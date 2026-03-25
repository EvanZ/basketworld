#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
import mlflow
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.backend.observations import _ensure_set_obs
from app.backend.selector_runtime import (
    apply_selected_offense_intent,
    selector_alpha_current,
    selector_neutralize_observation,
    selector_runtime_enabled,
    selector_sample_intent,
    selector_segment_boundary_reason,
)
from basketworld.envs.basketworld_env_v2 import Team
from basketworld.utils.action_resolution import IllegalActionStrategy
from basketworld.utils.callbacks import IntentDiversityCallback
from basketworld.utils.intent_discovery import (
    CompletedIntentEpisode,
    IntentDiscriminator,
    IntentTransition,
    build_padded_episode_batch,
    compute_episode_embeddings,
    extract_action_features_for_env,
    flatten_observation_for_env,
    load_intent_discriminator_from_checkpoint,
)
from basketworld.utils.mlflow_config import setup_mlflow
from basketworld.utils.mlflow_params import get_mlflow_params, get_mlflow_training_params
from basketworld.utils.policy_loading import load_ppo_for_inference
from basketworld.utils.self_play_wrapper import SelfPlayEnvWrapper
from train.config import get_args
from train.env_factory import setup_environment


def _extract_checkpoint_index(path_or_name: str) -> int | None:
    name = os.path.basename(str(path_or_name))
    match = re.search(r"_(?:alternation|iter)_(\d+)\.(?:pt|zip)$", name)
    if not match:
        return None
    return int(match.group(1))


def _resolve_artifact_uri(uri: str) -> tuple[str, str]:
    match = re.match(
        r"^mlflow-artifacts:/\d+/([0-9a-f]+)/artifacts/(.+)$",
        str(uri).strip(),
    )
    if not match:
        raise RuntimeError(f"Unsupported MLflow artifact URI: {uri}")
    return match.group(1), match.group(2)


def _download_matching_artifact(
    run_id: str,
    *,
    pattern: re.Pattern[str],
    checkpoint_idx: Optional[int],
    tmp_prefix: str,
) -> str:
    client = mlflow.tracking.MlflowClient()
    artifacts = client.list_artifacts(run_id, "models")
    candidates: list[tuple[int, str]] = []
    for item in artifacts:
        match = pattern.search(item.path)
        if match:
            candidates.append((int(match.group(1)), item.path))
    if not candidates:
        raise RuntimeError(
            f"No matching artifacts found in run {run_id} for pattern {pattern.pattern}"
        )

    artifact_path: Optional[str] = None
    if checkpoint_idx is not None:
        for idx, path in candidates:
            if idx == int(checkpoint_idx):
                artifact_path = path
                break
        if artifact_path is None:
            raise RuntimeError(
                f"No matching artifact with checkpoint/index={int(checkpoint_idx)} found in run {run_id}"
            )
    else:
        candidates.sort(key=lambda item: item[0])
        artifact_path = candidates[-1][1]

    tmpdir = tempfile.mkdtemp(prefix=tmp_prefix)
    return mlflow.tracking.MlflowClient().download_artifacts(run_id, artifact_path, tmpdir)


def _resolve_local_or_run_artifact(
    path_or_run_id: str,
    *,
    pattern: re.Pattern[str],
    checkpoint_idx: Optional[int],
    tmp_prefix: str,
) -> str:
    if os.path.isfile(path_or_run_id):
        return os.path.abspath(path_or_run_id)
    if os.path.isdir(path_or_run_id):
        if checkpoint_idx is None:
            raise RuntimeError(
                f"Directory input requires --alternation-index to select a checkpoint: {path_or_run_id}"
            )
        local_name = None
        if pattern.pattern.endswith(r"\.zip$"):
            local_name = f"unified_iter_{int(checkpoint_idx)}.zip"
        elif pattern.pattern.endswith(r"\.pt$"):
            local_name = f"intent_disc_iter_{int(checkpoint_idx)}.pt"
        if local_name is not None:
            candidate = os.path.join(path_or_run_id, local_name)
            if os.path.isfile(candidate):
                return os.path.abspath(candidate)
        raise RuntimeError(
            f"Could not find requested checkpoint/index={int(checkpoint_idx)} under directory {path_or_run_id}"
        )
    if str(path_or_run_id).startswith("mlflow-artifacts:/"):
        run_id, artifact_path = _resolve_artifact_uri(path_or_run_id)
        tmpdir = tempfile.mkdtemp(prefix=tmp_prefix)
        return mlflow.tracking.MlflowClient().download_artifacts(run_id, artifact_path, tmpdir)
    return _download_matching_artifact(
        path_or_run_id,
        pattern=pattern,
        checkpoint_idx=checkpoint_idx,
        tmp_prefix=tmp_prefix,
    )


def _resolve_policy_and_disc_paths(
    run_id_or_policy_path: str,
    *,
    checkpoint_idx: Optional[int],
    disc_path: Optional[str],
) -> tuple[str, str, str | None]:
    policy_pattern = re.compile(r"unified_(?:alternation|iter)_(\d+)\.zip$")
    disc_pattern = re.compile(r"intent_disc_(?:alternation|iter)_(\d+)\.pt$")
    policy_path = _resolve_local_or_run_artifact(
        run_id_or_policy_path,
        pattern=policy_pattern,
        checkpoint_idx=checkpoint_idx,
        tmp_prefix="intent_disc_single_policy_",
    )
    if disc_path:
        resolved_disc_path = _resolve_local_or_run_artifact(
            disc_path,
            pattern=disc_pattern,
            checkpoint_idx=checkpoint_idx,
            tmp_prefix="intent_disc_single_disc_",
        )
    else:
        run_id = None if os.path.isfile(run_id_or_policy_path) or os.path.isdir(run_id_or_policy_path) else run_id_or_policy_path
        if run_id is None:
            local_policy_dir = os.path.dirname(policy_path)
            local_idx = _extract_checkpoint_index(policy_path)
            if local_idx is not None:
                candidate = os.path.join(local_policy_dir, f"intent_disc_iter_{local_idx}.pt")
                if os.path.isfile(candidate):
                    resolved_disc_path = os.path.abspath(candidate)
                else:
                    raise RuntimeError(
                        "Could not infer a local discriminator checkpoint. Pass --disc-path explicitly."
                    )
            else:
                raise RuntimeError(
                    "Could not infer discriminator checkpoint index from policy path. Pass --disc-path explicitly."
                )
        else:
            resolved_disc_path = _download_matching_artifact(
                run_id,
                pattern=disc_pattern,
                checkpoint_idx=checkpoint_idx,
                tmp_prefix="intent_disc_single_disc_",
            )
    inferred_run_id = None
    if not os.path.isfile(run_id_or_policy_path) and not os.path.isdir(run_id_or_policy_path):
        inferred_run_id = run_id_or_policy_path
    return policy_path, resolved_disc_path, inferred_run_id


def _policy_uses_set_obs(model) -> bool:
    try:
        obs_space = getattr(getattr(model, "policy", None), "observation_space", None)
        return isinstance(obs_space, gym.spaces.Dict) and "players" in obs_space.spaces and "globals" in obs_space.spaces
    except Exception:
        return False


def _apply_run_params(args: argparse.Namespace, required: dict, optional: dict) -> None:
    for key, value in {**required, **optional}.items():
        setattr(args, key, value)
    if "pass_arc_degrees" in optional:
        args.pass_arc_start = optional["pass_arc_degrees"]
    if "pass_oob_turnover_prob" in optional:
        args.pass_oob_turnover_prob_start = optional["pass_oob_turnover_prob"]
    if "offensive_three_seconds_enabled" in optional:
        args.offensive_three_seconds = optional["offensive_three_seconds_enabled"]


def _build_env_args(run_id: str | None, model) -> argparse.Namespace:
    args = get_args([])
    if run_id is not None:
        client = mlflow.tracking.MlflowClient()
        required, optional = get_mlflow_params(client, run_id)
        training_params = get_mlflow_training_params(client, run_id)
        _apply_run_params(args, required, {**optional, **training_params})
    args.use_set_obs = _policy_uses_set_obs(model)
    args.enable_env_profiling = False
    if not bool(args.use_set_obs):
        args.mirror_episode_prob = 0.0
    return args


def _single_obs_to_batched(obs: dict) -> dict:
    batched: dict[str, Any] = {}
    for key, value in (obs or {}).items():
        if isinstance(value, np.ndarray):
            batched[key] = np.expand_dims(value, axis=0)
        else:
            batched[key] = value
    return batched


def _extract_obs_scalar(obs_payload, key: str, env_idx: int, default: float = 0.0) -> float:
    try:
        if not isinstance(obs_payload, dict) or key not in obs_payload:
            return float(default)
        arr = np.asarray(obs_payload[key], dtype=np.float32)
        if arr.ndim == 0:
            return float(arr)
        if arr.shape[0] <= int(env_idx):
            return float(default)
        return float(arr[int(env_idx)].reshape(-1)[0])
    except Exception:
        return float(default)


def _inspect_single_obs_input(obs: Any, *, target_intent: int) -> dict[str, Any]:
    batched_obs = _single_obs_to_batched(obs) if isinstance(obs, dict) else obs
    flattened = flatten_observation_for_env(batched_obs, 0)

    has_obs_key = isinstance(obs, dict) and "obs" in obs
    raw_obs_vec = None
    if has_obs_key:
        try:
            raw_obs_vec = np.asarray(obs["obs"], dtype=np.float32).reshape(-1)
        except Exception:
            raw_obs_vec = None

    raw_direct_intent_index = None
    raw_direct_intent_active = None
    if isinstance(obs, dict):
        if "intent_index" in obs:
            raw_direct_intent_index = int(
                max(0, int(_extract_obs_scalar(batched_obs, "intent_index", 0, default=0.0)))
            )
        if "intent_active" in obs:
            raw_direct_intent_active = bool(
                _extract_obs_scalar(batched_obs, "intent_active", 0, default=0.0) > 0.5
            )

    flatten_matches_obs_key = bool(
        raw_obs_vec is not None
        and flattened.shape == raw_obs_vec.shape
        and np.allclose(flattened, raw_obs_vec)
    )

    return {
        "has_obs_key": bool(has_obs_key),
        "has_players_key": bool(isinstance(obs, dict) and "players" in obs),
        "has_globals_key": bool(isinstance(obs, dict) and "globals" in obs),
        "has_direct_intent_index_key": bool(
            isinstance(obs, dict) and "intent_index" in obs
        ),
        "has_direct_intent_active_key": bool(
            isinstance(obs, dict) and "intent_active" in obs
        ),
        "raw_direct_intent_index": raw_direct_intent_index,
        "raw_direct_intent_active": raw_direct_intent_active,
        "raw_direct_intent_target_match": bool(
            raw_direct_intent_index is not None
            and raw_direct_intent_index == int(target_intent)
        ),
        "flatten_feature_dim": int(flattened.shape[0]),
        "raw_obs_dim": int(raw_obs_vec.shape[0]) if raw_obs_vec is not None else None,
        "flatten_matches_obs_key": bool(flatten_matches_obs_key),
    }


def _normalize_training_action_for_wrapper(
    action_payload,
    wrapper_env: SelfPlayEnvWrapper,
) -> np.ndarray:
    raw = np.asarray(action_payload, dtype=int).reshape(-1)
    expected = len(getattr(wrapper_env, "training_player_ids", []))
    total_players = int(getattr(wrapper_env.env.unwrapped, "n_players", raw.shape[0]))
    if raw.shape[0] == expected:
        return raw.astype(int, copy=False)
    if raw.shape[0] == total_players:
        indices = np.asarray(getattr(wrapper_env, "training_player_ids", []), dtype=int)
        return raw[indices].astype(int, copy=False)
    return raw[:expected].astype(int, copy=False)


def _build_transition_single_obs(
    obs: dict,
    training_action: np.ndarray,
    *,
    max_obs_dim: int,
    max_action_dim: int,
    rollout_step_idx: int,
) -> IntentTransition:
    batched_obs = _single_obs_to_batched(obs)
    batched_actions = np.asarray(training_action, dtype=np.int64).reshape(1, -1)
    obs_feat = flatten_observation_for_env(batched_obs, 0)
    act_feat = extract_action_features_for_env(batched_actions, 0)
    feat = np.zeros(max_obs_dim + max_action_dim, dtype=np.float32)
    obs_take = min(max_obs_dim, obs_feat.shape[0])
    if obs_take > 0:
        feat[:obs_take] = obs_feat[:obs_take]
    act_take = min(max_action_dim, act_feat.shape[0])
    if act_take > 0:
        feat[max_obs_dim : max_obs_dim + act_take] = act_feat[:act_take]

    role_flag = _extract_obs_scalar(batched_obs, "role_flag", 0, default=0.0)
    intent_active = _extract_obs_scalar(batched_obs, "intent_active", 0, default=0.0)
    intent_index = _extract_obs_scalar(batched_obs, "intent_index", 0, default=0.0)
    return IntentTransition(
        feature=feat,
        buffer_step_idx=int(rollout_step_idx),
        env_idx=0,
        role_flag=float(role_flag),
        intent_active=bool(intent_active > 0.5),
        intent_index=int(max(0, int(intent_index))),
    )


def _completed_pass_boundary(info: Any) -> bool:
    if not isinstance(info, dict):
        return False
    action_results = info.get("action_results", {})
    if not isinstance(action_results, dict):
        return False
    passes = action_results.get("passes", {})
    if not isinstance(passes, dict):
        return False
    for pass_result in passes.values():
        if isinstance(pass_result, dict) and bool(pass_result.get("success")):
            return True
    return False


def _segment_boundary_reason(
    *,
    info: Any,
    segment_length: int,
    min_play_steps: int,
    commitment_steps: int,
) -> str | None:
    if int(segment_length) >= int(commitment_steps):
        return "commitment_timeout"
    if int(segment_length) >= int(min_play_steps) and _completed_pass_boundary(info):
        return "completed_pass"
    return None


def _load_discriminator_checkpoint(
    path: str, device: str
) -> tuple[IntentDiscriminator, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or "state_dict" not in payload or "config" not in payload:
        raise RuntimeError(f"Unsupported discriminator checkpoint format: {path}")
    disc = load_intent_discriminator_from_checkpoint(payload, device=device)
    return disc, payload


def _prepare_obs(policy, wrapped_env: SelfPlayEnvWrapper, obs: Any) -> Any:
    obs = _ensure_set_obs(policy, wrapped_env, obs)
    if isinstance(obs, dict):
        obs["role_flag"] = np.array([1.0], dtype=np.float32)
    return obs


def _force_target_intent(
    wrapped_env: SelfPlayEnvWrapper,
    target_intent: int,
    *,
    commitment_steps: int,
) -> Any:
    wrapped_env.set_offense_intent_state(
        int(target_intent),
        intent_active=True,
        intent_age=0,
        intent_commitment_remaining=int(commitment_steps),
    )
    return getattr(wrapped_env, "_last_obs", None)


def _selector_training_params_dict(env_args: argparse.Namespace) -> dict[str, Any]:
    return dict(vars(env_args))


def _maybe_apply_natural_segment_start(
    wrapped_env: SelfPlayEnvWrapper,
    policy,
    obs: Any,
    *,
    training_params: dict[str, Any],
) -> tuple[Any, str]:
    base_env = getattr(wrapped_env, "unwrapped", wrapped_env)
    if not selector_runtime_enabled(training_params, policy):
        return obs, "selector_disabled"
    if not bool(getattr(base_env, "enable_intent_learning", False)):
        return obs, "intent_disabled"
    if not bool(getattr(base_env, "intent_active", False)):
        return obs, "inactive_at_start"
    selector_obs = selector_neutralize_observation(
        obs,
        max(1, int(getattr(base_env, "num_intents", 1))),
    )
    result = selector_sample_intent(
        training_params,
        policy,
        selector_obs,
        num_intents=max(1, int(getattr(base_env, "num_intents", 1))),
        allow_uniform_fallback=False,
        rng=getattr(base_env, "_rng", None),
    )
    intent_index = result.get("intent_index")
    if intent_index is None:
        return obs, "env_reset_random"
    apply_selected_offense_intent(
        wrapped_env,
        int(intent_index),
        intent_commitment_steps=int(getattr(base_env, "intent_commitment_steps", 0)),
    )
    return _prepare_obs(policy, wrapped_env, getattr(wrapped_env, "_last_obs", obs)), (
        "selector" if bool(result.get("used_selector", False)) else "env_reset_random"
    )


def _maybe_apply_natural_boundary_segment_start(
    wrapped_env: SelfPlayEnvWrapper,
    policy,
    obs: Any,
    *,
    training_params: dict[str, Any],
) -> tuple[Any, str]:
    base_env = getattr(wrapped_env, "unwrapped", wrapped_env)
    if not selector_runtime_enabled(training_params, policy):
        return obs, "selector_disabled"
    selector_obs = selector_neutralize_observation(
        obs,
        max(1, int(getattr(base_env, "num_intents", 1))),
    )
    result = selector_sample_intent(
        training_params,
        policy,
        selector_obs,
        num_intents=max(1, int(getattr(base_env, "num_intents", 1))),
        allow_uniform_fallback=True,
        rng=getattr(base_env, "_rng", None),
    )
    intent_index = result.get("intent_index")
    if intent_index is None:
        return obs, "no_selection"
    apply_selected_offense_intent(
        wrapped_env,
        int(intent_index),
        intent_commitment_steps=int(getattr(base_env, "intent_commitment_steps", 0)),
    )
    return _prepare_obs(policy, wrapped_env, getattr(wrapped_env, "_last_obs", obs)), (
        "selector" if bool(result.get("used_selector", False)) else "uniform_fallback"
    )


def collect_single_intent_segments(
    *,
    policy_path: str,
    env_args: argparse.Namespace,
    target_intent: int,
    num_segments: int,
    max_obs_dim: int,
    max_action_dim: int,
    player_deterministic: bool,
    opponent_deterministic: bool,
    device: str,
    opponent_policy_path: str | None,
) -> tuple[list[CompletedIntentEpisode], dict[str, Any]]:
    policy = load_ppo_for_inference(policy_path, device=device)
    wrapped_env = SelfPlayEnvWrapper(
        setup_environment(env_args, Team.OFFENSE),
        opponent_policy=str(opponent_policy_path or policy_path),
        training_strategy=IllegalActionStrategy.SAMPLE_PROB,
        opponent_strategy=IllegalActionStrategy.SAMPLE_PROB,
        deterministic_opponent=bool(opponent_deterministic),
    )
    episodes: list[CompletedIntentEpisode] = []
    boundary_counts: Counter[str] = Counter()
    possession_count = 0
    reset_count = 0
    rng = np.random.default_rng(0)
    input_diag_by_step: list[dict[str, Any]] = []
    kept_input_diags: list[dict[str, Any]] = []

    try:
        obs, _ = wrapped_env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        reset_count += 1
        obs = _prepare_obs(policy, wrapped_env, obs)
        obs = _prepare_obs(
            policy,
            wrapped_env,
            _force_target_intent(
                wrapped_env,
                int(target_intent),
                commitment_steps=int(getattr(env_args, "intent_commitment_steps", 4)),
            ),
        )
        possession_count += 1

        transitions: list[IntentTransition] = []
        rollout_step_idx = 0
        segment_length = 0

        while len(episodes) < int(num_segments):
            raw_action, _ = policy.predict(obs, deterministic=bool(player_deterministic))
            training_action = _normalize_training_action_for_wrapper(raw_action, wrapped_env)
            next_obs, _, terminated, truncated, info = wrapped_env.step(training_action)
            next_obs = _prepare_obs(policy, wrapped_env, next_obs)
            step_input_diag = _inspect_single_obs_input(
                next_obs,
                target_intent=int(target_intent),
            )

            transitions.append(
                _build_transition_single_obs(
                    next_obs,
                    training_action,
                    max_obs_dim=int(max_obs_dim),
                    max_action_dim=int(max_action_dim),
                    rollout_step_idx=rollout_step_idx,
                )
            )
            input_diag_by_step.append(step_input_diag)
            rollout_step_idx += 1
            segment_length += 1

            done = bool(terminated or truncated)
            boundary_reason = None
            if not done:
                boundary_reason = _segment_boundary_reason(
                    info=info,
                    segment_length=segment_length,
                    min_play_steps=int(getattr(env_args, "intent_selector_min_play_steps", 3)),
                    commitment_steps=int(getattr(env_args, "intent_commitment_steps", 4)),
                )

            if done or boundary_reason is not None:
                episode = CompletedIntentEpisode(
                    intent_index=int(target_intent),
                    transitions=list(transitions),
                    terminal_info=dict(info or {}),
                )
                if episode.role_is_offense and episode.active_prefix_length > 0:
                    episodes.append(episode)
                    boundary_counts[boundary_reason or "episode_end"] += 1
                    kept_input_diags.extend(
                        input_diag_by_step[: episode.active_prefix_length]
                    )
                transitions.clear()
                if not (episode.role_is_offense and episode.active_prefix_length > 0):
                    input_diag_by_step.clear()
                    segment_length = 0
                    if len(episodes) >= int(num_segments):
                        break
                    if done:
                        obs, _ = wrapped_env.reset(seed=int(rng.integers(0, 2**31 - 1)))
                        reset_count += 1
                        obs = _prepare_obs(policy, wrapped_env, obs)
                        possession_count += 1
                    obs = _prepare_obs(
                        policy,
                        wrapped_env,
                        _force_target_intent(
                            wrapped_env,
                            int(target_intent),
                            commitment_steps=int(getattr(env_args, "intent_commitment_steps", 4)),
                        ),
                    )
                    continue
                input_diag_by_step.clear()
                segment_length = 0
                if len(episodes) >= int(num_segments):
                    break
                if done:
                    obs, _ = wrapped_env.reset(seed=int(rng.integers(0, 2**31 - 1)))
                    reset_count += 1
                    obs = _prepare_obs(policy, wrapped_env, obs)
                    possession_count += 1
                obs = _prepare_obs(
                    policy,
                    wrapped_env,
                    _force_target_intent(
                        wrapped_env,
                        int(target_intent),
                        commitment_steps=int(getattr(env_args, "intent_commitment_steps", 4)),
                    ),
                )
                continue

            obs = next_obs
    finally:
        try:
            wrapped_env.close()
        except Exception:
            pass

    first_transition_labels = [
        int(ep.transitions[0].intent_index)
        for ep in episodes
        if ep.transitions
    ]
    first_transition_active = [
        bool(ep.transitions[0].intent_active)
        for ep in episodes
        if ep.transitions
    ]
    active_prefix_all_match = [
        bool(
            ep.active_prefix_length > 0
            and all(int(tr.intent_index) == int(target_intent) for tr in ep.active_prefix_transitions)
        )
        for ep in episodes
    ]
    flatten_feature_dims = [
        int(diag["flatten_feature_dim"])
        for diag in kept_input_diags
        if diag.get("flatten_feature_dim") is not None
    ]
    raw_obs_dims = [
        int(diag["raw_obs_dim"])
        for diag in kept_input_diags
        if diag.get("raw_obs_dim") is not None
    ]
    summary = {
        "possession_count": int(possession_count),
        "env_reset_count": int(reset_count),
        "boundary_reason_counts": dict(boundary_counts),
        "mean_active_prefix_length": float(
            np.mean([ep.active_prefix_length for ep in episodes]) if episodes else 0.0
        ),
        "mean_segment_length": float(
            np.mean([ep.length for ep in episodes]) if episodes else 0.0
        ),
        "collector_first_transition_intent_histogram": dict(
            sorted(Counter(first_transition_labels).items())
        ),
        "collector_first_transition_target_match_rate": float(
            np.mean([int(label) == int(target_intent) for label in first_transition_labels])
            if first_transition_labels
            else 0.0
        ),
        "collector_first_transition_active_rate": float(
            np.mean(first_transition_active) if first_transition_active else 0.0
        ),
        "collector_active_prefix_all_target_rate": float(
            np.mean(active_prefix_all_match) if active_prefix_all_match else 0.0
        ),
        "collector_obs_key_present_rate": float(
            np.mean([bool(diag.get("has_obs_key", False)) for diag in kept_input_diags])
            if kept_input_diags
            else 0.0
        ),
        "collector_set_obs_keys_present_rate": float(
            np.mean(
                [
                    bool(diag.get("has_players_key", False))
                    and bool(diag.get("has_globals_key", False))
                    for diag in kept_input_diags
                ]
            )
            if kept_input_diags
            else 0.0
        ),
        "collector_raw_direct_intent_index_present_rate": float(
            np.mean(
                [
                    bool(diag.get("has_direct_intent_index_key", False))
                    for diag in kept_input_diags
                ]
            )
            if kept_input_diags
            else 0.0
        ),
        "collector_raw_direct_intent_active_present_rate": float(
            np.mean(
                [
                    bool(diag.get("has_direct_intent_active_key", False))
                    for diag in kept_input_diags
                ]
            )
            if kept_input_diags
            else 0.0
        ),
        "collector_raw_direct_intent_target_match_rate": float(
            np.mean(
                [
                    bool(diag.get("raw_direct_intent_target_match", False))
                    for diag in kept_input_diags
                    if diag.get("has_direct_intent_index_key", False)
                ]
            )
            if any(diag.get("has_direct_intent_index_key", False) for diag in kept_input_diags)
            else 0.0
        ),
        "collector_raw_direct_intent_active_true_rate": float(
            np.mean(
                [
                    bool(diag.get("raw_direct_intent_active", False))
                    for diag in kept_input_diags
                    if diag.get("has_direct_intent_active_key", False)
                ]
            )
            if any(diag.get("has_direct_intent_active_key", False) for diag in kept_input_diags)
            else 0.0
        ),
        "collector_flatten_matches_obs_key_rate": float(
            np.mean(
                [
                    bool(diag.get("flatten_matches_obs_key", False))
                    for diag in kept_input_diags
                    if diag.get("has_obs_key", False)
                ]
            )
            if any(diag.get("has_obs_key", False) for diag in kept_input_diags)
            else 0.0
        ),
        "collector_flatten_feature_dim_histogram": dict(
            sorted(Counter(flatten_feature_dims).items())
        ),
        "collector_raw_obs_dim_histogram": dict(sorted(Counter(raw_obs_dims).items())),
    }
    return episodes, summary


def collect_natural_intent_segments(
    *,
    policy_path: str,
    env_args: argparse.Namespace,
    num_segments: int,
    max_obs_dim: int,
    max_action_dim: int,
    player_deterministic: bool,
    opponent_deterministic: bool,
    device: str,
    opponent_policy_path: str | None,
) -> tuple[list[CompletedIntentEpisode], dict[str, Any]]:
    policy = load_ppo_for_inference(policy_path, device=device)
    wrapped_env = SelfPlayEnvWrapper(
        setup_environment(env_args, Team.OFFENSE),
        opponent_policy=str(opponent_policy_path or policy_path),
        training_strategy=IllegalActionStrategy.SAMPLE_PROB,
        opponent_strategy=IllegalActionStrategy.SAMPLE_PROB,
        deterministic_opponent=bool(opponent_deterministic),
    )
    training_params = _selector_training_params_dict(env_args)
    episodes: list[CompletedIntentEpisode] = []
    boundary_counts: Counter[str] = Counter()
    segment_start_source_counts: Counter[str] = Counter()
    possession_count = 0
    reset_count = 0
    rng = np.random.default_rng(0)

    try:
        obs, _ = wrapped_env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        reset_count += 1
        obs = _prepare_obs(policy, wrapped_env, obs)
        obs, start_source = _maybe_apply_natural_segment_start(
            wrapped_env,
            policy,
            obs,
            training_params=training_params,
        )
        segment_start_source_counts[start_source] += 1
        possession_count += 1

        transitions: list[IntentTransition] = []
        rollout_step_idx = 0

        while len(episodes) < int(num_segments):
            raw_action, _ = policy.predict(obs, deterministic=bool(player_deterministic))
            training_action = _normalize_training_action_for_wrapper(raw_action, wrapped_env)
            next_obs, _, terminated, truncated, info = wrapped_env.step(training_action)
            next_obs = _prepare_obs(policy, wrapped_env, next_obs)

            tr = _build_transition_single_obs(
                next_obs,
                training_action,
                max_obs_dim=int(max_obs_dim),
                max_action_dim=int(max_action_dim),
                rollout_step_idx=rollout_step_idx,
            )
            transitions.append(tr)
            rollout_step_idx += 1

            done = bool(terminated or truncated)
            base_env = getattr(wrapped_env, "unwrapped", wrapped_env)
            boundary_reason = None
            if not done:
                boundary_reason = selector_segment_boundary_reason(
                    training_params,
                    segment_length=int(getattr(base_env, "intent_age", 0)),
                    info=info,
                    intent_commitment_steps=int(
                        getattr(base_env, "intent_commitment_steps", 0)
                    ),
                )

            if done or boundary_reason is not None:
                first_active_intent = 0
                for candidate in transitions:
                    if bool(candidate.intent_active):
                        first_active_intent = int(candidate.intent_index)
                        break
                episode = CompletedIntentEpisode(
                    intent_index=int(first_active_intent),
                    transitions=list(transitions),
                    terminal_info=dict(info or {}),
                )
                if episode.role_is_offense and episode.active_prefix_length > 0:
                    episodes.append(episode)
                    boundary_counts[boundary_reason or "episode_end"] += 1
                transitions.clear()
                if len(episodes) >= int(num_segments):
                    break
                if done:
                    obs, _ = wrapped_env.reset(seed=int(rng.integers(0, 2**31 - 1)))
                    reset_count += 1
                    obs = _prepare_obs(policy, wrapped_env, obs)
                    obs, start_source = _maybe_apply_natural_segment_start(
                        wrapped_env,
                        policy,
                        obs,
                        training_params=training_params,
                    )
                    segment_start_source_counts[start_source] += 1
                    possession_count += 1
                else:
                    obs, start_source = _maybe_apply_natural_boundary_segment_start(
                        wrapped_env,
                        policy,
                        next_obs,
                        training_params=training_params,
                    )
                    segment_start_source_counts[start_source] += 1
                continue

            obs = next_obs
    finally:
        try:
            wrapped_env.close()
        except Exception:
            pass

    labels = [int(ep.intent_index) for ep in episodes]
    summary = {
        "possession_count": int(possession_count),
        "env_reset_count": int(reset_count),
        "boundary_reason_counts": dict(boundary_counts),
        "segment_start_source_counts": dict(segment_start_source_counts),
        "mean_active_prefix_length": float(
            np.mean([ep.active_prefix_length for ep in episodes]) if episodes else 0.0
        ),
        "mean_segment_length": float(
            np.mean([ep.length for ep in episodes]) if episodes else 0.0
        ),
        "collected_intent_histogram": dict(sorted(Counter(labels).items())),
        "selector_runtime_active": bool(selector_runtime_enabled(training_params, policy)),
        "selector_alpha_current": float(selector_alpha_current(training_params, policy)),
    }
    return episodes, summary


def score_segments_general(
    episodes: list[CompletedIntentEpisode],
    *,
    disc_bundle: tuple[IntentDiscriminator, dict[str, Any]],
    device: str,
) -> dict[str, Any]:
    disc, payload = disc_bundle
    disc_config = dict(payload.get("config", {}) or {})
    encoder_type = str(disc_config.get("encoder_type", "mlp_mean")).strip().lower()
    max_obs_dim = int(disc_config.get("max_obs_dim", 256))
    max_action_dim = int(disc_config.get("max_action_dim", 16))

    if encoder_type == "gru":
        x_np, lengths_np, y_np = build_padded_episode_batch(
            episodes,
            max_obs_dim=max_obs_dim,
            max_action_dim=max_action_dim,
        )
        x = torch.as_tensor(x_np, dtype=torch.float32, device=device)
        lengths = torch.as_tensor(lengths_np, dtype=torch.long, device=device)
        with torch.no_grad():
            logits = disc(x, lengths)
    else:
        x_np, y_np = compute_episode_embeddings(
            episodes,
            max_obs_dim=max_obs_dim,
            max_action_dim=max_action_dim,
        )
        lengths_np = None
        x = torch.as_tensor(x_np, dtype=torch.float32, device=device)
        with torch.no_grad():
            logits = disc(x, None)

    logits_tensor = torch.as_tensor(logits, dtype=torch.float32, device=device)
    probs = (
        torch.softmax(logits_tensor, dim=-1)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
    )
    pred = np.argmax(probs, axis=1).astype(np.int64, copy=False)
    y_np = np.asarray(y_np, dtype=np.int64).reshape(-1)
    num_intents = int(disc_config.get("num_intents", probs.shape[1] if probs.ndim == 2 else 0))
    confusion = np.zeros((max(1, num_intents), max(1, num_intents)), dtype=np.int64)
    if y_np.size and pred.size:
        for true_z, pred_z in zip(y_np.tolist(), pred.tolist()):
            if 0 <= int(true_z) < confusion.shape[0] and 0 <= int(pred_z) < confusion.shape[1]:
                confusion[int(true_z), int(pred_z)] += 1
    row_den = np.clip(confusion.sum(axis=1, keepdims=True), 1, None)
    return {
        "num_segments": int(y_np.size),
        "true_class_histogram": np.bincount(y_np, minlength=max(1, num_intents)).tolist()
        if y_np.size
        else [],
        "predicted_class_histogram": np.bincount(pred, minlength=max(1, num_intents)).tolist()
        if pred.size
        else [],
        "recomputed_top1_acc": float(np.mean(pred == y_np)) if y_np.size else 0.0,
        "mean_max_probability": float(np.mean(np.max(probs, axis=1))) if probs.size else 0.0,
        "row_confusion_counts": np.bincount(pred, minlength=max(1, num_intents)).tolist()
        if pred.size
        else [],
        "confusion_counts": confusion.tolist(),
        "row_normalized_confusion": (confusion / row_den).tolist(),
        "lengths": None if lengths_np is None else lengths_np.tolist(),
        "recomputed_auc_ovr_macro": (
            float(
                IntentDiversityCallback._multiclass_auc_ovr_macro(
                    logits_tensor.detach().cpu().numpy(),
                    y_np,
                    num_classes=max(1, num_intents),
                )
            )
            if y_np.size and len(np.unique(y_np)) > 1
            else None
        ),
    }


def score_single_intent_segments(
    episodes: list[CompletedIntentEpisode],
    *,
    disc_bundle: tuple[IntentDiscriminator, dict[str, Any]],
    device: str,
) -> dict[str, Any]:
    general = score_segments_general(
        episodes,
        disc_bundle=disc_bundle,
        device=device,
    )
    y_np = np.asarray([int(ep.intent_index) for ep in episodes], dtype=np.int64).reshape(-1)
    target_intent = int(y_np[0]) if y_np.size else 0
    probs_placeholder = None
    disc, payload = disc_bundle
    disc_config = dict(payload.get("config", {}) or {})
    encoder_type = str(disc_config.get("encoder_type", "mlp_mean")).strip().lower()
    max_obs_dim = int(disc_config.get("max_obs_dim", 256))
    max_action_dim = int(disc_config.get("max_action_dim", 16))
    if encoder_type == "gru":
        x_np, lengths_np, y_np = build_padded_episode_batch(
            episodes,
            max_obs_dim=max_obs_dim,
            max_action_dim=max_action_dim,
        )
        x = torch.as_tensor(x_np, dtype=torch.float32, device=device)
        lengths = torch.as_tensor(lengths_np, dtype=torch.long, device=device)
        with torch.no_grad():
            logits = disc(x, lengths)
    else:
        x_np, y_np = compute_episode_embeddings(
            episodes,
            max_obs_dim=max_obs_dim,
            max_action_dim=max_action_dim,
        )
        x = torch.as_tensor(x_np, dtype=torch.float32, device=device)
        with torch.no_grad():
            logits = disc(x, None)
    probs_placeholder = (
        torch.softmax(torch.as_tensor(logits, dtype=torch.float32, device=device), dim=-1)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
    )
    return general | {
        "target_intent": int(target_intent),
        "target_prediction_rate": float(general.get("recomputed_top1_acc", 0.0)),
        "mean_target_probability": float(
            np.mean(probs_placeholder[:, target_intent])
        )
        if probs_placeholder.size and 0 <= target_intent < probs_placeholder.shape[1]
        else 0.0,
    }


def validate_result(
    result: dict[str, Any],
    *,
    min_target_rate: float | None,
) -> dict[str, Any]:
    failed: list[str] = []
    if min_target_rate is not None:
        if "per_intent" in result:
            for intent_key, intent_result in sorted(
                dict(result.get("per_intent", {})).items(),
                key=lambda item: int(item[0]),
            ):
                actual = float(
                    dict(intent_result or {}).get("target_prediction_rate", 0.0)
                )
                if actual < float(min_target_rate):
                    failed.append(
                        f"intent {int(intent_key)} target_prediction_rate {actual:.6f} < required {float(min_target_rate):.6f}"
                    )
        else:
            actual = float(result.get("target_prediction_rate", 0.0))
            if actual < float(min_target_rate):
                failed.append(
                    f"target_prediction_rate {actual:.6f} < required {float(min_target_rate):.6f}"
                )
    return {"passed": len(failed) == 0, "failed_checks": failed}


def _render_summary(result: dict[str, Any]) -> str:
    lines: list[str] = []
    if bool(result.get("natural_intents", False)):
        lines.append("natural_intents=true")
        lines.append(
            f"recomputed_top1_acc={float(result.get('recomputed_top1_acc', 0.0)):.4f}"
        )
        auc = result.get("recomputed_auc_ovr_macro", None)
        lines.append(
            "recomputed_auc_ovr_macro="
            + ("null" if auc is None else f"{float(auc):.4f}")
        )
        hist = dict(result.get("collection", {}).get("collected_intent_histogram", {}) or {})
        if hist:
            lines.append(f"collected_intent_histogram={json.dumps(hist, sort_keys=True)}")
    elif bool(result.get("all_intents", False)):
        lines.append("all_intents=true")
        lines.append(f"target_prediction_rate_mean={float(result.get('target_prediction_rate_mean', 0.0)):.4f}")
        lines.append(f"target_prediction_rate_min={float(result.get('target_prediction_rate_min', 0.0)):.4f}")
        dominant = dict(result.get("dominant_prediction_by_intent", {}) or {})
        if dominant:
            lines.append(f"dominant_prediction_by_intent={json.dumps(dominant, sort_keys=True)}")
    else:
        lines.append(f"target_intent={int(result.get('target_intent', 0))}")
        lines.append(f"target_prediction_rate={float(result.get('target_prediction_rate', 0.0)):.4f}")
        lines.append(f"mean_target_probability={float(result.get('mean_target_probability', 0.0)):.4f}")
        lines.append(f"mean_max_probability={float(result.get('mean_max_probability', 0.0)):.4f}")
    validation = dict(result.get("validation", {}) or {})
    lines.append(f"validation_passed={bool(validation.get('passed', False))}")
    failed_checks = list(validation.get("failed_checks", []) or [])
    if failed_checks:
        lines.append("failed_checks:")
        for item in failed_checks:
            lines.append(f"  - {item}")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect fresh segments outside training and score the discriminator on them."
        )
    )
    parser.add_argument(
        "run_id_or_policy_path",
        help="MLflow run id, local checkpoint path, local checkpoint directory, or MLflow artifact URI for the unified policy.",
    )
    parser.add_argument(
        "--alternation-index",
        type=int,
        default=None,
        help="Checkpoint index when resolving policy/discriminator from an MLflow run or local directory.",
    )
    parser.add_argument(
        "--disc-path",
        type=str,
        default=None,
        help="Optional local or MLflow artifact URI for the discriminator checkpoint.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--intent-index",
        type=int,
        default=None,
        help="Force this offense intent for every collected segment.",
    )
    group.add_argument(
        "--all-intents",
        action="store_true",
        help="Run the same fresh recollection test for every intent index 0..num_intents-1.",
    )
    group.add_argument(
        "--natural-intents",
        action="store_true",
        help="Collect fresh segments using the checkpoint's natural selector/runtime behavior instead of forcing intents.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of completed segments to collect for the forced intent.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device for policy/discriminator inference.",
    )
    parser.add_argument(
        "--player-deterministic",
        action="store_true",
        help="Use deterministic policy actions for the offense policy under test.",
    )
    parser.add_argument(
        "--opponent-deterministic",
        action="store_true",
        help="Use deterministic opponent actions.",
    )
    parser.add_argument(
        "--opponent-policy-path",
        type=str,
        default=None,
        help="Optional separate opponent policy path or MLflow artifact URI. Defaults to the same unified policy checkpoint.",
    )
    parser.add_argument(
        "--min-target-rate",
        type=float,
        default=None,
        help="Optional minimum required rate at which the discriminator predicts the forced intent.",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="Optional path to write the result JSON.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not os.path.isfile(args.run_id_or_policy_path) and not os.path.isdir(args.run_id_or_policy_path):
        setup_mlflow(verbose=False)
    policy_path, disc_path, inferred_run_id = _resolve_policy_and_disc_paths(
        args.run_id_or_policy_path,
        checkpoint_idx=args.alternation_index,
        disc_path=args.disc_path,
    )
    policy = load_ppo_for_inference(policy_path, device=args.device)
    env_args = _build_env_args(inferred_run_id, policy)
    disc_bundle = _load_discriminator_checkpoint(disc_path, device=args.device)
    disc_config = dict(disc_bundle[1].get("config", {}) or {})
    opponent_policy_path = (
        _resolve_local_or_run_artifact(
            args.opponent_policy_path,
            pattern=re.compile(r"unified_(?:alternation|iter)_(\d+)\.zip$"),
            checkpoint_idx=args.alternation_index,
            tmp_prefix="intent_disc_single_opp_",
        )
        if args.opponent_policy_path
        else policy_path
    )

    requested_intents: list[int]
    if bool(args.natural_intents):
        requested_intents = []
    elif bool(args.all_intents):
        requested_intents = list(range(int(disc_config.get("num_intents", 0))))
    else:
        requested_intents = [int(args.intent_index)]

    result = {
        "policy_path": os.path.abspath(policy_path),
        "discriminator_checkpoint_path": os.path.abspath(disc_path),
        "opponent_policy_path": os.path.abspath(opponent_policy_path),
        "requested_segments": int(args.episodes),
        "disc_checkpoint_config": disc_config,
        "disc_checkpoint_meta": dict(disc_bundle[1].get("meta", {}) or {}),
    }
    if bool(args.natural_intents):
        episodes, collection = collect_natural_intent_segments(
            policy_path=policy_path,
            env_args=env_args,
            num_segments=int(args.episodes),
            max_obs_dim=int(disc_config.get("max_obs_dim", 256)),
            max_action_dim=int(disc_config.get("max_action_dim", 16)),
            player_deterministic=bool(args.player_deterministic),
            opponent_deterministic=bool(args.opponent_deterministic),
            device=args.device,
            opponent_policy_path=opponent_policy_path,
        )
        result.update(
            score_segments_general(
                episodes,
                disc_bundle=disc_bundle,
                device=args.device,
            )
        )
        result["natural_intents"] = True
        result["collection"] = collection
    else:
        per_intent: dict[str, Any] = {}
        prediction_matrix_counts: list[list[int]] = []
        target_rates: list[float] = []
        dominant_prediction_by_intent: dict[str, int] = {}
        collection_by_intent: dict[str, Any] = {}

        for target_intent in requested_intents:
            episodes, collection = collect_single_intent_segments(
                policy_path=policy_path,
                env_args=env_args,
                target_intent=int(target_intent),
                num_segments=int(args.episodes),
                max_obs_dim=int(disc_config.get("max_obs_dim", 256)),
                max_action_dim=int(disc_config.get("max_action_dim", 16)),
                player_deterministic=bool(args.player_deterministic),
                opponent_deterministic=bool(args.opponent_deterministic),
                device=args.device,
                opponent_policy_path=opponent_policy_path,
            )
            scored = score_single_intent_segments(
                episodes,
                disc_bundle=disc_bundle,
                device=args.device,
            )
            intent_key = str(int(target_intent))
            per_intent[intent_key] = scored
            collection_by_intent[intent_key] = collection
            hist = list(scored.get("predicted_class_histogram", []))
            prediction_matrix_counts.append([int(v) for v in hist])
            target_rates.append(float(scored.get("target_prediction_rate", 0.0)))
            dominant_prediction_by_intent[intent_key] = (
                int(np.argmax(np.asarray(hist, dtype=np.int64)))
                if hist
                else -1
            )

        if len(requested_intents) == 1:
            intent_key = str(int(requested_intents[0]))
            result.update(per_intent[intent_key])
            result["collection"] = collection_by_intent[intent_key]
        else:
            result.update(
                {
                    "all_intents": True,
                    "num_intents": int(disc_config.get("num_intents", len(requested_intents))),
                    "per_intent": per_intent,
                    "collection_by_intent": collection_by_intent,
                    "dominant_prediction_by_intent": dominant_prediction_by_intent,
                    "target_prediction_rate_mean": float(np.mean(target_rates))
                    if target_rates
                    else 0.0,
                    "target_prediction_rate_min": float(np.min(target_rates))
                    if target_rates
                    else 0.0,
                    "forced_intent_prediction_matrix_counts": prediction_matrix_counts,
                }
            )
    result["validation"] = validate_result(
        result,
        min_target_rate=args.min_target_rate,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.json_out:
        out_path = os.path.abspath(args.json_out)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(rendered)
            fh.write("\n")
        print(_render_summary(result))
        print(f"json_out={out_path}")
    else:
        print(rendered)
    return 0 if bool(result["validation"]["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
