#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue
import re
import sys
import tempfile
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
import mlflow
import numpy as np
import torch
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.backend.observations import _ensure_set_obs, rebuild_observation_from_env
from app.backend.rollout_runtime import (
    initialize_rollout_selector_episode,
)
from app.backend.selector_runtime import (
    apply_selected_offense_intent,
    apply_rollout_segment_start,
    selector_alpha_current,
    selector_completed_pass_boundary,
    selector_runtime_enabled,
    selector_segment_boundary_reason,
)
from basketworld.envs.basketworld_env_v2 import Team
from basketworld.utils.action_resolution import IllegalActionStrategy
from basketworld.utils.callbacks import IntentDiversityCallback
from basketworld.utils.intent_discovery import (
    CompletedIntentEpisode,
    IntentEpisodeBuffer,
    IntentDiscriminator,
    IntentTransition,
    build_padded_episode_batch,
    compute_episode_embeddings,
    extract_action_features_for_env,
    flatten_observation_for_env,
    load_intent_discriminator_from_checkpoint,
)
from basketworld.utils.intent_policy_sensitivity import (
    clone_observation_dict,
    patch_intent_in_observation,
)
from basketworld.utils.mlflow_config import setup_mlflow
from basketworld.utils.mlflow_params import get_mlflow_params, get_mlflow_training_params
from basketworld.utils.policy_loading import load_ppo_for_inference
from basketworld.utils.self_play_wrapper import SelfPlayEnvWrapper
from train.config import get_args
from train.env_factory import setup_environment


def _make_collection_progress(*, total: int, desc: str):
    return tqdm(
        total=max(0, int(total)),
        desc=str(desc),
        leave=False,
        dynamic_ncols=True,
        disable=not sys.stderr.isatty(),
    )


_TRAINING_HOLDOUT_WORKER_STATE: dict[str, Any] | None = None


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


def _download_exact_artifact(
    run_id: str,
    *,
    artifact_path: str,
    tmp_prefix: str,
) -> str:
    tmpdir = tempfile.mkdtemp(prefix=tmp_prefix)
    return mlflow.tracking.MlflowClient().download_artifacts(
        run_id,
        artifact_path,
        tmpdir,
    )


def _infer_run_id(input_value: str) -> str | None:
    raw = str(input_value or "").strip()
    if not raw:
        return None
    if os.path.isfile(raw) or os.path.isdir(raw):
        return None
    if raw.startswith("mlflow-artifacts:/"):
        run_id, _ = _resolve_artifact_uri(raw)
        return run_id
    return raw


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
    require_disc: bool = True,
) -> tuple[str, str | None, str | None]:
    policy_pattern = re.compile(r"unified_(?:alternation|iter)_(\d+)\.zip$")
    disc_pattern = re.compile(r"intent_disc_(?:alternation|iter)_(\d+)\.pt$")
    policy_path = _resolve_local_or_run_artifact(
        run_id_or_policy_path,
        pattern=policy_pattern,
        checkpoint_idx=checkpoint_idx,
        tmp_prefix="intent_disc_single_policy_",
    )
    resolved_disc_path: str | None = None
    if disc_path:
        resolved_disc_path = _resolve_local_or_run_artifact(
            disc_path,
            pattern=disc_pattern,
            checkpoint_idx=checkpoint_idx,
            tmp_prefix="intent_disc_single_disc_",
        )
    elif require_disc:
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
    inferred_run_id = _infer_run_id(run_id_or_policy_path)
    return policy_path, resolved_disc_path, inferred_run_id


def _infer_num_intents(model: Any, env_args: argparse.Namespace) -> int:
    try:
        value = int(getattr(env_args, "num_intents", 0))
        if value > 0:
            return int(value)
    except Exception:
        pass
    try:
        policy_obj = getattr(model, "policy", None)
        extractor = getattr(policy_obj, "features_extractor", None)
        value = int(getattr(extractor, "num_intents", 0))
        if value > 0:
            return int(value)
    except Exception:
        pass
    try:
        value = int(getattr(getattr(model, "policy", None), "num_intents", 0))
        if value > 0:
            return int(value)
    except Exception:
        pass
    return 8


def _infer_feature_dim_from_episodes(
    episodes: list[CompletedIntentEpisode],
    *,
    default: int,
) -> int:
    for ep in episodes:
        for tr in list(getattr(ep, "transitions", []) or []):
            try:
                feat = np.asarray(getattr(tr, "feature", None), dtype=np.float32).reshape(-1)
            except Exception:
                feat = np.zeros((0,), dtype=np.float32)
            if feat.size > 0:
                return int(feat.size)
    return int(default)


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


def _download_artifact_cached(
    client: Any,
    run_id: str,
    artifact_path: str,
    *,
    cache_root: str,
    log_prefix: str = "[IntentDiscEval]",
) -> str:
    target_path = os.path.join(cache_root, artifact_path)
    if os.path.isfile(target_path):
        print(f"{log_prefix} Using cached artifact: {target_path}", flush=True)
        return target_path
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    print(
        f"{log_prefix} Downloading artifact: run_id={run_id} path={artifact_path}",
        flush=True,
    )
    return client.download_artifacts(run_id, artifact_path, cache_root)


def _list_unified_checkpoint_artifacts(
    client: Any,
    run_id: str,
    *,
    checkpoint_idx: int | None = None,
    include_current: bool = False,
) -> list[tuple[int, str]]:
    artifacts = client.list_artifacts(run_id, "models")
    pattern = re.compile(r"unified_(?:alternation|iter)_(\d+)\.zip$")
    candidates: list[tuple[int, str]] = []
    for item in artifacts:
        match = pattern.search(item.path)
        if not match:
            continue
        idx = int(match.group(1))
        if checkpoint_idx is not None:
            if include_current:
                if idx > int(checkpoint_idx):
                    continue
            elif idx >= int(checkpoint_idx):
                continue
        candidates.append((idx, item.path))
    candidates.sort(key=lambda item: item[0])
    return candidates


def _parse_logged_opponent_assignment_text(note_text: str) -> list[str]:
    env_matches: list[tuple[int, str]] = []
    for line in str(note_text).splitlines():
        match = re.search(r"Env\s+(\d+):\s+(\S+)", line)
        if match:
            env_matches.append((int(match.group(1)), str(match.group(2)).strip()))
    if env_matches:
        env_matches.sort(key=lambda item: item[0])
        return [name for _, name in env_matches]

    single_matches = re.findall(
        r"^\s+(\S+\.zip)\s*$",
        str(note_text),
        flags=re.MULTILINE,
    )
    if single_matches:
        return [str(single_matches[0]).strip()]
    return []


def _fit_logged_opponent_assignments(
    basenames: list[str],
    *,
    num_assignments: int,
) -> list[str]:
    if not basenames:
        return []
    target = max(1, int(num_assignments))
    if len(basenames) == target:
        return list(basenames)
    if len(basenames) == 1:
        return [str(basenames[0]) for _ in range(target)]
    if len(basenames) > target:
        return list(basenames[:target])
    fitted = list(basenames)
    while len(fitted) < target:
        fitted.extend(basenames[: max(0, target - len(fitted))])
    return fitted[:target]


def _training_offense_assignment_count(env_args: argparse.Namespace) -> int:
    total_envs = max(1, int(getattr(env_args, "num_envs", 1)))
    return max(1, total_envs // 2)


def _offense_subset_logged_assignments(
    basenames: list[str],
    *,
    env_args: argparse.Namespace,
) -> list[str]:
    if not basenames:
        return []
    offense_count = _training_offense_assignment_count(env_args)
    if len(basenames) <= offense_count:
        return list(basenames)
    return list(basenames[:offense_count])


def _resolve_training_holdout_match_opponent_policy_paths(
    *,
    run_id: str,
    policy_path: str,
    checkpoint_idx: int,
    env_args: argparse.Namespace,
) -> tuple[list[str], dict[str, Any]]:
    client = mlflow.tracking.MlflowClient()
    artifact_path = f"opponents/opponent_alt_{int(checkpoint_idx)}.txt"
    cached_note_path = _download_artifact_cached(
        client,
        run_id,
        artifact_path,
        cache_root=os.path.join(".opponent_cache", run_id),
        log_prefix="[IntentDiscEval]",
    )
    basenames = _parse_logged_opponent_assignment_text(
        Path(cached_note_path).read_text(encoding="utf-8")
    )
    if not basenames:
        raise RuntimeError(
            f"Logged opponent assignment artifact was empty or unparseable: {artifact_path}"
        )
    offense_basenames = _offense_subset_logged_assignments(
        basenames,
        env_args=env_args,
    )
    target_count = _training_offense_assignment_count(env_args)
    fitted_basenames = _fit_logged_opponent_assignments(
        offense_basenames,
        num_assignments=target_count,
    )
    candidates = _list_unified_checkpoint_artifacts(
        client,
        run_id,
        checkpoint_idx=int(checkpoint_idx),
        include_current=True,
    )
    model_artifact_map = {os.path.basename(path): path for _, path in candidates}
    local_cache: dict[str, str] = {}
    resolved_paths: list[str] = []
    missing_basenames: list[str] = []
    for basename in fitted_basenames:
        artifact = model_artifact_map.get(str(basename))
        if artifact is None:
            if str(basename) == os.path.basename(str(policy_path)):
                resolved_paths.append(str(policy_path))
                continue
            missing_basenames.append(str(basename))
            continue
        if artifact not in local_cache:
            local_cache[artifact] = _download_artifact_cached(
                client,
                run_id,
                artifact,
                cache_root=os.path.join(".opponent_cache", run_id),
                log_prefix="[IntentDiscEval]",
            )
        resolved_paths.append(local_cache[artifact])
    if missing_basenames:
        raise RuntimeError(
            "Logged opponent assignment referenced checkpoints that could not be resolved: "
            + ", ".join(sorted(set(missing_basenames)))
        )
    return resolved_paths, {
        "opponent_mode": "training_holdout_match",
        "opponent_source": f"logged_assignment_alt_{int(checkpoint_idx)}",
        "opponent_assignment_subset": "offense_only",
        "training_offense_assignment_count": int(target_count),
        "opponent_policy_basenames": fitted_basenames,
        "opponent_policy_paths": resolved_paths,
        "policy_checkpoint_index": int(checkpoint_idx),
    }


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


def _compute_policy_state_embedding(model: Any, obs: dict) -> np.ndarray:
    policy_obj = getattr(model, "policy", None)
    if policy_obj is None:
        raise RuntimeError("Loaded PPO model has no policy object.")

    obs_tensor, _ = policy_obj.obs_to_tensor(obs)
    with torch.no_grad():
        features = policy_obj.extract_features(obs_tensor)
    if isinstance(features, tuple):
        features = features[0]
    if features.ndim == 1:
        features = features.unsqueeze(0)

    if hasattr(policy_obj, "_split_tokens"):
        tokens = policy_obj._split_tokens(features)
        token_players = int(getattr(policy_obj, "token_players", tokens.shape[1]))
        pooled = torch.mean(tokens[:, :token_players, :], dim=1)
    else:
        pooled = features

    return (
        pooled[0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
        .reshape(-1)
    )


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


def _sample_rollout_action_like_ppo(
    model: Any,
    obs: Any,
    *,
    deterministic: bool,
) -> np.ndarray:
    policy_obj = getattr(model, "policy", None)
    if policy_obj is None:
        raw_action, _ = model.predict(obs, deterministic=bool(deterministic))
        return np.asarray(raw_action)

    with torch.no_grad():
        obs_tensor, _ = policy_obj.obs_to_tensor(obs)
        if bool(deterministic):
            actions = policy_obj._predict(obs_tensor, deterministic=True)
        else:
            actions, _, _ = policy_obj(obs_tensor)
    actions_np = actions.detach().cpu().numpy()

    action_space = getattr(policy_obj, "action_space", getattr(model, "action_space", None))
    if isinstance(action_space, gym.spaces.Box):
        if bool(getattr(policy_obj, "squash_output", False)):
            actions_np = policy_obj.unscale_action(actions_np)
        else:
            actions_np = np.clip(actions_np, action_space.low, action_space.high)
    return np.asarray(actions_np)


def _build_transition_single_obs(
    obs: dict,
    training_action: np.ndarray,
    *,
    max_obs_dim: int,
    max_action_dim: int,
    rollout_step_idx: int,
    env_idx: int = 0,
    feature_override: np.ndarray | None = None,
    info: dict[str, Any] | None = None,
) -> IntentTransition:
    obs_batch_idx = 0
    info_dict = dict(info or {})
    if feature_override is not None:
        batched_obs = _single_obs_to_batched(obs)
        role_flag = _extract_obs_scalar(
            batched_obs, "role_flag", obs_batch_idx, default=0.0
        )
        intent_active = _extract_obs_scalar(
            batched_obs,
            "intent_active",
            obs_batch_idx,
            default=float(info_dict.get("intent_active", 0.0)),
        )
        intent_index = _extract_obs_scalar(
            batched_obs,
            "intent_index",
            obs_batch_idx,
            default=float(info_dict.get("intent_index", 0.0)),
        )
        return IntentTransition(
            feature=np.asarray(feature_override, dtype=np.float32).reshape(-1),
            buffer_step_idx=int(rollout_step_idx),
            env_idx=int(env_idx),
            role_flag=float(role_flag),
            intent_active=bool(intent_active > 0.5),
            intent_index=int(max(0, int(intent_index))),
        )
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

    role_flag = _extract_obs_scalar(batched_obs, "role_flag", obs_batch_idx, default=0.0)
    intent_active = _extract_obs_scalar(
        batched_obs,
        "intent_active",
        obs_batch_idx,
        default=float(info_dict.get("intent_active", 0.0)),
    )
    intent_index = _extract_obs_scalar(
        batched_obs,
        "intent_index",
        obs_batch_idx,
        default=float(info_dict.get("intent_index", 0.0)),
    )
    return IntentTransition(
        feature=feat,
        buffer_step_idx=int(rollout_step_idx),
        env_idx=int(env_idx),
        role_flag=float(role_flag),
        intent_active=bool(intent_active > 0.5),
        intent_index=int(max(0, int(intent_index))),
    )


def _completed_pass_boundary(info: Any) -> bool:
    return bool(selector_completed_pass_boundary(info))


def _segment_boundary_reason(
    *,
    info: Any,
    segment_length: int,
    min_play_steps: int,
    commitment_steps: int,
) -> str | None:
    return selector_segment_boundary_reason(
        {
            "intent_selector_multiselect_enabled": True,
            "intent_selector_min_play_steps": int(min_play_steps),
        },
        segment_length=int(segment_length),
        info=info,
        intent_commitment_steps=int(commitment_steps),
    )


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
    return rebuild_observation_from_env(
        wrapped_env,
        current_obs=getattr(wrapped_env, "_last_obs", None),
        role_flag_value=1.0,
    )


def _selector_training_params_dict(env_args: argparse.Namespace) -> dict[str, Any]:
    return dict(vars(env_args))


def _ppo_like_apply_rollout_segment_start(
    *,
    env: Any,
    obs: Any,
    policy: Any,
    training_params: dict[str, Any] | None,
    allow_uniform_fallback: bool,
) -> dict[str, Any]:
    if not isinstance(obs, dict):
        return {"obs": obs, "used_selector": False, "intent_index": None}
    if not selector_runtime_enabled(training_params, policy):
        return {"obs": obs, "used_selector": False, "intent_index": None}

    batched_obs = _single_obs_to_batched(obs)
    if _extract_obs_scalar(batched_obs, "role_flag", 0, default=0.0) <= 0.0:
        return {"obs": obs, "used_selector": False, "intent_index": None}
    if _extract_obs_scalar(batched_obs, "intent_active", 0, default=0.0) <= 0.5:
        return {"obs": obs, "used_selector": False, "intent_index": None}

    base_env = getattr(env, "unwrapped", env)
    num_intents = max(1, int(getattr(base_env, "num_intents", 1) or 1))
    selector_obs = clone_observation_dict(obs)
    patch_intent_in_observation(
        selector_obs,
        0,
        num_intents,
        active=0.0,
        visible=0.0,
        age_norm=0.0,
    )

    used_selector = False
    chosen_intent: int | None = None
    alpha = float(selector_alpha_current(training_params, policy))
    if alpha > 0.0 and float(np.random.random()) < alpha:
        try:
            with torch.no_grad():
                logits, _ = policy.policy.get_intent_selector_outputs(selector_obs)
                dist = torch.distributions.Categorical(logits=logits)
                chosen = dist.sample().reshape(-1)
            chosen_intent = int(chosen[0].item())
            used_selector = True
        except Exception:
            chosen_intent = None

    if chosen_intent is None and bool(allow_uniform_fallback):
        chosen_intent = int(np.random.randint(0, num_intents))

    if chosen_intent is None:
        return {"obs": obs, "used_selector": False, "intent_index": None}

    setter = getattr(base_env, "set_offense_intent_state", None)
    if callable(setter):
        try:
            setter(int(chosen_intent), intent_active=True, intent_age=0)
        except Exception:
            pass

    visible = _extract_obs_scalar(batched_obs, "intent_visible", 0, default=1.0)
    patched_obs = clone_observation_dict(obs)
    patch_intent_in_observation(
        patched_obs,
        int(chosen_intent),
        num_intents,
        active=1.0,
        visible=float(visible),
        age_norm=0.0,
    )
    return {
        "obs": patched_obs,
        "used_selector": bool(used_selector),
        "intent_index": int(chosen_intent),
    }


def _build_callback_episode_collector(
    *,
    num_intents: int,
    max_obs_dim: int,
    max_action_dim: int,
) -> IntentDiversityCallback:
    return IntentDiversityCallback(
        enabled=True,
        num_intents=int(max(1, num_intents)),
        max_obs_dim=int(max_obs_dim),
        max_action_dim=int(max_action_dim),
    )


def _init_training_holdout_match_worker(
    env_args: argparse.Namespace,
    policy_path: str,
    device: str,
    progress_queue=None,
) -> None:
    global _TRAINING_HOLDOUT_WORKER_STATE
    import torch as _torch

    try:
        _torch.set_num_threads(1)
        _torch.set_num_interop_threads(1)
    except Exception:
        pass

    policy = load_ppo_for_inference(policy_path, device=device)
    _TRAINING_HOLDOUT_WORKER_STATE = {
        "env_args": env_args,
        "policy": policy,
        "device": str(device),
        "progress_queue": progress_queue,
        "wrapped_env_cache": {},
    }


def _get_or_create_training_holdout_wrapped_env(
    *,
    env_idx: int,
    opponent_policy_path: str,
    opponent_deterministic: bool,
) -> SelfPlayEnvWrapper:
    state = _TRAINING_HOLDOUT_WORKER_STATE or {}
    cache = state.setdefault("wrapped_env_cache", {})
    key = (int(env_idx), str(opponent_policy_path), bool(opponent_deterministic))
    wrapped_env = cache.get(key)
    if wrapped_env is not None:
        return wrapped_env

    env_args = state["env_args"]
    wrapped_env = SelfPlayEnvWrapper(
        setup_environment(env_args, Team.OFFENSE, env_idx=int(env_idx)),
        opponent_policy=str(opponent_policy_path),
        training_strategy=IllegalActionStrategy.SAMPLE_PROB,
        opponent_strategy=IllegalActionStrategy.SAMPLE_PROB,
        deterministic_opponent=bool(opponent_deterministic),
    )
    cache[key] = wrapped_env
    return wrapped_env


def _run_training_holdout_match_batch_worker(args: tuple[Any, ...]) -> dict[str, Any]:
    (
        rollout_steps_target,
        seed,
        env_idx,
        opponent_policy_path,
        player_deterministic,
        opponent_deterministic,
        max_obs_dim,
        max_action_dim,
        probe_feature_mode,
    ) = args

    state = _TRAINING_HOLDOUT_WORKER_STATE or {}
    env_args = state["env_args"]
    policy = state["policy"]
    progress_queue = state.get("progress_queue")
    training_params = _selector_training_params_dict(env_args)
    rng = np.random.default_rng(int(seed))

    wrapped_env = _get_or_create_training_holdout_wrapped_env(
        env_idx=int(env_idx),
        opponent_policy_path=str(opponent_policy_path),
        opponent_deterministic=bool(opponent_deterministic),
    )
    obs, _ = wrapped_env.reset(seed=int(rng.integers(0, 2**31 - 1)))
    selector_state = _ppo_like_apply_rollout_segment_start(
        env=wrapped_env,
        obs=obs,
        policy=policy,
        training_params=training_params,
        allow_uniform_fallback=False,
    )
    obs = selector_state["obs"]

    episodes: list[CompletedIntentEpisode] = []
    probe_episodes: list[CompletedIntentEpisode] = []
    collector = _build_callback_episode_collector(
        num_intents=int(getattr(env_args, "num_intents", 8)),
        max_obs_dim=int(max_obs_dim),
        max_action_dim=int(max_action_dim),
    )
    probe_buffer = IntentEpisodeBuffer() if probe_feature_mode else None
    boundary_counts: Counter[str] = Counter()
    segment_start_source_counts: Counter[str] = Counter()
    env_idx_histogram: Counter[int] = Counter()
    possession_count = 1
    reset_count = 1
    start_source = (
        "selector"
        if bool(selector_state.get("used_selector", False))
        else "env_reset_random"
    )
    segment_start_source_counts[start_source] += 1
    rollout_step_idx = 0

    for _ in range(max(1, int(rollout_steps_target))):
        raw_action = _sample_rollout_action_like_ppo(
            policy,
            obs,
            deterministic=bool(player_deterministic),
        )
        training_action = _normalize_training_action_for_wrapper(raw_action, wrapped_env)
        next_obs, _, terminated, truncated, info = wrapped_env.step(training_action)

        if progress_queue is not None:
            try:
                progress_queue.put_nowait(1)
            except Exception:
                pass

        done = bool(terminated or truncated)
        base_env = getattr(wrapped_env, "unwrapped", wrapped_env)
        boundary_reason = None
        if not done:
            boundary_reason = selector_segment_boundary_reason(
                training_params,
                segment_length=int(getattr(base_env, "intent_age", 0)),
                info=info,
                intent_commitment_steps=int(getattr(base_env, "intent_commitment_steps", 0)),
            )
        info_for_callback = dict(info or {})
        if boundary_reason is not None:
            info_for_callback["intent_segment_boundary"] = 1.0
            info_for_callback["intent_segment_boundary_reason"] = str(boundary_reason)
        completed_probe: list[CompletedIntentEpisode] = []
        if probe_buffer is not None:
            probe_transition = _build_probe_transition(
                feature_mode=str(probe_feature_mode),
                policy=policy,
                next_obs=next_obs,
                training_action=training_action,
                info=info_for_callback,
                rollout_step_idx=int(getattr(collector, "_rollout_step_idx", 0)),
                max_obs_dim=int(max_obs_dim),
                max_action_dim=int(max_action_dim),
                env_idx=int(env_idx),
            )
            completed_probe = _buffer_on_step(
                probe_buffer,
                transition=probe_transition,
                info=info_for_callback,
                done=done,
            )

        completed = _assign_episode_env_idx(
            _collector_on_step(
                collector,
                next_obs=next_obs,
                training_action=training_action,
                info=info_for_callback,
                done=done,
            ),
            env_idx=int(env_idx),
        )

        rollout_step_idx = int(getattr(collector, "_rollout_step_idx", rollout_step_idx + 1))
        if done or boundary_reason is not None:
            collected_now = 0
            boundary_label = str(boundary_reason or "episode_end")
            for episode in completed:
                if not (episode.role_is_offense and episode.active_prefix_length > 0):
                    continue
                episodes.append(episode)
                boundary_counts[boundary_label] += 1
                collected_now += 1
                if episode.transitions:
                    env_idx_histogram[int(episode.transitions[0].env_idx)] += 1
            for probe_episode in completed_probe:
                if not (
                    probe_episode.role_is_offense
                    and probe_episode.active_prefix_length > 0
                ):
                    continue
                probe_episodes.append(probe_episode)
            if done:
                obs, _ = wrapped_env.reset(seed=int(rng.integers(0, 2**31 - 1)))
                reset_count += 1
                obs = _prepare_obs(policy, wrapped_env, obs)
                selector_state = _ppo_like_apply_rollout_segment_start(
                    env=wrapped_env,
                    obs=obs,
                    policy=policy,
                    training_params=training_params,
                    allow_uniform_fallback=False,
                )
                obs = selector_state["obs"]
                start_source = (
                    "selector"
                    if bool(selector_state.get("used_selector", False))
                    else "env_reset_random"
                )
                segment_start_source_counts[start_source] += 1
                possession_count += 1
            else:
                segment_start = apply_rollout_segment_start(
                    wrapped_env,
                    next_obs,
                    training_params=training_params,
                    unified_policy=policy,
                    opponent_policy=None,
                    user_team=Team.OFFENSE,
                    role_flag_offense=1.0,
                    allow_uniform_fallback=True,
                )
                obs = _prepare_obs(
                    policy,
                    wrapped_env,
                    segment_start.get("obs", next_obs),
                )
                start_source = (
                    "selector"
                    if bool(segment_start.get("used_selector", False))
                    else "uniform_fallback"
                )
                segment_start_source_counts[str(start_source)] += 1
            continue
        obs = next_obs
    return {
        "episodes": episodes,
        "probe_episodes": probe_episodes if probe_buffer is not None else None,
        "summary": {
            "boundary_reason_counts": dict(boundary_counts),
            "segment_start_source_counts": dict(segment_start_source_counts),
            "env_idx_histogram": dict(sorted(env_idx_histogram.items())),
            "possession_count": int(possession_count),
            "env_reset_count": int(reset_count),
            "rollout_steps": int(max(1, int(rollout_steps_target))),
        },
    }


def _collector_on_step(
    collector: IntentDiversityCallback,
    *,
    next_obs: Any,
    training_action: np.ndarray,
    info: dict[str, Any],
    done: bool,
) -> list[CompletedIntentEpisode]:
    collector.locals = {
        "infos": [dict(info or {})],
        "dones": [bool(done)],
        "actions": np.asarray(training_action, dtype=np.int64).reshape(1, -1),
        "new_obs": _single_obs_to_batched(next_obs),
    }
    collector._on_step()
    return collector._episodes.pop_completed()


def _assign_episode_env_idx(
    episodes: list[CompletedIntentEpisode],
    *,
    env_idx: int,
) -> list[CompletedIntentEpisode]:
    for episode in episodes:
        for transition in getattr(episode, "transitions", []) or []:
            transition.env_idx = int(env_idx)
    return episodes


def _buffer_on_step(
    buffer: IntentEpisodeBuffer,
    *,
    transition: IntentTransition,
    info: dict[str, Any],
    done: bool,
) -> list[CompletedIntentEpisode]:
    current_intent = buffer.current_intent_index(0)
    if (
        current_intent is not None
        and int(current_intent) != int(transition.intent_index)
        and not bool(done)
    ):
        buffer.close_episode(0)
    buffer.append(0, transition)
    if bool(done):
        buffer.close_episode(0, terminal_info=info)
    elif float((info or {}).get("intent_segment_boundary", 0.0)) > 0.5:
        buffer.close_episode(0, terminal_info=info)
    return buffer.pop_completed()


def _build_probe_transition(
    *,
    feature_mode: str,
    policy: Any,
    next_obs: Any,
    training_action: np.ndarray,
    info: dict[str, Any] | None,
    rollout_step_idx: int,
    max_obs_dim: int,
    max_action_dim: int,
    env_idx: int = 0,
) -> IntentTransition:
    if str(feature_mode) == "set_attention_pool":
        feature_override = _compute_policy_state_embedding(policy, next_obs)
        return _build_transition_single_obs(
            next_obs,
            training_action,
            max_obs_dim=int(max_obs_dim),
            max_action_dim=0,
            rollout_step_idx=int(rollout_step_idx),
            env_idx=int(env_idx),
            feature_override=feature_override,
            info=info,
        )
    return _build_transition_single_obs(
        next_obs,
        training_action,
        max_obs_dim=int(max_obs_dim),
        max_action_dim=int(max_action_dim),
        rollout_step_idx=int(rollout_step_idx),
        env_idx=int(env_idx),
        info=info,
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
    probe_feature_mode: str | None = None,
) -> tuple[list[CompletedIntentEpisode], dict[str, Any], list[CompletedIntentEpisode] | None]:
    policy = load_ppo_for_inference(policy_path, device=device)
    wrapped_env = SelfPlayEnvWrapper(
        setup_environment(env_args, Team.OFFENSE),
        opponent_policy=str(opponent_policy_path or policy_path),
        training_strategy=IllegalActionStrategy.SAMPLE_PROB,
        opponent_strategy=IllegalActionStrategy.SAMPLE_PROB,
        deterministic_opponent=bool(opponent_deterministic),
    )
    episodes: list[CompletedIntentEpisode] = []
    probe_episodes: list[CompletedIntentEpisode] = []
    collector = _build_callback_episode_collector(
        num_intents=int(getattr(env_args, "num_intents", 8)),
        max_obs_dim=int(max_obs_dim),
        max_action_dim=int(max_action_dim),
    )
    probe_buffer = IntentEpisodeBuffer() if probe_feature_mode else None
    boundary_counts: Counter[str] = Counter()
    possession_count = 0
    reset_count = 0
    rng = np.random.default_rng(0)
    input_diag_by_step: list[dict[str, Any]] = []
    kept_input_diags: list[dict[str, Any]] = []
    progress = _make_collection_progress(
        total=int(num_segments),
        desc=f"Collect forced z={int(target_intent)} segments",
    )

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

        input_diag_by_step = []

        while len(episodes) < int(num_segments):
            raw_action = _sample_rollout_action_like_ppo(
                policy,
                obs,
                deterministic=bool(player_deterministic),
            )
            training_action = _normalize_training_action_for_wrapper(raw_action, wrapped_env)
            next_obs, _, terminated, truncated, info = wrapped_env.step(training_action)
            next_obs = _prepare_obs(policy, wrapped_env, next_obs)
            step_input_diag = _inspect_single_obs_input(
                next_obs,
                target_intent=int(target_intent),
            )
            input_diag_by_step.append(step_input_diag)

            done = bool(terminated or truncated)
            base_env = getattr(wrapped_env, "unwrapped", wrapped_env)
            boundary_reason = None
            if not done:
                boundary_reason = _segment_boundary_reason(
                    info=info,
                    segment_length=int(getattr(base_env, "intent_age", 0)),
                    min_play_steps=int(getattr(env_args, "intent_selector_min_play_steps", 3)),
                    commitment_steps=int(getattr(env_args, "intent_commitment_steps", 4)),
                )

            info_for_callback = dict(info or {})
            if boundary_reason is not None:
                info_for_callback["intent_segment_boundary"] = 1.0
                info_for_callback["intent_segment_boundary_reason"] = str(boundary_reason)
            completed_probe: list[CompletedIntentEpisode] = []
            if probe_buffer is not None:
                probe_transition = _build_probe_transition(
                    feature_mode=str(probe_feature_mode),
                    policy=policy,
                    next_obs=next_obs,
                    training_action=training_action,
                    info=info_for_callback,
                    rollout_step_idx=int(getattr(collector, "_rollout_step_idx", 0)),
                    max_obs_dim=int(max_obs_dim),
                    max_action_dim=int(max_action_dim),
                )
                completed_probe = _buffer_on_step(
                    probe_buffer,
                    transition=probe_transition,
                    info=info_for_callback,
                    done=done,
                )
            completed = _collector_on_step(
                collector,
                next_obs=next_obs,
                training_action=training_action,
                info=info_for_callback,
                done=done,
            )
            if done or boundary_reason is not None:
                before_count = len(episodes)
                for episode in completed:
                    if not (episode.role_is_offense and episode.active_prefix_length > 0):
                        continue
                    episodes.append(episode)
                    boundary_counts[boundary_reason or "episode_end"] += 1
                    kept_input_diags.extend(
                        input_diag_by_step[: episode.active_prefix_length]
                    )
                for probe_episode in completed_probe:
                    if not (
                        probe_episode.role_is_offense
                        and probe_episode.active_prefix_length > 0
                    ):
                        continue
                    probe_episodes.append(probe_episode)
                collected_now = max(0, len(episodes) - before_count)
                if collected_now > 0:
                    progress.update(collected_now)
                input_diag_by_step.clear()
                if len(episodes) >= int(num_segments):
                    break
                if done:
                    obs, _ = wrapped_env.reset(seed=int(rng.integers(0, 2**31 - 1)))
                    reset_count += 1
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
        progress.close()
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
    return episodes, summary, (probe_episodes if probe_buffer is not None else None)


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
    probe_feature_mode: str | None = None,
) -> tuple[list[CompletedIntentEpisode], dict[str, Any], list[CompletedIntentEpisode] | None]:
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
    probe_episodes: list[CompletedIntentEpisode] = []
    collector = _build_callback_episode_collector(
        num_intents=int(getattr(env_args, "num_intents", 8)),
        max_obs_dim=int(max_obs_dim),
        max_action_dim=int(max_action_dim),
    )
    probe_buffer = IntentEpisodeBuffer() if probe_feature_mode else None
    boundary_counts: Counter[str] = Counter()
    segment_start_source_counts: Counter[str] = Counter()
    possession_count = 0
    reset_count = 0
    rng = np.random.default_rng(0)
    progress = _make_collection_progress(
        total=int(num_segments),
        desc="Collect natural intent segments",
    )

    try:
        obs, _ = wrapped_env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        reset_count += 1
        obs = _prepare_obs(policy, wrapped_env, obs)
        selector_state = initialize_rollout_selector_episode(
            env=wrapped_env,
            obs=obs,
            training_params=training_params,
            unified_policy=policy,
            opponent_policy=None,
            user_team=Team.OFFENSE,
            role_flag_offense=1.0,
        )
        obs = selector_state["obs"]
        selector_segment_index = int(selector_state["selector_segment_index"])
        start_source = (
            "selector"
            if bool(selector_state.get("used_selector", False))
            else "env_reset_random"
        )
        segment_start_source_counts[start_source] += 1
        possession_count += 1

        while len(episodes) < int(num_segments):
            raw_action = _sample_rollout_action_like_ppo(
                policy,
                obs,
                deterministic=bool(player_deterministic),
            )
            training_action = _normalize_training_action_for_wrapper(raw_action, wrapped_env)
            next_obs, _, terminated, truncated, info = wrapped_env.step(training_action)
            next_obs = _prepare_obs(policy, wrapped_env, next_obs)

            done = bool(terminated or truncated)
            base_env = getattr(wrapped_env, "unwrapped", wrapped_env)
            boundary_reason = None
            if not done:
                boundary_reason = selector_segment_boundary_reason(
                    training_params,
                    segment_length=int(getattr(base_env, "intent_age", 0)),
                    info=info,
                    intent_commitment_steps=int(getattr(base_env, "intent_commitment_steps", 0)),
                )
            info_for_callback = dict(info or {})
            if boundary_reason is not None:
                info_for_callback["intent_segment_boundary"] = 1.0
                info_for_callback["intent_segment_boundary_reason"] = str(boundary_reason)
            completed_probe: list[CompletedIntentEpisode] = []
            if probe_buffer is not None:
                probe_transition = _build_probe_transition(
                    feature_mode=str(probe_feature_mode),
                    policy=policy,
                    next_obs=next_obs,
                    training_action=training_action,
                    info=info_for_callback,
                    rollout_step_idx=int(getattr(collector, "_rollout_step_idx", 0)),
                    max_obs_dim=int(max_obs_dim),
                    max_action_dim=int(max_action_dim),
                )
                completed_probe = _buffer_on_step(
                    probe_buffer,
                    transition=probe_transition,
                    info=info_for_callback,
                    done=done,
                )
            completed = _collector_on_step(
                collector,
                next_obs=next_obs,
                training_action=training_action,
                info=info_for_callback,
                done=done,
            )

            if done or boundary_reason is not None:
                before_count = len(episodes)
                for episode in completed:
                    if not (episode.role_is_offense and episode.active_prefix_length > 0):
                        continue
                    episodes.append(episode)
                    boundary_counts[boundary_reason or "episode_end"] += 1
                for probe_episode in completed_probe:
                    if not (
                        probe_episode.role_is_offense
                        and probe_episode.active_prefix_length > 0
                    ):
                        continue
                    probe_episodes.append(probe_episode)
                collected_now = max(0, len(episodes) - before_count)
                if collected_now > 0:
                    progress.update(collected_now)
                if len(episodes) >= int(num_segments):
                    break
                if done:
                    obs, _ = wrapped_env.reset(seed=int(rng.integers(0, 2**31 - 1)))
                    reset_count += 1
                    obs = _prepare_obs(policy, wrapped_env, obs)
                    selector_state = initialize_rollout_selector_episode(
                        env=wrapped_env,
                        obs=obs,
                        training_params=training_params,
                        unified_policy=policy,
                        opponent_policy=None,
                        user_team=Team.OFFENSE,
                        role_flag_offense=1.0,
                    )
                    obs = selector_state["obs"]
                    selector_segment_index = int(selector_state["selector_segment_index"])
                    start_source = (
                        "selector"
                        if bool(selector_state.get("used_selector", False))
                        else "env_reset_random"
                    )
                    segment_start_source_counts[start_source] += 1
                    possession_count += 1
                else:
                    segment_start = _ppo_like_apply_rollout_segment_start(
                        env=wrapped_env,
                        obs=next_obs,
                        policy=policy,
                        training_params=training_params,
                        allow_uniform_fallback=True,
                    )
                    obs = segment_start.get("obs", next_obs)
                    selector_segment_index += 1
                    start_source = str(
                        "selector"
                        if bool(segment_start.get("used_selector", False))
                        else "uniform_fallback"
                    )
                    segment_start_source_counts[start_source] += 1
                continue

            obs = next_obs
    finally:
        progress.close()
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
    return episodes, summary, (probe_episodes if probe_buffer is not None else None)


def collect_natural_intent_segments_training_holdout_match(
    *,
    run_id: str,
    checkpoint_idx: int,
    policy_path: str,
    env_args: argparse.Namespace,
    num_segments: int,
    max_obs_dim: int,
    max_action_dim: int,
    player_deterministic: bool,
    opponent_deterministic: bool,
    device: str,
    probe_feature_mode: str | None = None,
) -> tuple[list[CompletedIntentEpisode], dict[str, Any], list[CompletedIntentEpisode] | None]:
    opponent_policy_paths, opponent_meta = _resolve_training_holdout_match_opponent_policy_paths(
        run_id=str(run_id),
        policy_path=str(policy_path),
        checkpoint_idx=int(checkpoint_idx),
        env_args=env_args,
    )
    training_params = _selector_training_params_dict(env_args)
    summary_policy = load_ppo_for_inference(str(policy_path), device=str(device))
    episodes: list[CompletedIntentEpisode] = []
    probe_episodes: list[CompletedIntentEpisode] = []
    boundary_counts: Counter[str] = Counter()
    segment_start_source_counts: Counter[str] = Counter()
    env_idx_histogram: Counter[int] = Counter()
    possession_count = 0
    reset_count = 0
    rollout_steps_per_env = max(1, int(getattr(env_args, "n_steps", 1)))
    progress = _make_collection_progress(
        total=int(len(opponent_policy_paths) * rollout_steps_per_env),
        desc="Collect training-matched rollout window",
    )

    try:
        batches: list[tuple[Any, ...]] = []
        seed_rng = np.random.default_rng(0)
        for env_idx, opponent_path in enumerate(opponent_policy_paths):
            batches.append(
                (
                    int(rollout_steps_per_env),
                    int(seed_rng.integers(0, 2**31 - 1)),
                    int(env_idx),
                    str(opponent_path),
                    bool(player_deterministic),
                    bool(opponent_deterministic),
                    int(max_obs_dim),
                    int(max_action_dim),
                    None if probe_feature_mode is None else str(probe_feature_mode),
                )
            )

        ctx = mp.get_context("spawn")
        progress_queue = ctx.Queue()
        max_workers = min(
            max(1, len(opponent_policy_paths)),
            max(1, len(batches)),
            max(1, int(os.cpu_count() or 1)),
        )
        with ProcessPoolExecutor(
            max_workers=int(max_workers),
            mp_context=ctx,
            initializer=_init_training_holdout_match_worker,
            initargs=(
                env_args,
                str(policy_path),
                str(device),
                progress_queue,
            ),
        ) as executor:
            futures = {
                executor.submit(_run_training_holdout_match_batch_worker, batch): batch
                for batch in batches
            }
            pending = set(futures.keys())
            while pending:
                done, pending = wait(
                    pending,
                    timeout=1.0,
                    return_when=FIRST_COMPLETED,
                )
                while True:
                    try:
                        inc = int(progress_queue.get_nowait())
                        progress.update(
                            min(inc, max(0, int(progress.total) - int(progress.n)))
                        )
                    except queue.Empty:
                        break
                    except Exception:
                        break
                for future in done:
                    payload = future.result()
                    episodes.extend(payload.get("episodes", []) or [])
                    probe_episodes.extend(payload.get("probe_episodes", []) or [])
                    worker_summary = dict(payload.get("summary", {}) or {})
                    boundary_counts.update(
                        {
                            str(k): int(v)
                            for k, v in dict(
                                worker_summary.get("boundary_reason_counts", {}) or {}
                            ).items()
                        }
                    )
                    segment_start_source_counts.update(
                        {
                            str(k): int(v)
                            for k, v in dict(
                                worker_summary.get("segment_start_source_counts", {}) or {}
                            ).items()
                        }
                    )
                    env_idx_histogram.update(
                        {
                            int(k): int(v)
                            for k, v in dict(
                                worker_summary.get("env_idx_histogram", {}) or {}
                            ).items()
                        }
                    )
                    possession_count += int(worker_summary.get("possession_count", 0))
                    reset_count += int(worker_summary.get("env_reset_count", 0))

            while True:
                try:
                    inc = int(progress_queue.get_nowait())
                    progress.update(min(inc, max(0, int(progress.total) - int(progress.n))))
                except queue.Empty:
                    break
                except Exception:
                    break
    finally:
        progress.close()

    labels = [int(ep.intent_index) for ep in episodes]
    summary = {
        "collection_mode": "training_holdout_match",
        "num_eval_envs": int(len(opponent_policy_paths)),
        "collector": "process_pool_eval_workers",
        "rollout_steps_per_env": int(rollout_steps_per_env),
        "total_rollout_steps": int(len(opponent_policy_paths) * rollout_steps_per_env),
        "collected_segments": int(len(episodes)),
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
        "env_idx_histogram": dict(sorted(env_idx_histogram.items())),
        "selector_runtime_active": bool(
            selector_runtime_enabled(
                training_params,
                summary_policy,
            )
        ),
        "selector_alpha_current": float(
            selector_alpha_current(
                training_params,
                summary_policy,
            )
        ),
        "opponent_policy_paths": [str(path) for path in opponent_policy_paths],
        "opponent_policy_basenames": [
            os.path.basename(str(path)) for path in opponent_policy_paths
        ],
        "opponent_sampling_meta": opponent_meta,
    }
    return episodes, summary, (probe_episodes if probe_feature_mode else None)


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


def _stratified_train_test_split(
    y_np: np.ndarray,
    *,
    test_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_np, dtype=np.int64).reshape(-1)
    rng = np.random.default_rng(int(seed))
    train_idx: list[int] = []
    test_idx: list[int] = []
    for cls in sorted(np.unique(y).tolist()):
        cls_idx = np.flatnonzero(y == int(cls))
        if cls_idx.size == 0:
            continue
        cls_idx = np.array(cls_idx, copy=True)
        rng.shuffle(cls_idx)
        if cls_idx.size <= 1:
            train_idx.extend(cls_idx.tolist())
            continue
        n_test = int(round(float(cls_idx.size) * float(test_fraction)))
        n_test = max(1, min(int(cls_idx.size) - 1, n_test))
        test_idx.extend(cls_idx[:n_test].tolist())
        train_idx.extend(cls_idx[n_test:].tolist())
    return (
        np.asarray(sorted(train_idx), dtype=np.int64),
        np.asarray(sorted(test_idx), dtype=np.int64),
    )


def _fit_linear_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    num_classes: int,
    seed: int,
    device: str,
) -> tuple[torch.nn.Module, np.ndarray, np.ndarray]:
    x_np = np.asarray(x_train, dtype=np.float32)
    y_np = np.asarray(y_train, dtype=np.int64).reshape(-1)
    mean = x_np.mean(axis=0, keepdims=True).astype(np.float32, copy=False)
    std = x_np.std(axis=0, keepdims=True).astype(np.float32, copy=False)
    std = np.where(std > 1e-6, std, 1.0).astype(np.float32, copy=False)
    x_scaled = (x_np - mean) / std

    torch.manual_seed(int(seed))
    model = torch.nn.Linear(int(x_scaled.shape[1]), int(num_classes)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-2, weight_decay=1e-4)
    x_tensor = torch.as_tensor(x_scaled, dtype=torch.float32, device=device)
    y_tensor = torch.as_tensor(y_np, dtype=torch.long, device=device)

    model.train()
    for _ in range(400):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_tensor)
        loss = torch.nn.functional.cross_entropy(logits, y_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    return model, mean, std


def _knn_metrics(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    k: int,
) -> dict[str, float]:
    if x_train.shape[0] == 0 or x_test.shape[0] == 0:
        return {"knn_top1_acc": 0.0, "knn_purity": 0.0}
    k_eff = max(1, min(int(k), int(x_train.shape[0])))
    diff = x_test[:, None, :] - x_train[None, :, :]
    dist = np.sum(diff * diff, axis=2)
    nn_idx = np.argpartition(dist, kth=k_eff - 1, axis=1)[:, :k_eff]
    nn_labels = y_train[nn_idx]
    purity = np.mean(nn_labels == y_test[:, None])
    pred_labels: list[int] = []
    for row in nn_labels:
        counts = np.bincount(row, minlength=max(int(np.max(y_train)) + 1, 1))
        pred_labels.append(int(np.argmax(counts)))
    pred_np = np.asarray(pred_labels, dtype=np.int64)
    return {
        "knn_top1_acc": float(np.mean(pred_np == y_test)),
        "knn_purity": float(purity),
    }


def compute_heldout_probe_metrics(
    episodes: list[CompletedIntentEpisode],
    *,
    max_obs_dim: int,
    max_action_dim: int,
    num_intents: int,
    test_fraction: float,
    knn_k: int,
    seed: int,
    device: str,
) -> dict[str, Any]:
    x_np, y_np = compute_episode_embeddings(
        episodes,
        max_obs_dim=int(max_obs_dim),
        max_action_dim=int(max_action_dim),
    )
    y_np = np.asarray(y_np, dtype=np.int64).reshape(-1)
    result: dict[str, Any] = {
        "num_samples": int(y_np.size),
        "num_classes_present": int(len(np.unique(y_np))) if y_np.size else 0,
        "class_histogram": np.bincount(
            y_np, minlength=max(1, int(num_intents))
        ).tolist()
        if y_np.size
        else [],
        "test_fraction": float(test_fraction),
        "seed": int(seed),
        "knn_k": int(knn_k),
    }
    if y_np.size < 4 or len(np.unique(y_np)) < 2:
        result["error"] = "insufficient_samples_or_classes"
        return result

    train_idx, test_idx = _stratified_train_test_split(
        y_np,
        test_fraction=float(test_fraction),
        seed=int(seed),
    )
    if train_idx.size == 0 or test_idx.size == 0:
        result["error"] = "empty_train_or_test_split"
        return result

    x_train = np.asarray(x_np[train_idx], dtype=np.float32)
    y_train = np.asarray(y_np[train_idx], dtype=np.int64)
    x_test = np.asarray(x_np[test_idx], dtype=np.float32)
    y_test = np.asarray(y_np[test_idx], dtype=np.int64)
    result["train_class_histogram"] = np.bincount(
        y_train, minlength=max(1, int(num_intents))
    ).tolist()
    result["test_class_histogram"] = np.bincount(
        y_test, minlength=max(1, int(num_intents))
    ).tolist()
    result["num_train"] = int(y_train.size)
    result["num_test"] = int(y_test.size)

    probe, mean, std = _fit_linear_probe(
        x_train,
        y_train,
        num_classes=max(1, int(num_intents)),
        seed=int(seed),
        device=device,
    )
    x_test_scaled = ((x_test - mean) / std).astype(np.float32, copy=False)
    x_test_tensor = torch.as_tensor(x_test_scaled, dtype=torch.float32, device=device)
    with torch.no_grad():
        logits = probe(x_test_tensor)
        pred = torch.argmax(logits, dim=-1).detach().cpu().numpy().astype(np.int64, copy=False)
    logits_np = logits.detach().cpu().numpy().astype(np.float32, copy=False)
    result["linear_probe_top1_acc"] = float(np.mean(pred == y_test))
    result["linear_probe_auc_ovr_macro"] = (
        float(
            IntentDiversityCallback._multiclass_auc_ovr_macro(
                logits_np,
                y_test,
                num_classes=max(1, int(num_intents)),
            )
        )
        if y_test.size and len(np.unique(y_test)) > 1
        else None
    )
    result["linear_probe_predicted_class_histogram"] = np.bincount(
        pred, minlength=max(1, int(num_intents))
    ).tolist()
    row_confusion = np.zeros(
        (max(1, int(num_intents)), max(1, int(num_intents))), dtype=np.int64
    )
    for true_z, pred_z in zip(y_test.tolist(), pred.tolist()):
        row_confusion[int(true_z), int(pred_z)] += 1
    row_den = np.clip(row_confusion.sum(axis=1, keepdims=True), 1, None)
    result["linear_probe_confusion_counts"] = row_confusion.tolist()
    result["linear_probe_row_normalized_confusion"] = (row_confusion / row_den).tolist()

    x_train_scaled = ((x_train - mean) / std).astype(np.float32, copy=False)
    result.update(
        _knn_metrics(
            x_train_scaled,
            y_train,
            x_test_scaled,
            y_test,
            k=int(knn_k),
        )
    )
    return result


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
        if "recomputed_top1_acc" in result:
            lines.append(
                f"recomputed_top1_acc={float(result.get('recomputed_top1_acc', 0.0)):.4f}"
            )
        if "recomputed_auc_ovr_macro" in result:
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
        if "target_prediction_rate_mean" in result:
            lines.append(f"target_prediction_rate_mean={float(result.get('target_prediction_rate_mean', 0.0)):.4f}")
        if "target_prediction_rate_min" in result:
            lines.append(f"target_prediction_rate_min={float(result.get('target_prediction_rate_min', 0.0)):.4f}")
        dominant = dict(result.get("dominant_prediction_by_intent", {}) or {})
        if dominant:
            lines.append(f"dominant_prediction_by_intent={json.dumps(dominant, sort_keys=True)}")
    else:
        if "target_intent" in result:
            lines.append(f"target_intent={int(result.get('target_intent', 0))}")
        if "target_prediction_rate" in result:
            lines.append(f"target_prediction_rate={float(result.get('target_prediction_rate', 0.0)):.4f}")
        if "mean_target_probability" in result:
            lines.append(f"mean_target_probability={float(result.get('mean_target_probability', 0.0)):.4f}")
        if "mean_max_probability" in result:
            lines.append(f"mean_max_probability={float(result.get('mean_max_probability', 0.0)):.4f}")
    heldout = dict(result.get("heldout_probe", {}) or {})
    if heldout:
        feature_mode = str(heldout.get("feature_mode", "")).strip()
        if feature_mode:
            lines.append(f"heldout_probe_feature_mode={feature_mode}")
        lines.append(
            f"heldout_linear_probe_top1_acc={float(heldout.get('linear_probe_top1_acc', 0.0)):.4f}"
        )
        heldout_auc = heldout.get("linear_probe_auc_ovr_macro", None)
        lines.append(
            "heldout_linear_probe_auc_ovr_macro="
            + ("null" if heldout_auc is None else f"{float(heldout_auc):.4f}")
        )
        lines.append(
            f"heldout_knn_top1_acc={float(heldout.get('knn_top1_acc', 0.0)):.4f}"
        )
        lines.append(
            f"heldout_knn_purity={float(heldout.get('knn_purity', 0.0)):.4f}"
        )
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
        "--training-holdout-match",
        action="store_true",
        help=(
            "For --natural-intents only: recollect segments using the logged per-env "
            "training opponent assignment and offense-env count for this alternation."
        ),
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
    parser.add_argument(
        "--heldout-probe",
        action="store_true",
        help="Compute a held-out linear probe and kNN purity metric on the freshly collected segments.",
    )
    parser.add_argument(
        "--heldout-test-fraction",
        type=float,
        default=0.3,
        help="Test split fraction for the held-out probe.",
    )
    parser.add_argument(
        "--heldout-knn-k",
        type=int,
        default=5,
        help="Neighborhood size for the held-out kNN metrics.",
    )
    parser.add_argument(
        "--heldout-seed",
        type=int,
        default=0,
        help="Random seed for held-out probe train/test split and initialization.",
    )
    parser.add_argument(
        "--heldout-probe-feature-mode",
        type=str,
        default="disc_episode",
        choices=["disc_episode", "set_attention_pool"],
        help="Episode feature object to use for the held-out probe.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if bool(args.training_holdout_match):
        if not bool(args.natural_intents):
            raise RuntimeError("--training-holdout-match requires --natural-intents.")
        if args.opponent_policy_path:
            raise RuntimeError(
                "--training-holdout-match uses the logged per-env opponent assignment; "
                "do not also pass --opponent-policy-path."
            )
    if not os.path.isfile(args.run_id_or_policy_path) and not os.path.isdir(args.run_id_or_policy_path):
        setup_mlflow(verbose=False)
    heldout_feature_mode = str(args.heldout_probe_feature_mode)
    probe_only_without_disc = bool(args.heldout_probe) and heldout_feature_mode == "set_attention_pool" and not bool(args.disc_path)
    policy_path, disc_path, inferred_run_id = _resolve_policy_and_disc_paths(
        args.run_id_or_policy_path,
        checkpoint_idx=args.alternation_index,
        disc_path=args.disc_path,
        require_disc=not bool(probe_only_without_disc),
    )
    policy = load_ppo_for_inference(policy_path, device=args.device)
    env_args = _build_env_args(inferred_run_id, policy)
    disc_bundle = (
        _load_discriminator_checkpoint(str(disc_path), device=args.device)
        if disc_path
        else None
    )
    disc_config = dict(disc_bundle[1].get("config", {}) or {}) if disc_bundle else {}
    num_intents = int(disc_config.get("num_intents", _infer_num_intents(policy, env_args)))
    default_max_obs_dim = int(disc_config.get("max_obs_dim", 256))
    default_max_action_dim = int(disc_config.get("max_action_dim", 16))
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
        requested_intents = list(range(int(num_intents)))
    else:
        requested_intents = [int(args.intent_index)]

    result = {
        "policy_path": os.path.abspath(policy_path),
        "opponent_policy_path": os.path.abspath(opponent_policy_path),
        "requested_segments": int(args.episodes),
    }
    if disc_path is not None:
        result["discriminator_checkpoint_path"] = os.path.abspath(disc_path)
    if disc_bundle is not None:
        result["disc_checkpoint_config"] = disc_config
        result["disc_checkpoint_meta"] = dict(disc_bundle[1].get("meta", {}) or {})
    elif bool(probe_only_without_disc):
        result["discriminator_checkpoint_skipped"] = True
        result["discriminator_checkpoint_skip_reason"] = (
            "heldout_probe_set_attention_pool_does_not_require_discriminator"
        )
    if bool(args.natural_intents):
        if bool(args.training_holdout_match):
            checkpoint_idx = (
                int(args.alternation_index)
                if args.alternation_index is not None
                else _extract_checkpoint_index(policy_path)
            )
            if inferred_run_id is None or checkpoint_idx is None:
                raise RuntimeError(
                    "--training-holdout-match requires an MLflow-backed run and a resolvable alternation index."
                )
            episodes, collection, probe_episodes = (
                collect_natural_intent_segments_training_holdout_match(
                    run_id=str(inferred_run_id),
                    checkpoint_idx=int(checkpoint_idx),
                    policy_path=policy_path,
                    env_args=env_args,
                    num_segments=int(args.episodes),
                    max_obs_dim=int(default_max_obs_dim),
                    max_action_dim=int(default_max_action_dim),
                    player_deterministic=bool(args.player_deterministic),
                    opponent_deterministic=bool(args.opponent_deterministic),
                    device=args.device,
                    probe_feature_mode=(
                        str(args.heldout_probe_feature_mode)
                        if bool(args.heldout_probe)
                        else None
                    ),
                )
            )
        else:
            episodes, collection, probe_episodes = collect_natural_intent_segments(
                policy_path=policy_path,
                env_args=env_args,
                num_segments=int(args.episodes),
                max_obs_dim=int(default_max_obs_dim),
                max_action_dim=int(default_max_action_dim),
                player_deterministic=bool(args.player_deterministic),
                opponent_deterministic=bool(args.opponent_deterministic),
                device=args.device,
                opponent_policy_path=opponent_policy_path,
                probe_feature_mode=(
                    str(args.heldout_probe_feature_mode)
                    if bool(args.heldout_probe)
                    else None
                ),
            )
        if disc_bundle is not None:
            result.update(
                score_segments_general(
                    episodes,
                    disc_bundle=disc_bundle,
                    device=args.device,
                )
            )
        result["natural_intents"] = True
        result["collection"] = collection
        if bool(args.heldout_probe):
            heldout_episodes = (
                probe_episodes
                if str(args.heldout_probe_feature_mode) == "set_attention_pool"
                and probe_episodes is not None
                else episodes
            )
            heldout_max_obs_dim = (
                _infer_feature_dim_from_episodes(
                    heldout_episodes,
                    default=int(default_max_obs_dim),
                )
                if heldout_feature_mode == "set_attention_pool"
                else int(default_max_obs_dim)
            )
            result["heldout_probe"] = compute_heldout_probe_metrics(
                heldout_episodes,
                max_obs_dim=int(heldout_max_obs_dim),
                max_action_dim=(
                    0
                    if heldout_feature_mode == "set_attention_pool"
                    else int(default_max_action_dim)
                ),
                num_intents=int(num_intents),
                test_fraction=float(args.heldout_test_fraction),
                knn_k=int(args.heldout_knn_k),
                seed=int(args.heldout_seed),
                device=args.device,
            )
            result["heldout_probe"]["feature_mode"] = heldout_feature_mode
    else:
        per_intent: dict[str, Any] = {}
        prediction_matrix_counts: list[list[int]] = []
        target_rates: list[float] = []
        dominant_prediction_by_intent: dict[str, int] = {}
        collection_by_intent: dict[str, Any] = {}
        all_episodes: list[CompletedIntentEpisode] = []
        all_probe_episodes: list[CompletedIntentEpisode] = []

        for target_intent in requested_intents:
            episodes, collection, probe_episodes = collect_single_intent_segments(
                policy_path=policy_path,
                env_args=env_args,
                target_intent=int(target_intent),
                num_segments=int(args.episodes),
                max_obs_dim=int(default_max_obs_dim),
                max_action_dim=int(default_max_action_dim),
                player_deterministic=bool(args.player_deterministic),
                opponent_deterministic=bool(args.opponent_deterministic),
                device=args.device,
                opponent_policy_path=opponent_policy_path,
                probe_feature_mode=(
                    str(args.heldout_probe_feature_mode)
                    if bool(args.heldout_probe)
                    else None
                ),
            )
            intent_key = str(int(target_intent))
            collection_by_intent[intent_key] = collection
            all_episodes.extend(episodes)
            if probe_episodes is not None:
                all_probe_episodes.extend(probe_episodes)
            if disc_bundle is not None:
                scored = score_single_intent_segments(
                    episodes,
                    disc_bundle=disc_bundle,
                    device=args.device,
                )
                per_intent[intent_key] = scored
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
            if disc_bundle is not None:
                result.update(per_intent[intent_key])
            result["collection"] = collection_by_intent[intent_key]
        else:
            result.update(
                {
                    "all_intents": True,
                    "num_intents": int(num_intents),
                    "collection_by_intent": collection_by_intent,
                }
            )
            if disc_bundle is not None:
                result.update(
                    {
                        "per_intent": per_intent,
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
            if bool(args.heldout_probe):
                heldout_episodes = (
                    all_probe_episodes
                    if heldout_feature_mode == "set_attention_pool"
                    and all_probe_episodes
                    else all_episodes
                )
                heldout_max_obs_dim = (
                    _infer_feature_dim_from_episodes(
                        heldout_episodes,
                        default=int(default_max_obs_dim),
                    )
                    if heldout_feature_mode == "set_attention_pool"
                    else int(default_max_obs_dim)
                )
                result["heldout_probe"] = compute_heldout_probe_metrics(
                    heldout_episodes,
                    max_obs_dim=int(heldout_max_obs_dim),
                    max_action_dim=(
                        0
                        if heldout_feature_mode == "set_attention_pool"
                        else int(default_max_action_dim)
                    ),
                    num_intents=int(num_intents),
                    test_fraction=float(args.heldout_test_fraction),
                    knn_k=int(args.heldout_knn_k),
                    seed=int(args.heldout_seed),
                    device=args.device,
                )
                result["heldout_probe"]["feature_mode"] = heldout_feature_mode
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
