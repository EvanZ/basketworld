#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import json
import multiprocessing as mp
import os
import queue
import random
import re
import sys
import tempfile
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, silhouette_score
from sklearn.preprocessing import StandardScaler
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basketworld.envs.basketworld_env_v2 import Team
from basketworld.policies import SetAttentionDualCriticPolicy, SetAttentionExtractor
from basketworld.utils.action_resolution import (
    IllegalActionStrategy,
)
from basketworld.utils.intent_discovery import (
    CompletedIntentEpisode,
    IntentDiscriminator,
    IntentEpisodeBuffer,
    IntentTransition,
    build_padded_episode_batch,
    compute_episode_embeddings,
    extract_action_features_for_env,
    flatten_observation_for_env,
    load_intent_discriminator_from_checkpoint,
)
from basketworld.utils.intent_policy_sensitivity import (
    clone_observation_dict,
    infer_num_intents,
    patch_intent_in_observation,
)
from basketworld.utils.intent_pca import (
    SUMMARY_FEATURE_NAMES,
    build_summary_feature,
    infer_outcome_from_episode_info,
)
from basketworld.utils.mlflow_config import setup_mlflow
from basketworld.utils.mlflow_params import get_mlflow_params
from basketworld.utils.mlflow_params import get_mlflow_training_params
from basketworld.utils.policy_loading import load_ppo_for_inference
from basketworld.utils.policies import PassBiasDualCriticPolicy, PassBiasMultiInputPolicy
from basketworld.utils.self_play_wrapper import SelfPlayEnvWrapper
from train.config import get_args
from train.env_factory import setup_environment

_PCA_WORKER_STATE = {}


def _policy_has_intent_selector(model: PPO) -> bool:
    policy_obj = getattr(model, "policy", None)
    return bool(
        policy_obj is not None
        and hasattr(policy_obj, "has_intent_selector")
        and policy_obj.has_intent_selector()
    )


def _selector_alpha_current(
    model: PPO,
    env_args: argparse.Namespace,
) -> float:
    t = int(getattr(model, "num_timesteps", 0))
    start = float(getattr(env_args, "intent_selector_alpha_start", 0.0))
    end = float(getattr(env_args, "intent_selector_alpha_end", 1.0))
    warmup = max(0, int(getattr(env_args, "intent_selector_alpha_warmup_steps", 0)))
    ramp = max(0, int(getattr(env_args, "intent_selector_alpha_ramp_steps", 1)))
    if t < warmup:
        return float(start)
    if ramp <= 0:
        return float(end)
    progress = min(1.0, max(0.0, (t - warmup) / float(ramp)))
    return float(start + progress * (end - start))


def _resolve_intent_source_mode(
    requested_mode: str,
    model: PPO,
    env_args: argparse.Namespace,
) -> str:
    mode = str(requested_mode).strip().lower()
    if mode not in {"training_match", "env", "selector"}:
        raise ValueError(f"Unsupported intent source mode: {requested_mode!r}")
    selector_enabled = bool(getattr(env_args, "intent_selector_enabled", False))
    selector_available = selector_enabled and _policy_has_intent_selector(model)
    if mode == "env":
        return "env"
    if mode == "selector":
        return "selector" if selector_available else "env"
    return "selector" if selector_available else "env"


def _maybe_apply_selector_intent_start(
    *,
    obs: dict,
    wrapped_env: SelfPlayEnvWrapper,
    unified_policy: PPO,
    env_args: argparse.Namespace,
    rng: np.random.Generator,
    intent_source_mode: str,
) -> dict[str, object]:
    mode = _resolve_intent_source_mode(intent_source_mode, unified_policy, env_args)
    result: dict[str, object] = {
        "requested_intent_source_mode": str(intent_source_mode),
        "resolved_intent_source_mode": mode,
        "selector_enabled": bool(getattr(env_args, "intent_selector_enabled", False)),
        "selector_available": bool(_policy_has_intent_selector(unified_policy)),
        "selector_alpha": 0.0,
        "selector_applied": False,
        "selected_intent_index": None,
    }
    if mode != "selector":
        return result
    if not isinstance(obs, dict):
        return result
    if not bool(getattr(wrapped_env.unwrapped, "intent_active", False)):
        return result

    alpha = (
        1.0
        if str(intent_source_mode).strip().lower() == "selector"
        else _selector_alpha_current(unified_policy, env_args)
    )
    result["selector_alpha"] = float(alpha)
    if alpha <= 0.0:
        return result
    if str(intent_source_mode).strip().lower() == "training_match":
        if float(rng.random()) >= float(alpha):
            return result

    num_intents = int(
        max(
            1,
            getattr(env_args, "num_intents", infer_num_intents(unified_policy, default=8)),
        )
    )
    selector_obs = clone_observation_dict(obs)
    patch_intent_in_observation(
        selector_obs,
        0,
        num_intents,
        active=0.0,
        visible=0.0,
        age_norm=0.0,
    )
    with torch.no_grad():
        logits = unified_policy.policy.get_intent_selector_logits(selector_obs)
        dist = torch.distributions.Categorical(logits=logits)
        chosen_z = int(dist.sample().reshape(-1)[0].item())

    visible = 1.0
    try:
        visible = float(np.asarray(obs.get("intent_visible", 1.0), dtype=np.float32).reshape(-1)[0])
    except Exception:
        visible = 1.0
    wrapped_env.set_offense_intent_state(chosen_z, intent_active=True, intent_age=0)
    patch_intent_in_observation(
        obs,
        chosen_z,
        num_intents,
        active=1.0,
        visible=float(visible),
        age_norm=0.0,
    )
    result["selector_applied"] = True
    result["selected_intent_index"] = int(chosen_z)
    return result


def _latest_zip_in_dir(dir_path: str) -> str | None:
    candidates = sorted(glob.glob(os.path.join(dir_path, "*.zip")), key=os.path.getmtime)
    return candidates[-1] if candidates else None


def _find_unified_checkpoint_in_dir(dir_path: str, checkpoint_idx: int | None) -> str | None:
    candidates = sorted(glob.glob(os.path.join(dir_path, "*.zip")), key=os.path.getmtime)
    if checkpoint_idx is None:
        return candidates[-1] if candidates else None

    target_idx = int(checkpoint_idx)
    matches: list[tuple[int, str]] = []
    for path in candidates:
        idx = _extract_checkpoint_index(path)
        if idx == target_idx:
            matches.append((idx, path))
    return matches[-1][1] if matches else None


def _download_unified_checkpoint(run_id: str, checkpoint_idx: int | None) -> str:
    client = mlflow.tracking.MlflowClient()
    artifacts = client.list_artifacts(run_id, "models")
    pattern = re.compile(r"unified_(?:alternation|iter)_(\d+)\.zip$")
    candidates: list[tuple[int, str]] = []
    for item in artifacts:
        match = pattern.search(item.path)
        if match:
            candidates.append((int(match.group(1)), item.path))
    artifact_path = None
    if checkpoint_idx is not None:
        target_idx = int(checkpoint_idx)
        for idx, path in candidates:
            if idx == target_idx:
                artifact_path = path
                break
        if artifact_path is None:
            raise RuntimeError(
                f"No unified checkpoint with alternation/index={target_idx} found for run_id={run_id}"
            )
        print(
            f"[IntentPCA] Using requested unified checkpoint artifact: {artifact_path}",
            flush=True,
        )
    elif candidates:
        candidates.sort(key=lambda item: item[0])
        artifact_path = candidates[-1][1]
        print(
            f"[IntentPCA] Using latest unified checkpoint artifact: {artifact_path}",
            flush=True,
        )
    else:
        fallback = [item.path for item in artifacts if item.path.endswith("unified_latest.zip")]
        if not fallback:
            raise RuntimeError(f"No unified policy artifacts found for run_id={run_id}")
        artifact_path = fallback[-1]
        print(
            f"[IntentPCA] Using fallback unified checkpoint artifact: {artifact_path}",
            flush=True,
        )
    return _download_artifact_cached(
        client,
        run_id,
        artifact_path,
        cache_root=os.path.join(".opponent_cache", run_id),
        log_prefix="[IntentPCA]",
    )


def resolve_policy_path(input_arg: str, checkpoint_idx: int | None = None) -> tuple[str, str | None]:
    if os.path.isfile(input_arg):
        if checkpoint_idx is not None:
            print(
                f"[IntentPCA] Ignoring --alternation-index={int(checkpoint_idx)} because an explicit policy file was provided.",
                flush=True,
            )
        print(f"[IntentPCA] Using local policy file: {input_arg}", flush=True)
        return input_arg, None
    if os.path.isdir(input_arg):
        selected = _find_unified_checkpoint_in_dir(input_arg, checkpoint_idx)
        if selected is not None:
            if checkpoint_idx is None:
                print(f"[IntentPCA] Using latest local policy in directory: {selected}", flush=True)
            else:
                print(
                    f"[IntentPCA] Using requested local policy in directory: {selected}",
                    flush=True,
                )
            return selected, None
        if checkpoint_idx is not None:
            raise RuntimeError(
                f"No local unified checkpoint with alternation/index={int(checkpoint_idx)} found in {input_arg}"
            )
    cache_dir = os.path.join(".opponent_cache", input_arg)
    if os.path.isdir(cache_dir):
        selected = _find_unified_checkpoint_in_dir(cache_dir, checkpoint_idx)
        if selected is not None:
            if checkpoint_idx is None:
                print(f"[IntentPCA] Using cached policy checkpoint: {selected}", flush=True)
            else:
                print(
                    f"[IntentPCA] Using requested cached policy checkpoint: {selected}",
                    flush=True,
                )
            return selected, input_arg
        if checkpoint_idx is not None:
            print(
                "[IntentPCA] Requested checkpoint "
                f"alternation/index={int(checkpoint_idx)} not found in cache for run_id={input_arg}; "
                "trying MLflow artifacts.",
                flush=True,
            )
    if os.path.isfile(f"{input_arg}.zip"):
        if checkpoint_idx is not None:
            print(
                f"[IntentPCA] Ignoring --alternation-index={int(checkpoint_idx)} because an explicit policy file was provided.",
                flush=True,
            )
        print(f"[IntentPCA] Using local policy file: {input_arg}.zip", flush=True)
        return f"{input_arg}.zip", None
    return _download_unified_checkpoint(input_arg, checkpoint_idx), input_arg


def _extract_checkpoint_index(path_or_name: str) -> int | None:
    name = os.path.basename(str(path_or_name))
    match = re.search(r"_(?:alternation|iter)_(\d+)\.(?:zip|pt)$", name)
    if not match:
        return None
    return int(match.group(1))


def _intent_pca_mlflow_artifact_path(policy_path: str) -> str:
    name = os.path.basename(str(policy_path))
    match = re.search(r"unified_(alternation|iter)_(\d+)\.zip$", name)
    if match:
        return f"analysis/intent_pca/{match.group(1)}_{int(match.group(2))}"
    if name == "unified_latest.zip":
        return "analysis/intent_pca/latest"
    stem = Path(name).stem
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._-")
    if not safe_stem:
        safe_stem = "custom"
    return f"analysis/intent_pca/{safe_stem}"


def _download_matching_intent_disc(run_id: str, checkpoint_idx: int | None) -> str | None:
    client = mlflow.tracking.MlflowClient()
    artifacts = client.list_artifacts(run_id, "models")
    pattern = re.compile(r"intent_disc_(?:alternation|iter)_(\d+)\.pt$")
    candidates: list[tuple[int, str]] = []
    for item in artifacts:
        match = pattern.search(item.path)
        if match:
            candidates.append((int(match.group(1)), item.path))
    if not candidates:
        return None

    artifact_path = None
    if checkpoint_idx is not None:
        for idx, path in candidates:
            if idx == int(checkpoint_idx):
                artifact_path = path
                break
    if artifact_path is None:
        candidates.sort(key=lambda item: item[0])
        artifact_path = candidates[-1][1]

    print(
        f"[IntentPCA] Using discriminator checkpoint artifact: {artifact_path}",
        flush=True,
    )
    return _download_artifact_cached(
        client,
        run_id,
        artifact_path,
        cache_root=os.path.join(".opponent_cache", run_id),
        log_prefix="[IntentPCA]",
    )


def _download_artifact_cached(
    client,
    run_id: str,
    artifact_path: str,
    *,
    cache_root: str,
    log_prefix: str = "[IntentPCA]",
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
    client,
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


def _sample_geometric_index_with_rng(
    indices: list[int], beta: float, rng: random.Random
) -> int:
    if not indices:
        raise ValueError("indices must be non-empty")
    beta = float(beta)
    weights = [
        (1.0 - beta) * (beta ** (len(indices) - i))
        for i in range(1, len(indices) + 1)
    ]
    return int(rng.choices(indices, weights=weights, k=1)[0])


def _sample_training_matched_opponent_artifacts(
    candidates: list[tuple[int, str]],
    *,
    num_assignments: int,
    pool_size: int,
    beta: float,
    uniform_eps: float,
    per_env_sampling: bool,
    seed: int,
) -> list[str]:
    if not candidates:
        return []
    recent = candidates[-max(1, int(pool_size)) :]
    full_pool = list(candidates)
    rng = random.Random(int(seed))

    def _sample_one() -> str:
        if (
            rng.random() < float(uniform_eps)
            and len(full_pool) > len(recent)
        ):
            return str(rng.choice(full_pool)[1])
        recent_indices = list(range(len(recent)))
        chosen_idx = _sample_geometric_index_with_rng(
            recent_indices,
            float(beta),
            rng,
        )
        return str(recent[chosen_idx][1])

    if not bool(per_env_sampling):
        chosen = _sample_one()
        return [chosen for _ in range(max(1, int(num_assignments)))]
    return [_sample_one() for _ in range(max(1, int(num_assignments)))]


def _parse_logged_opponent_assignment_text(note_text: str) -> list[str]:
    env_matches: list[tuple[int, str]] = []
    for line in str(note_text).splitlines():
        match = re.search(r"Env\s+(\d+):\s+(\S+)", line)
        if match:
            env_matches.append((int(match.group(1)), str(match.group(2)).strip()))
    if env_matches:
        env_matches.sort(key=lambda item: item[0])
        return [name for _, name in env_matches]

    single_matches = re.findall(r"^\s+(\S+\.zip)\s*$", str(note_text), flags=re.MULTILINE)
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
    """Return the number of offense env slots used during mixed self-play training.

    Unified training uses a mixed vec env where the first half of workers train
    offense and the second half train defense. The discriminator callback only
    consumes offense episodes, so offline offense-only replay should mirror the
    offense half of the logged/sampled opponent assignment, not the full mixed
    assignment list.
    """

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


def _download_logged_opponent_assignments(
    client,
    run_id: str,
    *,
    checkpoint_idx: int | None,
) -> list[str]:
    if checkpoint_idx is None:
        return []
    artifact_path = f"opponents/opponent_alt_{int(checkpoint_idx)}.txt"
    local_cached_path = os.path.join(".opponent_cache", run_id, artifact_path)
    if os.path.isfile(local_cached_path):
        try:
            return _parse_logged_opponent_assignment_text(Path(local_cached_path).read_text())
        except Exception:
            return []
    print(
        "[IntentPCA] No cached logged opponent assignment note found; "
        "falling back to synthetic training-match sampling.",
        flush=True,
    )
    return []


def _resolve_opponent_policy_paths_for_collection(
    *,
    run_id: str | None,
    policy_path: str,
    env_args: argparse.Namespace,
    num_assignments: int,
    opponent_mode: str,
    seed: int,
) -> tuple[list[str], dict]:
    mode = str(opponent_mode).strip().lower()
    if mode == "self" or run_id is None:
        if mode == "training_match" and run_id is None:
            print(
                "[IntentPCA] No run_id available for training-matched opponent sampling; "
                "falling back to self-play opponent.",
                flush=True,
            )
        return [str(policy_path) for _ in range(max(1, int(num_assignments)))], {
            "opponent_mode": "self",
            "opponent_policy_paths": [str(policy_path)],
        }

    checkpoint_idx = _extract_checkpoint_index(policy_path)
    client = mlflow.tracking.MlflowClient()
    candidates = _list_unified_checkpoint_artifacts(
        client,
        run_id,
        checkpoint_idx=checkpoint_idx,
        include_current=False,
    )
    model_artifact_map = {os.path.basename(path): path for _, path in candidates}
    if checkpoint_idx is not None:
        current_candidates = _list_unified_checkpoint_artifacts(
            client,
            run_id,
            checkpoint_idx=checkpoint_idx,
            include_current=True,
        )
        for _, path in current_candidates:
            model_artifact_map.setdefault(os.path.basename(path), path)

    logged_basenames = _download_logged_opponent_assignments(
        client,
        run_id,
        checkpoint_idx=checkpoint_idx,
    )
    if logged_basenames:
        logged_basenames = _offense_subset_logged_assignments(
            logged_basenames,
            env_args=env_args,
        )
        fitted_basenames = _fit_logged_opponent_assignments(
            logged_basenames,
            num_assignments=max(1, int(num_assignments)),
        )
        local_cache: dict[str, str] = {}
        resolved_paths: list[str] = []
        missing_basenames: list[str] = []
        for basename in fitted_basenames:
            artifact_path = model_artifact_map.get(str(basename))
            if artifact_path is None:
                if str(basename) == os.path.basename(str(policy_path)):
                    resolved_paths.append(str(policy_path))
                    continue
                missing_basenames.append(str(basename))
                continue
            if artifact_path not in local_cache:
                local_cache[artifact_path] = _download_artifact_cached(
                    client,
                    run_id,
                    artifact_path,
                    cache_root=os.path.join(".opponent_cache", run_id),
                    log_prefix="[IntentPCA]",
                )
            resolved_paths.append(local_cache[artifact_path])
        if resolved_paths and not missing_basenames:
            return resolved_paths, {
                "opponent_mode": "training_match",
                "opponent_source": f"logged_assignment_alt_{checkpoint_idx}",
                "opponent_assignment_subset": "offense_only",
                "training_offense_assignment_count": _training_offense_assignment_count(
                    env_args
                ),
                "opponent_policy_basenames": fitted_basenames,
                "opponent_policy_paths": resolved_paths,
                "per_env_opponent_sampling": bool(
                    getattr(env_args, "per_env_opponent_sampling", False)
                ),
                "policy_checkpoint_index": checkpoint_idx,
            }
        if missing_basenames:
            print(
                "[IntentPCA] Logged opponent assignment contained unmatched checkpoint names; "
                "falling back to synthetic training-match sampling. "
                f"Missing: {sorted(set(missing_basenames))}",
                flush=True,
            )

    if not candidates:
        print(
            "[IntentPCA] No prior unified checkpoints found for training-matched opponent sampling; "
            "falling back to self-play opponent.",
            flush=True,
        )
        return [str(policy_path) for _ in range(max(1, int(num_assignments)))], {
            "opponent_mode": "self",
            "opponent_policy_paths": [str(policy_path)],
        }

    selected_artifacts = _sample_training_matched_opponent_artifacts(
        candidates,
        num_assignments=max(
            1,
            min(int(num_assignments), _training_offense_assignment_count(env_args))
            if bool(getattr(env_args, "per_env_opponent_sampling", False))
            else int(num_assignments),
        ),
        pool_size=int(getattr(env_args, "opponent_pool_size", 10)),
        beta=float(getattr(env_args, "opponent_pool_beta", 0.7)),
        uniform_eps=float(getattr(env_args, "opponent_pool_exploration", 0.15)),
        per_env_sampling=bool(getattr(env_args, "per_env_opponent_sampling", False)),
        seed=int(seed),
    )
    local_cache: dict[str, str] = {}
    resolved_paths: list[str] = []
    for artifact_path in selected_artifacts:
        if artifact_path not in local_cache:
            local_cache[artifact_path] = _download_artifact_cached(
                client,
                run_id,
                artifact_path,
                cache_root=os.path.join(".opponent_cache", run_id),
                log_prefix="[IntentPCA]",
            )
        resolved_paths.append(local_cache[artifact_path])
    resolved_paths = _fit_logged_opponent_assignments(
        resolved_paths,
        num_assignments=max(1, int(num_assignments)),
    )
    return resolved_paths, {
        "opponent_mode": "training_match",
        "opponent_source": "synthetic_sampling",
        "opponent_policy_artifacts": selected_artifacts,
        "opponent_policy_paths": resolved_paths,
        "opponent_assignment_subset": "offense_only"
        if bool(getattr(env_args, "per_env_opponent_sampling", False))
        else "all",
        "training_offense_assignment_count": _training_offense_assignment_count(
            env_args
        ),
        "opponent_pool_size": int(getattr(env_args, "opponent_pool_size", 10)),
        "opponent_pool_beta": float(getattr(env_args, "opponent_pool_beta", 0.7)),
        "opponent_pool_exploration": float(
            getattr(env_args, "opponent_pool_exploration", 0.15)
        ),
        "per_env_opponent_sampling": bool(
            getattr(env_args, "per_env_opponent_sampling", False)
        ),
        "policy_checkpoint_index": checkpoint_idx,
    }


def resolve_discriminator_path(
    run_id: str | None,
    policy_path: str,
) -> str | None:
    checkpoint_idx = _extract_checkpoint_index(policy_path)

    policy_dir = os.path.dirname(str(policy_path))
    if checkpoint_idx is not None:
        local_match = os.path.join(policy_dir, f"intent_disc_iter_{checkpoint_idx}.pt")
        if os.path.isfile(local_match):
            print(
                f"[IntentPCA] Using local discriminator checkpoint: {local_match}",
                flush=True,
            )
            return local_match
    pt_candidates = sorted(
        glob.glob(os.path.join(policy_dir, "intent_disc_*.pt")), key=os.path.getmtime
    )
    if checkpoint_idx is None and pt_candidates:
        local_path = pt_candidates[-1]
        print(
            f"[IntentPCA] Using latest local discriminator checkpoint: {local_path}",
            flush=True,
        )
        return local_path

    if run_id is None:
        return None
    return _download_matching_intent_disc(run_id, checkpoint_idx)


def _load_discriminator_checkpoint(path: str, device: str) -> tuple[IntentDiscriminator, dict]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or "state_dict" not in payload or "config" not in payload:
        raise RuntimeError(f"Unsupported discriminator checkpoint format: {path}")
    disc = load_intent_discriminator_from_checkpoint(payload, device=device)
    return disc, payload


def _custom_objects() -> dict:
    return {
        "PassBiasDualCriticPolicy": PassBiasDualCriticPolicy,
        "PassBiasMultiInputPolicy": PassBiasMultiInputPolicy,
        "SetAttentionDualCriticPolicy": SetAttentionDualCriticPolicy,
        "SetAttentionExtractor": SetAttentionExtractor,
    }


def _policy_uses_set_obs(model: PPO) -> bool:
    try:
        obs_space = getattr(model, "observation_space", None)
        spaces_dict = getattr(obs_space, "spaces", {})
        return "players" in spaces_dict and "globals" in spaces_dict
    except Exception:
        return False


def _apply_run_params(args, required: dict, optional: dict) -> None:
    for key, value in {**required, **optional}.items():
        setattr(args, key, value)
    if "pass_arc_degrees" in optional:
        args.pass_arc_start = optional["pass_arc_degrees"]
    if "pass_oob_turnover_prob" in optional:
        args.pass_oob_turnover_prob_start = optional["pass_oob_turnover_prob"]
    if "offensive_three_seconds_enabled" in optional:
        args.offensive_three_seconds = optional["offensive_three_seconds_enabled"]


def _build_env_args(run_id: str | None, model: PPO) -> argparse.Namespace:
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


def _make_offense_env_fn(
    env_args: argparse.Namespace, env_idx: int, opponent_policy_path: str
):
    def _thunk():
        base_env = setup_environment(env_args, Team.OFFENSE, env_idx=env_idx)
        return SelfPlayEnvWrapper(
            base_env,
            opponent_policy=str(opponent_policy_path),
            training_strategy=IllegalActionStrategy.SAMPLE_PROB,
            opponent_strategy=IllegalActionStrategy.SAMPLE_PROB,
            deterministic_opponent=False,
        )

    return _thunk


def _build_vec_env(
    env_args: argparse.Namespace,
    *,
    num_envs: int,
    backend: str,
    opponent_policy_path: str,
):
    env_fns = [
        _make_offense_env_fn(
            env_args,
            env_idx=i,
            opponent_policy_path=opponent_policy_path,
        )
        for i in range(int(num_envs))
    ]
    if backend == "dummy" or int(num_envs) <= 1:
        return DummyVecEnv(env_fns), "dummy"
    return SubprocVecEnv(env_fns, start_method="spawn"), "subproc"


def _unwrap_predicted_actions(predicted) -> np.ndarray:
    arr = np.asarray(predicted, dtype=int)
    if arr.ndim == 0:
        return np.array([[int(arr)]], dtype=int)
    if arr.ndim == 1:
        return arr.reshape(1, -1).astype(int, copy=False)
    return arr.reshape(arr.shape[0], -1).astype(int, copy=False)


def _extract_probs_for_env(probs_per_player, env_idx: int):
    if probs_per_player is None:
        return None
    extracted = []
    for probs in probs_per_player:
        arr = np.asarray(probs, dtype=np.float32)
        if arr.ndim >= 2:
            extracted.append(arr[int(env_idx)])
        else:
            extracted.append(arr)
    return extracted


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


def _single_obs_to_batched(obs: dict) -> dict:
    batched = {}
    for key, value in (obs or {}).items():
        if isinstance(value, np.ndarray):
            batched[key] = np.expand_dims(value, axis=0)
        else:
            batched[key] = value
    return batched


def _compute_policy_state_embedding(model: PPO, obs: dict) -> np.ndarray:
    policy_obj = getattr(model, "policy", None)
    if policy_obj is None:
        raise RuntimeError("Loaded PPO model has no policy object.")

    obs_tensor, _ = policy_obj.obs_to_tensor(obs)
    with (
        policy_obj.runtime_conditioning_context(obs_tensor)
        if hasattr(policy_obj, "runtime_conditioning_context")
        else nullcontext()
    ):
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


@contextmanager
def _temporarily_disable_set_intent_embedding(policy_obj):
    extractor = getattr(policy_obj, "features_extractor", None)
    if extractor is None:
        yield
        return

    attr_names = (
        "offense_intent_embedding",
        "defense_intent_embedding",
        "offense_intent_to_token",
        "defense_intent_to_token",
    )
    saved: dict[str, object] = {}
    try:
        for name in attr_names:
            if hasattr(extractor, name):
                saved[name] = getattr(extractor, name)
                setattr(extractor, name, None)
        yield
    finally:
        for name, value in saved.items():
            setattr(extractor, name, value)


def _compute_policy_state_embedding_intent_embedding_ablated(
    model: PPO, obs: dict
) -> np.ndarray:
    policy_obj = getattr(model, "policy", None)
    if policy_obj is None:
        raise RuntimeError("Loaded PPO model has no policy object.")
    with _temporarily_disable_set_intent_embedding(policy_obj):
        return _compute_policy_state_embedding(model, obs)


def _compute_policy_state_embedding_no_intent_signal(
    model: PPO, obs: dict
) -> np.ndarray:
    policy_obj = getattr(model, "policy", None)
    if policy_obj is None:
        raise RuntimeError("Loaded PPO model has no policy object.")
    obs_clone = clone_observation_dict(obs)
    num_intents = int(max(1, infer_num_intents(model, default=8)))
    patch_intent_in_observation(
        obs_clone,
        0,
        num_intents,
        active=0.0,
        visible=0.0,
        age_norm=0.0,
    )
    with _temporarily_disable_set_intent_embedding(policy_obj):
        return _compute_policy_state_embedding(model, obs_clone)


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


def _build_transition(
    obs_payload,
    resolved_action_batch: np.ndarray,
    info: dict,
    max_obs_dim: int,
    max_action_dim: int,
    rollout_step_idx: int,
    env_idx: int,
) -> IntentTransition:
    obs_feat = flatten_observation_for_env(obs_payload, env_idx)
    act_feat = extract_action_features_for_env(resolved_action_batch, env_idx)
    feat = np.zeros(max_obs_dim + max_action_dim, dtype=np.float32)
    obs_take = min(max_obs_dim, obs_feat.shape[0])
    if obs_take > 0:
        feat[:obs_take] = obs_feat[:obs_take]
    act_take = min(max_action_dim, act_feat.shape[0])
    if act_take > 0:
        feat[max_obs_dim : max_obs_dim + act_take] = act_feat[:act_take]

    role_flag = _extract_obs_scalar(obs_payload, "role_flag", env_idx, default=0.0)
    intent_active = _extract_obs_scalar(
        obs_payload,
        "intent_active",
        env_idx,
        default=float(info.get("intent_active", 0.0)),
    )
    intent_index = _extract_obs_scalar(
        obs_payload,
        "intent_index",
        env_idx,
        default=float(info.get("intent_index", 0.0)),
    )

    return IntentTransition(
        feature=feat,
        buffer_step_idx=int(rollout_step_idx),
        env_idx=int(env_idx),
        role_flag=float(role_flag),
        intent_active=bool(intent_active > 0.5),
        intent_index=int(max(0, int(intent_index))),
    )


def _build_transition_single_obs(
    obs: dict,
    training_action: np.ndarray,
    *,
    max_obs_dim: int,
    max_action_dim: int,
    rollout_step_idx: int,
    feature_override: np.ndarray | None = None,
) -> IntentTransition:
    if feature_override is not None:
        batched_obs = _single_obs_to_batched(obs)
        role_flag = _extract_obs_scalar(batched_obs, "role_flag", 0, default=0.0)
        intent_active = _extract_obs_scalar(batched_obs, "intent_active", 0, default=0.0)
        intent_index = _extract_obs_scalar(batched_obs, "intent_index", 0, default=0.0)
        return IntentTransition(
            feature=np.asarray(feature_override, dtype=np.float32).reshape(-1),
            buffer_step_idx=int(rollout_step_idx),
            env_idx=0,
            role_flag=float(role_flag),
            intent_active=bool(intent_active > 0.5),
            intent_index=int(max(0, int(intent_index))),
        )
    batched_obs = _single_obs_to_batched(obs)
    batched_actions = np.asarray(training_action, dtype=np.int64).reshape(1, -1)
    return _build_transition(
        batched_obs,
        batched_actions,
        {},
        max_obs_dim=max_obs_dim,
        max_action_dim=max_action_dim,
        rollout_step_idx=rollout_step_idx,
        env_idx=0,
    )


def _update_pass_counters(counters: dict, action_results: dict) -> None:
    passes = (action_results or {}).get("passes", {}) or {}
    counters["pass_attempts"] += len(passes)
    counters["pass_completions"] += sum(1 for pres in passes.values() if pres.get("success"))
    counters["pass_intercepts"] += sum(
        1
        for pres in passes.values()
        if (not pres.get("success")) and pres.get("reason") == "intercepted"
    )
    counters["pass_oob"] += sum(
        1
        for pres in passes.values()
        if (not pres.get("success")) and pres.get("reason") == "out_of_bounds"
    )


def _make_episode_metadata(
    episode_idx: int,
    start_intent_index: int,
    start_intent_active: bool,
    episode,
    episode_info: dict,
    counters: dict,
) -> dict:
    outcome, shot_type, points = infer_outcome_from_episode_info(episode_info)
    return {
        "episode_idx": int(episode_idx),
        "intent_index": int(start_intent_index),
        "intent_active_start": bool(start_intent_active),
        "episode_length": int(float(episode_info.get("l", episode.length))),
        "active_prefix_length": int(episode.active_prefix_length),
        "outcome": outcome,
        "shot_type": shot_type,
        "points": float(points),
        "pass_attempts": int(counters.get("pass_attempts", 0)),
        "pass_completions": int(counters.get("pass_completions", 0)),
        "pass_intercepts": int(counters.get("pass_intercepts", 0)),
        "pass_oob": int(counters.get("pass_oob", 0)),
        "assist_potential": float(
            episode_info.get("potential_assists", 0.0)
        ),
        "assist_full": float(episode_info.get("assists", 0.0)),
        "team_reward_offense": float(episode_info.get("r", 0.0)),
    }


def _episode_label_intent_index(
    transitions: list[IntentTransition] | None,
    fallback_intent_index: int,
) -> int:
    if transitions:
        try:
            return int(max(0, int(transitions[0].intent_index)))
        except Exception:
            pass
    return int(max(0, int(fallback_intent_index)))


def _init_intent_pca_worker(
    env_args: argparse.Namespace,
    unified_policy_path: str,
    device: str,
    feature_mode: str,
    progress_queue=None,
):
    global _PCA_WORKER_STATE

    import torch as _torch
    from app.backend.observations import (
        _ensure_set_obs as _obs_ensure_set_obs,
        validate_policy_observation_schema as _obs_validate_schema,
    )

    try:
        _torch.set_num_threads(1)
        _torch.set_num_interop_threads(1)
    except Exception:
        pass

    custom_objects = _custom_objects()
    unified_policy = load_ppo_for_inference(
        unified_policy_path,
        device=device,
        custom_objects=custom_objects,
    )

    try:
        env = setup_environment(env_args, Team.OFFENSE)
        obs0, _ = env.reset(seed=0)
        _obs_validate_schema(
            unified_policy, env, obs0, policy_label="intent_pca_unified_policy"
        )
        env.close()
    except Exception as e:
        raise RuntimeError(f"Intent PCA worker schema validation failed: {e}") from e

    _PCA_WORKER_STATE = {
        "env_args": env_args,
        "unified_policy": unified_policy,
        "ensure_set_obs": _obs_ensure_set_obs,
        "feature_mode": str(feature_mode),
        "progress_queue": progress_queue,
    }


def _run_intent_pca_batch_worker(args: tuple) -> dict:
    (
        accepted_target,
        seed,
        opponent_policy_path,
        intent_source_mode,
        require_active_intent,
        player_deterministic,
        opponent_deterministic,
        illegal_strategy_name,
        max_obs_dim,
        max_action_dim,
        collect_disc_episodes,
        disc_max_obs_dim,
        disc_max_action_dim,
    ) = args

    state = _PCA_WORKER_STATE
    env_args = state["env_args"]
    unified_policy = state["unified_policy"]
    ensure_set_obs = state["ensure_set_obs"]
    feature_mode = str(state.get("feature_mode", "active_prefix_flat"))
    progress_queue = state.get("progress_queue")
    rng = np.random.default_rng(int(seed))
    deterministic_opponent = bool(opponent_deterministic)

    if illegal_strategy_name == "noop":
        illegal_strategy = IllegalActionStrategy.NOOP
    elif illegal_strategy_name == "best":
        illegal_strategy = IllegalActionStrategy.BEST_PROB
    else:
        illegal_strategy = IllegalActionStrategy.SAMPLE_PROB

    episodes: list[CompletedIntentEpisode] = []
    disc_episodes: list[CompletedIntentEpisode] = []
    metadata: list[dict] = []
    accepted_idx = 0
    env = setup_environment(env_args, Team.OFFENSE)
    wrapped_env = SelfPlayEnvWrapper(
        env,
        opponent_policy=str(opponent_policy_path) if opponent_policy_path else None,
        training_strategy=illegal_strategy,
        opponent_strategy=illegal_strategy,
        deterministic_opponent=deterministic_opponent,
    )
    try:
        while accepted_idx < int(accepted_target):
            reset_seed = int(rng.integers(0, 2**31 - 1))
            obs, _ = wrapped_env.reset(seed=reset_seed)
            obs = ensure_set_obs(unified_policy, wrapped_env, obs)
            if isinstance(obs, dict):
                obs["role_flag"] = np.array([1.0], dtype=np.float32)

            selector_meta = _maybe_apply_selector_intent_start(
                obs=obs,
                wrapped_env=wrapped_env,
                unified_policy=unified_policy,
                env_args=env_args,
                rng=rng,
                intent_source_mode=str(intent_source_mode),
            )
            start_intent_active = bool(getattr(wrapped_env.unwrapped, "intent_active", False))
            start_intent_index = (
                int(getattr(wrapped_env.unwrapped, "intent_index", 0))
                if start_intent_active
                else 0
            )
            if bool(require_active_intent) and not start_intent_active:
                continue

            transitions: list[IntentTransition] = []
            disc_transitions: list[IntentTransition] | None = (
                [] if bool(collect_disc_episodes) else None
            )
            counters = {
                "pass_attempts": 0,
                "pass_completions": 0,
                "pass_intercepts": 0,
                "pass_oob": 0,
            }
            done = False
            rollout_step_idx = 0
            final_info = {}

            while not done:
                raw_training_action, _ = unified_policy.predict(
                    obs, deterministic=bool(player_deterministic)
                )
                training_action = _normalize_training_action_for_wrapper(
                    raw_training_action,
                    wrapped_env,
                )
                next_obs, _, terminated, truncated, info = wrapped_env.step(training_action)
                next_obs = ensure_set_obs(unified_policy, wrapped_env, next_obs)
                if isinstance(next_obs, dict):
                    next_obs["role_flag"] = np.array([1.0], dtype=np.float32)

                if disc_transitions is not None:
                    disc_transitions.append(
                        _build_transition_single_obs(
                            next_obs,
                            training_action,
                            max_obs_dim=int(disc_max_obs_dim),
                            max_action_dim=int(disc_max_action_dim),
                            rollout_step_idx=rollout_step_idx,
                        )
                    )

                if feature_mode in {
                    "set_attention_pool",
                    "set_attention_pool_no_intent_embedding",
                    "set_attention_pool_no_intent_signal",
                }:
                    feature_override = (
                        _compute_policy_state_embedding_no_intent_signal(
                            unified_policy, next_obs
                        )
                        if feature_mode == "set_attention_pool_no_intent_signal"
                        else (
                        _compute_policy_state_embedding_intent_embedding_ablated(
                            unified_policy, next_obs
                        )
                        if feature_mode == "set_attention_pool_no_intent_embedding"
                        else _compute_policy_state_embedding(unified_policy, next_obs)
                        )
                    )
                    transitions.append(
                        _build_transition_single_obs(
                            next_obs,
                            training_action,
                            max_obs_dim=int(max_obs_dim),
                            max_action_dim=int(max_action_dim),
                            rollout_step_idx=rollout_step_idx,
                            feature_override=feature_override,
                        )
                    )
                else:
                    transitions.append(
                        _build_transition_single_obs(
                            next_obs,
                            training_action,
                            max_obs_dim=int(max_obs_dim),
                            max_action_dim=int(max_action_dim),
                            rollout_step_idx=rollout_step_idx,
                        )
                    )

                obs = next_obs
                final_info = info or {}
                _update_pass_counters(counters, final_info.get("action_results", {}) or {})
                rollout_step_idx += 1
                done = bool(terminated or truncated)

            episode_label_intent_index = _episode_label_intent_index(
                transitions,
                start_intent_index,
            )
            episode = CompletedIntentEpisode(
                intent_index=int(episode_label_intent_index),
                transitions=transitions,
            )
            if bool(require_active_intent) and episode.active_prefix_length <= 0:
                continue
            if disc_transitions is not None:
                disc_episode_label_intent_index = _episode_label_intent_index(
                    disc_transitions,
                    start_intent_index,
                )
                disc_episode = CompletedIntentEpisode(
                    intent_index=int(disc_episode_label_intent_index),
                    transitions=disc_transitions,
                )
                if bool(require_active_intent) and disc_episode.active_prefix_length <= 0:
                    continue
            else:
                disc_episode_label_intent_index = None

            episode_info = dict(final_info.get("episode", {}) or {})
            metadata.append(
                {
                    **_make_episode_metadata(
                    accepted_idx,
                    start_intent_index,
                    start_intent_active,
                    episode,
                    episode_info,
                    counters,
                    ),
                    "selector_requested_mode": str(selector_meta.get("requested_intent_source_mode", intent_source_mode)),
                    "selector_resolved_mode": str(selector_meta.get("resolved_intent_source_mode", "env")),
                    "selector_alpha": float(selector_meta.get("selector_alpha", 0.0)),
                    "selector_applied": bool(selector_meta.get("selector_applied", False)),
                    "selector_selected_intent_index": selector_meta.get("selected_intent_index"),
                    "collector_episode_label_intent_index": int(episode_label_intent_index),
                    "collector_disc_label_intent_index": (
                        None
                        if disc_episode_label_intent_index is None
                        else int(disc_episode_label_intent_index)
                    ),
                    "collector_start_label_matches": bool(
                        int(episode_label_intent_index) == int(start_intent_index)
                    ),
                    "collector_disc_start_label_matches": (
                        None
                        if disc_episode_label_intent_index is None
                        else bool(
                            int(disc_episode_label_intent_index) == int(start_intent_index)
                        )
                    ),
                }
            )
            episodes.append(episode)
            if disc_transitions is not None:
                disc_episodes.append(disc_episode)
            accepted_idx += 1
            if progress_queue is not None:
                try:
                    progress_queue.put_nowait(1)
                except Exception:
                    pass
    finally:
        try:
            wrapped_env.close()
        except Exception:
            pass

    return {
        "episodes": episodes,
        "disc_episodes": disc_episodes if disc_episodes else None,
        "metadata": metadata,
    }


def _collect_episodes_parallel_workers(
    *,
    env_args: argparse.Namespace,
    policy_path: str,
    opponent_policy_paths: list[str],
    num_episodes: int,
    num_workers: int,
    device: str,
    feature_mode: str,
    player_deterministic: bool,
    opponent_deterministic: bool,
    intent_source_mode: str,
    illegal_strategy_name: str,
    require_active_intent: bool,
    max_obs_dim: int,
    max_action_dim: int,
    collect_disc_episodes: bool,
    disc_max_obs_dim: int,
    disc_max_action_dim: int,
    progress: tqdm,
) -> tuple[list[CompletedIntentEpisode], list[dict], list[CompletedIntentEpisode] | None]:
    num_workers = max(1, min(int(num_workers), int(num_episodes)))
    per_env_sampling = bool(getattr(env_args, "per_env_opponent_sampling", False))
    target_batches = max(1, num_workers if per_env_sampling else (num_workers * 4))
    batch_target = max(1, (int(num_episodes) + target_batches - 1) // target_batches)
    batches = []
    remaining = int(num_episodes)
    seed_rng = np.random.default_rng(0)
    while remaining > 0:
        accepted_target = min(batch_target, remaining)
        batch_idx = len(batches)
        opponent_path = (
            opponent_policy_paths[min(batch_idx, len(opponent_policy_paths) - 1)]
            if opponent_policy_paths
            else str(policy_path)
        )
        batches.append(
            (
                accepted_target,
                int(seed_rng.integers(0, 2**31 - 1)),
                str(opponent_path),
                str(intent_source_mode),
                bool(require_active_intent),
                bool(player_deterministic),
                bool(opponent_deterministic),
                str(illegal_strategy_name),
                int(max_obs_dim),
                int(max_action_dim),
                bool(collect_disc_episodes),
                int(disc_max_obs_dim),
                int(disc_max_action_dim),
            )
        )
        remaining -= accepted_target

    ctx = mp.get_context("spawn")
    progress_queue = ctx.Queue()
    episodes: list[CompletedIntentEpisode] = []
    disc_episodes: list[CompletedIntentEpisode] = []
    metadata: list[dict] = []

    with ProcessPoolExecutor(
        max_workers=num_workers,
        mp_context=ctx,
        initializer=_init_intent_pca_worker,
        initargs=(
            env_args,
            str(policy_path),
            str(device),
            str(feature_mode),
            progress_queue,
        ),
    ) as executor:
        futures = {executor.submit(_run_intent_pca_batch_worker, batch): batch for batch in batches}
        pending = set(futures.keys())

        while pending:
            done, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
            while True:
                try:
                    inc = int(progress_queue.get_nowait())
                    progress.update(min(inc, max(0, int(progress.total) - int(progress.n))))
                except queue.Empty:
                    break
                except Exception:
                    break
            for future in done:
                payload = future.result()
                episodes.extend(payload.get("episodes", []) or [])
                disc_episodes.extend(payload.get("disc_episodes", []) or [])
                metadata.extend(payload.get("metadata", []) or [])

        while True:
            try:
                inc = int(progress_queue.get_nowait())
                progress.update(min(inc, max(0, int(progress.total) - int(progress.n))))
            except queue.Empty:
                break
            except Exception:
                break

    if len(episodes) > int(num_episodes):
        episodes = episodes[: int(num_episodes)]
        metadata = metadata[: int(num_episodes)]
    if disc_episodes:
        disc_episodes = disc_episodes[: int(num_episodes)]
        return episodes, metadata, disc_episodes
    return episodes, metadata, None


def _build_feature_matrix(
    feature_mode: str,
    episodes,
    metadata: list[dict],
    max_obs_dim: int,
    max_action_dim: int,
    disc_bundle: tuple[IntentDiscriminator, dict] | None = None,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if feature_mode in {
        "set_attention_pool",
        "set_attention_pool_no_intent_embedding",
        "set_attention_pool_no_intent_signal",
    }:
        rows: list[np.ndarray] = []
        labels: list[int] = []
        for ep in episodes:
            feats = [
                np.asarray(tr.feature, dtype=np.float32).reshape(-1)
                for tr in ep.active_prefix_transitions
            ]
            if not feats:
                continue
            rows.append(np.mean(np.vstack(feats), axis=0).astype(np.float32, copy=False))
            labels.append(int(ep.intent_index))
        if not rows:
            return np.zeros((0, 1), dtype=np.float32), np.zeros((0,), dtype=np.int64), []
        x = np.vstack(rows).astype(np.float32, copy=False)
        y = np.asarray(labels, dtype=np.int64)
        feature_names = [f"attn_pool_{i:04d}" for i in range(int(x.shape[1]))]
        return x, y, feature_names

    if feature_mode == "summary":
        x = np.vstack([build_summary_feature(meta) for meta in metadata]).astype(np.float32, copy=False)
        y = np.asarray([int(meta["intent_index"]) for meta in metadata], dtype=np.int64)
        return x, y, list(SUMMARY_FEATURE_NAMES)

    if feature_mode == "mean_pool":
        x, y = compute_episode_embeddings(
            episodes,
            max_obs_dim=max_obs_dim,
            max_action_dim=max_action_dim,
        )
        feature_names = [f"mean_feat_{i:04d}" for i in range(int(x.shape[1]))]
        return x, y, feature_names

    if feature_mode == "active_prefix_flat":
        padded, _, y = build_padded_episode_batch(
            episodes,
            max_obs_dim=max_obs_dim,
            max_action_dim=max_action_dim,
        )
        x = padded.reshape(padded.shape[0], -1).astype(np.float32, copy=False)
        feature_names = [f"prefix_feat_{i:05d}" for i in range(int(x.shape[1]))]
        return x, y, feature_names

    if feature_mode == "disc_embedding":
        if disc_bundle is None:
            raise RuntimeError(
                "feature_mode='disc_embedding' requires a saved intent discriminator checkpoint."
            )
        disc, payload = disc_bundle
        config = dict(payload.get("config", {}) or {})
        disc_max_obs_dim = int(config.get("max_obs_dim", max_obs_dim))
        disc_max_action_dim = int(config.get("max_action_dim", max_action_dim))
        encoder_type = str(config.get("encoder_type", "mlp_mean")).strip().lower()
        if encoder_type == "gru":
            x_np, lengths_np, y = build_padded_episode_batch(
                episodes,
                max_obs_dim=disc_max_obs_dim,
                max_action_dim=disc_max_action_dim,
            )
            x_tensor = torch.as_tensor(x_np, dtype=torch.float32, device=device)
            lengths_tensor = torch.as_tensor(lengths_np, dtype=torch.long, device=device)
            with torch.no_grad():
                emb = disc.encode(x_tensor, lengths_tensor)
        else:
            x_np, y = compute_episode_embeddings(
                episodes,
                max_obs_dim=disc_max_obs_dim,
                max_action_dim=disc_max_action_dim,
            )
            x_tensor = torch.as_tensor(x_np, dtype=torch.float32, device=device)
            with torch.no_grad():
                emb = disc.encode(x_tensor, None)
        x = emb.detach().cpu().numpy().astype(np.float32, copy=False)
        feature_names = [f"disc_emb_{i:04d}" for i in range(int(x.shape[1]))]
        return x, y, feature_names

    raise ValueError(f"Unsupported feature_mode={feature_mode!r}")


def _plot_embedding_scatter(
    coords: np.ndarray,
    labels: np.ndarray,
    output_path: str,
    title: str,
    *,
    xlabel: str,
    ylabel: str,
) -> None:
    unique_labels = sorted({int(x) for x in labels.tolist()})
    cmap = plt.get_cmap("tab10" if len(unique_labels) <= 10 else "tab20")
    fig, ax = plt.subplots(figsize=(11, 9))
    for idx, z in enumerate(unique_labels):
        mask = labels == z
        color = cmap(idx % cmap.N)
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=12,
            alpha=0.55,
            label=f"z={z}",
            color=color,
            edgecolors="none",
        )
        centroid = np.mean(coords[mask], axis=0)
        ax.scatter(
            [centroid[0]],
            [centroid[1]],
            s=80,
            color=color,
            edgecolors="black",
            linewidths=0.8,
        )
        ax.text(centroid[0], centroid[1], f" {z}", fontsize=10, va="center")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="best", frameon=True, fontsize=9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_pca(coords: np.ndarray, labels: np.ndarray, output_path: str, title: str, explained: np.ndarray) -> None:
    _plot_embedding_scatter(
        coords,
        labels,
        output_path,
        title,
        xlabel=f"PC1 ({100.0 * float(explained[0]):.2f}% var)",
        ylabel=f"PC2 ({100.0 * float(explained[1]):.2f}% var)",
    )


def _write_embedding_points_csv(
    output_path: Path,
    metadata: list[dict],
    coords: np.ndarray,
    *,
    x_key: str,
    y_key: str,
) -> None:
    with output_path.open("w", newline="") as f:
        fieldnames = [
            "episode_idx",
            "intent_index",
            "intent_active_start",
            "outcome",
            "shot_type",
            "episode_length",
            "active_prefix_length",
            "pass_attempts",
            "pass_completions",
            "pass_intercepts",
            "pass_oob",
            "assist_potential",
            "assist_full",
            "points",
            "team_reward_offense",
            x_key,
            y_key,
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for meta, coord in zip(metadata, coords):
            row = {key: meta.get(key) for key in fieldnames if key not in {x_key, y_key}}
            row[x_key] = float(coord[0])
            row[y_key] = float(coord[1])
            writer.writerow(row)


def _fit_tsne_coords(
    x_scaled: np.ndarray,
    *,
    seed: int,
    perplexity: float,
) -> tuple[np.ndarray, float]:
    n_samples = int(x_scaled.shape[0])
    if n_samples < 3:
        raise RuntimeError("Need at least three episodes to run t-SNE.")
    effective_perplexity = float(
        min(float(perplexity), max(1.0, float(n_samples - 1) / 3.0))
    )
    tsne = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        learning_rate="auto",
        init="pca",
        random_state=int(seed),
    )
    return tsne.fit_transform(x_scaled), effective_perplexity


def _fit_umap_coords(
    x_scaled: np.ndarray,
    *,
    seed: int,
    n_neighbors: int,
    min_dist: float,
) -> tuple[np.ndarray, int]:
    try:
        import umap  # type: ignore
    except Exception as exc:
        raise ModuleNotFoundError(
            "UMAP requires the optional 'umap-learn' package."
        ) from exc
    n_samples = int(x_scaled.shape[0])
    if n_samples < 3:
        raise RuntimeError("Need at least three episodes to run UMAP.")
    effective_neighbors = int(min(max(2, int(n_neighbors)), max(2, n_samples - 1)))
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=effective_neighbors,
        min_dist=float(min_dist),
        random_state=int(seed),
    )
    return reducer.fit_transform(x_scaled), effective_neighbors


def _compute_confusion_matrix_tables(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    num_intents: int,
) -> tuple[list[int], np.ndarray, np.ndarray]:
    labels = list(range(int(max(1, num_intents))))
    counts = confusion_matrix(
        np.asarray(y_true, dtype=np.int64),
        np.asarray(y_pred, dtype=np.int64),
        labels=labels,
    ).astype(np.int64, copy=False)
    row_totals = counts.sum(axis=1, keepdims=True).astype(np.float64, copy=False)
    row_norm = np.divide(
        counts.astype(np.float64, copy=False),
        np.maximum(row_totals, 1.0),
    )
    return labels, counts, row_norm


def _resolve_discriminator_feature_dims(
    disc_bundle: tuple[IntentDiscriminator, dict] | None,
    *,
    default_max_obs_dim: int,
    default_max_action_dim: int,
) -> tuple[int, int]:
    if disc_bundle is None:
        return int(default_max_obs_dim), int(default_max_action_dim)
    config = dict((disc_bundle[1] or {}).get("config", {}) or {})
    return (
        int(config.get("max_obs_dim", default_max_obs_dim)),
        int(config.get("max_action_dim", default_max_action_dim)),
    )


def _select_discriminator_eval_episodes(
    feature_episodes: list[CompletedIntentEpisode],
    raw_step_episodes: list[CompletedIntentEpisode] | None,
) -> list[CompletedIntentEpisode]:
    if raw_step_episodes:
        return raw_step_episodes
    return feature_episodes


def _compute_discriminator_predictions(
    episodes,
    *,
    disc_bundle: tuple[IntentDiscriminator, dict],
    max_obs_dim: int,
    max_action_dim: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    disc, payload = disc_bundle
    disc_max_obs_dim, disc_max_action_dim = _resolve_discriminator_feature_dims(
        (disc, payload),
        default_max_obs_dim=max_obs_dim,
        default_max_action_dim=max_action_dim,
    )
    config = dict((payload or {}).get("config", {}) or {})
    encoder_type = str(config.get("encoder_type", "mlp_mean")).strip().lower()

    if encoder_type == "gru":
        x_np, lengths_np, y_true = build_padded_episode_batch(
            episodes,
            max_obs_dim=disc_max_obs_dim,
            max_action_dim=disc_max_action_dim,
        )
        x_tensor = torch.as_tensor(x_np, dtype=torch.float32, device=device)
        lengths_tensor = torch.as_tensor(lengths_np, dtype=torch.long, device=device)
        with torch.no_grad():
            logits = disc(x_tensor, lengths_tensor)
    else:
        x_np, y_true = compute_episode_embeddings(
            episodes,
            max_obs_dim=disc_max_obs_dim,
            max_action_dim=disc_max_action_dim,
        )
        lengths_np = None
        x_tensor = torch.as_tensor(x_np, dtype=torch.float32, device=device)
        with torch.no_grad():
            logits = disc(x_tensor, None)

    prob_tensor = torch.softmax(logits, dim=-1)
    y_pred = torch.argmax(prob_tensor, dim=-1).detach().cpu().numpy().astype(np.int64, copy=False)
    probs = prob_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
    return np.asarray(y_true, dtype=np.int64), y_pred, probs, lengths_np


def _write_confusion_matrix_csv(
    output_path: Path,
    labels: list[int],
    matrix: np.ndarray,
    *,
    float_format: str | None = None,
) -> None:
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true_intent"] + [f"pred_{int(z)}" for z in labels])
        for row_idx, z in enumerate(labels):
            row = matrix[row_idx]
            if float_format is None:
                values = [int(v) for v in row.tolist()]
            else:
                values = [format(float(v), float_format) for v in row.tolist()]
            writer.writerow([int(z)] + values)


def _plot_confusion_matrices(
    labels: list[int],
    counts: np.ndarray,
    row_norm: np.ndarray,
    output_path: str,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    matrices = [
        (counts.astype(np.float64, copy=False), "Counts", "viridis"),
        (row_norm, "Row-normalized", "magma"),
    ]
    for ax, (matrix, subtitle, cmap_name) in zip(axes, matrices):
        im = ax.imshow(matrix, interpolation="nearest", cmap=cmap_name, aspect="auto")
        ax.set_title(subtitle)
        ax.set_xlabel("Predicted intent")
        ax.set_ylabel("True intent")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        threshold = float(np.max(matrix) * 0.55) if matrix.size else 0.0
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                text = f"{value:.2f}" if subtitle == "Row-normalized" else f"{int(round(value))}"
                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if value > threshold else "black",
                )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run PCA over latent-intent episode features and color points by intent z."
    )
    parser.add_argument("run_id_or_path", help="MLflow run ID or path to a unified policy .zip")
    parser.add_argument(
        "--mlflow-run-id",
        default=None,
        help="Optional MLflow run ID to source env params when loading a local policy path.",
    )
    parser.add_argument(
        "--alternation-index",
        type=int,
        default=None,
        help="Optional unified checkpoint alternation/iteration index to analyze instead of the latest.",
    )
    parser.add_argument("--episodes", type=int, default=2000, help="Number of active-intent episodes to collect.")
    parser.add_argument(
        "--feature-mode",
        choices=[
            "active_prefix_flat",
            "mean_pool",
            "summary",
            "disc_embedding",
            "set_attention_pool",
            "set_attention_pool_no_intent_embedding",
            "set_attention_pool_no_intent_signal",
        ],
        default="active_prefix_flat",
        help="Episode representation to feed into PCA.",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed.")
    parser.add_argument("--device", default="cpu", help="Torch device for policy inference.")
    parser.add_argument(
        "--num-envs",
        type=int,
        default=8,
        help="Number of offense environments to use for parallel episode collection.",
    )
    parser.add_argument(
        "--vec-env",
        choices=["auto", "dummy", "subproc"],
        default="auto",
        help="Vector environment backend for episode collection.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Deprecated shorthand: use deterministic action selection for both player and opponent.",
    )
    parser.add_argument(
        "--player-deterministic",
        action="store_true",
        help="Use deterministic action selection for the analyzed offense policy.",
    )
    parser.add_argument(
        "--opponent-deterministic",
        action="store_true",
        help="Use deterministic action selection for the opponent policy.",
    )
    parser.add_argument(
        "--opponent-mode",
        choices=["training_match", "self"],
        default="training_match",
        help="Opponent policy source for episode collection. training_match samples prior checkpoints like training; self reuses the analyzed checkpoint as opponent.",
    )
    parser.add_argument(
        "--intent-source",
        choices=["training_match", "env", "selector"],
        default="training_match",
        help="Episode-start offense intent source. training_match uses selector-driven starts when the run/policy had mu enabled; env uses raw env-sampled z; selector forces mu selection on every eligible start.",
    )
    parser.add_argument(
        "--illegal-strategy",
        choices=["sample", "best", "noop"],
        default="sample",
        help="Fallback strategy when predicted actions are illegal.",
    )
    parser.add_argument(
        "--include-null-start-episodes",
        dest="require_active_intent",
        action="store_false",
        help="Do not filter out episodes that start with no active offense intent.",
    )
    parser.add_argument(
        "--override-intent-null-prob",
        type=float,
        default=None,
        help="Optional override for intent_null_prob during analysis (for cleaner sampling).",
    )
    parser.add_argument(
        "--max-obs-dim",
        type=int,
        default=256,
        help="Max flattened observation dims used in step features.",
    )
    parser.add_argument(
        "--max-action-dim",
        type=int,
        default=16,
        help="Max flattened action dims used in step features.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Defaults to analytics/intent_pca/<label>_<timestamp>/",
    )
    parser.add_argument(
        "--embedding-methods",
        nargs="+",
        choices=["pca", "tsne", "umap"],
        default=["pca"],
        help="Low-dimensional projections to render. PCA remains the primary summary projection.",
    )
    parser.add_argument(
        "--tsne-perplexity",
        type=float,
        default=30.0,
        help="Requested t-SNE perplexity. The effective value is clipped based on sample count.",
    )
    parser.add_argument(
        "--umap-n-neighbors",
        type=int,
        default=30,
        help="Requested UMAP n_neighbors. The effective value is clipped based on sample count.",
    )
    parser.add_argument(
        "--umap-min-dist",
        type=float,
        default=0.1,
        help="UMAP min_dist parameter.",
    )
    parser.add_argument(
        "--log-to-mlflow",
        action="store_true",
        help="Log plot/csv/json artifacts back to the run when a run ID is available.",
    )
    args = parser.parse_args()

    setup_mlflow()

    policy_path, inferred_run_id = resolve_policy_path(
        args.run_id_or_path,
        checkpoint_idx=args.alternation_index,
    )
    run_id = args.mlflow_run_id or inferred_run_id
    disc_bundle: tuple[IntentDiscriminator, dict] | None = None
    disc_path: str | None = None

    model = load_ppo_for_inference(
        policy_path,
        device=args.device,
        custom_objects=_custom_objects(),
    )
    disc_path = resolve_discriminator_path(run_id, policy_path)
    if args.feature_mode == "disc_embedding":
        if disc_path is None:
            raise RuntimeError(
                "No saved intent discriminator checkpoint found for this run/checkpoint. "
                "Train with the updated code that logs intent_disc_iter_<N>.pt artifacts."
            )
        disc_bundle = _load_discriminator_checkpoint(disc_path, args.device)
        disc_cfg = dict((disc_bundle[1] or {}).get("config", {}) or {})
        print(
            "[IntentPCA] Loaded discriminator embedding source: "
            f"path={disc_path} encoder={disc_cfg.get('encoder_type', 'unknown')} "
            f"hidden_dim={disc_cfg.get('hidden_dim', 'unknown')}",
            flush=True,
        )
    elif disc_path is not None:
        disc_bundle = _load_discriminator_checkpoint(disc_path, args.device)
        disc_cfg = dict((disc_bundle[1] or {}).get("config", {}) or {})
        print(
            "[IntentPCA] Loaded discriminator analysis source: "
            f"path={disc_path} encoder={disc_cfg.get('encoder_type', 'unknown')} "
            f"hidden_dim={disc_cfg.get('hidden_dim', 'unknown')}",
            flush=True,
        )
    env_args = _build_env_args(run_id, model)
    if args.override_intent_null_prob is not None:
        env_args.intent_null_prob = float(max(0.0, min(1.0, args.override_intent_null_prob)))
    requested_num_envs = max(1, int(args.num_envs))
    player_deterministic = bool(args.deterministic or args.player_deterministic)
    if bool(args.deterministic) or bool(args.opponent_deterministic):
        opponent_deterministic = True
    else:
        opponent_deterministic = bool(
            getattr(env_args, "deterministic_opponent", False)
            if str(args.opponent_mode).strip().lower() == "training_match"
            else False
        )
    opponent_policy_paths, opponent_meta = _resolve_opponent_policy_paths_for_collection(
        run_id=run_id,
        policy_path=str(policy_path),
        env_args=env_args,
        num_assignments=max(
            1,
            requested_num_envs
            if bool(getattr(env_args, "per_env_opponent_sampling", False))
            else (requested_num_envs * 4),
        ),
        opponent_mode=str(args.opponent_mode),
        seed=int(args.seed),
    )
    worker_mode = "eval_workers"
    resolved_intent_source = _resolve_intent_source_mode(
        str(args.intent_source), model, env_args
    )
    print(
        "[IntentPCA] Collection config: "
        f"episodes={int(args.episodes)} "
        f"feature_mode={args.feature_mode} "
        f"num_workers={requested_num_envs} "
        f"collector={worker_mode} "
        f"opponent_mode={opponent_meta.get('opponent_mode', str(args.opponent_mode))} "
        f"intent_source={resolved_intent_source} "
        f"player_deterministic={player_deterministic} "
        f"opponent_deterministic={opponent_deterministic} "
        f"require_active_intent={bool(getattr(args, 'require_active_intent', True))} "
        f"intent_null_prob={float(getattr(env_args, 'intent_null_prob', 0.0)):.3f}",
        flush=True,
    )
    if args.vec_env != "auto":
        print(
            f"[IntentPCA] Note: --vec-env={args.vec_env} is ignored in eval-worker collector mode.",
            flush=True,
        )

    if not bool(getattr(env_args, "enable_intent_learning", False)):
        raise RuntimeError("Intent PCA requires enable_intent_learning=true in the environment config.")

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        label = run_id or Path(policy_path).stem
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = REPO_ROOT / "analytics" / "intent_pca" / f"{label}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(int(args.seed))
    progress = tqdm(
        total=int(args.episodes),
        desc="Collecting intent episodes",
        unit="ep",
        dynamic_ncols=True,
    )
    progress.set_postfix(
        feature=args.feature_mode,
        workers=requested_num_envs,
        collector=worker_mode,
    )

    try:
        disc_max_obs_dim, disc_max_action_dim = _resolve_discriminator_feature_dims(
            disc_bundle,
            default_max_obs_dim=int(args.max_obs_dim),
            default_max_action_dim=int(args.max_action_dim),
        )
        episodes, metadata, disc_eval_episodes = _collect_episodes_parallel_workers(
            env_args=env_args,
            policy_path=str(policy_path),
            opponent_policy_paths=opponent_policy_paths,
            num_episodes=int(args.episodes),
            num_workers=requested_num_envs,
            device=str(args.device),
            feature_mode=str(args.feature_mode),
            player_deterministic=bool(player_deterministic),
            opponent_deterministic=bool(opponent_deterministic),
            intent_source_mode=str(args.intent_source),
            illegal_strategy_name=str(args.illegal_strategy),
            require_active_intent=bool(getattr(args, "require_active_intent", True)),
            max_obs_dim=int(args.max_obs_dim),
            max_action_dim=int(args.max_action_dim),
            collect_disc_episodes=disc_bundle is not None,
            disc_max_obs_dim=int(disc_max_obs_dim),
            disc_max_action_dim=int(disc_max_action_dim),
            progress=progress,
        )
    finally:
        progress.close()

    print(
        f"[IntentPCA] Collected {len(episodes)} accepted episodes using {requested_num_envs} eval workers.",
        flush=True,
    )

    x, labels, feature_names = _build_feature_matrix(
        args.feature_mode,
        episodes,
        metadata,
        max_obs_dim=int(args.max_obs_dim),
        max_action_dim=int(args.max_action_dim),
        disc_bundle=disc_bundle,
        device=str(args.device),
    )
    if x.shape[0] < 2:
        raise RuntimeError("Need at least two episodes to run PCA.")

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    pca = PCA(n_components=2, random_state=int(args.seed))
    coords = pca.fit_transform(x_scaled)
    explained = pca.explained_variance_ratio_

    requested_embedding_methods = list(dict.fromkeys(str(x).strip().lower() for x in args.embedding_methods))
    csv_path = output_dir / "intent_pca_points.csv"
    _write_embedding_points_csv(
        csv_path,
        metadata,
        coords,
        x_key="pca_x",
        y_key="pca_y",
    )

    plot_path = output_dir / "intent_pca_scatter.png"
    title = (
        f"Intent PCA ({args.feature_mode})\n"
        f"episodes={len(episodes)}  policy={Path(policy_path).name}"
    )
    _plot_pca(coords, labels, str(plot_path), title, explained)
    tsne_csv_path = output_dir / "intent_tsne_points.csv"
    tsne_plot_path = output_dir / "intent_tsne_scatter.png"
    umap_csv_path = output_dir / "intent_umap_points.csv"
    umap_plot_path = output_dir / "intent_umap_scatter.png"
    extra_embedding_outputs: dict[str, dict[str, object]] = {}
    if "tsne" in requested_embedding_methods:
        tsne_coords, effective_perplexity = _fit_tsne_coords(
            x_scaled,
            seed=int(args.seed),
            perplexity=float(args.tsne_perplexity),
        )
        _write_embedding_points_csv(
            tsne_csv_path,
            metadata,
            tsne_coords,
            x_key="tsne_x",
            y_key="tsne_y",
        )
        _plot_embedding_scatter(
            tsne_coords,
            labels,
            str(tsne_plot_path),
            (
                f"Intent t-SNE ({args.feature_mode})\n"
                f"episodes={len(episodes)}  policy={Path(policy_path).name}"
            ),
            xlabel="t-SNE 1",
            ylabel="t-SNE 2",
        )
        extra_embedding_outputs["tsne"] = {
            "csv_path": str(tsne_csv_path),
            "plot_path": str(tsne_plot_path),
            "effective_perplexity": float(effective_perplexity),
        }
    if "umap" in requested_embedding_methods:
        try:
            umap_coords, effective_neighbors = _fit_umap_coords(
                x_scaled,
                seed=int(args.seed),
                n_neighbors=int(args.umap_n_neighbors),
                min_dist=float(args.umap_min_dist),
            )
            _write_embedding_points_csv(
                umap_csv_path,
                metadata,
                umap_coords,
                x_key="umap_x",
                y_key="umap_y",
            )
            _plot_embedding_scatter(
                umap_coords,
                labels,
                str(umap_plot_path),
                (
                    f"Intent UMAP ({args.feature_mode})\n"
                    f"episodes={len(episodes)}  policy={Path(policy_path).name}"
                ),
                xlabel="UMAP 1",
                ylabel="UMAP 2",
            )
            extra_embedding_outputs["umap"] = {
                "csv_path": str(umap_csv_path),
                "plot_path": str(umap_plot_path),
                "effective_n_neighbors": int(effective_neighbors),
                "min_dist": float(args.umap_min_dist),
            }
        except ModuleNotFoundError as exc:
            print(f"[IntentPCA] Skipping UMAP: {exc}", flush=True)
            extra_embedding_outputs["umap"] = {
                "status": "skipped_missing_dependency",
                "reason": str(exc),
            }

    confusion_counts_path = output_dir / "intent_confusion_matrix_counts.csv"
    confusion_row_norm_path = output_dir / "intent_confusion_matrix_row_normalized.csv"
    confusion_plot_path = output_dir / "intent_confusion_matrix.png"
    pca_disc_eval_batch_path = output_dir / "intent_disc_eval_batch_from_pca.npz"

    summary = {
        "policy_path": policy_path,
        "run_id": run_id,
        "mlflow_artifact_path": _intent_pca_mlflow_artifact_path(policy_path) if run_id else None,
        "episodes": int(len(episodes)),
        "feature_mode": str(args.feature_mode),
        "embedding_methods": requested_embedding_methods,
        "opponent_mode": opponent_meta.get("opponent_mode", str(args.opponent_mode)),
        "intent_source_requested": str(args.intent_source),
        "intent_source_resolved": resolved_intent_source,
        "player_deterministic": bool(player_deterministic),
        "opponent_deterministic": bool(opponent_deterministic),
        "use_set_obs": bool(getattr(env_args, "use_set_obs", False)),
        "mirror_episode_prob": float(getattr(env_args, "mirror_episode_prob", 0.0)),
        "explained_variance_ratio": [float(x) for x in explained.tolist()],
        "intent_counts": {
            str(int(z)): int(np.sum(labels == z)) for z in sorted({int(v) for v in labels.tolist()})
        },
        "feature_dim": int(x.shape[1]),
        "summary_feature_names": feature_names if args.feature_mode == "summary" else None,
        "embedding_outputs": extra_embedding_outputs,
        "discriminator_checkpoint_meta": (
            (disc_bundle[1] or {}).get("meta", None) if disc_bundle is not None else None
        ),
        "discriminator_checkpoint_config": (
            (disc_bundle[1] or {}).get("config", None) if disc_bundle is not None else None
        ),
        "discriminator_checkpoint_path": disc_path,
        "discriminator_eval_representation": (
            "raw_step_features" if disc_bundle is not None else None
        ),
        "opponent_sampling_meta": opponent_meta,
        "selector_applied_episodes": int(
            sum(1 for meta in metadata if bool(meta.get("selector_applied", False)))
        ),
        "selector_applied_rate": float(
            (
                sum(1 for meta in metadata if bool(meta.get("selector_applied", False)))
                / float(max(1, len(metadata)))
            )
        ),
        "collector_start_label_match_rate": float(
            sum(1 for meta in metadata if bool(meta.get("collector_start_label_matches", False)))
            / float(max(1, len(metadata)))
        ),
        "collector_disc_start_label_match_rate": float(
            sum(
                1
                for meta in metadata
                if meta.get("collector_disc_start_label_matches", None) is True
            )
            / float(
                max(
                    1,
                    sum(
                        1
                        for meta in metadata
                        if meta.get("collector_disc_start_label_matches", None) is not None
                    ),
                )
            )
        ),
    }
    if len(set(labels.tolist())) > 1:
        try:
            summary["silhouette_score_pca2"] = float(silhouette_score(coords, labels))
        except Exception:
            pass

    if disc_bundle is not None:
        selected_disc_eval_episodes = _select_discriminator_eval_episodes(
            episodes, disc_eval_episodes
        )
        y_true_disc, y_pred_disc, probs_disc, lengths_disc = _compute_discriminator_predictions(
            selected_disc_eval_episodes,
            disc_bundle=disc_bundle,
            max_obs_dim=int(args.max_obs_dim),
            max_action_dim=int(args.max_action_dim),
            device=str(args.device),
        )
        disc_max_obs_dim, disc_max_action_dim = _resolve_discriminator_feature_dims(
            disc_bundle,
            default_max_obs_dim=int(args.max_obs_dim),
            default_max_action_dim=int(args.max_action_dim),
        )
        disc_cfg = dict((disc_bundle[1] or {}).get("config", {}) or {})
        disc_encoder_type = str(disc_cfg.get("encoder_type", "mlp_mean")).strip().lower()
        if disc_encoder_type == "gru":
            x_disc_np, lengths_save_np, y_save_np = build_padded_episode_batch(
                selected_disc_eval_episodes,
                max_obs_dim=disc_max_obs_dim,
                max_action_dim=disc_max_action_dim,
            )
        else:
            x_disc_np, y_save_np = compute_episode_embeddings(
                selected_disc_eval_episodes,
                max_obs_dim=disc_max_obs_dim,
                max_action_dim=disc_max_action_dim,
            )
            lengths_save_np = None
        np.savez_compressed(
            pca_disc_eval_batch_path,
            x=np.asarray(x_disc_np, copy=False),
            y=np.asarray(y_save_np, dtype=np.int64, copy=False),
            lengths=(
                np.asarray(lengths_save_np, dtype=np.int64, copy=False)
                if lengths_save_np is not None
                else np.asarray([], dtype=np.int64)
            ),
            y_pred=np.asarray(y_pred_disc, dtype=np.int64, copy=False),
            probs=np.asarray(probs_disc, dtype=np.float32, copy=False),
            encoder_type=np.asarray([disc_encoder_type]),
            max_obs_dim=np.asarray([int(disc_max_obs_dim)], dtype=np.int64),
            max_action_dim=np.asarray([int(disc_max_action_dim)], dtype=np.int64),
        )
        disc_num_intents = int(
            ((disc_bundle[1] or {}).get("config", {}) or {}).get("num_intents", 0)
            or max(1, np.max(y_true_disc) + 1)
        )
        cm_labels, cm_counts, cm_row_norm = _compute_confusion_matrix_tables(
            y_true_disc,
            y_pred_disc,
            num_intents=disc_num_intents,
        )
        _write_confusion_matrix_csv(confusion_counts_path, cm_labels, cm_counts)
        _write_confusion_matrix_csv(
            confusion_row_norm_path,
            cm_labels,
            cm_row_norm,
            float_format=".6f",
        )
        _plot_confusion_matrices(
            cm_labels,
            cm_counts,
            cm_row_norm,
            str(confusion_plot_path),
            (
                f"Intent Confusion Matrix\n"
                f"episodes={len(episodes)}  policy={Path(policy_path).name}"
            ),
        )
        summary["discriminator_eval_top1_acc"] = float(
            np.mean(y_true_disc == y_pred_disc)
        )
        summary["discriminator_eval_num_samples"] = int(y_true_disc.shape[0])
        summary["discriminator_eval_label_histogram"] = (
            np.bincount(y_true_disc, minlength=disc_num_intents).astype(int).tolist()
        )
        summary["discriminator_eval_predicted_histogram"] = (
            np.bincount(y_pred_disc, minlength=disc_num_intents).astype(int).tolist()
        )
        summary["discriminator_eval_batch_path"] = str(pca_disc_eval_batch_path)
        summary["confusion_matrix_labels"] = [int(z) for z in cm_labels]
        summary["confusion_matrix_counts"] = cm_counts.astype(int).tolist()
        summary["confusion_matrix_row_normalized"] = [
            [float(v) for v in row] for row in cm_row_norm.tolist()
        ]

    summary_path = output_dir / "intent_pca_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    if args.log_to_mlflow and run_id:
        try:
            client = mlflow.tracking.MlflowClient()
            artifact_path = _intent_pca_mlflow_artifact_path(policy_path)
            print(
                f"[IntentPCA] Logging MLflow artifacts under: {artifact_path}",
                flush=True,
            )
            client.log_artifact(run_id, str(csv_path), artifact_path=artifact_path)
            client.log_artifact(run_id, str(plot_path), artifact_path=artifact_path)
            if tsne_csv_path.exists():
                client.log_artifact(run_id, str(tsne_csv_path), artifact_path=artifact_path)
            if tsne_plot_path.exists():
                client.log_artifact(run_id, str(tsne_plot_path), artifact_path=artifact_path)
            if umap_csv_path.exists():
                client.log_artifact(run_id, str(umap_csv_path), artifact_path=artifact_path)
            if umap_plot_path.exists():
                client.log_artifact(run_id, str(umap_plot_path), artifact_path=artifact_path)
            if confusion_counts_path.exists():
                client.log_artifact(
                    run_id, str(confusion_counts_path), artifact_path=artifact_path
                )
            if confusion_row_norm_path.exists():
                client.log_artifact(
                    run_id, str(confusion_row_norm_path), artifact_path=artifact_path
                )
            if confusion_plot_path.exists():
                client.log_artifact(
                    run_id, str(confusion_plot_path), artifact_path=artifact_path
                )
            if pca_disc_eval_batch_path.exists():
                client.log_artifact(
                    run_id, str(pca_disc_eval_batch_path), artifact_path=artifact_path
                )
            client.log_artifact(run_id, str(summary_path), artifact_path=artifact_path)
        except Exception:
            pass

    print("\n=== Intent PCA Analysis ===")
    print(f"Policy: {policy_path}")
    print(f"Env params source: {run_id or 'local-only'}")
    print(f"Episodes: {len(episodes)}")
    print(f"Feature mode: {args.feature_mode}")
    print(f"Embedding methods: {', '.join(requested_embedding_methods)}")
    print(f"Collection workers: {requested_num_envs} (eval_workers)")
    print(f"Feature dim: {x.shape[1]}")
    print(
        "Explained variance: "
        f"PC1={100.0 * float(explained[0]):.2f}%  "
        f"PC2={100.0 * float(explained[1]):.2f}%"
    )
    if "silhouette_score_pca2" in summary:
        print(f"Silhouette (PCA-2): {float(summary['silhouette_score_pca2']):.4f}")
    if "discriminator_eval_top1_acc" in summary:
        print(
            f"Discriminator top1: {float(summary['discriminator_eval_top1_acc']):.4f}"
        )
        print(f"Confusion counts CSV: {confusion_counts_path}")
        print(f"Confusion row-norm CSV: {confusion_row_norm_path}")
        print(f"Confusion plot: {confusion_plot_path}")
        print(f"PCA disc eval batch: {pca_disc_eval_batch_path}")
    print(f"CSV: {csv_path}")
    print(f"Plot: {plot_path}")
    if tsne_plot_path.exists():
        print(f"t-SNE CSV: {tsne_csv_path}")
        print(f"t-SNE Plot: {tsne_plot_path}")
    if umap_plot_path.exists():
        print(f"UMAP CSV: {umap_csv_path}")
        print(f"UMAP Plot: {umap_plot_path}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
