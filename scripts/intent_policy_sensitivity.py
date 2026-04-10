#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import tempfile
from pathlib import Path

import mlflow
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3 import PPO

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basketworld.envs.basketworld_env_v2 import Team
from basketworld.policies import SetAttentionDualCriticPolicy, SetAttentionExtractor
from basketworld.utils.intent_policy_sensitivity import (
    clone_observation_dict,
    compute_policy_sensitivity_metrics,
    extract_single_env_observation,
    infer_num_intents,
)
from basketworld.utils.mlflow_config import setup_mlflow
from basketworld.utils.mlflow_params import get_mlflow_params
from basketworld.utils.policies import PassBiasDualCriticPolicy, PassBiasMultiInputPolicy
from train.config import get_args
from train.env_factory import setup_environment


def _latest_zip_in_dir(dir_path: str) -> str | None:
    candidates = sorted(glob.glob(os.path.join(dir_path, "*.zip")), key=os.path.getmtime)
    return candidates[-1] if candidates else None


def _download_latest_unified(run_id: str) -> str:
    client = mlflow.tracking.MlflowClient()
    artifacts = client.list_artifacts(run_id, "models")
    pattern = re.compile(r"unified_(?:alternation|iter)_(\d+)\.zip$")
    candidates: list[tuple[int, str]] = []
    for item in artifacts:
        match = pattern.search(item.path)
        if match:
            candidates.append((int(match.group(1)), item.path))
    if candidates:
        candidates.sort(key=lambda item: item[0])
        artifact_path = candidates[-1][1]
    else:
        fallback = [item.path for item in artifacts if item.path.endswith("unified_latest.zip")]
        if not fallback:
            raise RuntimeError(f"No unified policy artifacts found for run_id={run_id}")
        artifact_path = fallback[-1]
    tmpdir = tempfile.mkdtemp(prefix="intent_policy_sensitivity_")
    return client.download_artifacts(run_id, artifact_path, tmpdir)


def resolve_policy_path(input_arg: str) -> tuple[str, str | None]:
    if os.path.isfile(input_arg):
        return input_arg, None
    if os.path.isdir(input_arg):
        latest = _latest_zip_in_dir(input_arg)
        if latest is not None:
            return latest, None
    cache_dir = os.path.join(".opponent_cache", input_arg)
    if os.path.isdir(cache_dir):
        latest = _latest_zip_in_dir(cache_dir)
        if latest is not None:
            return latest, input_arg
    if os.path.isfile(f"{input_arg}.zip"):
        return f"{input_arg}.zip", None
    return _download_latest_unified(input_arg), input_arg


def _parse_intents(raw: str | None) -> list[int] | None:
    if raw is None or not raw.strip():
        return None
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _sample_random_actions(action_mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    actions = []
    for row in np.asarray(action_mask):
        legal = np.flatnonzero(np.asarray(row) == 1)
        if legal.size == 0:
            actions.append(0)
        else:
            actions.append(int(rng.choice(legal)))
    return np.asarray(actions, dtype=int)


def _sample_random_actions_batch(
    action_mask: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    mask = np.asarray(action_mask)
    if mask.ndim != 3:
        raise ValueError(
            f"Expected action_mask with shape (num_envs, num_players, num_actions), got {mask.shape}"
        )
    actions = np.zeros((mask.shape[0], mask.shape[1]), dtype=int)
    for env_idx in range(mask.shape[0]):
        actions[env_idx] = _sample_random_actions(mask[env_idx], rng)
    return actions


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


def _build_env_args(run_id: str | None, model: PPO) -> tuple[argparse.Namespace, int]:
    args = get_args([])
    mlflow_num_intents = None
    if run_id is not None:
        client = mlflow.tracking.MlflowClient()
        required, optional = get_mlflow_params(client, run_id)
        _apply_run_params(args, required, optional)
        mlflow_num_intents = optional.get("num_intents")
    args.use_set_obs = _policy_uses_set_obs(model)
    args.enable_env_profiling = False
    args.mirror_episode_prob = 0.0
    num_intents = int(mlflow_num_intents or infer_num_intents(model, default=8))
    return args, max(1, num_intents)


def _make_offense_env_fn(env_args: argparse.Namespace, env_idx: int):
    def _thunk():
        return setup_environment(env_args, Team.OFFENSE, env_idx=env_idx)

    return _thunk


def _make_collection_env(
    env_args: argparse.Namespace,
    *,
    num_envs: int,
    backend: str,
):
    env_fns = [_make_offense_env_fn(env_args, env_idx=i) for i in range(int(num_envs))]
    if backend == "dummy" or int(num_envs) <= 1:
        return DummyVecEnv(env_fns), "dummy"
    return SubprocVecEnv(env_fns, start_method="spawn"), "subproc"


def _collect_states(
    vec_env,
    *,
    rng: np.random.Generator,
    seed: int,
    warmup_steps: int,
    num_states: int,
) -> list[dict]:
    observations: list[dict] = []
    try:
        vec_env.seed(int(seed))
    except Exception:
        pass
    obs = vec_env.reset()
    steps_taken = 0

    while len(observations) < num_states:
        if steps_taken >= max(0, warmup_steps):
            batch_size = 0
            if isinstance(obs, dict):
                for value in obs.values():
                    arr = np.asarray(value)
                    if arr.ndim > 0:
                        batch_size = int(arr.shape[0])
                        break
            for env_idx in range(batch_size):
                if len(observations) >= num_states:
                    break
                observations.append(
                    clone_observation_dict(
                        extract_single_env_observation(
                            obs,
                            env_idx=env_idx,
                            expected_batch_size=batch_size,
                        )
                    )
                )

        action_mask = np.asarray(obs["action_mask"])
        actions = _sample_random_actions_batch(action_mask, rng)
        obs, _, _, _ = vec_env.step(actions)
        steps_taken += 1

    return observations


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure how strongly a policy's action distribution depends on latent intent."
    )
    parser.add_argument("run_id_or_path", help="MLflow run ID or path to a policy .zip")
    parser.add_argument(
        "--mlflow-run-id",
        default=None,
        help="Optional MLflow run ID to source environment params when loading a local policy path.",
    )
    parser.add_argument("--states", type=int, default=64, help="Number of states to evaluate.")
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=16,
        help="Random env steps before starting state collection.",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for env reset and rollouts.")
    parser.add_argument("--device", default="cpu", help="Torch device for policy inference.")
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Number of offense environments to use for parallel state collection.",
    )
    parser.add_argument(
        "--vec-env",
        choices=["auto", "dummy", "subproc"],
        default="auto",
        help="Vector environment backend for state collection.",
    )
    parser.add_argument(
        "--candidate-intents",
        default=None,
        help="Comma-separated subset of intent IDs to evaluate. Defaults to all intents.",
    )
    parser.add_argument(
        "--num-intents",
        type=int,
        default=None,
        help="Override num_intents when MLflow params are unavailable.",
    )
    args = parser.parse_args()

    setup_mlflow(verbose=False)

    policy_path, resolved_run_id = resolve_policy_path(args.run_id_or_path)
    env_run_id = args.mlflow_run_id or resolved_run_id
    model = PPO.load(policy_path, device=args.device, custom_objects=_custom_objects())
    env_args, inferred_num_intents = _build_env_args(env_run_id, model)
    num_intents = max(1, int(args.num_intents or inferred_num_intents))
    candidate_intents = _parse_intents(args.candidate_intents)

    backend = "dummy" if args.vec_env == "auto" and int(args.num_envs) <= 1 else (
        "subproc" if args.vec_env == "auto" else args.vec_env
    )
    env, backend_used = _make_collection_env(
        env_args,
        num_envs=max(1, int(args.num_envs)),
        backend=backend,
    )
    rng = np.random.default_rng(args.seed)
    try:
        observations = _collect_states(
            env,
            rng=rng,
            seed=args.seed,
            warmup_steps=args.warmup_steps,
            num_states=args.states,
        )
    finally:
        env.close()

    metrics = compute_policy_sensitivity_metrics(
        model,
        observations,
        num_intents=num_intents,
        candidate_intents=candidate_intents,
        active=1.0,
        visible=1.0,
        age_norm=0.0,
        commitment_remaining_norm=1.0,
    )

    print("=== Intent Policy Sensitivity ===")
    print(f"Policy: {policy_path}")
    print(f"Env params source: {env_run_id or 'train defaults'}")
    print(f"Use set obs: {bool(getattr(env_args, 'use_set_obs', False))}")
    print(f"State collection envs: {max(1, int(args.num_envs))} ({backend_used})")
    print(f"Num intents: {num_intents}")
    if candidate_intents is not None:
        print(f"Candidate intents: {candidate_intents}")
    print(f"States evaluated: {int(metrics['num_states'])}")
    print(f"Intent pairs evaluated: {int(metrics['num_pairs'])}")
    print(f"Symmetric KL mean: {metrics['policy_kl_mean']:.6f}")
    print(f"Symmetric KL max: {metrics['policy_kl_max']:.6f}")
    print(f"TV mean: {metrics['policy_tv_mean']:.6f}")
    print(f"Top-action flip rate: {metrics['action_flip_rate']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
