#!/usr/bin/env python3
import argparse
import glob
import os
import re
import tempfile

import numpy as np
import basketworld
from stable_baselines3 import PPO

from basketworld.utils.action_resolution import get_policy_action_probabilities
from basketworld.utils.policies import PassBiasDualCriticPolicy, PassBiasMultiInputPolicy
from basketworld.utils.wrappers import SetObservationWrapper
from basketworld.policies import SetAttentionDualCriticPolicy, SetAttentionExtractor


def _latest_zip_in_dir(dir_path: str) -> str | None:
    candidates = sorted(glob.glob(os.path.join(dir_path, "*.zip")), key=os.path.getmtime)
    return candidates[-1] if candidates else None


def _download_latest_unified(run_id: str) -> str:
    import mlflow

    client = mlflow.tracking.MlflowClient()
    artifacts = client.list_artifacts(run_id, "models")
    pattern = re.compile(r"unified_(?:alternation|iter)_(\d+)\\.zip$")
    candidates: list[tuple[int, str]] = []
    for item in artifacts:
        match = pattern.search(item.path)
        if match:
            candidates.append((int(match.group(1)), item.path))
    if candidates:
        candidates.sort(key=lambda v: v[0])
        artifact_path = candidates[-1][1]
    else:
        fallback = [item.path for item in artifacts if item.path.endswith("unified_latest.zip")]
        if not fallback:
            raise RuntimeError(f"No unified policy artifacts found for run_id={run_id}")
        artifact_path = fallback[-1]
    tmpdir = tempfile.mkdtemp(prefix="perm_kl_")
    return client.download_artifacts(run_id, artifact_path, tmpdir)


def resolve_policy_path(input_arg: str) -> str:
    if os.path.isfile(input_arg):
        return input_arg
    if os.path.isdir(input_arg):
        path = _latest_zip_in_dir(input_arg)
        if path:
            return path
    cache_dir = os.path.join(".opponent_cache", input_arg)
    if os.path.isdir(cache_dir):
        path = _latest_zip_in_dir(cache_dir)
        if path:
            return path
    if os.path.isfile(f"{input_arg}.zip"):
        return f"{input_arg}.zip"
    return _download_latest_unified(input_arg)


def _sample_random_actions(action_mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    actions = []
    for row in action_mask:
        legal = np.flatnonzero(np.asarray(row) == 1)
        if legal.size == 0:
            actions.append(0)
        else:
            actions.append(int(rng.choice(legal)))
    return np.asarray(actions, dtype=int)


def _normalize_probs(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, eps, None)
    return p / p.sum()


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = _normalize_probs(p)
    q = _normalize_probs(q)
    return float(np.sum(p * np.log(p / q)))


def _get_policy_probs(policy: PPO, obs: dict) -> list[np.ndarray]:
    probs = get_policy_action_probabilities(policy, obs)
    if probs is None:
        raise RuntimeError("Failed to compute action probabilities for policy.")
    return [np.asarray(p, dtype=np.float64).reshape(-1) for p in probs]


def _swap_players_obs(obs: dict, i: int, j: int) -> dict:
    swapped = dict(obs)
    swapped_players = np.copy(obs["players"])
    swapped_players[[i, j]] = swapped_players[[j, i]]
    swapped["players"] = swapped_players
    if "action_mask" in obs:
        swapped_mask = np.copy(obs["action_mask"])
        swapped_mask[[i, j]] = swapped_mask[[j, i]]
        swapped["action_mask"] = swapped_mask
    return swapped


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute KL divergence under player-token swap.")
    parser.add_argument("run_id_or_path", help="MLflow run_id or path to a policy .zip")
    parser.add_argument("--players", type=int, default=3, help="Players per side (3 = 3v3)")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for env reset")
    parser.add_argument("--warmup-steps", type=int, default=5, help="Random steps before measuring")
    parser.add_argument("--swap", default="0,1", help="Comma-separated player indices to swap")
    args = parser.parse_args()

    swap_parts = [int(p.strip()) for p in args.swap.split(",") if p.strip()]
    if len(swap_parts) != 2:
        raise SystemExit("Swap must be two comma-separated indices, e.g. 0,1")
    idx_a, idx_b = swap_parts

    policy_path = resolve_policy_path(args.run_id_or_path)
    print(f"Using policy: {policy_path}")

    env = SetObservationWrapper(basketworld.HexagonBasketballEnv(players=args.players, render_mode=None))
    rng = np.random.default_rng(args.seed)
    obs, _ = env.reset(seed=args.seed)
    for _ in range(max(0, args.warmup_steps)):
        actions = _sample_random_actions(obs["action_mask"], rng)
        obs, _, terminated, truncated, _ = env.step(actions)
        if terminated or truncated:
            obs, _ = env.reset(seed=args.seed)

    custom_objects = {
        "PassBiasDualCriticPolicy": PassBiasDualCriticPolicy,
        "PassBiasMultiInputPolicy": PassBiasMultiInputPolicy,
        "SetAttentionDualCriticPolicy": SetAttentionDualCriticPolicy,
        "SetAttentionExtractor": SetAttentionExtractor,
    }
    policy = PPO.load(policy_path, custom_objects=custom_objects, device="cpu")

    probs_orig = _get_policy_probs(policy, obs)
    n_players = len(probs_orig)
    if idx_a >= n_players or idx_b >= n_players:
        raise SystemExit(f"Swap indices out of range. n_players={n_players}")

    swapped_obs = _swap_players_obs(obs, idx_a, idx_b)
    probs_swapped = _get_policy_probs(policy, swapped_obs)

    perm = list(range(n_players))
    perm[idx_a], perm[idx_b] = perm[idx_b], perm[idx_a]
    probs_swapped_reordered = [probs_swapped[perm[k]] for k in range(n_players)]

    kls = [_kl_divergence(probs_orig[k], probs_swapped_reordered[k]) for k in range(n_players)]
    rev_kls = [_kl_divergence(probs_swapped_reordered[k], probs_orig[k]) for k in range(n_players)]
    sym_kls = [(kls[k] + rev_kls[k]) / 2.0 for k in range(n_players)]

    print(f"Swap indices: {idx_a} <-> {idx_b}")
    print(f"KL mean: {float(np.mean(kls)):.6f} | KL max: {float(np.max(kls)):.6f}")
    print(f"Sym KL mean: {float(np.mean(sym_kls)):.6f} | Sym KL max: {float(np.max(sym_kls)):.6f}")
    for k, val in enumerate(kls):
        print(f"P{k} KL: {val:.6f} | Sym KL: {sym_kls[k]:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
