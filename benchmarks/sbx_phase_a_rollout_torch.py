from __future__ import annotations

from pathlib import Path
from time import perf_counter_ns
from typing import Any, Sequence
import sys

import numpy as np

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

try:
    from benchmarks.common import (
        build_benchmark_parser,
        build_progress,
        resolve_training_team,
        write_json,
    )
    from benchmarks.sbx_phase_a import (
        build_phase_a_policy_spec,
        extract_phase_a_training_action_mask,
        flatten_phase_a_observation,
    )
    from benchmarks.sbx_phase_a_torch import (
        PhaseATorchPolicy,
        require_torch,
        run_torch_policy_once,
        _sync_torch,
    )
except ImportError:  # pragma: no cover - direct script execution path
    from common import build_benchmark_parser, build_progress, resolve_training_team, write_json  # type: ignore[no-redef]
    from sbx_phase_a import (  # type: ignore[no-redef]
        build_phase_a_policy_spec,
        extract_phase_a_training_action_mask,
        flatten_phase_a_observation,
    )
    from sbx_phase_a_torch import (  # type: ignore[no-redef]
        PhaseATorchPolicy,
        require_torch,
        run_torch_policy_once,
        _sync_torch,
    )

from train.env_factory import setup_environment


def parse_args(argv=None):
    parser = build_benchmark_parser(
        "Reduced Python rollout baseline matching the compiled JAX rollout structure as closely as possible."
    )
    parser.set_defaults(mode="throughput", runner="sequential", device="cpu")
    parser.add_argument(
        "--kernel-batch-size",
        type=int,
        default=256,
        help="Number of env instances to keep active in the rollout batch.",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=3,
        help="Number of rollout windows to run before timing.",
    )
    parser.add_argument(
        "--benchmark-iters",
        type=int,
        default=20,
        help="Number of timed rollout windows to measure.",
    )
    parser.add_argument(
        "--sample-reset-seed",
        type=int,
        default=0,
        help="Base seed used for initial resets and per-env auto-reset reseeding.",
    )
    parser.add_argument(
        "--policy-hidden-dims",
        type=int,
        nargs="+",
        default=[128, 128],
        help="Hidden layer widths for the reduced flat Torch policy.",
    )
    parser.add_argument(
        "--policy-seed",
        type=int,
        default=0,
        help="Random seed used to initialize the reduced flat Torch policy.",
    )
    return parser.parse_args(argv)


def _minimal_rollout_blockers(args) -> list[str]:
    blockers = []
    if bool(getattr(args, "enable_phi_shaping", False)):
        blockers.append("phi_shaping")
    if bool(getattr(args, "illegal_defense_enabled", False)):
        blockers.append("illegal_defense")
    if bool(getattr(args, "offensive_three_seconds", False)):
        blockers.append("offensive_three_seconds")
    return blockers


def _build_env_batch(args):
    training_team = resolve_training_team(args.training_team)
    return [setup_environment(args, training_team) for _ in range(int(args.kernel_batch_size))]


def _resolve_team_player_ids(env, training_team) -> tuple[np.ndarray, np.ndarray]:
    base_env = env.unwrapped
    if training_team == resolve_training_team("defense"):
        training_ids = np.asarray(base_env.defense_ids, dtype=np.int64)
        opponent_ids = np.asarray(base_env.offense_ids, dtype=np.int64)
    else:
        training_ids = np.asarray(base_env.offense_ids, dtype=np.int64)
        opponent_ids = np.asarray(base_env.defense_ids, dtype=np.int64)
    return training_ids, opponent_ids


def _initial_reset_seed(base_seed: int, env_idx: int) -> int:
    return int(base_seed) + (int(env_idx) * 1_000_003)


def _next_reset_seed(base_seed: int, env_idx: int, reset_count: int) -> int:
    return int(base_seed) + (int(env_idx) * 1_000_003) + int(reset_count)


def _prepare_policy_inputs(
    obs_batch: Sequence[dict[str, Any]],
    training_player_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flat_rows = [flatten_phase_a_observation(obs) for obs in obs_batch]
    training_masks = [
        extract_phase_a_training_action_mask(obs["action_mask"], training_player_ids)
        for obs in obs_batch
    ]
    full_masks = [np.asarray(obs["action_mask"], dtype=np.int8) for obs in obs_batch]
    return (
        np.stack(flat_rows, axis=0).astype(np.float32, copy=False),
        np.stack(training_masks, axis=0).astype(np.int8, copy=False),
        np.stack(full_masks, axis=0).astype(np.int8, copy=False),
    )


def _sample_random_team_actions(
    action_masks: np.ndarray,
    rngs: Sequence[np.random.Generator],
) -> np.ndarray:
    out = np.zeros(action_masks.shape[:2], dtype=np.int64)
    for env_idx in range(action_masks.shape[0]):
        for player_idx in range(action_masks.shape[1]):
            legal = np.flatnonzero(action_masks[env_idx, player_idx])
            out[env_idx, player_idx] = int(rngs[env_idx].choice(legal)) if legal.size else 0
    return out


def _assemble_full_actions(
    training_actions: np.ndarray,
    opponent_actions: np.ndarray,
    training_ids: np.ndarray,
    opponent_ids: np.ndarray,
    n_players: int,
) -> np.ndarray:
    batch_size = int(training_actions.shape[0])
    full_actions = np.zeros((batch_size, int(n_players)), dtype=np.int64)
    full_actions[:, training_ids] = np.asarray(training_actions, dtype=np.int64)
    full_actions[:, opponent_ids] = np.asarray(opponent_actions, dtype=np.int64)
    return full_actions


def _build_torch_policy(
    torch_mod,
    args,
    flat_obs_batch: np.ndarray,
    training_mask_batch: np.ndarray,
    device,
):
    spec = build_phase_a_policy_spec(
        flat_obs_batch,
        training_mask_batch,
        hidden_dims=args.policy_hidden_dims,
    )
    policy = PhaseATorchPolicy(
        torch_mod,
        spec,
        hidden_dims=args.policy_hidden_dims,
        seed=int(args.policy_seed),
        device=device,
    )
    return policy, spec


def _run_rollout_windows(
    *,
    args,
    envs,
    obs_batch,
    rngs,
    training_ids: np.ndarray,
    opponent_ids: np.ndarray,
    use_training_policy: bool,
    policy,
    torch_mod,
    device,
    progress,
    phase_label: str,
    iterations: int,
) -> tuple[dict[str, float], Sequence[dict[str, Any]]]:
    timings_ns = {
        "obs_prep_ns": 0,
        "training_policy_ns": 0,
        "training_random_ns": 0,
        "opponent_random_ns": 0,
        "assemble_actions_ns": 0,
        "env_step_ns": 0,
        "env_reset_ns": 0,
    }
    total_states = int(args.kernel_batch_size) * int(args.horizon) * int(iterations)
    auto_resets = 0
    total_done = 0
    total_truncated = 0
    reset_counts = np.ones((int(args.kernel_batch_size),), dtype=np.int64)
    n_players = int(obs_batch[0]["action_mask"].shape[0])

    for window_idx in range(int(iterations)):
        for _ in range(int(args.horizon)):
            prep_start_ns = perf_counter_ns()
            flat_obs_batch, training_mask_batch, full_action_mask_batch = _prepare_policy_inputs(
                obs_batch,
                training_ids,
            )
            opponent_mask_batch = full_action_mask_batch[:, opponent_ids, :]
            timings_ns["obs_prep_ns"] += perf_counter_ns() - prep_start_ns

            if use_training_policy:
                torch_obs = torch_mod.as_tensor(flat_obs_batch, dtype=torch_mod.float32, device=device)
                torch_mask = torch_mod.as_tensor(training_mask_batch, dtype=torch_mod.int8, device=device)
                policy_start_ns = perf_counter_ns()
                with torch_mod.inference_mode():
                    out = run_torch_policy_once(torch_mod, policy, torch_obs, torch_mask, policy.spec)
                    _sync_torch(torch_mod, device)
                timings_ns["training_policy_ns"] += perf_counter_ns() - policy_start_ns
                training_actions = (
                    out["sampled_actions"].detach().cpu().numpy().astype(np.int64, copy=False)
                )
            else:
                random_start_ns = perf_counter_ns()
                training_actions = _sample_random_team_actions(training_mask_batch, rngs)
                timings_ns["training_random_ns"] += perf_counter_ns() - random_start_ns

            opponent_start_ns = perf_counter_ns()
            opponent_actions = _sample_random_team_actions(opponent_mask_batch, rngs)
            timings_ns["opponent_random_ns"] += perf_counter_ns() - opponent_start_ns

            assemble_start_ns = perf_counter_ns()
            full_actions = _assemble_full_actions(
                training_actions,
                opponent_actions,
                training_ids,
                opponent_ids,
                n_players,
            )
            timings_ns["assemble_actions_ns"] += perf_counter_ns() - assemble_start_ns

            next_obs_batch: list[dict[str, Any]] = []
            for env_idx, env in enumerate(envs):
                step_start_ns = perf_counter_ns()
                next_obs, _, terminated, truncated, _ = env.step(full_actions[env_idx])
                timings_ns["env_step_ns"] += perf_counter_ns() - step_start_ns
                if bool(terminated or truncated):
                    reset_start_ns = perf_counter_ns()
                    reset_seed = _next_reset_seed(args.sample_reset_seed, env_idx, reset_counts[env_idx])
                    next_obs, _ = env.reset(seed=reset_seed)
                    timings_ns["env_reset_ns"] += perf_counter_ns() - reset_start_ns
                    reset_counts[env_idx] += 1
                    auto_resets += 1
                    total_done += int(bool(terminated))
                    total_truncated += int(bool(truncated))
                next_obs_batch.append(next_obs)
            obs_batch = next_obs_batch
        progress.update(1)
        progress.set_postfix_str(
            f"{phase_label} windows={window_idx + 1} states={(window_idx + 1) * int(args.kernel_batch_size) * int(args.horizon)}",
            refresh=False,
        )

    rollout_ns = sum(timings_ns.values())
    total_seconds = max(rollout_ns / 1e9, 1e-12)
    return (
        {
            "total_states": float(total_states),
            "rollout_time_sec": float(total_seconds),
            "states_per_sec": float(total_states) / total_seconds,
            "mean_window_latency_ms": (rollout_ns / 1e6) / max(1, int(iterations)),
            "obs_prep_time_sec": timings_ns["obs_prep_ns"] / 1e9,
            "training_policy_time_sec": timings_ns["training_policy_ns"] / 1e9,
            "training_random_time_sec": timings_ns["training_random_ns"] / 1e9,
            "opponent_random_time_sec": timings_ns["opponent_random_ns"] / 1e9,
            "assemble_actions_time_sec": timings_ns["assemble_actions_ns"] / 1e9,
            "env_step_time_sec": timings_ns["env_step_ns"] / 1e9,
            "env_reset_time_sec": timings_ns["env_reset_ns"] / 1e9,
            "auto_resets_from_done": float(auto_resets),
            "episodes_completed_done": float(total_done),
            "episodes_completed_truncated": float(total_truncated),
        },
        obs_batch,
    )


def _run_single_metric(args, *, use_training_policy: bool) -> dict[str, Any]:
    torch_mod = require_torch()
    blockers = _minimal_rollout_blockers(args)
    if blockers:
        raise SystemExit(
            "benchmarks/sbx_phase_a_rollout_torch.py only supports the reduced transition scope: "
            + ", ".join(blockers)
        )

    device_name = str(getattr(args, "device", "cpu") or "cpu")
    if device_name == "auto":
        device_name = "cuda" if torch_mod.cuda.is_available() else "cpu"
    device = torch_mod.device(device_name)
    training_team = resolve_training_team(args.training_team)
    envs = _build_env_batch(args)
    rngs = [
        np.random.default_rng(int(args.seed) + (env_idx * 10_000) + 17)
        for env_idx in range(int(args.kernel_batch_size))
    ]
    progress = build_progress(
        total=int(args.warmup_iters) + int(args.benchmark_iters),
        desc="phase_a_rollout_torch",
        disable=bool(args.no_progress),
        unit="window",
    )

    try:
        obs_batch: list[dict[str, Any]] = []
        for env_idx, env in enumerate(envs):
            obs, _ = env.reset(seed=_initial_reset_seed(args.sample_reset_seed, env_idx))
            obs_batch.append(obs)

        training_ids, opponent_ids = _resolve_team_player_ids(envs[0], training_team)
        flat_obs_batch, training_mask_batch, _ = _prepare_policy_inputs(obs_batch, training_ids)
        policy, spec = _build_torch_policy(
            torch_mod,
            args,
            flat_obs_batch,
            training_mask_batch,
            device,
        )

        _, obs_batch = _run_rollout_windows(
            args=args,
            envs=envs,
            obs_batch=obs_batch,
            rngs=rngs,
            training_ids=training_ids,
            opponent_ids=opponent_ids,
            use_training_policy=use_training_policy,
            policy=policy,
            torch_mod=torch_mod,
            device=device,
            progress=progress,
            phase_label="warmup",
            iterations=int(args.warmup_iters),
        )
        metrics, _ = _run_rollout_windows(
            args=args,
            envs=envs,
            obs_batch=obs_batch,
            rngs=rngs,
            training_ids=training_ids,
            opponent_ids=opponent_ids,
            use_training_policy=use_training_policy,
            policy=policy,
            torch_mod=torch_mod,
            device=device,
            progress=progress,
            phase_label="benchmark",
            iterations=int(args.benchmark_iters),
        )
    finally:
        progress.close()
        for env in envs:
            try:
                env.close()
            except Exception:
                pass

    return {
        "device": str(device),
        "kernel_batch_size": int(args.kernel_batch_size),
        "warmup_iters": int(args.warmup_iters),
        "benchmark_iters": int(args.benchmark_iters),
        "horizon": int(args.horizon),
        "training_player_ids": [int(v) for v in training_ids.tolist()],
        "opponent_player_ids": [int(v) for v in opponent_ids.tolist()],
        "flat_obs_dim": int(spec.flat_obs_dim),
        "training_player_count": int(spec.training_player_count),
        "action_dim_per_player": int(spec.action_dim_per_player),
        "total_action_dim": int(spec.total_action_dim),
        "rollout_states_per_sec": float(metrics["states_per_sec"]),
        "mean_window_latency_ms": float(metrics["mean_window_latency_ms"]),
        "obs_prep_time_sec": float(metrics["obs_prep_time_sec"]),
        "training_policy_time_sec": float(metrics["training_policy_time_sec"]),
        "training_random_time_sec": float(metrics["training_random_time_sec"]),
        "opponent_random_time_sec": float(metrics["opponent_random_time_sec"]),
        "assemble_actions_time_sec": float(metrics["assemble_actions_time_sec"]),
        "env_step_time_sec": float(metrics["env_step_time_sec"]),
        "env_reset_time_sec": float(metrics["env_reset_time_sec"]),
        "auto_resets_from_done": int(metrics["auto_resets_from_done"]),
        "episodes_completed_done": int(metrics["episodes_completed_done"]),
        "episodes_completed_truncated": int(metrics["episodes_completed_truncated"]),
    }


def run_phase_a_rollout_torch_benchmark(args) -> dict[str, Any]:
    policy_metric = _run_single_metric(args, use_training_policy=True)
    random_metric = _run_single_metric(args, use_training_policy=False)
    return {
        "script": "benchmarks/sbx_phase_a_rollout_torch.py",
        "training_team": resolve_training_team(args.training_team).name,
        "env_config": {
            key: getattr(args, key)
            for key in (
                "players",
                "court_rows",
                "court_cols",
                "shot_clock",
                "min_shot_clock",
                "pass_mode",
                "use_set_obs",
                "mask_occupied_moves",
                "enable_pass_gating",
                "enable_phi_shaping",
                "illegal_defense_enabled",
                "offensive_three_seconds",
            )
        },
        "compiled_rollout_python_legal_random_minimal": random_metric,
        "compiled_rollout_python_phase_a_policy_minimal": policy_metric,
    }


def main(argv=None):
    args = parse_args(argv)
    result = run_phase_a_rollout_torch_benchmark(args)

    print("Phase A Torch rollout baseline")
    for name in (
        "compiled_rollout_python_legal_random_minimal",
        "compiled_rollout_python_phase_a_policy_minimal",
    ):
        metrics = result[name]
        print(
            f"{name}:"
            f" states_per_sec={metrics['rollout_states_per_sec']:.2f}"
            f" mean_window_latency_ms={metrics['mean_window_latency_ms']:.4f}"
        )

    if args.output_json:
        write_json(args.output_json, result)
        print(f"wrote_json: {args.output_json}")


if __name__ == "__main__":
    main()
