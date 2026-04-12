from __future__ import annotations

import sys
import warnings
from pathlib import Path
from time import perf_counter_ns

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

# Support both `python -m benchmarks.rollout_baseline` and
# `python benchmarks/rollout_baseline.py` from the repo root.
if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent.parent))

warnings.filterwarnings(
    "ignore",
    message=r".*env\.reset_profile_stats to get variables from other wrappers is deprecated.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*env\.get_profile_stats to get variables from other wrappers is deprecated.*",
    category=UserWarning,
)

from benchmarks.common import (
    Timer,
    aggregate_profile_stats,
    benchmark_args_snapshot,
    build_progress,
    build_benchmark_parser,
    choose_legal_actions,
    extract_action_mask,
    generate_action_ranks,
    rank_profile_sections,
    resolve_training_team,
    write_json,
)
from train.env_factory import setup_environment


def parse_args(argv=None):
    parser = build_benchmark_parser(
        "Benchmark current BasketWorld rollout performance without policy inference."
    )
    return parser.parse_args(argv)


def configure_args_for_mode(args):
    args.enable_env_profiling = args.mode == "hotspot"
    if args.mode == "hotspot" and float(args.profiling_sample_rate) <= 0.0:
        raise SystemExit("--profiling-sample-rate must be > 0 in hotspot mode.")
    return args


def build_envs(args):
    training_team = resolve_training_team(args.training_team)
    return [setup_environment(args, training_team) for _ in range(int(args.num_envs))]


def build_vec_env(args):
    training_team = resolve_training_team(args.training_team)

    def _make_env(env_idx):
        def _thunk():
            return setup_environment(args, training_team, env_idx=env_idx)

        return _thunk

    env_fns = [_make_env(env_idx) for env_idx in range(int(args.num_envs))]
    if args.runner == "dummy_vec":
        return DummyVecEnv(env_fns)
    if args.runner == "subproc_vec":
        return SubprocVecEnv(env_fns, start_method="spawn")
    raise ValueError(f"Unsupported vector runner: {args.runner}")


def run_sequential_baseline(args):
    envs = build_envs(args)
    base_envs = [env.unwrapped for env in envs]

    if not envs:
        raise SystemExit("No environments were created. Check --num-envs.")

    for base_env in base_envs:
        try:
            base_env.reset_profile_stats()
        except Exception:
            pass

    n_players = int(base_envs[0].n_players)
    master_rng = np.random.default_rng(int(args.seed))
    pregenerated_ranks = None
    if args.action_mode == "pregenerated_legal":
        pregenerated_ranks = generate_action_ranks(
            rng=master_rng,
            num_envs=int(args.num_envs),
            episodes=int(args.episodes),
            horizon=int(args.horizon),
            n_players=n_players,
        )

    reset_ns_total = 0
    step_ns_total = 0
    total_resets = 0
    total_steps = 0
    total_done = 0
    total_truncated = 0
    total_horizon_capped = 0
    benchmark_start_ns = perf_counter_ns()
    progress = build_progress(
        total=int(args.num_envs) * int(args.episodes),
        desc=f"{args.runner}:{args.mode}",
        disable=bool(args.no_progress),
        unit="ep",
    )

    try:
        for env_idx, env in enumerate(envs):
            action_rng = np.random.default_rng(int(args.seed) + 10_000 * (env_idx + 1))

            for episode_idx in range(int(args.episodes)):
                reset_seed = int(args.seed) + (env_idx * 1_000_003) + episode_idx
                with Timer() as timer:
                    obs, _ = env.reset(seed=reset_seed)
                reset_ns_total += timer.elapsed_ns
                total_resets += 1
                episode_finished = False

                for step_idx in range(int(args.horizon)):
                    action_mask = extract_action_mask(obs)
                    action_ranks = None
                    if pregenerated_ranks is not None:
                        action_ranks = pregenerated_ranks[env_idx, episode_idx, step_idx]
                    actions = choose_legal_actions(
                        action_mask=action_mask,
                        rng=action_rng,
                        action_mode=args.action_mode,
                        action_ranks=action_ranks,
                    )

                    with Timer() as timer:
                        obs, _, done, truncated, _ = env.step(actions)
                    step_ns_total += timer.elapsed_ns
                    total_steps += 1

                    if done:
                        total_done += 1
                        episode_finished = True
                        break
                    if truncated:
                        total_truncated += 1
                        episode_finished = True
                        break

                if not episode_finished:
                    total_horizon_capped += 1

                progress.update(1)
                elapsed_sec = max((perf_counter_ns() - benchmark_start_ns) / 1e9, 1e-9)
                progress.set_postfix_str(
                    f"steps={total_steps} sps={total_steps / elapsed_sec:.1f}",
                    refresh=False,
                )
    finally:
        progress.close()

    wall_elapsed_sec = (perf_counter_ns() - benchmark_start_ns) / 1e9
    step_time_sec = step_ns_total / 1e9
    reset_time_sec = reset_ns_total / 1e9
    rollout_time_sec = step_time_sec + reset_time_sec

    metrics = {
        "total_steps": int(total_steps),
        "total_resets": int(total_resets),
        "episodes_completed_done": int(total_done),
        "episodes_completed_truncated": int(total_truncated),
        "episodes_capped_by_horizon": int(total_horizon_capped),
        "mean_episode_steps": (float(total_steps) / float(total_resets)) if total_resets else 0.0,
        "step_call_count": int(total_steps),
        "reset_call_count": int(total_resets),
        "env_step_time_sec": float(step_time_sec),
        "env_reset_time_sec": float(reset_time_sec),
        "rollout_time_sec": float(rollout_time_sec),
        "wall_elapsed_sec": float(wall_elapsed_sec),
        "steps_per_sec": (float(total_steps) / step_time_sec) if step_time_sec > 0.0 else 0.0,
        "resets_per_sec": (float(total_resets) / reset_time_sec) if reset_time_sec > 0.0 else 0.0,
        "rollout_steps_per_sec": (float(total_steps) / rollout_time_sec) if rollout_time_sec > 0.0 else 0.0,
        "mean_step_latency_ms": (step_ns_total / total_steps / 1e6) if total_steps else 0.0,
        "mean_reset_latency_ms": (reset_ns_total / total_resets / 1e6) if total_resets else 0.0,
        "mean_step_call_latency_ms": (step_ns_total / total_steps / 1e6) if total_steps else 0.0,
        "mean_env_step_latency_ms": (step_ns_total / total_steps / 1e6) if total_steps else 0.0,
        "mean_reset_call_latency_ms": (reset_ns_total / total_resets / 1e6) if total_resets else 0.0,
        "mean_env_reset_latency_ms": (reset_ns_total / total_resets / 1e6) if total_resets else 0.0,
        "auto_resets_from_done": 0,
        "runner_semantics": "exact per-env episodes with explicit reset calls",
    }

    profile_raw = {}
    profile_by_total = []
    profile_by_avg = []
    if args.mode == "hotspot":
        profile_raw = aggregate_profile_stats(
            base_env.get_profile_stats() for base_env in base_envs
        )
        profile_by_total = rank_profile_sections(
            profile_raw, sort_key="total_ms", limit=int(args.profile_top_k)
        )
        profile_by_avg = rank_profile_sections(
            profile_raw, sort_key="avg_us", limit=int(args.profile_top_k)
        )

    for env in envs:
        try:
            env.close()
        except Exception:
            pass

    return metrics, profile_raw, profile_by_total, profile_by_avg


def run_vecenv_baseline(args):
    vec_env = build_vec_env(args)
    num_envs = int(args.num_envs)
    n_players = int(args.players) * 2
    action_rngs = [
        np.random.default_rng(int(args.seed) + 10_000 * (env_idx + 1))
        for env_idx in range(num_envs)
    ]
    master_rng = np.random.default_rng(int(args.seed))
    pregenerated_ranks = None
    if args.action_mode == "pregenerated_legal":
        pregenerated_ranks = generate_action_ranks(
            rng=master_rng,
            num_envs=num_envs,
            episodes=int(args.episodes),
            horizon=int(args.horizon),
            n_players=n_players,
        )

    try:
        vec_env.env_method("reset_profile_stats")
    except Exception:
        pass

    reset_ns_total = 0
    step_ns_total = 0
    explicit_resets = 0
    total_steps = 0
    total_done = 0
    total_truncated = 0
    total_rollout_windows = num_envs * int(args.episodes)
    auto_resets_from_done = 0
    step_call_count = int(args.episodes) * int(args.horizon)
    reset_call_count = int(args.episodes)
    benchmark_start_ns = perf_counter_ns()
    progress = build_progress(
        total=num_envs * int(args.episodes),
        desc=f"{args.runner}:{args.mode}",
        disable=bool(args.no_progress),
        unit="ep",
    )

    try:
        for episode_idx in range(int(args.episodes)):
            base_seed = int(args.seed) + (episode_idx * 1_000_003)
            vec_env.seed(base_seed)
            with Timer() as timer:
                obs = vec_env.reset()
            reset_ns_total += timer.elapsed_ns
            explicit_resets += num_envs

            for step_idx in range(int(args.horizon)):
                action_masks = extract_action_mask(obs)
                actions = np.zeros((num_envs, n_players), dtype=np.int64)
                for env_idx in range(num_envs):
                    action_ranks = None
                    if pregenerated_ranks is not None:
                        action_ranks = pregenerated_ranks[env_idx, episode_idx, step_idx]
                    actions[env_idx] = choose_legal_actions(
                        action_mask=action_masks[env_idx],
                        rng=action_rngs[env_idx],
                        action_mode=args.action_mode,
                        action_ranks=action_ranks,
                    )

                with Timer() as timer:
                    obs, _, dones, infos = vec_env.step(actions)
                step_ns_total += timer.elapsed_ns
                total_steps += num_envs

                done_count = int(np.sum(dones))
                total_done += done_count
                auto_resets_from_done += done_count
                for env_idx, done in enumerate(dones):
                    if not done:
                        continue
                    if bool(infos[env_idx].get("TimeLimit.truncated", False)):
                        total_truncated += 1

            progress.update(num_envs)
            elapsed_sec = max((perf_counter_ns() - benchmark_start_ns) / 1e9, 1e-9)
            progress.set_postfix_str(
                f"steps={total_steps} sps={total_steps / elapsed_sec:.1f}",
                refresh=False,
            )
    finally:
        progress.close()

    wall_elapsed_sec = (perf_counter_ns() - benchmark_start_ns) / 1e9
    step_time_sec = step_ns_total / 1e9
    reset_time_sec = reset_ns_total / 1e9
    rollout_time_sec = step_time_sec + reset_time_sec

    metrics = {
        "total_steps": int(total_steps),
        "total_resets": int(explicit_resets),
        "explicit_resets": int(explicit_resets),
        "rollout_windows": int(total_rollout_windows),
        "auto_resets_from_done": int(auto_resets_from_done),
        "episodes_completed_done": int(total_done),
        "episodes_completed_truncated": int(total_truncated),
        "episodes_capped_by_horizon": 0,
        "mean_episode_steps": 0.0,
        "step_call_count": int(step_call_count),
        "reset_call_count": int(reset_call_count),
        "env_step_time_sec": float(step_time_sec),
        "env_reset_time_sec": float(reset_time_sec),
        "rollout_time_sec": float(rollout_time_sec),
        "wall_elapsed_sec": float(wall_elapsed_sec),
        "steps_per_sec": (float(total_steps) / step_time_sec) if step_time_sec > 0.0 else 0.0,
        "resets_per_sec": (float(explicit_resets) / reset_time_sec) if reset_time_sec > 0.0 else 0.0,
        "rollout_steps_per_sec": (float(total_steps) / rollout_time_sec) if rollout_time_sec > 0.0 else 0.0,
        "mean_step_latency_ms": (step_ns_total / total_steps / 1e6) if total_steps else 0.0,
        "mean_reset_latency_ms": (reset_ns_total / explicit_resets / 1e6) if explicit_resets else 0.0,
        "mean_step_call_latency_ms": (step_ns_total / step_call_count / 1e6)
        if step_call_count > 0
        else 0.0,
        "mean_env_step_latency_ms": (step_ns_total / total_steps / 1e6) if total_steps else 0.0,
        "mean_reset_call_latency_ms": (reset_ns_total / reset_call_count / 1e6)
        if reset_call_count > 0
        else 0.0,
        "mean_env_reset_latency_ms": (reset_ns_total / explicit_resets / 1e6)
        if explicit_resets > 0
        else 0.0,
        "runner_semantics": (
            "training-style vectorized rollout windows: explicit vec_env.reset() per outer "
            "window, auto-reset on done inside step timing"
        ),
    }

    profile_raw = {}
    profile_by_total = []
    profile_by_avg = []
    if args.mode == "hotspot":
        profile_raw = aggregate_profile_stats(vec_env.env_method("get_profile_stats"))
        profile_by_total = rank_profile_sections(
            profile_raw, sort_key="total_ms", limit=int(args.profile_top_k)
        )
        profile_by_avg = rank_profile_sections(
            profile_raw, sort_key="avg_us", limit=int(args.profile_top_k)
        )

    try:
        vec_env.close()
    except Exception:
        pass

    return metrics, profile_raw, profile_by_total, profile_by_avg


def run_baseline(args):
    args = configure_args_for_mode(args)
    if args.runner == "sequential":
        metrics, profile_raw, profile_by_total, profile_by_avg = run_sequential_baseline(args)
    else:
        metrics, profile_raw, profile_by_total, profile_by_avg = run_vecenv_baseline(args)

    result = {
        "script": "benchmarks/rollout_baseline.py",
        "mode": args.mode,
        "runner": args.runner,
        "action_mode": args.action_mode,
        "training_team": resolve_training_team(args.training_team).name,
        "seed": int(args.seed),
        "episodes_per_env": int(args.episodes),
        "num_envs": int(args.num_envs),
        "horizon": int(args.horizon),
        "profiling_sample_rate": float(args.profiling_sample_rate),
        "benchmark_args": {
            "mode": args.mode,
            "runner": args.runner,
            "action_mode": args.action_mode,
            "training_team": args.training_team,
            "seed": int(args.seed),
            "profile_top_k": int(args.profile_top_k),
            "output_json": args.output_json,
        },
        "env_config": benchmark_args_snapshot(args),
        "metrics": metrics,
        "profile": {
            "raw": profile_raw,
            "top_by_total_ms": profile_by_total,
            "top_by_avg_us": profile_by_avg,
        },
    }

    return result


def print_summary(result):
    metrics = result["metrics"]
    print("Baseline rollout benchmark")
    print(f"mode: {result['mode']}")
    print(f"runner: {result['runner']}")
    print(f"training_team: {result['training_team']}")
    print(f"action_mode: {result['action_mode']}")
    print(f"num_envs: {result['num_envs']}")
    print(f"episodes_per_env: {result['episodes_per_env']}")
    print(f"horizon: {result['horizon']}")
    print(f"total_steps: {metrics['total_steps']}")
    print(f"total_resets: {metrics['total_resets']}")
    print(f"steps_per_sec: {metrics['steps_per_sec']:.2f}")
    print(f"rollout_steps_per_sec: {metrics['rollout_steps_per_sec']:.2f}")
    print(f"resets_per_sec: {metrics['resets_per_sec']:.2f}")
    print(f"mean_env_step_latency_ms: {metrics['mean_env_step_latency_ms']:.4f}")
    print(f"mean_env_reset_latency_ms: {metrics['mean_env_reset_latency_ms']:.4f}")
    if result["runner"] != "sequential":
        print(f"mean_step_call_latency_ms: {metrics['mean_step_call_latency_ms']:.4f}")
        print(f"mean_reset_call_latency_ms: {metrics['mean_reset_call_latency_ms']:.4f}")
    if metrics.get("auto_resets_from_done", 0):
        print(f"auto_resets_from_done: {metrics['auto_resets_from_done']}")

    if result["mode"] == "hotspot":
        print("top_profile_sections_by_total_ms:")
        for item in result["profile"]["top_by_total_ms"]:
            print(
                "  "
                f"{item['section']}: total_ms={item['total_ms']:.3f}, "
                f"avg_us={item['avg_us']:.3f}, calls={item['calls']:.0f}"
            )


def main(argv=None):
    args = parse_args(argv)
    result = run_baseline(args)
    print_summary(result)
    if args.output_json:
        write_json(args.output_json, result)
        print(f"wrote_json: {args.output_json}")


if __name__ == "__main__":
    main()
