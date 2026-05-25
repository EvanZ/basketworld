from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from time import perf_counter_ns

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv

# Support both `python -m benchmarks.policy_bridge_baseline` and
# `python benchmarks/policy_bridge_baseline.py` from the repo root.
if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from basketworld.utils.action_resolution import IllegalActionStrategy
from basketworld.utils.self_play_wrapper import SelfPlayEnvWrapper
from benchmarks.common import (
    Timer,
    benchmark_args_snapshot,
    build_benchmark_parser,
    build_progress,
    resolve_training_team,
    write_json,
)
from benchmarks.jax_kernel import build_policy_benchmark_model
from train.env_factory import make_vector_env, setup_environment


def parse_args(argv=None):
    parser = build_benchmark_parser(
        "Benchmark the current BasketWorld policy + self-play rollout path without PPO updates."
    )
    parser.set_defaults(mode="throughput", runner="subproc_vec")
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=1,
        help="Number of untimed reset/predict/step warmup cycles before the timed benchmark.",
    )
    parser.add_argument(
        "--deterministic-training",
        action="store_true",
        help="Use deterministic training-policy predictions during the benchmark.",
    )
    return parser.parse_args(argv)


def _build_dummy_self_play_vec_env(args, opponent_policy):
    training_team = resolve_training_team(args.training_team)

    def _make_env(env_idx):
        def _thunk():
            base_env = setup_environment(args, training_team, env_idx=env_idx)
            return SelfPlayEnvWrapper(
                base_env,
                opponent_policy=opponent_policy,
                training_strategy=IllegalActionStrategy.SAMPLE_PROB,
                opponent_strategy=IllegalActionStrategy.SAMPLE_PROB,
                deterministic_opponent=bool(getattr(args, "deterministic_opponent", False)),
            )

        return _thunk

    env_fns = [_make_env(env_idx) for env_idx in range(int(args.num_envs))]
    return DummyVecEnv(env_fns)


def build_self_play_vec_env(args, opponent_policy):
    if args.runner == "dummy_vec":
        return _build_dummy_self_play_vec_env(args, opponent_policy)
    if args.runner == "subproc_vec":
        training_team = resolve_training_team(args.training_team)
        return make_vector_env(
            args,
            training_team,
            opponent_policy=opponent_policy,
            num_envs=int(args.num_envs),
            deterministic_opponent=bool(getattr(args, "deterministic_opponent", False)),
        )
    raise SystemExit(
        "benchmarks/policy_bridge_baseline.py only supports --runner dummy_vec or subproc_vec."
    )


def _warmup_policy_rollout(vec_env, training_model, steps: int, deterministic_training: bool):
    if int(steps) <= 0:
        return
    obs = vec_env.reset()
    for _ in range(int(steps)):
        actions, _ = training_model.predict(obs, deterministic=deterministic_training)
        obs, _, _, _ = vec_env.step(actions)


def run_policy_bridge_baseline(args):
    if args.runner == "sequential":
        raise SystemExit(
            "benchmarks/policy_bridge_baseline.py does not support --runner sequential."
        )

    training_model, policy_init_env = build_policy_benchmark_model(args)
    policy_artifact_dir = None
    vec_env = None

    try:
        opponent_policy = training_model
        if args.runner == "subproc_vec":
            policy_artifact_dir = tempfile.TemporaryDirectory(prefix="basketworld_policy_bridge_")
            opponent_path = Path(policy_artifact_dir.name) / "opponent_policy.zip"
            training_model.save(str(opponent_path))
            opponent_policy = str(opponent_path)

        vec_env = build_self_play_vec_env(args, opponent_policy=opponent_policy)
        _warmup_policy_rollout(
            vec_env,
            training_model,
            steps=int(args.warmup_steps),
            deterministic_training=bool(args.deterministic_training),
        )

        num_envs = int(args.num_envs)
        reset_ns_total = 0
        predict_ns_total = 0
        step_ns_total = 0
        explicit_resets = 0
        total_steps = 0
        auto_resets_from_done = 0
        benchmark_start_ns = perf_counter_ns()
        progress = build_progress(
            total=num_envs * int(args.episodes),
            desc=f"{args.runner}:policy_bridge",
            disable=bool(args.no_progress),
            unit="ep",
        )

        try:
            for _ in range(int(args.episodes)):
                with Timer() as timer:
                    obs = vec_env.reset()
                reset_ns_total += timer.elapsed_ns
                explicit_resets += 1

                for _ in range(int(args.horizon)):
                    with Timer() as timer:
                        actions, _ = training_model.predict(
                            obs,
                            deterministic=bool(args.deterministic_training),
                        )
                    predict_ns_total += timer.elapsed_ns

                    with Timer() as timer:
                        obs, _, dones, _ = vec_env.step(actions)
                    step_ns_total += timer.elapsed_ns
                    total_steps += num_envs
                    auto_resets_from_done += int(np.asarray(dones, dtype=np.int8).sum())

                progress.update(num_envs)
                elapsed_sec = max((perf_counter_ns() - benchmark_start_ns) / 1e9, 1e-9)
                progress.set_postfix_str(
                    f"steps={total_steps} sps={total_steps / elapsed_sec:.1f}",
                    refresh=False,
                )
        finally:
            progress.close()

        wall_elapsed_sec = (perf_counter_ns() - benchmark_start_ns) / 1e9
        reset_time_sec = reset_ns_total / 1e9
        predict_time_sec = predict_ns_total / 1e9
        step_time_sec = step_ns_total / 1e9
        bridge_time_sec = predict_time_sec + step_time_sec
        rollout_time_sec = bridge_time_sec + reset_time_sec
        step_call_count = int(args.episodes) * int(args.horizon)
        predict_call_count = step_call_count

        return {
            "script": "benchmarks/policy_bridge_baseline.py",
            "mode": "throughput",
            "runner": str(args.runner),
            "training_team": resolve_training_team(args.training_team).name,
            "episodes_per_env": int(args.episodes),
            "horizon": int(args.horizon),
            "num_envs": num_envs,
            "warmup_steps": int(args.warmup_steps),
            "env_config": benchmark_args_snapshot(args),
            "total_steps": int(total_steps),
            "total_resets": int(explicit_resets * num_envs),
            "explicit_reset_calls": int(explicit_resets),
            "step_call_count": int(step_call_count),
            "predict_call_count": int(predict_call_count),
            "auto_resets_from_done": int(auto_resets_from_done),
            "policy_predict_time_sec": float(predict_time_sec),
            "env_step_time_sec": float(step_time_sec),
            "env_reset_time_sec": float(reset_time_sec),
            "bridge_time_sec": float(bridge_time_sec),
            "rollout_time_sec": float(rollout_time_sec),
            "wall_elapsed_sec": float(wall_elapsed_sec),
            "policy_predict_steps_per_sec": (
                float(total_steps) / predict_time_sec
            )
            if predict_time_sec > 0.0
            else 0.0,
            "env_step_steps_per_sec": (
                float(total_steps) / step_time_sec
            )
            if step_time_sec > 0.0
            else 0.0,
            "bridge_steps_per_sec": (
                float(total_steps) / bridge_time_sec
            )
            if bridge_time_sec > 0.0
            else 0.0,
            "rollout_steps_per_sec": (
                float(total_steps) / rollout_time_sec
            )
            if rollout_time_sec > 0.0
            else 0.0,
            "mean_predict_call_latency_ms": (
                predict_ns_total / step_call_count / 1e6
            )
            if step_call_count
            else 0.0,
            "mean_step_call_latency_ms": (
                step_ns_total / step_call_count / 1e6
            )
            if step_call_count
            else 0.0,
            "mean_reset_call_latency_ms": (
                reset_ns_total / explicit_resets / 1e6
            )
            if explicit_resets
            else 0.0,
            "mean_policy_predict_env_latency_ms": (
                predict_ns_total / total_steps / 1e6
            )
            if total_steps
            else 0.0,
            "mean_env_step_latency_ms": (
                step_ns_total / total_steps / 1e6
            )
            if total_steps
            else 0.0,
            "mean_env_reset_latency_ms": (
                reset_ns_total / (explicit_resets * num_envs) / 1e6
            )
            if explicit_resets and num_envs
            else 0.0,
            "runner_semantics": (
                "training policy predict in main process plus current SelfPlayEnvWrapper step path"
            ),
        }
    finally:
        try:
            if vec_env is not None:
                vec_env.close()
        except Exception:
            pass
        try:
            policy_init_env.close()
        except Exception:
            pass
        if policy_artifact_dir is not None:
            policy_artifact_dir.cleanup()


def print_summary(result):
    print("Baseline policy/self-play benchmark")
    print(f"mode: {result['mode']}")
    print(f"runner: {result['runner']}")
    print(f"training_team: {result['training_team']}")
    print(f"num_envs: {result['num_envs']}")
    print(f"episodes_per_env: {result['episodes_per_env']}")
    print(f"horizon: {result['horizon']}")
    print(f"total_steps: {result['total_steps']}")
    print(f"total_resets: {result['total_resets']}")
    print(f"explicit_reset_calls: {result['explicit_reset_calls']}")
    print(f"policy_predict_steps_per_sec: {result['policy_predict_steps_per_sec']:.2f}")
    print(f"env_step_steps_per_sec: {result['env_step_steps_per_sec']:.2f}")
    print(f"bridge_steps_per_sec: {result['bridge_steps_per_sec']:.2f}")
    print(f"rollout_steps_per_sec: {result['rollout_steps_per_sec']:.2f}")
    print(f"mean_predict_call_latency_ms: {result['mean_predict_call_latency_ms']:.4f}")
    print(f"mean_step_call_latency_ms: {result['mean_step_call_latency_ms']:.4f}")
    print(f"mean_reset_call_latency_ms: {result['mean_reset_call_latency_ms']:.4f}")
    print(f"mean_policy_predict_env_latency_ms: {result['mean_policy_predict_env_latency_ms']:.4f}")
    print(f"mean_env_step_latency_ms: {result['mean_env_step_latency_ms']:.4f}")
    print(f"mean_env_reset_latency_ms: {result['mean_env_reset_latency_ms']:.4f}")
    print(f"auto_resets_from_done: {result['auto_resets_from_done']}")


def main(argv=None):
    args = parse_args(argv)
    result = run_policy_bridge_baseline(args)
    print_summary(result)
    if args.output_json:
        write_json(args.output_json, result)
        print(f"wrote_json: {Path(args.output_json).resolve()}")


if __name__ == "__main__":
    main()
