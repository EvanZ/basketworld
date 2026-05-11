#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from time import perf_counter
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basketworld_jax.train.main import parse_args as parse_train_args
from basketworld_jax.train.main import run_training_loop


DEFAULT_CONFIGS = ("1024x32", "2048x32", "2048x64", "4096x64")


def _parse_rollout_config(value: str) -> tuple[int, int]:
    token = str(value).lower().replace(":", "x").replace("*", "x")
    parts = [part.strip() for part in token.split("x") if part.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"Expected rollout config as BATCHxHORIZON, got {value!r}."
        )
    try:
        batch_size = int(parts[0])
        horizon = int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Expected integer BATCHxHORIZON, got {value!r}."
        ) from exc
    if batch_size <= 0 or horizon <= 0:
        raise argparse.ArgumentTypeError("Batch size and horizon must be positive.")
    return batch_size, horizon


def _base_train_args(*, include_sample_dumps: bool) -> list[str]:
    start_template_path = REPO_ROOT / "configs" / "start_templates_v2.json"
    args = [
        "--run-train-loop",
        "--no-progress",
        "--pass-mode",
        "pointer_targeted",
        "--num-updates",
        "1",
        "--log-every-updates",
        "1",
        "--eval-every-updates",
        "0",
        "--eval-horizon",
        "32",
        "--max-eval-dumps",
        "0",
        "--checkpoint-every-updates",
        "0",
        "--ent-coef-start",
        "1e-1",
        "--ent-coef-end",
        "1e-3",
        "--ent-schedule",
        "exp",
        "--grouped-opponent-sampling",
        "--opponent-group-count",
        "8",
        "--policy-update-epochs",
        "1",
        "--ppo-minibatches",
        "16",
        "--layup-pct",
        "0.60",
        "--three-pt-pct",
        "0.40",
        "--dunk-pct",
        "0.6",
        "--layup-std",
        "0.05",
        "--three-pt-std",
        "0.05",
        "--dunk-std",
        "0.3",
        "--three-pt-extra-hex-decay",
        "0.05",
        "--start-template-enabled",
        "true",
        "--start-template-library",
        str(start_template_path),
        "--start-template-prob",
        "1.0",
        "--start-template-jitter-scale",
        "1.0",
        "--start-template-mirror-prob",
        "0.5",
        "--start-template-strict",
        "true",
        "--illegal-defense-enabled",
        "true",
        "--offensive-three-seconds",
        "true",
        "--three-second-lane-width",
        "1",
        "--three-second-lane-height",
        "3",
        "--three-second-max-steps",
        "3",
        "--violation-reward",
        "2.0",
        "--learning-rate",
        "5e-4",
        "--policy-model",
        "attention",
        "--action-head-mode",
        "pointer_targeted",
        "--attention-embed-dim",
        "64",
        "--attention-num-heads",
        "4",
        "--attention-token-mlp-dim",
        "64",
        "--attention-cls-tokens",
        "2",
        "--attention-pi-head-hidden-dims",
        "64",
        "64",
        "64",
        "64",
        "--attention-vf-head-hidden-dims",
        "64",
        "64",
        "64",
        "64",
        "--attention-head-activation",
        "relu",
        "--intent-embedding-enabled",
        "--intent-embedding-dim",
        "16",
        "--enable-intent-learning",
        "true",
        "--enable-defense-intent-learning",
        "false",
        "--defense-intent-null-prob",
        "1.0",
        "--num-intents",
        "8",
        "--intent-commitment-steps",
        "8",
        "--intent-null-prob",
        "0.0",
        "--intent-visible-to-defense-prob",
        "0.0",
        "--task-reward-scale-start",
        "0.1",
        "--task-reward-scale-end",
        "1.0",
        "--task-reward-scale-warmup-updates",
        "2",
        "--task-reward-scale-ramp-updates",
        "5",
        "--intent-selector-enabled",
        "true",
        "--intent-selector-hidden-dim",
        "64",
        "--intent-selector-alpha-start",
        "0",
        "--intent-selector-alpha-end",
        "0.35",
        "--intent-selector-alpha-warmup-updates",
        "12",
        "--intent-selector-alpha-ramp-updates",
        "12",
        "--intent-selector-eps-start",
        "0.35",
        "--intent-selector-eps-end",
        "0.15",
        "--intent-selector-eps-warmup-updates",
        "12",
        "--intent-selector-eps-ramp-updates",
        "18",
        "--intent-selector-value-coef",
        "0.5",
        "--intent-selector-entropy-coef",
        "0.03",
        "--intent-selector-usage-reg-coef",
        "0.05",
        "--intent-selector-train-every-rollouts",
        "1",
        "--intent-selector-max-samples-per-update",
        "1024",
        "--intent-selector-multiselect-enabled",
        "true",
        "--intent-selector-min-play-steps",
        "4",
        "--intent-diversity-enabled",
        "true",
        "--intent-diversity-beta-target",
        "0.1",
        "--intent-diversity-warmup-updates",
        "0",
        "--intent-diversity-ramp-updates",
        "5",
        "--intent-diversity-clip",
        "2.0",
        "--intent-disc-encoder-type",
        "set_step",
        "--intent-disc-hidden-dim",
        "128",
        "--intent-disc-dropout",
        "0.1",
        "--intent-disc-batch-size",
        "512",
        "--intent-disc-updates-per-rollout",
        "1",
        "--intent-disc-eval-holdout-fraction",
        "0.10",
        "--intent-disc-current-policy-only",
        "true",
        "--intent-disc-include-shot-clock",
        "false",
        "--intent-disc-include-pressure-exposure",
        "false",
        "--disc-eval-batch-output",
        "true" if include_sample_dumps else "false",
        "--intent-sample-dump-size",
        "4096",
    ]
    return args


def _aggregate_metrics(
    history: list[dict[str, Any]],
    *,
    warmup_updates: int,
    steps_per_update: int,
) -> dict[str, Any]:
    measured = [
        item
        for item in history
        if int(item.get("update_index", 0)) > int(warmup_updates)
    ]
    if not measured:
        raise RuntimeError("No measured train-history rows were produced.")
    measured_steps = int(steps_per_update) * len(measured)
    compiled_sec = sum(float(item["end_to_end_elapsed_sec"]) for item in measured)
    outer_sec = sum(
        float(item.get("train_loop_elapsed_sec", item["end_to_end_elapsed_sec"]))
        for item in measured
    )
    return {
        "measured_updates": len(measured),
        "measured_steps": measured_steps,
        "compiled_rollout_update_steps_per_sec": measured_steps / max(compiled_sec, 1.0e-12),
        "outer_train_loop_steps_per_sec": measured_steps / max(outer_sec, 1.0e-12),
        "compiled_rollout_update_elapsed_sec": compiled_sec,
        "outer_train_loop_elapsed_sec": outer_sec,
        "mean_rollout_time_pct": statistics.fmean(
            float(item.get("rollout_time_pct", 0.0)) for item in measured
        ),
        "mean_ppo_update_time_pct": statistics.fmean(
            float(item.get("ppo_update_time_pct", 0.0)) for item in measured
        ),
        "mean_train_loop_overhead_sec": statistics.fmean(
            float(item.get("train_loop_overhead_sec", 0.0)) for item in measured
        ),
        "mean_approx_kl": statistics.fmean(
            float(item.get("approx_kl", 0.0)) for item in measured
        ),
        "mean_clip_fraction": statistics.fmean(
            float(item.get("clip_fraction", 0.0)) for item in measured
        ),
        "mean_entropy_bonus": statistics.fmean(
            float(item.get("entropy_bonus", 0.0)) for item in measured
        ),
    }


def _run_one(
    *,
    batch_size: int,
    horizon: int,
    warmup_updates: int,
    measure_updates: int,
    include_sample_dumps: bool,
    extra_train_args: list[str],
) -> dict[str, Any]:
    total_updates = int(warmup_updates) + int(measure_updates)
    train_args = _base_train_args(include_sample_dumps=include_sample_dumps)
    train_args.extend(
        [
            "--kernel-batch-size",
            str(int(batch_size)),
            "--rollout-horizon",
            str(int(horizon)),
            "--num-updates",
            str(total_updates),
        ]
    )
    train_args.extend(extra_train_args)
    parsed = parse_train_args(train_args)
    steps_per_update = int(batch_size) * int(horizon) * 2
    start = perf_counter()
    result = run_training_loop(parsed)
    wall_sec = perf_counter() - start
    metrics = _aggregate_metrics(
        list(result["train_history"]),
        warmup_updates=int(warmup_updates),
        steps_per_update=steps_per_update,
    )
    return {
        "config": f"{int(batch_size)}x{int(horizon)}",
        "kernel_batch_size": int(batch_size),
        "rollout_horizon": int(horizon),
        "steps_per_update": steps_per_update,
        "warmup_updates": int(warmup_updates),
        "measure_updates": int(measure_updates),
        "wall_elapsed_sec": wall_sec,
        **metrics,
    }


def _print_table(results: list[dict[str, Any]]) -> None:
    print(
        "rank config     steps/update  outer_sps  compiled_sps  rollout%  update%  overhead_ms"
    )
    for rank, item in enumerate(
        sorted(results, key=lambda row: row["outer_train_loop_steps_per_sec"], reverse=True),
        start=1,
    ):
        print(
            f"{rank:>4} {item['config']:>9}"
            f" {int(item['steps_per_update']):>12}"
            f" {item['outer_train_loop_steps_per_sec']:>10.0f}"
            f" {item['compiled_rollout_update_steps_per_sec']:>13.0f}"
            f" {item['mean_rollout_time_pct']:>8.1f}"
            f" {item['mean_ppo_update_time_pct']:>8.1f}"
            f" {item['mean_train_loop_overhead_sec'] * 1000.0:>11.1f}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Sweep JAX train-loop rollout sizes and rank end-to-end speed."
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        type=_parse_rollout_config,
        default=[_parse_rollout_config(item) for item in DEFAULT_CONFIGS],
        help="Rollout configs as BATCHxHORIZON, e.g. 1024x32 2048x64.",
    )
    parser.add_argument("--warmup-updates", type=int, default=2)
    parser.add_argument("--measure-updates", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--include-sample-dumps",
        action="store_true",
        help="Include discriminator sample dump construction in the benchmark.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "results" / "jax_e2e_speed_sweep.json",
    )
    args, extra_train_args = parser.parse_known_args(argv)
    if extra_train_args and extra_train_args[0] == "--":
        extra_train_args = extra_train_args[1:]

    results = []
    for batch_size, horizon in args.configs:
        repeat_results = []
        for repeat_idx in range(int(args.repeats)):
            print(
                f"\n[jax_e2e_speed_sweep] {batch_size}x{horizon}"
                f" repeat={repeat_idx + 1}/{int(args.repeats)}",
                flush=True,
            )
            repeat_results.append(
                _run_one(
                    batch_size=batch_size,
                    horizon=horizon,
                    warmup_updates=int(args.warmup_updates),
                    measure_updates=int(args.measure_updates),
                    include_sample_dumps=bool(args.include_sample_dumps),
                    extra_train_args=list(extra_train_args),
                )
            )
        primary = sorted(
            repeat_results,
            key=lambda row: row["outer_train_loop_steps_per_sec"],
        )[len(repeat_results) // 2]
        primary = {
            **primary,
            "repeat_results": repeat_results,
            "median_outer_train_loop_steps_per_sec": primary["outer_train_loop_steps_per_sec"],
        }
        results.append(primary)

    payload = {
        "script": "benchmarks/jax_e2e_speed_sweep.py",
        "warmup_updates": int(args.warmup_updates),
        "measure_updates": int(args.measure_updates),
        "repeats": int(args.repeats),
        "include_sample_dumps": bool(args.include_sample_dumps),
        "extra_train_args": list(extra_train_args),
        "results": sorted(
            results,
            key=lambda row: row["outer_train_loop_steps_per_sec"],
            reverse=True,
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _print_table(results)
    print(f"\nwrote_json: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
