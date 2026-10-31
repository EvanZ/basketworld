#!/usr/bin/env python3
"""Evaluate distinct trained AR and independent JAX policies head to head."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basketworld_jax.eval.policy_crossplay import (  # noqa: E402
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_EPISODES,
    run_policy_crossplay,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run matched direct cross-play between two distinct trained JAX "
            "checkpoints: one autoregressive and one independent. Decoder mode "
            "and ordering are always loaded from each checkpoint."
        )
    )
    parser.add_argument("--autoregressive-checkpoint", required=True)
    parser.add_argument("--independent-checkpoint", required=True)
    parser.add_argument(
        "--output-dir",
        required=True,
        help="New output directory; an existing path is never overwritten.",
    )
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--action-mode",
        choices=("deterministic", "sampled"),
        default="deterministic",
        help="Apply the same action-selection mode to both trained policies.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES
    )
    parser.add_argument(
        "--environment-overrides-json",
        default="",
        help=(
            "Optional JSON object of explicit environment overrides. By default "
            "checkpoint-owned environment semantics are used and must agree."
        ),
    )
    return parser.parse_args(argv)


def _load_environment_overrides(path: str) -> dict[str, Any]:
    if not str(path).strip():
        return {}
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Environment overrides JSON must contain an object.")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    result = run_policy_crossplay(
        autoregressive_checkpoint=args.autoregressive_checkpoint,
        independent_checkpoint=args.independent_checkpoint,
        output_dir=args.output_dir,
        episodes=int(args.episodes),
        seed=int(args.seed),
        action_mode=str(args.action_mode),
        batch_size=int(args.batch_size),
        bootstrap_samples=int(args.bootstrap_samples),
        environment_overrides=_load_environment_overrides(
            args.environment_overrides_json
        ),
        repo_root=REPO_ROOT,
        printer=lambda message: print(message, flush=True),
    )
    summary = result["summary_payload"]
    paired = summary["matched_scenarios"]
    win_score = paired["autoregressive_win_score"]
    margin = paired["autoregressive_margin"]
    if win_score["mean"] is None:
        print(
            "Completed direct cross-play, but no complete four-game matched "
            "scenario was available for paired inference.",
            flush=True,
        )
    else:
        print(
            "Completed direct cross-play: "
            f"AR win score {win_score['mean']:.3f} "
            "(paired 95% CI "
            f"{win_score['ci95_low']:.3f} to {win_score['ci95_high']:.3f}); "
            f"mean margin {margin['mean']:+.3f}.",
            flush=True,
        )
    print(f"Summary: {result['summary']}", flush=True)
    print(f"Games: {result['games']}", flush=True)
    print(f"Manifest: {result['manifest']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
