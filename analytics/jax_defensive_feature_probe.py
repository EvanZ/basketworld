#!/usr/bin/env python3
"""Run the offline defensive feature probe on completed JAX MCTS results."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basketworld_jax.eval.defensive_feature_probe import (  # noqa: E402
    ProbeTrainingConfig,
    run_defensive_feature_probe,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train controlled offline attention probes that predict defensive "
            "MCTS actions from current and explicitly relational observations."
        )
    )
    parser.add_argument(
        "--mcts-results",
        required=True,
        help="Completed root_action_comparison.json from the JAX MCTS study.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="New output directory; existing paths are never overwritten.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=None,
        help="Search depth to use; defaults to the source study primary depth.",
    )
    parser.add_argument(
        "--stability-result",
        action="append",
        default=[],
        help=(
            "Optional completed result over the same states at another budget "
            "or seed. Repeat to require agreement across several artifacts."
        ),
    )
    parser.add_argument("--split-seed", type=int, default=17)
    parser.add_argument("--training-seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--token-mlp-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--high-regret-quantile", type=float, default=0.75)
    parser.add_argument(
        "--learning-curve-fractions",
        type=float,
        nargs="+",
        default=[0.25, 0.50, 1.0],
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    config = ProbeTrainingConfig(
        split_seed=int(args.split_seed),
        training_seeds=tuple(int(value) for value in args.training_seeds),
        train_fraction=float(args.train_fraction),
        validation_fraction=float(args.validation_fraction),
        learning_rate=float(args.learning_rate),
        epochs=int(args.epochs),
        patience=int(args.patience),
        embed_dim=int(args.embed_dim),
        token_mlp_dim=int(args.token_mlp_dim),
        num_heads=int(args.num_heads),
        num_layers=int(args.num_layers),
        bootstrap_samples=int(args.bootstrap_samples),
        high_regret_quantile=float(args.high_regret_quantile),
        learning_curve_fractions=tuple(
            float(value) for value in args.learning_curve_fractions
        ),
    )
    result = run_defensive_feature_probe(
        Path(args.mcts_results).expanduser(),
        output_dir=Path(args.output_dir).expanduser(),
        config=config,
        depth=args.depth,
        stability_result_paths=[
            Path(value).expanduser() for value in args.stability_result
        ],
        progress_callback=lambda message: print(message, flush=True),
    )
    payload = result["payload"]
    print(f"Results: {Path(result['results']).resolve()}")
    for variant, variant_result in payload["feature_variants"].items():
        metric = variant_result["metrics"]["test"]["exact_player_action_accuracy"]
        print(
            f"{variant}: test exact action accuracy "
            f"{metric['estimate']:.1%} "
            f"(95% CI {metric['ci95_low']:.1%} to {metric['ci95_high']:.1%})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
