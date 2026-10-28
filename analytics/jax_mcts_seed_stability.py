#!/usr/bin/env python3
"""Analyze same-state MCTS action stability across search RNG seeds."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basketworld_jax.eval.mcts_seed_stability import (  # noqa: E402
    run_search_seed_stability_study,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure whether completed MCTS searches over identical states "
            "select the same actions when only the search RNG seed changes."
        )
    )
    parser.add_argument(
        "--result",
        action="append",
        required=True,
        help=(
            "Completed root_action_comparison.json. Repeat for each independent "
            "search seed; at least two are required."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="New output directory; existing paths are never overwritten.",
    )
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=17)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    result = run_search_seed_stability_study(
        [Path(value).expanduser() for value in args.result],
        output_dir=Path(args.output_dir).expanduser(),
        depth=args.depth,
        bootstrap_samples=int(args.bootstrap_samples),
        bootstrap_seed=int(args.bootstrap_seed),
    )
    summary = result["payload"]["summary"]
    player = summary["pairwise_player_action_agreement"]
    joint = summary["pairwise_joint_action_agreement"]
    print(f"Results: {Path(result['results']).resolve()}")
    print(
        f"{summary['run_count']} seeds over {summary['state_count']} states: "
        f"per-defender pairwise agreement {player['estimate']:.1%} "
        f"(95% CI {player['ci95_low']:.1%} to {player['ci95_high']:.1%}); "
        f"joint-action agreement {joint['estimate']:.1%} "
        f"(95% CI {joint['ci95_low']:.1%} to {joint['ci95_high']:.1%})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
