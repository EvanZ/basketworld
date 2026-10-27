#!/usr/bin/env python3
"""Run the standalone JAX MCTS policy-regret evaluator from two checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basketworld_jax.eval.mcts_regret import (  # noqa: E402
    DEFAULT_CALIBRATION_BUDGETS,
    DEFAULT_SEARCH_DEPTH,
    DEFAULT_SEARCH_DEPTHS,
    DEFAULT_SIMULATIONS_PER_ACTION,
    MctsRegretConfig,
)
from basketworld_jax.eval.mcts_regret_jax import (  # noqa: E402
    load_jax_counterfactual_runtime,
)
from basketworld_jax.eval.mcts_regret_study import (  # noqa: E402
    run_budget_calibration_study,
    run_root_only_study,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample replayable JAX policy states and compare the policy root "
            "action with possession-value MCTS under a fixed opponent policy."
        )
    )
    parser.add_argument("--evaluated-checkpoint", required=True)
    parser.add_argument("--opponent-checkpoint", required=True)
    parser.add_argument(
        "--output-dir",
        required=True,
        help="New output directory; existing paths are never overwritten.",
    )
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--planning-mode",
        choices=("root-only", "root-and-full-possession"),
        default="root-and-full-possession",
        help=(
            "Whether to stop after root-action comparison or also replan with "
            "primary-depth MCTS through possession end."
        ),
    )
    parser.add_argument(
        "--evaluated-action-mode",
        choices=("deterministic", "sampled"),
        default="deterministic",
    )
    parser.add_argument(
        "--opponent-action-mode",
        choices=("deterministic", "sampled"),
        default="deterministic",
    )
    parser.add_argument(
        "--search-depths",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEARCH_DEPTHS),
    )
    parser.add_argument(
        "--primary-search-depth",
        type=int,
        default=DEFAULT_SEARCH_DEPTH,
    )
    parser.add_argument(
        "--simulations-per-action",
        type=int,
        default=DEFAULT_SIMULATIONS_PER_ACTION,
        help="MCTS simulations used for each planner action selection.",
    )
    parser.add_argument(
        "--paired-rollouts",
        type=int,
        default=128,
        help="Matched rollout seeds used to estimate policy/planner regret.",
    )
    parser.add_argument("--max-joint-actions-per-node", type=int, default=64)
    parser.add_argument("--exploration-c", type=float, default=1.4)
    parser.add_argument("--progressive-widening-c", type=float, default=2.0)
    parser.add_argument("--progressive-widening-alpha", type=float, default=0.5)
    parser.add_argument("--discount", type=float, default=1.0)
    parser.add_argument("--max-rollout-steps", type=int, default=128)
    parser.add_argument("--max-episode-steps", type=int, default=10_000)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument(
        "--sample-metadata",
        default="",
        help=(
            "Optional sampled_states.json to replay instead of sampling new "
            "episodes. Checkpoint and environment provenance must match."
        ),
    )
    parser.add_argument(
        "--evaluated-role",
        choices=("offense", "defense"),
        default=None,
        help="Optionally analyze only sampled offense or defense decisions.",
    )
    parser.add_argument(
        "--calibration",
        action="store_true",
        help="Run one shared-state search-budget convergence study.",
    )
    parser.add_argument("--calibration-states", type=int, default=50)
    parser.add_argument(
        "--calibration-budgets",
        type=int,
        nargs="+",
        default=list(DEFAULT_CALIBRATION_BUDGETS),
    )
    parser.add_argument(
        "--environment-overrides-json",
        default="",
        help=(
            "Optional JSON object of explicit environment overrides. By default "
            "the evaluator uses checkpoint-owned environment semantics."
        ),
    )
    parser.add_argument(
        "--print-calibration-command",
        action="store_true",
        help=(
            "Print the agreed 50-state budget-sweep commands and exit without "
            "loading checkpoints."
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


def _print_calibration_commands(args: argparse.Namespace) -> None:
    base = (
        "python analytics/jax_mcts_policy_regret.py "
        f"--evaluated-checkpoint {args.evaluated_checkpoint} "
        f"--opponent-checkpoint {args.opponent_checkpoint} "
        f"--episodes 50 --seed {args.seed} "
        f"--evaluated-action-mode {args.evaluated_action_mode} "
        f"--opponent-action-mode {args.opponent_action_mode} "
        "--planning-mode root-only "
    )
    for budget in DEFAULT_CALIBRATION_BUDGETS:
        destination = Path(args.output_dir) / f"budget-{budget}"
        print(f"{base}--simulations-per-action {budget} " f"--output-dir {destination}")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.print_calibration_command:
        _print_calibration_commands(args)
        return 0
    protocol = MctsRegretConfig(
        search_depths=tuple(int(value) for value in args.search_depths),
        primary_search_depth=int(args.primary_search_depth),
        simulations_per_action=int(args.simulations_per_action),
        calibration_budgets=tuple(int(value) for value in args.calibration_budgets),
        exploration_c=float(args.exploration_c),
        progressive_widening_c=float(args.progressive_widening_c),
        progressive_widening_alpha=float(args.progressive_widening_alpha),
        max_joint_actions_per_node=int(args.max_joint_actions_per_node),
        discount=float(args.discount),
    )
    runtime = load_jax_counterfactual_runtime(
        evaluated_checkpoint_path=Path(args.evaluated_checkpoint).expanduser(),
        opponent_checkpoint_path=Path(args.opponent_checkpoint).expanduser(),
        evaluated_action_mode=args.evaluated_action_mode,
        opponent_action_mode=args.opponent_action_mode,
        environment_overrides=_load_environment_overrides(
            args.environment_overrides_json
        ),
        max_rollout_steps=int(args.max_rollout_steps),
        progress_callback=lambda phase: print(phase, flush=True),
    )

    def report_progress(completed: int, total: int) -> None:
        trajectory_path = Path(args.output_dir).expanduser() / "regret_trajectory.json"
        trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))[
            "trajectory"
        ]
        latest = trajectory[-1]
        print(
            f"Analyzed {completed}/{total} sampled states: "
            f"mean regret {latest['regret_mean']:+.4f}, "
            "95% CI "
            f"[{latest['regret_mean_ci95_low']:+.4f}, "
            f"{latest['regret_mean_ci95_high']:+.4f}].",
            flush=True,
        )

    if args.calibration:

        def report_calibration_progress(
            budget_index: int,
            budget_count: int,
            completed: int,
            total: int,
        ) -> None:
            print(
                f"Budget {budget_index}/{budget_count}: analyzed "
                f"{completed}/{total} states.",
                flush=True,
            )

        print(
            "Sampling episodes (the first run compiles the JAX collector).",
            flush=True,
        )
        result = run_budget_calibration_study(
            runtime,
            output_dir=Path(args.output_dir).expanduser(),
            episode_count=int(args.calibration_states),
            seed=int(args.seed),
            protocol=protocol,
            paired_rollout_count=int(args.paired_rollouts),
            max_episode_steps=int(args.max_episode_steps),
            bootstrap_samples=int(args.bootstrap_samples),
            progress_callback=report_calibration_progress,
        )
        evidence = result["convergence"]["main_budget_evidence"]
        print(f"Results: {Path(result['results']).resolve()}")
        print(
            f"{evidence['lower_budget']}→{evidence['upper_budget']} action "
            f"stability: {evidence['selected_action_stability_rate']:.1%}; "
            "mean absolute regret delta: "
            f"{evidence['regret_delta']['mean_absolute']:.4f}."
        )
        return 0

    print(
        "Sampling episodes (the first run compiles the JAX collector).",
        flush=True,
    )
    result = run_root_only_study(
        runtime,
        output_dir=Path(args.output_dir).expanduser(),
        episode_count=int(args.episodes),
        seed=int(args.seed),
        protocol=protocol,
        paired_rollout_count=int(args.paired_rollouts),
        max_episode_steps=int(args.max_episode_steps),
        bootstrap_samples=int(args.bootstrap_samples),
        include_full_possession=(args.planning_mode == "root-and-full-possession"),
        sample_metadata_path=(
            None
            if not str(args.sample_metadata).strip()
            else Path(args.sample_metadata).expanduser()
        ),
        evaluated_role=args.evaluated_role,
        progress_callback=report_progress,
    )
    aggregate = result["aggregate"]
    print(f"Results: {Path(result['results']).resolve()}")
    print(
        "Primary depth "
        f"{aggregate['primary_search_depth']}: "
        f"mean regret {aggregate['primary_regret_mean']:+.4f}, "
        "exact agreement "
        f"{aggregate['primary_exact_action_agreement_rate']:.1%}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
