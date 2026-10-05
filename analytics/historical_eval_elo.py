#!/usr/bin/env python3
"""Build a deterministic Elo ladder from current historical-eval artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Sequence

from basketworld.utils.historical_eval_elo import (
    DEFAULT_INITIAL_RATING,
    DEFAULT_K_FACTOR,
    DEFAULT_METHOD,
    DEFAULT_PRIOR_STDDEV,
    build_deterministic_elo_ladder,
    write_elo_outputs,
)


DEFAULT_MLFLOW_ARTIFACT_PATH = "results/historical_opponent_evaluation.json"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute an offline deterministic Elo ladder from BasketWorld's "
            "historical-opponent evaluation JSON."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--input-json",
        help="Local historical_opponent_evaluation.json path.",
    )
    source.add_argument(
        "--run-id",
        help="MLflow run ID containing the historical evaluation artifact.",
    )
    parser.add_argument(
        "--artifact-path",
        default=DEFAULT_MLFLOW_ARTIFACT_PATH,
        help="MLflow artifact path used with --run-id.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help=(
            "Output directory. Defaults to <input-dir>/elo_analysis for local "
            "input or ./elo_analysis/<run-id> for MLflow input."
        ),
    )
    parser.add_argument(
        "--initial-rating",
        type=float,
        default=DEFAULT_INITIAL_RATING,
    )
    parser.add_argument(
        "--method",
        choices=("batch", "chronological"),
        default=DEFAULT_METHOD,
        help=(
            "Rating method. 'batch' fits all completed outcomes at once; "
            "'chronological' applies the earlier aggregate K-factor updates."
        ),
    )
    parser.add_argument(
        "--prior-stddev",
        type=float,
        default=DEFAULT_PRIOR_STDDEV,
        help="Gaussian Elo prior standard deviation for --method batch.",
    )
    parser.add_argument(
        "--k-factor",
        type=float,
        default=DEFAULT_K_FACTOR,
        help="K-factor for --method chronological.",
    )
    parser.add_argument(
        "--anchor-update",
        type=int,
        help=(
            "Checkpoint update to assign --anchor-rating after fitting. Must be "
            "used together with --anchor-rating."
        ),
    )
    parser.add_argument(
        "--anchor-rating",
        type=float,
        help=(
            "Displayed Elo rating for --anchor-update. Applies one linear shift "
            "without changing rating gaps or expected scores."
        ),
    )
    return parser.parse_args(argv)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def _download_mlflow_artifact(run_id: str, artifact_path: str, target_dir: str) -> Path:
    from basketworld.utils.mlflow_config import setup_mlflow

    setup_mlflow(verbose=False)
    import mlflow

    downloaded = mlflow.tracking.MlflowClient().download_artifacts(
        run_id,
        artifact_path,
        target_dir,
    )
    return Path(downloaded)


def _print_rankings(result: dict[str, Any]) -> None:
    summary = result["summary"]
    print(
        "Accepted "
        f"{summary['accepted_matchup_count']} deterministic matchups; "
        f"skipped {summary['skipped_record_count']}."
    )
    print("Rank  Update       Elo       95% CI  Games     W-T-L  Score   Point diff")
    for row in result["rankings"]:
        score = "n/a" if row["win_score"] is None else f"{row['win_score']:.3f}"
        point_diff = (
            "n/a"
            if row["mean_point_differential"] is None
            else f"{row['mean_point_differential']:+.3f}"
        )
        record = f"{row['wins']}-{row['ties']}-{row['losses']}"
        interval = (
            "n/a"
            if row["rating_ci95_low"] is None
            else f"{row['rating_ci95_low']:.0f}..{row['rating_ci95_high']:.0f}"
        )
        print(
            f"{row['rank']:>4}  {row['update']:>6}  "
            f"{row['elo_rating']:>8.2f}  {interval:>11}  "
            f"{row['completed_games']:>5}  "
            f"{record:>10}  {score:>5}  {point_diff:>10}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.input_json:
        input_path = Path(args.input_json).expanduser().resolve()
        payload = _load_json(input_path)
        source = {"kind": "local_json", "path": str(input_path)}
        output_dir = (
            Path(args.output_dir).expanduser()
            if args.output_dir
            else input_path.parent / "elo_analysis"
        )
    else:
        run_id = str(args.run_id).strip()
        with TemporaryDirectory(prefix="basketworld_historical_elo_") as tmpdir:
            input_path = _download_mlflow_artifact(
                run_id,
                str(args.artifact_path),
                tmpdir,
            )
            payload = _load_json(input_path)
        source = {
            "kind": "mlflow",
            "run_id": run_id,
            "artifact_path": str(args.artifact_path),
        }
        output_dir = (
            Path(args.output_dir).expanduser()
            if args.output_dir
            else Path.cwd() / "elo_analysis" / run_id
        )

    result = build_deterministic_elo_ladder(
        payload,
        method=args.method,
        initial_rating=args.initial_rating,
        k_factor=args.k_factor,
        prior_stddev=args.prior_stddev,
        anchor_update=args.anchor_update,
        anchor_rating=args.anchor_rating,
        source=source,
    )
    json_path, csv_path = write_elo_outputs(result, output_dir)
    _print_rankings(result)
    print(f"JSON: {json_path.resolve()}")
    print(f"CSV:  {csv_path.resolve()}")
    if not result["summary"]["ratings_globally_comparable"]:
        print(
            "Warning: the accepted matchup graph has multiple disconnected "
            "components; compare ratings only within a component."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
