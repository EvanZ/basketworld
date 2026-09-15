#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from analytics.rebound_physics.dataset import (
    SHOT_TYPES,
    build_dataset_court,
    build_transition_arrays,
    build_transition_arrays_by_shot_type,
    dataclass_to_jsonable,
    save_transition_arrays,
    shot_type_for_cell,
    write_json,
    write_jsonl,
)
from analytics.rebound_physics.scale import (
    DEFAULT_BW_THREE_POINT_DISTANCE,
    NBA_MAX_THREE_POINT_DISTANCE_METERS,
    meters_per_bw_unit_for_three_point_radius,
)

TARGET_CELL_FIELDS = ("catch_cell_index", "landing_cell_index")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit/export a lightweight rebound table model from a generated MuJoCo rebound dataset. "
            "This does not rerun MuJoCo; it filters the completed sweep and writes runtime-ready arrays."
        )
    )
    parser.add_argument("--dataset-dir", type=Path, default=Path("analytics/rebound_physics/outputs/dataset_9x8"))
    parser.add_argument("--raw-jsonl", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--mapping-json", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--target-cell-field",
        choices=TARGET_CELL_FIELDS,
        default="catch_cell_index",
        help=(
            "Which per-sample target to fit. catch_cell_index is recommended for BW because it "
            "approximates the in-air rebound target; landing_cell_index fits first floor contact."
        ),
    )
    parser.add_argument(
        "--filter-behind-backboard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude missed shots whose selected target is behind the backboard.",
    )
    parser.add_argument(
        "--behind-backboard-y-threshold",
        type=float,
        default=None,
        help="Y threshold for behind-backboard filtering. Defaults to the source summary threshold or 0.381.",
    )
    parser.add_argument(
        "--pseudocount",
        type=float,
        default=0.05,
        help="Additive smoothing applied to every target cell in every fitted row.",
    )
    parser.add_argument("--court-rows", type=int, default=None)
    parser.add_argument("--court-cols", type=int, default=None)
    parser.add_argument("--three-point-distance", type=float, default=None)
    parser.add_argument("--three-point-short-distance", type=float, default=None)
    parser.add_argument("--three-point-radius-meters", type=float, default=NBA_MAX_THREE_POINT_DISTANCE_METERS)
    parser.add_argument("--meters-per-bw-unit", type=float, default=None)
    parser.add_argument("--layup-max-distance-hex", type=float, default=None)
    args = parser.parse_args(argv)
    if args.pseudocount < 0.0:
        raise SystemExit("--pseudocount must be non-negative")
    return args


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def _target_y(record: dict[str, Any], target_cell_field: str) -> float | None:
    if target_cell_field == "catch_cell_index":
        value = record.get("catch_y")
    elif target_cell_field == "landing_cell_index":
        value = record.get("landing_y")
    else:
        raise ValueError(f"Unsupported target field: {target_cell_field}")
    return None if value is None else float(value)


def _filter_reason(
    record: dict[str, Any],
    *,
    target_cell_field: str,
    filter_behind_backboard: bool,
    behind_backboard_y_threshold: float,
) -> str | None:
    if bool(record.get("made", False)):
        return "made"
    if record.get(target_cell_field) is None:
        return "missing_target"
    if filter_behind_backboard:
        target_y = _target_y(record, target_cell_field)
        if target_cell_field == "catch_cell_index" and bool(record.get("catch_behind_backboard", False)):
            return "behind_backboard"
        if target_y is not None and target_y > float(behind_backboard_y_threshold):
            return "behind_backboard"
    return None


def _filter_fit_records(
    records: list[dict[str, Any]],
    *,
    target_cell_field: str,
    filter_behind_backboard: bool,
    behind_backboard_y_threshold: float,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    kept: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    for record in records:
        reason = _filter_reason(
            record,
            target_cell_field=target_cell_field,
            filter_behind_backboard=filter_behind_backboard,
            behind_backboard_y_threshold=behind_backboard_y_threshold,
        )
        if reason is None:
            kept.append(record)
        else:
            rejected[reason] += 1
    return kept, rejected


def _summary_arg(source_summary: dict[str, Any], name: str, fallback: Any) -> Any:
    args = source_summary.get("args") if isinstance(source_summary.get("args"), dict) else {}
    return args.get(name, fallback)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    dataset_dir = args.dataset_dir
    raw_path = args.raw_jsonl or dataset_dir / "rebound_dataset_raw_samples.jsonl"
    summary_path = args.summary_json or dataset_dir / "rebound_dataset_summary.json"
    mapping_path = args.mapping_json or dataset_dir / "rebound_canonical_shot_mapping.json"
    target_name = str(args.target_cell_field).removesuffix("_cell_index")
    out_dir = args.out_dir or dataset_dir / f"fitted_{target_name}_model"

    if not raw_path.exists():
        raise SystemExit(f"Raw sample file does not exist: {raw_path}")
    if not summary_path.exists():
        raise SystemExit(f"Dataset summary does not exist: {summary_path}")

    source_summary = _read_json(summary_path)
    records = _read_jsonl(raw_path)
    backboard_y = (
        float(args.behind_backboard_y_threshold)
        if args.behind_backboard_y_threshold is not None
        else float(source_summary.get("behind_backboard_y_threshold", 0.381))
    )
    fit_records, rejected = _filter_fit_records(
        records,
        target_cell_field=args.target_cell_field,
        filter_behind_backboard=bool(args.filter_behind_backboard),
        behind_backboard_y_threshold=backboard_y,
    )
    if not fit_records:
        raise SystemExit("No records remained after filtering; cannot fit rebound model.")

    court_rows = int(args.court_rows or _summary_arg(source_summary, "court_rows", source_summary.get("court_rows", 9)))
    court_cols = int(args.court_cols or _summary_arg(source_summary, "court_cols", source_summary.get("court_cols", 8)))
    three_point_distance = float(
        args.three_point_distance
        if args.three_point_distance is not None
        else _summary_arg(source_summary, "three_point_distance", DEFAULT_BW_THREE_POINT_DISTANCE)
    )
    three_point_short_distance = float(
        args.three_point_short_distance
        if args.three_point_short_distance is not None
        else _summary_arg(source_summary, "three_point_short_distance", 3.0)
    )
    meters_per_bw_unit = (
        float(args.meters_per_bw_unit)
        if args.meters_per_bw_unit is not None
        else float(
            _summary_arg(
                source_summary,
                "meters_per_bw_unit",
                meters_per_bw_unit_for_three_point_radius(
                    three_point_distance=three_point_distance,
                    three_point_radius_meters=float(args.three_point_radius_meters),
                ),
            )
        )
    )
    layup_max_distance_hex = float(
        args.layup_max_distance_hex
        if args.layup_max_distance_hex is not None
        else _summary_arg(source_summary, "layup_max_distance_hex", 1.0)
    )
    court = build_dataset_court(
        rows=court_rows,
        cols=court_cols,
        three_point_distance=three_point_distance,
        three_point_short_distance=three_point_short_distance,
    )
    canonical_shot_indices = [
        int(record["canonical_shot_cell_index"])
        for record in source_summary.get("canonical_shot_cells", [])
    ]
    if not canonical_shot_indices:
        canonical_shot_indices = sorted({int(record["canonical_shot_cell_index"]) for record in fit_records})

    out_dir.mkdir(parents=True, exist_ok=True)
    transitions = build_transition_arrays(
        fit_records,
        canonical_shot_indices=canonical_shot_indices,
        cell_count=len(court.cells),
        target_cell_field=args.target_cell_field,
        pseudocount=float(args.pseudocount),
    )
    transitions_by_shot_type = build_transition_arrays_by_shot_type(
        fit_records,
        canonical_shot_indices=canonical_shot_indices,
        court=court,
        cell_count=len(court.cells),
        layup_max_distance_hex=layup_max_distance_hex,
        target_cell_field=args.target_cell_field,
        pseudocount=float(args.pseudocount),
    )

    fit_samples_path = out_dir / "rebound_fit_samples.jsonl"
    fit_summary_path = out_dir / "rebound_fit_summary.json"
    write_jsonl(fit_records, fit_samples_path)
    transition_paths = save_transition_arrays(transitions, out_dir)
    shot_type_paths: dict[str, Any] = {}
    for shot_type in SHOT_TYPES:
        type_transitions = transitions_by_shot_type.get(shot_type)
        if type_transitions is None:
            continue
        type_records = [record for record in fit_records if str(record.get("shot_type")) == shot_type]
        shot_type_paths[shot_type] = {
            "fit_samples": int(len(type_records)),
            "shot_cell_count": int(type_transitions.row_shot_cell_indices.size),
            "row_shot_cell_indices": type_transitions.row_shot_cell_indices.tolist(),
            "transition_arrays": save_transition_arrays(type_transitions, out_dir / "shot_type_models" / shot_type),
        }

    target_counts_by_shot = Counter(int(record["canonical_shot_cell_index"]) for record in fit_records)
    missing_rows = [idx for idx in canonical_shot_indices if int(target_counts_by_shot.get(int(idx), 0)) <= 0]
    fit_type_counts = Counter(str(record.get("shot_type", "unknown")) for record in fit_records)
    summary = {
        "source_dataset_dir": str(dataset_dir),
        "source_raw_jsonl": str(raw_path),
        "source_summary_json": str(summary_path),
        "source_mapping_json": str(mapping_path) if mapping_path.exists() else None,
        "target_cell_field": str(args.target_cell_field),
        "filter_behind_backboard": bool(args.filter_behind_backboard),
        "behind_backboard_y_threshold": float(backboard_y),
        "pseudocount": float(args.pseudocount),
        "counts_include_pseudocount": bool(float(args.pseudocount) > 0.0),
        "raw_samples_loaded": int(len(records)),
        "fit_samples": int(len(fit_records)),
        "rejected_samples": {key: int(value) for key, value in sorted(rejected.items())},
        "fit_shot_type_counts": {shot_type: int(fit_type_counts.get(shot_type, 0)) for shot_type in SHOT_TYPES},
        "court_rows": int(court_rows),
        "court_cols": int(court_cols),
        "court_cells": int(len(court.cells)),
        "basket_cell_index": int(court.basket_index),
        "meters_per_bw_unit": float(meters_per_bw_unit),
        "three_point_distance": float(three_point_distance),
        "three_point_short_distance": float(three_point_short_distance),
        "layup_max_distance_hex": float(layup_max_distance_hex),
        "canonical_shot_cell_count": int(len(canonical_shot_indices)),
        "missing_unsmoothed_rows": [int(idx) for idx in missing_rows],
        "canonical_shot_cells": [
            {
                "canonical_shot_cell_index": int(idx),
                "q": int(court.cells[int(idx)][0]),
                "r": int(court.cells[int(idx)][1]),
                "shot_type": shot_type_for_cell(court, int(idx), layup_max_distance_hex=layup_max_distance_hex),
                "fit_samples": int(target_counts_by_shot.get(int(idx), 0)),
            }
            for idx in canonical_shot_indices
        ],
        "fit_samples_jsonl": str(fit_samples_path),
        "transition_arrays": transition_paths,
        "shot_type_transition_arrays": shot_type_paths,
        "args": dataclass_to_jsonable(vars(args)),
    }
    write_json(summary, fit_summary_path)
    if mapping_path.exists():
        shutil.copyfile(mapping_path, out_dir / "rebound_canonical_shot_mapping.json")

    print(f"wrote {fit_samples_path}")
    print(f"wrote {fit_summary_path}")
    print(
        "fit rebound model "
        f"target={args.target_cell_field} samples={len(fit_records)} "
        f"rejected={dict(rejected)} rows={len(canonical_shot_indices)} out={out_dir}"
    )


if __name__ == "__main__":
    main()
