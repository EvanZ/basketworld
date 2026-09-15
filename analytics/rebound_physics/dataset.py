from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np

from analytics.rebound_physics.model import TrajectorySample, catch_xy_at_height, rim_outcome_label
from analytics.rebound_sim.model import Court, CourtSpec, build_court
from analytics.rebound_sim.plotting import _plot_hex_values


@dataclass(frozen=True)
class CanonicalShotMapping:
    shot_cell_index: int
    canonical_cell_index: int
    reflection_sign: int
    symmetry_class: str


@dataclass(frozen=True)
class TransitionArrays:
    row_shot_cell_indices: np.ndarray
    counts: np.ndarray
    probs: np.ndarray
    logits: np.ndarray


@dataclass(frozen=True)
class ShotMakeArrays:
    shot_cell_indices: np.ndarray
    attempts: np.ndarray
    makes: np.ndarray
    fg_pct: np.ndarray


SHOT_TYPES = ("dunk", "finger_roll", "layup", "jumper")


def build_dataset_court(
    *,
    rows: int = 9,
    cols: int = 8,
    three_point_distance: float = 4.25,
    three_point_short_distance: float = 3.0,
) -> Court:
    return build_court(
        CourtSpec(
            rows=rows,
            cols=cols,
            three_point_distance=three_point_distance,
            three_point_short_distance=three_point_short_distance,
        )
    )


def court_cell_mujoco_xy(court: Court, cell_index: int, meters_per_bw_unit: float) -> np.ndarray:
    """Map a BW axial hex center to MuJoCo x/y meters relative to the rim.

    MuJoCo x is lateral left/right. MuJoCo y is negative away from the rim,
    matching the current physics prototype coordinate system.
    """

    rel_xy = court.xy[int(cell_index)] - court.rim_xy
    return np.asarray(
        [float(rel_xy[1]) * meters_per_bw_unit, -float(rel_xy[0]) * meters_per_bw_unit],
        dtype=np.float64,
    )


def court_mujoco_xy(court: Court, meters_per_bw_unit: float) -> np.ndarray:
    rel_xy = court.xy - court.rim_xy[None, :]
    return np.column_stack(
        [
            rel_xy[:, 1] * meters_per_bw_unit,
            -rel_xy[:, 0] * meters_per_bw_unit,
        ]
    ).astype(np.float64)


def nearest_court_cell_index(court_xy_mujoco: np.ndarray, xy: tuple[float, float] | np.ndarray) -> int:
    point = np.asarray(xy, dtype=np.float64)
    distances = np.sum((court_xy_mujoco - point[None, :]) ** 2, axis=1)
    return int(np.argmin(distances))


def parse_shot_cells(value: str) -> list[tuple[int, int]]:
    cleaned = value.strip()
    if not cleaned:
        return []
    for char in "();":
        cleaned = cleaned.replace(char, ",")
    cleaned = cleaned.replace(" ", "")
    parts = [part for part in cleaned.split(",") if part]
    if len(parts) % 2 != 0:
        raise ValueError("--shot-cells expects an even number of comma-separated integers: q,r[,q,r...]")
    try:
        values = [int(part) for part in parts]
    except ValueError as exc:
        raise ValueError("--shot-cells only accepts integer axial coordinates") from exc
    return [(values[i], values[i + 1]) for i in range(0, len(values), 2)]


def shot_type_for_cell(
    court: Court,
    cell_index: int,
    *,
    layup_max_distance_hex: float = 1.0,
    finger_roll_lateral_tolerance: float = 1e-6,
    tolerance: float = 1e-6,
) -> str:
    idx = int(cell_index)
    if idx == int(court.basket_index):
        return "dunk"
    if float(court.distance_hex[idx]) <= float(layup_max_distance_hex) + float(tolerance):
        lateral = float(court.xy[idx, 1] - court.rim_xy[1])
        if abs(lateral) <= float(finger_roll_lateral_tolerance):
            return "finger_roll"
        return "layup"
    return "jumper"


def shot_indices_from_cells(
    court: Court,
    shot_cells: Iterable[tuple[int, int]] | None = None,
    *,
    include_basket: bool = False,
) -> list[int]:
    cells = list(shot_cells or [])
    if not cells:
        return [
            idx
            for idx in range(len(court.cells))
            if bool(include_basket) or idx != int(court.basket_index)
        ]
    shot_indices: list[int] = []
    for cell in cells:
        if cell not in court.cell_index:
            valid = ", ".join(f"{q},{r}" for q, r in court.cells[:8])
            raise ValueError(f"Shot cell {cell} is not on the court. Example valid cells: {valid}, ...")
        idx = int(court.cell_index[cell])
        if bool(include_basket) or idx != int(court.basket_index):
            shot_indices.append(idx)
    return list(dict.fromkeys(shot_indices))


def canonicalize_shot_cell(
    court: Court,
    cell_index: int,
    *,
    meters_per_bw_unit: float,
    court_xy_mujoco: np.ndarray | None = None,
    centerline_tolerance: float = 1e-8,
) -> CanonicalShotMapping:
    court_xy_mujoco = court_xy_mujoco if court_xy_mujoco is not None else court_mujoco_xy(court, meters_per_bw_unit)
    shot_xy = court_xy_mujoco[int(cell_index)]
    lateral = float(shot_xy[0])
    if abs(lateral) <= centerline_tolerance:
        reflection_sign = 1
        symmetry_class = "centerline"
    elif lateral < 0.0:
        reflection_sign = -1
        symmetry_class = "reflected_left_to_right"
    else:
        reflection_sign = 1
        symmetry_class = "canonical_right"
    canonical_xy = np.asarray([abs(lateral), float(shot_xy[1])], dtype=np.float64)
    canonical_cell_index = nearest_court_cell_index(court_xy_mujoco, canonical_xy)
    return CanonicalShotMapping(
        shot_cell_index=int(cell_index),
        canonical_cell_index=int(canonical_cell_index),
        reflection_sign=int(reflection_sign),
        symmetry_class=symmetry_class,
    )


def build_canonical_shot_mappings(
    court: Court,
    shot_indices: Iterable[int],
    *,
    meters_per_bw_unit: float,
) -> tuple[list[int], list[CanonicalShotMapping]]:
    court_xy = court_mujoco_xy(court, meters_per_bw_unit)
    mappings = [
        canonicalize_shot_cell(court, idx, meters_per_bw_unit=meters_per_bw_unit, court_xy_mujoco=court_xy)
        for idx in shot_indices
    ]
    canonical_indices = sorted(
        {mapping.canonical_cell_index for mapping in mappings},
        key=lambda idx: (court.offsets[int(idx)][1], court.offsets[int(idx)][0]),
    )
    return canonical_indices, mappings


def mapping_record(court: Court, mapping: CanonicalShotMapping, *, meters_per_bw_unit: float) -> dict[str, Any]:
    shot_q, shot_r = court.cells[mapping.shot_cell_index]
    canonical_q, canonical_r = court.cells[mapping.canonical_cell_index]
    shot_col, shot_row = court.offsets[mapping.shot_cell_index]
    canonical_col, canonical_row = court.offsets[mapping.canonical_cell_index]
    shot_xy = court_cell_mujoco_xy(court, mapping.shot_cell_index, meters_per_bw_unit)
    canonical_xy = court_cell_mujoco_xy(court, mapping.canonical_cell_index, meters_per_bw_unit)
    return {
        "shot_cell_index": int(mapping.shot_cell_index),
        "shot_q": int(shot_q),
        "shot_r": int(shot_r),
        "shot_offset_col": int(shot_col),
        "shot_offset_row": int(shot_row),
        "shot_x": float(shot_xy[0]),
        "shot_y": float(shot_xy[1]),
        "canonical_shot_cell_index": int(mapping.canonical_cell_index),
        "canonical_shot_q": int(canonical_q),
        "canonical_shot_r": int(canonical_r),
        "canonical_shot_offset_col": int(canonical_col),
        "canonical_shot_offset_row": int(canonical_row),
        "canonical_shot_x": float(canonical_xy[0]),
        "canonical_shot_y": float(canonical_xy[1]),
        "reflection_sign": int(mapping.reflection_sign),
        "symmetry_class": mapping.symmetry_class,
    }


def dataset_record_from_sample(
    sample: TrajectorySample,
    *,
    court: Court,
    court_xy_mujoco: np.ndarray,
    shot_cell_index: int,
    sample_index_within_cell: int,
    global_sample_index: int,
    meters_per_bw_unit: float,
    layup_max_distance_hex: float = 1.0,
    backboard_y: float = 0.381,
    catch_height: float = 2.6,
) -> dict[str, Any]:
    canonical_mapping = canonicalize_shot_cell(
        court,
        shot_cell_index,
        meters_per_bw_unit=meters_per_bw_unit,
        court_xy_mujoco=court_xy_mujoco,
    )
    shot_q, shot_r = court.cells[int(shot_cell_index)]
    shot_col, shot_row = court.offsets[int(shot_cell_index)]
    shot_xy = court_cell_mujoco_xy(court, int(shot_cell_index), meters_per_bw_unit)
    canonical_q, canonical_r = court.cells[int(canonical_mapping.canonical_cell_index)]
    shot_type = shot_type_for_cell(
        court,
        int(shot_cell_index),
        layup_max_distance_hex=layup_max_distance_hex,
    )

    landing_cell_index: int | None = None
    landing_q: int | None = None
    landing_r: int | None = None
    landing_col: int | None = None
    landing_row: int | None = None
    if sample.landing_xy is not None:
        landing_cell_index = nearest_court_cell_index(court_xy_mujoco, sample.landing_xy)
        landing_q, landing_r = court.cells[int(landing_cell_index)]
        landing_col, landing_row = court.offsets[int(landing_cell_index)]

    catch_xy = catch_xy_at_height(sample, catch_height)
    catch_cell_index: int | None = None
    catch_q: int | None = None
    catch_r: int | None = None
    catch_col: int | None = None
    catch_row: int | None = None
    catch_x: float | None = None
    catch_y: float | None = None
    if catch_xy is not None:
        catch_x = float(catch_xy[0])
        catch_y = float(catch_xy[1])
        catch_cell_index = nearest_court_cell_index(court_xy_mujoco, catch_xy)
        catch_q, catch_r = court.cells[int(catch_cell_index)]
        catch_col, catch_row = court.offsets[int(catch_cell_index)]

    rim_crossing_x: float | None = None
    rim_crossing_y: float | None = None
    if sample.rim_crossing_xy is not None:
        rim_crossing_x = float(sample.rim_crossing_xy[0])
        rim_crossing_y = float(sample.rim_crossing_xy[1])

    landing_x: float | None = None
    landing_y: float | None = None
    if sample.landing_xy is not None:
        landing_x = float(sample.landing_xy[0])
        landing_y = float(sample.landing_xy[1])

    settled_x: float | None = None
    settled_y: float | None = None
    if sample.settled_xy is not None:
        settled_x = float(sample.settled_xy[0])
        settled_y = float(sample.settled_xy[1])

    return {
        "global_sample_index": int(global_sample_index),
        "sample_index_within_cell": int(sample_index_within_cell),
        "seed": int(sample.seed),
        "shot_index": int(sample.shot_index),
        "shot_cell_index": int(shot_cell_index),
        "shot_q": int(shot_q),
        "shot_r": int(shot_r),
        "shot_offset_col": int(shot_col),
        "shot_offset_row": int(shot_row),
        "canonical_shot_cell_index": int(canonical_mapping.canonical_cell_index),
        "canonical_shot_q": int(canonical_q),
        "canonical_shot_r": int(canonical_r),
        "shot_type": shot_type,
        "shot_distance_hex": float(court.distance_hex[int(shot_cell_index)]),
        "symmetry_class": canonical_mapping.symmetry_class,
        "reflection_sign": int(canonical_mapping.reflection_sign),
        "shot_x": float(shot_xy[0]),
        "shot_y": float(shot_xy[1]),
        "shot_z": float(sample.shot.origin.z),
        "shot_model": sample.shot.shot_model,
        "target_x": float(sample.shot.target_x),
        "target_y": float(sample.shot.target_y),
        "target_z": float(sample.shot.target_z),
        "flight_time": float(sample.shot.flight_time),
        "entry_angle_degrees": (
            None if sample.shot.entry_angle_degrees is None else float(sample.shot.entry_angle_degrees)
        ),
        "release_speed_error": float(sample.shot.release_speed_error),
        "release_lateral_angle_error_degrees": float(sample.shot.release_lateral_angle_error_degrees),
        "release_vertical_angle_error_degrees": float(sample.shot.release_vertical_angle_error_degrees),
        "made": bool(sample.made),
        "rim_outcome": rim_outcome_label(sample),
        "first_contact": sample.first_contact,
        "contact_sequence": list(sample.contact_sequence),
        "contact_count": int(sample.contact_count),
        "rim_crossing_x": rim_crossing_x,
        "rim_crossing_y": rim_crossing_y,
        "rim_crossing_distance": (
            None if sample.rim_crossing_distance is None else float(sample.rim_crossing_distance)
        ),
        "rim_crossing_time": None if sample.rim_crossing_time is None else float(sample.rim_crossing_time),
        "landing_x": landing_x,
        "landing_y": landing_y,
        "landing_q": None if landing_q is None else int(landing_q),
        "landing_r": None if landing_r is None else int(landing_r),
        "landing_offset_col": None if landing_col is None else int(landing_col),
        "landing_offset_row": None if landing_row is None else int(landing_row),
        "landing_cell_index": landing_cell_index,
        "catch_height": float(catch_height),
        "catch_x": catch_x,
        "catch_y": catch_y,
        "catch_q": None if catch_q is None else int(catch_q),
        "catch_r": None if catch_r is None else int(catch_r),
        "catch_offset_col": None if catch_col is None else int(catch_col),
        "catch_offset_row": None if catch_row is None else int(catch_row),
        "catch_cell_index": catch_cell_index,
        "catch_behind_backboard": bool(catch_y is not None and float(catch_y) > float(backboard_y)),
        "settled_x": settled_x,
        "settled_y": settled_y,
        "max_height": float(sample.max_height),
        "sim_time": float(sample.sim_time),
    }


def write_jsonl(records: Iterable[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def write_json(data: dict[str, Any] | list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_transition_arrays(
    missed_records: Iterable[dict[str, Any]],
    *,
    canonical_shot_indices: Iterable[int],
    cell_count: int,
    empty_row_mode: str = "uniform",
    logit_epsilon: float = 1e-12,
    target_cell_field: str = "landing_cell_index",
    pseudocount: float = 0.0,
) -> TransitionArrays:
    if float(pseudocount) < 0.0:
        raise ValueError("pseudocount must be non-negative")
    row_shot_cell_indices = np.asarray(list(canonical_shot_indices), dtype=np.int32)
    shot_to_row = {int(cell_idx): row for row, cell_idx in enumerate(row_shot_cell_indices.tolist())}
    counts = np.zeros((row_shot_cell_indices.size, int(cell_count)), dtype=np.float64)
    for record in missed_records:
        shot_idx = int(record["canonical_shot_cell_index"])
        target_idx = record.get(target_cell_field)
        if target_idx is None:
            continue
        row = shot_to_row.get(shot_idx)
        if row is None:
            continue
        counts[int(row), int(target_idx)] += 1.0

    if float(pseudocount) > 0.0:
        counts += float(pseudocount)

    row_totals = np.sum(counts, axis=1, keepdims=True)
    probs = np.zeros_like(counts, dtype=np.float64)
    non_empty = row_totals[:, 0] > 0.0
    probs[non_empty] = counts[non_empty] / row_totals[non_empty]
    if np.any(~non_empty):
        if empty_row_mode != "uniform":
            raise ValueError(f"Unsupported empty_row_mode={empty_row_mode!r}")
        probs[~non_empty] = 1.0 / float(max(1, cell_count))
    logits = np.log(np.clip(probs, logit_epsilon, 1.0))
    return TransitionArrays(
        row_shot_cell_indices=row_shot_cell_indices,
        counts=counts,
        probs=probs,
        logits=logits,
    )


def build_transition_arrays_by_shot_type(
    missed_records: Iterable[dict[str, Any]],
    *,
    canonical_shot_indices: Iterable[int],
    court: Court,
    cell_count: int,
    layup_max_distance_hex: float = 1.0,
    target_cell_field: str = "landing_cell_index",
    pseudocount: float = 0.0,
) -> dict[str, TransitionArrays]:
    records = list(missed_records)
    canonical_list = [int(idx) for idx in canonical_shot_indices]
    output: dict[str, TransitionArrays] = {}
    for shot_type in SHOT_TYPES:
        type_indices = [
            idx
            for idx in canonical_list
            if shot_type_for_cell(court, idx, layup_max_distance_hex=layup_max_distance_hex) == shot_type
        ]
        if not type_indices:
            continue
        type_records = [record for record in records if record.get("shot_type") == shot_type]
        output[shot_type] = build_transition_arrays(
            type_records,
            canonical_shot_indices=type_indices,
            cell_count=cell_count,
            target_cell_field=target_cell_field,
            pseudocount=pseudocount,
        )
    return output


def build_shot_make_arrays(
    raw_records: Iterable[dict[str, Any]],
    *,
    canonical_shot_indices: Iterable[int],
    cell_count: int,
) -> ShotMakeArrays:
    shot_cell_indices = np.asarray(list(canonical_shot_indices), dtype=np.int32)
    attempts = np.zeros(int(cell_count), dtype=np.float64)
    makes = np.zeros(int(cell_count), dtype=np.float64)
    canonical_set = {int(idx) for idx in shot_cell_indices.tolist()}
    for record in raw_records:
        shot_idx = int(record["canonical_shot_cell_index"])
        if shot_idx not in canonical_set:
            continue
        attempts[shot_idx] += 1.0
        if bool(record["made"]):
            makes[shot_idx] += 1.0
    fg_pct = np.full(int(cell_count), np.nan, dtype=np.float64)
    non_empty = attempts > 0.0
    fg_pct[non_empty] = makes[non_empty] / attempts[non_empty]
    return ShotMakeArrays(
        shot_cell_indices=shot_cell_indices,
        attempts=attempts,
        makes=makes,
        fg_pct=fg_pct,
    )


def summarize_dataset_records(
    raw_records: list[dict[str, Any]],
    missed_records: list[dict[str, Any]],
    *,
    court: Court,
    canonical_shot_indices: Iterable[int],
    meters_per_bw_unit: float,
    layup_max_distance_hex: float = 1.0,
    backboard_y: float = 0.381,
    catch_height: float = 2.6,
) -> dict[str, Any]:
    canonical_list = [int(idx) for idx in canonical_shot_indices]
    raw_count = len(raw_records)
    missed_count = len(missed_records)
    made_count = sum(1 for record in raw_records if record["made"])
    raw_missed_count = raw_count - made_count
    landing_count = sum(1 for record in raw_records if record["landing_cell_index"] is not None)
    catch_records = [
        record
        for record in raw_records
        if not bool(record.get("made", False)) and record.get("catch_cell_index") is not None
    ]
    model_catch_records = [
        record
        for record in catch_records
        if not bool(record.get("catch_behind_backboard", False))
    ]
    make_arrays = build_shot_make_arrays(
        raw_records,
        canonical_shot_indices=canonical_list,
        cell_count=len(court.cells),
    )
    raw_type_counts = Counter(str(record.get("shot_type", "unknown")) for record in raw_records)
    missed_type_counts = Counter(str(record.get("shot_type", "unknown")) for record in missed_records)
    behind_backboard_records = [
        record
        for record in missed_records
        if record.get("landing_y") is not None and float(record["landing_y"]) > float(backboard_y)
    ]
    behind_backboard_first_contact_counts = Counter(
        str(record.get("first_contact") or "none") for record in behind_backboard_records
    )
    behind_backboard_sequence_counts = Counter(
        "->".join(str(item) for item in record.get("contact_sequence", [])) or "none"
        for record in behind_backboard_records
    )
    behind_backboard_rim_outcome_counts = Counter(
        str(record.get("rim_outcome") or "unknown") for record in behind_backboard_records
    )
    per_shot: list[dict[str, Any]] = []
    shot_make_grid: list[dict[str, Any]] = []
    for shot_idx in canonical_list:
        shot_records = [record for record in raw_records if int(record["canonical_shot_cell_index"]) == shot_idx]
        shot_missed = [record for record in missed_records if int(record["canonical_shot_cell_index"]) == shot_idx]
        shot_catches = [
            record
            for record in shot_records
            if not bool(record.get("made", False)) and record.get("catch_cell_index") is not None
        ]
        shot_model_catches = [record for record in shot_catches if not bool(record.get("catch_behind_backboard", False))]
        q, r = court.cells[shot_idx]
        col, row = court.offsets[shot_idx]
        shot_xy = court_cell_mujoco_xy(court, shot_idx, meters_per_bw_unit)
        shot_type = shot_type_for_cell(
            court,
            shot_idx,
            layup_max_distance_hex=layup_max_distance_hex,
        )
        attempts = int(make_arrays.attempts[shot_idx])
        makes = int(make_arrays.makes[shot_idx])
        fg_pct = None if attempts <= 0 else float(make_arrays.fg_pct[shot_idx])
        per_shot.append(
            {
                "canonical_shot_cell_index": int(shot_idx),
                "canonical_shot_q": int(q),
                "canonical_shot_r": int(r),
                "canonical_shot_x": float(shot_xy[0]),
                "canonical_shot_y": float(shot_xy[1]),
                "shot_type": shot_type,
                "shot_distance_hex": float(court.distance_hex[shot_idx]),
                "raw_samples": int(len(shot_records)),
                "missed_samples": int(len(shot_missed)),
                "catch_samples": int(len(shot_catches)),
                "model_catch_samples": int(len(shot_model_catches)),
                "made_samples": int(makes),
                "make_rate": float(0.0 if fg_pct is None else fg_pct),
                "landing_rate": float(
                    sum(1 for record in shot_records if record["landing_cell_index"] is not None)
                    / max(1, len(shot_records))
                ),
            }
        )
        shot_make_grid.append(
            {
                "cell_index": int(shot_idx),
                "q": int(q),
                "r": int(r),
                "offset_col": int(col),
                "offset_row": int(row),
                "shot_x": float(shot_xy[0]),
                "shot_y": float(shot_xy[1]),
                "shot_type": shot_type,
                "shot_distance_hex": float(court.distance_hex[shot_idx]),
                "attempts": int(attempts),
                "makes": int(makes),
                "fg_pct": fg_pct,
            }
        )
    return {
        "raw_samples": int(raw_count),
        "missed_samples": int(missed_count),
        "raw_missed_samples": int(raw_missed_count),
        "made_samples": int(made_count),
        "make_rate": float(made_count / max(1, raw_count)),
        "behind_backboard_y_threshold": float(backboard_y),
        "behind_backboard_miss_count": int(len(behind_backboard_records)),
        "behind_backboard_miss_rate": float(len(behind_backboard_records) / max(1, missed_count)),
        "behind_backboard_by_first_contact": dict(sorted(behind_backboard_first_contact_counts.items())),
        "behind_backboard_by_contact_sequence": dict(behind_backboard_sequence_counts.most_common(12)),
        "behind_backboard_by_rim_outcome": dict(sorted(behind_backboard_rim_outcome_counts.items())),
        "catch_height": float(catch_height),
        "missed_catch_samples": int(len(catch_records)),
        "missed_catch_rate": float(len(catch_records) / max(1, raw_missed_count)),
        "model_catch_samples": int(len(model_catch_records)),
        "excluded_behind_backboard_catch_samples": int(len(catch_records) - len(model_catch_records)),
        "excluded_behind_backboard_catch_rate": float((len(catch_records) - len(model_catch_records)) / max(1, len(catch_records))),
        "landing_samples": int(landing_count),
        "landing_rate": float(landing_count / max(1, raw_count)),
        "court_rows": int(court.spec.rows),
        "court_cols": int(court.spec.cols),
        "court_cells": int(len(court.cells)),
        "basket_cell_index": int(court.basket_index),
        "meters_per_bw_unit": float(meters_per_bw_unit),
        "canonical_shot_cell_count": int(len(canonical_list)),
        "canonical_shot_cells": per_shot,
        "shot_types": list(SHOT_TYPES),
        "shot_type_counts": {
            shot_type: {
                "raw_samples": int(raw_type_counts.get(shot_type, 0)),
                "missed_samples": int(missed_type_counts.get(shot_type, 0)),
            }
            for shot_type in SHOT_TYPES
        },
        "shot_make_grid": shot_make_grid,
        "shot_make_arrays": {
            "shot_cell_indices": make_arrays.shot_cell_indices.tolist(),
            "attempts": make_arrays.attempts.tolist(),
            "makes": make_arrays.makes.tolist(),
            "fg_pct": [None if not math.isfinite(float(value)) else float(value) for value in make_arrays.fg_pct],
        },
    }


def save_transition_arrays(transitions: TransitionArrays, out_dir: str | Path) -> dict[str, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "row_shot_cell_indices": out_dir / "rebound_transition_row_shot_cell_indices.npy",
        "counts": out_dir / "rebound_transition_counts.npy",
        "probs": out_dir / "rebound_transition_probs.npy",
        "logits": out_dir / "rebound_transition_logits.npy",
    }
    np.save(paths["row_shot_cell_indices"], transitions.row_shot_cell_indices)
    np.save(paths["counts"], transitions.counts)
    np.save(paths["probs"], transitions.probs)
    np.save(paths["logits"], transitions.logits)
    return {key: str(path) for key, path in paths.items()}


def plot_transition_heatmap(
    court: Court,
    values: np.ndarray,
    *,
    shot_cell_index: int,
    path: str | Path,
    title: str | None = None,
    integer: bool = True,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    q, r = court.cells[int(shot_cell_index)]
    title = title or f"Missed-shot rebound landing targets | shot q={q}, r={r}"
    fig, ax = plt.subplots(figsize=(9, 8))
    _plot_hex_values(court, ax, values, title, "Oranges", integer=integer)
    _overlay_three_point_line(court, ax)
    x, y = court.xy[int(shot_cell_index)]
    ax.scatter(
        [float(x)],
        [float(y)],
        marker="*",
        s=470,
        facecolor="#00e5ff",
        edgecolor="#111111",
        linewidth=1.4,
        zorder=20,
    )
    ax.text(
        float(x),
        float(y) - 0.9,
        "shot",
        ha="center",
        va="center",
        fontsize=8,
        fontweight="bold",
        color="#111111",
        bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "alpha": 0.80, "edgecolor": "none"},
        zorder=21,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_fg_pct_heatmap(
    court: Court,
    make_arrays: ShotMakeArrays,
    *,
    path: str | Path,
    title: str | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    title = title or "Field goal percentage by canonical shot location"
    fig, ax = plt.subplots(figsize=(9, 8))
    _plot_hex_values(court, ax, make_arrays.fg_pct, title, "RdYlGn", vmin=0.0, vmax=1.0, integer=False)
    _overlay_three_point_line(court, ax)

    for shot_cell_index in make_arrays.shot_cell_indices.tolist():
        attempts = int(make_arrays.attempts[int(shot_cell_index)])
        makes = int(make_arrays.makes[int(shot_cell_index)])
        if attempts <= 0:
            continue
        x, y = court.xy[int(shot_cell_index)]
        ax.text(
            float(x),
            float(y) + 0.34,
            f"{makes}/{attempts}",
            ha="center",
            va="center",
            fontsize=6,
            color="#111111",
            zorder=22,
        )

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _overlay_three_point_line(
    court: Court,
    ax,
    *,
    color: str = "#f8f3df",
    linewidth: float = 3.0,
    alpha: float = 0.96,
) -> None:
    if court.spec.three_point_distance <= 0.0:
        return
    hoop_x, hoop_y = court.rim_xy
    radius = float(court.spec.three_point_distance) * math.sqrt(3.0)
    short_distance = float(court.spec.three_point_short_distance) * math.sqrt(3.0)
    if radius <= 0.0:
        return

    xs = court.xy[:, 0]
    ys = court.xy[:, 1]
    min_x = float(np.min(xs) - 1.0)
    max_x = float(np.max(xs) + 1.0)
    min_y = float(np.min(ys) - 1.0)
    max_y = float(np.max(ys) + 1.0)

    if 0.0 < short_distance < radius:
        horizontal_reach = math.sqrt(max(0.0, radius * radius - short_distance * short_distance))
        arc_start = math.asin(max(-1.0, min(1.0, short_distance / radius)))
        theta = np.linspace(-arc_start, arc_start, 160)
        arc_x = hoop_x + radius * np.cos(theta)
        arc_y = hoop_y + radius * np.sin(theta)
        line_end_x = float(hoop_x + horizontal_reach)
        line_start_x = min_x
        for sign in (-1.0, 1.0):
            line_y = float(hoop_y + sign * short_distance)
            if min_y <= line_y <= max_y:
                ax.plot(
                    [line_start_x, line_end_x],
                    [line_y, line_y],
                    color=color,
                    linewidth=linewidth,
                    alpha=alpha,
                    solid_capstyle="round",
                    zorder=18,
                )
    else:
        theta = np.linspace(-math.pi / 2.0, math.pi / 2.0, 180)
        arc_x = hoop_x + radius * np.cos(theta)
        arc_y = hoop_y + radius * np.sin(theta)

    visible = (arc_x >= min_x) & (arc_x <= max_x) & (arc_y >= min_y) & (arc_y <= max_y)
    if np.any(visible):
        ax.plot(
            arc_x[visible],
            arc_y[visible],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            solid_capstyle="round",
            zorder=18,
        )


def dataclass_to_jsonable(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [dataclass_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): dataclass_to_jsonable(item) for key, item in value.items()}
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
