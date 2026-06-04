#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from analytics.rebound_physics.dataset import (
    build_canonical_shot_mappings,
    build_dataset_court,
    SHOT_TYPES,
    build_shot_make_arrays,
    build_transition_arrays,
    build_transition_arrays_by_shot_type,
    court_cell_mujoco_xy,
    court_mujoco_xy,
    dataclass_to_jsonable,
    dataset_record_from_sample,
    mapping_record,
    parse_shot_cells,
    plot_fg_pct_heatmap,
    plot_transition_heatmap,
    save_transition_arrays,
    shot_indices_from_cells,
    shot_type_for_cell,
    summarize_dataset_records,
    write_json,
    write_jsonl,
)
from analytics.rebound_physics.model import PhysicsConfig, ShotOrigin, ShotSamplerConfig, sample_shot_params
from analytics.rebound_physics.scale import (
    DEFAULT_BW_THREE_POINT_DISTANCE,
    NBA_MAX_THREE_POINT_DISTANCE_METERS,
    meters_per_bw_unit_for_three_point_radius,
)
from analytics.rebound_physics.simulate import run_shot


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a canonical quarter-court missed-shot rebound dataset from MuJoCo simulations."
    )
    parser.add_argument("--samples-per-cell", type=int, default=250)
    parser.add_argument(
        "--misses-per-cell",
        "--target-misses-per-cell",
        dest="misses_per_cell",
        type=int,
        default=0,
        help=(
            "If positive, sample each canonical shot cell until this many usable missed "
            "shots are collected instead of taking a fixed --samples-per-cell attempts."
        ),
    )
    parser.add_argument(
        "--miss-target",
        choices=("landing", "catch", "any"),
        default="landing",
        help=(
            "Target used by --misses-per-cell. landing counts misses with a valid floor "
            "landing cell, catch counts misses with a valid catch-height cell, and any "
            "counts every missed shot."
        ),
    )
    parser.add_argument(
        "--max-attempts-per-cell",
        type=int,
        default=100000,
        help=(
            "Safety cap for --misses-per-cell. 0 disables the cap; positive values abort if "
            "a cell cannot collect enough target misses within this many attempts."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--court-rows", type=int, default=9)
    parser.add_argument("--court-cols", type=int, default=8)
    parser.add_argument("--three-point-distance", type=float, default=DEFAULT_BW_THREE_POINT_DISTANCE)
    parser.add_argument("--three-point-short-distance", type=float, default=3.0)
    parser.add_argument(
        "--three-point-radius-meters",
        type=float,
        default=NBA_MAX_THREE_POINT_DISTANCE_METERS,
        help=(
            "Metric radius used to calibrate BW-to-meter scale when --meters-per-bw-unit "
            "is omitted. Defaults to the NBA max 3pt distance."
        ),
    )
    parser.add_argument(
        "--meters-per-bw-unit",
        type=float,
        default=None,
        help=(
            "Explicit BW cartesian-unit to meter scale. If omitted, derived as "
            "three_point_radius_meters / (three_point_distance * sqrt(3))."
        ),
    )
    parser.add_argument("--shot-z", type=float, default=2.0, help="Jumper release height in meters.")
    parser.add_argument(
        "--include-dunk-cell",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include the basket hex as the explicit dunk shot type in the dataset sweep.",
    )
    parser.add_argument(
        "--layup-max-distance-hex",
        type=float,
        default=1.0,
        help="Cells at or below this hex distance from the basket use the layup shot profile.",
    )
    parser.add_argument("--layup-shot-z", type=float, default=2.50)
    parser.add_argument(
        "--layup-target-kind",
        choices=("backboard_box_upper_corner", "backboard_box", "backboard_reflection"),
        default="backboard_box_upper_corner",
        help=(
            "Backboard aim point for layup cells. "
            "backboard_box_upper_corner matches the coaching sweet-spot heuristic; "
            "backboard_reflection uses mirror geometry toward rim center."
        ),
    )
    parser.add_argument(
        "--layup-board-impact-angle-degrees",
        type=float,
        default=8.0,
        help=(
            "Vertical angle at the backboard aim point for bank layups. "
            "Positive means the ball is still rising at glass contact, 0 is near-apex, "
            "and negative means descending."
        ),
    )
    parser.add_argument(
        "--layup-entry-angle-degrees",
        type=float,
        default=58.0,
        help=(
            "Deprecated for bank layups. Kept for compatibility; "
            "--layup-board-impact-angle-degrees controls the active layup solver."
        ),
    )
    parser.add_argument(
        "--layup-entry-angle-std-degrees",
        type=float,
        default=7.0,
        help=(
            "Deprecated for bank layups. Layup variability now comes from release speed, "
            "lateral-angle, and vertical-angle noise."
        ),
    )
    parser.add_argument("--layup-release-speed-noise-std", type=float, default=0.075)
    parser.add_argument("--layup-release-lateral-angle-std-degrees", type=float, default=3.5)
    parser.add_argument("--layup-release-vertical-angle-std-degrees", type=float, default=3.5)
    parser.add_argument("--finger-roll-shot-z", type=float, default=2.50)
    parser.add_argument("--finger-roll-target-vertical-angle-degrees", type=float, default=-45.0)
    parser.add_argument("--finger-roll-release-speed-noise-std", type=float, default=0.03)
    parser.add_argument("--finger-roll-release-lateral-angle-std-degrees", type=float, default=1.5)
    parser.add_argument("--finger-roll-release-vertical-angle-std-degrees", type=float, default=1.5)
    parser.add_argument("--dunk-shot-z", type=float, default=3.35)
    parser.add_argument("--dunk-target-error-x-std", type=float, default=0.06)
    parser.add_argument("--dunk-target-error-y-std", type=float, default=0.06)
    parser.add_argument("--dunk-target-error-z-std", type=float, default=0.02)
    parser.add_argument("--dunk-flight-time-mean", type=float, default=0.18)
    parser.add_argument("--dunk-flight-time-std", type=float, default=0.02)
    parser.add_argument("--dunk-flight-time-min", type=float, default=0.12)
    parser.add_argument("--dunk-flight-time-max", type=float, default=0.28)
    parser.add_argument(
        "--shot-cells",
        type=str,
        default="",
        help="Optional flat comma-separated axial q,r shot cells. Values are canonicalized by L/R symmetry.",
    )
    parser.add_argument(
        "--max-shot-cells",
        type=int,
        default=0,
        help="Cap canonical shot cells for smoke tests. 0 means all canonical cells.",
    )
    parser.add_argument(
        "--shot-model",
        choices=("target_noise", "release_noise"),
        default="release_noise",
        help="Shot generation model: noisy target solve or ideal release plus velocity noise.",
    )
    parser.add_argument("--target-error-x-std", type=float, default=0.18)
    parser.add_argument("--target-error-y-std", type=float, default=0.22)
    parser.add_argument("--target-error-z-std", type=float, default=0.05)
    parser.add_argument("--flight-time-mean", type=float, default=0.95)
    parser.add_argument("--flight-time-std", type=float, default=0.08)
    parser.add_argument("--flight-time-min", type=float, default=0.75)
    parser.add_argument("--flight-time-max", type=float, default=1.25)
    parser.add_argument("--entry-angle-degrees", type=float, default=47.0)
    parser.add_argument("--entry-angle-std-degrees", type=float, default=2.0)
    parser.add_argument("--entry-angle-min-degrees", type=float, default=38.0)
    parser.add_argument("--entry-angle-max-degrees", type=float, default=58.0)
    parser.add_argument("--release-speed-noise-std", type=float, default=0.01)
    parser.add_argument("--release-lateral-angle-std-degrees", type=float, default=0.5)
    parser.add_argument("--release-vertical-angle-std-degrees", type=float, default=0.5)
    parser.add_argument("--backspin-mean", type=float, default=22.0)
    parser.add_argument("--backspin-std", type=float, default=4.0)
    parser.add_argument("--sidespin-mean", type=float, default=0.0)
    parser.add_argument("--sidespin-std", type=float, default=2.0)
    parser.add_argument(
        "--net-catch-made",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When a shot is classified as made, drop it vertically like a net.",
    )
    parser.add_argument("--net-downward-speed", type=float, default=1.8)
    parser.add_argument(
        "--backboard-center-z-offset",
        type=float,
        default=0.381,
        help="Backboard center height above rim height, in meters. NBA 42-inch board default is 0.381m.",
    )
    parser.add_argument(
        "--catch-height",
        type=float,
        default=2.6,
        help="Height in meters used to estimate the in-air missed-shot rebound catch/intercept cell.",
    )
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--timestep", type=float, default=0.002)
    parser.add_argument("--contact-timeconst", type=float, default=0.030)
    parser.add_argument("--contact-dampratio", type=float, default=0.060)
    parser.add_argument("--contact-solimp-width", type=float, default=0.9)
    parser.add_argument("--contact-solimp-midpoint", type=float, default=0.95)
    parser.add_argument("--contact-solimp-power", type=float, default=0.001)
    parser.add_argument(
        "--rim-contact-timeconst",
        type=float,
        default=None,
        help="Optional rim-only solref time constant. Defaults to --contact-timeconst when omitted.",
    )
    parser.add_argument(
        "--rim-contact-dampratio",
        type=float,
        default=None,
        help="Optional rim-only solref damping ratio. Increase this to make rim caroms less bouncy.",
    )
    parser.add_argument("--rim-contact-solimp-width", type=float, default=None)
    parser.add_argument("--rim-contact-solimp-midpoint", type=float, default=None)
    parser.add_argument("--rim-contact-solimp-power", type=float, default=None)
    parser.add_argument(
        "--backboard-contact-timeconst",
        type=float,
        default=None,
        help="Optional backboard-only solref time constant. Defaults to --contact-timeconst when omitted.",
    )
    parser.add_argument(
        "--backboard-contact-dampratio",
        type=float,
        default=None,
        help="Optional backboard-only solref damping ratio. Increase this to soften glass rebounds.",
    )
    parser.add_argument("--backboard-contact-solimp-width", type=float, default=None)
    parser.add_argument("--backboard-contact-solimp-midpoint", type=float, default=None)
    parser.add_argument("--backboard-contact-solimp-power", type=float, default=None)
    parser.add_argument("--trajectory-stride", type=int, default=20)
    parser.add_argument("--out-dir", type=Path, default=Path("analytics/rebound_physics/outputs/dataset_9x8"))
    parser.add_argument("--no-plot", action="store_true", help="Skip per-shot PNG heatmaps.")
    argv_list = _normalize_shot_cells_arg(sys.argv[1:] if argv is None else list(argv))
    args = parser.parse_args(argv_list)
    if args.meters_per_bw_unit is None:
        args.meters_per_bw_unit = meters_per_bw_unit_for_three_point_radius(
            three_point_distance=args.three_point_distance,
            three_point_radius_meters=args.three_point_radius_meters,
        )
    return args


def _normalize_shot_cells_arg(argv: list[str]) -> list[str]:
    normalized: list[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--shot-cells" and i + 1 < len(argv):
            normalized.append(f"--shot-cells={argv[i + 1]}")
            i += 2
            continue
        normalized.append(arg)
        i += 1
    return normalized


def _record_counts_toward_miss_target(record: dict[str, object], miss_target: str) -> bool:
    if bool(record.get("made", False)):
        return False
    if miss_target == "any":
        return True
    if miss_target == "landing":
        return record.get("landing_cell_index") is not None
    if miss_target == "catch":
        return record.get("catch_cell_index") is not None and not bool(
            record.get("catch_behind_backboard", False)
        )
    raise ValueError(f"Unknown miss target: {miss_target}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.samples_per_cell <= 0:
        raise SystemExit("--samples-per-cell must be positive")
    if args.misses_per_cell < 0:
        raise SystemExit("--misses-per-cell must be non-negative")
    if args.max_attempts_per_cell < 0:
        raise SystemExit("--max-attempts-per-cell must be non-negative")
    if args.meters_per_bw_unit <= 0.0:
        raise SystemExit("--meters-per-bw-unit must be positive")

    court = build_dataset_court(
        rows=args.court_rows,
        cols=args.court_cols,
        three_point_distance=args.three_point_distance,
        three_point_short_distance=args.three_point_short_distance,
    )
    try:
        shot_cells = parse_shot_cells(args.shot_cells)
        requested_shot_indices = shot_indices_from_cells(
            court,
            shot_cells,
            include_basket=bool(args.include_dunk_cell),
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    canonical_shot_indices, mappings = build_canonical_shot_mappings(
        court,
        requested_shot_indices,
        meters_per_bw_unit=args.meters_per_bw_unit,
    )
    if args.max_shot_cells > 0:
        canonical_shot_indices = canonical_shot_indices[: args.max_shot_cells]
    if not canonical_shot_indices:
        raise SystemExit("No shot cells selected for the dataset sweep.")

    config = PhysicsConfig(
        duration=args.duration,
        timestep=args.timestep,
        trajectory_stride=args.trajectory_stride,
        net_catch_made=args.net_catch_made,
        net_downward_speed=args.net_downward_speed,
        backboard_center_z_offset=args.backboard_center_z_offset,
        contact_solref_timeconst=args.contact_timeconst,
        contact_solref_dampratio=args.contact_dampratio,
        contact_solimp_width=args.contact_solimp_width,
        contact_solimp_midpoint=args.contact_solimp_midpoint,
        contact_solimp_power=args.contact_solimp_power,
        rim_contact_solref_timeconst=args.rim_contact_timeconst,
        rim_contact_solref_dampratio=args.rim_contact_dampratio,
        rim_contact_solimp_width=args.rim_contact_solimp_width,
        rim_contact_solimp_midpoint=args.rim_contact_solimp_midpoint,
        rim_contact_solimp_power=args.rim_contact_solimp_power,
        backboard_contact_solref_timeconst=args.backboard_contact_timeconst,
        backboard_contact_solref_dampratio=args.backboard_contact_dampratio,
        backboard_contact_solimp_width=args.backboard_contact_solimp_width,
        backboard_contact_solimp_midpoint=args.backboard_contact_solimp_midpoint,
        backboard_contact_solimp_power=args.backboard_contact_solimp_power,
    )
    sampler_config = ShotSamplerConfig(
        shot_model=args.shot_model,
        target_error_x_std=args.target_error_x_std,
        target_error_y_std=args.target_error_y_std,
        target_error_z_std=args.target_error_z_std,
        flight_time_mean=args.flight_time_mean,
        flight_time_std=args.flight_time_std,
        flight_time_min=args.flight_time_min,
        flight_time_max=args.flight_time_max,
        entry_angle_degrees=args.entry_angle_degrees,
        entry_angle_std_degrees=args.entry_angle_std_degrees,
        entry_angle_min_degrees=args.entry_angle_min_degrees,
        entry_angle_max_degrees=args.entry_angle_max_degrees,
        release_speed_noise_std=args.release_speed_noise_std,
        release_lateral_angle_std_degrees=args.release_lateral_angle_std_degrees,
        release_vertical_angle_std_degrees=args.release_vertical_angle_std_degrees,
        backspin_mean=args.backspin_mean,
        backspin_std=args.backspin_std,
        sidespin_mean=args.sidespin_mean,
        sidespin_std=args.sidespin_std,
    )
    layup_sampler_config = replace(
        sampler_config,
        shot_model="release_noise",
        target_kind=args.layup_target_kind,
        target_vertical_angle_degrees=args.layup_board_impact_angle_degrees,
        entry_angle_degrees=args.layup_entry_angle_degrees,
        entry_angle_std_degrees=args.layup_entry_angle_std_degrees,
        release_speed_noise_std=args.layup_release_speed_noise_std,
        release_lateral_angle_std_degrees=args.layup_release_lateral_angle_std_degrees,
        release_vertical_angle_std_degrees=args.layup_release_vertical_angle_std_degrees,
    )
    finger_roll_sampler_config = replace(
        sampler_config,
        shot_model="release_noise",
        target_kind="rim",
        target_vertical_angle_degrees=args.finger_roll_target_vertical_angle_degrees,
        release_speed_noise_std=args.finger_roll_release_speed_noise_std,
        release_lateral_angle_std_degrees=args.finger_roll_release_lateral_angle_std_degrees,
        release_vertical_angle_std_degrees=args.finger_roll_release_vertical_angle_std_degrees,
    )
    dunk_sampler_config = replace(
        sampler_config,
        shot_model="target_noise",
        target_error_x_std=args.dunk_target_error_x_std,
        target_error_y_std=args.dunk_target_error_y_std,
        target_error_z_std=args.dunk_target_error_z_std,
        flight_time_mean=args.dunk_flight_time_mean,
        flight_time_std=args.dunk_flight_time_std,
        flight_time_min=args.dunk_flight_time_min,
        flight_time_max=args.dunk_flight_time_max,
    )
    sampler_configs_by_type = {
        "dunk": dunk_sampler_config,
        "finger_roll": finger_roll_sampler_config,
        "layup": layup_sampler_config,
        "jumper": sampler_config,
    }

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    court_xy_mujoco = court_mujoco_xy(court, args.meters_per_bw_unit)
    seed_rng = np.random.default_rng(args.seed)
    raw_records: list[dict[str, object]] = []
    missed_records: list[dict[str, object]] = []
    global_sample_index = 0
    target_miss_mode = args.misses_per_cell > 0

    for shot_row, shot_cell_index in enumerate(canonical_shot_indices):
        shot_q, shot_r = court.cells[int(shot_cell_index)]
        shot_type = shot_type_for_cell(
            court,
            int(shot_cell_index),
            layup_max_distance_hex=args.layup_max_distance_hex,
        )
        shot_z = {
            "dunk": args.dunk_shot_z,
            "finger_roll": args.finger_roll_shot_z,
            "layup": args.layup_shot_z,
        }.get(shot_type, args.shot_z)
        active_sampler_config = sampler_configs_by_type[shot_type]
        origin_xy = court_cell_mujoco_xy(court, int(shot_cell_index), args.meters_per_bw_unit)
        origin = ShotOrigin(x=float(origin_xy[0]), y=float(origin_xy[1]), z=float(shot_z))
        print(
            f"[{shot_row + 1}/{len(canonical_shot_indices)}] "
            f"{shot_type} q={shot_q}, r={shot_r}, origin=({origin.x:.2f}, {origin.y:.2f}, {origin.z:.2f})"
        )
        attempts_for_cell = 0
        target_misses_for_cell = 0
        while True:
            if target_miss_mode:
                if target_misses_for_cell >= args.misses_per_cell:
                    break
                if args.max_attempts_per_cell > 0 and attempts_for_cell >= args.max_attempts_per_cell:
                    raise SystemExit(
                        "Reached --max-attempts-per-cell="
                        f"{args.max_attempts_per_cell} before collecting "
                        f"{args.misses_per_cell} {args.miss_target} misses for "
                        f"{shot_type} q={shot_q}, r={shot_r}; collected "
                        f"{target_misses_for_cell} target misses."
                    )
            elif attempts_for_cell >= args.samples_per_cell:
                break

            shot_seed = int(seed_rng.integers(0, 2**31 - 1))
            shot_rng = np.random.default_rng(shot_seed)
            shot = sample_shot_params(
                shot_rng,
                origin=origin,
                config=config,
                sampler_config=active_sampler_config,
            )
            try:
                sample = run_shot(
                    shot,
                    config=config,
                    seed=shot_seed,
                    shot_index=global_sample_index,
                )
            except RuntimeError as exc:
                raise SystemExit(str(exc)) from exc
            record = dataset_record_from_sample(
                sample,
                court=court,
                court_xy_mujoco=court_xy_mujoco,
                shot_cell_index=int(shot_cell_index),
                sample_index_within_cell=attempts_for_cell,
                global_sample_index=global_sample_index,
                meters_per_bw_unit=args.meters_per_bw_unit,
                layup_max_distance_hex=args.layup_max_distance_hex,
                backboard_y=config.backboard_y,
                catch_height=args.catch_height,
            )
            raw_records.append(record)
            if not record["made"] and record["landing_cell_index"] is not None:
                missed_records.append(record)
            if target_miss_mode and _record_counts_toward_miss_target(record, args.miss_target):
                target_misses_for_cell += 1
            attempts_for_cell += 1
            global_sample_index += 1
        if target_miss_mode:
            print(
                f"  collected {target_misses_for_cell} {args.miss_target} misses "
                f"in {attempts_for_cell} attempts"
            )

    generated_canonical_set = set(int(idx) for idx in canonical_shot_indices)
    mapping_records = [
        mapping_record(court, mapping, meters_per_bw_unit=args.meters_per_bw_unit)
        for mapping in mappings
        if int(mapping.canonical_cell_index) in generated_canonical_set
    ]
    transitions = build_transition_arrays(
        missed_records,
        canonical_shot_indices=canonical_shot_indices,
        cell_count=len(court.cells),
    )
    transitions_by_shot_type = build_transition_arrays_by_shot_type(
        missed_records,
        canonical_shot_indices=canonical_shot_indices,
        court=court,
        cell_count=len(court.cells),
        layup_max_distance_hex=args.layup_max_distance_hex,
    )
    catch_records = [
        record
        for record in raw_records
        if not bool(record.get("made", False))
        and record.get("catch_cell_index") is not None
        and not bool(record.get("catch_behind_backboard", False))
    ]
    catch_transitions = build_transition_arrays(
        catch_records,
        canonical_shot_indices=canonical_shot_indices,
        cell_count=len(court.cells),
        target_cell_field="catch_cell_index",
    )
    catch_transitions_by_shot_type = build_transition_arrays_by_shot_type(
        catch_records,
        canonical_shot_indices=canonical_shot_indices,
        court=court,
        cell_count=len(court.cells),
        layup_max_distance_hex=args.layup_max_distance_hex,
        target_cell_field="catch_cell_index",
    )
    make_arrays = build_shot_make_arrays(
        raw_records,
        canonical_shot_indices=canonical_shot_indices,
        cell_count=len(court.cells),
    )
    summary = summarize_dataset_records(
        raw_records,
        missed_records,
        court=court,
        canonical_shot_indices=canonical_shot_indices,
        meters_per_bw_unit=args.meters_per_bw_unit,
        layup_max_distance_hex=args.layup_max_distance_hex,
        backboard_y=config.backboard_y,
        catch_height=args.catch_height,
    )
    summary["args"] = dataclass_to_jsonable(vars(args))
    summary["sampling"] = {
        "mode": "target_misses" if target_miss_mode else "fixed_attempts",
        "samples_per_cell": int(args.samples_per_cell),
        "misses_per_cell": int(args.misses_per_cell),
        "miss_target": str(args.miss_target),
        "max_attempts_per_cell": int(args.max_attempts_per_cell),
    }
    summary["physics_config"] = dataclass_to_jsonable(config)
    summary["shot_sampler_config"] = dataclass_to_jsonable(sampler_config)
    summary["shot_type_sampler_configs"] = {
        shot_type: dataclass_to_jsonable(type_sampler_config)
        for shot_type, type_sampler_config in sampler_configs_by_type.items()
    }
    summary["transition_arrays"] = save_transition_arrays(transitions, out_dir)
    summary["catch_transition_arrays"] = save_transition_arrays(catch_transitions, out_dir / "catch_model")
    summary["shot_type_transition_arrays"] = {}
    for shot_type in SHOT_TYPES:
        type_transitions = transitions_by_shot_type.get(shot_type)
        if type_transitions is None:
            continue
        type_records = [record for record in raw_records if record.get("shot_type") == shot_type]
        type_missed_records = [record for record in missed_records if record.get("shot_type") == shot_type]
        type_catch_records = [record for record in catch_records if record.get("shot_type") == shot_type]
        type_dir = out_dir / "shot_type_models" / shot_type
        type_catch_transitions = catch_transitions_by_shot_type.get(shot_type)
        summary["shot_type_transition_arrays"][shot_type] = {
            "raw_samples": int(len(type_records)),
            "missed_samples": int(len(type_missed_records)),
            "catch_samples": int(len(type_catch_records)),
            "shot_cell_count": int(type_transitions.row_shot_cell_indices.size),
            "row_shot_cell_indices": type_transitions.row_shot_cell_indices.tolist(),
            "transition_arrays": save_transition_arrays(type_transitions, type_dir),
            "catch_transition_arrays": None
            if type_catch_transitions is None
            else save_transition_arrays(type_catch_transitions, type_dir / "catch_model"),
        }
    fg_pct_heatmap_path = out_dir / "rebound_shot_fg_pct_heatmap.png"
    summary["diagnostic_plots"] = {
        "shot_fg_pct_heatmap": None if args.no_plot else str(fg_pct_heatmap_path),
    }

    raw_path = out_dir / "rebound_dataset_raw_samples.jsonl"
    missed_path = out_dir / "rebound_dataset_missed_samples.jsonl"
    catch_path = out_dir / "rebound_dataset_catch_samples.jsonl"
    summary_path = out_dir / "rebound_dataset_summary.json"
    mapping_path = out_dir / "rebound_canonical_shot_mapping.json"
    write_jsonl(raw_records, raw_path)
    write_jsonl(missed_records, missed_path)
    write_jsonl(catch_records, catch_path)
    write_json(summary, summary_path)
    write_json(mapping_records, mapping_path)

    written_paths = [raw_path, missed_path, catch_path, summary_path, mapping_path]
    if not args.no_plot:
        plot_fg_pct_heatmap(
            court,
            make_arrays,
            path=fg_pct_heatmap_path,
            title="MuJoCo shot make rate by canonical shot location",
        )
        written_paths.append(fg_pct_heatmap_path)
        heatmap_dir = out_dir / "heatmaps"
        for row_idx, shot_cell_index in enumerate(transitions.row_shot_cell_indices.tolist()):
            q, r = court.cells[int(shot_cell_index)]
            heatmap_path = heatmap_dir / f"shot_cell_q{q}_r{r}.png"
            shot_type = shot_type_for_cell(
                court,
                int(shot_cell_index),
                layup_max_distance_hex=args.layup_max_distance_hex,
            )
            plot_transition_heatmap(
                court,
                transitions.counts[row_idx],
                shot_cell_index=int(shot_cell_index),
                path=heatmap_path,
                title=f"Missed-shot {shot_type} rebound landing targets | shot q={q}, r={r}",
            )
            written_paths.append(heatmap_path)

            catch_heatmap_path = out_dir / "catch_heatmaps" / f"shot_cell_q{q}_r{r}.png"
            plot_transition_heatmap(
                court,
                catch_transitions.counts[row_idx],
                shot_cell_index=int(shot_cell_index),
                path=catch_heatmap_path,
                title=(
                    f"Missed-shot {shot_type} rebound catch targets "
                    f"z={args.catch_height:.2f}m | shot q={q}, r={r}"
                ),
            )
            written_paths.append(catch_heatmap_path)

    for path in written_paths:
        print(f"wrote {path}")
    print(
        "dataset "
        f"raw={summary['raw_samples']} missed={summary['missed_samples']} "
        f"made={summary['made_samples']} make_rate={summary['make_rate']:.3f} "
        f"canonical_shots={summary['canonical_shot_cell_count']}"
    )


if __name__ == "__main__":
    main()
