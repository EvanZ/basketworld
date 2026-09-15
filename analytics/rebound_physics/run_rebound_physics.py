from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from analytics.rebound_physics.model import (
    PhysicsConfig,
    ShotSamplerConfig,
    ShotOrigin,
    catch_xy_at_height,
    sample_shot_params,
    summarize_samples,
    write_samples_jsonl,
    write_summary_json,
)
from analytics.rebound_physics.plotting import (
    plot_3d_scene,
    plot_catch_heatmap,
    plot_contact_heatmaps,
    plot_landing_heatmap,
    plot_rim_outcomes,
    plot_shooter_view_trajectories,
    plot_side_trajectories,
    render_typical_shot_gif,
)
from analytics.rebound_physics.scale import (
    DEFAULT_BW_THREE_POINT_DISTANCE,
    NBA_MAX_THREE_POINT_DISTANCE_METERS,
    meters_per_bw_unit_for_three_point_radius,
)
from analytics.rebound_physics.simulate import run_batch, run_shot


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MuJoCo rebound physics samples.")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument(
        "--misses",
        "--target-misses",
        dest="misses",
        type=int,
        default=0,
        help=(
            "If positive, sample until this many usable missed shots are collected "
            "instead of taking a fixed --samples attempts."
        ),
    )
    parser.add_argument(
        "--miss-target",
        choices=("landing", "catch", "any"),
        default="landing",
        help=(
            "Target used by --misses. landing counts misses with a valid floor landing, "
            "catch counts misses with a valid catch-height intercept, and any counts every miss."
        ),
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=100000,
        help=(
            "Safety cap for --misses. 0 disables the cap; positive values abort if "
            "enough target misses are not collected within this many attempts."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--shot-type",
        choices=("custom", "jumper", "finger_roll", "layup", "dunk"),
        default="jumper",
        help=(
            "Shot profile for single-location calibration. custom/jumper use the base shot params; "
            "finger_roll, layup, and dunk use the same profile defaults as the dataset sweep."
        ),
    )
    parser.add_argument("--shot-x", type=float, default=4.0)
    parser.add_argument("--shot-y", type=float, default=-4.5)
    parser.add_argument("--shot-z", type=float, default=2.0)
    parser.add_argument("--three-point-distance", type=float, default=DEFAULT_BW_THREE_POINT_DISTANCE)
    parser.add_argument("--three-point-short-distance", type=float, default=3.0)
    parser.add_argument(
        "--three-point-radius-meters",
        type=float,
        default=NBA_MAX_THREE_POINT_DISTANCE_METERS,
        help=(
            "Metric radius used to calibrate the BW-to-meter plotting scale when "
            "--meters-per-bw-unit is omitted. Defaults to the NBA max 3pt distance."
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
    parser.add_argument(
        "--draw-three-point-line",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overlay the scaled Basketworld 3pt line on court-space plots, including heatmaps, rim outcomes, 3D scene, and GIF.",
    )
    parser.add_argument(
        "--draw-hex-grid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overlay faint 9x8 Basketworld hex outlines on single-shot landing and catch heatmaps.",
    )
    parser.add_argument(
        "--catch-height",
        type=float,
        default=2.6,
        help="Height in meters used to estimate the in-air rebound catch/intercept location for missed shots.",
    )
    parser.add_argument("--layup-shot-z", type=float, default=2.50)
    parser.add_argument(
        "--layup-target-kind",
        choices=("backboard_box_upper_corner", "backboard_box", "backboard_reflection"),
        default="backboard_box_upper_corner",
        help=(
            "Backboard aim point for --shot-type layup. "
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
        "--shot-model",
        choices=("target_noise", "release_noise"),
        default="target_noise",
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
    parser.add_argument("--release-speed-noise-std", type=float, default=0.025)
    parser.add_argument("--release-lateral-angle-std-degrees", type=float, default=1.0)
    parser.add_argument("--release-vertical-angle-std-degrees", type=float, default=1.0)
    parser.add_argument("--backspin-mean", type=float, default=22.0)
    parser.add_argument("--backspin-std", type=float, default=4.0)
    parser.add_argument("--sidespin-mean", type=float, default=0.0)
    parser.add_argument("--sidespin-std", type=float, default=2.0)
    parser.add_argument(
        "--net-catch-made",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When a shot is classified as made, remove horizontal velocity and drop it vertically like a net.",
    )
    parser.add_argument("--net-downward-speed", type=float, default=1.8)
    parser.add_argument(
        "--backboard-center-z-offset",
        type=float,
        default=0.381,
        help="Backboard center height above rim height, in meters. NBA 42-inch board default is 0.381m.",
    )
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--timestep", type=float, default=0.002)
    parser.add_argument("--contact-timeconst", type=float, default=0.015)
    parser.add_argument("--contact-dampratio", type=float, default=1.0)
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
    parser.add_argument("--trajectory-stride", type=int, default=10)
    parser.add_argument("--max-plot-trajectories", type=int, default=40)
    parser.add_argument("--make-gif", action="store_true")
    parser.add_argument("--gif-shot-index", type=int, default=None)
    parser.add_argument("--gif-fps", type=int, default=18)
    parser.add_argument("--gif-max-frames", type=int, default=120)
    parser.add_argument("--gif-trajectory-count", type=int, default=1)
    parser.add_argument(
        "--gif-spin-mode",
        choices=("none", "seam", "markers"),
        default="none",
        help=(
            "Tiny in-scene ball spin overlay for GIFs. The zoomed spin inset is enabled separately by default; "
            "use seam/markers here only when debugging the regulation-size ball in the 3D court view."
        ),
    )
    parser.add_argument(
        "--gif-spin-inset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw a zoomed side-view spin inset in GIF frames. This is the primary readable spin diagnostic.",
    )
    parser.add_argument(
        "--gif-spin-primary-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw spin overlay only on the first selected trajectory by default.",
    )
    parser.add_argument(
        "--gif-rim-inset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw top-down and frontal rim/backboard closeup insets in GIF frames.",
    )
    parser.add_argument("--gif-spin-alpha", type=float, default=0.75)
    parser.add_argument("--out-dir", type=Path, default=Path("analytics/rebound_physics/outputs"))
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)
    if args.meters_per_bw_unit is None:
        args.meters_per_bw_unit = meters_per_bw_unit_for_three_point_radius(
            three_point_distance=args.three_point_distance,
            three_point_radius_meters=args.three_point_radius_meters,
        )
    return args


def _base_sampler_config(args: argparse.Namespace) -> ShotSamplerConfig:
    return ShotSamplerConfig(
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


def _sample_counts_toward_miss_target(
    sample,
    miss_target: str,
    *,
    catch_height: float,
    config: PhysicsConfig,
) -> bool:
    if bool(sample.made):
        return False
    if miss_target == "any":
        return True
    if miss_target == "landing":
        return sample.landing_xy is not None
    if miss_target == "catch":
        catch_xy = catch_xy_at_height(sample, catch_height)
        return catch_xy is not None and float(catch_xy[1]) <= config.backboard_y
    raise ValueError(f"Unknown miss target: {miss_target}")


def _run_until_target_misses(
    *,
    misses: int,
    miss_target: str,
    max_attempts: int,
    catch_height: float,
    seed: int,
    origin: ShotOrigin,
    config: PhysicsConfig,
    sampler_config: ShotSamplerConfig,
):
    seed_rng = np.random.default_rng(seed)
    output = []
    target_misses = 0
    while target_misses < misses:
        if max_attempts > 0 and len(output) >= max_attempts:
            raise SystemExit(
                f"Reached --max-attempts={max_attempts} before collecting "
                f"{misses} {miss_target} misses; collected {target_misses}."
            )
        shot_seed = int(seed_rng.integers(0, 2**31 - 1))
        shot_rng = np.random.default_rng(shot_seed)
        shot = sample_shot_params(
            shot_rng,
            origin=origin,
            config=config,
            sampler_config=sampler_config,
        )
        sample = run_shot(shot, config=config, seed=shot_seed, shot_index=len(output))
        output.append(sample)
        if _sample_counts_toward_miss_target(
            sample,
            miss_target,
            catch_height=catch_height,
            config=config,
        ):
            target_misses += 1
    return output


def _resolve_shot_profile(args: argparse.Namespace) -> tuple[float, ShotSamplerConfig]:
    """Return release height and sampler config for the requested shot profile."""
    base_config = _base_sampler_config(args)
    if args.shot_type == "finger_roll":
        return args.finger_roll_shot_z, replace(
            base_config,
            shot_model="release_noise",
            target_kind="rim",
            target_vertical_angle_degrees=args.finger_roll_target_vertical_angle_degrees,
            release_speed_noise_std=args.finger_roll_release_speed_noise_std,
            release_lateral_angle_std_degrees=args.finger_roll_release_lateral_angle_std_degrees,
            release_vertical_angle_std_degrees=args.finger_roll_release_vertical_angle_std_degrees,
        )
    if args.shot_type == "layup":
        return args.layup_shot_z, replace(
            base_config,
            shot_model="release_noise",
            target_kind=args.layup_target_kind,
            target_vertical_angle_degrees=args.layup_board_impact_angle_degrees,
            entry_angle_degrees=args.layup_entry_angle_degrees,
            entry_angle_std_degrees=args.layup_entry_angle_std_degrees,
            release_speed_noise_std=args.layup_release_speed_noise_std,
            release_lateral_angle_std_degrees=args.layup_release_lateral_angle_std_degrees,
            release_vertical_angle_std_degrees=args.layup_release_vertical_angle_std_degrees,
        )
    if args.shot_type == "dunk":
        return args.dunk_shot_z, replace(
            base_config,
            shot_model="target_noise",
            target_error_x_std=args.dunk_target_error_x_std,
            target_error_y_std=args.dunk_target_error_y_std,
            target_error_z_std=args.dunk_target_error_z_std,
            flight_time_mean=args.dunk_flight_time_mean,
            flight_time_std=args.dunk_flight_time_std,
            flight_time_min=args.dunk_flight_time_min,
            flight_time_max=args.dunk_flight_time_max,
        )
    return args.shot_z, base_config


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.samples <= 0:
        raise SystemExit("--samples must be positive")
    if args.misses < 0:
        raise SystemExit("--misses must be non-negative")
    if args.max_attempts < 0:
        raise SystemExit("--max-attempts must be non-negative")
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
    shot_z, sampler_config = _resolve_shot_profile(args)
    origin = ShotOrigin(x=args.shot_x, y=args.shot_y, z=shot_z)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    target_miss_mode = args.misses > 0
    try:
        if target_miss_mode:
            samples = _run_until_target_misses(
                misses=args.misses,
                miss_target=args.miss_target,
                max_attempts=args.max_attempts,
                catch_height=args.catch_height,
                seed=args.seed,
                origin=origin,
                config=config,
                sampler_config=sampler_config,
            )
        else:
            samples = run_batch(
                samples=args.samples,
                seed=args.seed,
                origin=origin,
                config=config,
                sampler_config=sampler_config,
            )
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    target_misses_collected = sum(
        1
        for sample in samples
        if _sample_counts_toward_miss_target(
            sample,
            args.miss_target,
            catch_height=args.catch_height,
            config=config,
        )
    )

    samples_path = out_dir / "rebound_physics_samples.jsonl"
    summary_path = out_dir / "rebound_physics_summary.json"
    plot_paths = {
        "landing_heatmap": out_dir / "rebound_physics_landing_heatmap.png",
        "missed_landing_heatmap": out_dir / "rebound_physics_missed_landing_heatmap.png",
        "missed_catch_heatmap": out_dir / "rebound_physics_missed_catch_heatmap.png",
        "contact_heatmaps": out_dir / "rebound_physics_contact_heatmaps.png",
        "rim_outcomes": out_dir / "rebound_physics_rim_outcomes.png",
        "side_trajectories": out_dir / "rebound_physics_side_trajectories.png",
        "shooter_view_trajectories": out_dir / "rebound_physics_shooter_view_trajectories.png",
        "scene_3d": out_dir / "rebound_physics_3d_scene.png",
        "typical_shot_gif": out_dir / "rebound_physics_typical_shot.gif",
    }
    write_samples_jsonl(samples, samples_path)
    summary = summarize_samples(samples, config=config, catch_height=args.catch_height)
    summary["shot_type"] = args.shot_type
    summary["shot_origin"] = {"x": origin.x, "y": origin.y, "z": origin.z}
    summary["shot_sampler_config"] = sampler_config.__dict__
    summary["sampling"] = {
        "mode": "target_misses" if target_miss_mode else "fixed_attempts",
        "requested_samples": int(args.samples),
        "misses": int(args.misses),
        "miss_target": str(args.miss_target),
        "max_attempts": int(args.max_attempts),
        "attempts": int(len(samples)),
        "target_misses_collected": int(target_misses_collected),
    }
    summary["args"] = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    write_summary_json(summary, summary_path)
    if not args.no_plot:
        plot_landing_heatmap(
            samples,
            plot_paths["landing_heatmap"],
            config=config,
            draw_three_point_line=args.draw_three_point_line,
            draw_hex_grid=args.draw_hex_grid,
            three_point_distance=args.three_point_distance,
            three_point_short_distance=args.three_point_short_distance,
            meters_per_bw_unit=args.meters_per_bw_unit,
        )
        plot_landing_heatmap(
            samples,
            plot_paths["missed_landing_heatmap"],
            config=config,
            include_made=False,
            draw_three_point_line=args.draw_three_point_line,
            draw_hex_grid=args.draw_hex_grid,
            three_point_distance=args.three_point_distance,
            three_point_short_distance=args.three_point_short_distance,
            meters_per_bw_unit=args.meters_per_bw_unit,
        )
        plot_catch_heatmap(
            samples,
            plot_paths["missed_catch_heatmap"],
            catch_height=args.catch_height,
            config=config,
            include_made=False,
            draw_three_point_line=args.draw_three_point_line,
            draw_hex_grid=args.draw_hex_grid,
            three_point_distance=args.three_point_distance,
            three_point_short_distance=args.three_point_short_distance,
            meters_per_bw_unit=args.meters_per_bw_unit,
        )
        plot_contact_heatmaps(samples, plot_paths["contact_heatmaps"], config=config)
        plot_rim_outcomes(
            samples,
            plot_paths["rim_outcomes"],
            config=config,
            draw_three_point_line=args.draw_three_point_line,
            three_point_distance=args.three_point_distance,
            three_point_short_distance=args.three_point_short_distance,
            meters_per_bw_unit=args.meters_per_bw_unit,
        )
        plot_side_trajectories(
            samples,
            plot_paths["side_trajectories"],
            config=config,
            max_samples=args.max_plot_trajectories,
        )
        plot_shooter_view_trajectories(
            samples,
            plot_paths["shooter_view_trajectories"],
            config=config,
            max_samples=args.max_plot_trajectories,
        )
        plot_3d_scene(
            samples,
            plot_paths["scene_3d"],
            config=config,
            max_samples=args.max_plot_trajectories,
            draw_three_point_line=args.draw_three_point_line,
            three_point_distance=args.three_point_distance,
            three_point_short_distance=args.three_point_short_distance,
            meters_per_bw_unit=args.meters_per_bw_unit,
        )
        if args.make_gif:
            gif_samples = render_typical_shot_gif(
                samples,
                plot_paths["typical_shot_gif"],
                config=config,
                sample_index=args.gif_shot_index,
                fps=args.gif_fps,
                max_frames=args.gif_max_frames,
                trajectory_count=args.gif_trajectory_count,
                spin_mode=args.gif_spin_mode,
                spin_inset=args.gif_spin_inset,
                rim_inset=args.gif_rim_inset,
                spin_primary_only=args.gif_spin_primary_only,
                spin_alpha=args.gif_spin_alpha,
                draw_three_point_line=args.draw_three_point_line,
                three_point_distance=args.three_point_distance,
                three_point_short_distance=args.three_point_short_distance,
                meters_per_bw_unit=args.meters_per_bw_unit,
            )
            gif_indices = ",".join(str(sample.shot_index) for sample in gif_samples)
            print(f"gif_sample_shot_indices={gif_indices}")

    print(f"wrote {samples_path}")
    print(f"wrote {summary_path}")
    if not args.no_plot:
        for name, plot_path in plot_paths.items():
            if name == "typical_shot_gif" and not args.make_gif:
                continue
            print(f"wrote {plot_path}")
    print(
        f"make_rate={summary['make_rate']:.3f} landing_count={summary['landing_count']} "
        f"attempts={len(samples)} target_misses={target_misses_collected}"
    )


if __name__ == "__main__":
    main()
