from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PhysicsConfig:
    timestep: float = 0.002
    duration: float = 6.0
    gravity: float = 9.81
    floor_size: float = 12.0
    rim_height: float = 3.05
    rim_radius: float = 0.2286
    rim_tube_radius: float = 0.018
    rim_segments: int = 24
    make_radius: float = 0.16
    net_catch_made: bool = True
    net_downward_speed: float = 1.8
    ball_radius: float = 0.1213
    ball_mass: float = 0.624
    # Backboard face is behind the rim center by inner rim radius + 6 inches,
    # matching the regulation nearest-inside-edge spacing.
    backboard_y: float = 0.381
    # A 42-inch backboard spans from 9 ft 6 in to 13 ft above the floor.
    # With a 10 ft rim, its center is 15 in above rim height.
    backboard_center_z_offset: float = 0.381
    backboard_width: float = 1.83
    backboard_height: float = 1.07
    backboard_thickness: float = 0.05
    # Regulation target box is 24 in wide x 18 in high; its bottom edge is at rim height.
    backboard_box_width: float = 0.6096
    backboard_box_center_z_offset: float = 0.2286
    trajectory_stride: int = 10
    max_contact_points: int = 96
    contact_solref_timeconst: float = 0.015
    contact_solref_dampratio: float = 1.0
    contact_solimp_width: float = 0.9
    contact_solimp_midpoint: float = 0.95
    contact_solimp_power: float = 0.001
    rim_contact_solref_timeconst: float | None = None
    rim_contact_solref_dampratio: float | None = None
    rim_contact_solimp_width: float | None = None
    rim_contact_solimp_midpoint: float | None = None
    rim_contact_solimp_power: float | None = None
    backboard_contact_solref_timeconst: float | None = None
    backboard_contact_solref_dampratio: float | None = None
    backboard_contact_solimp_width: float | None = None
    backboard_contact_solimp_midpoint: float | None = None
    backboard_contact_solimp_power: float | None = None


@dataclass(frozen=True)
class ShotOrigin:
    x: float = 4.0
    y: float = -4.5
    z: float = 2.0


@dataclass(frozen=True)
class ShotSamplerConfig:
    shot_model: str = "target_noise"
    target_error_x_std: float = 0.18
    target_error_y_std: float = 0.22
    target_error_z_std: float = 0.05
    flight_time_mean: float = 0.95
    flight_time_std: float = 0.08
    flight_time_min: float = 0.75
    flight_time_max: float = 1.25
    entry_angle_degrees: float = 47.0
    entry_angle_std_degrees: float = 2.0
    entry_angle_min_degrees: float = 38.0
    entry_angle_max_degrees: float = 58.0
    release_speed_noise_std: float = 0.025
    release_lateral_angle_std_degrees: float = 1.0
    release_vertical_angle_std_degrees: float = 1.0
    # Optional target-contact vertical angle for release-noise shots. Positive
    # means the ball is still rising at the aim point; negative means descending.
    target_vertical_angle_degrees: float | None = None
    backspin_mean: float = 22.0
    backspin_std: float = 4.0
    sidespin_mean: float = 0.0
    sidespin_std: float = 2.0
    target_kind: str = "rim"


@dataclass(frozen=True)
class ShotParams:
    origin: ShotOrigin
    target_x: float
    target_y: float
    target_z: float
    flight_time: float
    velocity: tuple[float, float, float]
    angular_velocity: tuple[float, float, float]
    target_error_xy: tuple[float, float]
    target_error_z: float
    shot_model: str = "target_noise"
    target_kind: str = "rim"
    entry_angle_degrees: float | None = None
    target_vertical_angle_degrees: float | None = None
    release_speed_error: float = 0.0
    release_lateral_angle_error_degrees: float = 0.0
    release_vertical_angle_error_degrees: float = 0.0


@dataclass(frozen=True)
class ContactPoint:
    kind: str
    x: float
    y: float
    z: float
    time: float


@dataclass(frozen=True)
class TrajectorySample:
    seed: int
    shot_index: int
    shot: ShotParams
    made: bool
    first_contact: str | None
    contact_sequence: tuple[str, ...]
    contact_count: int
    landing_xy: tuple[float, float] | None
    settled_xy: tuple[float, float] | None
    max_height: float
    sim_time: float
    rim_crossing_xy: tuple[float, float] | None = None
    rim_crossing_distance: float | None = None
    rim_crossing_time: float | None = None
    contact_points: tuple[ContactPoint, ...] = field(default_factory=tuple)
    trajectory_xyz: tuple[tuple[float, float, float], ...] = field(default_factory=tuple)
    trajectory_quat_wxyz: tuple[tuple[float, float, float, float], ...] = field(default_factory=tuple)


def rim_outcome_label(sample: TrajectorySample) -> str:
    if sample.rim_crossing_xy is None:
        return "no downward crossing"
    if sample.made:
        return "made"
    if any(point.kind in {"rim", "backboard"} for point in sample.contact_points):
        return "rim/backboard miss"
    return "clean miss"


def catch_xy_at_height(sample: TrajectorySample, catch_height: float) -> tuple[float, float] | None:
    """Estimate the first descending x/y location where a miss reaches catch height.

    This is a lightweight proxy for an in-air rebound opportunity. It uses the
    decimated trajectory stored on each sample, so lower trajectory_stride values
    improve interpolation fidelity for this diagnostic/model target.
    """

    if sample.made or catch_height <= 0.0 or not sample.trajectory_xyz:
        return None
    trajectory = np.asarray(sample.trajectory_xyz, dtype=np.float64)
    if trajectory.ndim != 2 or trajectory.shape[1] < 3 or len(trajectory) < 2:
        return None
    finite = np.all(np.isfinite(trajectory[:, :3]), axis=1)
    trajectory = trajectory[finite]
    if len(trajectory) < 2:
        return None

    height = float(catch_height)
    for prev, curr in zip(trajectory[:-1], trajectory[1:]):
        prev_z = float(prev[2])
        curr_z = float(curr[2])
        if prev_z <= curr_z:
            continue
        if prev_z >= height >= curr_z:
            denom = prev_z - curr_z
            alpha = 0.0 if denom <= 1e-12 else (prev_z - height) / denom
            alpha = float(np.clip(alpha, 0.0, 1.0))
            xy = prev[:2] + alpha * (curr[:2] - prev[:2])
            return (float(xy[0]), float(xy[1]))
    return None


def _sample_spin(
    rng: np.random.Generator,
    sampler_config: ShotSamplerConfig,
    *,
    origin: ShotOrigin,
    target_x: float,
    target_y: float,
) -> tuple[float, float, float]:
    """Sample angular velocity using basketball-oriented spin semantics.

    Back/top spin is measured around the shot-relative lateral axis, so the
    same signed value works from either side of the floor. Positive values are
    the existing backspin convention; negative values produce topspin. Side
    spin / English is measured around the vertical axis.
    """

    backspin = float(
        np.clip(
            rng.normal(sampler_config.backspin_mean, sampler_config.backspin_std),
            -35.0,
            35.0,
        )
    )
    sidespin = float(
        np.clip(
            rng.normal(sampler_config.sidespin_mean, sampler_config.sidespin_std),
            -35.0,
            35.0,
        )
    )
    forward = np.asarray([float(target_x) - float(origin.x), float(target_y) - float(origin.y)], dtype=np.float64)
    norm = float(np.linalg.norm(forward))
    if norm < 1e-8:
        lateral = np.asarray([1.0, 0.0], dtype=np.float64)
    else:
        forward = forward / norm
        lateral = np.asarray([forward[1], -forward[0]], dtype=np.float64)
    angular = np.asarray([backspin * lateral[0], backspin * lateral[1], sidespin], dtype=np.float64)
    return (float(angular[0]), float(angular[1]), float(angular[2]))


def _resolve_shot_target(
    origin: ShotOrigin,
    *,
    config: PhysicsConfig,
    sampler_config: ShotSamplerConfig,
) -> tuple[float, float, float]:
    if sampler_config.target_kind == "rim":
        return (0.0, 0.0, float(config.rim_height))
    if sampler_config.target_kind == "backboard_box":
        return (
            0.0,
            float(config.backboard_y - config.ball_radius),
            float(config.rim_height + config.backboard_box_center_z_offset),
        )
    if sampler_config.target_kind == "backboard_box_upper_corner":
        side = 0.0 if abs(origin.x) < 1e-8 else float(np.sign(origin.x))
        return (
            float(side * config.backboard_box_width * 0.5),
            float(config.backboard_y - config.ball_radius),
            float(config.rim_height + 2.0 * config.backboard_box_center_z_offset),
        )
    if sampler_config.target_kind == "backboard_reflection":
        # Mirror the rim through the vertical glass plane in x/y only. The
        # vertical aim point is still a regulation glass-box target; otherwise
        # close/high releases can produce below-rim board targets.
        plane_y = float(config.backboard_y - config.ball_radius)
        mirrored_rim_y = 2.0 * plane_y
        denom = mirrored_rim_y - float(origin.y)
        target_z = float(config.rim_height + 2.0 * config.backboard_box_center_z_offset)
        if abs(denom) < 1e-8:
            return (0.0, plane_y, target_z)
        t = float(np.clip((plane_y - float(origin.y)) / denom, 0.0, 1.0))
        target_x = float(origin.x + t * (0.0 - float(origin.x)))
        return (target_x, plane_y, target_z)
    raise ValueError(f"Unsupported target_kind={sampler_config.target_kind!r}")


def _solve_velocity_to_target_time(
    origin: ShotOrigin,
    *,
    target_x: float,
    target_y: float,
    target_z: float,
    flight_time: float,
    config: PhysicsConfig,
) -> tuple[float, float, float]:
    vx = (target_x - origin.x) / flight_time
    vy = (target_y - origin.y) / flight_time
    vz = (target_z - origin.z + 0.5 * config.gravity * flight_time * flight_time) / flight_time
    return (float(vx), float(vy), float(vz))


def _solve_ideal_velocity_for_entry_angle(
    origin: ShotOrigin,
    *,
    target_x: float,
    target_y: float,
    target_z: float,
    entry_angle_degrees: float,
    config: PhysicsConfig,
    fallback_flight_time: float,
) -> tuple[tuple[float, float, float], float]:
    target_xy = np.array([target_x, target_y], dtype=np.float64)
    origin_xy = np.array([origin.x, origin.y], dtype=np.float64)
    delta_xy = target_xy - origin_xy
    horizontal_distance = float(np.linalg.norm(delta_xy))
    if horizontal_distance < 1e-8:
        velocity = _solve_velocity_to_target_time(
            origin,
            target_x=target_x,
            target_y=target_y,
            target_z=target_z,
            flight_time=fallback_flight_time,
            config=config,
        )
        return velocity, fallback_flight_time

    theta = np.deg2rad(entry_angle_degrees)
    height_delta = target_z - origin.z
    flight_term = 2.0 * (height_delta + horizontal_distance * np.tan(theta)) / config.gravity
    if not np.isfinite(flight_term) or flight_term <= 0.0:
        velocity = _solve_velocity_to_target_time(
            origin,
            target_x=target_x,
            target_y=target_y,
            target_z=target_z,
            flight_time=fallback_flight_time,
            config=config,
        )
        return velocity, fallback_flight_time

    flight_time = float(np.sqrt(flight_term))
    horizontal_speed = horizontal_distance / flight_time
    direction_xy = delta_xy / horizontal_distance
    vz = config.gravity * flight_time - horizontal_speed * np.tan(theta)
    velocity = (
        float(direction_xy[0] * horizontal_speed),
        float(direction_xy[1] * horizontal_speed),
        float(vz),
    )
    return velocity, flight_time


def _solve_ideal_velocity_for_target_vertical_angle(
    origin: ShotOrigin,
    *,
    target_x: float,
    target_y: float,
    target_z: float,
    target_vertical_angle_degrees: float,
    config: PhysicsConfig,
    fallback_flight_time: float,
) -> tuple[tuple[float, float, float], float]:
    target_xy = np.array([target_x, target_y], dtype=np.float64)
    origin_xy = np.array([origin.x, origin.y], dtype=np.float64)
    delta_xy = target_xy - origin_xy
    horizontal_distance = float(np.linalg.norm(delta_xy))
    if horizontal_distance < 1e-8:
        velocity = _solve_velocity_to_target_time(
            origin,
            target_x=target_x,
            target_y=target_y,
            target_z=target_z,
            flight_time=fallback_flight_time,
            config=config,
        )
        return velocity, fallback_flight_time

    phi = np.deg2rad(target_vertical_angle_degrees)
    height_delta = target_z - origin.z
    # At the target, constrain vertical velocity relative to horizontal speed:
    # final_vz = horizontal_speed * tan(phi). Positive phi means rising at glass.
    flight_term = 2.0 * (height_delta - horizontal_distance * np.tan(phi)) / config.gravity
    if not np.isfinite(flight_term) or flight_term <= 0.0:
        velocity = _solve_velocity_to_target_time(
            origin,
            target_x=target_x,
            target_y=target_y,
            target_z=target_z,
            flight_time=fallback_flight_time,
            config=config,
        )
        return velocity, fallback_flight_time

    flight_time = float(np.sqrt(flight_term))
    horizontal_speed = horizontal_distance / flight_time
    direction_xy = delta_xy / horizontal_distance
    vz = horizontal_speed * np.tan(phi) + config.gravity * flight_time
    velocity = (
        float(direction_xy[0] * horizontal_speed),
        float(direction_xy[1] * horizontal_speed),
        float(vz),
    )
    return velocity, flight_time


def _apply_release_noise(
    rng: np.random.Generator,
    ideal_velocity: tuple[float, float, float],
    origin: ShotOrigin,
    sampler_config: ShotSamplerConfig,
) -> tuple[tuple[float, float, float], float, float, float]:
    ideal = np.asarray(ideal_velocity, dtype=np.float64)
    speed = float(np.linalg.norm(ideal))
    horizontal_speed = float(np.linalg.norm(ideal[:2]))
    if speed < 1e-8 or horizontal_speed < 1e-8:
        return ideal_velocity, 0.0, 0.0, 0.0

    # Perturb around the ideal horizontal aim direction. For rim-targeted
    # jumpers this points at the rim; for bank layups it points at the glass.
    forward = ideal[:2] / horizontal_speed
    lateral = np.array([-forward[1], forward[0]], dtype=np.float64)

    speed_error = float(rng.normal(0.0, sampler_config.release_speed_noise_std))
    lateral_error_deg = float(rng.normal(0.0, sampler_config.release_lateral_angle_std_degrees))
    vertical_error_deg = float(rng.normal(0.0, sampler_config.release_vertical_angle_std_degrees))

    noisy_speed = max(0.1, speed * (1.0 + speed_error))
    base_elevation = float(np.arctan2(ideal[2], horizontal_speed))
    noisy_elevation = float(np.clip(base_elevation + np.deg2rad(vertical_error_deg), np.deg2rad(3.0), np.deg2rad(82.0)))
    noisy_azimuth = np.deg2rad(lateral_error_deg)
    horizontal_direction = np.cos(noisy_azimuth) * forward + np.sin(noisy_azimuth) * lateral
    noisy_horizontal_speed = noisy_speed * np.cos(noisy_elevation)
    noisy_velocity = (
        float(horizontal_direction[0] * noisy_horizontal_speed),
        float(horizontal_direction[1] * noisy_horizontal_speed),
        float(noisy_speed * np.sin(noisy_elevation)),
    )
    return noisy_velocity, speed_error, lateral_error_deg, vertical_error_deg


def sample_shot_params(
    rng: np.random.Generator,
    *,
    origin: ShotOrigin | None = None,
    config: PhysicsConfig | None = None,
    sampler_config: ShotSamplerConfig | None = None,
) -> ShotParams:
    """Sample a shot from either target-noise or release-noise generation."""
    origin = origin or ShotOrigin()
    config = config or PhysicsConfig()
    sampler_config = sampler_config or ShotSamplerConfig()

    if sampler_config.shot_model == "target_noise":
        err_x = float(rng.normal(0.0, sampler_config.target_error_x_std))
        err_y = float(rng.normal(0.0, sampler_config.target_error_y_std))
        err_z = float(rng.normal(0.0, sampler_config.target_error_z_std))
        flight_time = float(
            np.clip(
                rng.normal(sampler_config.flight_time_mean, sampler_config.flight_time_std),
                sampler_config.flight_time_min,
                sampler_config.flight_time_max,
            )
        )

        base_target_x, base_target_y, base_target_z = _resolve_shot_target(
            origin,
            config=config,
            sampler_config=sampler_config,
        )
        target_x = base_target_x + err_x
        target_y = base_target_y + err_y
        target_z = base_target_z + err_z
        velocity = _solve_velocity_to_target_time(
            origin,
            target_x=target_x,
            target_y=target_y,
            target_z=target_z,
            flight_time=flight_time,
            config=config,
        )
        spin = _sample_spin(
            rng,
            sampler_config,
            origin=origin,
            target_x=target_x,
            target_y=target_y,
        )
        return ShotParams(
            origin=origin,
            target_x=target_x,
            target_y=target_y,
            target_z=target_z,
            flight_time=flight_time,
            velocity=velocity,
            angular_velocity=spin,
            target_error_xy=(err_x, err_y),
            target_error_z=err_z,
            shot_model=sampler_config.shot_model,
            target_kind=sampler_config.target_kind,
        )

    if sampler_config.shot_model == "release_noise":
        target_vertical_angle = sampler_config.target_vertical_angle_degrees
        entry_angle = None
        if target_vertical_angle is None:
            entry_angle = float(
                np.clip(
                    rng.normal(sampler_config.entry_angle_degrees, sampler_config.entry_angle_std_degrees),
                    sampler_config.entry_angle_min_degrees,
                    sampler_config.entry_angle_max_degrees,
                )
            )
        fallback_flight_time = float(
            np.clip(
                rng.normal(sampler_config.flight_time_mean, sampler_config.flight_time_std),
                sampler_config.flight_time_min,
                sampler_config.flight_time_max,
            )
        )
        target_x, target_y, target_z = _resolve_shot_target(
            origin,
            config=config,
            sampler_config=sampler_config,
        )
        if target_vertical_angle is None:
            assert entry_angle is not None
            ideal_velocity, flight_time = _solve_ideal_velocity_for_entry_angle(
                origin,
                target_x=target_x,
                target_y=target_y,
                target_z=target_z,
                entry_angle_degrees=entry_angle,
                config=config,
                fallback_flight_time=fallback_flight_time,
            )
        else:
            ideal_velocity, flight_time = _solve_ideal_velocity_for_target_vertical_angle(
                origin,
                target_x=target_x,
                target_y=target_y,
                target_z=target_z,
                target_vertical_angle_degrees=float(target_vertical_angle),
                config=config,
                fallback_flight_time=fallback_flight_time,
            )
        velocity, speed_error, lateral_error_deg, vertical_error_deg = _apply_release_noise(
            rng,
            ideal_velocity,
            origin,
            sampler_config,
        )
        spin = _sample_spin(
            rng,
            sampler_config,
            origin=origin,
            target_x=target_x,
            target_y=target_y,
        )
        return ShotParams(
            origin=origin,
            target_x=target_x,
            target_y=target_y,
            target_z=target_z,
            flight_time=flight_time,
            velocity=velocity,
            angular_velocity=spin,
            target_error_xy=(0.0, 0.0),
            target_error_z=0.0,
            shot_model=sampler_config.shot_model,
            target_kind=sampler_config.target_kind,
            entry_angle_degrees=entry_angle,
            target_vertical_angle_degrees=target_vertical_angle,
            release_speed_error=speed_error,
            release_lateral_angle_error_degrees=lateral_error_deg,
            release_vertical_angle_error_degrees=vertical_error_deg,
        )

    raise ValueError(f"Unsupported shot_model={sampler_config.shot_model!r}")


def summarize_samples(
    samples: list[TrajectorySample],
    *,
    config: PhysicsConfig | None = None,
    behind_backboard_margin: float = 0.0,
    catch_height: float | None = None,
) -> dict[str, Any]:
    config = config or PhysicsConfig()
    landings = np.array([s.landing_xy for s in samples if s.landing_xy is not None], dtype=np.float64)
    rim_crossings = np.array([s.rim_crossing_xy for s in samples if s.rim_crossing_xy is not None], dtype=np.float64)
    contact_counts = Counter(s.first_contact or "none" for s in samples)
    sequence_counts = Counter("->".join(s.contact_sequence) or "none" for s in samples)
    point_counts = Counter(point.kind for sample in samples for point in sample.contact_points)
    rim_outcome_counts = Counter(rim_outcome_label(sample) for sample in samples)
    crossing_count = len(rim_crossings)
    clean_miss_count = sum(count for outcome, count in rim_outcome_counts.items() if outcome == "clean miss")
    missed_samples = [sample for sample in samples if not sample.made]
    missed_landing_samples = [sample for sample in missed_samples if sample.landing_xy is not None]
    behind_y_threshold = float(config.backboard_y + behind_backboard_margin)
    behind_backboard_samples = [
        sample
        for sample in missed_landing_samples
        if sample.landing_xy is not None and float(sample.landing_xy[1]) > behind_y_threshold
    ]
    behind_backboard_landings = np.array(
        [sample.landing_xy for sample in behind_backboard_samples],
        dtype=np.float64,
    )
    behind_backboard_first_contact_counts = Counter(
        sample.first_contact or "none" for sample in behind_backboard_samples
    )
    behind_backboard_sequence_counts = Counter(
        "->".join(sample.contact_sequence) or "none" for sample in behind_backboard_samples
    )
    behind_backboard_rim_outcome_counts = Counter(rim_outcome_label(sample) for sample in behind_backboard_samples)
    catch_height_value = None if catch_height is None else float(catch_height)
    missed_catch_points: list[tuple[float, float]] = []
    behind_backboard_catch_points: list[tuple[float, float]] = []
    if catch_height_value is not None:
        for sample in missed_landing_samples:
            catch_xy = catch_xy_at_height(sample, catch_height_value)
            if catch_xy is None:
                continue
            missed_catch_points.append(catch_xy)
            if float(catch_xy[1]) > behind_y_threshold:
                behind_backboard_catch_points.append(catch_xy)

    summary: dict[str, Any] = {
        "samples": len(samples),
        "made": int(sum(s.made for s in samples)),
        "make_rate": float(sum(s.made for s in samples) / max(1, len(samples))),
        "landing_count": int(len(landings)),
        "missed_landing_count": int(len(missed_landing_samples)),
        "behind_backboard_y_threshold": behind_y_threshold,
        "behind_backboard_miss_count": int(len(behind_backboard_samples)),
        "behind_backboard_miss_rate": float(
            len(behind_backboard_samples) / max(1, len(missed_landing_samples))
        ),
        "behind_backboard_by_first_contact": dict(sorted(behind_backboard_first_contact_counts.items())),
        "behind_backboard_by_contact_sequence": dict(behind_backboard_sequence_counts.most_common(12)),
        "behind_backboard_by_rim_outcome": dict(sorted(behind_backboard_rim_outcome_counts.items())),
        "catch_height": catch_height_value,
        "missed_catch_count": int(len(missed_catch_points)),
        "missed_catch_rate": float(len(missed_catch_points) / max(1, len(missed_landing_samples))),
        "behind_backboard_catch_count": int(len(behind_backboard_catch_points)),
        "behind_backboard_catch_rate": float(len(behind_backboard_catch_points) / max(1, len(missed_catch_points))),
        "rim_crossing_count": int(crossing_count),
        "clean_miss_count": int(clean_miss_count),
        "rim_outcome_counts": dict(sorted(rim_outcome_counts.items())),
        "first_contact_counts": dict(sorted(contact_counts.items())),
        "contact_point_counts": dict(sorted(point_counts.items())),
        "contact_sequence_counts": dict(sequence_counts.most_common(12)),
        "mean_max_height": float(np.mean([s.max_height for s in samples])) if samples else 0.0,
        "mean_sim_time": float(np.mean([s.sim_time for s in samples])) if samples else 0.0,
    }
    if len(landings):
        summary.update(
            {
                "landing_mean_xy": [float(v) for v in np.mean(landings, axis=0)],
                "landing_std_xy": [float(v) for v in np.std(landings, axis=0)],
                "landing_mean_distance_from_rim": float(np.mean(np.linalg.norm(landings, axis=1))),
            }
        )
    if len(rim_crossings):
        rim_distances = np.linalg.norm(rim_crossings, axis=1)
        summary.update(
            {
                "rim_crossing_mean_xy": [float(v) for v in np.mean(rim_crossings, axis=0)],
                "rim_crossing_std_xy": [float(v) for v in np.std(rim_crossings, axis=0)],
                "rim_crossing_mean_distance": float(np.mean(rim_distances)),
                "rim_crossing_std_distance": float(np.std(rim_distances)),
            }
        )
    if len(behind_backboard_landings):
        summary.update(
            {
                "behind_backboard_landing_mean_xy": [
                    float(v) for v in np.mean(behind_backboard_landings, axis=0)
                ],
                "behind_backboard_landing_std_xy": [
                    float(v) for v in np.std(behind_backboard_landings, axis=0)
                ],
            }
        )
    if len(missed_catch_points):
        catch_arr = np.asarray(missed_catch_points, dtype=np.float64)
        summary.update(
            {
                "missed_catch_mean_xy": [float(v) for v in np.mean(catch_arr, axis=0)],
                "missed_catch_std_xy": [float(v) for v in np.std(catch_arr, axis=0)],
            }
        )
    if len(behind_backboard_catch_points):
        behind_catch_arr = np.asarray(behind_backboard_catch_points, dtype=np.float64)
        summary.update(
            {
                "behind_backboard_catch_mean_xy": [
                    float(v) for v in np.mean(behind_catch_arr, axis=0)
                ],
                "behind_backboard_catch_std_xy": [
                    float(v) for v in np.std(behind_catch_arr, axis=0)
                ],
            }
        )
    for kind in sorted(point_counts):
        points = np.array(
            [[p.x, p.y, p.z] for sample in samples for p in sample.contact_points if p.kind == kind],
            dtype=np.float64,
        )
        if len(points):
            summary[f"{kind}_contact_mean_xyz"] = [float(v) for v in np.mean(points, axis=0)]
            summary[f"{kind}_contact_std_xyz"] = [float(v) for v in np.std(points, axis=0)]
    return summary


def write_samples_jsonl(samples: list[TrajectorySample], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(asdict(sample), sort_keys=True) + "\n")


def write_summary_json(summary: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
