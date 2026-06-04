from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from analytics.rebound_physics.model import (
    ContactPoint,
    PhysicsConfig,
    ShotSamplerConfig,
    ShotOrigin,
    TrajectorySample,
    catch_xy_at_height,
    sample_shot_params,
    summarize_samples,
    write_samples_jsonl,
    write_summary_json,
)
from analytics.rebound_physics.plotting import (
    _shot_frame,
    plot_3d_scene,
    plot_catch_heatmap,
    plot_contact_heatmaps,
    plot_landing_heatmap,
    plot_rim_outcomes,
    plot_shooter_view_trajectories,
    render_typical_shot_gif,
    plot_side_trajectories,
)
from analytics.rebound_physics.scene import build_scene_xml
from analytics.rebound_physics.run_rebound_physics import _sample_counts_toward_miss_target


def _fake_sample(seed: int = 0) -> TrajectorySample:
    rng = np.random.default_rng(seed)
    shot = sample_shot_params(rng, origin=ShotOrigin(x=1.0, y=-3.0, z=2.0))
    return TrajectorySample(
        seed=seed,
        shot_index=seed,
        shot=shot,
        made=False,
        first_contact="rim",
        contact_sequence=("rim", "floor"),
        contact_count=2,
        landing_xy=(0.5, -1.25),
        settled_xy=(0.55, -1.1),
        max_height=3.9,
        sim_time=2.4,
        rim_crossing_xy=(0.05, 0.03),
        rim_crossing_distance=float((0.05**2 + 0.03**2) ** 0.5),
        rim_crossing_time=0.75,
        contact_points=(
            ContactPoint(kind="rim", x=0.1, y=0.2, z=3.05, time=0.8),
            ContactPoint(kind="backboard", x=0.2, y=0.19, z=3.25, time=0.9),
            ContactPoint(kind="floor", x=0.5, y=-1.25, z=0.0, time=1.9),
        ),
        trajectory_xyz=(
            (1.0, -3.0, 2.0),
            (0.7, -2.0, 3.1),
            (0.1, -0.2, 3.7),
            (0.5, -1.25, 0.1),
        ),
        trajectory_quat_wxyz=(
            (1.0, 0.0, 0.0, 0.0),
            (0.99, 0.1, 0.0, 0.0),
            (0.96, 0.28, 0.0, 0.0),
            (0.90, 0.42, 0.0, 0.0),
        ),
    )


def test_scene_xml_contains_core_objects() -> None:
    xml = build_scene_xml(config=PhysicsConfig(rim_segments=8), shot_origin=ShotOrigin())
    assert 'name="floor"' in xml
    assert 'name="backboard"' in xml
    assert 'name="ball"' in xml
    assert 'name="rim_00"' in xml
    assert 'freejoint name="ball_free"' in xml


def test_default_backboard_spacing_matches_regulation_rim_offset() -> None:
    config = PhysicsConfig()
    assert config.backboard_y == pytest.approx(config.rim_radius + 0.1524, abs=5e-4)
    assert config.backboard_center_z_offset == pytest.approx(0.381, abs=5e-4)
    xml = build_scene_xml(config=config)
    expected_center_y = config.backboard_y + config.backboard_thickness * 0.5
    expected_center_z = config.rim_height + config.backboard_center_z_offset
    assert f'pos="0 {expected_center_y:.6f} {expected_center_z:.6f}"' in xml


def test_scene_xml_uses_contact_solver_config() -> None:
    config = PhysicsConfig(
        contact_solref_timeconst=0.021,
        contact_solref_dampratio=0.42,
        contact_solimp_width=0.81,
        contact_solimp_midpoint=0.91,
        contact_solimp_power=0.002,
    )
    xml = build_scene_xml(config=config, shot_origin=ShotOrigin())
    assert 'solref="0.021000 0.420000"' in xml
    assert 'solimp="0.810000 0.910000 0.002000"' in xml


def test_scene_xml_supports_rim_and_backboard_contact_overrides() -> None:
    config = PhysicsConfig(
        contact_solref_timeconst=0.030,
        contact_solref_dampratio=0.060,
        contact_solimp_width=0.90,
        contact_solimp_midpoint=0.95,
        contact_solimp_power=0.001,
        rim_contact_solref_dampratio=0.180,
        backboard_contact_solref_timeconst=0.040,
        backboard_contact_solref_dampratio=0.220,
        backboard_contact_solimp_width=0.70,
    )
    xml = build_scene_xml(config=config, shot_origin=ShotOrigin())
    assert 'name="rim_00"' in xml
    assert 'solref="0.030000 0.180000"' in xml
    assert 'name="backboard" type="box"' in xml
    assert 'solref="0.040000 0.220000"' in xml
    assert 'solimp="0.700000 0.950000 0.001000"' in xml


def test_shot_sampler_is_deterministic() -> None:
    first = sample_shot_params(np.random.default_rng(12))
    second = sample_shot_params(np.random.default_rng(12))
    assert first == second
    assert first.shot_model == "target_noise"
    assert first.flight_time > 0.0
    assert len(first.velocity) == 3

    release_config = ShotSamplerConfig(shot_model="release_noise")
    release_first = sample_shot_params(np.random.default_rng(12), sampler_config=release_config)
    release_second = sample_shot_params(np.random.default_rng(12), sampler_config=release_config)
    assert release_first == release_second
    assert release_first.shot_model == "release_noise"
    assert release_first.entry_angle_degrees is not None
    assert len(release_first.velocity) == 3


def test_shot_sampler_rejects_unknown_model() -> None:
    with pytest.raises(ValueError):
        sample_shot_params(np.random.default_rng(12), sampler_config=ShotSamplerConfig(shot_model="bad"))


def test_spin_sampler_preserves_signed_backspin() -> None:
    base_config = ShotSamplerConfig(
        target_error_x_std=0.0,
        target_error_y_std=0.0,
        target_error_z_std=0.0,
        flight_time_std=0.0,
        backspin_std=0.0,
        sidespin_mean=0.0,
        sidespin_std=0.0,
    )
    origin = ShotOrigin(x=0.0, y=-4.5, z=2.0)
    topspin = sample_shot_params(
        np.random.default_rng(12),
        origin=origin,
        sampler_config=replace(base_config, backspin_mean=-22.0),
    )
    backspin = sample_shot_params(
        np.random.default_rng(12),
        origin=origin,
        sampler_config=replace(base_config, backspin_mean=22.0),
    )

    assert topspin.angular_velocity == pytest.approx((-22.0, 0.0, 0.0))
    assert backspin.angular_velocity == pytest.approx((22.0, 0.0, 0.0))


def test_spin_sampler_maps_sidespin_to_vertical_axis() -> None:
    sampler_config = ShotSamplerConfig(
        target_error_x_std=0.0,
        target_error_y_std=0.0,
        target_error_z_std=0.0,
        flight_time_std=0.0,
        backspin_mean=0.0,
        backspin_std=0.0,
        sidespin_mean=12.0,
        sidespin_std=0.0,
    )
    shot = sample_shot_params(
        np.random.default_rng(12),
        origin=ShotOrigin(x=0.0, y=-4.5, z=2.0),
        sampler_config=sampler_config,
    )

    assert shot.angular_velocity == pytest.approx((0.0, 0.0, 12.0))




def test_plot_shot_frame_matches_shot_relative_spin_axis() -> None:
    config = PhysicsConfig()
    sampler_config = ShotSamplerConfig(
        shot_model="release_noise",
        target_kind="backboard_box_upper_corner",
        target_vertical_angle_degrees=8.0,
        flight_time_std=0.0,
        release_speed_noise_std=0.0,
        release_lateral_angle_std_degrees=0.0,
        release_vertical_angle_std_degrees=0.0,
        backspin_mean=22.0,
        backspin_std=0.0,
        sidespin_mean=0.0,
        sidespin_std=0.0,
    )
    shot = sample_shot_params(
        np.random.default_rng(12),
        origin=ShotOrigin(x=1.0, y=-1.0, z=2.5),
        config=config,
        sampler_config=sampler_config,
    )
    sample = TrajectorySample(
        seed=12,
        shot_index=0,
        shot=shot,
        made=False,
        first_contact=None,
        contact_sequence=(),
        contact_count=0,
        landing_xy=None,
        settled_xy=None,
        max_height=0.0,
        sim_time=0.0,
    )

    _, lateral = _shot_frame(sample)
    angular_xy = np.asarray(shot.angular_velocity[:2], dtype=np.float64)

    assert shot.angular_velocity[2] == pytest.approx(0.0)
    assert angular_xy == pytest.approx(22.0 * lateral)

def test_backboard_box_target_uses_regulation_box_center() -> None:
    config = PhysicsConfig()
    sampler_config = ShotSamplerConfig(
        shot_model="release_noise",
        target_kind="backboard_box",
        entry_angle_std_degrees=0.0,
        release_speed_noise_std=0.0,
        release_lateral_angle_std_degrees=0.0,
        release_vertical_angle_std_degrees=0.0,
    )
    shot = sample_shot_params(
        np.random.default_rng(12),
        origin=ShotOrigin(x=1.0, y=-1.0, z=2.5),
        config=config,
        sampler_config=sampler_config,
    )
    assert shot.target_kind == "backboard_box"
    assert shot.target_x == pytest.approx(0.0)
    assert shot.target_y == pytest.approx(config.backboard_y - config.ball_radius)
    assert shot.target_z == pytest.approx(config.rim_height + config.backboard_box_center_z_offset)


def test_backboard_upper_corner_target_tracks_shooter_side() -> None:
    config = PhysicsConfig()
    sampler_config = ShotSamplerConfig(
        shot_model="release_noise",
        target_kind="backboard_box_upper_corner",
        entry_angle_std_degrees=0.0,
        release_speed_noise_std=0.0,
        release_lateral_angle_std_degrees=0.0,
        release_vertical_angle_std_degrees=0.0,
    )

    right = sample_shot_params(
        np.random.default_rng(12),
        origin=ShotOrigin(x=1.0, y=-1.0, z=2.5),
        config=config,
        sampler_config=sampler_config,
    )
    left = sample_shot_params(
        np.random.default_rng(12),
        origin=ShotOrigin(x=-1.0, y=-1.0, z=2.5),
        config=config,
        sampler_config=sampler_config,
    )
    center = sample_shot_params(
        np.random.default_rng(12),
        origin=ShotOrigin(x=0.0, y=-1.0, z=2.5),
        config=config,
        sampler_config=sampler_config,
    )

    assert right.target_kind == "backboard_box_upper_corner"
    assert right.target_x == pytest.approx(config.backboard_box_width * 0.5)
    assert left.target_x == pytest.approx(-config.backboard_box_width * 0.5)
    assert center.target_x == pytest.approx(0.0)
    assert right.target_y == pytest.approx(config.backboard_y - config.ball_radius)
    assert right.target_z == pytest.approx(config.rim_height + 2.0 * config.backboard_box_center_z_offset)



def test_layup_board_impact_angle_can_be_rising_at_glass() -> None:
    config = PhysicsConfig()
    impact_angle = 8.0
    sampler_config = ShotSamplerConfig(
        shot_model="release_noise",
        target_kind="backboard_box_upper_corner",
        target_vertical_angle_degrees=impact_angle,
        flight_time_std=0.0,
        release_speed_noise_std=0.0,
        release_lateral_angle_std_degrees=0.0,
        release_vertical_angle_std_degrees=0.0,
        backspin_std=0.0,
        sidespin_std=0.0,
    )
    shot = sample_shot_params(
        np.random.default_rng(12),
        origin=ShotOrigin(x=1.0, y=-1.0, z=2.5),
        config=config,
        sampler_config=sampler_config,
    )

    vx, vy, vz = shot.velocity
    horizontal_speed = float(np.linalg.norm([vx, vy]))
    target_vz = vz - config.gravity * shot.flight_time

    assert shot.entry_angle_degrees is None
    assert shot.target_vertical_angle_degrees == pytest.approx(impact_angle)
    assert target_vz > 0.0
    assert target_vz / horizontal_speed == pytest.approx(np.tan(np.deg2rad(impact_angle)))


def test_backboard_reflection_target_uses_mirrored_horizontal_geometry() -> None:
    config = PhysicsConfig()
    sampler_config = ShotSamplerConfig(
        shot_model="release_noise",
        target_kind="backboard_reflection",
        entry_angle_std_degrees=0.0,
        release_speed_noise_std=0.0,
        release_lateral_angle_std_degrees=0.0,
        release_vertical_angle_std_degrees=0.0,
    )
    origin = ShotOrigin(x=1.0, y=-1.0, z=2.5)
    shot = sample_shot_params(
        np.random.default_rng(12),
        origin=origin,
        config=config,
        sampler_config=sampler_config,
    )

    plane_y = config.backboard_y - config.ball_radius
    mirrored_rim_y = 2.0 * plane_y
    t = (plane_y - origin.y) / (mirrored_rim_y - origin.y)

    assert shot.target_kind == "backboard_reflection"
    assert shot.target_x == pytest.approx(origin.x + t * (0.0 - origin.x))
    assert shot.target_y == pytest.approx(plane_y)
    assert shot.target_z == pytest.approx(config.rim_height + 2.0 * config.backboard_box_center_z_offset)
    assert shot.target_z > config.rim_height


def test_shot_sampler_rejects_unknown_target_kind() -> None:
    with pytest.raises(ValueError):
        sample_shot_params(
            np.random.default_rng(12),
            sampler_config=ShotSamplerConfig(shot_model="release_noise", target_kind="bad"),
        )


def test_single_location_miss_target_modes() -> None:
    sample = _fake_sample(1)
    made_sample = replace(sample, made=True)
    no_landing = replace(sample, landing_xy=None)

    assert not _sample_counts_toward_miss_target(
        made_sample, "any", catch_height=2.6, config=PhysicsConfig()
    )
    assert _sample_counts_toward_miss_target(
        sample, "any", catch_height=2.6, config=PhysicsConfig()
    )
    assert _sample_counts_toward_miss_target(
        sample, "landing", catch_height=2.6, config=PhysicsConfig()
    )
    assert not _sample_counts_toward_miss_target(
        no_landing, "landing", catch_height=2.6, config=PhysicsConfig()
    )
    assert _sample_counts_toward_miss_target(
        no_landing, "catch", catch_height=2.6, config=PhysicsConfig()
    )
    assert not _sample_counts_toward_miss_target(
        no_landing, "catch", catch_height=2.6, config=PhysicsConfig(backboard_y=-1.0)
    )
    with pytest.raises(ValueError):
        _sample_counts_toward_miss_target(sample, "bad", catch_height=2.6, config=PhysicsConfig())


def test_summary_and_writers(tmp_path) -> None:
    samples = [_fake_sample(1), _fake_sample(2)]
    summary = summarize_samples(samples)
    assert summary["samples"] == 2
    assert summary["landing_count"] == 2
    assert summary["first_contact_counts"]["rim"] == 2
    assert summary["contact_point_counts"]["rim"] == 2
    assert "rim_contact_mean_xyz" in summary
    assert summary["rim_crossing_count"] == 2
    assert summary["rim_outcome_counts"]["rim/backboard miss"] == 2
    assert "rim_crossing_mean_distance" in summary
    assert summary["behind_backboard_miss_count"] == 0

    catch_xy = catch_xy_at_height(samples[0], 2.6)
    assert catch_xy is not None
    catch_summary = summarize_samples(samples, catch_height=2.6)
    assert catch_summary["catch_height"] == pytest.approx(2.6)
    assert catch_summary["missed_catch_count"] == 2
    assert catch_summary["missed_catch_rate"] == pytest.approx(1.0)

    behind_summary = summarize_samples(samples, config=PhysicsConfig(backboard_y=-2.0))
    assert behind_summary["behind_backboard_miss_count"] == 2
    assert behind_summary["behind_backboard_by_first_contact"]["rim"] == 2
    assert behind_summary["behind_backboard_by_contact_sequence"]["rim->floor"] == 2

    jsonl_path = tmp_path / "samples.jsonl"
    summary_path = tmp_path / "summary.json"
    write_samples_jsonl(samples, jsonl_path)
    write_summary_json(summary, summary_path)
    assert len(jsonl_path.read_text(encoding="utf-8").splitlines()) == 2
    first_record = json.loads(jsonl_path.read_text(encoding="utf-8").splitlines()[0])
    assert first_record["contact_points"][0]["kind"] == "rim"
    assert first_record["trajectory_quat_wxyz"][0] == [1.0, 0.0, 0.0, 0.0]
    assert json.loads(summary_path.read_text(encoding="utf-8"))["samples"] == 2


def test_diagnostic_plots_write_pngs(tmp_path) -> None:
    samples = [_fake_sample(1), _fake_sample(2)]
    paths = [
        tmp_path / "landing.png",
        tmp_path / "catch.png",
        tmp_path / "contacts.png",
        tmp_path / "rim_outcomes.png",
        tmp_path / "side.png",
        tmp_path / "shooter.png",
        tmp_path / "scene3d.png",
    ]
    plot_landing_heatmap(samples, paths[0])
    plot_landing_heatmap(samples, tmp_path / "missed_landing.png", include_made=False)
    plot_catch_heatmap(samples, paths[1], catch_height=2.6)
    plot_contact_heatmaps(samples, paths[2])
    plot_rim_outcomes(samples, paths[3])
    plot_side_trajectories(samples, paths[4])
    plot_shooter_view_trajectories(samples, paths[5])
    plot_3d_scene(samples, paths[6])
    for path in paths:
        assert path.exists()
        assert path.stat().st_size > 0


def test_multi_trajectory_gif_writes(tmp_path) -> None:
    pytest.importorskip("PIL")
    samples = [_fake_sample(1), _fake_sample(2), _fake_sample(3)]
    path = tmp_path / "multi.gif"
    selected = render_typical_shot_gif(samples, path, trajectory_count=2, max_frames=4, fps=8)
    assert path.exists()
    assert path.stat().st_size > 0
    assert len(selected) == 2


def test_mujoco_scene_loads_when_dependency_is_installed() -> None:
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(build_scene_xml())
    assert model.nbody > 0
    assert model.ngeom > 0
