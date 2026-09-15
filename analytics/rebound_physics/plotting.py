from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

from analytics.rebound_physics.model import PhysicsConfig, TrajectorySample, catch_xy_at_height, rim_outcome_label
from analytics.rebound_physics.scale import DEFAULT_BW_THREE_POINT_DISTANCE, DEFAULT_METERS_PER_BW_UNIT
from analytics.rebound_sim.model import CourtSpec, build_court


def _trajectory_array(sample: TrajectorySample) -> np.ndarray:
    return np.asarray(sample.trajectory_xyz, dtype=np.float64)


def _trajectory_quat_array(sample: TrajectorySample) -> np.ndarray:
    return np.asarray(sample.trajectory_quat_wxyz, dtype=np.float64)


def _shot_frame(sample: TrajectorySample) -> tuple[np.ndarray, np.ndarray]:
    origin_xy = np.array([sample.shot.origin.x, sample.shot.origin.y], dtype=np.float64)
    target_xy = np.array([sample.shot.target_x, sample.shot.target_y], dtype=np.float64)
    forward = target_xy - origin_xy
    norm = float(np.linalg.norm(forward))
    if norm < 1e-8:
        forward = np.array([0.0, 1.0], dtype=np.float64)
    else:
        forward = forward / norm
    # Match model._sample_spin: back/top spin is around this shot-relative lateral axis.
    lateral = np.array([forward[1], -forward[0]], dtype=np.float64)
    return forward, lateral


def _contact_array(samples: list[TrajectorySample], kind: str) -> np.ndarray:
    return np.array(
        [[point.x, point.y, point.z] for sample in samples for point in sample.contact_points if point.kind == kind],
        dtype=np.float64,
    )


def _trajectory_subset(samples: list[TrajectorySample], max_samples: int) -> list[TrajectorySample]:
    usable = [sample for sample in samples if sample.trajectory_xyz]
    if len(usable) <= max_samples:
        return usable
    idx = np.linspace(0, len(usable) - 1, max_samples).astype(int)
    return [usable[int(i)] for i in idx]


def _three_point_line_segments_meters(
    *,
    three_point_distance: float,
    three_point_short_distance: float,
    meters_per_bw_unit: float,
) -> list[np.ndarray]:
    radius = float(three_point_distance) * float(meters_per_bw_unit) * np.sqrt(3.0)
    short = float(three_point_short_distance) * float(meters_per_bw_unit) * np.sqrt(3.0)
    if radius <= 0.0:
        return []
    if short <= 0.0 or short >= radius:
        theta = np.linspace(-np.pi, 0.0, 240)
        return [np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])]

    reach = float(np.sqrt(max(radius * radius - short * short, 0.0)))
    line_y = np.linspace(0.0, -reach, 64)
    theta_left = np.arctan2(-reach, -short)
    theta_right = np.arctan2(-reach, short)
    theta = np.linspace(theta_left, theta_right, 240)
    return [
        np.column_stack([np.full_like(line_y, short), line_y]),
        np.column_stack([np.full_like(line_y, -short), line_y]),
        np.column_stack([radius * np.cos(theta), radius * np.sin(theta)]),
    ]


def _plot_three_point_line_meters(
    ax,
    *,
    three_point_distance: float,
    three_point_short_distance: float,
    meters_per_bw_unit: float,
    color: str = "white",
    linewidth: float = 2.0,
    alpha: float = 0.88,
) -> None:
    segments = _three_point_line_segments_meters(
        three_point_distance=three_point_distance,
        three_point_short_distance=three_point_short_distance,
        meters_per_bw_unit=meters_per_bw_unit,
    )
    for idx, segment in enumerate(segments):
        ax.plot(
            segment[:, 0],
            segment[:, 1],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            label="3pt line" if idx == 0 else None,
        )


def _bw_plot_xy_to_mujoco_meters(
    xy: np.ndarray,
    *,
    rim_xy: np.ndarray,
    meters_per_bw_unit: float,
) -> np.ndarray:
    rel_xy = np.asarray(xy, dtype=np.float64) - np.asarray(rim_xy, dtype=np.float64)
    return np.column_stack(
        [
            rel_xy[:, 1] * float(meters_per_bw_unit),
            -rel_xy[:, 0] * float(meters_per_bw_unit),
        ]
    )


def _plot_hex_grid_meters(
    ax,
    *,
    rows: int = 9,
    cols: int = 8,
    three_point_distance: float = DEFAULT_BW_THREE_POINT_DISTANCE,
    three_point_short_distance: float = 3.0,
    meters_per_bw_unit: float = DEFAULT_METERS_PER_BW_UNIT,
    color: str = "white",
    linewidth: float = 0.65,
    alpha: float = 0.24,
) -> None:
    court = build_court(
        CourtSpec(
            rows=int(rows),
            cols=int(cols),
            three_point_distance=float(three_point_distance),
            three_point_short_distance=float(three_point_short_distance),
        )
    )
    angles = np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False) + (np.pi / 6.0)
    unit_hex = np.column_stack([0.98 * np.cos(angles), 0.98 * np.sin(angles)])
    for idx, center in enumerate(court.xy):
        vertices = _bw_plot_xy_to_mujoco_meters(
            center[None, :] + unit_hex,
            rim_xy=court.rim_xy,
            meters_per_bw_unit=meters_per_bw_unit,
        )
        is_basket = idx == int(court.basket_index)
        ax.add_patch(
            Polygon(
                vertices,
                closed=True,
                fill=False,
                edgecolor="#ffcf66" if is_basket else color,
                linewidth=linewidth,
                alpha=min(0.55, alpha + 0.18) if is_basket else alpha,
                zorder=5,
            )
        )


def _plot_three_point_line_3d_meters(
    ax,
    *,
    three_point_distance: float,
    three_point_short_distance: float,
    meters_per_bw_unit: float,
    z: float = 0.025,
    color: str = "white",
    linewidth: float = 2.0,
    alpha: float = 0.88,
) -> None:
    segments = _three_point_line_segments_meters(
        three_point_distance=three_point_distance,
        three_point_short_distance=three_point_short_distance,
        meters_per_bw_unit=meters_per_bw_unit,
    )
    for idx, segment in enumerate(segments):
        ax.plot(
            segment[:, 0],
            segment[:, 1],
            np.full(len(segment), z, dtype=np.float64),
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            label="3pt line" if idx == 0 else None,
        )


def _set_equal_3d_box_aspect(
    ax,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    z_bounds: tuple[float, float],
) -> None:
    x_range = max(float(x_bounds[1] - x_bounds[0]), 1e-6)
    y_range = max(float(y_bounds[1] - y_bounds[0]), 1e-6)
    z_range = max(float(z_bounds[1] - z_bounds[0]), 1e-6)
    ax.set_box_aspect((x_range, y_range, z_range))


def _set_equal_2d_data_aspect(
    ax,
    *,
    x_bounds: tuple[float, float] | None = None,
    y_bounds: tuple[float, float] | None = None,
) -> None:
    if x_bounds is None:
        x_bounds = tuple(float(v) for v in ax.get_xlim())
    if y_bounds is None:
        y_bounds = tuple(float(v) for v in ax.get_ylim())
    x0, x1 = x_bounds
    y0, y1 = y_bounds
    x_span = max(x1 - x0, 1e-6)
    y_span = max(y1 - y0, 1e-6)
    span = max(x_span, y_span)
    x_mid = 0.5 * (x0 + x1)
    y_mid = 0.5 * (y0 + y1)
    ax.set_xlim(x_mid - 0.5 * span, x_mid + 0.5 * span)
    ax.set_ylim(y_mid - 0.5 * span, y_mid + 0.5 * span)
    ax.set_aspect("equal", adjustable="box")


def _plot_shot_origin_projection_3d(
    ax,
    origins: np.ndarray,
    *,
    color: str = "lime",
    alpha: float = 0.58,
) -> None:
    origins = np.asarray(origins, dtype=np.float64).reshape(-1, 3)
    if len(origins) == 0:
        return

    # Many samples share the same shooter location. De-duplicate so the dashed
    # projection remains readable when plotting multiple trajectories.
    unique_origins = np.unique(np.round(origins, decimals=6), axis=0)
    for idx, origin in enumerate(unique_origins):
        x, y, z = (float(origin[0]), float(origin[1]), float(origin[2]))
        ax.plot(
            [x, x],
            [y, y],
            [0.0, z],
            color=color,
            linestyle="--",
            linewidth=1.15,
            alpha=alpha,
            label="shot-origin floor projection" if idx == 0 else None,
        )
    ax.scatter(
        unique_origins[:, 0],
        unique_origins[:, 1],
        np.zeros(len(unique_origins), dtype=np.float64),
        c=color,
        marker="x",
        s=42,
        alpha=min(1.0, alpha + 0.12),
    )


def _square_xy_bounds(points: list[np.ndarray], *, fallback_half_extent: float, min_span: float = 1.0) -> tuple[tuple[float, float], tuple[float, float]]:
    arrays = [np.asarray(point, dtype=np.float64).reshape(-1, 2) for point in points if np.asarray(point).size]
    arrays = [array[np.all(np.isfinite(array), axis=1)] for array in arrays]
    arrays = [array for array in arrays if len(array)]
    if not arrays:
        return (-fallback_half_extent, fallback_half_extent), (-fallback_half_extent, fallback_half_extent)

    combined = np.concatenate(arrays, axis=0)
    mins = np.min(combined, axis=0)
    maxs = np.max(combined, axis=0)
    center = (mins + maxs) * 0.5
    span = max(float(np.max(maxs - mins)), float(min_span))
    pad = max(0.35, span * 0.05)
    half = span * 0.5 + pad
    return (float(center[0] - half), float(center[0] + half)), (float(center[1] - half), float(center[1] + half))


def _three_point_context_points(
    *,
    draw_three_point_line: bool,
    three_point_distance: float,
    three_point_short_distance: float,
    meters_per_bw_unit: float,
) -> list[np.ndarray]:
    if not draw_three_point_line:
        return []
    return _three_point_line_segments_meters(
        three_point_distance=three_point_distance,
        three_point_short_distance=three_point_short_distance,
        meters_per_bw_unit=meters_per_bw_unit,
    )


def plot_landing_heatmap(
    samples: list[TrajectorySample],
    path: str | Path,
    *,
    config: PhysicsConfig | None = None,
    bins: int = 48,
    include_made: bool = True,
    draw_three_point_line: bool = True,
    draw_hex_grid: bool = True,
    three_point_distance: float = DEFAULT_BW_THREE_POINT_DISTANCE,
    three_point_short_distance: float = 3.0,
    meters_per_bw_unit: float = DEFAULT_METERS_PER_BW_UNIT,
) -> None:
    config = config or PhysicsConfig()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    selected_samples = samples if include_made else [sample for sample in samples if not sample.made]
    landings = np.array([s.landing_xy for s in selected_samples if s.landing_xy is not None], dtype=np.float64)
    context_points = [
        landings,
        np.array([[0.0, 0.0]], dtype=np.float64),
        np.array([[-config.backboard_width * 0.5, config.backboard_y], [config.backboard_width * 0.5, config.backboard_y]], dtype=np.float64),
    ]
    context_points.extend(
        _three_point_context_points(
            draw_three_point_line=draw_three_point_line,
            three_point_distance=three_point_distance,
            three_point_short_distance=three_point_short_distance,
            meters_per_bw_unit=meters_per_bw_unit,
        )
    )
    if samples:
        origin = samples[0].shot.origin
        context_points.append(np.array([[origin.x, origin.y]], dtype=np.float64))
    x_bounds, y_bounds = _square_xy_bounds(context_points, fallback_half_extent=config.floor_size * 0.5)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_aspect("equal")
    ax.set_xlim(*x_bounds)
    ax.set_ylim(*y_bounds)
    title_prefix = "MuJoCo rebound landing heatmap" if include_made else "MuJoCo missed-shot landing heatmap"
    ax.set_title(f"{title_prefix} (n={len(landings)})")
    ax.set_xlabel("x meters")
    ax.set_ylabel("y meters")

    if len(landings):
        hist = ax.hist2d(
            landings[:, 0],
            landings[:, 1],
            bins=bins,
            range=[
                [x_bounds[0], x_bounds[1]],
                [y_bounds[0], y_bounds[1]],
            ],
            cmap="inferno",
        )
        fig.colorbar(hist[3], ax=ax, label="landing count")

    ax.scatter([0.0], [0.0], c="white", edgecolors="black", s=80, label="rim")
    ax.plot(
        [-config.backboard_width * 0.5, config.backboard_width * 0.5],
        [config.backboard_y, config.backboard_y],
        color="cyan",
        linewidth=3,
        label="backboard",
    )
    if draw_three_point_line:
        _plot_three_point_line_meters(
            ax,
            three_point_distance=three_point_distance,
            three_point_short_distance=three_point_short_distance,
            meters_per_bw_unit=meters_per_bw_unit,
        )
    if draw_hex_grid:
        _plot_hex_grid_meters(
            ax,
            three_point_distance=three_point_distance,
            three_point_short_distance=three_point_short_distance,
            meters_per_bw_unit=meters_per_bw_unit,
        )
    if samples:
        origin = samples[0].shot.origin
        ax.scatter([origin.x], [origin.y], c="lime", edgecolors="black", s=80, label="shot origin")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)



def plot_catch_heatmap(
    samples: list[TrajectorySample],
    path: str | Path,
    *,
    catch_height: float = 2.6,
    config: PhysicsConfig | None = None,
    bins: int = 48,
    include_made: bool = False,
    draw_three_point_line: bool = True,
    draw_hex_grid: bool = True,
    three_point_distance: float = DEFAULT_BW_THREE_POINT_DISTANCE,
    three_point_short_distance: float = 3.0,
    meters_per_bw_unit: float = DEFAULT_METERS_PER_BW_UNIT,
) -> None:
    config = config or PhysicsConfig()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    selected_samples = samples if include_made else [sample for sample in samples if not sample.made]
    catches = np.array(
        [catch_xy for sample in selected_samples if (catch_xy := catch_xy_at_height(sample, catch_height)) is not None],
        dtype=np.float64,
    )
    context_points = [
        catches,
        np.array([[0.0, 0.0]], dtype=np.float64),
        np.array([[-config.backboard_width * 0.5, config.backboard_y], [config.backboard_width * 0.5, config.backboard_y]], dtype=np.float64),
    ]
    context_points.extend(
        _three_point_context_points(
            draw_three_point_line=draw_three_point_line,
            three_point_distance=three_point_distance,
            three_point_short_distance=three_point_short_distance,
            meters_per_bw_unit=meters_per_bw_unit,
        )
    )
    if samples:
        origin = samples[0].shot.origin
        context_points.append(np.array([[origin.x, origin.y]], dtype=np.float64))
    x_bounds, y_bounds = _square_xy_bounds(context_points, fallback_half_extent=config.floor_size * 0.5)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_aspect("equal")
    ax.set_xlim(*x_bounds)
    ax.set_ylim(*y_bounds)
    title_prefix = "MuJoCo rebound catch-height heatmap" if include_made else "MuJoCo missed-shot catch-height heatmap"
    ax.set_title(f"{title_prefix} z={catch_height:.2f}m (n={len(catches)})")
    ax.set_xlabel("x meters")
    ax.set_ylabel("y meters")

    if len(catches):
        hist = ax.hist2d(
            catches[:, 0],
            catches[:, 1],
            bins=bins,
            range=[
                [x_bounds[0], x_bounds[1]],
                [y_bounds[0], y_bounds[1]],
            ],
            cmap="viridis",
        )
        fig.colorbar(hist[3], ax=ax, label="catch-height count")

    ax.scatter([0.0], [0.0], c="white", edgecolors="black", s=80, label="rim")
    ax.plot(
        [-config.backboard_width * 0.5, config.backboard_width * 0.5],
        [config.backboard_y, config.backboard_y],
        color="cyan",
        linewidth=3,
        label="backboard",
    )
    if draw_three_point_line:
        _plot_three_point_line_meters(
            ax,
            three_point_distance=three_point_distance,
            three_point_short_distance=three_point_short_distance,
            meters_per_bw_unit=meters_per_bw_unit,
        )
    if draw_hex_grid:
        _plot_hex_grid_meters(
            ax,
            three_point_distance=three_point_distance,
            three_point_short_distance=three_point_short_distance,
            meters_per_bw_unit=meters_per_bw_unit,
        )
    if samples:
        origin = samples[0].shot.origin
        ax.scatter([origin.x], [origin.y], c="lime", edgecolors="black", s=80, label="shot origin")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_rim_outcomes(
    samples: list[TrajectorySample],
    path: str | Path,
    *,
    config: PhysicsConfig | None = None,
    draw_three_point_line: bool = True,
    three_point_distance: float = DEFAULT_BW_THREE_POINT_DISTANCE,
    three_point_short_distance: float = 3.0,
    meters_per_bw_unit: float = DEFAULT_METERS_PER_BW_UNIT,
) -> None:
    config = config or PhysicsConfig()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_title("Rim-plane outcomes at downward crossing")
    ax.set_aspect("equal")
    ax.set_xlabel("x meters")
    ax.set_ylabel("y meters")

    groups: dict[str, list[tuple[float, float]]] = {
        "made": [],
        "rim/backboard miss": [],
        "clean miss": [],
    }
    no_crossing_count = 0
    for sample in samples:
        outcome = rim_outcome_label(sample)
        if sample.rim_crossing_xy is None:
            no_crossing_count += 1
            continue
        groups.setdefault(outcome, []).append(sample.rim_crossing_xy)

    all_points = [point for pts in groups.values() for point in pts]
    if draw_three_point_line:
        context_points = [
            np.asarray(all_points, dtype=np.float64) if all_points else np.zeros((0, 2), dtype=np.float64),
            np.array([[0.0, 0.0]], dtype=np.float64),
            np.array([[-config.backboard_width * 0.5, config.backboard_y], [config.backboard_width * 0.5, config.backboard_y]], dtype=np.float64),
        ]
        context_points.extend(
            _three_point_context_points(
                draw_three_point_line=True,
                three_point_distance=three_point_distance,
                three_point_short_distance=three_point_short_distance,
                meters_per_bw_unit=meters_per_bw_unit,
            )
        )
        if samples:
            origin = samples[0].shot.origin
            context_points.append(np.array([[origin.x, origin.y]], dtype=np.float64))
        x_bounds, y_bounds = _square_xy_bounds(context_points, fallback_half_extent=config.floor_size * 0.5)
        ax.set_xlim(*x_bounds)
        ax.set_ylim(*y_bounds)
        _plot_three_point_line_meters(
            ax,
            three_point_distance=three_point_distance,
            three_point_short_distance=three_point_short_distance,
            meters_per_bw_unit=meters_per_bw_unit,
            color="0.35",
            linewidth=1.8,
            alpha=0.72,
        )
        ax.plot(
            [-config.backboard_width * 0.5, config.backboard_width * 0.5],
            [config.backboard_y, config.backboard_y],
            color="tab:cyan",
            linewidth=2.0,
            label="backboard",
        )
        if samples:
            origin = samples[0].shot.origin
            ax.scatter([origin.x], [origin.y], c="lime", edgecolors="black", s=70, label="shot origin")
    else:
        if all_points:
            arr_all = np.asarray(all_points, dtype=np.float64)
            lim = max(config.rim_radius + 0.45, float(np.max(np.abs(arr_all))) * 1.15 + 0.03)
        else:
            lim = config.rim_radius + 0.45
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)

    colors = {
        "made": "tab:green",
        "rim/backboard miss": "tab:orange",
        "clean miss": "tab:red",
    }
    markers = {
        "made": "o",
        "rim/backboard miss": "x",
        "clean miss": "+",
    }
    for name, pts in groups.items():
        if not pts:
            continue
        arr = np.asarray(pts, dtype=np.float64)
        ax.scatter(
            arr[:, 0],
            arr[:, 1],
            s=42,
            c=colors[name],
            marker=markers[name],
            alpha=0.75,
            label=f"{name} (n={len(arr)})",
        )

    theta = np.linspace(0.0, 2.0 * np.pi, 240)
    ax.plot(
        config.rim_radius * np.cos(theta),
        config.rim_radius * np.sin(theta),
        color="black",
        linewidth=2.0,
        label="rim outer edge",
    )
    ax.plot(
        config.make_radius * np.cos(theta),
        config.make_radius * np.sin(theta),
        color="tab:green",
        linestyle="--",
        linewidth=1.5,
        label="make threshold",
    )
    ax.axhline(0.0, color="0.8", linewidth=0.8)
    ax.axvline(0.0, color="0.8", linewidth=0.8)
    if no_crossing_count:
        ax.text(
            0.02,
            0.02,
            f"no downward crossing: {no_crossing_count}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            color="0.25",
        )
    if any(groups.values()):
        ax.legend(loc="upper right", fontsize=8)
    else:
        ax.text(0.5, 0.5, "no rim-plane crossings", ha="center", va="center", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_contact_heatmaps(
    samples: list[TrajectorySample],
    path: str | Path,
    *,
    config: PhysicsConfig | None = None,
    bins: int = 36,
) -> None:
    config = config or PhysicsConfig()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rim = _contact_array(samples, "rim")
    backboard = _contact_array(samples, "backboard")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    rim_ax, board_ax = axes

    rim_ax.set_title(f"Rim contacts, top view (n={len(rim)})")
    rim_ax.set_aspect("equal")
    rim_ax.set_xlabel("x meters")
    rim_ax.set_ylabel("y meters")
    lim = config.rim_radius + 0.2
    rim_ax.set_xlim(-lim, lim)
    rim_ax.set_ylim(-lim, lim)
    if len(rim):
        h = rim_ax.hist2d(rim[:, 0], rim[:, 1], bins=bins, range=[[-lim, lim], [-lim, lim]], cmap="magma")
        fig.colorbar(h[3], ax=rim_ax, label="contacts")
    else:
        rim_ax.text(0.5, 0.5, "no rim contacts", ha="center", va="center", transform=rim_ax.transAxes)
    theta = np.linspace(0.0, 2.0 * np.pi, 160)
    rim_ax.plot(config.rim_radius * np.cos(theta), config.rim_radius * np.sin(theta), color="white", linewidth=2)

    board_ax.set_title(f"Backboard contacts, face view (n={len(backboard)})")
    board_ax.set_aspect("equal")
    board_ax.set_xlabel("x meters")
    board_ax.set_ylabel("z meters")
    board_center_z = config.rim_height + config.backboard_center_z_offset
    board_half_w = config.backboard_width * 0.5
    board_half_h = config.backboard_height * 0.5
    board_ax.set_xlim(-board_half_w * 1.05, board_half_w * 1.05)
    board_ax.set_ylim(board_center_z - board_half_h * 1.15, board_center_z + board_half_h * 1.15)
    if len(backboard):
        h = board_ax.hist2d(
            backboard[:, 0],
            backboard[:, 2],
            bins=bins,
            range=[[-board_half_w, board_half_w], [board_center_z - board_half_h, board_center_z + board_half_h]],
            cmap="viridis",
        )
        fig.colorbar(h[3], ax=board_ax, label="contacts")
    else:
        board_ax.text(0.5, 0.5, "no backboard contacts", ha="center", va="center", transform=board_ax.transAxes)
    board_rect = plt.Rectangle(
        (-board_half_w, board_center_z - board_half_h),
        config.backboard_width,
        config.backboard_height,
        fill=False,
        edgecolor="white",
        linewidth=2,
    )
    board_ax.add_patch(board_rect)
    board_ax.axhline(config.rim_height, color="orange", linestyle="--", linewidth=1.5, label="rim height")
    board_ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_side_trajectories(
    samples: list[TrajectorySample],
    path: str | Path,
    *,
    config: PhysicsConfig | None = None,
    max_samples: int = 40,
) -> None:
    config = config or PhysicsConfig()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    subset = _trajectory_subset(samples, max_samples)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title("Shot trajectories, side view")
    ax.set_xlabel("distance along shot line (meters)")
    ax.set_ylabel("height z (meters)")
    for sample in subset:
        arr = _trajectory_array(sample)
        if len(arr) == 0:
            continue
        forward, _ = _shot_frame(sample)
        origin_xy = np.array([sample.shot.origin.x, sample.shot.origin.y], dtype=np.float64)
        forward_distance = (arr[:, :2] - origin_xy) @ forward
        color = "tab:green" if sample.made else "tab:orange"
        ax.plot(forward_distance, arr[:, 2], color=color, alpha=0.35, linewidth=1.2)
    if samples:
        sample = samples[0]
        forward, _ = _shot_frame(sample)
        origin_xy = np.array([sample.shot.origin.x, sample.shot.origin.y], dtype=np.float64)
        rim_distance = (np.array([0.0, 0.0]) - origin_xy) @ forward
        board_distance = (np.array([0.0, config.backboard_y]) - origin_xy) @ forward
        ax.scatter([rim_distance], [config.rim_height], c="black", s=70, label="rim")
        ax.axvline(board_distance, color="cyan", linestyle="--", linewidth=1.5, label="backboard plane")
    ax.set_ylim(0.0, max(config.rim_height + 1.2, 4.6))
    _set_equal_2d_data_aspect(ax)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_shooter_view_trajectories(
    samples: list[TrajectorySample],
    path: str | Path,
    *,
    config: PhysicsConfig | None = None,
    max_samples: int = 40,
) -> None:
    config = config or PhysicsConfig()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    subset = _trajectory_subset(samples, max_samples)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_title("Shot trajectories from shooter perspective")
    ax.set_xlabel("lateral offset from rim line (meters)")
    ax.set_ylabel("height z (meters)")
    for sample in subset:
        arr = _trajectory_array(sample)
        if len(arr) == 0:
            continue
        _, lateral = _shot_frame(sample)
        origin_xy = np.array([sample.shot.origin.x, sample.shot.origin.y], dtype=np.float64)
        lateral_offset = (arr[:, :2] - origin_xy) @ lateral
        color = "tab:green" if sample.made else "tab:orange"
        ax.plot(lateral_offset, arr[:, 2], color=color, alpha=0.35, linewidth=1.2)
    ax.scatter([0.0], [config.rim_height], c="black", s=70, label="rim center")
    ax.axhline(config.rim_height, color="orange", linestyle="--", linewidth=1.0, label="rim height")
    ax.set_ylim(0.0, max(config.rim_height + 1.2, 4.6))
    _set_equal_2d_data_aspect(ax)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _trajectory_xy_context(samples: list[TrajectorySample]) -> list[np.ndarray]:
    points: list[np.ndarray] = []
    for sample in samples:
        arr = _trajectory_array(sample)
        if len(arr):
            points.append(arr[:, :2])
        points.append(np.array([[sample.shot.origin.x, sample.shot.origin.y]], dtype=np.float64))
        if sample.landing_xy is not None:
            points.append(np.array([sample.landing_xy], dtype=np.float64))
        if sample.settled_xy is not None:
            points.append(np.array([sample.settled_xy], dtype=np.float64))
    return points


def _scene_xy_bounds(
    samples: list[TrajectorySample],
    config: PhysicsConfig,
    *,
    draw_three_point_line: bool = False,
    three_point_distance: float = DEFAULT_BW_THREE_POINT_DISTANCE,
    three_point_short_distance: float = 3.0,
    meters_per_bw_unit: float = DEFAULT_METERS_PER_BW_UNIT,
) -> tuple[tuple[float, float], tuple[float, float]]:
    theta = np.linspace(0.0, 2.0 * np.pi, 160)
    points = _trajectory_xy_context(samples)
    points.extend(
        [
            np.column_stack([config.rim_radius * np.cos(theta), config.rim_radius * np.sin(theta)]),
            np.array(
                [[-config.backboard_width * 0.5, config.backboard_y], [config.backboard_width * 0.5, config.backboard_y]],
                dtype=np.float64,
            ),
            np.array([[0.0, 0.0]], dtype=np.float64),
        ]
    )
    points.extend(
        _three_point_context_points(
            draw_three_point_line=draw_three_point_line,
            three_point_distance=three_point_distance,
            three_point_short_distance=three_point_short_distance,
            meters_per_bw_unit=meters_per_bw_unit,
        )
    )
    return _square_xy_bounds(points, fallback_half_extent=config.floor_size * 0.5, min_span=2.0)


def plot_3d_scene(
    samples: list[TrajectorySample],
    path: str | Path,
    *,
    config: PhysicsConfig | None = None,
    max_samples: int = 24,
    draw_three_point_line: bool = True,
    three_point_distance: float = DEFAULT_BW_THREE_POINT_DISTANCE,
    three_point_short_distance: float = 3.0,
    meters_per_bw_unit: float = DEFAULT_METERS_PER_BW_UNIT,
) -> None:
    config = config or PhysicsConfig()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    subset = _trajectory_subset(samples, max_samples)
    x_bounds, y_bounds = _scene_xy_bounds(
        samples,
        config,
        draw_three_point_line=draw_three_point_line,
        three_point_distance=three_point_distance,
        three_point_short_distance=three_point_short_distance,
        meters_per_bw_unit=meters_per_bw_unit,
    )

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("MuJoCo rebound scene and sampled trajectories")
    floor_x = [x_bounds[0], x_bounds[1], x_bounds[1], x_bounds[0], x_bounds[0]]
    floor_y = [y_bounds[0], y_bounds[0], y_bounds[1], y_bounds[1], y_bounds[0]]
    floor_z = [0.0] * 5
    ax.plot(floor_x, floor_y, floor_z, color="0.35", linewidth=1.5, label="plot bounds")

    theta = np.linspace(0.0, 2.0 * np.pi, 160)
    ax.plot(
        config.rim_radius * np.cos(theta),
        config.rim_radius * np.sin(theta),
        np.full_like(theta, config.rim_height),
        color="orange",
        linewidth=3,
        label="rim",
    )
    board_z = config.rim_height + config.backboard_center_z_offset
    hw = config.backboard_width * 0.5
    hh = config.backboard_height * 0.5
    by = config.backboard_y
    bx = [-hw, hw, hw, -hw, -hw]
    bz = [board_z - hh, board_z - hh, board_z + hh, board_z + hh, board_z - hh]
    ax.plot(bx, [by] * 5, bz, color="black", linewidth=1.5, label="backboard")
    if draw_three_point_line:
        _plot_three_point_line_3d_meters(
            ax,
            three_point_distance=three_point_distance,
            three_point_short_distance=three_point_short_distance,
            meters_per_bw_unit=meters_per_bw_unit,
            z=0.025,
            color="black",
            linewidth=2.0,
            alpha=0.82,
        )

    for sample in subset:
        arr = _trajectory_array(sample)
        if len(arr) == 0:
            continue
        color = "tab:green" if sample.made else "tab:orange"
        ax.plot(arr[:, 0], arr[:, 1], arr[:, 2], color=color, alpha=0.35, linewidth=1.0)
    if samples:
        origins = np.asarray([[sample.shot.origin.x, sample.shot.origin.y, sample.shot.origin.z] for sample in samples], dtype=np.float64)
        ax.scatter(origins[:, 0], origins[:, 1], origins[:, 2], c="lime", edgecolors="black", s=70, label="shot origin")
        _plot_shot_origin_projection_3d(ax, origins)
    z_bounds = (0.0, max(config.rim_height + 1.3, 4.6))
    ax.set_xlim(*x_bounds)
    ax.set_ylim(*y_bounds)
    ax.set_zlim(*z_bounds)
    _set_equal_3d_box_aspect(ax, x_bounds, y_bounds, z_bounds)
    ax.set_xlabel("x lateral (m)")
    ax.set_ylabel("y depth, negative away from rim (m)")
    ax.set_zlabel("z height (m)")
    ax.view_init(elev=22, azim=-62)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _usable_trajectory_samples(samples: list[TrajectorySample]) -> list[TrajectorySample]:
    usable = [sample for sample in samples if len(sample.trajectory_xyz) >= 2]
    if not usable:
        raise ValueError("No samples with trajectories are available for GIF rendering.")
    return usable


def _select_typical_sample(samples: list[TrajectorySample], sample_index: int | None = None) -> TrajectorySample:
    usable = _usable_trajectory_samples(samples)
    if sample_index is not None:
        for sample in usable:
            if sample.shot_index == sample_index:
                return sample
        raise ValueError(f"No sample with shot_index={sample_index} has a recorded trajectory.")
    distances = []
    for sample in usable:
        if sample.landing_xy is not None:
            distances.append(float(np.linalg.norm(np.asarray(sample.landing_xy, dtype=np.float64))))
        elif sample.settled_xy is not None:
            distances.append(float(np.linalg.norm(np.asarray(sample.settled_xy, dtype=np.float64))))
        else:
            distances.append(float(np.linalg.norm(_trajectory_array(sample)[-1, :2])))
    median_distance = float(np.median(distances))
    best_idx = int(np.argmin(np.abs(np.asarray(distances, dtype=np.float64) - median_distance)))
    return usable[best_idx]


def _select_gif_samples(
    samples: list[TrajectorySample],
    *,
    sample_index: int | None,
    trajectory_count: int,
) -> list[TrajectorySample]:
    usable = _usable_trajectory_samples(samples)
    count = max(1, min(int(trajectory_count), len(usable)))
    if count == 1:
        return [_select_typical_sample(samples, sample_index=sample_index)]

    selected: list[TrajectorySample] = []
    if sample_index is not None:
        selected.append(_select_typical_sample(samples, sample_index=sample_index))
        selected_ids = {selected[0].shot_index}
        remaining = [sample for sample in usable if sample.shot_index not in selected_ids]
    else:
        remaining = usable

    need = count - len(selected)
    if need > 0 and remaining:
        idxs = np.linspace(0, len(remaining) - 1, min(need, len(remaining))).astype(int)
        selected.extend(remaining[int(idx)] for idx in idxs)
    return selected


def _gif_resample_indices(length: int, max_frames: int) -> np.ndarray:
    if length <= max_frames:
        return np.arange(length, dtype=int)
    return np.linspace(0, length - 1, max_frames).astype(int)


def _resample_trajectory_for_gif(arr: np.ndarray, max_frames: int) -> np.ndarray:
    if len(arr) == 0:
        return arr
    return arr[_gif_resample_indices(len(arr), max_frames)]


def _quat_wxyz_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if norm < 1e-8:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = quat / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _ball_equator_marker_points(
    center: np.ndarray,
    quat: np.ndarray,
    radius: float,
    *,
    count: int = 10,
) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    local = np.column_stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)]) * float(radius)
    rotation = _quat_wxyz_to_rotation_matrix(quat)
    return np.asarray(center, dtype=np.float64) + local @ rotation.T


def _draw_ball_spin_overlay(
    ax,
    *,
    center: np.ndarray,
    quat: np.ndarray,
    color,
    config: PhysicsConfig,
    spin_mode: str,
    spin_alpha: float,
) -> None:
    if spin_mode == "none":
        return
    spin_alpha = float(np.clip(spin_alpha, 0.0, 1.0))
    marker_xyz = _ball_equator_marker_points(
        center,
        quat,
        config.ball_radius * 0.88,
        count=10,
    )
    marker_loop = np.vstack([marker_xyz, marker_xyz[:1]])
    ax.plot(
        marker_loop[:, 0],
        marker_loop[:, 1],
        marker_loop[:, 2],
        color=color,
        linewidth=0.9,
        alpha=spin_alpha,
    )
    ax.scatter(
        marker_xyz[:1, 0],
        marker_xyz[:1, 1],
        marker_xyz[:1, 2],
        c="black",
        edgecolors=[color],
        s=22,
        alpha=min(1.0, spin_alpha + 0.12),
    )
    if spin_mode == "markers":
        ax.scatter(
            marker_xyz[1:, 0],
            marker_xyz[1:, 1],
            marker_xyz[1:, 2],
            c="white",
            edgecolors=[color],
            s=8,
            alpha=max(0.2, spin_alpha * 0.65),
        )


def _draw_ball_spin_inset(
    fig,
    *,
    quat: np.ndarray,
    color,
    spin_alpha: float,
    forward_axis: np.ndarray | None = None,
) -> None:
    """Draw a zoomed shot-relative back/top-spin orientation inset."""
    spin_alpha = float(np.clip(spin_alpha, 0.0, 1.0))
    rotation = _quat_wxyz_to_rotation_matrix(quat)
    theta = np.linspace(0.0, 2.0 * np.pi, 180)

    # Back/top spin rotates around the shot-relative lateral axis, so the
    # side-view inset must plot shot-forward/z. Plotting lateral/z makes pure
    # backspin look like side spin on diagonal layups.
    forward = np.asarray([0.0, 1.0], dtype=np.float64) if forward_axis is None else np.asarray(forward_axis, dtype=np.float64)
    norm = float(np.linalg.norm(forward))
    if norm < 1e-8:
        forward = np.asarray([0.0, 1.0], dtype=np.float64)
    else:
        forward = forward / norm
    equator_local = np.column_stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)])
    meridian_local = np.column_stack([np.zeros_like(theta), np.cos(theta), np.sin(theta)])
    equator = equator_local @ rotation.T
    meridian = meridian_local @ rotation.T
    dots = np.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64) @ rotation.T

    equator_forward = equator[:, :2] @ forward
    meridian_forward = meridian[:, :2] @ forward
    dots_forward = dots[:, :2] @ forward

    inset_ax = fig.add_axes([0.705, 0.625, 0.22, 0.22])
    inset_ax.set_aspect("equal")
    inset_ax.set_xlim(-1.16, 1.16)
    inset_ax.set_ylim(-1.16, 1.16)
    inset_ax.set_facecolor((1.0, 1.0, 1.0, 0.82))
    inset_ax.set_title("back/top spin", fontsize=8, pad=2)
    inset_ax.set_xticks([])
    inset_ax.set_yticks([])
    for spine in inset_ax.spines.values():
        spine.set_color("0.25")
        spine.set_linewidth(0.8)

    ball = plt.Circle(
        (0.0, 0.0),
        1.0,
        facecolor=(1.0, 1.0, 1.0, 0.78),
        edgecolor="black",
        linewidth=1.1,
        zorder=1,
    )
    inset_ax.add_patch(ball)
    inset_ax.plot(
        equator_forward,
        equator[:, 2],
        color=color,
        linewidth=1.6,
        alpha=spin_alpha,
        zorder=2,
    )
    inset_ax.plot(
        meridian_forward,
        meridian[:, 2],
        color="0.15",
        linewidth=0.9,
        alpha=max(0.25, spin_alpha * 0.55),
        zorder=2,
    )
    inset_ax.scatter(
        dots_forward[:1],
        dots[:1, 2],
        c="black",
        edgecolors=[color],
        linewidths=0.8,
        s=30,
        zorder=3,
    )
    inset_ax.scatter(
        dots_forward[1:],
        dots[1:, 2],
        c=[color],
        edgecolors="black",
        linewidths=0.7,
        s=24,
        zorder=3,
    )



def _draw_rim_closeup_inset(
    fig,
    *,
    trajectories: list[np.ndarray],
    quaternions: list[np.ndarray],
    colors: np.ndarray,
    frame_i: int,
    frame_count: int,
    config: PhysicsConfig,
    spin_alpha: float,
) -> None:
    """Draw a zoomed top-down inset around the rim/backboard interaction."""
    inset_ax = fig.add_axes([0.685, 0.09, 0.24, 0.24])
    inset_ax.set_aspect("equal")
    inset_ax.set_facecolor((1.0, 1.0, 1.0, 0.84))
    inset_ax.set_title("rim closeup", fontsize=8, pad=2)
    inset_ax.set_xticks([])
    inset_ax.set_yticks([])
    for spine in inset_ax.spines.values():
        spine.set_color("0.25")
        spine.set_linewidth(0.8)

    theta = np.linspace(0.0, 2.0 * np.pi, 180)
    rim_x = config.rim_radius * np.cos(theta)
    rim_y = config.rim_radius * np.sin(theta)
    make_x = config.make_radius * np.cos(theta)
    make_y = config.make_radius * np.sin(theta)
    inset_ax.plot(rim_x, rim_y, color="orange", linewidth=2.0)
    inset_ax.plot(make_x, make_y, color="tab:green", linestyle="--", linewidth=1.0, alpha=0.9)
    inset_ax.plot(
        [-config.backboard_width * 0.5, config.backboard_width * 0.5],
        [config.backboard_y, config.backboard_y],
        color="black",
        linewidth=1.5,
    )

    view_half = max(0.72, config.rim_radius + config.ball_radius + 0.34)
    inset_ax.set_xlim(-view_half, view_half)
    inset_ax.set_ylim(min(config.backboard_y - 0.10, -view_half), view_half)
    inset_ax.axhline(0.0, color="0.75", linewidth=0.6)
    inset_ax.axvline(0.0, color="0.75", linewidth=0.6)

    for idx, arr in enumerate(trajectories):
        if len(arr) == 0:
            continue
        if frame_count <= 1:
            sample_frame = 0
        else:
            sample_frame = int(round(frame_i * (len(arr) - 1) / (frame_count - 1)))
        point = arr[sample_frame]
        color = colors[idx]
        trail_start = max(0, sample_frame - 18)
        trail = arr[trail_start : sample_frame + 1, :2]
        inset_ax.plot(trail[:, 0], trail[:, 1], color=color, linewidth=1.7, alpha=0.78)

        ball = plt.Circle(
            (float(point[0]), float(point[1])),
            config.ball_radius,
            facecolor=color,
            edgecolor="black",
            linewidth=0.75,
            alpha=0.82,
            zorder=3,
        )
        inset_ax.add_patch(ball)

        quat_arr = quaternions[idx] if idx < len(quaternions) else np.zeros((0, 4), dtype=np.float64)
        if len(quat_arr) > sample_frame:
            marker_xyz = _ball_equator_marker_points(
                np.asarray(point, dtype=np.float64),
                quat_arr[sample_frame],
                config.ball_radius * 0.82,
                count=18,
            )
            marker_loop = np.vstack([marker_xyz, marker_xyz[:1]])
            inset_ax.plot(
                marker_loop[:, 0],
                marker_loop[:, 1],
                color="black",
                linewidth=0.8,
                alpha=max(0.32, spin_alpha * 0.58),
                zorder=4,
            )
            inset_ax.scatter(
                marker_xyz[:1, 0],
                marker_xyz[:1, 1],
                c="white",
                edgecolors="black",
                linewidths=0.6,
                s=14,
                alpha=max(0.42, spin_alpha),
                zorder=5,
            )



def _draw_rim_front_closeup_inset(
    fig,
    *,
    trajectories: list[np.ndarray],
    quaternions: list[np.ndarray],
    colors: np.ndarray,
    frame_i: int,
    frame_count: int,
    config: PhysicsConfig,
    spin_alpha: float,
) -> None:
    """Draw a zoomed front view of the backboard/rim interaction."""
    inset_ax = fig.add_axes([0.075, 0.09, 0.24, 0.24])
    inset_ax.set_aspect("equal")
    inset_ax.set_facecolor((1.0, 1.0, 1.0, 0.84))
    inset_ax.set_title("front closeup", fontsize=8, pad=2)
    inset_ax.set_xticks([])
    inset_ax.set_yticks([])
    for spine in inset_ax.spines.values():
        spine.set_color("0.25")
        spine.set_linewidth(0.8)

    board_center_z = config.rim_height + config.backboard_center_z_offset
    board_half_w = config.backboard_width * 0.5
    board_half_h = config.backboard_height * 0.5
    board_rect = plt.Rectangle(
        (-board_half_w, board_center_z - board_half_h),
        config.backboard_width,
        config.backboard_height,
        fill=False,
        edgecolor="black",
        linewidth=1.4,
    )
    inset_ax.add_patch(board_rect)

    target_box_h = max(2.0 * config.backboard_box_center_z_offset, 1e-6)
    target_rect = plt.Rectangle(
        (-config.backboard_box_width * 0.5, config.rim_height),
        config.backboard_box_width,
        target_box_h,
        fill=False,
        edgecolor="tab:orange",
        linewidth=1.0,
        linestyle="--",
        alpha=0.85,
    )
    inset_ax.add_patch(target_rect)
    inset_ax.plot(
        [-config.rim_radius, config.rim_radius],
        [config.rim_height, config.rim_height],
        color="orange",
        linewidth=2.2,
    )
    inset_ax.plot(
        [-config.make_radius, config.make_radius],
        [config.rim_height, config.rim_height],
        color="tab:green",
        linewidth=1.1,
        linestyle="--",
        alpha=0.9,
    )

    x_half = max(board_half_w + 0.12, config.rim_radius + config.ball_radius + 0.42)
    z_min = config.rim_height - 0.62
    z_max = board_center_z + board_half_h + 0.14
    inset_ax.set_xlim(-x_half, x_half)
    inset_ax.set_ylim(z_min, z_max)
    inset_ax.axvline(0.0, color="0.78", linewidth=0.6)

    for idx, arr in enumerate(trajectories):
        if len(arr) == 0:
            continue
        if frame_count <= 1:
            sample_frame = 0
        else:
            sample_frame = int(round(frame_i * (len(arr) - 1) / (frame_count - 1)))
        point = arr[sample_frame]
        color = colors[idx]
        trail_start = max(0, sample_frame - 18)
        trail = arr[trail_start : sample_frame + 1]
        inset_ax.plot(trail[:, 0], trail[:, 2], color=color, linewidth=1.7, alpha=0.78)

        ball = plt.Circle(
            (float(point[0]), float(point[2])),
            config.ball_radius,
            facecolor=color,
            edgecolor="black",
            linewidth=0.75,
            alpha=0.82,
            zorder=3,
        )
        inset_ax.add_patch(ball)

        quat_arr = quaternions[idx] if idx < len(quaternions) else np.zeros((0, 4), dtype=np.float64)
        if len(quat_arr) > sample_frame:
            marker_xyz = _ball_equator_marker_points(
                np.asarray(point, dtype=np.float64),
                quat_arr[sample_frame],
                config.ball_radius * 0.82,
                count=18,
            )
            marker_loop = np.vstack([marker_xyz, marker_xyz[:1]])
            inset_ax.plot(
                marker_loop[:, 0],
                marker_loop[:, 2],
                color="black",
                linewidth=0.8,
                alpha=max(0.32, spin_alpha * 0.58),
                zorder=4,
            )
            inset_ax.scatter(
                marker_xyz[:1, 0],
                marker_xyz[:1, 2],
                c="white",
                edgecolors="black",
                linewidths=0.6,
                s=14,
                alpha=max(0.42, spin_alpha),
                zorder=5,
            )


def render_typical_shot_gif(
    samples: list[TrajectorySample],
    path: str | Path,
    *,
    config: PhysicsConfig | None = None,
    sample_index: int | None = None,
    fps: int = 18,
    max_frames: int = 120,
    trajectory_count: int = 1,
    spin_mode: str = "none",
    spin_inset: bool = True,
    rim_inset: bool = True,
    spin_primary_only: bool = True,
    spin_alpha: float = 0.75,
    draw_three_point_line: bool = True,
    three_point_distance: float = DEFAULT_BW_THREE_POINT_DISTANCE,
    three_point_short_distance: float = 3.0,
    meters_per_bw_unit: float = DEFAULT_METERS_PER_BW_UNIT,
) -> tuple[TrajectorySample, ...]:
    """Render one or more representative simulated shots as a 3D animated GIF."""
    import imageio.v2 as imageio

    if spin_mode not in {"none", "seam", "markers"}:
        raise ValueError(f"Unsupported spin_mode={spin_mode!r}; expected none, seam, or markers.")
    config = config or PhysicsConfig()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    selected = _select_gif_samples(samples, sample_index=sample_index, trajectory_count=trajectory_count)
    raw_trajectories = [_trajectory_array(sample) for sample in selected]
    resample_indices = [_gif_resample_indices(len(arr), max_frames) for arr in raw_trajectories]
    trajectories = [arr[idx] for arr, idx in zip(raw_trajectories, resample_indices, strict=False)]
    raw_quaternions = [_trajectory_quat_array(sample) for sample in selected]
    quaternions: list[np.ndarray] = []
    for arr, quat, idx in zip(raw_trajectories, raw_quaternions, resample_indices, strict=False):
        if len(quat) == len(arr) and len(idx):
            quaternions.append(quat[idx])
        else:
            quaternions.append(np.zeros((0, 4), dtype=np.float64))
    frame_count = max(len(arr) for arr in trajectories)
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(1, len(selected))))

    x_bounds, y_bounds = _scene_xy_bounds(
        selected,
        config,
        draw_three_point_line=draw_three_point_line,
        three_point_distance=three_point_distance,
        three_point_short_distance=three_point_short_distance,
        meters_per_bw_unit=meters_per_bw_unit,
    )
    theta = np.linspace(0.0, 2.0 * np.pi, 160)
    rim_x = config.rim_radius * np.cos(theta)
    rim_y = config.rim_radius * np.sin(theta)
    rim_z = np.full_like(theta, config.rim_height)
    board_z = config.rim_height + config.backboard_center_z_offset
    hw = config.backboard_width * 0.5
    hh = config.backboard_height * 0.5
    by = config.backboard_y
    board_x = [-hw, hw, hw, -hw, -hw]
    board_zs = [board_z - hh, board_z - hh, board_z + hh, board_z + hh, board_z - hh]
    contact_arrays = [
        np.asarray([[p.x, p.y, p.z] for p in sample.contact_points if p.kind in {"rim", "backboard"}], dtype=np.float64)
        for sample in selected
    ]

    frames: list[np.ndarray] = []
    for frame_i in range(frame_count):
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_title(f"{len(selected)} simulated shot trajectories")
        floor_x = [x_bounds[0], x_bounds[1], x_bounds[1], x_bounds[0], x_bounds[0]]
        floor_y = [y_bounds[0], y_bounds[0], y_bounds[1], y_bounds[1], y_bounds[0]]
        ax.plot(floor_x, floor_y, [0.0] * 5, color="0.35", linewidth=1.2)
        ax.plot(rim_x, rim_y, rim_z, color="orange", linewidth=3)
        ax.plot(board_x, [by] * 5, board_zs, color="black", linewidth=1.5)
        if draw_three_point_line:
            _plot_three_point_line_3d_meters(
                ax,
                three_point_distance=three_point_distance,
                three_point_short_distance=three_point_short_distance,
                meters_per_bw_unit=meters_per_bw_unit,
                z=0.025,
                color="black",
                linewidth=1.8,
                alpha=0.82,
            )
        origins = np.asarray([[sample.shot.origin.x, sample.shot.origin.y, sample.shot.origin.z] for sample in selected], dtype=np.float64)
        ax.scatter(origins[:, 0], origins[:, 1], origins[:, 2], c="lime", s=42, edgecolors="black", label="shot origins")
        _plot_shot_origin_projection_3d(ax, origins)
        inset_quat: np.ndarray | None = None
        inset_forward_axis: np.ndarray | None = None
        for idx, (sample, arr, quat_arr, contact_xyz) in enumerate(
            zip(selected, trajectories, quaternions, contact_arrays, strict=False)
        ):
            if frame_count <= 1:
                sample_frame = 0
            else:
                sample_frame = int(round(frame_i * (len(arr) - 1) / (frame_count - 1)))
            point = arr[sample_frame]
            color = colors[idx]
            outcome = "make" if sample.made else "miss"
            label = f"{sample.shot_index}: {outcome}, {sample.first_contact or 'none'}"
            ax.plot(
                arr[: sample_frame + 1, 0],
                arr[: sample_frame + 1, 1],
                arr[: sample_frame + 1, 2],
                color=color,
                linewidth=1.8,
                alpha=0.75,
                label=label,
            )
            trail_start = max(0, sample_frame - 12)
            ax.plot(
                arr[trail_start : sample_frame + 1, 0],
                arr[trail_start : sample_frame + 1, 1],
                arr[trail_start : sample_frame + 1, 2],
                color=color,
                linewidth=3.2,
                alpha=0.95,
            )
            ax.scatter([point[0]], [point[1]], [point[2]], c=[color], edgecolors="black", s=90)
            has_quat = len(quat_arr) > sample_frame
            if spin_inset and idx == 0 and has_quat:
                inset_quat = quat_arr[sample_frame]
                inset_forward_axis, _ = _shot_frame(sample)
            draw_spin = spin_mode != "none" and has_quat and (not spin_primary_only or idx == 0)
            if draw_spin:
                _draw_ball_spin_overlay(
                    ax,
                    center=point,
                    quat=quat_arr[sample_frame],
                    color=color,
                    config=config,
                    spin_mode=spin_mode,
                    spin_alpha=spin_alpha,
                )
            if len(contact_xyz):
                ax.scatter(contact_xyz[:, 0], contact_xyz[:, 1], contact_xyz[:, 2], c=[color], s=18, alpha=0.35)
        z_bounds = (0.0, max(config.rim_height + 1.3, 4.6))
        ax.set_xlim(*x_bounds)
        ax.set_ylim(*y_bounds)
        ax.set_zlim(*z_bounds)
        _set_equal_3d_box_aspect(ax, x_bounds, y_bounds, z_bounds)
        ax.set_xlabel("x lateral")
        ax.set_ylabel("y depth")
        ax.set_zlabel("z height")
        ax.view_init(elev=22, azim=-62)
        ax.legend(loc="upper left", fontsize=8)
        fig.tight_layout()
        if spin_inset and inset_quat is not None:
            _draw_ball_spin_inset(
                fig,
                quat=inset_quat,
                color=colors[0],
                spin_alpha=spin_alpha,
                forward_axis=inset_forward_axis,
            )
        if rim_inset:
            _draw_rim_closeup_inset(
                fig,
                trajectories=trajectories,
                quaternions=quaternions,
                colors=colors,
                frame_i=frame_i,
                frame_count=frame_count,
                config=config,
                spin_alpha=spin_alpha,
            )
            _draw_rim_front_closeup_inset(
                fig,
                trajectories=trajectories,
                quaternions=quaternions,
                colors=colors,
                frame_i=frame_i,
                frame_count=frame_count,
                config=config,
                spin_alpha=spin_alpha,
            )
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
        plt.close(fig)

    imageio.mimsave(path, frames, duration=1.0 / max(1, fps), loop=0)
    return tuple(selected)
