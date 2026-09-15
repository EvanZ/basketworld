from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps as mpl_cmaps
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch, RegularPolygon

from .model import Court, ReboundSimulationResult

FLOW_REGION_ORDER = (
    "right_baseline",
    "right_wing",
    "top_of_key",
    "paint",
    "left_wing",
    "left_baseline",
)
FLOW_REGION_DISPLAY = {
    "left_baseline": "Left baseline",
    "right_baseline": "Right baseline",
    "paint": "Paint",
    "left_wing": "Left wing",
    "right_wing": "Right wing",
    "top_of_key": "Top of key",
}
FLOW_REGION_COLORS = {
    "left_baseline": "#ff9f1c",
    "right_baseline": "#00a8ff",
    "paint": "#ef476f",
    "left_wing": "#06d6a0",
    "right_wing": "#118ab2",
    "top_of_key": "#ffd166",
}


def plot_rebound_summary(result: ReboundSimulationResult, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    court = result.court
    shot_counts = _counts_by_cell(court, result.shot_indices)
    target_counts = _counts_by_cell(court, result.target_indices)
    oreb_target_counts = _counts_by_cell(court, result.target_indices[result.winner_is_offense])
    target_total = np.maximum(target_counts, 1)
    oreb_rate_by_target = np.where(target_counts > 0, oreb_target_counts / target_total, np.nan)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    _plot_hex_values(court, axes[0, 0], shot_counts, "Shot locations", "Blues", integer=True)
    _plot_hex_values(court, axes[0, 1], target_counts, "Rebound landing targets", "Oranges", integer=True)
    _plot_hex_values(
        court,
        axes[1, 0],
        oreb_target_counts,
        "Offensive rebound targets",
        "Greens",
        integer=True,
    )
    _plot_hex_values(
        court,
        axes[1, 1],
        oreb_rate_by_target,
        "OREB% by landing target",
        "RdYlGn",
        vmin=0.0,
        vmax=1.0,
        integer=False,
    )
    summary = result.summary()
    fig.suptitle(
        "BW Rebound Prototype | "
        f"samples={summary['samples']} | "
        f"OREB={summary['offensive_rebound_rate']:.1%} | "
        f"winner dist={summary['mean_rebound_winner_distance']:.2f}",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_conditioned_rebound_heatmap(
    result: ReboundSimulationResult,
    path: str | Path,
    *,
    shot_cell: tuple[int, int] | None = None,
) -> None:
    """Plot rebound landing density for a fixed shot cell simulation."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    court = result.court
    target_counts = _counts_by_cell(court, result.target_indices)
    shot_idx: int | None = None
    if result.shot_indices.size:
        unique_shots = np.unique(result.shot_indices.astype(np.int64))
        shot_idx = int(unique_shots[0])
        if shot_cell is None:
            shot_cell = court.cells[shot_idx]
    summary = result.summary()
    cell_label = f"q={shot_cell[0]}, r={shot_cell[1]}" if shot_cell is not None else "unknown shot"
    title = (
        f"Rebound landing targets conditioned on shot {cell_label}\n"
        f"samples={summary['samples']} | OREB={summary['offensive_rebound_rate']:.1%} | "
        f"winner dist={summary['mean_rebound_winner_distance']:.2f}"
    )

    fig, ax = plt.subplots(figsize=(9, 8))
    _plot_hex_values(court, ax, target_counts, title, "Oranges", integer=True)
    if shot_idx is not None:
        x, y = court.xy[shot_idx]
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


def plot_rebound_region_flow(result: ReboundSimulationResult, path: str | Path, *, max_flows: int = 20) -> None:
    """Draw source shot-region to rebound-region flows over the hex court.

    This is intentionally a spatial Sankey-style overlay rather than a standard
    rectangular Sankey: region centroids stay on the court so the shot/rebound
    relationship remains visually interpretable.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    court = result.court
    source_regions = classify_rebound_flow_regions(court, result.shot_indices)
    target_regions = classify_rebound_flow_regions(court, result.target_indices)
    region_to_idx = {region: idx for idx, region in enumerate(FLOW_REGION_ORDER)}
    flow_counts = np.zeros((len(FLOW_REGION_ORDER), len(FLOW_REGION_ORDER)), dtype=np.float64)
    for source, target in zip(source_regions, target_regions, strict=True):
        flow_counts[region_to_idx[str(source)], region_to_idx[str(target)]] += 1.0

    fig, ax = plt.subplots(figsize=(12, 10))
    _draw_court_base(court, ax, title="Shot region -> rebound region flow")
    centroids = _region_centroids(court)
    source_anchors, target_anchors = _flow_region_anchors(court, centroids)
    total = float(np.sum(flow_counts)) or 1.0
    source_totals = np.maximum(np.sum(flow_counts, axis=1), 1.0)
    entries: list[tuple[float, int, int]] = []
    for i in range(flow_counts.shape[0]):
        for j in range(flow_counts.shape[1]):
            count = float(flow_counts[i, j])
            if count > 0.0:
                entries.append((count, i, j))
    entries.sort(reverse=True)
    entries = entries[: max(1, int(max_flows))]
    max_count = entries[0][0] if entries else 1.0

    for rank, (count, source_idx, target_idx) in enumerate(reversed(entries)):
        source = FLOW_REGION_ORDER[source_idx]
        target = FLOW_REGION_ORDER[target_idx]
        color = FLOW_REGION_COLORS[source]
        width = 0.8 + 7.0 * math.sqrt(count / max_count)
        alpha = 0.28 + 0.42 * math.sqrt(count / max_count)
        source_xy = source_anchors[source]
        target_xy = target_anchors[target]
        conditional = count / float(source_totals[source_idx])
        share = count / total
        if source_idx == target_idx:
            rad = 0.46
        else:
            rad = 0.18 if (source_idx + target_idx) % 2 == 0 else -0.18
        arrow = FancyArrowPatch(
            tuple(source_xy),
            tuple(target_xy),
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>",
            mutation_scale=10.0 + width,
            linewidth=width,
            color=color,
            alpha=alpha,
            zorder=4,
        )
        ax.add_patch(arrow)
        if rank >= len(entries) - 12:
            mid = (source_xy + target_xy) / 2.0
            ax.text(
                float(mid[0]),
                float(mid[1]),
                f"{conditional:.0%}",
                ha="center",
                va="center",
                fontsize=8,
                color="#111111",
                bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "alpha": 0.72, "edgecolor": "none"},
                zorder=8,
            )
        if share >= 0.10:
            ax.plot([source_xy[0]], [source_xy[1]], marker="o", markersize=4, color=color, zorder=9)

    for region in FLOW_REGION_ORDER:
        color = FLOW_REGION_COLORS[region]
        label_xy = centroids[region]
        source_xy = source_anchors[region]
        target_xy = target_anchors[region]
        ax.scatter(
            [source_xy[0]],
            [source_xy[1]],
            marker="s",
            s=210,
            facecolor=color,
            edgecolor="#111111",
            linewidth=1.1,
            alpha=0.96,
            zorder=10,
        )
        ax.scatter(
            [target_xy[0]],
            [target_xy[1]],
            marker="o",
            s=230,
            facecolor="white",
            edgecolor=color,
            linewidth=2.0,
            alpha=0.96,
            zorder=10,
        )
        ax.text(float(source_xy[0]), float(source_xy[1]), "S", ha="center", va="center", fontsize=8, fontweight="bold", color="#111111", zorder=11)
        ax.text(float(target_xy[0]), float(target_xy[1]), "R", ha="center", va="center", fontsize=8, fontweight="bold", color="#111111", zorder=11)
        ax.text(
            float(label_xy[0]),
            float(label_xy[1] + _region_label_y_offset(region)),
            FLOW_REGION_DISPLAY[region].replace(" ", "\n"),
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color="#111111",
            bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "alpha": 0.68, "edgecolor": "none"},
            zorder=11,
        )

    summary = result.summary()
    ax.text(
        0.01,
        0.02,
        f"samples={summary['samples']}  |  top {len(entries)} region flows  |  S=sources, R=rebound targets  |  labels=P(rebound region | shot region)",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color="#333333",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.82, "edgecolor": "#cccccc"},
        zorder=20,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def classify_rebound_flow_regions(court: Court, indices: np.ndarray) -> np.ndarray:
    """Map cells to coarse basketball regions for rebound flow visualization.

    Baseline means the basket-side short-corner/baseline area, not the full
    upper/lower edge of the rectangular plot. The current half-court UI labels
    the upper side as right and lower side as left, so y < rim_y maps to right
    regions and y > rim_y maps to left regions.
    """

    indices = np.asarray(indices, dtype=np.int64)
    xy = court.xy[indices]
    dx = xy[:, 0] - court.rim_xy[0]
    dy = xy[:, 1] - court.rim_xy[1]
    abs_dy = np.abs(dy)
    dist = court.distance_hex[indices]
    labels = np.full(indices.shape, "top_of_key", dtype=object)

    # Keep this simple and court-relative: the paint hugs the rim, baseline is
    # short-corner depth near the basket side, and wings are the side perimeter.
    baseline_depth = max(3.5, 0.30 * float(np.max(court.xy[:, 0]) - court.rim_xy[0]))
    paint = dist <= 2.35
    baseline = (~paint) & (dx <= baseline_depth) & (abs_dy >= 0.48 * court.vertical_extent)
    middle = (~paint) & (~baseline) & (abs_dy <= 0.30 * court.vertical_extent)
    left_side = dy > 0.0
    right_side = dy < 0.0
    wing = (~paint) & (~baseline) & (~middle)

    labels[paint] = "paint"
    labels[baseline & left_side] = "left_baseline"
    labels[baseline & right_side] = "right_baseline"
    labels[wing & left_side] = "left_wing"
    labels[wing & right_side] = "right_wing"
    labels[middle] = "top_of_key"
    return labels


def _counts_by_cell(court: Court, indices: np.ndarray) -> np.ndarray:
    counts = np.zeros(len(court.cells), dtype=np.float64)
    if indices.size:
        bincount = np.bincount(indices.astype(np.int64), minlength=len(court.cells))
        counts[: len(bincount)] = bincount[: len(counts)]
    return counts


def _region_centroids(court: Court) -> dict[str, np.ndarray]:
    all_regions = classify_rebound_flow_regions(court, np.arange(len(court.cells), dtype=np.int64))
    centroids: dict[str, np.ndarray] = {}
    for region in FLOW_REGION_ORDER:
        mask = all_regions == region
        if np.any(mask):
            centroids[region] = np.mean(court.xy[mask], axis=0)
        else:
            centroids[region] = np.mean(court.xy, axis=0)
    return centroids


def _flow_region_anchors(
    court: Court,
    centroids: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    span_x = float(np.max(court.xy[:, 0]) - np.min(court.xy[:, 0])) or 1.0
    offset = np.asarray([max(0.35, 0.035 * span_x), 0.0], dtype=np.float64)
    source_anchors = {region: xy + offset for region, xy in centroids.items()}
    target_anchors = {region: xy - offset for region, xy in centroids.items()}
    return source_anchors, target_anchors


def _region_label_y_offset(region: str) -> float:
    if region == "right_baseline":
        return -0.88
    if region == "left_baseline":
        return 0.88
    return 0.0



def _draw_court_base(court: Court, ax, title: str) -> None:
    ax.set_aspect("equal")
    ax.set_title(title)
    for idx, _cell in enumerate(court.cells):
        x, y = court.xy[idx]
        edge = "#aaaaaa"
        linewidth = 0.8
        if idx == court.basket_index:
            edge = "#ff9f1c"
            linewidth = 1.6
        elif court.three_point_mask[idx]:
            edge = "#8fd3ff"
        patch = RegularPolygon(
            (float(x), float(y)),
            numVertices=6,
            radius=0.98,
            orientation=0,
            facecolor=(0.93, 0.95, 0.98, 0.45),
            edgecolor=edge,
            linewidth=linewidth,
            zorder=1,
        )
        ax.add_patch(patch)
    xs = court.xy[:, 0]
    ys = court.xy[:, 1]
    ax.set_xlim(float(xs.min() - 1.5), float(xs.max() + 1.5))
    ax.set_ylim(float(ys.max() + 1.5), float(ys.min() - 1.5))
    ax.axis("off")


def _plot_hex_values(
    court: Court,
    ax,
    values: np.ndarray,
    title: str,
    cmap_name: str,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    integer: bool,
) -> None:
    ax.set_aspect("equal")
    ax.set_title(title)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        finite = np.asarray([0.0])
    if vmin is None:
        vmin = float(np.nanmin(finite))
    if vmax is None:
        vmax = float(np.nanmax(finite)) or 1.0
    if math.isclose(vmin, vmax):
        vmax = vmin + 1.0
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = mpl_cmaps.get_cmap(cmap_name)

    for idx, ((_q, _r), value) in enumerate(zip(court.cells, values, strict=True)):
        x, y = court.xy[idx]
        if np.isfinite(value):
            color = cmap(norm(float(value))) if float(value) > 0.0 else (0.90, 0.90, 0.92, 0.45)
        else:
            color = (0.90, 0.90, 0.92, 0.25)
        edge = "#222222"
        if idx == court.basket_index:
            edge = "#ff9f1c"
        elif court.three_point_mask[idx]:
            edge = "#00a8ff"
        patch = RegularPolygon(
            (float(x), float(y)),
            numVertices=6,
            radius=0.98,
            orientation=0,
            facecolor=color,
            edgecolor=edge,
            linewidth=1.0,
        )
        ax.add_patch(patch)
        if np.isfinite(value) and float(value) > 0.0:
            label = f"{int(value)}" if integer else f"{float(value):.0%}"
            ax.text(float(x), float(y), label, ha="center", va="center", fontsize=7)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
    xs = court.xy[:, 0]
    ys = court.xy[:, 1]
    ax.set_xlim(float(xs.min() - 1.5), float(xs.max() + 1.5))
    ax.set_ylim(float(ys.max() + 1.5), float(ys.min() - 1.5))
    ax.axis("off")
