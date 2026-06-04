from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from basketworld.envs.core import geometry

ReboundKind = Literal["short", "normal", "long"]
ReboundRegion = Literal["weak", "same", "middle"]


@dataclass(frozen=True)
class CourtSpec:
    """Hex court geometry used by the current JAX half-court training setup."""

    rows: int = 9
    cols: int = 8
    basket_col: int = 0
    basket_row: int | None = None
    three_point_distance: float = 4.25
    three_point_short_distance: float = 3.0


@dataclass(frozen=True)
class ReboundParams:
    """Tunable rebound model parameters.

    Distances are expressed in BW hex-distance units unless noted otherwise.
    The model is intentionally simple: sample a rebound landing hex, then
    sample a rebound winner from distance/skill/position logits.
    """

    shot_rim_weight: float = 0.16
    shot_paint_weight: float = 0.24
    shot_midrange_weight: float = 0.20
    shot_three_weight: float = 0.40
    shot_rim_center: float = 1.0
    shot_paint_center: float = 2.0
    shot_midrange_center: float = 3.4
    shot_three_center: float = 4.8
    shot_rim_sigma: float = 0.65
    shot_paint_sigma: float = 0.75
    shot_midrange_sigma: float = 0.95
    shot_three_sigma: float = 0.85

    short_rebound_base_hex: float = 1.0
    normal_rebound_base_hex: float = 2.0
    long_rebound_base_hex: float = 3.8
    short_rebound_slope: float = 0.10
    normal_rebound_slope: float = 0.18
    long_rebound_slope: float = 0.28
    short_rebound_sigma_hex: float = 0.90
    normal_rebound_sigma_hex: float = 1.10
    long_rebound_sigma_hex: float = 1.35
    lateral_bias_hex: float = 1.15

    centerline_deadband: float = 0.12
    short_weak_prob: float = 0.56
    short_same_prob: float = 0.25
    long_weak_prob: float = 0.22
    long_same_prob: float = 0.47

    defense_rebound_bias: float = 0.55
    boxout_bias: float = 0.35
    inside_position_weight: float = 0.24
    target_distance_weight: float = 1.10
    rebound_skill_weight: float = 0.35
    rebound_skill_std: float = 0.35
    winner_temperature: float = 0.75


@dataclass(frozen=True)
class Court:
    spec: CourtSpec
    cells: tuple[tuple[int, int], ...]
    offsets: tuple[tuple[int, int], ...]
    xy: np.ndarray
    basket_axial: tuple[int, int]
    basket_index: int
    distance_hex: np.ndarray
    distance_cart: np.ndarray
    three_point_mask: np.ndarray
    hex_distance_lut: np.ndarray
    cell_index: dict[tuple[int, int], int]

    @property
    def rim_xy(self) -> np.ndarray:
        return self.xy[self.basket_index]

    @property
    def vertical_extent(self) -> float:
        dy = np.abs(self.xy[:, 1] - self.rim_xy[1])
        return float(np.max(dy)) or 1.0


@dataclass(frozen=True)
class ReboundSimulationResult:
    params: ReboundParams
    court: Court
    shot_indices: np.ndarray
    target_indices: np.ndarray
    winner_indices: np.ndarray
    winner_is_offense: np.ndarray
    rebound_kinds: np.ndarray
    rebound_regions: np.ndarray
    shot_zones: np.ndarray
    winner_distances: np.ndarray

    def summary(self) -> dict[str, Any]:
        n = int(self.shot_indices.size)
        oreb = self.winner_is_offense.astype(np.float64)
        summary: dict[str, Any] = {
            "samples": n,
            "court_rows": int(self.court.spec.rows),
            "court_cols": int(self.court.spec.cols),
            "offensive_rebound_rate": float(np.mean(oreb)) if n else 0.0,
            "defensive_rebound_rate": float(1.0 - np.mean(oreb)) if n else 0.0,
            "mean_rebound_winner_distance": float(np.mean(self.winner_distances)) if n else 0.0,
            "params": asdict(self.params),
        }
        for name, values in (
            ("shot_zone", self.shot_zones),
            ("rebound_kind", self.rebound_kinds),
            ("rebound_region", self.rebound_regions),
        ):
            counts: dict[str, int] = {}
            oreb_rates: dict[str, float] = {}
            for value in sorted(set(values.tolist())):
                mask = values == value
                counts[str(value)] = int(np.sum(mask))
                oreb_rates[str(value)] = float(np.mean(oreb[mask])) if np.any(mask) else 0.0
            summary[f"{name}_counts"] = counts
            summary[f"{name}_offensive_rebound_rate"] = oreb_rates
        return summary

    def save_summary_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.summary(), indent=2, sort_keys=True) + "\n")


class _ThreePointAdapter:
    def __init__(self, spec: CourtSpec):
        self.court_width = int(spec.cols)
        self.court_height = int(spec.rows)
        self.three_point_distance = float(spec.three_point_distance)
        self.three_point_short_distance = float(spec.three_point_short_distance)
        row = self.court_height // 2 if spec.basket_row is None else int(spec.basket_row)
        self.basket_position = geometry.offset_to_axial_formula(int(spec.basket_col), row)

    @staticmethod
    def _offset_to_axial(col: int, row: int) -> tuple[int, int]:
        return geometry.offset_to_axial_formula(col, row)

    @staticmethod
    def _axial_to_cartesian(q: int, r: int) -> tuple[float, float]:
        return geometry.axial_to_cartesian_formula(q, r)


def build_court(spec: CourtSpec | None = None) -> Court:
    spec = spec or CourtSpec()
    adapter = _ThreePointAdapter(spec)
    cells: list[tuple[int, int]] = []
    offsets: list[tuple[int, int]] = []
    for row in range(spec.rows):
        for col in range(spec.cols):
            cell = geometry.offset_to_axial_formula(col, row)
            cells.append(cell)
            offsets.append((col, row))
    xy = np.asarray([geometry.axial_to_cartesian_formula(q, r) for q, r in cells], dtype=np.float64)
    basket = adapter.basket_position
    basket_index = cells.index(basket)
    lut, cell_index = geometry.precompute_hex_distance_lut(cells)
    basket_xy = xy[basket_index]
    distance_cart = np.linalg.norm(xy - basket_xy[None, :], axis=1)
    distance_hex = distance_cart / math.sqrt(3.0)
    three_point_hexes, _, _ = geometry.compute_three_point_geometry(adapter)
    three_point_mask = np.asarray([cell in three_point_hexes for cell in cells], dtype=bool)
    return Court(
        spec=spec,
        cells=tuple(cells),
        offsets=tuple(offsets),
        xy=xy,
        basket_axial=basket,
        basket_index=basket_index,
        distance_hex=distance_hex,
        distance_cart=distance_cart,
        three_point_mask=three_point_mask,
        hex_distance_lut=lut,
        cell_index=cell_index,
    )


def shot_location_weights(court: Court, params: ReboundParams | None = None) -> np.ndarray:
    params = params or ReboundParams()
    d = court.distance_hex
    weights = (
        params.shot_rim_weight * _gaussian(d, params.shot_rim_center, params.shot_rim_sigma)
        + params.shot_paint_weight * _gaussian(d, params.shot_paint_center, params.shot_paint_sigma)
        + params.shot_midrange_weight
        * _gaussian(d, params.shot_midrange_center, params.shot_midrange_sigma)
        + params.shot_three_weight * _gaussian(d, params.shot_three_center, params.shot_three_sigma)
    )
    weights = np.where(court.three_point_mask, weights * 1.2, weights)
    weights[court.basket_index] = 0.0
    return _normalize_weights(weights)


def rebound_kind_probabilities(distance_hex: float, is_three: bool) -> np.ndarray:
    long = np.clip(0.03 + 0.06 * float(distance_hex), 0.04, 0.35)
    short = np.clip(0.74 - 0.09 * float(distance_hex), 0.18, 0.74)
    if is_three:
        long += 0.08
        short -= 0.05
    normal = max(0.01, 1.0 - short - long)
    probs = np.asarray([short, normal, long], dtype=np.float64)
    return _normalize_weights(probs)


def rebound_region_probabilities(
    court: Court,
    shot_index: int,
    kind: ReboundKind,
    params: ReboundParams | None = None,
) -> dict[ReboundRegion, float]:
    params = params or ReboundParams()
    shot_xy = court.xy[int(shot_index)]
    dy = float(shot_xy[1] - court.rim_xy[1])
    side_strength = min(1.0, abs(dy) / court.vertical_extent)
    if side_strength < params.centerline_deadband:
        return {"weak": 0.0, "same": 0.0, "middle": 1.0}

    if kind == "long":
        weak = params.long_weak_prob * side_strength
        same = params.long_same_prob * side_strength
    else:
        weak = params.short_weak_prob * side_strength
        same = params.short_same_prob * side_strength
    middle = max(0.01, 1.0 - weak - same)
    total = weak + same + middle
    return {"weak": weak / total, "same": same / total, "middle": middle / total}


def rebound_target_weights(
    court: Court,
    shot_index: int,
    kind: ReboundKind,
    region: ReboundRegion,
    params: ReboundParams | None = None,
) -> np.ndarray:
    params = params or ReboundParams()
    shot_index = int(shot_index)
    shot_xy = court.xy[shot_index]
    rim_xy = court.rim_xy
    shot_vec = shot_xy - rim_xy
    shot_norm = float(np.linalg.norm(shot_vec))
    outward = shot_vec / shot_norm if shot_norm > 1e-9 else np.asarray([1.0, 0.0])
    side_sign = np.sign(float(shot_xy[1] - rim_xy[1]))
    side_strength = min(1.0, abs(float(shot_xy[1] - rim_xy[1])) / court.vertical_extent)
    vertical = np.asarray([0.0, 1.0], dtype=np.float64)

    base_hex, slope, sigma_hex = {
        "short": (
            params.short_rebound_base_hex,
            params.short_rebound_slope,
            params.short_rebound_sigma_hex,
        ),
        "normal": (
            params.normal_rebound_base_hex,
            params.normal_rebound_slope,
            params.normal_rebound_sigma_hex,
        ),
        "long": (
            params.long_rebound_base_hex,
            params.long_rebound_slope,
            params.long_rebound_sigma_hex,
        ),
    }[kind]
    distance_hex = base_hex + slope * float(court.distance_hex[shot_index])
    lateral_sign = 0.0
    if region == "weak":
        lateral_sign = -side_sign
    elif region == "same":
        lateral_sign = side_sign
    mean_xy = (
        rim_xy
        + outward * (distance_hex * math.sqrt(3.0))
        + vertical * lateral_sign * params.lateral_bias_hex * math.sqrt(3.0) * side_strength
    )
    sigma = max(1e-6, sigma_hex * math.sqrt(3.0))
    dist2 = np.sum((court.xy - mean_xy[None, :]) ** 2, axis=1)
    weights = np.exp(-dist2 / (2.0 * sigma * sigma))
    return _normalize_weights(weights)


def simulate_rebounds(
    samples: int,
    *,
    seed: int = 0,
    court: Court | None = None,
    params: ReboundParams | None = None,
    shot_indices: np.ndarray | None = None,
) -> ReboundSimulationResult:
    court = court or build_court()
    params = params or ReboundParams()
    rng = np.random.default_rng(seed)
    if shot_indices is None:
        samples = int(samples)
        shot_weights = shot_location_weights(court, params)
        shot_indices_arr = rng.choice(len(court.cells), size=samples, p=shot_weights).astype(np.int32)
    else:
        shot_indices_arr = np.asarray(shot_indices, dtype=np.int32).reshape(-1)
        samples = int(shot_indices_arr.size)
        invalid = (shot_indices_arr < 0) | (shot_indices_arr >= len(court.cells))
        if bool(np.any(invalid)):
            bad = int(shot_indices_arr[np.flatnonzero(invalid)[0]])
            raise ValueError(f"shot_indices contains invalid court cell index {bad}")
    target_indices = np.zeros(samples, dtype=np.int32)
    winner_indices = np.zeros(samples, dtype=np.int32)
    winner_is_offense = np.zeros(samples, dtype=bool)
    rebound_kinds = np.empty(samples, dtype=object)
    rebound_regions = np.empty(samples, dtype=object)
    shot_zones = np.empty(samples, dtype=object)
    winner_distances = np.zeros(samples, dtype=np.float32)

    for i, shot_idx in enumerate(shot_indices_arr):
        is_three = bool(court.three_point_mask[shot_idx])
        kind = _choice_label(rng, ("short", "normal", "long"), rebound_kind_probabilities(court.distance_hex[shot_idx], is_three))
        region_probs = rebound_region_probabilities(court, int(shot_idx), kind, params)
        region = _choice_label(
            rng,
            ("weak", "same", "middle"),
            np.asarray([region_probs["weak"], region_probs["same"], region_probs["middle"]]),
        )
        target_weights = rebound_target_weights(court, int(shot_idx), kind, region, params)
        target_idx = int(rng.choice(len(court.cells), p=target_weights))
        positions, shooter_id = _sample_player_positions(court, rng, int(shot_idx))
        winner_id, winner_distance = _sample_rebound_winner(
            court,
            rng,
            positions,
            target_idx,
            shooter_id,
            params,
        )
        target_indices[i] = target_idx
        winner_indices[i] = winner_id
        winner_is_offense[i] = winner_id < 5
        rebound_kinds[i] = kind
        rebound_regions[i] = region
        shot_zones[i] = classify_shot_zone(court, int(shot_idx))
        winner_distances[i] = winner_distance

    return ReboundSimulationResult(
        params=params,
        court=court,
        shot_indices=shot_indices_arr.astype(np.int32),
        target_indices=target_indices,
        winner_indices=winner_indices,
        winner_is_offense=winner_is_offense,
        rebound_kinds=rebound_kinds,
        rebound_regions=rebound_regions,
        shot_zones=shot_zones,
        winner_distances=winner_distances,
    )


def classify_shot_zone(court: Court, shot_index: int) -> str:
    d = float(court.distance_hex[int(shot_index)])
    if d <= 1.35:
        return "rim"
    if d <= 2.35:
        return "paint"
    if bool(court.three_point_mask[int(shot_index)]):
        side_strength = abs(float(court.xy[int(shot_index), 1] - court.rim_xy[1])) / court.vertical_extent
        return "corner_3" if side_strength >= 0.72 else "above_break_3"
    return "midrange"


def _sample_player_positions(court: Court, rng: np.random.Generator, shot_index: int) -> tuple[np.ndarray, int]:
    n_players = 10
    positions = np.full((n_players,), -1, dtype=np.int32)
    shooter_id = int(rng.integers(0, 5))
    positions[shooter_id] = int(shot_index)

    used = {int(shot_index)}
    offense_weights = _normalize_weights(
        0.45 * _gaussian(court.distance_hex, 2.2, 1.0)
        + 0.35 * _gaussian(court.distance_hex, 3.6, 1.1)
        + 0.20 * court.three_point_mask.astype(np.float64)
    )
    for pid in range(5):
        if pid == shooter_id:
            continue
        choice = _sample_unused(rng, offense_weights, used)
        positions[pid] = choice
        used.add(choice)

    for pid in range(5, 10):
        paired_offense = int(positions[pid - 5])
        guard_mean = court.xy[paired_offense] + 0.45 * (court.rim_xy - court.xy[paired_offense])
        dist2 = np.sum((court.xy - guard_mean[None, :]) ** 2, axis=1)
        guard_weights = np.exp(-dist2 / (2.0 * (1.55**2)))
        guard_weights += 0.04
        choice = _sample_unused(rng, _normalize_weights(guard_weights), used)
        positions[pid] = choice
        used.add(choice)
    return positions, shooter_id


def _sample_rebound_winner(
    court: Court,
    rng: np.random.Generator,
    positions: np.ndarray,
    target_index: int,
    shooter_id: int,
    params: ReboundParams,
) -> tuple[int, float]:
    dist_to_target = court.hex_distance_lut[positions, int(target_index)].astype(np.float64)
    dist_to_rim = court.distance_hex[positions]
    max_rim_dist = float(np.max(court.distance_hex)) or 1.0
    is_defense = np.arange(10) >= 5
    skills = rng.normal(0.0, params.rebound_skill_std, size=10)
    boxout = np.zeros(10, dtype=np.float64)
    for did in range(5, 10):
        oid = did - 5
        if dist_to_rim[did] <= dist_to_rim[oid] and court.hex_distance_lut[positions[did], positions[oid]] <= 2:
            boxout[did] = params.boxout_bias
    scores = (
        -params.target_distance_weight * dist_to_target
        + params.rebound_skill_weight * skills
        + params.inside_position_weight * ((max_rim_dist - dist_to_rim) / max_rim_dist)
        + params.defense_rebound_bias * is_defense.astype(np.float64)
        + boxout
    )
    # Shooters have a harder time rebounding their own miss in the immediate-resolution model.
    scores[int(shooter_id)] -= 0.15
    weights = _softmax(scores / max(1e-6, params.winner_temperature))
    winner = int(rng.choice(10, p=weights))
    return winner, float(dist_to_target[winner])


def _sample_unused(rng: np.random.Generator, weights: np.ndarray, used: set[int]) -> int:
    local = weights.copy()
    for idx in used:
        local[int(idx)] = 0.0
    if float(np.sum(local)) <= 0.0:
        local = np.ones_like(weights)
        for idx in used:
            local[int(idx)] = 0.0
    return int(rng.choice(len(local), p=_normalize_weights(local)))


def _choice_label(rng: np.random.Generator, labels: tuple[str, ...], probs: np.ndarray) -> str:
    return labels[int(rng.choice(len(labels), p=_normalize_weights(probs)))]


def _gaussian(x: np.ndarray, center: float, sigma: float) -> np.ndarray:
    sigma = max(1e-6, float(sigma))
    return np.exp(-0.5 * ((x - float(center)) / sigma) ** 2)


def _normalize_weights(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)
    total = float(np.sum(weights))
    if total <= 0.0:
        return np.ones_like(weights, dtype=np.float64) / float(weights.size)
    return weights / total


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - float(np.max(logits))
    exp = np.exp(shifted)
    return _normalize_weights(exp)
