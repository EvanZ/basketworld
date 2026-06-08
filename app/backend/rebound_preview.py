from __future__ import annotations

import os
from dataclasses import asdict, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from analytics.rebound_sim.model import Court, CourtSpec, ReboundParams, _softmax, build_court
from analytics.rebound_sim.table_model import FittedReboundTableModel
from app.backend.env_access import env_view

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_TABLE_MODEL_DIR = (
    _PROJECT_ROOT
    / "analytics"
    / "rebound_physics"
    / "outputs"
    / "dataset_9x8"
    / "fitted_catch_model"
)


def compute_rebound_preview(game_state: Any, request: Any) -> dict[str, Any]:
    """Compute a read-only rebound preview for the current terminal missed shot."""
    if request is not None and not bool(getattr(request, "enabled", True)):
        return _unavailable("disabled")

    state_payload = _current_state_payload(game_state)
    if not state_payload:
        return _unavailable("game_not_initialized")

    shot = _latest_terminal_missed_shot(state_payload, game_state)
    if not shot.get("available"):
        return _unavailable(str(shot.get("reason") or "terminal_not_missed_shot"))

    court = _court_from_state(state_payload)
    positions = _positions_from_state(state_payload)
    if not positions:
        return _unavailable("missing_positions")
    if any(tuple(pos) not in court.cell_index for pos in positions):
        return _unavailable("position_outside_rebound_table_court")

    shooter_id = int(shot["player_id"])
    if shooter_id < 0 or shooter_id >= len(positions):
        return _unavailable("shooter_position_unavailable")
    shot_cell = tuple(int(v) for v in positions[shooter_id])
    shot_index = int(court.cell_index[shot_cell])

    model_dir = _table_model_dir(request)
    if not model_dir.exists():
        return _unavailable("fitted_table_missing", table_model_dir=str(model_dir))
    model = _load_table_model(str(model_dir), _court_cache_key(court))

    target_probs = model.target_probabilities(court, shot_index)
    target_probs = _adjust_target_distribution(target_probs, request)
    shot_type = model.shot_type_for_shot(court, shot_index)

    rng = np.random.default_rng(_request_seed(request))
    target_index = int(rng.choice(len(court.cells), p=target_probs))
    params = _params_from_request(request)
    position_indices = np.asarray([court.cell_index[tuple(pos)] for pos in positions], dtype=np.int32)
    conditional_probs = _winner_probs_for_target(
        court,
        position_indices,
        target_index,
        params,
    )
    winner_slot = int(rng.choice(len(positions), p=conditional_probs))
    winner_id = int(winner_slot)

    marginal_probs = np.zeros(len(positions), dtype=np.float64)
    for idx, target_prob in enumerate(target_probs):
        if float(target_prob) <= 0.0:
            continue
        marginal_probs += float(target_prob) * _winner_probs_for_target(
            court,
            position_indices,
            idx,
            params,
        )
    marginal_probs = marginal_probs / max(1e-12, float(marginal_probs.sum()))

    target_cells = [
        {
            "index": int(idx),
            "q": int(cell[0]),
            "r": int(cell[1]),
            "prob": float(prob),
        }
        for idx, (cell, prob) in enumerate(zip(court.cells, target_probs, strict=True))
        if float(prob) > 0.0
    ]
    sampled_target = _target_payload(court, target_index, target_probs[target_index])
    winner_rows = _winner_rows(
        court,
        positions,
        position_indices,
        state_payload,
        marginal_probs,
        conditional_probs,
        target_index,
    )

    return {
        "status": "success",
        "available": True,
        "reason": None,
        "table_model_dir": str(model_dir),
        "shot": {
            "player_id": shooter_id,
            "team": _team_for_player(shooter_id, state_payload),
            "q": int(shot_cell[0]),
            "r": int(shot_cell[1]),
            "cell_index": shot_index,
            "shot_type": str(shot_type),
            "probability": _optional_float(shot.get("probability")),
            "expected_points": _optional_float(shot.get("expected_points")),
        },
        "target_cells": target_cells,
        "sampled_target": sampled_target,
        "sampled_winner": next((row for row in winner_rows if row["player_id"] == winner_id), None),
        "winner_probs": winner_rows,
        "params": asdict(params),
        "target_temperature": _positive_float(getattr(request, "target_temperature", 1.0), 1.0),
        "target_uniform_mix": _clamp01(getattr(request, "target_uniform_mix", 0.0)),
    }


def _unavailable(reason: str, **extra: Any) -> dict[str, Any]:
    return {"status": "success", "available": False, "reason": reason, **extra}


def _current_state_payload(game_state: Any) -> dict[str, Any]:
    runtime = getattr(game_state, "jax_runtime", None)
    if runtime is not None:
        return runtime.get_full_game_state(
            game_state,
            include_policy_probs=False,
            include_action_values=False,
            include_state_values=False,
        )
    env_obj = getattr(game_state, "env", None)
    if env_obj is None:
        return {}
    env = env_view(env_obj)
    return {
        "positions": [(int(q), int(r)) for q, r in env.positions],
        "last_action_results": getattr(env, "last_action_results", {}) or {},
        "offense_ids": [int(pid) for pid in (env.offense_ids or [])],
        "defense_ids": [int(pid) for pid in (env.defense_ids or [])],
        "basket_position": (int(env.basket_position[0]), int(env.basket_position[1])),
        "court_width": int(env.court_width or 8),
        "court_height": int(env.court_height or 9),
        "three_point_distance": float(env.three_point_distance or 4.25),
        "three_point_short_distance": (
            float(env.three_point_short_distance)
            if env.three_point_short_distance is not None
            else 3.0
        ),
        "done": bool(getattr(env, "episode_ended", False)),
    }


def _latest_terminal_missed_shot(state_payload: dict[str, Any], game_state: Any) -> dict[str, Any]:
    if not bool(state_payload.get("done")):
        return {"available": False, "reason": "episode_not_done"}
    results = state_payload.get("last_action_results") or {}
    shots = results.get("shots") if isinstance(results, dict) else None
    if isinstance(shots, dict) and shots:
        missed: list[dict[str, Any]] = []
        made = False
        for raw_pid, shot in shots.items():
            if not isinstance(shot, dict):
                continue
            entry = dict(shot)
            entry["player_id"] = int(raw_pid)
            if bool(shot.get("success", False)):
                made = True
            else:
                missed.append(entry)
        if missed:
            missed.sort(key=lambda row: int(row.get("player_id", 0)))
            missed[0]["available"] = True
            return missed[0]
        if made:
            return {"available": False, "reason": "last_terminal_shot_was_made"}

    shot_log = getattr(game_state, "shot_log", None) or []
    if shot_log:
        last = dict(shot_log[-1])
        if not bool(last.get("success", False)) and last.get("player_id") is not None:
            last["available"] = True
            return last
        return {"available": False, "reason": "last_terminal_shot_was_made"}
    return {"available": False, "reason": "terminal_not_missed_shot"}


def _positions_from_state(state_payload: dict[str, Any]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for raw in state_payload.get("positions") or []:
        if not isinstance(raw, (list, tuple)) or len(raw) < 2:
            return []
        out.append((int(raw[0]), int(raw[1])))
    return out


def _court_from_state(state_payload: dict[str, Any]) -> Court:
    rows = int(state_payload.get("court_height") or 9)
    cols = int(state_payload.get("court_width") or 8)
    basket = tuple(int(v) for v in (state_payload.get("basket_position") or (0, rows // 2)))
    basket_col, basket_row = _axial_to_offset_in_grid(basket, rows, cols)
    spec = CourtSpec(
        rows=rows,
        cols=cols,
        basket_col=basket_col,
        basket_row=basket_row,
        three_point_distance=float(state_payload.get("three_point_distance") or 4.25),
        three_point_short_distance=float(state_payload.get("three_point_short_distance") or 3.0),
    )
    court = build_court(spec)
    if tuple(court.basket_axial) != basket:
        raise ValueError(f"Runtime basket {basket} does not match rebound court basket {court.basket_axial}")
    return court


def _axial_to_offset_in_grid(cell: tuple[int, int], rows: int, cols: int) -> tuple[int, int]:
    target_q, target_r = int(cell[0]), int(cell[1])
    for row in range(rows):
        for col in range(cols):
            q = col - ((row - (row & 1)) >> 1)
            if q == target_q and row == target_r:
                return col, row
    return 0, rows // 2


def _court_cache_key(court: Court) -> tuple[int, int, int, int, float, float]:
    return (
        int(court.spec.rows),
        int(court.spec.cols),
        int(court.spec.basket_col),
        int(court.spec.basket_row or 0),
        float(court.spec.three_point_distance),
        float(court.spec.three_point_short_distance),
    )


@lru_cache(maxsize=8)
def _load_table_model(model_dir: str, court_key: tuple[int, int, int, int, float, float]) -> FittedReboundTableModel:
    rows, cols, basket_col, basket_row, three_point_distance, three_point_short_distance = court_key
    court = build_court(
        CourtSpec(
            rows=rows,
            cols=cols,
            basket_col=basket_col,
            basket_row=basket_row,
            three_point_distance=three_point_distance,
            three_point_short_distance=three_point_short_distance,
        )
    )
    return FittedReboundTableModel.load(model_dir, court=court)


def _table_model_dir(request: Any) -> Path:
    override = getattr(request, "table_model_dir", None) if request is not None else None
    raw = override or os.environ.get("BW_REBOUND_TABLE_MODEL_DIR") or _DEFAULT_TABLE_MODEL_DIR
    return Path(raw).expanduser().resolve()


def _request_seed(request: Any) -> int | None:
    raw = getattr(request, "seed", None) if request is not None else None
    if raw is None:
        return None
    return int(raw)


def _params_from_request(request: Any) -> ReboundParams:
    params = ReboundParams()
    overrides: dict[str, float] = {}
    for field in (
        "target_distance_weight",
        "winner_temperature",
    ):
        value = getattr(request, field, None) if request is not None else None
        if value is not None:
            overrides[field] = float(value)
    return replace(params, **overrides) if overrides else params


def _adjust_target_distribution(probs: np.ndarray, request: Any) -> np.ndarray:
    adjusted = np.asarray(probs, dtype=np.float64)
    temp = _positive_float(getattr(request, "target_temperature", 1.0), 1.0)
    if abs(temp - 1.0) > 1e-9:
        logits = np.log(np.maximum(adjusted, 1e-12)) / temp
        adjusted = _softmax(logits)
    mix = _clamp01(getattr(request, "target_uniform_mix", 0.0))
    if mix > 0.0:
        uniform = np.ones_like(adjusted, dtype=np.float64) / float(adjusted.size)
        adjusted = (1.0 - mix) * adjusted + mix * uniform
    total = float(adjusted.sum())
    if total <= 0.0 or not np.isfinite(total):
        return np.ones_like(adjusted, dtype=np.float64) / float(adjusted.size)
    return adjusted / total


def _winner_probs_for_target(
    court: Court,
    position_indices: np.ndarray,
    target_index: int,
    params: ReboundParams,
) -> np.ndarray:
    dist_to_target = court.hex_distance_lut[position_indices, int(target_index)].astype(np.float64)
    scores = -params.target_distance_weight * dist_to_target
    return _softmax(scores / max(1e-6, params.winner_temperature))


def _winner_rows(
    court: Court,
    positions: list[tuple[int, int]],
    position_indices: np.ndarray,
    state_payload: dict[str, Any],
    marginal_probs: np.ndarray,
    conditional_probs: np.ndarray,
    target_index: int,
) -> list[dict[str, Any]]:
    offense_ids = {int(pid) for pid in (state_payload.get("offense_ids") or [])}
    defense_ids = {int(pid) for pid in (state_payload.get("defense_ids") or [])}
    rows: list[dict[str, Any]] = []
    for pid, pos in enumerate(positions):
        team = "offense" if pid in offense_ids else ("defense" if pid in defense_ids else "unknown")
        rows.append(
            {
                "player_id": int(pid),
                "team": team,
                "q": int(pos[0]),
                "r": int(pos[1]),
                "prob": float(marginal_probs[pid]),
                "conditional_prob": float(conditional_probs[pid]),
                "distance_to_sampled_target": int(court.hex_distance_lut[int(position_indices[pid]), int(target_index)]),
            }
        )
    rows.sort(key=lambda row: (-float(row["conditional_prob"]), int(row["player_id"])))
    return rows


def _team_for_player(player_id: int, state_payload: dict[str, Any]) -> str:
    offense_ids = {int(pid) for pid in (state_payload.get("offense_ids") or [])}
    defense_ids = {int(pid) for pid in (state_payload.get("defense_ids") or [])}
    if int(player_id) in offense_ids:
        return "offense"
    if int(player_id) in defense_ids:
        return "defense"
    return "unknown"


def _target_payload(court: Court, target_index: int, prob: float) -> dict[str, Any]:
    q, r = court.cells[int(target_index)]
    return {"index": int(target_index), "q": int(q), "r": int(r), "prob": float(prob)}


def _clamp01(value: Any) -> float:
    try:
        num = float(value)
    except Exception:
        return 0.0
    if not np.isfinite(num):
        return 0.0
    return float(max(0.0, min(1.0, num)))


def _positive_float(value: Any, default: float) -> float:
    try:
        num = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(num) or num <= 0.0:
        return float(default)
    return float(num)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        num = float(value)
    except Exception:
        return None
    return float(num) if np.isfinite(num) else None
