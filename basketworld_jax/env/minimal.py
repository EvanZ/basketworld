from __future__ import annotations

import math
from typing import Any, NamedTuple, Sequence

import numpy as np

from basketworld.envs.basketworld_env_v2 import ActionType, Team
from basketworld.utils.start_templates import (
    _mirror_anchor,
    _project_anchor_to_valid_cell,
    resolve_start_template,
)
from basketworld_jax.train.cli import resolve_training_team
from train.env_factory import setup_environment


MOVE_ACTION_START = ActionType.MOVE_E.value
MOVE_ACTION_END = ActionType.MOVE_SE.value + 1
PASS_ACTION_START = ActionType.PASS_E.value
PASS_ACTION_END = ActionType.PASS_SE.value + 1
ACTION_COUNT = len(ActionType)
SQRT3 = float(np.sqrt(3.0))
TOKEN_OBS_PLAYER_DIM = 15
TOKEN_OBS_GLOBAL_DIM = 4
TOKEN_OBS_ROLE_FLAG_DIM = 1
TURNOVER_REASON_NONE = 0
TURNOVER_REASON_PASS_OUT_OF_BOUNDS = 1
TURNOVER_REASON_INTERCEPTED = 2
TURNOVER_REASON_DEFENDER_PRESSURE = 3
TURNOVER_REASON_MOVE_OUT_OF_BOUNDS = 4
TURNOVER_REASON_SHOT_CLOCK = 5
TURNOVER_REASON_OFFENSIVE_THREE_SECONDS = 6
SHOT_TYPE_NONE = 0
SHOT_TYPE_DUNK = 1
SHOT_TYPE_TWO = 2
SHOT_TYPE_THREE = 3


class KernelStatic(NamedTuple):
    cell_coords: Any
    basket_distance_by_cell: Any
    cell_distance_matrix: Any
    non_basket_cell_mask: Any
    offense_spawn_candidate_mask: Any
    move_mask_by_cell: Any
    three_point_by_cell: Any
    basket_position: Any
    hex_directions: Any
    offense_ids: Any
    defense_ids: Any
    role_encoding: Any
    opponent_mask: Any
    pointer_pass_slot_mask: Any
    pointer_pass_target_ids: Any
    start_template_enabled: Any
    start_template_prob: Any
    start_template_positions: Any
    start_template_ball_holders: Any
    start_template_shot_clocks: Any
    start_template_weights: Any
    start_template_entry_anchors: Any
    start_template_entry_jitter_radii: Any
    start_template_entry_has_ball: Any
    court_norm_den: Any
    offensive_lane_by_cell: Any
    defensive_lane_by_cell: Any
    allow_dunks: Any
    mask_occupied_moves: Any
    enable_pass_gating: Any
    shot_pressure_enabled: Any
    shot_pressure_max: Any
    shot_pressure_lambda: Any
    shot_pressure_cos_threshold: Any
    defender_pressure_distance: Any
    defender_pressure_turnover_chance: Any
    defender_pressure_decay_lambda: Any
    base_steal_rate: Any
    steal_perp_decay: Any
    steal_distance_factor: Any
    steal_position_weight_min: Any
    three_point_distance: Any
    three_pt_extra_hex_decay: Any
    shot_clock_min: Any
    shot_clock_max: Any
    three_second_max_steps: Any
    defense_min_spawn_distance: Any
    max_spawn_distance_enabled: Any
    max_spawn_distance: Any
    defender_spawn_distance: Any
    defender_guard_distance: Any
    illegal_defense_enabled: Any
    offensive_three_seconds_enabled: Any
    pass_reward: Any
    violation_reward: Any
    reward_shaping_gamma: Any
    enable_phi_shaping: Any
    phi_beta: Any
    phi_blend_weight: Any
    phi_use_ball_handler_only: Any
    pass_oob_turnover_prob: Any
    assist_window: Any
    potential_assist_pct: Any
    full_assist_bonus_pct: Any
    base_layup_pct: Any
    base_three_pt_pct: Any
    base_dunk_pct: Any
    layup_std: Any
    three_pt_std: Any
    dunk_std: Any
    training_player_mask: Any
    training_role_flag: Any
    task_reward_scale: Any
    enable_intent_learning: Any
    enable_defense_intent_learning: Any
    num_intents: Any
    intent_commitment_steps: Any
    intent_null_prob: Any
    defense_intent_null_prob: Any
    intent_visible_to_defense_prob: Any


class KernelState(NamedTuple):
    positions: Any
    ball_holder: Any
    shot_clock: Any
    step_count: Any
    episode_ended: Any
    pressure_exposure: Any
    offense_lane_steps: Any
    defense_lane_steps: Any
    cached_phi: Any
    offense_score: Any
    defense_score: Any
    assist_active: Any
    assist_passer: Any
    assist_recipient: Any
    assist_expires_at: Any
    intent_index: Any
    intent_active: Any
    intent_age: Any
    intent_commitment_remaining: Any
    intent_visible_to_defense: Any
    defense_intent_index: Any
    defense_intent_active: Any
    defense_intent_age: Any
    defense_intent_commitment_remaining: Any
    layup_pct: Any
    three_pt_pct: Any
    dunk_pct: Any


class StepBatchOutput(NamedTuple):
    state: KernelState
    rewards: Any
    done: Any
    pass_attempt: Any
    pass_passer: Any
    pass_receiver: Any
    completed_pass: Any
    assist: Any
    turnover: Any
    terminal_episode_steps: Any
    shot_attempt: Any
    shot_success: Any
    shot_shooter: Any
    shot_value: Any
    shot_expected_points: Any
    shot_distance: Any
    shot_type: Any
    shot_q: Any
    shot_r: Any
    potential_assist: Any
    assist_passer: Any
    turnover_player: Any
    turnover_reason: Any
    offensive_three_seconds: Any
    defensive_lane_violation: Any
    defensive_lane_violation_player: Any


def _player_skill_arrays(env) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    layup = np.full(env.n_players, float(env.layup_pct), dtype=np.float32)
    three = np.full(env.n_players, float(env.three_pt_pct), dtype=np.float32)
    dunk = np.full(env.n_players, float(env.dunk_pct), dtype=np.float32)
    layup[np.asarray(env.offense_ids, dtype=np.int32)] = np.asarray(
        env.offense_layup_pct_by_player,
        dtype=np.float32,
    )
    three[np.asarray(env.offense_ids, dtype=np.int32)] = np.asarray(
        env.offense_three_pt_pct_by_player,
        dtype=np.float32,
    )
    dunk[np.asarray(env.offense_ids, dtype=np.int32)] = np.asarray(
        env.offense_dunk_pct_by_player,
        dtype=np.float32,
    )
    return layup, three, dunk


def _lane_step_arrays(env) -> tuple[np.ndarray, np.ndarray]:
    offense = np.zeros(env.n_players, dtype=np.float32)
    defense = np.zeros(env.n_players, dtype=np.float32)
    for pid in range(env.n_players):
        offense[pid] = float(getattr(env, "_offensive_lane_steps", {}).get(pid, 0))
        defense[pid] = float(getattr(env, "_defender_in_key_steps", {}).get(pid, 0))
    return offense, defense


def snapshot_state_from_env(env) -> dict[str, np.ndarray | int]:
    layup, three, dunk = _player_skill_arrays(env)
    offense_lane_steps, defense_lane_steps = _lane_step_arrays(env)
    assist_candidate = getattr(env, "_assist_candidate", None)
    return {
        "positions": np.asarray(env.positions, dtype=np.int32).copy(),
        "ball_holder": int(env.ball_holder) if env.ball_holder is not None else -1,
        "shot_clock": int(env.shot_clock),
        "step_count": int(getattr(env, "step_count", 0)),
        "episode_ended": 1 if bool(getattr(env, "episode_ended", False)) else 0,
        "pressure_exposure": float(getattr(env, "pressure_exposure", 0.0)),
        "offense_lane_steps": offense_lane_steps,
        "defense_lane_steps": defense_lane_steps,
        "cached_phi": float(getattr(env, "_cached_phi", 0.0) or 0.0),
        "offense_score": float(getattr(env, "offense_score", 0.0)),
        "defense_score": float(getattr(env, "defense_score", 0.0)),
        "assist_active": 1 if assist_candidate is not None else 0,
        "assist_passer": int(assist_candidate.get("passer_id", -1)) if assist_candidate is not None else -1,
        "assist_recipient": int(assist_candidate.get("recipient_id", -1)) if assist_candidate is not None else -1,
        "assist_expires_at": int(assist_candidate.get("expires_at_step", -1)) if assist_candidate is not None else -1,
        "intent_index": int(getattr(env, "intent_index", 0)),
        "intent_active": 1 if bool(getattr(env, "intent_active", False)) else 0,
        "intent_age": int(getattr(env, "intent_age", 0)),
        "intent_commitment_remaining": int(getattr(env, "intent_commitment_remaining", 0)),
        "intent_visible_to_defense": 1 if bool(getattr(env, "_intent_visible_to_defense", False)) else 0,
        "defense_intent_index": int(getattr(env, "defense_intent_index", 0)),
        "defense_intent_active": 1 if bool(getattr(env, "defense_intent_active", False)) else 0,
        "defense_intent_age": int(getattr(env, "defense_intent_age", 0)),
        "defense_intent_commitment_remaining": int(
            getattr(env, "defense_intent_commitment_remaining", 0)
        ),
        "layup_pct": layup,
        "three_pt_pct": three,
        "dunk_pct": dunk,
    }


def stack_state_snapshots(
    snapshots: Sequence[dict[str, np.ndarray | int]],
    xp,
) -> KernelState:
    return KernelState(
        positions=xp.asarray(
            np.stack([np.asarray(item["positions"], dtype=np.int32) for item in snapshots], axis=0),
            dtype=xp.int32,
        ),
        ball_holder=xp.asarray(
            np.asarray([int(item["ball_holder"]) for item in snapshots], dtype=np.int32),
            dtype=xp.int32,
        ),
        shot_clock=xp.asarray(
            np.asarray([int(item["shot_clock"]) for item in snapshots], dtype=np.int32),
            dtype=xp.int32,
        ),
        step_count=xp.asarray(
            np.asarray([int(item["step_count"]) for item in snapshots], dtype=np.int32),
            dtype=xp.int32,
        ),
        episode_ended=xp.asarray(
            np.asarray([int(item["episode_ended"]) for item in snapshots], dtype=np.int8),
            dtype=xp.int8,
        ),
        pressure_exposure=xp.asarray(
            np.asarray([float(item["pressure_exposure"]) for item in snapshots], dtype=np.float32),
            dtype=xp.float32,
        ),
        offense_lane_steps=xp.asarray(
            np.stack(
                [np.asarray(item["offense_lane_steps"], dtype=np.float32) for item in snapshots],
                axis=0,
            ),
            dtype=xp.float32,
        ),
        defense_lane_steps=xp.asarray(
            np.stack(
                [np.asarray(item["defense_lane_steps"], dtype=np.float32) for item in snapshots],
                axis=0,
            ),
            dtype=xp.float32,
        ),
        cached_phi=xp.asarray(
            np.asarray([float(item["cached_phi"]) for item in snapshots], dtype=np.float32),
            dtype=xp.float32,
        ),
        offense_score=xp.asarray(
            np.asarray([float(item["offense_score"]) for item in snapshots], dtype=np.float32),
            dtype=xp.float32,
        ),
        defense_score=xp.asarray(
            np.asarray([float(item["defense_score"]) for item in snapshots], dtype=np.float32),
            dtype=xp.float32,
        ),
        assist_active=xp.asarray(
            np.asarray([int(item["assist_active"]) for item in snapshots], dtype=np.int8),
            dtype=xp.int8,
        ),
        assist_passer=xp.asarray(
            np.asarray([int(item["assist_passer"]) for item in snapshots], dtype=np.int32),
            dtype=xp.int32,
        ),
        assist_recipient=xp.asarray(
            np.asarray([int(item["assist_recipient"]) for item in snapshots], dtype=np.int32),
            dtype=xp.int32,
        ),
        assist_expires_at=xp.asarray(
            np.asarray([int(item["assist_expires_at"]) for item in snapshots], dtype=np.int32),
            dtype=xp.int32,
        ),
        intent_index=xp.asarray(
            np.asarray([int(item.get("intent_index", 0)) for item in snapshots], dtype=np.int32),
            dtype=xp.int32,
        ),
        intent_active=xp.asarray(
            np.asarray([int(item.get("intent_active", 0)) for item in snapshots], dtype=np.int8),
            dtype=xp.int8,
        ),
        intent_age=xp.asarray(
            np.asarray([int(item.get("intent_age", 0)) for item in snapshots], dtype=np.int32),
            dtype=xp.int32,
        ),
        intent_commitment_remaining=xp.asarray(
            np.asarray(
                [int(item.get("intent_commitment_remaining", 0)) for item in snapshots],
                dtype=np.int32,
            ),
            dtype=xp.int32,
        ),
        intent_visible_to_defense=xp.asarray(
            np.asarray([int(item.get("intent_visible_to_defense", 0)) for item in snapshots], dtype=np.int8),
            dtype=xp.int8,
        ),
        defense_intent_index=xp.asarray(
            np.asarray([int(item.get("defense_intent_index", 0)) for item in snapshots], dtype=np.int32),
            dtype=xp.int32,
        ),
        defense_intent_active=xp.asarray(
            np.asarray([int(item.get("defense_intent_active", 0)) for item in snapshots], dtype=np.int8),
            dtype=xp.int8,
        ),
        defense_intent_age=xp.asarray(
            np.asarray([int(item.get("defense_intent_age", 0)) for item in snapshots], dtype=np.int32),
            dtype=xp.int32,
        ),
        defense_intent_commitment_remaining=xp.asarray(
            np.asarray(
                [int(item.get("defense_intent_commitment_remaining", 0)) for item in snapshots],
                dtype=np.int32,
            ),
            dtype=xp.int32,
        ),
        layup_pct=xp.asarray(
            np.stack([np.asarray(item["layup_pct"], dtype=np.float32) for item in snapshots], axis=0),
            dtype=xp.float32,
        ),
        three_pt_pct=xp.asarray(
            np.stack([np.asarray(item["three_pt_pct"], dtype=np.float32) for item in snapshots], axis=0),
            dtype=xp.float32,
        ),
        dunk_pct=xp.asarray(
            np.stack([np.asarray(item["dunk_pct"], dtype=np.float32) for item in snapshots], axis=0),
            dtype=xp.float32,
        ),
    )


def _compiled_start_template_arrays(env) -> dict[str, np.ndarray | float | bool]:
    n_players = int(env.n_players)
    fallback_holder = int(env.offense_ids[0]) if len(env.offense_ids) else 0
    fallback = {
        "enabled": False,
        "prob": 0.0,
        "positions": np.zeros((1, n_players, 2), dtype=np.int32),
        "ball_holders": np.full((1,), fallback_holder, dtype=np.int32),
        "shot_clocks": np.full((1,), -1, dtype=np.int32),
        "weights": np.ones((1,), dtype=np.float32),
        "entry_anchors": np.zeros((1, n_players, 2), dtype=np.int32),
        "entry_jitter_radii": np.zeros((1, n_players), dtype=np.int32),
        "entry_has_ball": np.zeros((1, n_players), dtype=np.int8),
    }

    enabled = bool(getattr(env, "start_template_enabled", False))
    library = getattr(env, "start_template_library", None)
    prob = float(np.clip(float(getattr(env, "start_template_prob", 0.0)), 0.0, 1.0))
    if not enabled or library is None or prob <= 0.0:
        return fallback

    jitter_scale = float(getattr(env, "start_template_jitter_scale", 1.0))
    mirror_prob = float(np.clip(float(getattr(env, "start_template_mirror_prob", 0.0)), 0.0, 1.0))
    strict = bool(getattr(env, "start_template_strict", False))
    positions: list[np.ndarray] = []
    ball_holders: list[int] = []
    shot_clocks: list[int] = []
    weights: list[float] = []
    entry_anchors: list[np.ndarray] = []
    entry_jitter_radii: list[np.ndarray] = []
    entry_has_ball: list[np.ndarray] = []
    valid_cells = list(
        getattr(env, "_valid_axial", ())
        or getattr(env, "_cell_index", {}).keys()
    )

    for template in list(library.get("templates", []) or []):
        template_weight = max(0.0, float(template.get("weight", 1.0)))
        if template_weight <= 0.0:
            continue
        mirrorable = bool(template.get("mirrorable", False))
        variants: list[tuple[bool, float]]
        if mirrorable and mirror_prob > 0.0:
            variants = [(False, template_weight * (1.0 - mirror_prob))]
            variants.append((True, template_weight * mirror_prob))
        else:
            variants = [(False, template_weight)]

        for mirror, variant_weight in variants:
            if variant_weight <= 0.0:
                continue
            try:
                resolved = resolve_start_template(
                    env,
                    template,
                    jitter_scale=jitter_scale,
                    mirror=mirror,
                )
            except Exception:
                if strict:
                    raise
                continue
            variant_anchors: list[tuple[int, int]] = []
            variant_jitter: list[int] = []
            variant_has_ball: list[int] = []
            try:
                for team_name in ("offense", "defense"):
                    for entry in list(template.get(team_name, []) or []):
                        anchor = _project_anchor_to_valid_cell(
                            env,
                            (int(entry["anchor"][0]), int(entry["anchor"][1])),
                            valid_cells=valid_cells,
                        )
                        if mirror and bool(template.get("mirrorable", False)):
                            anchor = _project_anchor_to_valid_cell(
                                env,
                                _mirror_anchor(env, anchor),
                                valid_cells=valid_cells,
                            )
                        variant_anchors.append((int(anchor[0]), int(anchor[1])))
                        variant_jitter.append(
                            max(
                                0,
                                int(
                                    round(
                                        float(entry.get("jitter_radius", 0))
                                        * max(0.0, jitter_scale)
                                    )
                                ),
                            )
                        )
                        variant_has_ball.append(1 if bool(entry.get("has_ball", False)) else 0)
            except Exception:
                if strict:
                    raise
                continue
            if len(variant_anchors) != n_players:
                if strict:
                    raise ValueError(
                        f"start template '{template.get('id', '')}' has {len(variant_anchors)} entries, expected {n_players}"
                    )
                continue
            positions.append(
                np.asarray(resolved["initial_positions"], dtype=np.int32).reshape(n_players, 2)
            )
            ball_holders.append(int(resolved.get("ball_holder", -1)))
            shot_clocks.append(int(resolved.get("shot_clock", -1)))
            weights.append(float(variant_weight))
            entry_anchors.append(np.asarray(variant_anchors, dtype=np.int32).reshape(n_players, 2))
            entry_jitter_radii.append(np.asarray(variant_jitter, dtype=np.int32).reshape(n_players))
            entry_has_ball.append(np.asarray(variant_has_ball, dtype=np.int8).reshape(n_players))

    if not positions:
        if strict:
            raise ValueError("start-template library did not produce any valid JAX reset candidates")
        return fallback

    weight_array = np.asarray(weights, dtype=np.float32)
    if float(np.sum(weight_array)) <= 0.0:
        weight_array = np.ones_like(weight_array, dtype=np.float32)
    return {
        "enabled": True,
        "prob": prob,
        "positions": np.stack(positions, axis=0).astype(np.int32, copy=False),
        "ball_holders": np.asarray(ball_holders, dtype=np.int32),
        "shot_clocks": np.asarray(shot_clocks, dtype=np.int32),
        "weights": weight_array,
        "entry_anchors": np.stack(entry_anchors, axis=0).astype(np.int32, copy=False),
        "entry_jitter_radii": np.stack(entry_jitter_radii, axis=0).astype(np.int32, copy=False),
        "entry_has_ball": np.stack(entry_has_ball, axis=0).astype(np.int8, copy=False),
    }


def build_kernel_static_from_env(env, xp) -> KernelStatic:
    cells = sorted(env._move_mask_by_cell.keys())
    basket_position = np.asarray(env.basket_position, dtype=np.int32)
    basket_distance = np.asarray(
        [env._hex_distance(cell, env.basket_position) for cell in cells],
        dtype=np.int32,
    )
    cell_distance_matrix = np.asarray(
        [[env._hex_distance(src, dst) for dst in cells] for src in cells],
        dtype=np.int32,
    )
    non_basket_mask = np.asarray(
        [0 if np.array_equal(np.asarray(cell, dtype=np.int32), basket_position) else 1 for cell in cells],
        dtype=np.int8,
    )
    offense_min_spawn = max(0, int(getattr(env, "spawn_distance", 0)))
    defense_min_spawn = max(0, int(getattr(env, "spawn_distance", 0)) - 1)
    max_spawn_distance = getattr(env, "max_spawn_distance", None)
    max_spawn_enabled = max_spawn_distance is not None
    max_spawn_value = int(max_spawn_distance) if max_spawn_enabled else -1
    offense_spawn_candidate_mask = (
        (non_basket_mask == 1)
        & (basket_distance >= offense_min_spawn)
        & ((basket_distance <= max_spawn_value) if max_spawn_enabled else np.ones_like(basket_distance, dtype=bool))
    ).astype(np.int8)
    if int(np.sum(offense_spawn_candidate_mask)) < int(env.players_per_side):
        offense_spawn_candidate_mask = non_basket_mask.copy()
    move_masks = np.stack(
        [np.asarray(env._move_mask_by_cell[cell], dtype=np.int8) for cell in cells],
        axis=0,
    )
    three_point_mask = np.asarray(
        [1 if cell in getattr(env, "_three_point_hexes", set()) else 0 for cell in cells],
        dtype=np.int8,
    )
    offensive_lane_mask = np.asarray(
        [1 if cell in getattr(env, "offensive_lane_hexes", set()) else 0 for cell in cells],
        dtype=np.int8,
    )
    defensive_lane_mask = np.asarray(
        [1 if cell in getattr(env, "defensive_lane_hexes", set()) else 0 for cell in cells],
        dtype=np.int8,
    )
    player_is_offense = np.asarray(
        [1 if pid in env.offense_ids else 0 for pid in range(env.n_players)],
        dtype=np.int8,
    )
    if getattr(env, "training_team", Team.OFFENSE) == Team.DEFENSE:
        training_ids = np.asarray(env.defense_ids, dtype=np.int32)
        training_role_flag = -1.0
    else:
        training_ids = np.asarray(env.offense_ids, dtype=np.int32)
        training_role_flag = 1.0
    training_player_mask = np.zeros(env.n_players, dtype=np.float32)
    training_player_mask[training_ids] = 1.0
    role_encoding = np.where(player_is_offense == 1, 1.0, -1.0).astype(np.float32)
    opponent_mask = (player_is_offense[:, None] != player_is_offense[None, :]).astype(np.int8)
    pass_slot_mask = np.zeros((env.n_players, 6), dtype=np.int8)
    pass_target_ids = np.full((env.n_players, 6), -1, dtype=np.int32)
    for passer_id in range(env.n_players):
        if passer_id in env.offense_ids:
            teammates = [pid for pid in env.offense_ids if pid != passer_id]
        else:
            teammates = [pid for pid in env.defense_ids if pid != passer_id]
        teammates = sorted(int(pid) for pid in teammates)[:6]
        pass_slot_mask[passer_id, : len(teammates)] = 1
        pass_target_ids[passer_id, : len(teammates)] = np.asarray(teammates, dtype=np.int32)
    start_templates = _compiled_start_template_arrays(env)

    return KernelStatic(
        cell_coords=xp.asarray(np.asarray(cells, dtype=np.int32), dtype=xp.int32),
        basket_distance_by_cell=xp.asarray(basket_distance, dtype=xp.int32),
        cell_distance_matrix=xp.asarray(cell_distance_matrix, dtype=xp.int32),
        non_basket_cell_mask=xp.asarray(non_basket_mask, dtype=xp.int8),
        offense_spawn_candidate_mask=xp.asarray(offense_spawn_candidate_mask, dtype=xp.int8),
        move_mask_by_cell=xp.asarray(move_masks, dtype=xp.int8),
        three_point_by_cell=xp.asarray(three_point_mask, dtype=xp.int8),
        basket_position=xp.asarray(basket_position, dtype=xp.int32),
        hex_directions=xp.asarray(np.asarray(env.hex_directions, dtype=np.int32), dtype=xp.int32),
        offense_ids=xp.asarray(np.asarray(env.offense_ids, dtype=np.int32), dtype=xp.int32),
        defense_ids=xp.asarray(np.asarray(env.defense_ids, dtype=np.int32), dtype=xp.int32),
        role_encoding=xp.asarray(role_encoding, dtype=xp.float32),
        opponent_mask=xp.asarray(opponent_mask, dtype=xp.int8),
        pointer_pass_slot_mask=xp.asarray(pass_slot_mask, dtype=xp.int8),
        pointer_pass_target_ids=xp.asarray(pass_target_ids, dtype=xp.int32),
        start_template_enabled=xp.asarray(1 if bool(start_templates["enabled"]) else 0, dtype=xp.int8),
        start_template_prob=xp.asarray(float(start_templates["prob"]), dtype=xp.float32),
        start_template_positions=xp.asarray(start_templates["positions"], dtype=xp.int32),
        start_template_ball_holders=xp.asarray(start_templates["ball_holders"], dtype=xp.int32),
        start_template_shot_clocks=xp.asarray(start_templates["shot_clocks"], dtype=xp.int32),
        start_template_weights=xp.asarray(start_templates["weights"], dtype=xp.float32),
        start_template_entry_anchors=xp.asarray(start_templates["entry_anchors"], dtype=xp.int32),
        start_template_entry_jitter_radii=xp.asarray(start_templates["entry_jitter_radii"], dtype=xp.int32),
        start_template_entry_has_ball=xp.asarray(start_templates["entry_has_ball"], dtype=xp.int8),
        court_norm_den=xp.asarray(
            float(max(env.court_width, env.court_height)) if env.normalize_obs else 1.0,
            dtype=xp.float32,
        ),
        offensive_lane_by_cell=xp.asarray(offensive_lane_mask, dtype=xp.int8),
        defensive_lane_by_cell=xp.asarray(defensive_lane_mask, dtype=xp.int8),
        allow_dunks=xp.asarray(1 if env.allow_dunks else 0, dtype=xp.int8),
        mask_occupied_moves=xp.asarray(1 if env.mask_occupied_moves else 0, dtype=xp.int8),
        enable_pass_gating=xp.asarray(1 if env.enable_pass_gating else 0, dtype=xp.int8),
        shot_pressure_enabled=xp.asarray(1 if env.shot_pressure_enabled else 0, dtype=xp.int8),
        shot_pressure_max=xp.asarray(float(env.shot_pressure_max), dtype=xp.float32),
        shot_pressure_lambda=xp.asarray(float(env.shot_pressure_lambda), dtype=xp.float32),
        shot_pressure_cos_threshold=xp.asarray(float(math.cos(env.shot_pressure_arc_rad / 2.0)), dtype=xp.float32),
        defender_pressure_distance=xp.asarray(float(env.defender_pressure_distance), dtype=xp.float32),
        defender_pressure_turnover_chance=xp.asarray(float(env.defender_pressure_turnover_chance), dtype=xp.float32),
        defender_pressure_decay_lambda=xp.asarray(float(env.defender_pressure_decay_lambda), dtype=xp.float32),
        base_steal_rate=xp.asarray(float(env.base_steal_rate), dtype=xp.float32),
        steal_perp_decay=xp.asarray(float(env.steal_perp_decay), dtype=xp.float32),
        steal_distance_factor=xp.asarray(float(env.steal_distance_factor), dtype=xp.float32),
        steal_position_weight_min=xp.asarray(float(env.steal_position_weight_min), dtype=xp.float32),
        three_point_distance=xp.asarray(float(env.three_point_distance), dtype=xp.float32),
        three_pt_extra_hex_decay=xp.asarray(float(env.three_pt_extra_hex_decay), dtype=xp.float32),
        shot_clock_min=xp.asarray(int(env.min_shot_clock), dtype=xp.int32),
        shot_clock_max=xp.asarray(int(env.shot_clock_steps), dtype=xp.int32),
        three_second_max_steps=xp.asarray(float(env.three_second_max_steps), dtype=xp.float32),
        defense_min_spawn_distance=xp.asarray(float(defense_min_spawn), dtype=xp.float32),
        max_spawn_distance_enabled=xp.asarray(1 if max_spawn_enabled else 0, dtype=xp.int8),
        max_spawn_distance=xp.asarray(float(max_spawn_value), dtype=xp.float32),
        defender_spawn_distance=xp.asarray(float(env.defender_spawn_distance), dtype=xp.float32),
        defender_guard_distance=xp.asarray(float(env.defender_guard_distance), dtype=xp.float32),
        illegal_defense_enabled=xp.asarray(1 if env.illegal_defense_enabled else 0, dtype=xp.int8),
        offensive_three_seconds_enabled=xp.asarray(
            1 if env.offensive_three_seconds_enabled else 0,
            dtype=xp.int8,
        ),
        pass_reward=xp.asarray(float(env.pass_reward), dtype=xp.float32),
        violation_reward=xp.asarray(float(env.violation_reward), dtype=xp.float32),
        reward_shaping_gamma=xp.asarray(float(env.reward_shaping_gamma), dtype=xp.float32),
        enable_phi_shaping=xp.asarray(1 if env.enable_phi_shaping else 0, dtype=xp.int8),
        phi_beta=xp.asarray(float(env.phi_beta), dtype=xp.float32),
        phi_blend_weight=xp.asarray(float(env.phi_blend_weight), dtype=xp.float32),
        phi_use_ball_handler_only=xp.asarray(1 if env.phi_use_ball_handler_only else 0, dtype=xp.int8),
        pass_oob_turnover_prob=xp.asarray(float(env.pass_oob_turnover_prob), dtype=xp.float32),
        assist_window=xp.asarray(float(env.assist_window), dtype=xp.float32),
        potential_assist_pct=xp.asarray(float(env.potential_assist_pct), dtype=xp.float32),
        full_assist_bonus_pct=xp.asarray(float(env.full_assist_bonus_pct), dtype=xp.float32),
        base_layup_pct=xp.asarray(float(env.layup_pct), dtype=xp.float32),
        base_three_pt_pct=xp.asarray(float(env.three_pt_pct), dtype=xp.float32),
        base_dunk_pct=xp.asarray(float(env.dunk_pct), dtype=xp.float32),
        layup_std=xp.asarray(float(env.layup_std), dtype=xp.float32),
        three_pt_std=xp.asarray(float(env.three_pt_std), dtype=xp.float32),
        dunk_std=xp.asarray(float(env.dunk_std), dtype=xp.float32),
        training_player_mask=xp.asarray(training_player_mask, dtype=xp.float32),
        training_role_flag=xp.asarray(float(training_role_flag), dtype=xp.float32),
        task_reward_scale=xp.asarray(float(getattr(env, "task_reward_scale", 1.0)), dtype=xp.float32),
        enable_intent_learning=xp.asarray(
            1 if bool(getattr(env, "enable_intent_learning", False)) else 0,
            dtype=xp.int8,
        ),
        enable_defense_intent_learning=xp.asarray(
            1 if bool(getattr(env, "enable_defense_intent_learning", False)) else 0,
            dtype=xp.int8,
        ),
        num_intents=xp.asarray(max(1, int(getattr(env, "num_intents", 8))), dtype=xp.int32),
        intent_commitment_steps=xp.asarray(
            max(1, int(getattr(env, "intent_commitment_steps", 4))),
            dtype=xp.int32,
        ),
        intent_null_prob=xp.asarray(
            float(np.clip(float(getattr(env, "intent_null_prob", 0.2)), 0.0, 1.0)),
            dtype=xp.float32,
        ),
        defense_intent_null_prob=xp.asarray(
            float(np.clip(float(getattr(env, "defense_intent_null_prob", 1.0)), 0.0, 1.0)),
            dtype=xp.float32,
        ),
        intent_visible_to_defense_prob=xp.asarray(
            float(np.clip(float(getattr(env, "intent_visible_to_defense_prob", 0.0)), 0.0, 1.0)),
            dtype=xp.float32,
        ),
    )


def _axial_to_cartesian(q, r, jnp):
    qf = q.astype(jnp.float32)
    rf = r.astype(jnp.float32)
    x = (SQRT3 * qf) + ((SQRT3 / 2.0) * rf)
    y = 1.5 * rf
    return x, y


def _hex_distance(a, b, jnp):
    q1 = a[..., 0]
    r1 = a[..., 1]
    q2 = b[..., 0]
    r2 = b[..., 1]
    return (
        jnp.abs(q1 - q2)
        + jnp.abs((q1 + r1) - (q2 + r2))
        + jnp.abs(r1 - r2)
    ) // 2


def _lookup_cell_indices(cell_coords, positions, jnp):
    matches = jnp.all(positions[..., None, :] == cell_coords, axis=-1)
    indices = jnp.argmax(matches.astype(jnp.int32), axis=-1)
    found = jnp.any(matches, axis=-1)
    return indices, found


def _safe_ball_holder_positions(state: KernelState, jnp):
    n_players = state.positions.shape[1]
    safe_holder = jnp.clip(state.ball_holder, 0, n_players - 1)
    return jnp.take_along_axis(state.positions, safe_holder[:, None, None], axis=1)[:, 0, :]


def _single_state_to_batched(state: KernelState, jnp) -> KernelState:
    return KernelState(*(jnp.expand_dims(field, axis=0) for field in state))


def _replace_state(state: KernelState, **updates) -> KernelState:
    return state._replace(**updates)


def _advance_role_intent_single(static: KernelStatic, enabled, index, active, age, remaining, jnp):
    enabled_bool = enabled.astype(jnp.bool_)
    active_bool = active.astype(jnp.bool_)
    should_expire = enabled_bool & active_bool & (remaining <= 0)
    should_advance = enabled_bool & active_bool & (remaining > 0)
    next_active = jnp.where(should_expire, jnp.asarray(0, dtype=jnp.int8), active)
    next_age = jnp.where(should_advance, age + 1, age)
    next_remaining = jnp.where(
        should_expire,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.where(should_advance, jnp.maximum(0, remaining - 1), remaining),
    )
    disabled = ~enabled_bool
    return {
        "index": jnp.where(disabled, jnp.asarray(0, dtype=jnp.int32), index),
        "active": jnp.where(disabled, jnp.asarray(0, dtype=jnp.int8), next_active),
        "age": jnp.where(disabled, jnp.asarray(0, dtype=jnp.int32), next_age),
        "commitment_remaining": jnp.where(
            disabled,
            jnp.asarray(0, dtype=jnp.int32),
            next_remaining,
        ),
    }


def advance_intent_clock_single(static: KernelStatic, state: KernelState, jnp) -> KernelState:
    offense = _advance_role_intent_single(
        static,
        static.enable_intent_learning,
        state.intent_index,
        state.intent_active,
        state.intent_age,
        state.intent_commitment_remaining,
        jnp,
    )
    defense = _advance_role_intent_single(
        static,
        static.enable_defense_intent_learning,
        state.defense_intent_index,
        state.defense_intent_active,
        state.defense_intent_age,
        state.defense_intent_commitment_remaining,
        jnp,
    )
    return _replace_state(
        state,
        intent_index=offense["index"],
        intent_active=offense["active"],
        intent_age=offense["age"],
        intent_commitment_remaining=offense["commitment_remaining"],
        defense_intent_index=defense["index"],
        defense_intent_active=defense["active"],
        defense_intent_age=defense["age"],
        defense_intent_commitment_remaining=defense["commitment_remaining"],
    )


def _team_mask_for_holder(static: KernelStatic, ball_holder, jnp):
    is_offense_holder = jnp.any(ball_holder == static.offense_ids)
    return jnp.where(
        is_offense_holder,
        static.role_encoding > 0.0,
        static.role_encoding < 0.0,
    )


def _signed_angles(base_positions, target_positions, basket_position, jnp):
    basket_delta = basket_position - base_positions
    target_delta = target_positions - base_positions
    basket_x, basket_y = _axial_to_cartesian(basket_delta[..., 0], basket_delta[..., 1], jnp)
    target_x, target_y = _axial_to_cartesian(target_delta[..., 0], target_delta[..., 1], jnp)
    basket_mag = jnp.sqrt((basket_x**2) + (basket_y**2))
    target_mag = jnp.sqrt((target_x**2) + (target_y**2))
    dot = (basket_x * target_x) + (basket_y * target_y)
    cross = (basket_x * target_y) - (basket_y * target_x)
    signed_angle = jnp.arctan2(cross, dot) / jnp.pi
    return jnp.where((basket_mag == 0.0) | (target_mag == 0.0), 0.0, signed_angle)


def _unordered_teammate_distances(team_positions, jnp):
    batch_size = team_positions.shape[0]
    team_size = team_positions.shape[1]
    parts = []
    for idx in range(team_size - 1):
        part = _hex_distance(
            team_positions[:, idx : idx + 1, :],
            team_positions[:, idx + 1 :, :],
            jnp,
        ).astype(jnp.float32)
        parts.append(part.reshape(batch_size, -1))
    if not parts:
        return jnp.zeros((batch_size, 0), dtype=jnp.float32)
    return jnp.concatenate(parts, axis=1)


def _ordered_teammate_angles(team_positions, basket_position, jnp):
    batch_size = team_positions.shape[0]
    team_size = team_positions.shape[1]
    parts = []
    for idx in range(team_size):
        target_indices = [target_idx for target_idx in range(team_size) if target_idx != idx]
        if not target_indices:
            continue
        part = _signed_angles(
            team_positions[:, idx : idx + 1, :],
            team_positions[:, target_indices, :],
            basket_position,
            jnp,
        ).astype(jnp.float32)
        parts.append(part.reshape(batch_size, -1))
    if not parts:
        return jnp.zeros((batch_size, 0), dtype=jnp.float32)
    return jnp.concatenate(parts, axis=1)


def _point_to_segment_distance_and_projection(point_x, point_y, line_x, line_y, jnp):
    line_length_sq = (line_x**2) + (line_y**2)
    safe_line_length_sq = jnp.where(line_length_sq == 0.0, 1.0, line_length_sq)
    t_raw = ((point_x * line_x) + (point_y * line_y)) / safe_line_length_sq
    t_clipped = jnp.clip(t_raw, 0.0, 1.0)
    closest_x = line_x * t_clipped
    closest_y = line_y * t_clipped
    perp_distance = jnp.sqrt(((point_x - closest_x) ** 2) + ((point_y - closest_y) ** 2))
    perp_distance = jnp.where(line_length_sq == 0.0, jnp.sqrt((point_x**2) + (point_y**2)), perp_distance)
    return perp_distance, t_raw


def build_action_masks_batch(static: KernelStatic, state: KernelState, jnp):
    batch_size, n_players, _ = state.positions.shape
    masks = jnp.zeros((batch_size, n_players, ACTION_COUNT), dtype=jnp.int8)
    masks = masks.at[:, :, ActionType.NOOP.value].set(1)

    cell_indices, found = _lookup_cell_indices(static.cell_coords, state.positions, jnp)
    move_masks = static.move_mask_by_cell[cell_indices]
    move_masks = move_masks * found[..., None].astype(jnp.int8)

    neighbor_positions = state.positions[:, :, None, :] + static.hex_directions[None, None, :, :]
    occupied = jnp.any(
        jnp.all(
            neighbor_positions[:, :, :, None, :] == state.positions[:, None, None, :, :],
            axis=-1,
        ),
        axis=-1,
    )
    occupied_move_masks = move_masks * (1 - occupied.astype(jnp.int8))
    move_masks = jnp.where(
        static.mask_occupied_moves.astype(jnp.bool_),
        occupied_move_masks,
        move_masks,
    )
    masks = masks.at[:, :, MOVE_ACTION_START:MOVE_ACTION_END].set(move_masks)

    player_ids = jnp.arange(n_players, dtype=jnp.int32)
    holder_mask = (state.ball_holder[:, None] == player_ids[None, :]) & (state.ball_holder[:, None] >= 0)
    masks = masks.at[:, :, ActionType.SHOOT.value].set(holder_mask.astype(jnp.int8))

    pass_masks = holder_mask[:, :, None].astype(jnp.int8) * static.pointer_pass_slot_mask[None, :, :]
    masks = masks.at[:, :, PASS_ACTION_START:PASS_ACTION_END].set(pass_masks)
    return masks


def build_shot_profile_batch(static: KernelStatic, state: KernelState, jnp):
    batch_size, n_players, _ = state.positions.shape
    basket = jnp.broadcast_to(static.basket_position, (batch_size, n_players, 2))
    distances = _hex_distance(state.positions, basket, jnp).astype(jnp.int32)

    cell_indices, found = _lookup_cell_indices(static.cell_coords, state.positions, jnp)
    is_three = jnp.where(
        found,
        static.three_point_by_cell[cell_indices].astype(jnp.bool_),
        jnp.zeros((batch_size, n_players), dtype=jnp.bool_),
    )

    d0 = jnp.asarray(1.0, dtype=jnp.float32)
    d1 = jnp.maximum(static.three_point_distance + 1.0, d0 + 1.0)
    distances_f = distances.astype(jnp.float32)
    t = (distances_f - d0) / (d1 - d0)
    t = jnp.clip(t, 0.0, 1.0)
    base_prob = state.layup_pct + (state.three_pt_pct - state.layup_pct) * t
    base_prob = jnp.where(distances <= 1, state.layup_pct, base_prob)
    extra_hexes = jnp.maximum(0.0, distances_f - jnp.floor(d1))
    base_prob = jnp.where(
        distances_f > d1,
        base_prob - (static.three_pt_extra_hex_decay * extra_hexes),
        base_prob,
    )
    base_prob = jnp.where(
        static.allow_dunks.astype(jnp.bool_) & (distances == 0),
        state.dunk_pct,
        base_prob,
    )
    base_prob = jnp.clip(base_prob, 0.01, 0.99)

    shooter_pos = state.positions[:, :, None, :]
    defender_pos = state.positions[:, None, :, :]
    defender_delta = defender_pos - shooter_pos
    basket_delta = static.basket_position[None, None, :] - state.positions

    dir_x, dir_y = _axial_to_cartesian(basket_delta[..., 0], basket_delta[..., 1], jnp)
    vx, vy = _axial_to_cartesian(defender_delta[..., 0], defender_delta[..., 1], jnp)
    dir_norm = jnp.sqrt((dir_x**2) + (dir_y**2))
    dir_norm = jnp.where(dir_norm == 0.0, 1.0, dir_norm)
    vnorm = jnp.sqrt((vx**2) + (vy**2))
    safe_vnorm = jnp.where(vnorm == 0.0, 1.0, vnorm)
    cosang = (vx * dir_x[:, :, None] + vy * dir_y[:, :, None]) / (safe_vnorm * dir_norm[:, :, None])
    in_arc = cosang >= static.shot_pressure_cos_threshold
    defender_distance = _hex_distance(shooter_pos, defender_pos, jnp).astype(jnp.float32)
    valid_defender = (
        static.opponent_mask[None, :, :].astype(jnp.bool_)
        & (vnorm > 0.0)
        & in_arc
        & (defender_distance <= distances_f[:, :, None])
    )

    angle_factor = (cosang - static.shot_pressure_cos_threshold) / (1.0 - static.shot_pressure_cos_threshold)
    distance_reduction = static.shot_pressure_max * jnp.exp(
        -static.shot_pressure_lambda * (defender_distance - 1.0)
    )
    pressure_reduction = distance_reduction * (angle_factor**2)
    masked_reduction = jnp.where(
        valid_defender,
        pressure_reduction,
        jnp.full_like(pressure_reduction, -jnp.inf),
    )
    best_reduction = jnp.max(masked_reduction, axis=-1)
    has_pressure = jnp.any(valid_defender, axis=-1)
    pressure_multiplier = jnp.where(
        has_pressure,
        jnp.maximum(0.0, 1.0 - best_reduction),
        jnp.ones_like(best_reduction),
    )
    pressure_multiplier = jnp.where(
        static.shot_pressure_enabled.astype(jnp.bool_),
        pressure_multiplier,
        jnp.ones_like(pressure_multiplier),
    )

    probability = jnp.clip(base_prob * pressure_multiplier, 0.01, 0.99)
    shot_value = jnp.where(
        static.allow_dunks.astype(jnp.bool_) & (distances == 0),
        jnp.full_like(probability, 2.0),
        jnp.where(is_three, jnp.full_like(probability, 3.0), jnp.full_like(probability, 2.0)),
    )
    expected_points = shot_value * probability

    return {
        "distance": distances,
        "is_three": is_three,
        "base_probability": base_prob,
        "pressure_multiplier": pressure_multiplier,
        "probability": probability,
        "shot_value": shot_value,
        "expected_points": expected_points,
    }


def build_offense_expected_points_batch(static: KernelStatic, state: KernelState, jnp):
    profile = build_shot_profile_batch(static, state, jnp)
    return jnp.take(profile["expected_points"], static.offense_ids, axis=1)


def build_turnover_probabilities_batch(static: KernelStatic, state: KernelState, jnp):
    batch_size = state.positions.shape[0]
    offense_count = static.offense_ids.shape[0]
    passer_pos = _safe_ball_holder_positions(state, jnp)
    defense_positions = jnp.take(state.positions, static.defense_ids, axis=1)

    ball_holder_offense_mask = state.ball_holder[:, None] == static.offense_ids[None, :]
    has_offense_holder = jnp.any(ball_holder_offense_mask, axis=1)

    distances = _hex_distance(passer_pos[:, None, :], defense_positions, jnp).astype(jnp.float32)
    basket_delta = static.basket_position[None, :] - passer_pos
    defender_delta = defense_positions - passer_pos[:, None, :]
    basket_x, basket_y = _axial_to_cartesian(basket_delta[..., 0], basket_delta[..., 1], jnp)
    defender_x, defender_y = _axial_to_cartesian(defender_delta[..., 0], defender_delta[..., 1], jnp)
    basket_mag = jnp.sqrt((basket_x**2) + (basket_y**2))
    defender_mag = jnp.sqrt((defender_x**2) + (defender_y**2))
    safe_den = jnp.where((basket_mag[:, None] * defender_mag) == 0.0, 1.0, basket_mag[:, None] * defender_mag)
    cos_angle = ((basket_x[:, None] * defender_x) + (basket_y[:, None] * defender_y)) / safe_den
    cos_angle = jnp.where((basket_mag[:, None] == 0.0) | (defender_mag == 0.0), 0.0, cos_angle)

    valid = has_offense_holder[:, None] & (distances <= static.defender_pressure_distance) & (cos_angle >= 0.0)
    turnover_prob = static.defender_pressure_turnover_chance * jnp.exp(
        -static.defender_pressure_decay_lambda * jnp.maximum(0.0, distances - 1.0)
    )
    turnover_prob = jnp.where(valid, turnover_prob, 0.0)
    total_turnover = 1.0 - jnp.prod(1.0 - turnover_prob, axis=1)

    out = jnp.zeros((batch_size, offense_count), dtype=jnp.float32)
    return jnp.where(ball_holder_offense_mask, total_turnover[:, None], out)


def build_pass_steal_probabilities_batch(static: KernelStatic, state: KernelState, jnp):
    batch_size = state.positions.shape[0]
    offense_count = static.offense_ids.shape[0]
    passer_pos = _safe_ball_holder_positions(state, jnp)
    offense_positions = jnp.take(state.positions, static.offense_ids, axis=1)
    defense_positions = jnp.take(state.positions, static.defense_ids, axis=1)

    ball_holder_offense_mask = state.ball_holder[:, None] == static.offense_ids[None, :]
    has_offense_holder = jnp.any(ball_holder_offense_mask, axis=1)
    valid_receivers = has_offense_holder[:, None] & (~ball_holder_offense_mask)

    line_delta = offense_positions - passer_pos[:, None, :]
    line_x, line_y = _axial_to_cartesian(line_delta[..., 0], line_delta[..., 1], jnp)
    pass_distance = _hex_distance(passer_pos[:, None, :], offense_positions, jnp).astype(jnp.float32)

    defender_delta = defense_positions[:, None, :, :] - passer_pos[:, None, None, :]
    defender_x, defender_y = _axial_to_cartesian(defender_delta[..., 0], defender_delta[..., 1], jnp)
    dot = (defender_x * line_x[:, :, None]) + (defender_y * line_y[:, :, None])
    forward_defender = dot >= 0.0

    same_as_passer = jnp.all(defense_positions[:, None, :, :] == passer_pos[:, None, None, :], axis=-1)
    same_as_receiver = jnp.all(defense_positions[:, None, :, :] == offense_positions[:, :, None, :], axis=-1)
    perp_distance, position_t = _point_to_segment_distance_and_projection(
        defender_x,
        defender_y,
        line_x[:, :, None],
        line_y[:, :, None],
        jnp,
    )
    between_endpoints = (position_t > 0.0) & (position_t < 1.0)
    position_weight = static.steal_position_weight_min + ((1.0 - static.steal_position_weight_min) * position_t)
    steal_contrib = (
        static.base_steal_rate
        * jnp.exp(-static.steal_perp_decay * perp_distance)
        * (1.0 + (static.steal_distance_factor * pass_distance[:, :, None]))
        * position_weight
    )
    steal_contrib = jnp.clip(steal_contrib, 0.0, 1.0)
    steal_contrib = jnp.where(
        valid_receivers[:, :, None]
        & forward_defender
        & between_endpoints
        & (~same_as_passer)
        & (~same_as_receiver),
        steal_contrib,
        0.0,
    )
    total_steal = 1.0 - jnp.prod(1.0 - steal_contrib, axis=-1)
    return jnp.where(valid_receivers, total_steal, jnp.zeros((batch_size, offense_count), dtype=jnp.float32))


def build_observation_vector_batch(static: KernelStatic, state: KernelState, jnp):
    batch_size = state.positions.shape[0]
    n_players = state.positions.shape[1]
    norm_den = static.court_norm_den
    positions_norm = state.positions.astype(jnp.float32).reshape(batch_size, -1) / norm_den

    player_ids = jnp.arange(n_players, dtype=jnp.int32)
    ball_holder_one_hot = (
        (state.ball_holder[:, None] == player_ids[None, :]) & (state.ball_holder[:, None] >= 0)
    ).astype(jnp.float32)

    pressure_exposure = state.pressure_exposure[:, None]
    shot_clock_den = jnp.maximum(static.shot_clock_max.astype(jnp.float32), 1.0)
    shot_clock = (state.shot_clock.astype(jnp.float32) / shot_clock_den)[:, None]
    role_encoding = jnp.broadcast_to(static.role_encoding[None, :], (batch_size, n_players))

    passer_pos = _safe_ball_holder_positions(state, jnp).astype(jnp.float32)
    ball_handler_pos = jnp.where(
        (state.ball_holder[:, None] >= 0),
        passer_pos,
        jnp.broadcast_to(static.basket_position[None, :], passer_pos.shape).astype(jnp.float32),
    ) / norm_den
    hoop_vec = jnp.broadcast_to(static.basket_position.astype(jnp.float32)[None, :] / norm_den, (batch_size, 2))

    offense_positions = jnp.take(state.positions, static.offense_ids, axis=1)
    defense_positions = jnp.take(state.positions, static.defense_ids, axis=1)
    off_def_distances = _hex_distance(
        offense_positions[:, :, None, :],
        defense_positions[:, None, :, :],
        jnp,
    ).astype(jnp.float32)
    off_def_distances = off_def_distances.reshape(batch_size, -1) / norm_den
    off_def_angles = _signed_angles(
        offense_positions[:, :, None, :],
        defense_positions[:, None, :, :],
        static.basket_position,
        jnp,
    ).astype(jnp.float32).reshape(batch_size, -1)

    teammate_distances = jnp.concatenate(
        [
            _unordered_teammate_distances(offense_positions, jnp),
            _unordered_teammate_distances(defense_positions, jnp),
        ],
        axis=1,
    ) / norm_den
    teammate_angles = jnp.concatenate(
        [
            _ordered_teammate_angles(offense_positions, static.basket_position, jnp),
            _ordered_teammate_angles(defense_positions, static.basket_position, jnp),
        ],
        axis=1,
    )

    lane_steps = jnp.where(
        static.role_encoding[None, :] > 0.0,
        state.offense_lane_steps,
        state.defense_lane_steps,
    ).astype(jnp.float32)
    ep_values = build_offense_expected_points_batch(static, state, jnp)
    turnover_probs = build_turnover_probabilities_batch(static, state, jnp)
    steal_risks = build_pass_steal_probabilities_batch(static, state, jnp)

    return jnp.concatenate(
        [
            positions_norm,
            ball_holder_one_hot,
            shot_clock,
            pressure_exposure,
            role_encoding,
            ball_handler_pos,
            hoop_vec,
            off_def_distances,
            off_def_angles,
            teammate_distances,
            teammate_angles,
            lane_steps,
            ep_values,
            turnover_probs,
            steal_risks,
        ],
        axis=1,
    )


def build_offense_skill_deltas_batch(static: KernelStatic, state: KernelState, jnp):
    layup_delta = state.layup_pct[:, static.offense_ids] - static.base_layup_pct
    three_delta = state.three_pt_pct[:, static.offense_ids] - static.base_three_pt_pct
    dunk_delta = state.dunk_pct[:, static.offense_ids] - static.base_dunk_pct
    stacked = jnp.stack([layup_delta, three_delta, dunk_delta], axis=-1)
    return stacked.reshape(stacked.shape[0], -1).astype(jnp.float32)


def build_flat_observation_batch_with_role_flag(static: KernelStatic, state: KernelState, role_flag_value, jnp):
    batch_size = state.positions.shape[0]
    role_flag = jnp.full((batch_size, 1), role_flag_value, dtype=jnp.float32)
    return jnp.concatenate(
        [
            build_observation_vector_batch(static, state, jnp),
            role_flag,
            build_offense_skill_deltas_batch(static, state, jnp),
        ],
        axis=1,
    ).astype(jnp.float32)


def build_flat_observation_batch(static: KernelStatic, state: KernelState, jnp):
    return build_flat_observation_batch_with_role_flag(
        static,
        state,
        static.training_role_flag,
        jnp,
    )


def _scatter_offense_features(static: KernelStatic, offense_values, n_players: int, jnp):
    batch_size = offense_values.shape[0]
    out = jnp.zeros((batch_size, int(n_players)), dtype=jnp.float32)
    return out.at[:, static.offense_ids].set(offense_values.astype(jnp.float32))


def _nearest_masked_distance(distance_matrix, valid_mask, jnp):
    masked = jnp.where(
        valid_mask,
        distance_matrix,
        jnp.full(distance_matrix.shape, jnp.inf, dtype=jnp.float32),
    )
    nearest = jnp.min(masked, axis=-1)
    return jnp.where(jnp.isfinite(nearest), nearest, jnp.zeros_like(nearest))


def build_token_observation_components_batch(
    static: KernelStatic,
    state: KernelState,
    role_flag_value,
    jnp,
):
    """Build set-observation components matching the production token layout."""
    batch_size, n_players, _ = state.positions.shape
    norm_den = static.court_norm_den
    positions_norm = state.positions.astype(jnp.float32) / norm_den
    player_ids = jnp.arange(n_players, dtype=jnp.int32)
    role_encoding = jnp.broadcast_to(static.role_encoding[None, :], (batch_size, n_players))
    is_offense = role_encoding > 0.0
    has_ball = (
        (state.ball_holder[:, None] == player_ids[None, :]) & (state.ball_holder[:, None] >= 0)
    ).astype(jnp.float32)

    skill_gate = is_offense.astype(jnp.float32)
    layup = state.layup_pct.astype(jnp.float32) * skill_gate
    three = state.three_pt_pct.astype(jnp.float32) * skill_gate
    dunk = state.dunk_pct.astype(jnp.float32) * skill_gate

    max_lane_steps = jnp.maximum(static.three_second_max_steps, jnp.asarray(1.0, dtype=jnp.float32))
    lane_steps = jnp.where(
        is_offense,
        state.offense_lane_steps,
        state.defense_lane_steps,
    ).astype(jnp.float32)
    lane_steps_norm = jnp.clip(lane_steps / max_lane_steps, 0.0, 1.0)

    shot_profile = build_shot_profile_batch(static, state, jnp)
    expected_points = jnp.where(
        is_offense,
        shot_profile["expected_points"].astype(jnp.float32),
        jnp.zeros((batch_size, n_players), dtype=jnp.float32),
    )
    offense_expected_points = jnp.take(expected_points, static.offense_ids, axis=1)
    best_offense_slot = jnp.argmax(offense_expected_points, axis=1)
    best_ep_player = static.offense_ids[best_offense_slot]
    best_ep_pos = jnp.take_along_axis(
        state.positions,
        best_ep_player[:, None, None],
        axis=1,
    )[:, 0, :]

    turnover_probs = _scatter_offense_features(
        static,
        build_turnover_probabilities_batch(static, state, jnp),
        n_players,
        jnp,
    )
    steal_risks = _scatter_offense_features(
        static,
        build_pass_steal_probabilities_batch(static, state, jnp),
        n_players,
        jnp,
    )

    ball_pos = _safe_ball_holder_positions(state, jnp)
    dist_to_ball = _hex_distance(state.positions, ball_pos[:, None, :], jnp).astype(jnp.float32) / norm_den
    dist_to_ball = jnp.where(state.ball_holder[:, None] >= 0, dist_to_ball, jnp.zeros_like(dist_to_ball))
    dist_to_best_ep = _hex_distance(state.positions, best_ep_pos[:, None, :], jnp).astype(jnp.float32) / norm_den

    all_distances = _hex_distance(
        state.positions[:, :, None, :],
        state.positions[:, None, :, :],
        jnp,
    ).astype(jnp.float32)
    opponent_mask = static.opponent_mask.astype(jnp.bool_)[None, :, :]
    same_team_mask = (static.role_encoding[:, None] == static.role_encoding[None, :])[None, :, :]
    not_self_mask = ~jnp.eye(n_players, dtype=jnp.bool_)[None, :, :]
    dist_to_nearest_opp = _nearest_masked_distance(all_distances, opponent_mask, jnp) / norm_den
    dist_to_nearest_team = _nearest_masked_distance(all_distances, same_team_mask & not_self_mask, jnp) / norm_den

    players = jnp.stack(
        [
            positions_norm[..., 0],
            positions_norm[..., 1],
            role_encoding,
            has_ball,
            layup,
            three,
            dunk,
            lane_steps_norm,
            expected_points,
            turnover_probs,
            steal_risks,
            dist_to_ball,
            dist_to_best_ep,
            dist_to_nearest_opp,
            dist_to_nearest_team,
        ],
        axis=-1,
    ).astype(jnp.float32)
    globals_vec = jnp.stack(
        [
            state.shot_clock.astype(jnp.float32)
            / jnp.maximum(static.shot_clock_max.astype(jnp.float32), 1.0),
            state.pressure_exposure.astype(jnp.float32),
            jnp.full((batch_size,), static.basket_position[0].astype(jnp.float32) / norm_den, dtype=jnp.float32),
            jnp.full((batch_size,), static.basket_position[1].astype(jnp.float32) / norm_den, dtype=jnp.float32),
        ],
        axis=-1,
    ).astype(jnp.float32)
    role_flag = jnp.full((batch_size, 1), role_flag_value, dtype=jnp.float32)
    return players, globals_vec, role_flag


def build_token_observation_batch_with_role_flag(
    static: KernelStatic,
    state: KernelState,
    role_flag_value,
    jnp,
):
    players, globals_vec, role_flag = build_token_observation_components_batch(
        static,
        state,
        role_flag_value,
        jnp,
    )
    return jnp.concatenate(
        [
            players.reshape(players.shape[0], -1),
            globals_vec,
            role_flag,
        ],
        axis=1,
    ).astype(jnp.float32)


def build_token_observation_batch(static: KernelStatic, state: KernelState, jnp):
    return build_token_observation_batch_with_role_flag(
        static,
        state,
        static.training_role_flag,
        jnp,
    )


def build_policy_observation_batch_with_role_flag(
    static: KernelStatic,
    state: KernelState,
    role_flag_value,
    jnp,
    *,
    model_type: str,
):
    if str(model_type) == "attention":
        return build_token_observation_batch_with_role_flag(
            static,
            state,
            role_flag_value,
            jnp,
        )
    return build_flat_observation_batch_with_role_flag(
        static,
        state,
        role_flag_value,
        jnp,
    )


def build_policy_observation_batch(
    static: KernelStatic,
    state: KernelState,
    jnp,
    *,
    model_type: str,
):
    return build_policy_observation_batch_with_role_flag(
        static,
        state,
        static.training_role_flag,
        jnp,
        model_type=model_type,
    )


def build_policy_intent_context_batch_with_role_flag(
    static: KernelStatic,
    state: KernelState,
    role_flag_value,
    jnp,
) -> dict[str, Any]:
    """Return the runtime intent context consumed by intent-conditioned policies."""
    batch_size = state.positions.shape[0]
    role_flag = jnp.full((batch_size,), role_flag_value, dtype=jnp.float32)
    is_offense = role_flag > 0.0
    offense_gate = (
        static.enable_intent_learning.astype(jnp.bool_)
        & state.intent_active.astype(jnp.bool_)
    )
    defense_gate = (
        static.enable_defense_intent_learning.astype(jnp.bool_)
        & state.defense_intent_active.astype(jnp.bool_)
    )
    intent_index = jnp.where(is_offense, state.intent_index, state.defense_intent_index)
    intent_gate = jnp.where(is_offense, offense_gate, defense_gate)
    return {
        "intent_index": intent_index.astype(jnp.int32),
        "intent_gate": intent_gate.astype(jnp.float32),
    }


def build_policy_intent_context_batch(
    static: KernelStatic,
    state: KernelState,
    jnp,
) -> dict[str, Any]:
    return build_policy_intent_context_batch_with_role_flag(
        static,
        state,
        static.training_role_flag,
        jnp,
    )


def build_aggregated_reward_batch(static: KernelStatic, rewards, jnp):
    scaled = rewards.astype(jnp.float32) * static.training_player_mask[None, :]
    return jnp.sum(scaled, axis=1) * static.task_reward_scale


def _turnover_to_defense_single(static: KernelStatic, positions, from_player, jnp):
    from_pos = positions[from_player]
    offense_turnover = static.role_encoding[from_player] > 0.0
    candidate_ids = jnp.where(offense_turnover, static.defense_ids, static.offense_ids)
    candidate_positions = positions[candidate_ids]
    distances = _hex_distance(from_pos[None, :], candidate_positions, jnp)
    nearest_idx = jnp.argmin(distances)
    return candidate_ids[nearest_idx]


def _pass_steal_probs_single(static: KernelStatic, state: KernelState, jnp):
    batched_state = _single_state_to_batched(state, jnp)
    return build_pass_steal_probabilities_batch(static, batched_state, jnp)[0]


def _pressure_turnover_probs_single(static: KernelStatic, state: KernelState, jnp):
    batched_state = _single_state_to_batched(state, jnp)
    offense_probs = build_turnover_probabilities_batch(static, batched_state, jnp)[0]
    total_prob = jnp.max(offense_probs)

    passer_pos = state.positions[jnp.clip(state.ball_holder, 0, state.positions.shape[0] - 1)]
    defense_positions = state.positions[static.defense_ids]
    distances = _hex_distance(passer_pos[None, :], defense_positions, jnp).astype(jnp.float32)
    basket_delta = static.basket_position - passer_pos
    defender_delta = defense_positions - passer_pos[None, :]
    basket_x, basket_y = _axial_to_cartesian(basket_delta[0], basket_delta[1], jnp)
    defender_x, defender_y = _axial_to_cartesian(defender_delta[:, 0], defender_delta[:, 1], jnp)
    basket_mag = jnp.sqrt((basket_x**2) + (basket_y**2))
    defender_mag = jnp.sqrt((defender_x**2) + (defender_y**2))
    safe_den = jnp.where((basket_mag * defender_mag) == 0.0, 1.0, basket_mag * defender_mag)
    cos_angle = ((basket_x * defender_x) + (basket_y * defender_y)) / safe_den
    cos_angle = jnp.where((basket_mag == 0.0) | (defender_mag == 0.0), 0.0, cos_angle)
    valid = (
        jnp.any(state.ball_holder == static.offense_ids)
        & (distances <= static.defender_pressure_distance)
        & (cos_angle >= 0.0)
    )
    per_defender = static.defender_pressure_turnover_chance * jnp.exp(
        -static.defender_pressure_decay_lambda * jnp.maximum(0.0, distances - 1.0)
    )
    return jnp.where(valid, per_defender, 0.0), total_prob


def _positions_in_lane(static: KernelStatic, positions, lane_mask, jnp):
    cell_indices, found = _lookup_cell_indices(static.cell_coords, positions, jnp)
    return found & lane_mask[cell_indices].astype(jnp.bool_)


def _defender_guarding_offense_mask(static: KernelStatic, positions, jnp):
    defense_positions = positions[static.defense_ids]
    offense_positions = positions[static.offense_ids]
    distances = _hex_distance(
        defense_positions[:, None, :],
        offense_positions[None, :, :],
        jnp,
    ).astype(jnp.float32)
    guarding = jnp.any(distances <= static.defender_guard_distance, axis=1)
    return jnp.where(static.defender_guard_distance > 0.0, guarding, jnp.zeros_like(guarding))


def _apply_offensive_three_seconds_single(static: KernelStatic, state: KernelState, actions, jnp):
    n_players = state.positions.shape[0]
    player_ids = jnp.arange(n_players, dtype=jnp.int32)
    enabled = static.offensive_three_seconds_enabled.astype(jnp.bool_)
    in_lane = _positions_in_lane(static, state.positions, static.offensive_lane_by_cell, jnp)
    offense_mask = static.role_encoding > 0.0
    updated_steps = jnp.where(
        offense_mask & in_lane,
        state.offense_lane_steps + 1.0,
        jnp.zeros_like(state.offense_lane_steps),
    )
    updated_steps = jnp.where(enabled, updated_steps, state.offense_lane_steps)

    has_ball = player_ids == state.ball_holder
    threshold = static.three_second_max_steps
    non_holder_violation = offense_mask & in_lane & (~has_ball) & (updated_steps >= threshold)
    holder_violation = (
        offense_mask
        & in_lane
        & has_ball
        & (updated_steps > threshold)
        & (actions != ActionType.SHOOT.value)
    )
    violation_mask = enabled & (non_holder_violation | holder_violation)
    has_violation = jnp.any(violation_mask)
    violating_player = jnp.argmax(violation_mask.astype(jnp.int32)).astype(jnp.int32)
    turnover_from = jnp.where(
        state.ball_holder >= 0,
        state.ball_holder,
        violating_player,
    )
    new_holder = jnp.where(
        has_violation,
        _turnover_to_defense_single(static, state.positions, turnover_from, jnp),
        state.ball_holder,
    )
    next_state = _replace_state(
        state,
        ball_holder=new_holder,
        offense_lane_steps=updated_steps,
    )
    return next_state, has_violation, jnp.where(has_violation, violating_player, jnp.asarray(-1, dtype=jnp.int32))


def _apply_defensive_lane_rule_single(static: KernelStatic, state: KernelState, *, shot_active, jnp):
    enabled = static.illegal_defense_enabled.astype(jnp.bool_) & (~shot_active)
    n_players = state.positions.shape[0]
    player_ids = jnp.arange(n_players, dtype=jnp.int32)
    defense_mask = static.role_encoding < 0.0
    in_lane = _positions_in_lane(static, state.positions, static.defensive_lane_by_cell, jnp)
    guarding_by_defender = _defender_guarding_offense_mask(static, state.positions, jnp)
    guarding = jnp.zeros((n_players,), dtype=jnp.bool_)
    guarding = guarding.at[static.defense_ids].set(guarding_by_defender)
    should_increment = defense_mask & in_lane & (~guarding)
    updated_steps = jnp.where(
        should_increment,
        state.defense_lane_steps + 1.0,
        jnp.zeros_like(state.defense_lane_steps),
    )
    updated_steps = jnp.where(enabled, updated_steps, state.defense_lane_steps)
    violation_mask = enabled & should_increment & (updated_steps > static.three_second_max_steps)
    has_violation = jnp.any(violation_mask)
    violating_player = jnp.argmax(violation_mask.astype(jnp.int32)).astype(jnp.int32)
    updated_steps = jnp.where(
        has_violation & (player_ids == violating_player),
        jnp.zeros_like(updated_steps),
        updated_steps,
    )
    next_state = _replace_state(
        state,
        defense_lane_steps=updated_steps,
        offense_score=state.offense_score + has_violation.astype(jnp.float32),
    )
    return next_state, has_violation, jnp.where(has_violation, violating_player, jnp.asarray(-1, dtype=jnp.int32))


def _resolve_movement_single(static: KernelStatic, state: KernelState, actions, key, jax, jnp):
    n_players = state.positions.shape[0]
    current_positions = state.positions
    ball_holder = state.ball_holder
    move_keys = jax.random.uniform(key, shape=(n_players,), minval=0.0, maxval=1.0)

    intended_dest = current_positions
    requested_move = (actions >= MOVE_ACTION_START) & (actions < MOVE_ACTION_END)
    direction_idx = jnp.clip(actions - MOVE_ACTION_START, 0, 5)
    deltas = static.hex_directions[direction_idx]
    proposed = current_positions + deltas

    _, proposed_found = _lookup_cell_indices(static.cell_coords, proposed, jnp)
    basket_collision = jnp.all(proposed == static.basket_position, axis=-1) & (~static.allow_dunks.astype(jnp.bool_))
    valid_move = requested_move & proposed_found & (~basket_collision)
    intended_dest = jnp.where(valid_move[:, None], proposed, intended_dest)

    ball_holder_turnover = requested_move & (~valid_move) & (jnp.arange(n_players) == ball_holder)
    turnover_any = jnp.any(ball_holder_turnover)
    turnover_player = jnp.argmax(ball_holder_turnover.astype(jnp.int32))
    ball_holder = jnp.where(
        turnover_any,
        _turnover_to_defense_single(static, current_positions, turnover_player, jnp),
        ball_holder,
    )

    occupied_start = jnp.all(intended_dest[:, None, :] == current_positions[None, :, :], axis=-1)
    occupied_by_other = requested_move & jnp.any(
        occupied_start & (~jnp.eye(n_players, dtype=jnp.bool_)),
        axis=1,
    )
    valid_move = valid_move & (~occupied_by_other)
    intended_dest = jnp.where(valid_move[:, None], proposed, current_positions)

    final_positions = current_positions
    player_ids = jnp.arange(n_players)
    for dest_idx in range(static.cell_coords.shape[0]):
        dest = static.cell_coords[dest_idx]
        contenders = valid_move & jnp.all(intended_dest == dest, axis=-1)
        static_occupant = (~valid_move) & jnp.all(current_positions == dest, axis=-1)
        move_count = jnp.sum(contenders.astype(jnp.int32))
        winner_idx = jnp.argmax(jnp.where(contenders, move_keys, -jnp.ones_like(move_keys)))
        winner_mask = contenders & (player_ids == winner_idx)
        single_move = contenders & (move_count == 1) & (~jnp.any(static_occupant))
        collision_move = winner_mask & (move_count > 1) & (~jnp.any(static_occupant))
        applied = single_move | collision_move
        final_positions = jnp.where(applied[:, None], jnp.broadcast_to(dest, final_positions.shape), final_positions)

    turnover_player = jnp.where(
        turnover_any,
        turnover_player,
        jnp.asarray(-1, dtype=jnp.int32),
    )
    return final_positions, ball_holder, turnover_any, turnover_player


def _step_single_minimal(static: KernelStatic, state: KernelState, actions, key, jax, jnp):
    zero_rewards = jnp.zeros((state.positions.shape[0],), dtype=jnp.float32)
    zero_flag = jnp.asarray(0, dtype=jnp.int8)
    zero_steps = jnp.asarray(0, dtype=jnp.int32)
    zero_float = jnp.asarray(0.0, dtype=jnp.float32)
    no_player = jnp.asarray(-1, dtype=jnp.int32)
    no_reason = jnp.asarray(TURNOVER_REASON_NONE, dtype=jnp.int32)

    def _already_done(_):
        return StepBatchOutput(
            state=state,
            rewards=zero_rewards,
            done=jnp.asarray(True),
            pass_attempt=zero_flag,
            pass_passer=no_player,
            pass_receiver=no_player,
            completed_pass=zero_flag,
            assist=zero_flag,
            turnover=zero_flag,
            terminal_episode_steps=zero_steps,
            shot_attempt=zero_flag,
            shot_success=zero_flag,
            shot_shooter=no_player,
            shot_value=zero_float,
            shot_expected_points=zero_float,
            shot_distance=zero_float,
            shot_type=jnp.asarray(SHOT_TYPE_NONE, dtype=jnp.int32),
            shot_q=zero_steps,
            shot_r=zero_steps,
            potential_assist=zero_flag,
            assist_passer=no_player,
            turnover_player=no_player,
            turnover_reason=no_reason,
            offensive_three_seconds=zero_flag,
            defensive_lane_violation=zero_flag,
            defensive_lane_violation_player=no_player,
        )

    def _run_active(_):
        pressure_key, action_key, move_key = jax.random.split(key, 3)
        next_state = _replace_state(
            state,
            step_count=state.step_count + 1,
            episode_ended=jnp.asarray(0, dtype=state.episode_ended.dtype),
        )
        next_state = advance_intent_clock_single(static, next_state, jnp)

        pressure_probs, total_pressure_prob = _pressure_turnover_probs_single(static, next_state, jnp)
        next_state = _replace_state(
            next_state,
            pressure_exposure=next_state.pressure_exposure + jnp.maximum(0.0, total_pressure_prob),
        )
        pressure_draws = jax.random.uniform(pressure_key, shape=pressure_probs.shape)
        pressure_success = pressure_draws < pressure_probs
        pressure_turnover = jnp.any(pressure_success)
        pressure_def_idx = jnp.argmax(pressure_success.astype(jnp.int32))
        pressure_holder = static.defense_ids[pressure_def_idx]

        def _pressure_done(_):
            pressure_turnover_player = jnp.clip(next_state.ball_holder, 0, next_state.positions.shape[0] - 1)
            pressure_state = _replace_state(
                next_state,
                ball_holder=pressure_holder,
                episode_ended=jnp.asarray(1, dtype=next_state.episode_ended.dtype),
            )
            return StepBatchOutput(
                state=pressure_state,
                rewards=zero_rewards,
                done=jnp.asarray(True),
                pass_attempt=zero_flag,
                pass_passer=no_player,
                pass_receiver=no_player,
                completed_pass=zero_flag,
                assist=zero_flag,
                turnover=jnp.asarray(1, dtype=jnp.int8),
                terminal_episode_steps=pressure_state.step_count.astype(jnp.int32),
                shot_attempt=zero_flag,
                shot_success=zero_flag,
                shot_shooter=no_player,
                shot_value=zero_float,
                shot_expected_points=zero_float,
                shot_distance=zero_float,
                shot_type=jnp.asarray(SHOT_TYPE_NONE, dtype=jnp.int32),
                shot_q=zero_steps,
                shot_r=zero_steps,
                potential_assist=zero_flag,
                assist_passer=no_player,
                turnover_player=jnp.where(next_state.ball_holder >= 0, pressure_turnover_player, no_player),
                turnover_reason=jnp.asarray(TURNOVER_REASON_DEFENDER_PRESSURE, dtype=jnp.int32),
                offensive_three_seconds=zero_flag,
                defensive_lane_violation=zero_flag,
                defensive_lane_violation_player=no_player,
            )

        def _normal_step(_):
            shot_clock_state = _replace_state(next_state, shot_clock=next_state.shot_clock - 1)
            ball_holder = shot_clock_state.ball_holder
            safe_holder = jnp.clip(ball_holder, 0, actions.shape[0] - 1)
            holder_action = actions[safe_holder]
            holder_has_ball = ball_holder >= 0
            is_shot = holder_has_ball & (holder_action == ActionType.SHOOT.value)
            is_pass = holder_has_ball & (holder_action >= PASS_ACTION_START) & (holder_action < PASS_ACTION_END)
            pass_attempt = is_pass.astype(jnp.int8)

            shot_key, pass_key = jax.random.split(action_key)
            positions_after = shot_clock_state.positions
            ball_holder_after = shot_clock_state.ball_holder
            assist_active = shot_clock_state.assist_active
            assist_passer = shot_clock_state.assist_passer
            assist_recipient = shot_clock_state.assist_recipient
            assist_expires_at = shot_clock_state.assist_expires_at
            rewards = zero_rewards
            shot_active = jnp.asarray(False)
            shot_success = jnp.asarray(False)
            shot_expected_points = jnp.asarray(0.0, dtype=jnp.float32)
            shot_shooter = safe_holder
            turnover_from_action = jnp.asarray(False)
            pass_success = jnp.asarray(False)

            if_state = _single_state_to_batched(shot_clock_state, jnp)
            shot_profile = build_shot_profile_batch(static, if_state, jnp)
            shot_probabilities = shot_profile["probability"][0]
            shot_values = shot_profile["shot_value"][0]
            shot_ep_all = shot_profile["expected_points"][0]
            shot_distances = shot_profile["distance"][0]
            shot_is_three = shot_profile["is_three"][0]

            def _do_shot(_):
                draw = jax.random.uniform(shot_key)
                success = draw < shot_probabilities[safe_holder]
                new_holder = jnp.where(success, ball_holder_after, jnp.asarray(-1, dtype=jnp.int32))
                return (
                    new_holder,
                    assist_active,
                    assist_passer,
                    assist_recipient,
                    assist_expires_at,
                    jnp.asarray(True),
                    success,
                    shot_values[safe_holder],
                    shot_ep_all[safe_holder],
                    shot_distances[safe_holder].astype(jnp.float32),
                    jnp.asarray(False),
                    jnp.asarray(False),
                    no_reason,
                    no_player,
                )

            def _do_pass(_):
                slot_idx = holder_action - PASS_ACTION_START
                receiver = static.pointer_pass_target_ids[safe_holder, jnp.clip(slot_idx, 0, 5)]
                pass_probs = _pass_steal_probs_single(static, shot_clock_state, jnp)
                pass_draw = jax.random.uniform(pass_key)
                receiver_safe = jnp.clip(receiver, 0, pass_probs.shape[0] - 1)
                steal_prob = jnp.where(receiver >= 0, pass_probs[receiver_safe], 0.0)
                theft = (receiver < 0) | (pass_draw < steal_prob)
                steal_holder = _turnover_to_defense_single(static, shot_clock_state.positions, safe_holder, jnp)
                new_holder = jnp.where(theft, steal_holder, receiver)
                new_assist_active = jnp.where(theft, jnp.asarray(0, dtype=jnp.int8), jnp.asarray(1, dtype=jnp.int8))
                new_assist_passer = jnp.where(theft, jnp.asarray(-1, dtype=jnp.int32), safe_holder)
                new_assist_recipient = jnp.where(theft, jnp.asarray(-1, dtype=jnp.int32), receiver)
                new_assist_expires = jnp.where(
                    theft,
                    jnp.asarray(-1, dtype=jnp.int32),
                    shot_clock_state.step_count + static.assist_window.astype(jnp.int32),
                )
                turnover_reason = jnp.where(
                    receiver < 0,
                    jnp.asarray(TURNOVER_REASON_PASS_OUT_OF_BOUNDS, dtype=jnp.int32),
                    jnp.asarray(TURNOVER_REASON_INTERCEPTED, dtype=jnp.int32),
                )
                return (
                    new_holder,
                    new_assist_active,
                    new_assist_passer,
                    new_assist_recipient,
                    new_assist_expires,
                    jnp.asarray(False),
                    jnp.asarray(False),
                    jnp.asarray(0.0, dtype=jnp.float32),
                    jnp.asarray(0.0, dtype=jnp.float32),
                    jnp.asarray(0.0, dtype=jnp.float32),
                    theft,
                    ~theft,
                    turnover_reason,
                    receiver.astype(jnp.int32),
                )

            (
                ball_holder_after,
                assist_active,
                assist_passer,
                assist_recipient,
                assist_expires_at,
                shot_active,
                shot_success,
                shot_value,
                shot_expected_points,
                shot_distance,
                turnover_from_action,
                pass_success,
                action_turnover_reason,
                pass_receiver,
            ) = jax.lax.cond(
                is_shot,
                _do_shot,
                lambda _: jax.lax.cond(
                    is_pass,
                    _do_pass,
                    lambda __: (
                        ball_holder_after,
                        assist_active,
                        assist_passer,
                        assist_recipient,
                        assist_expires_at,
                        jnp.asarray(False),
                        jnp.asarray(False),
                        jnp.asarray(0.0, dtype=jnp.float32),
                        jnp.asarray(0.0, dtype=jnp.float32),
                        jnp.asarray(0.0, dtype=jnp.float32),
                        jnp.asarray(False),
                        jnp.asarray(False),
                        no_reason,
                        no_player,
                    ),
                    operand=None,
                ),
                operand=None,
            )

            movement_skipped = shot_active | turnover_from_action
            positions_after, ball_holder_after, movement_turnover, movement_turnover_player = jax.lax.cond(
                movement_skipped,
                lambda _: (positions_after, ball_holder_after, jnp.asarray(False), no_player),
                lambda _: _resolve_movement_single(
                    static,
                    _replace_state(shot_clock_state, positions=positions_after, ball_holder=ball_holder_after),
                    actions,
                    move_key,
                    jax,
                    jnp,
                ),
                operand=None,
            )

            final_state = _replace_state(
                shot_clock_state,
                positions=positions_after,
                ball_holder=ball_holder_after,
                assist_active=assist_active,
                assist_passer=assist_passer,
                assist_recipient=assist_recipient,
                assist_expires_at=assist_expires_at,
            )
            shooter_is_offense = static.role_encoding[shot_shooter] > 0.0
            scored_points = shot_value * shot_success.astype(jnp.float32)
            final_state = _replace_state(
                final_state,
                offense_score=final_state.offense_score + jnp.where(shooter_is_offense, scored_points, 0.0),
                defense_score=final_state.defense_score + jnp.where(shooter_is_offense, 0.0, scored_points),
            )

            per_team_pass = static.pass_reward / static.offense_ids.shape[0]
            offense_mask = static.role_encoding > 0.0
            rewards = rewards + (
                jnp.where(offense_mask, per_team_pass, -per_team_pass) * pass_success.astype(jnp.float32)
            )

            done = turnover_from_action | movement_turnover | shot_active
            per_team_shot = shot_expected_points / static.offense_ids.shape[0]
            rewards = rewards + (
                jnp.where(offense_mask, per_team_shot, -per_team_shot) * shot_active.astype(jnp.float32)
            )
            assist_valid = (
                final_state.assist_active.astype(jnp.bool_)
                & (final_state.assist_recipient == shot_shooter)
                & (final_state.step_count <= final_state.assist_expires_at)
            )
            assist_event = (assist_valid & shot_active & shot_success).astype(jnp.int8)
            potential_amt = static.potential_assist_pct * shot_expected_points
            full_amt = static.full_assist_bonus_pct * shot_expected_points
            rewards = rewards + (
                jnp.where(
                    offense_mask,
                    potential_amt / static.offense_ids.shape[0],
                    -potential_amt / static.offense_ids.shape[0],
                )
                * (assist_valid & shot_active).astype(jnp.float32)
            )
            rewards = rewards + (
                jnp.where(
                    offense_mask,
                    full_amt / static.offense_ids.shape[0],
                    -full_amt / static.offense_ids.shape[0],
                )
                * (assist_valid & shot_active & shot_success).astype(jnp.float32)
            )
            final_state = _replace_state(
                final_state,
                assist_active=jnp.where(shot_active, jnp.asarray(0, dtype=jnp.int8), final_state.assist_active),
                assist_passer=jnp.where(shot_active, jnp.asarray(-1, dtype=jnp.int32), final_state.assist_passer),
                assist_recipient=jnp.where(
                    shot_active,
                    jnp.asarray(-1, dtype=jnp.int32),
                    final_state.assist_recipient,
                ),
                assist_expires_at=jnp.where(
                    shot_active,
                    jnp.asarray(-1, dtype=jnp.int32),
                    final_state.assist_expires_at,
                ),
            )

            (
                final_state,
                offensive_three_seconds_turnover,
                offensive_three_seconds_player,
            ) = jax.lax.cond(
                shot_active | turnover_from_action,
                lambda _: (
                    final_state,
                    jnp.asarray(False),
                    no_player,
                ),
                lambda _: _apply_offensive_three_seconds_single(
                    static,
                    final_state,
                    actions,
                    jnp,
                ),
                operand=None,
            )
            (
                final_state,
                defensive_lane_violation,
                defensive_lane_violation_player,
            ) = _apply_defensive_lane_rule_single(
                static,
                final_state,
                shot_active=shot_active,
                jnp=jnp,
            )
            per_team_violation = static.violation_reward / static.offense_ids.shape[0]
            rewards = rewards + (
                jnp.where(offense_mask, per_team_violation, -per_team_violation)
                * defensive_lane_violation.astype(jnp.float32)
            )

            shot_clock_turnover = final_state.shot_clock <= 0
            done = done | offensive_three_seconds_turnover | defensive_lane_violation
            turnover_event = (
                turnover_from_action
                | movement_turnover
                | offensive_three_seconds_turnover
                | shot_clock_turnover
            ).astype(jnp.int8)
            turnover_player = jnp.where(
                turnover_from_action,
                safe_holder,
                jnp.where(
                    movement_turnover,
                    movement_turnover_player,
                    jnp.where(
                        offensive_three_seconds_turnover,
                        offensive_three_seconds_player,
                        jnp.where(shot_clock_turnover, safe_holder, no_player),
                    ),
                ),
            )
            turnover_reason = jnp.where(
                turnover_from_action,
                action_turnover_reason,
                jnp.where(
                    movement_turnover,
                    jnp.asarray(TURNOVER_REASON_MOVE_OUT_OF_BOUNDS, dtype=jnp.int32),
                    jnp.where(
                        offensive_three_seconds_turnover,
                        jnp.asarray(TURNOVER_REASON_OFFENSIVE_THREE_SECONDS, dtype=jnp.int32),
                        jnp.where(
                            shot_clock_turnover,
                            jnp.asarray(TURNOVER_REASON_SHOT_CLOCK, dtype=jnp.int32),
                            no_reason,
                        ),
                    ),
                ),
            )
            turnover_player = jnp.where(turnover_event.astype(jnp.bool_), turnover_player, no_player)
            turnover_reason = jnp.where(turnover_event.astype(jnp.bool_), turnover_reason, no_reason)
            done = done | shot_clock_turnover
            final_state = _replace_state(
                final_state,
                episode_ended=done.astype(final_state.episode_ended.dtype),
            )
            shot_position = shot_clock_state.positions[shot_shooter]
            shot_type = jnp.where(
                shot_active,
                jnp.where(
                    shot_distance <= 0.0,
                    jnp.asarray(SHOT_TYPE_DUNK, dtype=jnp.int32),
                    jnp.where(
                        shot_is_three[safe_holder].astype(jnp.bool_),
                        jnp.asarray(SHOT_TYPE_THREE, dtype=jnp.int32),
                        jnp.asarray(SHOT_TYPE_TWO, dtype=jnp.int32),
                    ),
                ),
                jnp.asarray(SHOT_TYPE_NONE, dtype=jnp.int32),
            )
            potential_assist_event = (assist_valid & shot_active).astype(jnp.int8)
            event_assist_passer = jnp.where(potential_assist_event.astype(jnp.bool_), assist_passer, no_player)
            return StepBatchOutput(
                state=final_state,
                rewards=rewards,
                done=done,
                pass_attempt=pass_attempt,
                pass_passer=jnp.where(is_pass, safe_holder.astype(jnp.int32), no_player),
                pass_receiver=jnp.where(is_pass, pass_receiver.astype(jnp.int32), no_player),
                completed_pass=pass_success.astype(jnp.int8),
                assist=assist_event,
                turnover=turnover_event,
                terminal_episode_steps=jnp.where(
                    done,
                    final_state.step_count.astype(jnp.int32),
                    jnp.asarray(0, dtype=jnp.int32),
                ),
                shot_attempt=shot_active.astype(jnp.int8),
                shot_success=shot_success.astype(jnp.int8),
                shot_shooter=jnp.where(shot_active, shot_shooter.astype(jnp.int32), no_player),
                shot_value=jnp.where(shot_active, shot_value, zero_float),
                shot_expected_points=jnp.where(shot_active, shot_expected_points, zero_float),
                shot_distance=jnp.where(shot_active, shot_distance, zero_float),
                shot_type=shot_type,
                shot_q=jnp.where(shot_active, shot_position[0], zero_steps),
                shot_r=jnp.where(shot_active, shot_position[1], zero_steps),
                potential_assist=potential_assist_event,
                assist_passer=event_assist_passer,
                turnover_player=turnover_player,
                turnover_reason=turnover_reason,
                offensive_three_seconds=offensive_three_seconds_turnover.astype(jnp.int8),
                defensive_lane_violation=defensive_lane_violation.astype(jnp.int8),
                defensive_lane_violation_player=defensive_lane_violation_player,
            )

        return jax.lax.cond(pressure_turnover, _pressure_done, _normal_step, operand=None)

    return jax.lax.cond(state.episode_ended.astype(jnp.bool_), _already_done, _run_active, operand=None)


def step_batch_minimal(static: KernelStatic, state: KernelState, actions, rng_keys, jax, jnp):
    per_state = lambda state_single, action_single, key_single: _step_single_minimal(
        static,
        state_single,
        action_single,
        key_single,
        jax,
        jnp,
    )
    return jax.vmap(per_state)(state, actions, rng_keys)


def resolve_team_player_ids(static, jax, jnp):
    is_training_offense = static.training_role_flag > 0.0
    training_ids = jax.lax.cond(
        is_training_offense,
        lambda _: static.offense_ids,
        lambda _: static.defense_ids,
        operand=None,
    )
    opponent_ids = jax.lax.cond(
        is_training_offense,
        lambda _: static.defense_ids,
        lambda _: static.offense_ids,
        operand=None,
    )
    return training_ids.astype(jnp.int32), opponent_ids.astype(jnp.int32)


def masked_categorical_actions_jax(logits, action_mask, sample_key, jax, jnp):
    legal = action_mask > 0
    has_legal = jnp.any(legal, axis=-1, keepdims=True)
    noop_mask = jnp.zeros_like(legal)
    noop_mask = noop_mask.at[..., 0].set(True)
    effective_legal = jnp.where(has_legal, legal, noop_mask)
    masked_logits = jnp.where(effective_legal, logits, jnp.full_like(logits, -1.0e9))
    sampled = jax.random.categorical(sample_key, masked_logits, axis=-1).astype(jnp.int32)
    deterministic = jnp.argmax(masked_logits, axis=-1).astype(jnp.int32)
    return sampled, deterministic, masked_logits


def sample_uniform_legal_actions_jax(action_mask, sample_key, jax, jnp):
    zero_logits = jnp.zeros(action_mask.shape, dtype=jnp.float32)
    sampled, _, _ = masked_categorical_actions_jax(
        zero_logits,
        action_mask,
        sample_key,
        jax,
        jnp,
    )
    return sampled


def assemble_full_actions_jax(training_actions, opponent_actions, training_ids, opponent_ids, n_players: int, jnp):
    batch_size = training_actions.shape[0]
    full_actions = jnp.zeros((batch_size, int(n_players)), dtype=jnp.int32)
    full_actions = full_actions.at[:, training_ids].set(training_actions)
    full_actions = full_actions.at[:, opponent_ids].set(opponent_actions)
    return full_actions


def replace_done_states(next_state: KernelState, reset_state: KernelState, done, jnp):
    done_bool = done.astype(jnp.bool_)
    replaced = []
    for current_value, reset_value in zip(next_state, reset_state):
        if getattr(current_value, "ndim", 0) <= 1:
            replaced.append(jnp.where(done_bool, reset_value, current_value))
        else:
            expand_shape = (done_bool.shape[0],) + (1,) * (current_value.ndim - 1)
            done_expand = done_bool.reshape(expand_shape)
            replaced.append(jnp.where(done_expand, reset_value, current_value))
    return KernelState(*replaced)


def set_offense_intent_state_batch(
    static: KernelStatic,
    state: KernelState,
    intent_index,
    intent_active,
    jnp,
) -> KernelState:
    """Batch equivalent of the Python env's set_offense_intent_state helper."""
    enabled = static.enable_intent_learning.astype(jnp.bool_)
    active = enabled & jnp.asarray(intent_active).astype(jnp.bool_)
    safe_index = jnp.clip(
        jnp.asarray(intent_index, dtype=jnp.int32),
        0,
        static.num_intents.astype(jnp.int32) - 1,
    )
    return _replace_state(
        state,
        intent_index=jnp.where(active, safe_index, jnp.asarray(0, dtype=jnp.int32)),
        intent_active=active.astype(jnp.int8),
        intent_age=jnp.zeros_like(state.intent_age, dtype=jnp.int32),
        intent_commitment_remaining=jnp.where(
            active,
            static.intent_commitment_steps.astype(jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
        ),
    )


def _sample_index_from_mask(mask, key, jax, jnp):
    mask_bool = mask.astype(jnp.bool_)
    logits = jnp.where(mask_bool, jnp.zeros(mask_bool.shape, dtype=jnp.float32), -jnp.inf)
    return jax.random.categorical(key, logits, axis=-1).astype(jnp.int32)


def _sample_unique_indices_from_mask(mask, count: int, key, jax, jnp):
    gumbels = jax.random.gumbel(key, shape=mask.shape, dtype=jnp.float32)
    masked_scores = jnp.where(mask.astype(jnp.bool_), gumbels, -jnp.inf)
    _, indices = jax.lax.top_k(masked_scores, int(count))
    return indices.astype(jnp.int32)


def _defender_spawn_candidate_mask(static: KernelStatic, offense_cell_idx, taken_mask, jnp):
    offense_dist = static.basket_distance_by_cell[offense_cell_idx].astype(jnp.float32)
    dist_to_offense = static.cell_distance_matrix[:, offense_cell_idx].astype(jnp.float32)
    non_basket_available = static.non_basket_cell_mask.astype(jnp.bool_) & (~taken_mask)
    within_max = jnp.where(
        static.max_spawn_distance_enabled.astype(jnp.bool_),
        static.basket_distance_by_cell.astype(jnp.float32) <= static.max_spawn_distance,
        jnp.ones_like(static.basket_distance_by_cell, dtype=jnp.bool_),
    )
    strict_mask = (
        non_basket_available
        & within_max
        & (static.basket_distance_by_cell.astype(jnp.float32) < offense_dist)
        & (static.basket_distance_by_cell.astype(jnp.float32) >= static.defense_min_spawn_distance)
        & (jnp.abs(dist_to_offense - static.defender_spawn_distance) <= 1.0)
    )
    closer_mask = (
        non_basket_available
        & within_max
        & (static.basket_distance_by_cell.astype(jnp.float32) < offense_dist)
        & (static.basket_distance_by_cell.astype(jnp.float32) >= static.defense_min_spawn_distance)
    )
    ranged_mask = (
        non_basket_available
        & within_max
        & (static.basket_distance_by_cell.astype(jnp.float32) >= static.defense_min_spawn_distance)
    )
    fallback_mask = non_basket_available
    return jnp.where(
        jnp.sum(strict_mask.astype(jnp.int32)) >= 2,
        strict_mask,
        jnp.where(
            jnp.any(closer_mask),
            closer_mask,
            jnp.where(jnp.any(ranged_mask), ranged_mask, fallback_mask),
        ),
    )


def _sample_clamped_probabilities(mean, std, shape, key, jax, jnp):
    std_scalar = jnp.asarray(std, dtype=jnp.float32)
    mean_scalar = jnp.asarray(mean, dtype=jnp.float32)
    sampled = mean_scalar + (std_scalar * jax.random.normal(key, shape=shape, dtype=jnp.float32))
    deterministic = jnp.full(shape, mean_scalar, dtype=jnp.float32)
    return jnp.clip(
        jnp.where(std_scalar > 0.0, sampled, deterministic),
        0.01,
        0.99,
    )


def _sample_reset_positions_single(static: KernelStatic, key, jax, jnp):
    offense_count = int(static.offense_ids.shape[0])
    cell_count = int(static.cell_coords.shape[0])
    offense_key, match_key, defense_key = jax.random.split(key, 3)
    offense_cell_indices = _sample_unique_indices_from_mask(
        static.offense_spawn_candidate_mask,
        offense_count,
        offense_key,
        jax,
        jnp,
    )

    taken_mask = jnp.zeros((cell_count,), dtype=jnp.bool_)
    taken_mask = taken_mask.at[offense_cell_indices].set(True)
    offense_match_order = jax.random.permutation(match_key, jnp.arange(offense_count, dtype=jnp.int32))
    defense_choice_keys = jax.random.split(defense_key, offense_count)
    defense_cell_indices = jnp.full((offense_count,), -1, dtype=jnp.int32)

    for idx in range(offense_count):
        offense_slot = offense_match_order[idx]
        offense_cell_idx = offense_cell_indices[offense_slot]
        dist_to_offense = static.cell_distance_matrix[:, offense_cell_idx].astype(jnp.float32)
        candidate_mask = _defender_spawn_candidate_mask(
            static,
            offense_cell_idx,
            taken_mask,
            jnp,
        )
        masked_distance = jnp.where(candidate_mask, dist_to_offense, jnp.full((cell_count,), jnp.inf, dtype=jnp.float32))
        min_distance = jnp.min(masked_distance)
        closest_mask = candidate_mask & (dist_to_offense == min_distance)
        chosen_cell_idx = _sample_index_from_mask(closest_mask, defense_choice_keys[idx], jax, jnp)
        defense_cell_indices = defense_cell_indices.at[idx].set(chosen_cell_idx)
        taken_mask = taken_mask.at[chosen_cell_idx].set(True)

    positions = jnp.zeros((int(static.role_encoding.shape[0]), 2), dtype=jnp.int32)
    positions = positions.at[static.offense_ids].set(static.cell_coords[offense_cell_indices])
    positions = positions.at[static.defense_ids].set(static.cell_coords[defense_cell_indices])
    return positions


def _sample_start_template_positions_single(static: KernelStatic, template_index, key, jax, jnp):
    """Resolve one sampled template variant on-device.

    Template rows are stored as offense entries followed by defense entries.
    Each reset re-samples team slot assignment and each entry's jittered cell.
    """
    n_players = int(static.role_encoding.shape[0])
    offense_count = int(static.offense_ids.shape[0])
    cell_count = int(static.cell_coords.shape[0])
    entry_count = int(static.start_template_entry_anchors.shape[1])
    offense_perm_key, defense_perm_key, placement_key = jax.random.split(key, 3)

    offense_assignment = jax.random.permutation(offense_perm_key, static.offense_ids)
    defense_assignment = jax.random.permutation(defense_perm_key, static.defense_ids)
    entry_player_ids = jnp.concatenate(
        [
            offense_assignment,
            defense_assignment,
        ],
        axis=0,
    ).astype(jnp.int32)
    anchors = static.start_template_entry_anchors[template_index]
    radii = static.start_template_entry_jitter_radii[template_index].astype(jnp.int32)
    has_ball = static.start_template_entry_has_ball[template_index].astype(jnp.bool_)
    order = jnp.argsort((radii * (n_players + 1)) + entry_player_ids)
    placement_keys = jax.random.split(placement_key, entry_count)

    positions = jnp.zeros((n_players, 2), dtype=jnp.int32)
    taken_mask = jnp.zeros((cell_count,), dtype=jnp.bool_)
    base_cell_available = jnp.where(
        static.allow_dunks.astype(jnp.bool_),
        jnp.ones((cell_count,), dtype=jnp.bool_),
        static.non_basket_cell_mask.astype(jnp.bool_),
    )

    for slot in range(entry_count):
        entry_idx = order[slot]
        anchor = anchors[entry_idx]
        radius = radii[entry_idx]
        player_id = entry_player_ids[entry_idx]
        available_mask = base_cell_available & (~taken_mask)
        distances = _hex_distance(static.cell_coords, anchor, jnp).astype(jnp.int32)
        in_radius_mask = available_mask & (distances <= radius)
        masked_distance = jnp.where(
            available_mask,
            distances,
            jnp.full((cell_count,), 1_000_000, dtype=jnp.int32),
        )
        nearest_distance = jnp.min(masked_distance)
        nearest_mask = available_mask & (distances == nearest_distance)
        candidate_mask = jnp.where(jnp.any(in_radius_mask), in_radius_mask, nearest_mask)
        chosen_cell_idx = _sample_index_from_mask(
            candidate_mask,
            placement_keys[slot],
            jax,
            jnp,
        )
        positions = positions.at[player_id].set(static.cell_coords[chosen_cell_idx])
        taken_mask = taken_mask.at[chosen_cell_idx].set(True)

    ball_entry_idx = jnp.argmax(has_ball.astype(jnp.int32)).astype(jnp.int32)
    ball_holder = jnp.where(
        jnp.any(has_ball),
        entry_player_ids[ball_entry_idx],
        jnp.asarray(-1, dtype=jnp.int32),
    )
    return positions, ball_holder


def _sample_role_intent_single(static: KernelStatic, enabled, null_prob, key, jax, jnp):
    draw_key, index_key = jax.random.split(key)
    enabled_bool = enabled.astype(jnp.bool_)
    active = enabled_bool & (jax.random.uniform(draw_key) >= null_prob)
    sampled_index = jax.random.randint(
        index_key,
        shape=(),
        minval=0,
        maxval=jnp.maximum(static.num_intents, jnp.asarray(1, dtype=jnp.int32)),
        dtype=jnp.int32,
    )
    return {
        "index": jnp.where(active, sampled_index, jnp.asarray(0, dtype=jnp.int32)),
        "active": active.astype(jnp.int8),
        "age": jnp.asarray(0, dtype=jnp.int32),
        "commitment_remaining": jnp.where(
            active,
            static.intent_commitment_steps.astype(jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
        ),
    }


def _reset_single_minimal(static: KernelStatic, key, jax, jnp):
    n_players = int(static.role_encoding.shape[0])
    offense_count = int(static.offense_ids.shape[0])
    (
        shot_clock_key,
        layup_key,
        three_key,
        dunk_key,
        positions_key,
        template_key,
        holder_key,
        offense_intent_key,
        defense_intent_key,
        intent_visible_key,
    ) = jax.random.split(key, 10)

    shot_clock = jax.random.randint(
        shot_clock_key,
        shape=(),
        minval=static.shot_clock_min,
        maxval=static.shot_clock_max + 1,
        dtype=jnp.int32,
    )
    layup_samples = _sample_clamped_probabilities(
        static.base_layup_pct,
        static.layup_std,
        (offense_count,),
        layup_key,
        jax,
        jnp,
    )
    three_samples = _sample_clamped_probabilities(
        static.base_three_pt_pct,
        static.three_pt_std,
        (offense_count,),
        three_key,
        jax,
        jnp,
    )
    dunk_samples = _sample_clamped_probabilities(
        static.base_dunk_pct,
        static.dunk_std,
        (offense_count,),
        dunk_key,
        jax,
        jnp,
    )
    layup_pct = jnp.full((n_players,), static.base_layup_pct, dtype=jnp.float32)
    layup_pct = layup_pct.at[static.offense_ids].set(layup_samples)
    three_pt_pct = jnp.full((n_players,), static.base_three_pt_pct, dtype=jnp.float32)
    three_pt_pct = three_pt_pct.at[static.offense_ids].set(three_samples)
    dunk_pct = jnp.full((n_players,), static.base_dunk_pct, dtype=jnp.float32)
    dunk_pct = dunk_pct.at[static.offense_ids].set(dunk_samples)

    positions = _sample_reset_positions_single(static, positions_key, jax, jnp)
    holder_offset = jax.random.randint(holder_key, shape=(), minval=0, maxval=offense_count, dtype=jnp.int32)
    ball_holder = static.offense_ids[holder_offset]
    template_draw_key, template_index_key, template_resolve_key = jax.random.split(template_key, 3)
    template_weights = jnp.maximum(static.start_template_weights.astype(jnp.float32), 0.0)
    template_weight_sum = jnp.sum(template_weights)
    template_count = static.start_template_weights.shape[0]
    safe_template_weights = jnp.where(
        template_weight_sum > 0.0,
        template_weights / template_weight_sum,
        jnp.ones_like(template_weights, dtype=jnp.float32) / float(template_count),
    )
    template_cdf = jnp.cumsum(safe_template_weights)
    template_draw = jax.random.uniform(template_index_key)
    template_index = jnp.argmax((template_draw <= template_cdf).astype(jnp.int32)).astype(jnp.int32)
    use_template = (
        static.start_template_enabled.astype(jnp.bool_)
        & (jax.random.uniform(template_draw_key) < static.start_template_prob)
    )
    template_positions, template_ball_holder = _sample_start_template_positions_single(
        static,
        template_index,
        template_resolve_key,
        jax,
        jnp,
    )
    template_shot_clock = static.start_template_shot_clocks[template_index]
    positions = jnp.where(use_template, template_positions, positions)
    ball_holder = jnp.where(
        use_template & (template_ball_holder >= 0),
        template_ball_holder,
        ball_holder,
    )
    shot_clock = jnp.where(
        use_template & (template_shot_clock >= 0),
        template_shot_clock,
        shot_clock,
    )
    offense_intent = _sample_role_intent_single(
        static,
        static.enable_intent_learning,
        static.intent_null_prob,
        offense_intent_key,
        jax,
        jnp,
    )
    defense_intent = _sample_role_intent_single(
        static,
        static.enable_defense_intent_learning,
        static.defense_intent_null_prob,
        defense_intent_key,
        jax,
        jnp,
    )
    intent_visible_to_defense = (
        static.enable_intent_learning.astype(jnp.bool_)
        & (jax.random.uniform(intent_visible_key) < static.intent_visible_to_defense_prob)
    )
    return KernelState(
        positions=positions,
        ball_holder=ball_holder,
        shot_clock=shot_clock,
        step_count=jnp.asarray(0, dtype=jnp.int32),
        episode_ended=jnp.asarray(0, dtype=jnp.int8),
        pressure_exposure=jnp.asarray(0.0, dtype=jnp.float32),
        offense_lane_steps=jnp.zeros((n_players,), dtype=jnp.float32),
        defense_lane_steps=jnp.zeros((n_players,), dtype=jnp.float32),
        cached_phi=jnp.asarray(0.0, dtype=jnp.float32),
        offense_score=jnp.asarray(0.0, dtype=jnp.float32),
        defense_score=jnp.asarray(0.0, dtype=jnp.float32),
        assist_active=jnp.asarray(0, dtype=jnp.int8),
        assist_passer=jnp.asarray(-1, dtype=jnp.int32),
        assist_recipient=jnp.asarray(-1, dtype=jnp.int32),
        assist_expires_at=jnp.asarray(-1, dtype=jnp.int32),
        intent_index=offense_intent["index"],
        intent_active=offense_intent["active"],
        intent_age=offense_intent["age"],
        intent_commitment_remaining=offense_intent["commitment_remaining"],
        intent_visible_to_defense=intent_visible_to_defense.astype(jnp.int8),
        defense_intent_index=defense_intent["index"],
        defense_intent_active=defense_intent["active"],
        defense_intent_age=defense_intent["age"],
        defense_intent_commitment_remaining=defense_intent["commitment_remaining"],
        layup_pct=layup_pct,
        three_pt_pct=three_pt_pct,
        dunk_pct=dunk_pct,
    )


def reset_batch_minimal(static: KernelStatic, rng_keys, jax, jnp):
    return jax.vmap(lambda key: _reset_single_minimal(static, key, jax, jnp))(rng_keys)


def sample_state_batch(args, xp) -> tuple[KernelStatic, KernelState]:
    training_team = resolve_training_team(args.training_team)
    wrapped_env = setup_environment(args, training_team)
    base_env = wrapped_env.unwrapped

    try:
        snapshots = []
        for batch_idx in range(int(args.kernel_batch_size)):
            base_env.reset(seed=int(args.sample_reset_seed) + batch_idx)
            snapshots.append(snapshot_state_from_env(base_env))

        static = build_kernel_static_from_env(base_env, xp=xp)
        state = stack_state_snapshots(snapshots, xp=xp)
    finally:
        try:
            wrapped_env.close()
        except Exception:
            pass

    return static, state
