from __future__ import annotations

import math
from typing import Any, NamedTuple, Sequence

import numpy as np

from basketworld.envs.basketworld_env_v2 import ActionType, Team
from basketworld_jax.train.cli import resolve_training_team
from train.env_factory import setup_environment


MOVE_ACTION_START = ActionType.MOVE_E.value
MOVE_ACTION_END = ActionType.MOVE_SE.value + 1
PASS_ACTION_START = ActionType.PASS_E.value
PASS_ACTION_END = ActionType.PASS_SE.value + 1
ACTION_COUNT = len(ActionType)
SQRT3 = float(np.sqrt(3.0))


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
    layup_pct: Any
    three_pt_pct: Any
    dunk_pct: Any


class StepBatchOutput(NamedTuple):
    state: KernelState
    rewards: Any
    done: Any
    pass_attempt: Any
    completed_pass: Any
    assist: Any
    turnover: Any
    terminal_episode_steps: Any


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
    position_weight = static.steal_position_weight_min + ((1.0 - static.steal_position_weight_min) * position_t)
    steal_contrib = (
        static.base_steal_rate
        * jnp.exp(-static.steal_perp_decay * perp_distance)
        * (1.0 + (static.steal_distance_factor * pass_distance[:, :, None]))
        * position_weight
    )
    steal_contrib = jnp.clip(steal_contrib, 0.0, 1.0)
    steal_contrib = jnp.where(
        valid_receivers[:, :, None] & forward_defender & (~same_as_passer) & (~same_as_receiver),
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
    shot_clock = state.shot_clock.astype(jnp.float32)[:, None]
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


def build_flat_observation_batch(static: KernelStatic, state: KernelState, jnp):
    batch_size = state.positions.shape[0]
    role_flag = jnp.full((batch_size, 1), static.training_role_flag, dtype=jnp.float32)
    return jnp.concatenate(
        [
            build_observation_vector_batch(static, state, jnp),
            role_flag,
            build_offense_skill_deltas_batch(static, state, jnp),
        ],
        axis=1,
    ).astype(jnp.float32)


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

    return final_positions, ball_holder, turnover_any


def _step_single_minimal(static: KernelStatic, state: KernelState, actions, key, jax, jnp):
    zero_rewards = jnp.zeros((state.positions.shape[0],), dtype=jnp.float32)
    zero_flag = jnp.asarray(0, dtype=jnp.int8)
    zero_steps = jnp.asarray(0, dtype=jnp.int32)

    def _already_done(_):
        return StepBatchOutput(
            state=state,
            rewards=zero_rewards,
            done=jnp.asarray(True),
            pass_attempt=zero_flag,
            completed_pass=zero_flag,
            assist=zero_flag,
            turnover=zero_flag,
            terminal_episode_steps=zero_steps,
        )

    def _run_active(_):
        pressure_key, action_key, move_key = jax.random.split(key, 3)
        next_state = _replace_state(
            state,
            step_count=state.step_count + 1,
            episode_ended=jnp.asarray(0, dtype=state.episode_ended.dtype),
        )

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
                completed_pass=zero_flag,
                assist=zero_flag,
                turnover=jnp.asarray(1, dtype=jnp.int8),
                terminal_episode_steps=pressure_state.step_count.astype(jnp.int32),
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
                    jnp.asarray(False),
                    jnp.asarray(False),
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
                    theft,
                    ~theft,
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
                turnover_from_action,
                pass_success,
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
                        jnp.asarray(False),
                        jnp.asarray(False),
                    ),
                    operand=None,
                ),
                operand=None,
            )

            movement_skipped = shot_active | turnover_from_action
            positions_after, ball_holder_after, movement_turnover = jax.lax.cond(
                movement_skipped,
                lambda _: (positions_after, ball_holder_after, jnp.asarray(False)),
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

            shot_clock_turnover = final_state.shot_clock <= 0
            turnover_event = (
                turnover_from_action | movement_turnover | shot_clock_turnover
            ).astype(jnp.int8)
            done = done | shot_clock_turnover
            final_state = _replace_state(
                final_state,
                episode_ended=done.astype(final_state.episode_ended.dtype),
            )
            return StepBatchOutput(
                state=final_state,
                rewards=rewards,
                done=done,
                pass_attempt=pass_attempt,
                completed_pass=pass_success.astype(jnp.int8),
                assist=assist_event,
                turnover=turnover_event,
                terminal_episode_steps=jnp.where(
                    done,
                    final_state.step_count.astype(jnp.int32),
                    jnp.asarray(0, dtype=jnp.int32),
                ),
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


def _sample_index_from_mask(mask, key, jax, jnp):
    mask_bool = mask.astype(jnp.bool_)
    logits = jnp.where(mask_bool, jnp.zeros(mask_bool.shape, dtype=jnp.float32), -jnp.inf)
    return jax.random.categorical(key, logits, axis=-1).astype(jnp.int32)


def _sample_unique_indices_from_mask(mask, count: int, key, jax, jnp):
    gumbels = jax.random.gumbel(key, shape=mask.shape, dtype=jnp.float32)
    masked_scores = jnp.where(mask.astype(jnp.bool_), gumbels, -jnp.inf)
    _, indices = jax.lax.top_k(masked_scores, int(count))
    return indices.astype(jnp.int32)


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
        candidate_mask = jnp.where(
            jnp.any(strict_mask),
            strict_mask,
            jnp.where(
                jnp.any(closer_mask),
                closer_mask,
                jnp.where(jnp.any(ranged_mask), ranged_mask, fallback_mask),
            ),
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


def _reset_single_minimal(static: KernelStatic, key, jax, jnp):
    n_players = int(static.role_encoding.shape[0])
    offense_count = int(static.offense_ids.shape[0])
    shot_clock_key, layup_key, three_key, dunk_key, positions_key, holder_key = jax.random.split(key, 6)

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
