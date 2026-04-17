from __future__ import annotations

import math
import sys
from contextlib import nullcontext
from pathlib import Path
from time import perf_counter_ns
from typing import Any, NamedTuple, Sequence

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Support both `python -m benchmarks.jax_kernel` and
# `python benchmarks/jax_kernel.py` from the repo root.
if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from basketworld.algorithms import IntegratedIntentSelectorPPO
from basketworld.envs.basketworld_env_v2 import ActionType, Team
from basketworld.policies import SetAttentionDualCriticPolicy
from basketworld.utils.action_resolution import (
    IllegalActionStrategy,
    get_policy_action_probabilities,
    resolve_illegal_actions,
)
from basketworld.utils.mask_agnostic_extractor import MaskAgnosticCombinedExtractor
from basketworld.utils.policies import PassBiasDualCriticPolicy, PassBiasMultiInputPolicy
from benchmarks.sbx_phase_a import PhaseAPolicySpec, init_phase_a_policy_params
from benchmarks.common import (
    Timer,
    benchmark_args_snapshot,
    build_progress,
    build_benchmark_parser,
    ensure_jax_available,
    resolve_training_team,
    write_json,
)
from train.env_factory import make_policy_init_env, setup_environment


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


def parse_args(argv=None):
    parser = build_benchmark_parser(
        "Prototype batched JAX kernels for BasketWorld hotspot-heavy env computations."
    )
    parser.set_defaults(mode="throughput", runner="sequential")
    parser.add_argument(
        "--kernel-batch-size",
        type=int,
        default=256,
        help="Number of sampled env states to pack into one batched kernel call.",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=3,
        help="Number of warm JIT iterations to run before timing.",
    )
    parser.add_argument(
        "--benchmark-iters",
        type=int,
        default=50,
        help="Number of timed iterations per compiled kernel.",
    )
    parser.add_argument(
        "--sample-reset-seed",
        type=int,
        default=0,
        help="Base seed used when sampling env snapshots into the JAX batch.",
    )
    parser.add_argument(
        "--phase-a-policy-hidden-dims",
        type=int,
        nargs="+",
        default=[128, 128],
        help="Hidden layer widths for the reduced flat JAX policy used in compiled rollout benchmarks.",
    )
    parser.add_argument(
        "--phase-a-policy-seed",
        type=int,
        default=0,
        help="Random seed used to initialize the reduced flat JAX policy for compiled rollout benchmarks.",
    )
    return parser.parse_args(argv)


def _require_supported_scope(args) -> None:
    unsupported = []
    if str(getattr(args, "pass_mode", "pointer_targeted")).lower() != "pointer_targeted":
        unsupported.append("pass_mode must be pointer_targeted")
    if getattr(args, "players", 0) <= 0:
        unsupported.append("players must be > 0")
    if not bool(getattr(args, "include_hoop_vector", True)):
        unsupported.append("include_hoop_vector must be true")
    if str(getattr(args, "phi_aggregation_mode", "team_best")).lower() != "team_best":
        unsupported.append("phi_aggregation_mode must be team_best")
    if bool(getattr(args, "phi_use_ball_handler_only", False)):
        unsupported.append("phi_use_ball_handler_only must be false")
    if unsupported:
        raise SystemExit(
            "benchmarks/jax_kernel.py only supports the current phase-2 hotspot kernel scope: "
            + "; ".join(unsupported)
        )


def _minimal_transition_scope_blockers(args) -> list[str]:
    blockers = []
    if bool(getattr(args, "enable_phi_shaping", False)):
        blockers.append("phi_shaping")
    if bool(getattr(args, "illegal_defense_enabled", False)):
        blockers.append("illegal_defense")
    if bool(getattr(args, "offensive_three_seconds_enabled", False)):
        blockers.append("offensive_three_seconds")
    return blockers


def _player_skill_arrays(env) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    layup = np.full(env.n_players, float(env.layup_pct), dtype=np.float32)
    three = np.full(env.n_players, float(env.three_pt_pct), dtype=np.float32)
    dunk = np.full(env.n_players, float(env.dunk_pct), dtype=np.float32)
    layup[np.asarray(env.offense_ids, dtype=np.int32)] = np.asarray(
        env.offense_layup_pct_by_player, dtype=np.float32
    )
    three[np.asarray(env.offense_ids, dtype=np.int32)] = np.asarray(
        env.offense_three_pt_pct_by_player, dtype=np.float32
    )
    dunk[np.asarray(env.offense_ids, dtype=np.int32)] = np.asarray(
        env.offense_dunk_pct_by_player, dtype=np.float32
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
        "assist_active": 1 if getattr(env, "_assist_candidate", None) is not None else 0,
        "assist_passer": int(getattr(env, "_assist_candidate", {}).get("passer_id", -1))
        if getattr(env, "_assist_candidate", None) is not None
        else -1,
        "assist_recipient": int(getattr(env, "_assist_candidate", {}).get("recipient_id", -1))
        if getattr(env, "_assist_candidate", None) is not None
        else -1,
        "assist_expires_at": int(getattr(env, "_assist_candidate", {}).get("expires_at_step", -1))
        if getattr(env, "_assist_candidate", None) is not None
        else -1,
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
            np.asarray(
                [float(item["pressure_exposure"]) for item in snapshots], dtype=np.float32
            ),
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
        [
            [env._hex_distance(src, dst) for dst in cells]
            for src in cells
        ],
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
    opponent_mask = (
        player_is_offense[:, None] != player_is_offense[None, :]
    ).astype(np.int8)
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
        shot_pressure_cos_threshold=xp.asarray(
            float(math.cos(env.shot_pressure_arc_rad / 2.0)), dtype=xp.float32
        ),
        defender_pressure_distance=xp.asarray(
            float(env.defender_pressure_distance), dtype=xp.float32
        ),
        defender_pressure_turnover_chance=xp.asarray(
            float(env.defender_pressure_turnover_chance), dtype=xp.float32
        ),
        defender_pressure_decay_lambda=xp.asarray(
            float(env.defender_pressure_decay_lambda), dtype=xp.float32
        ),
        base_steal_rate=xp.asarray(float(env.base_steal_rate), dtype=xp.float32),
        steal_perp_decay=xp.asarray(float(env.steal_perp_decay), dtype=xp.float32),
        steal_distance_factor=xp.asarray(float(env.steal_distance_factor), dtype=xp.float32),
        steal_position_weight_min=xp.asarray(
            float(env.steal_position_weight_min), dtype=xp.float32
        ),
        three_point_distance=xp.asarray(float(env.three_point_distance), dtype=xp.float32),
        three_pt_extra_hex_decay=xp.asarray(
            float(env.three_pt_extra_hex_decay), dtype=xp.float32
        ),
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
            1 if env.offensive_three_seconds_enabled else 0, dtype=xp.int8
        ),
        pass_reward=xp.asarray(float(env.pass_reward), dtype=xp.float32),
        violation_reward=xp.asarray(float(env.violation_reward), dtype=xp.float32),
        reward_shaping_gamma=xp.asarray(float(env.reward_shaping_gamma), dtype=xp.float32),
        enable_phi_shaping=xp.asarray(1 if env.enable_phi_shaping else 0, dtype=xp.int8),
        phi_beta=xp.asarray(float(env.phi_beta), dtype=xp.float32),
        phi_blend_weight=xp.asarray(float(env.phi_blend_weight), dtype=xp.float32),
        phi_use_ball_handler_only=xp.asarray(
            1 if env.phi_use_ball_handler_only else 0, dtype=xp.int8
        ),
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
        task_reward_scale=xp.asarray(
            float(getattr(env, "task_reward_scale", 1.0)), dtype=xp.float32
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


def _single_state_from_batched(state: KernelState) -> KernelState:
    return KernelState(*(field[0] for field in state))


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


def _pick_deterministic_legal_actions(masks, jnp):
    action_ids = jnp.arange(ACTION_COUNT, dtype=jnp.int32)
    invalid_scores = -jnp.ones_like(masks, dtype=jnp.int32)
    scores = jnp.where(masks > 0, action_ids[None, None, :], invalid_scores)
    return jnp.argmax(scores, axis=-1).astype(jnp.int32)


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
    holder_mask = (state.ball_holder[:, None] == player_ids[None, :]) & (
        state.ball_holder[:, None] >= 0
    )
    masks = masks.at[:, :, ActionType.SHOOT.value].set(holder_mask.astype(jnp.int8))

    pass_masks = (
        holder_mask[:, :, None].astype(jnp.int8)
        * static.pointer_pass_slot_mask[None, :, :]
    )
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
    extra_hexes = jnp.maximum(
        0.0,
        distances_f - jnp.floor(d1),
    )
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
    cosang = (vx * dir_x[:, :, None] + vy * dir_y[:, :, None]) / (
        safe_vnorm * dir_norm[:, :, None]
    )
    in_arc = cosang >= static.shot_pressure_cos_threshold
    defender_distance = _hex_distance(shooter_pos, defender_pos, jnp).astype(jnp.float32)
    valid_defender = (
        static.opponent_mask[None, :, :].astype(jnp.bool_)
        & (vnorm > 0.0)
        & in_arc
        & (defender_distance <= distances_f[:, :, None])
    )

    angle_factor = (cosang - static.shot_pressure_cos_threshold) / (
        1.0 - static.shot_pressure_cos_threshold
    )
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
    defender_x, defender_y = _axial_to_cartesian(
        defender_delta[..., 0], defender_delta[..., 1], jnp
    )
    basket_mag = jnp.sqrt((basket_x**2) + (basket_y**2))
    defender_mag = jnp.sqrt((defender_x**2) + (defender_y**2))
    safe_den = jnp.where((basket_mag[:, None] * defender_mag) == 0.0, 1.0, basket_mag[:, None] * defender_mag)
    cos_angle = ((basket_x[:, None] * defender_x) + (basket_y[:, None] * defender_y)) / safe_den
    cos_angle = jnp.where((basket_mag[:, None] == 0.0) | (defender_mag == 0.0), 0.0, cos_angle)

    valid = (
        has_offense_holder[:, None]
        & (distances <= static.defender_pressure_distance)
        & (cos_angle >= 0.0)
    )
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
    pass_distance = _hex_distance(passer_pos[:, None, :], offense_positions, jnp).astype(
        jnp.float32
    )

    defender_delta = defense_positions[:, None, :, :] - passer_pos[:, None, None, :]
    defender_x, defender_y = _axial_to_cartesian(
        defender_delta[..., 0], defender_delta[..., 1], jnp
    )
    dot = (defender_x * line_x[:, :, None]) + (defender_y * line_y[:, :, None])
    forward_defender = dot >= 0.0

    same_as_passer = jnp.all(
        defense_positions[:, None, :, :] == passer_pos[:, None, None, :], axis=-1
    )
    same_as_receiver = jnp.all(
        defense_positions[:, None, :, :] == offense_positions[:, :, None, :], axis=-1
    )
    perp_distance, position_t = _point_to_segment_distance_and_projection(
        defender_x,
        defender_y,
        line_x[:, :, None],
        line_y[:, :, None],
        jnp,
    )
    position_weight = static.steal_position_weight_min + (
        (1.0 - static.steal_position_weight_min) * position_t
    )
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
    hoop_vec = jnp.broadcast_to(
        static.basket_position.astype(jnp.float32)[None, :] / norm_den,
        (batch_size, 2),
    )

    offense_positions = jnp.take(state.positions, static.offense_ids, axis=1)
    defense_positions = jnp.take(state.positions, static.defense_ids, axis=1)
    off_def_distances = _hex_distance(
        offense_positions[:, :, None, :], defense_positions[:, None, :, :], jnp
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


def _expand_offense_values_to_all_players(static: KernelStatic, offense_values, jnp):
    batch_size = offense_values.shape[0]
    n_players = static.role_encoding.shape[0]
    full = jnp.zeros((batch_size, n_players), dtype=offense_values.dtype)
    return full.at[:, static.offense_ids].set(offense_values)


def build_offense_skill_deltas_batch(static: KernelStatic, state: KernelState, jnp):
    layup_delta = state.layup_pct[:, static.offense_ids] - static.base_layup_pct
    three_delta = state.three_pt_pct[:, static.offense_ids] - static.base_three_pt_pct
    dunk_delta = state.dunk_pct[:, static.offense_ids] - static.base_dunk_pct
    stacked = jnp.stack([layup_delta, three_delta, dunk_delta], axis=-1)
    return stacked.reshape(stacked.shape[0], -1).astype(jnp.float32)


def build_set_globals_batch(static: KernelStatic, state: KernelState, jnp):
    batch_size = state.positions.shape[0]
    hoop_vec = jnp.broadcast_to(
        static.basket_position.astype(jnp.float32)[None, :] / static.court_norm_den,
        (batch_size, 2),
    )
    return jnp.concatenate(
        [
            state.shot_clock.astype(jnp.float32)[:, None],
            state.pressure_exposure.astype(jnp.float32)[:, None],
            hoop_vec,
        ],
        axis=1,
    )


def build_set_player_tokens_batch(static: KernelStatic, state: KernelState, jnp):
    batch_size, n_players, _ = state.positions.shape
    norm_den = static.court_norm_den
    positions_f = state.positions.astype(jnp.float32)
    player_ids = jnp.arange(n_players, dtype=jnp.int32)
    role = jnp.broadcast_to(static.role_encoding[None, :], (batch_size, n_players))
    has_ball = (
        (state.ball_holder[:, None] == player_ids[None, :]) & (state.ball_holder[:, None] >= 0)
    ).astype(jnp.float32)
    raw_skills = jnp.stack([state.layup_pct, state.three_pt_pct, state.dunk_pct], axis=-1)
    skills = jnp.where(role[..., None] > 0.0, raw_skills, jnp.zeros_like(raw_skills))

    max_lane_steps = jnp.maximum(static.three_second_max_steps, 1.0)
    lane_steps = jnp.where(
        role > 0.0,
        state.offense_lane_steps,
        state.defense_lane_steps,
    ).astype(jnp.float32)
    lane_steps_norm = jnp.clip(lane_steps / max_lane_steps, 0.0, 1.0)

    shot_profile = build_shot_profile_batch(static, state, jnp)
    ep_full = jnp.where(role > 0.0, shot_profile["expected_points"], 0.0).astype(jnp.float32)

    turnover_offense = build_turnover_probabilities_batch(static, state, jnp)
    turnover_full = _expand_offense_values_to_all_players(static, turnover_offense, jnp)
    steal_offense = build_pass_steal_probabilities_batch(static, state, jnp)
    steal_full = _expand_offense_values_to_all_players(static, steal_offense, jnp)

    holder_valid = state.ball_holder[:, None] >= 0
    holder_pos = _safe_ball_holder_positions(state, jnp)
    dist_to_ball = _hex_distance(
        state.positions,
        holder_pos[:, None, :],
        jnp,
    ).astype(jnp.float32) / norm_den
    dist_to_ball = jnp.where(holder_valid, dist_to_ball, jnp.zeros_like(dist_to_ball))

    offense_ep = jnp.take(shot_profile["expected_points"], static.offense_ids, axis=1)
    best_ep_idx = jnp.argmax(offense_ep, axis=1)
    best_ep_pid = static.offense_ids[best_ep_idx]
    best_ep_pos = jnp.take_along_axis(
        state.positions,
        best_ep_pid[:, None, None],
        axis=1,
    )[:, 0, :]
    dist_to_best_ep = _hex_distance(
        state.positions,
        best_ep_pos[:, None, :],
        jnp,
    ).astype(jnp.float32) / norm_den

    pairwise = _hex_distance(
        state.positions[:, :, None, :],
        state.positions[:, None, :, :],
        jnp,
    ).astype(jnp.float32) / norm_den
    inf = jnp.asarray(jnp.inf, dtype=jnp.float32)
    same_player = jnp.eye(n_players, dtype=jnp.bool_)[None, :, :]
    opponent_mask = static.opponent_mask[None, :, :].astype(jnp.bool_)
    teammate_mask = (~opponent_mask) & (~same_player)
    nearest_opp = jnp.min(jnp.where(opponent_mask, pairwise, inf), axis=-1)
    nearest_team = jnp.min(jnp.where(teammate_mask, pairwise, inf), axis=-1)
    nearest_opp = jnp.where(jnp.isfinite(nearest_opp), nearest_opp, 0.0)
    nearest_team = jnp.where(jnp.isfinite(nearest_team), nearest_team, 0.0)

    return jnp.stack(
        [
            positions_f[..., 0] / norm_den,
            positions_f[..., 1] / norm_den,
            role,
            has_ball,
            skills[..., 0],
            skills[..., 1],
            skills[..., 2],
            lane_steps_norm,
            ep_full,
            turnover_full,
            steal_full,
            dist_to_ball,
            dist_to_best_ep,
            nearest_opp,
            nearest_team,
        ],
        axis=-1,
    ).astype(jnp.float32)


def build_set_observation_batch(static: KernelStatic, state: KernelState, jnp):
    batch_size = state.positions.shape[0]
    return {
        "obs": build_observation_vector_batch(static, state, jnp),
        "action_mask": build_action_masks_batch(static, state, jnp),
        "role_flag": jnp.full(
            (batch_size, 1),
            static.training_role_flag,
            dtype=jnp.float32,
        ),
        "skills": build_offense_skill_deltas_batch(static, state, jnp),
        "players": build_set_player_tokens_batch(static, state, jnp),
        "globals": build_set_globals_batch(static, state, jnp),
    }


def build_phase_a_flat_observation_batch(static: KernelStatic, state: KernelState, jnp):
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


def _expected_points_single(static: KernelStatic, state: KernelState, jnp):
    batched_state = _single_state_to_batched(state, jnp)
    return build_shot_profile_batch(static, batched_state, jnp)["expected_points"][0]


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
    basket_collision = jnp.all(proposed == static.basket_position, axis=-1) & (
        ~static.allow_dunks.astype(jnp.bool_)
    )
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

    occupied_start = jnp.all(
        intended_dest[:, None, :] == current_positions[None, :, :],
        axis=-1,
    )
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


def _phi_shot_quality_single(static: KernelStatic, state: KernelState, jnp):
    if static.phi_use_ball_handler_only.astype(jnp.bool_):
        eps = _expected_points_single(static, state, jnp)
        safe_holder = jnp.clip(state.ball_holder, 0, eps.shape[0] - 1)
        return eps[safe_holder]

    eps = _expected_points_single(static, state, jnp)
    team_mask = _team_mask_for_holder(static, state.ball_holder, jnp)
    team_eps = jnp.where(team_mask, eps, -jnp.inf)
    team_best = jnp.max(team_eps)
    safe_holder = jnp.clip(state.ball_holder, 0, eps.shape[0] - 1)
    ball_ep = eps[safe_holder]
    return ((1.0 - static.phi_blend_weight) * team_best) + (static.phi_blend_weight * ball_ep)


def _step_single_minimal(static: KernelStatic, state: KernelState, actions, key, jax, jnp):
    zero_rewards = jnp.zeros((state.positions.shape[0],), dtype=jnp.float32)

    def _already_done(_):
        return StepBatchOutput(state=state, rewards=zero_rewards, done=jnp.asarray(True))

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
            return StepBatchOutput(state=pressure_state, rewards=zero_rewards, done=jnp.asarray(True))

        def _normal_step(_):
            shot_clock_state = _replace_state(next_state, shot_clock=next_state.shot_clock - 1)
            ball_holder = shot_clock_state.ball_holder
            safe_holder = jnp.clip(ball_holder, 0, actions.shape[0] - 1)
            holder_action = actions[safe_holder]
            holder_has_ball = ball_holder >= 0
            is_shot = holder_has_ball & (holder_action == ActionType.SHOOT.value)
            is_pass = holder_has_ball & (holder_action >= PASS_ACTION_START) & (holder_action < PASS_ACTION_END)

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
                steal_prob = jnp.where(
                    receiver >= 0,
                    pass_probs[receiver_safe],
                    0.0,
                )
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
                offense_score=final_state.offense_score
                + jnp.where(shooter_is_offense, scored_points, 0.0),
                defense_score=final_state.defense_score
                + jnp.where(shooter_is_offense, 0.0, scored_points),
            )

            per_team_pass = static.pass_reward / static.offense_ids.shape[0]
            offense_mask = static.role_encoding > 0.0
            rewards = rewards + (
                jnp.where(offense_mask, per_team_pass, -per_team_pass)
                * pass_success.astype(jnp.float32)
            )

            done = turnover_from_action | movement_turnover | shot_active
            per_team_shot = shot_expected_points / static.offense_ids.shape[0]
            rewards = rewards + (
                jnp.where(offense_mask, per_team_shot, -per_team_shot)
                * shot_active.astype(jnp.float32)
            )
            assist_valid = (
                final_state.assist_active.astype(jnp.bool_)
                & (final_state.assist_recipient == shot_shooter)
                & (final_state.step_count <= final_state.assist_expires_at)
            )
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
                assist_active=jnp.where(
                    shot_active,
                    jnp.asarray(0, dtype=jnp.int8),
                    final_state.assist_active,
                ),
                assist_passer=jnp.where(
                    shot_active,
                    jnp.asarray(-1, dtype=jnp.int32),
                    final_state.assist_passer,
                ),
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

            done = done | (final_state.shot_clock <= 0)

            final_state = _replace_state(
                final_state,
                episode_ended=done.astype(final_state.episode_ended.dtype),
            )
            return StepBatchOutput(state=final_state, rewards=rewards, done=done)

        return jax.lax.cond(pressure_turnover, _pressure_done, _normal_step, operand=None)

    return jax.lax.cond(state.episode_ended.astype(jnp.bool_), _already_done, _run_active, operand=None)


def _step_batch_minimal_impl(static: KernelStatic, state: KernelState, actions, rng_keys, jax, jnp):
    per_state = lambda state_single, action_single, key_single: _step_single_minimal(
        static,
        state_single,
        action_single,
        key_single,
        jax,
        jnp,
    )
    return jax.vmap(per_state)(state, actions, rng_keys)


def _resolve_team_player_ids(static, jax, jnp):
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


def _phase_a_policy_forward_logits(params, flat_obs, jnp):
    x = flat_obs
    for layer_idx, (weights, bias) in enumerate(params):
        x = jnp.matmul(x, weights) + bias
        if layer_idx < len(params) - 1:
            x = jnp.tanh(x)
    return x


def _masked_categorical_actions_jax(logits, action_mask, sample_key, jax, jnp):
    legal = action_mask > 0
    has_legal = jnp.any(legal, axis=-1, keepdims=True)
    noop_mask = jnp.zeros_like(legal)
    noop_mask = noop_mask.at[..., 0].set(True)
    effective_legal = jnp.where(has_legal, legal, noop_mask)
    masked_logits = jnp.where(
        effective_legal,
        logits,
        jnp.full_like(logits, -1.0e9),
    )
    sampled = jax.random.categorical(sample_key, masked_logits, axis=-1).astype(jnp.int32)
    deterministic = jnp.argmax(masked_logits, axis=-1).astype(jnp.int32)
    return sampled, deterministic, masked_logits


def _sample_uniform_legal_actions_jax(action_mask, sample_key, jax, jnp):
    zero_logits = jnp.zeros(action_mask.shape, dtype=jnp.float32)
    sampled, _, _ = _masked_categorical_actions_jax(
        zero_logits,
        action_mask,
        sample_key,
        jax,
        jnp,
    )
    return sampled


def _assemble_full_actions_jax(
    training_actions,
    opponent_actions,
    training_ids,
    opponent_ids,
    n_players: int,
    jnp,
):
    batch_size = training_actions.shape[0]
    full_actions = jnp.zeros((batch_size, int(n_players)), dtype=jnp.int32)
    full_actions = full_actions.at[:, training_ids].set(training_actions)
    full_actions = full_actions.at[:, opponent_ids].set(opponent_actions)
    return full_actions


def _replace_done_states(next_state: KernelState, reset_state: KernelState, done, jnp):
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
    offense_match_order = jax.random.permutation(
        match_key,
        jnp.arange(offense_count, dtype=jnp.int32),
    )
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
        masked_distance = jnp.where(
            candidate_mask,
            dist_to_offense,
            jnp.full((cell_count,), jnp.inf, dtype=jnp.float32),
        )
        min_distance = jnp.min(masked_distance)
        closest_mask = candidate_mask & (dist_to_offense == min_distance)
        chosen_cell_idx = _sample_index_from_mask(
            closest_mask,
            defense_choice_keys[idx],
            jax,
            jnp,
        )
        defense_cell_indices = defense_cell_indices.at[idx].set(chosen_cell_idx)
        taken_mask = taken_mask.at[chosen_cell_idx].set(True)

    positions = jnp.zeros(
        (int(static.role_encoding.shape[0]), 2),
        dtype=jnp.int32,
    )
    positions = positions.at[static.offense_ids].set(static.cell_coords[offense_cell_indices])
    positions = positions.at[static.defense_ids].set(static.cell_coords[defense_cell_indices])
    return positions


def _reset_single_minimal(static: KernelStatic, key, jax, jnp):
    n_players = int(static.role_encoding.shape[0])
    offense_count = int(static.offense_ids.shape[0])
    (
        shot_clock_key,
        layup_key,
        three_key,
        dunk_key,
        positions_key,
        holder_key,
    ) = jax.random.split(key, 6)

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
    holder_offset = jax.random.randint(
        holder_key,
        shape=(),
        minval=0,
        maxval=offense_count,
        dtype=jnp.int32,
    )
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


def step_batch(static: KernelStatic, state: KernelState, actions, rng_keys, jax, jnp):
    if (
        bool(np.asarray(static.enable_phi_shaping))
        or bool(np.asarray(static.illegal_defense_enabled))
        or bool(np.asarray(static.offensive_three_seconds_enabled))
    ):
        raise NotImplementedError(
            "step_batch currently supports the minimal transition scope only: "
            "phi shaping, illegal defense, and offensive three-seconds must be disabled."
        )
    return _step_batch_minimal_impl(static, state, actions, rng_keys, jax, jnp)


def compile_hotspot_kernels(jax, jnp):
    return {
        "action_masks": jax.jit(lambda static, state: build_action_masks_batch(static, state, jnp)),
        "shot_profiles": jax.jit(lambda static, state: build_shot_profile_batch(static, state, jnp)),
        "offense_expected_points": jax.jit(
            lambda static, state: build_offense_expected_points_batch(static, state, jnp)
        ),
        "turnover_probabilities": jax.jit(
            lambda static, state: build_turnover_probabilities_batch(static, state, jnp)
        ),
        "pass_steal_probabilities": jax.jit(
            lambda static, state: build_pass_steal_probabilities_batch(static, state, jnp)
        ),
        "raw_observation_vector": jax.jit(
            lambda static, state: build_observation_vector_batch(static, state, jnp)
        ),
        "set_observation_payload": jax.jit(
            lambda static, state: build_set_observation_batch(static, state, jnp)
        ),
    }


def compile_transition_kernels(jax, jnp):
    def _reset_batch_kernel(static, rng_keys):
        return reset_batch_minimal(static, rng_keys, jax, jnp)

    def _step_batch_kernel(static, state, actions, rng_keys):
        return _step_batch_minimal_impl(static, state, actions, rng_keys, jax, jnp)

    def _rollout_like_kernel(static, state, rng_keys):
        masks = build_action_masks_batch(static, state, jnp)
        actions = _pick_deterministic_legal_actions(masks, jnp)
        out = _step_batch_minimal_impl(static, state, actions, rng_keys, jax, jnp)
        next_obs = build_observation_vector_batch(static, out.state, jnp)
        next_masks = build_action_masks_batch(static, out.state, jnp)
        return out.state, out.rewards, out.done, next_obs, next_masks

    def _sb3_payload_kernel(static, state, rng_keys):
        masks = build_action_masks_batch(static, state, jnp)
        actions = _pick_deterministic_legal_actions(masks, jnp)
        out = _step_batch_minimal_impl(static, state, actions, rng_keys, jax, jnp)
        next_obs = build_set_observation_batch(static, out.state, jnp)
        aggregated_reward = build_aggregated_reward_batch(static, out.rewards, jnp)
        return next_obs, aggregated_reward, out.done

    return {
        "reset_batch_minimal": jax.jit(_reset_batch_kernel),
        "step_batch_minimal": jax.jit(_step_batch_kernel),
        "rollout_like_minimal": jax.jit(_rollout_like_kernel),
        "sb3_payload_minimal_device": jax.jit(_sb3_payload_kernel),
    }


def compile_compiled_rollout_kernels(jax, jnp):
    def _compiled_rollout_random_opponent_kernel(
        static,
        initial_state,
        rollout_key,
        horizon: int,
    ):
        training_ids, opponent_ids = _resolve_team_player_ids(static, jax, jnp)
        n_players = int(static.role_encoding.shape[0])

        def _scan_step(carry, _):
            state, key = carry
            key, opponent_key, env_key, reset_key = jax.random.split(key, 4)
            masks = build_action_masks_batch(static, state, jnp)
            opponent_mask = masks[:, opponent_ids, :]
            opponent_actions = _sample_uniform_legal_actions_jax(
                opponent_mask,
                opponent_key,
                jax,
                jnp,
            )
            training_actions = _sample_uniform_legal_actions_jax(
                masks[:, training_ids, :],
                key,
                jax,
                jnp,
            )
            full_actions = _assemble_full_actions_jax(
                training_actions,
                opponent_actions,
                training_ids,
                opponent_ids,
                n_players,
                jnp,
            )
            env_keys = jax.random.split(env_key, initial_state.positions.shape[0])
            out = _step_batch_minimal_impl(static, state, full_actions, env_keys, jax, jnp)
            reset_keys = jax.random.split(reset_key, initial_state.positions.shape[0])
            reset_state = reset_batch_minimal(static, reset_keys, jax, jnp)
            next_state = _replace_done_states(out.state, reset_state, out.done, jnp)
            aggregated_reward = build_aggregated_reward_batch(static, out.rewards, jnp)
            return (next_state, key), (
                aggregated_reward,
                out.done.astype(jnp.int8),
            )

        return jax.lax.scan(
            _scan_step,
            (initial_state, rollout_key),
            xs=None,
            length=int(horizon),
        )

    def _compiled_rollout_phase_a_policy_kernel(
        static,
        initial_state,
        policy_params,
        rollout_key,
        horizon: int,
    ):
        training_ids, opponent_ids = _resolve_team_player_ids(static, jax, jnp)
        n_players = int(static.role_encoding.shape[0])
        training_player_count = int(training_ids.shape[0])
        action_dim = ACTION_COUNT

        def _scan_step(carry, _):
            state, key = carry
            key, policy_key, opponent_key, env_key, reset_key = jax.random.split(key, 5)
            masks = build_action_masks_batch(static, state, jnp)
            training_mask = masks[:, training_ids, :]
            opponent_mask = masks[:, opponent_ids, :]

            flat_obs = build_phase_a_flat_observation_batch(static, state, jnp)
            flat_logits = _phase_a_policy_forward_logits(policy_params, flat_obs, jnp)
            logits = flat_logits.reshape(
                flat_logits.shape[0],
                training_player_count,
                action_dim,
            )
            training_actions, _, _ = _masked_categorical_actions_jax(
                logits,
                training_mask,
                policy_key,
                jax,
                jnp,
            )
            opponent_actions = _sample_uniform_legal_actions_jax(
                opponent_mask,
                opponent_key,
                jax,
                jnp,
            )
            full_actions = _assemble_full_actions_jax(
                training_actions,
                opponent_actions,
                training_ids,
                opponent_ids,
                n_players,
                jnp,
            )
            env_keys = jax.random.split(env_key, initial_state.positions.shape[0])
            out = _step_batch_minimal_impl(static, state, full_actions, env_keys, jax, jnp)
            reset_keys = jax.random.split(reset_key, initial_state.positions.shape[0])
            reset_state = reset_batch_minimal(static, reset_keys, jax, jnp)
            next_state = _replace_done_states(out.state, reset_state, out.done, jnp)
            aggregated_reward = build_aggregated_reward_batch(static, out.rewards, jnp)
            return (next_state, key), (
                aggregated_reward,
                out.done.astype(jnp.int8),
            )

        return jax.lax.scan(
            _scan_step,
            (initial_state, rollout_key),
            xs=None,
            length=int(horizon),
        )

    return {
        "compiled_rollout_legal_random_minimal": jax.jit(
            _compiled_rollout_random_opponent_kernel,
            static_argnums=(3,),
        ),
        "compiled_rollout_phase_a_policy_minimal": jax.jit(
            _compiled_rollout_phase_a_policy_kernel,
            static_argnums=(4,),
        ),
    }


def _block_until_ready_tree(value):
    if isinstance(value, dict):
        for item in value.values():
            _block_until_ready_tree(item)
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            _block_until_ready_tree(item)
        return
    if hasattr(value, "block_until_ready"):
        value.block_until_ready()


def benchmark_compiled_kernel(
    fn,
    static,
    state,
    batch_size: int,
    iterations: int,
    progress=None,
    progress_label: str | None = None,
):
    return benchmark_compiled_call(
        fn,
        batch_size,
        iterations,
        static,
        state,
        progress=progress,
        progress_label=progress_label,
    )


def benchmark_compiled_call(
    fn,
    batch_size: int,
    iterations: int,
    *call_args,
    progress=None,
    progress_label: str | None = None,
):
    if progress is not None and progress_label:
        progress.set_postfix_str(progress_label, refresh=False)
    with Timer() as timer:
        result = None
        for _ in range(int(iterations)):
            result = fn(*call_args)
            if progress is not None:
                progress.update(1)
        _block_until_ready_tree(result)
    elapsed_sec = timer.elapsed_ns / 1e9
    total_items = int(batch_size) * int(iterations)
    return {
        "iterations": int(iterations),
        "batch_size": int(batch_size),
        "total_states": int(total_items),
        "elapsed_sec": float(elapsed_sec),
        "states_per_sec": (float(total_items) / elapsed_sec) if elapsed_sec > 0.0 else 0.0,
        "mean_batch_latency_ms": (timer.elapsed_ns / max(1, int(iterations)) / 1e6),
    }


def benchmark_host_adapter_call(
    fn,
    jax,
    batch_size: int,
    iterations: int,
    *call_args,
    progress=None,
    progress_label: str | None = None,
):
    if progress is not None and progress_label:
        progress.set_postfix_str(progress_label, refresh=False)
    host_result = None
    with Timer() as timer:
        for _ in range(int(iterations)):
            host_result = jax.device_get(fn(*call_args))
            if progress is not None:
                progress.update(1)
    # Touch representative fields so the benchmark includes Python-side payload access.
    if isinstance(host_result, tuple) and len(host_result) == 3:
        obs_dict, rewards, done = host_result
        _ = (
            np.asarray(obs_dict["players"]).shape[0],
            np.asarray(obs_dict["globals"]).shape[0],
            np.asarray(obs_dict["action_mask"]).shape[0],
            np.asarray(rewards).shape[0],
            np.asarray(done).shape[0],
        )
    elapsed_sec = timer.elapsed_ns / 1e9
    total_items = int(batch_size) * int(iterations)
    return {
        "iterations": int(iterations),
        "batch_size": int(batch_size),
        "total_states": int(total_items),
        "elapsed_sec": float(elapsed_sec),
        "states_per_sec": (float(total_items) / elapsed_sec) if elapsed_sec > 0.0 else 0.0,
        "mean_batch_latency_ms": (timer.elapsed_ns / max(1, int(iterations)) / 1e6),
    }


def benchmark_host_callable(
    fn,
    batch_size: int,
    iterations: int,
    *,
    progress=None,
    progress_label: str | None = None,
):
    if progress is not None and progress_label:
        progress.set_postfix_str(progress_label, refresh=False)
    result = None
    with Timer() as timer:
        for _ in range(int(iterations)):
            result = fn()
            if progress is not None:
                progress.update(1)
    if isinstance(result, np.ndarray):
        _ = result.shape[0] if result.ndim > 0 else int(result)
    elapsed_sec = timer.elapsed_ns / 1e9
    total_items = int(batch_size) * int(iterations)
    return {
        "iterations": int(iterations),
        "batch_size": int(batch_size),
        "total_states": int(total_items),
        "elapsed_sec": float(elapsed_sec),
        "states_per_sec": (float(total_items) / elapsed_sec) if elapsed_sec > 0.0 else 0.0,
        "mean_batch_latency_ms": (timer.elapsed_ns / max(1, int(iterations)) / 1e6),
    }


def benchmark_compiled_rollout_call(
    fn,
    batch_size: int,
    horizon: int,
    iterations: int,
    *call_args,
    progress=None,
    progress_label: str | None = None,
):
    if progress is not None and progress_label:
        progress.set_postfix_str(progress_label, refresh=False)
    with Timer() as timer:
        result = None
        for _ in range(int(iterations)):
            result = fn(*call_args)
            if progress is not None:
                progress.update(1)
        _block_until_ready_tree(result)
    elapsed_sec = timer.elapsed_ns / 1e9
    total_items = int(batch_size) * int(horizon) * int(iterations)
    return {
        "iterations": int(iterations),
        "batch_size": int(batch_size),
        "horizon": int(horizon),
        "total_states": int(total_items),
        "elapsed_sec": float(elapsed_sec),
        "states_per_sec": (float(total_items) / elapsed_sec) if elapsed_sec > 0.0 else 0.0,
        "mean_batch_latency_ms": (timer.elapsed_ns / max(1, int(iterations)) / 1e6),
    }


def build_policy_kwargs(args) -> dict[str, Any]:
    policy_kwargs: dict[str, Any] = {}
    if getattr(args, "net_arch", None) is not None:
        policy_kwargs["net_arch"] = args.net_arch
    else:
        policy_kwargs["net_arch"] = dict(
            pi=getattr(args, "net_arch_pi", [64, 64]),
            vf=getattr(args, "net_arch_vf", [64, 64]),
        )

    if bool(getattr(args, "use_set_obs", False)):
        policy_kwargs["embed_dim"] = int(getattr(args, "set_embed_dim", 64))
        policy_kwargs["n_heads"] = int(getattr(args, "set_heads", 4))
        policy_kwargs["token_mlp_dim"] = int(getattr(args, "set_token_mlp_dim", 64))
        policy_kwargs["num_cls_tokens"] = int(getattr(args, "set_cls_tokens", 2))
        policy_kwargs["token_activation"] = str(getattr(args, "set_token_activation", "relu"))
        policy_kwargs["head_activation"] = str(getattr(args, "set_head_activation", "tanh"))
        policy_kwargs["intent_embedding_enabled"] = bool(
            getattr(args, "set_intent_embedding_enabled", False)
        )
        policy_kwargs["intent_embedding_dim"] = int(
            getattr(args, "set_intent_embedding_dim", 16)
        )
        policy_kwargs["num_intents"] = int(getattr(args, "num_intents", 8))
        policy_kwargs["intent_selector_enabled"] = bool(
            getattr(args, "intent_selector_enabled", False)
        )
        policy_kwargs["intent_selector_hidden_dim"] = int(
            getattr(args, "intent_selector_hidden_dim", 64)
        )
    else:
        policy_kwargs["features_extractor_class"] = MaskAgnosticCombinedExtractor

    if bool(getattr(args, "use_dual_policy", False)):
        policy_kwargs["use_dual_policy"] = True
    return policy_kwargs


def build_policy_benchmark_model(args):
    temp_env = DummyVecEnv([lambda: make_policy_init_env(args)])
    use_integrated_intent_selector = bool(
        getattr(args, "intent_selector_enabled", False)
        and str(getattr(args, "intent_selector_mode", "callback")).lower() == "integrated"
    )
    algorithm_class = (
        IntegratedIntentSelectorPPO if use_integrated_intent_selector else PPO
    )
    algo_kwargs: dict[str, Any] = {}
    if use_integrated_intent_selector:
        algo_kwargs.update(
            intent_selector_enabled=True,
            num_intents=int(getattr(args, "num_intents", 8)),
            intent_selector_alpha_start=float(
                getattr(args, "intent_selector_alpha_start", 0.0)
            ),
            intent_selector_alpha_end=float(
                getattr(args, "intent_selector_alpha_end", 1.0)
            ),
            intent_selector_alpha_warmup_steps=int(
                getattr(args, "intent_selector_alpha_warmup_steps", 0)
            ),
            intent_selector_alpha_ramp_steps=int(
                getattr(args, "intent_selector_alpha_ramp_steps", 1)
            ),
            intent_selector_eps_start=float(
                getattr(args, "intent_selector_eps_start", 0.0)
            ),
            intent_selector_eps_end=float(
                getattr(args, "intent_selector_eps_end", 0.0)
            ),
            intent_selector_eps_warmup_steps=int(
                getattr(args, "intent_selector_eps_warmup_steps", 0)
            ),
            intent_selector_eps_ramp_steps=int(
                getattr(args, "intent_selector_eps_ramp_steps", 1)
            ),
            intent_selector_entropy_coef=float(
                getattr(args, "intent_selector_entropy_coef", 0.01)
            ),
            intent_selector_usage_reg_coef=float(
                getattr(args, "intent_selector_usage_reg_coef", 0.01)
            ),
            intent_selector_value_coef=float(
                getattr(args, "intent_selector_value_coef", 0.5)
            ),
            intent_selector_template_metrics_log_every_rollouts=int(
                getattr(args, "intent_selector_template_metrics_log_every_rollouts", 8)
            ),
            intent_selector_train_every_rollouts=int(
                getattr(args, "intent_selector_train_every_rollouts", 1)
            ),
            intent_selector_max_samples_per_update=int(
                getattr(args, "intent_selector_max_samples_per_update", 0)
            ),
            intent_selector_multiselect_enabled=bool(
                getattr(args, "intent_selector_multiselect_enabled", False)
            ),
            intent_selector_min_play_steps=int(
                getattr(args, "intent_selector_min_play_steps", 3)
            ),
            intent_commitment_steps=int(
                getattr(args, "intent_commitment_steps", 4)
            ),
        )

    if bool(getattr(args, "use_set_obs", False)):
        policy_class = SetAttentionDualCriticPolicy
    else:
        policy_class = (
            PassBiasDualCriticPolicy
            if bool(getattr(args, "use_dual_critic", False))
            else PassBiasMultiInputPolicy
        )

    model = algorithm_class(
        policy_class,
        temp_env,
        verbose=0,
        n_steps=max(8, int(getattr(args, "players", 3) or 3)),
        n_epochs=1,
        vf_coef=float(getattr(args, "vf_coef", 0.75) or 0.75),
        ent_coef=float(getattr(args, "ent_coef", 0.0) or 0.0),
        batch_size=max(8, int(getattr(args, "players", 3) or 3)),
        learning_rate=float(getattr(args, "learning_rate", 1e-3) or 1e-3),
        tensorboard_log=None,
        policy_kwargs=build_policy_kwargs(args),
        target_kl=getattr(args, "target_kl", None),
        device=getattr(args, "device", "auto"),
        **algo_kwargs,
    )
    try:
        policy_obj = getattr(model, "policy", None)
        if policy_obj is not None and hasattr(policy_obj, "set_pass_mode"):
            policy_obj.set_pass_mode(getattr(args, "pass_mode", "directional"))
        if policy_obj is not None and hasattr(policy_obj, "set_pass_prob_min"):
            policy_obj.set_pass_prob_min(float(getattr(args, "pass_prob_min", 0.0) or 0.0))
    except Exception:
        pass
    return model, temp_env


def _copy_host_obs(obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: np.array(value, copy=True) for key, value in obs.items()}


def build_opponent_observation_host(obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = _copy_host_obs(obs)
    out["role_flag"] = -np.asarray(out["role_flag"], dtype=np.float32)
    return out


def _normalize_batched_actions(actions, batch_size: int) -> np.ndarray:
    arr = np.asarray(actions, dtype=np.int64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[0] == 1 and batch_size > 1:
        arr = np.repeat(arr, batch_size, axis=0)
    return arr


def _normalize_probs_per_player(
    probs_per_player: list[np.ndarray] | None,
    batch_size: int,
) -> list[np.ndarray] | None:
    if probs_per_player is None:
        return None
    out: list[np.ndarray] = []
    for probs in probs_per_player:
        arr = np.asarray(probs, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.shape[0] == 1 and batch_size > 1:
            arr = np.repeat(arr, batch_size, axis=0)
        out.append(arr)
    return out


def _masked_sample_actions_from_prob_tensor(
    torch_mod,
    probs,
    action_mask,
    *,
    deterministic: bool,
):
    probs_t = probs.to(dtype=torch_mod.float32)
    if probs_t.ndim == 2:
        probs_t = probs_t.unsqueeze(0)
    mask_t = torch_mod.as_tensor(action_mask, dtype=torch_mod.float32, device=probs_t.device)
    if mask_t.ndim == 2:
        mask_t = mask_t.unsqueeze(0)

    masked = probs_t * mask_t
    totals = masked.sum(dim=-1, keepdim=True)
    has_legal = totals > 0.0

    fallback = torch_mod.zeros_like(masked)
    fallback[..., 0] = 1.0
    normalized = torch_mod.where(has_legal, masked / torch_mod.clamp_min(totals, 1e-12), fallback)

    if deterministic:
        sampled = torch_mod.argmax(normalized, dim=-1)
    else:
        sampled = torch_mod.distributions.Categorical(probs=normalized).sample()
    return sampled.to(dtype=torch_mod.int64), normalized


def _get_policy_action_probabilities_tensor(policy, obs):
    policy_obj = getattr(policy, "policy", None)
    if policy_obj is None:
        return None
    try:
        import torch
    except ImportError:
        return None

    try:
        obs_tensor = policy_obj.obs_to_tensor(obs)[0]
        with (
            policy_obj.runtime_conditioning_context(obs_tensor)
            if hasattr(policy_obj, "runtime_conditioning_context")
            else nullcontext()
        ):
            distributions = policy_obj.get_distribution(obs_tensor)
        if hasattr(distributions, "action_probabilities"):
            probs = distributions.action_probabilities()
            if probs.ndim == 2:
                probs = probs.unsqueeze(0)
            return probs.to(dtype=torch.float32)
        stacked = torch.stack(
            [dist.probs.to(dtype=torch.float32) for dist in distributions.distribution],
            dim=1,
        )
        return stacked
    except Exception:
        return None


def sample_masked_policy_actions_host(
    *,
    policy,
    obs,
    action_mask: np.ndarray,
    deterministic: bool,
) -> np.ndarray:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - torch is expected in benchmark env
        raise RuntimeError("PyTorch is required for masked policy sampling.") from exc

    probs = _get_policy_action_probabilities_tensor(policy, obs)
    if probs is None:
        raise RuntimeError("Policy probabilities are unavailable for masked sampling.")

    sampled, _ = _masked_sample_actions_from_prob_tensor(
        torch,
        probs,
        action_mask,
        deterministic=deterministic,
    )
    return sampled.detach().cpu().numpy().astype(np.int64, copy=False)


def _team_player_ids_from_static(static: KernelStatic) -> tuple[np.ndarray, np.ndarray]:
    training_mask = np.asarray(static.training_player_mask, dtype=np.float32)
    training_ids = np.flatnonzero(training_mask > 0.5).astype(np.int64)
    opponent_ids = np.flatnonzero(training_mask <= 0.5).astype(np.int64)
    return training_ids, opponent_ids


def run_self_play_bridge_once(
    *,
    training_policy,
    opponent_policy,
    host_obs: dict[str, np.ndarray],
    training_player_ids: np.ndarray,
    opponent_player_ids: np.ndarray,
    deterministic_training: bool = False,
    deterministic_opponent: bool = False,
) -> np.ndarray:
    batch_size = int(np.asarray(host_obs["role_flag"]).shape[0])
    action_mask = np.asarray(host_obs["action_mask"], dtype=np.int8)
    n_players = int(action_mask.shape[1])

    training_actions_raw, _ = training_policy.predict(
        host_obs, deterministic=deterministic_training
    )
    training_actions_raw = _normalize_batched_actions(training_actions_raw, batch_size)

    opponent_obs = build_opponent_observation_host(host_obs)
    opponent_actions_raw, _ = opponent_policy.predict(
        opponent_obs, deterministic=deterministic_opponent
    )
    opponent_actions_raw = _normalize_batched_actions(opponent_actions_raw, batch_size)
    opponent_probs = _normalize_probs_per_player(
        get_policy_action_probabilities(opponent_policy, opponent_obs),
        batch_size,
    )

    full_actions = np.zeros((batch_size, n_players), dtype=np.int64)
    training_action_dim = int(action_mask.shape[-1])
    for env_idx in range(batch_size):
        training_mask = action_mask[env_idx, training_player_ids]
        opponent_mask = action_mask[env_idx, opponent_player_ids]
        training_probs = [
            np.ones(training_action_dim, dtype=np.float32)
            for _ in range(int(training_player_ids.shape[0]))
        ]
        opponent_probs_env = None
        if opponent_probs is not None and len(opponent_probs) == int(opponent_player_ids.shape[0]):
            opponent_probs_env = [opponent_probs[idx][env_idx] for idx in range(len(opponent_probs))]

        resolved_training = resolve_illegal_actions(
            training_actions_raw[env_idx],
            training_mask,
            IllegalActionStrategy.SAMPLE_PROB,
            deterministic_training,
            training_probs,
        )
        resolved_opponent = resolve_illegal_actions(
            opponent_actions_raw[env_idx],
            opponent_mask,
            IllegalActionStrategy.SAMPLE_PROB,
            deterministic_opponent,
            opponent_probs_env,
        )

        full_actions[env_idx, training_player_ids] = resolved_training
        full_actions[env_idx, opponent_player_ids] = resolved_opponent
    return full_actions


def run_self_play_bridge_masked_sampling_once(
    *,
    training_policy,
    opponent_policy,
    host_obs: dict[str, np.ndarray],
    training_player_ids: np.ndarray,
    opponent_player_ids: np.ndarray,
    deterministic_training: bool = False,
    deterministic_opponent: bool = False,
) -> np.ndarray:
    action_mask = np.asarray(host_obs["action_mask"], dtype=np.int8)
    batch_size = int(action_mask.shape[0])
    n_players = int(action_mask.shape[1])

    training_mask = action_mask[:, training_player_ids, :]
    opponent_obs = build_opponent_observation_host(host_obs)
    opponent_mask = action_mask[:, opponent_player_ids, :]

    training_actions = sample_masked_policy_actions_host(
        policy=training_policy,
        obs=host_obs,
        action_mask=training_mask,
        deterministic=deterministic_training,
    )
    opponent_actions = sample_masked_policy_actions_host(
        policy=opponent_policy,
        obs=opponent_obs,
        action_mask=opponent_mask,
        deterministic=deterministic_opponent,
    )

    training_actions = _normalize_batched_actions(training_actions, batch_size)
    opponent_actions = _normalize_batched_actions(opponent_actions, batch_size)

    full_actions = np.zeros((batch_size, n_players), dtype=np.int64)
    full_actions[:, training_player_ids] = training_actions
    full_actions[:, opponent_player_ids] = opponent_actions
    return full_actions


def _transition_benchmark_inputs(static, state, batch_size: int, jax, jnp):
    masks = build_action_masks_batch(static, state, jnp)
    actions = _pick_deterministic_legal_actions(masks, jnp)
    rng_keys = jax.random.split(jax.random.PRNGKey(123), int(batch_size))
    return actions, rng_keys


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


def run_jax_kernel_benchmark(args):
    _require_supported_scope(args)
    jax, jnp = ensure_jax_available("benchmarks/jax_kernel.py")
    static, state = sample_state_batch(args, xp=jnp)
    kernels = compile_hotspot_kernels(jax, jnp)
    transition_blockers = _minimal_transition_scope_blockers(args)
    transition_kernels = None if transition_blockers else compile_transition_kernels(jax, jnp)
    compiled_rollout_kernels = (
        None if transition_blockers else compile_compiled_rollout_kernels(jax, jnp)
    )
    transition_kernel_count = 0 if transition_kernels is None else len(transition_kernels)
    adapter_metric_count = 0 if transition_kernels is None else 1
    policy_bridge_enabled = transition_kernels is not None and bool(getattr(args, "use_set_obs", False))
    policy_bridge_metric_count = 7 if policy_bridge_enabled else 0
    compiled_rollout_enabled = compiled_rollout_kernels is not None
    compiled_rollout_metric_count = 2 if compiled_rollout_enabled else 0
    total_progress_iters = (
        len(kernels) * (int(args.warmup_iters) + int(args.benchmark_iters))
        + transition_kernel_count * (int(args.warmup_iters) + int(args.benchmark_iters))
        + adapter_metric_count * (int(args.warmup_iters) + int(args.benchmark_iters))
        + policy_bridge_metric_count * (int(args.warmup_iters) + int(args.benchmark_iters))
        + compiled_rollout_metric_count * (int(args.warmup_iters) + int(args.benchmark_iters))
    )
    progress = build_progress(
        total=total_progress_iters,
        desc="jax_kernel",
        disable=bool(getattr(args, "no_progress", False)),
        unit="iter",
    )

    try:
        warmup_metrics = {}
        for name, fn in kernels.items():
            warmup_metrics[name] = benchmark_compiled_kernel(
                fn=fn,
                static=static,
                state=state,
                batch_size=int(args.kernel_batch_size),
                iterations=int(args.warmup_iters),
                progress=progress,
                progress_label=f"warmup:{name}",
            )

        metrics = {}
        for name, fn in kernels.items():
            metrics[name] = benchmark_compiled_kernel(
                fn=fn,
                static=static,
                state=state,
                batch_size=int(args.kernel_batch_size),
                iterations=int(args.benchmark_iters),
                progress=progress,
                progress_label=f"benchmark:{name}",
            )

        transition_scope = {
            "implemented": [
                "reduced reset_batch benchmark",
                "legal-action-only reduced step_batch",
                "pointer_targeted pass mode",
                "deterministic legal-action selection for benchmark rollouts",
                "rollout-like benchmark path: action masks -> step_batch -> raw observation -> next masks",
            ],
            "required_disabled_features": [
                "phi_shaping",
                "illegal_defense",
                "offensive_three_seconds",
            ],
        }
        policy_bridge_scope = {
            "required_features": [
                "use_set_obs",
                "reduced transition scope",
            ],
            "implemented": [
                "training policy predict on host set-observation payload",
                "opponent policy predict on flipped-role host payload",
                "opponent action-probability extraction",
                "self-play action resolution and full-action assembly",
                "masked policy-action sampling without host-side illegal-action repair",
                "end-to-end bridge benchmark including JAX adapter output",
            ],
        }
        compiled_rollout_scope = {
            "required_features": [
                "reduced transition scope",
            ],
            "implemented": [
                "pure JAX rollout loop via lax.scan",
                "reduced reset_batch sampled on device for done episodes",
                "on-device legal-random opponent actions",
                "reduced flat JAX policy inside the compiled loop",
                "on-device masked action sampling for training-team actions",
            ],
            "limitations": [
                "uses a reduced flat JAX policy, not the current SB3/Torch policy stack",
                "uses a legal-random opponent, not self-play parity",
            ],
        }
        transition_warmup_metrics = {}
        transition_metrics = {}
        adapter_warmup_metrics = {}
        adapter_metrics = {}
        policy_bridge_warmup_metrics = {}
        policy_bridge_metrics = {}
        compiled_rollout_warmup_metrics = {}
        compiled_rollout_metrics = {}
        if transition_blockers:
            transition_scope["status"] = "skipped"
            transition_scope["blocked_by"] = transition_blockers
            policy_bridge_scope["status"] = "skipped"
            policy_bridge_scope["blocked_by"] = list(transition_blockers)
            compiled_rollout_scope["status"] = "skipped"
            compiled_rollout_scope["blocked_by"] = list(transition_blockers)
        else:
            transition_scope["status"] = "enabled"
            actions, rng_keys = _transition_benchmark_inputs(
                static=static,
                state=state,
                batch_size=int(args.kernel_batch_size),
                jax=jax,
                jnp=jnp,
            )
            for name, fn in transition_kernels.items():
                call_args = (
                    (static, rng_keys)
                    if name == "reset_batch_minimal"
                    else (
                        (static, state, actions, rng_keys)
                        if name == "step_batch_minimal"
                        else (static, state, rng_keys)
                    )
                )
                transition_warmup_metrics[name] = benchmark_compiled_call(
                    fn,
                    int(args.kernel_batch_size),
                    int(args.warmup_iters),
                    *call_args,
                    progress=progress,
                    progress_label=f"warmup:{name}",
                )
            for name, fn in transition_kernels.items():
                call_args = (
                    (static, rng_keys)
                    if name == "reset_batch_minimal"
                    else (
                        (static, state, actions, rng_keys)
                        if name == "step_batch_minimal"
                        else (static, state, rng_keys)
                    )
                )
                transition_metrics[name] = benchmark_compiled_call(
                    fn,
                    int(args.kernel_batch_size),
                    int(args.benchmark_iters),
                    *call_args,
                    progress=progress,
                    progress_label=f"benchmark:{name}",
                )
            adapter_fn = transition_kernels["sb3_payload_minimal_device"]
            adapter_call_args = (static, state, rng_keys)
            adapter_warmup_metrics["sb3_payload_minimal_host"] = benchmark_host_adapter_call(
                adapter_fn,
                jax,
                int(args.kernel_batch_size),
                int(args.warmup_iters),
                *adapter_call_args,
                progress=progress,
                progress_label="warmup:sb3_payload_minimal_host",
            )
            adapter_metrics["sb3_payload_minimal_host"] = benchmark_host_adapter_call(
                adapter_fn,
                jax,
                int(args.kernel_batch_size),
                int(args.benchmark_iters),
                *adapter_call_args,
                progress=progress,
                progress_label="benchmark:sb3_payload_minimal_host",
            )

            if not bool(getattr(args, "use_set_obs", False)):
                policy_bridge_scope["status"] = "skipped"
                policy_bridge_scope["blocked_by"] = ["use_set_obs must be true"]
            else:
                policy_bridge_scope["status"] = "enabled"
                training_policy, policy_env = build_policy_benchmark_model(args)
                try:
                    adapter_fn = transition_kernels["sb3_payload_minimal_device"]
                    host_obs_seed, _, _ = jax.device_get(adapter_fn(static, state, rng_keys))
                    training_player_ids, opponent_player_ids = _team_player_ids_from_static(static)
                    opponent_obs_seed = build_opponent_observation_host(host_obs_seed)

                    def _training_predict_once():
                        return _normalize_batched_actions(
                            training_policy.predict(host_obs_seed, deterministic=False)[0],
                            int(args.kernel_batch_size),
                        )

                    def _opponent_predict_once():
                        return _normalize_batched_actions(
                            training_policy.predict(opponent_obs_seed, deterministic=False)[0],
                            int(args.kernel_batch_size),
                        )

                    def _opponent_probs_once():
                        probs = _normalize_probs_per_player(
                            get_policy_action_probabilities(training_policy, opponent_obs_seed),
                            int(args.kernel_batch_size),
                        )
                        if probs is None:
                            return np.zeros(
                                (int(args.kernel_batch_size), int(opponent_player_ids.shape[0]), ACTION_COUNT),
                                dtype=np.float32,
                            )
                        return np.stack(probs, axis=1)

                    def _bridge_host_once():
                        return run_self_play_bridge_once(
                            training_policy=training_policy,
                            opponent_policy=training_policy,
                            host_obs=host_obs_seed,
                            training_player_ids=training_player_ids,
                            opponent_player_ids=opponent_player_ids,
                            deterministic_training=False,
                            deterministic_opponent=False,
                        )

                    def _bridge_with_adapter_once():
                        host_obs, _, _ = jax.device_get(adapter_fn(static, state, rng_keys))
                        return run_self_play_bridge_once(
                            training_policy=training_policy,
                            opponent_policy=training_policy,
                            host_obs=host_obs,
                            training_player_ids=training_player_ids,
                            opponent_player_ids=opponent_player_ids,
                            deterministic_training=False,
                            deterministic_opponent=False,
                        )

                    def _bridge_masked_sampling_host_once():
                        return run_self_play_bridge_masked_sampling_once(
                            training_policy=training_policy,
                            opponent_policy=training_policy,
                            host_obs=host_obs_seed,
                            training_player_ids=training_player_ids,
                            opponent_player_ids=opponent_player_ids,
                            deterministic_training=False,
                            deterministic_opponent=False,
                        )

                    def _bridge_masked_sampling_with_adapter_once():
                        host_obs, _, _ = jax.device_get(adapter_fn(static, state, rng_keys))
                        return run_self_play_bridge_masked_sampling_once(
                            training_policy=training_policy,
                            opponent_policy=training_policy,
                            host_obs=host_obs,
                            training_player_ids=training_player_ids,
                            opponent_player_ids=opponent_player_ids,
                            deterministic_training=False,
                            deterministic_opponent=False,
                        )

                    policy_functions = {
                        "training_policy_predict_host": _training_predict_once,
                        "opponent_policy_predict_host": _opponent_predict_once,
                        "opponent_action_probabilities_host": _opponent_probs_once,
                        "self_play_bridge_host": _bridge_host_once,
                        "self_play_bridge_with_adapter": _bridge_with_adapter_once,
                        "self_play_bridge_masked_sampling_host": _bridge_masked_sampling_host_once,
                        "self_play_bridge_masked_sampling_with_adapter": _bridge_masked_sampling_with_adapter_once,
                    }
                    for name, fn in policy_functions.items():
                        policy_bridge_warmup_metrics[name] = benchmark_host_callable(
                            fn,
                            int(args.kernel_batch_size),
                            int(args.warmup_iters),
                            progress=progress,
                            progress_label=f"warmup:{name}",
                        )
                    for name, fn in policy_functions.items():
                        policy_bridge_metrics[name] = benchmark_host_callable(
                            fn,
                            int(args.kernel_batch_size),
                            int(args.benchmark_iters),
                            progress=progress,
                            progress_label=f"benchmark:{name}",
                        )
                finally:
                    try:
                        policy_env.close()
                    except Exception:
                        pass

            compiled_rollout_scope["status"] = "enabled"
            training_ids_np, _ = _team_player_ids_from_static(static)
            flat_obs_seed = np.asarray(
                jax.device_get(build_phase_a_flat_observation_batch(static, state, jnp)),
                dtype=np.float32,
            )
            mask_seed = np.asarray(
                jax.device_get(build_action_masks_batch(static, state, jnp))[:, training_ids_np, :],
                dtype=np.int8,
            )
            policy_spec = PhaseAPolicySpec(
                flat_obs_dim=int(flat_obs_seed.shape[1]),
                training_player_count=int(mask_seed.shape[1]),
                action_dim_per_player=int(mask_seed.shape[2]),
                total_action_dim=int(mask_seed.shape[1] * mask_seed.shape[2]),
                hidden_dims=tuple(int(v) for v in getattr(args, "phase_a_policy_hidden_dims", [128, 128])),
            )
            policy_params = init_phase_a_policy_params(
                jax,
                jnp,
                policy_spec,
                seed=int(getattr(args, "phase_a_policy_seed", 0)),
            )
            rollout_seed = jax.random.PRNGKey(int(getattr(args, "seed", 0)) + 4242)
            compiled_rollout_call_specs = {
                "compiled_rollout_legal_random_minimal": (
                    compiled_rollout_kernels["compiled_rollout_legal_random_minimal"],
                    (static, state, rollout_seed, int(args.horizon)),
                ),
                "compiled_rollout_phase_a_policy_minimal": (
                    compiled_rollout_kernels["compiled_rollout_phase_a_policy_minimal"],
                    (static, state, policy_params, rollout_seed, int(args.horizon)),
                ),
            }
            for name, (fn, call_args) in compiled_rollout_call_specs.items():
                compiled_rollout_warmup_metrics[name] = benchmark_compiled_rollout_call(
                    fn,
                    int(args.kernel_batch_size),
                    int(args.horizon),
                    int(args.warmup_iters),
                    *call_args,
                    progress=progress,
                    progress_label=f"warmup:{name}",
                )
            for name, (fn, call_args) in compiled_rollout_call_specs.items():
                compiled_rollout_metrics[name] = benchmark_compiled_rollout_call(
                    fn,
                    int(args.kernel_batch_size),
                    int(args.horizon),
                    int(args.benchmark_iters),
                    *call_args,
                    progress=progress,
                    progress_label=f"benchmark:{name}",
                )
    finally:
        progress.close()

    return {
        "script": "benchmarks/jax_kernel.py",
        "kernel_batch_size": int(args.kernel_batch_size),
        "warmup_iters": int(args.warmup_iters),
        "benchmark_iters": int(args.benchmark_iters),
        "seed": int(args.seed),
        "sample_reset_seed": int(args.sample_reset_seed),
        "training_team": resolve_training_team(args.training_team).name,
        "env_config": benchmark_args_snapshot(args),
        "devices": [str(device) for device in jax.devices()],
        "kernel_scope": {
            "implemented": [
                "batched action mask generation",
                "batched shot profile computation",
                "batched offensive expected-points computation",
                "batched defender-pressure turnover probabilities",
                "batched pass steal-probability computation",
                "batched raw observation vector assembly",
                "batched set-observation payload assembly",
                "reduced-scope legal-action step_batch benchmark",
                "reduced-scope reset_batch benchmark",
                "reduced-scope rollout-like benchmark path",
                "host-side SB3-style payload transfer benchmark",
                "env->kernel snapshot bridge for current benchmark config",
            ],
            "excluded": [
                "full-scope pure JAX step_batch",
                "intent observation fields",
                "full current self-play wrapper path",
                "SB3 policy inference cost",
            ],
        },
        "transition_scope": transition_scope,
        "policy_bridge_scope": policy_bridge_scope,
        "compiled_rollout_scope": compiled_rollout_scope,
        "warmup_metrics": warmup_metrics,
        "metrics": metrics,
        "transition_warmup_metrics": transition_warmup_metrics,
        "transition_metrics": transition_metrics,
        "adapter_warmup_metrics": adapter_warmup_metrics,
        "adapter_metrics": adapter_metrics,
        "policy_bridge_warmup_metrics": policy_bridge_warmup_metrics,
        "policy_bridge_metrics": policy_bridge_metrics,
        "compiled_rollout_warmup_metrics": compiled_rollout_warmup_metrics,
        "compiled_rollout_metrics": compiled_rollout_metrics,
    }


def print_summary(result):
    print("JAX hotspot kernel benchmark")
    print(f"kernel_batch_size: {result['kernel_batch_size']}")
    print(f"devices: {', '.join(result['devices'])}")
    for name, metrics in result["metrics"].items():
        print(
            f"{name}: states_per_sec={metrics['states_per_sec']:.2f} "
            f"mean_batch_latency_ms={metrics['mean_batch_latency_ms']:.4f}"
        )
    transition_scope = result.get("transition_scope", {})
    transition_metrics = result.get("transition_metrics", {})
    if transition_scope.get("status") == "enabled":
        for name, metrics in transition_metrics.items():
            print(
                f"{name}: states_per_sec={metrics['states_per_sec']:.2f} "
                f"mean_batch_latency_ms={metrics['mean_batch_latency_ms']:.4f}"
            )
        for name, metrics in result.get("adapter_metrics", {}).items():
            print(
                f"{name}: states_per_sec={metrics['states_per_sec']:.2f} "
                f"mean_batch_latency_ms={metrics['mean_batch_latency_ms']:.4f}"
            )
        for name, metrics in result.get("policy_bridge_metrics", {}).items():
            print(
                f"{name}: states_per_sec={metrics['states_per_sec']:.2f} "
                f"mean_batch_latency_ms={metrics['mean_batch_latency_ms']:.4f}"
            )
        for name, metrics in result.get("compiled_rollout_metrics", {}).items():
            print(
                f"{name}: states_per_sec={metrics['states_per_sec']:.2f} "
                f"mean_batch_latency_ms={metrics['mean_batch_latency_ms']:.4f}"
            )
    elif transition_scope.get("status") == "skipped":
        blocked = ", ".join(transition_scope.get("blocked_by", []))
        print(f"transition_scope: skipped ({blocked})")


def main(argv=None):
    args = parse_args(argv)
    result = run_jax_kernel_benchmark(args)
    print_summary(result)
    if args.output_json:
        write_json(args.output_json, result)
        print(f"wrote_json: {args.output_json}")


if __name__ == "__main__":
    main()
