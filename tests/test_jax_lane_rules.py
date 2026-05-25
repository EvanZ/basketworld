from __future__ import annotations

import numpy as np
import pytest

from basketworld.envs.basketworld_env_v2 import ActionType, HexagonBasketballEnv
from basketworld_jax.env.minimal import (
    TURNOVER_REASON_OFFENSIVE_THREE_SECONDS,
    build_kernel_static_from_env,
    snapshot_state_from_env,
    stack_state_snapshots,
    step_batch_minimal,
)


def _make_lane_env(
    *,
    illegal_defense_enabled: bool,
    offensive_three_seconds_enabled: bool,
) -> HexagonBasketballEnv:
    env = HexagonBasketballEnv(
        players=3,
        render_mode=None,
        pass_mode="pointer_targeted",
        allow_dunks=True,
        shot_pressure_enabled=False,
        defender_pressure_turnover_chance=0.0,
        base_steal_rate=0.0,
        enable_phi_shaping=False,
        illegal_defense_enabled=illegal_defense_enabled,
        offensive_three_seconds_enabled=offensive_three_seconds_enabled,
        three_second_max_steps=3,
        three_second_lane_width=1,
        three_second_lane_height=3,
        defender_guard_distance=0,
        violation_reward=1.0,
    )
    env.reset(seed=11)
    env.shot_clock = 12
    env.step_count = 0
    env.episode_ended = False
    env._assist_candidate = None
    return env


def _sorted_valid_cells(env: HexagonBasketballEnv) -> list[tuple[int, int]]:
    return sorted(
        env._move_mask_by_cell.keys(),
        key=lambda cell: (env._hex_distance(cell, env.basket_position), cell[0], cell[1]),
    )


def _pick_cells(
    env: HexagonBasketballEnv,
    *,
    count: int,
    forbidden: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    cells = []
    for cell in _sorted_valid_cells(env):
        if cell in forbidden:
            continue
        if cell == tuple(env.basket_position):
            continue
        cells.append(cell)
        if len(cells) == count:
            return cells
    raise AssertionError("Could not find enough legal test cells.")


def _kernel_step(env: HexagonBasketballEnv, actions: np.ndarray):
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    static = build_kernel_static_from_env(env, xp=jnp)
    state = stack_state_snapshots([snapshot_state_from_env(env)], xp=jnp)
    return step_batch_minimal(
        static,
        state,
        jnp.asarray(actions[None, :], dtype=jnp.int32),
        jnp.asarray([jax.random.PRNGKey(0)]),
        jax,
        jnp,
    )


def test_jax_defensive_lane_violation_matches_env_reward_and_score():
    env = _make_lane_env(
        illegal_defense_enabled=True,
        offensive_three_seconds_enabled=False,
    )
    lane_cell = sorted(env.defensive_lane_hexes)[0]
    filler = _pick_cells(env, count=5, forbidden={lane_cell})
    env.positions = [
        filler[0],
        filler[1],
        filler[2],
        lane_cell,
        filler[3],
        filler[4],
    ]
    env.ball_holder = env.offense_ids[0]
    env._defender_in_key_steps = {env.defense_ids[0]: 3, env.defense_ids[1]: 0, env.defense_ids[2]: 0}
    actions = np.full(env.n_players, ActionType.NOOP.value, dtype=np.int32)

    out = _kernel_step(env, actions)
    _, env_rewards, env_done, _, _ = env.step(actions)
    next_state = out.state

    assert bool(np.asarray(out.done)[0]) == bool(env_done)
    assert int(np.asarray(out.defensive_lane_violation)[0]) == 1
    assert int(np.asarray(out.defensive_lane_violation_player)[0]) == int(env.defense_ids[0])
    assert float(np.asarray(next_state.offense_score)[0]) == pytest.approx(float(env.offense_score), abs=1e-6)
    assert np.asarray(out.rewards)[0].tolist() == pytest.approx(env_rewards.tolist(), abs=1e-6)


def test_jax_offensive_three_seconds_violation_matches_env_turnover():
    env = _make_lane_env(
        illegal_defense_enabled=False,
        offensive_three_seconds_enabled=True,
    )
    lane_cell = sorted(env.offensive_lane_hexes)[0]
    filler = _pick_cells(env, count=5, forbidden={lane_cell})
    env.positions = [
        filler[0],
        lane_cell,
        filler[1],
        filler[2],
        filler[3],
        filler[4],
    ]
    env.ball_holder = env.offense_ids[0]
    env._offensive_lane_steps = {env.offense_ids[0]: 0, env.offense_ids[1]: 2, env.offense_ids[2]: 0}
    actions = np.full(env.n_players, ActionType.NOOP.value, dtype=np.int32)

    out = _kernel_step(env, actions)
    _, env_rewards, env_done, _, _ = env.step(actions)
    next_state = out.state

    assert bool(np.asarray(out.done)[0]) == bool(env_done)
    assert int(np.asarray(out.offensive_three_seconds)[0]) == 1
    assert int(np.asarray(out.turnover_player)[0]) == int(env.offense_ids[1])
    assert int(np.asarray(out.turnover_reason)[0]) == int(TURNOVER_REASON_OFFENSIVE_THREE_SECONDS)
    assert int(np.asarray(next_state.ball_holder)[0]) == int(env.ball_holder)
    assert np.asarray(out.rewards)[0].tolist() == pytest.approx(env_rewards.tolist(), abs=1e-6)
