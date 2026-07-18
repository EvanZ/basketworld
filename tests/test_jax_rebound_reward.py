from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from basketworld.envs.basketworld_env_v2 import ActionType, HexagonBasketballEnv
from basketworld_jax.env.minimal import (
    REBOUND_CONTEST_MODE_LOCAL,
    build_kernel_static_from_env,
    snapshot_state_from_env,
    stack_state_snapshots,
    step_batch_minimal,
)


def _reward_test_state(*, enabled: bool, once_per_possession: bool = True):
    env = HexagonBasketballEnv(
        players=2,
        render_mode=None,
        pass_mode="pointer_targeted",
        allow_dunks=False,
        layup_pct=0.0,
        three_pt_pct=0.0,
        dunk_pct=0.0,
        shot_pressure_enabled=False,
        defender_pressure_turnover_chance=0.0,
        base_steal_rate=0.0,
        illegal_defense_enabled=False,
        offensive_three_seconds_enabled=False,
        enable_phi_shaping=False,
        violation_reward=0.0,
    )
    env.rebound_target_temperature = 1.0
    env.rebound_target_uniform_mix = 0.0
    env.rebound_winner_distance_weight = 1.0
    env.rebound_basket_position_weight = 0.0
    env.rebound_winner_temperature = 1.0
    env.rebound_skill_weight = 0.0
    env.rebound_contest_mode = "local_contest"
    env.rebound_contest_radius = 0
    env.offensive_rebound_shot_clock_reset = 14
    env.enable_rebound_reward_redistribution = enabled
    env.offensive_rebound_reward_advance = 0.4
    env.rebound_reward_once_per_possession = once_per_possession
    env.reset(seed=19)
    env.shot_clock = 24
    env.step_count = 0
    env.episode_ended = False
    env._assist_candidate = None

    static = build_kernel_static_from_env(env, xp=jnp)
    state = stack_state_snapshots([snapshot_state_from_env(env)], xp=jnp)
    holder = int(np.asarray(state.ball_holder)[0])
    holder_position = np.asarray(state.positions)[0, holder]
    cell_coords = np.asarray(static.cell_coords)
    target_idx = int(np.flatnonzero(np.all(cell_coords == holder_position, axis=1))[0])
    rebound_probs = np.zeros(np.asarray(static.rebound_target_probs).shape, dtype=np.float32)
    rebound_probs[:, :, target_idx] = 1.0
    static = static._replace(
        enable_rebounds=jnp.asarray(1, dtype=jnp.int8),
        rebound_target_probs=jnp.asarray(rebound_probs, dtype=jnp.float32),
        rebound_contest_mode=jnp.asarray(REBOUND_CONTEST_MODE_LOCAL, dtype=jnp.int32),
        rebound_contest_radius=jnp.asarray(0, dtype=jnp.int32),
        enable_rebound_reward_redistribution=jnp.asarray(1 if enabled else 0, dtype=jnp.int8),
        offensive_rebound_reward_advance=jnp.asarray(0.4, dtype=jnp.float32),
        rebound_reward_once_per_possession=jnp.asarray(
            1 if once_per_possession else 0, dtype=jnp.int8
        ),
    )
    state = state._replace(
        layup_pct=jnp.zeros_like(state.layup_pct),
        three_pt_pct=jnp.zeros_like(state.three_pt_pct),
        dunk_pct=jnp.zeros_like(state.dunk_pct),
    )
    return static, state


def _step(static, state, *, shoot: bool, seed: int):
    actions = np.full(
        (1, int(static.role_encoding.shape[0])),
        ActionType.NOOP.value,
        dtype=np.int32,
    )
    if shoot:
        holder = int(np.asarray(state.ball_holder)[0])
        actions[0, holder] = ActionType.SHOOT.value
    return step_batch_minimal(
        static,
        state,
        jnp.asarray(actions, dtype=jnp.int32),
        jnp.asarray([jax.random.PRNGKey(seed)]),
        jax,
        jnp,
    )


def _offense_reward(static, out) -> float:
    rewards = np.asarray(out.rewards)[0]
    return float(rewards[np.asarray(static.offense_ids, dtype=np.int32)].sum())


def test_rebound_reward_redistribution_disabled_preserves_zero_miss_reward():
    static, state = _reward_test_state(enabled=False)

    out = _step(static, state, shoot=True, seed=1)

    assert int(np.asarray(out.offensive_rebound)[0]) == 1
    assert float(np.asarray(out.rebound_reward_advance)[0]) == pytest.approx(0.0)
    assert float(np.asarray(out.rebound_reward_settlement)[0]) == pytest.approx(0.0)
    assert float(np.asarray(out.state.rebound_reward_advance_paid)[0]) == pytest.approx(0.0)
    assert _offense_reward(static, out) == pytest.approx(0.0)


def test_rebound_reward_advance_is_deducted_from_later_made_shot():
    static, state = _reward_test_state(enabled=True)

    rebound_out = _step(static, state, shoot=True, seed=2)
    assert int(np.asarray(rebound_out.offensive_rebound)[0]) == 1
    assert float(np.asarray(rebound_out.rebound_reward_advance)[0]) == pytest.approx(0.4)
    assert _offense_reward(static, rebound_out) == pytest.approx(0.4)

    make_state = rebound_out.state._replace(
        layup_pct=jnp.ones_like(rebound_out.state.layup_pct),
        three_pt_pct=jnp.ones_like(rebound_out.state.three_pt_pct),
        dunk_pct=jnp.ones_like(rebound_out.state.dunk_pct),
    )
    make_out = _step(static, make_state, shoot=True, seed=3)
    shot_value = float(np.asarray(make_out.shot_value)[0])

    assert bool(np.asarray(make_out.done)[0])
    assert int(np.asarray(make_out.shot_success)[0]) == 1
    assert float(np.asarray(make_out.rebound_reward_settlement)[0]) == pytest.approx(-0.4)
    assert _offense_reward(static, make_out) == pytest.approx(shot_value - 0.4)
    assert _offense_reward(static, rebound_out) + _offense_reward(static, make_out) == pytest.approx(
        shot_value
    )
    assert float(np.asarray(make_out.state.rebound_reward_advance_paid)[0]) == pytest.approx(0.0)


def test_rebound_reward_settles_on_pressure_turnover_terminal_path():
    static, state = _reward_test_state(enabled=True)
    holder = int(np.asarray(state.ball_holder)[0])
    defense_id = int(np.asarray(static.defense_ids)[0])
    positions = np.asarray(state.positions).copy()
    positions[0, defense_id] = positions[0, holder]
    state = state._replace(
        positions=jnp.asarray(positions, dtype=jnp.int32),
        rebound_reward_advance_paid=jnp.asarray([0.4], dtype=jnp.float32),
    )
    static = static._replace(
        defender_pressure_distance=jnp.asarray(1.0, dtype=jnp.float32),
        defender_pressure_turnover_chance=jnp.asarray(1.0, dtype=jnp.float32),
        defender_pressure_decay_lambda=jnp.asarray(0.0, dtype=jnp.float32),
    )

    out = _step(static, state, shoot=False, seed=7)

    assert bool(np.asarray(out.done)[0])
    assert int(np.asarray(out.turnover)[0]) == 1
    assert float(np.asarray(out.rebound_reward_settlement)[0]) == pytest.approx(-0.4)
    assert _offense_reward(static, out) == pytest.approx(-0.4)
    assert float(np.asarray(out.state.rebound_reward_advance_paid)[0]) == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("once_per_possession", "expected_second_advance", "expected_ledger"),
    [(True, 0.0, 0.4), (False, 0.4, 0.8)],
)
def test_rebound_reward_multiple_orbs_settle_exactly(
    once_per_possession: bool,
    expected_second_advance: float,
    expected_ledger: float,
):
    static, state = _reward_test_state(
        enabled=True,
        once_per_possession=once_per_possession,
    )

    first_out = _step(static, state, shoot=True, seed=4)
    second_out = _step(static, first_out.state, shoot=True, seed=5)

    assert int(np.asarray(first_out.offensive_rebound)[0]) == 1
    assert int(np.asarray(second_out.offensive_rebound)[0]) == 1
    assert float(np.asarray(second_out.rebound_reward_advance)[0]) == pytest.approx(
        expected_second_advance
    )
    assert float(np.asarray(second_out.state.rebound_reward_advance_paid)[0]) == pytest.approx(
        expected_ledger
    )

    terminal_state = second_out.state._replace(
        shot_clock=jnp.asarray([1], dtype=second_out.state.shot_clock.dtype)
    )
    terminal_out = _step(static, terminal_state, shoot=False, seed=6)

    assert bool(np.asarray(terminal_out.done)[0])
    assert float(np.asarray(terminal_out.rebound_reward_settlement)[0]) == pytest.approx(
        -expected_ledger
    )
    total_offense_reward = sum(
        _offense_reward(static, out) for out in (first_out, second_out, terminal_out)
    )
    assert total_offense_reward == pytest.approx(0.0)
    assert float(np.asarray(terminal_out.state.rebound_reward_advance_paid)[0]) == pytest.approx(0.0)
