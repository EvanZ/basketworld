"""A technical restart must not also record the interrupted turnover."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from basketworld.domain import ActionType
from basketworld_jax.config import JaxEnvironmentConfig
from basketworld_jax.env.minimal import (
    EPISODE_END_GAME_CLOCK,
    EPISODE_END_POSSESSIONS,
    GAME_PHASE_AWAITING_INBOUND,
    GAME_PHASE_LIVE,
    MULTI_POSSESSION_REWARD_WIN_LOSS,
    PASS_ACTION_START,
    POSSESSION_END_DEFENSIVE_VIOLATION,
    TEAM_A,
    TEAM_B,
    TURNOVER_REASON_INTERCEPTED,
    TURNOVER_REASON_NONE,
    build_kernel_static_from_config,
    reset_batch_minimal,
    step_batch_game_clock,
    step_batch_minimal,
)


@pytest.fixture(scope="module")
def scenario():
    static = build_kernel_static_from_config(
        JaxEnvironmentConfig(
            players=2, grid_size=7, three_point_distance=2.0,
            enable_multi_possession=True, multi_possession_limit=6,
            multi_possession_reward_mode="scoring_events",
            illegal_defense_enabled=True, defender_guard_distance=0.0,
            three_second_max_steps=0, defender_pressure_turnover_chance=0.0,
            base_steal_rate=1.0e6, pass_reward=0.5,
            enable_rebound_reward_redistribution=True,
            enable_phi_shaping=False,
        ), xp=jnp,
    )
    # Static arrays are arguments, avoiding expensive constant folding of the
    # whole court during compilation. This exercises the compiled training path.
    reset = jax.jit(lambda st, keys: reset_batch_minimal(st, keys, jax, jnp))
    state = reset(static, jax.random.split(jax.random.PRNGKey(501), 1))
    cells = np.asarray(static.cell_coords)
    available = {tuple(c) for c in cells}
    for middle in cells[np.asarray(static.defensive_lane_by_cell, dtype=bool)]:
        left, right = middle - [2, 0], middle + [2, 0]
        if tuple(left) in available and tuple(right) in available:
            break
    else:
        raise AssertionError("No pass segment across the defensive lane")
    far = max(cells, key=lambda c: np.linalg.norm(c - middle))
    positions = np.asarray([left, right, middle, far], dtype=np.int32)
    return static, state, positions


_possession_step = jax.jit(
    lambda st, s, a, k: step_batch_minimal(st, s, a, k, jax, jnp)
)
_clock_step = jax.jit(
    lambda st, s, a, k: step_batch_game_clock(st, s, a, k, jax, jnp)
)


@pytest.mark.parametrize("team", [TEAM_A, TEAM_B])
@pytest.mark.parametrize("timed", [False, True])
def test_technical_cancels_interception_for_training_and_evaluation(scenario, team, timed):
    static, initial, positions = scenario
    mode = EPISODE_END_GAME_CLOCK if timed else EPISODE_END_POSSESSIONS
    static = static._replace(episode_end_mode=jnp.asarray(mode, dtype=jnp.int8))
    offense = np.asarray(static.offense_ids if team == TEAM_A else static.defense_ids)
    defense = np.asarray(static.defense_ids if team == TEAM_A else static.offense_ids)
    reordered = np.empty_like(positions)
    reordered[offense] = positions[:2]
    reordered[defense] = positions[2:]
    holder, receiver = map(int, offense)
    state = initial._replace(
        episode_end_mode=jnp.asarray([mode], dtype=jnp.int8),
        game_clock_remaining=jnp.asarray([281], dtype=jnp.int32),
        positions=jnp.asarray(reordered[None]),
        ball_holder=jnp.asarray([holder], dtype=jnp.int32),
        offense_team=jnp.asarray([team], dtype=jnp.int8),
        starting_offense_team=jnp.asarray([team], dtype=jnp.int8),
        game_phase=jnp.asarray([GAME_PHASE_LIVE], dtype=jnp.int8),
        clearance_achieved=jnp.asarray([0], dtype=jnp.int8),
        shot_clock=jnp.asarray([7], dtype=jnp.int32),
        rebound_reward_advance_paid=jnp.asarray([0.4], dtype=jnp.float32),
    )
    slot = int(np.flatnonzero(np.asarray(static.pointer_pass_target_ids)[holder] == receiver)[0])
    actions = jnp.full((1, 4), ActionType.NOOP.value, dtype=jnp.int32)
    actions = actions.at[0, holder].set(PASS_ACTION_START + slot)
    keys = jax.random.split(jax.random.PRNGKey(502), 1)
    step = _clock_step if timed else _possession_step

    # Control: the same action and RNG really do produce an interception.
    control = step(static._replace(illegal_defense_enabled=jnp.asarray(0, dtype=jnp.int8)), state, actions, keys)
    assert int(control.turnover[0]) == 1
    assert int(control.turnover_reason[0]) == TURNOVER_REASON_INTERCEPTED
    assert int(control.steal_player[0]) in defense
    assert int(control.state.offense_team[0]) == 1 - team

    for inbounds in [True, False]:
        st = static._replace(multi_possession_use_inbounds=jnp.asarray(inbounds, dtype=jnp.int8))
        out = step(st, state, actions, keys)
        assert int(out.defensive_lane_violation[0]) == 1
        assert int(out.turnover[0]) == 0
        assert int(out.turnover_reason[0]) == TURNOVER_REASON_NONE
        assert int(out.turnover_player[0]) == -1
        assert int(out.steal_player[0]) == -1
        assert int(out.pass_attempt[0]) == int(out.completed_pass[0]) == 0
        assert int(out.pass_passer[0]) == int(out.pass_receiver[0]) == -1
        assert int(out.possession_ended[0]) == 0
        assert int(out.possession_end_reason[0]) == POSSESSION_END_DEFENSIVE_VIOLATION
        assert int(out.state.completed_possessions[0]) == 0
        assert int(out.state.offense_team[0]) == team
        assert int(out.state.clearance_achieved[0]) == 0
        assert int(out.state.shot_clock[0]) == 14
        assert int(out.state.game_clock_remaining[0]) == (280 if timed else 281)
        assert float(out.state.rebound_reward_advance_paid[0]) == pytest.approx(0.4)
        assert float(out.rebound_reward_settlement[0]) == 0.0
        assert int(out.state.game_phase[0]) == (GAME_PHASE_AWAITING_INBOUND if inbounds else GAME_PHASE_LIVE)
        assert int(out.state.ball_holder[0]) in offense
        assert float(out.state.team_a_score[0]) == (1.0 if team == TEAM_A else 0.0)
        assert float(out.state.team_b_score[0]) == (1.0 if team == TEAM_B else 0.0)
        np.testing.assert_allclose(np.asarray(out.rewards)[0, offense], 0.5)
        np.testing.assert_allclose(np.asarray(out.rewards)[0, defense], -0.5)

    # A successful pass on the whistle is also cancelled, with no pass bonus.
    successful = step(static._replace(base_steal_rate=jnp.asarray(0.0, dtype=jnp.float32)), state, actions, keys)
    assert int(successful.defensive_lane_violation[0]) == 1
    assert int(successful.pass_attempt[0]) == int(successful.completed_pass[0]) == 0
    np.testing.assert_allclose(successful.rewards, out.rewards)

    # Auxiliary rewards must not leak a pass bonus or settle the still-open
    # possession's rebound advance when the technical interrupts that pass.
    auxiliary_static = static._replace(
        base_steal_rate=jnp.asarray(0.0, dtype=jnp.float32),
        multi_possession_reward_mode=jnp.asarray(MULTI_POSSESSION_REWARD_WIN_LOSS, dtype=jnp.int32),
        multi_possession_aux_rewards_enabled=jnp.asarray(1, dtype=jnp.int8),
    )
    auxiliary = step(auxiliary_static, state, actions, keys)
    expected = float(static.violation_reward) / len(offense)
    np.testing.assert_allclose(np.asarray(auxiliary.rewards)[0, offense], expected)
    np.testing.assert_allclose(np.asarray(auxiliary.rewards)[0, defense], -expected)

    clean_pass = step(auxiliary_static._replace(illegal_defense_enabled=jnp.asarray(0, dtype=jnp.int8)), state, actions, keys)
    assert int(clean_pass.pass_attempt[0]) == int(clean_pass.completed_pass[0]) == 1
    np.testing.assert_allclose(np.asarray(clean_pass.rewards)[0, offense], float(static.pass_reward) / len(offense))
