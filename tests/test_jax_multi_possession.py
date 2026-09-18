from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from basketworld.envs.basketworld_env_v2 import ActionType, HexagonBasketballEnv
from basketworld_jax.train.main import parse_args, validate_train_args
from basketworld_jax.env.minimal import (
    GAME_PHASE_AWAITING_INBOUND,
    GAME_PHASE_LIVE,
    PASS_ACTION_START,
    POSSESSION_END_DEFENSIVE_REBOUND,
    POSSESSION_END_MADE_BASKET,
    TEAM_A,
    TEAM_B,
    build_kernel_static_from_env,
    reset_batch_minimal,
    step_batch_minimal,
)


def _multi_possession_static():
    env = HexagonBasketballEnv(
        players=2,
        render_mode=None,
        pass_mode="pointer_targeted",
        allow_dunks=False,
        layup_pct=1.0,
        three_pt_pct=1.0,
        dunk_pct=1.0,
        shot_pressure_enabled=False,
        defender_pressure_turnover_chance=0.0,
        base_steal_rate=0.0,
        illegal_defense_enabled=False,
        offensive_three_seconds_enabled=False,
        enable_phi_shaping=False,
    )
    env.enable_multi_possession = True
    env.multi_possession_limit = 2
    return build_kernel_static_from_env(env, xp=jnp)


def test_multi_possession_cli_validates_limit_and_disables_start_templates():
    args = parse_args(["--enable-multi-possession", "--multi-possession-limit", "25"])
    assert args.enable_multi_possession is True
    assert args.multi_possession_limit == 25
    validate_train_args(args)

    with pytest.raises(SystemExit, match="multi-possession-limit"):
        validate_train_args(
            parse_args(["--enable-multi-possession", "--multi-possession-limit", "0"])
        )
    with pytest.raises(SystemExit, match="start-template-enabled"):
        validate_train_args(
            parse_args(["--enable-multi-possession", "--start-template-enabled", "true"])
        )


def _shoot(static, state, seed: int):
    batch_size, n_players, _ = state.positions.shape
    actions = jnp.full((batch_size, n_players), ActionType.NOOP.value, dtype=jnp.int32)
    actions = actions.at[jnp.arange(batch_size), state.ball_holder].set(ActionType.SHOOT.value)
    return step_batch_minimal(
        static,
        state,
        actions,
        jax.random.split(jax.random.PRNGKey(seed), batch_size),
        jax,
        jnp,
    )


def test_multi_possession_reset_uses_both_starting_teams_and_handoffs_after_terminal_shot():
    static = _multi_possession_static()
    state = reset_batch_minimal(
        static,
        jax.random.split(jax.random.PRNGKey(17), 4),
        jax,
        jnp,
    )
    starters = np.asarray(state.starting_offense_team, dtype=np.int8)
    holders = np.asarray(state.ball_holder, dtype=np.int32)
    team_a_ids = set(np.asarray(static.offense_ids, dtype=np.int32).tolist())
    team_b_ids = set(np.asarray(static.defense_ids, dtype=np.int32).tolist())

    assert {TEAM_A, TEAM_B}.issubset(set(starters.tolist()))
    for starter, holder in zip(starters, holders, strict=True):
        assert holder in (team_a_ids if starter == TEAM_A else team_b_ids)

    state = state._replace(
        layup_pct=jnp.ones_like(state.layup_pct),
        three_pt_pct=jnp.ones_like(state.three_pt_pct),
        dunk_pct=jnp.ones_like(state.dunk_pct),
    )
    out = _shoot(static, state, seed=3)
    assert np.all(np.asarray(out.possession_ended, dtype=np.int8) == 1)
    reasons = np.asarray(out.possession_end_reason, dtype=np.int32)
    assert set(reasons.tolist()).issubset(
        {POSSESSION_END_MADE_BASKET, POSSESSION_END_DEFENSIVE_REBOUND}
    )
    assert np.any(reasons == POSSESSION_END_MADE_BASKET)
    assert not np.any(np.asarray(out.done, dtype=bool))
    assert np.all(np.asarray(out.state.completed_possessions, dtype=np.int32) == 1)
    assert np.all(np.asarray(out.state.game_phase, dtype=np.int8) == GAME_PHASE_AWAITING_INBOUND)
    assert np.all(np.asarray(out.state.ball_holder, dtype=np.int32) == -1)
    assert np.all(np.asarray(out.state.inbound_player, dtype=np.int32) == -1)
    assert np.all(np.asarray(out.state.inbound_reason, dtype=np.int32) == reasons)
    assert np.all(np.asarray(out.state.clearance_achieved, dtype=np.int8) == 0)
    np.testing.assert_array_equal(np.asarray(out.state.positions), np.asarray(state.positions))
    np.testing.assert_array_equal(np.asarray(out.state.layup_pct), np.asarray(state.layup_pct))
    np.testing.assert_array_equal(np.asarray(out.state.rebound_skill), np.asarray(state.rebound_skill))
    np.testing.assert_array_equal(
        np.asarray(out.state.offense_team, dtype=np.int8),
        1 - starters,
    )
    np.testing.assert_array_equal(
        np.asarray(out.state.inbound_team, dtype=np.int8),
        1 - starters,
    )

    # #20 will replace this direct phase advance with the true baseline inbound.
    next_offense_ids = np.where(
        np.asarray(out.state.offense_team, dtype=np.int8)[:, None] == TEAM_A,
        np.asarray(static.offense_ids, dtype=np.int32)[None, :],
        np.asarray(static.defense_ids, dtype=np.int32)[None, :],
    )
    resumed = out.state._replace(
        game_phase=jnp.full_like(out.state.game_phase, GAME_PHASE_LIVE),
        ball_holder=jnp.asarray(next_offense_ids[:, 0], dtype=jnp.int32),
    )
    final_out = _shoot(static, resumed, seed=4)

    assert np.all(np.asarray(final_out.possession_ended, dtype=np.int8) == 1)
    assert np.all(np.asarray(final_out.done, dtype=bool))
    assert np.all(np.asarray(final_out.state.episode_ended, dtype=np.int8) == 1)
    assert np.all(np.asarray(final_out.state.completed_possessions, dtype=np.int32) == 2)
    assert np.all(
        np.asarray(final_out.state.team_a_score) >= np.asarray(out.state.team_a_score)
    )
    assert np.all(
        np.asarray(final_out.state.team_b_score) >= np.asarray(out.state.team_b_score)
    )


def test_team_b_can_take_a_live_possession_before_the_first_handoff():
    static = _multi_possession_static()
    state = reset_batch_minimal(
        static,
        jax.random.split(jax.random.PRNGKey(9), 1),
        jax,
        jnp,
    )
    team_b_holder = np.asarray(static.defense_ids, dtype=np.int32)[0]
    state = state._replace(
        offense_team=jnp.asarray([TEAM_B], dtype=jnp.int8),
        starting_offense_team=jnp.asarray([TEAM_B], dtype=jnp.int8),
        ball_holder=jnp.asarray([team_b_holder], dtype=jnp.int32),
    )
    actions = jnp.full((1, state.positions.shape[1]), ActionType.NOOP.value, dtype=jnp.int32)
    actions = actions.at[0, team_b_holder].set(PASS_ACTION_START)
    out = step_batch_minimal(
        static,
        state,
        actions,
        jax.random.split(jax.random.PRNGKey(10), 1),
        jax,
        jnp,
    )

    assert int(np.asarray(out.completed_pass)[0]) == 1
    assert not bool(np.asarray(out.done)[0])
    assert int(np.asarray(out.state.offense_team)[0]) == TEAM_B
    assert int(np.asarray(out.state.ball_holder)[0]) in set(
        np.asarray(static.defense_ids, dtype=np.int32).tolist()
    )


def test_live_steal_switches_possession_without_a_dead_ball_inbound():
    static = _multi_possession_static()._replace(
        defender_pressure_distance=jnp.asarray(100.0, dtype=jnp.float32),
        defender_pressure_turnover_chance=jnp.asarray(1.0, dtype=jnp.float32),
        defender_pressure_decay_lambda=jnp.asarray(0.0, dtype=jnp.float32),
    )
    state = reset_batch_minimal(
        static,
        jax.random.split(jax.random.PRNGKey(11), 1),
        jax,
        jnp,
    )
    team_b_holder = np.asarray(static.defense_ids, dtype=np.int32)[0]
    state = state._replace(
        offense_team=jnp.asarray([TEAM_B], dtype=jnp.int8),
        ball_holder=jnp.asarray([team_b_holder], dtype=jnp.int32),
    )
    actions = jnp.full((1, state.positions.shape[1]), ActionType.NOOP.value, dtype=jnp.int32)
    out = step_batch_minimal(
        static,
        state,
        actions,
        jax.random.split(jax.random.PRNGKey(12), 1),
        jax,
        jnp,
    )

    assert int(np.asarray(out.turnover)[0]) == 1
    assert int(np.asarray(out.possession_ended)[0]) == 1
    assert not bool(np.asarray(out.done)[0])
    assert int(np.asarray(out.state.offense_team)[0]) == TEAM_A
    assert int(np.asarray(out.state.game_phase)[0]) == GAME_PHASE_LIVE
    assert int(np.asarray(out.state.inbound_team)[0]) == -1
    assert int(np.asarray(out.state.clearance_achieved)[0]) == 0
    assert int(np.asarray(out.state.ball_holder)[0]) in set(
        np.asarray(static.offense_ids, dtype=np.int32).tolist()
    )
