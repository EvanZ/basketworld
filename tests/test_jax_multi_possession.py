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
    POSSESSION_END_DEFENSIVE_VIOLATION,
    POSSESSION_END_MADE_BASKET,
    POSSESSION_END_TURNOVER,
    REBOUND_CONTEST_MODE_LOCAL,
    TEAM_A,
    TEAM_B,
    build_kernel_static_from_env,
    reset_batch_minimal,
    step_batch_minimal,
)


def _multi_possession_static(
    *,
    possession_limit: int = 2,
    illegal_defense_enabled: bool = False,
):
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
        illegal_defense_enabled=illegal_defense_enabled,
        offensive_three_seconds_enabled=False,
        enable_phi_shaping=False,
    )
    env.enable_multi_possession = True
    env.multi_possession_limit = possession_limit
    return build_kernel_static_from_env(env, xp=jnp)


def test_multi_possession_cli_validates_limit_and_disables_start_templates():
    assert parse_args(["--enable-multi-possession"]).multi_possession_limit == 25
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
            parse_args(
                ["--enable-multi-possession", "--start-template-enabled", "true"]
            )
        )


def _shoot(static, state, seed: int):
    batch_size, n_players, _ = state.positions.shape
    actions = jnp.full((batch_size, n_players), ActionType.NOOP.value, dtype=jnp.int32)
    actions = actions.at[jnp.arange(batch_size), state.ball_holder].set(
        ActionType.SHOOT.value
    )
    return step_batch_minimal(
        static,
        state,
        actions,
        jax.random.split(jax.random.PRNGKey(seed), batch_size),
        jax,
        jnp,
    )


def _noops(state):
    return jnp.full(
        (state.positions.shape[0], state.positions.shape[1]),
        ActionType.NOOP.value,
        dtype=jnp.int32,
    )


def _resume_reserved_inbound(static, state):
    """Advance the #19 reserved dead-ball state for lifecycle-only tests.

    #20 replaces this test harness helper with an actual baseline inbound.  It
    deliberately leaves every player where the previous possession ended so
    these tests can verify the #19 identity/score contract independently of
    inbounder placement.
    """
    next_offense_ids = jnp.where(
        state.offense_team[:, None] == TEAM_A,
        static.offense_ids[None, :],
        static.defense_ids[None, :],
    )
    return state._replace(
        game_phase=jnp.full_like(state.game_phase, GAME_PHASE_LIVE),
        ball_holder=next_offense_ids[:, 0],
        inbound_team=jnp.full_like(state.inbound_team, -1),
        inbound_player=jnp.full_like(state.inbound_player, -1),
        inbound_reason=jnp.zeros_like(state.inbound_reason),
    )


def _force_rebound_winner(static, state, winner: int):
    """Use the local-contest table path to make a specific player rebound."""
    positions = np.asarray(state.positions)
    winner_position = positions[0, winner]
    target_idx = int(
        np.flatnonzero(
            np.all(np.asarray(static.cell_coords) == winner_position, axis=1)
        )[0]
    )
    rebound_probs = np.zeros(
        np.asarray(static.rebound_target_probs).shape, dtype=np.float32
    )
    rebound_probs[:, :, target_idx] = 1.0
    return static._replace(
        enable_rebounds=jnp.asarray(1, dtype=jnp.int8),
        rebound_target_probs=jnp.asarray(rebound_probs, dtype=jnp.float32),
        rebound_target_uniform_mix=jnp.asarray(0.0, dtype=jnp.float32),
        rebound_target_temperature=jnp.asarray(1.0, dtype=jnp.float32),
        rebound_contest_mode=jnp.asarray(REBOUND_CONTEST_MODE_LOCAL, dtype=jnp.int32),
        rebound_contest_radius=jnp.asarray(0, dtype=jnp.int32),
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
    repeated = reset_batch_minimal(
        static,
        jax.random.split(jax.random.PRNGKey(17), 4),
        jax,
        jnp,
    )
    holders = np.asarray(state.ball_holder, dtype=np.int32)
    team_a_ids = set(np.asarray(static.offense_ids, dtype=np.int32).tolist())
    team_b_ids = set(np.asarray(static.defense_ids, dtype=np.int32).tolist())

    assert {TEAM_A, TEAM_B}.issubset(set(starters.tolist()))
    np.testing.assert_array_equal(starters, np.asarray(repeated.starting_offense_team))
    starting_team_counts = {
        TEAM_A: int(np.sum(starters == TEAM_A)),
        TEAM_B: int(np.sum(starters == TEAM_B)),
    }
    assert starting_team_counts[TEAM_A] + starting_team_counts[TEAM_B] == len(starters)
    assert all(count > 0 for count in starting_team_counts.values())
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
    assert np.all(
        np.asarray(out.state.game_phase, dtype=np.int8) == GAME_PHASE_AWAITING_INBOUND
    )
    assert np.all(np.asarray(out.state.ball_holder, dtype=np.int32) == -1)
    assert np.all(np.asarray(out.state.inbound_player, dtype=np.int32) == -1)
    assert np.all(np.asarray(out.state.inbound_reason, dtype=np.int32) == reasons)
    assert np.all(np.asarray(out.state.clearance_achieved, dtype=np.int8) == 0)
    np.testing.assert_array_equal(
        np.asarray(out.state.positions), np.asarray(state.positions)
    )
    np.testing.assert_array_equal(
        np.asarray(out.state.layup_pct), np.asarray(state.layup_pct)
    )
    np.testing.assert_array_equal(
        np.asarray(out.state.rebound_skill), np.asarray(state.rebound_skill)
    )
    np.testing.assert_array_equal(
        np.asarray(out.state.offense_team, dtype=np.int8),
        1 - starters,
    )
    np.testing.assert_array_equal(
        np.asarray(out.state.inbound_team, dtype=np.int8),
        1 - starters,
    )

    resumed = _resume_reserved_inbound(static, out.state)
    final_out = _shoot(static, resumed, seed=4)

    assert np.all(np.asarray(final_out.possession_ended, dtype=np.int8) == 1)
    assert np.all(np.asarray(final_out.done, dtype=bool))
    assert np.all(np.asarray(final_out.state.episode_ended, dtype=np.int8) == 1)
    assert np.all(
        np.asarray(final_out.state.completed_possessions, dtype=np.int32) == 2
    )
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
    actions = jnp.full(
        (1, state.positions.shape[1]), ActionType.NOOP.value, dtype=jnp.int32
    )
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
    actions = jnp.full(
        (1, state.positions.shape[1]), ActionType.NOOP.value, dtype=jnp.int32
    )
    compiled_step = jax.jit(
        lambda state_arg, actions_arg, keys_arg: step_batch_minimal(
            static,
            state_arg,
            actions_arg,
            keys_arg,
            jax,
            jnp,
        )
    )
    out = compiled_step(
        state,
        actions,
        jax.random.split(jax.random.PRNGKey(12), 1),
    )

    assert int(np.asarray(out.turnover)[0]) == 1
    assert int(np.asarray(out.possession_ended)[0]) == 1
    assert int(np.asarray(out.possession_end_reason)[0]) == POSSESSION_END_TURNOVER
    assert not bool(np.asarray(out.done)[0])
    assert int(np.asarray(out.state.completed_possessions)[0]) == 1
    assert int(np.asarray(out.state.offense_team)[0]) == TEAM_A
    assert int(np.asarray(out.state.game_phase)[0]) == GAME_PHASE_LIVE
    assert int(np.asarray(out.state.inbound_team)[0]) == -1
    assert int(np.asarray(out.state.clearance_achieved)[0]) == 0
    assert int(np.asarray(out.state.ball_holder)[0]) in set(
        np.asarray(static.offense_ids, dtype=np.int32).tolist()
    )


def test_rebounds_and_dead_ball_offensive_violation_have_distinct_lifecycle_transitions():
    base_static = _multi_possession_static()
    base_state = reset_batch_minimal(
        base_static,
        jax.random.split(jax.random.PRNGKey(21), 1),
        jax,
        jnp,
    )
    team_a_holder = int(np.asarray(base_static.offense_ids)[0])
    team_b_rebounder = int(np.asarray(base_static.defense_ids)[0])
    base_state = base_state._replace(
        offense_team=jnp.asarray([TEAM_A], dtype=jnp.int8),
        starting_offense_team=jnp.asarray([TEAM_A], dtype=jnp.int8),
        ball_holder=jnp.asarray([team_a_holder], dtype=jnp.int32),
        layup_pct=jnp.zeros_like(base_state.layup_pct),
        three_pt_pct=jnp.zeros_like(base_state.three_pt_pct),
        dunk_pct=jnp.zeros_like(base_state.dunk_pct),
    )

    defensive_static = _force_rebound_winner(base_static, base_state, team_b_rebounder)
    defensive_out = _shoot(defensive_static, base_state, seed=22)
    assert int(np.asarray(defensive_out.defensive_rebound)[0]) == 1
    assert int(np.asarray(defensive_out.offensive_rebound)[0]) == 0
    assert int(np.asarray(defensive_out.possession_ended)[0]) == 1
    assert (
        int(np.asarray(defensive_out.possession_end_reason)[0])
        == POSSESSION_END_DEFENSIVE_REBOUND
    )
    assert int(np.asarray(defensive_out.state.completed_possessions)[0]) == 1
    assert int(np.asarray(defensive_out.state.offense_team)[0]) == TEAM_B
    assert int(np.asarray(defensive_out.state.game_phase)[0]) == GAME_PHASE_LIVE
    assert int(np.asarray(defensive_out.state.inbound_team)[0]) == -1
    assert int(np.asarray(defensive_out.state.ball_holder)[0]) == team_b_rebounder

    offensive_static = _force_rebound_winner(base_static, base_state, team_a_holder)
    offensive_out = _shoot(offensive_static, base_state, seed=23)
    assert int(np.asarray(offensive_out.offensive_rebound)[0]) == 1
    assert int(np.asarray(offensive_out.defensive_rebound)[0]) == 0
    assert int(np.asarray(offensive_out.possession_ended)[0]) == 0
    assert int(np.asarray(offensive_out.possession_end_reason)[0]) == 0
    assert int(np.asarray(offensive_out.state.completed_possessions)[0]) == 0
    assert int(np.asarray(offensive_out.state.offense_team)[0]) == TEAM_A
    assert int(np.asarray(offensive_out.state.game_phase)[0]) == GAME_PHASE_LIVE
    assert int(np.asarray(offensive_out.state.ball_holder)[0]) == team_a_holder

    violation_state = base_state._replace(
        shot_clock=jnp.asarray([1], dtype=base_state.shot_clock.dtype),
    )
    violation_out = step_batch_minimal(
        base_static,
        violation_state,
        _noops(violation_state),
        jax.random.split(jax.random.PRNGKey(24), 1),
        jax,
        jnp,
    )
    assert int(np.asarray(violation_out.turnover)[0]) == 1
    assert int(np.asarray(violation_out.possession_ended)[0]) == 1
    assert (
        int(np.asarray(violation_out.possession_end_reason)[0])
        == POSSESSION_END_TURNOVER
    )
    assert int(np.asarray(violation_out.state.completed_possessions)[0]) == 1
    assert int(np.asarray(violation_out.state.offense_team)[0]) == TEAM_B
    assert (
        int(np.asarray(violation_out.state.game_phase)[0])
        == GAME_PHASE_AWAITING_INBOUND
    )
    assert int(np.asarray(violation_out.state.inbound_team)[0]) == TEAM_B
    assert int(np.asarray(violation_out.state.ball_holder)[0]) == -1


def test_repeated_role_switches_preserve_fixed_teams_attributes_scores_and_final_boundary():
    static = _multi_possession_static(possession_limit=3)
    state = reset_batch_minimal(
        static,
        jax.random.split(jax.random.PRNGKey(25), 1),
        jax,
        jnp,
    )
    team_a_holder = int(np.asarray(static.offense_ids)[0])
    state = state._replace(
        offense_team=jnp.asarray([TEAM_A], dtype=jnp.int8),
        starting_offense_team=jnp.asarray([TEAM_A], dtype=jnp.int8),
        ball_holder=jnp.asarray([team_a_holder], dtype=jnp.int32),
        layup_pct=jnp.ones_like(state.layup_pct),
        three_pt_pct=jnp.ones_like(state.three_pt_pct),
        dunk_pct=jnp.ones_like(state.dunk_pct),
    )
    initial_positions = np.asarray(state.positions).copy()
    initial_layup = np.asarray(state.layup_pct).copy()
    initial_three = np.asarray(state.three_pt_pct).copy()
    initial_dunk = np.asarray(state.dunk_pct).copy()
    initial_rebound = np.asarray(state.rebound_skill).copy()
    stable_roster = sorted(
        np.asarray(static.offense_ids, dtype=np.int32).tolist()
        + np.asarray(static.defense_ids, dtype=np.int32).tolist()
    )
    assert stable_roster == list(range(state.positions.shape[1]))

    expected_team_a_score = 0.0
    expected_team_b_score = 0.0
    for possession_number, seed in enumerate((26, 27, 28), start=1):
        prior_offense_team = int(np.asarray(state.offense_team)[0])
        out = _shoot(static, state, seed=seed)
        scored = float(np.asarray(out.shot_value)[0]) * float(
            np.asarray(out.shot_success)[0]
        )
        if prior_offense_team == TEAM_A:
            expected_team_a_score += scored
        else:
            expected_team_b_score += scored

        assert int(np.asarray(out.possession_ended)[0]) == 1
        assert int(np.asarray(out.state.completed_possessions)[0]) == possession_number
        assert int(np.asarray(out.state.offense_team)[0]) == 1 - prior_offense_team
        np.testing.assert_array_equal(
            np.asarray(out.state.positions), initial_positions
        )
        np.testing.assert_array_equal(np.asarray(out.state.layup_pct), initial_layup)
        np.testing.assert_array_equal(np.asarray(out.state.three_pt_pct), initial_three)
        np.testing.assert_array_equal(np.asarray(out.state.dunk_pct), initial_dunk)
        np.testing.assert_array_equal(
            np.asarray(out.state.rebound_skill), initial_rebound
        )
        assert float(np.asarray(out.state.team_a_score)[0]) == pytest.approx(
            expected_team_a_score
        )
        assert float(np.asarray(out.state.team_b_score)[0]) == pytest.approx(
            expected_team_b_score
        )

        if possession_number < 3:
            assert not bool(np.asarray(out.done)[0])
            assert (
                int(np.asarray(out.state.game_phase)[0]) == GAME_PHASE_AWAITING_INBOUND
            )
            state = _resume_reserved_inbound(static, out.state)
        else:
            assert bool(np.asarray(out.done)[0])
            assert int(np.asarray(out.state.episode_ended)[0]) == 1
            # The final score resolves, but a completed game must not create a
            # fresh inbound state for a non-existent next possession.
            assert int(np.asarray(out.state.game_phase)[0]) == GAME_PHASE_LIVE
            assert int(np.asarray(out.state.inbound_team)[0]) == -1
            assert int(np.asarray(out.state.inbound_player)[0]) == -1


def test_defensive_lane_violation_awards_same_offense_one_point_and_dead_ball_restart():
    static = _multi_possession_static(illegal_defense_enabled=True)._replace(
        illegal_defense_enabled=jnp.asarray(1, dtype=jnp.int8),
        defender_guard_distance=jnp.asarray(0.0, dtype=jnp.float32),
        three_second_max_steps=jnp.asarray(0, dtype=jnp.int32),
    )
    state = reset_batch_minimal(
        static,
        jax.random.split(jax.random.PRNGKey(29), 1),
        jax,
        jnp,
    )
    team_b_holder = int(np.asarray(static.defense_ids)[0])
    team_a_defender = int(np.asarray(static.offense_ids)[0])
    lane_idx = int(np.flatnonzero(np.asarray(static.defensive_lane_by_cell))[0])
    positions = np.asarray(state.positions).copy()
    positions[0, team_a_defender] = np.asarray(static.cell_coords)[lane_idx]
    state = state._replace(
        positions=jnp.asarray(positions, dtype=jnp.int32),
        offense_team=jnp.asarray([TEAM_B], dtype=jnp.int8),
        starting_offense_team=jnp.asarray([TEAM_B], dtype=jnp.int8),
        ball_holder=jnp.asarray([team_b_holder], dtype=jnp.int32),
    )

    out = step_batch_minimal(
        static,
        state,
        _noops(state),
        jax.random.split(jax.random.PRNGKey(30), 1),
        jax,
        jnp,
    )

    assert int(np.asarray(out.defensive_lane_violation)[0]) == 1
    assert int(np.asarray(out.turnover)[0]) == 0
    assert int(np.asarray(out.possession_ended)[0]) == 1
    assert (
        int(np.asarray(out.possession_end_reason)[0])
        == POSSESSION_END_DEFENSIVE_VIOLATION
    )
    assert int(np.asarray(out.state.completed_possessions)[0]) == 1
    assert int(np.asarray(out.state.offense_team)[0]) == TEAM_B
    assert float(np.asarray(out.state.team_a_score)[0]) == pytest.approx(0.0)
    assert float(np.asarray(out.state.team_b_score)[0]) == pytest.approx(1.0)
    assert int(np.asarray(out.state.game_phase)[0]) == GAME_PHASE_AWAITING_INBOUND
    assert int(np.asarray(out.state.inbound_team)[0]) == TEAM_B
    assert (
        int(np.asarray(out.state.inbound_reason)[0])
        == POSSESSION_END_DEFENSIVE_VIOLATION
    )


def test_jit_vmap_mixed_phases_both_starters_final_score_and_legacy_mode():
    static = _multi_possession_static(possession_limit=1)
    state = reset_batch_minimal(
        static,
        jax.random.split(jax.random.PRNGKey(31), 3),
        jax,
        jnp,
    )
    team_a_holder = int(np.asarray(static.offense_ids)[0])
    team_b_holder = int(np.asarray(static.defense_ids)[0])
    state = state._replace(
        offense_team=jnp.asarray([TEAM_A, TEAM_B, TEAM_A], dtype=jnp.int8),
        starting_offense_team=jnp.asarray([TEAM_A, TEAM_B, TEAM_A], dtype=jnp.int8),
        ball_holder=jnp.asarray([team_a_holder, team_b_holder, -1], dtype=jnp.int32),
        game_phase=jnp.asarray(
            [GAME_PHASE_LIVE, GAME_PHASE_LIVE, GAME_PHASE_AWAITING_INBOUND],
            dtype=jnp.int8,
        ),
        inbound_team=jnp.asarray([-1, -1, TEAM_A], dtype=jnp.int8),
        layup_pct=jnp.ones_like(state.layup_pct),
        three_pt_pct=jnp.ones_like(state.three_pt_pct),
        dunk_pct=jnp.ones_like(state.dunk_pct),
    )
    actions = _noops(state)
    actions = actions.at[0, team_a_holder].set(ActionType.SHOOT.value)
    actions = actions.at[1, team_b_holder].set(ActionType.SHOOT.value)
    compiled_step = jax.jit(
        lambda state_arg, actions_arg, keys_arg: step_batch_minimal(
            static,
            state_arg,
            actions_arg,
            keys_arg,
            jax,
            jnp,
        )
    )
    out = compiled_step(
        state,
        actions,
        jax.random.split(jax.random.PRNGKey(32), 3),
    )

    assert np.asarray(out.possession_ended, dtype=np.int8).tolist() == [1, 1, 0]
    assert np.asarray(out.done, dtype=bool).tolist() == [True, True, False]
    assert np.asarray(out.state.completed_possessions, dtype=np.int32).tolist() == [
        1,
        1,
        0,
    ]
    assert np.asarray(out.state.starting_offense_team, dtype=np.int8).tolist() == [
        TEAM_A,
        TEAM_B,
        TEAM_A,
    ]
    assert int(np.asarray(out.state.inbound_team)[0]) == -1
    assert int(np.asarray(out.state.inbound_team)[1]) == -1
    assert int(np.asarray(out.state.inbound_team)[2]) == TEAM_A
    assert float(np.asarray(out.state.team_a_score)[0]) == pytest.approx(
        float(np.asarray(out.shot_value)[0]) * float(np.asarray(out.shot_success)[0])
    )
    assert float(np.asarray(out.state.team_b_score)[1]) == pytest.approx(
        float(np.asarray(out.shot_value)[1]) * float(np.asarray(out.shot_success)[1])
    )

    legacy_static = static._replace(
        enable_multi_possession=jnp.asarray(0, dtype=jnp.int8)
    )
    legacy_state = reset_batch_minimal(
        legacy_static,
        jax.random.split(jax.random.PRNGKey(33), 1),
        jax,
        jnp,
    )._replace(
        layup_pct=jnp.ones((1, state.positions.shape[1]), dtype=jnp.float32),
        three_pt_pct=jnp.ones((1, state.positions.shape[1]), dtype=jnp.float32),
        dunk_pct=jnp.ones((1, state.positions.shape[1]), dtype=jnp.float32),
    )
    legacy_out = _shoot(legacy_static, legacy_state, seed=34)
    assert bool(np.asarray(legacy_out.done)[0])
    assert int(np.asarray(legacy_out.state.completed_possessions)[0]) == 0
    assert int(np.asarray(legacy_out.state.offense_team)[0]) == TEAM_A
    assert int(np.asarray(legacy_out.state.game_phase)[0]) == GAME_PHASE_LIVE
    assert int(np.asarray(legacy_out.state.inbound_team)[0]) == -1
