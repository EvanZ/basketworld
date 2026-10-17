from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from basketworld.domain import ActionType
from basketworld_jax.checkpoints import (
    build_checkpoint_payload,
    load_checkpoint,
    save_checkpoint,
)
from basketworld_jax.config import JaxEnvironmentConfig
from basketworld_jax.env.minimal import (
    EPISODE_END_GAME_CLOCK,
    EPISODE_END_POSSESSIONS,
    GAME_PHASE_AWAITING_CHECK,
    GAME_PHASE_AWAITING_INBOUND,
    GAME_PHASE_CHECK_SETUP,
    GAME_PHASE_LIVE,
    PASS_ACTION_START,
    POSSESSION_END_MADE_BASKET,
    POSSESSION_END_INBOUND_VIOLATION,
    POSSESSION_END_PERIOD_EXPIRED,
    TEAM_A,
    TEAM_B,
    TRUNCATION_REASON_TOTAL_STEPS,
    TURNOVER_REASON_INBOUND_TIMEOUT,
    TURNOVER_REASON_NONE,
    build_action_masks_batch,
    build_kernel_static_from_config,
    build_multi_possession_observation_features_batch,
    reset_batch_minimal,
    step_batch_minimal,
)
from basketworld_jax.train.main import _restore_like_template


def _static(
    *,
    players: int = 1,
    mode: str = "game_clock",
    regulation: int = 5,
    overtime: int = 2,
    max_steps: int = 10_000,
    winner_reward: float = 0.0,
):
    config = JaxEnvironmentConfig(
        players=players,
        episode_end_mode=mode,
        game_clock_seconds=regulation,
        overtime_seconds=overtime,
        game_clock_max_total_steps=max_steps,
        enable_multi_possession=True,
        multi_possession_limit=2,
        game_winner_reward=winner_reward,
        check_deadline_steps=5,
        check_setup_steps=2,
        defender_pressure_turnover_chance=0.0,
        base_steal_rate=0.0,
        illegal_defense_enabled=False,
        offensive_three_seconds_enabled=False,
        shot_pressure_enabled=False,
        layup_pct=1.0,
        three_pt_pct=1.0,
        allow_dunks=False,
    )
    return build_kernel_static_from_config(config, jnp)


def _reset(static, seed: int = 1):
    return reset_batch_minimal(
        static,
        jax.random.split(jax.random.PRNGKey(seed), 1),
        jax,
        jnp,
    )


def _step(static, state, *, seed: int = 2, actions=None):
    if actions is None:
        actions = jnp.full(
            (1, state.positions.shape[1]),
            ActionType.NOOP.value,
            dtype=jnp.int32,
        )
    return step_batch_minimal(
        static,
        state,
        actions,
        jax.random.split(jax.random.PRNGKey(seed), 1),
        jax,
        jnp,
    )


def _phase_positions(static, state):
    cells = np.asarray(static.cell_coords, dtype=np.int32)
    check = np.asarray(static.check_position, dtype=np.int32)
    selected = [cell for cell in cells if not np.array_equal(cell, check)][:2]
    return state._replace(
        positions=jnp.asarray(np.asarray(selected)[None, :, :], dtype=jnp.int32)
    )


def _inbound_state(static, state, *, countdown: int = 5):
    active_team = int(np.asarray(state.offense_team)[0])
    active_ids = np.asarray(
        static.offense_ids if active_team == TEAM_A else static.defense_ids,
        dtype=np.int32,
    )
    opponent_ids = np.asarray(
        static.defense_ids if active_team == TEAM_A else static.offense_ids,
        dtype=np.int32,
    )
    inbounder = int(active_ids[0])
    positions = np.asarray(state.positions).copy()
    positions[0, inbounder] = np.asarray(static.inbound_position)
    positions[0, int(active_ids[1])] = np.asarray([0, 8], dtype=np.int32)
    for offset, player_id in enumerate(opponent_ids.tolist()):
        positions[0, player_id] = np.asarray([5 + offset, 5], dtype=np.int32)
    return (
        state._replace(
            positions=jnp.asarray(positions, dtype=jnp.int32),
            game_clock_remaining=jnp.asarray([8], dtype=jnp.int32),
            shot_clock=jnp.asarray([22], dtype=jnp.int32),
            game_phase=jnp.asarray([GAME_PHASE_AWAITING_INBOUND], dtype=jnp.int8),
            inbound_team=jnp.asarray([active_team], dtype=jnp.int8),
            inbound_player=jnp.asarray([inbounder], dtype=jnp.int32),
            inbound_steps_remaining=jnp.asarray([countdown], dtype=jnp.int32),
            ball_holder=jnp.asarray([inbounder], dtype=jnp.int32),
        ),
        inbounder,
    )


def test_game_clock_configuration_defaults_validation_and_possession_mode_parity():
    defaults = JaxEnvironmentConfig(players=1)
    assert defaults.episode_end_mode == "possessions"
    assert defaults.game_clock_seconds == 300
    assert defaults.overtime_seconds == 120
    assert defaults.game_clock_max_total_steps == 10_000

    with pytest.raises(ValueError, match="episode_end_mode"):
        JaxEnvironmentConfig(players=1, episode_end_mode="quarters")
    with pytest.raises(ValueError, match="game_clock_seconds"):
        JaxEnvironmentConfig(players=1, game_clock_seconds=0)

    with pytest.raises(ValueError, match="requires enable_multi_possession"):
        JaxEnvironmentConfig(players=1, episode_end_mode="game_clock")

    static = _static(mode="possessions", regulation=17, overtime=9)
    assert int(np.asarray(static.episode_end_mode)) == EPISODE_END_POSSESSIONS
    state = _reset(static)._replace(
        game_clock_remaining=jnp.asarray([17], dtype=jnp.int32)
    )
    output = _step(static, state)
    assert int(np.asarray(output.state.game_clock_remaining)[0]) == 17
    assert int(np.asarray(output.state.shot_clock)[0]) == int(
        np.asarray(state.shot_clock)[0]
    ) - 1
    assert int(np.asarray(output.state.elapsed_game_clock_seconds)[0]) == 0

    clock_static = _static()
    starters = {
        int(np.asarray(_reset(clock_static, seed=seed).offense_team)[0])
        for seed in range(8)
    }
    assert starters == {TEAM_A, TEAM_B}


def test_clock_observation_uses_fixed_unclipped_time_and_score_scales():
    static = _static(regulation=900)
    state = _reset(static)._replace(
        offense_team=jnp.asarray([TEAM_A], dtype=jnp.int8),
        game_clock_remaining=jnp.asarray([900], dtype=jnp.int32),
        team_a_score=jnp.asarray([50.0], dtype=jnp.float32),
        team_b_score=jnp.asarray([5.0], dtype=jnp.float32),
    )
    _, globals_vec = build_multi_possession_observation_features_batch(
        static,
        state,
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp,
    )
    assert float(np.asarray(globals_vec)[0, 0]) == pytest.approx(1.5)
    assert float(np.asarray(globals_vec)[0, 1]) == pytest.approx(1.25)

    possession_static = _static(mode="possessions")
    possession_state = _reset(possession_static)._replace(
        offense_team=jnp.asarray([TEAM_A], dtype=jnp.int8),
        team_a_score=jnp.asarray([3.0], dtype=jnp.float32),
        team_b_score=jnp.asarray([0.0], dtype=jnp.float32),
        team_a_completed_possessions=jnp.asarray([1], dtype=jnp.int32),
        episode_possession_limit=jnp.asarray([2], dtype=jnp.int32),
    )
    _, possession_globals = build_multi_possession_observation_features_batch(
        possession_static,
        possession_state,
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp,
    )
    assert float(np.asarray(possession_globals)[0, 0]) == pytest.approx(0.5)
    assert float(np.asarray(possession_globals)[0, 1]) == pytest.approx(0.5)


def test_mid_game_clock_state_survives_checkpoint_round_trip(tmp_path):
    static = _static(regulation=10)
    live = _step(static, _reset(static, seed=51), seed=52).state
    live = live._replace(
        overtime_round=jnp.asarray([2], dtype=jnp.int32),
        game_clock_remaining=jnp.asarray([7], dtype=jnp.int32),
    )
    payload = build_checkpoint_payload(
        update_index=4,
        trainer_config={},
        policy_spec={},
        frozen_config={},
        env_config={
            "enable_multi_possession": True,
            "episode_end_mode": "game_clock",
            "game_clock_seconds": 10,
            "overtime_seconds": 2,
        },
        params={},
        opt_state={},
        current_state={"offense": live},
        eval_initial_state={},
        base_key=jax.random.PRNGKey(53),
        eval_trajectories=[],
        last_metrics=None,
        observation_metadata={
            "episode_end_mode": "game_clock",
            "episode_budget_feature": {
                "semantic": "game_clock_seconds_remaining",
                "normalization": 720.0,
            },
            "relative_score_feature": {
                "semantic": "observing_team_score_difference",
                "normalization": 30.0,
            },
        },
    )
    checkpoint = tmp_path / "clock-checkpoint"
    save_checkpoint(checkpoint, payload)
    restored = _restore_like_template(
        load_checkpoint(checkpoint)["current_state"]["offense"],
        live,
    )

    np.testing.assert_array_equal(
        np.asarray(restored.game_clock_remaining), np.asarray([7], dtype=np.int32)
    )
    np.testing.assert_array_equal(
        np.asarray(restored.elapsed_game_clock_seconds),
        np.asarray(live.elapsed_game_clock_seconds),
    )
    np.testing.assert_array_equal(
        np.asarray(restored.overtime_round), np.asarray([2], dtype=np.int32)
    )
    np.testing.assert_array_equal(
        np.asarray(restored.episode_end_mode),
        np.asarray([EPISODE_END_GAME_CLOCK], dtype=np.int8),
    )


def test_clock_runs_live_and_during_pickup_but_stops_for_setup_and_inbound():
    static = _static(regulation=10)
    state = _reset(static)
    assert int(np.asarray(state.episode_end_mode)[0]) == EPISODE_END_GAME_CLOCK
    assert int(np.asarray(state.game_clock_remaining)[0]) == 10

    live = _step(static, state)
    assert int(np.asarray(live.state.game_clock_remaining)[0]) == 9
    assert int(np.asarray(live.state.shot_clock)[0]) == 23
    assert int(np.asarray(live.state.elapsed_game_clock_seconds)[0]) == 1
    assert int(np.asarray(live.state.possession_game_clock_seconds)[0]) == 1

    setup_state = _phase_positions(static, live.state)._replace(
        game_phase=jnp.asarray([GAME_PHASE_CHECK_SETUP], dtype=jnp.int8),
        ball_holder=jnp.asarray([-1], dtype=jnp.int32),
        check_team=live.state.offense_team,
        check_setup_steps_remaining=jnp.asarray([1], dtype=jnp.int32),
        check_steps_remaining=jnp.asarray([0], dtype=jnp.int32),
    )
    setup = _step(static, setup_state, seed=3)
    assert int(np.asarray(setup.state.game_clock_remaining)[0]) == 9
    assert int(np.asarray(setup.state.shot_clock)[0]) == 23
    assert int(np.asarray(setup.state.check_setup_steps_remaining)[0]) == 0
    assert int(np.asarray(setup.state.game_phase)[0]) == GAME_PHASE_AWAITING_CHECK
    assert int(np.asarray(setup.state.check_steps_remaining)[0]) == 5

    pickup_state = _phase_positions(static, setup.state)
    pickup = _step(static, pickup_state, seed=4)
    assert int(np.asarray(pickup.state.game_clock_remaining)[0]) == 8
    assert int(np.asarray(pickup.state.shot_clock)[0]) == 22
    assert int(np.asarray(pickup.state.possession_game_clock_seconds)[0]) == 2

    pickup_positions = np.asarray(pickup_state.positions).copy()
    active_team = int(np.asarray(pickup_state.offense_team)[0])
    active_ids = np.asarray(
        static.offense_ids if active_team == TEAM_A else static.defense_ids,
        dtype=np.int32,
    )
    pickup_positions[0, int(active_ids[0])] = np.asarray(static.check_position)
    successful_pickup = _step(
        static,
        pickup_state._replace(
            positions=jnp.asarray(pickup_positions, dtype=jnp.int32)
        ),
        seed=5,
    )
    assert int(np.asarray(successful_pickup.check_pickup)[0]) == 1
    assert int(np.asarray(successful_pickup.state.game_phase)[0]) == GAME_PHASE_LIVE
    assert int(np.asarray(successful_pickup.state.game_clock_remaining)[0]) == 8
    assert int(np.asarray(successful_pickup.state.shot_clock)[0]) == 22

    inbound_state = pickup.state._replace(
        game_phase=jnp.asarray([GAME_PHASE_AWAITING_INBOUND], dtype=jnp.int8),
        inbound_team=pickup.state.offense_team,
        inbound_player=pickup.state.ball_holder,
        inbound_steps_remaining=jnp.asarray([5], dtype=jnp.int32),
        ball_holder=jnp.asarray([0], dtype=jnp.int32),
    )
    inbound = _step(static, inbound_state, seed=6)
    assert int(np.asarray(inbound.state.game_clock_remaining)[0]) == 8
    assert int(np.asarray(inbound.state.shot_clock)[0]) == 22
    assert int(np.asarray(inbound.state.inbound_steps_remaining)[0]) == 4


def test_inbound_catch_keeps_both_clocks_frozen_until_the_following_live_step():
    static = _static(players=2, regulation=10)
    state, inbounder = _inbound_state(static, _reset(static, seed=7))
    masks = np.asarray(build_action_masks_batch(static, state, jnp), dtype=np.int8)
    legal_slots = np.flatnonzero(masks[0, inbounder, PASS_ACTION_START:])
    assert legal_slots.size > 0
    actions = jnp.full(
        (1, state.positions.shape[1]), ActionType.NOOP.value, dtype=jnp.int32
    ).at[0, inbounder].set(PASS_ACTION_START + int(legal_slots[0]))

    caught = _step(static, state, seed=8, actions=actions)
    assert int(np.asarray(caught.completed_pass)[0]) == 1
    assert int(np.asarray(caught.state.game_phase)[0]) == GAME_PHASE_LIVE
    assert int(np.asarray(caught.state.game_clock_remaining)[0]) == 8
    assert int(np.asarray(caught.state.shot_clock)[0]) == 22

    live = _step(static, caught.state, seed=9)
    assert int(np.asarray(live.state.game_clock_remaining)[0]) == 7
    assert int(np.asarray(live.state.shot_clock)[0]) == 21


def test_inbound_deadline_runs_while_clocks_stop_then_restarts_the_next_inbound():
    static = _static(players=2, regulation=10)
    state, _ = _inbound_state(static, _reset(static, seed=12), countdown=2)
    initial_team = int(np.asarray(state.offense_team)[0])

    first = _step(static, state, seed=13)
    assert int(np.asarray(first.state.inbound_steps_remaining)[0]) == 1
    assert int(np.asarray(first.state.game_clock_remaining)[0]) == 8
    assert int(np.asarray(first.state.shot_clock)[0]) == 22

    expired = _step(static, first.state, seed=14)
    assert int(np.asarray(expired.turnover_reason)[0]) == TURNOVER_REASON_INBOUND_TIMEOUT
    assert int(np.asarray(expired.possession_end_reason)[0]) == POSSESSION_END_INBOUND_VIOLATION
    assert int(np.asarray(expired.state.completed_possessions)[0]) == 1
    assert int(np.asarray(expired.state.game_phase)[0]) == GAME_PHASE_AWAITING_INBOUND
    assert int(np.asarray(expired.state.inbound_team)[0]) == 1 - initial_team
    assert int(np.asarray(expired.state.inbound_steps_remaining)[0]) == 5
    assert int(np.asarray(expired.state.game_clock_remaining)[0]) == 8
    assert int(np.asarray(expired.state.shot_clock)[0]) == 24


def test_period_expiration_starts_repeatable_paired_overtime_and_counts_possession():
    static = _static(regulation=1, overtime=2)
    base_state = _reset(static, seed=10)
    state = base_state._replace(
        team_a_score=jnp.asarray([7.0], dtype=jnp.float32),
        team_b_score=jnp.asarray([7.0], dtype=jnp.float32),
        completed_possessions=jnp.asarray([5], dtype=jnp.int32),
        team_a_completed_possessions=jnp.asarray([3], dtype=jnp.int32),
        team_b_completed_possessions=jnp.asarray([2], dtype=jnp.int32),
        episode_overtime_round_cap=jnp.asarray([1], dtype=jnp.int32),
        pressure_exposure=jnp.asarray([3.0], dtype=jnp.float32),
        offense_lane_steps=jnp.ones_like(base_state.offense_lane_steps),
        defense_lane_steps=jnp.ones_like(base_state.defense_lane_steps),
        assist_active=jnp.asarray([1], dtype=jnp.int8),
        assist_passer=jnp.asarray([0], dtype=jnp.int32),
        assist_recipient=jnp.asarray([1], dtype=jnp.int32),
    )
    starting_team = int(np.asarray(state.offense_team)[0])
    partner = state._replace(
        overtime_pair_inversion=jnp.asarray([1], dtype=jnp.int8),
        step_count=jnp.asarray([777], dtype=jnp.int32),
    )
    first = _step(static, state, seed=11)
    inverted = _step(static, partner, seed=11)

    assert not bool(np.asarray(first.done)[0])
    assert int(np.asarray(first.state.overtime_round)[0]) == 1
    assert int(np.asarray(first.state.game_clock_remaining)[0]) == 2
    assert int(np.asarray(first.state.completed_possessions)[0]) == 6
    assert int(np.asarray(first.state.team_a_completed_possessions)[0]) == 3 + (
        starting_team == TEAM_A
    )
    assert int(np.asarray(first.state.team_b_completed_possessions)[0]) == 2 + (
        starting_team == TEAM_B
    )
    assert int(np.asarray(first.possession_end_reason)[0]) == POSSESSION_END_PERIOD_EXPIRED
    assert int(np.asarray(first.completed_possession_game_clock_seconds)[0]) == 1
    assert float(np.asarray(first.state.team_a_score)[0]) == 7.0
    assert float(np.asarray(first.state.team_b_score)[0]) == 7.0
    assert int(np.asarray(first.state.elapsed_game_clock_seconds)[0]) == 1
    assert int(np.asarray(first.state.period_elapsed_game_clock_seconds)[0]) == 0
    assert int(np.asarray(first.state.shot_clock)[0]) == 24
    assert int(np.asarray(first.state.game_phase)[0]) == GAME_PHASE_LIVE
    assert int(np.asarray(first.state.inbound_player)[0]) == -1
    assert int(np.asarray(first.state.assist_active)[0]) == 0
    assert float(np.asarray(first.state.pressure_exposure)[0]) == 0.0
    assert np.all(np.asarray(first.state.offense_lane_steps) == 0.0)
    assert np.all(np.asarray(first.state.defense_lane_steps) == 0.0)
    np.testing.assert_array_equal(first.state.layup_pct, state.layup_pct)
    np.testing.assert_array_equal(first.state.pressure_skill, state.pressure_skill)
    assert int(np.asarray(first.state.offense_team)[0]) == 1 - int(
        np.asarray(inverted.state.offense_team)[0]
    )
    np.testing.assert_array_equal(
        np.asarray(first.state.positions), np.asarray(inverted.state.positions)
    )

    overtime_holder = int(np.asarray(first.state.ball_holder)[0])
    scoring_positions = np.asarray(first.state.positions).copy()
    scoring_positions[0, overtime_holder] = np.asarray(static.basket_position)
    scoring_state = first.state._replace(
        positions=jnp.asarray(scoring_positions, dtype=jnp.int32),
        clearance_achieved=jnp.asarray([1], dtype=jnp.int8),
        layup_pct=jnp.ones_like(first.state.layup_pct),
    )
    scoring_actions = jnp.full(
        (1, scoring_state.positions.shape[1]),
        ActionType.NOOP.value,
        dtype=jnp.int32,
    ).at[0, overtime_holder].set(ActionType.SHOOT.value)
    overtime_score = _step(
        static, scoring_state, seed=12, actions=scoring_actions
    )
    assert not bool(np.asarray(overtime_score.done)[0])
    assert int(np.asarray(overtime_score.state.overtime_round)[0]) == 1
    assert int(np.asarray(overtime_score.state.game_clock_remaining)[0]) == 1

    repeated = _step(
        static,
        first.state._replace(
            game_clock_remaining=jnp.asarray([1], dtype=jnp.int32)
        ),
        seed=99,
    )
    assert not bool(np.asarray(repeated.done)[0])
    assert int(np.asarray(repeated.state.overtime_round)[0]) == 2
    assert int(np.asarray(repeated.state.game_clock_remaining)[0]) == 2
    assert int(np.asarray(repeated.state.elapsed_game_clock_seconds)[0]) == 2


def test_final_tick_score_resolves_once_and_untied_finish_pays_winner_reward():
    static = _static(regulation=1, winner_reward=5.0)
    state = _reset(static, seed=20)
    holder = int(np.asarray(state.ball_holder)[0])
    positions = np.asarray(state.positions).copy()
    positions[0, holder] = np.asarray(static.basket_position)
    state = state._replace(
        positions=jnp.asarray(positions, dtype=jnp.int32),
        game_clock_remaining=jnp.asarray([1], dtype=jnp.int32),
        clearance_achieved=jnp.asarray([1], dtype=jnp.int8),
        layup_pct=jnp.ones_like(state.layup_pct),
    )
    actions = jnp.full(
        (1, state.positions.shape[1]), ActionType.NOOP.value, dtype=jnp.int32
    ).at[0, holder].set(ActionType.SHOOT.value)
    output = _step(static, state, seed=21, actions=actions)

    assert bool(np.asarray(output.done)[0])
    assert int(np.asarray(output.possession_end_reason)[0]) == POSSESSION_END_MADE_BASKET
    assert int(np.asarray(output.state.completed_possessions)[0]) == 1
    assert int(
        np.asarray(
            output.state.team_a_completed_possessions
            + output.state.team_b_completed_possessions
        )[0]
    ) == 1
    assert int(np.asarray(output.completed_possession_game_clock_seconds)[0]) == 1
    assert int(np.asarray(output.state.game_clock_remaining)[0]) == 0
    assert float(np.asarray(output.state.team_a_score + output.state.team_b_score)[0]) == 2.0
    winning_team = int(np.asarray(state.offense_team)[0])
    winning_ids = np.asarray(
        static.offense_ids if winning_team == TEAM_A else static.defense_ids,
        dtype=np.int32,
    )
    losing_ids = np.asarray(
        static.defense_ids if winning_team == TEAM_A else static.offense_ids,
        dtype=np.int32,
    )
    assert float(np.asarray(output.winner_reward)[0, winning_ids].sum()) == pytest.approx(
        5.0
    )
    assert float(np.asarray(output.winner_reward)[0, losing_ids].sum()) == pytest.approx(
        -5.0
    )

    repeated = _step(static, output.state, seed=22)
    assert bool(np.asarray(repeated.done)[0])
    np.testing.assert_array_equal(np.asarray(repeated.winner_reward), 0.0)
    assert int(np.asarray(repeated.state.completed_possessions)[0]) == 1


def test_period_expiration_supersedes_shot_clock_and_check_deadline_violations():
    static = _static(regulation=1, overtime=2)
    state = _reset(static, seed=30)._replace(
        shot_clock=jnp.asarray([1], dtype=jnp.int32),
        game_clock_remaining=jnp.asarray([1], dtype=jnp.int32),
    )
    shot_clock = _step(static, state, seed=31)
    assert not bool(np.asarray(shot_clock.turnover)[0])
    assert int(np.asarray(shot_clock.turnover_reason)[0]) == TURNOVER_REASON_NONE
    assert int(np.asarray(shot_clock.possession_end_reason)[0]) == POSSESSION_END_PERIOD_EXPIRED
    assert int(np.asarray(shot_clock.state.overtime_round)[0]) == 1

    check_state = _phase_positions(static, _reset(static, seed=32))._replace(
        game_phase=jnp.asarray([GAME_PHASE_AWAITING_CHECK], dtype=jnp.int8),
        ball_holder=jnp.asarray([-1], dtype=jnp.int32),
        check_team=jnp.asarray([TEAM_A], dtype=jnp.int8),
        offense_team=jnp.asarray([TEAM_A], dtype=jnp.int8),
        check_steps_remaining=jnp.asarray([1], dtype=jnp.int32),
        game_clock_remaining=jnp.asarray([1], dtype=jnp.int32),
        team_a_score=jnp.asarray([0.0], dtype=jnp.float32),
        team_b_score=jnp.asarray([0.0], dtype=jnp.float32),
    )
    check = _step(static, check_state, seed=33)
    assert int(np.asarray(check.state.overtime_round)[0]) == 1
    assert int(np.asarray(check.possession_end_reason)[0]) == POSSESSION_END_PERIOD_EXPIRED
    assert int(np.asarray(check.state.completed_possessions)[0]) == 1
    assert float(np.asarray(check.state.team_a_score)[0]) == 0.0
    assert float(np.asarray(check.state.team_b_score)[0]) == 0.0
    assert not bool(np.asarray(check.check_violation)[0])
    assert int(np.asarray(check.turnover_reason)[0]) == TURNOVER_REASON_NONE


def test_total_step_ceiling_truncates_stopped_clock_without_winner_reward():
    static = _static(regulation=10, max_steps=2, winner_reward=5.0)
    state = _reset(static, seed=40)._replace(
        step_count=jnp.asarray([1], dtype=jnp.int32),
        game_phase=jnp.asarray([GAME_PHASE_AWAITING_INBOUND], dtype=jnp.int8),
        inbound_team=jnp.asarray([TEAM_A], dtype=jnp.int8),
        inbound_player=jnp.asarray([0], dtype=jnp.int32),
        ball_holder=jnp.asarray([0], dtype=jnp.int32),
        inbound_steps_remaining=jnp.asarray([5], dtype=jnp.int32),
        team_a_score=jnp.asarray([3.0], dtype=jnp.float32),
        team_b_score=jnp.asarray([0.0], dtype=jnp.float32),
    )
    output = _step(static, state, seed=41)
    assert bool(np.asarray(output.done)[0])
    assert bool(np.asarray(output.truncated)[0])
    assert bool(np.asarray(output.state.episode_truncated)[0])
    assert int(np.asarray(output.truncation_reason)[0]) == TRUNCATION_REASON_TOTAL_STEPS
    assert int(np.asarray(output.state.game_clock_remaining)[0]) == 10
    assert int(np.asarray(output.state.inbound_steps_remaining)[0]) == 4
    np.testing.assert_array_equal(np.asarray(output.winner_reward), 0.0)
