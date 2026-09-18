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
    MOVE_ACTION_START,
    PASS_ACTION_START,
    PASS_ACTION_END,
    POSSESSION_END_INBOUND_VIOLATION,
    POSSESSION_END_DEFENSIVE_REBOUND,
    POSSESSION_END_DEFENSIVE_VIOLATION,
    POSSESSION_END_MADE_BASKET,
    POSSESSION_END_TURNOVER,
    REBOUND_CONTEST_MODE_LOCAL,
    TEAM_A,
    TEAM_B,
    TURNOVER_REASON_INBOUND_INVALID_PASS,
    TURNOVER_REASON_INBOUND_TIMEOUT,
    TURNOVER_REASON_INTERCEPTED,
    build_action_masks_batch,
    build_kernel_static_from_env,
    build_turnover_probabilities_batch,
    reset_batch_minimal,
    step_batch_minimal,
)


def _multi_possession_static(
    *,
    players: int = 2,
    possession_limit: int = 2,
    illegal_defense_enabled: bool = False,
):
    env = HexagonBasketballEnv(
        players=players,
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


def _inbound_state(
    static,
    *,
    team: int = TEAM_A,
    countdown: int = 5,
    shot_clock: int = 24,
    seed: int = 70,
    receiver_at_basket: bool = False,
):
    state = reset_batch_minimal(
        static,
        jax.random.split(jax.random.PRNGKey(seed), 1),
        jax,
        jnp,
    )
    team_ids = np.asarray(
        static.offense_ids if team == TEAM_A else static.defense_ids,
        dtype=np.int32,
    )
    opponent_ids = np.asarray(
        static.defense_ids if team == TEAM_A else static.offense_ids,
        dtype=np.int32,
    )
    inbounder = int(team_ids[0])
    positions = np.asarray(state.positions).copy()
    positions[0, inbounder] = np.asarray(static.inbound_position)
    if team_ids.size > 1:
        positions[0, int(team_ids[1])] = (
            np.asarray(static.basket_position)
            if receiver_at_basket
            else np.asarray([0, 8], dtype=np.int32)
        )
    for offset, player_id in enumerate(opponent_ids.tolist()):
        positions[0, player_id] = np.asarray([5 + offset, 5], dtype=np.int32)
    return state._replace(
        positions=jnp.asarray(positions, dtype=jnp.int32),
        offense_team=jnp.asarray([team], dtype=jnp.int8),
        starting_offense_team=jnp.asarray([team], dtype=jnp.int8),
        ball_holder=jnp.asarray([inbounder], dtype=jnp.int32),
        shot_clock=jnp.asarray([shot_clock], dtype=jnp.int32),
        game_phase=jnp.asarray([GAME_PHASE_AWAITING_INBOUND], dtype=jnp.int8),
        inbound_team=jnp.asarray([team], dtype=jnp.int8),
        inbound_player=jnp.asarray([inbounder], dtype=jnp.int32),
        inbound_reason=jnp.asarray([POSSESSION_END_MADE_BASKET], dtype=jnp.int32),
        inbound_steps_remaining=jnp.asarray([countdown], dtype=jnp.int32),
        clearance_achieved=jnp.asarray([0], dtype=jnp.int8),
    )


def test_multi_possession_cli_validates_limit_and_disables_start_templates():
    assert parse_args(["--enable-multi-possession"]).multi_possession_limit == 25
    assert parse_args(["--enable-multi-possession"]).inbound_deadline_steps == 5
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
    with pytest.raises(SystemExit, match="inbound-deadline-steps"):
        validate_train_args(
            parse_args(["--enable-multi-possession", "--inbound-deadline-steps", "0"])
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


def _complete_inbound(static, state, seed: int):
    masks = build_action_masks_batch(static, state, jnp)
    actions = _noops(state)
    inbounders = np.asarray(state.inbound_player, dtype=np.int32)
    masks_np = np.asarray(masks, dtype=np.int8)
    for batch_idx, inbounder in enumerate(inbounders.tolist()):
        legal_slots = np.flatnonzero(
            masks_np[batch_idx, inbounder, PASS_ACTION_START:]
        )
        assert legal_slots.size > 0
        actions = actions.at[batch_idx, inbounder].set(
            PASS_ACTION_START + int(legal_slots[0])
        )
    out = step_batch_minimal(
        static,
        state,
        actions,
        jax.random.split(jax.random.PRNGKey(seed), state.positions.shape[0]),
        jax,
        jnp,
    )
    assert np.all(np.asarray(out.completed_pass, dtype=np.int8) == 1)
    assert np.all(np.asarray(out.state.game_phase, dtype=np.int8) == GAME_PHASE_LIVE)
    live_state = out.state
    reentry_masks = np.asarray(
        build_action_masks_batch(static, live_state, jnp), dtype=np.int8
    )
    pending_inbounders = np.asarray(live_state.inbound_player, dtype=np.int32)
    blocked_rows = [
        batch_idx
        for batch_idx, inbounder in enumerate(pending_inbounders.tolist())
        if not np.any(reentry_masks[batch_idx, inbounder, 1:7])
    ]
    if blocked_rows:
        vacate_actions = _noops(live_state)
        positions = np.asarray(live_state.positions)
        live_masks = np.asarray(
            build_action_masks_batch(static, live_state, jnp), dtype=np.int8
        )
        for batch_idx in blocked_rows:
            basket_occupant = int(
                np.flatnonzero(
                    np.all(
                        positions[batch_idx] == np.asarray(static.basket_position),
                        axis=-1,
                    )
                )[0]
            )
            legal_moves = np.flatnonzero(live_masks[batch_idx, basket_occupant, 1:7])
            assert legal_moves.size > 0
            vacate_actions = vacate_actions.at[batch_idx, basket_occupant].set(
                1 + int(legal_moves[0])
            )
        live_state = step_batch_minimal(
            static,
            live_state,
            vacate_actions,
            jax.random.split(
                jax.random.PRNGKey(seed + 500), state.positions.shape[0]
            ),
            jax,
            jnp,
        ).state
        reentry_masks = np.asarray(
            build_action_masks_batch(static, live_state, jnp), dtype=np.int8
        )

    reentry_actions = _noops(live_state)
    pending_inbounders = np.asarray(live_state.inbound_player, dtype=np.int32)
    for batch_idx, inbounder in enumerate(pending_inbounders.tolist()):
        legal_moves = np.flatnonzero(
            reentry_masks[batch_idx, inbounder, 1:7]
        )
        assert legal_moves.size > 0
        reentry_actions = reentry_actions.at[batch_idx, inbounder].set(
            1 + int(legal_moves[0])
        )
    reentry_out = step_batch_minimal(
        static,
        live_state,
        reentry_actions,
        jax.random.split(
            jax.random.PRNGKey(seed + 1_000),
            state.positions.shape[0],
        ),
        jax,
        jnp,
    )
    assert np.all(np.asarray(reentry_out.state.inbound_player) == -1)
    # #21 owns the actual clearance transition.  Lifecycle-only tests mark it
    # complete so their next forced shot remains in scope for #19.
    return reentry_out.state._replace(
        clearance_achieved=jnp.ones_like(reentry_out.state.clearance_achieved)
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


def test_inbound_coordinate_entry_mapping_and_nearest_selection_for_both_teams():
    static = _multi_possession_static(possession_limit=4)
    expected_inbound = np.asarray(static.basket_position) + np.asarray(
        static.hex_directions
    )[ActionType.MOVE_W.value - MOVE_ACTION_START]
    np.testing.assert_array_equal(np.asarray(static.inbound_position), expected_inbound)
    court = {tuple(coord) for coord in np.asarray(static.cell_coords).tolist()}
    entry_cells = {
        tuple(np.asarray(static.inbound_position) + direction)
        for direction in np.asarray(static.hex_directions)
        if tuple(np.asarray(static.inbound_position) + direction) in court
    }
    assert entry_cells == {tuple(np.asarray(static.basket_position))}

    state = reset_batch_minimal(
        static,
        jax.random.split(jax.random.PRNGKey(71), 2),
        jax,
        jnp,
    )
    team_a = np.asarray(static.offense_ids, dtype=np.int32)
    team_b = np.asarray(static.defense_ids, dtype=np.int32)
    positions = np.asarray(state.positions).copy()
    positions[0, team_a[0]] = np.asarray([0, 0], dtype=np.int32)
    positions[0, team_a[1]] = np.asarray([1, 0], dtype=np.int32)
    positions[0, team_b[0]] = np.asarray(static.basket_position)
    positions[0, team_b[1]] = np.asarray([5, 5], dtype=np.int32)
    positions[1, team_b[0]] = np.asarray([0, 0], dtype=np.int32)
    positions[1, team_b[1]] = np.asarray([1, 0], dtype=np.int32)
    positions[1, team_a[0]] = np.asarray(static.basket_position)
    positions[1, team_a[1]] = np.asarray([5, 5], dtype=np.int32)
    state = state._replace(
        positions=jnp.asarray(positions, dtype=jnp.int32),
        offense_team=jnp.asarray([TEAM_A, TEAM_B], dtype=jnp.int8),
        ball_holder=jnp.asarray([team_a[0], team_b[0]], dtype=jnp.int32),
        layup_pct=jnp.ones_like(state.layup_pct),
        three_pt_pct=jnp.ones_like(state.three_pt_pct),
        dunk_pct=jnp.ones_like(state.dunk_pct),
    )
    out = _shoot(static, state, seed=72)

    assert np.asarray(out.shot_success, dtype=np.int8).tolist() == [1, 1]
    assert np.asarray(out.state.inbound_team, dtype=np.int8).tolist() == [
        TEAM_B,
        TEAM_A,
    ]
    assert np.asarray(out.state.inbound_player, dtype=np.int32).tolist() == [
        int(team_b[0]),
        int(team_a[0]),
    ]


def test_inbound_countdown_moves_everyone_else_consumes_clock_and_freezes_lane_counters():
    static = _multi_possession_static(possession_limit=4)._replace(
        defender_pressure_turnover_chance=jnp.asarray(1.0, dtype=jnp.float32)
    )
    state = _inbound_state(static, seed=73)
    state = state._replace(
        offense_lane_steps=jnp.full_like(state.offense_lane_steps, 2),
        defense_lane_steps=jnp.full_like(state.defense_lane_steps, 3),
    )
    inbounder = int(np.asarray(state.inbound_player)[0])
    teammate = int(np.asarray(static.offense_ids)[1])
    defender = int(np.asarray(static.defense_ids)[0])
    masks = np.asarray(build_action_masks_batch(static, state, jnp), dtype=np.int8)
    assert not np.any(masks[0, inbounder, MOVE_ACTION_START : MOVE_ACTION_START + 6])
    assert masks[0, inbounder, ActionType.SHOOT.value] == 0
    assert masks[0, inbounder, PASS_ACTION_START] == 1
    np.testing.assert_array_equal(
        np.asarray(build_turnover_probabilities_batch(static, state, jnp)),
        np.zeros((1, 2), dtype=np.float32),
    )

    move_choices: dict[int, int] = {}
    destinations: set[tuple[int, int]] = set()
    positions = np.asarray(state.positions)
    occupied_positions = {
        tuple(position) for position in positions[0].tolist()
    }
    directions = np.asarray(static.hex_directions)
    for player_id in (teammate, defender):
        for direction_idx in np.flatnonzero(
            masks[0, player_id, MOVE_ACTION_START : MOVE_ACTION_START + 6]
        ):
            destination = tuple(positions[0, player_id] + directions[direction_idx])
            if destination not in destinations and destination not in occupied_positions:
                move_choices[player_id] = MOVE_ACTION_START + int(direction_idx)
                destinations.add(destination)
                break
        assert player_id in move_choices
    actions = _noops(state)
    for player_id, action in move_choices.items():
        actions = actions.at[0, player_id].set(action)
    out = step_batch_minimal(
        static,
        state,
        actions,
        jax.random.split(jax.random.PRNGKey(74), 1),
        jax,
        jnp,
    )

    assert int(np.asarray(out.state.shot_clock)[0]) == 23
    assert int(np.asarray(out.state.inbound_steps_remaining)[0]) == 4
    np.testing.assert_array_equal(
        np.asarray(out.state.offense_lane_steps), np.asarray(state.offense_lane_steps)
    )
    np.testing.assert_array_equal(
        np.asarray(out.state.defense_lane_steps), np.asarray(state.defense_lane_steps)
    )
    np.testing.assert_array_equal(
        np.asarray(out.state.positions)[0, inbounder],
        np.asarray(static.inbound_position),
    )
    for player_id in (teammate, defender):
        assert not np.array_equal(
            np.asarray(out.state.positions)[0, player_id],
            np.asarray(state.positions)[0, player_id],
        )
    assert int(np.asarray(out.turnover)[0]) == 0


def test_deadline_release_is_accepted_and_under_basket_receiver_cannot_shoot_before_clearance():
    static = _multi_possession_static(possession_limit=4)
    state = _inbound_state(
        static,
        countdown=1,
        shot_clock=1,
        seed=75,
        receiver_at_basket=True,
    )
    inbounder = int(np.asarray(state.inbound_player)[0])
    receiver = int(np.asarray(static.offense_ids)[1])
    actions = _noops(state).at[0, inbounder].set(PASS_ACTION_START)
    out = step_batch_minimal(
        static,
        state,
        actions,
        jax.random.split(jax.random.PRNGKey(76), 1),
        jax,
        jnp,
    )

    assert int(np.asarray(out.completed_pass)[0]) == 1
    assert int(np.asarray(out.turnover)[0]) == 0
    assert int(np.asarray(out.possession_ended)[0]) == 0
    assert int(np.asarray(out.state.game_phase)[0]) == GAME_PHASE_LIVE
    assert int(np.asarray(out.state.ball_holder)[0]) == receiver
    assert int(np.asarray(out.state.shot_clock)[0]) == 0
    assert int(np.asarray(out.state.inbound_steps_remaining)[0]) == 0
    np.testing.assert_array_equal(
        np.asarray(out.state.positions)[0, receiver],
        np.asarray(static.basket_position),
    )
    post_masks = np.asarray(
        build_action_masks_batch(static, out.state, jnp), dtype=np.int8
    )
    assert post_masks[0, receiver, ActionType.SHOOT.value] == 0


def test_inbounder_waits_for_occupied_entry_then_reenters_under_basket_and_becomes_eligible():
    static = _multi_possession_static(possession_limit=4)
    state = _inbound_state(static, seed=77, receiver_at_basket=True)
    inbounder = int(np.asarray(state.inbound_player)[0])
    receiver = int(np.asarray(static.offense_ids)[1])
    pass_out = step_batch_minimal(
        static,
        state,
        _noops(state).at[0, inbounder].set(PASS_ACTION_START),
        jax.random.split(jax.random.PRNGKey(78), 1),
        jax,
        jnp,
    )
    blocked_masks = np.asarray(
        build_action_masks_batch(static, pass_out.state, jnp), dtype=np.int8
    )
    assert not np.any(
        blocked_masks[0, inbounder, MOVE_ACTION_START : MOVE_ACTION_START + 6]
    )
    blocked_out = step_batch_minimal(
        static,
        pass_out.state,
        _noops(pass_out.state).at[0, inbounder].set(ActionType.MOVE_E.value),
        jax.random.split(jax.random.PRNGKey(79), 1),
        jax,
        jnp,
    )
    np.testing.assert_array_equal(
        np.asarray(blocked_out.state.positions)[0, inbounder],
        np.asarray(static.inbound_position),
    )
    assert int(np.asarray(blocked_out.turnover)[0]) == 0

    receiver_out = step_batch_minimal(
        static,
        blocked_out.state,
        _noops(blocked_out.state).at[0, receiver].set(ActionType.MOVE_E.value),
        jax.random.split(jax.random.PRNGKey(80), 1),
        jax,
        jnp,
    )
    entry_masks = np.asarray(
        build_action_masks_batch(static, receiver_out.state, jnp), dtype=np.int8
    )
    assert entry_masks[0, inbounder, ActionType.MOVE_E.value] == 1
    reentry_out = step_batch_minimal(
        static,
        receiver_out.state,
        _noops(receiver_out.state).at[0, inbounder].set(ActionType.MOVE_E.value),
        jax.random.split(jax.random.PRNGKey(81), 1),
        jax,
        jnp,
    )
    np.testing.assert_array_equal(
        np.asarray(reentry_out.state.positions)[0, inbounder],
        np.asarray(static.basket_position),
    )
    assert int(np.asarray(reentry_out.state.inbound_player)[0]) == -1
    eligible_masks = np.asarray(
        build_action_masks_batch(static, reentry_out.state, jnp), dtype=np.int8
    )
    assert eligible_masks[0, receiver, PASS_ACTION_START] == 1


def test_inbound_pass_can_be_intercepted_into_live_possession_for_actual_defender():
    static = _multi_possession_static(possession_limit=4)._replace(
        base_steal_rate=jnp.asarray(1.0e6, dtype=jnp.float32)
    )
    state = _inbound_state(static, seed=82)
    inbounder = int(np.asarray(state.inbound_player)[0])
    defender = int(np.asarray(static.defense_ids)[0])
    positions = np.asarray(state.positions).copy()
    positions[0, defender] = np.asarray(static.basket_position)
    state = state._replace(positions=jnp.asarray(positions, dtype=jnp.int32))
    out = step_batch_minimal(
        static,
        state,
        _noops(state).at[0, inbounder].set(PASS_ACTION_START),
        jax.random.split(jax.random.PRNGKey(83), 1),
        jax,
        jnp,
    )

    assert int(np.asarray(out.turnover)[0]) == 1
    assert int(np.asarray(out.turnover_reason)[0]) == TURNOVER_REASON_INTERCEPTED
    assert int(np.asarray(out.steal_player)[0]) == defender
    assert int(np.asarray(out.state.ball_holder)[0]) == defender
    assert int(np.asarray(out.state.offense_team)[0]) == TEAM_B
    assert int(np.asarray(out.state.game_phase)[0]) == GAME_PHASE_LIVE
    assert int(np.asarray(out.state.completed_possessions)[0]) == 1


def test_empty_invalid_and_repeated_inbound_violations_switch_to_the_nonviolating_team():
    empty_static = _multi_possession_static(players=1, possession_limit=4)
    empty_state = _inbound_state(empty_static, countdown=1, seed=84)
    empty_inbounder = int(np.asarray(empty_state.inbound_player)[0])
    empty_masks = np.asarray(
        build_action_masks_batch(empty_static, empty_state, jnp), dtype=np.int8
    )
    assert not np.any(
        empty_masks[0, empty_inbounder, PASS_ACTION_START:PASS_ACTION_END]
    )

    static = _multi_possession_static(possession_limit=5)
    state = _inbound_state(static, countdown=1, seed=85)
    timeout_out = step_batch_minimal(
        static,
        state,
        _noops(state),
        jax.random.split(jax.random.PRNGKey(86), 1),
        jax,
        jnp,
    )
    assert int(np.asarray(timeout_out.turnover_reason)[0]) == TURNOVER_REASON_INBOUND_TIMEOUT
    assert int(np.asarray(timeout_out.possession_end_reason)[0]) == POSSESSION_END_INBOUND_VIOLATION
    assert int(np.asarray(timeout_out.state.offense_team)[0]) == TEAM_B
    assert int(np.asarray(timeout_out.state.inbound_team)[0]) == TEAM_B
    assert int(np.asarray(timeout_out.state.completed_possessions)[0]) == 1
    assert np.sum(
        np.all(
            np.asarray(timeout_out.state.positions)[0]
            == np.asarray(static.inbound_position),
            axis=-1,
        )
    ) == 1

    new_inbounder = int(np.asarray(timeout_out.state.inbound_player)[0])
    invalid_action = PASS_ACTION_START + 1
    invalid_masks = np.asarray(
        build_action_masks_batch(static, timeout_out.state, jnp), dtype=np.int8
    )
    assert invalid_masks[0, new_inbounder, invalid_action] == 0
    invalid_out = step_batch_minimal(
        static,
        timeout_out.state,
        _noops(timeout_out.state).at[0, new_inbounder].set(invalid_action),
        jax.random.split(jax.random.PRNGKey(87), 1),
        jax,
        jnp,
    )
    assert int(np.asarray(invalid_out.turnover_reason)[0]) == TURNOVER_REASON_INBOUND_INVALID_PASS
    assert int(np.asarray(invalid_out.possession_end_reason)[0]) == POSSESSION_END_INBOUND_VIOLATION
    assert int(np.asarray(invalid_out.state.offense_team)[0]) == TEAM_A
    assert int(np.asarray(invalid_out.state.inbound_team)[0]) == TEAM_A
    assert int(np.asarray(invalid_out.state.completed_possessions)[0]) == 2
    assert np.sum(
        np.all(
            np.asarray(invalid_out.state.positions)[0]
            == np.asarray(static.inbound_position),
            axis=-1,
        )
    ) == 1


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
    np.testing.assert_array_equal(
        np.asarray(out.state.ball_holder, dtype=np.int32),
        np.asarray(out.state.inbound_player, dtype=np.int32),
    )
    assert np.all(np.asarray(out.state.inbound_player, dtype=np.int32) >= 0)
    assert np.all(np.asarray(out.state.inbound_reason, dtype=np.int32) == reasons)
    assert np.all(np.asarray(out.state.inbound_steps_remaining, dtype=np.int32) == 5)
    assert np.all(np.asarray(out.state.clearance_achieved, dtype=np.int8) == 0)
    for row_idx, inbounder in enumerate(
        np.asarray(out.state.inbound_player, dtype=np.int32).tolist()
    ):
        assert tuple(np.asarray(out.state.positions)[row_idx, inbounder]) == tuple(
            np.asarray(static.inbound_position)
        )
        non_inbounders = [
            pid for pid in range(state.positions.shape[1]) if pid != inbounder
        ]
        np.testing.assert_array_equal(
            np.asarray(out.state.positions)[row_idx, non_inbounders],
            np.asarray(state.positions)[row_idx, non_inbounders],
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

    resumed = _complete_inbound(static, out.state, seed=40)
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
    assert int(np.asarray(violation_out.state.ball_holder)[0]) == int(
        np.asarray(violation_out.state.inbound_player)[0]
    )


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
        prior_positions = np.asarray(state.positions).copy()
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
        boundary_positions = np.asarray(out.state.positions)
        if possession_number < 3:
            inbounder = int(np.asarray(out.state.inbound_player)[0])
            other_players = np.arange(boundary_positions.shape[1]) != inbounder
            np.testing.assert_array_equal(
                boundary_positions[:, other_players],
                prior_positions[:, other_players],
            )
            np.testing.assert_array_equal(
                boundary_positions[0, inbounder],
                np.asarray(static.inbound_position),
            )
        else:
            np.testing.assert_array_equal(boundary_positions, prior_positions)
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
            state = _complete_inbound(static, out.state, seed=seed + 100)
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
    assert int(np.asarray(out.state.ball_holder)[0]) == int(
        np.asarray(out.state.inbound_player)[0]
    )
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
    positions = np.asarray(state.positions).copy()
    positions[2, team_a_holder] = np.asarray(static.inbound_position)
    state = state._replace(
        positions=jnp.asarray(positions, dtype=jnp.int32),
        offense_team=jnp.asarray([TEAM_A, TEAM_B, TEAM_A], dtype=jnp.int8),
        starting_offense_team=jnp.asarray([TEAM_A, TEAM_B, TEAM_A], dtype=jnp.int8),
        ball_holder=jnp.asarray(
            [team_a_holder, team_b_holder, team_a_holder], dtype=jnp.int32
        ),
        game_phase=jnp.asarray(
            [GAME_PHASE_LIVE, GAME_PHASE_LIVE, GAME_PHASE_AWAITING_INBOUND],
            dtype=jnp.int8,
        ),
        inbound_team=jnp.asarray([-1, -1, TEAM_A], dtype=jnp.int8),
        inbound_player=jnp.asarray([-1, -1, team_a_holder], dtype=jnp.int32),
        inbound_steps_remaining=jnp.asarray([0, 0, 5], dtype=jnp.int32),
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
