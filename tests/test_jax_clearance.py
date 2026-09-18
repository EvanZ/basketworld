from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from basketworld.envs.basketworld_env_v2 import ActionType
from basketworld_jax.env.minimal import (
    GAME_PHASE_AWAITING_INBOUND,
    GAME_PHASE_LIVE,
    MOVE_ACTION_START,
    PASS_ACTION_START,
    TEAM_A,
    TEAM_B,
    TURNOVER_REASON_CLEARANCE_VIOLATION,
    TURNOVER_REASON_INTERCEPTED,
    TURNOVER_REASON_OFFENSIVE_THREE_SECONDS,
    TURNOVER_REASON_SHOT_CLOCK,
    build_action_masks_batch,
    reset_batch_minimal,
    step_batch_minimal,
)
from tests.test_jax_multi_possession import (
    _force_rebound_winner,
    _inbound_state,
    _multi_possession_static,
    _noops,
    _shoot,
)


def _stack_states(*states):
    return type(states[0])(
        *(
            jnp.concatenate([getattr(state, field) for state in states], axis=0)
            for field in states[0]._fields
        )
    )


def _clearance_boundary(static) -> tuple[np.ndarray, int, np.ndarray]:
    cells = np.asarray(static.cell_coords, dtype=np.int32)
    outside = np.asarray(static.three_point_by_cell, dtype=bool)
    directions = np.asarray(static.hex_directions, dtype=np.int32)
    cell_lookup = {tuple(cell): idx for idx, cell in enumerate(cells.tolist())}
    for cell_idx, cell in enumerate(cells):
        if outside[cell_idx]:
            continue
        for direction_idx, direction in enumerate(directions):
            destination = cell + direction
            destination_idx = cell_lookup.get(tuple(destination))
            if destination_idx is not None and outside[destination_idx]:
                return cell, MOVE_ACTION_START + direction_idx, destination
    raise AssertionError("Court has no inside-to-outside clearance boundary.")


def _set_live_team_state(
    static, state, *, team: int, positions: np.ndarray, holder: int
):
    return state._replace(
        positions=jnp.asarray(positions[None, ...], dtype=jnp.int32),
        offense_team=jnp.asarray([team], dtype=jnp.int8),
        starting_offense_team=jnp.asarray([team], dtype=jnp.int8),
        ball_holder=jnp.asarray([holder], dtype=jnp.int32),
        game_phase=jnp.asarray([GAME_PHASE_LIVE], dtype=jnp.int8),
        inbound_team=jnp.asarray([-1], dtype=jnp.int8),
        inbound_player=jnp.asarray([-1], dtype=jnp.int32),
        clearance_achieved=jnp.asarray([0], dtype=jnp.int8),
    )


def test_move_to_clear_works_for_both_teams_without_bonus_or_scripted_action():
    static = _multi_possession_static(possession_limit=6)
    inside, move_action, outside = _clearance_boundary(static)
    team_a = np.asarray(static.offense_ids, dtype=np.int32)
    team_b = np.asarray(static.defense_ids, dtype=np.int32)
    rows = []
    for seed, team, holder in (
        (101, TEAM_A, int(team_a[0])),
        (102, TEAM_B, int(team_b[0])),
    ):
        state = reset_batch_minimal(
            static,
            jax.random.split(jax.random.PRNGKey(seed), 1),
            jax,
            jnp,
        )
        positions = np.asarray([[5, 5], [6, 5], [7, 5], [8, 5]], dtype=np.int32)
        positions[holder] = inside
        rows.append(
            _set_live_team_state(
                static,
                state,
                team=team,
                positions=positions,
                holder=holder,
            )._replace(
                shot_clock=jnp.asarray(
                    [int(np.asarray(static.shot_clock_max))], dtype=jnp.int32
                )
            )
        )
    state = _stack_states(*rows)
    holders = np.asarray(state.ball_holder, dtype=np.int32)
    masks = np.asarray(build_action_masks_batch(static, state, jnp), dtype=np.int8)
    for row, holder in enumerate(holders.tolist()):
        assert masks[row, holder, ActionType.SHOOT.value] == 1
    actions = _noops(state)
    for row, holder in enumerate(holders.tolist()):
        actions = actions.at[row, holder].set(move_action)
    out = step_batch_minimal(
        static,
        state,
        actions,
        jax.random.split(jax.random.PRNGKey(103), 2),
        jax,
        jnp,
    )

    assert np.asarray(out.state.clearance_achieved, dtype=np.int8).tolist() == [1, 1]
    assert np.asarray(out.clearance_event, dtype=np.int8).tolist() == [1, 1]
    assert np.asarray(out.clearance_elapsed_steps, dtype=np.int32).tolist() == [1, 1]
    for row, holder in enumerate(holders.tolist()):
        np.testing.assert_array_equal(
            np.asarray(out.state.positions)[row, holder], outside
        )
    np.testing.assert_array_equal(
        np.asarray(out.rewards), np.zeros((2, 4), dtype=np.float32)
    )

    direction_idx = move_action - MOVE_ACTION_START
    directions = np.asarray(static.hex_directions, dtype=np.int32)
    reverse_idx = int(
        np.flatnonzero(np.all(directions == -directions[direction_idx], axis=1))[0]
    )
    reverse_actions = _noops(out.state)
    for row, holder in enumerate(holders.tolist()):
        reverse_actions = reverse_actions.at[row, holder].set(
            MOVE_ACTION_START + reverse_idx
        )
    returned = step_batch_minimal(
        static,
        out.state,
        reverse_actions,
        jax.random.split(jax.random.PRNGKey(118), 2),
        jax,
        jnp,
    )
    assert np.asarray(returned.state.clearance_achieved, dtype=np.int8).tolist() == [
        1,
        1,
    ]
    assert np.asarray(returned.clearance_event, dtype=np.int8).tolist() == [0, 0]


def test_inbound_inside_catch_does_not_clear_but_outside_catch_does_for_other_team():
    static = _multi_possession_static(possession_limit=6)
    inside_state = _inbound_state(
        static,
        team=TEAM_A,
        seed=104,
        receiver_at_basket=True,
    )
    outside_state = _inbound_state(static, team=TEAM_B, seed=105)
    outside_receiver = int(np.asarray(static.defense_ids)[1])
    outside_idx = int(np.flatnonzero(np.asarray(static.three_point_by_cell))[0])
    outside_positions = np.asarray(outside_state.positions).copy()
    outside_positions[0, outside_receiver] = np.asarray(static.cell_coords)[outside_idx]
    outside_state = outside_state._replace(
        positions=jnp.asarray(outside_positions, dtype=jnp.int32)
    )
    state = _stack_states(inside_state, outside_state)
    actions = _noops(state)
    for row, inbounder in enumerate(
        np.asarray(state.inbound_player, dtype=np.int32).tolist()
    ):
        actions = actions.at[row, inbounder].set(PASS_ACTION_START)
    out = step_batch_minimal(
        static,
        state,
        actions,
        jax.random.split(jax.random.PRNGKey(106), 2),
        jax,
        jnp,
    )

    assert np.asarray(out.completed_pass, dtype=np.int8).tolist() == [1, 1]
    assert np.asarray(out.state.clearance_achieved, dtype=np.int8).tolist() == [0, 1]
    assert np.asarray(out.clearance_event, dtype=np.int8).tolist() == [0, 1]
    assert np.asarray(out.clearance_elapsed_steps, dtype=np.int32).tolist() == [0, 1]


def test_live_pass_to_teammate_outside_clears_for_both_teams():
    static = _multi_possession_static(possession_limit=6)._replace(
        base_steal_rate=jnp.asarray(0.0, dtype=jnp.float32)
    )
    team_a = np.asarray(static.offense_ids, dtype=np.int32)
    team_b = np.asarray(static.defense_ids, dtype=np.int32)
    outside_idx = int(np.flatnonzero(np.asarray(static.three_point_by_cell))[0])
    outside = np.asarray(static.cell_coords)[outside_idx]
    rows = []
    receivers = []
    pass_actions = []
    for seed, team, team_ids in (
        (119, TEAM_A, team_a),
        (120, TEAM_B, team_b),
    ):
        state = reset_batch_minimal(
            static,
            jax.random.split(jax.random.PRNGKey(seed), 1),
            jax,
            jnp,
        )
        holder = int(team_ids[0])
        receiver = int(team_ids[1])
        positions = np.asarray(state.positions)[0].copy()
        positions[holder] = np.asarray(static.basket_position)
        positions[receiver] = outside
        rows.append(
            _set_live_team_state(
                static,
                state,
                team=team,
                positions=positions,
                holder=holder,
            )
        )
        target_slots = np.flatnonzero(
            np.asarray(static.pointer_pass_target_ids)[holder] == receiver
        )
        assert target_slots.size == 1
        receivers.append(receiver)
        pass_actions.append(PASS_ACTION_START + int(target_slots[0]))

    state = _stack_states(*rows)
    actions = _noops(state)
    for row, holder in enumerate(np.asarray(state.ball_holder, dtype=np.int32)):
        actions = actions.at[row, holder].set(pass_actions[row])
    out = step_batch_minimal(
        static,
        state,
        actions,
        jax.random.split(jax.random.PRNGKey(121), 2),
        jax,
        jnp,
    )

    assert np.asarray(out.completed_pass, dtype=np.int8).tolist() == [1, 1]
    assert np.asarray(out.state.ball_holder, dtype=np.int32).tolist() == receivers
    assert np.asarray(out.state.clearance_achieved, dtype=np.int8).tolist() == [1, 1]
    assert np.asarray(out.clearance_event, dtype=np.int8).tolist() == [1, 1]


def test_preclear_shot_is_selectable_but_becomes_dead_ball_clearance_violation():
    static = _multi_possession_static(possession_limit=6)
    state = reset_batch_minimal(
        static,
        jax.random.split(jax.random.PRNGKey(107), 1),
        jax,
        jnp,
    )._replace(clearance_achieved=jnp.asarray([0], dtype=jnp.int8))
    holder = int(np.asarray(state.ball_holder)[0])
    assert (
        int(
            np.asarray(build_action_masks_batch(static, state, jnp))[
                0, holder, ActionType.SHOOT.value
            ]
        )
        == 1
    )
    out = step_batch_minimal(
        static,
        state,
        _noops(state).at[0, holder].set(ActionType.SHOOT.value),
        jax.random.split(jax.random.PRNGKey(108), 1),
        jax,
        jnp,
    )

    assert int(np.asarray(out.shot_attempt)[0]) == 0
    assert int(np.asarray(out.turnover)[0]) == 1
    assert (
        int(np.asarray(out.turnover_reason)[0]) == TURNOVER_REASON_CLEARANCE_VIOLATION
    )
    assert int(np.asarray(out.turnover_before_clearance)[0]) == 1
    assert int(np.asarray(out.state.offense_team)[0]) == TEAM_B
    assert int(np.asarray(out.state.game_phase)[0]) == GAME_PHASE_AWAITING_INBOUND


def test_outside_interceptor_clears_immediately_and_repeated_turnovers_recompute_for_both_teams():
    static = _multi_possession_static(possession_limit=6)._replace(
        base_steal_rate=jnp.asarray(1.0e6, dtype=jnp.float32)
    )
    state = reset_batch_minimal(
        static,
        jax.random.split(jax.random.PRNGKey(109), 1),
        jax,
        jnp,
    )
    team_a = np.asarray(static.offense_ids, dtype=np.int32)
    team_b = np.asarray(static.defense_ids, dtype=np.int32)
    positions = np.zeros((4, 2), dtype=np.int32)
    positions[team_a[0]] = np.asarray(static.basket_position)
    positions[team_a[1]] = np.asarray([4, 8], dtype=np.int32)
    positions[team_b[0]] = np.asarray([0, 8], dtype=np.int32)
    positions[team_b[1]] = np.asarray([5, 5], dtype=np.int32)
    state = _set_live_team_state(
        static,
        state,
        team=TEAM_A,
        positions=positions,
        holder=int(team_a[0]),
    )
    first = step_batch_minimal(
        static,
        state,
        _noops(state).at[0, int(team_a[0])].set(PASS_ACTION_START),
        jax.random.split(jax.random.PRNGKey(110), 1),
        jax,
        jnp,
    )
    assert int(np.asarray(first.turnover_reason)[0]) == TURNOVER_REASON_INTERCEPTED
    assert int(np.asarray(first.state.ball_holder)[0]) == int(team_b[0])
    assert int(np.asarray(first.state.offense_team)[0]) == TEAM_B
    assert int(np.asarray(first.state.clearance_achieved)[0]) == 1
    assert int(np.asarray(first.clearance_event)[0]) == 1
    assert int(np.asarray(first.turnover_before_clearance)[0]) == 1

    positions[team_b[0]] = np.asarray(static.basket_position)
    positions[team_b[1]] = np.asarray([4, 8], dtype=np.int32)
    positions[team_a[0]] = np.asarray([-3, 8], dtype=np.int32)
    positions[team_a[1]] = np.asarray([-2, 8], dtype=np.int32)
    second_state = first.state._replace(
        positions=jnp.asarray(positions[None, ...], dtype=jnp.int32),
        ball_holder=jnp.asarray([team_b[0]], dtype=jnp.int32),
        clearance_achieved=jnp.asarray([0], dtype=jnp.int8),
    )
    second = step_batch_minimal(
        static,
        second_state,
        _noops(second_state).at[0, int(team_b[0])].set(PASS_ACTION_START),
        jax.random.split(jax.random.PRNGKey(111), 1),
        jax,
        jnp,
    )
    assert int(np.asarray(second.turnover_reason)[0]) == TURNOVER_REASON_INTERCEPTED
    second_holder = int(np.asarray(second.state.ball_holder)[0])
    assert second_holder in set(team_a.tolist())
    holder_position = np.asarray(second.state.positions)[0, second_holder]
    holder_cell = int(
        np.flatnonzero(
            np.all(np.asarray(static.cell_coords) == holder_position, axis=1)
        )[0]
    )
    assert int(np.asarray(static.three_point_by_cell)[holder_cell]) == 0
    assert int(np.asarray(second.state.offense_team)[0]) == TEAM_A
    assert int(np.asarray(second.state.clearance_achieved)[0]) == 0
    assert int(np.asarray(second.clearance_event)[0]) == 0
    assert int(np.asarray(second.state.completed_possessions)[0]) == 2


def test_offensive_rebound_preserves_clearance_and_continuation_clock_rules():
    static = _multi_possession_static(possession_limit=6)
    state = reset_batch_minimal(
        static,
        jax.random.split(jax.random.PRNGKey(112), 1),
        jax,
        jnp,
    )
    holder = int(np.asarray(state.ball_holder)[0])
    state = state._replace(
        clearance_achieved=jnp.asarray([1], dtype=jnp.int8),
        layup_pct=jnp.zeros_like(state.layup_pct),
        three_pt_pct=jnp.zeros_like(state.three_pt_pct),
        dunk_pct=jnp.zeros_like(state.dunk_pct),
        shot_clock=jnp.asarray([5], dtype=jnp.int32),
    )
    static = _force_rebound_winner(static, state, holder)
    out = _shoot(static, state, seed=113)

    assert int(np.asarray(out.offensive_rebound)[0]) == 1
    assert int(np.asarray(out.possession_ended)[0]) == 0
    assert int(np.asarray(out.state.clearance_achieved)[0]) == 1
    assert int(np.asarray(out.clearance_event)[0]) == 0
    assert int(np.asarray(out.state.shot_clock)[0]) >= int(
        np.asarray(static.offensive_rebound_shot_clock_reset)
    )


def test_preclear_clock_and_offensive_lane_violations_use_dead_ball_restarts():
    static = _multi_possession_static(possession_limit=6)
    state = reset_batch_minimal(
        static,
        jax.random.split(jax.random.PRNGKey(114), 1),
        jax,
        jnp,
    )._replace(
        clearance_achieved=jnp.asarray([0], dtype=jnp.int8),
        shot_clock=jnp.asarray([1], dtype=jnp.int32),
    )
    clock_out = step_batch_minimal(
        static,
        state,
        _noops(state),
        jax.random.split(jax.random.PRNGKey(115), 1),
        jax,
        jnp,
    )
    assert int(np.asarray(clock_out.turnover_reason)[0]) == TURNOVER_REASON_SHOT_CLOCK
    assert int(np.asarray(clock_out.turnover_before_clearance)[0]) == 1
    assert int(np.asarray(clock_out.state.game_phase)[0]) == GAME_PHASE_AWAITING_INBOUND

    lane_static = _multi_possession_static(
        possession_limit=6,
        offensive_three_seconds_enabled=True,
    )._replace(
        three_second_max_steps=jnp.asarray(0, dtype=jnp.int32),
    )
    lane_state = reset_batch_minimal(
        lane_static,
        jax.random.split(jax.random.PRNGKey(116), 1),
        jax,
        jnp,
    )._replace(clearance_achieved=jnp.asarray([0], dtype=jnp.int8))
    lane_idx = int(np.flatnonzero(np.asarray(lane_static.offensive_lane_by_cell))[0])
    lane_positions = np.asarray(lane_state.positions).copy()
    lane_player = int(np.asarray(lane_static.offense_ids)[1])
    lane_positions[0, lane_player] = np.asarray(lane_static.cell_coords)[lane_idx]
    lane_state = lane_state._replace(
        positions=jnp.asarray(lane_positions, dtype=jnp.int32)
    )
    lane_out = step_batch_minimal(
        lane_static,
        lane_state,
        _noops(lane_state),
        jax.random.split(jax.random.PRNGKey(117), 1),
        jax,
        jnp,
    )
    assert (
        int(np.asarray(lane_out.turnover_reason)[0])
        == TURNOVER_REASON_OFFENSIVE_THREE_SECONDS
    )
    assert int(np.asarray(lane_out.turnover_before_clearance)[0]) == 1
    assert int(np.asarray(lane_out.state.game_phase)[0]) == GAME_PHASE_AWAITING_INBOUND
