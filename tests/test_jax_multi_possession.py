from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from basketworld.envs.basketworld_env_v2 import ActionType, HexagonBasketballEnv
from basketworld_jax.train.main import parse_args, validate_train_args
from basketworld_jax.train.runtime import (
    _build_spatial_step_metrics,
    _selector_segment_application_masks,
    summarize_completed_possession_pace,
)
from basketworld_jax.env.minimal import (
    GAME_PHASE_AWAITING_CHECK,
    GAME_PHASE_AWAITING_INBOUND,
    GAME_PHASE_LIVE,
    MOVE_ACTION_START,
    MOVE_ACTION_END,
    MULTI_POSSESSION_REWARD_SCORING_EVENTS,
    MADE_BASKET_RESTART_BASELINE_INBOUND,
    MADE_BASKET_RESTART_CHECK,
    PASS_ACTION_START,
    PASS_ACTION_END,
    POSSESSION_END_INBOUND_VIOLATION,
    POSSESSION_END_CHECK_VIOLATION,
    POSSESSION_END_DEFENSIVE_REBOUND,
    POSSESSION_END_DEFENSIVE_VIOLATION,
    POSSESSION_END_MADE_BASKET,
    POSSESSION_END_TURNOVER,
    REBOUND_CONTEST_MODE_LOCAL,
    TEAM_A,
    TEAM_B,
    TURNOVER_REASON_DEFENDER_PRESSURE,
    TURNOVER_REASON_INBOUND_INVALID_PASS,
    TURNOVER_REASON_INBOUND_TIMEOUT,
    TURNOVER_REASON_INTERCEPTED,
    _select_inbounder_single,
    _select_technical_inbound_position_single,
    _prepare_check_positions_single,
    build_action_masks_batch,
    build_kernel_static_from_env,
    build_spatial_diagnostics_batch,
    build_token_observation_components_batch,
    build_turnover_probabilities_batch,
    reset_batch_minimal,
    sample_state_batch,
    step_batch_minimal,
)


def _multi_possession_static(
    *,
    players: int = 2,
    possession_limit: int = 2,
    illegal_defense_enabled: bool = False,
    offensive_three_seconds_enabled: bool = False,
    court_rows: int | None = None,
    court_cols: int | None = None,
    three_point_distance: float = 4.0,
    three_point_short_distance: float | None = None,
    overtime_round_cap: int = 0,
):
    env = HexagonBasketballEnv(
        players=players,
        render_mode=None,
        court_rows=court_rows,
        court_cols=court_cols,
        pass_mode="pointer_targeted",
        allow_dunks=False,
        three_point_distance=three_point_distance,
        three_point_short_distance=three_point_short_distance,
        layup_pct=1.0,
        three_pt_pct=1.0,
        three_pt_extra_hex_decay=0.0,
        dunk_pct=1.0,
        shot_pressure_enabled=False,
        defender_pressure_turnover_chance=0.0,
        base_steal_rate=0.0,
        illegal_defense_enabled=illegal_defense_enabled,
        offensive_three_seconds_enabled=offensive_three_seconds_enabled,
        enable_phi_shaping=False,
    )
    env.enable_multi_possession = True
    env.multi_possession_limit = possession_limit
    env.multi_possession_overtime_round_cap = overtime_round_cap
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


def test_spatial_diagnostics_follow_active_roles_and_mark_corner_zone():
    static = _multi_possession_static(players=2)
    state = reset_batch_minimal(
        static,
        jax.random.split(jax.random.PRNGKey(41), 1),
        jax,
        jnp,
    )
    cells = np.asarray(static.cell_coords, dtype=np.int32)
    corner_index = int(np.flatnonzero(np.asarray(static.corner_cell_mask))[0])
    spread_indices = np.linspace(0, len(cells) - 1, num=2, dtype=np.int32)
    compact_team_a = np.repeat(cells[corner_index][None, :], repeats=2, axis=0)
    spread_team_b = cells[spread_indices]
    positions = jnp.asarray(
        np.concatenate([compact_team_a, spread_team_b], axis=0)[None, :, :],
        dtype=jnp.int32,
    )

    team_a_offense = state._replace(
        positions=positions,
        offense_team=jnp.asarray([TEAM_A], dtype=jnp.int8),
    )
    team_a_metrics = build_spatial_diagnostics_batch(static, team_a_offense, jnp)
    assert float(team_a_metrics["spatial_offense_teammate_pair_distance"][0]) == 0.0
    assert float(team_a_metrics["spatial_defense_teammate_pair_distance"][0]) > 0.0
    assert float(team_a_metrics["spatial_corner_player_fraction"][0]) >= 0.5

    team_b_offense = team_a_offense._replace(
        offense_team=jnp.asarray([TEAM_B], dtype=jnp.int8),
    )
    team_b_metrics = build_spatial_diagnostics_batch(static, team_b_offense, jnp)
    assert float(team_b_metrics["spatial_offense_teammate_pair_distance"][0]) > 0.0
    assert float(team_b_metrics["spatial_defense_teammate_pair_distance"][0]) == 0.0


def test_spatial_engagement_diagnostics_measure_coverage_and_gate_dead_ball_steps():
    static = _multi_possession_static(players=2)
    state = reset_batch_minimal(
        static,
        jax.random.split(jax.random.PRNGKey(42), 1),
        jax,
        jnp,
    )
    positions = jnp.asarray(
        [[[0, 0], [2, 0], [1, 0], [4, 0]]],
        dtype=jnp.int32,
    )
    live_state = state._replace(
        positions=positions,
        offense_team=jnp.asarray([TEAM_A], dtype=jnp.int8),
        ball_holder=jnp.asarray([0], dtype=jnp.int32),
        game_phase=jnp.asarray([GAME_PHASE_LIVE], dtype=jnp.int8),
    )

    diagnostics = build_spatial_diagnostics_batch(static, live_state, jnp)
    assert float(diagnostics["spatial_ball_handler_samples"][0]) == 1.0
    assert float(
        diagnostics["spatial_ball_handler_nearest_defender_distance"][0]
    ) == pytest.approx(1.0)
    assert float(diagnostics["spatial_ball_handler_pressured"][0]) == 1.0
    assert float(
        diagnostics["spatial_offense_nearest_defender_distance"][0]
    ) == pytest.approx(1.0)
    assert float(diagnostics["spatial_unguarded_offense_fraction"][0]) == 0.0
    assert float(diagnostics["spatial_team_centroid_distance"][0]) == pytest.approx(
        1.5
    )

    active = jnp.asarray([True], dtype=jnp.bool_)
    live_metrics = _build_spatial_step_metrics(static, live_state, active, jnp)
    assert float(live_metrics["spatial_live_steps"][0]) == 1.0

    check_state = live_state._replace(
        ball_holder=jnp.asarray([-1], dtype=jnp.int32),
        game_phase=jnp.asarray([GAME_PHASE_AWAITING_CHECK], dtype=jnp.int8),
    )
    dead_ball_metrics = _build_spatial_step_metrics(static, check_state, active, jnp)
    assert float(dead_ball_metrics["spatial_live_steps"][0]) == 0.0
    for key, value in dead_ball_metrics.items():
        if key != "spatial_live_steps":
            assert float(value[0]) == 0.0


def test_tied_closest_inbounders_are_selected_by_seeded_random_draw():
    static = _multi_possession_static()
    receiving_ids = np.asarray(static.offense_ids, dtype=np.int32)
    coords = np.asarray(static.cell_coords, dtype=np.int32)
    inbound_position = np.asarray(static.inbound_position, dtype=np.int32)
    delta_q = coords[:, 0] - inbound_position[0]
    delta_r = coords[:, 1] - inbound_position[1]
    distances = (np.abs(delta_q) + np.abs(delta_r) + np.abs(delta_q + delta_r)) // 2
    tied_candidates = next(
        indices
        for distance in np.unique(distances)
        if (indices := np.flatnonzero(distances == distance)).size >= 2
    )
    positions = np.zeros((int(static.role_encoding.shape[0]), 2), dtype=np.int32)
    positions[receiving_ids] = coords[tied_candidates[: receiving_ids.size]]
    positions = jnp.asarray(positions, dtype=jnp.int32)

    selected = jax.vmap(
        lambda key: _select_inbounder_single(
            static,
            positions,
            jnp.asarray(TEAM_A, dtype=jnp.int8),
            key,
            jax,
            jnp,
        )
    )(jax.random.split(jax.random.PRNGKey(97), 128))

    selected_ids = set(np.asarray(selected, dtype=np.int32).tolist())
    assert selected_ids == set(receiving_ids.tolist())


def test_equidistant_technical_inbound_sides_are_selected_by_seeded_random_draw():
    static = _multi_possession_static(players=5, court_rows=9, court_cols=8)
    distances = np.asarray(
        (
            np.abs(np.asarray(static.technical_inbound_positions)[:, 0] - int(static.check_position[0]))
            + np.abs(np.asarray(static.technical_inbound_positions)[:, 1] - int(static.check_position[1]))
            + np.abs(
                np.asarray(static.technical_inbound_positions)[:, 0]
                - int(static.check_position[0])
                + np.asarray(static.technical_inbound_positions)[:, 1]
                - int(static.check_position[1])
            )
        )
        // 2,
        dtype=np.int32,
    )
    assert distances[0] == distances[1]

    selected = jax.vmap(
        lambda key: _select_technical_inbound_position_single(
            static,
            static.check_position,
            key,
            jax,
            jnp,
        )
    )(jax.random.split(jax.random.PRNGKey(98), 128))

    selected_positions = {
        tuple(position.tolist()) for position in np.asarray(selected, dtype=np.int32)
    }
    expected_positions = {
        tuple(position.tolist())
        for position in np.asarray(static.technical_inbound_positions, dtype=np.int32)
    }
    assert selected_positions == expected_positions


def test_multi_possession_cli_validates_limit_and_disables_start_templates():
    defaults = parse_args(["--enable-multi-possession"])
    assert defaults.multi_possession_limit == 25
    assert defaults.multi_possession_limit_start is None
    assert defaults.multi_possession_limit_end is None
    assert defaults.multi_possession_limit_ramp_updates == 0
    assert defaults.multi_possession_overtime_round_cap == 0
    assert parse_args(["--enable-multi-possession"]).inbound_deadline_steps == 5
    assert parse_args(["--enable-multi-possession"]).check_deadline_steps == 5
    assert (
        parse_args(["--enable-multi-possession"]).made_basket_restart_mode
        == "baseline_inbound"
    )
    assert parse_args(["--enable-multi-possession"]).game_winner_reward == pytest.approx(0.0)
    assert parse_args(["--enable-multi-possession"]).multi_possession_use_inbounds
    assert not parse_args(
        ["--enable-multi-possession", "--no-multi-possession-use-inbounds"]
    ).multi_possession_use_inbounds
    args = parse_args(["--enable-multi-possession", "--multi-possession-limit", "25"])
    assert args.enable_multi_possession is True
    assert args.multi_possession_limit == 25
    validate_train_args(args)

    with pytest.raises(SystemExit, match="multi-possession-limit"):
        validate_train_args(
            parse_args(["--enable-multi-possession", "--multi-possession-limit", "0"])
        )
    with pytest.raises(SystemExit, match="multi-possession-limit-start"):
        validate_train_args(
            parse_args(
                ["--enable-multi-possession", "--multi-possession-limit-start", "0"]
            )
        )
    with pytest.raises(SystemExit, match="overtime-round-cap"):
        validate_train_args(
            parse_args(
                [
                    "--enable-multi-possession",
                    "--multi-possession-overtime-round-cap",
                    "-1",
                ]
            )
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
    with pytest.raises(SystemExit, match="check-deadline-steps"):
        validate_train_args(
            parse_args(["--enable-multi-possession", "--check-deadline-steps", "0"])
        )
    with pytest.raises(SystemExit, match="game-winner-reward"):
        validate_train_args(
            parse_args(["--enable-multi-possession", "--game-winner-reward", "-1"])
        )


def test_training_state_compilation_propagates_check_restart_configuration():
    args = parse_args(
        [
            "--enable-multi-possession",
            "--made-basket-restart-mode",
            "check",
            "--check-deadline-steps",
            "3",
            "--kernel-batch-size",
            "1",
            "--players",
            "2",
            "--court-rows",
            "9",
            "--court-cols",
            "8",
            "--pass-mode",
            "pointer_targeted",
        ]
    )

    static, _ = sample_state_batch(args, xp=jnp)

    assert int(np.asarray(static.made_basket_restart_mode)) == MADE_BASKET_RESTART_CHECK
    assert int(np.asarray(static.check_deadline_steps)) == 3


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


def _check_state(static, *, countdown: int = 5, seed: int = 310):
    state = reset_batch_minimal(
        static,
        jax.random.split(jax.random.PRNGKey(seed), 1),
        jax,
        jnp,
    )
    cells = np.asarray(static.cell_coords, dtype=np.int32)
    check = np.asarray(static.check_position, dtype=np.int32)
    directions = np.asarray(static.hex_directions, dtype=np.int32)
    entry_direction = next(
        idx
        for idx, direction in enumerate(directions)
        if np.any(np.all(cells == (check - direction)[None, :], axis=1))
    )
    entry = check - directions[entry_direction]
    positions = np.asarray(state.positions, dtype=np.int32).copy()
    offense_player = int(np.asarray(static.offense_ids, dtype=np.int32)[0])
    positions[0, offense_player] = entry
    occupied = {tuple(entry.tolist()), tuple(check.tolist())}
    cursor = 0
    for player_id in range(positions.shape[1]):
        if player_id == offense_player:
            continue
        while tuple(cells[cursor].tolist()) in occupied:
            cursor += 1
        positions[0, player_id] = cells[cursor]
        occupied.add(tuple(cells[cursor].tolist()))
        cursor += 1
    return (
        state._replace(
            positions=jnp.asarray(positions, dtype=jnp.int32),
            ball_holder=jnp.asarray([-1], dtype=jnp.int32),
            offense_team=jnp.asarray([TEAM_A], dtype=jnp.int8),
            game_phase=jnp.asarray([GAME_PHASE_AWAITING_CHECK], dtype=jnp.int8),
            check_team=jnp.asarray([TEAM_A], dtype=jnp.int8),
            check_steps_remaining=jnp.asarray([countdown], dtype=jnp.int32),
            check_steps_elapsed=jnp.asarray([0], dtype=jnp.int32),
            clearance_achieved=jnp.asarray([0], dtype=jnp.int8),
        ),
        offense_player,
        MOVE_ACTION_START + entry_direction,
    )


@pytest.mark.parametrize("countdown", [5, 1])
def test_check_pickup_succeeds_and_final_step_wins_over_timeout(countdown):
    static = _multi_possession_static()._replace(
        made_basket_restart_mode=jnp.asarray(
            MADE_BASKET_RESTART_CHECK,
            dtype=jnp.int8,
        ),
        check_deadline_steps=jnp.asarray(5, dtype=jnp.int32),
    )
    state, pickup_player, move_action = _check_state(static, countdown=countdown)
    actions = _noops(state).at[0, pickup_player].set(move_action)
    out = step_batch_minimal(
        static,
        state,
        actions,
        jax.random.split(jax.random.PRNGKey(311 + countdown), 1),
        jax,
        jnp,
    )

    assert int(np.asarray(out.check_pickup)[0]) == 1
    assert int(np.asarray(out.check_violation)[0]) == 0
    assert int(np.asarray(out.state.ball_holder)[0]) == pickup_player
    assert int(np.asarray(out.state.game_phase)[0]) == GAME_PHASE_LIVE
    assert int(np.asarray(out.state.clearance_achieved)[0]) == 1
    assert int(np.asarray(out.state.shot_clock)[0]) == int(np.asarray(state.shot_clock)[0])


def test_defense_cannot_enter_check_cell_and_timeout_awards_point_and_side_inbound():
    static = _multi_possession_static()._replace(
        made_basket_restart_mode=jnp.asarray(MADE_BASKET_RESTART_CHECK, dtype=jnp.int8),
        check_deadline_steps=jnp.asarray(1, dtype=jnp.int32),
        multi_possession_reward_mode=jnp.asarray(
            MULTI_POSSESSION_REWARD_SCORING_EVENTS,
            dtype=jnp.int32,
        ),
    )
    state, pickup_player, move_action = _check_state(static, countdown=1)
    check = np.asarray(static.check_position, dtype=np.int32)
    direction = np.asarray(static.hex_directions, dtype=np.int32)[move_action - MOVE_ACTION_START]
    defender = int(np.asarray(static.defense_ids, dtype=np.int32)[0])
    positions = np.asarray(state.positions, dtype=np.int32).copy()
    positions[0, defender] = check - direction
    occupied_without_pickup = {
        tuple(position.tolist())
        for player_id, position in enumerate(positions[0])
        if player_id != pickup_player
    }
    replacement = next(
        cell
        for cell in np.asarray(static.cell_coords, dtype=np.int32)
        if tuple(cell.tolist()) not in occupied_without_pickup
        and not np.array_equal(cell, check)
    )
    positions[0, pickup_player] = replacement
    state = state._replace(positions=jnp.asarray(positions, dtype=jnp.int32))
    masks = np.asarray(build_action_masks_batch(static, state, jnp), dtype=np.int8)
    assert masks[0, defender, move_action] == 0

    actions = _noops(state).at[0, defender].set(move_action)
    out = step_batch_minimal(
        static,
        state,
        actions,
        jax.random.split(jax.random.PRNGKey(320), 1),
        jax,
        jnp,
    )
    assert not np.array_equal(np.asarray(out.state.positions)[0, defender], check)
    assert int(np.asarray(out.check_violation)[0]) == 1
    assert int(np.asarray(out.possession_end_reason)[0]) == POSSESSION_END_CHECK_VIOLATION
    assert int(np.asarray(out.state.offense_team)[0]) == TEAM_B
    assert float(np.asarray(out.state.team_a_score)[0]) == pytest.approx(0.0)
    assert float(np.asarray(out.state.team_b_score)[0]) == pytest.approx(1.0)
    assert float(np.asarray(out.team_a_score_delta)[0]) == pytest.approx(0.0)
    assert float(np.asarray(out.team_b_score_delta)[0]) == pytest.approx(1.0)
    assert int(np.asarray(out.state.game_phase)[0]) == GAME_PHASE_AWAITING_INBOUND
    assert int(np.asarray(out.state.inbound_team)[0]) == TEAM_B
    assert int(np.asarray(out.state.ball_holder)[0]) == int(
        np.asarray(out.state.inbound_player)[0]
    )
    assert int(np.asarray(out.state.inbound_reason)[0]) == POSSESSION_END_CHECK_VIOLATION
    inbounder = int(np.asarray(out.state.inbound_player)[0])
    inbound_position = np.asarray(out.state.positions)[0, inbounder]
    assert any(
        np.array_equal(inbound_position, position)
        for position in np.asarray(static.technical_inbound_positions)
    )
    assert not np.array_equal(inbound_position, np.asarray(static.inbound_position))
    receiving_ids = np.asarray(static.defense_ids, dtype=np.int32)
    receiving_positions = np.asarray(state.positions)[0, receiving_ids]
    inbound_deltas = receiving_positions - inbound_position[None, :]
    inbound_distances = (
        np.abs(inbound_deltas[:, 0])
        + np.abs(inbound_deltas[:, 1])
        + np.abs(inbound_deltas[:, 0] + inbound_deltas[:, 1])
    ) // 2
    inbounder_slot = int(np.flatnonzero(receiving_ids == inbounder)[0])
    assert inbound_distances[inbounder_slot] == np.min(inbound_distances)
    assert int(np.asarray(out.check_opportunity)[0]) == 0
    np.testing.assert_allclose(
        np.asarray(out.game_reward)[0],
        np.asarray([-0.5, -0.5, 0.5, 0.5], dtype=np.float32),
    )


def test_check_transition_relocates_occupied_check_cell_with_seeded_tie_break():
    static = _multi_possession_static()
    state, occupant, _ = _check_state(static)
    positions = np.asarray(state.positions, dtype=np.int32).copy()
    positions[0, occupant] = np.asarray(static.check_position, dtype=np.int32)
    state = state._replace(positions=jnp.asarray(positions, dtype=jnp.int32))
    relocated = _prepare_check_positions_single(
        static,
        jax.tree_util.tree_map(lambda value: value[0], state),
        jax.random.PRNGKey(321),
        jax,
        jnp,
    )
    assert not np.array_equal(np.asarray(relocated)[occupant], np.asarray(static.check_position))


def test_baseline_inbound_remains_default_made_basket_restart():
    static = _multi_possession_static()
    assert int(np.asarray(static.made_basket_restart_mode)) == MADE_BASKET_RESTART_BASELINE_INBOUND


def test_made_basket_enters_configured_check_phase_with_full_frozen_clock():
    static = _multi_possession_static(possession_limit=2)._replace(
        made_basket_restart_mode=jnp.asarray(MADE_BASKET_RESTART_CHECK, dtype=jnp.int8),
        check_deadline_steps=jnp.asarray(5, dtype=jnp.int32),
    )
    state = reset_batch_minimal(
        static,
        jax.random.split(jax.random.PRNGKey(322), 1),
        jax,
        jnp,
    )._replace(
        layup_pct=jnp.ones((1, 4), dtype=jnp.float32),
        three_pt_pct=jnp.ones((1, 4), dtype=jnp.float32),
        clearance_achieved=jnp.asarray([1], dtype=jnp.int8),
    )
    prior_offense = int(np.asarray(state.offense_team)[0])
    out = _shoot(static, state, seed=323)

    assert int(np.asarray(out.shot_success)[0]) == 1
    assert int(np.asarray(out.state.game_phase)[0]) == GAME_PHASE_AWAITING_CHECK
    assert int(np.asarray(out.state.offense_team)[0]) == 1 - prior_offense
    assert int(np.asarray(out.state.ball_holder)[0]) == -1
    assert int(np.asarray(out.state.check_steps_remaining)[0]) == 5
    assert int(np.asarray(out.state.shot_clock)[0]) == 24
    assert int(np.asarray(out.check_opportunity)[0]) == 1
    assert not np.any(
        np.all(
            np.asarray(out.state.positions)[0]
            == np.asarray(static.check_position)[None, :],
            axis=-1,
        )
    )


def test_check_observation_encodes_loose_ball_phase_clock_and_distance():
    static = _multi_possession_static()._replace(
        made_basket_restart_mode=jnp.asarray(MADE_BASKET_RESTART_CHECK, dtype=jnp.int8),
        check_deadline_steps=jnp.asarray(5, dtype=jnp.int32),
    )
    state, _, _ = _check_state(static, countdown=3)
    players, globals_vec, _ = build_token_observation_components_batch(
        static,
        state,
        jnp.asarray([1.0], dtype=jnp.float32),
        jnp,
        multi_possession_features=True,
    )

    assert float(np.asarray(globals_vec)[0, -2]) == pytest.approx(2.0)
    assert float(np.asarray(globals_vec)[0, -1]) == pytest.approx(3.0 / 5.0)
    assert np.all(np.asarray(players)[0, :, 3] == 0.0)
    assert np.any(np.asarray(players)[0, :, 11] > 0.0)


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
    return reentry_out.state


def test_possession_live_step_counter_excludes_inbounds_and_restarts_offensive_intent():
    static = _multi_possession_static(possession_limit=3)._replace(
        enable_intent_learning=jnp.asarray(1, dtype=jnp.int8),
        intent_null_prob=jnp.asarray(0.0, dtype=jnp.float32),
    )
    state = reset_batch_minimal(
        static,
        jax.random.split(jax.random.PRNGKey(201), 1),
        jax,
        jnp,
    )._replace(clearance_achieved=jnp.asarray([1], dtype=jnp.int8))

    # The opening jump-ball winner begins an active offensive segment.
    assert int(np.asarray(state.intent_active)[0]) == 1
    assert int(np.asarray(state.intent_age)[0]) == 0

    live_out = step_batch_minimal(
        static,
        state,
        _noops(state),
        jax.random.split(jax.random.PRNGKey(202), 1),
        jax,
        jnp,
    )
    assert int(np.asarray(live_out.state.live_possession_steps)[0]) == 1
    assert int(np.asarray(live_out.completed_possession_live_steps)[0]) == 0

    made_out = _shoot(static, live_out.state, seed=203)
    assert int(np.asarray(made_out.shot_success)[0]) == 1
    assert int(np.asarray(made_out.possession_ended)[0]) == 1
    # The terminal live shot is included in the finished possession's count.
    assert int(np.asarray(made_out.completed_possession_live_steps)[0]) == 2
    assert int(np.asarray(made_out.state.live_possession_steps)[0]) == 0
    assert int(np.asarray(made_out.state.game_phase)[0]) == GAME_PHASE_AWAITING_INBOUND
    # The incoming offense has a fresh segment while it moves during the inbound.
    assert int(np.asarray(made_out.state.intent_active)[0]) == 1
    assert int(np.asarray(made_out.state.intent_age)[0]) == 0

    inbound_timeout_state = made_out.state._replace(
        inbound_steps_remaining=jnp.asarray([1], dtype=jnp.int32)
    )
    inbound_timeout_out = step_batch_minimal(
        static,
        inbound_timeout_state,
        _noops(inbound_timeout_state),
        jax.random.split(jax.random.PRNGKey(204), 1),
        jax,
        jnp,
    )
    assert int(np.asarray(inbound_timeout_out.possession_ended)[0]) == 1
    # A possession that ends before live play contributes no live ticks.
    assert int(np.asarray(inbound_timeout_out.completed_possession_live_steps)[0]) == 0
    assert int(np.asarray(inbound_timeout_out.state.intent_active)[0]) == 1
    assert int(np.asarray(inbound_timeout_out.state.intent_age)[0]) == 0

    _, possession_start, _, _, _, _, applied, _ = _selector_segment_application_masks(
        made_out.state,
        alpha_used=jnp.asarray([True]),
        multiselect_enabled=jnp.asarray(False),
        completed_pass_boundary=jnp.asarray([False]),
        offensive_rebound_boundary=jnp.asarray([False]),
        selector_min_play_steps=1,
        jnp=jnp,
    )
    assert bool(np.asarray(possession_start)[0]) is True
    assert bool(np.asarray(applied)[0]) is True


def test_completed_possession_pace_summary_uses_boundary_counters():
    metrics = summarize_completed_possession_pace(
        possession_ended=np.asarray([[0, 1], [1, 1]], dtype=np.int8),
        completed_possession_live_steps=np.asarray([[0, 2], [0, 5]], dtype=np.int32),
    )
    assert metrics == {
        "completed_possession_count": 3,
        "completed_possession_live_steps": 7,
        "mean_live_steps_per_completed_possession": pytest.approx(7.0 / 3.0),
    }


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
        # This fixture is exercising a made-basket inbound for either team,
        # rather than a pre-clearance shooting violation.
        clearance_achieved=jnp.ones_like(state.clearance_achieved),
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


def test_inbound_countdown_moves_everyone_else_and_freezes_shot_and_lane_clocks():
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

    assert int(np.asarray(out.state.shot_clock)[0]) == 24
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
    assert int(np.asarray(out.state.shot_clock)[0]) == 24
    assert int(np.asarray(out.state.inbound_steps_remaining)[0]) == 0
    np.testing.assert_array_equal(
        np.asarray(out.state.positions)[0, receiver],
        np.asarray(static.basket_position),
    )
    post_masks = np.asarray(
        build_action_masks_batch(static, out.state, jnp), dtype=np.int8
    )
    # Shooting remains selectable before clearance; #21 resolves the attempt
    # as a violation instead of silently masking it.
    assert post_masks[0, receiver, ActionType.SHOOT.value] == 1

    # The inbound pass establishes live play without spending a shot-clock
    # tick.  The next live action is the first one that consumes the clock.
    live_out = step_batch_minimal(
        static,
        out.state,
        _noops(out.state),
        jax.random.split(jax.random.PRNGKey(77), 1),
        jax,
        jnp,
    )
    assert int(np.asarray(live_out.state.shot_clock)[0]) == 23


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


def test_rebound_eligibility_requires_an_inbounder_to_be_on_court_after_shot_movement():
    def _live_state_with_waiting_inbounder(*, seed: int):
        base_static = _multi_possession_static(possession_limit=4)
        inbound_state = _inbound_state(base_static, seed=seed)
        inbounder = int(np.asarray(inbound_state.inbound_player)[0])
        receiver = int(np.asarray(base_static.offense_ids)[1])
        passed = step_batch_minimal(
            base_static,
            inbound_state,
            _noops(inbound_state).at[0, inbounder].set(PASS_ACTION_START),
            jax.random.split(jax.random.PRNGKey(seed + 1), 1),
            jax,
            jnp,
        )
        assert int(np.asarray(passed.completed_pass)[0]) == 1
        assert int(np.asarray(passed.state.game_phase)[0]) == GAME_PHASE_LIVE

        return base_static, passed.state._replace(
            clearance_achieved=jnp.asarray([1], dtype=jnp.int8),
            layup_pct=jnp.zeros_like(passed.state.layup_pct),
            three_pt_pct=jnp.zeros_like(passed.state.three_pt_pct),
            dunk_pct=jnp.zeros_like(passed.state.dunk_pct),
        ), inbounder, receiver

    def _static_with_single_target(base_static, target_cell_idx: int):
        target_probs = np.zeros(
            np.asarray(base_static.rebound_target_probs).shape,
            dtype=np.float32,
        )
        target_probs[:, :, target_cell_idx] = 1.0
        return base_static._replace(
            enable_rebounds=jnp.asarray(1, dtype=jnp.int8),
            rebound_target_probs=jnp.asarray(target_probs, dtype=jnp.float32),
            rebound_target_uniform_mix=jnp.asarray(0.0, dtype=jnp.float32),
            rebound_contest_mode=jnp.asarray(REBOUND_CONTEST_MODE_LOCAL, dtype=jnp.int32),
            rebound_contest_radius=jnp.asarray(0, dtype=jnp.int32),
        )

    # A waiting inbounder shares no court cell with the rebound target. Before
    # this guard, the safe lookup fallback incorrectly treated that player as
    # occupying cell zero and made them the sole local-contest candidate.
    base_static, waiting_state, inbounder, receiver = _live_state_with_waiting_inbounder(seed=130)
    external_target_idx = 0
    external_static = _static_with_single_target(base_static, external_target_idx)
    coords = np.asarray(external_static.cell_coords)
    positions = np.asarray(waiting_state.positions).copy()
    available_cells = [idx for idx in range(len(coords)) if idx != external_target_idx]
    for player_id, cell_idx in zip(
        [pid for pid in range(positions.shape[1]) if pid != inbounder],
        available_cells[: positions.shape[1] - 1],
        strict=True,
    ):
        positions[0, player_id] = coords[cell_idx]
    waiting_state = waiting_state._replace(positions=jnp.asarray(positions, dtype=jnp.int32))
    waiting_out = _shoot(external_static, waiting_state, seed=132)

    assert int(np.asarray(waiting_out.rebound_attempt)[0]) == 1
    assert int(np.asarray(waiting_out.rebound_winner)[0]) != inbounder

    # Re-entering on the same simultaneous shot action puts the inbounder on
    # court before rebound sampling, so they may contest normally.
    base_static, reentry_state, inbounder, receiver = _live_state_with_waiting_inbounder(seed=140)
    basket_idx = int(
        np.flatnonzero(
            np.all(
                np.asarray(base_static.cell_coords) == np.asarray(base_static.basket_position),
                axis=1,
            )
        )[0]
    )
    reentry_static = _static_with_single_target(base_static, basket_idx)
    coords = np.asarray(reentry_static.cell_coords)
    positions = np.asarray(reentry_state.positions).copy()
    available_cells = [idx for idx in range(len(coords)) if idx != basket_idx]
    for player_id, cell_idx in zip(
        [pid for pid in range(positions.shape[1]) if pid != inbounder],
        available_cells[: positions.shape[1] - 1],
        strict=True,
    ):
        positions[0, player_id] = coords[cell_idx]
    reentry_state = reentry_state._replace(positions=jnp.asarray(positions, dtype=jnp.int32))
    masks = np.asarray(build_action_masks_batch(reentry_static, reentry_state, jnp), dtype=np.int8)
    reentry_moves = np.flatnonzero(
        masks[0, inbounder, MOVE_ACTION_START:MOVE_ACTION_END]
    )
    assert reentry_moves.size == 1
    actions = _noops(reentry_state)
    actions = actions.at[0, receiver].set(ActionType.SHOOT.value)
    actions = actions.at[0, inbounder].set(MOVE_ACTION_START + int(reentry_moves[0]))
    reentry_out = step_batch_minimal(
        reentry_static,
        reentry_state,
        actions,
        jax.random.split(jax.random.PRNGKey(142), 1),
        jax,
        jnp,
    )

    assert int(np.asarray(reentry_out.rebound_attempt)[0]) == 1
    assert int(np.asarray(reentry_out.rebound_winner)[0]) == inbounder


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


def test_multi_possession_reset_starts_with_a_full_shot_clock():
    static = _multi_possession_static()._replace(
        shot_clock_min=jnp.asarray(6, dtype=jnp.int32),
        shot_clock_max=jnp.asarray(24, dtype=jnp.int32),
    )

    state = reset_batch_minimal(
        static,
        jax.random.split(jax.random.PRNGKey(16), 8),
        jax,
        jnp,
    )

    np.testing.assert_array_equal(
        np.asarray(state.shot_clock, dtype=np.int32),
        np.full((8,), 24, dtype=np.int32),
    )


def test_multi_possession_reset_samples_independent_shooting_skills_for_both_teams():
    static = _multi_possession_static()._replace(
        base_layup_pct=jnp.asarray(0.60, dtype=jnp.float32),
        base_three_pt_pct=jnp.asarray(0.37, dtype=jnp.float32),
        base_dunk_pct=jnp.asarray(0.60, dtype=jnp.float32),
        layup_std=jnp.asarray(0.05, dtype=jnp.float32),
        three_pt_std=jnp.asarray(0.05, dtype=jnp.float32),
        dunk_std=jnp.asarray(0.30, dtype=jnp.float32),
    )
    state = reset_batch_minimal(
        static,
        jax.random.split(jax.random.PRNGKey(161), 4),
        jax,
        jnp,
    )
    team_ids = (
        np.asarray(static.offense_ids, dtype=np.int32),
        np.asarray(static.defense_ids, dtype=np.int32),
    )

    for values, mean in (
        (np.asarray(state.layup_pct), 0.60),
        (np.asarray(state.three_pt_pct), 0.37),
        (np.asarray(state.dunk_pct), 0.60),
    ):
        assert np.all(values >= 0.01)
        assert np.all(values <= 0.99)
        assert np.any(np.abs(values - mean) > 1.0e-5)
        for ids in team_ids:
            assert np.ptp(values[:, ids]) > 1.0e-5


def test_multi_possession_reset_is_neutral_before_the_jump_ball_and_handoffs_after_terminal_shot():
    # One completed possession per team means two total possession endings.
    static = _multi_possession_static(possession_limit=1)
    reset_keys = jax.random.split(jax.random.PRNGKey(17), 4)
    state = reset_batch_minimal(
        static,
        reset_keys,
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

    positions = np.asarray(state.positions, dtype=np.int32)
    cell_set = {tuple(cell) for cell in np.asarray(static.cell_coords, dtype=np.int32)}
    basket = tuple(np.asarray(static.basket_position, dtype=np.int32))
    for row_positions in positions:
        assert len({tuple(cell) for cell in row_positions}) == row_positions.shape[0]
        assert all(tuple(cell) in cell_set and tuple(cell) != basket for cell in row_positions)

    # A jump-ball winner has to clear only when their starting cell is inside
    # the arc. Starting beyond it is already a valid cleared-ball position.
    cell_index_by_coord = {
        tuple(cell): idx for idx, cell in enumerate(np.asarray(static.cell_coords))
    }
    holder_is_beyond_arc = np.asarray(
        [
            static.three_point_by_cell[cell_index_by_coord[tuple(row[holder])]]
            for row, holder in zip(positions, holders, strict=True)
        ],
        dtype=np.int8,
    )
    np.testing.assert_array_equal(
        np.asarray(state.clearance_achieved, dtype=np.int8),
        holder_is_beyond_arc,
    )

    # Multi-possession layout does not consult any legacy offense/defense spawn
    # configuration. Restrict those fields to an intentionally unusable setup;
    # the same seeded neutral jump-ball positions must be unchanged.
    spawn_biased_static = static._replace(
        offense_spawn_candidate_mask=jnp.zeros_like(static.offense_spawn_candidate_mask),
        defense_min_spawn_distance=jnp.asarray(999.0, dtype=jnp.float32),
        max_spawn_distance_enabled=jnp.asarray(1, dtype=jnp.int8),
        max_spawn_distance=jnp.asarray(0.0, dtype=jnp.float32),
        defender_spawn_distance=jnp.asarray(999.0, dtype=jnp.float32),
    )
    spawn_biased_state = reset_batch_minimal(
        spawn_biased_static,
        reset_keys,
        jax,
        jnp,
    )
    np.testing.assert_array_equal(
        np.asarray(spawn_biased_state.positions),
        positions,
    )

    state = state._replace(
        layup_pct=jnp.ones_like(state.layup_pct),
        three_pt_pct=jnp.ones_like(state.three_pt_pct),
        dunk_pct=jnp.ones_like(state.dunk_pct),
        # This portion of the test exercises the terminal-shot handoff, not
        # the separately asserted opening clearance obligation.
        clearance_achieved=jnp.ones_like(state.clearance_achieved),
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
    tied = np.asarray(final_out.state.team_a_score) == np.asarray(
        final_out.state.team_b_score
    )
    np.testing.assert_array_equal(np.asarray(final_out.done, dtype=bool), ~tied)
    np.testing.assert_array_equal(
        np.asarray(final_out.state.episode_ended, dtype=bool), ~tied
    )
    np.testing.assert_array_equal(
        np.asarray(final_out.state.overtime_round, dtype=np.int32),
        tied.astype(np.int32),
    )
    assert np.all(
        np.asarray(final_out.state.completed_possessions, dtype=np.int32) == 2
    )
    assert np.all(
        np.asarray(final_out.state.team_a_score) >= np.asarray(out.state.team_a_score)
    )
    assert np.all(
        np.asarray(final_out.state.team_b_score) >= np.asarray(out.state.team_b_score)
    )


def test_no_inbounds_ablation_directly_handoffs_dead_ball_restarts_to_receiving_team():
    sample_count = 128
    static = _multi_possession_static()._replace(
        multi_possession_use_inbounds=jnp.asarray(0, dtype=jnp.int8)
    )
    state = reset_batch_minimal(
        static,
        jax.random.split(jax.random.PRNGKey(201), sample_count),
        jax,
        jnp,
    )
    team_a_holder = int(np.asarray(static.offense_ids, dtype=np.int32)[0])
    state = state._replace(
        offense_team=jnp.full((sample_count,), TEAM_A, dtype=jnp.int8),
        starting_offense_team=jnp.full((sample_count,), TEAM_A, dtype=jnp.int8),
        ball_holder=jnp.full((sample_count,), team_a_holder, dtype=jnp.int32),
        layup_pct=jnp.ones_like(state.layup_pct),
        three_pt_pct=jnp.ones_like(state.three_pt_pct),
        dunk_pct=jnp.ones_like(state.dunk_pct),
        clearance_achieved=jnp.ones_like(state.clearance_achieved),
    )
    positions_before = np.asarray(state.positions).copy()

    out = _shoot(static, state, seed=202)

    assert np.all(np.asarray(out.possession_ended, dtype=np.int8) == 1)
    made_mask = (
        np.asarray(out.possession_end_reason, dtype=np.int32)
        == POSSESSION_END_MADE_BASKET
    )
    # Some initial cells are forced into a terminal miss by the shot geometry;
    # use the many made-basket rows to isolate the dead-ball transition.
    assert int(made_mask.sum()) >= sample_count // 2
    assert np.all(np.asarray(out.state.offense_team, dtype=np.int8)[made_mask] == TEAM_B)
    assert np.all(
        np.asarray(out.state.game_phase, dtype=np.int8)[made_mask] == GAME_PHASE_LIVE
    )
    assert np.all(np.asarray(out.state.inbound_team, dtype=np.int8)[made_mask] == -1)
    assert np.all(np.asarray(out.state.inbound_player, dtype=np.int32)[made_mask] == -1)
    assert np.all(
        np.asarray(out.state.inbound_steps_remaining, dtype=np.int32)[made_mask] == 0
    )
    assert np.all(np.asarray(out.state.shot_clock, dtype=np.int32)[made_mask] == 24)
    np.testing.assert_array_equal(
        np.asarray(out.state.positions)[made_mask], positions_before[made_mask]
    )

    receiving_ids = np.asarray(static.defense_ids, dtype=np.int32)
    holders = np.asarray(out.state.ball_holder, dtype=np.int32)[made_mask]
    assert set(holders.tolist()).issubset(set(receiving_ids.tolist()))
    # The handoff is intentionally uniform rather than roster-order based.
    first_receiver_share = float(np.mean(holders == receiving_ids[0]))
    assert first_receiver_share == pytest.approx(0.5, abs=0.15)
    cell_indices = {
        tuple(cell): index
        for index, cell in enumerate(np.asarray(static.cell_coords, dtype=np.int32))
    }
    expected_clearance = np.asarray(
        [
            static.three_point_by_cell[
                cell_indices[tuple(row[holder])]
            ]
            for row, holder in zip(
                np.asarray(out.state.positions)[made_mask],
                holders,
                strict=True,
            )
        ],
        dtype=np.int8,
    )
    np.testing.assert_array_equal(
        np.asarray(out.state.clearance_achieved, dtype=np.int8)[made_mask],
        expected_clearance,
    )

    # The same no-inbounds rule applies to ordinary dead-ball turnovers while
    # retaining every player's existing location.
    turnover_state = state._replace(
        shot_clock=jnp.ones_like(state.shot_clock),
    )
    turnover_positions = np.asarray(turnover_state.positions).copy()
    turnover_out = step_batch_minimal(
        static,
        turnover_state,
        _noops(turnover_state),
        jax.random.split(jax.random.PRNGKey(203), sample_count),
        jax,
        jnp,
    )
    assert np.all(np.asarray(turnover_out.turnover, dtype=np.int8) == 1)
    assert np.all(np.asarray(turnover_out.state.game_phase, dtype=np.int8) == GAME_PHASE_LIVE)
    assert np.all(np.asarray(turnover_out.state.inbound_player, dtype=np.int32) == -1)
    assert set(np.asarray(turnover_out.state.ball_holder, dtype=np.int32).tolist()).issubset(
        set(receiving_ids.tolist())
    )
    np.testing.assert_array_equal(np.asarray(turnover_out.state.positions), turnover_positions)


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


def _forced_pressure_turnover_state(static, *, defender_distances: tuple[int, int] = (1, 1)):
    """Place a Team B ball handler at the basket and two pressuring Team A defenders."""
    state = reset_batch_minimal(
        static,
        jax.random.split(jax.random.PRNGKey(11), 1),
        jax,
        jnp,
    )
    team_a = np.asarray(static.offense_ids, dtype=np.int32)
    team_b = np.asarray(static.defense_ids, dtype=np.int32)
    holder = int(team_b[0])
    positions = np.asarray(state.positions).copy()
    coords = np.asarray(static.cell_coords, dtype=np.int32)
    basket = np.asarray(static.basket_position, dtype=np.int32)
    delta_q = coords[:, 0] - basket[0]
    delta_r = coords[:, 1] - basket[1]
    cell_distances = (np.abs(delta_q) + np.abs(delta_r) + np.abs(delta_q + delta_r)) // 2

    positions[0, holder] = basket
    used = {tuple(basket.tolist())}
    for defender, distance in zip(team_a.tolist(), defender_distances, strict=True):
        cell_idx = next(
            int(idx)
            for idx in np.flatnonzero(cell_distances == distance)
            if tuple(coords[idx].tolist()) not in used
        )
        positions[0, defender] = coords[cell_idx]
        used.add(tuple(coords[cell_idx].tolist()))
    for player_id in team_b[1:].tolist():
        cell_idx = next(
            int(idx)
            for idx in range(coords.shape[0])
            if tuple(coords[idx].tolist()) not in used
        )
        positions[0, player_id] = coords[cell_idx]
        used.add(tuple(coords[cell_idx].tolist()))

    return state._replace(
        positions=jnp.asarray(positions, dtype=jnp.int32),
        offense_team=jnp.asarray([TEAM_B], dtype=jnp.int8),
        starting_offense_team=jnp.asarray([TEAM_B], dtype=jnp.int8),
        ball_holder=jnp.asarray([holder], dtype=jnp.int32),
        clearance_achieved=jnp.asarray([0], dtype=jnp.int8),
    )


def test_forced_pressure_turnover_becomes_a_live_steal_that_must_clear():
    static = _multi_possession_static()._replace(
        defender_pressure_distance=jnp.asarray(100.0, dtype=jnp.float32),
        defender_pressure_turnover_chance=jnp.asarray(1.0, dtype=jnp.float32),
        defender_pressure_decay_lambda=jnp.asarray(0.0, dtype=jnp.float32),
    )
    state = _forced_pressure_turnover_state(static)
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
    assert int(np.asarray(out.turnover_reason)[0]) == TURNOVER_REASON_DEFENDER_PRESSURE
    stealer = int(np.asarray(out.steal_player)[0])
    assert stealer in set(np.asarray(static.offense_ids, dtype=np.int32).tolist())
    assert int(np.asarray(out.possession_ended)[0]) == 1
    assert int(np.asarray(out.possession_end_reason)[0]) == POSSESSION_END_TURNOVER
    assert not bool(np.asarray(out.done)[0])
    assert int(np.asarray(out.state.completed_possessions)[0]) == 1
    assert int(np.asarray(out.state.offense_team)[0]) == TEAM_A
    assert int(np.asarray(out.state.game_phase)[0]) == GAME_PHASE_LIVE
    assert int(np.asarray(out.state.inbound_team)[0]) == -1
    assert int(np.asarray(out.state.inbound_player)[0]) == -1
    assert int(np.asarray(out.state.ball_holder)[0]) == stealer
    assert int(np.asarray(out.state.inbound_steps_remaining)[0]) == 0
    assert int(np.asarray(out.state.clearance_achieved)[0]) == 0
    assert int(np.asarray(out.clearance_event)[0]) == 0


def test_pressure_stealer_is_sampled_from_softmax_pressure_strengths():
    # The basket holder makes both defenders valid irrespective of direction.
    # At distances one and two with decay ln(2), their pressure strengths are
    # 1.0 and 0.5, so softmax(log(strength)) assigns the first defender 2/3.
    static = _multi_possession_static()._replace(
        defender_pressure_distance=jnp.asarray(100.0, dtype=jnp.float32),
        defender_pressure_turnover_chance=jnp.asarray(1.0, dtype=jnp.float32),
        defender_pressure_decay_lambda=jnp.asarray(np.log(2.0), dtype=jnp.float32),
    )
    single_state = _forced_pressure_turnover_state(
        static,
        defender_distances=(1, 2),
    )
    sample_count = 1024
    state = jax.tree_util.tree_map(
        lambda value: jnp.repeat(value, sample_count, axis=0),
        single_state,
    )
    out = step_batch_minimal(
        static,
        state,
        _noops(state),
        jax.random.split(jax.random.PRNGKey(13), sample_count),
        jax,
        jnp,
    )

    team_a = np.asarray(static.offense_ids, dtype=np.int32)
    stealers = np.asarray(out.steal_player, dtype=np.int32)
    assert set(stealers.tolist()) == set(team_a.tolist())
    nearer_share = float(np.mean(stealers == team_a[0]))
    assert nearer_share == pytest.approx(2.0 / 3.0, abs=0.07)


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
        # Keep the completed regulation game out of overtime; this test is
        # about stable-roster role switches and score bookkeeping.
        team_a_score=jnp.asarray([100.0], dtype=jnp.float32),
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

    expected_team_a_score = 100.0
    expected_team_b_score = 0.0
    for possession_number, seed in enumerate((26, 27, 28, 29, 30, 31), start=1):
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
        if possession_number < 6:
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

        if possession_number < 6:
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

    assert int(np.asarray(out.state.team_a_completed_possessions)[0]) == 3
    assert int(np.asarray(out.state.team_b_completed_possessions)[0]) == 3
    assert int(np.asarray(out.state.completed_possessions)[0]) == 6


def test_clearance_requires_the_curved_arc_not_short_corner_three_point_cells():
    static = _multi_possession_static(
        players=2,
        possession_limit=1,
        court_rows=9,
        court_cols=8,
        three_point_distance=4.25,
        three_point_short_distance=3.0,
    )
    three_mask = np.asarray(static.three_point_by_cell, dtype=np.int8)
    clearance_mask = np.asarray(static.clearance_by_cell, dtype=np.int8)
    short_corner_indices = np.flatnonzero((three_mask == 1) & (clearance_mask == 0))
    curved_arc_indices = np.flatnonzero(clearance_mask == 1)
    assert short_corner_indices.size > 0
    assert curved_arc_indices.size > 0

    state = reset_batch_minimal(
        static,
        jax.random.split(jax.random.PRNGKey(119), 1),
        jax,
        jnp,
    )
    holder = int(np.asarray(static.offense_ids)[0])
    positions = np.asarray(state.positions).copy()
    positions[0, holder] = np.asarray(static.cell_coords)[short_corner_indices[0]]
    short_corner_state = state._replace(
        positions=jnp.asarray(positions, dtype=jnp.int32),
        offense_team=jnp.asarray([TEAM_A], dtype=jnp.int8),
        ball_holder=jnp.asarray([holder], dtype=jnp.int32),
        clearance_achieved=jnp.asarray([0], dtype=jnp.int8),
    )
    short_corner_out = step_batch_minimal(
        static,
        short_corner_state,
        _noops(short_corner_state),
        jax.random.split(jax.random.PRNGKey(120), 1),
        jax,
        jnp,
    )
    assert int(np.asarray(short_corner_out.state.clearance_achieved)[0]) == 0

    positions[0, holder] = np.asarray(static.cell_coords)[curved_arc_indices[0]]
    curved_arc_state = short_corner_state._replace(
        positions=jnp.asarray(positions, dtype=jnp.int32)
    )
    curved_arc_out = step_batch_minimal(
        static,
        curved_arc_state,
        _noops(curved_arc_state),
        jax.random.split(jax.random.PRNGKey(121), 1),
        jax,
        jnp,
    )
    assert int(np.asarray(curved_arc_out.state.clearance_achieved)[0]) == 1


def _tied_regulation_boundary_state(static, *, seed: int):
    state = reset_batch_minimal(
        static,
        jax.random.split(jax.random.PRNGKey(seed), 1),
        jax,
        jnp,
    )
    holder = int(np.asarray(static.defense_ids)[0])
    positions = np.asarray(state.positions).copy()
    positions[0, holder] = np.asarray(static.basket_position)
    return state._replace(
        positions=jnp.asarray(positions, dtype=jnp.int32),
        starting_offense_team=jnp.asarray([TEAM_A], dtype=jnp.int8),
        offense_team=jnp.asarray([TEAM_B], dtype=jnp.int8),
        ball_holder=jnp.asarray([holder], dtype=jnp.int32),
        team_a_completed_possessions=jnp.asarray([1], dtype=jnp.int32),
        team_b_completed_possessions=jnp.asarray([0], dtype=jnp.int32),
        completed_possessions=jnp.asarray([1], dtype=jnp.int32),
        team_a_score=jnp.asarray([2.0], dtype=jnp.float32),
        team_b_score=jnp.asarray([0.0], dtype=jnp.float32),
        clearance_achieved=jnp.asarray([1], dtype=jnp.int8),
        layup_pct=jnp.ones_like(state.layup_pct),
        three_pt_pct=jnp.ones_like(state.three_pt_pct),
        dunk_pct=jnp.ones_like(state.dunk_pct),
    )


def test_paired_overtime_waits_for_both_possessions_and_emits_one_winner_bonus():
    static = _multi_possession_static(possession_limit=1)._replace(
        multi_possession_use_inbounds=jnp.asarray(0, dtype=jnp.int8),
        multi_possession_reward_mode=jnp.asarray(
            MULTI_POSSESSION_REWARD_SCORING_EVENTS, dtype=jnp.int32
        ),
        game_winner_reward=jnp.asarray(5.0, dtype=jnp.float32),
    )
    regulation = _shoot(
        static,
        _tied_regulation_boundary_state(static, seed=122),
        seed=123,
    )
    assert not bool(np.asarray(regulation.done)[0])
    assert int(np.asarray(regulation.state.overtime_round)[0]) == 1
    assert int(np.asarray(regulation.state.overtime_possessions_completed)[0]) == 0
    assert int(np.asarray(regulation.state.offense_team)[0]) == TEAM_A
    np.testing.assert_array_equal(np.asarray(regulation.winner_reward), 0.0)

    first_holder = int(np.asarray(regulation.state.ball_holder)[0])
    first_positions = np.asarray(regulation.state.positions).copy()
    first_positions[0, first_holder] = np.asarray(static.basket_position)
    first_overtime = _shoot(
        static,
        regulation.state._replace(
            positions=jnp.asarray(first_positions, dtype=jnp.int32),
            clearance_achieved=jnp.asarray([1], dtype=jnp.int8),
            layup_pct=jnp.ones_like(regulation.state.layup_pct),
        ),
        seed=124,
    )
    assert not bool(np.asarray(first_overtime.done)[0])
    assert int(np.asarray(first_overtime.state.overtime_possessions_completed)[0]) == 1
    assert int(np.asarray(first_overtime.state.offense_team)[0]) == TEAM_B
    np.testing.assert_array_equal(np.asarray(first_overtime.winner_reward), 0.0)

    final_overtime = _shoot(
        static,
        first_overtime.state._replace(
            clearance_achieved=jnp.asarray([0], dtype=jnp.int8)
        ),
        seed=125,
    )
    assert bool(np.asarray(final_overtime.done)[0])
    assert int(np.asarray(final_overtime.state.overtime_possessions_completed)[0]) == 2
    team_a_bonus = float(
        np.asarray(final_overtime.winner_reward)[
            0, np.asarray(static.offense_ids, dtype=np.int32)
        ].sum()
    )
    team_b_bonus = float(
        np.asarray(final_overtime.winner_reward)[
            0, np.asarray(static.defense_ids, dtype=np.int32)
        ].sum()
    )
    assert team_a_bonus == pytest.approx(5.0)
    assert team_b_bonus == pytest.approx(-5.0)
    assert float(
        np.asarray(final_overtime.game_reward)[
            0, np.asarray(static.offense_ids, dtype=np.int32)
        ].sum()
    ) == pytest.approx(0.0)
    assert float(
        np.asarray(final_overtime.rewards)[
            0, np.asarray(static.offense_ids, dtype=np.int32)
        ].sum()
    ) == pytest.approx(5.0)

    repeated = step_batch_minimal(
        static,
        final_overtime.state,
        _noops(final_overtime.state),
        jax.random.split(jax.random.PRNGKey(126), 1),
        jax,
        jnp,
    )
    np.testing.assert_array_equal(np.asarray(repeated.winner_reward), 0.0)


def test_tied_overtime_pair_continues_and_alternates_the_next_round_starter():
    static = _multi_possession_static(
        possession_limit=1,
        overtime_round_cap=2,
    )._replace(
        multi_possession_use_inbounds=jnp.asarray(0, dtype=jnp.int8)
    )
    regulation = _shoot(
        static,
        _tied_regulation_boundary_state(static, seed=127),
        seed=128,
    )
    first = _shoot(
        static,
        regulation.state._replace(clearance_achieved=jnp.asarray([0], dtype=jnp.int8)),
        seed=129,
    )
    second = _shoot(
        static,
        first.state._replace(clearance_achieved=jnp.asarray([0], dtype=jnp.int8)),
        seed=130,
    )

    assert not bool(np.asarray(second.done)[0])
    assert int(np.asarray(second.state.overtime_round)[0]) == 2
    assert int(np.asarray(second.state.overtime_possessions_completed)[0]) == 0
    assert int(np.asarray(second.state.overtime_starting_team)[0]) == TEAM_B
    assert int(np.asarray(second.state.offense_team)[0]) == TEAM_B
    assert float(np.asarray(second.state.team_a_score)[0]) == pytest.approx(2.0)
    assert float(np.asarray(second.state.team_b_score)[0]) == pytest.approx(2.0)


def test_default_overtime_cap_equals_possession_limit_and_capped_tie_terminates():
    static = _multi_possession_static(possession_limit=1)._replace(
        multi_possession_use_inbounds=jnp.asarray(0, dtype=jnp.int8),
        multi_possession_reward_mode=jnp.asarray(
            MULTI_POSSESSION_REWARD_SCORING_EVENTS,
            dtype=jnp.int32,
        ),
        game_winner_reward=jnp.asarray(5.0, dtype=jnp.float32),
    )
    regulation_state = _tied_regulation_boundary_state(static, seed=227)
    assert int(np.asarray(regulation_state.episode_possession_limit)[0]) == 1
    assert int(np.asarray(regulation_state.episode_overtime_round_cap)[0]) == 1

    regulation = _shoot(static, regulation_state, seed=228)
    first = _shoot(
        static,
        regulation.state._replace(clearance_achieved=jnp.asarray([0], dtype=jnp.int8)),
        seed=229,
    )
    tied_final = _shoot(
        static,
        first.state._replace(clearance_achieved=jnp.asarray([0], dtype=jnp.int8)),
        seed=230,
    )

    assert bool(np.asarray(tied_final.done)[0])
    assert int(np.asarray(tied_final.state.overtime_round)[0]) == 1
    assert int(np.asarray(tied_final.state.overtime_possessions_completed)[0]) == 2
    assert float(np.asarray(tied_final.state.team_a_score)[0]) == pytest.approx(2.0)
    assert float(np.asarray(tied_final.state.team_b_score)[0]) == pytest.approx(2.0)
    np.testing.assert_array_equal(np.asarray(tied_final.winner_reward), 0.0)


def test_explicit_overtime_cap_is_copied_into_each_reset_episode():
    static = _multi_possession_static(
        possession_limit=3,
        overtime_round_cap=7,
    )
    state = reset_batch_minimal(
        static,
        jax.random.split(jax.random.PRNGKey(231), 2),
        jax,
        jnp,
    )

    np.testing.assert_array_equal(
        np.asarray(state.episode_possession_limit),
        np.full((2,), 3, dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.asarray(state.episode_overtime_round_cap),
        np.full((2,), 7, dtype=np.int32),
    )


@pytest.mark.parametrize(
    ("finishing_team", "team_a_before", "team_b_before"),
    [(TEAM_A, 24, 25), (TEAM_B, 25, 24)],
)
def test_25_possession_limit_is_per_team_and_terminates_at_50_combined(
    finishing_team,
    team_a_before,
    team_b_before,
):
    static = _multi_possession_static(possession_limit=25)
    state = reset_batch_minimal(
        static,
        jax.random.split(jax.random.PRNGKey(125 + finishing_team), 1),
        jax,
        jnp,
    )
    holder = int(
        np.asarray(
            static.offense_ids if finishing_team == TEAM_A else static.defense_ids
        )[0]
    )
    state = state._replace(
        offense_team=jnp.asarray([finishing_team], dtype=jnp.int8),
        ball_holder=jnp.asarray([holder], dtype=jnp.int32),
        team_a_completed_possessions=jnp.asarray([team_a_before], dtype=jnp.int32),
        team_b_completed_possessions=jnp.asarray([team_b_before], dtype=jnp.int32),
        completed_possessions=jnp.asarray([49], dtype=jnp.int32),
        clearance_achieved=jnp.asarray([1], dtype=jnp.int8),
        layup_pct=jnp.ones_like(state.layup_pct),
        three_pt_pct=jnp.ones_like(state.three_pt_pct),
        dunk_pct=jnp.ones_like(state.dunk_pct),
    )

    out = _shoot(static, state, seed=130 + finishing_team)

    assert bool(np.asarray(out.done)[0])
    assert int(np.asarray(out.state.team_a_completed_possessions)[0]) == 25
    assert int(np.asarray(out.state.team_b_completed_possessions)[0]) == 25
    assert int(np.asarray(out.state.completed_possessions)[0]) == 50


def test_defensive_lane_violation_awards_technical_and_sideline_restart_same_possession():
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
        shot_clock=jnp.asarray([7], dtype=jnp.int32),
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
    assert int(np.asarray(out.possession_ended)[0]) == 0
    assert (
        int(np.asarray(out.possession_end_reason)[0])
        == POSSESSION_END_DEFENSIVE_VIOLATION
    )
    assert int(np.asarray(out.state.completed_possessions)[0]) == 0
    assert int(np.asarray(out.state.offense_team)[0]) == TEAM_B
    assert float(np.asarray(out.state.team_a_score)[0]) == pytest.approx(0.0)
    assert float(np.asarray(out.state.team_b_score)[0]) == pytest.approx(1.0)
    assert int(np.asarray(out.state.game_phase)[0]) == GAME_PHASE_AWAITING_INBOUND
    assert int(np.asarray(out.state.shot_clock)[0]) == 14
    assert int(np.asarray(out.state.inbound_team)[0]) == TEAM_B
    assert int(np.asarray(out.state.ball_holder)[0]) == int(
        np.asarray(out.state.inbound_player)[0]
    )
    assert (
        int(np.asarray(out.state.inbound_reason)[0])
        == POSSESSION_END_DEFENSIVE_VIOLATION
    )
    inbounder = int(np.asarray(out.state.inbound_player)[0])
    inbound_position = np.asarray(out.state.positions)[0, inbounder]
    assert any(
        np.array_equal(inbound_position, position)
        for position in np.asarray(static.technical_inbound_positions)
    )
    assert not np.array_equal(inbound_position, np.asarray(static.inbound_position))

    # The one-step threshold is intentional for triggering the first call.
    # Restore the ordinary count before exercising entry after the inbound,
    # otherwise the unmoved defender immediately incurs a second technical.
    resumed_static = static._replace(
        three_second_max_steps=jnp.asarray(3, dtype=jnp.int32)
    )
    live_state = _complete_inbound(resumed_static, out.state, seed=31)
    reentry_masks = np.asarray(build_action_masks_batch(resumed_static, live_state, jnp))
    legal_moves = np.flatnonzero(
        reentry_masks[0, inbounder, MOVE_ACTION_START:MOVE_ACTION_END]
    )
    assert legal_moves.size > 0
    reentry_out = step_batch_minimal(
        resumed_static,
        live_state,
        _noops(live_state).at[0, inbounder].set(
            MOVE_ACTION_START + int(legal_moves[0])
        ),
        jax.random.split(jax.random.PRNGKey(32), 1),
        jax,
        jnp,
    )
    assert int(np.asarray(reentry_out.state.inbound_player)[0]) == -1

    # The same technical interruption becomes an immediate live restart in
    # the no-inbounds ablation. It preserves the offense, score, clock, and
    # positions, but assigns the ball to a random player on that offense.
    no_inbounds_static = static._replace(
        multi_possession_use_inbounds=jnp.asarray(0, dtype=jnp.int8)
    )
    positions_before = np.asarray(state.positions).copy()
    no_inbounds_out = step_batch_minimal(
        no_inbounds_static,
        state,
        _noops(state),
        jax.random.split(jax.random.PRNGKey(33), 1),
        jax,
        jnp,
    )
    assert int(np.asarray(no_inbounds_out.defensive_lane_violation)[0]) == 1
    assert int(np.asarray(no_inbounds_out.possession_ended)[0]) == 0
    assert int(np.asarray(no_inbounds_out.state.game_phase)[0]) == GAME_PHASE_LIVE
    assert int(np.asarray(no_inbounds_out.state.inbound_team)[0]) == -1
    assert int(np.asarray(no_inbounds_out.state.inbound_player)[0]) == -1
    assert int(np.asarray(no_inbounds_out.state.inbound_steps_remaining)[0]) == 0
    assert int(np.asarray(no_inbounds_out.state.shot_clock)[0]) == 14
    assert int(np.asarray(no_inbounds_out.state.ball_holder)[0]) in set(
        np.asarray(no_inbounds_static.defense_ids, dtype=np.int32).tolist()
    )
    np.testing.assert_array_equal(
        np.asarray(no_inbounds_out.state.positions), positions_before
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
        team_a_completed_possessions=jnp.asarray([0, 1, 0], dtype=jnp.int32),
        team_b_completed_possessions=jnp.asarray([1, 0, 0], dtype=jnp.int32),
        completed_possessions=jnp.asarray([1, 1, 0], dtype=jnp.int32),
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
        2,
        2,
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
