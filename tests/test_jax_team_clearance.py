from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from basketworld.domain import ActionType
from basketworld_jax.config import JaxEnvironmentConfig
from basketworld_jax.config.environment import normalize_runtime_environment_config
from basketworld_jax.env import minimal as env
from basketworld_jax.train.main import (
    _build_policy_spec,
    _checkpoint_observation_metadata,
    _compatible_env_config_for_resume,
    _jax_env_config_from_args,
    _restore_like_template,
    parse_args,
)


def _config(**kwargs):
    return JaxEnvironmentConfig(
        players=2,
        enable_multi_possession=True,
        require_all_offense_clear=True,
        multi_possession_limit=5,
        base_steal_rate=0.0,
        defender_pressure_turnover_chance=0.0,
        shot_pressure_enabled=False,
        illegal_defense_enabled=False,
        offensive_three_seconds_enabled=False,
        enable_phi_shaping=False,
        three_point_distance=4.0,
        **kwargs,
    )


def _reset(config=None, seed=73):
    static = env.build_kernel_static_from_config(config or _config(), jnp)
    state = env.reset_batch_minimal(
        static, jax.random.split(jax.random.PRNGKey(seed), 1), jax, jnp
    )
    return static, state


def _step(static, state, actions=None, seed=74):
    if actions is None:
        actions = jnp.zeros(state.positions.shape[:2], dtype=jnp.int32)
    # Exercise actual transitions eagerly here; compiled training/evaluation
    # coverage lives in the integration suites.
    with jax.disable_jit():
        out = env._step_single_minimal(
            static,
            jax.tree.map(lambda x: x[0], state),
            actions[0],
            jax.random.split(jax.random.PRNGKey(seed), 1)[0],
            jax,
            jnp,
            apply_game_clock=bool(int(static.episode_end_mode)),
        )
    return jax.tree.map(lambda x: x[None], out)


def _geometry_flags(static, state):
    cells = {
        tuple(p): bool(c)
        for p, c in zip(
            np.asarray(static.cell_coords), np.asarray(static.clearance_by_cell)
        )
    }
    team = int(state.offense_team[0])
    ids = np.asarray(static.offense_ids if team == env.TEAM_A else static.defense_ids)
    return np.asarray(
        [
            int(i in ids and cells.get(tuple(p), False))
            for i, p in enumerate(np.asarray(state.positions[0]))
        ]
    )


def _live(static, state, team=env.TEAM_A):
    # Use four distinct interior cells; all player flags are deliberately incomplete.
    cells = np.asarray(static.cell_coords)
    inside = cells[~np.asarray(static.clearance_by_cell, dtype=bool)][:4]
    holder = int(
        np.asarray(static.offense_ids if team == env.TEAM_A else static.defense_ids)[0]
    )
    return state._replace(
        positions=jnp.asarray(inside[None], dtype=jnp.int32),
        offense_team=jnp.asarray([team], dtype=jnp.int8),
        ball_holder=jnp.asarray([holder], dtype=jnp.int32),
        game_phase=jnp.asarray([env.GAME_PHASE_LIVE], dtype=jnp.int8),
        clearance_achieved=jnp.ones((1,), dtype=jnp.int8),
        offense_clearance=jnp.zeros_like(state.offense_clearance),
    )


@pytest.mark.parametrize("players", [1, 5])
def test_opening_initializes_only_current_offense_from_arc_geometry(players):
    static, state = _reset(replace(_config(), players=players))
    np.testing.assert_array_equal(
        state.offense_clearance[0], _geometry_flags(static, state)
    )
    assert int(static.multi_possession_schema_version) == 6


@pytest.mark.parametrize("mode", ["possessions", "game_clock"])
@pytest.mark.parametrize("team", [env.TEAM_A, env.TEAM_B])
@pytest.mark.parametrize(
    "ball_clear,team_clear", [(True, False), (False, True), (False, False)]
)
def test_premature_shot_is_selectable_and_costs_one_opponent_point(
    mode, team, ball_clear, team_clear
):
    static, state = _reset(
        _config(episode_end_mode=mode, multi_possession_reward_mode="scoring_events")
    )
    state = _live(static, state, team)
    ids = static.offense_ids if team == env.TEAM_A else static.defense_ids
    flags = jnp.zeros_like(state.offense_clearance).at[0, ids].set(int(team_clear))
    state = state._replace(
        offense_clearance=flags,
        clearance_achieved=jnp.asarray([int(ball_clear)], dtype=jnp.int8),
    )
    holder = int(state.ball_holder[0])
    assert (
        int(
            env.build_action_masks_batch(static, state, jnp)[
                0, holder, ActionType.SHOOT.value
            ]
        )
        == 1
    )
    actions = (
        jnp.zeros(state.positions.shape[:2], dtype=jnp.int32)
        .at[0, holder]
        .set(ActionType.SHOOT.value)
    )
    out = _step(static, state, actions)
    assert int(out.turnover_reason[0]) == env.TURNOVER_REASON_CLEARANCE_VIOLATION
    assert int(out.turnover[0]) == 1
    assert int(out.shot_attempt[0]) == 0
    assert int(out.possession_ended[0]) == 1
    assert int(out.state.offense_team[0]) == 1 - team
    assert float(out.state.team_a_score[0]) == (1.0 if team == env.TEAM_B else 0.0)
    assert float(out.state.team_b_score[0]) == (1.0 if team == env.TEAM_A else 0.0)
    assert float(jnp.sum(out.rewards[0, ids])) == pytest.approx(-1.0)
    np.testing.assert_array_equal(
        out.state.offense_clearance[0], _geometry_flags(static, out.state)
    )


def _boundary(static):
    cells = np.asarray(static.cell_coords)
    cleared = np.asarray(static.clearance_by_cell, dtype=bool)
    lookup = {tuple(p): i for i, p in enumerate(cells)}
    for i, p in enumerate(cells):
        if cleared[i]:
            continue
        for d, delta in enumerate(np.asarray(static.hex_directions)):
            q = p + delta
            if tuple(q) in lookup and cleared[lookup[tuple(q)]]:
                return p, q, env.MOVE_ACTION_START + d
    raise AssertionError("No arc boundary")


@pytest.mark.parametrize(
    "phase",
    [env.GAME_PHASE_LIVE, env.GAME_PHASE_CHECK_SETUP, env.GAME_PHASE_AWAITING_CHECK],
)
def test_movement_latches_clearance_in_every_phase_and_survives_cut_inside(phase):
    static, state = _reset(_config(check_setup_steps=3))
    state = _live(static, state)
    inside, outside, move = _boundary(static)
    cells = np.asarray(static.cell_coords)
    other = [p for p in cells if tuple(p) not in {tuple(inside), tuple(outside)}][-3:]
    positions = np.asarray([other[0], inside, other[1], other[2]])
    state = state._replace(
        positions=jnp.asarray(positions[None]),
        game_phase=jnp.asarray([phase], dtype=jnp.int8),
        check_setup_steps_remaining=jnp.asarray([3], dtype=jnp.int32),
        check_steps_remaining=jnp.asarray([3], dtype=jnp.int32),
        ball_holder=jnp.asarray(
            [0 if phase == env.GAME_PHASE_LIVE else -1], dtype=jnp.int32
        ),
        check_team=jnp.asarray([env.TEAM_A], dtype=jnp.int8),
        offense_clearance=jnp.asarray([[1, 0, 0, 0]], dtype=jnp.int8),
    )
    actions = jnp.zeros((1, 4), dtype=jnp.int32).at[0, 1].set(move)
    out = _step(static, state, actions)
    np.testing.assert_array_equal(out.state.positions[0, 1], outside)
    assert int(out.state.offense_clearance[0, 1]) == 1
    reverse = next(
        i
        for i, delta in enumerate(np.asarray(static.hex_directions))
        if np.array_equal(outside + delta, inside)
    )
    back = _step(
        static,
        out.state,
        jnp.zeros((1, 4), dtype=jnp.int32)
        .at[0, 1]
        .set(env.MOVE_ACTION_START + reverse),
    )
    assert int(back.state.offense_clearance[0, 1]) == 1
    if phase == env.GAME_PHASE_CHECK_SETUP:
        assert int(out.state.check_setup_steps_remaining[0]) == 2
        assert int(out.state.shot_clock[0]) == int(state.shot_clock[0])


def test_last_player_movement_cannot_retroactively_legalize_shot():
    static, state = _reset()
    state = _live(static, state)
    inside, outside, move = _boundary(static)
    positions = np.asarray(state.positions).copy()
    positions[0, 1] = inside
    state = state._replace(
        positions=jnp.asarray(positions),
        offense_clearance=jnp.asarray([[1, 0, 0, 0]], dtype=jnp.int8),
    )
    actions = (
        jnp.zeros((1, 4), dtype=jnp.int32)
        .at[0, 0]
        .set(ActionType.SHOOT.value)
        .at[0, 1]
        .set(move)
    )
    out = _step(static, state, actions)
    assert int(out.turnover_reason[0]) == env.TURNOVER_REASON_CLEARANCE_VIOLATION
    assert float(out.state.team_b_score[0]) == 1.0


@pytest.mark.parametrize("enabled", [False, True])
def test_valid_shot_and_disabled_behavior(enabled):
    static, state = _reset(replace(_config(), require_all_offense_clear=enabled))
    state = _live(static, state)._replace(
        offense_clearance=jnp.asarray([[1, 1, 0, 0]], dtype=jnp.int8)
    )
    out = _step(
        static,
        state,
        jnp.zeros((1, 4), dtype=jnp.int32).at[0, 0].set(ActionType.SHOOT.value),
    )
    assert int(out.shot_attempt[0]) == 1
    if not enabled:
        state = state._replace(clearance_achieved=jnp.zeros((1,), dtype=jnp.int8))
        out = _step(
            static,
            state,
            jnp.zeros((1, 4), dtype=jnp.int32).at[0, 0].set(ActionType.SHOOT.value),
        )
        assert int(out.turnover[0]) == 1
        assert float(out.state.team_b_score[0]) == 0.0


@pytest.mark.parametrize("model", ["mlp", "attention"])
@pytest.mark.parametrize("radius", [0, 1])
def test_observation_schema_metadata_and_state_restore(model, radius):
    args = parse_args(
        [
            "--enable-multi-possession",
            "--require-all-offense-clear",
            "--players",
            "2",
            "--policy-model",
            model,
            "--check-clearance",
            str(radius),
        ]
    )
    static, state = _reset(_config(check_clearance=radius, check_setup_steps=3))
    state = _live(static, state)._replace(
        offense_clearance=jnp.asarray([[1, 0, 0, 0]], dtype=jnp.int8)
    )
    obs = env.build_policy_observation_batch(
        static,
        state,
        jnp,
        model_type=model,
        multi_possession_features=True,
        check_clearance_features=True,
        team_clearance_features=True,
    )
    masks = env.build_action_masks_batch(static, state, jnp)[:, static.offense_ids]
    spec = _build_policy_spec(args, static, np.asarray(obs), np.asarray(masks))
    assert spec.observation_schema_version == 6
    metadata = _checkpoint_observation_metadata(spec, args=args)
    assert metadata["player_labels"][-1] == "offense_cleared"
    assert metadata["global_labels"][-1] == "clearance_ready_to_shoot"
    players, globals_ = env.build_team_clearance_observation_features_batch(
        static, state, jnp
    )
    np.testing.assert_array_equal(players[0, :, 0], [1, 0, 0, 0])
    assert float(globals_[0, 0]) == 0.0
    restored = _restore_like_template(
        state._asdict(),
        state._replace(offense_clearance=jnp.zeros_like(state.offense_clearance)),
    )
    np.testing.assert_array_equal(restored.offense_clearance, state.offense_clearance)
    assert _jax_env_config_from_args(args)["require_all_offense_clear"] is True
    assert _jax_env_config_from_args(args)["multi_possession_schema_version"] == 6


def test_historical_config_defaults_and_semantic_validation():
    old = JaxEnvironmentConfig(enable_multi_possession=True)
    assert not old.require_all_offense_clear
    assert (
        normalize_runtime_environment_config(old).multi_possession_schema_version == 4
    )
    assert (
        normalize_runtime_environment_config(_config()).multi_possession_schema_version
        == 6
    )
    with pytest.raises(ValueError, match="schema"):
        normalize_runtime_environment_config(
            replace(_config(), multi_possession_schema_version=4)
        )
    with pytest.raises(ValueError, match="enable_multi_possession"):
        JaxEnvironmentConfig(require_all_offense_clear=True)


def test_legacy_checkpoint_without_clearance_state_restores_disabled_behavior():
    _, template = _reset(replace(_config(), require_all_offense_clear=False))
    legacy_mapping = template._asdict()
    legacy_mapping.pop("offense_clearance")
    for saved in (legacy_mapping, tuple(template[:-1])):
        restored = _restore_like_template(saved, template)
        np.testing.assert_array_equal(restored.offense_clearance, 0)
        np.testing.assert_array_equal(restored.positions, template.positions)
        np.testing.assert_array_equal(
            restored.clearance_achieved, template.clearance_achieved
        )
    # Missing historical config must never inherit a newly requested true value.
    normalized = _compatible_env_config_for_resume(
        {}, {"require_all_offense_clear": True}
    )
    assert normalized["require_all_offense_clear"] is False


@pytest.mark.parametrize("winner", [0, 2])
def test_rebound_preserves_offense_flags_or_initializes_new_offense(winner):
    from tests.test_jax_multi_possession import _force_rebound_winner

    static, state = _reset()
    state = _live(static, state)._replace(
        offense_clearance=jnp.asarray([[1, 1, 0, 0]], dtype=jnp.int8),
        layup_pct=jnp.zeros_like(state.layup_pct),
        three_pt_pct=jnp.zeros_like(state.three_pt_pct),
        dunk_pct=jnp.zeros_like(state.dunk_pct),
    )
    static = _force_rebound_winner(static, state, winner)
    out = _step(
        static,
        state,
        jnp.zeros((1, 4), dtype=jnp.int32).at[0, 0].set(ActionType.SHOOT.value),
        seed=114,
    )
    assert int(out.shot_success[0]) == 0
    if winner == 0:
        assert int(out.offensive_rebound[0]) == 1
        assert int(out.possession_ended[0]) == 0
        np.testing.assert_array_equal(
            out.state.offense_clearance, state.offense_clearance
        )
    else:
        assert int(out.defensive_rebound[0]) == 1
        assert int(out.state.offense_team[0]) == env.TEAM_B
        np.testing.assert_array_equal(
            out.state.offense_clearance[0], _geometry_flags(static, out.state)
        )


def test_passes_and_catches_remain_allowed_before_clearance():
    static, state = _reset(_config(pass_mode="pointer_targeted"))
    state = _live(static, state)._replace(
        clearance_achieved=jnp.zeros((1,), dtype=jnp.int8)
    )
    slot = int(np.flatnonzero(np.asarray(static.pointer_pass_target_ids[0]) == 1)[0])
    action = env.PASS_ACTION_START + slot
    assert int(env.build_action_masks_batch(static, state, jnp)[0, 0, action]) == 1
    out = _step(static, state, jnp.zeros((1, 4), dtype=jnp.int32).at[0, 0].set(action))
    assert int(out.completed_pass[0]) == 1
    assert int(out.state.ball_holder[0]) == 1
    assert int(out.possession_ended[0]) == 0


def test_same_possession_technical_restart_keeps_partial_clearance():
    static, state = _reset()
    state = _live(static, state)._replace(
        offense_clearance=jnp.asarray([[1, 0, 0, 0]], dtype=jnp.int8)
    )
    single = jax.tree.map(lambda x: x[0], state)
    restarted = env._restart_after_defensive_technical_single(
        static,
        single,
        shot_clock_before_interruption=single.shot_clock,
        inbound_selection_key=jax.random.PRNGKey(77),
        jax=jax,
        jnp=jnp,
    )
    np.testing.assert_array_equal(restarted.offense_clearance, single.offense_clearance)


def test_setup_deadline_does_not_wait_for_team_clearance():
    static, state = _reset(_config(check_setup_steps=1))
    state = _live(static, state)._replace(
        game_phase=jnp.asarray([env.GAME_PHASE_CHECK_SETUP], dtype=jnp.int8),
        check_setup_steps_remaining=jnp.asarray([1], dtype=jnp.int32),
        ball_holder=jnp.asarray([-1], dtype=jnp.int32),
        check_team=jnp.asarray([env.TEAM_A], dtype=jnp.int8),
    )
    out = _step(static, state)
    assert int(out.state.game_phase[0]) == env.GAME_PHASE_AWAITING_CHECK
    assert not np.any(np.asarray(out.state.offense_clearance))
    assert int(out.state.shot_clock[0]) == int(state.shot_clock[0])


def test_last_second_penalty_counts_before_game_winner_is_decided():
    static, state = _reset(
        _config(
            episode_end_mode="game_clock",
            game_clock_seconds=1,
            multi_possession_reward_mode="scoring_events",
        )
    )
    state = _live(static, state)
    out = _step(
        static,
        state,
        jnp.zeros((1, 4), dtype=jnp.int32).at[0, 0].set(ActionType.SHOOT.value),
    )
    assert float(out.state.team_b_score[0]) == 1.0
    assert bool(out.done[0])
    assert int(out.state.overtime_round[0]) == 0


def test_team_clearance_violation_takes_precedence_over_defensive_lane_violation():
    static, state = _reset(replace(_config(), illegal_defense_enabled=True))
    state = _live(static, state)._replace(
        defense_lane_steps=jnp.full_like(state.defense_lane_steps, 100)
    )
    out = _step(
        static,
        state,
        jnp.zeros((1, 4), dtype=jnp.int32).at[0, 0].set(ActionType.SHOOT.value),
    )
    assert int(out.turnover[0]) == 1
    assert int(out.defensive_lane_violation[0]) == 0
    assert float(out.state.team_a_score[0]) == 0.0
    assert float(out.state.team_b_score[0]) == 1.0
    assert int(out.state.offense_team[0]) == env.TEAM_B


def test_interception_initializes_new_offense_clearance_from_current_positions():
    static, state = _reset(_config(pass_mode="pointer_targeted"))
    static = static._replace(base_steal_rate=jnp.asarray(1e6, dtype=jnp.float32))
    state = _live(static, state)._replace(
        positions=jnp.asarray(
            [[np.asarray(static.basket_position), [4, 8], [0, 8], [5, 5]]],
            dtype=jnp.int32,
        ),
        offense_clearance=jnp.asarray([[1, 1, 0, 0]], dtype=jnp.int8),
    )
    out = _step(
        static,
        state,
        jnp.zeros((1, 4), dtype=jnp.int32).at[0, 0].set(env.PASS_ACTION_START),
        seed=110,
    )
    assert int(out.turnover_reason[0]) == env.TURNOVER_REASON_INTERCEPTED
    assert int(out.state.offense_team[0]) == env.TEAM_B
    np.testing.assert_array_equal(
        out.state.offense_clearance[0], _geometry_flags(static, out.state)
    )


def test_overtime_opening_reinitializes_player_clearance():
    static, state = _reset(_config(episode_end_mode="game_clock"))
    state = state._replace(offense_clearance=jnp.ones_like(state.offense_clearance))
    single = jax.tree.map(lambda x: x[0], state)
    with jax.disable_jit():
        next_state = env._start_timed_overtime_single(
            static, single, jnp.asarray(1, dtype=jnp.int32), jax, jnp
        )
    next_state = jax.tree.map(lambda x: x[None], next_state)
    np.testing.assert_array_equal(
        next_state.offense_clearance[0], _geometry_flags(static, next_state)
    )


@pytest.mark.parametrize("scheduled", [False, True])
def test_evaluation_starter_override_reinitializes_clearance(scheduled):
    from basketworld_jax.eval.native import (
        _apply_paired_starting_teams,
        _apply_scheduled_starting_teams,
    )

    static, state = _reset()
    # Force the next selected starter to differ; stale latched flags must not survive.
    state = state._replace(
        offense_team=jnp.asarray([env.TEAM_B], dtype=jnp.int8),
        ball_holder=jnp.asarray([2], dtype=jnp.int32),
        offense_clearance=jnp.ones_like(state.offense_clearance),
    )
    if scheduled:
        out = _apply_scheduled_starting_teams(static, state, [env.TEAM_A], jnp=jnp)
    else:
        out = _apply_paired_starting_teams(static, state, episode_offset=0, jnp=jnp)
    np.testing.assert_array_equal(
        out.offense_clearance[0], _geometry_flags(static, out)
    )
    assert int(out.clearance_achieved[0]) == int(
        out.offense_clearance[0, int(out.ball_holder[0])]
    )


def test_partial_clearance_survives_durable_checkpoint_restore(tmp_path):
    from basketworld_jax.checkpoints.checkpoint import build_checkpoint_payload, save_checkpoint, load_checkpoint
    static, state = _reset()
    state = _live(static, state)._replace(offense_clearance=jnp.asarray([[1, 0, 0, 0]], dtype=jnp.int8))
    payload = build_checkpoint_payload(
        update_index=1, trainer_config={}, policy_spec={"observation_schema_version": 6},
        frozen_config={}, env_config={"require_all_offense_clear": True},
        params={}, opt_state={}, current_state={"offense": state},
        eval_initial_state={"offense": state}, base_key=jax.random.PRNGKey(73),
        eval_trajectories=[], last_metrics=None,
    )
    path = tmp_path / "team-clearance"
    save_checkpoint(path, payload)
    loaded = load_checkpoint(path)
    restored = _restore_like_template(loaded["current_state"]["offense"], state._replace(offense_clearance=jnp.zeros_like(state.offense_clearance)))
    np.testing.assert_array_equal(restored.offense_clearance, [[1, 0, 0, 0]])
    assert loaded["env_config"]["require_all_offense_clear"] is True
