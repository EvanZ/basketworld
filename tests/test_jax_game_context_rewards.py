from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from basketworld.envs.basketworld_env_v2 import ActionType
from basketworld_jax.env.minimal import (
    GAME_PHASE_AWAITING_INBOUND,
    GAME_PHASE_LIVE,
    MULTI_POSSESSION_REWARD_POINT_DIFFERENTIAL,
    MULTI_POSSESSION_SCHEMA_VERSION,
    PASS_ACTION_START,
    TEAM_A,
    TEAM_B,
    build_action_masks_batch,
    build_multi_possession_observation_features_batch,
    reset_batch_minimal,
    step_batch_minimal,
    token_observation_dims,
)
from basketworld_jax.eval.native import _adapt_policy_observation_to_spec
from basketworld_jax.models import ActorCriticSpec
from basketworld_jax.train.main import (
    _build_reward_component_arrays,
    _jax_env_config_from_args,
    parse_args,
    validate_train_args,
)
from tests.test_jax_multi_possession import (
    _force_rebound_winner,
    _multi_possession_static,
    _noops,
    _shoot,
)


def _team_total(values, ids) -> float:
    return float(np.asarray(values)[0, np.asarray(ids, dtype=np.int32)].sum())


def _terminal_clearance_violation(
    static, *, team_a_score: float, team_b_score: float, seed: int
):
    state = reset_batch_minimal(
        static,
        jax.random.split(jax.random.PRNGKey(seed), 1),
        jax,
        jnp,
    )
    holder = int(np.asarray(static.offense_ids)[0])
    state = state._replace(
        offense_team=jnp.asarray([TEAM_A], dtype=jnp.int8),
        ball_holder=jnp.asarray([holder], dtype=jnp.int32),
        completed_possessions=jnp.asarray(
            [int(np.asarray(static.multi_possession_limit)) - 1], dtype=jnp.int32
        ),
        team_a_score=jnp.asarray([team_a_score], dtype=jnp.float32),
        team_b_score=jnp.asarray([team_b_score], dtype=jnp.float32),
        clearance_achieved=jnp.asarray([0], dtype=jnp.int8),
        game_phase=jnp.asarray([GAME_PHASE_LIVE], dtype=jnp.int8),
    )
    actions = _noops(state).at[0, holder].set(ActionType.SHOOT.value)
    return step_batch_minimal(
        static,
        state,
        actions,
        jax.random.split(jax.random.PRNGKey(seed + 1), 1),
        jax,
        jnp,
    )


@pytest.mark.parametrize(
    ("score_a", "score_b", "expected"),
    [(5.0, 3.0, 1.0), (2.0, 4.0, -1.0), (3.0, 3.0, 0.0)],
)
def test_terminal_win_loss_rewards_are_fixed_team_zero_sum_and_not_duplicated(
    score_a,
    score_b,
    expected,
):
    static = _multi_possession_static(possession_limit=2)
    out = _terminal_clearance_violation(
        static,
        team_a_score=score_a,
        team_b_score=score_b,
        seed=201,
    )

    assert bool(np.asarray(out.done)[0])
    assert _team_total(out.game_reward, static.offense_ids) == pytest.approx(expected)
    assert _team_total(out.game_reward, static.defense_ids) == pytest.approx(-expected)
    assert float(np.asarray(out.game_reward).sum()) == pytest.approx(0.0)
    assert _team_total(out.auxiliary_reward, static.offense_ids) == pytest.approx(0.0)
    assert _team_total(out.rewards, static.offense_ids) == pytest.approx(expected)

    repeated = step_batch_minimal(
        static,
        out.state,
        _noops(out.state),
        jax.random.split(jax.random.PRNGKey(202), 1),
        jax,
        jnp,
    )
    np.testing.assert_array_equal(np.asarray(repeated.game_reward), 0.0)
    np.testing.assert_array_equal(np.asarray(repeated.rewards), 0.0)


def test_terminal_point_differential_is_separately_selectable():
    static = _multi_possession_static(possession_limit=2)._replace(
        multi_possession_reward_mode=jnp.asarray(
            MULTI_POSSESSION_REWARD_POINT_DIFFERENTIAL,
            dtype=jnp.int32,
        )
    )
    out = _terminal_clearance_violation(
        static,
        team_a_score=7.0,
        team_b_score=3.0,
        seed=203,
    )
    assert _team_total(out.game_reward, static.offense_ids) == pytest.approx(4.0)
    assert _team_total(out.game_reward, static.defense_ids) == pytest.approx(-4.0)


def test_score_potential_telescopes_across_role_flip_rollout_boundary_and_beta_change():
    static = _multi_possession_static(possession_limit=2)._replace(
        enable_phi_shaping=jnp.asarray(1, dtype=jnp.int8),
        phi_beta=jnp.asarray(0.5, dtype=jnp.float32),
        reward_shaping_gamma=jnp.asarray(1.0, dtype=jnp.float32),
        score_potential_scale=jnp.asarray(1.0, dtype=jnp.float32),
    )
    state = reset_batch_minimal(
        static,
        jax.random.split(jax.random.PRNGKey(204), 1),
        jax,
        jnp,
    )
    holder = int(np.asarray(static.offense_ids)[0])
    positions = np.asarray(state.positions).copy()
    positions[0, holder] = np.asarray(static.basket_position)
    state = state._replace(
        positions=jnp.asarray(positions, dtype=jnp.int32),
        offense_team=jnp.asarray([TEAM_A], dtype=jnp.int8),
        ball_holder=jnp.asarray([holder], dtype=jnp.int32),
        clearance_achieved=jnp.asarray([1], dtype=jnp.int8),
        layup_pct=jnp.ones_like(state.layup_pct),
        three_pt_pct=jnp.ones_like(state.three_pt_pct),
    )
    scored = _shoot(static, state, seed=205)
    assert not bool(np.asarray(scored.done)[0])
    assert int(np.asarray(scored.state.offense_team)[0]) == TEAM_B
    score_diff = float(
        np.asarray(scored.state.team_a_score - scored.state.team_b_score)[0]
    )
    phi_1 = _team_total(scored.rewards, static.offense_ids)
    assert phi_1 == pytest.approx(0.5 * score_diff)
    assert float(np.asarray(scored.team_a_score_delta)[0]) == pytest.approx(score_diff)
    assert float(np.asarray(scored.team_b_score_delta)[0]) == pytest.approx(0.0)
    assert _team_total(scored.game_reward, static.offense_ids) == pytest.approx(0.0)
    assert _team_total(scored.auxiliary_reward, static.offense_ids) == pytest.approx(
        0.0
    )

    # Simulate an update boundary that changes beta while the game state and
    # cached effective potential continue into the next rollout.
    changed_static = static._replace(phi_beta=jnp.asarray(0.8, dtype=jnp.float32))
    inbounder = int(np.asarray(scored.state.inbound_player)[0])
    masks = np.asarray(build_action_masks_batch(changed_static, scored.state, jnp))
    legal_slots = np.flatnonzero(masks[0, inbounder, PASS_ACTION_START:])
    assert legal_slots.size > 0
    inbound_actions = (
        _noops(scored.state)
        .at[0, inbounder]
        .set(PASS_ACTION_START + int(legal_slots[0]))
    )
    inbound = step_batch_minimal(
        changed_static,
        scored.state,
        inbound_actions,
        jax.random.split(jax.random.PRNGKey(206), 1),
        jax,
        jnp,
    )
    phi_2 = _team_total(inbound.rewards, static.offense_ids)
    assert phi_2 == pytest.approx((0.8 - 0.5) * score_diff)
    live_state = inbound.state._replace(
        clearance_achieved=jnp.asarray([0], dtype=jnp.int8)
    )
    live_holder = int(np.asarray(live_state.ball_holder)[0])
    terminal = step_batch_minimal(
        changed_static,
        live_state,
        _noops(live_state).at[0, live_holder].set(ActionType.SHOOT.value),
        jax.random.split(jax.random.PRNGKey(207), 1),
        jax,
        jnp,
    )
    phi_3 = _team_total(terminal.rewards - terminal.game_reward, static.offense_ids)
    assert phi_3 == pytest.approx(-0.8 * score_diff)
    assert phi_1 + phi_2 + phi_3 == pytest.approx(0.0, abs=1.0e-6)


def test_role_switch_and_offensive_rebound_do_not_reset_or_flip_score_potential():
    static = _multi_possession_static(possession_limit=4)._replace(
        enable_phi_shaping=jnp.asarray(1, dtype=jnp.int8),
        phi_beta=jnp.asarray(0.5, dtype=jnp.float32),
        reward_shaping_gamma=jnp.asarray(1.0, dtype=jnp.float32),
        score_potential_scale=jnp.asarray(1.0, dtype=jnp.float32),
        defender_pressure_distance=jnp.asarray(100.0, dtype=jnp.float32),
        defender_pressure_turnover_chance=jnp.asarray(1.0, dtype=jnp.float32),
        defender_pressure_decay_lambda=jnp.asarray(0.0, dtype=jnp.float32),
    )
    state = reset_batch_minimal(
        static,
        jax.random.split(jax.random.PRNGKey(208), 1),
        jax,
        jnp,
    )._replace(
        offense_team=jnp.asarray([TEAM_A], dtype=jnp.int8),
        team_a_score=jnp.asarray([2.0], dtype=jnp.float32),
        team_b_score=jnp.asarray([0.0], dtype=jnp.float32),
        cached_phi=jnp.asarray([1.0], dtype=jnp.float32),
    )
    switched = step_batch_minimal(
        static,
        state,
        _noops(state),
        jax.random.split(jax.random.PRNGKey(209), 1),
        jax,
        jnp,
    )
    assert int(np.asarray(switched.state.offense_team)[0]) == TEAM_B
    assert _team_total(switched.rewards, static.offense_ids) == pytest.approx(0.0)
    assert float(np.asarray(switched.state.cached_phi)[0]) == pytest.approx(1.0)

    rebound_static = _multi_possession_static(possession_limit=4)._replace(
        enable_phi_shaping=jnp.asarray(1, dtype=jnp.int8),
        phi_beta=jnp.asarray(0.5, dtype=jnp.float32),
        reward_shaping_gamma=jnp.asarray(1.0, dtype=jnp.float32),
        score_potential_scale=jnp.asarray(1.0, dtype=jnp.float32),
    )
    rebound_state = reset_batch_minimal(
        rebound_static,
        jax.random.split(jax.random.PRNGKey(210), 1),
        jax,
        jnp,
    )._replace(
        team_a_score=jnp.asarray([2.0], dtype=jnp.float32),
        team_b_score=jnp.asarray([0.0], dtype=jnp.float32),
        cached_phi=jnp.asarray([1.0], dtype=jnp.float32),
        layup_pct=jnp.zeros((1, 4), dtype=jnp.float32),
        three_pt_pct=jnp.zeros((1, 4), dtype=jnp.float32),
        dunk_pct=jnp.zeros((1, 4), dtype=jnp.float32),
    )
    rebound_holder = int(np.asarray(rebound_state.ball_holder)[0])
    rebound_static = _force_rebound_winner(
        rebound_static,
        rebound_state,
        rebound_holder,
    )
    rebound = _shoot(rebound_static, rebound_state, seed=211)
    assert int(np.asarray(rebound.offensive_rebound)[0]) == 1
    assert _team_total(rebound.rewards, rebound_static.offense_ids) == pytest.approx(
        0.0
    )
    assert float(np.asarray(rebound.state.cached_phi)[0]) == pytest.approx(1.0)


def test_game_context_observations_are_team_relative_and_cover_inbound_final_state():
    static = _multi_possession_static(possession_limit=5)
    state = reset_batch_minimal(
        static,
        jax.random.split(jax.random.PRNGKey(212), 1),
        jax,
        jnp,
    )
    inbounder = int(np.asarray(static.offense_ids)[0])
    positions = np.asarray(state.positions).copy()
    positions[0, inbounder] = np.asarray(static.inbound_position)
    state = state._replace(
        positions=jnp.asarray(positions, dtype=jnp.int32),
        offense_team=jnp.asarray([TEAM_A], dtype=jnp.int8),
        team_a_score=jnp.asarray([6.0], dtype=jnp.float32),
        team_b_score=jnp.asarray([3.0], dtype=jnp.float32),
        completed_possessions=jnp.asarray([2], dtype=jnp.int32),
        game_phase=jnp.asarray([GAME_PHASE_AWAITING_INBOUND], dtype=jnp.int8),
        inbound_player=jnp.asarray([inbounder], dtype=jnp.int32),
        inbound_steps_remaining=jnp.asarray([3], dtype=jnp.int32),
        clearance_achieved=jnp.asarray([0], dtype=jnp.int8),
    )
    offense_players, offense_globals = (
        build_multi_possession_observation_features_batch(static, state, 1.0, jnp)
    )
    _, defense_globals = build_multi_possession_observation_features_batch(
        static, state, -1.0, jnp
    )
    assert float(np.asarray(offense_globals)[0, 0]) == pytest.approx(3.0 / 15.0)
    assert float(np.asarray(defense_globals)[0, 0]) == pytest.approx(-3.0 / 15.0)
    np.testing.assert_allclose(
        np.asarray(offense_globals)[0, 1:],
        np.asarray([3.0 / 5.0, 1.0, 1.0, 3.0 / 5.0], dtype=np.float32),
    )
    assert np.asarray(offense_players)[0, inbounder].tolist() == [1.0, 0.0]

    reentered_positions = positions.copy()
    reentered_positions[0, inbounder] = np.asarray(static.basket_position)
    reentered = state._replace(
        positions=jnp.asarray(reentered_positions, dtype=jnp.int32),
        game_phase=jnp.asarray([GAME_PHASE_LIVE], dtype=jnp.int8),
        inbound_player=jnp.asarray([-1], dtype=jnp.int32),
        inbound_steps_remaining=jnp.asarray([0], dtype=jnp.int32),
    )
    reentered_players, _ = build_multi_possession_observation_features_batch(
        static, reentered, 1.0, jnp
    )
    assert np.asarray(reentered_players)[0, inbounder].tolist() == [0.0, 1.0]

    flipped = state._replace(offense_team=jnp.asarray([TEAM_B], dtype=jnp.int8))
    _, flipped_offense = build_multi_possession_observation_features_batch(
        static, flipped, 1.0, jnp
    )
    _, flipped_defense = build_multi_possession_observation_features_batch(
        static, flipped, -1.0, jnp
    )
    assert float(np.asarray(flipped_offense)[0, 0]) == pytest.approx(-3.0 / 15.0)
    assert float(np.asarray(flipped_defense)[0, 0]) == pytest.approx(3.0 / 15.0)

    final_state = state._replace(
        completed_possessions=jnp.asarray([5], dtype=jnp.int32),
        episode_ended=jnp.asarray([1], dtype=jnp.int8),
        game_phase=jnp.asarray([GAME_PHASE_LIVE], dtype=jnp.int8),
        inbound_player=jnp.asarray([-1], dtype=jnp.int32),
        inbound_steps_remaining=jnp.asarray([0], dtype=jnp.int32),
        clearance_achieved=jnp.asarray([1], dtype=jnp.int8),
    )
    _, final_globals = build_multi_possession_observation_features_batch(
        static, final_state, 1.0, jnp
    )
    np.testing.assert_allclose(np.asarray(final_globals)[0, 1:], 0.0)
    final_masks = np.asarray(build_action_masks_batch(static, final_state, jnp))
    assert not np.any(final_masks[..., 1:])


def test_multi_possession_observation_schema_and_cli_reject_incompatible_configs():
    assert token_observation_dims(False, True, True) == (20, 12)
    args = parse_args(["--enable-multi-possession"])
    assert args.multi_possession_reward_mode == "win_loss"
    assert args.score_potential_scale == pytest.approx(1.0)
    assert args.multi_possession_aux_rewards_enabled is False
    validate_train_args(args)
    assert (
        _jax_env_config_from_args(args)["multi_possession_schema_version"]
        == MULTI_POSSESSION_SCHEMA_VERSION
    )
    assert (
        _jax_env_config_from_args(parse_args([]))["multi_possession_schema_version"]
        == 1
    )

    with pytest.raises(SystemExit, match="requires --enable-multi-possession"):
        validate_train_args(parse_args(["--multi-possession-aux-rewards-enabled"]))
    with pytest.raises(SystemExit, match="phi-use-ball-handler-only"):
        validate_train_args(
            parse_args(
                [
                    "--enable-multi-possession",
                    "--enable-phi-shaping",
                    "true",
                    "--phi-use-ball-handler-only",
                    "true",
                ]
            )
        )

    spec = ActorCriticSpec(
        flat_obs_dim=12,
        training_player_count=2,
        action_dim_per_player=14,
        total_action_dim=28,
        hidden_dims=(16,),
        multi_possession_features=True,
        observation_schema_version=MULTI_POSSESSION_SCHEMA_VERSION,
    )
    with pytest.raises(ValueError, match="versioned checkpoint schema"):
        _adapt_policy_observation_to_spec(
            jnp.zeros((1, 11), dtype=jnp.float32),
            None,
            spec,
            jnp,
        )
    legacy = replace(
        spec, multi_possession_features=False, observation_schema_version=1
    )
    assert legacy != spec


def test_reward_components_remain_separate_for_training_logging():
    rollout = SimpleNamespace(
        trajectory=SimpleNamespace(
            rewards=jnp.asarray([[3.5]], dtype=jnp.float32),
            phi_r_shape=jnp.asarray([[0.25]], dtype=jnp.float32),
            game_rewards=jnp.asarray([[1.0]], dtype=jnp.float32),
            auxiliary_rewards=jnp.asarray([[0.5]], dtype=jnp.float32),
        )
    )
    static = SimpleNamespace(
        role_encoding=jnp.asarray([1.0, 1.0, -1.0, -1.0], dtype=jnp.float32),
        training_player_mask=jnp.asarray([1.0, 1.0, 0.0, 0.0], dtype=jnp.float32),
        task_reward_scale=jnp.asarray(2.0, dtype=jnp.float32),
    )
    components = _build_reward_component_arrays(
        rollout,
        static,
        task_reward_scale=3.0,
        jnp=jnp,
    )
    assert set(components) == {
        "task_reward",
        "phi_reward",
        "game_reward",
        "auxiliary_reward",
        "intent_bonus",
    }
    assert float(np.asarray(components["phi_reward"])[0, 0]) == pytest.approx(3.0)
    assert float(np.asarray(components["game_reward"])[0, 0]) == pytest.approx(6.0)
    assert float(np.asarray(components["auxiliary_reward"])[0, 0]) == pytest.approx(3.0)
    assert float(np.asarray(components["task_reward"])[0, 0]) == pytest.approx(0.5)
