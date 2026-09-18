from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from basketworld.envs.basketworld_env_v2 import ActionType
from basketworld_jax.checkpoints.checkpoint import load_checkpoint
from basketworld_jax.env.minimal import (
    TEAM_A,
    TEAM_B,
    build_aggregated_reward_batch,
    build_multi_possession_observation_features_batch,
    build_policy_intent_context_batch,
    build_policy_observation_batch,
    build_training_role_flags_batch,
    reset_batch_minimal,
    step_batch_minimal,
)
from basketworld_jax.train.main import (
    parse_args,
    run_training_loop,
    validate_train_args,
)
from basketworld_jax.train.runtime import update_pinned_opponent_episode_state
from basketworld_jax.train.types import (
    TrainerConfig,
    build_trajectory_training_masks,
    compute_gae_and_returns,
)
from tests.test_jax_multi_possession import _multi_possession_static


def _trainer_config(*, completed_only: bool = False) -> TrainerConfig:
    return TrainerConfig(
        kernel_batch_size=2,
        rollout_horizon=2,
        num_updates=1,
        gamma=1.0,
        gae_lambda=1.0,
        ppo_clip_range=0.2,
        value_coef=0.5,
        entropy_coef=0.0,
        learning_rate=3.0e-4,
        policy_update_epochs=1,
        ppo_minibatches=1,
        ppo_completed_episodes_only=completed_only,
    )


def test_multi_possession_rejects_rollout_modes_that_drop_or_reset_unfinished_games():
    with pytest.raises(SystemExit, match="single-episode-rollouts"):
        validate_train_args(
            parse_args(["--enable-multi-possession", "--single-episode-rollouts"])
        )
    with pytest.raises(SystemExit, match="ppo-completed-episodes-only"):
        validate_train_args(
            parse_args(["--enable-multi-possession", "--ppo-completed-episodes-only"])
        )


def test_fixed_team_role_flags_and_rewards_follow_possession_role_per_step():
    team_a_static = _multi_possession_static(possession_limit=5)
    team_b_mask = (
        jnp.zeros_like(team_a_static.training_player_mask)
        .at[team_a_static.defense_ids]
        .set(1.0)
    )
    team_b_static = team_a_static._replace(
        training_role_flag=jnp.asarray(-1.0, dtype=jnp.float32),
        training_player_mask=team_b_mask,
    )
    state = reset_batch_minimal(
        team_a_static,
        jax.random.split(jax.random.PRNGKey(301), 2),
        jax,
        jnp,
    )._replace(
        offense_team=jnp.asarray([TEAM_A, TEAM_B], dtype=jnp.int8),
        team_a_score=jnp.asarray([4.0, 4.0], dtype=jnp.float32),
        team_b_score=jnp.asarray([1.0, 1.0], dtype=jnp.float32),
        intent_index=jnp.asarray([2, 2], dtype=jnp.int32),
        intent_active=jnp.asarray([1, 1], dtype=jnp.int8),
        defense_intent_index=jnp.asarray([7, 7], dtype=jnp.int32),
        defense_intent_active=jnp.asarray([1, 1], dtype=jnp.int8),
    )

    np.testing.assert_array_equal(
        np.asarray(build_training_role_flags_batch(team_a_static, state, jnp)),
        [1.0, -1.0],
    )
    np.testing.assert_array_equal(
        np.asarray(build_training_role_flags_batch(team_b_static, state, jnp)),
        [-1.0, 1.0],
    )

    team_a_obs = build_policy_observation_batch(
        team_a_static,
        state,
        jnp,
        model_type="attention",
        multi_possession_features=True,
    )
    team_b_obs = build_policy_observation_batch(
        team_b_static,
        state,
        jnp,
        model_type="attention",
        multi_possession_features=True,
    )
    np.testing.assert_array_equal(np.asarray(team_a_obs[:, -1]), [1.0, -1.0])
    np.testing.assert_array_equal(np.asarray(team_b_obs[:, -1]), [-1.0, 1.0])

    team_a_context = build_policy_intent_context_batch(team_a_static, state, jnp)
    team_b_context = build_policy_intent_context_batch(team_b_static, state, jnp)
    np.testing.assert_array_equal(np.asarray(team_a_context["intent_index"]), [2, 7])
    np.testing.assert_array_equal(np.asarray(team_b_context["intent_index"]), [7, 2])

    _, team_a_globals = build_multi_possession_observation_features_batch(
        team_a_static,
        state,
        build_training_role_flags_batch(team_a_static, state, jnp),
        jnp,
    )
    _, team_b_globals = build_multi_possession_observation_features_batch(
        team_b_static,
        state,
        build_training_role_flags_batch(team_b_static, state, jnp),
        jnp,
    )
    np.testing.assert_allclose(np.asarray(team_a_globals[:, 0]), [0.2, 0.2])
    np.testing.assert_allclose(np.asarray(team_b_globals[:, 0]), [-0.2, -0.2])

    fixed_team_rewards = jnp.asarray(
        [[0.5, 0.5, -0.5, -0.5], [0.5, 0.5, -0.5, -0.5]],
        dtype=jnp.float32,
    )
    np.testing.assert_array_equal(
        np.asarray(
            build_aggregated_reward_batch(team_a_static, fixed_team_rewards, jnp)
        ),
        [1.0, 1.0],
    )
    np.testing.assert_array_equal(
        np.asarray(
            build_aggregated_reward_batch(team_b_static, fixed_team_rewards, jnp)
        ),
        [-1.0, -1.0],
    )


def test_turnover_role_switch_bootstraps_but_real_game_endings_cut_returns():
    # Mixed-sign fixed-team rewards straddle a nonterminal possession switch.
    rewards = jnp.asarray([[0.5], [-0.25]], dtype=jnp.float32)
    values = jnp.asarray([[0.25], [0.5]], dtype=jnp.float32)

    _, cut_returns = compute_gae_and_returns(
        rewards,
        values,
        jnp.asarray([[0], [0]], dtype=jnp.int8),
        jnp.asarray([3.0], dtype=jnp.float32),
        gamma=1.0,
        gae_lambda=1.0,
        jax=jax,
        jnp=jnp,
    )
    np.testing.assert_allclose(np.asarray(cut_returns[:, 0]), [3.25, 2.75])

    _, terminal_returns = compute_gae_and_returns(
        rewards,
        values,
        jnp.asarray([[0], [1]], dtype=jnp.int8),
        jnp.asarray([99.0], dtype=jnp.float32),
        gamma=1.0,
        gae_lambda=1.0,
        jax=jax,
        jnp=jnp,
    )
    np.testing.assert_allclose(np.asarray(terminal_returns[:, 0]), [0.25, -0.25])


def test_every_active_chunk_is_ppo_eligible_without_a_completed_game():
    trajectory = SimpleNamespace(
        active_mask=jnp.ones((3, 2), dtype=jnp.float32),
        episode_start=jnp.zeros((3, 2), dtype=jnp.int8),
        dones=jnp.zeros((3, 2), dtype=jnp.int8),
    )
    active_mask, loss_weights, denominator = build_trajectory_training_masks(
        trajectory,
        _trainer_config(completed_only=False),
        jax,
        jnp,
    )
    np.testing.assert_array_equal(np.asarray(active_mask), 1.0)
    np.testing.assert_array_equal(np.asarray(loss_weights), 1.0)
    assert float(np.asarray(denominator)) == pytest.approx(6.0)


def test_game_longer_than_horizon_contributes_each_chunk_and_finishes_without_respawn():
    static = _multi_possession_static(possession_limit=1)
    state = reset_batch_minimal(
        static,
        jax.random.split(jax.random.PRNGKey(401), 1),
        jax,
        jnp,
    )._replace(shot_clock=jnp.asarray([2], dtype=jnp.int32))
    actions = jnp.full(
        (1, state.positions.shape[1]),
        ActionType.NOOP.value,
        dtype=jnp.int32,
    )

    first_chunk = step_batch_minimal(
        static,
        state,
        actions,
        jax.random.split(jax.random.PRNGKey(402), 1),
        jax,
        jnp,
    )
    assert not bool(np.asarray(first_chunk.done)[0])
    assert int(np.asarray(first_chunk.state.step_count)[0]) == 1
    assert int(np.asarray(first_chunk.state.completed_possessions)[0]) == 0

    second_chunk = step_batch_minimal(
        static,
        first_chunk.state,
        actions,
        jax.random.split(jax.random.PRNGKey(403), 1),
        jax,
        jnp,
    )
    assert bool(np.asarray(second_chunk.done)[0])
    assert int(np.asarray(second_chunk.state.step_count)[0]) == 2
    assert int(np.asarray(second_chunk.state.completed_possessions)[0]) == 1

    for done in (first_chunk.done, second_chunk.done):
        trajectory = SimpleNamespace(
            active_mask=jnp.ones((1, 1), dtype=jnp.float32),
            episode_start=jnp.zeros((1, 1), dtype=jnp.int8),
            dones=done[None, :],
        )
        _, loss_weights, denominator = build_trajectory_training_masks(
            trajectory,
            _trainer_config(completed_only=False),
            jax,
            jnp,
        )
        np.testing.assert_array_equal(np.asarray(loss_weights), 1.0)
        assert float(np.asarray(denominator)) == pytest.approx(1.0)


def test_opponent_identity_and_action_mode_change_only_at_real_game_reset():
    assignment, mode = update_pinned_opponent_episode_state(
        jnp.asarray([0, 0, -1], dtype=jnp.int32),
        jnp.asarray([0, 1, 0], dtype=jnp.bool_),
        jnp.asarray([0, 1, 0], dtype=jnp.bool_),
        jnp.asarray([1, 1, 1], dtype=jnp.int32),
        jnp.asarray([1, 0, 1], dtype=jnp.bool_),
        jnp,
    )
    np.testing.assert_array_equal(np.asarray(assignment), [0, 1, -1])
    np.testing.assert_array_equal(np.asarray(mode), [False, False, False])


def test_fresh_launcher_keeps_the_agreed_budget_and_has_no_continuation_dependency():
    script = Path("scripts/run_jax_5v5_halfcourt_multi_possession.sh").read_text()
    assert "--enable-multi-possession" in script
    assert "--kernel-batch-size 512" in script
    assert "--rollout-horizon 64" in script
    assert "--policy-update-epochs 1" in script
    assert "--ppo-minibatches 16" in script
    assert "--gamma 1.0" in script
    assert "--start-template-enabled" not in script
    assert "--ppo-completed-episodes-only" not in script
    assert "--single-episode-rollouts" not in script
    assert "--continue-run-id" not in script
    assert "--resume-checkpoint" not in script
    assert "--frozen-opponent-checkpoint" not in script


def test_short_fresh_multi_possession_training_continues_and_resumes(tmp_path):
    checkpoint_dir = tmp_path / "continuous_ckpts"
    common_args = [
        "--run-train-loop",
        "--enable-multi-possession",
        "--multi-possession-limit",
        "25",
        "--kernel-batch-size",
        "1",
        "--rollout-horizon",
        "1",
        "--policy-update-epochs",
        "1",
        "--ppo-minibatches",
        "1",
        "--log-every-updates",
        "1",
        "--eval-every-updates",
        "0",
        "--eval-deploy-every-updates",
        "0",
        "--eval-deploy-batches",
        "0",
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--checkpoint-every-updates",
        "1",
        "--no-progress",
    ]
    first_args = parse_args([*common_args, "--num-updates", "1"])
    validate_train_args(first_args)
    first_result = run_training_loop(first_args)

    metrics = first_result["final_metrics"]
    assert metrics["steps_per_update"] == 2
    assert metrics["active_step_count"] == pytest.approx(2.0)
    assert metrics["ppo_batch_size"] == 2
    assert metrics["ppo_unused_active_step_count"] == pytest.approx(0.0)
    assert metrics["ppo_update_optimizer_samples_per_sec"] > 0.0
    assert np.isfinite(metrics["grad_norm"])
    # Metrics describe the pool used for this rollout. The update-1 snapshot is
    # appended immediately afterward and is present in the checkpoint below.
    assert metrics["opponent_pool_candidate_count"] == 0
    assert metrics["opponent_legal_random_game_count"] == 2
    assert set(first_result["training_player_ids"]) == {"offense", "defense"}
    for cohort in ("offense", "defense"):
        assert (
            metrics[f"{cohort}_learner_offense_step_fraction"]
            + metrics[f"{cohort}_learner_defense_step_fraction"]
        ) == pytest.approx(1.0)

    latest = checkpoint_dir / "latest"
    payload = load_checkpoint(latest)
    assert payload["update_index"] == 1
    assert set(payload["current_state"]) == {"offense", "defense"}
    assert payload["opponent_pool_state"]["enabled"] is True
    assert len(payload["opponent_pool_state"]["candidate_infos"]) == 1
    for role in ("offense", "defense"):
        state = payload["current_state"][role]
        assert {
            "team_a_score",
            "team_b_score",
            "completed_possessions",
            "shot_clock",
            "offense_team",
            "clearance_achieved",
        }.issubset(state)
        assert int(np.asarray(state["step_count"])[0]) == 1
        assert int(np.asarray(state["completed_possessions"])[0]) < 25
        np.testing.assert_array_equal(
            np.asarray(payload["opponent_pool_state"]["assignments"][role]),
            [-1],
        )
        np.testing.assert_array_equal(
            np.asarray(payload["opponent_pool_state"]["deterministic_modes"][role]),
            [False],
        )

    resumed_args = parse_args(
        [
            *common_args,
            "--num-updates",
            "2",
            "--resume-checkpoint",
            str(latest),
        ]
    )
    validate_train_args(resumed_args)
    resumed_result = run_training_loop(resumed_args)
    assert resumed_result["resumed_from_checkpoint"] == str(latest)
    assert resumed_result["final_metrics"]["update_index"] == 2
    assert resumed_result["final_metrics"][
        "ppo_unused_active_step_count"
    ] == pytest.approx(0.0)

    resumed_payload = load_checkpoint(latest)
    assert resumed_payload["update_index"] == 2
    for role in ("offense", "defense"):
        state = resumed_payload["current_state"][role]
        assert int(np.asarray(state["step_count"])[0]) == 2
        np.testing.assert_array_equal(
            np.asarray(resumed_payload["opponent_pool_state"]["assignments"][role]),
            [-1],
        )
        np.testing.assert_array_equal(
            np.asarray(
                resumed_payload["opponent_pool_state"]["deterministic_modes"][role]
            ),
            [False],
        )
