from __future__ import annotations

import pytest
import numpy as np

from basketworld_jax.checkpoints import build_checkpoint_payload, load_checkpoint
from basketworld_jax.env import (
    TOKEN_OBS_GLOBAL_DIM,
    TOKEN_OBS_PLAYER_DIM,
    TOKEN_OBS_ROLE_FLAG_DIM,
)
from basketworld_jax.intent.discriminator import (
    IntentDiscriminatorSpec,
    build_intent_step_features_from_rollout,
)
from basketworld_jax.models import ActorCriticSpec
from basketworld_jax.train.types import (
    RolloutOutput,
    SelectorBatch,
    TrainerConfig,
    TrajectoryBatch,
    build_ppo_batch,
    limit_selector_batch_samples,
)
from basketworld_jax.train.main import (
    TRAIN_FROZEN_VALUES,
    _checkpoint_interval_for_update,
    _checkpoint_trainer_config_from_args,
    _entropy_coef_for_update,
    _periodic_checkpoint_updates,
    _filter_mlflow_train_metrics,
    _phi_beta_for_update,
    _log_mlflow_params,
    _selector_learning_rate_for_args,
    _task_reward_scale_for_update,
    build_trainer_config,
    parse_args,
    run_train_scaffold,
    run_training_loop,
    validate_train_args,
)
from basketworld_jax.train.runtime import (
    _apply_selector_update_param_scope,
    _mask_selector_update_grads,
    _merge_selector_update_params,
    _selector_segment_application_masks,
    summarize_ppo_eligible_episode_metrics,
    summarize_reward_by_intent_metrics,
)


def test_trainer_parser_defaults_match_frozen_scope():
    args = parse_args([])

    for key, expected in TRAIN_FROZEN_VALUES.items():
        assert getattr(args, key) == expected


def test_fixed_checkpoint_schedule_preserves_modulo_cadence():
    args = parse_args(
        [
            "--num-updates",
            "10",
            "--checkpoint-every-updates",
            "3",
        ]
    )

    assert _checkpoint_interval_for_update(args, 1) == 3
    assert _periodic_checkpoint_updates(args) == {3, 6, 9}


def test_log_checkpoint_schedule_starts_frequent_and_caps_interval():
    args = parse_args(
        [
            "--num-updates",
            "20",
            "--checkpoint-every-updates",
            "10",
            "--checkpoint-schedule",
            "log",
            "--checkpoint-log-initial-updates",
            "2",
            "--checkpoint-log-ramp-updates",
            "20",
        ]
    )

    due_updates = sorted(_periodic_checkpoint_updates(args))
    intervals = [b - a for a, b in zip(due_updates, due_updates[1:], strict=False)]

    assert due_updates[0] == 2
    assert intervals
    assert intervals[0] < int(args.checkpoint_every_updates)
    assert max(intervals) <= int(args.checkpoint_every_updates)
    assert _checkpoint_interval_for_update(args, 20) == 10


def test_checkpoint_payload_preserves_selector_optimizer_state():
    selector_opt_state = {"count": np.asarray(3, dtype=np.int32)}

    payload = build_checkpoint_payload(
        update_index=1,
        trainer_config={},
        policy_spec={},
        frozen_config={},
        params={"w": np.asarray([1.0], dtype=np.float32)},
        opt_state={"count": np.asarray(2, dtype=np.int32)},
        selector_opt_state=selector_opt_state,
        current_state={},
        eval_initial_state={},
        base_key=np.asarray([0, 1], dtype=np.uint32),
        eval_trajectories=[],
        last_metrics=None,
    )

    assert payload["state"]["selector_opt_state"]["count"] == np.asarray(3, dtype=np.int32)


def test_checkpoint_trainer_config_persists_selector_runtime_fields():
    args = parse_args(
        [
            "--enable-intent-learning",
            "true",
            "--num-intents",
            "8",
            "--intent-commitment-steps",
            "8",
            "--intent-selector-enabled",
            "true",
            "--intent-selector-mode",
            "integrated",
            "--intent-selector-learning-rate",
            "0.0004",
            "--intent-selector-alpha-start",
            "0.25",
            "--intent-selector-alpha-end",
            "0.75",
            "--intent-selector-alpha-warmup-updates",
            "10",
            "--intent-selector-alpha-ramp-updates",
            "20",
            "--intent-selector-eps-start",
            "0.5",
            "--intent-selector-eps-end",
            "0.15",
            "--intent-selector-eps-warmup-updates",
            "30",
            "--intent-selector-eps-ramp-updates",
            "40",
            "--intent-selector-multiselect-enabled",
            "true",
            "--intent-selector-min-play-steps",
            "4",
            "--ppo-completed-episodes-only",
        ]
    )
    trainer_config = build_trainer_config(args)

    serialized = _checkpoint_trainer_config_from_args(trainer_config, args)

    assert serialized["ppo_completed_episodes_only"] is True
    assert serialized["enable_intent_learning"] is True
    assert serialized["num_intents"] == 8
    assert serialized["intent_commitment_steps"] == 8
    assert serialized["intent_selector_enabled"] is True
    assert serialized["intent_selector_mode"] == "integrated"
    assert serialized["intent_selector_learning_rate"] == pytest.approx(0.0004)
    assert serialized["intent_selector_alpha_start"] == pytest.approx(0.25)
    assert serialized["intent_selector_alpha_end"] == pytest.approx(0.75)
    assert serialized["intent_selector_alpha_warmup_updates"] == 10
    assert serialized["intent_selector_alpha_ramp_updates"] == 20
    assert serialized["intent_selector_eps_start"] == pytest.approx(0.5)
    assert serialized["intent_selector_eps_end"] == pytest.approx(0.15)
    assert serialized["intent_selector_eps_warmup_updates"] == 30
    assert serialized["intent_selector_eps_ramp_updates"] == 40
    assert serialized["intent_selector_multiselect_enabled"] is True
    assert serialized["intent_selector_min_play_steps"] == 4


def test_jax_trainer_uses_grouped_opponent_sampling_flag_name():
    args = parse_args(["--grouped-opponent-sampling"])

    assert args.grouped_opponent_sampling is True
    with pytest.raises(SystemExit):
        parse_args(["--per-env-opponent-sampling"])


def test_jax_trainer_allows_skill_distribution_overrides():
    args = parse_args(
        [
            "--layup-pct",
            "0.55",
            "--three-pt-pct",
            "0.37",
            "--dunk-pct",
            "0.6",
            "--layup-std",
            "0.05",
            "--three-pt-std",
            "0.05",
            "--dunk-std",
            "0.3",
        ]
    )

    validate_train_args(args)
    assert args.dunk_pct == 0.6


def test_jax_trainer_allows_lane_rule_overrides():
    args = parse_args(
        [
            "--illegal-defense-enabled",
            "true",
            "--offensive-three-seconds",
            "true",
        ]
    )

    validate_train_args(args)
    assert args.illegal_defense_enabled is True
    assert args.offensive_three_seconds is True


def test_jax_phi_beta_update_schedule():
    args = parse_args(
        [
            "--enable-phi-shaping",
            "true",
            "--phi-beta-start",
            "0.0",
            "--phi-beta-end",
            "0.15",
            "--phi-beta-warmup-updates",
            "2",
            "--phi-beta-ramp-updates",
            "4",
        ]
    )

    validate_train_args(args)
    assert _phi_beta_for_update(args, 1) == pytest.approx(0.0)
    assert _phi_beta_for_update(args, 2) == pytest.approx(0.0)
    assert _phi_beta_for_update(args, 4) == pytest.approx(0.075)
    assert _phi_beta_for_update(args, 6) == pytest.approx(0.15)

    disabled = parse_args(
        [
            "--enable-phi-shaping",
            "false",
            "--phi-beta-start",
            "0.0",
            "--phi-beta-end",
            "0.15",
        ]
    )
    validate_train_args(disabled)
    assert _phi_beta_for_update(disabled, 6) == pytest.approx(0.0)


def test_intent_discriminator_uses_training_mask_for_active_samples():
    jax = pytest.importorskip("jax")
    jnp = jax.numpy

    time_steps = 3
    batch_size = 2
    flat_obs_dim = 8
    player_count = 3
    zeros = jnp.zeros((time_steps, batch_size), dtype=jnp.float32)
    trajectory_data = {field: zeros for field in TrajectoryBatch._fields}
    trajectory_data.update(
        {
            "active_mask": jnp.ones((time_steps, batch_size), dtype=jnp.float32),
            "episode_start": jnp.zeros((time_steps, batch_size), dtype=jnp.float32),
            "flat_obs": jnp.zeros((time_steps, batch_size, flat_obs_dim), dtype=jnp.float32),
            "policy_intent_index": jnp.asarray([[0, 1], [2, 3], [1, 0]], dtype=jnp.int32),
            "policy_intent_gate": jnp.ones((time_steps, batch_size), dtype=jnp.float32),
            "action_mask": jnp.zeros((time_steps, batch_size, player_count, 5), dtype=jnp.float32),
            "actions": jnp.zeros((time_steps, batch_size, player_count), dtype=jnp.int32),
            "full_actions": jnp.zeros((time_steps, batch_size, player_count * 2), dtype=jnp.int32),
            "selected_log_probs": jnp.zeros((time_steps, batch_size, player_count), dtype=jnp.float32),
            "values": zeros,
            "rewards": zeros,
            "dones": zeros,
            "terminal_episode_steps": jnp.ones((time_steps, batch_size), dtype=jnp.float32),
        }
    )
    rollout = RolloutOutput(
        trajectory=TrajectoryBatch(**trajectory_data),
        final_state=None,
        bootstrap_values=jnp.zeros((batch_size,), dtype=jnp.float32),
        final_selector_values=jnp.zeros((batch_size,), dtype=jnp.float32),
        final_flat_obs=jnp.zeros((batch_size, flat_obs_dim), dtype=jnp.float32),
        final_action_mask=None,
    )
    spec = IntentDiscriminatorSpec(
        encoder_type="mlp_mean",
        input_dim=flat_obs_dim + player_count + 11,
        hidden_dim=8,
        num_intents=4,
        learning_rate=3e-4,
        batch_size=8,
        updates_per_rollout=1,
        beta_target=0.01,
        warmup_updates=0,
        ramp_updates=1,
        warmup_steps=0,
        ramp_steps=1,
        bonus_clip=2.0,
        eval_holdout_fraction=0.25,
        dropout=0.0,
        max_obs_dim=flat_obs_dim,
        action_dim_per_player=5,
        training_player_count=player_count,
        token_player_count=2,
        token_dim=3,
        global_dim=1,
        set_heads=1,
        set_cls_tokens=1,
        include_shot_clock=True,
        include_pressure_exposure=True,
    )
    training_mask = jnp.asarray([[1, 0], [0, 1], [1, 1]], dtype=jnp.float32)

    _, _, active_mask = build_intent_step_features_from_rollout(
        rollout,
        spec,
        jnp,
        training_mask=training_mask,
    )

    np.testing.assert_array_equal(np.asarray(active_mask), np.asarray(training_mask, dtype=bool))


def test_completed_episode_ppo_weights_sum_episode_losses():
    jax = pytest.importorskip("jax")
    jnp = jax.numpy

    time_steps = 4
    batch_size = 1
    flat_obs_dim = 3
    player_count = 3
    zeros = jnp.zeros((time_steps, batch_size), dtype=jnp.float32)
    trajectory_data = {field: zeros for field in TrajectoryBatch._fields}
    trajectory_data.update(
        {
            "active_mask": jnp.ones((time_steps, batch_size), dtype=jnp.float32),
            "episode_start": jnp.asarray([[1], [0], [1], [0]], dtype=jnp.int8),
            "flat_obs": jnp.zeros((time_steps, batch_size, flat_obs_dim), dtype=jnp.float32),
            "policy_intent_index": jnp.zeros((time_steps, batch_size), dtype=jnp.int32),
            "policy_intent_gate": jnp.zeros((time_steps, batch_size), dtype=jnp.float32),
            "action_mask": jnp.zeros((time_steps, batch_size, player_count, 5), dtype=jnp.float32),
            "actions": jnp.zeros((time_steps, batch_size, player_count), dtype=jnp.int32),
            "full_actions": jnp.zeros((time_steps, batch_size, player_count * 2), dtype=jnp.int32),
            "selected_log_probs": jnp.zeros((time_steps, batch_size, player_count), dtype=jnp.float32),
            "values": zeros,
            "rewards": jnp.asarray([[0.2], [0.3], [5.0], [5.0]], dtype=jnp.float32),
            "dones": jnp.asarray([[0], [1], [0], [0]], dtype=jnp.int8),
            "terminal_episode_steps": jnp.asarray([[0], [2], [0], [0]], dtype=jnp.int32),
        }
    )
    rollout = RolloutOutput(
        trajectory=TrajectoryBatch(**trajectory_data),
        final_state=None,
        bootstrap_values=jnp.zeros((batch_size,), dtype=jnp.float32),
        final_selector_values=jnp.zeros((batch_size,), dtype=jnp.float32),
        final_flat_obs=jnp.zeros((batch_size, flat_obs_dim), dtype=jnp.float32),
        final_action_mask=None,
    )
    config = TrainerConfig(
        kernel_batch_size=batch_size,
        rollout_horizon=time_steps,
        num_updates=1,
        gamma=1.0,
        gae_lambda=1.0,
        ppo_clip_range=0.2,
        value_coef=0.5,
        entropy_coef=0.0,
        learning_rate=1e-3,
        policy_update_epochs=1,
        ppo_minibatches=1,
        ppo_completed_episodes_only=True,
    )

    ppo_batch = build_ppo_batch(rollout, config, jax, jnp)

    np.testing.assert_array_equal(np.asarray(ppo_batch.active_mask), [1.0, 1.0, 0.0, 0.0])
    np.testing.assert_array_equal(np.asarray(ppo_batch.loss_weights), [1.0, 1.0, 0.0, 0.0])
    np.testing.assert_array_equal(np.asarray(ppo_batch.loss_denominator), [1.0, 1.0, 1.0, 1.0])


def test_ppo_eligible_episode_metrics_use_training_mask():
    jnp = pytest.importorskip("jax.numpy")

    time_steps = 4
    batch_size = 1
    zeros = jnp.zeros((time_steps, batch_size), dtype=jnp.float32)
    trajectory_data = {field: zeros for field in TrajectoryBatch._fields}
    trajectory_data.update(
        {
            "active_mask": jnp.ones((time_steps, batch_size), dtype=jnp.float32),
            "rewards": jnp.asarray([[0.2], [0.3], [5.0], [5.0]], dtype=jnp.float32),
            "dones": jnp.asarray([[0], [1], [0], [0]], dtype=jnp.int8),
            "pass_attempts": jnp.asarray([[1], [0], [1], [1]], dtype=jnp.float32),
            "turnovers": jnp.asarray([[0], [1], [0], [0]], dtype=jnp.float32),
            "learner_turnovers": jnp.asarray([[0], [1], [0], [0]], dtype=jnp.float32),
            "shot_attempts": jnp.asarray([[0], [0], [1], [1]], dtype=jnp.float32),
            "terminal_episode_steps": jnp.asarray([[0], [2], [0], [0]], dtype=jnp.int32),
            "turnover_intercepted": jnp.asarray([[0], [1], [0], [0]], dtype=jnp.float32),
        }
    )
    training_mask = jnp.asarray([[1], [1], [0], [0]], dtype=jnp.float32)

    metrics = summarize_ppo_eligible_episode_metrics(
        "test_ppo_eligible",
        TrajectoryBatch(**trajectory_data),
        training_mask,
    )

    assert metrics["test_ppo_eligible_active_step_count"] == pytest.approx(2.0)
    assert metrics["test_ppo_eligible_completed_episodes"] == pytest.approx(1.0)
    assert metrics["test_ppo_eligible_completed_episode_count"] == pytest.approx(1.0)
    assert metrics["test_ppo_eligible_completed_active_step_count"] == pytest.approx(2.0)
    assert metrics["test_ppo_eligible_reward_total"] == pytest.approx(0.5)
    assert metrics["test_ppo_eligible_reward_per_step"] == pytest.approx(0.25)
    assert metrics["test_ppo_eligible_reward_per_completed_episode"] == pytest.approx(0.5)
    assert metrics["test_ppo_eligible_mean_completed_episode_length"] == pytest.approx(2.0)
    assert metrics["test_ppo_eligible_pass_attempts_total"] == pytest.approx(1.0)
    assert metrics["test_ppo_eligible_terminal_turnover_share"] == pytest.approx(1.0)
    assert metrics["test_ppo_eligible_terminal_turnover_intercepted_share"] == pytest.approx(1.0)
    assert metrics["test_ppo_eligible_shot_attempts_total"] == pytest.approx(0.0)


def test_reward_by_intent_metrics_attribute_completed_episodes_to_start_intent():
    jnp = pytest.importorskip("jax.numpy")

    time_steps = 4
    batch_size = 2
    zeros = jnp.zeros((time_steps, batch_size), dtype=jnp.float32)
    trajectory_data = {field: zeros for field in TrajectoryBatch._fields}
    trajectory_data.update(
        {
            "active_mask": jnp.ones((time_steps, batch_size), dtype=jnp.float32),
            "episode_start": jnp.asarray(
                [[1, 1], [0, 0], [1, 1], [0, 0]],
                dtype=jnp.int8,
            ),
            "policy_intent_index": jnp.asarray(
                [[1, 2], [1, 2], [2, 1], [2, 1]],
                dtype=jnp.int32,
            ),
            "rewards": jnp.asarray(
                [[0.5, 1.0], [0.7, 2.0], [3.0, 4.0], [3.0, 5.0]],
                dtype=jnp.float32,
            ),
            "terminal_episode_steps": jnp.asarray(
                [[0, 0], [2, 2], [0, 0], [0, 2]],
                dtype=jnp.int32,
            ),
        }
    )

    metrics = summarize_reward_by_intent_metrics(
        "test_intent",
        TrajectoryBatch(**trajectory_data),
        num_intents=3,
    )

    assert metrics["test_intent_completed_episodes_by_intent/0"] == pytest.approx(0.0)
    assert metrics["test_intent_completed_episodes_by_intent/1"] == pytest.approx(2.0)
    assert metrics["test_intent_completed_episodes_by_intent/2"] == pytest.approx(1.0)
    assert metrics["test_intent_reward_per_completed_episode_by_intent/1"] == pytest.approx(5.1)
    assert metrics["test_intent_reward_per_completed_episode_by_intent/2"] == pytest.approx(3.0)

    training_mask = jnp.asarray(
        [[1, 1], [1, 1], [0, 0], [0, 0]],
        dtype=jnp.float32,
    )
    ppo_metrics = summarize_reward_by_intent_metrics(
        "test_ppo_intent",
        TrajectoryBatch(**trajectory_data),
        num_intents=3,
        training_mask=training_mask,
    )
    assert ppo_metrics["test_ppo_intent_completed_episodes_by_intent/1"] == pytest.approx(1.0)
    assert ppo_metrics["test_ppo_intent_completed_episodes_by_intent/2"] == pytest.approx(1.0)
    assert ppo_metrics["test_ppo_intent_reward_per_completed_episode_by_intent/1"] == pytest.approx(1.2)
    assert ppo_metrics["test_ppo_intent_reward_per_completed_episode_by_intent/2"] == pytest.approx(3.0)


def test_jax_trainer_allows_intent_runtime_overrides():
    args = parse_args(
        [
            "--enable-intent-learning",
            "true",
            "--enable-defense-intent-learning",
            "true",
            "--num-intents",
            "6",
            "--intent-commitment-steps",
            "5",
            "--intent-null-prob",
            "0.1",
            "--defense-intent-null-prob",
            "0.2",
            "--intent-visible-to-defense-prob",
            "0.3",
        ]
    )

    validate_train_args(args)
    assert args.enable_intent_learning is True
    assert args.enable_defense_intent_learning is True
    assert args.num_intents == 6
    assert args.intent_commitment_steps == 5
    assert args.intent_null_prob == 0.1
    assert args.defense_intent_null_prob == 0.2
    assert args.intent_visible_to_defense_prob == 0.3


def test_jax_trainer_attention_model_enables_set_obs():
    args = parse_args(["--policy-model", "attention"])

    validate_train_args(args)
    assert args.policy_model == "attention"
    assert args.use_set_obs is True


def test_jax_trainer_validates_intent_embedding_runtime_requirements():
    valid_args = parse_args(
        [
            "--policy-model",
            "attention",
            "--intent-embedding-enabled",
            "--enable-intent-learning",
            "true",
        ]
    )
    validate_train_args(valid_args)

    invalid_args = parse_args(["--intent-embedding-enabled"])
    with pytest.raises(SystemExit, match="requires --policy-model attention"):
        validate_train_args(invalid_args)

    no_runtime_args = parse_args(
        [
            "--policy-model",
            "attention",
            "--intent-embedding-enabled",
        ]
    )
    with pytest.raises(SystemExit, match="requires --enable-intent-learning"):
        validate_train_args(no_runtime_args)


def test_jax_trainer_validates_intent_selector_requirements():
    valid_args = parse_args(
        [
            "--policy-model",
            "attention",
            "--intent-embedding-enabled",
            "--enable-intent-learning",
            "true",
            "--intent-selector-enabled",
            "true",
        ]
    )
    validate_train_args(valid_args)

    missing_attention_args = parse_args(
        [
            "--intent-selector-enabled",
            "true",
            "--enable-intent-learning",
            "true",
            "--intent-embedding-enabled",
        ]
    )
    with pytest.raises(SystemExit, match="requires --policy-model attention"):
        validate_train_args(missing_attention_args)

    missing_embedding_args = parse_args(
        [
            "--policy-model",
            "attention",
            "--enable-intent-learning",
            "true",
            "--intent-selector-enabled",
            "true",
        ]
    )
    with pytest.raises(SystemExit, match="requires --intent-embedding-enabled"):
        validate_train_args(missing_embedding_args)


def test_jax_trainer_supports_selector_learning_rate_override():
    default_args = parse_args(["--learning-rate", "0.001"])
    default_config = build_trainer_config(default_args)
    assert _selector_learning_rate_for_args(default_args, default_config) == pytest.approx(0.001)

    override_args = parse_args(
        [
            "--learning-rate",
            "0.001",
            "--intent-selector-learning-rate",
            "0.0003",
        ]
    )
    validate_train_args(override_args)
    override_config = build_trainer_config(override_args)
    assert _selector_learning_rate_for_args(override_args, override_config) == pytest.approx(0.0003)

    invalid_args = parse_args(["--intent-selector-learning-rate", "0"])
    with pytest.raises(SystemExit, match="must be > 0"):
        validate_train_args(invalid_args)


def test_jax_trainer_validates_intent_diversity_requirements():
    scaffold_args = parse_args(
        [
            "--policy-model",
            "attention",
            "--intent-embedding-enabled",
            "--enable-intent-learning",
            "true",
            "--intent-diversity-enabled",
            "true",
        ]
    )
    with pytest.raises(SystemExit, match="supported only with --run-train-loop"):
        validate_train_args(scaffold_args)

    missing_embedding_args = parse_args(
        [
            "--run-train-loop",
            "--policy-model",
            "attention",
            "--enable-intent-learning",
            "true",
            "--intent-diversity-enabled",
            "true",
        ]
    )
    with pytest.raises(SystemExit, match="requires --intent-embedding-enabled"):
        validate_train_args(missing_embedding_args)

    gru_args = parse_args(
        [
            "--run-train-loop",
            "--policy-model",
            "attention",
            "--intent-embedding-enabled",
            "--enable-intent-learning",
            "true",
            "--intent-diversity-enabled",
            "true",
            "--intent-disc-encoder-type",
            "gru",
        ]
    )
    with pytest.raises(SystemExit, match="mlp_mean or set_step"):
        validate_train_args(gru_args)


def test_jax_trainer_validates_ppo_minibatches_divide_batch_size():
    args = parse_args(
        [
            "--run-train-loop",
            "--kernel-batch-size",
            "4",
            "--rollout-horizon",
            "4",
            "--ppo-minibatches",
            "3",
        ]
    )

    with pytest.raises(SystemExit, match="ppo_batch_size=32"):
        validate_train_args(args)


def test_limit_selector_batch_samples_compacts_active_rows():
    jnp = pytest.importorskip("jax.numpy")
    batch = SelectorBatch(
        flat_obs=jnp.arange(20, dtype=jnp.float32).reshape(10, 2),
        chosen_intents=jnp.arange(10, dtype=jnp.int32),
        old_log_probs=jnp.arange(10, dtype=jnp.float32) + 0.1,
        old_values=jnp.arange(10, dtype=jnp.float32) + 0.2,
        advantages=jnp.arange(10, dtype=jnp.float32) + 0.3,
        returns=jnp.arange(10, dtype=jnp.float32) + 0.4,
        active_mask=jnp.asarray([0, 1, 0, 1, 1, 0, 1, 0, 1, 0], dtype=jnp.float32),
    )

    limited = limit_selector_batch_samples(batch, jnp, max_samples=3)

    assert tuple(limited.flat_obs.shape) == (3, 2)
    np.testing.assert_array_equal(
        np.asarray(limited.flat_obs),
        np.asarray(batch.flat_obs)[[1, 3, 4]],
    )
    np.testing.assert_array_equal(np.asarray(limited.chosen_intents), [1, 3, 4])
    np.testing.assert_array_equal(np.asarray(limited.active_mask), [1.0, 1.0, 1.0])


def test_selector_update_helpers_freeze_shared_policy_params():
    jax = pytest.importorskip("jax")
    jnp = jax.numpy

    params = {
        "attention_block_0": {"kernel": jnp.asarray([1.0], dtype=jnp.float32)},
        "policy_head_offense": {"bias": jnp.asarray([2.0], dtype=jnp.float32)},
        "intent_selector_head_0": {"kernel": jnp.asarray([3.0], dtype=jnp.float32)},
        "intent_selector_head_out": {"bias": jnp.asarray([4.0], dtype=jnp.float32)},
        "intent_selector_value_head_0": {"kernel": jnp.asarray([5.0], dtype=jnp.float32)},
        "intent_selector_value_head_out": {"bias": jnp.asarray([6.0], dtype=jnp.float32)},
    }
    candidate_params = jax.tree_util.tree_map(lambda value: value + 10.0, params)

    merged = _merge_selector_update_params(params, candidate_params, jax)

    np.testing.assert_allclose(np.asarray(merged["attention_block_0"]["kernel"]), [1.0])
    np.testing.assert_allclose(np.asarray(merged["policy_head_offense"]["bias"]), [2.0])
    np.testing.assert_allclose(np.asarray(merged["intent_selector_head_0"]["kernel"]), [13.0])
    np.testing.assert_allclose(np.asarray(merged["intent_selector_head_out"]["bias"]), [14.0])
    np.testing.assert_allclose(
        np.asarray(merged["intent_selector_value_head_0"]["kernel"]),
        [15.0],
    )
    np.testing.assert_allclose(
        np.asarray(merged["intent_selector_value_head_out"]["bias"]),
        [16.0],
    )

    grads = jax.tree_util.tree_map(lambda value: jnp.ones_like(value), params)
    masked_grads = _mask_selector_update_grads(grads, jnp.asarray(1.0), jax, jnp)

    np.testing.assert_allclose(np.asarray(masked_grads["attention_block_0"]["kernel"]), [0.0])
    np.testing.assert_allclose(np.asarray(masked_grads["policy_head_offense"]["bias"]), [0.0])
    np.testing.assert_allclose(np.asarray(masked_grads["intent_selector_head_0"]["kernel"]), [1.0])
    np.testing.assert_allclose(np.asarray(masked_grads["intent_selector_head_out"]["bias"]), [1.0])
    np.testing.assert_allclose(
        np.asarray(masked_grads["intent_selector_value_head_0"]["kernel"]),
        [1.0],
    )
    np.testing.assert_allclose(
        np.asarray(masked_grads["intent_selector_value_head_out"]["bias"]),
        [1.0],
    )

    inactive_grads = _mask_selector_update_grads(grads, jnp.asarray(0.0), jax, jnp)
    assert all(
        np.allclose(np.asarray(leaf), 0.0)
        for leaf in jax.tree_util.tree_leaves(inactive_grads)
    )

    active_scoped_params = _apply_selector_update_param_scope(
        params,
        candidate_params,
        jnp.asarray(1.0),
        jax,
        jnp,
    )
    inactive_scoped_params = _apply_selector_update_param_scope(
        params,
        candidate_params,
        jnp.asarray(0.0),
        jax,
        jnp,
    )

    np.testing.assert_allclose(
        np.asarray(active_scoped_params["attention_block_0"]["kernel"]),
        [1.0],
    )
    np.testing.assert_allclose(
        np.asarray(active_scoped_params["intent_selector_head_0"]["kernel"]),
        [13.0],
    )
    np.testing.assert_allclose(
        np.asarray(inactive_scoped_params["attention_block_0"]["kernel"]),
        [1.0],
    )
    np.testing.assert_allclose(
        np.asarray(inactive_scoped_params["intent_selector_head_0"]["kernel"]),
        [3.0],
    )


def test_multiselect_boundaries_do_not_apply_random_fallback_intents():
    jnp = pytest.importorskip("jax.numpy")

    class State:
        intent_active = jnp.asarray([1, 1], dtype=jnp.int8)
        intent_age = jnp.asarray([6, 6], dtype=jnp.int32)
        intent_commitment_remaining = jnp.asarray([0, 2], dtype=jnp.int32)

    (
        _episode_start,
        commitment_timeout,
        completed_pass,
        used,
        applied,
        fallback_used,
    ) = _selector_segment_application_masks(
        State,
        alpha_used=jnp.asarray([False, False]),
        multiselect_enabled=jnp.asarray(True),
        completed_pass_boundary=jnp.asarray([False, True]),
        selector_min_play_steps=4,
        jnp=jnp,
    )

    np.testing.assert_array_equal(np.asarray(commitment_timeout), [True, False])
    np.testing.assert_array_equal(np.asarray(completed_pass), [False, True])
    np.testing.assert_array_equal(np.asarray(used), [False, False])
    np.testing.assert_array_equal(np.asarray(applied), [False, False])
    np.testing.assert_array_equal(np.asarray(fallback_used), [False, False])


def test_build_trainer_config_uses_training_args():
    args = parse_args(
        [
            "--kernel-batch-size",
            "32",
            "--rollout-horizon",
            "128",
            "--num-updates",
            "200",
            "--gae-lambda",
            "0.9",
            "--ppo-clip-range",
            "0.15",
            "--vf-coef",
            "0.7",
            "--ent-coef",
            "0.02",
            "--learning-rate",
            "0.001",
            "--ppo-minibatches",
            "16",
        ]
    )
    validate_train_args(args)
    config = build_trainer_config(args)

    assert config.kernel_batch_size == 32
    assert config.rollout_horizon == 128
    assert config.num_updates == 200
    assert config.gae_lambda == 0.9
    assert config.ppo_clip_range == 0.15
    assert config.value_coef == 0.7
    assert config.entropy_coef == 0.02
    assert config.learning_rate == 0.001
    assert config.ppo_minibatches == 16


def test_jax_task_reward_scale_prefers_update_schedule():
    args = parse_args(
        [
            "--kernel-batch-size",
            "4",
            "--rollout-horizon",
            "4",
            "--task-reward-scale-start",
            "0.1",
            "--task-reward-scale-end",
            "1.0",
            "--task-reward-scale-warmup-steps",
            "999999",
            "--task-reward-scale-ramp-steps",
            "999999",
            "--task-reward-scale-warmup-updates",
            "2",
            "--task-reward-scale-ramp-updates",
            "4",
        ]
    )
    validate_train_args(args)

    assert _task_reward_scale_for_update(args, 1) == pytest.approx(0.1)
    assert _task_reward_scale_for_update(args, 2) == pytest.approx(0.1)
    assert _task_reward_scale_for_update(args, 4) == pytest.approx(0.55)
    assert _task_reward_scale_for_update(args, 6) == pytest.approx(1.0)


def test_jax_entropy_coef_supports_linear_and_exp_schedules():
    linear_args = parse_args(
        [
            "--num-updates",
            "6",
            "--ent-coef",
            "0.01",
            "--ent-coef-start",
            "0.02",
            "--ent-coef-end",
            "0.002",
            "--ent-schedule",
            "linear",
        ]
    )
    validate_train_args(linear_args)
    assert _entropy_coef_for_update(linear_args, 1) == pytest.approx(0.02)
    assert _entropy_coef_for_update(linear_args, 6) == pytest.approx(0.002)

    exp_args = parse_args(
        [
            "--num-updates",
            "3",
            "--ent-coef-start",
            "0.02",
            "--ent-coef-end",
            "0.002",
            "--ent-schedule",
            "exp",
        ]
    )
    validate_train_args(exp_args)
    assert _entropy_coef_for_update(exp_args, 1) == pytest.approx(0.02)
    assert _entropy_coef_for_update(exp_args, 2) == pytest.approx(
        float(np.sqrt(0.02 * 0.002))
    )
    assert _entropy_coef_for_update(exp_args, 3) == pytest.approx(0.002)


def test_mlflow_metric_profile_defaults_to_core():
    args = parse_args([])

    assert args.mlflow_metric_profile == "core"


def test_core_mlflow_train_metric_filter_drops_redundant_aliases():
    metrics = {
        "learner_shot_attempts": 10,
        "offense_all_shot_attempts": 10,
        "offense_learner_shot_attempts": 10,
        "defense_all_shot_makes": 4,
        "defense_opponent_shot_makes": 4,
        "offense_offense_points_total": 12.0,
        "offense_learner_points_total": 12.0,
        "defense_offense_points_total": 7.0,
        "defense_opponent_points_total": 7.0,
        "offense_mean_reward": 0.1,
        "offense_learner_mean_reward": 0.1,
        "offense_learner_shot_dunk_share": 0.3,
        "offense_ppo_eligible_shot_attempts_total": 100,
        "offense_ppo_eligible_shot_attempts_per_step": 0.1,
        "offense_ppo_eligible_shot_attempts_per_completed_episode": 1.1,
        "offense_ppo_eligible_learner_shot_attempts_per_completed_episode": 0.9,
        "offense_ppo_eligible_terminal_shot_episodes": 10,
        "offense_ppo_eligible_terminal_shot_share": 0.8,
        "offense_ppo_eligible_reward_per_step": 0.09,
        "offense_intent_usage_count/0": 42,
        "offense_intent_usage_share/0": 0.25,
        "intent_disc_label_count_by_intent/0": 21,
        "intent_disc_label_prob_by_intent/0": 0.125,
        "selector_used_count": 500,
        "selector_usage_by_intent/0": 0.2,
        "end_to_end_steps_per_sec": 30000.0,
    }

    filtered = _filter_mlflow_train_metrics(metrics)

    assert "learner_shot_attempts" not in filtered
    assert "offense_all_shot_attempts" not in filtered
    assert "defense_all_shot_makes" not in filtered
    assert "offense_offense_points_total" not in filtered
    assert "defense_offense_points_total" not in filtered
    assert "offense_mean_reward" not in filtered
    assert "offense_ppo_eligible_shot_attempts_total" not in filtered
    assert "offense_ppo_eligible_shot_attempts_per_step" not in filtered
    assert "offense_ppo_eligible_shot_attempts_per_completed_episode" not in filtered
    assert "offense_ppo_eligible_terminal_shot_episodes" not in filtered
    assert "offense_intent_usage_count/0" not in filtered
    assert "intent_disc_label_count_by_intent/0" not in filtered
    assert "offense_learner_shot_attempts" not in filtered
    assert "defense_opponent_shot_makes" not in filtered
    assert "offense_learner_points_total" not in filtered
    assert "defense_opponent_points_total" not in filtered
    assert filtered["offense_learner_mean_reward"] == 0.1
    assert filtered["offense_learner_shot_dunk_share"] == 0.3
    assert filtered["offense_ppo_eligible_learner_shot_attempts_per_completed_episode"] == 0.9
    assert filtered["offense_ppo_eligible_terminal_shot_share"] == 0.8
    assert filtered["offense_ppo_eligible_reward_per_step"] == 0.09
    assert filtered["offense_intent_usage_share/0"] == 0.25
    assert filtered["intent_disc_label_prob_by_intent/0"] == 0.125
    assert filtered["selector_used_count"] == 500
    assert filtered["selector_usage_by_intent/0"] == 0.2
    assert filtered["end_to_end_steps_per_sec"] == 30000.0
    assert _filter_mlflow_train_metrics(metrics, profile="full") == metrics


def test_mlflow_params_include_jax_env_skill_stds():
    class Recorder:
        def __init__(self):
            self.params = None

        def log_params(self, params):
            self.params = dict(params)

    args = parse_args(
        [
            "--layup-std",
            "0.05",
            "--three-pt-std",
            "0.05",
            "--dunk-std",
            "0.3",
            "--dunk-pct",
            "0.6",
            "--ent-coef-start",
            "0.02",
            "--ent-coef-end",
            "0.003",
            "--ent-schedule",
            "exp",
            "--enable-intent-learning",
            "true",
            "--enable-defense-intent-learning",
            "true",
            "--num-intents",
            "6",
            "--intent-commitment-steps",
            "5",
            "--intent-null-prob",
            "0.1",
            "--defense-intent-null-prob",
            "0.2",
            "--intent-visible-to-defense-prob",
            "0.3",
            "--task-reward-scale-start",
            "0.1",
            "--task-reward-scale-end",
            "1.0",
            "--task-reward-scale-warmup-updates",
            "50",
            "--task-reward-scale-ramp-updates",
            "300",
            "--enable-phi-shaping",
            "true",
            "--reward-shaping-gamma",
            "0.97",
            "--phi-beta-start",
            "0.01",
            "--phi-beta-end",
            "0.15",
            "--phi-beta-warmup-updates",
            "50",
            "--phi-beta-ramp-updates",
            "200",
            "--phi-blend-weight",
            "0.5",
        ]
    )
    validate_train_args(args)
    recorder = Recorder()
    _log_mlflow_params(
        recorder,
        args,
        build_trainer_config(args),
        ActorCriticSpec(
            flat_obs_dim=91,
            training_player_count=3,
            action_dim_per_player=14,
            total_action_dim=42,
            hidden_dims=(128, 128),
        ),
    )

    assert recorder.params["jax/env/layup_std"] == 0.05
    assert recorder.params["jax/env/three_pt_std"] == 0.05
    assert recorder.params["jax/env/dunk_std"] == 0.3
    assert recorder.params["jax/env/allow_dunks"] is True
    assert recorder.params["jax/env/dunk_pct"] == 0.6
    assert recorder.params["jax/env/layup_pct"] == args.layup_pct
    assert recorder.params["jax/env/pass_mode"] == args.pass_mode
    assert recorder.params["jax/env/enable_intent_learning"] is True
    assert recorder.params["jax/env/enable_defense_intent_learning"] is True
    assert recorder.params["jax/env/num_intents"] == 6
    assert recorder.params["jax/env/intent_commitment_steps"] == 5
    assert recorder.params["jax/env/intent_null_prob"] == 0.1
    assert recorder.params["jax/env/defense_intent_null_prob"] == 0.2
    assert recorder.params["jax/env/intent_visible_to_defense_prob"] == 0.3
    assert recorder.params["jax/env/enable_phi_shaping"] is True
    assert recorder.params["jax/env/reward_shaping_gamma"] == 0.97
    assert recorder.params["jax/env/phi_beta_start"] == 0.01
    assert recorder.params["jax/env/phi_beta_end"] == 0.15
    assert recorder.params["jax/env/phi_beta_warmup_updates"] == 50
    assert recorder.params["jax/env/phi_beta_ramp_updates"] == 200
    assert recorder.params["jax/env/phi_blend_weight"] == 0.5
    assert "jax/enable_phi_shaping" not in recorder.params
    assert "jax/reward_shaping_gamma" not in recorder.params
    assert "jax/phi_beta_start" not in recorder.params
    assert "jax/phi_beta_end" not in recorder.params
    assert "jax/phi_beta_warmup_updates" not in recorder.params
    assert "jax/phi_beta_ramp_updates" not in recorder.params
    assert "jax/phi_blend_weight" not in recorder.params
    assert recorder.params["jax/intent_embedding_enabled"] is False
    assert recorder.params["jax/intent_embedding_dim"] == 16
    assert recorder.params["jax/num_intents"] == 8
    assert recorder.params["jax/task_reward_scale_start"] == 0.1
    assert recorder.params["jax/task_reward_scale_end"] == 1.0
    assert recorder.params["jax/task_reward_scale_warmup_updates"] == 50
    assert recorder.params["jax/task_reward_scale_ramp_updates"] == 300
    assert recorder.params["jax/ent_coef_start"] == 0.02
    assert recorder.params["jax/ent_coef_end"] == 0.003
    assert recorder.params["jax/ent_schedule"] == "exp"
    assert recorder.params["jax/mlflow_metric_profile"] == "core"


def test_train_scaffold_emits_rollout_trajectory_shapes():
    pytest.importorskip("jax")

    args = parse_args(
        [
            "--kernel-batch-size",
            "4",
            "--warmup-iters",
            "0",
            "--benchmark-iters",
            "1",
            "--rollout-horizon",
            "4",
            "--no-progress",
        ]
    )
    validate_train_args(args)
    result = run_train_scaffold(args)

    spec = result["trajectory_spec"]
    flat_obs_dim = 112
    assert spec["trajectory_flat_obs_shape"] == [4, 4, flat_obs_dim]
    assert spec["trajectory_policy_intent_index_shape"] == [4, 4]
    assert spec["trajectory_policy_intent_gate_shape"] == [4, 4]
    assert spec["trajectory_action_mask_shape"] == [4, 4, 3, 14]
    assert spec["trajectory_actions_shape"] == [4, 4, 3]
    assert spec["trajectory_full_actions_shape"] == [4, 4, 6]
    assert spec["trajectory_log_prob_shape"] == [4, 4, 3]
    assert spec["trajectory_values_shape"] == [4, 4]
    assert spec["trajectory_rewards_shape"] == [4, 4]
    assert spec["trajectory_dones_shape"] == [4, 4]
    assert spec["trajectory_pass_attempts_shape"] == [4, 4]
    assert spec["trajectory_completed_passes_shape"] == [4, 4]
    assert spec["trajectory_assists_shape"] == [4, 4]
    assert spec["trajectory_turnovers_shape"] == [4, 4]
    assert spec["trajectory_offensive_three_seconds_shape"] == [4, 4]
    assert spec["trajectory_defensive_lane_violations_shape"] == [4, 4]
    assert spec["trajectory_terminal_episode_steps_shape"] == [4, 4]
    assert spec["trajectory_offense_score_delta_shape"] == [4, 4]
    assert spec["trajectory_defense_score_delta_shape"] == [4, 4]
    assert spec["bootstrap_values_shape"] == [4]
    assert spec["ppo_batch_flat_obs_shape"] == [16, flat_obs_dim]
    assert spec["ppo_batch_policy_intent_index_shape"] == [16]
    assert spec["ppo_batch_policy_intent_gate_shape"] == [16]
    assert spec["ppo_batch_action_mask_shape"] == [16, 3, 14]
    assert spec["ppo_batch_actions_shape"] == [16, 3]
    assert spec["ppo_batch_old_log_probs_shape"] == [16, 3]
    assert spec["ppo_batch_advantages_shape"] == [16]
    assert spec["ppo_batch_returns_shape"] == [16]
    assert result["ppo_update_updates_per_sec"] > 0.0
    assert "total_loss" in result["ppo_update_final_metrics"]


def test_train_scaffold_supports_attention_policy_model():
    pytest.importorskip("jax")

    args = parse_args(
        [
            "--policy-model",
            "attention",
            "--action-head-mode",
            "pointer_targeted",
            "--attention-embed-dim",
            "16",
            "--attention-num-heads",
            "4",
            "--attention-token-mlp-dim",
            "12",
            "--attention-pi-head-hidden-dims",
            "8",
            "8",
            "--attention-vf-head-hidden-dims",
            "8",
            "8",
            "--attention-head-activation",
            "relu",
            "--intent-embedding-enabled",
            "--intent-embedding-dim",
            "6",
            "--enable-intent-learning",
            "true",
            "--enable-defense-intent-learning",
            "true",
            "--kernel-batch-size",
            "2",
            "--warmup-iters",
            "0",
            "--benchmark-iters",
            "1",
            "--rollout-horizon",
            "2",
            "--ppo-minibatches",
            "2",
            "--no-progress",
        ]
    )
    validate_train_args(args)
    result = run_train_scaffold(args)

    assert result["policy_spec"]["model_type"] == "attention"
    assert result["policy_spec"]["action_head_mode"] == "pointer_targeted"
    assert result["policy_spec"]["token_player_count"] == 6
    assert result["policy_spec"]["token_dim"] == TOKEN_OBS_PLAYER_DIM
    assert result["policy_spec"]["global_dim"] == TOKEN_OBS_GLOBAL_DIM
    assert result["policy_spec"]["attention_pi_head_hidden_dims"] == (8, 8)
    assert result["policy_spec"]["attention_vf_head_hidden_dims"] == (8, 8)
    assert result["policy_spec"]["attention_head_activation"] == "relu"
    assert result["policy_spec"]["intent_embedding_enabled"] is True
    assert result["policy_spec"]["intent_embedding_dim"] == 6
    assert result["trainer_config"]["ppo_minibatches"] == 2
    token_obs_dim = (6 * TOKEN_OBS_PLAYER_DIM) + TOKEN_OBS_GLOBAL_DIM + TOKEN_OBS_ROLE_FLAG_DIM
    assert result["trajectory_spec"]["flat_obs_shape"] == [2, token_obs_dim]
    assert result["trajectory_spec"]["trajectory_flat_obs_shape"] == [2, 2, token_obs_dim]
    assert result["ppo_update_final_metrics"]["total_loss"] != 0.0


def test_train_loop_emits_history_and_eval_dumps():
    pytest.importorskip("jax")

    args = parse_args(
        [
            "--run-train-loop",
            "--kernel-batch-size",
            "4",
            "--rollout-horizon",
            "4",
            "--num-updates",
            "2",
            "--policy-update-epochs",
            "1",
            "--ppo-minibatches",
            "2",
            "--log-every-updates",
            "1",
            "--eval-every-updates",
            "1",
            "--eval-horizon",
            "4",
            "--max-eval-dumps",
            "2",
            "--no-progress",
        ]
    )
    validate_train_args(args)
    result = run_training_loop(args)

    assert result["status"] == "train_loop"
    assert result["trainer_config"]["ppo_minibatches"] == 2
    assert set(result["training_player_ids"]) == {"offense", "defense"}
    assert len(result["train_history"]) == 2
    assert len(result["eval_trajectories"]) == 2
    assert result["final_metrics"]["update_index"] == 2
    assert "mean_reward" in result["final_metrics"]
    assert "offense_mean_reward" in result["final_metrics"]
    assert "defense_mean_reward" in result["final_metrics"]
    assert "offense_learner_mean_reward" in result["final_metrics"]
    assert "defense_learner_mean_reward" in result["final_metrics"]
    assert "offense_opponent_mean_reward" in result["final_metrics"]
    assert "defense_opponent_mean_reward" in result["final_metrics"]
    assert "offense_learner_points_per_completed_episode" in result["final_metrics"]
    assert "defense_opponent_points_per_completed_episode" in result["final_metrics"]
    assert "offense_ppo_eligible_completed_episodes" in result["final_metrics"]
    assert "offense_completed_episode_count" in result["final_metrics"]
    assert "offense_completed_active_step_count" in result["final_metrics"]
    assert "offense_ppo_eligible_completed_episode_count" in result["final_metrics"]
    assert "offense_ppo_eligible_completed_active_step_count" in result["final_metrics"]
    assert "defense_ppo_eligible_completed_episodes" in result["final_metrics"]
    assert "offense_ppo_eligible_reward_per_completed_episode" in result["final_metrics"]
    assert "offense_ppo_eligible_reward_per_step" in result["final_metrics"]
    assert "offense_ppo_eligible_terminal_turnover_share" in result["final_metrics"]
    assert "offense_ppo_eligible_terminal_shot_share" in result["final_metrics"]
    assert result["final_metrics"]["offense_opponent_mean_reward"] == pytest.approx(
        -result["final_metrics"]["offense_learner_mean_reward"]
    )
    assert result["final_metrics"]["defense_opponent_mean_reward"] == pytest.approx(
        -result["final_metrics"]["defense_learner_mean_reward"]
    )
    assert result["final_metrics"]["steps_per_update"] == 32
    assert result["final_metrics"]["ppo_batch_size"] == 32
    assert result["final_metrics"]["ppo_update_epochs"] == 1
    assert result["final_metrics"]["ppo_update_minibatches"] == 2
    assert result["final_metrics"]["ppo_update_minibatch_size"] == 16
    assert result["final_metrics"]["rollout_time_pct"] > 0.0
    assert result["final_metrics"]["ppo_update_time_pct"] > 0.0
    assert result["final_metrics"]["rollout_time_pct"] + result["final_metrics"]["ppo_update_time_pct"] == pytest.approx(
        100.0
    )
    assert result["final_metrics"]["ppo_update_optimizer_samples_per_sec"] > 0.0
    assert result["final_metrics"]["approx_kl"] > 0.0
    assert result["final_metrics"]["mean_abs_log_ratio"] > 0.0
    assert result["final_metrics"]["max_abs_log_ratio"] > 0.0
    assert "completed_episodes" in result["final_metrics"]
    assert "completed_episode_count" in result["final_metrics"]
    assert "completed_active_step_count" in result["final_metrics"]
    assert "active_step_count" in result["final_metrics"]
    assert "ppo_used_active_step_count" in result["final_metrics"]
    assert "ppo_unused_active_step_count" in result["final_metrics"]
    assert "ppo_used_completed_episode_count" in result["final_metrics"]
    assert "ppo_used_completed_active_step_count" in result["final_metrics"]
    assert "ppo_unused_completed_episode_count" in result["final_metrics"]
    assert "ppo_unused_completed_active_step_count" in result["final_metrics"]
    assert "cumulative_active_step_count" in result["final_metrics"]
    assert "cumulative_ppo_used_active_step_count" in result["final_metrics"]
    assert "cumulative_ppo_unused_active_step_count" in result["final_metrics"]
    assert "cumulative_completed_episode_count" in result["final_metrics"]
    assert "cumulative_completed_active_step_count" in result["final_metrics"]
    assert "cumulative_ppo_used_completed_episode_count" in result["final_metrics"]
    assert "cumulative_ppo_used_completed_active_step_count" in result["final_metrics"]
    assert "cumulative_ppo_unused_completed_episode_count" in result["final_metrics"]
    assert "cumulative_ppo_unused_completed_active_step_count" in result["final_metrics"]
    assert result["final_metrics"]["cumulative_completed_episode_count"] == pytest.approx(
        result["final_metrics"]["cumulative_ppo_used_completed_episode_count"]
        + result["final_metrics"]["cumulative_ppo_unused_completed_episode_count"]
    )
    assert result["final_metrics"]["cumulative_active_step_count"] == pytest.approx(
        result["final_metrics"]["cumulative_ppo_used_active_step_count"]
        + result["final_metrics"]["cumulative_ppo_unused_active_step_count"]
    )
    assert result["final_metrics"]["cumulative_completed_active_step_count"] == pytest.approx(
        result["final_metrics"]["cumulative_ppo_used_completed_active_step_count"]
        + result["final_metrics"]["cumulative_ppo_unused_completed_active_step_count"]
    )
    assert "mean_completed_episode_length" in result["final_metrics"]
    assert "mean_pass_attempts_per_completed_episode" in result["final_metrics"]
    assert "mean_assists_per_completed_episode" in result["final_metrics"]
    assert "mean_turnovers_per_completed_episode" in result["final_metrics"]
    assert "total_offensive_three_seconds" in result["final_metrics"]
    assert "total_3_second_violations" in result["final_metrics"]
    assert "mean_3_second_violations_per_completed_episode" in result["final_metrics"]
    assert "three_second_violation_rate_per_step" in result["final_metrics"]
    assert "mean_defensive_lane_violations_per_completed_episode" in result["final_metrics"]
    assert "total_defensive_lane_violations" in result["final_metrics"]
    first_eval = result["eval_trajectories"][0]
    assert first_eval["training_role"] in {"offense", "defense"}
    assert first_eval["trajectory_length"] == 4
    assert first_eval["positions"].shape == (4, 6, 2)
    assert first_eval["full_actions"].shape == (4, 6)
    assert first_eval["pass_attempts"].shape == (4,)
    assert first_eval["completed_passes"].shape == (4,)
    assert first_eval["assists"].shape == (4,)
    assert first_eval["turnovers"].shape == (4,)
    assert first_eval["offensive_three_seconds"].shape == (4,)
    assert first_eval["defensive_lane_violations"].shape == (4,)
    assert first_eval["terminal_episode_steps"].shape == (4,)


def test_train_loop_runs_intent_selector_segment_start_metrics():
    pytest.importorskip("jax")

    args = parse_args(
        [
            "--run-train-loop",
            "--policy-model",
            "attention",
            "--attention-embed-dim",
            "16",
            "--attention-num-heads",
            "4",
            "--attention-token-mlp-dim",
            "12",
            "--intent-embedding-enabled",
            "--intent-embedding-dim",
            "4",
            "--enable-intent-learning",
            "true",
            "--num-intents",
            "4",
            "--intent-commitment-steps",
            "2",
            "--intent-null-prob",
            "0.0",
            "--intent-selector-enabled",
            "true",
            "--intent-selector-hidden-dim",
            "8",
            "--intent-selector-alpha-start",
            "1.0",
            "--intent-selector-alpha-end",
            "1.0",
            "--ent-coef-start",
            "0.02",
            "--ent-coef-end",
            "0.02",
            "--task-reward-scale-start",
            "0.25",
            "--task-reward-scale-end",
            "0.25",
            "--intent-selector-multiselect-enabled",
            "true",
            "--kernel-batch-size",
            "4",
            "--rollout-horizon",
            "5",
            "--num-updates",
            "1",
            "--policy-update-epochs",
            "1",
            "--ppo-minibatches",
            "2",
            "--log-every-updates",
            "1",
            "--eval-every-updates",
            "0",
            "--no-progress",
        ]
    )
    validate_train_args(args)
    result = run_training_loop(args)

    metrics = result["final_metrics"]
    assert result["policy_spec"]["intent_selector_enabled"] is True
    assert result["policy_spec"]["intent_selector_hidden_dim"] == 8
    assert metrics["selector_alpha"] == pytest.approx(1.0)
    assert metrics["selector_eps"] == pytest.approx(0.0)
    assert metrics["entropy_coef"] == pytest.approx(0.02)
    assert metrics["task_reward_scale"] == pytest.approx(0.25)
    assert metrics["task_reward_scale_is_scheduled"] == pytest.approx(1.0)
    assert metrics["selector_used_count"] > 0
    assert metrics["selector_usage_rate"] > 0.0
    assert metrics["selector_applied_count"] >= metrics["selector_used_count"]
    assert metrics["selector_boundary_commitment_timeout_count"] > 0
    assert metrics["selector_entropy"] > 0.0
    assert metrics["selector_train_sample_count"] > 0.0
    assert "selector_train_loss" in metrics
    assert "selector_train_approx_kl" in metrics
    assert "selector_train_clip_fraction" in metrics
    assert "selector_train_usage_by_intent/0" in metrics
    assert "selector_usage_by_intent/0" in metrics


def test_train_loop_skips_empty_selector_update_during_warmup():
    pytest.importorskip("jax")

    args = parse_args(
        [
            "--run-train-loop",
            "--policy-model",
            "attention",
            "--attention-embed-dim",
            "16",
            "--attention-num-heads",
            "4",
            "--attention-token-mlp-dim",
            "12",
            "--intent-embedding-enabled",
            "--intent-embedding-dim",
            "4",
            "--enable-intent-learning",
            "true",
            "--num-intents",
            "4",
            "--intent-selector-enabled",
            "true",
            "--intent-selector-hidden-dim",
            "8",
            "--intent-selector-alpha-start",
            "0.0",
            "--intent-selector-alpha-end",
            "0.0",
            "--intent-selector-train-every-rollouts",
            "1",
            "--kernel-batch-size",
            "4",
            "--rollout-horizon",
            "4",
            "--num-updates",
            "1",
            "--policy-update-epochs",
            "1",
            "--ppo-minibatches",
            "2",
            "--log-every-updates",
            "1",
            "--eval-every-updates",
            "0",
            "--no-progress",
        ]
    )
    validate_train_args(args)
    result = run_training_loop(args)

    metrics = result["final_metrics"]
    assert metrics["selector_alpha"] == pytest.approx(0.0)
    assert metrics["selector_train_sample_count"] == pytest.approx(0.0)
    assert metrics["selector_train_skipped_empty"] == pytest.approx(1.0)
    assert "selector_train_loss" not in metrics


def test_train_loop_checkpoint_resume_round_trip(tmp_path):
    pytest.importorskip("jax")

    checkpoint_dir = tmp_path / "jax_ckpts"
    first_args = parse_args(
        [
            "--run-train-loop",
            "--kernel-batch-size",
            "4",
            "--rollout-horizon",
            "4",
            "--num-updates",
            "1",
            "--policy-update-epochs",
            "1",
            "--log-every-updates",
            "1",
            "--eval-every-updates",
            "1",
            "--eval-horizon",
            "4",
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--checkpoint-every-updates",
            "1",
            "--layup-std",
            "0.05",
            "--three-pt-std",
            "0.05",
            "--dunk-std",
            "0.3",
            "--dunk-pct",
            "0.6",
            "--no-progress",
        ]
    )
    validate_train_args(first_args)
    first_result = run_training_loop(first_args)

    latest_checkpoint = checkpoint_dir / "latest"
    assert first_result["latest_checkpoint_path"] == str(latest_checkpoint)
    assert latest_checkpoint.is_dir()
    assert (latest_checkpoint / "metadata.json").is_file()
    assert (latest_checkpoint / "state").is_dir()

    payload = load_checkpoint(latest_checkpoint)
    assert payload["update_index"] == 1
    assert "train_history" not in payload
    assert payload["env_config"]["layup_std"] == 0.05
    assert payload["env_config"]["three_pt_std"] == 0.05
    assert payload["env_config"]["dunk_std"] == 0.3
    assert payload["env_config"]["allow_dunks"] is True
    assert payload["env_config"]["dunk_pct"] == 0.6
    assert payload["play_name_metadata"]["backend"] == "jax"
    assert payload["play_name_metadata"]["pool_version"] >= 1
    assert payload["play_name_metadata"]["model_codename"]
    assert payload["play_name_metadata"]["num_intents"] == int(first_args.num_intents)
    assert len(payload["play_name_map"]) == int(first_args.num_intents)
    assert "learner_shot_dunk_share" in payload["last_metrics"]
    assert "learner_shot_two_share" in payload["last_metrics"]
    assert "learner_shot_three_share" in payload["last_metrics"]
    assert "opponent_shot_dunk_share" in payload["last_metrics"]
    assert "opponent_shot_two_share" in payload["last_metrics"]
    assert "opponent_shot_three_share" in payload["last_metrics"]
    assert set(payload["current_state"]) == {"offense", "defense"}
    assert set(payload["eval_initial_state"]) == {"offense", "defense"}

    resumed_args = parse_args(
        [
            "--run-train-loop",
            "--kernel-batch-size",
            "4",
            "--rollout-horizon",
            "4",
            "--num-updates",
            "2",
            "--policy-update-epochs",
            "1",
            "--log-every-updates",
            "1",
            "--eval-every-updates",
            "1",
            "--eval-horizon",
            "4",
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--checkpoint-every-updates",
            "1",
            "--resume-checkpoint",
            str(latest_checkpoint),
            "--layup-std",
            "0.05",
            "--three-pt-std",
            "0.05",
            "--dunk-std",
            "0.3",
            "--dunk-pct",
            "0.6",
            "--no-progress",
        ]
    )
    validate_train_args(resumed_args)
    resumed_result = run_training_loop(resumed_args)

    assert resumed_result["resumed_from_checkpoint"] == str(latest_checkpoint)
    assert resumed_result["final_metrics"]["update_index"] == 2
    assert resumed_result["final_metrics"]["cumulative_active_step_count"] >= (
        first_result["final_metrics"]["cumulative_active_step_count"]
    )
    assert resumed_result["final_metrics"]["cumulative_completed_episode_count"] >= (
        first_result["final_metrics"]["cumulative_completed_episode_count"]
    )
    assert resumed_result["final_metrics"]["cumulative_completed_active_step_count"] >= (
        first_result["final_metrics"]["cumulative_completed_active_step_count"]
    )
    assert len(resumed_result["train_history"]) == 1


def test_train_loop_accepts_frozen_opponent_checkpoint(tmp_path):
    pytest.importorskip("jax")

    checkpoint_dir = tmp_path / "opponent_ckpts"
    checkpoint_args = parse_args(
        [
            "--run-train-loop",
            "--kernel-batch-size",
            "4",
            "--rollout-horizon",
            "4",
            "--num-updates",
            "1",
            "--policy-update-epochs",
            "1",
            "--log-every-updates",
            "1",
            "--eval-every-updates",
            "3",
            "--eval-horizon",
            "4",
            "--max-eval-dumps",
            "2",
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--checkpoint-every-updates",
            "1",
            "--no-progress",
        ]
    )
    validate_train_args(checkpoint_args)
    first_result = run_training_loop(checkpoint_args)
    opponent_checkpoint = checkpoint_dir / "latest"
    assert first_result["latest_checkpoint_path"] == str(opponent_checkpoint)

    train_args = parse_args(
        [
            "--run-train-loop",
            "--kernel-batch-size",
            "4",
            "--rollout-horizon",
            "4",
            "--num-updates",
            "1",
            "--policy-update-epochs",
            "1",
            "--log-every-updates",
            "1",
            "--eval-every-updates",
            "1",
            "--eval-horizon",
            "4",
            "--max-eval-dumps",
            "2",
            "--frozen-opponent-checkpoint",
            str(opponent_checkpoint),
            "--no-progress",
        ]
    )
    validate_train_args(train_args)
    result = run_training_loop(train_args)

    assert result["active_opponent"]["source"] in {"checkpoint", "local_checkpoint"}
    assert result["active_opponent"].get("checkpoint_path", str(opponent_checkpoint)) == str(opponent_checkpoint)
    assert result["opponent_pool_size"] >= 1
    assert result["final_metrics"]["update_index"] == 1
    assert "offense_mean_reward" in result["final_metrics"]
    assert "defense_mean_reward" in result["final_metrics"]


def test_train_loop_samples_newly_saved_opponents_from_pool(tmp_path):
    pytest.importorskip("jax")

    checkpoint_dir = tmp_path / "pool_ckpts"
    args = parse_args(
        [
            "--run-train-loop",
            "--kernel-batch-size",
            "4",
            "--rollout-horizon",
            "4",
            "--num-updates",
            "2",
            "--policy-update-epochs",
            "1",
            "--log-every-updates",
            "1",
            "--eval-every-updates",
            "3",
            "--eval-horizon",
            "4",
            "--max-eval-dumps",
            "2",
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--checkpoint-every-updates",
            "1",
            "--no-progress",
        ]
    )
    validate_train_args(args)
    result = run_training_loop(args)

    assert result["opponent_pool_size"] == 2
    assert result["active_opponent"]["candidate_kind"] == "self_checkpoint"
    assert result["active_opponent"]["source"] == "local_checkpoint"
    assert result["final_metrics"]["opponent_source"] == "local_checkpoint"
    assert result["final_metrics"]["opponent_update_index"] == 1


def test_train_loop_groups_sampled_opponents_when_per_env_enabled(tmp_path):
    pytest.importorskip("jax")

    checkpoint_dir = tmp_path / "grouped_pool_ckpts"
    args = parse_args(
        [
            "--run-train-loop",
            "--kernel-batch-size",
            "4",
            "--rollout-horizon",
            "4",
            "--num-updates",
            "3",
            "--policy-update-epochs",
            "1",
            "--log-every-updates",
            "1",
            "--eval-every-updates",
            "3",
            "--eval-horizon",
            "4",
            "--max-eval-dumps",
            "2",
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--checkpoint-every-updates",
            "1",
            "--grouped-opponent-sampling",
            "--opponent-group-count",
            "2",
            "--no-progress",
        ]
    )
    validate_train_args(args)
    result = run_training_loop(args)

    assert result["opponent_pool_size"] == 3
    assert result["active_opponent"]["source"] == "grouped_pool"
    assert result["active_opponent"]["group_count"] == 2
    assert len(result["active_opponent"]["groups"]) == 2
    assert len(result["eval_trajectories"]) == 2
    assert result["final_metrics"]["opponent_source"] == "grouped_pool"
    assert result["final_metrics"]["opponent_group_count"] == 2


def test_train_loop_runs_offense_intent_discriminator_and_sample_dump(tmp_path):
    pytest.importorskip("jax")

    checkpoint_dir = tmp_path / "intent_disc_ckpts"
    args = parse_args(
        [
            "--run-train-loop",
            "--policy-model",
            "attention",
            "--action-head-mode",
            "pointer_targeted",
            "--attention-embed-dim",
            "16",
            "--attention-num-heads",
            "4",
            "--attention-token-mlp-dim",
            "12",
            "--intent-embedding-enabled",
            "--intent-embedding-dim",
            "4",
            "--enable-intent-learning",
            "true",
            "--num-intents",
            "4",
            "--intent-null-prob",
            "0.0",
            "--intent-diversity-enabled",
            "true",
            "--intent-diversity-warmup-updates",
            "0",
            "--intent-diversity-ramp-updates",
            "1",
            "--intent-diversity-beta-target",
            "0.01",
            "--intent-disc-hidden-dim",
            "16",
            "--intent-disc-dropout",
            "0.1",
            "--intent-disc-batch-size",
            "8",
            "--intent-disc-updates-per-rollout",
            "1",
            "--intent-disc-eval-holdout-fraction",
            "0.5",
            "--intent-disc-encoder-type",
            "set_step",
            "--intent-disc-include-shot-clock",
            "false",
            "--intent-disc-include-pressure-exposure",
            "false",
            "--kernel-batch-size",
            "2",
            "--rollout-horizon",
            "2",
            "--num-updates",
            "1",
            "--policy-update-epochs",
            "1",
            "--log-every-updates",
            "1",
            "--eval-every-updates",
            "0",
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--checkpoint-every-updates",
            "1",
            "--disc-eval-batch-output",
            "true",
            "--intent-sample-dump-size",
            "3",
            "--no-progress",
        ]
    )
    validate_train_args(args)
    result = run_training_loop(args)

    metrics = result["final_metrics"]
    assert metrics["intent_disc_active_count"] > 0.0
    assert metrics["intent_disc_loss"] > 0.0
    assert metrics["intent_disc_top1_acc_trainbatch"] >= 0.0
    assert metrics["intent_disc_top1_acc_holdout"] >= 0.0
    assert metrics["intent_disc_auc_ovr_macro_trainbatch"] >= 0.0
    assert metrics["intent_disc_auc_ovr_macro_holdout"] >= 0.0
    assert metrics["intent_disc_trainbatch_size"] > 0.0
    assert metrics["intent_disc_holdout_size"] > 0.0
    assert 0.0 <= metrics["intent_disc_holdout_fraction_realized"] <= 1.0
    assert "intent_disc_label_prob_by_intent/0" in metrics
    assert "intent_disc_pred_prob_by_intent/0" in metrics
    assert metrics["intent_bonus_beta"] == pytest.approx(0.01)
    assert metrics["intent_bonus_active_sample_count"] > 0
    assert result["intent_discriminator_config"]["num_intents"] == 4
    assert result["intent_discriminator_config"]["encoder_type"] == "set_step"
    assert result["intent_discriminator_config"]["dropout"] == pytest.approx(0.1)
    assert result["intent_discriminator_config"]["eval_holdout_fraction"] == pytest.approx(0.5)
    assert result["intent_discriminator_config"]["include_shot_clock"] is False
    assert result["intent_discriminator_config"]["include_pressure_exposure"] is False
    assert len(result["intent_sample_artifacts"]) == 1
    sample_path = result["intent_sample_artifacts"][0]
    payload = load_checkpoint(checkpoint_dir / "latest")
    assert "intent_discriminator_state" in payload
    assert payload["play_name_metadata"]["backend"] == "jax"
    assert payload["play_name_metadata"]["num_intents"] == 4
    assert set(payload["play_name_map"]) == {"0", "1", "2", "3"}
    assert result["play_name_map"] == payload["play_name_map"]
    with np.load(sample_path) as sample:
        assert sample["features"].shape[0] <= 3
        assert sample["embedding"].shape[0] == sample["features"].shape[0]
        assert sample["intent_index"].shape[0] == sample["features"].shape[0]
        assert sample["players"].shape[0] == sample["features"].shape[0]
        assert sample["globals"].shape[1] == TOKEN_OBS_GLOBAL_DIM
        assert np.all(sample["globals"][:, 0] == 0.0)
        assert np.all(sample["globals"][:, 1] == 0.0)


def test_train_loop_skips_intent_discriminator_during_warmup(tmp_path):
    pytest.importorskip("jax")

    checkpoint_dir = tmp_path / "intent_disc_warmup_ckpts"
    args = parse_args(
        [
            "--run-train-loop",
            "--policy-model",
            "attention",
            "--action-head-mode",
            "pointer_targeted",
            "--attention-embed-dim",
            "16",
            "--attention-num-heads",
            "4",
            "--attention-token-mlp-dim",
            "12",
            "--intent-embedding-enabled",
            "--intent-embedding-dim",
            "4",
            "--enable-intent-learning",
            "true",
            "--num-intents",
            "4",
            "--intent-null-prob",
            "0.0",
            "--intent-diversity-enabled",
            "true",
            "--intent-diversity-warmup-updates",
            "10",
            "--intent-diversity-ramp-updates",
            "1",
            "--intent-diversity-beta-target",
            "0.01",
            "--intent-disc-hidden-dim",
            "16",
            "--intent-disc-batch-size",
            "8",
            "--intent-disc-updates-per-rollout",
            "1",
            "--intent-disc-encoder-type",
            "set_step",
            "--kernel-batch-size",
            "2",
            "--rollout-horizon",
            "2",
            "--num-updates",
            "1",
            "--policy-update-epochs",
            "1",
            "--log-every-updates",
            "1",
            "--eval-every-updates",
            "0",
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--checkpoint-every-updates",
            "1",
            "--disc-eval-batch-output",
            "true",
            "--intent-sample-dump-size",
            "3",
            "--no-progress",
        ]
    )
    validate_train_args(args)
    result = run_training_loop(args)

    metrics = result["final_metrics"]
    assert metrics["intent_bonus_beta"] == pytest.approx(0.0)
    assert metrics["intent_disc_skipped_warmup"] == pytest.approx(1.0)
    assert "intent_disc_loss" not in metrics
    assert "intent_disc_top1_acc_trainbatch" not in metrics
    assert "intent_bonus_active_sample_count" not in metrics
    assert result["intent_sample_artifacts"] == []
