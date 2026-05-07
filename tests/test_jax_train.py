from __future__ import annotations

import pytest
import numpy as np

from basketworld_jax.checkpoints import load_checkpoint
from basketworld_jax.models import ActorCriticSpec
from basketworld_jax.train.main import (
    TRAIN_FROZEN_VALUES,
    _log_mlflow_params,
    build_trainer_config,
    parse_args,
    run_train_scaffold,
    run_training_loop,
    validate_train_args,
)


def test_trainer_parser_defaults_match_frozen_scope():
    args = parse_args([])

    for key, expected in TRAIN_FROZEN_VALUES.items():
        assert getattr(args, key) == expected


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
    assert recorder.params["jax/intent_embedding_enabled"] is False
    assert recorder.params["jax/intent_embedding_dim"] == 16
    assert recorder.params["jax/num_intents"] == 8


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
    assert spec["trajectory_flat_obs_shape"] == [4, 4, 91]
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
    assert spec["ppo_batch_flat_obs_shape"] == [16, 91]
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
    assert result["policy_spec"]["token_dim"] == 15
    assert result["policy_spec"]["global_dim"] == 4
    assert result["policy_spec"]["attention_pi_head_hidden_dims"] == (8, 8)
    assert result["policy_spec"]["attention_vf_head_hidden_dims"] == (8, 8)
    assert result["policy_spec"]["attention_head_activation"] == "relu"
    assert result["policy_spec"]["intent_embedding_enabled"] is True
    assert result["policy_spec"]["intent_embedding_dim"] == 6
    assert result["trainer_config"]["ppo_minibatches"] == 2
    assert result["trajectory_spec"]["flat_obs_shape"] == [2, 95]
    assert result["trajectory_spec"]["trajectory_flat_obs_shape"] == [2, 2, 95]
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
    assert "mean_completed_episode_length" in result["final_metrics"]
    assert "mean_pass_attempts_per_completed_episode" in result["final_metrics"]
    assert "mean_assists_per_completed_episode" in result["final_metrics"]
    assert "mean_turnovers_per_completed_episode" in result["final_metrics"]
    assert "total_offensive_three_seconds" in result["final_metrics"]
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
        assert sample["globals"].shape[1] == 4
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
