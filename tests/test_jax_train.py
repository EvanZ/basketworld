from __future__ import annotations

import pytest

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


def test_jax_trainer_attention_model_enables_set_obs():
    args = parse_args(["--policy-model", "attention"])

    validate_train_args(args)
    assert args.policy_model == "attention"
    assert args.use_set_obs is True


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
    assert recorder.params["jax/env/dunk_pct"] == 0.6
    assert recorder.params["jax/env/layup_pct"] == args.layup_pct
    assert recorder.params["jax/env/pass_mode"] == args.pass_mode


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
    assert payload["env_config"]["dunk_pct"] == 0.6
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
