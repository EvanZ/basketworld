from __future__ import annotations

import pytest

from basketworld_jax.checkpoints import load_checkpoint
from basketworld_jax.train.main import (
    TRAIN_FROZEN_VALUES,
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
    assert spec["trajectory_terminal_episode_steps_shape"] == [4, 4]
    assert spec["bootstrap_values_shape"] == [4]
    assert spec["ppo_batch_flat_obs_shape"] == [16, 91]
    assert spec["ppo_batch_action_mask_shape"] == [16, 3, 14]
    assert spec["ppo_batch_actions_shape"] == [16, 3]
    assert spec["ppo_batch_old_log_probs_shape"] == [16, 3]
    assert spec["ppo_batch_advantages_shape"] == [16]
    assert spec["ppo_batch_returns_shape"] == [16]
    assert result["ppo_update_updates_per_sec"] > 0.0
    assert "total_loss" in result["ppo_update_final_metrics"]


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
    assert len(result["train_history"]) == 2
    assert len(result["eval_trajectories"]) == 2
    assert result["final_metrics"]["update_index"] == 2
    assert "mean_reward" in result["final_metrics"]
    assert "completed_episodes" in result["final_metrics"]
    assert "mean_completed_episode_length" in result["final_metrics"]
    assert "mean_pass_attempts_per_completed_episode" in result["final_metrics"]
    assert "mean_assists_per_completed_episode" in result["final_metrics"]
    assert "mean_turnovers_per_completed_episode" in result["final_metrics"]
    first_eval = result["eval_trajectories"][0]
    assert first_eval["trajectory_length"] == 4
    assert first_eval["positions"].shape == (4, 6, 2)
    assert first_eval["full_actions"].shape == (4, 6)
    assert first_eval["pass_attempts"].shape == (4,)
    assert first_eval["completed_passes"].shape == (4,)
    assert first_eval["assists"].shape == (4,)
    assert first_eval["turnovers"].shape == (4,)
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
            "--no-progress",
        ]
    )
    validate_train_args(resumed_args)
    resumed_result = run_training_loop(resumed_args)

    assert resumed_result["resumed_from_checkpoint"] == str(latest_checkpoint)
    assert resumed_result["final_metrics"]["update_index"] == 2
    assert len(resumed_result["train_history"]) == 1
