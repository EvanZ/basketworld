from __future__ import annotations

import pytest

from benchmarks.jax_phase_a_train import (
    PHASE_A_TRAIN_FROZEN_VALUES,
    build_trainer_config,
    parse_args,
    run_phase_a_training_loop,
    run_phase_a_train_scaffold,
    validate_phase_a_train_args,
)


def test_trainer_parser_defaults_match_frozen_scope():
    args = parse_args([])

    for key, expected in PHASE_A_TRAIN_FROZEN_VALUES.items():
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
    validate_phase_a_train_args(args)
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
    validate_phase_a_train_args(args)
    result = run_phase_a_train_scaffold(args)

    spec = result["trajectory_spec"]
    assert spec["trajectory_flat_obs_shape"] == [4, 4, 91]
    assert spec["trajectory_action_mask_shape"] == [4, 4, 3, 14]
    assert spec["trajectory_actions_shape"] == [4, 4, 3]
    assert spec["trajectory_log_prob_shape"] == [4, 4, 3]
    assert spec["trajectory_values_shape"] == [4, 4]
    assert spec["trajectory_rewards_shape"] == [4, 4]
    assert spec["trajectory_dones_shape"] == [4, 4]
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
    validate_phase_a_train_args(args)
    result = run_phase_a_training_loop(args)

    assert result["status"] == "train_loop"
    assert len(result["train_history"]) == 2
    assert len(result["eval_trajectories"]) == 2
    assert result["final_metrics"]["update_index"] == 2
    assert "mean_reward" in result["final_metrics"]
    first_eval = result["eval_trajectories"][0]
    assert first_eval["trajectory_length"] == 4
    assert first_eval["positions"].shape == (4, 6, 2)
    assert first_eval["full_actions"].shape == (4, 6)
