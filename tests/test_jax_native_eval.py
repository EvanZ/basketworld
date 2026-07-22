from __future__ import annotations

import inspect

import numpy as np
import pytest

from basketworld.envs.basketworld_env_v2 import HexagonBasketballEnv, Team
from basketworld_jax.config import TRAIN_FROZEN_VALUES
from basketworld_jax.eval import can_run_native_jax_evaluation, run_native_jax_evaluation
from basketworld_jax.eval.native import (
    _phi_beta_for_eval,
    _post_orb_continuation_diagnostics_from_trace,
    _task_reward_scale_for_eval,
)
from basketworld_jax.train.main import parse_args, run_training_loop, validate_train_args


def _native_eval_env_params() -> dict:
    signature_keys = set(inspect.signature(HexagonBasketballEnv.__init__).parameters)
    params = {
        key: value
        for key, value in TRAIN_FROZEN_VALUES.items()
        if key in signature_keys
    }
    params["training_team"] = Team.OFFENSE
    params["offensive_three_seconds_enabled"] = bool(
        TRAIN_FROZEN_VALUES.get("offensive_three_seconds", False)
    )
    return params


def test_post_orb_continuation_diagnostics_compare_values_with_shaped_returns():
    trace = {
        "active": np.asarray([[1], [1], [1], [0]], dtype=np.int8),
        "terminal_episode_steps": np.asarray([[0], [0], [3], [0]], dtype=np.int32),
        "offensive_rebound": np.asarray([[1], [0], [0], [0]], dtype=np.int8),
        "shot_success": np.asarray([[0], [0], [1], [0]], dtype=np.int8),
        "shot_value": np.asarray([[0.0], [0.0], [2.0], [0.0]], dtype=np.float32),
        "offense_values": np.asarray([[0.0], [0.8], [0.0], [0.0]], dtype=np.float32),
        "defense_values": np.asarray([[0.0], [-0.7], [0.0], [0.0]], dtype=np.float32),
        "offense_rewards": np.zeros((4, 1), dtype=np.float32),
        "defense_rewards": np.zeros((4, 1), dtype=np.float32),
        "offense_training_rewards": np.asarray([[0.4], [0.1], [1.5], [0.0]], dtype=np.float32),
        "defense_training_rewards": np.asarray([[-0.4], [-0.1], [-1.5], [0.0]], dtype=np.float32),
        "done": np.asarray([[0], [0], [1], [0]], dtype=np.int8),
    }

    diagnostics = _post_orb_continuation_diagnostics_from_trace(
        trace,
        env_index=0,
        gamma=0.5,
    )

    assert diagnostics["post_orb_samples"] == 1
    assert diagnostics["post_orb_points_sum"] == pytest.approx(2.0)
    assert diagnostics["post_orb_consensus_value_sum"] == pytest.approx(0.75)
    assert diagnostics["post_orb_offense_shaped_return_sum"] == pytest.approx(0.85)
    assert diagnostics["post_orb_defense_shaped_return_sum"] == pytest.approx(-0.85)
    assert diagnostics["post_orb_consensus_shaped_return_sum"] == pytest.approx(0.85)

    trace["terminal_episode_steps"] = np.zeros((4, 1), dtype=np.int32)
    truncated = _post_orb_continuation_diagnostics_from_trace(
        trace,
        env_index=0,
        gamma=0.5,
    )
    assert truncated["post_orb_samples"] == 0
    assert truncated["post_orb_shaped_return_samples"] == 0


def test_task_reward_scale_for_eval_matches_update_schedule():
    payload = {"update_index": 250, "trainer_config": {}}
    params = {
        "jax/task_reward_scale_start": 0.1,
        "jax/task_reward_scale_end": 1.0,
        "jax/task_reward_scale_warmup_updates": 0,
        "jax/task_reward_scale_ramp_updates": 500,
    }

    assert _task_reward_scale_for_eval(params, payload) == pytest.approx(0.55)
    assert _task_reward_scale_for_eval({}, payload) == pytest.approx(1.0)

    phi_params = {
        "jax/enable_phi_shaping": True,
        "jax/phi_beta_start": 0.0,
        "jax/phi_beta_end": 0.25,
        "jax/phi_beta_warmup_updates": 0,
        "jax/phi_beta_ramp_updates": 500,
    }
    assert _phi_beta_for_eval(phi_params, payload, default=0.0) == pytest.approx(0.125)


def test_native_jax_evaluation_returns_stats_tab_payload(tmp_path):
    pytest.importorskip("jax")

    checkpoint_dir = tmp_path / "native_eval_ckpts"
    args = parse_args(
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
    validate_train_args(args)
    train_result = run_training_loop(args)
    checkpoint_path = train_result["latest_checkpoint_path"]

    assert can_run_native_jax_evaluation(
        unified_policy_path=checkpoint_path,
        opponent_policy_path=None,
        custom_setup=None,
        randomize_offense_permutation=False,
    )

    result = run_native_jax_evaluation(
        num_episodes=3,
        player_deterministic=False,
        opponent_deterministic=False,
        required_params={},
        optional_params=_native_eval_env_params(),
        unified_policy_path=checkpoint_path,
        opponent_policy_path=None,
        user_team_name="OFFENSE",
        role_flag_offense=1.0,
        role_flag_defense=-1.0,
        eval_seed=123,
    )

    assert len(result["results"]) == 3
    assert isinstance(result["shot_accumulator"], dict)
    assert set(result["per_player_stats"]) == set(range(6))
    assert "none" in result["per_intent_stats"]

    player_zero = result["per_player_stats"][0]
    for key in (
        "shots",
        "makes",
        "assists",
        "potential_assists",
        "turnovers",
        "points",
        "episodes",
        "steps",
        "shot_types",
        "shot_chart",
        "assist_full_by_type",
        "unassisted",
    ):
        assert key in player_zero

    diagnostics = result["eval_diagnostics"]
    assert diagnostics["jax_native_summary"]["eval_seed"] == 123
    assert diagnostics["jax_native_summary"]["allow_dunks"] is True
    assert "shot_dunk_share" in diagnostics["jax_native_summary"]
    assert "shot_two_share" in diagnostics["jax_native_summary"]
    assert "shot_three_share" in diagnostics["jax_native_summary"]
    assert "post_orb_shaped_return_sample_count" in diagnostics["jax_native_summary"]
    assert "post_orb_consensus_shaped_return_per_sample" in diagnostics["jax_native_summary"]
    assert "post_orb_critic_minus_shaped_return_per_sample" in diagnostics["jax_native_summary"]
    assert diagnostics["jax_native_summary"]["post_orb_shaped_return_includes_training_intent_bonus"] is False
    assert "post_orb_phi_beta" in diagnostics["jax_native_summary"]
    assert diagnostics["jax_native_summary"]["intent_active_episodes"] == 0
    assert diagnostics["jax_native_summary"]["intent_inactive_episodes"] == 3
    assert diagnostics["jax_native_summary"]["defense_intent_active_episodes"] == 0
    assert diagnostics["jax_native_summary"]["defense_intent_inactive_episodes"] == 3
    for key in (
        "action_mix",
        "reward_breakdown",
        "turnover_reasons",
        "assist_links",
        "potential_assist_links",
        "jax_native_summary",
    ):
        assert key in diagnostics

    for episode in result["results"]:
        assert episode["intent_index"] is None
        assert episode["intent_active"] is False
        assert episode["intent_visible_to_defense"] is False
        assert episode["defense_intent_index"] is None
        assert episode["defense_intent_active"] is False
        outcome_info = episode["outcome_info"]
        assert set(outcome_info) >= {
            "shots",
            "turnovers",
            "defensive_lane_violations",
            "shot_clock",
            "three_point_distance",
        }
