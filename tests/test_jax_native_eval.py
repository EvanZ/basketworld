from __future__ import annotations

import inspect

import pytest

from basketworld.envs.basketworld_env_v2 import HexagonBasketballEnv, Team
from basketworld_jax.config import TRAIN_FROZEN_VALUES
from basketworld_jax.eval import can_run_native_jax_evaluation, run_native_jax_evaluation
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
