from gymnasium import Wrapper
from types import SimpleNamespace
import pytest

import app.backend.evaluation as backend_evaluation
from app.backend.routes import evaluation_routes
from app.backend.schemas import EvaluationRequest
from basketworld.envs.basketworld_env_v2 import HexagonBasketballEnv, Team


def test_validate_custom_eval_setup_uses_unwrapped_env_for_position_validation():
    env = HexagonBasketballEnv(
        players=3,
        allow_dunks=True,
        enable_intent_learning=True,
        intent_null_prob=0.0,
        training_team=Team.OFFENSE,
    )
    env.reset(seed=123)
    wrapped_env = Wrapper(env)

    custom_setup = {
        "initial_positions": [tuple(pos) for pos in env.positions],
        "ball_holder": int(env.ball_holder),
        "shooting_mode": "random",
    }

    normalized = backend_evaluation.validate_custom_eval_setup(custom_setup, wrapped_env)

    assert normalized["initial_positions"] == [tuple(pos) for pos in env.positions]
    assert int(normalized["ball_holder"]) == int(env.ball_holder)


def test_pass_steal_preview_uses_unwrapped_env_for_position_validation(monkeypatch):
    env = HexagonBasketballEnv(
        players=3,
        allow_dunks=True,
        enable_intent_learning=True,
        intent_null_prob=0.0,
        training_team=Team.OFFENSE,
    )
    env.reset(seed=456)
    wrapped_env = Wrapper(env)

    monkeypatch.setattr(
        backend_evaluation,
        "_predict_policy_actions",
        lambda *args, **kwargs: (None, []),
    )

    result = backend_evaluation.pass_steal_preview(
        wrapped_env,
        [tuple(pos) for pos in env.positions],
        int(env.ball_holder),
    )

    assert "steal_probabilities" in result
    assert "policy_probabilities" in result

def test_validate_custom_eval_setup_accepts_constrained_rebound_skill_sampling():
    env = HexagonBasketballEnv(
        players=5,
        allow_dunks=True,
        enable_intent_learning=True,
        intent_null_prob=0.0,
        training_team=Team.OFFENSE,
    )
    env.reset(seed=789)

    normalized = backend_evaluation.validate_custom_eval_setup(
        {
            "shooting_mode": "random",
            "rebound_skill_sampling": {
                "mode": "constrained_gaussian",
                "std": 0.75,
                "target_edge": 1.25,
                "tolerance": 0.2,
                "max_attempts": 2500,
            },
        },
        env,
    )

    assert normalized["rebound_skill_sampling"] == {
        "mode": "constrained_gaussian",
        "std": 0.75,
        "target_edge": 1.25,
        "tolerance": 0.2,
        "max_attempts": 2500,
    }


def test_evaluation_multi_possession_overrides_are_validated_and_disable_templates(monkeypatch):
    request = EvaluationRequest(
        env_overrides={
            "enable_multi_possession": "true",
            "multi_possession_limit": "7",
            "multi_possession_overtime_round_cap": "9",
            "multi_possession_reward_mode": "point_differential",
            "score_potential_scale": "0.25",
            "game_winner_reward": "5.0",
            "multi_possession_aux_rewards_enabled": "false",
            "multi_possession_use_inbounds": "false",
            "inbound_deadline_steps": "4",
        },
        start_template_mode="enabled",
    )
    monkeypatch.setattr(evaluation_routes.game_state, "env_optional_params", {
        "start_template_enabled": True,
        "start_template_library": {"templates": [{"id": "legacy"}]},
    })
    monkeypatch.setattr(
        evaluation_routes.game_state,
        "mlflow_start_template_library",
        {"templates": [{"id": "legacy"}]},
    )

    optional_params, diagnostics = evaluation_routes._build_evaluation_optional_params(
        request
    )

    assert optional_params["enable_multi_possession"] is True
    assert optional_params["multi_possession_limit"] == 7
    assert optional_params["multi_possession_overtime_round_cap"] == 9
    assert optional_params["multi_possession_reward_mode"] == "point_differential"
    assert optional_params["score_potential_scale"] == pytest.approx(0.25)
    assert optional_params["game_winner_reward"] == pytest.approx(5.0)
    assert optional_params["multi_possession_aux_rewards_enabled"] is False
    assert optional_params["multi_possession_use_inbounds"] is False
    assert optional_params["inbound_deadline_steps"] == 4
    assert optional_params["start_template_enabled"] is False
    assert "start_template_library" not in optional_params
    assert diagnostics["start_template_disabled_reason"] == "multi_possession"

    with pytest.raises(Exception, match="multi_possession_limit"):
        evaluation_routes._coerce_eval_env_override("multi_possession_limit", "bad")
    with pytest.raises(Exception, match="inbound_deadline_steps"):
        evaluation_routes._coerce_eval_env_override("inbound_deadline_steps", 0)
    assert (
        evaluation_routes._coerce_eval_env_override(
            "multi_possession_overtime_round_cap", 0
        )
        == 0
    )
    with pytest.raises(Exception, match="multi_possession_overtime_round_cap"):
        evaluation_routes._coerce_eval_env_override(
            "multi_possession_overtime_round_cap", -1
        )

    assert (
        evaluation_routes._coerce_eval_env_override(
            "multi_possession_reward_mode", "scoring_events"
        )
        == "scoring_events"
    )


def test_evaluation_response_keeps_cutoff_games_incomplete(monkeypatch):
    fake_env = SimpleNamespace(shot_clock_steps=24, min_shot_clock=1)
    monkeypatch.setattr(evaluation_routes.game_state, "env", fake_env)
    monkeypatch.setattr(evaluation_routes.game_state, "unified_policy", object())
    monkeypatch.setattr(evaluation_routes.game_state, "defense_policy", None)
    monkeypatch.setattr(evaluation_routes.game_state, "env_required_params", {"players": 2})
    monkeypatch.setattr(
        evaluation_routes.game_state,
        "env_optional_params",
        {"enable_multi_possession": True, "multi_possession_limit": 3},
    )
    monkeypatch.setattr(evaluation_routes.game_state, "unified_policy_path", "/tmp/policy")
    monkeypatch.setattr(evaluation_routes.game_state, "opponent_policy_path", None)
    monkeypatch.setattr(evaluation_routes.game_state, "mlflow_training_params", {})
    monkeypatch.setattr(evaluation_routes.game_state, "user_team", Team.OFFENSE)
    monkeypatch.setattr(evaluation_routes.game_state, "unified_policy_key", "test")
    monkeypatch.setattr(evaluation_routes.game_state, "opponent_unified_policy_key", None)
    monkeypatch.setattr(evaluation_routes.game_state, "role_flag_offense", 1.0)
    monkeypatch.setattr(evaluation_routes.game_state, "role_flag_defense", -1.0)
    monkeypatch.setattr(evaluation_routes, "eval_validate_custom_eval_setup", lambda *_: {})
    monkeypatch.setattr(evaluation_routes, "reset_evaluation_progress", lambda *_: None)
    monkeypatch.setattr(evaluation_routes, "update_evaluation_progress", lambda *_: None)
    monkeypatch.setattr(evaluation_routes, "get_ui_game_state", lambda: {})
    monkeypatch.setattr(
        evaluation_routes,
        "eval_run_evaluation",
        lambda **_: {
            "results": [
                {
                    "episode": 1,
                    "steps": 36,
                    "completed": False,
                    "truncated": True,
                    "game": {
                        "completed": False,
                        "result": None,
                        "team_a_score": 4.0,
                        "team_b_score": 4.0,
                        "user_score": 4.0,
                        "opponent_score": 4.0,
                        "completed_possessions": 2,
                    },
                    "outcome_info": {},
                }
            ],
            "shot_accumulator": {},
            "eval_diagnostics": {},
        },
    )

    response = evaluation_routes.run_evaluation(EvaluationRequest(num_episodes=1))

    result = response["results"][0]
    assert result["completed"] is False
    assert result["truncated"] is True
    assert result["game"]["result"] is None
    assert result["final_state"]["done"] is False
    assert result["final_state"]["team_a_score"] == pytest.approx(4.0)
    assert result["final_state"]["completed_possessions"] == 2


def test_action_mode_matrix_runs_four_same_seed_native_evaluations(monkeypatch):
    fake_env = SimpleNamespace(shot_clock_steps=24, min_shot_clock=1)
    monkeypatch.setattr(evaluation_routes.game_state, "env", fake_env)
    monkeypatch.setattr(evaluation_routes.game_state, "unified_policy", object())
    monkeypatch.setattr(evaluation_routes.game_state, "defense_policy", None)
    monkeypatch.setattr(evaluation_routes.game_state, "env_required_params", {"players": 2})
    monkeypatch.setattr(
        evaluation_routes.game_state,
        "env_optional_params",
        {"enable_multi_possession": True, "multi_possession_limit": 1},
    )
    monkeypatch.setattr(evaluation_routes.game_state, "unified_policy_path", "/tmp/policy")
    monkeypatch.setattr(evaluation_routes.game_state, "opponent_policy_path", None)
    monkeypatch.setattr(evaluation_routes.game_state, "mlflow_training_params", {})
    monkeypatch.setattr(evaluation_routes.game_state, "user_team", Team.OFFENSE)
    monkeypatch.setattr(evaluation_routes.game_state, "unified_policy_key", "test")
    monkeypatch.setattr(evaluation_routes.game_state, "opponent_unified_policy_key", None)
    monkeypatch.setattr(evaluation_routes.game_state, "role_flag_offense", 1.0)
    monkeypatch.setattr(evaluation_routes.game_state, "role_flag_defense", -1.0)
    monkeypatch.setattr(evaluation_routes, "eval_validate_custom_eval_setup", lambda *_: {})
    monkeypatch.setattr(evaluation_routes, "can_run_native_jax_evaluation", lambda **_: True)
    monkeypatch.setattr(evaluation_routes, "get_ui_game_state", lambda: {})

    progress = []
    monkeypatch.setattr(evaluation_routes, "reset_evaluation_progress", lambda total: progress.append((0, total)))
    monkeypatch.setattr(
        evaluation_routes,
        "update_evaluation_progress",
        lambda completed, total: progress.append((completed, total)),
    )
    calls = []

    def _fake_run(**kwargs):
        calls.append(kwargs)
        kwargs["progress_callback"](kwargs["num_episodes"], kwargs["num_episodes"])
        player_argmax = bool(kwargs["player_deterministic"])
        ai_argmax = bool(kwargs["opponent_deterministic"])
        margin = float(int(player_argmax) - int(ai_argmax))
        return {
            "results": [
                {
                    "episode": index + 1,
                    "steps": 5,
                    "completed": True,
                    "game": {
                        "completed": True,
                        "user_score": 2.0,
                        "opponent_score": 1.0,
                    },
                    "outcome_info": {},
                }
                for index in range(kwargs["num_episodes"])
            ],
            "shot_accumulator": {},
            "eval_diagnostics": {
                "jax_native_summary": {
                    "num_episodes": kwargs["num_episodes"],
                    "eval_seed": kwargs["eval_seed"],
                    "completed_games": kwargs["num_episodes"],
                    "completion_rate": 1.0,
                    "win_count": 1,
                    "tie_count": 0,
                    "loss_count": 1,
                    "completed_margin_mean": margin,
                    "user_points_per_possession": 1.0,
                    "opponent_points_per_possession": 0.5,
                }
            },
        }

    monkeypatch.setattr(evaluation_routes, "eval_run_evaluation", _fake_run)

    response = evaluation_routes.run_evaluation(
        EvaluationRequest(
            num_episodes=2,
            action_mode_matrix=True,
            eval_seed=12345,
        )
    )

    assert len(calls) == 4
    assert {call["eval_seed"] for call in calls} == {12345}
    assert [
        (call["player_deterministic"], call["opponent_deterministic"])
        for call in calls
    ] == [(True, True), (False, True), (True, False), (False, False)]
    matrix = response["eval_diagnostics"]["action_mode_matrix"]
    assert matrix["episodes_per_mode"] == 2
    assert matrix["total_episode_count"] == 8
    assert set(matrix["cells"]) == {
        "argmax_vs_argmax",
        "sampled_vs_argmax",
        "argmax_vs_sampled",
        "sampled_vs_sampled",
    }
    assert progress[-1] == (8, 8)
