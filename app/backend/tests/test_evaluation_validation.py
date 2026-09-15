from gymnasium import Wrapper

import app.backend.evaluation as backend_evaluation
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
