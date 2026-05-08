from __future__ import annotations

import pytest
from fastapi import HTTPException

from app.backend.routes.evaluation_routes import _build_evaluation_optional_params
from app.backend.schemas import EvaluationRequest
from app.backend.state import game_state


def _library() -> dict:
    return {
        "version": 1,
        "players_per_side": 3,
        "templates": [
            {
                "id": "eval_template",
                "weight": 1,
                "mirrorable": True,
                "offense": [],
                "defense": [],
            }
        ],
    }


@pytest.fixture(autouse=True)
def restore_game_state_template_fields():
    previous_optional = game_state.env_optional_params
    previous_library = game_state.mlflow_start_template_library
    previous_source = game_state.start_template_library_source
    try:
        yield
    finally:
        game_state.env_optional_params = previous_optional
        game_state.mlflow_start_template_library = previous_library
        game_state.start_template_library_source = previous_source


def test_eval_enabled_template_mode_uses_loaded_session_library_without_mutating_defaults():
    game_state.env_optional_params = {"start_template_enabled": False}
    game_state.mlflow_start_template_library = _library()
    game_state.start_template_library_source = "local_file"

    params, diagnostics = _build_evaluation_optional_params(
        EvaluationRequest(
            start_template_mode="enabled",
            start_template_prob=0.75,
            start_template_jitter_scale=0.0,
            start_template_mirror_prob=0.25,
        )
    )

    assert params["start_template_enabled"] is True
    assert params["start_template_library"]["templates"][0]["id"] == "eval_template"
    assert params["start_template_prob"] == pytest.approx(0.75)
    assert params["start_template_jitter_scale"] == pytest.approx(0.0)
    assert params["start_template_mirror_prob"] == pytest.approx(0.25)
    assert diagnostics["start_template_mode"] == "enabled"
    assert diagnostics["start_template_source"] == "local_file"
    assert game_state.env_optional_params == {"start_template_enabled": False}


def test_eval_checkpoint_template_mode_does_not_force_loaded_library_when_checkpoint_disabled():
    game_state.env_optional_params = {"start_template_enabled": False}
    game_state.mlflow_start_template_library = _library()

    params, diagnostics = _build_evaluation_optional_params(
        EvaluationRequest(start_template_mode="checkpoint")
    )

    assert params["start_template_enabled"] is False
    assert "start_template_library" not in params
    assert diagnostics["start_template_enabled"] is False


def test_eval_enabled_template_mode_requires_loaded_library():
    game_state.env_optional_params = {}
    game_state.mlflow_start_template_library = None

    with pytest.raises(HTTPException) as exc_info:
        _build_evaluation_optional_params(
            EvaluationRequest(start_template_mode="enabled")
        )

    assert exc_info.value.status_code == 400
