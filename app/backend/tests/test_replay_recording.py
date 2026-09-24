import pytest
from fastapi import HTTPException

from app.backend.routes import analytics_routes
from app.backend.state import GameState


def _state(*, session_id=None, possessions=0, user_score=0, ai_score=0):
    state = {
        "completed_possessions": possessions,
        "user_score": user_score,
        "ai_score": ai_score,
    }
    if session_id is not None:
        state["replay_session_id"] = session_id
    return state


def test_replay_uses_only_the_active_recording_session(monkeypatch):
    state = GameState()
    state.env = object()
    state.replay_session_id = "active"
    state.episode_states = [
        _state(session_id="stale", possessions=3, user_score=4, ai_score=2),
        _state(session_id="active", possessions=0, user_score=0, ai_score=0),
        _state(session_id="active", possessions=1, user_score=0, ai_score=3),
    ]
    monkeypatch.setattr(analytics_routes, "game_state", state)

    body = analytics_routes.replay_last_episode()

    assert body["states"] == state.episode_states[1:]


def test_replay_rejects_a_non_monotonic_recording(monkeypatch):
    state = GameState()
    state.env = object()
    state.episode_states = [
        _state(possessions=3, user_score=4, ai_score=2),
        _state(possessions=2, user_score=0, ai_score=3),
    ]
    monkeypatch.setattr(analytics_routes, "game_state", state)

    with pytest.raises(HTTPException) as error:
        analytics_routes.replay_last_episode()

    assert error.value.status_code == 409
