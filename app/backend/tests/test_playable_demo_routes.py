from starlette.requests import Request
from starlette.responses import Response

from app.backend.routes import playable_routes
from app.backend.schemas import PlayableDemoTakeoverRequest, PlayableStepRequest
from app.backend.state import GameState


class DummyPlayableEnv:
    def __init__(self):
        self.reset_calls = []
        self.shot_clock = 24
        self.last_action_results = {}
        self.pass_mode = "directional"
        self.training_team = None
        self.shot_clock_steps = 24
        self.min_shot_clock = 24
        self.offense_layup_pct_by_player = [0.6]
        self.offense_three_pt_pct_by_player = [0.35]
        self.offense_dunk_pct_by_player = [0.2]

    def reset(self, options=None):
        self.reset_calls.append(options or {})
        if isinstance(options, dict) and "shot_clock" in options:
            self.shot_clock = int(options["shot_clock"])
        return {"obs": []}, {}


def _build_dummy_playable_state(*, demo_mode: bool = False) -> GameState:
    state = GameState()
    state.env = DummyPlayableEnv()
    state.playable_session = {
        "active": True,
        "players_per_side": 1,
        "difficulty": "demo" if demo_mode else "easy",
        "run_id": "demo-run" if demo_mode else "live-run",
        "checkpoint_index": 42,
        "policy_name": "unified_iter_42.zip",
        "score": {"user": 0, "ai": 0},
        "possession_number": 1,
        "user_on_offense": True,
        "side_skills": {"user": {}, "ai": {}},
        "period_mode": "period",
        "total_periods": 1,
        "current_period": 1,
        "period_length_minutes": 5,
        "period_seconds_remaining": 300,
        "game_over": False,
        "turn_index": 1,
        "demo_mode": demo_mode,
        "analytics": {
            "game_id": "game-id",
            "next_seq": 1,
        },
    }
    return state


def _make_request() -> Request:
    return Request({"type": "http", "headers": []})


def test_playable_step_defaults_missing_user_actions_to_noop(monkeypatch):
    target_state = _build_dummy_playable_state(demo_mode=False)
    captured = {}

    def fake_resolve_session_state(_request, *, require_active):
        assert require_active is True
        return "session-id", target_state

    def fake_lifecycle_step(action_request):
        captured["actions"] = dict(action_request.actions)
        captured["player_deterministic"] = action_request.player_deterministic
        return {
            "status": "success",
            "state": {"done": False},
            "actions_taken": {},
            "actions_taken_meta": {},
        }

    monkeypatch.setattr(playable_routes, "_resolve_session_state", fake_resolve_session_state)
    monkeypatch.setattr(playable_routes, "lifecycle_step", fake_lifecycle_step)
    monkeypatch.setattr(playable_routes, "_map_state_for_playable", lambda state, session: state)

    response = Response()
    payload = playable_routes.playable_step(
        PlayableStepRequest(actions={}, auto_user_actions=False),
        _make_request(),
        response,
    )

    assert payload["status"] == "success"
    assert captured["actions"] == {"0": "NOOP"}
    assert captured["player_deterministic"] is True


def test_playable_step_uses_policy_actions_for_demo_mode(monkeypatch):
    target_state = _build_dummy_playable_state(demo_mode=True)
    captured = {}

    def fake_resolve_session_state(_request, *, require_active):
        assert require_active is True
        return "session-id", target_state

    def fake_lifecycle_step(action_request):
        captured["actions"] = dict(action_request.actions)
        captured["player_deterministic"] = action_request.player_deterministic
        return {
            "status": "success",
            "state": {"done": False},
            "actions_taken": {},
            "actions_taken_meta": {},
        }

    monkeypatch.setattr(playable_routes, "_resolve_session_state", fake_resolve_session_state)
    monkeypatch.setattr(playable_routes, "lifecycle_step", fake_lifecycle_step)
    monkeypatch.setattr(playable_routes, "_map_state_for_playable", lambda state, session: state)

    response = Response()
    payload = playable_routes.playable_step(
        PlayableStepRequest(actions={}, auto_user_actions=False),
        _make_request(),
        response,
    )

    assert payload["status"] == "success"
    assert captured["actions"] == {}
    assert captured["player_deterministic"] is False


def test_playable_options_include_demo_config(monkeypatch):
    monkeypatch.setenv("BW_PLAYABLE_DEMO_ENABLED", "true")
    monkeypatch.setenv("BW_PLAYABLE_DEMO_RUN_ID", "demo-run-id")
    monkeypatch.setenv("BW_PLAYABLE_DEMO_ALTERNATION_NUMBER", "75")
    monkeypatch.delenv("BW_PLAYABLE_DEMO_CHECKPOINT", raising=False)

    body = playable_routes.get_playable_options()
    assert body["status"] == "success"
    assert body["demo"]["enabled"] is True
    assert body["demo"]["available"] is True
    assert body["demo"]["period_mode"] == "period"
    assert body["demo"]["period_length_minutes"] == 5
    assert body["demo"]["session_kind"] == "demo"


def test_playable_demo_takeover_turns_demo_into_live_game(monkeypatch):
    target_state = _build_dummy_playable_state(demo_mode=True)

    def fake_resolve_session_state(_request, *, require_active):
        assert require_active is True
        return "session-id", target_state

    monkeypatch.setattr(playable_routes, "_resolve_session_state", fake_resolve_session_state)
    monkeypatch.setattr(playable_routes, "_build_playable_side_skills", lambda: {"user": {}, "ai": {}})
    monkeypatch.setattr(playable_routes, "_reset_playable_possession", lambda session: {"done": False})
    monkeypatch.setattr(playable_routes, "_map_state_for_playable", lambda state, session: state)

    response = Response()
    payload = playable_routes.playable_demo_takeover(
        PlayableDemoTakeoverRequest(period_mode="quarters", period_length_minutes=7),
        _make_request(),
        response,
    )

    assert payload["status"] == "success"
    assert payload["config"]["demo_mode"] is False
    assert payload["config"]["session_kind"] == "human"
    assert payload["config"]["period_mode"] == "quarters"
    assert payload["config"]["period_length_minutes"] == 7
    assert target_state.playable_session["demo_mode"] is False


def test_reset_playable_possession_runs_selector_episode_init(monkeypatch):
    target_state = _build_dummy_playable_state(demo_mode=False)
    target_state.user_team = None
    captured = {"called": 0}

    def fake_selector_init():
        captured["called"] += 1

    monkeypatch.setattr(playable_routes, "game_state", target_state)
    monkeypatch.setattr(playable_routes, "_capture_turn_start_snapshot", lambda: None)
    monkeypatch.setattr(playable_routes, "get_ui_game_state", lambda: {"done": False})
    monkeypatch.setattr(
        playable_routes,
        "_initialize_app_selector_runtime_for_episode",
        fake_selector_init,
    )

    state = playable_routes._reset_playable_possession(target_state.playable_session)

    assert state == {"done": False}
    assert captured["called"] == 1


def test_reset_playable_possession_uses_jax_runtime_branch(monkeypatch):
    target_state = _build_dummy_playable_state(demo_mode=False)
    target_state.user_team = None
    target_state.playable_session["side_skills"] = {
        "user": {"layup": [0.7], "three_pt": [0.4], "dunk": [0.3]},
        "ai": {"layup": [0.5], "three_pt": [0.2], "dunk": [0.1]},
    }

    class DummyJaxRuntime:
        def __init__(self):
            self.calls = []

        def reset_playable_possession(self, **kwargs):
            self.calls.append(kwargs)
            kwargs["game_state"].env = target_state.env
            kwargs["game_state"].obs = {"obs": []}
            return {"done": False, "jax": True}

    jax_runtime = DummyJaxRuntime()
    target_state.jax_runtime = jax_runtime

    monkeypatch.setattr(playable_routes, "game_state", target_state)

    state = playable_routes._reset_playable_possession(target_state.playable_session)

    assert state == {"done": False, "jax": True}
    assert target_state.env.reset_calls == []
    assert len(jax_runtime.calls) == 1
    call = jax_runtime.calls[0]
    assert call["user_team"] == playable_routes.Team.OFFENSE
    assert call["shot_clock"] == 24
    assert call["offense_skills"] == {"layup": [0.7], "three_pt": [0.4], "dunk": [0.3]}
    assert target_state.episode_states == [{"done": False, "jax": True}]


def test_playable_matrix_json_accepts_policy_name_override(monkeypatch):
    monkeypatch.setenv(
        "BW_PLAYABLE_POLICY_MATRIX_JSON",
        '{"3":{"easy":{"run_id":"jax-run","checkpoint_index":150,"policy_name":"update_150"}}}',
    )

    matrix = playable_routes._get_playable_matrix_from_json()

    assert matrix[3]["easy"]["policy_name"] == "update_150"
