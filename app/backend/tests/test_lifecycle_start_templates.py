from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

import app.backend.observations as backend_observations
import app.backend.state as backend_state
from app.backend.routes import lifecycle_routes
from app.backend.schemas import InitGameRequest, StartSelfPlayRequest
from basketworld.envs.basketworld_env_v2 import HexagonBasketballEnv, Team
from basketworld.utils import mlflow_config
from basketworld.utils.start_templates import load_start_template_library


@pytest.fixture
def isolated_game_state(monkeypatch):
    fresh = backend_state.GameState()
    monkeypatch.setattr(backend_state, "game_state", fresh)
    monkeypatch.setattr(lifecycle_routes, "game_state", fresh)
    monkeypatch.setattr(backend_observations, "game_state", fresh)
    return fresh


def _library_path() -> Path:
    return Path(__file__).resolve().parents[3] / "configs" / "start_templates_v2.json"


def test_start_self_play_can_resolve_loaded_template_library(isolated_game_state):
    env = HexagonBasketballEnv(
        players=3,
        allow_dunks=True,
        training_team=Team.OFFENSE,
        render_mode=None,
    )
    obs, _ = env.reset(seed=11)
    isolated_game_state.env = env
    isolated_game_state.obs = obs
    isolated_game_state.user_team = Team.OFFENSE
    isolated_game_state.mlflow_training_params = {
        "start_template_jitter_scale": 0.0,
        "start_template_mirror_prob": 0.0,
    }
    isolated_game_state.mlflow_start_template_library = load_start_template_library(
        _library_path(),
        players_per_side=3,
    )
    isolated_game_state.start_template_library_source = "local_file"

    result = lifecycle_routes.start_self_play(
        StartSelfPlayRequest(
            template_id="top_entry_gap",
            template_mirrored=False,
            template_seed=1234,
        )
    )

    assert result["status"] == "success"
    assert result["start_template"]["template_id"] == "top_entry_gap"
    assert result["start_template"]["mirrored"] is False
    assert result["start_template"]["source"] == "local_file"
    assert isolated_game_state.replay_shot_clock == 24
    assert isolated_game_state.replay_ball_holder in env.offense_ids
    assert isolated_game_state.replay_initial_positions == [
        tuple(pos) for pos in env.positions
    ]
    assert len(set(isolated_game_state.replay_initial_positions)) == env.n_players


def test_init_game_preserves_session_template_library_when_run_has_no_artifact(
    isolated_game_state,
    monkeypatch,
):
    session_library = load_start_template_library(
        _library_path(),
        players_per_side=3,
    )
    isolated_game_state.mlflow_start_template_library = session_library
    isolated_game_state.start_template_library_source = "local_file"
    isolated_game_state.start_template_library_path = str(_library_path())

    class DummyPolicy:
        backend_kind = "sb3"
        metadata = {}

        def __init__(self):
            self.policy = type("DummyPolicyModule", (), {"pass_logit_bias": 0.0})()

    monkeypatch.setattr(mlflow_config, "setup_mlflow", lambda *args, **kwargs: None)
    monkeypatch.setattr(lifecycle_routes.mlflow.tracking, "MlflowClient", lambda: object())
    monkeypatch.setattr(
        lifecycle_routes,
        "get_unified_policy_path",
        lambda client, run_id, unified_name=None: "/tmp/fake.zip",
    )
    monkeypatch.setattr(
        lifecycle_routes,
        "load_inference_policy",
        lambda *args, **kwargs: DummyPolicy(),
    )
    monkeypatch.setattr(
        lifecycle_routes,
        "get_mlflow_params",
        lambda client, run_id: ({"players": 3}, {"allow_dunks": True}),
    )
    monkeypatch.setattr(lifecycle_routes, "get_mlflow_phi_shaping_params", lambda client, run_id: {})
    monkeypatch.setattr(lifecycle_routes, "get_mlflow_training_params", lambda client, run_id: {})
    monkeypatch.setattr(lifecycle_routes, "get_mlflow_start_template_library", lambda client, run_id: None)
    monkeypatch.setattr(lifecycle_routes, "validate_policy_observation_schema", lambda policy, env, obs, **kwargs: obs)
    monkeypatch.setattr(lifecycle_routes, "_compute_param_counts_from_policy", lambda policy: None)

    body = asyncio.run(
        lifecycle_routes.init_game(
            InitGameRequest(
                run_id="run-without-template-artifact",
                user_team_name="OFFENSE",
            )
        )
    )

    assert body["status"] == "success"
    assert body["state"]["start_template_library"]["templates"][0]["id"] == "wing_entry_help"
    assert isolated_game_state.start_template_library_source == "local_file"
    assert isolated_game_state.start_template_library_path == str(_library_path())
