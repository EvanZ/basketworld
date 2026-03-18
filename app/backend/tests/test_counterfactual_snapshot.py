import copy
import asyncio

import numpy as np
import pytest
from fastapi.encoders import jsonable_encoder
from fastapi import HTTPException

import app.backend.observations as backend_observations
import app.backend.state as backend_state
import basketworld.utils.mlflow_config as mlflow_config
from app.backend.routes import admin_routes
from app.backend.routes import lifecycle_routes
from app.backend.schemas import InitGameRequest, ReplayCounterfactualRequest
from basketworld.envs.basketworld_env_v2 import ActionType, HexagonBasketballEnv, Team


def _encode(obj):
    return jsonable_encoder(obj, custom_encoder={np.ndarray: lambda arr: arr.tolist()})


@pytest.fixture
def isolated_game_state(monkeypatch):
    fresh = backend_state.GameState()
    monkeypatch.setattr(backend_state, "game_state", fresh)
    monkeypatch.setattr(admin_routes, "game_state", fresh)
    monkeypatch.setattr(lifecycle_routes, "game_state", fresh)
    monkeypatch.setattr(backend_observations, "game_state", fresh)
    return fresh


def _init_live_game(state: backend_state.GameState) -> None:
    env = HexagonBasketballEnv(
        players=3,
        allow_dunks=True,
        enable_intent_learning=True,
        intent_null_prob=0.0,
        enable_defense_intent_learning=True,
        defense_intent_null_prob=0.0,
        intent_commitment_steps=4,
        training_team=Team.OFFENSE,
    )
    obs, _ = env.reset(seed=123)
    env.positions[env.offense_ids[0]] = env.basket_position
    env.ball_holder = env.offense_ids[0]
    env.shot_clock = 11
    env.intent_active = True
    env.intent_index = 3
    env.intent_age = 2
    env.intent_commitment_remaining = 2
    env.defense_intent_active = True
    env.defense_intent_index = 6
    env.defense_intent_age = 1
    env.defense_intent_commitment_remaining = 3
    env.offense_layup_pct_by_player = [0.91, 0.62, 0.58]
    env.offense_three_pt_pct_by_player = [0.35, 0.41, 0.29]
    env.offense_dunk_pct_by_player = [0.96, 0.83, 0.80]

    state.env = env
    state.user_team = Team.OFFENSE
    state.obs = obs
    state.prev_obs = copy.deepcopy(obs)
    state.sampled_offense_skills = {
        "layup": list(env.offense_layup_pct_by_player),
        "three_pt": list(env.offense_three_pt_pct_by_player),
        "dunk": list(env.offense_dunk_pct_by_player),
    }
    state.replay_offense_skills = copy.deepcopy(state.sampled_offense_skills)
    state.reward_history = [
        {"step": 1, "offense": 0.2, "defense": -0.1, "shot_clock": 12},
    ]
    state.episode_rewards = {"offense": 0.2, "defense": -0.1}
    state.shot_log = [{"step": 1, "player_id": 0, "success": False}]
    state.phi_log = [{"step": 1, "phi_r_shape": 0.0}]
    state.actions_log = [[0 for _ in range(env.n_players)]]
    state.episode_states = [{"step": 1, "shot_clock": 12}]
    state.playable_session = {
        "active": True,
        "score": {"user": 1, "ai": 0},
        "period_seconds_remaining": 244,
        "turn_index": 2,
        "user_on_offense": True,
    }
    backend_state._rebuild_cached_obs()
    backend_state._capture_turn_start_snapshot()


def test_counterfactual_snapshot_restores_state_and_rng(isolated_game_state):
    _init_live_game(isolated_game_state)
    env = isolated_game_state.env

    backend_state.capture_counterfactual_snapshot()

    actions = np.full(env.n_players, ActionType.NOOP.value, dtype=int)
    actions[env.ball_holder] = ActionType.SHOOT.value

    obs1, rewards1, done1, truncated1, info1 = env.step(actions)
    first_result = {
        "positions": copy.deepcopy(env.positions),
        "ball_holder": env.ball_holder,
        "shot_clock": env.shot_clock,
        "episode_ended": env.episode_ended,
        "last_action_results": _encode(env.last_action_results),
        "obs": _encode(obs1),
        "rewards": list(rewards1) if isinstance(rewards1, (list, tuple, np.ndarray)) else [rewards1],
        "done": done1,
        "truncated": truncated1,
        "info": _encode(info1),
    }

    env.positions[0] = (99, 99)
    env.ball_holder = 2
    env.shot_clock = 1
    env.intent_index = 7
    env.intent_age = 4
    env.offense_layup_pct_by_player[0] = 0.1
    isolated_game_state.reward_history.append({"step": 2, "offense": 9.9, "defense": -9.9})
    isolated_game_state.playable_session["score"]["user"] = 99

    backend_state.restore_counterfactual_snapshot()
    env2 = isolated_game_state.env
    obs2, rewards2, done2, truncated2, info2 = env2.step(actions)

    second_result = {
        "positions": copy.deepcopy(env2.positions),
        "ball_holder": env2.ball_holder,
        "shot_clock": env2.shot_clock,
        "episode_ended": env2.episode_ended,
        "last_action_results": _encode(env2.last_action_results),
        "obs": _encode(obs2),
        "rewards": list(rewards2) if isinstance(rewards2, (list, tuple, np.ndarray)) else [rewards2],
        "done": done2,
        "truncated": truncated2,
        "info": _encode(info2),
    }

    assert second_result == first_result
    assert isolated_game_state.playable_session["score"]["user"] == 1
    assert isolated_game_state.reward_history == [
        {"step": 1, "offense": 0.2, "defense": -0.1, "shot_clock": 12}
    ]


def test_counterfactual_snapshot_routes_capture_and_restore(isolated_game_state):
    _init_live_game(isolated_game_state)
    baseline_positions = [tuple(pos) for pos in isolated_game_state.env.positions]
    baseline_shot_clock = int(isolated_game_state.env.shot_clock)
    baseline_ball_holder = int(isolated_game_state.env.ball_holder)
    baseline_intent_index = int(isolated_game_state.env.intent_index)
    baseline_skill = float(isolated_game_state.env.offense_layup_pct_by_player[0])

    capture_body = admin_routes.capture_counterfactual_snapshot_route()
    assert capture_body["snapshot"]["available"] is True
    assert capture_body["state"]["counterfactual_snapshot_available"] is True

    isolated_game_state.env.positions[0] = (4, 4)
    isolated_game_state.env.ball_holder = 1
    isolated_game_state.env.shot_clock = 3
    isolated_game_state.env.intent_index = 0
    isolated_game_state.env.offense_layup_pct_by_player[0] = 0.15
    backend_state._rebuild_cached_obs()

    body = admin_routes.restore_counterfactual_snapshot_route()

    assert body["status"] == "success"
    assert body["state"]["counterfactual_snapshot_available"] is True
    assert [tuple(pos) for pos in body["state"]["positions"]] == baseline_positions
    assert int(body["state"]["shot_clock"]) == baseline_shot_clock
    assert int(body["state"]["ball_holder"]) == baseline_ball_holder
    assert int(body["state"]["intent_index_current"]) == baseline_intent_index
    assert float(isolated_game_state.env.offense_layup_pct_by_player[0]) == baseline_skill


def test_restore_counterfactual_snapshot_requires_existing_snapshot(isolated_game_state):
    _init_live_game(isolated_game_state)
    with pytest.raises(HTTPException) as exc:
        admin_routes.restore_counterfactual_snapshot_route()
    assert exc.value.status_code == 400
    assert "No counterfactual snapshot available" in str(exc.value.detail)


def test_replay_counterfactual_snapshot_route_replays_deterministically(isolated_game_state):
    _init_live_game(isolated_game_state)
    backend_state.capture_counterfactual_snapshot()

    first = admin_routes.replay_counterfactual_snapshot_route(
        ReplayCounterfactualRequest(
            player_deterministic=True,
            opponent_deterministic=True,
            max_steps=32,
        )
    )

    assert first["status"] == "success"
    assert first["terminated"] is True
    assert len(first["states"]) >= 2
    assert first["steps_taken"] == len(first["states"]) - 1
    assert first["state"] == first["states"][-1]

    backend_state.restore_counterfactual_snapshot()

    second = admin_routes.replay_counterfactual_snapshot_route(
        ReplayCounterfactualRequest(
            player_deterministic=True,
            opponent_deterministic=True,
            max_steps=32,
        )
    )

    assert _encode(second["states"]) == _encode(first["states"])


def test_init_game_clears_counterfactual_snapshot(monkeypatch, isolated_game_state):
    isolated_game_state.counterfactual_snapshot = {"available": True}

    class DummyPPO:
        def __init__(self):
            self.policy = type("DummyPolicy", (), {"pass_logit_bias": 0.0})()

    monkeypatch.setattr(mlflow_config, "setup_mlflow", lambda *args, **kwargs: None)
    monkeypatch.setattr(lifecycle_routes, "get_mlflow_params", lambda client, run_id: ({"players": 3}, {}))
    monkeypatch.setattr(lifecycle_routes, "get_mlflow_phi_shaping_params", lambda client, run_id: {})
    monkeypatch.setattr(lifecycle_routes, "get_mlflow_training_params", lambda client, run_id: {})
    monkeypatch.setattr(lifecycle_routes, "get_unified_policy_path", lambda client, run_id, unified_name=None: "/tmp/fake.zip")
    monkeypatch.setattr(lifecycle_routes.PPO, "load", lambda *args, **kwargs: DummyPPO())
    monkeypatch.setattr(lifecycle_routes, "validate_policy_observation_schema", lambda policy, env, obs, **kwargs: obs)
    monkeypatch.setattr(lifecycle_routes, "_compute_param_counts_from_policy", lambda policy: None)
    monkeypatch.setattr(lifecycle_routes.mlflow.tracking, "MlflowClient", lambda: object())

    body = asyncio.run(
        lifecycle_routes.init_game(
            InitGameRequest(
                run_id="dummy-run",
                user_team_name="OFFENSE",
            )
        )
    )

    assert body["status"] == "success"
    assert isolated_game_state.counterfactual_snapshot is None
    assert body["state"]["counterfactual_snapshot_available"] is False
