import copy
import asyncio
import queue
from concurrent.futures import Future

import numpy as np
import pytest
from fastapi.encoders import jsonable_encoder
from fastapi import HTTPException

import app.backend.observations as backend_observations
import app.backend.state as backend_state
import basketworld.utils.mlflow_config as mlflow_config
from app.backend.routes import admin_routes
from app.backend.routes import lifecycle_routes
from app.backend.schemas import InitGameRequest, PlaybookAnalysisRequest, ReplayCounterfactualRequest
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


def test_playbook_analysis_route_aggregates_and_restores_live_state(isolated_game_state):
    _init_live_game(isolated_game_state)
    backend_state.capture_counterfactual_snapshot()

    baseline_positions = [tuple(pos) for pos in isolated_game_state.env.positions]
    baseline_ball_holder = int(isolated_game_state.env.ball_holder)
    baseline_intent_index = int(isolated_game_state.env.intent_index)
    baseline_reward_history = copy.deepcopy(isolated_game_state.reward_history)
    baseline_snapshot = copy.deepcopy(isolated_game_state.counterfactual_snapshot)

    body = admin_routes.playbook_analysis_route(
        PlaybookAnalysisRequest(
            intent_indices=[0, 2],
            num_rollouts=2,
            max_steps=3,
            use_snapshot=True,
            player_deterministic=False,
            opponent_deterministic=True,
        )
    )

    assert body["status"] == "success"
    assert body["source"] == "snapshot"
    assert body["offense_ids"] == list(isolated_game_state.env.offense_ids)
    assert len(body["panels"]) == 2
    assert {panel["intent_index"] for panel in body["panels"]} == {0, 2}
    for panel in body["panels"]:
        assert panel["ball_heatmap"]
        assert isinstance(panel["player_heatmaps"], dict)
        assert isinstance(panel["player_shot_heatmaps"], dict)
        assert isinstance(panel["shot_stats"], dict)
        assert isinstance(panel["shot_stats"]["by_player"], dict)
        assert isinstance(panel["shot_stats"]["total"], dict)
        assert isinstance(panel["terminal_outcomes"], dict)
        assert isinstance(panel["turnover_reasons"], dict)
        assert panel["num_rollouts"] == 2
        assert panel["base_state"]
        assert isinstance(panel["player_path_segments"], dict)
        assert isinstance(panel["ball_path_segments"], list)
        assert isinstance(panel["pass_path_segments"], list)
        assert all(isinstance(segments, list) for segments in panel["player_path_segments"].values())
        for segment in panel["pass_path_segments"]:
            assert "passer_id" in segment
            assert "receiver_id" in segment

    assert [tuple(pos) for pos in isolated_game_state.env.positions] == baseline_positions
    assert int(isolated_game_state.env.ball_holder) == baseline_ball_holder
    assert int(isolated_game_state.env.intent_index) == baseline_intent_index
    assert isolated_game_state.reward_history == baseline_reward_history


def test_playbook_analysis_run_to_end_ignores_max_steps(isolated_game_state, monkeypatch):
    _init_live_game(isolated_game_state)
    backend_state.capture_counterfactual_snapshot()

    steps = {"count": 0}

    def fake_step(_req):
        steps["count"] += 1
        isolated_game_state.env.shot_clock -= 1
        done = steps["count"] >= 3
        isolated_game_state.env.episode_ended = done
        return {
            "state": {
                "done": done,
                "last_action_results": {},
            }
        }

    monkeypatch.setattr(lifecycle_routes, "step", fake_step)

    body = admin_routes.playbook_analysis_route(
        PlaybookAnalysisRequest(
            intent_indices=[0],
            num_rollouts=1,
            max_steps=1,
            run_to_end=True,
            use_snapshot=True,
            player_deterministic=False,
            opponent_deterministic=True,
        )
    )

    assert body["status"] == "success"
    assert body["run_to_end"] is True
    assert steps["count"] == 3
    assert body["panels"][0]["avg_steps"] == pytest.approx(3.0)
    assert isolated_game_state.counterfactual_snapshot is not None
    progress = admin_routes.playbook_progress()
    assert progress["status"] == "completed"
    assert progress["completed"] == 1
    assert progress["total"] == 1


def test_playbook_analysis_parallel_path_passes_training_params_and_merges_payload(
    isolated_game_state,
    monkeypatch,
):
    _init_live_game(isolated_game_state)
    backend_state.capture_counterfactual_snapshot()
    isolated_game_state.env_required_params = {
        "players": 3,
        "allow_dunks": True,
        "enable_intent_learning": True,
    }
    isolated_game_state.env_optional_params = {
        "intent_null_prob": 0.0,
        "enable_defense_intent_learning": True,
        "defense_intent_null_prob": 0.0,
        "intent_commitment_steps": 4,
    }
    isolated_game_state.mlflow_training_params = {
        "intent_selector_multiselect_enabled": True,
        "intent_selector_min_play_steps": 2,
    }
    isolated_game_state.unified_policy_path = "/tmp/unified_iter_1.zip"
    isolated_game_state.opponent_policy_path = "/tmp/opponent_iter_1.zip"
    isolated_game_state.role_flag_offense = 1.0
    isolated_game_state.role_flag_defense = -1.0

    captured: dict[str, object] = {}

    class _FakeContext:
        @staticmethod
        def Queue():
            return queue.Queue()

    class _FakeExecutor:
        def __init__(self, *, max_workers, mp_context, initializer, initargs):
            captured["max_workers"] = int(max_workers)
            captured["initargs"] = initargs
            initializer(*initargs)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, batch):
            future = Future()
            future.set_result(fn(batch))
            return future

    def fake_run_playbook_batch_worker(batch):
        (
            batch_specs,
            _parallel_base_env,
            _base_ui_state,
            _user_team_name,
            offense_ids,
            _commitment_steps,
            _max_steps,
            _run_to_end,
            _player_deterministic,
            _opponent_deterministic,
        ) = batch
        payload = {}
        for _, intent_index, _ in batch_specs:
            panel = payload.setdefault(int(intent_index), admin_routes._init_playbook_panel_accumulator(offense_ids))
            panel["num_rollouts"] += 1
            panel["base_state"] = panel["base_state"] or {"seeded": True}
        return payload

    monkeypatch.setattr(admin_routes.mp, "get_context", lambda _mode: _FakeContext())
    monkeypatch.setattr(admin_routes, "ProcessPoolExecutor", _FakeExecutor)
    monkeypatch.setattr(admin_routes, "_run_playbook_batch_worker", fake_run_playbook_batch_worker)
    monkeypatch.setattr(
        admin_routes.backend_evaluation,
        "_init_evaluation_worker",
        lambda *args, **kwargs: captured.setdefault("initializer_called", True),
    )

    body = admin_routes.playbook_analysis_route(
        PlaybookAnalysisRequest(
            intent_indices=[0, 2],
            num_rollouts=4,
            max_steps=3,
            use_snapshot=True,
            player_deterministic=False,
            opponent_deterministic=True,
        )
    )

    assert body["status"] == "success"
    assert captured["initargs"][2]["intent_selector_multiselect_enabled"] is False
    assert isolated_game_state.mlflow_training_params["intent_selector_multiselect_enabled"] is True
    assert captured["initargs"][3] == isolated_game_state.unified_policy_path
    assert captured["initargs"][4] == isolated_game_state.opponent_policy_path
    panels = {panel["intent_index"]: panel for panel in body["panels"]}
    assert panels[0]["num_rollouts"] == 4
    assert panels[2]["num_rollouts"] == 4
    assert panels[0]["base_state"] == {"seeded": True}
    assert panels[2]["base_state"] == {"seeded": True}
    assert body["play_name_map"]


def test_init_game_clears_counterfactual_snapshot(monkeypatch, isolated_game_state):
    isolated_game_state.counterfactual_snapshot = {"available": True}

    class DummyPPO:
        def __init__(self):
            self.policy = type("DummyPolicy", (), {"pass_logit_bias": 0.0})()

    monkeypatch.setattr(mlflow_config, "setup_mlflow", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        lifecycle_routes,
        "get_mlflow_params",
        lambda client, run_id: (
            {"players": 3},
            {
                "use_set_obs": True,
                "mirror_episode_prob": 0.5,
                "deterministic_opponent": True,
                "per_env_opponent_sampling": True,
                "opponent_pool_size": 16,
            },
        ),
    )
    monkeypatch.setattr(lifecycle_routes, "get_mlflow_phi_shaping_params", lambda client, run_id: {})
    monkeypatch.setattr(lifecycle_routes, "get_mlflow_training_params", lambda client, run_id: {})
    monkeypatch.setattr(lifecycle_routes, "get_unified_policy_path", lambda client, run_id, unified_name=None: "/tmp/fake.zip")
    monkeypatch.setattr(lifecycle_routes, "load_ppo_for_inference", lambda *args, **kwargs: DummyPPO())
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
