import copy

import numpy as np
import pytest
import torch

import app.backend.observations as backend_observations
import app.backend.state as backend_state
from app.backend.routes import lifecycle_routes
from app.backend.schemas import ActionRequest
from basketworld.envs.basketworld_env_v2 import ActionType, HexagonBasketballEnv, Team


@pytest.fixture
def isolated_game_state(monkeypatch):
    fresh = backend_state.GameState()
    monkeypatch.setattr(backend_state, "game_state", fresh)
    monkeypatch.setattr(lifecycle_routes, "game_state", fresh)
    monkeypatch.setattr(backend_observations, "game_state", fresh)
    return fresh


class _DummySelectorPolicy:
    def __init__(self, chosen_intent: int, num_intents: int):
        self.pass_logit_bias = 0.0
        self._chosen_intent = int(chosen_intent)
        self._num_intents = int(num_intents)

    def has_intent_selector(self) -> bool:
        return True

    def has_intent_selector_value_head(self) -> bool:
        return True

    def get_intent_selector_outputs(self, obs):
        logits = torch.full((1, self._num_intents), -1000.0, dtype=torch.float32)
        logits[0, self._chosen_intent] = 1000.0
        values = torch.tensor([0.25], dtype=torch.float32)
        return logits, values


class _DummyModel:
    def __init__(self, chosen_intent: int, num_intents: int, *, num_timesteps: int = 0):
        self.policy = _DummySelectorPolicy(chosen_intent, num_intents)
        self.num_timesteps = int(num_timesteps)


def _build_live_env(*, intent_commitment_steps: int) -> HexagonBasketballEnv:
    env = HexagonBasketballEnv(
        players=3,
        allow_dunks=True,
        enable_intent_learning=True,
        intent_null_prob=0.0,
        intent_commitment_steps=int(intent_commitment_steps),
        training_team=Team.OFFENSE,
    )
    env.reset(seed=123)
    return env


def _init_selector_state(
    state: backend_state.GameState,
    monkeypatch,
    *,
    chosen_intent: int,
    intent_commitment_steps: int,
    min_play_steps: int,
) -> HexagonBasketballEnv:
    env = _build_live_env(intent_commitment_steps=intent_commitment_steps)
    state.env = env
    state.user_team = Team.OFFENSE
    state.unified_policy = _DummyModel(chosen_intent, env.num_intents, num_timesteps=10)
    state.defense_policy = None
    state.obs = env._build_observation_dict(True)
    state.prev_obs = None
    state.reward_history = []
    state.episode_rewards = {"offense": 0.0, "defense": 0.0}
    state.frames = []
    state.actions_log = []
    state.shot_log = []
    state.phi_log = []
    state.episode_states = []
    state.playable_session = None
    state.sampled_offense_skills = {
        "layup": list(env.offense_layup_pct_by_player),
        "three_pt": list(env.offense_three_pt_pct_by_player),
        "dunk": list(env.offense_dunk_pct_by_player),
    }
    state.replay_offense_skills = copy.deepcopy(state.sampled_offense_skills)
    state.mlflow_training_params = {
        "intent_selector_enabled": True,
        "intent_selector_mode": "integrated",
        "intent_selector_alpha_start": 1.0,
        "intent_selector_alpha_end": 1.0,
        "intent_selector_alpha_warmup_steps": 0,
        "intent_selector_alpha_ramp_steps": 1,
        "intent_selector_multiselect_enabled": True,
        "intent_selector_min_play_steps": int(min_play_steps),
    }
    backend_state._rebuild_cached_obs()

    def _fake_predict(policy, ai_obs, env_obj, deterministic, strategy):
        probs = [
            np.full(len(ActionType), 1.0 / float(len(ActionType)), dtype=np.float32)
            for _ in range(env_obj.n_players)
        ]
        return np.zeros(env_obj.n_players, dtype=int), probs

    monkeypatch.setattr(lifecycle_routes, "_predict_policy_actions", _fake_predict)
    return env


def test_step_route_reselects_on_commitment_timeout(isolated_game_state, monkeypatch):
    env = _init_selector_state(
        isolated_game_state,
        monkeypatch,
        chosen_intent=5,
        intent_commitment_steps=4,
        min_play_steps=4,
    )
    env.intent_active = True
    env.intent_index = 1
    env.intent_age = 3
    env.intent_commitment_remaining = 1

    def _fake_step(actions):
        env.shot_clock = max(0, int(env.shot_clock) - 1)
        env.intent_active = True
        env.intent_age = 4
        env.intent_commitment_remaining = 0
        env.last_action_results = {}
        obs = env._build_observation_dict(True)
        info = {
            "action_results": {},
            "phi_ep_by_player": np.zeros(env.n_players, dtype=np.float32),
        }
        return obs, np.zeros(env.n_players, dtype=np.float32), False, False, info

    monkeypatch.setattr(env, "step", _fake_step)

    body = lifecycle_routes.step(
        ActionRequest(actions={}, player_deterministic=True, opponent_deterministic=True)
    )

    assert body["status"] == "success"
    assert int(env.intent_index) == 5
    assert bool(env.intent_active) is True
    assert int(env.intent_age) == 0
    assert int(env.intent_commitment_remaining) == 4
    assert body["state"]["intent_index_current"] == 5
    assert body["state"]["intent_active_current"] is True
    assert body["state"]["selector_segment_index_current"] == 1
    assert body["state"]["selector_last_boundary_reason"] == "commitment_timeout"


def test_step_route_reselects_on_completed_pass_after_min_steps(
    isolated_game_state, monkeypatch
):
    env = _init_selector_state(
        isolated_game_state,
        monkeypatch,
        chosen_intent=6,
        intent_commitment_steps=8,
        min_play_steps=4,
    )
    env.intent_active = True
    env.intent_index = 2
    env.intent_age = 3
    env.intent_commitment_remaining = 5
    passer_id = int(env.offense_ids[0])
    receiver_id = int(env.offense_ids[1])

    def _fake_step(actions):
        env.shot_clock = max(0, int(env.shot_clock) - 1)
        env.ball_holder = receiver_id
        env.intent_active = True
        env.intent_age = 4
        env.intent_commitment_remaining = 4
        env.last_action_results = {
            "passes": {
                str(passer_id): {
                    "success": True,
                    "target": receiver_id,
                }
            }
        }
        obs = env._build_observation_dict(True)
        info = {
            "action_results": copy.deepcopy(env.last_action_results),
            "phi_ep_by_player": np.zeros(env.n_players, dtype=np.float32),
        }
        return obs, np.zeros(env.n_players, dtype=np.float32), False, False, info

    monkeypatch.setattr(env, "step", _fake_step)

    body = lifecycle_routes.step(
        ActionRequest(actions={}, player_deterministic=True, opponent_deterministic=True)
    )

    assert body["status"] == "success"
    assert int(env.intent_index) == 6
    assert bool(env.intent_active) is True
    assert int(env.intent_age) == 0
    assert int(env.intent_commitment_remaining) == 8
    assert body["state"]["intent_index_current"] == 6
    assert body["state"]["intent_active_current"] is True
    assert body["state"]["selector_segment_index_current"] == 1
    assert body["state"]["selector_last_boundary_reason"] == "completed_pass"
