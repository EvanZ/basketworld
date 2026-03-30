import numpy as np
import torch

import app.backend.evaluation as backend_evaluation
from basketworld.envs.basketworld_env_v2 import ActionType, HexagonBasketballEnv, Team


class _DummySelectorPolicy:
    def __init__(self, chosen_intent: int, num_intents: int):
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


def test_sequential_evaluation_initializes_selector_runtime(monkeypatch):
    env = HexagonBasketballEnv(
        players=3,
        allow_dunks=True,
        enable_intent_learning=True,
        intent_null_prob=0.0,
        intent_commitment_steps=4,
        training_team=Team.OFFENSE,
    )
    env.reset(seed=123)
    chosen_intent = 5
    model = _DummyModel(chosen_intent, env.num_intents, num_timesteps=10)
    captured_initial_intents: list[int] = []

    def _fake_accumulate_intent_selection(eval_diagnostics, env_obj):
        captured_initial_intents.append(int(getattr(env_obj, "intent_index", -1)))

    def _fake_predict_joint_actions(**kwargs):
        env_obj = kwargs["env"]
        probs = [
            np.full(len(ActionType), 1.0 / float(len(ActionType)), dtype=np.float32)
            for _ in range(env_obj.n_players)
        ]
        zeros = np.zeros(env_obj.n_players, dtype=int)
        return {
            "resolved_unified": zeros.copy(),
            "unified_probs": probs,
            "resolved_opponent": zeros.copy(),
            "opponent_probs": probs,
        }

    def _fake_step(actions):
        env.shot_clock = max(0, int(env.shot_clock) - 1)
        env.last_action_results = {}
        obs = env._build_observation_dict(True)
        info = {
            "action_results": {},
            "phi_ep_by_player": np.zeros(env.n_players, dtype=np.float32),
        }
        return obs, np.zeros(env.n_players, dtype=np.float32), True, False, info

    monkeypatch.setattr(
        backend_evaluation,
        "_accumulate_intent_selection",
        _fake_accumulate_intent_selection,
    )
    monkeypatch.setattr(
        backend_evaluation,
        "predict_joint_policy_actions",
        _fake_predict_joint_actions,
    )
    monkeypatch.setattr(env, "step", _fake_step)

    backend_evaluation._run_sequential_evaluation(
        1,
        True,
        True,
        env,
        {
            "intent_selector_enabled": True,
            "intent_selector_mode": "integrated",
            "intent_selector_alpha_start": 1.0,
            "intent_selector_alpha_end": 1.0,
            "intent_selector_alpha_warmup_steps": 0,
            "intent_selector_alpha_ramp_steps": 1,
            "intent_selector_multiselect_enabled": True,
            "intent_selector_min_play_steps": 4,
        },
        model,
        None,
        Team.OFFENSE,
        1.0,
        -1.0,
    )

    assert captured_initial_intents == [chosen_intent]


def test_sequential_evaluation_best_intent_mode_bypasses_alpha_mixing(monkeypatch):
    env = HexagonBasketballEnv(
        players=3,
        allow_dunks=True,
        enable_intent_learning=True,
        intent_null_prob=0.0,
        intent_commitment_steps=4,
        training_team=Team.OFFENSE,
    )
    env.reset(seed=321)
    chosen_intent = 4
    model = _DummyModel(chosen_intent, env.num_intents, num_timesteps=0)
    captured_initial_intents: list[int] = []

    def _fake_accumulate_intent_selection(eval_diagnostics, env_obj):
        captured_initial_intents.append(int(getattr(env_obj, "intent_index", -1)))

    def _fake_predict_joint_actions(**kwargs):
        env_obj = kwargs["env"]
        probs = [
            np.full(len(ActionType), 1.0 / float(len(ActionType)), dtype=np.float32)
            for _ in range(env_obj.n_players)
        ]
        zeros = np.zeros(env_obj.n_players, dtype=int)
        return {
            "resolved_unified": zeros.copy(),
            "unified_probs": probs,
            "resolved_opponent": zeros.copy(),
            "opponent_probs": probs,
        }

    def _fake_step(actions):
        env.shot_clock = max(0, int(env.shot_clock) - 1)
        env.last_action_results = {}
        obs = env._build_observation_dict(True)
        info = {
            "action_results": {},
            "phi_ep_by_player": np.zeros(env.n_players, dtype=np.float32),
        }
        return obs, np.zeros(env.n_players, dtype=np.float32), True, False, info

    monkeypatch.setattr(
        backend_evaluation,
        "_accumulate_intent_selection",
        _fake_accumulate_intent_selection,
    )
    monkeypatch.setattr(
        backend_evaluation,
        "predict_joint_policy_actions",
        _fake_predict_joint_actions,
    )
    monkeypatch.setattr(env, "step", _fake_step)

    backend_evaluation._run_sequential_evaluation(
        1,
        True,
        True,
        env,
        {
            "intent_selector_enabled": True,
            "intent_selector_mode": "integrated",
            "intent_selector_alpha_start": 0.0,
            "intent_selector_alpha_end": 0.0,
            "intent_selector_alpha_warmup_steps": 0,
            "intent_selector_alpha_ramp_steps": 1,
            "intent_selector_multiselect_enabled": True,
            "intent_selector_min_play_steps": 4,
        },
        model,
        None,
        Team.OFFENSE,
        1.0,
        -1.0,
        intent_selection_mode="best_intent",
    )

    assert captured_initial_intents == [chosen_intent]
