import numpy as np
import torch

import app.backend.state as backend_state
from basketworld.envs.basketworld_env_v2 import HexagonBasketballEnv, Team


class _DummySelectorPolicy:
    def __init__(self, logits: list[float], value: float = 0.25):
        self.pass_logit_bias = 0.0
        self._logits = torch.tensor([logits], dtype=torch.float32)
        self._value = torch.tensor([value], dtype=torch.float32)

    def has_intent_selector(self) -> bool:
        return True

    def has_intent_selector_value_head(self) -> bool:
        return True

    def get_intent_selector_outputs(self, obs):
        return self._logits, self._value


class _DummyModel:
    def __init__(self, logits: list[float], *, num_timesteps: int = 10):
        self.policy = _DummySelectorPolicy(logits)
        self.num_timesteps = int(num_timesteps)


def test_get_full_game_state_includes_selector_intent_preferences():
    original_state = backend_state.game_state.__dict__.copy()
    try:
        env = HexagonBasketballEnv(
            players=3,
            render_mode=None,
            enable_intent_learning=True,
            intent_null_prob=0.0,
            training_team=Team.OFFENSE,
        )
        obs, _ = env.reset(seed=123)
        env.intent_active = True
        env.intent_index = 2
        backend_state.game_state.env = env
        backend_state.game_state.obs = obs
        backend_state.game_state.user_team = Team.OFFENSE
        backend_state.game_state.unified_policy = _DummyModel(
            [0.1, 0.2, 3.0, 1.5, -0.3, 0.0, 0.5, 0.4]
        )
        backend_state.game_state.defense_policy = None
        backend_state.game_state.reward_history = []
        backend_state.game_state.episode_rewards = {"offense": 0.0, "defense": 0.0}
        backend_state.game_state.shot_log = []
        backend_state.game_state.phi_log = []
        backend_state.game_state.actions_log = []
        backend_state.game_state.episode_states = []
        backend_state.game_state.mlflow_training_params = {
            "intent_selector_enabled": True,
            "intent_selector_mode": "integrated",
            "intent_selector_alpha_start": 1.0,
            "intent_selector_alpha_end": 1.0,
            "intent_selector_alpha_warmup_steps": 0,
            "intent_selector_alpha_ramp_steps": 1,
        }

        state = backend_state.get_full_game_state(include_policy_probs=True)

        prefs = state.get("selector_intent_preferences")
        assert prefs is not None
        assert np.isclose(float(prefs["alpha_current"]), 1.0)
        assert np.isclose(float(prefs["value_estimate"]), 0.25)
        assert int(prefs["current_intent_index"]) == 2
        ranked = prefs["intent_probs"]
        assert [int(item["intent_index"]) for item in ranked[:4]] == [2, 3, 6, 7]
        probs = [float(item["prob"]) for item in ranked]
        assert probs == sorted(probs, reverse=True)
    finally:
        backend_state.game_state.__dict__.clear()
        backend_state.game_state.__dict__.update(original_state)
