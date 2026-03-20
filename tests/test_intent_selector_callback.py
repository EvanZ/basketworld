import numpy as np
import torch

import basketworld.utils.callbacks as callbacks_module
from basketworld.utils.callbacks import IntentSelectorCallback


class _DummySelectorVecEnv:
    def __init__(self):
        self.calls = []

    def env_method(self, method_name, *args, indices=None, **kwargs):
        self.calls.append((method_name, args, indices, kwargs))
        return [None]


class _DummySelectorPolicy:
    def __init__(self, num_intents: int = 4):
        self.device = torch.device("cpu")
        self.num_intents = int(num_intents)
        self.bias = torch.nn.Parameter(
            torch.tensor([-4.0, -2.0, 4.0, -3.0], dtype=torch.float32)
        )
        self.optimizer = torch.optim.Adam([self.bias], lr=0.05)

    def has_intent_selector(self) -> bool:
        return True

    def obs_to_tensor(self, obs):
        tensor_obs = {
            key: torch.as_tensor(value, dtype=torch.float32)
            for key, value in obs.items()
        }
        return tensor_obs, None

    def get_intent_selector_logits(self, obs):
        obs_arr = obs["obs"]
        batch = 1 if len(obs_arr.shape) == 1 else int(obs_arr.shape[0])
        return self.bias.unsqueeze(0).expand(batch, -1)


class _DummySelectorModel:
    def __init__(self, n_envs: int = 2):
        self.policy = _DummySelectorPolicy()
        self._env = _DummySelectorVecEnv()
        self.num_timesteps = 100
        self._last_obs = {
            "obs": np.zeros((n_envs, 4), dtype=np.float32),
            "role_flag": np.array([[1.0], [-1.0]], dtype=np.float32)
            if n_envs == 2
            else np.array([[1.0]], dtype=np.float32),
            "intent_active": np.ones((n_envs, 1), dtype=np.float32),
            "intent_visible": np.ones((n_envs, 1), dtype=np.float32),
            "intent_index": np.zeros((n_envs, 1), dtype=np.float32),
            "intent_age_norm": np.zeros((n_envs, 1), dtype=np.float32),
        }
        self._last_episode_starts = np.ones(n_envs, dtype=bool)

    def get_env(self):
        return self._env


def test_intent_selector_callback_overrides_offense_episode_starts_only():
    model = _DummySelectorModel(n_envs=2)
    callback = IntentSelectorCallback(
        enabled=True,
        num_intents=4,
        alpha_start=1.0,
        alpha_end=1.0,
        warmup_steps=0,
        ramp_steps=0,
    )
    callback.init_callback(model)
    callback._on_training_start()

    callback._on_rollout_start()

    assert len(model.get_env().calls) == 1
    method_name, args, indices, kwargs = model.get_env().calls[0]
    assert method_name == "set_offense_intent_state"
    assert args == (2,)
    assert indices == [0]
    assert kwargs["intent_active"] is True
    assert kwargs["intent_age"] == 0
    assert float(model._last_obs["intent_index"][0, 0]) == 2.0
    assert float(model._last_obs["intent_active"][0, 0]) == 1.0
    assert 1 not in callback._episode_start_records_by_env
    assert callback._episode_start_records_by_env[0]["chosen_z"] == 2


def test_intent_selector_callback_trains_on_possession_return_and_logs_metrics(monkeypatch):
    model = _DummySelectorModel(n_envs=1)
    callback = IntentSelectorCallback(
        enabled=True,
        num_intents=4,
        alpha_start=1.0,
        alpha_end=1.0,
        warmup_steps=0,
        ramp_steps=0,
        entropy_coef=0.0,
        usage_reg_coef=0.0,
    )
    callback.init_callback(model)
    callback._on_training_start()

    class _FixedStats:
        mean = 0.0
        std = 1.0

        @staticmethod
        def update(_):
            return None

    callback._return_stats = _FixedStats()
    bias_before = model.policy.bias.detach().clone()

    callback._on_rollout_start()
    callback.locals = {
        "infos": [{"episode": {"r": 2.0}}],
        "dones": np.array([True], dtype=bool),
        "new_obs": {
            "obs": np.zeros((1, 4), dtype=np.float32),
            "role_flag": np.array([[1.0]], dtype=np.float32),
            "intent_active": np.ones((1, 1), dtype=np.float32),
            "intent_visible": np.ones((1, 1), dtype=np.float32),
            "intent_index": np.zeros((1, 1), dtype=np.float32),
            "intent_age_norm": np.zeros((1, 1), dtype=np.float32),
        },
    }
    assert callback._on_step() is True

    logged = {}
    monkeypatch.setattr(
        callbacks_module.mlflow,
        "log_metric",
        lambda name, value, step=None: logged.setdefault(name, (float(value), step)),
    )

    callback._on_rollout_end()

    bias_after = model.policy.bias.detach().clone()
    assert bias_after[2] > bias_before[2]
    assert "intent/selector_loss" in logged
    assert "intent/selector_alpha_current" in logged
    assert "intent/selector_return_mean" in logged
    assert "intent/selector_samples" in logged
    assert "intent/selector_usage_by_intent/2" in logged
    assert "intent/selector_prob_mean_by_intent/2" in logged
    assert "intent/selector_top1_by_intent/2" in logged
    assert "intent/selector_return_by_intent/2" in logged
    assert "intent/selector_confidence_mean" in logged
    assert "intent/selector_margin_mean" in logged
    assert np.isclose(logged["intent/selector_alpha_current"][0], 1.0)
    assert np.isclose(logged["intent/selector_return_mean"][0], 2.0)
    assert np.isclose(logged["intent/selector_samples"][0], 1.0)
    assert np.isclose(logged["intent/selector_usage_by_intent/2"][0], 1.0)
    assert np.isclose(logged["intent/selector_top1_by_intent/2"][0], 1.0)
    assert np.isclose(logged["intent/selector_return_by_intent/2"][0], 2.0)
    assert logged["intent/selector_prob_mean_by_intent/2"][0] > 0.99
    assert logged["intent/selector_confidence_mean"][0] > 0.99
    assert logged["intent/selector_margin_mean"][0] > 0.99
