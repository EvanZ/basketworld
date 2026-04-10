import numpy as np
import torch

from basketworld.utils.callbacks import IntentPolicySensitivityCallback
from basketworld.utils.intent_policy_sensitivity import (
    build_intent_variant_batch,
    compute_policy_sensitivity_metrics,
)


class _DummyDistribution:
    def __init__(self, probs: torch.Tensor):
        self._probs = probs

    def action_probabilities(self) -> torch.Tensor:
        return self._probs


class _DummySensitivityPolicy:
    def __init__(self, num_intents: int = 4, n_players: int = 2, n_actions: int = 4):
        self.device = torch.device("cpu")
        self.num_intents = int(num_intents)
        self.n_players = int(n_players)
        self.n_actions = int(n_actions)

    def obs_to_tensor(self, obs):
        tensor_obs = {
            key: torch.as_tensor(value, dtype=torch.float32)
            for key, value in obs.items()
        }
        return tensor_obs, None

    def get_distribution(self, obs_tensor):
        globals_vec = obs_tensor["globals"]
        idx_norm = globals_vec[:, -4]
        if self.num_intents > 1:
            intent_idx = torch.round(idx_norm * float(self.num_intents - 1)).long()
        else:
            intent_idx = torch.zeros_like(idx_norm, dtype=torch.long)

        logits = torch.full(
            (globals_vec.shape[0], self.n_players, self.n_actions),
            fill_value=-2.0,
            dtype=torch.float32,
        )
        for batch_idx in range(logits.shape[0]):
            preferred = int(intent_idx[batch_idx].item()) % self.n_actions
            secondary = (preferred + 1) % self.n_actions
            logits[batch_idx, :, secondary] = 0.5
            logits[batch_idx, :, preferred] = 2.0
        probs = torch.softmax(logits, dim=-1)
        return _DummyDistribution(probs)


class _DummyModel:
    def __init__(self, num_intents: int = 4):
        self.policy = _DummySensitivityPolicy(num_intents=num_intents)
        self.num_timesteps = 123


def _single_obs():
    return {
        "players": np.zeros((6, 15), dtype=np.float32),
        "globals": np.array([24.0, 0.0, 0.1, 0.2, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "action_mask": np.ones((2, 4), dtype=np.float32),
        "role_flag": np.array([1.0], dtype=np.float32),
        "intent_index": np.array([0.0], dtype=np.float32),
        "intent_active": np.array([0.0], dtype=np.float32),
        "intent_visible": np.array([0.0], dtype=np.float32),
        "intent_age_norm": np.array([0.0], dtype=np.float32),
    }


def test_build_intent_variant_batch_patches_scalar_and_set_fields():
    batch, intents = build_intent_variant_batch(_single_obs(), num_intents=4)
    assert intents == [0, 1, 2, 3]
    assert batch["globals"].shape == (4, 8)
    assert np.isclose(batch["globals"][2, -4], 2.0 / 3.0)
    assert np.isclose(batch["globals"][3, -3], 1.0)
    assert np.isclose(batch["globals"][3, -2], 1.0)
    assert np.isclose(batch["globals"][3, -1], 0.0)
    assert np.isclose(batch["intent_index"][3, 0], 3.0)
    assert np.isclose(batch["intent_active"][1, 0], 1.0)
    assert np.isclose(batch["intent_visible"][1, 0], 1.0)
    assert np.isclose(batch["intent_age_norm"][1, 0], 0.0)


def test_compute_policy_sensitivity_metrics_detects_intent_dependence():
    model = _DummyModel(num_intents=4)
    metrics = compute_policy_sensitivity_metrics(
        model,
        [_single_obs(), _single_obs()],
        num_intents=4,
    )
    assert metrics["num_states"] == 2.0
    assert metrics["num_pairs"] == 12.0
    assert metrics["policy_kl_mean"] > 0.1
    assert metrics["policy_kl_max"] >= metrics["policy_kl_mean"]
    assert metrics["policy_tv_mean"] > 0.1
    assert metrics["action_flip_rate"] > 0.1


def test_intent_policy_sensitivity_callback_logs_metrics(monkeypatch):
    model = _DummyModel(num_intents=4)
    callback = IntentPolicySensitivityCallback(
        enabled=True,
        num_intents=4,
        sample_states=4,
        log_freq_rollouts=1,
    )
    callback.init_callback(model)

    logged = []
    monkeypatch.setattr(
        "basketworld.utils.callbacks.mlflow.log_metric",
        lambda name, value, step=None: logged.append((name, float(value), step)),
    )

    callback._on_rollout_start()
    callback.locals = {
        "infos": [{}, {}],
        "new_obs": {
            "players": np.zeros((2, 6, 15), dtype=np.float32),
            "globals": np.array(
                [
                    [24.0, 0.0, 0.1, 0.2, 0.0, 0.0, 0.0, 0.0],
                    [24.0, 0.0, 0.1, 0.2, 0.0, 0.0, 0.0, 0.0],
                ],
                dtype=np.float32,
            ),
            "action_mask": np.ones((2, 2, 4), dtype=np.float32),
            "role_flag": np.array([[1.0], [-1.0]], dtype=np.float32),
            "intent_index": np.zeros((2, 1), dtype=np.float32),
            "intent_active": np.ones((2, 1), dtype=np.float32),
            "intent_visible": np.ones((2, 1), dtype=np.float32),
            "intent_age_norm": np.zeros((2, 1), dtype=np.float32),
        },
    }
    assert callback._on_step() is True
    callback._on_rollout_end()

    metric_names = {name for name, _, _ in logged}
    assert "intent/policy_kl_mean" in metric_names
    assert "intent/policy_kl_max" in metric_names
    assert "intent/policy_tv_mean" in metric_names
    assert "intent/action_flip_rate" in metric_names
    assert "intent/policy_sensitivity_samples" in metric_names
    assert "intent/policy_sensitivity_pairs" in metric_names
