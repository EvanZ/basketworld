import numpy as np
import torch

from basketworld.utils.callbacks import IntentDiversityCallback


class _DummyRolloutBuffer:
    def __init__(self, n_steps: int = 2, n_envs: int = 2):
        self.rewards = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.recomputed = False

    def compute_returns_and_advantage(self, last_values, dones):
        self.recomputed = True
        self._last_values = last_values
        self._dones = dones


class _DummyPolicy:
    def __init__(self):
        self.device = torch.device("cpu")

    def obs_to_tensor(self, obs):
        arr = torch.as_tensor(obs["obs"], dtype=torch.float32)
        return arr, None

    def predict_values(self, obs_tensor):
        if obs_tensor.ndim == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        return torch.zeros((obs_tensor.shape[0], 1), dtype=torch.float32)


class _DummyModel:
    def __init__(self):
        self.num_timesteps = 100
        self.policy = _DummyPolicy()
        self.rollout_buffer = _DummyRolloutBuffer()
        self._last_obs = {"obs": np.zeros((2, 4), dtype=np.float32)}
        self._last_episode_starts = np.array([False, False], dtype=bool)


def _make_step_obs(role0: float, role1: float):
    return {
        "obs": np.zeros((2, 4), dtype=np.float32),
        "role_flag": np.array([[role0], [role1]], dtype=np.float32),
        "intent_active": np.array([[1.0], [1.0]], dtype=np.float32),
        "intent_index": np.array([[1.0], [2.0]], dtype=np.float32),
    }


def test_intent_diversity_callback_disabled_noop():
    cb = IntentDiversityCallback(enabled=False)
    model = _DummyModel()
    cb.init_callback(model)

    cb._on_rollout_start()
    cb.locals = {
        "infos": [{}, {}],
        "dones": np.array([True, True], dtype=bool),
        "actions": np.zeros((2, 2), dtype=np.int64),
        "new_obs": _make_step_obs(1.0, -1.0),
    }
    cb._on_step()
    before = model.rollout_buffer.rewards.copy()
    cb._on_rollout_end()
    after = model.rollout_buffer.rewards.copy()

    assert np.array_equal(before, after)
    assert not model.rollout_buffer.recomputed


def test_intent_diversity_callback_injects_offense_only_bonus():
    cb = IntentDiversityCallback(
        enabled=True,
        num_intents=4,
        beta_target=1.0,
        warmup_steps=0,
        ramp_steps=1,
        bonus_clip=10.0,
    )
    model = _DummyModel()
    cb.init_callback(model)
    cb._on_training_start()

    # Stabilize normalization for deterministic bonus assertion.
    class _FixedStats:
        mean = 0.0
        std = 1.0

        @staticmethod
        def update(_):
            return None

    cb._bonus_stats = _FixedStats()
    cb._train_discriminator = lambda x_np, y_np: (0.1, 0.9)
    cb._compute_episode_bonus = lambda x_np, y_np: np.array([1.0], dtype=np.float32)

    cb._on_rollout_start()
    cb.locals = {
        "infos": [{}, {}],
        "dones": np.array([False, False], dtype=bool),
        "actions": np.zeros((2, 2), dtype=np.int64),
        "new_obs": _make_step_obs(1.0, -1.0),
    }
    cb._on_step()
    cb.locals = {
        "infos": [{}, {}],
        "dones": np.array([True, True], dtype=bool),
        "actions": np.zeros((2, 2), dtype=np.int64),
        "new_obs": _make_step_obs(1.0, -1.0),
    }
    cb._on_step()

    cb._on_rollout_end()

    # Only offense env (env 0) should receive bonus.
    assert np.any(model.rollout_buffer.rewards[:, 0] != 0.0)
    assert np.all(model.rollout_buffer.rewards[:, 1] == 0.0)
    assert model.rollout_buffer.recomputed


def test_intent_episode_metric_helpers():
    ep = {
        "made_2pt": 1.0,
        "made_3pt": 0.0,
        "made_dunk": 0.0,
        "defensive_lane_violation": 0.0,
        "attempts": 1.0,
        "turnover": 0.0,
        "shot_2pt": 1.0,
        "shot_3pt": 0.0,
        "shot_dunk": 0.0,
    }
    ppp = IntentDiversityCallback._episode_ppp(ep)
    three_share = IntentDiversityCallback._episode_shot_three_share(ep)
    assert np.isclose(ppp, 2.0)
    assert np.isclose(three_share, 0.0)


def test_binary_auc_helper():
    y_true = np.array([0, 0, 1, 1], dtype=np.int64)
    y_score = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float64)
    auc = IntentDiversityCallback._binary_auc_from_scores(y_true, y_score)
    assert auc is not None
    assert np.isclose(auc, 1.0)


def test_multiclass_auc_ovr_macro_helper():
    # 3-class logits with clear separation by class.
    logits = np.array(
        [
            [5.0, 0.1, 0.1],  # class 0
            [0.2, 4.5, 0.1],  # class 1
            [0.1, 0.2, 4.8],  # class 2
            [4.2, 0.3, 0.2],  # class 0
            [0.1, 4.0, 0.1],  # class 1
            [0.1, 0.2, 4.2],  # class 2
        ],
        dtype=np.float64,
    )
    y = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
    auc = IntentDiversityCallback._multiclass_auc_ovr_macro(logits, y, num_classes=3)
    assert auc is not None
    assert auc > 0.99
