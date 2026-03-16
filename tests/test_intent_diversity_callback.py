import numpy as np
import torch

import basketworld.utils.callbacks as callbacks_module
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
    def __init__(self, n_steps: int = 2, n_envs: int = 2):
        self.num_timesteps = 100
        self.policy = _DummyPolicy()
        self.rollout_buffer = _DummyRolloutBuffer(n_steps=n_steps, n_envs=n_envs)
        self._last_obs = {"obs": np.zeros((n_envs, 4), dtype=np.float32)}
        self._last_episode_starts = np.array([False] * n_envs, dtype=bool)


def _make_step_obs(
    role0: float,
    role1: float,
    *,
    active0: float = 1.0,
    active1: float = 1.0,
    idx0: float = 1.0,
    idx1: float = 2.0,
):
    return {
        "obs": np.zeros((2, 4), dtype=np.float32),
        "role_flag": np.array([[role0], [role1]], dtype=np.float32),
        "intent_active": np.array([[active0], [active1]], dtype=np.float32),
        "intent_index": np.array([[idx0], [idx1]], dtype=np.float32),
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
    shot_2pt_share, shot_3pt_share, shot_dunk_share = (
        IntentDiversityCallback._episode_shot_type_shares(ep)
    )
    assert np.isclose(ppp, 2.0)
    assert np.isclose(shot_2pt_share, 1.0)
    assert np.isclose(shot_3pt_share, 0.0)
    assert np.isclose(shot_dunk_share, 0.0)


def test_log_intent_behavior_metrics_logs_shot_type_shares():
    cb = IntentDiversityCallback(enabled=True, num_intents=4)
    model = _DummyModel()
    model.ep_info_buffer = [
        {
            "training_team": "offense",
            "intent_active": 1.0,
            "intent_index": 2.0,
            "made_2pt": 0.0,
            "made_3pt": 0.0,
            "made_dunk": 1.0,
            "defensive_lane_violation": 0.0,
            "attempts": 1.0,
            "turnover": 0.0,
            "passes": 3.0,
            "shot_2pt": 0.0,
            "shot_3pt": 0.0,
            "shot_dunk": 1.0,
        }
    ]
    cb.init_callback(model)

    logged = {}
    original_log_metric = callbacks_module.mlflow.log_metric

    def _capture_metric(key, value, step=None):
        logged[key] = (value, step)

    callbacks_module.mlflow.log_metric = _capture_metric
    try:
        cb._log_intent_behavior_metrics(global_step=123)
    finally:
        callbacks_module.mlflow.log_metric = original_log_metric

    assert "intent/ppp_by_intent/2" in logged
    assert "intent/pass_rate_by_intent/2" in logged
    assert "intent/shot_2pt_share_by_intent/2" in logged
    assert "intent/shot_3pt_share_by_intent/2" in logged
    assert "intent/shot_dunk_share_by_intent/2" in logged
    assert "intent/shot_dist_by_intent/2" in logged
    assert np.isclose(logged["intent/shot_2pt_share_by_intent/2"][0], 0.0)
    assert np.isclose(logged["intent/shot_3pt_share_by_intent/2"][0], 0.0)
    assert np.isclose(logged["intent/shot_dunk_share_by_intent/2"][0], 1.0)
    assert np.isclose(logged["intent/shot_dist_by_intent/2"][0], 0.0)


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


def test_intent_diversity_callback_gru_path_uses_sequence_lengths():
    cb = IntentDiversityCallback(
        enabled=True,
        num_intents=4,
        beta_target=1.0,
        warmup_steps=0,
        ramp_steps=1,
        bonus_clip=10.0,
        disc_encoder_type="gru",
        disc_step_dim=8,
    )
    model = _DummyModel()
    cb.init_callback(model)
    cb._on_training_start()

    class _FixedStats:
        mean = 0.0
        std = 1.0

        @staticmethod
        def update(_):
            return None

    cb._bonus_stats = _FixedStats()
    seen = {"train_lengths": None, "auc_lengths": None, "bonus_lengths": None}

    def _train(x_np, y_np, lengths_np=None):
        seen["train_lengths"] = None if lengths_np is None else lengths_np.copy()
        return 0.1, 0.9

    def _auc(x_np, y_np, lengths_np=None):
        seen["auc_lengths"] = None if lengths_np is None else lengths_np.copy()
        return 0.75

    def _bonus(x_np, y_np, lengths_np=None):
        seen["bonus_lengths"] = None if lengths_np is None else lengths_np.copy()
        return np.array([1.0], dtype=np.float32)

    cb._train_discriminator = _train
    cb._compute_disc_auc = _auc
    cb._compute_episode_bonus = _bonus

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

    assert seen["train_lengths"] is not None
    assert seen["auc_lengths"] is not None
    assert seen["bonus_lengths"] is not None
    assert seen["train_lengths"].tolist() == [2]
    assert np.any(model.rollout_buffer.rewards[:, 0] != 0.0)
    assert np.all(model.rollout_buffer.rewards[:, 1] == 0.0)


def test_intent_diversity_bonus_applies_only_to_active_prefix():
    cb = IntentDiversityCallback(
        enabled=True,
        num_intents=4,
        beta_target=1.0,
        warmup_steps=0,
        ramp_steps=1,
        bonus_clip=10.0,
    )
    model = _DummyModel(n_steps=3, n_envs=2)
    cb.init_callback(model)
    cb._on_training_start()

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
        "new_obs": _make_step_obs(1.0, -1.0, active0=1.0, active1=1.0),
    }
    cb._on_step()
    cb.locals = {
        "infos": [{}, {}],
        "dones": np.array([False, False], dtype=bool),
        "actions": np.zeros((2, 2), dtype=np.int64),
        "new_obs": _make_step_obs(1.0, -1.0, active0=1.0, active1=1.0),
    }
    cb._on_step()
    cb.locals = {
        "infos": [{}, {}],
        "dones": np.array([True, True], dtype=bool),
        "actions": np.zeros((2, 2), dtype=np.int64),
        "new_obs": _make_step_obs(1.0, -1.0, active0=0.0, active1=1.0),
    }
    cb._on_step()

    cb._on_rollout_end()

    assert np.all(model.rollout_buffer.rewards[:2, 0] != 0.0)
    assert np.all(model.rollout_buffer.rewards[2:, 0] == 0.0)
    assert np.all(model.rollout_buffer.rewards[:, 1] == 0.0)
