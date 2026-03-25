from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import basketworld.utils.callbacks as callbacks_module
from basketworld.utils.callbacks import IntentDiversityCallback
from basketworld.utils.intent_discovery import CompletedIntentEpisode, IntentTransition
from train.policy_utils import get_latest_discriminator_checkpoint_path


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
    cb._train_discriminator = lambda x_np, y_np, **kwargs: (0.1, 0.9)
    cb._compute_episode_bonus = (
        lambda x_np, y_np, **kwargs: np.array([1.0], dtype=np.float32)
    )

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


def test_intent_diversity_callback_logs_clipped_and_shaping_bonus_metrics():
    cb = IntentDiversityCallback(
        enabled=True,
        num_intents=4,
        beta_target=0.5,
        warmup_steps=0,
        ramp_steps=1,
        bonus_clip=1.0,
    )
    model = _DummyModel()
    model.num_timesteps = 100
    cb.init_callback(model)
    cb._on_training_start()

    class _FixedStats:
        mean = 0.0
        std = 1.0

        @staticmethod
        def update(_):
            return None

    cb._bonus_stats = _FixedStats()
    cb._train_discriminator = lambda x_np, y_np, **kwargs: (0.1, 0.9)
    cb._compute_disc_eval_top1_acc = lambda x_np, y_np, **kwargs: 0.42
    cb._compute_disc_auc = lambda x_np, y_np, **kwargs: 0.75
    cb._compute_episode_bonus = (
        lambda x_np, y_np, **kwargs: np.array([2.0], dtype=np.float32)
    )

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

    logged = {}
    original_log_metric = callbacks_module.mlflow.log_metric

    def _capture_metric(key, value, step=None):
        logged[key] = (value, step)

    callbacks_module.mlflow.log_metric = _capture_metric
    try:
        cb._on_rollout_end()
    finally:
        callbacks_module.mlflow.log_metric = original_log_metric

    assert np.isclose(logged["intent/bonus_raw_mean"][0], 2.0)
    assert np.isclose(logged["intent/bonus_raw_std"][0], 0.0)
    assert np.isclose(logged["intent/bonus_norm_mean"][0], 2.0)
    assert np.isclose(logged["intent/bonus_norm_std"][0], 0.0)
    assert np.isclose(logged["intent/bonus_clipped_mean"][0], 1.0)
    assert np.isclose(logged["intent/bonus_clipped_std"][0], 0.0)
    assert np.isclose(logged["intent/bonus_shaping_per_episode_mean"][0], 0.5)
    assert np.isclose(logged["intent/bonus_shaping_per_episode_std"][0], 0.0)
    assert np.isclose(logged["intent/bonus_shaping_per_step_mean"][0], 0.25)
    assert np.isclose(logged["intent/bonus_shaping_per_step_std"][0], 0.0)
    assert np.isclose(logged["intent/disc_top1_acc_eval"][0], 0.42)


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
    assert "intent/episodes_by_intent/2" in logged
    assert "intent/points_by_intent/2" in logged
    assert "intent/possessions_by_intent/2" in logged
    assert "intent/pass_rate_by_intent/2" in logged
    assert "intent/shot_2pt_share_by_intent/2" in logged
    assert "intent/shot_3pt_share_by_intent/2" in logged
    assert "intent/shot_dunk_share_by_intent/2" in logged
    assert "intent/shot_dist_by_intent/2" in logged
    assert np.isclose(logged["intent/episodes_by_intent/2"][0], 1.0)
    assert np.isclose(logged["intent/points_by_intent/2"][0], 2.0)
    assert np.isclose(logged["intent/possessions_by_intent/2"][0], 1.0)
    assert np.isclose(logged["intent/ppp_by_intent/2"][0], 2.0)
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


def test_build_disc_aux_targets_uses_segment_terminal_info():
    episode_with_shot = CompletedIntentEpisode(
        intent_index=1,
        transitions=[
            IntentTransition(
                feature=np.zeros((4,), dtype=np.float32),
                buffer_step_idx=0,
                env_idx=0,
                role_flag=1.0,
                intent_active=True,
                intent_index=1,
            )
        ],
        terminal_info={
            "action_results": {
                "shots": {
                    0: {"expected_points": 1.7, "success": False},
                }
            }
        },
    )
    episode_with_pass = CompletedIntentEpisode(
        intent_index=2,
        transitions=[
            IntentTransition(
                feature=np.zeros((4,), dtype=np.float32),
                buffer_step_idx=1,
                env_idx=0,
                role_flag=1.0,
                intent_active=True,
                intent_index=2,
            )
        ],
        terminal_info={
            "action_results": {
                "passes": {
                    0: {"success": True, "target": 1},
                }
            }
        },
    )

    shot_end, shot_quality, shot_quality_mask = (
        IntentDiversityCallback._build_disc_aux_targets(
            [episode_with_shot, episode_with_pass]
        )
    )

    assert np.allclose(shot_end, np.array([1.0, 0.0], dtype=np.float32))
    assert np.allclose(shot_quality, np.array([1.7, 0.0], dtype=np.float32))
    assert np.allclose(shot_quality_mask, np.array([1.0, 0.0], dtype=np.float32))


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

    def _train(x_np, y_np, lengths_np=None, **kwargs):
        seen["train_lengths"] = None if lengths_np is None else lengths_np.copy()
        return 0.1, 0.9

    def _auc(x_np, y_np, lengths_np=None, **kwargs):
        seen["auc_lengths"] = None if lengths_np is None else lengths_np.copy()
        return 0.75

    def _bonus(x_np, y_np, lengths_np=None, **kwargs):
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
    cb._train_discriminator = lambda x_np, y_np, **kwargs: (0.1, 0.9)
    cb._compute_episode_bonus = (
        lambda x_np, y_np, **kwargs: np.array([1.0], dtype=np.float32)
    )

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


def test_intent_diversity_callback_exports_discriminator_checkpoint(tmp_path: Path):
    cb = IntentDiversityCallback(enabled=True, num_intents=4, disc_encoder_type="gru")
    model = _DummyModel()
    cb.init_callback(model)
    cb._on_training_start()
    cb._maybe_build_discriminator(input_dim=12)

    out_path = tmp_path / "intent_disc_iter_1.pt"
    saved = cb.export_discriminator_checkpoint(
        str(out_path), global_step=1234, alternation_idx=7
    )

    payload = torch.load(out_path, map_location="cpu", weights_only=False)
    assert saved is True
    assert out_path.exists()
    assert "optimizer_state_dict" in payload
    assert payload["config"]["encoder_type"] == "gru"
    assert payload["config"]["input_dim"] == 272
    assert payload["meta"]["global_step"] == 1234
    assert payload["meta"]["alternation_idx"] == 7


def test_intent_diversity_callback_exports_latest_eval_batch(tmp_path: Path):
    cb = IntentDiversityCallback(enabled=True, num_intents=4, disc_encoder_type="gru")
    model = _DummyModel()
    model.num_timesteps = 4321
    cb.init_callback(model)
    cb._on_training_start()
    cb._last_disc_eval_acc = 0.36
    cb._last_disc_auc = 0.82
    cb._latest_disc_eval_batch = {
        "x": np.arange(24, dtype=np.float32).reshape(2, 3, 4),
        "y": np.array([1, 3], dtype=np.int64),
        "lengths": np.array([3, 2], dtype=np.int64),
    }

    out_path = tmp_path / "intent_disc_eval_batch_iter_5.npz"
    saved = cb.export_latest_discriminator_eval_batch(
        str(out_path), global_step=4321, alternation_idx=5
    )

    assert saved is True
    assert out_path.exists()
    payload = np.load(out_path, allow_pickle=True)
    assert payload["x"].shape == (2, 3, 4)
    assert payload["y"].tolist() == [1, 3]
    assert payload["lengths"].tolist() == [3, 2]
    assert int(payload["global_step"][0]) == 4321
    assert int(payload["alternation_idx"][0]) == 5
    assert np.isclose(float(payload["disc_top1_acc_eval"][0]), 0.36, atol=1e-6)
    assert np.isclose(float(payload["disc_auc_ovr_macro"][0]), 0.82, atol=1e-6)


def test_restore_discriminator_checkpoint_payload_restores_state_and_meta(tmp_path: Path):
    cb = IntentDiversityCallback(enabled=True, num_intents=4, disc_encoder_type="gru")
    model = _DummyModel()
    cb.init_callback(model)
    cb._on_training_start()
    cb._maybe_build_discriminator(input_dim=272)
    assert cb._disc is not None

    with torch.no_grad():
        for param in cb._disc.parameters():
            param.fill_(0.25)

    out_path = tmp_path / "intent_disc_iter_3.pt"
    cb._last_disc_loss = 0.12
    cb._last_disc_acc = 0.34
    cb._last_disc_eval_acc = 0.23
    cb._last_disc_auc = 0.56
    saved = cb.export_discriminator_checkpoint(
        str(out_path), global_step=4321, alternation_idx=9
    )
    assert saved is True

    restored_cb = IntentDiversityCallback(
        enabled=True, num_intents=4, disc_encoder_type="gru"
    )
    restored_model = _DummyModel()
    restored_cb.init_callback(restored_model)
    restored_cb._on_training_start()

    payload = torch.load(out_path, map_location="cpu", weights_only=False)
    restored = restored_cb.restore_discriminator_checkpoint_payload(
        payload, source="unit-test"
    )

    assert restored is True
    assert restored_cb._disc is not None
    for expected_param, restored_param in zip(
        cb._disc.parameters(), restored_cb._disc.parameters()
    ):
        assert torch.allclose(expected_param, restored_param)
    assert np.isclose(restored_cb._last_disc_loss, 0.12)
    assert np.isclose(restored_cb._last_disc_acc, 0.34)
    assert np.isclose(restored_cb._last_disc_eval_acc, 0.23)
    assert np.isclose(restored_cb._last_disc_auc, 0.56)
    assert restored_cb._rollout_counter == 0


def test_queue_discriminator_checkpoint_restore_rejects_incompatible_payload():
    cb = IntentDiversityCallback(enabled=True, num_intents=4, disc_encoder_type="gru")
    payload = {
        "state_dict": {},
        "config": {
            "input_dim": 999,
            "hidden_dim": 128,
            "num_intents": 4,
            "dropout": 0.1,
            "encoder_type": "gru",
            "step_dim": 64,
            "max_obs_dim": 256,
            "max_action_dim": 16,
        },
    }

    queued = cb.queue_discriminator_checkpoint_restore(payload, source="bad-payload")

    assert queued is False
    assert cb._pending_disc_restore_payload is None


def test_get_latest_discriminator_checkpoint_path_selects_highest_index():
    class _Client:
        @staticmethod
        def list_artifacts(_run_id, _artifact_path):
            return [
                SimpleNamespace(path="models/unified_iter_1.zip"),
                SimpleNamespace(path="models/intent_disc_iter_2.pt"),
                SimpleNamespace(path="models/intent_disc_iter_17.pt"),
                SimpleNamespace(path="models/intent_disc_iter_5.pt"),
            ]

    latest = get_latest_discriminator_checkpoint_path(_Client(), "dummy-run")

    assert latest == "models/intent_disc_iter_17.pt"
