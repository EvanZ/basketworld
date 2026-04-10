from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch
import copy
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv

from basketworld.algorithms import IntegratedIntentSelectorPPO
from basketworld.policies.set_attention_policy import SetAttentionDualCriticPolicy
from train.callbacks import build_intent_selector_callback


def _make_obs_space(
    n_players: int,
    token_dim: int = 15,
    globals_dim: int = 8,
    n_actions: int = 14,
):
    return spaces.Dict(
        {
            "players": spaces.Box(
                -np.inf, np.inf, (n_players, token_dim), np.float32
            ),
            "globals": spaces.Box(-np.inf, np.inf, (globals_dim,), np.float32),
            "action_mask": spaces.Box(0.0, 1.0, (n_players, n_actions), np.float32),
            "role_flag": spaces.Box(-1.0, 1.0, (1,), np.float32),
            "skills": spaces.Box(-np.inf, np.inf, (n_players // 2 * 3,), np.float32),
            "intent_index": spaces.Box(0.0, 8.0, (1,), np.float32),
            "intent_active": spaces.Box(0.0, 1.0, (1,), np.float32),
            "intent_visible": spaces.Box(0.0, 1.0, (1,), np.float32),
            "intent_age_norm": spaces.Box(0.0, 1.0, (1,), np.float32),
        }
    )


def _make_obs(
    n_players: int,
    *,
    role: float,
    intent_index: int = 0,
    intent_active: bool = True,
    intent_visible: bool = True,
    n_actions: int = 14,
    token_dim: int = 15,
    globals_dim: int = 8,
):
    globals_vec = np.zeros((globals_dim,), dtype=np.float32)
    if globals_dim >= 8:
        globals_vec[-4:] = np.asarray(
            [
                float(intent_index) / 3.0,
                1.0 if intent_active else 0.0,
                1.0 if intent_visible else 0.0,
                0.0,
            ],
            dtype=np.float32,
        )
    return {
        "players": np.zeros((n_players, token_dim), dtype=np.float32),
        "globals": globals_vec,
        "action_mask": np.ones((n_players, n_actions), dtype=np.float32),
        "role_flag": np.asarray([role], dtype=np.float32),
        "skills": np.zeros((n_players // 2 * 3,), dtype=np.float32),
        "intent_index": np.asarray([float(intent_index)], dtype=np.float32),
        "intent_active": np.asarray(
            [1.0 if intent_active else 0.0], dtype=np.float32
        ),
        "intent_visible": np.asarray(
            [1.0 if intent_visible else 0.0], dtype=np.float32
        ),
        "intent_age_norm": np.asarray([0.0], dtype=np.float32),
    }


class _DummySelectorEnv(gym.Env):
    metadata = {}

    def __init__(
        self,
        *,
        role: float = 1.0,
        episode_reward: float = 1.0,
        steps_per_episode: int = 1,
        pass_boundary_steps: tuple[int, ...] = (),
    ):
        super().__init__()
        self.n_players = 6
        self.players_per_side = 3
        self.n_actions = 14
        self.role = float(role)
        self.episode_reward = float(episode_reward)
        self.steps_per_episode = int(steps_per_episode)
        self.observation_space = _make_obs_space(self.n_players, n_actions=self.n_actions)
        self.action_space = spaces.MultiDiscrete([self.n_actions] * self.players_per_side)
        self._steps = 0
        self.intent_index = 0
        self.intent_active = True
        self.intent_visible = True
        self.set_intent_calls: list[tuple[int, bool, int]] = []
        self.pass_boundary_steps = tuple(int(step) for step in pass_boundary_steps)

    def _obs(self):
        return _make_obs(
            self.n_players,
            role=self.role,
            intent_index=self.intent_index,
            intent_active=self.intent_active,
            intent_visible=self.intent_visible,
            n_actions=self.n_actions,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._steps = 0
        self.intent_index = 0
        self.intent_active = True
        self.intent_visible = True
        return self._obs(), {}

    def step(self, action):
        self._steps += 1
        terminated = self._steps >= self.steps_per_episode
        reward = self.episode_reward if terminated else 0.0
        info = {}
        if (not terminated) and self._steps in self.pass_boundary_steps:
            info["action_results"] = {
                "passes": {
                    0: {"success": True, "target": 1},
                }
            }
        return self._obs(), reward, terminated, False, info

    def set_offense_intent_state(
        self,
        intent_index: int,
        *,
        intent_active: bool = True,
        intent_age: int = 0,
        intent_commitment_remaining=None,
    ):
        self.intent_index = int(intent_index)
        self.intent_active = bool(intent_active)
        self.set_intent_calls.append((int(intent_index), bool(intent_active), int(intent_age)))


def _make_model(env, **algo_kwargs):
    kwargs = dict(
        verbose=0,
        n_steps=1,
        batch_size=2,
        n_epochs=1,
        learning_rate=1e-3,
        gamma=0.0,
        device="cpu",
        policy_kwargs={
            "intent_embedding_enabled": True,
            "intent_embedding_dim": 8,
            "num_intents": 4,
            "intent_selector_enabled": True,
            "intent_selector_hidden_dim": 16,
        },
        intent_selector_enabled=True,
        num_intents=4,
        intent_selector_alpha_start=1.0,
        intent_selector_alpha_end=1.0,
        intent_selector_alpha_warmup_steps=0,
        intent_selector_alpha_ramp_steps=0,
        intent_selector_entropy_coef=0.0,
        intent_selector_usage_reg_coef=0.0,
        intent_selector_value_coef=0.5,
    )
    kwargs.update(algo_kwargs)
    model = IntegratedIntentSelectorPPO(
        SetAttentionDualCriticPolicy,
        env,
        **kwargs,
    )
    selector_head = model.policy.intent_selector_head
    assert selector_head is not None
    assert model.policy.intent_selector_value_head is not None
    with torch.no_grad():
        selector_head[0].weight.zero_()
        selector_head[0].bias.zero_()
        selector_head[-1].weight.zero_()
        selector_head[-1].bias.fill_(-1e9)
        selector_head[-1].bias[2] = 0.0
    return model


def test_integrated_selector_applies_only_to_offense_episode_starts(monkeypatch):
    env_refs = []

    def _factory(role: float):
        def _make():
            env = _DummySelectorEnv(role=role, episode_reward=1.0, steps_per_episode=1)
            env_refs.append(env)
            return env

        return _make

    env = DummyVecEnv([_factory(1.0), _factory(-1.0)])
    logged = {}
    monkeypatch.setattr(
        "basketworld.algorithms.integrated_mu_selector_ppo.mlflow.log_metric",
        lambda name, value, step=None: logged.setdefault(name, float(value)),
    )

    model = _make_model(env)
    model.learn(total_timesteps=2)

    offense_env, defense_env = env_refs
    assert len(offense_env.set_intent_calls) >= 1
    assert defense_env.set_intent_calls == []
    assert np.isclose(model._selector_last_metrics["intent/selector_samples"], 1.0)
    assert np.isclose(logged["intent/selector_usage_by_intent/2"], 1.0)


def test_integrated_selector_uses_full_possession_return(monkeypatch):
    env = DummyVecEnv(
        [lambda: _DummySelectorEnv(role=1.0, episode_reward=2.0, steps_per_episode=2)]
    )
    logged = {}
    monkeypatch.setattr(
        "basketworld.algorithms.integrated_mu_selector_ppo.mlflow.log_metric",
        lambda name, value, step=None: logged.setdefault(name, float(value)),
    )

    model = _make_model(env, n_steps=2)
    model.learn(total_timesteps=2)

    assert np.isclose(model._selector_last_metrics["intent/selector_return_mean"], 2.0)
    assert np.isclose(logged["intent/selector_return_mean"], 2.0)
    assert np.isclose(logged["intent/selector_return_by_intent/2"], 2.0)
    assert "intent/selector_value_loss" in logged
    assert "intent/selector_clip_fraction" in logged


def test_integrated_selector_eps_floor_preserves_effective_prob_mass():
    model = _make_model(
        DummyVecEnv(
            [
                lambda: _DummySelectorEnv(
                    role=1.0, episode_reward=1.0, steps_per_episode=1
                ),
                lambda: _DummySelectorEnv(
                    role=1.0, episode_reward=1.0, steps_per_episode=1
                ),
            ]
        ),
        intent_selector_eps_start=0.25,
        intent_selector_eps_end=0.25,
        intent_selector_eps_warmup_steps=0,
        intent_selector_eps_ramp_steps=0,
        n_steps=1,
        batch_size=2,
    )
    selector_probs = torch.tensor(
        [[0.0, 0.0, 1.0, 0.0]], dtype=torch.float32, device=model.device
    )
    mixed = model._selector_apply_eps_floor(selector_probs, 0.25)
    mixed_np = mixed.detach().cpu().numpy()

    assert np.isclose(model._selector_eps_current(), 0.25)
    assert np.isclose(mixed_np[0, 0], 0.0625, atol=1e-6)
    assert np.isclose(mixed_np[0, 1], 0.0625, atol=1e-6)
    assert np.isclose(mixed_np[0, 2], 0.8125, atol=1e-6)
    assert np.isclose(mixed_np[0, 3], 0.0625, atol=1e-6)


def test_callback_builder_skips_legacy_selector_when_integrated_mode():
    args = SimpleNamespace(
        intent_selector_enabled=True,
        intent_selector_mode="integrated",
        num_intents=4,
        intent_selector_alpha_start=0.0,
        intent_selector_alpha_end=1.0,
        intent_selector_alpha_warmup_steps=0,
        intent_selector_alpha_ramp_steps=1,
        intent_selector_entropy_coef=0.01,
        intent_selector_usage_reg_coef=0.01,
    )

    assert build_intent_selector_callback(args) is None


def test_integrated_selector_model_save_load_roundtrip(tmp_path):
    env = DummyVecEnv(
        [lambda: _DummySelectorEnv(role=1.0, episode_reward=1.0, steps_per_episode=1)]
    )
    model = _make_model(env, n_steps=2)

    save_path = tmp_path / "integrated_selector_model.zip"
    model.save(save_path)
    loaded = IntegratedIntentSelectorPPO.load(save_path, env=env, device="cpu")

    assert loaded.intent_selector_enabled is True
    assert loaded.intent_selector_num_intents == 4
    assert loaded._policy_has_intent_selector() is True
    assert loaded._policy_has_intent_selector_value_head() is True


def test_integrated_selector_multiselect_reselects_on_completed_pass(monkeypatch):
    env_ref: list[_DummySelectorEnv] = []

    def _make():
        env = _DummySelectorEnv(
            role=1.0,
            episode_reward=1.0,
            steps_per_episode=3,
            pass_boundary_steps=(1,),
        )
        env_ref.append(env)
        return env

    env = DummyVecEnv([_make])
    logged = {}
    monkeypatch.setattr(
        "basketworld.algorithms.integrated_mu_selector_ppo.mlflow.log_metric",
        lambda name, value, step=None: logged.setdefault(name, float(value)),
    )

    model = _make_model(
        env,
        n_steps=3,
        intent_selector_multiselect_enabled=True,
        intent_selector_min_play_steps=1,
    )
    model.learn(total_timesteps=3)

    test_env = env_ref[0]
    assert len(test_env.set_intent_calls) >= 2
    assert np.isclose(model._selector_last_metrics["intent/selector_samples"], 2.0)
    assert (
        model._selector_last_metrics["intent/selector_segment_steps_mean"] >= 1.0
    )
    assert np.isclose(logged["intent/multisegment_episode_rate"], 1.0)
    assert np.isclose(logged["intent/segments_per_episode_mean"], 2.0)
    assert np.isclose(
        logged["intent/segment_boundary_reason_completed_pass_rate"], 1.0
    )
    assert np.isclose(
        logged["intent/segment_boundary_reason_commitment_timeout_rate"], 0.0
    )
    assert np.isclose(logged["intent/selector_segment_index_count/0"], 1.0)
    assert np.isclose(logged["intent/selector_segment_index_count/1"], 1.0)
    assert np.isclose(logged["intent/selector_usage_by_segment/0/2"], 1.0)
    assert np.isclose(logged["intent/selector_usage_by_segment/1/2"], 1.0)


def test_integrated_selector_multiselect_reselects_on_commitment_timeout(monkeypatch):
    env_ref: list[_DummySelectorEnv] = []

    def _make():
        env = _DummySelectorEnv(
            role=1.0,
            episode_reward=1.0,
            steps_per_episode=3,
            pass_boundary_steps=(),
        )
        env_ref.append(env)
        return env

    env = DummyVecEnv([_make])
    logged = {}
    monkeypatch.setattr(
        "basketworld.algorithms.integrated_mu_selector_ppo.mlflow.log_metric",
        lambda name, value, step=None: logged.setdefault(name, float(value)),
    )

    model = _make_model(
        env,
        n_steps=3,
        intent_selector_multiselect_enabled=True,
        intent_selector_min_play_steps=3,
        intent_commitment_steps=2,
    )
    model.learn(total_timesteps=3)

    test_env = env_ref[0]
    assert len(test_env.set_intent_calls) >= 2
    assert np.isclose(model._selector_last_metrics["intent/selector_samples"], 2.0)
    assert (
        model._selector_last_metrics["intent/selector_segment_steps_mean"] >= 1.0
    )
    assert np.isclose(logged["intent/multisegment_episode_rate"], 1.0)
    assert np.isclose(logged["intent/segments_per_episode_mean"], 2.0)
    assert np.isclose(
        logged["intent/segment_boundary_reason_completed_pass_rate"], 0.0
    )
    assert np.isclose(
        logged["intent/segment_boundary_reason_commitment_timeout_rate"], 1.0
    )
    assert np.isclose(logged["intent/selector_segment_index_count/0"], 1.0)
    assert np.isclose(logged["intent/selector_segment_index_count/1"], 1.0)
    assert np.isclose(logged["intent/selector_usage_by_segment/0/2"], 1.0)
    assert np.isclose(logged["intent/selector_usage_by_segment/1/2"], 1.0)


def test_integrated_selector_loads_legacy_checkpoint_without_value_head():
    env = DummyVecEnv(
        [lambda: _DummySelectorEnv(role=1.0, episode_reward=1.0, steps_per_episode=1)]
    )
    model = _make_model(env, n_steps=2)
    params = copy.deepcopy(model.get_parameters())
    params["policy"] = {
        key: value
        for key, value in params["policy"].items()
        if "intent_selector_value_head" not in key
    }

    reloaded = _make_model(env, n_steps=2)
    reloaded.set_parameters(params, exact_match=True, device="cpu")

    assert reloaded._policy_has_intent_selector_value_head() is True
