import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.save_util import load_from_zip_file, save_to_zip_file
from gymnasium import spaces

from basketworld.envs.basketworld_env_v2 import HexagonBasketballEnv, Team
from basketworld.policies.set_attention_policy import SetAttentionDualCriticPolicy
from basketworld.utils.policy_loading import load_ppo_for_inference
from basketworld.utils.policy_proxy import FrozenPolicyProxy
from basketworld.utils.self_play_wrapper import SelfPlayEnvWrapper


class _DummyInnerPolicy:
    def __init__(self):
        self.pass_mode_calls: list[str] = []

    def set_pass_mode(self, mode: str) -> None:
        self.pass_mode_calls.append(str(mode))


class _DummySB3Policy:
    def __init__(self, output_actions):
        self.policy = _DummyInnerPolicy()
        self._output_actions = np.array(output_actions, dtype=int)
        self.last_obs = None

    def predict(self, obs, deterministic=False):
        self.last_obs = obs
        del deterministic
        return np.array(self._output_actions, dtype=int), None


class _MiniSetAttentionEnv(gym.Env):
    metadata = {}

    def __init__(self):
        super().__init__()
        self.n_players = 6
        self.players_per_side = 3
        self.n_actions = 14
        self.observation_space = spaces.Dict(
            {
                "players": spaces.Box(
                    -np.inf, np.inf, (self.n_players, 15), np.float32
                ),
                "globals": spaces.Box(-np.inf, np.inf, (8,), np.float32),
                "action_mask": spaces.Box(
                    0.0, 1.0, (self.n_players, self.n_actions), np.float32
                ),
                "role_flag": spaces.Box(-1.0, 1.0, (1,), np.float32),
                "skills": spaces.Box(
                    -np.inf, np.inf, (self.players_per_side * 3,), np.float32
                ),
                "intent_index": spaces.Box(0.0, 8.0, (1,), np.float32),
                "intent_active": spaces.Box(0.0, 1.0, (1,), np.float32),
                "intent_visible": spaces.Box(0.0, 1.0, (1,), np.float32),
                "intent_age_norm": spaces.Box(0.0, 1.0, (1,), np.float32),
            }
        )
        self.action_space = spaces.MultiDiscrete([self.n_actions] * self.players_per_side)

    def _obs(self):
        return {
            "players": np.zeros((self.n_players, 15), dtype=np.float32),
            "globals": np.zeros((8,), dtype=np.float32),
            "action_mask": np.ones((self.n_players, self.n_actions), dtype=np.float32),
            "role_flag": np.asarray([1.0], dtype=np.float32),
            "skills": np.zeros((self.players_per_side * 3,), dtype=np.float32),
            "intent_index": np.asarray([0.0], dtype=np.float32),
            "intent_active": np.asarray([1.0], dtype=np.float32),
            "intent_visible": np.asarray([1.0], dtype=np.float32),
            "intent_age_norm": np.asarray([0.0], dtype=np.float32),
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return self._obs(), {}

    def step(self, action):
        del action
        return self._obs(), 0.0, True, False, {}


def _make_env(training_team=Team.OFFENSE, pass_mode="pointer_targeted"):
    return HexagonBasketballEnv(
        players=3,
        render_mode=None,
        training_team=training_team,
        pass_mode=pass_mode,
    )


def test_wrapper_applies_pass_mode_to_opponent_policy_object():
    env = _make_env(training_team=Team.OFFENSE, pass_mode="pointer_targeted")
    opponent = _DummySB3Policy(output_actions=[0, 0, 0])

    wrapped = SelfPlayEnvWrapper(env, opponent_policy=opponent)
    wrapped.reset(seed=7)

    assert opponent.policy.pass_mode_calls, "Expected at least one pass mode application call."
    assert opponent.policy.pass_mode_calls[-1] == "pointer_targeted"


def test_wrapper_maps_full_length_opponent_actions_to_opponent_team(monkeypatch):
    env = _make_env(training_team=Team.OFFENSE, pass_mode="directional")
    n_players = env.n_players
    players_per_side = env.players_per_side

    # Legacy-style output: one action per absolute player id.
    # Defense ids are expected to receive the tail actions [4, 5, 6] for players=3.
    opponent = _DummySB3Policy(output_actions=[1, 2, 3, 4, 5, 6][:n_players])
    wrapped = SelfPlayEnvWrapper(env, opponent_policy=opponent)
    wrapped.reset(seed=11)

    # Keep resolver transparent so we can verify raw remapping behavior.
    monkeypatch.setattr(
        "basketworld.utils.self_play_wrapper.resolve_illegal_actions",
        lambda predicted_actions, action_mask, strategy, deterministic, probs_per_player=None: np.array(predicted_actions, dtype=int),
    )

    captured = {}

    def _fake_env_step(full_action):
        captured["full_action"] = np.array(full_action, dtype=int)
        obs = wrapped._last_obs
        reward = np.zeros(wrapped.env.unwrapped.n_players, dtype=np.float32)
        done = True
        truncated = False
        info = {"action_results": {}, "training_team": wrapped.env.unwrapped.training_team.name}
        return obs, reward, done, truncated, info

    monkeypatch.setattr(wrapped.env, "step", _fake_env_step)

    training_action = np.zeros(players_per_side, dtype=int)
    wrapped.step(training_action)

    assert "full_action" in captured
    full_action = captured["full_action"]
    defense_ids = wrapped.opponent_player_ids
    assert full_action[defense_ids].tolist() == [4, 5, 6][: len(defense_ids)]


def test_wrapper_applies_pass_mode_when_loading_opponent_from_path(monkeypatch):
    env = _make_env(training_team=Team.DEFENSE, pass_mode="pointer_targeted")
    fake_loaded = _DummySB3Policy(output_actions=[0, 0, 0])

    monkeypatch.setattr(
        "basketworld.utils.self_play_wrapper.load_ppo_for_inference",
        lambda *args, **kwargs: fake_loaded,
    )

    wrapped = SelfPlayEnvWrapper(env, opponent_policy="/tmp/fake_opp.zip")
    wrapped.reset(seed=13)
    wrapped.step(np.zeros(env.players_per_side, dtype=int))

    assert fake_loaded.policy.pass_mode_calls, "Expected pass mode to be set on loaded opponent policy."
    assert fake_loaded.policy.pass_mode_calls[-1] == "pointer_targeted"


def test_wrapper_reconditions_intent_fields_for_opponent_role():
    env = HexagonBasketballEnv(
        players=3,
        render_mode=None,
        training_team=Team.DEFENSE,
        enable_intent_learning=True,
        intent_null_prob=0.0,
        intent_visible_to_defense_prob=0.0,
    )
    opponent = _DummySB3Policy(output_actions=[0, 0, 0])
    wrapped = SelfPlayEnvWrapper(env, opponent_policy=opponent)

    obs, _ = wrapped.reset(seed=5)
    assert float(obs["intent_active"][0]) == 0.0
    assert float(obs["intent_visible"][0]) == 0.0

    wrapped.step(np.zeros(env.players_per_side, dtype=int))
    assert opponent.last_obs is not None
    assert float(opponent.last_obs["role_flag"][0]) == 1.0
    assert float(opponent.last_obs["intent_visible"][0]) == 1.0


def test_wrapper_set_offense_intent_state_reconditions_cached_training_obs():
    env = HexagonBasketballEnv(
        players=3,
        render_mode=None,
        training_team=Team.OFFENSE,
        enable_intent_learning=True,
        intent_null_prob=0.0,
    )
    opponent = _DummySB3Policy(output_actions=[0, 0, 0])
    wrapped = SelfPlayEnvWrapper(env, opponent_policy=opponent)

    wrapped.reset(seed=17)
    wrapped.set_offense_intent_state(2, intent_active=True, intent_age=0)

    assert wrapped._last_obs is not None
    assert float(wrapped._last_obs["intent_index"][0]) == 2.0
    assert float(wrapped._last_obs["intent_active"][0]) == 1.0
    assert float(wrapped._last_obs["intent_visible"][0]) == 1.0


def test_wrapper_get_offense_intent_override_defaults_to_none():
    env = _make_env(training_team=Team.OFFENSE, pass_mode="directional")
    opponent = _DummySB3Policy(output_actions=[0, 0, 0])
    wrapped = SelfPlayEnvWrapper(env, opponent_policy=opponent)

    wrapped.reset(seed=19)

    assert wrapped.get_offense_intent_override() is None


def test_inference_loader_skips_legacy_optimizer_state(tmp_path):
    env = _MiniSetAttentionEnv()
    model = PPO(
        SetAttentionDualCriticPolicy,
        env,
        n_steps=8,
        batch_size=4,
        n_epochs=1,
        learning_rate=1e-3,
        device="cpu",
        policy_kwargs={
            "intent_embedding_enabled": True,
            "intent_embedding_dim": 8,
            "num_intents": 8,
            "intent_selector_enabled": True,
        },
        verbose=0,
    )
    source_path = tmp_path / "current.zip"
    legacy_path = tmp_path / "legacy.zip"
    model.save(source_path)

    data, params, pytorch_variables = load_from_zip_file(source_path, device="cpu")
    assert data is not None
    assert params is not None

    params["policy"] = {
        key: value
        for key, value in params["policy"].items()
        if "intent_selector_value_head" not in key
    }
    optimizer_state = params["policy.optimizer"]
    for group in optimizer_state["param_groups"]:
        if len(group["params"]) >= 4:
            removed = set(group["params"][-4:])
            group["params"] = list(group["params"][:-4])
            for param_id in removed:
                optimizer_state["state"].pop(param_id, None)

    save_to_zip_file(
        legacy_path,
        data=data,
        params=params,
        pytorch_variables=pytorch_variables,
    )

    try:
        PPO.load(legacy_path, device="cpu")
        raised = False
    except ValueError as exc:
        raised = "parameter group" in str(exc)
    assert raised is True

    loaded = load_ppo_for_inference(str(legacy_path), device="cpu")
    obs, _ = env.reset(seed=23)
    action, _ = loaded.predict(obs, deterministic=False)
    assert action is not None


def test_policy_proxy_uses_inference_loader_for_legacy_checkpoint(tmp_path):
    env = _MiniSetAttentionEnv()
    model = PPO(
        SetAttentionDualCriticPolicy,
        env,
        n_steps=8,
        batch_size=4,
        n_epochs=1,
        learning_rate=1e-3,
        device="cpu",
        policy_kwargs={
            "intent_embedding_enabled": True,
            "intent_embedding_dim": 8,
            "num_intents": 8,
            "intent_selector_enabled": True,
        },
        verbose=0,
    )
    source_path = tmp_path / "proxy_source.zip"
    legacy_path = tmp_path / "proxy_legacy.zip"
    model.save(source_path)

    data, params, pytorch_variables = load_from_zip_file(source_path, device="cpu")
    assert data is not None
    assert params is not None

    params["policy"] = {
        key: value
        for key, value in params["policy"].items()
        if "intent_selector_value_head" not in key
    }
    optimizer_state = params["policy.optimizer"]
    for group in optimizer_state["param_groups"]:
        if len(group["params"]) >= 4:
            removed = set(group["params"][-4:])
            group["params"] = list(group["params"][:-4])
            for param_id in removed:
                optimizer_state["state"].pop(param_id, None)

    save_to_zip_file(
        legacy_path,
        data=data,
        params=params,
        pytorch_variables=pytorch_variables,
    )

    proxy = FrozenPolicyProxy(str(legacy_path), device="cpu")
    obs, _ = env.reset(seed=29)
    action, _ = proxy.predict(obs, deterministic=False)
    assert action is not None
