import numpy as np
import torch as th
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from basketworld.policies.set_attention_policy import SetAttentionDualCriticPolicy


def _make_obs_space(n_players: int, token_dim: int = 11, globals_dim: int = 3, n_actions: int = 14):
    return spaces.Dict(
        {
            "players": spaces.Box(-np.inf, np.inf, (n_players, token_dim), np.float32),
            "globals": spaces.Box(-np.inf, np.inf, (globals_dim,), np.float32),
            "action_mask": spaces.Box(0.0, 1.0, (n_players, n_actions), np.float32),
            "role_flag": spaces.Box(-1.0, 1.0, (1,), np.float32),
            "skills": spaces.Box(-np.inf, np.inf, (n_players // 2 * 3,), np.float32),
        }
    )


def _make_obs(
    n_players: int,
    token_dim: int = 11,
    globals_dim: int = 3,
    n_actions: int = 14,
    role: float = 1.0,
    players: np.ndarray | None = None,
    globals_vec: np.ndarray | None = None,
    action_mask: np.ndarray | None = None,
):
    return {
        "players": (
            np.zeros((n_players, token_dim), dtype=np.float32)
            if players is None
            else players.astype(np.float32)
        ),
        "globals": (
            np.zeros((globals_dim,), dtype=np.float32)
            if globals_vec is None
            else globals_vec.astype(np.float32)
        ),
        "action_mask": (
            np.ones((n_players, n_actions), dtype=np.float32)
            if action_mask is None
            else action_mask.astype(np.float32)
        ),
        "role_flag": np.array([role], dtype=np.float32),
        "skills": np.zeros((n_players // 2 * 3,), dtype=np.float32),
    }


def test_set_attention_policy_offense_team_action_shape():
    n_players = 6
    players_per_side = 3
    n_actions = 14
    obs_space = _make_obs_space(n_players, n_actions=n_actions)
    action_space = spaces.MultiDiscrete([n_actions] * players_per_side)

    policy = SetAttentionDualCriticPolicy(
        obs_space,
        action_space,
        lr_schedule=lambda _: 0.0003,
    )

    obs = _make_obs(n_players, n_actions=n_actions, role=1.0)
    tensor_obs, _ = policy.obs_to_tensor(obs)
    actions, values, log_prob = policy.forward(tensor_obs, deterministic=True)

    assert actions.shape == (1, players_per_side)
    assert values.shape == (1, 1)
    assert log_prob.shape == (1,)


def test_set_attention_policy_all_players_action_shape():
    n_players = 6
    n_actions = 14
    obs_space = _make_obs_space(n_players, n_actions=n_actions)
    action_space = spaces.MultiDiscrete([n_actions] * n_players)

    policy = SetAttentionDualCriticPolicy(
        obs_space,
        action_space,
        lr_schedule=lambda _: 0.0003,
    )

    obs = _make_obs(n_players, n_actions=n_actions, role=-1.0)
    tensor_obs, _ = policy.obs_to_tensor(obs)
    actions, values, log_prob = policy.forward(tensor_obs, deterministic=True)

    assert actions.shape == (1, n_players)
    assert values.shape == (1, 1)
    assert log_prob.shape == (1,)


def test_set_attention_policy_permutation_equivariance():
    rng = np.random.default_rng(0)
    n_players = 6
    n_actions = 14
    token_dim = 8
    globals_dim = 3

    obs_space = _make_obs_space(n_players, token_dim=token_dim, globals_dim=globals_dim, n_actions=n_actions)
    action_space = spaces.MultiDiscrete([n_actions] * n_players)
    policy = SetAttentionDualCriticPolicy(
        obs_space,
        action_space,
        lr_schedule=lambda _: 0.0003,
    )
    policy.eval()

    players = rng.normal(size=(n_players, token_dim)).astype(np.float32)
    globals_vec = rng.normal(size=(globals_dim,)).astype(np.float32)
    obs = _make_obs(
        n_players,
        token_dim=token_dim,
        globals_dim=globals_dim,
        n_actions=n_actions,
        role=1.0,
        players=players,
        globals_vec=globals_vec,
    )

    perm = rng.permutation(n_players)
    inv_perm = np.argsort(perm)
    perm_players = players[perm]
    perm_obs = _make_obs(
        n_players,
        token_dim=token_dim,
        globals_dim=globals_dim,
        n_actions=n_actions,
        role=1.0,
        players=perm_players,
        globals_vec=globals_vec,
    )

    obs_tensor, _ = policy.obs_to_tensor(obs)
    features = policy.extract_features(obs_tensor)
    latent_pi, latent_vf = policy.mlp_extractor(features)
    logits = policy._get_action_logits(latent_pi).reshape(1, n_players, n_actions)
    values = policy._get_value_from_latent(latent_vf, obs_tensor["role_flag"]).detach()

    perm_tensor, _ = policy.obs_to_tensor(perm_obs)
    perm_features = policy.extract_features(perm_tensor)
    perm_latent_pi, perm_latent_vf = policy.mlp_extractor(perm_features)
    perm_logits = policy._get_action_logits(perm_latent_pi).reshape(1, n_players, n_actions)
    perm_values = policy._get_value_from_latent(perm_latent_vf, perm_tensor["role_flag"]).detach()

    logits = logits.detach().cpu().numpy()
    perm_logits = perm_logits.detach().cpu().numpy()
    perm_reordered = perm_logits[:, inv_perm, :]
    np.testing.assert_allclose(logits, perm_reordered, atol=1e-5)
    np.testing.assert_allclose(values.cpu().numpy(), perm_values.cpu().numpy(), atol=1e-5)


def test_set_attention_policy_ppo_smoke_step():
    n_players = 6
    token_dim = 8
    globals_dim = 3
    n_actions = 14

    class DummySetObsEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self):
            self.action_space = spaces.MultiDiscrete([n_actions] * n_players)
            self.observation_space = _make_obs_space(
                n_players, token_dim=token_dim, globals_dim=globals_dim, n_actions=n_actions
            )
            self._steps = 0

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self._steps = 0
            obs = _make_obs(n_players, token_dim=token_dim, globals_dim=globals_dim, n_actions=n_actions)
            return obs, {}

        def step(self, action):
            self._steps += 1
            obs = _make_obs(n_players, token_dim=token_dim, globals_dim=globals_dim, n_actions=n_actions)
            reward = 0.0
            terminated = self._steps >= 1
            truncated = False
            return obs, reward, terminated, truncated, {}

    env = DummyVecEnv([lambda: DummySetObsEnv()])
    model = PPO(
        SetAttentionDualCriticPolicy,
        env,
        n_steps=2,
        batch_size=2,
        n_epochs=1,
        learning_rate=1e-3,
        gamma=0.0,
        device="cpu",
    )
    model.learn(total_timesteps=2)
