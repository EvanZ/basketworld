import numpy as np
import torch as th
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from basketworld.policies.set_attention_policy import (
    PointerTargetedMultiCategoricalDistribution,
    SetAttentionDualCriticPolicy,
)
from basketworld.envs.basketworld_env_v2 import ActionType


def _make_obs_space(n_players: int, token_dim: int = 15, globals_dim: int = 3, n_actions: int = 14):
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
    token_dim: int = 15,
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


def test_pointer_targeted_log_prob_matches_action_probabilities():
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
    policy.set_pass_mode("pointer_targeted")

    obs = _make_obs(n_players, n_actions=n_actions, role=1.0)
    tensor_obs, _ = policy.obs_to_tensor(obs)
    distribution = policy.get_distribution(tensor_obs)

    actions = th.tensor(
        [[ActionType.MOVE_E.value, ActionType.PASS_E.value, ActionType.PASS_NE.value]],
        dtype=th.long,
    )
    log_prob = distribution.log_prob(actions)
    probs = distribution.action_probabilities()
    manual = th.log(
        probs[0, th.arange(players_per_side), actions[0]] + 1e-12
    ).sum().unsqueeze(0)
    assert th.allclose(log_prob, manual, atol=1e-5)


def test_pointer_targeted_pass_prob_floor_applies_to_pass_mass():
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
    policy.set_pass_mode("pointer_targeted")
    policy.set_pass_prob_min(0.35)
    policy.set_pass_logit_bias(0.0)

    obs = _make_obs(n_players, n_actions=n_actions, role=1.0)
    tensor_obs, _ = policy.obs_to_tensor(obs)
    distribution = policy.get_distribution(tensor_obs)
    probs = distribution.action_probabilities()
    pass_mass = probs[
        ...,
        ActionType.PASS_E.value : ActionType.PASS_SE.value + 1,
    ].sum(dim=-1)

    assert th.all(pass_mass >= (0.35 - 1e-4))


def test_pointer_targeted_dual_policy_defense_smoke():
    n_players = 6
    players_per_side = 3
    n_actions = 14
    obs_space = _make_obs_space(n_players, n_actions=n_actions)
    action_space = spaces.MultiDiscrete([n_actions] * players_per_side)

    policy = SetAttentionDualCriticPolicy(
        obs_space,
        action_space,
        lr_schedule=lambda _: 0.0003,
        use_dual_policy=True,
    )
    policy.set_pass_mode("pointer_targeted")

    obs = _make_obs(n_players, n_actions=n_actions, role=-1.0)
    tensor_obs, _ = policy.obs_to_tensor(obs)
    actions, values, log_prob = policy.forward(tensor_obs, deterministic=True)

    assert actions.shape == (1, players_per_side)
    assert values.shape == (1, 1)
    assert log_prob.shape == (1,)
    assert th.all(actions >= 0)
    assert th.all(actions < n_actions)


def test_set_attention_policy_load_state_dict_allows_missing_pointer_keys():
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

    full_state = policy.state_dict()
    legacy_state = {k: v for k, v in full_state.items() if not k.startswith("pointer_")}
    result = policy.load_state_dict(legacy_state, strict=True)
    assert all(str(k).startswith("pointer_") for k in result.missing_keys)


def test_set_attention_policy_intent_embedding_smoke():
    n_players = 6
    players_per_side = 3
    n_actions = 14
    obs_space = _make_obs_space(n_players, globals_dim=8, n_actions=n_actions)
    action_space = spaces.MultiDiscrete([n_actions] * players_per_side)

    policy = SetAttentionDualCriticPolicy(
        obs_space,
        action_space,
        lr_schedule=lambda _: 0.0003,
        intent_embedding_enabled=True,
        intent_embedding_dim=12,
        num_intents=8,
    )

    globals_vec = np.array([24.0, 0.0, 0.0, 0.0, 0.42, 1.0, 1.0, 0.25], dtype=np.float32)
    obs = _make_obs(n_players, globals_dim=8, n_actions=n_actions, role=1.0, globals_vec=globals_vec)
    tensor_obs, _ = policy.obs_to_tensor(obs)
    actions, values, log_prob = policy.forward(tensor_obs, deterministic=True)

    assert actions.shape == (1, players_per_side)
    assert values.shape == (1, 1)
    assert log_prob.shape == (1,)


def test_set_attention_policy_intent_selector_logits_shape():
    n_players = 6
    players_per_side = 3
    n_actions = 14
    obs_space = _make_obs_space(n_players, globals_dim=8, n_actions=n_actions)
    action_space = spaces.MultiDiscrete([n_actions] * players_per_side)

    policy = SetAttentionDualCriticPolicy(
        obs_space,
        action_space,
        lr_schedule=lambda _: 0.0003,
        intent_embedding_enabled=True,
        intent_embedding_dim=12,
        num_intents=8,
        intent_selector_enabled=True,
        intent_selector_hidden_dim=32,
    )

    globals_vec = np.array([24.0, 0.0, 0.0, 0.0, 0.42, 1.0, 1.0, 0.25], dtype=np.float32)
    obs = _make_obs(
        n_players,
        globals_dim=8,
        n_actions=n_actions,
        role=1.0,
        globals_vec=globals_vec,
    )
    tensor_obs, _ = policy.obs_to_tensor(obs)
    logits = policy.get_intent_selector_logits(tensor_obs)

    assert logits.shape == (1, 8)
    assert th.isfinite(logits).all()


def test_set_attention_policy_load_state_dict_allows_missing_intent_embedding_keys():
    n_players = 6
    players_per_side = 3
    n_actions = 14
    obs_space = _make_obs_space(n_players, globals_dim=8, n_actions=n_actions)
    action_space = spaces.MultiDiscrete([n_actions] * players_per_side)

    policy = SetAttentionDualCriticPolicy(
        obs_space,
        action_space,
        lr_schedule=lambda _: 0.0003,
        intent_embedding_enabled=True,
        intent_embedding_dim=8,
        num_intents=8,
    )

    full_state = policy.state_dict()
    legacy_state = {
        k: v
        for k, v in full_state.items()
        if not (
            k.startswith("pointer_")
            or k.startswith("features_extractor.intent_embedding")
            or k.startswith("features_extractor.intent_to_global")
            or k.startswith("features_extractor.offense_intent_embedding")
            or k.startswith("features_extractor.defense_intent_embedding")
            or k.startswith("features_extractor.offense_intent_to_token")
            or k.startswith("features_extractor.defense_intent_to_token")
        )
    }
    result = policy.load_state_dict(legacy_state, strict=True)
    allowed_prefixes = (
        "pointer_",
        "features_extractor.intent_embedding",
        "features_extractor.intent_to_global",
        "features_extractor.offense_intent_embedding",
        "features_extractor.defense_intent_embedding",
        "features_extractor.offense_intent_to_token",
        "features_extractor.defense_intent_to_token",
    )
    assert all(str(k).startswith(allowed_prefixes) for k in result.missing_keys)


def test_set_attention_policy_load_state_dict_allows_missing_selector_keys():
    n_players = 6
    players_per_side = 3
    n_actions = 14
    obs_space = _make_obs_space(n_players, globals_dim=8, n_actions=n_actions)
    action_space = spaces.MultiDiscrete([n_actions] * players_per_side)

    policy = SetAttentionDualCriticPolicy(
        obs_space,
        action_space,
        lr_schedule=lambda _: 0.0003,
        intent_embedding_enabled=True,
        intent_embedding_dim=8,
        num_intents=8,
        intent_selector_enabled=True,
    )

    full_state = policy.state_dict()
    legacy_state = {
        k: v
        for k, v in full_state.items()
        if not (
            k.startswith("pointer_")
            or k.startswith("intent_selector_head")
            or k.startswith("intent_selector_value_head")
        )
    }
    result = policy.load_state_dict(legacy_state, strict=True)
    allowed_prefixes = ("pointer_", "intent_selector_head", "intent_selector_value_head")
    assert all(str(k).startswith(allowed_prefixes) for k in result.missing_keys)


def test_set_attention_policy_uses_role_selected_intent_embeddings():
    n_players = 6
    players_per_side = 3
    n_actions = 14
    obs_space = _make_obs_space(n_players, globals_dim=8, n_actions=n_actions)
    action_space = spaces.MultiDiscrete([n_actions] * players_per_side)

    policy = SetAttentionDualCriticPolicy(
        obs_space,
        action_space,
        lr_schedule=lambda _: 0.0003,
        intent_embedding_enabled=True,
        intent_embedding_dim=4,
        num_intents=8,
    )
    extractor = policy.features_extractor
    assert extractor.offense_intent_embedding is not None
    assert extractor.defense_intent_embedding is not None
    assert extractor.offense_intent_to_token is not None
    assert extractor.defense_intent_to_token is not None

    with th.no_grad():
        extractor.offense_intent_embedding.weight.zero_()
        extractor.defense_intent_embedding.weight.zero_()
        extractor.offense_intent_embedding.weight[3].fill_(1.0)
        extractor.defense_intent_embedding.weight[3].fill_(1.0)
        extractor.offense_intent_to_token.weight.fill_(0.5)
        extractor.defense_intent_to_token.weight.fill_(-0.5)

    globals_vec = np.array([24.0, 0.0, 0.0, 0.0, 3.0 / 7.0, 1.0, 1.0, 0.25], dtype=np.float32)
    obs_off = _make_obs(
        n_players,
        globals_dim=8,
        n_actions=n_actions,
        role=1.0,
        globals_vec=globals_vec,
    )
    obs_def = _make_obs(
        n_players,
        globals_dim=8,
        n_actions=n_actions,
        role=-1.0,
        globals_vec=globals_vec,
    )

    tensor_off, _ = policy.obs_to_tensor(obs_off)
    tensor_def, _ = policy.obs_to_tensor(obs_def)
    latent_off = extractor(tensor_off)
    latent_def = extractor(tensor_def)

    assert not th.allclose(latent_off, latent_def)


def test_pointer_targeted_backward_pass_no_inplace_error():
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
    policy.set_pass_mode("pointer_targeted")
    policy.set_pass_prob_min(0.35)

    obs = _make_obs(n_players, n_actions=n_actions, role=1.0)
    tensor_obs, _ = policy.obs_to_tensor(obs)
    actions, _, _ = policy.forward(tensor_obs, deterministic=False)
    values, log_prob, entropy = policy.evaluate_actions(tensor_obs, actions)

    loss = -(log_prob.mean() + 0.01 * entropy.mean()) + 0.1 * values.mean()
    loss.backward()

    grads = [p.grad for p in policy.parameters() if p.requires_grad]
    assert any(g is not None for g in grads)


def test_pointer_targeted_mode_uses_joint_action_argmax():
    # non-pass actions are 0..7, pass slots are 8..13
    dist = PointerTargetedMultiCategoricalDistribution(
        action_dim=14,
        non_pass_action_indices=list(range(8)),
        pass_action_indices=list(range(8, 14)),
    )

    # Build a case where PASS type has highest type-probability,
    # but each individual PASS action has lower probability than SHOOT (7).
    # If mode() incorrectly does hierarchical argmax, it would select PASS.
    type_probs = np.array(
        [0.037, 0.037, 0.037, 0.037, 0.037, 0.037, 0.038, 0.34, 0.40],
        dtype=np.float32,
    )
    slot_probs = np.array([0.30, 0.20, 0.10, 0.10, 0.15, 0.15], dtype=np.float32)

    action_type_logits = th.log(th.as_tensor(type_probs)).view(1, 1, -1)
    pass_target_logits = th.log(th.as_tensor(slot_probs)).view(1, 1, -1)

    dist = dist.proba_distribution(
        action_type_logits=action_type_logits,
        pass_target_logits=pass_target_logits,
    )
    actions = dist.mode()
    probs = dist.action_probabilities()
    expected = th.argmax(probs, dim=-1)

    assert actions.shape == (1, 1)
    assert actions.item() == int(expected.item())
    # Explicitly check we picked SHOOT (action 7), not PASS slots.
    assert actions.item() == 7


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
