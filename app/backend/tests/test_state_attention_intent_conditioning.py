import numpy as np
import torch

import app.backend.state as backend_state
from basketworld.envs.basketworld_env_v2 import HexagonBasketballEnv, Team
from basketworld.policies import SetAttentionDualCriticPolicy
from basketworld.utils.wrappers import SetObservationWrapper


class _DummyModel:
    def __init__(self, policy):
        self.policy = policy


def _make_policy_for_env(env: HexagonBasketballEnv) -> SetAttentionDualCriticPolicy:
    wrapped = SetObservationWrapper(env)
    policy = SetAttentionDualCriticPolicy(
        wrapped.observation_space,
        env.action_space,
        lr_schedule=lambda _: 1e-3,
        embed_dim=32,
        n_heads=2,
        token_mlp_dim=32,
        num_cls_tokens=2,
        intent_embedding_enabled=True,
        intent_embedding_dim=8,
        num_intents=max(2, int(getattr(env, "num_intents", 8))),
        intent_selector_enabled=False,
    )
    extractor = policy.features_extractor
    with torch.no_grad():
        extractor.offense_intent_embedding.weight.zero_()
        extractor.defense_intent_embedding.weight.zero_()
        extractor.offense_intent_embedding.weight[1].fill_(2.0)
        extractor.defense_intent_embedding.weight[1].fill_(-2.0)
        extractor.offense_intent_to_token.weight.normal_(mean=0.0, std=0.5)
        extractor.defense_intent_to_token.weight.normal_(mean=0.0, std=0.5)
        if extractor.offense_intent_to_token.bias is not None:
            extractor.offense_intent_to_token.bias.zero_()
        if extractor.defense_intent_to_token.bias is not None:
            extractor.defense_intent_to_token.bias.zero_()
    policy.eval()
    return policy


def test_attention_payload_changes_with_live_intent_override():
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
        env.intent_age = 0

        backend_state.game_state.env = env
        backend_state.game_state.obs = obs
        backend_state.game_state.user_team = Team.OFFENSE
        backend_state.game_state.unified_policy = _DummyModel(_make_policy_for_env(env))
        backend_state.game_state.defense_policy = None
        backend_state.game_state.reward_history = []
        backend_state.game_state.episode_rewards = {"offense": 0.0, "defense": 0.0}
        backend_state.game_state.shot_log = []
        backend_state.game_state.phi_log = []
        backend_state.game_state.actions_log = []
        backend_state.game_state.episode_states = []

        env.intent_index = 0
        state_z0 = backend_state.get_full_game_state()
        attn_z0 = np.asarray(state_z0["obs_tokens"]["attention"]["weights_avg"], dtype=np.float32)

        env.intent_index = 1
        state_z1 = backend_state.get_full_game_state()
        attn_z1 = np.asarray(state_z1["obs_tokens"]["attention"]["weights_avg"], dtype=np.float32)

        assert attn_z0.shape == attn_z1.shape
        assert not np.allclose(attn_z0, attn_z1)
    finally:
        backend_state.game_state.__dict__.clear()
        backend_state.game_state.__dict__.update(original_state)
