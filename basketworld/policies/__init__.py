"""Custom policies for BasketWorld RL training."""

from basketworld.policies.dual_critic_policy import DualCriticActorCriticPolicy
from basketworld.policies.set_attention_policy import (
    SetAttentionDualCriticPolicy,
    SetAttentionExtractor,
)

__all__ = [
    "DualCriticActorCriticPolicy",
    "SetAttentionDualCriticPolicy",
    "SetAttentionExtractor",
]
