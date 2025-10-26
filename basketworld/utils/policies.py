from __future__ import annotations

"""
Custom SB3 policy that adds an additive bias to the logits of PASS actions
in a MultiDiscrete action space (one categorical per player).

This is used to increase exploration of passes early in training and is
annealed toward zero via a scheduler.

Design notes:
- We subclass MultiInputActorCriticPolicy and override _get_action_dist_from_latent
  to add a scalar bias to PASS action logits before constructing the distribution.
- The PASS indices are the same for every player dimension: those whose action
  name starts with "PASS_". By convention in this env, ActionType PASS indices
  are contiguous and follow SHOOT: [8..13] for 14 total actions.
- We compute flat logits from self.action_net(latent_pi), then split per
  dimension using the action space's nvec to add the bias to the PASS slots
  in each categorical, then re-concatenate.
"""

from typing import Optional, Tuple, List

import torch as th
from torch import nn
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.distributions import Distribution
from basketworld.policies import DualCriticActorCriticPolicy


class PassBiasMultiInputPolicy(MultiInputActorCriticPolicy):
    """Multi-input Actor-Critic policy with additive pass-logit bias.

    Exposes set_pass_logit_bias(float) so a training callback can schedule it.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Scalar bias added to each PASS logit (same for all players). 0.0 disables.
        self.pass_logit_bias: float = 0.0
        # Cache per-dimension sizes from MultiDiscrete
        try:
            self._nvec: List[int] = list(self.action_space.nvec)
        except Exception:
            self._nvec = []
        # In this env, PASS action indices are 8..13 within each categorical of size 14
        # We derive pass index list from nvec if available, falling back to [8..13].
        self._pass_indices: List[int] = []
        if self._nvec and len(self._nvec) > 0:
            per_dim = int(self._nvec[0])
            # Defensive clamp
            start = min(8, per_dim - 1)
            end = min(per_dim - 1, 13)
            self._pass_indices = [i for i in range(start, end + 1) if i < per_dim]
        else:
            self._pass_indices = [8, 9, 10, 11, 12, 13]

    def set_pass_logit_bias(self, value: float) -> None:
        try:
            self.pass_logit_bias = float(value)
        except Exception:
            self.pass_logit_bias = 0.0

    def _get_action_dist_from_latent(
        self, latent_pi: th.Tensor, latent_sde: Optional[th.Tensor] = None
    ) -> Distribution:
        """Override to inject bias into PASS logits before building the distribution."""
        # Base logits from action head
        action_logits: th.Tensor = self.action_net(latent_pi)

        # No-op if not MultiDiscrete or bias is effectively zero
        if not self._nvec or abs(self.pass_logit_bias) <= 1e-12:
            return self.action_dist.proba_distribution(action_logits=action_logits)

        # Split flat logits into per-dimension chunks: [B, sum(nvec)] -> list of [B, n_i]
        with th.no_grad():
            # Prepare split sizes
            sizes = self._nvec
        chunks = th.split(action_logits, sizes, dim=1)

        # Add bias to PASS indices in each categorical
        biased_chunks: List[th.Tensor] = []
        bias = th.as_tensor(
            self.pass_logit_bias, device=action_logits.device, dtype=action_logits.dtype
        )
        for i, chunk in enumerate(chunks):
            if chunk.shape[1] == 0:
                biased_chunks.append(chunk)
                continue
            # Clone to avoid in-place on autograd graph of original logits
            c = chunk.clone()
            for idx in self._pass_indices:
                if idx < c.shape[1]:
                    c[:, idx] = c[:, idx] + bias
            biased_chunks.append(c)

        biased_logits = th.cat(biased_chunks, dim=1)
        return self.action_dist.proba_distribution(action_logits=biased_logits)


class PassBiasDualCriticPolicy(DualCriticActorCriticPolicy):
    """Multi-input Actor-Critic policy with dual critics AND additive pass-logit bias.
    
    Combines:
    - DualCriticActorCriticPolicy: Separate value networks for offense/defense
    - PassBias: Additive bias to PASS action logits for exploration
    
    This policy addresses the fundamental value function issue in zero-sum self-play
    while maintaining the pass exploration mechanism.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Scalar bias added to each PASS logit (same for all players). 0.0 disables.
        self.pass_logit_bias: float = 0.0
        # Cache per-dimension sizes from MultiDiscrete
        try:
            self._nvec: List[int] = list(self.action_space.nvec)
        except Exception:
            self._nvec = []
        # In this env, PASS action indices are 8..13 within each categorical of size 14
        # We derive pass index list from nvec if available, falling back to [8..13].
        self._pass_indices: List[int] = []
        if self._nvec and len(self._nvec) > 0:
            per_dim = int(self._nvec[0])
            # Defensive clamp
            start = min(8, per_dim - 1)
            end = min(per_dim - 1, 13)
            self._pass_indices = [i for i in range(start, end + 1) if i < per_dim]
        else:
            self._pass_indices = [8, 9, 10, 11, 12, 13]

    def set_pass_logit_bias(self, value: float) -> None:
        try:
            self.pass_logit_bias = float(value)
        except Exception:
            self.pass_logit_bias = 0.0

    def _get_action_dist_from_latent(
        self, latent_pi: th.Tensor, latent_sde: Optional[th.Tensor] = None
    ) -> Distribution:
        """Override to inject bias into PASS logits before building the distribution."""
        # Base logits from action head
        action_logits: th.Tensor = self.action_net(latent_pi)

        # No-op if not MultiDiscrete or bias is effectively zero
        if not self._nvec or abs(self.pass_logit_bias) <= 1e-12:
            return self.action_dist.proba_distribution(action_logits=action_logits)

        # Split flat logits into per-dimension chunks: [B, sum(nvec)] -> list of [B, n_i]
        with th.no_grad():
            # Prepare split sizes
            sizes = self._nvec
        chunks = th.split(action_logits, sizes, dim=1)

        # Add bias to PASS indices in each categorical
        biased_chunks: List[th.Tensor] = []
        bias = th.as_tensor(
            self.pass_logit_bias, device=action_logits.device, dtype=action_logits.dtype
        )
        for i, chunk in enumerate(chunks):
            if chunk.shape[1] == 0:
                biased_chunks.append(chunk)
                continue
            # Clone to avoid in-place on autograd graph of original logits
            c = chunk.clone()
            for idx in self._pass_indices:
                if idx < c.shape[1]:
                    c[:, idx] = c[:, idx] + bias
            biased_chunks.append(c)

        biased_logits = th.cat(biased_chunks, dim=1)
        return self.action_dist.proba_distribution(action_logits=biased_logits)



