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

import math
from typing import Optional, Tuple, List

import torch as th
from torch import nn
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.distributions import Distribution
from basketworld.policies import DualCriticActorCriticPolicy


def _compute_pass_bias_floor(
    logits: th.Tensor,
    pass_indices: List[int],
    base_bias: float,
    pass_prob_min: float,
) -> th.Tensor:
    batch = logits.shape[0]
    base = th.full((batch,), float(base_bias), device=logits.device, dtype=logits.dtype)
    if pass_prob_min is None or pass_prob_min <= 0.0:
        return base
    pass_idx = [idx for idx in pass_indices if idx < logits.shape[1]]
    if not pass_idx or len(pass_idx) == logits.shape[1]:
        return base
    non_pass_mask = th.ones(logits.shape[1], dtype=th.bool, device=logits.device)
    non_pass_mask[pass_idx] = False
    pass_logits = logits[:, pass_idx]
    non_pass_logits = logits[:, non_pass_mask]
    if non_pass_logits.shape[1] == 0:
        return base
    p_min = min(max(float(pass_prob_min), 0.0), 1.0 - 1e-6)
    log_p = math.log(p_min)
    log_1mp = math.log1p(-p_min)
    log_s = th.logsumexp(pass_logits, dim=1)
    log_r = th.logsumexp(non_pass_logits, dim=1)
    b_needed = log_p + log_r - log_1mp - log_s
    total = th.maximum(base, b_needed)
    valid = th.isfinite(log_s) & th.isfinite(log_r)
    return th.where(valid, total, base)


class PassBiasMultiInputPolicy(MultiInputActorCriticPolicy):
    """Multi-input Actor-Critic policy with additive pass-logit bias.

    Exposes set_pass_logit_bias(float) so a training callback can schedule it.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Scalar bias added to each PASS logit (same for all players). 0.0 disables.
        self.pass_logit_bias: float = 0.0
        self.pass_prob_min: float = 0.0
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

    def set_pass_prob_min(self, value: float) -> None:
        try:
            self.pass_prob_min = float(value)
        except Exception:
            self.pass_prob_min = 0.0

    def _apply_pass_bias(self, action_logits: th.Tensor) -> th.Tensor:
        base_bias = float(self.pass_logit_bias)
        pass_prob_min = float(getattr(self, "pass_prob_min", 0.0) or 0.0)
        if not self._nvec or (abs(base_bias) <= 1e-12 and pass_prob_min <= 0.0):
            return action_logits

        with th.no_grad():
            sizes = self._nvec
        chunks = th.split(action_logits, sizes, dim=1)

        biased_chunks: List[th.Tensor] = []
        for chunk in chunks:
            if chunk.shape[1] == 0:
                biased_chunks.append(chunk)
                continue
            c = chunk.clone()
            pass_idx = [idx for idx in self._pass_indices if idx < c.shape[1]]
            if pass_idx:
                total_bias = _compute_pass_bias_floor(
                    chunk, pass_idx, base_bias, pass_prob_min
                )
                c[:, pass_idx] = c[:, pass_idx] + total_bias.unsqueeze(-1)
            biased_chunks.append(c)

        return th.cat(biased_chunks, dim=1)

    def _get_action_dist_from_latent(
        self, latent_pi: th.Tensor, latent_sde: Optional[th.Tensor] = None
    ) -> Distribution:
        """Override to inject bias into PASS logits before building the distribution."""
        # Base logits from action head
        action_logits: th.Tensor = self.action_net(latent_pi)
        biased_logits = self._apply_pass_bias(action_logits)
        return self.action_dist.proba_distribution(action_logits=biased_logits)


class PassBiasDualCriticPolicy(DualCriticActorCriticPolicy):
    """Multi-input Actor-Critic policy with dual critics AND additive pass-logit bias.
    
    Combines:
    - DualCriticActorCriticPolicy: Separate value networks for offense/defense,
      and optionally separate action networks (dual policy)
    - PassBias: Additive bias to PASS action logits for exploration
    
    This policy addresses the fundamental value function issue in zero-sum self-play
    while maintaining the pass exploration mechanism.
    
    When use_dual_policy=True is passed to __init__, separate action networks are used
    for offense and defense, allowing each role to learn distinct strategies.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Scalar bias added to each PASS logit (same for all players). 0.0 disables.
        self.pass_logit_bias: float = 0.0
        self.pass_prob_min: float = 0.0
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

    def set_pass_prob_min(self, value: float) -> None:
        try:
            self.pass_prob_min = float(value)
        except Exception:
            self.pass_prob_min = 0.0

    def _apply_pass_bias(self, action_logits: th.Tensor) -> th.Tensor:
        """
        Apply pass logit bias to action logits.
        
        :param action_logits: Raw action logits
        :return: Biased action logits (or original if bias is zero)
        """
        base_bias = float(self.pass_logit_bias)
        pass_prob_min = float(getattr(self, "pass_prob_min", 0.0) or 0.0)
        # No-op if not MultiDiscrete or bias/prob floor is effectively zero
        if not self._nvec or (abs(base_bias) <= 1e-12 and pass_prob_min <= 0.0):
            return action_logits

        # Split flat logits into per-dimension chunks: [B, sum(nvec)] -> list of [B, n_i]
        with th.no_grad():
            # Prepare split sizes
            sizes = self._nvec
        chunks = th.split(action_logits, sizes, dim=1)

        # Add bias to PASS indices in each categorical
        biased_chunks: List[th.Tensor] = []
        for i, chunk in enumerate(chunks):
            if chunk.shape[1] == 0:
                biased_chunks.append(chunk)
                continue
            # Clone to avoid in-place on autograd graph of original logits
            c = chunk.clone()
            pass_idx = [idx for idx in self._pass_indices if idx < c.shape[1]]
            if pass_idx:
                total_bias = _compute_pass_bias_floor(
                    chunk, pass_idx, base_bias, pass_prob_min
                )
                c[:, pass_idx] = c[:, pass_idx] + total_bias.unsqueeze(-1)
            biased_chunks.append(c)

        return th.cat(biased_chunks, dim=1)

    def _get_action_dist_from_latent(
        self, latent_pi: th.Tensor, latent_sde: Optional[th.Tensor] = None
    ) -> Distribution:
        """
        Override to inject bias into PASS logits before building the distribution.
        
        Uses the base class's _get_action_logits to properly handle dual policy
        (selecting offense or defense action network based on role_flag), then
        applies pass bias on top.
        """
        # Get action logits from the appropriate action network
        # (handles dual policy routing if use_dual_policy=True)
        action_logits: th.Tensor = self._get_action_logits(latent_pi)
        
        # Apply pass bias
        biased_logits = self._apply_pass_bias(action_logits)
        
        return self.action_dist.proba_distribution(action_logits=biased_logits)


