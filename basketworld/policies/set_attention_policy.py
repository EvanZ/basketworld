from __future__ import annotations

from typing import Any, Dict, Optional, Type

import torch as th
from torch import nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from basketworld.policies.dual_critic_policy import DualCriticActorCriticPolicy


class SetAttentionExtractor(BaseFeaturesExtractor):
    """Encode player tokens with shared MLP + self-attention, return flattened tokens."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        embed_dim: int = 64,
        n_heads: int = 4,
        token_mlp_dim: int = 64,
        num_cls_tokens: int = 2,
    ):
        players_space = observation_space.spaces.get("players")
        globals_space = observation_space.spaces.get("globals")
        if players_space is None or globals_space is None:
            raise ValueError("SetAttentionExtractor requires 'players' and 'globals' in observation.")

        n_players, token_dim = players_space.shape
        global_dim = globals_space.shape[0]
        self.n_players = int(n_players)
        self.embed_dim = int(embed_dim)
        self.token_dim = int(token_dim)
        self.global_dim = int(global_dim)
        self.num_cls_tokens = int(num_cls_tokens)

        total_tokens = self.n_players + self.num_cls_tokens
        super().__init__(observation_space, features_dim=total_tokens * self.embed_dim)

        self.token_mlp = nn.Sequential(
            nn.Linear(self.token_dim + self.global_dim, token_mlp_dim),
            nn.ReLU(),
            nn.Linear(token_mlp_dim, self.embed_dim),
        )
        self.attn = nn.MultiheadAttention(self.embed_dim, n_heads, batch_first=True)
        self.attn_norm = nn.LayerNorm(self.embed_dim)
        if self.num_cls_tokens > 0:
            self.cls_tokens = nn.Parameter(th.zeros(self.num_cls_tokens, self.embed_dim))
        else:
            self.cls_tokens = None

    def forward(self, obs: Dict[str, th.Tensor]) -> th.Tensor:
        players = obs["players"]
        globals_vec = obs["globals"]
        g = globals_vec.unsqueeze(1).expand(-1, players.size(1), -1)
        tokens = th.cat([players, g], dim=-1)
        emb = self.token_mlp(tokens)
        if self.cls_tokens is not None:
            batch = emb.size(0)
            cls = self.cls_tokens.unsqueeze(0).expand(batch, -1, -1)
            emb = th.cat([emb, cls], dim=1)
        attn_out, _ = self.attn(emb, emb, emb, need_weights=False)
        attn_out = self.attn_norm(emb + attn_out)
        return attn_out.reshape(attn_out.size(0), -1)


class SetAttentionDualCriticPolicy(DualCriticActorCriticPolicy):
    """Dual-critic policy using set attention tokens for action/value heads."""

    def __init__(
        self,
        *args,
        embed_dim: int = 64,
        n_heads: int = 4,
        token_mlp_dim: int = 64,
        num_cls_tokens: int = 2,
        **kwargs,
    ):
        if "features_extractor_class" not in kwargs:
            kwargs["features_extractor_class"] = SetAttentionExtractor
        features_extractor_kwargs = dict(kwargs.get("features_extractor_kwargs") or {})
        features_extractor_kwargs.setdefault("embed_dim", embed_dim)
        features_extractor_kwargs.setdefault("n_heads", n_heads)
        features_extractor_kwargs.setdefault("token_mlp_dim", token_mlp_dim)
        features_extractor_kwargs.setdefault("num_cls_tokens", num_cls_tokens)
        kwargs["features_extractor_kwargs"] = features_extractor_kwargs
        effective_embed_dim = int(features_extractor_kwargs["embed_dim"])

        if "net_arch" not in kwargs:
            kwargs["net_arch"] = []

        super().__init__(*args, **kwargs)

        if not hasattr(self.action_space, "nvec"):
            raise ValueError("SetAttentionDualCriticPolicy requires MultiDiscrete action space.")

        obs_players = self.observation_space.spaces.get("players")
        if obs_players is None:
            raise ValueError("SetAttentionDualCriticPolicy requires 'players' in observation space.")

        self.token_players = int(obs_players.shape[0])
        self.num_cls_tokens = int(features_extractor_kwargs.get("num_cls_tokens", 0))
        self.action_players = int(len(self.action_space.nvec))
        self.actions_per_player = int(self.action_space.nvec[0])
        if any(int(n) != self.actions_per_player for n in self.action_space.nvec):
            raise ValueError("SetAttentionDualCriticPolicy requires uniform action dimensions.")

        self.pass_logit_bias: float = 0.0
        self._pass_indices = self._infer_pass_indices()

        self.embed_dim = int(effective_embed_dim)
        if self.embed_dim <= 0:
            raise ValueError("embed_dim must be positive.")

        if self.use_dual_policy:
            self.action_head_offense = nn.Linear(self.embed_dim, self.actions_per_player)
            self.action_head_defense = nn.Linear(self.embed_dim, self.actions_per_player)
        else:
            self.action_head = nn.Linear(self.embed_dim, self.actions_per_player)

        self.value_net_offense = nn.Linear(self.embed_dim, 1)
        self.value_net_defense = nn.Linear(self.embed_dim, 1)

        if self.ortho_init:
            for net, gain in [
                (self.value_net_offense, 1.0),
                (self.value_net_defense, 1.0),
            ]:
                nn.init.orthogonal_(net.weight, gain=gain)
                nn.init.constant_(net.bias, 0)
            if self.use_dual_policy:
                for net in [self.action_head_offense, self.action_head_defense]:
                    nn.init.orthogonal_(net.weight, gain=0.01)
                    nn.init.constant_(net.bias, 0)
            else:
                nn.init.orthogonal_(self.action_head.weight, gain=0.01)
                nn.init.constant_(self.action_head.bias, 0)

    def _infer_pass_indices(self) -> list[int]:
        per_dim = int(self.actions_per_player)
        if per_dim <= 0:
            return []
        start = min(8, per_dim - 1)
        end = min(per_dim - 1, 13)
        return [i for i in range(start, end + 1) if i < per_dim]

    def set_pass_logit_bias(self, value: float) -> None:
        try:
            self.pass_logit_bias = float(value)
        except Exception:
            self.pass_logit_bias = 0.0

    def _apply_pass_bias(self, logits: th.Tensor) -> th.Tensor:
        if abs(self.pass_logit_bias) <= 1e-12 or not self._pass_indices:
            return logits
        bias = th.as_tensor(
            self.pass_logit_bias, device=logits.device, dtype=logits.dtype
        )
        adjusted = logits.clone()
        for idx in self._pass_indices:
            if idx < adjusted.shape[-1]:
                adjusted[:, :, idx] = adjusted[:, :, idx] + bias
        return adjusted

    def _split_tokens(self, latent: th.Tensor) -> th.Tensor:
        batch = latent.shape[0]
        total_tokens = self.token_players + self.num_cls_tokens
        return latent.reshape(batch, total_tokens, self.embed_dim)

    def _get_action_logits(self, latent_pi: th.Tensor) -> th.Tensor:
        tokens = self._split_tokens(latent_pi)
        player_tokens = tokens[:, : self.token_players, :]
        if self.use_dual_policy and self._current_role_flags is not None:
            role_flags = self._current_role_flags.to(tokens.device)
            logits_off = self.action_head_offense(player_tokens)
            logits_def = self.action_head_defense(player_tokens)
            is_offense = (role_flags.squeeze(-1) > 0.0).unsqueeze(-1).unsqueeze(-1)
            logits = th.where(is_offense.expand_as(logits_off), logits_off, logits_def)
        else:
            logits = self.action_head(player_tokens)

        logits = self._apply_pass_bias(logits)

        if self.action_players == self.token_players:
            selected = logits
        elif self.action_players == (self.token_players // 2):
            if self._current_role_flags is None:
                selected = logits[:, : self.action_players, :]
            else:
                role_flags = self._current_role_flags.to(tokens.device)
                is_offense = (role_flags.squeeze(-1) > 0.0).unsqueeze(-1).unsqueeze(-1)
                offense_logits = logits[:, : self.action_players, :]
                defense_logits = logits[:, self.action_players :, :]
                selected = th.where(
                    is_offense.expand_as(offense_logits), offense_logits, defense_logits
                )
        else:
            raise ValueError(
                "Action space size must match all tokens or one team (players_per_side)."
            )
        return selected.reshape(selected.size(0), -1)

    def _get_value_from_latent(self, latent_vf: th.Tensor, role_flags: th.Tensor) -> th.Tensor:
        tokens = self._split_tokens(latent_vf)
        if self.num_cls_tokens < 2:
            raise ValueError("SetAttentionDualCriticPolicy requires 2 CLS tokens for critics.")
        offense_token = tokens[:, self.token_players, :]
        defense_token = tokens[:, self.token_players + 1, :]
        values_off = self.value_net_offense(offense_token)
        values_def = self.value_net_defense(defense_token)

        role_flags = role_flags.to(latent_vf.device)
        is_offense = (role_flags.squeeze(-1) > 0.0).unsqueeze(-1)
        return th.where(is_offense, values_off, values_def)
