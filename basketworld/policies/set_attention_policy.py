from __future__ import annotations

import math
from typing import Any, Dict, Optional, Type

import torch as th
from torch import nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from basketworld.policies.dual_critic_policy import DualCriticActorCriticPolicy


def _resolve_activation(name: str, fallback: str) -> Type[nn.Module]:
    key = (name or fallback).lower()
    if key == "relu":
        return nn.ReLU
    if key == "gelu":
        return nn.GELU
    if key in {"silu", "swish"}:
        return nn.SiLU
    return nn.Tanh


class SetAttentionExtractor(BaseFeaturesExtractor):
    """Encode per-player tokens with shared MLP + self-attention.

    Inputs (from set-observation wrapper):
      - obs["players"]: float tensor of shape (B, P, T)
        B = batch size, P = number of players, T = per-player token features.
      - obs["globals"]: float tensor of shape (B, G)
        G = number of global features (shot clock, hoop coords, etc.).

    Pipeline (with shapes):
      1) Broadcast globals to each player:
         globals -> (B, 1, G) -> (B, P, G)
      2) Concatenate player+global features:
         tokens = concat(players, globals) -> (B, P, T+G)
      3) Shared token MLP:
         token_mlp: (B, P, T+G) -> (B, P, D)
         D = embed_dim
      4) Optional learned CLS tokens:
         cls_tokens: (C, D) where C = num_cls_tokens
         append -> (B, P+C, D)
      5) Multihead self-attention + residual + LayerNorm:
         attn_out -> (B, P+C, D)
      6) Flatten for SB3 compatibility:
         output -> (B, (P+C) * D)

    Note: although SB3 expects a flat feature vector, we preserve token
    structure internally in the policy by reshaping the flattened output
    back to (B, P+C, D) for action/value heads.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        embed_dim: int = 64,
        n_heads: int = 4,
        token_mlp_dim: int = 64,
        num_cls_tokens: int = 2,
        token_activation: str = "relu",
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

        token_act = _resolve_activation(token_activation, "relu")
        self.token_mlp = nn.Sequential(
            nn.Linear(self.token_dim + self.global_dim, token_mlp_dim),
            token_act(),
            nn.Linear(token_mlp_dim, self.embed_dim),
        )
        self.attn = nn.MultiheadAttention(self.embed_dim, n_heads, batch_first=True)
        self.attn_norm = nn.LayerNorm(self.embed_dim)
        if self.num_cls_tokens > 0:
            self.cls_tokens = nn.Parameter(th.zeros(self.num_cls_tokens, self.embed_dim))
        else:
            self.cls_tokens = None

    def forward(self, obs: Dict[str, th.Tensor]) -> th.Tensor:
        """Compute flattened token embeddings from a set observation.

        Args:
            obs: dict with "players" (B, P, T) and "globals" (B, G).

        Returns:
            Flattened tokens of shape (B, (P + C) * D).
            This preserves token order: first P player tokens, then C CLS tokens.
        """
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
    """Dual-critic policy with token-based attention features.

    High-level flow:
      - Extract tokens with SetAttentionExtractor:
        latent = (B, (P+C) * D)
      - Reshape tokens: (B, P+C, D)
      - Action head(s) use ONLY player tokens (P tokens).
      - Value heads read CLS tokens (C tokens):
        CLS_OFF -> offense value, CLS_DEF -> defense value.

    Shapes and roles:
      - P = number of players (from obs["players"].shape[0]).
      - C = num_cls_tokens (typically 2: offense + defense).
      - D = embed_dim.
      - Action logits per player:
        (B, P, A) where A = actions_per_player.
      - Policy output flattened for SB3:
        (B, P * A)

    net_arch handling:
      - SB3 normally expects a flat feature vector and builds an MLP.
      - We bypass that and build optional per-token MLPs ourselves:
        - net_arch=[64,64] -> shared token MLP for both pi/vf.
        - net_arch={"pi":[...],"vf":[...]} -> separate token MLPs.
    """

    def __init__(
        self,
        *args,
        embed_dim: int = 64,
        n_heads: int = 4,
        token_mlp_dim: int = 64,
        num_cls_tokens: int = 2,
        token_activation: str = "relu",
        head_activation: str = "tanh",
        **kwargs,
    ):
        head_arch = kwargs.get("net_arch")
        self._head_activation = _resolve_activation(head_activation, "tanh")
        if "features_extractor_class" not in kwargs:
            kwargs["features_extractor_class"] = SetAttentionExtractor
        features_extractor_kwargs = dict(kwargs.get("features_extractor_kwargs") or {})
        features_extractor_kwargs.setdefault("embed_dim", embed_dim)
        features_extractor_kwargs.setdefault("n_heads", n_heads)
        features_extractor_kwargs.setdefault("token_mlp_dim", token_mlp_dim)
        features_extractor_kwargs.setdefault("num_cls_tokens", num_cls_tokens)
        features_extractor_kwargs.setdefault("token_activation", token_activation)
        kwargs["features_extractor_kwargs"] = features_extractor_kwargs
        effective_embed_dim = int(features_extractor_kwargs["embed_dim"])

        if "net_arch" not in kwargs:
            kwargs["net_arch"] = []
        else:
            # MLP extractor should not reshape token features; apply head MLPs manually instead.
            kwargs["net_arch"] = []

        super().__init__(*args, **kwargs)
        # Preserve the requested head architecture for logging/debugging.
        self.net_arch = head_arch

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
        self.pass_prob_min: float = 0.0
        self._pass_indices = self._infer_pass_indices()

        self.embed_dim = int(effective_embed_dim)
        if self.embed_dim <= 0:
            raise ValueError("embed_dim must be positive.")

        self.token_head_mlp_pi = None
        self.token_head_mlp_vf = None
        self.pi_embed_dim = self.embed_dim
        self.vf_embed_dim = self.embed_dim

        head_arch = self._normalize_head_arch(head_arch)
        if head_arch:
            if "shared" in head_arch:
                shared_mlp, shared_dim = self._build_token_head_mlp(
                    self.embed_dim, head_arch["shared"]
                )
                self.token_head_mlp_pi = shared_mlp
                self.token_head_mlp_vf = shared_mlp
                self.pi_embed_dim = shared_dim
                self.vf_embed_dim = shared_dim
            else:
                if "pi" in head_arch:
                    self.token_head_mlp_pi, self.pi_embed_dim = self._build_token_head_mlp(
                        self.embed_dim, head_arch["pi"]
                    )
                if "vf" in head_arch:
                    self.token_head_mlp_vf, self.vf_embed_dim = self._build_token_head_mlp(
                        self.embed_dim, head_arch["vf"]
                    )

        if self.use_dual_policy:
            self.action_head_offense = nn.Linear(self.pi_embed_dim, self.actions_per_player)
            self.action_head_defense = nn.Linear(self.pi_embed_dim, self.actions_per_player)
        else:
            self.action_head = nn.Linear(self.pi_embed_dim, self.actions_per_player)

        self.value_net_offense = nn.Linear(self.vf_embed_dim, 1)
        self.value_net_defense = nn.Linear(self.vf_embed_dim, 1)

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

        # Rebuild optimizer so newly created heads are trainable.
        try:
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.optimizer = self.optimizer_class(
                self.parameters(), lr=current_lr, **self.optimizer_kwargs
            )
            opt_params = {
                id(p): p
                for group in self.optimizer.param_groups
                for p in group.get("params", [])
            }
            opt_param_total = sum(p.numel() for p in opt_params.values())
            trainable_total = sum(
                p.numel() for p in self.parameters() if p.requires_grad
            )
            print(
                "[SetAttention] Optimizer params:",
                opt_param_total,
                "(trainable:",
                trainable_total,
                ")",
            )
        except Exception:
            pass

    def _infer_pass_indices(self) -> list[int]:
        """Infer which action indices correspond to pass actions (per player)."""
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

    def set_pass_prob_min(self, value: float) -> None:
        try:
            self.pass_prob_min = float(value)
        except Exception:
            self.pass_prob_min = 0.0

    def _apply_pass_bias(self, logits: th.Tensor) -> th.Tensor:
        base_bias = float(self.pass_logit_bias)
        pass_prob_min = float(getattr(self, "pass_prob_min", 0.0) or 0.0)
        if (abs(base_bias) <= 1e-12 and pass_prob_min <= 0.0) or not self._pass_indices:
            return logits
        pass_idx = [idx for idx in self._pass_indices if idx < logits.shape[-1]]
        if not pass_idx:
            return logits
        adjusted = logits.clone()
        base = th.full(
            adjusted.shape[:-1], base_bias, device=logits.device, dtype=logits.dtype
        )
        total_bias = base
        if pass_prob_min > 0.0 and len(pass_idx) < logits.shape[-1]:
            p_min = min(max(float(pass_prob_min), 0.0), 1.0 - 1e-6)
            pass_logits = logits[..., pass_idx]
            non_pass_mask = th.ones(logits.shape[-1], dtype=th.bool, device=logits.device)
            non_pass_mask[pass_idx] = False
            non_pass_logits = logits[..., non_pass_mask]
            if non_pass_logits.shape[-1] > 0:
                log_p = math.log(p_min)
                log_1mp = math.log1p(-p_min)
                log_s = th.logsumexp(pass_logits, dim=-1)
                log_r = th.logsumexp(non_pass_logits, dim=-1)
                b_needed = log_p + log_r - log_1mp - log_s
                total_bias = th.maximum(base, b_needed)
                valid = th.isfinite(log_s) & th.isfinite(log_r)
                total_bias = th.where(valid, total_bias, base)
        adjusted[..., pass_idx] = adjusted[..., pass_idx] + total_bias.unsqueeze(-1)
        return adjusted

    def _split_tokens(self, latent: th.Tensor) -> th.Tensor:
        """Reshape flat latent to (B, P+C, D).

        Args:
            latent: (B, (P+C) * D) from SetAttentionExtractor.
        """
        batch = latent.shape[0]
        total_tokens = self.token_players + self.num_cls_tokens
        return latent.reshape(batch, total_tokens, self.embed_dim)

    def _get_action_logits(self, latent_pi: th.Tensor) -> th.Tensor:
        """Compute per-player action logits from token embeddings.

        Inputs:
            latent_pi: (B, (P+C) * D)
        Steps:
            1) Reshape -> tokens: (B, P+C, D)
            2) Slice player tokens: (B, P, D)
            3) Optional token MLP (pi) -> (B, P, D')
            4) Action head(s) -> logits: (B, P, A)
            5) If action space is per-team, select the correct team slice.

        Returns:
            Flattened logits (B, P*A) or (B, (P/2)*A) depending on action space.
        """
        tokens = self._split_tokens(latent_pi)
        player_tokens = tokens[:, : self.token_players, :]
        if self.token_head_mlp_pi is not None:
            player_tokens = self.token_head_mlp_pi(player_tokens)
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
        """Compute value estimate from CLS tokens.

        Inputs:
            latent_vf: (B, (P+C) * D)
            role_flags: (B, 1) or (B, P, 1) with +1 for offense, -1 for defense.

        Steps:
            1) Reshape -> tokens: (B, P+C, D)
            2) Optional token MLP (vf) -> (B, P+C, D')
            3) Select CLS tokens:
               - offense_token = tokens[:, P, :]
               - defense_token = tokens[:, P+1, :]
            4) Apply value heads -> (B, 1) for each
            5) Select offense/defense value based on role_flags.
        """
        tokens = self._split_tokens(latent_vf)
        if self.token_head_mlp_vf is not None:
            tokens = self.token_head_mlp_vf(tokens)
        if self.num_cls_tokens < 2:
            raise ValueError("SetAttentionDualCriticPolicy requires 2 CLS tokens for critics.")
        offense_token = tokens[:, self.token_players, :]
        defense_token = tokens[:, self.token_players + 1, :]
        values_off = self.value_net_offense(offense_token)
        values_def = self.value_net_defense(defense_token)

        role_flags = role_flags.to(latent_vf.device)
        is_offense = (role_flags.squeeze(-1) > 0.0).unsqueeze(-1)
        return th.where(is_offense, values_off, values_def)

    @staticmethod
    def _normalize_head_arch(net_arch: Optional[Any]) -> Optional[Dict[str, list[int]]]:
        if net_arch is None:
            return None
        if isinstance(net_arch, list) and net_arch and isinstance(net_arch[0], dict):
            net_arch = net_arch[0]
        if isinstance(net_arch, dict):
            pi_arch = list(net_arch.get("pi", []) or [])
            vf_arch = list(net_arch.get("vf", []) or [])
            return {"pi": pi_arch, "vf": vf_arch}
        if isinstance(net_arch, (list, tuple)):
            return {"shared": list(net_arch)}
        return None

    def _build_token_head_mlp(self, input_dim: int, layers: list[int]):
        """Build a per-token MLP.

        This MLP is applied to each token independently (shared weights across tokens).
        It does not mix information across tokens; attention already handles that.
        """
        if not layers:
            return None, input_dim
        modules = []
        last_dim = int(input_dim)
        for size in layers:
            modules.append(nn.Linear(last_dim, int(size)))
            modules.append(self._head_activation())
            last_dim = int(size)
        return nn.Sequential(*modules), last_dim
