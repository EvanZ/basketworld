from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Type

import torch as th
from torch import nn
from torch.distributions import Categorical
from gymnasium import spaces
from stable_baselines3.common.distributions import Distribution
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
      3b) Optional intent embedding modulation:
         intent embedding -> projected to global dim and added to globals
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
        intent_embedding_enabled: bool = False,
        intent_embedding_dim: int = 16,
        num_intents: int = 8,
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
        self.intent_embedding_enabled = bool(intent_embedding_enabled)
        self.intent_embedding_dim = max(1, int(intent_embedding_dim))
        self.num_intents = max(1, int(num_intents))
        # Intent globals are currently appended as trailing [index_norm, active, visible].
        self._intent_globals_dim = 3
        self._intent_start_idx = max(0, self.global_dim - self._intent_globals_dim)
        self._has_intent_globals = self.global_dim >= (4 + self._intent_globals_dim)

        total_tokens = self.n_players + self.num_cls_tokens
        super().__init__(observation_space, features_dim=total_tokens * self.embed_dim)

        token_act = _resolve_activation(token_activation, "relu")
        self.token_mlp = nn.Sequential(
            nn.Linear(self.token_dim + self.global_dim, token_mlp_dim),
            token_act(),
            nn.Linear(token_mlp_dim, self.embed_dim),
        )
        if self.intent_embedding_enabled and self._has_intent_globals:
            self.intent_embedding = nn.Embedding(
                num_embeddings=self.num_intents,
                embedding_dim=self.intent_embedding_dim,
            )
            self.intent_to_global = nn.Linear(
                self.intent_embedding_dim, self.global_dim, bias=False
            )
            nn.init.normal_(self.intent_embedding.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.intent_to_global.weight)
        else:
            self.intent_embedding = None
            self.intent_to_global = None
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
        if self.intent_embedding is not None and self.intent_to_global is not None:
            try:
                start = int(self._intent_start_idx)
                idx_norm = globals_vec[:, start]
                active = globals_vec[:, start + 1]
                visible = globals_vec[:, start + 2]
                if self.num_intents > 1:
                    intent_idx = th.round(
                        idx_norm.clamp(0.0, 1.0) * float(self.num_intents - 1)
                    ).long()
                else:
                    intent_idx = th.zeros_like(idx_norm, dtype=th.long)
                intent_idx = intent_idx.clamp(min=0, max=self.num_intents - 1)
                intent_emb = self.intent_embedding(intent_idx)
                # Do not leak hidden intent through embedding modulation.
                intent_gate = (active * visible).unsqueeze(-1)
                intent_delta = self.intent_to_global(intent_emb * intent_gate)
                globals_vec = globals_vec + intent_delta
            except Exception:
                pass
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


class PointerTargetedMultiCategoricalDistribution(Distribution):
    """Factorized action distribution for pointer-targeted passing.

    This models each player's action as:
      1) `action_type` over non-pass actions + PASS
      2) `pass_target` over pass slots, conditioned on action_type=PASS

    It returns sampled actions in the original discrete action space (e.g. 0..13),
    so SB3 rollout buffers and env stepping stay unchanged.
    """

    def __init__(
        self,
        action_dim: int,
        non_pass_action_indices: List[int],
        pass_action_indices: List[int],
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.non_pass_action_indices = [int(i) for i in non_pass_action_indices]
        self.pass_action_indices = [int(i) for i in pass_action_indices]
        self.pass_type_index = len(self.non_pass_action_indices)

        if self.action_dim <= 0:
            raise ValueError("action_dim must be positive.")
        if len(self.non_pass_action_indices) == 0:
            raise ValueError("Pointer distribution requires at least one non-pass action.")
        if len(self.pass_action_indices) == 0:
            raise ValueError("Pointer distribution requires at least one pass action slot.")

        self._action_type_distribution: Optional[Categorical] = None
        self._pass_target_distribution: Optional[Categorical] = None

        # Lookup tensors initialized on first proba_distribution() call.
        self._non_pass_action_tensor: Optional[th.Tensor] = None
        self._pass_action_tensor: Optional[th.Tensor] = None
        self._action_to_type: Optional[th.Tensor] = None
        self._action_to_pass_slot: Optional[th.Tensor] = None
        self._action_is_pass: Optional[th.Tensor] = None

    def _ensure_lookup_tensors(self, device: th.device) -> None:
        if self._action_to_type is not None and self._action_to_type.device == device:
            return

        non_pass = th.as_tensor(
            self.non_pass_action_indices,
            dtype=th.long,
            device=device,
        )
        pass_actions = th.as_tensor(
            self.pass_action_indices,
            dtype=th.long,
            device=device,
        )
        action_to_type = th.full(
            (self.action_dim,),
            fill_value=-1,
            dtype=th.long,
            device=device,
        )
        action_to_pass_slot = th.zeros((self.action_dim,), dtype=th.long, device=device)
        action_is_pass = th.zeros((self.action_dim,), dtype=th.bool, device=device)

        for type_idx, action_idx in enumerate(self.non_pass_action_indices):
            if 0 <= action_idx < self.action_dim:
                action_to_type[action_idx] = int(type_idx)
        for slot_idx, action_idx in enumerate(self.pass_action_indices):
            if 0 <= action_idx < self.action_dim:
                action_to_type[action_idx] = int(self.pass_type_index)
                action_to_pass_slot[action_idx] = int(slot_idx)
                action_is_pass[action_idx] = True

        self._non_pass_action_tensor = non_pass
        self._pass_action_tensor = pass_actions
        self._action_to_type = action_to_type
        self._action_to_pass_slot = action_to_pass_slot
        self._action_is_pass = action_is_pass

    def _check_ready(self) -> None:
        if self._action_type_distribution is None or self._pass_target_distribution is None:
            raise RuntimeError("Pointer distribution parameters are not initialized.")

    def proba_distribution_net(self, *args, **kwargs):
        raise NotImplementedError(
            "PointerTargetedMultiCategoricalDistribution does not expose a single logits net."
        )

    def proba_distribution(
        self,
        action_type_logits: th.Tensor,
        pass_target_logits: th.Tensor,
    ) -> "PointerTargetedMultiCategoricalDistribution":
        if action_type_logits.ndim != 3:
            raise ValueError("Expected action_type_logits shape (B, P, T).")
        if pass_target_logits.ndim != 3:
            raise ValueError("Expected pass_target_logits shape (B, P, S).")
        if action_type_logits.shape[:2] != pass_target_logits.shape[:2]:
            raise ValueError("action_type_logits and pass_target_logits must share (B, P).")
        if action_type_logits.shape[-1] != (self.pass_type_index + 1):
            raise ValueError("Unexpected action_type logits dimension.")
        if pass_target_logits.shape[-1] != len(self.pass_action_indices):
            raise ValueError("Unexpected pass_target logits dimension.")

        self._ensure_lookup_tensors(action_type_logits.device)
        self._action_type_distribution = Categorical(logits=action_type_logits)
        self._pass_target_distribution = Categorical(logits=pass_target_logits)

        action_probs = self.action_probabilities()
        # Keep compatibility with existing code that expects a list of per-player Categoricals.
        self.distribution = [
            Categorical(probs=action_probs[:, dim_idx, :])
            for dim_idx in range(action_probs.shape[1])
        ]
        return self

    def action_probabilities(self) -> th.Tensor:
        self._check_ready()
        assert self._action_type_distribution is not None
        assert self._pass_target_distribution is not None
        assert self._non_pass_action_tensor is not None
        assert self._pass_action_tensor is not None

        type_probs = self._action_type_distribution.probs
        pass_probs = self._pass_target_distribution.probs
        out = type_probs.new_zeros(*type_probs.shape[:-1], self.action_dim)

        out[..., self._non_pass_action_tensor] = type_probs[..., : self.pass_type_index]
        pass_mass = type_probs[..., self.pass_type_index : self.pass_type_index + 1] * pass_probs
        out[..., self._pass_action_tensor] = pass_mass
        return out

    def _map_actions(self, actions: th.Tensor) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        self._check_ready()
        assert self._action_to_type is not None
        assert self._action_to_pass_slot is not None
        assert self._action_is_pass is not None

        if actions.ndim == 1:
            actions = actions.unsqueeze(0)
        actions = actions.long()
        safe_actions = actions.clamp(min=0, max=self.action_dim - 1)

        type_actions = self._action_to_type[safe_actions]
        pass_slots = self._action_to_pass_slot[safe_actions]
        is_pass = self._action_is_pass[safe_actions]
        valid = type_actions >= 0
        safe_type_actions = th.where(valid, type_actions, th.zeros_like(type_actions))
        return safe_type_actions, pass_slots, is_pass, valid

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        self._check_ready()
        assert self._action_type_distribution is not None
        assert self._pass_target_distribution is not None

        type_actions, pass_slots, is_pass, valid = self._map_actions(actions)
        log_prob_type = self._action_type_distribution.log_prob(type_actions)
        log_prob_pass = self._pass_target_distribution.log_prob(pass_slots)
        total = log_prob_type + th.where(is_pass, log_prob_pass, th.zeros_like(log_prob_pass))
        invalid_penalty = th.full_like(total, -1e9)
        total = th.where(valid, total, invalid_penalty)
        return total.sum(dim=1)

    def entropy(self) -> th.Tensor:
        self._check_ready()
        assert self._action_type_distribution is not None
        assert self._pass_target_distribution is not None
        type_entropy = self._action_type_distribution.entropy()
        pass_entropy = self._pass_target_distribution.entropy()
        pass_prob = self._action_type_distribution.probs[..., self.pass_type_index]
        return (type_entropy + pass_prob * pass_entropy).sum(dim=1)

    def _compose_actions(self, type_choice: th.Tensor, pass_choice: th.Tensor) -> th.Tensor:
        assert self._non_pass_action_tensor is not None
        assert self._pass_action_tensor is not None

        non_pass_idx = type_choice.clamp(max=self.pass_type_index - 1)
        non_pass_actions = self._non_pass_action_tensor[non_pass_idx]
        pass_actions = self._pass_action_tensor[pass_choice]
        is_pass = type_choice == self.pass_type_index
        return th.where(is_pass, pass_actions, non_pass_actions)

    def sample(self) -> th.Tensor:
        self._check_ready()
        assert self._action_type_distribution is not None
        assert self._pass_target_distribution is not None
        type_choice = self._action_type_distribution.sample()
        pass_choice = self._pass_target_distribution.sample()
        return self._compose_actions(type_choice, pass_choice)

    def mode(self) -> th.Tensor:
        # Deterministic action must maximize the FINAL action probabilities.
        # In the factorized parameterization, argmax(action_type) + argmax(slot)
        # can be wrong because pass action mass is split across slots:
        #   P(PASS->k) = P(type=PASS) * P(slot=k)
        # so the joint per-action argmax is not always the hierarchical argmax.
        probs = self.action_probabilities()
        return th.argmax(probs, dim=-1)

    def actions_from_params(
        self,
        action_type_logits: th.Tensor,
        pass_target_logits: th.Tensor,
        deterministic: bool = False,
    ) -> th.Tensor:
        self.proba_distribution(action_type_logits=action_type_logits, pass_target_logits=pass_target_logits)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(
        self,
        action_type_logits: th.Tensor,
        pass_target_logits: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor]:
        actions = self.actions_from_params(
            action_type_logits=action_type_logits,
            pass_target_logits=pass_target_logits,
            deterministic=False,
        )
        return actions, self.log_prob(actions)


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
        intent_embedding_enabled: bool = False,
        intent_embedding_dim: int = 16,
        num_intents: int = 8,
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
        features_extractor_kwargs.setdefault(
            "intent_embedding_enabled", intent_embedding_enabled
        )
        features_extractor_kwargs.setdefault(
            "intent_embedding_dim", intent_embedding_dim
        )
        features_extractor_kwargs.setdefault("num_intents", num_intents)
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
        self.pass_mode: str = "directional"
        self._pass_indices = self._infer_pass_indices()
        pass_index_set = set(self._pass_indices)
        self._non_pass_action_indices = [
            i for i in range(self.actions_per_player) if i not in pass_index_set
        ]
        self._pointer_pass_indices = list(self._pass_indices)
        self._pointer_pass_type_index = len(self._non_pass_action_indices)
        self._pointer_action_type_dim = self._pointer_pass_type_index + 1
        self._pointer_neg_inf = -1e9

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
            self.pointer_action_type_head_offense = nn.Linear(
                self.pi_embed_dim, self._pointer_action_type_dim
            )
            self.pointer_action_type_head_defense = nn.Linear(
                self.pi_embed_dim, self._pointer_action_type_dim
            )
            self.pointer_query_head_offense = nn.Linear(
                self.pi_embed_dim, self.pi_embed_dim, bias=False
            )
            self.pointer_query_head_defense = nn.Linear(
                self.pi_embed_dim, self.pi_embed_dim, bias=False
            )
            self.pointer_key_head_offense = nn.Linear(
                self.pi_embed_dim, self.pi_embed_dim, bias=False
            )
            self.pointer_key_head_defense = nn.Linear(
                self.pi_embed_dim, self.pi_embed_dim, bias=False
            )
        else:
            self.action_head = nn.Linear(self.pi_embed_dim, self.actions_per_player)
            self.pointer_action_type_head = nn.Linear(
                self.pi_embed_dim, self._pointer_action_type_dim
            )
            self.pointer_query_head = nn.Linear(self.pi_embed_dim, self.pi_embed_dim, bias=False)
            self.pointer_key_head = nn.Linear(self.pi_embed_dim, self.pi_embed_dim, bias=False)

        self.value_net_offense = nn.Linear(self.vf_embed_dim, 1)
        self.value_net_defense = nn.Linear(self.vf_embed_dim, 1)

        pointer_slot_table = self._build_pointer_slot_target_ids()
        self.register_buffer("pointer_slot_target_ids", pointer_slot_table, persistent=False)

        if self.ortho_init:
            for net, gain in [
                (self.value_net_offense, 1.0),
                (self.value_net_defense, 1.0),
            ]:
                nn.init.orthogonal_(net.weight, gain=gain)
                nn.init.constant_(net.bias, 0)
            if self.use_dual_policy:
                for net in [
                    self.action_head_offense,
                    self.action_head_defense,
                    self.pointer_action_type_head_offense,
                    self.pointer_action_type_head_defense,
                ]:
                    nn.init.orthogonal_(net.weight, gain=0.01)
                    nn.init.constant_(net.bias, 0)
                for net in [
                    self.pointer_query_head_offense,
                    self.pointer_query_head_defense,
                    self.pointer_key_head_offense,
                    self.pointer_key_head_defense,
                ]:
                    nn.init.orthogonal_(net.weight, gain=0.01)
            else:
                nn.init.orthogonal_(self.action_head.weight, gain=0.01)
                nn.init.constant_(self.action_head.bias, 0)
                nn.init.orthogonal_(self.pointer_action_type_head.weight, gain=0.01)
                nn.init.constant_(self.pointer_action_type_head.bias, 0)
                nn.init.orthogonal_(self.pointer_query_head.weight, gain=0.01)
                nn.init.orthogonal_(self.pointer_key_head.weight, gain=0.01)

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

    def set_pass_mode(self, mode: str) -> None:
        normalized = str(mode).lower()
        if normalized in ("directional", "pointer_targeted"):
            self.pass_mode = normalized

    def _use_pointer_factorization(self) -> bool:
        return (
            str(getattr(self, "pass_mode", "directional")).lower() == "pointer_targeted"
            and len(self._pointer_pass_indices) > 0
            and len(self._non_pass_action_indices) > 0
        )

    def _build_pointer_slot_target_ids(self) -> th.Tensor:
        """Build [player_id, pass_slot] -> teammate_id lookup (-1 for invalid slots)."""
        slot_count = max(1, len(self._pointer_pass_indices))
        table = th.full(
            (self.token_players, slot_count),
            fill_value=-1,
            dtype=th.long,
        )
        half = self.token_players // 2
        for pid in range(self.token_players):
            if self.token_players % 2 == 0:
                team_ids = list(range(0, half)) if pid < half else list(range(half, self.token_players))
            else:
                team_ids = list(range(self.token_players))
            teammates = [tid for tid in team_ids if tid != pid][:slot_count]
            for slot_idx, tid in enumerate(teammates):
                table[pid, slot_idx] = int(tid)
        return table

    def _extract_pi_player_tokens(self, latent_pi: th.Tensor) -> th.Tensor:
        tokens = self._split_tokens(latent_pi)
        player_tokens = tokens[:, : self.token_players, :]
        if self.token_head_mlp_pi is not None:
            player_tokens = self.token_head_mlp_pi(player_tokens)
        return player_tokens

    def _select_action_player_indices(self, batch_size: int, device: th.device) -> th.Tensor:
        if self.action_players == self.token_players:
            return th.arange(self.token_players, device=device).unsqueeze(0).expand(batch_size, -1)

        if self.action_players == (self.token_players // 2):
            half = self.action_players
            offense_ids = th.arange(half, device=device)
            defense_ids = th.arange(half, half * 2, device=device)
            if self._current_role_flags is None:
                return offense_ids.unsqueeze(0).expand(batch_size, -1)
            role_flags = self._current_role_flags.to(device)
            is_offense = role_flags.squeeze(-1) > 0.0
            return th.where(
                is_offense.unsqueeze(-1),
                offense_ids.unsqueeze(0).expand(batch_size, -1),
                defense_ids.unsqueeze(0).expand(batch_size, -1),
            )

        raise ValueError(
            "Action space size must match all tokens or one team (players_per_side)."
        )

    @staticmethod
    def _gather_players(values: th.Tensor, player_ids: th.Tensor) -> th.Tensor:
        gather_idx = player_ids.unsqueeze(-1).expand(-1, -1, values.size(-1))
        return th.gather(values, dim=1, index=gather_idx)

    def _get_all_directional_logits(self, player_tokens: th.Tensor) -> th.Tensor:
        if self.use_dual_policy:
            logits_off = self.action_head_offense(player_tokens)
            logits_def = self.action_head_defense(player_tokens)
            if self._current_role_flags is None:
                return logits_off
            role_flags = self._current_role_flags.to(player_tokens.device)
            is_offense = (role_flags.squeeze(-1) > 0.0).unsqueeze(-1).unsqueeze(-1)
            return th.where(is_offense, logits_off, logits_def)
        return self.action_head(player_tokens)

    def _get_pointer_qk(self, player_tokens: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        if self.use_dual_policy:
            q_off = self.pointer_query_head_offense(player_tokens)
            k_off = self.pointer_key_head_offense(player_tokens)
            q_def = self.pointer_query_head_defense(player_tokens)
            k_def = self.pointer_key_head_defense(player_tokens)
            if self._current_role_flags is None:
                return q_off, k_off
            role_flags = self._current_role_flags.to(player_tokens.device)
            is_offense = (role_flags.squeeze(-1) > 0.0).unsqueeze(-1).unsqueeze(-1)
            q = th.where(is_offense, q_off, q_def)
            k = th.where(is_offense, k_off, k_def)
            return q, k
        return self.pointer_query_head(player_tokens), self.pointer_key_head(player_tokens)

    def _get_pointer_action_type_logits(self, selected_tokens: th.Tensor) -> th.Tensor:
        if self.use_dual_policy:
            logits_off = self.pointer_action_type_head_offense(selected_tokens)
            logits_def = self.pointer_action_type_head_defense(selected_tokens)
            if self._current_role_flags is None:
                return logits_off
            role_flags = self._current_role_flags.to(selected_tokens.device)
            is_offense = (role_flags.squeeze(-1) > 0.0).unsqueeze(-1).unsqueeze(-1)
            return th.where(is_offense, logits_off, logits_def)
        return self.pointer_action_type_head(selected_tokens)

    def _get_pointer_pass_slot_logits(
        self,
        player_tokens: th.Tensor,
        selected_player_ids: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor]:
        q_all, k_all = self._get_pointer_qk(player_tokens)
        scale = math.sqrt(float(q_all.shape[-1])) if q_all.shape[-1] > 0 else 1.0
        pair_scores = th.matmul(q_all, k_all.transpose(-1, -2)) / scale

        slot_table = self.pointer_slot_target_ids.to(player_tokens.device)
        slot_target_ids = slot_table[selected_player_ids]  # (B, P, S)
        valid_slots = slot_target_ids >= 0
        safe_target_ids = slot_target_ids.clamp(min=0)

        batch_size, action_players, slot_count = slot_target_ids.shape
        batch_idx = (
            th.arange(batch_size, device=player_tokens.device)
            .view(batch_size, 1, 1)
            .expand(batch_size, action_players, slot_count)
        )
        passer_idx = selected_player_ids.unsqueeze(-1).expand(batch_size, action_players, slot_count)

        slot_logits = pair_scores[batch_idx, passer_idx, safe_target_ids]
        neg_inf = th.full_like(slot_logits, self._pointer_neg_inf)
        slot_logits = th.where(valid_slots, slot_logits, neg_inf)

        valid_any = valid_slots.any(dim=-1)  # (B, P)
        fallback = th.full_like(slot_logits, self._pointer_neg_inf)
        fallback[..., 0] = 0.0
        slot_logits = th.where(valid_any.unsqueeze(-1), slot_logits, fallback)
        return slot_logits, valid_any

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

    def _apply_pointer_pass_bias(
        self,
        action_type_logits: th.Tensor,
        valid_pass_target: th.Tensor,
    ) -> th.Tensor:
        """Apply pass bias/floor to action_type PASS logit in pointer mode."""
        adjusted = action_type_logits.clone()
        pass_idx = int(self._pointer_pass_type_index)
        pass_logit = adjusted[..., pass_idx]
        neg_inf = th.full_like(pass_logit, self._pointer_neg_inf)
        pass_logit = th.where(valid_pass_target, pass_logit, neg_inf)

        base_bias = float(self.pass_logit_bias)
        pass_prob_min = float(getattr(self, "pass_prob_min", 0.0) or 0.0)
        if abs(base_bias) > 1e-12:
            pass_logit = th.where(valid_pass_target, pass_logit + base_bias, pass_logit)

        if pass_prob_min > 0.0:
            p_min = min(max(pass_prob_min, 0.0), 1.0 - 1e-6)
            # Use the original logits tensor (not `adjusted`) to avoid
            # autograd versioning issues from later in-place updates.
            non_pass_logits = action_type_logits[..., :pass_idx]
            if non_pass_logits.shape[-1] > 0:
                log_p = math.log(p_min)
                log_1mp = math.log1p(-p_min)
                log_r = th.logsumexp(non_pass_logits, dim=-1)
                b_needed = log_p + log_r - log_1mp - pass_logit
                finite = valid_pass_target & th.isfinite(log_r) & th.isfinite(pass_logit)
                pass_logit = th.where(finite, pass_logit + b_needed.clamp(min=0.0), pass_logit)

        adjusted[..., pass_idx] = pass_logit
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
        player_tokens = self._extract_pi_player_tokens(latent_pi)
        logits_all = self._get_all_directional_logits(player_tokens)
        logits_all = self._apply_pass_bias(logits_all)
        selected_ids = self._select_action_player_indices(
            batch_size=player_tokens.shape[0],
            device=player_tokens.device,
        )
        selected = self._gather_players(logits_all, selected_ids)
        return selected.reshape(selected.size(0), -1)

    def _get_action_dist_from_latent(
        self,
        latent_pi: th.Tensor,
        latent_sde: Optional[th.Tensor] = None,
    ) -> Distribution:
        del latent_sde  # unused for this policy
        if not self._use_pointer_factorization():
            directional_logits = self._get_action_logits(latent_pi)
            return self.action_dist.proba_distribution(action_logits=directional_logits)

        player_tokens = self._extract_pi_player_tokens(latent_pi)
        selected_ids = self._select_action_player_indices(
            batch_size=player_tokens.shape[0],
            device=player_tokens.device,
        )
        selected_tokens = self._gather_players(player_tokens, selected_ids)
        pass_slot_logits, valid_any_pass_slot = self._get_pointer_pass_slot_logits(
            player_tokens,
            selected_ids,
        )
        action_type_logits = self._get_pointer_action_type_logits(selected_tokens)
        action_type_logits = self._apply_pointer_pass_bias(
            action_type_logits,
            valid_any_pass_slot,
        )

        dist = PointerTargetedMultiCategoricalDistribution(
            action_dim=self.actions_per_player,
            non_pass_action_indices=self._non_pass_action_indices,
            pass_action_indices=self._pointer_pass_indices,
        )
        return dist.proba_distribution(
            action_type_logits=action_type_logits,
            pass_target_logits=pass_slot_logits,
        )

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

    def load_state_dict(self, state_dict, strict: bool = True):  # type: ignore[override]
        """Allow backward-compatible loads that predate pointer-targeted heads."""
        if not strict:
            return super().load_state_dict(state_dict, strict=False)

        result = super().load_state_dict(state_dict, strict=False)
        allowed_prefixes = (
            "pointer_",
            "features_extractor.intent_embedding",
            "features_extractor.intent_to_global",
        )
        missing = [
            key
            for key in result.missing_keys
            if not str(key).startswith(allowed_prefixes)
        ]
        unexpected = [
            key
            for key in result.unexpected_keys
            if not str(key).startswith(allowed_prefixes)
        ]
        if missing or unexpected:
            missing_msg = ", ".join(missing) if missing else "None"
            unexpected_msg = ", ".join(unexpected) if unexpected else "None"
            raise RuntimeError(
                "Error(s) in loading state_dict for SetAttentionDualCriticPolicy: "
                f"Missing key(s): {missing_msg}; Unexpected key(s): {unexpected_msg}"
            )
        return result

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
