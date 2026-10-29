from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


NOOP_ACTION_INDEX = 0
MASKED_LOGIT_FLOOR = -1.0e9
PASS_ACTION_START = 8
PASS_ACTION_END = 14
ACTION_HEAD_MODE_FLAT = "flat"
ACTION_HEAD_MODE_POINTER_TARGETED = "pointer_targeted"
TEAM_ACTION_MODE_INDEPENDENT = "independent"
TEAM_ACTION_MODE_AUTOREGRESSIVE = "autoregressive"
TEAM_ACTION_ORDER_RANDOM = "random"
TEAM_ACTION_ORDER_BALL_HANDLER_LAST = "ball_handler_last"
TEAM_ACTION_ORDER_CANONICAL = "canonical"


@dataclass(frozen=True)
class ActorCriticSpec:
    flat_obs_dim: int
    training_player_count: int
    action_dim_per_player: int
    total_action_dim: int
    hidden_dims: tuple[int, ...]
    model_type: str = "mlp"
    token_player_count: int = 0
    token_dim: int = 0
    global_dim: int = 0
    attention_embed_dim: int = 64
    attention_num_heads: int = 4
    attention_num_layers: int = 1
    attention_token_mlp_dim: int = 64
    attention_num_cls_tokens: int = 2
    attention_pi_head_hidden_dims: tuple[int, ...] = ()
    attention_vf_head_hidden_dims: tuple[int, ...] = ()
    attention_head_activation: str = "tanh"
    action_head_mode: str = ACTION_HEAD_MODE_FLAT
    team_action_mode: str = TEAM_ACTION_MODE_INDEPENDENT
    offense_action_order: str = TEAM_ACTION_ORDER_RANDOM
    defense_action_order: str = TEAM_ACTION_ORDER_RANDOM
    pass_action_start: int = PASS_ACTION_START
    pass_action_end: int = PASS_ACTION_END
    intent_embedding_enabled: bool = False
    intent_embedding_dim: int = 16
    intent_conditioning_scale: float = 1.0
    num_intents: int = 8
    intent_selector_enabled: bool = False
    intent_selector_hidden_dim: int = 64
    rebound_win_prob_features: bool = False
    rebound_target_observation_features: bool = True
    rebound_critic_enabled: bool = False
    multi_possession_features: bool = False
    observation_schema_version: int = 1


def build_actor_critic_spec(
    flat_obs_batch: np.ndarray,
    action_mask_batch: np.ndarray,
    hidden_dims: Sequence[int],
    *,
    model_type: str = "mlp",
    token_player_count: int = 0,
    token_dim: int = 0,
    global_dim: int = 0,
    attention_embed_dim: int = 64,
    attention_num_heads: int = 4,
    attention_num_layers: int = 1,
    attention_token_mlp_dim: int = 64,
    attention_num_cls_tokens: int = 2,
    attention_pi_head_hidden_dims: Sequence[int] = (),
    attention_vf_head_hidden_dims: Sequence[int] = (),
    attention_head_activation: str = "tanh",
    action_head_mode: str = ACTION_HEAD_MODE_FLAT,
    team_action_mode: str = TEAM_ACTION_MODE_INDEPENDENT,
    offense_action_order: str = TEAM_ACTION_ORDER_RANDOM,
    defense_action_order: str = TEAM_ACTION_ORDER_RANDOM,
    pass_action_start: int = PASS_ACTION_START,
    pass_action_end: int = PASS_ACTION_END,
    intent_embedding_enabled: bool = False,
    intent_embedding_dim: int = 16,
    intent_conditioning_scale: float = 1.0,
    num_intents: int = 8,
    intent_selector_enabled: bool = False,
    intent_selector_hidden_dim: int = 64,
    rebound_win_prob_features: bool = False,
    rebound_target_observation_features: bool = True,
    rebound_critic_enabled: bool = False,
    multi_possession_features: bool = False,
    observation_schema_version: int = 1,
) -> ActorCriticSpec:
    if flat_obs_batch.ndim != 2:
        raise ValueError(
            f"Expected flat_obs batch shape (batch, dim), got {tuple(flat_obs_batch.shape)}."
        )
    if action_mask_batch.ndim != 3:
        raise ValueError(
            "Expected action_mask batch shape (batch, players, actions), got "
            f"{tuple(action_mask_batch.shape)}."
        )
    training_player_count = int(action_mask_batch.shape[1])
    action_dim_per_player = int(action_mask_batch.shape[2])
    normalized_model_type = str(model_type).lower()
    normalized_action_head_mode = str(action_head_mode).lower()
    normalized_team_action_mode = str(team_action_mode).lower()
    normalized_offense_action_order = str(offense_action_order).lower()
    normalized_defense_action_order = str(defense_action_order).lower()
    if normalized_model_type not in {"mlp", "attention"}:
        raise ValueError(f"Unsupported JAX policy model type {model_type!r}.")
    if normalized_action_head_mode not in {ACTION_HEAD_MODE_FLAT, ACTION_HEAD_MODE_POINTER_TARGETED}:
        raise ValueError(f"Unsupported JAX action head mode {action_head_mode!r}.")
    if normalized_action_head_mode == ACTION_HEAD_MODE_POINTER_TARGETED and normalized_model_type != "attention":
        raise ValueError("Pointer-targeted JAX action head currently requires --policy-model attention.")
    if normalized_team_action_mode not in {
        TEAM_ACTION_MODE_INDEPENDENT,
        TEAM_ACTION_MODE_AUTOREGRESSIVE,
    }:
        raise ValueError(f"Unsupported JAX team action mode {team_action_mode!r}.")
    valid_action_orders = {
        TEAM_ACTION_ORDER_RANDOM,
        TEAM_ACTION_ORDER_BALL_HANDLER_LAST,
        TEAM_ACTION_ORDER_CANONICAL,
    }
    if normalized_offense_action_order not in valid_action_orders:
        raise ValueError(f"Unsupported JAX offense action order {offense_action_order!r}.")
    if normalized_defense_action_order not in {
        TEAM_ACTION_ORDER_RANDOM,
        TEAM_ACTION_ORDER_CANONICAL,
    }:
        raise ValueError(f"Unsupported JAX defense action order {defense_action_order!r}.")
    if normalized_team_action_mode == TEAM_ACTION_MODE_AUTOREGRESSIVE and normalized_model_type != "attention":
        raise ValueError("Autoregressive JAX team actions require --policy-model attention.")
    intent_embedding_enabled = bool(intent_embedding_enabled)
    intent_embedding_dim = int(intent_embedding_dim)
    intent_conditioning_scale = float(intent_conditioning_scale)
    num_intents = int(num_intents)
    intent_selector_enabled = bool(intent_selector_enabled)
    intent_selector_hidden_dim = int(intent_selector_hidden_dim)
    if intent_embedding_enabled and normalized_model_type != "attention":
        raise ValueError("JAX intent embeddings require --policy-model attention.")
    if intent_selector_enabled and normalized_model_type != "attention":
        raise ValueError("JAX intent selector requires --policy-model attention.")
    if intent_embedding_enabled and intent_embedding_dim <= 0:
        raise ValueError("--intent-embedding-dim must be >= 1.")
    if intent_conditioning_scale < 0.0:
        raise ValueError("--intent-conditioning-scale must be >= 0.")
    if intent_selector_enabled and intent_selector_hidden_dim <= 0:
        raise ValueError("--intent-selector-hidden-dim must be >= 1.")
    if num_intents <= 0:
        raise ValueError("--num-intents must be >= 1.")
    if not intent_embedding_enabled and not intent_selector_enabled:
        num_intents = 8
    pass_action_start = int(pass_action_start)
    pass_action_end = int(pass_action_end)
    if not (0 <= pass_action_start < pass_action_end <= action_dim_per_player):
        raise ValueError(
            "Invalid pass action range for JAX actor-critic: "
            f"start={pass_action_start}, end={pass_action_end}, action_dim={action_dim_per_player}."
        )
    if normalized_model_type == "attention":
        token_player_count = int(token_player_count)
        token_dim = int(token_dim)
        global_dim = int(global_dim)
        attention_embed_dim = int(attention_embed_dim)
        attention_num_heads = int(attention_num_heads)
        attention_num_layers = int(attention_num_layers)
        if token_player_count <= 0 or token_dim <= 0 or global_dim <= 0:
            raise ValueError("Attention policy requires positive token/player/global dimensions.")
        expected_dim = (token_player_count * token_dim) + global_dim + 1
        if int(flat_obs_batch.shape[1]) != expected_dim:
            raise ValueError(
                f"Attention policy expected packed obs dim {expected_dim}, got {int(flat_obs_batch.shape[1])}."
            )
        if attention_embed_dim % attention_num_heads != 0:
            raise ValueError("--attention-embed-dim must be divisible by --attention-num-heads.")
        if attention_num_layers < 1:
            raise ValueError("--attention-num-layers must be >= 1.")
    return ActorCriticSpec(
        flat_obs_dim=int(flat_obs_batch.shape[1]),
        training_player_count=training_player_count,
        action_dim_per_player=action_dim_per_player,
        total_action_dim=training_player_count * action_dim_per_player,
        hidden_dims=tuple(int(v) for v in hidden_dims),
        model_type=normalized_model_type,
        token_player_count=int(token_player_count),
        token_dim=int(token_dim),
        global_dim=int(global_dim),
        attention_embed_dim=int(attention_embed_dim),
        attention_num_heads=int(attention_num_heads),
        attention_num_layers=int(attention_num_layers),
        attention_token_mlp_dim=int(attention_token_mlp_dim),
        attention_num_cls_tokens=int(attention_num_cls_tokens),
        attention_pi_head_hidden_dims=tuple(int(v) for v in attention_pi_head_hidden_dims),
        attention_vf_head_hidden_dims=tuple(int(v) for v in attention_vf_head_hidden_dims),
        attention_head_activation=str(attention_head_activation).lower(),
        action_head_mode=normalized_action_head_mode,
        team_action_mode=normalized_team_action_mode,
        offense_action_order=normalized_offense_action_order,
        defense_action_order=normalized_defense_action_order,
        pass_action_start=int(pass_action_start),
        pass_action_end=int(pass_action_end),
        intent_embedding_enabled=bool(intent_embedding_enabled),
        intent_embedding_dim=int(intent_embedding_dim),
        intent_conditioning_scale=float(intent_conditioning_scale),
        num_intents=int(num_intents),
        intent_selector_enabled=bool(intent_selector_enabled),
        intent_selector_hidden_dim=int(intent_selector_hidden_dim),
        rebound_win_prob_features=bool(rebound_win_prob_features),
        rebound_target_observation_features=bool(rebound_target_observation_features),
        rebound_critic_enabled=bool(rebound_critic_enabled),
        multi_possession_features=bool(multi_possession_features),
        observation_schema_version=int(observation_schema_version),
    )


def _non_pass_action_indices(spec: ActorCriticSpec) -> tuple[int, ...]:
    return tuple(
        action_idx
        for action_idx in range(int(spec.action_dim_per_player))
        if not (int(spec.pass_action_start) <= action_idx < int(spec.pass_action_end))
    )


def _pass_action_indices(spec: ActorCriticSpec) -> tuple[int, ...]:
    return tuple(range(int(spec.pass_action_start), int(spec.pass_action_end)))


def _pointer_action_type_dim(spec: ActorCriticSpec) -> int:
    return len(_non_pass_action_indices(spec)) + 1


def _build_pointer_slot_target_ids(spec: ActorCriticSpec) -> np.ndarray:
    token_players = int(spec.token_player_count)
    pass_slots = len(_pass_action_indices(spec))
    table = np.full((token_players, pass_slots), -1, dtype=np.int32)
    if token_players <= 0 or pass_slots <= 0:
        return table
    half = token_players // 2
    for player_id in range(token_players):
        if token_players % 2 == 0:
            team_ids = range(0, half) if player_id < half else range(half, token_players)
        else:
            team_ids = range(token_players)
        teammates = [int(pid) for pid in team_ids if int(pid) != int(player_id)]
        for slot_idx, teammate_id in enumerate(teammates[:pass_slots]):
            table[player_id, slot_idx] = int(teammate_id)
    return table


def _flax_dense_kernel_init(nn):
    return nn.initializers.variance_scaling(
        scale=2.0,
        mode="fan_in",
        distribution="truncated_normal",
    )


def _flax_dense_bias_init(nn):
    return nn.initializers.zeros_init()


def build_actor_critic_module(spec: ActorCriticSpec):
    import jax
    from flax import linen as nn
    import jax.numpy as jnp

    kernel_init = _flax_dense_kernel_init(nn)
    bias_init = _flax_dense_bias_init(nn)
    zero_init = nn.initializers.zeros_init()
    non_pass_indices = _non_pass_action_indices(spec)
    pass_indices = _pass_action_indices(spec)
    pointer_type_dim = _pointer_action_type_dim(spec)
    pointer_pass_slot_count = len(pass_indices)
    pointer_slot_target_ids_np = _build_pointer_slot_target_ids(spec)

    class ActorCriticModule(nn.Module):
        @nn.compact
        def _mlp_forward(self, flat_obs, include_rebound_critic=False):
            hidden = flat_obs.astype(jnp.float32)
            for hidden_dim in spec.hidden_dims:
                hidden = nn.Dense(
                    int(hidden_dim),
                    kernel_init=kernel_init,
                    bias_init=bias_init,
                )(hidden)
                hidden = nn.tanh(hidden)

            flat_policy_logits = nn.Dense(
                int(spec.total_action_dim),
                kernel_init=kernel_init,
                bias_init=bias_init,
                name="policy_head",
            )(hidden)
            values = nn.Dense(
                1,
                kernel_init=kernel_init,
                bias_init=bias_init,
                name="value_head",
            )(hidden)[..., 0]
            if bool(spec.rebound_critic_enabled) and bool(include_rebound_critic):
                rebound_values = nn.Dense(
                    1,
                    kernel_init=kernel_init,
                    bias_init=bias_init,
                    name="rebound_value_head",
                )(hidden)[..., 0]
            else:
                rebound_values = jnp.zeros_like(values)
            return {
                "hidden": hidden,
                "flat_policy_logits": flat_policy_logits,
                "attention_weights": jnp.zeros(
                    (flat_obs.shape[0], 0, 0, 0),
                    dtype=jnp.float32,
                ),
                "values": values,
                "rebound_values": rebound_values,
                "selector_logits": jnp.zeros(
                    (flat_obs.shape[0], int(spec.num_intents)),
                    dtype=jnp.float32,
                ),
                "selector_values": jnp.zeros((flat_obs.shape[0],), dtype=jnp.float32),
            }

        @nn.compact
        def _attention_layer(self, tokens, *, layer_index: int):
            embed_dim = int(spec.attention_embed_dim)
            num_heads = int(spec.attention_num_heads)
            head_dim = embed_dim // num_heads
            # Keep the first block's historical names so existing one-layer
            # checkpoints load without a parameter migration.
            suffix = "" if int(layer_index) == 0 else f"_{int(layer_index)}"
            qkv = nn.Dense(
                3 * embed_dim,
                kernel_init=kernel_init,
                bias_init=bias_init,
                name=f"attention_qkv{suffix}",
            )(tokens)
            qkv = qkv.reshape(tokens.shape[0], tokens.shape[1], 3, num_heads, head_dim)
            query = qkv[:, :, 0]
            key = qkv[:, :, 1]
            value = qkv[:, :, 2]
            scale = jnp.asarray(head_dim, dtype=jnp.float32) ** -0.5
            scores = jnp.einsum("bthd,bshd->bhts", query, key) * scale
            weights = nn.softmax(scores, axis=-1)
            attended = jnp.einsum("bhts,bshd->bthd", weights, value)
            attended = attended.reshape(tokens.shape[0], tokens.shape[1], embed_dim)
            projected = nn.Dense(
                embed_dim,
                kernel_init=kernel_init,
                bias_init=bias_init,
                name=f"attention_out{suffix}",
            )(attended)
            return nn.LayerNorm(name=f"attention_norm{suffix}")(
                tokens + projected
            ), weights

        def _unpack_token_observation(self, flat_obs):
            batch_size = flat_obs.shape[0]
            player_dim = int(spec.token_player_count) * int(spec.token_dim)
            global_start = player_dim
            global_end = global_start + int(spec.global_dim)
            players = flat_obs[:, :player_dim].reshape(
                batch_size,
                int(spec.token_player_count),
                int(spec.token_dim),
            )
            globals_vec = flat_obs[:, global_start:global_end]
            role_flag = flat_obs[:, global_end : global_end + 1]
            return players.astype(jnp.float32), globals_vec.astype(jnp.float32), role_flag.astype(jnp.float32)

        @nn.compact
        def _apply_intent_conditioning(self, token_hidden, role_flag, intent_context):
            if not bool(spec.intent_embedding_enabled):
                return token_hidden

            batch_size = token_hidden.shape[0]
            if intent_context is None:
                intent_index = jnp.zeros((batch_size,), dtype=jnp.int32)
                intent_gate = jnp.zeros((batch_size,), dtype=jnp.float32)
            else:
                intent_index = jnp.asarray(
                    intent_context.get("intent_index", jnp.zeros((batch_size,), dtype=jnp.int32)),
                    dtype=jnp.int32,
                ).reshape((batch_size,))
                intent_gate = jnp.asarray(
                    intent_context.get(
                        "intent_gate",
                        intent_context.get("intent_active", jnp.zeros((batch_size,), dtype=jnp.float32)),
                    ),
                    dtype=jnp.float32,
                ).reshape((batch_size,))

            safe_index = jnp.clip(intent_index, 0, int(spec.num_intents) - 1)
            offense_embeddings = self.param(
                "offense_intent_embedding",
                nn.initializers.normal(stddev=0.02),
                (int(spec.num_intents), int(spec.intent_embedding_dim)),
            )
            defense_embeddings = self.param(
                "defense_intent_embedding",
                nn.initializers.normal(stddev=0.02),
                (int(spec.num_intents), int(spec.intent_embedding_dim)),
            )
            offense_delta = nn.Dense(
                int(spec.attention_embed_dim),
                use_bias=False,
                kernel_init=kernel_init,
                name="offense_intent_projection",
            )(offense_embeddings[safe_index])
            defense_delta = nn.Dense(
                int(spec.attention_embed_dim),
                use_bias=False,
                kernel_init=kernel_init,
                name="defense_intent_projection",
            )(defense_embeddings[safe_index])
            is_offense = role_flag[:, 0:1] > 0.0
            delta = jnp.where(is_offense, offense_delta, defense_delta)
            gate = jnp.clip(intent_gate, 0.0, 1.0)
            conditioning_scale = jnp.asarray(
                float(spec.intent_conditioning_scale),
                dtype=jnp.float32,
            )
            return token_hidden + (
                gate[:, None, None]
                * conditioning_scale
                * delta[:, None, :]
            )

        def _select_training_player_tokens(self, player_tokens, role_flag):
            if int(spec.token_player_count) == int(spec.training_player_count):
                return player_tokens[:, : int(spec.training_player_count), :]
            if int(spec.token_player_count) == int(spec.training_player_count) * 2:
                offense_tokens = player_tokens[:, : int(spec.training_player_count), :]
                defense_tokens = player_tokens[
                    :,
                    int(spec.training_player_count) : int(spec.training_player_count) * 2,
                    :,
                ]
                is_offense = role_flag[:, None, :] > 0.0
                return jnp.where(is_offense, offense_tokens, defense_tokens)
            return player_tokens[:, : int(spec.training_player_count), :]

        def _select_training_player_indices(self, batch_size, role_flag):
            if int(spec.token_player_count) == int(spec.training_player_count):
                ids = jnp.arange(int(spec.training_player_count), dtype=jnp.int32)
                return jnp.broadcast_to(ids[None, :], (batch_size, int(spec.training_player_count)))
            if int(spec.token_player_count) == int(spec.training_player_count) * 2:
                offense_ids = jnp.arange(int(spec.training_player_count), dtype=jnp.int32)
                defense_ids = offense_ids + int(spec.training_player_count)
                offense_ids = jnp.broadcast_to(
                    offense_ids[None, :],
                    (batch_size, int(spec.training_player_count)),
                )
                defense_ids = jnp.broadcast_to(
                    defense_ids[None, :],
                    (batch_size, int(spec.training_player_count)),
                )
                is_offense = role_flag > 0.0
                return jnp.where(is_offense, offense_ids, defense_ids)
            ids = jnp.arange(int(spec.training_player_count), dtype=jnp.int32)
            return jnp.broadcast_to(ids[None, :], (batch_size, int(spec.training_player_count)))

        def _attention_head_mlp(self, hidden, hidden_dims, *, prefix: str):
            out = hidden
            for idx, hidden_dim in enumerate(hidden_dims):
                out = nn.Dense(
                    int(hidden_dim),
                    kernel_init=kernel_init,
                    bias_init=bias_init,
                    name=f"{prefix}_{idx}",
                )(out)
                activation = str(spec.attention_head_activation).lower()
                if activation == "relu":
                    out = nn.relu(out)
                elif activation == "gelu":
                    out = nn.gelu(out)
                elif activation in {"silu", "swish"}:
                    out = nn.silu(out)
                else:
                    out = nn.tanh(out)
            return out

        def _head_activation(self, hidden):
            activation = str(spec.attention_head_activation).lower()
            if activation == "relu":
                return nn.relu(hidden)
            if activation == "gelu":
                return nn.gelu(hidden)
            if activation in {"silu", "swish"}:
                return nn.silu(hidden)
            return nn.tanh(hidden)

        def _compose_pointer_flat_logits(self, action_type_logits, pass_target_logits):
            type_log_probs = jax.nn.log_softmax(action_type_logits, axis=-1)
            pass_log_probs = jax.nn.log_softmax(pass_target_logits, axis=-1)
            batch_size = action_type_logits.shape[0]
            flat = jnp.full(
                (
                    batch_size,
                    int(spec.training_player_count),
                    int(spec.action_dim_per_player),
                ),
                MASKED_LOGIT_FLOOR,
                dtype=jnp.float32,
            )
            for type_idx, action_idx in enumerate(non_pass_indices):
                flat = flat.at[..., int(action_idx)].set(type_log_probs[..., int(type_idx)])
            pass_type_idx = len(non_pass_indices)
            for slot_idx, action_idx in enumerate(pass_indices):
                flat = flat.at[..., int(action_idx)].set(
                    type_log_probs[..., pass_type_idx] + pass_log_probs[..., int(slot_idx)]
                )
            return flat.reshape(batch_size, int(spec.total_action_dim))

        def _pointer_pass_target_logits(
            self,
            query_player_tokens,
            key_player_tokens,
            selected_player_ids,
            role_flag,
        ):
            embed_dim = int(query_player_tokens.shape[-1])
            q_offense = nn.Dense(
                embed_dim,
                use_bias=False,
                kernel_init=kernel_init,
                name="pointer_query_head_offense",
            )(query_player_tokens)
            k_offense = nn.Dense(
                embed_dim,
                use_bias=False,
                kernel_init=kernel_init,
                name="pointer_key_head_offense",
            )(key_player_tokens)
            q_defense = nn.Dense(
                embed_dim,
                use_bias=False,
                kernel_init=kernel_init,
                name="pointer_query_head_defense",
            )(query_player_tokens)
            k_defense = nn.Dense(
                embed_dim,
                use_bias=False,
                kernel_init=kernel_init,
                name="pointer_key_head_defense",
            )(key_player_tokens)
            is_offense = (role_flag > 0.0)[:, None, :]
            query = jnp.where(is_offense, q_offense, q_defense)
            key = jnp.where(is_offense, k_offense, k_defense)
            scale = jnp.asarray(embed_dim, dtype=jnp.float32) ** -0.5
            pair_scores = jnp.einsum("bpd,btd->bpt", query, key) * scale

            slot_table = jnp.asarray(pointer_slot_target_ids_np, dtype=jnp.int32)
            slot_target_ids = slot_table[selected_player_ids]
            valid_slots = slot_target_ids >= 0
            safe_target_ids = jnp.clip(slot_target_ids, 0, int(spec.token_player_count) - 1)
            slot_logits = jnp.take_along_axis(pair_scores, safe_target_ids, axis=-1)
            slot_logits = jnp.where(
                valid_slots,
                slot_logits,
                jnp.full_like(slot_logits, MASKED_LOGIT_FLOOR),
            )
            fallback = jnp.full_like(slot_logits, MASKED_LOGIT_FLOOR)
            fallback = fallback.at[..., 0].set(0.0)
            has_valid = jnp.any(valid_slots, axis=-1, keepdims=True)
            return jnp.where(has_valid, slot_logits, fallback)

        @nn.compact
        def _autoregressive_team_decode(self, selected_tokens, action_order, teacher_actions):
            batch_size = selected_tokens.shape[0]
            player_count = int(spec.training_player_count)
            embed_dim = int(spec.attention_embed_dim)
            if action_order is None:
                action_order = jnp.broadcast_to(
                    jnp.arange(player_count, dtype=jnp.int32)[None, :],
                    (batch_size, player_count),
                )
            else:
                action_order = jnp.asarray(action_order, dtype=jnp.int32).reshape(
                    batch_size, player_count
                )
            if teacher_actions is None:
                teacher_actions = jnp.zeros(
                    (batch_size, player_count), dtype=jnp.int32
                )
            else:
                teacher_actions = jnp.asarray(teacher_actions, dtype=jnp.int32).reshape(
                    batch_size, player_count
                )

            ordered_tokens = jnp.take_along_axis(
                selected_tokens, action_order[..., None], axis=1
            )
            ordered_actions = jnp.take_along_axis(teacher_actions, action_order, axis=1)
            action_embedding = nn.Embed(
                num_embeddings=int(spec.action_dim_per_player),
                features=embed_dim,
                embedding_init=nn.initializers.normal(stddev=0.02),
                name="team_action_embedding",
            )(ordered_actions)
            decoder_state = nn.Dense(
                embed_dim,
                kernel_init=kernel_init,
                bias_init=bias_init,
                name="team_decoder_state_projection",
            )(ordered_tokens)

            start_token = self.param(
                "team_decoder_start_token",
                nn.initializers.zeros_init(),
                (1, embed_dim),
            )
            start_batch = jnp.broadcast_to(
                start_token[None, :, :],
                (batch_size, 1, embed_dim),
            )
            key_value_input = jnp.concatenate(
                [start_batch, decoder_state + action_embedding], axis=1
            )
            num_heads = int(spec.attention_num_heads)
            head_dim = embed_dim // num_heads
            query = nn.Dense(
                embed_dim,
                kernel_init=kernel_init,
                bias_init=bias_init,
                name="team_decoder_query",
            )(decoder_state).reshape(batch_size, player_count, num_heads, head_dim)
            key = nn.Dense(
                embed_dim,
                kernel_init=kernel_init,
                bias_init=bias_init,
                name="team_decoder_key",
            )(key_value_input).reshape(batch_size, player_count + 1, num_heads, head_dim)
            value = nn.Dense(
                embed_dim,
                kernel_init=kernel_init,
                bias_init=bias_init,
                name="team_decoder_value",
            )(key_value_input).reshape(batch_size, player_count + 1, num_heads, head_dim)
            scale = jnp.asarray(head_dim, dtype=jnp.float32) ** -0.5
            scores = jnp.einsum("bphd,bshd->bhps", query, key) * scale
            query_positions = jnp.arange(player_count, dtype=jnp.int32)[:, None]
            key_positions = jnp.arange(player_count + 1, dtype=jnp.int32)[None, :]
            causal_mask = (key_positions == 0) | (key_positions <= query_positions)
            scores = jnp.where(
                causal_mask[None, None, :, :],
                scores,
                jnp.full_like(scores, MASKED_LOGIT_FLOOR),
            )
            weights = nn.softmax(scores, axis=-1)
            attended = jnp.einsum("bhps,bshd->bphd", weights, value).reshape(
                batch_size, player_count, embed_dim
            )
            attended = nn.Dense(
                embed_dim,
                kernel_init=kernel_init,
                bias_init=bias_init,
                name="team_decoder_output",
            )(attended)
            decoded = nn.LayerNorm(name="team_decoder_attention_norm")(
                ordered_tokens + attended
            )
            feed_forward = nn.Dense(
                4 * embed_dim,
                kernel_init=kernel_init,
                bias_init=bias_init,
                name="team_decoder_mlp_in",
            )(decoded)
            feed_forward = nn.gelu(feed_forward)
            feed_forward = nn.Dense(
                embed_dim,
                kernel_init=kernel_init,
                bias_init=bias_init,
                name="team_decoder_mlp_out",
            )(feed_forward)
            decoded = nn.LayerNorm(name="team_decoder_mlp_norm")(
                decoded + feed_forward
            )
            canonical_tokens = jnp.zeros_like(decoded)
            batch_index = jnp.arange(batch_size, dtype=jnp.int32)[:, None]
            return canonical_tokens.at[batch_index, action_order].set(decoded)

        @nn.compact
        def _attention_forward(
            self,
            flat_obs,
            intent_context=None,
            include_rebound_critic=False,
            action_order=None,
            teacher_actions=None,
        ):
            players, globals_vec, role_flag = self._unpack_token_observation(flat_obs)
            globals_expanded = jnp.broadcast_to(
                globals_vec[:, None, :],
                (players.shape[0], players.shape[1], globals_vec.shape[-1]),
            )
            token_input = jnp.concatenate([players, globals_expanded], axis=-1)
            token_hidden = nn.Dense(
                int(spec.attention_token_mlp_dim),
                kernel_init=kernel_init,
                bias_init=bias_init,
                name="token_mlp_0",
            )(token_input)
            token_hidden = nn.relu(token_hidden)
            token_hidden = nn.Dense(
                int(spec.attention_embed_dim),
                kernel_init=kernel_init,
                bias_init=bias_init,
                name="token_mlp_1",
            )(token_hidden)
            token_hidden = self._apply_intent_conditioning(
                token_hidden,
                role_flag,
                intent_context,
            )

            cls_count = max(0, int(spec.attention_num_cls_tokens))
            if cls_count > 0:
                cls_tokens = self.param(
                    "cls_tokens",
                    nn.initializers.zeros_init(),
                    (cls_count, int(spec.attention_embed_dim)),
                )
                cls_batch = jnp.broadcast_to(
                    cls_tokens[None, :, :],
                    (token_hidden.shape[0], cls_count, int(spec.attention_embed_dim)),
                )
                token_hidden = jnp.concatenate([token_hidden, cls_batch], axis=1)

            attended = token_hidden
            attention_weights = None
            for layer_index in range(int(spec.attention_num_layers)):
                attended, attention_weights = self._attention_layer(
                    attended,
                    layer_index=layer_index,
                )
            assert attention_weights is not None
            player_tokens = attended[:, : int(spec.token_player_count), :]
            pi_player_tokens = self._attention_head_mlp(
                player_tokens,
                spec.attention_pi_head_hidden_dims,
                prefix="pi_head_mlp",
            )
            selected_player_ids = self._select_training_player_indices(player_tokens.shape[0], role_flag)
            selected_tokens = self._select_training_player_tokens(pi_player_tokens, role_flag)
            if str(spec.team_action_mode) == TEAM_ACTION_MODE_AUTOREGRESSIVE:
                selected_tokens = self._autoregressive_team_decode(
                    selected_tokens,
                    action_order,
                    teacher_actions,
                )
            policy_logits_offense = nn.Dense(
                int(spec.action_dim_per_player),
                kernel_init=kernel_init,
                bias_init=bias_init,
                name="policy_head_offense",
            )(selected_tokens)
            policy_logits_defense = nn.Dense(
                int(spec.action_dim_per_player),
                kernel_init=kernel_init,
                bias_init=bias_init,
                name="policy_head_defense",
            )(selected_tokens)
            is_offense = role_flag[:, None, :] > 0.0
            policy_logits = jnp.where(is_offense, policy_logits_offense, policy_logits_defense)
            if str(spec.action_head_mode) == ACTION_HEAD_MODE_POINTER_TARGETED:
                action_type_logits_offense = nn.Dense(
                    int(pointer_type_dim),
                    kernel_init=kernel_init,
                    bias_init=bias_init,
                    name="pointer_action_type_head_offense",
                )(selected_tokens)
                action_type_logits_defense = nn.Dense(
                    int(pointer_type_dim),
                    kernel_init=kernel_init,
                    bias_init=bias_init,
                    name="pointer_action_type_head_defense",
                )(selected_tokens)
                action_type_logits = jnp.where(
                    is_offense,
                    action_type_logits_offense,
                    action_type_logits_defense,
                )
                pass_target_logits = self._pointer_pass_target_logits(
                    selected_tokens,
                    pi_player_tokens,
                    selected_player_ids,
                    role_flag,
                )
                flat_policy_logits = self._compose_pointer_flat_logits(
                    action_type_logits,
                    pass_target_logits,
                )
            else:
                action_type_logits = jnp.zeros(
                    (
                        flat_obs.shape[0],
                        int(spec.training_player_count),
                        int(pointer_type_dim),
                    ),
                    dtype=jnp.float32,
                )
                pass_target_logits = jnp.zeros(
                    (
                        flat_obs.shape[0],
                        int(spec.training_player_count),
                        int(pointer_pass_slot_count),
                    ),
                    dtype=jnp.float32,
                )
                flat_policy_logits = policy_logits.reshape(
                    policy_logits.shape[0],
                    int(spec.total_action_dim),
                )

            if cls_count >= 2:
                offense_value_input = attended[:, int(spec.token_player_count), :]
                defense_value_input = attended[:, int(spec.token_player_count) + 1, :]
                selector_input = offense_value_input
            elif cls_count == 1:
                offense_value_input = attended[:, int(spec.token_player_count), :]
                defense_value_input = offense_value_input
                selector_input = offense_value_input
            else:
                pooled = jnp.mean(player_tokens, axis=1)
                offense_value_input = pooled
                defense_value_input = pooled
                selector_input = pooled

            value_inputs = jnp.stack([offense_value_input, defense_value_input], axis=1)
            value_latents = self._attention_head_mlp(
                value_inputs,
                spec.attention_vf_head_hidden_dims,
                prefix="vf_head_mlp",
            )
            offense_value_input = value_latents[:, 0, :]
            defense_value_input = value_latents[:, 1, :]
            values_offense = nn.Dense(
                1,
                kernel_init=kernel_init,
                bias_init=bias_init,
                name="value_head_offense",
            )(offense_value_input)[..., 0]
            values_defense = nn.Dense(
                1,
                kernel_init=kernel_init,
                bias_init=bias_init,
                name="value_head_defense",
            )(defense_value_input)[..., 0]
            values = jnp.where(role_flag[:, 0] > 0.0, values_offense, values_defense)
            if bool(spec.rebound_critic_enabled) and bool(include_rebound_critic):
                rebound_values_offense = nn.Dense(
                    1,
                    kernel_init=kernel_init,
                    bias_init=bias_init,
                    name="rebound_value_head_offense",
                )(offense_value_input)[..., 0]
                rebound_values_defense = nn.Dense(
                    1,
                    kernel_init=kernel_init,
                    bias_init=bias_init,
                    name="rebound_value_head_defense",
                )(defense_value_input)[..., 0]
                rebound_values = jnp.where(
                    role_flag[:, 0] > 0.0, rebound_values_offense, rebound_values_defense
                )
            else:
                rebound_values = jnp.zeros_like(values)
            if bool(spec.intent_selector_enabled):
                selector_hidden = nn.Dense(
                    int(spec.intent_selector_hidden_dim),
                    kernel_init=kernel_init,
                    bias_init=bias_init,
                    name="intent_selector_head_0",
                )(selector_input)
                selector_hidden = self._head_activation(selector_hidden)
                selector_logits = nn.Dense(
                    int(spec.num_intents),
                    kernel_init=zero_init,
                    bias_init=zero_init,
                    name="intent_selector_head_out",
                )(selector_hidden)
                selector_value_hidden = nn.Dense(
                    int(spec.intent_selector_hidden_dim),
                    kernel_init=kernel_init,
                    bias_init=bias_init,
                    name="intent_selector_value_head_0",
                )(selector_input)
                selector_value_hidden = self._head_activation(selector_value_hidden)
                selector_values = nn.Dense(
                    1,
                    kernel_init=zero_init,
                    bias_init=zero_init,
                    name="intent_selector_value_head_out",
                )(selector_value_hidden)[..., 0]
            else:
                selector_logits = jnp.zeros(
                    (flat_obs.shape[0], int(spec.num_intents)),
                    dtype=jnp.float32,
                )
                selector_values = jnp.zeros((flat_obs.shape[0],), dtype=jnp.float32)
            return {
                "hidden": attended.reshape(attended.shape[0], -1),
                "flat_policy_logits": flat_policy_logits,
                "action_type_logits": action_type_logits,
                "pass_target_logits": pass_target_logits,
                "attention_weights": attention_weights,
                "values": values,
                "rebound_values": rebound_values,
                "selector_logits": selector_logits,
                "selector_values": selector_values,
            }

        def __call__(
            self,
            flat_obs,
            intent_context=None,
            include_rebound_critic=False,
            action_order=None,
            teacher_actions=None,
        ):
            if str(spec.model_type) == "attention":
                return self._attention_forward(
                    flat_obs,
                    intent_context,
                    include_rebound_critic,
                    action_order,
                    teacher_actions,
                )
            return self._mlp_forward(flat_obs, include_rebound_critic)

    return ActorCriticModule()


def init_actor_critic_params(jax, jnp, spec: ActorCriticSpec, *, seed: int):
    from flax.core import unfreeze

    module = build_actor_critic_module(spec)
    sample_flat_obs = jnp.zeros((1, int(spec.flat_obs_dim)), dtype=jnp.float32)
    variables = module.init(
        jax.random.PRNGKey(int(seed)),
        sample_flat_obs,
        include_rebound_critic=bool(spec.rebound_critic_enabled),
    )
    return unfreeze(variables["params"])


def actor_critic_forward(
    params,
    flat_obs,
    spec: ActorCriticSpec,
    jnp,
    *,
    intent_context=None,
    include_rebound_critic: bool = False,
    action_order=None,
    teacher_actions=None,
):
    module = build_actor_critic_module(spec)
    return module.apply(
        {"params": params},
        flat_obs.astype(jnp.float32),
        intent_context,
        include_rebound_critic,
        action_order,
        teacher_actions,
    )


def team_action_order_uses_rng(spec: ActorCriticSpec) -> bool:
    if str(spec.team_action_mode) != TEAM_ACTION_MODE_AUTOREGRESSIVE:
        return False
    return (
        str(spec.offense_action_order)
        in {TEAM_ACTION_ORDER_RANDOM, TEAM_ACTION_ORDER_BALL_HANDLER_LAST}
        or str(spec.defense_action_order) == TEAM_ACTION_ORDER_RANDOM
    )


def build_team_action_order(
    flat_obs,
    spec: ActorCriticSpec,
    order_key,
    jax,
    jnp,
    *,
    ball_holder=None,
    controlled_player_ids=None,
):
    batch_size = int(flat_obs.shape[0])
    player_count = int(spec.training_player_count)
    canonical = jnp.broadcast_to(
        jnp.arange(player_count, dtype=jnp.int32)[None, :],
        (batch_size, player_count),
    )
    if str(spec.team_action_mode) != TEAM_ACTION_MODE_AUTOREGRESSIVE:
        return canonical

    if team_action_order_uses_rng(spec):
        random_scores = jax.random.uniform(
            order_key,
            shape=(batch_size, player_count),
            dtype=jnp.float32,
        )
        random_order = jnp.argsort(random_scores, axis=-1).astype(jnp.int32)
    else:
        random_order = canonical

    offense_order = random_order
    if str(spec.offense_action_order) == TEAM_ACTION_ORDER_CANONICAL:
        offense_order = canonical
    elif str(spec.offense_action_order) == TEAM_ACTION_ORDER_BALL_HANDLER_LAST:
        if ball_holder is not None and controlled_player_ids is not None:
            holder = jnp.asarray(ball_holder, dtype=jnp.int32).reshape(-1)
            if holder.shape[0] == 1 and batch_size != 1:
                holder = jnp.broadcast_to(holder, (batch_size,))
            player_ids = jnp.asarray(controlled_player_ids, dtype=jnp.int32)
            if player_ids.ndim == 1:
                player_ids = jnp.broadcast_to(
                    player_ids[None, :], (batch_size, player_count)
                )
            holder_matches = player_ids == holder[:, None]
            holder_local = jnp.argmax(holder_matches, axis=-1).astype(jnp.int32)
            holder_local = jnp.where(jnp.any(holder_matches, axis=-1), holder_local, -1)
            is_holder = random_order == holder_local[:, None]
            stable_rank = jnp.broadcast_to(
                jnp.arange(player_count, dtype=jnp.int32)[None, :],
                (batch_size, player_count),
            )
            move_to_end_rank = jnp.where(is_holder, player_count, stable_rank)
            reorder_index = jnp.argsort(move_to_end_rank, axis=-1)
            offense_order = jnp.take_along_axis(
                random_order, reorder_index, axis=-1
            )

    defense_order = (
        canonical
        if str(spec.defense_action_order) == TEAM_ACTION_ORDER_CANONICAL
        else random_order
    )
    role_index = (
        int(spec.token_player_count) * int(spec.token_dim) + int(spec.global_dim)
    )
    is_offense = flat_obs[:, role_index] > 0.0
    return jnp.where(is_offense[:, None], offense_order, defense_order)


def apply_action_mask(flat_policy_logits, action_mask, spec: ActorCriticSpec, jax, jnp):
    batch_size = int(flat_policy_logits.shape[0])
    logits = flat_policy_logits.reshape(
        batch_size,
        int(spec.training_player_count),
        int(spec.action_dim_per_player),
    )
    legal = action_mask > 0
    has_legal = jnp.any(legal, axis=-1, keepdims=True)
    noop_mask = jnp.zeros_like(legal)
    noop_mask = noop_mask.at[..., NOOP_ACTION_INDEX].set(True)
    effective_legal = jnp.where(has_legal, legal, noop_mask)
    masked_logits = jnp.where(
        effective_legal,
        logits,
        jnp.full_like(logits, MASKED_LOGIT_FLOOR),
    )
    log_probs = jax.nn.log_softmax(masked_logits, axis=-1)
    probs = jnp.exp(log_probs)
    deterministic_actions = jnp.argmax(masked_logits, axis=-1).astype(jnp.int32)
    entropy = -jnp.sum(probs * log_probs, axis=-1)
    return {
        "masked_logits": masked_logits,
        "log_probs": log_probs,
        "probs": probs,
        "deterministic_actions": deterministic_actions,
        "entropy": entropy,
    }


def sample_actions(masked_logits, sample_key, jax, jnp):
    sampled_actions = jax.random.categorical(
        sample_key,
        masked_logits,
        axis=-1,
    ).astype(jnp.int32)
    log_probs = jax.nn.log_softmax(masked_logits, axis=-1)
    selected_log_probs = jnp.take_along_axis(
        log_probs,
        sampled_actions[..., None],
        axis=-1,
    )[..., 0]
    return sampled_actions, selected_log_probs


def run_actor_critic(
    params,
    flat_obs,
    action_mask,
    spec: ActorCriticSpec,
    sample_key,
    jax,
    jnp,
    *,
    intent_context=None,
    include_rebound_critic: bool = False,
    ball_holder=None,
    controlled_player_ids=None,
    action_order=None,
    generate_deterministic_actions: bool = True,
):
    if str(spec.team_action_mode) == TEAM_ACTION_MODE_AUTOREGRESSIVE:
        if action_order is None:
            if team_action_order_uses_rng(spec):
                order_key, sample_key = jax.random.split(sample_key)
            else:
                order_key = sample_key
            action_order = build_team_action_order(
                flat_obs,
                spec,
                order_key,
                jax,
                jnp,
                ball_holder=ball_holder,
                controlled_player_ids=controlled_player_ids,
            )
        batch_size = int(flat_obs.shape[0])
        player_count = int(spec.training_player_count)

        def generate_actions(*, deterministic: bool):
            actions = jnp.zeros((batch_size, player_count), dtype=jnp.int32)
            selected_log_probs = jnp.zeros(
                (batch_size, player_count), dtype=jnp.float32
            )
            final_forward = None
            final_mask = None
            batch_index = jnp.arange(batch_size, dtype=jnp.int32)
            for step_index in range(player_count):
                final_forward = actor_critic_forward(
                    params,
                    flat_obs,
                    spec,
                    jnp,
                    intent_context=intent_context,
                    include_rebound_critic=include_rebound_critic,
                    action_order=action_order,
                    teacher_actions=actions,
                )
                final_mask = apply_action_mask(
                    final_forward["flat_policy_logits"],
                    action_mask,
                    spec,
                    jax,
                    jnp,
                )
                player_index = action_order[:, step_index]
                player_logits = final_mask["masked_logits"][batch_index, player_index]
                if deterministic:
                    chosen_action = jnp.argmax(player_logits, axis=-1).astype(jnp.int32)
                else:
                    chosen_action = jax.random.categorical(
                        jax.random.fold_in(sample_key, step_index),
                        player_logits,
                        axis=-1,
                    ).astype(jnp.int32)
                player_log_probs = jax.nn.log_softmax(player_logits, axis=-1)
                chosen_log_prob = jnp.take_along_axis(
                    player_log_probs,
                    chosen_action[:, None],
                    axis=-1,
                )[:, 0]
                actions = actions.at[batch_index, player_index].set(chosen_action)
                selected_log_probs = selected_log_probs.at[
                    batch_index, player_index
                ].set(chosen_log_prob)
            assert final_forward is not None and final_mask is not None
            return actions, selected_log_probs, final_forward, final_mask

        sampled_actions, selected_log_probs, forward_out, mask_out = generate_actions(
            deterministic=False
        )
        if generate_deterministic_actions:
            deterministic_actions, _, _, deterministic_mask_out = generate_actions(
                deterministic=True
            )
        else:
            deterministic_actions = mask_out["deterministic_actions"]
            deterministic_mask_out = mask_out
        return {
            **forward_out,
            **mask_out,
            "sampled_actions": sampled_actions,
            "deterministic_actions": deterministic_actions,
            "deterministic_masked_logits": deterministic_mask_out["masked_logits"],
            "deterministic_log_probs": deterministic_mask_out["log_probs"],
            "deterministic_probs": deterministic_mask_out["probs"],
            "deterministic_entropy": deterministic_mask_out["entropy"],
            "selected_log_probs": selected_log_probs,
            "action_order": action_order,
        }

    forward_out = actor_critic_forward(
        params,
        flat_obs,
        spec,
        jnp,
        intent_context=intent_context,
        include_rebound_critic=include_rebound_critic,
    )
    mask_out = apply_action_mask(
        forward_out["flat_policy_logits"],
        action_mask,
        spec,
        jax,
        jnp,
    )
    sampled_actions, selected_log_probs = sample_actions(
        mask_out["masked_logits"],
        sample_key,
        jax,
        jnp,
    )
    return {
        **forward_out,
        **mask_out,
        "sampled_actions": sampled_actions,
        "selected_log_probs": selected_log_probs,
        "action_order": build_team_action_order(
            flat_obs,
            spec,
            sample_key,
            jax,
            jnp,
        ),
    }
