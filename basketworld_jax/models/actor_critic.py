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
    attention_token_mlp_dim: int = 64
    attention_num_cls_tokens: int = 2
    attention_pi_head_hidden_dims: tuple[int, ...] = ()
    attention_vf_head_hidden_dims: tuple[int, ...] = ()
    attention_head_activation: str = "tanh"
    action_head_mode: str = ACTION_HEAD_MODE_FLAT
    pass_action_start: int = PASS_ACTION_START
    pass_action_end: int = PASS_ACTION_END


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
    attention_token_mlp_dim: int = 64,
    attention_num_cls_tokens: int = 2,
    attention_pi_head_hidden_dims: Sequence[int] = (),
    attention_vf_head_hidden_dims: Sequence[int] = (),
    attention_head_activation: str = "tanh",
    action_head_mode: str = ACTION_HEAD_MODE_FLAT,
    pass_action_start: int = PASS_ACTION_START,
    pass_action_end: int = PASS_ACTION_END,
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
    if normalized_model_type not in {"mlp", "attention"}:
        raise ValueError(f"Unsupported JAX policy model type {model_type!r}.")
    if normalized_action_head_mode not in {ACTION_HEAD_MODE_FLAT, ACTION_HEAD_MODE_POINTER_TARGETED}:
        raise ValueError(f"Unsupported JAX action head mode {action_head_mode!r}.")
    if normalized_action_head_mode == ACTION_HEAD_MODE_POINTER_TARGETED and normalized_model_type != "attention":
        raise ValueError("Pointer-targeted JAX action head currently requires --policy-model attention.")
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
        if token_player_count <= 0 or token_dim <= 0 or global_dim <= 0:
            raise ValueError("Attention policy requires positive token/player/global dimensions.")
        expected_dim = (token_player_count * token_dim) + global_dim + 1
        if int(flat_obs_batch.shape[1]) != expected_dim:
            raise ValueError(
                f"Attention policy expected packed obs dim {expected_dim}, got {int(flat_obs_batch.shape[1])}."
            )
        if attention_embed_dim % attention_num_heads != 0:
            raise ValueError("--attention-embed-dim must be divisible by --attention-num-heads.")
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
        attention_token_mlp_dim=int(attention_token_mlp_dim),
        attention_num_cls_tokens=int(attention_num_cls_tokens),
        attention_pi_head_hidden_dims=tuple(int(v) for v in attention_pi_head_hidden_dims),
        attention_vf_head_hidden_dims=tuple(int(v) for v in attention_vf_head_hidden_dims),
        attention_head_activation=str(attention_head_activation).lower(),
        action_head_mode=normalized_action_head_mode,
        pass_action_start=int(pass_action_start),
        pass_action_end=int(pass_action_end),
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
    non_pass_indices = _non_pass_action_indices(spec)
    pass_indices = _pass_action_indices(spec)
    pointer_type_dim = _pointer_action_type_dim(spec)
    pointer_pass_slot_count = len(pass_indices)
    pointer_slot_target_ids_np = _build_pointer_slot_target_ids(spec)

    class ActorCriticModule(nn.Module):
        @nn.compact
        def _mlp_forward(self, flat_obs):
            hidden = flat_obs.astype(np.float32)
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
            return {
                "hidden": hidden,
                "flat_policy_logits": flat_policy_logits,
                "attention_weights": jnp.zeros(
                    (flat_obs.shape[0], 0, 0, 0),
                    dtype=jnp.float32,
                ),
                "values": values,
            }

        @nn.compact
        def _attention_layer(self, tokens):
            embed_dim = int(spec.attention_embed_dim)
            num_heads = int(spec.attention_num_heads)
            head_dim = embed_dim // num_heads
            qkv = nn.Dense(
                3 * embed_dim,
                kernel_init=kernel_init,
                bias_init=bias_init,
                name="attention_qkv",
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
                name="attention_out",
            )(attended)
            return nn.LayerNorm(name="attention_norm")(tokens + projected), weights

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

        def _pointer_pass_target_logits(self, pi_player_tokens, selected_player_ids, role_flag):
            embed_dim = int(pi_player_tokens.shape[-1])
            q_offense = nn.Dense(
                embed_dim,
                use_bias=False,
                kernel_init=kernel_init,
                name="pointer_query_head_offense",
            )(pi_player_tokens)
            k_offense = nn.Dense(
                embed_dim,
                use_bias=False,
                kernel_init=kernel_init,
                name="pointer_key_head_offense",
            )(pi_player_tokens)
            q_defense = nn.Dense(
                embed_dim,
                use_bias=False,
                kernel_init=kernel_init,
                name="pointer_query_head_defense",
            )(pi_player_tokens)
            k_defense = nn.Dense(
                embed_dim,
                use_bias=False,
                kernel_init=kernel_init,
                name="pointer_key_head_defense",
            )(pi_player_tokens)
            is_offense = (role_flag > 0.0)[:, None, :]
            query = jnp.where(is_offense, q_offense, q_defense)
            key = jnp.where(is_offense, k_offense, k_defense)
            scale = jnp.asarray(embed_dim, dtype=jnp.float32) ** -0.5
            pair_scores = jnp.einsum("bpd,btd->bpt", query, key) * scale

            slot_table = jnp.asarray(pointer_slot_target_ids_np, dtype=jnp.int32)
            slot_target_ids = slot_table[selected_player_ids]
            valid_slots = slot_target_ids >= 0
            safe_target_ids = jnp.clip(slot_target_ids, 0, int(spec.token_player_count) - 1)
            batch_idx = jnp.arange(pi_player_tokens.shape[0], dtype=jnp.int32)[:, None, None]
            passer_idx = selected_player_ids[:, :, None]
            slot_logits = pair_scores[batch_idx, passer_idx, safe_target_ids]
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
        def _attention_forward(self, flat_obs):
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

            attended, attention_weights = self._attention_layer(token_hidden)
            player_tokens = attended[:, : int(spec.token_player_count), :]
            pi_player_tokens = self._attention_head_mlp(
                player_tokens,
                spec.attention_pi_head_hidden_dims,
                prefix="pi_head_mlp",
            )
            selected_player_ids = self._select_training_player_indices(player_tokens.shape[0], role_flag)
            selected_tokens = self._select_training_player_tokens(pi_player_tokens, role_flag)
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
            elif cls_count == 1:
                offense_value_input = attended[:, int(spec.token_player_count), :]
                defense_value_input = offense_value_input
            else:
                pooled = jnp.mean(player_tokens, axis=1)
                offense_value_input = pooled
                defense_value_input = pooled

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
            return {
                "hidden": attended.reshape(attended.shape[0], -1),
                "flat_policy_logits": flat_policy_logits,
                "action_type_logits": action_type_logits,
                "pass_target_logits": pass_target_logits,
                "attention_weights": attention_weights,
                "values": values,
            }

        def __call__(self, flat_obs):
            if str(spec.model_type) == "attention":
                return self._attention_forward(flat_obs)
            return self._mlp_forward(flat_obs)

    return ActorCriticModule()


def init_actor_critic_params(jax, jnp, spec: ActorCriticSpec, *, seed: int):
    from flax.core import unfreeze

    module = build_actor_critic_module(spec)
    sample_flat_obs = jnp.zeros((1, int(spec.flat_obs_dim)), dtype=jnp.float32)
    variables = module.init(jax.random.PRNGKey(int(seed)), sample_flat_obs)
    return unfreeze(variables["params"])


def actor_critic_forward(params, flat_obs, spec: ActorCriticSpec, jnp):
    module = build_actor_critic_module(spec)
    return module.apply({"params": params}, flat_obs.astype(jnp.float32))


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


def run_actor_critic(params, flat_obs, action_mask, spec: ActorCriticSpec, sample_key, jax, jnp):
    forward_out = actor_critic_forward(params, flat_obs, spec, jnp)
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
    }
