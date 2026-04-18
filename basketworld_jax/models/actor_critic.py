from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


NOOP_ACTION_INDEX = 0
MASKED_LOGIT_FLOOR = -1.0e9


@dataclass(frozen=True)
class ActorCriticSpec:
    flat_obs_dim: int
    training_player_count: int
    action_dim_per_player: int
    total_action_dim: int
    hidden_dims: tuple[int, ...]


def build_actor_critic_spec(
    flat_obs_batch: np.ndarray,
    action_mask_batch: np.ndarray,
    hidden_dims: Sequence[int],
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
    return ActorCriticSpec(
        flat_obs_dim=int(flat_obs_batch.shape[1]),
        training_player_count=training_player_count,
        action_dim_per_player=action_dim_per_player,
        total_action_dim=training_player_count * action_dim_per_player,
        hidden_dims=tuple(int(v) for v in hidden_dims),
    )


def _flax_dense_kernel_init(nn):
    return nn.initializers.variance_scaling(
        scale=2.0,
        mode="fan_in",
        distribution="truncated_normal",
    )


def _flax_dense_bias_init(nn):
    return nn.initializers.zeros_init()


def build_actor_critic_module(spec: ActorCriticSpec):
    from flax import linen as nn

    kernel_init = _flax_dense_kernel_init(nn)
    bias_init = _flax_dense_bias_init(nn)

    class ActorCriticModule(nn.Module):
        @nn.compact
        def __call__(self, flat_obs):
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
                "values": values,
            }

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
