from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


NOOP_ACTION_INDEX = 0
MASKED_LOGIT_FLOOR = -1.0e9


@dataclass(frozen=True)
class PhaseAActorCriticSpec:
    flat_obs_dim: int
    training_player_count: int
    action_dim_per_player: int
    total_action_dim: int
    hidden_dims: tuple[int, ...]


def build_phase_a_actor_critic_spec(
    flat_obs_batch: np.ndarray,
    action_mask_batch: np.ndarray,
    hidden_dims: Sequence[int],
) -> PhaseAActorCriticSpec:
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
    return PhaseAActorCriticSpec(
        flat_obs_dim=int(flat_obs_batch.shape[1]),
        training_player_count=training_player_count,
        action_dim_per_player=action_dim_per_player,
        total_action_dim=training_player_count * action_dim_per_player,
        hidden_dims=tuple(int(v) for v in hidden_dims),
    )


def _init_linear_params(jax, jnp, *, in_dim: int, out_dim: int, key):
    scale = np.sqrt(2.0 / max(1, int(in_dim)))
    weights = (
        jax.random.normal(
            key,
            shape=(int(in_dim), int(out_dim)),
            dtype=jnp.float32,
        )
        * jnp.asarray(scale, dtype=jnp.float32)
    )
    bias = jnp.zeros((int(out_dim),), dtype=jnp.float32)
    return weights, bias


def init_phase_a_actor_critic_params(jax, jnp, spec: PhaseAActorCriticSpec, *, seed: int):
    hidden_dims = [int(v) for v in spec.hidden_dims]
    trunk_dims = [int(spec.flat_obs_dim), *hidden_dims]
    num_keys = max(0, len(trunk_dims) - 1) + 2
    keys = iter(jax.random.split(jax.random.PRNGKey(int(seed)), num_keys))

    trunk = []
    for in_dim, out_dim in zip(trunk_dims[:-1], trunk_dims[1:]):
        trunk.append(
            _init_linear_params(
                jax,
                jnp,
                in_dim=int(in_dim),
                out_dim=int(out_dim),
                key=next(keys),
            )
        )

    trunk_out_dim = hidden_dims[-1] if hidden_dims else int(spec.flat_obs_dim)
    policy_head = _init_linear_params(
        jax,
        jnp,
        in_dim=trunk_out_dim,
        out_dim=int(spec.total_action_dim),
        key=next(keys),
    )
    value_head = _init_linear_params(
        jax,
        jnp,
        in_dim=trunk_out_dim,
        out_dim=1,
        key=next(keys),
    )
    return {
        "trunk": tuple(trunk),
        "policy_head": policy_head,
        "value_head": value_head,
    }


def actor_critic_forward(params, flat_obs, jnp):
    hidden = flat_obs.astype(jnp.float32)
    for weights, bias in params["trunk"]:
        hidden = jnp.tanh(jnp.matmul(hidden, weights) + bias)

    policy_weights, policy_bias = params["policy_head"]
    value_weights, value_bias = params["value_head"]
    flat_policy_logits = jnp.matmul(hidden, policy_weights) + policy_bias
    values = (jnp.matmul(hidden, value_weights) + value_bias)[..., 0]
    return {
        "hidden": hidden,
        "flat_policy_logits": flat_policy_logits,
        "values": values,
    }


def apply_action_mask(flat_policy_logits, action_mask, spec: PhaseAActorCriticSpec, jax, jnp):
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


def run_actor_critic(params, flat_obs, action_mask, spec: PhaseAActorCriticSpec, sample_key, jax, jnp):
    forward_out = actor_critic_forward(params, flat_obs, jnp)
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
