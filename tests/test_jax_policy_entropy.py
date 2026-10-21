"""Action entropy reporting must not inherit PPO loss normalization."""

import math

import numpy as np
import pytest

from basketworld_jax.models import ActorCriticSpec, init_actor_critic_params
from basketworld_jax.optim import build_adam_transform, init_optimizer_state
from basketworld_jax.train.main import _filter_mlflow_train_metrics
from basketworld_jax.train.runtime import build_jitted_ppo_update_runner
from basketworld_jax.train.types import PPOBatch, TrainerConfig


def _entropy_case(minibatches, eligibility):
    jax = pytest.importorskip("jax")
    jnp = jax.numpy
    optax = pytest.importorskip("optax")
    size, players, actions = 32, 2, 4
    spec = ActorCriticSpec(3, players, actions, players * actions, (4,))
    params = jax.tree.map(
        jnp.zeros_like, init_actor_critic_params(jax, jnp, spec, seed=17)
    )
    # Uniform distributions over different legal-action counts, including
    # forced actions, give an analytic entropy independent of model learning.
    legal_counts = np.column_stack(
        (np.arange(size) % actions + 1, (np.arange(size) // 2) % actions + 1)
    )
    mask = np.arange(actions)[None, None, :] < legal_counts[:, :, None]
    weights = np.ones(size, dtype=np.float32)
    if eligibility == "sparse":
        weights[:] = 0
        weights[::7] = 1  # Uneven counts and empty minibatches.
    elif eligibility == "empty":
        weights[:] = 0
    denominator = 2.0 if eligibility == "completed" else max(weights.sum(), 1.0)
    data = {field: jnp.zeros(size, dtype=jnp.float32) for field in PPOBatch._fields}
    data.update(
        flat_obs=jnp.zeros((size, 3)),
        policy_intent_index=jnp.zeros(size, dtype=jnp.int32),
        action_mask=jnp.asarray(mask),
        actions=jnp.zeros((size, players), dtype=jnp.int32),
        old_selected_log_probs=jnp.asarray(-np.log(legal_counts), dtype=jnp.float32),
        rebound_positioning_mask=jnp.zeros((size, players)),
        rebound_counterfactual_advantages=jnp.zeros((size, players)),
        rebound_counterfactual_raw_advantages=jnp.zeros((size, players)),
        rebound_counterfactual_mask=jnp.zeros((size, players)),
        active_mask=jnp.asarray(weights),
        loss_weights=jnp.asarray(weights),
        loss_denominator=jnp.full(size, denominator),
    )
    config = TrainerConfig(
        kernel_batch_size=size,
        rollout_horizon=1,
        num_updates=1,
        gamma=0.99,
        gae_lambda=0.95,
        ppo_clip_range=0.2,
        value_coef=0.5,
        entropy_coef=0.4,
        learning_rate=0.0,
        policy_update_epochs=1,
        ppo_minibatches=minibatches,
        ppo_completed_episodes_only=eligibility == "completed",
    )
    transform = build_adam_transform(optax, learning_rate=config.learning_rate)
    opt_state = init_optimizer_state(transform, params)
    entropy_sum = float(np.sum(np.log(legal_counts).mean(axis=-1) * weights))
    return spec, config, params, opt_state, PPOBatch(**data), entropy_sum, weights.sum()


@pytest.mark.parametrize("minibatches", [1, 4, 16])
@pytest.mark.parametrize("eligibility", ["all", "sparse", "completed", "empty"])
def test_policy_entropy_is_sample_mean_without_changing_legacy_loss(minibatches, eligibility):
    jax = pytest.importorskip("jax")
    spec, config, params, opt_state, batch, entropy_sum, count = _entropy_case(
        minibatches, eligibility
    )
    runner, _ = build_jitted_ppo_update_runner(jax, jax.numpy, spec, config)
    _, _, metrics = runner(params, opt_state, batch, jax.random.PRNGKey(31))

    expected = entropy_sum / max(float(count), 1.0)
    assert math.isfinite(float(metrics["policy_entropy"]))
    assert float(metrics["policy_entropy"]) == pytest.approx(expected, abs=1e-6)
    # Preserve both the historic metric and the actual entropy loss term,
    # including episode-count normalization and the minibatch contribution.
    legacy_entropy = entropy_sum / float(batch.loss_denominator[0]) / minibatches
    assert float(metrics["entropy_bonus"]) == pytest.approx(legacy_entropy, rel=1e-6, abs=1e-6)
    assert float(metrics["total_loss"]) == pytest.approx(
        -config.entropy_coef * legacy_entropy, abs=1e-6
    )


@pytest.mark.parametrize("profile", ["core", "full"])
def test_policy_entropy_is_available_in_both_mlflow_profiles(profile):
    metrics = {"policy_entropy": 1.76, "entropy_bonus": 0.11, "entropy_coef": 0.4}
    assert _filter_mlflow_train_metrics(metrics, profile=profile) == metrics
