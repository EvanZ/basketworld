from __future__ import annotations

import numpy as np
import pytest

from benchmarks.jax_phase_a_policy import (
    PhaseAActorCriticSpec,
    apply_action_mask,
    init_phase_a_actor_critic_params,
    run_actor_critic,
)


def test_actor_critic_forward_shapes():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    spec = PhaseAActorCriticSpec(
        flat_obs_dim=5,
        training_player_count=2,
        action_dim_per_player=3,
        total_action_dim=6,
        hidden_dims=(8,),
    )
    params = init_phase_a_actor_critic_params(jax, jnp, spec, seed=0)
    flat_obs = jnp.ones((4, 5), dtype=jnp.float32)
    action_mask = jnp.ones((4, 2, 3), dtype=jnp.int8)
    out = run_actor_critic(
        params,
        flat_obs,
        action_mask,
        spec,
        jax.random.PRNGKey(1),
        jax,
        jnp,
    )

    assert out["flat_policy_logits"].shape == (4, 6)
    assert out["masked_logits"].shape == (4, 2, 3)
    assert out["sampled_actions"].shape == (4, 2)
    assert out["selected_log_probs"].shape == (4, 2)
    assert out["values"].shape == (4,)


def test_apply_action_mask_respects_legality_and_noop_fallback():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    spec = PhaseAActorCriticSpec(
        flat_obs_dim=3,
        training_player_count=2,
        action_dim_per_player=3,
        total_action_dim=6,
        hidden_dims=(4,),
    )
    flat_logits = jnp.array(
        [
            [1.0, 3.0, 2.0, 5.0, 1.0, -2.0],
            [2.0, 1.0, 0.0, -1.0, -2.0, -3.0],
        ],
        dtype=jnp.float32,
    )
    action_mask = jnp.array(
        [
            [[0, 1, 0], [0, 0, 0]],
            [[1, 0, 1], [0, 1, 0]],
        ],
        dtype=jnp.int8,
    )
    out = apply_action_mask(flat_logits, action_mask, spec, jax, jnp)

    np.testing.assert_array_equal(
        np.asarray(out["deterministic_actions"], dtype=np.int32),
        np.array([[1, 0], [0, 1]], dtype=np.int32),
    )
    masked_logits = np.asarray(out["masked_logits"], dtype=np.float32)
    assert masked_logits[0, 0, 0] < -1.0e8
    assert masked_logits[0, 1, 1] < -1.0e8
    assert masked_logits[0, 1, 0] > -1.0
