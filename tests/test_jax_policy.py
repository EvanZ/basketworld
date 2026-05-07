from __future__ import annotations

import numpy as np
import pytest

from basketworld_jax.models.actor_critic import (
    ActorCriticSpec,
    actor_critic_forward,
    apply_action_mask,
    init_actor_critic_params,
    run_actor_critic,
)


def test_actor_critic_forward_shapes():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    spec = ActorCriticSpec(
        flat_obs_dim=5,
        training_player_count=2,
        action_dim_per_player=3,
        total_action_dim=6,
        hidden_dims=(8,),
    )
    params = init_actor_critic_params(jax, jnp, spec, seed=0)
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
    assert out["attention_weights"].shape == (4, 0, 0, 0)


def test_attention_actor_critic_forward_shapes():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    spec = ActorCriticSpec(
        flat_obs_dim=95,
        training_player_count=3,
        action_dim_per_player=14,
        total_action_dim=42,
        hidden_dims=(8,),
        model_type="attention",
        token_player_count=6,
        token_dim=15,
        global_dim=4,
        attention_embed_dim=16,
        attention_num_heads=4,
        attention_token_mlp_dim=12,
        attention_num_cls_tokens=2,
        attention_pi_head_hidden_dims=(8, 8),
        attention_vf_head_hidden_dims=(8, 8),
        attention_head_activation="relu",
        action_head_mode="pointer_targeted",
    )
    params = init_actor_critic_params(jax, jnp, spec, seed=0)
    flat_obs = jnp.ones((4, 95), dtype=jnp.float32)
    flat_obs = flat_obs.at[:, -1].set(1.0)
    action_mask = jnp.ones((4, 3, 14), dtype=jnp.int8)
    out = run_actor_critic(
        params,
        flat_obs,
        action_mask,
        spec,
        jax.random.PRNGKey(1),
        jax,
        jnp,
    )

    assert out["flat_policy_logits"].shape == (4, 42)
    assert out["masked_logits"].shape == (4, 3, 14)
    assert out["sampled_actions"].shape == (4, 3)
    assert out["selected_log_probs"].shape == (4, 3)
    assert out["values"].shape == (4,)
    assert out["action_type_logits"].shape == (4, 3, 9)
    assert out["pass_target_logits"].shape == (4, 3, 6)
    assert out["attention_weights"].shape == (4, 4, 8, 8)
    np.testing.assert_allclose(
        np.asarray(out["attention_weights"]).sum(axis=-1),
        np.ones((4, 4, 8), dtype=np.float32),
        atol=1e-5,
    )


def test_attention_intent_embedding_conditions_policy_when_active():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    spec = ActorCriticSpec(
        flat_obs_dim=95,
        training_player_count=3,
        action_dim_per_player=14,
        total_action_dim=42,
        hidden_dims=(8,),
        model_type="attention",
        token_player_count=6,
        token_dim=15,
        global_dim=4,
        attention_embed_dim=16,
        attention_num_heads=4,
        attention_token_mlp_dim=12,
        attention_num_cls_tokens=2,
        intent_embedding_enabled=True,
        intent_embedding_dim=4,
        num_intents=5,
    )
    params = init_actor_critic_params(jax, jnp, spec, seed=0)
    flat_obs = jnp.ones((3, 95), dtype=jnp.float32)
    flat_obs = flat_obs.at[:, -1].set(1.0)

    out_a = actor_critic_forward(
        params,
        flat_obs,
        spec,
        jnp,
        intent_context={
            "intent_index": jnp.full((3,), 1, dtype=jnp.int32),
            "intent_gate": jnp.ones((3,), dtype=jnp.float32),
        },
    )
    out_b = actor_critic_forward(
        params,
        flat_obs,
        spec,
        jnp,
        intent_context={
            "intent_index": jnp.full((3,), 2, dtype=jnp.int32),
            "intent_gate": jnp.ones((3,), dtype=jnp.float32),
        },
    )

    assert not np.allclose(
        np.asarray(out_a["flat_policy_logits"]),
        np.asarray(out_b["flat_policy_logits"]),
    )


def test_attention_intent_embedding_zero_gate_matches_no_context():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    spec = ActorCriticSpec(
        flat_obs_dim=95,
        training_player_count=3,
        action_dim_per_player=14,
        total_action_dim=42,
        hidden_dims=(8,),
        model_type="attention",
        token_player_count=6,
        token_dim=15,
        global_dim=4,
        attention_embed_dim=16,
        attention_num_heads=4,
        attention_token_mlp_dim=12,
        attention_num_cls_tokens=2,
        intent_embedding_enabled=True,
        intent_embedding_dim=4,
        num_intents=5,
    )
    params = init_actor_critic_params(jax, jnp, spec, seed=0)
    flat_obs = jnp.ones((3, 95), dtype=jnp.float32)
    flat_obs = flat_obs.at[:, -1].set(1.0)

    no_context = actor_critic_forward(params, flat_obs, spec, jnp)
    zero_gate = actor_critic_forward(
        params,
        flat_obs,
        spec,
        jnp,
        intent_context={
            "intent_index": jnp.full((3,), 4, dtype=jnp.int32),
            "intent_gate": jnp.zeros((3,), dtype=jnp.float32),
        },
    )

    np.testing.assert_allclose(
        np.asarray(no_context["flat_policy_logits"]),
        np.asarray(zero_gate["flat_policy_logits"]),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(no_context["values"]),
        np.asarray(zero_gate["values"]),
        atol=1e-6,
    )


def test_pointer_targeted_action_head_produces_final_action_distribution():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    spec = ActorCriticSpec(
        flat_obs_dim=95,
        training_player_count=3,
        action_dim_per_player=14,
        total_action_dim=42,
        hidden_dims=(8,),
        model_type="attention",
        token_player_count=6,
        token_dim=15,
        global_dim=4,
        attention_embed_dim=16,
        attention_num_heads=4,
        attention_token_mlp_dim=12,
        attention_num_cls_tokens=2,
        attention_pi_head_hidden_dims=(8,),
        attention_vf_head_hidden_dims=(8,),
        action_head_mode="pointer_targeted",
    )
    params = init_actor_critic_params(jax, jnp, spec, seed=0)
    flat_obs = jnp.ones((2, 95), dtype=jnp.float32)
    flat_obs = flat_obs.at[:, -1].set(1.0)
    action_mask = jnp.zeros((2, 3, 14), dtype=jnp.int8)
    action_mask = action_mask.at[..., 0].set(1)
    action_mask = action_mask.at[:, 0, 8:10].set(1)

    forward_out = actor_critic_forward(params, flat_obs, spec, jnp)
    masked = apply_action_mask(
        forward_out["flat_policy_logits"],
        action_mask,
        spec,
        jax,
        jnp,
    )

    assert forward_out["action_type_logits"].shape == (2, 3, 9)
    assert forward_out["pass_target_logits"].shape == (2, 3, 6)
    np.testing.assert_allclose(
        np.asarray(masked["probs"].sum(axis=-1)),
        np.ones((2, 3), dtype=np.float32),
        atol=1e-6,
    )
    assert np.asarray(masked["probs"][:, 0, 8:10]).sum() > 0.0
    assert np.asarray(masked["probs"][:, 1:, 8:]).sum() == pytest.approx(0.0, abs=1e-6)


def test_apply_action_mask_respects_legality_and_noop_fallback():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    spec = ActorCriticSpec(
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
