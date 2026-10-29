from __future__ import annotations

from dataclasses import asdict
import numpy as np
import pytest

from basketworld_jax.checkpoints import (
    build_checkpoint_payload,
    load_checkpoint,
    save_checkpoint,
)
from basketworld_jax.models.actor_critic import (
    ActorCriticSpec,
    actor_critic_forward,
    apply_action_mask,
    build_team_action_order,
    init_actor_critic_params,
    run_actor_critic,
    sample_actions,
)


def _autoregressive_attention_spec(**overrides):
    values = {
        "flat_obs_dim": 95,
        "training_player_count": 3,
        "action_dim_per_player": 14,
        "total_action_dim": 42,
        "hidden_dims": (8,),
        "model_type": "attention",
        "token_player_count": 6,
        "token_dim": 15,
        "global_dim": 4,
        "attention_embed_dim": 16,
        "attention_num_heads": 4,
        "attention_token_mlp_dim": 12,
        "attention_num_cls_tokens": 2,
        "team_action_mode": "autoregressive",
        "offense_action_order": "random",
        "defense_action_order": "random",
    }
    values.update(overrides)
    return ActorCriticSpec(**values)


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


def test_independent_mode_keeps_existing_sampling_key_and_canonical_storage_order():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    spec = ActorCriticSpec(5, 2, 3, 6, (8,))
    params = init_actor_critic_params(jax, jnp, spec, seed=2)
    flat_obs = jnp.ones((4, 5), dtype=jnp.float32)
    action_mask = jnp.ones((4, 2, 3), dtype=jnp.int8)
    sample_key = jax.random.PRNGKey(9)
    out = run_actor_critic(
        params, flat_obs, action_mask, spec, sample_key, jax, jnp
    )
    forward = actor_critic_forward(params, flat_obs, spec, jnp)
    masked = apply_action_mask(
        forward["flat_policy_logits"], action_mask, spec, jax, jnp
    )
    expected_actions, expected_log_probs = sample_actions(
        masked["masked_logits"], sample_key, jax, jnp
    )
    np.testing.assert_array_equal(np.asarray(out["sampled_actions"]), np.asarray(expected_actions))
    np.testing.assert_allclose(np.asarray(out["selected_log_probs"]), np.asarray(expected_log_probs))
    np.testing.assert_array_equal(
        np.asarray(out["action_order"]),
        np.asarray([[0, 1], [0, 1], [0, 1], [0, 1]], dtype=np.int32),
    )


def test_rebound_critic_head_is_opt_in_and_initialized_for_training():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    spec = ActorCriticSpec(
        flat_obs_dim=5,
        training_player_count=2,
        action_dim_per_player=3,
        total_action_dim=6,
        hidden_dims=(8,),
        rebound_critic_enabled=True,
    )
    params = init_actor_critic_params(jax, jnp, spec, seed=0)
    flat_obs = jnp.ones((4, 5), dtype=jnp.float32)

    training_out = actor_critic_forward(
        params,
        flat_obs,
        spec,
        jnp,
        include_rebound_critic=True,
    )
    inference_out = actor_critic_forward(params, flat_obs, spec, jnp)

    assert training_out["rebound_values"].shape == (4,)
    assert np.isfinite(np.asarray(training_out["rebound_values"])).all()
    np.testing.assert_allclose(np.asarray(inference_out["rebound_values"]), 0.0)

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
        attention_num_layers=2,
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
    assert out["selector_logits"].shape == (4, 8)
    assert out["selector_values"].shape == (4,)
    assert {
        "attention_qkv",
        "attention_out",
        "attention_norm",
        "attention_qkv_1",
        "attention_out_1",
        "attention_norm_1",
    }.issubset(params)
    np.testing.assert_allclose(
        np.asarray(out["attention_weights"]).sum(axis=-1),
        np.ones((4, 4, 8), dtype=np.float32),
        atol=1e-5,
    )


def test_autoregressive_decoder_is_causal_and_teacher_forced_log_probs_match_rollout():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    spec = _autoregressive_attention_spec(
        offense_action_order="canonical",
        defense_action_order="canonical",
    )
    params = init_actor_critic_params(jax, jnp, spec, seed=7)
    flat_obs = jnp.ones((2, 95), dtype=jnp.float32).at[:, -1].set(1.0)
    action_mask = jnp.ones((2, 3, 14), dtype=jnp.int8)
    action_mask = action_mask.at[:, 2, :].set(0)
    action_mask = action_mask.at[:, 2, 5].set(1)
    order = jnp.broadcast_to(jnp.arange(3, dtype=jnp.int32)[None, :], (2, 3))
    actions_a = jnp.zeros((2, 3), dtype=jnp.int32)
    actions_b = actions_a.at[:, 0].set(1)
    forward_a = actor_critic_forward(
        params,
        flat_obs,
        spec,
        jnp,
        action_order=order,
        teacher_actions=actions_a,
    )
    forward_b = actor_critic_forward(
        params,
        flat_obs,
        spec,
        jnp,
        action_order=order,
        teacher_actions=actions_b,
    )
    logits_a = np.asarray(forward_a["flat_policy_logits"]).reshape(2, 3, 14)
    logits_b = np.asarray(forward_b["flat_policy_logits"]).reshape(2, 3, 14)
    np.testing.assert_allclose(logits_a[:, 0], logits_b[:, 0], atol=1e-6)
    assert not np.allclose(logits_a[:, 1], logits_b[:, 1])

    rollout = run_actor_critic(
        params,
        flat_obs,
        action_mask,
        spec,
        jax.random.PRNGKey(11),
        jax,
        jnp,
    )
    np.testing.assert_array_equal(
        np.asarray(rollout["sampled_actions"][:, 2]), np.asarray([5, 5])
    )
    teacher_forced = actor_critic_forward(
        params,
        flat_obs,
        spec,
        jnp,
        action_order=rollout["action_order"],
        teacher_actions=rollout["sampled_actions"],
    )
    masked = apply_action_mask(
        teacher_forced["flat_policy_logits"], action_mask, spec, jax, jnp
    )
    selected = jnp.take_along_axis(
        masked["log_probs"], rollout["sampled_actions"][..., None], axis=-1
    )[..., 0]
    np.testing.assert_allclose(
        np.asarray(selected), np.asarray(rollout["selected_log_probs"]), atol=1e-6
    )
    ratio = jnp.exp(
        jnp.sum(selected - rollout["selected_log_probs"], axis=-1)
    )
    np.testing.assert_allclose(np.asarray(ratio), np.ones((2,), dtype=np.float32))


def test_ball_handler_last_order_moves_only_offense_holder_to_final_slot():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    spec = _autoregressive_attention_spec(offense_action_order="ball_handler_last")
    flat_obs = jnp.ones((3, 95), dtype=jnp.float32)
    flat_obs = flat_obs.at[0, -1].set(1.0)
    flat_obs = flat_obs.at[1, -1].set(1.0)
    flat_obs = flat_obs.at[2, -1].set(-1.0)
    order_key = jax.random.PRNGKey(13)
    order = build_team_action_order(
        flat_obs,
        spec,
        order_key,
        jax,
        jnp,
        ball_holder=jnp.asarray([1, -1, 4], dtype=jnp.int32),
        controlled_player_ids=jnp.asarray([0, 1, 2], dtype=jnp.int32),
    )
    random_order = build_team_action_order(
        flat_obs,
        _autoregressive_attention_spec(),
        order_key,
        jax,
        jnp,
        ball_holder=jnp.asarray([1, -1, 4], dtype=jnp.int32),
        controlled_player_ids=jnp.asarray([0, 1, 2], dtype=jnp.int32),
    )
    order_np = np.asarray(order, dtype=np.int32)
    assert order_np[0, -1] == 1
    np.testing.assert_array_equal(order_np[1], np.asarray(random_order)[1])
    np.testing.assert_array_equal(
        np.sort(order_np, axis=1),
        np.asarray([[0, 1, 2], [0, 1, 2], [0, 1, 2]]),
    )


def test_autoregressive_pointer_head_preserves_canonical_pass_target_slots():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    spec = _autoregressive_attention_spec(action_head_mode="pointer_targeted")
    params = init_actor_critic_params(jax, jnp, spec, seed=19)
    flat_obs = jnp.ones((1, 95), dtype=jnp.float32).at[:, -1].set(1.0)
    out = actor_critic_forward(
        params,
        flat_obs,
        spec,
        jnp,
        action_order=jnp.asarray([[2, 0, 1]], dtype=jnp.int32),
        teacher_actions=jnp.asarray([[8, 3, 9]], dtype=jnp.int32),
    )
    pass_logits = np.asarray(out["pass_target_logits"])
    assert pass_logits.shape == (1, 3, 6)
    assert np.isfinite(pass_logits[..., :2]).all()
    assert np.all(pass_logits[..., 2:] < -1.0e8)


def test_autoregressive_checkpoint_round_trip_preserves_decoder_metadata_and_params(tmp_path):
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    spec = _autoregressive_attention_spec(
        offense_action_order="ball_handler_last",
        defense_action_order="canonical",
    )
    params = init_actor_critic_params(jax, jnp, spec, seed=23)
    payload = build_checkpoint_payload(
        update_index=3,
        trainer_config={},
        policy_spec=asdict(spec),
        frozen_config={},
        params=params,
        opt_state={"count": np.asarray(0, dtype=np.int32)},
        current_state={"marker": np.asarray([1], dtype=np.int32)},
        eval_initial_state={"marker": np.asarray([2], dtype=np.int32)},
        base_key=np.asarray(jax.random.PRNGKey(5)),
        eval_trajectories=[],
        last_metrics={},
        resolved_training_config={
            "team_action_mode": "autoregressive",
            "offense_action_order": "ball_handler_last",
            "defense_action_order": "canonical",
        },
    )
    checkpoint_path = save_checkpoint(tmp_path / "checkpoint", payload)
    restored = load_checkpoint(checkpoint_path)
    assert ActorCriticSpec(**restored["policy_spec"]) == spec
    assert restored["resolved_training_config"]["team_action_mode"] == "autoregressive"
    assert "team_action_embedding" in restored["params"]


def test_single_attention_layer_keeps_legacy_parameter_names():
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
        attention_num_layers=1,
        attention_token_mlp_dim=12,
        attention_num_cls_tokens=2,
    )

    params = init_actor_critic_params(jax, jnp, spec, seed=0)

    assert "attention_qkv" in params
    assert "attention_out" in params
    assert "attention_norm" in params
    assert not any(name.startswith("attention_qkv_") for name in params)


def test_attention_intent_selector_head_starts_uniform_and_zero_value():
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
        intent_selector_enabled=True,
        intent_selector_hidden_dim=7,
    )
    params = init_actor_critic_params(jax, jnp, spec, seed=0)
    flat_obs = jnp.ones((3, 95), dtype=jnp.float32)
    flat_obs = flat_obs.at[:, -1].set(1.0)

    out = actor_critic_forward(
        params,
        flat_obs,
        spec,
        jnp,
        intent_context={
            "intent_index": jnp.zeros((3,), dtype=jnp.int32),
            "intent_gate": jnp.zeros((3,), dtype=jnp.float32),
        },
    )

    assert out["selector_logits"].shape == (3, 5)
    assert out["selector_values"].shape == (3,)
    np.testing.assert_allclose(
        np.asarray(out["selector_logits"]),
        np.zeros((3, 5), dtype=np.float32),
        atol=1e-7,
    )
    np.testing.assert_allclose(
        np.asarray(out["selector_values"]),
        np.zeros((3,), dtype=np.float32),
        atol=1e-7,
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
