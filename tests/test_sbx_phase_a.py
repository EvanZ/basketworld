from __future__ import annotations

import numpy as np
import pytest

from benchmarks.sbx_phase_a import (
    PHASE_A_FROZEN_VALUES,
    PhaseAPolicySpec,
    _build_jitted_policy_forward,
    build_phase_a_args,
    extract_phase_a_training_action_mask,
    flatten_phase_a_observation,
    flatten_phase_a_observation_batch,
)


def test_build_phase_a_args_uses_frozen_scope_defaults():
    args = build_phase_a_args([])

    for key, expected in PHASE_A_FROZEN_VALUES.items():
        actual = getattr(args, key)
        if isinstance(expected, float):
            assert np.isclose(float(actual), float(expected), atol=1e-8, rtol=0.0)
        else:
            assert actual == expected


def test_flatten_phase_a_observation_uses_obs_role_flag_and_skills_only():
    obs = {
        "obs": np.array([1.0, 2.0], dtype=np.float32),
        "role_flag": np.array([-1.0], dtype=np.float32),
        "skills": np.array([3.0, 4.0, 5.0], dtype=np.float32),
        "action_mask": np.ones((2, 3), dtype=np.int8),
        "intent_index": np.array([99.0], dtype=np.float32),
    }

    flat = flatten_phase_a_observation(obs)

    np.testing.assert_allclose(
        flat,
        np.array([1.0, 2.0, -1.0, 3.0, 4.0, 5.0], dtype=np.float32),
    )


def test_flatten_phase_a_observation_batch_concatenates_expected_keys():
    obs_batch = {
        "obs": np.array([[1.0, 2.0], [10.0, 20.0]], dtype=np.float32),
        "role_flag": np.array([[1.0], [-1.0]], dtype=np.float32),
        "skills": np.array([[3.0, 4.0], [30.0, 40.0]], dtype=np.float32),
        "action_mask": np.ones((2, 2, 3), dtype=np.int8),
    }

    flat = flatten_phase_a_observation_batch(obs_batch)

    np.testing.assert_allclose(
        flat,
        np.array(
            [
                [1.0, 2.0, 1.0, 3.0, 4.0],
                [10.0, 20.0, -1.0, 30.0, 40.0],
            ],
            dtype=np.float32,
        ),
    )


def test_extract_phase_a_training_action_mask_supports_single_and_batched_inputs():
    single = np.arange(24, dtype=np.int8).reshape(6, 4)
    batched = np.arange(48, dtype=np.int8).reshape(2, 6, 4)
    ids = [0, 2, 4]

    single_out = extract_phase_a_training_action_mask(single, ids)
    batched_out = extract_phase_a_training_action_mask(batched, ids)

    np.testing.assert_array_equal(single_out, single[ids])
    np.testing.assert_array_equal(batched_out, batched[:, ids, :])


def test_phase_a_masked_forward_never_selects_illegal_action():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    spec = PhaseAPolicySpec(
        flat_obs_dim=2,
        training_player_count=2,
        action_dim_per_player=4,
        total_action_dim=8,
        hidden_dims=(),
    )
    forward = _build_jitted_policy_forward(jax, jnp, spec)

    weights = jnp.zeros((2, 8), dtype=jnp.float32)
    bias = jnp.array(
        [0.0, 10.0, -1.0, -2.0, 7.0, 0.0, -3.0, -4.0],
        dtype=jnp.float32,
    )
    params = ((weights, bias),)
    flat_obs = jnp.zeros((1, 2), dtype=jnp.float32)
    action_mask = jnp.array(
        [[[1, 0, 1, 0], [0, 1, 0, 0]]],
        dtype=jnp.int8,
    )

    out = forward(params, flat_obs, action_mask, jax.random.PRNGKey(0))

    deterministic = np.asarray(out["deterministic_actions"], dtype=np.int32)
    probs = np.asarray(out["probs"], dtype=np.float32)

    np.testing.assert_array_equal(deterministic, np.array([[0, 1]], dtype=np.int32))
    assert probs[0, 0, 1] == pytest.approx(0.0)
    assert probs[0, 0, 3] == pytest.approx(0.0)
    assert probs[0, 1, 0] == pytest.approx(0.0)
    assert probs[0, 1, 2] == pytest.approx(0.0)
    assert probs[0, 1, 3] == pytest.approx(0.0)


def test_phase_a_masked_forward_falls_back_to_noop_when_mask_is_empty():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    spec = PhaseAPolicySpec(
        flat_obs_dim=1,
        training_player_count=1,
        action_dim_per_player=3,
        total_action_dim=3,
        hidden_dims=(),
    )
    forward = _build_jitted_policy_forward(jax, jnp, spec)

    params = ((jnp.zeros((1, 3), dtype=jnp.float32), jnp.array([3.0, 2.0, 1.0], dtype=jnp.float32)),)
    flat_obs = jnp.zeros((1, 1), dtype=jnp.float32)
    action_mask = jnp.zeros((1, 1, 3), dtype=jnp.int8)

    out = forward(params, flat_obs, action_mask, jax.random.PRNGKey(1))

    deterministic = np.asarray(out["deterministic_actions"], dtype=np.int32)
    probs = np.asarray(out["probs"], dtype=np.float32)

    np.testing.assert_array_equal(deterministic, np.array([[0]], dtype=np.int32))
    assert probs[0, 0, 0] == pytest.approx(1.0)
    assert probs[0, 0, 1] == pytest.approx(0.0)
    assert probs[0, 0, 2] == pytest.approx(0.0)
