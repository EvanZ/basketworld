from __future__ import annotations

import numpy as np
import pytest

from basketworld.envs.basketworld_env_v2 import HexagonBasketballEnv, Team
from basketworld_jax.env.minimal import (
    build_kernel_static_from_env,
    reset_batch_minimal,
    step_batch_minimal,
)


def _make_intent_env(**overrides):
    params = {
        "players": 3,
        "court_rows": 9,
        "court_cols": 8,
        "shot_clock": 24,
        "min_shot_clock": 14,
        "training_team": Team.OFFENSE,
        "pass_mode": "pointer_targeted",
        "allow_dunks": True,
        "shot_pressure_enabled": False,
        "defender_pressure_turnover_chance": 0.0,
        "illegal_defense_enabled": False,
        "offensive_three_seconds_enabled": False,
        "enable_intent_learning": True,
        "enable_defense_intent_learning": True,
        "num_intents": 5,
        "intent_commitment_steps": 3,
        "intent_null_prob": 0.0,
        "defense_intent_null_prob": 0.0,
        "intent_visible_to_defense_prob": 1.0,
    }
    params.update(overrides)
    return HexagonBasketballEnv(**params)


def test_jax_reset_samples_intent_state_deterministically():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    env = _make_intent_env()
    static = build_kernel_static_from_env(env, jnp)
    keys = jax.random.split(jax.random.PRNGKey(123), 8)

    state_a = reset_batch_minimal(static, keys, jax, jnp)
    state_b = reset_batch_minimal(static, keys, jax, jnp)

    for field in (
        "intent_index",
        "intent_active",
        "intent_age",
        "intent_commitment_remaining",
        "intent_visible_to_defense",
        "defense_intent_index",
        "defense_intent_active",
        "defense_intent_age",
        "defense_intent_commitment_remaining",
    ):
        np.testing.assert_array_equal(np.asarray(getattr(state_a, field)), np.asarray(getattr(state_b, field)))

    assert np.asarray(state_a.intent_active).tolist() == [1] * 8
    assert np.asarray(state_a.defense_intent_active).tolist() == [1] * 8
    assert np.asarray(state_a.intent_visible_to_defense).tolist() == [1] * 8
    assert np.asarray(state_a.intent_age).tolist() == [0] * 8
    assert np.asarray(state_a.defense_intent_age).tolist() == [0] * 8
    assert np.asarray(state_a.intent_commitment_remaining).tolist() == [3] * 8
    assert np.asarray(state_a.defense_intent_commitment_remaining).tolist() == [3] * 8
    assert np.all(np.asarray(state_a.intent_index) >= 0)
    assert np.all(np.asarray(state_a.intent_index) < 5)
    assert np.all(np.asarray(state_a.defense_intent_index) >= 0)
    assert np.all(np.asarray(state_a.defense_intent_index) < 5)


def test_jax_reset_zeroes_null_intent_commitment():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    env = _make_intent_env(
        intent_null_prob=1.0,
        defense_intent_null_prob=1.0,
        intent_visible_to_defense_prob=0.0,
    )
    static = build_kernel_static_from_env(env, jnp)
    state = reset_batch_minimal(static, jax.random.split(jax.random.PRNGKey(321), 4), jax, jnp)

    assert np.asarray(state.intent_active).tolist() == [0] * 4
    assert np.asarray(state.defense_intent_active).tolist() == [0] * 4
    assert np.asarray(state.intent_commitment_remaining).tolist() == [0] * 4
    assert np.asarray(state.defense_intent_commitment_remaining).tolist() == [0] * 4


def test_jax_step_advances_and_expires_intent_state():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    env = _make_intent_env(intent_commitment_steps=2)
    static = build_kernel_static_from_env(env, jnp)
    state = reset_batch_minimal(static, jax.random.split(jax.random.PRNGKey(456), 1), jax, jnp)
    actions = jnp.zeros((1, env.n_players), dtype=jnp.int32)

    out = step_batch_minimal(
        static,
        state,
        actions,
        jax.random.split(jax.random.PRNGKey(1), 1),
        jax,
        jnp,
    )
    state = out.state
    assert np.asarray(state.intent_active).tolist() == [1]
    assert np.asarray(state.defense_intent_active).tolist() == [1]
    assert np.asarray(state.intent_age).tolist() == [1]
    assert np.asarray(state.defense_intent_age).tolist() == [1]
    assert np.asarray(state.intent_commitment_remaining).tolist() == [1]
    assert np.asarray(state.defense_intent_commitment_remaining).tolist() == [1]

    out = step_batch_minimal(
        static,
        state,
        actions,
        jax.random.split(jax.random.PRNGKey(2), 1),
        jax,
        jnp,
    )
    state = out.state
    assert np.asarray(state.intent_active).tolist() == [1]
    assert np.asarray(state.defense_intent_active).tolist() == [1]
    assert np.asarray(state.intent_age).tolist() == [2]
    assert np.asarray(state.defense_intent_age).tolist() == [2]
    assert np.asarray(state.intent_commitment_remaining).tolist() == [0]
    assert np.asarray(state.defense_intent_commitment_remaining).tolist() == [0]

    out = step_batch_minimal(
        static,
        state,
        actions,
        jax.random.split(jax.random.PRNGKey(3), 1),
        jax,
        jnp,
    )
    state = out.state
    assert np.asarray(state.intent_active).tolist() == [0]
    assert np.asarray(state.defense_intent_active).tolist() == [0]
    assert np.asarray(state.intent_age).tolist() == [2]
    assert np.asarray(state.defense_intent_age).tolist() == [2]
    assert np.asarray(state.intent_commitment_remaining).tolist() == [0]
    assert np.asarray(state.defense_intent_commitment_remaining).tolist() == [0]
