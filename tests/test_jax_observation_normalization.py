from __future__ import annotations

import numpy as np
import pytest

from basketworld.envs.basketworld_env_v2 import HexagonBasketballEnv
from basketworld_jax.env.minimal import (
    build_kernel_static_from_env,
    build_observation_vector_batch,
    build_token_observation_components_batch,
    snapshot_state_from_env,
    stack_state_snapshots,
)


def _make_env() -> HexagonBasketballEnv:
    env = HexagonBasketballEnv(
        players=3,
        render_mode=None,
        pass_mode="pointer_targeted",
        shot_clock=24,
        min_shot_clock=14,
    )
    env.reset(seed=123)
    env.shot_clock = 12
    return env


def test_jax_flat_observation_normalizes_shot_clock_feature_only():
    pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    env = _make_env()
    static = build_kernel_static_from_env(env, xp=jnp)
    state = stack_state_snapshots([snapshot_state_from_env(env)], xp=jnp)

    obs = np.asarray(build_observation_vector_batch(static, state, jnp))[0]
    n_players = int(env.n_players)
    shot_clock_feature_idx = (n_players * 2) + n_players

    assert int(np.asarray(state.shot_clock)[0]) == 12
    assert obs[shot_clock_feature_idx] == pytest.approx(12.0 / 24.0, abs=1e-6)


def test_jax_token_observation_normalizes_shot_clock_global_only():
    pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    env = _make_env()
    static = build_kernel_static_from_env(env, xp=jnp)
    state = stack_state_snapshots([snapshot_state_from_env(env)], xp=jnp)

    _, globals_vec, _ = build_token_observation_components_batch(
        static,
        state,
        static.training_role_flag,
        jnp,
    )
    globals_np = np.asarray(globals_vec)[0]

    assert int(np.asarray(state.shot_clock)[0]) == 12
    assert globals_np[0] == pytest.approx(12.0 / 24.0, abs=1e-6)
