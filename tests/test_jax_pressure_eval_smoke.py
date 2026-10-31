"""Compiled evaluation paths must preserve the exact same live-state metrics."""

import numpy as np
import pytest


def test_compiled_role_and_deploy_esv_match_training_diagnostic():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from tests.test_jax_pressure_metrics import fixture
    from basketworld_jax.env import (
        build_policy_observation_batch,
        build_action_masks_batch,
    )
    from basketworld_jax.models import build_actor_critic_spec, init_actor_critic_params
    from basketworld_jax.train.pressure_metrics import (
        build_pressure_diagnostics,
        summarize_pressure_diagnostics,
    )
    from basketworld_jax.train.runtime import (
        build_compiled_eval_runner,
        build_compiled_deploy_eval_runner,
        summarize_deploy_eval_outputs,
    )

    static, state = fixture()
    obs = build_policy_observation_batch(
        static, state, jnp, model_type="mlp", multi_possession_features=True
    )
    masks = build_action_masks_batch(static, state, jnp)[:, static.offense_ids, :]
    spec = build_actor_critic_spec(
        np.asarray(obs), np.asarray(masks), (16,), multi_possession_features=True
    )
    params = init_actor_critic_params(jax, jnp, spec, seed=80)
    expected = summarize_pressure_diagnostics(
        build_pressure_diagnostics(static, state, jnp.ones(4), jnp)
    )
    _, trace = build_compiled_eval_runner(jax, jnp, spec)(
        static, state, params, jax.random.PRNGKey(80), 1, False, False, 3
    )
    role_summary = summarize_pressure_diagnostics(
        jax.device_get(trace.pressure_diagnostics)
    )
    assert role_summary == pytest.approx(expected)
    output = build_compiled_deploy_eval_runner(jax, jnp, spec)(
        static, state, params, jax.random.PRNGKey(80), 1, False, False, 3
    )
    jax.block_until_ready(output)
    deploy_summary = summarize_deploy_eval_outputs([output], batch_size=4)
    assert {
        key: value
        for key, value in deploy_summary.items()
        if key.startswith("pressure/")
    } == pytest.approx(expected)
