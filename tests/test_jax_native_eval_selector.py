from __future__ import annotations

import numpy as np
import pytest

from basketworld.envs.basketworld_env_v2 import HexagonBasketballEnv, Team
from basketworld_jax.env.minimal import (
    TOKEN_OBS_GLOBAL_DIM,
    TOKEN_OBS_PLAYER_DIM,
    build_action_masks_batch,
    build_kernel_static_from_env,
    build_policy_observation_batch_with_role_flag,
    reset_batch_minimal,
)
from basketworld_jax.eval.native import (
    _build_native_eval_runner,
    _selector_eval_settings,
)
from basketworld_jax.models import build_actor_critic_spec, init_actor_critic_params


def _make_intent_env():
    return HexagonBasketballEnv(
        players=3,
        court_rows=9,
        court_cols=8,
        shot_clock=8,
        min_shot_clock=8,
        training_team=Team.OFFENSE,
        pass_mode="pointer_targeted",
        allow_dunks=True,
        shot_pressure_enabled=False,
        defender_pressure_turnover_chance=0.0,
        illegal_defense_enabled=False,
        offensive_three_seconds_enabled=False,
        enable_intent_learning=True,
        enable_defense_intent_learning=True,
        num_intents=5,
        intent_commitment_steps=3,
        intent_null_prob=0.0,
        defense_intent_null_prob=0.0,
        intent_visible_to_defense_prob=1.0,
    )


def test_native_selector_eval_settings_use_checkpoint_update_schedule():
    spec = build_actor_critic_spec(
        np.zeros((1, (6 * TOKEN_OBS_PLAYER_DIM) + TOKEN_OBS_GLOBAL_DIM + 1), dtype=np.float32),
        np.ones((1, 3, 14), dtype=np.int8),
        hidden_dims=(),
        model_type="attention",
        token_player_count=6,
        token_dim=TOKEN_OBS_PLAYER_DIM,
        global_dim=TOKEN_OBS_GLOBAL_DIM,
        intent_embedding_enabled=True,
        num_intents=5,
        intent_selector_enabled=True,
        intent_selector_hidden_dim=8,
    )
    settings = _selector_eval_settings(
        payload={"update_index": 50},
        spec=spec,
        training_params={
            "intent_selector_enabled": True,
            "intent_selector_mode": "integrated",
            "intent_selector_alpha_start": 0.0,
            "intent_selector_alpha_end": 0.5,
            "intent_selector_alpha_warmup_steps": 0,
            "intent_selector_alpha_ramp_steps": 100,
            "intent_selector_eps_start": 0.5,
            "intent_selector_eps_end": 0.1,
            "intent_selector_eps_warmup_steps": 0,
            "intent_selector_eps_ramp_steps": 100,
            "intent_selector_multiselect_enabled": True,
            "intent_selector_min_play_steps": 7,
        },
        intent_selection_mode="learned_sample",
    )

    assert settings["enabled"] is True
    assert settings["mode_label"] == "learned_sample"
    assert settings["training_alpha"] == pytest.approx(0.25)
    assert settings["eval_alpha"] == pytest.approx(1.0)
    assert settings["eps"] == pytest.approx(0.3)
    assert settings["multiselect_enabled"] is True
    assert settings["min_play_steps"] == 7


def test_native_selector_eval_settings_prefers_checkpoint_selector_metadata_over_session_defaults():
    spec = build_actor_critic_spec(
        np.zeros((1, (6 * TOKEN_OBS_PLAYER_DIM) + TOKEN_OBS_GLOBAL_DIM + 1), dtype=np.float32),
        np.ones((1, 3, 14), dtype=np.int8),
        hidden_dims=(),
        model_type="attention",
        token_player_count=6,
        token_dim=TOKEN_OBS_PLAYER_DIM,
        global_dim=TOKEN_OBS_GLOBAL_DIM,
        intent_embedding_enabled=True,
        num_intents=5,
        intent_selector_enabled=True,
        intent_selector_hidden_dim=8,
    )
    settings = _selector_eval_settings(
        payload={
            "update_index": 10,
            "trainer_config": {
                "intent_selector_enabled": True,
                "intent_selector_mode": "integrated",
                "intent_selector_multiselect_enabled": True,
                "intent_selector_min_play_steps": 6,
            },
            "policy_spec": {"intent_selector_enabled": True},
        },
        spec=spec,
        training_params={
            "intent_selector_enabled": False,
            "intent_selector_mode": "callback",
            "intent_selector_multiselect_enabled": False,
            "intent_selector_min_play_steps": 2,
        },
        intent_selection_mode="learned_sample",
    )

    assert settings["enabled"] is True
    assert settings["spec_enabled"] is True
    assert settings["config_enabled"] is True
    assert settings["selector_mode"] == "integrated"
    assert settings["disabled_reason"] is None
    assert settings["multiselect_enabled"] is True
    assert settings["min_play_steps"] == 6


def test_native_selector_eval_settings_treats_callback_mode_as_legacy_jax_integrated():
    spec = build_actor_critic_spec(
        np.zeros((1, (6 * TOKEN_OBS_PLAYER_DIM) + TOKEN_OBS_GLOBAL_DIM + 1), dtype=np.float32),
        np.ones((1, 3, 14), dtype=np.int8),
        hidden_dims=(),
        model_type="attention",
        token_player_count=6,
        token_dim=TOKEN_OBS_PLAYER_DIM,
        global_dim=TOKEN_OBS_GLOBAL_DIM,
        intent_embedding_enabled=True,
        num_intents=5,
        intent_selector_enabled=True,
        intent_selector_hidden_dim=8,
    )
    settings = _selector_eval_settings(
        payload={
            "update_index": 10,
            "trainer_config": {
                "intent_selector_enabled": True,
                "intent_selector_mode": "callback",
            },
            "policy_spec": {"intent_selector_enabled": True},
        },
        spec=spec,
        training_params={"intent_selector_mode": "callback"},
        intent_selection_mode="learned_sample",
    )

    assert settings["enabled"] is True
    assert settings["selector_mode"] == "integrated"
    assert settings["disabled_reason"] is None


def test_native_selector_eval_settings_uses_policy_spec_when_training_params_are_stale():
    spec = build_actor_critic_spec(
        np.zeros((1, (6 * TOKEN_OBS_PLAYER_DIM) + TOKEN_OBS_GLOBAL_DIM + 1), dtype=np.float32),
        np.ones((1, 3, 14), dtype=np.int8),
        hidden_dims=(),
        model_type="attention",
        token_player_count=6,
        token_dim=TOKEN_OBS_PLAYER_DIM,
        global_dim=TOKEN_OBS_GLOBAL_DIM,
        intent_embedding_enabled=True,
        num_intents=5,
        intent_selector_enabled=True,
        intent_selector_hidden_dim=8,
    )
    settings = _selector_eval_settings(
        payload={
            "update_index": 10,
            "policy_spec": {"intent_selector_enabled": True},
        },
        spec=spec,
        training_params={
            "intent_selector_enabled": False,
            "intent_selector_mode": "callback",
        },
        intent_selection_mode="learned_sample",
    )

    assert settings["enabled"] is True
    assert settings["selector_mode"] == "integrated"
    assert settings["disabled_reason"] is None

def test_native_eval_runner_applies_selector_at_episode_start():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    env = _make_intent_env()
    static = build_kernel_static_from_env(env, jnp)
    initial_state = reset_batch_minimal(
        static,
        jax.random.split(jax.random.PRNGKey(10), 4),
        jax,
        jnp,
    )
    flat_obs = build_policy_observation_batch_with_role_flag(
        static,
        initial_state,
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp,
        model_type="attention",
    )
    action_mask = build_action_masks_batch(static, initial_state, jnp)[:, static.offense_ids, :]
    spec = build_actor_critic_spec(
        np.asarray(flat_obs),
        np.asarray(action_mask),
        hidden_dims=(),
        model_type="attention",
        token_player_count=env.n_players,
        token_dim=TOKEN_OBS_PLAYER_DIM,
        global_dim=TOKEN_OBS_GLOBAL_DIM,
        attention_embed_dim=16,
        attention_num_heads=4,
        attention_token_mlp_dim=16,
        attention_num_cls_tokens=2,
        intent_embedding_enabled=True,
        num_intents=env.num_intents,
        intent_selector_enabled=True,
        intent_selector_hidden_dim=8,
    )
    params = init_actor_critic_params(jax, jnp, spec, seed=123)
    runner = _build_native_eval_runner(jax, jnp, spec)

    trace = runner(
        static,
        initial_state,
        params,
        params,
        jax.random.PRNGKey(99),
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.asarray(-1.0, dtype=jnp.float32),
        1,
        True,
        True,
        True,
        0.0,
        False,
        3,
        1,
    )
    trace = jax.device_get(trace)

    np.testing.assert_array_equal(np.asarray(trace["selector_applied"][0]), np.ones(4, dtype=np.int8))
    np.testing.assert_array_equal(np.asarray(trace["selector_used"][0]), np.ones(4, dtype=np.int8))
    np.testing.assert_array_equal(
        np.asarray(trace["selector_boundary_episode_start"][0]),
        np.ones(4, dtype=np.int8),
    )
    np.testing.assert_array_equal(np.asarray(trace["selector_intent_index"][0]), np.zeros(4, dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(trace["intent_index"][0]), np.zeros(4, dtype=np.int32))
