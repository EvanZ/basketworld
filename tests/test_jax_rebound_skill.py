import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from basketworld_jax.env.minimal import (
    SHOT_TYPE_TWO,
    build_rebound_observation_features_batch,
    build_shot_profile_batch,
    build_token_observation_components_batch,
    reset_batch_minimal,
    sample_state_batch,
)
from basketworld_jax.train.main import parse_args


def _sample_static_state(extra_args=None):
    args = parse_args([
        "--kernel-batch-size",
        "1",
        "--players",
        "2",
        "--use-set-obs",
        "true",
        *(extra_args or []),
    ])
    return sample_state_batch(args, jnp)


def test_rebound_skill_reset_defaults_to_zero_and_obs_label_slot_exists():
    static, _ = _sample_static_state()
    state = reset_batch_minimal(static, jnp.asarray(jax.random.split(jax.random.PRNGKey(0), 2)), jax, jnp)

    assert np.allclose(np.asarray(state.rebound_skill), 0.0)
    players, globals_vec, role = build_token_observation_components_batch(
        static,
        state,
        static.training_role_flag,
        jnp,
    )
    assert players.shape[-1] == 18


def test_rebound_skill_std_samples_nonzero_values():
    static, _ = _sample_static_state(["--rebound-skill-std", "0.5"])
    state = reset_batch_minimal(static, jnp.asarray(jax.random.split(jax.random.PRNGKey(1), 8)), jax, jnp)

    assert not np.allclose(np.asarray(state.rebound_skill), 0.0)


def test_rebound_skill_weight_zero_matches_distance_only_equal_distance():
    static, state = _sample_static_state(["--rebound-skill-weight", "0.0"])
    coords = np.asarray(static.cell_coords, dtype=np.int32)
    target_idx = 0
    target = coords[target_idx]
    n_players = int(static.role_encoding.shape[0])
    rebound_probs = np.zeros(np.asarray(static.rebound_target_probs).shape, dtype=np.float32)
    rebound_probs[SHOT_TYPE_TWO, target_idx, target_idx] = 1.0
    static = static._replace(
        enable_rebounds=jnp.asarray(1, dtype=jnp.int8),
        rebound_target_probs=jnp.asarray(rebound_probs, dtype=jnp.float32),
        rebound_winner_distance_weight=jnp.asarray(1.0, dtype=jnp.float32),
        rebound_winner_temperature=jnp.asarray(1.0, dtype=jnp.float32),
        rebound_skill_weight=jnp.asarray(0.0, dtype=jnp.float32),
    )
    state = state._replace(
        positions=jnp.asarray([[target for _ in range(n_players)]], dtype=jnp.int32),
        ball_holder=jnp.asarray([0], dtype=jnp.int32),
        rebound_skill=jnp.asarray([[0.0, 5.0, 0.0, 0.0]], dtype=jnp.float32),
    )

    shot_profile = build_shot_profile_batch(static, state, jnp)
    features = build_rebound_observation_features_batch(static, state, shot_profile, jnp)
    probs = np.asarray(features["win_prob"])[0]

    assert probs[0] == pytest.approx(probs[1], abs=1e-6)


def test_higher_rebound_skill_increases_equal_distance_win_probability():
    static, state = _sample_static_state(["--rebound-skill-weight", "1.0"])
    coords = np.asarray(static.cell_coords, dtype=np.int32)
    target_idx = 0
    target = coords[target_idx]
    n_players = int(static.role_encoding.shape[0])
    rebound_probs = np.zeros(np.asarray(static.rebound_target_probs).shape, dtype=np.float32)
    rebound_probs[SHOT_TYPE_TWO, target_idx, target_idx] = 1.0
    static = static._replace(
        enable_rebounds=jnp.asarray(1, dtype=jnp.int8),
        rebound_target_probs=jnp.asarray(rebound_probs, dtype=jnp.float32),
        rebound_winner_distance_weight=jnp.asarray(1.0, dtype=jnp.float32),
        rebound_winner_temperature=jnp.asarray(1.0, dtype=jnp.float32),
        rebound_skill_weight=jnp.asarray(1.0, dtype=jnp.float32),
    )
    state = state._replace(
        positions=jnp.asarray([[target for _ in range(n_players)]], dtype=jnp.int32),
        ball_holder=jnp.asarray([0], dtype=jnp.int32),
        rebound_skill=jnp.asarray([[0.0, 1.0, 0.0, 0.0]], dtype=jnp.float32),
    )

    shot_profile = build_shot_profile_batch(static, state, jnp)
    features = build_rebound_observation_features_batch(static, state, shot_profile, jnp)
    probs = np.asarray(features["win_prob"])[0]

    assert probs[1] > probs[0]
