import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from basketworld_jax.env.minimal import (
    REBOUND_CONTEST_MODE_LOCAL,
    REBOUND_SKILL_SAMPLING_ONE_HIGH_PER_TEAM,
    TOKEN_OBS_GLOBAL_DIM,
    TOKEN_OBS_PLAYER_DIM,
    _local_rebound_contest_mask_from_distances,
    build_rebound_observation_features_batch,
    build_shot_profile_batch,
    build_token_observation_components_batch,
    reset_batch_minimal,
    sample_state_batch,
)
from basketworld_jax.train.main import parse_args
from basketworld_jax.eval.native import _apply_native_custom_setup


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


def _single_target_rebound_probs(static, holder_idx: int, target_idx: int):
    rebound_probs = np.zeros(np.asarray(static.rebound_target_probs).shape, dtype=np.float32)
    rebound_probs[:, holder_idx, target_idx] = 1.0
    return rebound_probs


def test_rebound_skill_reset_defaults_to_zero_and_obs_slots_exist():
    static, _ = _sample_static_state()
    state = reset_batch_minimal(static, jnp.asarray(jax.random.split(jax.random.PRNGKey(0), 2)), jax, jnp)

    assert np.allclose(np.asarray(state.rebound_skill), 0.0)
    assert np.allclose(np.asarray(state.rebound_skill_specialist), 0.0)
    players, globals_vec, role = build_token_observation_components_batch(
        static,
        state,
        static.training_role_flag,
        jnp,
    )
    assert players.shape[-1] == TOKEN_OBS_PLAYER_DIM == 18
    assert globals_vec.shape[-1] == TOKEN_OBS_GLOBAL_DIM == 7


def test_rebound_skill_std_samples_nonzero_values():
    static, _ = _sample_static_state(["--rebound-skill-std", "0.5"])
    state = reset_batch_minimal(static, jnp.asarray(jax.random.split(jax.random.PRNGKey(1), 8)), jax, jnp)

    assert not np.allclose(np.asarray(state.rebound_skill), 0.0)
    assert np.allclose(np.asarray(state.rebound_skill_specialist), 0.0)


def test_one_high_per_team_rebound_skill_sampling_disables_gaussian_noise():
    static, _ = _sample_static_state([
        "--rebound-skill-std",
        "99.0",
        "--rebound-skill-sampling-mode",
        "one_high_per_team",
        "--rebound-skill-high",
        "1.0",
        "--rebound-skill-low",
        "-0.25",
    ])
    state = reset_batch_minimal(static, jnp.asarray(jax.random.split(jax.random.PRNGKey(7), 16)), jax, jnp)

    skills = np.asarray(state.rebound_skill)
    specialists = np.asarray(state.rebound_skill_specialist)
    offense_ids = np.asarray(static.offense_ids, dtype=np.int32)
    defense_ids = np.asarray(static.defense_ids, dtype=np.int32)
    for row_skills, row_specialists in zip(skills, specialists):
        assert np.count_nonzero(np.isclose(row_skills[offense_ids], 1.0)) == 1
        assert np.count_nonzero(np.isclose(row_skills[defense_ids], 1.0)) == 1
        assert np.count_nonzero(np.isclose(row_specialists[offense_ids], 1.0)) == 1
        assert np.count_nonzero(np.isclose(row_specialists[defense_ids], 1.0)) == 1
        assert np.allclose(row_skills[row_specialists == 0.0], -0.25)



def test_custom_eval_rebound_specialist_respects_sampling_mode():
    static, _ = _sample_static_state()
    keys = jnp.asarray(jax.random.split(jax.random.PRNGKey(17), 2))
    state = reset_batch_minimal(static, keys, jax, jnp)
    batch_size = int(state.rebound_skill.shape[0])
    n_players = int(static.role_encoding.shape[0])
    values = np.linspace(-0.5, 1.0, n_players, dtype=np.float32)
    custom_setup = {"rebound_skills": values.tolist()}

    gaussian_state = _apply_native_custom_setup(static, state, custom_setup, batch_size, jnp)
    assert np.asarray(gaussian_state.rebound_skill)[0].tolist() == pytest.approx(values.tolist())
    assert np.allclose(np.asarray(gaussian_state.rebound_skill_specialist), 0.0)

    one_high_static = static._replace(
        rebound_skill_sampling_mode=jnp.asarray(REBOUND_SKILL_SAMPLING_ONE_HIGH_PER_TEAM, dtype=jnp.int32),
    )
    one_high_state = _apply_native_custom_setup(one_high_static, state, custom_setup, batch_size, jnp)
    assert np.asarray(one_high_state.rebound_skill)[0].tolist() == pytest.approx(values.tolist())
    assert np.asarray(one_high_state.rebound_skill_specialist)[0].tolist() == pytest.approx(
        (values > 0.0).astype(np.float32).tolist()
    )


def test_rebound_features_emit_centroid_and_player_distance_to_centroid():
    static, state = _sample_static_state()
    coords = np.asarray(static.cell_coords, dtype=np.int32)
    distance_matrix = np.asarray(static.cell_distance_matrix, dtype=np.int32)
    holder_idx = 0
    target_idx = int(np.where(distance_matrix[holder_idx] > 3)[0][0])
    n_players = int(static.role_encoding.shape[0])
    rebound_probs = _single_target_rebound_probs(static, holder_idx, target_idx)
    static = static._replace(
        enable_rebounds=jnp.asarray(1, dtype=jnp.int8),
        rebound_target_probs=jnp.asarray(rebound_probs, dtype=jnp.float32),
        rebound_target_uniform_mix=jnp.asarray(0.0, dtype=jnp.float32),
        rebound_target_temperature=jnp.asarray(1.0, dtype=jnp.float32),
    )
    positions = np.asarray(
        [[coords[holder_idx], coords[target_idx]] + [coords[holder_idx] for _ in range(n_players - 2)]],
        dtype=np.int32,
    )
    state = state._replace(
        positions=jnp.asarray(positions, dtype=jnp.int32),
        ball_holder=jnp.asarray([0], dtype=jnp.int32),
        rebound_skill=jnp.zeros((1, n_players), dtype=jnp.float32),
    )

    shot_profile = build_shot_profile_batch(static, state, jnp)
    features = build_rebound_observation_features_batch(static, state, shot_profile, jnp)

    norm_den = float(np.asarray(static.court_norm_den))
    assert "win_prob" not in features
    assert "orb_prob" not in features
    assert float(np.asarray(features["expected_target_q"])[0]) == pytest.approx(float(coords[target_idx, 0]) / norm_den, abs=1.0e-5)
    assert float(np.asarray(features["expected_target_r"])[0]) == pytest.approx(float(coords[target_idx, 1]) / norm_den, abs=1.0e-5)
    distances = np.asarray(features["dist_to_expected_target"])[0]
    assert distances[1] == pytest.approx(0.0, abs=1.0e-6)
    assert distances[0] == pytest.approx(float(distance_matrix[holder_idx, target_idx]) / norm_den, abs=1.0e-6)


def test_rebound_skill_and_specialist_are_last_token_player_features():
    static, state = _sample_static_state()
    n_players = int(static.role_encoding.shape[0])
    skills = np.linspace(-1.0, 1.0, n_players, dtype=np.float32)
    specialists = (skills > 0.0).astype(np.float32)
    state = state._replace(
        rebound_skill=jnp.asarray([skills], dtype=jnp.float32),
        rebound_skill_specialist=jnp.asarray([specialists], dtype=jnp.float32),
    )

    players, _globals_vec, _role = build_token_observation_components_batch(
        static,
        state,
        static.training_role_flag,
        jnp,
    )

    assert np.asarray(players)[0, :, -2].tolist() == pytest.approx(skills.tolist())
    assert np.asarray(players)[0, :, -1].tolist() == pytest.approx(specialists.tolist())


def test_local_rebound_contest_uses_fixed_target_radius_only():
    static, _ = _sample_static_state([
        "--rebound-contest-mode",
        "local_contest",
        "--rebound-contest-radius",
        "1",
    ])
    static = static._replace(rebound_contest_mode=jnp.asarray(REBOUND_CONTEST_MODE_LOCAL, dtype=jnp.int32))
    target_distances = jnp.asarray([1, 2, 2, 4], dtype=jnp.int32)
    player_player_distances = jnp.asarray(
        [
            [0, 1, 1, 4],
            [1, 0, 2, 4],
            [1, 2, 0, 2],
            [4, 4, 2, 0],
        ],
        dtype=jnp.int32,
    )

    eligible, radius_used, fallback = _local_rebound_contest_mask_from_distances(
        static,
        target_distances,
        player_player_distances,
        jnp,
    )

    assert int(np.asarray(radius_used)) == 1
    assert bool(np.asarray(fallback)) is False
    assert np.asarray(eligible).tolist() == [True, False, False, False]


def test_local_rebound_contest_falls_back_when_no_radius_has_eligible_player():
    static, _ = _sample_static_state([
        "--rebound-contest-mode",
        "local_contest",
        "--rebound-contest-radius",
        "0",
    ])
    static = static._replace(rebound_contest_mode=jnp.asarray(REBOUND_CONTEST_MODE_LOCAL, dtype=jnp.int32))
    target_distances = jnp.asarray([2, 3, 4, 5], dtype=jnp.int32)
    player_player_distances = jnp.asarray(
        [
            [0, 1, 2, 3],
            [1, 0, 1, 2],
            [2, 1, 0, 1],
            [3, 2, 1, 0],
        ],
        dtype=jnp.int32,
    )

    eligible, radius_used, fallback = _local_rebound_contest_mask_from_distances(
        static,
        target_distances,
        player_player_distances,
        jnp,
    )

    assert int(np.asarray(radius_used)) == -1
    assert bool(np.asarray(fallback)) is True
    assert np.asarray(eligible).tolist() == [True, True, True, True]


def test_rebound_obs_top_n_targets_truncates_centroid_features():
    static, state = _sample_static_state([
        "--rebound-contest-mode",
        "local_contest",
        "--rebound-obs-top-n-targets",
        "1",
    ])
    coords = np.asarray(static.cell_coords, dtype=np.int32)
    distance_matrix = np.asarray(static.cell_distance_matrix, dtype=np.int32)
    target_a_idx = 0
    target_b_idx = int(np.where(distance_matrix[target_a_idx] > 3)[0][0])
    other_idx = int(np.where((distance_matrix[target_a_idx] > 2) & (distance_matrix[target_b_idx] > 2))[0][0])
    n_players = int(static.role_encoding.shape[0])
    rebound_probs = np.zeros(np.asarray(static.rebound_target_probs).shape, dtype=np.float32)
    rebound_probs[:, target_a_idx, target_a_idx] = 0.7
    rebound_probs[:, target_a_idx, target_b_idx] = 0.3
    positions = np.asarray(
        [[coords[target_a_idx], coords[target_b_idx]] + [coords[other_idx] for _ in range(n_players - 2)]],
        dtype=np.int32,
    )
    static = static._replace(
        enable_rebounds=jnp.asarray(1, dtype=jnp.int8),
        rebound_target_probs=jnp.asarray(rebound_probs, dtype=jnp.float32),
        rebound_target_uniform_mix=jnp.asarray(0.0, dtype=jnp.float32),
        rebound_target_temperature=jnp.asarray(1.0, dtype=jnp.float32),
        rebound_contest_mode=jnp.asarray(REBOUND_CONTEST_MODE_LOCAL, dtype=jnp.int32),
        rebound_obs_top_n_targets=jnp.asarray(1, dtype=jnp.int32),
    )
    state = state._replace(
        positions=jnp.asarray(positions, dtype=jnp.int32),
        ball_holder=jnp.asarray([0], dtype=jnp.int32),
        rebound_skill=jnp.zeros((1, n_players), dtype=jnp.float32),
    )

    shot_profile = build_shot_profile_batch(static, state, jnp)
    top_features = build_rebound_observation_features_batch(static, state, shot_profile, jnp)

    exact_static = static._replace(rebound_obs_top_n_targets=jnp.asarray(0, dtype=jnp.int32))
    exact_features = build_rebound_observation_features_batch(exact_static, state, shot_profile, jnp)

    norm_den = float(np.asarray(static.court_norm_den))
    exact_q = ((0.7 * float(coords[target_a_idx, 0])) + (0.3 * float(coords[target_b_idx, 0]))) / norm_den
    exact_r = ((0.7 * float(coords[target_a_idx, 1])) + (0.3 * float(coords[target_b_idx, 1]))) / norm_den
    assert float(np.asarray(exact_features["expected_target_q"])[0]) == pytest.approx(exact_q, abs=1.0e-5)
    assert float(np.asarray(exact_features["expected_target_r"])[0]) == pytest.approx(exact_r, abs=1.0e-5)
    assert float(np.asarray(top_features["expected_target_q"])[0]) == pytest.approx(float(coords[target_a_idx, 0]) / norm_den, abs=1.0e-5)
    assert float(np.asarray(top_features["expected_target_r"])[0]) == pytest.approx(float(coords[target_a_idx, 1]) / norm_den, abs=1.0e-5)
