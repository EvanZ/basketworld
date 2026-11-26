"""Issue #81: shot selection, fitted rebound rows, and continuing games."""

from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from analytics.rebound_sim.model import build_court
from analytics.rebound_sim.table_model import FittedReboundTableModel
from basketworld.domain import ActionType
from basketworld_jax.config import JaxEnvironmentConfig
from basketworld_jax.env.minimal import (
    GAME_PHASE_LIVE,
    SHOT_TYPE_DUNK,
    SHOT_TYPE_TWO,
    _build_legacy_phi_expected_points_batch,
    _compiled_rebound_target_probs,
    build_kernel_static_from_config,
    build_rebound_observation_features_batch,
    build_shot_profile_batch,
    reset_batch_minimal,
    step_batch_minimal,
)
from basketworld_jax.train.runtime import _build_shot_type_transition_metrics
from tests.test_rebound_sim_table_model import _write_order_sensitive_table_model


@pytest.fixture(scope="module")
def shot_setup(tmp_path_factory):
    court = build_court()
    model_dir = _write_order_sensitive_table_model(
        tmp_path_factory.mktemp("layup-table"), court
    )
    config = JaxEnvironmentConfig(
        players=2,
        allow_dunks=True,
        prefer_higher_percentage_shot=True,
        enable_rebounds=True,
        court_rows=court.spec.rows,
        court_cols=court.spec.cols,
        three_point_distance=court.spec.three_point_distance,
        three_point_short_distance=court.spec.three_point_short_distance,
        rebound_table_model_dir=str(model_dir),
        enable_multi_possession=True,
        multi_possession_limit=8,
        shot_pressure_enabled=True,
        defender_pressure_turnover_chance=0.0,
        illegal_defense_enabled=False,
        offensive_three_seconds_enabled=False,
        enable_phi_shaping=False,
        check_clearance=0,
    )
    static = build_kernel_static_from_config(config, xp=jnp)
    state = reset_batch_minimal(
        static, jax.random.split(jax.random.PRNGKey(19), 32), jax, jnp
    )
    coords = np.asarray(static.cell_coords)
    basket = np.asarray(static.basket_position)
    far = np.flatnonzero(np.asarray(static.basket_distance_by_cell) > 2)[:3]
    positions = np.broadcast_to(np.vstack([basket, coords[far]]), (32, 4, 2)).copy()
    state = state._replace(
        positions=jnp.asarray(positions),
        ball_holder=jnp.zeros(32, dtype=jnp.int32),
        offense_team=jnp.zeros(32, dtype=jnp.int8),
        game_phase=jnp.full((32,), GAME_PHASE_LIVE, dtype=jnp.int8),
        clearance_achieved=jnp.ones(32, dtype=jnp.int8),
    )
    return static, state


def with_skills(state, dunk, layup):
    return state._replace(
        dunk_pct=jnp.full_like(state.dunk_pct, dunk),
        layup_pct=jnp.full_like(state.layup_pct, layup),
    )


@pytest.mark.parametrize(
    "dunk,layup,allow,kind,prob",
    [
        (0.85, 0.65, True, SHOT_TYPE_DUNK, 0.85),
        (0.65, 0.85, True, SHOT_TYPE_TWO, 0.85),
        (0.75, 0.75, True, SHOT_TYPE_DUNK, 0.75),
        (0.85, 0.65, False, SHOT_TYPE_TWO, 0.65),
    ],
)
def test_under_basket_profile_and_phi_select_same_shot(
    shot_setup, dunk, layup, allow, kind, prob
):
    static, state = shot_setup
    static = static._replace(allow_dunks=jnp.asarray(allow, dtype=jnp.int8))
    state = with_skills(state, dunk, layup)
    profile = build_shot_profile_batch(static, state, jnp)
    assert np.all(np.asarray(profile["shot_type"])[:, 0] == kind)
    np.testing.assert_allclose(profile["base_probability"][:, 0], prob)
    np.testing.assert_allclose(
        profile["probability"][:, 0], prob * profile["pressure_multiplier"][:, 0]
    )
    np.testing.assert_array_equal(profile["shot_value"][:, 0], 2)
    np.testing.assert_allclose(
        _build_legacy_phi_expected_points_batch(static, state, jnp)[:, 0],
        profile["expected_points"][:, 0],
    )


@pytest.mark.parametrize("reverse_order", [False, True])
def test_basket_layup_row_is_equal_mixture_of_physical_neighbors(
    tmp_path, reverse_order
):
    court = build_court()
    model_dir = _write_order_sensitive_table_model(tmp_path, court)
    model = FittedReboundTableModel.load(model_dir, court=court)
    cells = sorted(court.cells, reverse=reverse_order)
    env = SimpleNamespace(
        enable_rebounds=True,
        prefer_higher_percentage_shot=True,
        rebound_table_model_dir=str(model_dir),
        court_height=court.spec.rows,
        court_width=court.spec.cols,
        basket_position=court.basket_axial,
        three_point_distance=court.spec.three_point_distance,
        three_point_short_distance=court.spec.three_point_short_distance,
    )
    dense = _compiled_rebound_target_probs(env, cells)
    basket_idx = cells.index(court.basket_axial)
    q, r = court.basket_axial
    neighbors = [
        (q + dq, r + dr)
        for dq, dr in ((1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1))
        if (q + dq, r + dr) in court.cell_index
    ]
    assert len(neighbors) > 1
    assert {
        model.shot_type_for_shot(court, court.cell_index[c]) for c in neighbors
    } <= {"layup", "finger_roll"}
    rows = [model.target_probabilities(court, court.cell_index[c]) for c in neighbors]
    expected = np.mean(rows, axis=0)
    permutation = [court.cell_index[c] for c in cells]
    np.testing.assert_allclose(
        dense[SHOT_TYPE_TWO, basket_idx], expected[permutation], atol=1e-7
    )
    np.testing.assert_allclose(
        dense[SHOT_TYPE_DUNK, basket_idx],
        model.target_probabilities(court, court.cell_index[court.basket_axial])[
            permutation
        ],
    )
    assert not np.allclose(
        dense[SHOT_TYPE_TWO, basket_idx], dense[SHOT_TYPE_DUNK, basket_idx]
    )
    assert dense[SHOT_TYPE_TWO, basket_idx, basket_idx] < 1.0
    assert np.isfinite(dense).all()
    assert (dense >= 0).all()
    np.testing.assert_allclose(dense.sum(axis=-1), 1.0, atol=1e-6)


@jax.jit
def run_shots(static, state, keys):
    actions = (
        jnp.zeros(state.positions.shape[:2], dtype=jnp.int32)
        .at[:, 0]
        .set(ActionType.SHOOT.value)
    )
    return step_batch_minimal(static, state, actions, keys, jax, jnp)


@pytest.mark.parametrize(
    "dunk,layup,kind,enabled",
    [
        (0.8, 0.2, SHOT_TYPE_DUNK, True),
        (0.2, 0.8, SHOT_TYPE_TWO, True),
        (0.5, 0.5, SHOT_TYPE_DUNK, True),
        (0.2, 0.8, SHOT_TYPE_DUNK, False),
    ],
)
@pytest.mark.parametrize("mix,temperature", [(0.0, 1.0), (0.25, 0.7)])
def test_runtime_and_observations_use_selected_rebound_distribution(
    shot_setup, dunk, layup, kind, enabled, mix, temperature
):
    static, state = shot_setup
    static = static._replace(
        prefer_higher_percentage_shot=jnp.asarray(enabled, dtype=jnp.int8),
        rebound_target_uniform_mix=jnp.asarray(mix, dtype=jnp.float32),
        rebound_target_temperature=jnp.asarray(temperature, dtype=jnp.float32),
        rebound_obs_top_n_targets=jnp.asarray(0, dtype=jnp.int32),
    )
    state = with_skills(state, dunk, layup)
    profile = build_shot_profile_batch(static, state, jnp)
    features = build_rebound_observation_features_batch(static, state, profile, jnp)
    coords = np.asarray(static.cell_coords)
    basket_idx = np.flatnonzero(
        np.all(coords == np.asarray(static.basket_position), axis=1)
    )[0]
    row = np.asarray(static.rebound_target_probs)[kind, basket_idx]
    logits = np.log(np.maximum((1 - mix) * row + mix / len(row), 1e-8)) / temperature
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    expected_position = probs @ coords / float(static.court_norm_den)
    np.testing.assert_allclose(
        features["expected_target_q"], expected_position[0], atol=1e-6
    )
    np.testing.assert_allclose(
        features["expected_target_r"], expected_position[1], atol=1e-6
    )
    keys = jax.random.split(jax.random.PRNGKey(81), 32)
    out = run_shots(static, state, keys)
    assert np.all(np.asarray(out.shot_type) == kind)
    metrics = _build_shot_type_transition_metrics(static, out, jnp)
    np.testing.assert_array_equal(metrics["shot_dunks"], int(kind == SHOT_TYPE_DUNK))
    np.testing.assert_array_equal(metrics["shot_twos"], int(kind == SHOT_TYPE_TWO))
    # Reconstruct independent draws: observation summaries do not consume RNG.
    action_keys = jax.vmap(lambda key: jax.random.split(key, 5)[2])(keys)
    shot_keys = jax.vmap(lambda key: jax.random.split(key, 4)[0])(action_keys)
    target_keys = jax.vmap(lambda key: jax.random.split(key, 4)[2])(action_keys)
    draws = jax.vmap(jax.random.uniform)(shot_keys)
    made = np.asarray(draws < profile["probability"][:, 0])
    np.testing.assert_array_equal(out.shot_success, made)
    assert made.any() and (~made).any()
    np.testing.assert_array_equal(out.rebound_attempt, ~made)
    targets = jax.vmap(lambda key: jax.random.categorical(key, jnp.asarray(logits)))(
        target_keys
    )
    np.testing.assert_array_equal(
        np.asarray(out.rebound_target_cell)[~made], np.asarray(targets)[~made]
    )
    # Misses continue the multi-possession game; the virtual origin never moves the shooter.
    assert not np.asarray(out.done).any()
    np.testing.assert_array_equal(
        np.asarray(out.state.positions)[~made, 0], np.asarray(state.positions)[~made, 0]
    )
    assert np.all(
        np.asarray(out.offensive_rebound)[~made]
        + np.asarray(out.defensive_rebound)[~made]
        == 1
    )
    offensive = np.asarray(out.offensive_rebound).astype(bool)
    defensive = np.asarray(out.defensive_rebound).astype(bool)
    assert offensive.any() and defensive.any()
    np.testing.assert_array_equal(
        np.asarray(out.state.completed_possessions)[offensive],
        np.asarray(state.completed_possessions)[offensive],
    )
    np.testing.assert_array_equal(
        np.asarray(out.state.completed_possessions)[defensive],
        np.asarray(state.completed_possessions)[defensive] + 1,
    )
    # Calling again is deterministic despite prior sampling.
    again = build_rebound_observation_features_batch(static, state, profile, jnp)
    for name in features:
        np.testing.assert_array_equal(features[name], again[name])


@pytest.mark.parametrize("saved", [None, False, True])
def test_checkpoint_shot_rule_defaults_and_resume_guard(saved):
    from basketworld_jax.train.main import (
        parse_args,
        _effective_env_config_from_args,
        _compatible_env_config_for_resume,
        _validate_checkpoint_episode_semantics,
    )
    from basketworld_jax.eval.native import _resolve_native_environment_config

    metadata = {"env_config": {"allow_dunks": True}}
    if saved is not None:
        metadata["env_config"]["prefer_higher_percentage_shot"] = saved
    config = JaxEnvironmentConfig.from_checkpoint_metadata(metadata)
    assert config.prefer_higher_percentage_shot is bool(saved)
    static = build_kernel_static_from_config(config, xp=np)
    assert bool(static.prefer_higher_percentage_shot) is bool(saved)
    evaluated = _resolve_native_environment_config(
        required_params={},
        optional_params={},
        unified_payload=metadata,
    )
    assert evaluated.prefer_higher_percentage_shot is bool(saved)
    args = parse_args(["--prefer-higher-percentage-shot"] if saved else [])
    assert _effective_env_config_from_args(args)[
        "prefer_higher_percentage_shot"
    ] is bool(saved)
    _validate_checkpoint_episode_semantics(
        metadata, args=args, context="Resume checkpoint"
    )
    args.prefer_higher_percentage_shot = not bool(saved)
    with pytest.raises(SystemExit, match="prefer_higher_percentage_shot"):
        _validate_checkpoint_episode_semantics(
            metadata, args=args, context="Resume checkpoint"
        )
    # Missing legacy metadata cannot inherit an enabled new run's value.
    assert (
        _compatible_env_config_for_resume({}, {"prefer_higher_percentage_shot": True})[
            "prefer_higher_percentage_shot"
        ]
        is False
    )


def test_cross_checkpoint_evaluation_rejects_mixed_shot_rules():
    from basketworld_jax.eval.native import _resolve_native_environment_config

    with pytest.raises(ValueError, match="prefer_higher_percentage_shot"):
        _resolve_native_environment_config(
            required_params={},
            optional_params={},
            unified_payload={"env_config": {"prefer_higher_percentage_shot": True}},
            opponent_payload={"env_config": {}},
        )
