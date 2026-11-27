"""Approved #84 probability contract, independent of sampling noise."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from basketworld_jax.config import JaxEnvironmentConfig
from basketworld_jax.env.minimal import build_shot_profile_batch
from basketworld_jax.shot_defense import (
    RIM_PROTECTION_DEFAULTS,
    rim_skill_from_block_sample,
)
from tests.test_jax_shot_profile import (
    _state_for_shot_profile,
    _static_for_shot_profile,
)


def profile(*, shooter=(0, 0), defender=(1, 0), block_sample=0.02, **settings):
    cells = np.array([(q, r) for q in range(-4, 5) for r in range(-4, 5)])
    static = _static_for_shot_profile(cells, np.zeros(len(cells)), (0, 0))
    config = {**RIM_PROTECTION_DEFAULTS, "rim_protection_enabled": True, **settings}
    static = static._replace(
        **{
            key: jnp.asarray(
                value, dtype=jnp.int8 if isinstance(value, bool) else jnp.float32
            )
            for key, value in config.items()
        }
    )
    state = _state_for_shot_profile(np.array([[shooter, defender]]))
    state = state._replace(
        layup_pct=jnp.full((1, 2), 0.8),
        rim_protection_skill=jnp.asarray(
            [[0.0, float(rim_skill_from_block_sample(block_sample, 0.02, np))]]
        ),
    )
    return build_shot_profile_batch(static, state, jnp)


@pytest.mark.parametrize(
    "defender", [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
)
def test_under_basket_finish_is_contested_from_every_direction(defender):
    result = profile(defender=defender)
    assert float(result["block_probability"][0, 0]) == pytest.approx(0.02)
    assert float(result["conditional_probability"][0, 0]) == pytest.approx(0.6)
    assert float(result["probability"][0, 0]) == pytest.approx(0.588)
    assert bool(result["blocker_candidates"][0, 0, 1])


@pytest.mark.parametrize(
    "sample,expected", [(0.01, 0.642), (0.02, 0.588), (0.05, 0.495), (0.10, 0.405)]
)
def test_shared_skill_mapping_matches_approved_examples(sample, expected):
    result = profile(block_sample=sample)
    assert float(result["block_probability"][0, 0]) == pytest.approx(sample)
    assert float(result["probability"][0, 0]) == pytest.approx(expected, abs=0.001)


def test_no_adjacent_defender_and_disabled_rules_preserve_open_baseline():
    for result in (profile(defender=(2, 0)), profile(rim_protection_enabled=False)):
        assert float(result["probability"][0, 0]) == pytest.approx(0.8)
        assert float(result["block_probability"][0, 0]) == 0


def test_jumpers_are_not_rim_finishes():
    result = profile(shooter=(2, 0), defender=(1, 0))
    assert not bool(result["is_finish"][0, 0])
    assert float(result["block_probability"][0, 0]) == 0


@pytest.mark.parametrize(
    "block,contest,expected", [(0, 0, 0.8), (1, 0, 0), (0, 1, 0.01)]
)
def test_probability_endpoints(block, contest, expected):
    result = profile(
        rim_protection_block_base=block, rim_protection_contest_base=contest
    )
    assert float(result["probability"][0, 0]) == pytest.approx(expected)


@pytest.mark.parametrize(
    "name,value",
    [
        ("rim_protection_beta_alpha", 0),
        ("rim_protection_beta_beta", float("nan")),
        ("rim_protection_block_distribution_max", 1.1),
        ("rim_protection_skill_block_reference", 0),
        ("rim_protection_block_base", -0.1),
        ("rim_protection_contest_base", float("inf")),
        ("rim_protection_block_effect_weight", -1),
        ("rim_protection_contest_effect_weight", -1),
    ],
)
def test_invalid_parameters_rejected(name, value):
    with pytest.raises(ValueError, match=name):
        JaxEnvironmentConfig(**{name: value})


def test_beta_mapping_has_finite_endpoints_and_correct_default_distribution():
    samples = np.random.default_rng(84).beta(1, 3, size=100_000) * 0.1
    assert samples.mean() == pytest.approx(0.025, abs=0.0003)
    assert np.mean(samples < 0.02) == pytest.approx(0.488, abs=0.005)
    assert np.isfinite(
        rim_skill_from_block_sample(np.array([0.0, 1.0]), 0.02, np)
    ).all()


@pytest.mark.parametrize(
    "skill,expected", [(-1.0, 0.62245933), (0.0, 0.5), (1.0, 0.37754067)]
)
def test_contested_shooting_offsets_perimeter_suppression(skill, expected):
    cells = np.array([(q, r) for q in range(-4, 5) for r in range(-4, 5)])
    static = _static_for_shot_profile(cells, np.zeros(len(cells)), (0, 0))._replace(
        shot_pressure_enabled=jnp.asarray(1),
        shot_pressure_max=jnp.asarray(0.5),
        shot_pressure_cos_threshold=jnp.asarray(0.0),
        contested_shooting_enabled=jnp.asarray(1),
        contested_shooting_effect_weight=jnp.asarray(0.5),
    )
    state = _state_for_shot_profile(np.array([[(3, 0), (2, 0)]]))
    state = state._replace(contested_shooting_skill=jnp.array([[skill, 0.0]]))
    result = build_shot_profile_batch(static, state, jnp)
    assert float(result["suppression_after_resistance"][0, 0]) == pytest.approx(
        expected
    )
    assert float(result["suppression_before_resistance"][0, 0]) == pytest.approx(0.5)
    disabled = build_shot_profile_batch(
        static._replace(contested_shooting_enabled=jnp.asarray(0)), state, jnp
    )
    assert float(disabled["suppression_after_resistance"][0, 0]) == pytest.approx(0.5)


def test_trained_contract_rejects_equal_dimension_semantic_changes():
    from basketworld_jax.shot_defense import (
        shot_defense_contract,
        validate_trained_shot_defense,
    )
    from basketworld_jax.config.environment import normalize_runtime_environment_config

    config = JaxEnvironmentConfig(
        enable_multi_possession=True,
        rim_protection_enabled=True,
        contested_shooting_enabled=True,
    )
    assert (
        normalize_runtime_environment_config(config).multi_possession_schema_version
        == 7
    )
    metadata = {
        "env_config": config.as_metadata(),
        "policy_spec": {"observation_schema_version": 7},
        "observation_metadata": {
            "shot_defense_contract": shot_defense_contract(config)
        },
    }
    validate_trained_shot_defense(metadata, config)
    with pytest.raises(ValueError, match="trained shot rules"):
        validate_trained_shot_defense(
            metadata, {**config.as_metadata(), "contested_shooting_enabled": False}
        )
    with pytest.raises(ValueError, match="parameter mismatch"):
        validate_trained_shot_defense(
            metadata, {**config.as_metadata(), "rim_protection_block_base": 0.03}
        )
    with pytest.raises(ValueError, match="trained shot rules"):
        validate_trained_shot_defense({}, config)


def test_new_flags_and_parameters_round_trip_through_trainer_editor():
    from app.training_backend.application_training import load_application_config_schema
    from app.training_backend.config_schema import compile_overrides
    from basketworld_jax.train.main import parse_args, _jax_env_config_from_args
    from basketworld_jax.shot_defense import SHOT_DEFENSE_DEFAULTS
    from pathlib import Path

    schema = load_application_config_schema(str(Path(__file__).resolve().parents[1]))
    settings = {
        **SHOT_DEFENSE_DEFAULTS,
        "rim_protection_enabled": True,
        "contested_shooting_enabled": True,
        "rim_protection_beta_beta": 4.0,
        "contested_shooting_skill_std": 0.7,
    }
    args = parse_args(compile_overrides(settings, schema))
    resolved = _jax_env_config_from_args(args)
    for key, value in settings.items():
        assert resolved[key] == value
    assert resolved["multi_possession_schema_version"] == 7


def test_block_metrics_survive_default_mlflow_filters():
    from basketworld_jax.train.main import (
        _filter_mlflow_train_metrics,
        _filter_mlflow_deploy_metrics,
    )

    from basketworld_jax.block_metrics import (
        TRAIN_BLOCK_METRICS, DEPLOY_BLOCK_METRICS,
        TRAIN_RIM_CONTEST_METRICS, DEPLOY_RIM_CONTEST_METRICS,
        TRAIN_LAYUP_METRICS, DEPLOY_LAYUP_METRICS,
    )

    metrics = {key: 1.0 for key in TRAIN_BLOCK_METRICS | DEPLOY_BLOCK_METRICS
               | TRAIN_RIM_CONTEST_METRICS | DEPLOY_RIM_CONTEST_METRICS
               | TRAIN_LAYUP_METRICS | DEPLOY_LAYUP_METRICS}
    metrics.update(blocks=3.0, all_shot_defense_rim_opportunities=15.0)
    for profile in ("core", "full"):
        assert set(_filter_mlflow_train_metrics(metrics, profile=profile)) == TRAIN_BLOCK_METRICS | TRAIN_RIM_CONTEST_METRICS | TRAIN_LAYUP_METRICS
        assert set(_filter_mlflow_deploy_metrics(metrics, profile=profile)) == DEPLOY_BLOCK_METRICS | DEPLOY_RIM_CONTEST_METRICS | DEPLOY_LAYUP_METRICS
