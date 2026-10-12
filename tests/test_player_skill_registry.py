from dataclasses import replace

import pytest

from basketworld_jax.eval.skills import (
    ACTIVE_PLAYER_SKILL_REGISTRY,
    PlayerSkillDescriptor,
    PlayerSkillRegistry,
    PlayerSkillValidationError,
)


def _valid_profile() -> dict[str, float]:
    return {
        "layup": 0.60,
        "three_pt": 0.37,
        "dunk": 0.90,
        "rebound": 0.10,
        "pressure": -0.20,
        "steal": 0.15,
        "ball_handling": 0.25,
        "passing": -0.05,
    }


def test_active_registry_has_a_stable_hashed_schema():
    payload = ACTIVE_PLAYER_SKILL_REGISTRY.schema_payload()

    assert payload["schema_version"] == 1
    assert [item["key"] for item in payload["skills"]] == sorted(_valid_profile())
    assert len(ACTIVE_PLAYER_SKILL_REGISTRY.schema_hash) == 64
    assert (
        ACTIVE_PLAYER_SKILL_REGISTRY.schema_hash
        == ACTIVE_PLAYER_SKILL_REGISTRY.schema_hash
    )


def test_normalize_profile_validates_active_skills_and_preserves_unknown_fields():
    profile = _valid_profile()
    profile["future_contest"] = {"value": 0.4, "source": "future-schema"}

    normalized = ACTIVE_PLAYER_SKILL_REGISTRY.normalize_profile(profile)
    profile["future_contest"]["value"] = 99.0

    assert normalized["layup"] == 0.60
    assert normalized["future_contest"] == {"value": 0.4, "source": "future-schema"}


def test_normalize_profile_reports_missing_and_invalid_declared_skills():
    missing = _valid_profile()
    missing.pop("passing")

    with pytest.raises(PlayerSkillValidationError, match="is missing: passing"):
        ACTIVE_PLAYER_SKILL_REGISTRY.normalize_profile(missing)

    invalid = _valid_profile()
    invalid["three_pt"] = 1.1
    with pytest.raises(PlayerSkillValidationError, match="between 0 and 1"):
        ACTIVE_PLAYER_SKILL_REGISTRY.normalize_profile(invalid)


def test_registry_drives_a_synthetic_extra_numeric_skill():
    synthetic = PlayerSkillDescriptor(
        key="contest",
        display_label="Shot Contest",
        numeric_type="float",
        minimum=-1.0,
        maximum=1.0,
        sampling_distribution="normal",
        sampling_defaults=(("mean", 0.0), ("std", 0.1)),
        semantic_group="defense",
        runtime_field="contest_skill",
    )
    extended = replace(
        ACTIVE_PLAYER_SKILL_REGISTRY,
        descriptors=ACTIVE_PLAYER_SKILL_REGISTRY.descriptors + (synthetic,),
    )
    profile = _valid_profile()
    profile["contest"] = "0.75"

    normalized = extended.normalize_profile(profile)

    assert normalized["contest"] == 0.75
    assert extended.schema_hash != ACTIVE_PLAYER_SKILL_REGISTRY.schema_hash
