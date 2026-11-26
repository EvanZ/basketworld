"""Shared configuration and numerical contract for finishing defense."""

from __future__ import annotations

import math
from typing import Mapping


SHOT_DEFENSE_RULE_VERSION = 1
SHOT_DEFENSE_OBSERVATION_SCHEMA_VERSION = 7
SKILL_PROBABILITY_EPSILON = 1.0e-6

RIM_PROTECTION_DEFAULTS = {
    "rim_protection_enabled": False,
    "rim_protection_beta_alpha": 1.0,
    "rim_protection_beta_beta": 3.0,
    "rim_protection_block_distribution_max": 0.10,
    "rim_protection_skill_block_reference": 0.02,
    "rim_protection_block_base": 0.02,
    "rim_protection_block_effect_weight": 1.0,
    "rim_protection_contest_base": 0.25,
    "rim_protection_contest_effect_weight": 0.5,
}

CONTESTED_SHOOTING_DEFAULTS = {
    "contested_shooting_enabled": False,
    "contested_shooting_skill_mean": 0.0,
    "contested_shooting_skill_std": 1.0,
    "contested_shooting_effect_weight": 0.5,
}
SHOT_DEFENSE_DEFAULTS = {**RIM_PROTECTION_DEFAULTS, **CONTESTED_SHOOTING_DEFAULTS}


def validate_shot_defense(config) -> None:
    validate_rim_protection(config)
    for key, default in CONTESTED_SHOOTING_DEFAULTS.items():
        if isinstance(default, bool):
            continue
        value = float(
            config.get(key, default)
            if isinstance(config, Mapping)
            else getattr(config, key)
        )
        if not math.isfinite(value) or (
            key != "contested_shooting_skill_mean" and value < 0
        ):
            raise ValueError(
                f"{key} must be finite"
                + (
                    " and nonnegative."
                    if key != "contested_shooting_skill_mean"
                    else "."
                )
            )


def shot_defense_values(config) -> dict:
    return {
        key: (
            config.get(key, default)
            if isinstance(config, Mapping)
            else getattr(config, key, default)
        )
        for key, default in SHOT_DEFENSE_DEFAULTS.items()
    }


def shot_defense_enabled(config) -> bool:
    values = shot_defense_values(config)
    return bool(
        values["rim_protection_enabled"] or values["contested_shooting_enabled"]
    )


def shot_defense_contract(config) -> dict:
    return {
        "rule_version": SHOT_DEFENSE_RULE_VERSION,
        "skill_schema_version": 2,
        "rim_skill_distribution": "scaled_beta_logit_centered",
        "contested_shooting_distribution": "independent_normal",
        "parameters": shot_defense_values(config),
    }


def validate_trained_shot_defense(metadata, requested) -> None:
    """Reject semantic changes, even when both policies have identical shapes."""
    recorded = {}
    for key in (
        "frozen_config",
        "trainer_config",
        "env_config",
        "effective_env_config",
    ):
        recorded.update(metadata.get(key) or {})
    expected, actual = shot_defense_values(requested), shot_defense_values(recorded)
    for flag in ("rim_protection_enabled", "contested_shooting_enabled"):
        if bool(expected[flag]) != bool(actual[flag]):
            raise ValueError(
                f"Checkpoint {flag}={actual[flag]} does not match requested {expected[flag]}; trained shot rules cannot be changed."
            )
    if not shot_defense_enabled(actual):
        return
    if (
        int((metadata.get("policy_spec") or {}).get("observation_schema_version", 1))
        != SHOT_DEFENSE_OBSERVATION_SCHEMA_VERSION
    ):
        raise ValueError(
            "Checkpoint enables shot-defense skills without the required observation schema."
        )
    contract = (metadata.get("observation_metadata") or {}).get("shot_defense_contract")
    if contract != shot_defense_contract(actual):
        raise ValueError(
            "Checkpoint is missing or has inconsistent shot-defense rule/skill metadata."
        )
    for key in actual:
        if expected[key] != actual[key]:
            raise ValueError(
                f"Checkpoint shot-defense parameter mismatch for {key}; use a compatible environment."
            )


def validate_rim_protection(config) -> None:
    def value(key):
        raw = (
            config.get(key, RIM_PROTECTION_DEFAULTS[key])
            if isinstance(config, Mapping)
            else getattr(config, key)
        )
        result = float(raw)
        if not math.isfinite(result):
            raise ValueError(f"{key} must be finite.")
        return result

    for key in ("rim_protection_beta_alpha", "rim_protection_beta_beta"):
        if value(key) <= 0:
            raise ValueError(f"{key} must be positive.")
    key = "rim_protection_block_distribution_max"
    if not 0 < value(key) <= 1:
        raise ValueError(f"{key} must be in (0, 1].")
    key = "rim_protection_skill_block_reference"
    if not 0 < value(key) < 1:
        raise ValueError(f"{key} must be strictly between zero and one.")
    for key in ("rim_protection_block_base", "rim_protection_contest_base"):
        if not 0 <= value(key) <= 1:
            raise ValueError(f"{key} must be in [0, 1].")
    for key in (
        "rim_protection_block_effect_weight",
        "rim_protection_contest_effect_weight",
    ):
        if value(key) < 0:
            raise ValueError(f"{key} must be nonnegative.")


def rim_skill_from_block_sample(sample, reference, xp):
    """Convert a sampled probability to one finite, persistent player ability."""
    safe = xp.clip(sample, SKILL_PROBABILITY_EPSILON, 1.0 - SKILL_PROBABILITY_EPSILON)
    reference = xp.clip(
        reference, SKILL_PROBABILITY_EPSILON, 1.0 - SKILL_PROBABILITY_EPSILON
    )
    return xp.log(safe) - xp.log1p(-safe) - xp.log(reference) + xp.log1p(-reference)
