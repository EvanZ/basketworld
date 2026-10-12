"""Versioned metadata for episode-stable evaluation player skills.

The environment remains the source of truth for how a skill changes a game.
This module owns the evaluation contract around those skills: stable names,
validation, serialization, and the JAX state field that receives a pinned
value.  Pool and study artifacts can therefore record a self-describing skill
schema instead of depending on a positional, hard-coded array.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping


PLAYER_SKILL_SCHEMA_VERSION = 1


class PlayerSkillValidationError(ValueError):
    """Raised when a profile does not satisfy a registry descriptor."""

    def __init__(
        self,
        message: str,
        *,
        skill_key: str | None = None,
        missing_keys: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        self.skill_key = skill_key
        self.missing_keys = missing_keys


def _copy_unknown_value(value: Any) -> Any:
    """Copy an unknown artifact field, including immutable mapping proxies."""
    if isinstance(value, Mapping):
        return {str(key): _copy_unknown_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_copy_unknown_value(item) for item in value]
    return copy.deepcopy(value)


@dataclass(frozen=True)
class PlayerSkillDescriptor:
    """One episode-stable player skill exposed to evaluation tooling."""

    key: str
    display_label: str
    numeric_type: str
    minimum: float | None
    maximum: float | None
    sampling_distribution: str
    sampling_defaults: tuple[tuple[str, Any], ...]
    semantic_group: str
    runtime_field: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready, order-independent descriptor payload."""
        return {
            "key": self.key,
            "display_label": self.display_label,
            "numeric_type": self.numeric_type,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "sampling_distribution": self.sampling_distribution,
            "sampling_defaults": {key: value for key, value in self.sampling_defaults},
            "semantic_group": self.semantic_group,
            "runtime_field": self.runtime_field,
        }

    def normalize_value(self, value: Any) -> float:
        """Coerce and validate one numeric skill value."""
        try:
            normalized = float(value)
        except (TypeError, ValueError) as exc:
            raise PlayerSkillValidationError(
                "must be numeric",
                skill_key=self.key,
            ) from exc
        if not math.isfinite(normalized):
            raise PlayerSkillValidationError(
                "must be finite",
                skill_key=self.key,
            )
        if self.minimum is not None and normalized < self.minimum:
            raise PlayerSkillValidationError(
                self._bounds_message(),
                skill_key=self.key,
            )
        if self.maximum is not None and normalized > self.maximum:
            raise PlayerSkillValidationError(
                self._bounds_message(),
                skill_key=self.key,
            )
        return normalized

    def _bounds_message(self) -> str:
        if self.minimum is not None and self.maximum is not None:
            return f"must be between {self.minimum:g} and {self.maximum:g}"
        if self.minimum is not None:
            return f"must be at least {self.minimum:g}"
        return f"must be at most {self.maximum:g}"


@dataclass(frozen=True)
class PlayerSkillRegistry:
    """A versioned collection of descriptors with canonical serialization."""

    schema_version: int
    descriptors: tuple[PlayerSkillDescriptor, ...]

    def __post_init__(self) -> None:
        keys = [descriptor.key for descriptor in self.descriptors]
        if len(set(keys)) != len(keys):
            raise ValueError("Player-skill registry keys must be unique.")
        fields = [descriptor.runtime_field for descriptor in self.descriptors]
        if len(set(fields)) != len(fields):
            raise ValueError("Player-skill registry runtime fields must be unique.")
        if any(descriptor.numeric_type != "float" for descriptor in self.descriptors):
            raise ValueError("Only float player-skill descriptors are supported.")

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(descriptor.key for descriptor in self.descriptors)

    def descriptor_for(self, key: str) -> PlayerSkillDescriptor:
        for descriptor in self.descriptors:
            if descriptor.key == key:
                return descriptor
        raise KeyError(f"Unknown player skill: {key}")

    def schema_payload(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "skills": [
                descriptor.to_dict()
                for descriptor in sorted(self.descriptors, key=lambda item: item.key)
            ],
        }

    @property
    def schema_hash(self) -> str:
        encoded = json.dumps(
            self.schema_payload(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def normalize_profile(
        self,
        raw_profile: Mapping[str, Any],
        *,
        preserve_unknown: bool = True,
    ) -> dict[str, Any]:
        """Validate active skills and preserve optional future fields verbatim.

        Unknown values are deliberately retained for artifact round trips.  The
        native JAX application path uses only registered active descriptors, so
        an unknown field never silently changes a simulation.
        """
        if not isinstance(raw_profile, Mapping):
            raise PlayerSkillValidationError("must be an object")
        missing = tuple(key for key in self.keys if key not in raw_profile)
        if missing:
            raise PlayerSkillValidationError(
                f"is missing: {', '.join(missing)}",
                missing_keys=missing,
            )
        normalized: dict[str, Any] = {
            descriptor.key: descriptor.normalize_value(raw_profile[descriptor.key])
            for descriptor in self.descriptors
        }
        if preserve_unknown:
            known_keys = set(self.keys)
            for key, value in raw_profile.items():
                if key not in known_keys:
                    normalized[str(key)] = _copy_unknown_value(value)
        return normalized


def _sampling_defaults(**values: Any) -> tuple[tuple[str, Any], ...]:
    return tuple(sorted(values.items()))


ACTIVE_PLAYER_SKILL_REGISTRY = PlayerSkillRegistry(
    schema_version=PLAYER_SKILL_SCHEMA_VERSION,
    descriptors=(
        PlayerSkillDescriptor(
            key="layup",
            display_label="Layup",
            numeric_type="float",
            minimum=0.0,
            maximum=1.0,
            sampling_distribution="normal",
            sampling_defaults=_sampling_defaults(
                default_mean=0.60,
                default_std=0.0,
                mean_field="layup_pct",
                std_field="layup_std",
            ),
            semantic_group="shooting",
            runtime_field="layup_pct",
        ),
        PlayerSkillDescriptor(
            key="three_pt",
            display_label="Three Point",
            numeric_type="float",
            minimum=0.0,
            maximum=1.0,
            sampling_distribution="normal",
            sampling_defaults=_sampling_defaults(
                default_mean=0.37,
                default_std=0.0,
                mean_field="three_pt_pct",
                std_field="three_pt_std",
            ),
            semantic_group="shooting",
            runtime_field="three_pt_pct",
        ),
        PlayerSkillDescriptor(
            key="dunk",
            display_label="Dunk",
            numeric_type="float",
            minimum=0.0,
            maximum=1.0,
            sampling_distribution="normal",
            sampling_defaults=_sampling_defaults(
                default_mean=0.90,
                default_std=0.0,
                mean_field="dunk_pct",
                std_field="dunk_std",
            ),
            semantic_group="shooting",
            runtime_field="dunk_pct",
        ),
        PlayerSkillDescriptor(
            key="rebound",
            display_label="Rebounding",
            numeric_type="float",
            minimum=None,
            maximum=None,
            sampling_distribution="gaussian_or_specialist",
            sampling_defaults=_sampling_defaults(
                default_mean=0.0,
                default_std=0.0,
                std_field="rebound_skill_std",
            ),
            semantic_group="shared",
            runtime_field="rebound_skill",
        ),
        PlayerSkillDescriptor(
            key="pressure",
            display_label="Pressure",
            numeric_type="float",
            minimum=None,
            maximum=None,
            sampling_distribution="normal",
            sampling_defaults=_sampling_defaults(
                default_mean=0.0,
                default_std=0.0,
                mean_field="pressure_skill_mean",
                std_field="pressure_skill_std",
            ),
            semantic_group="defense",
            runtime_field="pressure_skill",
        ),
        PlayerSkillDescriptor(
            key="shot_contest",
            display_label="Shot Contest",
            numeric_type="float",
            minimum=None,
            maximum=None,
            sampling_distribution="normal",
            sampling_defaults=_sampling_defaults(
                default_mean=0.0,
                default_std=0.0,
                mean_field="shot_contest_skill_mean",
                std_field="shot_contest_skill_std",
            ),
            semantic_group="defense",
            runtime_field="shot_contest_skill",
        ),
        PlayerSkillDescriptor(
            key="steal",
            display_label="Steal",
            numeric_type="float",
            minimum=None,
            maximum=None,
            sampling_distribution="normal",
            sampling_defaults=_sampling_defaults(
                default_mean=0.0,
                default_std=0.0,
                mean_field="steal_skill_mean",
                std_field="steal_skill_std",
            ),
            semantic_group="defense",
            runtime_field="steal_skill",
        ),
        PlayerSkillDescriptor(
            key="ball_handling",
            display_label="Ball Handling",
            numeric_type="float",
            minimum=None,
            maximum=None,
            sampling_distribution="normal",
            sampling_defaults=_sampling_defaults(
                default_mean=0.0,
                default_std=0.0,
                mean_field="ball_handling_skill_mean",
                std_field="ball_handling_skill_std",
            ),
            semantic_group="offense",
            runtime_field="ball_handling_skill",
        ),
        PlayerSkillDescriptor(
            key="passing",
            display_label="Passing",
            numeric_type="float",
            minimum=None,
            maximum=None,
            sampling_distribution="normal",
            sampling_defaults=_sampling_defaults(
                default_mean=0.0,
                default_std=0.0,
                mean_field="passing_skill_mean",
                std_field="passing_skill_std",
            ),
            semantic_group="offense",
            runtime_field="passing_skill",
        ),
    ),
)
