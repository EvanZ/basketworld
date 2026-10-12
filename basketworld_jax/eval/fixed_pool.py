"""Immutable, versioned fixed-player-pool artifacts for evaluation studies."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from basketworld_jax.eval.skills import (
    ACTIVE_PLAYER_SKILL_REGISTRY,
    PlayerSkillDescriptor,
    PlayerSkillRegistry,
    PlayerSkillValidationError,
)


PLAYER_POOL_SCHEMA_VERSION = 1


class PlayerPoolError(ValueError):
    """Base error for fixed-player-pool artifacts."""


class PoolCompatibilityError(PlayerPoolError):
    """A historical pool cannot safely run under the requested skill schema."""


def _json_value(value: Any) -> Any:
    """Return a defensive JSON-compatible copy without dropping unknown fields."""
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return copy.deepcopy(value)


def _freeze_json(value: Any) -> Any:
    """Recursively freeze artifact data while retaining JSON serialization."""
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    return copy.deepcopy(value)


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        _json_value(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def pool_id_for_payload(payload: Mapping[str, Any]) -> str:
    """Compute the content-addressed ID for a pool payload.

    ``pool_id`` itself is excluded so an artifact can carry its identifier
    without making the hash recursive.
    """
    content = {key: value for key, value in payload.items() if key != "pool_id"}
    digest = hashlib.sha256(_canonical_json(content).encode("utf-8")).hexdigest()
    return f"pool-v{PLAYER_POOL_SCHEMA_VERSION}-{digest[:20]}"


@dataclass(frozen=True)
class PoolPlayer:
    player_id: str
    name: str
    skills: Mapping[str, Any]
    extra_fields: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "skills", _freeze_json(self.skills))
        object.__setattr__(self, "extra_fields", _freeze_json(self.extra_fields))

    def to_payload(self) -> dict[str, Any]:
        return {
            "player_id": self.player_id,
            "name": self.name,
            "skills": _json_value(self.skills),
            **_json_value(self.extra_fields),
        }


@dataclass(frozen=True)
class PlayerPool:
    schema_version: int
    pool_id: str
    skill_schema_version: int
    skill_schema_hash: str
    skill_descriptors: tuple[Mapping[str, Any], ...]
    pool_seed: int
    players: tuple[PoolPlayer, ...]
    parent_pool_id: str | None = None
    migration_seed: int | None = None
    extra_fields: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "skill_descriptors",
            tuple(_freeze_json(descriptor) for descriptor in self.skill_descriptors),
        )
        object.__setattr__(self, "extra_fields", _freeze_json(self.extra_fields or {}))

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": int(self.schema_version),
            "pool_id": self.pool_id,
            "skill_schema_version": int(self.skill_schema_version),
            "skill_schema_hash": self.skill_schema_hash,
            "skill_descriptors": [_json_value(item) for item in self.skill_descriptors],
            "pool_seed": int(self.pool_seed),
            "players": [player.to_payload() for player in self.players],
        }
        if self.parent_pool_id is not None:
            payload["parent_pool_id"] = self.parent_pool_id
        if self.migration_seed is not None:
            payload["migration_seed"] = int(self.migration_seed)
        if self.extra_fields:
            payload.update(_json_value(self.extra_fields))
        return payload

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        registry: PlayerSkillRegistry = ACTIVE_PLAYER_SKILL_REGISTRY,
        validate_active_skills: bool = True,
    ) -> "PlayerPool":
        if not isinstance(payload, Mapping):
            raise PlayerPoolError("Player-pool artifact must be a JSON object.")
        try:
            schema_version = int(payload["schema_version"])
            pool_id = str(payload["pool_id"])
            skill_schema_version = int(payload["skill_schema_version"])
            skill_schema_hash = str(payload["skill_schema_hash"])
            pool_seed = int(payload["pool_seed"])
        except (KeyError, TypeError, ValueError) as exc:
            raise PlayerPoolError(
                "Player-pool artifact is missing required metadata."
            ) from exc
        if schema_version != PLAYER_POOL_SCHEMA_VERSION:
            raise PlayerPoolError(
                f"Unsupported player-pool schema version {schema_version}; "
                f"expected {PLAYER_POOL_SCHEMA_VERSION}."
            )
        descriptors = payload.get("skill_descriptors")
        if not isinstance(descriptors, list) or not all(
            isinstance(descriptor, Mapping) for descriptor in descriptors
        ):
            raise PlayerPoolError(
                "Player-pool artifact must contain skill_descriptors."
            )
        raw_players = payload.get("players")
        if not isinstance(raw_players, list) or not raw_players:
            raise PlayerPoolError(
                "Player-pool artifact must contain at least one player."
            )
        players = tuple(_player_from_payload(player) for player in raw_players)
        player_ids = [player.player_id for player in players]
        if len(set(player_ids)) != len(player_ids):
            raise PlayerPoolError("Player-pool artifact contains duplicate player IDs.")
        known_fields = {
            "schema_version",
            "pool_id",
            "skill_schema_version",
            "skill_schema_hash",
            "skill_descriptors",
            "pool_seed",
            "players",
            "parent_pool_id",
            "migration_seed",
        }
        pool = cls(
            schema_version=schema_version,
            pool_id=pool_id,
            skill_schema_version=skill_schema_version,
            skill_schema_hash=skill_schema_hash,
            skill_descriptors=tuple(_json_value(item) for item in descriptors),
            pool_seed=pool_seed,
            players=players,
            parent_pool_id=(
                str(payload["parent_pool_id"])
                if payload.get("parent_pool_id") is not None
                else None
            ),
            migration_seed=(
                int(payload["migration_seed"])
                if payload.get("migration_seed") is not None
                else None
            ),
            extra_fields={
                str(key): _json_value(value)
                for key, value in payload.items()
                if key not in known_fields
            },
        )
        expected_id = pool_id_for_payload(pool.to_payload())
        if pool.pool_id != expected_id:
            raise PlayerPoolError(
                f"Pool ID {pool.pool_id!r} does not match artifact content; "
                f"expected {expected_id!r}."
            )
        if validate_active_skills:
            assert_pool_compatible(pool, registry=registry)
        return pool


def _player_from_payload(payload: Any) -> PoolPlayer:
    if not isinstance(payload, Mapping):
        raise PlayerPoolError("Every pool player must be a JSON object.")
    try:
        player_id = str(payload["player_id"])
        name = str(payload["name"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PlayerPoolError("Every pool player needs player_id and name.") from exc
    skills = payload.get("skills")
    if not isinstance(skills, Mapping):
        raise PlayerPoolError(f"Pool player {player_id!r} needs a skills object.")
    known_fields = {"player_id", "name", "skills"}
    return PoolPlayer(
        player_id=player_id,
        name=name,
        skills=_json_value(skills),
        extra_fields={
            str(key): _json_value(value)
            for key, value in payload.items()
            if key not in known_fields
        },
    )


def _registry_descriptor_payload(
    registry: PlayerSkillRegistry,
) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        _json_value(descriptor) for descriptor in registry.schema_payload()["skills"]
    )


def _sample_skill(
    descriptor: PlayerSkillDescriptor,
    rng: np.random.Generator,
    overrides: Mapping[str, Any] | None,
) -> float:
    defaults = dict(descriptor.sampling_defaults)
    override = dict(overrides or {})
    mean = float(override.get("mean", defaults.get("default_mean", 0.0)))
    std = float(override.get("std", defaults.get("default_std", 0.0)))
    if std < 0.0:
        raise PlayerPoolError(
            f"Sampling std for skill {descriptor.key!r} must be non-negative."
        )
    value = float(rng.normal(mean, std))
    if descriptor.minimum is not None:
        value = max(value, float(descriptor.minimum))
    if descriptor.maximum is not None:
        value = min(value, float(descriptor.maximum))
    return value


def _make_pool(
    *,
    registry: PlayerSkillRegistry,
    pool_seed: int,
    players: Sequence[PoolPlayer],
    parent_pool_id: str | None = None,
    migration_seed: int | None = None,
    extra_fields: Mapping[str, Any] | None = None,
) -> PlayerPool:
    provisional = {
        "schema_version": PLAYER_POOL_SCHEMA_VERSION,
        "skill_schema_version": int(registry.schema_version),
        "skill_schema_hash": registry.schema_hash,
        "skill_descriptors": list(_registry_descriptor_payload(registry)),
        "pool_seed": int(pool_seed),
        "players": [player.to_payload() for player in players],
        **(_json_value(extra_fields) if extra_fields else {}),
    }
    if parent_pool_id is not None:
        provisional["parent_pool_id"] = parent_pool_id
    if migration_seed is not None:
        provisional["migration_seed"] = int(migration_seed)
    pool_id = pool_id_for_payload(provisional)
    return PlayerPool(
        schema_version=PLAYER_POOL_SCHEMA_VERSION,
        pool_id=pool_id,
        skill_schema_version=int(registry.schema_version),
        skill_schema_hash=registry.schema_hash,
        skill_descriptors=_registry_descriptor_payload(registry),
        pool_seed=int(pool_seed),
        players=tuple(players),
        parent_pool_id=parent_pool_id,
        migration_seed=migration_seed,
        extra_fields=_json_value(extra_fields or {}),
    )


def generate_player_pool(
    *,
    player_count: int,
    seed: int,
    registry: PlayerSkillRegistry = ACTIVE_PLAYER_SKILL_REGISTRY,
    sampling_overrides: Mapping[str, Mapping[str, Any]] | None = None,
    names: Sequence[str] | None = None,
    extra_fields: Mapping[str, Any] | None = None,
) -> PlayerPool:
    """Generate a deterministic, self-describing pool from registry metadata."""
    if int(player_count) < 1:
        raise PlayerPoolError("Player-pool size must be at least one.")
    if names is not None and len(names) != int(player_count):
        raise PlayerPoolError("Player-pool names must match player_count exactly.")
    rng = np.random.default_rng(int(seed))
    players = []
    for index in range(int(player_count)):
        skills = {
            descriptor.key: _sample_skill(
                descriptor,
                rng,
                (sampling_overrides or {}).get(descriptor.key),
            )
            for descriptor in registry.descriptors
        }
        players.append(
            PoolPlayer(
                player_id=f"player-{index:04d}",
                name=(names[index] if names is not None else f"Player {index + 1:03d}"),
                skills=skills,
                extra_fields={},
            )
        )
    return _make_pool(
        registry=registry,
        pool_seed=int(seed),
        players=players,
        extra_fields=extra_fields,
    )


def assert_pool_compatible(
    pool: PlayerPool,
    *,
    registry: PlayerSkillRegistry = ACTIVE_PLAYER_SKILL_REGISTRY,
) -> None:
    """Ensure each active descriptor and skill value can run in this study."""
    artifact_descriptors = {
        str(descriptor.get("key")): descriptor
        for descriptor in pool.skill_descriptors
        if isinstance(descriptor, Mapping) and descriptor.get("key") is not None
    }
    for descriptor in registry.descriptors:
        artifact_descriptor = artifact_descriptors.get(descriptor.key)
        if artifact_descriptor is None:
            raise PoolCompatibilityError(
                f"Pool {pool.pool_id} lacks descriptor for active skill {descriptor.key!r}. "
                "Use the original compatible environment or create a versioned pool extension."
            )
        if _canonical_json(artifact_descriptor) != _canonical_json(
            descriptor.to_dict()
        ):
            raise PoolCompatibilityError(
                f"Pool {pool.pool_id} has an incompatible descriptor for active skill "
                f"{descriptor.key!r}. Use the original compatible environment or create "
                "a versioned pool extension."
            )
    for player in pool.players:
        try:
            registry.normalize_profile(player.skills, preserve_unknown=True)
        except PlayerSkillValidationError as exc:
            if exc.missing_keys:
                detail = f"missing active skills {', '.join(exc.missing_keys)}"
            elif exc.skill_key is not None:
                detail = f"invalid active skill {exc.skill_key!r}: {exc}"
            else:
                detail = str(exc)
            raise PoolCompatibilityError(
                f"Pool {pool.pool_id} player {player.player_id!r} is incompatible: {detail}. "
                "Create a versioned pool extension rather than modifying this artifact."
            ) from exc


def extend_player_pool(
    parent: PlayerPool,
    *,
    migration_seed: int,
    registry: PlayerSkillRegistry = ACTIVE_PLAYER_SKILL_REGISTRY,
) -> PlayerPool:
    """Create a deterministic child pool by sampling only newly required skills."""
    rng = np.random.default_rng(int(migration_seed))
    extended_players: list[PoolPlayer] = []
    added_skill_count = 0
    for player in parent.players:
        skills = _json_value(player.skills)
        for descriptor in registry.descriptors:
            if descriptor.key not in skills:
                skills[descriptor.key] = _sample_skill(descriptor, rng, None)
                added_skill_count += 1
        extended_players.append(
            PoolPlayer(
                player_id=player.player_id,
                name=player.name,
                skills=skills,
                extra_fields=_json_value(player.extra_fields),
            )
        )
    if added_skill_count == 0:
        raise PlayerPoolError(
            f"Pool {parent.pool_id} already contains every active skill; no extension is needed."
        )
    child_extra = _json_value(parent.extra_fields or {})
    child_extra["migration"] = {
        "added_skill_count": added_skill_count,
        "parent_skill_schema_hash": parent.skill_schema_hash,
    }
    child = _make_pool(
        registry=registry,
        pool_seed=parent.pool_seed,
        players=extended_players,
        parent_pool_id=parent.pool_id,
        migration_seed=int(migration_seed),
        extra_fields=child_extra,
    )
    assert_pool_compatible(child, registry=registry)
    return child


def write_player_pool(pool: PlayerPool, path: str | Path) -> Path:
    """Write an immutable artifact, rejecting a conflicting overwrite."""
    target = Path(path)
    payload = pool.to_payload()
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        existing = target.read_text(encoding="utf-8")
        if existing != text:
            raise FileExistsError(
                f"Refusing to overwrite immutable player-pool artifact: {target}"
            )
        return target
    target.write_text(text, encoding="utf-8")
    return target


def load_player_pool(
    path: str | Path,
    *,
    registry: PlayerSkillRegistry = ACTIVE_PLAYER_SKILL_REGISTRY,
    validate_active_skills: bool = True,
) -> PlayerPool:
    """Load a pool while preserving future fields and checking its identity."""
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PlayerPoolError(f"Invalid player-pool JSON: {source}") from exc
    return PlayerPool.from_payload(
        payload,
        registry=registry,
        validate_active_skills=validate_active_skills,
    )
