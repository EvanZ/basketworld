from dataclasses import replace

import pytest

from basketworld_jax.eval.fixed_pool import (
    PoolCompatibilityError,
    PlayerPool,
    assert_pool_compatible,
    extend_player_pool,
    generate_player_pool,
    load_player_pool,
    pool_id_for_payload,
    write_player_pool,
)
from basketworld_jax.eval.skills import ACTIVE_PLAYER_SKILL_REGISTRY


def test_pool_generation_is_deterministic_and_self_describing():
    overrides = {"layup": {"mean": 0.55, "std": 0.03}}

    first = generate_player_pool(player_count=12, seed=73, sampling_overrides=overrides)
    second = generate_player_pool(
        player_count=12, seed=73, sampling_overrides=overrides
    )

    assert first.to_payload() == second.to_payload()
    assert first.pool_id == second.pool_id
    assert first.skill_schema_hash == ACTIVE_PLAYER_SKILL_REGISTRY.schema_hash
    assert (
        first.to_payload()["skill_descriptors"]
        == ACTIVE_PLAYER_SKILL_REGISTRY.schema_payload()["skills"]
    )
    assert [player.player_id for player in first.players] == [
        f"player-{index:04d}" for index in range(12)
    ]


def test_pool_round_trip_preserves_unknown_fields_and_rejects_conflicting_overwrite(
    tmp_path,
):
    pool = generate_player_pool(player_count=2, seed=19)
    payload = pool.to_payload()
    payload["future_top_level"] = {"ledger_version": 9}
    payload["players"][0]["skills"]["future_contest"] = {"value": 0.75}
    payload["players"][0]["future_player_metadata"] = ["prospect"]
    payload["pool_id"] = pool_id_for_payload(payload)

    restored = PlayerPool.from_payload(payload)
    path = write_player_pool(restored, tmp_path / "pool.json")
    reloaded = load_player_pool(path)

    assert reloaded.to_payload() == restored.to_payload()
    assert reloaded.extra_fields["future_top_level"] == {"ledger_version": 9}
    assert reloaded.players[0].skills["future_contest"] == {"value": 0.75}
    assert list(reloaded.players[0].extra_fields["future_player_metadata"]) == [
        "prospect"
    ]
    with pytest.raises(TypeError):
        reloaded.players[0].skills["future_contest"] = {"value": 1.0}

    conflicting = generate_player_pool(player_count=2, seed=20)
    with pytest.raises(FileExistsError, match="immutable"):
        write_player_pool(conflicting, path)


def test_pool_extension_samples_only_missing_registered_skills_and_keeps_parent_unchanged():
    legacy_registry = replace(
        ACTIVE_PLAYER_SKILL_REGISTRY,
        descriptors=tuple(
            descriptor
            for descriptor in ACTIVE_PLAYER_SKILL_REGISTRY.descriptors
            if descriptor.key != "passing"
        ),
    )
    parent = generate_player_pool(player_count=3, seed=101, registry=legacy_registry)

    with pytest.raises(PoolCompatibilityError, match="lacks descriptor.*passing"):
        assert_pool_compatible(parent)

    child = extend_player_pool(parent, migration_seed=90210)
    replay = extend_player_pool(parent, migration_seed=90210)

    assert child.to_payload() == replay.to_payload()
    assert child.pool_id != parent.pool_id
    assert child.parent_pool_id == parent.pool_id
    assert child.migration_seed == 90210
    for parent_player, child_player in zip(parent.players, child.players, strict=True):
        assert "passing" not in parent_player.skills
        assert (
            child_player.skills["passing"]
            == replay.players[int(child_player.player_id.rsplit("-", 1)[1])].skills[
                "passing"
            ]
        )
        for key, value in parent_player.skills.items():
            assert child_player.skills[key] == value
