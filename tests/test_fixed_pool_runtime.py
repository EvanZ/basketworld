import numpy as np
import pytest

from basketworld_jax.eval.fixed_pool import generate_player_pool
from basketworld_jax.eval.fixed_pool_runtime import (
    FixedPoolRuntimeError,
    build_fixed_pool_game_batch,
)
from basketworld_jax.eval.lineup_schedule import build_balanced_lineup_schedule
from basketworld_jax.eval.skills import ACTIVE_PLAYER_SKILL_REGISTRY


def test_fixed_pool_batch_maps_global_players_to_stable_jax_slots():
    pool = generate_player_pool(player_count=12, seed=41)
    schedule = build_balanced_lineup_schedule(pool, game_count=2, seed=99)
    batch = build_fixed_pool_game_batch(
        pool,
        schedule,
        schedule.games,
        team_a_slots=range(5),
        team_b_slots=range(5, 10),
        total_slots=10,
    )

    assert batch.starting_team_codes.tolist() == [0, 1]
    assert batch.reset_rng_roots.tolist() == [
        schedule.games[0].reset_rng_root,
        schedule.games[1].reset_rng_root,
    ]
    assert batch.environment_rng_roots.tolist() == [
        schedule.games[0].environment_rng_root,
        schedule.games[1].environment_rng_root,
    ]
    assert batch.policy_rng_roots.tolist() == [
        schedule.games[0].policy_rng_root,
        schedule.games[1].policy_rng_root,
    ]
    for row, game in enumerate(schedule.games):
        player_ids = (*game.team_a_player_ids, *game.team_b_player_ids)
        for descriptor in ACTIVE_PLAYER_SKILL_REGISTRY.descriptors:
            expected = [
                pool.players[int(player_id[-4:])].skills[descriptor.key]
                for player_id in player_ids
            ]
            np.testing.assert_allclose(
                batch.player_skill_rows[descriptor.key][row], expected
            )
    assert batch.trace_metadata(0)["pair_id"] == schedule.games[0].pair_id
    assert batch.trace_metadata(0)["team_a_player_ids"] == list(
        schedule.games[0].team_a_player_ids
    )
    with pytest.raises(ValueError):
        batch.player_skill_rows["layup"][0, 0] = 0.0


def test_fixed_pool_batch_rejects_non_5v5_jax_slots():
    pool = generate_player_pool(player_count=10, seed=12)
    schedule = build_balanced_lineup_schedule(pool, game_count=2, seed=18)

    with pytest.raises(FixedPoolRuntimeError, match="five stable JAX slots"):
        build_fixed_pool_game_batch(
            pool,
            schedule,
            schedule.games,
            team_a_slots=range(3),
            team_b_slots=range(3, 6),
            total_slots=6,
        )
