from dataclasses import replace
import json

import pytest

from basketworld_jax.eval.fixed_pool import generate_player_pool
from basketworld_jax.eval.lineup_schedule import (
    LineupSchedule,
    LineupScheduleError,
    ScheduleCompatibilityError,
    ScheduledGame,
    build_balanced_lineup_schedule,
    build_lineup_schedule_extension,
    load_lineup_schedule,
    schedule_balance_diagnostics,
    schedule_id_for_payload,
    validate_lineup_schedule,
    write_lineup_schedule,
)


def test_lineup_schedule_is_deterministic_balanced_and_has_independent_games():
    pool = generate_player_pool(player_count=12, seed=31)

    first = build_balanced_lineup_schedule(pool, game_count=48, seed=912)
    replay = build_balanced_lineup_schedule(pool, game_count=48, seed=912)

    assert first.to_payload() == replay.to_payload()
    assert first.schedule_id == replay.schedule_id
    assert first.sampling_group_count == 48
    assert len(first.games) == 48
    assert first.diagnostics["game_appearance_spread"] <= 1
    assert first.diagnostics["teammate_pairs"]["total_assignments"] == 48 * 20
    assert first.diagnostics["opponent_pairs"]["total_assignments"] == 48 * 25
    assert first.diagnostics["teammate_pairs"]["distinct_pair_count"] == 66
    assert first.diagnostics["opponent_pairs"]["distinct_pair_count"] == 66
    assert first.diagnostics["teammate_pairs"]["spread"] <= 3
    assert first.diagnostics["opponent_pairs"]["spread"] <= 4

    assert first.diagnostics["evaluation_design"] == "independent_balanced_games"
    assert first.diagnostics["team_a_start_game_count"] == 24
    assert first.diagnostics["team_b_start_game_count"] == 24
    assert first.diagnostics["max_team_side_imbalance"] <= 4
    assert first.diagnostics["start_appearance_spread"] <= 6
    assert len({game.pair_id for game in first.games}) == len(first.games)
    assert len({game.environment_rng_root for game in first.games}) == len(first.games)
    for game in first.games:
        roster = (*game.team_a_player_ids, *game.team_b_player_ids)
        assert len(roster) == len(set(roster)) == 10


def test_saved_schedule_replays_against_its_exact_pool_and_is_immutable(tmp_path):
    pool = generate_player_pool(player_count=14, seed=12)
    schedule = build_balanced_lineup_schedule(pool, game_count=20, seed=444)

    path = write_lineup_schedule(schedule, tmp_path / "lineups.json")
    reloaded = load_lineup_schedule(path, pool=pool)

    assert reloaded.to_payload() == schedule.to_payload()
    assert (
        reloaded.diagnostics["player_game_appearances"]
        == schedule.diagnostics["player_game_appearances"]
    )
    with pytest.raises(TypeError):
        reloaded.diagnostics["game_appearance_spread"] = 0

    other_schedule = build_balanced_lineup_schedule(pool, game_count=20, seed=445)
    with pytest.raises(FileExistsError, match="immutable"):
        write_lineup_schedule(other_schedule, path)

    incompatible_pool = generate_player_pool(player_count=14, seed=13)
    with pytest.raises(ScheduleCompatibilityError, match="created for pool"):
        load_lineup_schedule(path, pool=incompatible_pool)


def test_schedule_extension_preserves_base_games_and_adds_only_new_ids():
    pool = generate_player_pool(player_count=12, seed=4)
    base = build_balanced_lineup_schedule(pool, game_count=24, seed=11)

    extension = build_lineup_schedule_extension(
        pool, base, additional_game_count=12, seed=13
    )

    assert len(extension.additional_schedule.games) == 12
    assert len(extension.combined_schedule.games) == 36
    assert extension.combined_schedule.games[:24] == base.games
    assert {game.game_id for game in extension.additional_schedule.games}.isdisjoint(
        {game.game_id for game in base.games}
    )
    assert extension.combined_schedule.extra_fields["extension"]["base_schedule_id"] == (
        base.schedule_id
    )


def test_schema_v2_paired_schedule_remains_loadable_for_historical_studies(tmp_path):
    pool = generate_player_pool(player_count=10, seed=42)
    player_ids = tuple(player.player_id for player in pool.players)
    teams = (player_ids[:5], player_ids[5:])
    games = tuple(
        ScheduledGame(
            game_id=f"legacy-pair-00000-{starter}-start",
            pair_id="legacy-pair-00000",
            starting_team=starter,
            team_a_player_ids=teams[0],
            team_b_player_ids=teams[1],
            reset_rng_root=1,
            environment_rng_root=2,
            policy_rng_root=3,
        )
        for starter in ("team_a", "team_b")
    )
    provisional = LineupSchedule(
        schema_version=2,
        schedule_id="",
        pool_id=pool.pool_id,
        pool_skill_schema_hash=pool.skill_schema_hash,
        pool_player_ids=player_ids,
        schedule_seed=9,
        team_size=5,
        optimizer_version="balanced-greedy-v1",
        candidate_count=128,
        games=games,
        diagnostics={},
    )
    diagnostics = schedule_balance_diagnostics(provisional)
    legacy = replace(provisional, diagnostics=diagnostics)
    payload = legacy.to_payload()
    payload["schedule_id"] = schedule_id_for_payload(payload)
    path = tmp_path / "legacy-paired-schedule.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    restored = load_lineup_schedule(path, pool=pool)

    assert restored.schema_version == 2
    assert restored.matchup_count == 1
    assert len(restored.games) == 2


def test_schedule_replay_validation_rejects_duplicate_sampling_groups():
    pool = generate_player_pool(player_count=10, seed=18)
    schedule = build_balanced_lineup_schedule(pool, game_count=3, seed=54)
    changed_game = replace(
        schedule.games[1],
        pair_id=schedule.games[0].pair_id,
    )
    changed = replace(
        schedule, games=(schedule.games[0], changed_game, *schedule.games[2:])
    )

    with pytest.raises(LineupScheduleError, match="sampling group"):
        validate_lineup_schedule(changed, pool=pool)


def test_schedule_payload_preserves_future_fields_and_rechecks_diagnostics():
    pool = generate_player_pool(player_count=11, seed=7)
    schedule = build_balanced_lineup_schedule(pool, game_count=5, seed=87)
    payload = schedule.to_payload()
    payload["future_schedule_setting"] = {"experiment": "next"}
    payload["games"][0]["future_game_setting"] = ["seeded"]
    payload["schedule_id"] = schedule_id_for_payload(payload)

    restored = LineupSchedule.from_payload(payload)

    assert restored.extra_fields["future_schedule_setting"] == {"experiment": "next"}
    assert list(restored.games[0].extra_fields["future_game_setting"]) == ["seeded"]

    invalid = restored.to_payload()
    invalid["diagnostics"]["game_appearance_spread"] = 999
    invalid["schedule_id"] = schedule_id_for_payload(invalid)
    with pytest.raises(LineupScheduleError, match="diagnostics"):
        LineupSchedule.from_payload(invalid)
