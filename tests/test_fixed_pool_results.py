from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from basketworld_jax.eval.fixed_pool import extend_player_pool, generate_player_pool
from basketworld_jax.eval.fixed_pool_models import fit_fixed_pool_study_models
from basketworld_jax.eval.fixed_pool_results import (
    EVENT_LEDGER_FIELDS,
    EPISODE_TABLE_FIELDS,
    FixedPoolStudyArtifactError,
    load_fixed_pool_study_artifacts,
    persist_fixed_pool_study_artifacts,
    possession_event_rows_from_trace,
)
from basketworld_jax.eval.lineup_schedule import build_balanced_lineup_schedule
from basketworld_jax.eval.skills import (
    ACTIVE_PLAYER_SKILL_REGISTRY,
    PlayerSkillDescriptor,
    PlayerSkillRegistry,
)


class _FakeMlflow:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None]] = []

    def log_artifacts(self, local_dir, artifact_path=None) -> None:
        self.calls.append((str(local_dir), artifact_path))


def _study_inputs():
    pool = generate_player_pool(player_count=10, seed=3)
    schedule = build_balanced_lineup_schedule(pool, game_count=2, seed=5)
    return pool, schedule


def _episode_result(pool, schedule, index: int, *, completed: bool = True):
    game = schedule.games[index]
    return {
        "episode": index + 1,
        "completed": completed,
        "fixed_pool_study": {
            "pool_id": pool.pool_id,
            "schedule_id": schedule.schedule_id,
            "action_mode": "direct",
            "unified_checkpoint_path": "/checkpoints/policy.msgpack",
            "unified_checkpoint_update_index": 12,
            "opponent_checkpoint_path": "/checkpoints/opponent.msgpack",
            "opponent_checkpoint_update_index": 8,
            "checkpoint_observation_schema_versions": [3],
            "environment_observation_schema_version": 3,
            "schedule_game_id": game.game_id,
            "pair_id": game.pair_id,
            "starting_team": game.starting_team,
            "team_a_player_ids": list(game.team_a_player_ids),
            "team_b_player_ids": list(game.team_b_player_ids),
            "reset_rng_root": game.reset_rng_root,
            "environment_rng_root": game.environment_rng_root,
            "policy_rng_root": game.policy_rng_root,
        },
        "game": {
            "completed": completed,
            "team_a_score": 12.0 if completed else 0.0,
            "team_b_score": 6.0 if completed else 3.0,
            "completed_possessions": 9 if completed else 2,
            "team_a_completed_possessions": 6 if completed else 0,
            "team_b_completed_possessions": 3 if completed else 2,
            "overtime_round": 0,
            "overtime_possessions_completed": 0,
        },
    }


def _event_trace():
    def values(first, second=0):
        return np.asarray([[first], [second]])

    return {
        "offense_team": values(0, 0),
        "possession_ended": values(0, 1),
        "completed_possessions": values(0, 1),
        "team_a_completed_possessions": values(0, 1),
        "team_b_completed_possessions": values(0, 0),
        "possession_end_reason": values(0, 1),
        "prior_game_phase": values(0, 0),
        "game_phase": values(0, 1),
        "team_a_score": values(0.0, 2.0),
        "team_b_score": values(0.0, 0.0),
        "shot_attempt": values(0, 1),
        "shot_success": values(0, 1),
        "shot_shooter": values(-1, 2),
        "shot_value": values(0.0, 2.0),
        "shot_expected_points": values(0.0, 1.4),
        "shot_distance": values(0.0, 3.0),
        "shot_type": values(-1, 0),
        "pass_attempts": values(1, 0),
        "pass_passer": values(0, -1),
        "pass_receiver": values(1, -1),
        "completed_passes": values(1, 0),
        "assists": values(1, 0),
        "potential_assist": values(1, 0),
        "turnovers": values(0, 1),
        "turnover_reason": values(0, 5),
        "turnover_player": values(-1, 3),
        "steal_player": values(-1, 5),
        "rebound_attempt": values(0, 1),
        "offensive_rebound": values(0, 0),
        "defensive_rebound": values(0, 1),
        "rebound_winner": values(-1, 6),
        "rebound_target_cell": values(-1, 9),
        "rebound_global_contest": values(0, 1),
        "clearance_events": values(0, 1),
        "clearance_elapsed_steps": values(0, 2),
        "turnovers_before_clearance": values(0, 1),
        "inbound_team": values(-1, 1),
        "inbound_player": values(-1, 5),
        "inbound_steps_remaining": values(0, 4),
        "check_opportunities": values(1, 0),
        "check_setup_opportunities": values(0, 0),
        "check_setup_steps": values(0, 0),
        "check_pickups": values(0, 0),
        "check_violations": values(0, 1),
        "check_pickup_steps": values(0, 3),
        "defensive_lane_violation": values(0, 1),
        "defensive_lane_violation_player": values(-1, 7),
        "offensive_three_seconds": values(0, 0),
    }


def test_fixed_pool_artifact_round_trip_normalizes_scores_and_isolates_namespace(
    tmp_path,
):
    pool, schedule = _study_inputs()
    results = [
        _episode_result(pool, schedule, 0),
        _episode_result(pool, schedule, 1, completed=False),
    ]
    game = schedule.games[0]
    events = possession_event_rows_from_trace(
        _event_trace(),
        env_index=0,
        active_steps=2,
        episode=1,
        game_metadata={
            "schedule_game_id": game.game_id,
            "pair_id": game.pair_id,
            "team_a_player_ids": list(game.team_a_player_ids),
            "team_b_player_ids": list(game.team_b_player_ids),
        },
    )
    mlflow = _FakeMlflow()
    paths = persist_fixed_pool_study_artifacts(
        pool=pool,
        schedule=schedule,
        episode_results=results,
        possession_events=events,
        output_dir=tmp_path,
        mlflow_client=mlflow,
    )

    assert paths.root.parent.name == "fixed_pool_studies"
    assert paths.manifest_path.is_file()
    assert paths.pool_manifest_path.is_file()
    assert paths.schedule_manifest_path.is_file()
    assert paths.episode_table_path.is_file()
    assert paths.event_ledger_path.is_file()
    assert mlflow.calls == [(str(paths.root), f"fixed_pool_studies/{paths.study_id}")]

    artifact = load_fixed_pool_study_artifacts(paths.root)
    assert artifact["manifest"]["tables"]["episodes"]["fields"] == list(
        EPISODE_TABLE_FIELDS
    )
    assert artifact["manifest"]["tables"]["possession_events"]["fields"] == list(
        EVENT_LEDGER_FIELDS
    )
    first, zero_possession = artifact["episodes"]
    assert first["team_a_minus_team_b_margin"] == 6.0
    assert first["team_a_points_per_offensive_possession"] == 2.0
    assert first["team_b_points_per_offensive_possession"] == 2.0
    assert first["team_a_minus_team_b_net_rating"] == 0.0
    assert first["analysis_default_included"] is True
    assert zero_possession["termination_status"] == "zero_possession"
    assert zero_possession["team_a_minus_team_b_net_rating"] is None
    assert zero_possession["analysis_default_included"] is False

    # The persisted tables and manifests are sufficient input to the offline
    # modeling layer; no evaluator objects or in-memory trace are needed.
    model_report = fit_fixed_pool_study_models(paths.root, bootstrap_samples=2)
    assert model_report["study_id"] == paths.study_id
    assert model_report["analysis_rows"]["included_game_count"] == 1
    assert model_report["action_mode"] == "direct"

    # Exact reruns are safe and preserve the immutable files byte-for-byte.
    repeated = persist_fixed_pool_study_artifacts(
        pool=pool,
        schedule=schedule,
        episode_results=results,
        possession_events=events,
        output_dir=tmp_path,
    )
    assert repeated.root == paths.root


def test_possession_ledger_captures_required_events_and_named_players():
    pool, schedule = _study_inputs()
    game = schedule.games[0]
    events = possession_event_rows_from_trace(
        _event_trace(),
        env_index=0,
        active_steps=2,
        episode=1,
        game_metadata={
            "schedule_game_id": game.game_id,
            "pair_id": game.pair_id,
            "team_a_player_ids": list(game.team_a_player_ids),
            "team_b_player_ids": list(game.team_b_player_ids),
        },
    )

    assert {event["event_type"] for event in events} >= {
        "shot",
        "pass",
        "turnover",
        "rebound",
        "clearance",
        "inbound",
        "check",
        "violation",
        "possession_end",
    }
    assert all(set(event) == set(EVENT_LEDGER_FIELDS) for event in events)
    shot = next(event for event in events if event["event_type"] == "shot")
    assert shot["actor_player_id"] == game.team_a_player_ids[2]
    assert shot["shot_expected_points"] == pytest.approx(1.4)
    turnover = next(event for event in events if event["event_type"] == "turnover")
    assert turnover["turnover_reason"] == "shot_clock_violation"
    assert turnover["actor_player_id"] == game.team_a_player_ids[3]
    assert turnover["target_player_id"] == game.team_b_player_ids[0]
    rebound = next(event for event in events if event["event_type"] == "rebound")
    assert rebound["actor_player_id"] == game.team_b_player_ids[1]
    assert rebound["rebound_defensive"] is True
    assert rebound["possession_number"] == 1


def test_fixed_pool_artifact_refuses_changed_outcomes_under_same_study_identity(
    tmp_path,
):
    pool, schedule = _study_inputs()
    results = [_episode_result(pool, schedule, 0), _episode_result(pool, schedule, 1)]
    persist_fixed_pool_study_artifacts(
        pool=pool,
        schedule=schedule,
        episode_results=results,
        possession_events=[],
        output_dir=tmp_path,
    )
    changed = deepcopy(results)
    changed[0]["game"]["team_a_score"] = 99.0
    with pytest.raises(FixedPoolStudyArtifactError, match="immutable"):
        persist_fixed_pool_study_artifacts(
            pool=pool,
            schedule=schedule,
            episode_results=changed,
            possession_events=[],
            output_dir=tmp_path,
        )


def test_raw_artifact_remains_readable_after_additive_pool_schema_evolution(tmp_path):
    legacy_registry = replace(
        ACTIVE_PLAYER_SKILL_REGISTRY,
        descriptors=tuple(
            descriptor
            for descriptor in ACTIVE_PLAYER_SKILL_REGISTRY.descriptors
            if descriptor.key != "passing"
        ),
    )
    parent = generate_player_pool(player_count=10, seed=83, registry=legacy_registry)
    evolved_pool = extend_player_pool(parent, migration_seed=89)
    schedule = build_balanced_lineup_schedule(evolved_pool, game_count=2, seed=97)
    paths = persist_fixed_pool_study_artifacts(
        pool=evolved_pool,
        schedule=schedule,
        episode_results=[
            _episode_result(evolved_pool, schedule, 0),
            _episode_result(evolved_pool, schedule, 1),
        ],
        possession_events=[],
        output_dir=tmp_path,
    )

    reloaded = load_fixed_pool_study_artifacts(paths.root)
    assert reloaded["pool_manifest"]["parent_pool_id"] == parent.pool_id
    assert all(
        "passing" in player["skills"] for player in reloaded["pool_manifest"]["players"]
    )
    # Historical raw studies retain their self-describing pool manifest after
    # additive evolution; later work uses a child pool instead of mutation.
    future_registry = PlayerSkillRegistry(
        schema_version=2,
        descriptors=(
            *ACTIVE_PLAYER_SKILL_REGISTRY.descriptors,
            PlayerSkillDescriptor(
                key="screen_navigation",
                display_label="Screen navigation",
                numeric_type="float",
                minimum=-2.0,
                maximum=2.0,
                sampling_distribution="normal",
                sampling_defaults=(("default_mean", 0.0), ("default_std", 0.5)),
                semantic_group="defense",
                runtime_field="screen_navigation",
            ),
        ),
    )
    future_child = extend_player_pool(
        evolved_pool, migration_seed=101, registry=future_registry
    )
    assert future_child.parent_pool_id == evolved_pool.pool_id
    assert (
        load_fixed_pool_study_artifacts(paths.root)["manifest"]["study_id"]
        == paths.study_id
    )
