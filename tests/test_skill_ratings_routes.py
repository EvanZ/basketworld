import json
from pathlib import Path

import pytest
from fastapi import HTTPException

from app.backend.routes import skill_ratings_routes as routes
from basketworld_jax.eval.fixed_pool import generate_player_pool, write_player_pool
from basketworld_jax.eval.lineup_schedule import (
    build_balanced_lineup_schedule,
    write_lineup_schedule,
)


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_report_payload_exposes_profiles_and_dynamic_skill_labels(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(routes, "REPO_ROOT", tmp_path)
    artifact_dir = tmp_path / "artifacts" / "skill_ratings" / "example"
    pool_path = _write_json(
        artifact_dir / "fixed_pool_studies" / "study-a" / "pool_manifest.json",
        {
            "skill_descriptors": [
                {
                    "key": "rebound",
                    "display_label": "Rebounding",
                    "semantic_group": "shared",
                },
                {
                    "key": "new_skill",
                    "display_label": "New Skill",
                    "semantic_group": "offense",
                },
            ],
            "players": [
                {
                    "player_id": "player-a",
                    "name": "Player A",
                    "skills": {"rebound": 1.5, "new_skill": -0.25},
                }
            ],
        },
    )
    report_path = _write_json(
        artifact_dir / "fixed_pool_reports" / "study-a" / "checkpoint_report.json",
        {
            "kind": "fixed_pool_checkpoint_report",
            "checkpoint": {"path": "checkpoint", "update_index": 12},
            "action_mode": "direct",
            "raw_artifact_dependencies": {
                "study_id": "study-a",
                "pool_manifest": str(pool_path),
            },
            "player_rapm": {
                "players": {
                    "player-a": {
                        "net_rating_points_vs_replacement": 3.5,
                        "bootstrap_interval": {"p05": 1.0, "p95": 5.0},
                    }
                },
                "fit": {"r_squared": 0.4},
                "validation": {"r_squared": 0.2},
            },
            "direct_skill_ratings": {
                "players": {
                    "player-a": {
                        "net_rating_points_vs_replacement": 2.75,
                        "bootstrap_interval": {"p05": 1.25, "p95": 4.25},
                        "apm_diagnostic_value": 3.5,
                        "apm_minus_skill_rating": 0.75,
                    }
                },
                "fit": {"r_squared": 0.3},
                "validation": {"r_squared": 0.1},
                "coefficients": {
                    "new_skill": {
                        "net_rating_points_per_pool_standard_deviation": 2.5,
                        "pool_mean": 0.0,
                        "pool_standard_deviation": 1.0,
                    }
                },
                "apm_agreement": {
                    "sample_count": 1,
                    "rating_pearson_correlation": None,
                    "ranking_spearman_correlation": None,
                    "overlap_k": 1,
                    "top_k_overlap_count": 1,
                    "bottom_k_overlap_count": 1,
                },
            },
            "skill_coefficients": {
                "coefficients": {
                    "new_skill": {
                        "net_rating_points_per_standard_deviation": 2.0,
                        "pool_mean": 0.0,
                        "pool_standard_deviation": 1.0,
                    }
                },
                "predicted_player_value": {"player-a": 2.0},
                "residual_player_value": {"player-a": 1.5},
            },
            "conventional_three_factor_impact": {
                "scenario_weights": {
                    "factor_weights": {
                        "effective_field_goal_pct": {
                            "percent_of_total_margin_delta": 50.0
                        }
                    }
                },
                "unavailable_factors": {"free_throw_rate": {"status": "unavailable"}},
            },
        },
    )

    payload = routes._ui_report(report_path)

    assert payload["summary"]["path"] == str(report_path)
    assert payload["summary"]["analysis_label"] == "Original report"
    assert payload["player_rows"][0]["name"] == "Player A"
    assert payload["player_rows"][0]["net_rating"] == 2.75
    assert payload["player_rows"][0]["apm_rating"] == 3.5
    assert payload["player_rows"][0]["apm_minus_skill_rating"] == 0.75
    assert payload["skill_apm_agreement"]["top_k_overlap_count"] == 1
    assert payload["player_rows"][0]["skills"][1] == {
        "key": "new_skill",
        "label": "New Skill",
        "group": "offense",
        "value": -0.25,
    }
    assert payload["skill_coefficients"][0]["net_rating_per_standard_deviation"] == 2.5
    assert (
        payload["three_factors"]["unavailable_factors"]["free_throw_rate"]["status"]
        == "unavailable"
    )


def test_command_uses_isolated_study_cli(tmp_path, monkeypatch):
    monkeypatch.setattr(routes, "REPO_ROOT", tmp_path)
    config = routes.SkillRatingsJobRequest(
        checkpoint_paths=["checkpoints/update_0000012"],
        pool_path="inputs/pool.json",
        schedule_path="inputs/schedule.json",
        artifact_dir="artifacts/skill_ratings/example",
    )
    resolved = {
        "checkpoint_paths": [tmp_path / "checkpoints" / "update_0000012"],
        "pool_path": tmp_path / "inputs" / "pool.json",
        "schedule_path": tmp_path / "inputs" / "schedule.json",
        "artifact_dir": tmp_path / "artifacts" / "skill_ratings" / "example",
        "report_dir": tmp_path / "artifacts" / "skill_ratings" / "example",
    }

    command = routes._build_command(config, resolved)

    assert "--run-fixed-pool-study" in command
    assert "--fixed-pool-create" in command
    assert "--fixed-pool-schedule-create" in command
    assert command[command.index("--fixed-pool-action-mode") + 1] == "direct"
    assert command[command.index("--fixed-pool-games") + 1] == "0"
    assert "--fixed-pool-ridge-alpha" not in command


def test_report_library_search_keeps_prior_timestamped_studies_visible(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(routes, "REPO_ROOT", tmp_path)
    _write_json(
        tmp_path
        / "artifacts"
        / "skill_ratings"
        / "older-study"
        / "fixed_pool_reports"
        / "study-a"
        / "checkpoint_report.json",
        {
            "kind": "fixed_pool_checkpoint_report",
            "checkpoint": {"path": "checkpoint", "update_index": 1},
            "raw_artifact_dependencies": {"study_id": "study-a"},
            "action_mode": "direct",
        },
    )
    _write_json(
        tmp_path
        / "artifacts"
        / "skill_ratings"
        / "newer-study"
        / "fixed_pool_reports"
        / "study-b"
        / "checkpoint_report.json",
        {
            "kind": "fixed_pool_checkpoint_report",
            "checkpoint": {"path": "checkpoint", "update_index": 2},
            "raw_artifact_dependencies": {"study_id": "study-b"},
            "action_mode": "direct",
        },
    )
    revision_path = _write_json(
        tmp_path
        / "artifacts"
        / "skill_ratings"
        / "newer-study"
        / "fixed_pool_reports"
        / "study-b"
        / "analysis_revisions"
        / "v3.json",
        {
            "schema_version": 4,
            "kind": "fixed_pool_checkpoint_reanalysis",
            "checkpoint": {"path": "checkpoint", "update_index": 2},
            "raw_artifact_dependencies": {"study_id": "study-b"},
            "action_mode": "direct",
            "reanalysis": {"schema_version": 3},
        },
    )

    payload = routes.list_skill_ratings_reports("artifacts/skill_ratings")

    assert {report["study_id"] for report in payload["reports"]} == {
        "study-a",
        "study-b",
    }
    revision = next(
        report for report in payload["reports"] if report["path"] == str(revision_path)
    )
    assert revision["analysis_label"] == "Reanalysis v3"
    assert revision["analysis_revision"] == 3


def test_active_job_snapshot_survives_a_tab_remount():
    manager = routes._SkillRatingsJobManager()
    manager._jobs["job-1"] = {
        "id": "job-1",
        "status": "running",
        "created_at": 1.0,
        "updated_at": 1.0,
        "started_at": 1.0,
        "finished_at": None,
        "config": {},
        "events": [{"phase": "batch_started", "at": 1.0}],
        "logs": [],
        "report_paths": [],
        "error": None,
        "return_code": None,
    }
    manager._active_job_id = "job-1"

    assert manager.active_snapshot()["id"] == "job-1"
    assert manager.active_snapshot()["events"][0]["phase"] == "batch_started"


def test_prelaunch_validation_rejects_schedule_from_a_different_pool(tmp_path):
    pool = generate_player_pool(player_count=10, seed=1)
    other_pool = generate_player_pool(player_count=10, seed=2)
    pool_path = write_player_pool(pool, tmp_path / "pool.json")
    schedule_path = write_lineup_schedule(
        build_balanced_lineup_schedule(other_pool, game_count=2, seed=3),
        tmp_path / "schedule.json",
    )
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    config = routes.SkillRatingsJobRequest(
        checkpoint_paths=[str(checkpoint)],
        pool_path=str(pool_path),
        create_pool=False,
        schedule_path=str(schedule_path),
        create_schedule=False,
        artifact_dir=str(tmp_path / "output"),
    )

    with pytest.raises(HTTPException, match="created for pool"):
        routes._validate_request_inputs(
            config,
            {
                "checkpoint_paths": [checkpoint],
                "pool_path": pool_path,
                "schedule_path": schedule_path,
            },
        )


def test_prelaunch_validation_rejects_new_pool_with_reused_schedule(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    schedule = tmp_path / "schedule.json"
    schedule.write_text("{}", encoding="utf-8")
    config = routes.SkillRatingsJobRequest(
        checkpoint_paths=[str(checkpoint)],
        pool_path=str(tmp_path / "new-pool.json"),
        create_pool=True,
        schedule_path=str(schedule),
        create_schedule=False,
        artifact_dir=str(tmp_path / "output"),
    )

    with pytest.raises(HTTPException, match="new player pool requires a new schedule"):
        routes._validate_request_inputs(
            config,
            {
                "checkpoint_paths": [checkpoint],
                "pool_path": tmp_path / "new-pool.json",
                "schedule_path": schedule,
            },
        )


def _synchronous_job_dependencies(monkeypatch, tmp_path, *, return_code=0):
    resolved = {
        "checkpoint_paths": [tmp_path / "checkpoint"],
        "checkpoint_root": None,
        "pool_path": tmp_path / "pool.json",
        "schedule_path": tmp_path / "schedule.json",
        "artifact_dir": tmp_path / "artifacts",
        "report_dir": tmp_path / "reports",
        "extension_source": None,
    }
    monkeypatch.setattr(routes, "_resolve_request_paths", lambda _config: resolved)
    monkeypatch.setattr(routes, "_validate_request_inputs", lambda *_args: None)
    monkeypatch.setattr(
        routes, "_validate_checkpoint_compatibility", lambda *_args: None
    )
    monkeypatch.setattr(routes, "_build_command", lambda *_args: ["fake-evaluator"])
    monkeypatch.setattr(
        routes,
        "_discover_reports",
        lambda *_args: [str(tmp_path / "reports" / "study" / "checkpoint_report.json")],
    )

    output = [
        "[fixed-pool] batch_started completed=0/128 batch=1 elapsed=0.0s\n",
        "[fixed-pool] batch_running elapsed=10.0s\n",
        "[fixed-pool] batch_completed completed=64/128 batch=1 elapsed=12.5s\n",
    ]
    if return_code:
        output.append("ScheduleCompatibilityError: schedule belongs to another pool\n")

    class FakeProcess:
        stdout = iter(output)

        def wait(self):
            return return_code

    class SynchronousThread:
        def __init__(self, *, target, args, **_kwargs):
            self.target = target
            self.args = args

        def start(self):
            self.target(*self.args)

    monkeypatch.setattr(
        routes.subprocess, "Popen", lambda *_args, **_kwargs: FakeProcess()
    )
    monkeypatch.setattr(routes.threading, "Thread", SynchronousThread)


def test_launch_progress_and_completed_job_lifecycle(monkeypatch, tmp_path):
    _synchronous_job_dependencies(monkeypatch, tmp_path)
    manager = routes._SkillRatingsJobManager()
    config = routes.SkillRatingsJobRequest(
        checkpoint_paths=["checkpoint"],
        pool_path="pool.json",
        schedule_path="schedule.json",
        artifact_dir="artifacts",
    )

    snapshot = manager.start(config)

    assert snapshot["status"] == "completed"
    assert snapshot["return_code"] == 0
    assert snapshot["report_paths"][0].endswith("checkpoint_report.json")
    assert [event["phase"] for event in snapshot["events"]] == [
        "batch_started",
        "batch_running",
        "batch_completed",
    ]
    assert snapshot["events"][-1]["completed"] == 64
    assert snapshot["events"][-1]["total"] == 128


def test_failed_job_keeps_actionable_evaluator_output(monkeypatch, tmp_path):
    _synchronous_job_dependencies(monkeypatch, tmp_path, return_code=2)
    manager = routes._SkillRatingsJobManager()
    config = routes.SkillRatingsJobRequest(
        checkpoint_paths=["checkpoint"],
        pool_path="pool.json",
        schedule_path="schedule.json",
        artifact_dir="artifacts",
    )

    snapshot = manager.start(config)

    assert snapshot["status"] == "failed"
    assert snapshot["return_code"] == 2
    assert snapshot["error"] == (
        "The Skill Ratings evaluator exited with code 2. "
        "Last output: ScheduleCompatibilityError: schedule belongs to another pool"
    )
    assert any(
        "ScheduleCompatibilityError" in entry["message"] for entry in snapshot["logs"]
    )
