from __future__ import annotations

import json
import time

import pytest

from app.training_backend.config import Settings
from app.training_backend.database import TrainingRepository
from app.training_backend.jobs import JobManager, ProvenanceConfirmationRequired
from app.training_backend.models import RunResumeRequest, TrainingRunConfig


def _write_minimal_checkpoint_metadata(path) -> None:
    (path / "state").mkdir(parents=True, exist_ok=True)
    (path / "metadata.json").write_text(
        json.dumps(
            {
                "checkpoint_version": 2,
                "saved_at": "2026-01-01T00:00:00+00:00",
                "update_index": 1,
                "trainer_config": {},
                "frozen_config": {},
                "policy_spec": {
                    "model_type": "attention",
                    "flat_obs_dim": 32,
                    "training_player_count": 2,
                    "action_dim_per_player": 16,
                    "total_action_dim": 32,
                    "observation_schema_version": 3,
                },
            }
        ),
        encoding="utf-8",
    )


@pytest.fixture
def paused_schedule_run(tmp_path, monkeypatch):
    settings = Settings(
        repo_root=tmp_path / "repo", data_dir=tmp_path / "data",
        database_path=tmp_path / "data" / "training.duckdb",
        cors_origins=(), poll_interval_seconds=0.01,
    )
    settings.ensure_directories()
    repository = TrainingRepository(settings.database_path)
    repository.initialize()
    manager = JobManager(settings, repository)
    # Exercise persistence and command construction without launching training.
    monkeypatch.setattr(manager, "_launch", lambda run: run)
    run = manager.create_and_launch(
        config=TrainingRunConfig(num_updates=30_000, historical_eval_updates=[100, 10_000, 30_000, 40_000]),
        idempotency_key="resume-schedule-test",
    )
    checkpoint = settings.data_dir / "runs" / run["id"] / "checkpoints" / "latest"
    _write_minimal_checkpoint_metadata(checkpoint)
    metadata_path = checkpoint / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["update_index"] = 10_000
    metadata_path.write_text(json.dumps(metadata))
    artifact = checkpoint.parent / "historical_opponent_evaluation.json"
    artifact.write_text('{"matches": [{"candidate_update": 10000}]}')
    run = repository.update_run(
        run["id"], status="paused", current_update=10_000,
        checkpoint_path=str(checkpoint), mlflow_run_id="same-mlflow-run",
    )
    return manager, repository, run, artifact


@pytest.mark.parametrize("added, error", [
    ([10_000], "after the current"), ([9999], "after the current"),
    ([30_001], "no later than"), ([15_000, 15_000], "only once"),
    ([30_000], "already in"), ([True], "whole-number"), ([15_000.5], "whole-number"),
])
def test_resume_rejects_invalid_schedule_without_mutation(paused_schedule_run, added, error):
    manager, repository, run, artifact = paused_schedule_run
    original_events = repository.list_events(run["id"])
    original_artifact = artifact.read_bytes()
    with pytest.raises(RuntimeError, match=error):
        manager.resume(run["id"], additional_historical_eval_updates=added)
    assert repository.get_run(run["id"]) == run
    assert repository.list_events(run["id"]) == original_events
    assert artifact.read_bytes() == original_artifact


def test_resume_appends_schedule_in_same_run_after_preflight(paused_schedule_run):
    manager, repository, run, artifact = paused_schedule_run
    original_artifact = artifact.read_bytes()
    with pytest.raises(ProvenanceConfirmationRequired):
        manager.resume(run["id"], additional_historical_eval_updates=[25_000, 15_000])
    assert repository.get_run(run["id"]) == run
    resumed = manager.resume(
        run["id"], additional_historical_eval_updates=[25_000, 15_000],
        confirm_provenance_mismatch=True,
    )
    assert resumed["id"] == run["id"]
    assert resumed["mlflow_run_id"] == "same-mlflow-run"
    assert resumed["target_updates"] == 30_000
    assert resumed["config"]["historical_eval_updates"] == [100, 10_000, 15_000, 25_000, 30_000, 40_000]
    command = resumed["command"]
    assert command[command.index("--historical-eval-updates") + 1] == "100,10000,15000,25000,30000"
    assert command[command.index("--resume-checkpoint") + 1] == run["checkpoint_path"]
    assert command[command.index("--mlflow-resume-run-id") + 1] == "same-mlflow-run"
    event = next(e for e in repository.list_events(run["id"]) if e["event_type"] == "resume_requested")
    assert event["payload"]["previous_historical_eval_updates"] == [100, 10_000, 30_000, 40_000]
    assert event["payload"]["added_historical_eval_updates"] == [15_000, 25_000]
    assert event["payload"]["checkpoint_update"] == 10_000
    assert artifact.read_bytes() == original_artifact


@pytest.mark.parametrize("saved, added, expected", [
    (None, [], None),
    (None, [15_000], [100, 500, 2500, 5000, 10_000, 15_000, 20_000, 30_000]),
    ([], [15_000], [15_000]),
])
def test_resume_preserves_default_or_disabled_schedules(paused_schedule_run, saved, added, expected):
    manager, repository, run, _ = paused_schedule_run
    config = {**run["config"], "historical_eval_updates": saved}
    repository.update_run(run["id"], config_json=json.dumps(config))
    resumed = manager.resume(
        run["id"], additional_historical_eval_updates=added,
        confirm_provenance_mismatch=True,
    )
    assert resumed["config"]["historical_eval_updates"] == expected


def test_resume_schedule_checks_durable_checkpoint_not_only_status(paused_schedule_run):
    manager, repository, run, _ = paused_schedule_run
    repository.update_run(run["id"], current_update=8000)
    with pytest.raises(RuntimeError, match="saved checkpoint"):
        manager.resume(run["id"], additional_historical_eval_updates=[9000], confirm_provenance_mismatch=True)
    assert repository.get_run(run["id"])["config"] == run["config"]


@pytest.mark.parametrize("updates", [[True], [1.5], [-1], [0], [15000, 15000]])
def test_resume_request_rejects_invalid_milestones(updates):
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        RunResumeRequest(additional_historical_eval_updates=updates)


def test_resume_api_keeps_additions_through_confirmation(paused_schedule_run, monkeypatch):
    from fastapi.testclient import TestClient
    from app.training_backend import main

    manager, repository, run, _ = paused_schedule_run
    monkeypatch.setattr(main, "manager", manager)
    client = TestClient(main.app)
    url = f"/api/v1/runs/{run['id']}/resume"
    body = {"additional_historical_eval_updates": [15_000, 20_000]}
    response = client.post(url, json=body)
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "provenance_confirmation_required"
    assert repository.get_run(run["id"])["config"] == run["config"]
    response = client.post(url, json={**body, "confirm_provenance_mismatch": True})
    assert response.status_code == 200
    assert response.json()["run"]["mlflow_run_id"] == "same-mlflow-run"
    assert response.json()["run"]["config"]["historical_eval_updates"] == [100, 10_000, 15_000, 20_000, 30_000, 40_000]


def test_job_manager_launches_and_completes_detached_worker(tmp_path):
    repo_root = tmp_path / "repo"
    python_bin = repo_root / ".env" / "bin" / "python"
    python_bin.parent.mkdir(parents=True)
    python_bin.write_text(
        "#!/usr/bin/env bash\necho fake-training-worker\n", encoding="utf-8"
    )
    python_bin.chmod(0o755)

    data_dir = tmp_path / "data"
    settings = Settings(
        repo_root=repo_root,
        data_dir=data_dir,
        database_path=data_dir / "training.duckdb",
        cors_origins=("http://localhost:5174",),
        poll_interval_seconds=0.01,
    )
    settings.ensure_directories()
    repository = TrainingRepository(settings.database_path)
    repository.initialize()
    manager = JobManager(settings, repository)

    run = manager.create_and_launch(
        config=TrainingRunConfig(num_updates=1),
        idempotency_key="job-launch-test",
    )
    assert run["status"] == "running"
    assert run["pid"] is not None

    for _ in range(100):
        manager.poll()
        run = repository.get_run(run["id"])
        if run["status"] == "completed":
            break
        time.sleep(0.01)

    assert run["status"] == "completed"
    assert "fake-training-worker" in open(run["log_file"], encoding="utf-8").read()
    assert (
        manager.create_and_launch(
            config=TrainingRunConfig(num_updates=1),
            idempotency_key="job-launch-test",
        )["id"]
        == run["id"]
    )


def test_job_manager_records_intentional_stop_as_stopped(tmp_path):
    repo_root = tmp_path / "repo"
    python_bin = repo_root / ".env" / "bin" / "python"
    python_bin.parent.mkdir(parents=True)
    python_bin.write_text(
        "#!/usr/bin/env bash\nexec sleep 30\n", encoding="utf-8"
    )
    python_bin.chmod(0o755)

    data_dir = tmp_path / "data"
    settings = Settings(
        repo_root=repo_root,
        data_dir=data_dir,
        database_path=data_dir / "training.duckdb",
        cors_origins=("http://localhost:5174",),
        poll_interval_seconds=0.01,
    )
    settings.ensure_directories()
    repository = TrainingRepository(settings.database_path)
    repository.initialize()
    manager = JobManager(settings, repository)

    run = manager.create_and_launch(
        config=TrainingRunConfig(num_updates=1),
        idempotency_key="job-stop-test",
    )
    assert run["status"] == "running"

    run = manager.request_stop(run["id"])
    assert run["status"] == "stopping"

    for _ in range(100):
        manager.poll()
        run = repository.get_run(run["id"])
        if run["status"] == "stopped":
            break
        time.sleep(0.01)

    assert run["status"] == "stopped"
    assert run["error_message"] is None


def test_job_manager_continues_completed_run_in_same_mlflow_run(tmp_path):
    repo_root = tmp_path / "repo"
    python_bin = repo_root / ".env" / "bin" / "python"
    python_bin.parent.mkdir(parents=True)
    python_bin.write_text(
        "#!/usr/bin/env bash\necho fake-training-worker\n", encoding="utf-8"
    )
    python_bin.chmod(0o755)

    data_dir = tmp_path / "data"
    settings = Settings(
        repo_root=repo_root,
        data_dir=data_dir,
        database_path=data_dir / "training.duckdb",
        cors_origins=("http://localhost:5174",),
        poll_interval_seconds=0.01,
    )
    settings.ensure_directories()
    repository = TrainingRepository(settings.database_path)
    repository.initialize()
    manager = JobManager(settings, repository)

    run = manager.create_and_launch(
        config=TrainingRunConfig(
            num_updates=1,
            historical_eval_updates=[1, 30_000],
        ),
        idempotency_key="job-continue-test",
    )
    for _ in range(100):
        manager.poll()
        run = repository.get_run(run["id"])
        if run["status"] == "completed":
            break
        time.sleep(0.01)
    assert run["status"] == "completed"

    checkpoint = data_dir / "runs" / run["id"] / "checkpoints" / "latest"
    _write_minimal_checkpoint_metadata(checkpoint)
    repository.update_run(
        run["id"],
        checkpoint_path=str(checkpoint),
        mlflow_run_id="original-mlflow-run-id",
    )

    with pytest.raises(RuntimeError, match="after the completed update"):
        manager.continue_completed(
            run["id"],
            additional_updates=10_000,
            additional_historical_eval_updates=[1],
        )
    with pytest.raises(RuntimeError, match="no later than"):
        manager.continue_completed(
            run["id"],
            additional_updates=10_000,
            additional_historical_eval_updates=[10_002],
        )
    with pytest.raises(RuntimeError, match="only once"):
        manager.continue_completed(
            run["id"],
            additional_updates=10_000,
            additional_historical_eval_updates=[2_500, 2_500],
        )
    with pytest.raises(ProvenanceConfirmationRequired):
        manager.continue_completed(
            run["id"],
            additional_updates=10_000,
            additional_historical_eval_updates=[2_500, 10_001],
        )

    continued = manager.continue_completed(
        run["id"],
        additional_updates=10_000,
        additional_historical_eval_updates=[2_500, 10_001],
        confirm_provenance_mismatch=True,
    )

    assert continued["status"] == "running"
    assert continued["target_updates"] == 10_001
    assert continued["config"]["num_updates"] == 10_001
    assert continued["config"]["historical_eval_updates"] == [
        1,
        2_500,
        10_001,
        30_000,
    ]
    assert "--resume-checkpoint" in continued["command"]
    assert str(checkpoint) in continued["command"]
    assert "--mlflow-resume-run-id" in continued["command"]
    assert "original-mlflow-run-id" in continued["command"]
    historical_index = continued["command"].index("--historical-eval-updates")
    assert continued["command"][historical_index + 1] == "1,2500,10001"
    events = repository.list_events(run["id"])
    assert events[0]["event_type"] == "started"
    assert events[1]["event_type"] == "continuation_requested"
    payload = events[1]["payload"]
    assert payload["additional_updates"] == 10_000
    assert payload["previous_target_updates"] == 1
    assert payload["target_updates"] == 10_001
    assert payload["previous_historical_eval_updates"] == [1, 30_000]
    assert payload["added_historical_eval_updates"] == [2_500, 10_001]
    assert payload["resulting_historical_eval_updates"] == [
        1,
        2_500,
        10_001,
        30_000,
    ]
    assert payload["checkpoint_path"] == str(checkpoint)
    assert payload["checkpoint_update"] == 1
    assert payload["checkpoint_provenance"] is None
    assert payload["provenance_comparison"]["status"] == "legacy"
    assert payload["provenance_confirmation"] == {
        "required": True,
        "confirmed": True,
    }


def test_job_manager_recovers_stopped_run_from_validated_checkpoint(tmp_path):
    repo_root = tmp_path / "repo"
    python_bin = repo_root / ".env" / "bin" / "python"
    python_bin.parent.mkdir(parents=True)
    python_bin.write_text(
        "#!/usr/bin/env bash\necho fake-training-worker\n", encoding="utf-8"
    )
    python_bin.chmod(0o755)

    data_dir = tmp_path / "data"
    settings = Settings(
        repo_root=repo_root,
        data_dir=data_dir,
        database_path=data_dir / "training.duckdb",
        cors_origins=("http://localhost:5174",),
        poll_interval_seconds=0.01,
    )
    settings.ensure_directories()
    repository = TrainingRepository(settings.database_path)
    repository.initialize()
    manager = JobManager(settings, repository)

    run = manager.create_and_launch(
        config=TrainingRunConfig(num_updates=5),
        idempotency_key="job-stopped-recovery-test",
    )
    for _ in range(100):
        manager.poll()
        run = repository.get_run(run["id"])
        if run["status"] == "completed":
            break
        time.sleep(0.01)
    assert run["status"] == "completed"

    checkpoint = data_dir / "runs" / run["id"] / "checkpoints" / "latest"
    _write_minimal_checkpoint_metadata(checkpoint)
    repository.update_run(
        run["id"], status="stopped", checkpoint_path=str(checkpoint)
    )

    recovered = manager.resume(
        run["id"], confirm_provenance_mismatch=True
    )

    assert recovered["status"] == "running"
    events = repository.list_events(run["id"])
    recovery = next(event for event in events if event["event_type"] == "stopped_recovery_requested")
    assert recovery["payload"]["checkpoint_path"] == str(checkpoint)
    assert recovery["payload"]["provenance_confirmation"]["confirmed"] is True


def test_resume_preflight_defaults_missing_team_clearance_to_false(tmp_path):
    settings = Settings(
        repo_root=tmp_path / "repo", data_dir=tmp_path / "data",
        database_path=tmp_path / "data" / "training.duckdb",
        cors_origins=(), poll_interval_seconds=0.01,
    )
    settings.ensure_directories()
    manager = JobManager(settings, TrainingRepository(settings.database_path))
    config = TrainingRunConfig()
    run_id = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
    preview = manager.preview(config, run_id=run_id)
    checkpoint = tmp_path / "checkpoint"
    _write_minimal_checkpoint_metadata(checkpoint)
    metadata_path = checkpoint / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    saved = manager._resolved_config_from_preview(preview)
    saved.pop("require_all_offense_clear")
    metadata["resolved_training_config"] = saved
    metadata_path.write_text(json.dumps(metadata))
    manager._preflight_resume(checkpoint=str(checkpoint), preview=preview)
    enabled_preview = manager.preview(
        config.model_copy(update={"require_all_offense_clear": True}), run_id=run_id
    )
    with pytest.raises(RuntimeError, match="require_all_offense_clear"):
        manager._preflight_resume(checkpoint=str(checkpoint), preview=enabled_preview)
