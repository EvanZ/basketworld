from __future__ import annotations

import json
import time

import pytest

from app.training_backend.config import Settings
from app.training_backend.database import TrainingRepository
from app.training_backend.jobs import JobManager, ProvenanceConfirmationRequired
from app.training_backend.models import TrainingRunConfig


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
