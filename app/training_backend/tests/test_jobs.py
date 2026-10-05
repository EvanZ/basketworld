from __future__ import annotations

import time

from app.training_backend.config import Settings
from app.training_backend.database import TrainingRepository
from app.training_backend.jobs import JobManager
from app.training_backend.models import TrainingRunConfig


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
