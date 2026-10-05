from __future__ import annotations

import json
from uuid import uuid4

from app.training_backend.database import TrainingRepository, utc_now


def _payload(run_id: str) -> dict:
    now = utc_now()
    return {
        "id": run_id,
        "idempotency_key": f"key-{run_id}",
        "name": "smoke run",
        "preset": "halfcourt_multi_possession",
        "status": "queued",
        "config_json": json.dumps({"num_updates": 5}),
        "command_json": json.dumps(["train", "--num-updates", "5"]),
        "environment_json": json.dumps({"NUM_UPDATES": "5"}),
        "cwd": "/repo",
        "created_at": now,
        "updated_at": now,
        "target_updates": 5,
        "control_file": "/tmp/control.json",
        "status_file": "/tmp/status.json",
        "log_file": "/tmp/train.log",
    }


def test_repository_round_trip_and_events(tmp_path):
    repository = TrainingRepository(tmp_path / "training.duckdb")
    repository.initialize()
    run_id = str(uuid4())

    created = repository.create_run(_payload(run_id))
    assert created["config"] == {"num_updates": 5}
    assert created["command"][-1] == "5"
    assert repository.get_run_by_idempotency_key(f"key-{run_id}")["id"] == run_id

    updated = repository.update_run(
        run_id,
        status="running",
        current_update=3,
        pid=123,
        process_start_token="456",
    )
    assert updated["status"] == "running"
    assert updated["current_update"] == 3
    assert repository.list_active_runs()[0]["id"] == run_id

    repository.append_event(
        {
            "id": str(uuid4()),
            "run_id": run_id,
            "event_type": "started",
            "status": "running",
            "message": None,
            "payload": {"pid": 123},
            "created_at": utc_now(),
        }
    )
    assert repository.list_events(run_id)[0]["payload"] == {"pid": 123}
