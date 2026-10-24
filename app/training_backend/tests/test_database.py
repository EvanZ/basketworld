from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from app.training_backend.database import TrainingRepository, utc_now
from app.training_backend.run_timing import active_training_seconds


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


def test_timing_history_is_complete_and_scoped_to_requested_runs(tmp_path):
    repository = TrainingRepository(tmp_path / "training.duckdb")
    repository.initialize()
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for run_id in ("resumed", "other", "queued"):
        repository.create_run(_payload(run_id))
    events = [("resumed", "started", 0), ("resumed", "paused", 20)]
    events += [("resumed", "checkpoint_requested", 21 + i) for i in range(205)]
    events += [
        ("resumed", "started", 300), ("resumed", "paused", 330),
        ("other", "started", 0), ("other", "completed", 100),
    ]
    for run_id, event_type, seconds in events:
        repository.append_event({
            "id": str(uuid4()), "run_id": run_id, "event_type": event_type,
            "created_at": start + timedelta(seconds=seconds),
        })

    history = repository.timing_events_by_run(["resumed", "queued"])
    assert set(history) == {"resumed", "queued"}
    assert len(history["resumed"]) == 4
    assert active_training_seconds(history["resumed"], now=start + timedelta(days=1)) == 50
    assert history["queued"] == []
    assert repository.timing_events_by_run([]) == {}


def test_runs_endpoint_includes_elapsed_training_time(tmp_path, monkeypatch):
    from app.training_backend import main
    from fastapi.testclient import TestClient

    repository = TrainingRepository(tmp_path / "training.duckdb")
    repository.initialize()
    repository.create_run(_payload("paused"))
    repository.update_run("paused", status="paused", current_update=2)
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for event_type, seconds in [("started", 0), ("paused", 40)]:
        repository.append_event({
            "id": str(uuid4()), "run_id": "paused", "event_type": event_type,
            "created_at": start + timedelta(seconds=seconds),
        })
    monkeypatch.setattr(main, "repository", repository)
    monkeypatch.setattr(main.manager, "poll", lambda: None)

    response = TestClient(main.app).get("/api/v1/runs")
    assert response.status_code == 200
    assert response.json()[0]["training_elapsed_seconds"] == 40
    assert response.json()[0]["current_update"] == 2


def test_runs_endpoint_elo_badges_show_latest_and_isolate_bad_artifacts(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from fastapi.testclient import TestClient
    from app.training_backend import main
    from app.training_backend.historical_elo import HISTORICAL_EVAL_FILENAME

    repository = TrainingRepository(tmp_path / "training.duckdb")
    repository.initialize()
    for run_id in ("ready", "waiting", "corrupt", "outside"):
        run_root = tmp_path / "runs" / run_id
        checkpoints = run_root / "checkpoints"
        checkpoints.mkdir(parents=True)
        payload = _payload(run_id)
        payload["status_file"] = str(run_root / "status.json")
        if run_id == "outside":
            payload["status_file"] = str(tmp_path / "outside" / "status.json")
        repository.create_run(payload)
        artifact = checkpoints / HISTORICAL_EVAL_FILENAME
        if run_id == "corrupt":
            artifact.write_text("{broken", encoding="utf-8")
        elif run_id == "ready":
            artifact.write_text(json.dumps({
                "schema_version": 2,
                "milestone_updates": [100, 250, 500],
                "milestone_checkpoints": {"100": {}, "250": {}, "500": {}},
                "matches": [
                    {
                        "action_mode": "deterministic",
                        "candidate_update": candidate,
                        "opponent_update": opponent,
                        "candidate_wins": wins,
                        "candidate_losses": losses,
                        "candidate_ties": 0,
                        "mean_candidate_point_differential": 0.0,
                        "completed_episode_count": wins + losses,
                        "truncated_episode_count": 1,
                    }
                    for candidate, opponent, wins, losses in [
                        (250, 100, 16, 4), (500, 250, 2, 18),
                    ]
                ],
            }), encoding="utf-8")
    monkeypatch.setattr(main, "repository", repository)
    monkeypatch.setattr(main, "settings", SimpleNamespace(data_dir=tmp_path))
    monkeypatch.setattr(main.manager, "poll", lambda: None)

    client = TestClient(main.app)
    response = client.get("/api/v1/runs")
    assert response.status_code == 200
    badges = {run["id"]: run["elo_badge"] for run in response.json()}
    detail = client.get("/api/v1/runs/ready/elo").json()
    assert badges["ready"]["status"] == "ready"
    assert badges["ready"]["update"] == 500
    assert badges["ready"]["rating"] == detail["latest"]["elo_rating"]
    assert badges["ready"]["rating"] < max(point["elo_rating"] for point in detail["points"])
    assert badges["ready"]["warnings"] == detail["warnings"]
    assert badges["waiting"]["status"] == "waiting"
    assert badges["waiting"]["rating"] is None
    for run_id in ("corrupt", "outside"):
        assert badges[run_id]["status"] == "unavailable"
        assert badges[run_id]["error"]
