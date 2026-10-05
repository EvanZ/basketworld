from __future__ import annotations

import json
import sys
from types import SimpleNamespace

from basketworld_jax.train.control import (
    acknowledge_control_request,
    atomic_write_json,
    read_control_request,
    write_training_status,
)
from basketworld_jax.train import main as train_main
from basketworld_jax.train.main import parse_args


def test_control_request_round_trip_and_acknowledgement(tmp_path):
    control = tmp_path / "control.json"
    atomic_write_json(
        control,
        {"request_id": "request-1", "action": "pause"},
    )

    request = read_control_request(control)
    assert request == {"request_id": "request-1", "action": "pause"}

    acknowledgement = acknowledge_control_request(
        control,
        request,
        update_index=17,
        checkpoint_path="/tmp/checkpoints/latest",
    )
    assert not control.exists()
    assert json.loads(acknowledgement.read_text())["update_index"] == 17


def test_invalid_control_request_is_ignored(tmp_path):
    control = tmp_path / "control.json"
    atomic_write_json(control, {"request_id": "request-1", "action": "delete"})
    assert read_control_request(control) is None


def test_training_status_contains_only_small_metric_summary(tmp_path):
    status = tmp_path / "status.json"
    write_training_status(
        status,
        state="running",
        completed_updates=8,
        target_updates=100,
        mlflow_run_id="run-123",
        checkpoint_path=None,
        metrics={
            "end_to_end_steps_per_sec": 1234.5,
            "large_internal_payload": [1, 2, 3],
        },
    )
    payload = json.loads(status.read_text())
    assert payload["completed_updates"] == 8
    assert payload["metrics"] == {"end_to_end_steps_per_sec": 1234.5}


def test_trainer_accepts_control_and_status_files(tmp_path):
    args = parse_args(
        [
            "--control-file",
            str(tmp_path / "control.json"),
            "--training-status-file",
            str(tmp_path / "status.json"),
        ]
    )
    assert args.control_file.endswith("control.json")
    assert args.training_status_file.endswith("status.json")


def test_resume_reopens_existing_mlflow_run(monkeypatch):
    calls = []
    fake_mlflow = SimpleNamespace(
        start_run=lambda **kwargs: calls.append(("start_run", kwargs)) or object(),
        set_experiment=lambda name: calls.append(("set_experiment", name)),
    )
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    monkeypatch.setattr(train_main, "setup_mlflow", lambda verbose=False: None)

    returned_mlflow, context = train_main._maybe_start_mlflow_run(
        SimpleNamespace(
            log_mlflow=True,
            mlflow_resume_run_id="existing-run-id",
            mlflow_experiment_name="experiment",
            mlflow_run_name="run name",
        ),
        mode="train",
    )

    assert returned_mlflow is fake_mlflow
    assert context is not None
    assert calls == [("start_run", {"run_id": "existing-run-id"})]
