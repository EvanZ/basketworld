import json
from pathlib import Path

import numpy as np

from basketworld_jax.train.main import (
    TRAIN_LOOP_SUMMARY_ARTIFACT_PATH,
    _build_train_loop_summary_payload,
    _log_mlflow_train_loop_summary,
)


class FakeMlflow:
    def __init__(self):
        self.logged_payload = None
        self.logged_artifact_path = None
        self.tags = {}

    def log_artifact(self, path, artifact_path=None):
        self.logged_payload = json.loads(Path(path).read_text())
        self.logged_artifact_path = artifact_path

    def set_tag(self, key, value):
        self.tags[key] = value


def test_log_mlflow_train_loop_summary_logs_json_artifact():
    mlflow = FakeMlflow()

    artifact_path = _log_mlflow_train_loop_summary(
        mlflow,
        {
            "status": "train_loop",
            "array_metric": np.asarray([1, 2], dtype=np.int32),
            "float_metric": np.float32(0.5),
            "train_history": [{"update_index": 1}],
            "eval_trajectories": [{"positions": [[0, 0]]}],
        },
    )

    assert artifact_path == TRAIN_LOOP_SUMMARY_ARTIFACT_PATH
    assert mlflow.logged_artifact_path == "results"
    assert mlflow.tags["jax_train_loop_summary_artifact"] == TRAIN_LOOP_SUMMARY_ARTIFACT_PATH
    assert mlflow.logged_payload == {
        "array_metric": [1, 2],
        "eval_trajectory_count": 1,
        "float_metric": 0.5,
        "status": "train_loop",
        "train_history_count": 1,
    }
    assert "train_history" not in mlflow.logged_payload
    assert "eval_trajectories" not in mlflow.logged_payload


def test_train_loop_summary_payload_drops_large_history_fields():
    payload = _build_train_loop_summary_payload(
        {
            "status": "train_loop",
            "final_metrics": {"end_to_end_steps_per_sec": 123.0},
            "train_history": [{"update_index": idx} for idx in range(100)],
            "eval_trajectories": [{"positions": [[idx, 0]]} for idx in range(3)],
        }
    )

    assert payload["status"] == "train_loop"
    assert payload["final_metrics"]["end_to_end_steps_per_sec"] == 123.0
    assert payload["train_history_count"] == 100
    assert payload["eval_trajectory_count"] == 3
    assert "train_history" not in payload
    assert "eval_trajectories" not in payload
