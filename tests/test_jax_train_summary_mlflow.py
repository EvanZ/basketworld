import json
from pathlib import Path

import numpy as np

from basketworld_jax.train.main import (
    TRAIN_LOOP_SUMMARY_ARTIFACT_PATH,
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
        },
    )

    assert artifact_path == TRAIN_LOOP_SUMMARY_ARTIFACT_PATH
    assert mlflow.logged_artifact_path == "results"
    assert mlflow.tags["jax_train_loop_summary_artifact"] == TRAIN_LOOP_SUMMARY_ARTIFACT_PATH
    assert mlflow.logged_payload == {
        "array_metric": [1, 2],
        "float_metric": 0.5,
        "status": "train_loop",
    }
