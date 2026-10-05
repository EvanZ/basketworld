from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.training_backend.mlflow_metrics import metric_catalog, metric_histories


class FakeClient:
    def get_run(self, run_id):
        assert run_id == "run-1"
        return SimpleNamespace(
            data=SimpleNamespace(metrics={"z": 3.0, "a": 1.5, "nan": float("nan")})
        )

    def get_metric_history(self, run_id, name):
        assert run_id == "run-1"
        assert name == "loss"
        return [
            SimpleNamespace(step=step, timestamp=1000 + step, value=float(step))
            for step in range(10)
        ]


def test_metric_catalog_is_sorted_and_includes_latest_values():
    assert metric_catalog(
        tracking_uri="http://mlflow", run_id="run-1", client=FakeClient()
    ) == [
        {"name": "a", "latest": 1.5},
        {"name": "nan", "latest": None},
        {"name": "z", "latest": 3.0},
    ]


def test_metric_histories_are_downsampled_with_endpoints_preserved():
    result = metric_histories(
        tracking_uri="http://mlflow",
        run_id="run-1",
        names=["loss", "loss"],
        max_points=4,
        client=FakeClient(),
    )

    assert len(result["loss"]) == 4
    assert result["loss"][0]["step"] == 0
    assert result["loss"][-1]["step"] == 9


def test_metric_histories_limit_selected_metrics():
    with pytest.raises(ValueError, match="At most 16"):
        metric_histories(
            tracking_uri="http://mlflow",
            run_id="run-1",
            names=[f"metric-{index}" for index in range(17)],
            client=FakeClient(),
        )
