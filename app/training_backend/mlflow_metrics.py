from __future__ import annotations

import math
from typing import Any


MAX_SELECTED_METRICS = 16
MAX_HISTORY_POINTS = 240


def _client(tracking_uri: str):
    import mlflow

    return mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)


def _downsample(points: list[dict[str, Any]], max_points: int) -> list[dict[str, Any]]:
    if len(points) <= max_points:
        return points
    if max_points < 2:
        return [points[-1]]
    last_index = len(points) - 1
    indices = [round(index * last_index / (max_points - 1)) for index in range(max_points)]
    return [points[index] for index in indices]


def metric_catalog(
    *, tracking_uri: str, run_id: str, client=None
) -> list[dict[str, Any]]:
    mlflow_client = client or _client(tracking_uri)
    run = mlflow_client.get_run(run_id)
    return [
        {
            "name": name,
            "latest": float(value) if math.isfinite(float(value)) else None,
        }
        for name, value in sorted(run.data.metrics.items())
    ]


def metric_histories(
    *,
    tracking_uri: str,
    run_id: str,
    names: list[str],
    max_points: int = MAX_HISTORY_POINTS,
    client=None,
) -> dict[str, list[dict[str, Any]]]:
    selected = list(dict.fromkeys(name.strip() for name in names if name.strip()))
    if len(selected) > MAX_SELECTED_METRICS:
        raise ValueError(
            f"At most {MAX_SELECTED_METRICS} metrics can be displayed at once."
        )
    mlflow_client = client or _client(tracking_uri)
    histories: dict[str, list[dict[str, Any]]] = {}
    for name in selected:
        history = sorted(
            mlflow_client.get_metric_history(run_id, name),
            key=lambda metric: (int(metric.step), int(metric.timestamp)),
        )
        points = []
        for metric in history:
            value = float(metric.value)
            if not math.isfinite(value):
                continue
            points.append(
                {
                    "step": int(metric.step),
                    "timestamp": int(metric.timestamp),
                    "value": value,
                }
            )
        histories[name] = _downsample(points, max_points)
    return histories
