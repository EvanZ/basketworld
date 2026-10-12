from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.backend.routes import training_routes


class _Response:
    def __init__(self, payload, *, error: Exception | None = None):
        self._payload = payload
        self._error = error

    def raise_for_status(self):
        if self._error is not None:
            raise self._error

    def json(self):
        return self._payload


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(training_routes.router)
    return TestClient(app)


def test_recent_training_runs_returns_safe_summary(monkeypatch):
    observed = {}

    def fake_get(url, *, params, timeout):
        observed.update(url=url, params=params, timeout=timeout)
        return _Response(
            [
                {
                    "id": "local-1",
                    "name": "half-court test",
                    "status": "running",
                    "mlflow_run_id": "abc123",
                    "current_update": 7500,
                    "target_updates": 30000,
                    "created_at": "2026-10-07T01:00:00Z",
                    "updated_at": "2026-10-07T02:00:00Z",
                    "command": ["secret", "internal"],
                }
            ]
        )

    monkeypatch.setattr(training_routes.httpx, "get", fake_get)
    response = _client().get("/api/training/runs?limit=5")

    assert response.status_code == 200
    assert observed["params"] == {"limit": 5}
    assert response.json() == {
        "available": True,
        "runs": [
            {
                "id": "local-1",
                "name": "half-court test",
                "status": "running",
                "mlflow_run_id": "abc123",
                "current_update": 7500,
                "target_updates": 30000,
                "created_at": "2026-10-07T01:00:00Z",
                "updated_at": "2026-10-07T02:00:00Z",
            }
        ],
    }


def test_recent_training_runs_degrades_when_trainer_is_offline(monkeypatch):
    def fake_get(*args, **kwargs):
        raise training_routes.httpx.ConnectError("connection refused")

    monkeypatch.setattr(training_routes.httpx, "get", fake_get)
    response = _client().get("/api/training/runs")

    assert response.status_code == 200
    assert response.json()["available"] is False
    assert response.json()["runs"] == []
