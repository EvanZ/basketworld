from __future__ import annotations

import os
from typing import Any

import httpx
from fastapi import APIRouter, Query


router = APIRouter()


def _training_api_root() -> str:
    return str(
        os.getenv("BW_TRAINING_API_URL", "http://localhost:8090/api/v1")
    ).rstrip("/")


def _recent_run_summary(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(raw.get("id") or ""),
        "name": str(raw.get("name") or "Unnamed run"),
        "status": str(raw.get("status") or "unknown"),
        "mlflow_run_id": str(raw.get("mlflow_run_id") or ""),
        "current_update": int(raw.get("current_update") or 0),
        "target_updates": int(raw.get("target_updates") or 0),
        "created_at": raw.get("created_at"),
        "updated_at": raw.get("updated_at"),
    }


@router.get("/api/training/runs")
def recent_training_runs(
    limit: int = Query(default=8, ge=1, le=50),
) -> dict[str, Any]:
    """Return recent trainer-managed runs without exposing trainer controls."""
    try:
        response = httpx.get(
            f"{_training_api_root()}/runs",
            params={"limit": limit},
            timeout=3.0,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            raise ValueError("Trainer API returned an invalid run list.")
        return {
            "available": True,
            "runs": [
                _recent_run_summary(item)
                for item in payload
                if isinstance(item, dict)
            ],
        }
    except (httpx.HTTPError, ValueError) as exc:
        return {
            "available": False,
            "runs": [],
            "detail": f"Trainer API unavailable: {exc}",
        }
