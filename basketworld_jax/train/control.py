from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


CONTROL_ACTIONS = frozenset({"checkpoint", "pause"})


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(target)


def read_control_request(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    target = Path(path)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    action = str(payload.get("action", "")).strip().lower()
    request_id = str(payload.get("request_id", "")).strip()
    if action not in CONTROL_ACTIONS or not request_id:
        return None
    return {**payload, "action": action, "request_id": request_id}


def acknowledge_control_request(
    path: str | Path,
    request: dict[str, Any],
    *,
    update_index: int,
    checkpoint_path: str | None,
) -> Path:
    target = Path(path)
    acknowledgement = target.with_name(f"{target.stem}.ack.json")
    atomic_write_json(
        acknowledgement,
        {
            "request_id": str(request["request_id"]),
            "action": str(request["action"]),
            "acknowledged_at": utc_timestamp(),
            "update_index": int(update_index),
            "checkpoint_path": checkpoint_path,
        },
    )
    try:
        current = read_control_request(target)
        if current and current.get("request_id") == request.get("request_id"):
            target.unlink(missing_ok=True)
    except OSError:
        pass
    return acknowledgement


def write_training_status(
    path: str | Path | None,
    *,
    state: str,
    completed_updates: int,
    target_updates: int,
    mlflow_run_id: str | None,
    checkpoint_path: str | None,
    metrics: dict[str, Any] | None = None,
) -> None:
    if not path:
        return
    summary = {}
    for key in (
        "end_to_end_steps_per_sec",
        "active_end_to_end_steps_per_sec",
        "train_loop_steps_per_sec",
        "train_loop_elapsed_sec",
    ):
        value = (metrics or {}).get(key)
        if isinstance(value, (int, float)):
            summary[key] = float(value)
    atomic_write_json(
        path,
        {
            "state": str(state),
            "completed_updates": int(completed_updates),
            "target_updates": int(target_updates),
            "mlflow_run_id": mlflow_run_id,
            "checkpoint_path": checkpoint_path,
            "metrics": summary,
            "updated_at": utc_timestamp(),
        },
    )
