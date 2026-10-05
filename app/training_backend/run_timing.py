from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


TERMINAL_ACTIVE_EVENTS = frozenset(
    {"paused", "stopped", "completed", "failed", "launch_failed"}
)


def active_training_seconds(
    events: list[dict[str, Any]], *, now: datetime | None = None
) -> float:
    current_time = now or datetime.now(timezone.utc)
    active_start: datetime | None = None
    elapsed = 0.0
    for event in sorted(events, key=lambda item: item["created_at"]):
        event_type = str(event.get("event_type", ""))
        created_at = event["created_at"]
        if event_type == "started":
            if active_start is not None:
                elapsed += max(0.0, (created_at - active_start).total_seconds())
            active_start = created_at
        elif event_type in TERMINAL_ACTIVE_EVENTS and active_start is not None:
            elapsed += max(0.0, (created_at - active_start).total_seconds())
            active_start = None
    if active_start is not None:
        elapsed += max(0.0, (current_time - active_start).total_seconds())
    return float(elapsed)
