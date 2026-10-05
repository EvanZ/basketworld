from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.training_backend.run_timing import active_training_seconds


def _event(event_type: str, created_at: datetime) -> dict:
    return {"event_type": event_type, "created_at": created_at}


def test_active_training_seconds_excludes_paused_time_across_resume():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    events = [
        _event("started", start),
        _event("paused", start + timedelta(seconds=20)),
        _event("resume_requested", start + timedelta(seconds=70)),
        _event("started", start + timedelta(seconds=80)),
    ]

    assert active_training_seconds(
        list(reversed(events)), now=start + timedelta(seconds=110)
    ) == 50.0


def test_active_training_seconds_stops_at_terminal_event():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    events = [
        _event("started", start),
        _event("completed", start + timedelta(seconds=45)),
    ]

    assert active_training_seconds(
        events, now=start + timedelta(days=1)
    ) == 45.0
