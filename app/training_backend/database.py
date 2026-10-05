from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import duckdb


SCHEMA_VERSION = 1


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class TrainingRepository:
    """Small DuckDB control-plane store with a single in-process writer lock."""

    def __init__(self, database_path: str | Path):
        self.database_path = Path(database_path)
        self._lock = threading.RLock()

    @contextmanager
    def _connection(self) -> Iterator[duckdb.DuckDBPyConnection]:
        connection = duckdb.connect(str(self.database_path))
        try:
            yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self._connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    applied_at TIMESTAMPTZ NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS training_runs (
                    id VARCHAR PRIMARY KEY,
                    idempotency_key VARCHAR NOT NULL UNIQUE,
                    name VARCHAR NOT NULL,
                    preset VARCHAR NOT NULL,
                    status VARCHAR NOT NULL,
                    config_json VARCHAR NOT NULL,
                    command_json VARCHAR NOT NULL,
                    environment_json VARCHAR NOT NULL,
                    cwd VARCHAR NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL,
                    started_at TIMESTAMPTZ,
                    finished_at TIMESTAMPTZ,
                    pid BIGINT,
                    process_start_token VARCHAR,
                    exit_code INTEGER,
                    mlflow_run_id VARCHAR,
                    current_update INTEGER NOT NULL DEFAULT 0,
                    target_updates INTEGER NOT NULL,
                    checkpoint_path VARCHAR,
                    control_file VARCHAR NOT NULL,
                    status_file VARCHAR NOT NULL,
                    log_file VARCHAR NOT NULL,
                    error_message VARCHAR
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS run_events (
                    id VARCHAR PRIMARY KEY,
                    run_id VARCHAR NOT NULL,
                    event_type VARCHAR NOT NULL,
                    status VARCHAR,
                    message VARCHAR,
                    payload_json VARCHAR NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL
                )
                """
            )
            connection.execute(
                """
                INSERT INTO schema_migrations (version, applied_at)
                SELECT ?, ?
                WHERE NOT EXISTS (
                    SELECT 1 FROM schema_migrations WHERE version = ?
                )
                """,
                [SCHEMA_VERSION, utc_now(), SCHEMA_VERSION],
            )

    @staticmethod
    def _row_dict(
        cursor: duckdb.DuckDBPyConnection, row: tuple[Any, ...]
    ) -> dict[str, Any]:
        columns = [description[0] for description in cursor.description]
        return dict(zip(columns, row, strict=True))

    @staticmethod
    def _decode_run(row: dict[str, Any] | None) -> dict[str, Any] | None:
        if row is None:
            return None
        decoded = dict(row)
        for source, target in (
            ("config_json", "config"),
            ("command_json", "command"),
            ("environment_json", "environment"),
        ):
            decoded[target] = json.loads(decoded.pop(source))
        return decoded

    def create_run(self, payload: dict[str, Any]) -> dict[str, Any]:
        columns = (
            "id",
            "idempotency_key",
            "name",
            "preset",
            "status",
            "config_json",
            "command_json",
            "environment_json",
            "cwd",
            "created_at",
            "updated_at",
            "target_updates",
            "control_file",
            "status_file",
            "log_file",
        )
        values = [payload[column] for column in columns]
        with self._lock, self._connection() as connection:
            connection.execute(
                f"INSERT INTO training_runs ({', '.join(columns)}) VALUES ({', '.join('?' for _ in columns)})",
                values,
            )
        return self.get_run(payload["id"])

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._lock, self._connection() as connection:
            cursor = connection.execute(
                "SELECT * FROM training_runs WHERE id = ?", [run_id]
            )
            row = cursor.fetchone()
            return self._decode_run(self._row_dict(cursor, row)) if row else None

    def get_run_by_idempotency_key(self, key: str) -> dict[str, Any] | None:
        with self._lock, self._connection() as connection:
            cursor = connection.execute(
                "SELECT * FROM training_runs WHERE idempotency_key = ?", [key]
            )
            row = cursor.fetchone()
            return self._decode_run(self._row_dict(cursor, row)) if row else None

    def list_runs(self, *, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock, self._connection() as connection:
            cursor = connection.execute(
                "SELECT * FROM training_runs ORDER BY created_at DESC LIMIT ?", [limit]
            )
            return [
                self._decode_run(self._row_dict(cursor, row))
                for row in cursor.fetchall()
            ]

    def list_active_runs(self) -> list[dict[str, Any]]:
        statuses = ["starting", "running", "pausing", "stopping"]
        with self._lock, self._connection() as connection:
            cursor = connection.execute(
                "SELECT * FROM training_runs WHERE status IN (?, ?, ?, ?)", statuses
            )
            return [
                self._decode_run(self._row_dict(cursor, row))
                for row in cursor.fetchall()
            ]

    def update_run(self, run_id: str, **changes: Any) -> dict[str, Any]:
        if not changes:
            current = self.get_run(run_id)
            if current is None:
                raise KeyError(run_id)
            return current
        allowed = {
            "status",
            "updated_at",
            "started_at",
            "finished_at",
            "pid",
            "process_start_token",
            "exit_code",
            "mlflow_run_id",
            "current_update",
            "checkpoint_path",
            "error_message",
            "command_json",
            "environment_json",
        }
        unknown = set(changes) - allowed
        if unknown:
            raise ValueError(f"Unsupported training_runs fields: {sorted(unknown)}")
        changes.setdefault("updated_at", utc_now())
        assignments = ", ".join(f"{column} = ?" for column in changes)
        values = list(changes.values()) + [run_id]
        with self._lock, self._connection() as connection:
            connection.execute(
                f"UPDATE training_runs SET {assignments} WHERE id = ?", values
            )
        updated = self.get_run(run_id)
        if updated is None:
            raise KeyError(run_id)
        return updated

    def append_event(self, payload: dict[str, Any]) -> None:
        with self._lock, self._connection() as connection:
            connection.execute(
                """
                INSERT INTO run_events (
                    id, run_id, event_type, status, message, payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    payload["id"],
                    payload["run_id"],
                    payload["event_type"],
                    payload.get("status"),
                    payload.get("message"),
                    json.dumps(payload.get("payload", {}), sort_keys=True),
                    payload["created_at"],
                ],
            )

    def list_events(self, run_id: str, *, limit: int = 200) -> list[dict[str, Any]]:
        with self._lock, self._connection() as connection:
            cursor = connection.execute(
                """
                SELECT * FROM run_events
                WHERE run_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                [run_id, limit],
            )
            events = []
            for row in cursor.fetchall():
                decoded = self._row_dict(cursor, row)
                decoded["payload"] = json.loads(decoded.pop("payload_json"))
                events.append(decoded)
            return events
