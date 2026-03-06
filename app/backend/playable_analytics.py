from __future__ import annotations

import atexit
import copy
import gzip
import json
import os
import socket
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from typing import Any

from basketworld.utils.mlflow_config import get_mlflow_config


def _is_truthy(raw: str | None) -> bool:
    return str(raw or "").strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    raw = os.getenv(name)
    try:
        value = int(str(raw).strip()) if raw is not None else int(default)
    except Exception:
        value = int(default)
    if value < minimum:
        return int(minimum)
    if value > maximum:
        return int(maximum)
    return int(value)


def _utc_iso(ts: float | None = None) -> str:
    moment = datetime.fromtimestamp(ts if ts is not None else time.time(), tz=timezone.utc)
    return moment.isoformat().replace("+00:00", "Z")


def _json_default(value: Any):
    # Keep this dependency-light and tolerant to numpy scalars if present.
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


class PlayableAnalyticsEmitter:
    """
    Asynchronous S3 emitter for playable analytics events.

    Events are batched into NDJSON payloads and uploaded as .jsonl.gz files.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._queue: list[dict[str, Any]] = []
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._s3_client = None
        self._warned_disabled = False
        self._warned_client_init = False
        self._dropped_events = 0
        self._uploaded_events = 0
        self._uploaded_batches = 0
        self._last_upload_iso: str | None = None
        self._debug_received: deque[dict[str, Any]] = deque(
            maxlen=self._debug_buffer_size()
        )
        self._debug_uploaded: deque[dict[str, Any]] = deque(
            maxlen=self._debug_buffer_size()
        )

    @staticmethod
    def _config() -> dict[str, Any]:
        prefix = str(os.getenv("BW_ANALYTICS_S3_PREFIX") or "basketworld/playable-analytics").strip().strip("/")
        return {
            "enabled": _is_truthy(os.getenv("BW_ANALYTICS_S3_ENABLED")),
            "debug_enabled": _is_truthy(os.getenv("BW_ANALYTICS_DEBUG_ENABLED")),
            "bucket": str(os.getenv("BW_ANALYTICS_S3_BUCKET") or "").strip(),
            "prefix": prefix,
            "flush_events": _env_int("BW_ANALYTICS_S3_FLUSH_EVENTS", 50, minimum=1, maximum=10_000),
            "flush_seconds": _env_int("BW_ANALYTICS_S3_FLUSH_SECONDS", 15, minimum=1, maximum=600),
            "max_queue_events": _env_int("BW_ANALYTICS_S3_MAX_QUEUE_EVENTS", 5_000, minimum=100, maximum=1_000_000),
            "gzip": True,
            "environment": str(os.getenv("BW_ANALYTICS_ENVIRONMENT") or "").strip() or None,
        }

    @staticmethod
    def _debug_buffer_size() -> int:
        return _env_int("BW_ANALYTICS_DEBUG_BUFFER_EVENTS", 200, minimum=20, maximum=10_000)

    def runtime_status(self) -> dict[str, Any]:
        cfg = self._config()
        with self._lock:
            return {
                "enabled": bool(cfg["enabled"]),
                "debug_enabled": bool(cfg["debug_enabled"]),
                "bucket_configured": bool(cfg["bucket"]),
                "prefix": cfg["prefix"],
                "flush_events": int(cfg["flush_events"]),
                "flush_seconds": int(cfg["flush_seconds"]),
                "max_queue_events": int(cfg["max_queue_events"]),
                "queued_events": int(len(self._queue)),
                "dropped_events": int(self._dropped_events),
                "uploaded_events": int(self._uploaded_events),
                "uploaded_batches": int(self._uploaded_batches),
                "last_upload_ts": self._last_upload_iso,
            }

    def debug_snapshot(self, limit: int = 50) -> dict[str, Any]:
        cfg = self._config()
        safe_limit = max(1, min(int(limit), 500))
        with self._lock:
            return {
                "enabled": bool(cfg["debug_enabled"]),
                "limit": int(safe_limit),
                "queue_size": int(len(self._queue)),
                "queued_events": copy.deepcopy(self._queue[-safe_limit:]),
                "recent_received_events": list(self._debug_received)[-safe_limit:],
                "recent_uploaded_events": list(self._debug_uploaded)[-safe_limit:],
            }

    def _ensure_s3_client(self):
        if self._s3_client is not None:
            return self._s3_client
        try:
            import boto3
        except Exception:
            if not self._warned_client_init:
                print("[PlayableAnalytics] boto3 unavailable; analytics uploads disabled.")
                self._warned_client_init = True
            return None

        try:
            mlflow_cfg = get_mlflow_config(load_env=True)
            mlflow_cfg.set_boto3_env()
        except Exception:
            # Best effort only; boto3 can still resolve credentials via IAM role/profile chain.
            pass

        endpoint = os.getenv("MLFLOW_S3_ENDPOINT_URL")
        region = os.getenv("AWS_DEFAULT_REGION") or os.getenv("MLFLOW_AWS_DEFAULT_REGION")
        kwargs: dict[str, Any] = {}
        if endpoint:
            kwargs["endpoint_url"] = endpoint
        if region:
            kwargs["region_name"] = region

        try:
            self._s3_client = boto3.client("s3", **kwargs)
            return self._s3_client
        except Exception as exc:
            if not self._warned_client_init:
                print(f"[PlayableAnalytics] Failed to initialize S3 client: {exc}")
                self._warned_client_init = True
            return None

    def _start_worker_locked(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_worker, name="playable-analytics-uploader", daemon=True)
        self._thread.start()

    def emit(self, event: dict[str, Any]) -> None:
        cfg = self._config()
        if not cfg["enabled"] or not cfg["bucket"]:
            if not self._warned_disabled and cfg["enabled"] and not cfg["bucket"]:
                print("[PlayableAnalytics] BW_ANALYTICS_S3_ENABLED=true but BW_ANALYTICS_S3_BUCKET is unset.")
                self._warned_disabled = True
            return

        now_iso = _utc_iso()
        record = dict(event or {})
        record.setdefault("emitted_at", now_iso)

        with self._lock:
            self._start_worker_locked()
            self._queue.append(record)
            if bool(cfg["debug_enabled"]):
                self._debug_received.append(copy.deepcopy(record))
            max_queue = int(cfg["max_queue_events"])
            overflow = len(self._queue) - max_queue
            if overflow > 0:
                # Drop oldest first to preserve the most recent gameplay context.
                del self._queue[:overflow]
                self._dropped_events += overflow
            self._wake.set()

    def flush(self, force: bool = False) -> None:
        self._flush_once(force=force)

    def shutdown(self) -> None:
        self._stop.set()
        self._wake.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        self._flush_once(force=True)

    def _run_worker(self) -> None:
        try:
            while not self._stop.is_set():
                self._wake.wait(timeout=1.0)
                self._wake.clear()
                self._flush_once(force=False)
        finally:
            self._flush_once(force=True)

    def _flush_once(self, force: bool) -> None:
        cfg = self._config()
        if not cfg["enabled"] or not cfg["bucket"]:
            return

        batch: list[dict[str, Any]] = []
        with self._lock:
            queue_size = len(self._queue)
            if queue_size == 0:
                return
            if not force and queue_size < int(cfg["flush_events"]):
                return
            batch = self._queue[:]
            self._queue.clear()

        client = self._ensure_s3_client()
        if client is None:
            # Re-queue with cap when client init fails.
            with self._lock:
                self._queue = batch + self._queue
                overflow = len(self._queue) - int(cfg["max_queue_events"])
                if overflow > 0:
                    del self._queue[:overflow]
                    self._dropped_events += overflow
            return

        lines = []
        for event in batch:
            try:
                lines.append(json.dumps(event, default=_json_default, separators=(",", ":")))
            except Exception:
                self._dropped_events += 1
        if not lines:
            return

        payload = ("\n".join(lines) + "\n").encode("utf-8")
        timestamp = datetime.now(timezone.utc)
        date_part = timestamp.strftime("%Y-%m-%d")
        hour_part = timestamp.strftime("%H")
        epoch_ms = int(timestamp.timestamp() * 1000)
        host = socket.gethostname().split(".")[0]
        uid = uuid.uuid4().hex[:10]
        prefix = cfg["prefix"]
        key_prefix = f"{prefix}/date={date_part}/hour={hour_part}" if prefix else f"date={date_part}/hour={hour_part}"
        key = f"{key_prefix}/events_{epoch_ms}_{host}_{uid}.jsonl.gz"

        try:
            compressed = gzip.compress(payload)
            client.put_object(
                Bucket=str(cfg["bucket"]),
                Key=key,
                Body=compressed,
                ContentType="application/x-ndjson",
                ContentEncoding="gzip",
            )
        except Exception as exc:
            print(f"[PlayableAnalytics] S3 upload failed: {exc}")
            # Re-queue for retry.
            with self._lock:
                self._queue = batch + self._queue
                overflow = len(self._queue) - int(cfg["max_queue_events"])
                if overflow > 0:
                    del self._queue[:overflow]
                    self._dropped_events += overflow
            return

        with self._lock:
            self._uploaded_events += len(lines)
            self._uploaded_batches += 1
            self._last_upload_iso = _utc_iso()
            if bool(cfg["debug_enabled"]):
                for event in batch:
                    self._debug_uploaded.append(copy.deepcopy(event))


playable_analytics_emitter = PlayableAnalyticsEmitter()
atexit.register(playable_analytics_emitter.shutdown)
