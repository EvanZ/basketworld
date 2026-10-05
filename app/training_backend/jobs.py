from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.training_backend.config import Settings
from app.training_backend.database import TrainingRepository, utc_now
from app.training_backend.models import RunStatus, TrainingRunConfig
from app.training_backend.presets import build_halfcourt_multi_possession_preview


ACTIVE_STATUSES = {
    RunStatus.STARTING.value,
    RunStatus.RUNNING.value,
    RunStatus.PAUSING.value,
    RunStatus.STOPPING.value,
}


def _process_start_token(pid: int) -> str | None:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        fields = raw[raw.rfind(")") + 2 :].split()
        return fields[19]
    except (OSError, IndexError):
        return None


def _process_matches(pid: int | None, token: str | None) -> bool:
    if not pid or not token:
        return False
    return _process_start_token(pid) == token


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)


def _read_json(path: str | Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return payload if isinstance(payload, dict) else None


class JobManager:
    def __init__(self, settings: Settings, repository: TrainingRepository):
        self.settings = settings
        self.repository = repository
        self._processes: dict[str, subprocess.Popen[bytes]] = {}
        self._lock = threading.RLock()

    def _event(
        self,
        run_id: str,
        event_type: str,
        *,
        status: str | None = None,
        message: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        self.repository.append_event(
            {
                "id": str(uuid4()),
                "run_id": run_id,
                "event_type": event_type,
                "status": status,
                "message": message,
                "payload": payload or {},
                "created_at": utc_now(),
            }
        )

    def preview(self, config: TrainingRunConfig, *, run_id: str | None = None):
        return build_halfcourt_multi_possession_preview(
            config=config,
            run_id=run_id or str(uuid4()),
            repo_root=self.settings.repo_root,
            data_dir=self.settings.data_dir,
        )

    def create_and_launch(
        self,
        *,
        config: TrainingRunConfig,
        idempotency_key: str,
    ) -> dict[str, Any]:
        with self._lock:
            existing = self.repository.get_run_by_idempotency_key(idempotency_key)
            if existing is not None:
                return existing
            active = self.repository.list_active_runs()
            if active:
                raise RuntimeError(
                    f"Training run {active[0]['id']} is already active; v0 permits one active worker."
                )
            run_id = str(uuid4())
            preview = self.preview(config, run_id=run_id)
            now = utc_now()
            created = self.repository.create_run(
                {
                    "id": run_id,
                    "idempotency_key": idempotency_key,
                    "name": config.name,
                    "preset": config.preset,
                    "status": RunStatus.QUEUED.value,
                    "config_json": config.model_dump_json(),
                    "command_json": json.dumps(preview.argv),
                    "environment_json": json.dumps(preview.environment, sort_keys=True),
                    "cwd": str(self.settings.repo_root),
                    "created_at": now,
                    "updated_at": now,
                    "target_updates": config.num_updates,
                    "control_file": preview.control_file,
                    "status_file": preview.status_file,
                    "log_file": preview.log_file,
                }
            )
            self._event(
                run_id,
                "created",
                status=RunStatus.QUEUED.value,
                payload={"display_command": preview.display_command},
            )
            try:
                return self._launch(created)
            except Exception as exc:
                failed = self.repository.update_run(
                    run_id,
                    status=RunStatus.FAILED.value,
                    finished_at=utc_now(),
                    error_message=str(exc),
                )
                self._event(
                    run_id,
                    "launch_failed",
                    status=RunStatus.FAILED.value,
                    message=str(exc),
                )
                return failed

    def _launch(self, run: dict[str, Any]) -> dict[str, Any]:
        run_id = run["id"]
        log_path = Path(run["log_file"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        Path(run["control_file"]).unlink(missing_ok=True)
        environment = os.environ.copy()
        environment.update(run["environment"])
        self.repository.update_run(run_id, status=RunStatus.STARTING.value)
        with log_path.open("ab", buffering=0) as log_handle:
            process = subprocess.Popen(
                run["command"],
                cwd=run["cwd"],
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        token = _process_start_token(process.pid)
        if token is None:
            process.terminate()
            raise RuntimeError(
                "Training worker started but its process identity could not be verified."
            )
        with self._lock:
            self._processes[run_id] = process
        updated = self.repository.update_run(
            run_id,
            status=RunStatus.RUNNING.value,
            pid=process.pid,
            process_start_token=token,
            started_at=utc_now(),
            finished_at=None,
            exit_code=None,
            error_message=None,
        )
        self._event(
            run_id,
            "started",
            status=RunStatus.RUNNING.value,
            payload={"pid": process.pid},
        )
        return updated

    def request_checkpoint(self, run_id: str) -> dict[str, Any]:
        run = self._require_status(run_id, {RunStatus.RUNNING.value})
        request_id = str(uuid4())
        _atomic_json(
            Path(run["control_file"]),
            {
                "request_id": request_id,
                "action": "checkpoint",
                "requested_at": utc_now().isoformat(),
            },
        )
        self._event(run_id, "checkpoint_requested", payload={"request_id": request_id})
        return self.repository.get_run(run_id)

    def request_pause(self, run_id: str) -> dict[str, Any]:
        run = self._require_status(run_id, {RunStatus.RUNNING.value})
        request_id = str(uuid4())
        _atomic_json(
            Path(run["control_file"]),
            {
                "request_id": request_id,
                "action": "pause",
                "requested_at": utc_now().isoformat(),
            },
        )
        updated = self.repository.update_run(run_id, status=RunStatus.PAUSING.value)
        self._event(
            run_id,
            "pause_requested",
            status=RunStatus.PAUSING.value,
            payload={"request_id": request_id},
        )
        return updated

    def request_stop(self, run_id: str, *, force: bool = False) -> dict[str, Any]:
        run = self._require_status(run_id, ACTIVE_STATUSES)
        if not _process_matches(run.get("pid"), run.get("process_start_token")):
            raise RuntimeError(
                "Training worker identity no longer matches the stored process."
            )
        os.killpg(int(run["pid"]), signal.SIGKILL if force else signal.SIGTERM)
        updated = self.repository.update_run(run_id, status=RunStatus.STOPPING.value)
        self._event(
            run_id,
            "force_stop_requested" if force else "stop_requested",
            status=RunStatus.STOPPING.value,
        )
        return updated

    def resume(self, run_id: str) -> dict[str, Any]:
        run = self._require_status(run_id, {RunStatus.PAUSED.value})
        config = TrainingRunConfig.model_validate(run["config"])
        checkpoint = run.get("checkpoint_path") or str(
            Path(run["status_file"]).parent / "checkpoints" / "latest"
        )
        if not Path(checkpoint).exists():
            raise RuntimeError(f"Pause checkpoint does not exist: {checkpoint}")
        preview = build_halfcourt_multi_possession_preview(
            config=config,
            run_id=run_id,
            repo_root=self.settings.repo_root,
            data_dir=self.settings.data_dir,
            resume_checkpoint=checkpoint,
            mlflow_resume_run_id=run.get("mlflow_run_id"),
        )
        updated = self.repository.update_run(
            run_id,
            command_json=json.dumps(preview.argv),
            environment_json=json.dumps(preview.environment, sort_keys=True),
            status=RunStatus.QUEUED.value,
            finished_at=None,
        )
        self._event(run_id, "resume_requested", status=RunStatus.QUEUED.value)
        return self._launch(updated)

    def poll(self) -> None:
        for run in self.repository.list_active_runs():
            self._sync_worker_status(run)
            run = self.repository.get_run(run["id"]) or run
            process = self._processes.get(run["id"])
            exit_code = process.poll() if process is not None else None
            alive = _process_matches(run.get("pid"), run.get("process_start_token"))
            if process is not None and exit_code is None:
                continue
            if process is None and alive:
                continue
            if process is not None:
                with self._lock:
                    self._processes.pop(run["id"], None)
            worker_status = _read_json(run["status_file"]) or {}
            worker_state = str(worker_status.get("state", ""))
            if run["status"] == RunStatus.STOPPING.value:
                final_status = RunStatus.STOPPED.value
            elif worker_state == "paused" or run["status"] == RunStatus.PAUSING.value:
                final_status = RunStatus.PAUSED.value
            elif exit_code == 0 or worker_state == "completed":
                final_status = RunStatus.COMPLETED.value
            else:
                final_status = RunStatus.FAILED.value
            error = None
            if final_status == RunStatus.FAILED.value:
                error = f"Training worker exited with code {exit_code}."
                if process is None:
                    error = "Training worker disappeared while the backend was not its parent."
            self.repository.update_run(
                run["id"],
                status=final_status,
                exit_code=exit_code,
                finished_at=utc_now(),
                error_message=error,
            )
            self._event(
                run["id"],
                final_status,
                status=final_status,
                message=error,
            )

    def _sync_worker_status(self, run: dict[str, Any]) -> None:
        status = _read_json(run["status_file"])
        if not status:
            return
        changes: dict[str, Any] = {}
        if status.get("completed_updates") is not None:
            changes["current_update"] = int(status["completed_updates"])
        if status.get("checkpoint_path"):
            changes["checkpoint_path"] = str(status["checkpoint_path"])
        if status.get("mlflow_run_id"):
            changes["mlflow_run_id"] = str(status["mlflow_run_id"])
        if changes:
            self.repository.update_run(run["id"], **changes)

    def _require_status(self, run_id: str, statuses: set[str]) -> dict[str, Any]:
        run = self.repository.get_run(run_id)
        if run is None:
            raise KeyError(run_id)
        if run["status"] not in statuses:
            raise RuntimeError(
                f"Run {run_id} is {run['status']}; expected one of {sorted(statuses)}."
            )
        return run

    def worker_status(self, run: dict[str, Any]) -> dict[str, Any] | None:
        return _read_json(run["status_file"])
