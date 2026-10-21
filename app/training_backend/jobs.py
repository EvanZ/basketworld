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
from app.training_backend.application_training import resolve_run_training_config
from basketworld_jax.checkpoints import CHECKPOINT_VERSION, load_checkpoint_metadata
from basketworld_jax.provenance import (
    capture_source_provenance,
    compare_source_provenance,
    resume_config_snapshot,
)


ACTIVE_STATUSES = {
    RunStatus.STARTING.value,
    RunStatus.RUNNING.value,
    RunStatus.PAUSING.value,
    RunStatus.STOPPING.value,
}


class ProvenanceConfirmationRequired(RuntimeError):
    """Raised when a restore source mismatch has not been approved explicitly."""

    def __init__(self, comparison: dict[str, Any]):
        super().__init__(str(comparison["message"]))
        self.comparison = comparison


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
                    "provenance_json": json.dumps(
                        {"schema_version": 1, "launches": []}, sort_keys=True
                    ),
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
        launch_provenance = capture_source_provenance(self.settings.repo_root)
        provenance = dict(run.get("provenance", {}) or {})
        launches = list(provenance.get("launches", []) or [])
        launches.append(
            {
                "at": utc_now().isoformat(),
                "source_provenance": launch_provenance,
            }
        )
        provenance.update({"schema_version": 1, "launches": launches})
        run = self.repository.update_run(
            run_id, provenance_json=json.dumps(provenance, sort_keys=True)
        )
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
            payload={"pid": process.pid, "source_provenance": launch_provenance},
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

    def resume(
        self, run_id: str, *, confirm_provenance_mismatch: bool = False
    ) -> dict[str, Any]:
        run = self._require_status(
            run_id, {RunStatus.PAUSED.value, RunStatus.STOPPED.value}
        )
        config = TrainingRunConfig.model_validate(run["config"])
        self._require_available_worker_slot(run_id)
        return self._resume_from_checkpoint(
            run,
            config,
            event_type=(
                "stopped_recovery_requested"
                if run["status"] == RunStatus.STOPPED.value
                else "resume_requested"
            ),
            confirm_provenance_mismatch=confirm_provenance_mismatch,
        )

    def continue_completed(
        self,
        run_id: str,
        *,
        additional_updates: int,
        additional_historical_eval_updates: list[int] | None = None,
        confirm_provenance_mismatch: bool = False,
    ) -> dict[str, Any]:
        run = self._require_status(run_id, {RunStatus.COMPLETED.value})
        self._require_available_worker_slot(run_id)
        if additional_updates < 1:
            raise RuntimeError("Continuation requires at least one additional update.")

        previous_target = max(
            int(run["target_updates"]), int(run["current_update"])
        )
        target_updates = previous_target + int(additional_updates)
        if target_updates > 1_000_000:
            raise RuntimeError(
                "Continuation target exceeds the maximum of 1,000,000 updates."
            )

        existing_config = TrainingRunConfig.model_validate(run["config"])
        existing_historical_updates = list(
            existing_config.historical_eval_updates or []
        )
        added_historical_updates = list(additional_historical_eval_updates or [])
        if any(
            isinstance(update, bool) or not isinstance(update, int)
            for update in added_historical_updates
        ):
            raise RuntimeError("Added Elo checkpoints must be whole-number updates.")
        if any(update < 1 for update in added_historical_updates):
            raise RuntimeError("Added Elo checkpoints must be positive.")
        if len(set(added_historical_updates)) != len(added_historical_updates):
            raise RuntimeError("Each added Elo checkpoint must appear only once.")
        if any(update <= previous_target for update in added_historical_updates):
            raise RuntimeError(
                "Added Elo checkpoints must be after the completed update "
                f"U{previous_target:,}."
            )
        if any(update > target_updates for update in added_historical_updates):
            raise RuntimeError(
                "Added Elo checkpoints must be no later than the continuation "
                f"target U{target_updates:,}."
            )
        duplicated = set(existing_historical_updates).intersection(
            added_historical_updates
        )
        if duplicated:
            duplicate_list = ", ".join(
                f"U{update:,}" for update in sorted(duplicated)
            )
            raise RuntimeError(
                "Added Elo checkpoints are already in the saved schedule: "
                f"{duplicate_list}."
            )

        resulting_historical_updates = sorted(
            set(existing_historical_updates).union(added_historical_updates)
        )
        config_updates: dict[str, Any] = {"num_updates": target_updates}
        if added_historical_updates:
            config_updates["historical_eval_updates"] = resulting_historical_updates
        config = existing_config.model_copy(update=config_updates)
        return self._resume_from_checkpoint(
            run,
            config,
            event_type="continuation_requested",
            event_payload={
                "additional_updates": int(additional_updates),
                "previous_target_updates": previous_target,
                "target_updates": target_updates,
                "previous_historical_eval_updates": existing_historical_updates,
                "added_historical_eval_updates": added_historical_updates,
                "resulting_historical_eval_updates": list(
                    config.historical_eval_updates or []
                ),
            },
            target_updates=target_updates,
            confirm_provenance_mismatch=confirm_provenance_mismatch,
        )

    def _resume_from_checkpoint(
        self,
        run: dict[str, Any],
        config: TrainingRunConfig,
        *,
        event_type: str,
        event_payload: dict[str, Any] | None = None,
        target_updates: int | None = None,
        confirm_provenance_mismatch: bool = False,
    ) -> dict[str, Any]:
        run_id = run["id"]
        checkpoint = run.get("checkpoint_path") or str(
            Path(run["status_file"]).parent / "checkpoints" / "latest"
        )
        if not Path(checkpoint).exists():
            raise RuntimeError(f"Resume checkpoint does not exist: {checkpoint}")
        preview = build_halfcourt_multi_possession_preview(
            config=config,
            run_id=run_id,
            repo_root=self.settings.repo_root,
            data_dir=self.settings.data_dir,
            resume_checkpoint=checkpoint,
            mlflow_resume_run_id=run.get("mlflow_run_id"),
        )
        checkpoint_metadata, provenance_comparison = self._preflight_resume(
            checkpoint=checkpoint,
            preview=preview,
        )
        if (
            provenance_comparison["requires_confirmation"]
            and not confirm_provenance_mismatch
        ):
            raise ProvenanceConfirmationRequired(provenance_comparison)
        updated = self.repository.update_run(
            run_id,
            command_json=json.dumps(preview.argv),
            environment_json=json.dumps(preview.environment, sort_keys=True),
            config_json=config.model_dump_json(),
            status=RunStatus.QUEUED.value,
            finished_at=None,
            target_updates=(
                int(run["target_updates"])
                if target_updates is None
                else target_updates
            ),
        )
        self._event(
            run_id,
            event_type,
            status=RunStatus.QUEUED.value,
            payload={
                **(event_payload or {}),
                "checkpoint_path": checkpoint,
                "checkpoint_update": int(checkpoint_metadata["update_index"]),
                "checkpoint_provenance": checkpoint_metadata.get(
                    "source_provenance"
                ),
                "provenance_comparison": provenance_comparison,
                "provenance_confirmation": {
                    "required": bool(provenance_comparison["requires_confirmation"]),
                    "confirmed": bool(confirm_provenance_mismatch),
                },
            },
        )
        return self._launch(updated)

    def provenance_comparison(self, run: dict[str, Any]) -> dict[str, Any] | None:
        """Return a display-safe checkpoint/current-checkout source comparison."""

        checkpoint = run.get("checkpoint_path") or str(
            Path(run["status_file"]).parent / "checkpoints" / "latest"
        )
        if not Path(checkpoint).exists():
            return {
                "status": "missing",
                "exact": False,
                "requires_confirmation": False,
                "message": "No durable checkpoint is available for restore.",
            }
        try:
            metadata = load_checkpoint_metadata(checkpoint)
        except (OSError, ValueError, KeyError) as exc:
            return {
                "status": "invalid",
                "exact": False,
                "requires_confirmation": False,
                "message": f"Checkpoint cannot be inspected: {exc}",
            }
        return compare_source_provenance(
            metadata.get("source_provenance"),
            capture_source_provenance(self.settings.repo_root),
        )

    @staticmethod
    def _resolved_config_from_preview(preview) -> dict[str, Any]:
        return resume_config_snapshot(resolve_run_training_config(tuple(preview.argv)))

    @staticmethod
    def _validate_checkpoint_metadata(metadata: dict[str, Any]) -> None:
        if int(metadata.get("checkpoint_version", 0)) != CHECKPOINT_VERSION:
            raise RuntimeError(
                "Resume checkpoint schema is unsupported: "
                f"expected version {CHECKPOINT_VERSION}."
            )
        policy_spec = metadata.get("policy_spec")
        if not isinstance(policy_spec, dict):
            raise RuntimeError("Resume checkpoint is missing policy_spec metadata.")
        required_shape_keys = (
            "model_type",
            "flat_obs_dim",
            "training_player_count",
            "action_dim_per_player",
            "total_action_dim",
            "observation_schema_version",
        )
        missing = [key for key in required_shape_keys if key not in policy_spec]
        if missing:
            raise RuntimeError(
                "Resume checkpoint policy/action/observation metadata is incomplete: "
                + ", ".join(missing)
                + "."
            )
        try:
            player_count = int(policy_spec["training_player_count"])
            action_dim = int(policy_spec["action_dim_per_player"])
            total_dim = int(policy_spec["total_action_dim"])
            observation_dim = int(policy_spec["flat_obs_dim"])
            schema = int(policy_spec["observation_schema_version"])
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "Resume checkpoint policy/action/observation metadata is invalid."
            ) from exc
        if min(player_count, action_dim, total_dim, observation_dim, schema) <= 0:
            raise RuntimeError(
                "Resume checkpoint policy/action/observation metadata has non-positive dimensions."
            )
        if total_dim != player_count * action_dim:
            raise RuntimeError(
                "Resume checkpoint action shape is inconsistent: total_action_dim must "
                "equal training_player_count × action_dim_per_player."
            )
        for key in ("trainer_config", "frozen_config"):
            if not isinstance(metadata.get(key), dict):
                raise RuntimeError(f"Resume checkpoint is missing {key} metadata.")

    def _preflight_resume(self, *, checkpoint: str, preview) -> tuple[dict[str, Any], dict[str, Any]]:
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.is_dir():
            raise RuntimeError(
                "Resume checkpoint must be a durable JAX checkpoint directory."
            )
        if not (checkpoint_path / "state").is_dir():
            raise RuntimeError("Resume checkpoint is missing its durable state directory.")
        try:
            metadata = load_checkpoint_metadata(checkpoint_path)
        except FileNotFoundError as exc:
            raise RuntimeError(f"Resume checkpoint metadata is missing: {exc}") from exc
        except (OSError, ValueError, KeyError) as exc:
            raise RuntimeError(f"Resume checkpoint cannot be inspected: {exc}") from exc
        self._validate_checkpoint_metadata(metadata)

        saved_config = metadata.get("resolved_training_config")
        if isinstance(saved_config, dict):
            saved_config = dict(saved_config)
            saved_config.setdefault("require_all_offense_clear", False)
            expected_config = self._resolved_config_from_preview(preview)
            changed = [
                key
                for key in sorted(set(saved_config) | set(expected_config))
                if saved_config.get(key) != expected_config.get(key)
            ]
            if changed:
                raise RuntimeError(
                    "Resume checkpoint resolved training config mismatch for "
                    + ", ".join(repr(key) for key in changed[:8])
                    + ("." if len(changed) <= 8 else ", and more.")
                )

        comparison = compare_source_provenance(
            metadata.get("source_provenance"),
            capture_source_provenance(self.settings.repo_root),
        )
        return metadata, comparison

    def _require_available_worker_slot(self, run_id: str) -> None:
        for active in self.repository.list_active_runs():
            if active["id"] != run_id:
                raise RuntimeError(
                    f"Training run {active['id']} is already active; v0 permits one active worker."
                )

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
        checkpoint_provenance = status.get("checkpoint_provenance")
        if isinstance(checkpoint_provenance, dict):
            provenance = dict(run.get("provenance", {}) or {})
            if provenance.get("latest_checkpoint") != checkpoint_provenance:
                provenance.update(
                    {
                        "schema_version": 1,
                        "latest_checkpoint": checkpoint_provenance,
                    }
                )
                changes["provenance_json"] = json.dumps(provenance, sort_keys=True)
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
