"""Background jobs and result views for the dev-app Skill Ratings workbench.

The underlying evaluator deliberately keeps its ``fixed_pool`` artifact names
because those files are durable study evidence.  This route is the boundary
where the dev app presents that workflow as *Skill Ratings* instead.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, model_validator


router = APIRouter()

REPO_ROOT = Path(__file__).resolve().parents[3]
_VENV_PYTHON = REPO_ROOT / ".env" / "bin" / "python"
_FIXED_POOL_REPORT_NAMESPACE = "fixed_pool_reports"
_MAX_LOG_LINES = 160
_PHASE_PATTERN = re.compile(r"^\[fixed-pool\]\s+(?P<phase>[\w_]+)(?P<tail>.*)$")
_PROGRESS_PATTERN = re.compile(
    r"completed=(?P<completed>\d+)/(?P<total>\d+)\s+batch=(?P<batch>\d+)"
)
_ELAPSED_PATTERN = re.compile(r"elapsed=(?P<elapsed>[0-9.]+)s")


class SkillRatingsJobRequest(BaseModel):
    """Public configuration for one isolated Skill Ratings study."""

    checkpoint_paths: list[str] = Field(default_factory=list)
    checkpoint_root: str | None = None
    checkpoint_updates: list[int] = Field(default_factory=list)
    pool_path: str
    create_pool: bool = True
    pool_size: int = Field(default=0, ge=0)
    pool_seed: int = 0
    schedule_path: str
    create_schedule: bool = True
    game_count: int = Field(default=0, ge=0)
    schedule_seed: int = 1
    artifact_dir: str
    report_dir: str | None = None
    action_mode: Literal["direct", "sampled"] = "direct"
    batch_size: int = Field(default=64, ge=1)
    eval_seed: int = 0
    # Omitted means deterministic grouped 5-fold selection.  Retained as
    # an API-only override for reproducing older analyses.
    ridge_alpha: float | None = Field(default=None, ge=0)
    validation_fraction: float = Field(default=0.2, gt=0, lt=1)
    split_seed: int = 0
    bootstrap_samples: int = Field(default=200, ge=1)
    bootstrap_seed: int = 1
    extension_source: str | None = None

    @model_validator(mode="after")
    def has_checkpoint_source(self) -> "SkillRatingsJobRequest":
        if not self.checkpoint_paths and not str(self.checkpoint_root or "").strip():
            raise ValueError("Select a checkpoint or checkpoint directory.")
        return self


class SkillRatingsExtensionRequest(BaseModel):
    """Request a continuation that evaluates only additional games."""

    report_path: str
    additional_games: int = Field(ge=1)
    schedule_seed: int | None = None
    batch_size: int | None = Field(default=None, ge=1)


class _SkillRatingsJobManager:
    """Own the one resource-intensive local study process at a time."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, dict[str, Any]] = {}
        self._active_job_id: str | None = None

    def start(self, config: SkillRatingsJobRequest) -> dict[str, Any]:
        resolved = _resolve_request_paths(config)
        _validate_request_inputs(config, resolved)
        # Metadata validation is intentionally done before dispatch so the UI
        # rejects a single-possession or non-5v5 checkpoint immediately.
        _validate_checkpoint_compatibility(resolved["checkpoint_paths"])

        with self._lock:
            if self._active_job_id is not None:
                active = self._jobs.get(self._active_job_id, {})
                if active.get("status") in {"queued", "running"}:
                    raise HTTPException(
                        status_code=409,
                        detail="A Skill Ratings study is already running in this dev app.",
                    )
            job_id = uuid.uuid4().hex
            now = time.time()
            job = {
                "id": job_id,
                "status": "queued",
                "created_at": now,
                "updated_at": now,
                "started_at": None,
                "finished_at": None,
                "config": _public_config(config, resolved),
                "events": [],
                "logs": [],
                "report_paths": [],
                "error": None,
                "return_code": None,
            }
            self._jobs[job_id] = job
            self._active_job_id = job_id

        worker = threading.Thread(
            target=self._run,
            args=(job_id, config, resolved),
            daemon=True,
            name=f"skill-ratings-{job_id[:8]}",
        )
        worker.start()
        return self.snapshot(job_id)

    def snapshot(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise HTTPException(
                    status_code=404, detail="Skill Ratings job not found."
                )
            return _snapshot_job(job)

    def active_snapshot(self) -> dict[str, Any] | None:
        """Return the running job so a remounted tab can reattach to it."""
        with self._lock:
            if self._active_job_id is None:
                return None
            job = self._jobs.get(self._active_job_id)
            if job is None or job.get("status") not in {"queued", "running"}:
                return None
            return _snapshot_job(job)

    def _append_log(self, job_id: str, line: str) -> None:
        text = line.rstrip()
        if not text:
            return
        now = time.time()
        with self._lock:
            job = self._jobs[job_id]
            logs = job["logs"]
            logs.append({"at": now, "message": text})
            del logs[:-_MAX_LOG_LINES]
            match = _PHASE_PATTERN.match(text)
            if match:
                event: dict[str, Any] = {"phase": match.group("phase"), "at": now}
                tail = match.group("tail")
                progress = _PROGRESS_PATTERN.search(tail)
                if progress:
                    event.update(
                        {key: int(value) for key, value in progress.groupdict().items()}
                    )
                elapsed = _ELAPSED_PATTERN.search(tail)
                if elapsed:
                    event["elapsed_sec"] = float(elapsed.group("elapsed"))
                job["events"].append(event)
            job["updated_at"] = now

    def _run(
        self,
        job_id: str,
        config: SkillRatingsJobRequest,
        resolved: dict[str, Any],
    ) -> None:
        started_at = time.time()
        with self._lock:
            job = self._jobs[job_id]
            job["status"] = "running"
            job["started_at"] = started_at
            job["updated_at"] = started_at
        command = _build_command(config, resolved)
        self._append_log(job_id, "Starting isolated Skill Ratings evaluator.")
        try:
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                self._append_log(job_id, line)
            return_code = process.wait()
        except Exception as exc:  # pragma: no cover - protects the job surface
            return_code = None
            self._append_log(job_id, f"Unable to start Skill Ratings evaluator: {exc}")
            error = str(exc)
        else:
            if return_code == 0:
                error = None
            else:
                with self._lock:
                    recent_output = [
                        str(entry.get("message") or "").strip()
                        for entry in self._jobs[job_id]["logs"]
                        if str(entry.get("message") or "").strip()
                    ]
                detail = (
                    recent_output[-1]
                    if recent_output
                    else "No evaluator output was captured."
                )
                error = (
                    f"The Skill Ratings evaluator exited with code {return_code}. "
                    f"Last output: {detail}"
                )

        finished_at = time.time()
        report_paths = (
            _discover_reports(resolved["report_dir"], started_at)
            if return_code == 0
            else []
        )
        if return_code == 0 and not report_paths:
            error = "The evaluator finished without writing a Skill Ratings report."
        with self._lock:
            job = self._jobs[job_id]
            job["return_code"] = return_code
            job["report_paths"] = report_paths
            job["error"] = error
            job["status"] = "completed" if error is None else "failed"
            job["finished_at"] = finished_at
            job["updated_at"] = finished_at
            if self._active_job_id == job_id:
                self._active_job_id = None


_JOB_MANAGER = _SkillRatingsJobManager()


def _require_repo_path(raw_path: str, *, label: str) -> Path:
    text = str(raw_path or "").strip()
    if not text:
        raise HTTPException(status_code=422, detail=f"{label} is required.")
    candidate = Path(text).expanduser()
    resolved = (
        (REPO_ROOT / candidate).resolve()
        if not candidate.is_absolute()
        else candidate.resolve()
    )
    try:
        resolved.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"{label} must be inside the BasketWorld repository.",
        ) from exc
    return resolved


def _resolve_request_paths(config: SkillRatingsJobRequest) -> dict[str, Any]:
    checkpoints = [
        _require_repo_path(path, label="Checkpoint path")
        for path in config.checkpoint_paths
        if str(path).strip()
    ]
    root = (
        _require_repo_path(config.checkpoint_root, label="Checkpoint directory")
        if str(config.checkpoint_root or "").strip()
        else None
    )
    if root is not None:
        if config.checkpoint_updates:
            checkpoints.extend(
                root / f"update_{int(index):07d}" for index in config.checkpoint_updates
            )
        else:
            numbered = sorted(root.glob("update_*"))
            if not numbered:
                raise HTTPException(
                    status_code=422,
                    detail="Checkpoint directory has no update_* entries.",
                )
            checkpoints.append(numbered[-1])
    deduplicated = list(dict.fromkeys(path.resolve() for path in checkpoints))
    report_dir = _require_repo_path(
        config.report_dir or config.artifact_dir, label="Report destination"
    )
    extension_source = str(config.extension_source or "").strip()
    return {
        "checkpoint_paths": deduplicated,
        "checkpoint_root": root,
        "pool_path": _require_repo_path(config.pool_path, label="Player pool path"),
        "schedule_path": _require_repo_path(
            config.schedule_path, label="Schedule path"
        ),
        "artifact_dir": _require_repo_path(
            config.artifact_dir, label="Artifact destination"
        ),
        "report_dir": report_dir,
        "extension_source": (
            _require_repo_path(extension_source, label="Extension source")
            if extension_source
            else None
        ),
    }


def _validate_request_inputs(
    config: SkillRatingsJobRequest, resolved: Mapping[str, Any]
) -> None:
    missing = [str(path) for path in resolved["checkpoint_paths"] if not path.is_dir()]
    if missing:
        raise HTTPException(
            status_code=422,
            detail="Checkpoint paths do not exist: " + ", ".join(missing),
        )
    pool_path = resolved["pool_path"]
    schedule_path = resolved["schedule_path"]
    if config.create_pool and pool_path.exists():
        raise HTTPException(
            status_code=409,
            detail="A player pool already exists at that immutable path.",
        )
    if not config.create_pool and not pool_path.is_file():
        raise HTTPException(
            status_code=422,
            detail="Select an existing player pool or create a new one.",
        )
    if config.create_schedule and schedule_path.exists():
        raise HTTPException(
            status_code=409, detail="A schedule already exists at that immutable path."
        )
    if not config.create_schedule and not schedule_path.is_file():
        raise HTTPException(
            status_code=422, detail="Select an existing schedule or create a new one."
        )
    if config.create_pool and not config.create_schedule:
        raise HTTPException(
            status_code=422,
            detail="A new player pool requires a new schedule created from that pool.",
        )
    if config.create_pool and 0 < int(config.pool_size) < 10:
        raise HTTPException(
            status_code=422,
            detail="A 5v5 Skill Ratings player pool must contain at least ten players.",
        )

    # Reused artifacts are immutable inputs, so validate their content and
    # their relationship before allocating a background worker.  In
    # particular, a schedule is content-bound to one exact pool ID.
    if not config.create_pool:
        try:
            from basketworld_jax.eval.fixed_pool import (
                PlayerPoolError,
                load_player_pool,
            )
            from basketworld_jax.eval.lineup_schedule import (
                LineupScheduleError,
                load_lineup_schedule,
            )

            pool = load_player_pool(pool_path)
            if len(pool.players) < 10:
                raise PlayerPoolError(
                    "A 5v5 Skill Ratings player pool must contain at least ten players."
                )
            if not config.create_schedule:
                load_lineup_schedule(schedule_path, pool=pool)
        except (OSError, PlayerPoolError, LineupScheduleError) as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Skill Ratings input artifacts are incompatible: {exc}",
            ) from exc
    extension_source = resolved.get("extension_source")
    if extension_source is not None:
        if not extension_source.is_dir():
            raise HTTPException(
                status_code=422, detail="Extension source study does not exist."
            )
        if not config.create_schedule:
            raise HTTPException(
                status_code=422, detail="An extension must create a new schedule."
            )
        if int(config.game_count) < 1:
            raise HTTPException(
                status_code=422, detail="Choose at least one additional game."
            )


def _validate_checkpoint_compatibility(checkpoint_paths: list[Path]) -> None:
    try:
        from basketworld_jax.eval.fixed_pool_study import (  # local-only metadata read
            FixedPoolStudyRunError,
            _checkpoint_env_params,
        )

        for path in checkpoint_paths:
            _checkpoint_env_params(path)
    except FixedPoolStudyRunError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Could not validate checkpoint compatibility: {exc}",
        ) from exc


def _build_command(
    config: SkillRatingsJobRequest, resolved: Mapping[str, Any]
) -> list[str]:
    executable = _VENV_PYTHON if _VENV_PYTHON.is_file() else Path(sys.executable)
    command = [
        str(executable),
        "-m",
        "basketworld_jax.train.main",
        "--run-fixed-pool-study",
    ]
    for path in resolved["checkpoint_paths"]:
        command.extend(["--fixed-pool-checkpoint", str(path)])
    command.extend(
        [
            "--fixed-pool-path",
            str(resolved["pool_path"]),
            "--fixed-pool-size",
            str(config.pool_size),
            "--fixed-pool-seed",
            str(config.pool_seed),
            "--fixed-pool-schedule-path",
            str(resolved["schedule_path"]),
            "--fixed-pool-games",
            str(config.game_count),
            "--fixed-pool-schedule-seed",
            str(config.schedule_seed),
            "--fixed-pool-action-mode",
            config.action_mode,
            "--fixed-pool-batch-size",
            str(config.batch_size),
            "--fixed-pool-artifact-dir",
            str(resolved["artifact_dir"]),
            "--fixed-pool-report-dir",
            str(resolved["report_dir"]),
            "--fixed-pool-eval-seed",
            str(config.eval_seed),
            "--fixed-pool-validation-fraction",
            str(config.validation_fraction),
            "--fixed-pool-split-seed",
            str(config.split_seed),
            "--fixed-pool-bootstrap-samples",
            str(config.bootstrap_samples),
            "--fixed-pool-bootstrap-seed",
            str(config.bootstrap_seed),
        ]
    )
    if config.ridge_alpha is not None:
        command.extend(["--fixed-pool-ridge-alpha", str(config.ridge_alpha)])
    if config.create_pool:
        command.append("--fixed-pool-create")
    if config.create_schedule:
        command.append("--fixed-pool-schedule-create")
    if resolved.get("extension_source") is not None:
        command.extend(
            ["--fixed-pool-extension-source", str(resolved["extension_source"])]
        )
    return command


def _public_config(
    config: SkillRatingsJobRequest, resolved: Mapping[str, Any]
) -> dict[str, Any]:
    public = config.model_dump()
    public.update(
        {
            "checkpoint_paths": [str(path) for path in resolved["checkpoint_paths"]],
            "pool_path": str(resolved["pool_path"]),
            "schedule_path": str(resolved["schedule_path"]),
            "artifact_dir": str(resolved["artifact_dir"]),
            "report_dir": str(resolved["report_dir"]),
            "extension_source": (
                str(resolved["extension_source"])
                if resolved.get("extension_source") is not None
                else None
            ),
        }
    )
    return public


def _snapshot_job(job: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": job["id"],
        "status": job["status"],
        "created_at": job["created_at"],
        "updated_at": job["updated_at"],
        "started_at": job["started_at"],
        "finished_at": job["finished_at"],
        "config": job["config"],
        "events": list(job["events"]),
        "logs": list(job["logs"]),
        "report_paths": list(job["report_paths"]),
        "error": job["error"],
        "return_code": job["return_code"],
    }


def _discover_reports(report_dir: Path, started_at: float = 0.0) -> list[str]:
    root = report_dir / _FIXED_POOL_REPORT_NAMESPACE
    if not root.is_dir():
        return []
    paths = list(root.glob("*/checkpoint_report.json")) + list(
        root.glob("*/analysis_revisions/*.json")
    )
    return [
        str(path)
        for path in sorted(paths, key=lambda item: item.stat().st_mtime, reverse=True)
        if path.stat().st_mtime >= started_at - 1.0
    ]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=422, detail=f"Could not read report: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=422, detail="The selected report is not a JSON object."
        )
    return payload


def _report_summary(path: Path) -> dict[str, Any]:
    report = _read_json(path)
    raw = report.get("raw_artifact_dependencies") or {}
    checkpoint = report.get("checkpoint") or {}
    is_reanalysis = "reanalysis" in report
    analysis_revision = (
        (report.get("reanalysis") or {}).get("schema_version")
        if is_reanalysis
        else report.get("schema_version")
    )
    return {
        "path": str(path),
        "kind": report.get("kind"),
        "study_id": raw.get("study_id"),
        "checkpoint": checkpoint.get("path"),
        "checkpoint_update": checkpoint.get("update_index"),
        "action_mode": report.get("action_mode"),
        "is_reanalysis": is_reanalysis,
        "analysis_revision": analysis_revision,
        "analysis_label": (
            f"Reanalysis v{analysis_revision}"
            if is_reanalysis
            else (
                f"Original report schema v{analysis_revision}"
                if analysis_revision is not None
                else "Original report"
            )
        ),
        "modified_at": path.stat().st_mtime,
    }


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _report_reproducibility(
    path: Path, report: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Recover a launchable configuration from a report's immutable inputs."""
    raw = report.get("raw_artifact_dependencies") or {}
    if not all(
        str(raw.get(key) or "").strip()
        for key in ("root", "pool_manifest", "schedule_manifest")
    ):
        return None
    raw_root = _require_repo_path(str(raw.get("root") or ""), label="Study artifact")
    pool_path = _require_repo_path(
        str(raw.get("pool_manifest") or ""), label="Report pool manifest"
    )
    schedule_path = _require_repo_path(
        str(raw.get("schedule_manifest") or ""), label="Report schedule manifest"
    )
    pool = _read_json(pool_path)
    schedule = _read_json(schedule_path)
    execution = report.get("execution") or {}
    recorded = (
        execution.get("reproducibility") if isinstance(execution, Mapping) else None
    )
    config = dict(recorded) if isinstance(recorded, Mapping) else {}
    checkpoint = report.get("checkpoint") or {}
    config.update(
        {
            "checkpoint_paths": [str(checkpoint.get("path") or "")],
            "pool_path": str(pool_path),
            "create_pool": False,
            "pool_size": len(pool.get("players") or []),
            "pool_seed": pool.get("pool_seed"),
            "schedule_path": str(schedule_path),
            "create_schedule": False,
            "game_count": len(schedule.get("games") or []),
            "schedule_seed": schedule.get("schedule_seed"),
            "artifact_dir": str(raw_root.parent.parent),
            "action_mode": report.get("action_mode"),
            "batch_size": config.get(
                "batch_size", execution.get("requested_batch_size", 64)
            ),
            "eval_seed": config.get("eval_seed"),
            "ridge_alpha": config.get("ridge_alpha"),
            "validation_fraction": config.get("validation_fraction", 0.2),
            "split_seed": config.get("split_seed"),
            "bootstrap_samples": config.get("bootstrap_samples", 200),
            "bootstrap_seed": config.get("bootstrap_seed"),
        }
    )
    missing_seed_keys = [
        key
        for key in ("eval_seed", "split_seed", "bootstrap_seed")
        if config.get(key) is None
    ]
    return {
        "config": config,
        "seeds": {
            "pool_seed": pool.get("pool_seed"),
            "schedule_seed": schedule.get("schedule_seed"),
            "evaluation_seed": config.get("eval_seed"),
            "split_seed": config.get("split_seed"),
            "bootstrap_seed": config.get("bootstrap_seed"),
        },
        # Early reports predate the persisted launch configuration.  Do not
        # claim their CLI defaults were necessarily used, and do not let an
        # extension silently replace those unknown inputs with defaults.
        "missing_seed_keys": missing_seed_keys,
        "raw_study_root": str(raw_root),
        "source_report_path": str(path),
    }


def _ui_report(path: Path) -> dict[str, Any]:
    report = _read_json(path)
    raw = report.get("raw_artifact_dependencies") or {}
    pool_path = raw.get("pool_manifest")
    pool = _read_json(
        _require_repo_path(str(pool_path or ""), label="Report pool manifest")
    )
    descriptors = [
        item for item in pool.get("skill_descriptors", []) if isinstance(item, dict)
    ]
    descriptor_by_key = {str(item.get("key")): item for item in descriptors}
    direct_skill_ratings = report.get("direct_skill_ratings") or {}
    apm_ratings = (report.get("player_rapm") or {}).get("players") or {}
    # New reports use the direct model as the primary profile rating.  The APM
    # output remains available alongside it as an empirical diagnostic.
    ratings = direct_skill_ratings.get("players") or apm_ratings
    apm_agreement = direct_skill_ratings.get("apm_agreement")
    if not isinstance(apm_agreement, Mapping) and direct_skill_ratings.get("players"):
        # Historical reports predate the persisted comparison block.  Derive it
        # from their immutable player values so they remain equally useful in
        # the workbench without rewriting the source artifact.
        from basketworld_jax.eval.fixed_pool_models import skill_apm_agreement

        apm_agreement = skill_apm_agreement(direct_skill_ratings["players"])
    predicted = (report.get("skill_coefficients") or {}).get(
        "predicted_player_value"
    ) or {}
    residual = (report.get("skill_coefficients") or {}).get(
        "residual_player_value"
    ) or {}
    rows = []
    for player in pool.get("players", []):
        if not isinstance(player, dict):
            continue
        player_id = str(player.get("player_id", ""))
        rating = ratings.get(player_id) or {}
        interval = rating.get("bootstrap_interval") or {}
        apm_rating = apm_ratings.get(player_id) or {}
        skills = []
        for descriptor in descriptors:
            key = str(descriptor.get("key", ""))
            skills.append(
                {
                    "key": key,
                    "label": descriptor.get("display_label", key),
                    "group": descriptor.get("semantic_group"),
                    "value": _number((player.get("skills") or {}).get(key)),
                }
            )
        rows.append(
            {
                "player_id": player_id,
                "name": player.get("name", player_id),
                "net_rating": _number(rating.get("net_rating_points_vs_replacement")),
                "interval_p05": _number(interval.get("p05")),
                "interval_p95": _number(interval.get("p95")),
                "apm_rating": _number(
                    rating.get(
                        "apm_diagnostic_value",
                        apm_rating.get("net_rating_points_vs_replacement"),
                    )
                ),
                "apm_minus_skill_rating": _number(rating.get("apm_minus_skill_rating")),
                "predicted_skill_value": _number(
                    rating.get("predicted_skill_value", predicted.get(player_id))
                ),
                "residual_skill_value": _number(
                    rating.get("residual_skill_value", residual.get(player_id))
                ),
                "skills": skills,
            }
        )
    rows.sort(
        key=lambda row: (
            row["net_rating"] if row["net_rating"] is not None else float("-inf")
        ),
        reverse=True,
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    coefficient_rows = []
    coefficients = (
        direct_skill_ratings.get("coefficients")
        or (report.get("skill_coefficients") or {}).get("coefficients")
        or {}
    )
    for key, value in coefficients.items():
        details = value if isinstance(value, dict) else {}
        descriptor = descriptor_by_key.get(str(key), {})
        coefficient_rows.append(
            {
                "key": key,
                "label": descriptor.get("display_label", key),
                "group": descriptor.get("semantic_group"),
                "net_rating_per_standard_deviation": _number(
                    details.get(
                        "net_rating_points_per_pool_standard_deviation",
                        details.get("net_rating_points_per_standard_deviation"),
                    )
                ),
                "pool_mean": _number(details.get("pool_mean")),
                "pool_standard_deviation": _number(
                    details.get("pool_standard_deviation")
                ),
            }
        )
    coefficient_rows.sort(
        key=lambda row: abs(row["net_rating_per_standard_deviation"] or 0), reverse=True
    )
    factors = report.get("conventional_three_factor_impact") or {}
    reproducibility = _report_reproducibility(path, report)
    return {
        "summary": _report_summary(path),
        "raw_artifact_dependencies": raw,
        "sizing": (report.get("sizing_and_rank_stability") or {}).get("sizing"),
        "player_rows": rows,
        "skill_descriptors": descriptors,
        "skill_coefficients": coefficient_rows,
        "skill_fit": (report.get("skill_coefficients") or {}).get("fit"),
        "player_rating_fit": direct_skill_ratings.get("fit")
        or (report.get("player_rapm") or {}).get("fit"),
        "player_rating_validation": direct_skill_ratings.get("validation")
        or (report.get("player_rapm") or {}).get("validation"),
        "apm_diagnostic_fit": (report.get("player_rapm") or {}).get("fit"),
        "skill_apm_agreement": apm_agreement,
        "reproducibility": reproducibility,
        "three_factors": {
            "scenario_weights": factors.get("scenario_weights"),
            "unavailable_factors": factors.get("unavailable_factors"),
            "definitions": factors.get("definitions"),
            "outcome_validation": factors.get("outcome_validation"),
        },
    }


@router.post("/api/skill-ratings/jobs")
def launch_skill_ratings_job(request: SkillRatingsJobRequest) -> dict[str, Any]:
    """Validate then start one isolated Skill Ratings evaluation process."""
    return _JOB_MANAGER.start(request)


@router.post("/api/skill-ratings/extensions")
def extend_skill_ratings_study(
    request: SkillRatingsExtensionRequest,
) -> dict[str, Any]:
    """Evaluate only extra games, then join them to a prior immutable study."""
    report_path = _require_repo_path(request.report_path, label="Report path")
    if not report_path.is_file():
        raise HTTPException(status_code=404, detail="Skill Ratings report not found.")
    report = _read_json(report_path)
    reproducibility = _report_reproducibility(report_path, report)
    if reproducibility is None:
        raise HTTPException(
            status_code=422,
            detail="Report lacks the immutable pool and schedule artifacts needed to extend it.",
        )
    missing_seed_keys = reproducibility.get("missing_seed_keys") or []
    if missing_seed_keys:
        labels = ", ".join(str(key).replace("_", " ") for key in missing_seed_keys)
        raise HTTPException(
            status_code=422,
            detail=(
                "This legacy report did not record its "
                f"{labels}. It cannot be safely extended; launch a new study instead."
            ),
        )
    source = reproducibility["config"]
    checkpoint_path = str((source.get("checkpoint_paths") or [""])[0] or "")
    if not checkpoint_path:
        raise HTTPException(
            status_code=422, detail="Report has no checkpoint path to extend."
        )
    source_study_id = str(
        (report.get("raw_artifact_dependencies") or {}).get("study_id") or "study"
    )
    extension_root = (
        Path(str(source["artifact_dir"]))
        / "extensions"
        / f"{source_study_id}-plus-{uuid.uuid4().hex[:8]}"
    )
    extension_seed = (
        int(request.schedule_seed)
        if request.schedule_seed is not None
        else int(source.get("schedule_seed") or 0) + 1
    )
    config = SkillRatingsJobRequest(
        checkpoint_paths=[checkpoint_path],
        pool_path=str(source["pool_path"]),
        create_pool=False,
        pool_size=int(source.get("pool_size") or 0),
        pool_seed=int(source.get("pool_seed") or 0),
        schedule_path=str(extension_root / "additional_schedule.json"),
        create_schedule=True,
        game_count=int(request.additional_games),
        schedule_seed=extension_seed,
        artifact_dir=str(extension_root),
        report_dir=str(extension_root),
        action_mode=str(source.get("action_mode") or "direct"),
        batch_size=int(request.batch_size or source.get("batch_size") or 64),
        eval_seed=int(source.get("eval_seed") or 0),
        ridge_alpha=(
            float(source["ridge_alpha"])
            if source.get("ridge_alpha") is not None
            else None
        ),
        validation_fraction=float(source.get("validation_fraction") or 0.2),
        split_seed=int(source.get("split_seed") or 0),
        bootstrap_samples=int(source.get("bootstrap_samples") or 200),
        bootstrap_seed=int(source.get("bootstrap_seed") or 1),
        extension_source=str(reproducibility["raw_study_root"]),
    )
    return _JOB_MANAGER.start(config)


@router.get("/api/skill-ratings/jobs/active")
def get_active_skill_ratings_job() -> dict[str, Any]:
    """Return the active job, if any, after a tab remount or browser reload."""
    return {"job": _JOB_MANAGER.active_snapshot()}


@router.get("/api/skill-ratings/jobs/{job_id}")
def get_skill_ratings_job(job_id: str) -> dict[str, Any]:
    return _JOB_MANAGER.snapshot(job_id)


@router.get("/api/skill-ratings/reports")
def list_skill_ratings_reports(artifact_dir: str) -> dict[str, Any]:
    report_dir = _require_repo_path(artifact_dir, label="Artifact destination")
    if not report_dir.is_dir():
        return {"reports": []}
    # ``artifact_dir`` may be one individual study folder or the durable
    # Skill Ratings library.  Search below it so completed reports do not
    # vanish merely because the launch form has generated a new timestamped
    # output folder.
    paths = set(
        report_dir.rglob(f"{_FIXED_POOL_REPORT_NAMESPACE}/*/checkpoint_report.json")
    )
    paths.update(
        report_dir.rglob(f"{_FIXED_POOL_REPORT_NAMESPACE}/*/analysis_revisions/*.json")
    )
    summaries = [_report_summary(path) for path in paths]
    summaries.sort(key=lambda report: report["modified_at"], reverse=True)
    return {"reports": summaries}


@router.get("/api/skill-ratings/report")
def get_skill_ratings_report(path: str) -> dict[str, Any]:
    report_path = _require_repo_path(path, label="Report path")
    if not report_path.is_file():
        raise HTTPException(status_code=404, detail="Skill Ratings report not found.")
    return _ui_report(report_path)
