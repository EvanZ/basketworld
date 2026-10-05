from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager, suppress
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles

from app.training_backend.config import Settings
from app.training_backend.application_training import (
    load_application_config_schema,
    resolve_run_config_schema,
)
from app.training_backend.database import TrainingRepository
from app.training_backend.jobs import JobManager
from app.training_backend.mlflow_import import import_mlflow_run
from app.training_backend.mlflow_metrics import (
    MAX_HISTORY_POINTS,
    metric_catalog,
    metric_histories,
)
from app.training_backend.presets import format_display_command
from app.training_backend.run_timing import active_training_seconds
from app.training_backend.models import (
    HealthResponse,
    MlflowImportRequest,
    RunActionResponse,
    RunCreateRequest,
    RunDetail,
    RunPreview,
    TrainingRun,
    TrainingRunConfig,
)


settings = Settings.from_env()
repository = TrainingRepository(settings.database_path)
manager = JobManager(settings, repository)


async def _poll_jobs() -> None:
    while True:
        await asyncio.to_thread(manager.poll)
        await asyncio.sleep(settings.poll_interval_seconds)


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings.ensure_directories()
    repository.initialize()
    manager.poll()
    poller = asyncio.create_task(_poll_jobs())
    try:
        yield
    finally:
        poller.cancel()
        with suppress(asyncio.CancelledError):
            await poller


app = FastAPI(
    title="BasketWorld Training Control",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _run_or_404(run_id: str) -> dict:
    run = repository.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Training run not found.")
    return run


def _action(run_id: str, callback, message: str) -> RunActionResponse:
    try:
        run = callback(run_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Training run not found.") from None
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return RunActionResponse(run=TrainingRun.model_validate(run), message=message)


@app.get("/api/v1/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(database_path=str(settings.database_path))


@app.get("/api/v1/presets")
def presets() -> list[dict]:
    return [
        {
            "id": "halfcourt_multi_possession",
            "name": "Halfcourt multi-possession",
            "description": "The current JAX 5v5 halfcourt experiment preset.",
            "defaults": TrainingRunConfig().model_dump(),
        }
    ]


@app.get("/api/v1/config-schema")
def config_schema() -> list[dict]:
    return load_application_config_schema(str(settings.repo_root))


@app.post("/api/v1/mlflow/import-run")
def import_run_from_mlflow(request: MlflowImportRequest) -> dict:
    try:
        return import_mlflow_run(
            tracking_uri=request.tracking_uri,
            run_id=request.run_id,
            schema=load_application_config_schema(str(settings.repo_root)),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502, detail=f"Unable to import MLflow run: {exc}"
        ) from exc


@app.post("/api/v1/runs/preview", response_model=RunPreview)
def preview_run(config: TrainingRunConfig) -> RunPreview:
    return manager.preview(config)


@app.post("/api/v1/runs", response_model=TrainingRun, status_code=201)
def create_run(request: RunCreateRequest) -> TrainingRun:
    try:
        run = manager.create_and_launch(
            config=request.config,
            idempotency_key=request.idempotency_key,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return TrainingRun.model_validate(run)


@app.get("/api/v1/runs", response_model=list[TrainingRun])
def list_runs(limit: int = Query(default=100, ge=1, le=1000)) -> list[TrainingRun]:
    manager.poll()
    return [
        TrainingRun.model_validate(run) for run in repository.list_runs(limit=limit)
    ]


@app.get("/api/v1/runs/{run_id}", response_model=RunDetail)
def get_run(run_id: str) -> RunDetail:
    manager.poll()
    run = _run_or_404(run_id)
    events = repository.list_events(run_id)
    return RunDetail.model_validate(
        {
            **run,
            "events": events,
            "worker_status": manager.worker_status(run),
            "display_command": format_display_command(
                run["command"], run["environment"]
            ),
            "resolved_config": resolve_run_config_schema(tuple(run["command"])),
            "training_elapsed_seconds": active_training_seconds(events),
        }
    )


@app.get("/api/v1/runs/{run_id}/metrics/catalog")
def get_run_metric_catalog(run_id: str) -> dict:
    run = _run_or_404(run_id)
    if not run.get("mlflow_run_id"):
        return {"metrics": []}
    try:
        metrics = metric_catalog(
            tracking_uri=run["config"]["mlflow_tracking_uri"],
            run_id=run["mlflow_run_id"],
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502, detail=f"Unable to read MLflow metrics: {exc}"
        ) from exc
    return {"metrics": metrics}


@app.get("/api/v1/runs/{run_id}/metrics")
def get_run_metric_histories(
    run_id: str,
    names: list[str] = Query(default=[]),
    max_points: int = Query(default=MAX_HISTORY_POINTS, ge=20, le=1000),
) -> dict:
    run = _run_or_404(run_id)
    if not run.get("mlflow_run_id") or not names:
        return {"series": {}}
    try:
        series = metric_histories(
            tracking_uri=run["config"]["mlflow_tracking_uri"],
            run_id=run["mlflow_run_id"],
            names=names,
            max_points=max_points,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502, detail=f"Unable to read MLflow metric history: {exc}"
        ) from exc
    return {"series": series}


@app.get("/api/v1/runs/{run_id}/logs", response_class=PlainTextResponse)
def get_run_logs(
    run_id: str,
    tail_bytes: int = Query(default=100_000, ge=1_000, le=2_000_000),
) -> str:
    run = _run_or_404(run_id)
    path = Path(run["log_file"]).resolve()
    allowed_root = (settings.data_dir / "runs").resolve()
    if not path.is_relative_to(allowed_root):
        raise HTTPException(
            status_code=400, detail="Run log path is outside the data directory."
        )
    try:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - tail_bytes))
            return handle.read().decode("utf-8", errors="replace")
    except FileNotFoundError:
        return ""


@app.post("/api/v1/runs/{run_id}/checkpoint", response_model=RunActionResponse)
def checkpoint_run(run_id: str) -> RunActionResponse:
    return _action(run_id, manager.request_checkpoint, "Checkpoint requested.")


@app.post("/api/v1/runs/{run_id}/pause", response_model=RunActionResponse)
def pause_run(run_id: str) -> RunActionResponse:
    return _action(run_id, manager.request_pause, "Pause requested.")


@app.post("/api/v1/runs/{run_id}/resume", response_model=RunActionResponse)
def resume_run(run_id: str) -> RunActionResponse:
    return _action(run_id, manager.resume, "Training resumed.")


@app.post("/api/v1/runs/{run_id}/stop", response_model=RunActionResponse)
def stop_run(run_id: str, force: bool = False) -> RunActionResponse:
    return _action(
        run_id,
        lambda value: manager.request_stop(value, force=force),
        "Force stop requested." if force else "Stop requested.",
    )


if os.getenv("BW_TRAINING_SERVE_FRONTEND", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}:
    frontend_dist = settings.repo_root / "app" / "training_frontend" / "dist"
    if not frontend_dist.is_dir():
        raise RuntimeError(
            f"Training frontend build not found at {frontend_dist}. Run npm build first."
        )
    app.mount(
        "/", StaticFiles(directory=frontend_dist, html=True), name="training_frontend"
    )
