from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class RunStatus(str, Enum):
    QUEUED = "queued"
    STARTING = "starting"
    RUNNING = "running"
    PAUSING = "pausing"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    COMPLETED = "completed"
    FAILED = "failed"


class TrainingRunConfig(BaseModel):
    name: str = Field(
        default="halfcourt-multi-possession", min_length=1, max_length=120
    )
    preset: Literal["halfcourt_multi_possession"] = "halfcourt_multi_possession"
    mlflow_tracking_uri: str = Field(
        default="http://localhost:5000", min_length=1, max_length=2048
    )
    mlflow_experiment_name: str = Field(
        default="halfcourt_multi_possessions", min_length=1, max_length=250
    )
    num_updates: int = Field(default=5000, ge=1, le=1_000_000)
    policy_seed: int = Field(default=0, ge=0, le=2_147_483_647)
    historical_eval_updates: list[int] | None = None
    historical_eval_episodes: int = Field(default=200, ge=2, le=100_000)
    possession_limit_start: int = Field(default=1, ge=1, le=1000)
    possession_limit_end: int = Field(default=25, ge=1, le=1000)
    possession_limit_ramp_updates: int = Field(default=5000, ge=0)
    made_basket_restart_mode: Literal["baseline_inbound", "check", "direct_handoff"] = (
        "check"
    )
    check_setup_steps: int = Field(default=0, ge=0, le=120)
    multi_possession_use_inbounds: bool = True
    overrides: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name", "mlflow_tracking_uri", "mlflow_experiment_name")
    @classmethod
    def normalize_required_text(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("Value cannot be blank.")
        return normalized

    @field_validator("historical_eval_episodes")
    @classmethod
    def require_paired_eval_episodes(cls, value: int) -> int:
        if value % 2:
            raise ValueError("Historical evaluation episodes must be even.")
        return value

    @model_validator(mode="after")
    def validate_cross_fields(self) -> "TrainingRunConfig":
        if self.possession_limit_start > self.possession_limit_end:
            raise ValueError(
                "possession_limit_start cannot exceed possession_limit_end."
            )
        if self.historical_eval_updates is not None:
            cleaned = sorted(set(self.historical_eval_updates))
            if any(update <= 0 for update in cleaned):
                raise ValueError("Historical evaluation updates must be positive.")
            if any(update > self.num_updates for update in cleaned):
                raise ValueError(
                    "Historical evaluation updates cannot exceed num_updates."
                )
            self.historical_eval_updates = cleaned
        return self


class RunCreateRequest(BaseModel):
    config: TrainingRunConfig
    idempotency_key: str = Field(min_length=8, max_length=200)


class MlflowImportRequest(BaseModel):
    tracking_uri: str = Field(min_length=1, max_length=2048)
    run_id: str = Field(min_length=1, max_length=128)


class RunPreview(BaseModel):
    preset: str
    argv: list[str]
    environment: dict[str, str]
    display_command: str
    checkpoint_dir: str
    control_file: str
    status_file: str
    log_file: str


class TrainingRun(BaseModel):
    id: str
    idempotency_key: str
    name: str
    preset: str
    status: RunStatus
    config: dict[str, Any]
    command: list[str]
    environment: dict[str, str]
    cwd: str
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    pid: int | None = None
    process_start_token: str | None = None
    exit_code: int | None = None
    mlflow_run_id: str | None = None
    current_update: int = 0
    target_updates: int
    checkpoint_path: str | None = None
    control_file: str
    status_file: str
    log_file: str
    error_message: str | None = None


class RunEvent(BaseModel):
    id: str
    run_id: str
    event_type: str
    status: RunStatus | None = None
    message: str | None = None
    payload: dict[str, Any]
    created_at: datetime


class RunDetail(TrainingRun):
    events: list[RunEvent] = Field(default_factory=list)
    worker_status: dict[str, Any] | None = None
    display_command: str
    resolved_config: list[dict[str, Any]] = Field(default_factory=list)
    training_elapsed_seconds: float = 0.0


class RunActionResponse(BaseModel):
    run: TrainingRun
    message: str


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    database_path: str
    now: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
