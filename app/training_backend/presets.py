from __future__ import annotations

import os
import shlex
from pathlib import Path
from uuid import UUID

from app.training_backend.models import RunPreview, TrainingRunConfig
from app.training_backend.application_training import (
    application_preset_values,
    load_application_config_schema,
)
from app.training_backend.config_schema import compile_config_values, compile_overrides


DEFAULT_HISTORICAL_UPDATES = (100, 500, 2500, 5000, 10000, 20000, 30000)


def format_display_command(argv: list[str], environment: dict[str, str]) -> str:
    env_prefix = " ".join(
        f"{key}={shlex.quote(value)}" for key, value in sorted(environment.items())
    )
    return f"{env_prefix} {shlex.join(argv)}".strip()


def _historical_updates(config: TrainingRunConfig) -> list[int]:
    if config.historical_eval_updates is not None:
        return list(config.historical_eval_updates)
    updates = [
        value for value in DEFAULT_HISTORICAL_UPDATES if value <= config.num_updates
    ]
    if config.num_updates not in updates:
        updates.append(config.num_updates)
    return sorted(set(updates))


def build_halfcourt_multi_possession_preview(
    *,
    config: TrainingRunConfig,
    run_id: str,
    repo_root: Path,
    data_dir: Path,
    resume_checkpoint: str | None = None,
    mlflow_resume_run_id: str | None = None,
) -> RunPreview:
    UUID(run_id)
    run_dir = data_dir / "runs" / run_id
    checkpoint_dir = run_dir / "checkpoints"
    control_file = run_dir / "control.json"
    status_file = run_dir / "worker_status.json"
    log_file = run_dir / "training.log"
    python_bin = repo_root / ".env" / "bin" / "python"

    schema = load_application_config_schema(str(repo_root))
    # Validate user-supplied values separately before merging them into the
    # trusted application recipe, which also contains internal runtime fields.
    compile_overrides(config.overrides, schema)
    values = {
        **application_preset_values(repo_root),
        **config.overrides,
        "policy_seed": config.policy_seed,
        "mlflow_run_name": config.name,
        "mlflow_experiment_name": config.mlflow_experiment_name,
        "num_updates": config.num_updates,
        "historical_eval_updates": ",".join(
            str(value) for value in _historical_updates(config)
        ),
        "historical_eval_episodes": config.historical_eval_episodes,
        "multi_possession_limit": config.possession_limit_end,
        "multi_possession_limit_start": config.possession_limit_start,
        "multi_possession_limit_end": config.possession_limit_end,
        "multi_possession_limit_ramp_updates": config.possession_limit_ramp_updates,
        "multi_possession_use_inbounds": config.multi_possession_use_inbounds,
        "made_basket_restart_mode": config.made_basket_restart_mode,
        "check_setup_steps": config.check_setup_steps,
        "checkpoint_dir": str(checkpoint_dir),
        "control_file": str(control_file),
        "training_status_file": str(status_file),
    }
    if resume_checkpoint:
        values["resume_checkpoint"] = str(resume_checkpoint)
    if mlflow_resume_run_id:
        values["mlflow_resume_run_id"] = str(mlflow_resume_run_id)
    trainer_argv = compile_config_values(values, schema, require_editable=False)
    argv = [
        str(python_bin),
        "-m",
        "basketworld_jax.train.main",
        *trainer_argv,
    ]

    environment = {
        "MLFLOW_TRACKING_URI": config.mlflow_tracking_uri,
        "MPLCONFIGDIR": "/tmp/basketworld-training-matplotlib",
        "PYTHONPATH": str(repo_root),
    }
    display_command = format_display_command(argv, environment)
    return RunPreview(
        preset=config.preset,
        argv=argv,
        environment=environment,
        display_command=display_command,
        checkpoint_dir=str(checkpoint_dir),
        control_file=str(control_file),
        status_file=str(status_file),
        log_file=str(log_file),
    )
