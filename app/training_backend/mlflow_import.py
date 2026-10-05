from __future__ import annotations

import ast
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from app.training_backend.models import TrainingRunConfig


RESOLVED_CONFIG_ARTIFACT = "metadata/resolved_training_config.json"


def _parse_value(raw: Any, field: dict[str, Any]) -> Any:
    if not isinstance(raw, str):
        return raw
    text = raw.strip()
    if text.lower() in {"none", "null", ""}:
        return None
    if field["type"] == "bool":
        if text.lower() in {"true", "1", "yes", "on"}:
            return True
        if text.lower() in {"false", "0", "no", "off"}:
            return False
        raise ValueError(f"Cannot parse {field['name']} as bool.")
    if field["type"] == "int":
        return int(text)
    if field["type"] == "float":
        return float(text)
    if field["type"] == "list":
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            value = ast.literal_eval(text)
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"Cannot parse {field['name']} as list.")
        item_type = field.get("item_type") or "str"
        return [_parse_value(str(item), {**field, "type": item_type}) for item in value]
    return text


def config_from_mlflow_values(
    *,
    values: dict[str, Any],
    params: dict[str, str],
    schema: list[dict[str, Any]],
    tracking_uri: str,
    experiment_name: str,
    run_name: str,
) -> tuple[TrainingRunConfig, list[str], list[str]]:
    fields = {field["name"]: field for field in schema}
    recovered: dict[str, Any] = {}
    for name, field in fields.items():
        candidates = (
            values.get(name),
            params.get(name),
            params.get(f"jax/{name}"),
            params.get(f"jax/env/{name}"),
        )
        raw = next(
            (candidate for candidate in candidates if candidate is not None), None
        )
        if raw is None:
            continue
        try:
            value = _parse_value(raw, field)
        except (TypeError, ValueError, SyntaxError):
            continue
        if value == -1 and field.get("default") is None:
            value = None
        recovered[name] = value

    base = TrainingRunConfig(
        name=f"{run_name or 'MLflow run'} copy",
        mlflow_tracking_uri=tracking_uri,
        mlflow_experiment_name=experiment_name,
    ).model_dump()
    application_field_mapping = {
        "num_updates": "num_updates",
        "policy_seed": "policy_seed",
        "historical_eval_episodes": "historical_eval_episodes",
        "multi_possession_limit_start": "possession_limit_start",
        "multi_possession_limit_end": "possession_limit_end",
        "multi_possession_limit_ramp_updates": "possession_limit_ramp_updates",
        "made_basket_restart_mode": "made_basket_restart_mode",
        "check_setup_steps": "check_setup_steps",
        "multi_possession_use_inbounds": "multi_possession_use_inbounds",
    }
    for source, target in application_field_mapping.items():
        if source in recovered and recovered[source] is not None:
            base[target] = recovered[source]
    historical_updates = recovered.get("historical_eval_updates")
    if isinstance(historical_updates, str):
        historical_updates = [
            int(item) for item in historical_updates.split(",") if item.strip()
        ]
    if isinstance(historical_updates, list):
        base["historical_eval_updates"] = historical_updates or None

    application_field_names = set(application_field_mapping) | {
        "historical_eval_updates",
        "mlflow_experiment_name",
        "mlflow_run_name",
    }
    base["overrides"] = {
        name: value
        for name, value in recovered.items()
        if name not in application_field_names and fields[name].get("editable", False)
    }
    config = TrainingRunConfig.model_validate(base)
    matched = sorted(recovered)
    missing = sorted(name for name in fields if name not in recovered)
    return config, matched, missing


def import_mlflow_run(
    *, tracking_uri: str, run_id: str, schema: list[dict[str, Any]]
) -> dict[str, Any]:
    import mlflow

    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    run = client.get_run(run_id)
    experiment = client.get_experiment(run.info.experiment_id)
    values: dict[str, Any] = {}
    source = "parameters"
    with TemporaryDirectory(prefix="basketworld-training-import-") as temp_dir:
        try:
            metadata_artifacts = client.list_artifacts(run_id, "metadata")
            if any(
                artifact.path == RESOLVED_CONFIG_ARTIFACT
                for artifact in metadata_artifacts
            ):
                path = client.download_artifacts(
                    run_id, RESOLVED_CONFIG_ARTIFACT, dst_path=temp_dir
                )
                payload = json.loads(Path(path).read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    values = payload
                    source = "resolved_config_artifact"
        except Exception:
            values = {}

    config, matched, missing = config_from_mlflow_values(
        values=values,
        params=dict(run.data.params),
        schema=schema,
        tracking_uri=tracking_uri,
        experiment_name=experiment.name,
        run_name=run.info.run_name or run_id[:8],
    )
    return {
        "config": config.model_dump(),
        "source": source,
        "source_run_id": run_id,
        "source_run_name": run.info.run_name,
        "experiment_name": experiment.name,
        "matched_fields": matched,
        "missing_fields": missing,
        "exact": source == "resolved_config_artifact",
    }
