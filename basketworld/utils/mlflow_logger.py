from typing import Any, Dict

import mlflow
from stable_baselines3.common.logger import KVWriter


class MLflowWriter(KVWriter):
    """A minimal SB3 writer that mirrors scalar metrics to MLflow."""

    def __init__(self, team_name: str):
        self.team_name = team_name

    def write(self, key_values: Dict[str, Any], key_excluded: Dict[str, Any], step: int = 0) -> None:  # type: ignore[override]
        for key, value in key_values.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(f"{self.team_name} {key}", float(value), step=int(step))

    def close(self) -> None:  # noqa: D401
        pass

    def flush(self) -> None:  # noqa: D401
        pass 