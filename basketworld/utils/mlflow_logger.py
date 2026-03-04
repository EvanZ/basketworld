from typing import Any, Dict

import mlflow
from stable_baselines3.common.logger import KVWriter


class MLflowWriter(KVWriter):
    """A minimal SB3 writer that mirrors scalar metrics to MLflow."""

    def __init__(self, team_name: str, log_every_n_writes: int = 1):
        self.team_name = team_name
        self.log_every_n_writes = max(1, int(log_every_n_writes))
        self._write_calls = 0

    def write(self, key_values: Dict[str, Any], key_excludes: Dict[str, str], step: int = 0) -> None:
        self._write_calls += 1
        if self._write_calls % self.log_every_n_writes != 0:
            return
        for key, value in key_values.items():
            # Handle tensor values by converting them to a scalar float
            if hasattr(value, "item"):
                value = value.item()
            
            if isinstance(value, (int, float)):
                mlflow.log_metric(f"{self.team_name} {key}", float(value), step=int(step))

    def close(self) -> None:  # noqa: D401
        pass

    def flush(self) -> None:  # noqa: D401
        pass
