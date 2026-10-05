from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    repo_root: Path
    data_dir: Path
    database_path: Path
    cors_origins: tuple[str, ...]
    poll_interval_seconds: float

    @classmethod
    def from_env(cls) -> "Settings":
        default_root = Path(__file__).resolve().parents[2]
        repo_root = Path(os.getenv("BW_TRAINING_REPO_ROOT", default_root)).resolve()
        data_dir = Path(
            os.getenv("BW_TRAINING_DATA_DIR", repo_root / "var" / "training_app")
        ).resolve()
        database_path = Path(
            os.getenv("BW_TRAINING_DB_PATH", data_dir / "training.duckdb")
        ).resolve()
        origins = tuple(
            value.strip()
            for value in os.getenv(
                "BW_TRAINING_CORS_ORIGINS",
                "http://localhost:5174,http://127.0.0.1:5174",
            ).split(",")
            if value.strip()
        )
        return cls(
            repo_root=repo_root,
            data_dir=data_dir,
            database_path=database_path,
            cors_origins=origins,
            poll_interval_seconds=float(
                os.getenv("BW_TRAINING_POLL_INTERVAL_SECONDS", "2")
            ),
        )

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
