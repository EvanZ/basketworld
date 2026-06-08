from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from analytics.rebound_physics.dataset import (
    court_mujoco_xy,
    nearest_court_cell_index,
    shot_type_for_cell,
)
from analytics.rebound_sim.model import Court


@dataclass(frozen=True)
class ShotCellMapping:
    shot_cell_index: int
    canonical_cell_index: int
    reflection_sign: int
    symmetry_class: str


@dataclass(frozen=True)
class FittedReboundTableModel:
    """Runtime sampler for fitted MuJoCo rebound target tables.

    The fitted artifacts are built only for canonical shot cells. At runtime we
    map the shot cell into that canonical side, sample a canonical rebound
    target, then reflect the target back when the original shot was mirrored.
    """

    model_dir: Path
    meters_per_bw_unit: float
    mapping_by_shot_index: dict[int, ShotCellMapping]
    probs_by_shot_type: dict[str, np.ndarray]
    row_lookup_by_shot_type: dict[str, dict[int, int]]
    court_xy_mujoco: np.ndarray
    summary: dict[str, Any]

    @classmethod
    def load(cls, model_dir: str | Path, *, court: Court) -> "FittedReboundTableModel":
        model_dir = Path(model_dir)
        summary_path = model_dir / "rebound_fit_summary.json"
        mapping_path = model_dir / "rebound_canonical_shot_mapping.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing fitted rebound summary: {summary_path}")
        if not mapping_path.exists():
            raise FileNotFoundError(f"Missing fitted rebound mapping: {mapping_path}")

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rows = int(summary.get("court_rows", court.spec.rows))
        cols = int(summary.get("court_cols", court.spec.cols))
        if rows != int(court.spec.rows) or cols != int(court.spec.cols):
            raise ValueError(
                "Fitted rebound table court size does not match runtime court: "
                f"table={rows}x{cols}, runtime={court.spec.rows}x{court.spec.cols}"
            )
        meters_per_bw_unit = float(summary["meters_per_bw_unit"])
        court_xy = court_mujoco_xy(court, meters_per_bw_unit)

        mapping_records = json.loads(mapping_path.read_text(encoding="utf-8"))
        mapping_by_shot_index = {
            int(record["shot_cell_index"]): ShotCellMapping(
                shot_cell_index=int(record["shot_cell_index"]),
                canonical_cell_index=int(record["canonical_shot_cell_index"]),
                reflection_sign=int(record["reflection_sign"]),
                symmetry_class=str(record["symmetry_class"]),
            )
            for record in mapping_records
        }

        probs_by_shot_type: dict[str, np.ndarray] = {}
        row_lookup_by_shot_type: dict[str, dict[int, int]] = {}
        shot_type_dir = model_dir / "shot_type_models"
        for type_dir in sorted(path for path in shot_type_dir.iterdir() if path.is_dir()):
            probs_path = type_dir / "rebound_transition_probs.npy"
            rows_path = type_dir / "rebound_transition_row_shot_cell_indices.npy"
            if not probs_path.exists() or not rows_path.exists():
                continue
            probs = np.load(probs_path).astype(np.float64)
            row_indices = np.load(rows_path).astype(np.int32)
            if probs.ndim != 2 or probs.shape[0] != row_indices.size or probs.shape[1] != len(court.cells):
                raise ValueError(
                    f"Invalid fitted rebound arrays for shot_type={type_dir.name!r}: "
                    f"probs={probs.shape}, rows={row_indices.shape}, cells={len(court.cells)}"
                )
            probs_by_shot_type[type_dir.name] = probs
            row_lookup_by_shot_type[type_dir.name] = {
                int(cell_index): int(row) for row, cell_index in enumerate(row_indices.tolist())
            }
        if not probs_by_shot_type:
            raise ValueError(f"No shot-type transition arrays found under {shot_type_dir}")

        return cls(
            model_dir=model_dir,
            meters_per_bw_unit=meters_per_bw_unit,
            mapping_by_shot_index=mapping_by_shot_index,
            probs_by_shot_type=probs_by_shot_type,
            row_lookup_by_shot_type=row_lookup_by_shot_type,
            court_xy_mujoco=court_xy,
            summary=summary,
        )

    def sample_target_index(self, rng: np.random.Generator, court: Court, shot_index: int) -> int:
        probs = self.target_probabilities(court, shot_index)
        return int(rng.choice(len(court.cells), p=probs))

    def target_probabilities(self, court: Court, shot_index: int) -> np.ndarray:
        """Return runtime rebound target probabilities for a shot cell.

        Fitted rows live in canonical court coordinates. This maps the runtime
        shot cell to its canonical row and reflects the full target distribution
        back when the original shot was mirrored.
        """
        mapping = self.mapping_for_shot(shot_index)
        shot_type = self.shot_type_for_shot(court, shot_index)
        probs = self.probs_by_shot_type.get(shot_type)
        row_lookup = self.row_lookup_by_shot_type.get(shot_type)
        if probs is None or row_lookup is None:
            raise ValueError(f"Fitted rebound table has no arrays for shot_type={shot_type!r}")
        row = row_lookup.get(mapping.canonical_cell_index)
        if row is None:
            raise ValueError(
                f"Fitted rebound table has no row for shot_type={shot_type!r}, "
                f"canonical_cell_index={mapping.canonical_cell_index}"
            )
        canonical_probs = np.asarray(probs[int(row)], dtype=np.float64)
        if int(mapping.reflection_sign) >= 0:
            runtime_probs = canonical_probs.copy()
        else:
            runtime_probs = np.zeros_like(canonical_probs)
            for canonical_target, prob in enumerate(canonical_probs):
                runtime_target = self.reflect_target_index(canonical_target, mapping.reflection_sign)
                runtime_probs[int(runtime_target)] += float(prob)
        total = float(runtime_probs.sum())
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError(f"Fitted rebound table row has no probability mass for shot_cell_index={shot_index}")
        return runtime_probs / total

    def shot_type_for_shot(self, court: Court, shot_index: int) -> str:
        return shot_type_for_cell(
            court,
            int(shot_index),
            layup_max_distance_hex=float(self.summary.get("layup_max_distance_hex", 1.0)),
        )

    def mapping_for_shot(self, shot_index: int) -> ShotCellMapping:
        mapping = self.mapping_by_shot_index.get(int(shot_index))
        if mapping is None:
            raise ValueError(f"Fitted rebound table has no mapping for shot_cell_index={shot_index}")
        return mapping

    def reflect_target_index(self, canonical_target_index: int, reflection_sign: int) -> int:
        if int(reflection_sign) >= 0:
            return int(canonical_target_index)
        target_xy = self.court_xy_mujoco[int(canonical_target_index)].copy()
        target_xy[0] *= -1.0
        return nearest_court_cell_index(self.court_xy_mujoco, target_xy)
