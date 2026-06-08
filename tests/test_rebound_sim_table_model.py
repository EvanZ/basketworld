from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from analytics.rebound_physics.dataset import court_mujoco_xy, mapping_record, nearest_court_cell_index
from analytics.rebound_sim.model import build_court, simulate_rebounds
from analytics.rebound_sim.table_model import FittedReboundTableModel


def test_fitted_table_model_samples_and_reflects_targets(tmp_path: Path) -> None:
    court = build_court()
    model_dir, left_shot_idx, right_shot_idx, canonical_target_idx = _write_minimal_table_model(tmp_path, court)
    model = FittedReboundTableModel.load(model_dir, court=court)

    rng = np.random.default_rng(123)
    right_target = model.sample_target_index(rng, court, right_shot_idx)
    assert right_target == canonical_target_idx

    rng = np.random.default_rng(123)
    left_target = model.sample_target_index(rng, court, left_shot_idx)
    expected_left_target = _mirror_target_index(model, canonical_target_idx)
    assert left_target == expected_left_target
    assert left_target != canonical_target_idx



def test_fitted_table_model_target_probabilities_reflect_rows(tmp_path: Path) -> None:
    court = build_court()
    model_dir, left_shot_idx, right_shot_idx, canonical_target_idx = _write_minimal_table_model(tmp_path, court)
    model = FittedReboundTableModel.load(model_dir, court=court)

    right_probs = model.target_probabilities(court, right_shot_idx)
    assert right_probs.shape == (len(court.cells),)
    assert float(right_probs.sum()) == pytest.approx(1.0)
    assert int(np.argmax(right_probs)) == canonical_target_idx

    left_probs = model.target_probabilities(court, left_shot_idx)
    expected_left_target = _mirror_target_index(model, canonical_target_idx)
    assert left_probs.shape == (len(court.cells),)
    assert float(left_probs.sum()) == pytest.approx(1.0)
    assert int(np.argmax(left_probs)) == expected_left_target
    assert expected_left_target != canonical_target_idx


def test_simulate_rebounds_can_use_fitted_table_target_sampler(tmp_path: Path) -> None:
    court = build_court()
    model_dir, left_shot_idx, _, canonical_target_idx = _write_minimal_table_model(tmp_path, court)
    model = FittedReboundTableModel.load(model_dir, court=court)
    expected_left_target = _mirror_target_index(model, canonical_target_idx)

    result = simulate_rebounds(
        16,
        seed=7,
        court=court,
        shot_indices=np.full(16, left_shot_idx, dtype=np.int32),
        target_sampler=model.sample_target_index,
    )

    assert set(result.rebound_kinds.tolist()) == {"table"}
    assert set(result.rebound_regions.tolist()) == {"table"}
    np.testing.assert_array_equal(result.target_indices, np.full(16, expected_left_target, dtype=np.int32))


def test_fitted_table_model_rejects_court_size_mismatch(tmp_path: Path) -> None:
    court = build_court()
    model_dir, _, _, _ = _write_minimal_table_model(tmp_path, court)
    bad_court = build_court(type(court.spec)(rows=8, cols=8))
    with pytest.raises(ValueError, match="court size"):
        FittedReboundTableModel.load(model_dir, court=bad_court)


def _write_minimal_table_model(tmp_path: Path, court):
    model_dir = tmp_path / "model"
    shot_type_dir = model_dir / "shot_type_models" / "jumper"
    shot_type_dir.mkdir(parents=True)
    meters_per_bw_unit = 0.9835331644547978
    court_xy = court_mujoco_xy(court, meters_per_bw_unit)

    left_shot_idx = 0
    right_shot_idx = 64
    canonical_shot_idx = 64
    canonical_target_idx = 71
    probs = np.zeros((1, len(court.cells)), dtype=np.float64)
    probs[0, canonical_target_idx] = 1.0
    np.save(shot_type_dir / "rebound_transition_probs.npy", probs)
    np.save(shot_type_dir / "rebound_transition_row_shot_cell_indices.npy", np.asarray([canonical_shot_idx], dtype=np.int32))

    (model_dir / "rebound_fit_summary.json").write_text(
        json.dumps(
            {
                "court_rows": court.spec.rows,
                "court_cols": court.spec.cols,
                "meters_per_bw_unit": meters_per_bw_unit,
                "layup_max_distance_hex": 1.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    records = [
        mapping_record(
            court,
            type("Mapping", (), {
                "shot_cell_index": left_shot_idx,
                "canonical_cell_index": canonical_shot_idx,
                "reflection_sign": -1,
                "symmetry_class": "reflected_left_to_right",
            })(),
            meters_per_bw_unit=meters_per_bw_unit,
        ),
        mapping_record(
            court,
            type("Mapping", (), {
                "shot_cell_index": right_shot_idx,
                "canonical_cell_index": canonical_shot_idx,
                "reflection_sign": 1,
                "symmetry_class": "canonical_right",
            })(),
            meters_per_bw_unit=meters_per_bw_unit,
        ),
    ]
    (model_dir / "rebound_canonical_shot_mapping.json").write_text(
        json.dumps(records) + "\n",
        encoding="utf-8",
    )
    assert court_xy[canonical_target_idx, 0] > 0.0
    return model_dir, left_shot_idx, right_shot_idx, canonical_target_idx


def _mirror_target_index(model: FittedReboundTableModel, canonical_target_idx: int) -> int:
    target_xy = model.court_xy_mujoco[int(canonical_target_idx)].copy()
    target_xy[0] *= -1.0
    return nearest_court_cell_index(model.court_xy_mujoco, target_xy)
