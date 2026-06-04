from __future__ import annotations

import numpy as np
import pytest

from analytics.rebound_physics.dataset import (
    build_canonical_shot_mappings,
    build_dataset_court,
    build_shot_make_arrays,
    build_transition_arrays,
    build_transition_arrays_by_shot_type,
    canonicalize_shot_cell,
    court_mujoco_xy,
    nearest_court_cell_index,
    parse_shot_cells,
    shot_indices_from_cells,
    shot_type_for_cell,
)
from analytics.rebound_physics.run_rebound_dataset import _record_counts_toward_miss_target
from analytics.rebound_physics.run_rebound_fit import _filter_fit_records


def test_court_mujoco_mapping_round_trips_cell_centers() -> None:
    court = build_dataset_court(rows=9, cols=8)
    xy = court_mujoco_xy(court, meters_per_bw_unit=0.5)
    for idx in range(len(court.cells)):
        assert nearest_court_cell_index(xy, xy[idx]) == idx


def test_canonicalization_maps_to_nonnegative_lateral_side() -> None:
    court = build_dataset_court(rows=9, cols=8)
    xy = court_mujoco_xy(court, meters_per_bw_unit=0.5)
    for idx in range(len(court.cells)):
        mapping = canonicalize_shot_cell(
            court,
            idx,
            meters_per_bw_unit=0.5,
            court_xy_mujoco=xy,
        )
        assert xy[mapping.canonical_cell_index, 0] >= -1e-8
        if xy[idx, 0] < -1e-8:
            assert mapping.reflection_sign == -1
        else:
            assert mapping.reflection_sign == 1


def test_build_canonical_mappings_reduces_symmetric_shot_cells() -> None:
    court = build_dataset_court(rows=9, cols=8)
    shot_indices = shot_indices_from_cells(court)
    canonical_indices, mappings = build_canonical_shot_mappings(
        court,
        shot_indices,
        meters_per_bw_unit=0.5,
    )
    assert len(mappings) == len(shot_indices)
    assert 0 < len(canonical_indices) < len(shot_indices)
    assert court.basket_index not in canonical_indices


def test_shot_indices_can_explicitly_include_basket_for_dunks() -> None:
    court = build_dataset_court(rows=9, cols=8)
    basket_cell = court.cells[int(court.basket_index)]
    default_indices = shot_indices_from_cells(court, [basket_cell])
    dunk_indices = shot_indices_from_cells(court, [basket_cell], include_basket=True)
    assert default_indices == []
    assert dunk_indices == [int(court.basket_index)]


def test_shot_type_for_cell_splits_dunk_finger_roll_layup_and_jumper() -> None:
    court = build_dataset_court(rows=9, cols=8)
    basket_idx = int(court.basket_index)
    finger_roll_idx = next(
        idx
        for idx, distance in enumerate(court.distance_hex.tolist())
        if idx != basket_idx
        and float(distance) <= 1.0 + 1e-6
        and abs(float(court.xy[idx, 1] - court.rim_xy[1])) <= 1e-6
    )
    layup_idx = next(
        idx
        for idx, distance in enumerate(court.distance_hex.tolist())
        if idx != basket_idx
        and float(distance) <= 1.0 + 1e-6
        and abs(float(court.xy[idx, 1] - court.rim_xy[1])) > 1e-6
    )
    jumper_idx = next(
        idx
        for idx, distance in enumerate(court.distance_hex.tolist())
        if float(distance) > 1.0 + 1e-6
    )
    assert shot_type_for_cell(court, basket_idx) == "dunk"
    assert shot_type_for_cell(court, finger_roll_idx) == "finger_roll"
    assert shot_type_for_cell(court, layup_idx) == "layup"
    assert shot_type_for_cell(court, jumper_idx) == "jumper"


def test_parse_shot_cells_accepts_flat_coordinate_string() -> None:
    assert parse_shot_cells("-4,8,3,4;(3,1)") == [(-4, 8), (3, 4), (3, 1)]
    assert parse_shot_cells("") == []
    with pytest.raises(ValueError):
        parse_shot_cells("-4,8,3")


def test_record_counts_toward_miss_target_modes() -> None:
    made = {"made": True, "landing_cell_index": 1, "catch_cell_index": 2}
    clean_miss = {"made": False, "landing_cell_index": 1, "catch_cell_index": 2}
    air_miss = {"made": False, "landing_cell_index": None, "catch_cell_index": 2}
    behind_catch = {
        "made": False,
        "landing_cell_index": 1,
        "catch_cell_index": 2,
        "catch_behind_backboard": True,
    }

    assert not _record_counts_toward_miss_target(made, "any")
    assert _record_counts_toward_miss_target(clean_miss, "any")
    assert _record_counts_toward_miss_target(clean_miss, "landing")
    assert not _record_counts_toward_miss_target(air_miss, "landing")
    assert _record_counts_toward_miss_target(air_miss, "catch")
    assert not _record_counts_toward_miss_target(behind_catch, "catch")
    with pytest.raises(ValueError):
        _record_counts_toward_miss_target(clean_miss, "bad")


def test_transition_arrays_count_missed_records() -> None:
    records = [
        {"canonical_shot_cell_index": 2, "landing_cell_index": 5},
        {"canonical_shot_cell_index": 2, "landing_cell_index": 5},
        {"canonical_shot_cell_index": 2, "landing_cell_index": 6},
        {"canonical_shot_cell_index": 3, "landing_cell_index": 6},
        {"canonical_shot_cell_index": 3, "landing_cell_index": None},
    ]
    transitions = build_transition_arrays(records, canonical_shot_indices=[2, 3, 4], cell_count=8)
    assert transitions.counts.shape == (3, 8)
    assert transitions.probs.shape == (3, 8)
    assert transitions.logits.shape == (3, 8)
    assert transitions.counts[0, 5] == 2.0
    assert transitions.counts[0, 6] == 1.0
    assert transitions.counts[1, 6] == 1.0
    np.testing.assert_allclose(transitions.probs[0].sum(), 1.0)
    np.testing.assert_allclose(transitions.probs[1].sum(), 1.0)
    np.testing.assert_allclose(transitions.probs[2], np.full(8, 1.0 / 8.0))



def test_transition_arrays_can_use_catch_target_field() -> None:
    records = [
        {"canonical_shot_cell_index": 2, "landing_cell_index": 5, "catch_cell_index": 1},
        {"canonical_shot_cell_index": 2, "landing_cell_index": 5, "catch_cell_index": 1},
        {"canonical_shot_cell_index": 2, "landing_cell_index": 6, "catch_cell_index": 4},
        {"canonical_shot_cell_index": 3, "landing_cell_index": 6, "catch_cell_index": None},
    ]
    transitions = build_transition_arrays(
        records,
        canonical_shot_indices=[2, 3],
        cell_count=8,
        target_cell_field="catch_cell_index",
    )
    assert transitions.counts[0, 1] == 2.0
    assert transitions.counts[0, 4] == 1.0
    assert transitions.counts[0, 5] == 0.0
    np.testing.assert_allclose(transitions.probs[0].sum(), 1.0)
    np.testing.assert_allclose(transitions.probs[1], np.full(8, 1.0 / 8.0))


def test_transition_arrays_by_shot_type_use_separate_rows() -> None:
    court = build_dataset_court(rows=9, cols=8)
    basket_idx = int(court.basket_index)
    finger_roll_idx = next(
        idx
        for idx, distance in enumerate(court.distance_hex.tolist())
        if idx != basket_idx
        and float(distance) <= 1.0 + 1e-6
        and abs(float(court.xy[idx, 1] - court.rim_xy[1])) <= 1e-6
    )
    layup_idx = next(
        idx
        for idx, distance in enumerate(court.distance_hex.tolist())
        if idx != basket_idx
        and float(distance) <= 1.0 + 1e-6
        and abs(float(court.xy[idx, 1] - court.rim_xy[1])) > 1e-6
    )
    jumper_idx = next(
        idx
        for idx, distance in enumerate(court.distance_hex.tolist())
        if float(distance) > 1.0 + 1e-6
    )
    records = [
        {"canonical_shot_cell_index": basket_idx, "landing_cell_index": 1, "shot_type": "dunk"},
        {"canonical_shot_cell_index": finger_roll_idx, "landing_cell_index": 2, "shot_type": "finger_roll"},
        {"canonical_shot_cell_index": layup_idx, "landing_cell_index": 3, "shot_type": "layup"},
        {"canonical_shot_cell_index": jumper_idx, "landing_cell_index": 4, "shot_type": "jumper"},
    ]
    transitions = build_transition_arrays_by_shot_type(
        records,
        canonical_shot_indices=[basket_idx, finger_roll_idx, layup_idx, jumper_idx],
        court=court,
        cell_count=len(court.cells),
    )
    assert set(transitions) == {"dunk", "finger_roll", "layup", "jumper"}
    assert transitions["dunk"].row_shot_cell_indices.tolist() == [basket_idx]
    assert transitions["finger_roll"].row_shot_cell_indices.tolist() == [finger_roll_idx]
    assert transitions["layup"].row_shot_cell_indices.tolist() == [layup_idx]
    assert transitions["jumper"].row_shot_cell_indices.tolist() == [jumper_idx]
    assert transitions["dunk"].counts[0, 1] == 1.0
    assert transitions["finger_roll"].counts[0, 2] == 1.0
    assert transitions["layup"].counts[0, 3] == 1.0
    assert transitions["jumper"].counts[0, 4] == 1.0


def test_shot_make_arrays_count_attempts_and_makes() -> None:
    records = [
        {"canonical_shot_cell_index": 2, "made": True},
        {"canonical_shot_cell_index": 2, "made": False},
        {"canonical_shot_cell_index": 3, "made": True},
        {"canonical_shot_cell_index": 7, "made": True},
    ]
    make_arrays = build_shot_make_arrays(records, canonical_shot_indices=[2, 3, 4], cell_count=8)
    assert make_arrays.attempts[2] == 2.0
    assert make_arrays.makes[2] == 1.0
    assert make_arrays.attempts[3] == 1.0
    assert make_arrays.makes[3] == 1.0
    assert make_arrays.attempts[4] == 0.0
    np.testing.assert_allclose(make_arrays.fg_pct[2], 0.5)
    np.testing.assert_allclose(make_arrays.fg_pct[3], 1.0)
    assert np.isnan(make_arrays.fg_pct[4])



def test_transition_arrays_apply_pseudocount_smoothing() -> None:
    records = [
        {"canonical_shot_cell_index": 2, "landing_cell_index": 5},
        {"canonical_shot_cell_index": 2, "landing_cell_index": 5},
    ]
    transitions = build_transition_arrays(
        records,
        canonical_shot_indices=[2, 3],
        cell_count=8,
        pseudocount=0.5,
    )
    assert transitions.counts[0, 5] == 2.5
    assert transitions.counts[0, 0] == 0.5
    assert transitions.counts[1, 0] == 0.5
    np.testing.assert_allclose(transitions.counts[0].sum(), 6.0)
    np.testing.assert_allclose(transitions.counts[1].sum(), 4.0)
    np.testing.assert_allclose(transitions.probs[0].sum(), 1.0)
    np.testing.assert_allclose(transitions.probs[1].sum(), 1.0)
    with pytest.raises(ValueError):
        build_transition_arrays(records, canonical_shot_indices=[2], cell_count=8, pseudocount=-0.1)


def test_fit_filter_excludes_makes_missing_targets_and_behind_backboard() -> None:
    records = [
        {"made": True, "catch_cell_index": 1, "catch_y": 0.0},
        {"made": False, "catch_cell_index": None, "catch_y": 0.0},
        {"made": False, "catch_cell_index": 2, "catch_y": 0.5, "catch_behind_backboard": True},
        {"made": False, "catch_cell_index": 3, "catch_y": 0.0, "catch_behind_backboard": False},
    ]
    kept, rejected = _filter_fit_records(
        records,
        target_cell_field="catch_cell_index",
        filter_behind_backboard=True,
        behind_backboard_y_threshold=0.381,
    )
    assert kept == [records[3]]
    assert rejected == {"made": 1, "missing_target": 1, "behind_backboard": 1}


def test_fit_filter_can_use_landing_y_for_behind_backboard() -> None:
    records = [
        {"made": False, "landing_cell_index": 1, "landing_y": 0.5},
        {"made": False, "landing_cell_index": 2, "landing_y": -1.0},
    ]
    kept, rejected = _filter_fit_records(
        records,
        target_cell_field="landing_cell_index",
        filter_behind_backboard=True,
        behind_backboard_y_threshold=0.381,
    )
    assert kept == [records[1]]
    assert rejected == {"behind_backboard": 1}
