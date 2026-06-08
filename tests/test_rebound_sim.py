from __future__ import annotations

import numpy as np

from analytics.rebound_sim.model import (
    ReboundParams,
    _sample_rebound_winner,
    build_court,
    rebound_region_probabilities,
    simulate_rebounds,
)


def test_rebound_sim_uses_current_jax_court_geometry_defaults():
    court = build_court()

    assert court.spec.rows == 9
    assert court.spec.cols == 8
    assert len(court.cells) == 72
    assert court.basket_axial == court.cells[court.basket_index]
    assert int(np.sum(court.three_point_mask)) > 0


def test_corner_short_rebound_biases_weak_side_more_than_same_side():
    court = build_court()
    rim_y = court.rim_xy[1]
    candidates = [
        idx
        for idx, is_three in enumerate(court.three_point_mask)
        if is_three and abs(float(court.xy[idx, 1] - rim_y)) / court.vertical_extent > 0.85
    ]
    assert candidates
    shot_idx = max(candidates, key=lambda idx: abs(float(court.xy[idx, 1] - rim_y)))

    probs = rebound_region_probabilities(court, shot_idx, "normal")

    assert probs["weak"] > probs["same"]
    assert probs["middle"] > 0.0


def test_rebound_winner_depends_only_on_distance_to_target():
    court = build_court()
    target_idx = court.basket_index
    far_idx = int(np.argmax(court.distance_hex))
    positions = np.full(10, far_idx, dtype=np.int32)
    positions[0] = target_idx
    positions[5] = target_idx
    rng = np.random.default_rng(2026)
    params = ReboundParams(
        target_distance_weight=8.0,
        winner_temperature=0.25,
    )

    winners = np.asarray(
        [
            _sample_rebound_winner(court, rng, positions, target_idx, params=params)[0]
            for _ in range(4000)
        ]
    )
    contested = winners[(winners == 0) | (winners == 5)]

    assert len(contested) == len(winners)
    defense_share = float(np.mean(contested == 5))
    assert 0.44 <= defense_share <= 0.56


def test_rebound_simulation_outputs_valid_rates_and_cell_indices():
    result = simulate_rebounds(1000, seed=123)
    summary = result.summary()

    assert 0.0 < summary["offensive_rebound_rate"] < 1.0
    assert result.target_indices.min() >= 0
    assert result.target_indices.max() < len(result.court.cells)
    assert result.winner_indices.min() >= 0
    assert result.winner_indices.max() < 10


def test_fixed_shot_indices_condition_rebound_simulation():
    court = build_court()
    shot_idx = court.cell_index[(-4, 8)]
    result = simulate_rebounds(50, seed=789, court=court, shot_indices=np.full(50, shot_idx, dtype=np.int32))

    assert set(result.shot_indices.tolist()) == {shot_idx}
    assert result.summary()["shot_zone_counts"]


def test_conditioned_rebound_heatmap_writes_png(tmp_path):
    from analytics.rebound_sim.plotting import plot_conditioned_rebound_heatmap

    court = build_court()
    shot_cell = (-4, 8)
    shot_idx = court.cell_index[shot_cell]
    result = simulate_rebounds(120, seed=321, court=court, shot_indices=np.full(120, shot_idx, dtype=np.int32))
    path = tmp_path / "rebound_heatmap_shot_q-4_r8.png"

    plot_conditioned_rebound_heatmap(result, path, shot_cell=shot_cell)

    assert path.exists()
    assert path.stat().st_size > 0


def test_rebound_region_flow_plot_writes_png(tmp_path):
    from analytics.rebound_sim.plotting import (
        FLOW_REGION_ORDER,
        classify_rebound_flow_regions,
        plot_rebound_region_flow,
    )

    result = simulate_rebounds(250, seed=456)
    regions = classify_rebound_flow_regions(result.court, np.arange(len(result.court.cells)))

    assert set(regions.tolist()) == set(FLOW_REGION_ORDER)

    path = tmp_path / "rebound_sim_region_flow.png"
    plot_rebound_region_flow(result, path, max_flows=12)

    assert path.exists()
    assert path.stat().st_size > 0
