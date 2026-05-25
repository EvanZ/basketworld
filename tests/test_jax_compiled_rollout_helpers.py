from __future__ import annotations

import numpy as np
import pytest

from benchmarks.jax_kernel import (
    KernelState,
    _assemble_full_actions_jax,
    _replace_done_states,
    _sample_uniform_legal_actions_jax,
    parse_args,
    reset_batch_minimal,
    sample_state_batch,
)


def _dummy_state(jnp):
    return KernelState(
        positions=jnp.array(
            [
                [[0, 0], [1, 0]],
                [[2, 0], [3, 0]],
            ],
            dtype=jnp.int32,
        ),
        ball_holder=jnp.array([0, 1], dtype=jnp.int32),
        shot_clock=jnp.array([5, 7], dtype=jnp.int32),
        step_count=jnp.array([1, 2], dtype=jnp.int32),
        episode_ended=jnp.array([0, 1], dtype=jnp.int8),
        pressure_exposure=jnp.array([0.1, 0.2], dtype=jnp.float32),
        offense_lane_steps=jnp.zeros((2, 2), dtype=jnp.float32),
        defense_lane_steps=jnp.zeros((2, 2), dtype=jnp.float32),
        cached_phi=jnp.array([0.0, 0.0], dtype=jnp.float32),
        offense_score=jnp.array([1.0, 2.0], dtype=jnp.float32),
        defense_score=jnp.array([0.0, 1.0], dtype=jnp.float32),
        assist_active=jnp.array([0, 1], dtype=jnp.int8),
        assist_passer=jnp.array([-1, 0], dtype=jnp.int32),
        assist_recipient=jnp.array([-1, 1], dtype=jnp.int32),
        assist_expires_at=jnp.array([-1, 3], dtype=jnp.int32),
        layup_pct=jnp.ones((2, 2), dtype=jnp.float32),
        three_pt_pct=jnp.ones((2, 2), dtype=jnp.float32),
        dunk_pct=jnp.ones((2, 2), dtype=jnp.float32),
    )


def test_sample_uniform_legal_actions_respects_mask_and_noop_fallback():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    mask = jnp.array(
        [
            [[0, 1, 0], [0, 0, 0]],
            [[1, 0, 1], [0, 1, 0]],
        ],
        dtype=jnp.int8,
    )

    actions = _sample_uniform_legal_actions_jax(mask, jax.random.PRNGKey(0), jax, jnp)
    actions_np = np.asarray(actions, dtype=np.int32)

    assert actions_np[0, 0] == 1
    assert actions_np[0, 1] == 0
    assert actions_np[1, 1] == 1
    assert actions_np[1, 0] in (0, 2)


def test_assemble_full_actions_scatter_by_team_ids():
    jnp = pytest.importorskip("jax.numpy")

    training_actions = jnp.array([[1, 2], [3, 4]], dtype=jnp.int32)
    opponent_actions = jnp.array([[5, 6], [7, 8]], dtype=jnp.int32)
    training_ids = jnp.array([0, 2], dtype=jnp.int32)
    opponent_ids = jnp.array([1, 3], dtype=jnp.int32)

    full = _assemble_full_actions_jax(
        training_actions,
        opponent_actions,
        training_ids,
        opponent_ids,
        4,
        jnp,
    )

    np.testing.assert_array_equal(
        np.asarray(full, dtype=np.int32),
        np.array([[1, 5, 2, 6], [3, 7, 4, 8]], dtype=np.int32),
    )


def test_replace_done_states_swaps_only_finished_rows():
    jnp = pytest.importorskip("jax.numpy")

    current = _dummy_state(jnp)
    reset = KernelState(
        *[
            value + 100 if value.dtype.kind in {"i", "u", "f"} else value
            for value in current
        ]
    )
    done = jnp.array([False, True])

    replaced = _replace_done_states(current, reset, done, jnp)

    np.testing.assert_array_equal(np.asarray(replaced.ball_holder), np.array([0, 101]))
    np.testing.assert_array_equal(
        np.asarray(replaced.positions),
        np.array(
            [
                [[0, 0], [1, 0]],
                [[102, 100], [103, 100]],
            ],
            dtype=np.int32,
        ),
    )


def test_reset_batch_minimal_produces_legal_reduced_states():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    args = parse_args(
        [
            "--kernel-batch-size",
            "4",
            "--players",
            "3",
            "--court-rows",
            "9",
            "--court-cols",
            "8",
            "--pass-mode",
            "pointer_targeted",
            "--use-set-obs",
            "true",
            "--enable-phi-shaping",
            "false",
            "--illegal-defense-enabled",
            "false",
            "--offensive-three-seconds",
            "false",
        ]
    )
    static, _ = sample_state_batch(args, xp=jnp)
    keys = jax.random.split(jax.random.PRNGKey(7), 4)
    reset_state = reset_batch_minimal(static, keys, jax, jnp)

    positions = np.asarray(reset_state.positions, dtype=np.int32)
    ball_holders = np.asarray(reset_state.ball_holder, dtype=np.int32)
    shot_clock = np.asarray(reset_state.shot_clock, dtype=np.int32)
    layup_pct = np.asarray(reset_state.layup_pct, dtype=np.float32)
    three_pt_pct = np.asarray(reset_state.three_pt_pct, dtype=np.float32)
    dunk_pct = np.asarray(reset_state.dunk_pct, dtype=np.float32)
    offense_ids = np.asarray(static.offense_ids, dtype=np.int32)
    defense_ids = np.asarray(static.defense_ids, dtype=np.int32)
    valid_cells = {tuple(cell) for cell in np.asarray(static.cell_coords, dtype=np.int32)}
    basket_distance_by_cell = {
        tuple(cell): int(dist)
        for cell, dist in zip(
            np.asarray(static.cell_coords, dtype=np.int32),
            np.asarray(static.basket_distance_by_cell, dtype=np.int32),
        )
    }
    basket = tuple(np.asarray(static.basket_position, dtype=np.int32))
    min_shot_clock = int(np.asarray(static.shot_clock_min))
    max_shot_clock = int(np.asarray(static.shot_clock_max))
    spawn_distance = int(np.ceil(float(np.asarray(static.defense_min_spawn_distance)))) + 1

    np.testing.assert_array_equal(np.asarray(reset_state.step_count), np.zeros((4,), dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(reset_state.episode_ended), np.zeros((4,), dtype=np.int8))
    np.testing.assert_array_equal(np.asarray(reset_state.assist_passer), -np.ones((4,), dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(reset_state.assist_recipient), -np.ones((4,), dtype=np.int32))

    assert np.all((shot_clock >= min_shot_clock) & (shot_clock <= max_shot_clock))
    assert np.all(np.isin(ball_holders, offense_ids))
    assert np.all((layup_pct >= 0.01) & (layup_pct <= 0.99))
    assert np.all((three_pt_pct >= 0.01) & (three_pt_pct <= 0.99))
    assert np.all((dunk_pct >= 0.01) & (dunk_pct <= 0.99))

    for batch_idx in range(positions.shape[0]):
        row_positions = positions[batch_idx]
        assert len({tuple(cell) for cell in row_positions}) == row_positions.shape[0]
        for cell in row_positions:
            assert tuple(cell) in valid_cells
            assert tuple(cell) != basket
        offense_positions = row_positions[offense_ids]
        defense_positions = row_positions[defense_ids]
        for cell in offense_positions:
            assert basket_distance_by_cell[tuple(cell)] >= spawn_distance
        assert defense_positions.shape[0] == defense_ids.shape[0]
