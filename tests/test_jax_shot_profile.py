import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from basketworld_jax.env.minimal import KernelState, KernelStatic, build_shot_profile_batch


def _hex_distance(a, b):
    q1, r1 = a
    q2, r2 = b
    return (abs(q1 - q2) + abs((q1 + r1) - (q2 + r2)) + abs(r1 - r2)) // 2


def _cartesian_distance_hex_units(a, b):
    dq = float(b[0] - a[0])
    dr = float(b[1] - a[1])
    x = (np.sqrt(3.0) * dq) + ((np.sqrt(3.0) / 2.0) * dr)
    y = 1.5 * dr
    return float(np.sqrt((x * x) + (y * y)) / np.sqrt(3.0))


def _static_for_shot_profile(cell_coords, three_point_by_cell, basket_position):
    values = {field: jnp.asarray(0) for field in KernelStatic._fields}
    values.update(
        cell_coords=jnp.asarray(cell_coords, dtype=jnp.int32),
        three_point_by_cell=jnp.asarray(three_point_by_cell, dtype=jnp.int8),
        basket_position=jnp.asarray(basket_position, dtype=jnp.int32),
        opponent_mask=jnp.asarray([[0, 1], [1, 0]], dtype=jnp.int8),
        allow_dunks=jnp.asarray(0, dtype=jnp.int8),
        shot_pressure_enabled=jnp.asarray(0, dtype=jnp.int8),
        shot_pressure_max=jnp.asarray(0.0, dtype=jnp.float32),
        shot_pressure_lambda=jnp.asarray(0.0, dtype=jnp.float32),
        shot_pressure_cos_threshold=jnp.asarray(1.0, dtype=jnp.float32),
        three_point_distance=jnp.asarray(4.25, dtype=jnp.float32),
        three_pt_extra_hex_decay=jnp.asarray(0.0, dtype=jnp.float32),
    )
    return KernelStatic(**values)


def _state_for_shot_profile(positions):
    batch_size, n_players, _ = positions.shape
    values = {field: jnp.zeros((batch_size,), dtype=jnp.int32) for field in KernelState._fields}
    values.update(
        positions=jnp.asarray(positions, dtype=jnp.int32),
        ball_holder=jnp.zeros((batch_size,), dtype=jnp.int32),
        layup_pct=jnp.full((batch_size, n_players), 0.60, dtype=jnp.float32),
        three_pt_pct=jnp.full((batch_size, n_players), 0.37, dtype=jnp.float32),
        dunk_pct=jnp.full((batch_size, n_players), 0.85, dtype=jnp.float32),
    )
    return KernelState(**values)


def test_jax_corner_three_ep_uses_continuous_distance_not_hex_bucket():
    basket_position = (0, 4)
    same_hex_a = (0, 0)
    same_hex_b = (2, 0)
    dummy_defender = (1, 1)

    assert _hex_distance(same_hex_a, basket_position) == _hex_distance(same_hex_b, basket_position)
    assert _cartesian_distance_hex_units(same_hex_a, basket_position) != pytest.approx(
        _cartesian_distance_hex_units(same_hex_b, basket_position)
    )

    closer, farther = sorted(
        [same_hex_a, same_hex_b],
        key=lambda pos: _cartesian_distance_hex_units(pos, basket_position),
    )
    static = _static_for_shot_profile(
        cell_coords=np.asarray([closer, farther, dummy_defender], dtype=np.int32),
        three_point_by_cell=np.asarray([1, 1, 0], dtype=np.int8),
        basket_position=basket_position,
    )
    state = _state_for_shot_profile(
        np.asarray(
            [
                [closer, dummy_defender],
                [farther, dummy_defender],
            ],
            dtype=np.int32,
        )
    )

    profile = build_shot_profile_batch(static, state, jnp)
    ep = np.asarray(profile["expected_points"])[:, 0]
    probability = np.asarray(profile["probability"])[:, 0]
    shot_value = np.asarray(profile["shot_value"])[:, 0]

    assert shot_value.tolist() == [3.0, 3.0]
    assert probability[0] > probability[1]
    assert ep[0] > ep[1]
