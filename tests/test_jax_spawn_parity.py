from __future__ import annotations

import numpy as np
import pytest

from basketworld.envs.basketworld_env_v2 import HexagonBasketballEnv
from basketworld_jax.env.minimal import (
    _defender_spawn_candidate_mask,
    build_kernel_static_from_env,
)


def test_jax_defender_spawn_broadens_single_strict_candidate_like_python_env():
    jnp = pytest.importorskip("jax.numpy")

    env = HexagonBasketballEnv(
        players=3,
        render_mode=None,
        pass_mode="pointer_targeted",
        court_rows=9,
        court_cols=8,
        spawn_distance=4,
        max_spawn_distance=7,
        defender_spawn_distance=2,
    )
    static = build_kernel_static_from_env(env, xp=jnp)
    cells = np.asarray(static.cell_coords, dtype=np.int32)
    cell_to_index = {tuple(map(int, cell)): idx for idx, cell in enumerate(cells)}

    # This reproduces the historical edge case: the strict defender-distance
    # filter has one legal cell, so Python falls back to the broader closer-mask
    # to avoid deterministic single-cell defender placement.
    offense_cell = (0, 0)
    taken_cells = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)]
    offense_cell_idx = cell_to_index[offense_cell]
    taken_mask = jnp.zeros((cells.shape[0],), dtype=jnp.bool_)
    for cell in taken_cells:
        taken_mask = taken_mask.at[cell_to_index[cell]].set(True)

    dist_to_basket = np.asarray(static.basket_distance_by_cell, dtype=np.float32)
    dist_to_offense = np.asarray(static.cell_distance_matrix[:, offense_cell_idx], dtype=np.float32)
    available = np.asarray(static.non_basket_cell_mask, dtype=np.int8).astype(bool) & ~np.asarray(taken_mask)
    within_max = dist_to_basket <= float(np.asarray(static.max_spawn_distance))
    closer_mask = (
        available
        & within_max
        & (dist_to_basket < dist_to_basket[offense_cell_idx])
        & (dist_to_basket >= float(np.asarray(static.defense_min_spawn_distance)))
    )
    strict_mask = closer_mask & (
        np.abs(dist_to_offense - float(np.asarray(static.defender_spawn_distance))) <= 1.0
    )

    actual = np.asarray(
        _defender_spawn_candidate_mask(
            static,
            jnp.asarray(offense_cell_idx, dtype=jnp.int32),
            taken_mask,
            jnp,
        ),
        dtype=bool,
    )

    assert int(strict_mask.sum()) == 1
    assert int(closer_mask.sum()) > int(strict_mask.sum())
    np.testing.assert_array_equal(actual, closer_mask)
