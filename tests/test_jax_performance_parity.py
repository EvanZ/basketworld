from __future__ import annotations

import hashlib

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from basketworld_jax.env.minimal import (
    GAME_PHASE_AWAITING_CHECK,
    GAME_PHASE_CHECK_SETUP,
    TEAM_A,
    TEAM_B,
    build_action_masks_batch,
    sample_state_batch,
    sample_uniform_legal_actions_jax,
    step_batch_minimal,
)
from basketworld_jax.train.main import parse_args


def _tree_digest(tree) -> str:
    digest = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(tree):
        array = np.ascontiguousarray(np.asarray(leaf))
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def test_shared_check_phase_movement_is_bit_exact_with_preoptimization_reference():
    """Guard the performance refactor with a complete output-tree golden value.

    The reference digest was captured immediately before check setup and
    ordinary check handling began sharing their movement-resolution call.
    The fixture mixes both phases, both stable offensive teams, and seeded
    sampled legal actions. It therefore covers state, action, RNG, transition,
    reward, and diagnostic outputs in one exact comparison on the reference
    CPU backend.
    """
    if jax.default_backend() != "cpu":
        pytest.skip("Golden parity digest is intentionally backend-specific")

    args = parse_args(
        [
            "--enable-multi-possession",
            "--players",
            "5",
            "--court-rows",
            "9",
            "--court-cols",
            "8",
            "--kernel-batch-size",
            "8",
            "--made-basket-restart-mode",
            "check",
            "--check-setup-steps",
            "5",
        ]
    )
    static, state = sample_state_batch(args, jnp)
    row_ids = jnp.arange(8, dtype=jnp.int32)
    phases = jnp.where(
        (row_ids % 2) == 0,
        GAME_PHASE_CHECK_SETUP,
        GAME_PHASE_AWAITING_CHECK,
    ).astype(jnp.int8)
    offense_teams = jnp.where((row_ids % 3) == 0, TEAM_B, TEAM_A).astype(jnp.int8)
    state = state._replace(
        game_phase=phases,
        offense_team=offense_teams,
        ball_holder=jnp.full((8,), -1, dtype=jnp.int32),
        check_team=offense_teams,
        check_setup_steps_remaining=jnp.where(
            phases == GAME_PHASE_CHECK_SETUP,
            3,
            0,
        ).astype(jnp.int32),
        check_steps_remaining=jnp.where(
            phases == GAME_PHASE_AWAITING_CHECK,
            4,
            0,
        ).astype(jnp.int32),
        check_steps_elapsed=jnp.where(
            phases == GAME_PHASE_AWAITING_CHECK,
            1,
            0,
        ).astype(jnp.int32),
    )
    masks = build_action_masks_batch(static, state, jnp)
    actions = sample_uniform_legal_actions_jax(
        masks,
        jax.random.PRNGKey(910),
        jax,
        jnp,
    )
    output = step_batch_minimal(
        static,
        state,
        actions,
        jax.random.split(jax.random.PRNGKey(911), 8),
        jax,
        jnp,
    )
    jax.block_until_ready(output)

    assert _tree_digest(output) == (
        "0f588f230b64a521fc31d799c4ac2713e5bd078a0633f470c124feca29afeff0"
    )
