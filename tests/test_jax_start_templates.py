from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from basketworld.envs.basketworld_env_v2 import HexagonBasketballEnv, Team
from basketworld.utils.start_templates import load_start_template_library
from basketworld_jax.env.minimal import build_kernel_static_from_env, reset_batch_minimal
from basketworld_jax.train.main import (
    _jax_env_config_from_args,
    parse_args,
    validate_train_args,
)


def _template_library_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "start_templates_v2.json"


def test_jax_reset_uses_compiled_start_template_candidates():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    library = load_start_template_library(_template_library_path(), players_per_side=3)
    env = HexagonBasketballEnv(
        players=3,
        render_mode=None,
        training_team=Team.OFFENSE,
        pass_mode="pointer_targeted",
        court_rows=9,
        court_cols=8,
        shot_clock=24,
        min_shot_clock=14,
        start_template_enabled=True,
        start_template_library=library,
        start_template_prob=1.0,
        start_template_jitter_scale=0.0,
        start_template_mirror_prob=0.0,
        start_template_strict=True,
    )
    static = build_kernel_static_from_env(env, xp=jnp)
    state = reset_batch_minimal(
        static,
        jax.random.split(jax.random.PRNGKey(17), 16),
        jax,
        jnp,
    )

    candidate_positions = np.asarray(static.start_template_positions, dtype=np.int32)
    candidate_shot_clocks = set(np.asarray(static.start_template_shot_clocks, dtype=np.int32).tolist())
    candidate_position_sets = {
        tuple(sorted(tuple(cell) for cell in candidate.tolist()))
        for candidate in candidate_positions
    }
    reset_positions = np.asarray(state.positions, dtype=np.int32)

    assert int(np.asarray(static.start_template_enabled)) == 1
    assert candidate_positions.shape[0] == len(library["templates"])
    assert np.asarray(static.start_template_entry_jitter_radii).shape == (
        len(library["templates"]),
        env.n_players,
    )
    assert np.asarray(static.start_template_prob) == pytest.approx(1.0)
    assert np.all(np.isin(np.asarray(state.ball_holder, dtype=np.int32), list(env.offense_ids)))
    assert np.all(np.isin(np.asarray(state.shot_clock, dtype=np.int32), list(candidate_shot_clocks)))
    for row in reset_positions:
        assert tuple(sorted(tuple(cell) for cell in row.tolist())) in candidate_position_sets


def test_jax_start_templates_resample_assignment_and_jitter_each_reset():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    library = load_start_template_library(_template_library_path(), players_per_side=3)
    env = HexagonBasketballEnv(
        players=3,
        render_mode=None,
        training_team=Team.OFFENSE,
        pass_mode="pointer_targeted",
        court_rows=9,
        court_cols=8,
        shot_clock=24,
        min_shot_clock=14,
        start_template_enabled=True,
        start_template_library=library,
        start_template_prob=1.0,
        start_template_jitter_scale=1.0,
        start_template_mirror_prob=0.5,
        start_template_strict=True,
    )
    static = build_kernel_static_from_env(env, xp=jnp)
    state = reset_batch_minimal(
        static,
        jax.random.split(jax.random.PRNGKey(23), 64),
        jax,
        jnp,
    )

    unique_position_rows = {
        tuple(tuple(cell) for cell in row.tolist())
        for row in np.asarray(state.positions, dtype=np.int32)
    }

    assert int(np.asarray(static.start_template_enabled)) == 1
    assert np.max(np.asarray(static.start_template_entry_jitter_radii)) > 0
    assert len(unique_position_rows) > len(library["templates"])


def test_jax_train_config_accepts_and_logs_start_template_overrides():
    path = str(_template_library_path())
    args = parse_args(
        [
            "--start-template-enabled",
            "true",
            "--start-template-library",
            path,
            "--start-template-prob",
            "1.0",
            "--start-template-jitter-scale",
            "0.0",
            "--start-template-mirror-prob",
            "0.0",
            "--start-template-strict",
            "true",
        ]
    )

    validate_train_args(args)
    env_config = _jax_env_config_from_args(args)

    assert env_config["start_template_enabled"] is True
    assert env_config["start_template_library"] == path
    assert env_config["start_template_prob"] == pytest.approx(1.0)
    assert env_config["start_template_jitter_scale"] == pytest.approx(0.0)
    assert env_config["start_template_mirror_prob"] == pytest.approx(0.0)
    assert env_config["start_template_strict"] is True
