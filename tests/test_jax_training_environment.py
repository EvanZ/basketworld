from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from basketworld.domain import ActionType
from basketworld_jax.env.minimal import (
    EPISODE_END_GAME_CLOCK,
    EPISODE_END_POSSESSIONS,
    GAME_PHASE_LIVE,
    build_action_masks_batch,
    build_policy_observation_batch,
    sample_state_batch,
    step_batch_minimal,
)
from basketworld_jax.train.main import parse_args, validate_train_args


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _reset_args(mode: str, *, batch_size: int = 3, seed: int = 17):
    argv = [
        "--players",
        "2",
        "--court-rows",
        "9",
        "--court-cols",
        "8",
        "--kernel-batch-size",
        str(batch_size),
        "--sample-reset-seed",
        str(seed),
        "--pass-mode",
        "pointer_targeted",
    ]
    if mode in {"possessions", "game_clock"}:
        argv.append("--enable-multi-possession")
    if mode == "game_clock":
        argv.extend(
            [
                "--episode-end-mode",
                "game_clock",
                "--game-clock-seconds",
                "300",
                "--overtime-seconds",
                "120",
            ]
        )
    return parse_args(argv)


def test_game_clock_training_requires_multi_possession():
    args = parse_args(["--episode-end-mode", "game_clock"])
    with pytest.raises(SystemExit, match="requires --enable-multi-possession"):
        validate_train_args(args)


@pytest.mark.parametrize(
    ("mode", "expected_mode"),
    [
        ("single_possession", EPISODE_END_POSSESSIONS),
        ("possessions", EPISODE_END_POSSESSIONS),
        ("game_clock", EPISODE_END_GAME_CLOCK),
    ],
)
@pytest.mark.parametrize("batch_size", [1, 3])
def test_jax_training_reset_is_seeded_and_deterministic(mode, expected_mode, batch_size):
    args = _reset_args(mode, batch_size=batch_size)
    static_a, state_a = sample_state_batch(args, jnp)
    static_b, state_b = sample_state_batch(args, jnp)

    for first, second in zip(jax.tree_util.tree_leaves(static_a), jax.tree_util.tree_leaves(static_b)):
        np.testing.assert_array_equal(np.asarray(first), np.asarray(second))
    for first, second in zip(jax.tree_util.tree_leaves(state_a), jax.tree_util.tree_leaves(state_b)):
        np.testing.assert_array_equal(np.asarray(first), np.asarray(second))

    assert state_a.positions.shape == (batch_size, 4, 2)
    np.testing.assert_array_equal(
        np.asarray(state_a.episode_end_mode),
        np.full((batch_size,), expected_mode, dtype=np.int8),
    )
    expected_clock = 300 if mode == "game_clock" else 0
    np.testing.assert_array_equal(
        np.asarray(state_a.game_clock_remaining),
        np.full((batch_size,), expected_clock, dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.asarray(state_a.step_count), np.zeros((batch_size,), dtype=np.int32)
    )
    np.testing.assert_array_equal(
        np.asarray(state_a.team_a_score), np.zeros((batch_size,), dtype=np.float32)
    )
    np.testing.assert_array_equal(
        np.asarray(state_a.team_b_score), np.zeros((batch_size,), dtype=np.float32)
    )
    np.testing.assert_array_equal(
        np.asarray(state_a.game_phase),
        np.full((batch_size,), GAME_PHASE_LIVE, dtype=np.int8),
    )
    np.testing.assert_array_equal(
        np.asarray(state_a.inbound_player), -np.ones((batch_size,), dtype=np.int32)
    )
    assert state_a.layup_pct.shape == (batch_size, 4)
    assert state_a.pressure_skill.shape == (batch_size, 4)
    if mode == "single_possession":
        np.testing.assert_array_equal(
            np.asarray(state_a.offense_team), np.zeros((batch_size,), dtype=np.int8)
        )
        shot_clock = np.asarray(state_a.shot_clock)
        assert np.all(shot_clock >= int(np.asarray(static_a.shot_clock_min)))
        assert np.all(shot_clock <= int(np.asarray(static_a.shot_clock_max)))
    else:
        np.testing.assert_array_equal(
            np.asarray(state_a.shot_clock),
            np.full(
                (batch_size,), int(np.asarray(static_a.shot_clock_max)), dtype=np.int32
            ),
        )


def test_single_possession_reset_builds_observations_masks_and_terminal_transition():
    args = _reset_args("single_possession", batch_size=2, seed=31)
    static, state = sample_state_batch(args, jnp)
    observations = build_policy_observation_batch(
        static,
        state,
        jnp,
        model_type="mlp",
    )
    masks = build_action_masks_batch(static, state, jnp)

    assert observations.shape[0] == 2
    assert np.all(np.isfinite(np.asarray(observations)))
    assert masks.shape == (2, 4, len(ActionType))
    holder_masks = np.asarray(masks)[
        np.arange(2), np.asarray(state.ball_holder, dtype=np.int32)
    ]
    assert np.all(holder_masks[:, ActionType.SHOOT.value] == 1)

    actions = jnp.full((2, 4), ActionType.NOOP.value, dtype=jnp.int32)
    actions = actions.at[jnp.arange(2), state.ball_holder].set(ActionType.SHOOT.value)
    output = step_batch_minimal(
        static,
        state,
        actions,
        jax.random.split(jax.random.PRNGKey(32), 2),
        jax,
        jnp,
    )
    np.testing.assert_array_equal(np.asarray(output.done), np.ones((2,), dtype=bool))


def test_jax_training_source_has_no_retired_runtime_imports():
    forbidden = (
        "train.env_factory",
        "basketworld.envs.basketworld_env_v2",
        "stable_baselines3",
    )
    paths = [PROJECT_ROOT / "basketworld_jax" / "env" / "minimal.py"]
    paths.extend(sorted((PROJECT_ROOT / "basketworld_jax" / "train").glob("*.py")))

    for path in paths:
        tree = ast.parse(path.read_text())
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        for module in imported:
            assert not module.startswith(forbidden), (path, module)


def test_jax_training_scaffold_runs_with_sb3_and_legacy_env_unavailable():
    script = r'''
import importlib.abc
import sys

FORBIDDEN = (
    "basketworld.envs.basketworld_env_v2",
    "stable_baselines3",
    "train.env_factory",
)

class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + ".") for name in FORBIDDEN):
            raise AssertionError(f"forbidden import: {fullname}")
        return None

sys.meta_path.insert(0, Blocker())
from basketworld_jax.train.main import parse_args, run_train_scaffold

base_args = [
    "--players", "1",
    "--court-rows", "9",
    "--court-cols", "8",
    "--kernel-batch-size", "1",
    "--rollout-horizon", "2",
    "--warmup-iters", "0",
    "--benchmark-iters", "1",
    "--pass-mode", "pointer_targeted",
    "--no-progress",
]
for mode_args, expected_mode in (
    ([], "possessions"),
    ([
        "--enable-multi-possession",
        "--episode-end-mode", "game_clock",
        "--game-clock-seconds", "300",
        "--overtime-seconds", "120",
    ], "game_clock"),
):
    result = run_train_scaffold(parse_args(base_args + mode_args))
    assert result["status"] == "trajectory_and_update_scaffold"
    assert result["trajectory_spec"]["full_action_shape"] == [1, 2]
    assert result["env_config"]["episode_end_mode"] == expected_mode
'''
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert completed.returncode == 0, completed.stderr
