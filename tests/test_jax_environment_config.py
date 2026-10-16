from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")

from basketworld.domain import ActionType, Team
from basketworld.envs.basketworld_env_v2 import HexagonBasketballEnv
from basketworld.utils.start_templates import load_start_template_library
from basketworld_jax.config import JaxEnvironmentConfig
from basketworld_jax.env.minimal import (
    build_kernel_static_from_config,
    build_kernel_static_from_env,
)


def _assert_static_equal(legacy_static, config_static) -> None:
    assert legacy_static._fields == config_static._fields
    for name, legacy_value, config_value in zip(
        legacy_static._fields, legacy_static, config_static
    ):
        legacy_array = np.asarray(legacy_value)
        config_array = np.asarray(config_value)
        assert legacy_array.shape == config_array.shape, name
        np.testing.assert_allclose(
            legacy_array,
            config_array,
            atol=0.0,
            rtol=0.0,
            equal_nan=True,
            err_msg=name,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "players": 1,
            "court_rows": 9,
            "court_cols": 8,
        },
        {
            "players": 3,
            "court_rows": 9,
            "court_cols": 8,
            "allow_dunks": True,
            "three_point_distance": 4.25,
            "three_point_short_distance": 3.0,
            "illegal_defense_enabled": False,
            "offensive_three_seconds_enabled": True,
            "pressure_skill_mean": 0.2,
            "pressure_skill_std": 0.4,
            "shot_contest_skill_mean": -0.1,
            "shot_contest_skill_std": 0.3,
            "ball_handling_skill_mean": 0.5,
            "ball_handling_skill_std": 0.2,
            "steal_skill_mean": 0.1,
            "steal_skill_std": 0.6,
            "passing_skill_mean": -0.2,
            "passing_skill_std": 0.7,
        },
    ],
)
def test_jax_owned_static_geometry_matches_legacy_builder(kwargs):
    env = HexagonBasketballEnv(render_mode=None, seed=0, **kwargs)
    config = JaxEnvironmentConfig.from_legacy_env(env)

    _assert_static_equal(
        build_kernel_static_from_env(env, jnp),
        build_kernel_static_from_config(config, jnp),
    )


@pytest.mark.parametrize("mode", ["possessions", "game_clock"])
def test_jax_owned_static_multi_possession_modes_match_legacy_builder(mode):
    env = HexagonBasketballEnv(
        players=2,
        court_rows=16,
        court_cols=16,
        render_mode=None,
        seed=0,
        episode_end_mode=mode,
        game_clock_seconds=17,
        overtime_seconds=9,
    )
    overrides = {
        "enable_multi_possession": True,
        "multi_possession_limit": 4,
        "multi_possession_reward_mode": "scoring_events",
        "game_winner_reward": 2.0,
        "check_clearance": 1,
        "check_setup_steps": 2,
        "check_deadline_steps": 6,
        "random_check_location": True,
    }
    for key, value in overrides.items():
        setattr(env, key, value)

    config = JaxEnvironmentConfig.from_legacy_env(env)
    _assert_static_equal(
        build_kernel_static_from_env(env, jnp),
        build_kernel_static_from_config(config, jnp),
    )


def test_jax_owned_start_template_compilation_matches_legacy_builder():
    path = Path(__file__).resolve().parents[1] / "configs" / "start_templates_v2.json"
    library = load_start_template_library(path, players_per_side=3)
    env = HexagonBasketballEnv(
        players=3,
        court_rows=9,
        court_cols=8,
        shot_clock=24,
        min_shot_clock=14,
        render_mode=None,
        seed=0,
        start_template_enabled=True,
        start_template_library=library,
        start_template_prob=1.0,
        start_template_jitter_scale=0.0,
        start_template_mirror_prob=0.0,
        start_template_strict=True,
    )
    config = JaxEnvironmentConfig.from_legacy_env(env)

    _assert_static_equal(
        build_kernel_static_from_env(env, jnp),
        build_kernel_static_from_config(config, jnp),
    )


def test_environment_config_adapters_share_one_contract(tmp_path):
    args = Namespace(
        grid_size=12,
        court_rows=9,
        court_cols=8,
        players=3,
        shot_clock=24,
        min_shot_clock=14,
        training_team="defense",
        enable_multi_possession=True,
        episode_end_mode="game_clock",
        game_clock_seconds=300,
        overtime_seconds=120,
    )
    from_args = JaxEnvironmentConfig.from_args(args)
    assert from_args.training_team == "defense"
    assert from_args.players == 3
    assert from_args.episode_end_mode == "game_clock"

    template_path = (
        Path(__file__).resolve().parents[1] / "configs" / "start_templates_v2.json"
    )
    template_args = Namespace(
        players=3,
        start_template_enabled=True,
        start_template_library=str(template_path),
    )
    from_template_args = JaxEnvironmentConfig.from_args(template_args)
    assert isinstance(from_template_args.start_template_library, dict)
    assert from_template_args.start_template_library["templates"]

    metadata = {
        "frozen_config": {"players": 2, "court_rows": 10, "court_cols": 7},
        "env_config": {
            "enable_multi_possession": True,
            "episode_end_mode": "game_clock",
            "game_clock_seconds": 90,
        },
    }
    from_checkpoint = JaxEnvironmentConfig.from_checkpoint_metadata(metadata)
    assert from_checkpoint.players == 2
    assert from_checkpoint.court_rows == 10
    assert from_checkpoint.game_clock_seconds == 90

    from_eval = JaxEnvironmentConfig.from_evaluation_request(
        {"players_per_side": 1, "court_height": 8, "court_width": 6},
        {"shot_clock_steps": 18},
    )
    assert (from_eval.players, from_eval.court_rows, from_eval.court_cols) == (1, 8, 6)
    assert from_eval.shot_clock == 18

    from_dev = JaxEnvironmentConfig.from_dev_game_params(
        {"players": 1}, {"offensive_three_seconds": True}
    )
    assert from_dev.offensive_three_seconds_enabled is True

    metadata_path = tmp_path / "env.json"
    metadata_path.write_text(json.dumps(from_checkpoint.as_metadata(), sort_keys=True))
    assert json.loads(metadata_path.read_text())["training_team"] == "offense"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"players": 0}, "players"),
        ({"shot_clock": 0}, "shot_clock"),
        ({"min_shot_clock": 25, "shot_clock": 24}, "min_shot_clock"),
        ({"episode_end_mode": "quarters"}, "episode_end_mode"),
        (
            {"episode_end_mode": "game_clock", "enable_multi_possession": False},
            "enable_multi_possession",
        ),
        ({"check_clearance": -1}, "check_clearance"),
    ],
)
def test_environment_config_validation_names_the_invalid_field(kwargs, message):
    with pytest.raises(ValueError, match=message):
        JaxEnvironmentConfig(**kwargs)


def test_domain_types_keep_legacy_values_and_identity():
    from basketworld.envs.basketworld_env_v2 import ActionType as LegacyActionType
    from basketworld.envs.basketworld_env_v2 import Team as LegacyTeam

    assert LegacyActionType is ActionType
    assert LegacyTeam is Team
    assert [action.value for action in ActionType] == list(range(14))


def test_jax_config_and_static_builder_import_without_legacy_or_sb3():
    project_root = Path(__file__).resolve().parents[1]
    script = r'''
import importlib.abc
import sys

FORBIDDEN = (
    "basketworld.envs.basketworld_env_v2",
    "gymnasium",
    "stable_baselines3",
    "torch",
)

class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + ".") for name in FORBIDDEN):
            raise AssertionError(f"forbidden import: {fullname}")
        return None

sys.meta_path.insert(0, Blocker())
import numpy as np
from basketworld.domain import ActionType, Team
from basketworld_jax.config.environment import JaxEnvironmentConfig
from basketworld_jax.env.minimal import build_kernel_static_from_config

static = build_kernel_static_from_config(
    JaxEnvironmentConfig(players=1, court_rows=9, court_cols=8),
    np,
)
assert static.cell_coords.shape == (72, 2)
assert ActionType.SHOOT.value == 7
assert Team.OFFENSE.value == 0
'''
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
