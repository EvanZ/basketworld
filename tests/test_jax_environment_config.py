from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path
import subprocess
import sys

import pytest

jnp = pytest.importorskip("jax.numpy")

from basketworld.domain import ActionType, Team
from basketworld.utils.start_templates import load_start_template_library
from basketworld_jax.config import JaxEnvironmentConfig
from basketworld_jax.env.minimal import build_kernel_static_from_config


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
        gamma=0.97,
        reward_shaping_gamma=None,
        phi_beta_start=0.4,
        pass_oob_turnover_prob_start=0.25,
    )
    from_args = JaxEnvironmentConfig.from_args(args)
    assert from_args.training_team == "defense"
    assert from_args.players == 3
    assert from_args.episode_end_mode == "game_clock"
    assert from_args.reward_shaping_gamma == 0.97
    assert from_args.phi_beta == 0.4
    assert from_args.pass_oob_turnover_prob == 0.25

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


def test_domain_types_keep_stable_wire_values():
    assert [action.value for action in ActionType] == list(range(14))
    assert Team.OFFENSE.value == 0
    assert Team.DEFENSE.value == 1


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
