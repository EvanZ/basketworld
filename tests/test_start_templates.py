from pathlib import Path

import numpy as np

from basketworld.envs.basketworld_env_v2 import HexagonBasketballEnv
from basketworld.utils.start_templates import (
    _mirror_anchor,
    load_start_template_library,
    resolve_start_template,
)


def _sample_library_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "start_templates_v1.json"


def _extreme_library_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "start_templates_v2.json"


def test_load_and_resolve_sample_start_template_library():
    library = load_start_template_library(_sample_library_path(), players_per_side=3)
    assert library["version"] == 1
    assert library["players_per_side"] == 3
    assert len(library["templates"]) >= 1
    assert all("slot" not in entry for entry in library["templates"][0]["offense"])
    assert sum(
        1
        for team_name in ("offense", "defense")
        for entry in library["templates"][0][team_name]
        if entry.get("has_ball", False)
    ) == 1

    env = HexagonBasketballEnv(players=3, render_mode=None, allow_dunks=True)
    resolved = resolve_start_template(
        env,
        library["templates"][0],
        jitter_scale=1.0,
        mirror=False,
    )
    assert resolved["template_id"] == library["templates"][0]["id"]
    assert len(resolved["initial_positions"]) == env.n_players
    assert len(set(resolved["initial_positions"])) == env.n_players
    assert resolved["ball_holder"] in env.offense_ids


def test_env_reset_uses_template_when_enabled():
    library = load_start_template_library(_sample_library_path(), players_per_side=3)
    env = HexagonBasketballEnv(
        players=3,
        render_mode=None,
        allow_dunks=True,
        min_shot_clock=14,
        shot_clock=24,
        start_template_enabled=True,
        start_template_library=library,
        start_template_prob=1.0,
        start_template_jitter_scale=1.0,
        start_template_mirror_prob=0.0,
    )
    _, info = env.reset(seed=7)
    assert info["start_template_used"] == 1.0
    assert info["start_template_id"] in {template["id"] for template in library["templates"]}
    assert info["start_template_mirrored"] == 0.0
    assert len(env.positions) == env.n_players
    assert len(set(env.positions)) == env.n_players


def test_explicit_reset_override_bypasses_templates():
    library = load_start_template_library(_sample_library_path(), players_per_side=3)
    env = HexagonBasketballEnv(
        players=3,
        render_mode=None,
        allow_dunks=True,
        start_template_enabled=True,
        start_template_library=library,
        start_template_prob=1.0,
    )
    explicit_positions = [(-1, 2), (1, 2), (2, 3), (-1, 4), (0, 3), (1, 4)]
    _, info = env.reset(
        seed=3,
        options={
            "initial_positions": explicit_positions,
            "ball_holder": 1,
            "shot_clock": 19,
        },
    )
    assert env.positions == explicit_positions
    assert env.ball_holder == 1
    assert env.shot_clock == 19
    assert info["start_template_used"] == 0.0
    assert info["start_template_id"] == ""


def test_template_mirror_flips_sideline_axis_not_basket_axis():
    env = HexagonBasketballEnv(players=3, render_mode=None, allow_dunks=True)

    chosen_anchor = None
    for cell in sorted(env._valid_axial):
        col, row = env._axial_to_offset(int(cell[0]), int(cell[1]))
        mirrored = _mirror_anchor(env, tuple(cell))
        if (
            tuple(mirrored) in env._valid_axial
            and tuple(mirrored) != tuple(cell)
            and env._axial_to_offset(int(mirrored[0]), int(mirrored[1]))[0] == col
        ):
            chosen_anchor = tuple(cell)
            break

    assert chosen_anchor is not None

    original_col, original_row = env._axial_to_offset(
        int(chosen_anchor[0]), int(chosen_anchor[1])
    )
    mirrored = _mirror_anchor(env, chosen_anchor)
    mirrored_col, mirrored_row = env._axial_to_offset(int(mirrored[0]), int(mirrored[1]))

    assert mirrored_col == original_col
    assert mirrored_row == env.court_height - 1 - original_row


def test_resolved_template_mirror_is_exact_reflection_of_concrete_positions():
    library = load_start_template_library(_sample_library_path(), players_per_side=3)
    template = library["templates"][0]

    env_a = HexagonBasketballEnv(players=3, render_mode=None, allow_dunks=True)
    env_b = HexagonBasketballEnv(players=3, render_mode=None, allow_dunks=True)
    env_a._rng = np.random.default_rng(1234)
    env_b._rng = np.random.default_rng(1234)

    base = resolve_start_template(env_a, template, jitter_scale=1.0, mirror=False)
    mirrored = resolve_start_template(env_b, template, jitter_scale=1.0, mirror=True)

    assert mirrored["ball_holder"] == base["ball_holder"]
    assert mirrored["mirrored"] is True
    assert mirrored["initial_positions"] == [
        _mirror_anchor(env_a, tuple(pos)) for pos in base["initial_positions"]
    ]


def test_resolve_start_template_projects_invalid_anchor_and_keeps_positions_in_bounds():
    env = HexagonBasketballEnv(players=3, render_mode=None, allow_dunks=True)
    env._rng = np.random.default_rng(7)
    template = {
        "id": "oob_anchor",
        "mirrorable": True,
        "offense": [
            {"anchor": [999, 999], "jitter_radius": 1, "has_ball": True},
            {"anchor": [2, 4], "jitter_radius": 0},
            {"anchor": [1, 7], "jitter_radius": 1},
        ],
        "defense": [
            {"anchor": [1, 4], "jitter_radius": 0},
            {"anchor": [2, 2], "jitter_radius": 1},
            {"anchor": [0, 6], "jitter_radius": 1},
        ],
    }

    resolved = resolve_start_template(env, template, jitter_scale=1.0, mirror=False)

    assert len(set(resolved["initial_positions"])) == env.n_players
    assert all(tuple(pos) in env._valid_axial for pos in resolved["initial_positions"])


def test_all_v2_templates_resolve_in_bounds_with_jitter_and_mirror():
    library = load_start_template_library(_extreme_library_path(), players_per_side=3)

    for template in library["templates"]:
        for mirror in (False, True):
            for seed in range(16):
                env = HexagonBasketballEnv(players=3, render_mode=None, allow_dunks=True)
                env._rng = np.random.default_rng(seed)
                resolved = resolve_start_template(
                    env,
                    template,
                    jitter_scale=1.0,
                    mirror=mirror,
                )

                assert len(set(resolved["initial_positions"])) == env.n_players
                assert all(
                    tuple(pos) in env._valid_axial
                    for pos in resolved["initial_positions"]
                )
