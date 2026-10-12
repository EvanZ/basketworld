from types import SimpleNamespace

import numpy as np
import pytest

from basketworld.envs.basketworld_env_v2 import HexagonBasketballEnv
from basketworld.envs.core.shooting import compute_shot_pressure_multiplier

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from basketworld_jax.env.minimal import KernelState, KernelStatic, build_shot_profile_batch
from basketworld_jax.inference.policy import _adapt_observation_to_checkpoint
from basketworld_jax.models import ActorCriticSpec


POSITIONS = np.asarray([[(0, 0), (0, 1), (0, 2), (2, 0)]], dtype=np.int32)


def _axial_to_cartesian(q, r):
    return np.sqrt(3.0) * (q + (r / 2.0)), 1.5 * r


def _hex_distance(first, second):
    dq = int(first[0]) - int(second[0])
    dr = int(first[1]) - int(second[1])
    return (abs(dq) + abs(dr) + abs(dq + dr)) // 2


def _python_env(*, skills, weight):
    return SimpleNamespace(
        shot_pressure_enabled=True,
        basket_position=(0, 4),
        offense_ids=[0],
        defense_ids=[1, 2, 3],
        positions=[tuple(value) for value in POSITIONS[0]],
        shot_pressure_arc_rad=np.deg2rad(60.0),
        shot_pressure_max=0.5,
        shot_pressure_lambda=1.0,
        shot_contest_skill_effect_weight=float(weight),
        shot_contest_skill_by_player=list(skills),
        _axial_to_cartesian=_axial_to_cartesian,
        _hex_distance=_hex_distance,
    )


def _jax_profile(*, skills, weight):
    static_values = {field: jnp.asarray(0) for field in KernelStatic._fields}
    opponent_mask = np.ones((4, 4), dtype=np.int8) - np.eye(4, dtype=np.int8)
    static_values.update(
        cell_coords=jnp.asarray(POSITIONS[0], dtype=jnp.int32),
        three_point_by_cell=jnp.zeros((4,), dtype=jnp.int8),
        basket_position=jnp.asarray((0, 4), dtype=jnp.int32),
        opponent_mask=jnp.asarray(opponent_mask, dtype=jnp.int8),
        allow_dunks=jnp.asarray(0, dtype=jnp.int8),
        shot_pressure_enabled=jnp.asarray(1, dtype=jnp.int8),
        shot_pressure_max=jnp.asarray(0.5, dtype=jnp.float32),
        shot_pressure_lambda=jnp.asarray(1.0, dtype=jnp.float32),
        shot_pressure_cos_threshold=jnp.asarray(
            np.cos(np.deg2rad(30.0)), dtype=jnp.float32
        ),
        shot_contest_skill_effect_weight=jnp.asarray(weight, dtype=jnp.float32),
        three_point_distance=jnp.asarray(4.25, dtype=jnp.float32),
        three_pt_extra_hex_decay=jnp.asarray(0.0, dtype=jnp.float32),
    )
    state_values = {
        field: jnp.zeros((1,), dtype=jnp.int32) for field in KernelState._fields
    }
    state_values.update(
        positions=jnp.asarray(POSITIONS, dtype=jnp.int32),
        ball_holder=jnp.asarray([0], dtype=jnp.int32),
        layup_pct=jnp.full((1, 4), 0.6, dtype=jnp.float32),
        three_pt_pct=jnp.full((1, 4), 0.35, dtype=jnp.float32),
        dunk_pct=jnp.full((1, 4), 0.9, dtype=jnp.float32),
        shot_contest_skill=jnp.asarray([skills], dtype=jnp.float32),
    )
    return build_shot_profile_batch(
        KernelStatic(**static_values), KernelState(**state_values), jnp
    )


def _python_multiplier(*, skills, weight):
    env = _python_env(skills=skills, weight=weight)
    return compute_shot_pressure_multiplier(env, 0, (0, 0), 4)


def test_shot_contest_skill_sampling_is_seeded_independent_and_episode_stable():
    env = HexagonBasketballEnv(
        players=3,
        render_mode=None,
        shot_contest_skill_mean=0.25,
        shot_contest_skill_std=0.8,
    )
    env.reset(seed=17)
    sampled = np.asarray(env.shot_contest_skill_by_player)

    assert sampled.shape == (6,)
    assert float(np.std(sampled)) > 0.0
    env._compute_shot_pressure_multiplier(
        env.offense_ids[0], env.positions[env.offense_ids[0]], 4
    )
    np.testing.assert_array_equal(env.shot_contest_skill_by_player, sampled)

    env.reset(seed=17)
    np.testing.assert_array_equal(env.shot_contest_skill_by_player, sampled)

    pinned = np.linspace(-0.5, 0.5, env.n_players)
    env.reset(seed=99, options={"shot_contest_skills": pinned})
    np.testing.assert_allclose(env.shot_contest_skill_by_player, pinned)


def test_contest_skill_changes_only_eligible_defenders_and_uses_strongest_effective_contest():
    neutral = [0.0, 0.0, 0.0, 0.0]
    near_high = [0.0, 2.0, -2.0, 100.0]
    far_high = [0.0, -2.0, 4.0, 100.0]

    neutral_multiplier = _python_multiplier(skills=neutral, weight=1.0)
    near_multiplier = _python_multiplier(skills=near_high, weight=1.0)
    far_multiplier = _python_multiplier(skills=far_high, weight=1.0)
    without_ineligible_extreme = _python_multiplier(
        skills=[0.0, -2.0, 4.0, -100.0], weight=1.0
    )

    assert near_multiplier < neutral_multiplier
    assert far_multiplier < neutral_multiplier
    assert far_multiplier == pytest.approx(without_ineligible_extreme, abs=1e-12)


@pytest.mark.parametrize(
    "skills,weight",
    [
        ([0.0, 1.5, -0.5, 50.0], 1.25),
        ([0.0, -2.0, 4.0, -50.0], 0.75),
        ([100.0, -100.0, 100.0, -100.0], 0.0),
    ],
)
def test_python_and_jax_contest_pressure_are_equivalent(skills, weight):
    python_multiplier = _python_multiplier(skills=skills, weight=weight)
    jax_multiplier = float(
        np.asarray(_jax_profile(skills=skills, weight=weight)["pressure_multiplier"])[
            0, 0
        ]
    )

    assert jax_multiplier == pytest.approx(python_multiplier, abs=2e-6)


def test_zero_contest_weight_preserves_legacy_pressure_exactly():
    neutral = _python_multiplier(skills=[0.0] * 4, weight=0.0)
    extreme = _python_multiplier(
        skills=[1000.0, -1000.0, 1000.0, -1000.0], weight=0.0
    )
    assert extreme == neutral

    neutral_jax = np.asarray(
        _jax_profile(skills=[0.0] * 4, weight=0.0)["pressure_multiplier"]
    )
    extreme_jax = np.asarray(
        _jax_profile(
            skills=[1000.0, -1000.0, 1000.0, -1000.0], weight=0.0
        )["pressure_multiplier"]
    )
    np.testing.assert_array_equal(extreme_jax, neutral_jax)


@pytest.mark.parametrize(
    "schema_version,checkpoint_dim,expected_prefix",
    [(2, 27, 23), (3, 35, 31)],
)
def test_inference_adapter_preserves_older_checkpoint_feature_order(
    schema_version, checkpoint_dim, expected_prefix
):
    spec = ActorCriticSpec(
        flat_obs_dim=checkpoint_dim,
        training_player_count=1,
        action_dim_per_player=14,
        total_action_dim=14,
        hidden_dims=(16,),
        multi_possession_features=True,
        observation_schema_version=schema_version,
    )
    static = SimpleNamespace(
        role_encoding=jnp.zeros((2,), dtype=jnp.float32),
        offense_ids=jnp.zeros((1,), dtype=jnp.int32),
    )
    current = jnp.arange(37, dtype=jnp.float32)[None, :]

    adapted = np.asarray(
        _adapt_observation_to_checkpoint(current, static, spec, jnp)
    )

    np.testing.assert_allclose(
        adapted,
        np.concatenate(
            [np.arange(expected_prefix), np.arange(33, 37)]
        )[None, :],
    )
