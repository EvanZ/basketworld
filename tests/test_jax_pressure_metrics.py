import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from basketworld_jax.train.pressure_metrics import (
    build_pressure_diagnostics,
    summarize_pressure_diagnostics,
)
from basketworld_jax.env.minimal import (
    GAME_PHASE_LIVE,
    build_shot_profile_batch,
    reset_batch_minimal,
)
from basketworld_jax.train.runtime import _terminal_learner_win_score
from tests.test_jax_multi_possession import _multi_possession_static


def fixture():
    static = _multi_possession_static(players=2)
    state = reset_batch_minimal(
        static, jax.random.split(jax.random.PRNGKey(80), 4), jax, jnp
    )
    return static, state._replace(
        game_phase=jnp.full((4,), GAME_PHASE_LIVE, dtype=jnp.int8),
        offense_team=jnp.array([0, 1, 0, 1], dtype=jnp.int8),
        ball_holder=jnp.array([0, 2, -1, 2], dtype=jnp.int32),
        episode_ended=jnp.array([0, 0, 0, 1], dtype=jnp.int8),
    )


def test_pressure_role_split_masks_and_zero_contest():
    static, state = fixture()
    values = build_pressure_diagnostics(static, state, jnp.ones(4), jnp)
    summary = summarize_pressure_diagnostics(values)
    attack, defend = "pressure/learner_attacking/", "pressure/learner_defending/"
    assert summary[attack + "live_samples"] == 2
    assert summary[attack + "ball_handler_samples"] == 1
    assert summary[defend + "live_samples"] == 1
    assert summary[defend + "ball_handler_samples"] == 1
    assert summary[attack + "mean_ball_handler_esv_reduction"] == 0
    assert summary[defend + "mean_ball_handler_esv_reduction"] == 0
    empty = summarize_pressure_diagnostics(values * 0)
    assert empty[attack + "live_samples"] == 0
    assert attack + "mean_ball_handler_defended_esv" not in empty
    swapped = static._replace(training_player_mask=1 - static.training_player_mask)
    np.testing.assert_allclose(
        build_pressure_diagnostics(swapped, state, jnp.ones(4), jnp),
        np.asarray(values)[:, ::-1],
    )


@pytest.mark.parametrize("pressure_max", [0.0, 0.5, 2.0])
@pytest.mark.parametrize("skill_weight", [0.0, 1.0])
@pytest.mark.parametrize("three_point", [False, True])
def test_esv_matches_actual_shot_profile_with_clipping_and_contest(
    pressure_max, skill_weight, three_point
):
    static, state = fixture()
    static = static._replace(
        shot_pressure_enabled=jnp.asarray(1, jnp.int8),
        shot_pressure_max=jnp.asarray(pressure_max),
        shot_contest_skill_effect_weight=jnp.asarray(skill_weight),
        three_point_by_cell=jnp.full_like(static.three_point_by_cell, int(three_point)),
    )
    # A defender directly between the holder and basket; another behind.
    basket = np.asarray(static.basket_position)
    positions = np.stack(
        [basket + [0, 3], basket + [1, 2], basket + [0, 2], basket + [0, 4]]
    )
    state = state._replace(
        positions=jnp.broadcast_to(jnp.asarray(positions), state.positions.shape),
        shot_contest_skill=jnp.ones_like(state.shot_contest_skill),
    )
    profile = build_shot_profile_batch(static, state, jnp)
    assert float(profile["shot_value"][0, 0]) == (3 if three_point else 2)
    values = build_pressure_diagnostics(static, state, jnp.ones(4), jnp)
    summary = summarize_pressure_diagnostics(values)
    prefix = "pressure/learner_attacking/mean_ball_handler_"
    raw = float(profile["shot_value"][0, 0] * profile["base_probability"][0, 0])
    defended = float(profile["expected_points"][0, 0])
    assert summary[prefix + "uncontested_esv"] == pytest.approx(raw)
    assert summary[prefix + "defended_esv"] == pytest.approx(defended)
    assert summary[prefix + "esv_reduction"] == pytest.approx(raw - defended)
    if pressure_max > 0:
        assert summary[prefix + "esv_reduction"] > 0
    if pressure_max == 2.0:
        assert float(profile["probability"][0, 0]) == pytest.approx(0.01)
    np.testing.assert_allclose(
        jax.jit(build_pressure_diagnostics, static_argnums=(3,))(
            static, state, jnp.ones(4), jnp
        ),
        values,
        rtol=1e-5,
        atol=1e-6,
    )


def test_terminal_win_score_uses_stable_team_not_possession_or_reward():
    static, state = fixture()
    state = state._replace(
        team_a_score=jnp.array([3, 1, 2, 1]), team_b_score=jnp.array([1, 3, 2, 1])
    )
    np.testing.assert_array_equal(
        _terminal_learner_win_score(static, state, jnp), [1, 0, 0.5, 0.5]
    )
    swapped = static._replace(training_player_mask=1 - static.training_player_mask)
    np.testing.assert_array_equal(
        _terminal_learner_win_score(swapped, state, jnp), [0, 1, 0.5, 0.5]
    )


def test_sampling_uses_next_mode_and_preserves_zero_probability():
    from basketworld_jax.train.runtime import sample_pinned_opponent_assignment

    modes = jnp.array([False, True] * 16)
    probs = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    sampled = sample_pinned_opponent_assignment(
        probs, modes, jax.random.PRNGKey(88), jax, jnp
    )
    np.testing.assert_array_equal(sampled, modes.astype(jnp.int32))
    legacy = sample_pinned_opponent_assignment(
        probs[0], modes, jax.random.PRNGKey(88), jax, jnp
    )
    np.testing.assert_array_equal(legacy, np.zeros(32))


def test_offcourt_offensive_player_is_excluded_from_team_mean():
    static, state = fixture()
    state = state._replace(
        positions=state.positions.at[:, 1, :].set(jnp.array([100, 100]))
    )
    summary = summarize_pressure_diagnostics(
        build_pressure_diagnostics(static, state, jnp.ones(4), jnp)
    )
    assert summary[
        "pressure/learner_attacking/mean_offense_defended_esv"
    ] == pytest.approx(
        float(
            build_shot_profile_batch(static, state, jnp)["expected_points"][
                [0, 2], 0
            ].mean()
        )
    )
