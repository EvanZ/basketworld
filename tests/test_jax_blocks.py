"""Compiled block attribution and ordinary-rebound parity for both teams."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp
from basketworld.domain import ActionType
from basketworld_jax.config import JaxEnvironmentConfig
from basketworld_jax.env.minimal import (
    build_kernel_static_from_config,
    reset_batch_minimal,
    step_batch_minimal,
    step_batch_game_clock,
    GAME_PHASE_LIVE,
)
from basketworld_jax.shot_defense_metrics import (
    build_shot_defense_diagnostics,
    summarize_shot_defense_diagnostics,
)


_timed_step = jax.jit(lambda st, s, a, k: step_batch_game_clock(st, s, a, k, jax, jnp))


@pytest.fixture(scope="module")
def fixture():
    static = build_kernel_static_from_config(
        JaxEnvironmentConfig(
            players=2,
            grid_size=7,
            three_point_distance=2.0,
            enable_multi_possession=True,
            multi_possession_limit=6,
            rim_protection_enabled=True,
            contested_shooting_enabled=True,
            rim_protection_block_base=1.0,
            rim_protection_contest_base=1.0,
            illegal_defense_enabled=False,
            defender_pressure_turnover_chance=0.0,
            enable_phi_shaping=False,
        ),
        jnp,
    )
    # Use the ordinary fallback rebound table, keeping this test independent of
    # external fitted rebound-model artifacts.
    static = static._replace(enable_rebounds=jnp.asarray(1, jnp.int8))
    reset = jax.jit(lambda st, keys: reset_batch_minimal(st, keys, jax, jnp))
    state = reset(static, jax.random.split(jax.random.PRNGKey(840), 16))
    step = jax.jit(lambda st, s, a, keys: step_batch_minimal(st, s, a, keys, jax, jnp))
    return static, state, step


@pytest.mark.parametrize("team", [0, 1])
@pytest.mark.parametrize("timed", [False, True])
def test_blocks_credit_one_defender_and_use_normal_rebounds(fixture, team, timed):
    static, initial, step = fixture
    if timed:
        static = static._replace(episode_end_mode=jnp.asarray(1, jnp.int8))
        initial = initial._replace(
            episode_end_mode=jnp.ones((16,), jnp.int8),
            game_clock_remaining=jnp.full((16,), 120, jnp.int32),
        )
        step = _timed_step
    offense = np.asarray(static.offense_ids if team == 0 else static.defense_ids)
    defense = np.asarray(static.defense_ids if team == 0 else static.offense_ids)
    basket = np.asarray(static.basket_position)
    cells = np.asarray(static.cell_coords)
    adjacent = [
        c
        for c in cells
        if max(abs(c[0] - basket[0]), abs(c[1] - basket[1]), abs((c - basket).sum()))
        == 1
    ]
    positions = np.zeros((4, 2), np.int32)
    positions[offense[0]] = basket
    positions[offense[1]] = adjacent[-1]
    positions[defense] = adjacent[:2]
    state = initial._replace(
        positions=jnp.broadcast_to(positions, (16, 4, 2)),
        ball_holder=jnp.full((16,), offense[0], jnp.int32),
        offense_team=jnp.full((16,), team, jnp.int8),
        game_phase=jnp.full((16,), GAME_PHASE_LIVE, jnp.int8),
        clearance_achieved=jnp.ones((16,), jnp.int8),
        shot_clock=jnp.full((16,), 12, jnp.int32),
        rim_protection_skill=jnp.zeros((16, 4)),
    )
    actions = (
        jnp.full((16, 4), ActionType.NOOP.value, jnp.int32)
        .at[:, offense[0]]
        .set(ActionType.SHOOT.value)
    )
    keys = jax.random.split(jax.random.PRNGKey(841), 16)
    blocked = step(static, state, actions, keys)
    control = step(
        static._replace(rim_protection_block_base=jnp.asarray(0.0, jnp.float32)),
        state,
        actions,
        keys,
    )
    np.testing.assert_array_equal(blocked.shot_blocked, 1)
    np.testing.assert_array_equal(blocked.shot_success, 0)
    assert set(np.asarray(blocked.shot_blocker)) == set(defense)
    np.testing.assert_array_equal(blocked.turnover, 0)
    np.testing.assert_array_equal(blocked.steal_player, -1)
    np.testing.assert_array_equal(blocked.assist, 0)
    misses = np.asarray(control.shot_success) == 0
    assert misses.sum() >= 12
    for field in (
        "rebound_target_cell",
        "rebound_winner",
        "offensive_rebound",
        "defensive_rebound",
    ):
        np.testing.assert_array_equal(
            np.asarray(getattr(blocked, field))[misses],
            np.asarray(getattr(control, field))[misses],
        )
    metrics = summarize_shot_defense_diagnostics(
        build_shot_defense_diagnostics(static, blocked, jnp.ones((16,), bool), jnp)
    )
    assert metrics["blocks"] == 16
    assert metrics["all_shot_defense_block_rate_per_eligible_attempt"] == 1
    assert (
        metrics["all_shot_defense_blocked_offensive_recoveries"]
        + metrics["all_shot_defense_blocked_defensive_recoveries"]
        == 16
    )


@pytest.mark.parametrize(
    "rim,shooting", [(False, False), (True, False), (False, True), (True, True)]
)
def test_attention_observation_contract_and_inactive_skill_mask(fixture, rim, shooting):
    from basketworld_jax.env.minimal import (
        build_token_observation_components_batch,
        token_observation_dims,
        token_observation_labels,
    )

    static, state, _ = fixture
    static = static._replace(
        rim_protection_enabled=jnp.asarray(rim, jnp.int8),
        contested_shooting_enabled=jnp.asarray(shooting, jnp.int8),
    )
    features = dict(
        multi_possession_features=True,
        check_clearance_features=True,
        team_clearance_features=True,
        shooting_skill_features=True,
    )
    players, globals_, role = build_token_observation_components_batch(
        static, state, 1.0, jnp, **features
    )
    dimensions = token_observation_dims(**features)
    labels = token_observation_labels(**features)
    assert (players.shape[-1], globals_.shape[-1]) == dimensions
    assert tuple(map(len, labels)) == dimensions
    for field, enabled in [
        ("rim_protection_skill", rim),
        ("contested_shooting_skill", shooting),
    ]:
        index = labels[0].index(field)
        np.testing.assert_array_equal(
            players[:, :, index],
            getattr(state, field) if enabled else np.zeros_like(getattr(state, field)),
        )


def test_strongest_eligible_defender_wins_without_compounding(fixture):
    from basketworld_jax.env.minimal import build_shot_profile_batch
    from basketworld_jax.shot_defense import rim_skill_from_block_sample

    static, state, _ = fixture
    static = static._replace(
        rim_protection_block_base=jnp.asarray(0.02),
        rim_protection_contest_base=jnp.asarray(0.25),
        allow_dunks=jnp.asarray(0, jnp.int8),
    )
    basket = np.asarray(static.basket_position)
    cells = np.asarray(static.cell_coords)
    adjacent = [
        c
        for c in cells
        if max(abs(c[0] - basket[0]), abs(c[1] - basket[1]), abs((c - basket).sum()))
        == 1
    ]
    positions = np.asarray(state.positions).copy()
    positions[:, 0] = basket
    positions[:, 2] = adjacent[0]
    positions[:, 3] = adjacent[1]
    skills = np.zeros((16, 4), np.float32)
    skills[:, 1] = 100.0
    skills[:, 3] = rim_skill_from_block_sample(0.08, 0.02, np)
    state = state._replace(
        positions=jnp.asarray(positions),
        rim_protection_skill=jnp.asarray(skills),
        layup_pct=jnp.full((16, 4), 0.8),
    )
    result = build_shot_profile_batch(static, state, jnp)
    np.testing.assert_allclose(result["block_probability"][:, 0], 0.08, rtol=1e-5)
    np.testing.assert_array_equal(
        np.asarray(result["blocker_candidates"])[:, 0],
        np.broadcast_to([False, False, False, True], (16, 4)),
    )
    # Moving the strongest defender away restores the weaker adjacent defender's chance.
    far = max(cells, key=lambda c: np.linalg.norm(c - basket))
    positions[:, 3] = far
    result = build_shot_profile_batch(
        static, state._replace(positions=jnp.asarray(positions)), jnp
    )
    np.testing.assert_allclose(result["block_probability"][:, 0], 0.02, rtol=1e-5)
