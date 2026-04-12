import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from basketworld.envs.basketworld_env_v2 import HexagonBasketballEnv
from basketworld.utils.wrappers import SetObservationWrapper
from benchmarks.jax_kernel import (
    build_action_masks_batch,
    build_kernel_static_from_env,
    build_observation_vector_batch,
    build_pass_steal_probabilities_batch,
    build_set_observation_batch,
    build_shot_profile_batch,
    snapshot_state_from_env,
    step_batch,
    stack_state_snapshots,
)


def _make_env() -> HexagonBasketballEnv:
    env = HexagonBasketballEnv(
        players=3,
        render_mode=None,
        pass_mode="pointer_targeted",
        allow_dunks=True,
        shot_pressure_enabled=True,
        mask_occupied_moves=False,
        three_point_distance=4.25,
        three_point_short_distance=3.0,
    )
    env.reset(seed=17)
    valid_cells = sorted(
        env._move_mask_by_cell.keys(),
        key=lambda cell: (env._hex_distance(cell, env.basket_position), cell[0], cell[1]),
    )
    assert len(valid_cells) >= env.n_players
    env.positions = [
        valid_cells[1],
        valid_cells[0],
        valid_cells[2],
        valid_cells[3],
        valid_cells[4],
        valid_cells[5],
    ]
    env.ball_holder = env.offense_ids[0]
    env.shot_clock = 11
    env.pressure_exposure = 0.75
    env._offensive_lane_steps = {env.offense_ids[0]: 2, env.offense_ids[1]: 1}
    env._defender_in_key_steps = {env.defense_ids[0]: 3, env.defense_ids[1]: 1}
    env.offense_layup_pct_by_player = [0.64, 0.57, 0.60]
    env.offense_three_pt_pct_by_player = [0.34, 0.39, 0.41]
    env.offense_dunk_pct_by_player = [0.72, 0.56, 0.61]
    return env


def _make_kernel_inputs(env: HexagonBasketballEnv):
    static = build_kernel_static_from_env(env, xp=jnp)
    state = stack_state_snapshots([snapshot_state_from_env(env)], xp=jnp)
    return static, state


class _FakeRng:
    def __init__(self, random_values):
        self._random_values = list(float(v) for v in random_values)

    def random(self):
        if not self._random_values:
            return 0.5
        return float(self._random_values.pop(0))

    def choice(self, items):
        if len(items) == 0:
            raise AssertionError("Unexpected empty RNG.choice() call")
        return items[0]


def test_jax_action_masks_match_env_for_fixed_pointer_targeted_state():
    env = _make_env()
    static, state = _make_kernel_inputs(env)

    jax_masks = np.asarray(build_action_masks_batch(static, state, jnp))[0]
    env_masks = env._get_action_masks()

    assert jax_masks.shape == env_masks.shape
    assert np.array_equal(jax_masks, env_masks)


def test_jax_shot_profiles_match_env_for_fixed_state():
    env = _make_env()
    static, state = _make_kernel_inputs(env)

    profile = build_shot_profile_batch(static, state, jnp)
    jax_prob = np.asarray(profile["probability"])[0]
    jax_ep = np.asarray(profile["expected_points"])[0]

    env_prob = []
    env_ep = []
    for player_id in range(env.n_players):
        pos = env.positions[player_id]
        distance = env._hex_distance(pos, env.basket_position)
        env_prob.append(float(env._calculate_shot_probability(player_id, distance)))
        shot_value = 2.0
        if not (env.allow_dunks and distance == 0) and env._is_three_point_hex(tuple(pos)):
            shot_value = 3.0
        env_ep.append(float(shot_value * env_prob[-1]))

    assert jax_prob.tolist() == pytest.approx(env_prob, abs=1e-6)
    assert jax_ep.tolist() == pytest.approx(env_ep, abs=1e-6)


def test_jax_pass_steal_probabilities_match_env_for_fixed_state():
    env = _make_env()
    static, state = _make_kernel_inputs(env)

    jax_probs = np.asarray(build_pass_steal_probabilities_batch(static, state, jnp))[0]
    env_probs_dict = env.calculate_pass_steal_probabilities(env.ball_holder)
    env_probs = np.zeros(env.players_per_side, dtype=np.float32)
    for idx, offense_id in enumerate(env.offense_ids):
        if offense_id == env.ball_holder:
            continue
        env_probs[idx] = float(env_probs_dict.get(offense_id, 0.0))

    assert jax_probs.tolist() == pytest.approx(env_probs.tolist(), abs=1e-6)


def test_jax_raw_observation_vector_matches_env_for_fixed_state():
    env = _make_env()
    static, state = _make_kernel_inputs(env)

    jax_obs = np.asarray(build_observation_vector_batch(static, state, jnp))[0]
    env_obs = env._get_observation()

    assert jax_obs.shape == env_obs.shape
    assert jax_obs.tolist() == pytest.approx(env_obs.tolist(), abs=1e-6)


def test_jax_set_observation_payload_matches_wrapper_for_fixed_state():
    env = _make_env()
    static, state = _make_kernel_inputs(env)

    jax_obs = {
        key: np.asarray(value)[0]
        for key, value in build_set_observation_batch(static, state, jnp).items()
    }
    wrapped = SetObservationWrapper(env)
    wrapped_obs = wrapped.observation(env._build_observation_dict(observer_is_offense=True))

    assert np.array_equal(jax_obs["action_mask"], np.asarray(wrapped_obs["action_mask"]))
    for key in ("obs", "role_flag", "skills", "players", "globals"):
        wrapped_value = np.asarray(wrapped_obs[key])
        assert jax_obs[key].shape == wrapped_value.shape
        np.testing.assert_allclose(jax_obs[key], wrapped_value, atol=1e-6, rtol=0.0)


def _make_minimal_step_env() -> HexagonBasketballEnv:
    env = HexagonBasketballEnv(
        players=3,
        render_mode=None,
        pass_mode="pointer_targeted",
        allow_dunks=True,
        shot_pressure_enabled=False,
        defender_pressure_turnover_chance=0.0,
        base_steal_rate=0.0,
        enable_phi_shaping=False,
        illegal_defense_enabled=False,
        offensive_three_seconds_enabled=False,
    )
    env.reset(seed=5)
    env.positions = [
        (-2, 4),
        (-1, 4),
        (0, 4),
        (-3, 4),
        (-3, 5),
        (-2, 5),
    ]
    env.ball_holder = env.offense_ids[0]
    env.shot_clock = 12
    env.step_count = 0
    env.episode_ended = False
    env.pressure_exposure = 0.0
    env._cached_phi = 0.0
    env._assist_candidate = None
    return env


def test_jax_step_batch_matches_env_for_deterministic_noop_step():
    env = _make_minimal_step_env()
    static, state = _make_kernel_inputs(env)
    actions = np.array([[0] * env.n_players], dtype=np.int32)
    key = jax.random.PRNGKey(0)

    env._rng = _FakeRng([])
    env_obs, env_rewards, env_done, env_truncated, _ = env.step(actions[0])
    out = step_batch(static, state, jnp.asarray(actions), jnp.asarray([key]), jax, jnp)
    next_state = out.state

    assert env_truncated is False
    assert bool(np.asarray(out.done)[0]) == bool(env_done)
    assert np.asarray(next_state.positions)[0].tolist() == [list(pos) for pos in env.positions]
    assert int(np.asarray(next_state.ball_holder)[0]) == int(env.ball_holder)
    assert int(np.asarray(next_state.shot_clock)[0]) == int(env.shot_clock)
    assert np.asarray(out.rewards)[0].tolist() == pytest.approx(env_rewards.tolist(), abs=1e-6)
    assert np.asarray(build_observation_vector_batch(static, next_state, jnp))[0].tolist() == pytest.approx(
        env_obs["obs"].tolist(), abs=1e-6
    )


def test_jax_step_batch_matches_env_for_pointer_pass_success():
    env = _make_minimal_step_env()
    static, state = _make_kernel_inputs(env)
    actions = np.array([[0] * env.n_players], dtype=np.int32)
    actions[0, env.ball_holder] = 8  # PASS_E -> first teammate slot in pointer_targeted mode
    key = jax.random.PRNGKey(1)

    env._rng = _FakeRng([0.9])
    _, env_rewards, env_done, _, _ = env.step(actions[0])
    out = step_batch(static, state, jnp.asarray(actions), jnp.asarray([key]), jax, jnp)
    next_state = out.state

    assert bool(np.asarray(out.done)[0]) == bool(env_done)
    assert int(np.asarray(next_state.ball_holder)[0]) == int(env.ball_holder)
    assert np.asarray(next_state.positions)[0].tolist() == [list(pos) for pos in env.positions]
    assert int(np.asarray(next_state.assist_active)[0]) == 1
    assert int(np.asarray(next_state.assist_recipient)[0]) == int(env._assist_candidate["recipient_id"])
    assert np.asarray(out.rewards)[0].tolist() == pytest.approx(env_rewards.tolist(), abs=1e-6)


def test_jax_step_batch_matches_env_for_deterministic_shot():
    env = _make_minimal_step_env()
    env.positions[env.ball_holder] = env.basket_position
    key = jax.random.PRNGKey(3)
    shot_draw = float(jax.random.uniform(key))
    env.offense_dunk_pct_by_player[0] = max(0.01, min(0.99, shot_draw + 0.05))
    static, state = _make_kernel_inputs(env)
    actions = np.array([[0] * env.n_players], dtype=np.int32)
    actions[0, env.ball_holder] = 7  # SHOOT

    env._rng = _FakeRng([shot_draw])
    _, env_rewards, env_done, _, _ = env.step(actions[0])
    out = step_batch(static, state, jnp.asarray(actions), jnp.asarray([key]), jax, jnp)
    next_state = out.state

    assert bool(np.asarray(out.done)[0]) == bool(env_done)
    assert int(np.asarray(next_state.ball_holder)[0]) == int(-1 if env.ball_holder is None else env.ball_holder)
    assert np.asarray(out.rewards)[0].tolist() == pytest.approx(env_rewards.tolist(), abs=1e-6)
