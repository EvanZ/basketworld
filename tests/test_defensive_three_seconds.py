import numpy as np

from basketworld.envs.basketworld_env_v2 import ActionType, HexagonBasketballEnv


def _lane_steps_start_index(env: HexagonBasketballEnv) -> int:
    n_players = int(env.n_players)
    n_offense = len(env.offense_ids)
    n_defense = len(env.defense_ids)
    all_pairs = n_offense * n_defense
    teammate_distance_pairs = (
        (n_offense * (n_offense - 1)) // 2
        + (n_defense * (n_defense - 1)) // 2
    )
    teammate_angle_pairs = (
        n_offense * (n_offense - 1)
        + n_defense * (n_defense - 1)
    )
    return (
        (n_players * 2)  # player positions
        + n_players  # ball holder one-hot
        + 1  # shot clock
        + 1  # pressure exposure
        + n_players  # team encoding
        + 2  # ball handler position
        + (2 if env.include_hoop_vector else 0)
        + all_pairs  # offense-defense distances
        + all_pairs  # offense-defense angles
        + teammate_distance_pairs
        + teammate_angle_pairs
    )


def _find_non_lane_offense_position(env: HexagonBasketballEnv, defender_lane_pos: tuple[int, int]) -> tuple[int, int]:
    for q in range(-env.grid_size, env.grid_size + 1):
        for r in range(-env.grid_size, env.grid_size + 1):
            pos = (q, r)
            if not env._is_valid_position(*pos):
                continue
            if pos in env.offensive_lane_hexes:
                continue
            if env._hex_distance(pos, defender_lane_pos) <= env.defender_guard_distance:
                continue
            return pos
    raise AssertionError("Failed to find a valid non-lane offense position for defensive-lane test")


def test_defensive_three_seconds_counter_updates_and_violation_fires():
    env = HexagonBasketballEnv(
        players=1,
        render_mode=None,
        illegal_defense_enabled=True,
        offensive_three_seconds_enabled=False,
        three_second_max_steps=3,
    )
    env.reset(seed=0)

    offense_id = env.offense_ids[0]
    defender_id = env.defense_ids[0]
    defender_lane_pos = sorted(env.defensive_lane_hexes)[0]
    offense_pos = _find_non_lane_offense_position(env, defender_lane_pos)

    env.positions[defender_id] = defender_lane_pos
    env.positions[offense_id] = offense_pos
    env.ball_holder = offense_id

    lane_start = _lane_steps_start_index(env)
    noop_actions = np.array([ActionType.NOOP.value] * env.n_players)

    for expected_steps in (1, 2, 3):
        obs, _, _, _, info = env.step(noop_actions)
        assert env._defender_in_key_steps[defender_id] == expected_steps
        assert float(obs["obs"][lane_start + defender_id]) == float(expected_steps)
        assert not info.get("action_results", {}).get("defensive_lane_violations", [])

    obs, _, _, _, info = env.step(noop_actions)
    violations = info.get("action_results", {}).get("defensive_lane_violations", [])
    assert violations
    assert int(violations[0]["player_id"]) == defender_id
    assert int(violations[0]["steps"]) == int(env.three_second_max_steps) + 1
    assert env._defender_in_key_steps[defender_id] == 0
    assert float(obs["obs"][lane_start + defender_id]) == 0.0
