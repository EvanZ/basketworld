import numpy as np

from basketworld.envs.basketworld_env_v2 import ActionType, HexagonBasketballEnv


def test_pass_steal_probabilities_shape():
    env = HexagonBasketballEnv(players=2, render_mode=None)
    env.reset(seed=0)
    env.ball_holder = env.offense_ids[0]
    env.positions[env.offense_ids[0]] = env.basket_position
    env.positions[env.offense_ids[1]] = (env.basket_position[0] + 1, env.basket_position[1])

    probs = env.calculate_pass_steal_probabilities(env.ball_holder)
    assert set(probs.keys()) == {env.offense_ids[1]}
    val = probs[env.offense_ids[1]]
    assert 0.0 <= val <= 1.0


def test_pointer_targeted_pass_uses_explicit_receiver():
    env = HexagonBasketballEnv(
        players=2,
        render_mode=None,
        pass_mode="pointer_targeted",
        base_steal_rate=0.0,
    )
    env.reset(seed=3)
    passer = env.offense_ids[0]
    receiver = env.offense_ids[1]
    env.ball_holder = passer

    passer_pos = env.positions[passer]
    receiver_pos = env.positions[receiver]
    vec_x, vec_y = env._axial_to_cartesian(
        receiver_pos[0] - passer_pos[0],
        receiver_pos[1] - passer_pos[1],
    )
    direction_idx = 0
    min_dot = float("inf")
    for idx, (dq, dr) in enumerate(env.hex_directions):
        dir_x, dir_y = env._axial_to_cartesian(dq, dr)
        dot = vec_x * dir_x + vec_y * dir_y
        if dot < min_dot:
            min_dot = dot
            direction_idx = idx

    results = {"passes": {}, "turnovers": []}
    env._attempt_pass(
        passer_id=passer,
        direction_idx=direction_idx,
        results=results,
        explicit_target_id=receiver,
    )

    assert env.ball_holder == receiver
    assert passer in results["passes"]
    assert results["passes"][passer]["success"] is True
    assert results["passes"][passer]["target"] == receiver
    assert results["passes"][passer]["intended_target"] == receiver


def test_pointer_targeted_pass_rejects_illegal_receiver():
    env = HexagonBasketballEnv(
        players=2,
        render_mode=None,
        pass_mode="pointer_targeted",
        base_steal_rate=0.0,
    )
    env.reset(seed=5)
    passer = env.offense_ids[0]
    env.ball_holder = passer

    illegal_target = env.defense_ids[0]
    results = {"passes": {}, "turnovers": []}
    env._attempt_pass(
        passer_id=passer,
        direction_idx=0,
        results=results,
        explicit_target_id=illegal_target,
    )

    assert env.ball_holder == passer
    assert passer in results["passes"]
    assert results["passes"][passer]["success"] is False
    assert results["passes"][passer]["reason"] == "illegal_target"


def test_pointer_targeted_pass_slot_mapping_uses_teammate_order():
    env = HexagonBasketballEnv(
        players=3,
        render_mode=None,
        pass_mode="pointer_targeted",
        base_steal_rate=0.0,
    )
    env.reset(seed=9)
    passer = env.offense_ids[0]
    teammates = [pid for pid in env.offense_ids if pid != passer]
    assert teammates == sorted(teammates)
    env.ball_holder = passer

    # Slot 1 corresponds to PASS_NE (index 1) and should target teammates[1].
    actions = np.array([ActionType.NOOP.value] * env.n_players, dtype=int)
    actions[passer] = ActionType.PASS_NE.value

    _, _, _, _, info = env.step(actions)
    action_results = info.get("action_results", {})
    pass_res = action_results.get("passes", {}).get(passer)

    assert pass_res is not None
    assert pass_res.get("success") is True
    assert pass_res.get("intended_target") == teammates[1]
    assert pass_res.get("target") == teammates[1]
    assert env.ball_holder == teammates[1]


def test_pointer_targeted_missing_target_is_rejected():
    env = HexagonBasketballEnv(
        players=2,
        render_mode=None,
        pass_mode="pointer_targeted",
        base_steal_rate=0.0,
    )
    env.reset(seed=11)
    passer = env.offense_ids[0]
    env.ball_holder = passer

    results = {"passes": {}, "turnovers": []}
    env._attempt_pass(
        passer_id=passer,
        direction_idx=0,
        results=results,
        explicit_target_id=None,
    )

    assert env.ball_holder == passer
    assert passer in results["passes"]
    assert results["passes"][passer]["success"] is False
    assert results["passes"][passer]["reason"] == "missing_target"
