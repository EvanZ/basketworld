"""Translate resolved kernel events without inferring steals from restarts."""

from types import SimpleNamespace

import numpy as np
import pytest

from app.backend.jax_dev_runtime import JaxDevRuntime
from basketworld_jax.env.minimal import (
    GAME_PHASE_AWAITING_INBOUND,
    GAME_PHASE_LIVE,
    TURNOVER_REASON_INTERCEPTED,
    TURNOVER_REASON_NONE,
    TURNOVER_REASON_PASS_OUT_OF_BOUNDS,
)


def _events(*, turnover, reason, stealer, technical=False):
    runtime = object.__new__(JaxDevRuntime)
    runtime.jax = SimpleNamespace(device_get=lambda value: value)
    runtime.static = SimpleNamespace(role_encoding=np.array([1, 1, -1, -1]))
    runtime._reset_box_score()
    previous = SimpleNamespace(
        positions=np.array([[[0, 0], [1, 0], [2, 0], [3, 0]]]),
        ball_holder=np.array([0]),
        game_phase=np.array([GAME_PHASE_LIVE]),
    )
    output = SimpleNamespace(
        state=SimpleNamespace(
            ball_holder=np.array([1]),  # Inbounder differs from the interceptor.
            game_phase=np.array([GAME_PHASE_AWAITING_INBOUND]),
        ),
        **{key: np.array([value]) for key, value in {
            "shot_attempt": 0, "pass_attempt": 1, "pass_passer": 0,
            "pass_receiver": 1, "completed_pass": 0,
            "turnover": turnover, "turnover_reason": reason,
            "turnover_player": 0 if turnover else -1, "steal_player": stealer,
            "offensive_three_seconds": 0, "defensive_lane_violation": technical,
            "defensive_lane_violation_player": 2 if technical else -1,
        }.items()},
    )
    return runtime, previous, output


def test_incomplete_pass_without_turnover_does_not_fabricate_a_steal():
    runtime, previous, output = _events(
        turnover=0, reason=TURNOVER_REASON_NONE, stealer=-1, technical=True,
    )
    results = runtime._action_results_from_step(previous, output)
    assert results["turnovers"] == []
    assert results["passes"] == {}
    assert results["defensive_lane_violations"][0]["player_id"] == 2
    runtime._record_box_score_step(previous, output)
    assert all(line["stl"] == line["tov"] == 0 for line in runtime._box_score_players.values())


@pytest.mark.parametrize("reason,stealer,expected_reason,expected_stealer", [
    (TURNOVER_REASON_INTERCEPTED, 3, "steal", 3),
    (TURNOVER_REASON_PASS_OUT_OF_BOUNDS, -1, "pass_out_of_bounds", None),
])
def test_pass_turnover_uses_authoritative_interceptor(reason, stealer, expected_reason, expected_stealer):
    runtime, previous, output = _events(turnover=1, reason=reason, stealer=stealer)
    turnover, = runtime._action_results_from_step(previous, output)["turnovers"]
    assert turnover["player_id"] == 0
    assert turnover["reason"] == expected_reason
    assert turnover["stolen_by"] == expected_stealer
    runtime._record_box_score_step(previous, output)
    assert runtime._box_score_players["0"]["tov"] == 1
    assert runtime._box_score_players["1"]["stl"] == 0
    assert runtime._box_score_players["3"]["stl"] == int(stealer == 3)


def test_block_is_a_missed_attempt_and_one_defenders_stat():
    runtime, previous, output = _events(turnover=0, reason=TURNOVER_REASON_NONE, stealer=-1)
    for key, value in {
        'pass_attempt':0, 'shot_attempt':1, 'shot_shooter':0, 'shot_type':0,
        'shot_success':0, 'shot_blocked':1, 'shot_blocker':2,
        'shot_value':2., 'shot_expected_points':1.176, 'shot_distance':1.,
        'shot_open_probability':.8, 'shot_conditional_probability':.6,
        'shot_block_probability':.02,
    }.items():
        setattr(output,key,np.array([value]))
    runtime._record_box_score_step(previous,output)
    assert runtime._box_score_players['0']['fga'] == 1
    assert runtime._box_score_players['0']['fgm'] == 0
    assert runtime._box_score_players['2']['blk'] == 1
    assert all(line['stl'] == line['tov'] == line['ast'] == 0 for line in runtime._box_score_players.values())
    shot = runtime._action_results_from_step(previous,output)['shots']['0']
    assert shot['blocked'] and shot['blocker_id'] == 2
    assert shot['probability'] == pytest.approx(.588)
    assert shot['conditional_probability'] == pytest.approx(.6)
