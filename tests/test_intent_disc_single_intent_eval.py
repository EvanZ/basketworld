import numpy as np

from analytics.intent_disc_single_intent_eval import (
    _inspect_single_obs_input,
    _completed_pass_boundary,
    _segment_boundary_reason,
    score_single_intent_segments,
    validate_result,
)
from basketworld.utils.intent_discovery import CompletedIntentEpisode, IntentTransition


class _DummyDisc:
    def __init__(self, logits):
        self._logits = logits

    def __call__(self, x, lengths):
        return self._logits


def _episode(intent_index: int, steps: int = 3) -> CompletedIntentEpisode:
    transitions = [
        IntentTransition(
            feature=np.ones((8,), dtype=np.float32),
            buffer_step_idx=t,
            env_idx=0,
            role_flag=1.0,
            intent_active=True,
            intent_index=int(intent_index),
        )
        for t in range(steps)
    ]
    return CompletedIntentEpisode(intent_index=int(intent_index), transitions=transitions)


def test_completed_pass_boundary_detects_successful_pass():
    info = {"action_results": {"passes": {"0": {"success": True}}}}
    assert _completed_pass_boundary(info) is True
    assert (
        _segment_boundary_reason(
            info=info,
            segment_length=4,
            min_play_steps=4,
            commitment_steps=8,
        )
        == "completed_pass"
    )


def test_segment_boundary_reason_prefers_commitment_timeout():
    info = {"action_results": {"passes": {"0": {"success": True}}}}
    assert (
        _segment_boundary_reason(
            info=info,
            segment_length=8,
            min_play_steps=4,
            commitment_steps=8,
        )
        == "commitment_timeout"
    )


def test_score_single_intent_segments_reports_target_rate():
    logits = np.array(
        [
            [0.1, 3.0, 0.2],
            [0.1, 2.5, 0.2],
            [0.1, 1.0, 2.0],
        ],
        dtype=np.float32,
    )
    disc = _DummyDisc(logits=np.asarray(logits))
    payload = {
        "config": {
            "encoder_type": "mlp_mean",
            "max_obs_dim": 4,
            "max_action_dim": 4,
            "num_intents": 3,
        }
    }
    episodes = [_episode(1), _episode(1), _episode(1)]
    result = score_single_intent_segments(
        episodes,
        disc_bundle=(disc, payload),
        device="cpu",
    )
    assert result["target_intent"] == 1
    assert abs(float(result["target_prediction_rate"]) - (2.0 / 3.0)) < 1e-6
    assert result["predicted_class_histogram"] == [0, 2, 1]


def test_validate_result_handles_all_intents_mode():
    result = {
        "per_intent": {
            "0": {"target_prediction_rate": 0.91},
            "1": {"target_prediction_rate": 0.42},
        }
    }
    validation = validate_result(result, min_target_rate=0.80)
    assert validation["passed"] is False
    assert validation["failed_checks"] == [
        "intent 1 target_prediction_rate 0.420000 < required 0.800000"
    ]


def test_inspect_single_obs_input_reports_raw_intent_fields_and_flatten_source():
    obs = {
        "obs": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "players": np.ones((3, 4), dtype=np.float32),
        "globals": np.ones((8,), dtype=np.float32),
        "intent_index": np.array([2.0], dtype=np.float32),
        "intent_active": np.array([1.0], dtype=np.float32),
    }
    diag = _inspect_single_obs_input(obs, target_intent=2)
    assert diag["has_obs_key"] is True
    assert diag["has_players_key"] is True
    assert diag["has_globals_key"] is True
    assert diag["has_direct_intent_index_key"] is True
    assert diag["has_direct_intent_active_key"] is True
    assert diag["raw_direct_intent_index"] == 2
    assert diag["raw_direct_intent_active"] is True
    assert diag["raw_direct_intent_target_match"] is True
    assert diag["flatten_matches_obs_key"] is True
    assert diag["flatten_feature_dim"] == 3
    assert diag["raw_obs_dim"] == 3
