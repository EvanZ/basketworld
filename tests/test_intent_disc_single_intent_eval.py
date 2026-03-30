import numpy as np

from analytics.intent_disc_single_intent_eval import (
    _build_transition_single_obs,
    _inspect_single_obs_input,
    _infer_run_id,
    _parse_logged_opponent_assignment_text,
    _fit_logged_opponent_assignments,
    _resolve_policy_and_disc_paths,
    _completed_pass_boundary,
    _segment_boundary_reason,
    compute_heldout_probe_metrics,
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


def test_build_transition_single_obs_uses_env_idx_as_metadata_not_obs_batch_index():
    obs = {
        "obs": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "action_mask": np.ones((4,), dtype=np.float32),
        "role_flag": np.array([1.0], dtype=np.float32),
        "intent_index": np.array([3.0], dtype=np.float32),
        "intent_active": np.array([1.0], dtype=np.float32),
    }
    transition = _build_transition_single_obs(
        obs,
        np.array([1], dtype=np.int64),
        max_obs_dim=8,
        max_action_dim=4,
        rollout_step_idx=0,
        env_idx=5,
    )
    assert transition.env_idx == 5
    assert transition.role_flag == 1.0
    assert transition.intent_active is True
    assert transition.intent_index == 3


def test_build_transition_single_obs_uses_info_fallback_for_intent_fields():
    obs = {
        "obs": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "action_mask": np.ones((4,), dtype=np.float32),
        "role_flag": np.array([1.0], dtype=np.float32),
    }
    transition = _build_transition_single_obs(
        obs,
        np.array([1], dtype=np.int64),
        max_obs_dim=8,
        max_action_dim=4,
        rollout_step_idx=0,
        env_idx=0,
        info={"intent_active": 1.0, "intent_index": 6.0},
    )
    assert transition.intent_active is True
    assert transition.intent_index == 6


def test_compute_heldout_probe_metrics_succeeds_on_separable_data():
    episodes = []
    centers = {
        0: -4.0,
        1: 0.0,
        2: 4.0,
    }
    for intent_index, center in centers.items():
        for episode_idx in range(12):
            base = center + float(episode_idx) * 0.01
            transitions = [
                IntentTransition(
                    feature=np.array([base, base + 0.5, 0.0, 0.0], dtype=np.float32),
                    buffer_step_idx=t,
                    env_idx=0,
                    role_flag=1.0,
                    intent_active=True,
                    intent_index=int(intent_index),
                )
                for t in range(3)
            ]
            episodes.append(
                CompletedIntentEpisode(
                    intent_index=int(intent_index),
                    transitions=transitions,
                )
            )

    result = compute_heldout_probe_metrics(
        episodes,
        max_obs_dim=4,
        max_action_dim=0,
        num_intents=3,
        test_fraction=0.25,
        knn_k=3,
        seed=0,
        device="cpu",
    )
    assert "error" not in result
    assert result["num_train"] > 0
    assert result["num_test"] > 0
    assert float(result["linear_probe_top1_acc"]) >= 0.9
    assert float(result["knn_top1_acc"]) >= 0.9


def test_resolve_policy_and_disc_paths_allows_missing_disc_for_probe_only(tmp_path):
    policy_path = tmp_path / "unified_iter_44.zip"
    policy_path.write_bytes(b"stub")

    resolved_policy, resolved_disc, inferred_run_id = _resolve_policy_and_disc_paths(
        str(policy_path),
        checkpoint_idx=44,
        disc_path=None,
        require_disc=False,
    )

    assert resolved_policy == str(policy_path.resolve())
    assert resolved_disc is None
    assert inferred_run_id is None


def test_infer_run_id_handles_mlflow_artifact_uri():
    run_id = _infer_run_id(
        "mlflow-artifacts:/7/794ae78c93f14b42a9f740753b40ff6b/artifacts/models/unified_iter_28.zip"
    )
    assert run_id == "794ae78c93f14b42a9f740753b40ff6b"


def test_parse_logged_opponent_assignment_text_preserves_env_order():
    parsed = _parse_logged_opponent_assignment_text(
        "\n".join(
            [
                "Env 3: unified_iter_12.zip",
                "Env 1: unified_iter_10.zip",
                "Env 2: unified_iter_11.zip",
            ]
        )
    )
    assert parsed == [
        "unified_iter_10.zip",
        "unified_iter_11.zip",
        "unified_iter_12.zip",
    ]


def test_fit_logged_opponent_assignments_repeats_single_assignment():
    fitted = _fit_logged_opponent_assignments(
        ["unified_iter_9.zip"],
        num_assignments=4,
    )
    assert fitted == [
        "unified_iter_9.zip",
        "unified_iter_9.zip",
        "unified_iter_9.zip",
        "unified_iter_9.zip",
    ]
