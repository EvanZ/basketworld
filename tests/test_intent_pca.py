import numpy as np

from analytics.intent_pca import _build_feature_matrix
from basketworld.utils.intent_discovery import CompletedIntentEpisode, IntentTransition
from basketworld.utils.intent_pca import (
    SUMMARY_FEATURE_NAMES,
    build_summary_feature,
    infer_outcome_from_episode_info,
)


def test_infer_outcome_from_episode_info_prefers_dunk_over_turnover_flags():
    outcome, shot_type, points = infer_outcome_from_episode_info(
        {
            "shot_dunk": 1,
            "made_dunk": 1,
            "turnover": 1,
            "turnover_pressure": 1,
        }
    )
    assert outcome == "Made Dunk"
    assert shot_type == "dunk"
    assert points == 2.0


def test_infer_outcome_from_episode_info_handles_turnover():
    outcome, shot_type, points = infer_outcome_from_episode_info(
        {
            "turnover": 1,
            "turnover_intercepted": 1,
        }
    )
    assert outcome == "Turnover (Intercepted)"
    assert shot_type == "turnover"
    assert points == 0.0


def test_build_summary_feature_shape_matches_feature_names():
    feat = build_summary_feature(
        {
            "episode_length": 6,
            "active_prefix_length": 4,
            "pass_attempts": 2,
            "pass_completions": 1,
            "pass_intercepts": 1,
            "pass_oob": 0,
            "assist_potential": 0,
            "assist_full": 1,
            "points": 3,
            "team_reward_offense": 2.5,
            "outcome": "Made 3pt",
        }
    )
    assert feat.shape == (len(SUMMARY_FEATURE_NAMES),)
    # Made 3pt bucket should be active.
    assert feat[11] == 1.0


def test_set_attention_pool_feature_mode_means_active_prefix_embeddings():
    ep = CompletedIntentEpisode(
        intent_index=3,
        transitions=[
            IntentTransition(
                feature=np.array([1.0, 2.0], dtype=np.float32),
                buffer_step_idx=0,
                env_idx=0,
                role_flag=1.0,
                intent_active=True,
                intent_index=3,
            ),
            IntentTransition(
                feature=np.array([3.0, 4.0], dtype=np.float32),
                buffer_step_idx=1,
                env_idx=0,
                role_flag=1.0,
                intent_active=True,
                intent_index=3,
            ),
            IntentTransition(
                feature=np.array([99.0, 99.0], dtype=np.float32),
                buffer_step_idx=2,
                env_idx=0,
                role_flag=1.0,
                intent_active=False,
                intent_index=3,
            ),
        ],
    )

    x, y, feature_names = _build_feature_matrix(
        "set_attention_pool",
        [ep],
        metadata=[],
        max_obs_dim=256,
        max_action_dim=16,
    )

    assert x.shape == (1, 2)
    assert y.tolist() == [3]
    assert feature_names == ["attn_pool_0000", "attn_pool_0001"]
    assert np.allclose(x[0], [2.0, 3.0])
