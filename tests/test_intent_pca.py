import numpy as np
from pathlib import Path
import torch

from analytics.intent_pca import (
    _build_env_args,
    _build_feature_matrix,
    _compute_confusion_matrix_tables,
    _episode_label_intent_index,
    _fit_logged_opponent_assignments,
    _intent_pca_mlflow_artifact_path,
    _maybe_apply_selector_intent_start,
    _offense_subset_logged_assignments,
    _parse_logged_opponent_assignment_text,
    _resolve_intent_source_mode,
    _resolve_discriminator_feature_dims,
    _selector_alpha_current,
    _sample_training_matched_opponent_artifacts,
    _training_offense_assignment_count,
    _select_discriminator_eval_episodes,
    resolve_policy_path,
)
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


def test_resolve_policy_path_uses_requested_checkpoint_in_local_directory(tmp_path: Path):
    older = tmp_path / "unified_iter_12.zip"
    newer = tmp_path / "unified_iter_45.zip"
    alt = tmp_path / "unified_alternation_20.zip"
    for path in (older, newer, alt):
        path.write_bytes(b"")

    selected, inferred_run_id = resolve_policy_path(str(tmp_path), checkpoint_idx=20)

    assert selected == str(alt)
    assert inferred_run_id is None


def test_resolve_policy_path_errors_when_requested_checkpoint_missing(tmp_path: Path):
    (tmp_path / "unified_iter_12.zip").write_bytes(b"")

    try:
        resolve_policy_path(str(tmp_path), checkpoint_idx=99)
    except RuntimeError as exc:
        assert "alternation/index=99" in str(exc)
    else:
        raise AssertionError("Expected resolve_policy_path to raise for missing checkpoint")


def test_intent_pca_mlflow_artifact_path_uses_checkpoint_index():
    assert (
        _intent_pca_mlflow_artifact_path("/tmp/unified_iter_30.zip")
        == "analysis/intent_pca/iter_30"
    )
    assert (
        _intent_pca_mlflow_artifact_path("/tmp/unified_alternation_12.zip")
        == "analysis/intent_pca/alternation_12"
    )
    assert (
        _intent_pca_mlflow_artifact_path("/tmp/unified_latest.zip")
        == "analysis/intent_pca/latest"
    )


def test_compute_confusion_matrix_tables_row_normalizes_counts():
    labels, counts, row_norm = _compute_confusion_matrix_tables(
        np.array([0, 0, 1, 1, 1], dtype=np.int64),
        np.array([0, 1, 1, 1, 0], dtype=np.int64),
        num_intents=3,
    )

    assert labels == [0, 1, 2]
    assert counts.tolist() == [
        [1, 1, 0],
        [1, 2, 0],
        [0, 0, 0],
    ]
    assert np.allclose(row_norm[0], [0.5, 0.5, 0.0])
    assert np.allclose(row_norm[1], [1.0 / 3.0, 2.0 / 3.0, 0.0])
    assert np.allclose(row_norm[2], [0.0, 0.0, 0.0])


def test_resolve_discriminator_feature_dims_prefers_checkpoint_config():
    disc_bundle = (
        object(),
        {
            "config": {
                "max_obs_dim": 192,
                "max_action_dim": 12,
            }
        },
    )

    obs_dim, action_dim = _resolve_discriminator_feature_dims(
        disc_bundle,
        default_max_obs_dim=256,
        default_max_action_dim=16,
    )

    assert obs_dim == 192
    assert action_dim == 12


def test_select_discriminator_eval_episodes_prefers_raw_step_collection():
    feature_episode = CompletedIntentEpisode(
        intent_index=0,
        transitions=[
            IntentTransition(
                feature=np.array([99.0, 99.0], dtype=np.float32),
                buffer_step_idx=0,
                env_idx=0,
                role_flag=1.0,
                intent_active=True,
                intent_index=0,
            )
        ],
    )
    raw_episode = CompletedIntentEpisode(
        intent_index=0,
        transitions=[
            IntentTransition(
                feature=np.array([1.0, 2.0], dtype=np.float32),
                buffer_step_idx=0,
                env_idx=0,
                role_flag=1.0,
                intent_active=True,
                intent_index=0,
            )
        ],
    )

    selected = _select_discriminator_eval_episodes([feature_episode], [raw_episode])

    assert selected is not None
    assert len(selected) == 1
    assert np.allclose(selected[0].transitions[0].feature, [1.0, 2.0])


def test_episode_label_intent_index_prefers_first_transition_label():
    transitions = [
        IntentTransition(
            feature=np.array([1.0, 2.0], dtype=np.float32),
            buffer_step_idx=0,
            env_idx=0,
            role_flag=1.0,
            intent_active=True,
            intent_index=6,
        )
    ]

    assert _episode_label_intent_index(transitions, fallback_intent_index=2) == 6
    assert _episode_label_intent_index([], fallback_intent_index=2) == 2


def test_build_env_args_preserves_mirror_prob_for_set_obs(monkeypatch):
    class DummyModel:
        observation_space = type(
            "ObsSpace",
            (),
            {"spaces": {"players": object(), "globals": object()}},
        )()

    monkeypatch.setattr(
        "analytics.intent_pca.get_args",
        lambda argv: type(
            "Args",
            (),
            {
                "use_set_obs": False,
                "enable_env_profiling": True,
                "mirror_episode_prob": 0.0,
            },
        )(),
    )
    monkeypatch.setattr(
        "analytics.intent_pca.get_mlflow_params",
        lambda client, run_id: ({}, {"mirror_episode_prob": 0.35}),
    )
    monkeypatch.setattr(
        "analytics.intent_pca.get_mlflow_training_params",
        lambda client, run_id: {},
    )

    env_args = _build_env_args("dummy_run", DummyModel())

    assert env_args.use_set_obs is True
    assert env_args.enable_env_profiling is False
    assert env_args.mirror_episode_prob == 0.35


def test_parse_logged_opponent_assignment_text_reads_env_mapping():
    parsed = _parse_logged_opponent_assignment_text(
        "\n".join(
            [
                "Per-environment opponent sampling (geometric distribution):",
                "  Env  0: unified_iter_61.zip",
                "  Env  1: unified_iter_64.zip",
                "",
                "Summary:",
            ]
        )
    )

    assert parsed == ["unified_iter_61.zip", "unified_iter_64.zip"]


def test_parse_logged_opponent_assignment_text_reads_single_policy_note():
    parsed = _parse_logged_opponent_assignment_text(
        "\n".join(
            [
                "Single policy for all environments:",
                "  unified_iter_64.zip",
            ]
        )
    )

    assert parsed == ["unified_iter_64.zip"]


def test_fit_logged_opponent_assignments_expands_single_policy():
    fitted = _fit_logged_opponent_assignments(
        ["unified_iter_64.zip"],
        num_assignments=4,
    )

    assert fitted == ["unified_iter_64.zip"] * 4


def test_fit_logged_opponent_assignments_truncates_to_target_size():
    fitted = _fit_logged_opponent_assignments(
        [
            "unified_iter_61.zip",
            "unified_iter_62.zip",
            "unified_iter_63.zip",
        ],
        num_assignments=2,
    )

    assert fitted == ["unified_iter_61.zip", "unified_iter_62.zip"]


def test_training_offense_assignment_count_uses_half_of_mixed_envs():
    env_args = type("Args", (), {"num_envs": 16})()
    assert _training_offense_assignment_count(env_args) == 8


def test_offense_subset_logged_assignments_uses_offense_half_only():
    env_args = type("Args", (), {"num_envs": 16})()
    basenames = [f"unified_iter_{i}.zip" for i in range(16)]

    subset = _offense_subset_logged_assignments(basenames, env_args=env_args)

    assert subset == [f"unified_iter_{i}.zip" for i in range(8)]


def test_sample_training_matched_opponent_artifacts_uses_single_opponent_when_not_per_env():
    candidates = [
        (11, "models/unified_iter_11.zip"),
        (12, "models/unified_iter_12.zip"),
        (13, "models/unified_iter_13.zip"),
    ]

    selected = _sample_training_matched_opponent_artifacts(
        candidates,
        num_assignments=4,
        pool_size=2,
        beta=0.7,
        uniform_eps=0.0,
        per_env_sampling=False,
        seed=123,
    )

    assert len(selected) == 4
    assert len(set(selected)) == 1
    assert selected[0] in {
        "models/unified_iter_12.zip",
        "models/unified_iter_13.zip",
    }


def test_sample_training_matched_opponent_artifacts_can_draw_older_history_with_exploration():
    candidates = [
        (9, "models/unified_iter_9.zip"),
        (10, "models/unified_iter_10.zip"),
        (11, "models/unified_iter_11.zip"),
        (12, "models/unified_iter_12.zip"),
    ]

    selected = _sample_training_matched_opponent_artifacts(
        candidates,
        num_assignments=20,
        pool_size=2,
        beta=0.7,
        uniform_eps=1.0,
        per_env_sampling=True,
        seed=7,
    )

    assert len(selected) == 20
    assert "models/unified_iter_9.zip" in selected or "models/unified_iter_10.zip" in selected


class _SelectorTestPolicy:
    def __init__(self):
        self._enabled = True

    def has_intent_selector(self):
        return True

    def get_intent_selector_logits(self, _obs):
        return torch.tensor([[0.0, 0.0, 10.0, 0.0]], dtype=torch.float32)


class _SelectorTestModel:
    def __init__(self, num_timesteps: int = 75):
        self.num_timesteps = num_timesteps
        self.policy = _SelectorTestPolicy()


class _SelectorTestEnv:
    def __init__(self):
        self.intent_active = True
        self.intent_index = 1
        self.last_set = None

    @property
    def unwrapped(self):
        return self

    def set_offense_intent_state(self, intent_index: int, *, intent_active=True, intent_age=0):
        self.intent_active = bool(intent_active)
        self.intent_index = int(intent_index)
        self.last_set = (int(intent_index), bool(intent_active), int(intent_age))


class _SelectorWrappedEnv:
    def __init__(self):
        self.env = _SelectorTestEnv()

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def set_offense_intent_state(self, intent_index: int, *, intent_active=True, intent_age=0):
        return self.env.set_offense_intent_state(
            intent_index,
            intent_active=intent_active,
            intent_age=intent_age,
        )


def test_selector_alpha_current_matches_training_schedule():
    env_args = type(
        "Args",
        (),
        {
            "intent_selector_alpha_start": 0.2,
            "intent_selector_alpha_end": 1.0,
            "intent_selector_alpha_warmup_steps": 50,
            "intent_selector_alpha_ramp_steps": 100,
        },
    )()

    assert np.isclose(_selector_alpha_current(_SelectorTestModel(0), env_args), 0.2)
    assert np.isclose(_selector_alpha_current(_SelectorTestModel(50), env_args), 0.2)
    assert np.isclose(_selector_alpha_current(_SelectorTestModel(100), env_args), 0.6)
    assert np.isclose(_selector_alpha_current(_SelectorTestModel(200), env_args), 1.0)


def test_resolve_intent_source_mode_prefers_selector_when_enabled():
    env_args = type("Args", (), {"intent_selector_enabled": True})()
    resolved = _resolve_intent_source_mode("training_match", _SelectorTestModel(), env_args)
    assert resolved == "selector"


def test_maybe_apply_selector_intent_start_patches_obs_and_env():
    obs = {
        "intent_index": np.array([1.0], dtype=np.float32),
        "intent_active": np.array([1.0], dtype=np.float32),
        "intent_visible": np.array([1.0], dtype=np.float32),
        "intent_age_norm": np.array([0.4], dtype=np.float32),
        "globals": np.zeros((8,), dtype=np.float32),
    }
    wrapped_env = _SelectorWrappedEnv()
    model = _SelectorTestModel(num_timesteps=1000)
    env_args = type(
        "Args",
        (),
        {
            "intent_selector_enabled": True,
            "intent_selector_alpha_start": 0.0,
            "intent_selector_alpha_end": 1.0,
            "intent_selector_alpha_warmup_steps": 0,
            "intent_selector_alpha_ramp_steps": 1,
            "num_intents": 4,
        },
    )()

    result = _maybe_apply_selector_intent_start(
        obs=obs,
        wrapped_env=wrapped_env,
        unified_policy=model,
        env_args=env_args,
        rng=np.random.default_rng(0),
        intent_source_mode="training_match",
    )

    assert result["selector_applied"] is True
    assert int(result["selected_intent_index"]) == 2
    assert wrapped_env.unwrapped.last_set == (2, True, 0)
    assert int(np.asarray(obs["intent_index"]).reshape(-1)[0]) == 2
    assert float(np.asarray(obs["intent_active"]).reshape(-1)[0]) == 1.0
