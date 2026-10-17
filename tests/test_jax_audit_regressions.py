"""Lightweight regression checks for the baseline functional audit."""

from dataclasses import asdict, replace
import json

import numpy as np
import pytest

from basketworld_jax.config import JaxEnvironmentConfig
from basketworld_jax.checkpoints import build_checkpoint_payload
from basketworld_jax.checkpoints.checkpoint import (
    _metadata_from_payload,
    _payload_from_metadata,
    load_checkpoint_metadata,
)
from basketworld_jax.env.minimal import build_kernel_static_from_config
from basketworld_jax.models import ActorCriticSpec
from basketworld_jax.train.main import (
    TRAIN_FROZEN_VALUES,
    _effective_env_config_from_args,
    _checkpoint_observation_metadata,
    _jax_env_config_from_args,
    _log_mlflow_params,
    _validate_resume_checkpoint_payload,
    _validate_checkpoint_episode_semantics,
    build_trainer_config,
    parse_args,
    validate_train_args,
    build_train_config_schema,
)


@pytest.mark.parametrize(
    ("alias", "canonical", "value"),
    [
        ("shot_clock_steps", "shot_clock", 18),
        ("court_height", "court_rows", 11),
        ("court_width", "court_cols", 10),
        ("players_per_side", "players", 2),
        ("offensive_three_seconds", "offensive_three_seconds_enabled", True),
        ("phi_beta_start", "phi_beta", 0.25),
        ("pass_oob_turnover_prob_start", "pass_oob_turnover_prob", 0.4),
    ],
)
def test_alias_overrides_checkpoint_and_required_layer(alias, canonical, value):
    metadata = {
        "env_config": JaxEnvironmentConfig(players=5, shot_clock=24).as_metadata()
    }
    config = JaxEnvironmentConfig.from_evaluation_request(
        {canonical: metadata["env_config"][canonical]},
        {alias: value},
        checkpoint_metadata=metadata,
    )
    assert getattr(config, canonical) == value
    assert config.layup_pct == metadata["env_config"]["layup_pct"]


def test_alias_precedence_within_layer_and_across_checkpoint_layers():
    config = JaxEnvironmentConfig.from_checkpoint_metadata(
        {
            "frozen_config": {"shot_clock": 24, "players": 5},
            "env_config": {"shot_clock_steps": 18},
        }
    )
    assert (config.shot_clock, config.players) == (18, 5)
    config = JaxEnvironmentConfig.from_evaluation_request(
        {"shot_clock_steps": 18, "shot_clock": 20},
    )
    assert config.shot_clock == 20


def _spec():
    return ActorCriticSpec(
        flat_obs_dim=91,
        training_player_count=3,
        action_dim_per_player=14,
        total_action_dim=42,
        hidden_dims=(128, 128),
    )


class Recorder:
    def __init__(self):
        self.params = None
        self.artifacts = {}

    def log_params(self, values):
        self.params = values

    def log_dict(self, values, path):
        self.artifacts[path] = values

    def set_tag(self, *args):
        pass


@pytest.mark.parametrize("explicit_gamma", [None, 1.0])
def test_effective_rewards_match_kernel_and_preserve_requested_values(explicit_gamma):
    argv = [
        "--gamma",
        "0.97",
        "--policy-seed",
        "37",
        "--rebound-terminal-reward-mode",
        "last_shot_ep",
    ]
    if explicit_gamma is not None:
        argv += ["--reward-shaping-gamma", str(explicit_gamma)]
    args = parse_args(argv)
    effective = _effective_env_config_from_args(args)
    static = build_kernel_static_from_config(JaxEnvironmentConfig.from_args(args), np)
    assert effective["reward_shaping_gamma"] == pytest.approx(
        float(static.reward_shaping_gamma)
    )
    assert effective["rebound_terminal_reward_mode"] == "last_shot_ep"
    assert int(static.rebound_terminal_reward_mode) == 2
    assert _jax_env_config_from_args(args)["reward_shaping_gamma"] == explicit_gamma
    recorder = Recorder()
    _log_mlflow_params(recorder, args, build_trainer_config(args), _spec())
    assert recorder.params["jax/policy_seed"] == 37
    assert (
        recorder.params["jax/env/reward_shaping_gamma"]
        == effective["reward_shaping_gamma"]
    )
    assert (
        recorder.artifacts["metadata/resolved_training_config.json"][
            "reward_shaping_gamma"
        ]
        == explicit_gamma
    )
    assert recorder.artifacts["metadata/effective_env_config.json"] == effective


def test_effective_environment_survives_metadata_and_payload_roundtrip(tmp_path):
    effective = {
        "reward_shaping_gamma": 0.97,
        "rebound_terminal_reward_mode": "last_shot_ep",
    }
    payload = build_checkpoint_payload(
        update_index=1,
        trainer_config={"gamma": 0.97},
        policy_spec={},
        frozen_config={},
        env_config={"reward_shaping_gamma": None},
        effective_env_config=effective,
        params={},
        opt_state={},
        current_state={},
        eval_initial_state={},
        base_key=None,
        eval_trajectories=[],
        last_metrics=None,
    )
    metadata = _metadata_from_payload(payload)
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    assert load_checkpoint_metadata(tmp_path)["effective_env_config"] == effective
    restored = _payload_from_metadata(metadata, payload["state"])
    assert restored["effective_env_config"] == effective
    assert (
        JaxEnvironmentConfig.from_checkpoint_metadata(restored).reward_shaping_gamma
        == 0.97
    )
    assert restored["env_config"]["reward_shaping_gamma"] is None


def test_continuation_logs_effective_values_without_rewriting_params():
    args = parse_args(
        [
            "--mlflow-resume-run-id",
            "existing",
            "--num-updates",
            "100",
            "--gamma",
            "0.97",
        ]
    )
    recorder = Recorder()
    _log_mlflow_params(recorder, args, build_trainer_config(args), _spec())
    assert recorder.params is None
    assert (
        recorder.artifacts["metadata/continuations/target_0000100_effective_env.json"][
            "reward_shaping_gamma"
        ]
        == 0.97
    )


@pytest.mark.parametrize("recorded_effective", [False, True])
def test_resume_preserves_effective_gamma(recorded_effective):
    args = parse_args(["--gamma", "0.97"])
    trainer = build_trainer_config(args)
    payload = {
        "trainer_config": asdict(trainer),
        "policy_spec": asdict(_spec()),
        "frozen_config": {key: getattr(args, key) for key in TRAIN_FROZEN_VALUES},
        "env_config": _jax_env_config_from_args(args),
    }
    if recorded_effective:
        payload["effective_env_config"] = _effective_env_config_from_args(args)
        _validate_resume_checkpoint_payload(
            payload, trainer_config=trainer, spec=_spec(), args=args
        )
    else:
        with pytest.raises(
            SystemExit, match="does not record effective reward_shaping_gamma"
        ):
            _validate_resume_checkpoint_payload(
                payload, trainer_config=trainer, spec=_spec(), args=args
            )
        args.reward_shaping_gamma = 1.0
        _validate_resume_checkpoint_payload(
            payload, trainer_config=trainer, spec=_spec(), args=args
        )


def test_game_clock_training_configuration_is_accepted():
    args = parse_args(["--enable-multi-possession", "--episode-end-mode", "game_clock"])
    validate_train_args(args)


def test_native_game_clock_evaluation_is_rejected_before_jax_initialization(
    monkeypatch,
):
    import basketworld_jax.eval.native as native

    payload = {
        "env_config": {
            "enable_multi_possession": True,
            "episode_end_mode": "game_clock",
        }
    }
    monkeypatch.setattr(
        native, "_load_checkpoint_params", lambda path: (payload, {}, _spec())
    )

    def unexpected(*args):
        pytest.fail(
            "Initialized JAX before rejecting unsupported game-clock evaluation"
        )

    monkeypatch.setattr(native, "ensure_jax_available", unexpected)
    with pytest.raises(ValueError, match="issue #65"):
        native.run_native_jax_evaluation(
            num_episodes=1,
            player_deterministic=True,
            opponent_deterministic=True,
            required_params={},
            optional_params={},
            unified_policy_path="unused",
            opponent_policy_path=None,
            user_team_name="OFFENSE",
            role_flag_offense=1.0,
            role_flag_defense=-1.0,
        )


def test_trainer_schema_exposes_both_episode_modes():
    fields = {
        field["name"]: field for field in build_train_config_schema(parse_args([]))
    }
    assert fields["episode_end_mode"]["choices"] == ["possessions", "game_clock"]


def test_checkpoint_semantics_reject_silent_cross_mode_reuse():
    clock_args = parse_args(
        ["--enable-multi-possession", "--episode-end-mode", "game_clock"]
    )
    with pytest.raises(SystemExit, match="different observation semantics"):
        _validate_checkpoint_episode_semantics(
            {"env_config": {"episode_end_mode": "possessions"}},
            args=clock_args,
            context="Frozen opponent checkpoint",
        )

    clock_spec = replace(
        _spec(),
        multi_possession_features=True,
        observation_schema_version=4,
    )
    observation_metadata = _checkpoint_observation_metadata(
        clock_spec,
        args=clock_args,
    )
    assert observation_metadata["episode_budget_feature"]["normalization"] == 720.0
    assert observation_metadata["relative_score_feature"]["normalization"] == 30.0
    _validate_checkpoint_episode_semantics(
        {
            "env_config": {"episode_end_mode": "game_clock"},
            "observation_metadata": observation_metadata,
        },
        args=clock_args,
        context="Resume checkpoint",
    )

    possession_args = parse_args(["--enable-multi-possession"])
    _validate_checkpoint_episode_semantics(
        {"env_config": {}},
        args=possession_args,
        context="Legacy possession checkpoint",
    )


def test_game_clock_mlflow_metadata_records_mode_durations_and_observation_scales():
    args = parse_args(
        [
            "--enable-multi-possession",
            "--episode-end-mode",
            "game_clock",
            "--game-clock-seconds",
            "300",
            "--overtime-seconds",
            "120",
        ]
    )
    spec = replace(
        _spec(),
        multi_possession_features=True,
        observation_schema_version=4,
    )
    recorder = Recorder()
    _log_mlflow_params(recorder, args, build_trainer_config(args), spec)

    assert recorder.params["jax/env/episode_end_mode"] == "game_clock"
    assert recorder.params["jax/env/game_clock_seconds"] == 300
    assert recorder.params["jax/env/overtime_seconds"] == 120
    assert recorder.params["jax/observation/game_clock_time_normalization_seconds"] == 720.0
    assert recorder.params["jax/observation/score_difference_normalization"] == 30.0
    observation = recorder.artifacts["metadata/observation.json"]
    assert observation["episode_budget_feature"]["normalization"] == 720.0
    assert observation["relative_score_feature"]["normalization"] == 30.0


def test_native_resolver_respects_alias_overrides_on_both_checkpoints():
    from basketworld_jax.eval.native import _resolve_native_environment_config

    config = _resolve_native_environment_config(
        required_params={},
        optional_params={"shot_clock_steps": 18, "court_height": 11},
        unified_payload={"env_config": {"shot_clock": 24, "court_rows": 9}},
        opponent_payload={"env_config": {"shot_clock": 20, "court_rows": 10}},
    )
    assert config.shot_clock == 18
    assert config.court_rows == 11


def test_resume_rejects_changed_recorded_effective_gamma():
    args = parse_args(["--gamma", "0.97"])
    trainer = build_trainer_config(args)
    payload = {
        "trainer_config": asdict(trainer),
        "policy_spec": asdict(_spec()),
        "frozen_config": {key: getattr(args, key) for key in TRAIN_FROZEN_VALUES},
        "env_config": _jax_env_config_from_args(args),
        "effective_env_config": {"reward_shaping_gamma": 1.0},
    }
    with pytest.raises(SystemExit, match="effective reward_shaping_gamma differs"):
        _validate_resume_checkpoint_payload(
            payload, trainer_config=trainer, spec=_spec(), args=args
        )


def test_resume_can_explicitly_resolve_post_migration_unrecorded_gamma():
    args = parse_args(["--gamma", "0.97"])
    trainer = build_trainer_config(args)
    payload = {
        "trainer_config": asdict(trainer),
        "policy_spec": asdict(_spec()),
        "frozen_config": {key: getattr(args, key) for key in TRAIN_FROZEN_VALUES},
        "env_config": _jax_env_config_from_args(args),
    }
    args.reward_shaping_gamma = 0.97
    _validate_resume_checkpoint_payload(
        payload, trainer_config=trainer, spec=_spec(), args=args
    )
