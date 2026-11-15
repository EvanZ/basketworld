from __future__ import annotations

import json
from pathlib import Path

import pytest
from types import SimpleNamespace

from basketworld_jax.eval.fixed_pool import generate_player_pool
from basketworld_jax.eval.fixed_pool_results import (
    load_fixed_pool_study_artifacts,
    persist_fixed_pool_study_artifacts,
)
from basketworld_jax.eval.fixed_pool_study import (
    FixedPoolStudyRunError,
    _study_skill_sampling_overrides,
    run_fixed_pool_study,
    write_fixed_pool_model_reanalysis,
)
from basketworld_jax.train.main import parse_args, validate_train_args


def _args(tmp_path, checkpoint_path, *, create: bool):
    return SimpleNamespace(
        fixed_pool_checkpoint=[str(checkpoint_path)],
        fixed_pool_checkpoint_dir="",
        fixed_pool_checkpoint_update=[],
        fixed_pool_path=str(tmp_path / "inputs" / "pool.json"),
        fixed_pool_create=create,
        fixed_pool_size=10,
        fixed_pool_seed=3,
        fixed_pool_schedule_path=str(tmp_path / "inputs" / "schedule.json"),
        fixed_pool_schedule_create=create,
        fixed_pool_games=2,
        fixed_pool_schedule_seed=5,
        fixed_pool_artifact_dir=str(tmp_path / "artifacts"),
        fixed_pool_report_dir="",
        fixed_pool_action_mode="direct",
        fixed_pool_batch_size=2,
        fixed_pool_eval_seed=7,
        fixed_pool_ridge_alpha=0.5,
        fixed_pool_validation_fraction=0.2,
        fixed_pool_split_seed=0,
        fixed_pool_bootstrap_samples=2,
        fixed_pool_bootstrap_seed=1,
        log_mlflow=False,
        mlflow_experiment_name="Default",
        mlflow_run_name="",
    )


def _mark_jax_checkpoint(path):
    path.mkdir()
    (path / "metadata.json").write_text("{}", encoding="utf-8")
    (path / "state").mkdir()


def _fake_native_runner(**kwargs):
    pool = kwargs["fixed_player_pool"]
    schedule = kwargs["fixed_lineup_schedule"]
    checkpoint_path = kwargs["unified_policy_path"]
    progress = kwargs["progress_event_callback"]
    progress({"phase": "native_initializing", "total_episodes": len(schedule.games)})
    progress(
        {
            "phase": "runner_ready",
            "total_episodes": len(schedule.games),
            "batch_size": kwargs["evaluation_batch_size"],
            "horizon": 12,
        }
    )
    progress(
        {
            "phase": "batch_started",
            "completed_episodes": 0,
            "batch_episode_count": len(schedule.games),
            "total_episodes": len(schedule.games),
        }
    )
    progress(
        {
            "phase": "batch_completed",
            "completed_episodes": len(schedule.games),
            "batch_episode_count": len(schedule.games),
            "total_episodes": len(schedule.games),
        }
    )
    results = []
    for episode, game in enumerate(schedule.games, start=1):
        results.append(
            {
                "episode": episode,
                "completed": True,
                "fixed_pool_study": {
                    "pool_id": pool.pool_id,
                    "schedule_id": schedule.schedule_id,
                    "action_mode": "direct",
                    "unified_checkpoint_path": checkpoint_path,
                    "unified_checkpoint_update_index": 4,
                    "opponent_checkpoint_path": checkpoint_path,
                    "opponent_checkpoint_update_index": 4,
                    "checkpoint_observation_schema_versions": [3],
                    "environment_observation_schema_version": 3,
                    "schedule_game_id": game.game_id,
                    "pair_id": game.pair_id,
                    "starting_team": game.starting_team,
                    "team_a_player_ids": list(game.team_a_player_ids),
                    "team_b_player_ids": list(game.team_b_player_ids),
                    "reset_rng_root": game.reset_rng_root,
                    "environment_rng_root": game.environment_rng_root,
                    "policy_rng_root": game.policy_rng_root,
                },
                "game": {
                    "completed": True,
                    "team_a_score": 10.0,
                    "team_b_score": 8.0,
                    "completed_possessions": 10,
                    "team_a_completed_possessions": 5,
                    "team_b_completed_possessions": 5,
                    "overtime_round": 0,
                    "overtime_possessions_completed": 0,
                },
            }
        )
    paths = persist_fixed_pool_study_artifacts(
        pool=pool,
        schedule=schedule,
        episode_results=results,
        possession_events=[],
        output_dir=kwargs["fixed_pool_artifact_dir"],
        mlflow_client=kwargs["fixed_pool_mlflow_client"],
    )
    return {
        "results": results,
        "fixed_pool_artifacts": paths.to_payload(),
        "eval_diagnostics": {
            "jax_native_summary": {
                "batch_size": kwargs["evaluation_batch_size"],
                "horizon": 12,
                "elapsed_sec": 0.5,
                "episodes_per_sec": 4.0,
                "states_per_sec": 48.0,
            }
        },
    }


def test_fixed_pool_cli_controls_are_opt_in_and_validate_study_only_requirements():
    normal = parse_args([])
    assert normal.run_fixed_pool_study is False

    args = parse_args(
        [
            "--run-fixed-pool-study",
            "--fixed-pool-checkpoint",
            "/tmp/checkpoint",
            "--fixed-pool-path",
            "/tmp/pool.json",
            "--fixed-pool-schedule-path",
            "/tmp/schedule.json",
            "--fixed-pool-artifact-dir",
            "/tmp/artifacts",
            "--fixed-pool-action-mode",
            "sampled",
        ]
    )
    validate_train_args(args)
    assert args.fixed_pool_action_mode == "sampled"
    assert args.run_train_loop is False


def test_study_pool_sampling_has_support_for_each_active_skill():
    overrides = _study_skill_sampling_overrides()
    pool = generate_player_pool(
        player_count=16,
        seed=19,
        sampling_overrides=overrides,
    )
    for skill_name, override in overrides.items():
        values = {player.skills[skill_name] for player in pool.players}
        assert override["std"] > 0.0
        assert len(values) > 1


@pytest.mark.parametrize("team_size", [3, 4, 5])
def test_operational_runner_creates_then_reuses_immutable_inputs(
    tmp_path, monkeypatch, team_size
):
    checkpoint = tmp_path / "update_0000004"
    _mark_jax_checkpoint(checkpoint)
    monkeypatch.setattr(
        "basketworld_jax.eval.fixed_pool_study.load_checkpoint_metadata",
        lambda _path: {
            "env_config": {"players": team_size, "enable_multi_possession": True},
            "trainer_config": {"eval_horizon": 3},
        },
    )
    monkeypatch.setattr(
        "basketworld_jax.eval.fixed_pool_study.load_checkpoint",
        lambda _path: {"params": {}, "policy_spec": {}},
    )

    created = run_fixed_pool_study(
        _args(tmp_path, checkpoint, create=True), native_runner=_fake_native_runner
    )
    assert created["enabled"] is True
    assert created["pool"]["created"] is True
    assert created["schedule"]["created"] is True
    assert created["schedule"]["team_size"] == team_size
    assert len(created["checkpoint_reports"]) == 1
    report = created["checkpoint_reports"][0]
    assert report["raw_artifact_dependencies"]["root"].endswith(
        report["raw_artifact_dependencies"]["study_id"]
    )
    assert report["report_path"].startswith(
        str(tmp_path / "artifacts" / "fixed_pool_reports")
    )
    assert "skill_schema_hash" in report["schema_compatibility"]
    assert "specialist_vs_balanced" in report
    assert report["execution"]["requested_batch_size"] == 2
    assert report["execution"]["actual_batch_size"] == 2
    assert [event["phase"] for event in report["runtime_observation"]["progress_events"]] == [
        "study_dispatch",
        "checkpoint_environment_loading",
        "checkpoint_environment_loaded",
        "checkpoint_restore_loading",
        "checkpoint_restore_loaded",
        "native_initializing",
        "runner_ready",
        "batch_started",
        "batch_completed",
        "native_evaluation_completed",
        "model_fitting_started",
        "model_fitting_completed",
    ]

    reused = run_fixed_pool_study(
        _args(tmp_path, checkpoint, create=False), native_runner=_fake_native_runner
    )
    assert reused["pool"]["created"] is False
    assert reused["schedule"]["created"] is False
    assert reused["pool"]["id"] == created["pool"]["id"]
    assert reused["schedule"]["id"] == created["schedule"]["id"]

    reanalysis_path = write_fixed_pool_model_reanalysis(
        report["raw_artifact_dependencies"]["root"],
        report_root=tmp_path / "reanalysis",
        bootstrap_samples=2,
    )
    reanalysis = json.loads(reanalysis_path.read_text(encoding="utf-8"))
    assert reanalysis["kind"] == "fixed_pool_checkpoint_reanalysis"
    assert reanalysis["reanalysis"]["raw_study_id"] == report[
        "raw_artifact_dependencies"
    ]["study_id"]
    conventional = reanalysis["conventional_three_factor_impact"]
    assert conventional["unavailable_factors"]["free_throw_rate"]["status"] == (
        "unavailable"
    )
    assert conventional["scenario_weights"]["status"] == "excluded"


@pytest.mark.parametrize("team_size", [3, 4, 5])
def test_operational_runner_extends_raw_evidence_without_replaying_base_games(
    tmp_path, monkeypatch, team_size
):
    checkpoint = tmp_path / "update_0000004"
    _mark_jax_checkpoint(checkpoint)
    monkeypatch.setattr(
        "basketworld_jax.eval.fixed_pool_study.load_checkpoint_metadata",
        lambda _path: {
            "env_config": {"players": team_size, "enable_multi_possession": True},
            "trainer_config": {"eval_horizon": 3},
        },
    )
    monkeypatch.setattr(
        "basketworld_jax.eval.fixed_pool_study.load_checkpoint",
        lambda _path: {"params": {}, "policy_spec": {}},
    )
    base = run_fixed_pool_study(
        _args(tmp_path, checkpoint, create=True), native_runner=_fake_native_runner
    )
    extension_args = _args(tmp_path, checkpoint, create=False)
    extension_args.fixed_pool_extension_source = base["checkpoint_reports"][0][
        "raw_artifact_dependencies"
    ]["root"]
    extension_args.fixed_pool_games = 3
    extension_args.fixed_pool_schedule_seed = 9
    extension_args.fixed_pool_schedule_path = str(tmp_path / "inputs" / "extension.json")
    extension_args.fixed_pool_schedule_create = True
    extension_args.fixed_pool_artifact_dir = str(tmp_path / "extension_artifacts")

    base_root = Path(extension_args.fixed_pool_extension_source)
    original_files = {p: p.read_bytes() for p in base_root.iterdir() if p.is_file()}
    extended = run_fixed_pool_study(extension_args, native_runner=_fake_native_runner)
    assert all(p.read_bytes() == content for p, content in original_files.items())
    assert extended["schedule"]["team_size"] == team_size

    report = extended["checkpoint_reports"][0]
    artifact = load_fixed_pool_study_artifacts(report["raw_artifact_dependencies"]["root"])
    assert len(artifact["episodes"]) == 5
    assert artifact["manifest"]["source_studies"][0]["role"] == "base"
    assert artifact["manifest"]["source_studies"][1]["role"] == "additional"
    assert extended["schedule"]["game_count"] == 3
    assert report["execution"]["reproducibility"]["extension_source_root"] == (
        extension_args.fixed_pool_extension_source
    )


@pytest.mark.parametrize(
    "failure", ["mixed", "unsupported", "single_possession", "small_pool"]
)
def test_preflight_failure_does_not_publish_inputs(tmp_path, monkeypatch, failure):
    checkpoint = tmp_path / "checkpoint"
    other = tmp_path / "other"
    _mark_jax_checkpoint(checkpoint)
    _mark_jax_checkpoint(other)
    args = _args(tmp_path, checkpoint, create=True)
    args.fixed_pool_checkpoint.append(str(other))
    args.fixed_pool_size = 5 if failure == "small_pool" else 10
    monkeypatch.setattr(
        "basketworld_jax.eval.fixed_pool_study.load_checkpoint_metadata",
        lambda path: {
            "env_config": {
                "players": (
                    2
                    if failure == "unsupported"
                    else 4 if failure == "mixed" and path == other else 3
                ),
                "enable_multi_possession": failure != "single_possession",
            }
        },
    )
    with pytest.raises(FixedPoolStudyRunError):
        run_fixed_pool_study(
            args, native_runner=lambda **kwargs: pytest.fail("Must not evaluate")
        )
    assert not Path(args.fixed_pool_path).exists()
    assert not Path(args.fixed_pool_schedule_path).exists()
    assert not Path(args.fixed_pool_artifact_dir).exists()


@pytest.mark.parametrize("extension", [False, True])
def test_preflight_rejects_wrong_size_schedule_without_writes(
    tmp_path, monkeypatch, extension
):
    from basketworld_jax.eval.fixed_pool import write_player_pool
    from basketworld_jax.eval.lineup_schedule import (
        build_balanced_lineup_schedule,
        write_lineup_schedule,
    )

    checkpoint = tmp_path / "checkpoint"
    _mark_jax_checkpoint(checkpoint)
    args = _args(tmp_path, checkpoint, create=False)
    pool = generate_player_pool(player_count=10, seed=3)
    write_player_pool(pool, args.fixed_pool_path)
    schedule = build_balanced_lineup_schedule(pool, game_count=2, seed=5)
    write_lineup_schedule(schedule, args.fixed_pool_schedule_path)
    if extension:
        from tests.test_fixed_pool_results import _episode_result

        raw = persist_fixed_pool_study_artifacts(
            pool=pool,
            schedule=schedule,
            episode_results=[_episode_result(pool, schedule, i) for i in range(2)],
            possession_events=[],
            output_dir=tmp_path / "base",
        )
        args.fixed_pool_extension_source = str(raw.root)
        args.fixed_pool_schedule_create = True
        args.fixed_pool_schedule_path = str(tmp_path / "new_schedule.json")
    original_files = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    monkeypatch.setattr(
        "basketworld_jax.eval.fixed_pool_study.load_checkpoint_metadata",
        lambda path: {"env_config": {"players": 4, "enable_multi_possession": True}},
    )
    with pytest.raises(FixedPoolStudyRunError, match="does not match"):
        run_fixed_pool_study(
            args, native_runner=lambda **kwargs: pytest.fail("Must not evaluate")
        )
    assert {
        p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()
    } == original_files
