from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.training_backend.models import TrainingRunConfig
from app.training_backend.presets import (
    build_halfcourt_multi_possession_preview,
    format_display_command,
)


def test_short_run_filters_default_historical_milestones(tmp_path):
    config = TrainingRunConfig(num_updates=500)
    preview = build_halfcourt_multi_possession_preview(
        config=config,
        run_id="aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
        repo_root=tmp_path,
        data_dir=tmp_path / "data",
    )

    historical_index = preview.argv.index("--historical-eval-updates")
    assert preview.argv[historical_index + 1] == "100,500"
    assert preview.argv[:3] == [
        str(tmp_path / ".env" / "bin" / "python"),
        "-m",
        "basketworld_jax.train.main",
    ]
    assert preview.environment["MLFLOW_TRACKING_URI"] == "http://localhost:5000"
    experiment_index = preview.argv.index("--mlflow-experiment-name")
    assert preview.argv[experiment_index + 1] == "halfcourt_multi_possessions"
    assert "--control-file" in preview.argv
    assert "--training-status-file" in preview.argv


def test_explicit_historical_milestone_after_target_is_rejected():
    with pytest.raises(ValidationError, match="cannot exceed num_updates"):
        TrainingRunConfig(num_updates=500, historical_eval_updates=[100, 1000])


def test_historical_eval_episode_count_must_be_even():
    with pytest.raises(ValidationError, match="must be even"):
        TrainingRunConfig(historical_eval_episodes=101)


def test_preview_keeps_run_name_as_single_argv_value(tmp_path):
    config = TrainingRunConfig(name="experiment; touch nope")
    preview = build_halfcourt_multi_possession_preview(
        config=config,
        run_id="bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
        repo_root=tmp_path,
        data_dir=tmp_path / "data",
    )
    index = preview.argv.index("--mlflow-run-name")
    assert preview.argv[index + 1] == "experiment; touch nope"


def test_preview_persists_explicit_mlflow_location(tmp_path):
    config = TrainingRunConfig(
        mlflow_tracking_uri="http://mlflow.internal:5050",
        mlflow_experiment_name="halfcourt_cpu_runs",
    )
    preview = build_halfcourt_multi_possession_preview(
        config=config,
        run_id="dddddddd-dddd-4ddd-8ddd-dddddddddddd",
        repo_root=tmp_path,
        data_dir=tmp_path / "data",
    )

    assert preview.environment["MLFLOW_TRACKING_URI"] == ("http://mlflow.internal:5050")
    experiment_index = preview.argv.index("--mlflow-experiment-name")
    assert preview.argv[experiment_index + 1] == "halfcourt_cpu_runs"
    assert "--mlflow-experiment-name halfcourt_cpu_runs" in preview.display_command


def test_preview_compiles_typed_overrides(tmp_path):
    preview = build_halfcourt_multi_possession_preview(
        config=TrainingRunConfig(
            overrides={"gamma": 0.97, "single_episode_rollouts": True}
        ),
        run_id="eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee",
        repo_root=tmp_path,
        data_dir=tmp_path / "data",
    )

    assert "--gamma" in preview.argv
    assert preview.argv[preview.argv.index("--gamma") + 1] == "0.97"
    assert "--single-episode-rollouts" in preview.argv


def test_resume_uses_full_state_checkpoint_without_reset_flags(tmp_path):
    preview = build_halfcourt_multi_possession_preview(
        config=TrainingRunConfig(num_updates=5000),
        run_id="cccccccc-cccc-4ccc-8ccc-cccccccccccc",
        repo_root=tmp_path,
        data_dir=tmp_path / "data",
        resume_checkpoint="/tmp/checkpoints/latest",
        mlflow_resume_run_id="original-mlflow-run-id",
    )
    resume_index = preview.argv.index("--resume-checkpoint")
    assert preview.argv[resume_index + 1] == "/tmp/checkpoints/latest"
    mlflow_index = preview.argv.index("--mlflow-resume-run-id")
    assert preview.argv[mlflow_index + 1] == "original-mlflow-run-id"
    assert "--resume-reset-env-state" not in preview.argv
    assert "--resume-reset-intent-discriminator-state" not in preview.argv


def test_display_command_shell_quotes_environment_and_arguments():
    command = format_display_command(
        ["/repo/.env/bin/python", "-m", "trainer", "--name", "a run; safe"],
        {
            "PYTHONPATH": "/repo path",
            "MLFLOW_TRACKING_URI": "http://localhost:5000",
        },
    )

    assert "MLFLOW_TRACKING_URI=http://localhost:5000" in command
    assert "PYTHONPATH='/repo path'" in command
    assert "--name 'a run; safe'" in command
