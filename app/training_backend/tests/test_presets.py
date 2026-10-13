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
    for option in (
        "--shot-contest-skill-mean",
        "--shot-contest-skill-std",
        "--shot-contest-skill-effect-weight",
    ):
        index = preview.argv.index(option)
        assert preview.argv[index + 1] == "0.0"


def test_shortened_run_saves_full_schedule_but_only_executes_in_range(tmp_path):
    schedule = [100, 250, 500, 1000, 2500, 5000, 10000, 30000]
    config = TrainingRunConfig(num_updates=500, historical_eval_updates=schedule)

    def command_schedule(run_config):
        preview = build_halfcourt_multi_possession_preview(
            config=run_config,
            run_id="aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
            repo_root=tmp_path,
            data_dir=tmp_path / "data",
        )
        index = preview.argv.index("--historical-eval-updates")
        return preview.argv[index + 1]

    assert command_schedule(config) == "100,250,500"
    saved = TrainingRunConfig.model_validate_json(config.model_dump_json())
    assert saved.historical_eval_updates == schedule
    extended = saved.model_copy(update={"num_updates": 30000})
    assert command_schedule(extended) == ",".join(map(str, schedule))
    assert config.historical_eval_updates == schedule


@pytest.mark.parametrize("schedule", [[], [1000, 2500]])
def test_no_in_range_explicit_milestones_does_not_fall_back_to_defaults(tmp_path, schedule):
    config = TrainingRunConfig(num_updates=500, historical_eval_updates=schedule)
    preview = build_halfcourt_multi_possession_preview(
        config=config,
        run_id="aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
        repo_root=tmp_path,
        data_dir=tmp_path / "data",
    )
    index = preview.argv.index("--historical-eval-updates")
    assert preview.argv[index + 1] == ""
    assert config.historical_eval_updates == schedule


@pytest.mark.parametrize("milestone", [0, -1])
def test_nonpositive_historical_milestone_is_rejected(milestone):
    with pytest.raises(ValidationError, match="must be positive"):
        TrainingRunConfig(historical_eval_updates=[milestone])


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


def test_preview_compiles_shot_contest_overrides(tmp_path):
    preview = build_halfcourt_multi_possession_preview(
        config=TrainingRunConfig(
            overrides={
                "shot_contest_skill_mean": -0.2,
                "shot_contest_skill_std": 0.7,
                "shot_contest_skill_effect_weight": 1.3,
            }
        ),
        run_id="ffffffff-ffff-4fff-8fff-ffffffffffff",
        repo_root=tmp_path,
        data_dir=tmp_path / "data",
    )

    expected = {
        "--shot-contest-skill-mean": "-0.2",
        "--shot-contest-skill-std": "0.7",
        "--shot-contest-skill-effect-weight": "1.3",
    }
    for option, value in expected.items():
        index = preview.argv.index(option)
        assert preview.argv[index + 1] == value


def test_preview_carries_check_clearance_controls_and_validates_check_mode(tmp_path):
    config = TrainingRunConfig(
        check_deadline_steps=7,
        check_setup_steps=5,
        check_clearance=2,
        random_check_location=False,
    )
    preview = build_halfcourt_multi_possession_preview(
        config=config,
        run_id="12121212-1212-4121-8121-121212121212",
        repo_root=tmp_path,
        data_dir=tmp_path / "data",
    )
    expected = {
        "--check-deadline-steps": "7",
        "--check-setup-steps": "5",
        "--check-clearance": "2",
    }
    for option, value in expected.items():
        index = preview.argv.index(option)
        assert preview.argv[index + 1] == value
    assert "--no-random-check-location" in preview.argv

    with pytest.raises(ValidationError, match="requires made_basket_restart_mode"):
        TrainingRunConfig(
            made_basket_restart_mode="baseline_inbound",
            check_clearance=1,
        )
    with pytest.raises(ValidationError, match="requires at least one"):
        TrainingRunConfig(check_clearance=1, check_setup_steps=0)


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
