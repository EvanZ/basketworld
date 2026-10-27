from __future__ import annotations

from types import SimpleNamespace

from analytics import jax_mcts_policy_regret as cli


def _required_args(tmp_path) -> list[str]:
    return [
        "--evaluated-checkpoint",
        "evaluated",
        "--opponent-checkpoint",
        "opponent",
        "--output-dir",
        str(tmp_path / "study"),
    ]


def test_cli_defaults_match_issue_76_protocol(tmp_path) -> None:
    args = cli._parse_args(_required_args(tmp_path))

    assert args.episodes == 1000
    assert args.search_depths == [1, 3, 5]
    assert args.primary_search_depth == 5
    assert args.simulations_per_action == 512
    assert args.planning_mode == "root-and-full-possession"
    assert args.evaluated_action_mode == "deterministic"
    assert args.opponent_action_mode == "deterministic"
    assert args.calibration is False
    assert args.calibration_states == 50
    assert args.calibration_budgets == [64, 128, 256, 512, 1024]


def test_cli_calibration_commands_reuse_same_seed_and_50_states(
    tmp_path, capsys
) -> None:
    result = cli.main(
        _required_args(tmp_path) + ["--seed", "19", "--print-calibration-command"]
    )

    lines = capsys.readouterr().out.strip().splitlines()
    assert result == 0
    assert len(lines) == 5
    assert all("--episodes 50 --seed 19" in line for line in lines)
    assert all("--planning-mode root-only" in line for line in lines)
    assert [
        int(line.split("--simulations-per-action ", 1)[1].split()[0]) for line in lines
    ] == [64, 128, 256, 512, 1024]


def test_cli_wires_explicit_checkpoints_and_protocol(monkeypatch, tmp_path) -> None:
    observed = {}
    runtime = SimpleNamespace()

    def fake_load(**kwargs):
        observed["load"] = kwargs
        return runtime

    def fake_run(passed_runtime, **kwargs):
        observed["run"] = (passed_runtime, kwargs)
        return {
            "results": str(tmp_path / "study" / "root_action_comparison.json"),
            "aggregate": {
                "primary_search_depth": 3,
                "primary_regret_mean": 0.25,
                "primary_exact_action_agreement_rate": 0.5,
            },
        }

    monkeypatch.setattr(cli, "load_jax_counterfactual_runtime", fake_load)
    monkeypatch.setattr(cli, "run_root_only_study", fake_run)

    result = cli.main(
        _required_args(tmp_path)
        + [
            "--episodes",
            "12",
            "--search-depths",
            "1",
            "3",
            "--primary-search-depth",
            "3",
            "--simulations-per-action",
            "64",
            "--paired-rollouts",
            "20",
            "--evaluated-action-mode",
            "sampled",
            "--sample-metadata",
            str(tmp_path / "source" / "sampled_states.json"),
            "--evaluated-role",
            "defense",
        ]
    )

    assert result == 0
    assert str(observed["load"]["evaluated_checkpoint_path"]) == "evaluated"
    assert str(observed["load"]["opponent_checkpoint_path"]) == "opponent"
    assert observed["load"]["evaluated_action_mode"] == "sampled"
    passed_runtime, run_kwargs = observed["run"]
    assert passed_runtime is runtime
    assert run_kwargs["episode_count"] == 12
    assert run_kwargs["paired_rollout_count"] == 20
    assert run_kwargs["include_full_possession"] is True
    assert (
        run_kwargs["sample_metadata_path"]
        == tmp_path / "source" / "sampled_states.json"
    )
    assert run_kwargs["evaluated_role"] == "defense"
    assert run_kwargs["protocol"].search_depths == (1, 3)
    assert run_kwargs["protocol"].simulations_per_action == 64


def test_cli_runs_calibration_as_one_shared_state_study(monkeypatch, tmp_path) -> None:
    observed = {}
    runtime = SimpleNamespace()

    monkeypatch.setattr(
        cli,
        "load_jax_counterfactual_runtime",
        lambda **kwargs: runtime,
    )

    def fake_calibration(passed_runtime, **kwargs):
        observed["call"] = (passed_runtime, kwargs)
        return {
            "results": str(tmp_path / "calibration" / "budget_calibration.json"),
            "convergence": {
                "main_budget_evidence": {
                    "lower_budget": 512,
                    "upper_budget": 1024,
                    "selected_action_stability_rate": 0.95,
                    "regret_delta": {"mean_absolute": 0.02},
                }
            },
        }

    monkeypatch.setattr(cli, "run_budget_calibration_study", fake_calibration)
    monkeypatch.setattr(
        cli,
        "run_root_only_study",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("root study should not run")
        ),
    )

    result = cli.main(
        _required_args(tmp_path)
        + [
            "--calibration",
            "--calibration-states",
            "12",
            "--calibration-budgets",
            "64",
            "512",
            "1024",
        ]
    )

    assert result == 0
    passed_runtime, kwargs = observed["call"]
    assert passed_runtime is runtime
    assert kwargs["episode_count"] == 12
    assert kwargs["protocol"].calibration_budgets == (64, 512, 1024)
