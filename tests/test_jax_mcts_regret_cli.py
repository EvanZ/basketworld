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
    assert args.evaluated_action_mode == "deterministic"
    assert args.opponent_action_mode == "deterministic"


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
    assert run_kwargs["protocol"].search_depths == (1, 3)
    assert run_kwargs["protocol"].simulations_per_action == 64
