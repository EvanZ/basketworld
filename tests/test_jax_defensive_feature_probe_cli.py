from __future__ import annotations

from analytics import jax_defensive_feature_probe as cli


def test_cli_defaults_match_issue_77_study() -> None:
    args = cli._parse_args(
        [
            "--mcts-results",
            "root_action_comparison.json",
            "--output-dir",
            "probe-output",
        ]
    )

    assert args.training_seeds == [1, 2, 3]
    assert args.train_fraction == 0.70
    assert args.validation_fraction == 0.15
    assert args.learning_curve_fractions == [0.25, 0.50, 1.0]
    assert args.stability_result == []


def test_cli_wires_source_stability_and_training_config(monkeypatch, tmp_path) -> None:
    observed = {}

    def fake_run(results_path, **kwargs):
        observed["call"] = (results_path, kwargs)
        return {
            "results": str(tmp_path / "output" / "defensive_feature_probe.json"),
            "payload": {
                "feature_variants": {
                    name: {
                        "metrics": {
                            "test": {
                                "exact_player_action_accuracy": {
                                    "estimate": 0.4,
                                    "ci95_low": 0.3,
                                    "ci95_high": 0.5,
                                },
                                "negative_log_likelihood": {"estimate": 1.5},
                                "legal_action_macro_roc_auc": {"estimate": 0.6},
                                "legal_action_macro_pr_auc": {"estimate": 0.3},
                            }
                        }
                    }
                    for name in (
                        "current_observation",
                        "directional_geometry",
                        "threat_and_responsibility",
                    )
                }
            },
        }

    monkeypatch.setattr(cli, "run_defensive_feature_probe", fake_run)

    result = cli.main(
        [
            "--mcts-results",
            str(tmp_path / "source.json"),
            "--output-dir",
            str(tmp_path / "output"),
            "--stability-result",
            str(tmp_path / "budget-256.json"),
            "--training-seeds",
            "4",
            "5",
            "--epochs",
            "12",
            "--learning-curve-fractions",
            "0.5",
            "1.0",
        ]
    )

    assert result == 0
    results_path, kwargs = observed["call"]
    assert results_path == tmp_path / "source.json"
    assert kwargs["output_dir"] == tmp_path / "output"
    assert kwargs["stability_result_paths"] == [tmp_path / "budget-256.json"]
    assert kwargs["config"].training_seeds == (4, 5)
    assert kwargs["config"].epochs == 12
    assert kwargs["config"].learning_curve_fractions == (0.5, 1.0)
