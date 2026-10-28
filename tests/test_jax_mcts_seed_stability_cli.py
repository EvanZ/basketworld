from __future__ import annotations

from analytics import jax_mcts_seed_stability as cli


def test_seed_stability_cli_wires_inputs_and_bootstrap(monkeypatch, tmp_path) -> None:
    observed = {}

    def fake_run(result_paths, **kwargs):
        observed["call"] = (result_paths, kwargs)
        return {
            "results": str(tmp_path / "output" / "mcts_search_seed_stability.json"),
            "payload": {
                "summary": {
                    "run_count": 3,
                    "state_count": 50,
                    "pairwise_player_action_agreement": {
                        "estimate": 0.7,
                        "ci95_low": 0.6,
                        "ci95_high": 0.8,
                    },
                    "pairwise_joint_action_agreement": {
                        "estimate": 0.2,
                        "ci95_low": 0.1,
                        "ci95_high": 0.3,
                    },
                }
            },
        }

    monkeypatch.setattr(cli, "run_search_seed_stability_study", fake_run)

    result = cli.main(
        [
            "--result",
            str(tmp_path / "seed-1.json"),
            "--result",
            str(tmp_path / "seed-2.json"),
            "--output-dir",
            str(tmp_path / "output"),
            "--depth",
            "5",
            "--bootstrap-samples",
            "100",
            "--bootstrap-seed",
            "19",
        ]
    )

    assert result == 0
    paths, kwargs = observed["call"]
    assert paths == [tmp_path / "seed-1.json", tmp_path / "seed-2.json"]
    assert kwargs["output_dir"] == tmp_path / "output"
    assert kwargs["depth"] == 5
    assert kwargs["bootstrap_samples"] == 100
    assert kwargs["bootstrap_seed"] == 19
