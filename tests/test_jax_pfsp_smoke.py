"""Small compiled CPU exercise; no MLflow or active-run mutation."""

import numpy as np
import pytest


@pytest.mark.parametrize("snapshot_interval", [0, 2])
def test_compiled_pfsp_training_resume_and_metrics(tmp_path, snapshot_interval):
    jax = pytest.importorskip("jax")
    from basketworld_jax.train.main import parse_args, run_training_loop
    from basketworld_jax.checkpoints import load_checkpoint

    common = [
        "--run-train-loop",
        "--enable-multi-possession",
        "--players",
        "2",
        "--multi-possession-limit",
        "2",
        "--multi-possession-overtime-round-cap",
        "0",
        "--shot-clock",
        "1",
        "--min-shot-clock",
        "1",
        "--check-setup-steps",
        "0",
        "--kernel-batch-size",
        "2",
        "--rollout-horizon",
        "32",
        "--policy-hidden-dims",
        "16",
        "--policy-update-epochs",
        "1",
        "--ppo-minibatches",
        "1",
        "--checkpoint-every-updates",
        "0" if snapshot_interval else "1",
        "--opponent-snapshot-every-updates",
        str(snapshot_interval),
        "--opponent-pool-size",
        "2",
        "--log-every-updates",
        "1",
        "--eval-every-updates",
        "0",
        "--eval-deploy-every-updates",
        "0",
        "--eval-deploy-batches",
        "0",
        "--opponent-pool-sampling",
        "pfsp",
        "--no-progress",
    ]
    if snapshot_interval:
        common += [
            "--policy-model",
            "attention",
            "--team-action-mode",
            "autoregressive",
            "--attention-embed-dim",
            "8",
            "--attention-token-mlp-dim",
            "8",
        ]
    directory = tmp_path / "initial"
    target_update = 5 if snapshot_interval else 3
    first = run_training_loop(
        parse_args(
            [
                *common,
                "--checkpoint-dir",
                str(directory),
                "--num-updates",
                str(target_update),
            ]
        )
    )
    saved = load_checkpoint(directory / "latest")
    state = saved["opponent_pool_state"]["pfsp_state"]
    assert (
        sum(row["sampled"]["completed_games"] for row in state["entries"].values()) > 0
    )
    assert all(
        row["deterministic"]["completed_games"] == 0
        for row in state["entries"].values()
    )
    assert np.isfinite(first["final_metrics"]["total_loss"])
    assert any(
        key.startswith("pressure/learner_defending/") for key in first["final_metrics"]
    )
    assert any(key.startswith("pfsp/w/") for key in first["final_metrics"])
    if snapshot_interval:
        assert sorted(path.name for path in directory.iterdir()) == [
            "latest",
            "update_0000005",
        ]
        pool = saved["opponent_pool_state"]
        assert len(pool["candidate_infos"]) == 2
        assert all(info["update_index"] != 5 for info in pool["candidate_infos"])
        assert pool["cache_state"]["capacity"] == 2
        assert "opponent_cache/usage_percent/self_u1" in first["final_metrics"]

    # Resume the exact same saved game/RNG/pool twice: resulting policies,
    # assignments, and estimator histories must agree, not just config values.
    resumed = []
    for name in ("resume_a", "resume_b"):
        target = tmp_path / name
        run_training_loop(
            parse_args(
                [
                    *common,
                    "--checkpoint-dir",
                    str(target),
                    "--num-updates",
                    str(target_update + 1),
                    "--resume-checkpoint",
                    str(directory / "latest"),
                ]
            )
        )
        resumed.append(load_checkpoint(target / "latest"))
    assert (
        resumed[0]["opponent_pool_state"]["pfsp_state"]
        == resumed[1]["opponent_pool_state"]["pfsp_state"]
    )
    assert resumed[0]["opponent_pool_state"].get("cache_state") == resumed[1][
        "opponent_pool_state"
    ].get("cache_state")
    for left, right in zip(
        jax.tree_util.tree_leaves(
            resumed[0]["opponent_pool_state"]["candidate_params"]
        ),
        jax.tree_util.tree_leaves(
            resumed[1]["opponent_pool_state"]["candidate_params"]
        ),
        strict=True,
    ):
        np.testing.assert_array_equal(left, right)
    for left, right in zip(
        jax.tree_util.tree_leaves(resumed[0]["params"]),
        jax.tree_util.tree_leaves(resumed[1]["params"]),
        strict=True,
    ):
        np.testing.assert_array_equal(left, right)
    for role in ("offense", "defense"):
        np.testing.assert_array_equal(
            resumed[0]["opponent_pool_state"]["assignments"][role],
            resumed[1]["opponent_pool_state"]["assignments"][role],
        )
