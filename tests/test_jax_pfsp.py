from types import SimpleNamespace

import numpy as np
import pytest

from basketworld_jax.train.pfsp import PFSPState, opponent_id, probabilities


def candidates():
    return [
        {"info": {"candidate_kind": "self_checkpoint", "update_index": i}}
        for i in (100, 500, 1000)
    ]


@pytest.mark.parametrize("alpha", [0, 0.25, 0.5, 1])
@pytest.mark.parametrize("epsilon", [0, 0.1, 1])
def test_probability_formula(alpha, epsilon):
    w = np.array([0.1, 0.5, 0.9])
    even, hard = w * (1 - w), (1 - w) ** 2
    expected = (1 - epsilon) * (
        (1 - alpha) * even / even.sum() + alpha * hard / hard.sum()
    ) + epsilon / 3
    np.testing.assert_allclose(probabilities(w, alpha, epsilon), expected, rtol=1e-6)


def test_probability_fallbacks_and_validation():
    assert probabilities([], 0.5, 0.1).size == 0
    np.testing.assert_array_equal(probabilities([1], 0, 0), [1])
    np.testing.assert_allclose(probabilities([1, 1], 1, 0), [0.5, 0.5])
    np.testing.assert_allclose(probabilities([0, 1], 0, 0), [0.5, 0.5])
    for value in (float("nan"), float("inf"), -0.1, 1.1):
        with pytest.raises(ValueError):
            probabilities([0.5], value, 0.1)
        with pytest.raises(ValueError):
            probabilities([0.5], 0.5, value)


def test_estimates_modes_age_prior_and_resume():
    state = PFSPState()
    pool = candidates()
    state.observe(
        "self_u100", "sampled", [1, 1, 0.5, 0], update=10, role="offense", truncated=2
    )
    probs, metrics = state.sampling_snapshot(pool, update=10, alpha=0.5, epsilon=0.1)
    assert metrics["pfsp/w/self_u100/sampled"] == pytest.approx(3.5 / 6)
    assert metrics["pfsp/w/self_u100/deterministic"] == 0.5
    assert metrics["pfsp/prior_only/self_u500/sampled"] == 1
    assert metrics["pfsp/truncated_games/self_u100/sampled"] == 2
    assert np.all(probs > 0)
    _, aged = state.sampling_snapshot(pool, update=510, alpha=0.5, epsilon=0.1)
    assert aged["pfsp/w/self_u100/sampled"] == pytest.approx(2.25 / 4)
    assert aged["pfsp/age_updates/self_u100/sampled"] == 500
    restored = PFSPState(state.payload())
    restored_probs, restored_metrics = restored.sampling_snapshot(
        pool, update=510, alpha=0.5, epsilon=0.1
    )
    np.testing.assert_allclose(
        restored_probs,
        state.sampling_snapshot(pool, update=510, alpha=0.5, epsilon=0.1)[0],
    )
    assert restored_metrics == aged
    assert (
        opponent_id(
            {
                "candidate_kind": "self_checkpoint",
                "source": "local_checkpoint",
                "update_index": 100,
            }
        )
        == "self_u100"
    )


def test_terminal_outcomes_use_pre_reset_identity_and_not_rewards():
    state = PFSPState()
    trace = SimpleNamespace(
        active_mask=np.array([[1, 1, 0], [1, 1, 1]]),
        dones=np.array([[0, 1, 1], [1, 1, 1]]),
        truncated=np.array([[0, 0, 0], [0, 1, 0]]),
        opponent_assignment=np.array([[0, 1, 2], [0, 2, -1]]),
        opponent_deterministic_episode=np.array([[0, 1, 0], [0, 0, 0]]),
        terminal_learner_win_score=np.array([[0, 0.5, 1], [1, 0, 1]]),
    )
    state.observe_rollout(trace, candidates(), update=20, role="defense")
    _, metrics = state.sampling_snapshot(
        candidates(), update=21, alpha=0.5, epsilon=0.1
    )
    assert metrics["pfsp/completed_games/self_u100/sampled"] == 1
    assert metrics["pfsp/completed_games/self_u500/deterministic"] == 1
    assert metrics["pfsp/w/self_u500/deterministic"] == 0.5
    assert metrics["pfsp/completed_games/self_u100/deterministic"] == 0
    assert metrics["pfsp/truncated_games/self_u1000/sampled"] == 1
    assert metrics["pfsp/completed_games/self_u1000/sampled"] == 0
    assert metrics["pfsp/defense_start_games/self_u100/sampled"] == 1


def test_config_schema_and_logging_round_trip():
    from basketworld_jax.train.main import (
        parse_args,
        build_train_config_schema,
        validate_train_args,
        _filter_mlflow_train_metrics,
        _filter_mlflow_deploy_metrics,
    )
    from app.training_backend.config_schema import compile_overrides

    args = parse_args(
        [
            "--enable-multi-possession",
            "--opponent-pool-sampling",
            "pfsp",
            "--opponent-pfsp-alpha",
            "0.8",
            "--opponent-pfsp-epsilon",
            "0.2",
        ]
    )
    validate_train_args(args)
    schema = build_train_config_schema(args)
    fields = {row["name"]: row for row in schema}
    names = ("opponent_pool_sampling", "opponent_pfsp_alpha", "opponent_pfsp_epsilon")
    overrides = {name: fields[name]["effective_value"] for name in names}
    restored = parse_args(compile_overrides(overrides, schema))
    assert [getattr(restored, name) for name in names] == ["pfsp", 0.8, 0.2]
    assert parse_args([]).opponent_pool_sampling == "recency"
    for value in ("nan", "inf", "-1", "2"):
        with pytest.raises(SystemExit, match="PFSP"):
            validate_train_args(parse_args(["--opponent-pfsp-alpha", value]))
    metrics = {
        "pfsp/w/self_u100/sampled": 0.7,
        "pressure/learner_defending/mean_ball_handler_defended_esv": 0.5,
    }
    assert _filter_mlflow_train_metrics(metrics) == metrics
    assert _filter_mlflow_deploy_metrics(metrics) == {
        k: v for k, v in metrics.items() if k.startswith("pressure/")
    }


def test_checkpoint_payload_retains_estimator_and_sampling_configuration():
    from basketworld_jax.checkpoints import build_checkpoint_payload

    state = PFSPState()
    state.observe("self_u100", "sampled", [1], update=3, role="offense")
    config = {"mode": "pfsp", "alpha": 0.5, "epsilon": 0.1}
    payload = build_checkpoint_payload(
        update_index=3,
        trainer_config={},
        policy_spec={},
        frozen_config={},
        params={},
        opt_state={},
        current_state={},
        eval_initial_state={},
        base_key=None,
        eval_trajectories=[],
        last_metrics=None,
        opponent_pool_state={"pfsp_state": state.payload(), "sampling_config": config},
    )
    assert payload["opponent_pool_state"]["pfsp_state"] == state.payload()
    assert payload["opponent_pool_state"]["sampling_config"] == config


def test_evidence_cap_combines_both_sides_before_discounting():
    state = PFSPState()
    traces = {}
    for role, score in (("offense", 1.0), ("defense", 0.0)):
        traces[role] = SimpleNamespace(
            active_mask=np.ones((1, 300)),
            dones=np.ones((1, 300)),
            truncated=np.zeros((1, 300)),
            opponent_assignment=np.zeros((1, 300)),
            opponent_deterministic_episode=np.zeros((1, 300)),
            terminal_learner_win_score=np.full((1, 300), score),
        )
    state.observe_rollouts(
        traces,
        candidates(),
        update=100,
    )
    _, metrics = state.sampling_snapshot(
        candidates(), update=100, alpha=0.5, epsilon=0.1
    )
    assert metrics["pfsp/w/self_u100/sampled"] == 0.5
    assert metrics["pfsp/effective_games/self_u100/sampled"] == 256
    assert metrics["pfsp/completed_games/self_u100/sampled"] == 600
    assert metrics["pfsp/offense_start_games/self_u100/sampled"] == 300
    assert metrics["pfsp/defense_start_games/self_u100/sampled"] == 300


def test_dynamic_opponent_metrics_are_batched_without_dropping_values():
    from basketworld_jax.train.main import _log_mlflow_metrics

    calls = []
    client = SimpleNamespace(
        log_metrics=lambda values, step: calls.append((values, step))
    )
    metrics = {f"pfsp/w/opponent_{index}/sampled": 0.5 for index in range(510)}
    metrics["unavailable"] = None
    _log_mlflow_metrics(client, metrics, step=50, prefix="jax/train")
    assert len(calls) == 2
    assert sum(len(values) for values, _ in calls) == 510
    assert all(step == 50 for _, step in calls)
    assert calls[0][0]["jax/train/pfsp/w/opponent_0/sampled"] == 0.5
