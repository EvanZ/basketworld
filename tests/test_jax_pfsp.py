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
    assert metrics["pfsp/wins/self_u100/sampled"] == 2
    assert metrics["pfsp/win_rate_games/self_u100/sampled"] == 4
    assert metrics["pfsp/win_rate/self_u100/sampled"] == 0.5
    assert "pfsp/win_rate/self_u500/sampled" not in metrics
    assert np.all(probs > 0)
    _, aged = state.sampling_snapshot(pool, update=510, alpha=0.5, epsilon=0.1)
    assert aged["pfsp/w/self_u100/sampled"] == pytest.approx(2.25 / 4)
    assert aged["pfsp/age_updates/self_u100/sampled"] == 500
    assert aged["pfsp/win_rate_games/self_u100/sampled"] == 4
    assert aged["pfsp/win_rate/self_u100/sampled"] == 0.5
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
    assert metrics["pfsp/win_rate/self_u100/sampled"] == 1
    assert metrics["pfsp/win_rate/self_u500/deterministic"] == 0
    assert metrics["pfsp/win_rate_games/self_u1000/sampled"] == 0


def test_legacy_resume_tracks_only_known_win_outcomes_without_changing_sampling():
    state = PFSPState()
    state.observe("self_u100", "sampled", [1, 1, 0.5, 0], update=10, role="offense")
    expected_probs, _ = state.sampling_snapshot(
        candidates(), update=11, alpha=0.5, epsilon=0.1
    )
    legacy = state.payload()
    for modes in legacy["entries"].values():
        for entry in modes.values():
            del entry["wins"], entry["win_rate_games"]
    restored = PFSPState(legacy)
    probs, metrics = restored.sampling_snapshot(
        candidates(), update=11, alpha=0.5, epsilon=0.1
    )
    np.testing.assert_array_equal(probs, expected_probs)
    assert metrics["pfsp/completed_games/self_u100/sampled"] == 4
    assert metrics["pfsp/win_rate_games/self_u100/sampled"] == 0
    assert "pfsp/win_rate/self_u100/sampled" not in metrics
    restored.observe("self_u100", "sampled", [1, 0.5, 0], update=11, role="defense")
    _, metrics = PFSPState(restored.payload()).sampling_snapshot(
        candidates(), update=12, alpha=0.5, epsilon=0.1
    )
    assert metrics["pfsp/completed_games/self_u100/sampled"] == 7
    assert metrics["pfsp/wins/self_u100/sampled"] == 1
    assert metrics["pfsp/win_rate_games/self_u100/sampled"] == 3
    assert metrics["pfsp/win_rate/self_u100/sampled"] == pytest.approx(1 / 3)


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
        "pfsp/win_rate/self_u100/sampled": 0.6,
        "pfsp/wins/self_u100/sampled": 60,
        "pfsp/win_rate_games/self_u100/sampled": 100,
        "pfsp/ppp_scored/self_u100/sampled": 1.2,
        "pfsp/ppp_allowed/self_u100/sampled": 0.9,
        "pfsp/ppp_offensive_possessions/self_u100/sampled": 200,
        "pfsp/current_ppp_scored/self_u100/sampled": 1.1,
        "pfsp/current_ppp_update/self_u100/sampled": 100,
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
    assert metrics["pfsp/wins/self_u100/sampled"] == 300
    assert metrics["pfsp/win_rate_games/self_u100/sampled"] == 600
    assert metrics["pfsp/win_rate/self_u100/sampled"] == 0.5


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


def possession_trace(*, assignments, modes, active, attacking, ended, own, other, role):
    shape = (1, len(assignments))
    return SimpleNamespace(
        active_mask=np.array([active]),
        opponent_assignment=np.array([assignments]),
        opponent_deterministic_episode=np.array([modes]),
        training_role=np.array([attacking]),
        possession_ended=np.array([ended]),
        offense_score_delta=np.array([own if role == "offense" else other]),
        defense_score_delta=np.array([other if role == "offense" else own]),
        # No completed games: PPP must still use observed possession results.
        dones=np.zeros(shape),
        truncated=np.zeros(shape),
        terminal_learner_win_score=np.zeros(shape),
    )


def test_opponent_ppp_uses_actual_roles_scores_and_pre_reset_assignments():
    traces = {
        "offense": possession_trace(
            assignments=[0, 0, 0, 1, 1, -1, 0],
            modes=[0, 0, 0, 1, 1, 0, 0],
            active=[1, 1, 1, 1, 1, 1, 0],
            attacking=[1, 1, -1, -1, 1, 1, 1],
            ended=[1, 0, 1, 1, 0, 1, 1],
            # The 1 point while defending represents an awarded violation point.
            own=[2, 0, 1, 0, 0, 99, 99],
            other=[0, 0, 3, 2, 0, 99, 99],
            role="offense",
        ),
        "defense": possession_trace(
            assignments=[0, 0, 1],
            modes=[0, 0, 1],
            active=[1, 1, 1],
            attacking=[-1, 1, 1],
            ended=[1, 1, 1],
            own=[0, 3, 0],
            other=[2, 0, 0],
            role="defense",
        ),
    }
    # Completed possessions in a truncated game still count, matching training PPP.
    traces["defense"].dones[0, 0] = 1
    traces["defense"].truncated[0, 0] = 1
    state = PFSPState()
    current = state.observe_rollouts(traces, candidates(), update=10)
    probs, metrics = state.sampling_snapshot(
        candidates(), update=11, alpha=0.5, epsilon=0.1
    )
    expected = {
        "ppp_points_scored": 6,
        "ppp_points_allowed": 5,
        "ppp_offensive_possessions": 2,
        "ppp_defensive_possessions": 2,
        "ppp_scored": 3,
        "ppp_allowed": 2.5,
        "ppp_tracking_start_update": 10,
    }
    for key, value in expected.items():
        assert metrics[f"pfsp/{key}/self_u100/sampled"] == value
        if key != "ppp_tracking_start_update":
            assert current[f"pfsp/current_{key}/self_u100/sampled"] == value
    assert current["pfsp/current_ppp_update/self_u100/sampled"] == 10
    assert current["pfsp/current_ppp_scored/self_u500/deterministic"] == 0
    assert current["pfsp/current_ppp_allowed/self_u500/deterministic"] == 2
    assert current["pfsp/current_ppp_offensive_possessions/self_u1000/sampled"] == 0
    assert "pfsp/current_ppp_scored/self_u1000/sampled" not in current
    assert metrics["pfsp/ppp_scored/self_u500/deterministic"] == 0
    assert metrics["pfsp/ppp_allowed/self_u500/deterministic"] == 2
    assert metrics["pfsp/win_rate_games/self_u100/sampled"] == 0
    assert "pfsp/ppp_scored/self_u1000/sampled" not in metrics
    assert metrics["pfsp/ppp_tracking_start_update/self_u1000/sampled"] == -1
    # No diagnostic-only totals may change the sampler.
    np.testing.assert_array_equal(probs, np.full((2, 3), 1 / 3, dtype=np.float32))

    restored = PFSPState(state.payload())
    assert (
        restored.sampling_snapshot(candidates(), update=11, alpha=0.5, epsilon=0.1)[1]
        == metrics
    )
    latest = restored.observe_rollout(
        possession_trace(
            assignments=[0],
            modes=[0],
            active=[1],
            attacking=[1],
            ended=[1],
            own=[0],
            other=[0],
            role="offense",
        ),
        candidates(),
        update=12,
        role="offense",
    )
    _, updated = restored.sampling_snapshot(
        candidates(), update=13, alpha=0.5, epsilon=0.1
    )
    assert (
        updated["pfsp/ppp_scored/self_u100/sampled"] == 2
    )  # 6 / 3, not a mean of rollout ratios.
    assert updated["pfsp/ppp_allowed/self_u100/sampled"] == 2.5
    assert updated["pfsp/ppp_tracking_start_update/self_u100/sampled"] == 10
    assert latest["pfsp/current_ppp_update/self_u100/sampled"] == 12
    assert latest["pfsp/current_ppp_scored/self_u100/sampled"] == 0
    assert latest["pfsp/current_ppp_offensive_possessions/self_u100/sampled"] == 1
    assert latest["pfsp/current_ppp_defensive_possessions/self_u100/sampled"] == 0
    assert "pfsp/current_ppp_allowed/self_u100/sampled" not in latest
    # No stale contribution from the previous update or another opponent/mode.
    assert latest["pfsp/current_ppp_points_allowed/self_u500/deterministic"] == 0
    assert "pfsp/current_ppp_allowed/self_u500/deterministic" not in latest


def test_current_ppp_empty_candidates_and_legacy_traces_are_unavailable():
    state = PFSPState()
    assert state.observe_rollouts({}, [], update=1) == {}
    assert state.observe_rollouts({}, candidates(), update=1) == {}


def test_legacy_resume_initializes_only_new_possession_counters():
    state = PFSPState()
    state.observe("self_u100", "sampled", [1, 0], update=10, role="offense")
    legacy = state.payload()
    del legacy["entries"]["self_u100"]["sampled"]["possession_stats"]
    restored = PFSPState(legacy)
    _, empty = restored.sampling_snapshot(
        candidates(), update=20, alpha=0.5, epsilon=0.1
    )
    assert "pfsp/ppp_allowed/self_u100/sampled" not in empty
    assert empty["pfsp/win_rate_games/self_u100/sampled"] == 2
    # Points can be observed before an offensive possession ends across a rollout boundary.
    trace = possession_trace(
        assignments=[0],
        modes=[0],
        active=[1],
        attacking=[1],
        ended=[0],
        own=[1],
        other=[0],
        role="offense",
    )
    restored.observe_rollout(trace, candidates(), update=20, role="offense")
    _, pending = restored.sampling_snapshot(
        candidates(), update=21, alpha=0.5, epsilon=0.1
    )
    assert pending["pfsp/ppp_points_scored/self_u100/sampled"] == 1
    assert "pfsp/ppp_scored/self_u100/sampled" not in pending
    again = PFSPState(restored.payload())
    trace.offense_score_delta[0, 0] = 0
    trace.possession_ended[0, 0] = 1
    again.observe_rollout(trace, candidates(), update=21, role="offense")
    _, metrics = again.sampling_snapshot(
        candidates(), update=22, alpha=0.5, epsilon=0.1
    )
    assert metrics["pfsp/ppp_scored/self_u100/sampled"] == 1
    assert "pfsp/ppp_allowed/self_u100/sampled" not in metrics
    assert metrics["pfsp/ppp_tracking_start_update/self_u100/sampled"] == 20
    assert metrics["pfsp/win_rate_games/self_u100/sampled"] == 2
