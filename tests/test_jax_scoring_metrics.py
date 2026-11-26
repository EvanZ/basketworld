from types import SimpleNamespace as NS

import numpy as np
import pytest

from basketworld_jax.scoring_metrics import (
    summarize_training_scoring,
    summarize_learner_games,
    summarize_self_play_scoring,
    separate_scoring_diagnostics,
    read_native_scoring_summary,
)


def rollout(
    *,
    learner_a=True,
    truncated=False,
    terminal_scores=(12, 8),
    points=(4, 1),
    starter=True,
    possession_roles=(1, -1),
):
    a, b = points if learner_a else points[::-1]
    # Only the final two steps of a longer game are in this rollout.
    t = NS(
        active_mask=np.ones((2, 1)),
        terminal_episode_steps=np.array([[0], [30]]),
        truncated=np.array([[False], [truncated]]),
        terminal_learner_score=np.array([[0], [terminal_scores[0]]]),
        terminal_opponent_score=np.array([[0], [terminal_scores[1]]]),
        learner_started_with_possession=np.array([[starter], [starter]]),
        team_a_score_delta=np.array([[a], [0]]),
        team_b_score_delta=np.array([[0], [b]]),
        possession_ended=np.ones((2, 1)),
        training_role=np.asarray(possession_roles).reshape(2, 1),
    )
    return NS(trajectory=t)


def test_completed_game_scores_use_full_terminal_score_not_observed_fragment():
    result = summarize_training_scoring({"offense": rollout()})
    assert result["learner_points_scored_per_completed_game"] == 12
    assert result["learner_points_allowed_per_completed_game"] == 8
    assert (
        result["learner_completed_game_points_scored"]
        + result["learner_completed_game_points_allowed"]
        == 20
    )
    assert result["learner_points_scored_observed"] == 4
    assert result["learner_points_allowed_observed"] == 1  # violation point included
    assert result["learner_points_allowed_per_completed_defensive_possession"] == 1


def test_high_scoring_truncation_does_not_change_completed_games_but_contributes_to_observed_ppp():
    base = summarize_training_scoring({"offense": rollout()})
    mixed = summarize_training_scoring(
        {
            "offense": rollout(),
            "defense": rollout(
                learner_a=False,
                truncated=True,
                terminal_scores=(999, 500),
                points=(8, 3),
            ),
        }
    )
    for key in (
        "learner_points_scored_per_completed_game",
        "learner_points_allowed_per_completed_game",
        "learner_completed_game_win_rate",
        "learner_completed_game_win_score",
    ):
        assert mixed[key] == base[key]
    assert mixed["completed_game_count"] == 1
    assert mixed["truncated_game_count"] == 1
    assert mixed["learner_points_scored_per_completed_offensive_possession"] == 6


def test_combined_ppp_is_ratio_of_totals_with_unequal_lane_denominators():
    a = rollout(points=(10, 2), possession_roles=(1, -1))
    b = rollout(learner_a=False, points=(4, 2), possession_roles=(1, 1))
    result = summarize_training_scoring({"offense": a, "defense": b})
    assert result[
        "learner_points_scored_per_completed_offensive_possession"
    ] == pytest.approx(14 / 3)
    assert (
        result["learner_points_scored_per_completed_offensive_possession"]
        != (10 + 2) / 2
    )


def test_roster_swap_preserves_learner_outcomes_and_swaps_assignment_diagnostics():
    a = summarize_training_scoring({"offense": rollout(starter=False)})
    b = summarize_training_scoring({"defense": rollout(learner_a=False, starter=False)})
    for key in (
        "learner_score_margin_per_completed_game",
        "learner_completed_game_win_score",
        "learner_points_scored_per_completed_offensive_possession",
    ):
        assert a[key] == b[key]
    assert (
        a["learner_roster_assignment_a_completed_game_count"]
        == b["learner_roster_assignment_b_completed_game_count"]
        == 1
    )
    assert (
        a["opponent_started_with_possession_completed_game_count"]
        == b["opponent_started_with_possession_completed_game_count"]
        == 1
    )


def test_actual_opening_possession_and_ties_use_their_own_counts():
    result = summarize_training_scoring(
        {
            "offense": rollout(starter=False, terminal_scores=(8, 8)),
            "defense": rollout(learner_a=False, starter=True),
        }
    )
    assert result["learner_started_with_possession_completed_game_count"] == 1
    assert result["opponent_started_with_possession_completed_game_count"] == 1
    assert result["opening_possession_score_margin_delta"] == 4
    assert result["learner_completed_game_win_rate"] == 0.5
    assert result["learner_completed_game_win_score"] == 0.75


def test_no_completed_games_or_possessions_gives_gaps():
    empty = summarize_learner_games([])
    assert empty["completed_game_count"] == 0
    assert empty["learner_points_scored_per_completed_game"] is None
    assert empty["learner_completed_game_win_rate"] is None
    deploy = summarize_self_play_scoring(0, 0, 0, 0, 0, 5)
    assert deploy["self_play_points_per_completed_offensive_possession"] is None


def test_deploy_uses_only_completed_final_scores_and_ratio_of_possession_totals():
    from basketworld_jax.train.runtime import summarize_deploy_eval_outputs
    from basketworld_jax.train.types import DeployEvalOutput, DeployEvalTotals
    from basketworld_jax.env.minimal import ReboundDiagnosticTotals
    from basketworld_jax.shot_defense_metrics import SHOT_DEFENSE_SHAPE

    totals = {field: np.zeros(3) for field in DeployEvalTotals._fields}
    totals.update(
        shot_defense_diagnostics=np.zeros(SHOT_DEFENSE_SHAPE),
        team_a_completed_offensive_possessions=np.array([1, 100, 100]),
        team_b_completed_offensive_possessions=np.array([3, 100, 100]),
    )
    output = DeployEvalOutput(
        final_state=NS(
            episode_ended=np.array([True, True, False]),
            episode_truncated=np.array([False, True, False]),
            offense_score=np.array([6, 999, 999]),
            defense_score=np.array([2, 999, 999]),
        ),
        totals=DeployEvalTotals(**totals),
        rebound_diagnostic_totals=ReboundDiagnosticTotals(
            **{f: 0 for f in ReboundDiagnosticTotals._fields}
        ),
    )
    result = summarize_deploy_eval_outputs([output], batch_size=3)
    assert result["self_play_points_per_team_per_completed_game"] == 4
    assert result["self_play_points_per_completed_offensive_possession"] == 2
    assert result["completed_game_count"] == 1 and result["truncated_game_count"] == 2
    assert "mean_offense_score" not in result
    assert "clock_regulation_segment_1_team_a_points" not in result
    # Paired identical-policy games swapping roster scores have exactly zero assignment margin.
    paired = summarize_self_play_scoring(8, 8, 4, 4, 2, 0)
    assert paired["roster_assignment_score_margin_per_completed_game"] == 0


def test_retired_keys_are_removed_and_accounting_is_artifact_only_in_both_profiles():
    from basketworld_jax.train.main import _filter_mlflow_train_metrics

    metrics = {
        "offense_learner_points_per_completed_episode": 99,
        "team_a_score_delta_total": 12,
        "offense_regulation_segment_1_team_b_points": 4,
        "defense_ppo_eligible_learner_points_total": 3,
        "learner_points_scored_per_completed_game": 7,
        "mean_reward": 0.1,
    }
    clean = separate_scoring_diagnostics(metrics)
    assert clean["scoring_diagnostics"]["roster_slot_a_points_observed"] == 12
    assert "offense_learner_points_per_completed_episode" not in clean
    for profile in ("core", "full"):
        filtered = _filter_mlflow_train_metrics(clean, profile=profile)
        assert filtered == {
            "learner_points_scored_per_completed_game": 7,
            "mean_reward": 0.1,
        }


def test_legacy_native_reader_preserves_input_and_reads_new_explicit_ppp_names():
    old = {"user_points_per_possession": 1.2, "opponent_points_per_possession": 0.9}
    migrated = read_native_scoring_summary(old)
    assert migrated["user_points_per_completed_offensive_possession"] == 1.2
    assert "user_points_per_completed_offensive_possession" not in old
    assert (
        read_native_scoring_summary(
            {"user_points_per_completed_offensive_possession": 2, **old}
        )["user_points_per_completed_offensive_possession"]
        == 2
    )


@pytest.mark.parametrize("team", [0, 1])
def test_real_violation_scoreboard_is_attributed_to_persistent_learner(team):
    import jax.numpy as jnp
    from basketworld.domain import ActionType
    from tests.test_jax_team_clearance import _config, _reset, _live, _step
    from basketworld_jax.train.runtime import _terminal_scoring_diagnostics

    static, state = _reset(_config())
    state = _live(static, state, team)._replace(
        offense_clearance=jnp.zeros_like(state.offense_clearance),
        clearance_achieved=jnp.zeros_like(state.clearance_achieved),
    )
    actions = (
        jnp.zeros(state.positions.shape[:2], dtype=jnp.int32)
        .at[0, int(state.ball_holder[0])]
        .set(ActionType.SHOOT.value)
    )
    out = _step(static, state, actions)
    diag = _terminal_scoring_diagnostics(static, out.state, jnp)
    assert float(diag["terminal_learner_score"][0]) == (1 if team == 1 else 0)
    assert float(diag["terminal_opponent_score"][0]) == (1 if team == 0 else 0)
    sample = rollout(points=(0, 0))
    t = sample.trajectory
    t.team_a_score_delta = np.asarray(out.team_a_score_delta).reshape(1, 1)
    t.team_b_score_delta = np.asarray(out.team_b_score_delta).reshape(1, 1)
    t.active_mask = np.ones((1, 1))
    t.truncated = np.zeros((1, 1), bool)
    t.terminal_episode_steps = np.zeros((1, 1))
    t.possession_ended = np.asarray(out.possession_ended).reshape(1, 1)
    t.training_role = np.array([[1 if team == 0 else -1]])
    for key, value in diag.items():
        setattr(t, key, np.asarray(value).reshape(1, 1))
    metrics = summarize_training_scoring({"offense": sample})
    assert metrics["learner_points_scored_observed"] == (1 if team == 1 else 0)
    assert metrics["learner_points_allowed_observed"] == (1 if team == 0 else 0)


def test_schema_status_and_checkpoint_round_trip_do_not_require_policy_migration(
    tmp_path,
):
    from basketworld_jax.train.control import write_training_status
    from basketworld_jax.checkpoints.checkpoint import (
        build_checkpoint_payload,
        _metadata_from_payload,
        _payload_from_metadata,
    )

    # Use the existing checkpoint contract: schema metadata is additive and not a policy field.
    kwargs = dict(
        update_index=3,
        trainer_config={},
        policy_spec={},
        frozen_config={},
        params={},
        opt_state={},
        current_state={},
        eval_initial_state={},
        base_key=np.array([0, 1]),
        eval_trajectories=[],
        last_metrics={},
    )
    payload = build_checkpoint_payload(**kwargs)
    metadata = _metadata_from_payload(payload)
    assert metadata["scoring_metric_schema_version"] == 2
    assert (
        _payload_from_metadata(metadata, payload["state"])[
            "scoring_metric_schema_version"
        ]
        == 2
    )
    legacy = dict(metadata)
    legacy.pop("scoring_metric_schema_version")
    assert (
        _payload_from_metadata(legacy, payload["state"])[
            "scoring_metric_schema_version"
        ]
        == 1
    )
    import json

    status = tmp_path / "status.json"
    write_training_status(
        status,
        state="running",
        completed_updates=4,
        target_updates=10,
        mlflow_run_id="same-run",
        checkpoint_path=None,
        metrics={
            "learner_points_scored_per_completed_game": None,
            "completed_game_count": 0,
        },
    )
    recorded = json.loads(status.read_text())
    assert recorded["scoring_metric_schema_version"] == 2
    assert recorded["mlflow_run_id"] == "same-run"
    assert recorded["metrics"]["learner_points_scored_per_completed_game"] is None


def test_logging_new_updates_preserves_legacy_history_and_skips_undefined_rates():
    from basketworld_jax.train.main import (
        _log_mlflow_metrics,
        _filter_mlflow_train_metrics,
    )

    class ExistingRun:
        def __init__(self):
            self.history = [
                ("jax/train/offense_learner_points_per_completed_episode", 99, 1)
            ]

        def log_metrics(self, values, step):
            self.history.extend((key, value, step) for key, value in values.items())

    run = ExistingRun()
    original = list(run.history)
    metrics = summarize_learner_games([])
    _log_mlflow_metrics(
        run, _filter_mlflow_train_metrics(metrics), step=2, prefix="jax/train"
    )
    assert run.history[:1] == original
    assert ("jax/train/scoring_metric_schema_version", 2.0, 2) in run.history
    assert not any(
        key.endswith("per_completed_game") and step == 2 for key, _, step in run.history
    )


def test_legacy_historical_ppp_reader_is_additive_and_nonmutating():
    from basketworld_jax.scoring_metrics import read_historical_scoring_summary

    old = {"candidate_points_per_completed_possession": 1.3}
    assert (
        read_historical_scoring_summary(old)[
            "candidate_points_per_completed_offensive_possession"
        ]
        == 1.3
    )
    assert list(old) == ["candidate_points_per_completed_possession"]


def test_legacy_pickle_checkpoint_infers_schema_without_rewriting_file(tmp_path):
    import pickle
    from basketworld_jax.checkpoints.checkpoint import load_checkpoint, load_checkpoint_metadata

    path = tmp_path / "legacy.pkl"
    original = pickle.dumps({"checkpoint_version": 1, "update_index": 7})
    path.write_bytes(original)
    assert load_checkpoint(path)["scoring_metric_schema_version"] == 1
    assert load_checkpoint_metadata(path)["scoring_metric_schema_version"] == 1
    assert path.read_bytes() == original
