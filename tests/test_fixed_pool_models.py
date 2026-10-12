from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from basketworld_jax.eval.fixed_pool import generate_player_pool
from basketworld_jax.eval.fixed_pool_models import (
    _matrix_diagnostics,
    _fit_conventional_three_factor_scenario_weights,
    _fit_conventional_three_factor_outcome_model,
    _conventional_three_factor_outcome_importance,
    FixedPoolModelError,
    assert_studies_schema_compatible,
    build_signed_player_design_matrix,
    derive_conventional_three_factor_features,
    fit_fixed_pool_studies,
    fit_fixed_pool_study_models,
    fit_skill_regression,
    fixed_pool_study_budget,
    paired_group_kfold,
    paired_group_split,
)
from basketworld_jax.eval.fixed_pool_results import (
    FIXED_POOL_EVENT_LEDGER_SCHEMA_VERSION,
    FIXED_POOL_STUDY_SCHEMA_VERSION,
)
from basketworld_jax.eval.lineup_schedule import build_balanced_lineup_schedule
from basketworld_jax.eval.skills import (
    ACTIVE_PLAYER_SKILL_REGISTRY,
    PlayerSkillDescriptor,
    PlayerSkillRegistry,
)


def test_matrix_diagnostics_encodes_infinite_condition_for_json_reports():
    diagnostics = _matrix_diagnostics(
        np.asarray([[1.0, 1.0], [1.0, 1.0]], dtype=np.float64)
    )
    assert diagnostics["condition_number"] is None
    assert diagnostics["condition_number_is_infinite"] is True
    assert diagnostics["rank_deficient"] is True


def test_signed_player_design_excludes_the_known_value_gauge_from_rank_warning():
    episodes = [
        {
            "team_a_player_ids": ["a"],
            "team_b_player_ids": ["b"],
            "starting_team": "team_a",
            "overtime_round": 0,
        },
        {
            "team_a_player_ids": ["b"],
            "team_b_player_ids": ["a"],
            "starting_team": "team_b",
            "overtime_round": 1,
        },
        {
            "team_a_player_ids": ["a"],
            "team_b_player_ids": ["b"],
            "starting_team": "team_b",
            "overtime_round": 0,
        },
    ]
    diagnostics = build_signed_player_design_matrix(
        episodes, player_ids=["a", "b"]
    )["diagnostics"]
    assert diagnostics["structural_nullity"] == 1
    assert diagnostics["unidentified_nullity"] == 0
    assert diagnostics["rank_deficient"] is False


def _synthetic_study(*, action_mode: str = "direct", game_count: int = 64):
    pool = generate_player_pool(player_count=10, seed=51)
    schedule = build_balanced_lineup_schedule(
        pool, game_count=game_count, seed=71
    )
    player_ids = [player.player_id for player in pool.players]
    true_values = dict(zip(player_ids, np.linspace(-4.5, 4.5, len(player_ids))))
    episodes = []
    events = []
    for episode_number, game in enumerate(schedule.games, start=1):
        team_a_effect = sum(
            true_values[player_id] for player_id in game.team_a_player_ids
        )
        team_b_effect = sum(
            true_values[player_id] for player_id in game.team_b_player_ids
        )
        starting_effect = 0.75 if game.starting_team == "team_a" else -0.75
        net_rating = team_a_effect - team_b_effect + starting_effect
        episodes.append(
            {
                "episode": episode_number,
                "action_mode": action_mode,
                "pair_id": game.pair_id,
                "schedule_game_id": game.game_id,
                "starting_team": game.starting_team,
                "team_a_player_ids": list(game.team_a_player_ids),
                "team_b_player_ids": list(game.team_b_player_ids),
                "analysis_default_included": True,
                "analysis_exclusion_reason": None,
                "team_a_completed_possessions": 100,
                "team_b_completed_possessions": 100,
                "completed_possessions": 200,
                "team_a_minus_team_b_net_rating": float(net_rating),
                "checkpoint_observation_schema_versions": [3],
                "environment_observation_schema_version": 3,
                "overtime_round": 0,
            }
        )
        # One expected-value shot per side makes this factor exactly the
        # signed player effect (the common baseline cancels in the rate).
        events.extend(
            [
                {
                    "schedule_game_id": game.game_id,
                    "actor_team": "team_a",
                    "event_type": "shot",
                    "shot_expected_points": 20.0 + team_a_effect,
                    "shot_success": False,
                    "shot_value": 0.0,
                },
                {
                    "schedule_game_id": game.game_id,
                    "actor_team": "team_b",
                    "event_type": "shot",
                    "shot_expected_points": 20.0 + team_b_effect,
                    "shot_success": False,
                    "shot_value": 0.0,
                },
            ]
        )
    study = {
        "manifest": {
            "schema_version": FIXED_POOL_STUDY_SCHEMA_VERSION,
            "study_id": f"synthetic-{action_mode}",
            "skill_schema_hash": pool.skill_schema_hash,
            "tables": {
                "possession_events": {
                    "schema_version": FIXED_POOL_EVENT_LEDGER_SCHEMA_VERSION
                }
            },
        },
        "pool_manifest": pool.to_payload(),
        "episodes": episodes,
        "possession_events": events,
    }
    return study, true_values


def test_skill_scaled_budget_and_paired_split_keep_groups_disjoint():
    assert fixed_pool_study_budget(8) == {
        "effective_skill_count": 8,
        "seeded_players": 64,
        "game_appearances_per_player": 100,
        "games": 640,
    }
    split = paired_group_split(["p0", "p0", "p1", "p1", "p2", "p2"], seed=8)
    assert set(split["train_pair_ids"]).isdisjoint(split["validation_pair_ids"])
    assert set(split["train_indices"]).isdisjoint(split["validation_indices"])


def test_paired_group_five_fold_cv_keeps_a_b_starts_together():
    pair_ids = [f"pair-{index // 2}" for index in range(20)]
    folds = paired_group_kfold(pair_ids, n_splits=5, seed=13)

    assert folds["available"] is True
    assert folds["fold_count"] == 5
    for fold in folds["folds"]:
        assert set(fold["train_pair_ids"]).isdisjoint(fold["validation_pair_ids"])
        for pair_id in fold["validation_pair_ids"]:
            assert sum(pair == pair_id for pair in pair_ids) == 2


def test_conventional_three_factors_use_event_ledger_and_mark_ftr_unavailable():
    features = derive_conventional_three_factor_features(
        [{"schedule_game_id": "game-1"}],
        [
            {
                "schedule_game_id": "game-1",
                "actor_team": "team_a",
                "event_type": "shot",
                "shot_success": True,
                "shot_value": 3.0,
            },
            {
                "schedule_game_id": "game-1",
                "actor_team": "team_a",
                "event_type": "shot",
                "shot_success": False,
                "shot_value": 2.0,
            },
            {
                "schedule_game_id": "game-1",
                "actor_team": "team_b",
                "event_type": "shot",
                "shot_success": True,
                "shot_value": 2.0,
            },
            {
                "schedule_game_id": "game-1",
                "actor_team": "team_b",
                "event_type": "shot",
                "shot_success": False,
                "shot_value": 3.0,
            },
            {
                "schedule_game_id": "game-1",
                "actor_team": "team_a",
                "event_type": "turnover",
            },
            {
                "schedule_game_id": "game-1",
                "actor_team": "team_a",
                "event_type": "rebound",
                "rebound_offensive": True,
            },
            {
                "schedule_game_id": "game-1",
                "actor_team": "team_b",
                "event_type": "rebound",
                "rebound_defensive": True,
            },
            {
                "schedule_game_id": "game-1",
                "actor_team": "team_a",
                "event_type": "rebound",
                "rebound_defensive": True,
            },
        ],
    )
    assert features["targets"]["effective_field_goal_pct"] == [pytest.approx(25.0)]
    assert features["targets"]["turnover_pct"] == [pytest.approx(100.0 / 3.0)]
    assert features["targets"]["offensive_rebound_pct"] == [pytest.approx(50.0)]
    assert features["unavailable_factors"]["free_throw_rate"]["status"] == "unavailable"


def test_conventional_three_factor_outcome_model_reports_held_out_effects():
    features = {
        "factor_names": [
            "effective_field_goal_pct",
            "turnover_pct",
            "offensive_rebound_pct",
        ],
        "targets": {
            "effective_field_goal_pct": [1.0, 2.0, 4.0, 3.0, 5.0, 6.0],
            "turnover_pct": [0.0, 2.0, 1.0, 3.0, 1.0, 4.0],
            "offensive_rebound_pct": [2.0, 0.0, 3.0, 1.0, 4.0, 2.0],
        },
        "units": "team_a_minus_team_b_percentage_points",
        "definitions": {},
    }
    matrix = np.asarray(
        [
            features["targets"][name]
            for name in features["factor_names"]
        ],
        dtype=np.float64,
    ).T
    outcome = 2.0 * matrix[:, 0] - 3.0 * matrix[:, 1] + 0.5 * matrix[:, 2]
    report = _fit_conventional_three_factor_outcome_model(
        features,
        outcome_target=outcome,
        train_indices=np.asarray([0, 1, 2, 3]),
        validation_indices=np.asarray([4, 5]),
        episodes=[{"pair_id": f"pair-{index}"} for index in range(6)],
        ridge_alpha=0.0,
    )
    assert report["status"] == "fit"
    assert report["coefficients"]["effective_field_goal_pct"][
        "net_rating_points_per_percentage_point"
    ] == pytest.approx(2.0)
    assert report["coefficients"]["turnover_pct"][
        "net_rating_points_per_percentage_point"
    ] == pytest.approx(-3.0)
    assert report["validation"]["r_squared"] == pytest.approx(1.0)
    importance = _conventional_three_factor_outcome_importance(
        features,
        outcome_target=outcome,
        train_indices=np.asarray([0, 1, 2, 3]),
        validation_indices=np.asarray([4, 5]),
        episodes=[{"pair_id": f"pair-{index}"} for index in range(6)],
        ridge_alpha=0.0,
    )
    assert importance["status"] == "fit"
    assert sum(
        factor["percent_of_explained_improvement"]
        for factor in importance["factor_contributions"].values()
    ) == pytest.approx(100.0)


def test_conventional_scenario_weights_match_the_two_sided_methodology():
    factor_names = [
        "effective_field_goal_pct",
        "turnover_pct",
        "offensive_rebound_pct",
    ]
    rng = np.random.default_rng(17)
    matrix = rng.normal(30.0, 4.0, size=(10, 6))
    features = {
        "factor_names": factor_names,
        "team_rates_percentage_points": {
            "team_a": {
                name: matrix[:, index].tolist()
                for index, name in enumerate(factor_names)
            },
            "team_b": {
                name: matrix[:, index + len(factor_names)].tolist()
                for index, name in enumerate(factor_names)
            },
        },
        "definitions": {},
    }
    margin = 3.0 + matrix @ np.asarray([1.5, -0.8, 0.3, -1.2, 0.6, -0.4])
    report = _fit_conventional_three_factor_scenario_weights(
        features,
        point_margin=margin,
        train_indices=np.arange(8),
        validation_indices=np.asarray([8, 9]),
        episodes=[{"pair_id": f"pair-{index}"} for index in range(10)],
    )
    assert report["status"] == "fit"
    assert report["fit"]["estimator"] == "ordinary_least_squares"
    assert report["validation"]["r_squared"] == pytest.approx(1.0)
    assert set(report["one_standard_deviation_scenarios"]) == {
        "own_effective_field_goal_pct",
        "opponent_effective_field_goal_pct",
        "own_turnover_pct",
        "opponent_turnover_pct",
        "own_offensive_rebound_pct",
        "opponent_offensive_rebound_pct",
    }
    assert sum(
        item["percent_of_total_margin_delta"]
        for item in report["factor_weights"].values()
    ) == pytest.approx(100.0)


def test_adjusted_plus_minus_recovers_known_effects_and_factor_groups():
    study, true_values = _synthetic_study()
    report = fit_fixed_pool_study_models(
        study,
        ridge_alpha=0.01,
        validation_fraction=0.25,
        split_seed=4,
        bootstrap_samples=12,
        bootstrap_seed=9,
    )

    estimated = np.asarray(
        [
            report["adjusted_plus_minus"]["players"][player_id][
                "net_rating_points_vs_replacement"
            ]
            for player_id in sorted(true_values)
        ]
    )
    expected = np.asarray([true_values[player_id] for player_id in sorted(true_values)])
    assert np.corrcoef(estimated, expected)[0, 1] > 0.99
    assert report["adjusted_plus_minus"]["design_effects"][
        "team_a_starting_side"
    ] == pytest.approx(0.75, abs=0.08)
    assert set(report["adjusted_plus_minus"]["train_pair_ids"]).isdisjoint(
        report["adjusted_plus_minus"]["validation"]["pair_ids"]
    )
    factor_report = report["factor_models"]
    assert set(factor_report["factor_estimation_pair_ids"]).isdisjoint(
        factor_report["outcome_validation_pair_ids"]
    )
    conventional = report["conventional_three_factor_models"]
    assert conventional["unavailable_factors"]["free_throw_rate"]["status"] == "unavailable"
    assert conventional["definitions"]["effective_field_goal_pct"] == (
        "(FGM + 0.5 * 3PM) / FGA"
    )
    shot_factor = factor_report["models"]["shot_expected_value"]
    assert shot_factor["status"] == "fit"
    estimated_factor = np.asarray(
        [
            shot_factor["player_factor_impact"][player_id]
            for player_id in sorted(true_values)
        ]
    )
    assert np.corrcoef(estimated_factor, expected)[0, 1] > 0.99
    assert report["sizing"]["under_sized"] is True
    assert "under_sized_relative_to_skill_scaled_budget" in report["warnings"]


def test_direct_skill_ratings_fit_game_outcomes_without_apm_as_the_target():
    study, true_values = _synthetic_study(game_count=64)
    for player in study["pool_manifest"]["players"]:
        player["skills"]["rebound"] = true_values[player["player_id"]]
    report = fit_fixed_pool_study_models(
        study,
        ridge_alpha=0.0,
        validation_fraction=0.25,
        split_seed=2,
        bootstrap_samples=2,
    )

    direct = np.asarray(
        [
            report["direct_skill_ratings"]["players"][player_id][
                "net_rating_points_vs_replacement"
            ]
            for player_id in sorted(true_values)
        ]
    )
    expected = np.asarray([true_values[player_id] for player_id in sorted(true_values)])
    assert np.corrcoef(direct, expected)[0, 1] > 0.99
    assert report["direct_skill_ratings"]["fit"]["target"] == (
        "team_a_minus_team_b_net_rating"
    )
    agreement = report["direct_skill_ratings"]["apm_agreement"]
    assert agreement["sample_count"] == len(true_values)
    assert agreement["overlap_k"] == len(true_values)
    assert agreement["rating_pearson_correlation"] is not None
    assert agreement["ranking_spearman_correlation"] is not None


def test_ridge_lambdas_are_selected_by_deterministic_five_fold_cv():
    study, _ = _synthetic_study(game_count=48)
    first = fit_fixed_pool_study_models(
        study,
        validation_fraction=0.2,
        split_seed=7,
        bootstrap_samples=2,
        bootstrap_seed=3,
    )
    second = fit_fixed_pool_study_models(
        study,
        validation_fraction=0.2,
        split_seed=7,
        bootstrap_samples=2,
        bootstrap_seed=3,
    )

    selection = first["adjusted_plus_minus"]["fit"]["ridge_selection"]
    assert selection["method"] == "grouped_k_fold_cross_validation"
    assert selection["fold_count"] == 5
    assert selection["selected_alpha"] in selection["candidate_alphas"]
    assert selection["selected_alpha"] == second["adjusted_plus_minus"]["fit"][
        "ridge_selection"
    ]["selected_alpha"]


def test_skill_regression_handles_zero_variance_and_declared_extra_skill():
    extra = PlayerSkillDescriptor(
        key="screen_navigation",
        display_label="Screen navigation",
        numeric_type="float",
        minimum=-2.0,
        maximum=2.0,
        sampling_distribution="normal",
        sampling_defaults=(("default_mean", 0.0), ("default_std", 0.5)),
        semantic_group="defense",
        runtime_field="screen_navigation",
    )
    registry = PlayerSkillRegistry(
        schema_version=2,
        descriptors=(*ACTIVE_PLAYER_SKILL_REGISTRY.descriptors, extra),
    )
    pool = generate_player_pool(
        player_count=12,
        seed=17,
        registry=registry,
        sampling_overrides={"layup": {"mean": 0.5, "std": 0.0}},
    )
    player_ids = [player.player_id for player in pool.players]
    report = fit_skill_regression(
        {"pool_manifest": pool.to_payload()},
        player_ids=player_ids,
        player_values=np.linspace(-3.0, 3.0, len(player_ids)),
        ridge_alpha=0.1,
    )
    assert "screen_navigation" in report["coefficients"]
    assert {entry["skill"] for entry in report["excluded_skills"]} >= {"layup"}
    auto_report = fit_skill_regression(
        {"pool_manifest": pool.to_payload()},
        player_ids=player_ids,
        player_values=np.linspace(-3.0, 3.0, len(player_ids)),
    )
    assert auto_report["fit"]["ridge_selection"]["fold_count"] == 5


def test_action_modes_are_reported_separately_and_cross_schema_fails():
    direct, _ = _synthetic_study(action_mode="direct", game_count=8)
    sampled, _ = _synthetic_study(action_mode="sampled", game_count=8)
    reports = fit_fixed_pool_studies(
        [direct, sampled], ridge_alpha=0.1, bootstrap_samples=2
    )
    assert reports["action_modes_are_separate"] is True
    assert set(reports["reports_by_action_mode"]) == {"direct", "sampled"}

    incompatible = deepcopy(sampled)
    incompatible["manifest"]["skill_schema_hash"] = "other-schema"
    with pytest.raises(FixedPoolModelError, match="cross-schema"):
        assert_studies_schema_compatible([direct, incompatible])
