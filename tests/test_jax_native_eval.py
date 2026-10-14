from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from basketworld.envs.basketworld_env_v2 import HexagonBasketballEnv, Team
from basketworld_jax.config import TRAIN_FROZEN_VALUES
from basketworld_jax.eval import (
    can_run_native_jax_evaluation,
    run_native_jax_evaluation,
)
from basketworld_jax.eval.fixed_pool import generate_player_pool
from basketworld_jax.eval.fixed_pool_results import load_fixed_pool_study_artifacts
from basketworld_jax.eval.lineup_schedule import build_balanced_lineup_schedule
from basketworld_jax.eval.native import (
    _apply_native_profile_rows,
    _apply_native_custom_setup,
    _apply_paired_starting_teams,
    _apply_scheduled_starting_teams,
    _init_eval_diagnostics,
    _native_eval_horizon,
    _phi_beta_for_eval,
    _post_orb_continuation_diagnostics_from_trace,
    _record_assist_event,
    _record_pass_link_diagnostics,
    _task_reward_scale_for_eval,
)
from basketworld_jax.eval.skills import ACTIVE_PLAYER_SKILL_REGISTRY
from basketworld_jax.env.minimal import (
    MULTI_POSSESSION_SCHEMA_VERSION,
    TEAM_A,
    TEAM_B,
    reset_batch_minimal,
)
from tests.test_jax_multi_possession import _multi_possession_static
from basketworld_jax.train.main import (
    parse_args,
    run_training_loop,
    validate_train_args,
)


def _native_eval_env_params() -> dict:
    signature_keys = set(inspect.signature(HexagonBasketballEnv.__init__).parameters)
    params = {
        key: value
        for key, value in TRAIN_FROZEN_VALUES.items()
        if key in signature_keys
    }
    params["training_team"] = Team.OFFENSE
    params["offensive_three_seconds_enabled"] = bool(
        TRAIN_FROZEN_VALUES.get("offensive_three_seconds", False)
    )
    return params


def test_native_flow_diagnostics_preserve_both_stable_teams():
    diagnostics = _init_eval_diagnostics()
    stats = {
        player_id: {"potential_assists": 0, "assists": 0} for player_id in range(10)
    }

    _record_pass_link_diagnostics(
        diagnostics,
        passer_id=5,
        receiver_id=6,
        completed=True,
        user_team_ids_set={0, 1, 2, 3, 4},
    )
    _record_assist_event(
        stats=stats,
        eval_diagnostics=diagnostics,
        passer_id=5,
        shooter_id=6,
        shot_type="three",
        potential=True,
        full=True,
        user_team_ids_set={0, 1, 2, 3, 4},
    )

    assert diagnostics["pass_links"] == {"5->6": 1}
    assert diagnostics["completed_pass_links"] == {"5->6": 1}
    assert diagnostics["assist_links"] == {"5->6": 1}
    assert diagnostics["assist_links_by_type"]["three"] == {"5->6": 1}


def test_post_orb_continuation_diagnostics_compare_values_with_shaped_returns():
    trace = {
        "active": np.asarray([[1], [1], [1], [0]], dtype=np.int8),
        "terminal_episode_steps": np.asarray([[0], [0], [3], [0]], dtype=np.int32),
        "offensive_rebound": np.asarray([[1], [0], [0], [0]], dtype=np.int8),
        "shot_success": np.asarray([[0], [0], [1], [0]], dtype=np.int8),
        "shot_value": np.asarray([[0.0], [0.0], [2.0], [0.0]], dtype=np.float32),
        "offense_values": np.asarray([[0.0], [0.8], [0.0], [0.0]], dtype=np.float32),
        "defense_values": np.asarray([[0.0], [-0.7], [0.0], [0.0]], dtype=np.float32),
        "offense_rewards": np.zeros((4, 1), dtype=np.float32),
        "defense_rewards": np.zeros((4, 1), dtype=np.float32),
        "offense_training_rewards": np.asarray(
            [[0.4], [0.1], [1.5], [0.0]], dtype=np.float32
        ),
        "defense_training_rewards": np.asarray(
            [[-0.4], [-0.1], [-1.5], [0.0]], dtype=np.float32
        ),
        "done": np.asarray([[0], [0], [1], [0]], dtype=np.int8),
    }

    diagnostics = _post_orb_continuation_diagnostics_from_trace(
        trace,
        env_index=0,
        gamma=0.5,
    )

    assert diagnostics["post_orb_samples"] == 1
    assert diagnostics["post_orb_points_sum"] == pytest.approx(2.0)
    assert diagnostics["post_orb_consensus_value_sum"] == pytest.approx(0.75)
    assert diagnostics["post_orb_offense_shaped_return_sum"] == pytest.approx(0.85)
    assert diagnostics["post_orb_defense_shaped_return_sum"] == pytest.approx(-0.85)
    assert diagnostics["post_orb_consensus_shaped_return_sum"] == pytest.approx(0.85)

    trace["terminal_episode_steps"] = np.zeros((4, 1), dtype=np.int32)
    truncated = _post_orb_continuation_diagnostics_from_trace(
        trace,
        env_index=0,
        gamma=0.5,
    )
    assert truncated["post_orb_samples"] == 0
    assert truncated["post_orb_shaped_return_samples"] == 0


def test_task_reward_scale_for_eval_matches_update_schedule():
    payload = {"update_index": 250, "trainer_config": {}}
    params = {
        "jax/task_reward_scale_start": 0.1,
        "jax/task_reward_scale_end": 1.0,
        "jax/task_reward_scale_warmup_updates": 0,
        "jax/task_reward_scale_ramp_updates": 500,
    }

    assert _task_reward_scale_for_eval(params, payload) == pytest.approx(0.55)
    assert _task_reward_scale_for_eval({}, payload) == pytest.approx(1.0)

    phi_params = {
        "jax/enable_phi_shaping": True,
        "jax/phi_beta_start": 0.0,
        "jax/phi_beta_end": 0.25,
        "jax/phi_beta_warmup_updates": 0,
        "jax/phi_beta_ramp_updates": 500,
    }
    assert _phi_beta_for_eval(phi_params, payload, default=0.0) == pytest.approx(0.125)


def test_native_eval_horizon_includes_regulation_and_capped_paired_overtime():
    env = SimpleNamespace(
        shot_clock_steps=24,
        enable_rebounds=False,
        enable_multi_possession=True,
        multi_possession_limit=3,
        multi_possession_overtime_round_cap=0,
        inbound_deadline_steps=5,
    )
    # Base per-possession guard is 26 + 5 inbound steps + 2 transition steps.
    assert _native_eval_horizon(env, {}, {}) == 2 * (3 + 3) * 33

    env.multi_possession_overtime_round_cap = 1
    assert _native_eval_horizon(env, {}, {}) == 2 * (3 + 1) * 33


def test_native_eval_pairs_reset_rows_with_swapped_starting_teams_without_moving_players():
    static = _multi_possession_static(players=2, possession_limit=3)
    pair_keys = jax.random.split(jax.random.PRNGKey(19), 2)
    state = reset_batch_minimal(static, jnp.repeat(pair_keys, 2, axis=0), jax, jnp)

    paired = _apply_paired_starting_teams(static, state, episode_offset=0, jnp=jnp)

    assert np.asarray(paired.starting_offense_team).tolist() == [
        TEAM_A,
        TEAM_B,
        TEAM_A,
        TEAM_B,
    ]
    assert np.asarray(paired.offense_team).tolist() == [TEAM_A, TEAM_B, TEAM_A, TEAM_B]
    team_a_ids = set(np.asarray(static.offense_ids, dtype=np.int32).tolist())
    team_b_ids = set(np.asarray(static.defense_ids, dtype=np.int32).tolist())
    for idx, holder in enumerate(
        np.asarray(paired.ball_holder, dtype=np.int32).tolist()
    ):
        assert holder in (team_a_ids if idx % 2 == 0 else team_b_ids)

    np.testing.assert_array_equal(
        np.asarray(paired.positions), np.asarray(state.positions)
    )


def test_native_custom_setup_freezes_every_skill_for_every_player():
    static = _multi_possession_static(players=2, possession_limit=3)
    batch_size = 3
    reset_keys = jax.random.split(jax.random.PRNGKey(71), batch_size)
    state = reset_batch_minimal(static, reset_keys, jax, jnp)
    player_count = int(static.role_encoding.shape[0])
    profiles = {
        str(player_id): {
            "layup": 0.20 + player_id * 0.01,
            "three_pt": 0.30 + player_id * 0.01,
            "dunk": 0.40 + player_id * 0.01,
            "rebound": -0.30 + player_id * 0.10,
            "pressure": -0.20 + player_id * 0.10,
            "shot_contest": -0.15 + player_id * 0.10,
            "steal": -0.10 + player_id * 0.10,
            "ball_handling": player_id * 0.10,
            "passing": 0.10 + player_id * 0.10,
        }
        for player_id in range(player_count)
    }

    pinned = _apply_native_custom_setup(
        static,
        state,
        {"player_skills": profiles},
        batch_size,
        jnp,
    )

    for descriptor in ACTIVE_PLAYER_SKILL_REGISTRY.descriptors:
        expected = np.asarray(
            [
                profiles[str(player_id)][descriptor.key]
                for player_id in range(player_count)
            ],
            dtype=np.float32,
        )
        actual = np.asarray(getattr(pinned, descriptor.runtime_field), dtype=np.float32)
        assert actual.shape == (batch_size, player_count)
        np.testing.assert_allclose(
            actual,
            np.broadcast_to(expected[None, :], actual.shape),
            atol=1e-6,
        )


def test_fixed_pool_profile_rows_and_scheduled_starters_preserve_stable_slots():
    static = _multi_possession_static(players=5, possession_limit=3)
    batch_size = 2
    state = reset_batch_minimal(
        static,
        jax.random.split(jax.random.PRNGKey(32), batch_size),
        jax,
        jnp,
    )
    player_count = int(static.role_encoding.shape[0])
    profile_rows = {
        descriptor.key: np.asarray(
            [
                [row + player_id / 100.0 for player_id in range(player_count)]
                for row in range(batch_size)
            ],
            dtype=np.float32,
        )
        for descriptor in ACTIVE_PLAYER_SKILL_REGISTRY.descriptors
    }

    pinned = _apply_native_profile_rows(
        static,
        state,
        profile_rows,
        batch_size,
        jnp,
    )
    scheduled = _apply_scheduled_starting_teams(
        static,
        pinned,
        jnp.asarray([TEAM_A, TEAM_B], dtype=jnp.int8),
        jnp=jnp,
    )

    for descriptor in ACTIVE_PLAYER_SKILL_REGISTRY.descriptors:
        np.testing.assert_allclose(
            np.asarray(getattr(scheduled, descriptor.runtime_field)),
            profile_rows[descriptor.key],
        )
    assert np.asarray(scheduled.starting_offense_team).tolist() == [TEAM_A, TEAM_B]
    assert np.asarray(scheduled.offense_team).tolist() == [TEAM_A, TEAM_B]
    assert int(np.asarray(scheduled.ball_holder)[0]) in set(
        np.asarray(static.offense_ids).tolist()
    )
    assert int(np.asarray(scheduled.ball_holder)[1]) in set(
        np.asarray(static.defense_ids).tolist()
    )
    np.testing.assert_array_equal(
        np.asarray(scheduled.positions), np.asarray(state.positions)
    )


def test_native_jax_evaluation_returns_stats_tab_payload(tmp_path):
    pytest.importorskip("jax")

    checkpoint_dir = tmp_path / "native_eval_ckpts"
    args = parse_args(
        [
            "--run-train-loop",
            "--kernel-batch-size",
            "4",
            "--rollout-horizon",
            "4",
            "--num-updates",
            "1",
            "--policy-update-epochs",
            "1",
            "--eval-every-updates",
            "1",
            "--eval-horizon",
            "4",
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--checkpoint-every-updates",
            "1",
            "--no-progress",
        ]
    )
    validate_train_args(args)
    train_result = run_training_loop(args)
    checkpoint_path = train_result["latest_checkpoint_path"]

    assert can_run_native_jax_evaluation(
        unified_policy_path=checkpoint_path,
        opponent_policy_path=None,
        custom_setup=None,
        randomize_offense_permutation=False,
    )

    result = run_native_jax_evaluation(
        num_episodes=3,
        player_deterministic=False,
        opponent_deterministic=False,
        required_params={},
        optional_params=_native_eval_env_params(),
        unified_policy_path=checkpoint_path,
        opponent_policy_path=None,
        user_team_name="OFFENSE",
        role_flag_offense=1.0,
        role_flag_defense=-1.0,
        eval_seed=123,
    )

    assert len(result["results"]) == 3
    assert result["fixed_pool_artifacts"] is None
    assert "fixed_pool_study" not in result["eval_diagnostics"]["jax_native_summary"]
    assert all("fixed_pool_study" not in episode for episode in result["results"])
    assert isinstance(result["shot_accumulator"], dict)
    assert set(result["per_player_stats"]) == set(range(6))
    assert "none" in result["per_intent_stats"]

    player_zero = result["per_player_stats"][0]
    for key in (
        "shots",
        "makes",
        "assists",
        "potential_assists",
        "turnovers",
        "points",
        "episodes",
        "steps",
        "shot_types",
        "shot_chart",
        "assist_full_by_type",
        "unassisted",
        "sampled_skills",
        "sampled_skill_count",
    ):
        assert key in player_zero
    assert player_zero["sampled_skill_count"] == 3
    assert set(player_zero["sampled_skills"]) == set(ACTIVE_PLAYER_SKILL_REGISTRY.keys)
    assert all(np.isfinite(value) for value in player_zero["sampled_skills"].values())

    diagnostics = result["eval_diagnostics"]
    assert diagnostics["jax_native_summary"]["eval_seed"] == 123
    assert diagnostics["jax_native_summary"]["allow_dunks"] is True
    assert "shot_dunk_share" in diagnostics["jax_native_summary"]
    assert "shot_two_share" in diagnostics["jax_native_summary"]
    assert "shot_three_share" in diagnostics["jax_native_summary"]
    assert "post_orb_shaped_return_sample_count" in diagnostics["jax_native_summary"]
    assert (
        "post_orb_consensus_shaped_return_per_sample"
        in diagnostics["jax_native_summary"]
    )
    assert (
        "post_orb_critic_minus_shaped_return_per_sample"
        in diagnostics["jax_native_summary"]
    )
    assert (
        diagnostics["jax_native_summary"][
            "post_orb_shaped_return_includes_training_intent_bonus"
        ]
        is False
    )
    assert "post_orb_phi_beta" in diagnostics["jax_native_summary"]
    assert diagnostics["jax_native_summary"]["intent_active_episodes"] == 0
    assert diagnostics["jax_native_summary"]["intent_inactive_episodes"] == 3
    assert diagnostics["jax_native_summary"]["defense_intent_active_episodes"] == 0
    assert diagnostics["jax_native_summary"]["defense_intent_inactive_episodes"] == 3
    for key in (
        "action_mix",
        "reward_breakdown",
        "turnover_reasons",
        "assist_links",
        "potential_assist_links",
        "jax_native_summary",
    ):
        assert key in diagnostics

    for episode in result["results"]:
        assert episode["intent_index"] is None
        assert episode["intent_active"] is False
        assert episode["intent_visible_to_defense"] is False
        assert episode["defense_intent_index"] is None
        assert episode["defense_intent_active"] is False
        outcome_info = episode["outcome_info"]
        assert set(outcome_info) >= {
            "shots",
            "turnovers",
            "defensive_lane_violations",
            "shot_clock",
            "three_point_distance",
        }


def test_native_multi_possession_evaluation_pairs_starters_and_separates_cutoffs(
    tmp_path,
):
    checkpoint_dir = tmp_path / "native_multi_eval_ckpts"
    args = parse_args(
        [
            "--run-train-loop",
            "--enable-multi-possession",
            "--multi-possession-limit",
            "1",
            "--inbound-deadline-steps",
            "2",
            "--kernel-batch-size",
            "1",
            "--rollout-horizon",
            "1",
            "--policy-update-epochs",
            "1",
            "--ppo-minibatches",
            "1",
            "--num-updates",
            "1",
            "--eval-every-updates",
            "0",
            "--eval-deploy-every-updates",
            "0",
            "--eval-deploy-batches",
            "0",
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--checkpoint-every-updates",
            "1",
            "--no-progress",
        ]
    )
    validate_train_args(args)
    checkpoint_path = run_training_loop(args)["latest_checkpoint_path"]
    optional_params = _native_eval_env_params()
    optional_params.update(
        {
            "enable_multi_possession": True,
            "multi_possession_limit": 1,
            "inbound_deadline_steps": 2,
        }
    )

    result = run_native_jax_evaluation(
        num_episodes=4,
        player_deterministic=True,
        opponent_deterministic=True,
        required_params={},
        optional_params=optional_params,
        unified_policy_path=checkpoint_path,
        opponent_policy_path=None,
        user_team_name="OFFENSE",
        role_flag_offense=1.0,
        role_flag_defense=-1.0,
        eval_seed=321,
    )

    summary = result["eval_diagnostics"]["jax_native_summary"]
    pairing = summary["paired_starting_team_evaluation"]
    assert pairing["enabled"] is True
    assert pairing["paired_game_count"] == 2
    assert pairing["starting_team_a_count"] == 2
    assert pairing["starting_team_b_count"] == 2
    assert summary["completed_games"] + summary["incomplete_games"] == 4
    assert (
        summary["win_count"] + summary["loss_count"] + summary["tie_count"]
        == summary["completed_games"]
    )
    assert summary["evaluation_horizon_guard"]["cutoff_is_completed_game"] is False
    assert summary["evaluation_horizon_guard"]["cutoff_is_tie"] is False
    assert "inbound_timeout_turnover_count" in summary
    assert "turnovers_before_clearance_count" in summary
    assert "completed_possession_count" in summary
    assert "completed_user_offensive_possessions" in summary
    assert "completed_opponent_offensive_possessions" in summary
    assert "completed_user_score_total" in summary
    assert "completed_opponent_score_total" in summary
    assert (
        summary["completed_user_offensive_possessions"]
        + summary["completed_opponent_offensive_possessions"]
        == summary["completed_possessions_total"]
    )
    assert "completed_possession_live_steps" in summary
    assert "mean_live_steps_per_completed_possession" in summary
    assert "winner_reward_per_episode" in summary
    assert "overtime_game_count" in summary
    assert "overtime_rounds_mean" in summary
    assert "overtime_rounds_max" in summary
    assert "check_opportunity_count" in summary
    assert "check_pickup_count" in summary
    assert "check_violation_count" in summary
    assert "check_space_defense_violation_count" in summary
    assert "check_space_defense_violation_rate" in summary
    assert "check_pickup_rate" in summary
    assert "check_mean_pickup_steps" in summary
    assert summary["spatial_live_step_count"] > 0
    assert np.isfinite(summary["mean_live_all_player_pair_distance"])
    assert np.isfinite(summary["mean_live_offense_teammate_pair_distance"])
    assert np.isfinite(summary["mean_live_defense_teammate_pair_distance"])
    assert summary["ball_handler_sample_count"] > 0
    assert np.isfinite(summary["mean_live_ball_handler_nearest_defender_distance"])
    assert 0.0 <= summary["live_ball_handler_pressure_rate"] <= 1.0
    assert np.isfinite(summary["mean_live_offense_nearest_defender_distance"])
    assert 0.0 <= summary["mean_live_unguarded_offense_fraction"] <= 1.0
    assert np.isfinite(summary["mean_live_team_centroid_distance"])
    if summary["completed_user_offensive_possessions"] > 0:
        assert np.isclose(
            summary["user_points_per_possession"],
            summary["completed_user_score_total"]
            / summary["completed_user_offensive_possessions"],
        )
    if summary["completed_opponent_offensive_possessions"] > 0:
        assert np.isclose(
            summary["opponent_points_per_possession"],
            summary["completed_opponent_score_total"]
            / summary["completed_opponent_offensive_possessions"],
        )
    for episode in result["results"]:
        assert set(episode["game"]) >= {
            "completed",
            "result",
            "team_a_score",
            "team_b_score",
            "completed_possessions",
            "team_a_completed_possessions",
            "team_b_completed_possessions",
            "starting_offense_team",
            "overtime_round",
            "overtime_possessions_completed",
        }
        if episode["completed"]:
            assert episode["game"]["team_a_completed_possessions"] >= 1
            assert episode["game"]["team_b_completed_possessions"] >= 1
            assert episode["game"]["completed_possessions"] == (
                episode["game"]["team_a_completed_possessions"]
                + episode["game"]["team_b_completed_possessions"]
            )
        if not episode["completed"]:
            assert episode["game"]["result"] is None


def test_native_fixed_pool_evaluation_uses_scheduled_profiles_and_rng_streams(tmp_path):
    checkpoint_dir = tmp_path / "native_fixed_pool_eval_ckpts"
    args = parse_args(
        [
            "--run-train-loop",
            "--players",
            "5",
            "--enable-multi-possession",
            "--multi-possession-limit",
            "1",
            "--inbound-deadline-steps",
            "2",
            "--kernel-batch-size",
            "1",
            "--rollout-horizon",
            "1",
            "--policy-update-epochs",
            "1",
            "--ppo-minibatches",
            "1",
            "--num-updates",
            "1",
            "--eval-every-updates",
            "0",
            "--eval-deploy-every-updates",
            "0",
            "--eval-deploy-batches",
            "0",
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--checkpoint-every-updates",
            "1",
            "--no-progress",
        ]
    )
    validate_train_args(args)
    checkpoint_path = run_training_loop(args)["latest_checkpoint_path"]
    optional_params = _native_eval_env_params()
    optional_params.update(
        {
            "players": 5,
            "enable_multi_possession": True,
            "multi_possession_limit": 1,
            "inbound_deadline_steps": 2,
        }
    )
    pool = generate_player_pool(player_count=10, seed=808)
    schedule = build_balanced_lineup_schedule(pool, game_count=2, seed=909)

    result = run_native_jax_evaluation(
        num_episodes=len(schedule.games),
        player_deterministic=True,
        opponent_deterministic=True,
        required_params={},
        optional_params=optional_params,
        unified_policy_path=checkpoint_path,
        opponent_policy_path=None,
        user_team_name="OFFENSE",
        role_flag_offense=1.0,
        role_flag_defense=-1.0,
        eval_seed=333,
        fixed_player_pool=pool,
        fixed_lineup_schedule=schedule,
        fixed_pool_artifact_dir=tmp_path / "fixed_pool_artifacts",
    )

    summary = result["eval_diagnostics"]["jax_native_summary"]["fixed_pool_study"]
    assert summary["pool_id"] == pool.pool_id
    assert summary["schedule_id"] == schedule.schedule_id
    assert summary["action_mode"] == "direct"
    assert summary["sampling_group_count"] == len(schedule.games)
    assert summary["rng_streams"] == ["reset", "environment", "policy"]
    assert result["fixed_pool_artifacts"] is not None
    fixed_pool_artifact = load_fixed_pool_study_artifacts(
        result["fixed_pool_artifacts"]["root"]
    )
    assert len(fixed_pool_artifact["episodes"]) == len(schedule.games)
    assert all(
        row["checkpoint_observation_schema_versions"]
        == [int(MULTI_POSSESSION_SCHEMA_VERSION)]
        for row in fixed_pool_artifact["episodes"]
    )
    assert "possession_events" in fixed_pool_artifact["manifest"]["tables"]
    assert len(result["results"]) == 2
    for episode, game in zip(result["results"], schedule.games, strict=True):
        trace = episode["fixed_pool_study"]
        assert trace["schedule_game_id"] == game.game_id
        assert trace["pair_id"] == game.pair_id
        assert trace["team_a_player_ids"] == list(game.team_a_player_ids)
        assert trace["team_b_player_ids"] == list(game.team_b_player_ids)
        assert trace["reset_rng_root"] == game.reset_rng_root
        assert trace["environment_rng_root"] == game.environment_rng_root
        assert trace["policy_rng_root"] == game.policy_rng_root
        assert episode["game"]["starting_offense_team"] == game.starting_team

    expected_slot_players = (
        *schedule.games[0].team_a_player_ids,
        *schedule.games[0].team_b_player_ids,
    )
    for slot, player_id in enumerate(expected_slot_players):
        expected_player = pool.players[int(player_id[-4:])]
        assert result["per_player_stats"][slot]["sampled_skills"][
            "layup"
        ] == pytest.approx(expected_player.skills["layup"])

    sampled_result = run_native_jax_evaluation(
        num_episodes=len(schedule.games),
        player_deterministic=False,
        opponent_deterministic=False,
        required_params={},
        optional_params=optional_params,
        unified_policy_path=checkpoint_path,
        opponent_policy_path=None,
        user_team_name="OFFENSE",
        role_flag_offense=1.0,
        role_flag_defense=-1.0,
        eval_seed=333,
        fixed_player_pool=pool,
        fixed_lineup_schedule=schedule,
    )
    assert (
        sampled_result["eval_diagnostics"]["jax_native_summary"]["fixed_pool_study"][
            "action_mode"
        ]
        == "sampled"
    )
    assert [
        episode["fixed_pool_study"]["policy_rng_root"]
        for episode in sampled_result["results"]
    ] == [game.policy_rng_root for game in schedule.games]
