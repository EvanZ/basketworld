import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

import basketworld
from basketworld.envs.basketworld_env_v2 import Team
from basketworld.utils.action_resolution import IllegalActionStrategy
from basketworld.utils.self_play_wrapper import SelfPlayEnvWrapper
from basketworld.utils.wrappers import (
    RewardAggregationWrapper,
    EpisodeStatsWrapper,
    BetaSetterWrapper,
    EnvIndexWrapper,
    SetObservationWrapper,
    MirrorObservationWrapper,
)


def setup_environment(args, training_team, env_idx=None):
    """Construct a single environment wrapped for training/eval."""
    env = basketworld.HexagonBasketballEnv(
        grid_size=args.grid_size,
        court_rows=getattr(args, "court_rows", None),
        court_cols=getattr(args, "court_cols", None),
        players=args.players,
        shot_clock_steps=args.shot_clock,
        min_shot_clock=getattr(args, "min_shot_clock", 10),
        defender_pressure_distance=args.defender_pressure_distance,
        defender_pressure_turnover_chance=args.defender_pressure_turnover_chance,
        defender_pressure_decay_lambda=getattr(args, "defender_pressure_decay_lambda", 1.0),
        base_steal_rate=getattr(args, "base_steal_rate", 0.35),
        steal_perp_decay=getattr(args, "steal_perp_decay", 1.5),
        steal_distance_factor=getattr(args, "steal_distance_factor", 0.08),
        steal_position_weight_min=getattr(args, "steal_position_weight_min", 0.3),
        three_point_distance=args.three_point_distance,
        three_point_short_distance=getattr(args, "three_point_short_distance", None),
        layup_pct=args.layup_pct,
        layup_std=getattr(args, "layup_std", 0.0),
        three_pt_pct=args.three_pt_pct,
        three_pt_std=getattr(args, "three_pt_std", 0.0),
        allow_dunks=args.allow_dunks,
        dunk_pct=args.dunk_pct,
        dunk_std=getattr(args, "dunk_std", 0.0),
        shot_pressure_enabled=args.shot_pressure_enabled,
        shot_pressure_max=args.shot_pressure_max,
        shot_pressure_lambda=args.shot_pressure_lambda,
        shot_pressure_arc_degrees=args.shot_pressure_arc_degrees,
        pass_arc_degrees=getattr(args, "pass_arc_start", 60.0),
        pass_oob_turnover_prob=getattr(args, "pass_oob_turnover_prob_start", 1.0),
        spawn_distance=getattr(args, "spawn_distance", 3),
        max_spawn_distance=getattr(args, "max_spawn_distance", None),
        defender_spawn_distance=getattr(args, "defender_spawn_distance", 0),
        defender_guard_distance=getattr(args, "defender_guard_distance", 1),
        offense_spawn_boundary_margin=getattr(args, "offense_spawn_boundary_margin", 0),
        pass_reward=getattr(args, "pass_reward", 0.0),
        turnover_penalty=getattr(args, "turnover_penalty", 0.0),
        violation_reward=getattr(args, "violation_reward", 1.0),
        made_shot_reward_inside=getattr(args, "made_shot_reward_inside", 2.0),
        made_shot_reward_three=getattr(args, "made_shot_reward_three", 3.0),
        missed_shot_penalty=getattr(args, "missed_shot_penalty", 0.0),
        potential_assist_reward=getattr(args, "potential_assist_reward", 0.1),
        full_assist_bonus=getattr(args, "full_assist_bonus", 0.2),
        assist_window=getattr(args, "assist_window", getattr(args, "assist_window", 2)),
        potential_assist_pct=getattr(args, "potential_assist_pct", 0.10),
        full_assist_bonus_pct=getattr(args, "full_assist_bonus_pct", 0.05),
        enable_phi_shaping=getattr(args, "enable_phi_shaping", False),
        reward_shaping_gamma=getattr(args, "reward_shaping_gamma", args.gamma),
        phi_beta=getattr(args, "phi_beta_start", 0.0),
        phi_use_ball_handler_only=getattr(args, "phi_use_ball_handler_only", False),
        phi_aggregation_mode=getattr(args, "phi_aggregation_mode", "team_best"),
        phi_blend_weight=getattr(args, "phi_blend_weight", 0.0),
        enable_profiling=args.enable_env_profiling,
        profiling_sample_rate=getattr(args, "profiling_sample_rate", 1.0),
        training_team=training_team,
        use_egocentric_obs=args.use_egocentric_obs,
        egocentric_rotate_to_hoop=args.egocentric_rotate_to_hoop,
        include_hoop_vector=args.include_hoop_vector,
        normalize_obs=args.normalize_obs,
        mask_occupied_moves=args.mask_occupied_moves,
        enable_pass_gating=getattr(args, "enable_pass_gating", True),
        three_second_lane_width=getattr(args, "three_second_lane_width", 1),
        three_second_lane_height=getattr(args, "three_second_lane_height", 1),
        three_second_max_steps=getattr(args, "three_second_max_steps", 3),
        illegal_defense_enabled=args.illegal_defense_enabled,
        offensive_three_seconds_enabled=getattr(args, "offensive_three_seconds", False),
    )
    env = EpisodeStatsWrapper(env)
    env = RewardAggregationWrapper(env)
    env = BetaSetterWrapper(env)
    if getattr(args, "use_set_obs", False):
        env = SetObservationWrapper(env)
        mirror_prob = float(getattr(args, "mirror_episode_prob", 0.0))
        if mirror_prob > 0.0:
            env = MirrorObservationWrapper(env, mirror_prob=mirror_prob)
    monitored_env = Monitor(
        env,
        info_keywords=(
            "training_team",
            "shot_dunk",
            "shot_2pt",
            "shot_3pt",
            "assisted_dunk",
            "assisted_2pt",
            "assisted_3pt",
            "potential_assisted_dunk",
            "potential_assisted_2pt",
            "potential_assisted_3pt",
            "potential_assists",
            "passes",
            "turnover",
            "turnover_pass_oob",
            "turnover_intercepted",
            "turnover_pressure",
            "turnover_offensive_lane",
            "defensive_lane_violation",
            "move_rejected_occupied",
            "made_dunk",
            "made_2pt",
            "made_3pt",
            "attempts",
            "pressure_exposure",
            "legal_actions_offense",
            "legal_actions_defense",
            "phi_beta",
            "phi_prev",
            "phi_next",
            "gt_is_three",
            "gt_is_dunk",
            "gt_points",
            "gt_shooter_off",
            "gt_shooter_q",
            "gt_shooter_r",
            "gt_distance",
            "basket_q",
            "basket_r",
        ),
    )
    if env_idx is not None:
        monitored_env = EnvIndexWrapper(monitored_env, env_idx)
    return monitored_env


def make_vector_env(
    args,
    training_team: Team,
    opponent_policy,
    num_envs: int,
    deterministic_opponent: bool,
) -> SubprocVecEnv:
    """Return a SubprocVecEnv with `num_envs` copies of the self-play environment."""
    def _make_env_with_opponent(env_idx: int, opp_policy_path) -> gym.Env:  # type: ignore[name-defined]
        def _thunk():
            base_env = setup_environment(args, training_team, env_idx=env_idx)
            return SelfPlayEnvWrapper(
                base_env,
                opponent_policy=opp_policy_path,
                training_strategy=IllegalActionStrategy.SAMPLE_PROB,
                opponent_strategy=IllegalActionStrategy.SAMPLE_PROB,
                deterministic_opponent=deterministic_opponent,
            )
        return _thunk

    def _make_env() -> gym.Env:  # type: ignore[name-defined]
        def _thunk():
            base_env = setup_environment(args, training_team)
            return SelfPlayEnvWrapper(
                base_env,
                opponent_policy=opponent_policy,
                training_strategy=IllegalActionStrategy.SAMPLE_PROB,
                opponent_strategy=IllegalActionStrategy.SAMPLE_PROB,
                deterministic_opponent=deterministic_opponent,
            )
        return _thunk

    if isinstance(opponent_policy, list):
        env_fns = [
            _make_env_with_opponent(i, opponent_policy[i % len(opponent_policy)])
            for i in range(num_envs)
        ]
    else:
        env_fns = [_make_env() for i in range(num_envs)]

    return SubprocVecEnv(env_fns, start_method="spawn")


def make_policy_init_env(args):
    """Create a single wrapped env for policy initialization (offense team)."""
    base_env = setup_environment(args, Team.OFFENSE)
    return SelfPlayEnvWrapper(
        base_env,
        opponent_policy=None,
        training_strategy=IllegalActionStrategy.SAMPLE_PROB,
        opponent_strategy=IllegalActionStrategy.SAMPLE_PROB,
        deterministic_opponent=False,
    )


def make_mixed_vector_env(
    args,
    opponent_policy,
    num_envs: int,
    deterministic_opponent: bool,
) -> SubprocVecEnv:
    """
    Return a SubprocVecEnv with num_envs split evenly: half train offense, half train defense.
    """
    num_offense = num_envs // 2
    num_defense = num_envs - num_offense

    if args.per_env_opponent_sampling:
        def _make_env_with_opponent(env_idx: int, training_team: Team, opp_policy_path) -> gym.Env:  # type: ignore[name-defined]
            def _thunk():
                base_env = setup_environment(args, training_team, env_idx=env_idx)
                return SelfPlayEnvWrapper(
                    base_env,
                    opponent_policy=opp_policy_path,
                    training_strategy=IllegalActionStrategy.SAMPLE_PROB,
                    opponent_strategy=IllegalActionStrategy.SAMPLE_PROB,
                    deterministic_opponent=deterministic_opponent,
                )
            return _thunk

        env_fns = [
            _make_env_with_opponent(i, Team.OFFENSE, opponent_policy[i % len(opponent_policy)])
            for i in range(num_offense)
        ]
        env_fns.extend([
            _make_env_with_opponent(num_offense + i, Team.DEFENSE, opponent_policy[(num_offense + i) % len(opponent_policy)])
            for i in range(num_defense)
        ])
    else:
        def _make_env_with_team(env_idx: int, training_team: Team) -> gym.Env:
            def _thunk():
                base_env = setup_environment(args, training_team, env_idx=env_idx)
                return SelfPlayEnvWrapper(
                    base_env,
                    opponent_policy=opponent_policy,
                    training_strategy=IllegalActionStrategy.SAMPLE_PROB,
                    opponent_strategy=IllegalActionStrategy.SAMPLE_PROB,
                    deterministic_opponent=deterministic_opponent,
                )
            return _thunk

        env_fns = [
            _make_env_with_team(i, Team.OFFENSE)
            for i in range(num_offense)
        ]
        env_fns.extend([
            _make_env_with_team(num_offense + i, Team.DEFENSE)
            for i in range(num_defense)
        ])

    return SubprocVecEnv(env_fns, start_method="spawn")
