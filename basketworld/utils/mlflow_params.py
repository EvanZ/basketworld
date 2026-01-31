import mlflow
from typing import Any, Callable, Dict, Tuple


def _get_param(
    params_dict: Dict[str, str],
    names: list[str],
    cast: Callable[[Any], Any],
    default: Any,
) -> Any:
    """Try multiple parameter names, cast if found, else return default.

    Params in MLflow are stored as strings. This helper searches for the first
    available alias in names, attempts to cast it, and falls back to default if
    missing or invalid.
    """
    for name in names:
        if name in params_dict and params_dict[name] != "":
            try:
                return cast(params_dict[name])
            except Exception:
                pass
    return default


def get_mlflow_params(
    client: mlflow.tracking.MlflowClient, run_id: str
) -> Tuple[dict, dict]:
    """Fetch and normalize environment-related params from an MLflow run.

    Returns a tuple (required, optional) where:
      - required contains required fields (grid_size, players, shot_clock)
      - optional contains all optional knobs with sensible defaults
    """
    run = client.get_run(run_id)
    params = run.data.params

    # Required
    required = {
        "grid_size": int(params["grid_size"]) if "grid_size" in params else 16,
        "players": int(params["players"]) if "players" in params else 3,
        "shot_clock": int(params["shot_clock"]) if "shot_clock" in params else 24,
    }

    # Optional
    optional = {}
    optional["court_rows"] = _get_param(
        params,
        ["court_rows", "court-rows"],
        lambda v: None if v == "" or v == "None" else int(v),
        None,
    )
    optional["court_cols"] = _get_param(
        params,
        ["court_cols", "court-cols"],
        lambda v: None if v == "" or v == "None" else int(v),
        None,
    )
    optional["three_point_distance"] = _get_param(
        params,
        [
            "three_point_distance",
            "three-point-distance",
            "three_pt_distance",
            "three-pt-distance",
        ],
        float,
        4.0,
    )
    optional["three_point_short_distance"] = _get_param(
        params,
        ["three_point_short_distance", "three-point-short-distance"],
        lambda v: None if v == "" or v == "None" else float(v),
        None,
    )
    optional["layup_pct"] = _get_param(params, ["layup_pct", "layup-pct"], float, 0.60)
    optional["three_pt_pct"] = _get_param(
        params, ["three_pt_pct", "three-pt-pct"], float, 0.37
    )
    # Per-player shooting variability (std dev) for episode sampling
    optional["layup_std"] = _get_param(params, ["layup_std", "layup-std"], float, 0.0)
    optional["three_pt_std"] = _get_param(
        params, ["three_pt_std", "three-pt-std"], float, 0.0
    )
    optional["spawn_distance"] = _get_param(
        params, ["spawn_distance", "spawn-distance"], int, 3
    )
    optional["max_spawn_distance"] = _get_param(
        params,
        ["max_spawn_distance", "max-spawn-distance"],
        lambda v: None if v == "" or v == "None" else int(v),
        None,
    )
    optional["defender_spawn_distance"] = _get_param(
        params, ["defender_spawn_distance", "defender-spawn-distance"], int, 0
    )
    optional["defender_guard_distance"] = _get_param(
        params, ["defender_guard_distance", "defender-guard-distance"], int, 1
    )
    optional["offense_spawn_boundary_margin"] = _get_param(
        params,
        ["offense_spawn_boundary_margin", "offense-spawn-boundary-margin"],
        int,
        0,
    )
    optional["allow_dunks"] = _get_param(
        params,
        ["allow_dunks", "allow-dunks"],
        lambda v: str(v).lower() in ["1", "true", "yes", "y", "t"],
        False,
    )
    optional["dunk_pct"] = _get_param(params, ["dunk_pct", "dunk-pct"], float, 0.90)
    optional["dunk_std"] = _get_param(params, ["dunk_std", "dunk-std"], float, 0.0)
    optional["shot_pressure_enabled"] = _get_param(
        params,
        ["shot_pressure_enabled", "shot-pressure-enabled"],
        lambda v: str(v).lower() in ["1", "true", "yes", "y", "t"],
        True,
    )
    optional["shot_pressure_max"] = _get_param(
        params, ["shot_pressure_max", "shot-pressure-max"], float, 0.5
    )
    optional["shot_pressure_lambda"] = _get_param(
        params, ["shot_pressure_lambda", "shot-pressure-lambda"], float, 1.0
    )
    optional["shot_pressure_arc_degrees"] = _get_param(
        params, ["shot_pressure_arc_degrees", "shot-pressure-arc-degrees"], float, 60.0
    )
    optional["defender_pressure_distance"] = _get_param(
        params, ["defender_pressure_distance", "defender-pressure-distance"], int, 1
    )
    optional["defender_pressure_turnover_chance"] = _get_param(
        params,
        ["defender_pressure_turnover_chance", "defender-pressure-turnover-chance"],
        float,
        0.05,
    )
    optional["defender_pressure_decay_lambda"] = _get_param(
        params,
        ["defender_pressure_decay_lambda", "defender-pressure-decay-lambda"],
        float,
        1.0,
    )
    # Minimum randomized shot clock at reset
    optional["min_shot_clock"] = _get_param(params, ["min_shot_clock"], int, 10)
    optional["mask_occupied_moves"] = _get_param(
        params,
        ["mask_occupied_moves", "mask-occupied-moves"],
        lambda v: str(v).lower() in ["1", "true", "yes", "y", "t"],
        False,
    )
    # 3-second violation parameters (shared between offense and defense)
    optional["three_second_lane_width"] = _get_param(
        params,
        [
            "three_second_lane_width",
            "three-second-lane-width",
            "offensive_three_second_lane_width",  # old name for backward compat
            "offensive-three-second-lane-width",
        ],
        int,
        1,
    )
    optional["three_second_lane_height"] = _get_param(
        params,
        [
            "three_second_lane_height",
            "three-second-lane-height",
        ],
        int,
        3,
    )
    optional["three_second_max_steps"] = _get_param(
        params,
        [
            "three_second_max_steps",
            "three-second-max-steps",
            "illegal_defense_max_steps",  # old name for backward compat
            "illegal-defense-max-steps",
            "offensive_three_second_max_steps",  # old name for backward compat
            "offensive-three-second-max-steps",
        ],
        int,
        3,
    )
    optional["illegal_defense_enabled"] = _get_param(
        params,
        ["illegal_defense_enabled", "illegal-defense-enabled"],
        lambda v: str(v).lower() in ["1", "true", "yes", "y", "t"],
        False,
    )
    optional["offensive_three_seconds_enabled"] = _get_param(
        params,
        [
            "offensive_three_seconds_enabled",
            "offensive-three-seconds-enabled",
            "offensive_three_seconds",  # CLI flag name
            "offensive-three-seconds",
        ],
        lambda v: str(v).lower() in ["1", "true", "yes", "y", "t"],
        False,
    )

    # Observation controls (optional; used by backend/main.py)
    optional["use_egocentric_obs"] = _get_param(
        params,
        ["use_egocentric_obs", "use-egocentric-obs"],
        lambda v: str(v).lower() in ["1", "true", "yes", "y", "t"],
        True,
    )
    optional["egocentric_rotate_to_hoop"] = _get_param(
        params,
        ["egocentric_rotate_to_hoop", "egocentric-rotate-to-hoop"],
        lambda v: str(v).lower() in ["1", "true", "yes", "y", "t"],
        True,
    )
    optional["include_hoop_vector"] = _get_param(
        params,
        ["include_hoop_vector", "include-hoop-vector"],
        lambda v: str(v).lower() in ["1", "true", "yes", "y", "t"],
        True,
    )
    optional["normalize_obs"] = _get_param(
        params,
        ["normalize_obs", "normalize-obs"],
        lambda v: str(v).lower() in ["1", "true", "yes", "y", "t"],
        True,
    )

    # Reward parameters (to ensure evaluation uses the same reward shaping)
    optional["pass_reward"] = _get_param(
        params, ["pass_reward", "pass-reward"], float, 0.0
    )
    optional["turnover_penalty"] = _get_param(
        params, ["turnover_penalty", "turnover-penalty"], float, 0.0
    )
    optional["violation_reward"] = _get_param(
        params, ["violation_reward", "violation-reward"], float, 2.0
    )
    optional["made_shot_reward_inside"] = _get_param(
        params, ["made_shot_reward_inside", "made-shot-reward-inside"], float, 2.0
    )
    optional["made_shot_reward_three"] = _get_param(
        params, ["made_shot_reward_three", "made-shot-reward-three"], float, 3.0
    )
    optional["missed_shot_penalty"] = _get_param(
        params, ["missed_shot_penalty", "missed-shot-penalty"], float, 0.0
    )
    optional["potential_assist_reward"] = _get_param(
        params, ["potential_assist_reward", "potential-assist-reward"], float, 0.1
    )
    optional["full_assist_bonus"] = _get_param(
        params, ["full_assist_bonus", "full-assist-bonus"], float, 0.2
    )
    # Backward-compat: support both names; map to unified 'assist_window'
    optional["assist_window"] = _get_param(
        params,
        [
            "assist_window",
            "assist-window",
            "assist_window_steps",
            "assist-window-steps",
        ],
        int,
        2,
    )
    optional["potential_assist_pct"] = _get_param(
        params, ["potential_assist_pct", "potential-assist-pct"], float, 0.10
    )
    optional["full_assist_bonus_pct"] = _get_param(
        params, ["full_assist_bonus_pct", "full-assist-bonus-pct"], float, 0.05
    )
    # Realistic passing steal parameters
    optional["base_steal_rate"] = _get_param(
        params, ["base_steal_rate", "base-steal-rate"], float, 0.35
    )
    optional["steal_perp_decay"] = _get_param(
        params, ["steal_perp_decay", "steal-perp-decay"], float, 1.5
    )
    optional["steal_distance_factor"] = _get_param(
        params, ["steal_distance_factor", "steal-distance-factor"], float, 0.08
    )
    optional["steal_position_weight_min"] = _get_param(
        params, ["steal_position_weight_min", "steal-position-weight-min"], float, 0.3
    )
    # Pass parameters
    optional["pass_arc_degrees"] = _get_param(
        params, ["pass_arc_degrees", "pass-arc-degrees"], float, 60.0
    )
    optional["pass_oob_turnover_prob"] = _get_param(
        params, ["pass_oob_turnover_prob", "pass-oob-turnover-prob"], float, 1.0
    )
    optional["enable_pass_gating"] = _get_param(
        params,
        ["enable_pass_gating", "enable-pass-gating"],
        lambda v: str(v).lower() in ["1", "true", "yes", "y", "t"],
        True,
    )
    # Illegal action policy
    optional["illegal_action_policy"] = _get_param(
        params, ["illegal_action_policy", "illegal-action-policy"], str, "noop"
    )
    
    # Role flag encoding (for backward compatibility with old models)
    optional["role_flag_offense_value"] = _get_param(
        params,
        ["role_flag_offense_value", "role-flag-offense-value"],
        float,
        1.0,  # Default to legacy encoding
    )
    optional["role_flag_defense_value"] = _get_param(
        params,
        ["role_flag_defense_value", "role-flag-defense-value"],
        float,
        0.0,  # Default to legacy encoding
    )
    optional["role_flag_encoding_version"] = _get_param(
        params,
        ["role_flag_encoding_version", "role-flag-encoding-version"],
        str,
        "legacy",  # Default to legacy encoding
    )
    
    # Note: use_vec_normalize is deprecated and not passed to environment
    # (kept in train.py for MLflow compatibility only)
    return required, optional


def get_mlflow_phi_shaping_params(
    client: mlflow.tracking.MlflowClient, run_id: str
) -> dict:
    """Fetch phi shaping parameters from an MLflow run.

    These are returned separately so they can be used for reward calculation
    independently from the Phi Shaping tab (which is for experimentation).

    Returns a dict with:
      - enable_phi_shaping: bool
      - phi_beta: float (uses phi_beta_end if available, else phi_beta_start)
      - reward_shaping_gamma: float
      - phi_use_ball_handler_only: bool
      - phi_blend_weight: float
      - phi_aggregation_mode: str
    """
    run = client.get_run(run_id)
    params = run.data.params

    phi_params = {}

    # Enable phi shaping
    phi_params["enable_phi_shaping"] = _get_param(
        params,
        ["enable_phi_shaping", "enable-phi-shaping"],
        lambda v: str(v).lower() in ["1", "true", "yes", "y", "t"],
        False,
    )

    # Phi beta - prefer phi_beta_end (final value) over phi_beta_start
    # This is what the model was trained with at the end of training
    phi_beta_end = _get_param(
        params,
        ["phi_beta_end", "phi-beta-end"],
        float,
        None,
    )
    phi_beta_start = _get_param(
        params,
        ["phi_beta_start", "phi-beta-start", "phi_beta", "phi-beta"],
        float,
        0.0,
    )
    phi_params["phi_beta"] = (
        phi_beta_end if phi_beta_end is not None else phi_beta_start
    )

    # Reward shaping gamma (should match training gamma)
    phi_params["reward_shaping_gamma"] = _get_param(
        params,
        ["reward_shaping_gamma", "reward-shaping-gamma"],
        float,
        1.0,
    )

    # Ball handler only mode
    phi_params["phi_use_ball_handler_only"] = _get_param(
        params,
        ["phi_use_ball_handler_only", "phi-use-ball-handler-only"],
        lambda v: str(v).lower() in ["1", "true", "yes", "y", "t"],
        False,
    )

    # Blend weight
    phi_params["phi_blend_weight"] = _get_param(
        params,
        ["phi_blend_weight", "phi-blend-weight"],
        float,
        0.0,
    )

    # Aggregation mode
    phi_params["phi_aggregation_mode"] = _get_param(
        params,
        ["phi_aggregation_mode", "phi-aggregation-mode"],
        str,
        "team_best",
    )

    return phi_params


def get_mlflow_training_params(
    client: mlflow.tracking.MlflowClient, run_id: str
) -> dict:
    """Fetch PPO training parameters from an MLflow run.

    Returns a dict with training hyperparameters like learning_rate, gamma, etc.
    """
    run = client.get_run(run_id)
    params = run.data.params

    training_params = {}

    # PPO Core Hyperparameters
    training_params["learning_rate"] = _get_param(
        params, ["learning_rate", "learning-rate"], float, 2.5e-4
    )
    training_params["n_steps"] = _get_param(
        params, ["n_steps", "n-steps"], int, 2048
    )
    training_params["batch_size"] = _get_param(
        params, ["batch_size", "batch-size"], int, 64
    )
    training_params["n_epochs"] = _get_param(
        params, ["n_epochs", "n-epochs"], int, 10
    )
    training_params["gamma"] = _get_param(
        params, ["gamma"], float, 0.99
    )
    training_params["gae_lambda"] = _get_param(
        params, ["gae_lambda", "gae-lambda"], float, 0.95
    )
    training_params["clip_range"] = _get_param(
        params, ["clip_range", "clip-range"], float, 0.2
    )
    training_params["vf_coef"] = _get_param(
        params, ["vf_coef", "vf-coef"], float, 0.5
    )
    training_params["ent_coef"] = _get_param(
        params, ["ent_coef", "ent-coef"], float, 0.0
    )
    training_params["ent_coef_start"] = _get_param(
        params, ["ent_coef_start", "ent-coef-start"], 
        lambda v: None if v == "" or v == "None" else float(v), 
        None
    )
    training_params["ent_coef_end"] = _get_param(
        params, ["ent_coef_end", "ent-coef-end"],
        lambda v: None if v == "" or v == "None" else float(v),
        None
    )

    # Network Architecture
    # net_arch_used is logged after policy creation and shows actual architecture
    training_params["net_arch_used"] = _get_param(
        params, ["net_arch_used", "net-arch-used"], str, None
    )
    # Original net_arch from CLI (may be None if using pi/vf separately)
    training_params["net_arch"] = _get_param(
        params, ["net_arch", "net-arch"], str, None
    )
    # Separate pi/vf architectures - stored as list strings like "[64, 64]"
    training_params["net_arch_pi"] = _get_param(
        params, ["net_arch_pi", "net-arch-pi", "net_arch-pi"], str, None
    )
    training_params["net_arch_vf"] = _get_param(
        params, ["net_arch_vf", "net-arch-vf", "net_arch-vf"], str, None
    )
    training_params["use_dual_critic"] = _get_param(
        params,
        ["use_dual_critic", "use-dual-critic"],
        lambda v: str(v).lower() in ["1", "true", "yes", "y", "t"],
        False,
    )
    training_params["policy_class"] = _get_param(
        params, ["policy_class", "policy-class"], str, "PassBiasMultiInputPolicy"
    )

    # Training Setup
    training_params["num_envs"] = _get_param(
        params, ["num_envs", "num-envs"], int, 8
    )
    # Alternation settings
    training_params["alternations"] = _get_param(
        params, ["alternations"], int, 10
    )
    training_params["steps_per_alternation"] = _get_param(
        params, ["steps_per_alternation", "steps-per-alternation"], int, 1
    )
    training_params["steps_per_alternation_end"] = _get_param(
        params, ["steps_per_alternation_end", "steps-per-alternation-end"], 
        lambda v: None if v in ("None", "null", "") else int(v), 
        None
    )
    training_params["steps_per_alternation_schedule"] = _get_param(
        params, ["steps_per_alternation_schedule", "steps-per-alternation-schedule"], 
        str, 
        "linear"
    )
    # Total planned timesteps (calculated, accounts for SPA schedule)
    training_params["total_timesteps_planned"] = _get_param(
        params, ["total_timesteps_planned"], 
        int, 
        None
    )

    # Self-play and Opponent Sampling
    training_params["deterministic_opponent"] = _get_param(
        params,
        ["deterministic_opponent", "deterministic-opponent"],
        lambda v: str(v).lower() in ["1", "true", "yes", "y", "t"],
        False,
    )
    training_params["per_env_opponent_sampling"] = _get_param(
        params,
        ["per_env_opponent_sampling", "per-env-opponent-sampling"],
        lambda v: str(v).lower() in ["1", "true", "yes", "y", "t"],
        False,
    )
    training_params["opponent_sample_k"] = _get_param(
        params, ["opponent_sample_k", "opponent-sample-k"], int, 5
    )
    training_params["opponent_pool_beta"] = _get_param(
        params, ["opponent_pool_beta", "opponent-pool-beta"], float, 0.5
    )
    training_params["opponent_pool_exploration"] = _get_param(
        params, ["opponent_pool_exploration", "opponent-pool-exploration"], float, 0.5
    )

    # Pass Logit Bias
    training_params["pass_logit_bias"] = _get_param(
        params, ["pass_logit_bias", "pass-logit-bias"], float, 0.0
    )
    training_params["pass_prob_min"] = _get_param(
        params, ["pass_prob_min", "pass-prob-min"], float, 0.0
    )

    return training_params
