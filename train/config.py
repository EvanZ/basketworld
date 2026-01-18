import argparse


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train PPO models using self-play.")
    parser.add_argument(
        "--grid-size", type=int, default=12, help="The size of the grid."
    )
    parser.add_argument(
        "--court-rows",
        dest="court_rows",
        type=int,
        default=None,
        help="Number of rows in the court (defaults to grid-size if None).",
    )
    parser.add_argument(
        "--court-cols",
        dest="court_cols",
        type=int,
        default=None,
        help="Number of columns in the court (defaults to grid-size if None).",
    )
    parser.add_argument(
        "--layup-pct", type=float, default=0.60, help="Percentage of layups."
    )
    parser.add_argument(
        "--layup-std",
        type=float,
        default=0.0,
        help="Std dev for per-player layup percentage sampling.",
    )
    parser.add_argument(
        "--three-pt-pct", type=float, default=0.37, help="Percentage of three-pointers."
    )
    parser.add_argument(
        "--three-pt-std",
        type=float,
        default=0.0,
        help="Std dev for per-player three-point percentage sampling.",
    )
    parser.add_argument(
        "--three-point-distance",
        type=float,
        default=4.0,
        help="Hex distance defining the three-point line.",
    )
    parser.add_argument(
        "--three-point-short-distance",
        type=float,
        default=None,
        help="Optional short corner distance for 3pt line (like NBA). If None, uses circular arc.",
    )
    parser.add_argument(
        "--players", type=int, default=2, help="Number of players per side."
    )
    parser.add_argument(
        "--shot-clock", type=int, default=20, help="Steps in the shot clock."
    )
    parser.add_argument(
        "--min-shot-clock",
        dest="min_shot_clock",
        type=int,
        default=10,
        help="Minimum steps for randomly initialized shot clock at reset.",
    )
    parser.add_argument(
        "--alternations",
        type=int,
        default=10,
        help="Number of times to alternate training.",
    )
    parser.add_argument(
        "--steps-per-alternation",
        type=int,
        default=1,
        help="Starting timesteps to train each policy per alternation (or constant if no end specified).",
    )
    parser.add_argument(
        "--steps-per-alternation-end",
        type=int,
        default=None,
        help="Ending timesteps per alternation. If specified, steps will be scheduled from start to end.",
    )
    parser.add_argument(
        "--steps-per-alternation-schedule",
        type=str,
        default="linear",
        choices=["linear", "log", "constant"],
        help="Schedule type for steps-per-alternation: 'linear' interpolates linearly, 'log' uses logarithmic curve (slower increase early, faster late), 'constant' uses start value only.",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=2048,
        help="PPO hyperparameter: Number of steps to run for each environment per update.",
    )
    parser.add_argument(
        "--target-kl",
        type=float,
        default=0.025,
        help="PPO hyperparameter: Target KL divergence for early stopping.",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=10,
        help="PPO hyperparameter: Number of epochs when optimizing the surrogate.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="PPO hyperparameter: Discount factor for future rewards.",
    )
    parser.add_argument(
        "--vf-coef",
        type=float,
        default=0.5,
        help="PPO hyperparameter: Weight for value function loss.",
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=0,
        help="PPO hyperparameter: Weight for entropy loss.",
    )
    # Optional entropy schedule across entire training
    parser.add_argument(
        "--ent-coef-start",
        type=float,
        default=None,
        help="If set, start entropy coefficient at this value and decay linearly.",
    )
    parser.add_argument(
        "--ent-coef-end",
        type=float,
        default=None,
        help="If set with --ent-coef-start, end entropy coefficient at this value.",
    )
    parser.add_argument(
        "--ent-schedule",
        type=str,
        choices=["linear", "exp"],
        default="linear",
        help="Entropy schedule type when start/end are provided.",
    )
    parser.add_argument(
        "--ent-bump-updates",
        type=int,
        default=0,
        help="If >0 with schedule, number of PPO updates to multiply entropy at start of each segment.",
    )
    parser.add_argument(
        "--ent-bump-rollouts",
        type=int,
        default=0,
        help="Deprecated alias of --ent-bump-updates; counted as updates.",
    )
    parser.add_argument(
        "--ent-bump-multiplier",
        type=float,
        default=1.0,
        help="Multiplier applied to entropy during bump rollouts (>=1.0).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="PPO hyperparameter: Minibatch size."
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2.5e-4,
        help="Learning rate for PPO optimizers.",
    )
    parser.add_argument(
        "--net-arch",
        type=int,
        nargs="+",
        default=None,
        help="The size of the neural network layers (e.g., 128 128). Default is SB3's default.",
    )
    parser.add_argument(
        "--net-arch-pi",
        type=int,
        nargs="+",
        default=[64, 64],
        help="Actor (policy) MLP hidden sizes, e.g. 64 64. Ignored if --net-arch is set.",
    )
    parser.add_argument(
        "--net-arch-vf",
        type=int,
        nargs="+",
        default=[64, 64],
        help="Critic (value) MLP hidden sizes, e.g. 64 64. Ignored if --net-arch is set.",
    )
    parser.add_argument(
        "--use-dual-critic",
        action="store_true",
        default=False,
        help="Use separate value networks for offense and defense (recommended for zero-sum self-play).",
    )
    parser.add_argument(
        "--use-dual-policy",
        action="store_true",
        default=False,
        help="Use separate action networks for offense and defense. Enables distinct strategies for each role. Implies --use-dual-critic.",
    )
    parser.add_argument(
        "--set-embed-dim",
        type=int,
        default=64,
        help="Set-attention token embedding dimension.",
    )
    parser.add_argument(
        "--set-heads",
        type=int,
        default=4,
        help="Number of attention heads for set-attention policy.",
    )
    parser.add_argument(
        "--set-token-mlp-dim",
        type=int,
        default=64,
        help="Hidden dimension for the set-attention token MLP.",
    )
    parser.add_argument(
        "--set-cls-tokens",
        type=int,
        default=2,
        help="Number of CLS tokens for set-attention policy (2 for dual critics).",
    )
    parser.add_argument(
        "--init-critic-from-run",
        type=str,
        default=None,
        help="MLflow run_id to initialize critic weights from (transfer learning). Only value heads are transferred.",
    )
    parser.add_argument(
        "--continue-run-id",
        type=str,
        default=None,
        help="If set, load latest offense/defense policies from this MLflow run and continue training. Also appends new artifacts using continued alternation indices.",
    )
    parser.add_argument(
        "--continue-schedule-mode",
        type=str,
        choices=["extend", "constant", "restart"],
        default="extend",
        help=(
            "How to handle schedules when continuing training: "
            "'extend' (default) - continue schedules from where they left off, adding more training to the original total; "
            "'constant' - use the final schedule values (where the previous run ended) as constants; "
            "'restart' - restart schedules from scratch using new parameters."
        ),
    )
    parser.add_argument(
        "--restart-entropy-on-continue",
        dest="restart_entropy_on_continue",
        type=lambda v: str(v).lower() in ["1", "true", "yes", "y", "t"],
        default=False,
        help="DEPRECATED: Use --continue-schedule-mode=restart instead. When continuing from a run, reset num_timesteps and reinitialize ent_coef to the schedule start.",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=2,
        help="Run evaluation every N alternations. Set to 0 to disable.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="Number of episodes to run for each evaluation.",
    )
    parser.add_argument(
        "--defender-pressure-distance",
        type=int,
        default=1,
        help="Distance at which defender pressure is applied.",
    )
    parser.add_argument(
        "--defender-pressure-turnover-chance",
        type=float,
        default=0.05,
        help="Chance of a defender pressure turnover.",
    )
    parser.add_argument(
        "--defender-pressure-decay-lambda",
        type=float,
        default=1.0,
        help="Exponential decay rate for defender pressure.",
    )
    parser.add_argument(
        "--tensorboard-path",
        type=str,
        default=None,
        help="Path to save TensorBoard logs (set to None if using MLflow).",
    )
    parser.add_argument(
        "--mlflow-experiment-name",
        type=str,
        default="BasketWorld_Training",
        help="Name of the MLflow experiment.",
    )
    parser.add_argument(
        "--mlflow-run-name", type=str, default=None, help="Name of the MLflow run."
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=8,
        help="Number of parallel environments to run for each policy during training.",
    )
    parser.add_argument(
        "--use-vec-normalize",
        type=lambda v: str(v).lower() in ["1", "true", "yes", "y", "t"],
        default=False,
        help="(DEPRECATED - no longer used) Previously used VecNormalize wrapper. "
        "Kept for MLflow compatibility.",
    )
    parser.add_argument(
        "--shot-pressure-enabled",
        type=lambda v: str(v).lower() in ["1", "true", "yes", "y", "t"],
        default=True,
        help="Enable defender shot pressure model.",
    )
    parser.add_argument(
        "--shot-pressure-max",
        type=float,
        default=0.5,
        help="Max multiplicative reduction at distance 1 (e.g., 0.5 -> up to -50%).",
    )
    parser.add_argument(
        "--shot-pressure-lambda",
        type=float,
        default=1.0,
        help="Exponential decay rate per hex for shot pressure.",
    )
    parser.add_argument(
        "--shot-pressure-arc-degrees",
        type=float,
        default=60.0,
        help="Arc width centered toward basket for pressure eligibility.",
    )
    parser.add_argument(
        "--enable-env-profiling",
        type=lambda v: str(v).lower() in ["1", "true", "yes", "y", "t"],
        default=False,
        help="Enable timing instrumentation inside the environment and log averages to MLflow after each alternation.",
    )
    parser.add_argument(
        "--profiling-sample-rate",
        type=float,
        default=1.0,
        help="Fraction of episodes to profile when profiling is enabled (0.0-1.0). Lower values reduce overhead. Default: 1.0 (profile all episodes).",
    )
    parser.add_argument(
        "--spawn-distance",
        type=int,
        default=3,
        help="minimum distance from basket at which players spawn.",
    )
    parser.add_argument(
        "--max-spawn-distance",
        dest="max_spawn_distance",
        type=lambda v: None if v == "" or str(v).lower() == "none" else int(v),
        default=None,
        help="maximum distance from basket at which players spawn (None = unlimited). Use with --spawn-distance for curriculum learning.",
    )
    parser.add_argument(
        "--defender-spawn-distance",
        dest="defender_spawn_distance",
        type=int,
        default=0,
        help="randomize defender spawn distance from matched offense player (0 = spawn adjacent; N = spawn 1-N hexes away).",
    )
    parser.add_argument(
        "--defender-guard-distance",
        dest="defender_guard_distance",
        type=int,
        default=1,
        help=(
            "Hex distance (N) within which a defender reset their lane counter if guarding "
            "an offensive player while in the lane. 0 disables guarding resets."
        ),
    )
    parser.add_argument(
        "--deterministic-opponent",
        type=lambda v: str(v).lower() in ["1", "true", "yes", "y", "t"],
        default=False,
        help="Use deterministic opponent actions.",
    )
    parser.add_argument(
        "--opponent-pool-size",
        type=int,
        default=10,
        help="Number of recent checkpoints to keep in opponent pool (K parameter).",
    )
    parser.add_argument(
        "--opponent-pool-beta",
        type=float,
        default=0.7,
        help="Geometric decay factor for opponent sampling (0=uniform, 1=most recent only).",
    )
    parser.add_argument(
        "--opponent-pool-exploration",
        type=float,
        default=0.15,
        help="Probability of sampling from ALL history instead of just recent pool (0-1).",
    )
    parser.add_argument(
        "--per-env-opponent-sampling",
        action="store_true",
        help="Sample different opponents for each parallel environment using geometric distribution (prevents forgetting). Each of the --num-envs workers independently samples from last K checkpoints with recency bias. Default: single opponent per alternation.",
    )
    parser.add_argument(
        "--allow-dunks",
        type=lambda v: str(v).lower() in ["1", "true", "yes", "y", "t"],
        default=True,
        help="Allow players to enter basket hex and enable dunk shots from basket cell.",
    )
    parser.add_argument(
        "--dunk-pct",
        type=float,
        default=0.90,
        help="Probability of a dunk (shot from basket cell).",
    )
    parser.add_argument(
        "--dunk-std",
        type=float,
        default=0.0,
        help="Std dev for per-player dunk percentage sampling.",
    )
    parser.add_argument(
        "--use-egocentric-obs",
        type=lambda v: str(v).lower() in ["1", "true", "yes", "y", "t"],
        default=True,
        help="[DEPRECATED] Observations now use absolute coordinates. This flag is ignored.",
    )
    parser.add_argument(
        "--egocentric-rotate-to-hoop",
        type=lambda v: str(v).lower() in ["1", "true", "yes", "y", "t"],
        default=True,
        help="[DEPRECATED] Rotation is no longer used with absolute coordinates. This flag is ignored.",
    )
    parser.add_argument(
        "--include-hoop-vector",
        type=lambda v: str(v).lower() in ["1", "true", "yes", "y", "t"],
        default=True,
        help="Append hoop position vector (absolute coordinates) to observation.",
    )
    parser.add_argument(
        "--normalize-obs",
        type=lambda v: str(v).lower() in ["1", "true", "yes", "y", "t"],
        default=True,
        help="Normalize relative coordinates to roughly [-1,1].",
    )
    parser.add_argument(
        "--use-set-obs",
        dest="use_set_obs",
        type=lambda v: str(v).lower() in ["1", "true", "yes", "y", "t"],
        default=False,
        help="Expose set-based token observations under 'players' and 'globals' "
        "(keeps existing obs keys).",
    )
    parser.add_argument(
        "--mask-occupied-moves",
        type=lambda v: str(v).lower() in ["1", "true", "yes", "y", "t"],
        default=True,
        help="Disallow moves into currently occupied neighboring hexes.",
    )
    parser.add_argument(
        "--enable-pass-gating",
        dest="enable_pass_gating",
        type=lambda v: str(v).lower() in ["1", "true", "yes", "y", "t"],
        default=True,
        help="Mask out pass actions that don't have a teammate in the arc. "
        "This prevents learning to avoid passing due to OOB turnovers.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use for training ('cuda', 'cpu', or 'auto').",
    )
    parser.add_argument(
        "--three-second-lane-width",
        type=int,
        default=1,
        help="Width of the lane in hexes (shared by offense and defense). 1 = 1 hex on each side of center line.",
    )
    parser.add_argument(
        "--three-second-lane-height",
        type=int,
        default=3,
        help="Height of the lane in hexes (shared by offense and defense). 1 = 1 hex on each side of center line.",
    )
    parser.add_argument(
        "--three-second-max-steps",
        type=int,
        default=3,
        help="Maximum steps a player can stay in the lane (shared by offense and defense).",
    )
    parser.add_argument(
        "--illegal-defense-enabled",
        type=lambda v: str(v).lower() in ["1", "true", "yes", "y", "t"],
        default=True,
        help="Enable illegal defense (defensive 3-second) rule.",
    )
    parser.add_argument(
        "--offensive-three-seconds",
        type=lambda v: str(v).lower() in ["1", "true", "yes", "y", "t"],
        default=True,
        help="Enable offensive 3-second violation rule.",
    )
    parser.add_argument(
        "--pass-reward",
        dest="pass_reward",
        type=float,
        default=0.0,
        help="Reward for successful pass (team-averaged).",
    )
    parser.add_argument(
        "--turnover-penalty",
        dest="turnover_penalty",
        type=float,
        default=0.0,
        help="Penalty for turnover (team-averaged).",
    )
    parser.add_argument(
        "--made-shot-reward-inside",
        dest="made_shot_reward_inside",
        type=float,
        default=2.0,
        help="Reward for made 2pt (team-averaged).",
    )
    parser.add_argument(
        "--made-shot-reward-three",
        dest="made_shot_reward_three",
        type=float,
        default=3.0,
        help="Reward for made 3pt (team-averaged).",
    )
    parser.add_argument(
        "--violation-reward",
        dest="violation_reward",
        type=float,
        default=2.0,
        help="Reward for violation (team-averaged).",
    )
    parser.add_argument(
        "--missed-shot-penalty",
        dest="missed_shot_penalty",
        type=float,
        default=0.0,
        help="Penalty for missed shot (team-averaged).",
    )
    parser.add_argument(
        "--potential-assist-reward",
        dest="potential_assist_reward",
        type=float,
        default=0,
        help="Reward for potential assist within window (team-averaged).",
    )
    parser.add_argument(
        "--full-assist-bonus",
        dest="full_assist_bonus",
        type=float,
        default=0,
        help="Additional reward for made shot within assist window (team-averaged).",
    )
    parser.add_argument(
        "--assist-window",
        dest="assist_window",
        type=int,
        default=3,
        help="Steps after pass that count toward assist window.",
    )
    parser.add_argument(
        "--potential-assist-pct",
        dest="potential_assist_pct",
        type=float,
        default=0,
        help="Potential assist reward as % of shot reward.",
    )
    parser.add_argument(
        "--full-assist-bonus-pct",
        dest="full_assist_bonus_pct",
        type=float,
        default=0,
        help="Full assist bonus as % of shot reward.",
    )
    parser.add_argument(
        "--base-steal-rate",
        dest="base_steal_rate",
        type=float,
        default=0.35,
        help="Base steal rate when defender is directly on pass line.",
    )
    parser.add_argument(
        "--steal-perp-decay",
        dest="steal_perp_decay",
        type=float,
        default=1.5,
        help="Exponential decay rate for steal chance perpendicular to pass line.",
    )
    parser.add_argument(
        "--steal-distance-factor",
        dest="steal_distance_factor",
        type=float,
        default=0.08,
        help="Factor by which pass distance increases steal chance.",
    )
    parser.add_argument(
        "--steal-position-weight-min",
        dest="steal_position_weight_min",
        type=float,
        default=0.3,
        help="Minimum steal weight for defenders near passer (1.0 at receiver). Defenders closer to receiver are more dangerous.",
    )
    parser.add_argument(
        "--episode-sample-prob",
        dest="episode_sample_prob",
        type=float,
        default=1e-2,
        help="Probability of sampling an episode for logging.",
    )
    parser.add_argument(
        "--log-episode-artifacts",
        type=lambda v: str(v).lower() in ["1", "true", "yes", "y", "t"],
        default=False,
        help="Log episode CSVs as MLflow artifacts during training. Set to False to reduce I/O overhead and keep timing charts clean. Episodes are still tracked internally.",
    )
    parser.add_argument(
        "--enable-phi-shaping",
        dest="enable_phi_shaping",
        type=lambda v: str(v).lower() in ["1", "true", "yes", "y", "t"],
        default=True,
        help="Enable potential-based reward shaping using best current shot quality.",
    )
    parser.add_argument(
        "--reward-shaping-gamma",
        dest="reward_shaping_gamma",
        type=float,
        default=None,
        help="Discount gamma used inside shaping term (should match PPO gamma).",
    )
    parser.add_argument(
        "--phi-use-ball-handler-only",
        dest="phi_use_ball_handler_only",
        type=lambda v: str(v).lower() in ["1", "true", "yes", "y", "t"],
        default=False,
        help="Use only ball-handler make prob for Phi instead of team best.",
    )
    parser.add_argument(
        "--phi-blend-weight",
        dest="phi_blend_weight",
        type=float,
        default=0.0,
        help="Blend weight w in [0,1] for Phi=(1-w)*aggregate_EP + w*ball_EP (ignored if ball-handler-only).",
    )
    parser.add_argument(
        "--phi-aggregation-mode",
        dest="phi_aggregation_mode",
        type=str,
        choices=[
            "team_best",
            "teammates_best",
            "teammates_avg",
            "team_avg",
            "team_worst",
            "teammates_worst",
        ],
        default="team_best",
        help="How to aggregate teammate EPs: 'team_best' (max including ball), 'teammates_best' (max excluding ball), 'teammates_avg' (mean excluding ball), 'team_avg' (mean including ball), 'team_worst' (min including ball), 'teammates_worst' (min excluding ball).",
    )
    parser.add_argument(
        "--phi-beta-start",
        dest="phi_beta_start",
        type=float,
        default=0.0,
        help="Initial beta multiplier for Phi shaping.",
    )
    parser.add_argument(
        "--phi-beta-end",
        dest="phi_beta_end",
        type=float,
        default=0.0,
        help="Final beta multiplier for Phi shaping (decays to this).",
    )
    parser.add_argument(
        "--phi-bump-updates",
        dest="phi_bump_updates",
        type=int,
        default=0,
        help="Number of PPO updates to bump phi_beta at start of each segment.",
    )
    parser.add_argument(
        "--phi-bump-multiplier",
        dest="phi_bump_multiplier",
        type=float,
        default=1.0,
        help="Multiplier applied to phi_beta during bump updates (>=1.0).",
    )
    parser.add_argument(
        "--pass-arc-start",
        dest="pass_arc_start",
        type=float,
        default=60,
        help="Initial passing arc degrees (e.g., 120).",
    )
    parser.add_argument(
        "--pass-arc-end",
        dest="pass_arc_end",
        type=float,
        default=60,
        help="Final passing arc degrees (e.g., 60).",
    )
    parser.add_argument(
        "--pass-oob-turnover-prob-start",
        dest="pass_oob_turnover_prob_start",
        type=float,
        default=1,
        help="Initial probability that pass without receiver is OOB turnover (e.g., 0.1).",
    )
    parser.add_argument(
        "--pass-oob-turnover-prob-end",
        dest="pass_oob_turnover_prob_end",
        type=float,
        default=1,
        help="Final OOB turnover probability when no receiver (e.g., 1.0).",
    )
    parser.add_argument(
        "--pass-arc-power",
        dest="pass_arc_power",
        type=float,
        default=1.0,
        help="Power applied to arc curriculum progress for steeper initial decay (default: 2.0, use 1.0 for linear).",
    )
    parser.add_argument(
        "--pass-oob-power",
        dest="pass_oob_power",
        type=float,
        default=1.0,
        help="Power applied to OOB curriculum progress for steeper initial decay (default: 2.0, use 1.0 for linear).",
    )
    parser.add_argument(
        "--pass-logit-bias-enabled",
        dest="pass_logit_bias_enabled",
        type=lambda v: str(v).lower() in ["1", "true", "yes", "y", "t"],
        default=False,
        help="Enable additive pass-logit bias.",
    )
    parser.add_argument(
        "--pass-logit-bias-start",
        dest="pass_logit_bias_start",
        type=float,
        default=None,
        help="Initial additive bias added to PASS action logits (e.g., 0.8).",
    )
    parser.add_argument(
        "--pass-logit-bias-end",
        dest="pass_logit_bias_end",
        type=float,
        default=None,
        help="Final additive bias (0 to disable at end).",
    )
    return parser


def get_args(argv=None):
    return get_parser().parse_args(argv)
