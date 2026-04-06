from typing import List, Literal, Optional

from pydantic import BaseModel


class CustomEvalSetup(BaseModel):
    initial_positions: list[tuple[int, int]] | None = None
    ball_holder: int | None = None
    shooting_mode: Literal["random", "fixed"] = "random"
    offense_skills: dict[str, list[float]] | None = None


class InitGameRequest(BaseModel):
    run_id: str
    user_team_name: str  # "OFFENSE" or "DEFENSE"
    run_name: str | None = None
    unified_policy_name: str | None = None
    opponent_unified_policy_name: str | None = None
    # Optional overrides
    spawn_distance: int | None = None
    defender_spawn_distance: int | None = None
    allow_dunks: bool | None = None
    dunk_pct: float | None = None


class TemplateBootstrapRequest(BaseModel):
    run_id: str | None = None
    user_team_name: str = "OFFENSE"
    players_per_side: int = 3
    allow_dunks: bool | None = None


class ApplyStartTemplateRequest(BaseModel):
    template_id: str
    mirrored: bool | None = None
    apply_to_state: bool = True
    seed: int | None = None


class ListPoliciesRequest(BaseModel):
    run_id: str


class SwapPoliciesRequest(BaseModel):
    user_policy_name: str | None = None
    opponent_policy_name: str | None = None


class PassStealPreviewRequest(BaseModel):
    positions: list[tuple[int, int]]
    ball_holder: int


class ActionRequest(BaseModel):
    actions: dict[str, object]  # Accept ints, action names, or structured payloads like {type:"PASS", target:id}.
    player_deterministic: bool | None = None
    opponent_deterministic: bool | None = None
    use_mcts: bool | None = None
    team: str | None = None  # Optional override of training_team ("OFFENSE"/"DEFENSE")
    mcts_player_id: int | None = None
    mcts_player_ids: list[int] | None = None
    mcts_max_depth: int | None = None
    mcts_time_budget_ms: int | None = None
    mcts_exploration_c: float | None = None
    mcts_use_priors: bool | None = None


class MCTSAdviseRequest(BaseModel):
    player_id: int | None = None
    max_depth: int | None = None
    time_budget_ms: int | None = None
    exploration_c: float | None = None
    use_priors: bool | None = True


class SetPhiParamsRequest(BaseModel):
    enable_phi_shaping: bool | None = None
    phi_beta: float | None = None
    reward_shaping_gamma: float | None = None
    phi_use_ball_handler_only: bool | None = None
    phi_blend_weight: float | None = None
    phi_aggregation_mode: str | None = None


class EvaluationRequest(BaseModel):
    num_episodes: int = 100
    player_deterministic: bool = True
    opponent_deterministic: bool = True
    custom_setup: CustomEvalSetup | None = None
    randomize_offense_permutation: bool = False
    intent_selection_mode: Literal["learned_sample", "best_intent", "uniform_random"] = "learned_sample"


class SaveEpisodeRequest(BaseModel):
    frames: List[str]  # Base64-encoded PNG images
    durations: Optional[List[float]] = None  # Optional per-frame durations in seconds
    step_duration_ms: Optional[float] = None  # Optional fallback duration per step in milliseconds


class UpdatePositionRequest(BaseModel):
    player_id: int
    q: int
    r: int


class UpdateShotClockRequest(BaseModel):
    delta: int


class SetBallHolderRequest(BaseModel):
    player_id: int


class BatchUpdatePositionRequest(BaseModel):
    updates: List[UpdatePositionRequest]


class SetIntentStateRequest(BaseModel):
    active: bool
    intent_index: int
    intent_age: int


class ReplayCounterfactualRequest(BaseModel):
    player_deterministic: bool = True
    opponent_deterministic: bool = True
    max_steps: int = 256


class PlaybookAnalysisRequest(BaseModel):
    intent_indices: List[int]
    num_rollouts: int = 16
    max_steps: int = 8
    run_to_end: bool = False
    use_snapshot: bool = True
    player_deterministic: bool = False
    opponent_deterministic: bool = True


class OffenseSkillsPayload(BaseModel):
    layup: List[float]
    three_pt: List[float]
    dunk: List[float]


class SetOffenseSkillsRequest(BaseModel):
    skills: OffenseSkillsPayload | None = None
    reset_to_sampled: bool = False


class SetPassTargetStrategyRequest(BaseModel):
    strategy: str


class SetPassLogitBiasRequest(BaseModel):
    bias: float | None = None


class SetPressureParamsRequest(BaseModel):
    reset_to_mlflow_defaults: bool = False
    scope: str | None = None
    reset_group: str | None = None
    reset_keys: List[str] | None = None
    # Shot make probability distance decay
    three_pt_extra_hex_decay: float | None = None
    # Shot pressure
    shot_pressure_enabled: bool | None = None
    shot_pressure_max: float | None = None
    shot_pressure_lambda: float | None = None
    shot_pressure_arc_degrees: float | None = None
    # Pass interception
    base_steal_rate: float | None = None
    steal_perp_decay: float | None = None
    steal_distance_factor: float | None = None
    steal_position_weight_min: float | None = None
    # Defender turnover pressure
    defender_pressure_distance: int | None = None
    defender_pressure_turnover_chance: float | None = None
    defender_pressure_decay_lambda: float | None = None


class PlayableStartRequest(BaseModel):
    players_per_side: int
    difficulty: Literal["easy", "medium", "hard"]
    period_mode: Literal["period", "halves", "quarters"] = "period"
    period_length_minutes: int = 5


class PlayableDemoTakeoverRequest(BaseModel):
    period_mode: Literal["period", "halves", "quarters"] = "period"
    period_length_minutes: int = 5


class PlayableStepRequest(BaseModel):
    actions: dict[str, object]
    auto_user_actions: bool = False
