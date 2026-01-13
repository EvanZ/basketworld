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


class ListPoliciesRequest(BaseModel):
    run_id: str


class SwapPoliciesRequest(BaseModel):
    user_policy_name: str | None = None
    opponent_policy_name: str | None = None


class PassStealPreviewRequest(BaseModel):
    positions: list[tuple[int, int]]
    ball_holder: int


class ActionRequest(BaseModel):
    actions: dict[str, object]  # Accept nested dicts or ints; normalized in the route.
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


class OffenseSkillsPayload(BaseModel):
    layup: List[float]
    three_pt: List[float]
    dunk: List[float]


class SetOffenseSkillsRequest(BaseModel):
    skills: OffenseSkillsPayload | None = None
    reset_to_sampled: bool = False


class SetPassTargetStrategyRequest(BaseModel):
    strategy: str
