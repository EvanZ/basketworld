import re
import os
import time
import sys
from pathlib import Path
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
import numpy as np
import basketworld
from basketworld.envs.basketworld_env_v2 import Team, ActionType
from stable_baselines3 import PPO
# Import custom policies so they're available when loading saved models
from basketworld.utils.policies import PassBiasMultiInputPolicy, PassBiasDualCriticPolicy
import mlflow
import torch
import copy
from datetime import datetime
import imageio
from basketworld.utils.action_resolution import IllegalActionStrategy
from basketworld.utils.evaluation_helpers import get_outcome_category
from basketworld.utils.mlflow_params import (
    get_mlflow_params,
    get_mlflow_phi_shaping_params,
    get_mlflow_training_params,
)

# Ensure repository root is on sys.path so absolute imports work when launched as a script
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.backend.schemas import (
    ActionRequest,
    BatchUpdatePositionRequest,
    EvaluationRequest,
    InitGameRequest,
    ListPoliciesRequest,
    MCTSAdviseRequest,
    OffenseSkillsPayload,
    PassStealPreviewRequest,
    SaveEpisodeRequest,
    SetBallHolderRequest,
    SetOffenseSkillsRequest,
    SetPassTargetStrategyRequest,
    SetPhiParamsRequest,
    SwapPoliciesRequest,
    UpdatePositionRequest,
    UpdateShotClockRequest,
)
from app.backend.state import (
    GameState,
    _capture_turn_start_snapshot,
    _role_flag_value_for_team,
    get_full_game_state,
    game_state,
)
from app.backend.policies import (
    _compute_param_counts_from_policy,
    get_latest_policies_from_run,
    get_unified_policy_path,
    list_policies_from_run,
)
from app.backend.observations import (
    _build_role_conditioned_obs,
    _compute_policy_probabilities_for_obs,
    _compute_q_values_for_player,
    _compute_state_values_from_obs,
    _predict_policy_actions,
    calculate_phi_from_ep_data,
    compute_policy_probabilities,
)
from app.backend.mcts import _run_mcts_advisor
from app.backend.routes.evaluation_routes import router as evaluation_router
from app.backend.routes.analytics_routes import router as analytics_router
from app.backend.routes.media_routes import router as media_router
from app.backend.routes.admin_routes import router as admin_router
from app.backend.routes.lifecycle_routes import router as lifecycle_router
from app.backend.routes.policy_routes import router as policy_router


# --- FastAPI App ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routes
app.include_router(evaluation_router)
app.include_router(analytics_router)
app.include_router(media_router)
app.include_router(admin_router)
app.include_router(lifecycle_router)
app.include_router(policy_router)
