import tempfile
import re
import os
import multiprocessing as mp
import hashlib
import time
import math
from typing import Dict, List, Optional, Tuple, Literal
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
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
from basketworld.utils.evaluation_helpers import get_outcome_category
from basketworld.utils.action_resolution import (
    IllegalActionStrategy,
    get_policy_action_probabilities,
    resolve_illegal_actions,
)
from basketworld.utils.mlflow_params import (
    get_mlflow_params,
    get_mlflow_phi_shaping_params,
    get_mlflow_training_params,
)


# --- Globals ---
# This is a simple way to manage state for a single-user demo.
# For a multi-user app, you would need a more robust session management system.


def _compute_param_counts_from_policy(policy_obj):
    """Return trainable parameter counts for shared trunk, policy heads, and value heads."""
    try:
        model = policy_obj.policy
    except Exception:
        return None

    def count_params(module):
        try:
            return sum(p.numel() for p in module.parameters() if getattr(p, "requires_grad", False))
        except Exception:
            return 0

    shared_trunk = 0
    for attr in ("features_extractor", "mlp_extractor"):
        if hasattr(model, attr):
            shared_trunk += count_params(getattr(model, attr))

    policy_heads = 0
    for attr in ("action_net", "action_net_offense", "action_net_defense"):
        if hasattr(model, attr):
            policy_heads += count_params(getattr(model, attr))

    log_std_count = 0
    if hasattr(model, "log_std") and isinstance(model.log_std, torch.nn.Parameter):
        if model.log_std.requires_grad:
            try:
                log_std_count = int(model.log_std.numel())
            except Exception:
                log_std_count = 0

    value_heads = 0
    for attr in ("value_net", "value_net_offense", "value_net_defense"):
        if hasattr(model, attr):
            value_heads += count_params(getattr(model, attr))

    total = shared_trunk + policy_heads + value_heads + log_std_count
    return {
        "total": int(total),
        "shared_trunk": int(shared_trunk),
        "policy_heads": int(policy_heads + log_std_count),
        "value_heads": int(value_heads),
        "log_std": int(log_std_count),
    }


class GameState:
    def __init__(self):
        self.env = None
        self.offense_policy = None
        self.defense_policy = None
        self.unified_policy = None
        self.user_team: Team = None
        self.obs = None
        self.frames = []  # List of RGB frames for the current episode
        self.reward_history = []  # Track rewards for each step
        self.episode_rewards = {"offense": 0.0, "defense": 0.0}  # Running totals
        self.shot_log = []  # Per-step shot attempts with probability and result
        self.phi_log = []  # Per-step Phi diagnostics and EPs
        # Track which policies are currently loaded so we can persist logs across episodes
        self.offense_policy_key: str | None = None
        self.defense_policy_key: str | None = None
        self.unified_policy_key: str | None = None
        # Opponent unified policy (if different from unified)
        self.opponent_unified_policy_key: str | None = None
        # Self-play / replay tracking
        self.self_play_active: bool = False
        self.replay_seed: int | None = None
        self.replay_initial_positions: list[tuple[int, int]] | None = None
        self.replay_ball_holder: int | None = None
        self.replay_shot_clock: int | None = None
        self.replay_offense_skills: dict | None = None  # Store sampled skills for consistency
        self.sampled_offense_skills: dict | None = None  # Baseline skills from initial game creation
        self.actions_log: list[list[int]] = []  # full action arrays per step
        # General replay buffers (manual or AI). We store full game states for instant replay
        self.episode_states: list[dict] = (
            []
        )  # includes initial state and each post-step state
        # MLflow run metadata
        self.run_id: str | None = None
        self.run_name: str | None = None
        # MLflow phi shaping parameters (used for Rewards tab calculations)
        # This is separate from env.phi_beta etc which can be modified in Phi Shaping tab
        self.mlflow_phi_shaping_params: dict | None = None
        # MLflow training parameters (PPO hyperparameters)
        self.mlflow_training_params: dict | None = None
        # Role flag encoding (for backward compatibility with old models)
        self.role_flag_offense: float = 1.0  # Default to new encoding
        self.role_flag_defense: float = -1.0  # Default to new encoding
        # Cache previous observation to handle race condition between move-recorded and step
        self.prev_obs: dict | None = None
        # Turn-start snapshot for frontend resets
        self.turn_start_positions: list[tuple[int, int]] | None = None
        self.turn_start_ball_holder: int | None = None
        self.turn_start_shot_clock: int | None = None
        # Parallel evaluation support - store params/paths for worker recreation
        self.env_required_params: dict | None = None
        self.env_optional_params: dict | None = None
        self.unified_policy_path: str | None = None
        self.opponent_policy_path: str | None = None


game_state = GameState()


def _role_flag_value_for_team(team: Team) -> float:
    if team == Team.OFFENSE:
        value = getattr(game_state, "role_flag_offense", None)
        return float(value if value is not None else 1.0)
    value = getattr(game_state, "role_flag_defense", None)
    return float(value if value is not None else -1.0)


def _capture_turn_start_snapshot():
    """Store the current environment positions/ball holder/shot clock as the baseline for the turn."""
    if not game_state.env:
        return
    env = game_state.env
    try:
        game_state.turn_start_positions = [
            (int(pos[0]), int(pos[1])) for pos in getattr(env, "positions", [])
        ]
    except Exception:
        game_state.turn_start_positions = None
    game_state.turn_start_ball_holder = (
        int(env.ball_holder) if getattr(env, "ball_holder", None) is not None else None
    )
    game_state.turn_start_shot_clock = int(getattr(env, "shot_clock", 0))


def _clone_obs_with_role_flag(obs: Dict, role_flag_value: float) -> Dict:
    cloned = {
        "obs": np.copy(obs["obs"]),
        "action_mask": obs["action_mask"],
        "role_flag": np.array([role_flag_value], dtype=np.float32),
    }
    skills = obs.get("skills")
    if skills is not None:
        cloned["skills"] = np.copy(skills)
    else:
        cloned["skills"] = None
    return cloned


def _team_player_ids(env, team: Team) -> List[int]:
    if team == Team.OFFENSE:
        return list(getattr(env, "offense_ids", []))
    return list(getattr(env, "defense_ids", []))


def _predict_actions_for_team(
    policy: PPO,
    base_obs: Dict,
    env,
    team: Team,
    deterministic: bool,
    strategy: IllegalActionStrategy,
) -> Tuple[Dict[int, int], Dict[int, np.ndarray]]:
    actions_by_player: Dict[int, int] = {}
    probs_by_player: Dict[int, np.ndarray] = {}

    if policy is None or base_obs is None or env is None:
        return actions_by_player, probs_by_player

    team_ids = _team_player_ids(env, team)
    if not team_ids:
        return actions_by_player, probs_by_player

    role_flag_value = _role_flag_value_for_team(team)
    conditioned_obs = _clone_obs_with_role_flag(base_obs, role_flag_value)

    try:
        raw_actions, _ = policy.predict(conditioned_obs, deterministic=deterministic)
    except Exception:
        return actions_by_player, probs_by_player

    raw_actions = np.array(raw_actions).reshape(-1)
    action_len = raw_actions.shape[0]
    team_mask = base_obs["action_mask"][team_ids]

    # Legacy policies output actions for every player; new policies output players_per_side only.
    if action_len == len(team_ids):
        team_pred_actions = raw_actions
    elif action_len == getattr(env, "n_players", action_len):
        team_pred_actions = raw_actions[team_ids]
    else:
        # Fallback: truncate/pad to team size
        team_pred_actions = raw_actions[: len(team_ids)]

    probs = get_policy_action_probabilities(policy, conditioned_obs)
    if probs is not None:
        probs = [
            np.asarray(p, dtype=np.float32) for p in probs
        ]
        if len(probs) == getattr(env, "n_players", len(probs)):
            team_probs = [probs[int(pid)] for pid in team_ids]
        else:
            team_probs = probs[: len(team_ids)]
    else:
        team_probs = None

    resolved_actions = resolve_illegal_actions(
        np.array(team_pred_actions),
        team_mask,
        strategy,
        deterministic,
        team_probs,
    )

    for idx, pid in enumerate(team_ids):
        actions_by_player[int(pid)] = int(resolved_actions[idx])
        if team_probs is not None and idx < len(team_probs):
            probs_by_player[int(pid)] = np.asarray(team_probs[idx], dtype=np.float32)

    return actions_by_player, probs_by_player


def _predict_policy_actions(
    policy: Optional[PPO],
    base_obs: Dict,
    env,
    deterministic: bool,
    strategy: IllegalActionStrategy,
) -> Tuple[Optional[np.ndarray], Optional[List[np.ndarray]]]:
    if policy is None or base_obs is None or env is None:
        return None, None

    num_players = env.n_players
    num_actions = len(ActionType)
    full_actions = np.zeros(num_players, dtype=int)
    probs_per_player: List[np.ndarray] = [
        np.zeros(num_actions, dtype=np.float32) for _ in range(num_players)
    ]

    for team in (Team.OFFENSE, Team.DEFENSE):
        team_actions, team_probs = _predict_actions_for_team(
            policy,
            base_obs,
            env,
            team,
            deterministic,
            strategy,
        )
        for pid, action in team_actions.items():
            full_actions[int(pid)] = int(action)
        for pid, prob_vec in team_probs.items():
            probs_per_player[int(pid)] = prob_vec

    return full_actions, probs_per_player


# --- Parallel Evaluation Helpers ---
# These functions enable multi-process evaluation for speedup on multi-core systems

# Worker-local storage (each process has its own copy)
_worker_state = {}


def _init_evaluation_worker(
    required_params: dict,
    optional_params: dict,
    unified_policy_path: str,
    opponent_policy_path: str | None,
    user_team_name: str,
    role_flag_offense: float,
    role_flag_defense: float,
):
    """Initialize a worker process with its own environment and policies.
    
    This function is called once per worker when the ProcessPoolExecutor starts.
    Each worker maintains its own copies of env and policies in _worker_state.
    """
    global _worker_state
    
    # Import all required modules for worker functions
    import numpy as np
    import basketworld
    from stable_baselines3 import PPO
    from basketworld.utils.policies import PassBiasMultiInputPolicy, PassBiasDualCriticPolicy
    from basketworld.envs.basketworld_env_v2 import Team
    from basketworld.utils.action_resolution import (
        IllegalActionStrategy,
        get_policy_action_probabilities,
        resolve_illegal_actions,
    )
    
    # Create environment with same parameters
    env = basketworld.HexagonBasketballEnv(
        **required_params,
        **optional_params,
        render_mode=None,  # No rendering needed for evaluation
    )
    
    # Load policies
    custom_objects = {
        "policy_class": PassBiasDualCriticPolicy,
        "PassBiasDualCriticPolicy": PassBiasDualCriticPolicy,
        "PassBiasMultiInputPolicy": PassBiasMultiInputPolicy,
    }
    unified_policy = PPO.load(unified_policy_path, custom_objects=custom_objects)
    opponent_policy = (
        PPO.load(opponent_policy_path, custom_objects=custom_objects)
        if opponent_policy_path
        else None
    )
    
    # Parse user team
    user_team = Team.OFFENSE if user_team_name == "OFFENSE" else Team.DEFENSE
    
    # Store in worker-local state (including imports needed by worker functions)
    _worker_state = {
        "env": env,
        "unified_policy": unified_policy,
        "opponent_policy": opponent_policy,
        "user_team": user_team,
        "role_flag_offense": role_flag_offense,
        "role_flag_defense": role_flag_defense,
        # Store imports for worker functions
        "np": np,
        "Team": Team,
        "IllegalActionStrategy": IllegalActionStrategy,
        "get_policy_action_probabilities": get_policy_action_probabilities,
        "resolve_illegal_actions": resolve_illegal_actions,
    }


def _worker_role_flag_value(team) -> float:
    """Get the role flag value for a team in the worker context."""
    Team = _worker_state["Team"]
    if team == Team.OFFENSE:
        return _worker_state.get("role_flag_offense", 1.0)
    return _worker_state.get("role_flag_defense", -1.0)


def _worker_clone_obs_with_role_flag(obs: dict, role_flag_value: float) -> dict:
    """Clone observation with role flag for worker context."""
    np = _worker_state["np"]
    cloned = {
        "obs": np.copy(obs["obs"]),
        "action_mask": obs["action_mask"],
        "role_flag": np.array([role_flag_value], dtype=np.float32),
    }
    skills = obs.get("skills")
    if skills is not None:
        cloned["skills"] = np.copy(skills)
    else:
        cloned["skills"] = None
    return cloned


# --- Evaluation helpers ---
def _init_player_stats(n_players: int) -> dict:
    """Create an empty per-player stats dictionary."""
    stats = {}
    for pid in range(n_players):
        stats[pid] = {
            "shots": 0,
            "makes": 0,
            "shot_types": {
                "dunk": [0, 0],
                "two": [0, 0],
                "three": [0, 0],
            },
            "assist_full_by_type": {
                "dunk": 0,
                "two": 0,
                "three": 0,
            },
            "assists": 0,
            "potential_assists": 0,
            "turnovers": 0,
            "points": 0.0,
            "episodes": 0,
            "steps": 0,
            "shot_chart": {},
            "unassisted": {
                "dunk": 0,
                "two": 0,
                "three": 0,
            },
        }
    return stats


def _merge_player_stats(dest: dict, src: dict) -> dict:
    """Merge per-player stats structures (summing counts in-place)."""
    if dest is None:
        dest = {}
    for pid_raw, src_stats in (src or {}).items():
        pid = int(pid_raw)
        if pid not in dest:
            dest[pid] = {
                "shots": 0,
                "makes": 0,
                "shot_types": {
                    "dunk": [0, 0],
                    "two": [0, 0],
                    "three": [0, 0],
                },
                "assists": 0,
                "potential_assists": 0,
                "turnovers": 0,
                "points": 0.0,
                "episodes": 0,
                "steps": 0,
                "shot_chart": {},
                "unassisted": {
                    "dunk": 0,
                    "two": 0,
                    "three": 0,
                },
            }
        dst_stats = dest[pid]
        dst_stats["shots"] += int(src_stats.get("shots", 0))
        dst_stats["makes"] += int(src_stats.get("makes", 0))
        dst_stats["assists"] += int(src_stats.get("assists", 0))
        dst_stats["potential_assists"] += int(src_stats.get("potential_assists", 0))
        dst_stats["turnovers"] += int(src_stats.get("turnovers", 0))
        dst_stats["points"] += float(src_stats.get("points", 0.0))
        dst_stats["episodes"] += int(src_stats.get("episodes", 0))
        dst_stats["steps"] += int(src_stats.get("steps", 0))

        # Merge shot types
        for shot_type in ("dunk", "two", "three"):
            src_pair = src_stats.get("shot_types", {}).get(shot_type, [0, 0])
            dst_pair = dst_stats["shot_types"].setdefault(shot_type, [0, 0])
            dst_pair[0] += int(src_pair[0] if isinstance(src_pair, (list, tuple)) else 0)
            dst_pair[1] += int(src_pair[1] if isinstance(src_pair, (list, tuple)) else 0)
            dst_stats.setdefault("assist_full_by_type", {}).setdefault(shot_type, 0)
            dst_stats["assist_full_by_type"][shot_type] += int(src_stats.get("assist_full_by_type", {}).get(shot_type, 0) or 0)

        # Merge shot chart (location -> [att, made])
        src_chart = src_stats.get("shot_chart", {}) or {}
        for loc, vals in src_chart.items():
            dst_pair = dst_stats["shot_chart"].setdefault(loc, [0, 0])
            try:
                att = int(vals[0]) if isinstance(vals, (list, tuple)) and len(vals) > 0 else 0
                mk = int(vals[1]) if isinstance(vals, (list, tuple)) and len(vals) > 1 else 0
            except Exception:
                att, mk = 0, 0
            dst_pair[0] += att
            dst_pair[1] += mk
        # Merge unassisted counts
        src_un = src_stats.get("unassisted", {}) or {}
        dst_un = dst_stats.setdefault("unassisted", {"dunk": 0, "two": 0, "three": 0})
        for key in ("dunk", "two", "three"):
            dst_un[key] = dst_un.get(key, 0) + int(src_un.get(key, 0) or 0)
    return dest


def _record_shot_for_stats(stats: dict, shooter_id: int, env, success: bool, assist_full: bool = False):
    """Update per-player stats for a shot attempt."""
    if stats is None or shooter_id not in stats:
        return
    try:
        pos = env.positions[shooter_id]
        dist = env._hex_distance(pos, env.basket_position)
        is_three = bool(env._is_three_point_hex(tuple(pos)))
    except Exception:
        # Fall back if env helpers are unavailable
        pos = (0, 0)
        dist = None
        is_three = False

    shot_type = "three" if is_three else "two"
    if dist == 0 and getattr(env, "allow_dunks", True):
        shot_type = "dunk"

    player_stats = stats[shooter_id]
    player_stats["shots"] += 1
    player_stats["makes"] += 1 if success else 0
    player_stats["points"] += 3 if (shot_type == "three" and success) else (2 if success else 0)
    pair = player_stats["shot_types"].setdefault(shot_type, [0, 0])
    pair[0] += 1
    if success:
        pair[1] += 1
        # Track assisted counts for later unassisted derivation
        afbt = player_stats.setdefault("assist_full_by_type", {"dunk": 0, "two": 0, "three": 0})
        if assist_full:
            afbt[shot_type] = afbt.get(shot_type, 0) + 1
        if not assist_full:
            # Track unassisted makes per type
            ua = player_stats.setdefault("unassisted", {"dunk": 0, "two": 0, "three": 0})
            ua[shot_type] = ua.get(shot_type, 0) + 1

    # Shot chart aggregation by (q,r)
    try:
        key = f"{int(pos[0])},{int(pos[1])}"
        chart_pair = player_stats["shot_chart"].setdefault(key, [0, 0])
        chart_pair[0] += 1
        if success:
            chart_pair[1] += 1
    except Exception:
        pass


def _record_assist_for_stats(stats: dict, passer_id: Optional[int], full: bool, potential: bool):
    if stats is None or passer_id is None or passer_id not in stats:
        return
    if full:
        stats[passer_id]["assists"] += 1
    if potential:
        stats[passer_id]["potential_assists"] += 1


def _record_turnover_for_stats(stats: dict, player_id: Optional[int]):
    if stats is None or player_id is None:
        return
    pid = int(player_id)
    if pid in stats:
        stats[pid]["turnovers"] += 1


def _build_reset_options_for_custom_setup(custom_setup: dict | None, enforce_fixed_skills: bool = False) -> dict:
    """Translate a validated custom setup dict into env.reset options."""
    if not custom_setup:
        return {}
    opts: dict = {}
    if custom_setup.get("initial_positions"):
        opts["initial_positions"] = [tuple(p) for p in custom_setup["initial_positions"]]
    if custom_setup.get("ball_holder") is not None:
        opts["ball_holder"] = int(custom_setup["ball_holder"])
    shooting_mode = custom_setup.get("shooting_mode") or "random"
    if enforce_fixed_skills and shooting_mode == "fixed" and custom_setup.get("offense_skills"):
        opts["offense_skills"] = copy.deepcopy(custom_setup["offense_skills"])
    return opts


def _sample_offense_permutation(env, rng=None) -> list[int]:
    """Return a permutation that shuffles offense slots while leaving defense slots in place."""
    if env is None or not hasattr(env, "offense_ids"):
        return []
    rng = rng or np.random.default_rng()
    perm = list(range(env.n_players))
    try:
        shuffled = list(rng.permutation(env.offense_ids))
    except Exception:
        shuffled = list(env.offense_ids)
    for dest_idx, src_idx in zip(env.offense_ids, shuffled):
        perm[dest_idx] = int(src_idx)
    return perm


def _invert_permutation(perm: list[int]) -> list[int]:
    inv = [0] * len(perm)
    for dest, src in enumerate(perm):
        inv[src] = dest
    return inv


def _permute_base_obs(base_obs: dict, env, perm: list[int]) -> tuple[dict, list[int]]:
    """Return a copy of base_obs with obs/action_mask (and skills) rows permuted."""
    if not perm or base_obs is None or env is None:
        return base_obs, []
    obs_arr = np.asarray(base_obs.get("obs"))
    mask_arr = np.asarray(base_obs.get("action_mask"))
    if obs_arr.ndim < 1 or mask_arr.ndim < 1:
        return base_obs, []
    inv = _invert_permutation(perm)
    permuted = dict(base_obs)
    try:
        permuted["obs"] = np.take(obs_arr, perm, axis=0)
    except Exception:
        permuted["obs"] = obs_arr
    try:
        permuted["action_mask"] = np.take(mask_arr, perm, axis=0)
    except Exception:
        permuted["action_mask"] = mask_arr

    # Permute offense skills (flat array length players_per_side*3)
    skills = base_obs.get("skills")
    try:
        offense_perm = [perm[idx] for idx in env.offense_ids]
        if skills is not None:
            skills_arr = np.asarray(skills)
            if skills_arr.size == env.players_per_side * 3:
                reshaped = skills_arr.reshape(env.players_per_side, 3)
                permuted_skills = reshaped[offense_perm].reshape(-1)
                permuted["skills"] = permuted_skills
    except Exception:
        permuted["skills"] = skills

    return permuted, inv


def _unpermute_actions(actions: np.ndarray, perm: list[int]) -> np.ndarray:
    """Map actions from permuted order back to original player order."""
    if actions is None or not perm:
        return actions
    actions = np.asarray(actions)
    restored = np.zeros_like(actions)
    for dest_idx, src_idx in enumerate(perm):
        restored[src_idx] = actions[dest_idx]
    return restored


def _apply_offense_permutation_in_env(env, perm: list[int]):
    """Physically permute offense player assignments in the env state (positions, ball holder, skills).

    perm maps dest_idx -> src_idx for ALL players, but only offense_ids are shuffled.
    """
    if not perm or env is None or not hasattr(env, "offense_ids"):
        return
    try:
        positions = list(env.positions)
        new_positions = list(positions)
        for dest_idx, src_idx in enumerate(perm):
            if dest_idx in env.offense_ids:
                new_positions[dest_idx] = positions[src_idx]
        env.positions = new_positions

        # Keep ball holder and skills tied to player IDs; only positions are permuted
    except Exception:
        return


def _worker_predict_actions_for_team(
    policy,
    base_obs: dict,
    env,
    team,
    deterministic: bool,
    strategy,
) -> dict[int, int]:
    """Predict actions for a team in worker context."""
    np = _worker_state["np"]
    Team = _worker_state["Team"]
    get_policy_action_probabilities = _worker_state["get_policy_action_probabilities"]
    resolve_illegal_actions = _worker_state["resolve_illegal_actions"]
    
    actions_by_player: dict[int, int] = {}
    
    if policy is None or base_obs is None or env is None:
        return actions_by_player
    
    team_ids = list(env.offense_ids if team == Team.OFFENSE else env.defense_ids)
    if not team_ids:
        return actions_by_player
    
    role_flag_value = _worker_role_flag_value(team)
    conditioned_obs = _worker_clone_obs_with_role_flag(base_obs, role_flag_value)
    
    try:
        raw_actions, _ = policy.predict(conditioned_obs, deterministic=deterministic)
    except Exception:
        return actions_by_player
    
    raw_actions = np.array(raw_actions).reshape(-1)
    action_len = raw_actions.shape[0]
    team_mask = base_obs["action_mask"][team_ids]
    
    # Legacy policies output actions for every player; new policies output players_per_side only.
    if action_len == len(team_ids):
        team_pred_actions = raw_actions
    elif action_len == getattr(env, "n_players", action_len):
        team_pred_actions = raw_actions[team_ids]
    else:
        team_pred_actions = raw_actions[: len(team_ids)]
    
    # Get probabilities for sampling strategy
    probs = get_policy_action_probabilities(policy, conditioned_obs)
    if probs is not None:
        probs = [np.asarray(p, dtype=np.float32) for p in probs]
        if len(probs) == getattr(env, "n_players", len(probs)):
            team_probs = [probs[int(pid)] for pid in team_ids]
        else:
            team_probs = probs[: len(team_ids)]
    else:
        team_probs = None
    
    resolved_actions = resolve_illegal_actions(
        np.array(team_pred_actions),
        team_mask,
        strategy,
        deterministic,
        team_probs,
    )
    
    for idx, pid in enumerate(team_ids):
        actions_by_player[int(pid)] = int(resolved_actions[idx])
    
    return actions_by_player


def _worker_predict_policy_actions(
    policy,
    base_obs: dict,
    env,
    deterministic: bool,
    strategy,
):
    """Predict actions for all players using a policy in worker context."""
    np = _worker_state["np"]
    Team = _worker_state["Team"]
    
    if policy is None or base_obs is None or env is None:
        return None
    
    num_players = env.n_players
    full_actions = np.zeros(num_players, dtype=int)
    
    for team in (Team.OFFENSE, Team.DEFENSE):
        team_actions = _worker_predict_actions_for_team(
            policy, base_obs, env, team, deterministic, strategy
        )
        for pid, action in team_actions.items():
            full_actions[int(pid)] = int(action)
    
    return full_actions


def _run_episode_batch_worker(args: tuple) -> dict:
    """Run a batch of evaluation episodes in a worker process.
    
    Args:
        args: Tuple of (episode_specs, player_deterministic, opponent_deterministic, custom_setup, randomize_offense_permutation)
              where episode_specs is a list of (episode_index, seed) tuples
        
    Returns:
        Dict with episode results and per-batch shot accumulator
    """
    np = _worker_state["np"]
    Team = _worker_state["Team"]
    IllegalActionStrategy = _worker_state["IllegalActionStrategy"]
    episode_specs, player_deterministic, opponent_deterministic, custom_setup, randomize_offense_permutation = args

    env = _worker_state["env"]
    unified_policy = _worker_state["unified_policy"]
    opponent_policy = _worker_state["opponent_policy"]
    user_team = _worker_state["user_team"]
    
    player_strategy = (
        IllegalActionStrategy.BEST_PROB if player_deterministic
        else IllegalActionStrategy.SAMPLE_PROB
    )
    opponent_strategy = (
        IllegalActionStrategy.BEST_PROB if opponent_deterministic
        else IllegalActionStrategy.SAMPLE_PROB
    )
    
    results = []
    shot_accumulator: dict[str, list[int]] = {}
    per_player_stats = _init_player_stats(env.n_players)
    
    for ep_idx, seed in episode_specs:
        # Reset environment with seed and custom setup (if provided)
        reset_opts = _build_reset_options_for_custom_setup(custom_setup, enforce_fixed_skills=True)
        obs, _ = env.reset(seed=seed, options=reset_opts)
        policy_order = None
        if randomize_offense_permutation:
            try:
                policy_order = _build_offense_policy_order(env, np.random.default_rng(int(seed)))
            except Exception:
                policy_order = None
        policy_env = _build_policy_env_view(env) if policy_order else env
        episode_shots: dict[str, list[int]] = {}
        
        done = False
        step_count = 0
        episode_rewards = {"offense": 0.0, "defense": 0.0}
        # Track participation per player
        for pid in per_player_stats:
            per_player_stats[pid]["episodes"] += 1
        
        # Run episode until done
        while not done and step_count < 1000:
            policy_obs = obs
            if policy_order:
                policy_obs = _build_permuted_policy_obs(
                    env,
                    policy_order,
                    _worker_state["role_flag_offense"],
                    normalize_obs=getattr(env, "normalize_obs", True),
                )

            # Get actions from unified policy (for user team)
            resolved_unified = _worker_predict_policy_actions(
                unified_policy, policy_obs, policy_env, player_deterministic, player_strategy
            )
            if resolved_unified is None:
                resolved_unified = np.zeros(env.n_players, dtype=int)
            elif policy_order:
                resolved_unified = _unpermute_actions_policy_to_env(resolved_unified, policy_order, env.n_players)
            
            # Get actions from opponent policy (or use unified if no separate opponent)
            if opponent_policy is not None:
                resolved_opponent = _worker_predict_policy_actions(
                    opponent_policy, policy_obs, policy_env, opponent_deterministic, opponent_strategy
                )
            else:
                resolved_opponent = np.array(resolved_unified)
            if policy_order:
                resolved_opponent = _unpermute_actions_policy_to_env(resolved_opponent, policy_order, env.n_players)
            
            # Combine actions based on team roles
            final_action = np.zeros(env.n_players, dtype=np.int32)
            
            if user_team == Team.OFFENSE:
                for idx in env.offense_ids:
                    final_action[idx] = resolved_unified[idx]
                for idx in env.defense_ids:
                    final_action[idx] = resolved_opponent[idx]
            else:
                for idx in env.defense_ids:
                    final_action[idx] = resolved_unified[idx]
                for idx in env.offense_ids:
                    final_action[idx] = resolved_opponent[idx]
            
            # Execute step
            obs, reward, terminated, truncated, info = env.step(final_action)
            done = terminated or truncated
            step_count += 1
            
            # Track rewards
            if isinstance(reward, np.ndarray):
                rewards_list = reward.tolist()
            elif isinstance(reward, (list, tuple)):
                rewards_list = list(reward)
            else:
                rewards_list = [reward]
            
            if len(rewards_list) > 1:
                for i, r in enumerate(rewards_list):
                    if i in env.offense_ids:
                        episode_rewards["offense"] += float(r)
                    elif i in env.defense_ids:
                        episode_rewards["defense"] += float(r)
            else:
                episode_rewards["offense"] += float(rewards_list[0])
        
        # Extract outcome information from last_action_results
        last_action_results = getattr(env, "last_action_results", {}) or {}
        shot_clock = int(getattr(env, "shot_clock", 0))
        three_point_distance = float(getattr(env, "three_point_distance", 4.0))
        
        # One shot per episode: accumulate at episode end
        shots_for_episode = last_action_results.get("shots", {}) if isinstance(last_action_results, dict) else {}
        for shooter_id, shot_res in shots_for_episode.items():
            try:
                sid = int(shooter_id)
                pos = env.positions[sid]
                q, r = int(pos[0]), int(pos[1])
                key = f"{q},{r}"
                if key not in shot_accumulator:
                    shot_accumulator[key] = [0, 0]
                if key not in episode_shots:
                    episode_shots[key] = [0, 0]
                shot_accumulator[key][0] += 1
                episode_shots[key][0] += 1
                if bool(shot_res.get("success", False)):
                    shot_accumulator[key][1] += 1
                    episode_shots[key][1] += 1
                _record_shot_for_stats(per_player_stats, sid, env, bool(shot_res.get("success", False)), bool(shot_res.get("assist_full", False)))
                if "assist_passer_id" in shot_res:
                    _record_assist_for_stats(
                        per_player_stats,
                        int(shot_res.get("assist_passer_id")),
                        bool(shot_res.get("assist_full", False)),
                        bool(shot_res.get("assist_potential", False)),
                    )
            except Exception:
                continue

        # Serialize last_action_results to be JSON-safe with full shot details
        shots_info = {}
        if last_action_results.get("shots"):
            for k, v in last_action_results["shots"].items():
                if isinstance(v, dict):
                    shots_info[str(k)] = {
                        "success": bool(v.get("success", False)),
                        "distance": int(v.get("distance", 9999)),
                        "assist_full": bool(v.get("assist_full", False)),
                        "assist_potential": bool(v.get("assist_potential", False)),
                    }
                else:
                    shots_info[str(k)] = {
                        "success": False,
                        "distance": 9999,
                        "assist_full": False,
                        "assist_potential": False,
                    }
        
        # Turnovers - preserve the list structure if it exists
        turnovers_raw = last_action_results.get("turnovers", [])
        turnovers_info = list(turnovers_raw) if isinstance(turnovers_raw, (list, tuple)) else []
        if isinstance(turnovers_raw, (list, tuple)):
            for t in turnovers_raw:
                try:
                    _record_turnover_for_stats(per_player_stats, t.get("player_id"))
                except Exception:
                    continue

        # Per-player step totals
        for pid in per_player_stats:
            per_player_stats[pid]["steps"] += step_count
        
        results.append({
            "episode": ep_idx + 1,
            "steps": step_count,
            "episode_rewards": episode_rewards,
            "outcome_info": {
                "shots": shots_info,
                "turnovers": turnovers_info,
                "shot_clock": shot_clock,
                "three_point_distance": three_point_distance,
            },
            "shot_counts": episode_shots,
        })
    
    return {
        "results": results,
        "shot_accumulator": shot_accumulator,
        "per_player_stats": per_player_stats,
    }


def _run_sequential_evaluation(
    num_episodes: int,
    player_deterministic: bool,
    opponent_deterministic: bool,
    shot_accumulator: dict | None = None,
    per_player_stats: dict | None = None,
    custom_setup: dict | None = None,
    randomize_offense_permutation: bool = False,
) -> dict:
    """Run evaluation episodes sequentially in the main process.
    
    This is faster than parallel for small episode counts because it avoids
    the overhead of spawning worker processes and loading policies.
    Uses the already-loaded game_state.env and policies.
    """
    results = []
    if shot_accumulator is None:
        shot_accumulator = {}
    env = game_state.env
    if (per_player_stats is None or len(per_player_stats) == 0) and env is not None:
        per_player_stats = _init_player_stats(env.n_players)

    player_strategy = (
        IllegalActionStrategy.BEST_PROB if player_deterministic
        else IllegalActionStrategy.SAMPLE_PROB
    )
    opponent_strategy = (
        IllegalActionStrategy.BEST_PROB if opponent_deterministic
        else IllegalActionStrategy.SAMPLE_PROB
    )
    
    for ep_idx in range(num_episodes):
        # Reset environment with random seed
        seed = int(np.random.randint(0, 2**31 - 1))
        reset_opts = _build_reset_options_for_custom_setup(custom_setup, enforce_fixed_skills=True)
        obs, _ = game_state.env.reset(seed=seed, options=reset_opts)
        policy_order = None
        if randomize_offense_permutation:
            try:
                policy_order = _build_offense_policy_order(game_state.env, np.random.default_rng(seed))
            except Exception:
                policy_order = None
        policy_env = _build_policy_env_view(game_state.env) if policy_order else game_state.env
        
        episode_shots: dict[str, list[int]] = {}
        
        done = False
        step_count = 0
        episode_rewards = {"offense": 0.0, "defense": 0.0}
        if per_player_stats is not None:
            for pid in per_player_stats:
                per_player_stats[pid]["episodes"] += 1
        
        # Run episode until done
        while not done and step_count < 1000:
            # Build policy-facing observation (optionally permuted)
            policy_obs = obs
            if policy_order:
                policy_obs = _build_permuted_policy_obs(
                    game_state.env,
                    policy_order,
                    game_state.role_flag_offense,
                    normalize_obs=getattr(game_state.env, "normalize_obs", True),
                )

            # Get actions from unified policy (for user team)
            resolved_unified, _ = _predict_policy_actions(
                game_state.unified_policy,
                policy_obs,
                policy_env,
                deterministic=player_deterministic,
                strategy=player_strategy,
            )
            if resolved_unified is None:
                resolved_unified = np.zeros(game_state.env.n_players, dtype=int)
            elif policy_order:
                resolved_unified = _unpermute_actions_policy_to_env(resolved_unified, policy_order, game_state.env.n_players)
            
            # Get actions from opponent policy (or use unified if no separate opponent)
            if game_state.defense_policy is not None:
                resolved_opponent, _ = _predict_policy_actions(
                    game_state.defense_policy,
                    policy_obs,
                    policy_env,
                    deterministic=opponent_deterministic,
                    strategy=opponent_strategy,
                )
            else:
                resolved_opponent = np.array(resolved_unified)
            if policy_order:
                resolved_opponent = _unpermute_actions_policy_to_env(resolved_opponent, policy_order, game_state.env.n_players)
            
            # Combine actions based on team roles
            final_action = np.zeros(game_state.env.n_players, dtype=np.int32)
            
            if game_state.user_team == Team.OFFENSE:
                for idx in game_state.env.offense_ids:
                    final_action[idx] = resolved_unified[idx]
                for idx in game_state.env.defense_ids:
                    final_action[idx] = resolved_opponent[idx]
            else:
                for idx in game_state.env.defense_ids:
                    final_action[idx] = resolved_unified[idx]
                for idx in game_state.env.offense_ids:
                    final_action[idx] = resolved_opponent[idx]
            
            # Execute step
            obs, reward, terminated, truncated, info = game_state.env.step(final_action)
            done = terminated or truncated
            step_count += 1

            # Track rewards
            if isinstance(reward, np.ndarray):
                rewards_list = reward.tolist()
            elif isinstance(reward, (list, tuple)):
                rewards_list = list(reward)
            else:
                rewards_list = [reward]
            
            if len(rewards_list) > 1:
                for i, r in enumerate(rewards_list):
                    if i in game_state.env.offense_ids:
                        episode_rewards["offense"] += float(r)
                    elif i in game_state.env.defense_ids:
                        episode_rewards["defense"] += float(r)
            else:
                episode_rewards["offense"] += float(rewards_list[0])
        
        # Extract outcome information
        env = game_state.env
        last_action_results = getattr(env, "last_action_results", {}) or {}
        shot_clock = int(getattr(env, "shot_clock", 0))
        three_point_distance = float(getattr(env, "three_point_distance", 4.0))

        # One shot per episode: accumulate at episode end
        shots_for_episode = last_action_results.get("shots", {}) if isinstance(last_action_results, dict) else {}
        for shooter_id, shot_res in shots_for_episode.items():
            try:
                sid = int(shooter_id)
                pos = game_state.env.positions[sid]
                q, r = int(pos[0]), int(pos[1])
                key = f"{q},{r}"
                if shot_accumulator is not None:
                    if key not in shot_accumulator:
                        shot_accumulator[key] = [0, 0]
                if key not in episode_shots:
                    episode_shots[key] = [0, 0]
                if shot_accumulator is not None:
                    shot_accumulator[key][0] += 1
                episode_shots[key][0] += 1
                if bool(shot_res.get("success", False)):
                    if shot_accumulator is not None:
                        shot_accumulator[key][1] += 1
                    episode_shots[key][1] += 1
                _record_shot_for_stats(per_player_stats, sid, env, bool(shot_res.get("success", False)), bool(shot_res.get("assist_full", False)))
                if "assist_passer_id" in shot_res:
                    _record_assist_for_stats(
                        per_player_stats,
                        int(shot_res.get("assist_passer_id")),
                        bool(shot_res.get("assist_full", False)),
                        bool(shot_res.get("assist_potential", False)),
                    )
            except Exception:
                continue
        
        # Serialize shot details
        shots_info = {}
        if last_action_results.get("shots"):
            for k, v in last_action_results["shots"].items():
                if isinstance(v, dict):
                    shots_info[str(k)] = {
                        "success": bool(v.get("success", False)),
                        "distance": int(v.get("distance", 9999)),
                        "assist_full": bool(v.get("assist_full", False)),
                        "assist_potential": bool(v.get("assist_potential", False)),
                    }
                else:
                    shots_info[str(k)] = {
                        "success": False,
                        "distance": 9999,
                        "assist_full": False,
                        "assist_potential": False,
                    }
        
        turnovers_raw = last_action_results.get("turnovers", [])
        turnovers_info = list(turnovers_raw) if isinstance(turnovers_raw, (list, tuple)) else []
        if isinstance(turnovers_raw, (list, tuple)):
            for t in turnovers_raw:
                try:
                    _record_turnover_for_stats(per_player_stats, t.get("player_id"))
                except Exception:
                    continue
        
        # Steps per player
        if per_player_stats is not None:
            for pid in per_player_stats:
                per_player_stats[pid]["steps"] += step_count
        
        results.append({
            "episode": ep_idx + 1,
            "steps": step_count,
            "episode_rewards": episode_rewards,
            "outcome_info": {
                "shots": shots_info,
                "turnovers": turnovers_info,
                "shot_clock": shot_clock,
                "three_point_distance": three_point_distance,
            },
            "shot_counts": episode_shots,
        })
    
    return {
        "results": results,
        "shot_accumulator": shot_accumulator if shot_accumulator is not None else {},
        "per_player_stats": per_player_stats or {},
    }


def _run_parallel_evaluation(
    num_episodes: int,
    player_deterministic: bool,
    opponent_deterministic: bool,
    required_params: dict,
    optional_params: dict,
    unified_policy_path: str,
    opponent_policy_path: str | None,
    user_team_name: str,
    role_flag_offense: float,
    role_flag_defense: float,
    shot_accumulator: dict[str, list[int]] | None = None,
    custom_setup: dict | None = None,
    randomize_offense_permutation: bool = False,
    num_workers: int | None = None,
) -> dict:
    """Run evaluation episodes in parallel using ProcessPoolExecutor with batching.
    
    Each worker processes multiple episodes to minimize IPC overhead.
    
    Args:
        num_episodes: Number of episodes to run
        player_deterministic: Whether to use deterministic actions for player
        opponent_deterministic: Whether to use deterministic actions for opponent
        required_params: Required environment parameters
        optional_params: Optional environment parameters
        unified_policy_path: Path to the unified policy file
        opponent_policy_path: Path to the opponent policy file (or None)
        user_team_name: "OFFENSE" or "DEFENSE"
        role_flag_offense: Role flag value for offense team
        role_flag_defense: Role flag value for defense team
        num_workers: Number of worker processes (default: CPU count)
        
    Returns:
        List of episode results
    """
    from concurrent.futures import ProcessPoolExecutor
    
    if num_workers is None:
        # Use number of physical cores, capped at 16
        num_workers = min(mp.cpu_count(), 16)
    
    # For small number of episodes, use fewer workers
    num_workers = min(num_workers, num_episodes)
    
    # Generate episode specs: (index, seed) for each episode
    episode_specs = [
        (i, int(np.random.randint(0, 2**31 - 1)))
        for i in range(num_episodes)
    ]
    
    # Divide episodes into batches for each worker (reduces IPC overhead)
    # Each worker gets a chunk of episodes to process sequentially
    batch_size = (num_episodes + num_workers - 1) // num_workers  # Ceiling division
    batches = []
    for i in range(0, num_episodes, batch_size):
        batch = episode_specs[i:i + batch_size]
        if batch:
            batches.append((batch, player_deterministic, opponent_deterministic, custom_setup, randomize_offense_permutation))
    
    print(f"[Parallel Evaluation] Using {len(batches)} worker processes for {num_episodes} episodes ({batch_size} episodes/batch)")
    
    # Use spawn context for safety with PyTorch/CUDA
    ctx = mp.get_context("spawn")
    
    with ProcessPoolExecutor(
        max_workers=num_workers,
        mp_context=ctx,
        initializer=_init_evaluation_worker,
        initargs=(
            required_params,
            optional_params,
            unified_policy_path,
            opponent_policy_path,
            user_team_name,
            role_flag_offense,
            role_flag_defense,
        ),
    ) as executor:
        batch_results = list(executor.map(_run_episode_batch_worker, batches))
    
    # Flatten results from all batches and merge shot counts
    results = []
    merged_player_stats: dict = {}
    for payload in batch_results:
        if not payload:
            continue
        if isinstance(payload, dict):
            batch_res = payload.get("results", [])
            results.extend(batch_res)
            if shot_accumulator is not None:
                batch_shots = payload.get("shot_accumulator", {}) or {}
                for key, vals in batch_shots.items():
                    try:
                        att = int(vals[0]) if isinstance(vals, (list, tuple)) and len(vals) > 0 else 0
                        mk = int(vals[1]) if isinstance(vals, (list, tuple)) and len(vals) > 1 else 0
                    except Exception:
                        att, mk = 0, 0
                    if key not in shot_accumulator:
                        shot_accumulator[key] = [0, 0]
                    shot_accumulator[key][0] += att
                    shot_accumulator[key][1] += mk
            merged_player_stats = _merge_player_stats(merged_player_stats, payload.get("per_player_stats") or {})
        elif isinstance(payload, list):
            # Backward compatibility: older worker may return list of results only
            results.extend(payload)
    
    return {
        "results": results,
        "shot_accumulator": shot_accumulator if shot_accumulator is not None else {},
        "per_player_stats": merged_player_stats,
    }


# --- API Models ---
class CustomEvalSetup(BaseModel):
    initial_positions: list[tuple[int, int]] | None = None
    ball_holder: int | None = None
    shooting_mode: Literal["random", "fixed"] = "random"
    offense_skills: dict[str, list[float]] | None = None


class InitGameRequest(BaseModel):
    run_id: str
    user_team_name: str  # "OFFENSE" or "DEFENSE"
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
    actions: dict[
        str, int
    ]  # JSON keys are strings, so we accept strings and convert later.
    player_deterministic: bool | None = None
    opponent_deterministic: bool | None = None
    use_mcts: bool | None = None
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


# --- FastAPI App ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def list_policies_from_run(client, run_id):
    """Return sorted list of unified policy artifact paths for a run."""
    artifacts = client.list_artifacts(run_id, "models")
    unified = [
        f.path for f in artifacts if f.path.endswith(".zip") and "unified" in f.path
    ]

    # sort by number embedded at end _<n>.zip if present
    def sort_key(p):
        import re

        m = re.search(r"_(\d+)\.zip$", p)
        return int(m.group(1)) if m else 0

    unified.sort(key=sort_key)
    return unified


def get_unified_policy_path(client, run_id, policy_name: str | None):
    """Return artifact path for unified policy (downloaded locally). If name None, use latest."""
    # Use a persistent cache directory to avoid deletion before PPO.load
    cache_dir = os.path.join("episodes", "_policy_cache")
    os.makedirs(cache_dir, exist_ok=True)

    unified_paths = list_policies_from_run(client, run_id)
    choices = unified_paths
    if not choices:
        raise HTTPException(
            status_code=404, detail=f"No unified policy artifacts found."
        )

    chosen_artifact = None
    if policy_name and any(p.endswith(policy_name) for p in choices):
        # find full path that ends with provided name
        chosen_artifact = next(p for p in choices if p.endswith(policy_name))
    else:
        chosen_artifact = choices[-1]  # latest

    return client.download_artifacts(run_id, chosen_artifact, cache_dir)


# existing helper kept for internal use
def get_latest_policies_from_run(client, run_id, tmpdir):
    """Downloads the latest policies from a given MLflow run."""
    artifacts = client.list_artifacts(run_id, "models")
    if not artifacts:
        raise HTTPException(
            status_code=404, detail="No model artifacts found in the specified run."
        )

    latest_offense_path = max(
        [f.path for f in artifacts if "offense" in f.path],
        key=lambda p: int(re.search(r"_(\d+)\.zip", p).group(1)),
    )
    latest_defense_path = max(
        [f.path for f in artifacts if "defense" in f.path],
        key=lambda p: int(re.search(r"_(\d+)\.zip", p).group(1)),
    )

    offense_local_path = client.download_artifacts(run_id, latest_offense_path, tmpdir)
    defense_local_path = client.download_artifacts(run_id, latest_defense_path, tmpdir)

    return offense_local_path, defense_local_path


@app.post("/api/list_policies")
def list_policies(request: ListPoliciesRequest):
    """Return available unified policy filenames for a run."""
    try:
        # Set tracking URI once at startup (avoid race conditions from repeated calls)
        current_uri = mlflow.get_tracking_uri()
        if "localhost" not in current_uri:
            mlflow.set_tracking_uri("http://localhost:5000")
        
        client = mlflow.tracking.MlflowClient()
        unified_paths = list_policies_from_run(client, request.run_id)
        
        # Return only basenames to frontend
        if not unified_paths:
            return {"unified": []}
        
        return {
            "unified": [os.path.basename(p) for p in unified_paths],
        }
    except Exception as e:
        # Log the error for debugging
        import traceback
        print(f"Error listing policies: {e}")
        traceback.print_exc()
        # Return empty list instead of 500 error (graceful degradation)
        return {"unified": []}


@app.post("/api/init_game")
async def init_game(request: InitGameRequest):
    """Initializes a new game from an MLflow run.

    We persist the global GameState instance so that shot logs can continue across
    episodes as long as the loaded policies do not change. If policies change, we
    reset the shot log.
    """
    global game_state

    from basketworld.utils.mlflow_config import setup_mlflow

    try:
        setup_mlflow(verbose=False)
    except (ImportError, ValueError) as e:
        raise HTTPException(status_code=500, detail=f"Failed to set up MLflow: {e}")

    client = mlflow.tracking.MlflowClient()

    try:
        required, optional = get_mlflow_params(client, request.run_id)

        # Fetch run metadata
        run = client.get_run(request.run_id)
        run_name = run.data.tags.get("mlflow.runName") if run and run.data else None

        # Load phi shaping parameters from MLflow
        # These will be used for Rewards tab calculations (independent of Phi Shaping tab)
        mlflow_phi_params = get_mlflow_phi_shaping_params(client, request.run_id)
        
        # Load training parameters from MLflow (PPO hyperparameters)
        mlflow_training_params = dict(get_mlflow_training_params(client, request.run_id) or {})
        
        # Load environment parameters including role_flag encoding
        required, optional = get_mlflow_params(client, request.run_id)

        # Extract role_flag encoding for backward compatibility (not passed to env)
        game_state.role_flag_offense = optional.pop("role_flag_offense_value")
        game_state.role_flag_defense = optional.pop("role_flag_defense_value")
        encoding_version = optional.pop("role_flag_encoding_version")
        
        if encoding_version == "symmetric":
            print(f"[INIT] Using SYMMETRIC role_flag encoding: offense={game_state.role_flag_offense}, defense={game_state.role_flag_defense}")
        else:
            print(f"[INIT] Using LEGACY role_flag encoding: offense={game_state.role_flag_offense}, defense={game_state.role_flag_defense}")

        # Apply request overrides for optional parameters
        if request.spawn_distance is not None:
            optional["spawn_distance"] = request.spawn_distance
        if request.defender_spawn_distance is not None:
            optional["defender_spawn_distance"] = request.defender_spawn_distance
        if request.allow_dunks is not None:
            optional["allow_dunks"] = request.allow_dunks
        if request.dunk_pct is not None:
            optional["dunk_pct"] = request.dunk_pct

        # Unified-only
        unified_path = get_unified_policy_path(
            client, request.run_id, request.unified_policy_name
        )
        opponent_unified_path = None
        if request.opponent_unified_policy_name:
            opponent_unified_path = get_unified_policy_path(
                client, request.run_id, request.opponent_unified_policy_name
            )
        unified_key = os.path.basename(unified_path)
        print(f"[INIT] Loading unified policy: {unified_key} ({unified_path})")

        # Reset per-episode stats/logs and set policy keys for UI on every init
        game_state.shot_log = []
        game_state.unified_policy_key = unified_key
        game_state.offense_policy_key = None
        game_state.defense_policy_key = None
        game_state.opponent_unified_policy_key = (
            os.path.basename(opponent_unified_path) if opponent_unified_path else None
        )

        # (Re)load policies from the selected paths
        # Provide custom_objects so SB3 can deserialize custom policy classes
        custom_objects = {
            "policy_class": PassBiasDualCriticPolicy,
            "PassBiasDualCriticPolicy": PassBiasDualCriticPolicy,
            "PassBiasMultiInputPolicy": PassBiasMultiInputPolicy,
        }
        game_state.unified_policy = PPO.load(unified_path, custom_objects=custom_objects)
        game_state.offense_policy = None
        game_state.defense_policy = (
            PPO.load(opponent_unified_path, custom_objects=custom_objects) if opponent_unified_path else None
        )
        if opponent_unified_path:
            print(f"[INIT] Loading opponent policy: {os.path.basename(opponent_unified_path)} ({opponent_unified_path})")
        else:
            print("[INIT] Opponent policy: mirror unified (no separate policy loaded)")

        # Compute parameter counts for UI display
        try:
            counts = _compute_param_counts_from_policy(game_state.unified_policy)
            if counts:
                mlflow_training_params["param_counts"] = counts
        except Exception:
            pass

        game_state.env = basketworld.HexagonBasketballEnv(
            **required,
            **optional,
            render_mode="rgb_array",
        )
        game_state.obs, _ = game_state.env.reset()
        game_state.prev_obs = None  # No previous observation at start
        
        # Store params and paths for parallel evaluation workers
        game_state.env_required_params = copy.deepcopy(required)
        game_state.env_optional_params = copy.deepcopy(optional)
        game_state.unified_policy_path = unified_path
        game_state.opponent_policy_path = opponent_unified_path
        
        # Capture the sampled skills for this episode so they remain consistent across resets
        sampled_skills = {
            "layup": list(game_state.env.offense_layup_pct_by_player),
            "three_pt": list(game_state.env.offense_three_pt_pct_by_player),
            "dunk": list(game_state.env.offense_dunk_pct_by_player),
        }
        game_state.replay_offense_skills = copy.deepcopy(sampled_skills)
        game_state.sampled_offense_skills = copy.deepcopy(sampled_skills)
        
        _capture_turn_start_snapshot()

        # Log shot clock configuration for debugging
        print(
            f"[Environment Config] Shot clock settings (loaded from MLflow run {request.run_id}):"
        )
        print(
            f"  - shot_clock (max): {required.get('shot_clock', 24)} (from MLflow param 'shot_clock')"
        )
        print(
            f"  - min_shot_clock: {optional.get('min_shot_clock', 10)} (from MLflow param 'min_shot_clock')"
        )
        print(f"  - Initial shot clock for this episode: {game_state.env.shot_clock}")
        print(
            f"  - Episode length range: [{game_state.env.min_shot_clock}, {game_state.env.shot_clock_steps}] steps"
        )

        # Set user team and ensure tracking containers start empty for the episode
        game_state.user_team = Team[request.user_team_name.upper()]
        # Store MLflow run metadata on game state for later use (saving, UI)
        game_state.run_id = request.run_id
        game_state.run_name = run_name or request.run_id
        game_state.mlflow_phi_shaping_params = mlflow_phi_params
        game_state.mlflow_training_params = mlflow_training_params
        game_state.frames = []
        game_state.reward_history = []
        game_state.episode_rewards = {"offense": 0.0, "defense": 0.0}
        # Clear any prior replay buffers and phi logs
        game_state.actions_log = []
        game_state.episode_states = []
        game_state.phi_log = []

        # Capture and keep the initial frame so saved episodes include the starting court state
        try:
            frame = game_state.env.render()
            if frame is not None:
                game_state.frames.append(frame)
        except Exception as e:
            print(f"Warning: Failed to capture initial frame: {e}")
            import traceback
            traceback.print_exc()

        # Record initial state for replay (manual or self-play) with policy probs
        initial_state = get_full_game_state(
            include_policy_probs=True,
            include_action_values=True,
            include_state_values=True,
        )
        game_state.episode_states.append(initial_state)

        # Record initial phi values (step 0) by computing EP for all players
        try:
            env = game_state.env
            ep_by_player = []
            for pid in range(env.n_players):
                pos = env.positions[pid]
                dist = env._hex_distance(pos, env.basket_position)
                is_three = env.is_three_point_location(pos)
                if getattr(env, "allow_dunks", True) and dist == 0:
                    shot_value = 2.0
                else:
                    shot_value = 3.0 if is_three else 2.0
                p = float(env._calculate_shot_probability(pid, dist))
                ep = float(shot_value * p)
                ep_by_player.append(ep)

            # Calculate phi_next for initial state using the same logic as in the env
            ball_holder_id = int(env.ball_holder) if env.ball_holder is not None else -1
            offense_ids = list(env.offense_ids)

            # Compute phi based on aggregation mode (same as during steps)
            if ball_holder_id >= 0 and len(offense_ids) > 0:
                ball_ep = ep_by_player[ball_holder_id]
                mode = getattr(env, "phi_aggregation_mode", "team_best")

                if mode == "team_avg":
                    phi_next = sum(ep_by_player[int(pid)] for pid in offense_ids) / max(
                        1, len(offense_ids)
                    )
                else:
                    teammate_eps = [
                        ep_by_player[int(pid)]
                        for pid in offense_ids
                        if int(pid) != ball_holder_id
                    ]
                    if not teammate_eps:
                        teammate_aggregate = ball_ep
                    elif mode == "teammates_best":
                        teammate_aggregate = max(teammate_eps)
                    elif mode == "teammates_avg":
                        teammate_aggregate = sum(teammate_eps) / len(teammate_eps)
                    elif mode == "teammates_worst":
                        teammate_aggregate = min(teammate_eps)
                    elif mode == "team_worst":
                        teammate_aggregate = min(min(teammate_eps), ball_ep)
                    else:  # "team_best"
                        teammate_aggregate = max(max(teammate_eps), ball_ep)

                    # Blend
                    w = float(max(0.0, min(1.0, getattr(env, "phi_blend_weight", 0.0))))
                    phi_next = (1.0 - w) * float(teammate_aggregate) + w * float(
                        ball_ep
                    )

                team_best_ep = max(ep_by_player[int(pid)] for pid in offense_ids)
                ball_handler_ep = ball_ep
            else:
                phi_next = 0.0
                team_best_ep = 0.0
                ball_handler_ep = 0.0

            initial_phi_entry = {
                "step": 0,
                "phi_prev": 0.0,  # No previous state
                "phi_next": float(phi_next),
                "phi_beta": float(getattr(env, "phi_beta", 0.0)),
                "phi_r_shape": 0.0,  # No shaping reward for initial state
                "ball_handler": ball_holder_id,
                "offense_ids": offense_ids,
                "defense_ids": list(env.defense_ids),
                "shot_clock": int(env.shot_clock) if hasattr(env, "shot_clock") else -1,
                "ep_by_player": ep_by_player,
                "team_best_ep": float(team_best_ep),
                "ball_handler_ep": float(ball_handler_ep),
                "is_terminal": False,  # Initial state is never terminal
                # Note: best_ep_player is calculated on the frontend from ep_by_player
            }
            game_state.phi_log.append(initial_phi_entry)
        except Exception as e:
            # If we fail to compute initial phi, just skip it
            print(f"Failed to compute initial phi: {e}")
            pass

        return {"status": "success", "state": initial_state}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize game: {e}")


def calculate_phi_from_ep_data(
    ep_by_player: list[float],
    ball_handler_id: int,
    offense_ids: list[int],
    phi_params: dict,
) -> float:
    """Calculate phi value from EP data using specified parameters.

    This allows us to recalculate phi with different parameters (e.g., MLflow params)
    independently of the environment's current phi settings.
    """
    if not ep_by_player or ball_handler_id < 0 or not offense_ids:
        return 0.0

    ball_ep = (
        ep_by_player[ball_handler_id] if ball_handler_id < len(ep_by_player) else 0.0
    )

    if phi_params.get("phi_use_ball_handler_only", False):
        return ball_ep

    mode = phi_params.get("phi_aggregation_mode", "team_best")
    blend_weight = phi_params.get("phi_blend_weight", 0.0)

    # Get EPs for offensive team
    offense_eps = [ep_by_player[pid] for pid in offense_ids if pid < len(ep_by_player)]

    if not offense_eps:
        return ball_ep

    if mode == "team_avg":
        # Simple average of all players (including ball handler)
        return sum(offense_eps) / len(offense_eps)

    # For other modes, separate ball handler from teammates
    teammate_eps = [
        ep_by_player[pid]
        for pid in offense_ids
        if pid != ball_handler_id and pid < len(ep_by_player)
    ]

    if not teammate_eps:
        # No teammates (1v1 or edge case)
        return ball_ep

    # Aggregate teammate EPs based on mode
    if mode == "teammates_best":
        teammate_aggregate = max(teammate_eps)
    elif mode == "teammates_avg":
        teammate_aggregate = sum(teammate_eps) / len(teammate_eps)
    elif mode == "teammates_worst":
        teammate_aggregate = min(teammate_eps)
    elif mode == "team_worst":
        # Include ball handler in the "worst" calculation
        teammate_aggregate = min(min(teammate_eps), ball_ep)
    else:  # "team_best" (default/legacy behavior)
        # Include ball handler in the "best" calculation
        teammate_aggregate = max(max(teammate_eps), ball_ep)

    # Blend teammate aggregate with ball handler EP
    w = max(0.0, min(1.0, blend_weight))
    return (1.0 - w) * teammate_aggregate + w * ball_ep


def _compute_q_values_for_player(player_id: int, game_state: GameState) -> dict:
    """Helper function to compute Q-values for all actions for a given player."""
    action_values = {}
    
    if not game_state.env or game_state.obs is None:
        return action_values
    
    value_policy = game_state.unified_policy
    gamma = value_policy.gamma
    
    # Get the list of all possible action names from the enum
    possible_actions = [action.name for action in ActionType]
    
    for action_name in possible_actions:
        action_id = ActionType[action_name].value
        
        # --- Simulate one step forward ---
        temp_env = copy.deepcopy(game_state.env)
        
        # Construct the full action array for the simulation
        # Need to use correct policy for each player (main vs opponent)
        sim_action = np.zeros(temp_env.n_players, dtype=int)

        player_strategy = IllegalActionStrategy.BEST_PROB
        full_actions_main, _ = _predict_policy_actions(
            game_state.unified_policy,
            game_state.obs,
            game_state.env,
            deterministic=True,
            strategy=player_strategy,
        )
        if full_actions_main is None:
            full_actions_main = np.zeros(temp_env.n_players, dtype=int)

        opponent_strategy = IllegalActionStrategy.BEST_PROB
        full_actions_opponent, _ = _predict_policy_actions(
            game_state.defense_policy,
            game_state.obs,
            game_state.env,
            deterministic=True,
            strategy=opponent_strategy,
        )
        
        # Determine which team the evaluating player is on
        is_player_on_user_team = (
            (player_id in game_state.env.offense_ids and game_state.user_team == Team.OFFENSE) or
            (player_id in game_state.env.defense_ids and game_state.user_team == Team.DEFENSE)
        )
        
        for i in range(temp_env.n_players):
            if i == player_id:
                # The player we're evaluating takes the specific action
                sim_action[i] = action_id
            else:
                # Other players: use appropriate policy based on their team
                is_i_on_user_team = (
                    (i in game_state.env.offense_ids and game_state.user_team == Team.OFFENSE) or
                    (i in game_state.env.defense_ids and game_state.user_team == Team.DEFENSE)
                )
                
                if is_i_on_user_team:
                    sim_action[i] = full_actions_main[i]
                elif full_actions_opponent is not None:
                    sim_action[i] = full_actions_opponent[i]
                else:
                    sim_action[i] = full_actions_main[i]
        
        # Step the temporary environment
        next_obs, reward, _, _, _ = temp_env.step(sim_action)
        
        # Get the value of the resulting state with proper role_flag conditioning
        # Determine which role_flag to use based on which team the player is on
        is_offense = player_id in game_state.env.offense_ids
        role_flag_value = game_state.role_flag_offense if is_offense else game_state.role_flag_defense
        
        # Create role-conditioned observation for value prediction
        conditioned_next_obs = {
            "obs": np.copy(next_obs["obs"]),
            "action_mask": next_obs["action_mask"],
            "role_flag": np.array([role_flag_value], dtype=np.float32),
            "skills": np.copy(next_obs.get("skills")) if next_obs.get("skills") is not None else None,
        }
        
        next_obs_tensor, _ = value_policy.policy.obs_to_tensor(conditioned_next_obs)
        with torch.no_grad():
            next_value = value_policy.policy.predict_values(next_obs_tensor)
        
        # Calculate the Q-value
        team_reward = reward[player_id]
        q_value = team_reward + gamma * next_value.item()
        
        # Debug: log if Q-value seems anomalous
        if abs(q_value) > 2.5:
            print(f"[Q-VALUE WARNING] Player {player_id} action {action_name}: Q={q_value:.3f}, r={team_reward:.3f}, V(s')={next_value.item():.3f}, role_flag={role_flag_value}, gamma={gamma}")
        
        action_values[action_name] = q_value
    
    return action_values


def _build_role_conditioned_obs(base_obs: dict | None, role_flag_value: float):
    """Prepare an observation payload with a specific role flag for value prediction."""
    if base_obs is None or "obs" not in base_obs or "action_mask" not in base_obs:
        return None
    return {
        "obs": np.copy(base_obs["obs"]),
        "action_mask": base_obs["action_mask"],
        "role_flag": np.array([role_flag_value], dtype=np.float32),
        "skills": np.copy(base_obs.get("skills")) if base_obs.get("skills") is not None else None,
    }


def _compute_state_values_from_obs(obs_dict: dict | None):
    """Compute offensive/defensive value estimates for a given observation snapshot."""
    if not game_state.unified_policy or obs_dict is None:
        return None
    
    value_policy = game_state.unified_policy
    if not hasattr(value_policy, "policy"):
        return None
    
    try:
        offense_obs = _build_role_conditioned_obs(obs_dict, game_state.role_flag_offense)
        defense_obs = _build_role_conditioned_obs(obs_dict, game_state.role_flag_defense)
        if offense_obs is None or defense_obs is None:
            return None
        
        offense_tensor, _ = value_policy.policy.obs_to_tensor(offense_obs)
        defense_tensor, _ = value_policy.policy.obs_to_tensor(defense_obs)
        
        with torch.no_grad():
            offense_value = float(value_policy.policy.predict_values(offense_tensor).item())
            defense_value = float(value_policy.policy.predict_values(defense_tensor).item())
        
        return {
            "offensive_value": offense_value,
            "defensive_value": defense_value,
        }
    except Exception as err:
        print(f"[STATE_VALUES] Failed to compute state values: {err}")
        import traceback
        traceback.print_exc()
        return None


def _compute_policy_probabilities_for_obs(base_obs: dict, env) -> dict | None:
    """Compute policy probabilities for a provided observation snapshot."""
    if not game_state.unified_policy or base_obs is None or env is None:
        return None
    try:
        # User team uses unified policy; opponent may use defense_policy if present
        _, raw_probs_main = _predict_policy_actions(
            game_state.unified_policy,
            base_obs,
            env,
            deterministic=False,
            strategy=IllegalActionStrategy.SAMPLE_PROB,
        )

        raw_probs_opponent = None
        if game_state.defense_policy is not None:
            _, raw_probs_opponent = _predict_policy_actions(
                game_state.defense_policy,
                base_obs,
                env,
                deterministic=False,
                strategy=IllegalActionStrategy.SAMPLE_PROB,
            )

        if raw_probs_main is None:
            return None

        action_mask = base_obs["action_mask"]
        probs_list = []

        for pid in range(env.n_players):
            is_user_team = (
                (pid in env.offense_ids and game_state.user_team == Team.OFFENSE)
                or (pid in env.defense_ids and game_state.user_team == Team.DEFENSE)
            )
            if is_user_team or raw_probs_opponent is None:
                probs = raw_probs_main[pid]
            else:
                probs = raw_probs_opponent[pid]
            masked = probs * action_mask[pid]
            total = masked.sum()
            if total > 0:
                masked = masked / total
            probs_list.append(masked.tolist())

        return {player_id: probs for player_id, probs in enumerate(probs_list)}
    except Exception as err:
        print(f"[policy_prob_preview] Failed to compute policy probabilities: {err}")
        return None


# --- MCTS Advisor ---
class _MCTSNode:
    def __init__(self, legal_actions: list[int], priors: Optional[np.ndarray]):
        self.legal_actions = legal_actions
        self.priors = priors if priors is not None else None
        self.children: dict[int, dict] = {}  # action -> {"visits", "value_sum", "state_key"}
        self.visits = 0
        self.value_sum = 0.0

    def add_value(self, value: float):
        self.visits += 1
        self.value_sum += float(value)


class MCTSAdvisor:
    def __init__(
        self,
        unified_policy: PPO,
        opponent_policy: Optional[PPO],
        user_team: Team,
        role_flag_offense: float,
        role_flag_defense: float,
        target_player_id: int,
        max_depth: int = 3,
        time_budget_ms: int = 200,
        exploration_c: float = 1.4,
        use_priors: bool = True,
    ):
        self.unified_policy = unified_policy
        self.opponent_policy = opponent_policy
        self.user_team = user_team
        self.role_flag_offense = role_flag_offense
        self.role_flag_defense = role_flag_defense
        self.target_player_id = target_player_id
        self.max_depth = max(1, int(max_depth))
        self.time_budget_ms = max(1, int(time_budget_ms))
        self.exploration_c = float(exploration_c)
        self.use_priors = bool(use_priors)
        self.gamma = float(getattr(unified_policy, "gamma", 0.99))

        self.nodes: dict[str, _MCTSNode] = {}
        self.nodes_expanded = 0
        self.max_depth_reached = 0
        self.cache_hits = 0

    # --- Hashing helpers (partial state) ---
    def _hash_state(self, env, obs: dict) -> str:
        positions_raw = getattr(env, "positions", None) or []
        positions = tuple((int(p[0]), int(p[1])) for p in positions_raw)
        bh_val = getattr(env, "ball_holder", -1)
        ball_holder = -1 if bh_val is None else int(bh_val)
        sc_val = getattr(env, "shot_clock", 0)
        shot_clock = 0 if sc_val is None else int(sc_val)
        offense_lane_dict = getattr(env, "_offensive_lane_steps", None) or {}
        offense_lane = tuple(
            sorted((int(k), int(v)) for k, v in offense_lane_dict.items())
        )
        defense_lane_dict = getattr(env, "_defender_in_key_steps", None) or {}
        defense_lane = tuple(
            sorted((int(k), int(v)) for k, v in defense_lane_dict.items())
        )
        skills = None
        try:
            skill_arr = obs.get("skills") if isinstance(obs, dict) else None
            if skill_arr is not None:
                skills = tuple(float(x) for x in np.asarray(skill_arr).flatten())
        except Exception:
            skills = None

        key_tuple = (positions, ball_holder, shot_clock, offense_lane, defense_lane, skills)
        return hashlib.sha1(str(key_tuple).encode("utf-8")).hexdigest()

    def _clone_env(self, env):
        try:
            return copy.deepcopy(env)
        except Exception as err:
            print(f"[MCTS] Failed to clone env with deepcopy: {err}")
            raise

    # --- Policy helpers ---
    def _team_for_player(self, env, player_id: int) -> Team:
        if player_id in getattr(env, "offense_ids", []):
            return Team.OFFENSE
        return Team.DEFENSE

    def _policy_for_player(self, player_id: int):
        player_team = self._team_for_player(game_state.env, player_id)
        if player_team == self.user_team:
            return self.unified_policy
        return self.opponent_policy or self.unified_policy

    def _build_policy_priors(self, obs: dict, env, player_id: int) -> Optional[np.ndarray]:
        policy = self._policy_for_player(player_id)
        if policy is None or not self.use_priors:
            return None
        try:
            _, probs = _predict_policy_actions(
                policy,
                obs,
                env,
                deterministic=True,
                strategy=IllegalActionStrategy.BEST_PROB,
            )
            if probs is None or player_id >= len(probs):
                return None
            return np.asarray(probs[player_id], dtype=np.float32)
        except Exception as err:
            print(f"[MCTS] Failed to build priors: {err}")
            return None

    def _estimate_value(self, obs: dict, env) -> float:
        policy = self.unified_policy
        if policy is None or not hasattr(policy, "policy"):
            return 0.0
        try:
            role_flag = (
                self.role_flag_offense
                if self.target_player_id in getattr(env, "offense_ids", [])
                else self.role_flag_defense
            )
            conditioned = _build_role_conditioned_obs(obs, role_flag)
            if conditioned is None:
                return 0.0
            obs_tensor, _ = policy.policy.obs_to_tensor(conditioned)
            with torch.no_grad():
                return float(policy.policy.predict_values(obs_tensor).item())
        except Exception as err:
            print(f"[MCTS] Value estimation failed: {err}")
            return 0.0

    def _reward_for_player(self, reward, player_id: int) -> float:
        if isinstance(reward, np.ndarray):
            reward_list = reward.tolist()
        elif isinstance(reward, (list, tuple)):
            reward_list = list(reward)
        else:
            reward_list = [reward]
        if player_id < len(reward_list):
            return float(reward_list[player_id])
        return float(reward_list[0]) if reward_list else 0.0

    def _build_action_array(self, obs: dict, env, target_action: int) -> np.ndarray:
        # Start with policy-driven actions for all players
        resolved_unified, _ = _predict_policy_actions(
            self.unified_policy,
            obs,
            env,
            deterministic=True,
            strategy=IllegalActionStrategy.BEST_PROB,
        )
        if resolved_unified is None:
            resolved_unified = np.zeros(env.n_players, dtype=int)

        resolved_opponent = None
        if self.opponent_policy is not None:
            resolved_opponent, _ = _predict_policy_actions(
                self.opponent_policy,
                obs,
                env,
                deterministic=True,
                strategy=IllegalActionStrategy.BEST_PROB,
            )

        final_action = np.zeros(env.n_players, dtype=np.int32)
        if self.user_team == Team.OFFENSE:
            for idx in env.offense_ids:
                final_action[idx] = resolved_unified[idx]
            for idx in env.defense_ids:
                final_action[idx] = (
                    resolved_opponent[idx] if resolved_opponent is not None else resolved_unified[idx]
                )
        else:
            for idx in env.defense_ids:
                final_action[idx] = resolved_unified[idx]
            for idx in env.offense_ids:
                final_action[idx] = (
                    resolved_opponent[idx] if resolved_opponent is not None else resolved_unified[idx]
                )

        # Override the target player's action with the MCTS choice (validated earlier)
        if 0 <= self.target_player_id < env.n_players:
            final_action[self.target_player_id] = int(target_action)

        return final_action

    def _select_action(self, node: _MCTSNode) -> int:
        # UCB1 with optional priors as bias
        best_action = node.legal_actions[0]
        best_score = float("-inf")
        log_parent = math.log(max(1, node.visits)) if node.visits > 0 else 0.0
        for action in node.legal_actions:
            child_stats = node.children.get(action)
            visits = child_stats["visits"] if child_stats else 0
            value_sum = child_stats["value_sum"] if child_stats else 0.0
            prior_bonus = 0.0
            if node.priors is not None and action < len(node.priors):
                prior_bonus = float(node.priors[action])

            if visits == 0:
                score = float("inf")
            else:
                q = value_sum / visits
                score = q + self.exploration_c * math.sqrt(log_parent / visits) + prior_bonus

            if score > best_score:
                best_score = score
                best_action = action
        return best_action

    def _simulate(self, env, obs: dict, depth: int) -> float:
        state_key = self._hash_state(env, obs)
        node = self.nodes.get(state_key)
        if node is None:
            legal_mask = obs.get("action_mask") if isinstance(obs, dict) else None
            if legal_mask is None or self.target_player_id >= len(legal_mask):
                return 0.0
            player_mask = legal_mask[self.target_player_id]
            legal_actions = [int(i) for i, allowed in enumerate(player_mask) if allowed == 1]
            priors = self._build_policy_priors(obs, env, self.target_player_id)
            node = _MCTSNode(legal_actions, priors)
            self.nodes[state_key] = node
            self.nodes_expanded += 1
            leaf_value = self._estimate_value(obs, env)
            node.add_value(leaf_value)
            self.max_depth_reached = max(self.max_depth_reached, depth)
            return leaf_value

        self.cache_hits += 1

        if depth >= self.max_depth or not node.legal_actions:
            value = self._estimate_value(obs, env)
            node.add_value(value)
            self.max_depth_reached = max(self.max_depth_reached, depth)
            return value

        action = self._select_action(node)
        next_env = self._clone_env(env)
        full_action = self._build_action_array(obs, next_env, action)
        next_obs, reward, terminated, truncated, _ = next_env.step(full_action)
        done = terminated or truncated
        immediate_reward = self._reward_for_player(reward, self.target_player_id)

        if done or depth + 1 >= self.max_depth:
            rollout_value = 0.0 if done else self._estimate_value(next_obs, next_env)
        else:
            rollout_value = self._simulate(next_env, next_obs, depth + 1)

        total_value = immediate_reward + (self.gamma * rollout_value)

        child_stats = node.children.get(action)
        if child_stats is None:
            child_stats = {"visits": 0, "value_sum": 0.0, "state_key": None}
            node.children[action] = child_stats
        child_stats["visits"] += 1
        child_stats["value_sum"] += float(total_value)
        child_stats["state_key"] = self._hash_state(next_env, next_obs)

        node.add_value(total_value)
        self.max_depth_reached = max(self.max_depth_reached, depth)
        return total_value

    def advise(self, env, obs: dict) -> dict:
        if env is None or obs is None:
            raise HTTPException(status_code=400, detail="Game not initialized")

        root_key = self._hash_state(env, obs)
        start = time.perf_counter()
        iterations = 0

        while (time.perf_counter() - start) * 1000.0 < self.time_budget_ms:
            sim_env = self._clone_env(env)
            sim_obs = copy.deepcopy(obs)
            self._simulate(sim_env, sim_obs, depth=0)
            iterations += 1

        root_node = self.nodes.get(root_key)
        if root_node is None or not root_node.legal_actions:
            raise HTTPException(status_code=500, detail="MCTS failed to build root node")

        action_visits = {}
        for act in root_node.legal_actions:
            child = root_node.children.get(act)
            action_visits[act] = child["visits"] if child else 0

        best_action = max(action_visits.items(), key=lambda kv: kv[1])[0]
        total_child_visits = sum(action_visits.values())
        if total_child_visits > 0:
            policy = [action_visits.get(i, 0) / total_child_visits for i in range(len(ActionType))]
        else:
            policy = [0.0 for _ in range(len(ActionType))]

        best_stats = root_node.children.get(best_action)
        q_estimate = None
        if best_stats and best_stats["visits"] > 0:
            q_estimate = best_stats["value_sum"] / best_stats["visits"]

        duration_ms = (time.perf_counter() - start) * 1000.0
        return {
            "action": int(best_action),
            "policy": policy,
            "q_estimate": q_estimate,
            "visits": action_visits,
            "nodes_expanded": self.nodes_expanded,
            "iterations": iterations,
            "duration_ms": duration_ms,
            "nodes_per_sec": (iterations / (duration_ms / 1000.0)) if duration_ms > 0 else None,
            "max_depth_reached": self.max_depth_reached,
            "cache_hits": self.cache_hits,
        }


def _run_mcts_advisor(
    player_id: Optional[int],
    obs: dict,
    env,
    max_depth: Optional[int] = None,
    time_budget_ms: Optional[int] = None,
    exploration_c: Optional[float] = None,
    use_priors: Optional[bool] = True,
):
    if env is None:
        raise HTTPException(status_code=400, detail="Game not initialized")

    target_player = player_id if player_id is not None else getattr(env, "ball_holder", 0)
    if target_player is None:
        target_player = 0

    advisor = MCTSAdvisor(
        unified_policy=game_state.unified_policy,
        opponent_policy=game_state.defense_policy,
        user_team=game_state.user_team,
        role_flag_offense=game_state.role_flag_offense,
        role_flag_defense=game_state.role_flag_defense,
        target_player_id=int(target_player),
        max_depth=max_depth or 3,
        time_budget_ms=time_budget_ms or 200,
        exploration_c=exploration_c if exploration_c is not None else 1.4,
        use_priors=True if use_priors is None else bool(use_priors),
    )
    return advisor.advise(env, obs)


@app.post("/api/step")
def take_step(request: ActionRequest):
    """Takes a single step in the environment."""
    if not game_state.env:
        raise HTTPException(status_code=400, detail="Game not initialized.")

    # Get AI actions (unified-only)
    ai_obs = game_state.obs
    mcts_results: dict[str, dict] = {}
    # Default to deterministic=True if not provided to preserve previous behavior
    player_deterministic = (
        True if request.player_deterministic is None else bool(request.player_deterministic)
    )
    opponent_deterministic = (
        True if request.opponent_deterministic is None else bool(request.opponent_deterministic)
    )

    player_team_strategy = (
        IllegalActionStrategy.BEST_PROB
        if player_deterministic
        else IllegalActionStrategy.SAMPLE_PROB
    )
    opponent_team_strategy = (
        IllegalActionStrategy.BEST_PROB
        if opponent_deterministic
        else IllegalActionStrategy.SAMPLE_PROB
    )

    resolved_unified, unified_probs = _predict_policy_actions(
        game_state.unified_policy,
        ai_obs,
        game_state.env,
        deterministic=player_deterministic,
        strategy=player_team_strategy,
    )
    if resolved_unified is None:
        resolved_unified = np.zeros(game_state.env.n_players, dtype=int)
        unified_probs = [np.zeros(len(ActionType), dtype=np.float32) for _ in range(game_state.env.n_players)]

    resolved_opponent = None
    opponent_probs = None
    if game_state.defense_policy is not None:
        resolved_opponent, opponent_probs = _predict_policy_actions(
            game_state.defense_policy,
            ai_obs,
            game_state.env,
            deterministic=opponent_deterministic,
            strategy=opponent_team_strategy,
        )

    if request.use_mcts:
        try:
            target_pids: list[int] = []
            if request.mcts_player_ids:
                target_pids = [int(pid) for pid in request.mcts_player_ids if pid is not None]
            elif request.mcts_player_id is not None:
                target_pids = [int(request.mcts_player_id)]
            else:
                default_pid = getattr(game_state.env, "ball_holder", None)
                target_pids = [int(default_pid) if default_pid is not None else 0]

            if not target_pids:
                target_pids = [0]

            unique_targets = []
            for pid in target_pids:
                if pid not in unique_targets:
                    unique_targets.append(pid)

            for target_pid in unique_targets:
                result = _run_mcts_advisor(
                    player_id=target_pid,
                    obs=ai_obs,
                    env=game_state.env,
                    max_depth=request.mcts_max_depth,
                    time_budget_ms=request.mcts_time_budget_ms,
                    exploration_c=request.mcts_exploration_c,
                    use_priors=request.mcts_use_priors,
                )
                mcts_results[str(target_pid)] = result
                best_action = result.get("action") if isinstance(result, dict) else None
                if best_action is not None:
                    # Ensure arrays exist before assignment
                    if resolved_unified is None:
                        resolved_unified = np.zeros(game_state.env.n_players, dtype=int)
                    if target_pid in getattr(game_state.env, "offense_ids", []):
                        resolved_unified[target_pid] = int(best_action)
                    elif resolved_opponent is not None:
                        resolved_opponent[target_pid] = int(best_action)
                    else:
                        resolved_unified[target_pid] = int(best_action)
        except Exception as err:
            print(f"[MCTS] Failed to run advisor inside step: {err}")

    action_mask = ai_obs["action_mask"]

    # Combine user and AI actions
    full_action = np.zeros(game_state.env.n_players, dtype=int)

    for i in range(game_state.env.n_players):
        # Check if action is explicitly provided in the request
        # We treat the request as the source of truth if an action is present.
        # This supports manual overrides for ANY player (user or opponent).
        proposed = request.actions.get(str(i))

        if proposed is not None:
             # Validate action against mask (security/rule check)
            if action_mask[i][proposed] == 1:
                full_action[i] = proposed
            else:
                # Illegal action defaults to NOOP (0)
                full_action[i] = 0
        else:
            # If not in request, fallback to AI logic
            # This branch is hit if frontend doesn't send moves for everyone (e.g. older client)
            # or if AI Mode logic on frontend omitted some players.
            is_user_offense = game_state.user_team == Team.OFFENSE
            use_opponent = (
                (is_user_offense and i in game_state.env.defense_ids)
                or ((not is_user_offense) and i in game_state.env.offense_ids)
            ) and (resolved_opponent is not None)

            if use_opponent:
                full_action[i] = int(resolved_opponent[i])
            else:
                full_action[i] = int(resolved_unified[i])

    # Calculate state values BEFORE the step using Q-values: V(s) = Σ π(a|s) * Q(s,a)
    # This is more reliable than calling the critic directly
    pre_step_offensive_value = None
    pre_step_defensive_value = None
    try:
        # Get Q-values and policy probs for offensive team's active player (ball handler or representative)
        offense_player_ids = list(game_state.env.offense_ids)
        defense_player_ids = list(game_state.env.defense_ids)
        
        if offense_player_ids and unified_probs is not None:
            # Use ball handler if available, otherwise first offensive player
            offense_rep = game_state.env.ball_holder if game_state.env.ball_holder in offense_player_ids else offense_player_ids[0]
            offense_q_values = _compute_q_values_for_player(offense_rep, game_state)
            offense_probs = unified_probs
            # V(s) = Σ π(a|s) * Q(s,a)
            pre_step_offensive_value = sum(
                offense_probs[offense_rep][i] * offense_q_values.get(ActionType(i).name, 0.0)
                for i in range(len(offense_probs[offense_rep]))
            )
        
        if defense_player_ids and unified_probs is not None:
            # Use first defensive player as representative
            defense_rep = defense_player_ids[0]
            defense_q_values = _compute_q_values_for_player(defense_rep, game_state)
            defense_probs = unified_probs
            # V(s) = Σ π(a|s) * Q(s,a)
            pre_step_defensive_value = sum(
                defense_probs[defense_rep][i] * defense_q_values.get(ActionType(i).name, 0.0)
                for i in range(len(defense_probs[defense_rep]))
            )
        
        print(f"[STATE_VALUES] Offensive V(s) = {pre_step_offensive_value:.3f} (from Q-values)")
        print(f"[STATE_VALUES] Defensive V(s) = {pre_step_defensive_value:.3f} (from Q-values)")
    except Exception as e:
        print(f"[WARNING] Failed to calculate pre-step state values: {e}")
        import traceback
        traceback.print_exc()
    
    # Cache the current observation before stepping (for backward compat with /api/state_values)
    game_state.prev_obs = game_state.obs
    
    game_state.obs, rewards, done, _, info = game_state.env.step(full_action)

    # Record the full action array for replay (both manual and self-play)
    try:
        game_state.actions_log.append([int(a) for a in full_action.tolist()])
    except Exception:
        # Ensure we still append something even if not a numpy array
        game_state.actions_log.append([int(a) for a in list(full_action)])

    # Track rewards
    # Convert rewards to a list if it's a numpy array
    if isinstance(rewards, np.ndarray):
        rewards_list = rewards.tolist()
    elif isinstance(rewards, (list, tuple)):
        rewards_list = list(rewards)
    else:
        # Single scalar value
        rewards_list = [rewards]

    if len(rewards_list) > 1:
        step_rewards = {"offense": 0.0, "defense": 0.0}
        # Sum rewards by team using actual team assignments
        for i, reward in enumerate(rewards_list):
            if i in game_state.env.offense_ids:
                team = "offense"
            elif i in game_state.env.defense_ids:
                team = "defense"
            else:
                continue  # Skip if player not in either team
            step_rewards[team] += float(reward)
            game_state.episode_rewards[team] += float(reward)
    else:
        # Single agent case
        step_rewards = {"offense": float(rewards_list[0]), "defense": 0.0}
        game_state.episode_rewards["offense"] += float(rewards_list[0])

    # Determine reward types based on actual action results from environment
    offense_reasons = []
    defense_reasons = []

    # Get action results from the environment info
    action_results = info.get("action_results", {}) if info else {}

    # Log any shots with their computed probability for auditability (after action_results is defined)
    if action_results.get("shots"):
        for pid, shot_res in action_results["shots"].items():
            try:
                pid_int = int(pid)
            except Exception:
                pid_int = pid
            # Use the shot distance at the time of the attempt for labeling
            dist_at_shot = int(shot_res.get("distance", 0))
            is_three = bool(
                shot_res.get("is_three")
                if "is_three" in shot_res
                else dist_at_shot >= game_state.env.three_point_distance
            )
            game_state.shot_log.append(
                {
                    "step": int(len(game_state.reward_history) + 1),
                    "player_id": int(pid_int),
                    "distance": dist_at_shot,
                    "probability": float(shot_res.get("probability", 0.0)),
                    "success": bool(shot_res.get("success", False)),
                    "is_three": is_three,
                    "rng": float(shot_res.get("rng", -1.0)),
                    "base_probability": float(shot_res.get("base_probability", -1.0)),
                    "pressure_multiplier": float(
                        shot_res.get("pressure_multiplier", -1.0)
                    ),
                    "expected_points": float(shot_res.get("expected_points", 0.0)),
                    # Adjusted FG% at time of shot (pressure applied and clamped)
                    "shooter_fg_pct": float(shot_res.get("probability", 0.0)),
                }
            )

    # Debug logging for action results
    if action_results:
        print(f"[Action Results] Step {len(game_state.reward_history) + 1}:")
        for key, value in action_results.items():
            if value:  # Only log non-empty results
                print(f"  {key}: {value}")

    # Check for specific actions that occurred
    # 1. Check for shots
    if action_results.get("shots"):
        for player_id, shot_result in action_results["shots"].items():
            ep_val = float(shot_result.get("expected_points", 0.0))
            ep_str = f"{ep_val:.2f}"
            if shot_result.get("success"):
                offense_reasons.append(f"Shot Make (EP={ep_str})")
                defense_reasons.append(f"Opp Shot (EP={ep_str})")
            else:
                offense_reasons.append(f"Shot Miss (EP={ep_str})")
                defense_reasons.append(f"Opp Shot (EP={ep_str})")

    # 2. Check for passes
    if action_results.get("passes"):
        successful_passes = 0
        for player_id, pass_result in action_results["passes"].items():
            if pass_result.get("success"):
                successful_passes += 1

        if successful_passes > 0:
            if successful_passes == 1:
                offense_reasons.append("Pass")
                defense_reasons.append("Opp Pass")
            else:
                offense_reasons.append(f"{successful_passes} Passes")
                defense_reasons.append(f"Opp {successful_passes} Passes")

    # 3. Check for turnovers
    if action_results.get("turnovers"):
        for turnover_info in action_results["turnovers"]:
            reason = turnover_info.get("reason", "Unknown")
            if reason == "out_of_bounds":
                offense_reasons.append("TO - OOB")
                defense_reasons.append("Forced TO - OOB")
            elif reason == "pressure":
                offense_reasons.append("TO - Pressure")
                defense_reasons.append("Forced TO - Pressure")
            elif reason == "steal":
                offense_reasons.append("TO - Steal")
                defense_reasons.append("Forced TO - Steal")
            else:
                offense_reasons.append(f"TO - {reason}")
                defense_reasons.append(f"Forced TO - {reason}")

    # 4. Check for movement penalties (OOB moves)
    if action_results.get("moves"):
        for player_id, move_result in action_results["moves"].items():
            if (
                not move_result.get("success", True)
                and move_result.get("reason") == "out_of_bounds"
            ):
                if player_id in game_state.env.offense_ids:
                    offense_reasons.append("OOB Move")
                    defense_reasons.append("Opp OOB")
                else:
                    defense_reasons.append("OOB Move")
                    offense_reasons.append("Opp OOB")

    # 5. If no specific actions detected but there are rewards, use generic labels
    off_reward = step_rewards["offense"]
    def_reward = step_rewards["defense"]

    if not offense_reasons and not defense_reasons:
        if abs(off_reward) < 0.001 and abs(def_reward) < 0.001:
            offense_reasons.append("None")
            defense_reasons.append("None")
        else:
            if off_reward > 0:
                offense_reasons.append("Positive")
            elif off_reward < 0:
                offense_reasons.append("Negative")
            else:
                offense_reasons.append("None")

            if def_reward > 0:
                defense_reasons.append("Positive")
            elif def_reward < 0:
                defense_reasons.append("Negative")
            else:
                defense_reasons.append("None")

    # Get phi shaping reward for this step (per team member)
    # This is from the environment's phi shaping (using env params)
    phi_r_shape_per_team = float(info.get("phi_r_shape", 0.0)) if info else 0.0

    # Store EP data for MLflow-based phi calculation in Rewards tab
    # Calculate it directly from environment state if not provided
    ep_by_player = []
    if info and "phi_ep_by_player" in info:
        try:
            ep_by_player = [float(x) for x in info.get("phi_ep_by_player", [])]
        except Exception:
            pass

    # If EP data not provided (phi shaping disabled in env), calculate it ourselves
    if not ep_by_player and game_state.env:
        try:
            env = game_state.env
            for pid in range(env.n_players):
                pos = env.positions[pid]
                dist = env._hex_distance(pos, env.basket_position)
                is_three = env.is_three_point_location(pos)
                if getattr(env, "allow_dunks", False) and dist == 0:
                    shot_value = 2.0
                else:
                    shot_value = 3.0 if is_three else 2.0
                p = float(env._calculate_shot_probability(pid, dist))
                ep_by_player.append(float(shot_value * p))
        except Exception as e:
            # If calculation fails, log and continue with empty list
            print(f"[WARNING] Failed to calculate EP data: {e}")
            ep_by_player = []

    game_state.reward_history.append(
        {
            "step": len(game_state.reward_history) + 1,
            "offense": float(step_rewards["offense"]),
            "defense": float(step_rewards["defense"]),
            "offense_reason": ", ".join(offense_reasons) if offense_reasons else "None",
            "defense_reason": ", ".join(defense_reasons) if defense_reasons else "None",
            "phi_r_shape": phi_r_shape_per_team,  # From env (for debugging)
            "ep_by_player": ep_by_player,  # For MLflow phi calculation
            "ball_handler": (
                int(game_state.env.ball_holder)
                if game_state.env.ball_holder is not None
                else -1
            ),
            "offense_ids": list(game_state.env.offense_ids),
            "is_terminal": bool(done),
            "shot_clock": int(game_state.env.shot_clock),  # Shot clock after action executed (when reward received)
        }
    )

    # Record Phi shaping diagnostics and per-player EPs for this step (if available)
    try:
        entry = {
            "step": int(len(game_state.reward_history)),
            "phi_prev": float(info.get("phi_prev", -1.0)) if info else -1.0,
            "phi_next": float(info.get("phi_next", -1.0)) if info else -1.0,
            "phi_beta": float(info.get("phi_beta", -1.0)) if info else -1.0,
            "phi_r_shape": float(info.get("phi_r_shape", 0.0)) if info else 0.0,
            "ball_handler": (
                int(game_state.env.ball_holder)
                if game_state.env.ball_holder is not None
                else -1
            ),
            "offense_ids": list(game_state.env.offense_ids),
            "defense_ids": list(game_state.env.defense_ids),
            "shot_clock": (
                int(game_state.env.shot_clock)
                if hasattr(game_state.env, "shot_clock")
                else -1
            ),
            "is_terminal": bool(done),  # Flag for terminal states
        }
        if info and "phi_ep_by_player" in info:
            try:
                ep_list = list(info.get("phi_ep_by_player", []))
                entry["ep_by_player"] = [float(x) for x in ep_list]
            except Exception:
                entry["ep_by_player"] = []
        if info and "phi_team_best_ep" in info:
            entry["team_best_ep"] = float(info.get("phi_team_best_ep", -1.0))
        if info and "phi_ball_handler_ep" in info:
            entry["ball_handler_ep"] = float(info.get("phi_ball_handler_ep", -1.0))
        # Note: best_ep_player is calculated on the frontend from ep_by_player
        game_state.phi_log.append(entry)
    except Exception:
        pass

    # Capture frame after step
    try:
        frame = game_state.env.render()
        if frame is not None:
            game_state.frames.append(frame)
    except Exception as e:
        print(f"Warning: Failed to capture frame at step {len(game_state.frames)}: {e}")
        import traceback
        traceback.print_exc()
    
    # Map action indices to names for frontend display (do this BEFORE storing episode state)
    action_names = [a.name for a in ActionType]
    actions_taken = {}
    for pid, act_idx in enumerate(full_action):
        if pid < len(full_action): 
            actions_taken[str(pid)] = action_names[act_idx] if act_idx < len(action_names) else "UNKNOWN"
    
    # Record resulting state for replay with policy probs AND the actions that led to this state
    try:
        state_with_actions = get_full_game_state(
            include_policy_probs=True,
            include_action_values=True,
            include_state_values=True,
        )
        # Store the actions that were taken to reach this state
        state_with_actions["actions_taken"] = actions_taken
        game_state.episode_states.append(state_with_actions)
    except Exception:
        pass
    _capture_turn_start_snapshot()
    # End of self-play: mark inactive when episode is done
    if game_state.self_play_active and done:
        game_state.self_play_active = False

    # Package state with policy probabilities so the frontend can render fresh annotations
    state_with_policy = get_full_game_state(
        include_policy_probs=True,
        include_state_values=True,
    )

    return {
        "status": "success",
        "state": state_with_policy,
        "actions_taken": actions_taken,
        "step_rewards": {
            "offense": float(step_rewards["offense"]),
            "defense": float(step_rewards["defense"]),
        },
        "episode_rewards": {
            "offense": float(game_state.episode_rewards["offense"]),
            "defense": float(game_state.episode_rewards["defense"]),
        },
        "pre_step_state_values": {
            "offensive_value": float(pre_step_offensive_value) if pre_step_offensive_value is not None else None,
            "defensive_value": float(pre_step_defensive_value) if pre_step_defensive_value is not None else None,
        },
        "mcts": jsonable_encoder(mcts_results) if mcts_results else None,
    }


@app.get("/api/phi_params")
def get_phi_params():
    # Return default values if game not initialized (graceful handling)
    if not game_state.env:
        return {
            "enable_phi_shaping": False,
            "phi_beta": 0.0,
            "reward_shaping_gamma": 1.0,
            "phi_use_ball_handler_only": False,
            "phi_blend_weight": 0.0,
            "phi_aggregation_mode": "team_best",
        }
    env = game_state.env
    return {
        "enable_phi_shaping": bool(getattr(env, "enable_phi_shaping", False)),
        "phi_beta": float(getattr(env, "phi_beta", 0.0)),
        "reward_shaping_gamma": float(getattr(env, "reward_shaping_gamma", 1.0)),
        "phi_use_ball_handler_only": bool(
            getattr(env, "phi_use_ball_handler_only", False)
        ),
        "phi_blend_weight": float(getattr(env, "phi_blend_weight", 0.0)),
        "phi_aggregation_mode": str(getattr(env, "phi_aggregation_mode", "team_best")),
    }


class SetPhiParamsRequest(BaseModel):
    enable_phi_shaping: bool | None = None
    phi_beta: float | None = None
    reward_shaping_gamma: float | None = None
    phi_use_ball_handler_only: bool | None = None
    phi_blend_weight: float | None = None
    phi_aggregation_mode: str | None = None


@app.post("/api/mcts_advise")
def mcts_advise(req: MCTSAdviseRequest):
    """Return an MCTS-recommended action without advancing the environment."""
    if not game_state.env or game_state.obs is None:
        raise HTTPException(status_code=400, detail="Game not initialized.")
    advice = _run_mcts_advisor(
        player_id=req.player_id,
        obs=game_state.obs,
        env=game_state.env,
        max_depth=req.max_depth,
        time_budget_ms=req.time_budget_ms,
        exploration_c=req.exploration_c,
        use_priors=req.use_priors,
    )
    return {"status": "success", "advice": jsonable_encoder(advice)}


@app.post("/api/phi_params")
def set_phi_params(req: SetPhiParamsRequest):
    if not game_state.env:
        raise HTTPException(status_code=400, detail="Game not initialized")
    try:
        env = game_state.env
        if req.enable_phi_shaping is not None:
            env.enable_phi_shaping = bool(req.enable_phi_shaping)
        if req.phi_beta is not None:
            env.phi_beta = float(req.phi_beta)
        if req.reward_shaping_gamma is not None:
            env.reward_shaping_gamma = float(req.reward_shaping_gamma)
        if req.phi_use_ball_handler_only is not None:
            env.phi_use_ball_handler_only = bool(req.phi_use_ball_handler_only)
        if req.phi_blend_weight is not None:
            try:
                env.phi_blend_weight = float(max(0.0, min(1.0, req.phi_blend_weight)))
            except Exception:
                pass
        if req.phi_aggregation_mode is not None:
            valid_modes = ["team_best", "teammates_best", "teammates_avg", "team_avg"]
            if str(req.phi_aggregation_mode) in valid_modes:
                env.phi_aggregation_mode = str(req.phi_aggregation_mode)
        return {"status": "success", "params": get_phi_params()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to set phi params: {e}")


@app.get("/api/phi_log")
def get_phi_log():
    if not game_state.env:
        raise HTTPException(status_code=400, detail="Game not initialized")
    return {"phi_log": list(game_state.phi_log)}


@app.post("/api/start_self_play")
def start_self_play():
    """Prepare deterministic self-play by resetting with current initial conditions and a fixed seed.

    This ensures that replaying with the same seed and overrides reproduces outcomes.
    """
    if not game_state.env:
        raise HTTPException(status_code=400, detail="Game not initialized.")

    # Snapshot current initial conditions
    init_positions = [(int(q), int(r)) for (q, r) in game_state.env.positions]
    init_ball_holder = (
        int(game_state.env.ball_holder)
        if game_state.env.ball_holder is not None
        else None
    )
    init_shot_clock = int(getattr(game_state.env, "shot_clock", 24))

    # Choose a seed for the episode and store everything for replay
    # Use numpy to generate a large random seed if none present
    import numpy as _np

    episode_seed = int(_np.random.SeedSequence().entropy % (2**32 - 1))

    game_state.replay_seed = episode_seed
    game_state.replay_initial_positions = init_positions
    game_state.replay_ball_holder = init_ball_holder
    game_state.replay_shot_clock = init_shot_clock
    game_state.actions_log = []
    game_state.self_play_active = True
    game_state.frames = []
    game_state.reward_history = []
    game_state.episode_rewards = {"offense": 0.0, "defense": 0.0}
    game_state.episode_states = []  # Reset episode states for new self-play
    if game_state.replay_offense_skills is None:
        game_state.replay_offense_skills = {
            "layup": list(game_state.env.offense_layup_pct_by_player),
            "three_pt": list(game_state.env.offense_three_pt_pct_by_player),
            "dunk": list(game_state.env.offense_dunk_pct_by_player),
        }
    if game_state.sampled_offense_skills is None and game_state.replay_offense_skills is not None:
        game_state.sampled_offense_skills = copy.deepcopy(game_state.replay_offense_skills)

    # Reset environment using overrides to avoid RNG draws during reset
    # Include offense_skills to maintain consistent skills throughout the episode
    options = {
        "initial_positions": init_positions,
        "ball_holder": init_ball_holder,
        "shot_clock": init_shot_clock,
        "offense_skills": game_state.replay_offense_skills,
    }
    game_state.obs, _ = game_state.env.reset(seed=episode_seed, options=options)
    game_state.prev_obs = None  # No previous observation at start
    _capture_turn_start_snapshot()

    # Capture initial frame
    try:
        frame = game_state.env.render()
        if frame is not None:
            game_state.frames.append(frame)
    except Exception:
        pass

    # Record initial state for replay (no actions_taken for initial state)
    initial_state = get_full_game_state(
        include_policy_probs=True,
        include_action_values=True,
        include_state_values=True,
    )
    game_state.episode_states.append(initial_state)

    return {
        "status": "success",
        "state": get_full_game_state(include_state_values=True),
        "seed": episode_seed,
    }


class EvaluationRequest(BaseModel):
    num_episodes: int = 100
    player_deterministic: bool = True
    opponent_deterministic: bool = True
    custom_setup: CustomEvalSetup | None = None
    randomize_offense_permutation: bool = False


def _validate_custom_eval_setup(custom_setup: CustomEvalSetup | dict | None, env) -> dict:
    """Validate and normalize a custom eval setup against the active environment."""
    if not custom_setup:
        return {}
    # Convert to plain dict
    setup = custom_setup.dict() if hasattr(custom_setup, "dict") else dict(custom_setup)
    if env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized for custom setup validation.")

    normalized: dict = {}
    shooting_mode = str(setup.get("shooting_mode") or "random").lower()
    if shooting_mode not in ("random", "fixed"):
        raise HTTPException(status_code=400, detail=f"Invalid shooting_mode: {shooting_mode}")
    normalized["shooting_mode"] = shooting_mode

    if setup.get("initial_positions") is not None:
        positions_raw = setup["initial_positions"]
        if not isinstance(positions_raw, (list, tuple)):
            raise HTTPException(status_code=400, detail="initial_positions must be a list.")
        if len(positions_raw) != env.n_players:
            raise HTTPException(
                status_code=400,
                detail=f"initial_positions must have {env.n_players} entries (one per player).",
            )
        positions: list[tuple[int, int]] = []
        seen = set()
        for pos in positions_raw:
            if not isinstance(pos, (list, tuple)) or len(pos) != 2:
                raise HTTPException(status_code=400, detail=f"Invalid position entry: {pos}")
            q, r = int(pos[0]), int(pos[1])
            if not env._is_valid_position(q, r):
                raise HTTPException(status_code=400, detail=f"Position {(q, r)} is out of bounds.")
            if (q, r) in seen:
                raise HTTPException(status_code=400, detail=f"Duplicate position {(q, r)} is not allowed.")
            seen.add((q, r))
            positions.append((q, r))
        normalized["initial_positions"] = positions

    if setup.get("ball_holder") is not None:
        bh = int(setup["ball_holder"])
        if bh < 0 or bh >= env.n_players:
            raise HTTPException(status_code=400, detail=f"Invalid ball_holder id: {bh}")
        normalized["ball_holder"] = bh

    # Validate fixed offense skills if requested
    if shooting_mode == "fixed":
        offense_ids = getattr(env, "offense_ids", [])
        offense_count = len(offense_ids)
        skills = setup.get("offense_skills")
        if not skills or not isinstance(skills, dict):
            raise HTTPException(status_code=400, detail="offense_skills are required when shooting_mode='fixed'.")
        normalized_skills: dict[str, list[float]] = {}
        for key in ("layup", "three_pt", "dunk"):
            arr = skills.get(key)
            if arr is None or len(arr) != offense_count:
                raise HTTPException(
                    status_code=400,
                    detail=f"offense_skills.{key} must have {offense_count} values.",
                )
            vals: list[float] = []
            for v in arr:
                try:
                    fv = float(v)
                except Exception:
                    raise HTTPException(status_code=400, detail=f"Invalid offense skill value: {v}")
                if fv < 0 or fv > 1:
                    raise HTTPException(status_code=400, detail="Offense skill probabilities must be between 0 and 1.")
                vals.append(fv)
            normalized_skills[key] = vals
        normalized["offense_skills"] = normalized_skills

    return normalized


@app.post("/api/pass_steal_preview")
def pass_steal_preview(req: PassStealPreviewRequest):
    """Return pass steal probabilities for a hypothetical placement (positions + ball holder).

    Does NOT mutate the active environment state; we temporarily set positions/ball_holder,
    compute probabilities, and restore.
    """
    if not game_state.env:
        raise HTTPException(status_code=400, detail="Game not initialized.")

    env = game_state.env

    if not isinstance(req.positions, (list, tuple)):
        raise HTTPException(status_code=400, detail="positions must be a list")

    if len(req.positions) != env.n_players:
        raise HTTPException(
            status_code=400,
            detail=f"positions must have {env.n_players} entries (got {len(req.positions)})",
        )

    # Validate positions are in-bounds and non-overlapping
    seen = set()
    validated_positions: list[tuple[int, int]] = []
    for pos in req.positions:
        if not isinstance(pos, (list, tuple)) or len(pos) != 2:
            raise HTTPException(status_code=400, detail=f"Invalid position entry: {pos}")
        q, r = int(pos[0]), int(pos[1])
        if not env._is_valid_position(q, r):
            raise HTTPException(status_code=400, detail=f"Position {(q, r)} is out of bounds.")
        if (q, r) in seen:
            raise HTTPException(status_code=400, detail=f"Duplicate position {(q, r)} not allowed.")
        seen.add((q, r))
        validated_positions.append((q, r))

    ball_holder = int(req.ball_holder)
    if ball_holder < 0 or ball_holder >= env.n_players:
        raise HTTPException(status_code=400, detail=f"Invalid ball_holder id: {ball_holder}")

    # Snapshot current state to restore later
    orig_positions = list(env.positions)
    orig_ball_holder = env.ball_holder
    orig_obs = game_state.obs
    orig_prev_obs = game_state.prev_obs

    try:
        env.positions = list(validated_positions)
        env.ball_holder = ball_holder
        # Build a fresh observation/action mask snapshot for preview calculations
        obs_vec = env._get_observation()
        action_mask = env._get_action_masks()
        base_obs = {
            "obs": obs_vec,
            "action_mask": action_mask,
            "role_flag": np.array([game_state.role_flag_offense], dtype=np.float32),
            "skills": env._get_offense_skills_array(),
        }
        state_values = _compute_state_values_from_obs(base_obs) or {}
        policy_probs = _compute_policy_probabilities_for_obs(base_obs, env) or {}
        entropy = {}
        if policy_probs:
            for pid, probs in policy_probs.items():
                try:
                    arr = np.array(probs, dtype=np.float32)
                    mask = (arr > 0) & np.isfinite(arr)
                    if not mask.any():
                        entropy[pid] = 0.0
                    else:
                        p = arr[mask]
                        p = p / p.sum()
                        entropy_val = float(-(p * np.log(p + 1e-9)).sum())
                        entropy[pid] = entropy_val
                except Exception:
                    entropy[pid] = 0.0

        probs = env.calculate_pass_steal_probabilities(ball_holder) or {}
        pass_probs = {int(k): float(v) for k, v in probs.items()}
        return {
            "pass_steal_probabilities": pass_probs,
            "policy_probabilities": policy_probs,
            "state_values": state_values,
            "entropy": entropy,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to compute pass steal preview: {e}")
    finally:
        env.positions = orig_positions
        env.ball_holder = orig_ball_holder
        game_state.obs = orig_obs
        game_state.prev_obs = orig_prev_obs


@app.post("/api/run_evaluation")
def run_evaluation(request: EvaluationRequest):
    """Run N episodes of self-play for evaluation purposes.

    Returns final state of each episode for stats tracking.
    Uses parallel execution on multiple CPU cores for speedup.

    Note: Uses the environment initialized in /api/init_game which loads
    all parameters (including min_shot_clock and shot_clock) from MLflow.
    """
    import time
    
    if not game_state.env:
        raise HTTPException(
            status_code=400, detail="Game not initialized. Call /api/init_game first."
        )

    if not game_state.unified_policy:
        raise HTTPException(
            status_code=400, detail="Unified policy required for evaluation."
        )
    
    # Check if we have the params needed for parallel evaluation
    if game_state.env_required_params is None or game_state.unified_policy_path is None:
        raise HTTPException(
            status_code=400, 
            detail="Missing environment parameters. Please re-initialize the game with /api/init_game."
        )

    num_episodes = max(1, min(request.num_episodes, 1000000))  # Cap at 1M for safety
    player_deterministic = request.player_deterministic
    opponent_deterministic = request.opponent_deterministic
    custom_setup = _validate_custom_eval_setup(request.custom_setup, game_state.env)
    randomize_offense_perm = bool(getattr(request, "randomize_offense_permutation", False))

    # Log shot clock configuration before evaluation
    print(f"[Evaluation] Starting {num_episodes} episodes (parallel)")
    print(f"[Evaluation] Configuration:")
    print(f"  - Player deterministic: {player_deterministic}")
    print(f"  - Opponent deterministic: {opponent_deterministic}")
    print(f"  - Using opponent policy: {game_state.defense_policy is not None}")
    print(f"  - User team: {game_state.user_team.name}")
    print(f"  - Unified policy (user): {game_state.unified_policy_key}")
    print(
        f"  - Opponent policy: {game_state.opponent_unified_policy_key or 'same as unified'}"
    )
    print(f"  - shot_clock (max): {game_state.env.shot_clock_steps}")
    print(f"  - min_shot_clock: {game_state.env.min_shot_clock}")
    print(
        f"  - Each episode starts with random shot clock in range: [{game_state.env.min_shot_clock}, {game_state.env.shot_clock_steps}] steps"
    )

    # Log policy assignment to teams
    if game_state.user_team == Team.OFFENSE:
        print(f"\n[Policy Assignment]")
        print(f"  - OFFENSE: {game_state.unified_policy_key} (user policy)")
        print(
            f"  - DEFENSE: {game_state.opponent_unified_policy_key or game_state.unified_policy_key} (opponent policy)"
        )
    else:
        print(f"\n[Policy Assignment]")
        print(
            f"  - OFFENSE: {game_state.opponent_unified_policy_key or game_state.unified_policy_key} (opponent policy)"
        )
        print(f"  - DEFENSE: {game_state.unified_policy_key} (user policy)")

    start_time = time.time()
    shot_accumulator: dict[str, list[int]] = {}
    per_player_stats: dict = {}
    
    # Choose between sequential and parallel based on episode count
    # Parallel has ~15-20s startup overhead (spawning workers, loading policies)
    # Sequential runs at ~100+ ep/s, so parallel only wins for large runs
    PARALLEL_THRESHOLD = 1000  # Only use parallel for 1000+ episodes
    
    if num_episodes >= PARALLEL_THRESHOLD and game_state.env_required_params is not None:
        print(f"[Evaluation] Using parallel execution ({num_episodes} >= {PARALLEL_THRESHOLD})")
        raw_results = _run_parallel_evaluation(
            num_episodes=num_episodes,
            player_deterministic=player_deterministic,
            opponent_deterministic=opponent_deterministic,
            required_params=game_state.env_required_params,
            optional_params=game_state.env_optional_params,
            unified_policy_path=game_state.unified_policy_path,
            opponent_policy_path=game_state.opponent_policy_path,
            user_team_name=game_state.user_team.name,
            role_flag_offense=game_state.role_flag_offense,
            role_flag_defense=game_state.role_flag_defense,
            shot_accumulator=shot_accumulator,
            custom_setup=custom_setup,
            randomize_offense_permutation=randomize_offense_perm,
        )
    else:
        print(f"[Evaluation] Using sequential execution ({num_episodes} < {PARALLEL_THRESHOLD})")
        raw_results = _run_sequential_evaluation(
            num_episodes=num_episodes,
            player_deterministic=player_deterministic,
            opponent_deterministic=opponent_deterministic,
            shot_accumulator=shot_accumulator,
            per_player_stats=per_player_stats,
            custom_setup=custom_setup,
            randomize_offense_permutation=randomize_offense_perm,
        )
    
    if isinstance(raw_results, dict):
        per_player_stats = raw_results.get("per_player_stats", {}) or {}
    else:
        per_player_stats = {}

    elapsed_time = time.time() - start_time
    
    # raw_results now wraps both episode list and shot accumulator
    if isinstance(raw_results, dict):
        episode_payload = raw_results.get("results", [])
        returned_shots = raw_results.get("shot_accumulator", {}) or {}
    else:
        episode_payload = raw_results
        returned_shots = {}

    # Merge shot data if workers provided it (parallel or sequential) and we don't already have counts
    merged_shots = False
    if returned_shots and isinstance(returned_shots, dict):
        if returned_shots is not shot_accumulator:
            for key, vals in returned_shots.items():
                try:
                    att = int(vals[0]) if isinstance(vals, (list, tuple)) and len(vals) > 0 else 0
                    mk = int(vals[1]) if isinstance(vals, (list, tuple)) and len(vals) > 1 else 0
                except Exception:
                    att, mk = 0, 0
                if key not in shot_accumulator:
                    shot_accumulator[key] = [0, 0]
                shot_accumulator[key][0] += att
                shot_accumulator[key][1] += mk
        merged_shots = True
    if not merged_shots and isinstance(episode_payload, list):
        # Fallback: merge per-episode shot_counts if provided
        for r in episode_payload:
            ep_counts = r.get("shot_counts", {}) if isinstance(r, dict) else {}
            for key, vals in (ep_counts or {}).items():
                try:
                    att = int(vals[0]) if isinstance(vals, (list, tuple)) and len(vals) > 0 else 0
                    mk = int(vals[1]) if isinstance(vals, (list, tuple)) and len(vals) > 1 else 0
                except Exception:
                    att, mk = 0, 0
                if key not in shot_accumulator:
                    shot_accumulator[key] = [0, 0]
                shot_accumulator[key][0] += att
                shot_accumulator[key][1] += mk

    # Convert results to expected format
    # Both sequential and parallel return outcome_info; we reconstruct final_state for frontend
    episode_results = []
    for r in episode_payload:
        outcome_info = r.get("outcome_info", {})
        # Reconstruct final_state with all fields needed by frontend stats recording
        final_state = {
            "last_action_results": {
                "shots": outcome_info.get("shots", {}),
                "turnovers": outcome_info.get("turnovers", []),
            },
            "shot_clock": outcome_info.get("shot_clock", 0),
            "three_point_distance": outcome_info.get("three_point_distance", 4.0),
            "user_team_name": game_state.user_team.name,
            "done": True,  # All evaluation episodes are complete
        }
        episode_results.append({
            "episode": r["episode"],
            "final_state": final_state,
            "steps": r["steps"],
            "episode_rewards": r["episode_rewards"],
        })

    # Log evaluation summary statistics with outcome analysis
    if episode_results:
        episode_lengths = [r["steps"] for r in episode_results]
        avg_length = sum(episode_lengths) / len(episode_lengths)
        min_length = min(episode_lengths)
        max_length = max(episode_lengths)

        # Analyze how episodes ended
        outcomes = {
            "made_shot": 0,
            "turnover": 0,
            "shot_clock_violation": 0,
            "other": 0,
        }
        for r in episode_results:
            final_state = r.get("final_state", {})
            last_action_results = final_state.get("last_action_results", {})
            shot_clock = final_state.get("shot_clock", 0)

            if last_action_results.get("shots"):
                # Check if shot was made
                shots = last_action_results["shots"]
                if shots:
                    first_shot = list(shots.values())[0]
                    if first_shot.get("success"):
                        outcomes["made_shot"] += 1
                    else:
                        outcomes["turnover"] += 1  # Missed shot counts as end
            elif last_action_results.get("turnovers"):
                outcomes["turnover"] += 1
            elif shot_clock <= 0:
                outcomes["shot_clock_violation"] += 1
            else:
                outcomes["other"] += 1

        print(f"[Evaluation Complete] Episode length statistics:")
        print(f"  - Average: {avg_length:.1f} steps")
        print(f"  - Min: {min_length} steps")
        print(f"  - Max: {max_length} steps")
        print(f"  - Total episodes: {len(episode_results)}")
        print(f"  - Total time: {elapsed_time:.1f}s ({num_episodes/elapsed_time:.1f} episodes/sec)")

        print(f"[Evaluation Complete] Episode outcomes:")
        print(
            f"  - Made shots: {outcomes['made_shot']} ({100*outcomes['made_shot']/len(episode_results):.1f}%)"
        )
        print(
            f"  - Turnovers: {outcomes['turnover']} ({100*outcomes['turnover']/len(episode_results):.1f}%)"
        )
        print(
            f"  - Shot clock violations: {outcomes['shot_clock_violation']} ({100*outcomes['shot_clock_violation']/len(episode_results):.1f}%)"
        )
        print(
            f"  - Other: {outcomes['other']} ({100*outcomes['other']/len(episode_results):.1f}%)"
        )

    # Reset environment to a fresh playable state after evaluation
    # This ensures the user can continue playing individual games
    print("[Evaluation] Resetting environment to playable state...")
    game_state.obs, _ = game_state.env.reset()
    game_state.prev_obs = None
    game_state.episode_states = []
    game_state.actions_log = []
    game_state.frames = []
    game_state.reward_history = []
    game_state.episode_rewards = {"offense": 0.0, "defense": 0.0}
    _capture_turn_start_snapshot()
    
    # Record initial state for potential replay
    current_game_state = get_full_game_state(
        include_policy_probs=True,
        include_action_values=True,
        include_state_values=True,
    )
    game_state.episode_states.append(current_game_state)

    try:
        # Log accumulated shot locations (q,r) -> (attempts, makes)
        sorted_items = sorted(shot_accumulator.items(), key=lambda kv: kv[0])
        print("[Evaluation] Shot location totals (q,r -> (FGA, FGM)):")
        if not sorted_items:
            print("  (no shots recorded during evaluation)")
        else:
            for loc, vals in sorted_items:
                att, mk = vals
                print(f"  {loc}: ({att}, {mk})")
    except Exception:
        pass

    return {
        "status": "success",
        "num_episodes": len(episode_results),
        "results": episode_results,
        "current_state": current_game_state,  # Fresh game state for UI to use
        "shot_accumulator": shot_accumulator,
        "per_player_stats": per_player_stats,
    }


@app.post("/api/replay_last_episode")
def replay_last_episode():
    """Return the recorded sequence of states for the last episode (manual or self-play).

    If full recorded states are available, return those directly. Otherwise, fall back
    to deterministic reconstruction using seed/initial overrides and action log.
    """
    if not game_state.env:
        raise HTTPException(status_code=400, detail="Game not initialized.")

    # Preferred: if we have recorded states, return them directly
    if (
        getattr(game_state, "episode_states", None)
        and len(game_state.episode_states) > 0
    ):
        return {"status": "success", "states": list(game_state.episode_states)}

    # Fallback to deterministic reconstruction using stored seed and actions
    if (
        game_state.replay_seed is None
        or game_state.replay_initial_positions is None
        or game_state.actions_log is None
    ):
        raise HTTPException(status_code=400, detail="No episode available to replay.")

    options = {
        "initial_positions": game_state.replay_initial_positions,
        "ball_holder": game_state.replay_ball_holder,
        "shot_clock": game_state.replay_shot_clock,
        "offense_skills": game_state.replay_offense_skills,
    }
    obs, _ = game_state.env.reset(seed=game_state.replay_seed, options=options)

    states = [get_full_game_state(include_state_values=True)]
    for action in game_state.actions_log:
        obs, _, _, _, _ = game_state.env.step(action)
        try:
            frame = game_state.env.render()
            if frame is not None:
                game_state.frames.append(frame)
        except Exception:
            pass
        states.append(get_full_game_state(include_state_values=True))

    game_state.obs = obs
    return {"status": "success", "states": states}


@app.get("/api/shot_stats")
def get_shot_stats():
    """Return raw shot log and simple aggregates to compare displayed probabilities vs outcomes."""
    logs = list(game_state.shot_log)
    total = len(logs)
    made = sum(1 for s in logs if s.get("success"))
    avg_prob = (sum(s.get("probability", 0.0) for s in logs) / total) if total else 0.0
    avg_base = sum(
        s.get("base_probability", 0.0)
        for s in logs
        if s.get("base_probability", -1.0) >= 0
    ) / max(1, sum(1 for s in logs if s.get("base_probability", -1.0) >= 0))
    avg_pressure_mult = sum(
        s.get("pressure_multiplier", 0.0)
        for s in logs
        if s.get("pressure_multiplier", -1.0) >= 0
    ) / max(1, sum(1 for s in logs if s.get("pressure_multiplier", -1.0) >= 0))
    total_three = sum(1 for s in logs if s.get("is_three"))
    made_three = sum(1 for s in logs if s.get("is_three") and s.get("success"))
    avg_prob_three = (
        sum(s.get("probability", 0.0) for s in logs if s.get("is_three")) / total_three
        if total_three
        else 0.0
    )
    avg_base_three = sum(
        s.get("base_probability", 0.0)
        for s in logs
        if s.get("is_three") and s.get("base_probability", -1.0) >= 0
    ) / max(
        1,
        sum(
            1
            for s in logs
            if s.get("is_three") and s.get("base_probability", -1.0) >= 0
        ),
    )
    avg_pressure_three = sum(
        s.get("pressure_multiplier", 0.0)
        for s in logs
        if s.get("is_three") and s.get("pressure_multiplier", -1.0) >= 0
    ) / max(
        1,
        sum(
            1
            for s in logs
            if s.get("is_three") and s.get("pressure_multiplier", -1.0) >= 0
        ),
    )
    # Group by distance
    by_distance = {}
    for s in logs:
        d = int(s.get("distance", -1))
        if d not in by_distance:
            by_distance[d] = {
                "attempts": 0,
                "made": 0,
                "avg_prob": 0.0,
                "_prob_sum": 0.0,
            }
        by_distance[d]["attempts"] += 1
        by_distance[d]["made"] += 1 if s.get("success") else 0
        by_distance[d]["_prob_sum"] += float(s.get("probability", 0.0))
    for d, agg in by_distance.items():
        attempts = max(1, agg["attempts"])  # avoid div by zero
        agg["avg_prob"] = agg["_prob_sum"] / attempts
        del agg["_prob_sum"]
    return {
        "total_attempts": total,
        "made": made,
        "make_rate": (made / total) if total else 0.0,
        "avg_prob": avg_prob,
        "avg_base_probability": avg_base,
        "avg_pressure_multiplier": avg_pressure_mult,
        "three_point": {
            "attempts": total_three,
            "made": made_three,
            "make_rate": (made_three / total_three) if total_three else 0.0,
            "avg_prob": avg_prob_three,
            "avg_base_probability": avg_base_three,
            "avg_pressure_multiplier": avg_pressure_three,
        },
        "by_distance": by_distance,
        "log": logs[-100:],  # return last 100 for brevity
    }


@app.get("/api/debug/frames")
def debug_frames():
    """Debug endpoint to check frame capture status."""
    return {
        "frames_count": len(game_state.frames) if game_state.frames else 0,
        "env_exists": game_state.env is not None,
        "render_mode": getattr(game_state.env, "render_mode", None) if game_state.env else None,
        "has_offensive_lane_hexes": hasattr(game_state.env, "offensive_lane_hexes") if game_state.env else False,
    }

class SaveEpisodeRequest(BaseModel):
    frames: List[str]  # Base64-encoded PNG images
    durations: Optional[List[float]] = None  # Optional per-frame durations in seconds
    step_duration_ms: Optional[float] = None  # Optional fallback duration per step in milliseconds


@app.post("/api/save_episode")
def save_episode():
    """Saves the recorded episode frames to a GIF in ./episodes and returns the file path."""
    print(f"[SAVE_EPISODE] Frames count: {len(game_state.frames)}")
    print(f"[SAVE_EPISODE] Env exists: {game_state.env is not None}")
    
    if not game_state.frames:
        raise HTTPException(status_code=400, detail=f"No episode frames to save. Frames list is empty (length: {len(game_state.frames) if game_state.frames else 0}).")
    # Determine directory using MLflow run_id if available
    base_dir = "episodes"
    if getattr(game_state, "run_id", None):
        base_dir = os.path.join(base_dir, str(game_state.run_id))
    os.makedirs(base_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Determine outcome label with assist annotations for shot outcomes
    outcome = "Unknown"
    category = None
    try:
        ar = game_state.env.last_action_results or {}
        if ar.get("shots"):
            # Take first shot result
            shooter_id_str = list(ar["shots"].keys())[0]
            shot_res = ar["shots"][shooter_id_str]
            distance = int(shot_res.get("distance", 999))
            is_dunk = distance == 0
            is_three = bool(
                shot_res.get("is_three")
                if "is_three" in shot_res
                else distance >= game_state.env.three_point_distance
            )
            success = bool(shot_res.get("success"))
            assist_full = bool(shot_res.get("assist_full", False))
            assist_potential = bool(shot_res.get("assist_potential", False))

            # Build detailed category
            shot_type = "dunk" if is_dunk else ("3pt" if is_three else "2pt")
            if success:
                outcome = (
                    f"Made {shot_type.upper()}" if shot_type != "dunk" else "Made Dunk"
                )
                category = (
                    f"made_assisted_{shot_type}"
                    if assist_full
                    else f"made_unassisted_{shot_type}"
                )
            else:
                outcome = (
                    f"Missed {shot_type.upper()}"
                    if shot_type != "dunk"
                    else "Missed Dunk"
                )
                if assist_potential:
                    category = f"missed_potentially_assisted_{shot_type}"
                else:
                    category = f"missed_{shot_type}"
        elif ar.get("turnovers"):
            reason = ar["turnovers"][0].get("reason", "turnover")
            if reason == "intercepted":
                outcome = "Turnover (Intercepted)"
            elif reason in ("pass_out_of_bounds", "move_out_of_bounds"):
                outcome = "Turnover (OOB)"
            elif reason == "defender_pressure":
                outcome = "Turnover (Pressure)"
            else:
                outcome = f"Turnover ({reason})"
        elif getattr(game_state.env, "shot_clock", 1) <= 0:
            outcome = "Turnover (Shot Clock Violation)"
    except Exception:
        pass

    # If we didn't build a detailed category (e.g., turnover), fall back to generic mapping
    if category is None:
        category = get_outcome_category(outcome)
    file_path = os.path.join(base_dir, f"episode_{timestamp}_{category}.gif")

    # Write frames to GIF (filter any None frames)
    try:
        valid_frames = [f for f in game_state.frames if f is not None]
        if not valid_frames:
            raise HTTPException(status_code=400, detail="No valid frames to save.")
        imageio.mimsave(file_path, valid_frames, fps=1, loop=0)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save GIF: {e}")

    # Clear frames list so next episode starts fresh
    game_state.frames = []

    return {"status": "success", "file_path": file_path}


@app.post("/api/save_episode_from_pngs")
def save_episode_from_pngs(request: SaveEpisodeRequest):
    """Saves episode from base64-encoded PNG frames sent from frontend."""
    import base64
    from PIL import Image
    import io
    
    if not request.frames or len(request.frames) == 0:
        raise HTTPException(status_code=400, detail="No frames provided")
    
    # Determine directory using MLflow run_id if available
    base_dir = "episodes"
    if getattr(game_state, "run_id", None):
        base_dir = os.path.join(base_dir, str(game_state.run_id))
    os.makedirs(base_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Determine outcome label
    outcome = "Unknown"
    category = None
    try:
        ar = game_state.env.last_action_results or {}
        if ar.get("shots"):
            shooter_id_str = list(ar["shots"].keys())[0]
            shot_res = ar["shots"][shooter_id_str]
            distance = int(shot_res.get("distance", 999))
            is_dunk = distance == 0
            is_three = bool(
                shot_res.get("is_three")
                if "is_three" in shot_res
                else distance >= game_state.env.three_point_distance
            )
            success = bool(shot_res.get("success"))
            assist_full = bool(shot_res.get("assist_full", False))
            assist_potential = bool(shot_res.get("assist_potential", False))

            shot_type = "dunk" if is_dunk else ("3pt" if is_three else "2pt")
            if success:
                outcome = f"Made {shot_type.upper()}" if shot_type != "dunk" else "Made Dunk"
                category = f"made_assisted_{shot_type}" if assist_full else f"made_unassisted_{shot_type}"
            else:
                outcome = f"Missed {shot_type.upper()}" if shot_type != "dunk" else "Missed Dunk"
                category = f"missed_potentially_assisted_{shot_type}" if assist_potential else f"missed_{shot_type}"
        elif ar.get("turnovers"):
            reason = ar["turnovers"][0].get("reason", "turnover")
            if reason == "intercepted":
                outcome = "Turnover (Intercepted)"
            elif reason in ("pass_out_of_bounds", "move_out_of_bounds"):
                outcome = "Turnover (OOB)"
            elif reason == "defender_pressure":
                outcome = "Turnover (Pressure)"
            else:
                outcome = f"Turnover ({reason})"
        elif getattr(game_state.env, "shot_clock", 1) <= 0:
            outcome = "Turnover (Shot Clock Violation)"
    except Exception:
        pass

    if category is None:
        category = get_outcome_category(outcome)
    
    file_path = os.path.join(base_dir, f"episode_{timestamp}_{category}.gif")
    
    # Decode base64 PNG frames and convert to numpy arrays
    try:
        pil_frames = []
        for base64_frame in request.frames:
            # Remove data URL prefix if present
            if ',' in base64_frame:
                base64_frame = base64_frame.split(',')[1]
            
            # Decode base64 to bytes
            img_bytes = base64.b64decode(base64_frame)
            
            # Open with PIL
            img = Image.open(io.BytesIO(img_bytes))
            
            # Convert to RGB (remove alpha channel if present)
            if img.mode == 'RGBA':
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])  # Use alpha channel as mask
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            pil_frames.append(img)

        durations_sec = None
        if request.durations and len(request.durations) > 0:
            durations_sec = [max(0.01, float(d)) for d in request.durations]
        elif request.step_duration_ms:
            dur = max(10.0, float(request.step_duration_ms))
            durations_sec = [dur / 1000.0] * len(pil_frames)
        else:
            durations_sec = [1.0] * len(pil_frames)

        if len(durations_sec) != len(pil_frames):
            # Pad or trim to match frame count
            if len(durations_sec) < len(pil_frames):
                last_d = durations_sec[-1] if durations_sec else 1.0
                durations_sec.extend([last_d] * (len(pil_frames) - len(durations_sec)))
            durations_sec = durations_sec[: len(pil_frames)]

        durations_ms = [max(10, int(round(d * 1000))) for d in durations_sec]

        if not pil_frames:
            raise HTTPException(status_code=400, detail="No frames provided after decoding")

        # Save GIF using Pillow directly to ensure per-frame durations are honored
        pil_frames[0].save(
            file_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=durations_ms,
            loop=0,
            optimize=False,
            disposal=2,
        )
        
        return {"status": "success", "file_path": file_path}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to save GIF from PNGs: {e}")


@app.get("/api/policy_probabilities")
def get_policy_probabilities():
    """
    Gets the action probabilities from the policy for the user's team,
    given the current observation.
    """
    if not game_state.env or not game_state.user_team:
        raise HTTPException(status_code=400, detail="Game not initialized.")

    try:
        response = compute_policy_probabilities()
        if response is None:
            raise HTTPException(
                status_code=500, detail="Failed to compute policy probabilities"
            )
        return jsonable_encoder(response)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get policy probabilities: {e}"
        )


@app.get("/api/debug/action_masks")
def get_action_masks_debug():
    """
    Debug endpoint to inspect raw action masks from the environment.
    Returns the action masks and key environment parameters.
    """
    if not game_state.env or game_state.obs is None:
        raise HTTPException(status_code=400, detail="Game not initialized.")

    try:
        action_mask = game_state.obs["action_mask"]

        # Get pass-related parameters
        debug_info = {
            "enable_pass_gating": getattr(game_state.env, "enable_pass_gating", None),
            "pass_arc_degrees": getattr(game_state.env, "pass_arc_degrees", None),
            "ball_holder": (
                int(game_state.env.ball_holder)
                if game_state.env.ball_holder is not None
                else None
            ),
            "positions": [(int(q), int(r)) for q, r in game_state.env.positions],
            "offense_ids": list(game_state.env.offense_ids),
            "defense_ids": list(game_state.env.defense_ids),
            "action_masks": {},
        }

        # For each player, show their action mask
        action_names = [
            "NOOP",
            "MOVE_E",
            "MOVE_NE",
            "MOVE_NW",
            "MOVE_W",
            "MOVE_SW",
            "MOVE_SE",
            "SHOOT",
            "PASS_E",
            "PASS_NE",
            "PASS_NW",
            "PASS_W",
            "PASS_SW",
            "PASS_SE",
        ]

        for player_id in range(game_state.env.n_players):
            mask = action_mask[player_id].tolist()
            debug_info["action_masks"][player_id] = {
                "mask": mask,
                "legal_actions": [
                    action_names[i] for i, m in enumerate(mask) if m == 1
                ],
                "pass_mask": mask[8:14],  # Pass actions
                "num_legal_passes": sum(mask[8:14]),
            }

        # If pass gating is enabled and ball holder exists, debug the arc logic
        if debug_info["enable_pass_gating"] and debug_info["ball_holder"] is not None:
            ball_holder = debug_info["ball_holder"]
            debug_info["pass_gating_debug"] = {}

            # Check each pass direction
            for dir_idx in range(6):
                has_teammate = game_state.env._has_teammate_in_pass_arc(
                    ball_holder, dir_idx
                )
                debug_info["pass_gating_debug"][f"direction_{dir_idx}"] = {
                    "has_teammate_in_arc": has_teammate,
                    "pass_action": action_names[8 + dir_idx],
                    "is_legal": bool(action_mask[ball_holder][8 + dir_idx] == 1),
                }

        return jsonable_encoder(debug_info)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get action masks: {e}")


@app.get("/api/action_values/{player_id}")
def get_action_values(player_id: int):
    """
    Calculates the Q-value (state-action value) for all possible actions for a given player.
    This is done via a one-step lookahead simulation.
    """
    if not game_state.env or game_state.obs is None:
        raise HTTPException(status_code=400, detail="Game not initialized.")

    # Check if the episode has ended - if so, return empty values
    if game_state.env.episode_ended:
        print(f"[API] Episode has ended, returning empty action values for player {player_id}")
        return jsonable_encoder({})

    # print(f"\n[API] Received request for Q-values for player {player_id}")
    action_values = _compute_q_values_for_player(player_id, game_state)

    # print(f"[API] Sending action values for player {player_id}:")
    import json

    # print(json.dumps(action_values, indent=2))
    return jsonable_encoder(action_values)


@app.get("/api/state_values")
def get_state_values():
    """Get the value function estimates for the PRE-STEP state from both perspectives.
    
    Uses the learned value function with role_flag conditioning.
    Uses prev_obs (cached before step) to avoid race condition where step completes first.
    
    Returns:
        offensive_value: Value function estimate for offensive team
        defensive_value: Value function estimate for defensive team
    """
    if not game_state.env or game_state.obs is None:
        raise HTTPException(status_code=400, detail="Game not initialized")
    
    if game_state.env.episode_ended:
        return {
            "offensive_value": 0.0,
            "defensive_value": 0.0
        }
    
    try:
        # Use prev_obs if available (set right before step), otherwise current obs
        # This handles race condition where step completes before this API call
        obs_to_use = game_state.prev_obs if game_state.prev_obs is not None else game_state.obs
        
        state_values = _compute_state_values_from_obs(obs_to_use)
        
        # Clear prev_obs after using it (consumed)
        game_state.prev_obs = None
        
        if state_values:
            # print(f"[STATE_VALUES] Offensive value (role_flag={game_state.role_flag_offense}): {state_values['offensive_value']:.3f}")
            # print(f"[STATE_VALUES] Defensive value (role_flag={game_state.role_flag_defense}): {state_values['defensive_value']:.3f}")
            return state_values
        else:
            return {
                "offensive_value": 0.0,
                "defensive_value": 0.0
            }
    except Exception as e:
        print(f"[ERROR] Failed to calculate state values: {e}")
        import traceback
        traceback.print_exc()
        return {
            "offensive_value": 0.0,
            "defensive_value": 0.0,
            "error": str(e)
        }


@app.get("/api/shot_probability/{player_id}")
def get_shot_probability(player_id: int):
    """Get the shot probability for a specific player.

    Returns base (unpressured) probability for UI display plus the
    pressure-adjusted probability for diagnostics.
    """
    if game_state.env is None:
        raise HTTPException(status_code=400, detail="Game not initialized")

    try:
        # Calculate distance from player to basket
        player_pos = game_state.env.positions[player_id]
        basket_pos = game_state.env.basket_position
        distance = game_state.env._hex_distance(player_pos, basket_pos)

        # Debug: Log the basic parameters
        # print(
        #     f"[SHOT_PROB_DEBUG] Player {player_id} at {player_pos}, basket at {basket_pos}, distance: {distance}"
        # )
        # print(
        #     f"[SHOT_PROB_DEBUG] Environment params: layup_pct={game_state.env.layup_pct}, three_pt_pct={game_state.env.three_pt_pct}, three_point_distance={game_state.env.three_point_distance}"
        # )
        # print(
        #     f"[SHOT_PROB_DEBUG] Shot pressure params: enabled={game_state.env.shot_pressure_enabled}, max={game_state.env.shot_pressure_max}, lambda={game_state.env.shot_pressure_lambda}"
        # )

        # Calculate base probability first (without pressure)
        d0 = 1
        d1 = max(game_state.env.three_point_distance, d0 + 1)
        p0 = game_state.env.layup_pct
        p1 = game_state.env.three_pt_pct

        if distance <= d0:
            base_prob = p0
        else:
            t = (distance - d0) / (d1 - d0)
            base_prob = p0 + (p1 - p0) * t

        # print(f"[SHOT_PROB_DEBUG] Base probability before pressure: {base_prob:.3f}")

        # Calculate pressure-adjusted probability (for logging/diagnostics)
        final_prob = game_state.env._calculate_shot_probability(player_id, distance)
        # print(
        #     f"[SHOT_PROB_DEBUG] Final shot probability after pressure: {final_prob:.3f}"
        # )

        return {
            "player_id": player_id,
            "shot_probability": float(base_prob),
            "shot_probability_final": float(final_prob),
            "distance": int(distance),
        }
    except Exception as e:
        return {"player_id": player_id, "shot_probability": 0.0, "error": str(e)}


@app.get("/api/pass_steal_probabilities")
def get_pass_steal_probabilities():
    """Get steal probabilities for passes from ball handler to each teammate."""
    if game_state.env is None:
        raise HTTPException(status_code=400, detail="Game not initialized")
    
    # Check if there is a ball handler
    if game_state.env.ball_holder is None:
        return {}
    
    try:
        steal_probs = game_state.env.calculate_pass_steal_probabilities(game_state.env.ball_holder)
        # Convert numpy types to standard Python types
        return {int(k): float(v) for k, v in steal_probs.items()}
    except Exception as e:
        print(f"[ERROR] Failed to calculate pass steal probabilities: {e}")
        import traceback
        traceback.print_exc()
        return {}


@app.get("/api/rewards")
def get_rewards():
    """Get the current reward history and episode totals."""
    import sys

    print("=" * 80, flush=True)
    # print("[DEBUG] /api/rewards endpoint called", flush=True)
    sys.stdout.flush()

    # Calculate phi shaping rewards using MLflow parameters (if available)
    # This is separate from the Phi Shaping tab which is for experimentation
    mlflow_phi_params = game_state.mlflow_phi_shaping_params
    # print(f"[DEBUG] mlflow_phi_params = {mlflow_phi_params}", flush=True)
    sys.stdout.flush()

    # Calculate MLflow-based phi shaping rewards
    mlflow_phi_r_shape_values = []
    if mlflow_phi_params and mlflow_phi_params.get("enable_phi_shaping", False):
        beta = mlflow_phi_params.get("phi_beta", 0.0)
        gamma = mlflow_phi_params.get("reward_shaping_gamma", 1.0)

        # print(
        #     f"[MLflow Phi] Calculating rewards with beta={beta}, gamma={gamma}, mode={mlflow_phi_params.get('phi_aggregation_mode')}, history_length={len(game_state.reward_history)}"
        # )

        # Calculate phi for initial state (step 0) from phi_log if available
        phi_prev = 0.0
        if game_state.phi_log and len(game_state.phi_log) > 0:
            initial_entry = game_state.phi_log[0]
            if initial_entry.get("step") == 0:
                # Recalculate initial phi using MLflow params
                initial_ep = initial_entry.get("ep_by_player", [])
                initial_ball = initial_entry.get("ball_handler", -1)
                initial_offense = initial_entry.get("offense_ids", [])
                # print(
                #     f"[MLflow Phi] Initial state: ep_by_player={initial_ep}, ball={initial_ball}, offense={initial_offense}"
                # )
                if initial_ep and initial_ball >= 0 and initial_offense:
                    phi_prev = calculate_phi_from_ep_data(
                        initial_ep, initial_ball, initial_offense, mlflow_phi_params
                    )
                    # print(
                    #     f"[MLflow Phi] Initial state phi_prev = {phi_prev} (expected from Phi Shaping tab)"
                    # )
                else:
                    print(
                        f"[MLflow Phi] WARNING: Could not calculate initial phi - missing data"
                    )

        for i, reward in enumerate(game_state.reward_history):
            ep_by_player = reward.get("ep_by_player", [])
            ball_handler = reward.get("ball_handler", -1)
            offense_ids = reward.get("offense_ids", [])
            is_terminal = reward.get("is_terminal", False)

            # Calculate phi_next using MLflow params
            phi_next = 0.0  # Terminal states have phi_next = 0
            if not is_terminal and ep_by_player:
                phi_next = calculate_phi_from_ep_data(
                    ep_by_player, ball_handler, offense_ids, mlflow_phi_params
                )
            elif not is_terminal and not ep_by_player:
                print(
                    f"[MLflow Phi] WARNING: No EP data for step {i+1}, cannot calculate phi"
                )

            # Calculate shaping reward: beta * (gamma * phi_next - phi_prev)
            # This is the TOTAL team shaping reward
            r_shape = beta * (gamma * phi_next - phi_prev)

            mlflow_phi_r_shape_values.append(r_shape)

            # Update phi_prev for next step
            phi_prev = phi_next
    else:
        # No MLflow phi shaping - all values are 0
        if mlflow_phi_params:
            print(
                f"[MLflow Phi] Phi shaping disabled in MLflow params: {mlflow_phi_params}"
            )
        else:
            print("[MLflow Phi] No MLflow phi params loaded")
        mlflow_phi_r_shape_values = [0.0] * len(game_state.reward_history)

    # Calculate phi potential values for display
    mlflow_phi_potential_values = []
    if mlflow_phi_params and mlflow_phi_params.get("enable_phi_shaping", False):
        for i, reward in enumerate(game_state.reward_history):
            ep_by_player = reward.get("ep_by_player", [])
            ball_handler = reward.get("ball_handler", -1)
            offense_ids = reward.get("offense_ids", [])
            is_terminal = reward.get("is_terminal", False)

            # Calculate phi potential using MLflow params
            phi_potential = 0.0  # Terminal states have phi = 0
            if not is_terminal and ep_by_player:
                phi_potential = calculate_phi_from_ep_data(
                    ep_by_player, ball_handler, offense_ids, mlflow_phi_params
                )
            mlflow_phi_potential_values.append(phi_potential)
    else:
        mlflow_phi_potential_values = [0.0] * len(game_state.reward_history)

    # Ensure all values are JSON serializable
    serialized_history = []

    # Add initial state (step 0) to show starting phi potential
    if mlflow_phi_params and mlflow_phi_params.get("enable_phi_shaping", False):
        initial_phi = 0.0
        if game_state.phi_log and len(game_state.phi_log) > 0:
            initial_entry = game_state.phi_log[0]
            if initial_entry.get("step") == 0:
                initial_ep = initial_entry.get("ep_by_player", [])
                initial_ball = initial_entry.get("ball_handler", -1)
                initial_offense = initial_entry.get("offense_ids", [])
                if initial_ep and initial_ball >= 0 and initial_offense:
                    initial_phi = calculate_phi_from_ep_data(
                        initial_ep, initial_ball, initial_offense, mlflow_phi_params
                    )

        serialized_history.append(
            {
                "step": 0,
                "shot_clock": 24,  # Initial shot clock
                "offense": 0.0,
                "defense": 0.0,
                "offense_reason": "Initial State",
                "defense_reason": "Initial State",
                "mlflow_phi_potential": float(initial_phi),
            }
        )

    for i, reward in enumerate(game_state.reward_history):
        mlflow_phi_r_shape = (
            mlflow_phi_r_shape_values[i] if i < len(mlflow_phi_r_shape_values) else 0.0
        )
        mlflow_phi_potential = (
            mlflow_phi_potential_values[i]
            if i < len(mlflow_phi_potential_values)
            else 0.0
        )

        # Get environment's phi shaping (per-player) and number of players
        env_phi_r_shape_per_player = reward.get("phi_r_shape", 0.0)
        offense_ids = reward.get("offense_ids", [])
        num_offensive_players = len(offense_ids) if offense_ids else 3  # Default to 3

        # Remove environment's phi shaping and add MLflow's phi shaping
        # mlflow_phi_r_shape is now the TOTAL team reward (not per-player)
        # env_phi_r_shape_per_player is per-player, so multiply by num_players for total
        env_phi_r_shape_total = env_phi_r_shape_per_player * num_offensive_players

        base_offense = float(reward["offense"]) - env_phi_r_shape_total
        base_defense = float(reward["defense"]) + env_phi_r_shape_total

        offense_with_mlflow = base_offense + mlflow_phi_r_shape
        defense_with_mlflow = base_defense - mlflow_phi_r_shape

        serialized_history.append(
            {
                "step": int(reward["step"]),
                "shot_clock": int(reward.get("shot_clock", 0)),  # Shot clock after action executed
                "offense": float(offense_with_mlflow),  # With MLflow phi shaping
                "defense": float(defense_with_mlflow),  # With MLflow phi shaping
                "offense_reason": reward.get("offense_reason", "Unknown"),
                "defense_reason": reward.get("defense_reason", "Unknown"),
                "mlflow_phi_potential": float(
                    mlflow_phi_potential
                ),  # Phi potential (Φ_next) for reference
            }
        )

    # Include reward shaping parameters so frontend can display them in Rewards tab
    env = game_state.env
    reward_params = {}
    try:
        reward_params = {
            # Absolute team-averaged rewards
            "pass_reward": float(getattr(env, "pass_reward", 0.0)),
            "turnover_reward": 0.0,  # Explicitly zeroed for dense reward scheme
            # Shot rewards now use expected points (shot value × pressure-adjusted make prob)
            "shot_reward_type": "expected_points",
            "shot_reward_description": "Reward = shot_value × pressure-adjusted make probability (applies to makes and misses)",
            "violation_reward": float(getattr(env, "violation_reward", 0.0)),
            # Percentage-based assist shaping
            "potential_assist_pct": float(getattr(env, "potential_assist_pct", 0.0)),
            "full_assist_bonus_pct": float(getattr(env, "full_assist_bonus_pct", 0.0)),
            "assist_window": int(getattr(env, "assist_window", 2)),
        }
    except Exception:
        reward_params = {}

    # Include MLflow phi shaping parameters
    mlflow_phi_params_serialized = {}
    if mlflow_phi_params:
        try:
            mlflow_phi_params_serialized = {
                "enable_phi_shaping": bool(
                    mlflow_phi_params.get("enable_phi_shaping", False)
                ),
                "phi_beta": float(mlflow_phi_params.get("phi_beta", 0.0)),
                "reward_shaping_gamma": float(
                    mlflow_phi_params.get("reward_shaping_gamma", 1.0)
                ),
                "phi_aggregation_mode": str(
                    mlflow_phi_params.get("phi_aggregation_mode", "team_best")
                ),
                "phi_use_ball_handler_only": bool(
                    mlflow_phi_params.get("phi_use_ball_handler_only", False)
                ),
                "phi_blend_weight": float(
                    mlflow_phi_params.get("phi_blend_weight", 0.0)
                ),
            }
        except Exception:
            pass

    return {
        "reward_history": serialized_history,
        "episode_rewards": {
            "offense": float(game_state.episode_rewards["offense"]),
            "defense": float(game_state.episode_rewards["defense"]),
        },
        "reward_params": reward_params,
        "mlflow_phi_params": mlflow_phi_params_serialized,
    }


def compute_policy_probabilities():
    """
    Helper function to compute policy probabilities for the current observation.
    Returns a dict mapping player_id to their action probabilities, or None if error.
    """
    if not game_state.env or not game_state.unified_policy or game_state.obs is None:
        return None

    try:
        # Get probabilities from unified_policy (for user team or both if single policy)
        _, raw_probs_main = _predict_policy_actions(
            game_state.unified_policy,
            game_state.obs,
            game_state.env,
            deterministic=False,
            strategy=IllegalActionStrategy.SAMPLE_PROB,
        )

        # Get probabilities from defense_policy (for opponent team if available)
        raw_probs_opponent = None
        if game_state.defense_policy is not None:
            _, raw_probs_opponent = _predict_policy_actions(
                game_state.defense_policy,
                game_state.obs,
                game_state.env,
                deterministic=False,
                strategy=IllegalActionStrategy.SAMPLE_PROB,
            )

        if raw_probs_main is None:
            return None

        # Apply action mask so illegal actions have probability 0, then renormalize.
        action_mask = game_state.obs["action_mask"]  # shape (n_players, n_actions)
        probs_list = []
        
        for pid in range(game_state.env.n_players):
            # Determine if this player should use the main policy (user team) or opponent policy
            is_user_team = (
                (pid in game_state.env.offense_ids and game_state.user_team == Team.OFFENSE) or
                (pid in game_state.env.defense_ids and game_state.user_team == Team.DEFENSE)
            )
            
            if is_user_team or raw_probs_opponent is None:
                probs = raw_probs_main[pid]
            else:
                probs = raw_probs_opponent[pid]

            masked = probs * action_mask[pid]
            total = masked.sum()
            if total > 0:
                masked = masked / total
            probs_list.append(masked.tolist())

        # Return as a dictionary mapping player_id to their list of probabilities
        # We now include ALL players (offense and defense)
        response = {
            player_id: probs
            for player_id, probs in enumerate(probs_list)
        }
        return response
    except Exception as e:
        print(f"[compute_policy_probabilities] Error: {e}")
        return None


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


@app.post("/api/offense_skills")
def set_offense_skills(req: SetOffenseSkillsRequest):
    """
    Override or reset the per-offensive-player shooting percentages for the current episode.
    """
    if not game_state.env:
        raise HTTPException(status_code=400, detail="Game not initialized.")

    env = game_state.env
    count = env.players_per_side

    def _normalize(values: List[float] | None, name: str) -> List[float]:
        if values is None:
            raise HTTPException(status_code=400, detail=f"Missing {name} values.")
        if len(values) != count:
            raise HTTPException(
                status_code=400,
                detail=f"{name} must include {count} values (one per offensive player).",
            )
        normalized: List[float] = []
        for v in values:
            try:
                val = float(v)
            except Exception:
                raise HTTPException(
                    status_code=400, detail=f"Invalid {name} value: {v}"
                )
            normalized.append(float(max(0.01, min(0.99, val))))
        return normalized

    target_skills: dict | None = None
    if req.reset_to_sampled:
        baseline = game_state.sampled_offense_skills or {}
        target_skills = {
            "layup": _normalize(
                baseline.get("layup")
                or list(env.offense_layup_pct_by_player),
                "layup",
            ),
            "three_pt": _normalize(
                baseline.get("three_pt")
                or list(env.offense_three_pt_pct_by_player),
                "three_pt",
            ),
            "dunk": _normalize(
                baseline.get("dunk")
                or list(env.offense_dunk_pct_by_player),
                "dunk",
            ),
        }
    elif req.skills is not None:
        target_skills = {
            "layup": _normalize(req.skills.layup, "layup"),
            "three_pt": _normalize(req.skills.three_pt, "three_pt"),
            "dunk": _normalize(req.skills.dunk, "dunk"),
        }
    else:
        raise HTTPException(
            status_code=400,
            detail="Provide skills or set reset_to_sampled=true to revert.",
        )

    # Apply to environment
    for i in range(count):
        env.offense_layup_pct_by_player[i] = float(target_skills["layup"][i])
        env.offense_three_pt_pct_by_player[i] = float(target_skills["three_pt"][i])
        env.offense_dunk_pct_by_player[i] = float(target_skills["dunk"][i])

    # Update cached skills for replay/self-play consistency
    game_state.replay_offense_skills = copy.deepcopy(target_skills)

    # Refresh skills in observation to keep UI/policies consistent
    try:
        skills_array = env._get_offense_skills_array()
    except Exception:
        skills_array = None

    if game_state.obs is not None and skills_array is not None:
        game_state.obs["skills"] = skills_array
        game_state.prev_obs = None

    return {
        "status": "success",
        "state": get_full_game_state(
            include_policy_probs=True,
            include_action_values=True,
            include_state_values=True,
        ),
    }


@app.post("/api/batch_update_player_positions")
def batch_update_player_positions(req: BatchUpdatePositionRequest):
    """
    Updates positions for multiple players at once.
    Useful for resetting state or bulk moves.
    """
    if not game_state.env or game_state.obs is None:
        raise HTTPException(status_code=400, detail="Game not initialized.")
    
    if game_state.env.episode_ended:
        raise HTTPException(status_code=400, detail="Cannot move players after episode has ended.")

    # Validate all moves first before applying any
    # Check bounds and duplicate targets
    target_positions = {}
    for update in req.updates:
        pid = update.player_id
        pos = (update.q, update.r)
        
        if pid < 0 or pid >= game_state.env.n_players:
             raise HTTPException(status_code=400, detail=f"Invalid player ID: {pid}")
        
        if not game_state.env._is_valid_position(*pos):
             raise HTTPException(status_code=400, detail=f"Position {pos} is out of bounds.")
             
        if pos in target_positions.values():
             raise HTTPException(status_code=400, detail=f"Duplicate target position {pos} in batch.")
        
        target_positions[pid] = pos

    # Check collisions with players NOT being moved
    # (If we swap two players, that's fine, but we can't move into a static player's hex)
    current_positions = game_state.env.positions
    moving_pids = set(target_positions.keys())
    
    for i, pos in enumerate(current_positions):
        if i not in moving_pids:
            # This player is static. Check if anyone is moving into their spot.
            if pos in target_positions.values():
                raise HTTPException(status_code=400, detail=f"Position {pos} is occupied by static Player {i}.")

    # Apply updates
    for pid, pos in target_positions.items():
        game_state.env.positions[pid] = pos

    # Refresh Observation
    try:
        obs_vec = game_state.env._get_observation()
        action_mask = game_state.env._get_action_masks()
        
        new_obs_dict = {
            "obs": obs_vec,
            "action_mask": action_mask,
            "role_flag": np.array(
                [1.0 if game_state.env.training_team == Team.OFFENSE else -1.0],
                dtype=np.float32,
            ),
            "skills": game_state.env._get_offense_skills_array(),
        }
        
        game_state.obs = new_obs_dict
        game_state.prev_obs = None
        
        return {
            "status": "success",
            "state": get_full_game_state(
                include_policy_probs=True,
                include_action_values=True,
                include_state_values=True,
            ),
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to batch update positions: {e}")


@app.post("/api/update_player_position")
def update_player_position(req: UpdatePositionRequest):
    """
    Updates a player's position during an ongoing episode.
    Recalculates observation and policy probabilities.
    Does NOT advance the simulation step.
    """
    if not game_state.env or game_state.obs is None:
        raise HTTPException(status_code=400, detail="Game not initialized.")
    
    if game_state.env.episode_ended:
        raise HTTPException(status_code=400, detail="Cannot move players after episode has ended.")

    pid = req.player_id
    new_pos = (req.q, req.r)

    # 1. Validate Player ID
    if pid < 0 or pid >= game_state.env.n_players:
        raise HTTPException(status_code=400, detail=f"Invalid player ID: {pid}")

    # 2. Validate Board Bounds
    if not game_state.env._is_valid_position(*new_pos):
        raise HTTPException(status_code=400, detail=f"Position {new_pos} is out of bounds.")

    # 3. Validate Occupancy
    # Check if any OTHER player is at this position
    for i, pos in enumerate(game_state.env.positions):
        if i != pid and pos == new_pos:
             raise HTTPException(status_code=400, detail=f"Position {new_pos} is occupied by Player {i}.")

    # 4. Update Position
    # Update the list in place
    game_state.env.positions[pid] = new_pos

    # 5. Refresh Observation
    # We must regenerate the observation since positions changed
    # We need to call reset() style logic but without resetting everything?
    # Actually env._get_obs() is what generates the observation dict from current state.
    # BasketWorldEnv usually returns (obs, info) from step/reset.
    # We can just call _get_obs() manually if we access the private method, 
    # or better, ensure we have a way to refresh `game_state.obs`.
    
    try:
        # Re-generate observation based on new positions
        # Manually construct the observation dictionary since _get_obs() doesn't exist
        # (BasketWorldEnv constructs it manually in step/reset)
        obs_vec = game_state.env._get_observation()
        action_mask = game_state.env._get_action_masks()
        
        # Construct the full dict as expected by the agent/frontend
        new_obs_dict = {
            "obs": obs_vec,
            "action_mask": action_mask,
            "role_flag": np.array(
                [1.0 if game_state.env.training_team == Team.OFFENSE else -1.0],
                dtype=np.float32,
            ),
            "skills": game_state.env._get_offense_skills_array(),
        }
        
        # Update the game state's observation
        game_state.obs = new_obs_dict
        
        # Clear cached previous observation to avoid staleness in state value calculations
        game_state.prev_obs = None
        
        # 6. Return updated state (including recomputed policy probs)
        return {
            "status": "success",
            "state": get_full_game_state(
                include_policy_probs=True,
                include_action_values=True,
                include_state_values=True,
            ),
        }
    except Exception as e:
        print(f"[update_player_position] Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to update position: {e}")


@app.post("/api/set_shot_clock")
def set_shot_clock(req: UpdateShotClockRequest):
    if not game_state.env or game_state.obs is None:
        raise HTTPException(status_code=400, detail="Game not initialized.")
    if game_state.env.episode_ended:
        raise HTTPException(status_code=400, detail="Cannot adjust shot clock after episode has ended.")

    min_clock = 0
    max_candidate = getattr(game_state.env, "shot_clock_steps", None)
    raw_value = getattr(game_state.env, "shot_clock", 0)
    # Allow adjustments up to whichever is larger: configured max, current clock, or turn-start clock
    candidate_values = [
        max_candidate,
        raw_value,
        getattr(game_state, "turn_start_shot_clock", None),
    ]
    max_clock = max(
        (int(val) for val in candidate_values if val is not None),
        default=24,
    )
    proposed = raw_value + int(req.delta)
    new_value = max(min_clock, min(max_clock, proposed))

    try:
        game_state.env.shot_clock = new_value

        obs_vec = game_state.env._get_observation()
        action_mask = game_state.env._get_action_masks()
        new_obs_dict = {
            "obs": obs_vec,
            "action_mask": action_mask,
            "role_flag": np.array(
                [1.0 if game_state.env.training_team == Team.OFFENSE else -1.0],
                dtype=np.float32,
            ),
            "skills": game_state.env._get_offense_skills_array(),
        }
        game_state.obs = new_obs_dict
        game_state.prev_obs = None

        return {
            "status": "success",
            "state": get_full_game_state(
                include_policy_probs=True,
                include_action_values=True,
                include_state_values=True,
            ),
        }
    except Exception as e:
        print(f"[set_shot_clock] Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to set shot clock: {e}")


@app.post("/api/set_ball_holder")
def set_ball_holder(req: SetBallHolderRequest):
    """Manually set the ball handler during a live game (offense only)."""
    if not game_state.env or game_state.obs is None:
        raise HTTPException(status_code=400, detail="Game not initialized.")
    if game_state.env.episode_ended:
        raise HTTPException(status_code=400, detail="Cannot set ball holder after episode has ended.")

    pid = int(req.player_id)
    if pid not in getattr(game_state.env, "offense_ids", []):
        raise HTTPException(status_code=400, detail="Ball holder must be an offensive player.")

    try:
        game_state.env.ball_holder = pid

        obs_vec = game_state.env._get_observation()
        action_mask = game_state.env._get_action_masks()
        new_obs_dict = {
            "obs": obs_vec,
            "action_mask": action_mask,
            "role_flag": np.array(
                [1.0 if game_state.env.training_team == Team.OFFENSE else -1.0],
                dtype=np.float32,
            ),
            "skills": game_state.env._get_offense_skills_array(),
        }
        game_state.obs = new_obs_dict
        game_state.prev_obs = None

        return {
            "status": "success",
            "state": get_full_game_state(
                include_policy_probs=True,
                include_action_values=True,
                include_state_values=True,
            ),
        }
    except Exception as e:
        print(f"[set_ball_holder] Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to set ball holder: {e}")


@app.post("/api/set_pass_target_strategy")
def set_pass_target_strategy(req: SetPassTargetStrategyRequest):
    if not game_state.env or game_state.obs is None:
        raise HTTPException(status_code=400, detail="Game not initialized.")
    if game_state.env.episode_ended:
        raise HTTPException(status_code=400, detail="Cannot adjust pass target strategy after episode has ended.")

    try:
        strategy = str(req.strategy).lower()
        if strategy not in ("nearest", "best_ev"):
            raise HTTPException(status_code=400, detail="Invalid pass target strategy.")

        game_state.env.set_pass_target_strategy(strategy)

        obs_vec = game_state.env._get_observation()
        action_mask = game_state.env._get_action_masks()
        new_obs_dict = {
            "obs": obs_vec,
            "action_mask": action_mask,
            "role_flag": np.array(
                [1.0 if game_state.env.training_team == Team.OFFENSE else -1.0],
                dtype=np.float32,
            ),
            "skills": game_state.env._get_offense_skills_array(),
        }
        game_state.obs = new_obs_dict
        game_state.prev_obs = None

        return {
            "status": "success",
            "state": get_full_game_state(
                include_policy_probs=True,
                include_action_values=True,
                include_state_values=True,
            ),
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"[set_pass_target_strategy] Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/swap_policies")
def swap_policies(req: SwapPoliciesRequest):
    """Swap the active PPO policies without resetting the environment."""
    if not game_state.env or game_state.obs is None:
        raise HTTPException(status_code=400, detail="Game not initialized.")
    if not game_state.run_id:
        raise HTTPException(status_code=400, detail="No MLflow run associated with current game.")

    requested_user_policy = req.user_policy_name
    requested_opponent_policy = req.opponent_policy_name

    # Ignore no-op requests
    if requested_user_policy is None and requested_opponent_policy is None:
        raise HTTPException(status_code=400, detail="No policy requested for swap.")

    client = mlflow.tracking.MlflowClient()
    custom_objects = {
        "policy_class": PassBiasDualCriticPolicy,
        "PassBiasDualCriticPolicy": PassBiasDualCriticPolicy,
        "PassBiasMultiInputPolicy": PassBiasMultiInputPolicy,
    }

    policies_changed = False

    if requested_user_policy is not None:
        # Avoid reloading the same policy
        if requested_user_policy != game_state.unified_policy_key:
            try:
                user_path = get_unified_policy_path(client, game_state.run_id, requested_user_policy)
                game_state.unified_policy = PPO.load(user_path, custom_objects=custom_objects)
                game_state.unified_policy_key = os.path.basename(user_path)
                game_state.unified_policy_path = user_path  # keep eval workers in sync
                print(f"[SWAP] Loaded user policy: {game_state.unified_policy_key} ({user_path})")
                try:
                    counts = _compute_param_counts_from_policy(game_state.unified_policy)
                    if counts:
                        if game_state.mlflow_training_params is None:
                            game_state.mlflow_training_params = {}
                        game_state.mlflow_training_params["param_counts"] = counts
                except Exception:
                    pass
                policies_changed = True
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to load user policy '{requested_user_policy}': {e}")

    if requested_opponent_policy is not None:
        # Empty string indicates "mirror user policy"
        if requested_opponent_policy == "":
            if game_state.defense_policy is not None or game_state.opponent_unified_policy_key is not None:
                game_state.defense_policy = None
                game_state.opponent_unified_policy_key = None
                game_state.opponent_policy_path = None
                print("[SWAP] Opponent policy set to mirror user policy")
                policies_changed = True
        elif requested_opponent_policy != game_state.opponent_unified_policy_key:
            try:
                opp_path = get_unified_policy_path(client, game_state.run_id, requested_opponent_policy)
                game_state.defense_policy = PPO.load(opp_path, custom_objects=custom_objects)
                game_state.opponent_unified_policy_key = os.path.basename(opp_path)
                game_state.opponent_policy_path = opp_path  # keep eval workers in sync
                print(f"[SWAP] Loaded opponent policy: {game_state.opponent_unified_policy_key} ({opp_path})")
                # For completeness, keep param counts aligned with current unified policy
                try:
                    counts = _compute_param_counts_from_policy(game_state.unified_policy)
                    if counts:
                        if game_state.mlflow_training_params is None:
                            game_state.mlflow_training_params = {}
                        game_state.mlflow_training_params["param_counts"] = counts
                except Exception:
                    pass
                policies_changed = True
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to load opponent policy '{requested_opponent_policy}': {e}")

    if not policies_changed:
        return {
            "status": "no_change",
            "state": get_full_game_state(
                include_policy_probs=True,
                include_action_values=True,
                include_state_values=True,
            ),
        }

    updated_state = get_full_game_state(
        include_policy_probs=True,
        include_action_values=True,
        include_state_values=True,
    )

    # Update the latest episode snapshot so manual replay reflects new policy context
    if game_state.episode_states:
        game_state.episode_states[-1] = updated_state

    return {"status": "success", "state": updated_state}


@app.post("/api/reset_turn_state")
def reset_turn_state():
    """Restore positions/ball holder/shot clock to the start-of-turn snapshot."""
    if not game_state.env or game_state.obs is None:
        raise HTTPException(status_code=400, detail="Game not initialized.")
    if not game_state.turn_start_positions:
        raise HTTPException(status_code=400, detail="No turn snapshot available.")

    env = game_state.env
    try:
        for idx, pos in enumerate(game_state.turn_start_positions):
            if idx < len(env.positions):
                env.positions[idx] = (int(pos[0]), int(pos[1]))
        env.ball_holder = (
            int(game_state.turn_start_ball_holder)
            if game_state.turn_start_ball_holder is not None
            else None
        )
        if game_state.turn_start_shot_clock is not None:
            env.shot_clock = int(game_state.turn_start_shot_clock)

        obs_vec = env._get_observation()
        action_mask = env._get_action_masks()
        new_obs_dict = {
            "obs": obs_vec,
            "action_mask": action_mask,
            "role_flag": np.array(
                [1.0 if env.training_team == Team.OFFENSE else -1.0],
                dtype=np.float32,
            ),
            "skills": env._get_offense_skills_array(),
        }
        game_state.obs = new_obs_dict
        game_state.prev_obs = None

        return {
            "status": "success",
            "state": get_full_game_state(
                include_policy_probs=True,
                include_action_values=True,
                include_state_values=True,
            ),
        }
    except Exception as e:
        print(f"[reset_turn_state] Error: {e}")
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to reset turn: {e}")


def get_full_game_state(
    include_policy_probs: bool = False,
    include_action_values: bool = False,
    include_state_values: bool = False,
):
    """Helper function to construct the full game state dictionary."""
    if not game_state.env:
        return {}

    # Use FastAPI's own jsonable_encoder with custom rules for numpy types.
    # This is the robust way to handle serialization.
    custom_encoder = {
        np.integer: int,
        np.floating: float,
        np.bool_: bool,
    }

    last_action_results_py = jsonable_encoder(
        game_state.env.last_action_results, custom_encoder=custom_encoder
    )

    # Convert numpy types to standard Python types for JSON serialization
    positions_py = [(int(q), int(r)) for q, r in game_state.env.positions]
    ball_holder_py = (
        int(game_state.env.ball_holder)
        if game_state.env.ball_holder is not None
        else None
    )
    basket_pos_py = (
        int(game_state.env.basket_position[0]),
        int(game_state.env.basket_position[1]),
    )
    action_mask_py = game_state.obs["action_mask"].tolist()

    # Calculate ball handler's pressure-adjusted shot probability for replay
    ball_handler_shot_prob = None
    if ball_holder_py is not None:
        try:
            player_pos = game_state.env.positions[ball_holder_py]
            basket_pos = game_state.env.basket_position
            distance = game_state.env._hex_distance(player_pos, basket_pos)
            ball_handler_shot_prob = float(
                game_state.env._calculate_shot_probability(ball_holder_py, distance)
            )
        except Exception:
            ball_handler_shot_prob = None

    # Calculate pass steal probabilities for replay
    pass_steal_probs = {}
    if ball_holder_py is not None:
        try:
            steal_probs = game_state.env.calculate_pass_steal_probabilities(ball_holder_py)
            pass_steal_probs = {int(k): float(v) for k, v in steal_probs.items()}
        except Exception as e:
            print(f"[get_full_game_state] Failed to calculate pass steal probabilities: {e}")
            pass_steal_probs = {}

    # Calculate EP (expected points) for all players
    ep_by_player = []
    try:
        env = game_state.env
        for pid in range(env.n_players):
            pos = env.positions[pid]
            dist = env._hex_distance(pos, env.basket_position)
            is_three = env.is_three_point_location(pos)
            if getattr(env, "allow_dunks", True) and dist == 0:
                shot_value = 2.0
            else:
                shot_value = 3.0 if is_three else 2.0
            p = float(env._calculate_shot_probability(pid, dist))
            ep = float(shot_value * p)
            ep_by_player.append(ep)
    except Exception:
        # If EP calculation fails, use empty list
        ep_by_player = []

    sampled_offense_skills = getattr(game_state, "sampled_offense_skills", None) or {}

    state = {
        "players_per_side": int(getattr(game_state.env, "players_per_side", 3)),
        "players": int(getattr(game_state.env, "players_per_side", 3)),
        "positions": positions_py,
        "ball_holder": ball_holder_py,
        "ball_handler_shot_probability": ball_handler_shot_prob,
        "pass_steal_probabilities": pass_steal_probs,
        "shot_clock": int(game_state.env.shot_clock),
        "min_shot_clock": int(getattr(game_state.env, "min_shot_clock", 10)),
        "shot_clock_steps": int(
            getattr(game_state.env, "shot_clock_steps", getattr(game_state.env, "shot_clock", 24))
        ),
        "user_team_name": game_state.user_team.name,
        "done": game_state.env.episode_ended,
        "training_team": (
            getattr(game_state.env, "training_team", None).name
            if getattr(game_state.env, "training_team", None)
            else None
        ),
        "action_space": {action.name: action.value for action in ActionType},
        "action_mask": action_mask_py,
        # Observation vector (main state) - all features the RL agent sees
        "obs": game_state.obs["obs"].tolist() if game_state.obs and "obs" in game_state.obs else [],
        "last_action_results": last_action_results_py,
        "offense_ids": game_state.env.offense_ids,
        "defense_ids": game_state.env.defense_ids,
        "basket_position": basket_pos_py,
        "court_width": game_state.env.court_width,
        "court_height": game_state.env.court_height,
        "three_point_distance": float(getattr(game_state.env, "three_point_distance", 4.0)),
        "three_point_short_distance": (
            float(getattr(game_state.env, "three_point_short_distance"))
            if getattr(game_state.env, "three_point_short_distance", None) is not None
            else None
        ),
        "three_point_hexes": [
            (int(q), int(r))
            for q, r in getattr(game_state.env, "_three_point_hexes", set())
        ],
        "three_point_line_hexes": [
            (int(q), int(r))
            for q, r in getattr(game_state.env, "_three_point_line_hexes", set())
        ],
        "three_point_outline": [
            (float(x), float(y))
            for x, y in getattr(game_state.env, "_three_point_outline_points", [])
        ],
        "shot_probs": getattr(game_state.env, "shot_probs", None),
        # Expose global shot means/stds used to sample per-player skills each episode
        "shot_params": {
            "layup_pct": float(getattr(game_state.env, "layup_pct", 0.0)),
            "three_pt_pct": float(getattr(game_state.env, "three_pt_pct", 0.0)),
            "dunk_pct": float(getattr(game_state.env, "dunk_pct", 0.0)),
            "layup_std": float(getattr(game_state.env, "layup_std", 0.0)),
            "three_pt_std": float(getattr(game_state.env, "three_pt_std", 0.0)),
            "dunk_std": float(getattr(game_state.env, "dunk_std", 0.0)),
            "allow_dunks": bool(getattr(game_state.env, "allow_dunks", False)),
        },
        "defender_pressure_distance": int(
            getattr(game_state.env, "defender_pressure_distance", 1)
        ),
        "defender_pressure_turnover_chance": float(
            getattr(game_state.env, "defender_pressure_turnover_chance", 0.05)
        ),
        "defender_pressure_decay_lambda": float(
            getattr(game_state.env, "defender_pressure_decay_lambda", 1.0)
        ),
        "base_steal_rate": float(getattr(game_state.env, "base_steal_rate", 0.35)),
        "steal_perp_decay": float(getattr(game_state.env, "steal_perp_decay", 1.5)),
        "steal_distance_factor": float(getattr(game_state.env, "steal_distance_factor", 0.08)),
        "steal_position_weight_min": float(getattr(game_state.env, "steal_position_weight_min", 0.3)),
        "spawn_distance": int(getattr(game_state.env, "spawn_distance", 3)),
        "max_spawn_distance": (
            int(getattr(game_state.env, "max_spawn_distance", None))
            if getattr(game_state.env, "max_spawn_distance", None) is not None
            else None
        ),
        "defender_spawn_distance": int(getattr(game_state.env, "defender_spawn_distance", 0)),
        "defender_guard_distance": int(getattr(game_state.env, "defender_guard_distance", 1)),
        "shot_pressure_enabled": bool(
            getattr(game_state.env, "shot_pressure_enabled", True)
        ),
        "shot_pressure_max": float(getattr(game_state.env, "shot_pressure_max", 0.5)),
        "shot_pressure_lambda": float(
            getattr(game_state.env, "shot_pressure_lambda", 1.0)
        ),
        "shot_pressure_arc_degrees": float(
            getattr(game_state.env, "shot_pressure_arc_degrees", 60.0)
        ),
        "mask_occupied_moves": bool(
            getattr(game_state.env, "mask_occupied_moves", False)
        ),
        # 3-second violation rules (shared configuration)
        "three_second_lane_width": int(
            getattr(game_state.env, "three_second_lane_width", 1)
        ),
        "three_second_lane_height": int(
            getattr(game_state.env, "three_second_lane_height", 3)
        ),
        "three_second_max_steps": int(
            getattr(game_state.env, "three_second_max_steps", 3)
        ),
        "illegal_defense_enabled": bool(
            getattr(game_state.env, "illegal_defense_enabled", False)
        ),
        "offensive_three_seconds_enabled": bool(
            getattr(game_state.env, "offensive_three_seconds_enabled", False)
        ),
        # Observation space configuration
        "include_hoop_vector": bool(
            getattr(game_state.env, "include_hoop_vector", False)
        ),
        # Lane hexes for visualization (convert set to list of tuples)
        "offensive_lane_hexes": [
            (int(q), int(r))
            for q, r in getattr(game_state.env, "offensive_lane_hexes", set())
        ],
        "defensive_lane_hexes": [
            (int(q), int(r))
            for q, r in getattr(game_state.env, "defensive_lane_hexes", set())
        ],
        # Lane step counts for all players
        "offensive_lane_steps": {
            int(pid): int(steps)
            for pid, steps in getattr(game_state.env, "_offensive_lane_steps", {}).items()
        },
        "defensive_lane_steps": {
            int(pid): int(steps)
            for pid, steps in getattr(game_state.env, "_defender_in_key_steps", {}).items()
        },
        # Pass parameters
        "pass_arc_degrees": float(getattr(game_state.env, "pass_arc_degrees", 60.0)),
        "pass_oob_turnover_prob": float(
            getattr(game_state.env, "pass_oob_turnover_prob", 1.0)
        ),
        "pass_target_strategy": getattr(game_state.env, "pass_target_strategy", "nearest"),
        # Illegal action policy
        "illegal_action_policy": (
            getattr(game_state.env, "illegal_action_policy", None).value
            if getattr(game_state.env, "illegal_action_policy", None)
            else "noop"
        ),
        # Pass logit bias (policy parameter, not env)
        "pass_logit_bias": float(
            getattr(game_state.unified_policy.policy, "pass_logit_bias", 0.0)
            if game_state.unified_policy
            and hasattr(game_state.unified_policy, "policy")
            else 0.0
        ),
        # MLflow metadata for UI display
        "run_id": getattr(game_state, "run_id", None),
        "run_name": getattr(game_state, "run_name", None),
        # Training parameters from MLflow (PPO hyperparameters)
        "training_params": getattr(game_state, "mlflow_training_params", None),
        # Policies in use
        "unified_policy_name": getattr(game_state, "unified_policy_key", None),
        "opponent_unified_policy_name": getattr(
            game_state, "opponent_unified_policy_key", None
        ),
        # Per-episode sampled shooting skills (offense only), aligned by offense index
        "offense_shooting_pct_by_player": {
            "layup": [
                float(x)
                for x in getattr(game_state.env, "offense_layup_pct_by_player", [])
            ],
            "three_pt": [
                float(x)
                for x in getattr(game_state.env, "offense_three_pt_pct_by_player", [])
            ],
            "dunk": [
                float(x)
                for x in getattr(game_state.env, "offense_dunk_pct_by_player", [])
            ],
        },
        "offense_shooting_pct_sampled": {
            "layup": [
                float(x)
                for x in sampled_offense_skills.get("layup", [])
            ],
            "three_pt": [
                float(x)
                for x in sampled_offense_skills.get("three_pt", [])
            ],
            "dunk": [
                float(x)
                for x in sampled_offense_skills.get("dunk", [])
            ],
        },
        # Expected points for all players (indexed by player ID)
        "ep_by_player": ep_by_player,
    }

    # Optionally include policy probabilities
    if include_policy_probs:
        policy_probs = compute_policy_probabilities()
        if policy_probs is not None:
            state["policy_probabilities"] = policy_probs

    if include_action_values:
        try:
            action_values_by_player = {}
            for pid in range(game_state.env.n_players):
                action_values_by_player[str(pid)] = _compute_q_values_for_player(pid, game_state)
            state["action_values"] = action_values_by_player
        except Exception as e:
            print(f"[get_full_game_state] Failed to compute action values: {e}")

    if include_state_values:
        state_values = _compute_state_values_from_obs(game_state.obs)
        if state_values:
            state["state_values"] = state_values

    return state
