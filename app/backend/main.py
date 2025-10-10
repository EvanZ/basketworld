import tempfile
import re
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder
import numpy as np
import basketworld
from basketworld.envs.basketworld_env_v2 import Team, ActionType
from stable_baselines3 import PPO
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
from basketworld.utils.action_resolution import (
    IllegalActionStrategy,
    get_policy_action_probabilities,
    resolve_illegal_actions,
)
from basketworld.utils.mlflow_params import (
    get_mlflow_params,
    get_mlflow_phi_shaping_params,
)


# --- Globals ---
# This is a simple way to manage state for a single-user demo.
# For a multi-user app, you would need a more robust session management system.
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


game_state = GameState()


# --- API Models ---
class InitGameRequest(BaseModel):
    run_id: str
    user_team_name: str  # "OFFENSE" or "DEFENSE"
    unified_policy_name: str | None = None
    opponent_unified_policy_name: str | None = None
    # Optional overrides
    spawn_distance: int | None = None
    allow_dunks: bool | None = None
    dunk_pct: float | None = None


class ListPoliciesRequest(BaseModel):
    run_id: str


class ActionRequest(BaseModel):
    actions: dict[
        str, int
    ]  # JSON keys are strings, so we accept strings and convert later.
    deterministic: bool | None = None


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
    mlflow.set_tracking_uri("http://localhost:5000")
    client = mlflow.tracking.MlflowClient()
    try:
        unified_paths = list_policies_from_run(client, request.run_id)
        # return only basenames to frontend
        return {
            "unified": [os.path.basename(p) for p in unified_paths],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list policies: {e}")


@app.post("/api/init_game")
async def init_game(request: InitGameRequest):
    """Initializes a new game from an MLflow run.

    We persist the global GameState instance so that shot logs can continue across
    episodes as long as the loaded policies do not change. If policies change, we
    reset the shot log.
    """
    global game_state

    mlflow.set_tracking_uri("http://localhost:5000")
    client = mlflow.tracking.MlflowClient()

    try:
        required, optional = get_mlflow_params(client, request.run_id)

        # Fetch run metadata
        run = client.get_run(request.run_id)
        run_name = run.data.tags.get("mlflow.runName") if run and run.data else None

        # Load phi shaping parameters from MLflow
        # These will be used for Rewards tab calculations (independent of Phi Shaping tab)
        mlflow_phi_params = get_mlflow_phi_shaping_params(client, request.run_id)

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

        # Reset per-episode stats/logs and set policy keys for UI on every init
        game_state.shot_log = []
        game_state.unified_policy_key = unified_key
        game_state.offense_policy_key = None
        game_state.defense_policy_key = None
        game_state.opponent_unified_policy_key = (
            os.path.basename(opponent_unified_path) if opponent_unified_path else None
        )

        # (Re)load policies from the selected paths
        game_state.unified_policy = PPO.load(unified_path)
        game_state.offense_policy = None
        game_state.defense_policy = (
            PPO.load(opponent_unified_path) if opponent_unified_path else None
        )

        game_state.env = basketworld.HexagonBasketballEnv(
            **required,
            **optional,
            render_mode="rgb_array",
        )
        game_state.obs, _ = game_state.env.reset()

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
        except Exception:
            pass

        # Record initial state for replay (manual or self-play) with policy probs
        initial_state = get_full_game_state(include_policy_probs=True)
        game_state.episode_states.append(initial_state)

        # Record initial phi values (step 0) by computing EP for all players
        try:
            env = game_state.env
            ep_by_player = []
            for pid in range(env.n_players):
                pos = env.positions[pid]
                dist = env._hex_distance(pos, env.basket_position)
                shot_value = (
                    2.0
                    if (getattr(env, "allow_dunks", True) and dist == 0)
                    else (3.0 if dist >= env.three_point_distance else 2.0)
                )
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


@app.post("/api/step")
def take_step(request: ActionRequest):
    """Takes a single step in the environment."""
    if not game_state.env:
        raise HTTPException(status_code=400, detail="Game not initialized.")

    # Get AI actions (unified-only)
    ai_obs = game_state.obs
    # Default to deterministic=True if not provided to preserve previous behavior
    pred_deterministic = (
        True if request.deterministic is None else bool(request.deterministic)
    )
    full_action_ai, _ = game_state.unified_policy.predict(
        ai_obs, deterministic=pred_deterministic
    )
    full_action_ai_opponent = None
    if game_state.defense_policy is not None:
        try:
            # Flip role flag for opponent so it conditions on opposite role
            opp_obs = ai_obs
            try:
                opp_obs = {
                    "obs": np.copy(ai_obs["obs"]),
                    "action_mask": ai_obs["action_mask"],
                    "role_flag": np.copy(ai_obs.get("role_flag")),
                    "skills": np.copy(ai_obs.get("skills")),
                }
                if opp_obs.get("role_flag") is not None:
                    opp_obs["role_flag"] = 1.0 - opp_obs["role_flag"]
            except Exception:
                opp_obs = ai_obs
            full_action_ai_opponent, _ = game_state.defense_policy.predict(
                opp_obs, deterministic=True
            )
        except Exception:
            full_action_ai_opponent = None

    # Prepare action masks and probability vectors
    action_mask = ai_obs["action_mask"]
    unified_probs = get_policy_action_probabilities(game_state.unified_policy, ai_obs)
    opponent_probs = (
        get_policy_action_probabilities(game_state.defense_policy, opp_obs)
        if full_action_ai_opponent is not None
        else None
    )
    player_team_strategy = (
        IllegalActionStrategy.BEST_PROB
        if pred_deterministic
        else IllegalActionStrategy.SAMPLE_PROB
    )
    resolved_unified = resolve_illegal_actions(
        np.array(full_action_ai),
        action_mask,
        player_team_strategy,
        pred_deterministic,
        unified_probs,
    )
    resolved_opponent = (
        resolve_illegal_actions(
            np.array(full_action_ai_opponent),
            action_mask,
            IllegalActionStrategy.BEST_PROB,
            True,
            opponent_probs,
        )
        if full_action_ai_opponent is not None
        else None
    )

    # Combine user and AI actions
    full_action = np.zeros(game_state.env.n_players, dtype=int)

    for i in range(game_state.env.n_players):
        is_user_player = (
            i in game_state.env.offense_ids and game_state.user_team == Team.OFFENSE
        ) or (i in game_state.env.defense_ids and game_state.user_team == Team.DEFENSE)

        if is_user_player:
            # Action comes from the user request
            # Convert player_id (int) to string for dict lookup
            proposed = request.actions.get(str(i), 0)
            # Enforce action mask for user as well
            if action_mask[i][proposed] == 1:
                full_action[i] = proposed
            else:
                full_action[i] = 0
        else:
            is_user_offense = game_state.user_team == Team.OFFENSE
            use_opponent = (
                (is_user_offense and i in game_state.env.defense_ids)
                or ((not is_user_offense) and i in game_state.env.offense_ids)
            ) and (resolved_opponent is not None)

            if use_opponent:
                full_action[i] = int(resolved_opponent[i])
            else:
                full_action[i] = int(resolved_unified[i])

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
            is_three = bool(dist_at_shot >= game_state.env.three_point_distance)
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
            if shot_result.get("success"):
                # Determine if it was a 2PT or 3PT based on distance
                shooter_pos = game_state.env.positions[player_id]
                dist_to_basket = game_state.env._hex_distance(
                    shooter_pos, game_state.env.basket_position
                )
                if dist_to_basket >= game_state.env.three_point_distance:
                    offense_reasons.append("Made 3PT")
                else:
                    offense_reasons.append("Made 2PT")
                defense_reasons.append("Opp Made Shot")
            else:
                offense_reasons.append("Missed Shot")
                defense_reasons.append("Opp Missed")

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
                shot_value = (
                    2.0
                    if (getattr(env, "allow_dunks", False) and dist == 0)
                    else (3.0 if dist >= env.three_point_distance else 2.0)
                )
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
    except Exception:
        pass
    # Record resulting state for replay with policy probs
    try:
        game_state.episode_states.append(get_full_game_state(include_policy_probs=True))
    except Exception:
        pass
    # End of self-play: mark inactive when episode is done
    if game_state.self_play_active and done:
        game_state.self_play_active = False

    return {
        "status": "success",
        "state": get_full_game_state(),
        "step_rewards": {
            "offense": float(step_rewards["offense"]),
            "defense": float(step_rewards["defense"]),
        },
        "episode_rewards": {
            "offense": float(game_state.episode_rewards["offense"]),
            "defense": float(game_state.episode_rewards["defense"]),
        },
    }


@app.get("/api/phi_params")
def get_phi_params():
    if not game_state.env:
        raise HTTPException(status_code=400, detail="Game not initialized")
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

    # Reset environment using overrides to avoid RNG draws during reset
    options = {
        "initial_positions": init_positions,
        "ball_holder": init_ball_holder,
        "shot_clock": init_shot_clock,
    }
    game_state.obs, _ = game_state.env.reset(seed=episode_seed, options=options)

    # Capture initial frame
    try:
        frame = game_state.env.render()
        if frame is not None:
            game_state.frames.append(frame)
    except Exception:
        pass

    return {"status": "success", "state": get_full_game_state(), "seed": episode_seed}


class EvaluationRequest(BaseModel):
    num_episodes: int = 100
    deterministic: bool = True


@app.post("/api/run_evaluation")
def run_evaluation(request: EvaluationRequest):
    """Run N episodes of self-play for evaluation purposes.

    Returns final state of each episode for stats tracking.
    Runs in deterministic mode using the unified policy.

    Note: Uses the environment initialized in /api/init_game which loads
    all parameters (including min_shot_clock and shot_clock) from MLflow.
    """
    if not game_state.env:
        raise HTTPException(
            status_code=400, detail="Game not initialized. Call /api/init_game first."
        )

    if not game_state.unified_policy:
        raise HTTPException(
            status_code=400, detail="Unified policy required for evaluation."
        )

    num_episodes = max(1, min(request.num_episodes, 10000))  # Cap at 10000 for safety
    deterministic = request.deterministic

    episode_results = []

    # Track entropy for diagnosis (to compare with training entropy)
    episode_entropies = []

    # Log shot clock configuration before evaluation
    # These values come from the environment initialized with MLflow parameters
    print(f"[Evaluation] Starting {num_episodes} episodes")
    print(f"[Evaluation] Configuration:")
    print(f"  - Deterministic policy: {deterministic}")
    print(f"  - Using opponent policy: {game_state.defense_policy is not None}")
    print(f"  - shot_clock (max): {game_state.env.shot_clock_steps}")
    print(f"  - min_shot_clock: {game_state.env.min_shot_clock}")
    print(
        f"  - Each episode starts with random shot clock in range: [{game_state.env.min_shot_clock}, {game_state.env.shot_clock_steps}] steps"
    )

    for ep_idx in range(num_episodes):
        # Reset environment for new episode
        episode_seed = int(np.random.randint(0, 2**31 - 1))
        obs, _ = game_state.env.reset(seed=episode_seed)
        game_state.obs = obs

        # Log shot clock for first few episodes to verify randomization
        if ep_idx < 5:
            print(
                f"  Episode {ep_idx + 1} starting shot clock: {game_state.env.shot_clock}"
            )

        done = False
        step_count = 0
        episode_rewards = {"offense": 0.0, "defense": 0.0}
        episode_entropy_sum = 0.0
        episode_entropy_count = 0

        # Run episode until done
        while not done and step_count < 1000:  # Safety limit
            # Get AI actions for both teams
            full_action_ai, _ = game_state.unified_policy.predict(
                obs, deterministic=deterministic
            )

            # Calculate policy entropy for diagnostics
            try:
                obs_tensor = game_state.unified_policy.policy.obs_to_tensor(obs)[0]
                distributions = game_state.unified_policy.policy.get_distribution(
                    obs_tensor
                )
                # Calculate entropy across all players
                for dist in distributions.distribution:
                    entropy = dist.entropy().mean().item()
                    episode_entropy_sum += entropy
                    episode_entropy_count += 1
            except Exception:
                pass

            opponent_obs = obs
            if game_state.defense_policy is not None:
                try:
                    # Flip role flag for opponent
                    opponent_obs = {
                        "obs": np.copy(obs["obs"]),
                        "action_mask": obs["action_mask"],
                        "role_flag": np.copy(obs.get("role_flag")),
                        "skills": np.copy(obs.get("skills")),
                    }
                    if opponent_obs.get("role_flag") is not None:
                        opponent_obs["role_flag"] = 1.0 - opponent_obs["role_flag"]
                    full_action_ai_opponent, _ = game_state.defense_policy.predict(
                        opponent_obs, deterministic=True
                    )
                except Exception:
                    full_action_ai_opponent = None
            else:
                full_action_ai_opponent = None

            # Resolve actions using the same logic as step endpoint
            action_mask = obs["action_mask"]
            unified_probs = get_policy_action_probabilities(
                game_state.unified_policy, obs
            )
            opponent_probs = (
                get_policy_action_probabilities(game_state.defense_policy, opponent_obs)
                if full_action_ai_opponent is not None
                else None
            )

            player_team_strategy = (
                IllegalActionStrategy.BEST_PROB
                if deterministic
                else IllegalActionStrategy.SAMPLE_PROB
            )

            resolved_unified = resolve_illegal_actions(
                np.array(full_action_ai),
                action_mask,
                player_team_strategy,
                deterministic,
                unified_probs,
            )

            if full_action_ai_opponent is not None:
                resolved_opponent = resolve_illegal_actions(
                    np.array(full_action_ai_opponent),
                    action_mask,
                    IllegalActionStrategy.BEST_PROB,
                    True,
                    opponent_probs,
                )
            else:
                resolved_opponent = np.array(full_action_ai)

            # Combine actions based on team roles
            final_action = np.array(resolved_unified, dtype=np.int32)
            for idx in game_state.env.defense_ids:
                final_action[idx] = resolved_opponent[idx]

            # Execute step
            obs, reward, terminated, truncated, info = game_state.env.step(final_action)
            done = terminated or truncated
            step_count += 1

            # Track rewards by team
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

        # Capture final state
        game_state.obs = obs
        final_state = get_full_game_state()

        # Calculate average entropy for this episode
        avg_entropy = (
            episode_entropy_sum / episode_entropy_count
            if episode_entropy_count > 0
            else 0.0
        )
        episode_entropies.append(avg_entropy)

        episode_results.append(
            {
                "episode": ep_idx + 1,
                "final_state": final_state,
                "steps": step_count,
                "episode_rewards": episode_rewards,
            }
        )

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

        # Log average policy entropy (to compare with training)
        if episode_entropies:
            avg_entropy = sum(episode_entropies) / len(episode_entropies)
            print(f"[Evaluation Complete] Policy entropy:")
            print(f"  - Average entropy: {avg_entropy:.3f}")
            print(
                f"  - NOTE: Compare with training 'train/entropy_loss' metric in MLflow"
            )
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

    return {
        "status": "success",
        "num_episodes": len(episode_results),
        "results": episode_results,
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
    }
    obs, _ = game_state.env.reset(seed=game_state.replay_seed, options=options)

    states = [get_full_game_state()]
    for action in game_state.actions_log:
        obs, _, _, _, _ = game_state.env.step(action)
        try:
            frame = game_state.env.render()
            if frame is not None:
                game_state.frames.append(frame)
        except Exception:
            pass
        states.append(get_full_game_state())

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


@app.post("/api/save_episode")
def save_episode():
    """Saves the recorded episode frames to a GIF in ./episodes and returns the file path."""
    if not game_state.frames:
        raise HTTPException(status_code=400, detail="No episode frames to save.")
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
            is_three = distance >= game_state.env.three_point_distance
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


@app.get("/api/action_values/{player_id}")
def get_action_values(player_id: int):
    """
    Calculates the Q-value (state-action value) for all possible actions for a given player.
    This is done via a one-step lookahead simulation.
    """
    if not game_state.env or game_state.obs is None:
        raise HTTPException(status_code=400, detail="Game not initialized.")

    print(f"\n[API] Received request for Q-values for player {player_id}")
    action_values = {}

    # Unified-only
    value_policy = game_state.unified_policy
    gamma = value_policy.gamma

    # Get the list of all possible action names from the enum
    possible_actions = [action.name for action in ActionType]

    for action_name in possible_actions:
        action_id = ActionType[action_name].value

        # --- Simulate one step forward ---
        temp_env = copy.deepcopy(game_state.env)

        # Construct the full action array for the simulation
        # The target player takes the action, others act based on their policy
        sim_action = np.zeros(temp_env.n_players, dtype=int)

        full_actions, _ = game_state.unified_policy.predict(
            game_state.obs, deterministic=True
        )

        for i in range(temp_env.n_players):
            if i == player_id:
                sim_action[i] = action_id
            else:
                sim_action[i] = full_actions[i]

        # Step the temporary environment
        next_obs, reward, _, _, _ = temp_env.step(sim_action)

        # Get the value of the resulting state
        # Convert the next observation to a tensor for the policy
        next_obs_tensor, _ = value_policy.policy.obs_to_tensor(next_obs)
        with torch.no_grad():
            next_value = value_policy.policy.predict_values(next_obs_tensor)

        # Calculate the Q-value
        # We need the specific reward for the team being evaluated
        team_reward = reward[player_id]
        q_value = team_reward + gamma * next_value.item()

        print(
            f"  - Simulating '{action_name}': "
            f"Immediate Reward = {team_reward:.3f}, "
            f"Next State Value = {next_value.item():.3f}, "
            f"Q-Value = {q_value:.3f}"
        )

        action_values[action_name] = q_value

    print(f"[API] Sending action values for player {player_id}:")
    import json

    print(json.dumps(action_values, indent=2))
    return jsonable_encoder(action_values)


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
        print(
            f"[SHOT_PROB_DEBUG] Player {player_id} at {player_pos}, basket at {basket_pos}, distance: {distance}"
        )
        print(
            f"[SHOT_PROB_DEBUG] Environment params: layup_pct={game_state.env.layup_pct}, three_pt_pct={game_state.env.three_pt_pct}, three_point_distance={game_state.env.three_point_distance}"
        )
        print(
            f"[SHOT_PROB_DEBUG] Shot pressure params: enabled={game_state.env.shot_pressure_enabled}, max={game_state.env.shot_pressure_max}, lambda={game_state.env.shot_pressure_lambda}"
        )

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

        print(f"[SHOT_PROB_DEBUG] Base probability before pressure: {base_prob:.3f}")

        # Calculate pressure-adjusted probability (for logging/diagnostics)
        final_prob = game_state.env._calculate_shot_probability(player_id, distance)
        print(
            f"[SHOT_PROB_DEBUG] Final shot probability after pressure: {final_prob:.3f}"
        )

        return {
            "player_id": player_id,
            "shot_probability": float(base_prob),
            "shot_probability_final": float(final_prob),
            "distance": int(distance),
        }
    except Exception as e:
        return {"player_id": player_id, "shot_probability": 0.0, "error": str(e)}


@app.get("/api/rewards")
def get_rewards():
    """Get the current reward history and episode totals."""
    import sys

    print("=" * 80, flush=True)
    print("[DEBUG] /api/rewards endpoint called", flush=True)
    sys.stdout.flush()

    # Calculate phi shaping rewards using MLflow parameters (if available)
    # This is separate from the Phi Shaping tab which is for experimentation
    mlflow_phi_params = game_state.mlflow_phi_shaping_params
    print(f"[DEBUG] mlflow_phi_params = {mlflow_phi_params}", flush=True)
    sys.stdout.flush()

    # Calculate MLflow-based phi shaping rewards
    mlflow_phi_r_shape_values = []
    if mlflow_phi_params and mlflow_phi_params.get("enable_phi_shaping", False):
        beta = mlflow_phi_params.get("phi_beta", 0.0)
        gamma = mlflow_phi_params.get("reward_shaping_gamma", 1.0)

        print(
            f"[MLflow Phi] Calculating rewards with beta={beta}, gamma={gamma}, mode={mlflow_phi_params.get('phi_aggregation_mode')}, history_length={len(game_state.reward_history)}"
        )

        # Calculate phi for initial state (step 0) from phi_log if available
        phi_prev = 0.0
        if game_state.phi_log and len(game_state.phi_log) > 0:
            initial_entry = game_state.phi_log[0]
            if initial_entry.get("step") == 0:
                # Recalculate initial phi using MLflow params
                initial_ep = initial_entry.get("ep_by_player", [])
                initial_ball = initial_entry.get("ball_handler", -1)
                initial_offense = initial_entry.get("offense_ids", [])
                print(
                    f"[MLflow Phi] Initial state: ep_by_player={initial_ep}, ball={initial_ball}, offense={initial_offense}"
                )
                if initial_ep and initial_ball >= 0 and initial_offense:
                    phi_prev = calculate_phi_from_ep_data(
                        initial_ep, initial_ball, initial_offense, mlflow_phi_params
                    )
                    print(
                        f"[MLflow Phi] Initial state phi_prev = {phi_prev} (expected from Phi Shaping tab)"
                    )
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
            "turnover_penalty": float(getattr(env, "turnover_penalty", 0.0)),
            "made_shot_reward_inside": float(
                getattr(env, "made_shot_reward_inside", 0.0)
            ),
            "made_shot_reward_three": float(
                getattr(env, "made_shot_reward_three", 0.0)
            ),
            "missed_shot_penalty": float(getattr(env, "missed_shot_penalty", 0.0)),
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
        policy = game_state.unified_policy
        # Convert observation to the format the policy expects
        obs_tensor = policy.policy.obs_to_tensor(game_state.obs)[0]
        # Get the distribution over actions from the policy object
        distributions = policy.policy.get_distribution(obs_tensor)
        # Extract raw probabilities for each player
        raw_probs = [
            dist.probs.detach().cpu().numpy().squeeze()
            for dist in distributions.distribution
        ]
        # Apply action mask so illegal actions have probability 0, then renormalize.
        action_mask = game_state.obs["action_mask"]  # shape (n_players, n_actions)
        probs_list = []
        for pid, probs in enumerate(raw_probs):
            masked = probs * action_mask[pid]
            total = masked.sum()
            if total > 0:
                masked = masked / total
            probs_list.append(masked.tolist())
        # Return as a dictionary mapping player_id to their list of probabilities
        response = {
            player_id: probs
            for player_id, probs in enumerate(probs_list)
            if player_id in game_state.env.offense_ids
            or player_id in game_state.env.defense_ids
        }
        return response
    except Exception as e:
        print(f"[compute_policy_probabilities] Error: {e}")
        return None


def get_full_game_state(include_policy_probs=False):
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

    # Calculate EP (expected points) for all players
    ep_by_player = []
    try:
        env = game_state.env
        for pid in range(env.n_players):
            pos = env.positions[pid]
            dist = env._hex_distance(pos, env.basket_position)
            shot_value = (
                2.0
                if (getattr(env, "allow_dunks", True) and dist == 0)
                else (3.0 if dist >= env.three_point_distance else 2.0)
            )
            p = float(env._calculate_shot_probability(pid, dist))
            ep = float(shot_value * p)
            ep_by_player.append(ep)
    except Exception:
        # If EP calculation fails, use empty list
        ep_by_player = []

    state = {
        "players_per_side": int(getattr(game_state.env, "players_per_side", 3)),
        "players": int(getattr(game_state.env, "players_per_side", 3)),
        "positions": positions_py,
        "ball_holder": ball_holder_py,
        "ball_handler_shot_probability": ball_handler_shot_prob,
        "shot_clock": int(game_state.env.shot_clock),
        "min_shot_clock": int(getattr(game_state.env, "min_shot_clock", 10)),
        "user_team_name": game_state.user_team.name,
        "done": game_state.env.episode_ended,
        "training_team": (
            getattr(game_state.env, "training_team", None).name
            if getattr(game_state.env, "training_team", None)
            else None
        ),
        "action_space": {action.name: action.value for action in ActionType},
        "action_mask": action_mask_py,
        "last_action_results": last_action_results_py,
        "offense_ids": game_state.env.offense_ids,
        "defense_ids": game_state.env.defense_ids,
        "basket_position": basket_pos_py,
        "court_width": game_state.env.court_width,
        "court_height": game_state.env.court_height,
        "three_point_distance": int(getattr(game_state.env, "three_point_distance", 4)),
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
        "steal_chance": float(getattr(game_state.env, "steal_chance", 0.05)),
        "spawn_distance": int(getattr(game_state.env, "spawn_distance", 3)),
        "max_spawn_distance": (
            int(getattr(game_state.env, "max_spawn_distance", None))
            if getattr(game_state.env, "max_spawn_distance", None) is not None
            else None
        ),
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
        "illegal_defense_enabled": bool(
            getattr(game_state.env, "illegal_defense_enabled", False)
        ),
        "illegal_defense_max_steps": int(
            getattr(game_state.env, "illegal_defense_max_steps", 3)
        ),
        # Pass parameters
        "pass_arc_degrees": float(getattr(game_state.env, "pass_arc_degrees", 60.0)),
        "pass_oob_turnover_prob": float(
            getattr(game_state.env, "pass_oob_turnover_prob", 1.0)
        ),
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
        # Expected points for all players (indexed by player ID)
        "ep_by_player": ep_by_player,
    }

    # Optionally include policy probabilities
    if include_policy_probs:
        policy_probs = compute_policy_probabilities()
        if policy_probs is not None:
            state["policy_probabilities"] = policy_probs

    return state
