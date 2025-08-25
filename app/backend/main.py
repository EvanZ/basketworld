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

# --- Globals ---
# This is a simple way to manage state for a single-user demo.
# For a multi-user app, you would need a more robust session management system.
class GameState:
    def __init__(self):
        self.env = None
        self.offense_policy = None
        self.defense_policy = None
        self.user_team: Team = None
        self.obs = None
        self.frames = []  # List of RGB frames for the current episode
        self.reward_history = []  # Track rewards for each step
        self.episode_rewards = {"offense": 0.0, "defense": 0.0}  # Running totals
        self.shot_log = []  # Per-step shot attempts with probability and result
        # Track which policies are currently loaded so we can persist logs across episodes
        self.offense_policy_key: str | None = None
        self.defense_policy_key: str | None = None

game_state = GameState()

# --- API Models ---
class InitGameRequest(BaseModel):
    run_id: str
    user_team_name: str  # "OFFENSE" or "DEFENSE"
    offense_policy_name: str | None = None  # optional specific policy filename
    defense_policy_name: str | None = None
    # Optional overrides
    spawn_distance: int | None = None
    allow_dunks: bool | None = None
    dunk_pct: float | None = None


class ListPoliciesRequest(BaseModel):
    run_id: str

class ActionRequest(BaseModel):
    actions: dict[str, int] # JSON keys are strings, so we accept strings and convert later.
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
    """Return sorted lists of offense and defense policy artifact paths for a run."""
    artifacts = client.list_artifacts(run_id, "models")
    offense = [f.path for f in artifacts if f.path.endswith(".zip") and "offense" in f.path]
    defense = [f.path for f in artifacts if f.path.endswith(".zip") and "defense" in f.path]

    # sort by number embedded at end _<n>.zip if present
    def sort_key(p):
        import re
        m = re.search(r"_(\d+)\.zip$", p)
        return int(m.group(1)) if m else 0
    offense.sort(key=sort_key)
    defense.sort(key=sort_key)
    return offense, defense


def get_policy_path(client, run_id, policy_name: str | None, team_prefix: str):
    """Return artifact path for given policy (downloaded locally). If name None, use latest."""
    # Use a persistent cache directory to avoid deletion before PPO.load
    cache_dir = os.path.join("episodes", "_policy_cache")
    os.makedirs(cache_dir, exist_ok=True)

    offense_paths, defense_paths = list_policies_from_run(client, run_id)
    choices = offense_paths if team_prefix == "offense" else defense_paths
    if not choices:
        raise HTTPException(status_code=404, detail=f"No {team_prefix} policy artifacts found.")

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
        raise HTTPException(status_code=404, detail="No model artifacts found in the specified run.")

    latest_offense_path = max([f.path for f in artifacts if "offense" in f.path], key=lambda p: int(re.search(r'_(\d+)\.zip', p).group(1)))
    latest_defense_path = max([f.path for f in artifacts if "defense" in f.path], key=lambda p: int(re.search(r'_(\d+)\.zip', p).group(1)))
    
    offense_local_path = client.download_artifacts(run_id, latest_offense_path, tmpdir)
    defense_local_path = client.download_artifacts(run_id, latest_defense_path, tmpdir)

    return offense_local_path, defense_local_path

@app.post("/api/list_policies")
def list_policies(request: ListPoliciesRequest):
    """Return available offense and defense policy filenames for a run."""
    mlflow.set_tracking_uri("http://localhost:5000")
    client = mlflow.tracking.MlflowClient()
    try:
        offense_paths, defense_paths = list_policies_from_run(client, request.run_id)
        # return only basenames to frontend
        return {
            "offense": [os.path.basename(p) for p in offense_paths],
            "defense": [os.path.basename(p) for p in defense_paths],
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
        run = client.get_run(request.run_id)
        params = run.data.params
        grid_size = int(params["grid_size"])
        players = int(params["players"])
        shot_clock = int(params["shot_clock"])

        def get_param(params_dict, names, cast, default):
            for n in names:
                if n in params_dict and params_dict[n] != "":
                    try:
                        return cast(params_dict[n])
                    except Exception:
                        pass
            return default

        # Optional params (added later); try multiple name variants, fall back to defaults
        three_point_distance = get_param(
            params,
            [
                "three_point_distance",
                "three-point-distance",
                "three_pt_distance",
                "three-pt-distance",
            ],
            int,
            4,
        )
        layup_pct = get_param(params, ["layup_pct", "layup-pct"], float, 0.60)
        three_pt_pct = get_param(params, ["three_pt_pct", "three-pt-pct"], float, 0.37)
        spawn_distance = get_param(params, ["spawn_distance", "spawn-distance"], int, 3)
        # Dunk params (optional)
        allow_dunks = get_param(params, ["allow_dunks", "allow-dunks"], lambda v: str(v).lower() in ["1","true","yes","y","t"], False)
        dunk_pct = get_param(params, ["dunk_pct", "dunk-pct"], float, 0.90)
        # Shot pressure params (optional)
        shot_pressure_enabled = get_param(params, ["shot_pressure_enabled", "shot-pressure-enabled"], lambda v: str(v).lower() in ["1","true","yes","y","t"], True)
        shot_pressure_max = get_param(params, ["shot_pressure_max", "shot-pressure-max"], float, 0.5)
        shot_pressure_lambda = get_param(params, ["shot_pressure_lambda", "shot-pressure-lambda"], float, 1.0)
        shot_pressure_arc_degrees = get_param(params, ["shot_pressure_arc_degrees", "shot-pressure-arc-degrees"], float, 60.0)
        # Defender pressure params (optional)
        defender_pressure_distance = get_param(params, ["defender_pressure_distance", "defender-pressure-distance"], int, 1)
        defender_pressure_turnover_chance = get_param(params, ["defender_pressure_turnover_chance", "defender-pressure-turnover-chance"], float, 0.05)
        # Movement mask (optional)
        mask_occupied_moves = get_param(params, ["mask_occupied_moves", "mask-occupied-moves"], lambda v: str(v).lower() in ["1","true","yes","y","t"], False)

        # Observation controls (optional)
        use_egocentric_obs = get_param(params, ["use_egocentric_obs", "use-egocentric-obs"], lambda v: str(v).lower() in ["1","true","yes","y","t"], True)
        egocentric_rotate_to_hoop = get_param(params, ["egocentric_rotate_to_hoop", "egocentric-rotate-to-hoop"], lambda v: str(v).lower() in ["1","true","yes","y","t"], True)
        include_hoop_vector = get_param(params, ["include_hoop_vector", "include-hoop-vector"], lambda v: str(v).lower() in ["1","true","yes","y","t"], True)
        normalize_obs = get_param(params, ["normalize_obs", "normalize-obs"], lambda v: str(v).lower() in ["1","true","yes","y","t"], True)
        
        # Apply request overrides if provided
        if request.spawn_distance is not None:
            spawn_distance = int(request.spawn_distance)
        if request.allow_dunks is not None:
            allow_dunks = bool(request.allow_dunks)
        if request.dunk_pct is not None:
            dunk_pct = float(request.dunk_pct)

        print(
            f"[init_game] Using params: grid={grid_size}, players={players}, shot_clock={shot_clock}, "
            f"three_point_distance={three_point_distance}, layup_pct={layup_pct}, three_pt_pct={three_pt_pct}, "
            f"shot_pressure_enabled={shot_pressure_enabled}, shot_pressure_max={shot_pressure_max}, "
            f"shot_pressure_lambda={shot_pressure_lambda}, shot_pressure_arc_degrees={shot_pressure_arc_degrees}, "
            f"mask_occupied_moves={mask_occupied_moves}, allow_dunks={allow_dunks}, dunk_pct={dunk_pct}, spawn_distance={spawn_distance}"
        )

        # Download selected or latest policies and determine keys
        offense_path = get_policy_path(client, request.run_id, request.offense_policy_name, "offense")
        defense_path = get_policy_path(client, request.run_id, request.defense_policy_name, "defense")
        offense_key = os.path.basename(offense_path)
        defense_key = os.path.basename(defense_path)

        # Reset shot log only if policies changed
        if game_state.offense_policy_key != offense_key or game_state.defense_policy_key != defense_key:
            game_state.shot_log = []
            game_state.offense_policy_key = offense_key
            game_state.defense_policy_key = defense_key

        # (Re)load policies from the selected paths
        game_state.offense_policy = PPO.load(offense_path)
        game_state.defense_policy = PPO.load(defense_path)

        game_state.env = basketworld.HexagonBasketballEnv(
            grid_size=grid_size,
            players_per_side=players,
            shot_clock_steps=shot_clock,
            render_mode="rgb_array",  # enable frame rendering
            three_point_distance=three_point_distance,
            layup_pct=layup_pct,
            three_pt_pct=three_pt_pct,
            allow_dunks=allow_dunks,
            dunk_pct=dunk_pct,
            shot_pressure_enabled=shot_pressure_enabled,
            shot_pressure_max=shot_pressure_max,
            shot_pressure_lambda=shot_pressure_lambda,
            shot_pressure_arc_degrees=shot_pressure_arc_degrees,
            spawn_distance=spawn_distance,
            defender_pressure_distance=defender_pressure_distance,
            defender_pressure_turnover_chance=defender_pressure_turnover_chance,
            # Observation controls
            use_egocentric_obs=use_egocentric_obs,
            egocentric_rotate_to_hoop=egocentric_rotate_to_hoop,
            include_hoop_vector=include_hoop_vector,
            normalize_obs=normalize_obs,
            mask_occupied_moves=mask_occupied_moves,
        )
        game_state.obs, _ = game_state.env.reset()

        # Set user team and ensure tracking containers start empty for the episode
        game_state.user_team = Team[request.user_team_name.upper()]
        game_state.frames = []
        game_state.reward_history = []
        game_state.episode_rewards = {"offense": 0.0, "defense": 0.0}

        # Capture and keep the initial frame so saved episodes include the starting court state
        try:
            frame = game_state.env.render()
            game_state.frames.append(frame)
        except Exception:
            pass

        return {"status": "success", "state": get_full_game_state()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize game: {e}")

@app.post("/api/step")
def take_step(request: ActionRequest):
    """Takes a single step in the environment."""
    if not game_state.env:
        raise HTTPException(status_code=400, detail="Game not initialized.")

    # Get AI actions
    ai_obs = game_state.obs
    # Default to deterministic=True if not provided to preserve previous behavior
    pred_deterministic = True if request.deterministic is None else bool(request.deterministic)
    offense_action_raw, _ = game_state.offense_policy.predict(ai_obs, deterministic=pred_deterministic)
    defense_action_raw, _ = game_state.defense_policy.predict(ai_obs, deterministic=pred_deterministic)

    # Combine user and AI actions
    full_action = np.zeros(game_state.env.n_players, dtype=int)
    action_mask = ai_obs['action_mask']

    for i in range(game_state.env.n_players):
        is_user_player = (i in game_state.env.offense_ids and game_state.user_team == Team.OFFENSE) or \
                         (i in game_state.env.defense_ids and game_state.user_team == Team.DEFENSE)

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
            # Action comes from the AI policy
            if i in game_state.env.offense_ids:
                predicted_action = offense_action_raw[i]
            else:
                predicted_action = defense_action_raw[i]
            
            # Enforce action mask and add logging
            if action_mask[i][predicted_action] == 1:
                print(f"[AI] Player {i} taking legal action: {ActionType(predicted_action).name}")
                full_action[i] = predicted_action
            else:
                # Fallback: choose the first legal action instead of NOOP
                legal = np.where(action_mask[i] == 1)[0]
                fallback = int(legal[0]) if len(legal) > 0 else 0
                print(f"[AI] Player {i} tried illegal action {ActionType(predicted_action).name}, taking {ActionType(fallback).name} instead.")
                full_action[i] = fallback

    game_state.obs, rewards, done, _, info = game_state.env.step(full_action)

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
            game_state.shot_log.append({
                "step": int(len(game_state.reward_history) + 1),
                "player_id": int(pid_int),
                "distance": dist_at_shot,
                "probability": float(shot_res.get("probability", 0.0)),
                "success": bool(shot_res.get("success", False)),
                "is_three": is_three,
                "rng": float(shot_res.get("rng", -1.0)),
                "base_probability": float(shot_res.get("base_probability", -1.0)),
                "pressure_multiplier": float(shot_res.get("pressure_multiplier", -1.0)),
            })
    
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
                dist_to_basket = game_state.env._hex_distance(shooter_pos, game_state.env.basket_position)
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
            if not move_result.get("success", True) and move_result.get("reason") == "out_of_bounds":
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

    game_state.reward_history.append({
        "step": len(game_state.reward_history) + 1,
        "offense": float(step_rewards["offense"]),
        "defense": float(step_rewards["defense"]),
        "offense_reason": ", ".join(offense_reasons) if offense_reasons else "None",
        "defense_reason": ", ".join(defense_reasons) if defense_reasons else "None"
    })

    # Capture frame after step
    try:
        frame = game_state.env.render()
        game_state.frames.append(frame)
    except Exception:
        pass
    
    return {
        "status": "success", 
        "state": get_full_game_state(),
        "step_rewards": {
            "offense": float(step_rewards["offense"]),
            "defense": float(step_rewards["defense"])
        },
        "episode_rewards": {
            "offense": float(game_state.episode_rewards["offense"]),
            "defense": float(game_state.episode_rewards["defense"])
        }
    }

@app.get("/api/shot_stats")
def get_shot_stats():
    """Return raw shot log and simple aggregates to compare displayed probabilities vs outcomes."""
    logs = list(game_state.shot_log)
    total = len(logs)
    made = sum(1 for s in logs if s.get("success"))
    avg_prob = (sum(s.get("probability", 0.0) for s in logs) / total) if total else 0.0
    avg_base = (sum(s.get("base_probability", 0.0) for s in logs if s.get("base_probability", -1.0) >= 0) / max(1, sum(1 for s in logs if s.get("base_probability", -1.0) >= 0)))
    avg_pressure_mult = (sum(s.get("pressure_multiplier", 0.0) for s in logs if s.get("pressure_multiplier", -1.0) >= 0) / max(1, sum(1 for s in logs if s.get("pressure_multiplier", -1.0) >= 0)))
    total_three = sum(1 for s in logs if s.get("is_three"))
    made_three = sum(1 for s in logs if s.get("is_three") and s.get("success"))
    avg_prob_three = (
        sum(s.get("probability", 0.0) for s in logs if s.get("is_three")) / total_three
        if total_three else 0.0
    )
    avg_base_three = (
        sum(s.get("base_probability", 0.0) for s in logs if s.get("is_three") and s.get("base_probability", -1.0) >= 0) / max(1, sum(1 for s in logs if s.get("is_three") and s.get("base_probability", -1.0) >= 0))
    )
    avg_pressure_three = (
        sum(s.get("pressure_multiplier", 0.0) for s in logs if s.get("is_three") and s.get("pressure_multiplier", -1.0) >= 0) / max(1, sum(1 for s in logs if s.get("is_three") and s.get("pressure_multiplier", -1.0) >= 0))
    )
    # Group by distance
    by_distance = {}
    for s in logs:
        d = int(s.get("distance", -1))
        if d not in by_distance:
            by_distance[d] = {"attempts": 0, "made": 0, "avg_prob": 0.0, "_prob_sum": 0.0}
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

    os.makedirs("episodes", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Determine outcome label (carry dunk logic like evaluation/ui)
    outcome = "Unknown"
    try:
        ar = game_state.env.last_action_results or {}
        if ar.get("shots"):
            # Take first shot result
            shooter_id_str = list(ar["shots"].keys())[0]
            shot_res = ar["shots"][shooter_id_str]
            distance = int(shot_res.get("distance", 999))
            is_dunk = (distance == 0)
            if is_dunk:
                outcome = "Made Dunk" if shot_res.get("success") else "Missed Dunk"
            else:
                # Determine 2 vs 3 by distance to basket at shot time
                if shot_res.get("success"):
                    outcome = "Made 3pt" if distance >= game_state.env.three_point_distance else "Made 2pt"
                else:
                    outcome = "Missed 3pt" if distance >= game_state.env.three_point_distance else "Missed 2pt"
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

    category = get_outcome_category(outcome)
    file_path = os.path.join("episodes", f"episode_{timestamp}_{category}.gif")

    # Write frames to GIF
    try:
        imageio.mimsave(file_path, game_state.frames, fps=1, loop=0)
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
        policy = game_state.offense_policy if game_state.user_team == Team.OFFENSE else game_state.defense_policy
        
        # Convert observation to the format the policy expects
        obs_tensor = policy.policy.obs_to_tensor(game_state.obs)[0]
        
        # Get the distribution over actions from the policy object
        distributions = policy.policy.get_distribution(obs_tensor)
        
        # Extract raw probabilities for each player
        raw_probs = [dist.probs.detach().cpu().numpy().squeeze() for dist in distributions.distribution]

        # Apply action mask so illegal actions have probability 0, then renormalize.
        action_mask = game_state.obs['action_mask']  # shape (n_players, n_actions)
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
            if player_id in game_state.env.offense_ids or player_id in game_state.env.defense_ids
        }
        return jsonable_encoder(response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get policy probabilities: {e}")


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
    
    # Determine which policy to use for value prediction
    player_is_offense = player_id in game_state.env.offense_ids
    value_policy = game_state.offense_policy if player_is_offense else game_state.defense_policy
    gamma = value_policy.gamma # Get the discount factor

    # Get the list of all possible action names from the enum
    possible_actions = [action.name for action in ActionType]

    for action_name in possible_actions:
        action_id = ActionType[action_name].value
        
        # --- Simulate one step forward ---
        temp_env = copy.deepcopy(game_state.env)
        
        # Construct the full action array for the simulation
        # The target player takes the action, others act based on their policy
        sim_action = np.zeros(temp_env.n_players, dtype=int)
        
        offense_actions, _ = game_state.offense_policy.predict(game_state.obs, deterministic=True)
        defense_actions, _ = game_state.defense_policy.predict(game_state.obs, deterministic=True)

        for i in range(temp_env.n_players):
            if i == player_id:
                sim_action[i] = action_id
            elif i in temp_env.offense_ids:
                sim_action[i] = offense_actions[i]
            else:
                sim_action[i] = defense_actions[i]

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
        
        print(f"  - Simulating '{action_name}': "
              f"Immediate Reward = {team_reward:.3f}, "
              f"Next State Value = {next_value.item():.3f}, "
              f"Q-Value = {q_value:.3f}")

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
        print(f"[SHOT_PROB_DEBUG] Player {player_id} at {player_pos}, basket at {basket_pos}, distance: {distance}")
        print(f"[SHOT_PROB_DEBUG] Environment params: layup_pct={game_state.env.layup_pct}, three_pt_pct={game_state.env.three_pt_pct}, three_point_distance={game_state.env.three_point_distance}")
        print(f"[SHOT_PROB_DEBUG] Shot pressure params: enabled={game_state.env.shot_pressure_enabled}, max={game_state.env.shot_pressure_max}, lambda={game_state.env.shot_pressure_lambda}")
        
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
        print(f"[SHOT_PROB_DEBUG] Final shot probability after pressure: {final_prob:.3f}")
        
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
    # Ensure all values are JSON serializable
    serialized_history = []
    for reward in game_state.reward_history:
        serialized_history.append({
            "step": int(reward["step"]),
            "offense": float(reward["offense"]),
            "defense": float(reward["defense"]),
            "offense_reason": reward.get("offense_reason", "Unknown"),
            "defense_reason": reward.get("defense_reason", "Unknown")
        })
    
    return {
        "reward_history": serialized_history,
        "episode_rewards": {
            "offense": float(game_state.episode_rewards["offense"]),
            "defense": float(game_state.episode_rewards["defense"])
        }
    }

def get_full_game_state():
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
    positions_py = [
        (int(q), int(r)) for q, r in game_state.env.positions
    ]
    ball_holder_py = int(game_state.env.ball_holder) if game_state.env.ball_holder is not None else None
    basket_pos_py = (int(game_state.env.basket_position[0]), int(game_state.env.basket_position[1]))
    action_mask_py = game_state.obs['action_mask'].tolist()
        
    return {
        "players_per_side": int(getattr(game_state.env, "players_per_side", 3)),
        "positions": positions_py,
        "ball_holder": ball_holder_py,
        "shot_clock": int(game_state.env.shot_clock),
        "user_team_name": game_state.user_team.name,
        "done": game_state.env.episode_ended,
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
        "shot_params": getattr(game_state.env, "shot_params", None),
        "defender_pressure_distance": int(getattr(game_state.env, "defender_pressure_distance", 1)),
        "defender_pressure_turnover_chance": float(getattr(game_state.env, "defender_pressure_turnover_chance", 0.05)),
        "spawn_distance": int(getattr(game_state.env, "spawn_distance", 3)),
        "shot_pressure_enabled": bool(getattr(game_state.env, "shot_pressure_enabled", True)),
        "shot_pressure_max": float(getattr(game_state.env, "shot_pressure_max", 0.5)),
        "shot_pressure_lambda": float(getattr(game_state.env, "shot_pressure_lambda", 1.0)),
        "shot_pressure_arc_degrees": float(getattr(game_state.env, "shot_pressure_arc_degrees", 60.0)),
        "mask_occupied_moves": bool(getattr(game_state.env, "mask_occupied_moves", False)),
    } 