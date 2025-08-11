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

game_state = GameState()

# --- API Models ---
class InitGameRequest(BaseModel):
    run_id: str
    user_team_name: str  # "OFFENSE" or "DEFENSE"
    offense_policy_name: str | None = None  # optional specific policy filename
    defense_policy_name: str | None = None


class ListPoliciesRequest(BaseModel):
    run_id: str

class ActionRequest(BaseModel):
    actions: dict[str, int] # JSON keys are strings, so we accept strings and convert later.

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
def init_game(request: InitGameRequest):
    """Initializes a new game from an MLflow run."""
    global game_state
    game_state = GameState() # Reset state

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

        print(
            f"[init_game] Using params: grid={grid_size}, players={players}, shot_clock={shot_clock}, "
            f"three_point_distance={three_point_distance}, layup_pct={layup_pct}, three_pt_pct={three_pt_pct}"
        )

        # Download selected or latest policies
        game_state.offense_policy = PPO.load(
            get_policy_path(client, request.run_id, request.offense_policy_name, "offense")
        )
        game_state.defense_policy = PPO.load(
            get_policy_path(client, request.run_id, request.defense_policy_name, "defense")
        )

        game_state.env = basketworld.HexagonBasketballEnv(
            grid_size=grid_size,
            players_per_side=players,
            shot_clock_steps=shot_clock,
            render_mode="rgb_array",  # enable frame rendering
            three_point_distance=three_point_distance,
            layup_pct=layup_pct,
            three_pt_pct=three_pt_pct,
        )
        game_state.obs, _ = game_state.env.reset()

        # Capture the initial frame
        try:
            frame = game_state.env.render()
            game_state.frames.append(frame)
        except Exception:
            pass
        game_state.user_team = Team[request.user_team_name.upper()]

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
    offense_action_raw, _ = game_state.offense_policy.predict(ai_obs, deterministic=True)
    defense_action_raw, _ = game_state.defense_policy.predict(ai_obs, deterministic=True)

    # Combine user and AI actions
    full_action = np.zeros(game_state.env.n_players, dtype=int)
    action_mask = ai_obs['action_mask']

    for i in range(game_state.env.n_players):
        is_user_player = (i in game_state.env.offense_ids and game_state.user_team == Team.OFFENSE) or \
                         (i in game_state.env.defense_ids and game_state.user_team == Team.DEFENSE)

        if is_user_player:
            # Action comes from the user request
            # Convert player_id (int) to string for dict lookup
            full_action[i] = request.actions.get(str(i), 0) # Default to NOOP if not provided
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
                print(f"[AI] Player {i} tried illegal action {ActionType(predicted_action).name}, taking NOOP instead.")
                full_action[i] = 0

    game_state.obs, _, done, _, _ = game_state.env.step(full_action)

    # Capture frame after step
    try:
        frame = game_state.env.render()
        game_state.frames.append(frame)
    except Exception:
        pass
    
    return {"status": "success", "state": get_full_game_state()}

@app.post("/api/save_episode")
def save_episode():
    """Saves the recorded episode frames to a GIF in ./episodes and returns the file path."""
    if not game_state.frames:
        raise HTTPException(status_code=400, detail="No episode frames to save.")

    os.makedirs("episodes", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join("episodes", f"episode_{timestamp}.gif")

    # Write frames to GIF
    try:
        imageio.mimsave(file_path, game_state.frames, fps=2, loop=0)
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
    } 