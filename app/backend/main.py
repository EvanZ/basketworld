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

game_state = GameState()

# --- API Models ---
class InitGameRequest(BaseModel):
    run_id: str
    user_team_name: str # "OFFENSE" or "DEFENSE"

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

        with tempfile.TemporaryDirectory() as tmpdir:
            offense_path, defense_path = get_latest_policies_from_run(client, request.run_id, tmpdir)
            game_state.offense_policy = PPO.load(offense_path)
            game_state.defense_policy = PPO.load(defense_path)

        game_state.env = basketworld.HexagonBasketballEnv(
            grid_size=grid_size,
            players_per_side=players,
            shot_clock_steps=shot_clock,
        )
        game_state.obs, _ = game_state.env.reset()
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
    
    return {"status": "success", "state": get_full_game_state()}

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
        "court_height": game_state.env.court_height
    } 