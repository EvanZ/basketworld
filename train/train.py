#!/usr/bin/env python3
"""
Main training script for the HexagonBasketballEnv using an alternating
self-play strategy with Proximal Policy Optimization (PPO).

This script implements a self-play loop where two policies, one for offense
and one for defense, are trained against each other in an alternating fashion.
A custom gym.Wrapper is used to manage the opponent's actions during training.
"""
import argparse
import os
from datetime import datetime
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Logger, HumanOutputFormat
from basketworld.utils.mlflow_logger import MLflowWriter
from basketworld.utils.callbacks import RolloutUpdateTimingCallback

from basketworld.utils.evaluation_helpers import get_outcome_category, create_and_log_gif
import imageio
from collections import defaultdict
import re

import mlflow
import sys
import tempfile
import random
from typing import Optional

import basketworld
from basketworld.envs.basketworld_env_v2 import Team

# --- Custom MLflow Callback ---

class MLflowCallback(BaseCallback):
    """
    A custom callback for logging metrics to MLflow.
    This callback logs the mean reward and episode length periodically.
    """
    def __init__(self, team_name: str, offense_policy, defense_policy, log_freq: int = 2048, verbose=0):
        super(MLflowCallback, self).__init__(verbose)
        self.team_name = team_name
        self.log_freq = log_freq
        self.offense_policy = offense_policy
        self.defense_policy = defense_policy

    def _on_step(self) -> bool:
        # Log metrics periodically to avoid performance overhead
        if self.n_calls % self.log_freq == 0:
            # The ep_info_buffer contains info from the last 100 episodes
            if self.model.ep_info_buffer:
                # Calculate the global step by summing timesteps from both policies
                global_step = self.offense_policy.num_timesteps + self.defense_policy.num_timesteps
                
                # Calculate the mean reward and length
                ep_rew_mean = np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
                ep_len_mean = np.mean([ep_info["l"] for ep_info in self.model.ep_info_buffer])
                
                # Log to MLflow
                mlflow.log_metric(f"{self.team_name} Mean Episode Reward", ep_rew_mean, step=global_step)
                mlflow.log_metric(f"{self.team_name} Mean Episode Length", ep_len_mean, step=global_step)
        return True

# --- Custom Reward Wrapper for Multi-Agent Aggregation ---

class RewardAggregationWrapper(gym.Wrapper):
    """
    A wrapper to aggregate multi-agent rewards for the Monitor wrapper.
    It sums the rewards of the team currently being trained.
    """
    def __init__(self, env):
        super().__init__(env)
    
    def step(self, action):
        obs, rewards, done, truncated, info = self.env.step(action)
        
        # Determine which player IDs belong to the training team
        if self.env.unwrapped.training_team == Team.OFFENSE:
            training_player_ids = self.env.unwrapped.offense_ids
        else:
            training_player_ids = self.env.unwrapped.defense_ids
            
        # Sum the rewards for only the players on the training team
        aggregated_reward = sum(rewards[i] for i in training_player_ids)
        
        return obs, aggregated_reward, done, truncated, info

# --- Custom Environment Wrapper for Self-Play ---

class SelfPlayEnvWrapper(gym.Wrapper):
    """
    A wrapper that manages the opponent's policy in a self-play setup.

    When the learning agent takes a step, this wrapper intercepts the action,
    gets an action from the frozen opponent policy, combines them, and passes
    the full action to the underlying environment.
    """
    def __init__(self, env, opponent_policy):
        super().__init__(env)
        self.opponent_policy = opponent_policy
        self._set_team_ids()

    def _set_team_ids(self):
        """Determine which player IDs belong to the training team and opponent."""
        if self.env.unwrapped.training_team == Team.OFFENSE:
            self.training_player_ids = self.env.unwrapped.offense_ids
            self.opponent_player_ids = self.env.unwrapped.defense_ids
        else:
            self.training_player_ids = self.env.unwrapped.defense_ids
            self.opponent_player_ids = self.env.unwrapped.offense_ids

    def reset(self, **kwargs):
        """Reset the environment and store the initial observation."""
        obs, info = self.env.reset(**kwargs)
        self.last_obs = obs
        return obs, info

    def step(self, action):
        """
        Take a step in the environment.
        'action' comes from the learning agent and is for ALL players.
        We replace the opponent's actions with predictions from the frozen policy.
        """
        # Get action from the frozen opponent policy using the last observation
        opponent_action_raw, _ = self.opponent_policy.predict(self.last_obs, deterministic=True)
        action_mask = self.last_obs['action_mask']

        # Combine the actions, ensuring the opponent's actions are legal
        full_action = np.zeros(self.env.unwrapped.n_players, dtype=int)
        for i in range(self.env.unwrapped.n_players):
            if i in self.training_player_ids:
                full_action[i] = action[i]
            else: # This is an opponent player
                predicted_action = opponent_action_raw[i]
                # Enforce the action mask for the opponent
                if action_mask[i][predicted_action] == 1:
                    full_action[i] = predicted_action
                else:
                    full_action[i] = 0 # Fallback to NOOP if illegal
                
        # Step the underlying environment with the combined action
        obs, reward, done, truncated, info = self.env.step(full_action)
        
        # Store the latest observation for the opponent's next decision
        self.last_obs = obs
        
        return obs, reward, done, truncated, info

# --- Main Training Logic ---

def sample_geometric(indices: list[int], beta: float) -> int:
    """Return index sampled with decayed probability (newest highest)."""
    K = len(indices)
    # newest has i = K, oldest i=1
    weights = [(1 - beta) * (beta ** (K - i)) for i in range(1, K + 1)]
    total = sum(weights)
    probs = [w / total for w in weights]
    return random.choices(indices, weights=probs, k=1)[0]


def get_random_policy_from_artifacts(
    client,
    run_id,
    team_prefix,
    tmpdir,
    K: int = 20,
    beta: float = 0.8,
    uniform_eps: float = 0.10,
):
    """Sample an opponent checkpoint using a geometric decay over recent K snapshots.

    Args:
        client: MLflow client
        run_id: experiment run
        team_prefix: "offense" or "defense"
        tmpdir: temp dir to download artifact
        K: reservoir size (keep last K)
        beta: geometric decay factor (0<beta<1)
        uniform_eps: probability of picking uniformly among all snapshots.
    """
    artifact_path = "models"
    all_artifacts = client.list_artifacts(run_id, artifact_path)

    # Extract paths for team
    team_policies = [
        f.path
        for f in all_artifacts
        if f.path.startswith(f"{artifact_path}/{team_prefix}") and f.path.endswith(".zip")
    ]

    if not team_policies:
        return None

    # sort chronologically by alternation number embedded at end _<n>.zip
    def sort_key(p):
        m = re.search(r"_(\d+)\.zip$", p)
        return int(m.group(1)) if m else 0

    team_policies.sort(key=sort_key)

    # keep last K
    recent_pols = team_policies[-K:]

    # with small probability sample uniform over all for coverage
    if random.random() < uniform_eps:
        chosen = random.choice(team_policies)
    else:
        # geometric sampling over recent_pols
        # indices list 0..len-1 correspond to oldest..newest in recent_pols
        idx = sample_geometric(list(range(len(recent_pols))), beta)
        chosen = recent_pols[idx]

    print(f"  - Selected opponent policy: {os.path.basename(chosen)}")
    local_path = client.download_artifacts(run_id, chosen, tmpdir)
    return PPO.load(local_path)

# --- Continuation helpers ---

def get_latest_policy_path(client, run_id: str, team_prefix: str) -> Optional[str]:
    artifact_path = "models"
    all_artifacts = client.list_artifacts(run_id, artifact_path)
    candidates = [f.path for f in all_artifacts if f.path.startswith(f"{artifact_path}/{team_prefix}") and f.path.endswith('.zip')]
    if not candidates:
        return None
    def sort_key(p):
        m = re.search(r"_(\d+)\.zip$", p)
        return int(m.group(1)) if m else 0
    candidates.sort(key=sort_key)
    return candidates[-1]

def get_max_alternation_index(client, run_id: str) -> int:
    """Return the max alternation index already present in the run (0 if none)."""
    artifact_path = "models"
    all_artifacts = client.list_artifacts(run_id, artifact_path)
    idxs = []
    for f in all_artifacts:
        m = re.search(r"_(\d+)\.zip$", f.path)
        if m:
            idxs.append(int(m.group(1)))
    return max(idxs) if idxs else 0

def setup_environment(args, training_team):
    """Create, configure, and wrap the environment for training."""
    env = basketworld.HexagonBasketballEnv(
        grid_size=args.grid_size,
        players_per_side=args.players,
        shot_clock_steps=args.shot_clock,
        defender_pressure_distance=args.defender_pressure_distance,
        defender_pressure_turnover_chance=args.defender_pressure_turnover_chance,
        three_point_distance=args.three_point_distance,
        layup_pct=args.layup_pct,
        three_pt_pct=args.three_pt_pct,
        shot_pressure_enabled=args.shot_pressure_enabled,
        shot_pressure_max=args.shot_pressure_max,
        shot_pressure_lambda=args.shot_pressure_lambda,
        shot_pressure_arc_degrees=args.shot_pressure_arc_degrees,
        enable_profiling=args.enable_env_profiling,
        training_team=training_team # Critical for correct rewards
    )
    # IMPORTANT: Aggregate rewards BEFORE monitoring
    env = RewardAggregationWrapper(env)
    return Monitor(env)

# ----------------------------------------------------------
# Helper function to create a vectorized self-play environment
# ----------------------------------------------------------


def make_vector_env(
    args,
    training_team: Team,
    opponent_policy,
    num_envs: int,
) -> DummyVecEnv:
    """Return a DummyVecEnv with `num_envs` copies of the self-play environment.

    Each copy is wrapped with `SelfPlayEnvWrapper` so that the opponent's
    behaviour is provided by the frozen `opponent_policy`.
    """

    def _single_env_factory() -> gym.Env:  # type: ignore[name-defined]
        # We capture the current parameters via default args so that each lambda
        # has its own bound values (important inside list comprehension).
        return SelfPlayEnvWrapper(
            setup_environment(args, training_team),
            opponent_policy=opponent_policy,
        )

    return DummyVecEnv([_single_env_factory for _ in range(num_envs)])

def main(args):
    """Main training function."""

    # --- Set up MLflow Tracking ---
    # MLflow requires a running server to log artifacts correctly.
    tracking_uri = "http://localhost:5000"
    mlflow.set_tracking_uri(tracking_uri)
    
    # Set the experiment name. This will create it if it doesn't exist.
    mlflow.set_experiment(args.mlflow_experiment_name)
    
    try:
        # Check if the server is reachable by trying to get the current experiment
        mlflow.get_experiment_by_name(args.mlflow_experiment_name)
    except mlflow.exceptions.MlflowException as e:
        print(f"Could not connect to MLflow tracking server at {tracking_uri}.", file=sys.stderr)
        print("Please ensure the MLflow UI server is running in a separate terminal with `mlflow ui`.", file=sys.stderr)
        sys.exit(1)
    
    with mlflow.start_run(run_name=args.mlflow_run_name) as run:
        print("MLflow tracking URI:", mlflow.get_tracking_uri())
        # Log hyperparameters
        mlflow.log_params(vars(args))
        print(f"MLflow Run ID: {run.info.run_id}")

        # --- Define Policy Kwargs ---
        # This allows us to set the network architecture from the command line.
        policy_kwargs = {}
        if args.net_arch is not None:
            policy_kwargs['net_arch'] = args.net_arch

        # The save_path is no longer needed as models are saved to a temp dir
        # timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # save_path = os.path.join(args.save_path, f"basketworld_selfplay_{timestamp}")
        # os.makedirs(save_path, exist_ok=True)
        
        # --- Initialize Base Environment (just for policy creation) ---
        # The model must be created with the same number of parallel envs that will be
        # used later (SB3 stores this value internally).
        temp_env = DummyVecEnv([
            (lambda: setup_environment(args, Team.OFFENSE)) for _ in range(args.num_envs)
        ])
        
        # --- Initialize Timing Callbacks ---
        offense_timing_callback = RolloutUpdateTimingCallback()
        defense_timing_callback = RolloutUpdateTimingCallback()

        print("Initializing policies...")
        offense_policy = None
        defense_policy = None

        if args.continue_run_id:
            print(f"Continuing from run {args.continue_run_id}...")
            client = mlflow.tracking.MlflowClient()
            with tempfile.TemporaryDirectory() as tmpd:
                off_art = get_latest_policy_path(client, args.continue_run_id, "offense")
                def_art = get_latest_policy_path(client, args.continue_run_id, "defense")
                if off_art and def_art:
                    off_local = client.download_artifacts(args.continue_run_id, off_art, tmpd)
                    def_local = client.download_artifacts(args.continue_run_id, def_art, tmpd)
                    offense_policy = PPO.load(off_local, env=temp_env)
                    defense_policy = PPO.load(def_local, env=temp_env)
                    print(f"  - Loaded latest offense: {os.path.basename(off_art)}")
                    print(f"  - Loaded latest defense: {os.path.basename(def_art)}")

        if offense_policy is None or defense_policy is None:
            offense_policy = PPO(
                "MultiInputPolicy", 
                temp_env, 
                verbose=1, 
                n_steps=args.n_steps, 
                vf_coef=args.vf_coef,
                ent_coef=args.ent_coef,
                batch_size=args.batch_size,
                tensorboard_log=None, # Disable TensorBoard if using MLflow
                policy_kwargs=policy_kwargs
            )
            defense_policy = PPO(
                "MultiInputPolicy", 
                temp_env, 
                verbose=1, 
                n_steps=args.n_steps, 
                vf_coef=args.vf_coef,
                ent_coef=args.ent_coef,
                batch_size=args.batch_size,
                tensorboard_log=None, # Disable TensorBoard if using MLflow
                policy_kwargs=policy_kwargs
            )
        temp_env.close()

        # --- Log the actual network architecture used ---
        # This ensures we capture the default if no custom arch is provided.
        actual_net_arch = str(offense_policy.policy.net_arch)
        mlflow.log_param("net_arch_used", actual_net_arch)
        print(f"  - Using network architecture: {actual_net_arch}")

        # --- Alternating Training Loop ---
        # Determine starting alternation index when continuing in-place
        base_alt_idx = 0
        if args.continue_run_id:
            base_alt_idx = get_max_alternation_index(mlflow.tracking.MlflowClient(), args.continue_run_id)
            print(f"Resuming alternations from index {base_alt_idx + 1}")

        for i in range(args.alternations):
            print("-" * 50)
            global_alt = base_alt_idx + i + 1
            print(f"Alternation {global_alt} (segment {i + 1} / {args.alternations})")
            print("-" * 50)

            with tempfile.TemporaryDirectory() as tmpdir:
                # --- Load a random historical opponent for this alternation ---
                print("\nLoading historical opponent policy...")
                opponent_for_offense = get_random_policy_from_artifacts(
                    mlflow.tracking.MlflowClient(), run.info.run_id, "defense", tmpdir
                ) or defense_policy # Fallback to the latest if none are downloaded

            # --- 1. Train Offense against frozen Defense ---
            print(f"\nTraining Offense...")
            offense_env = make_vector_env(
                args,
                training_team=Team.OFFENSE,
                opponent_policy=opponent_for_offense,
                num_envs=args.num_envs,
            )
            offense_policy.set_env(offense_env)
            
            offense_mlflow_callback = MLflowCallback(
                team_name="Offense", 
                offense_policy=offense_policy, 
                defense_policy=defense_policy, 
                log_freq=args.n_steps
            )
            
            offense_logger = Logger(folder=None, output_formats=[HumanOutputFormat(sys.stdout), MLflowWriter("Offense")])
            offense_policy.set_logger(offense_logger)

            offense_policy.learn(
                total_timesteps=args.steps_per_alternation, 
                reset_num_timesteps=False,
                callback=[offense_mlflow_callback, offense_timing_callback]
            )
            offense_env.close()
            
            with tempfile.TemporaryDirectory() as tmpdir:
                offense_model_path = os.path.join(tmpdir, f"offense_policy_alt_{global_alt}.zip")
                offense_policy.save(offense_model_path)
                mlflow.log_artifact(offense_model_path, artifact_path="models")
            print(f"Logged offense model for alternation {global_alt} to MLflow")

            with tempfile.TemporaryDirectory() as tmpdir:
                print("\nLoading historical opponent policy...")
                opponent_for_defense = get_random_policy_from_artifacts(
                    mlflow.tracking.MlflowClient(), run.info.run_id, "offense", tmpdir
                ) or offense_policy # Fallback to the latest if none are downloaded

            # --- 2. Train Defense against frozen Offense ---
            print(f"\nTraining Defense...")
            defense_env = make_vector_env(
                args,
                training_team=Team.DEFENSE,
                opponent_policy=opponent_for_defense,
                num_envs=args.num_envs,
            )
            defense_policy.set_env(defense_env)

            defense_mlflow_callback = MLflowCallback(
                team_name="Defense", 
                offense_policy=offense_policy, 
                defense_policy=defense_policy, 
                log_freq=args.n_steps
            )

            defense_logger = Logger(folder=None, output_formats=[HumanOutputFormat(sys.stdout), MLflowWriter("Defense")])
            defense_policy.set_logger(defense_logger)

            defense_policy.learn(
                total_timesteps=args.steps_per_alternation, 
                reset_num_timesteps=False,
                callback=[defense_mlflow_callback, defense_timing_callback]
            )
            defense_env.close()

            with tempfile.TemporaryDirectory() as tmpdir:
                defense_model_path = os.path.join(tmpdir, f"defense_policy_alt_{global_alt}.zip")
                defense_policy.save(defense_model_path)
                mlflow.log_artifact(defense_model_path, artifact_path="models")
            print(f"Logged defense model for alternation {global_alt} to MLflow")
 
            # --- 3. Run Evaluation Phase ---
            if args.eval_freq > 0 and (i + 1) % args.eval_freq == 0:
                print(f"\n--- Running Evaluation for Alternation {global_alt} ---")
                
                # Create a renderable environment for evaluation
                eval_env = basketworld.HexagonBasketballEnv(
                    grid_size=args.grid_size,
                    players_per_side=args.players,
                    shot_clock_steps=args.shot_clock,
                    render_mode="rgb_array",
                    three_point_distance=args.three_point_distance,
                    layup_pct=args.layup_pct,
                    three_pt_pct=args.three_pt_pct,
                    shot_pressure_enabled=args.shot_pressure_enabled,
                    shot_pressure_max=args.shot_pressure_max,
                    shot_pressure_lambda=args.shot_pressure_lambda,
                    shot_pressure_arc_degrees=args.shot_pressure_arc_degrees,
                    enable_profiling=args.enable_env_profiling,
                )

                with tempfile.TemporaryDirectory() as temp_dir:
                    for ep_num in range(args.eval_episodes):
                        obs, info = eval_env.reset()
                        done = False
                        episode_frames = []
                        
                        while not done:
                            # Get actions from the latest policies
                            offense_action, _ = offense_policy.predict(obs, deterministic=True)
                            defense_action, _ = defense_policy.predict(obs, deterministic=True)

                            full_action = np.zeros(eval_env.n_players, dtype=int)
                            for player_id in range(eval_env.n_players):
                                if player_id in eval_env.offense_ids:
                                    full_action[player_id] = offense_action[player_id]
                                else:
                                    full_action[player_id] = defense_action[player_id]
                            
                            obs, reward, done, _, info = eval_env.step(full_action)
                            frame = eval_env.render()
                            episode_frames.append(frame)

                        # Post-episode analysis to determine outcome
                        final_info = info
                        action_results = final_info.get('action_results', {})
                        outcome = "Unknown" # Default outcome

                        if action_results.get('shots'):
                            shooter_id = list(action_results['shots'].keys())[0]
                            shot_result = list(action_results['shots'].values())[0]
                            # Determine 2 or 3 based on position at shot
                            shooter_pos = eval_env.positions[int(shooter_id)]
                            bq, br = eval_env.basket_position
                            dist = (abs(shooter_pos[0] - bq) + abs((shooter_pos[0] + shooter_pos[1]) - (bq + br)) + abs(shooter_pos[1] - br)) // 2
                            is_three = dist >= getattr(eval_env, 'three_point_distance', 4)
                            if shot_result['success']:
                                outcome = "Made 3" if is_three else "Made 2"
                            else:
                                outcome = "Missed 3" if is_three else "Missed 2"
                        elif action_results.get('turnovers'):
                            turnover_reason = action_results['turnovers'][0]['reason']
                            if turnover_reason == 'intercepted':
                                outcome = "Turnover (Intercepted)"
                            elif turnover_reason == 'pass_out_of_bounds':
                                outcome = "Turnover (OOB)"
                            elif turnover_reason == 'move_out_of_bounds':
                                outcome = "Turnover (OOB)"
                            elif turnover_reason == 'defender_pressure':
                                outcome = "Turnover (Pressure)"
                        elif eval_env.unwrapped.shot_clock <= 0:
                            outcome = "Turnover (Shot Clock Violation)"

                        # Define the artifact path for this specific evaluation context
                        artifact_path = f"training_eval/alternation_{global_alt}"
                        create_and_log_gif(
                            frames=episode_frames, 
                            episode_num=ep_num, 
                            outcome=outcome, 
                            temp_dir=temp_dir,
                            artifact_path=artifact_path
                        )

                eval_env.close()
                print(f"--- Evaluation for Alternation {global_alt} Complete ---")

            # Log environment profiling if enabled
            if args.enable_env_profiling:
                try:
                    prof = offense_env.envs[0].unwrapped.get_profile_stats()
                    for k, v in prof.items():
                        mlflow.log_metric(f"env_prof_{k}_avg_us_offense", v.get("avg_us", 0.0), step=global_alt)
                    offense_env.envs[0].unwrapped.reset_profile_stats()
                except Exception:
                    pass
                try:
                    prof = defense_env.envs[0].unwrapped.get_profile_stats()
                    for k, v in prof.items():
                        mlflow.log_metric(f"env_prof_{k}_avg_us_defense", v.get("avg_us", 0.0), step=global_alt)
                    defense_env.envs[0].unwrapped.reset_profile_stats()
                except Exception:
                    pass

            # --- 4. Optional GIF Evaluation ---

        print("\n--- Training Complete ---")

        # --- Log final performance metrics ---
        if offense_timing_callback.rollout_times:
            mean_rollout_offense = np.mean(offense_timing_callback.rollout_times)
            mean_update_offense = np.mean(offense_timing_callback.update_times)
            print(f"Offense Mean Rollout Time: {mean_rollout_offense:.3f} s")
            print(f"Offense Mean Update Time:  {mean_update_offense:.3f} s")
            mlflow.log_param("perf_mean_rollout_sec_offense", f"{mean_rollout_offense:.3f}")
            mlflow.log_param("perf_mean_update_sec_offense", f"{mean_update_offense:.3f}")

        if defense_timing_callback.rollout_times:
            mean_rollout_defense = np.mean(defense_timing_callback.rollout_times)
            mean_update_defense = np.mean(defense_timing_callback.update_times)
            print(f"Defense Mean Rollout Time: {mean_rollout_defense:.3f} s")
            print(f"Defense Mean Update Time:  {mean_update_defense:.3f} s")
            mlflow.log_param("perf_mean_rollout_sec_defense", f"{mean_rollout_defense:.3f}")
            mlflow.log_param("perf_mean_update_sec_defense", f"{mean_update_defense:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO models using self-play.")
    parser.add_argument("--grid-size", type=int, default=12, help="The size of the grid.")
    parser.add_argument("--layup-pct", type=float, default=0.60, help="Percentage of layups.")
    parser.add_argument("--three-pt-pct", type=float, default=0.37, help="Percentage of three-pointers.")
    parser.add_argument("--three-point-distance", type=int, default=4, help="Hex distance defining the three-point line.")
    parser.add_argument("--players", type=int, default=2, help="Number of players per side.")
    parser.add_argument("--shot-clock", type=int, default=20, help="Steps in the shot clock.")
    parser.add_argument("--alternations", type=int, default=10, help="Number of times to alternate training.")
    parser.add_argument("--steps-per-alternation", type=int, default=20_000, help="Timesteps to train each policy per alternation.")
    parser.add_argument("--n-steps", type=int, default=2048, help="PPO hyperparameter: Number of steps to run for each environment per update.")
    parser.add_argument("--gamma", type=float, default=0.99, help="PPO hyperparameter: Discount factor for future rewards.")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="PPO hyperparameter: Weight for value function loss.")
    parser.add_argument("--ent-coef", type=float, default=0, help="PPO hyperparameter: Weight for entropy loss.")
    parser.add_argument("--batch-size", type=int, default=64, help="PPO hyperparameter: Minibatch size.")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4, help="Learning rate for PPO optimizers.")
    parser.add_argument("--net-arch", type=int, nargs='+', default=None, help="The size of the neural network layers (e.g., 128 128). Default is SB3's default.")
    parser.add_argument("--continue-run-id", type=str, default=None, help="If set, load latest offense/defense policies from this MLflow run and continue training. Also appends new artifacts using continued alternation indices.")
    parser.add_argument("--eval-freq", type=int, default=2, help="Run evaluation every N alternations. Set to 0 to disable.")
    parser.add_argument("--eval-episodes", type=int, default=10, help="Number of episodes to run for each evaluation.")
    # The --save-path argument is no longer needed
    # parser.add_argument("--save-path", type=str, default="models/", help="Path to save the trained models.")
    parser.add_argument("--defender-pressure-distance", type=int, default=1, help="Distance at which defender pressure is applied.")
    parser.add_argument("--defender-pressure-turnover-chance", type=float, default=0.05, help="Chance of a defender pressure turnover.")
    parser.add_argument("--tensorboard-path", type=str, default=None, help="Path to save TensorBoard logs (set to None if using MLflow).")
    parser.add_argument("--mlflow-experiment-name", type=str, default="BasketWorld_Training", help="Name of the MLflow experiment.")
    parser.add_argument("--mlflow-run-name", type=str, default=None, help="Name of the MLflow run.")
    parser.add_argument("--num-envs", type=int, default=8, help="Number of parallel environments to run for each policy during training.")
    parser.add_argument("--shot-pressure-enabled", type=lambda v: str(v).lower() in ["1","true","yes","y","t"], default=True, help="Enable defender shot pressure model.")
    parser.add_argument("--shot-pressure-max", type=float, default=0.5, help="Max multiplicative reduction at distance 1 (e.g., 0.5 -> up to -50%).")
    parser.add_argument("--shot-pressure-lambda", type=float, default=1.0, help="Exponential decay rate per hex for shot pressure.")
    parser.add_argument("--shot-pressure-arc-degrees", type=float, default=60.0, help="Arc width centered toward basket for pressure eligibility.")
    parser.add_argument("--enable-env-profiling", type=lambda v: str(v).lower() in ["1","true","yes","y","t"], default=False, help="Enable timing instrumentation inside the environment and log averages to MLflow after each alternation.")
    parser.add_argument("--spawn-distance", type=int, default=3, help="minimum distance from 3pt line at which players spawn.")
    args = parser.parse_args()
 
    main(args) 