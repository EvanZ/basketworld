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

def get_random_policy_from_artifacts(client, run_id, team_prefix, tmpdir):
    """
    Downloads a random historical policy for a given team from the MLflow run.
    If no policies are found, returns None.
    """
    artifact_path = "models"
    all_artifacts = client.list_artifacts(run_id, artifact_path)
    
    # Filter for the specific team's policy files (.zip)
    team_policies = [
        f.path for f in all_artifacts 
        if f.path.startswith(f"{artifact_path}/{team_prefix}") and f.path.endswith(".zip")
    ]

    if not team_policies:
        return None # No historical policies found for this team

    # Select a random policy from the list
    random_policy_path = random.choice(team_policies)
    print(f"  - Selected random opponent policy: {os.path.basename(random_policy_path)}")

    # Download the selected artifact
    local_path = client.download_artifacts(run_id, random_policy_path, tmpdir)
    return PPO.load(local_path)

def setup_environment(args, training_team):
    """Create, configure, and wrap the environment for training."""
    env = basketworld.HexagonBasketballEnv(
        grid_size=args.grid_size,
        players_per_side=args.players,
        shot_clock_steps=args.shot_clock,
        defender_pressure_distance=args.defender_pressure_distance,
        defender_pressure_turnover_chance=args.defender_pressure_turnover_chance,
        training_team=training_team # Critical for correct rewards
    )
    # IMPORTANT: Aggregate rewards BEFORE monitoring
    env = RewardAggregationWrapper(env)
    return Monitor(env)

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
        # SB3 requires an environment to initialize a policy. We'll create a temporary one.
        temp_env = DummyVecEnv([lambda: setup_environment(args, Team.OFFENSE)])
        
        # --- Initialize Timing Callbacks ---
        offense_timing_callback = RolloutUpdateTimingCallback()
        defense_timing_callback = RolloutUpdateTimingCallback()

        print("Initializing policies...")
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
        for i in range(args.alternations):
            print("-" * 50)
            print(f"Alternation {i + 1} / {args.alternations}")
            print("-" * 50)

            with tempfile.TemporaryDirectory() as tmpdir:
                # --- Load a random historical opponent for this alternation ---
                print("\nLoading historical opponent policy...")
                opponent_for_offense = get_random_policy_from_artifacts(
                    mlflow.tracking.MlflowClient(), run.info.run_id, "defense", tmpdir
                ) or defense_policy # Fallback to the latest if none are downloaded

            # --- 1. Train Offense against frozen Defense ---
            print(f"\nTraining Offense...")
            offense_env = DummyVecEnv([
                lambda: SelfPlayEnvWrapper(
                    setup_environment(args, Team.OFFENSE),
                    opponent_policy=opponent_for_offense
                )
            ])
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
                offense_model_path = os.path.join(tmpdir, f"offense_policy_alt_{i+1}.zip")
                offense_policy.save(offense_model_path)
                mlflow.log_artifact(offense_model_path, artifact_path="models")
            print(f"Logged offense model for alternation {i+1} to MLflow")

            with tempfile.TemporaryDirectory() as tmpdir:
                print("\nLoading historical opponent policy...")
                opponent_for_defense = get_random_policy_from_artifacts(
                    mlflow.tracking.MlflowClient(), run.info.run_id, "offense", tmpdir
                ) or offense_policy # Fallback to the latest if none are downloaded

            # --- 2. Train Defense against frozen Offense ---
            print(f"\nTraining Defense...")
            defense_env = DummyVecEnv([
                lambda: SelfPlayEnvWrapper(
                    setup_environment(args, Team.DEFENSE),
                    opponent_policy=opponent_for_defense
                )
            ])
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
                defense_model_path = os.path.join(tmpdir, f"defense_policy_alt_{i+1}.zip")
                defense_policy.save(defense_model_path)
                mlflow.log_artifact(defense_model_path, artifact_path="models")
            print(f"Logged defense model for alternation {i+1} to MLflow")
 
            # --- 3. Run Evaluation Phase ---
            if args.eval_freq > 0 and (i + 1) % args.eval_freq == 0:
                print(f"\n--- Running Evaluation for Alternation {i + 1} ---")
                
                # Create a renderable environment for evaluation
                eval_env = basketworld.HexagonBasketballEnv(
                    grid_size=args.grid_size,
                    players_per_side=args.players,
                    shot_clock_steps=args.shot_clock,
                    render_mode="rgb_array"
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
                            shot_result = list(action_results['shots'].values())[0]
                            outcome = "Made Shot" if shot_result['success'] else "Missed Shot"
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
                        artifact_path = f"training_eval/alternation_{i + 1}"
                        create_and_log_gif(
                            frames=episode_frames, 
                            episode_num=ep_num, 
                            outcome=outcome, 
                            temp_dir=temp_dir,
                            artifact_path=artifact_path
                        )

                eval_env.close()
                print(f"--- Evaluation for Alternation {i + 1} Complete ---")

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
    parser.add_argument("--players", type=int, default=2, help="Number of players per side.")
    parser.add_argument("--shot-clock", type=int, default=20, help="Steps in the shot clock.")
    parser.add_argument("--alternations", type=int, default=10, help="Number of times to alternate training.")
    parser.add_argument("--steps-per-alternation", type=int, default=20_000, help="Timesteps to train each policy per alternation.")
    parser.add_argument("--n-steps", type=int, default=2048, help="PPO hyperparameter: Number of steps to run for each environment per update.")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="PPO hyperparameter: Weight for value function loss.")
    parser.add_argument("--ent-coef", type=float, default=0, help="PPO hyperparameter: Weight for entropy loss.")
    parser.add_argument("--batch-size", type=int, default=64, help="PPO hyperparameter: Minibatch size.")
    parser.add_argument("--net-arch", type=int, nargs='+', default=None, help="The size of the neural network layers (e.g., 128 128). Default is SB3's default.")
    parser.add_argument("--eval-freq", type=int, default=2, help="Run evaluation every N alternations. Set to 0 to disable.")
    parser.add_argument("--eval-episodes", type=int, default=10, help="Number of episodes to run for each evaluation.")
    # The --save-path argument is no longer needed
    # parser.add_argument("--save-path", type=str, default="models/", help="Path to save the trained models.")
    parser.add_argument("--defender-pressure-distance", type=int, default=1, help="Distance at which defender pressure is applied.")
    parser.add_argument("--defender-pressure-turnover-chance", type=float, default=0.05, help="Chance of a defender pressure turnover.")
    parser.add_argument("--tensorboard-path", type=str, default=None, help="Path to save TensorBoard logs (set to None if using MLflow).")
    parser.add_argument("--mlflow-experiment-name", type=str, default="BasketWorld_Training", help="Name of the MLflow experiment.")
    parser.add_argument("--mlflow-run-name", type=str, default=None, help="Name of the MLflow run.")
    
    args = parser.parse_args()
 
    main(args) 