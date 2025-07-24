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

import basketworld
from basketworld.envs.basketworld_env_v2 import Team

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

def setup_environment(args, training_team):
    """Create, configure, and wrap the environment for training."""
    env = basketworld.HexagonBasketballEnv(
        grid_size=args.grid_size,
        players_per_side=args.players,
        shot_clock_steps=args.shot_clock,
        training_team=training_team # Critical for correct rewards
    )
    # IMPORTANT: Aggregate rewards BEFORE monitoring
    env = RewardAggregationWrapper(env)
    return Monitor(env)

def main(args):
    """Main training function."""
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(args.save_path, f"basketworld_selfplay_{timestamp}")
    os.makedirs(save_path, exist_ok=True)
    
    # --- Initialize Base Environment (just for policy creation) ---
    # SB3 requires an environment to initialize a policy. We'll create a temporary one.
    temp_env = DummyVecEnv([lambda: setup_environment(args, Team.OFFENSE)])
    
    print("Initializing policies...")
    offense_policy = PPO("MultiInputPolicy", temp_env, verbose=1, n_steps=args.n_steps, batch_size=args.batch_size)
    defense_policy = PPO("MultiInputPolicy", temp_env, verbose=1, n_steps=args.n_steps, batch_size=args.batch_size)
    temp_env.close()

    # --- Alternating Training Loop ---
    for i in range(args.alternations):
        print("-" * 50)
        print(f"Alternation {i + 1} / {args.alternations}")
        print("-" * 50)

        # --- 1. Train Offense against frozen Defense ---
        print(f"\nTraining Offense...")
        offense_env = DummyVecEnv([
            lambda: SelfPlayEnvWrapper(
                setup_environment(args, Team.OFFENSE),
                opponent_policy=defense_policy
            )
        ])
        offense_policy.set_env(offense_env)
        offense_policy.learn(total_timesteps=args.steps_per_alternation, reset_num_timesteps=False)
        offense_env.close()
        
        offense_model_path = os.path.join(save_path, f"offense_policy_alt_{i+1}.zip")
        offense_policy.save(offense_model_path)
        print(f"Saved offense model to {offense_model_path}")

        # --- 2. Train Defense against frozen Offense ---
        print(f"\nTraining Defense...")
        defense_env = DummyVecEnv([
            lambda: SelfPlayEnvWrapper(
                setup_environment(args, Team.DEFENSE),
                opponent_policy=offense_policy
            )
        ])
        defense_policy.set_env(defense_env)
        defense_policy.learn(total_timesteps=args.steps_per_alternation, reset_num_timesteps=False)
        defense_env.close()

        defense_model_path = os.path.join(save_path, f"defense_policy_alt_{i+1}.zip")
        defense_policy.save(defense_model_path)
        print(f"Saved defense model to {defense_model_path}")

    print("\n--- Training Complete ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO models using self-play.")
    parser.add_argument("--grid-size", type=int, default=12, help="The size of the grid.")
    parser.add_argument("--players", type=int, default=2, help="Number of players per side.")
    parser.add_argument("--shot-clock", type=int, default=20, help="Steps in the shot clock.")
    parser.add_argument("--alternations", type=int, default=10, help="Number of times to alternate training.")
    parser.add_argument("--steps-per-alternation", type=int, default=20_000, help="Timesteps to train each policy per alternation.")
    parser.add_argument("--n-steps", type=int, default=2048, help="PPO hyperparameter: Number of steps to run for each environment per update.")
    parser.add_argument("--batch-size", type=int, default=64, help="PPO hyperparameter: Minibatch size.")
    parser.add_argument("--save-path", type=str, default="models/", help="Path to save the trained models.")
    
    args = parser.parse_args()
    main(args) 