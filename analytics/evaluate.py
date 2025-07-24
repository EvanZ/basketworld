#!/usr/bin/env python3
"""
Evaluation script to analyze the performance of trained self-play agents.

This script loads a trained offense and defense policy and runs them against
each other for a specified number of episodes. It computes and prints key
performance metrics, such as scoring rate and turnover types, and can
optionally generate a GIF of a sample game.
"""
import argparse
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import basketworld
from basketworld.envs.basketworld_env_v2 import Team
import imageio
from collections import defaultdict
from tqdm import tqdm

def get_outcome_category(outcome_str: str) -> str:
    """Categorizes a detailed outcome string into a simple category for filenames."""
    if "Made Shot" in outcome_str:
        return "made_shot"
    if "Missed Shot" in outcome_str:
        return "missed_shot"
    if "Turnover" in outcome_str:
        return "turnover"
    return "unknown"

def setup_environment(args):
    """Create and wrap the environment for evaluation."""
    
    # Set render mode only if we need to generate the GIF
    render_mode = "rgb_array" if not args.no_render else None

    env = basketworld.HexagonBasketballEnv(
        grid_size=args.grid_size,
        players_per_side=args.players,
        shot_clock_steps=args.shot_clock,
        render_mode=render_mode
    )
    return env

def analyze_results(results: list, num_episodes: int):
    """Analyzes and prints the evaluation results."""
    print("\n--- Evaluation Results ---")
    
    outcomes = defaultdict(int)
    episode_lengths = []
    for res in results:
        outcomes[res['outcome']] += 1
        episode_lengths.append(res['length'])
        
    print(f"Total Episodes: {num_episodes}\n")
    
    # --- Episode Length Statistics ---
    avg_len = np.mean(episode_lengths)
    std_len = np.std(episode_lengths)
    min_len = np.min(episode_lengths)
    max_len = np.max(episode_lengths)
    print("Episode Length Stats:")
    print(f"  - Mean: {avg_len:.2f}")
    print(f"  - Std Dev: {std_len:.2f}")
    print(f"  - Min/Max: {min_len}/{max_len}\n")

    # --- Scoring and Outcome Statistics ---
    made_shots = outcomes.get("Made Shot", 0)
    score_rate = (made_shots / num_episodes) * 100
    print(f"Offensive Score Rate: {score_rate:.2f}%")
    
    print("\nEpisode Termination Breakdown:")
    for outcome, count in sorted(outcomes.items()):
        percentage = (count / num_episodes) * 100
        print(f"- {outcome}: {count}/{num_episodes} ({percentage:.2f}%)")

def main(args):
    """Main evaluation function."""
    
    if not os.path.exists(args.offense_policy) or not os.path.exists(args.defense_policy):
        print("Error: Policy files not found. Please check the paths.")
        return

    # --- Setup ---
    print("Setting up environment for evaluation...")
    env = setup_environment(args)

    print("Loading policies...")
    offense_policy = PPO.load(args.offense_policy)
    defense_policy = PPO.load(args.defense_policy)

    # --- Evaluation Loop ---
    print(f"\nRunning {args.episodes} evaluation episodes...")
    
    num_episodes = args.episodes
    episode_results = []
    
    # This dictionary will store the frames of the longest episode for each outcome type
    longest_episodes = {}

    for i in tqdm(range(num_episodes), desc="Running Evaluation"):
        obs, info = env.reset()
        done = False
        
        offense_ids = env.offense_ids
        
        # We now render frames for every episode, if rendering is enabled
        episode_frames = []
        if not args.no_render:
            frame = env.render()
            episode_frames.append(frame)

        while not done:
            offense_action, _ = offense_policy.predict(obs, deterministic=True)
            defense_action, _ = defense_policy.predict(obs, deterministic=True)

            full_action = np.zeros(env.n_players, dtype=int)
            for player_id in range(env.n_players):
                if player_id in offense_ids:
                    full_action[player_id] = offense_action[player_id]
                else:
                    full_action[player_id] = defense_action[player_id]
            
            obs, reward, done, _, info = env.step(full_action)
            
            if not args.no_render:
                frame = env.render()
                episode_frames.append(frame)

        # --- Post-episode analysis ---
        final_info = info
        action_results = final_info.get('action_results', {})
        outcome = "Unknown"

        if action_results.get('shots'):
            shot_result = list(action_results['shots'].values())[0]
            outcome = "Made Shot" if shot_result['success'] else "Missed Shot"
        elif action_results.get('passes'):
            pass_result = list(action_results['passes'].values())[0]
            if pass_result.get('turnover'):
                outcome = "Turnover (Intercepted)"
        elif action_results.get('out_of_bounds_turnover'):
            outcome = "Turnover (Move Out of Bounds)"
        elif final_info.get('shot_clock', 1) <= 0:
            outcome = "Turnover (Shot Clock Violation)"

        episode_results.append({
            "outcome": outcome,
            "length": env.step_count
        })

        # --- Check if this is the new longest episode for its category ---
        if not args.no_render:
            category = get_outcome_category(outcome)
            current_length = env.step_count
            
            if category not in longest_episodes or current_length > longest_episodes[category]['length']:
                longest_episodes[category] = {
                    'length': current_length,
                    'frames': episode_frames
                }

    # --- Final Report ---
    analyze_results(episode_results, num_episodes)

    # --- Save GIFs of the longest episodes ---
    if not args.no_render:
        print("\nSaving longest episode GIFs...")
        for category, data in longest_episodes.items():
            if data['frames']:
                gif_path = f"{category}_viz.gif"
                print(f"  - Saving {gif_path} (length: {data['length']} steps)")
                imageio.mimsave(gif_path, data['frames'], fps=5)
        print("Done.")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate trained BasketWorld agents.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--offense-policy", type=str, required=True, help="Path to the trained offense policy (.zip).")
    parser.add_argument("--defense-policy", type=str, required=True, help="Path to the trained defense policy (.zip).")
    parser.add_argument(
        "--grid-size", 
        type=int, 
        default=12, 
        help="The size of the grid. MUST match the training configuration."
    )
    parser.add_argument(
        "--players", 
        type=int, 
        default=2, 
        help="Number of players per side. MUST match the training configuration."
    )
    parser.add_argument(
        "--shot-clock", 
        type=int, 
        default=20, 
        help="Steps in the shot clock. MUST match the training configuration."
    )
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to run for evaluation.")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering a sample GIF.")
    
    args = parser.parse_args()
    main(args) 