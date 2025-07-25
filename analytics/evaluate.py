#!/usr/bin/env python3
"""
Evaluation script to analyze the performance of trained self-play agents.

This script connects to an MLflow run, downloads the latest trained policies,
runs them against each other, computes performance metrics, and logs the
resulting analysis and visualization GIFs back to the original MLflow run.
"""
import argparse
import os
import numpy as np
import tempfile
import re
from stable_baselines3 import PPO
import basketworld
from basketworld.envs.basketworld_env_v2 import Team
import imageio
from collections import defaultdict
from tqdm import tqdm
import mlflow

def get_outcome_category(outcome_str: str) -> str:
    """Categorizes a detailed outcome string into a simple category for filenames."""
    if "Made Shot" in outcome_str:
        return "made_shot"
    if "Missed Shot" in outcome_str:
        return "missed_shot"
    if "Turnover" in outcome_str:
        return "turnover"
    return "unknown"

def setup_environment(grid_size: int, players: int, shot_clock: int, no_render: bool):
    """Create and wrap the environment for evaluation."""
    
    render_mode = "rgb_array" if not no_render else None

    env = basketworld.HexagonBasketballEnv(
        grid_size=grid_size,
        players_per_side=players,
        shot_clock_steps=shot_clock,
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
    
    # --- Set up MLflow Tracking ---
    tracking_uri = "http://localhost:5000"
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()

    try:
        run = client.get_run(args.run_id)
    except Exception as e:
        print(f"Error: Could not find MLflow run with ID '{args.run_id}'. Please ensure the Run ID is correct and the MLflow server is running.")
        print(e)
        return

    # --- Get Hyperparameters from MLflow Run ---
    print("Fetching hyperparameters from MLflow run...")
    run_params = run.data.params
    # Parameters are logged as strings, so we must cast them to integers
    try:
        grid_size = int(run_params["grid_size"])
        players = int(run_params["players"])
        shot_clock = int(run_params["shot_clock"])
        print(f"  - Grid Size: {grid_size}")
        print(f"  - Players: {players}")
        print(f"  - Shot Clock: {shot_clock}")
    except KeyError as e:
        print(f"Error: Run {args.run_id} is missing a required parameter: {e}")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        # --- Download Model Artifacts ---
        print(f"Fetching latest models from MLflow Run ID: {args.run_id}")
        
        artifacts = client.list_artifacts(args.run_id, "models")
        
        # Find the latest offense and defense policies by alternation number
        latest_offense = max([f.path for f in artifacts if "offense" in f.path], key=lambda p: int(re.search(r'_(\d+)\.zip', p).group(1)))
        latest_defense = max([f.path for f in artifacts if "defense" in f.path], key=lambda p: int(re.search(r'_(\d+)\.zip', p).group(1)))
        
        offense_policy_path = client.download_artifacts(args.run_id, latest_offense, tmpdir)
        defense_policy_path = client.download_artifacts(args.run_id, latest_defense, tmpdir)

        print(f"  - Downloaded Offense Policy: {os.path.basename(latest_offense)}")
        print(f"  - Downloaded Defense Policy: {os.path.basename(latest_defense)}")
        
        # --- Setup ---
        print("\nSetting up environment for evaluation...")
        env = setup_environment(grid_size, players, shot_clock, args.no_render)

        print("Loading policies...")
        offense_policy = PPO.load(offense_policy_path)
        defense_policy = PPO.load(defense_policy_path)

        # --- Evaluation Loop ---
        print(f"\nRunning {args.episodes} evaluation episodes...")
        
        num_episodes = args.episodes
        episode_results = []
        
        longest_episodes = {}

        for i in tqdm(range(num_episodes), desc="Running Evaluation"):
            obs, info = env.reset()
            done = False
            
            offense_ids = env.offense_ids
            
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
            print("\nSaving and logging longest episode GIFs to MLflow...")
            with mlflow.start_run(run_id=args.run_id): # Re-opens the run to log artifacts
                for category, data in longest_episodes.items():
                    if data['frames']:
                        gif_path = os.path.join(tmpdir, f"{category}_viz.gif")
                        print(f"  - Saving {os.path.basename(gif_path)} (length: {data['length']} steps)")
                        imageio.mimsave(gif_path, data['frames'], fps=5, loop=0)
                        mlflow.log_artifact(gif_path, artifact_path="gifs")
            print("Done.")

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate trained BasketWorld agents from an MLflow run.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--run-id", type=str, required=True, help="The MLflow Run ID to evaluate.")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to run for evaluation.")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering a sample GIF.")
    
    args = parser.parse_args()
    main(args) 