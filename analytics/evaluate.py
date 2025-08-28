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
import mlflow
from basketworld.utils.evaluation_helpers import get_outcome_category, create_and_log_gif
from collections import defaultdict
from tqdm import tqdm

from basketworld.utils.mlflow_params import get_mlflow_params
from basketworld.envs.basketworld_env_v2 import HexagonBasketballEnv



def analyze_results(results: list, num_episodes: int):
    """Analyzes and prints the evaluation results."""
    print("\n--- Evaluation Results ---")
    
    outcomes = defaultdict(int)
    episode_lengths = []
    shot_probabilities = []
    shot_distances = []
    total_pass_attempts = 0
    total_pass_completions = 0
    total_pass_intercepts = 0
    total_pass_oob = 0
    for res in results:
        outcomes[res['outcome']] += 1
        episode_lengths.append(res['length'])
        total_pass_attempts += res.get('pass_attempts', 0)
        total_pass_completions += res.get('pass_completions', 0)
        total_pass_intercepts += res.get('pass_intercepts', 0)
        total_pass_oob += res.get('pass_oob', 0)
        # shot_probabilities.append(res['probabilities'])
        # shot_distances.append(res['distances'])
    # print(f"Shot Probabilities Mean: {np.mean(shot_probabilities)}")
    # print(f"Shot Probabilities Std: {np.std(shot_probabilities)}")
    print(f"Shot Distances Mean: {np.mean(shot_distances)}")
    print(f"Shot Distances Std: {np.std(shot_distances)}")
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
    made_2pts = outcomes.get("Made 2pt", 0)
    made_3pts = outcomes.get("Made 3pt", 0)
    missed_2pts = outcomes.get("Missed 2pt", 0)
    missed_3pts = outcomes.get("Missed 3pt", 0)
    made_dunks = outcomes.get("Made Dunk", 0)
    missed_dunks = outcomes.get("Missed Dunk", 0)
    score_rate = (made_2pts + made_3pts + made_dunks) / num_episodes
    turnovers = (
        outcomes.get('Turnover (Pressure)', 0)
        + outcomes.get('Turnover (OOB - Move)', 0)
        + outcomes.get('Turnover (OOB - Pass)', 0)
        + outcomes.get('Turnover (Intercepted)', 0)
        + outcomes.get('Turnover (Shot Clock Violation)', 0)
    )
    total_made = made_2pts + made_3pts + made_dunks
    total_missed = missed_2pts + missed_3pts + missed_dunks
    total_shots = total_made + total_missed
    print(f"Offensive Score Rate: {100.0 * score_rate:.2f}%")
    print(f"Made 2pts: {made_2pts}")
    print(f"Missed 2pts: {missed_2pts}")
    print(f"Made 3pts: {made_3pts}")
    print(f"Missed 3pts: {missed_3pts}")
    print(f"Made Dunks: {made_dunks}")
    print(f"Missed Dunks: {missed_dunks}")
    print(f"Total made shots: {total_made}")
    print(f"Total missed shots: {total_missed}")
    print(f"Total shots: {total_shots}")
    print(f"Total turnovers: {turnovers}")
    if turnovers > 0:
        print(f"  - OOB (Move): {outcomes.get('Turnover (OOB - Move)', 0)}")
        print(f"  - OOB (Pass): {outcomes.get('Turnover (OOB - Pass)', 0)}")
        print(f"  - Intercepted: {outcomes.get('Turnover (Intercepted)', 0)}")
        print(f"  - Pressure: {outcomes.get('Turnover (Pressure)', 0)}")
        print(f"  - Shot Clock: {outcomes.get('Turnover (Shot Clock Violation)', 0)}")
    # --- Passing Statistics ---
    print("\nPassing Stats:")
    print(f"Pass attempts: {total_pass_attempts}")
    print(f"Completed passes: {total_pass_completions}")
    if total_pass_attempts > 0:
        print(f"Pass completion%: {100.0 * total_pass_completions / total_pass_attempts:.2f}%")
    else:
        print("Pass completion%: N/A")
    print(f"Intercepted passes: {total_pass_intercepts}")
    print(f"Out-of-bounds passes: {total_pass_oob}")
    # Traditional 2PT% excludes dunks for separate reporting
    if (made_2pts + missed_2pts) > 0:
        print(f"2PT% (non-dunk): {100.0 * made_2pts / (made_2pts + missed_2pts):.2f}%")
    else:
        print("2PT% (non-dunk): N/A")
    if (made_dunks + missed_dunks) > 0:
        print(f"Dunk%: {100.0 * made_dunks / (made_dunks + missed_dunks):.2f}%")
    else:
        print("Dunk%: N/A")
    if made_3pts+missed_3pts > 0:
        print(f"3PT%: {100.0 * made_3pts / (made_3pts + missed_3pts):.2f}%")
    else:
        print("3PT%: N/A")
    if total_shots > 0:
        print(f"FG%: {100.0 * total_made / total_shots:.2f}%")
        print(f"EFG%: {100.0 * (made_2pts + made_dunks + made_3pts * 1.5) / total_shots:.2f}%")    
    else:
        print("FG%: N/A")
        print("EFG%: N/A")
    if (total_shots + turnovers) > 0:
        print(f"PPP: {2.0 * (made_2pts + made_dunks + made_3pts * 1.5) / (total_shots + turnovers):.2f}")
    else:
        print("PPP: N/A")
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
    try:
        required, optional = get_mlflow_params(client, args.run_id)

    except KeyError as e:
        print(f"Error: Run {args.run_id} is missing a required parameter: {e}")
        return

    # Re-open the original run context to log new artifacts to the correct run
    with mlflow.start_run(run_id=args.run_id):
        with tempfile.TemporaryDirectory() as temp_dir:
            # --- Download Model Artifacts ---
            print(f"Fetching latest models from MLflow Run ID: {args.run_id}")
            
            artifacts = client.list_artifacts(args.run_id, "models")
            
            # Find the latest offense and defense policies by alternation number
            latest_offense = max([f.path for f in artifacts if "offense" in f.path], key=lambda p: int(re.search(r'_(\d+)\.zip', p).group(1)))
            latest_defense = max([f.path for f in artifacts if "defense" in f.path], key=lambda p: int(re.search(r'_(\d+)\.zip', p).group(1)))
            
            offense_policy_path = client.download_artifacts(args.run_id, latest_offense, temp_dir)
            defense_policy_path = client.download_artifacts(args.run_id, latest_defense, temp_dir)

            print(f"  - Downloaded Offense Policy: {os.path.basename(latest_offense)}")
            print(f"  - Downloaded Defense Policy: {os.path.basename(latest_defense)}")
            
            # --- Setup ---

            env = HexagonBasketballEnv(
                **required,
                **optional,
            )

            print("Loading policies...")
            offense_policy = PPO.load(offense_policy_path)
            defense_policy = PPO.load(defense_policy_path)

            # --- Evaluation Loop ---
            print(f"\nRunning {args.episodes} evaluation episodes...")
            
            num_episodes = args.episodes
            results = []

            for i in tqdm(range(num_episodes), desc="Running Evaluation"):
                obs, info = env.reset()
                done = False
                
                offense_ids = env.offense_ids
                
                episode_frames = []
                # Per-episode passing counters
                pass_attempts = 0
                pass_completions = 0
                pass_intercepts = 0
                pass_oob = 0
                if not args.no_render:
                    frame = env.render()
                    episode_frames.append(frame)

                while not done:
                    offense_action, _ = offense_policy.predict(obs, deterministic=args.deterministic_offense)
                    defense_action, _ = defense_policy.predict(obs, deterministic=args.deterministic_defense)

                    full_action = np.zeros(env.n_players, dtype=int)
                    for player_id in range(env.n_players):
                        if player_id in offense_ids:
                            full_action[player_id] = offense_action[player_id]
                        else:
                            full_action[player_id] = defense_action[player_id]
                    
                    obs, reward, done, _, info = env.step(full_action)
                    # Count pass attempts and completions from this step
                    step_results = info.get('action_results', {})
                    if step_results.get('passes'):
                        attempts_this_step = len(step_results['passes'])
                        completions_this_step = sum(1 for _pid, pres in step_results['passes'].items() if pres.get('success'))
                        intercepts_this_step = sum(1 for _pid, pres in step_results['passes'].items() if not pres.get('success') and pres.get('reason') == 'intercepted')
                        oob_this_step = sum(1 for _pid, pres in step_results['passes'].items() if not pres.get('success') and pres.get('reason') == 'out_of_bounds')
                        pass_attempts += attempts_this_step
                        pass_completions += completions_this_step
                        pass_intercepts += intercepts_this_step
                        pass_oob += oob_this_step
                    
                    if not args.no_render:
                        frame = env.render()
                        episode_frames.append(frame)

                # --- Post-episode analysis ---
                final_info = info
                action_results = final_info.get('action_results', {})
                outcome = "Unknown" # Default outcome
                three_point_distance = env.three_point_distance
                if action_results.get('shots'):
                    shot_result = list(action_results['shots'].values())[0]
                    is_dunk = (shot_result.get('distance', 999) == 0)
                    if is_dunk:
                        outcome = "Made Dunk" if shot_result['success'] else "Missed Dunk"
                    elif shot_result['success'] and shot_result['distance'] < three_point_distance:
                        outcome = "Made 2pt"
                    elif shot_result['success'] and shot_result['distance'] >= three_point_distance:
                        outcome = "Made 3pt"
                    elif not shot_result['success'] and shot_result['distance'] < three_point_distance:
                        outcome = "Missed 2pt"
                    elif not shot_result['success'] and shot_result['distance'] >= three_point_distance:
                        outcome = "Missed 3pt"
                    else:
                        outcome = "Unknown"
                elif action_results.get('turnovers'):
                    turnover_reason = action_results['turnovers'][0]['reason']
                    if turnover_reason == 'intercepted':
                        outcome = "Turnover (Intercepted)"
                    elif turnover_reason == 'pass_out_of_bounds':
                        outcome = "Turnover (OOB - Pass)"
                    elif turnover_reason == 'move_out_of_bounds':
                        outcome = "Turnover (OOB - Move)"
                    elif turnover_reason == 'defender_pressure':
                        outcome = "Turnover (Pressure)"
                # Check the env state directly for shot clock violation, as info can be off by one step
                elif env.unwrapped.shot_clock <= 0:
                    outcome = "Turnover (Shot Clock Violation)"
                
                # Store results for final summary
                results.append({
                    "outcome": outcome,
                    "length": env.unwrapped.step_count,
                    "episode_num": i,
                    "pass_attempts": pass_attempts,
                    "pass_completions": pass_completions,
                    "pass_intercepts": pass_intercepts,
                    "pass_oob": pass_oob,
                    # "probabilities": shot_result['probability'],
                    # "distances": shot_result['distance'],
                })

                # --- Save and log GIF for this episode ---
                if not args.no_render:
                    create_and_log_gif(
                        frames=episode_frames,
                        episode_num=i,
                        outcome=outcome,
                        temp_dir=temp_dir,
                        artifact_path=f"gifs/{get_outcome_category(outcome)}"
                    )

            # --- Final Analysis ---
            analyze_results(results, num_episodes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate trained BasketWorld agents from an MLflow run.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--run-id", type=str, required=True, help="The MLflow Run ID to evaluate.")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to run for evaluation.")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering a sample GIF.")
    parser.add_argument("--deterministic-offense", type=lambda v: str(v).lower() in ["1","true","yes","y","t"], default=False, help="Use deterministic offense actions.")
    parser.add_argument("--deterministic-defense", type=lambda v: str(v).lower() in ["1","true","yes","y","t"], default=False, help="Use deterministic defense actions.")
    parser.add_argument("--mask-occupied-moves", type=lambda v: str(v).lower() in ["1","true","yes","y","t"], default=None, help="If set, disallow moves into currently occupied neighboring hexes.")
    # Optional overrides for dunk and spawn evaluation
    parser.add_argument("--allow-dunks", type=lambda v: str(v).lower() in ["1","true","yes","y","t"], default=None, help="Override run param to enable/disable dunks during evaluation.")
    parser.add_argument("--dunk-pct", type=float, default=None, help="Override run param for dunk make probability during evaluation.")
    parser.add_argument("--spawn-distance", type=int, default=None, help="Override run param for spawn distance relative to basket (can be negative).")
    
    args = parser.parse_args()
    main(args) 