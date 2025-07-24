#!/usr/bin/env python3
"""
Generates a GIF visualization of a single episode played by trained agents.

This script loads a trained offense and defense policy, runs them against each
other for one full episode, logs the step-by-step details to the console,
and saves the resulting animation as a GIF.
"""
import argparse
import os
import numpy as np
import imageio
from stable_baselines3 import PPO

import basketworld

def main():
    """Main function to run the visualization."""
    parser = argparse.ArgumentParser(
        description="Visualize a single episode played by trained BasketWorld agents.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--offense-policy", type=str, required=True, help="Path to the trained offense policy (.zip).")
    parser.add_argument("--defense-policy", type=str, required=True, help="Path to the trained defense policy (.zip).")
    parser.add_argument(
        "--grid-size", type=int, default=12,
        help="The size of the grid. MUST match the training configuration."
    )
    parser.add_argument(
        "--players", type=int, default=2,
        help="Number of players per side. MUST match the training configuration."
    )
    parser.add_argument(
        "--shot-clock", type=int, default=20,
        help="Steps in the shot clock. MUST match the training configuration."
    )
    parser.add_argument(
        "--loop", type=int, default=0,
        help="How many times the GIF should loop (0 for infinite)."
    )
    parser.add_argument(
        "--save-name", type=str, default="visualized_episode.gif",
        help="Filename for the output GIF."
    )
    args = parser.parse_args()

    # --- Environment and Policy Setup ---
    print("Setting up environment...")
    env = basketworld.HexagonBasketballEnv(
        grid_size=args.grid_size,
        players_per_side=args.players,
        shot_clock_steps=args.shot_clock,
        render_mode="rgb_array"
    )

    print("Loading policies...")
    if not os.path.exists(args.offense_policy) or not os.path.exists(args.defense_policy):
        print("Error: Policy file(s) not found. Please check the paths.")
        return
        
    offense_policy = PPO.load(args.offense_policy)
    defense_policy = PPO.load(args.defense_policy)

    # --- Simulation Loop ---
    obs, info = env.reset()
    done = False
    frames = []
    
    offense_ids = env.offense_ids
    
    print("\n--- Running Episode Visualization ---")

    while not done:
        # Render the current frame and add it to our list
        frame = env.render()
        frames.append(frame)
        
        # Get actions from the policies
        offense_action_raw, _ = offense_policy.predict(obs, deterministic=True)
        defense_action_raw, _ = defense_policy.predict(obs, deterministic=True)
        
        # Combine actions into a single array, ensuring they are legal
        actions = np.zeros(env.n_players, dtype=int)
        action_mask = obs['action_mask']

        for i in range(env.n_players):
            # Determine which policy's prediction to use for the current player
            if i in offense_ids:
                predicted_action = offense_action_raw[i]
            else:
                predicted_action = defense_action_raw[i]

            # Enforce the action mask. If predicted action is illegal, fall back to NOOP.
            if action_mask[i][predicted_action] == 1:
                actions[i] = predicted_action
            else:
                # Add a diagnostic print to see what illegal action was attempted
                action_name = basketworld.envs.basketworld_env_v2.ActionType(predicted_action).name
                print(f"  > Player {i} (Defense) tried illegal action: {action_name}. Overriding to NOOP.")
                actions[i] = 0  # ActionType.NOOP.value

        # Step the environment
        obs, rewards, done, _, info = env.step(actions)
        
        # --- Log Step Information ---
        print(f"Shot Clock: {info.get('shot_clock', 'N/A')}, Ball Holder: Player {env.ball_holder}")
        print(f"  Actions: {actions}")
        print(f"  Rewards: {rewards}")

        # Log shot results if a shot was taken
        if info['action_results'].get('shots'):
            shooter_id, result = list(info['action_results']['shots'].items())[0]
            outcome = "MAKES" if result['success'] else "MISSES"
            print(f"  SHOT by Player {shooter_id}: {outcome} the shot!")

        if done:
            # Determine the final reason for the episode ending
            episode_end_reason = "Unknown"
            if info['action_results'].get('shots'):
                episode_end_reason = "Shot Attempt"
            elif info['action_results'].get('passes'):
                pass_result = list(info['action_results']['passes'].values())[0]
                if pass_result.get("turnover"):
                    reason = pass_result.get("reason", "unknown")
                    episode_end_reason = f"Turnover (Pass {reason})"
            elif info['action_results'].get('out_of_bounds_turnover'):
                episode_end_reason = "Turnover (Move Out of Bounds)"
            elif info.get('shot_clock', 1) <= 0:
                episode_end_reason = "Turnover (Shot Clock Violation)"
            
            print(f"Episode terminated. Reason: {episode_end_reason}")

    # Append the final frame to the animation
    final_frame = env.render()
    frames.append(final_frame)
    
    # Save the animation as a GIF
    print(f"\nSaving episode animation to {args.save_name}...")
    imageio.mimsave(args.save_name, frames, fps=5, loop=args.loop)
    print("Done.")

    env.close()

if __name__ == "__main__":
    main() 