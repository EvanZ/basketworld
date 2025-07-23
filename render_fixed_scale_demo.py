#!/usr/bin/env python3
"""
Demo script to render and save a visual image of the FIXED SCALE hexagon basketball environment.
"""
import basketworld
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def sample_legal_actions(env, obs):
    """Samples a legal action for each player based on the action mask."""
    action_masks = obs['action_mask']
    actions = []
    for mask in action_masks:
        legal_actions = np.where(mask == 1)[0]
        action = env._rng.choice(legal_actions)
        actions.append(action)
    return np.array(actions)

def main():
    print("Creating Fixed-Scale Rectangular Hexagon Basketball Environment...")
    
    # Create a 3v3 environment with a 16x13 court
    env = basketworld.HexagonBasketballEnv(
        grid_size=16,
        players_per_side=3,
        shot_clock_steps=24
    )
    
    # Reset to get initial state
    obs, info = env.reset()
    
    frames = []
    
    # Run a few random steps to show some action
    for i in range(25): # Increased step limit to see more action
        # Log who has the ball at the start of the step
        print(f"Step {i+1}: Ball is with Player {env.ball_holder}")

        # Render frame
        rgb_array = env.render(mode="rgb_array")
        frames.append(rgb_array)
        
        # Sample a legal action using the action mask
        actions = sample_legal_actions(env, obs)
        obs, rewards, done, truncated, info = env.step(actions)
        
        # Log the actions and rewards for the current step
        print(f"  Actions: {actions}")
        print(f"  Rewards: {rewards}")
        
        # Log shot attempts and their results
        if info['action_results']['shots']:
            for shooter_id, shot_result in info['action_results']['shots'].items():
                result_text = "MADE" if shot_result['success'] else "MISSED"
                print(f"--- Shot by Player {shooter_id}: {result_text}! ---")
        
        if done:
            # Render one last frame after episode ends
            rgb_array = env.render(mode="rgb_array")
            frames.append(rgb_array)
            
            # --- Log the reason for the episode ending ---
            end_reason = "Unknown"
            if info['action_results']['shots']:
                end_reason = "Shot Attempt"
            # Check for pass turnovers and their specific reasons
            elif info['action_results']['passes']:
                for passer_id, pass_result in info['action_results']['passes'].items():
                    if pass_result.get("turnover"):
                        reason = pass_result.get("reason", "unknown")
                        if reason == "intercepted":
                            end_reason = "Turnover (Pass Intercepted by Defense)"
                        elif reason == "out_of_bounds":
                            end_reason = "Turnover (Pass Sailed Out of Bounds)"
                        else:
                            end_reason = f"Turnover on Pass ({reason})"
                        break # Found the turnover, stop looking
            elif info['action_results'].get('out_of_bounds_turnover'):
                end_reason = "Turnover (Player Stepped Out of Bounds)"
            elif env.shot_clock <= 0:
                end_reason = "Shot Clock Violation"
            
            print(f"\nEpisode ended after {i+1} steps. Reason: {end_reason}")
            break
            
    # Save the last frame as a static image
    if frames:
        img = Image.fromarray(frames[-1])
        filename = "fixed_scale_basketball.png"
        img.save(filename)
        print(f"✅ Final step saved as '{filename}'")

    # Create and save an animation
    if len(frames) > 1:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.axis('off')
        
        im = ax.imshow(frames[0])
        
        def animate(i):
            im.set_array(frames[i])
            return [im]
            
        anim = FuncAnimation(fig, animate, frames=len(frames), interval=400, blit=True)
        
        gif_filename = "fixed_scale_basketball.gif"
        anim.save(gif_filename, writer='pillow', fps=2.5)
        print(f"✅ Animation saved as '{gif_filename}'")
        plt.close(fig)

if __name__ == "__main__":
    main() 