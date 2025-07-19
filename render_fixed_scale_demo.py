#!/usr/bin/env python3
"""
Demo script to render and save a visual image of the FIXED SCALE hexagon basketball environment.
"""
import basketworld
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def main():
    print("Creating Fixed-Scale Rectangular Hexagon Basketball Environment...")
    
    # Create a 3v3 environment with a 16x13 court
    env = basketworld.HexagonBasketballEnv(
        grid_size=16,
        players_per_side=3,
        shot_clock_steps=24,
        seed=42
    )
    
    # Reset to get initial state
    obs, info = env.reset()
    
    frames = []
    
    # Run a few random steps to show some action
    for i in range(15):
        # Render frame
        rgb_array = env.render(mode="rgb_array")
        frames.append(rgb_array)
        
        # Take a random step
        actions = env.action_space.sample()
        obs, rewards, done, truncated, info = env.step(actions)
        
        # Print info for the first step
        if i == 0:
            print("Action taken in first step:", actions)
            print("Rewards:", rewards)
        
        if done:
            # Render one last frame after episode ends
            rgb_array = env.render(mode="rgb_array")
            frames.append(rgb_array)
            print(f"Episode ended after {i+1} steps.")
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