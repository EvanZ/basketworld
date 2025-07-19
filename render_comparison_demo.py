#!/usr/bin/env python3
"""
Demo script to create a comparison of different hexagon basketball configurations.
"""

import basketworld
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def main():
    print("Creating Hexagon Basketball Environment Comparison...")
    
    # Create environments with different configurations
    configs = [
        {"players_per_side": 2, "title": "2v2 Quick Game"},
        {"players_per_side": 3, "title": "3v3 Standard"},
        {"players_per_side": 4, "title": "4v4 Extended"}
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Hexagon Basketball Environment - Different Configurations', 
                 fontsize=16, fontweight='bold')
    
    for i, config in enumerate(configs):
        print(f"\nCreating {config['title']} environment...")
        
        env = basketworld.HexagonBasketballEnv(
            players_per_side=config["players_per_side"],
            shot_clock_steps=24,
            seed=42 + i  # Different seed for each
        )
        
        obs, info = env.reset()
        
        # Take a random action to show some movement
        actions = env.action_space.sample()
        try:
            obs, rewards, done, truncated, info = env.step(actions)
        except:
            pass  # Use initial state if step fails
        
        # Render to get positions
        positions = env.positions
        ball_holder = env.ball_holder
        offense_ids = env.offense_ids
        
        ax = axes[i]
        ax.set_aspect('equal')
        ax.set_title(config['title'], fontweight='bold')
        
        # Convert axial to cartesian coordinates for rendering  
        def axial_to_cartesian(q, r):
            # Standard conversion for flat-top hexagons
            x = 1.5 * q
            y = np.sqrt(3) * (r + q / 2)
            return x, y
        
        # Draw a simple hexagonal court outline with proper tessellation
        from matplotlib.patches import RegularPolygon
        hex_radius = 0.7  # Proper size for tessellation in comparison view
        grid_radius = 4  # Smaller for comparison
        for q in range(-grid_radius, grid_radius + 1):
            for r in range(-grid_radius, grid_radius + 1):
                if abs(q) <= grid_radius and abs(r) <= grid_radius and abs(q + r) <= grid_radius:
                    x, y = axial_to_cartesian(q, r)
                    hex_patch = RegularPolygon((x, y), 6, radius=hex_radius, 
                                             facecolor='lightgreen', 
                                             edgecolor='darkgreen', alpha=0.3, 
                                             linewidth=0.5, orientation=0)
                    ax.add_patch(hex_patch)
        
        # Draw basket at the environment's basket position
        basket_q, basket_r = env.basket_position
        basket_x, basket_y = axial_to_cartesian(basket_q, basket_r)
        basket = RegularPolygon((basket_x, basket_y), 6, radius=hex_radius, 
                               color='orange', alpha=0.8, zorder=5)
        ax.add_patch(basket)
        ax.text(basket_x, basket_y, 'B', ha='center', va='center', 
               fontsize=10, fontweight='bold', color='white', zorder=6)
        
        # Draw players
        for j, (q, r) in enumerate(positions):
            x, y = axial_to_cartesian(q, r)
            
            # Determine player color
            if j in offense_ids:
                color = 'blue'
            else:
                color = 'red'
            
            # Draw player
            player_circle = plt.Circle((x, y), 0.25, color=color, alpha=0.8, zorder=10)
            ax.add_patch(player_circle)
            
            # Add player number
            ax.text(x, y, str(j), ha='center', va='center', 
                   fontsize=8, fontweight='bold', color='white', zorder=11)
            
            # Mark ball holder
            if j == ball_holder:
                ball_ring = plt.Circle((x, y), 0.35, fill=False, color='gold', 
                                     linewidth=2, zorder=9)
                ax.add_patch(ball_ring)
                ax.text(x, y + 0.5, '●', ha='center', va='center', 
                       fontsize=12, color='orange', fontweight='bold', zorder=12)
        
        # Set limits
        all_x = [axial_to_cartesian(q, r)[0] for q, r in positions]
        all_y = [axial_to_cartesian(q, r)[1] for q, r in positions]
        margin = 1
        ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
        ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
        
        # Add info
        ax.text(0.02, 0.98, f'Players: {config["players_per_side"]}v{config["players_per_side"]}\n'
                           f'Ball: Player {ball_holder}\n'
                           f'Shot Clock: {env.shot_clock}', 
               transform=ax.transAxes, va='top', ha='left',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
               fontsize=9)
        
        ax.grid(True, alpha=0.2)
        ax.set_xlabel('Q Coordinate')
        ax.set_ylabel('R Coordinate')
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                  markersize=8, label='Offense'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                  markersize=8, label='Defense'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', 
                  markersize=8, label='Basket'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gold', 
                  markersize=10, label='Ball Holder', markeredgewidth=2, 
                  markeredgecolor='gold', fillstyle='none')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, 
              bbox_to_anchor=(0.5, -0.05))
    
    plt.tight_layout()
    filename = "hexagon_basketball_comparison.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Comparison image saved as '{filename}'")

if __name__ == "__main__":
    main() 