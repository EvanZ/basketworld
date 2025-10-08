#!/usr/bin/env python3
"""
Visualize the 5 spatial observation channels to understand what the CNN sees.
"""
import numpy as np
import matplotlib.pyplot as plt
from basketworld.envs.basketworld_env_v2 import HexagonBasketballEnv


def visualize_spatial_channels(save_path="spatial_channels_visualization.png"):
    """Create visualization of all 5 spatial channels."""
    
    # Create environment with spatial obs
    env = HexagonBasketballEnv(
        grid_size=12,
        players=3,  # 3v3 for more interesting visualization
        use_spatial_obs=True,
        seed=42,
    )
    
    obs, info = env.reset()
    spatial = obs["spatial"]
    
    # Create figure with 5 subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("BasketWorld CNN Spatial Observation Channels", fontsize=16, fontweight='bold')
    
    channel_names = [
        "Channel 0: Players\n(+1=offense, -1=defense)",
        "Channel 1: Ball Holder\n(binary)",
        "Channel 2: Basket Position\n(binary)",
        "Channel 3: Expected Points\n(shot_value × probability)",
        "Channel 4: Three-Point Arc\n(0=inside, 1=outside)"
    ]
    
    cmaps = ['RdBu_r', 'Oranges', 'Reds', 'YlGnBu', 'binary']
    
    for i in range(5):
        ax = axes[i // 3, i % 3]
        channel_data = spatial[i]
        
        # Plot
        im = ax.imshow(channel_data, cmap=cmaps[i], interpolation='nearest')
        ax.set_title(channel_names[i], fontsize=12, fontweight='bold')
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        
        # Add grid
        ax.grid(True, which='both', color='gray', linewidth=0.5, alpha=0.3)
        ax.set_xticks(np.arange(-0.5, channel_data.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, channel_data.shape[0], 1), minor=True)
        
        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Add text annotations for key channels
        if i == 0:  # Players channel
            for row in range(channel_data.shape[0]):
                for col in range(channel_data.shape[1]):
                    val = channel_data[row, col]
                    if val == 1.0:
                        ax.text(col, row, 'O', ha='center', va='center', 
                               color='blue', fontweight='bold', fontsize=10)
                    elif val == -1.0:
                        ax.text(col, row, 'D', ha='center', va='center',
                               color='red', fontweight='bold', fontsize=10)
        
        elif i == 1:  # Ball holder channel
            ball_pos = np.argwhere(channel_data == 1.0)
            if len(ball_pos) > 0:
                row, col = ball_pos[0]
                ax.text(col, row, '⚽', ha='center', va='center', fontsize=16)
        
        elif i == 2:  # Basket channel
            basket_pos = np.argwhere(channel_data == 1.0)
            if len(basket_pos) > 0:
                row, col = basket_pos[0]
                ax.text(col, row, '🏀', ha='center', va='center', fontsize=16)
    
    # Hide the 6th subplot (we only have 5 channels)
    axes[1, 2].axis('off')
    
    # Add description text in the empty subplot
    desc_text = """
    What the CNN sees:
    
    • Channel 0: All player positions with team identity
    • Channel 1: Who has the ball right now
    • Channel 2: Where the basket/goal is located
    • Channel 3: Expected scoring value from each position
    • Channel 4: Strategic zones (inside vs outside arc)
    
    The CNN learns spatial patterns across all channels
    to make decisions about movement, passing, and shooting.
    """
    axes[1, 2].text(0.1, 0.5, desc_text, transform=axes[1, 2].transAxes,
                   fontsize=10, verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved to: {save_path}")
    plt.close()
    
    # Print statistics
    print("\n" + "=" * 60)
    print("Spatial Observation Statistics")
    print("=" * 60)
    for i, name in enumerate(channel_names):
        channel_data = spatial[i]
        print(f"\n{name.split(':')[1].strip().split('(')[0]}")
        print(f"  Shape: {channel_data.shape}")
        print(f"  Range: [{np.min(channel_data):.3f}, {np.max(channel_data):.3f}]")
        print(f"  Mean: {np.mean(channel_data):.3f}")
        print(f"  Non-zero cells: {np.count_nonzero(channel_data)}")
    
    return spatial


def compare_scenarios():
    """Compare spatial observations across different game scenarios."""
    
    print("\n" + "=" * 60)
    print("Comparing Different Scenarios")
    print("=" * 60)
    
    scenarios = [
        ("2v2 Near Basket", dict(grid_size=12, players=2, spawn_distance=1)),
        ("3v3 Far from Basket", dict(grid_size=12, players=3, spawn_distance=5)),
        ("2v2 Small Court", dict(grid_size=8, players=2, spawn_distance=2)),
    ]
    
    fig, axes = plt.subplots(len(scenarios), 3, figsize=(12, 4 * len(scenarios)))
    fig.suptitle("Expected Points Heatmap Across Scenarios", fontsize=14, fontweight='bold')
    
    for idx, (name, kwargs) in enumerate(scenarios):
        env = HexagonBasketballEnv(**kwargs, use_spatial_obs=True, seed=42)
        obs, _ = env.reset()
        spatial = obs["spatial"]
        
        # Show channels 0 (players), 3 (expected points), 4 (3pt arc)
        channels_to_show = [0, 3, 4]
        titles = ["Players", "Expected Points", "3PT Arc"]
        
        for i, (ch_idx, title) in enumerate(zip(channels_to_show, titles)):
            ax = axes[idx, i] if len(scenarios) > 1 else axes[i]
            im = ax.imshow(spatial[ch_idx], cmap='YlOrRd', interpolation='nearest')
            ax.set_title(f"{name}\n{title}", fontsize=10)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig("spatial_scenarios_comparison.png", dpi=150, bbox_inches='tight')
    print("✓ Scenario comparison saved to: spatial_scenarios_comparison.png")
    plt.close()


def main():
    print("=" * 60)
    print("Spatial Observation Visualization Tool")
    print("=" * 60 + "\n")
    
    # Main visualization
    print("Creating main channel visualization...")
    spatial = visualize_spatial_channels()
    
    # Scenario comparison
    print("\nCreating scenario comparison...")
    compare_scenarios()
    
    print("\n" + "=" * 60)
    print("✓ Visualizations complete!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - spatial_channels_visualization.png")
    print("  - spatial_scenarios_comparison.png")
    print("\nThese show what the CNN 'sees' as input.")


if __name__ == "__main__":
    main()

