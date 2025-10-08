#!/usr/bin/env python3
"""
Visualize per-player skill heatmaps to show individual player differences.
"""
import numpy as np
import matplotlib.pyplot as plt
from basketworld.envs.basketworld_env_v2 import HexagonBasketballEnv


def visualize_skill_comparison(save_path="skill_heatmaps_comparison.png"):
    """Compare skill heatmaps across different players."""
    
    # Create 3v3 environment with varied skills
    env = HexagonBasketballEnv(
        grid_size=12,
        players=3,
        use_spatial_obs=True,
        use_pure_cnn=True,
        layup_std=0.10,    # High variability
        three_pt_std=0.10,
        dunk_std=0.05,
        seed=42,
    )
    
    obs, info = env.reset()
    spatial = obs["spatial"]
    
    # Extract skill heatmaps (channels 7-9)
    skill_heatmaps = spatial[7:10]  # Shape: (3, 12, 12)
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Per-Player Skill Heatmaps (Pure CNN Mode)", fontsize=16, fontweight='bold')
    
    # Plot each player's heatmap
    for i in range(3):
        ax = axes[0, i]
        heatmap = skill_heatmaps[i]
        
        im = ax.imshow(heatmap, cmap='YlOrRd', interpolation='nearest', vmin=0.0, vmax=1.0)
        ax.set_title(f"Player {i} Skill Heatmap", fontsize=14, fontweight='bold')
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Shooting Probability', rotation=270, labelpad=20)
        
        # Add statistics
        stats_text = f"Min: {np.min(heatmap):.3f}\nMax: {np.max(heatmap):.3f}\nMean: {np.mean(heatmap):.3f}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               verticalalignment='top', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Plot differences between players
    diff_pairs = [(0, 1), (0, 2), (1, 2)]
    titles = ["Player 0 vs 1", "Player 0 vs 2", "Player 1 vs 2"]
    
    for idx, (p1, p2) in enumerate(diff_pairs):
        ax = axes[1, idx]
        diff = skill_heatmaps[p1] - skill_heatmaps[p2]
        
        # Use diverging colormap centered at zero
        vmax = max(abs(np.min(diff)), abs(np.max(diff)))
        im = ax.imshow(diff, cmap='RdBu_r', interpolation='nearest', vmin=-vmax, vmax=vmax)
        ax.set_title(titles[idx], fontsize=14, fontweight='bold')
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Skill Difference', rotation=270, labelpad=20)
        
        # Add statistics
        avg_diff = np.mean(np.abs(diff))
        max_diff = np.max(np.abs(diff))
        stats_text = f"Avg |diff|: {avg_diff:.3f}\nMax |diff|: {max_diff:.3f}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               verticalalignment='top', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Skill heatmap comparison saved to: {save_path}")
    plt.close()
    
    # Print player skill summary
    print("\n" + "=" * 60)
    print("Player Skill Summary")
    print("=" * 60)
    
    for i in range(3):
        print(f"\nPlayer {i}:")
        print(f"  Layup %:  {env.offense_layup_pct_by_player[i]:.3f}")
        print(f"  3PT %:    {env.offense_three_pt_pct_by_player[i]:.3f}")
        print(f"  Dunk %:   {env.offense_dunk_pct_by_player[i]:.3f}")
        
        heatmap = skill_heatmaps[i]
        print(f"  Heatmap range: [{np.min(heatmap):.3f}, {np.max(heatmap):.3f}]")
        print(f"  Heatmap mean: {np.mean(heatmap):.3f}")
    
    # Print skill specializations
    print("\n" + "=" * 60)
    print("Skill Specializations (vs baseline)")
    print("=" * 60)
    
    baseline_layup = env.layup_pct
    baseline_three = env.three_pt_pct
    
    for i in range(3):
        layup_delta = env.offense_layup_pct_by_player[i] - baseline_layup
        three_delta = env.offense_three_pt_pct_by_player[i] - baseline_three
        
        print(f"\nPlayer {i}:")
        
        if abs(layup_delta) > abs(three_delta):
            if layup_delta > 0:
                spec = "Layup Specialist (drives to basket)"
            else:
                spec = "Weak at layups (stay outside)"
        else:
            if three_delta > 0:
                spec = "3PT Specialist (perimeter shooter)"
            else:
                spec = "Weak at 3PT (avoid perimeter)"
        
        print(f"  Layup delta: {layup_delta:+.3f}")
        print(f"  3PT delta:   {three_delta:+.3f}")
        print(f"  → {spec}")


def main():
    print("=" * 60)
    print("Per-Player Skill Heatmap Visualization")
    print("=" * 60 + "\n")
    
    print("Creating skill heatmap comparison...")
    visualize_skill_comparison()
    
    print("\n" + "=" * 60)
    print("✓ Visualization complete!")
    print("=" * 60)
    print("\nThis shows:")
    print("  - Each player has their own skill heatmap")
    print("  - Skills vary by player (sampled per episode)")
    print("  - Differences are spatially visible")
    print("  - CNN can learn player-specific strategies")


if __name__ == "__main__":
    main()

