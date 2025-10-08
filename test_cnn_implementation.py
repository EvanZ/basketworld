#!/usr/bin/env python3
"""
Quick test script to verify CNN implementation is working correctly.
Tests spatial observation generation and CNN feature extractor.
"""
import numpy as np
import torch as th
from basketworld.envs.basketworld_env_v2 import HexagonBasketballEnv, Team
from basketworld.utils.cnn_feature_extractor import HexCNNFeatureExtractor


def test_spatial_observation():
    """Test that spatial observations are generated correctly."""
    print("=" * 60)
    print("Testing Spatial Observation Generation")
    print("=" * 60)
    
    # Create environment with spatial obs enabled
    env = HexagonBasketballEnv(
        grid_size=12,
        players=2,
        use_spatial_obs=True,
        seed=42,
    )
    
    obs, info = env.reset()
    
    # Check that spatial key exists
    assert "spatial" in obs, "Spatial observation not in obs dict"
    
    spatial = obs["spatial"]
    print(f"✓ Spatial observation shape: {spatial.shape}")
    
    # Verify shape (5 channels, height, width)
    expected_shape = (5, env.court_height, env.court_width)
    assert spatial.shape == expected_shape, f"Expected {expected_shape}, got {spatial.shape}"
    print(f"✓ Expected shape matches: {expected_shape}")
    
    # Verify channel contents
    print("\nChannel verification:")
    
    # Channel 0: Players (should have +1 and -1 values)
    players_channel = spatial[0]
    offense_count = np.sum(players_channel == 1.0)
    defense_count = np.sum(players_channel == -1.0)
    print(f"  Channel 0 (Players): {offense_count} offense, {defense_count} defense")
    assert offense_count == env.players_per_side, f"Expected {env.players_per_side} offense, got {offense_count}"
    assert defense_count == env.players_per_side, f"Expected {env.players_per_side} defense, got {defense_count}"
    
    # Channel 1: Ball holder (should have exactly 1 cell with value 1.0)
    ball_channel = spatial[1]
    ball_count = np.sum(ball_channel == 1.0)
    print(f"  Channel 1 (Ball): {ball_count} cell(s) marked")
    assert ball_count == 1, f"Expected 1 ball holder, got {ball_count}"
    
    # Channel 2: Basket (should have exactly 1 cell with value 1.0)
    basket_channel = spatial[2]
    basket_count = np.sum(basket_channel == 1.0)
    print(f"  Channel 2 (Basket): {basket_count} cell(s) marked")
    assert basket_count == 1, f"Expected 1 basket, got {basket_count}"
    
    # Channel 3: Expected points (should have some positive values if ball holder is offense)
    ep_channel = spatial[3]
    ep_max = np.max(ep_channel)
    ep_min = np.min(ep_channel)
    print(f"  Channel 3 (Expected Points): min={ep_min:.3f}, max={ep_max:.3f}")
    assert ep_max > 0, "Expected points should have positive values"
    assert ep_max <= 3.0, f"Expected points max should be <= 3.0, got {ep_max}"
    
    # Channel 4: Three-point arc (should have 1.0 for cells at/outside arc)
    arc_channel = spatial[4]
    arc_count = np.sum(arc_channel == 1.0)
    print(f"  Channel 4 (3PT Arc): {arc_count} cells at/outside arc")
    assert arc_count > 0, "Expected some cells marked as outside 3PT arc"
    
    print("\n✓ All spatial observation tests passed!")
    
    # Test that observations persist through steps
    actions = env.action_space.sample()
    obs, rewards, done, truncated, info = env.step(actions)
    assert "spatial" in obs, "Spatial observation not in obs after step"
    print("✓ Spatial observation persists after env.step()")
    
    return env


def test_cnn_feature_extractor():
    """Test that CNN feature extractor works correctly."""
    print("\n" + "=" * 60)
    print("Testing CNN Feature Extractor")
    print("=" * 60)
    
    # Create environment with spatial obs
    env = HexagonBasketballEnv(
        grid_size=12,
        players=2,
        use_spatial_obs=True,
        seed=42,
    )
    
    obs, info = env.reset()
    
    # Create feature extractor
    extractor = HexCNNFeatureExtractor(
        observation_space=env.observation_space,
        cnn_features_dim=256,
        mlp_features_dim=128,
        cnn_channels=(32, 64, 64),
    )
    
    print(f"✓ Feature extractor created")
    print(f"  Output dimension: {extractor.features_dim}")
    print(f"  CNN enabled: {extractor.cnn_enabled}")
    
    # Convert observation to torch tensors
    obs_tensor = {
        key: th.from_numpy(val).unsqueeze(0) if isinstance(val, np.ndarray) else val
        for key, val in obs.items()
    }
    
    # Forward pass
    features = extractor(obs_tensor)
    print(f"✓ Forward pass successful")
    print(f"  Output shape: {features.shape}")
    
    expected_dim = 256 + 128  # cnn_features_dim + mlp_features_dim
    assert features.shape == (1, expected_dim), f"Expected shape (1, {expected_dim}), got {features.shape}"
    print(f"✓ Output dimension matches expected: {expected_dim}")
    
    # Test with batch
    batch_size = 4
    obs_batch = {
        key: th.from_numpy(np.stack([val] * batch_size)) if isinstance(val, np.ndarray) else val
        for key, val in obs.items()
    }
    features_batch = extractor(obs_batch)
    print(f"✓ Batch forward pass successful")
    print(f"  Batch output shape: {features_batch.shape}")
    assert features_batch.shape == (batch_size, expected_dim), f"Expected shape ({batch_size}, {expected_dim})"
    
    print("\n✓ All CNN feature extractor tests passed!")


def test_without_spatial_obs():
    """Test that everything still works without spatial obs (backward compatibility)."""
    print("\n" + "=" * 60)
    print("Testing Backward Compatibility (no spatial obs)")
    print("=" * 60)
    
    # Create environment WITHOUT spatial obs
    env = HexagonBasketballEnv(
        grid_size=12,
        players=2,
        use_spatial_obs=False,  # Disabled
        seed=42,
    )
    
    obs, info = env.reset()
    
    # Check that spatial key does NOT exist
    assert "spatial" not in obs, "Spatial observation should not be in obs when disabled"
    print("✓ Spatial observation correctly absent when disabled")
    
    # Verify other keys still exist
    expected_keys = {"obs", "action_mask", "role_flag", "skills"}
    assert set(obs.keys()) == expected_keys, f"Expected keys {expected_keys}, got {set(obs.keys())}"
    print(f"✓ All expected observation keys present: {expected_keys}")
    
    # Test step
    actions = env.action_space.sample()
    obs, rewards, done, truncated, info = env.step(actions)
    assert "spatial" not in obs, "Spatial observation should not be in obs after step"
    print("✓ Backward compatibility maintained")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("CNN IMPLEMENTATION TEST SUITE")
    print("=" * 60 + "\n")
    
    try:
        test_spatial_observation()
        test_cnn_feature_extractor()
        test_without_spatial_obs()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nYour CNN implementation is ready to use!")
        print("\nTo train with CNN:")
        print("  python train/train.py --use-spatial-obs --cnn-channels 32 64 64")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

