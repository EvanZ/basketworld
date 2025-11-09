# Observation Space Migration: Egocentric → Absolute Coordinates

## Overview

The environment has been migrated from an **egocentric (ball-handler-relative) observation space** to an **absolute court coordinate observation space**. This change enables the network to learn location-dependent basketball strategies where court position matters (center vs. sidelines).

**Migration Date:** November 6, 2025

---

## What Changed

### Core Change: Observation Coordinates

**Before (Egocentric):**
- All player positions were **relative to the ball handler**
- Coordinates were computed as: `(q_player - q_ball_handler, r_player - r_ball_handler)`
- Optional rotation to align basket direction
- Provided translation and rotation invariance

**After (Absolute):**
- All player positions use **absolute court coordinates**
- Coordinates are in the original court frame: `(q_player, r_player)`
- No rotation applied
- Allows network to distinguish court regions (center vs. sides)

### New Feature: Explicit Ball Handler Position

Added a dedicated feature encoding the absolute position of the ball handler:
- **Position in observation vector:** Index 19-20 (after shot clock)
- **Purpose:** Explicitly tells the network where on the court play is happening
- **Format:** Normalized absolute coordinates `(q, r) / max(court_width, court_height)`
- **Fallback:** Uses basket position if no ball holder (terminal states)

### Observation Structure Changes

| Component | Before | After |
|-----------|--------|-------|
| Player positions | Relative to ball handler | Absolute coordinates |
| Hoop vector | Relative to ball handler, rotated | Absolute coordinates (basket position) |
| Ball handler position | N/A | **NEW: Explicit absolute position** |
| Rotation | Applied if `egocentric_rotate_to_hoop=True` | **Not applied** |

### Observation Vector Size

**3v3 Example (with hoop vector):**

```
Before: 54 elements
- Player positions (relative): 12
- Ball holder one-hot: 6
- Shot clock: 1
- Hoop vector (rotated): 2
- [rest unchanged]

After: 56 elements (+2 new elements)
- Player positions (absolute): 12
- Ball holder one-hot: 6
- Shot clock: 1
- Ball handler position (NEW): 2  ← NEW
- Hoop vector (absolute): 2
- [rest unchanged]
```

---

## Files Modified

### Core Environment
- **`basketworld/envs/basketworld_env_v2.py`**
  - Modified `_get_observation()` method (lines 2300-2344)
  - Removed rotation logic
  - Added absolute ball handler position feature
  - Removed dependency on `egocentric_rotate_to_hoop` flag

### Documentation
- **`readmes/observation_space.md`**
  - Updated component descriptions (A-K)
  - Updated design principles
  - Updated configuration flags section
  - Updated size calculation example
  - Added notes about the architecture change

### Training Script
- **`train/train.py`**
  - Updated help text for deprecated flags
  - `--use-egocentric-obs`: Marked as deprecated (still accepted for backward compatibility)
  - `--egocentric-rotate-to-hoop`: Marked as deprecated (still accepted for backward compatibility)
  - `--include-hoop-vector`: Updated help text to reflect absolute coordinates

### Utility Files (Already Compatible)
- **`basketworld/utils/mlflow_params.py`**
  - Still recognizes deprecated flags (for experiment reproducibility)
  - No changes needed

---

## Deprecated Parameters

The following command-line flags are now **deprecated but still accepted** for backward compatibility:

- `--use-egocentric-obs`: Now ignored (observations are always absolute)
- `--egocentric-rotate-to-hoop`: Now ignored (no rotation applied)

These flags are accepted but do nothing, ensuring old training commands won't break.

---

## What This Means for Learning

### Advantages of Absolute Coordinates

1. **Location-Aware Strategies**
   - Network can learn that center court offers more space
   - Network can learn that sidelines constrain movement options
   - More realistic basketball decision-making

2. **Better Generalization on Position-Dependent Tasks**
   - Different optimal strategies for different court positions
   - Example: Kick-out spacing vs. drive spacing

3. **Explicit Position Encoding**
   - Ball handler position is now a standalone feature
   - Reduces need for implicit positional inference

4. **Real Basketball Dynamics**
   - Actual basketball has strong position-dependent strategies
   - Network can now learn these dynamics explicitly

### Trade-offs

1. **Loss of Translation Invariance**
   - Network must learn similar behaviors from different positions
   - Could require more data to learn well
   - No automatic generalization across court positions

2. **Increased State Space**
   - More unique observations for same game state
   - Slightly longer training time needed

---

## Migration Guide for Existing Code

### If You Have Custom Evaluation Scripts

**Update your environment instantiation from:**
```python
env = HexagonBasketballEnv(
    use_egocentric_obs=True,
    egocentric_rotate_to_hoop=True,
    ...
)
```

**To:**
```python
env = HexagonBasketballEnv(
    # These parameters are now deprecated and can be omitted
    ...
)
```

The new defaults will automatically use absolute coordinates.

### Policy Loading

**Important:** Policies trained with the old egocentric setup are **NOT compatible** with the new absolute coordinate system. They expect different observation shapes and value ranges.

- Old policies: `obs` shape with 54 elements (3v3)
- New policies: `obs` shape with 56 elements (3v3)

You must retrain models from scratch for the new observation space.

### Tests

A test file has been created to verify the new observation structure:
```bash
python3 test_absolute_obs.py
```

This test verifies:
- Correct observation vector size (56 for 3v3)
- All values are finite
- Ball holder one-hot sums to 1
- Observations remain consistent over multiple steps

---

## Why This Change?

The original ego-centric approach was designed for **complete translation and rotation invariance**—useful for generic policies that apply anywhere. However, real basketball is strongly position-dependent:

- **Center court:** More drive options, more spacing
- **Sidelines:** Constrained movement, tighter spacing
- **Baseline:** Different strategic options than perimeter

By using absolute coordinates, the network can now learn these crucial distinctions explicitly, leading to more realistic and intelligent play.

---

## Next Steps

1. **Retrain models** from scratch with new observation space
2. **Monitor learning curves** to ensure convergence with absolute positions
3. **Test policy quality** to verify location-dependent strategies emerge
4. **Compare against old approach** if needed for research purposes

---

## Rollback (If Needed)

To temporarily revert to egocentric coordinates:

```bash
git diff basketworld/envs/basketworld_env_v2.py  # Review changes
git checkout basketworld/envs/basketworld_env_v2.py  # Revert
```

But note that you'll also need to retrain models if you do this.

---

## Questions?

Refer to the updated `readmes/observation_space.md` for complete documentation of the new observation structure.



