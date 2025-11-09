# Observation Space Migration Summary

## ✅ Migration Complete

Your environment has been successfully migrated from **egocentric (ball-handler-relative) coordinates** to **absolute court coordinates**.

---

## What You Asked For

> "I think we should try going back to absolute positions (not egocentric relative to ball handler) in our observation_space."

**Status:** ✅ **Done**

---

## Changes Made

### 1. Core Environment File
**File:** `basketworld/envs/basketworld_env_v2.py`

**Modified function:** `_get_observation()` (lines 2300-2344)

**Changes:**
- ❌ Removed egocentric centering on ball handler
- ❌ Removed 6-way rotation logic to face basket
- ✅ Now uses absolute court coordinates for all positions
- ✅ Added explicit ball handler position (new feature)
- ✅ Simplified observation construction

**Code removed:** ~28 lines of centering/rotation logic
**Code added:** ~16 lines of absolute coordinate logic

### 2. Observation Structure Changes

| Feature | Before | After | Impact |
|---------|--------|-------|--------|
| Player positions | `(q - q_ball, r - r_ball)` | `(q, r)` absolute | Enables position-aware learning |
| Ball handler pos | Implicit (at origin) | Explicit feature | New signal for court region |
| Hoop vector | `(q_hoop - q_ball, rotated)` | `(q_hoop, r_hoop)` absolute | Simpler, explicit basket location |
| Rotation | Applied (6-way) | None | No longer needed |
| **Total elements** | 54 (3v3) | 56 (3v3) | +2 new elements |

### 3. Documentation Updated
**File:** `readmes/observation_space.md`

**Updates:**
- ✅ Changed design principles (egocentric → absolute)
- ✅ Added explicit Ball Handler Position section
- ✅ Updated component descriptions (A→K)
- ✅ Updated size calculation example
- ✅ Added "Notes on Architecture Change" section
- ✅ Documented deprecated flags

### 4. Training Script Updated
**File:** `train/train.py`

**Updates:**
- ✅ Marked `--use-egocentric-obs` as deprecated
- ✅ Marked `--egocentric-rotate-to-hoop` as deprecated
- ✅ Updated help text for `--include-hoop-vector`
- ✅ Maintained backward compatibility (flags still accepted but ignored)

### 5. Documentation Created
**New files:**
- `ABSOLUTE_COORDINATES_MIGRATION.md` - Complete migration guide
- `MIGRATION_SUMMARY.md` - This summary
- `test_absolute_obs.py` - Test script for verification

---

## Key Benefits

### 🎯 Location-Aware Learning
The network can now learn that:
- **Center court** = more space, more drive options
- **Sidelines** = constrained, tighter spacing
- **Different regions** → different optimal strategies

### 🎯 Realistic Basketball
Models the actual game where position matters:
- Fast breaks vs. sideline drives differ
- Spacing options vary by court location
- Pressure dynamics change with position

### 🎯 Explicit Features
Ball handler position is now:
- Clearly encoded as a standalone feature
- Not hidden in coordinate transformations
- Available for network to use directly

---

## Important Notes for Training

### ⚠️ Models Must Be Retrained

**Old trained models WILL NOT work with the new observation space.**

**Reason:** 
- Old models expect 54-element observations (3v3)
- New observations have 56 elements (3v3)
- Coordinate values and ranges have changed

**Action required:** Train models from scratch

### ✅ Backward Compatibility

The deprecated flags are still accepted:
```bash
# Old command format still works (flags are ignored)
python train.py --use-egocentric-obs=true --egocentric-rotate-to-hoop=true ...

# Preferred new format (no deprecated flags)
python train.py ...
```

---

## Technical Details

### Observation Vector Structure (3v3 with hoop vector)

```
[0-11]    : Player positions (absolute, 6 players × 2)
[12-17]   : Ball holder one-hot (6 elements)
[18]      : Shot clock (1 element)
[19-20]   : Ball handler position (absolute, NEW!)
[21-22]   : Hoop position (absolute, 2 elements)
[23-31]   : All-pairs offense-defense distances (9 elements)
[32-40]   : All-pairs offense-defense angles (9 elements)
[41-46]   : Lane step counts (6 elements)
[47-49]   : Expected Points (3 elements)
[50-52]   : Turnover probabilities (3 elements)
[53-55]   : Steal risks (3 elements)

Total: 56 elements
```

### Value Ranges (Normalized)

```
Player positions:     [-1, +1] (normalized by max(court_width, court_height))
Ball handler pos:     [-1, +1] (same normalization)
Hoop position:        [-1, +1] (same normalization)
Ball holder one-hot:  0 or 1
Shot clock:           [0, 24] (unnormalized)
Distances:            [0, ~1] (after normalization)
Angles:               [-1, +1] (cosines)
Lane steps:           [0, 3] (unnormalized)
EP values:            [0, 3] (pressure-adjusted)
Turnover prob:        [0, 1] (probability)
Steal risks:          [0, 1] (probability)
```

---

## Backward Compatibility Notes

### In MLFlow Params
The file `basketworld/utils/mlflow_params.py` still recognizes the deprecated flags:
- `use_egocentric_obs`
- `egocentric_rotate_to_hoop`

This ensures that old experiment records can still be loaded for reproducibility.

### The `_rotate_k60_axial()` Function
The rotation utility function is still in the code but no longer used:
- Kept for potential future use or debugging
- Can be safely ignored
- No performance impact

---

## Validation

### ✅ Code Compiles
- Environment file: `basketworld/envs/basketworld_env_v2.py` ✓
- Training script: `train/train.py` ✓
- Test script: `test_absolute_obs.py` ✓
- No syntax errors found by linter

### ✅ Observation Size
The new observation size (56 elements for 3v3) is correct and includes:
- 12 player position elements
- 6 one-hot ball holder
- 1 shot clock
- 2 ball handler position elements (NEW)
- 2 hoop vector elements
- Plus all the defensive, EP, and risk features (unchanged)

### ✅ Ready for Training
All files have been updated and are ready for:
- Training new models from scratch
- Testing the absolute coordinate system
- Generating position-aware policies

---

## Next Steps

1. **Train new models** with the updated environment
2. **Monitor convergence** - note that position-aware learning may take longer
3. **Evaluate policies** - look for evidence of location-dependent strategies
4. **Compare results** - see if absolute coordinates improve performance

---

## Questions or Issues?

Refer to:
- `ABSOLUTE_COORDINATES_MIGRATION.md` - Full migration guide
- `readmes/observation_space.md` - Complete observation space documentation
- `basketworld/envs/basketworld_env_v2.py` - Source code

---

**Migration completed:** November 6, 2025
**Files modified:** 3 main files + 3 documentation files
**Status:** ✅ Ready for training



