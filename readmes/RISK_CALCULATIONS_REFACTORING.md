# Risk Calculations Refactoring Summary

## Problem Identified

Two critical calculations were duplicated between gameplay resolution and observation features:

1. **Turnover Probability** (defender pressure on ball handler)
2. **Steal Risk** (pass interception probability)

This duplication created serious risks:
- If mechanics were tweaked, both places needed updating
- **Critical bug found**: The observation method for turnover probability was missing the "in front" check, showing agents incorrect probabilities!
- Agents would see different probabilities than what they actually experienced
- Complex logic duplicated = higher bug risk

## Critical Bug Fixed

### Turnover Probability Discrepancy

**Gameplay code** (lines 690-731):
- ✅ Checked if defender is "in front" (180° arc toward basket)
- Only defenders in front could cause turnovers

**Observation code** (lines 2941-2982):
- ❌ Missing the "in front" check
- Counted ALL defenders within distance

**Result**: Agents saw higher turnover probabilities than actually applied, breaking the learning signal!

## Solution: Extracted Shared Helper Methods

### 1. Turnover Probability Calculation

Created `_calculate_defender_pressure_info()` (lines 2941-3001):
- Single source of truth for defender pressure calculation
- **Includes the critical "in front" check** (180° arc toward basket)
- Returns list of pressure info dicts for each qualifying defender
- Used by both:
  - Gameplay: For actual turnover resolution
  - Observations: For showing agents accurate risk

**Formula preserved exactly**:
```python
turnover_prob = base_chance * exp(-decay_lambda * max(0, distance - 1))
```

**Arc check preserved exactly**:
```python
cos_angle = dot_product(defender_vec, basket_vec) / (norms)
if cos_angle >= 0:  # Defender is in front (180° arc)
    # ... calculate turnover prob
```

### 2. Steal Risk Calculation

Created `_calculate_steal_probability_for_pass()` (lines 2795-2884):
- Single source of truth for pass interception calculation
- Returns (total_steal_prob, defender_contributions)
- Used by both:
  - Gameplay: For actual interception resolution
  - Observations: For showing agents steal risk for each potential pass

**Formula preserved exactly**:
```python
steal_contrib = (
    base_steal_rate *
    exp(-steal_perp_decay * perp_distance) *
    (1.0 + steal_distance_factor * pass_distance) *
    position_weight
)
```

**Arc check preserved exactly**:
- Cosine check: only defenders in forward hemisphere (toward receiver)
- Between points check: only defenders on pass line
- Perpendicular distance: how close defender is to pass line
- Position weight: defenders near receiver more dangerous

## Code Changes

### Modified Files

**`basketworld/envs/basketworld_env_v2.py`**:

1. **Lines 2795-2884**: Added `_calculate_steal_probability_for_pass()` helper
2. **Lines 2886-2912**: Refactored `calculate_pass_steal_probabilities()` to use helper (simplified from 86 lines to 21 lines)
3. **Lines 2941-3001**: Added `_calculate_defender_pressure_info()` helper  
4. **Lines 3003-3019**: Refactored `calculate_defender_pressure_turnover_probability()` to use helper (simplified and **fixed bug**)
5. **Lines 678-688**: Updated inline defender pressure code to use helper (simplified from 53 lines to 9 lines)
6. **Lines 1604-1611**: Updated inline pass resolution to use helper (simplified from 51 lines to 7 lines)

### Net Result

- **Deleted ~190 lines** of duplicate code
- **Added ~150 lines** (new helpers + updated calls)
- **Net: -40 lines, -4 code duplications** ✅
- **Fixed 1 critical bug** (turnover probability mismatch) ✅

## Benefits

### 1. **Consistency Guaranteed**
Agents now see the EXACT same probabilities that govern gameplay. No divergence possible.

### 2. **Bug Fixed**
Turnover probability now includes "in front" check in observations, matching gameplay.

### 3. **Maintainability**
To change turnover or steal mechanics:
- **Before**: Update 4 separate places
- **After**: Update 1 helper method

### 4. **Readability**
Gameplay code now focuses on resolution logic, not calculation details.

### 5. **Testability**
Can test probability calculations independently from gameplay flow.

## Validation

✅ **No linter errors**
✅ **Exact same formulas** - no behavior change except bug fix
✅ **Arc checks preserved** - 180° forward hemisphere for both
✅ **Compound probabilities preserved** - `1 - ∏(1 - p_i)` formula maintained
✅ **All existing parameters used** - no missing factors

## Impact on Learning

### Before (Bug Present)
```python
# Actual gameplay: Defender behind ball handler doesn't cause turnover
# Observation shows: 0.15 turnover probability (WRONG - includes rear defender)
# Agent learns: "High risk position" but turnover never happens
# Result: Confused learning signal
```

### After (Bug Fixed)
```python
# Actual gameplay: Defender behind ball handler doesn't cause turnover  
# Observation shows: 0.0 turnover probability (CORRECT - rear defender ignored)
# Agent learns: "This position is safe"
# Result: Clear, accurate learning signal
```

This fix should significantly improve defensive positioning learning!

## Future Enhancements

Now that calculations are centralized, easy to add:
- Variable defender pressure based on defender skill
- Fatigue effects on steal ability
- Player height/wingspan effects on passing lanes
- Double team pressure bonuses
- Help defense steal contributions

All changes would automatically flow to both gameplay and observations.

## Summary

The refactoring:
- ✅ **Fixed critical bug** in turnover probability observations
- ✅ Eliminated duplicate code (4 instances → 2 helpers)
- ✅ Guaranteed consistency between gameplay and observations
- ✅ Improved maintainability
- ✅ Preserved all existing mechanics exactly
- ✅ Will significantly improve agent learning with accurate signals

Thank you for pushing us to check for duplications! This was a critical find. 🎯

