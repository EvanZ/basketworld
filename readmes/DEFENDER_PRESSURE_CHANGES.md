# Defender Pressure Mechanic Changes

## Summary

Modified the defender pressure mechanic to be more realistic by:
1. **Exponential decay**: Turnover probability now decays exponentially with distance instead of using a hard threshold
2. **Directional awareness**: Defender pressure only applies when the defender is in front of the ball handler (180° arc toward basket)

## Changes Made

### 1. Core Environment (`basketworld/envs/basketworld_env_v2.py`)

**New Parameter:**
- `defender_pressure_decay_lambda` (float, default=1.0): Controls the exponential decay rate of turnover probability with distance

**Modified Logic (lines 646-776):**
- Calculate direction from ball handler to basket (assumed facing direction)
- For each nearby defender (within `defender_pressure_distance`):
  - Use dot product to check if defender is in front (180° arc toward basket)
  - Calculate turnover probability with exponential decay:
    ```python
    turnover_prob = base_chance * exp(-lambda * distance)
    ```
  - Only apply pressure if defender is in front AND random roll succeeds

**Key Implementation Details:**
- Uses cartesian coordinate conversion (`_axial_to_cartesian`) for accurate angle calculations
- Dot product `cos(angle) >= 0` means defender is within [-90°, 90°] arc (i.e., in front)
- At distance=0, probability equals baseline; as distance increases, probability decays exponentially

### 2. Training Configuration (`basketworld/utils/mlflow_params.py`)

Added parameter parsing for:
- `defender_pressure_decay_lambda` / `defender-pressure-decay-lambda`
- Default value: 1.0

### 3. App Backend (`app/backend/main.py`)

Added parameter exposure in game state serialization:
- `defender_pressure_decay_lambda` included in environment configuration

## Example Behavior

With default settings:
- `defender_pressure_distance = 1` (max range)
- `defender_pressure_turnover_chance = 0.05` (baseline at distance=1, i.e., adjacent)
- `defender_pressure_decay_lambda = 1.0` (decay rate)

Turnover probabilities by distance:
- Distance 1 (adjacent): 0.0500 (5.0%) - full baseline
- Distance 2: 0.0184 (1.84%)
- Distance 3: 0.0068 (0.68%)

**Important:** These probabilities only apply if the defender is in front of the ball handler (between handler and basket).

## Configuration

To adjust the mechanic during training, use MLflow parameters:
```bash
--defender-pressure-distance 2 \
--defender-pressure-turnover-chance 0.08 \
--defender-pressure-decay-lambda 0.5
```

Lower `lambda` = slower decay (pressure effective over longer distances)
Higher `lambda` = faster decay (pressure only effective very close)

## Backward Compatibility

All changes are backward compatible:
- Default `defender_pressure_decay_lambda=1.0` provides reasonable behavior
- Existing configurations will work without modification
- The directional check is automatic and always enabled

