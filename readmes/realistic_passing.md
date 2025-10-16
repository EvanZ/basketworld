# Realistic Passing Mechanics

## Overview

This document describes the line-of-sight based passing system implemented in BasketWorld. The system makes passing more realistic by considering defender positioning along the pass line and the distance of the pass.

## Key Features

- **Arc-based action selection preserved** - No changes to action space; players still pass in one of 6 hexagonal directions
- **180° forward hemisphere filtering** - Only defenders in the forward half-plane (chosen direction ± 1 adjacent) can intercept; defenders behind cannot
- **Line-of-sight steal mechanics** - Defenders positioned between passer and receiver can intercept based on their proximity to the pass line
- **Open passes guaranteed** - If no defenders are between passer and receiver in the forward hemisphere, the pass always succeeds
- **Distance-dependent risk** - Longer passes are inherently more risky
- **Multiple defender compounding** - Multiple defenders near the pass line compound steal probabilities
- **Detailed tracking** - Full diagnostic information available for analysis and debugging

## How It Works

### 1. Pass Target Selection (Unchanged)

When a player chooses a pass action:
1. A configurable arc (default 60°) is centered on the chosen direction
2. All teammates within this arc are eligible receivers
3. The **nearest teammate** in the arc becomes the target receiver
4. If no teammate is in the arc, the pass goes out of bounds (with configurable turnover probability)

### 2. Line-of-Sight Steal Evaluation (New)

Once a valid receiver is identified, the system evaluates potential interceptions:

#### Defender Eligibility
Only defenders that meet ALL of these criteria are evaluated:
- Defender is in the **180° forward hemisphere** relative to the pass direction
  - This is the chosen direction plus its two adjacent hex directions
  - Example: PASS_NW checks defenders in W, NW, and NE directions
  - Defenders behind the passer (in rear hemisphere) cannot intercept
- Defender is **between** the passer and receiver along the pass line
  - Mathematically: the defender's projection onto the pass line has parameter `0 < t < 1`
  - Defenders behind the passer or past the receiver are ignored

#### Steal Contribution Calculation

For each eligible defender, the steal contribution is calculated using:

```
steal_contribution = base_steal_rate × exp(-steal_perp_decay × perp_distance) × (1 + steal_distance_factor × pass_distance)
```

Where:
- `base_steal_rate` - Maximum steal probability when defender is directly on the line (default: 0.35)
- `perp_distance` - Perpendicular distance from defender to the pass line (in Cartesian space)
- `steal_perp_decay` - Exponential decay rate (default: 1.5)
- `pass_distance` - Total distance between passer and receiver in hex units
- `steal_distance_factor` - Linear factor for distance effect (default: 0.08)

**Key insights**: 
- The 180° arc filter ensures only defenders in the forward hemisphere are considered (defenders behind cannot intercept)
- The exponential decay with perpendicular distance means defenders far from the pass line contribute negligibly
- This creates realistic passing lanes where positioning matters strategically

#### Compounding Multiple Defenders

When multiple defenders are between passer and receiver, their steal probabilities compound:

```
total_steal_prob = 1 - ∏(1 - steal_i) for all eligible defenders
```

This ensures that:
- With no defenders between passer/receiver: `total_steal_prob ≈ 0` → pass always succeeds ✓
- With one defender on line: `total_steal_prob = steal_1`
- With multiple defenders: probability is higher than any individual contribution

#### Interception Resolution

If `random() < total_steal_prob`:
- Interception occurs
- The defender with the **highest steal contribution** receives the ball
- Turnover is recorded with full diagnostic information

Otherwise:
- Pass succeeds
- Ball transfers to intended receiver
- Assist window begins (if applicable)

## Parameters

### New Pass Interception Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `base_steal_rate` | 0.35 | Maximum steal probability when defender is directly on the pass line (perpendicular distance = 0) |
| `steal_perp_decay` | 1.5 | Controls how quickly steal probability decays as defender moves perpendicular to pass line. Higher values = faster decay. |
| `steal_distance_factor` | 0.08 | Linear increase in steal chance per hex of pass distance. e.g., 0.08 means 8% additional chance per hex. |

### Existing Pass Parameters (Unchanged)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pass_arc_degrees` | 60.0 | Width of arc for determining eligible receivers |
| `pass_oob_turnover_prob` | 1.0 | Probability that pass with no receiver results in turnover vs NOOP |
| `enable_pass_gating` | True | Whether to mask pass actions that have no eligible receivers |

## Example Scenarios

### Scenario 1: Open Pass
```
Passer: (0, 0)
Pass Direction: PASS_E
Receiver: (4, 0)
Defenders: None in forward hemisphere between passer and receiver

Result: Pass always succeeds (total_steal_prob ≈ 0)
```

### Scenario 1b: Defender Behind Passer (180° Arc Filter)
```
Passer: (0, 0)
Pass Direction: PASS_E (East)
Receiver: (4, 0)
Defender: (-2, 0) [directly behind passer, on the line if extended]

Calculation:
- Defender is in rear hemisphere (not in 180° forward arc)
- Defender is filtered out before line-of-sight check
- total_steal_prob = 0

Result: Pass always succeeds - defenders behind cannot intercept
```

### Scenario 2: Defender Directly On Line
```
Passer: (0, 0)
Receiver: (4, 0)  
Defender: (2, 0) [on the line, pass_distance = 4]

Calculation:
- perp_distance = 0
- steal_contrib = 0.35 × exp(0) × (1 + 0.08 × 4)
                = 0.35 × 1.0 × 1.32
                = 0.462 (46.2%)

Result: 46.2% chance of interception
```

### Scenario 3: Defender One Hex Off Line
```
Passer: (0, 0)
Receiver: (4, 0)
Defender: (2, 1) [one hex off line, pass_distance = 4]

Calculation:
- perp_distance ≈ 1.0 (in Cartesian space)
- steal_contrib = 0.35 × exp(-1.5 × 1.0) × 1.32
                = 0.35 × 0.223 × 1.32
                = 0.103 (10.3%)

Result: 10.3% chance of interception
```

### Scenario 4: Multiple Defenders
```
Passer: (0, 0)
Receiver: (6, 0)
Defender 1: (2, 0) [on line] → contrib = 0.51
Defender 2: (4, 1) [one hex off] → contrib = 0.12

Calculation:
- total_steal_prob = 1 - (1 - 0.51) × (1 - 0.12)
                   = 1 - 0.49 × 0.88
                   = 1 - 0.431
                   = 0.569 (56.9%)

Result: 56.9% chance of interception (highest contributor gets ball)
```

### Scenario 5: Longer Pass
```
Passer: (0, 0)
Receiver: (8, 0)
Defender: (4, 0) [on line, pass_distance = 8]

Calculation:
- perp_distance = 0
- steal_contrib = 0.35 × exp(0) × (1 + 0.08 × 8)
                = 0.35 × 1.0 × 1.64
                = 0.574 (57.4%)

Result: 57.4% chance of interception (vs 46% for distance 4)
```

## Steal Probability Reference Table

With default parameters (`base_steal_rate=0.35`, `steal_perp_decay=1.5`, `steal_distance_factor=0.08`):

| Perpendicular Distance | Pass Distance 2 | Pass Distance 4 | Pass Distance 6 | Pass Distance 8 |
|------------------------|-----------------|-----------------|-----------------|-----------------|
| 0.0 (on line)          | 40.9%           | 46.2%           | 51.5%           | 57.4%           |
| 0.5                    | 26.9%           | 30.4%           | 33.9%           | 37.8%           |
| 1.0                    | 9.1%            | 10.3%           | 11.5%           | 12.8%           |
| 1.5                    | 3.1%            | 3.5%            | 3.9%            | 4.3%            |
| 2.0                    | 1.0%            | 1.2%            | 1.3%            | 1.5%            |

**Key observations**:
- Steal probability drops dramatically as defenders move away from the pass line
- Longer passes are consistently riskier
- Defenders more than 2 hexes from the line have minimal impact (<2%)

## Implementation Details

### Geometry Helper Methods

Two new helper methods handle the geometric calculations:

#### `_point_to_line_distance(point, line_start, line_end) -> float`
Calculates the perpendicular distance from a point to a line segment in Cartesian space.

**Algorithm**:
1. Convert axial coordinates to Cartesian using `_axial_to_cartesian()`
2. Project point onto line: `t = [(P-S) · (E-S)] / |E-S|²`
3. Clamp `t` to `[0, 1]` to stay within line segment
4. Calculate distance from point to closest point on segment

**Returns**: Distance in Cartesian units (corresponding to hexagon geometry)

#### `_is_between_points(point, line_start, line_end) -> bool`
Checks if a point's projection onto the line falls between the start and end points.

**Algorithm**:
1. Convert axial coordinates to Cartesian
2. Project point onto line: `t = [(P-S) · (E-S)] / |E-S|²`
3. Check if `0 < t < 1` (strictly between, not including endpoints)

**Returns**: `True` if defender is between passer and receiver, `False` otherwise

### Updated Pass Results Tracking

The `results["passes"][passer_id]` dictionary now includes:

```python
{
    "success": bool,                    # Whether pass succeeded
    "target": int,                      # Receiver ID (if applicable)
    "pass_distance": float,             # Hex distance between passer and receiver
    "total_steal_prob": float,          # Computed steal probability [0, 1]
    "defenders_evaluated": [            # List of defenders considered
        {
            "id": int,                  # Defender ID
            "steal_contribution": float, # Individual steal probability
            "perp_distance": float      # Distance from pass line
        },
        ...
    ],
    "reason": str,                      # "completed", "intercepted", "out_of_bounds", etc.
    "interceptor_id": int               # (Only if intercepted)
}
```

This detailed tracking enables:
- Post-game analysis of passing effectiveness
- Debugging and parameter tuning
- Training reward shaping based on pass difficulty
- Visualization of dangerous passes

## Usage Examples

### Basic Environment Setup

```python
from basketworld.envs.basketworld_env_v2 import HexagonBasketballEnv

# Using default parameters
env = HexagonBasketballEnv(
    grid_size=16,
    players=3,
    seed=42
)

# Custom passing parameters
env = HexagonBasketballEnv(
    grid_size=16,
    players=3,
    seed=42,
    base_steal_rate=0.4,          # More aggressive steals
    steal_perp_decay=2.0,         # Faster decay (tighter defense)
    steal_distance_factor=0.1     # Higher distance penalty
)
```

### Analyzing Pass Results

```python
obs, _ = env.reset()
actions = {0: 8, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}  # Player 0 passes East
obs, reward, done, truncated, info = env.step(actions)

# Check pass outcome
if 0 in info["passes"]:
    pass_info = info["passes"][0]
    print(f"Pass success: {pass_info['success']}")
    print(f"Pass distance: {pass_info['pass_distance']} hexes")
    print(f"Steal probability: {pass_info['total_steal_prob']:.1%}")
    
    # Analyze defenders
    for defender in pass_info['defenders_evaluated']:
        print(f"  Defender {defender['id']}: "
              f"{defender['steal_contribution']:.1%} contribution, "
              f"{defender['perp_distance']:.2f} from line")
```

### Training with Custom Parameters

```bash
python train/train.py \
  --players 3 \
  --base-steal-rate 0.4 \
  --steal-perp-decay 1.8 \
  --steal-distance-factor 0.1 \
  --pass-arc-degrees 60 \
  --learning-rate 3e-4 \
  # ... other training args
```

### Loading Parameters from MLflow

The system automatically reads pass parameters from MLflow runs:

```python
from basketworld.utils.mlflow_params import load_env_from_mlflow_params

# Load environment with parameters from previous run
env = load_env_from_mlflow_params(mlflow_params_dict)
```

Supported parameter names (both underscore and hyphen versions):
- `base_steal_rate` / `base-steal-rate`
- `steal_perp_decay` / `steal-perp-decay`  
- `steal_distance_factor` / `steal-distance-factor`

## Tuning Guidelines

### Making Passes Easier (More Offensive)

- **Decrease `base_steal_rate`** (e.g., 0.25) - Lower maximum steal chance
- **Increase `steal_perp_decay`** (e.g., 2.0) - Steals only work if defender is very close to line
- **Decrease `steal_distance_factor`** (e.g., 0.05) - Less penalty for long passes

### Making Passes Harder (More Defensive)

- **Increase `base_steal_rate`** (e.g., 0.45) - Higher maximum steal chance
- **Decrease `steal_perp_decay`** (e.g., 1.0) - Defenders can intercept from further away
- **Increase `steal_distance_factor`** (e.g., 0.12) - Stronger penalty for long passes

### Training Curriculum Approach

For reinforcement learning, consider gradually increasing difficulty:

```python
# Early training (episode 0-1000)
env.base_steal_rate = 0.25
env.steal_perp_decay = 2.0
env.steal_distance_factor = 0.05

# Mid training (episode 1000-5000)
env.base_steal_rate = 0.30
env.steal_perp_decay = 1.7
env.steal_distance_factor = 0.07

# Late training (episode 5000+)
env.base_steal_rate = 0.35  # Default
env.steal_perp_decay = 1.5  # Default
env.steal_distance_factor = 0.08  # Default
```

## Hexagonal Direction System and 180° Arc

BasketWorld uses a hexagonal grid with 6 primary directions:
- 0: **E** (East)
- 1: **NE** (Northeast)  
- 2: **NW** (Northwest)
- 3: **W** (West)
- 4: **SW** (Southwest)
- 5: **SE** (Southeast)

### 180° Forward Hemisphere

When you pass in a direction, defenders in the **180° forward arc** are checked:
- **Chosen direction** ± **1 adjacent direction** = 3 directions total
- This forms a forward-facing hemisphere

**Examples**:
- **PASS_E** (direction 0): Checks defenders in **SE (5), E (0), NE (1)**
- **PASS_NW** (direction 2): Checks defenders in **NE (1), NW (2), W (3)**
- **PASS_SW** (direction 4): Checks defenders in **W (3), SW (4), SE (5)**

**Rationale**:
- Physically realistic: you can only intercept passes coming toward you
- Strategically meaningful: passing away from defenders works
- Avoids asymmetries: consistent defensive coverage

## Comparison with Previous System

### Old System (Removed)
- Fixed `steal_chance` parameter (default 5%)
- Applied if ANY defender in arc was closer than receiver
- No consideration of defender positioning relative to pass line
- No distance-based risk
- No directional filtering

### New System
- Three parameters controlling different aspects of steal probability
- 180° arc filter: only defenders in forward hemisphere considered
- Only defenders **between** passer and receiver are evaluated
- Steal chance based on perpendicular distance from pass line
- Distance-based risk modeling
- Multiple defenders compound realistically
- Open passes guaranteed to succeed

**Result**: More realistic passing where positioning matters, longer passes are riskier, passing away from defenders works, and players can find open passing lanes.

## Integration with Other Systems

### Shot Pressure System
The passing system uses similar geometric principles to the shot pressure system (`_compute_shot_pressure_multiplier`):
- Both use Cartesian coordinates for accurate geometry
- Both use arc-based eligibility
- Both use exponential decay for distance effects

### Assist System
The assist tracking system (`_assist_candidate`) is unchanged and works seamlessly:
- Successful passes still trigger assist windows
- Assists are awarded when receiver scores within window
- Percentage-based assist rewards work with new system

### Action Masking
Pass gating (`enable_pass_gating`) is unchanged:
- Passes with no eligible receivers in arc are masked
- Based on `_has_teammate_in_pass_arc()` helper
- Prevents wasted actions and speeds training

## Testing and Validation

### Unit Tests
Test the geometry helpers:

```python
env = HexagonBasketballEnv(grid_size=16, players=2, seed=42)

# Test perpendicular distance
dist = env._point_to_line_distance((2, 0), (0, 0), (4, 0))
assert dist < 0.1, "Defender on line should have ~0 distance"

dist = env._point_to_line_distance((2, 1), (0, 0), (4, 0))
assert dist > 0.5, "Defender off line should have >0 distance"

# Test between-ness
assert env._is_between_points((2, 0), (0, 0), (4, 0)) == True
assert env._is_between_points((-1, 0), (0, 0), (4, 0)) == False
assert env._is_between_points((5, 0), (0, 0), (4, 0)) == False
```

### Integration Tests
Test pass scenarios:

```python
# Open pass should always succeed
env = HexagonBasketballEnv(
    grid_size=16, players=2, seed=42,
    initial_positions=[(0,0), (4,0), (10,10), (12,12)],
    initial_ball_holder=0
)

successes = 0
for trial in range(10):
    obs, _ = env.reset(seed=42 + trial)
    obs, _, _, _, info = env.step({0: 8, 1: 0, 2: 0, 3: 0})
    if info["passes"][0]["success"]:
        successes += 1

assert successes == 10, "Open passes should always succeed"
```

## Future Enhancements

Potential improvements to consider:

1. **Velocity-based stealing** - Account for defender momentum/direction
2. **Passer skill ratings** - Different players have different passing abilities
3. **Receiver skill ratings** - Some players are better at catching contested passes
4. **Pass trajectory types** - Bounce pass, lob pass, chest pass with different properties
5. **Fatigue effects** - Tired players make worse passes
6. **Weather/court conditions** - Environmental factors affecting pass accuracy

## Migration Notes

### Updating Existing Code

If you have existing code using `steal_chance`:

**Before**:
```python
env = HexagonBasketballEnv(..., steal_chance=0.05)
```

**After**:
```python
env = HexagonBasketballEnv(
    ...,
    base_steal_rate=0.35,
    steal_perp_decay=1.5,
    steal_distance_factor=0.08
)
```

### Parameter Equivalence

To roughly match the old 5% steal chance behavior:
```python
base_steal_rate=0.05,       # Low base rate
steal_perp_decay=0.5,       # Slow decay (defenders far away still contribute)
steal_distance_factor=0.0   # No distance effect
```

However, we **recommend using the new defaults** for more realistic gameplay.

## References

- Source file: `basketworld/envs/basketworld_env_v2.py`
- Related documentation:
  - `readmes/pass_actions.md` - Arc-based pass action system
  - `readmes/action_masking.md` - Pass gating and action masking
  - `readmes/observation_space.md` - How pass results affect observations

## Changelog

- **2025-10-16**: Initial implementation of line-of-sight based passing
  - Added `base_steal_rate`, `steal_perp_decay`, `steal_distance_factor` parameters
  - Added `_point_to_line_distance()` and `_is_between_points()` geometry helpers
  - Completely rewrote `_attempt_pass()` interception logic with 180° arc filtering
  - Only defenders in forward hemisphere (chosen direction ± 1) can intercept
  - Removed legacy `steal_chance` parameter
  - Updated all dependent files (train.py, mlflow_params.py, backend API, frontend UI)

