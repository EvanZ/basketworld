# Offensive 3-Second Violation Implementation

## Overview

This document summarizes the implementation of the offensive 3-second violation rule in BasketWorld, as requested. This feature creates a "lane" area near the basket where offensive players cannot camp out for more than a specified number of steps.

## Feature Specifications

### Rule Definition
- **Lane Area**: Extends from the basket along the +q axis (toward the offensive side) up to (but not including) the 3-point line distance
- **Lane Width**: Symmetric on both sides of the center line (e.g., `lane_width=1` means 1 hex to the left and right)
- **Max Steps**: Default 3 steps (configurable)
- **Ball Handler Exception** (Offense only): If a player has the ball or receives it on the last permitted step, they can shoot on the next step without violation

### Violation Consequences
- **Offensive**: Results in a turnover (possession to defense), logged with reason `"offensive_three_seconds"`
  - Offense receives negative reward (turnover penalty)
  - Defense receives positive reward
  - Episode ends
- **Defensive**: Awards offense 1 point (like a technical free throw) **and ends the episode**, logged in `defensive_lane_violations`
  - Offense receives **+1.0 reward** (divided among offense players)
  - Defense receives **-1.0 reward** (divided among defense players)
  - Episode ends
- **Visual feedback**: Painted lane area shown in light red in both web UI and rendered GIFs/frames

### Updated: Defensive 3-Second Rule
The defensive 3-second (illegal defense) rule has been updated to use the **same full lane area** as the offensive rule:
- **Previously**: Defenders were only tracked at the basket hex, violations only masked NOOP
- **Now**: 
  - Defenders cannot camp anywhere in the full painted lane area
  - **Violations award offense 1 point and end the episode** (like a technical free throw)
  - Both offensive and defensive lanes are identical (same hexes)
  - Violations automatically trigger when defender exceeds max steps in lane
  - No shooting exception for defenders (unlike offense)

### Configuration
- **Disabled by default** - must be explicitly enabled
- **Shared Configuration** (used by both offensive and defensive rules):
  - `--three-second-lane-width` (int, default: 1) - Width of the lane in hexes
  - `--three-second-max-steps` (int, default: 3) - Max steps allowed in lane
- **Individual Enable Flags**:
  - `--offensive-three-seconds` (boolean, default: False) - Enable offensive rule
  - `--illegal-defense-enabled` (boolean, default: False) - Enable defensive rule

## Implementation Details

### 1. Environment Changes (`basketworld/envs/basketworld_env_v2.py`)

#### New Parameters
Added shared configuration parameters (used by both offense and defense):
- `three_second_lane_width: int = 1` - Width of the lane (shared)
- `three_second_max_steps: int = 3` - Max steps in lane (shared)

Enable flags (separate for offense and defense):
- `offensive_three_seconds_enabled: bool = False` - Enable offensive rule
- `illegal_defense_enabled: bool = True` - Enable defensive rule (existing)

#### Lane Calculation
- `_calculate_offensive_lane_hexes()`: Calculates all hexes in the painted area
  - Iterates through distances 0 to `three_point_distance - 1`
  - For each distance, includes hexes within `lane_width` perpendicular to center line
  - Returns a set of (q, r) tuples
  - **Includes the basket hex** (distance 0)
- `_calculate_defensive_lane_hexes()`: **Updated** to return the full painted area
  - Now calls `_calculate_offensive_lane_hexes()` to ensure both lanes are identical
  - Previously only returned the basket hex
  - This makes the defensive 3-second rule consistent with the offensive rule

#### Step Tracking
- `_offensive_lane_steps: Dict[int, int]`: Tracks steps in lane for each offensive player
- `_defender_in_key_steps: Dict[int, int]`: **Updated** to track steps in full lane (not just basket)
- Updated in `step()` method after action processing:
  - Increments counter if player is in lane
  - Resets to 0 if player leaves lane
  - **Changed**: Defensive tracking now checks `tuple(pos) in self.defensive_lane_hexes` instead of `tuple(pos) == self.basket_position`

#### Violation Detection
Added violation check in `_process_simultaneous_actions()` before returning:
- Checks each offensive player's lane occupancy
- Violations occur when:
  1. Player at `max_steps` without ball (must leave)
  2. Ball handler at `max_steps + 1` without shooting (must shoot or leave)
- Creates turnover with reason `"offensive_three_seconds"`

#### Action Masking
Extended `_get_action_masks()` to enforce both offensive and defensive rules:
- **Offensive** players at `max_steps` in lane without ball: NOOP masked (must move)
- **Offensive** ball handler at `max_steps + 1` in lane: Only SHOOT allowed
- **Defensive** players at `max_steps` in lane: NOOP masked (must move)
  - **Changed**: Now checks `tuple(pos) in self.defensive_lane_hexes` instead of just basket hex
  - Applies to entire lane area, not just basket

#### Observation Space
**Important**: Added lane step counts to observation for BOTH offensive and defensive players:
- Observation size increased by `n_players` (4 in a 2v2 game)
- For offensive players: tracks steps in offensive lane
- For defensive players: tracks steps in defensive lane (basket hex)
- This allows agents to learn to manage their time in the lane

**Observation structure**:
```python
base_len = (n_players * 2)      # player positions
         + n_players             # ball holder one-hot
         + 1                     # shot clock
         + players_per_side      # nearest defender distances
         + n_players             # ← NEW: lane step counts
         + (2 if hoop_vector else 0)  # hoop vector
```

### 2. Training Script Changes (`train/train.py`)

Added CLI arguments:
```python
--offensive-three-seconds                # Enable the rule
--offensive-three-second-lane-width      # Lane width (default: 1)
--offensive-three-second-max-steps       # Max steps (default: 3)
```

Updated `setup_environment()` to pass these parameters to the environment.

### 3. Backend API Changes (`app/backend/main.py`)

Updated `get_full_game_state()` to expose:
- `offensive_three_seconds_enabled`: Rule enabled status
- `offensive_three_second_lane_width`: Configured lane width
- `offensive_three_second_max_steps`: Max steps allowed
- `offensive_lane_hexes`: List of (q, r) tuples for visualization
- `defensive_lane_hexes`: List of (q, r) tuples (basket hex)
- `offensive_lane_steps`: Dict mapping player_id → steps in lane
- `defensive_lane_steps`: Dict mapping player_id → steps in key

### 4. Visualization

**Web UI (`app/frontend/src/components/GameBoard.vue`):**
- `offensiveLaneHexes` computed property: Maps lane hexes to screen coordinates
- SVG layer renders lane with light red fill: `rgba(255, 100, 100, 0.15)`
- Subtle stroke for definition: `rgba(255, 100, 100, 0.3)`
- Rendered after court hexes but before 3PT line

**Environment Rendering (`basketworld_env_v2.py`):**
- Lane visualization in `_render_visual()` method for GIF/frame rendering
- Light red hexagons (alpha 0.15) with matching outline
- Only visible when violation rules are enabled
- Consistent appearance with web UI

## Testing

Successfully tested with a test script that verified:
1. ✅ Environment creates with rule enabled
2. ✅ Lane hexes calculated correctly (10 hexes for default 2v2 setup)
3. ✅ Observation includes lane step counts (21 dimensions vs 17 without)
4. ✅ Lane tracking increments and resets properly
5. ✅ Action masking would trigger at max steps

## MLflow Metrics

The following metrics are logged for each training phase:

### When Training Offense
- **`Offense 3-Second Violation`** - Offensive camping violations committed by the training offense team (turnovers)
- **`Offense Illegal Defense Violation`** - Defensive camping violations committed by the training offense team (should be rare/zero, as offense is on offense)

### When Training Defense
- **`Defense 3-Second Violation`** - Offensive camping violations committed by the frozen opponent offense (turnovers by opponent)
- **`Defense Illegal Defense Violation`** - Defensive camping violations committed by the training defense team

These metrics help you track:
- Whether offense is learning to avoid camping in the lane
- Whether defense is learning to avoid illegal defense violations
- How often the frozen opponent commits violations

## Usage Examples

### Training with the Rule

```bash
python train/train.py \
  --grid-size 12 \
  --players 3 \
  --alternations 10 \
  --steps-per-alternation 20000 \
  --num-envs 8 \
  --three-second-lane-width 1 \
  --three-second-max-steps 3 \
  --offensive-three-seconds true \
  --illegal-defense-enabled true
```

### Creating an Environment Directly

```python
import basketworld
from basketworld.envs.basketworld_env_v2 import Team

env = basketworld.HexagonBasketballEnv(
    grid_size=12,
    players=3,
    # Shared lane configuration
    three_second_lane_width=1,
    three_second_max_steps=3,
    # Enable both offensive and defensive rules
    offensive_three_seconds_enabled=True,
    illegal_defense_enabled=True,
    training_team=Team.OFFENSE,
)

obs, info = env.reset()
# obs['obs'] now includes lane step counts for all players
```

## Design Decisions

1. **Lane Shape**: Simple rectangular shape extending from basket along +q axis
   - Easy to understand and visualize
   - Consistent with real basketball "key" concept

2. **Ball Handler Exception**: Follows NBA rules
   - Player with ball gets one extra step to shoot
   - Prevents unfair turnovers when receiving pass on last permitted step

3. **Observation Space**: Added step counts for ALL players (not just violating player)
   - Allows agents to learn proactive lane management
   - Defensive players also get their illegal defense counter exposed
   - Increases observation dimension but provides crucial information

4. **Default Disabled**: Rule is opt-in via CLI flag
   - Preserves backward compatibility
   - Existing training runs unaffected

5. **Turnover Detail**: Logged with specific reason for MLflow tracking
   - Enables analysis of how often rule triggers
   - Can be used to adjust difficulty during curriculum learning

## Future Enhancements

Potential improvements for future iterations:

1. **Curriculum Learning**: Gradually decrease `max_steps` during training
2. **Per-Team Configuration**: Different rules for offense vs defense
3. **Dynamic Lane Width**: Change lane width based on game state
4. **Violation Warnings**: Visual/audio feedback in web UI when close to violation
5. **Analytics Dashboard**: Track violation frequency over training

## Files Modified

1. `/Users/evanzamir/projects/basketworld/basketworld/envs/basketworld_env_v2.py`
2. `/Users/evanzamir/projects/basketworld/train/train.py`
3. `/Users/evanzamir/projects/basketworld/app/backend/main.py`
4. `/Users/evanzamir/projects/basketworld/app/frontend/src/components/GameBoard.vue`

## Verification

All changes have been:
- ✅ Implemented (offensive and defensive rules)
- ✅ Linted (no errors)
- ✅ Tested with sample scripts
- ✅ Integrated with existing features (illegal defense, action masking, etc.)
- ✅ Documented
- ✅ **Updated defensive 3-second rule to use full lane area**
  - Verified offensive and defensive lanes are identical (10 hexes in 2v2 with default settings)
  - Confirmed basket hex is included in both lanes
  - Tested defensive tracking works in full lane area

The implementation is production-ready and can be enabled in training runs immediately.

### Test Results
- ✓ Defensive violations correctly award 1 point to offense
- ✓ Defensive violations give +1.0 reward to offense, -1.0 to defense
- ✓ Defensive violations properly end the episode
- ✓ Offensive violations correctly result in turnovers
- ✓ Offensive violations give turnover penalty rewards
- ✓ Offensive violations properly end the episode (via turnover)
- ✓ Lane tracking works for all players
- ✓ Violations are logged in action_results
- ✓ MLflow metrics properly track both violation types
- ✓ Shared parameters work correctly for both rules

### Key Changes Summary
1. **Offensive 3-second rule**: New feature, creates lane and enforces camping limits via turnover
2. **Defensive 3-second rule**: Updated to use same full lane area, award offense 1 point, and end episode per violation
3. **Observation space**: Now includes lane step counts for all players
4. **Visualization**: Painted lane area visible in web UI and rendered GIFs/frames
5. **Parameter naming**: Unified to use shared `three_second_lane_width` and `three_second_max_steps` for both offense and defense
6. **MLflow metrics**: Tracks 3-second violations and illegal defense violations per episode as `{team_name} 3-Second Violation` and `{team_name} Illegal Defense Violation`
7. **Episode termination**: Both violation types now end the episode (turnovers already did, defensive now does too)

