# Web App Defender Pressure Display

## Summary

Added defender pressure turnover probability display to the web app, similar to how pass interception probability is shown.

## Changes Made

### 1. Backend (Environment)

**File: `basketworld/envs/basketworld_env_v2.py`**

- Modified defender pressure mechanic to collect pressure information before checking for turnovers
- Added `defender_pressure` section to action results dictionary
- Tracks for each defender applying pressure:
  - `defender_id`: Which defender is applying pressure
  - `distance`: Distance from ball handler
  - `turnover_prob`: Individual turnover probability
- Also calculates `total_pressure_prob`: Combined probability from all defenders (compound probability)

**Key Code Changes:**
```python
# In step() function (lines 647-700):
# - Collect all defender pressures first
# - Calculate compound probability: total = 1 - ∏(1 - prob_i)
# - Then check for turnovers

# Added to action_results (lines 872-876):
if defender_pressure_info and self.ball_holder in self.offense_ids:
    action_results["defender_pressure"][self.ball_holder] = {
        "defenders": defender_pressure_info,
        "total_pressure_prob": total_pressure_prob,
    }
```

### 2. Frontend

**File: `app/frontend/src/components/PlayerControls.vue`**

#### Added Function (lines 473-485):
```javascript
function getDefenderPressureProbability(move, playerId) {
  if (!move.actionResults || !move.actionResults.defender_pressure) {
    return null;
  }
  
  const pressureInfo = move.actionResults.defender_pressure[playerId];
  if (!pressureInfo || pressureInfo.total_pressure_prob === undefined) {
    return null;
  }
  
  return pressureInfo.total_pressure_prob;
}
```

#### Display in Moves Table (lines 1030-1032):
```html
<div v-if="getDefenderPressureProbability(move, playerId) !== null" 
     class="defender-pressure-info">
  ({{ (getDefenderPressureProbability(move, playerId) * 100).toFixed(1) }}% turnover risk)
</div>
```

#### CSS Styling (lines 1540-1544):
```css
.defender-pressure-info {
  font-size: 0.8em;
  color: #ff6b35;  /* Orange color to distinguish from pass steal (red)
  font-style: italic;
}
```

#### Parameters Tab (lines 1125-1128):
Added display of `defender_pressure_decay_lambda` parameter:
```html
<div class="param-item">
  <span class="param-name">Decay lambda:</span>
  <span class="param-value">{{ props.gameState.defender_pressure_decay_lambda || 'N/A' }}</span>
</div>
```

## How It Works

1. **When the ball handler has the ball**, the environment calculates defender pressure from all defenders in front (within the 180° arc toward basket)

2. **For each defender** applying pressure:
   - Distance is measured
   - Individual turnover probability is calculated: `base * exp(-lambda * (distance - 1))`

3. **Total pressure probability** is calculated using compound probability:
   - `total = 1 - ∏(1 - individual_prob)`
   - This represents the probability that AT LEAST ONE defender forces a turnover

4. **In the web app**, this information is displayed in the moves table:
   - Shows as "(X.X% turnover risk)" below the ball handler's action
   - Displayed in orange color to distinguish from pass steal risk (red)
   - Only shown when defenders are applying pressure

## Example Display

```
Turn | Shot Clock | Player 0        | Player 1
-----|------------|-----------------|----------
  5  |    20      | MOVE_E          | NOOP
     |            | (3.2% turnover  |
     |            |  risk)          |
  6  |    19      | PASS_E          | NOOP
     |            | (15.4% steal    |
     |            |  risk)          |
```

## Benefits

- **Visibility**: Players can see the turnover risk from defender pressure
- **Strategic Feedback**: Helps understand when the ball handler is in danger
- **Consistent UI**: Matches the existing pass steal probability display pattern
- **Compound Probability**: Shows realistic total risk when multiple defenders are pressuring

