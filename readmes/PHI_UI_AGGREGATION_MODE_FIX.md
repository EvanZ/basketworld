# Phi Shaping UI Aggregation Mode Implementation

## Changes Made

### Backend (`app/backend/main.py`)

1. **Added `phi_aggregation_mode` to GET /api/phi_params:**
   - Now returns the current aggregation mode from the environment
   - Default: `"team_best"` for backward compatibility

2. **Added `phi_aggregation_mode` to SetPhiParamsRequest:**
   - New optional field to change aggregation mode via API
   - Validates against allowed modes: `["team_best", "teammates_best", "teammates_avg", "team_avg"]`

3. **Updated POST /api/phi_params handler:**
   - Sets `env.phi_aggregation_mode` when provided
   - Validates mode before setting

### Frontend (`app/frontend/src/components/PhiShaping.vue`)

1. **Added aggregation mode dropdown:**
   - Shows all four aggregation modes with descriptive labels
   - Syncs with backend via API

2. **Conditional blend weight display:**
   - Hides blend weight slider when `team_avg` mode is selected (not used in that mode)
   - Shows for all other modes

3. **Updated label text:**
   - Changed "Blend w (Team vs Ball Φ)" to "Blend w (Aggregate vs Ball Φ)" for clarity

## Why Phi Shaping Was Showing Zeros

The phi shaping values in the UI show zeros when:

1. **`phi_beta = 0.0`** (most common)
   - Phi shaping reward = β × (γ × Φ_next - Φ_prev)
   - If β = 0, all shaping rewards are 0
   - **Solution:** Set beta to non-zero value (e.g., 0.10-0.20)

2. **`enable_phi_shaping = false`**
   - Environment still calculates phi values but doesn't apply them
   - Shows calculated values in log but they don't affect actual rewards
   - **Solution:** Click "Apply" in Phi Shaping tab (auto-enables phi shaping)

3. **Model was trained with phi shaping disabled:**
   - If loaded model was trained without phi shaping, beta defaults to 0
   - **Solution:** Manually set beta in UI after loading model

## How to Use the New UI

### Step 1: Load a Game
Initialize a game with a trained model (or start fresh).

### Step 2: Go to Phi Shaping Tab
Click "Phi Shaping" in the Player Controls tabs.

### Step 3: Configure Parameters
**Note:** Phi shaping is automatically enabled when using this tab.

- **Beta:** Set to 0.10-0.20 (controls signal strength)
- **Gamma:** Should match your training gamma (0.99 or 1.0)
- **Aggregation Mode:** Select mode:
  - `team_best`: Original (max including ball handler)
  - `teammates_best`: Recommended (max excluding ball handler)
  - `teammates_avg`: Mean of teammates
  - `team_avg`: Mean of all (no blend parameter)
- **Blend weight:** Adjust w between 0-1 (if not using team_avg)

### Step 4: Apply
Click "Apply" to set parameters and enable phi shaping.

### Step 5: Play and Observe
Take steps and click "Refresh" to see phi shaping values in the log table.

## Interpreting the Log

The log shows per-step phi shaping diagnostics:

| Column | Meaning |
|--------|---------|
| Step | Step number in episode |
| β | Beta value at this step |
| Φprev | Potential before action |
| Φnext | Potential after action |
| TeamBestEP | Best EP among all teammates (depends on mode) |
| BallEP | Ball handler's EP |
| r_shape/team | Actual shaping reward per team |

**Total row:** Sum of all r_shape values over the episode.

### What to Look For

1. **Non-zero r_shape values:**
   - Positive: State improved (better shot quality)
   - Negative: State degraded (worse shot quality)
   - Zero: No change (or β=0)

2. **Φprev vs Φnext:**
   - Φnext > Φprev → Good action (improved position)
   - Φnext < Φprev → Bad action (degraded position)

3. **TeamBestEP vs BallEP:**
   - Shows how aggregation mode affects Φ calculation
   - With `teammates_best`, TeamBestEP excludes ball handler
   - With `team_avg`, Φ = average of all EPs (shown in Φnext)

## Troubleshooting

### "All phi values are still zero"
- Check that Beta > 0 (most common issue!)
- Click Apply after changing parameters (auto-enables phi shaping)
- Click Refresh to update log
- If still zero, check backend logs for errors

### "Blend weight has no effect"
- If using `team_avg` mode, blend weight is ignored (correct behavior)
- Otherwise, change blend weight and take a step to see effect

### "Can't see aggregation mode options"
- Refresh page if dropdown doesn't appear
- Backend needs to be restarted to load new API endpoints
- Check browser console for errors

## Testing Different Modes

Try this scenario to see the difference:

1. Set up a state where:
   - Ball handler has EP = 0.8
   - Teammate has EP = 1.2

2. Take an action that **degrades ball handler's shot** (EP 0.8 → 0.6)

3. Compare modes with w=0.3:
   - `team_best`: Φ changes from 0.7×1.2 + 0.3×0.8 = 1.08 to 0.7×1.2 + 0.3×0.6 = 1.02 (small penalty)
   - `teammates_best`: Φ changes from 0.7×1.2 + 0.3×0.8 = 1.08 to 0.7×1.2 + 0.3×0.6 = 1.02 (same in this case)
   - `teammates_avg`: Uses mean of teammates
   - `team_avg`: Uses mean of all players (no blend)

The difference becomes apparent when the ball handler has the best shot initially.

## For Training

To use these modes in training, use the new CLI parameter:

```bash
--phi-aggregation-mode teammates_best \
--phi-blend-weight 0.5 \
--phi-beta-start 0.15 \
--phi-beta-end 0.15
```

See `readmes/phi_aggregation_modes.md` for full training recommendations.

