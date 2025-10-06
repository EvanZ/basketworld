# Phi Shaping UI Auto-Update Feature

## Changes Made

The Phi Shaping tab now features **automatic updates** and **client-side recalculation**:

### 1. Auto-Refresh on Every Step
- Table automatically updates after each game step
- No manual "Refresh" button needed
- Uses Vue watchers to detect gameState changes

### 2. Live Parameter Recalculation
- All rows recalculate instantly when you change parameters
- Move sliders or change dropdowns → see updated values immediately
- Useful for exploring "what would phi shaping have been?" scenarios

### 3. Removed Manual Refresh Button
- No longer needed with auto-updates
- "Apply" button now reads "Apply & Update Environment"
- Clearer workflow: adjust params → apply → see results automatically

## How It Works

### Backend Data Storage
The backend stores raw data for each step:
- `team_best_ep`: Best expected points among offensive team
- `ball_handler_ep`: Ball handler's expected points
- Step number and other metadata

### Client-Side Recalculation
The frontend recalculates on the fly:

```javascript
// When parameters change, recalculate Φ for all rows
Φ(s) = (1 - w) × teamBestEP + w × ballHandlerEP

// Recalculate shaping reward
r_shape = β × (γ × Φ_next - Φ_prev)
```

This means you can:
- Change **beta** → see how signal strength would differ
- Change **gamma** → see how discounting affects values
- Change **blend weight** → see impact of individual vs team balance
- Change **aggregation mode** → compare different Φ definitions

All without re-running the episode!

## User Experience

### Before (Old Behavior)
1. Take a step
2. Manually click "Refresh" to see new data
3. Change parameters
4. Manually click "Refresh" again
5. Table shows old values (from when steps were taken)

### After (New Behavior)
1. Take a step → **table updates automatically**
2. Change parameters (beta, mode, blend) → **table recalculates instantly**
3. See "what if" scenarios in real-time
4. Click "Apply" only when you want to change environment settings

## Use Cases

### Use Case 1: Finding the Right Beta
```
1. Play through an episode
2. Adjust beta slider: 0.05 → 0.10 → 0.15 → 0.20
3. Watch r_shape values change in real-time
4. Pick beta that gives meaningful signal (not too weak, not overwhelming)
5. Click Apply to set it
```

### Use Case 2: Comparing Aggregation Modes
```
1. Play through an episode
2. Switch aggregation mode dropdown:
   - team_best → see current values
   - teammates_best → see how excluding ball handler changes Φ
   - teammates_avg → see smoother aggregation
   - team_avg → see simple average
3. Compare Total row to see cumulative impact
4. Pick mode that best penalizes bad decisions
5. Click Apply to set it
```

### Use Case 3: Understanding Past Episodes
```
1. Load a saved episode replay
2. Go to Phi Shaping tab
3. Adjust parameters to see what phi shaping would have been
4. Identify which parameters would have provided best signal
5. Use those parameters for next training run
```

## Technical Details

### Reactivity
- Uses Vue `computed` properties for instant reactivity
- Parameters trigger full table recalculation (<1ms for 200 rows)
- No network calls needed for parameter changes

### Data Flow
```
Backend (stores raw data)
    ↓
rawLogData (fetched via API)
    ↓
displayRows (computed, recalculates with current params)
    ↓
Template (renders table)
```

### Approximations
For `team_avg` mode, we approximate using `(teamBestEP + ballEP) / 2` since we don't store all individual player EPs. This is close enough for visualization but may not be exact.

For other modes, we use the stored `team_best_ep` which may have been calculated with a different aggregation mode originally. The recalculation assumes the mode-specific teamBestEP would be similar.

## Benefits

1. **Better UX**: No manual refresh needed
2. **Instant Feedback**: See parameter effects immediately
3. **Exploration**: Try different parameters without re-playing
4. **Learning**: Understand how phi shaping works by tweaking values
5. **Debugging**: Quickly identify if parameters are working correctly

## Limitations

### Cannot Change
- **TeamBestEP and BallEP**: These are stored from when the step was taken
- **Step order**: Cannot reorder or filter steps
- **Add new steps**: Must take actions in the game

### Can Change
- **Beta (β)**: See how signal strength affects rewards
- **Gamma (γ)**: See how discounting changes shaping
- **Blend weight (w)**: See individual vs team balance
- **Aggregation mode**: See different Φ definitions
- **Ball-handler-only**: Toggle between modes

## Future Enhancements

Possible improvements:
1. Store per-player EPs for exact team_avg recalculation
2. Add "diff" view showing change from current environment params
3. Export recalculated values to CSV
4. Highlight rows where shaping reward changed significantly
5. Add chart visualization of Φ over time

## Troubleshooting

### "Table doesn't update after taking a step"
- Check that gameState prop is being passed to PhiShaping component
- Look for JavaScript errors in browser console
- Verify backend is storing phi_log data correctly

### "Recalculated values seem wrong"
- For team_avg mode, values are approximated
- For other modes, teamBestEP from backend may differ if mode changed
- Click "Apply" to ensure environment matches UI parameters

### "Total is different than during actual play"
- Expected! You're seeing what phi would have been with current params
- To see actual values from play, reset params to match environment
- Or click Apply first, then take new steps


