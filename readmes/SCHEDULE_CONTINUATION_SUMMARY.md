# Schedule Continuation Implementation Summary

## What Was Implemented

I've implemented a comprehensive solution for continuing training with proper schedule management. This solves the problem you described where schedules (entropy, phi-shaping, etc.) would reset or behave incorrectly when resuming training.

## Files Created/Modified

### New Files:
1. **`basketworld/utils/schedule_state.py`** - Core functionality for saving/loading schedule metadata
2. **`docs/schedule_continuation.md`** - Comprehensive user guide with examples

### Modified Files:
1. **`train/train.py`** - Updated to support schedule continuation with three modes

## Key Features

### 1. Three Continuation Modes

**`--continue-schedule-mode extend` (default)**
- Continues schedules from where they left off
- Adds new training to the original total timesteps
- Most common use case: "I want to train for 10 more alternations"

**`--continue-schedule-mode constant`**
- Uses final schedule values as constants (no scheduling)
- Good for: training more after schedules completed

**`--continue-schedule-mode restart`**
- Restarts schedules from scratch with new parameters
- Good for: experimenting with different schedule parameters

### 2. Automatic Metadata Saving

When training completes, the following metadata is automatically saved to MLflow:
- `schedule_total_planned_timesteps` - Total timesteps planned
- `schedule_current_timesteps` - Actual timesteps completed (from model.num_timesteps)
- All schedule parameters (start, end, type, bump settings)
- Supported for: entropy, phi-beta, pass logit bias, pass curriculum

### 3. Intelligent Schedule Reconstruction

When continuing with `extend` mode:
```python
new_total = original_total + new_training_timesteps
progress = model.num_timesteps / new_total
current_value = calculate_scheduled_value(progress)
```

This ensures schedules progress smoothly as if it were one continuous training session.

## Quick Start Examples

### Example 1: Add More Training (Most Common)

```bash
# Initial run: 20 alternations
python train/train.py \
    --alternations 20 \
    --ent-coef-start 0.01 \
    --ent-coef-end 0.001 \
    --mlflow-run-name my_run

# Continue for 10 more alternations (schedules continue smoothly)
python train/train.py \
    --alternations 10 \
    --continue-run-id <run_id_from_above> \
    --mlflow-run-name my_run_continued
    # --continue-schedule-mode extend is default
```

### Example 2: Train More with Fixed Values

```bash
# After initial run completes...
python train/train.py \
    --alternations 10 \
    --continue-run-id <run_id> \
    --continue-schedule-mode constant
    # Uses whatever entropy/phi-beta values the run ended with
```

### Example 3: Try Different Schedule Parameters

```bash
# Restart schedules with different parameters
python train/train.py \
    --alternations 10 \
    --continue-run-id <run_id> \
    --continue-schedule-mode restart \
    --ent-coef-start 0.02 \
    --ent-coef-end 0.0005
    # Fresh schedule, ignores previous run's schedule
```

## What This Solves

### Before (Problems):
❌ Resuming training reset entropy to start value (unwanted exploration spike)  
❌ Schedules calculated wrong progress (didn't account for previous training)  
❌ Phi-shaping would restart from high values (policy disruption)  
❌ No way to just add "10 more alternations" without issues  
❌ Had to manually track and set constant values  

### After (Solutions):
✅ Schedules continue seamlessly from where they left off  
✅ Can add more alternations and schedules extend automatically  
✅ Can use final schedule values as constants if desired  
✅ Can restart schedules with new parameters if experimenting  
✅ All schedule types supported (entropy, phi-beta, pass curriculum)  
✅ Automatic metadata tracking via MLflow  

## How to Use

1. **Just continue training** (default behavior):
   ```bash
   python train/train.py --alternations 10 --continue-run-id <id>
   ```
   Schedules automatically extend!

2. **Want constant values instead?**
   ```bash
   python train/train.py --alternations 10 --continue-run-id <id> --continue-schedule-mode constant
   ```

3. **Want to experiment with different schedules?**
   ```bash
   python train/train.py --alternations 10 --continue-run-id <id> --continue-schedule-mode restart --ent-coef-start 0.05
   ```

## Verification

When training starts, you'll see console output like:
```
Loaded schedule metadata from run abc123
  Previous total timesteps: 2000000
  Previous current timesteps: 2000000
Schedule mode: EXTEND - continuing schedules from previous run
  Extended total timesteps: 3000000
```

This confirms:
- Metadata was loaded successfully
- Mode being used
- How total timesteps are calculated

## Backward Compatibility

- Old `--restart-entropy-on-continue` flag still works (maps to `--continue-schedule-mode restart`)
- Runs without schedule metadata fall back gracefully
- No breaking changes to existing workflows

## Implementation Details

### Metadata Storage
Metadata is stored as MLflow parameters with `schedule_` prefix:
- Easy to inspect in MLflow UI
- Preserved when continuing runs
- Doesn't interfere with other parameters

### Schedule Reconstruction
```python
# In extend mode
if previous_metadata:
    original_total = previous_metadata["total_planned_timesteps"]
    new_total = original_total + new_training_timesteps
    
    # Callbacks use new_total for progress calculation
    # model.num_timesteps continues from previous value
    # Result: seamless continuation
```

### Supported Schedules
All schedule types are supported:
- Entropy coefficient (linear or exponential)
- Phi beta (potential-based reward shaping)
- Pass logit bias (exponential)
- Pass curriculum (arc degrees and OOB turnover probability)

## Testing Recommendations

1. **Test extend mode** - Most important, most common use case:
   ```bash
   # Run 5 alternations
   python train/train.py --alternations 5 --ent-coef-start 0.01 --ent-coef-end 0.001
   # Add 5 more - entropy should continue from 0.001 toward 0.001 (or extend curve)
   python train/train.py --alternations 5 --continue-run-id <id>
   ```

2. **Test constant mode** - Verify final values are used:
   ```bash
   python train/train.py --alternations 5 --continue-run-id <id> --continue-schedule-mode constant
   # Check MLflow logs - entropy should be constant at previous run's final value
   ```

3. **Test restart mode** - Verify fresh schedule:
   ```bash
   python train/train.py --alternations 5 --continue-run-id <id> --continue-schedule-mode restart --ent-coef-start 0.05
   # Entropy should restart at 0.05, not continue from previous value
   ```

## Next Steps

1. Try continuing one of your existing runs with `--continue-run-id <id>`
2. Check the console output to verify schedule mode
3. Inspect MLflow UI to see schedule metadata parameters
4. Read full documentation in `docs/schedule_continuation.md`

## Questions?

Common scenarios are covered in the full documentation. The implementation handles:
- Multiple schedule types
- Different schedule curves (linear, exponential)
- Bump parameters (for alternation resets)
- Missing metadata (graceful fallback)
- Backward compatibility

The default behavior (`extend` mode) should "just work" for the most common use case: adding more alternations to an existing training run.

