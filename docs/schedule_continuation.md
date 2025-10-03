# Schedule Continuation Guide

## Overview

When continuing training from a previous run using `--continue-run-id`, you now have fine-grained control over how schedules (entropy, phi-shaping, pass curriculum, etc.) behave. This guide explains the three continuation modes and when to use each.

## The Problem

Previously, when resuming training from a checkpoint, schedules would either:
- **Reset to start values**, causing unwanted spikes in entropy or phi-beta
- **Continue with incorrect progress**, because the new run's `total_planned_timesteps` didn't account for the previous training

This made it difficult to:
- Add more alternations to an existing training run
- Continue with consistent exploration/exploitation balance
- Resume phi-shaping or pass curriculum from where they left off

## The Solution

The new `--continue-schedule-mode` flag provides three options:

### 1. **`extend` (Default)** - Continue Schedules Seamlessly

This mode treats your continued training as an extension of the original training session. Schedules pick up exactly where they left off and continue toward their end values over the extended total timesteps.

**When to use:**
- You want to add more training to an existing run
- Schedules should continue their original trajectory
- Most common use case

**Example:**
```bash
# Original run: 20 alternations with entropy 0.01→0.001
python train/train.py \
    --alternations 20 \
    --ent-coef-start 0.01 \
    --ent-coef-end 0.001 \
    --mlflow-run-name my_initial_run

# Continue for 10 more alternations (extends the schedule)
python train/train.py \
    --alternations 10 \
    --continue-run-id <run_id_from_above> \
    --continue-schedule-mode extend \
    --mlflow-run-name my_continued_run
```

In this example:
- Original run: entropy decays from 0.01 to 0.001 over 20 alternations
- After 20 alternations, entropy reached 0.001
- **Extend mode**: Continues from 0.001, extending toward the original end value over 10 more alternations
- Total training: 30 alternations with smooth, uninterrupted schedule

**How it works:**
- Loads schedule metadata from previous run (start, end, type, etc.)
- Calculates `new_total_timesteps = original_total + new_training_timesteps`
- Schedules use model's actual `num_timesteps` for progress calculation
- Result: Seamless continuation as if it were one long training session

### 2. **`constant`** - Use Final Schedule Values

This mode takes whatever values the schedules reached at the end of the previous run and uses them as constants (no scheduling) for the continued training.

**When to use:**
- Schedules already reached their end values and you want to keep them fixed
- You've tuned exploration/shaping to a good level and want more training without changes
- Fine-tuning or exploitation-focused training

**Example:**
```bash
# Original run reached entropy=0.001, phi_beta=0.5 at the end
python train/train.py \
    --alternations 10 \
    --continue-run-id <previous_run_id> \
    --continue-schedule-mode constant
```

In this example:
- Entropy stays at 0.001 (constant, no schedule)
- Phi-beta stays at 0.5 (constant, no schedule)
- Good for adding more training after schedules completed

**How it works:**
- Loads final schedule values from previous run
- Sets those as constant hyperparameters
- No scheduling callbacks are created
- Result: Stable training with fixed hyperparameters

### 3. **`restart`** - Start Schedules Fresh

This mode ignores previous schedule state and restarts all schedules from scratch using the parameters you provide in the current command.

**When to use:**
- You want to try different schedule parameters
- Previous schedules weren't working well
- Experimenting with different exploration strategies
- You want to "refresh" entropy after policy has converged

**Example:**
```bash
# Try a different entropy schedule on the continued model
python train/train.py \
    --alternations 10 \
    --continue-run-id <previous_run_id> \
    --continue-schedule-mode restart \
    --ent-coef-start 0.02 \
    --ent-coef-end 0.0005 \
    --ent-schedule exp
```

In this example:
- Model checkpoint is loaded from previous run
- Entropy schedule restarts fresh: 0.02 → 0.0005 over the new 10 alternations
- Independent of what entropy was in the previous run

**How it works:**
- Ignores previous schedule metadata
- Uses current command-line arguments to create new schedules
- `total_planned_timesteps` is calculated only from new training duration
- Result: Fresh schedules as if starting a new run (but with trained model)

## Supported Schedules

All of the following schedules support continuation:

- **Entropy coefficient** (`--ent-coef-start`, `--ent-coef-end`, `--ent-schedule`)
- **Phi beta (reward shaping)** (`--phi-beta-start`, `--phi-beta-end`, `--phi-beta-schedule`)
- **Pass logit bias** (`--pass-logit-bias-start`, `--pass-logit-bias-end`)
- **Pass curriculum** (`--pass-arc-start`, `--pass-arc-end`, `--pass-oob-turnover-prob-start`, `--pass-oob-turnover-prob-end`)

## Technical Details

### Metadata Saved to MLflow

When training completes, the following parameters are logged to MLflow for schedule continuation:

- `schedule_total_planned_timesteps`: Total timesteps planned in this run
- `schedule_current_timesteps`: Actual timesteps completed (`model.num_timesteps`)
- `schedule_ent_coef_start`, `schedule_ent_coef_end`, `schedule_ent_schedule`, etc.
- All schedule parameters for each supported schedule type

### How `extend` Mode Calculates Progress

```python
# Original run
original_total_timesteps = 2 * alternations * steps_per_alt * num_envs * n_steps
original_current_timesteps = model.num_timesteps  # e.g., 1,000,000

# Continued run
new_training_timesteps = 2 * new_alternations * steps_per_alt * num_envs * n_steps
extended_total_timesteps = original_total_timesteps + new_training_timesteps

# Schedule progress
progress = model.num_timesteps / extended_total_timesteps
current_value = start + (end - start) * progress  # Linear
# or: current_value = end * (start/end) ** (1.0 - progress)  # Exponential
```

This ensures schedules progress smoothly across the extended training period.

## Examples

### Example 1: Simple Extension

Train for 50 alternations, then add 25 more:

```bash
# Initial training
python train/train.py \
    --alternations 50 \
    --ent-coef-start 0.01 \
    --ent-coef-end 0.0001 \
    --mlflow-run-name phase1

# Extend with 25 more alternations
python train/train.py \
    --alternations 25 \
    --continue-run-id <run_id> \
    --continue-schedule-mode extend \
    --mlflow-run-name phase2
```

Result: Smooth 75-alternation training session with continuous entropy decay.

### Example 2: Constant Values After Schedule Completes

Train with schedule until it completes, then train more with fixed values:

```bash
# Initial training with phi-shaping schedule
python train/train.py \
    --alternations 40 \
    --enable-phi-shaping \
    --phi-beta-start 2.0 \
    --phi-beta-end 0.0 \
    --phi-beta-schedule exp \
    --mlflow-run-name with_shaping

# Continue without shaping (phi_beta=0.0)
python train/train.py \
    --alternations 20 \
    --continue-run-id <run_id> \
    --continue-schedule-mode constant \
    --mlflow-run-name post_shaping
```

Result: 40 alternations with decaying phi-beta, then 20 alternations with phi_beta=0.0 (no shaping).

### Example 3: Restart with Different Parameters

Model converged but policy is stale - refresh with new entropy:

```bash
# Original training
python train/train.py \
    --alternations 30 \
    --ent-coef-start 0.01 \
    --ent-coef-end 0.0001 \
    --mlflow-run-name initial

# Refresh exploration with new entropy burst
python train/train.py \
    --alternations 10 \
    --continue-run-id <run_id> \
    --continue-schedule-mode restart \
    --ent-coef-start 0.05 \
    --ent-coef-end 0.001 \
    --mlflow-run-name refreshed
```

Result: Model weights continue from alternation 30, but entropy gets a fresh high→low schedule to encourage renewed exploration.

## Backward Compatibility

The old `--restart-entropy-on-continue` flag is deprecated but still works:
- `--restart-entropy-on-continue true` → equivalent to `--continue-schedule-mode restart`

## Troubleshooting

### "Warning: Could not load schedule metadata"

This appears when continuing from a run that predates this feature. In this case:
- **Extend mode** falls back to restart behavior (no metadata available)
- **Constant mode** won't work (no previous values to use)
- **Restart mode** works normally

### Schedules Not Behaving as Expected

Check the console output when training starts:
```
Loaded schedule metadata from run abc123
  Previous total timesteps: 2000000
  Previous current timesteps: 2000000
Schedule mode: EXTEND - continuing schedules from previous run
  Extended total timesteps: 3000000
```

This shows exactly how schedules are being configured.

### Want to See Schedule Values

Check MLflow UI → Run → Parameters → search for "schedule_" to see all saved metadata.

## Best Practices

1. **Default to `extend`** - It's the most intuitive and handles continuation seamlessly
2. **Use `constant` after convergence** - When schedules have done their job and you just want more training
3. **Use `restart` for experiments** - When you want to try different hyperparameter trajectories
4. **Check the console output** - Always verify schedule mode and timestep calculations when training starts
5. **Log meaningful run names** - Use `--mlflow-run-name` to distinguish continuation runs

## Summary

| Mode | Use Case | Schedule Behavior |
|------|----------|------------------|
| `extend` | Add more alternations to existing run | Continues from where it left off, extends to new total |
| `constant` | Train more with fixed hyperparameters | Uses final schedule values as constants |
| `restart` | Try different schedule parameters | Ignores previous run, uses new parameters |

