# Per-Episode Opponent Sampling

## Problem

When training with self-play and sampling opponents only **once per alternation**, the agent trains against the same opponent for potentially millions of timesteps. This leads to:

1. **Strategic Forgetting**: Agent learns to beat current opponent but forgets how to handle strategies from earlier training
2. **Overfitting**: Agent becomes specialized for one opponent rather than generalizing
3. **Non-Transitive Results**: Later models can lose to much earlier models (e.g., Alt 100 loses to Alt 18)

## Solution

**Per-Episode Opponent Sampling**: Instead of selecting one opponent per alternation, maintain a pool of diverse opponents and randomly sample a new opponent for **each episode**.

### Key Changes

#### 1. Modified `SelfPlayEnvWrapper`

The wrapper now accepts either:
- **Single policy** (original behavior): `opponent_policy="/path/to/model.zip"`
- **Policy pool** (new): `opponent_policy=["/path/to/model1.zip", "/path/to/model2.zip", ...]`

When given a pool, it samples a new opponent on every `reset()` call (every episode).

#### 2. New Training Function

`get_opponent_policy_pool()` builds a diverse pool of opponents:
- **70% recent policies** (last 7 of 10 policies)
- **30% historical policies** (sampled from earlier checkpoints)
- All policies are downloaded and ready for sampling

#### 3. Command-Line Control

```bash
# Enable per-episode sampling (recommended for robust training)
python train/train.py \
    --per-env-opponent-sampling \
    --opponent-pool-size 10 \
    --opponent-pool-exploration 0.15

# Default behavior (sample once per alternation)
python train/train.py
    # ... other args
```

## Sampling Frequency Comparison

### Old Behavior (Once per Alternation)
```
Alternation 50:
├─ Select Alt 47 as opponent
├─ Train Offense: 100,000 timesteps vs Alt 47
└─ Train Defense: 100,000 timesteps vs Alt 47

Result: 200,000 timesteps against SINGLE opponent
```

### New Behavior (Per-Episode Sampling)
```
Alternation 50:
├─ Build pool: [Alt 42, 44, 45, 46, 47, 48, 49, 25, 18, 12]
├─ Train Offense: 100,000 timesteps
│   ├─ Episode 1: vs Alt 49 (20 steps)
│   ├─ Episode 2: vs Alt 18 (35 steps)  ← old strategy!
│   ├─ Episode 3: vs Alt 47 (28 steps)
│   ├─ Episode 4: vs Alt 45 (22 steps)
│   └─ ... (~3000 episodes, each with different opponent)
└─ Train Defense: 100,000 timesteps
    └─ (same per-episode sampling)

Result: ~6000 episodes, each against DIFFERENT opponent
```

## Parameters

### `--per-env-opponent-sampling`
- **Type**: Flag (boolean)
- **Default**: `False` (off)
- **Effect**: When enabled, samples new opponent for each episode from a pool

### `--opponent-pool-size`
- **Type**: Integer
- **Default**: `10`
- **Effect**: Number of policies to include in the pool
- **Recommendation**: 
  - 10-15 for balanced diversity and efficiency
  - 20+ if you have many checkpoints and want maximum diversity
  - 5-7 for faster training with less diversity

### `--opponent-pool-exploration`
- **Type**: Float (0-1)
- **Default**: `0.15`
- **Effect**: Probability weight for sampling from ALL history (not just recent)
- **Recommendation**:
  - 0.15-0.25 for good balance
  - 0.0 for purely recent opponents
  - 0.5+ for maximum historical diversity

### `--opponent-pool-beta` (single-policy mode only)
- **Type**: Float (0-1)
- **Default**: `0.7`
- **Effect**: When NOT using per-episode sampling, controls recency bias
- **Not used** in per-episode mode

## Expected Benefits

1. **Prevents Strategic Forgetting**: Continuously trains against old strategies
2. **Better Generalization**: Learns robust policies that work against diverse opponents
3. **Monotonic Improvement**: Later alternations should consistently beat earlier ones
4. **Higher ELO Ratings**: More stable skill progression

## Performance Impact

### Memory
- **Old**: 1 opponent policy loaded per alternation
- **New**: N policies loaded (e.g., 10 policies × 50MB each = 500MB additional memory)
- **Mitigation**: Policies are loaded lazily and cached efficiently

### Speed
- **Loading**: ~5-10 seconds one-time cost per alternation to download pool
- **Sampling**: Negligible (< 1ms per episode)
- **Total**: < 1% slower overall

### Disk
- Opponent cache directory grows with pool size (e.g., 10 policies × 50MB = 500MB)
- Cleaned up at end of run

## Recommended Settings for Next Training Run

```bash
python train/train.py \
    --per-env-opponent-sampling \
    --opponent-pool-size 12 \
    --opponent-pool-exploration 0.2 \
    --alternations 100 \
    # ... your other hyperparameters
```

This will:
- Sample ~8-9 recent opponents + ~3-4 historical opponents
- Give each episode a different opponent
- Include 20% weight to very old checkpoints for robustness

## Verification

After training with per-episode sampling, run ELO analysis to verify improvement:

```bash
python analytics/elo_evolution.py \
    --run-id <your_run_id> \
    --tournament-mode sequential \
    --window-size 5 \
    --episodes 200 \
    --output-heatmap
```

Look for:
- ✅ **Monotonic ELO increase** (later models consistently beat earlier ones)
- ✅ **No red spots in heatmap** (no surprising losses to old models)
- ✅ **Smooth progression** (no erratic jumps or plateaus)

## Migration Path

1. **Current runs**: Finish with existing behavior (no changes needed)
2. **Next run**: Add `--per-env-opponent-sampling` flag
3. **Compare**: Use ELO analysis to compare old vs new runs
4. **Iterate**: Adjust `--opponent-pool-size` based on results

## Technical Details

### How SelfPlayEnvWrapper Handles Pools

```python
# In reset():
if self.use_pool and self.opponent_pool:
    # Sample new opponent path from pool
    sampled_path = random.choice(self.opponent_pool)
    self.opponent_policy = sampled_path  # Will be lazy-loaded on first use

# In step():
self._ensure_opponent_loaded()  # Loads policy if it's still a path string
opponent_action, _ = self.opponent_policy.predict(obs)
```

### Memory Management

- Policies are loaded on-demand (lazy loading)
- Previous policy is garbage collected when new one is loaded
- Only 1-2 policies in memory at a time per subprocess
- With 16 parallel environments, ~16-32 policies in memory total

### Reproducibility

- Add seed to control opponent sampling:
  ```python
  random.seed(args.seed)
  ```
- Or sample deterministically based on episode number for reproducible experiments

## Future Enhancements

Potential improvements:
1. **Prioritized sampling**: Weight pool by recency or performance
2. **Adaptive pool**: Add/remove policies based on training progress
3. **Curriculum learning**: Start with easy opponents, increase difficulty
4. **Multi-level pools**: Different pools for offense vs defense
5. **Performance tracking**: Log which opponents cause most learning

## References

- OpenAI Five: Used opponent pools of 5-10 past checkpoints
- AlphaStar: Maintained league of diverse opponents
- Dota 2 AI: "Past Self Play" with rolling window of historical agents

