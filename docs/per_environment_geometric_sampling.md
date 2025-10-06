# Per-Environment Geometric Sampling

## Final Implementation

After testing different approaches, we've settled on **per-environment geometric sampling** as the optimal balance between diversity, performance, and control.

## How It Works

### 1. Sampling Phase (once per alternation)

Each parallel environment **independently** samples an opponent using the same geometric distribution:

```python
# With 16 environments, K=10, beta=0.7, eps=0.15:
available_policies = [Alt_40, Alt_41, ..., Alt_49]  # Last K=10 policies

for env_id in range(16):
    if random() < 0.15:  # eps chance
        opponent[env_id] = random.choice(ALL_HISTORY)  # Uniform from all history
    else:
        idx = sample_geometric([0,1,2,...,9], beta=0.7)  # Favor recent
        opponent[env_id] = available_policies[idx]

# Example result:
# Env 0:  Alt_49 (most recent)
# Env 1:  Alt_48 (recent)
# Env 2:  Alt_49 (duplicate is fine!)
# Env 3:  Alt_47 (recent)
# Env 4:  Alt_18 (from historical exploration)
# Env 5:  Alt_46 (recent)
# ...
# Env 15: Alt_45 (recent)
```

### 2. Loading Phase (once per alternation)

Download only the **unique** policies that were sampled:

```python
unique_policies = {Alt_49, Alt_48, Alt_47, Alt_46, Alt_45, Alt_18, ...}
# Typically 8-12 unique policies from 16 samples with beta=0.7

for policy in unique_policies:
    download_and_cache(policy)  # Download once, use in multiple envs
```

### 3. Training Phase (entire alternation)

Each environment uses its assigned opponent for all ~50K timesteps:

```
Worker 0 → Alt_49 → generates 50K timesteps
Worker 1 → Alt_48 → generates 50K timesteps
Worker 2 → Alt_49 → generates 50K timesteps (same as Worker 0, that's OK)
...
Worker 15 → Alt_45 → generates 50K timesteps

PPO.learn() aggregates all 800K timesteps across all opponents
→ Policy update uses gradients from 8-12 different opponent strategies
```

## Parameters

### `--opponent-pool-size` (K)
- **Type**: Integer
- **Default**: `10`
- **Effect**: How many recent checkpoints to consider
- **Tuning**:
  - Smaller (5-7): More biased to very recent opponents
  - Larger (15-20): More historical diversity
  - Recommended: `10-12` for good balance

### `--opponent-pool-beta` (β)
- **Type**: Float (0-1)
- **Default**: `0.7`
- **Effect**: Recency bias in geometric distribution
- **Tuning**:
  - `0.5`: Moderately favor recent (gentle geometric decay)
  - `0.7` ⭐: Good balance (default)
  - `0.85`: Strongly favor recent (sharp geometric decay)
  - `0.95`: Almost always pick most recent from pool

### `--opponent-pool-exploration` (ε)
- **Type**: Float (0-1)
- **Default**: `0.15`
- **Effect**: Probability of sampling from ALL history instead of recent K
- **Tuning**:
  - `0.0`: Never look at old policies (pure geometric)
  - `0.15` ⭐: 15% chance of old policy (default, prevents forgetting)
  - `0.3`: 30% chance of old policy (maximum diversity)

## Expected Behavior

### Early Training (Alternation 5, only 5 policies exist)

```
Pool: [Alt_1, Alt_2, Alt_3, Alt_4, Alt_5]  (K=10, but only 5 exist)

16 independent samples with beta=0.7:
- ~50% get Alt_5 (most recent)
- ~30% get Alt_4
- ~15% get Alt_3
- ~5% get Alt_1 or Alt_2

Unique policies downloaded: ~4-5
Diversity: 4-5 different opponents (vs 1 without flag)
```

### Mid Training (Alternation 50, 50 policies exist)

```
Pool: [Alt_40, Alt_41, ..., Alt_49]  (Last K=10)
Plus 15% exploration from Alt_1 through Alt_39

16 independent samples with beta=0.7, eps=0.15:
- ~40% get Alt_49 or Alt_48 (most recent 2)
- ~35% get Alt_47, Alt_46, Alt_45 (recent)
- ~15% get Alt_44, Alt_43, Alt_42, Alt_41, Alt_40 (older in pool)
- ~15% get Alt_1 through Alt_39 (historical exploration)

Unique policies downloaded: ~9-11
Diversity: 9-11 different opponents (vs 1 without flag)
```

### Late Training (Alternation 100, 100 policies exist)

```
Pool: [Alt_90, Alt_91, ..., Alt_99]  (Last K=10)
Plus 15% exploration from Alt_1 through Alt_89

16 independent samples:
- Heavy bias toward Alt_99, Alt_98, Alt_97
- Moderate representation of Alt_96-90
- Occasional very old policies (Alt_18, Alt_25, etc.)

Unique policies downloaded: ~10-12
Diversity: 10-12 different opponents including historical
```

## Performance Characteristics

| Metric | Value |
|--------|-------|
| **Speed** | Same as baseline (no per-episode reload) |
| **Memory** | +500-600 MB (10-12 policies × 50MB) |
| **Diversity** | 8-12 unique opponents per alternation |
| **Effective gradients** | Every batch samples from 8-12 strategies |
| **Prevents forgetting?** | ✅ Yes (continuous historical exposure) |

## Comparison with Alternatives

| Approach | Unique Opponents | Speed | Control |
|----------|-----------------|-------|---------|
| **Single opponent** | 1 | ⚡⚡⚡⚡ 1.0x | None |
| **Per-episode (v1)** | ~3000 | 🐌 4.0x | High |
| **Per-env round-robin** | 10-16 | ⚡⚡⚡⚡ 1.0x | Medium |
| **Per-env geometric** ⭐ | 8-12 | ⚡⚡⚡⚡ 1.0x | **High** |

## Why Geometric Sampling?

1. **Recency bias is important**: Recent opponents are most relevant to current training
2. **Prevents catastrophic forgetting**: Historical sampling (ε) ensures old strategies don't disappear
3. **Smooth transition**: Gradual decay from recent to old avoids sharp discontinuities
4. **Tunable**: Can adjust K, β, ε to control exploration/exploitation tradeoff

## Example Distribution

With `K=10, beta=0.7, eps=0.15` and 16 samples:

```
Probability of selecting each policy from recent pool [0-9]:
Index 9 (newest): ~28% → ~4-5 envs get this
Index 8:          ~20% → ~3 envs
Index 7:          ~14% → ~2 envs
Index 6:          ~10% → ~1-2 envs
Index 5:          ~7%  → ~1 env
Index 4:          ~5%  → ~1 env
Index 3:          ~4%  → ~0-1 envs
Index 2:          ~3%  → ~0 envs
Index 1:          ~2%  → ~0 envs
Index 0 (oldest): ~1%  → ~0 envs
Historical (ε):   ~15% → ~2-3 envs

Result: 8-10 unique policies from recent pool + 2-3 from history = 10-12 total
```

## Usage Example

```bash
python train/train.py \
    --per-env-opponent-sampling \
    --opponent-pool-size 12 \        # Consider last 12 checkpoints
    --opponent-pool-beta 0.75 \      # Moderate-strong recency bias
    --opponent-pool-exploration 0.2 \ # 20% chance of very old policy
    --num-envs 16 \                  # 16 independent samples
    --alternations 100
```

### Tuning Recommendations

**For maximum diversity** (prevent forgetting at all costs):
```bash
--opponent-pool-size 15
--opponent-pool-beta 0.6       # Gentler bias
--opponent-pool-exploration 0.3  # More historical
```

**For stability** (focus on recent performance):
```bash
--opponent-pool-size 8
--opponent-pool-beta 0.85      # Strong recency bias
--opponent-pool-exploration 0.1  # Less historical
```

**Balanced** (recommended default):
```bash
--opponent-pool-size 10
--opponent-pool-beta 0.7
--opponent-pool-exploration 0.15
```

## Verification

Run ELO analysis to verify improvement:

```bash
python analytics/elo_evolution.py \
    --run-id <your_run_id> \
    --tournament-mode sequential \
    --window-size 5 \
    --episodes 200 \
    --output-heatmap
```

Expected results:
- ✅ Monotonic ELO progression
- ✅ No "late loses to early" surprises
- ✅ Smooth learning curves
- ✅ No red spots in heatmap

## Technical Notes

### Why Multiple Envs Can Sample Same Policy

This is intentional and beneficial:
- Popular (recent/good) opponents get more training exposure
- Natural weighting emerges from probability distribution
- No artificial constraints on sampling

### Memory Efficiency

Only unique policies are downloaded:
```python
sampled = [Alt_49, Alt_49, Alt_48, Alt_49, Alt_47, ...]  # 16 samples
unique = {Alt_49, Alt_48, Alt_47, ...}  # ~10 unique
# Download only 10, not 16
```

### Gradient Diversity

Even with duplicates, diversity is high:
```
16 environments × 2048 steps = 32,768 timesteps per rollout
If 10 unique opponents across 16 workers:
→ Average ~3,200 steps per opponent per rollout
→ Each policy update uses gradients from 10 different strategies
→ Cannot overfit to single opponent
```

## Future Enhancements

1. **Adaptive sampling**: Adjust β based on training stage
2. **Performance-weighted**: Sample "challenging" opponents more often
3. **Stratified sampling**: Ensure representation from each quintile of history
4. **Dynamic K**: Increase pool size as training progresses

