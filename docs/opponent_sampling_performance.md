# Opponent Sampling Performance Comparison

## The Problem with Per-Episode Sampling

Initial implementation: sample new opponent on every episode reset
- **Result**: 4x slowdown! 
- **Cause**: Reloading policy from disk every ~30 steps
- **Why**: Lazy loading + string paths means each episode triggers disk I/O

## Solution: Per-Environment Assignment

**New approach**: Assign different opponents to each parallel worker once per alternation

### How It Works

```python
# With 16 parallel environments and pool of 10 opponents:
opponent_pool = [Alt_42, Alt_44, Alt_45, Alt_46, Alt_47, Alt_48, Alt_49, Alt_25, Alt_18, Alt_12]

# Assign to workers (cycle through pool):
Worker 0:  Alt_42  ← trains against this for entire alternation
Worker 1:  Alt_44
Worker 2:  Alt_45
Worker 3:  Alt_46
Worker 4:  Alt_47
Worker 5:  Alt_48
Worker 6:  Alt_49
Worker 7:  Alt_25
Worker 8:  Alt_18
Worker 9:  Alt_12
Worker 10: Alt_42  ← cycles back
Worker 11: Alt_44
...
Worker 15: Alt_46
```

### Performance Characteristics

| Approach | Opponents Loaded | Reload Frequency | Diversity | Speed |
|----------|-----------------|------------------|-----------|-------|
| **Old (Single)** | 1 | Once per alternation | 1x | ⚡️⚡️⚡️⚡️ Baseline |
| **Per-Episode** | 10 | Every ~30 steps | 3000x | 🐌 4x slower |
| **Per-Environment** ⭐ | 10-16 | Once per alternation | 16x | ⚡️⚡️⚡️⚡️ Same as baseline! |

### Why It's Fast

1. **One-time load**: Each subprocess loads its assigned policy once
2. **No disk I/O**: Policy stays in memory for entire alternation  
3. **No overhead**: Zero additional compute beyond single-opponent case
4. **Parallelism-native**: Leverages existing multi-process architecture

### Why It's Effective

With 16 workers and 10-opponent pool:
- Get **16 different opponents** simultaneously
- Each worker's gradients computed against different opponent
- Batch update aggregates across all 16 opponents
- **Equivalent diversity** to 16 episodes with different opponents
- But at **baseline speed**!

## Usage

Same flag, better performance:

```bash
python train/train.py \
    --per-env-opponent-sampling \  # Now uses per-environment assignment
    --opponent-pool-size 12 \
    --opponent-pool-exploration 0.2 \
    --num-envs 16  # More workers = more diversity!
```

### Tuning for Maximum Diversity

If you have 16 workers:
- `--opponent-pool-size 16`: Each worker gets unique opponent (best diversity)
- `--opponent-pool-size 8`: Some workers share opponents (still good)
- `--opponent-pool-size 24`: All workers get unique opponent + extras for variety

**Recommendation**: Set `pool_size ≥ num_envs` for maximum diversity

### Expected Results

With 16 workers and 12-opponent pool:
- **Speed**: Same as single-opponent baseline (~1.0x)
- **Diversity**: 12-16 unique opponents per alternation
- **Effective episodes**: Every batch samples from 16 different opponent matchups
- **Memory**: +600MB (12 policies × 50MB)

## Technical Details

### Distribution Logic

```python
# If pool has 10 policies and 16 workers:
worker_opponents = [pool[i % len(pool)] for i in range(16)]
# Result: [0,1,2,3,4,5,6,7,8,9, 0,1,2,3,4,5]
# Workers 0 and 10 get same opponent (Alt_42)
# Workers 1 and 11 get same opponent (Alt_44)
# etc.
```

### Gradient Aggregation

PPO collects experiences from all workers, so:
```
Single batch of 32,768 timesteps (16 envs × 2048 steps):
  - 2048 steps vs Alt_42 (worker 0)
  - 2048 steps vs Alt_44 (worker 1)
  - 2048 steps vs Alt_45 (worker 2)
  - ...
  - 2048 steps vs Alt_12 (worker 9)
  - 2048 steps vs Alt_42 (worker 10, same as worker 0)
  - ...
  
→ Policy update uses gradient from 10+ different opponents
→ Forces generalization across opponent strategies
```

### Why This Prevents Forgetting

- **Simultaneous exposure**: Model sees old and new strategies in same batch
- **Balanced gradients**: Can't overfit to single opponent's patterns
- **Continuous coverage**: Always training against diverse historical checkpoints
- **No temporal bias**: Old strategies don't "disappear" from training distribution

## Comparison with Alternatives

### Option A: Per-Episode Sampling
```python
for episode in range(3000):
    opponent = random.choice(pool)  # ← reload from disk
    run_episode(opponent)
```
- ✅ Maximum diversity (3000 different opponents)
- ❌ 4x slower (disk I/O overhead)
- ❌ Complex caching required

### Option B: Per-Rollout Sampling  
```python
for rollout in range(50):  # 50 rollouts per alternation
    opponent = random.choice(pool)  # ← reload every 2048 steps
    collect_rollout(opponent, steps=2048)
```
- ✅ Good diversity (50 different opponents)
- ⚠️ ~1.5-2x slower (less frequent reloading)
- ❌ Requires modifying PPO's learn() loop

### Option C: Per-Environment Assignment ⭐
```python
for worker_id in range(16):
    opponent = pool[worker_id % len(pool)]  # ← load once
    assign_to_worker(worker_id, opponent)

# Then train normally - no reloading!
for alternation in range(100):
    ppo.learn(total_timesteps=100000)  # Uses all 16 opponents naturally
```
- ✅ Excellent diversity (10-16 opponents)
- ✅ Zero overhead (baseline speed)
- ✅ Trivial implementation
- ✅ Scales with parallel workers

## Future Improvements

1. **Dynamic pool refresh**: Update opponent pool mid-alternation
2. **Adaptive assignment**: Give more workers to "challenging" opponents  
3. **Stratified sampling**: Ensure historical coverage (e.g., 4 old, 4 medium, 8 recent)
4. **Performance tracking**: Log which opponents cause most learning

## Verification

After training with per-environment assignment:

```bash
python analytics/elo_evolution.py \
    --run-id <run_id> \
    --tournament-mode sequential \
    --window-size 5 \
    --episodes 200 \
    --output-heatmap
```

Expected improvements:
- ✅ Monotonic ELO progression
- ✅ No "late model loses to early model" anomalies
- ✅ Smoother learning curves
- ✅ Faster training time than per-episode approach

