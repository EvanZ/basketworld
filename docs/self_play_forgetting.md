# Strategic Forgetting in Self-Play Training

## Problem

When training with self-play, agents can exhibit **strategic forgetting** or **cycling** where later models perform worse against earlier models despite improving against their current training opponents.

### Example
- Alt 100 Offense beats Alt 100 Defense: 65% win rate ✓
- Alt 100 Offense vs Alt 18 Defense: 38% win rate ✗

This violates our expectation that later models should dominate earlier ones.

## Root Causes

### 1. Co-evolutionary Narrowing
- Offense and defense evolve together as a coupled system
- They become specialized to beat *each other*, not to be generally robust
- Early defensive strategies get "forgotten" because they never appear in training

### 2. Curriculum Collapse
- As training progresses, the agent only sees one opponent (itself)
- Loses ability to generalize to diverse strategies
- Similar to overfitting in supervised learning

### 3. Non-Transitive Strategy Spaces
- Some games have rock-paper-scissors dynamics
- No single "best" strategy exists
- Training can cycle through strategies indefinitely

## Solutions

### 1. Historical Opponent Pool
Instead of always training against the most recent model, maintain a pool of past opponents:

```python
# Pseudocode for train.py modification
opponent_pool = []  # List of (alternation_idx, policy) tuples

for alternation in range(num_alternations):
    # Add current policy to pool
    if alternation % 5 == 0:  # Every 5 alternations
        opponent_pool.append((alternation, copy.deepcopy(unified_policy)))
    
    # Keep pool size manageable
    if len(opponent_pool) > 20:
        opponent_pool.pop(0)  # Remove oldest
    
    # Sample opponent from pool (with recency bias)
    weights = [2**(i - len(opponent_pool)) for i in range(len(opponent_pool))]
    opponent_idx = np.random.choice(len(opponent_pool), p=normalize(weights))
    opponent = opponent_pool[opponent_idx]
    
    # Train against sampled opponent instead of most recent
```

### 2. Periodic Evaluation Against Fixed Benchmarks
```python
# Keep a fixed set of "milestone" checkpoints
milestone_alts = [5, 20, 40, 60, 80, 100]

# During training, periodically test against these
if alternation % 10 == 0:
    for milestone_alt in milestone_alts:
        win_rate = evaluate_against(current_model, milestone_models[milestone_alt])
        mlflow.log_metric(f"winrate_vs_alt_{milestone_alt}", win_rate)
        
        # Alert if win rate drops below threshold
        if milestone_alt < alternation - 20 and win_rate < 0.5:
            print(f"⚠️ WARNING: Losing to much earlier model (alt {milestone_alt})")
```

### 3. Diversity Regularization
Add entropy bonus or other diversity rewards to prevent strategy collapse:

```python
# In environment or reward function
if encourage_diversity:
    # Penalize agents for using same strategy repeatedly
    strategy_diversity_bonus = calculate_action_entropy(recent_episodes)
    reward += diversity_weight * strategy_diversity_bonus
```

### 4. Explicit Robustness Training
Occasionally inject random or scripted opponents:

```python
if alternation % 10 == 0:
    # Train for a few iterations against a scripted "hard-coded" defense
    # that uses known-effective strategies
    train_against_scripted_opponent(unified_policy, "aggressive_pressure")
    train_against_scripted_opponent(unified_policy, "zone_defense")
```

### 5. Mixed Training with Multiple Past Opponents
Sample from multiple recent checkpoints during each training iteration:

```python
# During rollout, randomly assign different opponent versions
for env_idx in range(num_envs):
    lookback = np.random.randint(0, min(10, alternation))
    opponent_alt = alternation - lookback
    env.set_opponent(load_checkpoint(opponent_alt))
```

## Monitoring and Diagnostics

### Use ELO Evolution Analysis
```bash
python analytics/elo_evolution.py \
    --run-id <run_id> \
    --tournament-mode full \
    --episodes 200 \
    --output-heatmap
```

Look for:
- **Red spots in heatmap** = Non-transitive relationships
- **Declining ELO** = Overtraining / forgetting
- **Plateau** = Convergence (good)

### Log Cross-Epoch Performance
Track performance against fixed checkpoints throughout training:

```python
# In train.py
if alternation % eval_freq == 0:
    for ref_alt in [0, 20, 40, 60, 80]:
        if ref_alt <= alternation:
            win_rate = quick_eval(current_policy, checkpoints[ref_alt])
            mlflow.log_metric(f"cross_epoch/vs_alt_{ref_alt}", win_rate, step=alternation)
```

## Recommended Default Configuration

For robust self-play training:

1. **Maintain opponent pool of 10-20 past checkpoints** (evenly spaced)
2. **Sample opponents with 80% current, 20% from pool**
3. **Run cross-epoch evaluation every 10 alternations**
4. **Alert if win rate vs any past opponent (>20 alternations ago) drops below 50%**
5. **Use ELO analysis post-training to verify monotonic improvement**

## References

- "Emergent Complexity via Multi-Agent Competition" (OpenAI, 2017)
- "Stabilizing Experience Replay for Deep Multi-Agent RL" (Foerster et al., 2017)
- "On the Evaluation of Learned Policies" (Henderson et al., 2018)

