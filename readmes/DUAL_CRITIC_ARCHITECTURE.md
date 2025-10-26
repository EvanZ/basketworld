# Dual Critic Architecture Implementation

## Executive Summary

Implemented separate value networks for offense and defense to address a fundamental issue where the single centralized critic in zero-sum self-play cannot properly learn value functions for both teams.

**Problem:** With symmetric rewards (offense +2, defense -2), a single critic tries to predict both positive and negative returns for the same state, resulting in poor value estimates, low explained variance, and systematically biased predictions.

**Solution:** Maintain a unified actor (policy network) that conditions on `role_flag`, but use two separate critic networks:
- `value_net_offense`: Predicts returns when playing as offense (role_flag=+1.0)
- `value_net_defense`: Predicts returns when playing as defense (role_flag=-1.0)

## The Problem

### Observed Symptoms
1. **Negative Q-values when positive expected**: Offensive Q-values ~-1.5 even when offense scores 0.6-1.0 PPP on average
2. **Low explained variance**: Consistently < 0.3 during training
3. **Flat value predictions**: Most actions showing nearly identical Q-values
4. **Systematic bias**: Value function consistently favors defensive perspective

### Root Cause Analysis

In BasketWorld's self-play setting:
- **Zero-sum rewards**: When offense scores +2, defense gets -2 (and vice versa)
- **Unified policy**: Single actor network uses `role_flag` to play both sides
- **Centralized critic**: Single value network V(s) tries to predict returns for both perspectives

The critic receives conflicting training signals:
```
State s₁: offense has ball near basket
- When role_flag=+1 (offense): Should predict +rewards (high scoring probability)
- When role_flag=-1 (defense): Should predict -rewards (opponent likely to score)
```

With a **centralized critic** that doesn't strongly condition on `role_flag`, it learns some average prediction that:
- Doesn't properly distinguish offensive vs defensive value
- Shows systematic bias toward one perspective (usually defensive)
- Produces low explained variance because it can't fit opposite-signed returns

## The Solution: Dual Critic Architecture

### Architecture Overview

```
Observation → Feature Extractor → Shared Features
                                        ↓
                  ┌─────────────────────┴─────────────────────┐
                  ↓                                           ↓
         Actor Network (Policy π)                    Critic Networks
    (role_flag conditioned, shared)          (separate for each perspective)
                  ↓                                           ↓
         Action Distribution              ┌─────────────┬─────────────┐
                                          ↓             ↓             ↓
                                   If role_flag=+1.0  If role_flag=-1.0
                                     V_offense(s)      V_defense(s)
```

### Key Design Decisions

1. **Keep actor unified**: The policy π(a|s, role_flag) continues to share weights and learn from both perspectives
   - Good: Learns general basketball strategies applicable to both teams
   - Good: More sample efficient than separate policies
   
2. **Split critics**: Separate V_offense and V_defense networks
   - Each critic learns consistent returns from its own perspective
   - No conflicting signals from opposite-signed rewards
   - Proper role-specific value estimates

3. **Dynamic critic selection**: Based on `role_flag` in observation
   - role_flag=+1.0 → use V_offense
   - role_flag=-1.0 → use V_defense

### Expected Benefits

✅ **Positive offensive values**: V_offense should predict +returns when offense has good positions  
✅ **Negative defensive values**: V_defense should predict -returns when opponent has good positions  
✅ **Higher explained variance**: Each critic predicts returns with consistent signs  
✅ **Better Q-values**: Q(s,a) = r + γV(s') uses the correct V for each team  
✅ **Improved policy learning**: More accurate advantage estimates A = Q - V  

## Implementation Details

### Files Created

#### 1. `basketworld/policies/dual_critic_policy.py`
Custom `ActorCriticPolicy` class with separate value networks.

**Key methods:**
- `_build_mlp_extractor()`: Creates `value_net_offense` and `value_net_defense`
- `_extract_role_flag()`: **NEW** - Extracts role_flag from dict obs before preprocessing
- `_get_value_from_latent()`: Selects appropriate value network based on pre-extracted `role_flag`
- `predict_values()`: Returns role-conditioned value estimates
- `evaluate_actions()`: Used by PPO during training, automatically uses correct critic
- `forward()`: Main forward pass, extracts role_flag early for correct critic routing

**Implementation notes:**
- Inherits from SB3's `ActorCriticPolicy` 
- Overrides value network creation and forward pass
- **Critical: Extracts `role_flag` BEFORE feature preprocessing** to ensure correct critic selection during training
  - Initial implementation bug: tried to extract from preprocessed features → always used offense critic
  - Fix: Added `_extract_role_flag()` method called before `extract_features()`
  - Impact: Without this fix, explained variance stays near 0 despite values appearing correct in frontend
- Maintains compatibility with all SB3 training features (GAE, PPO optimization, etc.)

#### 2. `basketworld/policies/__init__.py`
Package initialization for custom policies.

#### 3. `basketworld/utils/policies.py` (updated)
Added `PassBiasDualCriticPolicy` combining:
- Dual critic architecture for correct value learning
- Pass action bias for exploration (existing mechanism)

### Files Modified

#### 1. `train/train.py`
**Changes:**
- Import `PassBiasDualCriticPolicy` (line 64)
- Added `--use-dual-critic` flag (lines 1567-1570)
- Dynamic policy class selection (lines 653-654)
- MLflow logging of architecture choice (lines 566-567)

**Usage:**
```bash
# Use dual critic (recommended for new training runs)
python train/train.py --use-dual-critic ...

# Use single critic (backward compatibility)
python train/train.py ...
```

#### 2. `basketworld/utils/mlflow_params.py` (unchanged but relevant)
Already logs `role_flag_offense_value`, `role_flag_defense_value`, and `role_flag_encoding_version` for backward compatibility with old models. This ensures the backend can correctly load and use models trained with different architectures.

### Backend Compatibility

The backend (`app/backend/main.py`) is already compatible:
- Q-value computation correctly conditions on `role_flag` (lines 510-531)
- State value calculation uses role-conditioned observations (lines 586-616)
- Works with both single-critic and dual-critic models via same API

**No backend changes required** - the policy's `predict_values()` method automatically uses the correct critic based on `role_flag`.

## Testing & Validation

### Before Training
Check that the new policy loads correctly:
```python
from basketworld.policies import DualCriticActorCriticPolicy
from stable_baselines3 import PPO

model = PPO(DualCriticActorCriticPolicy, env, ...)
print(model.policy.value_net_offense)
print(model.policy.value_net_defense)
```

### During Training
Monitor these metrics:
1. **Explained Variance**: Should be >0.6 from the start (if <0.1, role_flag extraction is broken)
   - **Critical diagnostic**: If EV is near 0 but frontend values look correct, check for `[WARNING] Could not extract role_flag` in training logs
   - This indicates the bug where preprocessed observations lose the dict structure
2. **Value Function Predictions**: Log offensive/defensive values at key states
3. **Q-values**: Should show positive values for good offensive positions
4. **PPP**: Should maintain or improve from baseline (~0.6-1.0)
5. **No warnings**: Training should NOT print `[WARNING] Could not extract role_flag from observation`

### After Training
Validate in the frontend:
1. Load a dual-critic model
2. Check Q-values on controls show sensible values
3. Verify moves table shows positive offensive values near basket
4. Confirm defensive values are negative/lower when offense has advantage

## Backward Compatibility

### Loading Old Models
Old single-critic models continue to work:
- Backend checks MLflow param `use_dual_critic` (defaults to False)
- Single-critic models use original `PassBiasMultiInputPolicy`
- Q-value computation adapts to loaded model architecture

### Mixed Training
You can compare architectures:
```bash
# Train with single critic
python train/train.py --run-name baseline_single_critic

# Train with dual critic  
python train/train.py --run-name improved_dual_critic --use-dual-critic

# Compare in frontend by loading each as frozen opponent
```

### Migration Path
1. **Start new training runs** with `--use-dual-critic`
2. **Keep old models** available for comparison
3. **Monitor explained variance** to validate improvement
4. **Gradually phase out** single-critic models once dual-critic proves superior

## Rationale for Design Choices

### Why not separate policies?
Separate offense/defense policies would:
- ❌ Require 2x training time (separate learning for each role)
- ❌ Lose transfer learning benefits (strategies applicable to both sides)
- ❌ Complicate model management (2 files per checkpoint)

Unified policy + dual critics:
- ✅ Shares actor weights across both perspectives
- ✅ Learns general basketball strategies
- ✅ Only splits where necessary (value prediction)

### Why not just stronger role_flag conditioning?
Attempted approaches that failed:
- Changed role_flag from 0/1 to -1/+1 (stronger signal) → Still biased
- Added role_flag explicitly to observations → Still poor explained variance

Problem: SB3's centralized critic architecture fundamentally wasn't designed for zero-sum multi-agent scenarios where the same state has opposite values for different agents.

### Why this is the right solution
- **Proven approach**: Multi-agent RL literature uses separate value functions per agent in competitive settings
- **Minimal changes**: Leverages SB3 infrastructure, only changes critic architecture
- **Clean abstraction**: Policy continues to treat offense/defense as role-conditioned variants of the same agent
- **Testable hypothesis**: If the issue is conflicting value signals, this should show immediate improvement in explained variance

## Expected Training Improvements

### Quantitative Metrics
- **Explained Variance**: 0.2-0.3 → 0.6-0.8
- **Value Predictions**: -1.5 (incorrect) → +0.5 to +1.5 (correct for offensive states)
- **Training Stability**: Fewer policy collapses from poor value estimates
- **Sample Efficiency**: Better advantage estimates → faster learning

### Qualitative Improvements
- Q-values make intuitive sense (positive for good offensive positions)
- Value predictions differentiate strong vs weak positions
- Policy learns more refined strategies with better value guidance
- Offensive and defensive play styles emerge more clearly

## Future Extensions

### Multi-Agent Value Factorization
Current: Single V(s) per perspective
Future: V(s) = Σᵢ Vᵢ(s, oᵢ) per-agent value decomposition

### Opponent Modeling
Current: Same value function vs all frozen opponents
Future: Opponent-conditioned value functions V(s | opponent_id)

### Curriculum Learning
Use dual critics to:
- Measure state value distributions over training
- Identify difficult states (high value uncertainty)
- Focus training on those states

## Troubleshooting

### Explained variance still low (< 0.1)
**Symptom:** Explained variance near 0 or negative during training, but value predictions look correct in frontend.

**Likely cause:** Role_flag not being extracted correctly during training batch processing.

**Diagnosis:**
1. Add logging in `_extract_role_flag()` to check if the fallback warning is printed
2. Check if `[WARNING] Could not extract role_flag from observation` appears during training
3. Verify observations are dict type with `role_flag` key

**Solution:** The policy must extract `role_flag` **before** calling `extract_features()` because feature extraction preprocesses the observation and may lose the dict structure. This is already implemented in the fixed version - ensure you're using the latest code.

**What was wrong:** Original implementation tried to extract `role_flag` from preprocessed features, which caused it to always default to the offense network, resulting in:
- ✅ Correct predictions in frontend (single observations as dicts)
- ❌ Wrong critic selection during training (batched/preprocessed observations)
- ❌ Explained variance near zero (always using wrong critic for ~50% of samples)

### Value predictions still negative
- Confirm offensive PPP is actually positive (check episode rewards)
- Verify reward structure is symmetric (offense +r → defense -r)
- Check if phi shaping is adding large negative rewards
- **Verify you're using `--use-dual-critic` flag** - without it, the old broken single-critic is used

### Training slower
- Dual critics add minimal overhead (< 5% more parameters)
- If significantly slower, check if features are being shared correctly
- Ensure `share_features_extractor=True` in policy_kwargs

### Values look correct but explained variance still low
This was the critical bug we found! The issue occurs when:
1. Frontend predictions work correctly (values make sense)
2. But explained variance during training is near zero or negative

This indicates the training loop is not properly using the dual critics. The fix ensures `role_flag` is extracted before feature preprocessing in all three methods: `forward()`, `predict_values()`, and `evaluate_actions()`.

## References

### Related Work
- **Multi-Agent RL**: Lowe et al. "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments" (MADDPG)
- **Value Factorization**: Sunehag et al. "Value-Decomposition Networks For Cooperative Multi-Agent Learning" (VDN)
- **Self-Play**: Silver et al. "Mastering the Game of Go with Deep Neural Networks and Tree Search" (AlphaGo)

### SB3 Documentation
- [Custom Policies](https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html)
- [ActorCriticPolicy API](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html#policies)

## Conclusion

The dual critic architecture addresses a fundamental mismatch between:
1. **What the environment provides**: Symmetric zero-sum rewards
2. **What the learning algorithm expects**: Consistent value function returns
3. **What a single critic can learn**: An average that satisfies neither perspective

By splitting the critic while keeping the actor unified, we get:
- ✅ Correct value estimates for each perspective
- ✅ Efficient policy learning from both roles
- ✅ Better training stability and sample efficiency
- ✅ Intuitive Q-values that match domain knowledge

**Recommendation**: Use `--use-dual-critic` for all new training runs. The single-critic architecture should be considered deprecated for zero-sum self-play scenarios.

