# Critic Transfer Learning

## Overview

This feature enables **transfer learning for value networks** when training new policies. By initializing a new policy's critic weights from a pre-trained model, we can significantly accelerate training and reduce the initial "thrashing" phase where both actor and critic learn from scratch.

## Motivation

In reinforcement learning, the critic (value function) learns to estimate expected returns from states. When training a fresh policy:

1. **Cold start problem**: Both actor and critic start with random weights
2. **Early instability**: Random value estimates lead to noisy policy gradients
3. **Slower convergence**: The critic must learn value functions from scratch even when similar policies have already learned them

By transferring a pre-trained critic:
- Skip the early phase of learning basic value estimates
- Provide stable policy gradients from the start
- Accelerate convergence to better policies
- Leverage knowledge from previous training runs

## How It Works

### What Gets Transferred

**The complete value network** including feature extraction is transferred:
- ✅ Value feature extractor (`mlp_extractor.value_net`) - processes observations into latent features
- ✅ Offense critic head (`value_net_offense`) - final layer predicting offense value
- ✅ Defense critic head (`value_net_defense`) - final layer predicting defense value
- ❌ Policy feature extractor (`mlp_extractor.policy_net`) - remains random for actor
- ❌ Actor/policy network (randomly initialized)

This ensures the value networks have both good feature representations AND good value estimates, while the actor learns from scratch.

### Safety Checks

The implementation includes several architecture compatibility checks:

1. **Dual critic requirement**: Source run must use dual critic; target automatically enables it
2. **Architecture verification**: Compares `net_arch` between source and target
3. **Dimension matching**: Uses PyTorch's `load_state_dict()` which validates weight dimensions
4. **Graceful failure**: Falls back to random initialization if transfer fails

### Usage

```bash
python train/train.py \
  --init-critic-from-run <SOURCE_RUN_ID> \
  [other training args...]
```

**Note:** Specifying `--init-critic-from-run` automatically enables dual critic mode (no need for `--use-dual-critic`).

**Example:**
```bash
# Train a new policy with critic initialized from run abc123
python train/train.py \
  --init-critic-from-run 2b1c63b5c1bd470286c4953dc82a5afc \
  --total-timesteps 10000000 \
  --n-envs 16
```

### What Happens During Training

1. **Initialization Phase**:
   - New policy created with random weights (both actor and critic)
   - Source model loaded from MLflow using the provided `run_id`
   - Architecture compatibility verified
   - Critic weights copied from source to target
   - Actor remains randomly initialized

2. **Training Phase**:
   - Critic **continues learning** (not frozen)
   - Actor learns from scratch with guidance from warm-started critic
   - Both networks adapt to each other through normal PPO updates

3. **MLflow Logging**:
   - `critic_transfer_enabled`: `true` or `false`
   - `critic_transfer_source_run`: Source run ID
   - `critic_transfer_error`: Error message if transfer failed

## Expected Benefits

### Faster Initial Learning
- **Better value estimates from step 1**: Pre-trained critic already understands state values
- **Stable policy gradients**: Reduces noise in early training updates
- **Skip exploration phase**: Critic doesn't need to learn basic value patterns

### Improved Sample Efficiency
- **Fewer wasted samples**: Early training steps are more productive
- **Faster convergence**: Policy can focus on improvement rather than exploration
- **Better asymptotic performance**: Potentially reach higher final skill levels

### Reduced Training Time
- **10-30% time savings**: Expected based on typical transfer learning gains
- **Fewer total timesteps needed**: May reach target performance earlier
- **Lower compute costs**: Less wallclock time and GPU usage

## Requirements and Limitations

### Requirements
- **Source must use dual critic**: Cannot transfer from single-critic models
- **Target auto-enables dual critic**: Automatically enabled when `--init-critic-from-run` is specified
- **Compatible architectures**: Same `net_arch` for value networks (checked automatically)
- **MLflow access**: Must have access to source run's artifacts

### Limitations
- **Only for dual critic**: Single-critic models not supported
- **Same observation space**: Source and target must have identical obs spaces
- **Policy feature extractor not transferred**: Actor's feature extractor remains random
- **No actor transfer**: Policy network always starts fresh
- **Architecture must match**: `net_arch` for value network must be compatible

### When to Use
✅ **Good use cases:**
- Training variants of similar policies (e.g., different hyperparameters)
- Continuing after architecture changes to actor
- Exploring different training strategies with same value function
- Fine-tuning for different opponent distributions

❌ **Not recommended:**
- Significantly different environment parameters
- Different observation spaces or features
- When you want to study learning dynamics from scratch

## Implementation Details

### Architecture Compatibility Check

The system performs a multi-level compatibility check:

```python
# 1. Auto-enable dual critic if transfer learning is requested
if args.init_critic_from_run is not None:
    args.use_dual_critic = True

# 2. Check source uses dual critic
if not source_use_dual_critic:
    raise ValueError("Source run must use dual critic architecture")

# 3. Extract and compare value network architectures
source_vf_arch = extract_vf_arch(source_net_arch_used)
target_vf_arch = extract_vf_arch(target_net_arch_used)

# 4. Warn if architectures differ (but attempt transfer anyway)
if str(source_vf_arch) != str(target_vf_arch):
    print("WARNING: Value network architectures differ!")

# 5. PyTorch validates weight dimensions during load_state_dict()
target_policy_net.value_net_offense.load_state_dict(
    source_policy_net.value_net_offense.state_dict()
)
```

### Weight Transfer Process

```python
# Load source model
source_policy = PPO.load(source_model_path, custom_objects={...})

# Extract policy networks
source_policy_net = source_policy.policy
target_policy_net = unified_policy.policy

# Transfer value feature extractor (critical!)
if hasattr(source_policy_net.mlp_extractor, 'value_net'):
    target_policy_net.mlp_extractor.value_net.load_state_dict(
        source_policy_net.mlp_extractor.value_net.state_dict()
    )

# Copy offense critic head
target_policy_net.value_net_offense.load_state_dict(
    source_policy_net.value_net_offense.state_dict()
)

# Copy defense critic head
target_policy_net.value_net_defense.load_state_dict(
    source_policy_net.value_net_defense.state_dict()
)
```

**Key insight:** Transferring only the final value heads is insufficient because they operate on latent features from `mlp_extractor.value_net`. Without transferring the feature extractor, the value heads receive random/meaningless features and cannot produce good estimates.

### Error Handling

The implementation uses defensive programming:
- All operations wrapped in try-except
- Detailed logging at each step
- Graceful fallback to random initialization on failure
- MLflow logs success/failure and error messages

## Example Output

### Successful Transfer
```
================================================================================
[Critic Transfer] Loading critic weights from run: 2b1c63b5c1bd470286c4953dc82a5afc
================================================================================
[Critic Transfer] Loading source model from: mlartifacts/.../unified_policy_final
[Critic Transfer] Source net_arch: [{'pi': [256, 256], 'vf': [256, 256]}]
[Critic Transfer] Target net_arch: [{'pi': [256, 256], 'vf': [256, 256]}]
[Critic Transfer] Transferring offense critic weights...
[Critic Transfer] Transferring defense critic weights...
[Critic Transfer] ✓ Successfully transferred critic weights from run: 2b1c63b5c1bd470286c4953dc82a5afc
[Critic Transfer] Actor network remains randomly initialized for fresh policy learning.
================================================================================
```

### Architecture Mismatch Warning
```
[Critic Transfer] WARNING: Value network architectures differ!
  Source vf: [256, 256]
  Target vf: [128, 128]
[Critic Transfer] Attempting transfer anyway - weights will be copied where dimensions match.
[Critic Transfer] ERROR: Failed to transfer critic weights: Error loading state_dict...
[Critic Transfer] Continuing with randomly initialized critics.
```

## Best Practices

1. **Use compatible architectures**: Ensure source and target have same `--net-arch-vf`
2. **Same observation space**: Don't change environment features between runs
3. **Document transfer chains**: Keep track of which runs were used as sources
4. **Monitor explained variance**: Check that critic maintains good performance
5. **Compare with baseline**: Run control experiments without transfer to measure benefit

## Future Enhancements

Possible extensions:
- Support for feature extractor transfer
- Partial weight transfer (fine-tuning)
- Transfer from single-critic to dual-critic (with duplication)
- Automatic architecture adaptation (reshape weights)
- Transfer learning curriculum (progressive unfreezing)

## Related Documentation

- [Dual Critic Architecture](./DUAL_CRITIC_ARCHITECTURE.md)
- [Training Guide](../README.md)
- [MLflow Setup](../scripts/REMOTE_MLFLOW_QUICKSTART.md)

