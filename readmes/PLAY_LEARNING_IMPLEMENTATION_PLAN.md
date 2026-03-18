# Emergent Play Learning Plan (No Handcrafted Playbook)

## Objective

Learn reusable multi-step offensive behaviors ("plays") directly from reward, without manually defining named plays, scripted phases, or basketball doctrine templates.

In this document, "play" means an emergent sequence pattern discovered by optimization, not a handcrafted finite state machine.

Maintain backward compatibility for both:

1. Playable app mode.
2. Development mode (`App.vue` / local backend workflows),

when using existing non-intent (legacy) models.

## Non-goals

1. No hardcoded plays like `GIVE_AND_GO` / `DRIVE_KICK`.
2. No hand-authored phase transitions.
3. No training setup where defense depends on privileged offense labels unavailable against humans.

## Compatibility requirements

1. Default-off behavior:
   - `enable_intent_learning` remains `False` by default.
   - Legacy training/eval/playable paths remain unchanged unless explicitly enabled.

2. API compatibility:
   - Keep action contracts unchanged for `/api/step` and `/api/playable/step`.
   - No required frontend payload changes for playable or dev mode.

3. Checkpoint compatibility:
   - Legacy checkpoints that were trained without intent features must continue to load and run.
   - Intent-enabled checkpoints should be clearly tagged via run metadata/params to avoid accidental mismatch.

4. Observation compatibility:
   - When intent learning is disabled, intent features are all zeros or omitted in a backward-compatible way.
   - Wrapper dimensions must match model expectations in both legacy and intent-enabled configurations.

5. Runtime safeguards:
   - Add explicit validation errors for model/observation shape mismatches.
   - Fail fast with actionable messages instead of silent misbehavior.

## Core approach

Use latent intent/options:

1. Sample or choose a latent intent variable `z` at possession start.
2. Keep `z` fixed for a commitment window.
3. Train primitive policy as `pi(a_t | s_t, z)`.
4. Add diversity pressure so different `z` values produce distinct trajectory modes.
5. Keep defense robust by training heavily on no-latent/unknown-latent episodes.

This produces emergent behavior modes that can later be interpreted as learned plays.

## Why this fits current codebase

Current constraints:

1. Environment action API is primitive per-player `MultiDiscrete`:
   - `HexagonBasketballEnv.action_space`
   - File: `basketworld/envs/basketworld_env_v2.py`

2. `SelfPlayEnvWrapper` expects primitive vectors for training and opponent:
   - File: `basketworld/utils/self_play_wrapper.py`

3. Episode currently corresponds to one possession (terminal on shot/turnover/violation):
   - Reward termination logic in `basketworld/envs/core/rewards.py`

Therefore: phase 1 should keep primitive action space unchanged and add only latent conditioning + losses.

## High-level design

## 1) Latent intent runtime state in env

Add environment-level latent state:

1. `intent_id` (discrete int in `[0, num_intents-1]`)
2. `intent_age`
3. `intent_commitment_remaining`
4. `intent_known` flag (for observability control)

Since training episodes are currently single-possession, sample `intent_id` in `reset()`.
If later you move to multi-possession episodes, sample at each possession boundary instead.

## 2) Policy conditioning

Condition low-level policy on `intent_id`:

1. Simple path (first): append scalar/one-hot intent features to observation.
2. Better path: learn an intent embedding and concatenate to actor/critic inputs.

The actor still outputs primitive `ActionType` actions.

## 3) Diversity objective (critical)

Without extra objective, all intents can collapse to one behavior.

Add one of:

1. DIAYN-style discriminator `q(z | trajectory_window)` with MI bonus.
2. Option-contrastive objective on trajectory embeddings.

Reward shape:

`r_total = r_env + beta_intent * r_diversity`

Keep `beta_intent` small and scheduled.

## 4) Defense asymmetry handling

Defense should not rely on offense-private intent labels.

Training distribution:

1. `p_known_intent` episodes: offense intent visible only when appropriate.
2. `p_unknown_intent` episodes: defense gets `intent_known=0`, `intent_id=0`.
3. `p_no_intent` episodes: offense itself runs with null intent.

At inference vs human:

1. Default `intent_known=0`
2. Defense behaves from geometry/risk cues, not privileged latent.

## File/class modification plan

## A) Environment

File: `basketworld/envs/basketworld_env_v2.py`
Class: `HexagonBasketballEnv`

Modify:

1. `__init__`:
   - Add args:
     - `enable_intent_learning: bool = False`
     - `num_intents: int = 8`
     - `intent_commitment_steps: int = 4`
     - `intent_visible_to_defense_prob: float = 0.0`
     - `intent_null_prob: float = 0.2`
     - `intent_obs_mode: str = "private_offense"`
   - Add state:
     - `self.intent_id`, `self.intent_age`, `self.intent_commitment_remaining`
     - `self.intent_known_for_training_obs`

2. `reset()`:
   - Initialize latent intent runtime:
     - Sample `intent_id` unless null intent sampled.
     - Reset age/counters.

3. `step()`:
   - Increment `intent_age`.
   - Decrement commitment counter.
   - Include intent diagnostics in `info` for logging:
     - `intent_id`, `intent_age`, `intent_known`.

4. Observation construction (`_get_observation` and related vector assembly):
   - Add intent features to observation with masking policy.

## B) Core state helpers

Likely new file: `basketworld/envs/core/intent.py`

Add:

1. `sample_intent_id(env)`
2. `mask_intent_for_perspective(env, role_flag, obs)`
3. `encode_intent_features(env)`

This keeps `basketworld_env_v2.py` from growing further.

## C) Self-play wrapper observability fix

File: `basketworld/utils/self_play_wrapper.py`
Class: `SelfPlayEnvWrapper`

Why needed:

`step()` currently flips `role_flag` for opponent observation by copying dict and changing only `role_flag`.

If intent visibility depends on perspective, wrapper must also rewrite intent fields for opponent obs.

Modify:

1. Add helper:
   - `_recondition_intent_fields_for_role(obs_dict, opponent_is_offense)`
2. Call it before opponent `predict()`.

This prevents accidental privileged leakage.

## D) Set observation wrappers

File: `basketworld/utils/wrappers.py`

Classes:

1. `SetObservationWrapper`
   - Increase `GLOBAL_DIM` and/or token dims to include intent features.
   - Add intent fields in `globals`.

2. `MirrorObservationWrapper`
   - Ensure intent fields are preserved and not corrupted during mirroring.

## E) Training config + env factory

Files:

1. `train/config.py`
   - Add CLI args:
     - `--enable-intent-learning`
     - `--num-intents`
     - `--intent-commitment-steps`
     - `--intent-null-prob`
     - `--intent-visible-to-defense-prob`
     - `--intent-diversity-beta`
     - `--intent-diversity-warmup-steps`
     - `--intent-noise-prob`

2. `train/env_factory.py`
   - Forward new args into `HexagonBasketballEnv(...)`.

3. `train/train.py`
   - Log all intent params to MLflow.
   - Add training mixture schedule for known/unknown/no-intent episodes.

## F) Policy classes

### Phase 1 (minimal invasive)

No new policy head required if intent is in observation vector and policy can consume it.

Touched classes only for dimensional compatibility:

1. `PassBiasMultiInputPolicy` (`basketworld/utils/policies.py`)
2. `PassBiasDualCriticPolicy` (`basketworld/utils/policies.py`)
3. `SetAttentionDualCriticPolicy` (`basketworld/policies/set_attention_policy.py`)

### Phase 2 (recommended for quality)

Add explicit intent embedding module:

1. Multi-input policies:
   - Add `nn.Embedding(num_intents, d_intent)` and concatenate in actor/critic trunks.

2. Set-attention policy:
   - Add intent embedding into `globals` processing path.

## G) Diversity/discovery auxiliary module

New file suggestion:

`basketworld/utils/intent_discovery.py`

Add:

1. `TrajectoryEncoder`
2. `IntentDiscriminator`
3. `compute_intent_diversity_bonus(...)`

Then integrate via callbacks or custom PPO training step.

### Current implementation status

The codebase currently implements a simplified first-pass discriminator, not the richer
`TrajectoryEncoder` design described above.

Files:

1. `basketworld/utils/intent_discovery.py`
2. `basketworld/utils/callbacks.py`

What is implemented today:

1. `IntentEpisodeBuffer`
   - Stores per-step transitions until an episode ends.

2. `IntentTransition.feature`
   - Per step, build one flat feature vector by concatenating:
     - flattened observation for that env index
     - flattened action for that env index

3. `compute_episode_embeddings(...)`
   - Stack all step features in the episode.
   - Take the mean over time.
   - Pad or truncate to a fixed length of `max_obs_dim + max_action_dim`.
   - Current callback defaults:
     - `max_obs_dim = 256`
     - `max_action_dim = 16`
     - total discriminator input dim = `272`

4. `IntentDiscriminator`
   - A small MLP:
     - `Linear(input_dim, hidden_dim)`
     - `ReLU`
     - `Dropout`
     - `Linear(hidden_dim, num_intents)`
   - This is a classifier over the mean-pooled episode embedding.

5. Diversity bonus
   - Compute `log q(z | episode_embedding) + log(num_intents)`.
   - Normalize with a running mean/std.
   - Clip.
   - Multiply by `beta`.
   - Spread evenly across the episode steps in the rollout buffer.

Important limitation of the current implementation:

1. Temporal order is discarded.
2. Pass-receive order, cut timing, and multi-step structure are not modeled explicitly.
3. The discriminator is therefore weaker than the original plan's intended
   trajectory-aware version.

### Worked numerical example of the current discriminator

This example uses small vectors for readability. The real code uses up to `256`
observation dims plus `16` action dims before padding/truncation.

Assume:

1. `num_intents = 8`
2. true sampled intent for this episode is `z = 2`
3. episode length is `3` steps
4. for illustration, we show only the first `4` observation dims and first `2` action dims

Per-step features:

1. Step 0
   - flattened observation slice = `[0.80, -0.25, 0.33, 1.00]`
   - flattened action slice = `[4.00, 0.00]`
   - concatenated step feature = `[0.80, -0.25, 0.33, 1.00, 4.00, 0.00]`

2. Step 1
   - flattened observation slice = `[0.78, -0.10, 0.40, 1.00]`
   - flattened action slice = `[7.00, 1.00]`
   - concatenated step feature = `[0.78, -0.10, 0.40, 1.00, 7.00, 1.00]`

3. Step 2
   - flattened observation slice = `[0.75, 0.05, 0.52, 1.00]`
   - flattened action slice = `[9.00, 0.00]`
   - concatenated step feature = `[0.75, 0.05, 0.52, 1.00, 9.00, 0.00]`

Episode embedding:

1. Stack the 3 step vectors.
2. Mean-pool over time:

`x_episode = mean(step_0, step_1, step_2)`

`x_episode = [0.7767, -0.1000, 0.4167, 1.0000, 6.6667, 0.3333]`

3. In the real implementation, this vector is then padded/truncated to length `272`.

Classifier pass:

1. Feed `x_episode` into the MLP.
2. Suppose the discriminator outputs logits:

`[0.2, -0.4, 1.1, 0.3, -0.2, 0.0, -0.5, 0.1]`

3. Softmax gives intent probabilities:

`q(z | episode) = [0.116, 0.064, 0.286, 0.128, 0.078, 0.095, 0.058, 0.105]`

4. Since the true intent is `z = 2`, the chosen probability is:

`q(z=2 | episode) = 0.286`

Raw diversity bonus:

1. Current formula:

`bonus_raw = log q(z | episode) + log(num_intents)`

2. With `num_intents = 8`:

`bonus_raw = log(0.286) + log(8)`

`bonus_raw = -1.2528 + 2.0794 = 0.8266`

Normalization and reward injection:

1. Suppose the running bonus statistics at this point are:
   - running mean = `0.5000`
   - running std = `0.2500`

2. Normalized bonus:

`bonus_norm = (0.8266 - 0.5000) / 0.2500 = 1.3064`

3. Suppose clip range is `[-2, 2]`, so clipped bonus stays `1.3064`.

4. Suppose current `beta = 0.05` and episode length is `3`:

`per_step_bonus = beta * clipped_bonus / episode_length`

`per_step_bonus = 0.05 * 1.3064 / 3 = 0.0218`

5. The callback adds `+0.0218` to each of the 3 offense steps in the rollout buffer for this
   episode before recomputing returns and advantages.

Interpretation:

1. If the discriminator can confidently identify the sampled intent from the episode summary,
   the episode receives positive shaping reward.
2. If the discriminator is uncertain, the raw bonus is smaller or negative.
3. Because the current embedding is only a mean over flat step features, the discriminator can
   miss sequence structure even when the policy is using the latent.

### Proposed trajectory encoder upgrade (GRU first)

The simplest meaningful upgrade is to replace mean pooling with a small GRU over the
step sequence.

Current path:

1. `step feature`
2. mean over time
3. MLP
4. logits over intents

Proposed path:

1. `step feature`
2. step encoder MLP
3. GRU over time
4. final hidden state
5. discriminator head
6. logits over intents

This keeps the reward-shaping logic unchanged while preserving order information.

#### Files and classes to change

File: `basketworld/utils/intent_discovery.py`

Add or replace:

1. `build_padded_episode_batch(...)`
   - Input: `List[CompletedIntentEpisode]`
   - Output:
     - `x_steps: [B, T_max, D]`
     - `lengths: [B]`
     - `y: [B]`

2. `StepEncoder`
   - Small MLP applied independently to each step.
   - Example:
     - `Linear(D, d_step)`
     - `ReLU`
     - optional dropout

3. `TrajectoryEncoderGRU`
   - `GRU(input_size=d_step, hidden_size=d_hidden, batch_first=True)`
   - Returns one episode embedding per sequence.

4. `IntentDiscriminator`
   - Wrap:
     - step encoder
     - GRU
     - final classifier head

File: `basketworld/utils/callbacks.py`

Modify:

1. `_train_discriminator(...)`
   - Train on padded sequences plus lengths, not a pre-averaged matrix.

2. `_compute_episode_bonus(...)`
   - Run the same GRU discriminator forward pass and compute:
     - `log q(z | trajectory) + log(num_intents)`

3. Keep rollout-buffer reward injection exactly the same.

#### Tensor shapes

Recommended first cut:

1. raw step feature dim:
   - `D = 272`
   - from `256` obs dims + `16` action dims

2. step encoder dim:
   - `d_step = 64`

3. GRU hidden dim:
   - `d_hidden = 128`

4. intents:
   - `K = 8`

For a minibatch of `B = 32` episodes with max episode length `T_max = 6`:

1. raw padded batch:
   - `x_steps.shape = [32, 6, 272]`

2. after step MLP:
   - `x_step_emb.shape = [32, 6, 64]`

3. after GRU:
   - `h_seq.shape = [32, 6, 128]`

4. final episode embedding:
   - `h_episode.shape = [32, 128]`

5. logits:
   - `logits.shape = [32, 8]`

#### Variable-length handling

Episodes are short but not identical in length, so the GRU should use sequence lengths.

Implementation approach:

1. Build padded tensor with zeros on the right.
2. Build `lengths` array with true episode lengths.
3. Use `torch.nn.utils.rnn.pack_padded_sequence(...)`.
4. Use the GRU final hidden state `h_n[-1]` as the episode embedding.

That avoids letting padding steps affect the episode representation.

#### Numerical example

Suppose one episode has 3 steps and each step feature is already reduced to 4 dims for
illustration only:

1. `x1 = [0.8, 0.1, 0.0, 1.0]`
2. `x2 = [0.4, 0.9, 0.0, 0.0]`
3. `x3 = [0.2, 0.3, 1.0, 0.0]`

Let the GRU hidden size be 2. Conceptually:

1. `h1 = GRU(x1, h0)`
   - `h1 = [0.30, -0.10]`

2. `h2 = GRU(x2, h1)`
   - `h2 = [0.52, 0.08]`

3. `h3 = GRU(x3, h2)`
   - `h3 = [0.71, 0.41]`

Episode embedding:

1. `h_episode = h3 = [0.71, 0.41]`

Classifier head:

1. logits over 4 intents for illustration:
   - `[0.1, -0.3, 1.2, 0.0]`

2. softmax:
   - `[0.171, 0.115, 0.518, 0.196]`

3. if true intent is `z = 2`:
   - `q(z=2 | trajectory) = 0.518`

Compare with the same three steps in a different order:

1. `x1 = [0.2, 0.3, 1.0, 0.0]`
2. `x2 = [0.4, 0.9, 0.0, 0.0]`
3. `x3 = [0.8, 0.1, 0.0, 1.0]`

The mean of the three vectors is identical, so the current implementation cannot
distinguish them after pooling.

But the GRU hidden-state path will generally be different:

1. `h1' = [0.12, 0.28]`
2. `h2' = [0.33, 0.35]`
3. `h3' = [0.44, 0.09]`

So:

1. current discriminator:
   - same episode embedding for both orders
   - same logits

2. GRU discriminator:
   - different final hidden state
   - different logits

This is exactly why a trajectory encoder is a better fit for learned plays.

#### What stays the same

1. `IntentEpisodeBuffer` can stay.
2. Per-episode bonus formula can stay.
3. Running mean/std normalization can stay.
4. Reward injection into rollout buffer can stay.
5. Logged metrics can stay:
   - `intent/disc_loss`
   - `intent/disc_top1_acc`
   - `intent/disc_auc_ovr_macro`
   - `intent/beta_current`

#### Minimal config additions

If we implement this cleanly, add discriminator-specific config like:

1. `--intent-disc-encoder-type gru`
2. `--intent-disc-step-dim 64`
3. `--intent-disc-hidden-dim 128`
4. `--intent-disc-dropout 0.1`

Defaults should keep the current MLP/mean-pooling path available for ablations.

#### Implementation checklist (ready to code)

This is the concrete order of work to replace the current mean-pooled discriminator with
the GRU version while minimizing risk.

##### Step 1: Extend `intent_discovery.py`

Add these classes and functions.

1. New step encoder:

```python
class StepEncoder(nn.Module):
    def __init__(self, input_dim: int, step_dim: int, dropout: float = 0.1) -> None: ...
    def forward(self, x_steps: torch.Tensor) -> torch.Tensor: ...
```

Expected shapes:

1. input: `x_steps.shape == [B, T, D]`
2. output: `x_step_emb.shape == [B, T, d_step]`

2. New GRU trajectory encoder:

```python
class TrajectoryEncoderGRU(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1, dropout: float = 0.0) -> None: ...
    def forward(self, x_steps: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor: ...
```

Expected behavior:

1. use `pack_padded_sequence`
2. return final hidden state with shape `[B, d_hidden]`

3. Replace current `IntentDiscriminator` with an encoder-aware version:

```python
class IntentDiscriminator(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_intents: int,
        encoder_type: str = "mlp_mean",
        hidden_dim: int = 128,
        step_dim: int = 64,
        dropout: float = 0.1,
    ) -> None: ...

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor: ...
```

Expected modes:

1. `encoder_type == "mlp_mean"`
   - preserve current behavior for ablation/backward comparison
2. `encoder_type == "gru"`
   - use `StepEncoder + TrajectoryEncoderGRU + classifier head`

4. Add padded-batch builder:

```python
def build_padded_episode_batch(
    episodes: List[CompletedIntentEpisode],
    max_obs_dim: int = 256,
    max_action_dim: int = 16,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ...
```

Return:

1. `x_steps`: shape `[B, T_max, D]`
2. `lengths`: shape `[B]`
3. `labels`: shape `[B]`

Implementation notes:

1. reuse the current `IntentTransition.feature`
2. pad on the right with zeros
3. skip empty episodes
4. keep `D = max_obs_dim + max_action_dim`

##### Step 2: Update discriminator training in `callbacks.py`

Modify the current callback methods.

1. `_maybe_build_discriminator(...)`

Current:

```python
def _maybe_build_discriminator(self, input_dim: int) -> None: ...
```

Recommended:

```python
def _maybe_build_discriminator(self, input_dim: int) -> None: ...
```

Same signature, but instantiate `IntentDiscriminator` with:

1. `encoder_type`
2. `step_dim`
3. `hidden_dim`
4. `dropout`

2. `_train_discriminator(...)`

Current:

```python
def _train_discriminator(self, x_np: np.ndarray, y_np: np.ndarray) -> tuple[float, float]: ...
```

Recommended replacement:

```python
def _train_discriminator(
    self,
    x_np: np.ndarray,
    y_np: np.ndarray,
    lengths_np: np.ndarray | None = None,
) -> tuple[float, float]:
    ...
```

Behavior:

1. if `encoder_type == "mlp_mean"`:
   - ignore `lengths_np`
   - preserve current path
2. if `encoder_type == "gru"`:
   - sample minibatch indices over episodes
   - pass both `xb` and `lengths_b` into discriminator

3. `_compute_episode_bonus(...)`

Current:

```python
def _compute_episode_bonus(self, x_np: np.ndarray, y_np: np.ndarray) -> np.ndarray: ...
```

Recommended replacement:

```python
def _compute_episode_bonus(
    self,
    x_np: np.ndarray,
    y_np: np.ndarray,
    lengths_np: np.ndarray | None = None,
) -> np.ndarray:
    ...
```

4. `_compute_disc_auc(...)`

Current:

```python
def _compute_disc_auc(self, x_np: np.ndarray, y_np: np.ndarray) -> Optional[float]: ...
```

Recommended replacement:

```python
def _compute_disc_auc(
    self,
    x_np: np.ndarray,
    y_np: np.ndarray,
    lengths_np: np.ndarray | None = None,
) -> Optional[float]:
    ...
```

##### Step 3: Replace rollout-end feature preparation

Current rollout-end path:

```python
x_np, y_np = compute_episode_embeddings(...)
```

Replace with:

```python
if self.disc_encoder_type == "gru":
    x_np, lengths_np, y_np = build_padded_episode_batch(...)
else:
    x_np, y_np = compute_episode_embeddings(...)
    lengths_np = None
```

Then pass `lengths_np` through:

1. `_train_discriminator(...)`
2. `_compute_disc_auc(...)`
3. `_compute_episode_bonus(...)`

##### Step 4: Add config plumbing

File: `train/config.py`

Add CLI args:

```python
parser.add_argument("--intent-disc-encoder-type", choices=["mlp_mean", "gru"], default="mlp_mean")
parser.add_argument("--intent-disc-step-dim", type=int, default=64)
parser.add_argument("--intent-disc-hidden-dim", type=int, default=128)
parser.add_argument("--intent-disc-dropout", type=float, default=0.1)
```

File: `train/callbacks.py`

Pass the new args into the diversity callback constructor.

File: `basketworld/utils/mlflow_params.py`

Parse and expose the new fields so dev mode can inspect them.

##### Step 5: Keep metrics and reward shaping unchanged

Do not change these initially:

1. `intent/disc_loss`
2. `intent/disc_top1_acc`
3. `intent/disc_auc_ovr_macro`
4. `intent/bonus_raw_mean`
5. `intent/bonus_norm_mean`
6. `intent/beta_current`

Do not change:

1. `RunningMeanStd`
2. clipping logic
3. reward injection into rollout buffer
4. return/advantage recomputation

This isolates the experiment to the discriminator representation.

##### Step 6: Add tests before long runs

Add:

1. `tests/test_intent_discovery_gru.py`

Recommended test cases:

1. `build_padded_episode_batch` returns correct shapes for variable-length episodes
2. padded timesteps do not change the final hidden-state embedding
3. `IntentDiscriminator(..., encoder_type="gru")` returns logits of shape `[B, K]`
4. reordered step sequences produce different GRU embeddings
5. `encoder_type="mlp_mean"` path still matches current behavior

2. Update:

1. `tests/test_intent_diversity_callback.py`
   - cover the new `lengths_np` path
   - ensure bonus computation still returns finite values

##### Step 7: Diagnostic run sequence

Run these in order:

1. `encoder_type = mlp_mean`
   - confirm baseline reproduces current behavior

2. `encoder_type = gru`, `beta = 0`
   - confirm code path runs
   - confirm no NaNs / shape errors

3. `encoder_type = gru`, diversity enabled
   - compare:
     - `intent/disc_auc_ovr_macro`
     - `intent/disc_top1_acc`
     - `intent/policy_kl_mean`
     - `intent/action_flip_rate`
     - PPP

##### Acceptance criteria for the GRU upgrade

Treat the GRU upgrade as justified if:

1. `policy_kl_mean` remains nonzero
2. `disc_auc_ovr_macro` rises above the mean-pooling baseline
3. `disc_top1_acc` rises above the mean-pooling baseline
4. PPP does not collapse further versus the current intent-enabled baseline

If AUC/top1 do not improve materially over mean pooling, then the next bottleneck is
likely the step features themselves, not just sequence aggregation.

## H) Callbacks + metrics

Files:

1. `basketworld/utils/callbacks.py`
2. `train/callbacks.py`

Add metrics:

1. `intent_usage_entropy`
2. `intent_balance_minmax`
3. `intent_mutual_info_estimate`
4. `intent_cluster_ppp` (points per possession by intent)
5. `defense_no_intent_eval_score`

## Training curriculum

Recommended schedule:

1. Stage 0:
   - `beta_intent=0`
   - intent present but no diversity reward
   - verify stability

2. Stage 1:
   - warm up `beta_intent` from `0 -> target`
   - enforce intent commitment window

3. Stage 2:
   - increase unknown/no-intent fraction for defense robustness
   - periodic eval against intent-free offense

4. Stage 3:
   - keep low-level play execution `pi(a | s, z)` and learned latent play space
   - introduce a learned high-level play selector `mu(z | s_context)`
   - ramp selector influence from uniform intent sampling to selector-driven intent sampling
   - preserve coverage regularization so previously learned plays do not collapse

## Stage 3 extension: learned play selection `mu(z | s)`

Once the latent play space is clearly formed, the next architectural step is to learn
which play to call, not just how to execute a sampled play.

Current system:

1. offense intent `z` is sampled externally
2. policy learns `pi(a_t | s_t, z)`
3. diversity objective shapes the latent play space

Stage 3 system:

1. high-level selector learns `mu(z | s_context)`
2. selector chooses or samples `z` at possession start
3. low-level policy executes `pi(a_t | s_t, z)` for the commitment window

This turns the current latent play discovery setup into a hierarchical play-calling
architecture.

### Why stage it instead of learning `mu` from the beginning

Jointly learning:

1. what each play means
2. how to execute it
3. when to select it

is possible, but substantially less stable.

Main failure modes:

1. selector collapse
   - `mu` quickly concentrates on a few lucky intents
   - other intents receive too little data to become useful plays

2. moving semantics
   - while `pi(a | s, z)` is still changing what each `z` means, `mu` is trying to
     optimize on top of a drifting codebook

3. premature rejection
   - an intent that starts weak can be abandoned before it has enough experience to improve

The current random-sampling phase is therefore best understood as forced coverage for
playbook formation.

### High-level selector design

The minimal selector should choose offense intent once per possession from a compact
context representation:

`z ~ mu(z | s_context)`

Recommended `s_context` contents:

1. offense player positions
2. defense player positions
3. ball holder
4. shot clock / game clock context
5. score differential / period context if available

The selector does not need the full primitive action history. It only needs enough
context to decide which latent play is appropriate for the current possession start.

### Recommended implementation choice

Recommended first implementation:

1. reuse the existing offense state encoder / set-attention trunk
2. add a separate selector head on top of that shared representation
3. output categorical logits over `num_intents`
4. call the selector once per possession, not every step
5. keep the selected `z` fixed for the existing commitment window

This is the best first version because:

1. it matches the semantics of a play call
2. it reuses the strongest existing court-state representation
3. it avoids duplicating a second full state encoder
4. it is easier to stabilize than a per-step selector

So the intended architecture is:

1. shared state encoder
2. selector head `mu(z | s_context)`
3. low-level action policy `pi(a | s, z)`

and not:

1. a fully separate selector network as the first implementation
2. a per-step selector that changes `z` every timestep

Per-step `mu(z_t | s_t)` is possible in principle, but without extra switching costs or
dwell constraints it stops behaving like a play caller and starts behaving like a noisy
latent mode switcher.

### Runtime sampling schedule

Do not hard-switch from:

1. `z ~ Uniform`

to:

2. `z ~ mu(z | s_context)`

Instead, use a mixture schedule:

`z ~ (1 - alpha) * Uniform + alpha * mu(z | s_context)`

where:

1. `alpha = 0` during latent play discovery
2. `alpha` is ramped upward only after the play space is stable

Recommended selector ramp:

1. `0M - 150M`:
   - `alpha = 0`
   - pure random coverage

2. `150M - 250M`:
   - ramp `alpha: 0 -> 1`
   - keep selector entropy high
   - optionally freeze or partially freeze low-level policy for part of this window

3. `250M+`:
   - allow selector-dominant play calling
   - optionally fine-tune jointly

### Training objective with selector

Low-level policy remains conditioned on `z`:

`pi(a_t | s_t, z)`

High-level selector receives task reward through the chosen play and should also be
regularized for coverage:

`L_total = L_PPO_low_level + lambda_sel * L_selector + lambda_cov * L_usage_reg`

Where:

1. `L_PPO_low_level`
   - normal PPO objective for the primitive policy

2. `L_selector`
   - selector objective for choosing better plays under task reward
   - implementation can be PPO-style if the selector is treated as a once-per-possession action

3. `L_usage_reg`
   - entropy/KL/usage regularization to prevent immediate collapse onto a few intents

### How `mu` actually learns which play to call

The selector should be treated as a slower-timescale policy whose action is the choice
of latent play at possession start.

Mechanism:

1. selector sees possession-start context `s_context`
2. selector chooses `z`
3. low-level policy executes the possession under `pi(a | s, z)`
4. possession ends and produces return `R_possession`
5. selector is updated so that choices of `z` that led to better returns in that context
   become more likely

So the learning signal is:

1. not "which play looks good in the abstract"
2. but "which play choice improved downstream possession reward in this context"

This is why Stage 3 only makes sense after the latent play space is already somewhat
stable. If `z` semantics are still drifting, then `mu` is trying to optimize over labels
whose meaning is moving underneath it.

Recommended credit assignment model:

1. treat selector choice as a once-per-possession categorical action
2. assign selector return from the resulting possession return
3. optimize selector with PPO-style policy gradient or equivalent episodic objective
4. keep low-level `pi(a | s, z)` training separate but concurrent

In other words:

1. `mu` learns play selection from possession outcomes
2. `pi(a | s, z)` learns play execution from step-level PPO updates

This separation of timescales is the core reason the architecture is learnable.

### Recommended stabilization strategy

When `mu` turns on:

1. initialize selector with near-uniform logits
2. keep high selector entropy early
3. add KL-to-uniform or minimum-usage regularization
4. use lower LR for low-level `pi(a | s, z)` during the first selector phase
5. consider freezing low-level policy for a short warm-start window while `mu` begins learning

This makes Stage 3 mostly about learning play selection, not rewriting the meaning of the
latent plays again from scratch.

### File/class modification plan for `mu`

#### A) Environment intent sampling

File: `basketworld/envs/basketworld_env_v2.py`

Current reset path samples `intent_id` from RNG.

For Stage 3, refactor reset-time offense intent sampling into a strategy hook:

1. default:
   - uniform random sampler (current behavior)

2. selector-enabled:
   - query high-level selector for possession-start offense intent

Recommended helper:

1. `sample_offense_intent_id(...)`
2. `sample_defense_intent_id(...)` if defense selector is ever added later

#### B) High-level selector module

New file suggestion:

`basketworld/utils/intent_selector.py`

Add:

1. `IntentSelector`
   - selector head over the existing shared offense encoder representation
   - outputs logits over `num_intents`
   - intended to run once per possession

2. `sample_intent_with_mixture(...)`
   - combines uniform prior with selector distribution using current `alpha`

3. `compute_selector_usage_metrics(...)`
   - selector entropy
   - selector top-1 frequency
   - per-intent usage histogram

#### C) Training integration

Files:

1. `train/train.py`
2. `train/callbacks.py`
3. optionally `basketworld/utils/callbacks.py`

Add:

1. selector creation and optimizer
2. selector schedule params:
   - `selector_enabled`
   - `selector_alpha_start`
   - `selector_alpha_end`
   - `selector_alpha_warmup_steps`
   - `selector_alpha_ramp_steps`
   - `selector_entropy_coef`
   - `selector_usage_reg_coef`

3. MLflow logging:
   - `intent/selector_entropy`
   - `intent/selector_usage_by_intent/<z>`
   - `intent/selector_top1_by_intent/<z>`
   - `intent/selector_alpha_current`

#### D) Policy/runtime boundary

Keep the low-level policy interface unchanged:

1. policy still receives `z` in observation/globals
2. low-level policy still outputs primitive actions
3. selector head only runs at possession start / play-selection time

This keeps Stage 3 compatible with the current low-level architecture and isolates the
new complexity to possession-start play selection.

### Clean integrated implementation target

The callback-based selector prototype is acceptable for smoke testing, but it is not the
final architecture. The clean solution is to keep `mu` inside the same policy object and
the same PPO optimization pass, while still treating it as a slower-timescale decision.

Desired end state:

1. one shared set-attention encoder
2. one low-level action head for primitive actions
3. one high-level selector head `mu(z | s_context)`
4. one PPO training loop that optimizes both heads together

That means:

1. `mu` is not a separate standalone policy/network
2. `mu` is also not trained as a callback-side afterthought
3. selector log-prob, entropy, and advantage should be first-class rollout data

#### Why this is the clean solution

It removes the main ambiguity in the prototype:

1. selector is produced by the same policy object that owns the low-level action head
2. selector exploration/entropy is part of the same training step
3. selector credit assignment is explicit in rollout storage, not inferred later by a callback
4. selector and low-level policy can share a coherent optimizer/update schedule

In other words, this is still "one policy" in the architectural sense:

1. shared encoder
2. multiple heads
3. unified training graph

#### Recommended algorithm structure

Add a custom PPO variant for hierarchical play selection.

New file suggestion:

`basketworld/algorithms/hierarchical_intent_ppo.py`

This should subclass SB3 PPO rather than trying to force the logic through callbacks.

Recommended helper classes:

1. `HierarchicalIntentRolloutBuffer`
2. `SelectorDecisionRecord`

The reason is simple: selector decisions are not emitted every timestep, so they do not
fit cleanly into the existing per-step-only rollout schema without explicit bookkeeping.

#### Rollout-time semantics

Selector should run only at possession start or play-selection boundary.

For each environment `env_idx`:

1. detect whether this timestep is a selector boundary
   - initially: episode start / possession start
   - later: optionally commitment-window boundary

2. build selector context from the current observation
   - clone observation
   - neutralize current intent fields so selector does not read the already-sampled `z`
   - feed this context into selector head

3. compute selector distribution
   - `p_sel = mu(z | s_context)`
   - optionally mix with uniform:
     - `p_mix = (1 - alpha) * Uniform + alpha * p_sel`

4. sample or argmax a play `z`
   - during training: sample from `p_mix`
   - during deterministic eval: argmax or deterministic mixture rule

5. write selected `z` into env state and observation fields
   - this becomes the play seen by the low-level policy for the commitment window

6. store one selector-decision record
   - selector log-prob
   - selector entropy
   - selected `z`
   - boundary timestep
   - env index
   - optional selector value estimate if using a selector critic

Then normal low-level PPO rollout collection continues step-by-step.

#### Recommended rollout buffer additions

Minimal buffer additions:

1. `selector_episode_start_mask[t, env]`
   - whether a selector decision happened here

2. `selector_chosen_z[t, env]`
   - chosen intent ID for this boundary

3. `selector_log_prob[t, env]`
   - log-prob of chosen `z` under the selector distribution used at rollout time

4. `selector_entropy[t, env]`
   - entropy of selector distribution at decision time

5. `selector_alpha[t, env]`
   - current mixture coefficient `alpha`

6. `selector_return[t, env]`
   - possession/window return assigned later

7. `selector_advantage[t, env]`
   - normalized advantage used for selector loss

Recommended if doing full PPO-style selector learning:

8. `selector_value[t, env]`
   - selector baseline estimate

9. `selector_value_target[t, env]`
   - target for selector value loss

10. `selector_valid_mask[t, env]`
   - whether this timestep holds a real selector decision to optimize

#### Recommended selector credit assignment

The selector action should receive credit from the possession it initiated.

For the initial clean implementation:

1. selector decision happens at possession start
2. possession ends with return `R_possession`
3. assign `R_possession` back to the selector decision that chose `z`

Two acceptable versions:

1. REINFORCE-style integrated loss
   - `A_selector = normalized(R_possession)`
   - simplest first clean version

2. PPO-style selector critic
   - add selector value head
   - `A_selector = R_possession - V_selector(s_context)`
   - preferred long-term version

Recommended implementation order:

1. first clean version: integrated REINFORCE-style selector loss
2. second refinement: add selector critic and clipped PPO-style selector objective

#### Recommended policy changes

File:

`basketworld/policies/set_attention_policy.py`

The policy already has the right high-level shape:

1. shared set-attention encoder
2. low-level action head
3. selector head over shared features

For the clean version, extend policy API with explicit selector methods:

1. `get_selector_context(obs)`
   - returns shared context tensor for selector

2. `get_intent_selector_distribution(obs, alpha)`
   - returns selector distribution and optionally mixture distribution

3. `sample_intent_selector(obs, alpha, deterministic=False)`
   - returns:
     - selected `z`
     - selector log-prob
     - selector entropy
     - optional selector value

These should be used by the custom PPO algorithm during rollout collection, not by a callback.

#### Recommended training losses

Unified objective should look like:

`L_total = L_PPO_low_level + lambda_sel * L_selector_policy + lambda_sel_v * L_selector_value - lambda_sel_ent * H_selector + lambda_usage * L_usage_reg`

Where:

1. `L_PPO_low_level`
   - existing primitive-action PPO loss

2. `L_selector_policy`
   - selector policy gradient term
   - REINFORCE-style first, PPO-style later if selector critic is added

3. `L_selector_value`
   - optional selector baseline loss

4. `H_selector`
   - selector entropy bonus

5. `L_usage_reg`
   - KL-to-uniform or equivalent coverage penalty over selector usage

Important:

1. low-level PPO entropy and selector entropy are distinct terms
2. both live in the same total loss, but they regularize different heads

#### Recommended file modification plan for the clean version

1. `basketworld/policies/set_attention_policy.py`
   - expose selector distribution/log-prob/value helpers

2. `basketworld/algorithms/hierarchical_intent_ppo.py`
   - custom PPO subclass
   - rollout collection with selector-boundary handling
   - unified selector + low-level loss computation

3. `basketworld/algorithms/hierarchical_rollout_buffer.py`
   - selector decision storage
   - possession-return assignment
   - selector advantage computation

4. `basketworld/envs/basketworld_env_v2.py`
   - selector boundary hook remains possession start initially
   - later optional support for commitment-window reselection

5. `train/train.py`
   - instantiate hierarchical PPO when selector is enabled in clean mode

6. `train/config.py`
   - keep selector schedule/regularization flags
   - add explicit switch between:
     - callback prototype
     - clean integrated PPO path

#### Recommended implementation order

1. keep current callback selector only as a temporary prototype
2. implement custom rollout buffer for selector decisions
3. implement custom PPO subclass that samples `z` at possession start
4. move selector loss out of callback and into PPO train step
5. add selector critic only after REINFORCE-style integrated version is stable

#### Success criteria for the clean version

The clean implementation should replace the prototype only if it achieves:

1. same or better latent-play separability
2. same or better PPP / possession efficiency
3. selector metrics that are easier to interpret
4. no callback-side mutation of rollout starts
5. explicit first-class selector data in rollout storage

### Evaluation criteria for `mu`

Treat the selector as successful if:

1. latent play separability remains strong
   - `disc_auc_ovr_macro`
   - `disc_top1_acc`

2. task performance improves
   - PPP / win rate / turnover rate

3. selector does not collapse immediately
   - healthy selector entropy
   - nontrivial usage across intents during ramp

4. counterfactual play selection makes sense in fixed states
   - same state, different selected `z`, different continuation
   - similar states, selector chooses similar plays

### Practical recommendation

Do not implement Stage 3 until:

1. latent play space is visibly present in metrics and analysis
2. at least some intents are behaviorally interpretable
3. low-level execution is stable enough that `z` semantics are not drifting wildly

At the current stage of the project, Stage 3 should be treated as the next architecture
branch after latent play discovery is sufficiently mature, not as a config toggle to
turn on immediately.

## Real examples (emergent, not handcrafted)

## Example 1: What a learned intent looks like in practice

After training with `num_intents=8`, analysis might show:

1. `z=2`:
   - early pass frequency high
   - shot locations mostly perimeter
   - longer possession length

2. `z=6`:
   - immediate rim pressure
   - higher turnover risk, higher dunk rate

No template was coded; this is inferred from trajectory statistics.

## Example 2: Human offense (no declared intent)

At inference vs human:

1. System sets offense intent as null/unknown.
2. Defense sees:
   - `intent_known=0`
   - neutral `intent_id`
3. Defense still performs because it was trained with unknown/no-intent episodes.

## Example 3: Detecting mode collapse

Bad run symptom:

1. `intent_usage_entropy` high but trajectory clusters overlap heavily.
2. Discriminator accuracy near random.
3. Performance differences by intent nearly zero.

Fix:

1. Raise `beta_intent` gradually.
2. Increase commitment window.
3. Add short trajectory encoder context (not single-step classifier).

## Tests to add

1. `tests/test_intent_state_reset.py`
   - intent sampled/reset correctly each episode.

2. `tests/test_intent_observation_masking.py`
   - offense/defense visibility rules hold, including wrapper role-flip path.

3. `tests/test_self_play_wrapper_intent_reconditioning.py`
   - opponent obs in `SelfPlayEnvWrapper` does not leak private intent.

4. `tests/test_intent_disabled_backward_compat.py`
   - with feature disabled, behavior and obs dims remain backward compatible.

5. `tests/test_intent_diversity_bonus_smoke.py`
   - auxiliary reward term finite and stable.

6. `app/backend/tests/test_playable_legacy_model_compat.py`
   - playable routes work with legacy non-intent models and unchanged request schema.

7. `app/backend/tests/test_dev_mode_legacy_model_compat.py`
   - lifecycle/dev routes work with legacy non-intent models.

## Rollout plan

1. Implement phase 1 only (latent state + observation + masking + logging).
2. Run short smoke training and verify no regressions.
3. Add diversity objective (phase 2) and track intent separability.
4. Harden defense with unknown/no-intent curriculum.
5. Add post-hoc clustering notebook/report to interpret learned intents.
6. After latent play space stabilizes, add Stage 3 selector `mu(z | s_context)` with
   mixture ramp from uniform to selector-driven play calling.

## Success criteria

1. Behavior modes are statistically distinct across `z` without handcrafted labels.
2. Overall reward/PPP improves or stays neutral versus baseline.
3. Defense remains strong when offense intent is unknown (human proxy).
4. Turning off intent learning returns baseline behavior cleanly.
5. Playable app and development mode remain functional with legacy non-intent models and unchanged API contracts.

## Future work

### 1) Full DIAYN skill pretraining track

If joint task+diversity training under PPO does not produce strong intent separation, add a two-stage pipeline:

1. Stage A: skill discovery pretraining in the same environment dynamics with task reward disabled (intrinsic-only objective).
2. Stage B: downstream task training where PPO learns to use discovered skills for scoring/winning.

Design notes:

1. Keep latent skill/intent `z` discrete (`num_intents = K`) with commitment window.
2. Train discriminator `q(z | trajectory)` and maximize MI-style bonus.
3. Preserve environment dynamics/opponents/observation schema so transfer is not confounded by distribution shift.
4. Evaluate pretraining quality before transfer using:
   - `intent/disc_top1_acc`
   - `intent/disc_auc_ovr_macro`
   - `intent/usage_entropy`
   - per-intent behavior spread metrics.
5. Transfer options:
   - Freeze low-level skill-conditioned policy and train only high-level chooser.
   - Or fine-tune low-level policy with small LR after warmup.

Implementation caution:

1. Canonical DIAYN uses entropy-maximizing actor-critic (often SAC). Since current stack is PPO + discrete actions, this should be treated as a dedicated research branch, not an in-place toggle.
2. Keep backward compatibility by version-tagging checkpoints and observation schema.

### 2) Learned play selector track

If the latent play space becomes stable and interpretable, add the high-level
selector described in Stage 3:

1. selector chooses offense play `z`
2. low-level policy executes `pi(a | s, z)`
3. selector is ramped in gradually from uniform sampling

This should be treated as a hierarchical-control branch, not folded into the original
latent-play discovery code path without clear ablations.

### 3) Defensive latent plays track

Extend latent play learning to defense with its own latent variable, separate from offense.

Core idea:

1. Offense uses `z_off`.
2. Defense uses `z_def`.
3. Each side gets its own embedding/discriminator/diversity bonus.

Why separate latents first:

1. Reduces role interference versus forcing one shared latent to serve both teams.
2. Makes diagnostics clearer (`offense_*` metrics vs `defense_*` metrics).

Proposed training sequence:

1. Stabilize offense-intent learning first.
2. Enable defense-intent with smaller `beta` and longer warmup/ramp.
3. Add role-specific monitoring:
   - `intent/offense_disc_*`, `intent/defense_disc_*`
   - role-specific usage entropy/min-prob
   - defense robustness vs unknown offense intent.

Security/observability rule:

1. No privileged leakage across roles:
   - defense never directly observes offense-private latent embedding;
   - offense never directly observes defense-private latent embedding unless an explicit experiment enables it.

Longer-term generalization:

1. Explore shared latent space with role conditioning (`z + role_flag`) only after separate-latent baselines are stable.
2. Compare against separate-latent baseline on robustness and exploitability before adopting shared-latent design.

## ChatGPT 5.4 Critique of Plan

1. The biggest gap is backward compatibility. The plan says intent features can be “all zeros or omitted” when disabled, but the current runtime builds fixed observation spaces, and existing checkpoints depend on those exact shapes. See [PLAY_LEARNING_IMPLEMENTATION_PLAN.md#L36](/home/ubuntu/basketworld/readmes/PLAY_LEARNING_IMPLEMENTATION_PLAN.md#L36), [basketworld_env_v2.py#L316](/home/ubuntu/basketworld/basketworld/envs/basketworld_env_v2.py#L316), and [wrappers.py#L36](/home/ubuntu/basketworld/basketworld/utils/wrappers.py#L36). I would not modify the legacy schema in place. Add an explicit `obs_schema_version` and only construct intent-augmented spaces for intent-enabled runs.

2. The null/unknown encoding is flawed as written. The plan defines `intent_id` in `[0, num_intents-1]` and later uses `intent_id=0` for unknown/no-intent episodes, which collides with a real learned intent. See [PLAY_LEARNING_IMPLEMENTATION_PLAN.md#L78](/home/ubuntu/basketworld/readmes/PLAY_LEARNING_IMPLEMENTATION_PLAN.md#L78) and [PLAY_LEARNING_IMPLEMENTATION_PLAN.md#L117](/home/ubuntu/basketworld/readmes/PLAY_LEARNING_IMPLEMENTATION_PLAN.md#L117). Use separate fields like `intent_index`, `intent_active`, and `intent_visible`, or reserve a dedicated null index outside the learned range.

3. The implementation surface is wider than the file plan says. Updating `SelfPlayEnvWrapper` is necessary, but it is not sufficient. The backend has multiple role-conditioned observation clone paths that would also need intent reconditioning and schema awareness: [self_play_wrapper.py#L121](/home/ubuntu/basketworld/basketworld/utils/self_play_wrapper.py#L121), [observations.py#L22](/home/ubuntu/basketworld/app/backend/observations.py#L22), [evaluation.py#L140](/home/ubuntu/basketworld/app/backend/evaluation.py#L140), and [lifecycle_routes.py#L96](/home/ubuntu/basketworld/app/backend/routes/lifecycle_routes.py#L96). The plan should explicitly include these files, plus [mlflow_params.py#L26](/home/ubuntu/basketworld/basketworld/utils/mlflow_params.py#L26), or the backend will reconstruct the wrong env for saved runs.

4. Phase 2 is not implementation-ready yet. “Integrate via callbacks or custom PPO training step” is the hard part, not a small detail. See [PLAY_LEARNING_IMPLEMENTATION_PLAN.md#L260](/home/ubuntu/basketworld/readmes/PLAY_LEARNING_IMPLEMENTATION_PLAN.md#L260) versus current stock SB3 PPO usage in [train.py#L388](/home/ubuntu/basketworld/train/train.py#L388) and [train.py#L835](/home/ubuntu/basketworld/train/train.py#L835). You need to specify where trajectory windows are stored, when the discriminator is updated, and how the bonus is injected into the rollout buffer.

5. The commitment-window state looks premature for phase 1. With one possession per episode and `intent_id` sampled in `reset()`, `intent_commitment_remaining` has no real effect unless phase 1 also allows mid-possession resampling or a higher-level chooser. See [PLAY_LEARNING_IMPLEMENTATION_PLAN.md#L49](/home/ubuntu/basketworld/readmes/PLAY_LEARNING_IMPLEMENTATION_PLAN.md#L49), [PLAY_LEARNING_IMPLEMENTATION_PLAN.md#L80](/home/ubuntu/basketworld/readmes/PLAY_LEARNING_IMPLEMENTATION_PLAN.md#L80), and [PLAY_LEARNING_IMPLEMENTATION_PLAN.md#L83](/home/ubuntu/basketworld/readmes/PLAY_LEARNING_IMPLEMENTATION_PLAN.md#L83). I would remove it from the first cut.

**Suggestions**

- Split this into `Phase 1A` and `Phase 2`. `Phase 1A`: env state, schema/versioning, backend/wrapper conditioning, logging, tests. No diversity loss yet.
- For set-attention models, avoid stuffing a large one-hot into `globals` if you can. `globals` gets broadcast to every player token in [set_attention_policy.py#L37](/home/ubuntu/basketworld/basketworld/policies/set_attention_policy.py#L37) and [set_attention_policy.py#L105](/home/ubuntu/basketworld/basketworld/policies/set_attention_policy.py#L105). A compact flag plus learned embedding is cleaner.
- Add explicit runtime validation that compares model metadata against env schema before the first `predict()`, not just generic shape-mismatch errors.
- Expand tests to include MLflow param round-trip and backend role-conditioned clone helpers, not just env reset and self-play wrapper tests.
- Tighten success criteria with thresholds: separability metric, PPP delta vs baseline, and unknown-intent defense degradation budget.

The overall direction is good and it does fit this codebase. I would call it good enough to implement a narrowed phase 1 after the compatibility and schema issues are fixed. I would not start the diversity/discovery phase from this plan as-is.

## Critique from Codex 5.3

**Critical Findings**
1. `intent_id` semantics conflict: you define `intent_id` in `[0..num_intents-1]`, but later use `intent_id=0` for unknown/no-intent, which collides with a real learned intent ([plan](/home/ubuntu/basketworld/readmes/PLAY_LEARNING_IMPLEMENTATION_PLAN.md#L78), [plan](/home/ubuntu/basketworld/readmes/PLAY_LEARNING_IMPLEMENTATION_PLAN.md#L117)).
2. Backward-compatibility is under-specified for observation shape. Current env/wrappers use fixed dict spaces and fixed dims; adding intent fields in-place can break legacy checkpoints ([env obs space](/home/ubuntu/basketworld/basketworld/envs/basketworld_env_v2.py#L378), [set wrapper dims](/home/ubuntu/basketworld/basketworld/utils/wrappers.py#L8), [plan](/home/ubuntu/basketworld/readmes/PLAY_LEARNING_IMPLEMENTATION_PLAN.md#L36)).
3. `SelfPlayEnvWrapper` changes are necessary but not sufficient. Role-conditioned obs cloning also exists in backend inference/eval paths and must be updated too, or you’ll get train/serve mismatch ([self-play wrapper](/home/ubuntu/basketworld/basketworld/utils/self_play_wrapper.py#L121), [backend observations](/home/ubuntu/basketworld/app/backend/observations.py#L22), [backend evaluation](/home/ubuntu/basketworld/app/backend/evaluation.py#L140)).
4. Diversity objective integration is too vague for implementation. “callbacks or custom PPO step” is the hard part; with current SB3 flow this needs explicit rollout-buffer integration details ([plan](/home/ubuntu/basketworld/readmes/PLAY_LEARNING_IMPLEMENTATION_PLAN.md#L260), [training loop](/home/ubuntu/basketworld/train/train.py#L835)).
5. Commitment-window state adds complexity without effect in current single-possession episodes; it should be deferred unless you actually resample intent intra-episode ([plan](/home/ubuntu/basketworld/readmes/PLAY_LEARNING_IMPLEMENTATION_PLAN.md#L83), [plan](/home/ubuntu/basketworld/readmes/PLAY_LEARNING_IMPLEMENTATION_PLAN.md#L49)).

**What I’d Change**
1. Split into phases: `Phase 1A` (schema-safe latent plumbing + masking + logging + tests), `Phase 1B` (training mixture schedule), `Phase 2` (diversity/discovery objective).
2. Introduce explicit observation schema/version metadata and hard validation before `predict()` to prevent silent mismatches.
3. Replace ambiguous intent encoding with separate fields: `intent_index`, `intent_active`, `intent_visible` (or reserve a dedicated null index outside learned range).
4. Expand file plan to include backend runtime/config paths (`app/backend/observations.py`, `app/backend/evaluation.py`, `app/backend/routes/lifecycle_routes.py`, `basketworld/utils/mlflow_params.py`), not only env/wrapper/train files.
5. Add concrete PPO integration design for diversity reward: where trajectory windows live, update frequency, and how bonus enters rollout rewards.
6. Tighten success criteria with thresholds (separability metric, PPP delta, unknown-intent robustness delta).

The plan is good directionally and implementable after these adjustments. I’d approve starting a narrowed `Phase 1A` now, and block `Phase 2` until the PPO integration spec is concrete.
