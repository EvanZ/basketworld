# Single-Step State-Only DIAYN Plan

## Goal

Replace the current trajectory-level, state+action discriminator with a discriminator that is:

1. **single-step**
2. **state-only**
3. **set/token based**
4. **closer to vanilla DIAYN**

Target behavior:

- each active offense step contributes one labeled discriminator example
- the discriminator predicts play `z` from a single state
- the intrinsic reward is computed per step, not per completed segment
- the discriminator input respects the permutation/invariance goals of the token/set-attention design

## Critical Update

On `2026-04-06` we confirmed that the current `set_step` discriminator could
recover much of its AUC from temporal-global shortcut features rather than
geometry-driven play structure.

Main shortcut features:

- `shot_clock`
- `pressure_exposure`

Observed post-hoc evidence on the saved eval batch:

- full baseline AUC: `0.6916`
- mask `has_ball`: `0.7065`
- mask ball-identity bundle: `0.6979`
- position only: `0.5196`
- position + role only: `0.5100`
- mask `shot_clock + pressure_exposure`: `0.4850`
- `shot_clock` alone in the feature-family sweep: `0.7410`

Interpretation:

1. the main shortcut was not ball-holder identity
2. the main shortcut was temporal-global leakage
3. prior discriminator AUC should not be treated as clean evidence of
   geometry-based play separation

Immediate design consequence for this plan:

- a valid single-step state-only discriminator should **exclude**
  `shot_clock` and `pressure_exposure` from the discriminator input path
- those features may still remain available to the policy; this restriction is
  discriminator-specific

Operational follow-up:

- future discriminator eval should also move from **per-step shuffled holdout**
  toward **episode-level / rollout-blocked holdout**

## Why Change It

The current discriminator path has several issues:

1. **Trajectory-based data introduces nuisance structure**
   - sequence length
   - padding
   - segment boundary definitions
   - per-segment reward spreading back across steps

2. **Current input is flat and player-order sensitive**
   - discriminator sees fixed player-index slots
   - this conflicts with the invariance goal of the set-attention policy architecture

3. **Action features can dominate**
   - the discriminator may learn action signatures rather than state consequences of a play

4. **This is farther from standard DIAYN**
   - vanilla DIAYN predicts the skill from state
   - not from an entire trajectory segment

## Target Design

### Training Example

One training example should be:

- `x = single state`
- `y = current play z`

Recommended default:

- use **post-transition state** `s_{t+1}`

Reason:

- `s_{t+1}` is the consequence of behavior under the current play
- it is a cleaner fit to the DIAYN intuition than pre-action state alone

### Input Features

Use:

- `players`
- `globals`
- `role_flag` if needed

But for `globals`, exclude temporal shortcut channels:

- do **not** include `shot_clock`
- do **not** include `pressure_exposure`

Do **not** use:

- actions
- action masks
- flattened player-index vectors as the primary path
- time-based / cumulative globals that make intent classification trivial

### Encoder

Use a set/token encoder aligned with the policy observation design.

The discriminator should use a **similar set-attention encoder model to the current policy architecture**.

That means:

- player tokens
- globals concatenated or injected in the same general style
- small transformer-style self-attention block(s)
- pooled context vector for classification

The goal is architectural consistency:

- the low-level policy already uses a compact set-attention transformer-style encoder
- the new discriminator should follow the same set-based modeling assumptions
- this avoids the current mismatch where the policy is set-based but the discriminator is flat and order-sensitive

Recommended first version:

1. player-token MLP
2. optional CLS token(s)
3. self-attention block(s)
4. pooled context vector
5. linear classifier over `num_intents`

This can be:

- a new dedicated discriminator encoder
- or a small reused/simplified version of the set-attention extractor

Preferred direction:

- implement a **similar but separate** set-attention encoder, using the policy extractor as the architectural template

Do **not** start by reusing the full low-level policy trunk directly.

Reason:

- discriminator and policy should stay separable
- reuse of exact policy features can create coupling that is harder to reason about

## Reward Definition

Per active offense step:

- compute logits from discriminator on the step state
- compute intrinsic reward from:
  - `log q(z | s_{t+1}) - log p(z)`

For uniform prior over plays:

- `log p(z) = -log(num_intents)`

Then:

1. normalize with running mean/std
2. clip
3. multiply by current `beta`
4. add directly to that step’s rollout reward

This replaces the current pattern:

- train on completed segments
- compute one segment bonus
- spread it across active-prefix steps

## Scope

### In Scope

- discriminator data model
- discriminator encoder
- discriminator reward path
- logging/eval changes needed to support per-step batches

### Out of Scope

- selector redesign
- play-boundary redesign
- UI changes
- changing nominal play conditioning itself

## Proposed Architecture

### 1. New Step Example Container

Add a new lightweight step-level record type, conceptually similar to the current transition/episode types but much simpler.

Suggested fields:

- `feature_obs`
  - the set observation dict for one env at one step, preferably `s_{t+1}`
- `intent_index`
- `intent_active`
- `role_is_offense`
- `env_idx`
- `buffer_step_idx`
- `global_step`
- optional diagnostics:
  - `shot_clock`
  - `ball_holder`
  - `done`
  - `boundary_reason`

Important:

- do not flatten to fixed player-index vectors at collection time
- preserve the set observation structure until batching

### 2. New Step-Level Batching Path

Add batching utilities that:

1. stack dict observations into batched tensors
2. stack integer labels
3. optionally stack provenance arrays for eval export

This should replace the current `build_padded_episode_batch(...)` / `compute_episode_embeddings(...)` path for the new mode.

### 3. New Discriminator Encoder Type

Introduce a new discriminator encoder mode, for example:

- `disc_encoder_type = "set_step"`

Behavior:

- input is a batched dict observation
- output is logits over `num_intents`

Recommended minimal architecture:

1. token projection for each player token
2. optional projection of globals into each token or separate CLS token initialization
3. self-attention block(s)
4. pooled context:
   - CLS token if present
   - otherwise mean over player tokens
5. classifier head

Optional auxiliary heads can remain future work.

### 4. New Callback Data Flow

Current callback behavior:

- wait for completed intent episodes
- build segment batch
- train discriminator
- compute segment bonus
- write bonus back across active-prefix rollout steps

New callback behavior:

1. on each rollout step, collect active offense step examples
2. at rollout end:
   - build a step batch
   - split train/holdout by step or by env-step provenance
   - train discriminator on step examples
   - compute per-step intrinsic bonus
   - add the bonus directly to matching rollout-buffer rewards
3. recompute returns/advantages

Recommended filtering:

- offense steps only
- `intent_active == True`
- current policy source only, if that remains the desired filter

### 5. Logging / Metrics

Keep:

- discriminator loss
- top-1 accuracy
- AUC

Add:

- number of active offense steps in discriminator batch
- mean intrinsic reward per step
- mean normalized/clipped intrinsic reward
- holdout metrics for step-level examples

Optional but useful:

- accuracy/AUC by play
- accuracy/AUC by shot-clock bucket
- accuracy/AUC by step position within possession, if that metadata is available

## Implementation Plan

### Phase 0. Guardrails

Before changing the main path:

1. keep the current discriminator implementation intact
2. add the new one behind a config flag / encoder mode
3. make it easy to A/B compare old vs new behavior on the same nominal-conditioning architecture

Suggested config additions:

- `--disc-granularity step|trajectory`
- `--disc-state-only true|false`
- `--disc-step-state-source next_obs|current_obs`
- `--disc-encoder-type set_step|gru|mlp`

Recommended first run:

- `disc-granularity=step`
- `disc-state-only=true`
- `disc-step-state-source=next_obs`
- `disc-encoder-type=set_step`

### Phase 1. Data Plumbing

Files likely involved:

- [basketworld/utils/callbacks.py](/home/evanzamir/basketworld/basketworld/utils/callbacks.py)
- [basketworld/utils/intent_discovery.py](/home/evanzamir/basketworld/basketworld/utils/intent_discovery.py)

Tasks:

1. add a step-level example structure
2. collect single-step state examples during rollout
3. preserve set observation structure instead of flattening
4. build batched tensors for the new discriminator mode

Design requirement:

- the collection path must not depend on top-level legacy `intent_*` observation keys

### Phase 2. Encoder Implementation

Files likely involved:

- discriminator model definition currently used by callback training
- [basketworld/policies/set_attention_policy.py](/home/evanzamir/basketworld/basketworld/policies/set_attention_policy.py) for architectural reference only

Tasks:

1. add a set-based step discriminator model
2. support dict observations as model input
3. return logits over `num_intents`

Design requirement:

- keep this encoder independent from the main policy extractor implementation
- but keep it intentionally similar in structure to the policy’s set-attention encoder

### Phase 3. Reward Path Refactor

Files likely involved:

- [basketworld/utils/callbacks.py](/home/evanzamir/basketworld/basketworld/utils/callbacks.py)

Tasks:

1. compute intrinsic reward per step instead of per segment
2. add it directly to matching rollout-buffer steps
3. remove segment-bonus spreading when the new mode is active
4. recompute returns/advantages

Design requirement:

- new mode should not rely on `active_prefix_length`

### Phase 4. Evaluation / Export

Files likely involved:

- analytics scripts that currently expect trajectory-level exported batches

Tasks:

1. add eval-batch export for step-level discriminator data
2. include provenance fields so held-out replay remains possible
3. add comparison tooling for:
   - old trajectory discriminator
   - new single-step discriminator

Recommended output fields:

- `y`
- batched set-observation tensors or a serialized equivalent
- `env_idx`
- `buffer_step_idx`
- `global_step`
- `boundary_reason` if available
- `shot_clock`

### Phase 5. Cleanup / Deprecation Decision

After the new path works:

1. compare learning curves
2. compare playbook behavior
3. compare robustness of holdout/replay eval

Then decide whether to:

- keep both modes for experimentation
- or deprecate the old trajectory discriminator

## Validation Plan

### A. Correctness Checks

1. discriminator batch contains only active offense steps
2. labels match the current play `z`
3. reward is added to the correct rollout step only once
4. holdout split is reproducible
5. no action features leak into the new path

### B. Invariance Checks

1. confirm the new encoder consumes token/set input rather than flat player-index vectors
2. run player-permutation probes if feasible
3. verify that arbitrary player-index dependence is reduced relative to the current path

### C. Training Checks

1. loss/top-1/AUC become nontrivial over training
2. intrinsic reward remains numerically stable
3. PPO learning does not destabilize due to noisier per-step rewards

### D. Behavioral Checks

1. Playbook behavior remains differentiated by play
2. selector still learns usable preferences
3. matched-start behavior tests remain at least as strong as before

## Risks

### 1. Weaker Signal

Removing actions and trajectories may reduce discriminator power substantially.

That is acceptable if:

- the signal becomes conceptually cleaner
- and still remains useful enough to shape play-conditioned behavior

### 2. Per-Step Reward Noise

Per-step DIAYN rewards may be noisier than segment-level rewards.

Possible mitigations:

- stronger running normalization
- lower `beta`
- larger discriminator batch sizes
- optional averaging over a small local window as a later experiment

### 3. Encoder Complexity

A set-based step discriminator is architecturally cleaner, but it is more work than reusing the current flat MLP/GRU path.

That tradeoff is worth it if invariance is a core design goal.

## Open Decisions

1. use `s_t` or `s_{t+1}` as the default state input?
   - recommendation: `s_{t+1}`

2. should `globals` be injected into each token or handled separately?
   - recommendation: start with simple token concatenation, similar to the policy extractor

3. should we keep any trajectory discriminator path at all?
   - recommendation: keep temporarily for A/B comparison only

4. should holdout splits be by step or by higher-level provenance group?
   - recommendation: start with step-level holdout, then consider stronger grouped holdouts if needed

## Recommended First Milestone

Implement the smallest end-to-end version:

1. collect active offense **next-state** set observations per step
2. train a **set-step** discriminator on those states only
   - using a similar set-attention encoder model to the current policy
3. compute per-step DIAYN reward
4. log holdout top-1/AUC
5. verify that training still shows a nontrivial learning curve

If that works, then decide whether the old trajectory discriminator should remain only as a comparison baseline.
