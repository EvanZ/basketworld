# Nominal Play Conditioning Plan

## Status

Proposal only. No training-path implementation has been done yet.

## Motivation

The current intent/play conditioning path mixes two different ideas:

1. `z` as a nominal play identity
2. `z` as an ordered numeric feature in observation space

That creates multiple problems:

- policy-side geometry is contaminated by direct play conditioning
- `intent_index_norm` leaks ordinal structure that should not exist
- representation plots can reflect label injection rather than learned play structure
- UI currently encourages thinking about plays as numbered bins instead of named categories

The goal of this redesign is to preserve play-conditioned behavior while removing ordinal leakage and making the system conceptually cleaner.

## Core Design

### Current Design

Low-level policy conditioning currently uses both:

1. explicit intent globals in observation space
   - `intent_index_norm`
   - `intent_active`
   - `intent_visible`
   - `intent_age_norm`

2. learned set-attention intent embedding injection
   - offense/defense embedding lookup
   - projected into token space
   - added to player tokens

This means the policy sees play identity both as:

- an ordered scalar signal
- a categorical learned embedding

### Proposed Design

Intent/play identity should be treated as **nominal only**.

That means:

1. remove scalar numeric play identity from the low-level policy observation path
2. keep low-level policy conditioning on play identity
3. condition only through a policy-side embedding lookup

In practical terms:

- selector still chooses an internal play id `z in {0, ..., N-1}`
- low-level policy receives `z` separately from the environment observation
- set-attention extractor looks up learned embedding `e_z`
- extractor injects `e_z` into token features
- no `intent_index_norm` is exposed to the low-level policy as an ordered scalar

## Desired Semantics

Play ids should behave like:

- `"Kansas"`
- `"Horns Twist"`
- `"Ghost Loop"`

not like:

- `0.00`
- `0.14`
- `0.29`

Permutation of labels should not change the task definition.

## Scope

### In Scope

- low-level policy conditioning path
- selector-to-policy communication path
- UI/display naming for plays
- metadata persistence for per-run play names
- analysis/evaluation updates needed to handle nominal play labels

### Out of Scope

- changing the fundamental selector objective
- changing the discriminator target semantics
- rewriting Playbook UX from scratch

## Architecture Changes

### 1. Separate Play Identity From Observation Space

Current behavior:

- env/wrapper writes play identity into observation globals
- low-level policy reads play identity from observation

Proposed behavior:

- env may still maintain current play state internally
- selector chooses `z`
- policy forward pass receives `z` through a dedicated conditioning channel
- low-level observation no longer includes ordered scalar play identity

Open design choice:

- whether to also remove `intent_active`, `intent_visible`, and `intent_age_norm` from low-level observation

Current recommendation:

- remove all play-identity conditioning fields from the low-level observation path
- if any remain, they should be justified individually rather than inherited from the current design

### 2. Embedding Lookup Only

Use the existing set-attention embedding mechanism as the basis for the redesign:

- keep offense/defense play embedding tables
- keep projection into token dimension
- keep token-space injection

But stop reconstructing the embedding id from `intent_index_norm`.

Instead:

- pass the current `z` directly into the policy/extractor
- perform lookup from that direct id

### 3. Selector Remains Conceptually Separate

Selector should continue to:

- observe intent-neutralized state
- choose play id `z`
- learn from selector return signal

The selector should not need major conceptual changes.

The main change is:

- selector output should be routed into policy conditioning directly
- not by patching ordered play scalars into observation space

## UI / Naming Plan

### Goal

Expose plays as nominal labels rather than numbered indices.

### Proposal

Maintain a larger pool of candidate play names, for example `100`.

For each model/run:

1. sample `N` unique names where `N = num_intents`
2. assign each sampled name to one internal play id `z`
3. persist that mapping as run metadata/artifact
4. use the same mapping everywhere in UI and analytics

### Requirements

- mapping must be stable per run/checkpoint family
- display name must never be re-randomized mid-run
- internal canonical id `z` must still be preserved for debugging and storage

### Naming Style

Desired properties:

- short
- memorable
- easy to distinguish
- mildly whimsical is acceptable

Examples:

- `Ghost Elbow`
- `Ivory Loop`
- `Comet Horn`
- `Raven Twist`
- `Glass Pin`

### Storage

Need a single canonical per-run mapping source, likely one of:

- MLflow param/artifact
- JSON metadata artifact under `analysis/` or `models/`
- checkpoint-side metadata

Current recommendation:

- persist as a small JSON artifact and also expose it in any UI state endpoint that needs it

## Migration Plan

### Phase 1. Policy Conditioning Refactor

1. identify all places where low-level policy reads play identity from observation
2. add dedicated policy-side play-id input path
3. switch set-attention extractor to direct play-id embedding lookup
4. remove ordered scalar play-id usage from low-level extractor

### Phase 2. Selector Routing Refactor

1. keep selector start-of-segment logic
2. replace observation-patching communication path with direct policy-side conditioning state
3. ensure VecEnv / rollout code carries current `z` cleanly per env

### Phase 3. UI Naming Layer

1. define play-name pool
2. add per-run name mapping generation
3. persist mapping in artifacts
4. display names in Playbook, selector diagnostics, admin endpoints, and any analytics surfaces

### Phase 4. Analysis Cleanup

1. update analysis tools to treat play labels as nominal
2. stop relying on policy-side latent geometry as primary evidence
3. prefer causal/behavioral evaluation

## Evaluation Implications

This redesign should improve interpretability, but it will not by itself prove “play learning.”

After implementation, the main evaluation questions should be:

1. does forcing different plays causally change behavior?
2. do selector preferences correspond to stable behavior differences?
3. do play-conditioned rollouts produce meaningfully distinct Playbook trajectories?

Recommended post-change checks:

- forced-play rollout comparisons from matched starts
- per-play behavioral statistics
- Playbook trajectory overlays
- selector preference diagnostics using named plays

## Risks

### 1. Plumbing Complexity

Removing observation-based play conditioning means the rollout stack must carry current `z` explicitly.

### 2. Backward Compatibility

Old checkpoints assume current observation-based play conditioning.

Possible outcome:

- new nominal-conditioning checkpoints are not directly comparable to old checkpoints

### 3. Hidden Couplings

Some backend/evaluation paths currently rely on `patch_intent_in_observation(...)`.

These will need review.

## Open Questions

1. Should the environment still emit play identity in info dicts for logging?
2. Should `intent_visible` survive anywhere outside logging/UI?
3. Should defense and offense keep separate embedding tables?
4. How should per-run play-name mapping be versioned across continuations?
5. Should play names be sampled once per run or once per alternation family?

## Recommended Default Decisions

Unless a stronger reason appears, use:

1. embedding-lookup-only conditioning for low-level policy
2. no ordered scalar play id in low-level observation
3. stable per-run sampled play-name mapping
4. internal `z` retained for debugging, external UI shows nominal labels

## Immediate Next Step

Do not implement yet.

When ready, start with:

1. a narrow design pass on how current `z` is carried from selector to policy
2. identify every place that currently depends on observation-patched intent fields
3. implement policy-side conditioning first, before UI naming
