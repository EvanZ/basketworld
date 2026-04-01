# Nominal Play Conditioning Plan

## Status

Phase 1 and Phase 2 are implemented:

- low-level policy conditioning now uses policy-side runtime play ids plus embedding lookup
- low-level set observations no longer expose legacy `intent_*` conditioning keys
- selector/runtime paths have been refactored onto explicit policy-side conditioning
- nominal play-name mapping is live in backend/UI

The remaining work is follow-up evaluation, refinement, and future experiments.

## Compatibility Stance

Backward compatibility with the old numeric-intent observation design is **not required**.

That older design should be treated as deprecated and replaced rather than preserved.

Implications:

- no long-term dual-mode conditioning path
- no need to preserve low-level policy support for scalar numeric play identity
- implementation should optimize for conceptual clarity, not compatibility shims

## Locked Initial Decisions

The following decisions are now part of the initial implementation plan:

1. remove **all** low-level intent globals from the low-level policy input path
   - `intent_index` / `intent_index_norm`
   - `intent_active`
   - `intent_visible`
   - `intent_age_norm`
2. keep intent/play runtime state in the environment and selector machinery
3. use true policy-side conditioning rather than a dedicated observation key
4. for the first implementation, carry current play ids as per-env runtime state on the policy/algorithm side rather than threading an explicit tensor through every SB3 call
5. defer nominal UI play naming to phase 2, after runtime/training conditioning is stable

These choices are intended to reduce architectural ambiguity before implementation starts.

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
- environment/runtime still tracks play state internally
- low-level policy receives current `z` separately from the environment observation
- set-attention extractor looks up learned embedding `e_z`
- extractor injects `e_z` into token features
- no low-level play-conditioning fields are exposed through observation space

This is intended to **replace** the old numeric-intent observation path, not coexist with it indefinitely.

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

Implementation decision:

- remove all low-level play-conditioning fields from the low-level observation path
- keep `intent_active`, `intent_visible`, `intent_age`, and related commitment state only as environment/runtime state, logging state, and UI/diagnostic state
- do not retain a permanent legacy numeric-intent observation mode

### 2. Embedding Lookup Only

Use the existing set-attention embedding mechanism as the basis for the redesign:

- keep offense/defense play embedding tables
- keep projection into token dimension
- keep token-space injection

But stop reconstructing the embedding id from `intent_index_norm`.

Instead:

- pass the current `z` directly into the policy/extractor
- perform lookup from that direct id

For the first implementation, the transport mechanism will be:

- maintain per-env current play ids as runtime state on the policy/algorithm side
- update that state when selector decisions or forced-play assignments occur
- have the extractor read the current play ids from that runtime state during forward passes

This is less invasive than threading an explicit play-id tensor through every SB3 entry point, while still achieving true policy-side conditioning.

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
2. remove all low-level intent globals from the low-level policy observation path
3. add dedicated policy-side play-id conditioning state
4. switch set-attention extractor to direct play-id embedding lookup
5. store current play ids per env on the policy/algorithm runtime side
6. remove legacy numeric play-conditioning code instead of preserving it as a supported path

### Phase 2. Selector Routing Refactor

1. keep selector start-of-segment logic
2. replace observation-patching communication path with direct policy-side conditioning state
3. ensure VecEnv / rollout code carries current `z` cleanly per env
4. ensure forced-play / evaluation helpers write the same runtime conditioning state

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

This is acceptable.

Expected outcome:

- new nominal-conditioning checkpoints will not be directly comparable to old checkpoints
- old numeric-conditioned checkpoints can be treated as legacy artifacts rather than a supported target

### 3. Hidden Couplings

Some backend/evaluation paths currently rely on `patch_intent_in_observation(...)`.

These will need review.

## Open Questions

1. Should the environment still emit play identity in info dicts for logging?
2. Should `intent_visible` survive anywhere outside logging/UI?
3. Should defense and offense keep separate embedding tables?
4. How should per-run play-name mapping be versioned across continuations?
5. Should play names be sampled once per run or once per alternation family?

## Future Experiments

### 1. Discriminator on State/Position Only

Test a discriminator variant that uses state or position features only, without action features.

Motivation:

- reduce reliance on superficial action-signature cues
- push the DIAYN signal toward distinct spatial/state consequences
- check whether the current discriminator signal is mostly action-driven

Recommended ablations:

1. `obs + actions`
2. `obs only`
3. `actions only`

### 2. Learned Play-Boundary / Reselection Head

Add a separate head that learns whether to keep the current play or sample a new one, instead of relying entirely on deterministic play boundaries.

Motivation:

- let the hierarchy learn play duration explicitly
- separate "which play" from "when to resample"
- reduce dependence on hard-coded commitment / pass boundary rules

Main risk:

- easy to collapse into resampling too often or never resampling without additional regularization

### 3. Direct Signal Attribution for the Discriminator

Run controlled replay/eval passes that zero or drop subsets of discriminator inputs.

Examples:

- remove action features
- remove non-positional observation features
- scramble temporal order
- mean-pool sequence inputs before evaluation

Goal:

- identify what the discriminator is actually using to separate plays

### 4. Behavioral Evaluation From Matched Starts

Prioritize causal evaluation over latent geometry.

Examples:

- same initial state, force different plays, compare trajectories
- per-play distributions of pass count, shot type, shot quality, turnover reason, possession length
- occupancy / endpoint heatmaps by play

Goal:

- measure whether plays produce stable behavioral differences that matter

### 5. Embedding Structure and Sharing Ablations

Evaluate whether offense and defense should continue to use separate embedding tables and whether embedding dimension should be reduced or regularized.

Examples:

- shared embedding table with role-conditioned projection
- smaller `intent_embedding_dim`
- embedding norm penalty or diversity regularization

Goal:

- test whether the current embedding parameterization is larger or less constrained than necessary

## Recommended Default Decisions

Unless a stronger reason appears, use:

1. embedding-lookup-only conditioning for low-level policy
2. no low-level intent globals in the low-level observation path
3. policy-side runtime-state transport for current play ids in the first implementation
4. stable per-run sampled play-name mapping
5. internal `z` retained for debugging, external UI shows nominal labels
6. no long-term backward-compatibility layer for numeric-intent observation conditioning

## Immediate Next Step

Immediate next steps should be evaluation-oriented:

1. complete at least one real training run under the nominal-conditioning architecture
2. monitor selector behavior, play-conditioned sensitivity, and backend/UI stability
3. use the future experiments above to decide whether to change the discriminator input or selector boundary design next
