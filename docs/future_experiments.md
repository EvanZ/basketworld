# Future Experiments

Focused follow-up tracker for the nominal play-conditioning architecture.

This file is intentionally narrower than
[nominal_play_conditioning_plan.md](/home/evanzamir/basketworld/docs/nominal_play_conditioning_plan.md).
Use it to prioritize experiments after the core refactor, not to restate the whole design.

## Status

Current architecture baseline:

- policy-side nominal play conditioning is implemented
- low-level policy no longer consumes legacy numeric intent observation fields
- nominal play names are live in backend/UI

So the next questions are empirical:

1. what signal is actually driving the discriminator?
2. do different plays create meaningful behavioral differences?
3. should the selector boundary logic become learnable?

## Priority Summary

| Priority | Experiment | Cost | Expected Insight |
| --- | --- | --- | --- |
| P0 | Real nominal-conditioning training run review | Small | Confirms baseline stability before new research changes |
| P1 | Discriminator signal attribution | Small | Tells us whether signal comes from actions, state, or time structure |
| P1 | Matched-start behavioral evaluation | Medium | Best evidence for whether plays are behaviorally real |
| P2 | State/position-only discriminator | Medium | Tests whether spatial consequences alone support play learning |
| P3 | Embedding structure ablations | Medium | Clarifies whether current `e_z` parameterization is over-flexible |
| P4 | Learned play-boundary / reselection head | Large | Expands hierarchy expressiveness, but with larger tuning risk |

## Experiments

### P0. Real Run Review

Goal:

- verify the new architecture behaves sanely in a full training run before adding more moving parts

What to inspect:

- selector probabilities over time
- intent policy sensitivity
- discriminator metrics
- playbook qualitative behavior
- any reset / alternation / self-play instability

Cost:

- small

Expected insight:

- distinguishes architecture issues from research issues

Recommended decision rule:

- do not start new architecture experiments until at least one real run looks operationally stable

### P1. Discriminator Signal Attribution

Goal:

- identify what the discriminator is actually using

Recommended ablations:

1. `obs + actions`
2. `obs only`
3. `actions only`
4. time-scrambled sequence
5. mean-pooled sequence

Questions answered:

- is the discriminator mostly reading action signatures?
- is there usable state-only signal?
- is sequence order important?
- are we still vulnerable to nuisance structure?

Cost:

- small

Expected insight:

- high

Suggested output:

- table of top-1 / AUC under each ablation
- one short interpretation paragraph per condition

### P1. Matched-Start Behavioral Evaluation

Goal:

- test whether forcing different plays from the same initial state causes different downstream behavior

Recommended metrics:

- pass count
- shot attempt rate
- shot type mix
- shot quality
- turnover reason
- possession length
- endpoint / occupancy summaries

Why this matters:

- this is stronger evidence than latent-space plots
- it tests the thing we actually care about: behavioral distinctness

Cost:

- medium

Expected insight:

- very high

Success criterion:

- repeated matched starts show stable behavior differences across at least some play pairs

### P2. State/Position-Only Discriminator

Goal:

- make the discriminator depend on spatial/state consequences rather than action signatures

Variants:

1. full observation state only
2. position-focused state only
3. position + minimal game context

Potential upside:

- cleaner notion of “play” as a state-distribution difference

Main risk:

- the signal may become too weak and collapse

Cost:

- medium

Expected insight:

- high if paired with the signal-attribution ablations above

Recommended order:

- do not change training objective until the attribution ablations suggest action features are dominating

### P3. Embedding Structure Ablations

Goal:

- test whether the current play embedding parameterization is more flexible than necessary

Variants:

1. separate offense / defense tables
2. shared table with role-conditioned projection
3. smaller `intent_embedding_dim`
4. embedding norm penalty
5. embedding diversity regularization

Questions answered:

- do offense and defense need separate play embeddings?
- is current embedding capacity larger than necessary?
- can we stabilize play semantics with stronger constraints?

Cost:

- medium

Expected insight:

- medium

### P4. Learned Play-Boundary / Reselection Head

Goal:

- learn when to keep or refresh the current play instead of relying only on deterministic boundaries

Concept:

- separate “which play” from “when to resample”

Potential upside:

- more expressive hierarchical control
- less reliance on hand-coded pass / timeout boundary rules

Main risks:

- trivial collapse to constant resampling
- trivial collapse to never resampling
- harder credit assignment
- significantly larger debugging surface

Cost:

- large

Expected insight:

- potentially high, but only after the simpler diagnostics above are done

Recommendation:

- do this only after the discriminator and behavioral evidence are better understood

## Suggested Order

1. complete and review a real nominal-conditioning run
2. run discriminator signal-attribution ablations
3. run matched-start behavioral evaluation
4. decide whether the discriminator should move toward state-only inputs
5. only then consider selector-boundary learning

## Notes

- latent PCA/t-SNE should remain secondary diagnostics, not primary evidence
- Playbook remains the main qualitative surface
- future experiments should optimize for causal and behavioral clarity first
