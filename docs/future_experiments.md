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

Critical current finding:

- for the `set_step` discriminator, `shot_clock` and `pressure_exposure`
  produced a strong shortcut
- masking those two globals dropped saved-batch AUC from `0.6916` to `0.4850`
- masking ball-identity features did **not** materially reduce AUC

Immediate implication:

- future discriminator-focused runs should disable those two globals in the
  discriminator input path by default
- discriminator AUC from older runs should be interpreted cautiously

So the next questions are empirical:

1. what signal is actually driving the discriminator?
2. do different plays create meaningful behavioral differences?
3. should the selector boundary logic become learnable?

## Priority Summary

| Priority | Experiment | Cost | Expected Insight |
| --- | --- | --- | --- |
| P0 | Real nominal-conditioning training run review | Small | Confirms baseline stability before new research changes |
| P1 | Frozen-start parallel counterfactual evaluation | Medium | Measures whether different plays actually diverge when rolled out from the exact same starting state |
| P1 | Discriminator signal attribution | Small | Tells us whether signal comes from actions, state, or time structure |
| P1 | Matched-start behavioral evaluation | Medium | Best evidence for whether plays are behaviorally real |
| P2 | State/position-only discriminator | Medium | Tests whether spatial consequences alone support play learning |
| P2 | DIAYN-first pretraining / reward curriculum | Medium | Tests whether plays separate better when discovery precedes strong task reward |
| P2 | Start-state curriculum / formation restriction | Medium | Tests whether plays become more coherent when early discovery sees fewer starting contexts |
| P2 | Mirror-symmetry exploitation / augmentation | Medium | Tests whether left-right symmetry can reduce exploration burden and improve play coherence |
| P3 | Selector abstain / null-play option | Medium | Tests whether the hierarchy benefits from an explicit “do not call a play” choice instead of forcing every possession into one of the learned plays |
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

### P1. Frozen-Start Parallel Counterfactual Evaluation

Goal:

- take a single captured starting state, replicate it across many envs, and roll out different plays and/or stochastic variants in parallel

Motivation:

- this is a stronger diagnostic than averaging over many unrelated starts
- it cleanly separates play-dependent divergence from variation in starting state
- it uses the parallel env stack to cheaply generate many counterfactuals from the exact same state

Primary use:

- evaluation / diagnostics first
- not a training default

Recommended setup:

1. capture a single valid offense possession start
2. clone that state across many envs
3. assign different plays and/or multiple stochastic rollouts per play
4. compare actions, trajectories, occupancy, and outcomes

Questions answered:

- from the same exact start, do different plays lead to meaningfully different behavior?
- is divergence mostly in immediate action choice, or in downstream state trajectories?
- how much variance comes from play identity versus rollout stochasticity?

Recommended metrics:

- first-action divergence
- pass count
- shot attempt rate
- shot type mix
- shot quality
- turnover reason
- possession length
- endpoint / occupancy summaries

Potential upside:

- one of the clearest diagnostics for whether plays are behaviorally real
- easy to compare across checkpoints
- naturally compatible with Playbook-style analysis

Main risks:

- overinterpretation from a tiny set of chosen starts
- frozen starts may not represent the broader training distribution
- rollout stochasticity can still obscure small play differences if sample counts are too low

Cost:

- medium

Expected insight:

- very high

Recommended first version:

1. build this as an evaluation tool only
2. run it on a small library of representative captured starts
3. compare checkpoints and play pairs on the same frozen-start panels

Success criterion:

- repeated frozen-start panels show that some play pairs reliably produce distinct trajectories or outcomes beyond immediate action variation

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

### P2. DIAYN-First Pretraining / Reward Curriculum

Goal:

- let the model learn distinguishable plays before strong point-scoring incentives pull all plays toward generic competent offense

Motivation:

- in vanilla DIAYN, skills are discovered before downstream task reward dominates
- in the current setup, task reward and play-discovery reward are active together
- that may encourage early convergence toward similar “good offense” behaviors before plays become meaningfully distinct

Variants:

1. pure DIAYN warmup
- suppress or zero task reward for an initial phase
- train only on intent/diversity reward
- later restore normal task reward

2. weighted curriculum
- begin with low task reward weight and high DIAYN influence
- gradually ramp task reward back to normal

3. staged fine-tuning
- phase A: train only low-level `pi(a|s,z)` play-conditioned behavior
- phase B: introduce or strengthen selector/task optimization after plays have separated

Questions answered:

- are plays failing to emerge because task reward dominates too early?
- does earlier play separation produce stronger later selector preferences?
- does a DIAYN-first phase produce more state-divergent counterfactuals from matched starts?

Potential upside:

- cleaner skill discovery
- stronger play-conditioned behavior before selector exploitation begins
- closer alignment with the original DIAYN training philosophy

Main risks:

- learned plays become diverse but strategically useless
- transfer into the later task-reward phase is weaker than expected
- staged schedules add more training complexity and bookkeeping

Cost:

- medium

Expected insight:

- high

Recommended first version:

1. start with a weighted curriculum, not pure reward removal
2. keep environment/task reward nonzero but strongly downweighted early
3. reduce defensive turnover / interception pressure during the early discovery phase
4. ramp toward the current reward mix and full defensive pressure later in training

Candidate early-phase environment relaxations:

- lower `defender_pressure_turnover_chance`
- lower `base_steal_rate`
- possibly lower `steal_distance_factor`

Rationale:

- early play discovery should favor controllable distinct behaviors
- strong turnover pressure may collapse exploration into conservative, similar policies before plays separate

Success criterion:

- compared with the baseline, the model shows:
  - earlier or stronger discriminator takeoff
  - more durable discriminator improvement
  - more distinct matched-start play rollouts

### P2. Start-State Curriculum / Formation Restriction

Goal:

- reduce early contextual entropy so play labels can become coherent before being asked to cover a broad distribution of possession starts

Implementation note:

- detailed v1 design is captured in
  [start_template_curriculum_plan.md](/home/evanzamir/basketworld/docs/start_template_curriculum_plan.md)

Motivation:

- starting state likely has a large influence on what a sensible play can be
- if the starting-state distribution is too broad, one play label may be forced to span many incompatible contexts
- that can weaken both discriminator signal and behavioral coherence

Key distinction:

- selector choice should still depend on state
- this experiment narrows the state distribution seen during early discovery; it does not remove state-dependent play choice

Variants:

1. limited spawn templates
- sample from a small set of offense/defense formation templates

2. restricted ball-holder starts
- limit which offensive player starts with the ball
- limit initial ball-holder court regions

3. phased start-state widening
- early phase: narrow formation family
- later phase: broaden toward the current full spawn/state distribution

Questions answered:

- do plays become more behaviorally coherent when initial contexts are less diverse?
- does discriminator signal strengthen when examples are drawn from more comparable starting states?
- do learned plays later generalize when the start-state distribution is widened again?

Potential upside:

- cleaner early play discovery
- easier matched-start behavioral interpretation
- stronger selector semantics once broader contexts are reintroduced

Main risks:

- plays overfit to a narrow family of starts
- later widening causes partial collapse or relabeling of play semantics
- curriculum design adds more environment complexity

Cost:

- medium

Expected insight:

- high

Recommended first version:

1. define a small library of half-court spawn templates
2. train early with only those templates
3. widen toward the current random spawn process later in training

Success criterion:

- compared with the baseline, the model shows:
  - stronger or earlier discriminator takeoff
  - more interpretable matched-start counterfactual differences
  - less apparent play collapse across widely different possession starts

### P2. Mirror-Symmetry Exploitation / Augmentation

Goal:

- reduce redundant exploration and let the model reuse play structure across left/right mirrored situations

Motivation:

- many half-court situations are approximately symmetric with respect to the basket axis
- without exploiting that symmetry, the model may need to relearn analogous left-side and right-side plays separately
- that can waste discriminator capacity and fragment play semantics

Variants:

1. training-time data augmentation
- mirror states during policy and/or discriminator training
- mirror actions consistently where needed

2. state canonicalization
- map states to a canonical left/right orientation before feeding policy/discriminator

3. evaluation-time symmetry checks
- compare matched-start counterfactuals under mirrored initial states
- test whether the same play behaves like a mirrored counterpart

Questions answered:

- does symmetry handling improve sample efficiency?
- do play labels become more abstract and less tied to court side?
- does the discriminator stop wasting capacity on orientation-specific distinctions?

Potential upside:

- lower effective exploration space
- cleaner play semantics
- faster discriminator takeoff
- better generalization across mirrored possessions

Main risks:

- action mirroring logic may be subtle and easy to get wrong
- some court situations may not be perfectly symmetric because of implementation details
- canonicalization may hide asymmetries that the policy should actually respect

Cost:

- medium

Expected insight:

- medium to high

Recommended first version:

1. start with evaluation-time symmetry checks
2. then add discriminator-side mirroring augmentation
3. only later consider full policy-side canonicalization

Success criterion:

- compared with baseline, the model shows:
  - faster early discriminator improvement
  - better consistency of plays across mirrored starts
  - fewer obviously duplicated “left version / right version” play behaviors

### P3. Selector Abstain / Null-Play Option

Goal:

- let the selector choose “no structured play” explicitly when that is better than forcing one of the learned intents

Concept:

- add a selector-level abstain / null action
- if chosen, set `intent_active = false` and run the low-level policy unconditioned
- do **not** treat this as a normal play intent class for the discriminator

Important design constraint:

- this should be a separate selector action, not just “one more intent `z`”
- otherwise the discriminator would be pushed to make “default / null behavior” artificially distinguishable, which conflicts with the intended semantics

Questions answered:

- are some states better handled by the default policy than by any learned play?
- does selector collapse reduce if it can abstain instead of overusing one mediocre play?
- do learned play families become cleaner when “no play” is separated from actual play selection?

Potential upside:

- cleaner selector semantics
- safer fallback in broken-floor / late-clock states
- less pressure to force every possession into the nearest available play family

Main risks:

- selector may collapse toward abstain because it is safe and generic
- play usage could shrink too much early in training
- requires careful scheduling or penalties so abstain does not dominate before play families are mature

Cost:

- medium

Expected insight:

- medium to high

Recommended first version:

1. keep the current play vocabulary unchanged
2. add one extra selector action for `null / abstain`
3. exclude abstain steps from discriminator training exactly as inactive-intent steps are excluded today
4. only enable or relax abstain after low-level play learning is already stable

Success criterion:

- abstain appears mainly in states where forcing a play is low-value
- selector no longer needs to overuse weak intents as a fallback
- play-conditioned behavior becomes cleaner rather than less used overall

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
5. test whether a DIAYN-first reward curriculum improves play separation
6. test whether a narrower early start-state distribution improves play coherence
7. consider whether the selector needs an explicit abstain / null-play option
8. only then consider selector-boundary learning

## Notes

- latent PCA/t-SNE should remain secondary diagnostics, not primary evidence
- Playbook remains the main qualitative surface
- future experiments should optimize for causal and behavioral clarity first
