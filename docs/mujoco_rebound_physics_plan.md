# MuJoCo Rebound Physics Sidequest Plan

## Goal

Build an offline physics simulator that generates missed-shot rebound landing data, then fit a lightweight Basketworld/JAX-compatible rebound model from that data.

MuJoCo is only for data generation and calibration. It should not become part of Basketworld training, inference, UI runtime, or JAX rollout execution.

## Current Status

As of May 30, 2026:

- Milestone 1 is complete. We have shown that the MuJoCo prototype can generate plausible trajectories, contact diagnostics, landing heatmaps, side/3D views, and animated GIFs.
- Milestone 2 is deferred. The current `target_noise` and `release_noise` shot controls are enough for the next phase. More detailed miss-type conditioning can be added later if fitting exposes a gap.
- Milestone 3 is implemented and being revised to export separate rebound tables by explicit shot type. We can now generate shot-location-conditioned rebound data for canonical court locations, retain raw made/missed diagnostics, filter made baskets out of the fitting dataset, and export transition tables plus per-shot heatmaps.
- Milestone 4 has a working table export: `run_rebound_fit.py` exports a filtered, pseudocount-smoothed table model from the completed sweep without rerunning MuJoCo. The default model uses catch-height targets and filters behind-backboard artifacts.
- Milestone 6 has started: `analytics/rebound_sim` can load the fitted catch table, sample target cells with canonical-shot reflection, and reuse the existing rebound-winner prototype logic.

## Rationale

Rebounding is likely to matter a lot for a future full-court version of Basketworld. The current lightweight rebound prototype is useful for tuning high-level behavior, but it still relies on hand-authored assumptions about where missed shots land.

A MuJoCo sidequest gives us a way to generate physically plausible priors for:

- Where missed shots tend to rebound from each shot location.
- How rebound distance changes with shot distance.
- How rim, backboard, and ball contact sequences affect landing location.
- How corner/wing/top-of-key shots differ in weak-side vs same-side rebound behavior.

The output should be a compact fitted model that can be used by the JAX environment through tables or simple vectorized formulas.

## Non-Goals

- Do not run MuJoCo inside BW training.
- Do not make MuJoCo a dependency of the app runtime.
- Do not simulate player bodies or player rebounding decisions in MuJoCo initially.
- Do not try to make the first version NBA-accurate. The first target is plausibility and parameter sensitivity.

## Milestone 1: Minimal Physics Prototype - Complete

Created `analytics/rebound_physics/`.

Implemented a single-shot MuJoCo prototype with:

- Floor plane.
- Backboard.
- Rim approximation.
- Basketball.
- Gravity and contact dynamics.
- Configurable shot origin.
- Randomized launch parameters.
- Target-noise and release-noise shot generation modes.
- Contact solver calibration controls.

Records:

- Whether the shot was made or missed.
- First contact type, if any.
- Contact sequence: rim, backboard, floor.
- Rim/backboard/floor contact points.
- First floor landing location.
- Settled location, if useful.
- Decimated trajectory points.

Outputs:

- JSONL trajectory summaries.
- Aggregate JSON summaries.
- Landing heatmap.
- Rim/backboard contact diagnostics.
- Rim outcome plot.
- Side-view and shooter-view trajectory plots.
- 3D scene diagnostic plot.
- Single or multi-trajectory animated GIFs.

Status:

- Complete enough for dataset generation.
- Physics parameters can still be tuned later, but more physics iteration is not blocking the next phase.

## Milestone 2: Shot Parameter Sampling - Deferred

Original goal: add a more elaborate shot trajectory sampler with tunable miss categories such as short, long, left, right, front rim, back rim, and backboard.

Decision:

- Defer this milestone for now.
- The current prototype already supports two useful modes: `target_noise` and `release_noise`.
- We can generate a meaningful rebound dataset now and only add explicit miss-type conditioning if the fitted model shows systematic weaknesses.

Deferred examples:

- Explicit miss-type conditioning.
- Better empirical shot error distributions.
- Player-specific release profiles.
- More detailed spin/friction calibration.

## Milestone 3: Hex Court Data Generation - Implemented

Map Basketworld hex cells to MuJoCo court coordinates, canonicalize symmetric shot cells, then generate conditioned rebound samples for canonical shot locations only.

Core tasks:

1. Reuse the Basketworld 9-row, 8-column half-court geometry as the initial data-generation target.
2. Define a deterministic `shot_cell -> MuJoCo x,y,z` mapping.
3. Define `landing_x,y -> nearest legal rebound hex` mapping.
4. Enumerate legal shot cells. Initially include all reasonable offensive shot cells, not just three-point locations.
5. Use court symmetry to reduce simulation cost: simulate only canonical shot cells, such as one lateral half plus centerline cells, rather than duplicating both sides.
6. Run a fixed number of shot simulations per canonical shot cell using the calibrated/current release-noise settings.
7. Save per-sample records plus per-canonical-shot-cell aggregate summaries.
8. Assign each canonical shot cell to an explicit shot type: `dunk` for the basket hex, `layup` for one-hex basket-adjacent cells by default, and `jumper` for all other cells.
9. Generate per-canonical-shot-cell rebound heatmaps for visual inspection.
10. Export separate transition arrays for each shot type so Basketworld can choose `P(landing_hex | shot_type, canonical_shot_hex)`.

Recommended raw sample schema:

```text
shot_q, shot_r, shot_cell_index,
canonical_shot_q, canonical_shot_r, symmetry_class, reflection_sign,
shot_x, shot_y, shot_z,
shot_model, shot_type, shot_distance_hex,
made, rim_outcome, first_contact, contact_count,
rim_crossing_x, rim_crossing_y, rim_crossing_distance,
landing_x, landing_y,
landing_q, landing_r, landing_cell_index,
settled_x, settled_y,
seed
```

Important filtering rule:

- Keep made baskets in the raw diagnostic dataset so make-rate and shot-quality diagnostics remain available.
- Exclude made baskets from the fitted rebound landing dataset, because only missed shots produce rebound opportunities.

Recommended aggregate artifacts:

```text
rebound_dataset_raw_samples.jsonl
rebound_dataset_missed_samples.jsonl
rebound_dataset_summary.json
rebound_transition_counts.npy
rebound_transition_probs.npy
rebound_transition_logits.npy
shot_type_models/<shot_type>/rebound_transition_counts.npy
shot_type_models/<shot_type>/rebound_transition_probs.npy
shot_type_models/<shot_type>/rebound_transition_logits.npy
shot_cell_<q>_<r>_heatmap.png
```

Success criteria:

- Every selected shot hex can produce a rebound target heatmap.
- Corner, wing, and top-of-key shots have visibly distinct rebound profiles.
- Same-side vs weak-side rebound tendencies are measurable.
- Dataset generation is deterministic for a fixed seed/config.
- Canonical shot cells cover the court under the chosen symmetry transform, and inference can reflect predictions back to the original side.
- Made shots are retained for diagnostics but excluded from rebound fitting artifacts.

Implemented artifacts:

- `analytics/rebound_physics/dataset.py`
- `analytics/rebound_physics/run_rebound_dataset.py`
- `tests/test_rebound_physics_dataset.py`
- VS Code launch config: `MuJoCo Rebound Dataset Sweep`

Implementation notes:

- The first dataset format retains the pooled table model target `P(landing_hex | canonical_shot_hex)` for diagnostics, but the intended BW model is now split by explicit shot type: `P(landing_hex | shot_type, canonical_shot_hex)`.
- Canonicalization reflects left/right shot cells into one simulated side to avoid duplicating symmetric shots.
- The raw JSONL keeps makes and misses for diagnostics.
- The missed JSONL and transition arrays exclude made baskets and are intended for fitting.
- `shot_type_models/{dunk,layup,jumper}/rebound_transition_probs.npy` and logits are emitted, so Milestone 4 can start from these arrays.
- Current layup sampling defaults are calibrated as a first-pass approximation to ~60% make rate for one-hex layups, so the layup rebound table contains actual miss samples rather than a uniform empty-row fallback.

## Milestone 4: Fit Lightweight Model - Started

Fit a compact model to the MuJoCo-generated missed-shot samples. The first implementation is a filtered table export that can use either catch-height targets or first-floor-contact landing targets.

Recommended first model:

- Table model: `P(target_hex | shot_type, shot_hex)`, defaulting to catch-height targets.
- Runtime representation: `target_logits[canonical_shot_cell_index, canonical_target_cell_index]`.
- JAX sampling path: canonicalize the shot cell, sample a canonical target, then reflect the sampled target back to the original side.

This is the best first implementation because it is faithful, simple, vectorized, and easy to inspect. More portable parametric models can come later.

Candidate later model families:

- Smoothed table model with Dirichlet prior / pseudocounts.
- Factorized model: `P(region, distance, side | shot_hex)`.
- Parametric model: landing mean/covariance as functions of shot geometry.
- Small learned model that maps shot features to rebound target logits.

Fitting tasks:

1. Load generated MuJoCo samples.
2. Filter to missed shots only for rebound distributions. This is mandatory: made shots should not contribute rebound targets. Behind-backboard artifacts are filtered for the first BW-ready table export.
3. Count `canonical_shot_cell -> canonical_landing_cell` transitions.
4. Store the symmetry metadata needed to canonicalize shot cells and de-canonicalize sampled landing cells at inference time.
5. Add pseudocount smoothing so every canonical shot cell has safe logits.
6. Normalize to probabilities and logits.
7. Save NumPy artifacts plus a JSON metadata file with geometry/config/symmetry provenance.
8. Produce fit diagnostics.

Evaluation:

- Compare fitted distributions to MuJoCo samples with KL divergence or cross entropy.
- Compare per-canonical-shot heatmaps visually.
- Compare aggregate rebound distance distributions.
- Compare weak-side vs same-side rebound rates.
- Confirm table sampling reproduces MuJoCo aggregate behavior with a standalone simulation.

Success criteria:

- The lightweight fitted model reproduces MuJoCo landing distributions closely enough for BW use.
- The model can be represented as JAX arrays or simple formulas.
- Inference is cheap enough for batched BW rollouts.

## Milestone 5: Render Shareable Shot Animations

Use the MuJoCo simulation traces to render shareable shot/rebound animations.

Status:

- Mostly complete for the prototype.
- We can already generate diagnostic animated GIFs with one or more trajectories.
- Future work is polish: camera choices, overlays, and blog-quality exports.

Target outputs:

- GIF for quick sharing.
- MP4 or WebM for blog-quality embeds.
- Optional PNG frame sequences for debugging.

Recommended views:

- Cinematic side/angle view showing ball, rim, backboard, and floor.
- Diagnostic top-down view showing shot origin, ball trajectory, rim/backboard contacts, and rebound landing location.
- Optional hex-court overlay showing the nearest Basketworld shot and landing cells.

Success criteria:

- We can generate a readable animation for a specific shot origin and seed.
- The animation shows the miss, rim/backboard interaction, and rebound landing.
- The output is suitable for blog posts or debugging notes.

## Milestone 6: Integrate With Current Rebound Prototype

Add a fitted-model mode to `analytics/rebound_sim`.

The existing rebound simulator should support:

- Hand-tuned parametric landing model.
- MuJoCo-fitted landing model.
- Same player rebound-winner logic on top of either landing source.

This lets us compare the current hand-authored rebound assumptions against the physics-calibrated version before touching the JAX environment.

Success criteria:

- The current rebound sim can load fitted landing distributions. Complete for the fitted catch table via `--fitted-target-model-dir`.
- Target sampling uses the fitted shot-type table: `dunk`, `finger_roll`, `layup`, or `jumper`. Complete.
- Symmetry is handled at runtime by canonicalizing the shot and reflecting sampled target cells back to the original side. Complete.
- Existing rebound-winner logic can run on top of fitted target samples. Complete.
- Remaining before JAX: validate fitted-vs-heuristic aggregate behavior, decide whether winner logic stays heuristic or gets table-derived inputs, then port the table lookup/sampling path to `basketworld_jax`.
- Existing heatmap visualizations work for fitted distributions.
- Player rebound winner logic remains independent from landing physics.

## Milestone 7: Port To JAX

Convert the final fitted model into JAX-friendly data structures.

Likely runtime representation:

```text
target_logits[canonical_shot_cell_index, canonical_target_cell_index]
```

JAX sampling path:

```text
canonical_shot_index, reflection = canonicalize_shot_cell(shot_index)
canonical_target_index = jax.random.categorical(key, target_logits[canonical_shot_index])
target_index = reflect_target_cell(canonical_target_index, reflection)
```

Requirements:

- No MuJoCo dependency.
- No Python loops in rollout-time sampling.
- Works with batched environments.
- Compatible with future 5-on-5/full-court configurations.

Success criteria:

- JAX environment can sample rebound landing locations from the fitted model.
- Runtime remains vectorized and cheap.
- Rebound behavior matches the offline prototype distribution.

## Risks

- Contact parameters may be physically plausible but not basketball-realistic without calibration.
- Rim approximation matters. A simple rigid cylinder may produce unrealistic bounce behavior.
- Shot launch distribution may dominate outcomes more than rim/backboard physics.
- Backspin and ball/rim friction can substantially change rebound patterns.
- The table model is geometry-specific unless we add a geometry-aware fitting layer.
- MuJoCo-generated priors should still be sanity-checked against basketball intuition and available real-world rebounding studies.

## Recommended Next Slice

Build the data-generation and fitting pipeline:

1. Add a script that sweeps canonical legal Basketworld shot hexes and writes MuJoCo rebound samples.
2. Add deterministic hex-to-meter and meter-to-hex mapping helpers, including canonicalize/reflect transforms.
3. Emit both raw samples and missed-shot-only fitting samples.
4. Add per-canonical-shot-cell heatmap outputs.
5. Add a fitter that produces canonical `target_probs`, canonical `target_logits`, and symmetry metadata from missed shots only.
6. Add a standalone validation script that samples from the fitted canonical table, reflects predictions back to full-court cells, and compares it to the MuJoCo source data.

Do not integrate with the JAX environment until the fitted table reproduces the offline MuJoCo distributions.
