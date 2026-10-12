# Fixed-player-pool evaluation study

This opt-in study asks a different question from normal evaluation: which
stable simulated players does a checkpoint make more valuable after lineup and
starting-side effects are controlled? It never changes standard evaluation,
historical-opponent evaluation, or Elo artifacts.

## Supported checkpoints

Use a current JAX **5v5 multi-possession** checkpoint whose observation schema
matches the current runtime schema. Older single-possession checkpoints are
intentionally unsupported. The runner validates this before any game is run.

Pool identity is an evaluation artifact, not a policy observation. The study
pins named player skill profiles into the existing ten stable JAX slots; no
player ID is supplied to either policy.

New study pools use `independent_skill_coverage_v1`: percentage skills have
0.15 standard-deviation variation and unbounded standardized skills have 1.0.
Normal environment defaults remain unchanged; this variation is specific to
the study because fitted skill coefficients need nonzero support.

## Create and run

Create the player pool and balanced independent-game schedule once, then keep
their JSON files immutable. The default size and schedule length are
skill-scaled (with the current registry: 64 players and 640 games). Every game
has a distinct balanced lineup, an independently seeded RNG stream, and a
balanced starting side. The smaller values below are a smoke test and will be
flagged as under-sized in the report.

```bash
.env/bin/python -m basketworld_jax.train.main \
  --run-fixed-pool-study \
  --fixed-pool-checkpoint /path/to/checkpoints/update_0005000 \
  --fixed-pool-path artifacts/fixed_pool_inputs/pool.json \
  --fixed-pool-create --fixed-pool-size 10 --fixed-pool-seed 73 \
  --fixed-pool-schedule-path artifacts/fixed_pool_inputs/schedule.json \
  --fixed-pool-schedule-create --fixed-pool-games 8 --fixed-pool-schedule-seed 912 \
  --fixed-pool-batch-size 2 \
  --fixed-pool-action-mode direct \
  --fixed-pool-artifact-dir artifacts/fixed_pool
```

For a full initial study, omit `--fixed-pool-size` and
`--fixed-pool-games`; the active skill registry determines both. Direct and
sampled action modes are separate studies and must not be pooled.

Fixed-pool studies default to 64 games per compiled batch. The runner prints a
heartbeat before and after every batch and a liveness event every ten seconds
while a compiled batch is still executing. It reports elapsed/per-phase time
without pretending to know intra-batch completion. The immutable checkpoint
report records the execution configuration; wall-clock observations remain
runtime-only so replaying identical evidence does not change the report. Use a
small batch (for example, `--fixed-pool-batch-size 2`) for a smoke run to
measure compilation and CPU throughput before launching a full study.

## Reuse and compare checkpoints

Load the same immutable inputs for each compatible checkpoint. Specify paths
directly or choose numbered updates from a checkpoint directory:

```bash
.env/bin/python -m basketworld_jax.train.main \
  --run-fixed-pool-study \
  --fixed-pool-checkpoint-dir /path/to/checkpoints \
  --fixed-pool-checkpoint-update 1000 \
  --fixed-pool-checkpoint-update 5000 \
  --fixed-pool-path artifacts/fixed_pool_inputs/pool.json \
  --fixed-pool-schedule-path artifacts/fixed_pool_inputs/schedule.json \
  --fixed-pool-action-mode direct \
  --fixed-pool-artifact-dir artifacts/fixed_pool
```

Each checkpoint produces a unique raw-study directory under
`fixed_pool_studies/` and a checkpoint report under `fixed_pool_reports/`.
The longitudinal index records action mode, skill-schema hash, observation
schema, raw-study dependency, and whether cross-checkpoint comparison is
compatible. Add `--log-mlflow` to mirror these same isolated namespaces to the
active MLflow run.

## Analyze the results

Each raw study contains immutable `pool_manifest.json` and
`schedule_manifest.json`, plus typed `episodes.csv` and
`possession_events.csv` tables. Episode rows retain incomplete and
zero-possession games explicitly; the documented default model rule is to fit
only completed games with both teams recording offensive possessions.

The checkpoint report contains:

- ridge adjusted plus-minus in net-rating points versus a replacement player,
  paired-group bootstrap intervals, starting-side/overtime controls, design
  rank, and collinearity diagnostics;
- skill coefficients in net-rating points per study-pool standard deviation,
  predicted player value, residual value, and excluded zero-variance skills;
- possession-ledger factors for shot expected value/conversion, ball security,
  second-chance creation, and possession organization;
- conventional three-factor estimates for effective field-goal percentage,
  turnover percentage, and offensive-rebound percentage, with player-adjusted
  impact estimates, a joint held-out outcome model, and a separate-own/opponent
  ordinary-least-squares scenario analysis. The latter moves each term one
  training-set standard deviation in its favorable direction from a
  mean-reference team, then normalizes the resulting point-margin deltas to
  weights that sum to 100%. Free-throw rate is explicitly reported as
  unavailable until fouls and free throws exist in the environment;
- a clearly labeled specialist-versus-balanced summary based on registered
  skills, not a fixed skill list.

The signed player design has one intentional null direction: adding the same
constant to every player does not change a Team-A-minus-Team-B outcome. The
report labels this `structural_nullity: 1` and evaluates rank and condition
number on the remaining identified subspace; it does not confuse that required
replacement-level gauge with a schedule-collinearity warning.

If the skill registry gains a new skill, extend the prior pool into a new
immutable child pool and create a schedule bound to that child. Do not mutate
historical pools or schedules; their manifests remain readable for previous
studies, while the longitudinal compatibility report prevents unreported
cross-schema comparisons.
