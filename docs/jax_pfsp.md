# Prioritized self-play and defensive pressure metrics

Issue #80 adds opt-in PFSP to the JAX multi-possession trainer. Independent
and autoregressive policies use the same sampler. Existing recency behavior
and the opponent determinism schedule remain unchanged unless explicitly edited.

## Configuration

The trainer app exposes these fields under Opponent sampling and pins them on
the configuration page. They are included in saved configs, copies, launch
commands, and resume checks.

- `opponent_pool_sampling` / `--opponent-pool-sampling`: `recency` (default) or `pfsp`.
- `opponent_pfsp_alpha` / `--opponent-pfsp-alpha`: default `0.5`; 0 favors even matches, 1 favors hard opponents.
- `opponent_pfsp_epsilon` / `--opponent-pfsp-epsilon`: default `0.1`; uniform exploration across all eligible candidates.

These numeric defaults are initial implementation choices, not experimentally
validated settings. Alpha and epsilon must be finite and in [0,1]. PFSP requires
multi-possession games and an enabled opponent pool. It ignores the legacy
recency size/beta/exploration weights when selecting among archived candidates.
With the legacy snapshot interval of zero, it does not change which checkpoints
are eligible. The independent snapshot cache below uses its own explicit schedule;
unscheduled pause/stop checkpoints still do not create training opponents.

## In-memory opponent snapshot cache

New trainer runs default to:

- `opponent_snapshot_every_updates=50` (`--opponent-snapshot-every-updates`):
  capture at **U1, U50, U100, ...**. U1 bootstraps self-play; until then an
  unseeded run uses uniformly sampled legal actions. Existing games retain that
  random opponent until reset. Explicit frozen opponents can seed the pool.
- `opponent_pool_size=600` (`--opponent-pool-size`): maximum resident opponent
  slots in cache mode, not the number selected each update. Both recency and
  PFSP sampling can use the cache. Capacity must be at least two.

CLI defaults retain interval zero for compatibility. Zero selects the original
disk-checkpoint-based admission schedule and keeps pool size's old recent-window
meaning. Saved legacy trainer runs and imported legacy configs explicitly retain
zero; they are never silently converted. Editing a copied config to interval 50
and capacity 600 starts a new cache experiment; resuming an existing run cannot
change its cache mode, capacity, cadence, or PFSP configuration.

Snapshots contain frozen policy weights, **not optimizer or environment state**.
There is no per-snapshot file, upload, deploy evaluation, or Elo match. Disk and
evaluation schedules are unchanged. An ordinary recovery checkpoint contains
the active pool and any pending weights, so resume remains exact; multiple
durable checkpoints still each include their pool state (no deduplication claim).

The weight bank has a fixed allocated shape from the start. Admission/replacement
writes a slot; unchanged updates reuse the same bank object. Small probability
vectors and outcome statistics still update every rollout. Unused and retiring
slots have exactly zero sampling probability. A completely empty cache keeps the
legal-random fallback. Legacy pools also reuse their stacked bank between
admissions, although their shape still grows when candidates are added.

### Eviction and pinned games

When full, choose the least recently used **eligible** opponent. Usage is the
number of newly assigned games across both modes/sides, exponentially decayed
with a 500-update half-life—not lifetime games, episode length, or win score.
Protect newly admitted opponents for two snapshot intervals, the newest one or
two snapshots depending on capacity, and up to three spread-out historical
anchors when capacity permits. These are fixed implementation rules, not extra
tuning knobs.

A victim first stops receiving new games; its weights remain unchanged until
all pinned games finish. At most one pending snapshot waits outside the resident
cache. If another scheduled snapshot arrives while waiting, it replaces the
pending snapshot, not an active opponent. Admission may therefore be delayed
under capacity/protection pressure; pending/superseded counts make this visible.
Resident storage never exceeds capacity, plus one bounded pending weight tree.

Resume restores slot identities/weights, usage, retirement and pending state,
admission/eviction history, PFSP evidence, pinned games/modes, and RNG. A cache is
a bounded opponent population, **not full-history FSP**; pruning can remove a
strategy that would become useful later.

### Usage chart and cache diagnostics

The trainer's primary pool view is a bar chart: **opponent snapshot update** on
the x-axis and **percentage of newly assigned games in the latest logged rollout**
on the y-axis. Both starting sides/action modes are combined; legal-random games
are excluded. Zero assignments show an unavailable sample, not invented usage.
This is observed usage, not the next-game PFSP selection probability. Old runs
without per-rollout counters fall back to clearly labeled lifetime assignments.
Two overlaid lines show **PPP scored** (green, solid) and **PPP allowed**
(amber, dashed) on a separate right-hand PPP axis; usage retains the left-hand
percentage axis. To match the combined-mode usage bars, the chart sums points and
possession counts across sampled/deterministic modes before dividing, rather
than averaging their ratios or weighting by usage. The details table still
separates modes. Hover or focus a bar/point for both PPP values and usage.
Missing or zero-denominator PPP stays blank and breaks the corresponding line;
measured zero PPP is plotted.
The **PPP view** selector has only two choices, both with opponent snapshot
update on the x-axis and just two PPP lines:

- **Current update** (default): score deltas and completed offensive possessions
  from that one logged training update's rollouts, separately for each opponent.
  It is not a difference between sparse cumulative logs or a multi-update window.
- **Pooled (cumulative)**: all tracked training history for each opponent since
  PPP tracking began, preserving the original overview.

Usage bars stay based on the latest rollout in both views. The current-update
view requires `pfsp/current_ppp_*` metrics from the same update as usage; absent
or stale data stays unavailable rather than falling back to cumulative values.
Hover/focus shows current possession sample counts. One update can have few or
no completed possessions against a particular opponent, making PPP noisy or
unavailable. Scores retain the existing training PPP convention, including
violation points and unfinished/truncated games; no new evaluations are run.
Current metrics reuse the existing per-rollout aggregates with a small host-side
logging addition. Existing workers need a pause/resume with the updated code.
There is no opponent-history overlay or extra history-fetching UI.
PFSP evidence and probabilities remain in collapsed details; inactive modes and
evicted identities are hidden from the current-pool view. History remains intact.

The **Win %** column is actual learner wins divided by tracked completed games,
separately per opponent and action mode. Ties count as non-wins; truncations are
excluded. Unlike `w`, this is cumulative, with no decay, cap, or prior, and does
not affect sampling. `pfsp/wins`, `pfsp/win_rate_games`, and `pfsp/win_rate`
(each suffixed with `/<opponent>/<mode>`) expose the numerator, denominator, and
ratio. Zero tracked games or missing legacy metrics display as unavailable.
New checkpoints preserve the counters. Resuming an older checkpoint starts
these counters at zero without altering its PFSP evidence or total Games;
the UI labels this **Partial history** and shows the counted games on hover.
Existing workers must resume/relaunch with the new code to collect these metrics;
old raw win counts cannot be reconstructed from the smoothed `w` score.

**PPP scored** and **PPP allowed** are cumulative per-opponent training diagnostics:
learner points / learner completed offensive possessions, and opponent points /
opponent completed offensive possessions, respectively. They combine both starting
sides but use the actual possession role, and remain separate by opponent action
mode. These use actual team score, including awarded violation points, not PPO
rewards or expected shot value. Higher scored and lower allowed are preferable,
but these are evolving-training-policy results, not fixed-checkpoint evaluations.

Like the existing training efficiency metrics, they count observed active steps
and completed possessions, including those in ongoing or later truncated games;
they do not wait for a completed game. Unfinished possessions do not enter the
denominator. Scores and possession counts accumulate across rollout boundaries.
Zero possessions display `—`, not zero efficiency. Counters have no decay, cap,
or prior and do not change opponent selection. Hover over each value to see its
points, possession denominator, and first tracked update.

Metrics under `jax/train/pfsp/` include `ppp_scored`, `ppp_allowed`,
`ppp_points_scored`, `ppp_points_allowed`, `ppp_offensive_possessions`,
`ppp_defensive_possessions`, and `ppp_tracking_start_update`, each followed by
`/<opponent>/<mode>`. They persist with the PFSP checkpoint state. Legacy resumes
initialize only these new counters; existing win estimates, histories, and pool
state remain intact. PPP collection starts after resuming with the new code,
without backfilling earlier results.

`jax/train/pfsp/current_ppp_` metrics include `points_scored`, `points_allowed`,
`offensive_possessions`, `defensive_possessions`, `scored`, `allowed`, and `update`,
again followed by `/<opponent>/<mode>`. Unlike the cumulative pre-rollout PFSP
snapshot, these are emitted after observing exactly that update's two starting-
side rollouts. Counts reset for every update, including unobserved opponents;
ratios are absent with zero denominators. They need no checkpoint state and do
not affect sampling, rewards, or model updates.

`jax/train/opponent_cache/` includes `capacity`, `size`, `interval_updates`,
`newest_snapshot_update`, `pending_snapshot_update`, `retiring_count`, `admissions`,
`evictions`, `bank_writes`, `bank_bytes`, and `superseded_pending`.
Per-opponent `usage_percent/<id>`, `assigned_games_latest/<id>`,
`checkpoint_update/<id>`, and `snapshot_update/<id>` identify the observed rollout.
`recent_assigned_games/<id>` is the decayed eviction score. Pool/PFSP snapshots
describe the pre-rollout population; observed usage is collected during that
rollout. End-of-update admissions appear in the next rollout's metrics.

For the measured 4v4, one-layer AR policy: 98,480 float32 parameters = 393,920
bytes (0.376 MiB) per snapshot. A 600-slot bank is 225.4 MiB; resident individual
trees plus the bank are about 450.8 MiB, excluding pending/checkpoint copies,
temporary buffers, compilation, and other training memory. U1 plus every 50
updates produces 601 snapshot requests over 30K updates, so a 600-slot cache
eventually evicts or stages one. This is a storage calculation, not a measured
whole-training throughput improvement.

A warmed CPU-only bank-maintenance check using that exact tensor shape and 600
slots measured 0.00055 ms median for an unchanged `advance` call and 0.185 ms
median / 0.313 ms p95 for a donated slot write (30 samples, one compiled variant).
Compilation, rollout/PPO work, logging, eviction bookkeeping, and disk IO were
excluded. This is not a production throughput benchmark.

For recent win score `w` against each candidate:

```text
even = normalize(w * (1 - w))
hard = normalize((1 - w)^2)
P = (1 - epsilon) * ((1 - alpha) * even + alpha * hard) + epsilon / N
```

Normalize each component separately. An all-zero component becomes uniform.
An empty archive retains the existing legal-random fallback. A singleton has
probability one. New opponents start with `w=0.5` and receive nonzero coverage.

## Evidence, ownership, and resume

Only completed game scores count: win 1, tie 0.5, loss 0. Truncations are counted
separately and do not update w. The learner's stable team identity determines
the result, regardless of who holds possession at the final step. The trajectory
stores the terminal result and the opponent identity/mode **before reset**.

Evidence decays with a 500-learner-update half-life and is capped at 256 effective
games per opponent/mode, with a Beta(1,1) neutral prior. Both starting-side batches
are aggregated before applying the cap so side iteration order does not bias w.
Lifetime game counts, starting-side counts, truncated games, assigned-game and
active-step exposure, and last-observed update remain available separately.
With no recent evidence the estimate relaxes toward 0.5.

Sampled and deterministic opponents have independent estimates and conditional
sampling distributions. Choose the next game's mode, then sample from that
mode's distribution. Both remain pinned until the game actually resets.
Distributions are updated at learner-update boundaries; completions within one
rollout inform the next update, not the middle of the compiled rollout.

Checkpoints retain estimator state/config, stable candidate identity, pinned
assignments/modes, environment state, and RNG state. Legacy checkpoints without
PFSP evidence initialize neutral priors with an explicit resume message; unchanged
legacy recency configs remain compatible. Resume rejects changes to an existing
PFSP sampler configuration. No active run is converted automatically.

## Training observability

`jax/train/pfsp/<field>/<opponent_id>/<action_mode>` logs the pre-rollout sampling
snapshot. In particular `w` is exactly the estimate used to construct that
update's probabilities, not a post-rollout estimate. Fields include `probability`,
`checkpoint_update`, `completed_games`, `truncated_games`, `effective_games`,
`prior_only`, `last_observed_update`, `age_updates`, `offense_start_games`,
`defense_start_games`, `assigned_games`, and `active_steps`. Age/last-observed
are -1 for unobserved entries; this is not a zero-game measured win rate.

The trainer shows the combined usage/PPP chart and a per-opponent details table. All fields are
independently selectable in the existing metric picker. The core MLflow profile
retains PFSP/pressure diagnostics. Metric writes are batched to avoid one HTTP
request per opponent-field pair.

## Expected shot value and defensive pressure

The same additive diagnostic is used by training, deploy evaluation, and the
role-specific evaluation paths:

```text
<prefix>/pressure/<role>/<metric>
```

Prefixes are `jax/train`, `jax/eval_deploy`, `jax/eval_offense`, and
`jax/eval_defense`. Roles are `learner_attacking` (pressure received from the
opponent) and `learner_defending` (pressure applied by the learner), based on
**current possession**, not the initial rollout group.

Metrics:

- `live_samples`, `ball_handler_samples`.
- `mean_offense_nearest_defender_distance`, `mean_ball_handler_nearest_defender_distance` (hex-cell steps).
- `ball_handler_pressure_rate`, `unguarded_offense_fraction` (fractions).
- `mean_ball_handler_uncontested_esv`, `mean_ball_handler_defended_esv`, `mean_ball_handler_esv_reduction`.
- `mean_offense_uncontested_esv`, `mean_offense_defended_esv`, `mean_offense_esv_reduction`.

ESV means shot points times make probability, in expected points per hypothetical
shot, **not PPP or the policy value estimate**. Uncontested ESV removes only the
shot-contest effect while holding position and shooting skills fixed. Defended
ESV reuses the actual JAX shot profile, including angular eligibility, distance,
contest skills, strongest eligible defender, and final probability clipping.
Suppression is uncontested minus defended ESV on identical samples.

Sample all active, non-ended live states, not just shots selected. Include valid
zero-pressure states. Ball-handler measurements require an on-court offensive
holder; loose balls are excluded. Team opportunity means first average the
on-court offensive players within a state, then average states. States with no
on-court defenders are outside the valid live-state diagnostic sample. Zero
samples have no estimate, not an invented zero. Existing pooled metrics remain
unchanged; old runs without new metrics are not backfilled.

This measures direct contest effect, not all defensive value. Forcing worse shot
locations, denying passes, and causing turnovers are separate effects. Deploy
currently uses the same policy on both sides: its role split is still self-play,
not an isolated test of defense against a fixed offense. Trainer metric cards
explain these semantics.

## Historical evaluation win score

Historical/Elo evaluation additionally exposes empirical per-opponent
`w_eval = (wins + 0.5 * ties) / completed games`. The trainer shows counts and
mode alongside Elo, including unavailable estimates for zero completed games.
Existing artifacts can supply this view without modification or reruns. New
evaluations log `jax/eval_historical/w_eval/u<opponent_update>/deterministic`
at the candidate update, plus completed/truncated games and W/T/L counts.
This is not the fitted Elo score and never updates the training estimator.

## Verification and experiment boundary

Focused tests cover the formula/fallbacks, mode separation, aging, terminal
ownership, config and persistence, ESV parity, and frontend display. A separate
compiled CPU smoke test exercises real training and deterministic resume.
No extra evaluation games or MCTS are required for these metrics.

A warmed CPU diagnostic-kernel measurement (100 alternating calls, 512 2v2
states) measured median 0.465 ms for existing spatial diagnostics and 0.610 ms
including the new role/ESV diagnostics: approximately 0.145 ms additional per
512 states. This is a diagnostic-only microbenchmark, not an estimate of
whole-trainer or production GPU overhead; compilation is excluded.

For the initial PFSP experiment, keep opponents sampled-only and compare with
the legacy sampler at matched training budgets and wall-clock cost. Use training
metrics/opponent exposure for process diagnosis and common-reference evaluation
or balanced cross-play for strength. PFSP does not guarantee 50% win rates or
equilibrium, and a changing pool makes aggregate training win rates incomparable
without exposure context.
