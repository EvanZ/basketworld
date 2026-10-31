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
It does not change which checkpoints are eligible: unscheduled pause/stop
snapshots remain resume artifacts, not training opponents.

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

The trainer shows a per-opponent snapshot and a Chart w button. All fields are
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
