# Halfcourt Rebounding Implementation Plan

Status: implementation in progress

This plan tracks the first halfcourt rebounding milestone for Basketworld JAX. The goal is not full-court multi-possession basketball yet. The goal is to let missed shots produce rebound outcomes inside the existing halfcourt possession model.

## Core Semantics

- [x] A made shot ends the episode as it does today.
- [x] A missed shot with rebounds disabled preserves current behavior and ends the episode.
- [x] A missed shot with a defensive rebound ends the episode.
- [x] A missed shot with an offensive rebound continues the same possession.
- [x] If the offense rebounds and the shot clock is below 14 seconds, reset it to 14 seconds.
- [x] Offensive rebounds produce no immediate task reward.
- [x] Defensive rebounds produce no immediate task reward.
- [x] Task shot reward becomes actual made points, not expected points, when rebounds are enabled.
- [x] Expected points remains available as diagnostics and optionally as phi potential input.

## Configuration

- [x] Add `--enable-rebounds`.
- [x] Add `--rebound-table-model-dir`.
- [x] Add `--rebound-target-temperature`.
- [x] Add `--rebound-target-uniform-mix`.
- [x] Add `--rebound-winner-distance-weight`.
- [x] Add `--rebound-winner-temperature`.
- [x] Add `--offensive-rebound-shot-clock-reset`, default `14`.
- [x] Add reward-mode config if needed, with rebound-enabled default set to actual points. Not needed for v1; `--enable-rebounds` selects actual points.
- [x] Log all rebound config params to MLflow.

## Rebound Table Artifact Contract

- [x] Define the JAX-ready artifact format. Runtime uses dense `target_probs[shot_type, shot_cell, target_cell]`.
- [x] Convert the fitted MuJoCo/table model into dense arrays:
  - [x] `target_probs[shot_type, shot_cell, target_cell]`
  - [ ] optional `shot_type_by_cell[shot_cell]`
  - [ ] optional `valid_target_mask[target_cell]`
- [x] Pre-expand reflection/canonical mapping before JIT so runtime does not need Python table logic.
- [x] Validate table shape against the active court geometry.
- [x] Disable rebounds or raise a clear config error if the table/court geometry does not match.

## JAX Env Changes

- [x] Extend `KernelStatic` with rebound config and dense target-probability arrays.
- [x] Extend `StepBatchOutput` with:
  - [x] `rebound_attempt`
  - [x] `rebound_target_cell`
  - [x] `rebound_winner`
  - [x] `offensive_rebound`
  - [x] `defensive_rebound`
  - [x] `shot_clock_reset_14`
- [x] Replace current `done = ... | shot_active` logic with rebound-aware shot resolution.
- [x] Resolve non-shooter movement on shot steps before rebound sampling.
- [x] Keep the shooter fixed on the shot step.
- [x] Treat non-holder pass actions as noops on shot steps.
- [ ] Suppress illegal non-shooter movement turnovers on shot steps if needed. Current training path relies on legal action masks; non-holder pass actions noop.
- [x] Sample rebound target from `P(target_cell | shot_type, shot_cell)`.
- [x] Sample rebound winner using distance-only logits:

```text
score_i = -distance_weight * distance(player_i, target_cell)
P(winner=i) = softmax(score_i / winner_temperature)
```

- [x] Use masks to apply rebound effects only for missed shots.
- [x] Keep the implementation JIT-friendly and vectorized.
- [x] Do not call MuJoCo or Python table-model code inside JAX training.

## Policy Observation Features

- [x] Add rebound-derived global features to the JAX observation:
  - [x] `expected_rebound_target_q`
  - [x] `expected_rebound_target_r`
  - [x] `target_entropy`
  - [x] `orb_prob_if_current_shot_misses`
- [x] Do not add `drb_prob_if_current_shot_misses`; it is exactly `1 - orb_prob_if_current_shot_misses`.
- [x] Add rebound-derived per-player features to the JAX observation:
  - [x] `dist_to_expected_rebound_target`
  - [x] `rebound_win_prob_if_current_shot_misses`
- [x] Gate these features to zero when rebounds are disabled, there is no valid ball holder, or the holder is not an offensive player.
- [x] Compute features from the fitted target table before the random rebound target is sampled, so the model gets distributional context but not future outcome leakage.
- [x] Update attention observation dimensions from `15` player features / `4` globals to `17` player features / `8` globals.
- [x] Update JAX train scaffold tests for the new observation dimensions.

## Offensive Rebound Continuation

- [x] Set `ball_holder` to the rebound winner.
- [x] Keep player positions unchanged after the rebound result itself; only pre-rebound shot-step movement can change positions.
- [x] Reset assist state after any shot attempt.
- [x] Reset both offensive and defensive lane counters after an offensive rebound.
- [x] Always mark an offensive rebound as an eligible selector boundary.
- [x] Preserve episode state as non-terminal after offensive rebound.
- [x] Apply shot-clock reset to 14 only if current remaining shot clock is below 14.
- [x] Treat the sampled rebound target as ball catch/landing metadata, not a hidden player movement.

## Reward Semantics

- [x] Replace direct shot expected-points task reward with actual made points when rebounds are enabled:

```text
made 2P shot -> +2 task points
made 3P shot -> +3 task points
miss -> +0 task points
offensive rebound -> +0 immediate task points, continue
defensive rebound -> +0 immediate task points, terminal
```

- [ ] Keep phi shaping as transition-based potential shaping:

```text
phi_reward = beta * (gamma * Phi(next_state) - Phi(prev_state))
```

- [ ] Treat made shots and defensive rebounds as terminal for phi.
- [x] Treat offensive rebounds as non-terminal for phi.
- [ ] Log task reward, phi reward, and total reward separately.

## Training Metrics

- [x] Add per-update metrics:
  - [x] `rebound_attempts`
  - [x] `offensive_rebounds`
  - [x] `defensive_rebounds`
  - [x] `offensive_rebound_rate`
  - [x] `defensive_rebound_rate`
  - [x] `shot_clock_reset_14_count`
- [ ] Add completed-episode metrics:
  - `offensive_rebounds_per_completed_episode`
  - `defensive_rebounds_per_completed_episode`
  - `points_after_offensive_rebound`
  - `possessions_extended_by_rebound`
- [ ] Add reward breakdown:
  - `reward/task_actual_points`
  - `reward/phi_shaping`
  - `reward/total`
- [ ] Keep EP diagnostics:
  - `shot_expected_points`
  - `shot_expected_points_per_attempt`
  - `actual_points_minus_ep`

## Eval And UI Follow-Up

- [ ] Surface rebound config in the Training/Eval UI.
- [ ] Show actual rebound outcome after missed shots:
  - sampled target
  - rebound winner
  - offensive/defensive rebound flag
  - shot-clock reset event
- [ ] Convert current rebound preview into an explanation view using the env-produced rebound result as source of truth.
- [ ] Add rebound stats to Eval/Stats:
  - offensive rebounds
  - defensive rebounds
  - rebound opportunities
  - offensive rebound rate
  - defensive rebound rate
- [ ] Update reward UI to separate actual task points, phi shaping, and total reward.
- [ ] Include compact rebound fields in checkpoint summaries and sample dumps:
  - target cell
  - winner
  - offensive/defensive rebound flags
  - rebound counts
- [ ] Do not dump full rebound target distributions per step.

## Tests

- [ ] Unit test: made shot still terminates.
- [ ] Unit test: miss with rebounds disabled terminates.
- [ ] Unit test: miss plus defensive rebound terminates.
- [ ] Unit test: miss plus offensive rebound continues.
- [ ] Unit test: non-shooter move actions resolve on shot steps before rebound sampling.
- [ ] Unit test: shooter does not move on shot steps.
- [ ] Unit test: offensive rebound sets ball holder to rebound winner.
- [ ] Unit test: offensive rebound resets shot clock to 14 when below 14.
- [ ] Unit test: offensive rebound does not reduce shot clock if already above 14.
- [ ] Unit test: winner distribution favors closer players.
- [ ] Unit test: missed shot does not receive expected-points task reward.
- [ ] Unit test: phi treats offensive rebound as non-terminal.
- [x] Integration smoke: forced-miss JAX step with rebounds enabled produces rebound metrics.
- [x] Integration smoke: compiled JAX rollout with rebounds enabled carries rebound fields through `TrajectoryBatch`.

## Resolved Design Decisions

- [x] Offensive rebounds reset both offensive and defensive lane counters. A shot/rebound event breaks the prior lane-possession context.
- [x] Offensive rebounds always create a selector boundary. The possession continues, but the play context should reset.
- [x] Rebound winners do not move to the sampled target cell in v1. The ball holder changes, but the rebound event itself does not move players.
- [x] Shot steps should still resolve non-shooter movement before rebound sampling so positioning matters.
- [x] The sampled rebound target is still shown in UI/diagnostics as the modeled catch/landing location.
- [x] Rebound stats are included in checkpoint summaries and sample dumps as compact fields only.
- [x] Full rebound target distributions are not dumped per step because they would bloat artifacts.
- [x] Non-shooter out-of-bounds movement turnovers are suppressed on shot steps. Legal action masks should prevent them; if an illegal non-holder move appears defensively, it noops.
- [x] Rebound reward/PPP parity note: `last_shot_ep` rewards the terminal shot by EP regardless of make/miss; `last_shot_ep_on_defensive_rebound` is legacy and biased high relative to actual PPP.

## Suggested Implementation Order

- [x] 1. Add config and artifact loading.
- [x] 2. Add `KernelStatic` and `StepBatchOutput` fields.
- [x] 3. Patch shot transition semantics behind `--enable-rebounds`.
- [x] 4. Add rebound terminal reward modes: `actual_points` for strict outcome rewards and `last_shot_ep` for unbiased terminal-shot EP rewards. Avoid `last_shot_ep_on_defensive_rebound` for reward-vs-PPP comparisons because it pays actual points on makes and EP on terminal misses.
- [x] 5. Add metrics and MLflow logging.
- [ ] 6. Add JAX unit tests.
- [ ] 7. Run a short smoke training run with rebounds enabled.
- [ ] 8. Patch backend/UI to display env-produced rebound results.
