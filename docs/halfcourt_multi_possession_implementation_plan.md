# Half-court multi-possession implementation plan

Status: implementation underway; issue #19 establishes the game-state and
possession-lifecycle foundation. Baseline inbounding remains in issue #20.

Parent tracker: [#18](https://github.com/EvanZ/basketworld/issues/18)

Working branch: `codex/halfcourt-multi-possession`

Baseline: [c9c5112](https://github.com/EvanZ/basketworld/commit/c9c5112ee4c2a9d603bec6ff0e295c9157b63539) on `main`, including the [rebounding milestone](https://github.com/EvanZ/basketworld/issues/17).

## Goal

Let fixed teams play repeated possessions at the existing half-court hoop. A defensive rebound, made basket, or possession-ending turnover becomes a transition within a game. An episode ends after a configurable number of completed possessions.

This milestone introduces inbounding and clearance before a later full-court environment. It does not introduce a game clock or periods.

## Agreed rules

### Teams, spawning, and possessions

- Keep stable team identities, player membership, and scores while offensive/defensive roles change.
- Use the existing initial spawn and holder selection with start templates disabled. This is the only full spawn during a game.
- At each game reset, sample the starting offense with equal probability for either team. Treat this as the result of a jump ball; do not simulate the jump itself. Apply the existing holder-selection logic to the selected offense and record the result for reproducibility and diagnostics.
- Preserve positions across live possession changes. For dead-ball restarts, relocate only the inbounder; do not reset the remaining formation.
- Preserve player attributes, including shooting and rebound skills, throughout the game.
- Made baskets, defensive rebounds, and possession-ending turnovers complete a possession. An offensive rebound extends the current possession.
- Reset the shot clock and possession-scoped assist/lane/play state for a new possession. Retain the existing offensive-rebound continuation and shot-clock reset behavior.
- Count completed possessions, including a possession lost during inbounding, exactly once. Resolve the final score before computing the game outcome, and do not start a new inbound after game termination.
- Use a configurable possession limit, initially around 25. Support odd/even limits; evaluate with paired starting-team swaps. Choose the precise default before the initial launcher is finalized.
- Keep the existing single-possession mode available. Configuration must make mode selection and incompatible combinations explicit.

### Baseline inbounding

- Made baskets and possession-ending violations use an actual baseline inbound.
- Start from a designated location outside the baseline. Select the nearest player on the team receiving possession, with deterministic tie-breaking, and relocate that player.
- Give the inbounder a configurable five-second release deadline.
- Start the shot clock as soon as the inbounder has the ball. Other players move while the countdown runs.
- Reuse ordinary passing/interception behavior with an outside-baseline origin. Protect the held ball from direct steals while the inbounder is outside; a released pass can be intercepted.
- Apply the deadline to pass release, not catch/flight completion.
- After release, the inbounder can legally re-enter and become a normal receiver, including under the basket.
- An inbound timeout or failed pass with no controller produces another dead-ball restart for the appropriate team. A defensive interception starts live possession for the actual interceptor.
- Define baseline coordinates, blocked-entry handling, event ordering, and lane-counter treatment in the inbound issue before finalizing those mechanics.
- Explicitly classify violations by the offending team. A defensive lane violation keeps its existing one-point award, completes the possession, and gives the same offense a baseline inbound; it is counted exactly once and never switches the ball to the violating defense.

### Clearance

- The new offense must have an inbounds player control the ball beyond the court's existing three-point geometry before shooting.
- Movement or a completed pass can establish clearance. Merely passing across the line, or sampling a rebound target beyond it, does not count.
- A new holder already beyond the line is immediately cleared.
- The outside-baseline inbounder cannot clear the ball. An inbound receiver outside the arc can; a receiver under the basket cannot.
- Once cleared, the ball may return inside without clearing again. Offensive rebounds preserve clearance.
- Mask shooting before clearance; retain ordinary movement/passing and the running shot clock.
- Another turnover during clearing starts the other team's possession with clearance recomputed from its holder.
- Let the policy learn to clear. Initially add no scripted clearance controller, clearance reward, or remedial curriculum. Measure success/time and failures first.

## Rewards and learning objective

Start training from random initialization. There is no pretrained-policy or continuation-run requirement.

The proposed primary experiment uses +1 for a win, -1 for a loss, and 0 for a tie, awarded once at actual game termination. Retain a separately selectable point-differential baseline for comparison.

For score-potential shaping, define potential from each fixed team's perspective:

```text
Phi_A(state) = scale * (score_A - score_B)
Phi_B(state) = -Phi_A(state)
shaping_A = gamma * Phi_A(next_state) - Phi_A(state)
```

Force terminal potential to zero only at the actual end of the game, with the corresponding final adjustment. Preserve potential through ordinary possession changes and rollout boundaries. Test the telescoping sum for gamma=1. If the scale/beta is scheduled across updates, retain the necessary previous potential or otherwise ensure that the schedule does not silently invalidate this property.

Current pass/assist/violation/EP and intent bonuses must be configured explicitly, rather than silently changing the win-plus-score-potential objective. Log raw outcome reward, score changes, phi, and enabled auxiliary terms separately.

Leave optional rebound reward advances disabled initially. If supported, explicitly settle any possession-scoped ledger once at possession end without treating the game critic as terminal.

## Training implications

The current rebound launch configurations collect two batches of 512 environments for 64 steps, combine offense/defense samples, and run one PPO epoch with 16 minibatches. This is 65,536 nominal team-step slots per update. The scripts use gamma=1; GAE lambda defaults to 0.95.

- Preserve the total simulation budget while replacing permanently assigned offense/defense rollouts with fixed learner teams and per-step roles. Two cohorts can start in opposite roles.
- Continue unfinished games across updates. Bootstrap the critic at a rollout boundary and cut game returns only at true game termination.
- Disable completed-episodes-only filtering: it currently excludes a game unless both its start and finish occur within the same rollout.
- Keep single-episode-rollouts disabled: that path resets state between updates, including unfinished episodes.
- Audit loss normalization/sample utilization when changing that filter.
- Collect selector/intent samples according to the current role within either cohort. A new possession can reset intent without ending the game value trajectory.
- Pin opponent identity and deterministic/stochastic action mode for the game. Publishing new opponent checkpoints must not change a currently running game's opponent.
- Save enough state for faithful resume, including scores, possession count, clocks, clearance, intent context, and opponent assignment.
- Build a fresh-run launcher without continuation arguments and with templates disabled. Historical opponents, if used, are snapshots from this run.
- Start by comparing against the existing horizon/epoch/minibatch scale. Longer horizons or different lambda are experiments, not assumed requirements.
- Track completed games and outcome frequency alongside steps and updates. A game can span several updates, so these measures are not interchangeable.
- Track starting-team counts and outcome splits. Training uses the sampled 50/50 start; paired evaluation still swaps the starter to reduce comparison variance.

## Implementation issues

### 1. [game state and possession lifecycle (#19)](https://github.com/EvanZ/basketworld/issues/19)

Introduce an opt-in JAX game mode in which teams retain their identities and alternate offense/defense across multiple possessions at the same hoop. A made basket, defensive rebound, or possession-ending turnover ends a possession; only the configured game limit ends an episode.

The state contract also reserves inbound team/player/reason and clearance fields now. They are populated at a handoff but are not yet playable or enforced; #20 assigns the inbounder and #21 enforces clearance.

Dependencies: none.

### 2. [baseline inbounding and dead-ball restarts (#20)](https://github.com/EvanZ/basketworld/issues/20)

Made baskets and possession-ending violations lead to a baseline inbound. Other players can move while the inbounder finds a receiver.

Dependencies: [#19](https://github.com/EvanZ/basketworld/issues/19).

### 3. [clearance and live possession switches (#21)](https://github.com/EvanZ/basketworld/issues/21)

After taking possession, a team must control the ball beyond the three-point line before shooting. Policies learn to clear through ordinary movement and passing.

Dependencies: [#19](https://github.com/EvanZ/basketworld/issues/19), [#20](https://github.com/EvanZ/basketworld/issues/20).

### 4. [observations and game-level rewards (#22)](https://github.com/EvanZ/basketworld/issues/22)

Both teams can observe enough game context to act, and rewards consistently represent each fixed team's result across offensive/defensive role changes.

Dependencies: [#19](https://github.com/EvanZ/basketworld/issues/19), [#20](https://github.com/EvanZ/basketworld/issues/20), [#21](https://github.com/EvanZ/basketworld/issues/21).

### 5. [continuous PPO and team-consistent self-play (#23)](https://github.com/EvanZ/basketworld/issues/23)

Train a policy from scratch on fixed rollout chunks of continuous games. Updates may happen midgame; the learner keeps controlling the same team across possession switches.

Dependencies: [#22](https://github.com/EvanZ/basketworld/issues/22).

### 6. [evaluation, UI, and end-to-end validation (#24)](https://github.com/EvanZ/basketworld/issues/24)

Users can run and inspect full half-court games, compare policies fairly, and distinguish correct mechanics from whether clearing has been learned.

Dependencies: [#19](https://github.com/EvanZ/basketworld/issues/19), [#20](https://github.com/EvanZ/basketworld/issues/20), [#21](https://github.com/EvanZ/basketworld/issues/21), [#22](https://github.com/EvanZ/basketworld/issues/22), [#23](https://github.com/EvanZ/basketworld/issues/23).

The individual issues contain detailed scope, primary code areas, and acceptance tests. Each implementation increment owns its focused tests; the final issue covers integration and user-facing validation.

## Validation strategy

- Test the state machine with deterministic, short possession chains before long stochastic games.
- Cover both teams and mixed phases in the same JAX batch, including exact clock deadlines and repeated turnovers.
- Exercise spawn, made-basket inbound, inside/outside receptions, clearance, offensive rebound, defensive rebound, live steal, violation restart, and final-game scoring.
- Test zero-sum rewards and potential sums independently of learned behavior.
- Run a PPO smoke where games outlast the rollout horizon and contribute samples before completion; verify bootstrap/terminal masks and resume.
- Verify native evaluation and interactive backend agreement for equivalent seeded action traces.
- Report wins/losses/ties only for completed games. Report evaluation cutoffs/incomplete games separately; they are not draws.
- Build the frontend and run focused legacy single-possession/rebounding regressions.
- Learned clearing performance is an experimental measurement, not a requirement for mechanics tests to pass.

## Remaining decisions

1. Exact possession-limit default (25 versus a nearby even number), retaining configurability and paired evaluation.
2. Exact outside-baseline coordinate, legal entry mapping, blocked-entry behavior, and lane-counter treatment during inbounding.
3. Same-step priority among legal last-moment release, interception, shot-clock expiration, and game termination.
4. Potential coefficient scheduling across update boundaries.

Resolve these in the owning issues and update this document before shipping the corresponding behavior.

## Deferred

Full-court geometry, game clock, periods and period respawns, ordinary out-of-bounds movement turnovers, play-start templates, pretrained-policy reuse, and clearing assistance. A later strategy for failed clearance learning should be justified by recorded diagnostics.
