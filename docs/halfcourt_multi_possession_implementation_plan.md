# Half-court multi-possession implementation plan

Status: issues #19 through #23 are implemented, acceptance-tested, and
accepted. Issue #24 is integrating evaluation, the development UI, and
end-to-end validation; it remains open pending maintainer approval.

Parent tracker: [#18](https://github.com/EvanZ/basketworld/issues/18)

Working branch: `codex/halfcourt-multi-possession`

Baseline: [c9c5112](https://github.com/EvanZ/basketworld/commit/c9c5112ee4c2a9d603bec6ff0e295c9157b63539) on `main`, including the [rebounding milestone](https://github.com/EvanZ/basketworld/issues/17).

## Goal

Let fixed teams play repeated possessions at the existing half-court hoop. A defensive rebound, made basket, or possession-ending turnover becomes a transition within a game. An episode ends after a configurable number of completed possessions.

This milestone introduces inbounding and clearance before a later full-court environment. It does not introduce a game clock or periods.

## Agreed rules

### Teams, spawning, and possessions

- Keep stable team identities, player membership, and scores while offensive/defensive roles change.
- Sample each stable team's players into distinct, neutral on-court cells at game start, before either team has possession. Then assign the 50/50 jump-ball winner and its ball-handler. Do not use offensive spawn-distance or defender-matching rules in this mode. Start templates are disabled. This is the only full spawn during a game.
- Start the opening possession with the full shot clock; do not randomize its
  initial value in continuous mode.
- At each game reset, sample the starting offense with equal probability for either team. Treat this as the result of a jump ball; do not simulate the jump itself. Apply the existing holder-selection logic to the selected offense and record the result for reproducibility and diagnostics.
- Preserve positions across live possession changes. For dead-ball restarts, relocate only the inbounder; do not reset the remaining formation.
- Preserve player attributes, including shooting and rebound skills, throughout the game.
- Made baskets, defensive rebounds, and possession-ending turnovers complete a possession. An offensive rebound extends the current possession.
- Reset the shot clock and possession-scoped assist/lane/play state for a new possession. Retain the existing offensive-rebound continuation and shot-clock reset behavior.
- Count completed possessions, including a possession lost during inbounding, exactly once. Resolve the final score before computing the game outcome, and do not start a new inbound after game termination.
- Use a configurable possession limit with a default of 25. Support odd/even
  limits; a seeded 50/50 opening-possession draw keeps training fair in
  expectation, while paired evaluation swaps the starter to remove the
  remaining odd-game variance.
- Keep the existing single-possession mode available. Configuration must make mode selection and incompatible combinations explicit.

### Baseline inbounding

- Made baskets and possession-ending violations use an actual baseline inbound.
- Start from the axial coordinate one `MOVE_W` step behind the basket (currently
  basket `[-4, 8]`, inbound `[-5, 8]`). Select the nearest player on the team
  receiving possession, with stable player-ID tie-breaking, and relocate only
  that player.
- Give the inbounder a configurable five-second release deadline.
- Start the shot clock as soon as the inbounder has the ball. Other players move while the countdown runs.
- Reuse ordinary passing/interception behavior with an outside-baseline origin. Protect the held ball from direct steals while the inbounder is outside; a released pass can be intercepted.
- Apply the deadline to pass release, not catch/flight completion.
- After release, the inbounder can legally re-enter and become a normal
  receiver. In the current geometry the only adjacent inbounds cell is the
  basket cell, so this entry remains legal even when ordinary dunk-position
  movement is disabled. If it is occupied, entry is masked until it becomes
  free; moving occupants do not vacate it early within the same simultaneous
  movement step.
- An inbound timeout or failed pass with no controller produces another dead-ball restart for the appropriate team. A defensive interception starts live possession for the actual interceptor.
- On a new dead-ball restart before the prior inbounder has re-entered, restore
  that player to the nearest free court cell before selecting the new inbounder;
  this prevents repeated violations from stacking players outside the baseline.
- Reset lane counters at the possession boundary, freeze them during the
  inbound countdown, and resume them on the first live-ball step.
- Resolve a valid released pass first on a deadline step, including its
  completion or interception. Only when there is no valid release do invalid
  target, inbound-deadline, and shot-clock violations apply, in that order.
- Explicitly classify violations by the offending team. A defensive lane violation keeps its existing one-point award, completes the possession, and gives the same offense a baseline inbound; it is counted exactly once and never switches the ball to the violating defense.

### Clearance

- The new offense must have an inbounds player control the ball beyond the court's existing three-point geometry before shooting.
- Movement or a completed pass can establish clearance. Merely passing across the line, or sampling a rebound target beyond it, does not count.
- A new holder already beyond the line is immediately cleared.
- The outside-baseline inbounder cannot clear the ball. An inbound receiver outside the arc can; a receiver under the basket cannot.
- Once cleared, the ball may return inside without clearing again. Offensive rebounds preserve clearance.
- Keep shooting selectable before clearance, but resolve the attempt as a
  clearance violation: record no shot attempt, complete the possession once,
  and give the other team a baseline inbound. Retain ordinary movement/passing
  and the running shot clock while clearing.
- Another turnover during clearing starts the other team's possession with clearance recomputed from its holder.
- Let the policy learn to clear. Initially add no scripted clearance controller, clearance reward, or remedial curriculum. Measure success/time and failures first.

## Rewards and learning objective

Start training from random initialization. There is no pretrained-policy or continuation-run requirement.

The primary `win_loss` objective uses +1 for a win, -1 for a loss, and 0 for a
tie, awarded once at actual game termination. The separately selectable
`point_differential` objective awards the final fixed-team score difference at
the same boundary. Both are zero-sum.

For score-potential shaping, define potential from each fixed team's perspective:

```text
Phi_A(state) = scale * (score_A - score_B)
Phi_B(state) = -Phi_A(state)
shaping_A = gamma * Phi_A(next_state) - Phi_A(state)
```

Force terminal potential to zero only at the actual end of the game, with the
corresponding final adjustment. Preserve the effective beta-weighted potential
through ordinary possession changes and rollout boundaries. When beta changes
between updates, subtract the cached prior effective potential and use the new
beta for the next potential; this preserves telescoping for gamma=1.

Legacy pass/assist/violation/EP/rebound rewards are disabled in the initial
multi-possession objective. They can be added only with the explicit
`multi_possession_aux_rewards_enabled` configuration and are logged separately
from raw game reward, fixed-team score changes, and phi shaping.

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

The state contract reserves inbound team/player/reason and clearance fields.
#20 assigns and runs the inbounder; #21 completes live clearance detection.

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

Implementation notes:

- every active rollout step is PPO-eligible even when its game continues beyond
  the update boundary; GAE bootstraps across rollout cuts and stops only at a
  true game terminal;
- the two fixed-team cohorts share one actor-critic update while observations,
  intent context, selector eligibility, and diagnostics follow each team's
  current offense/defense role;
- historical opponent identity and deterministic/stochastic action mode are
  pinned for each game row and change only after a real game reset;
- checkpoints preserve continuous environment state, the opponent candidate
  pool, per-row opponent assignments, action modes, and opponent RNG state;
- `scripts/run_jax_5v5_halfcourt_multi_possession.sh` is the fresh-training
  launcher for the agreed 25-possession, 1,024-environment configuration.

### 6. [evaluation, UI, and end-to-end validation (#24)](https://github.com/EvanZ/basketworld/issues/24)

Users can run and inspect full half-court games, compare policies fairly, and distinguish correct mechanics from whether clearing has been learned.

Dependencies: [#19](https://github.com/EvanZ/basketworld/issues/19), [#20](https://github.com/EvanZ/basketworld/issues/20), [#21](https://github.com/EvanZ/basketworld/issues/21), [#22](https://github.com/EvanZ/basketworld/issues/22), [#23](https://github.com/EvanZ/basketworld/issues/23).

The individual issues contain detailed scope, primary code areas, and acceptance tests. Each implementation increment owns its focused tests; the final issue covers integration and user-facing validation.

#### #24 implementation checklist

- [x] Carry continuous-game static configuration through checkpoint hydration,
  native evaluation, and the interactive JAX runtime.
- [x] Keep policies attached to stable Team A/B rosters while their observed
  offense/defense role changes possession by possession.
- [x] Expose score, active offense, possession count, inbound countdown,
  clearance state, baseline inbounder, and legal re-entry cells in the UI.
- [x] Disable and explain start-template controls in continuous mode.
- [x] Pair native-evaluation reset seeds with swapped starters and report
  completed W/L/T separately from horizon-truncated games.
- [x] Publish inbound, clearance, turnover-before-clearance, rebound, score,
  margin, possession, and points-per-possession diagnostics.
- [x] Complete final focused mechanics, PPO/resume, backend, frontend-build,
  and legacy-regression validation; request maintainer approval before issue
  closure.

### 7. [randomized check location and check clearance (#62)](https://github.com/EvanZ/basketworld/issues/62)

Status: the initial implementation landed on 2026-10-12 in
`f10805b` (`Implement configurable check clearance`). All identified rule
decisions are agreed, including the double-violation limit, random penalty,
and pickup/reentry precedence. The unchecked acceptance items below remain
the release-validation and learning-study work; they are intentionally not
claimed by the implementation commit.

#### Completion review (2026-10-12)

- [x] Add `check_clearance`, separate setup/pickup timers, legacy-compatible
  defaults, environment validation, and eligible outside-arc pickup sampling
  with a full clearance-plus-one inbounds margin.
- [x] Implement per-environment sampled check locations and retry counters,
  setup/pickup state transitions, single-team technical consequences, two
  no-score double-violation retries, and the seeded third-violation draw.
- [x] Wire the new state through JAX training, native evaluation, inference,
  checkpoint snapshots, aggregate check diagnostics, and the JAX dev-runtime
  state payload.
- [x] Add the schema-five policy features: sampled coordinates, clearance
  radius, double-violation count, actual active-phase time, dynamic
  ball-distance, and `inside_check_clearance`.
- [x] Add trainer form/backend/preset/import support for the three settings,
  including five-second new-run defaults and the nonzero-clearance setup-time
  validation.
- [x] Document the implemented rule in the repository overview and trainer
  application guide, and run the initial focused JAX mechanics and trainer
  application checks.

This section governs the new check-clearance behavior rather than changing
the historical implementation decisions above.

#### Configuration and compatibility

- Add nonnegative integer `check_clearance`, measured in hex rings around
  the loose check ball. Radius `1` protects the six adjacent tiles; radius
  `2` protects the first two rings. The ball tile is also protected.
- `check_clearance=0` disables the new clearance/random-location rules and
  retains the legacy check mechanic, including its existing ball-cell handling.
- `random_check_location` defaults to disabled. Nonzero-clearance checks use
  one deterministic eligible anchor nearest the legacy check location while
  retaining the clearance rule and observations; random locations are an
  explicit future-experiment opt-in.
- The intended default for new configurations is **five setup seconds,
  followed by a separate five-second pickup deadline**. One simulation step
  represents one countdown second. Currently setup defaults to zero and
  pickup defaults to five; update the setup default consistently across
  training and app configuration.
- Preserve explicit saved timer values. Restore older configurations and
  checkpoints with their legacy defaults when these fields are absent,
  rather than silently changing an existing run's rules.
- Nonzero clearance requires positive setup time. Validate radius, court
  geometry, and pickup deadline together so clearing the protected area
  does not make timely pickup geometrically impossible.

#### Random pickup location

- For nonzero clearance, sample uniformly from eligible court tiles once
  when a new check begins, using the environment RNG. Do not require those
  tiles or their surrounding areas to be initially unoccupied: moving clear
  is part of the task the policies should learn.
- The pickup tile must be outside the existing three-point arc, because
  collecting the check ball establishes offensive clearance to shoot.
- Require the entire hex disk of radius `check_clearance + 1` around the
  pickup tile to lie inside the court. This leaves one full ring in which
  players can stand outside the protected radius without being squeezed
  against a boundary. Radius `1` therefore needs two complete inbounds
  rings; radius `2` needs three.
- Reject configurations with no eligible pickup tiles; do not silently
  relax the boundary margin, arc requirement, or radius.
- Reveal the sampled location to both teams immediately and keep it fixed
  throughout setup, pickup, and any repeat setup caused by a double
  violation. A violation must not offer a new draw of the location.
- Preserve every player's on-court position when the check begins. In this
  mode, even a player already on the ball tile must learn to leave it;
  do not automatically relocate that player or allow setup-time pickup.

#### Setup, pickup, and violations

- During setup, players can move out of or through the surrounding
  protected area. Evaluate positions after the final setup movement step.
  Every player on both teams must then have hex distance strictly greater
  than `check_clearance` from the ball. No pickup occurs during setup.
- Only a successful setup starts the full pickup countdown. Keep the shot
  clock frozen during setup and pickup, as in the existing check phases.
- During pickup, offensive players may enter the protected area to collect
  the ball. Defenders must remain outside until pickup makes play live;
  defensive reentry is a check-space violation. Allow policies to learn
  surrounding-ring compliance rather than masking away those violations.
- On a step that begins in the pickup phase, resolve movement and check
  defensive clearance before accepting an offensive pickup. If a defender
  enters the protected area on the same step an offensive player reaches
  the ball, apply the defensive violation and its restart; do not record a
  successful pickup. Defenders must remain outside on the pickup step and
  may enter starting on the next step after a valid pickup makes play live.
- A defense-only violation awards one point to the offense, retains the
  same possession, and uses the existing defensive technical restart:
  sideline inbound, or direct handoff when inbounds are disabled. It does
  not complete a possession or terminate the game trajectory by itself.
- An offense-only setup violation awards one point to the defense and
  ends the offending offense's possession exactly once. The new offense
  receives the technical restart, mirroring the existing pickup-timeout
  consequence. Respect the normal game-end check before starting a restart.
- Multiple offending players on one team produce one team violation and
  one point, not one penalty per player.
- On the first and second consecutive double violations at the setup
  boundary, award **no points**, retain the same offense and possession
  count, and repeat the full setup period at the same location. Preserve
  positions and keep the pickup clock stopped.
- On the **third consecutive double violation**, randomly assign a single
  violation to one team with equal probability, using the environment RNG.
  Apply that team's normal single-team penalty and technical restart as
  defined above. Penalizing the offense completes its possession once;
  penalizing the defense retains the possession. Award only one point to
  the non-penalized team, never offsetting points to both teams.
- The limit counts double-violation events, not retries: the initial setup
  can fail twice and be repeated twice; the third failure resolves the
  check without another setup. Initialize the count to zero for each new
  check, preserve it across repeat setup periods and checkpoint resume,
  and clear it when setup succeeds or a penalty resolves the check.
- Keep the existing offensive pickup-timeout penalty after a valid setup.
  After a valid pickup, the check-space restriction ends and normal live
  play resumes.

#### Implementation scope

- Carry configuration through training CLI/presets, checkpoint and env
  metadata, native evaluation, training UI, and dev app controls.
- Store the sampled location and consecutive double-violation count in
  per-environment state. Use that location consistently in movement rules, observations,
  pickup detection, snapshots, checkpoint resume, and app rendering rather
  than assuming one static check coordinate for the whole batch.
- Expose the current ball location, protected area, phase, separate
  countdowns, and double-violation count/limit in the dev app. Both policies
  must observe the count, location, and enough rule context to respond to it.
- Record distinct offense, defense, and double check-space violations,
  offending players, setup retries, random limit resolutions and selected
  penalized teams, pickup success/timeouts, pickup latency,
  and stable-team score/possession consequences. Keep these separate from
  live-ball clearance violations and the existing pickup-timeout event.
- Preserve continuous PPO/GAE behavior across setup, retries, and ordinary
  possession boundaries. Verify score changes feed the existing game
  reward and score-shaping paths exactly once.

#### Approved policy observations

- Reuse existing player positions, offense/defense roles, restart phase,
  and normalized active countdown. The phase continues to distinguish
  setup from pickup; the role determines which players remain restricted
  during pickup.
- Add four global values for the check mechanic: sampled pickup `q` and
  `r` coordinates, `check_clearance` radius, and consecutive double-violation
  count. Both actor and critic, for both teams, receive these features.
  The count makes the different consequences after zero, one, or two
  double violations observable even when positions and clocks match.
- Update the existing per-player distance-to-ball feature to use the
  sampled per-environment check location during setup and pickup, rather
  than the old fixed coordinate. Expose coordinates directly as well so
  policies can determine the direction to the randomized destination.
- Add one derived per-player boolean, `inside_check_clearance`, indicating
  hex distance less than or equal to the configured radius, including the
  ball tile, during a nonzero-clearance check. This reports geometry, not
  a completed violation: occupancy is allowed before the setup boundary,
  and offensive players may enter during pickup. Role and phase provide
  that context. Clear the flag outside the relevant check phases.
- Also expose the actual remaining time of the active setup/pickup phase
  as a global value. Reuse the phase indicator to identify which timer it
  describes. If scaled, use a fixed documented scale so different timer
  configurations with the same remaining fraction do not appear identical.
  Retain the existing normalized countdown. This adds five global values
  in total, including the four check-context values above.
- Define consistent scaling and inactive-phase values for the new fields;
  do not expose a stale sampled location or retry count after the check.
  Do not expose the future random penalty draw.
- Introduce a new observation schema version for this extension. Preserve
  old checkpoint dimensions, feature ordering, and legacy observation
  semantics. Persist the version in model/checkpoint metadata and use the
  matching dimensions and labels in training, evaluation, and the dev app.
  Nonzero-clearance policies train from scratch with the new schema.

#### Trainer app configuration requirements

- Include the trainer app (`app/training_frontend` and
  `app/training_backend`) in this feature's implementation scope, in
  addition to the separate interactive dev app.
- Provide labeled trainer controls for `check_clearance` (hex rings),
  `check_setup_steps` (default five seconds for new configurations), and
  `check_deadline_steps` (default five seconds). Every additional training
  configuration option introduced for #62 must also be available in the
  trainer app, rather than requiring manual CLI arguments or raw overrides.
- Explain the effective rules beside the controls: nonzero clearance uses a
  safe eligible pickup location, `random_check_location` selects randomized
  locations or a deterministic anchor, boundary margin is clearance
  plus one ring, and the third consecutive double violation triggers a
  50/50 assignment of the normal single-team penalty. These agreed rules
  need not become separate toggles simply to appear in the UI.
- Keep frontend defaults, backend models/validation, application presets,
  and compiled training arguments consistent. Validate check-mode
  applicability, positive setup time for nonzero clearance, available
  pickup locations, and radius/deadline compatibility before launching.
- Preserve the settings through saved run configurations, run duplication,
  MLflow parameter import, checkpoint metadata, and resume. Retain explicit
  historical values and legacy fallback behavior for missing fields.
- Show the effective settings in run configuration/details so the operator
  can verify what was actually passed to training. Ensure dedicated form
  fields and advanced overrides cannot silently disagree.
- Primary integration points include
  `app/training_frontend/src/App.vue`, `app/training_backend/models.py`,
  `app/training_backend/presets.py`,
  `app/training_backend/application_training.py`, and
  `app/training_backend/mlflow_import.py`.

#### User documentation requirements

- Update the user-facing documentation as part of implementing #62; this
  plan and the GitHub issue alone are not sufficient documentation of the
  shipped mechanics. Until implementation lands, label the new behavior
  as planned rather than describing it as already available.
- Describe the full check lifecycle in the environment/movement rules:
  sampled location outside the arc, clearance-plus-one boundary margin,
  unchanged player positions, five-second setup, both-team clearance,
  five-second pickup, defender-only restrictions during pickup, and the
  transition to live play. Distinguish check-space clearance from the
  existing requirement to clear the ball beyond the arc before shooting.
- Document offense-only and defense-only violations, their point and
  possession consequences, same-step defensive-violation precedence, and
  the first two no-score double-violation retries followed by the third
  violation's 50/50 single-team penalty. Include a short worked example.
- Update the training and trainer app guides with configuration names,
  units, defaults, validation constraints, valid example settings, trainer
  controls, saved settings, and checkpoint compatibility. Explain radius
  zero and legacy timer behavior without implying old runs change rules.
- Update the dev app guide to explain the displayed pickup location,
  protected area, setup/pickup timers, double-violation count, and events.
  Describe the diagnostics used to assess learned compliance and pickups.
- Document the approved observation fields, per-player geometry flag,
  actual-time encoding, phase semantics, scaling, and schema compatibility
  in the actions/observations reference.
- Primary destinations are `docs/site/physics/movement-rules.md`,
  `docs/site/physics/rewards-termination.md`,
  `docs/site/environment/actions-observations.md`,
  `docs/site/jax/training.md`, `docs/training_app.md`, and `app/README.md`.
  Add or adjust overview links where needed and reconcile conflicting
  fixed-location or timer descriptions in the affected documentation.

#### Acceptance and validation

- [ ] Test hex radii `0`, `1`, and `2`, including the ball tile, exact radius
  boundaries, both teams, and multiple offending players.
- [ ] Test seeded eligible-location sampling, outside-arc geometry, full
  boundary margins, empty candidate rejection, and deadline feasibility.
- [ ] Test five setup steps followed by five pickup steps, final-step
  movement, frozen shot clock, and no premature pickup.
- [ ] Test both single-team penalties, unchanged possession accounting for
  defensive violations, one completed possession for offensive violations,
  double violations, fixed-location retries, and defensive reentry.
- [ ] Test no penalty on double violations one and two, exactly one seeded
  50/50 team selection on the third, both resulting penalty paths, counter
  reset/persistence, and no fourth setup period. Verify reproducibility and
  equal-probability selection independent of stable team identity.
- [ ] Test that same-step defensive reentry takes precedence over offensive
  pickup, applies only the defensive penalty, and records no pickup. Also
  test valid pickup with defenders outside and legal defensive entry on
  the following live step.
- [ ] Test no automatic relocation for nonzero clearance, including an
  initial occupant of the sampled ball tile.
- [ ] Test legacy radius-zero configurations, explicit saved timers, old
  checkpoint defaults, seeded resume, and mixed-phase JAX batches with
  different sampled locations.
- [ ] Verify training, evaluation, and dev runtime agreement; build affected
  frontends and run focused regression tests.
- [ ] Test observed sampled coordinates and distances for distinct check
  locations in one batch; radius and retry-count visibility for both teams;
  inside-clearance flags at the ball tile and radius boundary; setup versus
  pickup semantics; and inactive-phase values. Verify actual remaining
  time distinguishes different durations with equal normalized countdowns.
- [ ] Test new schema dimensions, labels, actor/critic inputs, metadata
  round trips, and unchanged legacy checkpoint observation layouts across
  training, native evaluation, and the dev runtime.
- [ ] Verify trainer form/API values reach the generated training command
  and recorded environment settings unchanged. Cover new-run defaults,
  validation errors, saved/duplicated/imported configurations, and legacy
  resume compatibility; build `app/training_frontend` as well as any
  affected dev frontend.
- [ ] Update the rules, training, trainer app, and dev app documentation;
  verify examples/defaults against the implemented configuration, check
  links, and run the repository's documentation build/checks where available.
- [ ] Train a fresh policy for nonzero clearance and evaluate setup
  compliance, retries, and pickup behavior separately. Random locations
  may improve adaptation; measure that rather than assuming improvement.
- [ ] Track how often the three-violation limit is reached and whether
  compliance improves. Random assignment ends setup deadlocks without a
  preferred team in the draw, but does not guarantee improved learning;
  check that policies do not learn to tolerate or exploit the lottery.

#### Decision status

No identified rule decisions remain open. The implementation work listed in
the completion review is complete; the unchecked acceptance checks above are
still pending.

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

No unresolved decisions remain for issues #19 through #23.

No identified rule decisions remain open for #62. See the check-clearance
section above for the agreed rules and pending implementation checks.

## Deferred

Full-court geometry, game clock, periods and period respawns, ordinary out-of-bounds movement turnovers, play-start templates, pretrained-policy reuse, and clearing assistance. A later strategy for failed clearance learning should be justified by recorded diagnostics.
