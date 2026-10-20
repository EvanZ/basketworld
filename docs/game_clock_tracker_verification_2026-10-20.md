# Game-clock tracker #49 closure verification — 2026-10-20

This record combines the child-issue acceptance evidence with actual 5v5
300-second regulation / 120-second overtime training and evaluation artifacts.
It does not claim a fresh full-suite run or numerical training parity.

## Implementation and acceptance mapping

The authoritative design remains [#49](https://github.com/EvanZ/basketworld/issues/49).
Numbering below follows its 23 main acceptance criteria (A1–A23) and seven
additional approved-decision checks (D1–D7).

- **Possession parity and reversible configuration (A1, A3, A5):**
  [#63](https://github.com/EvanZ/basketworld/issues/63#issuecomment-6218109019)
  records the retained possession phase/overtime tests and clock-setting
  isolation. #64/#66 cover CLI, serialization, switching, and compatibility.
  The user additionally confirmed loading and playing an older possession policy.
- **Configuration and propagation (A2, A4, A6, A7):**
  [#64](https://github.com/EvanZ/basketworld/issues/64#issuecomment-6235216935)
  maps fixed 300/120 durations, no clock curriculum, continuous PPO, mode metadata,
  and checkpoints to implementation and tests. The real-run evidence below
  verifies these settings beyond the earlier short-duration integration smoke.
- **Clock ordering, overtime, rewards, and mechanics coverage (A8–A14, A23;
  D5–D6):** #63 covers phase-at-start setup/pickup/inbound timing, inbound catch
  and deadline restarts, final-action precedence, repeated overtime, preserved
  identities/skills/statistics, terminal reward idempotence, buzzer possession
  accounting, and explicit safety truncation. Its recorded known baseline
  failures/partial broad-suite runs remain historical limitations; they are not
  presented here as a full-suite pass.
- **Observation and checkpoint semantics (A15–A16; D1–D3):** #64 covers fixed,
  unclipped seconds/720 and observing-team score/30 semantics, legacy possession
  checkpoints, cross-mode rejection, mid-game round trips, continuous rollout
  state, and final-observation truncation bootstrapping. Actual checkpoint/MLflow
  metadata and saved-checkpoint evaluation provide integration evidence.
- **Diagnostics and paired starts (A17–A18; D4, D7):**
  [#65](https://github.com/EvanZ/basketworld/issues/65#issuecomment-6238268905)
  maps native/deploy/historical evaluation, paired regulation/OT starts independent
  of step count, PPP, elapsed-clock pace, possession duration, regulation segments,
  raw event timestamps, and separate cutoff groups. The small closure fixes below
  preserve truncated-game clock/period coordinates and correct pace wording.
- **Settings and recorded presentation (A19–A22):**
  [#66](https://github.com/EvanZ/basketworld/issues/66#issuecomment-6239744916)
  records mode-aware controls, retained values, save/copy/load/launch plumbing,
  actual-state serialization, live/replay scoreboard rendering, 60 responsive
  layout checks, browser-decoded overtime GIF, and GIF/MP4 route tests in both
  modes. Mode selection configures a new game/run. The gameplay setup screen
  disappears after loading a game; refreshing to return to setup is not an
  in-progress mode conversion and adds no closure requirement.
- **JAX-only prerequisite:**
  [#67](https://github.com/EvanZ/basketworld/issues/67#issuecomment-6234497025)
  records the migrated runtime/dependency boundary. Its historical clock guards
  were subsequently replaced by #64/#65 compiled integration. Its earlier
  full-suite waiver is not a claim of a new full-suite pass here.

## Actual default-duration run

Trainer run `02f412e3-4323-4d26-81fe-8cd8e123293a`, MLflow run
`641fc8c3901345f7b24ac67d02de32a6`, uses 5 players per team,
`episode_end_mode=game_clock`, 300-second regulation and 120-second overtime.
Checkpoint observation metadata explicitly records the 720/30 normalization
scales. MLflow records the same episode mode and durations.

Immutable local evidence under `var/training_app/runs/<trainer-run>/checkpoints/`:

- `update_0014000/metadata.json`, saved `2026-10-20T17:49:23+00:00`;
  SHA-256 `702c2f05c74a5b53dbafeb1942caf809edb93884e6eea9e1bdcb2b9805ba9bb9`.
- `historical_opponent_evaluations/update_0015000.json`;
  SHA-256 `1b3dc19c4b429fedef23616c23b701d4f7bc31c0b4ea02bd5c02f9c71dde3a99`.
  Seventeen matchups completed all **3,400 requested games**, with zero safety
  or horizon cutoffs. There were **77 overtime games**, **83 overtime periods**,
  and raw events reaching period index 2 (second overtime). Each matchup has
  balanced candidate Team A/Team B assignments and separate completed/truncated
  summaries with per-team PPP, pace, possession duration, and scoring segments.
- MLflow deploy metrics at step **15,000**: **1,024 completed games**, zero safety
  or horizon cutoffs, **39 overtime games**, mean **1.1282051282051282** overtime
  periods among overtime games, **312,480 elapsed game-clock seconds**. Team A/B
  PPP: **1.3932161984402893 / 1.304771725883566**; pace:
  **2.0435867895545314 / 2.048195084485407** completed offensive possessions per
  elapsed game-clock minute.

The checkpoint's source provenance is commit
`6fc1324425f78105eafbd35990e5950d90af43e3` with `worktree_dirty=true` and tracked-diff
SHA-256 `361d404b9f2c0a0b79293623c92bb365521458d0bbbee8b7896e4d18079c8f99`.
These results must not be attributed to a clean checkout of that commit. The
review did not modify this run, its artifacts, or its training process.

## Closure fixes and focused verification

Historical summaries now retain the final state's remaining clock and overtime
round for both safety and horizon cutoffs. Completed-game-only diagnostic counters
cannot overwrite those state values with zero. Native evaluation records the
post-step clock and selects each game's actual last active step, including cutoffs,
rather than relying on completed-game counters. Completed and truncated groups
and completed-game win-rate denominators remain separate.

The shared pace description now explicitly includes overtime and check pickup
and excludes check setup and inbounds. The arithmetic is unchanged. Previously
saved reports remain immutable; the corrections apply to newly generated reports.

- **8 passed:** new cutoff-reporting regressions plus existing historical clock,
  training clock, core deploy pace, and truncation-bootstrap checks. Cases include
  safety/horizon cutoffs in OT2 with 70 seconds left, a completed overtime game,
  misleading zero terminal counters, and padded native trace rows.
- **43 passed:** trainer presets, lifecycle metadata/compatibility, replay recording,
  and media routes, including GIF/MP4 in both modes.
- **2 frontend test files passed:** game-clock presentation and trainer configuration
  editing. `git diff --check` passed.
- **1 compiled integration test passed in 340.23 seconds:**
  `tests/test_jax_native_eval.py::test_game_clock_native_and_historical_evaluation_emit_clock_diagnostics`.
  This creates a small temporary 4/2-second clock checkpoint and evaluates it
  through both historical and native paths. It verifies the changed compiled
  trace plumbing; default-duration evidence comes from the real run above.

## Real-checkpoint presentation and export check

Loaded the existing `update_0014000` checkpoint into an isolated CPU
`JaxDevRuntime`, using its saved effective 5v5 environment configuration. Started
a new game with seed 49 and recorded the initial state plus two deterministic
policy steps. Recorded clocks were **300, 299, 298 seconds**, all in `game_clock`
mode. The active user game and trainer were not reset or stepped.

The existing frontend capture harness rendered these actual recorded states
with the real `GameBoard` component. The live view showed **REGULATION 5:00**
and secondary shot clock **24**; replay and decoded GIF/MP4 final frames showed
**REGULATION 4:58** and shot clock **22**, with the same scores and possession
indicator. The real media routes encoded three frames into a **1280×2005 GIF**
and a **1280×2006 H.264 MP4** (40 encoded frames, 20 fps, two seconds). Both
formats were decoded and visually inspected. The harness additionally passed
**68 live/export layout checks**, including explicit setup/inbound/OT fixtures
at full and compact widths.

This is a short real-policy replay excerpt, not a newly simulated full game or a
full-duration video. Actual default-duration completion and repeated overtime
are established by the real evaluation artifacts; phase-boundary correctness
is established by the recorded kernel regressions. The export harness displayed
a missing logo image in encoded frames even though the live logo loaded; this
is outside the clock-mode acceptance check and no claim of a completely clean
branding/export visual audit is made.

Local reproducible evidence is retained under
`artifacts/audits/game-clock-2026-10-20/`: recorded states, live/replay screenshots,
decoded final frames, GIF, MP4, and `verification.json` with asset SHA-256 hashes.
Historical artifacts were not rewritten.
