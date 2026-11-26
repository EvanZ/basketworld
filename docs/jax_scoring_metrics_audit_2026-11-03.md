# JAX scoring metrics audit

Date: 2026-11-03  
Status: Implemented in the working tree on 2026-11-15; verification recorded below
Scope: JAX training, role-specific evaluation, deploy evaluation, historical evaluation, PFSP, native evaluation artifacts, and the trainer views derived from them

## Executive summary

The current scoring metrics mix four different axes:

1. The current learner versus its sampled training opponent.
2. Two persistent internal roster slots.
3. Which team currently has possession.
4. Which team received the opening possession.

In the original single-possession environment these axes mostly coincided. They no longer coincide in a multi-possession game. The current names still assume that they do.

There are also calculation problems. In particular, several training metrics divide points accumulated across every active rollout fragment by only the number of completed games. Points from ongoing and truncated games can therefore enter the numerator without a corresponding game in the denominator.

The main recommendation is to replace the current training score charts with a small learner-centered set calculated from completed games, retain possession-normalized PPP, and move roster-slot and period splits to explicitly named diagnostics. Historical evaluation, PFSP PPP, and expected shot value are substantially better defined and should mostly remain.

## Recommendation labels

- **Keep unchanged**: the name, calculation, and presentation are useful as they are.
- **Keep changed**: the underlying information is useful, but the name, denominator, grouping, or presentation should change.
- **Remove**: stop producing or displaying the metric because it is redundant, misleading, or not decision-useful.
- **Artifact only**: preserve it in detailed evaluation output for debugging, but do not create a normal MLflow chart or trainer headline from it.

## Terms used in this audit

### Learner and opponent

The **learner** is the policy currently being updated. The **opponent** is the sampled frozen policy used for that game. A score difference between them measures learner-versus-pool performance; it does not by itself measure an offense/defense or starting-possession advantage.

### Internal roster slots

The environment has two persistent player rosters so identities, skills, scores, and statistics survive possession changes. The code historically calls these rosters `offense`/Team A and `defense`/Team B. In a multi-possession game, both rosters play offense and defense.

The two training rollout lanes currently called `offense` and `defense` mean:

- `offense` lane: attach the learner policy to roster slot A.
- `defense` lane: attach the learner policy to roster slot B.

They do **not** mean that the learner only plays that basketball role, and they do **not** identify which team received the opening possession. These lane names should never be presented as basketball outcomes.

### Actual possession role

`trajectory.training_role > 0` means the learner is currently attacking. `trajectory.training_role < 0` means the learner is currently defending. This is the correct source for possession-level offense and defense attribution.

### Opening possession

In multi-possession mode the opening team is sampled independently. Opening-possession effects must be grouped using the recorded starting team, not the rollout-lane name.

### Points, PPP, win rate, and win score

- **Points** are actual scoreboard points, including points awarded for violations.
- **PPP scored** is learner points divided by completed learner offensive possessions.
- **PPP allowed** is opponent points divided by completed opponent offensive possessions.
- **Win rate** is wins divided by completed games; ties are not wins.
- **Win score** is `(wins + 0.5 * ties) / completed games`.
- **Expected shot value (ESV)** is shot value multiplied by make probability for a hypothetical shot. It is not actual points, PPP, reward, or a value-function estimate.

## 1. Training metrics: normal `jax/train` profile

### 1.1 Current points per completed episode

| Current metrics | Precise current meaning | Problem | Recommendation | Replacement |
| --- | --- | --- | --- | --- |
| `offense_learner_points_per_completed_episode` | All points scored by the learner while it occupies roster slot A during active steps in the current rollout, divided by completed non-truncated games observed in that lane. | `offense` means roster assignment, not offense or starter. The numerator includes partial and truncated games while the denominator includes only completed games. | **Remove** | `learner_points_scored_per_completed_game`, calculated from terminal scores of completed games across both lanes. |
| `offense_opponent_points_per_completed_episode` | All points scored by the opponent occupying roster slot B in the same lane, divided by completed non-truncated games in that lane. | Same lane-name and denominator mismatch. | **Remove** | `learner_points_allowed_per_completed_game`. |
| `defense_learner_points_per_completed_episode` | All points scored by the learner while it occupies roster slot B during active steps, divided by completed non-truncated games in that lane. | It does not mean learner defense and does not mean the learner started on defense. It also has the numerator/denominator mismatch. | **Remove** | Combined learner scoring metrics plus a separately named roster-symmetry diagnostic if needed. |
| `defense_opponent_points_per_completed_episode` | All points scored by the opponent occupying roster slot A in that lane, divided by completed non-truncated games. | It does not mean points allowed by learner defense in defense-start games. It also has the numerator/denominator mismatch. | **Remove** | `learner_points_allowed_per_completed_game`. |

The four metrics above are the immediate source of the current confusion. Their arithmetic can be reproduced from the rollout, but their names do not describe their meaning and their per-game denominators are not valid.

### 1.2 Current training offensive-efficiency metrics

| Current metrics | Precise current meaning | Problem | Recommendation | Replacement |
| --- | --- | --- | --- | --- |
| `offense_learner_offensive_efficiency` | Learner points in roster-A rollouts divided by completed possessions during which that learner was actually attacking. | The ratio itself is a meaningful observed-rollout PPP, but the lane prefix is an internal assignment. It includes observed fragments from ongoing/truncated games. | **Keep changed** | Combine both lanes as `learner_points_scored_per_completed_offensive_possession`. State explicitly whether it is observed-rollout or completed-game-only. |
| `offense_opponent_offensive_efficiency` | Opponent points in roster-A learner rollouts divided by completed opponent offensive possessions. | Meaningful PPP with an internal lane prefix. | **Keep changed** | Combine both lanes as `learner_points_allowed_per_completed_defensive_possession`. |
| `defense_learner_offensive_efficiency` | Learner points in roster-B rollouts divided by completed learner offensive possessions. | Meaningful PPP with an internal lane prefix. | **Keep changed** | Merge into the learner PPP metric above. |
| `defense_opponent_offensive_efficiency` | Opponent points in roster-B learner rollouts divided by completed opponent offensive possessions. | Meaningful PPP with an internal lane prefix. | **Keep changed** | Merge into the learner PPP-allowed metric above. |

For a combined ratio, sum points and possessions first, then divide. Do not average the two lane-level PPP values.

### 1.3 Regulation-segment and overtime score metrics

Current core metrics include all of these for both rollout lanes:

```text
{lane}_regulation_segment_{1..4}_team_a_points
{lane}_regulation_segment_{1..4}_team_b_points
{lane}_regulation_segment_{1..4}_point_differential
{lane}_overtime_team_a_points
{lane}_overtime_team_b_points
{lane}_overtime_point_differential
```

These are raw point totals from active rollout steps. They are not normalized by games, possessions, or elapsed time; they include partial games; `team_a`/`team_b` are internal roster slots; and `point_differential` is always A minus B even when the learner is B.

Recommendation: **remove from the core MLflow profile**. Preserve period totals in detailed artifacts only. If a period trend becomes useful, expose completed-game learner-versus-opponent rates such as `learner_points_per_completed_game_regulation_segment_1`, with a completed-game count beside it.

### 1.4 Reward metrics that resemble scoring metrics

```text
offense_learner_mean_reward
offense_opponent_mean_reward
defense_learner_mean_reward
defense_opponent_mean_reward
mean_reward
game_reward_mean
```

These are policy-learning rewards, not basketball points. They can contain scoring reward, winner reward, shaping, assistance bonuses, violation terms, or other configured components.

Recommendation: **keep unchanged in an RL-health section**, but remove them from any scoring or basketball-performance summary. The trainer UI should label them as reward, never score or efficiency.

## 2. Training metrics: full/debug profile

The full profile emits several additional score families.

### 2.1 Raw stable-roster totals

```text
team_a_score_delta_total
team_b_score_delta_total
{lane}_team_a_score_delta_total
{lane}_team_b_score_delta_total
```

These are sums of scoreboard changes over active rollout steps. They scale with batch size, horizon, and active-step count. In multi-possession mode the historical `offense` score field is roster A and the historical `defense` score field is roster B.

Recommendation: **artifact only**. Rename to `roster_slot_a_points_observed` and `roster_slot_b_points_observed` if retained. They are useful for accounting invariants, not model comparison.

### 2.2 Lane-oriented point totals and rates

For each lane, the full profile emits:

```text
{lane}_offense_points_total
{lane}_defense_points_total
{lane}_offense_points_per_completed_episode
{lane}_defense_points_per_completed_episode
{lane}_learner_points_total
{lane}_opponent_points_total
{lane}_learner_points_per_completed_episode
{lane}_opponent_points_per_completed_episode
```

`offense` and `defense` inside these names refer to the historical persistent roster fields, not the team currently possessing the ball. The `learner` and `opponent` totals correctly remap those roster fields for each lane, but the lane prefix remains internal. Every `per_completed_episode` value uses the mismatched all-active-points/completed-games denominator described above.

Recommendation:

- **Remove** every `*_points_per_completed_episode` variant.
- **Artifact only** for raw totals, with explicit roster or learner/opponent names.
- Replace normal charts with completed-game learner metrics and possession-normalized learner PPP.

### 2.3 PPO-eligible score families

For each lane, `summarize_ppo_eligible_episode_metrics` emits these patterns:

```text
{lane}_ppo_eligible_{offense|defense}_points_{total|per_step|per_completed_episode}
{lane}_ppo_eligible_{learner|opponent}_points_{total|per_step|per_completed_episode}
```

These describe score events on PPO-eligible steps, not complete game outcomes. The `per_completed_episode` variants again combine a step-filtered numerator with a completed-game denominator.

Recommendation:

- **Remove** all PPO-eligible `per_completed_episode` score metrics.
- **Artifact only** for totals and per-step rates when diagnosing PPO sample selection.
- Do not present PPO-eligibility metrics as basketball performance.

## 3. PFSP opponent metrics

PFSP metrics are split by opponent identity and opponent action mode.

### 3.1 PFSP outcome estimates

| Current metric | Precise definition | Recommendation |
| --- | --- | --- |
| `pfsp/w/{opponent}/{mode}` | A recency-weighted completed-game win score with win = 1, tie = 0.5, loss = 0; 500-update half-life; effective evidence capped at 256 games; Beta(1,1) prior. It is the quantity used by PFSP sampling. | **Keep unchanged**. Continue calling it `w`, not win rate or Elo. |
| `pfsp/win_rate/{opponent}/{mode}` | Lifetime learner wins divided by tracked completed games. Ties count as non-wins; truncations are excluded. No decay and no prior. | **Keep unchanged**. UI label `Win %` is accurate. |
| `pfsp/wins`, `pfsp/win_rate_games` | Lifetime win numerator and completed-game denominator for the displayed win rate. | **Keep unchanged**. These denominators make the rate auditable. |

### 3.2 PFSP cumulative PPP

```text
pfsp/ppp_points_scored/{opponent}/{mode}
pfsp/ppp_points_allowed/{opponent}/{mode}
pfsp/ppp_offensive_possessions/{opponent}/{mode}
pfsp/ppp_defensive_possessions/{opponent}/{mode}
pfsp/ppp_scored/{opponent}/{mode}
pfsp/ppp_allowed/{opponent}/{mode}
```

These are cumulative observed-rollout counters since PPP tracking began. `ppp_scored` is learner points divided by learner completed offensive possessions. `ppp_allowed` is opponent points divided by opponent completed offensive possessions. Points include violation awards. Ongoing and truncated games contribute observed points and completed possessions. There is no recency decay and these values do not affect PFSP sampling.

Recommendation: **keep unchanged**. The raw point and possession counters should remain because they make the ratios auditable and allow modes to be combined correctly.

### 3.3 PFSP current-update PPP

```text
pfsp/current_ppp_points_scored/{opponent}/{mode}
pfsp/current_ppp_points_allowed/{opponent}/{mode}
pfsp/current_ppp_offensive_possessions/{opponent}/{mode}
pfsp/current_ppp_defensive_possessions/{opponent}/{mode}
pfsp/current_ppp_scored/{opponent}/{mode}
pfsp/current_ppp_allowed/{opponent}/{mode}
```

These have the same definitions as the cumulative PFSP PPP metrics but use only the current training update.

Recommendation: **keep unchanged**. This is the correct source for the current-update opponent chart. Continue showing gaps rather than zero when there are no completed possessions.

## 4. Role-specific evaluation: `jax/eval_offense` and `jax/eval_defense`

Each evaluation lane currently logs:

```text
mean_final_offense_score
mean_final_defense_score
mean_final_score_margin
```

In multi-possession mode, `offense_score` is the final score of persistent roster slot A and `defense_score` is the final score of persistent roster slot B. The outer prefix determines which slot contains the learner. The mean includes every evaluation environment at the horizon, including incomplete/truncated games.

Consequences:

- `jax/eval_offense/mean_final_offense_score` happens to be learner score because the learner is in slot A.
- `jax/eval_offense/mean_final_defense_score` is opponent score.
- `jax/eval_defense/mean_final_defense_score` is learner score because the learner is in slot B.
- `jax/eval_defense/mean_final_offense_score` is opponent score.
- `mean_final_score_margin` is always slot A minus slot B, so its learner-facing sign flips between lanes.

Recommendation: **remove all three current metrics**. Replace them with completed-game, learner-oriented metrics aggregated across both lane assignments. Evaluation should also report the completed-game denominator and truncation count.

## 5. Deploy evaluation: `jax/eval_deploy`

Deploy evaluation currently runs the same policy on both teams in sampled action mode. It is therefore a self-play behavior evaluation, not a direct measure of strength against the training pool.

### 5.0 Does deploy evaluation record points per episode?

Not under an explicit or reliable `points_per_episode` metric. The closest fields are `mean_offense_score` and `mean_defense_score`. They average the two persistent roster-slot final scores over every deploy environment. When every deploy game completes, those values happen to equal roster-slot points per episode. If any game is truncated, they silently become a mixture of completed-game scores and partial-game scores.

Therefore the underlying final-score data exists, but the current metric contract does **not** provide an explicit completed-game points-per-episode measure. It also does not provide a learner-versus-opponent split because deploy evaluation uses the same policy on both sides.

This distinction matters: the gap is not “no score data at all”; it is “no clearly named, completion-safe scoring outcome.” The replacement must be calculated only from completed-game final scores and must publish its completed-game denominator.

### 5.1 Current mean score metrics

| Current metric | Precise current meaning | Recommendation |
| --- | --- | --- |
| `mean_offense_score` | Mean final score of roster slot A over all deploy environments, including incomplete/truncated games. | **Remove**. The name is stale and the completion groups are mixed. |
| `mean_defense_score` | Mean final score of roster slot B over all deploy environments, including incomplete/truncated games. | **Remove**. |
| `mean_score_margin` | Mean roster-slot-A score minus roster-slot-B score over all deploy environments. | **Keep changed** as a symmetry diagnostic calculated on completed games and named `roster_assignment_score_margin_per_completed_game`; do not use it as a performance headline. |

Add explicit completed-game scoring fields:

```text
self_play_roster_slot_a_points_per_completed_game
self_play_roster_slot_b_points_per_completed_game
self_play_points_per_team_per_completed_game
self_play_combined_points_per_completed_game
```

The first two are symmetry diagnostics. `self_play_points_per_team_per_completed_game` is `(slot A points + slot B points) / (2 * completed games)`. `self_play_combined_points_per_completed_game` is total points by both teams divided by completed games.

### 5.2 Deploy clock efficiency

```text
clock_team_a_offensive_efficiency_points_per_possession
clock_team_b_offensive_efficiency_points_per_possession
```

These use only completed deploy games. Each is the corresponding stable roster's final points divided by that roster's completed offensive possessions.

The ratios are arithmetically valid, but Team A/Team B are implementation slots and the same policy controls both sides.

Recommendation: **keep changed**:

- Add one primary `self_play_points_per_completed_offensive_possession`, calculated as total points across both sides divided by total completed offensive possessions across both sides.
- Retain the two slot-specific values only under a symmetry-diagnostics section with explicit `roster_slot_a`/`roster_slot_b` names.

### 5.3 Deploy period and overtime scoring

```text
clock_regulation_segment_{1..4}_team_a_points
clock_regulation_segment_{1..4}_team_b_points
clock_regulation_segment_{1..4}_team_a_point_differential
clock_overtime_team_a_points
clock_overtime_team_b_points
clock_overtime_team_a_point_differential
```

These are raw totals over completed deploy games. They grow with the evaluation batch count and are not useful as ordinary time series.

Recommendation: **artifact only**. If period scoring is promoted later, normalize it per completed game and use self-play or learner/opponent terminology appropriate to that evaluator.

## 6. Historical checkpoint evaluation

Historical evaluation is substantially cleaner because it explicitly treats one checkpoint as candidate, one as opponent, balances persistent roster assignments, records opening possession, and excludes truncations from completed-game outcomes.

### 6.1 MLflow historical outcome metrics

```text
jax/eval_historical/w_eval/{opponent}/deterministic
jax/eval_historical/completed_games/{opponent}/deterministic
jax/eval_historical/truncated_games/{opponent}/deterministic
jax/eval_historical/wins/{opponent}/deterministic
jax/eval_historical/ties/{opponent}/deterministic
jax/eval_historical/losses/{opponent}/deterministic
```

`w_eval` is `(wins + 0.5 * ties) / completed games`. Truncated games are excluded.

Recommendation: **keep unchanged**. This is an auditable direct comparison between two trained checkpoints.

### 6.2 Historical artifact scoring summaries

| Current fields | Precise definition | Recommendation |
| --- | --- | --- |
| `mean_candidate_score`, `mean_opponent_score` | Mean final scoreboard points over completed games only. | **Keep unchanged**. |
| `mean_candidate_point_differential` | Mean candidate score minus opponent score over completed games. | **Keep unchanged**. |
| `candidate_win_rate`, `candidate_tie_rate`, `candidate_loss_rate` | Outcome counts divided by completed games. | **Keep unchanged**. |
| `w_eval` | Win score with ties worth one half. | **Keep unchanged**. |
| `candidate_points_per_completed_possession`, `opponent_points_per_completed_possession` | Completed-game points divided by the corresponding side's completed offensive possessions. | **Keep changed** only in naming: use `*_points_per_completed_offensive_possession` to state the denominator explicitly. |
| `mean_combined_score` | Candidate plus opponent points divided by completed games. | **Keep unchanged** as a pace/style diagnostic, not a strength metric. |
| `scoreline_histogram`, `point_differential_histogram` | Counts of completed-game final scorelines and candidate margins. | **Artifact only**. |

### 6.3 Historical assignment and starter splits

The artifact already reports outcome summaries for:

- candidate assigned to roster slot A;
- candidate assigned to roster slot B;
- candidate received opening possession;
- opponent received opening possession.

Recommendation: **keep unchanged in the artifact** and promote a small subset as fairness diagnostics:

- completed-game count in each split;
- mean candidate margin in each split;
- win score in each split;
- assignment-margin delta and opening-possession-margin delta.

These are the correct diagnostics for determining whether an apparent bias comes from internal roster assignment or opening possession.

### 6.4 Historical clock metrics

`clock_metrics_completed_games` contains candidate/opponent PPP, pace, regulation-segment scoring, and overtime scoring for completed games. `clock_metrics_truncated_games` describes cutoffs separately.

Recommendation: **keep unchanged in the artifact**. Promote only completed-game candidate/opponent PPP and their denominators. Truncated scoring should remain diagnostic and must never be blended into completed-game averages.

## 7. Native evaluation summary

### 7.1 Correct identity-oriented scoring fields

```text
completed_user_score_mean
completed_opponent_score_mean
completed_user_score_total
completed_opponent_score_total
completed_margin_mean
user_points_per_possession
opponent_points_per_possession
win_count
loss_count
tie_count
completed_games
```

These use stable user/opponent identity and completed games. The PPP denominators are each side's completed offensive possessions.

Recommendation: **keep unchanged**, with two additions derived from existing counts:

- `completed_win_rate = win_count / completed_games`
- `completed_win_score = (win_count + 0.5 * tie_count) / completed_games`

For consistency, rename the two PPP fields to `user_points_per_completed_offensive_possession` and `opponent_points_per_completed_offensive_possession` in a versioned schema, while keeping compatibility readers for older artifacts.

### 7.2 Stale role-oriented native fields

```text
team_a_score_delta_per_episode
team_b_score_delta_per_episode
offense_points_per_episode
defense_points_per_episode
score_margin_per_episode
total_offense_points
total_defense_points
```

The `offense` and `defense` fields are persistent roster-slot scoreboard deltas in multi-possession games. These values also average or total all evaluation episodes rather than completed games only. They duplicate the clearer user/opponent completed-score fields.

Recommendation: **remove** from new native-summary schemas. Preserve old artifact readers but do not display these fields.

### 7.3 Native clock stable-team fields

The nested clock summary preserves `team_a` and `team_b` for reproducible low-level accounting and also maps them to `user` and `opponent`.

Recommendation: **keep unchanged in artifacts**. User-facing displays should use only the `user` and `opponent` views. Stable-team fields are useful for symmetry tests but should not be general metrics.

### 7.4 Post-offensive-rebound scoring

```text
post_orb_points_total
post_orb_points_per_sample
post_orb_sample_count
```

`post_orb_points_per_sample` is future actual points later in the same possession after an offensive rebound, averaged over offensive-rebound continuation samples. Multiple offensive rebounds in one possession create multiple samples.

Recommendation: **keep unchanged in detailed native evaluation**. Always display the sample count. Do not mix this with ordinary PPP because its denominator is rebound continuation states, not possessions.

## 8. Expected shot value and defensive pressure

For both `learner_attacking` and `learner_defending`, the system reports:

```text
mean_ball_handler_uncontested_esv
mean_ball_handler_defended_esv
mean_ball_handler_esv_reduction
mean_offense_uncontested_esv
mean_offense_defended_esv
mean_offense_esv_reduction
```

Definitions:

- `uncontested_esv`: shot value times the make probability with the contest effect removed, holding location and shooting skill fixed.
- `defended_esv`: shot value times the actual contest-adjusted make probability.
- `esv_reduction`: uncontested ESV minus defended ESV on the same samples.
- `ball_handler`: only live states with an on-court offensive ball handler.
- `offense`: first average over on-court offensive players within each live state, then average across states.

Recommendation: **keep unchanged**. These metrics use actual current-possession roles and directly measure the shot-value suppression applied or received by the learner. Keep their sample counts and continue labeling them ESV rather than expected points or PPP.

## 9. Elo and other outcome-strength metrics

Historical Elo is fitted from completed deterministic cross-checkpoint outcomes. It is a relative rating within connected matchup components; it is not basketball points and is not PFSP `w`.

```text
elo_rating
elo_delta
rating_ci95_low
rating_ci95_high
win_score
mean_point_differential
```

Recommendation: **keep unchanged**. Continue showing completed-game counts, confidence intervals, connected-component warnings, and evaluation-rule warnings. `mean_point_differential` is useful supporting evidence but should not replace the fitted outcome rating.

## 10. Proposed canonical scoring metrics

### 10.1 Training core

The normal training profile should contain only these basketball scoring outcomes:

```text
learner_points_scored_per_completed_game
learner_points_allowed_per_completed_game
learner_score_margin_per_completed_game
learner_points_scored_per_completed_offensive_possession
learner_points_allowed_per_completed_defensive_possession
learner_completed_game_win_rate
learner_completed_game_win_score
completed_game_count
truncated_game_count
```

All score-per-game numerators must come from terminal scores of the same completed games used in the denominator. All PPP values must be calculated by summing points and completed offensive possessions before division.

Because current trajectories do not expose all terminal score and opening-possession fields needed for a clean host summary, implementing these metrics should add explicit terminal learner score, terminal opponent score, and learner-started-with-possession values to the rollout diagnostics rather than reconstructing them from ambiguous aliases.

### 10.2 Training fairness diagnostics

Keep these outside the primary score panel:

```text
learner_roster_assignment_a_completed_game_count
learner_roster_assignment_a_score_margin_per_completed_game
learner_roster_assignment_b_completed_game_count
learner_roster_assignment_b_score_margin_per_completed_game
roster_assignment_score_margin_delta
learner_started_with_possession_completed_game_count
learner_started_with_possession_score_margin_per_completed_game
opponent_started_with_possession_completed_game_count
opponent_started_with_possession_score_margin_per_completed_game
opening_possession_score_margin_delta
```

The words offense and defense should not appear in roster-assignment metric names.

### 10.3 Deploy self-play core

Because deploy evaluation currently uses the same policy on both teams, its primary scoring metrics should describe self-play behavior:

```text
self_play_points_per_team_per_completed_game
self_play_combined_points_per_completed_game
self_play_points_per_completed_offensive_possession
completed_game_count
truncated_game_count
```

Roster-slot margin and slot-specific PPP belong in symmetry diagnostics, not the primary panel.

### 10.4 Direct strength evaluation

Use historical or cross-policy evaluation—not same-policy deploy—for strength:

```text
candidate_points_per_completed_game
opponent_points_per_completed_game
candidate_score_margin_per_completed_game
candidate_points_per_completed_offensive_possession
opponent_points_per_completed_offensive_possession
candidate_win_rate
w_eval
completed_game_count
truncated_game_count
```

## 11. Migration and compatibility recommendation

1. Add a scoring-metric schema version to training status, checkpoints, MLflow parameters, and evaluation artifacts.
2. Stop writing the misleading keys for new updates; do not silently change their meaning under the same names.
3. Preserve existing MLflow history and old artifact readers. Label old charts as legacy where they remain visible.
4. Build new values from additive totals and explicit denominators, then derive ratios once on the host.
5. Never average precomputed lane, mode, or opponent PPP ratios.
6. Keep completed and truncated games separate.
7. Keep internal roster-slot fields in artifacts for accounting and symmetry tests, but not in the normal metric picker.
8. Update the checkpoint summary so it prints the canonical learner metrics rather than selected lane aliases.

## 12. Required verification before implementation is considered complete

1. **Accounting invariant:** learner points plus opponent points equal the corresponding scoreboard totals for every completed game.
2. **Denominator invariant:** every per-completed-game numerator contains only those same completed games.
3. **PPP invariant:** combined PPP equals combined raw points divided by combined completed offensive possessions, never a mean of subgroup PPP values.
4. **Role-swap test:** swapping the learner between roster assignments swaps assignment diagnostics but does not change combined learner metrics.
5. **Starter-swap test:** paired games with inverted opening possession populate the correct starter split.
6. **Self-play symmetry test:** an identical policy on both roster assignments has no systematic assignment margin beyond sampling uncertainty.
7. **Truncation test:** adding a truncated high-scoring partial game does not change completed-game scoring metrics.
8. **Violation-point test:** awarded violation points enter the correct team's score and PPP numerator.
9. **Resume test:** a paused run can resume under the new schema without rewriting its earlier MLflow history.
10. **UI test:** primary panels contain only learner/opponent or self-play names; roster-slot diagnostics are clearly separated.

## Code locations inspected

- `basketworld_jax/train/main.py`: core/full metric filters, role rollout summaries, role evaluation, historical evaluation, MLflow logging, and checkpoint summary.
- `basketworld_jax/train/runtime.py`: training-step summaries, deploy-evaluation summaries, point totals, and clock metrics.
- `basketworld_jax/train/pfsp.py`: PFSP outcome evidence and current/cumulative PPP.
- `basketworld_jax/train/pressure_metrics.py`: expected shot value and pressure diagnostics.
- `basketworld_jax/eval/clock_metrics.py`: shared PPP, pace, period, and overtime denominators.
- `basketworld_jax/eval/native.py`: native evaluation scoring, completed-game identity mapping, PPP, and post-rebound continuation points.
- `app/training_backend/historical_elo.py`: trainer Elo and empirical historical win scores.
- `app/training_frontend/src/pfsp.js` and `OpponentUsageChart.vue`: PFSP and PPP presentation semantics.


## Implementation notes (2026-11-15)

The recommendations above are implemented as scoring metric schema 2. Training
PPP is explicitly observed-rollout; completed-game score metrics instead read
new terminal learner/opponent scoreboard diagnostics captured before reset.
Zero-denominator ratios are unavailable, not zero. Combined role evaluation is
published under `jax/eval`, and deploy scoring is explicitly same-policy
self-play. Fairness diagnostics use the actual roster and starting-team fields.

Raw roster, period, and PPO-selected scoring counters are preserved under
`scoring_diagnostics` in detailed results, excluded from both MLflow profiles.
Retired per-game aliases are no longer written. Historical and native PPP names
now explicitly identify completed offensive possessions; compatibility readers
retain support for pre-v2 artifacts. PFSP, ESV, Elo, and reward formulas remain
unchanged. No existing run history or evaluation artifact is rewritten.

See `docs/site/jax/training.md` for the reporting contract. The implementation
uses an additive schema field, a per-update schema metric, and the versioned
MLflow parameter `jax/scoring_metric_schema/v2` so old runs can resume without
changing an existing MLflow parameter.

### Verification

- Focused Python checks passed for terminal-score accounting, unequal possession
  denominators, roster/starter mapping, high-scoring truncations, real clearance
  violation awards, core/full MLflow filtering, historical summaries, and artifact
  compatibility. The main focused selection passed 34 tests; final filter and
  legacy-pickle checks also passed.
- Training with role/deploy evaluation, grouped opponents, and checkpoint resume
  passed. A separate resume check removed the scoring schema from a temporary
  checkpoint and confirmed that the next update emits schema 2. Legacy MLflow
  history preservation is covered with an existing-run logging fixture.
- Native evaluation passed its paired-starter and completion/cutoff accounting
  scenario. The final migration/native rerun passed both tests, including the
  renamed PPP fields and unavailable zero-denominator rates.
- All 17 dev-app and 7 trainer-app frontend test files passed, and both production
  builds passed. A live trainer run-detail check verified the learner-scoring
  section and opt-in legacy metric discovery.
- This is focused regression coverage, not a fresh full repository suite or a
  long-run statistical study of self-play assignment bias. Symmetry accounting
  is checked with balanced paired scores; native starter coverage is exercised
  with an actual identical-policy evaluation.
