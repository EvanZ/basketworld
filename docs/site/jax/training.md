# RL training

The current trainer implements PPO explicitly with JAX, Flax, and Optax.
Every update collects both offense-role and defense-role experience, combines
the resulting samples, and updates one policy parameter tree.

## One update

For each update index:

1. evaluate entropy, task-reward, phi, selector, and opponent schedules;
2. collect a rollout for offense;
3. collect a rollout for defense;
4. apply optional intent-diversity bonuses;
5. compute GAE and returns for each role;
6. concatenate both role batches;
7. run PPO epochs/minibatches;
8. optionally run a selector-only update;
9. log, evaluate, and checkpoint.

With batch size \(B\) and horizon \(H\), the ordinary combined PPO sample count
is:

\[
M = 2BH.
\]

The factor of two is the offense and defense role collection. Each sample
contains \(N\) controlled-player actions but one scalar team value, reward,
advantage, and return.

## Trajectory contents

`TrajectoryBatch` has leading shape `(H, B, ...)` and includes:

- packed observation and active intent context;
- controlled-team mask, action, and selected log probability;
- full offense-plus-defense joint action;
- scalar value, reward, and terminal flag;
- episode boundaries and active masks;
- pass, shot, turnover, rebound, lane, selector, and intent diagnostics.

The rollout output also carries final observations and bootstrap values.

## Generalized advantage estimation

For reward \(r_t\), value \(V_t\), and terminal flag \(d_t\):

\[
\delta_t =
r_t + \gamma(1-d_t)V_{t+1}-V_t,
\]

\[
A_t =
\delta_t + \gamma\lambda(1-d_t)A_{t+1}.
\]

Returns are:

\[
R_t = A_t + V_t.
\]

The scan runs backward through time. Terminal flags prevent bootstrapping
across possession boundaries. Advantages are normalized over active samples
before flattening into `PPOBatch`.

## PPO objective

The team joint log probability is the sum of selected controlled-player log
probabilities. Let:

\[
r_t(\theta) =
\exp\left[
\log\pi_\theta(\mathbf a_t|s_t)
-
\log\pi_{\text{old}}(\mathbf a_t|s_t)
\right].
\]

The clipped policy loss is:

\[
L_\pi =
-\mathbb E
\left[
\min\left(
r_t A_t,
\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)A_t
\right)
\right].
\]

Value loss is un-clipped mean squared error:

\[
L_V = \mathbb E[(V_\theta(s_t)-R_t)^2].
\]

The total optimized loss is:

\[
L =
L_\pi + c_VL_V - c_H\mathcal H.
\]

The runtime also reports approximate KL, clipping fraction, log-ratio
magnitudes, gradient norm, value diagnostics, and active sample counts.

## Epochs and minibatches

`policy_update_epochs` repeats optimization over the collected PPO batch.
`ppo_minibatches` shuffles sample indices for each epoch and scans over equal
minibatches.

The minibatch count must evenly divide \(2BH\) in training-loop mode. The
parser rejects invalid combinations before compilation. The compiled update
runner also fixes its first observed batch size and rejects later changes.

## Completed-episode masking

`--ppo-completed-episodes-only` limits loss eligibility to episodes that both
start and terminate within the collected rollout. Leading and trailing
fragments are excluded.

This is useful when short possessions and reset frequency could otherwise
produce many partial returns. It reduces sample utilization and may require a
longer horizon.

`--single-episode-rollouts` serves a different purpose: each environment row
stops contributing after its first terminal event instead of being reset
inside the rollout.

## Self-play opponent actions

The simplest compiled runner controls one role with the learned policy and
samples uniform legal actions for the other role. Production training can
replace that opponent with:

- a fixed frozen checkpoint;
- a checkpoint sampled from a historical pool;
- several sampled checkpoints assigned to contiguous environment groups.

The opponent action choice can be stochastic or episode-stable deterministic
argmax. See [Training operations](operations.md).

## Scheduling

The outer loop can schedule:

- entropy coefficient;
- task-reward scale;
- phi beta;
- selector alpha and epsilon;
- intent-diversity beta;
- deterministic-opponent probability;
- checkpoint interval.

Schedule values are calculated once per update and included in metrics and
checkpoint metadata. They are Python orchestration values, while the rollout
and optimizer remain compiled array programs.

## Protected multi-possession checks

For check-based made-basket restarts, new configurations use five
`check_setup_steps` and a five-step `check_deadline_steps` pickup clock. A
positive `check_clearance` is a game rule from update one, not a curriculum
flag. It selects the check-clearance observation schema, so train that policy
from scratch.

The movement constraints and final-boundary violation mechanics are described
in [Movement and rules](../physics/movement-rules.md). Evaluation uses the
shared game rules of the earlier checkpoint when comparing checkpoint pairs;
policy-internal schedules such as the intent selector remain checkpoint-local.

## Scoring metrics (schema 2)

Training score charts combine both learner roster assignments. The normal
`jax/train` basketball metrics are:

- `learner_points_scored_per_completed_game`,
  `learner_points_allowed_per_completed_game`, and
  `learner_score_margin_per_completed_game`: terminal scoreboard totals divided
  by the number of those same completed games, including games spanning updates.
- `learner_completed_game_win_rate`: wins / completed games.
  `learner_completed_game_win_score`: (wins + half the ties) / completed games.
- `learner_points_scored_per_completed_offensive_possession` and
  `learner_points_allowed_per_completed_defensive_possession`: **observed-rollout
  PPP**, including points and completed possessions from ongoing or truncated
  games. Both lanes' numerators and denominators are summed before division.
- `completed_game_count` and `truncated_game_count`: denominators for the update.

Scoreboard points include violation awards. Raw completed-game score totals,
win/tie counts, observed point totals, and possession counts accompany the
rates. A missing denominator produces `null` in artifacts/status and no MLflow
rate sample. It does not mean zero performance. Ongoing games that simply cross
a rollout boundary are not truncations.

Combined role evaluation logs under `jax/eval`: all its score and PPP metrics
use completed games only. `jax/eval_deploy` runs the same policy on both sides;
its `self_play_points_per_team_per_completed_game`,
`self_play_combined_points_per_completed_game`, and
`self_play_points_per_completed_offensive_possession` describe self-play
behavior, not strength against another checkpoint. Historical evaluation
retains completed-game outcomes and adds explicit offensive-possession PPP
names and roster/starter fairness splits.

The trainer groups basketball scoring separately from RL rewards, self-play,
and roster/opening-possession diagnostics. The metric picker hides legacy
scoring and fairness diagnostics unless explicitly included. Existing pinned
legacy charts remain readable under **Legacy scoring (schema 1)**. An internal
roster assignment is not the same as attacking, defending, or opening possession.

Raw roster/period and PPO-selected score totals live in `scoring_diagnostics`
inside detailed training/checkpoint/deploy results. They are excluded from both
core and full MLflow chart profiles. Invalid `*_points_per_completed_episode`
keys and old mean roster-score aliases are no longer emitted. PFSP, ESV,
historical Elo, and learning reward calculations are unchanged.

New status files, checkpoints, and evaluation artifacts record
`scoring_metric_schema_version: 2`. MLflow records
`jax/scoring_metric_schema/v2=true` and a per-update schema metric; the
version-specific parameter allows an existing run to resume without overwriting
immutable parameters or metric history. This is a reporting change: no policy
weights, actions, observation schema, or environment rules are migrated.

### Rim protection experiment

Enable **Rim protection** and **Contested shooting** together in the trainer
configuration editor, or pass `--rim-protection-enabled
--contested-shooting-enabled` when starting a new training run. Both are opt-in;
existing checkpoints retain their trained rules. All thirteen flags/parameters
are saved in the configuration, checkpoint, and MLflow. See
[shooting pressure](../physics/shooting-pressure.md#opt-in-rim-protection-and-contested-shooting-issue-84)
for distributions, defaults, and diagnostic denominators. In the metric picker,
**Shot defense and blocks** groups the new measurements.
