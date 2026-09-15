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
