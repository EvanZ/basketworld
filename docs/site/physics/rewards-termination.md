# Rewards and termination

BasketWorld constructs a zero-sum vector over all players, then aggregates the
controlled team's entries into the scalar PPO reward. Task reward, potential
shaping, and optional intent bonuses are conceptually separate even when they
are combined for optimization.

## Zero-sum allocation

For an offense-valued event \(R\) in \(N\)-on-\(N\), each offense player
receives \(R/N\) and each defender receives \(-R/N\). Summing the controlled
team therefore returns \(R\) from the offense perspective and \(-R\) from the
defense perspective.

The JAX helper applies `training_player_mask`, sums the selected player
entries, and multiplies by `task_reward_scale`.

## Task-reward components

### Completed pass

A successful pass contributes `pass_reward`. Interceptions and invalid pass
slots terminate but do not have a separate JAX turnover-penalty term.

### Shot

With rebounds disabled, every shot terminates and the shot reward is its
pressure-adjusted expected points, independent of the sampled make result.

With rebounds enabled, `rebound_terminal_reward_mode` controls the task value:

| Mode | Made shot | Miss + defensive rebound | Miss + offensive rebound |
|---|---:|---:|---:|
| `actual_points` | actual 2 or 3 | 0 | 0 |
| `last_shot_ep_on_defensive_rebound` | actual 2 or 3 | shot EP | 0 |
| `last_shot_ep` | shot EP | shot EP | 0 |

`actual_points` is the direct outcome objective. `last_shot_ep` is useful when
the terminal shot should receive a lower-variance expected-points target.

### Assist shaping

If a valid pass recipient attempts a shot before the assist window expires:

- potential-assist reward is
  `potential_assist_pct * shot_expected_points`;
- a make adds
  `full_assist_bonus_pct * shot_expected_points`.

The first component applies on the attempt; the second applies only on a make.

### Defensive lane violation

A defensive lane violation adds `violation_reward` to the offense perspective,
increments the offense score by one, and terminates the episode.

## Rebound reward redistribution

When enabled, an offensive rebound can pay
`offensive_rebound_reward_advance`. The amount is tracked in state and
subtracted when the possession eventually terminates:

\[
R_{\text{settlement}} =
-R_{\text{advances paid}}.
\]

`rebound_reward_once_per_possession` restricts the advance to the first
offensive rebound. Without redistribution, rebounds themselves have zero task
reward.

## Potential-based shaping

The potential \(\Phi(s)\) is derived from current pressure-adjusted shot
quality. Depending on `phi_aggregation_mode`, it can use the best, worst, or
average expected points over the possession team or over non-holder teammates.
`phi_blend_weight` blends that aggregate with the ball handler's EP, and
`phi_use_ball_handler_only` selects only the handler.

For transition \(s\rightarrow s'\):

\[
r_\Phi =
\beta\left(\gamma\Phi(s')-\Phi(s)\right).
\]

Terminal states force \(\Phi(s')=0\). An offensive rebound is non-terminal, so
its next-state potential is retained.

The shaped value is divided across the offense and negated for defense before
team aggregation. `phi_beta` can be scheduled during training; when shaping is
disabled, all phi diagnostics and contributions are zero.

## Training-time scaling and intent bonus

The outer training loop can schedule task-reward scale across PPO updates.
This scales the collected environment reward before optimization.

When intent diversity is enabled, a separately computed discriminator bonus is
added to the relevant rollout rewards. Returns and advantages are then built
from the combined reward. Metrics retain task, phi, and intent components so
research runs can distinguish behavior changes from reward rescaling.

## Terminal matrix

| Event | Terminal? |
|---|---|
| Defender-pressure turnover | Yes |
| Pass interception or invalid slot | Yes |
| Ball-handler move out of bounds | Yes |
| Shot-clock expiration without a shot | Yes |
| Offensive three seconds | Yes |
| Defensive lane violation | Yes |
| Made shot | Yes |
| Miss, rebounds disabled | Yes |
| Defensive rebound | Yes |
| Offensive rebound | No |

The rollout runner may reset a terminal row for continued batch utilization,
but the terminal flag remains part of the trajectory and cuts GAE
bootstrapping across that possession boundary.
