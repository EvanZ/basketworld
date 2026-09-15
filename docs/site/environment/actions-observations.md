# Actions and observations

## Joint action space

The environment receives one discrete action for every player. With \(2N\)
players, the Gymnasium representation is:

```python
spaces.MultiDiscrete([14] * (2 * players_per_side))
```

All choices describe the same simulation interval. They are not executed as
alternating individual turns.

| ID | Action | Meaning |
|---:|---|---|
| 0 | `NOOP` | Stay in place and take no ball action |
| 1 | `MOVE_E` | Move east |
| 2 | `MOVE_NE` | Move north-east |
| 3 | `MOVE_NW` | Move north-west |
| 4 | `MOVE_W` | Move west |
| 5 | `MOVE_SW` | Move south-west |
| 6 | `MOVE_SE` | Move south-east |
| 7 | `SHOOT` | Shoot; legal only for the ball holder |
| 8 | `PASS_E` | Pass slot 0 |
| 9 | `PASS_NE` | Pass slot 1 |
| 10 | `PASS_NW` | Pass slot 2 |
| 11 | `PASS_W` | Pass slot 3 |
| 12 | `PASS_SW` | Pass slot 4 |
| 13 | `PASS_SE` | Pass slot 5 |

The historical direction names remain in the enum. In the primary
`pointer_targeted` mode, IDs 8–13 are fixed teammate slots rather than
geometric directions. For each passer, teammates are sorted by player ID and
assigned to available slots; unused slots are masked.

## Action masks

Every player always has `NOOP`. Movement is legal when the neighboring axial
cell exists and, if `mask_occupied_moves` is enabled, is not currently
occupied. `SHOOT` and pass slots are legal only for the ball holder.

Masks have shape:

```text
(batch, all_players, 14)
```

The policy receives only the controlled team’s rows. Before sampling, illegal
logits are replaced with a large negative value. If a row somehow has no legal
entry, the policy falls back to `NOOP`.

Masks prevent most invalid actions, but the transition function remains
defensive: it handles out-of-bounds movement, invalid pass slots, and movement
conflicts explicitly.

## Role-conditioned observations

The same policy parameter tree is used from both perspectives. A scalar role
flag disambiguates them:

- offense: \(+1\);
- defense: \(-1\).

The policy chooses the role-specific actor/value path from this flag. Intent
conditioning also selects offense or defense embeddings with it.

## Attention observation

The attention observation is the primary JAX policy representation. It is
transported as one flat array and unpacked by the actor-critic:

```text
[player 0 token, ..., player P-1 token] + [globals] + [observer-role flag]
```

Let `N` be the number of players per side and `P = 2N` the total number of
players. Tokens are ordered by player ID: offense first, then defense.

| Schema | Player width | Global width | Packed dimension |
|---|---:|---:|---:|
| Default | 18 | 7 | `(P × 18) + 7 + 1` |
| Skill-only rebound observations (`--no-rebound-target-observation-features`) | 16 | 4 | `(P × 16) + 4 + 1` |
| `--rebound-win-prob-features` | 19 | 8 | `(P × 19) + 8 + 1` |

For 5-on-5, these dimensions are 188 by default, 165 in skill-only mode, and
199 with the optional rebound probability features.

`--rebound-target-observation-features` defaults to enabled; disable it for the
skill-only ablation. It removes all target-derived policy inputs and the
specialist marker, leaving `rebound_skill` as the only rebound-specific input.
`--rebound-win-prob-features` defaults to disabled. Both switches change the
model input shape and are saved in `ActorCriticSpec`, so neither can be toggled
when loading an existing checkpoint trained with another schema.

### Player token features

Every player token uses this exact order:

| Index | Feature | Range or normalization | Meaning |
|---:|---|---|---|
| 0 | `q_norm` | `q / court_norm_den` | Player axial court coordinate |
| 1 | `r_norm` | `r / court_norm_den` | Player axial court coordinate |
| 2 | `role` | `+1` offense, `-1` defense | Physical team of this player |
| 3 | `has_ball` | 0 or 1 | Ball-holder indicator |
| 4 | `layup_pct` | probability; defense is 0 | Offensive player layup probability |
| 5 | `three_pt_pct` | probability; defense is 0 | Offensive player three-point probability |
| 6 | `dunk_pct` | probability; defense is 0 | Offensive player dunk probability |
| 7 | `lane_steps_norm` | clipped to [0, 1] | Active offensive or defensive lane counter divided by the configured maximum |
| 8 | `expected_points` | points per attempt; defense is 0 | Current pressure-adjusted shot expected points |
| 9 | `turnover_probability` | probability; defense is 0 | Defender-pressure turnover risk for this offensive player |
| 10 | `pass_steal_probability` | probability; defense is 0 | Interception risk if the holder passes to this offensive player |
| 11 | `distance_to_ball` | normalized hex distance | Distance to the ball holder; 0 when there is no holder |
| 12 | `distance_to_best_ep_player` | normalized hex distance | Distance to the offensive player with the highest current expected points |
| 13 | `distance_to_nearest_opponent` | normalized hex distance | Distance to the nearest opposing player |
| 14 | `distance_to_nearest_teammate` | normalized hex distance | Distance to the nearest teammate, excluding self |
| 15 | `distance_to_expected_rebound_target` | normalized hex distance | **Target-observation feature.** Distance to the mean of the current rebound-target distribution |
| 16 | `rebound_skill` | sampled scalar | Per-episode skill used in rebound winner logits; index 15 in skill-only mode |
| 17 | `rebound_skill_specialist` | 0 or 1 | **Target-observation feature.** Specialist marker in `one_high_per_team` mode; otherwise 0 |
| 18 | `rebound_win_probability` | probability | **Optional.** Conditional probability this player wins the rebound if the current holder misses now |

Shooting and passing features are deliberately zeroed on defensive rows.
Rebound skill and rebound probability apply to both teams.

### Global features

The global vector uses this exact order:

| Index | Feature | Range or normalization | Meaning |
|---:|---|---|---|
| 0 | `shot_clock_norm` | current clock / configured maximum | Remaining possession clock |
| 1 | `pressure_exposure` | accumulated probability-like state | Cumulative defender-pressure exposure |
| 2 | `hoop_q_norm` | hoop `q / court_norm_den` | Basket axial coordinate |
| 3 | `hoop_r_norm` | hoop `r / court_norm_den` | Basket axial coordinate |
| 4 | `expected_rebound_target_q` | expected `q / court_norm_den` | **Target-observation feature.** Mean target coordinate under the rebound table |
| 5 | `expected_rebound_target_r` | expected `r / court_norm_den` | **Target-observation feature.** Mean target coordinate under the rebound table |
| 6 | `rebound_target_entropy` | entropy / `log(number_of_cells)` | **Target-observation feature.** Uncertainty of the rebound-target distribution |
| 7 | `offensive_rebound_probability` | probability | **Optional.** Conditional probability any offensive player wins the rebound if the current holder misses now |

### Observer-role flag

The final scalar identifies which team the shared policy controls:

| Value | Observer |
|---:|---|
| `+1` | offense |
| `-1` | defense |

This differs from player feature 2: every observation contains both physical
teams, while the observer flag selects the role-specific policy/value path.
Intent state is supplied separately as policy context and is not inserted into
the player or global arrays.

### Rebound-derived observation semantics

The kernel constructs rebound observations before sampling any future target.
For the current offensive holder it selects the fitted distribution for the
holder shot type and cell, applies uniform mixing and target temperature, and
then computes the expected target and normalized entropy.

`--rebound-obs-top-n-targets 0` uses the full fitted target distribution for
observations. A positive value keeps and renormalizes only the top N targets
for observation features; it does not change the distribution used by the
actual rebound event.

Rebound-derived values are zero when rebounds are disabled, there is no valid
ball holder, the holder is not offensive, or its court cell cannot be
resolved. They never expose the future sampled target or winner.

When `--rebound-win-prob-features` is enabled, the kernel computes the winner
softmax for every possible target using the same distance, basket-position,
skill, winner-temperature, and local/global eligibility rules as the
environment. It then marginalizes over target cells:

```text
player_win_probability[i]
  = sum_target P(target | current shot) × P(player i wins | target)

offensive_rebound_probability
  = sum_i_in_offense player_win_probability[i]
```

When a rebound is available, player probabilities sum to 1 across both teams.
The features are conditional on a miss: they do not encode the chance that the
shot misses. They use positions at observation time; movement selected on the
subsequent simultaneous shot step can change the positions used by the actual
rebound event.

## Flat observation

The MLP baseline uses a larger ordered flat vector. It remains useful for
performance baselines and debugging, while the attention representation is
required by the current pointer-targeted action head and integrated intent
selector.

Its default dimension is `5N² + 19N + 10`. Disabling rebound-target
observations removes `4N + 3` values, producing `5N² + 15N + 7`. Enabling
`--rebound-win-prob-features` adds `2N + 1` values to either schema.

| Order | Feature group | Width |
|---:|---|---:|
| 1 | Normalized player positions | `4N` |
| 2 | Ball-holder one-hot | `2N` |
| 3 | Normalized shot clock | 1 |
| 4 | Pressure exposure | 1 |
| 5 | Player team-role encodings | `2N` |
| 6 | Normalized ball-handler position | 2 |
| 7 | Normalized hoop position | 2 |
| 8 | All offense-defense distances | `N²` |
| 9 | All offense-defense signed angles | `N²` |
| 10 | Unordered within-team distances | `N(N-1)` |
| 11 | Ordered within-team signed angles | `2N(N-1)` |
| 12 | Per-player active lane counters | `2N` |
| 13 | Offensive-player expected points | `N` |
| 14 | Offensive-player turnover probabilities | `N` |
| 15 | Offensive receiver steal probabilities | `N` |
| 16 | **Target-observation feature:** per-player distance to expected rebound target | `2N` |
| 17 | Per-player rebound skill | `2N` |
| 18 | **Target-observation feature:** per-player rebound specialist indicator | `2N` |
| 19 | **Target-observation feature:** expected rebound target `q`, `r`, and entropy | 3 |
| 20 | **Optional:** per-player rebound-win probability | `2N` |
| 21 | **Optional:** offensive-rebound probability | 1 |
| 22 | Observer-role flag | 1 |
| 23 | Offensive layup, three-point, and dunk deltas from base skill | `3N` |

Unlike player token feature 7, the flat lane-counter group contains the raw
active counters rather than values divided by the configured maximum.

## Checkpoint, evaluation, and UI consistency

The selected schema is persisted as `rebound_win_prob_features` and
`rebound_target_observation_features` in `ActorCriticSpec`, and as
`jax/rebound_win_prob_features` and `jax/rebound_target_observation_features`
in MLflow. Training, deploy evaluation, checkpoint inference, and the web
application Observation tab reconstruct the schema from the saved policy
specification.
