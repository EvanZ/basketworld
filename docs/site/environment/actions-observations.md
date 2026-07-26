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

The current attention observation packs:

```text
[all player tokens flattened] + [globals] + [role flag]
```

For \(2N\) players its dimension is:

\[
(2N \times 18) + 7 + 1.
\]

### Player token: 18 features

| Feature | Interpretation |
|---|---|
| `q`, `r` | Normalized axial position |
| team role | \(+1\) offense, \(-1\) defense |
| has ball | One-hot ball-holder indicator |
| layup, three, dunk skill | Per-episode skill for offense; zeroed for defense |
| lane steps | Normalized active lane counter |
| expected points | Current pressure-adjusted EP for offense; zeroed for defense |
| turnover risk | Defender-pressure turnover risk, scattered to the offense slot |
| steal risk | Targeted-pass interception risk, scattered to offense |
| distance to ball | Normalized hex distance |
| distance to best-EP player | Normalized hex distance |
| nearest opponent distance | Normalized hex distance |
| nearest teammate distance | Normalized hex distance |
| expected rebound target distance | Distance to the distribution’s expected target |
| rebound skill | Per-episode scalar |
| rebound specialist | Specialist indicator for the discrete skill mode |

### Globals: 7 features

| Feature | Interpretation |
|---|---|
| shot clock | Fraction of configured maximum |
| pressure exposure | Cumulative possession pressure |
| hoop `q`, `r` | Normalized hoop coordinate |
| expected rebound target `q`, `r` | Mean target under the current rebound table |
| rebound target entropy | Normalized uncertainty of the target distribution |

Rebound-derived values are zero when rebounds are unavailable, there is no
valid offensive ball holder, or the relevant table row cannot be used. The
observation describes the pre-sample distribution and never reveals the future
random rebound draw.

## Flat observation

The MLP baseline uses a larger flat feature vector. It includes absolute
positions, ball-holder one-hot state, clock and pressure, role encodings, ball
and hoop coordinates, pairwise offense-defense geometry, teammate geometry,
lane counters, EP, turnover/pass risks, rebound context, role flag, and
offensive skill deltas.

The flat representation is valuable for speed and debugging. The attention
representation is better aligned with the entity structure of the game and is
required by the current pointer-targeted action head and intent selector.
