# Movement and rules

BasketWorld resolves a joint action for all players. A step is simultaneous at
the policy interface, but the transition kernel uses a fixed internal order so
that stochastic events, movement, rules, and rewards are reproducible.

## Resolution order

For an active state, the JAX kernel:

1. increments the episode step and advances intent clocks;
2. samples a defender-pressure turnover against the current ball holder;
3. decrements the shot clock;
4. resolves the ball holder's shot or pass;
5. resolves player movement;
6. samples shot and rebound outcomes;
7. applies offensive and defensive lane rules;
8. applies shot-clock expiration;
9. constructs rewards and potential shaping.

A pressure turnover ends the step immediately, so the chosen action and
movement do not occur.

## Proposed movement

Actions 1–6 add one of the six axial direction vectors to the player's current
position. A proposed move is valid only if:

- the destination is a legal court cell;
- the destination is not the basket cell when dunks are disabled;
- no stationary player currently occupies it.

When `mask_occupied_moves` is enabled, moves into cells occupied at the start
of the step are also removed from the policy action mask. Transition-time
checks remain in place regardless of the mask setting.

An invalid move by the ball holder is a turnover. Possession transfers to the
nearest opposing player, measured from the turnover location, and the episode
ends. An invalid move by a non-holder leaves that player in place.

## Collisions

Several moving players may request the same otherwise-empty destination. The
kernel draws one uniform tie-break value per player and moves only the
highest-valued contender. All losing contenders remain at their original
cells.

A contender cannot displace a player who remains stationary on the
destination. Position swapping is also rejected because each destination is
tested against positions occupied at the start of movement.

This is a collision-resolution rule, not physical momentum: players have no
velocity, mass, or acceleration state.

## Ball actions and movement

Only the ball holder's `SHOOT` or pass action changes possession. Other
players' pass actions normally have no ball effect.

On a shot step:

- the shooter is forced to `NOOP` for movement;
- non-holder pass actions are treated as `NOOP`;
- other legal movers resolve before the rebound target and winner are sampled.

This makes pre-rebound positioning strategically relevant without moving the
shooter and without treating the sampled rebound target as a player
destination.

On a pass step, the holder transfers or loses the ball first. Other players
then resolve their movement for the same simulation interval.

## Shot clock

The shot clock is decremented before the chosen ball action. If the resulting
clock is at or below zero and no shot occurred, the episode ends as a
shot-clock turnover.

An offensive rebound can raise the remaining clock to the configured reset
value:

\[
c_{\text{next}} = \max(c_{\text{remaining}}, c_{\text{ORB reset}}).
\]

The default research configuration uses a 14-step offensive-rebound reset,
but the value is configurable.

## Offensive three seconds

When enabled, each offensive player has a lane counter.

- A player outside the offensive lane resets to zero.
- A player inside increments by one.
- A non-ball-handler violates when its updated counter is greater than or
  equal to `three_second_max_steps`.
- The ball handler violates only when its counter is strictly greater than the
  threshold and it is not shooting.

The ball transfers to the nearest defender and the episode ends. The holder's
one-step grace at the threshold permits a shot.

## Defensive lane rule

When illegal defense is enabled, a defender increments its lane counter only
while:

- it is in the defensive lane; and
- it is not within `defender_guard_distance` of any offensive player.

Leaving the lane or guarding an offensive player resets the counter. A
violation occurs once the counter is strictly greater than
`three_second_max_steps`.

The offense receives one score point plus the configured violation reward, and
the episode ends. Defensive lane counting is disabled on a shot step.

## Episode boundaries

The environment is possession-oriented rather than a full-game simulator.
Turnovers and defensive rebounds therefore terminate instead of continuing in
the opposite direction. An offensive rebound retains the offense and is the
exception that extends a possession.

The compiled rollout runner can reset terminal batch rows immediately and
continue collecting. With `--single-episode-rollouts`, post-terminal slots are
instead masked so a row contributes at most one possession to that update.
