# Passing and turnovers

The primary JAX path treats pass actions as teammate-target slots. Pass safety
depends on the passer, receiver, intervening defenders, and the selected
interception model.

## Pointer-targeted receiver

For each player, `KernelStatic` stores up to six teammate IDs in stable,
player-ID order. Pass action IDs 8–13 select slots 0–5. The action mask exposes
only populated slots for the current ball holder.

An unavailable slot is treated as an out-of-bounds pass turnover if it reaches
the transition function. Legal action masks normally prevent this case.

## Independent hazard aggregation

The pass model first computes a contribution \(h_i\in[0,1]\) for each defender
against a particular receiver. The total interception probability is:

\[
p_{\text{steal}} = 1-\prod_i(1-h_i).
\]

This gives multiple defenders independent opportunities without simply
summing probabilities above one.

If the pass is stolen, the interceptor is sampled from defenders in proportion
to their positive contributions. If no usable weights exist, possession falls
back to the nearest defender.

## Line model

The `line` model measures each defender's perpendicular distance to the
passer-receiver segment and its projection \(t\) along that segment. Only
defenders strictly between the endpoints contribute.

A simplified view of the lane contribution is:

\[
h_{\text{lane},i} =
b\,
\exp(-\lambda_\perp d_{\perp,i})\,
(1+\alpha d_{\text{pass}})\,
w(t_i).
\]

Here \(b\) is `base_steal_rate`,
\(\lambda_\perp\) is `steal_perp_decay`, \(\alpha\) is
`steal_distance_factor`, and \(w(t)\) interpolates from
`steal_position_weight_min` near the passer toward one near the receiver.

The contribution is clipped after the enabled factors are combined.

## Lob-aware model

The `lob_aware` model starts from the line contribution and can reduce the
interior lane term for receivers near the basket through
`pass_lob_lane_multiplier`.

It then adds:

- passer-pressure risk from nearby, aligned defenders;
- receiver-pressure risk from nearby defenders aligned against the pass
  direction.

The associated weights allow experiments to trade central-lane interceptions
against endpoint pressure.

## Reaction model

The `reaction` model uses pass time and defender reach instead of the legacy
lane score as its primary mechanism.

For pass distance \(d_p\), pass speed \(v_p\), reaction delay \(\tau\),
defender speed \(v_d\), and reach radius \(r_d\):

\[
t_p = \frac{d_p}{\max(v_p,0.1)},
\]

\[
r_{\text{reachable}} =
v_d\max(0,t_p-\tau)+r_d.
\]

Receiver hazard uses a softened reach test:

\[
p_{\text{react},i} =
\sigma\left(
\frac{r_{\text{reachable}}-d_{\text{receiver},i}}
     {s_{\text{reaction}}}
\right),
\]

then scales it by receiver risk and a line-alignment multiplier. A separate
passer hazard decays exponentially with defender distance. An optional
`lane_weight` restores some line-lane risk.

The three hazards are combined as:

\[
h_i =
1-(1-h_{\text{passer},i})
(1-h_{\text{receiver},i})
(1-h_{\text{lane},i}).
\]

This model makes longer, slower passes easier to reach while still representing
pressure at either endpoint.

## Defender-pressure turnover

Before the chosen action is resolved, nearby defenders can cause a possession
turnover against an offensive ball holder. A defender qualifies when it is:

- within `defender_pressure_distance`; and
- in the forward half-plane from the handler toward the hoop.

At distance \(d\), its turnover probability is:

\[
p_i =
p_0\exp[-\lambda\max(0,d-1)].
\]

Each defender receives an independent draw. A success immediately transfers
the ball to a successful defender and terminates the possession. The policy
observation exposes the aggregate risk
\(1-\prod_i(1-p_i)\), while `pressure_exposure` accumulates that aggregate
through the possession.

## Completed passes and assists

A completed pass:

- moves the ball to the selected teammate;
- earns the configured pass reward;
- records the passer, receiver, and expiration step as an assist candidate.

If that receiver shoots before the candidate expires, the environment records
a potential assist and applies a reward proportional to shot EP. A made shot
adds the configured full-assist bonus and records an assist. Any shot attempt
clears the candidate.

Pass interception ends the possession, so no assist state remains.
