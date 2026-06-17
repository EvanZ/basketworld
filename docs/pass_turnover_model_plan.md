# Simplified Pass Turnover Model Plan

## Motivation

The current pass interception model is too geometric: it mostly asks whether a defender is near the exact pass line. That misses an important 5-on-5 effect: long passes give nearby defenders more reaction time, especially defenders near the receiver, even when they are not directly on the line segment.

For the next iteration, keep the model simple and tuneable. Do not model lobs or explicit pass speed yet. Split turnover risk into two interpretable components:

- Passer-side disruption: nearby defenders make the release harder.
- Receiver-side reaction/catch risk: defenders near the receiver become more dangerous as pass distance grows.

## Core Formula

Use noisy-or composition:

```text
p_turnover = 1 - (1 - p_passer) * (1 - p_receiver)
completion_pct = 1 - p_turnover
```

Keep the existing pass UI display as a completion percentage, but internally treat this as pass turnover probability rather than pure steal probability.

## Passer-Side Risk

Passer-side risk depends on nearest defender pressure around the ball handler.

```text
nearest_dist = min_i distance(passer, defender_i)
passer_pressure = exp(-passer_pressure_decay * max(0, nearest_dist - 1))
p_passer = base_passer_risk * passer_pressure
```

Optional extension if needed:

```text
p_passer = 1 - prod_i(1 - base_passer_risk * pressure_i)
```

For now, nearest-defender pressure is probably enough and easier to tune.

## Receiver-Side Risk

Receiver-side risk is the main missing piece. Each defender near the receiver contributes a hazard. That hazard grows with pass distance because longer passes provide more reaction time.

```text
receiver_hazard_i =
    base_receiver_risk
    * pass_distance_multiplier
    * receiver_distance_multiplier_i
    * receiver_alignment_multiplier_i
```

Where:

```text
pass_distance_multiplier = 1 + reaction_distance_factor * max(0, pass_distance - safe_pass_distance)

receiver_distance_multiplier_i = exp(
    -receiver_pressure_decay * max(0, distance(receiver, defender_i) - 1)
)

receiver_alignment_multiplier_i = receiver_alignment_min + (1 - receiver_alignment_min) * alignment_i
```

`alignment_i` should be high when the defender is positioned between the passer and receiver or near the catch path. It should be low but not necessarily zero for defenders simply close to the receiver, because they can still contest the catch.

Combine receiver hazards with noisy-or:

```text
p_receiver = 1 - prod_i(1 - clamp(receiver_hazard_i, 0, max_receiver_hazard))
```

## Optional Legacy Lane Risk

The old line-interception model can remain available as a small optional component, but it should not be the dominant model in 5-on-5.

```text
p_lane = old_line_model * lane_weight
p_turnover = 1 - (1 - p_passer) * (1 - p_receiver) * (1 - p_lane)
```

Initial recommendation: set `lane_weight = 0` or very small while calibrating the new model.

## Proposed Config Params

Use a new model name so old checkpoints and configs remain interpretable:

- `pass_turnover_model`: `line` or `reaction`
- `pass_passer_base_risk`
- `pass_passer_pressure_decay`
- `pass_receiver_base_risk`
- `pass_receiver_pressure_decay`
- `pass_reaction_distance_factor`
- `pass_safe_distance`
- `pass_receiver_alignment_min`
- `pass_receiver_max_hazard`
- `pass_lane_weight`

Existing params can remain for `pass_turnover_model=line`:

- `base_steal_rate`
- `steal_perp_decay`
- `steal_distance_factor`
- `steal_position_weight_min`

## Calibration Targets

Start with board-level qualitative calibration before training:

- Open short pass: 90-98% completion.
- Open long pass: 80-95% completion depending on receiver pressure.
- Long pass to a receiver with a defender nearby: materially risky.
- Defender directly on the passing lane: risky, but not automatically 100% unless also close to passer/receiver.
- Multiple defenders near receiver: risk compounds through noisy-or.

## Implementation Steps

1. Add `reaction` model config fields to the Python env and backend schema.

2. Implement a Python helper for board/UI previews that returns per-receiver turnover probabilities.

3. Implement the same formula in the JAX env as a vectorized function.

4. Keep current UI labels initially, but update copy from “steal probability” toward “pass turnover probability” where feasible.

5. Add a dev UI playground tab or panel for pass model calibration:

```text
- draggable players
- selected passer / receiver
- editable pass model params
- displayed p_passer, p_receiver, p_lane, p_turnover
- displayed completion percentage for each legal receiver
```

6. Add Python/JAX parity tests for fixed board states.

7. Run 5-on-5 ablations comparing `line` vs `reaction`, tracking pass attempts, completed passes, pass turnovers, PPP, reward, ORB%, and DRB%.

## Deferred

Do not implement these yet:

- Lob-specific rules.
- Explicit pass speed choice.
- Separate catch-error event types.
- Player pass/catch/deflection skills.
- Full physical ball trajectory simulation.

Those can come later if the simple reaction model is not expressive enough.
