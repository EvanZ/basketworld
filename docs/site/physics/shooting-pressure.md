# Shooting and pressure

A shot outcome combines the shooter's sampled skill, court position, shot
classification, and the most influential valid defender. The environment
computes the same pressure-adjusted probability for action outcomes,
observations, expected-points diagnostics, and potential shaping.

## Per-episode skills

At reset, each offensive player's layup, three-point, and dunk percentages are
sampled from configurable normal distributions and clipped to
\([0.01, 0.99]\). A standard deviation of zero makes a skill deterministic.

Every player also receives an episode-stable shot-contest skill sampled from
`Normal(shot_contest_skill_mean, shot_contest_skill_std)`. Contest skill is an
unbounded latent value: positive values strengthen a defender's eligible shot
pressure and negative values weaken it. In continuous multi-possession games,
the value stays attached to the stable player roster when offense and defense
switch.

Defense player entries retain the environment's base percentages internally,
but offensive skill features are gated to zero in policy tokens for defense.

## Distance interpolation

Let:

- \(p_L\) be the player's layup percentage;
- \(p_3\) be the player's three-point percentage;
- \(d_c\) be projected Cartesian shot distance in hex-scale units;
- \(D_3\) be `three_point_distance`;
- \(d_0=1\);
- \(d_1=\max(D_3+1,2)\).

The interpolation coordinate is:

\[
t = \operatorname{clip}\left(
  \frac{d_c-d_0}{d_1-d_0}, 0, 1
\right),
\]

and the base probability is initially:

\[
p_{\text{base}} = p_L + t(p_3-p_L).
\]

Discrete hex distance at most one forces \(p_L\). For projected distance
\(d_c>d_1\), the long-range adjustment is:

\[
p_{\text{long}} =
p_{\text{base}} -
\delta_3\max\left(0,d_c-\lfloor d_1\rfloor\right),
\]

where \(\delta_3\) is `three_pt_extra_hex_decay`. The result is clipped to
\([0.01,0.99]\).

When dunks are enabled and discrete hoop distance is zero, the sampled dunk
percentage replaces this curve. With `prefer_higher_percentage_shot=true`,
that override applies only if `dunk_pct >= layup_pct`. Otherwise the
player attempts a layup using `layup_pct`. Ties choose a dunk; the existing
contest adjustment applies after selecting the skill.

## Shot value and classification

Shot value comes from the precomputed court mask:

- basket cell: 2-point dunk or layup, selected by the rule above;
- cell outside the three-point line: 3 points;
- all other valid cells: 2 points.

The three-point cell mask and the continuous distance used by the make curve
are related but deliberately separate. This supports corner-like three-point
geometry while maintaining a smooth skill curve.

## Pressure cone

For a shooter, define a vector toward the hoop and a vector toward each
opponent. A defender can pressure the shot only when:

- it lies within the configured angular cone centered toward the basket;
- it is no farther from the shooter than the shooter's discrete distance to
  the basket;
- it is not at the same position.

If \(c\) is the cosine between those vectors and \(c_{\min}\) is the cosine of
half the configured pressure arc, the angular factor is:

\[
a = \frac{c-c_{\min}}{1-c_{\min}}.
\]

For defender hex distance \(d\), its geometric reduction is:

\[
\rho(d,a)=\rho_{\max}\exp\left[-\lambda(d-1)\right]a^2.
\]

The code uses
`shot_pressure_max` for \(\rho_{\max}\) and
`shot_pressure_lambda` for \(\lambda\).

When the shot-contest effect weight \(w_c\) is positive, eligible defender
\(i\)'s reduction is adjusted in logit space using its sampled contest skill
\(s_i\):

\[
\rho_i' = \sigma\left(\operatorname{logit}(\rho_i) + w_c s_i\right).
\]

The adjustment applies to each eligible defender before selecting the largest
reduction. A zero effect weight preserves the geometric pressure calculation
exactly.

Only the largest valid defender reduction is used:

\[
m_{\text{pressure}} =
\max\left(0,\,1-\max_i\rho_i'\right).
\]

When shot pressure is disabled or no defender qualifies,
\(m_{\text{pressure}}=1\).

## Final make probability

\[
p_{\text{make}} =
\operatorname{clip}
\left(p_{\text{base}}m_{\text{pressure}},\,0.01,\,0.99\right).
\]

The make result is a Bernoulli draw from this probability.

Expected points are:

\[
\operatorname{EP} = v_{\text{shot}}p_{\text{make}},
\]

where \(v_{\text{shot}}\) is two or three. EP is always retained as a
diagnostic. Whether it is also the task reward depends on rebound settings and
the configured rebound terminal reward mode; see
[Rewards and termination](rewards-termination.md).

## Shot-contest configuration and inspection

| Environment field | Training CLI flag | Meaning |
|---|---|---|
| `shot_contest_skill_mean` | `--shot-contest-skill-mean` | Mean of the per-player episode draw |
| `shot_contest_skill_std` | `--shot-contest-skill-std` | Standard deviation of the independent per-player draw; must be non-negative |
| `shot_contest_skill_effect_weight` | `--shot-contest-skill-effect-weight` | Non-negative logit-space weight \(w_c\); zero disables the skill effect |

These values are stored in checkpoint configuration and logged to MLflow. The
gameplay Environment tab exposes the three controls under **Shot Pressure** and
shows each sampled value in the **Defensive Skills (Episode)** table under
**Contest**. The Observation tab exposes `shot_contest_skill` when the current
multi-possession observation schema includes episode-stable player skills.
Evaluation player profiles use the key `shot_contest`.

!!! note "Model boundary"

    Shot probability is a compact positional model. It does not simulate a
    ball trajectory. The optional rebound table is generated by a separate
    offline physics process and is consulted only after a miss.

## Opt-in rim protection and contested shooting (issue #84)

These are separate **training flags**: `rim_protection_enabled` and
`contested_shooting_enabled`, both false by default. CLI options use hyphens
(`--rim-protection-enabled`, `--contested-shooting-enabled`); the trainer's
configuration editor exposes both flags and every parameter below. Sampling
parameters change the player population; effect weights change how strongly an
ability changes a shot. They are not interchangeable.

Rim protection applies only to layups and dunks at hex distance zero or one.
Any opponent exactly one hex from the shooter can contest, from any direction.
This replaces perimeter shot pressure on those finishes. Jump shots retain the
perimeter geometry. Among eligible defenders, use the greatest non-block
suppression and the greatest block probability, with one block roll. Tied
strongest blockers receive a seeded random attribution. Adding identical
adjacent defenders does not compound block probability.

Each player receives one persistent rim ability at reset:

```text
u ~ Beta(rim_protection_beta_alpha=1, rim_protection_beta_beta=3)
b = rim_protection_block_distribution_max=0.10 * u
s = logit(b) - logit(rim_protection_skill_block_reference=0.02)
block chance = sigmoid(logit(rim_protection_block_base=0.02)
                       + rim_protection_block_effect_weight=1.0 * s)
suppression = sigmoid(logit(rim_protection_contest_base=0.25)
                      + rim_protection_contest_effect_weight=0.5 * s)
```

`rim_protection_block_distribution_max` limits sampled abilities, not the effective
block probability after changing the mechanical baseline or effect weight.

Sampled probabilities are clipped to `[0.000001, 0.999999]` before logit to
keep skills finite. Baseline mechanic probabilities of zero or one remain exact. Conditional make
probabilities retain the existing 1%–99% clamp; the separate block roll can
reduce marginal make probability all the way to zero.
Defaults yield a mean sampled block probability of 2.5%, about 48.8% below 2%,
and a thin upper tail approaching 10%. These are probabilities **per eligible
finish**, not league-style blocks divided by all opponent field-goal attempts.
At sampled block probabilities 1%, 2%, 5%, and 10%, an 80% open finisher has
approximately 64.2%, 58.8%, 49.5%, and 40.5% marginal make probabilities.

Contested shooting is independent of open shooting accuracy:
`contested_shooting_skill_mean=0`, `contested_shooting_skill_std=1`, and
`contested_shooting_effect_weight=0.5`. On jumpers it subtracts
`effect_weight * shooter_skill` in the same log-odds calculation that adds the
defender's shot-contest ability. At geometric suppression 50%, neutral defender,
and shooter skill +1, suppression is 37.75%. It changes neither rim defense nor
open-shot accuracy. A disabled skill is zeroed in policy inputs.

Resolve the block first, then the make conditional on no block. With neutral
rim ability and an 80% open shot, conditional make probability is 60% and
marginal probability is `0.60 * (1 - 0.02) = 58.8%`. Expected points and shot
previews use the marginal probability. Dunk versus layup selection still uses
open percentages; when the higher-percentage-shot rule is enabled, ties select
a dunk. A block is one missed field-goal attempt and follows the ordinary
rebound calculation for that shot type and location. It is not a steal,
turnover, or automatic possession change; the blocker need not rebound it.

Enabling either flag requires observation schema 7. Checkpoints store the rule
version, all parameters, and skill-distribution metadata. Resume and inference
reject changed flags or settings even when network dimensions match. Old
checkpoints remain on their recorded rules. The dev UI displays the trained
flags under Shot Parameters; loading an old checkpoint does not enable blocks.

A block is recorded as `shot_blocked` and `shot_blocker`. The dev box score has
BLK, the play-by-play identifies shooter and blocker with shot type, distance,
and clock, and live/replay visuals show the credited defender's swat.

MLflow train, evaluation, and deployment diagnostics include `blocks`,
`learner_blocks`, `opponent_blocks`, and `blocks_per_completed_possession`.
`{learner,opponent,all}_shot_defense_*` counters group by **shooting side**,
including attempts, makes, eligible rim opportunities, blocks suffered, and
blocked-shot offensive/defensive recoveries. Rates include blocks per eligible
attempt and per shot, and recovery fractions per blocked shot. Open/contested
finish and jumper make rates, separate layup/dunk rates, open/conditional/
marginal probabilities, and jump suppression before/after resistance aid
interpretation. These count observed active steps, including partial games;
zero-denominator rates are omitted. They are behavior diagnostics, not Elo.

Skill Ratings schema 2 declares all eleven skills: layup, three-point shooting,
dunk, rebound, pressure, shot contest, steal, ball handling, passing, rim
protection, and contested shooting. Pools persist sampled abilities and their
sampling parameters. New studies record the effective environment and rule
contract in their immutable identity; extensions reject different contracts.
New studies for checkpoints with both new flags disabled use the original
nine-skill registry, including the existing perimeter shot-contest skill.
Their default pool and schedule sizes use those nine skills. Enabling either
new rule selects the eleven-skill schema, with the independently disabled skill
explicitly marked inactive. Legacy nine-skill pools remain readable for legacy rules. A new-rule study
requires a new compatible pool instead of silently adding abilities to old
players. Disabled skills and zero-variance skills are labeled not estimable in
the report, rather than shown as measured zero effects. Team-strength estimates
require a matching checkpoint and shot-defense contract.

The Direct skill effects table includes 90% percentile bootstrap intervals in
net-rating points per pool standard deviation. Whole sampling groups are drawn
with replacement from all included games, keeping paired games together and
holding player abilities and the selected ridge penalty fixed. Intervals that
include zero do not establish the direction of the effect. These intervals do
not include uncertainty from selecting the penalty or drawing a new player pool.
Disabled, constant, and otherwise unestimated coefficients have no interval;
neither do coefficients with fewer than two identifiable bootstrap fits.
Older reports remain unchanged and show unavailable intervals. A new analysis
revision can calculate intervals from their saved games without new simulation.
