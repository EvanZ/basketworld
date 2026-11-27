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

MLflow records exactly nine block series in both the core and full profiles:

- `jax/train/learner_block_pct`
- `jax/train/learner_blocks_per_defensive_possession`
- `jax/train/learner_blocks_per_episode`
- `jax/train/opponent_block_pct`
- `jax/train/opponent_blocks_per_defensive_possession`
- `jax/train/opponent_blocks_per_episode`
- `jax/eval_deploy/block_pct`
- `jax/eval_deploy/blocks_per_possession`
- `jax/eval_deploy/blocks_per_episode`

Block percentage is blocks divided by opposing layup + dunk attempts, multiplied
by 100. Open and contested finishes are included; jump shots are excluded.
Training credits each defending side separately; deploy combines both teams.
Per-possession and percentage metrics count events only from completed possessions,
with matching denominators. Per-episode metrics count all blocks in completed
games, excluding partial and truncated games; deploy counts each game once.
Training accumulates between logging points and carries pending possessions and
games across updates. After resuming, a partial possession/game whose beginning
was not observed is excluded until a complete one can be measured.
Zero-denominator values are unavailable, not zero.

Detailed `shot_defense_*` diagnostics remain available in evaluation artifacts;
they and the old raw block aliases are no longer logged to MLflow, including
role-specific evaluation. Historical MLflow data remains unchanged. No training
rule or checkpoint schema change is needed; resume a paused worker to use the
new logging code.

Three additional rim-contest metrics measure the probability reduction separately
from blocks:

- `jax/train/learner_rim_contest_mean_reduction_pp`
- `jax/train/opponent_rim_contest_mean_reduction_pp`
- `jax/eval_deploy/rim_contest_mean_reduction_pp`

Each is the average of `100 × (open make probability − conditional make
probability if unblocked)` over contested layups and dunks with rim protection
enabled, using completed possessions. For example, 80% reduced to 60% contributes
20 percentage points. Blocked attempts are included using their conditional
probability, so the independent block outcome/probability does not affect this
metric. Open finishes and perimeter shots are excluded. Training credits the
defending team; deploy pools attempts from both teams before dividing. No
qualifying attempts means unavailable. These use the existing diagnostics group
in the trainer UI and require a worker restart/resume before new values log.

Rim-contest frequency uses six additional metrics:

- `jax/train/learner_rim_contested_shots_per_episode`
- `jax/train/learner_rim_contested_shot_pct`
- `jax/train/opponent_rim_contested_shots_per_episode`
- `jax/train/opponent_rim_contested_shot_pct`
- `jax/eval_deploy/rim_contested_shots_per_episode`
- `jax/eval_deploy/rim_contested_shot_pct`

These six use **completed games only**, excluding partial and truncated games.
Training credits the named defending team: its contested layup/dunk attempts
divided by completed games, and its contested attempts divided by all opposing
layup/dunk attempts times 100. Deployment pools both teams; its per-episode
count is divided by **twice the completed game count**, reporting an average
per team per game. The deploy percentage pools contested attempts and all
layup/dunk attempts before dividing; it is not an average of team percentages.
A contest requires an adjacent eligible defender with rim protection enabled;
blocked attempts count. Zero denominators are unavailable. Training aggregates
games completed since the previous log; deployment aggregates its evaluation
batch. Pending game counts carry across updates and logging points. These
appear in the existing diagnostics group, with no additional raw interval counts.
All six are searchable and pinnable before their first logged value. The picker
labels them "Not logged yet" until MLflow has observations; empty charts do not
display invented zero values. A worker started before this logging update must
be paused and resumed to begin producing the new series.

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


### Compact layup metrics

The normal trainer picker replaces 108 historical shot-defense layup diagnostics
with nine selectable metrics in the existing diagnostics group:

- `jax/train/learner_layup_attempts_per_episode`
- `jax/train/learner_layup_makes_per_episode`
- `jax/train/learner_layup_fg_pct`
- `jax/train/opponent_layup_attempts_per_episode`
- `jax/train/opponent_layup_makes_per_episode`
- `jax/train/opponent_layup_fg_pct`
- `jax/eval_deploy/layup_attempts_per_episode`
- `jax/eval_deploy/layup_makes_per_episode`
- `jax/eval_deploy/layup_fg_pct`

All use completed games only, excluding partial and truncated games. Training
credits the named shooting team: attempts or makes divided by completed games.
Deploy divides both teams' counts by **2 × completed games**, an average per team
per game. FG% pools makes and attempts before dividing and multiplying by 100;
it is never divided by two. Blocked layups count as missed attempts. Dunks and
jump shots are excluded. Zero denominators mean unavailable; valid zero counts
are reported as zero. Training pools games completed since the previous log;
deploy pools the evaluation batch. Unfinished games carry across logging windows.

All nine can be pinned before any values log and show "Not logged yet".
Historical diagnostics are preserved and can be found through "Include scoring
diagnostics and legacy metrics"; existing pinned history is preserved. Workers
started before this update require pause/resume to produce the new series.
