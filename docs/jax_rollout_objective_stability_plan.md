# JAX Rollout Objective Stability Plan

## Problem

The current JAX trainer is fast because it runs fixed-shape compiled rollouts. The downside is that a fixed rollout budget can accidentally reward policies that terminate possessions early.

The suspicious pattern is:

- completed episodes increase sharply
- shot attempts or turnovers increase sharply
- reward per completed episode decreases
- mean reward per rollout step stays flat or increases

That can happen when the trainer optimizes a fixed number of rollout slots and immediately resets after `done`. A policy that ends episodes faster can create more terminal events inside the same rollout budget. If per-step averages or unbalanced transition sampling dominate, the model can learn to maximize throughput of episodes rather than quality per possession.

The real basketball objective is closer to **return per possession / episode**, not “reward density per compiled rollout slot.”

## Numerical Example

Assume a rollout has:

- `kernel_batch_size = 2048`
- `roles = 2`
- `rollout_horizon = 24`

Physical rollout slots:

```text
2048 * 2 * 24 = 98,304 slots/update
```

Suppose two policies see the same physical rollout budget.

Policy A:

- average episode length = 24 steps
- completed episodes per update ≈ 4,096
- reward per episode = 1.0
- total episode reward in rollout ≈ 4,096

Policy B:

- average episode length = 6 steps
- completed episodes per update ≈ 16,384
- reward per episode = 0.4
- total episode reward in rollout ≈ 6,554

Policy B is worse per possession, but it can look better under a fixed-slot objective because it produces more episodes. This is the exploit we need to remove.

## Candidate Fixes

## 1. Completed-Episode PPO Buffering

### Mechanism

Keep the fast fixed compiled rollout with immediate resets, but do not train directly on every raw transition in that rollout.

Instead:

- collect fixed-shape rollouts as usual
- identify episodes that fully start and finish inside the collected window
- exclude incomplete prefix and tail episodes from PPO
- compute returns over completed episodes
- weight transitions so each completed episode contributes equal total PPO weight
- eventually carry incomplete tail episodes into the next collection window with a fixed-shape buffer

This matches the intended objective better: optimize average possession quality, not reward density per rollout slot.

### Numerical Example

Assume:

```text
kernel_batch_size = 2048
roles = 2
rollout_horizon = 24
physical slots = 98,304
```

Suppose one env slot produces:

```text
episode A: 24 steps, reward = 1.0
```

Each transition gets:

```text
weight = 1 / 24
episode total weight = 1.0
```

Another env slot produces:

```text
episode B: 6 steps, reward = 0.4
episode C: 6 steps, reward = 0.3
episode D: 6 steps, reward = 0.5
episode E: 6 steps, reward = 0.4
```

Each transition gets:

```text
weight = 1 / 6
each episode total weight = 1.0
```

The PPO objective becomes an average over completed episodes. Short episodes no longer dominate because they have fewer transitions; each episode contributes one unit of loss weight.

### Important Caveat

If we use **all** completed episodes from a fixed rollout, a short-episode policy can still produce more completed episodes per update. That does not directly increase reward density if the loss is normalized by total episode weight, but it can change sampling variance and update composition.

The cleaner production version should therefore add one of:

- a target number of completed episodes/update
- a fixed completed-episode transition buffer
- carryover of incomplete episodes across rollout boundaries

The first implementation slice can still be useful without carryover: it removes incomplete fragments and gives each completed episode equal weight.

### Pros

- Keeps immediate resets, so physical throughput should remain close to the current fast path.
- Aligns PPO training with completed possessions.
- Avoids wasting post-terminal scan slots.
- Avoids the most obvious fixed-slot reward-density exploit.
- More natural than single-episode masking for production training.

### Cons

- More complex than the current raw rollout PPO batch.
- Needs episode-start tracking inside the trajectory.
- Needs weighted PPO loss, weighted advantage normalization, and careful metrics.
- Full carryover requires a fixed-shape cross-update buffer.
- Selector and discriminator should eventually use the same completed-episode semantics.

### Difficulty

Medium for the first implementation slice.

Medium-to-hard for the full carryover version.

Estimated implementation:

- completed-episode mask and episode-weighted PPO: 1 day
- selector/discriminator alignment: 1-2 additional days
- fixed-shape carryover buffer: 2-4 additional days

### When To Use

This is now the preferred production candidate if the early-termination exploit is real. It preserves the JAX speed advantage better than single-episode rollouts while moving the objective toward per-possession return.

## 2. Single-Episode Rollouts

### Mechanism

Each env slot contributes at most one episode per update.

When an env reaches `done`:

- do not reset it inside the current rollout
- mask all later slots from PPO
- mask all later slots from selector training
- mask all later slots from discriminator training
- reset the env only at the next PPO update boundary

This is now implemented behind:

```text
--single-episode-rollouts
```

### Numerical Example

With:

```text
kernel_batch_size = 2048
roles = 2
rollout_horizon = 24
```

Physical slots remain:

```text
98,304 slots/update
```

If the mean episode length is 10:

```text
active samples ≈ 2048 * 2 * 10 = 40,960
active fraction ≈ 40,960 / 98,304 = 41.7%
```

The compiled rollout still scans 98,304 slots, but only 40,960 contribute to learning.

### Pros

- Conceptually clean.
- Directly prevents “more short episodes per update” from increasing sample weight.
- Easy to reason about.
- Good diagnostic baseline for whether early termination was causing collapse.

### Cons

- Wastes post-terminal scan slots.
- `end_to_end_steps_per_sec` can look unchanged while useful learning throughput drops.
- Needs `active_end_to_end_steps_per_sec` and `ppo_active_sample_fraction` to understand real speed.
- May require larger `kernel_batch_size` if average episode length is much less than horizon.

### Difficulty

Low to medium. The core implementation is straightforward, but all losses, metrics, selector samples, and discriminator samples must respect the active mask.

### When To Use

Use as the first correctness check. If this removes the exploit, it proves the problem is rollout/objective weighting rather than model capacity or environment physics.

## 3. Episode-Balanced PPO Weighting

### Mechanism

Keep immediate resets inside the rollout, but change the loss weighting so short episodes do not get more influence merely because there are more of them.

Instead of every transition having weight `1`, give each completed episode approximately equal total weight.

One simple weighting:

```text
transition_weight = 1 / episode_length
```

Then normalize weights within the PPO batch.

### Numerical Example

Consider one env slot during a 24-step rollout.

Case A:

```text
one episode, length 24
each transition weight = 1/24
episode total weight = 1.0
```

Case B:

```text
four episodes, each length 6
each transition weight = 1/6
each episode total weight = 1.0
slot total weight = 4.0
```

This still gives more weight to a slot that completes more episodes. A stricter variant gives each env slot fixed total weight:

```text
transition_weight = 1 / (episode_count_in_slot * episode_length)
```

For four 6-step episodes:

```text
each transition weight = 1 / (4 * 6) = 1/24
slot total weight = 1.0
```

This removes the incentive to produce more episodes per env slot while preserving all rollout slots.

### Pros

- Preserves immediate resets and therefore most physical throughput.
- Avoids wasting post-terminal slots.
- Directly corrects the sampling/objective mismatch.
- Likely the best speed/correctness compromise.

### Cons

- More complex than masking.
- Need episode IDs or per-slot episode counters inside the compiled rollout.
- Need weighted advantage normalization, PPO loss, value loss, entropy, KL, selector loss, and discriminator sampling.
- Need decide whether weighting should be per completed episode, per env slot, or per role.

### Difficulty

Medium. It is easier than dynamic episode collection but requires careful weighting through all training paths.

### When To Use

This is the strongest candidate for the production training path if single-episode rollouts confirm the exploit.

## 4. Episode-Quota Rollouts With Fixed Max Scan

### Mechanism

Collect until a target number of completed episodes is reached, but keep JAX-compatible static shapes by scanning up to a fixed maximum.

Example config:

```text
target_completed_episodes = 4096
max_collection_steps = 32
kernel_batch_size = 2048
roles = 2
```

The scan runs at most 32 steps, but once the target episode quota is reached:

- additional transitions are masked
- additional episodes are ignored
- PPO uses only the transitions required to hit the quota

### Numerical Example

If average episode length is 12:

```text
episodes after 12 steps ≈ 2048 * 2 = 4096
quota reached around step 12
active slots ≈ 2048 * 2 * 12 = 49,152
physical max slots = 2048 * 2 * 32 = 131,072
active fraction ≈ 37.5%
```

If average episode length is 24:

```text
quota reached around step 24
active slots ≈ 98,304
physical max slots = 131,072
active fraction ≈ 75%
```

If average episode length is longer than `max_collection_steps`, the quota is not reached and the update uses a partial episode set.

### Pros

- Closer to “train on N possessions/update.”
- Avoids unlimited reward from generating extra episodes.
- Still mostly compatible with static-shape JAX.
- Cleaner conceptual match to basketball possessions.

### Cons

- Still wastes tail slots after quota.
- Needs robust quota masking logic across roles.
- Update-to-update active sample count varies.
- More complicated than single-episode masking.
- If quota is reached early, active fraction can be poor.

### Difficulty

Medium to hard.

Estimated implementation:

- prototype: 1-2 days
- robust PPO/selector/discriminator/metrics integration: 2-4 days

### When To Use

Use if we want a possession-count objective but do not want to implement true dynamic compaction yet.

## 5. True Dynamic Episode Compaction

### Mechanism

Maintain active env slots and a fixed-size transition buffer. Whenever an env finishes:

- write the completed episode or its transitions into the buffer
- immediately reset that env slot
- continue collecting
- stop when the buffer contains the target number of completed episodes or transitions

This is the closest JAX version of “collect exactly N completed possessions.”

### Numerical Example

Target:

```text
target_completed_episodes = 4096
```

If mean episode length is 6:

```text
needed active transitions ≈ 24,576
```

If mean episode length is 24:

```text
needed active transitions ≈ 98,304
```

The collection cost scales with actual episode length, not with a fixed horizon. That is correct conceptually, but it means update sizes and compute patterns are harder to manage.

### Pros

- Best conceptual fit for per-possession learning.
- Minimal wasted rollout work.
- Directly removes the “more episodes per fixed rollout” exploit.
- Could be fastest in useful samples/sec once mature.

### Cons

- Hardest to implement in JAX.
- Requires fixed maximum buffers anyway.
- Requires scatter/ring-buffer logic.
- Harder to make compatible with selector state, intent labels, grouped opponents, offense/defense role training, and metric summaries.
- More risk of subtle bugs.

### Difficulty

Hard.

Estimated implementation:

- prototype: several days
- reliable trainer integration: about a week or more

### When To Use

Only after the simpler methods prove the objective correction is necessary and worth optimizing further.

## 6. Reward Tweaks / Terminal Penalties

### Mechanism

Add or tune rewards to discourage short bad episodes, for example:

- terminal turnover penalty
- minimum possession-length bonus
- shot-clock survival bonus
- per-possession scoring bonus

### Numerical Example

If early turnovers are worth `-0.5`, increase them to `-1.0`. If the policy is ending possessions early, this may suppress the behavior.

### Pros

- Easy to implement.
- Can be useful for domain-specific shaping.
- May improve behavior quickly.

### Cons

- Does not fix the sampling/objective mismatch.
- Can hide a trainer bug.
- Requires tuning.
- Risk of creating new reward hacks.

### Difficulty

Low.

### When To Use

Use only after the objective/sampling issue is addressed. Reward shaping should refine behavior, not compensate for a bad training objective.

## Throughput Metrics To Track

For all approaches, log both physical throughput and useful learning throughput.

Important metrics:

- `end_to_end_steps_per_sec`: physical scanned slots/sec
- `active_end_to_end_steps_per_sec`: active learning samples/sec
- `ppo_active_sample_fraction`: active PPO fraction
- `rollout_active_step_fraction`: active rollout fraction
- `completed_episodes`
- `mean_completed_episode_length`
- `learner_reward_per_completed_episode`
- `learner_points_per_completed_episode`
- `mean_reward`
- `done_rate`
- turnover rates by reason
- shot type shares

For this specific bug, the key comparison is:

```text
learner_reward_per_completed_episode vs mean_reward
```

If mean reward rises while reward per completed episode falls, the policy may still be exploiting the training objective.

## Batch Size Guidance

Do not tune batch size from physical slots alone.

Current physical samples/update:

```text
kernel_batch_size * rollout_horizon * roles
```

Useful samples/update in single-episode mode:

```text
kernel_batch_size * min(mean_episode_length, rollout_horizon) * roles
```

Example:

```text
kernel_batch_size = 2048
rollout_horizon = 24
roles = 2
mean_episode_length = 10
```

Useful samples:

```text
2048 * 10 * 2 = 40,960
```

Physical samples:

```text
2048 * 24 * 2 = 98,304
```

Active fraction:

```text
40,960 / 98,304 = 41.7%
```

If this is too noisy, increasing `kernel_batch_size` to 4096 gives:

```text
4096 * 10 * 2 = 81,920 useful samples
```

But increasing batch size should come after observing `ppo_active_sample_fraction` and learning stability.

## Recommendation

Use this sequence:

1. Run `--single-episode-rollouts` with `--rollout-horizon 24` as the diagnostic baseline if we need the cleanest possible correctness check.
2. Implement `--ppo-completed-episodes-only` as the production-oriented speed-preserving candidate.
3. Compare completed-episode PPO against the previous fast immediate-reset run.
4. If completed-episode PPO removes the collapse, add fixed-shape carryover for incomplete episodes.
5. Only pursue episode-quota or true compaction if completed-episode PPO is not enough.

The current best bet is:

```text
single-episode rollouts for diagnosis
completed-episode PPO for production speed
fixed-shape carryover as the next correctness improvement
episode-quota collection as a later optimization if needed
```
