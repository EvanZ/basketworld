# Intent and play discovery

The intent system gives the low-level policy a discrete latent variable
\(z\)—a play identifier—and trains auxiliary components to choose and
differentiate those latents.

It consists of three separate mechanisms:

1. runtime intent state in the environment;
2. policy conditioning and an optional selector head;
3. a separate discriminator and diversity bonus.

## Runtime intent

Offense and defense have independent fields for:

- intent index;
- active flag;
- age;
- commitment steps remaining.

At reset, an enabled role samples an active intent unless its configured null
probability produces no intent. Active intents age on each step and expire
after their commitment.

The policy receives only the role-appropriate index and gate. Offense and
defense use separate embedding tables.

## Integrated selector

The attention policy can add:

- logits over `num_intents`;
- a scalar selector value.

Both heads read the offense CLS representation. Selector inference neutralizes
the low-level intent context, mixes the learned probabilities with a uniform
epsilon floor, and samples a new intent.

The selector can run at:

- the beginning of an active intent segment;
- commitment expiration when multiselect is enabled;
- a completed-pass or offensive-rebound boundary after the minimum play
  length when multiselect is enabled.

`selector_alpha` is the probability of using the selector at an eligible
boundary. `selector_eps` mixes its distribution with uniform exploration:

\[
\mu_\epsilon(z|s) =
(1-\epsilon)\mu(z|s) + \frac{\epsilon}{K}.
\]

Both values have update-based warmup and ramp schedules.

## Selector objective

Selector training builds one sample per boundary at which a selector choice
was used. Segment returns accumulate low-level rewards until the next selector
boundary, episode end, or rollout bootstrap.

The selector objective contains:

- clipped PPO policy loss for the chosen intent;
- selector value MSE;
- entropy regularization;
- KL from mean selector usage to the uniform distribution.

\[
L_{\text{selector}} =
L_\pi
+c_VL_V
-c_H\mathcal H
+c_U D_{\mathrm{KL}}(\bar\mu\,\|\,U).
\]

The selector has a separate Optax state. Its update is explicitly masked so
only the four selector policy/value head modules change; the attention
backbone and low-level actor remain unchanged by this auxiliary update.

## Intent discriminator

The discriminator is separate from the actor-critic. Its primary
`set_step` encoder reconstructs player tokens, globals, and role flag from
rollout observations, then applies:

1. a token MLP;
2. a learned CLS token;
3. multi-head self-attention with dropout;
4. an intent classifier.

The discriminator predicts the active intent from rollout states. Alternative
flat encoders can combine truncated observations, actions, and event features,
but `set_step` matches the entity structure of the current attention policy.

Training uses active intent samples, a train/holdout split, cross-entropy, and
multiple updates per rollout. Diagnostics include loss, top-1 accuracy,
entropy, macro one-vs-rest AUC, per-intent usage, and optional embedding/sample
dumps.

## Diversity bonus

The discriminator supplies a raw score for how recognizable the active intent
is. The current complete-episode path:

1. averages raw scores over intent-active steps in each complete possession;
2. standardizes episode means across the role rollout;
3. clips and re-centers them to zero mean;
4. scales each episode total by the scheduled beta;
5. distributes that total uniformly over its intent-active steps.

Leading and trailing incomplete episodes are not shaped. Re-centering avoids a
systematic positive reward offset, and episode-level allocation avoids making
the bonus grow merely because a possession lasts longer.

The resulting bonus is added to rollout reward before GAE is computed. Offense
and defense discriminators can be enabled independently and have separate
parameter, optimizer, and normalization state.

## Required configuration relationships

The trainer enforces:

- intent embeddings require `--policy-model attention`;
- embeddings require offense or defense intent learning;
- the selector requires attention, intent learning, and intent embeddings;
- intent-diversity training requires the corresponding intent runtime.

These checks prevent configurations in which a selector or discriminator is
nominally active but cannot affect the policy.

## What to inspect

Healthy intent learning is not established by classifier accuracy alone.
Inspect:

- selector usage and entropy;
- per-intent selection frequency;
- discriminator holdout metrics;
- policy sensitivity when the same state is conditioned on different intents;
- reward and episode outcomes by intent;
- state/action embeddings and qualitative trajectories;
- bonus magnitude relative to task return.
