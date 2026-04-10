# The Play's the Thing: Post Outline and Technical Guide

## Purpose

This document is a drafting guide for a Substack-style post about the learned-play work in
BasketWorld.

It is not a polished article. It is a structure you can write from without drifting away from
the actual implementation.

Working title:

`The Play's the Thing: Learning a Latent Play Space with a DIAYN-Inspired Discriminator`

---

## One-sentence thesis

Instead of hand-authoring a playbook, we can train a policy to condition on a latent variable
`z`, reward it for making different `z` values produce behaviorally distinct multi-step
trajectories, and thereby discover a latent play space directly from reward.

---

## Short abstract

Modern deep RL policies are usually trained to choose the next action well. Basketball, however,
is not only about the next action. It is about coordinated multi-step sequences: drive-and-kick
patterns, decoy actions, cuts that only make sense two beats later, and possessions whose meaning
is only visible in hindsight. This project explores whether a policy can learn those multi-step
structures without a handcrafted playbook. The core idea is to condition the offense on a latent
intent `z`, embed that intent into the policy, and add a DIAYN-inspired discriminator that rewards
the policy when different latent choices lead to distinguishable trajectory segments. The result is
not a hand-scripted finite state machine, but a learned latent play space.

---

## Recommended article arc

Use this sequence. It gives the post a clear narrative:

1. Start from the basketball intuition.
2. Explain why standard RL is weak at "plays".
3. Introduce the latent play idea.
4. Explain the DIAYN connection.
5. Show the actual architecture.
6. Explain the implementation mistakes and what changed.
7. Show the metrics that indicate the system is finally working.
8. End with limitations and why this is not the final form.

The post will be stronger if it reads like an engineering research note rather than a hype piece.

---

## Suggested outline

## 1. Opening hook: a play is not a single action

Goal:

Explain why the object you want to learn is inherently multi-step.

Points to make:

1. A basketball "play" is not "shoot now" or "pass now".
2. It is a structured sequence whose meaning unfolds over several decisions.
3. Most RL setups do not explicitly model that structure.
4. If you train only for per-step reward optimization, the agent may become effective without ever
   developing reusable play families.

Possible opening lines:

1. "A play is only obvious after several actions have already happened."
2. "The hardest part of learning basketball is that the unit of meaning is not the action, but the
   sequence."
3. "A possession can look random at step one and coherent at step five."

---

## 2. The core problem: standard policies do not naturally produce reusable plays

Goal:

Describe the failure mode you are addressing.

Points to make:

1. Standard PPO-style policies optimize expected return over primitive actions.
2. That setup can learn good local decisions without learning distinct reusable strategic modes.
3. If you introduce a latent variable `z` but do nothing else, the policy can simply ignore it.
4. Therefore the challenge is not just "add a latent", but "make the latent behaviorally matter".

Good phrasing:

1. "Without additional pressure, the latent collapses into irrelevance."
2. "The policy has no reason to use `z` unless different `z` values help either reward or
   identifiability."

Technical framing:

- Baseline form:
  - `pi(a_t | s_t)`
- Latent-conditioned form:
  - `pi(a_t | s_t, z)`
- Collapse risk:
  - `pi(a_t | s_t, z_1) ~= pi(a_t | s_t, z_2)` for all `z`

---

## 3. The main idea: learn a latent play space

Goal:

Define the concept cleanly.

Suggested definition:

A latent play space is a set of discrete latent codes `z in {0, ..., K-1}` whose learned
embeddings `e_z` condition the policy so that different codes correspond to different multi-step
behavior families.

Important clarification:

1. The system does not hand-label plays like `PICK_AND_ROLL` or `HORNS`.
2. It does not encode coaching doctrine.
3. It learns latent play identities that may later be interpreted by a human.

Precise language to use:

1. "`z` is a discrete latent play identity."
2. "`e_z` is the learned embedding associated with that latent play identity."
3. "The model is learning a latent play space, not a manually authored playbook."

Avoid saying:

1. "The model stores plays exactly like a coach's clipboard."
2. "Each latent directly corresponds to a canonical real-world play."

---

## 4. Why this is DIAYN-inspired, but not just DIAYN

Goal:

Be technically honest about the lineage.

What to say:

1. The design borrows the mutual-information idea from DIAYN.
2. A discriminator `q(z | tau)` tries to infer which latent generated a trajectory.
3. The policy receives a diversity bonus when different `z` values lead to distinguishable
   trajectories.
4. But this is not pure DIAYN:
   - DIAYN is usually unsupervised skill discovery.
   - DIAYN is commonly paired with SAC.
   - This system is task-conditioned and built into a PPO training loop.

Useful framing:

1. "DIAYN gives the right inductive bias: if the discriminator can infer the latent from the
   trajectory, the latent is behaviorally meaningful."
2. "The project keeps that principle, but applies it inside a reward-driven basketball task rather
   than a purely unsupervised skill-discovery setting."

Equation to include:

`J_total = J_env + beta * J_div`

where

`J_div ~ E[log q(z | tau) - log p(z)]`

with uniform latent prior

`p(z) = 1 / K`

Equivalent reward-shaping form:

`r_total_t = r_env_t + beta * b_t`

and at the episode or segment level:

`B(tau, z) = log q(z | tau) + log K`

Clarify:

1. The discriminator loss is not directly added to PPO loss.
2. Instead, the discriminator produces a bonus that is injected into rollout rewards.

That distinction matters and is worth spelling out.

---

## 5. The actual architecture

Goal:

Show the full stack clearly enough that a technical reader can understand what was built.

Recommended subsection order:

### 5.1 Environment-level latent state

What to say:

1. At possession start, the environment samples a latent intent `z`.
2. That latent is active for a commitment window of `K` steps.
3. The offense sees its own latent.
4. The defense does not see the offense latent.
5. After the commitment window expires, the current implementation deactivates the latent and the
   rest of the possession runs without active intent conditioning.

Important current semantics:

1. This is offense-first.
2. `z_off` and `z_def` are architecturally separated, but current training focuses on offense
   intent learning.
3. The discriminator and diversity bonus only use the active-intent prefix.

### 5.2 Set observation and intent globals

What to say:

1. The policy uses set observations rather than a flat monolithic vector.
2. Global features include intent-related context.
3. The current intent globals used by the set-attention path are:
   - `intent_index_norm`
   - `intent_active`
   - `intent_visible`
   - `intent_age_norm`

Why `intent_age_norm` matters:

1. The policy needs some notion of where it is within the play window.
2. Even a simple normalized age is better than giving the policy only a latent identity with no
   temporal position signal.

### 5.3 Policy conditioning with learned intent embeddings

What to say:

1. The policy does not use the raw scalar intent index directly.
2. It looks up a learned embedding `e_z`.
3. That embedding is injected into the set-attention processing path.
4. The result is that the same physical state can produce different action distributions depending
   on `z`.

Important conceptual point:

1. `e_z` is an embedding of the latent play identity.
2. It is not a post-hoc embedding of a realized play trajectory.

### 5.4 Why the first discriminator was too weak

This is a strong middle section because it shows engineering iteration.

What to say:

1. The first discriminator took per-step flat observation/action features.
2. It mean-pooled them over time.
3. That destroyed temporal order.
4. Two different sequences with similar averages became hard to distinguish.
5. As a result, policy sensitivity could rise while discriminator accuracy stayed near chance.

This is an important insight. It is the bridge between the original idea and the improved version.

### 5.5 GRU discriminator over the active prefix

This should be one of the most technical sections in the post.

What to say:

1. The discriminator was upgraded from mean pooling to a GRU-based trajectory encoder.
2. Each timestep contributes a step feature built from flattened observation and flattened action.
3. A step encoder projects those features.
4. A GRU processes the ordered step sequence.
5. The final hidden state becomes the episode or segment embedding.
6. A classifier head predicts `q(z | tau_prefix)`.

Important current semantics:

1. The GRU sees only the active-intent prefix, not the full possession.
2. This aligns the discriminator with the commitment window.
3. Late-possession cleanup actions no longer dilute the intent label.

Useful pseudo-equations:

`x_t = concat(flat_obs_t, flat_action_t)`

`h_t = GRU(phi_step(x_t), h_{t-1})`

`h_episode = h_T`

`q(z | tau) = softmax(W h_episode + b)`

---

## 6. Why the commitment window matters

Goal:

Explain the design decision cleanly.

What to say:

1. A latent play should not necessarily govern an entire possession indefinitely.
2. In the current implementation, the latent is active for a fixed number of steps.
3. After that, `intent_active` becomes false.
4. The discriminator and diversity bonus use only the active-intent prefix.

Why this matters:

1. It gives the latent a real temporal scope.
2. It reduces label noise in the discriminator.
3. It better matches the intuition that a play is a setup sequence, not necessarily the entire
   possession.

Good honest caveat:

1. This is still a simplified commitment model.
2. There is no higher-level controller that selects a new play mid-possession.
3. The system currently learns "one latent-guided setup phase, then generic continuation."

---

## 7. What the system is actually learning

Goal:

State the claim at exactly the right level.

Recommended phrasing:

1. "The system is learning latent-conditioned multi-step behavior families."
2. "When successful, those families are readable from trajectories and can be interpreted as
   learned plays."
3. "This is stronger than one-step action modulation, but weaker than a fully symbolic play
   library."

This is the central epistemic paragraph of the article. Get it right.

Avoid overclaiming:

1. Do not say the model has discovered canonical NBA plays.
2. Do not say the latent is guaranteed to be semantically clean.
3. Do not say interpretability is solved.

Do say:

1. The latent is becoming behaviorally meaningful.
2. The model can be pushed toward reusable multi-step modes.
3. The resulting structure is measurable, not purely anecdotal.

---

## 8. The metrics section: how to tell if latent plays are emerging

Goal:

Show that this was measured, not guessed.

Use this structure:

### 8.1 Policy sensitivity metrics

Explain:

These ask whether the policy changes when only `z` changes and the state is held fixed.

Metrics:

1. `intent/policy_kl_mean`
2. `intent/policy_tv_mean`
3. `intent/action_flip_rate`

Interpretation:

1. If these stay near zero, the policy is ignoring `z`.
2. If they rise, the policy is using the latent.

### 8.2 Discriminator metrics

Explain:

These ask whether trajectory segments are actually separable by latent.

Metrics:

1. `intent/disc_top1_acc`
2. `intent/disc_auc_ovr_macro`
3. `intent/disc_loss`

Interpretation:

1. With `K=8`, chance top-1 accuracy is `1/8 = 0.125`.
2. AUC near `0.5` means weak or no separability.
3. AUC significantly above `0.5` means the latent is being expressed in behavior.

### 8.3 Behavioral metrics by intent

Explain:

These ask whether different intents correspond to different kinds of possessions.

Metrics already worth mentioning:

1. `intent/ppp_by_intent/<z>`
2. `intent/pass_rate_by_intent/<z>`
3. `intent/shot_2pt_share_by_intent/<z>`
4. `intent/shot_3pt_share_by_intent/<z>`
5. `intent/shot_dunk_share_by_intent/<z>`

Possible sentence:

"A useful latent should not only be classifier-readable; it should correspond to different shot
profiles, passing tendencies, or efficiency tradeoffs."

### 8.4 Usage balance

Metric:

1. `intent/usage_entropy`

Why it matters:

1. A high-separation latent that collapses onto one or two intent IDs is not very interesting.
2. The model should ideally use multiple regions of the latent play space.

---

## 9. A good middle-section narrative: what failed first

This section will make the post much stronger because it shows the real research path.

Suggested sequence:

1. We first added a latent and a diversity objective.
2. The policy initially ignored the latent.
3. Policy-sensitivity diagnostics showed almost zero change under different `z`.
4. Strengthening the policy-side embedding path helped.
5. Then policy sensitivity rose, but the discriminator still stayed near chance.
6. That suggested a readout bottleneck rather than a policy-conditioning bottleneck.
7. Replacing mean pooling with a GRU over the active-intent prefix finally produced meaningful
   discriminator lift.

This sequence is compelling because it shows:

1. diagnostics mattered
2. architecture mattered
3. not all "latent-variable RL" failures are the same failure

---

## 10. Suggested figures and tables

Use visuals aggressively. This topic benefits from them.

Recommended figures:

1. Architecture diagram
   - environment samples `z`
   - policy uses `e_z`
   - trajectory prefix goes to GRU discriminator
   - bonus goes back into PPO rewards

2. Policy sensitivity curve over training
   - `policy_kl_mean`
   - `action_flip_rate`

3. Discriminator improvement curve
   - `disc_auc_ovr_macro`
   - `disc_top1_acc`

4. Per-intent shot-type table
   - 2PT share
   - 3PT share
   - dunk share

5. Shot-chart comparison
   - older non-intent model
   - newer latent-conditioned model

6. Failure mode diagram
   - two sequences with same mean-pooled features
   - GRU distinguishes them, MLP-mean does not

Optional table:

| Version | Policy sensitivity | Disc AUC | PPP | Interpretation |
| --- | --- | --- | --- | --- |
| No latent usage | near zero | ~0.5 | baseline | latent ignored |
| Stronger policy conditioning only | rising | ~0.5 | mixed | latent affects actions but not coherent trajectories |
| GRU active-prefix discriminator | rising | clearly above 0.5 | TBD | latent-conditioned play structure emerging |

---

## 11. Section on theory: why a latent play space is a useful abstraction

This should be a short theory section, not a full proof section.

Suggested argument:

1. Human basketball naturally uses hierarchical abstraction.
2. At the low level, players execute primitive motions and passes.
3. At a higher level, coaches and players think in coordinated multi-step patterns.
4. A latent play space is a learned analogue of that higher-level abstraction.
5. The abstraction is useful if it supports:
   - behavioral diversity
   - reuse
   - compositionality
   - interpretability

Good line:

"The point of a latent play space is not to imitate human naming conventions, but to give the
policy a reusable internal basis for multi-step behavior."

---

## 12. What to say about offense-defense asymmetry

You should address this directly because it is one of the intellectually interesting parts.

Suggested structure:

1. Offense is conditioned on its own latent play identity.
2. Defense does not get privileged access to the offense latent.
3. This prevents the system from solving defense with information that would be unavailable against
   a human.
4. The design keeps the latent private to the acting side.

Future-looking sentence:

"The same architecture can be extended to defensive latent plays, but the two latent spaces should
remain role-specific and private."

---

## 13. Limitations section

This section is essential.

You should explicitly say:

1. This is not pure DIAYN.
2. This is not yet a hierarchical controller that selects and chains multiple plays per
   possession.
3. The latent is discrete and learned indirectly; it is not a symbolic play representation.
4. A strong discriminator score does not automatically imply human-interpretable basketball
   structure.
5. There is still a tradeoff between task reward and diversity pressure.
6. Some runs can become more structured while becoming less efficient.

That last point is important. It keeps the post honest.

---

## 14. Future-work section

Suggested bullets:

1. Move from a single active intent window to multi-phase or reselected intents.
2. Add a defensive latent play space with private defense conditioning.
3. Explore a full DIAYN pretraining branch before PPO task training.
4. Learn richer trajectory embeddings post hoc for interpretability and retrieval.
5. Replace fixed commitment windows with learned or adaptive play durations.
6. Use the latent play space to constrain search or planning later.

If you mention MCTS, frame it as future work, not present capability.

---

## 15. Suggested closing paragraph

You want a close that is ambitious but not inflated.

Possible structure:

1. Reiterate that the goal is not to hardcode basketball wisdom.
2. Reiterate that the goal is to let play structure emerge from reward plus the right
   representation pressure.
3. Emphasize that the interesting result is not just higher return, but a policy whose behavior can
   be organized into latent multi-step modes.

Possible ending:

"A learned playbook is not a list of named actions. It is a space of reusable multi-step
possibilities. The deeper question is whether an agent can discover that space for itself. This
work suggests that with the right latent conditioning, the right discriminator, and the right
metrics, the answer may be yes."

---

## Technical appendix notes you can turn into prose

Use these if you want a more technical appendix or boxed sidebar.

### A. Current system summary

1. Environment samples offense latent intent `z`.
2. Set observation exposes intent globals to the offense:
   - `intent_index_norm`
   - `intent_active`
   - `intent_visible`
   - `intent_age_norm`
3. Policy looks up learned embedding `e_z`.
4. The set-attention policy uses that embedding to modulate its action computation.
5. The intent is active only for a fixed commitment window.
6. The discriminator processes only the active-intent prefix.
7. The discriminator is a GRU-based classifier over ordered step features.
8. The diversity bonus is added into PPO rollout rewards.

### B. Step feature construction

Each step feature is built from:

1. flattened observation
2. flattened action

Then:

1. step encoder projects it
2. GRU reads the sequence
3. final hidden state is classified into intent ID

### C. Why GRU instead of mean pooling

Mean pooling cannot distinguish:

1. `pass -> cut -> kick -> dunk`
2. `drive -> reset -> pass -> dunk`

if their average per-step features are similar.

A GRU can distinguish them because order changes the hidden-state trajectory.

### D. Why the reward bonus is not "just another loss term"

The discriminator is optimized separately.

Its output affects PPO through reward shaping:

1. train discriminator on completed trajectory prefixes
2. compute per-prefix bonus
3. normalize and clip
4. inject into rollout rewards
5. recompute returns and advantages
6. run normal PPO update

That is conceptually cleaner to explain than saying "we just added another term to the PPO loss,"
because that is not what the implementation does.

---

## Claims that are safe to make

1. "We are learning latent-conditioned multi-step behavior, not hand-authoring a playbook."
2. "The discriminator is DIAYN-inspired, but the training setup is task-conditioned PPO, not pure
   DIAYN."
3. "The first mean-pooled discriminator was too weak because it discarded sequence order."
4. "The GRU discriminator materially improved trajectory separability."
5. "Policy-sensitivity metrics and discriminator metrics answer different questions, and both were
   needed."

---

## Claims to avoid or qualify carefully

1. "The model discovered real basketball plays."
   - Better: "the model learned separable latent behavior families that may be interpretable as
     plays."

2. "The latent embedding is a direct embedding of observed trajectories."
   - Better: "the policy is conditioned on an embedding of a discrete latent play identity."

3. "The discriminator proves semantic interpretability."
   - Better: "the discriminator shows that latent-conditioned trajectories are behaviorally
     separable."

4. "This is DIAYN."
   - Better: "This is DIAYN-inspired."

---

## Optional article subtitles

If you want alternatives:

1. `Why one-step reinforcement learning misses the real structure of basketball`
2. `From primitive actions to latent multi-step behavior`
3. `How a GRU discriminator turned a dead latent into a readable play space`

---

## Fast drafting checklist

Before publishing, make sure the post does all of the following:

1. defines what a "play" means in this project
2. explains why a latent alone is not enough
3. explains the DIAYN connection accurately
4. explains why mean pooling failed
5. explains why the GRU active-prefix discriminator is better
6. distinguishes policy sensitivity from trajectory separability
7. includes at least one metric chart and one behavioral visual
8. includes a limitations section
9. avoids claiming the model has solved basketball strategy

