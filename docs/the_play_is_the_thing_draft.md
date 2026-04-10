# The Play Is The Thing

_How structured starts, intent-conditioned control, and a hard lesson about shortcut features finally produced the first evidence of learned basketball plays_

> Draft note:
> This is written as a publishable first draft, not just a bullet outline.
> The safest claim level is: **emerging play families**, not “solved play learning.”

## Optional Subtitle

What changed when we stopped asking reinforcement learning to invent basketball structure from random chaos.

## Suggested Hero

Use either:

- a 4-up collage of the strongest playbook trajectory screenshots, or
- one playbook image paired with one holdout latent plot.

Suggested local sources:

- `/home/evanzamir/Downloads/horns_turbo_spiral.png`
- `/home/evanzamir/Downloads/horns_jet_snap.png`
- `/home/evanzamir/Downloads/horns_comet_screen.png`
- `/home/evanzamir/Downloads/horns_cinder_pin.png`
- `/home/evanzamir/Downloads/disc_eval_batch_tsne (5).png`

## Opening

For a long time, our basketball agents could do interesting things without doing anything I would call a play.

They could move. They could score. They could discover useful local habits. But if you asked the harder question, “is this system actually learning reusable offensive structure?”, the honest answer was no.

There was no reason to believe the model had learned anything like a coordinated action family. It looked more like end-to-end control stumbling into decent possessions.

That changed only after we stopped treating play learning like generic reinforcement learning and started building a system that gave structure a chance to appear.

The phrase that ended up feeling right was this:

> The play is the thing.

Not the score alone. Not the classifier metric alone. Not the visualization alone. The thing that matters is whether the model organizes coordinated, role-dependent, reusable patterns of behavior that look like actual basketball actions.

This post is about the architecture and debugging work that finally started producing evidence of that.

## The Old Setup Was Too Underspecified

Our older baseline trained a policy directly from random spawns with no explicit play-selection mechanism and no learned intent discriminator.

That setup had one big virtue: it was cheap. It moved fast. At a given wall-clock budget, it let us chew through a lot of environment steps.

But it also had a major weakness: it asked the model to discover all higher-level structure from scratch in a state distribution that was too broad and too noisy.

In practice that meant:

- random starting states
- no reusable start-of-possession structure
- no explicit latent play variable
- no incentive for consistent role-specialized behavior
- weak tools for telling whether the agent had actually learned a play or just a bag of locally useful reactions

The result was competent behavior, but not organized offense.

## The New Stack

We introduced three structural ideas:

1. **Starting templates**
2. **Intent-conditioned low-level policy execution**
3. **A separate intent discriminator plus selector**

At a high level, the system now works like this:

- a possession can start from a structured offensive template instead of a fully random spawn
- a high-level selector can choose an intent `z`
- the low-level set-attention policy is explicitly conditioned on that intent
- a separate discriminator learns to distinguish intent-conditioned behavior and supplies a DIAYN-style shaping bonus

That gave the agent a way to learn not just “what action is good here?” but also “what kind of possession am I trying to run?”

For the detailed diagrams, see [current_model_architecture.md](/home/evanzamir/basketworld/docs/current_model_architecture.md).

## We Found a Shortcut Before We Found a Play

The first major breakthrough was not a success. It was a failure.

We noticed that discriminator AUC was becoming strong, but the visualizations and behavior were still not convincing. That raised the obvious suspicion: the discriminator might be solving the wrong problem.

That suspicion was correct.

Post-hoc ablations on saved discriminator holdout batches showed that the discriminator was leaning heavily on temporal global features, especially:

- `shot_clock`
- `pressure_exposure`

These are easy signals. They correlate with where a possession is in its trajectory. If you let a discriminator use them, it can look smart without learning anything like play identity.

That was the turning point. Once we knew the discriminator was taking a shortcut, the next step was clear: remove those features from the discriminator input path and rebuild the evaluation.

This was also a useful negative result because it sharpened the scientific question. The issue was not “the model does not learn anything.” The issue was “the current objective is rewarding the wrong thing.”

## We Tightened the Evaluation Too

Fixing the feature leak was necessary, but not sufficient.

We also changed the holdout construction. The earlier split operated at the individual-step level, which made it too easy for train and holdout to contain nearly identical nearby states from the same possession.

So we moved to an **episode-level holdout**:

- entire possessions go to train or holdout
- neighboring steps from the same sequence do not get split across both

That made the discriminator metrics more honest.

And importantly, the system still held up after the change.

## Figure 1: Holdout Curves

Suggested local images:

- `/home/evanzamir/Downloads/intent_disc_auc_ovr_macro_holdout (11).png`
- `/home/evanzamir/Downloads/intent_disc_top1_acc_holdout (5).png`

Suggested caption:

> After removing the temporal shortcut features and moving to an episode-level holdout, the discriminator still learns a strong intent signal. Macro AUC rises above 0.9, while top-1 accuracy climbs far above chance on an 8-way task.

Suggested embed for local preview:

```md
![Holdout AUC](/home/evanzamir/Downloads/intent_disc_auc_ovr_macro_holdout%20(11).png)
![Holdout Top-1](/home/evanzamir/Downloads/intent_disc_top1_acc_holdout%20(5).png)
```

Why this matters:

- AUC rose first, meaning the model learned ranking signal before it learned sharp exact classification.
- Top-1 rose later, which suggests the latent structure eventually became good enough to support much cleaner intent separation.

In other words, the model was not just learning a vague manifold. It was starting to learn identifiable intent-conditioned modes.

## The Templates Were Not the Shortcut

Once the temporal shortcut was removed, another concern remained: maybe the latent was just clustering by starting template.

That would have been disappointing for a different reason. It would mean the model had learned to identify the setup, not the play.

So we added template provenance to the saved holdout batch and plotted the learned discriminator latent both:

- colored by intent
- colored by start template

The result was encouraging:

- intent structure became visible
- templates were broadly mixed across the manifold
- the latent did **not** collapse into one cluster per template

That does not mean templates do nothing. They clearly matter as curriculum and context. But they were not simply swallowing the representation.

## Figure 2: Holdout Latent Structure

Suggested local images:

- `/home/evanzamir/Downloads/disc_eval_batch_tsne (5).png`
- `/home/evanzamir/Downloads/disc_eval_batch_pca (2).png`

Suggested caption:

> On a held-out discriminator batch, several strong intent families separate in latent space while weaker intents remain partially entangled. Template-colored views showed that start templates were mixed across the manifold rather than trivially owning separate regions.

Suggested embed for local preview:

```md
![Held-out t-SNE](/home/evanzamir/Downloads/disc_eval_batch_tsne%20(5).png)
![Held-out PCA](/home/evanzamir/Downloads/disc_eval_batch_pca%20(2).png)
```

At this point the evidence started to line up in a more interesting way:

- the discriminator holdout metrics were strong
- the latent plots showed real structure
- templates were not the trivial explanation
- and the selector was increasingly favoring the same strong intents that separated in latent space

That combination is much harder to dismiss than any single chart alone.

## The Most Important Result: We Started Seeing Plays

Metrics were useful, but they were not enough. What mattered was whether behavior looked like a play.

The first convincing evidence came from the Playbook tool using fixed starting templates and pinned intents.

For the same template, different intent labels began to produce:

- different primary routes
- different first-shot timing
- different primary-shooter distributions
- different turnover profiles
- different supporting-player assignments

That is the point where this stopped looking like classifier games and started looking like offensive structure.

For `empty_corner_dunker`, the strongest intents produced visibly different `P1` route families. Some looked like quick-hitting primary actions. Others looked slower, more read-based actions with secondary involvement.

For `double_high_horns`, the evidence got stronger. `P2`, an off-ball player, took clearly different jobs depending on the intent:

- weak-side spacing / hold behavior
- top lift / cut behavior
- interior diagonal movement
- lower-lane curl / baseline-to-slot behavior

That matters because it shows multi-player coordination, not just one scorer taking different shots.

## Figure 3: Same Template, Different Plays

Suggested local images:

- `/home/evanzamir/Downloads/horns_turbo_spiral.png`
- `/home/evanzamir/Downloads/horns_jet_snap.png`
- `/home/evanzamir/Downloads/horns_comet_screen.png`
- `/home/evanzamir/Downloads/horns_cinder_pin.png`

Suggested caption:

> Same starting template, same tracked player, different intent labels. The route families are meaningfully different, including different off-ball role assignments. This is the strongest evidence so far that the system is beginning to learn actual plays rather than generic scoring behavior.

Suggested embed for local preview:

```md
![Double High Horns - Turbo Spiral](/home/evanzamir/Downloads/horns_turbo_spiral.png)
![Double High Horns - Jet Snap](/home/evanzamir/Downloads/horns_jet_snap.png)
![Double High Horns - Comet Screen](/home/evanzamir/Downloads/horns_comet_screen.png)
![Double High Horns - Cinder Pin](/home/evanzamir/Downloads/horns_cinder_pin.png)
```

If I had to summarize the most important behavioral finding in one sentence, it would be this:

> When the off-ball players start doing different jobs by intent, you are no longer just training a scorer. You are training offense.

## A Useful Surprise: Better Behavior Despite Slower Training

One tradeoff here is obvious. The new system is slower.

Compared with the older “pure policy + random spawn” setup, we now pay for:

- template handling
- selector logic
- discriminator training
- more diagnostics and saved evaluation artifacts

But the surprising part is that the extra structure seems to be worth it.

At the same nominal step count, the current model feels much smarter than the older baseline ever did. Even before multiselect, even before more compute, the policy is behaving on a more organized manifold.

That is the core trade:

- less raw throughput
- much better behavioral structure

In this domain, that seems like the right trade.

## What the System Has Probably Learned So Far

I do not think the honest claim is “the model learned eight clean plays.”

The more defensible claim is:

- the system currently seems to have learned several strong play families
- some of the eight intents are clearly more meaningful than others
- the effective vocabulary is smaller than the nominal latent size

That is not a failure. It is what you would expect from a system that is just beginning to organize behavior at this level.

The important thing is that the strong modes now appear to be **real**, in the sense that they show up simultaneously in:

- discriminator holdout metrics
- latent visualizations
- selector preferences
- and actual role-structured playbook trajectories

That combination is much harder to dismiss than any single metric alone.

## What Comes Next

The next major experiment is probably **multiselect**.

Right now the selector chooses once at the beginning of a possession. That is already useful, but it limits the selector to something like ex-ante play calling.

Multiselect would let the selector choose again after the play starts to unfold, which could enable:

- chained actions
- counters after a pass
- more adaptive sequencing inside a possession

But it also makes the system harder to interpret.

So the current phase has been about getting the single-select version to the point where it clearly means something. Only after that does it make sense to ask for more.

## A Careful Summary

If I had to summarize the project honestly right now, I would say this:

We are not done. We have not solved play learning. We do not yet have a clean, balanced vocabulary of eight fully distinct offensive schemes.

But for the first time, we have credible evidence that the model is beginning to learn coordinated, reusable, intent-conditioned offensive structure.

And once that starts to happen, the research question changes.

It is no longer “can we get anything play-like at all?”

Now it is:

> how far can we push structured play learning once the system finally has something real to build on?

## Closing Line Options

Option 1:

The play is the thing. Everything else is just how we learned to see it.

Option 2:

The real milestone was not a prettier chart. It was the moment the trajectories started looking like assignments.

Option 3:

When the off-ball players start behaving differently by intent, you are no longer just training a scorer. You are training offense.

## Editor Notes

If you publish this, I would recommend:

1. keep the claims disciplined
2. keep one section on the shortcut failure and ablation fix
3. use 2-4 images total, not all of them
4. show one holdout metric figure and one playbook figure at minimum
5. mention multiselect only as the next step, not as part of the current result
6. before publishing externally, copy the images out of `Downloads/` into a stable repo or blog assets directory
