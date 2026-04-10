# The Play Is The Thing
## How BasketWorld started to learn something that actually looks like offense
---
### The question I actually care about

For a while now I've had a kind of private benchmark in my head for BasketWorld that is more demanding than simply asking whether the agents score points. It's not enough for an offense to become "good" in some abstract reinforcement learning sense. It's not enough for the reward curve to go up. It's not enough for the animations to look energetic. What I really want to know is: **is the system learning anything that deserves to be called a play?**

That is a much harder question than it sounds. A basketball possession can look competent without being organized. Players can move with purpose without the underlying policy having discovered reusable structure. In earlier versions of BasketWorld I often felt like I was seeing fragments of basketball logic, but not the thing itself. The offense could drive, shoot, occasionally pass, and generally survive in the environment, but it still felt like end-to-end control groping its way toward local solutions rather than discovering a family of coordinated actions.

Recently that started to change.

I do not want to oversell it. I do not think I have "solved play learning". But for the first time I think I have credible evidence that BasketWorld is beginning to learn **emerging play families** rather than just generic scoring behavior.

And the road to getting there was not what I expected.

### Why the older setup hit a ceiling

The older versions of BasketWorld asked a lot from pure policy learning. The setup was relatively simple: random spawns, no explicit selector, no learned intent discriminator, and no real start-of-possession structure beyond whatever the environment happened to generate. That had one obvious advantage: it was cheap. Training moved quickly. At a fixed wall-clock budget I could chew through a lot of environment steps.

But the downside was equally obvious in hindsight. I was effectively asking the model to discover all higher-level offensive structure from a very broad and noisy state distribution. That's a tall order. If you give a policy random starts, sparse strategic cues, and no explicit mechanism for separating one kind of possession from another, it is much more likely to learn useful local habits than anything resembling a reusable set action.

To put it differently, I was hoping the agents would invent offense while giving them very little reason to organize themselves offensively.

So the newer architecture introduced three ingredients that were missing before:

* structured **starting templates**
* an **intent-conditioned** low-level policy
* a separate **selector + discriminator** stack to encourage distinct intent-conditioned behavior

The basic idea is straightforward. A possession can begin from a recognizable starting shape. A selector can choose a latent intent `z`. The low-level set-attention policy is explicitly conditioned on that intent. And a discriminator tries to tell those intent-conditioned behaviors apart, feeding back a DIAYN-style shaping signal.

In principle that gives the system a chance to learn not just *what action is good here?* but something closer to *what kind of possession are we running?*

Of course, the phrase "in principle" is doing a lot of work there.

### The first breakthrough was discovering a failure mode

What finally moved the project forward was not the moment a metric got high. It was the moment I realized a metric was lying to me.

I had discriminator AUC climbing nicely, which at first looked promising. But the latent visualizations were not very convincing and the behavior itself was not yet screaming "play". That mismatch made me suspicious. If the discriminator was becoming strong, what exactly was it learning?

The answer turned out to be: not the thing I wanted.

After building some post-hoc ablation tools and scoring saved holdout batches directly, I found that the step-based discriminator had learned to lean heavily on temporal global features, especially:

* `shot_clock`
* `pressure_exposure`

That is a classic shortcut. Those variables tell the discriminator a lot about *where a possession is in time* without forcing it to learn *what kind of possession this is*. In hindsight this makes perfect sense. If you hand a model an easy temporal clue, it will happily use it.

This was a painful but useful lesson. The issue was not that the architecture was incapable of learning anything interesting. The issue was that I had made the objective too easy to hack.

So I removed those globals from the discriminator input path and tightened the evaluation procedure.

### Making the evaluation harder, not easier

Removing shortcut features was only half the fix. The evaluation itself also needed to be more honest.

Originally, the discriminator holdout construction worked at the individual-step level. That meant train and holdout could contain nearby steps from the same possession, which is exactly the kind of leakage that makes a model look cleaner than it really is. So I changed the split to an **episode-level holdout**: entire possessions go to train or holdout, and neighboring steps from the same trajectory no longer get divided across both.

That made the discriminator's job harder. Which is precisely why I wanted it.

And importantly, the model still held up.

![Holdout Top-1](/home/evanzamir/Downloads/intent_disc_top1_acc_holdout%20(5).png)

![Holdout AUC](/home/evanzamir/Downloads/intent_disc_auc_ovr_macro_holdout%20(11).png)

What I like about these curves is not just that they go up. It's *how* they go up. The macro one-vs-rest AUC improves first, while top-1 accuracy lags behind and only later starts to climb aggressively. That suggests the discriminator learns broad ranking structure before it learns sharp class boundaries. In other words, it first gets a sense that different possession types exist, and only later gets better at naming them cleanly.

This matters because it feels much less like a cheap classifier trick and much more like a representation actually beginning to organize itself.

### The templates were not the shortcut either

Once the temporal leak was gone, I had another worry. Maybe the latent wasn't learning intent structure at all. Maybe it was just clustering by start template.

That would have been disappointing for a different reason. It would mean the system had learned to identify the setup rather than the play.

So I added template provenance to the saved holdout batches and plotted the discriminator latent in two ways: once colored by intent, and once colored by starting template. The result was encouraging. The intent-colored views showed clear structure, but the template-colored views were broadly mixed across the same manifold. The templates were helping as curriculum and context, but they were not simply swallowing the representation.

![Held-out t-SNE](/home/evanzamir/Downloads/disc_eval_batch_tsne%20(5).png)

![Held-out PCA](/home/evanzamir/Downloads/disc_eval_batch_pca%20(2).png)

This was one of those moments where several lines of evidence finally started pointing in the same direction:

* holdout AUC and top-1 were strong
* the held-out latent had visible structure
* the structure was not trivially explained by start template
* and the selector was increasingly preferring the same intents that seemed to separate cleanly in latent space

That does not prove that every intent corresponds to a perfect play family. But it is much harder to dismiss than a single chart in isolation.

### Going to the tape

Still, none of that was enough by itself. As I wrote in my previous post, reinforcement learning has a way of making you fluent in metrics while leaving you half-blind about the thing you actually care about. At some point you have to go to the tape.

This is where the Playbook tooling became indispensable. By pinning the starting template and the intent label, I could inspect actual player trajectories and stop pretending that a scalar metric alone was going to answer the basketball question.

The first really convincing results came from the `empty_corner_dunker` template. For the same starting setup, different intents produced different `P1` route families, different first-shot timing, different primary shooter distributions, and different turnover profiles. Some looked like quick-hitting actions. Others looked slower and more read-based. That was already encouraging.

But what really made me sit up was what happened with the `double_high_horns` template.

I looked at `P2`, an off-ball player, across four different pinned intents. And instead of seeing minor random variation around the same job, I saw clearly different assignments. One intent kept `P2` mostly in a spacing/hold role. Another sent `P2` on a top lift or cut. Another pushed `P2` through a more interior diagonal route. Another looked more like a lower-lane curl or baseline-to-slot movement.

That matters because once the *off-ball* players are doing systematically different jobs by intent, the system is no longer just varying how one scorer gets to the rim. It is starting to organize offense.

![Double High Horns - Turbo Spiral](/home/evanzamir/Downloads/horns_turbo_spiral.png)

![Double High Horns - Jet Snap](/home/evanzamir/Downloads/horns_jet_snap.png)

![Double High Horns - Comet Screen](/home/evanzamir/Downloads/horns_comet_screen.png)

![Double High Horns - Cinder Pin](/home/evanzamir/Downloads/horns_cinder_pin.png)

This is the point where the title of this post started feeling earned.

The play is the thing.

Not the chart. Not the latent manifold. Not the accuracy number. Those things matter, but only because they eventually line up with the behavioral evidence. The real milestone is when the trajectories begin to look like roles and assignments rather than just movement.

### A note on compute, because it matters

One of the tradeoffs here is obvious. The newer architecture is slower. Structured templates, selector logic, discriminator training, better evaluation, and better logging all cost time. In fact one of the things I had to debug recently was a brutal slowdown right when selector training turned on. After throttling how often the selector updates and capping selector samples per update, the per-update time dropped from roughly 36 seconds back down to around 10 or 11.

That said, I would still take the current system over the older pure-policy baseline in a heartbeat.

At 50 million steps, this model feels much smarter than one of my older pre-selector, pre-discriminator models did at the same nominal step count. That is a very important point. What I have lost in raw throughput, I seem to be gaining back in **sample usefulness**. The agent is spending its experience budget on a much better-shaped problem.

So the trade now looks something like this:

* less raw throughput
* much better behavioral organization

And for this problem, that seems like the correct trade.

### What I think BasketWorld has learned so far

I do **not** think the honest claim is that BasketWorld has learned eight perfectly distinct plays.

What I do think is this:

* the system appears to have learned several strong play families
* some of the nominal intents are much more meaningful than others
* the effective vocabulary is probably smaller than the nominal latent size right now

That is not a failure. If anything, it is exactly what I would expect from a system that is only beginning to organize behavior at this level. The important thing is that the strongest modes now appear to be real in a way they were not before. They show up simultaneously in discriminator holdout metrics, latent visualizations, selector preferences, and the actual multi-player trajectories shown in the Playbook tool.

That combination is a lot harder to wave away as noise.

### What comes next

The obvious next experiment is **multiselect**.

Right now the selector chooses once at the beginning of a possession. That is already useful, but it still makes the selector more like a play caller than a fully reactive strategist. Multiselect would let the selector choose again once the possession starts to unfold, which should make it possible to chain actions, respond to the defense, and sequence counters more naturally.

But multiselect also makes the system harder to interpret, which is exactly why I did **not** want to jump there too early. I wanted the single-select version to mean something first. I wanted to see credible evidence that the system could learn a play family before asking it to learn a full play sequence.

I think BasketWorld is finally at that point.

### Closing thought

If I had to summarize where the project stands right now, I would put it this way:

I have not solved play learning.

But for the first time, I think I have a system that is beginning to learn coordinated, reusable, intent-conditioned offensive structure. And once that happens, the research question changes.

It is no longer *can we get anything play-like at all?*

Now it is:

> how far can we push structured play learning once the system finally has something real to build on?

### Technical Details

For those interested in the technical details, the current BasketWorld stack uses a set-based observation wrapper, an intent-conditioned set-attention policy, an integrated selector, and a step-based intent discriminator with DIAYN-style shaping. Training and experiment tracking are built around Gymnasium, Stable-Baselines3 PPO, and MLflow. I also wrote up a more schematic architecture overview in [current_model_architecture.md](/home/evanzamir/basketworld/docs/current_model_architecture.md).

The project is open source on github under a MIT license. If you like the project please support it by giving it a star. And, as always, I'd love your feedback because most of this project has been built by chasing questions that emerged only after I could actually watch the agents play.

https://github.com/EvanZ/basketworld
