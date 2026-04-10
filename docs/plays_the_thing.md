Short version:

### High-level stack

We currently use an **integrated, hierarchical DIAYN-style setup**, not a separate “pretrain skills first, then finetune” pipeline.

1. **Starting templates**

- possessions begin from structured formations to reduce random chaos

2. **Selector / coach**

- a selector chooses an intent `z` at possession start
- current mainline is still effectively **single-select**
- this is the high-level “play call”

3. **Intent-conditioned low-level policy**

- the actual action policy is:
- `π_θ(a_t | s_t, z)`
- implemented by adding a learned intent embedding into the set-attention token stream before self-attention

4. **Step discriminator**

- a separate discriminator tries to predict the current intent from active offense states:
- `q_φ(z | x_t)`
- where `x_t` is the step observation used for the discriminator

5. **DIAYN-style intrinsic bonus**

- if the discriminator can identify the intent from behavior, the policy gets bonus reward
- that pushes different intents to produce distinguishable trajectories

### Main separation equation

The core DIAYN objective is to maximize mutual information between the latent intent and visited states:

\[
I(Z;X) \approx \mathbb{E}[\log q_\phi(z \mid x) - \log p(z)]
\]

With a uniform prior over `K` intents, that becomes:

\[
\mathbb{E}[\log q_\phi(z \mid x) + \log K]
\]

That is the main “separation” signal.

### What we actually inject into training

In the current implementation, the raw discriminator bonus is:

\[
b_{\text{raw}}(x_t, z) = \log q_\phi(z \mid x_t) + \log K
\]

Then we normalize and clip it before adding it to PPO reward:

\[

r_t^ {\text{DIAYN}}=

_\beta \cdot
\mathrm{clip}
\left(
\frac{b_{\text{raw}} - \mu}{\sigma},
-c, c
\right)
\]

where:

- `β` = current intent bonus scale
- `μ, σ` = running mean/std of raw bonus
- `c` = bonus clip threshold

### One important current nuance

This is **not** “pure DIAYN” in the original paper’s sense.

We currently do:

- **joint training**
- selector + low-level policy + discriminator all in the same overall training loop

rather than:

- learn skills first
- freeze them
- learn a downstream controller later

### One more useful equation

At the policy level, intent conditioning is approximately:

\[
\text {token}_i\[
\text {token}_i

\text{token\_mlp}(player_i, globals)
+
gate \cdot W_{\text{role}} e(z)
\]

So the intent changes the token embeddings before attention, which then changes downstream actions.

If you want, I can also turn this into a blog-ready paragraph version instead of equations-and-bullets.
