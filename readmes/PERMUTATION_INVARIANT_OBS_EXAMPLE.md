# Permutation-Invariant Observation Example (3v3)

This note shows:
1) an explicit example of the *current* ordered observation vector for a single 3v3 state, and
2) one way to convert that same state into a permutation-invariant representation using shared
embeddings + pooling.

All numbers below are from a concrete, hand-picked 3v3 state in the current env
(`HexagonBasketballEnv(players=3, grid_size=16)` with `include_hoop_vector=True`,
`normalize_obs=True`).

## Example State

- Court: 16x16 (normalization denominator = 16)
- Basket axial: (-4, 8)  -> normalized (-0.25, 0.5)
- Offense ids: [0, 1, 2]
- Defense ids: [3, 4, 5]
- Ball holder: player 1
- Shot clock: 17
- Player axial positions (q, r):
  - O0 (id 0): (2, 8)
  - O1 (id 1): (4, 6)
  - O2 (id 2): (2, 10)
  - D3 (id 3): (2, 7)
  - D4 (id 4): (0, 9)
  - D5 (id 5): (-1, 8)

## Current Ordered Observation (Length = 80)

Each section below is appended in order in `basketworld/envs/core/observations.py`.

1) Player positions (q,r) normalized, ordered by player id:
```
[0.1250, 0.5000,   0.2500, 0.3750,   0.1250, 0.6250,
 0.1250, 0.4375,   0.0000, 0.5625,  -0.0625, 0.5000]
```

2) Ball holder one-hot (len=6):
```
[0, 1, 0, 0, 0, 0]
```

3) Shot clock (raw):
```
17
```

4) Team encoding (+1 offense, -1 defense):
```
[ 1, 1, 1, -1, -1, -1 ]
```

5) Ball handler position (normalized q,r):
```
[0.2500, 0.3750]
```

6) Hoop vector (normalized q,r):
```
[-0.2500, 0.5000]
```

7) All-pairs offense-defense distances (O0..O2 x D3..D5, 9 values):
```
[0.0625, 0.1250, 0.1875,
 0.1250, 0.2500, 0.3125,
 0.1875, 0.1875, 0.3125]
```

8) All-pairs offense-defense signed angles (normalized to [-1, 1] by dividing by pi):
```
[ 0.3333, -0.1667,  0.0000,
 -0.0895, -0.1789, -0.0529,
  0.2561,  0.0289,  0.0529]
```

9) Teammate distances (unordered pairs per team):
- Offense pairs: (0,1), (0,2), (1,2)
- Defense pairs: (3,4), (3,5), (4,5)
```
[0.1250, 0.1250, 0.2500, 0.1250, 0.1875, 0.1250]
```

10) Teammate signed angles (ordered pairs per team):
- Offense: (0,1), (0,2), (1,0), (1,2), (2,0), (2,1)
- Defense: (3,4), (3,5), (4,3), (4,5), (5,3), (5,4)
```
[ 0.6667, -0.6667, -0.2561, -0.4228,  0.2561,  0.4228,
 -0.2836, -0.0564,  0.6061,  0.1061,  0.8939, -0.8333]
```

11) Lane steps (per player, len=6):
```
[0, 0, 0, 0, 0, 0]
```

12) Expected points (offense only, len=3):
```
[0.6060, 0.1725, 0.1779]
```

13) Turnover probs (offense only, len=3):
```
[0, 0, 0]
```

14) Steal risks (offense only, len=3):
```
[0.0422, 0.0000, 0.0050]
```

## No Pooling: Per-Teammate Embeddings + Globals

This keeps *individual* teammate states (no pooling). For a centralized policy,
you can feed each player:

```
player_input =
  [self_features,
   teammate_embeddings (one vector per teammate, no pooling),
   global_context]
```

Use the same per-player feature definition as below:
```
[q_norm, r_norm, role(+1/-1), has_ball, dist_to_ball_norm, dist_to_hoop_norm]
```

### Example for the ball handler (P1)

Self features:
```
P1: [ 0.2500, 0.3750, 1, 1, 0.0000, 0.5000]
```

Teammate embeddings (ordered by teammate id, no pooling):
```
P0: [ 0.1250, 0.5000, 1, 0, 0.1250, 0.3750]
P2: [ 0.1250, 0.6250, 1, 0, 0.2500, 0.5000]
```

Global context (example):
```
shot_clock: 17
hoop: [-0.2500, 0.5000]
```

Flat input (no pooling):
```
[ 0.2500, 0.3750, 1.0000, 1.0000, 0.0000, 0.5000,
  0.1250, 0.5000, 1.0000, 0.0000, 0.1250, 0.3750,
  0.1250, 0.6250, 1.0000, 0.0000, 0.2500, 0.5000,
  17.0000, -0.2500, 0.5000]
```

Notes:
- This preserves per-teammate state without pooling.
- You can append opponent embeddings the same way (one vector per opponent) if needed.
- Ordering is fixed here (by id) only so the shape is stable.

### No-pooling set encoder (order-agnostic) with explicit math

To remove slot-specific weights, use a shared encoder per teammate and a
permutation-equivariant block (e.g., self-attention without positional encodings).
Below is a concrete numeric example using a tiny "toy" encoder so the math is visible.

**Step 1: Build tokens (self + teammates), appending globals to every token**

We concatenate `[self_or_teammate_features, shot_clock, hoop]`.

```
x_self = [0.2500, 0.3750, 1, 1, 0.0000, 0.5000, 17.0000, -0.2500, 0.5000]
x_P0   = [0.1250, 0.5000, 1, 0, 0.1250, 0.3750, 17.0000, -0.2500, 0.5000]
x_P2   = [0.1250, 0.6250, 1, 0, 0.2500, 0.5000, 17.0000, -0.2500, 0.5000]
```

**Step 2: Shared encoder f(x) -> e (same weights for every token)**

For a toy, let `f(x)` just keep `[q_norm, r_norm]` so we can compute by hand.
This is only for illustration.

```
e_self = [0.2500, 0.3750]
e_P0   = [0.1250, 0.5000]
e_P2   = [0.1250, 0.6250]
```

**Step 3: Self-attention for the self token (no positional enc)**

Use dot-product attention with `Q=K=V=identity` (toy). Scores:

```
score_self = e_self · e_self = 0.203125
score_P0   = e_self · e_P0   = 0.218750
score_P2   = e_self · e_P2   = 0.265625
```

Softmax weights (approx):
```
w_self = 0.3246
w_P0   = 0.3297
w_P2   = 0.3457
```

Context for self (weighted sum):
```
context = w_self*e_self + w_P0*e_P0 + w_P2*e_P2
        = [0.1630, 0.5130]  (approx)
```

**Key property (order-agnostic):**
If you swap P0 and P2 in the input list, the scores and weights swap too,
but the weighted sum is identical. That means the self output is invariant
to teammate order even though you kept per-teammate embeddings and did not pool.

In a real model, `f` is a learned MLP and attention has learned projections
and multiple heads, but the permutation-equivariant property holds as long as
there are no positional encodings or index-specific parameters.

## SB3 + PPO: End-to-End Set Policy (No Pooling)

This section shows how the set approach plugs into SB3 + PPO without pooling,
while still producing concrete actions.

### 1) Observation structure (dict via wrapper)

Instead of a flat vector, expose a dict for the policy:
```
obs = {
  "self":      shape (F,),
  "teammates": shape (T, F),
  "opponents": shape (O, F),
  "globals":   shape (G,),
}
```

Example features (same as above):
```
F = [q_norm, r_norm, role(+1/-1), has_ball, dist_to_ball_norm, dist_to_hoop_norm]
G = [shot_clock, hoop_q_norm, hoop_r_norm]
```

From the example state:
```
self = [0.2500, 0.3750, 1, 1, 0.0000, 0.5000]
teammates =
  [[0.1250, 0.5000, 1, 0, 0.1250, 0.3750],
   [0.1250, 0.6250, 1, 0, 0.2500, 0.5000]]
opponents =
  [[0.1250, 0.4375, -1, 0, 0.1250, 0.3750],
   [0.0000, 0.5625, -1, 0, 0.2500, 0.3125],
   [-0.0625, 0.5000, -1, 0, 0.3125, 0.1875]]
globals = [17.0, -0.2500, 0.5000]
```

### 2) Token construction (no pooling)

Append globals to every token:
```
token_self = concat(self, globals)         # shape (F+G,)
token_tm   = concat(teammate_i, globals)   # shape (T, F+G)
token_op   = concat(opponent_i, globals)   # shape (O, F+G)
tokens = [token_self] + token_tm + token_op
```

For 3v3, tokens = 1 + 2 + 3 = 6 tokens.

### 3) Shared encoder + set transformer

Use a shared MLP `f` on each token, then a permutation-equivariant block
(attention without positional encodings):
```
E = f(tokens)          # shape (N_tokens, D)
Z = SetAttn(E)         # shape (N_tokens, D)  (order-agnostic)
z_self = Z[0]          # self token output
```

Because there are no positional encodings, reordering teammate tokens does not
change `z_self`. You still keep individual teammate states inside the attention.

Self-token only means: the action head reads *only* the output embedding
for the controlled player (that player's token). Teammates still influence it
through attention, but you do not concatenate all tokens into one vector.

### 4) Action head (concrete SB3 MultiDiscrete output)

In this env, actions are per-player with a shared action set
(e.g., MOVE_*, SHOOT, PASS_*). For PPO in SB3:

```
logits_all = action_head(Z)   # shape (N_players, N_actions), shared head
logits_flat = logits_all.reshape(-1)  # SB3 MultiCategorical expects flat
```

If training offense only, take `logits_all[offense_ids]` and fill defense
actions from the opponent policy (self-play wrapper).

### 5) "Which teammate is which token" for passing?

The model does not rely on a fixed teammate slot. Instead, each teammate token
contains its own *state* (position, distances, etc). The policy can learn to
pass based on *geometry*:

- With the current action space (PASS_E, PASS_NE, ...), it only needs to decide
  a *direction*, so a set representation is enough.
- If you ever change to "pass to teammate X", use a pointer head:
  `pass_target = softmax(attn_scores_over_teammate_tokens)` and map to that id.

The key is: identity is encoded by the teammate's features, not by index.

### 6) SB3 wiring (high-level)

- Wrap env to emit the dict observation above.
- Create a custom `BaseFeaturesExtractor` that implements steps 2-3 and returns
  `Z` (or just `z_self`).
- Override the policy to use a shared action head on tokens (for MultiDiscrete).

This is fully compatible with PPO in SB3, but it requires a custom policy
because the default MLP expects a flat vector, not a set of tokens.

### 7) Minimal SB3 policy skeleton (no flattening of features)

This is a compact, end-to-end sketch. It is not drop-in, but the flow is complete.
Note: the *features* stay as tokens (B, N, D). We only flatten logits because
SB3's MultiCategorical expects a flat vector.

```python
import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy


class SetObsWrapper(gym.ObservationWrapper):
    def __init__(self, env, players_per_side, n_players):
        super().__init__(env)
        self.players_per_side = players_per_side
        self.n_players = n_players
        self.F = 6
        self.G = 3
        self.observation_space = spaces.Dict(
            {
                "players": spaces.Box(-np.inf, np.inf, (n_players, self.F), np.float32),
                "globals": spaces.Box(-np.inf, np.inf, (self.G,), np.float32),
            }
        )

    def observation(self, obs):
        env = self.env.unwrapped
        norm_den = float(max(env.court_width, env.court_height)) or 1.0
        if not env.normalize_obs:
            norm_den = 1.0
        players = []
        for pid in range(env.n_players):
            q, r = env.positions[pid]
            role = 1.0 if pid in env.offense_ids else -1.0
            has_ball = 1.0 if env.ball_holder == pid else 0.0
            dist_ball = 0.0 if env.ball_holder is None else float(
                env._hex_distance(env.positions[pid], env.positions[env.ball_holder])
            )
            dist_hoop = float(env._hex_distance(env.positions[pid], env.basket_position))
            players.append(
                [
                    q / norm_den,
                    r / norm_den,
                    role,
                    has_ball,
                    dist_ball / norm_den,
                    dist_hoop / norm_den,
                ]
            )
        globals_vec = np.array(
            [float(env.shot_clock), env.basket_position[0] / norm_den, env.basket_position[1] / norm_den],
            dtype=np.float32,
        )
        return {"players": np.array(players, dtype=np.float32), "globals": globals_vec}


class SetEncoder(nn.Module):
    def __init__(self, n_players, F, G, embed_dim=64, n_heads=4):
        super().__init__()
        self.n_players = int(n_players)
        self.embed_dim = int(embed_dim)
        self.num_cls_tokens = 2
        self.token_mlp = nn.Sequential(
            nn.Linear(F + G, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.cls_tokens = nn.Parameter(th.zeros(self.num_cls_tokens, embed_dim))

    def forward(self, players, globals_vec):
        g = globals_vec.unsqueeze(1).expand(-1, players.size(1), -1)
        tokens = th.cat([players, g], dim=-1)  # (B, N, F+G)
        E = self.token_mlp(tokens)             # (B, N, D)
        cls = self.cls_tokens.unsqueeze(0).expand(E.size(0), -1, -1)
        E = th.cat([E, cls], dim=1)            # append CLS tokens
        Z, _ = self.attn(E, E, E, need_weights=False)
        return Z                               # (B, N, D)


class SetPolicy(ActorCriticPolicy):
    def __init__(self, *args, n_players, n_actions, embed_dim=64, **kwargs):
        self.n_players = int(n_players)
        self.n_actions = int(n_actions)
        self.embed_dim = int(embed_dim)
        super().__init__(*args, net_arch=[], **kwargs)
        obs_space = self.observation_space
        F = obs_space["players"].shape[1]
        G = obs_space["globals"].shape[0]
        self.set_encoder = SetEncoder(self.n_players, F, G, embed_dim=embed_dim)
        self.action_head = nn.Linear(embed_dim, n_actions)
        self.value_head = nn.Linear(embed_dim, 1)

    def forward(self, obs, deterministic=False):
        Z = self.set_encoder(obs["players"], obs["globals"])  # (B, N, D)
        logits = self.action_head(Z).reshape(-1, self.n_players * self.n_actions)
        dist = self.action_dist.proba_distribution(logits)
        actions = dist.get_actions(deterministic=deterministic)
        values = self.value_head(Z[:, -1, :])  # last CLS token (use 2 heads for dual critics)
        return actions, values, dist.log_prob(actions)


# Wiring into PPO
# env = SetObsWrapper(HexagonBasketballEnv(...), players_per_side=3, n_players=6)
# model = PPO(
#     SetPolicy,
#     env,
#     policy_kwargs={"n_players": 6, "n_actions": len(ActionType), "embed_dim": 64},
#     verbose=1,
# )
# model.learn(1_000_000)
```

If you want *zero* flattening (even for logits), you can define a custom
distribution that accepts `(B, N, A)` logits and samples per-player actions.
SB3's default MultiCategorical expects a flat `(B, N*A)` vector, which is why
the example flattens logits only at the very end.

Notes:
- This uses a shared action head across tokens, so it cannot memorize player slots.
- For defense-only training, use the same policy but slice tokens for the defense team
  when mapping logits to actions (or train a separate defense policy).

## Permutation-Invariant Version (Embedding + Pooling)

One simple approach is a DeepSets-style representation:

1) Build a *per-player* feature vector (orderless set).
2) Apply a shared embedding function f( ) to each player.
3) Pool over the offense set and defense set (mean/sum/max).
4) Concatenate pooled features with global scalars (shot clock, hoop, ball pos, etc).

### Example per-player features (simple illustrative choice)

Here we use a very small feature vector for each player:
```
[q_norm, r_norm, role(+1/-1), has_ball]
```

That yields:
```
P0: [ 0.1250, 0.5000,  1, 0]
P1: [ 0.2500, 0.3750,  1, 1]
P2: [ 0.1250, 0.6250,  1, 0]
P3: [ 0.1250, 0.4375, -1, 0]
P4: [ 0.0000, 0.5625, -1, 0]
P5: [-0.0625, 0.5000, -1, 0]
```

### Pooling (mean) by team

Offense mean:
```
[0.1667, 0.5000, 1.0000, 0.3333]
```

Defense mean:
```
[0.0208, 0.5000, -1.0000, 0.0000]
```

### Final permutation-invariant observation (example)

```
perm_inv_obs =
  [offense_mean (4),
   defense_mean (4),
   shot_clock (1),
   ball_pos (2),
   hoop (2)]

= [0.1667, 0.5000, 1.0000, 0.3333,
   0.0208, 0.5000,-1.0000, 0.0000,
   17.0000,
   0.2500, 0.3750,
  -0.2500, 0.5000]
```

### Notes

- In practice, f( ) would be a learned MLP (e.g., 16 or 32 dims), and you might pool
  using mean+max or an attention pooler.
- You can also add *pairwise* features in a permutation-invariant way via a graph
  (edges = relative positions, pooled messages).
- The key difference from the current vector is that swapping player ids does
  not change the pooled representation, which reduces index bias.

## Permutation-Equivariant Version (Per-Player Outputs + Pooled Context)

Pooling alone is *not* enough for action selection. A common fix is to keep a
per-player embedding (so each player keeps their own identity via features),
and *add* pooled team/opponent context.

### Example per-player features (more explicit)

Use a shared feature vector for each player:
```
[q_norm, r_norm, role(+1/-1), has_ball, dist_to_ball_norm, dist_to_hoop_norm]
```

Computed for the same state:
```
P0: [ 0.1250, 0.5000,  1, 0, 0.1250, 0.3750]
P1: [ 0.2500, 0.3750,  1, 1, 0.0000, 0.5000]
P2: [ 0.1250, 0.6250,  1, 0, 0.2500, 0.5000]
P3: [ 0.1250, 0.4375, -1, 0, 0.1250, 0.3750]
P4: [ 0.0000, 0.5625, -1, 0, 0.2500, 0.3125]
P5: [-0.0625, 0.5000, -1, 0, 0.3125, 0.1875]
```

### Pooled context (mean) by team

Offense mean:
```
[0.1667, 0.5000, 1.0000, 0.3333, 0.1250, 0.4583]
```

Defense mean:
```
[0.0208, 0.5000, -1.0000, 0.0000, 0.2292, 0.2917]
```

### Example policy input for a *single player*

For player 1 (ball handler), build:
```
player_input =
  [player_features (6),
   offense_mean (6),
   defense_mean (6),
   shot_clock (1),
   hoop (2)]
```

Concrete example:
```
[ 0.2500, 0.3750, 1.0000, 1.0000, 0.0000, 0.5000,
  0.1667, 0.5000, 1.0000, 0.3333, 0.1250, 0.4583,
  0.0208, 0.5000,-1.0000, 0.0000, 0.2292, 0.2917,
  17.0000, -0.2500, 0.5000]
```

The key property: if you permute players, each player’s *own* input vector
changes accordingly, but the pooled context stays the same. The policy is
permutation-equivariant: swapping players swaps their logits/actions.

### Optional: Ball-handler attention (example)

If you want the ball handler to “query” teammates without fixed ordering, use
attention. A toy example using only (q_norm, r_norm) dot products:
```
ball handler (P1) xy = [0.2500, 0.3750]
teammate xy: P0 [0.1250, 0.5000], P2 [0.1250, 0.6250]
dot sims = [0.2188, 0.2656]
softmax weights = [0.4883, 0.5117]
```

That gives a weighted teammate summary without assuming a fixed order.
