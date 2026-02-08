# SetAttention Policy: Numerical Walkthrough (Toy Example)

This is a *small numeric* example that mirrors the real architecture but uses
tiny dimensions so the math is easy to follow by hand. The real model uses
more features, more heads, and larger embeddings, but the operations are the
same.

## 0) Toy configuration (small numbers for clarity)

We pick a minimal setup:
- Players per side: 3 (P=3 offense + 3 defense in the real env)
- **Toy P=3 tokens** (pretend we only have three player tokens)
- Token feature size: T=2 (we'll only use `[q_norm, r_norm]`)
- Globals size: G=1 (shot clock only)
- Embed dim: D=2
- CLS tokens: C=2 (CLS_OFF, CLS_DEF)
- Attention heads: 1 (real model uses 4 by default)

**Important:** the real model uses T=11 player features, G=3 globals, D=64,
and 4 heads. This is just to show the mechanics numerically.

## 1) Inputs (players + globals)

Players (B=1, P=3, T=2):

```
P0 = [0.20, 0.00]
P1 = [0.00, 0.30]
P2 = [0.20, 0.20]
```

Globals (B=1, G=1):

```
G = [0.10]  # shot_clock normalized
```

## 2) Broadcast globals and concatenate

Globals are appended to every player token:

```
P0 + G -> [0.20, 0.00, 0.10]
P1 + G -> [0.00, 0.30, 0.10]
P2 + G -> [0.20, 0.20, 0.10]
```

Shape now: (B=1, P=3, T+G=3)

## 3) Token MLP (toy weights)

The real model uses two linear layers with activation:

```
Linear(T+G -> token_mlp_dim) -> activation -> Linear(token_mlp_dim -> D)
```

For a clean numeric example, assume this toy MLP simply selects the first
two dimensions:

```
W1 = [[1, 0, 0],
      [0, 1, 0]]
b1 = [0, 0]
ReLU
W2 = I
b2 = [0, 0]
```

So embeddings are just `[q_norm, r_norm]`:

```
E0 = [0.20, 0.00]
E1 = [0.00, 0.30]
E2 = [0.20, 0.20]
```

## 4) Append CLS tokens

CLS tokens are learned parameters. Pick simple toy values:

```
CLS_OFF = [ 0.10, 0.10]
CLS_DEF = [-0.10, 0.10]
```

Token sequence (P+C=5):

```
T0 = [ 0.20, 0.00]  (P0)
T1 = [ 0.00, 0.30]  (P1)
T2 = [ 0.20, 0.20]  (P2)
T3 = [ 0.10, 0.10]  (CLS_OFF)
T4 = [-0.10, 0.10]  (CLS_DEF)
```

## 5) Self-attention (single head, identity projections)

The real model uses learned Q/K/V projections. For clarity, assume identity
projections (Q=K=V=T) and one head.

Score for token i attending to token j:

```
score_ij = (Qi · Kj) / sqrt(D)
```

### Example: CLS_OFF attends to all tokens

Q = T3 = [0.10, 0.10], D=2 -> sqrt(D)=1.414

Dot products:

```
T0: 0.02 -> 0.0141
T1: 0.03 -> 0.0212
T2: 0.04 -> 0.0283
T3: 0.02 -> 0.0141
T4: 0.00 -> 0.0000
```

Softmax weights (approx):

```
[0.1997, 0.2011, 0.2026, 0.1997, 0.1969]
```

Weighted sum (V = T):

```
attn_out(CLS_OFF) = 0.1997*T0 + ... + 0.1969*T4
                  ≈ [0.0807, 0.1405]
```

Residual + LayerNorm (toy):

```
CLS_OFF + attn_out = [0.10, 0.10] + [0.0807, 0.1405]
                   = [0.1807, 0.2405]
LayerNorm -> approx [-1.0, 1.0]
```

### Example: P0 attends to all tokens

Q = T0 = [0.20, 0.00]

Scores:

```
T0: 0.04 -> 0.0283
T1: 0.00 -> 0.0000
T2: 0.04 -> 0.0283
T3: 0.02 -> 0.0141
T4: -0.02 -> -0.0141
```

Softmax weights (approx):

```
[0.2034, 0.1977, 0.2034, 0.2005, 0.1949]
```

Weighted sum:

```
attn_out(P0) ≈ [0.0819, 0.1395]
P0 + attn_out = [0.2819, 0.1395]
LayerNorm -> approx [1.0, -1.0]
```

This is the per-token embedding used by the policy/value heads.

## 6) Action logits (per player token)

Action head is a linear layer applied to each player token. Suppose we have
3 actions (toy) with weight matrix:

```
W_action = [[ 1,  0],
            [ 0,  1],
            [ 1, -1]]
b_action = [0, 0, 0]
```

For P0 token after attention, use `[1.0, -1.0]`:

```
logits = W_action * [1.0, -1.0] = [ 1.0, -1.0, 2.0 ]
softmax ≈ [0.259, 0.035, 0.705]
```

That gives per-player action probabilities.

## 7) Value heads (CLS tokens)

The critic reads CLS tokens:

```
offense_token = token[P]     (CLS_OFF)
defense_token = token[P + 1] (CLS_DEF)
```

Toy value head:

```
V_off = [0.5, 0.5] · CLS_OFF
V_def = [0.5, 0.5] · CLS_DEF
```

Using CLS_OFF = [-1, 1] from the LN example gives:

```
V_off = 0.0
```

In the real model, these weights are learned and CLS outputs depend on the
current state, so values vary across steps.

## 8) How this maps to the real model

Real (default) shapes:
- Players P = 6 (3 offense + 3 defense)
- Token features T = 11
- Globals G = 3
- Embed dim D = 64
- CLS tokens C = 2
- Attention heads = 4

So the extractor outputs:

```
(B, (P + C) * D) = (B, 8 * 64) = (B, 512)
```

The policy then reshapes back to (B, 8, 64), uses the first 6 tokens for
action logits, and the last 2 tokens for offense/defense value heads.

---

If you want, I can add a second example using the **actual token features**
and a real head configuration (still with small numbers), or include a diagram
that mirrors this walkthrough.***
