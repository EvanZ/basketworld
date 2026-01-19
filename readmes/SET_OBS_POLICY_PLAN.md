# Set-Based Observation Policy Plan (SB3 + PPO)

## Introduction

We want a policy that **sees each player as an individual token** (self, teammates, opponents),
while **avoiding slot/index bias** from flattening fixed-order vectors. The core idea is:
build per-player feature tokens, pass them through a **shared encoder** and a **permutation‑equivariant
attention block** (no positional encodings), and let each player’s action head read its **own
token output**. This preserves individual teammate state (no pooling), but makes the policy
robust to arbitrary player ordering.

## Goals

- Keep individual player information (no teammate pooling).
- Avoid slot/index bias in the policy network.
- Integrate cleanly with SB3 + PPO and MultiDiscrete actions.
- Maintain backward compatibility with the web app by default.
- Add targeted tests to prevent regressions.

## Non-Goals

- Rewriting the env core observation vector for gameplay/UI.
- Changing the action space (still MultiDiscrete with MOVE/PASS/SHOOT).
- Removing action masks or role flags.

## Plan

### Phase 0: Define the token contract

- Token schema (per player):
  - `[q_norm, r_norm, role(+1/-1), has_ball, skill_layup, skill_3pt, skill_dunk, lane_steps_norm]`
  - Skill values are absolute sampled percentages in [0, 1] (not deltas).
- Global schema:
  - `[shot_clock, hoop_q_norm, hoop_r_norm]`
- Token set:
  - **All players** (self + teammates + opponents) as tokens.
- Globals usage:
  - Decision: **append globals to each token** (simple and explicit conditioning).
  - Alternative (not chosen): inject globals via a separate MLP + FiLM/add.
- Output:
  - Action logits per player from that player’s token output.

**Offense vs defense note**

Use one schema for all players. For fields that only apply to one side:

- `has_ball`: set to 1 only for the current ball handler; 0 otherwise (both teams).
- `lane_steps_norm`: offense uses offensive lane steps; defense uses defender-in-key steps.
- Any offense-only field not listed above should be set to 0 for defense (and vice-versa)
  to keep a consistent token shape.

### Phase 1: Observation wrapper (training-only, no env change)

- Add a new wrapper class in `basketworld/utils/wrappers.py` that exposes:
  - `players: (n_players, F)` and `globals: (G,)`
  - Keep existing keys (`action_mask`, `role_flag`, `skills`) unchanged.
  - Tokens will include role/skills as features, but the legacy `role_flag` and
    `skills` keys remain for backward compatibility.
- Keep the env’s current flat `obs` for UI/debugging.
- Add a feature flag in training config (ex: `--use-set-obs`) to enable the wrapper.

### Phase 2: SB3 policy implementation

- Create a policy class (ex: `SetAttentionPolicy`) that:
  - Uses a **shared token encoder** (MLP) + `nn.MultiheadAttention` (no positional encodings).
  - Produces **token outputs** `Z` shaped `(B, N, D)`.
  - Computes per-player logits with a **shared action head**: `logits_i = g(z_i)`.
  - Flattens logits only at the very end for SB3 MultiCategorical (shape `(B, N*A)`).
- Use **residual + LayerNorm** around attention output to avoid head collapse.
- Decision: keep the existing **dual-critic** structure (offense/defense value heads).
  - Use **team CLS tokens** (one offense, one defense) for value heads to avoid pooling player tokens.
- Decision: keep **pass-logit bias** support for parity with existing training schedules.
- Defaults: `set_embed_dim=64`, `set_heads=4`, `set_token_mlp_dim=64`, `set_cls_tokens=2`.

### Phase 3: Training/Eval integration

- Update `train/env_factory.py` to apply the set wrapper when flag enabled.
- Update `train/train.py` and `train/policy_utils.py` to register the new policy.
- Ensure `basketworld/utils/self_play_wrapper.py` passes through the new dict obs.
- Log set-attention hyperparameters to MLflow (`set_embed_dim`, `set_heads`, `set_token_mlp_dim`, `set_cls_tokens`).

### Phase 3b: Tests/Validation

- Unit tests for wrapper:
  - Shape checks for `(n_players, F)` and `(G,)`.
  - Deterministic values from a fixed env state.
- Model tests:
  - Token permutation test: reorder player tokens and confirm logits are unchanged
    (up to permutation for non-self tokens).
- Integration test:
  - One PPO step with the new policy, ensure no runtime errors.

### Phase 4: Backend (web app) impacts

- If the web app keeps using the old flat obs:
  - **No backend changes required** for the observation table.
  - Only policy inference code needs to accept the dict obs used by the new policy.
- Decision: expose tokens to the UI.
  - Add fields to `get_full_game_state`:
    - `obs_tokens.players`, `obs_tokens.globals`
    - `obs_tokens.attention` (avg + per-head weights, labels)
  - Add a version flag to avoid breaking existing clients.

### Phase 5: Frontend (web app) impacts

- Keep current observation table intact by default (flat obs).
- Decision: add an “Attention” tab with:
  - Token View (per-player tokens + globals).
  - Attention map table + per-head selector.
  - Heatmap coloring with min-max normalization.
  - Download PNG for attention maps.
- Note: training/inference do not request attention weights; `average_attn_weights`
  only affects debug/inspection output (per-head vs mean).

## Rollout Strategy

- Keep set-based policy behind a flag.
- Train a short run on 3v3 and compare:
  - Policy entropy, pass frequency, shot distribution bias.
- Only swap the web app policy after parity on core behavior.

## Open Questions

- Decision: action heads for **all players** (matches MultiDiscrete env shape + UI).
- Decision: do not add interdependent angles/distances to tokens for now; revisit if learning is slow.
