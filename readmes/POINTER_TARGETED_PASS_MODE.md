# Pointer-Targeted Passing Mode

This document describes the shipped `pointer_targeted` passing mode, how it differs from legacy `directional`, and how it is represented across training, backend, and UI.

## Why This Mode Exists

Directional pass actions (`PASS_E..PASS_SE`) are useful, but in crowded states the receiver can be ambiguous.  
`pointer_targeted` removes that ambiguity by making the intended teammate explicit.

## Modes

### `directional` (legacy/default)
- Pass intent is a direction.
- Environment resolves which teammate (if any) receives the pass based on arc/strategy rules.

### `pointer_targeted`
- Pass intent is an explicit receiver ID.
- Environment executes pass to that receiver directly (with the same steal/OOB mechanics).

## How To Enable

Set pass mode at training time:

```bash
python train/train.py ... --pass-mode pointer_targeted
```

Notes:
- CLI supports `--pass-mode directional|pointer_targeted` (default `directional`).
- `pass_mode` is logged in MLflow params.
- Backend reads `pass_mode` from the run/env config when loading a game.

## End-to-End Flow (Runtime)

1. Frontend reads `state.pass_mode` from backend snapshot.
2. In pointer mode, UI pass selection is explicit teammate intent (`PASS->id` conceptually).
3. Step payload can include structured pass action: `{type:"PASS", target:id}`.
4. Backend normalizes payload and stores intended targets via `env.set_pointer_pass_targets(...)`.
5. Environment step still receives discrete action indices, but pointer targets are consumed during pass execution.
6. Response and replay snapshots include both selected actions and pass intent metadata.

## Action Contract

Backend step accepts both forms:

1. Directional/legacy action:

```json
{ "2": "PASS_NE" }
```

2. Pointer-targeted pass:

```json
{ "2": { "type": "PASS", "target": 4 } }
```

`ActionRequest.actions` supports mixed payloads and remains backward-compatible.

## Slot Mapping (Policy + Masks)

To preserve the existing discrete action interface, pointer mode reuses pass action slots:

- `PASS_E` -> teammate slot 0
- `PASS_NE` -> teammate slot 1
- `PASS_NW` -> teammate slot 2
- `PASS_W` -> teammate slot 3
- `PASS_SW` -> teammate slot 4
- `PASS_SE` -> teammate slot 5

Teammate slot order is deterministic:
- same-team player IDs excluding self
- sorted ascending
- capped at 6 slots

Action masks in pointer mode expose only valid pass slots for the current ball handler.

## Backend Normalization Details

`/api/step` keeps the old discrete action path but augments it for pointer mode:

- Structured pass payload is normalized to:
  - discrete action override (`PASS_E` index placeholder),
  - metadata: `{"type":"PASS","target":<id>}`.
- Backend builds `pointer_targets: Dict[player_id, target_id]` from that metadata.
- It calls `env.set_pointer_pass_targets(pointer_targets)` before stepping.
- Legality enforcement is still mask-first when constructing `full_action`:
  - invalid/unmasked actions are replaced with `NOOP`.

This means pointer targeting is additive and backward-compatible with existing step plumbing.

## Environment Internals

### 1) Pointer pass target storage

- `set_pointer_pass_targets()` stores a sanitized `Dict[int,int]` for the next step only.

### 2) Action mask rewrite in pointer mode

For current ball handler only:
- all pass slots `PASS_E..PASS_SE` are first zeroed,
- slot `k` is enabled iff teammate slot `k` exists.

Non-ball-handlers keep pass slots masked as before.

### 3) Slot resolution fallback

During `_process_simultaneous_actions`:
- if pass action is selected and `pass_mode == "pointer_targeted"`:
  - explicit target is read from pending pointer targets when present,
  - else target falls back to slot mapping from selected pass action index.

### 4) Pass execution branch

`basketworld/envs/core/passing.py::attempt_pass(...)` branches on `pass_mode`.

In pointer mode:
- validates `intended_target` exists and is a legal teammate.
- rejects with:
  - `reason="missing_target"` when no target is available,
  - `reason="illegal_target"` when target is invalid/self/opponent/out-of-range.
- on valid target, it runs the same steal/OOB/interception mechanics as directional path.

In directional mode:
- receiver is selected from teammates in directional arc via `nearest` or `best_ev`.

## Steal/Interception Math (Applied in Both Modes)

Given passer `p`, receiver `r`, and eligible defenders `d` between them:

- `pass_distance = hex_distance(p, r)`
- defender contribution:
  - `steal_i = base_steal_rate * exp(-steal_perp_decay * perp_dist_i) * (1 + steal_distance_factor * pass_distance) * position_weight_i`
  - `position_weight_i = steal_position_weight_min + (1 - steal_position_weight_min) * t_i`
  - `t_i in [0,1]` is defender position along pass line (near receiver is more dangerous)
- aggregate steal probability:
  - `total_steal = 1 - Π_i (1 - steal_i)`

If interception occurs:
- defender with max individual contribution gets possession.

## Policy Architecture (Pointer Factorization)

In set-attention policy, pointer mode uses a factorized distribution:

Architecture diagram (current code, Mermaid):

- `./SET_ATTENTION_POLICY_VALUE_ARCHITECTURE.md`

- `action_type` categorical over:
  - all non-pass actions + one PASS type
- `pass_target` categorical over pass slots (0..5), conditioned on PASS.

Combined log-prob and entropy are used by PPO, while sampled final actions are mapped back into original discrete action IDs so SB3 buffers remain unchanged.

Implementation highlights:
- `PointerTargetedMultiCategoricalDistribution` handles factorization/mapping.
- pass-slot logits are built from query-key token scores over teammate token IDs.
- pass bias controls (`pass_logit_bias`, `pass_prob_min`) are applied to PASS type logit in pointer mode (not directly to target-slot logits).
- if no valid target slots exist, PASS type is effectively masked.

### PPO math for pointer factorization

Per player, define:
- `a_t` = sampled action type (`non-pass` or `PASS`)
- `s_t` = sampled pass slot (only meaningful if `a_t = PASS`)

The policy uses:
- `pi_type(a_t | x)`
- `pi_slot(s_t | x, a_t = PASS)`

Log-prob used by PPO is:

- `log pi(a | x) = log pi_type(a_t | x) + 1[a_t = PASS] * log pi_slot(s_t | x, PASS)`

Entropy term is decomposed consistently:
- type entropy always applies
- slot entropy contributes only on PASS branch

This is implemented inside the pointer distribution/evaluate path while still emitting legacy discrete action IDs to SB3 buffers.

### Pass bias and pass floor in pointer mode

In directional mode, pass bias/floor affects aggregate pass-direction logits.  
In pointer mode:
- bias and `pass_prob_min` are applied to `PASS` in `action_type` logits,
- slot logits are left as target-preference scores (plus legality masking),
- if no legal target slot exists, PASS is masked regardless of bias/floor.

## Replay and State Metadata

Per-step snapshots expose:

- `state.pass_mode`
- `state.actions_taken`
- `state.actions_taken_meta`
- `state.last_action_results.passes[*].intended_target`
- turnover entries with `intended_target` when applicable

`actions_taken_meta` for passes is filled with this precedence:

1. explicit target from request metadata
2. pass result `intended_target` (or `target`)
3. turnover `intended_target` / `pass_target`

This is what makes replay and move-history show `PASS->id` consistently.

## Pointer Metrics

Episode wrappers collect pointer-mode diagnostics:

- `pointer_pass_attempts`
- `pointer_pass_intent_match_rate`
- `pointer_pass_target_entropy`
- `pointer_pass_target_entropy_norm`
- `pointer_pass_target_kl_uniform`

These are aggregated and logged to MLflow as:
- `Pass IntentVsOutcomeMatch`
- `Pass Target Entropy`
- `Pass Target Entropy Norm`
- `Pass Target KL Uniform`

## Frontend Behavior

`PlayerControls` derives mode from `gameState.pass_mode`:
- directional mode: directional pass controls.
- pointer mode: teammate target controls and display normalization to `PASS->id`.

Submission behavior in pointer mode:
- pass actions are emitted as structured payloads with explicit target.
- display/replay rendering consumes state metadata, not local heuristics.

## Constraints and Edge Cases

- Max pass slots: 6 (bounded by action space `PASS_E..PASS_SE`).
- Teammate slot ordering is ID-sorted and deterministic.
- Pointer mode ignores directional receiver strategy selection (`nearest`/`best_ev`) for target choice because target is explicit.
- Only ball handler pass actions are meaningful; non-ball-handler pass actions are masked/ignored via existing legality path.
- Missing `pass_mode` metadata defaults to `directional` for compatibility.

## Debugging Tips

- Use `/api/debug/action_masks` to confirm legal pointer slots for current ball handler.
- Inspect `last_action_results.passes` and `turnovers` for `intended_target` vs `target`.
- In replay/manual stepping, verify `actions_taken_meta` contains `{"type":"PASS","target":...}` for pointer passes.

## Backward Compatibility

- Missing `pass_mode` defaults to `directional`.
- Legacy pass submissions remain accepted.
- Older directional checkpoints continue to work unchanged.
