# Pointer-Targeted Passing Mode Plan

## Goal

Add a new passing mode (`pointer_targeted`) that lets the policy and UI choose an explicit teammate receiver, while preserving current directional passing (`PASS_E..PASS_SE`) as a fully supported default mode.

## Current Status (In Progress)

Completed:
- `pass_mode` plumbed through env init, backend state serialization, MLflow param parsing, and training CLI/env factory.
- Backend step route accepts structured pointer pass payloads (`{type:"PASS", target:id}`) with backward-compatible legacy payload handling.
- Environment supports explicit passer->target execution path in `pointer_targeted`, including legality checks and intended-target metadata.
- For policy-driven actions in `pointer_targeted`, pass action indices now map to teammate target slots (`PASS_E` => slot 0, `PASS_NE` => slot 1, ...), enabling immediate trainability without PPO algorithm changes.
- Phase 3 core policy path is implemented for set-attention: factorized pointer actor (`action_type` + conditional `pass_target`) with combined PPO log-prob/entropy via a custom pointer distribution.
- Mode-aware pointer metrics are logged to MLflow via episode summaries:
  - `Pass IntentVsOutcomeMatch`
  - `Pass Target Entropy`
  - `Pass Target Entropy Norm`
  - `Pass Target KL Uniform`
- Frontend controls render teammate target buttons in pointer mode and submit structured pass actions.
- Replay/move-history uses `actions_taken_meta` to display pointer intent (`PASS->id`) when available.

Pending:
- Optional additions only (if desired): richer pointer diagnostics such as per-player target entropy and policy-vs-empirical target KL.

## Why

In 5-on-5, directional pass resolution can become a bottleneck:

- Crowded spacing makes direction-only intent ambiguous.
- The resolver can choose a different receiver than policy intent.
- This mismatch reduces policy control and interpretability.

`pointer_targeted` removes that mismatch by making receiver choice explicit.

## Non-Goals (v1)

- Replace directional mode.
- Support cross-loading directional checkpoints as pointer models.
- Change core steal/OOB physics formulas.

## Design Principles

1. Strict mode boundary:
   - `directional` and `pointer_targeted` execution paths are separate.
2. Backward compatibility:
   - Missing mode metadata defaults to `directional`.
3. Backend is source of truth:
   - UI should read `pass_mode` from backend game state, not infer directly from MLflow in frontend.
4. Positional/permutation invariance:
   - Pointer target scoring is done over teammate tokens with shared scoring weights.

## Modes

## `directional` (existing)

- Action semantics: existing `ActionType` including `PASS_E..PASS_SE`.
- Receiver selection: current environment logic.
- UI controls: directional pass arrows/buttons.

## `pointer_targeted` (new)

- Pass semantics:
  - Actor decides `PASS` vs non-pass.
  - If `PASS`, actor selects explicit `pass_target_id`.
- Environment executes pass directly passer->target.
- UI controls: `p-1` teammate-target pass buttons (exclude self).

## End-to-End Data Contract

## Policy output (conceptual)

- `action_type` head: `{NOOP, MOVE_*, SHOOT, PASS}`.
- `pass_target` head (conditional): logits over teammate candidates.

Training log-prob factorization:

- `logp = logp(action_type) + 1[action_type=PASS] * logp(pass_target | PASS)`

## Step submission payload (frontend -> backend)

Support both schemas:

1. Legacy directional:
   - `{ "2": "PASS_NE" }`
2. Pointer-targeted:
   - `{ "2": { "type": "PASS", "target": 4 } }`

Backend normalizes to an internal action representation per `pass_mode`.

## Step response / replay snapshot

Include in state history:

- `pass_mode`
- action intent fields (when applicable):
  - `actions_taken[player_id]` (existing)
  - `actions_taken_meta[player_id] = { type, target? }` (new)

For pass outcomes:

- `last_action_results.passes[passer_id]` includes:
  - `intended_target` (pointer mode)
  - `actual_target`
  - `success`, turnover metadata (existing patterns)

## Backend / Env Plan

## 1) Mode plumbing

Add `pass_mode` in:

- training config/params
- environment init
- backend game state serialization (`get_full_game_state`)
- policy metadata / MLflow params

Default:

- if not provided or missing in loaded model metadata -> `directional`.

## 2) Internal action representation

Add a normalized internal action object (example):

- `kind`: move/shoot/pass/noop
- `direction` (directional mode only)
- `target_id` (pointer mode only)

Keep external compatibility by accepting legacy string actions and mapping to this object.

## 3) Environment pass execution

Branch early on `pass_mode`:

- `directional`: existing code path.
- `pointer_targeted`: direct passer->target execution with current steal/OOB mechanics applied on that line.

Key requirement:

- No directional resolver in pointer mode.

## 4) Legal masks / validation

Pointer mode pass target must be:

- teammate
- not self
- in-bounds valid player id

Illegal target handling:

- follow existing illegal action policy behavior (`noop`/error), but log clear diagnostics.

## Policy / Training Plan

## 1) Mode-specific actor path

Keep existing actor path for directional.

Add pointer path with:

- `action_type` logits
- pointer target logits over teammate tokens

If `use_dual_policy=True`, add role-specific pointer heads:

- `pass_target_head_offense`
- `pass_target_head_defense`

## 2) Invariance

Pointer logits computed by shared scorer over teammate tokens:

- permutation equivariant over token ordering
- consistent under set-based architecture assumptions

## 3) Loss integration

PPO evaluate_actions path computes:

- action_type log-prob/entropy
- conditional pass_target log-prob/entropy

and combines them for policy gradient.

## 4) Metrics

Log mode-specific diagnostics:

- `PassTargetEntropy`
- `PassTargetKL`
- `PassIntentVsOutcomeMatch`
- pass turnover breakdown by mode

## 5) Pass Logit Bias Compatibility

Keep the existing pass-bias API/scheduler/admin controls, but make application mode-aware:

- `directional`:
  - current behavior remains (bias PASS direction logits, e.g. `PASS_E..PASS_SE`)
  - `pass_prob_min` remains minimum total mass over pass-direction actions
- `pointer_targeted`:
  - apply bias to `action_type=PASS` only (not to teammate target logits)
  - reinterpret `pass_prob_min` as minimum `P(action_type=PASS)`

Safety rule:

- if no legal pass targets exist, mask `PASS` regardless of bias settings.

## Frontend Plan

## 1) Controls mode switch

`PlayerControls` renders by backend `pass_mode`:

- `directional`: current directional controls
- `pointer_targeted`: teammate id pass buttons (`team_size - 1`)

## 2) UX behavior

Pointer mode buttons:

- only shown/enabled for current ball handler
- disabled for illegal targets
- label includes teammate id (optional: risk/EV tooltip later)

## 3) Selection model

Support union action selection shape:

- string action (legacy)
- structured pass action object (`{ type: "PASS", target: id }`)

Keep replay/manual stepping backward compatible by reading mode from each stored state.

## 4) Replay / Visualization

For pass annotations:

- pointer mode should show explicit intended receiver from snapshot metadata.
- directional mode remains unchanged.

## MLflow / Policy Loading Compatibility

On training:

- log `pass_mode` in run params and embed in policy metadata.

On policy load:

- backend resolves model `pass_mode`.
- if absent, use `directional`.

On policy swap:

- backend updates current `pass_mode` in returned state so UI re-renders controls immediately.

## Test Plan

## Unit tests

1. Mode resolution:
   - metadata present vs missing -> correct fallback.
2. Action normalization:
   - legacy string actions and structured pointer actions parse correctly.
3. Pointer legality:
   - self/opponent/invalid target rejected per policy.
4. Env execution:
   - pointer mode uses explicit target (no resolver drift).

## Policy tests

1. Pointer logits shape/masking for variable team sizes.
2. Permutation equivariance on teammate token reorder.
3. `evaluate_actions` log-prob composition sanity checks.

## Integration tests

1. End-to-end directional regression (unchanged behavior).
2. End-to-end pointer pass:
   - UI select target -> backend step -> replay shows intended target.
3. Policy swap directional <-> pointer updates controls and stepping correctly.

## Rollout Plan

1. Phase 1: Mode plumbing + metadata + backend state exposure.
2. Phase 2: Env pointer execution path + action normalization.
3. Phase 3: Policy pointer heads + PPO loss integration.
4. Phase 4: UI pointer controls + replay metadata usage.
5. Phase 5: Tests, ablation, and default tuning.

## Suggested Config Surface

- `--pass-mode directional|pointer_targeted` (default `directional`)

Optional later:

- `--pointer-pass-temperature`
- `--pointer-pass-topk-mask`

## Open Decisions

1. Action-space encoding for PPO internals:
   - separate channels (`action_type`, `pass_target`) vs packed encoding.
2. Whether pointer mode requires teammate arc constraints or allows full teammate set always.
3. How to represent non-pass `pass_target` in buffers (sentinel vs masked-only).
