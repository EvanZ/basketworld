# N-Player Curriculum Implementation Plan

## Goal

Support a curriculum that trains across increasing players-per-side in one workflow, e.g.:

- `2v2 -> 3v3 -> 5v5`

while preserving as much learned policy/value structure as possible at stage boundaries.

## Current Constraints

1. `players` changes both observation and action spaces:
   - `n_players = 2 * players_per_side`
   - `action_space = MultiDiscrete([len(ActionType)] * n_players)`
   - `obs["obs"]` length changes with `players_per_side`
2. Opponent sampling currently pools all `models/unified_iter_*.zip` artifacts without filtering by player count.
3. Existing transfer utility (`train/policy_utils.py`) only transfers critics and assumes compatible shapes.
4. `continue-run-id` currently assumes same environment topology.

Implication:

- Direct checkpoint continuation from `2v2` to `5v5` is not shape-compatible.
- We need explicit stage-boundary re-init + selective weight transfer.

## Design Choice (Recommended)

Implement **stage-based curriculum in one training run** with explicit model rebuild at each player-count transition.

Core principles:

1. Use one stage at a time with fixed `players`.
2. At stage transition, instantiate a new policy for new env shapes.
3. Transfer only compatible/shared modules from previous stage.
4. Keep opponent pools stage-local (same player count only).

## Scope

### In Scope (v1)

- CLI-configured player curriculum stages.
- Stage-local training loop with per-stage alternation counts.
- Cross-stage transfer for set-attention policies (recommended path).
- Stage-aware artifact naming and opponent filtering.
- Stage metadata logging to MLflow.

### Out of Scope (v1)

- Automatic transfer for non-set MLP policies across player counts.
- Cross-stage mixed-opponent training (e.g., 3v3 policy vs 5v5 policy).
- Full hyperparameter auto-tuning per stage.

## API / CLI Proposal

Add to `train/config.py`:

- `--players-curriculum`: string, stage spec.
  - Example: `"2:20,3:20,5:60"` (players_per_side:alternations)
- `--players-curriculum-strict`: bool (default true)
  - Fail if stage transfer cannot be applied safely.
- `--players-curriculum-transfer`: choice
  - `none|critic_only|set_shared|set_shared_plus_heads`
  - default: `set_shared`
- `--players-curriculum-final-players`: optional int guardrail
  - Validates last stage target.

Rules:

- If `--players-curriculum` is provided, ignore scalar `--players` except as fallback default.
- Sum of stage alternations replaces `--alternations` as total curriculum workload.

## Stage Model

Add a stage data model in a new module (e.g. `train/player_curriculum.py`):

- `players_per_side: int`
- `alternations: int`
- `stage_idx: int`
- `stage_name: str` (e.g. `p2`, `p3`, `p5`)

Utility functions:

- `parse_players_curriculum(spec: str) -> list[Stage]`
- `validate_players_curriculum(stages)`
- `iter_stage_alternations(stages)` with global alternation indexing.

## Training Loop Refactor

Modify `train/train.py`:

1. Parse stages at startup.
2. For each stage:
   - set `args.players = stage.players_per_side`
   - create policy-init env for that stage
   - either:
     - stage 0: init policy normally
     - stage >0: build new policy and transfer from previous stage
3. Run stage-local alternations using existing mixed offense/defense loop.
4. Save artifacts with stage-qualified names.

Suggested artifact naming:

- `models/unified_p{players}_iter_{global_alt}.zip`
- Optional compatibility alias for final stage:
  - `models/unified_iter_{global_alt}.zip` (final stage only)

## Stage-Aware Opponent Sampling

Update `train/policy_utils.py`:

- Extend artifact filtering regex to include player-count tag.
- Add required `players_per_side` filter parameter.
- Only sample opponent checkpoints from same stage/player count.

Why:

- Prevent loading incompatible opponent models with mismatched action/obs shapes.

## Transfer Strategy Across Player Counts

### Recommended path: set-attention policies (`--use-set-obs true`)

Transferable modules (shape-compatible across `n_players`):

- `SetAttentionExtractor.token_mlp`
- `SetAttentionExtractor.attn`
- `SetAttentionExtractor.attn_norm`
- `SetAttentionExtractor.cls_tokens` (if same `num_cls_tokens`)
- Optional token head MLPs (`token_head_mlp_pi`, `token_head_mlp_vf`) if dimensions match
- Action/value heads if dimensions match (they usually do when action count per player is unchanged)

Non-transferable:

- Anything tied to flattened token count (`features_dim` dependent internals), if present.

### Non-set policies

`critic_only` may fail when first layer input dims change with obs size. For v1:

- either disallow non-set curriculum transitions, or
- allow with `transfer=none` and log warning.

## New Transfer Utility

Add in `train/policy_utils.py` (or new file):

- `transfer_across_player_count(source_policy, target_policy, mode)`

Behavior:

1. Verify policy classes and extractor types.
2. Copy module `state_dict`s only when tensor shapes match.
3. Log copied vs skipped modules clearly.
4. Rebuild optimizer after copying (as already done in set-attention init path).

Safety:

- No silent partial failures.
- In strict mode, fail stage transition if required modules cannot transfer.

## Schedule Handling

Keep existing schedule system (`continue_schedule_mode`, SPA/entropy/phi/pass) and layer stage semantics on top:

1. Global schedule mode remains unchanged.
2. Stage boundaries should not reset schedules unless explicitly requested.
3. Log stage-local metadata:
   - stage start/end global alternation
   - stage players_per_side
   - transfer mode and transfer summary

Optional v2:

- stage-specific schedule overrides.

## MLflow Logging / Artifacts

Log new params:

- `players_curriculum_enabled`
- `players_curriculum_spec`
- `players_curriculum_transfer_mode`
- `players_curriculum_num_stages`

Per-stage metrics:

- `stage_{k}_players`
- `stage_{k}_alternations`
- `stage_{k}_transfer_modules_copied`
- `stage_{k}_transfer_modules_skipped`

Artifact:

- `curriculum/players_curriculum_plan.md` (resolved stage table with global alternations)

## Testing Plan

### Unit tests

1. `tests/train/test_player_curriculum_parse.py`
   - valid specs, invalid specs, ordering, duplicates.
2. `tests/train/test_player_curriculum_transfer.py`
   - set-attention transfer copies expected modules.
   - incompatible modules are skipped/fail in strict mode.
3. `tests/train/test_policy_utils_stage_filter.py`
   - opponent sampling only returns same-player checkpoints.

### Integration tests (small)

1. 2-stage smoke run: `2:1,3:1` with tiny timesteps.
2. Verify:
   - model trains both stages
   - stage transition happens once
   - opponent sampler uses stage-compatible artifacts only
   - final checkpoint loads for final stage.

## Rollout Plan

1. **Phase 1**: Parser + logging + stage loop scaffolding (no transfer, no-op rebuild).
2. **Phase 2**: Stage-aware artifact naming and opponent filtering.
3. **Phase 3**: Set-attention cross-stage transfer utility + strict mode.
4. **Phase 4**: Tests + docs + migration notes.
5. **Phase 5 (optional)**: Non-set fallback transfer improvements and stage-specific schedule overrides.

## Migration / Compatibility Notes

1. Keep old behavior when `--players-curriculum` is absent.
2. Keep legacy artifact regex path for existing single-player-count runs.
3. For curriculum runs, prefer stage-qualified artifact names to avoid accidental incompatible loads.

## Example Target Usage

```bash
python train/train.py \
  --use-set-obs true \
  --players-curriculum "2:20,3:20,5:60" \
  --players-curriculum-transfer set_shared \
  --players-curriculum-strict true \
  --num-envs 16
```

## Open Decisions

1. Should curriculum be alternation-based only, or support timestep-based stage budgets too?
2. Should stage transitions also reset opponent history, or keep same-stage history only?
3. Do we enforce `--use-set-obs true` when `--players-curriculum` is enabled?
4. Should final stage also emit legacy checkpoint names for tooling compatibility?
