# Emergent Play Learning Plan (No Handcrafted Playbook)

## Objective

Learn reusable multi-step offensive behaviors ("plays") directly from reward, without manually defining named plays, scripted phases, or basketball doctrine templates.

In this document, "play" means an emergent sequence pattern discovered by optimization, not a handcrafted finite state machine.

Maintain backward compatibility for both:

1. Playable app mode.
2. Development mode (`App.vue` / local backend workflows),

when using existing non-intent (legacy) models.

## Non-goals

1. No hardcoded plays like `GIVE_AND_GO` / `DRIVE_KICK`.
2. No hand-authored phase transitions.
3. No training setup where defense depends on privileged offense labels unavailable against humans.

## Compatibility requirements

1. Default-off behavior:
   - `enable_intent_learning` remains `False` by default.
   - Legacy training/eval/playable paths remain unchanged unless explicitly enabled.

2. API compatibility:
   - Keep action contracts unchanged for `/api/step` and `/api/playable/step`.
   - No required frontend payload changes for playable or dev mode.

3. Checkpoint compatibility:
   - Legacy checkpoints that were trained without intent features must continue to load and run.
   - Intent-enabled checkpoints should be clearly tagged via run metadata/params to avoid accidental mismatch.

4. Observation compatibility:
   - When intent learning is disabled, intent features are all zeros or omitted in a backward-compatible way.
   - Wrapper dimensions must match model expectations in both legacy and intent-enabled configurations.

5. Runtime safeguards:
   - Add explicit validation errors for model/observation shape mismatches.
   - Fail fast with actionable messages instead of silent misbehavior.

## Core approach

Use latent intent/options:

1. Sample or choose a latent intent variable `z` at possession start.
2. Keep `z` fixed for a commitment window.
3. Train primitive policy as `pi(a_t | s_t, z)`.
4. Add diversity pressure so different `z` values produce distinct trajectory modes.
5. Keep defense robust by training heavily on no-latent/unknown-latent episodes.

This produces emergent behavior modes that can later be interpreted as learned plays.

## Why this fits current codebase

Current constraints:

1. Environment action API is primitive per-player `MultiDiscrete`:
   - `HexagonBasketballEnv.action_space`
   - File: `basketworld/envs/basketworld_env_v2.py`

2. `SelfPlayEnvWrapper` expects primitive vectors for training and opponent:
   - File: `basketworld/utils/self_play_wrapper.py`

3. Episode currently corresponds to one possession (terminal on shot/turnover/violation):
   - Reward termination logic in `basketworld/envs/core/rewards.py`

Therefore: phase 1 should keep primitive action space unchanged and add only latent conditioning + losses.

## High-level design

## 1) Latent intent runtime state in env

Add environment-level latent state:

1. `intent_id` (discrete int in `[0, num_intents-1]`)
2. `intent_age`
3. `intent_commitment_remaining`
4. `intent_known` flag (for observability control)

Since training episodes are currently single-possession, sample `intent_id` in `reset()`.
If later you move to multi-possession episodes, sample at each possession boundary instead.

## 2) Policy conditioning

Condition low-level policy on `intent_id`:

1. Simple path (first): append scalar/one-hot intent features to observation.
2. Better path: learn an intent embedding and concatenate to actor/critic inputs.

The actor still outputs primitive `ActionType` actions.

## 3) Diversity objective (critical)

Without extra objective, all intents can collapse to one behavior.

Add one of:

1. DIAYN-style discriminator `q(z | trajectory_window)` with MI bonus.
2. Option-contrastive objective on trajectory embeddings.

Reward shape:

`r_total = r_env + beta_intent * r_diversity`

Keep `beta_intent` small and scheduled.

## 4) Defense asymmetry handling

Defense should not rely on offense-private intent labels.

Training distribution:

1. `p_known_intent` episodes: offense intent visible only when appropriate.
2. `p_unknown_intent` episodes: defense gets `intent_known=0`, `intent_id=0`.
3. `p_no_intent` episodes: offense itself runs with null intent.

At inference vs human:

1. Default `intent_known=0`
2. Defense behaves from geometry/risk cues, not privileged latent.

## File/class modification plan

## A) Environment

File: `basketworld/envs/basketworld_env_v2.py`
Class: `HexagonBasketballEnv`

Modify:

1. `__init__`:
   - Add args:
     - `enable_intent_learning: bool = False`
     - `num_intents: int = 8`
     - `intent_commitment_steps: int = 4`
     - `intent_visible_to_defense_prob: float = 0.0`
     - `intent_null_prob: float = 0.2`
     - `intent_obs_mode: str = "private_offense"`
   - Add state:
     - `self.intent_id`, `self.intent_age`, `self.intent_commitment_remaining`
     - `self.intent_known_for_training_obs`

2. `reset()`:
   - Initialize latent intent runtime:
     - Sample `intent_id` unless null intent sampled.
     - Reset age/counters.

3. `step()`:
   - Increment `intent_age`.
   - Decrement commitment counter.
   - Include intent diagnostics in `info` for logging:
     - `intent_id`, `intent_age`, `intent_known`.

4. Observation construction (`_get_observation` and related vector assembly):
   - Add intent features to observation with masking policy.

## B) Core state helpers

Likely new file: `basketworld/envs/core/intent.py`

Add:

1. `sample_intent_id(env)`
2. `mask_intent_for_perspective(env, role_flag, obs)`
3. `encode_intent_features(env)`

This keeps `basketworld_env_v2.py` from growing further.

## C) Self-play wrapper observability fix

File: `basketworld/utils/self_play_wrapper.py`
Class: `SelfPlayEnvWrapper`

Why needed:

`step()` currently flips `role_flag` for opponent observation by copying dict and changing only `role_flag`.

If intent visibility depends on perspective, wrapper must also rewrite intent fields for opponent obs.

Modify:

1. Add helper:
   - `_recondition_intent_fields_for_role(obs_dict, opponent_is_offense)`
2. Call it before opponent `predict()`.

This prevents accidental privileged leakage.

## D) Set observation wrappers

File: `basketworld/utils/wrappers.py`

Classes:

1. `SetObservationWrapper`
   - Increase `GLOBAL_DIM` and/or token dims to include intent features.
   - Add intent fields in `globals`.

2. `MirrorObservationWrapper`
   - Ensure intent fields are preserved and not corrupted during mirroring.

## E) Training config + env factory

Files:

1. `train/config.py`
   - Add CLI args:
     - `--enable-intent-learning`
     - `--num-intents`
     - `--intent-commitment-steps`
     - `--intent-null-prob`
     - `--intent-visible-to-defense-prob`
     - `--intent-diversity-beta`
     - `--intent-diversity-warmup-steps`
     - `--intent-noise-prob`

2. `train/env_factory.py`
   - Forward new args into `HexagonBasketballEnv(...)`.

3. `train/train.py`
   - Log all intent params to MLflow.
   - Add training mixture schedule for known/unknown/no-intent episodes.

## F) Policy classes

### Phase 1 (minimal invasive)

No new policy head required if intent is in observation vector and policy can consume it.

Touched classes only for dimensional compatibility:

1. `PassBiasMultiInputPolicy` (`basketworld/utils/policies.py`)
2. `PassBiasDualCriticPolicy` (`basketworld/utils/policies.py`)
3. `SetAttentionDualCriticPolicy` (`basketworld/policies/set_attention_policy.py`)

### Phase 2 (recommended for quality)

Add explicit intent embedding module:

1. Multi-input policies:
   - Add `nn.Embedding(num_intents, d_intent)` and concatenate in actor/critic trunks.

2. Set-attention policy:
   - Add intent embedding into `globals` processing path.

## G) Diversity/discovery auxiliary module

New file suggestion:

`basketworld/utils/intent_discovery.py`

Add:

1. `TrajectoryEncoder`
2. `IntentDiscriminator`
3. `compute_intent_diversity_bonus(...)`

Then integrate via callbacks or custom PPO training step.

## H) Callbacks + metrics

Files:

1. `basketworld/utils/callbacks.py`
2. `train/callbacks.py`

Add metrics:

1. `intent_usage_entropy`
2. `intent_balance_minmax`
3. `intent_mutual_info_estimate`
4. `intent_cluster_ppp` (points per possession by intent)
5. `defense_no_intent_eval_score`

## Training curriculum

Recommended schedule:

1. Stage 0:
   - `beta_intent=0`
   - intent present but no diversity reward
   - verify stability

2. Stage 1:
   - warm up `beta_intent` from `0 -> target`
   - enforce intent commitment window

3. Stage 2:
   - increase unknown/no-intent fraction for defense robustness
   - periodic eval against intent-free offense

## Real examples (emergent, not handcrafted)

## Example 1: What a learned intent looks like in practice

After training with `num_intents=8`, analysis might show:

1. `z=2`:
   - early pass frequency high
   - shot locations mostly perimeter
   - longer possession length

2. `z=6`:
   - immediate rim pressure
   - higher turnover risk, higher dunk rate

No template was coded; this is inferred from trajectory statistics.

## Example 2: Human offense (no declared intent)

At inference vs human:

1. System sets offense intent as null/unknown.
2. Defense sees:
   - `intent_known=0`
   - neutral `intent_id`
3. Defense still performs because it was trained with unknown/no-intent episodes.

## Example 3: Detecting mode collapse

Bad run symptom:

1. `intent_usage_entropy` high but trajectory clusters overlap heavily.
2. Discriminator accuracy near random.
3. Performance differences by intent nearly zero.

Fix:

1. Raise `beta_intent` gradually.
2. Increase commitment window.
3. Add short trajectory encoder context (not single-step classifier).

## Tests to add

1. `tests/test_intent_state_reset.py`
   - intent sampled/reset correctly each episode.

2. `tests/test_intent_observation_masking.py`
   - offense/defense visibility rules hold, including wrapper role-flip path.

3. `tests/test_self_play_wrapper_intent_reconditioning.py`
   - opponent obs in `SelfPlayEnvWrapper` does not leak private intent.

4. `tests/test_intent_disabled_backward_compat.py`
   - with feature disabled, behavior and obs dims remain backward compatible.

5. `tests/test_intent_diversity_bonus_smoke.py`
   - auxiliary reward term finite and stable.

6. `app/backend/tests/test_playable_legacy_model_compat.py`
   - playable routes work with legacy non-intent models and unchanged request schema.

7. `app/backend/tests/test_dev_mode_legacy_model_compat.py`
   - lifecycle/dev routes work with legacy non-intent models.

## Rollout plan

1. Implement phase 1 only (latent state + observation + masking + logging).
2. Run short smoke training and verify no regressions.
3. Add diversity objective (phase 2) and track intent separability.
4. Harden defense with unknown/no-intent curriculum.
5. Add post-hoc clustering notebook/report to interpret learned intents.

## Success criteria

1. Behavior modes are statistically distinct across `z` without handcrafted labels.
2. Overall reward/PPP improves or stays neutral versus baseline.
3. Defense remains strong when offense intent is unknown (human proxy).
4. Turning off intent learning returns baseline behavior cleanly.
5. Playable app and development mode remain functional with legacy non-intent models and unchanged API contracts.

## ChatGPT 5.4 Critique of Plan

1. The biggest gap is backward compatibility. The plan says intent features can be “all zeros or omitted” when disabled, but the current runtime builds fixed observation spaces, and existing checkpoints depend on those exact shapes. See [PLAY_LEARNING_IMPLEMENTATION_PLAN.md#L36](/home/ubuntu/basketworld/readmes/PLAY_LEARNING_IMPLEMENTATION_PLAN.md#L36), [basketworld_env_v2.py#L316](/home/ubuntu/basketworld/basketworld/envs/basketworld_env_v2.py#L316), and [wrappers.py#L36](/home/ubuntu/basketworld/basketworld/utils/wrappers.py#L36). I would not modify the legacy schema in place. Add an explicit `obs_schema_version` and only construct intent-augmented spaces for intent-enabled runs.

2. The null/unknown encoding is flawed as written. The plan defines `intent_id` in `[0, num_intents-1]` and later uses `intent_id=0` for unknown/no-intent episodes, which collides with a real learned intent. See [PLAY_LEARNING_IMPLEMENTATION_PLAN.md#L78](/home/ubuntu/basketworld/readmes/PLAY_LEARNING_IMPLEMENTATION_PLAN.md#L78) and [PLAY_LEARNING_IMPLEMENTATION_PLAN.md#L117](/home/ubuntu/basketworld/readmes/PLAY_LEARNING_IMPLEMENTATION_PLAN.md#L117). Use separate fields like `intent_index`, `intent_active`, and `intent_visible`, or reserve a dedicated null index outside the learned range.

3. The implementation surface is wider than the file plan says. Updating `SelfPlayEnvWrapper` is necessary, but it is not sufficient. The backend has multiple role-conditioned observation clone paths that would also need intent reconditioning and schema awareness: [self_play_wrapper.py#L121](/home/ubuntu/basketworld/basketworld/utils/self_play_wrapper.py#L121), [observations.py#L22](/home/ubuntu/basketworld/app/backend/observations.py#L22), [evaluation.py#L140](/home/ubuntu/basketworld/app/backend/evaluation.py#L140), and [lifecycle_routes.py#L96](/home/ubuntu/basketworld/app/backend/routes/lifecycle_routes.py#L96). The plan should explicitly include these files, plus [mlflow_params.py#L26](/home/ubuntu/basketworld/basketworld/utils/mlflow_params.py#L26), or the backend will reconstruct the wrong env for saved runs.

4. Phase 2 is not implementation-ready yet. “Integrate via callbacks or custom PPO training step” is the hard part, not a small detail. See [PLAY_LEARNING_IMPLEMENTATION_PLAN.md#L260](/home/ubuntu/basketworld/readmes/PLAY_LEARNING_IMPLEMENTATION_PLAN.md#L260) versus current stock SB3 PPO usage in [train.py#L388](/home/ubuntu/basketworld/train/train.py#L388) and [train.py#L835](/home/ubuntu/basketworld/train/train.py#L835). You need to specify where trajectory windows are stored, when the discriminator is updated, and how the bonus is injected into the rollout buffer.

5. The commitment-window state looks premature for phase 1. With one possession per episode and `intent_id` sampled in `reset()`, `intent_commitment_remaining` has no real effect unless phase 1 also allows mid-possession resampling or a higher-level chooser. See [PLAY_LEARNING_IMPLEMENTATION_PLAN.md#L49](/home/ubuntu/basketworld/readmes/PLAY_LEARNING_IMPLEMENTATION_PLAN.md#L49), [PLAY_LEARNING_IMPLEMENTATION_PLAN.md#L80](/home/ubuntu/basketworld/readmes/PLAY_LEARNING_IMPLEMENTATION_PLAN.md#L80), and [PLAY_LEARNING_IMPLEMENTATION_PLAN.md#L83](/home/ubuntu/basketworld/readmes/PLAY_LEARNING_IMPLEMENTATION_PLAN.md#L83). I would remove it from the first cut.

**Suggestions**

- Split this into `Phase 1A` and `Phase 2`. `Phase 1A`: env state, schema/versioning, backend/wrapper conditioning, logging, tests. No diversity loss yet.
- For set-attention models, avoid stuffing a large one-hot into `globals` if you can. `globals` gets broadcast to every player token in [set_attention_policy.py#L37](/home/ubuntu/basketworld/basketworld/policies/set_attention_policy.py#L37) and [set_attention_policy.py#L105](/home/ubuntu/basketworld/basketworld/policies/set_attention_policy.py#L105). A compact flag plus learned embedding is cleaner.
- Add explicit runtime validation that compares model metadata against env schema before the first `predict()`, not just generic shape-mismatch errors.
- Expand tests to include MLflow param round-trip and backend role-conditioned clone helpers, not just env reset and self-play wrapper tests.
- Tighten success criteria with thresholds: separability metric, PPP delta vs baseline, and unknown-intent defense degradation budget.

The overall direction is good and it does fit this codebase. I would call it good enough to implement a narrowed phase 1 after the compatibility and schema issues are fixed. I would not start the diversity/discovery phase from this plan as-is.

## Critique from Codex 5.3

**Critical Findings**
1. `intent_id` semantics conflict: you define `intent_id` in `[0..num_intents-1]`, but later use `intent_id=0` for unknown/no-intent, which collides with a real learned intent ([plan](/home/ubuntu/basketworld/readmes/PLAY_LEARNING_IMPLEMENTATION_PLAN.md#L78), [plan](/home/ubuntu/basketworld/readmes/PLAY_LEARNING_IMPLEMENTATION_PLAN.md#L117)).
2. Backward-compatibility is under-specified for observation shape. Current env/wrappers use fixed dict spaces and fixed dims; adding intent fields in-place can break legacy checkpoints ([env obs space](/home/ubuntu/basketworld/basketworld/envs/basketworld_env_v2.py#L378), [set wrapper dims](/home/ubuntu/basketworld/basketworld/utils/wrappers.py#L8), [plan](/home/ubuntu/basketworld/readmes/PLAY_LEARNING_IMPLEMENTATION_PLAN.md#L36)).
3. `SelfPlayEnvWrapper` changes are necessary but not sufficient. Role-conditioned obs cloning also exists in backend inference/eval paths and must be updated too, or you’ll get train/serve mismatch ([self-play wrapper](/home/ubuntu/basketworld/basketworld/utils/self_play_wrapper.py#L121), [backend observations](/home/ubuntu/basketworld/app/backend/observations.py#L22), [backend evaluation](/home/ubuntu/basketworld/app/backend/evaluation.py#L140)).
4. Diversity objective integration is too vague for implementation. “callbacks or custom PPO step” is the hard part; with current SB3 flow this needs explicit rollout-buffer integration details ([plan](/home/ubuntu/basketworld/readmes/PLAY_LEARNING_IMPLEMENTATION_PLAN.md#L260), [training loop](/home/ubuntu/basketworld/train/train.py#L835)).
5. Commitment-window state adds complexity without effect in current single-possession episodes; it should be deferred unless you actually resample intent intra-episode ([plan](/home/ubuntu/basketworld/readmes/PLAY_LEARNING_IMPLEMENTATION_PLAN.md#L83), [plan](/home/ubuntu/basketworld/readmes/PLAY_LEARNING_IMPLEMENTATION_PLAN.md#L49)).

**What I’d Change**
1. Split into phases: `Phase 1A` (schema-safe latent plumbing + masking + logging + tests), `Phase 1B` (training mixture schedule), `Phase 2` (diversity/discovery objective).
2. Introduce explicit observation schema/version metadata and hard validation before `predict()` to prevent silent mismatches.
3. Replace ambiguous intent encoding with separate fields: `intent_index`, `intent_active`, `intent_visible` (or reserve a dedicated null index outside learned range).
4. Expand file plan to include backend runtime/config paths (`app/backend/observations.py`, `app/backend/evaluation.py`, `app/backend/routes/lifecycle_routes.py`, `basketworld/utils/mlflow_params.py`), not only env/wrapper/train files.
5. Add concrete PPO integration design for diversity reward: where trajectory windows live, update frequency, and how bonus enters rollout rewards.
6. Tighten success criteria with thresholds (separability metric, PPP delta, unknown-intent robustness delta).

The plan is good directionally and implementable after these adjustments. I’d approve starting a narrowed `Phase 1A` now, and block `Phase 2` until the PPO integration spec is concrete.