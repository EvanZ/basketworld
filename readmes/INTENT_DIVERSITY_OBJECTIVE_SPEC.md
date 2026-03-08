# Intent Diversity Objective Spec (Review Draft)

## 1. Purpose

Define an implementation-ready spec for the diversity objective used in emergent play learning with latent intent `z`.

This document answers three open questions explicitly:

1. Is a diversity objective necessary?
2. Is this approach variational?
3. How should commitment window state be represented and exposed?

## 2. Key Decisions

1. We do **not** handcraft named plays.
2. We learn a latent intent-conditioned policy `pi(a_t | s_t, z)`.
3. We add a diversity objective so different `z` values correspond to behaviorally distinct trajectories.
4. Diversity objective is optional and default-off.
5. Backward compatibility with playable/dev and legacy non-intent checkpoints is mandatory.

## 3. Is Diversity Objective Necessary?

Short answer:

1. Not required to improve raw reward in general.
2. Required if we want `z` to reliably encode multiple distinct reusable behavior modes ("plays").

Without diversity pressure, the policy can ignore `z` and collapse to one dominant behavior mode.

## 4. Is This Variational?

Yes, in the DIAYN-style design.

We use a discriminator `q_psi(z|tau)` as a variational approximation to maximize a lower bound on mutual information `I(z; tau)`.

This is related to variational methods, but not a VAE reconstruction objective.

## 5. Scope and Non-Goals

In scope:

1. Objective definition and rollout integration.
2. Discriminator/data pipeline design.
3. Config surface, logging, tests, and compatibility constraints.

Out of scope:

1. Hand-authored playbook templates.
2. Frontend UI changes.
3. Multi-possession environment redesign.

## 6. Compatibility Requirements

1. `enable_intent_learning=false` keeps current behavior unchanged.
2. No request/response schema changes for `/api/step` and `/api/playable/step`.
3. Legacy checkpoints continue to load in playable and dev mode.
4. Intent-specific logic only activates when both model and env config support it.
5. Strict schema/version checks fail fast on mismatch.

## 7. Variables and Contracts

1. `K`: number of learned intents.
2. `z`: intent index sampled from `Uniform({0,...,K-1})` when intent is active.
3. `tau`: trajectory segment (initially full possession).
4. `intent_active`: binary field; 1 means intent conditioning is active.
5. `intent_visible`: binary field from observer perspective.
6. `intent_index`: integer index, meaningful only when `intent_active=1` and visible.

Do not overload `intent_index=0` for unknown/null. Unknown/null is represented by flags.

## 8. Commitment Window Definition

1. `intent_commitment_steps` is an integer hyperparameter (for example `4`).
2. It is environment runtime state, not just model state.
3. In current one-possession episodes, intent is sampled at `reset()` and usually remains fixed, so commitment mostly prepares future multi-possession support.
4. Observation exposure:
   - offense-private view may include `intent_age_norm` and `intent_commitment_remaining_norm`,
   - defense/human-facing view should mask these unless explicitly allowed by config.

## 9. Objective

Base task objective:

`J_env = E[sum_t gamma^t * r_env_t]`

Diversity objective (DIAYN-style lower bound):

`J_div = E[log q_psi(z|tau) - log p(z)]`, with `p(z)=1/K`

Total objective for policy:

`J_total = J_env + beta_div(t) * J_div`

Reward shaping form used by PPO:

`r_total_t = r_env_t + beta_div(t) * b_t`

where `b_t` is derived from episode/segment bonus:

`B(tau, z) = log q_psi(z|tau) - log(1/K)`

MVP distribution rule:

1. Compute `B` once per completed episode.
2. Spread uniformly over the episode steps: `b_t = B / L_episode`.

Optional anti-collapse regularizer:

`L_balance = KL(U_K || p_hat(z))`

where `p_hat(z)` is empirical intent usage over recent episodes.

## 10. Offense/Defense Asymmetry Policy

Diversity bonus is applied only to offense transitions:

1. `role_flag > 0`
2. `intent_active=1`
3. run has intent-diversity enabled

Defense training receives no direct `r_div`.

Against humans:

1. defense receives masked/unknown intent features (`intent_visible=0`)
2. behavior relies on observable geometry and dynamics, not privileged labels

## 11. SB3 Integration Design

## 11.1 Injection Point

Primary path:

1. Add `IntentDiversityCallback` and use `_on_rollout_end()`.
2. Update discriminator from collected completed episodes.
3. Inject bonus into `model.rollout_buffer.rewards`.
4. Recompute returns/advantages if needed before PPO update.

Safety rule:

1. If current SB3 callback timing does not allow reliable recomputation, move to a small custom PPO subclass with explicit hook before `compute_returns_and_advantage`.
2. Do not ship partial integration where `r_div` is logged but not used by policy gradients.

## 11.2 Trajectory Data Capture

From callback locals/rollout buffer, capture per transition:

1. env index
2. rollout step index
3. `done`
4. offense/defense role (`role_flag`)
5. intent fields (`intent_index`, `intent_active`, `intent_visible`)
6. observation summary for discriminator
7. action

Maintain carry-over buffers for episodes crossing rollout boundaries.

## 11.3 Discriminator Update

Per rollout:

1. build completed episode dataset `(tau_i, z_i)`
2. run `N_disc_updates` minibatch updates on `q_psi`
3. compute per-episode `B(tau_i, z_i)`
4. normalize and clip bonus
5. write step bonuses back into rollout rewards

## 12. Discriminator Module

New module:

`basketworld/utils/intent_discovery.py`

Required classes/functions:

1. `TrajectoryEncoder`
2. `IntentDiscriminator`
3. `IntentEpisodeBuffer`
4. `compute_diversity_bonus(...)`
5. `assign_bonus_to_rollout(...)`

MVP encoder:

1. state projection + action embedding
2. temporal mean pooling MLP

Recommended V2:

1. GRU encoder

## 13. File/Class Touchpoints

Environment and observation:

1. `basketworld/envs/basketworld_env_v2.py`
   - class `HexagonBasketballEnv`
   - add intent runtime state and config args
2. `basketworld/envs/core/observations.py`
   - `build_observation(...)` intent feature append/masking hooks
3. `basketworld/utils/wrappers.py`
   - `SetObservationWrapper` add intent globals
   - `MirrorObservationWrapper` preserve intent fields correctly
4. `basketworld/utils/self_play_wrapper.py`
   - role-conditioned opponent obs must recondition intent visibility fields

Training:

1. `train/config.py`
   - add CLI args for intent + diversity knobs
2. `train/env_factory.py`
   - pass intent configs into env ctor
3. `train/callbacks.py`
   - wire `IntentDiversityCallback` in mixed callback list
4. `train/train.py`
   - log params, callback config, compatibility metadata
5. `basketworld/utils/callbacks.py`
   - add `IntentDiversityCallback`

Backend compatibility:

1. `basketworld/utils/mlflow_params.py`
   - parse intent/diversity params for env reconstruction
2. `app/backend/observations.py`
   - clone/recondition new intent fields in role-conditioned obs copies
3. `app/backend/evaluation.py`
   - worker role-conditioned clones include intent fields

## 14. Config Surface (New)

Core intent:

1. `--enable-intent-learning` (bool, default `False`)
2. `--num-intents` (int, default `8`)
3. `--intent-commitment-steps` (int, default `4`)
4. `--intent-null-prob` (float, default `0.2`)
5. `--intent-visible-to-defense-prob` (float, default `0.0`)

Diversity:

1. `--intent-diversity-enabled` (bool, default `False`)
2. `--intent-diversity-beta-target` (float, default `0.05`)
3. `--intent-diversity-warmup-steps` (int, default `1_000_000`)
4. `--intent-diversity-ramp-steps` (int, default `1_000_000`)
5. `--intent-diversity-clip` (float, default `2.0`)
6. `--intent-disc-lr` (float, default `3e-4`)
7. `--intent-disc-batch-size` (int, default `256`)
8. `--intent-disc-updates-per-rollout` (int, default `2`)
9. `--intent-disc-encoder` (`mlp_pool|gru`, default `mlp_pool`)
10. `--intent-balance-weight` (float, default `0.0`)

## 15. Metrics

Core:

1. `intent/disc_loss`
2. `intent/disc_top1_acc`
3. `intent/bonus_raw_mean`
4. `intent/bonus_norm_mean`
5. `intent/bonus_norm_std`
6. `intent/beta_current`
7. `intent/usage_entropy`
8. `intent/usage_min_prob`

Behavior diagnostics:

1. `intent/ppp_by_intent/<z>`
2. `intent/pass_rate_by_intent/<z>`
3. `intent/shot_dist_by_intent/<z>`

Robustness:

1. `intent/defense_unknown_intent_ppp`
2. `intent/defense_unknown_intent_delta_vs_baseline`

## 16. Tests

Unit/integration tests to add:

1. `tests/test_intent_encoding_contract.py`
2. `tests/test_intent_visibility_masking.py`
3. `tests/test_self_play_wrapper_intent_reconditioning.py`
4. `tests/test_intent_diversity_callback_reward_injection.py`
5. `tests/test_intent_diversity_returns_recomputed.py`
6. `tests/test_intent_diversity_offense_only.py`
7. `tests/test_intent_diversity_disabled_noop.py`
8. `app/backend/tests/test_playable_legacy_non_intent_models.py`
9. `app/backend/tests/test_eval_worker_intent_obs_clone.py`

## 17. Phased Delivery

1. D0: latent intent plumbing + schema/version safeguards only.
2. D1: diversity callback + discriminator + rollout reward injection.
3. D2: GRU encoder and stronger diagnostics.
4. D3: robustness tuning for unknown/no-intent offense (human proxy).

## 18. Acceptance Criteria

1. Non-collapse:
   - discriminator top-1 accuracy is materially above random (`>= 2x random baseline`).
2. Diversity:
   - intent usage entropy above agreed floor and no single-intent domination.
3. Task quality:
   - offense PPP non-inferior to baseline within agreed margin.
4. Robustness:
   - defense unknown-intent degradation within agreed budget.
5. Compatibility:
   - playable/dev flows unchanged for legacy non-intent checkpoints when feature disabled.

## 19. D1 Callback Pseudocode (Exact Flow)

Reference target class:

1. `basketworld/utils/callbacks.py::IntentDiversityCallback`

Pseudo-implementation:

```python
class IntentDiversityCallback(BaseCallback):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.ep_buffer = IntentEpisodeBuffer()  # stores partial episodes across rollouts
        self.disc = IntentDiscriminator(cfg).to(cfg.device)
        self.opt = torch.optim.Adam(self.disc.parameters(), lr=cfg.intent_disc_lr)
        self.running_bonus_norm = RunningMeanStd()

    def _on_training_start(self):
        # Optional: assert expected obs keys exist when feature enabled
        # Optional: cache whether rollout buffer stores dict obs in this SB3 build
        pass

    def _on_step(self):
        # Called every env step during rollout collection.
        # locals keys used in current codebase callbacks:
        # - infos, dones, rewards, actions, new_obs/obs (SB3-version dependent)
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        actions = self.locals.get("actions", None)
        obs = self.locals.get("new_obs", None) or self.locals.get("obs", None)

        for env_i in range(len(infos)):
            tr = extract_transition(obs, actions, infos, env_i)
            # extract_transition must include:
            # role_flag, intent_index, intent_active, intent_visible
            # compact state features + action embedding input
            self.ep_buffer.append(env_i, tr)

            if dones[env_i]:
                self.ep_buffer.close_episode(env_i)
        return True

    def _on_rollout_end(self):
        # 1) Build completed-episode dataset for offense + intent-active only.
        episodes = self.ep_buffer.pop_completed(
            filter_fn=lambda ep: ep.role_is_offense and ep.intent_active
        )
        if len(episodes) == 0:
            return

        # 2) Train discriminator for N updates.
        ds = build_dataset(episodes)  # (tau, z)
        for _ in range(self.cfg.intent_disc_updates_per_rollout):
            batch = ds.sample(self.cfg.intent_disc_batch_size)
            logits = self.disc(batch.tau)
            loss = F.cross_entropy(logits, batch.z)
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()

        # 3) Compute episode bonus B = log q(z|tau) - log(1/K).
        bonuses = []
        for ep in episodes:
            logp = log_softmax(self.disc(ep.tau))[ep.z]
            B = logp - math.log(1.0 / self.cfg.num_intents)
            bonuses.append(B)

        # 4) Normalize + clip B, then spread across steps.
        B_norm = normalize_with_running_stats(bonuses, self.running_bonus_norm)
        B_clip = np.clip(B_norm, -self.cfg.intent_diversity_clip, self.cfg.intent_diversity_clip)
        beta = beta_schedule(
            num_timesteps=self.model.num_timesteps,
            warmup=self.cfg.intent_diversity_warmup_steps,
            ramp=self.cfg.intent_diversity_ramp_steps,
            target=self.cfg.intent_diversity_beta_target,
        )

        # 5) Inject into rollout buffer rewards using rollout indices captured in ep_buffer.
        # Each episode stores (buffer_step_idx, env_idx) per transition.
        rb = self.model.rollout_buffer
        for ep, Bc in zip(episodes, B_clip):
            per_step_bonus = (beta * Bc) / max(1, ep.length)
            for (t_idx, env_i) in ep.buffer_indices:
                rb.rewards[t_idx, env_i] += per_step_bonus

        # 6) Ensure PPO update uses modified rewards.
        # If returns/advantages already computed at this point, recompute here.
        maybe_recompute_returns_and_advantages(self.model, self.locals)

        # 7) Log metrics.
        log_intent_metrics(...)
```

Helper contract details:

1. `IntentEpisodeBuffer.append(env_i, tr)` stores transition and its `(t_idx, env_i)` rollout index.
2. `extract_transition(...)` must be robust to SB3 locals differences (`obs` vs `new_obs`).
3. `maybe_recompute_returns_and_advantages(...)`:
   - preferred: call `rollout_buffer.compute_returns_and_advantage(last_values, dones)` with values from callback locals,
   - fallback: if required values are not reliably available, move D1 to custom PPO subclass hook before return computation.
4. Defense transitions are ignored for diversity reward in D1.

## 20. Open Review Questions

1. Do we keep D1 bonus as episode-uniform or invest immediately in per-step attribution?
2. Should we include `intent_age_norm` in offense observation at D0, or add only in D2?
3. Do we enforce balance regularization in D1 (`intent-balance-weight > 0`) or keep off initially?
