## Additive Pass-Logit Bias (Exploration Aid)

To ensure we explore passes early in training (when turnovers make the policy collapse to low pass logits), we add an **additive bias** to the actor’s PASS logits and anneal it to zero.

### What it does
- For each player’s action categorical, we add a scalar `b(t)` to the PASS action logits (indices 8..13 in this env).
- This increases the probability mass on passes without hard floors or zeroing gradients.
- `b(t)` is decayed exponentially over the planned timesteps, so the aid disappears once the policy learns to pass on its own.

### Files changed
- `basketworld/utils/policies.py`:
  - `PassBiasMultiInputPolicy`: subclass of SB3 `MultiInputActorCriticPolicy` that overrides `_get_action_dist_from_latent` to add `pass_logit_bias` to PASS logits.
  - Verbose comments explain splitting flat logits into per-dimension chunks, applying the bias, and recombining.

- `basketworld/utils/callbacks.py`:
  - `PassLogitBiasExpScheduleCallback(start, end, total_planned_timesteps)`: computes an exponential schedule and updates the policy via `policy.set_pass_logit_bias(current)` each rollout/update.
  - Also updated `PotentialBetaExpScheduleCallback` to call env methods where relevant.

- `train/train.py`:
  - Use `PassBiasMultiInputPolicy` for PPO policy class.
  - CLI flags: `--pass-logit-bias-start`, `--pass-logit-bias-end`.
  - Attaches `PassLogitBiasExpScheduleCallback` to offense/defense segments.

### Code references
Policy (additive bias):

```1465:1517:basketworld/utils/policies.py
class PassBiasMultiInputPolicy(MultiInputActorCriticPolicy):
    def _get_action_dist_from_latent(self, latent_pi, latent_sde=None):
        action_logits = self.action_net(latent_pi)
        if not self._nvec or abs(self.pass_logit_bias) <= 1e-12:
            return self.action_dist.proba_distribution(action_logits=action_logits)
        sizes = self._nvec
        chunks = th.split(action_logits, sizes, dim=1)
        biased_chunks = []
        bias = th.as_tensor(self.pass_logit_bias, device=action_logits.device, dtype=action_logits.dtype)
        for chunk in chunks:
            c = chunk.clone()
            for idx in self._pass_indices:
                if idx < c.shape[1]:
                    c[:, idx] = c[:, idx] + bias
            biased_chunks.append(c)
        biased_logits = th.cat(biased_chunks, dim=1)
        return self.action_dist.proba_distribution(action_logits=biased_logits)
```

Callback (schedule):

```270:321:basketworld/utils/callbacks.py
class PassLogitBiasExpScheduleCallback(BaseCallback):
    def _apply_current(self):
        t = int(getattr(self.model, "num_timesteps", 0))
        current = self._scheduled_value(t)
        if hasattr(self.model.policy, "set_pass_logit_bias"):
            self.model.policy.set_pass_logit_bias(float(current))
```

Trainer wiring:

```440:486:train/train.py
unified_policy = PPO(PassBiasMultiInputPolicy, ...)
pass_bias_callback = PassLogitBiasExpScheduleCallback(p_start, p_end, total_ts)
```

### How to use
- Start training with e.g.:

```bash
--pass-logit-bias-start 1.0 --pass-logit-bias-end 0.0
```

- Monitor pass rate, assist pct, turnover pct; shorten the decay if turnovers spike.
- Combine with Φ-shaping (hybrid w) to give direct pass credit and team coordination.

### Reverting
- To disable, drop the callback flags and switch the PPO policy class back to `"MultiInputPolicy"`.
- Or call `policy.set_pass_logit_bias(0.0)` in a one-off callback.

I'll scan the codebase to locate where rewards are computed in the environment and where shot probability/contest models live so I can add Phi(s) shaping and wire an exponential beta schedule from the trainer.

[3 tools called]

I'll look up how entropy scheduling callback is implemented to mirror it for beta, then read `HexagonBasketballEnv.step` and `_check_termination_and_rewards` to insert potential shaping.

[2 tools called]

I'll open the environment file around `step` and `_check_termination_and_rewards` to insert Phi-based shaping, and scan for an existing beta parameter.

[10 tools called]

I'll add potential-based reward shaping into the environment with configurable Phi options, wire new CLI args, and create a beta exponential scheduler callback that updates all vectorized env workers. Then I'll run lints for changed files.

[1 tool called]

I'll add a new PotentialBetaExpScheduleCallback to update `phi_beta` via exponential decay, wire new CLI args and environment setup, and include the callback in the training loop. Then I'll run lint checks for the edited files.

[21 tools called]

I’ll add potential-based shaping in the env, wire CLI flags, and create an exponential beta scheduler; then I’ll surface the shaping diagnostics in info so you can audit it during training.

- Implemented Phi(s) policy-invariant shaping in `HexagonBasketballEnv` with configurable options and applied at both the normal step path and early turnover path.
- Added `_phi_shot_quality()` that uses your existing shot probability model to compute team-best or ball-handler-only make probability.
- Added `PotentialBetaExpScheduleCallback` for exponential decay of beta across planned timesteps, with optional short “bump” at alternation starts (mirrors your entropy scheduler).
- Wired new CLI args and env constructor plumbing; scheduler attaches to both offense/defense segments.
- Shaping stats included in `info`: `phi_prev`, `phi_next`, `phi_beta` for debugging.

Code references:
```318:336:/home/ubuntu/basketworld/basketworld/envs/basketworld_env_v2.py
        # --- Potential-based shaping (Phi) configuration ---
        self.enable_phi_shaping: bool = bool(enable_phi_shaping)
        # Must match the agent's discount to preserve policy invariance
        self.reward_shaping_gamma: float = (
            float(reward_shaping_gamma) if reward_shaping_gamma is not None else 1.0
        )
        # Beta can be scheduled during training via VecEnv.set_attr
        self.phi_beta: float = float(phi_beta)
        # If True, use only the ball handler's make prob; else best among offense
        self.phi_use_ball_handler_only: bool = bool(phi_use_ball_handler_only)
```

```612:632:/home/ubuntu/basketworld/basketworld/envs/basketworld_env_v2.py
        # Apply potential-based reward shaping R' = R + beta*(gamma*Phi(s') - Phi(s))
        if self.enable_phi_shaping:
            try:
                phi_next = float(self._phi_shot_quality())
            except Exception:
                phi_next = 0.0
            r_shape = (
                float(self.reward_shaping_gamma) * float(phi_next)
                - float(phi_prev if phi_prev is not None else 0.0)
            )
            shaped = float(self.phi_beta) * float(r_shape)
            rewards[self.offense_ids] += shaped / self.players_per_side
            rewards[self.defense_ids] -= shaped / self.players_per_side
            info["phi_prev"] = float(phi_prev if phi_prev is not None else 0.0)
            info["phi_next"] = float(phi_next)
            info["phi_beta"] = float(self.phi_beta)
```

```1502:1536:/home/ubuntu/basketworld/basketworld/envs/basketworld_env_v2.py
    # -------------------- Potential Function Phi(s) --------------------
    def _phi_shot_quality(self) -> float:
        """Potential function Phi(s): team's current best shot quality.
...
        if self.phi_use_ball_handler_only:
            player_id = int(self.ball_holder)
            player_pos = self.positions[player_id]
            dist = self._hex_distance(player_pos, self.basket_position)
            return float(self._calculate_shot_probability(player_id, dist))
        # Else, best make probability among teammates on current possession team
        best = 0.0
        for pid in team_ids:
            player_pos = self.positions[pid]
            dist = self._hex_distance(player_pos, self.basket_position)
            p = float(self._calculate_shot_probability(pid, dist))
            if p > best:
                best = p
        return float(best)
```

```247:321:/home/ubuntu/basketworld/basketworld/utils/callbacks.py
class PotentialBetaExpScheduleCallback(BaseCallback):
    """Exponential decay for phi_beta across the planned training timesteps.
...
    def _apply_current(self):
        try:
            t = int(getattr(self.model, "num_timesteps", 0))
            current = self._scheduled_value(t)
            if self._bump_updates_remaining > 0:
                current = float(current * self.bump_multiplier)
            # Update all env workers' phi_beta
            vecenv = self.model.get_env()
            if vecenv is not None and hasattr(vecenv, "set_attr"):
                try:
                    vecenv.set_attr("phi_beta", float(current))
                except Exception:
                    pass
        except Exception:
            pass
```

```212:248:/home/ubuntu/basketworld/train/train.py
    env = basketworld.HexagonBasketballEnv(
...
        # Phi shaping config
        enable_phi_shaping=getattr(args, "enable_phi_shaping", False),
        reward_shaping_gamma=getattr(args, "reward_shaping_gamma", args.gamma),
        phi_beta=getattr(args, "phi_beta_start", 0.0),
        phi_use_ball_handler_only=getattr(args, "phi_use_ball_handler_only", False),
```

How to use it:
- Key flags
  - enable shaping: `--enable-phi-shaping`
  - ensure policy invariance: set `--reward-shaping-gamma` equal to PPO `--gamma` (default already uses `args.gamma`)
  - choose potential: `--phi-use-ball-handler-only` (False = team-best, True = ball-handler-only)
  - schedule beta exponentially from start to end across the planned timesteps:
    - `--phi-beta-start 0.15 --phi-beta-end 0.0 --phi-beta-schedule exp`
    - optional “bump” at alternation starts: `--phi-bump-updates 2 --phi-bump-multiplier 1.5`

Example:
```bash
python /home/ubuntu/basketworld/train/train.py \
  --alternations 20 --steps-per-alternation 1 --num-envs 8 --n-steps 2048 \
  --gamma 0.99 \
  --enable-phi-shaping true \
  --reward-shaping-gamma 0.99 \
  --phi-use-ball-handler-only false \
  --phi-beta-start 0.15 --phi-beta-end 0.0 --phi-beta-schedule exp \
  --phi-bump-updates 1 --phi-bump-multiplier 1.25
```

Notes:
- This shaping is policy-invariant if `reward_shaping_gamma` equals the agent’s discount.
- Start beta in the 0.05–0.2 range; anneal toward 0 over training as you suggested.
- Shaping applies symmetrically as team-averaged reward for the possession team to align with your existing team-aggregated reward scheme.

I found and edited the env to add Phi shaping and added a beta scheduler callback, plus CLI and trainer wiring. I also included shaping diagnostics in `info`. Existing linter warnings/errors shown are from unrelated legacy code; the shaping-specific paths type-check and don’t add new type errors.

- Added potential-based shaping in `HexagonBasketballEnv` and `_phi_shot_quality`.
- Created `PotentialBetaExpScheduleCallback` and attached it during both segments.
- Added CLI flags and `setup_environment` wiring; shaping gamma defaults to PPO gamma.
- Included `phi_prev`, `phi_next`, `phi_beta` in info for auditability.

I'll update Phi(s) to use expected points (2 vs 3 weighting) by modifying `_phi_shot_quality` in the environment.

[1 tool called]

I updated Φ(s) to use expected points instead of raw make probability.

- Φ now computes, for each candidate shooter, shot_value × p_make where shot_value is 3 beyond the arc (non-dunk) and 2 otherwise; then uses the max among teammates (or ball-handler only if that mode is enabled).
- This keeps pressure effects intact and properly prioritizes 3PT opportunities over 2PT.

If you want Φ normalized (e.g., divide by 3) to keep the shaping term’s scale closer to [0, 1], I can add a flag; otherwise you may want to slightly reduce your initial β (e.g., from 0.15 to ~0.05–0.10).

### What Φ-shaping is rewarding
- It’s dense: you get shaping every step, not just on shots.
- It rewards moving to states with higher expected points Φ(s) (we use team-best EP unless you set ball-handler-only).
- Any action that raises team-best shot quality (drive, off-ball cut, screen, pass) yields positive shaping; drops yield negative shaping; no change ≈ 0.

### Why it still preserves your objective
- Shaping each step is F(s,a,s′) = γ·Φ(s′) − Φ(s).
- Across a possession, these terms telescope; sum depends only on start/end potentials, not on which actions you took. With γ matching PPO’s γ, the optimal policy doesn’t change.

### Concrete example (numbers)
Assume:
- γ = 0.99, β = 0.1
- Φ is team-best expected points (EP): 2pt=2×p, 3pt=3×p
- Start possession Φ(s0)=0.80

Step 1 (drive draws help; corner teammate gets more open)
- Φ prev = 0.80 → Φ next = 1.20
- Shaping = β·(γ·1.20 − 0.80) = 0.1·(1.188 − 0.80) = +0.0388

Step 2 (kick-out pass; team-best stays 1.20)
- Φ prev = 1.20 → Φ next = 1.20
- Shaping = 0.1·(0.99·1.20 − 1.20) ≈ −0.012 (small drift due to γ<1; if the pass strictly increased EP it’d be positive)

Step 3 (shot ends episode; terminal Φ=0 by convention)
- Φ prev = 1.20 → Φ next = 0
- Shaping = 0.1·(0 − 1.20) = −0.12
- Base reward: +3 if the 3 goes in (team-averaged), else 0 (or your miss penalty).

Interpretation:
- Early steps that improve shot quality are rewarded (credit to drives, passes, off-ball).
- The terminal step “pays back” the accumulated potential (negative), so the total shaping over the possession depends only on start/end Φ, not the path. That’s why it doesn’t change the optimal policy, but it does move credit earlier to the actions that created the shot.

### About teammate EP without a pass
- Yes, with team-best Φ, you get positive shaping even if the ball-handler doesn’t pass but a teammate’s EP improves (e.g., spacing/cuts). If you want shaping to reflect only the ball-handler’s current look, set `phi_use_ball_handler_only=true`.

### When to use which Φ
- Team-best EP: encourages coordinated play (spacing, cuts, kick-outs).
- Ball-handler EP: pushes the handler to improve their own look or pass when theirs is poor.

If you want, I can add a log for per-player EPs each step so you can see exactly which player drove the Φ increase.

Short answer: With team-best Φ, passing is encouraged only when the pass increases team-best EP. If a teammate already has the best EP and a pass doesn’t change that EP immediately, the shaping at the pass step is ≈0, so credit is more diffuse (you still get shaping when the pass leads to a further EP rise, or when the shot happens). Ball‑handler‑only Φ gives an immediate positive bump when transferring possession to the higher‑EP teammate, so it directly incentivizes the pass.

If you want team‑coordination but also direct pass credit, use a hybrid Φ that’s still a state-only function:
- Option A: Φ(s) = max_team_EP(s) + α·max(0, max_team_EP(s) − ball_handler_EP(s))
- Option B: Φ(s) = (1−w)·max_team_EP(s) + w·ball_handler_EP(s), with small w (e.g., 0.2)

Both preserve policy invariance (state-only) and give an immediate positive shaping when the handler is below the best option (favoring passes), while retaining team-best incentives for spacing/off-ball movement.

- Team-best Φ alone: may not directly incentivize passing if the teammate already has the highest EP and a pass doesn’t increase team-best EP at that moment. It can still lead to passing indirectly (when the pass increases EP or enables the shot), but the pass itself often gets ≈0 shaping.
- Ball-handler-only Φ: directly incentivizes passing when a teammate’s EP > ball-handler’s EP, because Φ jumps to the recipient’s EP right after the pass.

Recommendation:
- If your goal is “pass when a teammate has a better shot,” use ball-handler-only Φ or a hybrid:
  - Hybrid (policy‑invariant, state‑only):
    - Φ(s) = max_team_EP(s) + α · max(0, max_team_EP(s) − ball_handler_EP(s)) with small α (e.g., 0.2), or
    - Φ(s) = (1−w)·max_team_EP(s) + w·ball_handler_EP(s) with w≈0.2–0.3.
- Practical: start with ball-handler‑only (or hybrid with α/w > 0) to teach passing, then anneal α/w → 0 to revert toward pure team‑best for broader coordination.

Use a hybrid Φ so you get both behaviors.

- What to use
  - Φ(s) = (1 − w) · max_team_EP(s) + w · ball_handler_EP(s)
  - Or Φ(s) = max_team_EP(s) + α · max(0, max_team_EP(s) − ball_handler_EP(s))

- How it helps
  - The max_team_EP term rewards off-ball movement/spacing (teammates creating better looks).
  - The ball_handler term (or the gap term) gives immediate credit for passing when a teammate’s shot is better.

- Recommended settings
  - Start with w ≈ 0.2–0.4 (or α ≈ 0.2–0.4), β ≈ 0.1–0.2.
  - Anneal w (or α) → 0 over training to drift toward pure team-best once passing is learned.
  - Keep γ equal to the agent’s γ.

If you want, I can swap the boolean to a slider and wire the hybrid in the env and UI.

[10 tools called]

Added a hybrid potential so you get both behaviors (passing + off-ball movement).

- Env
  - New `phi_blend_weight` in `HexagonBasketballEnv`:
    - Φ(s) = (1 − w)·team_best_EP + w·ball_handler_EP, clamped w∈[0,1].
    - If `phi_use_ball_handler_only=true`, it still forces ball-only (w=1 effective).
  - Terminal Φnext is now 0 for all terminal steps (made miss/turnover/shot-clock), preserving policy invariance.
  - r_shape/team is recorded in `info` and shown in the Phi tab.

- Trainer
  - CLI flag `--phi-blend-weight` and wiring into env setup.

How to use
- In training: `--phi-blend-weight 0.2` (and leave `--phi-use-ball-handler-only` false). This encourages off-ball spacing (team_best term) and gives immediate credit when handler’s EP is below the best (ball term).
- In UI (Phi tab): keep Enable on, adjust Beta/Gamma as before; blend exposed via the CLI for now. I can add a UI slider next if you want live tuning.

I'll add a blend weight slider to the Phi Shaping tab and wire it to backend params (get/set) so it persists and takes effect immediately.

[4 tools called]

Added a blend weight slider to the Phi Shaping tab.

- Backend
  - Exposed `phi_blend_weight` on GET/POST `/api/phi_params`.
  - Clamps to [0,1] and applies immediately.

- Env
  - Φ(s) now supports blending:
    - Φ = (1 − w)·team_best_EP + w·ball_handler_EP
    - If ball-handler-only is enabled, it overrides the blend (w=1 effective).

- Frontend
  - Phi tab shows “Blend w (Team vs Ball Φ)” slider (0→team-best only, 1→ball-only).
  - Apply button sends the value to backend; step refresh only refreshes logs, not params.

Use case:
- Start w≈0.2–0.4 to encourage passing and off-ball movement together; anneal toward 0 as policies mature.

- At each step, the policy gets a reward = base task reward (points or PPP/residual) + shaping r_shape/team = β·(γ·Φnext − Φprev)/players.
- PPO computes returns/advantages (e.g., GAE) from these per-step rewards. Actions that increased Φ (positive r_shape/team) get higher advantages; actions that decreased Φ get lower/negative advantages.
- The actor update pushes up the probability of actions with positive advantages and down for negative ones. Concretely:
  - Drive that raises team-best EP → positive shaping → higher advantage → the policy repeats that drive in similar states.
  - Off-ball cut/screen that raises team-best EP → same reinforcement → teammates learn to move into better spots.
  - Pass when teammate’s EP > ball-handler’s EP:
    - With ball-handler-only or hybrid (w>0), the moment the ball transfers, Φ typically jumps toward the recipient’s EP → positive shaping at the pass step → pass is directly credited.
    - With pure team-best only, passing is directly credited only if team-best EP actually increases due to the pass (e.g., lowers pressure or sets up the shot); otherwise credit arrives when the subsequent action raises Φ.
- The terminal step has negative shaping “payback,” but the outcome reward (2/3 or PPP/residual) is what determines shot selection; shaping mainly redistributes credit to earlier creators.
- Over training, this shifts the policy toward sequences that monotonically raise Φ: drive to draw help, off-ball positioning to raise team-best EP, then pass/shoot when it maximizes expected points. Adjust w to balance direct pass incentives (w↑) with team coordination (w↓), and anneal β/w down once behaviors are learned.

- It won’t sum to 0. With potential shaping r_shape = β·(γ·Φnext − Φprev)/players, the episode sum telescopes to:
  - If γ = 1 and Φ(terminal) = 0: Total = −β · Φ(start) / players
  - If γ < 1 and Φ(terminal) = 0: Total = −β · Φ(start) / players + β · (γ − 1) · Σt>0 Φ(t)/players, which is even more negative because (γ − 1) < 0 and Φ ≥ 0.

- Why it’s always ≤ 0 in your tables:
  - Φ(start) ≥ 0, so −β·Φ(start)/players ≤ 0.
  - With γ < 1, the extra (γ − 1)·ΣΦ term makes it more negative.
  - Directional credit still works: steps that increase Φ get positive r_shape; the terminal “payback” makes the total a constant offset determined by the start Φ, not by the action sequence.

- Why that’s OK:
  - PPO optimizes advantages; the constant episode offset does not change which actions are good. The shaping still moves credit earlier (drives, cuts, passes that raise Φ), while the base reward (2/3 pts or PPP/residual) decides the outcome.

If you want the table to feel “zero-sum,” we can add a derived column that shows cumulative shaping plus a constant +β·Φ(start)/players, which will end at ~0 by construction.