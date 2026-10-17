# JAX Environment Ownership

BasketWorld's supported environment configuration belongs to the JAX stack.
The dependency direction is:

```text
basketworld.domain                 ActionType and Team
basketworld.geometry               dependency-neutral court math
        ↓
basketworld_jax.config.environment typed configuration and static court source
        ↓
basketworld_jax.env.minimal        KernelStatic, KernelState, reset, and step
```

None of these foundational modules may import Gymnasium, Stable-Baselines3,
PyTorch, `train.env_factory`, or `basketworld.envs.basketworld_env_v2`.

## Configuration contract

`JaxEnvironmentConfig` is the canonical typed input for JAX environment
construction. It covers the court, players, clocks, rules, skills, rewards,
intent settings, templates, rebounds, multi-possession behavior, and restart
settings used by `KernelStatic` and reset.

Use the adapter matching the caller:

- `from_args(...)` for the JAX trainer CLI namespace;
- `from_checkpoint_metadata(...)` for JAX checkpoint and MLflow metadata;
- `from_evaluation_request(...)` for native/fixed-pool evaluation;
- `from_dev_game_params(...)` for interactive backend requests;
- `from_mapping(...)` for an already normalized configuration mapping.

Aliases such as `players_per_side`, `court_height`, `court_width`,
`shot_clock_steps`, `phi_beta_start`, and `pass_oob_turnover_prob_start` are
accepted at the boundary and converted to the canonical metadata names.
Validation occurs when the configuration is created and names the invalid
field.

## Static construction

`build_kernel_static_from_config(config, xp)` is the JAX-owned construction
entrypoint. It creates court coordinates, movement and distance tables,
three-point and clearance regions, lane masks, inbound/check positions, player
indices, template arrays, and all rule constants without constructing a Gym
environment.

Training, native evaluation, historical-opponent evaluation, fixed-pool studies,
Skill Ratings, inference, and the interactive backend all construct static state
through `build_kernel_static_from_config(...)`.

## Evaluation and inference

Native evaluation resolves one `JaxEnvironmentConfig` from checkpoint/MLflow
metadata plus explicit request overrides. Historical checkpoint pairs are
cross-validated for court, player, observation, possession, check, and clock
semantics before execution. Fixed-pool and Skill Ratings studies recover the
same typed contract directly from checkpoint metadata.

`JAXInferenceModel` builds role-specific static state from checkpoint metadata
and can prepare directly from a JAX state snapshot mapping. Supported public
evaluation and inference loaders accept Orbax checkpoint directories with
`metadata.json` and `state/`; SB3/PyTorch artifacts receive an explicit
unsupported-format error.

## Reset and training construction

`sample_state_batch(args, xp)` converts the JAX trainer namespace through
`JaxEnvironmentConfig.from_args(...)`, builds `KernelStatic` with
`build_kernel_static_from_config(...)`, and initializes every batch row with
`reset_batch_minimal(...)`. Possession and game-clock modes therefore share the
same JAX-native seeded reset path. Training does not construct
`HexagonBasketballEnv`, import `train.env_factory`, or require
Stable-Baselines3.

The JAX trainer owns `episode_end_mode`, `game_clock_seconds`,
`overtime_seconds`, and `game_clock_max_total_steps`.
Compiled training selects a mode-specialized stepper when each rollout runner
is built: possession runs retain `step_batch_minimal`, while timed runs use
`step_batch_game_clock`. Safety truncations reset the environment but bootstrap
the learner from the final pre-reset observation. Native and historical timed
evaluation use the same clock stepper and size their compiled scans with an
explicit horizon. They report a safety-cap termination separately from a scan
horizon cutoff; neither contributes to completed-game win rates.

Game-clock policies observe seconds remaining divided by the fixed scale 720
and observing-team score difference divided by 30. These semantics and scales
are stored in checkpoint and MLflow observation metadata. Same-width policies
from possession and clock modes are rejected as incompatible unless a caller
uses an explicit adaptation path.

Training trajectories retain the raw period index and pre-step clock value for
each transition. MLflow training metrics use completed offensive possessions
for per-team efficiency, elapsed game-clock minutes for per-team pace, and
report regulation completions, overtime games/periods, terminal clock values,
four regulation scoring segments, overtime scoring, possession duration, and
safety truncations separately.

Native and historical evaluation use the same denominators. Offensive
efficiency is points per completed offensive possession. Pace is completed
offensive possessions per elapsed game-clock minute, including overtime and
clock time consumed during check pickup while excluding stopped setup/inbound
steps. Possession duration sums elapsed game-clock seconds through every
completed possession, including `period_expired`, exactly once. Ratios with no
denominator are reported as `0.0`. Completed and truncated groups are preserved
separately, and raw scoring/possession events retain period index and pre-step
seconds remaining for reanalysis.

Adjacent native evaluation rows share reset randomness, start with opposite
stable teams, and set opposite overtime inversion flags. Historical matches
reuse the same seeded game row across candidate Team A/Team B assignments, so
the episode-stable overtime draw is inverted from the candidate's perspective
without depending on how many steps either game used to reach overtime.

## Effective configuration and reproducibility

Checkpoint `env_config` and `resolved_training_config` preserve requested
settings. New checkpoints additionally store `effective_env_config`, including
the resolved shaping discount and rebound terminal reward mode. Evaluation and
inference prefer these effective values, then apply explicit request overrides.
Each input layer normalizes aliases before merging; for example, an explicit
`shot_clock_steps=18` overrides a checkpoint's `shot_clock=24`. Within one layer,
the canonical key wins when both spellings are present.

New MLflow runs log `jax/policy_seed` and effective `jax/env/*` values, plus
`metadata/effective_env_config.json`. Continuations preserve existing immutable
parameters and add an effective-config artifact under
`metadata/continuations/target_<update>_effective_env.json`. These environment
records retain the final possession-limit convention used by evaluation;
per-update curriculum schedules remain in the requested configuration.

New training inherits PPO `gamma` when `reward_shaping_gamma` is unset. The
pre-migration constructor instead used `1.0`. Older checkpoints with a null
request and no effective record cannot distinguish these cases when PPO gamma
is not `1.0`; resume requires an explicit `--reward-shaping-gamma` after checking
the source used for that run. For a pre-migration reproduction, set it to `1.0`.
Evaluation retains its existing historical `1.0` fallback for such metadata;
override it explicitly when the checkpoint was trained with the newer default.
Historical metadata is never rewritten. Similarly, old recorded rebound reward
requests may differ from the pre-migration kernel's always-`actual_points`
behavior; reproductions must explicitly select the effective historical mode.

## Compatibility policy

JAX checkpoints are the only supported checkpoint format. Configuration and
MLflow metadata names remain stable unless a versioned migration explicitly
changes them. Additive multi-possession checkpoint schemas 2, 3, and 4 are
normalized to runtime schema 4; check-clearance schema 5 remains strict. No
Python/Gym environment or non-JAX checkpoint path remains as a runtime oracle.
