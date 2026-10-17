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

## Compatibility policy

JAX checkpoints are the only supported checkpoint format. Configuration and
MLflow metadata names remain stable unless a versioned migration explicitly
changes them. Additive multi-possession checkpoint schemas 2, 3, and 4 are
normalized to runtime schema 4; check-clearance schema 5 remains strict. No
Python/Gym environment or non-JAX checkpoint path remains as a runtime oracle.
