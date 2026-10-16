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

Aliases such as `players_per_side`, `court_height`, `court_width`, and
`shot_clock_steps` are accepted at the boundary and converted to the canonical
metadata names. Validation occurs when the configuration is created and names
the invalid field.

## Static construction

`build_kernel_static_from_config(config, xp)` is the JAX-owned construction
entrypoint. It creates court coordinates, movement and distance tables,
three-point and clearance regions, lane masks, inbound/check positions, player
indices, template arrays, and all rule constants without constructing a Gym
environment.

`build_kernel_static_from_env(...)` remains temporarily as a parity adapter for
unmigrated callers. Issue #69 migrates reset/training, #70 migrates evaluation
and inference, and #71 migrates the interactive runtime and rendering. Issue
#72 removes the adapter and retired runtime after those migrations.

## Compatibility policy

JAX checkpoints are the only supported checkpoint format. Configuration and
MLflow metadata names remain stable unless a versioned migration explicitly
changes them. During the unwind, deterministic tests compare the JAX-owned
static output against the legacy implementation; the legacy implementation is
a temporary oracle, not a supported final runtime.
