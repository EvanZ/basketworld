"""JAX-native trainer package."""

from basketworld_jax.train.types import (
    EvalTrace,
    PPOBatch,
    RolloutOutput,
    TrajectoryBatch,
    TrainerConfig,
    build_ppo_batch,
    compute_gae_and_returns,
)

__all__ = [
    "EvalTrace",
    "PPOBatch",
    "RolloutOutput",
    "TrajectoryBatch",
    "TrainerConfig",
    "build_ppo_batch",
    "compute_gae_and_returns",
]
