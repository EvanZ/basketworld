"""JAX-native latent intent learning helpers."""

from basketworld_jax.intent.discriminator import (
    IntentDiscriminatorSpec,
    apply_intent_bonus_to_rollout,
    build_intent_discriminator_spec,
    build_intent_discriminator_update_runner,
    build_intent_sample_dump,
    build_intent_step_features_from_rollout,
    compute_intent_beta,
    compute_normalized_intent_bonus,
    init_bonus_stats,
    init_intent_discriminator_params,
    update_bonus_stats,
)

__all__ = [
    "IntentDiscriminatorSpec",
    "apply_intent_bonus_to_rollout",
    "build_intent_discriminator_spec",
    "build_intent_discriminator_update_runner",
    "build_intent_sample_dump",
    "build_intent_step_features_from_rollout",
    "compute_intent_beta",
    "compute_normalized_intent_bonus",
    "init_bonus_stats",
    "init_intent_discriminator_params",
    "update_bonus_stats",
]
