function isObject(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

/**
 * Return whether a live game state represents a JAX policy.
 *
 * `model_backend` is the primary signal, but it can be absent from a compact
 * state update after a policy has already been loaded.  JAX inference exposes
 * a checkpoint path together with its actor-critic policy spec, which is an
 * equally authoritative fallback for UI capability gating.
 */
export function isJaxModelState(gameState) {
  const backend = String(gameState?.model_backend || '').toLowerCase();
  if (backend.startsWith('jax')) return true;

  const metadata = gameState?.model_metadata;
  if (!isObject(metadata) || !isObject(metadata.policy_spec)) return false;
  return Boolean(metadata.checkpoint_path || metadata.unified_checkpoint_path);
}
