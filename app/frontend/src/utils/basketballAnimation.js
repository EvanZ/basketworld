export const BASKETBALL_SPIN_TURNS_PER_HEX = Object.freeze({
  pass: 0.45,
  shot: 0.7,
  rebound: 0.35,
});

function clamp01(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return 0;
  return Math.max(0, Math.min(1, numeric));
}

/**
 * Return a deterministic SVG rotation for a ball-flight frame.
 *
 * Rotation depends only on explicit frame progress and travel distance. That
 * keeps live play, replay, screenshots, and GIF export on the same seam angle
 * without relying on a wall-clock CSS animation.
 */
export function basketballRotationDegrees(
  progress,
  distance,
  kind,
  direction = 1,
) {
  const normalizedProgress = clamp01(progress);
  const normalizedDistance = Math.max(0, Number(distance) || 0);
  const turnsPerHex = BASKETBALL_SPIN_TURNS_PER_HEX[kind] || 0;
  const spinDirection = Number(direction) < 0 ? -1 : 1;
  return normalizedProgress * normalizedDistance * turnsPerHex * 360 * spinDirection;
}
