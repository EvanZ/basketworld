function clamp01(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return 0;
  return Math.max(0, Math.min(1, numeric));
}

function smoothstep(value) {
  const t = clamp01(value);
  return t * t * (3 - 2 * t);
}

function ramp(progress, start, end) {
  const width = Math.max(1e-9, Number(end) - Number(start));
  return smoothstep((clamp01(progress) - Number(start)) / width);
}

function finiteOr(value, fallback) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : fallback;
}

/**
 * Opacity envelope for the contextual hand pair attached to a player.
 *
 * Release gestures are strongest at the start of the ball flight and fade as
 * the ball leaves the player. Catch gestures appear during the latter half of
 * the flight and remain visible through the catch frame. Keeping this entirely
 * progress-driven makes live animation and deterministic GIF capture agree.
 */
export function interactionHandOpacity(phase, progress) {
  const t = clamp01(progress);
  if (phase === 'release') {
    return 1 - ramp(t, 0.58, 0.82);
  }
  if (phase === 'catch') {
    return ramp(t, 0.18, 0.48);
  }
  return 0;
}

/** Keep short, distance-gated ball flights readable at normal play speed. */
export function interactionHandDuration(actionDurationMs, minimumDurationMs = 850) {
  const actionDuration = Math.max(0, finiteOr(actionDurationMs, 0));
  const minimumDuration = Math.max(0, finiteOr(minimumDurationMs, 850));
  return Math.max(actionDuration, minimumDuration);
}

/** Position a pair of cartoony jazz hands outside the player side edges. */
export function interactionHandPairPose({
  center,
  kind,
  playerRadius,
  progress,
}) {
  const cx = finiteOr(center?.x, 0);
  const cy = finiteOr(center?.y, 0);
  const radius = Math.max(0, finiteOr(playerRadius, 0));
  const t = clamp01(progress);
  const normalizedKind = String(kind || '').toLowerCase();

  let sideOffset = radius * 0.93;
  let verticalOffset = 0;
  let leftBaseAngleDeg = 0;
  let rightBaseAngleDeg = 0;

  if (normalizedKind === 'shot-release') {
    sideOffset = radius * 0.32;
    verticalOffset = -radius * 0.72;
    leftBaseAngleDeg = 90;
    rightBaseAngleDeg = -90;
  } else if (normalizedKind === 'rebound-catch') {
    sideOffset = radius * 0.62;
    verticalOffset = -radius * 0.38;
    leftBaseAngleDeg = 45;
    rightBaseAngleDeg = -45;
  }

  return {
    x: cx,
    y: cy,
    sideOffset,
    verticalOffset,
    handScale: radius * 0.40,
    leftBaseAngleDeg,
    rightBaseAngleDeg,
    jazzAngleDeg: Math.sin(t * Math.PI * 4) * 7,
  };
}
