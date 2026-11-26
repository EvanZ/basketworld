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

export function shotReleaseGesture(isDunk, distance) {
  if (isDunk) return { kind: 'dunk-release', label: 'Two-handed slam' };
  if (Number(distance) <= 1) return { kind: 'layup-release', label: 'Basket-side layup' };
  return { kind: 'shot-release', label: 'Releasing shot' };
}

/** Position contextual hands, including shot-specific release gestures. */
export function interactionHandPairPose({
  center,
  basket,
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
  let showLeft = true;
  let showRight = true;
  let jazzAngleDeg = Math.sin(t * Math.PI * 4) * 7;

  if (normalizedKind === 'block-swat') {
    const reach = Math.sin(Math.PI * ramp(t, 0, 0.85));
    showLeft = finiteOr(basket?.x, cx) <= cx;
    showRight = !showLeft;
    sideOffset = radius * (0.45 + 0.55 * reach);
    verticalOffset = -radius * (0.35 + 1.15 * reach);
    leftBaseAngleDeg = 90 + 70 * ramp(t, 0.30, 0.65);
    rightBaseAngleDeg = -leftBaseAngleDeg;
    jazzAngleDeg = 0;
  } else if (normalizedKind === 'dunk-release') {
    const lift = ramp(t, 0, 0.22);
    const slam = ramp(t, 0.26, 0.55);
    sideOffset = radius * (0.42 - 0.16 * lift + 0.12 * slam);
    verticalOffset = radius * (-0.45 - 0.80 * lift + 0.95 * slam);
    leftBaseAngleDeg = 90 + 35 * slam;
    rightBaseAngleDeg = -leftBaseAngleDeg;
    jazzAngleDeg = 0;
  } else if (normalizedKind === 'layup-release') {
    // Select the hand nearest the hoop, not a fixed anatomical hand. If the
    // player occupies the basket cell, use the left hand as the tie-breaker.
    const dx = finiteOr(basket?.x, cx) - cx;
    const dy = finiteOr(basket?.y, cy) - cy;
    showLeft = dx <= 0;
    showRight = !showLeft;
    const reach = ramp(t, 0.02, 0.50);
    sideOffset = radius * (0.68 + 0.45 * reach);
    verticalOffset = -radius * (0.15 + 0.65 * reach);
    const aim = Math.atan2(dy || -radius * 0.5, dx || -radius) * 180 / Math.PI;
    const fingerRoll = 18 * ramp(t, 0.40, 0.65);
    leftBaseAngleDeg = aim + 180 + fingerRoll;
    rightBaseAngleDeg = aim - fingerRoll;
    jazzAngleDeg = 0;
  } else if (normalizedKind === 'shot-release') {
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
    jazzAngleDeg,
    showLeft,
    showRight,
  };
}
