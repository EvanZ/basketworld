const TIMING_PROFILES = Object.freeze({
  pass: Object.freeze({ limits: [1, 3, 5], scales: [0.45, 0.65, 0.82, 1], minFrames: 2 }),
  shot: Object.freeze({ limits: [0, 2, 4], scales: [0.45, 0.65, 0.82, 1], minFrames: 2 }),
  rebound: Object.freeze({ limits: [1, 3, 5], scales: [0.45, 0.65, 0.82, 1], minFrames: 3 }),
});

export function hexDistance(a, b) {
  if (!Array.isArray(a) || !Array.isArray(b) || a.length < 2 || b.length < 2) return 0;
  const [q1, r1] = a.map(Number);
  const [q2, r2] = b.map(Number);
  if (![q1, r1, q2, r2].every(Number.isFinite)) return 0;
  return (Math.abs(q1 - q2) + Math.abs(q1 + r1 - q2 - r2) + Math.abs(r1 - r2)) / 2;
}

export function actionAnimationTiming(kind, distance, baseDurationMs) {
  const profile = TIMING_PROFILES[kind] || TIMING_PROFILES.pass;
  const normalizedDistance = Math.max(0, Number(distance) || 0);
  const index = profile.limits.findIndex((limit) => normalizedDistance <= limit);
  const scale = profile.scales[index < 0 ? profile.scales.length - 1 : index];
  const base = Math.max(1, Number(baseDurationMs) || 0);
  return {
    distance: normalizedDistance,
    scale,
    durationMs: Math.max(120, Math.round(base * scale)),
  };
}

export function actionAnimationFrameCount(kind, distance, maxFrames) {
  const profile = TIMING_PROFILES[kind] || TIMING_PROFILES.pass;
  const maximum = Math.max(1, Math.trunc(Number(maxFrames) || 1));
  const timing = actionAnimationTiming(kind, distance, 1);
  return Math.min(maximum, Math.max(Math.min(profile.minFrames, maximum), Math.round(maximum * timing.scale)));
}
