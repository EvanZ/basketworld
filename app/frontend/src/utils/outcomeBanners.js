export function defensiveLaneViolationBannerPayload(actionResults) {
  if (!Array.isArray(actionResults?.defensive_lane_violations)
    || actionResults.defensive_lane_violations.length === 0) {
    return null;
  }
  return {
    text: 'Defensive 3 Seconds — Technical',
    made: false,
    kind: 'violation',
    reboundText: '+1 point · Offense retains possession',
  };
}

export function nonShotOutcomeBannerPayload(actionResults) {
  const checks = Array.isArray(actionResults?.checks) ? actionResults.checks : [];
  if (checks.some((entry) => entry?.violation)) {
    return { text: 'Check Violation', made: false, kind: 'violation', reboundText: null };
  }
  // The technical determines the restart, even in an older recording that
  // also contains the now-cancelled interception event.
  const technical = defensiveLaneViolationBannerPayload(actionResults);
  if (technical) return technical;
  const turnovers = Array.isArray(actionResults?.turnovers) ? actionResults.turnovers : [];
  const turnover = turnovers.find((entry) => entry && typeof entry === 'object');
  if (!turnover) return null;
  const reason = String(turnover.reason || '').toLowerCase();
  const isSteal = reason === 'steal';
  const isLaneViolation = reason === 'offensive_three_seconds'
    || reason === 'offensive_lane_violation'
    || Boolean(actionResults?.offensive_lane_violations?.length);
  return {
    text: isSteal ? 'Steal' : (isLaneViolation ? 'Turnover - Lane Violation' : 'Turnover'),
    made: false,
    kind: isSteal ? 'steal' : 'turnover',
    reboundText: null,
  };
}
