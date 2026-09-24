export function defensiveLaneViolationBannerPayload(actionResults) {
  if (!Array.isArray(actionResults?.defensive_lane_violations)
    || actionResults.defensive_lane_violations.length === 0) {
    return null;
  }
  return {
    text: 'Defensive 3 Seconds — Technical',
    made: false,
    kind: 'violation',
    reboundText: null,
  };
}
