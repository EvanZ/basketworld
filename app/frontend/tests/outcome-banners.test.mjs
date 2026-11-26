import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const source = await readFile(
  new URL('../src/utils/outcomeBanners.js', import.meta.url),
  'utf8',
);
const { defensiveLaneViolationBannerPayload, nonShotOutcomeBannerPayload } = await import(
  `data:text/javascript;base64,${Buffer.from(source).toString('base64')}`,
);

test('defensive three seconds is labeled as a technical violation, not turnover', () => {
  assert.deepEqual(
    defensiveLaneViolationBannerPayload({ defensive_lane_violations: [{ player_id: 6 }] }),
    {
      text: 'Defensive 3 Seconds — Technical',
      made: false,
      kind: 'violation',
      reboundText: '+1 point · Offense retains possession',
    },
  );
  assert.equal(defensiveLaneViolationBannerPayload({ defensive_lane_violations: [] }), null);
});

test('defensive technical wins over a canceled interception in older replays', () => {
  const results = {
    defensive_lane_violations: [{ player_id: 4 }],
    turnovers: [{ player_id: 0, reason: 'steal', stolen_by: 3 }],
  };
  assert.deepEqual(nonShotOutcomeBannerPayload(results), defensiveLaneViolationBannerPayload(results));
});

test('ordinary turnovers and check violations retain their labels', () => {
  for (const [reason, text, kind] of [
    ['steal', 'Steal', 'steal'],
    ['offensive_three_seconds', 'Turnover - Lane Violation', 'turnover'],
    ['pass_out_of_bounds', 'Turnover', 'turnover'],
  ]) {
    assert.deepEqual(nonShotOutcomeBannerPayload({ turnovers: [{ reason }] }),
      { text, made: false, kind, reboundText: null });
  }
  assert.equal(nonShotOutcomeBannerPayload({ checks: [{ violation: true }] }).text, 'Check Violation');
  assert.equal(nonShotOutcomeBannerPayload({}), null);
  assert.equal(nonShotOutcomeBannerPayload(null), null);
});
