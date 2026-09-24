import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const source = await readFile(
  new URL('../src/utils/outcomeBanners.js', import.meta.url),
  'utf8',
);
const { defensiveLaneViolationBannerPayload } = await import(
  `data:text/javascript;base64,${Buffer.from(source).toString('base64')}`,
);

test('defensive three seconds is labeled as a technical violation, not turnover', () => {
  assert.deepEqual(
    defensiveLaneViolationBannerPayload({ defensive_lane_violations: [{ player_id: 6 }] }),
    {
      text: 'Defensive 3 Seconds — Technical',
      made: false,
      kind: 'violation',
      reboundText: null,
    },
  );
  assert.equal(defensiveLaneViolationBannerPayload({ defensive_lane_violations: [] }), null);
});
