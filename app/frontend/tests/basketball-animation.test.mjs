import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const source = await readFile(
  new URL('../src/utils/basketballAnimation.js', import.meta.url),
  'utf8',
);
const {
  BASKETBALL_SPIN_TURNS_PER_HEX,
  basketballRotationDegrees,
} = await import(
  `data:text/javascript;base64,${Buffer.from(source).toString('base64')}`
);

test('ball rotation is deterministic from frame progress and distance', () => {
  const first = basketballRotationDegrees(0.5, 4, 'pass');
  const replay = basketballRotationDegrees(0.5, 4, 'pass');

  assert.equal(first, replay);
  assert.equal(first, 0.5 * 4 * BASKETBALL_SPIN_TURNS_PER_HEX.pass * 360);
});

test('longer flights make proportionally more turns', () => {
  const short = basketballRotationDegrees(1, 2, 'rebound');
  const long = basketballRotationDegrees(1, 6, 'rebound');

  assert.equal(long, short * 3);
});

test('shots use faster backspin than passes and rebounds', () => {
  const rebound = Math.abs(basketballRotationDegrees(1, 3, 'rebound'));
  const pass = Math.abs(basketballRotationDegrees(1, 3, 'pass'));
  const shot = basketballRotationDegrees(1, 3, 'shot', -1);

  assert.ok(rebound < pass);
  assert.ok(pass < Math.abs(shot));
  assert.ok(shot < 0);
});

test('stationary balls do not rotate and progress is clamped', () => {
  assert.equal(basketballRotationDegrees(1, 0, 'pass'), 0);
  assert.equal(basketballRotationDegrees(-1, 4, 'pass'), 0);
  assert.equal(
    basketballRotationDegrees(2, 4, 'pass'),
    basketballRotationDegrees(1, 4, 'pass'),
  );
});
