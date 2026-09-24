import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

// Load the Vite ES module in Node without changing the app's package mode.
const source = await readFile(new URL('../src/utils/actionAnimationTiming.js', import.meta.url), 'utf8');
const {
  actionAnimationFrameCount,
  actionAnimationTiming,
  hexDistance,
} = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);

test('hex distance matches the axial court geometry', () => {
  assert.equal(hexDistance([0, 0], [1, -1]), 1);
  assert.equal(hexDistance([0, 0], [3, -2]), 3);
  assert.equal(hexDistance([0, 0], [5, 0]), 5);
});

test('passes and shots use less time and fewer capture frames at short range', () => {
  const shortPass = actionAnimationTiming('pass', 1, 1100);
  const longPass = actionAnimationTiming('pass', 6, 1100);
  const shortShot = actionAnimationTiming('shot', 0, 1100);
  const longShot = actionAnimationTiming('shot', 5, 1100);

  assert.equal(shortPass.durationMs, 495);
  assert.equal(longPass.durationMs, 1100);
  assert.equal(shortShot.durationMs, 495);
  assert.equal(longShot.durationMs, 1100);
  assert.ok(actionAnimationFrameCount('pass', 1, 7) < actionAnimationFrameCount('pass', 6, 7));
  assert.ok(actionAnimationFrameCount('shot', 0, 7) < actionAnimationFrameCount('shot', 5, 7));
});

test('rebound flights also scale by basket-to-winner distance', () => {
  const shortRebound = actionAnimationTiming('rebound', 1, 3000);
  const longRebound = actionAnimationTiming('rebound', 6, 3000);

  assert.equal(shortRebound.durationMs, 1350);
  assert.equal(longRebound.durationMs, 3000);
  assert.ok(actionAnimationFrameCount('rebound', 1, 9) < actionAnimationFrameCount('rebound', 6, 9));
});
