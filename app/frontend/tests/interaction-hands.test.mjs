import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const source = await readFile(
  new URL('../src/utils/interactionHands.js', import.meta.url),
  'utf8',
);
const {
  interactionHandDuration,
  interactionHandOpacity,
  interactionHandPairPose,
} = await import(
  `data:text/javascript;base64,${Buffer.from(source).toString('base64')}`
);

test('release and catch gestures occupy opposite ends of an action', () => {
  assert.equal(interactionHandOpacity('release', 0), 1);
  assert.equal(interactionHandOpacity('release', 1), 0);
  assert.equal(interactionHandOpacity('catch', 0), 0);
  assert.equal(interactionHandOpacity('catch', 1), 1);
  assert.equal(interactionHandOpacity('release', 0.5), 1);
  assert.equal(interactionHandOpacity('catch', 0.5), 1);
});

test('interaction hands remain readable after short distance-gated flights', () => {
  assert.equal(interactionHandDuration(280), 850);
  assert.equal(interactionHandDuration(1200), 1200);
  assert.equal(interactionHandDuration(280, 700), 700);
});

test('jazz hands sit outside both player sides and scale from player radius', () => {
  const pose = interactionHandPairPose({
    center: { x: 10, y: 20 },
    playerRadius: 36,
    progress: 0,
  });

  assert.equal(pose.x, 10);
  assert.equal(pose.y, 20);
  assert.equal(pose.sideOffset, 36 * 0.93);
  assert.equal(pose.verticalOffset, 0);
  assert.equal(pose.handScale, 36 * 0.40);
  assert.equal(pose.leftBaseAngleDeg, 0);
  assert.equal(pose.rightBaseAngleDeg, 0);
  assert.equal(pose.jazzAngleDeg, 0);
});

test('shots raise close hands while rebounds use an intermediate angle', () => {
  const shot = interactionHandPairPose({
    center: { x: 0, y: 0 },
    kind: 'shot-release',
    playerRadius: 40,
    progress: 0,
  });
  assert.equal(shot.sideOffset, 40 * 0.32);
  assert.equal(shot.verticalOffset, -40 * 0.72);
  assert.equal(shot.leftBaseAngleDeg, 90);
  assert.equal(shot.rightBaseAngleDeg, -90);

  const rebound = interactionHandPairPose({
    center: { x: 0, y: 0 },
    kind: 'rebound-catch',
    playerRadius: 40,
    progress: 0,
  });
  assert.equal(rebound.sideOffset, 40 * 0.62);
  assert.equal(rebound.verticalOffset, -40 * 0.38);
  assert.equal(rebound.leftBaseAngleDeg, 45);
  assert.equal(rebound.rightBaseAngleDeg, -45);
});

test('jazz hand wiggle is deterministic from animation progress', () => {
  const first = interactionHandPairPose({
    center: { x: 50, y: 50 },
    playerRadius: 20,
    progress: 0.125,
  });
  const replay = interactionHandPairPose({
    center: { x: 50, y: 50 },
    playerRadius: 20,
    progress: 0.125,
  });

  assert.equal(first.jazzAngleDeg, 7);
  assert.deepEqual(replay, first);
});
