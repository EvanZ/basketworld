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
  shotReleaseGesture,
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

test('dunks lift both hands then slam down', () => {
  const pose = (progress) => interactionHandPairPose({
    center: { x: 50, y: 50 }, kind: 'dunk-release', playerRadius: 30, progress,
  });
  const start = pose(0);
  const lifted = pose(0.25);
  const slam = pose(0.55);
  for (const frame of [start, lifted, slam]) {
    assert.equal(frame.showLeft, true);
    assert.equal(frame.showRight, true);
    assert.equal(frame.leftBaseAngleDeg, -frame.rightBaseAngleDeg);
    assert.equal(frame.jazzAngleDeg, 0);
  }
  assert.ok(lifted.verticalOffset < start.verticalOffset);
  assert.ok(slam.verticalOffset > lifted.verticalOffset);
});

test('layups use only the hand nearest the basket on either side', () => {
  for (const basketX of [0, 100]) {
    const options = {
      center: { x: 50, y: 50 }, basket: { x: basketX, y: 30 },
      kind: 'layup-release', playerRadius: 30,
    };
    const start = interactionHandPairPose({ ...options, progress: 0 });
    const release = interactionHandPairPose({ ...options, progress: 0.5 });
    assert.equal(release.showLeft, basketX < 50);
    assert.equal(release.showRight, basketX > 50);
    assert.ok(release.sideOffset > start.sideOffset);
    assert.ok(release.verticalOffset < start.verticalOffset);
    assert.equal(release.jazzAngleDeg, 0);
    assert.deepEqual(release, interactionHandPairPose({ ...options, progress: 0.5 }));
  }
});

test('basket-cell layups have a stable single-hand tie-breaker', () => {
  const pose = interactionHandPairPose({
    center: { x: 50, y: 50 }, basket: { x: 50, y: 50 },
    kind: 'layup-release', playerRadius: 30, progress: 0.4,
  });
  assert.equal(pose.showLeft, true);
  assert.equal(pose.showRight, false);
  assert.ok(Number.isFinite(pose.leftBaseAngleDeg));
});

test('selected shot determines the gesture for both live and captured frames', () => {
  assert.equal(shotReleaseGesture(true, 0).kind, 'dunk-release');
  assert.equal(shotReleaseGesture(false, 0).kind, 'layup-release');
  assert.equal(shotReleaseGesture(false, 1).kind, 'layup-release');
  assert.equal(shotReleaseGesture(false, 2).kind, 'shot-release');
});

test('a block swat raises only the basket-side hand', () => {
  const base = { center: { x: 0, y: 0 }, basket: { x: -30, y: 0 }, playerRadius: 20, kind: 'block-swat' };
  const early = interactionHandPairPose({ ...base, progress: 0 });
  const reach = interactionHandPairPose({ ...base, progress: 0.4 });
  assert.equal(reach.showLeft, true);
  assert.equal(reach.showRight, false);
  assert.ok(reach.verticalOffset < early.verticalOffset);
});
