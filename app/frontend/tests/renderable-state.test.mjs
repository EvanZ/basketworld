import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const source = await readFile(
  new URL('../src/utils/renderableState.js', import.meta.url),
  'utf8',
);
const { resolveRenderableState } = await import(
  `data:text/javascript;base64,${Buffer.from(source).toString('base64')}`,
);

test('a completed pass renders its post-step receiver rather than its animation origin', () => {
  const state = {
    positions: [[1, 1], [4, 2]],
    ball_holder: 1,
    last_action_results: {
      pre_action_positions: [[1, 1], [4, 2]],
      pre_action_ball_holder: 0,
      passes: { 0: { success: true, target: 1 } },
    },
  };

  assert.deepEqual(resolveRenderableState(state), {
    positions: state.positions,
    ballHolder: 1,
  });
});

test('a missing ball holder remains absent instead of being coerced to player zero', () => {
  assert.deepEqual(resolveRenderableState({ positions: [[1, 1]], ball_holder: null }), {
    positions: [[1, 1]],
    ballHolder: null,
  });
});
