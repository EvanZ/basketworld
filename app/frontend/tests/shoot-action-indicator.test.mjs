import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const gameBoardSource = await readFile(
  new URL('../src/components/GameBoard.vue', import.meta.url),
  'utf8',
);

test('the board does not render a target graphic for shoot actions', () => {
  assert.doesNotMatch(gameBoardSource, /type:\s*['"]shoot['"]/);
  assert.doesNotMatch(gameBoardSource, /action-shoot-(?:outer|middle|inner)/);
});
