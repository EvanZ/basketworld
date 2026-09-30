import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const source = await readFile(new URL('../src/utils/episodeFrameRange.js', import.meta.url), 'utf8');
const {
  buildEpisodeFrameSelection,
  fullEpisodeFrameRange,
  normalizeEpisodeFrameRange,
  updateEpisodeFrameBoundary,
} = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);

test('defaults to the full episode and handles an empty recording', () => {
  assert.deepEqual(fullEpisodeFrameRange(0), {
    start: 0,
    end: 0,
    frameCount: 0,
    selectedFrameCount: 0,
  });
  assert.deepEqual(fullEpisodeFrameRange(31), {
    start: 1,
    end: 31,
    frameCount: 31,
    selectedFrameCount: 31,
  });
});

test('normalizes reversed and out-of-bounds ranges', () => {
  assert.deepEqual(normalizeEpisodeFrameRange(30, 20, 25), {
    start: 20,
    end: 25,
    frameCount: 25,
    selectedFrameCount: 6,
  });
  assert.deepEqual(normalizeEpisodeFrameRange(-4, 99, 25), {
    start: 1,
    end: 25,
    frameCount: 25,
    selectedFrameCount: 25,
  });
});

test('each slider boundary stops at the other boundary', () => {
  const range = { start: 10, end: 20 };
  assert.equal(updateEpisodeFrameBoundary(range, 'start', 25, 30).start, 20);
  assert.equal(updateEpisodeFrameBoundary(range, 'end', 5, 30).end, 10);
});

test('selects frames 20 through 30 inclusively and in order', () => {
  const states = Array.from({ length: 40 }, (_value, index) => ({ id: index + 1 }));
  const { range, entries } = buildEpisodeFrameSelection(states, 20, 30);

  assert.equal(range.selectedFrameCount, 11);
  assert.deepEqual(entries.map((entry) => entry.frameNumber),
    [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]);
  assert.deepEqual(entries.map((entry) => entry.state.id),
    [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]);
  assert.equal(entries[0].previousState.id, 19);
});

test('the default selection preserves full-episode export behavior', () => {
  const states = [{ id: 1 }, { id: 2 }, { id: 3 }];
  const full = fullEpisodeFrameRange(states.length);
  const { entries } = buildEpisodeFrameSelection(states, full.start, full.end);
  assert.deepEqual(entries.map((entry) => entry.state.id), [1, 2, 3]);
});
