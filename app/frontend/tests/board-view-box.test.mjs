import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const source = await readFile(
  new URL('../src/utils/boardViewBox.js', import.meta.url),
  'utf8',
);
const { stableBoardViewBox } = await import(
  `data:text/javascript;base64,${Buffer.from(source).toString('base64')}`,
);

const courtLayout = [
  { x: 0, y: 0 },
  { x: 62, y: 0 },
  { x: 0, y: 216 },
  { x: 62, y: 216 },
];

test('multi-possession board permanently reserves both sideline inbound rows', () => {
  const radius = 36;
  const [minX, minY, width, height] = stableBoardViewBox({
    courtLayout,
    minimalChrome: true,
    hexRadius: radius,
    multiPossession: true,
  }).split(' ').map(Number);

  const maxY = minY + height;
  const expectedMargin = radius * 1.8;

  assert.equal(minY, -(radius * 1.5) - expectedMargin);
  assert.equal(maxY, 216 + (radius * 1.5) + expectedMargin);
  assert.ok(minX < -(Math.sqrt(3) * radius));
  assert.ok(width > 0);
});

test('the multi-possession view box depends only on court geometry', () => {
  const input = {
    courtLayout,
    minimalChrome: false,
    hexRadius: 36,
    multiPossession: true,
  };

  const beforeInbound = stableBoardViewBox(input);
  const duringTopInbound = stableBoardViewBox(input);
  const duringBottomInbound = stableBoardViewBox(input);

  assert.equal(beforeInbound, duringTopInbound);
  assert.equal(beforeInbound, duringBottomInbound);
});

test('legacy board keeps the original court-only bounds', () => {
  assert.equal(
    stableBoardViewBox({
      courtLayout,
      minimalChrome: true,
      hexRadius: 36,
      multiPossession: false,
    }),
    '-64.8 -64.8 191.6 345.6',
  );
});
