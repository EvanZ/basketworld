import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

const source = await readFile(
  new URL('../src/utils/evalTeamVisualizations.js', import.meta.url),
  'utf8',
);
const {
  aggregatePlayerShotCharts,
  stableEvalTeamIds,
} = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);

test('resolves stable Player and AI ids independently of the current offense', () => {
  const state = {
    enable_multi_possession: true,
    team_a_ids: [0, 1, 2, 3, 4],
    team_b_ids: [5, 6, 7, 8, 9],
    user_player_ids: [0, 1, 2, 3, 4],
    offense_ids: [5, 6, 7, 8, 9],
    defense_ids: [0, 1, 2, 3, 4],
  };

  assert.deepEqual(stableEvalTeamIds(state, 'player'), [0, 1, 2, 3, 4]);
  assert.deepEqual(stableEvalTeamIds(state, 'ai'), [5, 6, 7, 8, 9]);
});

test('aggregates shot charts only for the selected stable team', () => {
  const stats = {
    0: { shot_chart: { '1,2': [3, 2], '2,2': [1, 0] } },
    1: { shot_chart: { '1,2': [4, 1] } },
    5: { shot_chart: { '1,2': [20, 10], '5,5': [2, 2] } },
  };

  assert.deepEqual(aggregatePlayerShotCharts(stats, [0, 1]), {
    '1,2': [7, 3],
    '2,2': [1, 0],
  });
  assert.deepEqual(aggregatePlayerShotCharts(stats, [5]), {
    '1,2': [20, 10],
    '5,5': [2, 2],
  });
});
