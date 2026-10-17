import test from 'node:test';
import assert from 'node:assert/strict';

import { readFile } from 'node:fs/promises';

const source = await readFile(
  new URL('../src/utils/gameClockPresentation.js', import.meta.url),
  'utf8',
);
const {
  clockPresentation,
  formatGameClock,
  recordedEpisodeEndMode,
} = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);

test('legacy recorded states retain possession presentation', () => {
  assert.equal(recordedEpisodeEndMode({ enable_multi_possession: true }), 'possessions');
  assert.equal(clockPresentation({ game_clock_remaining: 300 }).gameClockMode, false);
})

test('clock presentation formats regulation, overtime, and running state', () => {
  assert.equal(formatGameClock(300), '5:00');
  assert.deepEqual(clockPresentation({
    episode_end_mode: 'game_clock',
    game_clock_remaining: 119,
    overtime_round: 2,
    game_phase: 'awaiting_check',
  }), {
    mode: 'game_clock',
    gameClockMode: true,
    gameClock: '1:59',
    periodLabel: 'OT 2',
    clockRunning: true,
  });
  assert.equal(clockPresentation({
    episode_end_mode: 'game_clock', game_phase: 'check_setup',
  }).clockRunning, false);
  assert.equal(clockPresentation({
    episode_end_mode: 'game_clock', game_phase: 'awaiting_inbound',
  }).clockRunning, false);
  assert.equal(clockPresentation({
    episode_end_mode: 'game_clock', game_phase: 'live', done: true,
  }).clockRunning, false);
})
