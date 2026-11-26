import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const source = await readFile(new URL('../src/utils/playByPlay.js', import.meta.url), 'utf8');
const { buildPlayByPlay, playByPlayForStep } = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);
const initial = {
  team_a_ids: [0, 1], team_b_ids: [2, 3], offense_ids: [0, 1], defense_ids: [2, 3],
  team_a_score: 0, team_b_score: 0, episode_end_mode: 'game_clock',
  game_clock_remaining: 300, overtime_round: 0, shot_clock: 24,
};
const step = (results, extra = {}) => ({ ...initial, game_clock_remaining: 299, last_action_results: results, ...extra });

test('shot types, hex distances, outcomes and completed assists are recorded', () => {
  const rows = playByPlayForStep(step({ shots: {
    0: { is_dunk: true, distance: 0, success: true, assist_full: true, assist_passer_id: 1 },
    1: { distance: 1, success: false, assist_full: true, assist_passer_id: 0 },
    2: { distance: 3, success: true, assist_full: true, assist_passer_id: null },
    3: { is_three: true, distance: 5, success: false },
  } }, { team_a_score: 2 }), initial, { 0: 'SMITH', 1: 'JONES' });
  assert.equal(rows[0].event, 'Dunk by #0 SMITH from 0 hexes: made. Assist: #1 JONES.');
  assert.match(rows[1].event, /Layup.*1 hex: missed/);
  assert.doesNotMatch(rows[1].event, /Assist/);
  assert.match(rows[2].event, /2-point shot.*3 hexes: made/);
  assert.doesNotMatch(rows[2].event, /Assist/);
  assert.match(rows[3].event, /3-point shot.*5 hexes: missed/);
  assert.deepEqual([rows[0].period, rows[0].clock, rows[0].score], ['Regulation', '4:59', '2–0']);
});

test('turnover and steal use stable rosters after possession changes; null is not player zero', () => {
  const rows = playByPlayForStep(step({ turnovers: [
    { player_id: 0, reason: 'steal', stolen_by: 2 },
    { player_id: 3, reason: 'pass_out_of_bounds', stolen_by: null },
    { player_id: null, reason: 'shot_clock_violation', stolen_by: null },
  ] }, { offense_ids: [2, 3], defense_ids: [0, 1] }), initial);
  assert.equal(rows[0].team, 'Team A');
  assert.equal(rows[0].event, 'Turnover by #0 (steal). Steal by #2.');
  assert.equal(rows[1].event, 'Turnover by #3 (pass out of bounds).');
  assert.equal(rows[2].team, '—');
  assert.equal(rows[2].kind, 'violation');
  assert.doesNotMatch(rows[2].event, /#0|Steal/);
});

test('lane, clearance, inbound and check violations are surfaced without duplicate offensive three-seconds', () => {
  const rows = playByPlayForStep(step({
    turnovers: [
      { player_id: 0, reason: 'offensive_three_seconds' },
      { player_id: 1, reason: 'clearance_violation' },
      { player_id: 1, reason: 'inbound_timeout' },
    ],
    offensive_lane_violations: [{ player_id: 0 }],
    defensive_lane_violations: [{ player_id: 2 }],
    checks: [
      { team: 'team_a', violation: true, defense_space_violation: true },
      { team: 'team_b', violation: true },
    ],
  }), initial);
  assert.equal(rows.length, 6);
  assert.equal(rows.filter((r) => /three-second/.test(r.event)).length, 1);
  assert.ok(rows.every((r) => r.kind === 'violation'));
  assert.equal(rows[4].team, 'Team B');
  assert.match(rows[4].event, /check-space/);
  assert.match(rows[5].event, /deadline violation; turnover/);
});

test('final-tick events retain the expired period clock when overtime resets it', () => {
  const results = { shots: { 0: { distance: 1, success: true } } };
  const row = playByPlayForStep(step(results, { overtime_round: 1, game_clock_remaining: 120 }),
    { ...initial, game_clock_remaining: 1 })[0];
  assert.deepEqual([row.period, row.clock], ['Regulation', '0:00']);
  const next = playByPlayForStep(step(results, { overtime_round: 2, game_clock_remaining: 120 }),
    { ...initial, overtime_round: 1, game_clock_remaining: 1 })[0];
  assert.deepEqual([next.period, next.clock], ['OT 1', '0:00']);
});

test('possession games label the pre-action shot clock instead of a newly reset clock', () => {
  const row = playByPlayForStep(step({ turnovers: [{ player_id: 1, reason: 'steal' }] },
    { episode_end_mode: 'possessions', shot_clock: 24 }), { ...initial, shot_clock: 7 })[0];
  assert.equal(row.clock, 'Shot clock 7');
  assert.equal(row.period, 'Possession');
});

test('history follows replay position, clears on new game, and keeps real repeated events', () => {
  const shot = step({ shots: { 0: { distance: 0, is_dunk: true, success: true } } });
  const steal = step({ turnovers: [{ player_id: 0, reason: 'steal', stolen_by: 2 }] });
  const history = [initial, shot, shot, steal];
  assert.equal(buildPlayByPlay(history).length, 3);
  assert.equal(buildPlayByPlay(history)[0].kind, 'steal');
  assert.equal(buildPlayByPlay(history.slice(0, 2)).length, 1);
  assert.deepEqual(buildPlayByPlay([initial]), []);
  assert.deepEqual(buildPlayByPlay([]), []);
  assert.equal(buildPlayByPlay(history, {}, 2).length, 2);
  assert.equal(buildPlayByPlay(history)[0].id, '3-0');
});

test('older recordings with missing fields do not invent distances, scores or assists', () => {
  const row = playByPlayForStep({ last_action_results: { shots: { 0: { success: true } } } })[0];
  assert.match(row.event, /distance unavailable/);
  assert.equal(row.clock, 'Shot clock —');
  assert.equal(row.score, '—');
});

test('blocks identify the shooter and blocker with shot type, distance and clock', () => {
  const rows = playByPlayForStep({
    episode_end_mode: 'game_clock', game_clock_remaining: 123,
    last_action_results: { shots: { 0: { success: false, blocked: true, blocker_id: 3, is_dunk: true, distance: 0 } } },
  });
  assert.equal(rows[0].kind, 'block');
  assert.equal(rows[0].clock, '2:03');
  assert.match(rows[0].event, /Dunk by #0 from 0 hexes: blocked by #3/);
});
