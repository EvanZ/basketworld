import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import { teamStrengthFixture } from './team-strength-fixtures.mjs';

const source = await readFile(new URL('../src/utils/teamStrength.js', import.meta.url), 'utf8');
const { calculateTeamStrength } = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);
const close = (actual, expected) => assert.ok(Math.abs(actual - expected) < 1e-10, `${actual} != ${expected}`);

test('centers skills, reverses SD scaling and sums players for 3v3, 4v4 and 5v5', () => {
  for (const size of [3, 4, 5]) {
    const { report, state } = teamStrengthFixture(size);
    const result = calculateTeamStrength(report, state);
    assert.equal(result.available, true);
    close(result.players[0].rating, 5); // (.5-.4)/.1*2 + (1-0)/2*6
    close(result.blue, size * 5);
    close(result.red, size * -2);
    close(result.advantage, size * 7);
    assert.deepEqual(result.players[0].contributions.map((c) => c.key), ['three_pt', 'rebound']);
    // Offense changes are not roster changes.
    state.offense_ids = state.team_b_ids;
    state.defense_ids = state.team_a_ids;
    close(calculateTeamStrength(report, state).blue, result.blue);
    state.player_rebound_skills[0] = 2;
    close(calculateTeamStrength(report, state).blue, result.blue + 3);
  }
});

test('rejects mismatched model, opponent, update, rosters and APM-only reports', () => {
  const changes = [
    (r, s) => { s.model_metadata.checkpoint_path = '/different/run/update_2500'; },
    (r, s) => { s.model_metadata.update_index = 3000; },
    (r, s) => { s.opponent_model_metadata.checkpoint_path = '/different/opponent'; },
    (r, s) => { s.opponent_model_metadata = {}; s.opponent_unified_policy_name = 'unknown'; },
    (r, s) => { s.team_b_ids = [3, 4, 5, 6]; },
    (r, s) => { s.team_a_ids.pop(); },
    (r) => { r.reproducibility.team_size = null; },
    (r) => { r.player_rating_fit = { r_squared: .9 }; },
    (r) => { r.skill_coefficients.pop(); },
    (r) => { r.skill_coefficients[1] = r.skill_coefficients[0]; },
  ];
  for (const change of changes) {
    const { report, state } = teamStrengthFixture();
    change(report, state);
    assert.equal(calculateTeamStrength(report, state).available, false);
  }
});

test('missing skills, nulls, nonfinite values and invalid coefficients do not become zero ratings', () => {
  for (const value of [null, undefined, NaN, Infinity, '']) {
    const { report, state } = teamStrengthFixture();
    state.player_shooting_skills[0].three_pt = value;
    assert.match(calculateTeamStrength(report, state).reason, /missing a valid three_pt/);
  }
  const { report, state } = teamStrengthFixture();
  report.skill_coefficients[0].pool_mean = null;
  assert.match(calculateTeamStrength(report, state).reason, /invalid coefficient/);
});

test('unvaried skills can only rate the same constant value', () => {
  const { report, state } = teamStrengthFixture();
  report.skill_coefficients[1].pool_standard_deviation = 0;
  report.skill_coefficients[1].net_rating_per_standard_deviation = 0;
  assert.match(calculateTeamStrength(report, state).reason, /did not vary rebound/);
  for (const id of [...state.team_a_ids, ...state.team_b_ids]) state.player_rebound_skills[id] = 0;
  assert.equal(calculateTeamStrength(report, state).available, true);
});

test('warns on extrapolation and different action settings while preserving signed skill effects', () => {
  const { report, state } = teamStrengthFixture();
  report.skill_coefficients[0].net_rating_per_standard_deviation = -2;
  state.player_rebound_skills[0] = 10;
  const result = calculateTeamStrength(report, state, { playerDeterministic: true, opponentDeterministic: false });
  assert.equal(result.available, true);
  assert.equal(result.warnings.length, 2);
  close(result.players[0].rating, 28);
});

test('no report or no game yields a clear unavailable state', () => {
  assert.equal(calculateTeamStrength(null, {}).available, false);
  assert.equal(calculateTeamStrength({}, null).available, false);
});
