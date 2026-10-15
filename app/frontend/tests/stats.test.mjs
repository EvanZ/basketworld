import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const source = await readFile(new URL('../src/services/stats.js', import.meta.url), 'utf8');
const {
  addEpisodeCheckPickupStats,
  countTeamOffensivePossessions,
  getDefaultStats,
  getEpisodeMultiPossessionScoreTotals,
  getNativeMultiPossessionScoreTotals,
} = await import(
  `data:text/javascript;base64,${Buffer.from(source).toString('base64')}`,
);

test('accumulates check pickups by stable Player and AI teams', () => {
  const stats = getDefaultStats();
  addEpisodeCheckPickupStats(
    stats,
    {
      team_a: { opportunities: 4, pickups: 3 },
      team_b: { opportunities: 3, pickups: 1 },
    },
    { team_a: 'user', team_b: 'ai' },
  );
  addEpisodeCheckPickupStats(
    stats,
    {
      team_a: { opportunities: 2, pickups: 2 },
      // Pickups are capped at opportunities when a malformed payload arrives.
      team_b: { opportunities: 1, pickups: 9 },
    },
    { team_a: 'user', team_b: 'ai' },
  );

  assert.deepEqual(stats.checkPickups, {
    player: { opportunities: 6, pickups: 5 },
    ai: { opportunities: 4, pickups: 2 },
  });
});

test('splits legacy aggregate counters by the actual jump-ball starter', () => {
  assert.equal(countTeamOffensivePossessions(25, 'team_a', true), 13);
  assert.equal(countTeamOffensivePossessions(25, 'team_b', true), 12);
  assert.equal(countTeamOffensivePossessions(25, 'team_a', false), 12);
  assert.equal(countTeamOffensivePossessions(25, 'team_b', false), 13);
});

test('counts completed possessions only and rejects an unknown starter', () => {
  assert.equal(countTeamOffensivePossessions(24, 'team_a', true), 12);
  assert.equal(countTeamOffensivePossessions(0, 'team_a', true), 0);
  assert.equal(countTeamOffensivePossessions(25, null, true), 0);
});

test('uses stable player and AI score totals rather than dynamic offense totals', () => {
  const totals = getNativeMultiPossessionScoreTotals({
    completed_user_score_total: 18,
    completed_opponent_score_total: 27,
    completed_possessions_total: 50,
    completed_user_offensive_possessions: 25,
    completed_opponent_offensive_possessions: 25,
    // These dynamic-role totals are deliberately different and must not be
    // used to produce the Player PPP.
    total_offense_points: 58,
    total_defense_points: 0,
  });

  assert.deepEqual(totals, {
    playerPoints: 18,
    aiPoints: 27,
    totalPossessions: 50,
    playerOffensivePossessions: 25,
    aiOffensivePossessions: 25,
  });
  assert.equal(totals.playerPoints / totals.playerOffensivePossessions, 0.72);
  assert.equal(totals.aiPoints / totals.aiOffensivePossessions, 27 / 25);
});

test('recovers stable PPP totals from completed episode scoreboards', () => {
  const totals = getEpisodeMultiPossessionScoreTotals([
    {
      completed: true,
      game: {
        completed: true,
        user_score: 17,
        opponent_score: 23,
        completed_possessions: 50,
        team_a_completed_possessions: 25,
        team_b_completed_possessions: 25,
        starting_offense_team: 'team_a',
      },
    },
    {
      completed: true,
      game: {
        completed: true,
        user_score: 14,
        opponent_score: 4,
        completed_possessions: 50,
        team_a_completed_possessions: 25,
        team_b_completed_possessions: 25,
        starting_offense_team: 'team_b',
      },
    },
    {
      completed: false,
      game: {
        completed: false,
        user_score: 99,
        opponent_score: 99,
        completed_possessions: 24,
        starting_offense_team: 'team_a',
      },
    },
  ]);

  assert.deepEqual(totals, {
    playerPoints: 31,
    aiPoints: 27,
    totalPossessions: 100,
    playerOffensivePossessions: 50,
    aiOffensivePossessions: 50,
  });
});
