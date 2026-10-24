import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const source = await readFile(
  new URL('../src/utils/episodeSkillMetadata.js', import.meta.url),
  'utf8',
);
const { hydrateEpisodeSkillMetadata, resolveEpisodeSkillMetadata } = await import(
  `data:text/javascript;base64,${Buffer.from(source).toString('base64')}`,
);

test('compact replay frames inherit episode-stable skill metadata', () => {
  const initial = {
    player_episode_skills: { 0: { passing: 1.2 } },
    player_rebound_skills: { 0: -0.4 },
    player_shooting_skills: { 0: { layup: 0.7 } },
  };
  const compactCurrent = { positions: [[2, 3]], ball_holder: 0 };

  assert.deepEqual(
    resolveEpisodeSkillMetadata(compactCurrent, [initial, compactCurrent]),
    {
      player_episode_skills: initial.player_episode_skills,
      player_rebound_skills: initial.player_rebound_skills,
      player_shooting_skills: initial.player_shooting_skills,
    },
  );
});

test('current frame metadata wins over older replay metadata', () => {
  const initial = { player_rebound_skills: { 0: -1 } };
  const current = { player_rebound_skills: { 0: 2 } };

  assert.deepEqual(
    resolveEpisodeSkillMetadata(current, [initial, current]),
    { player_rebound_skills: current.player_rebound_skills },
  );
});

test('automatic end-game replay keeps every skill field on compact and final frames', () => {
  const skills = {
    player_episode_skills: { 0: { passing: 1.2, pressure: -0.3, shot_contest: 0.1, steal: 0.2, ball_handling: 0.5 } },
    player_rebound_skills: { 0: -0.4 },
    player_shooting_skills: { 0: { layup: 0.7, three_pt: 0.4, dunk: 0.8 } },
    player_rebound_skill_specialists: { 0: true },
    team_a_shooting_pct_by_player: { layup: [0.7] },
    team_b_shooting_pct_by_player: { layup: [0.6] },
  };
  const recorded = [
    { ...skills, done: false, positions: [[1, 2]] },
    { done: false, positions: [[2, 2]] },
    { done: true, positions: [[3, 2]], box_score: { player: { score: 12 } } },
  ];
  const original = structuredClone(recorded);
  const replay = hydrateEpisodeSkillMetadata(recorded);
  for (const frame of replay) {
    for (const [field, values] of Object.entries(skills)) {
      assert.deepEqual(frame[field], values);
    }
  }
  assert.equal(replay.at(-1).done, true);
  assert.deepEqual(replay.at(-1).positions, [[3, 2]]);
  assert.deepEqual(replay.at(-1).box_score, recorded.at(-1).box_score);
  assert.deepEqual(recorded, original);
});

test('replay preserves skill changes without leaking future or previous-episode skills', () => {
  const replay = hydrateEpisodeSkillMetadata([
    { player_rebound_skills: { 0: -1 } },
    {},
    { player_rebound_skills: { 0: 2 } },
    { done: true },
  ]);
  assert.deepEqual(replay.map((frame) => frame.player_rebound_skills[0]), [-1, -1, 2, 2]);
  assert.deepEqual(hydrateEpisodeSkillMetadata([{ done: true }]), [{ done: true }]);
});
