import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const source = await readFile(
  new URL('../src/utils/episodeSkillMetadata.js', import.meta.url),
  'utf8',
);
const { resolveEpisodeSkillMetadata } = await import(
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
