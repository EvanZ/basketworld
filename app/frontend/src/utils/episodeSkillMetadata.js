const EPISODE_SKILL_FIELDS = [
  'player_episode_skills',
  'player_rebound_skills',
  'player_shooting_skills',
];

function hasField(state, field) {
  const value = state?.[field];
  return value !== null && typeof value === 'object';
}

/**
 * Resolve episode-stable player skills for compact live and replay states.
 * Fast-mode payloads include these fields on the initial state only, so later
 * replay frames inherit them from an earlier frame in the same episode.
 */
export function resolveEpisodeSkillMetadata(currentState, history = []) {
  const result = {};
  for (const field of EPISODE_SKILL_FIELDS) {
    if (hasField(currentState, field)) {
      result[field] = currentState[field];
      continue;
    }
    for (let index = history.length - 1; index >= 0; index -= 1) {
      if (hasField(history[index], field)) {
        result[field] = history[index][field];
        break;
      }
    }
  }
  return result;
}
