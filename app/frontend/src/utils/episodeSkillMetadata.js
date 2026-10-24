const EPISODE_SKILL_FIELDS = [
  'player_episode_skills',
  'player_rebound_skills',
  'player_shooting_skills',
  'player_rebound_skill_specialists',
  'team_a_shooting_pct_by_player',
  'team_b_shooting_pct_by_player',
];

function hasField(state, field) {
  const value = state?.[field];
  return value !== null && typeof value === 'object';
}

/**
 * Resolve episode-stable player skills for compact live and replay states.
 * Fast-mode payloads omit these fields on intermediate states, so later
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

/** Make each replay frame usable by both the board and player inspectors. */
export function hydrateEpisodeSkillMetadata(states = []) {
  let skills = {};
  return states.map((state) => {
    // Use only this episode's preceding frames, never a future frame or the
    // currently displayed game (which may belong to another episode).
    skills = { ...skills, ...resolveEpisodeSkillMetadata(state) };
    return { ...state, ...skills };
  });
}
