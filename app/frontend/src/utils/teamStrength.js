const finite = (value) => typeof value === 'number' && Number.isFinite(value);
const path = (value) => String(value || '').replaceAll('\\', '/').replace(/\/+$/, '');
const checkpointPath = (metadata) => path(metadata?.checkpoint_path || metadata?.unified_checkpoint_path);

function skillsForPlayer(state, id) {
  return {
    ...state.player_episode_skills?.[id],
    ...state.player_shooting_skills?.[id],
    rebound: state.player_rebound_skills?.[id] ?? state.player_episode_skills?.[id]?.rebound,
  };
}

// The report's displayed effect is beta * pool SD. Undo that scaling and
// center on the same pooled profile as the direct model's player ratings.
// Never use the APM diagnostic or add individual confidence bounds together.
export function calculateTeamStrength(report, state, options = {}) {
  const unavailable = (reason) => ({ available: false, reason });
  if (!report) return unavailable('Select a completed study to rate the current teams.');
  if (!state?.enable_multi_possession) return unavailable('Load a multi-possession game to rate its teams.');
  if (report.player_rating_fit?.feature_definition !== 'sum(team_a_player_skills) - sum(team_b_player_skills)') {
    return unavailable('This report has no direct skill-to-game model. Select a newer Skill Ratings study.');
  }
  const size = report.reproducibility?.team_size;
  const rosters = [state.team_a_ids, state.team_b_ids];
  if (![3, 4, 5].includes(size)) return unavailable('The study does not record a supported team size.');
  if (rosters.some((ids) => !Array.isArray(ids) || ids.length !== size)) {
    return unavailable(`This study is for ${size}-on-${size}; select a study matching the current team size.`);
  }
  const ids = rosters.flat();
  if (ids.some((id) => !Number.isInteger(id) || id < 0) || new Set(ids).size !== size * 2) {
    return unavailable('Current team rosters are incomplete or overlap.');
  }
  const studyPath = path(report.summary?.checkpoint);
  const update = report.summary?.checkpoint_update;
  if (!studyPath || !Number.isInteger(update)) return unavailable('The study is missing its checkpoint identity.');
  const matches = (metadata) => checkpointPath(metadata) === studyPath
    && Number.isInteger(metadata?.update_index) && metadata.update_index === update;
  if (!matches(state.model_metadata)) {
    return unavailable('The selected study does not match the current player checkpoint.');
  }
  const opponent = state.opponent_model_metadata;
  if (checkpointPath(opponent)) {
    if (!matches(opponent)) return unavailable('This study uses one checkpoint for both teams. The current opponent uses a different checkpoint.');
  } else if (state.opponent_unified_policy_name
    && state.opponent_unified_policy_name !== state.unified_policy_name) {
    return unavailable('The current opponent checkpoint cannot be verified against this study.');
  }

  const trained = report.environment_contract?.shot_defense?.parameters;
  const current = state.shot_defense_parameters || {};
  if (trained && Object.entries(trained).some(([key, value]) => current[key] !== value)) {
    return unavailable('The study and current teams use different shot-defense rules.');
  }
  if (!trained && (current.rim_protection_enabled || current.contested_shooting_enabled)) {
    return unavailable('This study predates the current shot-defense rules.');
  }
  const descriptors = report.skill_descriptors || [];
  const coefficients = report.skill_coefficients || [];
  const keys = descriptors.map((skill) => skill.key);
  if (!keys.length || new Set(keys).size !== keys.length
    || coefficients.length !== keys.length
    || new Set(coefficients.map((skill) => skill.key)).size !== keys.length
    || coefficients.some((skill) => !keys.includes(skill.key))) {
    return unavailable('The study is missing coefficients for its full skill profile.');
  }
  const warnings = new Set();
  const players = [];
  for (const [index, roster] of rosters.entries()) {
    for (const id of roster) {
      const skills = skillsForPlayer(state, id);
      const contributions = [];
      for (const coefficient of coefficients) {
        const { key, pool_mean: mean, pool_standard_deviation: sd, net_rating_per_standard_deviation: effect } = coefficient;
        const value = skills[key];
        if (coefficient.status === 'not_estimable') {
          if (coefficient.reason === 'no_lineup_variation') return unavailable(`The study has no lineup variation for ${key}; extend its schedule before rating new teams.`);
          if (coefficient.reason === 'zero_pool_variance' && (!finite(value) || Math.abs(value - mean) > 1e-7)) {
            return unavailable(`The study did not vary ${key}, so it cannot rate this player.`);
          }
          warnings.add(`${coefficient.label || key}: ${coefficient.reason?.replaceAll('_', ' ')}.`);
          contributions.push({ key, label: coefficient.label || key, value: 0, status: coefficient.status });
          continue;
        }
        if (!finite(value)) return unavailable(`Player #${id} is missing a valid ${key} skill. Ratings need the full profile.`);
        if (!finite(mean) || !finite(sd) || sd < 0 || !finite(effect)) {
          return unavailable(`The study has an invalid coefficient for ${key}.`);
        }
        if (sd === 0 && (effect !== 0 || Math.abs(value - mean) > 1e-7)) {
          return unavailable(`The study did not vary ${key}, so it cannot rate the current value for player #${id}.`);
        }
        const contribution = sd > 0 ? (value - mean) / sd * effect : 0;
        if (!finite(contribution)) return unavailable(`The ${key} contribution is not finite.`);
        contributions.push({ key, label: coefficient.label || key, value: contribution });
        if (sd > 0 && Math.abs((value - mean) / sd) > 3) {
          warnings.add('Some current skills are more than three study standard deviations from average; these estimates extrapolate beyond typical study players.');
        }
      }
      const rating = contributions.reduce((sum, skill) => sum + skill.value, 0);
      players.push({ id, team: index === 0 ? 'Blue' : 'Red', rating, contributions });
    }
  }
  const blue = players.filter((p) => p.team === 'Blue').reduce((sum, p) => sum + p.rating, 0);
  const red = players.filter((p) => p.team === 'Red').reduce((sum, p) => sum + p.rating, 0);
  if (![blue, red, blue - red].every(finite)) return unavailable('Team ratings could not be calculated as finite values.');
  const direct = report.summary.action_mode === 'direct';
  if (typeof options.playerDeterministic === 'boolean' && typeof options.opponentDeterministic === 'boolean'
    && (options.playerDeterministic !== direct || options.opponentDeterministic !== direct)) {
    warnings.add(`The study used ${direct ? 'deterministic' : 'sampled'} actions; current action settings differ.`);
  }
  return { available: true, blue, red, advantage: blue - red, players, warnings: [...warnings] };
}
