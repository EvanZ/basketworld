function normalizeIds(values) {
  if (!Array.isArray(values)) return [];
  return Array.from(new Set(
    values
      .map((value) => Number(value))
      .filter((value) => Number.isFinite(value)),
  )).sort((a, b) => a - b);
}

export function stableEvalTeamIds(gameState, teamKey, perPlayerStats = {}) {
  const state = gameState || {};
  const allIds = normalizeIds([
    ...(state.team_a_ids || []),
    ...(state.team_b_ids || []),
    ...(state.offense_ids || []),
    ...(state.defense_ids || []),
    ...Object.keys(perPlayerStats || {}),
  ]);

  let playerIds = normalizeIds(state.user_player_ids || []);
  if (playerIds.length === 0) {
    if (state.enable_multi_possession) {
      playerIds = normalizeIds(state.team_a_ids || []);
    } else {
      const userTeam = String(state.user_team_name || 'OFFENSE').toUpperCase();
      playerIds = normalizeIds(
        userTeam === 'DEFENSE' ? state.defense_ids : state.offense_ids,
      );
    }
  }

  if (String(teamKey || 'player').toLowerCase() === 'player') {
    return playerIds;
  }
  const playerSet = new Set(playerIds);
  return allIds.filter((playerId) => !playerSet.has(playerId));
}

export function aggregatePlayerShotCharts(perPlayerStats, playerIds) {
  const result = {};
  for (const rawPlayerId of playerIds || []) {
    const playerId = Number(rawPlayerId);
    if (!Number.isFinite(playerId)) continue;
    const stats = perPlayerStats?.[playerId] || perPlayerStats?.[String(playerId)];
    for (const [location, values] of Object.entries(stats?.shot_chart || {})) {
      if (!Array.isArray(values) || values.length < 2) continue;
      const target = result[String(location)] || [0, 0];
      target[0] += Number(values[0]) || 0;
      target[1] += Number(values[1]) || 0;
      result[String(location)] = target;
    }
  }
  return result;
}
