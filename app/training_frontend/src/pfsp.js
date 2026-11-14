export function pfspRows(metrics = [], { includeInactiveModes = false } = {}) {
  const values = new Map(metrics.map(({ name, latest }) => [name, latest]))
  const prefix = 'jax/train/pfsp/'
  const modeProbability = values.get('jax/train/opponent_deterministic_episode_prob')
  const snapshotUpdates = metrics.filter(({ name }) => name.startsWith(`${prefix}snapshot_update/`)).map(({ latest }) => latest)
  const latestSnapshot = snapshotUpdates.length ? Math.max(...snapshotUpdates) : null
  return metrics.filter(({ name }) => name.startsWith(`${prefix}w/`)).map(({ name, latest }) => {
    const [opponent, mode] = name.slice(`${prefix}w/`.length).split('/')
    const get = (field) => values.get(`${prefix}${field}/${opponent}/${mode}`) ?? null
    const wins = get('wins'), winRateGames = get('win_rate_games')
    return { name, opponent, mode, w: latest, update: get('checkpoint_update'),
      wins, winRateGames, winRate: Number.isFinite(wins) && winRateGames > 0 ? wins / winRateGames : null,
      pointsScored: get('ppp_points_scored'), pointsAllowed: get('ppp_points_allowed'),
      offensivePossessions: get('ppp_offensive_possessions'), defensivePossessions: get('ppp_defensive_possessions'),
      pppStartUpdate: get('ppp_tracking_start_update'),
      pppScored: get('ppp_offensive_possessions') > 0 ? get('ppp_scored') : null,
      pppAllowed: get('ppp_defensive_possessions') > 0 ? get('ppp_allowed') : null,
      probability: get('probability'), games: get('completed_games'), age: get('age_updates'),
      assignedGames: get('assigned_games'), snapshotUpdate: get('snapshot_update'), retiring: get('retiring') === 1,
      priorOnly: get('prior_only') === 1, effectiveGames: get('effective_games'),
      truncated: get('truncated_games') }
  }).filter(row => (latestSnapshot === null || row.snapshotUpdate === latestSnapshot)
    && (includeInactiveModes || row.mode !== 'deterministic' || row.games !== 0)
    && (includeInactiveModes || !(modeProbability === 0 && row.mode === 'deterministic'))
    && (includeInactiveModes || !(modeProbability === 1 && row.mode === 'sampled')))
    .sort((a, b) => (a.update - b.update) || a.opponent.localeCompare(b.opponent) || a.mode.localeCompare(b.mode))
}

export function opponentUsage(metrics = []) {
  const values = new Map(metrics.map(({ name, latest }) => [name, latest]))
  const prefix = 'jax/train/opponent_cache/'
  const update = values.get(`${prefix}usage_update`)
  if (Number.isFinite(update)) {
    const rows = metrics.filter(({ name }) => name.startsWith(`${prefix}usage_percent/`)).map(({ name, latest }) => {
      const opponent = name.slice(`${prefix}usage_percent/`.length)
      return { opponent, name, usage: latest, update: values.get(`${prefix}checkpoint_update/${opponent}`),
        snapshotUpdate: values.get(`${prefix}snapshot_update/${opponent}`),
        assignedGames: values.get(`${prefix}assigned_games_latest/${opponent}`) }
    }).filter(row => row.snapshotUpdate === update && Number.isFinite(row.usage) && Number.isFinite(row.update))
      .sort((a, b) => a.update - b.update || a.opponent.localeCompare(b.opponent))
    return { rows, update, basis: 'latest rollout', games: values.get(`${prefix}assigned_games_latest`) ?? 0,
      capacity: values.get(`${prefix}capacity`) ?? null }
  }
  // Old runs logged lifetime assignments, not per-rollout usage. Label that
  // fallback explicitly; never present sampling probabilities as observations.
  const grouped = new Map()
  for (const row of pfspRows(metrics, { includeInactiveModes: true })) {
    if (!Number.isFinite(row.assignedGames) || !Number.isFinite(row.update)) continue
    const previous = grouped.get(row.opponent) || { ...row, assignedGames: 0 }
    grouped.set(row.opponent, { ...previous, assignedGames: previous.assignedGames + row.assignedGames })
  }
  const games = [...grouped.values()].reduce((sum, row) => sum + row.assignedGames, 0)
  return { rows: [...grouped.values()].map(row => ({ ...row, usage: games ? 100 * row.assignedGames / games : 0 })),
    update: null, basis: 'lifetime assignments (legacy run)', games, capacity: null }
}

export function opponentChartData(metrics = [], { basis = 'pooled' } = {}) {
  const usage = opponentUsage(metrics)
  const values = new Map(metrics.map(({ name, latest }) => [name, latest]))
  const currentRows = basis === 'current' ? usage.rows.flatMap(row => ['sampled', 'deterministic'].map(mode => {
    const get = field => values.get(`jax/train/pfsp/current_ppp_${field}/${row.opponent}/${mode}`)
    return { opponent: row.opponent, snapshotUpdate: get('update'), pointsScored: get('points_scored'), pointsAllowed: get('points_allowed'),
      offensivePossessions: get('offensive_possessions'), defensivePossessions: get('defensive_possessions') }
  })).filter(row => Number.isFinite(usage.update) && row.snapshotUpdate === usage.update) : []
  const totals = new Map()
  // Usage combines action modes, so combine PPP by summing score/possession
  // counters too. Averaging mode ratios or weighting by usage would bias PPP.
  for (const row of basis === 'current' ? currentRows : pfspRows(metrics, { includeInactiveModes: true })) {
    // The MLflow catalog is not an atomic snapshot: usage can advance just
    // before or after the cumulative PFSP counters. Keep the latest internally
    // consistent pooled snapshot visible through that brief skew. Current-
    // update counters remain strict because they describe one exact rollout.
    if (basis === 'current' && usage.update !== null && row.snapshotUpdate !== usage.update) continue
    const total = totals.get(row.opponent) ?? {
      pointsScored: 0, pointsAllowed: 0, offensivePossessions: 0, defensivePossessions: 0, pppStartUpdate: null,
    }
    let tracked = false
    for (const [points, possessions] of [['pointsScored', 'offensivePossessions'], ['pointsAllowed', 'defensivePossessions']]) {
      if (!Number.isFinite(row[points]) || row[points] < 0 || !Number.isFinite(row[possessions]) || row[possessions] < 0) continue
      total[points] += row[points]
      total[possessions] += row[possessions]
      tracked ||= row[possessions] > 0 || row[points] > 0
    }
    if (tracked && Number.isFinite(row.pppStartUpdate) && row.pppStartUpdate >= 0) {
      total.pppStartUpdate = total.pppStartUpdate === null ? row.pppStartUpdate : Math.min(total.pppStartUpdate, row.pppStartUpdate)
    }
    totals.set(row.opponent, total)
  }
  return { ...usage, currentAvailable: currentRows.length > 0, rows: usage.rows.map(row => {
    const total = totals.get(row.opponent)
    return { ...row, ...total,
      pppScored: total?.offensivePossessions > 0 ? total.pointsScored / total.offensivePossessions : null,
      pppAllowed: total?.defensivePossessions > 0 ? total.pointsAllowed / total.defensivePossessions : null,
    }
  }) }
}

export function metricDefinition(name) {
  if (name.includes('/pfsp/current_ppp_')) return 'PPP from this training update only, by opponent. Actual learner/opponent score deltas divided by their completed offensive possessions during these rollouts; includes violation points and ongoing/truncated games. No cumulative history, smoothing, or evaluation. Zero completed possessions means unavailable, not zero PPP.'
  if (name.includes('/pfsp/ppp_')) return 'Cumulative training efficiency against this opponent since PPP tracking began. PPP scored is learner points / learner completed offensive possessions; PPP allowed is opponent points / opponent completed offensive possessions. Actual score includes violation points, not shaped rewards. Includes observed steps and completed possessions in ongoing or truncated games, unlike game-level Win %. No decay or prior; not used for PFSP sampling.'
  if (name.includes('/pressure/')) return 'Live-state opportunities, split by current possession: learner attacking means pressure received; learner defending means pressure applied. ESV is expected points per hypothetical shot (not PPP). Uncontested ESV holds position and skill fixed with no shot contest; defended ESV uses the actual contest model. Reduction is their difference. Ball-handler samples exclude loose balls; offense averages on-court offensive players. Zero samples means unavailable. Deploy may use the same policy on both teams.'
  if (name.includes('/pfsp/')) return 'Pre-rollout PFSP sampling snapshot, separate sampled/deterministic opponents. w is a recent smoothed completed-game score (win 1, tie 0.5, loss 0), not an Elo rating. Prior-only w is 0.5. Evidence has a 500-update half-life, capped at 256 effective games plus a Beta(1,1) prior. Win % is actual learner wins / tracked completed games, without decay or a prior; ties are non-wins and truncations are excluded. Older runs may lack these counts; partial history is labeled. Probability is conditional on opponent action mode. Counts and freshness refer to the evidence used for this distribution.'
  if (name.includes('/w_eval/')) return 'Empirical completed-game win score: (wins + 0.5 × ties) / completed games against this frozen opponent. Excludes truncations. Not the Elo-fitted score and never used to update training PFSP estimates.'
  return ''
}
