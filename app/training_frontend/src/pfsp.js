export function pfspRows(metrics = [], { includeInactiveModes = false } = {}) {
  const values = new Map(metrics.map(({ name, latest }) => [name, latest]))
  const prefix = 'jax/train/pfsp/'
  const modeProbability = values.get('jax/train/opponent_deterministic_episode_prob')
  const snapshotUpdates = metrics.filter(({ name }) => name.startsWith(`${prefix}snapshot_update/`)).map(({ latest }) => latest)
  const latestSnapshot = snapshotUpdates.length ? Math.max(...snapshotUpdates) : null
  return metrics.filter(({ name }) => name.startsWith(`${prefix}w/`)).map(({ name, latest }) => {
    const [opponent, mode] = name.slice(`${prefix}w/`.length).split('/')
    const get = (field) => values.get(`${prefix}${field}/${opponent}/${mode}`) ?? null
    return { name, opponent, mode, w: latest, update: get('checkpoint_update'),
      probability: get('probability'), games: get('completed_games'), age: get('age_updates'),
      assignedGames: get('assigned_games'), snapshotUpdate: get('snapshot_update'), retiring: get('retiring') === 1,
      priorOnly: get('prior_only') === 1, effectiveGames: get('effective_games'),
      truncated: get('truncated_games') }
  }).filter(row => (latestSnapshot === null || row.snapshotUpdate === latestSnapshot)
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
        snapshotUpdate: values.get(`${prefix}snapshot_update/${opponent}`) }
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

export function metricDefinition(name) {
  if (name.includes('/pressure/')) return 'Live-state opportunities, split by current possession: learner attacking means pressure received; learner defending means pressure applied. ESV is expected points per hypothetical shot (not PPP). Uncontested ESV holds position and skill fixed with no shot contest; defended ESV uses the actual contest model. Reduction is their difference. Ball-handler samples exclude loose balls; offense averages on-court offensive players. Zero samples means unavailable. Deploy may use the same policy on both teams.'
  if (name.includes('/pfsp/')) return 'Pre-rollout PFSP sampling snapshot, separate sampled/deterministic opponents. w is a recent smoothed completed-game score (win 1, tie 0.5, loss 0), not an Elo rating. Prior-only w is 0.5. Evidence has a 500-update half-life, capped at 256 effective games plus a Beta(1,1) prior. Probability is conditional on opponent action mode. Counts and freshness refer to the evidence used for this distribution.'
  if (name.includes('/w_eval/')) return 'Empirical completed-game win score: (wins + 0.5 × ties) / completed games against this frozen opponent. Excludes truncations. Not the Elo-fitted score and never used to update training PFSP estimates.'
  return ''
}
