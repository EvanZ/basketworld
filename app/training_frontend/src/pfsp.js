export function pfspRows(metrics = []) {
  const values = new Map(metrics.map(({ name, latest }) => [name, latest]))
  const prefix = 'jax/train/pfsp/'
  return metrics.filter(({ name }) => name.startsWith(`${prefix}w/`)).map(({ name, latest }) => {
    const [opponent, mode] = name.slice(`${prefix}w/`.length).split('/')
    const get = (field) => values.get(`${prefix}${field}/${opponent}/${mode}`) ?? null
    return { name, opponent, mode, w: latest, update: get('checkpoint_update'),
      probability: get('probability'), games: get('completed_games'), age: get('age_updates'),
      priorOnly: get('prior_only') === 1, effectiveGames: get('effective_games'),
      truncated: get('truncated_games') }
  }).sort((a, b) => (a.update - b.update) || a.opponent.localeCompare(b.opponent) || a.mode.localeCompare(b.mode))
}

export function metricDefinition(name) {
  if (name.includes('/pressure/')) return 'Live-state opportunities, split by current possession: learner attacking means pressure received; learner defending means pressure applied. ESV is expected points per hypothetical shot (not PPP). Uncontested ESV holds position and skill fixed with no shot contest; defended ESV uses the actual contest model. Reduction is their difference. Ball-handler samples exclude loose balls; offense averages on-court offensive players. Zero samples means unavailable. Deploy may use the same policy on both teams.'
  if (name.includes('/pfsp/')) return 'Pre-rollout PFSP sampling snapshot, separate sampled/deterministic opponents. w is a recent smoothed completed-game score (win 1, tie 0.5, loss 0), not an Elo rating. Prior-only w is 0.5. Evidence has a 500-update half-life, capped at 256 effective games plus a Beta(1,1) prior. Probability is conditional on opponent action mode. Counts and freshness refer to the evidence used for this distribution.'
  if (name.includes('/w_eval/')) return 'Empirical completed-game win score: (wins + 0.5 × ties) / completed games against this frozen opponent. Excludes truncations. Not the Elo-fitted score and never used to update training PFSP estimates.'
  return ''
}
