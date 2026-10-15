export function historicalEvalUpdatesForTarget(updates, previousTarget, nextTarget) {
  if (!Array.isArray(updates) || !Number.isInteger(nextTarget) || nextTarget < 1) {
    return updates
  }

  const previousWasFinalMilestone = Number.isInteger(previousTarget)
    && updates.includes(previousTarget)
  const retained = updates.filter((update) => (
    Number.isInteger(update)
    && update > 0
  ))

  // Keep future milestones in the saved config so copying a shorter run cannot
  // lose them. The backend filters only the command used for this run.
  if (previousWasFinalMilestone) retained.push(nextTarget)
  return [...new Set(retained)].sort((left, right) => left - right)
}

export function parseHistoricalEvalUpdates(rawValue) {
  const raw = String(rawValue ?? '').trim()
  if (!raw) return { updates: [], error: '' }

  const items = raw.split(/[\s,]+/).filter(Boolean)
  if (items.some((item) => !/^\d+$/.test(item))) {
    return { updates: [], error: 'Elo checkpoints must be positive whole-number updates.' }
  }

  const updates = items.map(Number)
  if (updates.some((update) => !Number.isSafeInteger(update) || update < 1)) {
    return { updates: [], error: 'Elo checkpoints must be positive whole-number updates.' }
  }
  if (new Set(updates).size !== updates.length) {
    return { updates: [], error: 'Each added Elo checkpoint must appear only once.' }
  }
  return { updates: updates.sort((left, right) => left - right), error: '' }
}

export function continuationHistoricalEvalSchedule(
  existingUpdates,
  addedUpdates,
  previousTarget,
  nextTarget,
) {
  const existing = Array.isArray(existingUpdates)
    ? [...new Set(existingUpdates.filter((update) => Number.isSafeInteger(update) && update > 0))]
      .sort((left, right) => left - right)
    : []
  const added = Array.isArray(addedUpdates) ? addedUpdates : []

  if (!Number.isInteger(previousTarget) || !Number.isInteger(nextTarget)) {
    return { updates: existing, error: 'Enter at least one additional update.' }
  }
  if (added.some((update) => update <= previousTarget)) {
    return {
      updates: existing,
      error: `Added Elo checkpoints must be after U${previousTarget.toLocaleString()}.`,
    }
  }
  if (added.some((update) => update > nextTarget)) {
    return {
      updates: existing,
      error: `Added Elo checkpoints must be no later than U${nextTarget.toLocaleString()}.`,
    }
  }
  if (added.some((update) => existing.includes(update))) {
    return { updates: existing, error: 'An added Elo checkpoint is already in the saved schedule.' }
  }
  return {
    updates: [...new Set([...existing, ...added])].sort((left, right) => left - right),
    error: '',
  }
}
