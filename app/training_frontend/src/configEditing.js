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
