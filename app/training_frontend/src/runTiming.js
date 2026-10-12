export function estimateRemainingTrainingSeconds(run) {
  const completedUpdates = Number(run?.current_update)
  const targetUpdates = Number(run?.target_updates)
  const elapsedSeconds = Number(run?.training_elapsed_seconds)

  if (!Number.isFinite(completedUpdates)
    || !Number.isFinite(targetUpdates)
    || !Number.isFinite(elapsedSeconds)
    || targetUpdates <= 0
    || completedUpdates < 0
    || elapsedSeconds < 0) return null
  if (completedUpdates >= targetUpdates) return 0
  if (completedUpdates === 0 || elapsedSeconds === 0) return null

  return (elapsedSeconds / completedUpdates) * (targetUpdates - completedUpdates)
}

export function remainingTrainingTimeCaption(run, estimate) {
  if (estimate === 0) return 'complete'
  if (run?.status === 'paused') return 'active time · paused'
  if (run?.status === 'stopped') return 'stopped before completion'
  if (run?.status === 'failed') return 'run failed'
  if (estimate === null) return 'waiting for first update'
  return 'estimated active time'
}
