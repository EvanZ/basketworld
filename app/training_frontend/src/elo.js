export function prepareEloSeries(summary) {
  const raw = Array.isArray(summary?.points) ? summary.points : []
  return raw
    .filter((point) => point?.elo_rating !== null && point?.elo_rating !== '')
    .map((point) => ({ ...point, update: Number(point.update), elo_rating: Number(point.elo_rating) }))
    .filter((point) => Number.isFinite(point.update) && Number.isFinite(point.elo_rating))
    .sort((left, right) => left.update - right.update)
}

export function eloSparklineCoordinates(points, width = 420, height = 112, padding = 18) {
  if (!points.length) return []
  const values = points.map((point) => point.elo_rating)
  const updates = points.map((point) => point.update)
  const min = Math.min(...values)
  const max = Math.max(...values)
  const span = max - min
  const minUpdate = Math.min(...updates)
  const maxUpdate = Math.max(...updates)
  const updateSpan = maxUpdate - minUpdate
  return points.map((point) => ({
    ...point,
    x: updateSpan === 0
      ? width / 2
      : padding + ((point.update - minUpdate) / updateSpan) * (width - padding * 2),
    y: span === 0
      ? height / 2
      : padding + (1 - ((point.elo_rating - min) / span)) * (height - padding * 2),
  }))
}

export function eloSparklinePoints(points) {
  return eloSparklineCoordinates(points)
    .map((point) => `${point.x.toFixed(2)},${point.y.toFixed(2)}`)
    .join(' ')
}

export function eloWarningMessages(summary) {
  return Array.isArray(summary?.warnings)
    ? summary.warnings.filter((warning) => typeof warning === 'string' && warning.trim())
    : []
}

export function fitLogElo(points) {
  const samples = points
    .filter((point) => Number.isFinite(point?.update) && point.update > 0
      && Number.isFinite(point?.elo_rating))
    .map((point) => ({ x: Math.log(point.update), y: point.elo_rating }))
  if (samples.length < 3) return null
  const meanX = samples.reduce((sum, sample) => sum + sample.x, 0) / samples.length
  const meanY = samples.reduce((sum, sample) => sum + sample.y, 0) / samples.length
  const denominator = samples.reduce((sum, sample) => sum + ((sample.x - meanX) ** 2), 0)
  if (denominator <= 0) return null
  const slope = samples.reduce(
    (sum, sample) => sum + ((sample.x - meanX) * (sample.y - meanY)),
    0,
  ) / denominator
  const intercept = meanY - (slope * meanX)
  const residualSquares = samples.reduce((sum, sample) => {
    const residual = sample.y - (intercept + (slope * sample.x))
    return sum + (residual ** 2)
  }, 0)
  const totalSquares = samples.reduce((sum, sample) => sum + ((sample.y - meanY) ** 2), 0)
  return {
    slope,
    intercept,
    rSquared: totalSquares > 0 ? Math.max(0, 1 - (residualSquares / totalSquares)) : 1,
    sampleCount: samples.length,
  }
}

export function latestEloGainPerThousandUpdates(points) {
  if (points.length < 2) return null
  const previous = points[points.length - 2]
  const latest = points[points.length - 1]
  const updateDelta = latest.update - previous.update
  if (!Number.isFinite(updateDelta) || updateDelta <= 0) return null
  return ((latest.elo_rating - previous.elo_rating) / updateDelta) * 1000
}
