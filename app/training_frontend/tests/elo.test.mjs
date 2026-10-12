import test from 'node:test'
import assert from 'node:assert/strict'

import {
  eloSparklineCoordinates,
  eloSparklinePoints,
  eloWarningMessages,
  fitLogElo,
  latestEloGainPerThousandUpdates,
  prepareEloSeries,
} from '../src/elo.js'

test('prepares a sorted Elo series and drops zero-game placeholder rows', () => {
  const series = prepareEloSeries({
    points: [
      { update: 500, elo_rating: 740 },
      { update: 100, elo_rating: 400 },
      { update: 2500, elo_rating: null },
    ],
  })

  assert.deepEqual(series.map((point) => point.update), [100, 500])
  assert.deepEqual(series.map((point) => point.elo_rating), [400, 740])
})

test('creates bounded sparkline coordinates including both endpoints', () => {
  const points = [
    { update: 100, elo_rating: 400 },
    { update: 500, elo_rating: 700 },
    { update: 2500, elo_rating: 600 },
  ]
  const plotted = eloSparklineCoordinates(points)

  assert.equal(plotted[0].x, 18)
  assert.equal(plotted.at(-1).x, 402)
  assert.ok(plotted[1].x < 100, 'U500 should be much closer to U100 than U2500')
  assert.ok(plotted.every((point) => point.y >= 18 && point.y <= 94))
  assert.match(eloSparklinePoints(points), /^18\.00,94\.00 /)
})

test('handles an empty Elo state', () => {
  assert.deepEqual(prepareEloSeries({ status: 'waiting', points: [] }), [])
  assert.equal(eloSparklinePoints([]), '')
  assert.deepEqual(eloWarningMessages({ warnings: [] }), [])
})

test('retains concise backend warning states for display', () => {
  assert.deepEqual(
    eloWarningMessages({ warnings: ['29 truncated episodes were excluded.', '', null] }),
    ['29 truncated episodes were excluded.'],
  )
})

test('fits logarithmic Elo improvement and reports the latest marginal rate', () => {
  const points = [
    { update: 100, elo_rating: 400 + (120 * Math.log(100)) },
    { update: 500, elo_rating: 400 + (120 * Math.log(500)) },
    { update: 2500, elo_rating: 400 + (120 * Math.log(2500)) },
  ]
  const fit = fitLogElo(points)

  assert.ok(Math.abs(fit.slope - 120) < 1e-9)
  assert.equal(fit.rSquared, 1)
  assert.ok(Math.abs(latestEloGainPerThousandUpdates(points) - 96.566274746046) < 1e-9)
})
