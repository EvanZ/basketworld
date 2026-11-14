import test from 'node:test'
import assert from 'node:assert/strict'
import { opponentChartData } from '../src/pfsp.js'
import { opponentUsagePlot } from '../src/opponentUsagePlot.js'

function cache(opponents = [1, 50, 100]) {
  return [
    { name: 'jax/train/opponent_cache/usage_update', latest: 200 },
    { name: 'jax/train/opponent_cache/assigned_games_latest', latest: 10 },
    ...opponents.flatMap(update => Object.entries({ snapshot_update: 200, checkpoint_update: update,
      usage_percent: update === 1 ? 100 : 0, assigned_games_latest: update === 1 ? 10 : 0 })
      .map(([field, latest]) => ({ name: `jax/train/opponent_cache/${field}/self_u${update}`, latest }))),
  ]
}
function evidence(update, mode, counters, snapshot = 200) {
  return Object.entries({ w: 0.5, snapshot_update: snapshot, checkpoint_update: update, completed_games: 0, ...counters })
    .map(([field, latest]) => ({ name: `jax/train/pfsp/${field}/self_u${update}/${mode}`, latest }))
}

test('chart combines action modes by possession counts, independently of usage and game completion', () => {
  const result = opponentChartData([
    ...cache(), { name: 'jax/train/opponent_deterministic_episode_prob', latest: 0 },
    ...evidence(50, 'sampled', { ppp_points_scored: 2, ppp_offensive_possessions: 2, ppp_points_allowed: 6, ppp_defensive_possessions: 3, ppp_tracking_start_update: 100 }),
    ...evidence(50, 'deterministic', { ppp_points_scored: 24, ppp_offensive_possessions: 8, ppp_points_allowed: 1, ppp_defensive_possessions: 1, ppp_tracking_start_update: 150 }),
  ])
  const row = result.rows.find(row => row.update === 50)
  assert.equal(row.usage, 0, 'zero recent usage must not hide cumulative PPP')
  assert.equal(result.rows[0].assignedGames, 10)
  assert.equal(row.assignedGames, 0)
  assert.equal(row.pppScored, 2.6, 'sum points / sum possessions, not mean of mode ratios')
  assert.equal(row.pppAllowed, 1.75)
  assert.equal(row.offensivePossessions, 10)
  assert.equal(row.pppStartUpdate, 100)
  assert.equal(result.rows[0].pppScored, null)
})

test('pooled chart survives a non-atomic catalog refresh between usage and PFSP metrics', () => {
  const result = opponentChartData([
    ...cache([1]),
    ...evidence(1, 'sampled', { ppp_points_scored: 12, ppp_offensive_possessions: 10,
      ppp_points_allowed: 8, ppp_defensive_possessions: 10 }, 199),
  ])
  assert.equal(result.update, 200)
  assert.equal(result.rows[0].pppScored, 1.2)
  assert.equal(result.rows[0].pppAllowed, 0.8)
})

test('chart leaves missing, zero-denominator, invalid and stale PPP blank but preserves measured zero', () => {
  const result = opponentChartData([
    ...cache([1, 50, 100, 150, 175]),
    ...evidence(1, 'sampled', { ppp_points_scored: 0, ppp_offensive_possessions: 10, ppp_points_allowed: 0, ppp_defensive_possessions: 0 }),
    ...evidence(50, 'sampled', { ppp_points_scored: 20, ppp_offensive_possessions: 10 }, 199),
    ...evidence(100, 'sampled', { ppp_scored: 99, ppp_offensive_possessions: 10 }),
    ...evidence(150, 'sampled', { ppp_points_scored: NaN, ppp_offensive_possessions: 10, ppp_points_allowed: 1, ppp_defensive_possessions: Infinity }),
    ...evidence(175, 'sampled', { ppp_points_scored: 2, ppp_offensive_possessions: -1 }),
  ])
  assert.deepEqual(result.rows.map(row => row.pppScored), [0, null, null, null, null])
  assert.ok(result.rows.every(row => row.pppAllowed === null))
  assert.deepEqual(opponentChartData([]).rows, [])
})

test('legacy chart keeps lifetime usage with PPP unavailable when counters are absent', () => {
  const result = opponentChartData(evidence(1, 'sampled', { assigned_games: 10 }))
  assert.match(result.basis, /lifetime/)
  assert.equal(result.rows[0].usage, 100)
  assert.equal(result.rows[0].pppScored, null)
})

test('current-update chart uses only matching rollout counters, not cumulative or stale ratios', () => {
  const metrics = [...cache([1, 50, 100, 150]),
    ...evidence(1, 'sampled', { ppp_points_scored: 999, ppp_offensive_possessions: 10 })]
  for (const [update, mode, at, scored, offensive, allowed, defensive] of [
    [1, 'sampled', 200, 0, 2, 4, 2], [1, 'deterministic', 200, 3, 1, 0, 0],
    [50, 'sampled', 199, 500, 10, 500, 10], [100, 'sampled', 200, 0, 0, 0, 0],
    [150, 'sampled', 200, 1, 0, 0, 1], [150, 'deterministic', 200, 3, 1, 0, 0],
  ]) {
    for (const [field, latest] of Object.entries({ update: at, points_scored: scored, offensive_possessions: offensive,
      points_allowed: allowed, defensive_possessions: defensive, scored: 99, allowed: 99 })) {
      metrics.push({ name: `jax/train/pfsp/current_ppp_${field}/self_u${update}/${mode}`, latest })
    }
  }
  const current = opponentChartData(metrics, { basis: 'current' })
  assert.equal(current.currentAvailable, true)
  assert.deepEqual(current.rows.map(row => [row.pppScored, row.pppAllowed]), [[1, 2], [null, null], [null, null], [4, 0]])
  assert.equal(opponentChartData(metrics).rows[0].pppScored, 99.9)
  const old = opponentChartData(metrics.filter(m => !m.name.includes('/current_ppp_')), { basis: 'current' })
  assert.equal(old.currentAvailable, false)
  assert.ok(old.rows.every(row => row.pppScored === null && row.pppAllowed === null))
})

test('combining modes includes score awarded before a possession ends in either PPP view', () => {
  const metrics = [...cache(),
    ...evidence(1, 'sampled', { ppp_points_scored: 1, ppp_offensive_possessions: 0 }),
    ...evidence(1, 'deterministic', { ppp_points_scored: 3, ppp_offensive_possessions: 1 })]
  assert.equal(opponentChartData(metrics).rows[0].pppScored, 4)
})

test('PPP shares a right-axis scale distinct from usage and leaves gaps in missing data', () => {
  const rows = [0, 1, 2, 3].map(i => ({ opponent: `u${i}`, update: i * 50, usage: 25,
    pppScored: [1, null, 0, 3][i], pppAllowed: [2, 2, null, 2][i] }))
  const plot = opponentUsagePlot(rows, 640)
  assert.equal(plot.y(0), plot.bottom)
  assert.equal(plot.pppY(0), plot.bottom)
  assert.ok(plot.pppY(3) >= plot.top)
  assert.ok(plot.yTicks.at(-1) >= 25)
  assert.ok(plot.pppTicks.at(-1) >= 3 && plot.pppTicks.at(-1) < 25)
  assert.equal(plot.series[0].points.length, 3)
  assert.equal((plot.series[0].path.match(/M/g) ?? []).length, 2)
  assert.equal((plot.series[0].path.match(/L/g) ?? []).length, 1)
  assert.equal(plot.series[0].points[1].y, plot.bottom, 'zero PPP lies on the axis')
})

test('empty, legacy, singleton, duplicate-update and dense charts have finite bounded geometry', () => {
  for (const count of [0, 1, 2, 600]) {
    const rows = Array.from({ length: count }, (_, i) => ({ opponent: `opponent${i}`, update: count === 2 ? 1 : i * 50,
      usage: 100 / count, pppScored: i % 2 ? 0 : 1.5, pppAllowed: null }))
    for (const width of [280, 400, 900]) {
      const plot = opponentUsagePlot(rows, width)
      assert.ok(plot.bars.every(bar => Number.isFinite(bar.x) && Number.isFinite(bar.y) && bar.width > 0 && bar.x >= plot.left && bar.x <= plot.right))
      assert.ok(plot.series.every(series => !/NaN|Infinity/.test(series.path)))
      assert.equal(plot.series[1].points.length, 0)
      if (count === 2) assert.notEqual(plot.bars[0].x, plot.bars[1].x)
      if (!count) assert.deepEqual(plot.pppTicks, [])
    }
  }
  const legacy = opponentUsagePlot([{ opponent: 'legacy', update: 1, usage: 100 }], 640)
  assert.deepEqual(legacy.pppTicks, [])
  assert.equal(legacy.series[0].path, '')
})
