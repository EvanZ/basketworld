import test from 'node:test'
import assert from 'node:assert/strict'
import { pfspRows, metricDefinition, opponentUsage } from '../src/pfsp.js'

test('PFSP table separates identity, mode, counts and neutral priors', () => {
  const metrics = [
    { name: 'jax/train/pfsp/w/self_u100/sampled', latest: 0.6 },
    { name: 'jax/train/pfsp/completed_games/self_u100/sampled', latest: 20 },
    { name: 'jax/train/pfsp/w/self_u100/deterministic', latest: 0.5 },
    { name: 'jax/train/pfsp/prior_only/self_u100/deterministic', latest: 1 },
  ]
  const rows = pfspRows(metrics)
  assert.equal(rows.length, 2)
  assert.equal(rows.find(row => row.mode === 'sampled').games, 20)
  assert.equal(rows.find(row => row.mode === 'deterministic').priorOnly, true)
  assert.deepEqual(pfspRows([]), [])
})

test('definitions distinguish shot value, PFSP and empirical evaluation', () => {
  assert.match(metricDefinition('jax/train/pressure/learner_defending/mean_ball_handler_esv_reduction'), /not PPP/)
  assert.match(metricDefinition('jax/train/pfsp/w/x/sampled'), /500-update half-life/)
  assert.match(metricDefinition('jax/eval_historical/w_eval/u100/deterministic'), /Excludes truncations/)
})

test('Win % uses raw wins and its tracked denominator, never the smoothed score', () => {
  const metrics = []
  for (const [update, wins, tracked] of [[1, 1, 4], [50, null, null], [100, 0, 0], [150, 0, 4]]) {
    for (const [field, latest] of [['w', 0.8], ['completed_games', 100], ['checkpoint_update', update], ['wins', wins], ['win_rate_games', tracked]]) {
      if (latest !== null) metrics.push({ name: `jax/train/pfsp/${field}/self_u${update}/sampled`, latest })
    }
  }
  const rows = pfspRows(metrics)
  assert.deepEqual(rows.map(row => row.winRate), [0.25, null, null, 0])
  assert.equal(rows[0].winRateGames, 4)
  assert.equal(rows[0].wins, 1)
  assert.equal(rows[0].games, 100, 'legacy total must not dilute the known win percentage')
  assert.match(metricDefinition('jax/train/pfsp/win_rate/self_u1/sampled'), /ties are non-wins and truncations are excluded/)
})

test('PFSP table hides zero-game deterministic rows without changing usage counts', () => {
  const metrics = []
  for (const [update, mode, games, assigned] of [[100, 'deterministic', 0, 5], [100, 'sampled', 0, 15], [250, 'deterministic', 1, 20]]) {
    for (const [field, latest] of [['w', 0.5], ['completed_games', games], ['assigned_games', assigned], ['checkpoint_update', update]]) {
      metrics.push({ name: `jax/train/pfsp/${field}/self_u${update}/${mode}`, latest })
    }
  }
  for (const modeMetrics of [[], [{ name: 'jax/train/opponent_deterministic_episode_prob', latest: 0.5 }]]) {
    const input = [...metrics, ...modeMetrics]
    assert.deepEqual(pfspRows(input).map(row => [row.update, row.mode]), [[100, 'sampled'], [250, 'deterministic']])
    assert.equal(pfspRows(input, { includeInactiveModes: true }).length, 3)
    assert.equal(opponentUsage(input).games, 40)
    assert.deepEqual(opponentUsage(input).rows.map(row => row.usage), [50, 50])
  }
})

test('per-opponent PPP preserves zero efficiency, missing data, and tracked denominators', () => {
  const metrics = []
  for (const [update, scored, allowed, offensive, defensive] of [[1, 0, 1.5, 10, 20], [50, null, null, null, null], [100, 9, 9, 0, 0]]) {
    for (const [field, latest] of [['w', 0.7], ['checkpoint_update', update], ['ppp_scored', scored], ['ppp_allowed', allowed], ['ppp_offensive_possessions', offensive], ['ppp_defensive_possessions', defensive], ['ppp_tracking_start_update', 250], ['ppp_points_scored', 0], ['ppp_points_allowed', 30]]) {
      if (latest !== null) metrics.push({ name: `jax/train/pfsp/${field}/self_u${update}/sampled`, latest })
    }
  }
  const rows = pfspRows(metrics)
  assert.deepEqual(rows.map(row => [row.pppScored, row.pppAllowed]), [[0, 1.5], [null, null], [null, null]])
  assert.equal(rows[0].offensivePossessions, 10)
  assert.equal(rows[0].defensivePossessions, 20)
  assert.equal(rows[0].pointsAllowed, 30)
  assert.equal(rows[0].pppStartUpdate, 250)
  assert.match(metricDefinition('jax/train/pfsp/ppp_allowed/self_u1/sampled'), /opponent completed offensive possessions/)
  assert.match(metricDefinition('jax/train/pfsp/ppp_scored/self_u1/sampled'), /not used for PFSP sampling/)
})

test('current pool hides evicted identities and inactive action modes', () => {
  const metrics = [{ name: 'jax/train/opponent_deterministic_episode_prob', latest: 0 }]
  for (const [id, update] of [['self_u1', 100], ['self_u50', 150]]) {
    for (const mode of ['sampled', 'deterministic']) {
      metrics.push({ name: `jax/train/pfsp/w/${id}/${mode}`, latest: 0.5 },
        { name: `jax/train/pfsp/snapshot_update/${id}/${mode}`, latest: update })
    }
  }
  const rows = pfspRows(metrics)
  assert.equal(rows.length, 1)
  assert.equal(rows[0].opponent, 'self_u50')
  assert.equal(rows[0].mode, 'sampled')
})

test('usage chart uses measured latest-rollout percentages, not sampling probabilities', () => {
  const metrics = [
    { name: 'jax/train/opponent_cache/usage_update', latest: 101 },
    { name: 'jax/train/opponent_cache/assigned_games_latest', latest: 20 },
    { name: 'jax/train/opponent_cache/capacity', latest: 600 },
    { name: 'jax/train/pfsp/probability/self_u50/sampled', latest: 0.9 },
  ]
  for (const [id, checkpoint, observed, share] of [['self_u1', 1, 100, 100], ['self_u50', 50, 101, 25], ['self_u100', 100, 101, 75]]) {
    for (const [field, value] of [['usage_percent', share], ['checkpoint_update', checkpoint], ['snapshot_update', observed]]) {
      metrics.push({ name: `jax/train/opponent_cache/${field}/${id}`, latest: value })
    }
  }
  const result = opponentUsage(metrics)
  assert.deepEqual(result.rows.map(row => [row.update, row.usage]), [[50, 25], [100, 75]])
  assert.equal(result.update, 101)
  assert.equal(result.games, 20)
  assert.equal(result.capacity, 600)
  assert.equal(result.basis, 'latest rollout')
})

test('legacy usage is explicitly lifetime, missing and zero-count states stay honest', () => {
  const metrics = []
  for (const [id, update, count] of [['self_u100', 100, 30], ['self_u250', 250, 70]]) {
    for (const [field, value] of [['w', 0.5], ['assigned_games', count], ['checkpoint_update', update]]) {
      metrics.push({ name: `jax/train/pfsp/${field}/${id}/sampled`, latest: value })
    }
  }
  assert.deepEqual(opponentUsage(metrics).rows.map(row => row.usage), [30, 70])
  assert.match(opponentUsage(metrics).basis, /lifetime/)
  assert.deepEqual(opponentUsage([]).rows, [])
  assert.equal(opponentUsage([{ name: 'jax/train/opponent_cache/usage_update', latest: 1 }]).games, 0)
})
