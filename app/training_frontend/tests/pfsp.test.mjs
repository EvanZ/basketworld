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
