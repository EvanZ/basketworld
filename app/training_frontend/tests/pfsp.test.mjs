import test from 'node:test'
import assert from 'node:assert/strict'
import { pfspRows, metricDefinition } from '../src/pfsp.js'

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
