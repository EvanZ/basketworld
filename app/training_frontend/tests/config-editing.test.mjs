import test from 'node:test'
import assert from 'node:assert/strict'

import { responseErrorMessage } from '../src/api.js'
import {
  continuationHistoricalEvalSchedule,
  historicalEvalUpdatesForTarget,
  parseHistoricalEvalUpdates,
} from '../src/configEditing.js'

test('keeps future evaluation milestones when a copied run is shortened', () => {
  assert.deepEqual(
    historicalEvalUpdatesForTarget(
      [100, 500, 2500, 5000, 7500, 10000, 30000],
      30000,
      5000,
    ),
    [100, 500, 2500, 5000, 7500, 10000, 30000],
  )
})

test('preserves copied milestones while a replacement target is typed', () => {
  const source = [100, 500, 2500, 5000, 7500, 10000, 30000]

  assert.deepEqual(historicalEvalUpdatesForTarget(source, 30000, 6), [6, ...source])
  assert.deepEqual(historicalEvalUpdatesForTarget(source, 30000, 60), [60, ...source])
  assert.deepEqual(
    historicalEvalUpdatesForTarget(source, 30000, 600),
    [100, 500, 600, 2500, 5000, 7500, 10000, 30000],
  )
  assert.deepEqual(
    historicalEvalUpdatesForTarget(source, 30000, 6000),
    [100, 500, 2500, 5000, 6000, 7500, 10000, 30000],
  )
})

test('preserves an existing final evaluation milestone when updates increase', () => {
  assert.deepEqual(
    historicalEvalUpdatesForTarget([100, 500, 5000], 5000, 8000),
    [100, 500, 5000, 8000],
  )
})

test('successive copied run extensions keep all earlier evaluation milestones', () => {
  const original = [100, 250, 500]
  const extended = historicalEvalUpdatesForTarget(original, 500, 2500)
  const copiedAgain = historicalEvalUpdatesForTarget(extended, 2500, 5000)

  assert.deepEqual(extended, [100, 250, 500, 2500])
  assert.deepEqual(copiedAgain, [100, 250, 500, 2500, 5000])
  assert.deepEqual(original, [100, 250, 500])
})

test('copying a saved shortened run can restore its original full schedule', () => {
  const original = [100, 250, 500, 1000, 2500, 5000, 10000, 30000]
  const shortened = historicalEvalUpdatesForTarget(original, 30000, 500)
  const savedConfig = JSON.parse(JSON.stringify({
    num_updates: 500,
    historical_eval_updates: shortened,
  }))
  const copiedAgain = historicalEvalUpdatesForTarget(
    savedConfig.historical_eval_updates,
    savedConfig.num_updates,
    30000,
  )

  assert.deepEqual(savedConfig.historical_eval_updates, original)
  assert.deepEqual(copiedAgain, original)
})

test('typing a longer target restores every milestone from the copied schedule', () => {
  const source = [100, 250, 500, 1000, 2500, 5000]
  for (const target of [1, 10, 100, 1000]) {
    historicalEvalUpdatesForTarget(source, 5000, target)
  }
  assert.deepEqual(
    historicalEvalUpdatesForTarget(source, 5000, 10000),
    [...source, 10000],
  )
})

test('keeps unchanged and disabled schedules intact', () => {
  assert.deepEqual(
    historicalEvalUpdatesForTarget([100, 250, 500], 500, 500),
    [100, 250, 500],
  )
  assert.deepEqual(historicalEvalUpdatesForTarget([], 500, 2500), [])
})

test('does not add a final milestone when the explicit schedule omitted one', () => {
  assert.deepEqual(
    historicalEvalUpdatesForTarget([100, 500], 5000, 300),
    [100, 500],
  )
  assert.equal(historicalEvalUpdatesForTarget(null, 5000, 300), null)
})

test('parses append-only checkpoints for a completed-run continuation', () => {
  assert.deepEqual(parseHistoricalEvalUpdates(' 7500, 10000 ').updates, [7500, 10000])
  assert.match(parseHistoricalEvalUpdates('7500, nope').error, /whole-number/)
  assert.match(parseHistoricalEvalUpdates('7500, 7500').error, /only once/)
})

test('merges new continuation checkpoints without replacing the saved schedule', () => {
  assert.deepEqual(
    continuationHistoricalEvalSchedule(
      [100, 250, 500, 5000, 30000],
      [7500, 10000],
      5000,
      10000,
    ),
    {
      updates: [100, 250, 500, 5000, 7500, 10000, 30000],
      error: '',
    },
  )
  assert.match(
    continuationHistoricalEvalSchedule([5000], [5000], 5000, 10000).error,
    /after U5,000/,
  )
  assert.match(
    continuationHistoricalEvalSchedule([5000], [10001], 5000, 10000).error,
    /no later than U10,000/,
  )
})

test('shows FastAPI validation messages instead of a generic 422 error', () => {
  assert.equal(responseErrorMessage({
    detail: [{
      type: 'value_error',
      loc: ['body'],
      msg: 'Value error, Historical evaluation updates must be positive.',
    }],
  }, 422), 'Historical evaluation updates must be positive.')
  assert.equal(responseErrorMessage({}, 422), 'Request failed (422)')
})
