import test from 'node:test'
import assert from 'node:assert/strict'

import { responseErrorMessage } from '../src/api.js'
import { historicalEvalUpdatesForTarget } from '../src/configEditing.js'

test('retargets a copied run historical evaluation schedule when updates decrease', () => {
  assert.deepEqual(
    historicalEvalUpdatesForTarget(
      [100, 500, 2500, 5000, 7500, 10000, 30000],
      30000,
      5000,
    ),
    [100, 500, 2500, 5000],
  )
})

test('preserves copied milestones while a replacement target is typed', () => {
  const source = [100, 500, 2500, 5000, 7500, 10000, 30000]

  assert.deepEqual(historicalEvalUpdatesForTarget(source, 30000, 6), [6])
  assert.deepEqual(historicalEvalUpdatesForTarget(source, 30000, 60), [60])
  assert.deepEqual(historicalEvalUpdatesForTarget(source, 30000, 600), [100, 500, 600])
  assert.deepEqual(
    historicalEvalUpdatesForTarget(source, 30000, 6000),
    [100, 500, 2500, 5000, 6000],
  )
})

test('moves an existing final evaluation milestone when updates increase', () => {
  assert.deepEqual(
    historicalEvalUpdatesForTarget([100, 500, 5000], 5000, 8000),
    [100, 500, 8000],
  )
})

test('does not add a final milestone when the explicit schedule omitted one', () => {
  assert.deepEqual(
    historicalEvalUpdatesForTarget([100, 500], 5000, 300),
    [100],
  )
  assert.equal(historicalEvalUpdatesForTarget(null, 5000, 300), null)
})

test('shows FastAPI validation messages instead of a generic 422 error', () => {
  assert.equal(responseErrorMessage({
    detail: [{
      type: 'value_error',
      loc: ['body'],
      msg: 'Value error, Historical evaluation updates cannot exceed num_updates.',
    }],
  }, 422), 'Historical evaluation updates cannot exceed num_updates.')
  assert.equal(responseErrorMessage({}, 422), 'Request failed (422)')
})
