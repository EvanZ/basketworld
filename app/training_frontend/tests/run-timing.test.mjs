import test from 'node:test'
import assert from 'node:assert/strict'

import {
  estimateRemainingTrainingSeconds,
  remainingTrainingTimeCaption,
} from '../src/runTiming.js'

test('estimates remaining active time from completed updates', () => {
  assert.equal(estimateRemainingTrainingSeconds({
    current_update: 250,
    target_updates: 1000,
    training_elapsed_seconds: 500,
  }), 1500)
})

test('waits for progress and reaches zero at the target', () => {
  assert.equal(estimateRemainingTrainingSeconds({
    current_update: 0,
    target_updates: 1000,
    training_elapsed_seconds: 45,
  }), null)
  assert.equal(estimateRemainingTrainingSeconds({
    current_update: 1000,
    target_updates: 1000,
    training_elapsed_seconds: 2000,
  }), 0)
})

test('describes paused and terminal estimates', () => {
  assert.equal(remainingTrainingTimeCaption({ status: 'paused' }, 900), 'active time · paused')
  assert.equal(remainingTrainingTimeCaption({ status: 'stopped' }, 900), 'stopped before completion')
  assert.equal(remainingTrainingTimeCaption({ status: 'completed' }, 0), 'complete')
})
