import test from 'node:test'
import assert from 'node:assert/strict'

import {
  episodeEndMode,
  modeConfigFieldIsVisible,
  modeConfigValue,
  withModeConfigValue,
} from '../src/gameModeConfig.js'

test('legacy trainer configs default to possession mode', () => {
  assert.equal(episodeEndMode({ overrides: {} }), 'possessions')
})

test('mode-specific fields are only visible for their active mode', () => {
  assert.equal(modeConfigFieldIsVisible('game_clock_seconds', 'game_clock'), true)
  assert.equal(modeConfigFieldIsVisible('game_clock_seconds', 'possessions'), false)
  assert.equal(modeConfigFieldIsVisible('multi_possession_limit_end', 'possessions'), true)
  assert.equal(modeConfigFieldIsVisible('multi_possession_limit_end', 'game_clock'), false)
  assert.equal(modeConfigFieldIsVisible('gamma', 'game_clock'), true)
})

test('switching modes preserves both mode value sets', () => {
  let config = { overrides: { game_clock_seconds: 300, overtime_seconds: 120 } }
  config = withModeConfigValue(config, 'episode_end_mode', 'game_clock')
  config = withModeConfigValue(config, 'episode_end_mode', 'possessions')
  assert.equal(episodeEndMode(config), 'possessions')
  assert.equal(modeConfigValue(config, 'game_clock_seconds', 720), 300)
  assert.equal(modeConfigValue(config, 'overtime_seconds', 60), 120)
})
