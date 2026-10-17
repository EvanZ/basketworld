export const POSSESSION_MODE = 'possessions'
export const GAME_CLOCK_MODE = 'game_clock'

export const CLOCK_CONFIG_FIELDS = new Set([
  'game_clock_seconds',
  'overtime_seconds',
  'game_clock_max_total_steps',
])

export const POSSESSION_CONFIG_FIELDS = new Set([
  'multi_possession_limit',
  'multi_possession_limit_start',
  'multi_possession_limit_end',
  'multi_possession_limit_ramp_updates',
  'multi_possession_overtime_round_cap',
])

export function episodeEndMode(config) {
  const value = config?.overrides?.episode_end_mode ?? config?.episode_end_mode
  return value === GAME_CLOCK_MODE ? GAME_CLOCK_MODE : POSSESSION_MODE
}

export function modeConfigFieldIsVisible(fieldName, mode) {
  if (fieldName === 'episode_end_mode') return false
  if (CLOCK_CONFIG_FIELDS.has(fieldName)) return mode === GAME_CLOCK_MODE
  if (POSSESSION_CONFIG_FIELDS.has(fieldName)) return mode === POSSESSION_MODE
  return true
}

export function modeConfigValue(config, fieldName, fallback) {
  if (Object.prototype.hasOwnProperty.call(config?.overrides || {}, fieldName)) {
    return config.overrides[fieldName]
  }
  return fallback
}

export function withModeConfigValue(config, fieldName, value) {
  return {
    ...config,
    overrides: {
      ...(config?.overrides || {}),
      [fieldName]: value,
    },
  }
}
