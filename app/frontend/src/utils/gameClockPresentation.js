export const POSSESSION_MODE = 'possessions';
export const GAME_CLOCK_MODE = 'game_clock';

export function recordedEpisodeEndMode(state) {
  return state?.episode_end_mode === GAME_CLOCK_MODE
    ? GAME_CLOCK_MODE
    : POSSESSION_MODE;
}

export function formatGameClock(totalSeconds) {
  const seconds = Math.max(0, Math.floor(Number(totalSeconds) || 0));
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}`;
}

export function clockPresentation(state) {
  const mode = recordedEpisodeEndMode(state);
  const overtimeRound = Math.max(0, Number(state?.overtime_round ?? 0));
  const phase = String(state?.game_phase || 'live');
  return {
    mode,
    gameClockMode: mode === GAME_CLOCK_MODE,
    gameClock: formatGameClock(state?.game_clock_remaining),
    periodLabel: overtimeRound > 0 ? `OT ${overtimeRound}` : 'REGULATION',
    clockRunning: mode === GAME_CLOCK_MODE
      && state?.done !== true
      && ['live', 'awaiting_check'].includes(phase),
  };
}
