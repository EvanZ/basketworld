// Render recorded outcomes, not inferred actions. Rebuilding from the visible
// history keeps live play, fast self-play, and replay navigation in sync.
const number = (value) => value !== null && value !== undefined && value !== ''
  && Number.isFinite(Number(value)) ? Number(value) : null;
const playerId = (value) => Number.isInteger(number(value)) && number(value) >= 0 ? number(value) : null;

function formatClock(seconds) {
  if (number(seconds) === null) return '—';
  const value = Math.max(0, Math.floor(Number(seconds)));
  return `${Math.floor(value / 60)}:${String(value % 60).padStart(2, '0')}`;
}

function eventClock(state, previous) {
  if (state.episode_end_mode !== 'game_clock') {
    // The next state may already have reset the possession's shot clock.
    return { period: 'Possession', clock: `Shot clock ${number(previous.shot_clock) ?? number(state.shot_clock) ?? '—'}` };
  }
  const previousOvertime = number(previous.overtime_round) ?? 0;
  const overtime = number(state.overtime_round) ?? 0;
  const periodEnded = overtime > previousOvertime;
  return {
    period: (periodEnded ? previousOvertime : overtime) > 0
      ? `OT ${periodEnded ? previousOvertime : overtime}` : 'Regulation',
    clock: periodEnded ? '0:00' : formatClock(state.game_clock_remaining),
  };
}

const reasons = {
  steal: 'steal', intercepted: 'intercepted pass', defender_pressure: 'defender pressure',
  move_out_of_bounds: 'out of bounds', pass_out_of_bounds: 'pass out of bounds',
  shot_clock_violation: 'shot-clock violation', offensive_three_seconds: 'offensive three-second violation',
  clearance_violation: 'clearance violation', inbound_timeout: 'inbound timeout violation',
  inbound_invalid_pass: 'invalid inbound pass',
};

export function playByPlayForStep(state, previous = {}, names = {}) {
  if (!state) return [];
  const results = state.last_action_results || {};
  const rows = [];
  const player = (raw) => {
    const id = playerId(raw);
    return id === null ? 'Unknown player' : `#${id}${names[id] ? ` ${names[id]}` : ''}`;
  };
  // Stable roster IDs must win over offense/defense roles after a turnover.
  const team = (raw) => {
    const id = playerId(raw);
    if (id === null) return '—';
    if ((state.team_a_ids || previous.team_a_ids || []).map(Number).includes(id)) return 'Team A';
    if ((state.team_b_ids || previous.team_b_ids || []).map(Number).includes(id)) return 'Team B';
    if ((previous.offense_ids || state.offense_ids || []).map(Number).includes(id)) return 'Offense';
    if ((previous.defense_ids || state.defense_ids || []).map(Number).includes(id)) return 'Defense';
    return '—';
  };
  const timing = eventClock(state, previous);
  const a = number(state.team_a_score);
  const b = number(state.team_b_score);
  const push = (kind, side, event) => rows.push({
    ...timing, kind, team: side, event, score: a !== null && b !== null ? `${a}–${b}` : '—',
  });

  for (const [id, shot] of Object.entries(results.shots || {})) {
    if (!shot || typeof shot !== 'object') continue;
    const distance = number(shot.distance);
    const type = shot.is_dunk ? 'Dunk' : shot.is_three ? '3-point shot'
      : distance !== null && distance <= 1 ? 'Layup' : '2-point shot';
    const outcome = shot.blocked ? `blocked by ${player(shot.blocker_id)}` : shot.success === true ? 'made' : shot.success === false ? 'missed' : 'outcome unavailable';
    const range = distance === null ? 'distance unavailable' : `${distance} ${distance === 1 ? 'hex' : 'hexes'}`;
    const assist = shot.success === true && shot.assist_full && playerId(shot.assist_passer_id) !== null
      ? ` Assist: ${player(shot.assist_passer_id)}.` : '';
    push(shot.blocked ? 'block' : 'shot', team(id), `${type} by ${player(id)} from ${range}: ${outcome}.${assist}`);
  }

  const turnovers = Array.isArray(results.turnovers) ? results.turnovers : [];
  for (const turnover of turnovers) {
    const reason = String(turnover.reason || 'turnover');
    const label = reasons[reason] || reason.replaceAll('_', ' ');
    const stealer = playerId(turnover.stolen_by);
    const violation = /violation|three_seconds|inbound_timeout/.test(reason);
    const steal = stealer !== null ? ` Steal by ${player(stealer)}.` : '';
    push(stealer !== null ? 'steal' : violation ? 'violation' : 'turnover', team(turnover.player_id),
      `Turnover by ${player(turnover.player_id)} (${label}).${steal}`);
  }
  for (const [key, label] of [
    ['offensive_lane_violations', 'Offensive three-second violation'],
    ['defensive_lane_violations', 'Defensive lane violation'],
  ]) {
    for (const violation of results[key] || []) {
      if (key === 'offensive_lane_violations' && turnovers.some((t) =>
        t.reason === 'offensive_three_seconds' && playerId(t.player_id) === playerId(violation.player_id))) continue;
      push('violation', team(violation.player_id), `${label} by ${player(violation.player_id)}.`);
    }
  }
  for (const check of results.checks || []) {
    const offense = check.team === 'team_a' ? 'Team A' : check.team === 'team_b' ? 'Team B' : '—';
    if (check.defense_space_violation) {
      push('violation', offense === 'Team A' ? 'Team B' : offense === 'Team B' ? 'Team A' : '—',
        'Defensive check-space violation.');
    } else if (check.violation) {
      push('violation', offense, 'Check pickup deadline violation; turnover.');
    }
  }
  return rows;
}

export function buildPlayByPlay(states = [], names = {}, limit = 240) {
  const rows = [];
  // State zero is the starting position, not another execution of its last action.
  for (let i = states.length - 1; i >= 1 && rows.length < limit; i -= 1) {
    const events = playByPlayForStep(states[i], states[i - 1], names);
    events.forEach((event, index) => rows.push({ ...event, id: `${i}-${index}` }));
  }
  return rows.slice(0, limit);
}
