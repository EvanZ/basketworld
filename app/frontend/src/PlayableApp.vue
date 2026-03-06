<script setup>
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue';
import GameBoard from './components/GameBoard.vue';
import PlayableControls from './components/PlayableControls.vue';
import PlayableStatsPanel from './components/PlayableStatsPanel.vue';
import PlayableEnvironmentInfo from './components/PlayableEnvironmentInfo.vue';
import basketworldLogo from './assets/basketworld-logo.jpg';
import {
  getPlayableOptions,
  newPlayableGame,
  startPlayableGame,
  stepPlayableGame,
} from './services/api';

const optionsLoading = ref(false);
const actionLoading = ref(false);
const error = ref('');
const isDevBuild = Boolean(import.meta.env?.DEV);

const optionsPayload = ref(null);
const selectedPlayersPerSide = ref(1);
const selectedDifficulty = ref('easy');
const selectedPeriodMode = ref('period');
const selectedPeriodLengthMinutes = ref(5);

const gameState = ref(null);
const gameHistory = ref([]);
const activePlayerId = ref(null);
const boardSelections = ref({});
const controlsRef = ref(null);
const playerDisplayNames = ref({});
const playerJerseyNumbers = ref({});

const score = ref({ user: 0, ai: 0 });
const possession = ref(null);
const sessionConfig = ref(null);
function createDefaultGameClock() {
  return {
    period_mode: 'period',
    total_periods: 1,
    current_period: 1,
    period_length_minutes: 5,
    seconds_remaining: 300,
    display: '05:00',
    segment_label: 'Period 1',
  };
}

function createDefaultGameResult() {
  return {
    game_over: false,
    winner: null,
    message: '',
    score: { user: 0, ai: 0 },
  };
}

const gameClock = ref(createDefaultGameClock());
const gameResult = ref(createDefaultGameResult());
const lastPossessionResult = ref(null);
const transitionRunning = ref(false);
const forceGameOverPreview = ref(false);
const violationOverlayPreview = ref(null);
const passChordPPressed = ref(false);
const periodModeLabelMap = {
  period: '1 period',
  halves: '2 halves',
  quarters: '4 quarters',
};
const playableKeyboardShortcuts = [
  {
    key: 'Q',
    label: 'Move NW',
    action: 'MOVE_NW',
  },
  {
    key: 'A',
    label: 'Move W',
    action: 'MOVE_W',
  },
  {
    key: 'Z',
    label: 'Move SW',
    action: 'MOVE_SW',
  },
  {
    key: 'E',
    label: 'Move NE',
    action: 'MOVE_NE',
  },
  {
    key: 'D',
    label: 'Move E',
    action: 'MOVE_E',
  },
  {
    key: 'C',
    label: 'Move SE',
    action: 'MOVE_SE',
  },
  {
    key: 'S',
    label: 'Shoot',
    action: 'SHOOT',
  },
  {
    key: 'N',
    label: 'New Game',
  },
  {
    key: 'T',
    label: 'Submit Turn',
  },
  {
    key: 'P+0-9',
    label: 'Pass Target',
  },
  {
    key: '0-9',
    label: 'Player ID',
  },
];
const MOVE_HOTKEY_ACTIONS = {
  q: 'MOVE_NW',
  a: 'MOVE_W',
  z: 'MOVE_SW',
  e: 'MOVE_NE',
  d: 'MOVE_E',
  c: 'MOVE_SE',
  s: 'SHOOT',
};
const MOVE_HOTKEY_ACTIONS_BY_CODE = {
  KeyQ: 'MOVE_NW',
  KeyA: 'MOVE_W',
  KeyZ: 'MOVE_SW',
  KeyE: 'MOVE_NE',
  KeyD: 'MOVE_E',
  KeyC: 'MOVE_SE',
  KeyS: 'SHOOT',
};

const PLAYABLE_SURNAME_POOL = [
  'Smith', 'Jones', 'Brown', 'Davis', 'Miller', 'Wilson', 'Moore', 'Taylor',
  'Thomas', 'White', 'Clark', 'Lewis', 'Lee', 'Walker', 'Hall', 'Allen',
  'Young', 'King', 'Wright', 'Scott', 'Torres', 'Nguyen', 'Hill', 'Green',
  'Adams', 'Baker', 'Nelson', 'Carter', 'Perez', 'Turner', 'Parker', 'Evans',
  'Morris', 'Rogers', 'Reed', 'Cook', 'Morgan', 'Bell', 'Murphy', 'Bailey',
  'Rivera', 'Cooper', 'Cox', 'Gray', 'James', 'Watson', 'Brooks', 'Kelly',
  'Price', 'Wood', 'Barnes', 'Ross', 'Cole', 'Perry', 'Powell', 'Long',
  'Patel', 'Flores', 'Hughes', 'Butler', 'Foster', 'Bryant', 'Diaz', 'Hayes',
  'Myers', 'Ford', 'Graham', 'Woods', 'West', 'Jordan', 'Owens', 'Fisher',
  'Ellis', 'Harris', 'Hudson', 'Ryan', 'Porter', 'Hunter', 'Hicks', 'Henry',
  'Bishop', 'Dean', 'Mason', 'Hart', 'Willis', 'Lane', 'Riley', 'Rose',
  'Stone', 'Dunn', 'Payne', 'Ray', 'Berry', 'Arnold', 'Wagner', 'Weaver',
  'Burke', 'Lynch', 'Hanson', 'Day', 'Fox', 'Ramsey', 'Shaw', 'Burns',
  'Gordon', 'Warren', 'Dixon', 'Ramos', 'Reyes', 'Cruz', 'Ortiz', 'Soto',
  'Kim', 'Li', 'Ho', 'Tran', 'Pham', 'Le',
];

const PLAYABLE_JERSEY_NUMBER_WEIGHTS = [
  { number: '00', weight: 14 },
  { number: '0', weight: 27 },
  { number: '1', weight: 24 },
  { number: '2', weight: 23 },
  { number: '3', weight: 23 },
  { number: '4', weight: 23 },
  { number: '5', weight: 27 },
  { number: '6', weight: 2 },
  { number: '7', weight: 20 },
  { number: '8', weight: 28 },
  { number: '9', weight: 23 },
  { number: '10', weight: 19 },
  { number: '11', weight: 22 },
  { number: '12', weight: 19 },
  { number: '13', weight: 23 },
  { number: '14', weight: 16 },
  { number: '15', weight: 18 },
  { number: '16', weight: 7 },
  { number: '17', weight: 16 },
  { number: '18', weight: 8 },
  { number: '19', weight: 5 },
  { number: '20', weight: 17 },
  { number: '21', weight: 19 },
  { number: '22', weight: 24 },
  { number: '23', weight: 17 },
  { number: '24', weight: 17 },
  { number: '25', weight: 15 },
  { number: '26', weight: 6 },
  { number: '27', weight: 9 },
  { number: '28', weight: 8 },
  { number: '29', weight: 2 },
  { number: '30', weight: 11 },
  { number: '31', weight: 7 },
  { number: '32', weight: 8 },
  { number: '33', weight: 11 },
  { number: '34', weight: 6 },
  { number: '35', weight: 10 },
  { number: '36', weight: 2 },
  { number: '37', weight: 1 },
  { number: '40', weight: 3 },
  { number: '41', weight: 3 },
  { number: '42', weight: 3 },
  { number: '43', weight: 2 },
  { number: '44', weight: 6 },
  { number: '45', weight: 5 },
  { number: '46', weight: 1 },
  { number: '50', weight: 3 },
  { number: '54', weight: 1 },
  { number: '55', weight: 9 },
  { number: '58', weight: 1 },
  { number: '61', weight: 1 },
  { number: '65', weight: 1 },
  { number: '67', weight: 1 },
  { number: '71', weight: 1 },
  { number: '76', weight: 1 },
  { number: '77', weight: 6 },
  { number: '88', weight: 4 },
  { number: '91', weight: 1 },
  { number: '94', weight: 1 },
  { number: '99', weight: 1 },
];
const PLAYABLE_JERSEY_TOTAL_WEIGHT = PLAYABLE_JERSEY_NUMBER_WEIGHTS.reduce(
  (acc, row) => acc + Math.max(0, Number(row?.weight || 0)),
  0,
);

function formatClockDisplay(rawSeconds) {
  const total = Math.max(0, Number(rawSeconds || 0));
  const minutes = Math.floor(total / 60);
  const seconds = Math.floor(total % 60);
  return `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
}

function segmentLabelForMode(mode, currentPeriod) {
  const idx = Math.max(1, Number(currentPeriod || 1));
  if (mode === 'halves') return `Half ${idx}`;
  if (mode === 'quarters') return `Quarter ${idx}`;
  return `Period ${idx}`;
}

function normalizeGameClock(raw) {
  const mode = ['period', 'halves', 'quarters'].includes(String(raw?.period_mode || '').toLowerCase())
    ? String(raw.period_mode).toLowerCase()
    : 'period';
  const totalDefault = mode === 'halves' ? 2 : (mode === 'quarters' ? 4 : 1);
  const totalPeriods = Math.max(1, Number(raw?.total_periods || totalDefault));
  const currentPeriod = Math.min(totalPeriods, Math.max(1, Number(raw?.current_period || 1)));
  const periodLengthMinutes = Math.max(1, Number(raw?.period_length_minutes || selectedPeriodLengthMinutes.value || 5));
  const secondsRemaining = Math.max(0, Number(raw?.seconds_remaining ?? periodLengthMinutes * 60));
  const display = String(raw?.display || formatClockDisplay(secondsRemaining));
  const segmentLabel = String(raw?.segment_label || segmentLabelForMode(mode, currentPeriod));
  return {
    period_mode: mode,
    total_periods: totalPeriods,
    current_period: currentPeriod,
    period_length_minutes: periodLengthMinutes,
    seconds_remaining: secondsRemaining,
    display,
    segment_label: segmentLabel,
  };
}

function normalizeGameResult(raw) {
  const gameOver = Boolean(raw?.game_over);
  const winnerRaw = String(raw?.winner || '').toLowerCase();
  const winner = ['user', 'ai', 'tie'].includes(winnerRaw) ? winnerRaw : null;
  const scoreRaw = raw?.score && typeof raw.score === 'object' ? raw.score : {};
  return {
    game_over: gameOver,
    winner,
    message: gameOver ? String(raw?.message || 'Game over.') : '',
    score: {
      user: Number(scoreRaw.user ?? score.value.user ?? 0) || 0,
      ai: Number(scoreRaw.ai ?? score.value.ai ?? 0) || 0,
    },
  };
}

function shuffleArray(items) {
  const arr = [...items];
  for (let i = arr.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

function getPlayableAllPlayerIds(state) {
  const userIds = Array.isArray(state?.playable_user_ids) ? state.playable_user_ids : [];
  const aiIds = Array.isArray(state?.playable_ai_ids) ? state.playable_ai_ids : [];
  const merged = [...userIds, ...aiIds]
    .map((v) => Number(v))
    .filter((v) => Number.isFinite(v));
  if (merged.length > 0) {
    return Array.from(new Set(merged)).sort((a, b) => a - b);
  }
  const positions = Array.isArray(state?.positions) ? state.positions : [];
  return Array.from({ length: positions.length }, (_, idx) => idx);
}

function assignRandomPlayablePlayerNames(state) {
  const ids = getPlayableAllPlayerIds(state);
  if (ids.length === 0) {
    playerDisplayNames.value = {};
    return;
  }
  const shuffled = shuffleArray(PLAYABLE_SURNAME_POOL);
  const next = {};
  for (let i = 0; i < ids.length; i += 1) {
    const pid = ids[i];
    next[pid] = String(shuffled[i] || `P${pid}`).toUpperCase();
  }
  playerDisplayNames.value = next;
}

function ensurePlayablePlayerNames(state) {
  const ids = getPlayableAllPlayerIds(state);
  if (ids.length === 0) {
    playerDisplayNames.value = {};
    return;
  }
  const current = playerDisplayNames.value && typeof playerDisplayNames.value === 'object'
    ? playerDisplayNames.value
    : {};
  const next = {};
  const used = new Set();
  const missing = [];

  for (const pid of ids) {
    const raw = current[pid] ?? current[String(pid)];
    const name = typeof raw === 'string' ? raw.trim().toUpperCase() : '';
    if (name) {
      next[pid] = name;
      used.add(name);
    } else {
      missing.push(pid);
    }
  }

  if (missing.length > 0) {
    const candidates = shuffleArray(
      PLAYABLE_SURNAME_POOL.filter((name) => !used.has(String(name).toUpperCase())),
    );
    for (let i = 0; i < missing.length; i += 1) {
      const pid = missing[i];
      const fallback = String(candidates[i] || `P${pid}`).toUpperCase();
      next[pid] = fallback;
      used.add(fallback);
    }
  }

  playerDisplayNames.value = next;
}

function getPlayablePlayerName(playerId) {
  const id = Number(playerId);
  if (!Number.isFinite(id)) return '';
  const map = playerDisplayNames.value && typeof playerDisplayNames.value === 'object'
    ? playerDisplayNames.value
    : {};
  const raw = map[id] ?? map[String(id)];
  return typeof raw === 'string' ? raw.trim() : '';
}

function formatPlayablePlayerRef(playerId) {
  const id = Number(playerId);
  if (!Number.isFinite(id)) return 'Unknown';
  const surname = getPlayablePlayerName(id);
  if (surname) return `${surname} #${id}`;
  return `Player ${id}`;
}

function getWeightedRandomJerseyNumber() {
  if (PLAYABLE_JERSEY_TOTAL_WEIGHT <= 0) return String(Math.max(0, Number(0)));
  let draw = Math.random() * PLAYABLE_JERSEY_TOTAL_WEIGHT;
  for (const row of PLAYABLE_JERSEY_NUMBER_WEIGHTS) {
    draw -= Math.max(0, Number(row?.weight || 0));
    if (draw <= 0) return String(row.number);
  }
  return String(PLAYABLE_JERSEY_NUMBER_WEIGHTS[PLAYABLE_JERSEY_NUMBER_WEIGHTS.length - 1]?.number || '0');
}

function drawTeamJerseyNumber(usedByTeam) {
  const maxTries = 48;
  for (let i = 0; i < maxTries; i += 1) {
    const candidate = getWeightedRandomJerseyNumber();
    if (!usedByTeam.has(candidate)) return candidate;
  }
  for (const row of PLAYABLE_JERSEY_NUMBER_WEIGHTS) {
    const candidate = String(row.number);
    if (!usedByTeam.has(candidate)) return candidate;
  }
  return getWeightedRandomJerseyNumber();
}

function assignRandomPlayableJerseyNumbers(state) {
  const ids = getPlayableAllPlayerIds(state);
  if (ids.length === 0) {
    playerJerseyNumbers.value = {};
    return;
  }

  const userIds = (Array.isArray(state?.playable_user_ids) ? state.playable_user_ids : [])
    .map((v) => Number(v))
    .filter((v) => Number.isFinite(v))
    .sort((a, b) => a - b);
  const aiIds = (Array.isArray(state?.playable_ai_ids) ? state.playable_ai_ids : [])
    .map((v) => Number(v))
    .filter((v) => Number.isFinite(v))
    .sort((a, b) => a - b);

  const userSet = new Set(userIds);
  const aiSet = new Set(aiIds);
  const next = {};
  const userUsed = new Set();
  const aiUsed = new Set();

  for (const pid of userIds) {
    const jersey = drawTeamJerseyNumber(userUsed);
    next[pid] = jersey;
    userUsed.add(jersey);
  }

  for (const pid of aiIds) {
    const jersey = drawTeamJerseyNumber(aiUsed);
    next[pid] = jersey;
    aiUsed.add(jersey);
  }

  for (const pid of ids) {
    if (next[pid] !== undefined) continue;
    const useUserPool = userSet.has(pid);
    const useAiPool = aiSet.has(pid);
    const used = useUserPool ? userUsed : (useAiPool ? aiUsed : new Set());
    const jersey = drawTeamJerseyNumber(used);
    next[pid] = jersey;
    used.add(jersey);
  }

  playerJerseyNumbers.value = next;
}

function ensurePlayableJerseyNumbers(state) {
  const ids = getPlayableAllPlayerIds(state);
  if (ids.length === 0) {
    playerJerseyNumbers.value = {};
    return;
  }

  const current = playerJerseyNumbers.value && typeof playerJerseyNumbers.value === 'object'
    ? playerJerseyNumbers.value
    : {};
  const userIds = (Array.isArray(state?.playable_user_ids) ? state.playable_user_ids : [])
    .map((v) => Number(v))
    .filter((v) => Number.isFinite(v))
    .sort((a, b) => a - b);
  const aiIds = (Array.isArray(state?.playable_ai_ids) ? state.playable_ai_ids : [])
    .map((v) => Number(v))
    .filter((v) => Number.isFinite(v))
    .sort((a, b) => a - b);
  const userSet = new Set(userIds);
  const aiSet = new Set(aiIds);

  const next = {};
  const userUsed = new Set();
  const aiUsed = new Set();

  const copyExisting = (pid, teamUsed) => {
    const raw = current[pid] ?? current[String(pid)];
    const jersey = typeof raw === 'string' ? raw.trim() : String(raw ?? '').trim();
    if (!jersey) return false;
    if (teamUsed.has(jersey)) return false;
    next[pid] = jersey;
    teamUsed.add(jersey);
    return true;
  };

  for (const pid of userIds) {
    if (!copyExisting(pid, userUsed)) {
      const jersey = drawTeamJerseyNumber(userUsed);
      next[pid] = jersey;
      userUsed.add(jersey);
    }
  }

  for (const pid of aiIds) {
    if (!copyExisting(pid, aiUsed)) {
      const jersey = drawTeamJerseyNumber(aiUsed);
      next[pid] = jersey;
      aiUsed.add(jersey);
    }
  }

  for (const pid of ids) {
    if (next[pid] !== undefined) continue;
    const useUserPool = userSet.has(pid);
    const useAiPool = aiSet.has(pid);
    const used = useUserPool ? userUsed : (useAiPool ? aiUsed : new Set());
    if (!copyExisting(pid, used)) {
      const jersey = drawTeamJerseyNumber(used);
      next[pid] = jersey;
      used.add(jersey);
    }
  }

  playerJerseyNumbers.value = next;
}

function createShotLine() {
  return { attempts: 0, made: 0 };
}

function createPlayerStatsLine() {
  return {
    shots: {
      total: createShotLine(),
      twoPt: createShotLine(),
      threePt: createShotLine(),
      dunk: createShotLine(),
    },
    assists: 0,
    turnovers: 0,
    points: 0,
    actions: {
      noop: 0,
      move: 0,
      shoot: 0,
      pass: 0,
      other: 0,
      total: 0,
    },
  };
}

function createTeamStatsLine() {
  return {
    shots: {
      total: createShotLine(),
      twoPt: createShotLine(),
      threePt: createShotLine(),
      dunk: createShotLine(),
    },
    assists: 0,
    turnovers: 0,
    defensiveViolations: 0,
    offensiveViolations: 0,
    actions: {
      noop: 0,
      move: 0,
      shoot: 0,
      pass: 0,
      other: 0,
      total: 0,
    },
    perPlayer: {},
  };
}

function createPlayableStats() {
  return {
    turns: 0,
    possessions: 0,
    userOffensiveTurns: 0,
    userDefensiveTurns: 0,
    byTeam: {
      user: createTeamStatsLine(),
      ai: createTeamStatsLine(),
    },
    turnoverReasons: {},
  };
}

const playableStats = ref(createPlayableStats());
const playablePlayByPlay = ref([]);
const playByPlayEventCounter = ref(1);

function resetPlayableStats() {
  playableStats.value = createPlayableStats();
}

function resetPlayablePlayByPlay() {
  playablePlayByPlay.value = [];
  playByPlayEventCounter.value = 1;
}

function toIdSet(values) {
  const out = new Set();
  if (!Array.isArray(values)) return out;
  for (const raw of values) {
    const id = Number(raw);
    if (Number.isFinite(id)) out.add(id);
  }
  return out;
}

function resolveTeamKey(playerId, userIds, aiIds) {
  if (userIds.has(playerId)) return 'user';
  if (aiIds.has(playerId)) return 'ai';
  return null;
}

function classifyActionBucket(actionName, actionMeta = null) {
  const rawName = String(actionName || '').toUpperCase();
  const metaType = String(actionMeta?.type || '').toUpperCase();
  const name = metaType || rawName;

  if (name === 'NOOP') return 'noop';
  if (name === 'SHOOT' || rawName === 'SHOOT') return 'shoot';
  if (name === 'PASS' || rawName.startsWith('PASS')) return 'pass';
  if (name.startsWith('MOVE') || rawName.startsWith('MOVE_')) return 'move';
  return 'other';
}

function recordPlayableStepStats(payload, context = {}) {
  const stats = playableStats.value;
  if (!stats || typeof stats !== 'object') return;

  stats.turns = Number(stats.turns || 0) + 1;
  if (context.userOnOffense) {
    stats.userOffensiveTurns = Number(stats.userOffensiveTurns || 0) + 1;
  } else {
    stats.userDefensiveTurns = Number(stats.userDefensiveTurns || 0) + 1;
  }
  if (payload?.possession_ended) {
    stats.possessions = Number(stats.possessions || 0) + 1;
  }

  const userIds = context.userIds instanceof Set ? context.userIds : new Set();
  const aiIds = context.aiIds instanceof Set ? context.aiIds : new Set();
  const ensurePlayerLine = (teamKey, playerId) => {
    if (!teamKey) return null;
    const teamLine = stats.byTeam?.[teamKey];
    if (!teamLine) return null;
    const key = String(playerId);
    if (!teamLine.perPlayer || typeof teamLine.perPlayer !== 'object') {
      teamLine.perPlayer = {};
    }
    if (!teamLine.perPlayer[key]) {
      teamLine.perPlayer[key] = createPlayerStatsLine();
    }
    return teamLine.perPlayer[key];
  };

  const actionsTaken = payload?.actions_taken || {};
  const actionsTakenMeta = payload?.actions_taken_meta || {};
  for (const [pidRaw, actionName] of Object.entries(actionsTaken)) {
    const pid = Number(pidRaw);
    if (!Number.isFinite(pid)) continue;
    const teamKey = resolveTeamKey(pid, userIds, aiIds);
    if (!teamKey) continue;
    const actionMeta = actionsTakenMeta?.[pidRaw] ?? actionsTakenMeta?.[String(pid)] ?? null;
    const bucket = classifyActionBucket(actionName, actionMeta);
    const teamActions = stats.byTeam?.[teamKey]?.actions;
    if (!teamActions) continue;
    teamActions[bucket] = Number(teamActions[bucket] || 0) + 1;
    teamActions.total = Number(teamActions.total || 0) + 1;
    const playerLine = ensurePlayerLine(teamKey, pid);
    if (playerLine?.actions) {
      playerLine.actions[bucket] = Number(playerLine.actions[bucket] || 0) + 1;
      playerLine.actions.total = Number(playerLine.actions.total || 0) + 1;
    }
  }

  const stepResults = payload?.possession_ended
    ? payload?.ended_state?.last_action_results
    : payload?.state?.last_action_results;
  if (!stepResults || typeof stepResults !== 'object') return;

  const shots = stepResults?.shots && typeof stepResults.shots === 'object' ? stepResults.shots : {};
  for (const [shooterRaw, shotRaw] of Object.entries(shots)) {
    const shooterId = Number(shooterRaw);
    if (!Number.isFinite(shooterId)) continue;
    const teamKey = resolveTeamKey(shooterId, userIds, aiIds);
    if (!teamKey) continue;
    const shot = shotRaw && typeof shotRaw === 'object' ? shotRaw : {};
    const isDunk = Boolean(shot.is_dunk);
    const isThree = !isDunk && Boolean(shot.is_three);
    const shotKey = isDunk ? 'dunk' : (isThree ? 'threePt' : 'twoPt');
    const teamShots = stats.byTeam?.[teamKey]?.shots;
    if (!teamShots) continue;
    const playerLine = ensurePlayerLine(teamKey, shooterId);
    teamShots.total.attempts = Number(teamShots.total.attempts || 0) + 1;
    teamShots[shotKey].attempts = Number(teamShots[shotKey].attempts || 0) + 1;
    if (playerLine?.shots) {
      playerLine.shots.total.attempts = Number(playerLine.shots.total.attempts || 0) + 1;
      playerLine.shots[shotKey].attempts = Number(playerLine.shots[shotKey].attempts || 0) + 1;
    }
    if (Boolean(shot.success)) {
      const points = isThree ? 3 : 2;
      teamShots.total.made = Number(teamShots.total.made || 0) + 1;
      teamShots[shotKey].made = Number(teamShots[shotKey].made || 0) + 1;
      if (playerLine?.shots) {
        playerLine.shots.total.made = Number(playerLine.shots.total.made || 0) + 1;
        playerLine.shots[shotKey].made = Number(playerLine.shots[shotKey].made || 0) + 1;
      }
      if (playerLine) {
        playerLine.points = Number(playerLine.points || 0) + points;
      }
    }

    const assistPasserId = Number(shot?.assist_passer_id);
    if (Boolean(shot?.assist_full) && Number.isFinite(assistPasserId)) {
      const passerTeamKey = resolveTeamKey(assistPasserId, userIds, aiIds);
      if (passerTeamKey && stats.byTeam?.[passerTeamKey]) {
        stats.byTeam[passerTeamKey].assists = Number(stats.byTeam[passerTeamKey].assists || 0) + 1;
        const passerLine = ensurePlayerLine(passerTeamKey, assistPasserId);
        if (passerLine) {
          passerLine.assists = Number(passerLine.assists || 0) + 1;
        }
      }
    }
  }

  const turnovers = Array.isArray(stepResults?.turnovers) ? stepResults.turnovers : [];
  for (const turnover of turnovers) {
    const playerId = Number(turnover?.player_id);
    if (!Number.isFinite(playerId)) continue;
    const teamKey = resolveTeamKey(playerId, userIds, aiIds);
    if (!teamKey) continue;
    stats.byTeam[teamKey].turnovers = Number(stats.byTeam[teamKey].turnovers || 0) + 1;
    const playerLine = ensurePlayerLine(teamKey, playerId);
    if (playerLine) {
      playerLine.turnovers = Number(playerLine.turnovers || 0) + 1;
    }
    const reason = String(turnover?.reason || 'unknown');
    stats.turnoverReasons[reason] = Number(stats.turnoverReasons[reason] || 0) + 1;
  }

  const defensiveViolations = Array.isArray(stepResults?.defensive_lane_violations)
    ? stepResults.defensive_lane_violations
    : [];
  for (const violation of defensiveViolations) {
    const playerId = Number(violation?.player_id);
    if (!Number.isFinite(playerId)) continue;
    const teamKey = resolveTeamKey(playerId, userIds, aiIds);
    if (!teamKey) continue;
    stats.byTeam[teamKey].defensiveViolations = Number(stats.byTeam[teamKey].defensiveViolations || 0) + 1;
  }

  const offensiveViolations = Array.isArray(stepResults?.offensive_lane_violations)
    ? stepResults.offensive_lane_violations
    : [];
  for (const violation of offensiveViolations) {
    const playerId = Number(violation?.player_id);
    if (!Number.isFinite(playerId)) continue;
    const teamKey = resolveTeamKey(playerId, userIds, aiIds);
    if (!teamKey) continue;
    stats.byTeam[teamKey].offensiveViolations = Number(stats.byTeam[teamKey].offensiveViolations || 0) + 1;
  }
}

const PLAY_BY_PLAY_MAX_ROWS = 240;

function asTeamLabel(teamKey) {
  if (teamKey === 'user') return 'YOU';
  if (teamKey === 'ai') return 'AI';
  return '-';
}

function formatTurnoverReason(reasonRaw) {
  const reason = String(reasonRaw || '').trim().toLowerCase();
  if (!reason) return 'turnover';
  if (reason === 'defender_pressure') return 'defender pressure';
  if (reason === 'offensive_three_seconds') return 'offensive 3-second violation';
  if (reason === 'move_out_of_bounds') return 'out of bounds';
  if (reason === 'pass_out_of_bounds') return 'pass out of bounds';
  return reason.replace(/_/g, ' ');
}

function getStepResults(payload) {
  const fromEnded = payload?.ended_state?.last_action_results;
  const fromState = payload?.state?.last_action_results;
  if (payload?.possession_ended && fromEnded && typeof fromEnded === 'object') return fromEnded;
  if (fromState && typeof fromState === 'object') return fromState;
  return null;
}

function nextPlayByPlayEntry(base = {}) {
  return {
    id: playByPlayEventCounter.value++,
    period: String(base.period || '-'),
    clock: String(base.clock || '--:--'),
    team: String(base.team || '-'),
    score: String(base.score || '0-0'),
    event: String(base.event || ''),
  };
}

function formatPlayByPlayScore(rawScore) {
  const scoreObj = rawScore && typeof rawScore === 'object' ? rawScore : {};
  const user = Number(scoreObj.user ?? score.value?.user ?? 0) || 0;
  const ai = Number(scoreObj.ai ?? score.value?.ai ?? 0) || 0;
  return `${user}-${ai}`;
}

function appendPlayByPlayEntries(rows) {
  if (!Array.isArray(rows) || rows.length === 0) return;
  const valid = rows.filter((row) => row && typeof row.event === 'string' && row.event.trim().length > 0);
  if (valid.length === 0) return;
  playablePlayByPlay.value = [...valid, ...playablePlayByPlay.value].slice(0, PLAY_BY_PLAY_MAX_ROWS);
}

function recordPlayableStartPlayByPlay(payload) {
  const possessionInfo = payload?.possession || {};
  const offenseTeam = possessionInfo.offense_team === 'ai' ? 'AI' : 'YOU';
  const clockInfo = payload?.game_clock || {};
  const scoreLabel = formatPlayByPlayScore(payload?.score);
  appendPlayByPlayEntries([
    nextPlayByPlayEntry({
      period: String(clockInfo.segment_label || 'Period 1'),
      clock: String(clockInfo.display || '00:00'),
      team: offenseTeam,
      score: scoreLabel,
      event: `${offenseTeam} won the tip and starts with possession.`,
    }),
  ]);
}

function recordPlayableStepPlayByPlay(payload, context = {}) {
  const userIds = context.userIds instanceof Set ? context.userIds : new Set();
  const aiIds = context.aiIds instanceof Set ? context.aiIds : new Set();
  const resolveTeam = (playerId) => resolveTeamKey(Number(playerId), userIds, aiIds);
  const clockInfo = payload?.game_clock || {};
  let period = String(clockInfo.segment_label || gameClock.value?.segment_label || '-');
  const baseClock = String(clockInfo.display || gameClock.value?.display || '--:--');
  const clock = payload?.period_ended ? '00:00' : baseClock;
  const periodMessage = String(payload?.period_result?.message || '');
  if (payload?.period_ended && periodMessage) {
    const periodMatch = periodMessage.match(/^End of\s+(.+?)\.?$/i);
    if (periodMatch?.[1]) {
      period = String(periodMatch[1]).trim();
    }
  }
  const scoreLabel = formatPlayByPlayScore(payload?.score);
  const rows = [];
  const pushEvent = (teamKey, eventText) => {
    rows.push(
      nextPlayByPlayEntry({
        period,
        clock,
        team: asTeamLabel(teamKey),
        score: scoreLabel,
        event: eventText,
      }),
    );
  };

  const stepResults = getStepResults(payload);
  if (stepResults && typeof stepResults === 'object') {
    const turnovers = Array.isArray(stepResults?.turnovers) ? stepResults.turnovers : [];
    const turnoverPlayerIds = new Set(
      turnovers
        .map((turnover) => Number(turnover?.player_id))
        .filter((playerId) => Number.isFinite(playerId)),
    );

    const shots = stepResults?.shots && typeof stepResults.shots === 'object' ? stepResults.shots : {};
    for (const [shooterRaw, shotRaw] of Object.entries(shots)) {
      const shooterId = Number(shooterRaw);
      if (!Number.isFinite(shooterId)) continue;
      const shot = shotRaw && typeof shotRaw === 'object' ? shotRaw : {};
      const isDunk = Boolean(shot.is_dunk) || Number(shot.distance) === 0;
      const isThree = !isDunk && Boolean(shot.is_three);
      const shotLabel = isDunk ? 'Dunk' : (isThree ? '3pt shot' : '2pt shot');
      let event = `${shotLabel} ${Boolean(shot.success) ? 'made' : 'missed'} by ${formatPlayablePlayerRef(shooterId)}.`;
      const assistPasserId = Number(shot.assist_passer_id);
      if (Boolean(shot.success) && Boolean(shot.assist_full) && Number.isFinite(assistPasserId)) {
        event = `${event.slice(0, -1)} (assist: ${formatPlayablePlayerRef(assistPasserId)}).`;
      }
      pushEvent(resolveTeam(shooterId), event);
    }

    const passes = stepResults?.passes && typeof stepResults.passes === 'object' ? stepResults.passes : {};
    for (const [passerRaw, passRaw] of Object.entries(passes)) {
      const passerId = Number(passerRaw);
      if (!Number.isFinite(passerId)) continue;
      if (turnoverPlayerIds.has(passerId)) continue;
      const passInfo = passRaw && typeof passRaw === 'object' ? passRaw : {};
      if (Boolean(passInfo.success)) {
        const targetId = Number(passInfo.target);
        if (Number.isFinite(targetId)) {
          pushEvent(
            resolveTeam(passerId),
            `Pass completed: ${formatPlayablePlayerRef(passerId)} to ${formatPlayablePlayerRef(targetId)}.`,
          );
        } else {
          pushEvent(resolveTeam(passerId), `Pass completed by ${formatPlayablePlayerRef(passerId)}.`);
        }
      }
    }

    for (const turnover of turnovers) {
      const playerId = Number(turnover?.player_id);
      if (!Number.isFinite(playerId)) continue;
      const stolenBy = Number(turnover?.stolen_by);
      const reason = formatTurnoverReason(turnover?.reason);
      if (Number.isFinite(stolenBy)) {
        pushEvent(
          resolveTeam(stolenBy),
          `Steal by ${formatPlayablePlayerRef(stolenBy)} from ${formatPlayablePlayerRef(playerId)}.`,
        );
      } else {
        pushEvent(resolveTeam(playerId), `Turnover by ${formatPlayablePlayerRef(playerId)} (${reason}).`);
      }
    }

    const defensiveViolations = Array.isArray(stepResults?.defensive_lane_violations)
      ? stepResults.defensive_lane_violations
      : [];
    for (const violation of defensiveViolations) {
      const playerId = Number(violation?.player_id);
      if (!Number.isFinite(playerId)) continue;
      const scoringTeam = String(payload?.possession_result?.scoring_team || '').toLowerCase();
      const bonusTeam = scoringTeam === 'user' || scoringTeam === 'ai'
        ? ` (+1 ${asTeamLabel(scoringTeam)})`
        : '';
      pushEvent(
        resolveTeam(playerId),
        `Defensive lane violation by ${formatPlayablePlayerRef(playerId)}.${bonusTeam}`,
      );
    }

    const offensiveViolations = Array.isArray(stepResults?.offensive_lane_violations)
      ? stepResults.offensive_lane_violations
      : [];
    for (const violation of offensiveViolations) {
      const playerId = Number(violation?.player_id);
      if (!Number.isFinite(playerId)) continue;
      pushEvent(resolveTeam(playerId), `Offensive lane violation by ${formatPlayablePlayerRef(playerId)}.`);
    }
  }

  if (rows.length === 0) {
    const possessionMessage = String(payload?.possession_result?.message || '').trim();
    if (possessionMessage) {
      const scoringTeam = String(payload?.possession_result?.scoring_team || '').toLowerCase();
      const teamKey = scoringTeam === 'user' || scoringTeam === 'ai' ? scoringTeam : null;
      pushEvent(teamKey, possessionMessage);
    }
  }

  if (payload?.period_ended && payload?.period_result?.message) {
    pushEvent(null, String(payload.period_result.message));
  }

  if (payload?.game_result?.game_over) {
    const winner = String(payload?.game_result?.winner || '').toLowerCase();
    const winnerTeam = winner === 'user' || winner === 'ai' ? winner : null;
    pushEvent(winnerTeam, String(payload?.game_result?.message || 'Game over.'));
  }

  appendPlayByPlayEntries(rows);
}

const playerChoices = computed(() => {
  const values = optionsPayload.value?.players_per_side;
  const baseChoices = (
    Array.isArray(values) && values.length > 0
      ? values
      : [1, 2, 3, 4, 5]
  )
    .map((v) => Number(v))
    .filter((v) => Number.isFinite(v));

  const matrix = optionsPayload.value?.options;
  if (!matrix || typeof matrix !== 'object') return baseChoices;

  return baseChoices.filter((players) => {
    const row = matrix[String(players)];
    if (!row || typeof row !== 'object') return false;
    return Object.values(row).some((cfg) => Boolean(cfg?.available));
  });
});

const difficultyChoices = computed(() => {
  const values = optionsPayload.value?.difficulties;
  if (!Array.isArray(values) || values.length === 0) return ['easy', 'medium', 'hard'];
  return values.map((v) => String(v || '').toLowerCase());
});

const periodModeChoices = computed(() => {
  const values = optionsPayload.value?.period_modes;
  const fallback = ['period', 'halves', 'quarters'];
  const modes = Array.isArray(values) && values.length > 0 ? values : fallback;
  return modes
    .map((v) => String(v || '').toLowerCase())
    .filter((v) => ['period', 'halves', 'quarters'].includes(v));
});

const periodLengthBounds = computed(() => {
  const src = optionsPayload.value?.period_length_minutes || {};
  const min = Math.max(1, Number(src.min || 1));
  const max = Math.min(60, Math.max(min, Number(src.max || 60)));
  return { min, max };
});

const periodLengthChoices = computed(() => {
  const out = [];
  for (let m = periodLengthBounds.value.min; m <= periodLengthBounds.value.max; m += 1) {
    out.push(m);
  }
  return out;
});

const difficultyEntries = computed(() => {
  const matrix = optionsPayload.value?.options || {};
  const row = matrix[String(selectedPlayersPerSide.value)] || {};
  return difficultyChoices.value.map((difficulty) => {
    const cfg = row[difficulty] || {};
    return {
      value: difficulty,
      available: Boolean(cfg.available),
      reason: cfg.reason ? String(cfg.reason) : '',
    };
  });
});

const currentDifficultyConfig = computed(() => {
  return difficultyEntries.value.find((entry) => entry.value === selectedDifficulty.value) || null;
});

const hasGame = computed(() => !!gameState.value);
const isGameOver = computed(() => Boolean(gameResult.value?.game_over));
const showGameOverOverlay = computed(() => Boolean(isGameOver.value || forceGameOverPreview.value));

const canStartGame = computed(() => {
  return (
    !optionsLoading.value
    && !actionLoading.value
    && !transitionRunning.value
    && Boolean(currentDifficultyConfig.value?.available)
    && (!hasGame.value || isGameOver.value)
  );
});
const controlsDisabled = computed(() => actionLoading.value || transitionRunning.value || isGameOver.value);
const setupLocked = computed(() => optionsLoading.value || actionLoading.value || transitionRunning.value || (hasGame.value && !isGameOver.value));

const userPlayerIds = computed(() => {
  const ids = gameState.value?.playable_user_ids;
  if (Array.isArray(ids) && ids.length > 0) {
    return ids
      .map((v) => Number(v))
      .filter((v) => Number.isFinite(v))
      .sort((a, b) => a - b);
  }
  const players = Number(sessionConfig.value?.players_per_side || selectedPlayersPerSide.value || 0);
  return Array.from({ length: Math.max(0, players) }, (_, idx) => idx);
});

const aiPlayerIds = computed(() => {
  const ids = gameState.value?.playable_ai_ids;
  if (Array.isArray(ids) && ids.length > 0) {
    return ids
      .map((v) => Number(v))
      .filter((v) => Number.isFinite(v))
      .sort((a, b) => a - b);
  }
  const players = Number(sessionConfig.value?.players_per_side || selectedPlayersPerSide.value || 0);
  return Array.from({ length: Math.max(0, players) }, (_, idx) => idx + players);
});

function laneStepValueForPlayer(stepMap, playerId) {
  if (!stepMap || typeof stepMap !== 'object') return 0;
  const direct = Number(stepMap[playerId]);
  if (Number.isFinite(direct)) return Math.max(0, Math.trunc(direct));
  const byString = Number(stepMap[String(playerId)]);
  if (Number.isFinite(byString)) return Math.max(0, Math.trunc(byString));
  return 0;
}

function buildScoreLaneMeter(teamKey) {
  const gs = gameState.value;
  if (!gs || !hasGame.value) {
    return {
      role: '-',
      current: 0,
      max: 3,
      violation: false,
      lights: [],
      title: 'Lane steps: 0/3',
    };
  }

  const maxSteps = Math.max(1, Math.trunc(Number(gs.three_second_max_steps || 3)));
  const teamIds = (teamKey === 'user' ? userPlayerIds.value : aiPlayerIds.value)
    .map((id) => Number(id))
    .filter((id) => Number.isFinite(id));
  const teamIdSet = new Set(teamIds);

  const offenseIds = (Array.isArray(gs.offense_ids) ? gs.offense_ids : [])
    .map((id) => Number(id))
    .filter((id) => Number.isFinite(id));
  const defenseIds = (Array.isArray(gs.defense_ids) ? gs.defense_ids : [])
    .map((id) => Number(id))
    .filter((id) => Number.isFinite(id));

  const teamOnOffense = offenseIds.some((id) => teamIdSet.has(id));
  const role = teamOnOffense ? 'O' : 'D';
  const sourceMap = teamOnOffense ? gs.offensive_lane_steps : gs.defensive_lane_steps;
  const roleIds = (teamOnOffense ? offenseIds : defenseIds).filter((id) => teamIdSet.has(id));
  const idsToRead = roleIds.length > 0 ? roleIds : teamIds;

  let current = 0;
  for (const pid of idsToRead) {
    current = Math.max(current, laneStepValueForPlayer(sourceMap, pid));
  }
  current = Math.max(0, current);

  const clamped = Math.min(maxSteps, current);
  const lights = Array.from({ length: maxSteps }, (_, idx) => ({
    key: `${teamKey}-lane-${idx}`,
    lit: idx >= (maxSteps - clamped),
  }));

  return {
    role,
    current,
    max: maxSteps,
    violation: current >= maxSteps,
    lights,
    title: `Lane steps (${role === 'O' ? 'Offense' : 'Defense'}): ${current}/${maxSteps}`,
  };
}

const userLaneMeter = computed(() => buildScoreLaneMeter('user'));
const aiLaneMeter = computed(() => buildScoreLaneMeter('ai'));

const forcedEpisodeOutcomePreview = computed(() => {
  const mode = String(violationOverlayPreview.value || '').toLowerCase();
  if (!mode || !hasGame.value || !gameState.value) return null;

  if (mode === 'defensive') {
    const aiId = aiPlayerIds.value.length > 0 ? Number(aiPlayerIds.value[0]) : null;
    return {
      type: 'DEFENSIVE_VIOLATION',
      playerId: Number.isFinite(aiId) ? aiId : null,
    };
  }

  if (mode === 'offensive') {
    const userId = userPlayerIds.value.length > 0 ? Number(userPlayerIds.value[0]) : null;
    return {
      type: 'OFFENSIVE_VIOLATION',
      playerId: Number.isFinite(userId) ? userId : null,
    };
  }

  return null;
});

const possessionSummary = computed(() => {
  if (!possession.value) return '';
  const offenseTeam = possession.value.offense_team === 'user' ? 'You (Blue)' : 'AI (Red)';
  return `Possession ${possession.value.number}: ${offenseTeam} on offense`;
});

const selectedModeLabel = computed(() => {
  if (!sessionConfig.value) return '';
  const modeLabel = periodModeLabelMap[String(sessionConfig.value.period_mode || 'period')] || '1 period';
  const minutes = Number(sessionConfig.value.period_length_minutes || 5);
  return (
    `${sessionConfig.value.players_per_side}v${sessionConfig.value.players_per_side}`
    + ` · ${sessionConfig.value.difficulty}`
    + ` · ${modeLabel} x ${minutes} min`
  );
});

const formattedUserScore = computed(() => String(Number(score.value?.user || 0)).padStart(2, '0'));
const formattedAiScore = computed(() => String(Number(score.value?.ai || 0)).padStart(2, '0'));
const gameClockDisplay = computed(() => String(gameClock.value?.display || '00:00'));
const gameClockSegmentLabel = computed(() => String(gameClock.value?.segment_label || 'Period 1'));
const gameOverFinalUserScore = computed(() => Number(gameResult.value?.score?.user ?? score.value?.user ?? 0));
const gameOverFinalAiScore = computed(() => Number(gameResult.value?.score?.ai ?? score.value?.ai ?? 0));

const gameOverWinnerKey = computed(() => {
  const winnerRaw = String(gameResult.value?.winner || '').toLowerCase();
  if (winnerRaw === 'user' || winnerRaw === 'ai' || winnerRaw === 'tie') return winnerRaw;
  if (gameOverFinalUserScore.value > gameOverFinalAiScore.value) return 'user';
  if (gameOverFinalAiScore.value > gameOverFinalUserScore.value) return 'ai';
  return 'tie';
});

const gameOverWinnerLabel = computed(() => {
  if (!showGameOverOverlay.value) return '';
  if (gameOverWinnerKey.value === 'user') return 'YOU WIN';
  if (gameOverWinnerKey.value === 'ai') return 'AI WINS';
  return 'TIE GAME';
});

const gameOverScoreLabel = computed(() => {
  const user = gameOverFinalUserScore.value;
  const ai = gameOverFinalAiScore.value;
  if (gameOverWinnerKey.value === 'user') return `${user}-${ai}`;
  if (gameOverWinnerKey.value === 'ai') return `${ai}-${user}`;
  if (user === ai) return `${user}-${ai}`;
  return `${Math.max(user, ai)}-${Math.min(user, ai)}`;
});

const gameOverBannerText = computed(() => {
  if (!showGameOverOverlay.value) return '';
  const suffix = gameOverWinnerKey.value === 'tie' ? '' : '!';
  return `${gameOverWinnerLabel.value} ${gameOverScoreLabel.value}${suffix}`;
});

function toggleViolationOverlayPreview(mode) {
  const normalized = String(mode || '').toLowerCase();
  if (!['defensive', 'offensive'].includes(normalized)) return;
  violationOverlayPreview.value = violationOverlayPreview.value === normalized ? null : normalized;
}

watch(
  playerChoices,
  (choices) => {
    if (choices.length === 0) return;
    if (!choices.includes(Number(selectedPlayersPerSide.value))) {
      selectedPlayersPerSide.value = choices[0];
    }
  },
  { immediate: true },
);

watch(
  [difficultyEntries, selectedPlayersPerSide],
  ([entries]) => {
    if (!entries || entries.length === 0) return;
    const currentlySelected = entries.find((entry) => entry.value === selectedDifficulty.value);
    if (currentlySelected?.available) return;
    const firstAvailable = entries.find((entry) => entry.available);
    if (firstAvailable) {
      selectedDifficulty.value = firstAvailable.value;
      return;
    }
    selectedDifficulty.value = entries[0].value;
  },
  { immediate: true },
);

watch(
  periodModeChoices,
  (choices) => {
    if (!choices || choices.length === 0) return;
    if (!choices.includes(String(selectedPeriodMode.value))) {
      selectedPeriodMode.value = choices[0];
    }
  },
  { immediate: true },
);

watch(
  [periodLengthChoices, periodLengthBounds],
  ([choices, bounds]) => {
    if (!Array.isArray(choices) || choices.length === 0) return;
    const current = Number(selectedPeriodLengthMinutes.value || 5);
    if (choices.includes(current)) return;
    selectedPeriodLengthMinutes.value = Math.min(bounds.max, Math.max(bounds.min, current));
  },
  { immediate: true },
);

watch(
  userPlayerIds,
  (ids) => {
    if (!ids || ids.length === 0) {
      activePlayerId.value = null;
      return;
    }
    if (!ids.includes(Number(activePlayerId.value))) {
      activePlayerId.value = ids[0];
    }
  },
  { immediate: true },
);

function applyStatePayload(payload) {
  const hadGameBefore = Boolean(gameState.value);
  const state = payload?.state || null;
  gameState.value = state;
  gameHistory.value = state ? [state] : [];
  ensurePlayablePlayerNames(state);
  ensurePlayableJerseyNumbers(state);

  if (payload?.score) {
    score.value = {
      user: Number(payload.score.user) || 0,
      ai: Number(payload.score.ai) || 0,
    };
  }

  if (payload?.possession) {
    possession.value = payload.possession;
  }

  if (payload?.config) {
    const cfgMode = String(payload.config.period_mode || selectedPeriodMode.value || 'period').toLowerCase();
    const cfgMinutes = Number(payload.config.period_length_minutes ?? selectedPeriodLengthMinutes.value ?? 5);
    sessionConfig.value = {
      players_per_side: Number(payload.config.players_per_side),
      difficulty: String(payload.config.difficulty || '').toLowerCase(),
      period_mode: ['period', 'halves', 'quarters'].includes(cfgMode) ? cfgMode : 'period',
      total_periods: Number(payload.config.total_periods || (cfgMode === 'halves' ? 2 : (cfgMode === 'quarters' ? 4 : 1))),
      period_length_minutes: Math.max(1, cfgMinutes),
    };
    selectedPeriodMode.value = sessionConfig.value.period_mode;
    selectedPeriodLengthMinutes.value = sessionConfig.value.period_length_minutes;
  }

  if (payload?.game_clock) {
    gameClock.value = normalizeGameClock(payload.game_clock);
  } else if (payload?.config) {
    gameClock.value = normalizeGameClock({
      period_mode: sessionConfig.value?.period_mode || selectedPeriodMode.value,
      total_periods: sessionConfig.value?.total_periods,
      current_period: 1,
      period_length_minutes: sessionConfig.value?.period_length_minutes || selectedPeriodLengthMinutes.value,
      seconds_remaining: (sessionConfig.value?.period_length_minutes || selectedPeriodLengthMinutes.value || 5) * 60,
    });
  }

  if (payload?.game_result) {
    gameResult.value = normalizeGameResult(payload.game_result);
  } else {
    gameResult.value = normalizeGameResult({
      game_over: false,
      score: payload?.score || score.value,
    });
  }

  if (!state) {
    activePlayerId.value = null;
    return;
  }

  const userIds = Array.isArray(state?.playable_user_ids)
    ? state.playable_user_ids
      .map((v) => Number(v))
      .filter((v) => Number.isFinite(v))
      .sort((a, b) => a - b)
    : [];

  if (userIds.length === 0) {
    activePlayerId.value = null;
    return;
  }

  const currentActive = Number(activePlayerId.value);
  const hasValidActive = userIds.includes(currentActive);
  const userOnOffense = Boolean(payload?.possession?.user_on_offense ?? state?.playable_user_on_offense);
  const ballHolder = Number(state?.ball_holder);
  const ballHolderIsUser = Number.isFinite(ballHolder) && userIds.includes(ballHolder);

  if (!hadGameBefore && userOnOffense && ballHolderIsUser) {
    activePlayerId.value = ballHolder;
    return;
  }

  if (!hasValidActive) {
    activePlayerId.value = userOnOffense && ballHolderIsUser ? ballHolder : userIds[0];
  }
}

function formatActionForDisplay(actionName, actionMeta) {
  const name = typeof actionName === 'string' ? actionName : 'NOOP';
  const metaType = String(actionMeta?.type || '').toUpperCase();
  if ((metaType === 'PASS' || name.startsWith('PASS')) && actionMeta?.target !== undefined && actionMeta?.target !== null) {
    const target = Number(actionMeta.target);
    if (Number.isFinite(target)) {
      return `PASS->${target}`;
    }
  }
  return name;
}

function buildBoardSelectionActions(actionsTaken, actionsTakenMeta = null) {
  const mapped = {};
  if (!actionsTaken || typeof actionsTaken !== 'object') return mapped;
  for (const [pid, actionName] of Object.entries(actionsTaken)) {
    const numericPid = Number(pid);
    if (!Number.isFinite(numericPid)) continue;
    const meta = actionsTakenMeta && typeof actionsTakenMeta === 'object'
      ? actionsTakenMeta[pid]
      : null;
    mapped[numericPid] = formatActionForDisplay(actionName, meta);
  }
  return mapped;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function loadOptions() {
  optionsLoading.value = true;
  error.value = '';
  try {
    optionsPayload.value = await getPlayableOptions();
  } catch (err) {
    error.value = err?.message || 'Failed to load playable options.';
  } finally {
    optionsLoading.value = false;
  }
}

async function onStartGame() {
  if (!canStartGame.value) return;
  actionLoading.value = true;
  error.value = '';
  lastPossessionResult.value = null;
  forceGameOverPreview.value = false;
  violationOverlayPreview.value = null;

  try {
    const payload = await startPlayableGame(
      selectedPlayersPerSide.value,
      selectedDifficulty.value,
      selectedPeriodMode.value,
      selectedPeriodLengthMinutes.value,
    );
    resetPlayableStats();
    resetPlayablePlayByPlay();
    applyStatePayload(payload);
    assignRandomPlayablePlayerNames(payload?.state);
    assignRandomPlayableJerseyNumbers(payload?.state);
    recordPlayableStartPlayByPlay(payload);
    boardSelections.value = {};
  } catch (err) {
    error.value = err?.message || 'Failed to start playable game.';
  } finally {
    actionLoading.value = false;
  }
}

async function onNewGame() {
  if (!hasGame.value || actionLoading.value) return;
  actionLoading.value = true;
  error.value = '';
  lastPossessionResult.value = null;
  forceGameOverPreview.value = false;
  violationOverlayPreview.value = null;

  try {
    const payload = await newPlayableGame();
    resetPlayableStats();
    resetPlayablePlayByPlay();
    applyStatePayload(payload);
    assignRandomPlayablePlayerNames(payload?.state);
    assignRandomPlayableJerseyNumbers(payload?.state);
    recordPlayableStartPlayByPlay(payload);
    boardSelections.value = {};
  } catch (err) {
    error.value = err?.message || 'Failed to reset playable game.';
  } finally {
    actionLoading.value = false;
  }
}

function onResetSetup() {
  if (actionLoading.value || transitionRunning.value) return;

  error.value = '';
  lastPossessionResult.value = null;
  forceGameOverPreview.value = false;
  violationOverlayPreview.value = null;

  gameState.value = null;
  gameHistory.value = [];
  activePlayerId.value = null;
  boardSelections.value = {};
  score.value = { user: 0, ai: 0 };
  possession.value = null;
  sessionConfig.value = null;
  playerDisplayNames.value = {};
  playerJerseyNumbers.value = {};

  const modeRaw = String(selectedPeriodMode.value || 'period').toLowerCase();
  const periodMode = ['period', 'halves', 'quarters'].includes(modeRaw) ? modeRaw : 'period';
  const totalPeriods = periodMode === 'halves' ? 2 : (periodMode === 'quarters' ? 4 : 1);
  const periodLengthMinutes = Math.max(1, Number(selectedPeriodLengthMinutes.value || 5));
  gameClock.value = normalizeGameClock({
    period_mode: periodMode,
    total_periods: totalPeriods,
    current_period: 1,
    period_length_minutes: periodLengthMinutes,
    seconds_remaining: periodLengthMinutes * 60,
  });
  gameResult.value = createDefaultGameResult();

  resetPlayableStats();
  resetPlayablePlayByPlay();
}

async function onActionsSubmitted(actions) {
  if (!hasGame.value || actionLoading.value || transitionRunning.value || isGameOver.value) return;
  actionLoading.value = true;
  error.value = '';

  try {
    violationOverlayPreview.value = null;
    const userIds = toIdSet(userPlayerIds.value);
    const aiIds = toIdSet(gameState.value?.playable_ai_ids);
    const userOnOffense = Boolean(possession.value?.user_on_offense);

    const payload = await stepPlayableGame(actions || {});
    recordPlayableStepStats(payload, { userIds, aiIds, userOnOffense });
    recordPlayableStepPlayByPlay(payload, { userIds, aiIds, userOnOffense });
    const possessionEnded = Boolean(payload?.possession_ended);
    const periodEnded = Boolean(payload?.period_ended);
    const gameEnded = Boolean(payload?.game_result?.game_over);
    const possessionMessage = String(payload?.possession_result?.message || '').trim();
    const periodMessage = String(payload?.period_result?.message || '').trim();

    if (gameEnded) {
      lastPossessionResult.value = null;
    } else if (possessionEnded && possessionMessage) {
      lastPossessionResult.value = {
        message: periodEnded && periodMessage
          ? `${possessionMessage} ${periodMessage}`
          : possessionMessage,
      };
    } else if (periodEnded && periodMessage) {
      lastPossessionResult.value = { message: periodMessage };
    } else {
      lastPossessionResult.value = null;
    }

    if (possessionEnded && payload?.ended_state && !Boolean(payload?.game_result?.game_over)) {
      transitionRunning.value = true;
      gameState.value = payload.ended_state;
      gameHistory.value = [payload.ended_state];
      boardSelections.value = buildBoardSelectionActions(payload?.actions_taken, payload?.actions_taken_meta);
      await sleep(1125);
      applyStatePayload(payload);
      boardSelections.value = {};
      transitionRunning.value = false;
      return;
    }

    applyStatePayload(payload);
    boardSelections.value = {};
  } catch (err) {
    error.value = err?.message || 'Failed to process turn.';
  } finally {
    actionLoading.value = false;
    if (!actionLoading.value) {
      transitionRunning.value = false;
    }
  }
}

function onSelectionsChanged(nextSelections) {
  boardSelections.value = nextSelections && typeof nextSelections === 'object' ? nextSelections : {};
}

function getNumericKeyValue(event) {
  const code = String(event?.code || '');
  const key = String(event?.key || '');

  const numpadMatch = code.match(/^Numpad([0-9])$/);
  if (numpadMatch) {
    return Number(numpadMatch[1]);
  }

  if (/^[0-9]$/.test(key)) {
    return Number(key);
  }

  return null;
}

function getMoveActionFromEvent(event) {
  const byCode = MOVE_HOTKEY_ACTIONS_BY_CODE[String(event?.code || '')];
  if (byCode) return byCode;
  const key = String(event?.key || '').toLowerCase();
  return MOVE_HOTKEY_ACTIONS[key] || null;
}

function onGlobalKeydown(event) {
  const tag = String(event?.target?.tagName || '').toLowerCase();
  if (tag === 'input' || tag === 'textarea' || tag === 'select') return;
  if (event?.target?.isContentEditable) return;
  const key = String(event?.key || '').toLowerCase();
  if (key === 'p') {
    passChordPPressed.value = true;
    return;
  }
  const digit = getNumericKeyValue(event);
  if (digit !== null) {
    if (passChordPPressed.value) {
      const passHandled = Boolean(controlsRef.value?.applyPointerPassHotkey?.(digit));
      if (passHandled) {
        event.preventDefault();
      }
      return;
    }

    const selectHandled = Boolean(controlsRef.value?.applyNumericHotkey?.(digit));
    if (selectHandled) {
      event.preventDefault();
    }
    return;
  }

  const moveAction = getMoveActionFromEvent(event);
  if (moveAction) {
    const handled = Boolean(controlsRef.value?.applyActionHotkey?.(moveAction));
    if (handled) {
      event.preventDefault();
    }
    return;
  }

  if (key === 't') {
    if (hasGame.value && !controlsDisabled.value) {
      controlsRef.value?.submitActions?.();
      event.preventDefault();
    }
    return;
  }

  if (key === 'n') {
    if (hasGame.value && !actionLoading.value && !transitionRunning.value) {
      onNewGame();
      event.preventDefault();
      return;
    }
    if (!hasGame.value && canStartGame.value) {
      onStartGame();
      event.preventDefault();
    }
    return;
  }

  if (key === 'g') {
    if (isDevBuild && hasGame.value && !isGameOver.value) {
      forceGameOverPreview.value = !forceGameOverPreview.value;
      event.preventDefault();
    }
    return;
  }
}

function onGlobalKeyup(event) {
  const key = String(event?.key || '').toLowerCase();
  if (key === 'p') {
    passChordPPressed.value = false;
  }
}

function onWindowBlur() {
  passChordPPressed.value = false;
}

onMounted(() => {
  window.addEventListener('keydown', onGlobalKeydown);
  window.addEventListener('keyup', onGlobalKeyup);
  window.addEventListener('blur', onWindowBlur);
  loadOptions();
});

onBeforeUnmount(() => {
  window.removeEventListener('keydown', onGlobalKeydown);
  window.removeEventListener('keyup', onGlobalKeyup);
  window.removeEventListener('blur', onWindowBlur);
});
</script>

<template>
  <div class="playable-app">
    <header class="playable-header">
      <div class="header-brand">
        <img class="header-logo" :src="basketworldLogo" alt="BasketWorld logo" />
        <h1>Basketworld</h1>
        <p>Human vs AI with alternating possessions and live score tracking.</p>
      </div>
    </header>

    <section class="setup-panel">
      <div class="setup-settings-row">
        <div class="setup-item">
          <label for="players-select">Players Per Side</label>
          <select id="players-select" v-model.number="selectedPlayersPerSide" :disabled="setupLocked">
            <option v-for="players in playerChoices" :key="`players-${players}`" :value="players">
              {{ players }}v{{ players }}
            </option>
          </select>
        </div>

        <div class="setup-item">
          <label for="difficulty-select">Difficulty</label>
          <select id="difficulty-select" v-model="selectedDifficulty" :disabled="setupLocked">
            <option
              v-for="entry in difficultyEntries"
              :key="`difficulty-${entry.value}`"
              :value="entry.value"
              :disabled="!entry.available"
            >
              {{ `${entry.value}${entry.available ? '' : ' (unavailable)'}` }}
            </option>
          </select>
        </div>

        <div class="setup-item">
          <label for="period-mode-select">Game Format</label>
          <select id="period-mode-select" v-model="selectedPeriodMode" :disabled="setupLocked">
            <option v-for="mode in periodModeChoices" :key="`period-mode-${mode}`" :value="mode">
              {{ periodModeLabelMap[mode] || mode }}
            </option>
          </select>
        </div>

        <div class="setup-item">
          <label for="period-length-select">Length Per Segment</label>
          <select id="period-length-select" v-model.number="selectedPeriodLengthMinutes" :disabled="setupLocked">
            <option v-for="minutes in periodLengthChoices" :key="`period-length-${minutes}`" :value="minutes">
              {{ minutes }} min
            </option>
          </select>
        </div>
      </div>

      <div class="setup-actions">
        <button :disabled="!canStartGame" @click="onStartGame">
          {{ actionLoading ? 'Loading...' : 'Start Game' }}
        </button>
        <button :disabled="!hasGame || actionLoading || transitionRunning" @click="onNewGame">New Game</button>
        <button :disabled="!hasGame || actionLoading || transitionRunning" @click="onResetSetup">Reset</button>
        <button
          v-if="isDevBuild && hasGame && !isGameOver"
          :disabled="actionLoading || transitionRunning"
          @click="forceGameOverPreview = !forceGameOverPreview"
        >
          {{ forceGameOverPreview ? 'Hide End Banner' : 'Preview End Banner' }}
        </button>
        <button
          v-if="isDevBuild && hasGame"
          :disabled="actionLoading || transitionRunning"
          @click="toggleViolationOverlayPreview('defensive')"
        >
          {{ violationOverlayPreview === 'defensive' ? 'Hide Def Violation' : 'Demo Def Violation' }}
        </button>
        <button
          v-if="isDevBuild && hasGame"
          :disabled="actionLoading || transitionRunning"
          @click="toggleViolationOverlayPreview('offensive')"
        >
          {{ violationOverlayPreview === 'offensive' ? 'Hide Off Violation' : 'Demo Off Violation' }}
        </button>
      </div>

      <div class="status-row">
        <span v-if="selectedModeLabel">Mode: {{ selectedModeLabel }}</span>
        <span v-if="possessionSummary">{{ possessionSummary }}</span>
        <span v-if="violationOverlayPreview === 'defensive'">Demo: Defensive violation overlay</span>
        <span v-if="violationOverlayPreview === 'offensive'">Demo: Offensive violation overlay</span>
      </div>

      <p v-if="lastPossessionResult" class="possession-result">
        {{ lastPossessionResult.message }}
      </p>
      <p v-if="error" class="error">{{ error }}</p>
    </section>

    <section v-if="hasGame" class="playable-layout">
      <div class="playable-scoreboard" aria-label="Scoreboard">
        <div class="score-side score-side-you">
          <div class="score-lane-meter score-lane-meter-left" :title="userLaneMeter.title">
            <span class="score-lane-role">{{ userLaneMeter.role }}</span>
            <span
              v-for="light in userLaneMeter.lights"
              :key="light.key"
              class="score-lane-light"
              :class="{ lit: light.lit, violation: userLaneMeter.violation && light.lit }"
            />
          </div>
          <div class="score-side-main">
            <span class="score-team">YOU</span>
            <span class="score-digits">{{ formattedUserScore }}</span>
          </div>
        </div>
        <div class="score-center">
          <span class="score-period-label">{{ gameClockSegmentLabel }}</span>
          <span class="clock-digits">{{ gameClockDisplay }}</span>
        </div>
        <div class="score-side score-side-ai">
          <div class="score-side-main">
            <span class="score-team">AI</span>
            <span class="score-digits">{{ formattedAiScore }}</span>
          </div>
          <div class="score-lane-meter score-lane-meter-right" :title="aiLaneMeter.title">
            <span class="score-lane-role">{{ aiLaneMeter.role }}</span>
            <span
              v-for="light in aiLaneMeter.lights"
              :key="light.key"
              class="score-lane-light"
              :class="{ lit: light.lit, violation: aiLaneMeter.violation && light.lit }"
            />
          </div>
        </div>
      </div>
      <div class="left-shell">
        <div class="board-shell">
          <div class="board-stage">
            <GameBoard
              :game-history="gameHistory"
              :active-player-id="activePlayerId"
              :player-display-names="playerDisplayNames"
              :player-jersey-numbers="playerJerseyNumbers"
              :forced-episode-outcome="forcedEpisodeOutcomePreview"
              :selected-actions="boardSelections"
              :is-shot-clock-updating="false"
              :allow-shot-clock-adjustment="false"
              :disable-backend-value-fetches="true"
              :allow-position-drag="false"
              :minimal-chrome="true"
              @update:activePlayerId="activePlayerId = $event"
            />
            <div v-if="showGameOverOverlay" class="playable-gameover-overlay" role="status" aria-live="polite">
              <div class="playable-gameover-banner" :class="`winner-${gameOverWinnerKey}`">
                <p class="gameover-banner-main">{{ gameOverBannerText }}</p>
                <p class="gameover-banner-sub">Final score · Press N for New Game</p>
              </div>
            </div>
          </div>
        </div>
        <div class="environment-shell">
          <PlayableEnvironmentInfo
            :game-state="gameState"
            :player-display-names="playerDisplayNames"
            :player-jersey-numbers="playerJerseyNumbers"
          />
        </div>
      </div>

      <div class="sidebar-shell">
        <div class="controls-shell">
          <PlayableControls
            ref="controlsRef"
            :game-state="gameState"
            :active-player-id="activePlayerId"
            :user-player-ids="userPlayerIds"
            :player-display-names="playerDisplayNames"
            :shortcuts="playableKeyboardShortcuts"
            :disabled="controlsDisabled"
            @update:activePlayerId="activePlayerId = $event"
            @actions-submitted="onActionsSubmitted"
            @selections-changed="onSelectionsChanged"
          />
        </div>

        <div class="stats-shell">
          <PlayableStatsPanel
            :game-state="gameState"
            :user-player-ids="userPlayerIds"
            :score="score"
            :stats="playableStats"
            :play-by-play="playablePlayByPlay"
            :player-display-names="playerDisplayNames"
            :player-jersey-numbers="playerJerseyNumbers"
          />
        </div>
      </div>

    </section>

    <section v-else class="empty-state">
      <p>Choose a players-per-side and available difficulty, then start a game.</p>
    </section>
  </div>
</template>

<style scoped>
.playable-app {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.playable-header {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 1rem;
  flex-wrap: wrap;
}

.playable-header h1 {
  margin: 0;
  font-size: 2rem;
}

.playable-header p {
  margin: 0.35rem 0 0;
  color: var(--app-text-muted);
}

.header-brand {
  display: grid;
  grid-template-columns: auto 1fr;
  column-gap: 0.75rem;
  align-items: center;
}

.header-logo {
  width: 96px;
  height: 96px;
  border-radius: 10px;
  object-fit: cover;
  grid-row: 1 / span 2;
  box-shadow: 0 6px 16px rgba(2, 6, 23, 0.4);
}

.header-brand h1 {
  font-family: 'Roboto', sans-serif;
  font-size: 2rem;
  font-weight: bold;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--app-accent);
  text-shadow: 0 0 3px var(--app-accent), 0 0 6px var(--app-accent);
  grid-column: 2;
}

.header-brand p {
  grid-column: 2;
}

.setup-panel {
  background: var(--app-panel);
  border: 1px solid var(--app-panel-border);
  border-radius: 20px;
  padding: 0.95rem 1rem;
  display: flex;
  flex-direction: column;
  gap: 0.7rem;
}

.setup-settings-row {
  display: grid;
  grid-template-columns: repeat(4, max-content);
  gap: 0.65rem 0.9rem;
  align-items: end;
  justify-content: start;
}

.setup-item {
  display: flex;
  flex-direction: column;
  gap: 0.32rem;
  min-width: 0;
}

.setup-item label,
.setup-item span {
  font-size: 0.8rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--app-text-muted);
}

#players-select,
#difficulty-select,
#period-mode-select,
#period-length-select {
  min-width: 150px;
}

.setup-actions {
  display: flex;
  gap: 0.5rem;
  flex-wrap: wrap;
}

.status-row {
  display: flex;
  flex-wrap: wrap;
  gap: 0.7rem;
  font-size: 0.82rem;
  color: var(--app-text-muted);
}

.possession-result {
  color: var(--app-warning);
}

.error {
  color: #fca5a5;
}

.playable-layout {
  display: grid;
  grid-template-columns: minmax(0, 1024px) minmax(375px, 450px);
  justify-content: center;
  gap: 1rem;
  align-items: start;
  padding: 1rem;
  border-radius: 20px;
  border: 1px solid var(--app-panel-border);
  background: var(--app-panel);
}

.board-shell,
.left-shell,
.sidebar-shell,
.controls-shell,
.stats-shell,
.environment-shell,
.empty-state {
  background: transparent;
  border: none;
  border-radius: 0;
  padding: 0;
}

.board-shell {
  display: flex;
  flex-direction: column;
}

.left-shell {
  grid-column: 1;
  display: flex;
  flex-direction: column;
  gap: 0.85rem;
  align-self: start;
  width: min(100%, 1024px);
}

.board-stage {
  position: relative;
}

.playable-scoreboard {
  grid-column: 1 / -1;
  justify-self: center;
  display: inline-flex;
  align-items: stretch;
  gap: 0.8rem;
  min-width: 380px;
  padding: 0.48rem 1rem;
  border-radius: 8px;
  border: 1px solid #333;
  background: rgba(26, 26, 26, 0.96);
  box-shadow: 0 10px 20px rgba(2, 6, 23, 0.55);
  pointer-events: none;
}

.playable-gameover-overlay {
  position: absolute;
  inset: 0;
  z-index: 14;
  display: flex;
  align-items: center;
  justify-content: center;
  pointer-events: none;
  background: radial-gradient(circle at center, rgba(2, 6, 23, 0.12) 0%, rgba(2, 6, 23, 0.6) 68%);
}

.playable-gameover-banner {
  min-width: min(88%, 560px);
  padding: 0.95rem 1.35rem;
  border-radius: 12px;
  border: 1px solid rgba(255, 77, 77, 0.68);
  background: rgba(10, 14, 26, 0.9);
  box-shadow: 0 0 18px rgba(255, 77, 77, 0.22), 0 16px 26px rgba(2, 6, 23, 0.6);
  text-align: center;
}

.playable-gameover-banner.winner-user {
  border-color: rgba(56, 189, 248, 0.78);
  box-shadow: 0 0 18px rgba(56, 189, 248, 0.24), 0 16px 26px rgba(2, 6, 23, 0.6);
}

.playable-gameover-banner.winner-tie {
  border-color: rgba(148, 163, 184, 0.72);
  box-shadow: 0 0 14px rgba(148, 163, 184, 0.24), 0 16px 26px rgba(2, 6, 23, 0.6);
}

.gameover-banner-main {
  margin: 0;
  font-family: 'DSEG7 Classic', sans-serif;
  font-size: clamp(1.55rem, 4.2vw, 2.45rem);
  line-height: 1.08;
  color: #ff4d4d;
  text-shadow: 0 0 6px rgba(255, 77, 77, 0.95), 0 0 14px rgba(255, 77, 77, 0.6);
  letter-spacing: 0.03em;
}

.playable-gameover-banner.winner-user .gameover-banner-main {
  color: #38bdf8;
  text-shadow: 0 0 6px rgba(56, 189, 248, 0.92), 0 0 14px rgba(56, 189, 248, 0.58);
}

.playable-gameover-banner.winner-tie .gameover-banner-main {
  color: #cbd5e1;
  text-shadow: 0 0 6px rgba(203, 213, 225, 0.82), 0 0 12px rgba(148, 163, 184, 0.45);
}

.gameover-banner-sub {
  margin: 0.48rem 0 0;
  font-size: 0.72rem;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: rgba(226, 232, 240, 0.86);
}

.score-side {
  display: inline-flex;
  flex-direction: row;
  align-items: center;
  justify-content: center;
  gap: 0.52rem;
  min-width: 92px;
  padding: 0.35rem 0.6rem;
  border-radius: 6px;
  border: 1px solid rgba(148, 163, 184, 0.28);
}

.score-side-main {
  display: inline-flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 0.28rem;
}

.score-side-you {
  background: #007bff;
  border-color: #ffffff;
}

.score-side-ai {
  background: #dc3545;
  border-color: #ffffff;
}

.score-team {
  font-size: 0.72rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: rgba(248, 250, 252, 0.95);
}

.score-lane-meter {
  display: inline-flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 0.43rem;
}

.score-lane-role {
  font-size: 0.72rem;
  font-weight: 800;
  line-height: 1;
  color: rgba(248, 250, 252, 0.95);
  letter-spacing: 0.08em;
}

.score-lane-light {
  width: 0.82rem;
  height: 0.82rem;
  border-radius: 999px;
  border: 1px solid rgba(248, 250, 252, 0.72);
  background: rgba(15, 23, 42, 0.35);
}

.score-lane-light.lit {
  background: #ffffff;
  border-color: rgba(255, 255, 255, 0.98);
  box-shadow: 0 0 6px rgba(255, 255, 255, 0.95), 0 0 12px rgba(255, 255, 255, 0.55);
}

.score-lane-light.violation {
  box-shadow: 0 0 7px rgba(248, 250, 252, 0.9);
}

.clock-digits {
  font-family: 'DSEG7 Classic', sans-serif;
  color: #ff4d4d;
  text-shadow: 0 0 5px #ff4d4d, 0 0 10px #ff4d4d;
}

.score-digits {
  font-family: 'DSEG7 Classic', sans-serif;
  color: #f8fafc;
  text-shadow: 0 0 3px rgba(248, 250, 252, 0.65), 0 0 8px rgba(248, 250, 252, 0.2);
  font-size: 3.45rem;
  line-height: 1;
}

.score-center {
  display: inline-flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-width: 132px;
  padding: 0 0.5rem;
  border-left: 1px solid rgba(148, 163, 184, 0.25);
  border-right: 1px solid rgba(148, 163, 184, 0.25);
}

.score-period-label {
  font-size: 0.68rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: rgba(226, 232, 240, 0.78);
  margin-bottom: 0.25rem;
}

.clock-digits {
  font-size: 3.05rem;
  line-height: 1;
}

.controls-shell {
  display: flex;
  flex-direction: column;
  gap: 0.8rem;
}

.sidebar-shell {
  grid-column: 2;
  padding-left: 1rem;
  border-left: 1px solid rgba(148, 163, 184, 0.3);
  display: flex;
  flex-direction: column;
  gap: 0.8rem;
  align-self: start;
}

.stats-shell {
  display: flex;
  flex-direction: column;
  gap: 0.8rem;
}

.empty-state p {
  margin: 0;
}

@media (max-width: 1080px) {
  .setup-settings-row {
    grid-template-columns: repeat(2, max-content);
    align-items: end;
    justify-content: start;
  }

  .playable-layout {
    grid-template-columns: 1fr;
  }

  .left-shell {
    display: contents;
  }

  .board-shell {
    order: 1;
  }

  .sidebar-shell {
    order: 2;
    grid-column: auto;
    padding-left: 0;
    padding-top: 1rem;
    border-left: none;
    border-top: 1px solid rgba(148, 163, 184, 0.3);
  }

  .environment-shell {
    order: 3;
  }

  .playable-scoreboard {
    grid-column: auto;
    justify-self: center;
    min-width: 300px;
    gap: 0.7rem;
    padding: 0.4rem 0.8rem;
  }

  .score-digits {
    font-size: 2rem;
  }

  .clock-digits {
    font-size: 1.6rem;
  }
}

@media (max-width: 720px) {
  .setup-settings-row {
    grid-template-columns: 1fr 1fr;
  }
}
</style>
