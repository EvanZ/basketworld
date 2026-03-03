<script setup>
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue';
import GameBoard from './components/GameBoard.vue';
import KeyboardLegend from './components/KeyboardLegend.vue';
import PlayableControls from './components/PlayableControls.vue';
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

const optionsPayload = ref(null);
const selectedPlayersPerSide = ref(1);
const selectedDifficulty = ref('easy');

const gameState = ref(null);
const gameHistory = ref([]);
const activePlayerId = ref(null);
const boardSelections = ref({});
const controlsRef = ref(null);

const score = ref({ user: 0, ai: 0 });
const possession = ref(null);
const sessionConfig = ref(null);
const lastPossessionResult = ref(null);
const transitionRunning = ref(false);
const playableKeyboardShortcuts = [
  {
    key: 'N',
    label: 'New Game',
  },
  {
    key: 'T',
    label: 'Submit Turn',
  },
  {
    key: '0-9',
    label: 'Select Player or Pass Target (by ID, top row or numpad)',
  },
];

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

function resetPlayableStats() {
  playableStats.value = createPlayableStats();
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

const playerChoices = computed(() => {
  const values = optionsPayload.value?.players_per_side;
  if (!Array.isArray(values) || values.length === 0) return [1, 2, 3, 4, 5];
  return values.map((v) => Number(v)).filter((v) => Number.isFinite(v));
});

const difficultyChoices = computed(() => {
  const values = optionsPayload.value?.difficulties;
  if (!Array.isArray(values) || values.length === 0) return ['easy', 'medium', 'hard'];
  return values.map((v) => String(v || '').toLowerCase());
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

const canStartGame = computed(() => {
  return !optionsLoading.value && !actionLoading.value && !transitionRunning.value && Boolean(currentDifficultyConfig.value?.available);
});

const hasGame = computed(() => !!gameState.value);
const controlsDisabled = computed(() => actionLoading.value || transitionRunning.value);

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

const possessionSummary = computed(() => {
  if (!possession.value) return '';
  const offenseTeam = possession.value.offense_team === 'user' ? 'You (Blue)' : 'AI (Red)';
  return `Possession ${possession.value.number}: ${offenseTeam} on offense`;
});

const selectedModeLabel = computed(() => {
  if (!sessionConfig.value) return '';
  return `${sessionConfig.value.players_per_side}v${sessionConfig.value.players_per_side} · ${sessionConfig.value.difficulty}`;
});

const formattedUserScore = computed(() => String(Number(score.value?.user || 0)).padStart(2, '0'));
const formattedAiScore = computed(() => String(Number(score.value?.ai || 0)).padStart(2, '0'));

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
  const state = payload?.state || null;
  gameState.value = state;
  gameHistory.value = state ? [state] : [];

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
    sessionConfig.value = {
      players_per_side: Number(payload.config.players_per_side),
      difficulty: String(payload.config.difficulty || '').toLowerCase(),
    };
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

  try {
    const payload = await startPlayableGame(selectedPlayersPerSide.value, selectedDifficulty.value);
    resetPlayableStats();
    applyStatePayload(payload);
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

  try {
    const payload = await newPlayableGame();
    resetPlayableStats();
    applyStatePayload(payload);
    boardSelections.value = {};
  } catch (err) {
    error.value = err?.message || 'Failed to reset playable game.';
  } finally {
    actionLoading.value = false;
  }
}

async function onActionsSubmitted(actions) {
  if (!hasGame.value || actionLoading.value || transitionRunning.value) return;
  actionLoading.value = true;
  error.value = '';

  try {
    const userIds = toIdSet(userPlayerIds.value);
    const aiIds = toIdSet(gameState.value?.playable_ai_ids);
    const userOnOffense = Boolean(possession.value?.user_on_offense);

    const payload = await stepPlayableGame(actions || {});
    recordPlayableStepStats(payload, { userIds, aiIds, userOnOffense });
    const possessionEnded = Boolean(payload?.possession_ended);
    lastPossessionResult.value = possessionEnded ? payload?.possession_result || null : null;

    if (possessionEnded && payload?.ended_state) {
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

function onGlobalKeydown(event) {
  const tag = String(event?.target?.tagName || '').toLowerCase();
  if (tag === 'input' || tag === 'textarea' || tag === 'select') return;
  if (event?.target?.isContentEditable) return;
  const key = String(event?.key || '').toLowerCase();
  const digit = getNumericKeyValue(event);
  if (digit !== null) {
    const handled = Boolean(controlsRef.value?.applyNumericHotkey?.(digit));
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
}

onMounted(() => {
  window.addEventListener('keydown', onGlobalKeydown);
  loadOptions();
});

onBeforeUnmount(() => {
  window.removeEventListener('keydown', onGlobalKeydown);
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
      <div class="setup-row">
        <label for="players-select">Players Per Side</label>
        <select id="players-select" v-model.number="selectedPlayersPerSide" :disabled="optionsLoading || actionLoading">
          <option v-for="players in playerChoices" :key="`players-${players}`" :value="players">
            {{ players }}v{{ players }}
          </option>
        </select>
      </div>

      <div class="setup-row difficulty-row">
        <span>Difficulty</span>
        <div class="difficulty-buttons">
          <button
            v-for="entry in difficultyEntries"
            :key="`difficulty-${entry.value}`"
            class="difficulty-button"
            :class="{ active: selectedDifficulty === entry.value }"
            :disabled="!entry.available || optionsLoading || actionLoading"
            @click="selectedDifficulty = entry.value"
          >
            {{ entry.value }}<span v-if="!entry.available"> (unavailable)</span>
          </button>
        </div>
      </div>

      <div class="setup-actions">
        <button :disabled="!canStartGame" @click="onStartGame">
          {{ actionLoading ? 'Loading...' : 'Start Game' }}
        </button>
        <button :disabled="!hasGame || actionLoading || transitionRunning" @click="onNewGame">New Game</button>
      </div>

      <div class="status-row">
        <span v-if="selectedModeLabel">Mode: {{ selectedModeLabel }}</span>
        <span v-if="possessionSummary">{{ possessionSummary }}</span>
      </div>

      <p v-if="lastPossessionResult" class="possession-result">
        {{ lastPossessionResult.message }}
      </p>
      <p v-if="error" class="error">{{ error }}</p>
    </section>

    <section v-if="hasGame" class="playable-layout">
      <div class="board-shell">
        <div class="board-stage">
          <GameBoard
            :game-history="gameHistory"
            :active-player-id="activePlayerId"
            :selected-actions="boardSelections"
            :is-shot-clock-updating="false"
            :allow-shot-clock-adjustment="false"
            :disable-backend-value-fetches="true"
            :allow-position-drag="false"
            :minimal-chrome="true"
            @update:activePlayerId="activePlayerId = $event"
          />
          <div class="playable-scoreboard" aria-label="Scoreboard">
            <div class="score-side">
              <span class="score-team">YOU</span>
              <span class="score-digits">{{ formattedUserScore }}</span>
            </div>
            <span class="score-divider">:</span>
            <div class="score-side">
              <span class="score-team">AI</span>
              <span class="score-digits">{{ formattedAiScore }}</span>
            </div>
          </div>
        </div>
        <PlayableEnvironmentInfo :game-state="gameState" />
      </div>

      <div class="controls-shell">
        <PlayableControls
          ref="controlsRef"
          :game-state="gameState"
          :active-player-id="activePlayerId"
          :user-player-ids="userPlayerIds"
          :score="score"
          :stats="playableStats"
          :disabled="controlsDisabled"
          @update:activePlayerId="activePlayerId = $event"
          @actions-submitted="onActionsSubmitted"
          @selections-changed="onSelectionsChanged"
        />
        <KeyboardLegend :shortcuts="playableKeyboardShortcuts" class="playable-kb-legend" />
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
  gap: 0.8rem;
}

.setup-row {
  display: flex;
  align-items: center;
  gap: 0.8rem;
  flex-wrap: wrap;
}

.setup-row label,
.setup-row span {
  font-size: 0.8rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--app-text-muted);
}

#players-select {
  min-width: 120px;
}

.difficulty-row {
  align-items: flex-start;
}

.difficulty-buttons {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
}

.difficulty-button {
  text-transform: none;
  letter-spacing: 0.02em;
  padding: 0.4rem 0.85rem;
  font-size: 0.78rem;
}

.difficulty-button.active {
  border-color: var(--app-accent-strong);
  color: var(--app-accent);
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
  grid-template-columns: minmax(0, 1fr) minmax(300px, 360px);
  gap: 1rem;
  align-items: start;
  padding: 1rem;
  border-radius: 20px;
  border: 1px solid var(--app-panel-border);
  background: var(--app-panel);
}

.board-shell,
.controls-shell,
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

.board-stage {
  position: relative;
}

.playable-scoreboard {
  position: absolute;
  top: 10px;
  left: 50%;
  transform: translateX(-50%);
  z-index: 13;
  display: inline-flex;
  align-items: center;
  gap: 0.95rem;
  min-width: 300px;
  padding: 0.55rem 1.05rem;
  border-radius: 8px;
  border: 1px solid #333;
  background: rgba(26, 26, 26, 0.96);
  box-shadow: 0 10px 20px rgba(2, 6, 23, 0.55);
  pointer-events: none;
}

.score-side {
  display: inline-flex;
  align-items: baseline;
  gap: 0.35rem;
}

.score-team {
  font-size: 0.78rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: rgba(226, 232, 240, 0.85);
}

.score-digits,
.score-divider {
  font-family: 'DSEG7 Classic', sans-serif;
  color: #ff4d4d;
  text-shadow: 0 0 5px #ff4d4d, 0 0 10px #ff4d4d;
}

.score-digits {
  font-size: 3rem;
  line-height: 1;
}

.score-divider {
  font-size: 2.35rem;
  line-height: 1;
}

.controls-shell {
  padding-left: 1rem;
  border-left: 1px solid rgba(148, 163, 184, 0.3);
  display: flex;
  flex-direction: column;
  gap: 0.8rem;
}

.playable-kb-legend {
  margin-top: 0.2rem;
}

.empty-state p {
  margin: 0;
}

@media (max-width: 1080px) {
  .playable-layout {
    grid-template-columns: 1fr;
  }

  .controls-shell {
    padding-left: 0;
    padding-top: 1rem;
    border-left: none;
    border-top: 1px solid rgba(148, 163, 184, 0.3);
  }

  .playable-scoreboard {
    min-width: 240px;
    gap: 0.7rem;
    padding: 0.4rem 0.8rem;
  }

  .score-digits {
    font-size: 2.3rem;
  }

  .score-divider {
    font-size: 2rem;
  }
}
</style>
