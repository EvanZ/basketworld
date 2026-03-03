<script setup>
import { computed, ref, watch } from 'vue';
import HexagonControlPad from './HexagonControlPad.vue';

const props = defineProps({
  gameState: {
    type: Object,
    default: null,
  },
  activePlayerId: {
    type: Number,
    default: null,
  },
  userPlayerIds: {
    type: Array,
    default: () => [],
  },
  disabled: {
    type: Boolean,
    default: false,
  },
  score: {
    type: Object,
    default: () => ({ user: 0, ai: 0 }),
  },
  stats: {
    type: Object,
    default: null,
  },
});

const emit = defineEmits([
  'update:activePlayerId',
  'actions-submitted',
  'selections-changed',
]);

const selectedActions = ref({});

const ACTION_NAMES = [
  'NOOP',
  'MOVE_E',
  'MOVE_NE',
  'MOVE_NW',
  'MOVE_W',
  'MOVE_SW',
  'MOVE_SE',
  'SHOOT',
  'PASS_E',
  'PASS_NE',
  'PASS_NW',
  'PASS_W',
  'PASS_SW',
  'PASS_SE',
];

const passMode = computed(() => String(props.gameState?.pass_mode || 'directional').toLowerCase());
const isPointerPassMode = computed(() => passMode.value === 'pointer_targeted');

function makeEmptyShotLine() {
  return { attempts: 0, made: 0 };
}

function makeEmptyTeamStats() {
  return {
    shots: {
      total: makeEmptyShotLine(),
      twoPt: makeEmptyShotLine(),
      threePt: makeEmptyShotLine(),
      dunk: makeEmptyShotLine(),
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

function makeEmptyPlayerStats() {
  return {
    shots: {
      total: makeEmptyShotLine(),
      twoPt: makeEmptyShotLine(),
      threePt: makeEmptyShotLine(),
      dunk: makeEmptyShotLine(),
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

function makeEmptyStats() {
  return {
    turns: 0,
    possessions: 0,
    userOffensiveTurns: 0,
    userDefensiveTurns: 0,
    byTeam: {
      user: makeEmptyTeamStats(),
      ai: makeEmptyTeamStats(),
    },
    turnoverReasons: {},
  };
}

function asCount(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? Math.max(0, parsed) : 0;
}

function fgPercent(made, attempts) {
  const mk = asCount(made);
  const att = asCount(attempts);
  if (att <= 0) return '0.0';
  return ((mk / att) * 100).toFixed(1);
}

function formatShotCell(line) {
  const made = asCount(line?.made);
  const attempts = asCount(line?.attempts);
  return `${made}-${attempts} (${fgPercent(made, attempts)}%)`;
}

const statsSafe = computed(() => {
  if (props.stats && typeof props.stats === 'object') return props.stats;
  return makeEmptyStats();
});

const userPlayerIdsForStats = computed(() => {
  const ids = Array.isArray(props.userPlayerIds) ? props.userPlayerIds : [];
  return ids.map((id) => Number(id)).filter((id) => Number.isFinite(id)).sort((a, b) => a - b);
});

const aiPlayerIdsForStats = computed(() => {
  const ids = props.gameState?.playable_ai_ids;
  if (!Array.isArray(ids)) return [];
  return ids.map((id) => Number(id)).filter((id) => Number.isFinite(id)).sort((a, b) => a - b);
});

const boxScoreRows = computed(() => {
  const byTeam = statsSafe.value?.byTeam || {};
  const getPlayerLine = (teamKey, playerId) => {
    const team = byTeam[teamKey] || {};
    const map = team.perPlayer && typeof team.perPlayer === 'object' ? team.perPlayer : {};
    return map[String(playerId)] || map[playerId] || makeEmptyPlayerStats();
  };

  const buildRows = (teamKey, playerIds) => {
    const line = byTeam[teamKey] || makeEmptyTeamStats();
    const shots = line.shots || {};
    const rows = playerIds.map((pid) => {
      const playerLine = getPlayerLine(teamKey, pid);
      const playerShots = playerLine.shots || {};
      return {
        key: `${teamKey}-player-${pid}`,
        player: `${pid}`,
        isTotal: false,
        points: asCount(playerLine.points),
        fg: formatShotCell(playerShots.total),
        twoPt: formatShotCell(playerShots.twoPt),
        threePt: formatShotCell(playerShots.threePt),
        dunk: formatShotCell(playerShots.dunk),
        assists: asCount(playerLine.assists),
        turnovers: asCount(playerLine.turnovers),
      };
    });

    rows.push({
      key: `${teamKey}-total`,
      player: teamKey === 'user' ? 'You' : 'AI',
      isTotal: true,
      points: asCount(props.score?.[teamKey]),
      fg: formatShotCell(shots.total),
      twoPt: formatShotCell(shots.twoPt),
      threePt: formatShotCell(shots.threePt),
      dunk: formatShotCell(shots.dunk),
      assists: asCount(line.assists),
      turnovers: asCount(line.turnovers),
    });

    return rows;
  };

  return [
    ...buildRows('user', userPlayerIdsForStats.value),
    ...buildRows('ai', aiPlayerIdsForStats.value),
  ];
});

const orderedUserPlayerIds = computed(() => {
  const ids = Array.isArray(props.userPlayerIds) ? props.userPlayerIds : [];
  return ids
    .map((id) => Number(id))
    .filter((id) => Number.isFinite(id))
    .sort((a, b) => a - b);
});

const activePlayer = computed(() => {
  const active = Number(props.activePlayerId);
  if (Number.isFinite(active) && orderedUserPlayerIds.value.includes(active)) {
    return active;
  }
  return orderedUserPlayerIds.value.length > 0 ? orderedUserPlayerIds.value[0] : null;
});

watch(
  activePlayer,
  (nextActive) => {
    if (nextActive === null || nextActive === undefined) return;
    if (Number(props.activePlayerId) !== Number(nextActive)) {
      emit('update:activePlayerId', Number(nextActive));
    }
  },
  { immediate: true },
);

function getLegalActions(playerId) {
  const mask = props.gameState?.action_mask?.[Number(playerId)];
  if (!Array.isArray(mask)) return [];
  const legal = [];
  for (let idx = 0; idx < mask.length && idx < ACTION_NAMES.length; idx += 1) {
    if (Number(mask[idx]) === 1) {
      legal.push(ACTION_NAMES[idx]);
    }
  }
  return legal;
}

function getControlPadLegalActions(playerId) {
  const legal = getLegalActions(playerId);
  if (!isPointerPassMode.value) return legal;
  return legal.filter((action) => !String(action).startsWith('PASS_'));
}

function getPointerPassTeammatesForPlayer(playerId) {
  const gs = props.gameState;
  if (!gs || playerId === null || playerId === undefined) return [];
  const pid = Number(playerId);
  const offense = Array.isArray(gs.offense_ids) ? gs.offense_ids : [];
  const defense = Array.isArray(gs.defense_ids) ? gs.defense_ids : [];
  const team = offense.includes(pid) ? offense : defense.includes(pid) ? defense : [];
  return team
    .map((id) => Number(id))
    .filter((id) => Number.isFinite(id) && id !== pid)
    .sort((a, b) => a - b)
    .slice(0, 6);
}

function canPlayerSelectPointerPass(playerId) {
  if (!isPointerPassMode.value) return false;
  const pid = Number(playerId);
  if (!Number.isFinite(pid)) return false;
  if (Number(props.gameState?.ball_holder) !== pid) return false;
  const legal = getLegalActions(pid);
  return legal.some((action) => String(action).startsWith('PASS_'));
}

const activePassTargets = computed(() => {
  if (activePlayer.value === null || activePlayer.value === undefined) return [];
  return getPointerPassTeammatesForPlayer(activePlayer.value);
});

const activeCanSelectPointerPass = computed(() => {
  if (activePlayer.value === null || activePlayer.value === undefined) return false;
  return canPlayerSelectPointerPass(activePlayer.value);
});

function isPointerPassButtonSelected(targetId) {
  if (activePlayer.value === null || activePlayer.value === undefined) return false;
  return selectedActions.value?.[Number(activePlayer.value)] === `PASS->${Number(targetId)}`;
}

function handlePointerPassTargetSelected(targetId) {
  if (props.disabled) return;
  if (!activeCanSelectPointerPass.value) return;
  const pid = Number(activePlayer.value);
  const tid = Number(targetId);
  if (!Number.isFinite(pid) || !Number.isFinite(tid)) return;

  if (selectedActions.value?.[pid] === `PASS->${tid}`) {
    const next = { ...selectedActions.value };
    delete next[pid];
    selectedActions.value = next;
    return;
  }

  selectedActions.value = {
    ...selectedActions.value,
    [pid]: `PASS->${tid}`,
  };
}

function sanitizeSelections() {
  const next = {};
  for (const pid of orderedUserPlayerIds.value) {
    const action = selectedActions.value?.[pid];
    if (typeof action !== 'string') continue;
    if (isPointerPassMode.value && action.startsWith('PASS->')) {
      if (!canPlayerSelectPointerPass(pid)) continue;
      const parsedTarget = Number(String(action).replace('PASS->', ''));
      const legalTargets = getPointerPassTeammatesForPlayer(pid);
      if (Number.isFinite(parsedTarget) && legalTargets.includes(parsedTarget)) {
        next[pid] = `PASS->${parsedTarget}`;
      }
      continue;
    }
    const legal = getLegalActions(pid);
    if (legal.includes(action)) {
      next[pid] = action;
    }
  }
  selectedActions.value = next;
}

watch(
  () => props.gameState,
  () => {
    sanitizeSelections();
  },
  { deep: true },
);

watch(
  selectedActions,
  (nextSelections) => {
    emit('selections-changed', { ...(nextSelections || {}) });
  },
  { deep: true },
);

function handleActionSelected(actionName) {
  if (props.disabled) return;
  const pid = activePlayer.value;
  if (pid === null || pid === undefined) return;
  if (isPointerPassMode.value && String(actionName).startsWith('PASS_')) return;
  const legal = getLegalActions(pid);
  if (!legal.includes(actionName)) return;
  selectedActions.value = {
    ...selectedActions.value,
    [pid]: actionName,
  };
}

const selectedActionRows = computed(() => {
  return orderedUserPlayerIds.value.map((pid) => ({
    playerId: pid,
    action: selectedActions.value?.[pid] || 'NOOP',
  }));
});

const canSubmit = computed(() => {
  if (props.disabled) return false;
  if (!props.gameState) return false;
  if (props.gameState.done) return false;
  return orderedUserPlayerIds.value.length > 0;
});

function submitActions() {
  if (!canSubmit.value) return;

  const payload = {};
  for (const pid of orderedUserPlayerIds.value) {
    const action = String(selectedActions.value?.[pid] || 'NOOP');
    if (isPointerPassMode.value && action.startsWith('PASS->')) {
      const target = Number(action.replace('PASS->', ''));
      if (Number.isFinite(target)) {
        payload[String(pid)] = {
          type: 'PASS',
          target,
        };
        continue;
      }
    }
    payload[String(pid)] = action;
  }

  emit('actions-submitted', payload);
  selectedActions.value = {};
}

function applyNumericHotkey(digit) {
  const value = Number(digit);
  if (!Number.isFinite(value)) return false;
  if (props.disabled) return false;

  if (orderedUserPlayerIds.value.includes(value)) {
    emit('update:activePlayerId', value);
    return true;
  }

  if (isPointerPassMode.value && activeCanSelectPointerPass.value && activePassTargets.value.includes(value)) {
    handlePointerPassTargetSelected(value);
    return true;
  }

  return false;
}

defineExpose({
  submitActions,
  applyNumericHotkey,
});
</script>

<template>
  <div class="playable-controls">
    <div class="controls-pane">
      <h3>Turn Controls</h3>

      <div class="player-tabs">
        <button
          v-for="pid in orderedUserPlayerIds"
          :key="`user-tab-${pid}`"
          class="player-tab"
          :class="{ active: Number(activePlayerId) === Number(pid) }"
          :disabled="disabled"
          @click="emit('update:activePlayerId', Number(pid))"
        >
          Player {{ pid }}
        </button>
      </div>

      <div v-if="activePlayer !== null" class="control-pad-wrap">
        <HexagonControlPad
          :legal-actions="getControlPadLegalActions(activePlayer)"
          :selected-action="selectedActions[activePlayer]?.startsWith('PASS->') ? '' : (selectedActions[activePlayer] || '')"
          :pass-probabilities="null"
          :action-values="null"
          :value-range="{ min: 0, max: 0 }"
          :is-defense="false"
          @action-selected="handleActionSelected"
        />
      </div>

      <div v-if="isPointerPassMode && activePlayer !== null" class="pointer-pass-wrap">
        <p class="pointer-pass-label">Pass Target</p>
        <div class="pointer-pass-targets">
          <button
            v-for="targetId in activePassTargets"
            :key="`pointer-target-${activePlayer}-${targetId}`"
            class="pointer-pass-btn"
            :class="{ selected: isPointerPassButtonSelected(targetId) }"
            :disabled="disabled || !activeCanSelectPointerPass"
            @click="handlePointerPassTargetSelected(targetId)"
          >
            Player {{ targetId }}
          </button>
        </div>
        <p v-if="!activeCanSelectPointerPass" class="pointer-pass-help">
          Select the ball handler to choose a pass target.
        </p>
      </div>

      <div class="selected-grid">
        <div
          v-for="row in selectedActionRows"
          :key="`row-${row.playerId}`"
          class="selected-row"
        >
          <span>Player {{ row.playerId }}</span>
          <strong>{{ row.action }}</strong>
        </div>
      </div>

      <button class="submit-btn" :disabled="!canSubmit" @click="submitActions">
        Submit Turn
      </button>

      <h3>Box Score</h3>
      <div class="table-scroll">
        <table class="box-score-table">
          <thead>
            <tr>
              <th>Player</th>
              <th>PTS</th>
              <th>FG</th>
              <th>2PT</th>
              <th>3PT</th>
              <th>Dunk</th>
              <th>AST</th>
              <th>TOV</th>
            </tr>
          </thead>
          <tbody>
            <tr
              v-for="row in boxScoreRows"
              :key="row.key"
              :class="{ 'total-row': row.isTotal }"
            >
              <td class="player-cell">{{ row.player }}</td>
              <td>{{ row.points }}</td>
              <td>{{ row.fg }}</td>
              <td>{{ row.twoPt }}</td>
              <td>{{ row.threePt }}</td>
              <td>{{ row.dunk }}</td>
              <td>{{ row.assists }}</td>
              <td>{{ row.turnovers }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>
</template>

<style scoped>
.playable-controls {
  padding: 0.2rem 0.4rem;
  display: flex;
  flex-direction: column;
  gap: 0.9rem;
}

h3 {
  margin: 0;
  color: var(--app-accent);
  letter-spacing: 0.08em;
  text-transform: uppercase;
  font-size: 0.95rem;
}

.controls-pane {
  display: flex;
  flex-direction: column;
  gap: 0.85rem;
}

.player-tabs {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
}

.player-tab {
  border-radius: 999px;
  border: 1px solid rgba(148, 163, 184, 0.4);
  background: transparent;
  color: var(--app-text);
  padding: 0.4rem 0.85rem;
  font-size: 0.85rem;
}

.player-tab.active {
  border-color: var(--app-accent-strong);
  color: var(--app-accent);
}

.control-pad-wrap {
  display: flex;
  justify-content: center;
}

.pointer-pass-wrap {
  display: flex;
  flex-direction: column;
  gap: 0.4rem;
}

.pointer-pass-label {
  font-size: 0.76rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--app-text-muted);
}

.pointer-pass-targets {
  display: flex;
  flex-wrap: wrap;
  gap: 0.45rem;
}

.pointer-pass-btn {
  border-radius: 999px;
  border: 1px solid rgba(148, 163, 184, 0.4);
  background: transparent;
  color: var(--app-text);
  padding: 0.32rem 0.72rem;
  font-size: 0.76rem;
  text-transform: none;
  letter-spacing: 0.02em;
}

.pointer-pass-btn.selected {
  border-color: var(--app-accent-strong);
  color: var(--app-accent);
}

.pointer-pass-help {
  font-size: 0.76rem;
  color: var(--app-text-muted);
}

.selected-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
  gap: 0.4rem;
}

.selected-row {
  display: flex;
  justify-content: space-between;
  gap: 0.5rem;
  border: 1px solid rgba(148, 163, 184, 0.25);
  border-radius: 10px;
  padding: 0.35rem 0.5rem;
  font-size: 0.8rem;
}

.submit-btn {
  border-radius: 999px;
  border: 1px solid rgba(148, 163, 184, 0.4);
  background: transparent;
  color: var(--app-text);
  padding: 0.55rem 1rem;
  text-transform: uppercase;
  letter-spacing: 0.1em;
}

.submit-btn:hover:not(:disabled) {
  border-color: var(--app-accent-strong);
  color: var(--app-accent);
}

.submit-btn:disabled {
  opacity: 0.45;
  cursor: not-allowed;
}

.box-score-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.74rem;
}

.table-scroll {
  width: 100%;
  overflow-x: auto;
}

.box-score-table th,
.box-score-table td {
  border: 1px solid rgba(148, 163, 184, 0.22);
  padding: 0.34rem 0.4rem;
  text-align: right;
}

.box-score-table th:first-child,
.box-score-table td:first-child {
  text-align: left;
}

.box-score-table th {
  color: var(--app-text-muted);
  background: rgba(15, 23, 42, 0.45);
  font-weight: 600;
}

.box-score-table tbody tr:nth-child(even) {
  background: rgba(15, 23, 42, 0.25);
}

.player-cell {
  letter-spacing: 0.06em;
  font-weight: 500;
}

.box-score-table .total-row td {
  font-weight: 700;
  background: rgba(59, 130, 246, 0.1);
}
</style>
