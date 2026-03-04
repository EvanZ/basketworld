<script setup>
import { computed, ref, watch } from 'vue';
import HexagonControlPad from './HexagonControlPad.vue';
import KeyboardLegend from './KeyboardLegend.vue';

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
  playByPlay: {
    type: Array,
    default: () => [],
  },
  shortcuts: {
    type: Array,
    default: () => [],
  },
});

const emit = defineEmits([
  'update:activePlayerId',
  'actions-submitted',
  'selections-changed',
]);

const selectedActions = ref({});
const showPolicyHints = ref(false);

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

function formatPolicyPercent(value) {
  const prob = Number(value);
  if (!Number.isFinite(prob) || prob <= 0) return '0.0%';
  return `${(prob * 100).toFixed(1)}%`;
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

const playByPlayRows = computed(() => {
  const rows = Array.isArray(props.playByPlay) ? props.playByPlay : [];
  return rows
    .map((row, idx) => ({
      key: row?.id ?? `pbp-${idx}`,
      period: String(row?.period || '-'),
      clock: String(row?.clock || '--:--'),
      team: String(row?.team || '-'),
      score: String(row?.score || '0-0'),
      event: String(row?.event || ''),
    }))
    .filter((row) => row.event.length > 0);
});

const policyHintRows = computed(() => {
  const gs = props.gameState;
  if (!gs || typeof gs !== 'object') return [];

  const userIdsRaw = Array.isArray(props.userPlayerIds) ? props.userPlayerIds : [];
  const userIds = userIdsRaw
    .map((id) => Number(id))
    .filter((id) => Number.isFinite(id))
    .sort((a, b) => a - b);
  if (userIds.length === 0) return [];

  const policyMap = gs.policy_probabilities && typeof gs.policy_probabilities === 'object'
    ? gs.policy_probabilities
    : {};
  const actionMask = Array.isArray(gs.action_mask) ? gs.action_mask : [];
  const ballHolder = Number(gs?.ball_holder);
  const userOnOffense = Boolean(gs?.playable_user_on_offense);
  const sideLabel = userOnOffense ? 'Offense' : 'Defense';

  const rows = userIds.map((pid) => {
    const probs = policyMap[String(pid)] ?? policyMap[pid];
    const playerMask = Array.isArray(actionMask?.[pid]) ? actionMask[pid] : null;
    const ranked = [];

    if (Array.isArray(probs)) {
      const bound = Math.min(probs.length, ACTION_NAMES.length);
      for (let idx = 0; idx < bound; idx += 1) {
        if (playerMask && Number(playerMask[idx]) !== 1) continue;
        const prob = Number(probs[idx]);
        if (!Number.isFinite(prob) || prob <= 0) continue;
        ranked.push({
          key: `policy-hint-${pid}-${idx}`,
          action: ACTION_NAMES[idx],
          prob,
        });
      }
    }

    ranked.sort((a, b) => b.prob - a.prob);
    return {
      playerId: pid,
      isBallHandler: pid === ballHolder,
      roleLabel: sideLabel,
      hints: ranked.slice(0, 3),
    };
  });

  rows.sort((a, b) => Number(b.isBallHandler) - Number(a.isBallHandler));
  return rows;
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

const activePolicyHintRow = computed(() => {
  const pid = Number(activePlayer.value);
  if (!Number.isFinite(pid)) return null;

  const existing = policyHintRows.value.find((row) => Number(row.playerId) === pid);
  if (existing) return existing;

  const gs = props.gameState || {};
  return {
    playerId: pid,
    isBallHandler: pid === Number(gs?.ball_holder),
    roleLabel: Boolean(gs?.playable_user_on_offense) ? 'Offense' : 'Defense',
    hints: [],
  };
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
  selectActionForActivePlayer(actionName);
}

function selectActionForActivePlayer(actionName) {
  if (props.disabled) return false;
  const pid = activePlayer.value;
  if (pid === null || pid === undefined) return false;
  if (isPointerPassMode.value && String(actionName).startsWith('PASS_')) return false;
  const legal = getLegalActions(pid);
  if (!legal.includes(actionName)) return false;

  const currentlySelected = String(selectedActions.value?.[pid] || '');
  if (currentlySelected === actionName) {
    const next = { ...selectedActions.value };
    delete next[pid];
    selectedActions.value = next;
    return true;
  }

  selectedActions.value = {
    ...selectedActions.value,
    [pid]: actionName,
  };
  return true;
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

function applyPointerPassHotkey(digit) {
  const value = Number(digit);
  if (!Number.isFinite(value)) return false;
  if (props.disabled) return false;
  if (!isPointerPassMode.value) return false;
  if (!activeCanSelectPointerPass.value) return false;
  if (!activePassTargets.value.includes(value)) return false;

  handlePointerPassTargetSelected(value);
  return true;
}

function applyActionHotkey(actionName) {
  if (typeof actionName !== 'string' || actionName.length === 0) return false;
  return selectActionForActivePlayer(actionName);
}

function onShortcutLegendClicked(shortcut) {
  const action = String(shortcut?.action || '');
  if (!action) return;
  applyActionHotkey(action);
}

defineExpose({
  submitActions,
  applyNumericHotkey,
  applyPointerPassHotkey,
  applyActionHotkey,
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

      <div class="ai-hints-wrap">
        <button
          class="ai-hints-toggle"
          :class="{ active: showPolicyHints }"
          @click="showPolicyHints = !showPolicyHints"
        >
          {{ showPolicyHints ? 'Hide Policy Hints' : 'Show Policy Hints' }}
        </button>
      </div>

      <div v-if="activePlayer !== null" class="control-pad-zone">
        <div v-if="showPolicyHints" class="ai-hints-panel">
          <p class="ai-hints-title">Policy Hints</p>
          <p class="ai-hints-player">
            You {{ activePolicyHintRow?.playerId }}
            <span v-if="activePolicyHintRow?.isBallHandler"> (Ball)</span>
          </p>
          <ul v-if="activePolicyHintRow && activePolicyHintRow.hints.length > 0" class="ai-hints-list">
            <li
              v-for="hint in activePolicyHintRow.hints"
              :key="hint.key"
              class="ai-hints-list-item"
              :style="{ '--hint-fill': `${Math.max(0, Math.min(100, Number(hint.prob || 0) * 100)).toFixed(1)}%` }"
            >
              <span class="ai-hints-action">{{ hint.action }}</span>
              <span class="ai-hints-prob">{{ formatPolicyPercent(hint.prob) }}</span>
            </li>
          </ul>
          <div v-else class="ai-hints-empty">No legal actions</div>
          <p class="ai-hints-note">Role: {{ activePolicyHintRow?.roleLabel || 'Offense' }}</p>
        </div>

        <div class="control-pad-wrap">
          <HexagonControlPad
            :legal-actions="getControlPadLegalActions(activePlayer)"
            :selected-action="selectedActions[activePlayer]?.startsWith('PASS->') ? '' : (selectedActions[activePlayer] || '')"
            :pass-probabilities="null"
            :action-values="null"
            :value-range="{ min: 0, max: 0 }"
            :is-defense="false"
            layout-variant="court"
            @action-selected="handleActionSelected"
          />
        </div>
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
      <KeyboardLegend
        :shortcuts="shortcuts"
        class="playable-shortcuts-legend"
        @shortcut-clicked="onShortcutLegendClicked"
      />

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

      <h3>Play-By-Play</h3>
      <div class="table-scroll play-by-play-scroll">
        <table class="play-by-play-table">
          <thead>
            <tr>
              <th>Period</th>
              <th>Clock</th>
              <th>Score</th>
              <th>Team</th>
              <th>Event</th>
            </tr>
          </thead>
          <tbody>
            <template v-if="playByPlayRows.length === 0">
              <tr>
                <td colspan="5" class="play-by-play-empty">No events yet.</td>
              </tr>
            </template>
            <template v-else>
              <tr
                v-for="row in playByPlayRows"
                :key="row.key"
              >
                <td>{{ row.period }}</td>
                <td>{{ row.clock }}</td>
                <td class="play-by-play-score">{{ row.score }}</td>
                <td class="play-by-play-team">{{ row.team }}</td>
                <td class="play-by-play-event">{{ row.event }}</td>
              </tr>
            </template>
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

.control-pad-zone {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.7rem;
}

.control-pad-wrap {
  display: flex;
  justify-content: flex-start;
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

.ai-hints-wrap {
  display: flex;
  align-items: center;
}

.ai-hints-toggle {
  border-radius: 999px;
  border: 1px solid rgba(148, 163, 184, 0.4);
  background: transparent;
  color: var(--app-text);
  padding: 0.35rem 0.78rem;
  font-size: 0.76rem;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}

.ai-hints-toggle:hover {
  border-color: var(--app-accent-strong);
  color: var(--app-accent);
}

.ai-hints-toggle.active {
  border-color: var(--app-accent-strong);
  color: var(--app-accent);
}

.ai-hints-panel {
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
  border: 1px solid rgba(148, 163, 184, 0.24);
  border-radius: 10px;
  padding: 0.42rem 0.48rem;
  background: rgba(15, 23, 42, 0.2);
  width: 180px;
  max-width: 180px;
}

.ai-hints-title {
  margin: 0;
  font-size: 0.66rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--app-text-muted);
}

.ai-hints-player {
  margin: 0;
  font-size: 0.74rem;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--app-text);
}

.ai-hints-list {
  margin: 0;
  padding: 0;
  list-style: none;
  display: flex;
  flex-direction: column;
  gap: 0.3rem;
}

.ai-hints-list-item {
  --hint-fill: 0%;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.45rem;
  position: relative;
  overflow: hidden;
  border: 1px solid rgba(148, 163, 184, 0.2);
  border-radius: 8px;
  padding: 0.2rem 0.34rem;
  background: rgba(15, 23, 42, 0.28);
}

.ai-hints-list-item::before {
  content: '';
  position: absolute;
  inset: 0 auto 0 0;
  width: var(--hint-fill);
  background: linear-gradient(90deg, rgba(56, 189, 248, 0.18), rgba(56, 189, 248, 0.08));
  pointer-events: none;
}

.ai-hints-action {
  font-size: 0.67rem;
  letter-spacing: 0.02em;
  color: var(--app-text);
  white-space: nowrap;
  position: relative;
  z-index: 1;
}

.ai-hints-prob {
  font-size: 0.68rem;
  color: var(--app-accent);
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
  position: relative;
  z-index: 1;
}

.ai-hints-empty {
  font-size: 0.72rem;
  color: var(--app-text-muted);
}

.ai-hints-note {
  margin: 0;
  font-size: 0.64rem;
  color: var(--app-text-muted);
  letter-spacing: 0.05em;
  text-transform: uppercase;
}

@media (max-width: 780px) {
  .control-pad-zone {
    flex-direction: column;
    align-items: stretch;
    gap: 0.55rem;
  }

  .ai-hints-panel {
    width: 100%;
    max-width: none;
  }

  .control-pad-wrap {
    justify-content: center;
  }
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

.playable-shortcuts-legend {
  margin-top: 0.2rem;
}

.box-score-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.74rem;
}

.play-by-play-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.72rem;
}

.table-scroll {
  width: 100%;
  overflow-x: auto;
}

.play-by-play-scroll {
  max-height: 270px;
  overflow-y: auto;
}

.box-score-table th,
.box-score-table td,
.play-by-play-table th,
.play-by-play-table td {
  border: 1px solid rgba(148, 163, 184, 0.22);
  padding: 0.34rem 0.4rem;
  text-align: right;
}

.box-score-table th:first-child,
.box-score-table td:first-child,
.play-by-play-table th:first-child,
.play-by-play-table td:first-child {
  text-align: left;
}

.box-score-table th {
  color: var(--app-text-muted);
  background: rgba(15, 23, 42, 0.45);
  font-weight: 600;
}

.play-by-play-table th {
  color: var(--app-text-muted);
  background: rgba(15, 23, 42, 0.45);
  font-weight: 600;
}

.box-score-table tbody tr:nth-child(even) {
  background: rgba(15, 23, 42, 0.25);
}

.play-by-play-table tbody tr:nth-child(even) {
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

.play-by-play-table th:nth-child(1),
.play-by-play-table td:nth-child(1) {
  min-width: 86px;
}

.play-by-play-table th:nth-child(2),
.play-by-play-table td:nth-child(2) {
  min-width: 62px;
}

.play-by-play-table th:nth-child(3),
.play-by-play-table td:nth-child(3) {
  min-width: 90px;
  text-align: center;
}

.play-by-play-table th:nth-child(4),
.play-by-play-table td:nth-child(4) {
  min-width: 54px;
  text-align: center;
}

.play-by-play-table th:nth-child(5),
.play-by-play-table td:nth-child(5) {
  text-align: left;
}

.play-by-play-team {
  letter-spacing: 0.06em;
  font-weight: 600;
}

.play-by-play-event {
  white-space: normal;
  line-height: 1.3;
}

.play-by-play-score {
  font-variant-numeric: tabular-nums;
  letter-spacing: 0.04em;
}

.play-by-play-empty {
  text-align: center !important;
  color: var(--app-text-muted);
}
</style>
