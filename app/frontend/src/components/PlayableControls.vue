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
  playerDisplayNames: {
    type: Object,
    default: () => ({}),
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

function formatPolicyPercent(value) {
  const prob = Number(value);
  if (!Number.isFinite(prob) || prob <= 0) return '0.0%';
  return `${(prob * 100).toFixed(1)}%`;
}

function getPlayerSurname(playerId) {
  const id = Number(playerId);
  if (!Number.isFinite(id)) return '';
  const map = props.playerDisplayNames && typeof props.playerDisplayNames === 'object'
    ? props.playerDisplayNames
    : {};
  const raw = map[id] ?? map[String(id)];
  return typeof raw === 'string' ? raw.trim() : '';
}

function formatPlayerNameWithId(playerId) {
  const id = Number(playerId);
  if (!Number.isFinite(id)) return 'Unknown';
  const surname = getPlayerSurname(id);
  if (surname) return `${surname} #${id}`;
  return `Player ${id}`;
}

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

const activePassSuccessProbabilities = computed(() => {
  if (!activeCanSelectPointerPass.value) return {};
  const gs = props.gameState;
  if (!gs || typeof gs !== 'object') return {};
  const rawStealMap = gs.pass_steal_probabilities && typeof gs.pass_steal_probabilities === 'object'
    ? gs.pass_steal_probabilities
    : {};
  const result = {};
  for (const targetId of activePassTargets.value) {
    const raw = rawStealMap[targetId] ?? rawStealMap[String(targetId)];
    if (raw === null || raw === undefined) continue;
    const stealNum = Number(raw);
    if (!Number.isFinite(stealNum)) continue;
    const stealFraction = Math.max(0, Math.min(1, stealNum > 1 ? stealNum / 100 : stealNum));
    result[targetId] = Math.max(0, Math.min(1, 1 - stealFraction));
  }
  return result;
});

function getActivePassSuccessProbability(targetId) {
  const tid = Number(targetId);
  if (!Number.isFinite(tid)) return null;
  const raw = activePassSuccessProbabilities.value[tid] ?? activePassSuccessProbabilities.value[String(tid)];
  const prob = Number(raw);
  return Number.isFinite(prob) ? Math.max(0, Math.min(1, prob)) : null;
}

function formatPointerPassPercent(targetId) {
  const prob = getActivePassSuccessProbability(targetId);
  if (prob === null) return '—';
  return `${(prob * 100).toFixed(1)}%`;
}

function pointerPassButtonStyle(targetId) {
  const prob = getActivePassSuccessProbability(targetId);
  const fillPct = prob === null ? 0 : Math.max(0, Math.min(100, prob * 100));
  return {
    '--pass-fill': `${fillPct.toFixed(1)}%`,
  };
}

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
          {{ formatPlayerNameWithId(pid) }}
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
            You {{ formatPlayerNameWithId(activePolicyHintRow?.playerId) }}
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
            :style="pointerPassButtonStyle(targetId)"
            :disabled="disabled || !activeCanSelectPointerPass"
            @click="handlePointerPassTargetSelected(targetId)"
          >
            <span class="pointer-pass-name">{{ formatPlayerNameWithId(targetId) }}</span>
            <span v-if="activeCanSelectPointerPass" class="pointer-pass-prob">{{ formatPointerPassPercent(targetId) }}</span>
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
          <span>{{ formatPlayerNameWithId(row.playerId) }}</span>
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
  --pass-fill: 0%;
  position: relative;
  overflow: hidden;
  display: inline-flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.45rem;
  border-radius: 999px;
  border: 1px solid rgba(148, 163, 184, 0.4);
  background: transparent;
  color: var(--app-text);
  padding: 0.32rem 0.72rem;
  font-size: 0.76rem;
  text-transform: none;
  letter-spacing: 0.02em;
}

.pointer-pass-btn::before {
  content: '';
  position: absolute;
  inset: 0 auto 0 0;
  width: var(--pass-fill);
  background: linear-gradient(90deg, rgba(56, 189, 248, 0.18), rgba(56, 189, 248, 0.08));
  pointer-events: none;
}

.pointer-pass-btn.selected {
  border-color: var(--app-accent-strong);
  color: var(--app-accent);
}

.pointer-pass-name,
.pointer-pass-prob {
  position: relative;
  z-index: 1;
}

.pointer-pass-prob {
  font-size: 0.68rem;
  color: var(--app-accent);
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
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
</style>
