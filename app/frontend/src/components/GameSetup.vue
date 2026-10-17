<script setup>
import { onMounted, onUnmounted, ref, watch } from 'vue';
import { listPolicies, listRecentTrainingRuns } from '@/services/api';

const emit = defineEmits(['game-started', 'template-sandbox-started']);

const runId = ref('');
const userTeam = ref('OFFENSE');

const unifiedPolicies = ref([]);
const selectedUnifiedPolicy = ref(null);
const useDifferentOpponentPolicy = ref(false);
const selectedOpponentUnifiedPolicy = ref(null);
const isMultiPossessionRun = ref(false);
const episodeEndMode = ref('possessions');
const gameClockSeconds = ref(300);
const overtimeSeconds = ref(120);
const recentRuns = ref([]);
const recentRunsLoading = ref(false);
const trainerAvailable = ref(true);
const copiedRunId = ref('');
let recentRunsTimer = null;

async function refreshRecentRuns({ showLoading = true } = {}) {
  if (showLoading) recentRunsLoading.value = true;
  try {
    const result = await listRecentTrainingRuns(8);
    trainerAvailable.value = result?.available !== false;
    recentRuns.value = Array.isArray(result?.runs) ? result.runs : [];
  } catch (error) {
    trainerAvailable.value = false;
    recentRuns.value = [];
    console.warn('[GameSetup] Unable to load recent trainer runs:', error);
  } finally {
    if (showLoading) recentRunsLoading.value = false;
  }
}

function useRecentRun(run) {
  const mlflowRunId = String(run?.mlflow_run_id || '').trim();
  if (mlflowRunId) runId.value = mlflowRunId;
}

async function copyRecentRunId(run) {
  const mlflowRunId = String(run?.mlflow_run_id || '').trim();
  if (!mlflowRunId) return;
  try {
    const copyInput = document.createElement('textarea');
    copyInput.value = mlflowRunId;
    copyInput.setAttribute('readonly', '');
    copyInput.style.position = 'fixed';
    copyInput.style.opacity = '0';
    document.body.appendChild(copyInput);
    copyInput.select();
    const copiedBySelection = document.execCommand('copy');
    document.body.removeChild(copyInput);
    if (!copiedBySelection) {
      if (!navigator.clipboard?.writeText) throw new Error('Clipboard is unavailable.');
      await navigator.clipboard.writeText(mlflowRunId);
    }
    copiedRunId.value = mlflowRunId;
    window.setTimeout(() => {
      if (copiedRunId.value === mlflowRunId) copiedRunId.value = '';
    }, 1600);
  } catch (error) {
    console.warn('[GameSetup] Unable to copy MLflow run ID:', error);
  }
}

function formatRunProgress(run) {
  const current = Math.max(0, Number(run?.current_update) || 0);
  const target = Math.max(0, Number(run?.target_updates) || 0);
  if (!target) return `U${current.toLocaleString()}`;
  return `U${current.toLocaleString()} / ${target.toLocaleString()}`;
}

function statusClass(status) {
  return `status-${String(status || 'unknown').toLowerCase().replace(/[^a-z0-9_-]/g, '-')}`;
}

onMounted(() => {
  refreshRecentRuns();
  recentRunsTimer = window.setInterval(
    () => refreshRecentRuns({ showLoading: false }),
    5000,
  );
});

onUnmounted(() => {
  if (recentRunsTimer !== null) window.clearInterval(recentRunsTimer);
});

async function fetchPolicies() {
  if (!runId.value) {
    unifiedPolicies.value = [];
    selectedUnifiedPolicy.value = null;
    selectedOpponentUnifiedPolicy.value = null;
    isMultiPossessionRun.value = false;
    return;
  }
  const requestedRunId = runId.value;
  try {
    const res = await listPolicies(runId.value);
    if (requestedRunId !== runId.value) return;
    unifiedPolicies.value = res.unified || [];
    selectedUnifiedPolicy.value = unifiedPolicies.value.at(-1) || null;
    selectedOpponentUnifiedPolicy.value = selectedUnifiedPolicy.value;
    isMultiPossessionRun.value = res.enable_multi_possession === true;
    episodeEndMode.value = res.episode_end_mode === 'game_clock' ? 'game_clock' : 'possessions';
    gameClockSeconds.value = Math.max(1, Number(res.game_clock_seconds) || 300);
    overtimeSeconds.value = Math.max(1, Number(res.overtime_seconds) || 120);
    if (isMultiPossessionRun.value) {
      // Team A is the blue player roster. Keep the legacy enum only as the
      // API's stable internal representation of that roster.
      userTeam.value = 'OFFENSE';
    }
  } catch (e) {
    if (requestedRunId !== runId.value) return;
    isMultiPossessionRun.value = false;
    console.error('Failed to fetch policies', e);
  }
}

// fetch whenever runId changes with debounce-like watch
watch(runId, () => {
  isMultiPossessionRun.value = false;
  fetchPolicies();
});

// This component now only needs to emit the user's choices.
// The parent App.vue will handle the API call and loading state.
function startGame() {
    if (runId.value) {
        const payload = {
            runId: runId.value,
            userTeam: isMultiPossessionRun.value ? 'OFFENSE' : userTeam.value,
            unifiedPolicyName: selectedUnifiedPolicy.value,
            opponentUnifiedPolicyName: useDifferentOpponentPolicy.value
                ? (selectedOpponentUnifiedPolicy.value || selectedUnifiedPolicy.value)
                : null,
            episodeEndMode: episodeEndMode.value,
            gameClockSeconds: gameClockSeconds.value,
            overtimeSeconds: overtimeSeconds.value,
        };
        console.log('[GameSetup] Emitting game-started event with:', payload);
        emit('game-started', payload);
    }
}

function openTemplateSandbox() {
    emit('template-sandbox-started', {
        runId: runId.value || null,
        userTeam: isMultiPossessionRun.value ? 'OFFENSE' : userTeam.value,
    });
}
</script>

<template>
    <div class="setup-container">
        <h2>Game Setup</h2>
        <p>Enter an MLflow Run ID to load a JAX checkpoint artifact from that run.</p>

        <section class="recent-runs-panel" aria-labelledby="recent-runs-heading">
          <div class="recent-runs-header">
            <div>
              <h3 id="recent-runs-heading">Recent Trainer Runs</h3>
              <span>Click an MLflow ID to copy it, or use a run directly.</span>
            </div>
            <button
              type="button"
              class="recent-refresh-btn"
              :disabled="recentRunsLoading"
              @click="refreshRecentRuns()"
            >
              {{ recentRunsLoading ? 'Loading…' : 'Refresh' }}
            </button>
          </div>
          <div v-if="!trainerAvailable" class="recent-runs-empty">
            Trainer API is offline.
          </div>
          <div v-else-if="!recentRunsLoading && recentRuns.length === 0" class="recent-runs-empty">
            No trainer-managed runs yet.
          </div>
          <div v-else class="recent-runs-list">
            <article v-for="run in recentRuns" :key="run.id" class="recent-run-row">
              <div class="recent-run-summary">
                <div class="recent-run-title-row">
                  <strong>{{ run.name }}</strong>
                  <span class="run-status" :class="statusClass(run.status)">{{ run.status }}</span>
                </div>
                <span class="run-progress">{{ formatRunProgress(run) }}</span>
                <button
                  v-if="run.mlflow_run_id"
                  type="button"
                  class="run-id-link"
                  :title="`Copy MLflow run ID ${run.mlflow_run_id}`"
                  @click="copyRecentRunId(run)"
                >
                  {{ run.mlflow_run_id }}
                  <span class="copy-state">{{ copiedRunId === run.mlflow_run_id ? 'Copied!' : 'Copy' }}</span>
                </button>
                <span v-else class="run-id-pending">Waiting for MLflow run ID…</span>
              </div>
              <button
                type="button"
                class="recent-use-btn"
                :disabled="!run.mlflow_run_id"
                @click="useRecentRun(run)"
              >
                Use
              </button>
            </article>
          </div>
        </section>
        
        <form @submit.prevent="startGame">
            <div class="form-group">
                <label for="runId">MLflow Run ID:</label>
                <input
                  type="text"
                  id="runId"
                  v-model.trim="runId"
                  placeholder="e.g., ab0f402bc060442fb669c60f696af773"
                  required
                >
            </div>
            
            <div v-if="!isMultiPossessionRun" class="form-group">
                <p>Choose your team:</p>
                <label>
                    <input type="radio" v-model="userTeam" value="OFFENSE">
                    Offense
                </label>
                <label>
                    <input type="radio" v-model="userTeam" value="DEFENSE">
                    Defense
                </label>
            </div>
            <div v-else class="form-group fixed-team-summary">
              <p>Playing as the Blue Team</p>
              <span>Teams alternate offense and defense; the opening possession is decided by the jump ball.</span>
            </div>

            <fieldset class="game-mode-settings">
              <legend>Episode format</legend>
              <div class="mode-options">
                <label :class="{ active: episodeEndMode === 'possessions' }">
                  <input v-model="episodeEndMode" type="radio" value="possessions">
                  Possessions
                </label>
                <label :class="{ active: episodeEndMode === 'game_clock' }">
                  <input v-model="episodeEndMode" type="radio" value="game_clock">
                  Game clock
                </label>
              </div>
              <div v-if="episodeEndMode === 'game_clock'" class="clock-duration-grid">
                <label>
                  Regulation seconds
                  <input v-model.number="gameClockSeconds" type="number" min="1" required>
                  <span>Default 5:00</span>
                </label>
                <label>
                  Overtime seconds
                  <input v-model.number="overtimeSeconds" type="number" min="1" required>
                  <span>Default 2:00</span>
                </label>
              </div>
              <p v-else>Uses the run's possession limit and possession overtime settings.</p>
              <small>Changing this starts a new game; it does not convert an active game.</small>
            </fieldset>

            <div class="form-group" v-if="unifiedPolicies.length > 0">
                <label for="unifiedPol">Unified Policy:</label>
                <select id="unifiedPol" v-model="selectedUnifiedPolicy">
                    <option v-for="name in unifiedPolicies" :key="name" :value="name">{{ name }}</option>
                </select>
            </div>

            <div class="form-group" v-if="unifiedPolicies.length > 0">
                <label>
                    <input type="checkbox" v-model="useDifferentOpponentPolicy">
                    {{ isMultiPossessionRun ? 'Use different policy for opponent team' : 'Use different policy for frozen opponent' }}
                </label>
            </div>
            <div class="form-group" v-if="useDifferentOpponentPolicy && unifiedPolicies.length > 0">
                <label for="opponentUnifiedPol">Opponent Policy:</label>
                <select id="opponentUnifiedPol" v-model="selectedOpponentUnifiedPolicy">
                    <option v-for="name in unifiedPolicies" :key="name" :value="name">{{ name }}</option>
                </select>
            </div>

            <button type="submit">
                Start Game
            </button>
            <button type="button" class="secondary-btn" @click="openTemplateSandbox">
                Open Template Sandbox
            </button>
            <p class="sandbox-hint">
                Uses the entered run ID to copy court and player configuration when available. No model is loaded.
            </p>
        </form>
    </div>
</template>

<style scoped>
.setup-container {
  max-width: 540px;
  margin: 1.5rem auto;
  padding: 1.75rem;
  background: var(--app-panel);
  border: 1px solid var(--app-panel-border);
  border-radius: 24px;
  box-shadow: 0 20px 45px rgba(2, 6, 23, 0.45);
}

.setup-container h2 {
  margin-bottom: 0.6rem;
  color: var(--app-accent);
  letter-spacing: 0.1em;
  text-transform: uppercase;
  font-size: 0.95rem;
}

.setup-container p {
  color: var(--app-text-muted);
  margin-bottom: 1.5rem;
  font-size: 0.9rem;
}

.recent-runs-panel {
  margin: 1rem 0 1.4rem;
  padding: 0.9rem;
  border: 1px solid rgba(56, 189, 248, 0.24);
  border-radius: 16px;
  background: rgba(7, 13, 28, 0.72);
}

.recent-runs-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 0.75rem;
  margin-bottom: 0.75rem;
}

.recent-runs-header h3 {
  margin: 0 0 0.2rem;
  color: var(--app-text);
  font-size: 0.82rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.recent-runs-header span,
.recent-runs-empty {
  color: var(--app-text-muted);
  font-size: 0.72rem;
}

.recent-runs-list {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  max-height: 18rem;
  overflow-y: auto;
}

.recent-run-row {
  display: flex;
  align-items: center;
  gap: 0.65rem;
  padding: 0.65rem;
  border: 1px solid rgba(148, 163, 184, 0.16);
  border-radius: 12px;
  background: rgba(15, 23, 42, 0.62);
}

.recent-run-summary {
  min-width: 0;
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}

.recent-run-title-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.5rem;
}

.recent-run-title-row strong {
  overflow: hidden;
  color: var(--app-text);
  font-size: 0.78rem;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.run-status {
  flex: 0 0 auto;
  padding: 0.12rem 0.42rem;
  border-radius: 999px;
  background: rgba(148, 163, 184, 0.15);
  color: var(--app-text-muted);
  font-size: 0.62rem;
  letter-spacing: 0.07em;
  text-transform: uppercase;
}

.run-status.status-running,
.run-status.status-starting {
  background: rgba(34, 197, 94, 0.16);
  color: #86efac;
}

.run-status.status-paused,
.run-status.status-pausing {
  background: rgba(250, 204, 21, 0.15);
  color: #fde047;
}

.run-status.status-failed,
.run-status.status-stopped {
  background: rgba(248, 113, 113, 0.15);
  color: #fca5a5;
}

.run-progress,
.run-id-pending {
  color: var(--app-text-muted);
  font-size: 0.68rem;
}

.run-id-link {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.5rem;
  width: 100%;
  margin: 0;
  padding: 0;
  border: 0;
  border-radius: 0;
  background: transparent;
  color: var(--app-accent);
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  font-size: 0.68rem;
  letter-spacing: 0;
  text-align: left;
  text-transform: none;
}

.run-id-link:hover {
  transform: none;
  color: var(--app-accent-strong);
}

.copy-state {
  flex: 0 0 auto;
  color: var(--app-text-muted);
  font-family: inherit;
  font-size: 0.62rem;
}

.recent-refresh-btn,
.recent-use-btn {
  margin: 0;
  padding: 0.34rem 0.65rem;
  font-size: 0.65rem;
}

.recent-use-btn {
  flex: 0 0 auto;
}

.recent-refresh-btn:disabled,
.recent-use-btn:disabled {
  cursor: not-allowed;
  opacity: 0.45;
}

.form-group {
  margin-bottom: 1.2rem;
  display: flex;
  flex-direction: column;
  gap: 0.6rem;
}

.sandbox-hint {
  margin-top: 0.75rem;
  margin-bottom: 0;
  font-size: 0.8rem;
  color: var(--app-text-muted);
}

.fixed-team-summary p {
  margin: 0;
  color: var(--app-text);
  font-size: 0.9rem;
}

.fixed-team-summary span {
  color: var(--app-text-muted);
  font-size: 0.8rem;
  line-height: 1.4;
}

label {
  color: var(--app-text);
  font-size: 0.85rem;
  letter-spacing: 0.04em;
}

input[type="text"],
input[type="number"],
select {
  width: 100%;
  background: rgba(13, 20, 38, 0.85);
  border: 1px solid rgba(56, 189, 248, 0.35);
  color: var(--app-text);
  border-radius: 18px;
  padding: 0.6rem 0.9rem;
  font-family: inherit;
  letter-spacing: 0.04em;
  transition: border 0.2s ease, box-shadow 0.2s ease;
}

.game-mode-settings {
  margin: 0 0 1.2rem;
  padding: 0.9rem;
  border: 1px solid rgba(56, 189, 248, 0.3);
  border-radius: 16px;
  background: rgba(7, 13, 28, 0.58);
}

.game-mode-settings legend {
  padding: 0 0.4rem;
  color: var(--app-text);
  font-size: 0.78rem;
  font-weight: 800;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.mode-options { display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; }
.mode-options label { display: flex; align-items: center; gap: 0.45rem; padding: 0.55rem 0.7rem; border: 1px solid rgba(148, 163, 184, 0.25); border-radius: 12px; }
.mode-options label.active { border-color: var(--app-accent); background: rgba(56, 189, 248, 0.12); color: var(--app-accent); }
.clock-duration-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.65rem; margin-top: 0.8rem; }
.clock-duration-grid label { display: grid; gap: 0.35rem; }
.clock-duration-grid span, .game-mode-settings small { color: var(--app-text-muted); font-size: 0.7rem; }
.game-mode-settings p { margin: 0.8rem 0 0.4rem; }

input:focus,
select:focus {
  outline: none;
  border-color: var(--app-accent-strong);
  box-shadow: 0 0 12px rgba(56, 189, 248, 0.35);
}

.radio-row {
  display: flex;
  gap: 1rem;
  flex-wrap: wrap;
}

.radio-row label {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  padding: 0.35rem 0.75rem;
  border-radius: 16px;
  background: rgba(15, 23, 42, 0.7);
  border: 1px solid rgba(148, 163, 184, 0.3);
}

button {
  margin-top: 0.5rem;
  padding: 0.65rem 1.4rem;
  border-radius: 999px;
  border: 1px solid rgba(148, 163, 184, 0.4);
  background: transparent;
  color: var(--app-text);
  letter-spacing: 0.1em;
  text-transform: uppercase;
  font-size: 0.8rem;
  transition: color 0.15s ease, border 0.15s ease, transform 0.15s ease;
}

button:hover {
  transform: translateY(-1px);
  color: var(--app-accent);
  border-color: var(--app-accent-strong);
}

.secondary-btn {
  margin-left: 0.75rem;
}

.error-message {
  margin-top: 1rem;
  color: #f87171;
}

</style>
