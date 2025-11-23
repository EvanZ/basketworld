<script setup>
import { ref, watch } from 'vue';
import { listPolicies } from '@/services/api';

const emit = defineEmits(['game-started']);

const runId = ref('');
const userTeam = ref('OFFENSE');

const unifiedPolicies = ref([]);
const selectedUnifiedPolicy = ref(null);
const useDifferentOpponentPolicy = ref(false);
const selectedOpponentUnifiedPolicy = ref(null);

async function fetchPolicies() {
  if (!runId.value) return;
  try {
    const res = await listPolicies(runId.value);
    unifiedPolicies.value = res.unified || [];
    selectedUnifiedPolicy.value = unifiedPolicies.value.at(-1) || null;
    selectedOpponentUnifiedPolicy.value = selectedUnifiedPolicy.value;
  } catch (e) {
    console.error('Failed to fetch policies', e);
  }
}

// fetch whenever runId changes with debounce-like watch
watch(runId, () => { fetchPolicies(); });

// This component now only needs to emit the user's choices.
// The parent App.vue will handle the API call and loading state.
function startGame() {
    if (runId.value) {
        const payload = {
            runId: runId.value,
            userTeam: userTeam.value,
            unifiedPolicyName: selectedUnifiedPolicy.value,
            opponentUnifiedPolicyName: useDifferentOpponentPolicy.value
                ? (selectedOpponentUnifiedPolicy.value || selectedUnifiedPolicy.value)
                : null,
        };
        console.log('[GameSetup] Emitting game-started event with:', payload);
        emit('game-started', payload);
    }
}
</script>

<template>
    <div class="setup-container">
        <h2>Game Setup</h2>
        <p>Enter an MLflow Run ID to load the trained agents.</p>
        
        <form @submit.prevent="startGame">
            <div class="form-group">
                <label for="runId">MLflow Run ID:</label>
                <input type="text" id="runId" v-model.trim="runId" placeholder="e.g., ab0f402bc060442fb669c60f696af773" required>
            </div>
            
            <div class="form-group">
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
            
            <div class="form-group" v-if="unifiedPolicies.length > 0">
                <label for="unifiedPol">Unified Policy:</label>
                <select id="unifiedPol" v-model="selectedUnifiedPolicy">
                    <option v-for="name in unifiedPolicies" :key="name" :value="name">{{ name }}</option>
                </select>
            </div>

            <div class="form-group" v-if="unifiedPolicies.length > 0">
                <label>
                    <input type="checkbox" v-model="useDifferentOpponentPolicy">
                    Use different policy for frozen opponent
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

.form-group {
  margin-bottom: 1.2rem;
  display: flex;
  flex-direction: column;
  gap: 0.6rem;
}

label {
  color: var(--app-text);
  font-size: 0.85rem;
  letter-spacing: 0.04em;
}

input[type="text"],
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

.error-message {
  margin-top: 1rem;
  color: #f87171;
}
</style>