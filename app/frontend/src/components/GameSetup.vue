<script setup>
import { ref, watch } from 'vue';
import { listPolicies } from '@/services/api';

const emit = defineEmits(['game-started']);

const runId = ref('');
const userTeam = ref('OFFENSE');

const unifiedPolicies = ref([]);
const selectedUnifiedPolicy = ref(null);

async function fetchPolicies() {
  if (!runId.value) return;
  try {
    const res = await listPolicies(runId.value);
    unifiedPolicies.value = res.unified || [];
    selectedUnifiedPolicy.value = unifiedPolicies.value.at(-1) || null;
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
        console.log('[GameSetup] Emitting game-started event with:', { runId: runId.value, userTeam: userTeam.value, unifiedPolicyName: selectedUnifiedPolicy.value });
        emit('game-started', { runId: runId.value, userTeam: userTeam.value, unifiedPolicyName: selectedUnifiedPolicy.value });
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

            

            <button type="submit">
                Start Game
            </button>
        </form>
    </div>
</template>

<style scoped>
.setup-container {
    max-width: 500px;
    margin: 2rem auto;
    padding: 2rem;
    border: 1px solid #ccc;
    border-radius: 8px;
}
.form-group {
    margin-bottom: 1.5rem;
}
label {
    margin-right: 1rem;
}
input[type="text"] {
    width: 100%;
    padding: 0.5rem;
}
button {
    padding: 0.75rem 1.5rem;
    cursor: pointer;
}
.error-message {
    margin-top: 1rem;
    color: red;
}
</style> 