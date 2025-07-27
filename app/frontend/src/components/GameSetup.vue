<script setup>
import { ref } from 'vue';

const emit = defineEmits(['game-started']);

const runId = ref('');
const userTeam = ref('OFFENSE');

// This component now only needs to emit the user's choices.
// The parent App.vue will handle the API call and loading state.
function startGame() {
    if (runId.value) {
        console.log('[GameSetup] Emitting game-started event with:', { runId: runId.value, userTeam: userTeam.value });
        emit('game-started', { runId: runId.value, userTeam: userTeam.value });
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