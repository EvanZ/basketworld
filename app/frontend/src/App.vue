<script setup>
import { ref, watch } from 'vue';
import GameSetup from './components/GameSetup.vue';
import GameBoard from './components/GameBoard.vue';
import PlayerControls from './components/PlayerControls.vue';
import { initGame, stepGame, getPolicyProbs } from './services/api';

const gameState = ref(null);      // For current state and UI logic
const gameHistory = ref([]);     // For ghost trails
const policyProbs = ref(null);   // For AI suggestions
const isLoading = ref(false);
const error = ref(null);
const initialSetup = ref(null);
const activePlayerId = ref(null);
const autoPlay = ref(false); // AI mode toggle

// Auto-play watcher: whenever toggled ON (and a game is running) start looping
watch(autoPlay, (newVal) => {
  if (newVal) {
    autoPlayLoop();
  }
});

async function autoPlayLoop() {
  // Stop conditions
  if (!autoPlay.value || !gameState.value || gameState.value.done) return;

  // Build actions for the players that would normally be user-controlled so that
  // they also follow the learnt policy instead of defaulting to NOOP.
  const autoActions = {};
  if (policyProbs.value) {
    const controlledIds = gameState.value.user_team_name === 'OFFENSE'
      ? gameState.value.offense_ids
      : gameState.value.defense_ids;
    for (const pid of controlledIds) {
      const probs = policyProbs.value[pid];
      if (Array.isArray(probs)) {
        // pick argmax
        let best = 0;
        for (let i = 1; i < probs.length; i++) {
          if (probs[i] > probs[best]) best = i;
        }
        autoActions[pid] = best;
      }
    }
  }

  try {
    const response = await stepGame(autoActions);
    if (response.status === 'success') {
      gameState.value = response.state;
      gameHistory.value.push(response.state);
    }
  } catch (err) {
    error.value = err.message;
    console.error('[App] AutoPlay step failed:', err);
    // Abort autoplay on error
    autoPlay.value = false;
    return;
  }

  // Continue loop after a short delay to keep UI responsive
  setTimeout(autoPlayLoop, 150); // ~7 frames per second; adjust as desired
}

watch(gameState, async (newState) => {
    if (newState && !newState.done) {
        try {
            console.log('[App] Game state changed, fetching policy probabilities...');
            const probs = await getPolicyProbs();
            console.log('[App] Received policy probabilities:', probs);
            policyProbs.value = probs;
        } catch (e) {
            console.error("[App] Failed to fetch policy probabilities:", e);
            policyProbs.value = null;
        }
    }

    // If the episode ended, automatically stop AI mode so that the user can start
    // a fresh game without needing to untick the toggle first.
    if (newState && newState.done) {
        autoPlay.value = false;
    }
});

async function handleGameStarted(setupData) {
  isLoading.value = true;
  error.value = null;
  initialSetup.value = setupData;
  gameState.value = null;      // Ensure old board is cleared
  gameHistory.value = [];      // Clear history
  try {
    const response = await initGame(setupData.runId, setupData.userTeam);
    if (response.status === 'success') {
      gameState.value = response.state;
      gameHistory.value.push(response.state);
    } else {
      throw new Error(response.message || 'Failed to start game.');
    }
  } catch (err) {
    error.value = err.message;
    console.error(err);
  } finally {
    isLoading.value = false;
  }
}

async function handleActionsSubmitted(actions) {
  if (!gameState.value) return;
  // No loading indicator for steps, feels more responsive
  try {
    const response = await stepGame(actions);
     if (response.status === 'success') {
      gameState.value = response.state;
      gameHistory.value.push(response.state);
    } else {
      throw new Error(response.message || 'Failed to process step.');
    }
  } catch (err) {
    error.value = err.message;
    console.error(err);
  }
}

function handlePlayAgain() {
  gameState.value = null;
  gameHistory.value = [];
  policyProbs.value = null;
  activePlayerId.value = null;
  if (initialSetup.value) {
    handleGameStarted(initialSetup.value);
  }
}
</script>

<template>
  <main>
    <header>
      <h1>Welcome to BasketWorld</h1>
    </header>

    <GameSetup v-if="!gameState && !initialSetup" @game-started="handleGameStarted" />
    
    <div v-if="isLoading && !gameState" class="loading">Loading Game...</div>
    <div v-if="error" class="error-message">{{ error }}</div>

    <div v-if="gameState" class="ai-toggle">
      <label>
        <input type="checkbox" v-model="autoPlay" /> AI Mode
      </label>
    </div>

    <div v-if="gameState" class="game-container">
      <GameBoard 
        :game-history="gameHistory" 
        :active-player-id="activePlayerId"
        :policy-probabilities="policyProbs"
      />
      <div class="controls-area">
        <PlayerControls 
          :game-state="gameState" 
          v-model:activePlayerId="activePlayerId"
          :disabled="autoPlay && !gameState.done"
          @actions-submitted="handleActionsSubmitted" 
          @play-again="handlePlayAgain"
        />
      </div>
    </div>
  </main>
</template>

<style scoped>
header {
  text-align: center;
  margin-bottom: 2rem;
}
.loading {
  text-align: center;
  font-size: 1.5rem;
  margin-top: 2rem;
}
.error-message {
  color: red;
  text-align: center;
  margin-top: 1rem;
}

.ai-toggle {
  text-align: center;
  margin-bottom: 1rem;
}
.game-container {
  display: flex;
  flex-direction: row; /* Changed to row */
  justify-content: center; /* Center items horizontally */
  align-items: flex-start; /* Align items to the top */
  gap: 2rem; /* Increased gap */
}

.controls-area {
  width: 400px; /* Give the controls area a fixed width */
  flex-shrink: 0; /* Prevent the controls area from shrinking */
}
</style>
