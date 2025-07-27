<script setup>
import { ref, watch } from 'vue';
import GameSetup from './components/GameSetup.vue';
import GameBoard from './components/GameBoard.vue';
import PlayerControls from './components/PlayerControls.vue';
import GameOver from './components/GameOver.vue';
import { initGame, stepGame, getPolicyProbs } from './services/api';

const gameState = ref(null);      // For current state and UI logic
const gameHistory = ref([]);     // For ghost trails
const policyProbs = ref(null);   // For AI suggestions
const isLoading = ref(false);
const error = ref(null);
const initialSetup = ref(null);
const activePlayerId = ref(null);

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

    <div v-if="gameState" class="game-container">
      <GameBoard 
        :game-history="gameHistory" 
        :active-player-id="activePlayerId"
        :policy-probabilities="policyProbs"
      />
      <div class="controls-area">
        <PlayerControls 
          v-if="!gameState.done" 
          :game-state="gameState" 
          v-model:activePlayerId="activePlayerId"
          @actions-submitted="handleActionsSubmitted" 
        />
      </div>
    </div>

    <GameOver 
      v-if="gameState && gameState.done" 
      :results="gameState.last_action_results" 
      @play-again="handlePlayAgain" 
    />
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
