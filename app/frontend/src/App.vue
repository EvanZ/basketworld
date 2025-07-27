<script setup>
import { ref, computed } from 'vue';
import GameSetup from './components/GameSetup.vue';
import GameBoard from './components/GameBoard.vue';
import PlayerControls from './components/PlayerControls.vue';
import GameOver from './components/GameOver.vue';
import { initGame, stepGame } from './services/api';

const gameState = ref(null);
const isLoading = ref(false);
const error = ref(null);
const initialSetup = ref(null);
const activePlayerId = ref(null);

async function handleGameStarted(setupData) {
  isLoading.value = true;
  error.value = null;
  initialSetup.value = setupData; // Save setup for play again
  try {
    const response = await initGame(setupData.runId, setupData.userTeam);
    if (response.status === 'success') {
      gameState.value = response.state;
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
  console.log('[App] Received actions-submitted with:', actions);
  isLoading.value = true;
  error.value = null;
  try {
    const response = await stepGame(actions);
     if (response.status === 'success') {
      console.log('[App] Step successful, updating gameState.');
      gameState.value = response.state;
    } else {
      console.error('[App] Step API returned an error status:', response);
      throw new Error(response.message || 'Failed to process step.');
    }
  } catch (err) {
    console.error('[App] Error during stepGame call:', err);
    error.value = err.message;
    console.error(err);
  } finally {
    isLoading.value = false;
  }
}

function handlePlayAgain() {
  gameState.value = null; // Clear the board
  activePlayerId.value = null; // Reset active player
  if (initialSetup.value) {
    handleGameStarted(initialSetup.value); // Start a new game with same settings
  }
}
</script>

<template>
  <main>
    <header>
      <h1>Welcome to BasketWorld</h1>
    </header>

    <GameSetup v-if="!gameState && !initialSetup" @game-started="handleGameStarted" :is-loading="isLoading" />

    <div v-if="isLoading && !gameState" class="loading">Loading Game...</div>
    <div v-if="error" class="error-message">{{ error }}</div>

    <div v-if="gameState" class="game-container">
      <GameBoard :game-state="gameState" :active-player-id="activePlayerId" />
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
