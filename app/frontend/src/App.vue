<script setup>
import { ref, watch } from 'vue';
import GameSetup from './components/GameSetup.vue';
import GameBoard from './components/GameBoard.vue';
import PlayerControls from './components/PlayerControls.vue';
import { initGame, stepGame, getPolicyProbs, saveEpisode } from './services/api';

const gameState = ref(null);      // For current state and UI logic
const gameHistory = ref([]);     // For ghost trails
const policyProbs = ref(null);   // For AI suggestions
const isLoading = ref(false);
const error = ref(null);
const initialSetup = ref(null);
const activePlayerId = ref(null);

// AI Mode for pre-selecting actions, not automatic play
const aiMode = ref(false);
const deterministic = ref(true);

// Watch for when episodes end to stop auto-play behavior
watch(gameState, async (newState) => {
    if (newState && !newState.done) {
        try {
            const response = await getPolicyProbs();
            policyProbs.value = response;
        } catch (err) {
            console.error('[App] Failed to fetch policy probs:', err);
        }
    }
    // When an episode ends, disable AI mode to allow starting a new game
    if (newState && newState.done) {
        aiMode.value = false;
    }
});

async function handleGameStarted(setupData) {
  isLoading.value = true;
  error.value = null;
  initialSetup.value = setupData;
  gameState.value = null;      // Ensure old board is cleared
  gameHistory.value = [];      // Clear history
  try {
    const response = await initGame(setupData.runId, setupData.userTeam, setupData.offensePolicyName, setupData.defensePolicyName);
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

// New function for self-play mode (runs full episode)
async function handleSelfPlay() {
  if (!gameState.value || !aiMode.value) return;
  
  // Run full episode with AI controlling all players
  while (gameState.value && !gameState.value.done) {
    try {
      // Get AI actions for user-controlled players
      let aiActions = {};
      
      if (policyProbs.value) {
        // Determine which players are user-controlled
        const userControlledIds = gameState.value.user_team_name === 'OFFENSE' 
          ? gameState.value.offense_ids 
          : gameState.value.defense_ids;
        
        // Get AI actions for user-controlled players
        for (const playerId of userControlledIds) {
          const probs = policyProbs.value[playerId];
          if (Array.isArray(probs)) {
            // Pick action with highest probability (argmax)
            let bestActionIndex = 0;
            for (let i = 1; i < probs.length; i++) {
              if (probs[i] > probs[bestActionIndex]) {
                bestActionIndex = i;
              }
            }
            aiActions[playerId] = bestActionIndex;
          }
        }
      }
      
      const response = await stepGame(aiActions);
      if (response.status === 'success') {
        gameState.value = response.state;
        gameHistory.value.push(response.state);
        // Small delay to make the progression visible
        await new Promise(resolve => setTimeout(resolve, 100));
      } else {
        throw new Error(response.message || 'Failed to process step.');
      }
    } catch (err) {
      error.value = err.message;
      console.error(err);
      break;
    }
  }
}

async function handleSaveEpisode() {
  try {
    const res = await saveEpisode();
    alert(`Episode saved to ${res.file_path}`);
  } catch (e) {
    alert(`Failed to save episode: ${e.message}`);
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

    <GameSetup v-if="!gameState" @game-started="handleGameStarted" />
    
    <div v-if="isLoading && !gameState" class="loading">Loading Game...</div>
    <div v-if="error" class="error-message">{{ error }}</div>
    


    <div v-if="gameState" class="ai-toggle">
      <label>
        <input type="checkbox" v-model="aiMode" /> AI Mode
      </label>
      <label v-if="aiMode" style="margin-left: 10px;">
        <input type="checkbox" v-model="deterministic" /> Deterministic (Q-values)
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
            :disabled="false"
            :ai-mode="aiMode"
            :deterministic="deterministic"
            @actions-submitted="handleActionsSubmitted" 
            @play-again="handlePlayAgain"
            @self-play="handleSelfPlay"
        />

        <button v-if="gameState.done" @click="handleSaveEpisode" class="save-episode-button">
          Save Episode
        </button>
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

.save-episode-button {
  margin-top: 1rem;
  padding: 0.5rem 1rem;
}
</style>
