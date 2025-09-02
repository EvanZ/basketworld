<script setup>
import { ref, watch } from 'vue';
import GameSetup from './components/GameSetup.vue';
import GameBoard from './components/GameBoard.vue';
import PlayerControls from './components/PlayerControls.vue';
import { initGame, stepGame, getPolicyProbs, saveEpisode, startSelfPlay, replayLastEpisode } from './services/api';

const gameState = ref(null);      // For current state and UI logic
const gameHistory = ref([]);     // For ghost trails
const policyProbs = ref(null);   // For AI suggestions
const isLoading = ref(false);
const error = ref(null);
const initialSetup = ref(null);
const activePlayerId = ref(null);
// Reflect the actions being applied on each step to keep UI tabs in sync during self-play
const currentSelections = ref(null);
const isSelfPlaying = ref(false);
const canReplay = ref(false);
const isReplaying = ref(false);
// Force remount of PlayerControls to clear internal state between games
const controlsKey = ref(0);

// AI Mode for pre-selecting actions, not automatic play
const aiMode = ref(true);
const deterministic = ref(false);

// Shared move tracking between manual and AI play
const moveHistory = ref([]);

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
        aiMode.value = true;
    }
});

async function handleGameStarted(setupData) {
  console.log('[App] Starting game with data:', setupData);
  
  // Clear move history for new game
  moveHistory.value = [];
  // Clear any self-play selections state
  currentSelections.value = null;
  isSelfPlaying.value = false;
  // Bump key to reset PlayerControls' internal state (like selectedActions)
  controlsKey.value += 1;
  
  isLoading.value = true;
  error.value = null;
  initialSetup.value = setupData;
  gameState.value = null;      // Ensure old board is cleared
  gameHistory.value = [];      // Clear history
  try {
    const response = await initGame(
      setupData.runId,
      setupData.userTeam,
      setupData.offensePolicyName,
      setupData.defensePolicyName,
      setupData.unifiedPolicyName ?? null,
    );
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
    const response = await stepGame(actions, deterministic.value);
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

function handleMoveRecorded(moveData) {
    console.log('[App] Recording move:', moveData);
    moveHistory.value.push(moveData);
}

// New function for self-play mode (runs full episode)
async function handleSelfPlay(preselected = null) {
  if (!gameState.value || !aiMode.value) return;
  // Start deterministic self-play on backend: snapshot seed and initial state
  try {
    const res = await startSelfPlay();
    if (res && res.status === 'success' && res.state) {
      // Reset UI to backend's reset state so trajectories align
      gameState.value = res.state;
      gameHistory.value = [res.state];
      moveHistory.value = [];
      currentSelections.value = null;
    }
  } catch (e) {
    console.error('[App] Failed to start self-play on backend:', e);
  }
  isSelfPlaying.value = true;
  currentSelections.value = null;
  
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
          const actionMask = gameState.value.action_mask[playerId];

          // If this is the first loop iteration and preselected is provided, honor it
          if (preselected && preselected[`$${playerId}`] === undefined && preselected[playerId] === undefined) {
            // normalize to string and numeric lookup
          }
          if (preselected) {
            let chosen = null;
            const preVal = preselected[playerId] ?? preselected[`$${playerId}`];
            if (typeof preVal === 'string') {
              const names = [
                "NOOP","MOVE_E","MOVE_NE","MOVE_NW","MOVE_W","MOVE_SW","MOVE_SE",
                "SHOOT","PASS_E","PASS_NE","PASS_NW","PASS_W","PASS_SW","PASS_SE"
              ];
              const idx = names.indexOf(preVal);
              if (idx >= 0 && actionMask[idx] === 1) chosen = idx;
            } else if (typeof preVal === 'number') {
              if (preVal >= 0 && preVal < actionMask.length && actionMask[preVal] === 1) chosen = preVal;
            }
            if (chosen != null) {
              aiActions[playerId] = chosen;
              continue; // do not override a valid preselection
            }
          }
          
          if (Array.isArray(probs) && Array.isArray(actionMask)) {
            // Pick action with highest probability (argmax) among LEGAL actions only
            let bestActionIndex = 0;
            let bestProb = -1;
            
            for (let i = 0; i < probs.length && i < actionMask.length; i++) {
              // Only consider legal actions (action_mask[i] === 1)
              if (actionMask[i] === 1 && probs[i] > bestProb) {
                bestProb = probs[i];
                bestActionIndex = i;
              }
            }
            
            console.log(`[App] Self-play selected action ${bestActionIndex} for player ${playerId} (legal: ${actionMask[bestActionIndex] === 1}, prob: ${bestProb})`);
            aiActions[playerId] = bestActionIndex;
          }
        }
      }
      // Clear preselected after applying for the first step
      preselected = null;
      
      // Track moves for AI self-play
      if (Object.keys(aiActions).length > 0) {
        const currentTurn = moveHistory.value.length + 1;
        const teamMoves = {};
        
        // Convert action indices back to action names for tracking
        const actionNames = [
          "NOOP", "MOVE_E", "MOVE_NE", "MOVE_NW", "MOVE_W", "MOVE_SW", "MOVE_SE", 
          "SHOOT", "PASS_E", "PASS_NE", "PASS_NW", "PASS_W", "PASS_SW", "PASS_SE"
        ];
        
        const appliedSelections = {};
        for (const [playerId, actionIndex] of Object.entries(aiActions)) {
          const actionName = actionNames[actionIndex] || 'UNKNOWN';
          teamMoves[`Player ${playerId}`] = actionName;
          appliedSelections[playerId] = actionName;
        }
        // Update UI tabs to mirror applied actions
        currentSelections.value = appliedSelections;
        
        moveHistory.value.push({
          turn: currentTurn,
          moves: teamMoves
        });
      }
      
      const response = await stepGame(aiActions, deterministic.value);
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
  // Self-play finished
  isSelfPlaying.value = false;
  currentSelections.value = null;
  canReplay.value = true;
}

async function handleSaveEpisode() {
  try {
    const res = await saveEpisode();
    alert(`Episode saved to ${res.file_path}`);
  } catch (e) {
    alert(`Failed to save episode: ${e.message}`);
  }
}

async function handleReplay() {
  try {
    const res = await replayLastEpisode();
    if (res.status === 'success' && Array.isArray(res.states) && res.states.length > 0) {
      // Animate the replay so it is visible in the UI
      isReplaying.value = true;
      gameHistory.value = [];
      for (const s of res.states) {
        gameState.value = s;
        gameHistory.value.push(s);
        // Small delay between frames
        // eslint-disable-next-line no-await-in-loop
        await new Promise(resolve => setTimeout(resolve, 300));
      }
      isReplaying.value = false;
    } else {
      throw new Error('Invalid replay response');
    }
  } catch (e) {
    alert(`Failed to replay episode: ${e.message}`);
  }
}

function handlePlayAgain() {
  gameState.value = null;
  gameHistory.value = [];
  policyProbs.value = null;
  activePlayerId.value = null;
  currentSelections.value = null;
  isSelfPlaying.value = false;
  controlsKey.value += 1;
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
      <button class="toggle-btn" @click="aiMode = !aiMode">
        <font-awesome-icon :icon="aiMode ? ['fas','toggle-on'] : ['fas','toggle-off']" />
        <span class="toggle-label">AI Mode</span>
      </button>
      <button class="toggle-btn" @click="deterministic = !deterministic" :disabled="!aiMode">
        <font-awesome-icon :icon="deterministic ? ['fas','toggle-on'] : ['fas','toggle-off']" />
        <span class="toggle-label">Deterministic</span>
      </button>
    </div>

    <div v-if="gameState" class="game-container">
      <GameBoard 
        :game-history="gameHistory" 
        :active-player-id="activePlayerId"
        :policy-probabilities="policyProbs"
      />
      <div class="controls-area">
        <PlayerControls 
            :key="controlsKey"
            :game-state="gameState" 
            v-model:activePlayerId="activePlayerId"
            :disabled="false"
            :ai-mode="aiMode"
            :deterministic="deterministic"
            :move-history="moveHistory"
            :external-selections="isSelfPlaying ? currentSelections : null"
            @actions-submitted="handleActionsSubmitted" 
            @play-again="handlePlayAgain"
            @self-play="handleSelfPlay"
            @move-recorded="handleMoveRecorded"
        />

        <button v-if="gameState.done" @click="handleSaveEpisode" class="save-episode-button">
          Save Episode
        </button>
        <button v-if="gameState.done && canReplay" @click="handleReplay" class="save-episode-button" style="margin-left: 0.5rem;">
          <font-awesome-icon :icon="['fas', 'redo']" />
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
.toggle-btn {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.4rem 0.8rem;
  border: 1px solid #ddd;
  border-radius: 6px;
  background: #fff;
  cursor: pointer;
  margin: 0 0.5rem;
  font-size: 1.2rem;
}
.toggle-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
.toggle-label {
  font-weight: 500;
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
