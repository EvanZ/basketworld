<script setup>
import { ref, computed, watch, onMounted } from 'vue';
import HexagonControlPad from './HexagonControlPad.vue';
import { getActionValues, getShotProbability, getRewards } from '@/services/api';

const props = defineProps({
  gameState: { // This is now the currentGameState computed property from App.vue
    type: Object,
    required: true,
  },
  activePlayerId: {
    type: Number,
    default: null,
  },
  disabled: {
    type: Boolean,
    default: false,
  },
});

const emit = defineEmits(['actions-submitted', 'update:activePlayerId', 'play-again']);

const selectedActions = ref({});
const actionValues = ref(null);
const valueRange = ref({ min: 0, max: 0 });

// Add rewards tracking
const activeTab = ref('controls');
const rewardHistory = ref([]);
const episodeRewards = ref({ offense: 0.0, defense: 0.0 });

const isDefense = computed(() => {
  if (!props.gameState || props.activePlayerId === null) return false;
  return props.gameState.defense_ids.includes(props.activePlayerId);
});

const userControlledPlayerIds = computed(() => {
  if (!props.gameState || !props.gameState.user_team_name) {
    return [];
  }
  return props.gameState.user_team_name === 'OFFENSE' 
    ? props.gameState.offense_ids 
    : props.gameState.defense_ids;
});

// Watch for the active player ID to change, then fetch the action values and env shot prob for that player.
watch(() => props.activePlayerId, async (newPlayerId) => {
    actionValues.value = null; // Clear previous values
    valueRange.value = { min: 0, max: 0 };
    if (newPlayerId !== null && props.gameState && !props.gameState.done) {
        console.log(`[PlayerControls] Active player changed to ${newPlayerId}. Fetching action values...`);
        try {
            const values = await getActionValues(newPlayerId);
            console.log('[PlayerControls] Received action values from API:', values);
            actionValues.value = values;

            // Calculate min and max for color scaling
            const numericValues = Object.values(values).filter(v => typeof v === 'number');
            if (numericValues.length > 0) {
                valueRange.value.min = Math.min(...numericValues);
                valueRange.value.max = Math.max(...numericValues);
            }

        } catch (error) {
            console.error("Failed to fetch action values:", error);
            actionValues.value = { error: "Failed to load" }; // Show an error state
        }

        // Fetch backend-computed shot probability to ensure parity with environment logic
        try {
            const sp = await getShotProbability(newPlayerId);
            // Overwrite the computed shotProbability by setting a side value we use in the compute below
            _backendShotProb.value = sp.shot_probability;
            console.log('[PlayerControls] Fetched backend shot prob:', sp.shot_probability);
        } catch (e) {
            console.warn('[PlayerControls] Failed to fetch backend shot probability', e);
            _backendShotProb.value = null;
        }
    } else {
        console.log('[PlayerControls] No active player or game is done. Clearing action values.');
        actionValues.value = null;
    }
}, { immediate: true });

// Watch for the list of players to be populated, then set the first one as active.
// The `immediate` flag ensures this runs on component creation.
watch(userControlledPlayerIds, (newPlayerIds) => {
    if (newPlayerIds && newPlayerIds.length > 0 && props.activePlayerId === null) {
        emit('update:activePlayerId', newPlayerIds[0]);
    }
}, { immediate: true });

const actionNames = Object.values({
  0: "NOOP", 
  1: "MOVE_E", 2: "MOVE_NE", 3: "MOVE_NW", 4: "MOVE_W", 5: "MOVE_SW", 6: "MOVE_SE", 
  7: "SHOOT", 
  8: "PASS_E", 9: "PASS_NE", 10: "PASS_NW", 11: "PASS_W", 12: "PASS_SW", 13: "PASS_SE"
});

function getLegalActions(playerId) {
  if (!props.gameState.action_mask || !props.gameState.action_mask[playerId]) {
    return [];
  }
  const mask = props.gameState.action_mask[playerId];
  const legalActions = [];
  for (let i = 0; i < mask.length; i++) {
    if (mask[i] === 1 && i < actionNames.length) {
      legalActions.push(actionNames[i]);
    }
  }
  return legalActions;
}

function handleActionSelected(action) {
  if (props.disabled) return; // ignore clicks when disabled
  if (props.activePlayerId !== null) {
    // If the same action is clicked again, deselect it. Otherwise, select the new one.
    if (selectedActions.value[props.activePlayerId] === action) {
      delete selectedActions.value[props.activePlayerId];
    } else {
      selectedActions.value[props.activePlayerId] = action;
      // Optional: automatically switch to next player only when a new action is chosen
      const currentIndex = userControlledPlayerIds.value.indexOf(props.activePlayerId);
      const nextIndex = (currentIndex + 1) % userControlledPlayerIds.value.length;
      emit('update:activePlayerId', userControlledPlayerIds.value[nextIndex]);
    }
  }
}

function submitActions() {
  const userActions = {};
  for (const playerId of userControlledPlayerIds.value) {
    const actionName = selectedActions.value[playerId] || 'NOOP';
    userActions[playerId] = actionNames.indexOf(actionName);
  }
  console.log('[PlayerControls] Emitting actions-submitted with payload:', userActions);
  emit('actions-submitted', userActions);
  selectedActions.value = {};
  if (userControlledPlayerIds.value.length > 0) {
    emit('update:activePlayerId', userControlledPlayerIds.value[0]);
  }
}

// --- Shot Probability Logic (matches Python env) ---
function hexDistance(pos1, pos2) {
  const [q1, r1] = pos1;
  const [q2, r2] = pos2;
  // Must match Python _hex_distance: integer result
  return (Math.abs(q1 - q2) + Math.abs(q1 + r1 - q2 - r2) + Math.abs(r1 - r2)) / 2;
}

// Calculate shot probabilities for validation/display
const calculateShotProbability = (distance) => {
  if (!props.gameState?.shot_params) return 0;
  
  const { layup_pct, three_pt_pct } = props.gameState.shot_params;
  const three_point_distance = props.gameState.three_point_distance || 4;
  
  // Linear interpolation between layup and 3pt percentages
  const t = Math.min(distance / three_point_distance, 1.0);
  return layup_pct * (1 - t) + three_pt_pct * t;
};

// Fetch rewards from API
const fetchRewards = async () => {
  try {
    const data = await getRewards();
    rewardHistory.value = data.reward_history || [];
    episodeRewards.value = data.episode_rewards || { offense: 0.0, defense: 0.0 };
    console.log('[Rewards] Fetched rewards. History length:', rewardHistory.value.length, 'Episode totals:', episodeRewards.value);
  } catch (error) {
    console.error('Failed to fetch rewards:', error);
  }
};

// Watch for game state changes to update rewards
watch(() => props.gameState, () => {
  if (props.gameState) {
    console.log('[Rewards] Game state changed, fetching rewards. Done:', props.gameState.done);
    fetchRewards();
  }
}, { deep: true });

onMounted(() => {
  fetchRewards();
});

// Hold backend probability when available
const _backendShotProb = ref(null);

const shotProbability = computed(() => {
    if (props.activePlayerId === null || !props.gameState || !props.gameState.positions[props.activePlayerId]) {
        console.log('[ShotProb] Early return: no active player or game state');
        return null;
    }
    if (_backendShotProb.value !== null && _backendShotProb.value !== undefined) {
        console.log('[ShotProb] Using backend prob:', _backendShotProb.value);
        return _backendShotProb.value;
    }
    const playerPos = props.gameState.positions[props.activePlayerId];
    const basketPos = props.gameState.basket_position; 

    // --- DEBUG LOG ---
    console.log('[ShotCalc] Player Pos:', JSON.stringify(playerPos));
    console.log('[ShotCalc] Basket Pos:', JSON.stringify(basketPos));

    if (!playerPos || !basketPos) {
        console.error('[ShotCalc] Error: One of the positions is missing.');
        return null;
    }
    
    const distance = hexDistance(playerPos, basketPos);
    const prob = calculateShotProbability(distance);
    console.log('[ShotProb] Calculated prob:', prob, 'for distance:', distance);
    return prob;
});

</script>

<template>
  <div class="player-controls-container" :class="{ disabled: disabled && !gameState.done }">
    <h3>Player Controls</h3>
    
    <!-- Tab Navigation -->
    <div class="tab-navigation">
      <button 
        :class="{ active: activeTab === 'controls' }"
        @click="activeTab = 'controls'"
      >
        Controls
      </button>
      <button 
        :class="{ active: activeTab === 'rewards' }"
        @click="activeTab = 'rewards'"
      >
        Rewards
      </button>
    </div>

    <!-- Controls Tab -->
    <div v-if="activeTab === 'controls'" class="tab-content">
      <div class="player-tabs">
          <button 
              v-for="playerId in userControlledPlayerIds" 
              :key="playerId"
              :class="{ active: activePlayerId === playerId }"
              @click="$emit('update:activePlayerId', playerId)"
              :disabled="disabled && !gameState.done"
          >
              Player {{ playerId }}
              <span v-if="selectedActions[playerId]">
                ({{ selectedActions[playerId].startsWith('MOVE') ? 'M' : selectedActions[playerId].startsWith('PASS') ? 'P' : selectedActions[playerId] }})
              </span>
          </button>
      </div>
      
      <div class="control-pad-wrapper" v-if="activePlayerId !== null">
          <HexagonControlPad 
              :legal-actions="getLegalActions(activePlayerId)"
              :selected-action="selectedActions[activePlayerId]"
              :shot-probability="shotProbability"
              :pass-probabilities="passProbabilities"
              @action-selected="handleActionSelected"
              :action-values="actionValues"
              :value-range="valueRange"
              :is-defense="isDefense"
          />
          <p v-if="selectedActions[activePlayerId]">
              Selected for Player {{ activePlayerId }}: <strong>{{ selectedActions[activePlayerId] }}</strong>
          </p>
      </div>

      <button @click="submitActions" class="submit-button" :disabled="gameState.done || (disabled && !gameState.done)">
        {{ gameState.done ? 'Game Over' : disabled ? 'AI Playing' : 'Submit Turn' }}
      </button>
      
      <button @click="$emit('play-again')" class="new-game-button">
        New Game
      </button>
    </div>

    <!-- Rewards Tab -->
    <div v-if="activeTab === 'rewards'" class="tab-content">
      <div class="rewards-section">
        <h4>Episode Totals</h4>
        <div class="episode-totals">
          <div class="total-item">
            <span class="team-label offense">Offense:</span>
            <span class="reward-value">{{ episodeRewards.offense.toFixed(2) }}</span>
          </div>
          <div class="total-item">
            <span class="team-label defense">Defense:</span>
            <span class="reward-value">{{ episodeRewards.defense.toFixed(2) }}</span>
          </div>
        </div>

        <h4>Turn History</h4>
        <div class="reward-history">
          <div v-if="rewardHistory.length === 0" class="no-rewards">
            No rewards recorded yet.
          </div>
          <div v-else class="reward-table">
            <div class="reward-header">
              <span>Turn</span>
              <span>Offense</span>
              <span>Off. Reason</span>
              <span>Defense</span>
              <span>Def. Reason</span>
            </div>
            <div 
              v-for="reward in rewardHistory" 
              :key="reward.step"
              class="reward-row"
            >
              <span>{{ reward.step }}</span>
              <span :class="{ positive: reward.offense > 0, negative: reward.offense < 0 }">
                {{ reward.offense.toFixed(2) }}
              </span>
              <span class="reason-text">{{ reward.offense_reason }}</span>
              <span :class="{ positive: reward.defense > 0, negative: reward.defense < 0 }">
                {{ reward.defense.toFixed(2) }}
              </span>
              <span class="reason-text">{{ reward.defense_reason }}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.player-controls-container {
  padding: 1rem;
  background-color: #f8f9fa;
  border-radius: 8px;
  display: flex;
  flex-direction: column;
  gap: 1rem;
  max-height: 80vh;
  overflow-y: auto;
}

.tab-navigation {
  display: flex;
  gap: 0.5rem;
  border-bottom: 1px solid #dee2e6;
  margin-bottom: 1rem;
}

.tab-navigation button {
  padding: 0.5rem 1rem;
  border: none;
  background-color: transparent;
  cursor: pointer;
  border-bottom: 2px solid transparent;
  font-weight: 500;
  transition: all 0.2s ease;
}

.tab-navigation button:hover {
  background-color: #e9ecef;
}

.tab-navigation button.active {
  border-bottom-color: #007bff;
  color: #007bff;
}

.tab-content {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

/* Existing styles */
.player-tabs {
  display: flex;
  gap: 0.5rem;
  flex-wrap: wrap;
}

.player-tabs button {
  padding: 0.5rem 1rem;
  border: 1px solid #ccc;
  background-color: white;
  cursor: pointer;
  border-radius: 4px;
  transition: all 0.2s ease;
}

.player-tabs button:hover {
  background-color: #e9ecef;
}

.player-tabs button.active {
  background-color: #007bff;
  color: white;
  border-color: #007bff;
}

.player-tabs button:disabled {
  background-color: #e9ecef;
  color: #6c757d;
  cursor: not-allowed;
  opacity: 0.6;
}

/* Rewards styles */
.rewards-section {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.episode-totals {
  display: flex;
  gap: 1rem;
  padding: 1rem;
  background-color: white;
  border-radius: 4px;
  border: 1px solid #dee2e6;
}

.total-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.25rem;
}

.team-label {
  font-weight: bold;
  font-size: 0.9rem;
}

.team-label.offense {
  color: #dc3545;
}

.team-label.defense {
  color: #007bff;
}

.reward-value {
  font-size: 1.2rem;
  font-weight: bold;
}

.reward-history {
  max-height: 300px;
  overflow-y: auto;
}

.no-rewards {
  text-align: center;
  padding: 2rem;
  color: #6c757d;
  font-style: italic;
}

.reward-table {
  background-color: white;
  border-radius: 4px;
  border: 1px solid #dee2e6;
  overflow: hidden;
}

.reward-header {
  display: grid;
  grid-template-columns: 1fr 1fr 1.5fr 1fr 1.5fr;
  padding: 0.75rem;
  background-color: #f8f9fa;
  font-weight: bold;
  border-bottom: 1px solid #dee2e6;
  text-align: center;
}

.reward-row {
  display: grid;
  grid-template-columns: 1fr 1fr 1.5fr 1fr 1.5fr;
  padding: 0.5rem 0.75rem;
  border-bottom: 1px solid #f1f3f4;
  text-align: center;
}

.reward-row:last-child {
  border-bottom: none;
}

.reward-row:hover {
  background-color: #f8f9fa;
}

.positive {
  color: #28a745;
  font-weight: bold;
}

.negative {
  color: #dc3545;
  font-weight: bold;
}

.reason-text {
  font-size: 0.85rem;
  color: #6c757d;
  font-style: italic;
}

/* Existing control styles */
.control-pad-wrapper {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 1rem;
}

.submit-button, .new-game-button {
  padding: 0.75rem 1.5rem;
  font-size: 1rem;
  font-weight: bold;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  transition: background-color 0.2s ease;
  min-width: 120px;
}

.submit-button {
  background-color: #28a745;
  color: white;
}

.submit-button:disabled {
  background-color: #6c757d;
  cursor: not-allowed;
}

.submit-button:hover:not(:disabled) {
  background-color: #218838;
}

.new-game-button {
  background-color: #007bff;
  color: white;
}

.new-game-button:hover {
  background-color: #0056b3;
}

.disabled {
  opacity: 0.7;
  pointer-events: none;
}
</style> 