<script setup>
import { ref, computed, watch } from 'vue';
import HexagonControlPad from './HexagonControlPad.vue';
import { getActionValues } from '@/services/api';

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

// Watch for the active player ID to change, then fetch the action values for that player.
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

// --- Shot Probability Logic (Ported from Python) ---
function hexDistance(pos1, pos2) {
  const [q1, r1] = pos1;
  const [q2, r2] = pos2;
  // This formula must exactly match the Python environment's _hex_distance method.
  return (Math.abs(q1 - q2) + Math.abs(q1 + r1 - q2 - r2) + Math.abs(r1 - r2)) / 2;
}

function calculateShotProbability(distance) {
  if (distance <= 1) return 0.9;
  if (distance <= 3) return 0.5;
  if (distance <= 5) return 0.2;
  return 0.05;
}

const shotProbability = computed(() => {
    if (props.activePlayerId === null || !props.gameState || !props.gameState.positions[props.activePlayerId]) {
        return null;
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
    return calculateShotProbability(distance);
});

</script>

<template>
  <div class="player-controls-container" :class="{ disabled: disabled && !gameState.done }">
    <h3>Player Controls</h3>
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
</template>

<style scoped>
.player-controls-container {
  padding: 1rem;
  background-color: #f8f9fa;
  border-radius: 8px;
  display: flex;
  flex-direction: column;
}
.player-tabs {
    display: flex;
    margin-bottom: 1rem;
}
.player-tabs button {
    flex-grow: 1;
    padding: 0.5rem;
    border: 1px solid #ccc;
    background: #fff;
    cursor: pointer;
}
.player-tabs button.active {
    background-color: #007bff;
    color: white;
    border-color: #007bff;
}
.control-pad-wrapper {
    border: 1px solid #dee2e6;
    border-radius: 4px;
    padding: 1rem;
    margin-bottom: 1rem;
    text-align: center;
}
.new-game-button {
    margin-top: 1rem;
    width: 100%;
    padding: 0.75rem;
    background-color: #28a745;
    color: white;
    border: none;
    border-radius: 4px;
    font-size: 1rem;
    cursor: pointer;
}
.submit-button {
  margin-top: auto; /* Pushes button to the bottom */
  padding: 0.75rem;
  background-color: #28a745;
  color: white;
  border: none;
  font-size: 1.2rem;
}
.submit-button:disabled {
  background-color: #6c757d;
  cursor: not-allowed;
}

.player-controls-container.disabled {
  opacity: 0.5;
  pointer-events: none;
}
</style> 