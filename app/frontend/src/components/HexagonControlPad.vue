<script setup>
import { computed } from 'vue';

const props = defineProps({
  legalActions: {
    type: Array,
    required: true,
  },
  selectedAction: {
    type: String,
    default: null,
  },
  // shotProbability removed (shown on board instead)
  passProbabilities: {
    type: Object,
    default: null,
  },
  actionValues: {
    type: Object,
    default: null,
  },
  valueRange: {
    type: Object,
    default: () => ({ min: 0, max: 0 }),
  },
  isDefense: {
    type: Boolean,
    default: false,
  }
});

const emit = defineEmits(['action-selected']);

const visibleActions = computed(() => {
    return props.legalActions.filter(action => action !== 'NOOP');
});

function getActionColor(action) {
    if (!props.actionValues || props.actionValues[action] === undefined) {
        return '#f0f0f0'; // Default background color
    }

    const value = props.actionValues[action];
    const { min, max } = props.valueRange;

    if (min === max) {
        return 'rgba(255, 165, 0, 0.5)'; // Orange if all values are the same
    }

    // Normalize the value to a 0-1 range
    let normalized = (value - min) / (max - min);

    // If the player is on defense, a lower Q-value is better, so we flip the scale.
    if (props.isDefense) {
        normalized = 1 - normalized;
    }

    // Linear interpolation between blue and orange
    // Blue: rgb(0, 0, 255), Orange: rgb(255, 165, 0)
    const r = 0 + normalized * (255 - 0);
    const g = 0 + normalized * (165 - 0);
    const b = 255 * (1 - normalized);

    return `rgba(${r}, ${g}, ${b}, 0.5)`;
}

// --- Icon Definitions (Font Awesome components) ---
const faIcon = {
  move: ['fas', 'location-arrow'],
  pass: ['far', 'hand-pointer'],
  shoot: ['fas', 'bullseye'],
};

// --- Button Layout and Icon Configuration ---
const buttonConfig = {
  SHOOT: { style: { top: '50%', left: '50%', transform: 'translate(-50%, -50%)', width: '60px', borderRadius: '8px' }, icon: faIcon.shoot, rotation: 0, offset: 0 },
  // Moves
  MOVE_E:  { style: { top: '50%', left: '100%', transform: 'translate(-50%, -50%)' }, icon: faIcon.move, rotation: 0, offset: 45 },
  MOVE_NE: { style: { top: '15%', left: '75%', transform: 'translate(-50%, -50%)' }, icon: faIcon.move, rotation: -60, offset: 45 },
  MOVE_NW: { style: { top: '15%', left: '25%', transform: 'translate(-50%, -50%)' }, icon: faIcon.move, rotation: -120, offset: 45 },
  MOVE_W:  { style: { top: '50%', left: '0%', transform: 'translate(-50%, -50%)' }, icon: faIcon.move, rotation: 180, offset: 45 },
  MOVE_SW: { style: { top: '85%', left: '25%', transform: 'translate(-50%, -50%)' }, icon: faIcon.move, rotation: 120, offset: 45 },
  MOVE_SE: { style: { top: '85%', left: '75%', transform: 'translate(-50%, -50%)' }, icon: faIcon.move, rotation: 60, offset: 45 },
  // Passes
  PASS_E:  { style: { top: '50%', left: '135%', transform: 'translate(-50%, -50%)' }, icon: faIcon.pass, rotation: 0, offset: 90 },
  PASS_NE: { style: { top: '-5%', left: '90%', transform: 'translate(-50%, -50%)' }, icon: faIcon.pass, rotation: -60, offset: 90 },
  PASS_NW: { style: { top: '-5%', left: '10%', transform: 'translate(-50%, -50%)' }, icon: faIcon.pass, rotation: -120, offset: 90 },
  PASS_W:  { style: { top: '50%', left: '-35%', transform: 'translate(-50%, -50%)' }, icon: faIcon.pass, rotation: 180, offset: 90 },
  PASS_SW: { style: { top: '105%', left: '10%', transform: 'translate(-50%, -50%)' }, icon: faIcon.pass, rotation: 120, offset: 90 },
  PASS_SE: { style: { top: '105%', left: '90%', transform: 'translate(-50%, -50%)' }, icon: faIcon.pass, rotation: 60, offset: 90 },
};

function selectAction(action) {
  emit('action-selected', action);
}
</script>

<template>
  <div class="control-pad-container">
    <div class="control-pad">
      <button
        v-for="action in visibleActions"
        :key="action"
        :style="{ ...buttonConfig[action].style, backgroundColor: getActionColor(action) }"
        @click="selectAction(action)"
        class="action-button"
        :class="{ selected: action === selectedAction }"
        :title="action"
      >
        <span class="icon-wrapper" :style="{ transform: `rotate(${buttonConfig[action].rotation + (buttonConfig[action].offset || 0)}deg)` }">
          <font-awesome-icon :icon="buttonConfig[action].icon" />
        </span>
        <!-- Removed shot % display from control button -->
        <span v-if="passProbabilities && passProbabilities[action] !== undefined" class="prob-tooltip">
            {{ Math.round(passProbabilities[action] * 100) }}%
        </span>
        <span v-if="actionValues && actionValues[action] !== undefined" class="value-display">
            {{ actionValues[action].toFixed(2) }}
        </span>
      </button>
    </div>
  </div>
</template>

<style scoped>
.control-pad-container {
  display: flex;
  justify-content: center;
  align-items: center;
  padding: 2rem 0;
  margin-top: 1rem;
}
.control-pad {
  position: relative;
  width: 180px; /* Increased size to accommodate icons */
  height: 180px;
}
.action-button {
  position: absolute;
  width: 44px; /* Standardize width */
  height: 44px; /* Standardize height */
  padding: 0;
  display: flex;
  flex-direction: column; /* Stack icon and value vertically */
  justify-content: center;
  align-items: center;
  font-size: 14px; /* Slightly larger font */
  border-radius: 50%; /* Make buttons circular */
  cursor: pointer;
  border: 1px solid #ccc;
  background-color: #f0f0f0;
  transition: background-color 0.2s, color 0.2s;
}

.value-display {
  font-size: .85rem;
  color: var(--app-text);
}

.action-button.selected {
  background-color: #007bff;
  color: white;
  border-color: #0056b3;
}

.icon-wrapper {
  display: flex;
  justify-content: center;
  align-items: center;
}
</style> 