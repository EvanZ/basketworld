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
});

const emit = defineEmits(['action-selected']);

const visibleActions = computed(() => {
    return props.legalActions.filter(action => action !== 'NOOP');
});

// --- Icon Definitions ---
const icons = {
    move: `<svg xmlns="http://www.w3.org/2000/svg" height="24" viewBox="0 0 24 24" width="24" fill="currentColor"><path d="M0 0h24v24H0z" fill="none"/><path d="M12 4l-1.41 1.41L16.17 11H4v2h12.17l-5.58 5.59L12 20l8-8z"/></svg>`,
    pass: `<svg xmlns="http://www.w3.org/2000/svg" height="24" viewBox="0 0 24 24" width="24" fill="currentColor"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" fill="none"/><path d="M2.01 3L2 10l15 2-15 2 .01 7L23 12z"/></svg>`,
    shoot: `<svg xmlns="http://www.w3.org/2000/svg" height="24" viewBox="0 0 24 24" width="24" fill="currentColor"><path d="M0 0h24v24H0z" fill="none"/><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.42 0-8-3.58-8-8s3.58-8 8-8 8 3.58 8 8-3.58 8-8 8zM9.5 16.5c0-1.38 1.12-2.5 2.5-2.5s2.5 1.12 2.5 2.5c0 .82-.4 1.54-1 2h-3c-.6-.46-1-1.18-1-2zm2.5-11c1.38 0 2.5 1.12 2.5 2.5S13.38 10.5 12 10.5 9.5 9.38 9.5 8s1.12-2.5 2.5-2.5z"/></svg>`,
};

// --- Button Layout and Icon Configuration ---
const buttonConfig = {
  SHOOT: { style: { top: '50%', left: '50%', transform: 'translate(-50%, -50%)' }, icon: icons.shoot, rotation: 0 },
  // Moves
  MOVE_E:  { style: { top: '50%', left: '100%', transform: 'translate(-50%, -50%)' }, icon: icons.move, rotation: 0 },
  MOVE_NE: { style: { top: '15%', left: '75%', transform: 'translate(-50%, -50%)' }, icon: icons.move, rotation: -60 },
  MOVE_NW: { style: { top: '15%', left: '25%', transform: 'translate(-50%, -50%)' }, icon: icons.move, rotation: -120 },
  MOVE_W:  { style: { top: '50%', left: '0%', transform: 'translate(-50%, -50%)' }, icon: icons.move, rotation: 180 },
  MOVE_SW: { style: { top: '85%', left: '25%', transform: 'translate(-50%, -50%)' }, icon: icons.move, rotation: 120 },
  MOVE_SE: { style: { top: '85%', left: '75%', transform: 'translate(-50%, -50%)' }, icon: icons.move, rotation: 60 },
  // Passes
  PASS_E:  { style: { top: '50%', left: '135%', transform: 'translate(-50%, -50%)' }, icon: icons.pass, rotation: 0 },
  PASS_NE: { style: { top: '-5%', left: '90%', transform: 'translate(-50%, -50%)' }, icon: icons.pass, rotation: -45 },
  PASS_NW: { style: { top: '-5%', left: '10%', transform: 'translate(-50%, -50%)' }, icon: icons.pass, rotation: -135 },
  PASS_W:  { style: { top: '50%', left: '-35%', transform: 'translate(-50%, -50%)' }, icon: icons.pass, rotation: 180 },
  PASS_SW: { style: { top: '105%', left: '10%', transform: 'translate(-50%, -50%)' }, icon: icons.pass, rotation: 135 },
  PASS_SE: { style: { top: '105%', left: '90%', transform: 'translate(-50%, -50%)' }, icon: icons.pass, rotation: 45 },
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
        :style="buttonConfig[action].style"
        @click="selectAction(action)"
        class="action-button"
        :class="{ selected: action === selectedAction }"
        :title="action"
      >
        <span class="icon-wrapper" :style="{ transform: `rotate(${buttonConfig[action].rotation}deg)` }" v-html="buttonConfig[action].icon"></span>
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
  justify-content: center;
  align-items: center;
  font-size: 14px; /* Slightly larger font */
  border-radius: 50%; /* Make buttons circular */
  cursor: pointer;
  border: 1px solid #ccc;
  background-color: #f0f0f0;
  transition: background-color 0.2s, color 0.2s;
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