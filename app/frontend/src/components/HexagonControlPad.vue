<script setup>
import { computed } from 'vue';
const SQRT3 = Math.sqrt(3);

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
  },
  layoutVariant: {
    type: String,
    default: 'classic',
  }
});

const emit = defineEmits(['action-selected']);
const isCourtVariant = computed(() => String(props.layoutVariant || '').toLowerCase() === 'court');

const visibleActions = computed(() => {
    return props.legalActions.filter(action => action !== 'NOOP');
});

function getActionColor(action, isSelected = false) {
    // Let selected classes control the neon selected styling.
    if (isSelected) return null;
    if (!props.actionValues || props.actionValues[action] === undefined) {
        return null;
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

const courtActionCoords = {
  SHOOT: [0, 0],
  MOVE_E: [1, 0],
  MOVE_NE: [1, -1],
  MOVE_NW: [0, -1],
  MOVE_W: [-1, 0],
  MOVE_SW: [-1, 1],
  MOVE_SE: [0, 1],
  PASS_E: [2, 0],
  PASS_NE: [2, -2],
  PASS_NW: [0, -2],
  PASS_W: [-2, 0],
  PASS_SW: [-2, 2],
  PASS_SE: [0, 2],
};

const COURT_HEX_RADIUS_PX = 30;
const COURT_HEX_WIDTH_PX = SQRT3 * COURT_HEX_RADIUS_PX;
const COURT_HEX_HEIGHT_PX = COURT_HEX_RADIUS_PX * 2;
const COURT_HEX_HALF_WIDTH_PX = COURT_HEX_WIDTH_PX / 2;
const COURT_HEX_HALF_HEIGHT_PX = COURT_HEX_HEIGHT_PX / 2;

function axialToPixel(q, r, radius = COURT_HEX_RADIUS_PX) {
  return {
    x: radius * SQRT3 * (q + (r / 2)),
    y: radius * 1.5 * r,
  };
}

function getActionVisual(action) {
  return buttonConfig[action] || {
    style: { top: '50%', left: '50%', transform: 'translate(-50%, -50%)' },
    icon: faIcon.move,
    rotation: 0,
    offset: 0,
  };
}

const courtButtons = computed(() => {
  return visibleActions.value.map((action) => {
    const coord = courtActionCoords[action] || [0, 0];
    const [q, r] = coord;
    const { x, y } = axialToPixel(q, r);
    return { action, x, y };
  });
});

const courtBounds = computed(() => {
  if (courtButtons.value.length === 0) {
    return {
      minX: -COURT_HEX_HALF_WIDTH_PX,
      maxX: COURT_HEX_HALF_WIDTH_PX,
      minY: -COURT_HEX_HALF_HEIGHT_PX,
      maxY: COURT_HEX_HALF_HEIGHT_PX,
      width: COURT_HEX_WIDTH_PX,
      height: COURT_HEX_HEIGHT_PX,
      offsetX: COURT_HEX_HALF_WIDTH_PX,
      offsetY: COURT_HEX_HALF_HEIGHT_PX,
    };
  }

  const xs = courtButtons.value.map((btn) => btn.x);
  const ys = courtButtons.value.map((btn) => btn.y);

  const minX = Math.min(...xs) - COURT_HEX_HALF_WIDTH_PX;
  const maxX = Math.max(...xs) + COURT_HEX_HALF_WIDTH_PX;
  const minY = Math.min(...ys) - COURT_HEX_HALF_HEIGHT_PX;
  const maxY = Math.max(...ys) + COURT_HEX_HALF_HEIGHT_PX;

  return {
    minX,
    maxX,
    minY,
    maxY,
    width: maxX - minX,
    height: maxY - minY,
    offsetX: -minX,
    offsetY: -minY,
  };
});

const courtPadStyle = computed(() => {
  return {
    width: `${courtBounds.value.width}px`,
    height: `${courtBounds.value.height}px`,
    '--hex-radius': `${COURT_HEX_RADIUS_PX}px`,
    '--hex-width': `${COURT_HEX_WIDTH_PX}px`,
    '--hex-height': `${COURT_HEX_HEIGHT_PX}px`,
  };
});

function getCourtButtonStyle(button) {
  return {
    left: `${button.x + courtBounds.value.offsetX}px`,
    top: `${button.y + courtBounds.value.offsetY}px`,
  };
}

function selectAction(action) {
  emit('action-selected', action);
}
</script>

<template>
  <div class="control-pad-container" :class="{ 'court-layout': isCourtVariant }">
    <div v-if="isCourtVariant" class="control-pad-court" :style="courtPadStyle">
      <button
        v-for="button in courtButtons"
        :key="button.action"
        :style="{ ...getCourtButtonStyle(button), '--tile-fill': getActionColor(button.action, button.action === selectedAction) || 'transparent' }"
        @click="selectAction(button.action)"
        class="action-button hex-action-button"
        :class="{ selected: button.action === selectedAction }"
        :title="button.action"
      >
        <svg class="hex-tile-bg" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
          <polygon points="50,0 93.3,25 93.3,75 50,100 6.7,75 6.7,25" />
        </svg>
        <span class="icon-wrapper" :style="{ transform: `rotate(${getActionVisual(button.action).rotation + (getActionVisual(button.action).offset || 0)}deg)` }">
          <font-awesome-icon :icon="getActionVisual(button.action).icon" />
        </span>
        <span v-if="passProbabilities && passProbabilities[button.action] !== undefined" class="prob-tooltip">
          {{ Math.round(passProbabilities[button.action] * 100) }}%
        </span>
        <span v-if="actionValues && actionValues[button.action] !== undefined" class="value-display">
          {{ actionValues[button.action].toFixed(2) }}
        </span>
      </button>
    </div>
    <div v-else class="control-pad">
      <button
        v-for="action in visibleActions"
        :key="action"
        :style="{ ...getActionVisual(action).style, backgroundColor: getActionColor(action, action === selectedAction) }"
        @click="selectAction(action)"
        class="action-button classic-action-button"
        :class="{ selected: action === selectedAction }"
        :title="action"
      >
        <span class="icon-wrapper" :style="{ transform: `rotate(${getActionVisual(action).rotation + (getActionVisual(action).offset || 0)}deg)` }">
          <font-awesome-icon :icon="getActionVisual(action).icon" />
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

.control-pad-container.court-layout {
  padding: 0.45rem 0 0.1rem;
  margin-top: 0.2rem;
}

.control-pad {
  position: relative;
  width: 180px; /* Increased size to accommodate icons */
  height: 180px;
}

.control-pad-court {
  position: relative;
}

.action-button {
  position: absolute;
  padding: 0;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  font-size: 14px;
  cursor: pointer;
  border: 1px solid #ccc;
  background-color: #f0f0f0;
  transition: background-color 0.2s, color 0.2s, border-color 0.2s;
}

.classic-action-button {
  width: 44px; /* Standardize width */
  height: 44px; /* Standardize height */
  border-radius: 50%; /* Make buttons circular */
}

.hex-action-button {
  transform: translate(-50%, -50%);
  width: var(--hex-width, 52px);
  height: var(--hex-height, 60px);
  border: none;
  background: transparent;
  box-shadow: none;
  overflow: visible;
  transition: transform 0.15s ease, box-shadow 0.15s ease, color 0.15s ease;
}

.hex-tile-bg {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
}

.hex-tile-bg polygon {
  fill: var(--tile-fill, transparent);
  stroke: rgba(148, 163, 184, 0.55);
  stroke-width: 1.5;
  vector-effect: non-scaling-stroke;
}

.hex-action-button:hover {
  color: var(--app-accent);
  transform: translate(-50%, -50%) translateY(-1px);
  z-index: 3;
  box-shadow: 0 12px 25px rgba(14, 165, 233, 0.35);
}

.hex-action-button:hover .hex-tile-bg polygon {
  stroke: rgba(56, 189, 248, 0.75);
}

.value-display {
  font-size: .85rem;
  color: var(--app-text);
}

.action-button.selected {
  color: var(--app-accent);
  border-color: rgba(56, 189, 248, 0.95);
  box-shadow:
    0 0 0 1px rgba(56, 189, 248, 0.5),
    0 0 18px rgba(14, 165, 233, 0.35);
}

.classic-action-button.selected {
  background: transparent;
}

.hex-action-button.selected {
  background: transparent;
  box-shadow: none;
}

.hex-action-button.selected:hover {
  box-shadow: 0 12px 25px rgba(14, 165, 233, 0.35);
}

.hex-action-button.selected .hex-tile-bg polygon {
  stroke: rgba(56, 189, 248, 0.95);
  filter: drop-shadow(0 0 8px rgba(14, 165, 233, 0.4));
}

.icon-wrapper {
  display: flex;
  justify-content: center;
  align-items: center;
}

.hex-action-button .icon-wrapper {
  font-size: 1rem;
}
</style>
