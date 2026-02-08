<script setup>
import { computed, ref, watch, onMounted, onBeforeUnmount } from 'vue';
import { getShotProbability, getPassStealProbabilities } from '@/services/api';

const props = defineProps({
  gameHistory: {
    type: Array,
    required: true,
  },
  activePlayerId: {
    type: Number,
    default: null,
  },
  policyProbabilities: {
    type: Object,
    default: null,
  },
  isManualStepping: {
    type: Boolean,
    default: false,
  },
  isShotClockUpdating: {
    type: Boolean,
    default: false,
  },
  selectedActions: {
    type: Object,
    default: () => ({}),
  },
  shotAccumulator: {
    type: Object,
    default: () => ({}),
  },
  disableTransitions: {
    type: Boolean,
    default: false,
  },
  moveProgress: {
    type: Number,
    default: 1,
  },
  shotChartLabel: {
    type: String,
    default: '',
  },
  placementMode: {
    type: Boolean,
    default: false,
  },
  placementPositions: {
    type: Array,
    default: null,
  },
  placementBallHolder: {
    type: Number,
    default: null,
  },
  placementEditable: {
    type: Boolean,
    default: false,
  },
  placementPassProbs: {
    type: Object,
    default: () => ({}),
  },
});

const emit = defineEmits(['update:activePlayerId', 'update-player-position', 'adjust-shot-clock', 'update-placement']);

// ------------------------------------------------------------
//  HEXAGON GEOMETRY — POINTY-TOP, ODD-R OFFSET  (matches Python)
// ------------------------------------------------------------

const HEX_RADIUS = 24;  // pixel radius of one hexagon corner-to-center
const SQRT3 = Math.sqrt(3);
const PASS_COS_EPS = 1e-9;
// Axial direction vectors (q, r) aligned with ActionType ordering (E, NE, NW, W, SW, SE)
const HEX_DIRECTIONS = [
  [1, 0],
  [1, -1],
  [0, -1],
  [-1, 0],
  [-1, 1],
  [0, 1],
];
const PASS_ACTION_TO_DIR = {
  PASS_E: 0,
  PASS_NE: 1,
  PASS_NW: 2,
  PASS_W: 3,
  PASS_SW: 4,
  PASS_SE: 5,
};

// Axial (q,r) → pixel cartesian (x,y) for pointy-topped hexes.
// Formula identical to the one in basketworld_env_v2.py:_render_visual.
function axialToCartesian(q, r) {
  const x = HEX_RADIUS * (Math.sqrt(3) * q + Math.sqrt(3) / 2 * r);
  // Positive Y increases downward, matching the environment's coordinate system
  const y = HEX_RADIUS * (1.5 * r);
  return { x, y };
}
// Helper function from Python environment to get axial coordinates for "odd-r"
function offsetToAxial(col, row) {
  const q = col - ((row - (row & 1)) >> 1);
  const r = row;
  return { q, r };
}

// Hex distance on axial coords (matches env._hex_distance)
function hexDistance(a, b) {
  const [q1, r1] = a;
  const [q2, r2] = b;
  return (Math.abs(q1 - q2) + Math.abs(q1 + r1 - q2 - r2) + Math.abs(r1 - r2)) / 2;
}

function getRenderablePlayers(gameState) {
  if (!gameState || !gameState.positions) return [];
  return gameState.positions.map((pos, index) => {
    const [q, r] = pos;
    const { x, y } = axialToCartesian(q, r);
    const isOffense = gameState.offense_ids.includes(index);
    const hasBall = gameState.ball_holder === index;
    return { id: index, x, y, isOffense, hasBall };
  });
}

const svgRef = ref(null);
const draggedPlayerId = ref(null);
const draggedPlayerPos = ref({ x: 0, y: 0 });
const isDragging = ref(false);
const passStealProbs = ref({});
const ballColor = '#ffa500';
const PASS_FLASH_DURATION_MS = 1100;
const SHOT_FLASH_DURATION_MS = 1100;
const passFlash = ref(null);
const passFlashTimeout = ref(null);
const shotFlash = ref(null);
const shotFlashTimeout = ref(null);
const shotJumpPlayerId = ref(null);
const shotJumpTimeout = ref(null);
const shotJumpIsDunk = ref(false);
const shotInFlightPlayerId = computed(() => {
  const shots = currentGameState.value?.last_action_results?.shots;
  if (!shots || Object.keys(shots).length === 0) return null;
  const entry = Object.entries(shots)[0];
  if (!entry || entry.length < 2) return null;
  const shooterId = Number(entry[0]);
  return Number.isNaN(shooterId) ? null : shooterId;
});
const policyVisibility = ref(new Set()); // Player IDs with policy overlays shown
const clickTimeout = ref(null);
const SINGLE_CLICK_DELAY = 220;
const DRIBBLE_PERIOD_SECONDS = 0.5;
const DRIBBLE_AMPLITUDE_PX = HEX_RADIUS * 0.33;
const SHOOT_JUMP_PERIOD_SECONDS = 2.0;
const SHOOT_JUMP_AMPLITUDE_PX = HEX_RADIUS * 1.0;
const SHOOT_DUNK_AMPLITUDE_PX = HEX_RADIUS * 1.33;
const SHOOT_JUMP_SCALE = 1.2;
const SHOOT_DUNK_SCALE = 1.5;

function clearClickTimeout() {
  if (clickTimeout.value) {
    clearTimeout(clickTimeout.value);
    clickTimeout.value = null;
  }
}

function togglePolicyVisibility(playerId) {
  const next = new Set(policyVisibility.value);
  if (next.has(playerId)) {
    next.delete(playerId);
  } else {
    next.add(playerId);
  }
  policyVisibility.value = next;
}

function showPoliciesForAllPlayers() {
  const positions = currentGameState.value?.positions;
  if (!positions || positions.length === 0) {
    policyVisibility.value = new Set();
    return;
  }
  policyVisibility.value = new Set(positions.map((_, idx) => idx));
}

function hideAllPolicies() {
  policyVisibility.value = new Set();
}

function toggleAllPolicies() {
  if (allPoliciesVisible.value) {
    hideAllPolicies();
  } else {
    showPoliciesForAllPlayers();
  }
}

function isPolicyVisible(playerId) {
  return policyVisibility.value.has(playerId);
}

function onPlayerClick(_event, player) {
  // If dragging, don't trigger click
  if (isDragging.value) return;
  if (!player) return;

  clearClickTimeout();
  clickTimeout.value = setTimeout(() => {
    togglePolicyVisibility(player.id);
    clickTimeout.value = null;
  }, SINGLE_CLICK_DELAY);
}

function onPlayerDoubleClick(_event, player) {
  if (!player) return;
  clearClickTimeout();
  emit('update:activePlayerId', player.id);
}

function getSvgPoint(clientX, clientY) {
  if (!svgRef.value) return { x: 0, y: 0 };
  const pt = svgRef.value.createSVGPoint();
  pt.x = clientX;
  pt.y = clientY;
  return pt.matrixTransform(svgRef.value.getScreenCTM().inverse());
}

function onMouseDown(event, player) {
  if (!player) return;
  if (props.isManualStepping) return;
  event.preventDefault();
  
  draggedPlayerId.value = player.id;
  // Initialize drag position to player's current center
  draggedPlayerPos.value = { x: player.x, y: player.y };
  
  isDragging.value = false; // Not dragging yet until moved
  
  // Global listeners for drag/up to handle out-of-element movement
  window.addEventListener('mousemove', onGlobalMouseMove);
  window.addEventListener('mouseup', onGlobalMouseUp);
}

function onGlobalMouseMove(event) {
  if (draggedPlayerId.value !== null) {
    if (!isDragging.value) isDragging.value = true;
    const { x, y } = getSvgPoint(event.clientX, event.clientY);
    draggedPlayerPos.value = { x, y };
  }
}

function onGlobalMouseUp(event) {
  if (draggedPlayerId.value !== null && isDragging.value) {
    const { x, y } = getSvgPoint(event.clientX, event.clientY);
    
    // Find nearest hex
    let bestHex = null;
    let minDist = Infinity;
    
    for (const hex of courtLayout.value) {
      const dist = Math.sqrt((hex.x - x) ** 2 + (hex.y - y) ** 2);
      if (dist < minDist) {
        minDist = dist;
        bestHex = hex;
      }
    }
    
    if (bestHex && minDist < HEX_RADIUS * 1.5) { // Threshold to snap
      // Check if valid move
      const pid = draggedPlayerId.value;
      const currentPos = currentGameState.value.positions[pid];
      const newPos = [bestHex.q, bestHex.r];
      
      // Check occupancy (except self)
      const isOccupied = currentGameState.value.positions.some((p, idx) => 
        idx !== pid && p[0] === newPos[0] && p[1] === newPos[1]
      );
      
      if (!isOccupied && (currentPos[0] !== newPos[0] || currentPos[1] !== newPos[1])) {
         if (props.placementMode && props.placementEditable) {
           emit('update-placement', { playerId: pid, q: newPos[0], r: newPos[1] });
         } else if (!props.placementMode) {
           emit('update-player-position', { playerId: pid, q: newPos[0], r: newPos[1] });
         }
      }
    }
  }
  
  draggedPlayerId.value = null;
  isDragging.value = false;
  window.removeEventListener('mousemove', onGlobalMouseMove);
  window.removeEventListener('mouseup', onGlobalMouseUp);
}


const currentGameState = computed(() => {
  const base = props.gameHistory.length > 0 ? props.gameHistory[props.gameHistory.length - 1] : null;
  if (!base) return null;
  if (!props.placementMode) return base;
  const cloned = { ...base };
  if (Array.isArray(props.placementPositions) && props.placementPositions.length === (base.positions?.length || 0)) {
    cloned.positions = props.placementPositions.map((pos) => [pos[0], pos[1]]);
  }
  if (props.placementBallHolder !== null && props.placementBallHolder !== undefined) {
    cloned.ball_holder = props.placementBallHolder;
  }
  return cloned;
});

const allPoliciesVisible = computed(() => {
  const positions = currentGameState.value?.positions;
  if (!positions || positions.length === 0) return false;
  return policyVisibility.value.size === positions.length;
});

const offenseStateValue = computed(() => {
  const state = currentGameState.value;
  if (!state || !state.state_values) return null;
  const val = state.state_values.offensive_value;
  return typeof val === 'number' ? val : null;
});

const defenseStateValue = computed(() => {
  const state = currentGameState.value;
  if (!state || !state.state_values) return null;
  const val = state.state_values.defensive_value;
  return typeof val === 'number' ? val : null;
});

const sortedPlayers = computed(() => {
  const gs = currentGameState.value;
  if (!gs) return [];
  const players = getRenderablePlayers(gs);
  const activeId = props.activePlayerId;
  const ballHolderId = gs.ball_holder;

  const others = players.filter(
    (p) => p.id !== activeId && p.id !== ballHolderId
  );
  const ballHolderPlayer =
    ballHolderId !== undefined && ballHolderId !== null && ballHolderId !== activeId
      ? players.find((p) => p.id === ballHolderId)
      : null;
  const activePlayer = activeId !== null ? players.find((p) => p.id === activeId) : null;

  return [
    ...others,
    ...(ballHolderPlayer ? [ballHolderPlayer] : []),
    ...(activePlayer ? [activePlayer] : []),
  ];
});

function dribbleDelay(playerId) {
  return `${(playerId % 3) * 0.08}s`;
}

// no-op

const courtLayout = computed(() => {
    const hexes = [];
    if (!currentGameState.value) return [];
    for (let r_off = 0; r_off < currentGameState.value.court_height; r_off++) {
        for (let c_off = 0; c_off < currentGameState.value.court_width; c_off++) {
            const { q, r } = offsetToAxial(c_off, r_off);
            const { x, y } = axialToCartesian(q, r);
            hexes.push({ q, r, x, y, key: `${q},${r}` });
        }
    }
    return hexes;
});

const shotCountsMap = computed(() => props.shotAccumulator || {});
const shotChartLabel = computed(() => props.shotChartLabel || '');
const shotChartTitlePos = computed(() => {
  const vbString = viewBox.value;
  if (!vbString) return { x: 0, y: 0 };
  const parts = vbString.split(' ').map((val) => Number(val));
  if (parts.length !== 4 || parts.some((n) => Number.isNaN(n))) {
    return { x: 0, y: 0 };
  }
  const [minX, minY, width, height] = parts;
  const paddingX = HEX_RADIUS * 1.5;
  const paddingY = HEX_RADIUS * 0.8;
  return {
    x: minX + width - paddingX,
    y: minY + height - paddingY,
  };
});

const shotCountsNormalized = computed(() => {
  const out = {};
  const src = shotCountsMap.value || {};
  for (const [rawKey, val] of Object.entries(src)) {
    if (!Array.isArray(val) || val.length < 2) continue;
    const attempts = Number(val[0]) || 0;
    const makes = Number(val[1]) || 0;
    if (attempts <= 0 && makes <= 0) continue;
    const key = String(rawKey || '').trim();
    const [qStr, rStr] = key.split(',');
    const q = Number(qStr);
    const r = Number(rStr);
    if (Number.isNaN(q) || Number.isNaN(r)) continue;
    const normalizedKey = `${q},${r}`;
    out[normalizedKey] = { attempts, makes };
  }
  return out;
});

function getShotCount(q, r) {
  if (q === undefined || r === undefined || q === null || r === null) return null;
  const key = `${q},${r}`;
  return shotCountsNormalized.value[key] || null;
}

const shotCountList = computed(() => {
  const entries = Object.entries(shotCountsNormalized.value || {});
  entries.sort((a, b) => a[0].localeCompare(b[0]));
  return entries.map(([key, val]) => ({
    key,
    q: Number(key.split(',')[0]),
    r: Number(key.split(',')[1]),
    attempts: val.attempts,
    makes: val.makes,
  }));
});

const hasShotCounts = computed(() => !props.placementMode && shotCountList.value.length > 0);

const shotOverlayPoints = computed(() => {
  if (!hasShotCounts.value) return [];
  return shotCountList.value.map((entry) => {
    const { x, y } = axialToCartesian(entry.q, entry.r);
    return {
      key: entry.key,
      x,
      y,
      attempts: entry.attempts,
      makes: entry.makes,
    };
  });
});

const maxShotAttempts = computed(() => {
  let max = 0;
  for (const pt of shotOverlayPoints.value) {
    if (pt.attempts > max) max = pt.attempts;
  }
  return max || 1;
});

function volumeFill(att) {
  const t = Math.max(0, Math.min(1, att / maxShotAttempts.value));
  // Lerp from deep navy to accent orange
  const start = [15, 23, 42];     // dark base
  const end = [251, 146, 60];     // warm accent
  const r = Math.round(start[0] + (end[0] - start[0]) * t);
  const g = Math.round(start[1] + (end[1] - start[1]) * t);
  const b = Math.round(start[2] + (end[2] - start[2]) * t);
  return `rgba(${r}, ${g}, ${b}, ${0.75})`;
}

function volumeStroke(att) {
  const t = Math.max(0, Math.min(1, att / maxShotAttempts.value));
  // Slightly brighter stroke
  const start = [56, 189, 248];
  const end = [251, 191, 36];
  const r = Math.round(start[0] + (end[0] - start[0]) * t);
  const g = Math.round(start[1] + (end[1] - start[1]) * t);
  const b = Math.round(start[2] + (end[2] - start[2]) * t);
  return `rgba(${r}, ${g}, ${b}, 0.9)`;
}

function hexPointsFor(x, y, radius = HEX_RADIUS) {
  const pts = [];
  for (let i = 0; i < 6; i += 1) {
    const angleDeg = 60 * i + 30;
    const angleRad = (Math.PI / 180) * angleDeg;
    const px = x + radius * Math.cos(angleRad);
    const py = y + radius * Math.sin(angleRad);
    pts.push(`${px},${py}`);
  }
  return pts.join(' ');
}

const showPlayers = computed(() => !hasShotCounts.value);
const showValueAnnotations = computed(() => !hasShotCounts.value);

const shotLegendConfig = computed(() => {
  if (!hasShotCounts.value || courtLayout.value.length === 0) return null;
  const parts = viewBox.value.split(' ').map((v) => Number(v));
  if (parts.length !== 4 || parts.some((n) => Number.isNaN(n))) return null;
  const [minX, minY, width, height] = parts;
  const legendWidth = HEX_RADIUS * 10;
  const legendHeight = HEX_RADIUS * 1.5;
  const margin = HEX_RADIUS * 0.1;
  const x = minX + (width - legendWidth) / 2;
  const y = minY + height - legendHeight - margin;
  return { x, y, width: legendWidth, height: legendHeight };
});

watch(
  () => props.shotAccumulator,
  (val) => {
    try {
      const keys = val ? Object.keys(val) : [];
      console.log('[GameBoard] shotAccumulator updated, keys=', keys.length);
    } catch (_) { /* ignore */ }
  },
  { deep: true }
);

watch(
  () => shotCountList.value,
  (list) => {
    try {
      console.log('[GameBoard] shotCountList size=', list.length, 'sample=', list.slice(0, 3));
    } catch (_) { /* ignore */ }
  },
  { deep: true }
);

const basketPosition = computed(() => {
    if (!currentGameState.value) return { x: 0, y: 0 };
    const [q, r] = currentGameState.value.basket_position;
    // The basket axial coordinates already match the environment; no offset needed.
    return axialToCartesian(q, r);
});

// Action indicator configuration
// Position angles for hex faces (pointy-top hex)
// These are the angles from center to each hex face
const POSITION_ANGLES = {
  'MOVE_E':  0,      // Right
  'MOVE_NE': -60,    // Upper-right
  'MOVE_NW': -120,   // Upper-left
  'MOVE_W':  180,    // Left
  'MOVE_SW': 120,    // Lower-left
  'MOVE_SE': 60,     // Lower-right
  'PASS_E':  0,
  'PASS_NE': -60,
  'PASS_NW': -120,
  'PASS_W':  180,
  'PASS_SW': 120,
  'PASS_SE': 60,
};

// Icon rotation angles (matching HexagonControlPad)
const ICON_ROTATIONS = {
  'MOVE_E':  0 + 90,
  'MOVE_NE': -60 + 90,
  'MOVE_NW': -120 + 90,
  'MOVE_W':  180 + 90,
  'MOVE_SW': 120 + 90,
  'MOVE_SE': 60 + 90,
  'PASS_E':  0 + 90,
  'PASS_NE': -60 + 90,
  'PASS_NW': -120 + 90,
  'PASS_W':  180 + 90,
  'PASS_SW': 120 + 90,
  'PASS_SE': 60 + 90,
};

// Get action indicator data for a player
function getActionIndicator(playerId, playerX, playerY, hasBall) {
  const action = props.selectedActions[playerId];
  if (!action || action === 'NOOP') return null;
  
  const indicatorRadius = HEX_RADIUS * 0.55; // Distance from player center to indicator
  
  if (action.startsWith('MOVE_')) {
    const posAngle = POSITION_ANGLES[action];
    const rad = posAngle * Math.PI / 180;
    return {
      type: 'move',
      x: playerX + indicatorRadius * Math.cos(rad),
      y: playerY + indicatorRadius * Math.sin(rad),
      rotation: ICON_ROTATIONS[action],
    };
  }
  
  if (action.startsWith('PASS_') && hasBall) {
    const posAngle = POSITION_ANGLES[action];
    const rad = posAngle * Math.PI / 180;
    return {
      type: 'pass',
      x: playerX + indicatorRadius * Math.cos(rad),
      y: playerY + indicatorRadius * Math.sin(rad),
      rotation: ICON_ROTATIONS[action],
    };
  }
  
  if (action === 'SHOOT' && hasBall) {
    // Point toward basket
    const basket = basketPosition.value;
    const dx = basket.x - playerX;
    const dy = basket.y - playerY;
    const posAngle = Math.atan2(dy, dx) * 180 / Math.PI;
    const rad = Math.atan2(dy, dx);
    return {
      type: 'shoot',
      x: playerX + indicatorRadius * Math.cos(rad),
      y: playerY + indicatorRadius * Math.sin(rad),
      rotation: 0, // Target icon doesn't need rotation
    };
  }
  
  return null;
}

// Board transform - no flip needed, render as-is
const boardTransform = computed(() => {
  return '';
});

const threePointQualifiedSet = computed(() => {
  const gs = currentGameState.value;
  if (!gs || !gs.three_point_hexes) return new Set();
  return new Set(gs.three_point_hexes.map(([q, r]) => `${q},${r}`));
});

const buildArcPoints = (hoop, radius, startAngle, endAngle, steps = 160) => {
  const pts = [];
  const dir = endAngle >= startAngle ? 1 : -1;
  const total = Math.abs(endAngle - startAngle);
  for (let i = 0; i <= steps; i += 1) {
    const t = startAngle + dir * (i / steps) * total;
    const x = hoop.x + radius * Math.cos(t);
    const y = hoop.y + radius * Math.sin(t);
    pts.push({ x, y });
  }
  return pts;
};

const threePointArcPath = computed(() => {
  const gs = currentGameState.value;
  const hoop = basketPosition.value;
  if (!gs || !hoop) return '';
  const radiusPx = (gs.three_point_distance ?? 5) * HEX_RADIUS * SQRT3;
  if (radiusPx <= 0) return '';

  const segs = [];
  const shortDist = gs.three_point_short_distance;

  if (shortDist === null || shortDist === undefined) {
    const pts = buildArcPoints(hoop, radiusPx, Math.PI / 2, -Math.PI / 2, 240);
    pts.forEach((pt, idx) => {
      segs.push(`${idx === 0 ? 'M' : 'L'} ${pt.x} ${pt.y}`);
    });
    return segs.join(' ');
  }

  const shortPxRaw = shortDist * HEX_RADIUS * SQRT3;
  const shortPx = Math.min(shortPxRaw, radiusPx * 0.999);
  const connectX = Math.sqrt(Math.max(radiusPx * radiusPx - shortPx * shortPx, 0));
  const theta = Math.asin(shortPx / radiusPx);

  segs.push(`M ${hoop.x} ${hoop.y - shortPx}`);
  segs.push(`L ${hoop.x + connectX} ${hoop.y - shortPx}`);

  const arcPts = buildArcPoints(hoop, radiusPx, -theta, theta, 200);
  arcPts.forEach((pt) => segs.push(`L ${pt.x} ${pt.y}`));

  segs.push(`L ${hoop.x} ${hoop.y + shortPx}`);

  return segs.join(' ');
});

const offensiveLaneHexes = computed(() => {
  const gs = currentGameState.value;
  if (!gs || !gs.offensive_three_seconds_enabled || !gs.offensive_lane_hexes) return [];
  
  return gs.offensive_lane_hexes.map(([q, r]) => {
    const { x, y } = axialToCartesian(q, r);
    return { x, y, key: `lane-${q},${r}` };
  });
});

const shotClockValue = computed(() => currentGameState.value?.shot_clock ?? 0);
const shotClockMax = computed(() => {
  const state = currentGameState.value;
  if (!state) return 24;
  const candidates = [24];
  const maxParam = Number(state.shot_clock_steps);
  const current = Number(state.shot_clock);
  if (!Number.isNaN(maxParam)) candidates.push(maxParam);
  if (!Number.isNaN(current)) candidates.push(current);
  return Math.max(...candidates);
});
const isShotClockEditable = computed(() => !!currentGameState.value && !currentGameState.value.done);
const canIncrementShotClock = computed(
  () => isShotClockEditable.value && !props.isShotClockUpdating && shotClockValue.value < shotClockMax.value
);
const canDecrementShotClock = computed(
  () => isShotClockEditable.value && !props.isShotClockUpdating && shotClockValue.value > 0
);
const pendingShotClock = ref(null);
const displayedShotClockValue = computed(() => {
  if (props.isShotClockUpdating && pendingShotClock.value !== null) {
    return pendingShotClock.value;
  }
  return shotClockValue.value;
});

watch(shotClockValue, (newVal) => {
  if (!props.isShotClockUpdating) {
    pendingShotClock.value = null;
  }
});

watch(() => props.isShotClockUpdating, (updating) => {
  if (!updating) {
    pendingShotClock.value = null;
  }
});

// Fetch pass steal probabilities when game state changes
watch(currentGameState, async (newState) => {
  if (!newState || newState.ball_holder === null || newState.ball_holder === undefined) {
    passStealProbs.value = {};
    return;
  }
  
  // If the state has stored pass steal probabilities (from replay or recorded episode), use those
  if (newState.pass_steal_probabilities) {
    passStealProbs.value = newState.pass_steal_probabilities;
    console.log('[GameBoard] Using stored pass steal probabilities from state:', newState.pass_steal_probabilities);
    return;
  }
  
  // During manual stepping without stored probs, keep existing values (don't fetch)
  if (props.isManualStepping) {
    return;
  }
  
  // During live gameplay, fetch from API
  try {
    const probs = await getPassStealProbabilities();
    passStealProbs.value = probs || {};
  } catch (err) {
    console.error('[GameBoard] Failed to fetch pass steal probabilities:', err);
    passStealProbs.value = {};
  }
}, { immediate: true });

// Reset policy visibility when the game history is cleared (new game)
watch(
  () => props.gameHistory.length,
  (len) => {
    if (len === 0) {
      policyVisibility.value = new Set();
    }
  }
);

// Keep policy visibility aligned to existing player IDs
watch(
  () => currentGameState.value?.positions?.length,
  () => {
    const positions = currentGameState.value?.positions;
    if (!positions || positions.length === 0) {
      policyVisibility.value = new Set();
      return;
    }
    const validIds = new Set(positions.map((_, idx) => idx));
    const filtered = new Set([...policyVisibility.value].filter((id) => validIds.has(id)));
    if (filtered.size !== policyVisibility.value.size) {
      policyVisibility.value = filtered;
    }
  }
);

function adjustShotClock(delta) {
  if (!isShotClockEditable.value) return;
  const minClock = 0;
  const maxClock = shotClockMax.value;
  const baseValue = pendingShotClock.value !== null ? pendingShotClock.value : shotClockValue.value;
  const newValue = Math.max(minClock, Math.min(maxClock, baseValue + delta));
  if (newValue === baseValue) return;
  pendingShotClock.value = newValue;
  emit('adjust-shot-clock', delta);
}

const viewBox = computed(() => {
    if (courtLayout.value.length === 0) return "-100 -100 200 200";
    
    const allX = courtLayout.value.map(h => h.x);
    const allY = courtLayout.value.map(h => h.y);
    
    const margin = HEX_RADIUS * 3; // Increased margin for more padding
    const minX = Math.min(...allX) - margin;
    const maxX = Math.max(...allX) + margin;
    const minY = Math.min(...allY) - margin;
    const maxY = Math.max(...allY) + margin;

    const width = maxX - minX;
    const height = maxY - minY;

    return `${minX} ${minY} ${width} ${height}`;
});

const courtCenter = computed(() => {
  if (courtLayout.value.length === 0) return { x: 0, y: 0 };
  const xs = courtLayout.value.map((h) => h.x);
  const ys = courtLayout.value.map((h) => h.y);
  return {
    x: (Math.min(...xs) + Math.max(...xs)) / 2,
    y: (Math.min(...ys) + Math.max(...ys)) / 2,
  };
});

const courtBounds = computed(() => {
  if (courtLayout.value.length === 0) return { minX: 0, maxX: 0, minY: 0, maxY: 0 };
  const xs = courtLayout.value.map((h) => h.x);
  const ys = courtLayout.value.map((h) => h.y);
  return {
    minX: Math.min(...xs),
    maxX: Math.max(...xs),
    minY: Math.min(...ys),
    maxY: Math.max(...ys),
  };
});

const basketMarkerPosition = computed(() => {
  const basket = basketPosition.value;
  return {
    x: basket.x - HEX_RADIUS * 1.4,
    y: basket.y,
  };
});

const halfcourtMarkerPosition = computed(() => {
  return {
    x: courtBounds.value.maxX + HEX_RADIUS * 1.4,
    y: courtCenter.value.y,
  };
});

const rightSidelineMarkerPosition = computed(() => {
  return {
    x: courtCenter.value.x,
    y: courtBounds.value.minY - HEX_RADIUS * 1.6,
  };
});

const leftSidelineMarkerPosition = computed(() => {
  return {
    x: courtCenter.value.x,
    y: courtBounds.value.maxY + HEX_RADIUS * 1.6,
  };
});

const stateValueBoxWidth = HEX_RADIUS * 3;
const stateValueBoxBaseHeight = HEX_RADIUS * 1.3;
const stateValueBoxHeight = computed(() =>
  defenseStateValue.value !== null ? stateValueBoxBaseHeight * 2 : stateValueBoxBaseHeight
);

const stateValueAnchor = computed(() => {
  const vbString = viewBox.value;
  if (!vbString) return { x: 0, y: 0 };
  const parts = vbString.split(' ').map((val) => Number(val));
  if (parts.length !== 4 || parts.some((n) => Number.isNaN(n))) {
    return { x: 0, y: 0 };
  }
  const [minX, minY, width, height] = parts;
  const padding = HEX_RADIUS * 0.75;
  return {
    x: minX + width - stateValueBoxWidth - padding,
    // push it even lower to avoid overlapping the court area
    y: minY + height - stateValueBoxHeight.value - padding - HEX_RADIUS * 0.3,
  };
});

// This is a direct mapping from our ActionType enum for moves
const moveActionIndices = {
    MOVE_E: 1, MOVE_NE: 2, MOVE_NW: 3, MOVE_W: 4, MOVE_SW: 5, MOVE_SE: 6
};
const hexDirections = [
    {q: +1, r:  0}, {q: +1, r: -1}, {q:  0, r: -1}, 
    {q: -1, r:  0}, {q: -1, r: +1}, {q:  0, r: +1}
];

const policyProbsForDisplay = computed(() => {
  const gs = currentGameState.value;
  if (props.isManualStepping && gs?.policy_probabilities) {
    return gs.policy_probabilities;
  }
  return props.policyProbabilities;
});

function getPolicyProbsForPlayer(playerId) {
  const probs = policyProbsForDisplay.value;
  return probs?.[playerId] ?? probs?.[String(playerId)] ?? null;
}

function playerLabelTransform(player) {
  const isDragged = draggedPlayerId.value === player.id;
  const x = isDragged ? draggedPlayerPos.value.x : player.x;
  const y = isDragged ? draggedPlayerPos.value.y : player.y;
  return `translate(${x}, ${y})`;
}

function probToAlpha(prob) {
  const val = Number(prob);
  if (Number.isNaN(val)) return 0;
  // Keep very small probabilities faint but still visible
  return Math.max(0.2, Math.min(1, val));
}

function probToStealAlpha(prob) {
  const val = Number(prob);
  if (Number.isNaN(val)) return 0;
  // Accept values as fractions (0-1) or percents (0-100)
  const normalized = Math.max(0, Math.min(1, val > 1 ? val / 100 : val));
  // Slight gamma curve to widen contrast while keeping a visible floor
  const scaled = Math.pow(normalized, 0.7);
  return Math.max(0.1, Math.min(1, scaled));
}

const policySuggestions = computed(() => {
  const gs = currentGameState.value;
  if (!gs) return [];

  const visibleIds = Array.from(policyVisibility.value);
  if (visibleIds.length === 0) return [];

  const probsByPlayer = policyProbsForDisplay.value;
  if (!probsByPlayer) return [];

  const suggestions = [];

  for (const pid of visibleIds) {
    const probs = getPolicyProbsForPlayer(pid);
    const currentPlayerPos = gs.positions?.[pid];
    const mask = gs.action_mask?.[pid];
    if (!probs || !currentPlayerPos) continue;

    for (let i = 0; i < hexDirections.length; i++) {
      const dir = hexDirections[i];
      const moveActionIndex = i + 1; // MOVE_E .. MOVE_SE
      const passActionIndex = 8 + i; // PASS_E .. PASS_SE

      const targetPos = { q: currentPlayerPos[0] + dir.q, r: currentPlayerPos[1] + dir.r };
      const cartesianPos = axialToCartesian(targetPos.q, targetPos.r);
      const moveAllowed = Array.isArray(mask) ? mask[moveActionIndex] > 0 : true;
      const passAllowed = Array.isArray(mask) ? mask[passActionIndex] > 0 : true;
      const moveProb = moveAllowed ? (probs[moveActionIndex] ?? 0) : null;
      const passProb = passAllowed ? (probs[passActionIndex] ?? 0) : null;
      const moveOpacity = moveProb !== null ? probToAlpha(moveProb) : 0;
      const passOpacity = passProb !== null ? probToAlpha(passProb) : 0;

      suggestions.push({
        x: cartesianPos.x,
        y: cartesianPos.y,
        moveProb,
        passProb,
        moveOpacity,
        passOpacity,
        key: `sugg-${pid}-${i}`
      });
    }
  }

  return suggestions;
});

const ballHandlerShotProb = computed(() => {
    if (!currentGameState.value || currentGameState.value.ball_holder === null || !policyProbsForDisplay.value) {
        return null;
    }
    const ballHolderId = currentGameState.value.ball_holder;
    const probs = getPolicyProbsForPlayer(ballHolderId);
    if (!probs) {
        return null;
    }
    // From ActionType enum in the backend, SHOOT is at index 7
    return probs[7];
});

// Backend-calculated conditional make probability for the ball handler (pressure-adjusted)
const ballHandlerMakeProb = ref(null);

async function fetchBallHandlerMakeProb() {
  const gs = currentGameState.value;
  if (!gs || gs.ball_holder === null) {
    ballHandlerMakeProb.value = null;
    return;
  }
  
  // During manual stepping (replay), use the stored shot probability from the state
  if (props.isManualStepping && typeof gs.ball_handler_shot_probability === 'number') {
    ballHandlerMakeProb.value = gs.ball_handler_shot_probability;
    console.log('[GameBoard] Using stored shot probability from replay state:', gs.ball_handler_shot_probability);
    return;
  }
  
  // Otherwise, fetch from API (live gameplay)
  try {
    const resp = await getShotProbability(gs.ball_holder);
    const p = resp?.shot_probability_final ?? resp?.shot_probability ?? null;
    ballHandlerMakeProb.value = typeof p === 'number' ? p : null;
  } catch (e) {
    console.warn('[GameBoard] Failed to fetch ball-handler make prob', e);
    ballHandlerMakeProb.value = null;
  }
}

onMounted(() => {
  fetchBallHandlerMakeProb();
});

watch(
  () => ({
    ball_holder: currentGameState.value?.ball_holder,
    positions: currentGameState.value?.positions,
    shot_clock: currentGameState.value?.shot_clock,
    isManualStepping: props.isManualStepping,
  }),
  () => {
    fetchBallHandlerMakeProb();
  },
  { deep: true }
);

const episodeOutcome = computed(() => {
    if (!currentGameState.value || !currentGameState.value.done) {
        return null; // Game is not over
    }

    const results = currentGameState.value.last_action_results;
    if (!results) return null;

    // Check for shot results
    if (results.shots && Object.keys(results.shots).length > 0) {
        const shooterId = Object.keys(results.shots)[0];
        const shotResult = results.shots[shooterId];
        const pid = parseInt(shooterId, 10);
        const pos = currentGameState.value.positions[pid];
        // isThree comes directly from the backend shot result
        const isThree = shotResult.is_three;
        const [q, r] = pos;
        const bq = currentGameState.value.basket_position[0];
        const br = currentGameState.value.basket_position[1];
        // Simple distance for dunk check
        const dist = (Math.abs(q - bq) + Math.abs((q + r) - (bq + br)) + Math.abs(r - br)) / 2;
        const isDunk = dist === 0;
        return {
            type: shotResult.success ? 'MADE_SHOT' : 'MISSED_SHOT',
            isThree,
            isDunk,
            playerId: pid,
        };
    }

    // Check for defensive lane violations
    if (results.defensive_lane_violations && results.defensive_lane_violations.length > 0) {
        const violation = results.defensive_lane_violations[0];
        if (violation.position) {
            const { x, y } = axialToCartesian(violation.position[0], violation.position[1]);
            return { type: 'DEFENSIVE_VIOLATION', x, y, playerId: violation.player_id };
        }
        return { type: 'DEFENSIVE_VIOLATION' };
    }

    // Check for turnover results
    let allTurnovers = results.turnovers ? [...results.turnovers] : [];
    if (results.passes) {
        for (const pass_res of Object.values(results.passes)) {
            if (pass_res.turnover) {
                allTurnovers.push(pass_res);
            }
        }
    }

    if (allTurnovers.length > 0 && allTurnovers[0].turnover_pos) {
        const { x, y } = axialToCartesian(allTurnovers[0].turnover_pos[0], allTurnovers[0].turnover_pos[1]);
        return { type: 'TURNOVER', x, y };
    }

    // Check for shot clock violation
    if (currentGameState.value.shot_clock <= 0) {
        return { type: 'SHOT_CLOCK_VIOLATION' };
    }

    return null; // No definitive outcome found
});

const playerTransitions = computed(() => {
  if (props.gameHistory.length < 2) {
    return [];
  }
  const transitions = [];
  const progress = Math.max(0, Math.min(1, Number(props.moveProgress ?? 1)));
  // Start from the second state, as moves happen between states.
  for (let step = 1; step < props.gameHistory.length; step++) {
    const previousGameState = props.gameHistory[step - 1];
    const currentGameState = props.gameHistory[step];
    const isLastStep = step === props.gameHistory.length - 1;
    // Opacity should match the destination ghost cell's opacity.
    const opacity = 0.1 + (0.2 * (step - 1) / (props.gameHistory.length - 1));

    for (let playerId = 0; playerId < currentGameState.positions.length; playerId++) {
      const prevPos = previousGameState.positions[playerId];
      const currentPos = currentGameState.positions[playerId];

      // Check if the position has changed
      if (prevPos[0] !== currentPos[0] || prevPos[1] !== currentPos[1]) {
        const { x: startX, y: startY } = axialToCartesian(prevPos[0], prevPos[1]);
        const { x: fullEndX, y: fullEndY } = axialToCartesian(currentPos[0], currentPos[1]);
        const endX = isLastStep && progress < 1 ? startX + (fullEndX - startX) * progress : fullEndX;
        const endY = isLastStep && progress < 1 ? startY + (fullEndY - startY) * progress : fullEndY;
        
        const isOffense = currentGameState.offense_ids.includes(playerId);

        transitions.push({
          key: `arrow-${step}-${playerId}`,
          startX,
          startY,
          endX,
          endY,
          opacity,
          isOffense,
        });
      }
    }
  }
  return transitions;
});

// Compute pass rays from ball handler to teammates with steal probabilities
const passRays = computed(() => {
  const gs = currentGameState.value;
  if (!gs || gs.ball_holder === null || gs.ball_holder === undefined) return [];
  
  const ballHandlerId = gs.ball_holder;
  const ballHandlerPos = gs.positions[ballHandlerId];
  if (!ballHandlerPos) return [];
  
  const [bhQ, bhR] = ballHandlerPos;
  const bhCoords = axialToCartesian(bhQ, bhR);
  
  const isOffense = gs.offense_ids.includes(ballHandlerId);
  const teamIds = isOffense ? gs.offense_ids : gs.defense_ids;
  
  const rays = [];
  const usePlacementPreview = !!(props.placementMode && props.placementEditable && props.placementPassProbs && Object.keys(props.placementPassProbs || {}).length);
  const stealMap = usePlacementPreview ? props.placementPassProbs : (passStealProbs.value || {});
  for (const teammateId of teamIds) {
    if (teammateId === ballHandlerId) continue;
    
    const teammatePos = gs.positions[teammateId];
    if (!teammatePos) continue;
    
    const [tmQ, tmR] = teammatePos;
    const tmCoords = axialToCartesian(tmQ, tmR);
    
    const stealProbRaw = stealMap ? stealMap[teammateId] : null;
    const hasProb = stealProbRaw !== null && stealProbRaw !== undefined && !Number.isNaN(Number(stealProbRaw));
    // In live mode, if no probability is available, skip drawing this ray (preserves old behavior)
    if (!hasProb && !usePlacementPreview) continue;
    const stealProb = hasProb ? Number(stealProbRaw) : 0;
    const probFraction = stealProb > 1 ? stealProb / 100 : stealProb;
    const stealPercent = hasProb ? probFraction * 100 : null;
    const stealLabel = hasProb ? `${stealPercent.toFixed(1)}%` : '—';
    const stealOpacity = hasProb ? probToStealAlpha(probFraction) : 0.3;
    
    // Calculate midpoint for label placement
    const midX = (bhCoords.x + tmCoords.x) / 2;
    const midY = (bhCoords.y + tmCoords.y) / 2;
    
    rays.push({
      x1: bhCoords.x,
      y1: bhCoords.y,
      x2: tmCoords.x,
      y2: tmCoords.y,
      midX,
      midY,
      stealProb: stealPercent, // Rounded to nearest percent (nullable)
      stealLabel,
      stealOpacity,
      teammateId,
    });
  }
  
  return rays;
});

// Preview which teammate will receive a pass based on current selection/strategy
const passTargetPreview = computed(() => {
  const gs = currentGameState.value;
  if (!gs || gs.ball_holder === null || gs.ball_holder === undefined) return null;

  const passerId = gs.ball_holder;
  const action = props.selectedActions?.[passerId];
  if (!action || !action.startsWith('PASS_')) return null;

  const dirIdx = PASS_ACTION_TO_DIR[action];
  if (dirIdx === undefined) return null;

  const passerPos = gs.positions?.[passerId];
  if (!passerPos) return null;

  const arcDegrees = gs.pass_arc_degrees ?? 60;
  const halfAngleRad = (Math.max(1, Math.min(360, arcDegrees)) * Math.PI) / 360;
  const dirVec = HEX_DIRECTIONS[dirIdx];
  const dirCart = axialToCartesian(dirVec[0], dirVec[1]);
  const dirNorm = Math.hypot(dirCart.x, dirCart.y) || 1;
  const cosThreshold = Math.cos(halfAngleRad) - PASS_COS_EPS;

  const inArc = (targetPos) => {
    const [tq, tr] = targetPos;
    const vx = tq - passerPos[0];
    const vy = tr - passerPos[1];
    const vCart = axialToCartesian(vx, vy);
    const vNorm = Math.hypot(vCart.x, vCart.y);
    if (vNorm === 0) return false;
    const cosang = (vCart.x * dirCart.x + vCart.y * dirCart.y) / (vNorm * dirNorm);
    return cosang >= cosThreshold;
  };

  const isOffense = gs.offense_ids?.includes(passerId);
  const teamIds = isOffense ? gs.offense_ids : gs.defense_ids;
  if (!teamIds) return null;

  const strategy = (gs.pass_target_strategy || 'nearest').toLowerCase();
  let best = null;

  for (const tid of teamIds) {
    if (tid === passerId) continue;
    const tPos = gs.positions?.[tid];
    if (!tPos || !inArc(tPos)) continue;

    const distance = hexDistance(passerPos, tPos);

    if (strategy === 'best_ev') {
      const ep = (gs.ep_by_player && gs.ep_by_player[tid] !== undefined)
        ? Number(gs.ep_by_player[tid])
        : 0;
      const stealProb = passStealProbs.value?.[tid] ?? 0;
      const value = (1 - stealProb) * ep;
      const candidate = {
        receiverId: tid,
        distance,
        value,
        stealProb,
        ep,
      };
      if (
        !best ||
        candidate.value > best.value + 1e-9 ||
        (Math.abs(candidate.value - best.value) < 1e-9 && distance < best.distance) ||
        (Math.abs(candidate.value - best.value) < 1e-9 && distance === best.distance && tid < best.receiverId)
      ) {
        best = candidate;
      }
    } else {
      const candidate = { receiverId: tid, distance };
      if (!best || distance < best.distance || (distance === best.distance && tid < best.receiverId)) {
        best = candidate;
      }
    }
  }

  if (!best) return null;

  const passerCoords = axialToCartesian(passerPos[0], passerPos[1]);
  const recvPos = gs.positions[best.receiverId];
  const recvCoords = axialToCartesian(recvPos[0], recvPos[1]);

  return {
    passerId,
    receiverId: best.receiverId,
    start: passerCoords,
    end: recvCoords,
    strategy,
    distance: best.distance,
    value: best.value ?? null,
    stealProb: best.stealProb ?? null,
    ep: best.ep ?? null,
  };
});

function clearPassFlash() {
  if (passFlashTimeout.value) {
    clearTimeout(passFlashTimeout.value);
    passFlashTimeout.value = null;
  }
  passFlash.value = null;
}

function triggerPassFlash(passerId, receiverId, start, end) {
  if (passFlashTimeout.value) {
    clearTimeout(passFlashTimeout.value);
    passFlashTimeout.value = null;
  }

  passFlash.value = {
    passerId,
    receiverId,
    x1: start.x,
    y1: start.y,
    x2: end.x,
    y2: end.y,
    labelX: (start.x + end.x) / 2,
    labelY: (start.y + end.y) / 2 - HEX_RADIUS * 0.6,
  };

  passFlashTimeout.value = setTimeout(() => {
    passFlash.value = null;
    passFlashTimeout.value = null;
  }, PASS_FLASH_DURATION_MS);
}

function buildShotArcPath(start, end) {
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  const distance = Math.hypot(dx, dy) || 1;

  const midX = (start.x + end.x) / 2;
  const midY = (start.y + end.y) / 2;

  // Perpendicular direction to shot line
  const perpX = -dy / distance;
  const perpY = dx / distance;

  // Flip curvature as shooter crosses the court midline (sideline to sideline)
  const sideSign = start.y >= courtCenter.value.y ? -1 : 1;
  // Add arc height that scales with shot distance so deeper shots arc more
  const arcHeight = Math.min(HEX_RADIUS * 8, HEX_RADIUS * 0.4 + distance * 0.25);

  const controlX = midX + perpX * arcHeight * sideSign;
  const controlY = midY + perpY * arcHeight * sideSign;

  return `M ${start.x} ${start.y} Q ${controlX} ${controlY} ${end.x} ${end.y}`;
}

function clearShotFlash() {
  if (shotFlashTimeout.value) {
    clearTimeout(shotFlashTimeout.value);
    shotFlashTimeout.value = null;
  }
  shotFlash.value = null;
}

function clearShotJump() {
  if (shotJumpTimeout.value) {
    clearTimeout(shotJumpTimeout.value);
    shotJumpTimeout.value = null;
  }
  shotJumpPlayerId.value = null;
  shotJumpIsDunk.value = false;
}

function triggerShotJump(shooterId, isDunk = false) {
  if (shooterId === null || shooterId === undefined) return;
  if (shotJumpTimeout.value) {
    clearTimeout(shotJumpTimeout.value);
    shotJumpTimeout.value = null;
  }
  shotJumpPlayerId.value = shooterId;
  shotJumpIsDunk.value = !!isDunk;
  shotJumpTimeout.value = setTimeout(() => {
    clearShotJump();
  }, SHOOT_JUMP_PERIOD_SECONDS * 1000);
}

function triggerShotFlash(shooterId, start, end, success, isDunk = false) {
  if (shotFlashTimeout.value) {
    clearTimeout(shotFlashTimeout.value);
    shotFlashTimeout.value = null;
  }

  shotFlash.value = {
    shooterId,
    x1: start.x,
    y1: start.y,
    x2: end.x,
    y2: end.y,
    color: success ? '#22c55e' : '#ef4444',
    path: buildShotArcPath(start, end),
  };

  shotFlashTimeout.value = setTimeout(() => {
    shotFlash.value = null;
    shotFlashTimeout.value = null;
  }, SHOT_FLASH_DURATION_MS);
  triggerShotJump(shooterId, isDunk);
}

watch(
  currentGameState,
  (state) => {
    if (!state) {
      clearPassFlash();
      return;
    }

    const passes = state.last_action_results?.passes;
    if (!passes || Object.keys(passes).length === 0) {
      clearPassFlash();
      return;
    }

    let successfulPass = null;
    for (const [passerId, passResult] of Object.entries(passes)) {
      if (passResult && passResult.success && typeof passResult.target === 'number') {
        successfulPass = { passerId: Number(passerId), receiverId: Number(passResult.target) };
        break;
      }
    }

    if (!successfulPass) {
      clearPassFlash();
      return;
    }

    const passerPos = state.positions?.[successfulPass.passerId];
    const receiverPos = state.positions?.[successfulPass.receiverId];
    if (!passerPos || !receiverPos) {
      clearPassFlash();
      return;
    }

    const start = axialToCartesian(passerPos[0], passerPos[1]);
    const end = axialToCartesian(receiverPos[0], receiverPos[1]);
    triggerPassFlash(successfulPass.passerId, successfulPass.receiverId, start, end);
  },
  { immediate: true }
);

watch(
  currentGameState,
  (state) => {
    if (!state) {
      clearShotFlash();
      clearShotJump();
      return;
    }

    const shots = state.last_action_results?.shots;
    if (!shots || Object.keys(shots).length === 0) {
      clearShotFlash();
      clearShotJump();
      return;
    }

    let shotData = null;
    for (const [shooterId, shotResult] of Object.entries(shots)) {
      if (shotResult) {
        shotData = { shooterId: Number(shooterId), result: shotResult };
        break;
      }
    }

    if (!shotData) {
      clearShotFlash();
      clearShotJump();
      return;
    }

    const shooterPos = state.positions?.[shotData.shooterId];
    const basketPos = state.basket_position;
    if (!shooterPos || !basketPos) {
      clearShotFlash();
      clearShotJump();
      return;
    }

    const start = axialToCartesian(shooterPos[0], shooterPos[1]);
    const end = axialToCartesian(basketPos[0], basketPos[1]);
    const success = !!shotData.result.success;
    const isDunk =
      (shotData.result && typeof shotData.result.is_dunk === 'boolean' && shotData.result.is_dunk) ||
      hexDistance(shooterPos, basketPos) === 0;
    triggerShotFlash(shotData.shooterId, start, end, success, isDunk);
  },
  { immediate: true }
);

async function downloadBoardAsImage() {
  if (!svgRef.value) return;
  
  try {
    // Helper to inline computed styles from source to target
    const inlineStyles = (source, target) => {
      const computed = window.getComputedStyle(source);
      const properties = [
        'fill', 'stroke', 'stroke-width', 'stroke-dasharray',
        'opacity', 'font-family', 'font-size', 'font-weight',
        'text-anchor', 'dominant-baseline', 'paint-order',
        'transform', 'transform-origin', 'transform-box'
      ];
      properties.forEach(prop => {
        // Only set if not default/empty to keep it clean, 
        // but essential for class-based styles to persist
        const val = computed.getPropertyValue(prop);
        if (val) target.style[prop] = val;
      });
      
      for (let i = 0; i < source.children.length; i++) {
        if (target.children[i]) {
          inlineStyles(source.children[i], target.children[i]);
        }
      }
    };

    // Clone the SVG to avoid modifying the original
    const svgClone = svgRef.value.cloneNode(true);
    
    // Inline styles to ensure they are captured (since classes won't work in standalone SVG)
    inlineStyles(svgRef.value, svgClone);
    
    // Get the viewBox dimensions
    const viewBox = svgRef.value.getAttribute('viewBox').split(' ').map(Number);
    const [minX, minY, width, height] = viewBox;
    
    // Set explicit width and height for rendering
    svgClone.setAttribute('width', width);
    svgClone.setAttribute('height', height);
    
    // Serialize the SVG to a string
    const serializer = new XMLSerializer();
    let svgString = serializer.serializeToString(svgClone);
    
    // Add XML declaration and ensure proper encoding
    svgString = '<?xml version="1.0" encoding="UTF-8"?>' + svgString;
    
    // Create a blob from the SVG string
    const svgBlob = new Blob([svgString], { type: 'image/svg+xml;charset=utf-8' });
    const url = URL.createObjectURL(svgBlob);
    
    // Create an image element to load the SVG
    const img = new Image();
    
    // Capture state variables before async operations to ensure consistency
    const shotClock = currentGameState.value?.shot_clock;
    const hasShotClock = shotClock !== undefined && shotClock !== null;
    const shouldDrawShotClock = !hasShotCounts.value && hasShotClock;
    const shotClockVal = String(shotClock);

    img.onload = () => {
      // Create a canvas with the SVG dimensions
      const canvas = document.createElement('canvas');
      const scale = 2; // Higher resolution
      canvas.width = width * scale;
      canvas.height = height * scale;
      
      const ctx = canvas.getContext('2d');
      
      // Fill with dark background to match web app
      ctx.fillStyle = '#0a0f1e';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      
      // Draw the SVG onto the canvas (scaled up)
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      
      // Draw Shot Clock if available and not in shot overlay mode
      if (shouldDrawShotClock) {
        const fontSize = 48 * scale;
        const paddingX = 16 * scale;
        const paddingY = 4 * scale;
        const margin = 20 * scale;
        
        ctx.font = `${fontSize}px "DSEG7 Classic", monospace`;
        const textMetrics = ctx.measureText(shotClockVal);
        const textWidth = textMetrics.width;
        const boxWidth = textWidth + (paddingX * 2);
        const boxHeight = fontSize + (paddingY * 2);
        
        const x = canvas.width - boxWidth - margin;
        const y = margin;
        
        ctx.fillStyle = '#1a1a1a';
        ctx.fillRect(x, y, boxWidth, boxHeight);
        ctx.strokeStyle = '#333';
        ctx.lineWidth = 2 * scale;
        ctx.strokeRect(x, y, boxWidth, boxHeight);
        
        ctx.fillStyle = '#ff4d4d';
        ctx.shadowColor = '#ff4d4d';
        ctx.shadowBlur = 10 * scale;
        ctx.textBaseline = 'top';
        ctx.fillText(shotClockVal, x + paddingX, y + paddingY);
        ctx.shadowBlur = 0;
      }
      
      // Convert canvas to PNG and download (PNG supports transparency)
      canvas.toBlob((blob) => {
        const link = document.createElement('a');
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
        link.download = `basketworld-board-${timestamp}.png`;
        link.href = URL.createObjectURL(blob);
        link.click();
        
        // Cleanup
        URL.revokeObjectURL(url);
        URL.revokeObjectURL(link.href);
      }, 'image/png');
    };
    
    img.src = url;
  } catch (err) {
    console.error('[GameBoard] Failed to download image:', err);
    alert('Failed to download board image');
  }
}

// Expose method to render current state as PNG (for episode saving)
async function renderStateToPng() {
  if (!svgRef.value) return null;
  
  try {
    // Helper to inline computed styles
    const inlineStyles = (source, target) => {
      const computed = window.getComputedStyle(source);
      const properties = [
        'fill', 'stroke', 'stroke-width', 'stroke-dasharray',
        'opacity', 'font-family', 'font-size', 'font-weight',
        'text-anchor', 'dominant-baseline', 'paint-order',
        'transform', 'transform-origin', 'transform-box'
      ];
      properties.forEach(prop => {
        const val = computed.getPropertyValue(prop);
        if (val) target.style[prop] = val;
      });
      
      for (let i = 0; i < source.children.length; i++) {
        if (target.children[i]) {
          inlineStyles(source.children[i], target.children[i]);
        }
      }
    };

    const svgClone = svgRef.value.cloneNode(true);
    inlineStyles(svgRef.value, svgClone);
    
    const viewBox = svgRef.value.getAttribute('viewBox').split(' ').map(Number);
    const [minX, minY, width, height] = viewBox;
    
    svgClone.setAttribute('width', width);
    svgClone.setAttribute('height', height);
    
    const serializer = new XMLSerializer();
    let svgString = serializer.serializeToString(svgClone);
    svgString = '<?xml version="1.0" encoding="UTF-8"?>' + svgString;
    
    const svgBlob = new Blob([svgString], { type: 'image/svg+xml;charset=utf-8' });
    const url = URL.createObjectURL(svgBlob);
    
    // Return a promise that resolves with the PNG data URL
    return new Promise((resolve, reject) => {
      const img = new Image();
      const shotClock = currentGameState.value?.shot_clock;
      const hasShotClock = shotClock !== undefined && shotClock !== null;
      const shouldDrawShotClock = !hasShotCounts.value && hasShotClock;
      const shotClockVal = String(shotClock);

      img.onload = () => {
        try {
          const canvas = document.createElement('canvas');
          const scale = 2;
          canvas.width = width * scale;
          canvas.height = height * scale;
          
          const ctx = canvas.getContext('2d');
          
          // Fill with dark background to match web app
          ctx.fillStyle = '#0a0f1e';
          ctx.fillRect(0, 0, canvas.width, canvas.height);
          
          ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
          
          // Draw Shot Clock if available and not in shot overlay mode
          if (shouldDrawShotClock) {
            const fontSize = 48 * scale;
            const paddingX = 16 * scale;
            const paddingY = 4 * scale;
            const margin = 20 * scale;
            
            ctx.font = `${fontSize}px "DSEG7 Classic", monospace`;
            const textMetrics = ctx.measureText(shotClockVal);
            const textWidth = textMetrics.width;
            const boxWidth = textWidth + (paddingX * 2);
            const boxHeight = fontSize + (paddingY * 2);
            
            const x = canvas.width - boxWidth - margin;
            const y = margin;
            
            ctx.fillStyle = '#1a1a1a';
            ctx.fillRect(x, y, boxWidth, boxHeight);
            ctx.strokeStyle = '#333';
            ctx.lineWidth = 2 * scale;
            ctx.strokeRect(x, y, boxWidth, boxHeight);
            
            ctx.fillStyle = '#ff4d4d';
            ctx.shadowColor = '#ff4d4d';
            ctx.shadowBlur = 10 * scale;
            ctx.textBaseline = 'top';
            ctx.fillText(shotClockVal, x + paddingX, y + paddingY);
            ctx.shadowBlur = 0;
          }
          
          // Convert canvas to PNG data URL
          const dataUrl = canvas.toDataURL('image/png');
          
          // Cleanup
          URL.revokeObjectURL(url);
          
          resolve(dataUrl);
        } catch (err) {
          URL.revokeObjectURL(url);
          reject(err);
        }
      };
      
      img.onerror = () => {
        URL.revokeObjectURL(url);
        reject(new Error('Failed to load SVG image'));
      };
      
      img.src = url;
    });
  } catch (err) {
    console.error('[GameBoard] Failed to render PNG:', err);
    return null;
  }
}

// Expose method to parent component
defineExpose({
  renderStateToPng
});

onBeforeUnmount(() => {
  clearClickTimeout();
  window.removeEventListener('mousemove', onGlobalMouseMove);
  window.removeEventListener('mouseup', onGlobalMouseUp);
  clearPassFlash();
  clearShotFlash();
  clearShotJump();
});

</script>

<template>
  <div class="game-board-container" :class="{ 'no-move-transitions': disableTransitions }">
    <div class="board-toolbar">
      <button 
        class="download-button" 
        @click="downloadBoardAsImage"
        title="Download board as PNG"
      >
        📥
      </button>
      <button
        class="policy-toggle-button"
        @click="toggleAllPolicies"
        :aria-pressed="allPoliciesVisible"
        title="Show or hide policy probabilities for all players"
      >
        {{ allPoliciesVisible ? 'Hide Policies' : 'Show Policies' }}
      </button>
    </div>
    <svg :viewBox="viewBox" preserveAspectRatio="xMidYMid meet" ref="svgRef">
      <defs>
        <marker
          id="arrowhead-offense"
          viewBox="0 0 10 10"
          refX="9"
          refY="5"
          markerWidth="5"
          markerHeight="5"
          orient="auto"
        >
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#007bff" />
        </marker>
        <marker
          id="arrowhead-defense"
          viewBox="0 0 10 10"
          refX="9"
          refY="5"
          markerWidth="5"
          markerHeight="5"
          orient="auto"
        >
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#dc3545" />
        </marker>
        <linearGradient id="shot-volume-gradient" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stop-color="rgba(15,23,42,0.85)" />
          <stop offset="100%" stop-color="rgba(251,146,60,0.95)" />
        </linearGradient>
      </defs>
      <g :transform="boardTransform">
        <!-- Draw qualified (blue) and unqualified (dark) court hexes -->
        <polygon
          v-for="hex in courtLayout"
          :key="hex.key"
          :points="[...Array(6)].map((_, i) => {
            const angle_deg = 60 * i + 30;
            const angle_rad = Math.PI / 180 * angle_deg;
            const xPoint = hex.x + HEX_RADIUS * Math.cos(angle_rad);
            const yPoint = hex.y + HEX_RADIUS * Math.sin(angle_rad);
            return `${xPoint},${yPoint}`;
          }).join(' ')"
          :class="['court-hex', threePointQualifiedSet.has(`${hex.q},${hex.r}`) ? 'qualified' : 'unqualified']"
        />

        <!-- Offensive Lane (painted area) -->
        <polygon
          v-for="hex in offensiveLaneHexes"
          :key="hex.key"
          :points="[...Array(6)].map((_, i) => {
            const angle_deg = 60 * i + 30;
            const angle_rad = Math.PI / 180 * angle_deg;
            const xPoint = hex.x + HEX_RADIUS * Math.cos(angle_rad);
            const yPoint = hex.y + HEX_RADIUS * Math.sin(angle_rad);
            return `${xPoint},${yPoint}`;
          }).join(' ')"
          class="offensive-lane"
        />

        <!-- 3PT line outline -->
        <path
          v-if="threePointArcPath"
          :d="threePointArcPath"
          class="three-point-arc"
        />

        <!-- Court reference labels -->
        <text
          v-if="currentGameState && courtLayout.length"
          :x="basketMarkerPosition.x"
          :y="basketMarkerPosition.y"
          dy=".35em"
          text-anchor="middle"
          class="court-marker basket-marker"
        >
          B
        </text>
        <text
          v-if="currentGameState && courtLayout.length"
          :x="halfcourtMarkerPosition.x"
          :y="halfcourtMarkerPosition.y"
          dy=".35em"
          text-anchor="middle"
          class="court-marker halfcourt-marker"
        >
          H
        </text>
        <text
          v-if="currentGameState && courtLayout.length"
          :x="rightSidelineMarkerPosition.x"
          :y="rightSidelineMarkerPosition.y"
          dy=".35em"
          text-anchor="middle"
          class="court-marker sideline-marker"
        >
          R
        </text>
        <text
          v-if="currentGameState && courtLayout.length"
          :x="leftSidelineMarkerPosition.x"
          :y="leftSidelineMarkerPosition.y"
          dy=".35em"
          text-anchor="middle"
          class="court-marker sideline-marker"
        >
          L
        </text>

        <!-- Draw the basket -->
        <circle :cx="basketPosition.x" :cy="basketPosition.y" :r="HEX_RADIUS * 0.8" class="basket-rim" />

        <!-- Draw Ghost Trails -->
        <g 
          v-if="showPlayers"
          v-for="(gameState, step) in gameHistory" 
          :key="`step-${step}`" 
          :style="{ opacity: 0.1 + (0.2 * step / (gameHistory.length - 1)) }"
        >
          <g v-for="player in getRenderablePlayers(gameState)" :key="player.id">
            <circle 
              v-if="step < gameHistory.length - 1"
              :cx="player.x" 
              :cy="player.y" 
              :r="HEX_RADIUS * 0.8" 
              :class="player.isOffense ? 'player-offense' : 'player-defense'"
              class="ghost"
            />
            <text 
              v-if="step < gameHistory.length - 1"
              :x="player.x" 
              :y="player.y" 
              dy=".3em" 
              text-anchor="middle" 
              class="player-text ghost-text"
            >
              {{ player.id }}
            </text>
          </g>
        </g>
        
        <!-- Draw Transition Arrows -->
        <g v-if="showPlayers" v-for="move in playerTransitions" :key="move.key" :style="{ opacity: move.opacity }">
          <line
            :x1="move.startX"
            :y1="move.startY"
            :x2="move.endX"
            :y2="move.endY"
            :stroke="move.isOffense ? '#007bff' : '#dc3545'"
            stroke-width="3"
            :marker-end="move.isOffense ? 'url(#arrowhead-offense)' : 'url(#arrowhead-defense)'"
          />
        </g>

        <!-- Pass target preview (selected receiver) -->
        <g v-if="passTargetPreview && showPlayers" class="pass-preview-group">
          <line
            :x1="passTargetPreview.start.x"
            :y1="passTargetPreview.start.y"
            :x2="passTargetPreview.end.x"
            :y2="passTargetPreview.end.y"
            class="pass-preview-line"
          />
        </g>

        <!-- Draw the current players on top -->
        <g v-if="currentGameState && showPlayers">
        <g
          v-for="player in sortedPlayers"
          :key="player.id"
          :class="[
            'player-group',
            { 'ball-handler-bounce': player.hasBall && draggedPlayerId !== player.id && shotJumpPlayerId !== player.id && shotInFlightPlayerId !== player.id },
            { 'shoot-jump': shotJumpPlayerId === player.id && draggedPlayerId !== player.id },
            { 'shoot-jump-dunk': shotJumpPlayerId === player.id && shotJumpIsDunk && draggedPlayerId !== player.id }
          ]"
          :style="{
            ...(player.hasBall && draggedPlayerId !== player.id ? {
              '--dribble-amp': `${DRIBBLE_AMPLITUDE_PX}px`,
              '--dribble-period': `${DRIBBLE_PERIOD_SECONDS}s`,
              '--dribble-delay': dribbleDelay(player.id)
            } : {}),
            ...(shotJumpPlayerId === player.id && draggedPlayerId !== player.id ? {
              '--jump-amp': `${shotJumpIsDunk ? SHOOT_DUNK_AMPLITUDE_PX : SHOOT_JUMP_AMPLITUDE_PX}px`,
              '--jump-period': `${SHOOT_JUMP_PERIOD_SECONDS}s`,
              '--jump-scale-peak': `${shotJumpIsDunk ? SHOOT_DUNK_SCALE : SHOOT_JUMP_SCALE}`
            } : {})
          }"
        >
            <!-- If dragging this player, show it at dragged pos, otherwise at hex pos -->
            <circle 
              :cx="draggedPlayerId === player.id ? draggedPlayerPos.x : player.x" 
              :cy="draggedPlayerId === player.id ? draggedPlayerPos.y : player.y" 
              :r="HEX_RADIUS * 0.8" 
              :class="[
                player.isOffense ? 'player-offense' : 'player-defense',
                { 'active-player-hex': player.id === activePlayerId },
                { 'dragging': draggedPlayerId === player.id },
                { 'pass-target-preview': passTargetPreview && passTargetPreview.receiverId === player.id }
              ]"
              @mousedown="onMouseDown($event, player)"
              @click="onPlayerClick($event, player)"
              @dblclick.stop="onPlayerDoubleClick($event, player)"
              style="cursor: grab;"
            />
            <text 
              :transform="playerLabelTransform(player)"
              x="0" 
              y="0" 
              dy="0.3em" 
              text-anchor="middle" 
              class="player-text"
              style="pointer-events: none;" 
            >{{ player.id }}</text>
            <!-- EP (Expected Points) label above player ID for offensive players -->
            <text
              v-if="player.isOffense && currentGameState.ep_by_player && currentGameState.ep_by_player[player.id] !== undefined && draggedPlayerId !== player.id"
              :x="player.x"
              :y="player.y"
              dy="-1.0em"
              text-anchor="middle"
              class="noop-prob-text"
            >
              {{ Number(currentGameState.ep_by_player[player.id]).toFixed(2) }}
            </text>
            <!-- NOOP probability label (index 0) for the player -->
            <text
              v-if="isPolicyVisible(player.id) && getPolicyProbsForPlayer(player.id) && getPolicyProbsForPlayer(player.id)[0] !== undefined && draggedPlayerId !== player.id"
              :x="player.x"
              :y="player.y"
              dy="1.2em"
              text-anchor="middle"
              class="noop-prob-text"
              :opacity="probToAlpha(getPolicyProbsForPlayer(player.id)[0])"
            >
              {{ Number(getPolicyProbsForPlayer(player.id)[0]).toFixed(2) }}
            </text>
            <!-- Display policy attempt probability for ball handler -->
            <text 
              v-if="player.hasBall && isPolicyVisible(player.id) && ballHandlerShotProb !== null && draggedPlayerId !== player.id && showValueAnnotations"
              :x="player.x" 
              :y="player.y" 
              dy="0.4em" 
              dx="3.4em"
              text-anchor="middle" 
              class="shot-prob-text"
            >
              {{ ballHandlerShotProb.toFixed(2) }}
            </text>
            <!-- Display conditional make percentage for ball handler -->
            <text 
              v-if="player.hasBall && isPolicyVisible(player.id) && ballHandlerMakeProb !== null && draggedPlayerId !== player.id && showValueAnnotations"
              :x="player.x" 
              :y="player.y" 
              dy="-0.4em" 
              dx="3.4em"
              text-anchor="middle" 
              class="shot-prob-text"
            >
              {{ Math.round(ballHandlerMakeProb * 100) }}%
            </text>
            <!-- Ball handler indicator -->
            <circle 
                v-if="player.hasBall && shotJumpPlayerId !== player.id && shotInFlightPlayerId !== player.id" 
                :cx="draggedPlayerId === player.id ? draggedPlayerPos.x : player.x" 
                :cy="draggedPlayerId === player.id ? draggedPlayerPos.y : player.y" 
                :r="HEX_RADIUS * 0.9" 
                class="ball-indicator" 
                style="pointer-events: none;"
            />
            <!-- Action indicator (move arrow, pass hand, or shoot target) using native SVG -->
            <g 
              v-if="draggedPlayerId !== player.id && getActionIndicator(player.id, player.x, player.y, player.hasBall)"
              class="action-indicator"
              :transform="`translate(${getActionIndicator(player.id, player.x, player.y, player.hasBall).x}, ${getActionIndicator(player.id, player.x, player.y, player.hasBall).y})`"
            >
              <!-- Move arrow icon (location-arrow style) -->
              <g 
                v-if="getActionIndicator(player.id, player.x, player.y, player.hasBall).type === 'move'"
                :transform="`rotate(${getActionIndicator(player.id, player.x, player.y, player.hasBall).rotation})`"
              >
                <path 
                  d="M0,-7 L5,5 L0,2 L-5,5 Z" 
                  :class="['action-arrow', player.isOffense ? 'offense' : 'defense']"
                />
              </g>
              <!-- Pass indicator (hand-pointer style arrow) -->
              <g 
                v-if="getActionIndicator(player.id, player.x, player.y, player.hasBall).type === 'pass'"
                :transform="`rotate(${getActionIndicator(player.id, player.x, player.y, player.hasBall).rotation})`"
              >
                <path 
                  d="M0,-7 L4,3 L1,1 L1,7 L-1,7 L-1,1 L-4,3 Z" 
                  class="action-pass"
                />
              </g>
              <!-- Shoot indicator (bullseye style target) -->
              <g v-if="getActionIndicator(player.id, player.x, player.y, player.hasBall).type === 'shoot'">
                <circle r="6" class="action-shoot-outer" />
                <circle r="3" class="action-shoot-middle" />
                <circle r="1.5" class="action-shoot-inner" />
              </g>
            </g>
          </g>
        </g>
        
        <!-- Shot count annotations (from evaluation) -->
        <g class="shot-count-layer" v-if="hasShotCounts">
        <text
          v-if="shotChartLabel"
          :x="shotChartTitlePos.x"
          :y="shotChartTitlePos.y"
          text-anchor="end"
          class="shot-chart-title"
        >
          {{ shotChartLabel }}
        </text>
        <polygon
          v-for="pt in shotOverlayPoints"
          :key="`shot-poly-${pt.key}`"
          :points="hexPointsFor(pt.x, pt.y, HEX_RADIUS)"
          :fill="volumeFill(pt.attempts)"
            :stroke="volumeStroke(pt.attempts)"
            stroke-width="1.4"
          />
          <text
            v-for="pt in shotOverlayPoints"
            :key="`shot-${pt.key}`"
            :x="pt.x"
            :y="pt.y - HEX_RADIUS * 0.05"
            text-anchor="middle"
            class="shot-count-text"
          >
            {{ pt.attempts > 0 ? Math.round((pt.makes / pt.attempts) * 100) : 0 }}%
          </text>
         <text
            v-for="pt in shotOverlayPoints"
            :key="`shot-atts-${pt.key}`"
            :x="pt.x"
            :y="pt.y + HEX_RADIUS * 0.55"
            text-anchor="middle"
            class="shot-count-attempts"
          >
            {{ pt.attempts }}
          </text>
        </g>
        
        <!-- Draw Pass Rays (ball handler to teammates with steal probabilities) - drawn after players for visibility -->
        <g v-if="showPlayers" v-for="ray in passRays" :key="`pass-ray-${ray.teammateId}`" class="pass-ray-group">
          <line
            :x1="ray.x1"
            :y1="ray.y1"
            :x2="ray.x2"
            :y2="ray.y2"
            class="pass-ray"
            :opacity="ray.stealOpacity"
          />
          <text
            :x="ray.midX"
            :y="ray.midY"
            text-anchor="middle"
            dominant-baseline="middle"
            class="steal-prob-label"
            :opacity="ray.stealOpacity"
          >
            {{ ray.stealLabel }}
          </text>
        </g>

        <!-- Flash effect for completed passes -->
        <g v-if="passFlash && showPlayers" class="pass-flash-group">
          <line
            :x1="passFlash.x1"
            :y1="passFlash.y1"
            :x2="passFlash.x2"
            :y2="passFlash.y2"
            :stroke="ballColor"
            class="pass-flash-line"
          />
          <text
            :x="passFlash.labelX"
            :y="passFlash.labelY"
            text-anchor="middle"
            dominant-baseline="middle"
            :fill="ballColor"
            class="pass-flash-text"
          >
            PASS {{ passFlash.passerId }} -> {{ passFlash.receiverId }}
          </text>
        </g>

        <!-- Flash effect for shot attempts -->
        <g v-if="shotFlash && showPlayers" class="shot-flash-group">
          <path
            :d="shotFlash.path"
            :stroke="shotFlash.color"
            class="shot-flash-line"
            fill="none"
            :style="{ filter: `drop-shadow(0 0 10px ${shotFlash.color})` }"
          />
        </g>
        
        <!-- Draw Policy Suggestions -->
        <g v-if="policySuggestions.length > 0">
          <text 
            v-for="sugg in policySuggestions"
            :key="sugg.key"
            :x="sugg.x"
            :y="sugg.y"
            text-anchor="middle"
            class="policy-suggestion-text"
          >
            <tspan
              v-if="sugg.moveProb !== null && sugg.moveProb !== undefined"
              :x="sugg.x"
              dy="-0.4em"
              :opacity="sugg.moveOpacity"
            >
              {{ Number(sugg.moveProb).toFixed(3) }}
            </tspan>
            <tspan
              v-if="sugg.passProb !== null && sugg.passProb !== undefined"
              :x="sugg.x"
              :dy="sugg.moveProb !== null && sugg.moveProb !== undefined ? '1.4em' : '0'"
              class="policy-pass-prob"
              :opacity="sugg.passOpacity"
            >
              {{ Number(sugg.passProb).toFixed(3) }}
            </tspan>
          </text>
        </g>

        <!-- Draw Episode Outcome Indicators -->
        <g v-if="episodeOutcome" class="outcome-overlay">
            <!-- Basket Fill for Shots -->
            <circle v-if="episodeOutcome.type === 'MADE_SHOT'" :cx="basketPosition.x" :cy="basketPosition.y" :r="HEX_RADIUS" class="basket-fill-made" />
            <circle v-if="episodeOutcome.type === 'MISSED_SHOT'" :cx="basketPosition.x" :cy="basketPosition.y" :r="HEX_RADIUS" class="basket-fill-missed" />

            <!-- Turnover 'X' -->
            <text v-if="episodeOutcome.type === 'TURNOVER'" :x="episodeOutcome.x" :y="episodeOutcome.y" class="turnover-x">X</text>
            
            <!-- Defensive Violation indicator -->
            <text v-if="episodeOutcome.type === 'DEFENSIVE_VIOLATION' && episodeOutcome.x" :x="episodeOutcome.x" :y="episodeOutcome.y" class="violation-marker">!</text>
        </g>

        <!-- State-value overlay -->
        <g v-if="(offenseStateValue !== null || defenseStateValue !== null) && showValueAnnotations" class="state-value-overlay">
          <rect
            :x="stateValueAnchor.x"
            :y="stateValueAnchor.y"
            :width="stateValueBoxWidth"
            :height="stateValueBoxHeight"
            rx="12"
            ry="12"
          />
          <text
            :x="stateValueAnchor.x + stateValueBoxWidth / 2"
            :y="stateValueAnchor.y + stateValueBoxHeight / 2 - (defenseStateValue !== null ? HEX_RADIUS * 0.35 : 0)"
            text-anchor="middle"
            dominant-baseline="middle"
            class="state-value-text"
          >
            V<tspan baseline-shift="-35%" font-size="65%">o</tspan>
            {{ offenseStateValue !== null ? offenseStateValue.toFixed(2) : '—' }}
          </text>
          <text
            v-if="defenseStateValue !== null"
            :x="stateValueAnchor.x + stateValueBoxWidth / 2"
            :y="stateValueAnchor.y + stateValueBoxHeight / 2 + HEX_RADIUS * 0.45"
            text-anchor="middle"
            dominant-baseline="middle"
            class="state-value-text"
          >
            V<tspan baseline-shift="-35%" font-size="65%">d</tspan>
            {{ defenseStateValue.toFixed(2) }}
          </text>
        </g>
      </g>

      <!-- Outcome Text (drawn outside the transformed group to keep it upright) -->
      <g v-if="episodeOutcome" class="outcome-text-group">
          <text v-if="episodeOutcome.type === 'MADE_SHOT'" x="50%" y="15%" class="outcome-text made">
              <tspan class="player-outcome-text" x="50%" dy="-1.2em">Player {{ episodeOutcome.playerId }}</tspan>
              <tspan x="50%" dy="1.2em">{{ episodeOutcome.isDunk ? 'Made Dunk!' : (episodeOutcome.isThree ? 'Made 3!' : 'Made 2!') }}</tspan>
          </text>
          <text v-if="episodeOutcome.type === 'MISSED_SHOT'" x="50%" y="15%" class="outcome-text missed">
              <tspan class="player-outcome-text" x="50%" dy="-1.2em">Player {{ episodeOutcome.playerId }}</tspan>
              <tspan x="50%" dy="1.2em">{{ episodeOutcome.isDunk ? 'Missed Dunk!' : (episodeOutcome.isThree ? 'Missed 3!' : 'Missed 2!') }}</tspan>
          </text>
          <text v-if="episodeOutcome.type === 'TURNOVER'" x="50%" y="15%" class="outcome-text turnover">TURNOVER!</text>
          <text v-if="episodeOutcome.type === 'SHOT_CLOCK_VIOLATION'" x="50%" y="15%" class="outcome-text turnover long-outcome-text">SHOT CLOCK!</text>
          <text v-if="episodeOutcome.type === 'DEFENSIVE_VIOLATION'" x="50%" y="15%" class="outcome-text violation long-outcome-text">
              <tspan class="player-outcome-text" x="50%" dy="-1.2em">Player {{ episodeOutcome.playerId }}</tspan>
              <tspan x="50%" dy="1.2em">Violation - Defense!</tspan>
          </text>
      </g>

      <!-- In-canvas shot legend -->
      <g v-if="shotLegendConfig" class="legend-overlay">
        <rect
          :x="shotLegendConfig.x"
          :y="shotLegendConfig.y"
          :width="shotLegendConfig.width"
          :height="shotLegendConfig.height"
          class="legend-bg"
          rx="10"
          ry="10"
        />
        <text
          :x="shotLegendConfig.x + shotLegendConfig.width / 2"
          :y="shotLegendConfig.y + shotLegendConfig.height * 0.32"
          text-anchor="middle"
          class="legend-label-text"
        >
          Volume
        </text>
        <rect
          :x="shotLegendConfig.x + shotLegendConfig.width * 0.08"
          :y="shotLegendConfig.y + shotLegendConfig.height * 0.48"
          :width="shotLegendConfig.width * 0.84"
          :height="shotLegendConfig.height * 0.32"
          fill="url(#shot-volume-gradient)"
          class="legend-gradient-rect"
          rx="6"
          ry="6"
        />
        <text
          :x="shotLegendConfig.x + shotLegendConfig.width * 0.08"
          :y="shotLegendConfig.y + shotLegendConfig.height * 0.75"
          text-anchor="start"
          class="legend-scale-text"
        >
          0
        </text>
        <text
          :x="shotLegendConfig.x + shotLegendConfig.width * 0.92"
          :y="shotLegendConfig.y + shotLegendConfig.height * 0.75"
          text-anchor="end"
          class="legend-scale-text"
        >
          {{ maxShotAttempts }}
        </text>
      </g>

      <!-- MLflow Run ID label (top-left, outside transformed group) -->
      <text
        v-if="currentGameState && currentGameState.run_id"
        :x="parseFloat(viewBox.split(' ')[0]) + 100"
        :y="parseFloat(viewBox.split(' ')[1]) + 10"
        text-anchor="start"
        dominant-baseline="hanging"
        class="run-id-label"
      >
        {{ currentGameState.run_id }}
      </text>
    </svg>
    <div class="shot-clock-wrapper" v-if="!hasShotCounts">
      <div class="shot-clock-overlay">
        {{ displayedShotClockValue }}
      </div>
      <div class="shot-clock-controls">
        <button
          class="shot-clock-button"
          :disabled="!canIncrementShotClock"
          @click="adjustShotClock(1)"
          aria-label="Increase shot clock"
        >
          ▲
        </button>
        <button
          class="shot-clock-button"
          :disabled="!canDecrementShotClock"
          @click="adjustShotClock(-1)"
          aria-label="Decrease shot clock"
        >
          ▼
        </button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.game-board-container {
  position: relative; /* Needed for overlay positioning */
  flex: 1; /* Allow this component to grow and fill available space */
  min-width: 400px; /* Ensure it doesn't get too small */
  width: 100%;
  margin: 0; /* Remove auto margin which conflicts with flexbox */
  border-radius: 8px;
  overflow: visible; /* Allow the shot clock to be positioned outside */
  background: radial-gradient(circle at 30% 50%, #0f172a, #01010a 70%);
  border: 1px solid rgba(148, 163, 184, 0.35);
  box-shadow: 0 20px 45px rgba(2, 6, 23, 0.65);
}

.shot-clock-overlay {
  position: relative;
  font-family: 'DSEG7 Classic', sans-serif;
  font-size: 5rem;
  color: #ff4d4d; /* Bright red for the LED color */
  background-color: #1a1a1a; /* Dark background for contrast */
  padding: 2px 8px;
  border-radius: 5px;
  border: 1px solid #333;
  text-shadow: 0 0 5px #ff4d4d, 0 0 10px #ff4d4d; /* Glowing effect */
  pointer-events: none; /* Make it non-interactive */
  z-index: 10; /* Ensure it's above the SVG */
}

.shot-clock-wrapper {
  position: absolute;
  top: 5px;
  right: 5px;
  display: flex;
  align-items: center;
  gap: 6px;
  z-index: 11;
}

.shot-clock-controls {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.shot-clock-button {
  width: 30px;
  height: 24px;
  border-radius: 4px;
  border: 1px solid rgba(0, 0, 0, 0.6);
  background: rgba(255, 255, 255, 0.9);
  color: #222;
  font-size: 12px;
  line-height: 1;
  cursor: pointer;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.3);
}

.shot-clock-button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.board-toolbar {
  position: absolute;
  top: 10px;
  left: 10px;
  display: flex;
  gap: 8px;
  align-items: center;
  z-index: 12;
}

.download-button {
  background: rgba(255, 255, 255, 0.9);
  border: 1px solid #ddd;
  border-radius: 6px;
  padding: 8px 12px;
  cursor: pointer;
  font-size: 2rem;
  color: #333;
  transition: all 0.2s ease;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.download-button:hover {
  background: rgb(13, 9, 223);
  box-shadow: 0 4px 8px rgba(0,0,0,0.15);
  transform: translateY(-1px);
}

.download-button:active {
  transform: translateY(0);
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.policy-toggle-button {
  background: rgba(255, 255, 255, 0.92);
  border: 1px solid #ddd;
  border-radius: 6px;
  padding: 10px 12px;
  cursor: pointer;
  font-size: 0.95rem;
  color: #0b172d;
  transition: all 0.2s ease;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
  margin-left: auto;
}

.shot-status {
  display: inline-flex;
  align-items: center;
  gap: 0.25rem;
  margin-left: 0.75rem;
  padding: 6px 10px;
  border-radius: 8px;
  background: rgba(15, 23, 42, 0.8);
  border: 1px solid rgba(148, 163, 184, 0.35);
  color: var(--app-text);
  font-size: 0.8rem;
}

.shot-status .status-label {
  color: var(--app-text-muted);
  letter-spacing: 0.04em;
}

.policy-toggle-button:hover {
  background: #0d59df;
  color: #f8fafc;
  box-shadow: 0 4px 8px rgba(0,0,0,0.15);
  transform: translateY(-1px);
}

.policy-toggle-button:active {
  transform: translateY(0);
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.state-value-overlay rect {
  fill: rgba(15, 15, 20, 0.9);
  stroke: rgba(255, 255, 255, 0.7);
  stroke-width: 1;
  filter: drop-shadow(0px 0px 2px rgba(0, 0, 0, 0.6));
}

.state-value-text {
  fill: #fffbf2;
  font-size: 1.1rem;
  font-weight: 600;
  letter-spacing: 0.3px;
}

/* Removed rotation; court now renders in original orientation */
svg {
  display: block;
  width: 100%;
  height: auto;
}
.court-hex {
  stroke: rgba(15, 23, 42, 0.6);
  stroke-width: 0.1rem;
}
.court-hex.qualified {
  fill: rgba(59, 130, 246, 0.35);
  stroke: #fef3c781;
  stroke-width: 0.05rem;
}
.court-hex.unqualified {
  fill: rgba(38, 47, 77, 0.89);
  stroke: rgba(15, 23, 42, 0.95);
}
.three-point-arc {
  fill: none;
  stroke: #fb923c;
  stroke-width: 0.3rem;
  stroke-linecap: square;
  /* stroke-dasharray: 4 8; */
  filter: drop-shadow(0px 0px 4px rgba(0, 0, 0, 0.6));
}
.offensive-lane {
  fill: rgba(243, 4, 104, 0.397);
  stroke: rgba(255, 140, 140, 0.5);
  stroke-width: 1;
}
.player-offense {
  fill: #007bff;
  stroke: white;
  stroke-width: 0.05rem;
  transition: cx 0.26s ease, cy 0.26s ease, r 0.12s ease;
}
.player-defense {
  fill: #dc3545;
  stroke: white;
  stroke-width: 0.05rem;
  transition: cx 0.26s ease, cy 0.26s ease, r 0.12s ease;
}
.player-group {
  transform-origin: center;
  transform-box: fill-box; /* keep scale/translate centered on the marker */
}
.ball-handler-bounce {
  animation: dribble-bounce var(--dribble-period, 0.95s) ease-in-out infinite;
  animation-delay: var(--dribble-delay, 0s);
  will-change: transform;
}
.shoot-jump {
  animation: shoot-jump var(--jump-period, 1.35s) ease-out forwards;
  will-change: transform;
}
.shoot-jump-dunk {
  animation-name: shoot-jump;
}
.active-player-hex {
  stroke: #62ff3b; /* Bright yellow */
  stroke-width: 0.15rem; /* Increased width for better visibility */
}
.pass-target-preview {
  stroke: #f8e71c;
  stroke-width: 0.2rem;
  filter: drop-shadow(0 0 6px rgba(248, 231, 28, 0.6));
}
.dragging {
  opacity: 0.8;
  stroke: white;
  stroke-dasharray: 4 2;
  transition: none !important;
}
.player-text {
  fill: white;
  font-weight: bold;
  font-size: 0.85rem;
  paint-order: stroke;
  stroke: black;
  stroke-width: 0.1rem;
  transition: transform 0.26s ease;
}
.ghost-text {
  font-size: 10px;
  opacity: 0.7;
  stroke-width: 0.2;
}
.court-marker {
  fill: rgba(255, 255, 255, 0.5);
  font-weight: 700;
  font-size: 1rem;
  paint-order: stroke;
  stroke: rgba(0, 0, 0, 0.45);
  stroke-width: 0.08rem;
  pointer-events: none;
}
.basket-marker {
  font-size: 1.1rem;
  opacity: 0.7;
}
.halfcourt-marker {
  font-size: 1.1rem;
  opacity: 0.5;
}
.sideline-marker {
  font-size: 1rem;
  opacity: 0.55;
}
.basket-rim {
  fill: none;
  stroke: #ff8c00;
  stroke-width: 0.25rem;
}
.ball-indicator {
  fill: none;
  stroke: orange;
  stroke-linecap: round;
  stroke-width: 0.25rem;
  stroke-dasharray: 4 8;
  transition: cx 0.26s ease, cy 0.26s ease, r 0.12s ease;
}

/* Action indicator styles */
.action-indicator {
  pointer-events: none;
  transition: transform 0.26s ease;
}
.action-arrow {
  stroke: #000;
  stroke-width: 1;
  filter: drop-shadow(0 0 2px rgba(0,0,0,0.8));
}
.action-arrow.offense {
  fill: #ffd700;
}
.action-arrow.defense {
  fill: #ff6b6b;
}
.action-pass {
  fill: #90EE90;
  stroke: #000;
  stroke-width: 1;
  filter: drop-shadow(0 0 2px rgba(0,0,0,0.8));
}
.action-shoot-outer {
  fill: none;
  stroke: #ff4500;
  stroke-width: 2;
  filter: drop-shadow(0 0 2px rgba(0,0,0,0.8));
}
.action-shoot-middle {
  fill: none;
  stroke: #ff4500;
  stroke-width: 1.5;
}
.action-shoot-inner {
  fill: #ff4500;
}

.ghost {
  stroke: none;
}
.policy-suggestion-text {
  font-size: 0.65rem;
  font-weight: bold;
  fill: white;
  paint-order: stroke;
  stroke: black;
  stroke-width: 0.1rem;
  pointer-events: none;
}
.policy-pass-prob {
  fill: #f97316;
}
.shot-prob-text {
  font-size: 1.5rem;
  font-weight: bold;
  fill: greenyellow;
  paint-order: stroke;
  stroke: black;
  stroke-width: 0.25rem;
  transition: x 0.26s ease, y 0.26s ease;
}

.no-move-transitions .player-offense,
.no-move-transitions .player-defense,
.no-move-transitions .player-text,
.no-move-transitions .shot-prob-text,
.no-move-transitions .noop-prob-text,
.no-move-transitions .ball-indicator,
.no-move-transitions .action-indicator {
  transition: none !important;
}

.noop-prob-text {
  font-size: 10px;
  font-weight: 700;
  fill: #111;
  paint-order: stroke;
  stroke: #fff;
  stroke-width: 0.6px;
  transition: x 0.26s ease, y 0.26s ease;
}

.shot-count-layer {
  pointer-events: none;
}

.shot-count-text {
  fill: #f8fafc;
  font-size: 0.9rem;
  font-weight: 800;
  paint-order: stroke;
  stroke: rgba(2, 6, 23, 0.85);
  stroke-width: 1.1px;
}

.shot-count-attempts {
  fill: #e2e8f0;
  font-size: 0.60rem;
  font-weight: 500;
  text-shadow: 0 0 3px rgba(0, 0, 0, 0.6);
}
.shot-chart-title {
  fill: #fbbf24;
  font-size: 0.9rem;
  font-weight: 800;
  text-shadow: 0 0 8px rgba(0,0,0,0.65);
}

.legend-overlay {
  pointer-events: none;
}

.legend-bg {
  fill: rgba(15, 23, 42, 0.82);
  stroke: rgba(148, 163, 184, 0.4);
  stroke-width: 0.02rem;
  filter: drop-shadow(0 4px 12px rgba(0, 0, 0, 0.45));
}

.legend-label-text {
  fill: var(--app-text, #e2e8f0);
  font-size: 0.6rem;
  font-weight: 500;
  letter-spacing: 0.08em;
}

.legend-gradient-rect {
  stroke: rgba(148, 163, 184, 0.45);
  stroke-width: 0.02rem;
}

.legend-scale-text {
  fill: var(--app-text-muted, #cbe1e0);
  font-size: 0.8rem;
  font-weight: 500;
}

/* --- Outcome Indicator Styles --- */
.basket-fill-made {
    fill: green;
    opacity: 0.6;
}
.basket-fill-missed {
    fill: red;
    opacity: 0.6;
}
.turnover-x {
    font-size: 48px;
    fill: darkred;
    font-weight: bold;
    text-anchor: middle;
    dominant-baseline: central;
    transform: scale(1, -1); /* Counteract the group flip */
}
.violation-marker {
    font-size: 48px;
    fill: orange;
    font-weight: bold;
    text-anchor: middle;
    dominant-baseline: central;
    transform: scale(1, -1); /* Counteract the group flip */
}
.outcome-text-group {
    pointer-events: none; /* Make it non-interactive */
}
.outcome-text {
    font-size: 64px;
    font-weight: bold;
    text-anchor: middle;
    paint-order: stroke;
    stroke-width: 2px;
    stroke: black;
}
.player-outcome-text {
    font-size: 32px; /* Smaller font for the player ID */
}
.long-outcome-text {
    font-size: 54px; /* A smaller font size for longer text */
}
.made { fill: lightgreen; }
.missed { fill: #ff4d4d; }
.turnover { fill: #ff4d4d; }
.violation { fill: orange; }

/* Pass rays and steal probability labels */
.pass-ray {
  stroke: rgba(255, 255, 255, 0.3);
  stroke-width: 2;
  stroke-dasharray: 8, 4;
  pointer-events: none;
}

.steal-prob-label {
  font-size: 14px;
  font-weight: bold;
  fill: white;
  paint-order: stroke;
  stroke: black;
  stroke-width: 2px;
  pointer-events: none;
}

.pass-preview-line {
  stroke: rgba(248, 231, 28, 0.9);
  stroke-width: 4px;
  stroke-dasharray: 10 6;
  filter: drop-shadow(0 0 6px rgba(248, 231, 28, 0.5));
  pointer-events: none;
}

.pass-flash-line {
  stroke-width: 8;
  stroke-linecap: round;
  filter: drop-shadow(0 0 8px rgba(255, 165, 0, 0.6));
  animation: pass-flash-line 1s ease-out forwards;
}

.pass-flash-text {
  font-size: 18px;
  font-weight: 800;
  paint-order: stroke;
  stroke: #0a0f1e;
  stroke-width: 3px;
  letter-spacing: 0.5px;
  animation: pass-flash-text 1s ease-out forwards;
}

@keyframes pass-flash-line {
  0% { opacity: 1; stroke-width: 10; }
  60% { opacity: 0.75; stroke-width: 6; }
  100% { opacity: 0; stroke-width: 2; }
}

@keyframes pass-flash-text {
  0% { opacity: 1; transform: scale(1); }
  70% { opacity: 0.85; transform: scale(1.06); }
  100% { opacity: 0; transform: scale(1.08); }
}

@keyframes dribble-bounce {
  0%, 100% { transform: translateY(0); }
  38% { transform: translateY(calc(-1 * var(--dribble-amp, 6px))); }
  50% { transform: translateY(calc(-1 * var(--dribble-amp, 6px))); }
  72% { transform: translateY(calc(-0.25 * var(--dribble-amp, 6px))); }
}

@keyframes shoot-jump {
  0% { transform: translateY(0) scale(1); }
  28% { transform: translateY(calc(-1 * var(--jump-amp, 8px))) scale(var(--jump-scale-peak, 1)); }
  55% { transform: translateY(calc(-0.82 * var(--jump-amp, 8px))) scale(var(--jump-scale-peak, 1)); }
  85% { transform: translateY(calc(-0.12 * var(--jump-amp, 8px))) scale(1.04); }
  100% { transform: translateY(0) scale(1); }
}

.shot-flash-line {
  stroke-width: 10;
  stroke-linecap: round;
  animation: pass-flash-line 1s ease-out forwards;
}

</style>

<style scoped>
.run-id-label {
  font-size: 0.8rem;
  font-weight: normal;
  fill: white;
  paint-order: stroke;
  stroke: black;
  stroke-width: 0.1rem;
  pointer-events: none;
}
</style>
