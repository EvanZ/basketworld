<script setup>
import { computed, ref, watch, onMounted, onBeforeUnmount } from 'vue';
import { getShotProbability, getPassStealProbabilities, renderGifFromPngs } from '@/services/api';

const props = defineProps({
  gameHistory: {
    type: Array,
    required: true,
  },
  playbookOverlay: {
    type: Object,
    default: null,
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
  passAnimationStyle: {
    type: String,
    default: 'projectile',
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
  allowShotClockAdjustment: {
    type: Boolean,
    default: true,
  },
  disableBackendValueFetches: {
    type: Boolean,
    default: false,
  },
  allowPositionDrag: {
    type: Boolean,
    default: true,
  },
  minimalChrome: {
    type: Boolean,
    default: false,
  },
  playerDisplayNames: {
    type: Object,
    default: () => ({}),
  },
  playerJerseyNumbers: {
    type: Object,
    default: () => ({}),
  },
  forcedEpisodeOutcome: {
    type: Object,
    default: null,
  },
});

const emit = defineEmits(['update:activePlayerId', 'update-player-position', 'adjust-shot-clock', 'update-placement']);

// ------------------------------------------------------------
//  HEXAGON GEOMETRY — POINTY-TOP, ODD-R OFFSET  (matches Python)
// ------------------------------------------------------------

const HEX_RADIUS = 24;  // pixel radius of one hexagon corner-to-center
const SQRT3 = Math.sqrt(3);
const HEX_HALF_WIDTH = HEX_RADIUS * SQRT3 * 0.5;
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
const PLAYBOOK_TRAJECTORY_COLORS = ['#38bdf8', '#34d399', '#f472b6', '#f59e0b', '#a78bfa', '#fb7185'];
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

function toIdSet(values) {
  const out = new Set();
  if (!Array.isArray(values)) return out;
  for (const raw of values) {
    const pid = Number(raw);
    if (Number.isFinite(pid)) out.add(pid);
  }
  return out;
}

function getPlayableOwnershipSets(gameState) {
  const userIds = toIdSet(gameState?.playable_user_ids);
  const aiIds = toIdSet(gameState?.playable_ai_ids);
  return { userIds, aiIds };
}

function getPlayerOwner(gameState, playerId) {
  if (!gameState) return null;
  const { userIds, aiIds } = getPlayableOwnershipSets(gameState);
  if (userIds.has(playerId)) return 'user';
  if (aiIds.has(playerId)) return 'ai';
  return null;
}

function getPlayerDisplayName(playerId) {
  const id = Number(playerId);
  if (!Number.isFinite(id)) return '';
  const map = props.playerDisplayNames && typeof props.playerDisplayNames === 'object'
    ? props.playerDisplayNames
    : {};
  const raw = map[id] ?? map[String(id)];
  if (typeof raw !== 'string') return '';
  return raw.trim().toUpperCase();
}

function getPlayerJerseyNumber(playerId) {
  const id = Number(playerId);
  if (!Number.isFinite(id)) return '';
  const map = props.playerJerseyNumbers && typeof props.playerJerseyNumbers === 'object'
    ? props.playerJerseyNumbers
    : {};
  const raw = map[id] ?? map[String(id)];
  const jersey = typeof raw === 'string' ? raw.trim() : String(raw ?? '').trim();
  if (!jersey) return String(id);
  return jersey;
}

function getOutcomePlayerLabel(playerId) {
  const id = Number(playerId);
  if (!Number.isFinite(id)) return 'Unknown';
  const displayName = getPlayerDisplayName(id);
  if (displayName) return displayName;
  return `Player ${id}`;
}

function hasOutcomePlayer(playerId) {
  return Number.isFinite(Number(playerId));
}

function getTurnoverOutcomeText(outcome) {
  const stolenBy = Number(outcome?.stolenBy);
  if (Number.isFinite(stolenBy)) {
    return `Stolen by ${getOutcomePlayerLabel(stolenBy)}!`;
  }
  return 'Turnover!';
}

function expectedValueLabelDy() {
  return props.minimalChrome ? '1.9em' : '-1.0em';
}

function isPointerPassMode(gs) {
  return String(gs?.pass_mode || 'directional').toLowerCase() === 'pointer_targeted';
}

function getPointerPassTeammates(gs, passerId) {
  if (!gs || passerId === null || passerId === undefined) return [];
  const pid = Number(passerId);
  const offense = Array.isArray(gs.offense_ids) ? gs.offense_ids : [];
  const defense = Array.isArray(gs.defense_ids) ? gs.defense_ids : [];
  const team = offense.includes(pid) ? offense : defense.includes(pid) ? defense : [];
  return team
    .filter((tid) => Number(tid) !== pid)
    .map((tid) => Number(tid))
    .filter((tid) => Number.isFinite(tid))
    .sort((a, b) => a - b)
    .slice(0, 6);
}

function resolvePassTargetFromAction(gs, passerId, action) {
  if (!gs || !action || passerId === null || passerId === undefined) return null;

  const directMatch = String(action).match(/^PASS->(\d+)$/);
  if (directMatch) {
    const targetId = Number(directMatch[1]);
    return Number.isFinite(targetId) ? targetId : null;
  }

  if (!String(action).startsWith('PASS_')) return null;

  if (isPointerPassMode(gs)) {
    const slotIdx = PASS_ACTION_TO_DIR[action];
    if (slotIdx === undefined) return null;
    const teammates = getPointerPassTeammates(gs, passerId);
    if (slotIdx < 0 || slotIdx >= teammates.length) return null;
    return teammates[slotIdx];
  }

  return null;
}

function getRenderablePlayers(gameState) {
  if (!gameState || !gameState.positions) return [];
  return gameState.positions.map((pos, index) => {
    const [q, r] = pos;
    const { x, y } = axialToCartesian(q, r);
    const isOffense = gameState.offense_ids.includes(index);
    const hasBall = gameState.ball_holder === index;
    const owner = getPlayerOwner(gameState, index);
    return { id: index, x, y, isOffense, hasBall, owner };
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
const PROJECTILE_ARROW_LENGTH_SCALE = 0.5;
const passFlash = ref(null);
let passFlashSerial = 0;
const passFlashNowMs = ref(0);
const passFlashRaf = ref(null);
const passFlashTimeout = ref(null);
const shotFlash = ref(null);
let shotFlashSerial = 0;
const shotFlashNowMs = ref(0);
const shotFlashRaf = ref(null);
const shotFlashTimeout = ref(null);
const lastShotAnimationKey = ref(null);
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
const showShotPressureRing = ref(true);
const showDefenderPressureShake = ref(true);
const policyVisibility = ref(new Set()); // Player IDs with policy overlays shown
const showDownloadMenu = ref(false);
const downloadMenuRef = ref(null);
const isDownloadRunning = ref(false);
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

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function closeDownloadMenu() {
  showDownloadMenu.value = false;
}

function toggleDownloadMenu() {
  if (isDownloadRunning.value) return;
  showDownloadMenu.value = !showDownloadMenu.value;
}

function onGlobalMouseDown(event) {
  if (!showDownloadMenu.value) return;
  const root = downloadMenuRef.value;
  if (!root) return;
  if (root.contains(event.target)) return;
  closeDownloadMenu();
}

async function downloadBoardAsGif() {
  if (isDownloadRunning.value) return;
  isDownloadRunning.value = true;

  try {
    const targetDurationMs = 840; // Keep similar overall speed as prior single-step GIF capture.
    const frameCount = 24;
    const targetStepDurationMs = targetDurationMs / Math.max(1, frameCount - 1);
    const capturedFrames = [];
    const durations = [];
    const captureStartMs = performance.now();

    for (let i = 0; i < frameCount; i += 1) {
      const targetCaptureMs = captureStartMs + (i * targetStepDurationMs);
      const waitMs = targetCaptureMs - performance.now();
      if (waitMs > 0) {
        // eslint-disable-next-line no-await-in-loop
        await sleep(waitMs);
      }

      const capturedAtMs = performance.now();
      // eslint-disable-next-line no-await-in-loop
      const pngDataUrl = await renderStateToPng();
      if (pngDataUrl) {
        capturedFrames.push({ pngDataUrl, capturedAtMs });
      }
    }

    if (capturedFrames.length < 2) {
      throw new Error('Not enough frames to build GIF');
    }

    const frames = capturedFrames.map((f) => f.pngDataUrl);
    for (let i = 0; i < capturedFrames.length - 1; i += 1) {
      const deltaMs = Math.max(
        10,
        Number(capturedFrames[i + 1].capturedAtMs) - Number(capturedFrames[i].capturedAtMs),
      );
      durations.push(deltaMs / 1000);
    }
    durations.push(durations.length > 0 ? durations[durations.length - 1] : (targetStepDurationMs / 1000));

    const gifBlob = await renderGifFromPngs(frames, durations, targetStepDurationMs);
    const url = URL.createObjectURL(gifBlob);
    const link = document.createElement('a');
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
    link.download = `basketworld-board-${timestamp}.gif`;
    link.href = url;
    link.click();
    URL.revokeObjectURL(url);
  } catch (err) {
    console.error('[GameBoard] Failed to download GIF:', err);
    alert('Failed to download animated GIF');
  } finally {
    isDownloadRunning.value = false;
  }
}

async function handleDownloadChoice(format) {
  closeDownloadMenu();
  if (format === 'gif') {
    await downloadBoardAsGif();
    return;
  }
  await downloadBoardAsImage();
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
  if (!props.allowPositionDrag) return;
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

const courtHexPolygons = computed(() =>
  courtLayout.value.map((hex) => ({
    ...hex,
    points: hexPointsFor(hex.x, hex.y, HEX_RADIUS),
  }))
);

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
  return volumeFillFor(att, maxShotAttempts.value, 0.75);
}

function volumeFillFor(att, maxAttempts, alpha = 0.75) {
  const denom = Number(maxAttempts) > 0 ? Number(maxAttempts) : 1;
  const t = Math.max(0, Math.min(1, att / denom));
  // Lerp from deep navy to accent orange
  const start = [15, 23, 42];     // dark base
  const end = [251, 146, 60];     // warm accent
  const r = Math.round(start[0] + (end[0] - start[0]) * t);
  const g = Math.round(start[1] + (end[1] - start[1]) * t);
  const b = Math.round(start[2] + (end[2] - start[2]) * t);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

function volumeStroke(att) {
  return volumeStrokeFor(att, maxShotAttempts.value, 0.9);
}

function volumeStrokeFor(att, maxAttempts, alpha = 0.9) {
  const denom = Number(maxAttempts) > 0 ? Number(maxAttempts) : 1;
  const t = Math.max(0, Math.min(1, att / denom));
  // Slightly brighter stroke
  const start = [56, 189, 248];
  const end = [251, 191, 36];
  const r = Math.round(start[0] + (end[0] - start[0]) * t);
  const g = Math.round(start[1] + (end[1] - start[1]) * t);
  const b = Math.round(start[2] + (end[2] - start[2]) * t);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
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

function normalizeTrajectorySegment(rawSegment) {
  if (!rawSegment || typeof rawSegment !== 'object') return null;
  const start = Array.isArray(rawSegment.from) ? rawSegment.from : null;
  const end = Array.isArray(rawSegment.to) ? rawSegment.to : null;
  if (!start || !end || start.length < 2 || end.length < 2) return null;
  const startQ = Number(start[0]);
  const startR = Number(start[1]);
  const endQ = Number(end[0]);
  const endR = Number(end[1]);
  const count = Number(rawSegment.count || 0);
  if (
    [startQ, startR, endQ, endR, count].some((value) => Number.isNaN(value))
    || count <= 0
  ) {
    return null;
  }
  return {
    from: [startQ, startR],
    to: [endQ, endR],
    count,
  };
}

function buildArrowheadPoints(tipX, tipY, dirX, dirY, length, width) {
  const norm = Math.hypot(dirX, dirY) || 1;
  const ux = dirX / norm;
  const uy = dirY / norm;
  const px = -uy;
  const py = ux;
  const baseX = tipX - (ux * length);
  const baseY = tipY - (uy * length);
  const leftX = baseX + (px * width * 0.5);
  const leftY = baseY + (py * width * 0.5);
  const rightX = baseX - (px * width * 0.5);
  const rightY = baseY - (py * width * 0.5);
  return `${tipX},${tipY} ${leftX},${leftY} ${rightX},${rightY}`;
}

const hasPlaybookOverlay = computed(() => {
  const overlay = props.playbookOverlay;
  if (!overlay || typeof overlay !== 'object') return false;
  if (overlay.base_state) return true;
  const hasBallSegments = Array.isArray(overlay.ball_path_segments) && overlay.ball_path_segments.length > 0;
  const byPlayer = overlay.player_path_segments && typeof overlay.player_path_segments === 'object'
    ? Object.values(overlay.player_path_segments)
    : [];
  const hasPlayerSegments = byPlayer.some((segments) => Array.isArray(segments) && segments.length > 0);
  return hasBallSegments || hasPlayerSegments;
});

const showPlayers = computed(() => !hasShotCounts.value && !hasPlaybookOverlay.value);
const showValueAnnotations = computed(() => !hasShotCounts.value && !hasPlaybookOverlay.value);

const playbookPlayerColorMap = computed(() => {
  const gs = currentGameState.value;
  const offenseIds = Array.isArray(gs?.offense_ids) ? gs.offense_ids : [];
  const colorMap = {};
  offenseIds.forEach((rawId, idx) => {
    const pid = Number(rawId);
    if (!Number.isFinite(pid)) return;
    colorMap[pid] = PLAYBOOK_TRAJECTORY_COLORS[idx % PLAYBOOK_TRAJECTORY_COLORS.length];
  });
  return colorMap;
});

const playbookSelectedPlayerId = computed(() => {
  const raw = String(props.playbookOverlay?.display?.playerFilter || 'all');
  if (raw === 'all') return null;
  const pid = Number(raw);
  return Number.isFinite(pid) ? pid : null;
});

const playbookShowPlayerPaths = computed(() =>
  hasPlaybookOverlay.value && props.playbookOverlay?.display?.showPlayerPaths !== false
);

const playbookShowBallPaths = computed(() =>
  hasPlaybookOverlay.value && props.playbookOverlay?.display?.showBallPaths !== false
);

const playbookShowPassPaths = computed(() =>
  hasPlaybookOverlay.value && props.playbookOverlay?.display?.showPassPaths === true
);

const playbookShowShotHeatmap = computed(() =>
  hasPlaybookOverlay.value && props.playbookOverlay?.display?.showShotHeatmap === true
);

const playbookMaxPlayerSegmentCount = computed(() => {
  if (!hasPlaybookOverlay.value) return 1;
  let maxCount = 1;
  const byPlayer = props.playbookOverlay?.player_path_segments || {};
  for (const segments of Object.values(byPlayer)) {
    if (!Array.isArray(segments)) continue;
    for (const segment of segments) {
      const normalized = normalizeTrajectorySegment(segment);
      if (normalized && normalized.count > maxCount) {
        maxCount = normalized.count;
      }
    }
  }
  return maxCount;
});

const playbookMaxBallSegmentCount = computed(() => {
  if (!hasPlaybookOverlay.value) return 1;
  let maxCount = 1;
  const segments = Array.isArray(props.playbookOverlay?.ball_path_segments)
    ? props.playbookOverlay.ball_path_segments
    : [];
  for (const segment of segments) {
    const normalized = normalizeTrajectorySegment(segment);
    if (normalized && normalized.count > maxCount) {
      maxCount = normalized.count;
    }
  }
  return maxCount;
});

const playbookPlayerPathSegments = computed(() => {
  if (!playbookShowPlayerPaths.value) return [];
  const byPlayer = props.playbookOverlay?.player_path_segments || {};
  const segments = [];
  const colorMap = playbookPlayerColorMap.value;
  const selectedPid = playbookSelectedPlayerId.value;

  for (const [rawPid, rawSegments] of Object.entries(byPlayer)) {
    const pid = Number(rawPid);
    if (selectedPid !== null && pid !== selectedPid) continue;
    const color = colorMap[pid] || PLAYBOOK_TRAJECTORY_COLORS[Math.abs(pid) % PLAYBOOK_TRAJECTORY_COLORS.length];
    if (!Array.isArray(rawSegments)) continue;
    for (const rawSegment of rawSegments) {
      const normalized = normalizeTrajectorySegment(rawSegment);
      if (!normalized) continue;
      const start = axialToCartesian(normalized.from[0], normalized.from[1]);
      const end = axialToCartesian(normalized.to[0], normalized.to[1]);
      const ratio = Math.max(0, Math.min(1, normalized.count / playbookMaxPlayerSegmentCount.value));
      segments.push({
        key: `playbook-player-${pid}-${normalized.from.join(',')}-${normalized.to.join(',')}`,
        playerId: pid,
        color,
        count: normalized.count,
        x1: start.x,
        y1: start.y,
        x2: end.x,
        y2: end.y,
        strokeWidth: 1.5 + (6 * Math.sqrt(ratio)),
        opacity: 0.14 + (0.72 * Math.sqrt(ratio)),
      });
    }
  }

  segments.sort((a, b) => a.count - b.count);
  return segments;
});

const playbookBallPathSegments = computed(() => {
  if (!playbookShowBallPaths.value) return [];
  const rawSegments = Array.isArray(props.playbookOverlay?.ball_path_segments)
    ? props.playbookOverlay.ball_path_segments
    : [];
  const segments = [];
  for (const rawSegment of rawSegments) {
    const normalized = normalizeTrajectorySegment(rawSegment);
    if (!normalized) continue;
    const start = axialToCartesian(normalized.from[0], normalized.from[1]);
    const end = axialToCartesian(normalized.to[0], normalized.to[1]);
    const ratio = Math.max(0, Math.min(1, normalized.count / playbookMaxBallSegmentCount.value));
    segments.push({
      key: `playbook-ball-${normalized.from.join(',')}-${normalized.to.join(',')}`,
      count: normalized.count,
      x1: start.x,
      y1: start.y,
      x2: end.x,
      y2: end.y,
      strokeWidth: 2 + (7 * Math.sqrt(ratio)),
      opacity: 0.16 + (0.7 * Math.sqrt(ratio)),
    });
  }
  segments.sort((a, b) => a.count - b.count);
  return segments;
});

const playbookMaxPassSegmentCount = computed(() => {
  if (!hasPlaybookOverlay.value) return 1;
  let maxCount = 1;
  const segments = Array.isArray(props.playbookOverlay?.pass_path_segments)
    ? props.playbookOverlay.pass_path_segments
    : [];
  for (const segment of segments) {
    const normalized = normalizeTrajectorySegment(segment);
    if (normalized && normalized.count > maxCount) {
      maxCount = normalized.count;
    }
  }
  return maxCount;
});

const playbookPassPathSegments = computed(() => {
  if (!playbookShowPassPaths.value) return [];
  const rawSegments = Array.isArray(props.playbookOverlay?.pass_path_segments)
    ? props.playbookOverlay.pass_path_segments
    : [];
  const segments = [];
  for (const rawSegment of rawSegments) {
    const normalized = normalizeTrajectorySegment(rawSegment);
    if (!normalized) continue;
    const passerId = Number(rawSegment?.passer_id);
    const receiverId = Number(rawSegment?.receiver_id);
    if (
      playbookSelectedPlayerId.value !== null
      && Number.isFinite(passerId)
      && passerId !== playbookSelectedPlayerId.value
    ) {
      continue;
    }
    const start = axialToCartesian(normalized.from[0], normalized.from[1]);
    const end = axialToCartesian(normalized.to[0], normalized.to[1]);
    const dx = end.x - start.x;
    const dy = end.y - start.y;
    const norm = Math.hypot(dx, dy);
    if (norm <= 1e-6) continue;
    const hexDist = Math.max(1, hexDistance(normalized.from, normalized.to));
    const distanceScale = Math.pow(hexDist, 0.8);
    const ratio = Math.max(0, Math.min(1, normalized.count / playbookMaxPassSegmentCount.value));
    const startOffset = HEX_RADIUS * 0.45;
    const endOffset = HEX_RADIUS * 0.7;
    const headLength = HEX_RADIUS * (0.12 + (0.10 * distanceScale));
    const headWidth = HEX_RADIUS * (0.10 + (0.08 * distanceScale));
    const barHalfWidth = HEX_RADIUS * 0.21;
    const barStrokeWidth = 1.6 + (1.0 * distanceScale);
    const ux = dx / norm;
    const uy = dy / norm;
    const nx = -uy;
    const ny = ux;
    const startTipX = start.x + ((dx / norm) * startOffset);
    const startTipY = start.y + ((dy / norm) * startOffset);
    const endBarX = end.x - (ux * endOffset);
    const endBarY = end.y - (uy * endOffset);
    segments.push({
      key: `playbook-pass-${normalized.from.join(',')}-${normalized.to.join(',')}`,
      passerColor: playbookPlayerColorMap.value[passerId] || '#fbbf24',
      receiverColor: playbookPlayerColorMap.value[receiverId] || '#fbbf24',
      count: normalized.count,
      opacity: 0.18 + (0.5 * Math.sqrt(ratio)),
      startPoints: buildArrowheadPoints(startTipX, startTipY, dx, dy, headLength, headWidth),
      receiveBarX1: endBarX + (nx * barHalfWidth),
      receiveBarY1: endBarY + (ny * barHalfWidth),
      receiveBarX2: endBarX - (nx * barHalfWidth),
      receiveBarY2: endBarY - (ny * barHalfWidth),
      receiveBarStrokeWidth: barStrokeWidth,
    });
  }
  segments.sort((a, b) => a.count - b.count);
  return segments;
});

const playbookStartMarkers = computed(() => {
  if (!playbookShowPlayerPaths.value) return [];
  const gs = currentGameState.value;
  if (!gs || !Array.isArray(gs.positions)) return [];
  const offenseIds = Array.isArray(gs.offense_ids) ? gs.offense_ids : [];
  const ballHolder = Number(gs.ball_holder);
  const selectedPid = playbookSelectedPlayerId.value;
  return offenseIds
    .map((rawPid) => {
      const pid = Number(rawPid);
      if (selectedPid !== null && pid !== selectedPid) return null;
      const pos = gs.positions?.[pid];
      if (!Number.isFinite(pid) || !Array.isArray(pos) || pos.length < 2) return null;
      const point = axialToCartesian(Number(pos[0]), Number(pos[1]));
      return {
        key: `playbook-start-${pid}`,
        id: pid,
        x: point.x,
        y: point.y,
        color: playbookPlayerColorMap.value[pid] || PLAYBOOK_TRAJECTORY_COLORS[Math.abs(pid) % PLAYBOOK_TRAJECTORY_COLORS.length],
        hasBall: pid === ballHolder,
        label: getPlayerJerseyNumber(pid),
      };
    })
    .filter(Boolean);
});

const playbookShotHeatmapCounts = computed(() => {
  if (!playbookShowShotHeatmap.value) return {};
  const selectedPid = playbookSelectedPlayerId.value;
  const src = selectedPid === null
    ? (props.playbookOverlay?.shot_heatmap || {})
    : (props.playbookOverlay?.player_shot_heatmaps?.[String(selectedPid)] || {});
  const out = {};
  for (const [rawKey, val] of Object.entries(src)) {
    if (!Array.isArray(val) || val.length < 1) continue;
    const attempts = Number(val[0]) || 0;
    if (attempts <= 0) continue;
    const key = String(rawKey || '').trim();
    const [qStr, rStr] = key.split(',');
    const q = Number(qStr);
    const r = Number(rStr);
    if (Number.isNaN(q) || Number.isNaN(r)) continue;
    out[`${q},${r}`] = { attempts };
  }
  return out;
});

const playbookShotOverlayPoints = computed(() => {
  if (!playbookShowShotHeatmap.value) return [];
  const entries = Object.entries(playbookShotHeatmapCounts.value || {});
  entries.sort((a, b) => a[0].localeCompare(b[0]));
  return entries.map(([key, value]) => {
    const q = Number(key.split(',')[0]);
    const r = Number(key.split(',')[1]);
    const { x, y } = axialToCartesian(q, r);
    return {
      key,
      x,
      y,
      attempts: Number(value.attempts || 0),
    };
  });
});

const playbookMaxShotAttempts = computed(() => {
  let max = 0;
  for (const pt of playbookShotOverlayPoints.value) {
    if (pt.attempts > max) max = pt.attempts;
  }
  return max || 1;
});

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
  const rawAction = props.selectedActions[playerId];
  const action = typeof rawAction === 'string' ? rawAction : null;
  if (!action || action === 'NOOP') return null;
  
  const indicatorRadius = HEX_RADIUS * 0.55; // Distance from player center to indicator

  const gs = currentGameState.value;
  const resolvedTargetId = resolvePassTargetFromAction(gs, playerId, action);
  if (resolvedTargetId !== null && (hasBall || props.disableTransitions)) {
    const targetId = Number(resolvedTargetId);
    const targetPos = currentGameState.value?.positions?.[targetId];
    if (targetPos) {
      const target = axialToCartesian(targetPos[0], targetPos[1]);
      const dx = target.x - playerX;
      const dy = target.y - playerY;
      const rad = Math.atan2(dy, dx);
      const angleDeg = Math.atan2(dy, dx) * 180 / Math.PI;
      return {
        type: 'pass',
        x: playerX + indicatorRadius * Math.cos(rad),
        y: playerY + indicatorRadius * Math.sin(rad),
        rotation: angleDeg + 90,
      };
    }
  }
  
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
  
  if (action.startsWith('PASS_') && (hasBall || props.disableTransitions)) {
    const posAngle = POSITION_ANGLES[action];
    const rad = posAngle * Math.PI / 180;
    return {
      type: 'pass',
      x: playerX + indicatorRadius * Math.cos(rad),
      y: playerY + indicatorRadius * Math.sin(rad),
      rotation: ICON_ROTATIONS[action],
    };
  }
  
  if (action === 'SHOOT' && (hasBall || props.disableTransitions)) {
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

function toFiniteInt(value, fallback = 0) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return fallback;
  return Math.trunc(parsed);
}

function laneStepMapValue(map, playerId) {
  if (!map || typeof map !== 'object') return 0;
  const direct = map[playerId];
  if (direct !== undefined) return toFiniteInt(direct, 0);
  const byString = map[String(playerId)];
  return toFiniteInt(byString, 0);
}

function sideLaneStepsMax(map, sideIds) {
  if (!Array.isArray(sideIds) || sideIds.length === 0) return 0;
  let maxValue = 0;
  for (const rawId of sideIds) {
    const pid = toFiniteInt(rawId, Number.NaN);
    if (!Number.isFinite(pid)) continue;
    const value = laneStepMapValue(map, pid);
    if (value > maxValue) {
      maxValue = value;
    }
  }
  return Math.max(0, maxValue);
}

const laneStepIndicatorStacks = computed(() => {
  const gs = currentGameState.value;
  if (!gs || !basketPosition.value) return [];

  const offenseIds = Array.isArray(gs.offense_ids) ? gs.offense_ids : [];
  const defenseIds = Array.isArray(gs.defense_ids) ? gs.defense_ids : [];
  if (offenseIds.length === 0 && defenseIds.length === 0) return [];

  const maxSteps = Math.max(1, toFiniteInt(gs.three_second_max_steps, 3));
  const offenseSteps = sideLaneStepsMax(gs.offensive_lane_steps, offenseIds);
  const defenseSteps = sideLaneStepsMax(gs.defensive_lane_steps, defenseIds);

  const resolveSideColor = (sideIds, fallbackColor) => {
    if (!Array.isArray(sideIds)) return fallbackColor;
    for (const rawId of sideIds) {
      const pid = toFiniteInt(rawId, Number.NaN);
      if (!Number.isFinite(pid)) continue;
      const owner = getPlayerOwner(gs, pid);
      if (owner === 'user') return '#007bff';
      if (owner === 'ai') return '#dc3545';
    }
    return fallbackColor;
  };

  const offenseColor = resolveSideColor(offenseIds, '#007bff');
  const defenseColor = resolveSideColor(defenseIds, '#dc3545');

  const radius = HEX_RADIUS * 0.14;
  const spacing = HEX_RADIUS * 0.48;
  const topY = basketPosition.value.y - ((maxSteps - 1) * spacing) / 2;
  const columnGap = HEX_RADIUS * 0.62;
  const desiredDefenseX = basketPosition.value.x - HEX_RADIUS * 1.55;
  const desiredOffenseX = desiredDefenseX + columnGap;
  const minCenterX = courtBounds.value.minX + (HEX_RADIUS * 0.28);
  const shiftRight = Math.max(0, minCenterX - desiredDefenseX);
  const defenseX = desiredDefenseX + shiftRight;
  const offenseX = desiredOffenseX + shiftRight;

  const buildStack = (key, label, shortLabel, steps, color, x) => {
    const clamped = Math.max(0, Math.min(maxSteps, steps));
    const lights = [];
    for (let i = 0; i < maxSteps; i += 1) {
      lights.push({
        key: `${key}-light-${i}`,
        x,
        y: topY + (i * spacing),
        radius,
        lit: i < clamped,
        color,
        violation: steps >= maxSteps,
      });
    }
    return {
      key,
      label,
      shortLabel,
      color,
      steps: Math.max(0, steps),
      maxSteps,
      labelX: x,
      labelY: topY - (radius * 2.4),
      lights,
    };
  };

  return [
    buildStack('offense', 'Offense', 'O', offenseSteps, offenseColor, offenseX),
    buildStack('defense', 'Defense', 'D', defenseSteps, defenseColor, defenseX),
  ];
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
const isShotClockEditable = computed(
  () => !!props.allowShotClockAdjustment && !!currentGameState.value && !currentGameState.value.done
);
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

  if (props.disableBackendValueFetches) {
    passStealProbs.value = {};
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
    
    const margin = props.minimalChrome ? HEX_RADIUS * 1.8 : HEX_RADIUS * 3;
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

const courtBackdropRect = computed(() => {
  if (courtLayout.value.length === 0) return null;
  const { minX, maxX, minY, maxY } = courtBounds.value;
  const epsilon = 0.75;
  return {
    x: minX - HEX_HALF_WIDTH - epsilon,
    y: minY - HEX_RADIUS - epsilon,
    width: (maxX - minX) + (2 * HEX_HALF_WIDTH) + (2 * epsilon),
    height: (maxY - minY) + (2 * HEX_RADIUS) + (2 * epsilon),
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
const ballHandlerBaseProb = ref(null);
const DEFENDER_PRESSURE_SHAKE_EPS = 0.001;
const OFFENSE_SHELL_LOW = [17, 60, 131];
const OFFENSE_SHELL_HIGH = [30, 192, 30];
const OFFENSE_EP_COLOR_MIN = 0.5;
const OFFENSE_EP_COLOR_MAX = 2;

function clamp01(value) {
  const num = Number(value);
  if (!Number.isFinite(num)) return 0;
  return Math.max(0, Math.min(1, num));
}

function lerpChannel(a, b, t) {
  return Math.round(a + ((b - a) * t));
}

function offenseShellColor(playerId) {
  const epRaw = Number(currentGameState.value?.ep_by_player?.[playerId]);
  const ep = Number.isFinite(epRaw) ? epRaw : 0;
  const epRange = Math.max(1e-9, OFFENSE_EP_COLOR_MAX - OFFENSE_EP_COLOR_MIN);
  const t = clamp01((ep - OFFENSE_EP_COLOR_MIN) / epRange);
  const r = lerpChannel(OFFENSE_SHELL_LOW[0], OFFENSE_SHELL_HIGH[0], t);
  const g = lerpChannel(OFFENSE_SHELL_LOW[1], OFFENSE_SHELL_HIGH[1], t);
  const b = lerpChannel(OFFENSE_SHELL_LOW[2], OFFENSE_SHELL_HIGH[2], t);
  return `rgb(${r}, ${g}, ${b})`;
}

function playerTeamClass(player) {
  if (player?.owner === 'user') return 'player-user';
  if (player?.owner === 'ai') return 'player-ai';
  return player?.isOffense ? 'player-offense' : 'player-defense';
}

function playerCircleStyle(player) {
  const style = { cursor: 'grab' };
  if (player?.owner === 'user' || player?.owner === 'ai') {
    return style;
  }
  if (player?.isOffense) {
    style.fill = offenseShellColor(player.id);
  }
  return style;
}

function shotPressureColor(pressureReduction, maxReduction = 1) {
  const denom = Number.isFinite(Number(maxReduction)) && Number(maxReduction) > 0
    ? Number(maxReduction)
    : 1;
  const p = clamp01(Number(pressureReduction) / denom);
  // 0.0 => orange (low pressure), 1.0 => red (high pressure)
  const hue = 28 * (1 - p);
  return `hsl(${hue}, 92%, 54%)`;
}

const dominantShotPressureDefender = computed(() => {
  const gs = currentGameState.value;
  if (!gs || gs.ball_holder === null || gs.ball_holder === undefined) return null;

  const shooterId = Number(gs.ball_holder);
  if (!Number.isFinite(shooterId)) return null;

  const positions = Array.isArray(gs.positions) ? gs.positions : null;
  const shooterPos = positions?.[shooterId];
  const basketPos = Array.isArray(gs.basket_position) ? gs.basket_position : null;
  if (!Array.isArray(shooterPos) || !Array.isArray(basketPos)) return null;

  const dir = axialToCartesian(
    Number(basketPos[0]) - Number(shooterPos[0]),
    Number(basketPos[1]) - Number(shooterPos[1]),
  );
  const dirNorm = Math.hypot(dir.x, dir.y) || 1;
  const arcDegRaw = Number(gs.shot_pressure_arc_degrees ?? 60);
  const arcDeg = Number.isFinite(arcDegRaw) ? Math.max(1, Math.min(360, arcDegRaw)) : 60;
  const cosThreshold = Math.cos((arcDeg * Math.PI) / 360);
  const distanceToBasket = hexDistance(shooterPos, basketPos);

  const offenseIds = Array.isArray(gs.offense_ids) ? gs.offense_ids : [];
  const defenseIds = Array.isArray(gs.defense_ids) ? gs.defense_ids : [];
  const opponentIds = offenseIds.includes(shooterId) ? defenseIds : offenseIds;

  const maxRaw = Number(gs.shot_pressure_max ?? 0.5);
  const shotPressureMax = Number.isFinite(maxRaw) ? Math.max(0, maxRaw) : 0.5;
  const lambdaRaw = Number(gs.shot_pressure_lambda ?? 1.0);
  const shotPressureLambda = Number.isFinite(lambdaRaw) ? Math.max(0, lambdaRaw) : 1.0;

  let best = null;
  for (const rawDefenderId of opponentIds) {
    const defenderId = Number(rawDefenderId);
    if (!Number.isFinite(defenderId)) continue;
    const defenderPos = positions?.[defenderId];
    if (!Array.isArray(defenderPos)) continue;

    const dq = Number(defenderPos[0]) - Number(shooterPos[0]);
    const dr = Number(defenderPos[1]) - Number(shooterPos[1]);
    const v = axialToCartesian(dq, dr);
    const vNorm = Math.hypot(v.x, v.y);
    if (vNorm === 0) continue;

    const cosang = (v.x * dir.x + v.y * dir.y) / (vNorm * dirNorm);
    const dDef = hexDistance(shooterPos, defenderPos);
    if (cosang < cosThreshold || dDef > distanceToBasket) continue;

    const angleFactor = cosThreshold < 1 ? (cosang - cosThreshold) / (1 - cosThreshold) : 1;
    const distanceReduction = shotPressureMax * Math.exp(-shotPressureLambda * (dDef - 1));
    const reduction = clamp01(distanceReduction * (angleFactor ** 2));

    if (!best || reduction > best.reduction) {
      best = { defenderId, reduction };
    }
  }
  return best;
});

const ballHandlerPressureReduction = computed(() => {
  const rawReduction = clamp01(Number(dominantShotPressureDefender.value?.reduction ?? 0));
  // Treat tiny numeric noise as no pressure so the ring disappears cleanly.
  return rawReduction <= 0.01 ? 0 : rawReduction;
});

const ballHandlerShotPressureRing = computed(() => {
  const gs = currentGameState.value;
  if (!gs || !showShotPressureRing.value) return null;
  if (!gs.shot_pressure_enabled) return null;
  if (gs.ball_holder === null || gs.ball_holder === undefined) return null;

  const pressureReduction = Number(ballHandlerPressureReduction.value);
  if (!Number.isFinite(pressureReduction) || pressureReduction <= 0) return null;

  const pid = Number(dominantShotPressureDefender.value?.defenderId);
  if (!Number.isFinite(pid)) return null;

  const scaleDenom = Number(gs.shot_pressure_max);
  const normalizedPressure = clamp01(
    pressureReduction / (Number.isFinite(scaleDenom) && scaleDenom > 0 ? scaleDenom : 1),
  );
  const opacity = 0.35 + (0.55 * normalizedPressure);
  const strokeWidth = HEX_RADIUS * (0.065 + 0.1 * normalizedPressure);
  const playerMarkerRadius = HEX_RADIUS * 0.8;
  const ringGapPx = 1.0;
  const ringRadius = playerMarkerRadius + ringGapPx + (strokeWidth / 2);
  const pressurePct = pressureReduction * 100;
  const pulseEnabled = normalizedPressure >= 0.2;
  const pulseDurationSec = 1.7 - (0.9 * normalizedPressure);
  const pulseScale = 1.0 + (0.02 + 0.05 * normalizedPressure);
  const basePct = Number.isFinite(Number(ballHandlerBaseProb.value))
    ? Number(ballHandlerBaseProb.value) * 100
    : null;
  const finalPct = Number.isFinite(Number(ballHandlerMakeProb.value))
    ? Number(ballHandlerMakeProb.value) * 100
    : null;

  return {
    playerId: pid,
    color: shotPressureColor(pressureReduction, scaleDenom),
    opacity,
    strokeWidth,
    radius: ringRadius,
    normalizedPressure,
    pulseEnabled,
    pulseDurationSec,
    pulseScale,
    pressurePct,
    basePct,
    finalPct,
  };
});

const defenderTurnoverPressureById = computed(() => {
  const out = {};
  const gs = currentGameState.value;
  if (!gs) return out;

  const setPressure = (defenderId, turnoverProb) => {
    const did = Number(defenderId);
    const prob = clamp01(Number(turnoverProb));
    if (!Number.isFinite(did) || prob <= DEFENDER_PRESSURE_SHAKE_EPS) return;
    const existing = Number(out[did] || 0);
    out[did] = Math.max(existing, prob);
  };

  // Primary source: backend's per-step defender pressure diagnostics.
  const pressureByBallHandler = gs.last_action_results?.defender_pressure;
  if (pressureByBallHandler && typeof pressureByBallHandler === 'object') {
    const ballHolderId = gs.ball_holder;
    const ballHolderNum = Number(ballHolderId);
    const ballHandlerPressure = pressureByBallHandler[ballHolderId]
      ?? pressureByBallHandler[String(ballHolderId)]
      ?? (Number.isFinite(ballHolderNum) ? pressureByBallHandler[ballHolderNum] : null);
    const defenders = Array.isArray(ballHandlerPressure?.defenders) ? ballHandlerPressure.defenders : [];
    for (const defenderInfo of defenders) {
      setPressure(defenderInfo?.defender_id, defenderInfo?.turnover_prob);
    }
  }

  // Fallback source: replicate backend geometry so pressure is still visible
  // if last_action_results is stale or missing in the current render frame.
  const ballHolderId = Number(gs.ball_holder);
  const offenseIds = Array.isArray(gs.offense_ids) ? gs.offense_ids.map(Number) : [];
  const defenseIds = Array.isArray(gs.defense_ids) ? gs.defense_ids.map(Number) : [];
  const positions = Array.isArray(gs.positions) ? gs.positions : null;
  const basketPos = Array.isArray(gs.basket_position) ? gs.basket_position : null;
  if (!Number.isFinite(ballHolderId) || !positions || !Array.isArray(basketPos)) return out;
  if (!offenseIds.includes(ballHolderId)) return out;
  const ballPos = positions?.[ballHolderId];
  if (!Array.isArray(ballPos)) return out;

  const toBasket = axialToCartesian(
    Number(basketPos[0]) - Number(ballPos[0]),
    Number(basketPos[1]) - Number(ballPos[1]),
  );
  const basketNorm = Math.hypot(toBasket.x, toBasket.y);
  if (basketNorm <= 0) return out;

  const distLimitRaw = Number(gs.defender_pressure_distance ?? 1);
  const distLimit = Number.isFinite(distLimitRaw) ? Math.max(0, distLimitRaw) : 1;
  const baseTurnoverRaw = Number(gs.defender_pressure_turnover_chance ?? 0.05);
  const baseTurnover = Number.isFinite(baseTurnoverRaw) ? Math.max(0, baseTurnoverRaw) : 0.05;
  const decayRaw = Number(gs.defender_pressure_decay_lambda ?? 1.0);
  const decayLambda = Number.isFinite(decayRaw) ? Math.max(0, decayRaw) : 1.0;

  for (const defenderIdRaw of defenseIds) {
    const defenderId = Number(defenderIdRaw);
    if (!Number.isFinite(defenderId)) continue;
    const defenderPos = positions?.[defenderId];
    if (!Array.isArray(defenderPos)) continue;

    const dDef = hexDistance(ballPos, defenderPos);
    if (!Number.isFinite(dDef) || dDef > distLimit) continue;

    const toDefender = axialToCartesian(
      Number(defenderPos[0]) - Number(ballPos[0]),
      Number(defenderPos[1]) - Number(ballPos[1]),
    );
    const defenderNorm = Math.hypot(toDefender.x, toDefender.y);
    if (defenderNorm <= 0) continue;

    const cosAngle = ((toBasket.x * toDefender.x) + (toBasket.y * toDefender.y)) / (basketNorm * defenderNorm);
    if (!Number.isFinite(cosAngle) || cosAngle < 0) continue;

    const turnoverProb = baseTurnover * Math.exp(-decayLambda * Math.max(0, dDef - 1));
    setPressure(defenderId, turnoverProb);
  }

  return out;
});

const ballHandlerTurnoverPressureMeter = computed(() => {
  const gs = currentGameState.value;
  const ballHolderId = Number(gs?.ball_holder);
  if (!gs || !Number.isFinite(ballHolderId)) {
    return {
      actualProb: 0,
      normalized: 0,
      maxPossible: 0,
      color: '#f59e0b',
      title: 'Turnover pressure 0.0%',
    };
  }

  const offenseIds = Array.isArray(gs.offense_ids) ? gs.offense_ids.map(Number).filter((id) => Number.isFinite(id)) : [];
  const defenseIds = Array.isArray(gs.defense_ids) ? gs.defense_ids.map(Number).filter((id) => Number.isFinite(id)) : [];
  const defenderIds = offenseIds.includes(ballHolderId) ? defenseIds : offenseIds;

  const perDefenderMap = defenderTurnoverPressureById.value || {};
  let complement = 1.0;
  for (const defenderId of defenderIds) {
    const p = clamp01(
      Number(
        perDefenderMap[defenderId]
        ?? perDefenderMap[String(defenderId)]
        ?? 0,
      ),
    );
    complement *= (1.0 - p);
  }
  const actualProb = clamp01(1.0 - complement);

  const baseChance = clamp01(Number(gs.defender_pressure_turnover_chance ?? 0.05));
  const maxPossible = defenderIds.length > 0
    ? clamp01(1.0 - Math.pow(1.0 - baseChance, defenderIds.length))
    : 0;
  const normalized = maxPossible > 1e-6 ? clamp01(actualProb / maxPossible) : 0;

  const hue = Math.round(48 * (1 - normalized)); // yellow -> red as pressure rises
  const color = `hsl(${hue}, 100%, 58%)`;
  const title = `Turnover pressure ${(actualProb * 100).toFixed(1)}% (meter ${(normalized * 100).toFixed(1)}% of max ${(maxPossible * 100).toFixed(1)}%)`;

  return {
    actualProb,
    normalized,
    maxPossible,
    color,
    title,
  };
});

function getDefenderTurnoverPressure(playerId) {
  const pressure = Number(
    defenderTurnoverPressureById.value[playerId]
      ?? defenderTurnoverPressureById.value[String(playerId)]
      ?? 0,
  );
  return Number.isFinite(pressure) ? Math.max(0, pressure) : 0;
}

function getDefenderPressureNormalized(turnoverPressure) {
  const gs = currentGameState.value;
  const base = Number(gs?.defender_pressure_turnover_chance);
  const denom = Number.isFinite(base) && base > 0 ? base : 0.05;
  return clamp01(Number(turnoverPressure) / denom);
}

function getDefenderPressureShakeStyle(playerId) {
  const turnoverPressure = getDefenderTurnoverPressure(playerId);
  if (turnoverPressure <= DEFENDER_PRESSURE_SHAKE_EPS) return {};

  const normalized = getDefenderPressureNormalized(turnoverPressure);
  const ampPx = 0.55 + (2.4 * normalized);
  const durationSec = 0.40 - (0.12 * normalized);
  const rotDeg = 0.18 + (0.95 * normalized);

  return {
    '--defender-pressure-shake-amp': `${ampPx.toFixed(2)}px`,
    '--defender-pressure-shake-duration': `${Math.max(0.20, durationSec).toFixed(3)}s`,
    '--defender-pressure-shake-rot': `${rotDeg.toFixed(2)}deg`,
  };
}

function formatShotPressureRingTitle(ring) {
  if (!ring) return '';
  const pressure = Number(ring.pressurePct);
  const basePct = Number(ring.basePct);
  const finalPct = Number(ring.finalPct);
  const defenderLabel = `Defender ${Number(ring.playerId)}`;
  if (Number.isFinite(basePct) && Number.isFinite(finalPct)) {
    return `${defenderLabel} pressure ${pressure.toFixed(1)}% (${basePct.toFixed(1)}% -> ${finalPct.toFixed(1)}%)`;
  }
  return `${defenderLabel} pressure ${pressure.toFixed(1)}%`;
}

function formatDefenderTurnoverPressureTitle(playerId) {
  const pressure = getDefenderTurnoverPressure(playerId);
  if (!Number.isFinite(pressure) || pressure <= DEFENDER_PRESSURE_SHAKE_EPS) return '';
  return `Turnover pressure ${(pressure * 100).toFixed(1)}%`;
}

function getPlayerRenderCenter(player) {
  if (!player) return { x: 0, y: 0 };
  if (draggedPlayerId.value === player.id) {
    return { x: draggedPlayerPos.x, y: draggedPlayerPos.y };
  }
  return { x: player.x, y: player.y };
}

function getBallPressureMeterPath(player) {
  const center = getPlayerRenderCenter(player);
  const r = HEX_RADIUS * 0.9;
  const x = Number(center.x || 0);
  const y = Number(center.y || 0);
  return `M ${x} ${y - r} A ${r} ${r} 0 0 1 ${x} ${y + r}`;
}

function getBallPressureSeamPath(player) {
  const center = getPlayerRenderCenter(player);
  const r = HEX_RADIUS * 0.9;
  const x = Number(center.x || 0);
  const y = Number(center.y || 0);
  // Left-side seam from 6 -> 12 o'clock (opposite the TO meter arc).
  return `M ${x} ${y + r} A ${r} ${r} 0 0 1 ${x} ${y - r}`;
}

function getBallPressureMeterStyle() {
  const meter = ballHandlerTurnoverPressureMeter.value;
  const r = HEX_RADIUS * 0.9;
  const halfCircumference = Math.PI * r;
  const filled = Math.max(0, Math.min(halfCircumference, halfCircumference * Number(meter?.normalized || 0)));
  return {
    stroke: meter?.color || '#f59e0b',
    strokeDasharray: `${filled.toFixed(2)} ${halfCircumference.toFixed(2)}`,
  };
}

function formatBallPressureMeterLabel() {
  const meter = ballHandlerTurnoverPressureMeter.value;
  const pct = Math.max(0, Math.min(100, Number(meter?.actualProb || 0) * 100));
  return `TO ${pct.toFixed(1)}%`;
}

function formatPlayerTooltipTitle(playerId) {
  const parts = [];
  const ring = ballHandlerShotPressureRing.value;
  if (ring && Number(ring.playerId) === Number(playerId)) {
    parts.push(formatShotPressureRingTitle(ring));
  }
  const turnoverLabel = formatDefenderTurnoverPressureTitle(playerId);
  if (turnoverLabel) parts.push(turnoverLabel);
  return parts.join(' | ');
}

async function fetchBallHandlerMakeProb() {
  const gs = currentGameState.value;
  if (!gs || gs.ball_holder === null) {
    ballHandlerMakeProb.value = null;
    ballHandlerBaseProb.value = null;
    return;
  }
  
  // Prefer the backend-provided state value when available.
  if (typeof gs.ball_handler_shot_probability === 'number') {
    ballHandlerMakeProb.value = gs.ball_handler_shot_probability;
    // Replay snapshots currently store final probability only.
    ballHandlerBaseProb.value = null;
    console.log('[GameBoard] Using stored shot probability from state:', gs.ball_handler_shot_probability);
    return;
  }

  if (props.disableBackendValueFetches) {
    ballHandlerMakeProb.value = null;
    ballHandlerBaseProb.value = null;
    return;
  }
  
  // Otherwise, fetch from API (live gameplay)
  try {
    const resp = await getShotProbability(gs.ball_holder);
    const base = resp?.shot_probability ?? null;
    const p = resp?.shot_probability_final ?? resp?.shot_probability ?? null;
    ballHandlerBaseProb.value = typeof base === 'number' ? base : null;
    ballHandlerMakeProb.value = typeof p === 'number' ? p : null;
  } catch (e) {
    console.warn('[GameBoard] Failed to fetch ball-handler make prob', e);
    ballHandlerMakeProb.value = null;
    ballHandlerBaseProb.value = null;
  }
}

onMounted(() => {
  fetchBallHandlerMakeProb();
  window.addEventListener('mousedown', onGlobalMouseDown);
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

function resolveOutcomePointForPlayer(playerId) {
    const pid = Number(playerId);
    if (!Number.isFinite(pid)) return null;
    const playerPos = currentGameState.value?.positions?.[pid];
    if (!Array.isArray(playerPos) || playerPos.length < 2) return null;
    const [pq, pr] = playerPos;
    return axialToCartesian(pq, pr);
}

function normalizeForcedEpisodeOutcome(rawOutcome) {
    if (!rawOutcome || typeof rawOutcome !== 'object' || !currentGameState.value) return null;
    const type = String(rawOutcome.type || '').trim().toUpperCase();
    if (!type) return null;

    const allowedTypes = new Set([
        'DEFENSIVE_VIOLATION',
        'OFFENSIVE_VIOLATION',
        'TURNOVER',
        'SHOT_CLOCK_VIOLATION',
        'MADE_SHOT',
        'MISSED_SHOT',
    ]);
    if (!allowedTypes.has(type)) return null;

    const outcome = { type };
    const forcedPlayerId = Number(rawOutcome.playerId ?? rawOutcome.player_id);
    if (Number.isFinite(forcedPlayerId)) {
        outcome.playerId = forcedPlayerId;
    }

    const forcedX = Number(rawOutcome.x);
    const forcedY = Number(rawOutcome.y);
    if (Number.isFinite(forcedX) && Number.isFinite(forcedY)) {
        outcome.x = forcedX;
        outcome.y = forcedY;
    } else if (Number.isFinite(forcedPlayerId)) {
        const point = resolveOutcomePointForPlayer(forcedPlayerId);
        if (point) {
            outcome.x = point.x;
            outcome.y = point.y;
        }
    }

    if (type === 'MADE_SHOT' || type === 'MISSED_SHOT') {
        outcome.isThree = Boolean(rawOutcome.isThree ?? rawOutcome.is_three ?? false);
        outcome.isDunk = Boolean(rawOutcome.isDunk ?? rawOutcome.is_dunk ?? false);
    }

    return outcome;
}

const episodeOutcome = computed(() => {
    const forcedOutcome = normalizeForcedEpisodeOutcome(props.forcedEpisodeOutcome);
    if (forcedOutcome) return forcedOutcome;

    if (!currentGameState.value || !currentGameState.value.done) {
        return null; // Game is not over
    }

    const results = currentGameState.value.last_action_results;
    if (!results) return null;

    const resolveViolationPosition = (violation) => {
        if (!violation || typeof violation !== 'object') return null;
        if (Array.isArray(violation.position) && violation.position.length >= 2) {
            const [vq, vr] = violation.position;
            return axialToCartesian(vq, vr);
        }
        return resolveOutcomePointForPlayer(violation.player_id);
    };

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
        const point = resolveViolationPosition(violation);
        if (point) {
            return { type: 'DEFENSIVE_VIOLATION', x: point.x, y: point.y, playerId: violation.player_id };
        }
        return { type: 'DEFENSIVE_VIOLATION', playerId: violation?.player_id };
    }

    // Check for offensive lane violations
    if (results.offensive_lane_violations && results.offensive_lane_violations.length > 0) {
        const violation = results.offensive_lane_violations[0];
        const point = resolveViolationPosition(violation);
        if (point) {
            return { type: 'OFFENSIVE_VIOLATION', x: point.x, y: point.y, playerId: violation.player_id };
        }
        return { type: 'OFFENSIVE_VIOLATION', playerId: violation?.player_id };
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

    if (allTurnovers.length > 0) {
        const turnover = allTurnovers[0] && typeof allTurnovers[0] === 'object' ? allTurnovers[0] : {};
        const outcome = {
            type: 'TURNOVER',
            playerId: turnover.player_id,
            stolenBy: turnover.stolen_by,
            reason: turnover.reason,
        };
        if (Array.isArray(turnover.turnover_pos) && turnover.turnover_pos.length >= 2) {
            const { x, y } = axialToCartesian(turnover.turnover_pos[0], turnover.turnover_pos[1]);
            outcome.x = x;
            outcome.y = y;
        }
        return outcome;
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
        const owner = getPlayerOwner(currentGameState, playerId);
        const arrowTone = owner === 'user' || owner === 'ai'
          ? owner
          : (isOffense ? 'user' : 'ai');

        transitions.push({
          key: `arrow-${step}-${playerId}`,
          startX,
          startY,
          endX,
          endY,
          opacity,
          isOffense,
          arrowTone,
        });
      }
    }
  }
  return transitions;
});

// Compute pass rays from ball handler to teammates with pass success probabilities
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
    const stealFraction = Math.max(0, Math.min(1, stealProb > 1 ? stealProb / 100 : stealProb));
    const passSuccessFraction = 1 - stealFraction;
    const passSuccessPercent = hasProb ? passSuccessFraction * 100 : null;
    const passSuccessLabel = hasProb ? `${passSuccessPercent.toFixed(1)}%` : '—';
    // Keep original visual emphasis behavior (based on steal probability),
    // while displaying pass-success percentage text.
    const passSuccessOpacity = hasProb ? probToStealAlpha(stealFraction) : 0.3;
    
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
      passSuccessProb: passSuccessPercent, // Rounded to nearest percent (nullable)
      passSuccessLabel,
      passSuccessOpacity,
      teammateId,
    });
  }
  
  return rays;
});

// Preview which teammate will receive a pass based on current selection/strategy
const passTargetPreview = computed(() => {
  const gs = currentGameState.value;
  if (!gs || gs.ball_holder === null || gs.ball_holder === undefined) return null;

  let passerId = gs.ball_holder;
  let action = props.selectedActions?.[passerId];
  // During capture, selected actions are historical actions_taken for the rendered step.
  // A successful pass changes ball_holder, so recover passer from selected actions.
  if (props.disableTransitions && props.selectedActions) {
    const passEntries = Object.entries(props.selectedActions).filter(
      ([, act]) => typeof act === 'string' && /^PASS(?:_|->)/.test(act)
    );
    if (passEntries.length === 1) {
      passerId = Number(passEntries[0][0]);
      action = passEntries[0][1];
    }
  }
  if (!action || !/^PASS(?:_|->)/.test(action)) return null;

  const pointerTargetId = resolvePassTargetFromAction(gs, passerId, action);
  if (pointerTargetId !== null) {
    const passerPos = gs.positions?.[passerId];
    const recvPos = gs.positions?.[pointerTargetId];
    if (!passerPos || !recvPos) return null;
    const passerCoords = axialToCartesian(passerPos[0], passerPos[1]);
    const recvCoords = axialToCartesian(recvPos[0], recvPos[1]);
    return {
      passerId,
      receiverId: pointerTargetId,
      start: passerCoords,
      end: recvCoords,
      strategy: 'pointer_targeted',
      distance: hexDistance(passerPos, recvPos),
      value: null,
      stealProb: (passStealProbs.value?.[pointerTargetId] ?? null),
      ep: (gs.ep_by_player && gs.ep_by_player[pointerTargetId] !== undefined)
        ? Number(gs.ep_by_player[pointerTargetId])
        : null,
    };
  }

  if (!action.startsWith('PASS_')) return null;

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

const normalizedPassAnimationStyle = computed(() => {
  const mode = String(props.passAnimationStyle || 'projectile').toLowerCase();
  return mode === 'ray' ? 'ray' : 'projectile';
});

const passFlashProgress = computed(() => {
  const flash = passFlash.value;
  if (!flash) return 0;
  if (props.disableTransitions) {
    return Math.max(0, Math.min(1, Number(props.moveProgress ?? 1)));
  }
  const startedAt = Number(flash.startedAtMs ?? passFlashNowMs.value ?? performance.now());
  const elapsed = Math.max(0, Number(passFlashNowMs.value ?? performance.now()) - startedAt);
  return Math.max(0, Math.min(1, elapsed / PASS_FLASH_DURATION_MS));
});

const passFlashOpacity = computed(() => {
  const t = passFlashProgress.value;
  if (t <= 0.75) return 1;
  return Math.max(0, 1 - (t - 0.75) / 0.25);
});

const passProjectile = computed(() => {
  const flash = passFlash.value;
  if (!flash) return null;

  const dx = flash.x2 - flash.x1;
  const dy = flash.y2 - flash.y1;
  const dist = Math.hypot(dx, dy);
  if (!Number.isFinite(dist) || dist < 1e-6) return null;

  const ux = dx / dist;
  const uy = dy / dist;
  const px = -uy;
  const py = ux;

  const t = passFlashProgress.value;
  const eased = 1 - ((1 - t) * (1 - t));
  const tipX = flash.x1 + dx * eased;
  const tipY = flash.y1 + dy * eased;

  const headLength = Math.min(HEX_RADIUS * 0.92, dist * 0.20) * PROJECTILE_ARROW_LENGTH_SCALE;
  const headHalfWidth = headLength * 0.58;
  const headBaseX = tipX - ux * headLength;
  const headBaseY = tipY - uy * headLength;

  const leftX = headBaseX + px * headHalfWidth;
  const leftY = headBaseY + py * headHalfWidth;
  const rightX = headBaseX - px * headHalfWidth;
  const rightY = headBaseY - py * headHalfWidth;

  const shaftLength = Math.min(HEX_RADIUS * 1.45, dist * 0.30) * PROJECTILE_ARROW_LENGTH_SCALE;
  const shaftX1 = headBaseX;
  const shaftY1 = headBaseY;
  const shaftX2 = headBaseX - ux * shaftLength;
  const shaftY2 = headBaseY - uy * shaftLength;

  const impactPhase = Math.max(0, Math.min(1, (eased - 0.82) / 0.18));
  const impactOpacity = impactPhase > 0 ? (1 - impactPhase) * passFlashOpacity.value : 0;
  const impactRadius = HEX_RADIUS * (0.35 + impactPhase * 1.05);

  return {
    laneOpacity: 0.22 + passFlashOpacity.value * 0.28,
    projectileOpacity: 0.35 + passFlashOpacity.value * 0.65,
    shaftX1,
    shaftY1,
    shaftX2,
    shaftY2,
    headPoints: `${tipX},${tipY} ${leftX},${leftY} ${rightX},${rightY}`,
    impactX: flash.x2,
    impactY: flash.y2,
    impactOpacity,
    impactRadius,
  };
});

const passBallOutline = computed(() => {
  const flash = passFlash.value;
  if (!flash) return null;

  const dx = flash.x2 - flash.x1;
  const dy = flash.y2 - flash.y1;
  const dist = Math.hypot(dx, dy);
  if (!Number.isFinite(dist) || dist < 1e-6) return null;

  const t = passFlashProgress.value;
  const eased = 1 - ((1 - t) * (1 - t));
  const x = flash.x1 + dx * eased;
  const y = flash.y1 + dy * eased;
  const radius = HEX_RADIUS * (0.7 + 0.06 * Math.sin(Math.PI * t));
  const edgeFade = Math.max(0, Math.min(1, t / 0.12, (1 - t) / 0.12));
  const opacity = edgeFade * (0.6 + 0.4 * passFlashOpacity.value);
  const dashOffset = (1 - t) * 22;

  return {
    x,
    y,
    radius,
    opacity,
    dashOffset,
  };
});

function stopPassFlashClock() {
  if (passFlashRaf.value !== null) {
    cancelAnimationFrame(passFlashRaf.value);
    passFlashRaf.value = null;
  }
}

function startPassFlashClock() {
  stopPassFlashClock();
  const tick = (ts) => {
    passFlashNowMs.value = ts;
    if (passFlash.value && !props.disableTransitions) {
      passFlashRaf.value = requestAnimationFrame(tick);
    } else {
      passFlashRaf.value = null;
    }
  };
  passFlashRaf.value = requestAnimationFrame(tick);
}

function clearPassFlash() {
  if (passFlashTimeout.value) {
    clearTimeout(passFlashTimeout.value);
    passFlashTimeout.value = null;
  }
  stopPassFlashClock();
  passFlash.value = null;
}

function triggerPassFlash(passerId, receiverId, start, end) {
  if (passFlashTimeout.value) {
    clearTimeout(passFlashTimeout.value);
    passFlashTimeout.value = null;
  }

  const startedAtMs = performance.now();
  passFlashNowMs.value = startedAtMs;
  passFlash.value = {
    flashKey: ++passFlashSerial,
    startedAtMs,
    passerId,
    receiverId,
    x1: start.x,
    y1: start.y,
    x2: end.x,
    y2: end.y,
    labelX: (start.x + end.x) / 2,
    labelY: (start.y + end.y) / 2 - HEX_RADIUS * 0.6,
  };
  if (!props.disableTransitions) {
    startPassFlashClock();
  }

  passFlashTimeout.value = setTimeout(() => {
    passFlash.value = null;
    passFlashTimeout.value = null;
  }, PASS_FLASH_DURATION_MS);
}

function buildShotArcGeometry(start, end) {
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

  return {
    path: `M ${start.x} ${start.y} Q ${controlX} ${controlY} ${end.x} ${end.y}`,
    controlX,
    controlY,
  };
}

function quadraticBezierPoint(start, control, end, t) {
  const clampedT = Math.max(0, Math.min(1, t));
  const oneMinusT = 1 - clampedT;
  return {
    x:
      oneMinusT * oneMinusT * start.x +
      2 * oneMinusT * clampedT * control.x +
      clampedT * clampedT * end.x,
    y:
      oneMinusT * oneMinusT * start.y +
      2 * oneMinusT * clampedT * control.y +
      clampedT * clampedT * end.y,
  };
}

function quadraticBezierTangent(start, control, end, t) {
  const clampedT = Math.max(0, Math.min(1, t));
  return {
    x: 2 * (1 - clampedT) * (control.x - start.x) + 2 * clampedT * (end.x - control.x),
    y: 2 * (1 - clampedT) * (control.y - start.y) + 2 * clampedT * (end.y - control.y),
  };
}

const shotFlashProgress = computed(() => {
  const flash = shotFlash.value;
  if (!flash) return 0;
  if (props.disableTransitions) {
    return Math.max(0, Math.min(1, Number(props.moveProgress ?? 1)));
  }
  const startedAt = Number(flash.startedAtMs ?? shotFlashNowMs.value ?? performance.now());
  const elapsed = Math.max(0, Number(shotFlashNowMs.value ?? performance.now()) - startedAt);
  return Math.max(0, Math.min(1, elapsed / SHOT_FLASH_DURATION_MS));
});

const shotFlashOpacity = computed(() => {
  const t = shotFlashProgress.value;
  if (t <= 0.75) return 1;
  return Math.max(0, 1 - (t - 0.75) / 0.25);
});

const shotLaneOpacity = computed(() => 0.22 + shotFlashOpacity.value * 0.28);

const shotProjectile = computed(() => {
  const flash = shotFlash.value;
  if (!flash) return null;

  const start = { x: flash.x1, y: flash.y1 };
  const control = { x: flash.controlX, y: flash.controlY };
  const end = { x: flash.x2, y: flash.y2 };
  const t = shotFlashProgress.value;
  const eased = 1 - ((1 - t) * (1 - t));

  const tip = quadraticBezierPoint(start, control, end, eased);
  const tangent = quadraticBezierTangent(start, control, end, eased);
  const tangentNorm = Math.hypot(tangent.x, tangent.y) || 1;
  const ux = tangent.x / tangentNorm;
  const uy = tangent.y / tangentNorm;
  const px = -uy;
  const py = ux;

  const totalDist = Math.hypot(end.x - start.x, end.y - start.y);
  const headLength = Math.min(HEX_RADIUS * 0.95, totalDist * 0.18) * PROJECTILE_ARROW_LENGTH_SCALE;
  const headHalfWidth = headLength * 0.56;
  const headBaseX = tip.x - ux * headLength;
  const headBaseY = tip.y - uy * headLength;
  const leftX = headBaseX + px * headHalfWidth;
  const leftY = headBaseY + py * headHalfWidth;
  const rightX = headBaseX - px * headHalfWidth;
  const rightY = headBaseY - py * headHalfWidth;

  const shaftLength = Math.min(HEX_RADIUS * 1.5, totalDist * 0.28) * PROJECTILE_ARROW_LENGTH_SCALE;
  const shaftX1 = headBaseX;
  const shaftY1 = headBaseY;
  const shaftX2 = headBaseX - ux * shaftLength;
  const shaftY2 = headBaseY - uy * shaftLength;

  const impactPhase = Math.max(0, Math.min(1, (eased - 0.82) / 0.18));
  const impactOpacity = impactPhase > 0 ? (1 - impactPhase) * shotFlashOpacity.value : 0;
  const impactRadius = HEX_RADIUS * (0.35 + impactPhase * 1.0);

  return {
    projectileOpacity: 0.35 + shotFlashOpacity.value * 0.65,
    shaftX1,
    shaftY1,
    shaftX2,
    shaftY2,
    headPoints: `${tip.x},${tip.y} ${leftX},${leftY} ${rightX},${rightY}`,
    impactX: end.x,
    impactY: end.y,
    impactOpacity,
    impactRadius,
  };
});

const shotBallOutline = computed(() => {
  const flash = shotFlash.value;
  if (!flash) return null;

  const start = { x: flash.x1, y: flash.y1 };
  const control = { x: flash.controlX, y: flash.controlY };
  const end = { x: flash.x2, y: flash.y2 };
  const t = shotFlashProgress.value;
  const eased = 1 - ((1 - t) * (1 - t));
  const point = quadraticBezierPoint(start, control, end, eased);
  const edgeFade = Math.max(0, Math.min(1, t / 0.12, (1 - t) / 0.12));
  const opacity = edgeFade * (0.6 + 0.4 * shotFlashOpacity.value);

  return {
    x: point.x,
    y: point.y,
    radius: HEX_RADIUS * (0.69 + 0.06 * Math.sin(Math.PI * t)),
    opacity,
    dashOffset: (1 - t) * 22,
  };
});

function stopShotFlashClock() {
  if (shotFlashRaf.value !== null) {
    cancelAnimationFrame(shotFlashRaf.value);
    shotFlashRaf.value = null;
  }
}

function startShotFlashClock() {
  stopShotFlashClock();
  const tick = (ts) => {
    shotFlashNowMs.value = ts;
    if (shotFlash.value && !props.disableTransitions) {
      shotFlashRaf.value = requestAnimationFrame(tick);
    } else {
      shotFlashRaf.value = null;
    }
  };
  shotFlashRaf.value = requestAnimationFrame(tick);
}

function clearShotFlash() {
  if (shotFlashTimeout.value) {
    clearTimeout(shotFlashTimeout.value);
    shotFlashTimeout.value = null;
  }
  stopShotFlashClock();
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

  const startedAtMs = performance.now();
  shotFlashNowMs.value = startedAtMs;
  const arc = buildShotArcGeometry(start, end);

  shotFlash.value = {
    flashKey: ++shotFlashSerial,
    startedAtMs,
    shooterId,
    x1: start.x,
    y1: start.y,
    x2: end.x,
    y2: end.y,
    controlX: arc.controlX,
    controlY: arc.controlY,
    color: success ? '#22c55e' : '#ef4444',
    path: arc.path,
  };
  if (!props.disableTransitions) {
    startShotFlashClock();
  }

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

    const actionResults = state.last_action_results;
    if (!actionResults || typeof actionResults !== 'object') {
      clearPassFlash();
      return;
    }
    const passes = actionResults?.passes && typeof actionResults.passes === 'object'
      ? actionResults.passes
      : {};

    let passAnimation = null;
    for (const [passerId, passResult] of Object.entries(passes)) {
      const targetId = Number(passResult?.target);
      if (passResult && passResult.success && Number.isFinite(targetId)) {
        passAnimation = { passerId: Number(passerId), receiverId: targetId };
        break;
      }
    }

    // If pass was intercepted, animate the ball path to the stealing defender.
    if (!passAnimation) {
      const turnovers = Array.isArray(actionResults?.turnovers)
        ? actionResults.turnovers
        : [];
      const stealTurnover = turnovers.find((turnover) => {
        if (!turnover || typeof turnover !== 'object') return false;
        const reason = String(turnover.reason || '').toLowerCase();
        if (reason !== 'steal') return false;
        return Number.isFinite(Number(turnover.player_id)) && Number.isFinite(Number(turnover.stolen_by));
      });
      if (stealTurnover) {
        passAnimation = {
          passerId: Number(stealTurnover.player_id),
          receiverId: Number(stealTurnover.stolen_by),
        };
      }
    }

    // Extra fallback: some payloads may encode intercepted passes in the pass record itself.
    if (!passAnimation) {
      for (const [passerId, passResult] of Object.entries(passes)) {
        if (!passResult || typeof passResult !== 'object') continue;
        const turnoverFlag = Boolean(passResult.turnover) || String(passResult.reason || '').toLowerCase() === 'steal';
        const stolenBy = Number(passResult.stolen_by);
        if (turnoverFlag && Number.isFinite(stolenBy)) {
          passAnimation = { passerId: Number(passerId), receiverId: stolenBy };
          break;
        }
      }
    }

    if (!passAnimation) {
      clearPassFlash();
      return;
    }

    const passerPos = state.positions?.[passAnimation.passerId];
    const receiverPos = state.positions?.[passAnimation.receiverId];
    if (!passerPos || !receiverPos) {
      clearPassFlash();
      return;
    }

    const start = axialToCartesian(passerPos[0], passerPos[1]);
    const end = axialToCartesian(receiverPos[0], receiverPos[1]);
    triggerPassFlash(passAnimation.passerId, passAnimation.receiverId, start, end);
  },
  { immediate: true }
);

watch(
  () => props.disableTransitions,
  (disabled) => {
    if (disabled) {
      stopPassFlashClock();
      stopShotFlashClock();
      return;
    }
    if (passFlash.value) {
      startPassFlashClock();
    }
    if (shotFlash.value) {
      startShotFlashClock();
    }
  }
);

watch(
  currentGameState,
  (state) => {
    if (!state) {
      clearShotFlash();
      clearShotJump();
      lastShotAnimationKey.value = null;
      return;
    }

    const shots = state.last_action_results?.shots;
    if (!shots || Object.keys(shots).length === 0) {
      clearShotFlash();
      clearShotJump();
      lastShotAnimationKey.value = null;
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
      lastShotAnimationKey.value = null;
      return;
    }

    const start = axialToCartesian(shooterPos[0], shooterPos[1]);
    const end = axialToCartesian(basketPos[0], basketPos[1]);
    const success = !!shotData.result.success;
    const isDunk =
      (shotData.result && typeof shotData.result.is_dunk === 'boolean' && shotData.result.is_dunk) ||
      hexDistance(shooterPos, basketPos) === 0;

    const shotAnimationKey = [
      Number(state.shot_clock ?? -1),
      Number(shotData.shooterId),
      success ? 1 : 0,
      isDunk ? 1 : 0,
      Number(shotData.result?.distance ?? -1),
      Number(shooterPos[0]),
      Number(shooterPos[1]),
      Number(basketPos[0]),
      Number(basketPos[1]),
    ].join('|');
    if (shotAnimationKey === lastShotAnimationKey.value) {
      return;
    }
    lastShotAnimationKey.value = shotAnimationKey;

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
        'stroke-linecap', 'stroke-opacity',
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
        'stroke-linecap', 'stroke-opacity',
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
  closeDownloadMenu();
  clearClickTimeout();
  window.removeEventListener('mousedown', onGlobalMouseDown);
  window.removeEventListener('mousemove', onGlobalMouseMove);
  window.removeEventListener('mouseup', onGlobalMouseUp);
  clearPassFlash();
  clearShotFlash();
  clearShotJump();
});

</script>

<template>
  <div
    class="game-board-container"
    :class="{
      'no-move-transitions': disableTransitions,
      'minimal-chrome': minimalChrome,
    }"
  >
    <div class="board-toolbar">
      <div class="download-menu" ref="downloadMenuRef">
        <button
          class="download-button"
          @click="toggleDownloadMenu"
          :disabled="isDownloadRunning"
          title="Download board"
        >
          📥
        </button>
        <div v-if="showDownloadMenu" class="download-menu-popover">
          <button
            class="download-menu-item"
            @click="handleDownloadChoice('png')"
            :disabled="isDownloadRunning"
          >
            Download PNG
          </button>
          <button
            class="download-menu-item"
            @click="handleDownloadChoice('gif')"
            :disabled="isDownloadRunning"
          >
            Download GIF
          </button>
        </div>
      </div>
      <button
        v-if="!minimalChrome"
        class="toggle-btn board-toggle-btn"
        @click="toggleAllPolicies"
        :aria-pressed="allPoliciesVisible"
        title="Show or hide policy probabilities for all players"
      >
        <font-awesome-icon :icon="allPoliciesVisible ? ['fas','toggle-on'] : ['fas','toggle-off']" />
        <span class="toggle-label">Show Policies</span>
      </button>
      <div v-if="!minimalChrome" class="pressure-controls-row">
        <button
          class="toggle-btn board-toggle-btn"
          @click="showShotPressureRing = !showShotPressureRing"
          :aria-pressed="showShotPressureRing"
          title="Toggle live shot-pressure ring around the ball handler"
        >
          <font-awesome-icon :icon="showShotPressureRing ? ['fas','toggle-on'] : ['fas','toggle-off']" />
          <span class="toggle-label">Shot Pressure</span>
        </button>
        <button
          class="toggle-btn board-toggle-btn"
          @click="showDefenderPressureShake = !showDefenderPressureShake"
          :aria-pressed="showDefenderPressureShake"
          title="Toggle defender turnover-pressure shake animation"
        >
          <font-awesome-icon :icon="showDefenderPressureShake ? ['fas','toggle-on'] : ['fas','toggle-off']" />
          <span class="toggle-label">Defender Motion</span>
        </button>
      </div>
    </div>
    <svg class="board-svg" :viewBox="viewBox" preserveAspectRatio="xMidYMid meet" ref="svgRef">
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
        <mask
          v-if="courtBackdropRect"
          id="court-gap-mask"
          maskUnits="userSpaceOnUse"
          maskContentUnits="userSpaceOnUse"
        >
          <rect
            :x="courtBackdropRect.x"
            :y="courtBackdropRect.y"
            :width="courtBackdropRect.width"
            :height="courtBackdropRect.height"
            fill="white"
          />
          <polygon
            v-for="hex in courtHexPolygons"
            :key="`court-gap-mask-${hex.key}`"
            :points="hex.points"
            fill="black"
            stroke="black"
            stroke-width="2"
          />
        </mask>
      </defs>
      <g :transform="boardTransform">
        <rect
          v-if="courtBackdropRect"
          :x="courtBackdropRect.x"
          :y="courtBackdropRect.y"
          :width="courtBackdropRect.width"
          :height="courtBackdropRect.height"
          class="court-gap-fill"
          mask="url(#court-gap-mask)"
        />

        <!-- Draw qualified (blue) and unqualified (dark) court hexes -->
        <polygon
          v-for="hex in courtHexPolygons"
          :key="hex.key"
          :points="hex.points"
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

        <rect
          v-if="courtBackdropRect"
          :x="courtBackdropRect.x"
          :y="courtBackdropRect.y"
          :width="courtBackdropRect.width"
          :height="courtBackdropRect.height"
          class="court-gap-boundary"
        />

        <!-- 3PT line outline -->
        <path
          v-if="threePointArcPath"
          :d="threePointArcPath"
          class="three-point-arc"
        />

        <!-- Court reference labels -->
        <text
          v-if="!minimalChrome && currentGameState && courtLayout.length"
          :x="basketMarkerPosition.x"
          :y="basketMarkerPosition.y"
          dy=".35em"
          text-anchor="middle"
          class="court-marker basket-marker"
        >
          B
        </text>
        <text
          v-if="!minimalChrome && currentGameState && courtLayout.length"
          :x="halfcourtMarkerPosition.x"
          :y="halfcourtMarkerPosition.y"
          dy=".35em"
          text-anchor="middle"
          class="court-marker halfcourt-marker"
        >
          H
        </text>
        <text
          v-if="!minimalChrome && currentGameState && courtLayout.length"
          :x="rightSidelineMarkerPosition.x"
          :y="rightSidelineMarkerPosition.y"
          dy=".35em"
          text-anchor="middle"
          class="court-marker sideline-marker"
        >
          R
        </text>
        <text
          v-if="!minimalChrome && currentGameState && courtLayout.length"
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

        <!-- Aggregated Playbook overlay -->
        <g v-if="hasPlaybookOverlay" class="playbook-overlay">
          <polygon
            v-for="pt in playbookShotOverlayPoints"
            :key="`playbook-shot-${pt.key}`"
            :points="hexPointsFor(pt.x, pt.y, HEX_RADIUS)"
            :fill="volumeFillFor(pt.attempts, playbookMaxShotAttempts, 0.48)"
            :stroke="volumeStrokeFor(pt.attempts, playbookMaxShotAttempts, 0.72)"
            stroke-width="1.1"
            class="playbook-shot-heat-hex"
          />
          <text
            v-for="pt in playbookShotOverlayPoints"
            :key="`playbook-shot-label-${pt.key}`"
            :x="pt.x"
            :y="pt.y"
            text-anchor="middle"
            dominant-baseline="middle"
            class="playbook-shot-count-text"
          >
            {{ pt.attempts }}
          </text>
          <line
            v-for="segment in playbookPlayerPathSegments"
            :key="segment.key"
            :x1="segment.x1"
            :y1="segment.y1"
            :x2="segment.x2"
            :y2="segment.y2"
            :stroke="segment.color"
            :stroke-width="segment.strokeWidth"
            :opacity="segment.opacity"
            class="playbook-player-segment"
          />
          <line
            v-for="segment in playbookBallPathSegments"
            :key="segment.key"
            :x1="segment.x1"
            :y1="segment.y1"
            :x2="segment.x2"
            :y2="segment.y2"
            :stroke="ballColor"
            :stroke-width="segment.strokeWidth"
            :opacity="segment.opacity"
            class="playbook-ball-segment"
          />
          <g
            v-for="segment in playbookPassPathSegments"
            :key="segment.key"
            class="playbook-pass-segment"
            :style="{ opacity: segment.opacity }"
          >
            <polygon
              :points="segment.startPoints"
              :fill="segment.passerColor"
              class="playbook-pass-origin-head"
            />
            <line
              :x1="segment.receiveBarX1"
              :y1="segment.receiveBarY1"
              :x2="segment.receiveBarX2"
              :y2="segment.receiveBarY2"
              :stroke="segment.receiverColor"
              :stroke-width="segment.receiveBarStrokeWidth"
              class="playbook-pass-receive-bar"
            />
          </g>
          <g
            v-for="marker in playbookStartMarkers"
            :key="marker.key"
            class="playbook-start-marker"
          >
            <circle
              :cx="marker.x"
              :cy="marker.y"
              :r="HEX_RADIUS * 0.46"
              :stroke="marker.color"
              class="playbook-start-shell"
            />
            <circle
              v-if="marker.hasBall"
              :cx="marker.x"
              :cy="marker.y"
              :r="HEX_RADIUS * 0.62"
              class="playbook-start-ball-ring"
            />
            <text
              :x="marker.x"
              :y="marker.y"
              dy=".32em"
              text-anchor="middle"
              class="playbook-start-label"
            >
              {{ marker.label }}
            </text>
          </g>
        </g>

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
              :class="playerTeamClass(player)"
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
              {{ getPlayerJerseyNumber(player.id) }}
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
            :stroke="move.arrowTone === 'user' ? '#007bff' : '#dc3545'"
            stroke-width="3"
            :marker-end="move.arrowTone === 'user' ? 'url(#arrowhead-offense)' : 'url(#arrowhead-defense)'"
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
            { 'shoot-jump-dunk': shotJumpPlayerId === player.id && shotJumpIsDunk && draggedPlayerId !== player.id },
            {
              'defender-pressure-shake':
                showDefenderPressureShake
                && !player.isOffense
                && !player.hasBall
                && draggedPlayerId !== player.id
                && shotJumpPlayerId !== player.id
                && shotInFlightPlayerId !== player.id
                && getDefenderTurnoverPressure(player.id) > DEFENDER_PRESSURE_SHAKE_EPS,
            },
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
            } : {}),
            ...(
              showDefenderPressureShake
              && !player.isOffense
              && !player.hasBall
              && draggedPlayerId !== player.id
              && shotJumpPlayerId !== player.id
              && shotInFlightPlayerId !== player.id
                ? getDefenderPressureShakeStyle(player.id)
                : {}
            ),
          }"
        >
            <!-- If dragging this player, show it at dragged pos, otherwise at hex pos -->
            <circle
              v-if="ballHandlerShotPressureRing && ballHandlerShotPressureRing.playerId === player.id"
              :cx="draggedPlayerId === player.id ? draggedPlayerPos.x : player.x"
              :cy="draggedPlayerId === player.id ? draggedPlayerPos.y : player.y"
              :r="ballHandlerShotPressureRing.radius"
              :class="[
                'shot-pressure-ring',
                {
                  'shot-pressure-ring-pulse': ballHandlerShotPressureRing.pulseEnabled && !disableTransitions,
                },
              ]"
              :stroke="ballHandlerShotPressureRing.color"
              :stroke-opacity="ballHandlerShotPressureRing.opacity"
              :stroke-width="ballHandlerShotPressureRing.strokeWidth"
              :style="{
                pointerEvents: 'none',
                '--ring-pulse-duration': `${ballHandlerShotPressureRing.pulseDurationSec.toFixed(2)}s`,
                '--ring-pulse-scale': `${ballHandlerShotPressureRing.pulseScale.toFixed(3)}`,
              }"
            >
              <title>{{ formatShotPressureRingTitle(ballHandlerShotPressureRing) }}</title>
            </circle>
            <circle 
              :cx="draggedPlayerId === player.id ? draggedPlayerPos.x : player.x" 
              :cy="draggedPlayerId === player.id ? draggedPlayerPos.y : player.y" 
              :r="HEX_RADIUS * 0.8" 
              :class="[
                playerTeamClass(player),
                { 'dragging': draggedPlayerId === player.id },
                { 'pass-target-preview': passTargetPreview && passTargetPreview.receiverId === player.id },
                { 'playable-active-player': minimalChrome && player.id === activePlayerId },
              ]"
              @mousedown="onMouseDown($event, player)"
              @click="onPlayerClick($event, player)"
              @dblclick.stop="onPlayerDoubleClick($event, player)"
              :style="playerCircleStyle(player)"
            >
              <title v-if="formatPlayerTooltipTitle(player.id)">
                {{ formatPlayerTooltipTitle(player.id) }}
              </title>
            </circle>
            <circle
              v-if="player.isOffense && !player.owner"
              :cx="draggedPlayerId === player.id ? draggedPlayerPos.x : player.x"
              :cy="draggedPlayerId === player.id ? draggedPlayerPos.y : player.y"
              :r="HEX_RADIUS * 0.67"
              class="player-offense-core"
              style="pointer-events: none;"
            />
            <text 
              v-if="minimalChrome && getPlayerDisplayName(player.id)"
              :transform="playerLabelTransform(player)"
              x="0"
              y="0"
              dy="-0.85em"
              text-anchor="middle"
              class="player-name-text"
              style="pointer-events: none;"
            >{{ getPlayerDisplayName(player.id) }}</text>
            <text
              v-if="minimalChrome"
              :x="(draggedPlayerId === player.id ? draggedPlayerPos.x : player.x) - (HEX_RADIUS * 0.6)"
              :y="draggedPlayerId === player.id ? draggedPlayerPos.y : player.y"
              dy="0.3em"
              text-anchor="middle"
              class="player-index-text"
              style="pointer-events: none;"
            >
              {{ player.id }}
            </text>
            <text 
              :transform="playerLabelTransform(player)"
              x="0" 
              y="0" 
              :dy="minimalChrome && getPlayerDisplayName(player.id) ? '0.48em' : '0.3em'"
              text-anchor="middle" 
              :class="[
                'player-text',
                { 'active-player-text': player.id === activePlayerId },
              ]"
              style="pointer-events: none;" 
            >{{ getPlayerJerseyNumber(player.id) }}</text>
            <!-- EP (Expected Points) label above player ID for offensive players -->
            <text
              v-if="player.isOffense && currentGameState.ep_by_player && currentGameState.ep_by_player[player.id] !== undefined && draggedPlayerId !== player.id"
              :x="player.x"
              :y="player.y"
              :dy="expectedValueLabelDy()"
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
            <!-- Ball handler indicator / turnover pressure meter -->
            <g
              v-if="player.hasBall && !passFlash && !shotFlash && shotJumpPlayerId !== player.id && shotInFlightPlayerId !== player.id"
              class="ball-indicator-wrap"
              style="pointer-events: none;"
            >
              <template v-if="minimalChrome">
                <path
                  :d="getBallPressureSeamPath(player)"
                  class="ball-pressure-meter-seam"
                />
                <path
                  :d="getBallPressureMeterPath(player)"
                  class="ball-pressure-meter-track"
                />
                <path
                  :d="getBallPressureMeterPath(player)"
                  class="ball-pressure-meter-fill"
                  :style="getBallPressureMeterStyle()"
                />
                <text
                  :x="getPlayerRenderCenter(player).x + (HEX_RADIUS * 1.02)"
                  :y="getPlayerRenderCenter(player).y + (HEX_RADIUS * 0.02)"
                  text-anchor="start"
                  dominant-baseline="middle"
                  class="ball-pressure-meter-label"
                  :style="{ fill: ballHandlerTurnoverPressureMeter.color }"
                >
                  {{ formatBallPressureMeterLabel() }}
                </text>
                <title>{{ ballHandlerTurnoverPressureMeter.title }}</title>
              </template>
              <circle
                v-else
                :cx="draggedPlayerId === player.id ? draggedPlayerPos.x : player.x"
                :cy="draggedPlayerId === player.id ? draggedPlayerPos.y : player.y"
                :r="HEX_RADIUS * 0.9"
                class="ball-indicator"
              />
            </g>
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
        
        <!-- Draw Pass Rays (ball handler to teammates with pass success probabilities) - drawn after players for visibility -->
        <g v-if="showPlayers" v-for="ray in passRays" :key="`pass-ray-${ray.teammateId}`" class="pass-ray-group">
          <line
            :x1="ray.x1"
            :y1="ray.y1"
            :x2="ray.x2"
            :y2="ray.y2"
            class="pass-ray"
            :opacity="ray.passSuccessOpacity"
          />
          <text
            v-if="!minimalChrome"
            :x="ray.midX"
            :y="ray.midY"
            text-anchor="middle"
            dominant-baseline="middle"
            class="steal-prob-label"
            :opacity="ray.passSuccessOpacity"
          >
            {{ ray.passSuccessLabel }}
          </text>
        </g>

        <!-- Flash effect for completed passes -->
        <g v-if="passFlash && showPlayers" :key="`pass-flash-${passFlash.flashKey}`" class="pass-flash-group">
          <template v-if="normalizedPassAnimationStyle === 'projectile' && passProjectile">
            <line
              :x1="passFlash.x1"
              :y1="passFlash.y1"
              :x2="passFlash.x2"
              :y2="passFlash.y2"
              class="pass-projectile-lane"
              :opacity="passProjectile.laneOpacity"
            />
            <line
              :x1="passProjectile.shaftX1"
              :y1="passProjectile.shaftY1"
              :x2="passProjectile.shaftX2"
              :y2="passProjectile.shaftY2"
              class="pass-projectile-shaft"
              :opacity="passProjectile.projectileOpacity"
            />
            <polygon
              :points="passProjectile.headPoints"
              class="pass-projectile-head"
              :opacity="passProjectile.projectileOpacity"
            />
            <circle
              v-if="passProjectile.impactOpacity > 0.01"
              :cx="passProjectile.impactX"
              :cy="passProjectile.impactY"
              :r="passProjectile.impactRadius"
              class="pass-projectile-impact"
              :opacity="passProjectile.impactOpacity"
            />
          </template>
          <circle
            v-if="passBallOutline"
            :cx="passBallOutline.x"
            :cy="passBallOutline.y"
            :r="passBallOutline.radius"
            class="pass-ball-outline"
            :opacity="passBallOutline.opacity"
            :style="{ strokeDashoffset: `${passBallOutline.dashOffset}` }"
          />
          <line
            v-if="normalizedPassAnimationStyle !== 'projectile' || !passProjectile"
            :x1="passFlash.x1"
            :y1="passFlash.y1"
            :x2="passFlash.x2"
            :y2="passFlash.y2"
            :stroke="ballColor"
            class="pass-flash-line"
            :opacity="passFlashOpacity"
          />
          <text
            :x="passFlash.labelX"
            :y="passFlash.labelY"
            text-anchor="middle"
            dominant-baseline="middle"
            :fill="ballColor"
            class="pass-flash-text"
            :opacity="passFlashOpacity"
          >
            {{ getOutcomePlayerLabel(passFlash.passerId) }} -> {{ getOutcomePlayerLabel(passFlash.receiverId) }}
          </text>
        </g>

        <!-- Flash effect for shot attempts -->
        <g v-if="shotFlash && showPlayers" :key="`shot-flash-${shotFlash.flashKey}`" class="shot-flash-group">
          <path
            :d="shotFlash.path"
            :stroke="shotFlash.color"
            class="shot-flash-line"
            fill="none"
            :opacity="shotLaneOpacity"
            :style="{ filter: `drop-shadow(0 0 10px ${shotFlash.color})` }"
          />
          <line
            v-if="shotProjectile"
            :x1="shotProjectile.shaftX1"
            :y1="shotProjectile.shaftY1"
            :x2="shotProjectile.shaftX2"
            :y2="shotProjectile.shaftY2"
            class="shot-projectile-shaft"
            :stroke="shotFlash.color"
            :opacity="shotProjectile.projectileOpacity"
          />
          <polygon
            v-if="shotProjectile"
            :points="shotProjectile.headPoints"
            class="shot-projectile-head"
            :fill="shotFlash.color"
            :opacity="shotProjectile.projectileOpacity"
          />
          <circle
            v-if="shotBallOutline"
            :cx="shotBallOutline.x"
            :cy="shotBallOutline.y"
            :r="shotBallOutline.radius"
            class="shot-ball-outline"
            :opacity="shotBallOutline.opacity"
            :style="{ strokeDashoffset: `${shotBallOutline.dashOffset}` }"
          />
          <circle
            v-if="shotProjectile && shotProjectile.impactOpacity > 0.01"
            :cx="shotProjectile.impactX"
            :cy="shotProjectile.impactY"
            :r="shotProjectile.impactRadius"
            class="shot-projectile-impact"
            :stroke="shotFlash.color"
            :opacity="shotProjectile.impactOpacity"
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
            <text
              v-if="episodeOutcome.type === 'TURNOVER' && episodeOutcome.x !== undefined && episodeOutcome.y !== undefined"
              :x="episodeOutcome.x"
              :y="episodeOutcome.y"
              class="turnover-x"
            >X</text>
            
            <!-- Defensive Violation indicator -->
            <text v-if="episodeOutcome.type === 'DEFENSIVE_VIOLATION' && episodeOutcome.x" :x="episodeOutcome.x" :y="episodeOutcome.y" class="violation-marker defensive">!</text>

            <!-- Offensive Violation indicator -->
            <text v-if="episodeOutcome.type === 'OFFENSIVE_VIOLATION' && episodeOutcome.x" :x="episodeOutcome.x" :y="episodeOutcome.y" class="violation-marker offensive">!</text>
        </g>

        <!-- State-value overlay -->
        <g v-if="!minimalChrome && (offenseStateValue !== null || defenseStateValue !== null) && showValueAnnotations" class="state-value-overlay">
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
              <tspan class="player-outcome-text" x="50%" dy="-1.2em">{{ getOutcomePlayerLabel(episodeOutcome.playerId) }}</tspan>
              <tspan x="50%" dy="1.2em">{{ episodeOutcome.isDunk ? 'Made Dunk!' : (episodeOutcome.isThree ? 'Made 3!' : 'Made 2!') }}</tspan>
          </text>
          <text v-if="episodeOutcome.type === 'MISSED_SHOT'" x="50%" y="15%" class="outcome-text missed">
              <tspan class="player-outcome-text" x="50%" dy="-1.2em">{{ getOutcomePlayerLabel(episodeOutcome.playerId) }}</tspan>
              <tspan x="50%" dy="1.2em">{{ episodeOutcome.isDunk ? 'Missed Dunk!' : (episodeOutcome.isThree ? 'Missed 3!' : 'Missed 2!') }}</tspan>
          </text>
          <text v-if="episodeOutcome.type === 'TURNOVER'" x="50%" y="15%" class="outcome-text turnover long-outcome-text">
              <tspan v-if="hasOutcomePlayer(episodeOutcome.playerId)" class="player-outcome-text" x="50%" dy="-1.2em">
                {{ getOutcomePlayerLabel(episodeOutcome.playerId) }}
              </tspan>
              <tspan x="50%" :dy="hasOutcomePlayer(episodeOutcome.playerId) ? '1.2em' : '0'">{{ getTurnoverOutcomeText(episodeOutcome) }}</tspan>
          </text>
          <text v-if="episodeOutcome.type === 'SHOT_CLOCK_VIOLATION'" x="50%" y="15%" class="outcome-text turnover long-outcome-text">SHOT CLOCK!</text>
          <text v-if="episodeOutcome.type === 'DEFENSIVE_VIOLATION'" x="50%" y="15%" class="outcome-text violation long-outcome-text">
              <tspan class="player-outcome-text" x="50%" dy="-1.2em">{{ getOutcomePlayerLabel(episodeOutcome.playerId) }}</tspan>
              <tspan x="50%" dy="1.2em">Violation - Defense!</tspan>
          </text>
          <text v-if="episodeOutcome.type === 'OFFENSIVE_VIOLATION'" x="50%" y="15%" class="outcome-text violation-offense long-outcome-text">
              <tspan class="player-outcome-text" x="50%" dy="-1.2em">{{ getOutcomePlayerLabel(episodeOutcome.playerId) }}</tspan>
              <tspan x="50%" dy="1.2em">Violation - Offense!</tspan>
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

    </svg>
    <div class="shot-clock-wrapper" v-if="!hasShotCounts">
      <div v-if="!minimalChrome && currentGameState && laneStepIndicatorStacks.length" class="lane-step-clock-indicators">
        <div
          v-for="stack in laneStepIndicatorStacks"
          :key="`clock-${stack.key}`"
          class="lane-step-clock-stack"
          :title="`${stack.label} lane steps: ${stack.steps}/${stack.maxSteps}`"
        >
          <span class="lane-step-clock-label" :style="{ color: stack.color }">{{ stack.shortLabel }}</span>
          <span class="lane-step-clock-lights">
            <span
              v-for="light in stack.lights"
              :key="`clock-${light.key}`"
              class="lane-step-clock-light"
              :class="{ lit: light.lit, violation: light.violation }"
              :style="{ '--lane-color': light.color }"
            ></span>
          </span>
        </div>
      </div>
      <div class="shot-clock-overlay">
        {{ displayedShotClockValue }}
      </div>
      <div v-if="allowShotClockAdjustment" class="shot-clock-controls">
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

.game-board-container.minimal-chrome {
  border: none;
  box-shadow: none;
}

.shot-clock-overlay {
  position: relative;
  font-family: 'DSEG7 Classic', sans-serif;
  font-size: 4.1rem;
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
  align-items: flex-end;
  gap: 6px;
  z-index: 11;
}

.lane-step-clock-indicators {
  display: flex;
  align-items: flex-end;
  gap: 0.3rem;
  margin-right: 2px;
  pointer-events: none;
}

.lane-step-clock-stack {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.34rem;
}

.lane-step-clock-label {
  font-size: 1rem;
  font-weight: 800;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  line-height: 1;
  text-shadow: 0 1px 2px rgba(2, 6, 23, 0.75);
}

.lane-step-clock-lights {
  display: flex;
  flex-direction: column-reverse;
  align-items: center;
  gap: 1rem;
}

.lane-step-clock-light {
  width: 1rem;
  height: 1rem;
  border-radius: 999px;
  background: rgba(148, 163, 184, 0.2);
  border: 1px solid rgba(15, 23, 42, 0.75);
  transition: background 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
}

.lane-step-clock-light.lit {
  background: #ffffff;
  border-color: var(--lane-color);
  box-shadow: 0 0 6px rgba(255, 255, 255, 0.95), 0 0 12px var(--lane-color);
}

.lane-step-clock-light.violation {
  box-shadow: 0 0 8px rgba(255, 255, 255, 0.98), 0 0 16px var(--lane-color);
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
  flex-wrap: wrap;
  gap: 8px;
  align-items: center;
  max-width: min(92%, 560px);
  z-index: 12;
}

.pressure-controls-row {
  display: inline-flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 8px;
  flex-basis: 100%;
}

.download-menu {
  position: relative;
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

.download-button:disabled {
  opacity: 0.45;
  cursor: not-allowed;
}

.download-menu-popover {
  position: absolute;
  top: calc(100% + 6px);
  left: 0;
  display: flex;
  flex-direction: column;
  min-width: 132px;
  padding: 4px;
  border-radius: 8px;
  border: 1px solid rgba(148, 163, 184, 0.45);
  background: rgba(15, 23, 42, 0.96);
  box-shadow: 0 10px 24px rgba(2, 6, 23, 0.55);
  z-index: 40;
}

.download-menu-item {
  border: 1px solid transparent;
  border-radius: 6px;
  padding: 8px 10px;
  background: transparent;
  color: #e2e8f0;
  font-size: 0.83rem;
  text-align: left;
  cursor: pointer;
  transition: border-color 0.15s ease, color 0.15s ease, background 0.15s ease;
}

.download-menu-item:hover:not(:disabled) {
  border-color: rgba(56, 189, 248, 0.55);
  color: #f8fafc;
  background: rgba(30, 41, 59, 0.75);
}

.download-menu-item:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.board-toggle-btn {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 1rem;
  border-radius: 999px;
  border: 1px solid rgba(148, 163, 184, 0.4);
  background: transparent;
  color: var(--app-text);
  font-size: 1rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  cursor: pointer;
  transition: color 0.15s ease, border 0.15s ease;
}

.board-toggle-btn :deep(svg) {
  width: 1.15em;
  height: 1.15em;
  flex: 0 0 auto;
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

.board-toggle-btn:hover:not(:disabled) {
  color: var(--app-accent);
  border-color: var(--app-accent-strong);
}

.board-toggle-btn:disabled {
  opacity: 0.35;
  cursor: not-allowed;
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
.board-svg {
  display: block;
  width: 100%;
  height: auto;
}
.court-gap-fill {
  fill: rgba(58, 70, 108, 0.92);
  pointer-events: none;
}
.court-gap-boundary {
  fill: none;
  stroke: white;
  stroke-width: 0.08rem;
  stroke-linejoin: round;
  pointer-events: none;
  filter: drop-shadow(0px 0px 5px rgba(203, 240, 69, 0.35));
}
.court-hex {
  stroke: rgba(15, 23, 42, 0.6);
  stroke-width: 0.1rem;
}
.court-hex.qualified {
  fill: rgba(44, 91, 246, 0.35);
  stroke: #fef3c781;
  stroke-width: 0.05rem;
}
.court-hex.unqualified {
  fill: rgba(41, 49, 88, 0.89);
  stroke: rgba(15, 23, 42, 0.95);
}
.three-point-arc {
  fill: none;
  stroke: #f5f5e5;
  stroke-width: 0.22rem;
  stroke-linecap: round;
  /* stroke-dasharray: 4 8; */
  filter: drop-shadow(0px 0px 3px rgba(225, 244, 223, 0.6));
}
.offensive-lane {
  fill: rgba(218, 3, 68, 0.631);
  stroke: rgba(255, 140, 140, 0.5);
  stroke-width: 1;
}
.player-offense {
  stroke: white;
  stroke-width: 0.05rem;
  transition: cx 0.26s ease, cy 0.26s ease, r 0.12s ease, fill 0.2s ease;
  filter: drop-shadow(0px 0px 1px rgba(204, 218, 246, 0.892));
}
.player-user {
  fill: #007bff;
  stroke: white;
  stroke-width: 0.05rem;
  transition: cx 0.26s ease, cy 0.26s ease, r 0.12s ease;
}
.player-offense-core {
  fill: #007bff;
  transition: cx 0.26s ease, cy 0.26s ease, r 0.12s ease;
}
.player-defense {
  fill: #dc3545;
  stroke: white;
  stroke-width: 0.05rem;
  transition: cx 0.26s ease, cy 0.26s ease, r 0.12s ease;
  filter: drop-shadow(0px 0px 1px rgba(242, 165, 242, 0.88));
}
.player-ai {
  fill: #dc3545;
  stroke: white;
  stroke-width: 0.05rem;
  transition: cx 0.26s ease, cy 0.26s ease, r 0.12s ease;
}
.player-group {
  transform-origin: center;
  transform-box: fill-box; /* keep scale/translate centered on the marker */
}
.shot-pressure-ring {
  fill: none;
  stroke-linecap: round;
  filter: drop-shadow(0 0 4px rgba(15, 23, 42, 0.8));
  transition: cx 0.26s ease, cy 0.26s ease, stroke 0.18s ease, stroke-width 0.18s ease;
}
.shot-pressure-ring-pulse {
  transform-origin: center;
  transform-box: fill-box;
  animation: shot-pressure-ring-pulse var(--ring-pulse-duration, 1.2s) ease-in-out infinite;
}

@keyframes shot-pressure-ring-pulse {
  0%, 100% {
    opacity: 1;
    transform: scale(1);
  }
  50% {
    opacity: 0.55;
    transform: scale(var(--ring-pulse-scale, 1.05));
  }
}

.defender-pressure-shake {
  animation: defender-pressure-shake var(--defender-pressure-shake-duration, 0.32s) ease-in-out infinite;
  will-change: transform;
}

@keyframes defender-pressure-shake {
  0%, 100% {
    transform: translate(0, 0) rotate(0deg);
  }
  20% {
    transform: translate(calc(-1 * var(--defender-pressure-shake-amp, 1px)), 0)
      rotate(calc(-1 * var(--defender-pressure-shake-rot, 0.5deg)));
  }
  40% {
    transform: translate(0, var(--defender-pressure-shake-amp, 1px))
      rotate(var(--defender-pressure-shake-rot, 0.5deg));
  }
  60% {
    transform: translate(var(--defender-pressure-shake-amp, 1px), 0)
      rotate(calc(-1 * var(--defender-pressure-shake-rot, 0.5deg)));
  }
  80% {
    transform: translate(0, calc(-1 * var(--defender-pressure-shake-amp, 1px)))
      rotate(var(--defender-pressure-shake-rot, 0.5deg));
  }
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
.pass-target-preview {
  stroke: #f8e71c;
  stroke-width: 0.2rem;
  filter: drop-shadow(0 0 6px rgba(248, 231, 28, 0.6));
}
.playable-active-player {
  stroke: #f8e71c;
  stroke-width: 0.07rem;
  filter: drop-shadow(0 0 2px rgba(248, 231, 28, 0.7)) drop-shadow(0 0 10px rgba(248, 231, 28, 0.35));
}
.dragging {
  opacity: 0.8;
  stroke: white;
  stroke-dasharray: 4 2;
  transition: none !important;
}
.player-text {
  fill: white;
  font-weight: 400;
  font-size: 0.65rem;
  paint-order: stroke;
  stroke: black;
  stroke-width: 0.05rem;
  transition: transform 0.26s ease;
}
.player-name-text {
  fill: #dbeafe;
  font-weight: 500;
  font-size: 0.45rem;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  paint-order: stroke;
  stroke: rgba(2, 6, 23, 0.9);
  stroke-width: 0.04rem;
  transition: transform 0.26s ease;
}
.player-index-text {
  fill: #111;
  font-weight: 500;
  font-size: 0.48rem;
  paint-order: stroke;
  stroke: rgba(255, 255, 255, 0.85);
  stroke-width: 0.02rem;
  transition: x 0.26s ease, y 0.26s ease;
}
.active-player-text {
  fill: #f8e71c;
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
.lane-step-indicators {
  pointer-events: none;
}
.lane-step-label {
  font-size: 0.36rem;
  font-weight: 800;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  paint-order: stroke;
  stroke: rgba(15, 23, 42, 0.95);
  stroke-width: 0.05rem;
}
.lane-step-light {
  stroke-width: 0.05rem;
  transition: fill 0.18s ease, opacity 0.18s ease;
}
.lane-step-light.lit {
  opacity: 0.95;
  filter: drop-shadow(0 0 3px rgba(248, 250, 252, 0.3));
}
.lane-step-light.violation {
  filter: drop-shadow(0 0 5px rgba(248, 250, 252, 0.45));
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
.ball-indicator-wrap {
  pointer-events: none;
}
.ball-pressure-meter-track {
  fill: none;
  stroke: rgba(148, 163, 184, 0.45);
  stroke-width: 0.25rem;
  stroke-linecap: round;
}
.ball-pressure-meter-seam {
  fill: none;
  stroke: orangered;
  stroke-width: 0.25rem;
  stroke-linecap: round;
  stroke-dasharray: 3 6;
}
.ball-pressure-meter-fill {
  fill: none;
  stroke: #f59e0b;
  stroke-width: 0.25rem;
  stroke-linecap: round;
  filter: drop-shadow(0 0 5px rgba(239, 68, 68, 0.35));
  transition: stroke-dasharray 0.2s ease, stroke 0.2s ease;
}
.ball-pressure-meter-label {
  font-size: 0.34rem;
  font-weight: 700;
  letter-spacing: 0.01em;
  paint-order: stroke;
  stroke: rgba(2, 6, 23, 0.95);
  stroke-width: 0.04rem;
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
.no-move-transitions .player-user,
.no-move-transitions .player-offense-core,
.no-move-transitions .player-defense,
.no-move-transitions .player-ai,
.no-move-transitions .player-name-text,
.no-move-transitions .player-index-text,
.no-move-transitions .player-text,
.no-move-transitions .shot-prob-text,
.no-move-transitions .noop-prob-text,
.no-move-transitions .ball-indicator,
.no-move-transitions .shot-pressure-ring,
.no-move-transitions .action-indicator {
  transition: none !important;
}

.noop-prob-text {
  font-size: 0.45rem;
  font-weight: 500;
  fill: #f9fcf8;
  paint-order: stroke;
  stroke: rgba(2, 6, 23, 0.85);
  stroke-width: 0.04rem;
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

.playbook-overlay {
  pointer-events: none;
}

.playbook-shot-heat-hex {
  mix-blend-mode: screen;
}

.playbook-shot-count-text {
  fill: rgba(248, 250, 252, 0.96);
  font-size: 0.72rem;
  font-weight: 800;
  letter-spacing: 0.01em;
  stroke: rgba(15, 23, 42, 0.85);
  stroke-width: 2.25px;
  paint-order: stroke fill;
  pointer-events: none;
}

.playbook-player-segment {
  fill: none;
  stroke-linecap: round;
  mix-blend-mode: screen;
}

.playbook-ball-segment {
  fill: none;
  stroke-linecap: round;
  stroke-dasharray: 10 7;
  filter: drop-shadow(0 0 7px rgba(255, 165, 0, 0.28));
}

.playbook-pass-segment {
  pointer-events: none;
}

.playbook-pass-origin-head {
  filter: drop-shadow(0 0 2px rgba(15, 23, 42, 0.35));
}

.playbook-pass-receive-bar {
  stroke-linecap: round;
  filter: drop-shadow(0 0 2px rgba(15, 23, 42, 0.35));
}

.playbook-start-shell {
  fill: rgba(15, 23, 42, 0.82);
  stroke-width: 3;
}

.playbook-start-ball-ring {
  fill: none;
  stroke: #f59e0b;
  stroke-width: 3.5;
  stroke-dasharray: 7 5;
}

.playbook-start-label {
  fill: #f8fafc;
  font-size: 0.78rem;
  font-weight: 800;
  paint-order: stroke;
  stroke: rgba(15, 23, 42, 0.95);
  stroke-width: 2px;
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
    font-weight: bold;
    text-anchor: middle;
    dominant-baseline: central;
    transform: scale(1, -1); /* Counteract the group flip */
}
.violation-marker.defensive {
    fill: #f59e0b;
}
.violation-marker.offensive {
    fill: #fb7185;
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
.violation-offense { fill: #fb7185; }

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

.pass-flash-group {
  pointer-events: none;
}

.pass-flash-line {
  stroke-width: 7;
  stroke-dasharray: 10 7;
  stroke-opacity: 0.85;
  stroke-linecap: round;
}

.pass-projectile-lane {
  stroke: rgba(255, 200, 80, 0.65);
  stroke-width: 4.5;
  stroke-linecap: round;
  stroke-dasharray: 12 8;
}

.pass-projectile-shaft {
  stroke: #ffe6a7;
  stroke-width: 7.5;
  stroke-linecap: round;
}

.pass-projectile-head {
  fill: #ff9f1c;
}

.pass-projectile-impact {
  fill: none;
  stroke: #ffd27a;
  stroke-width: 3;
}

.pass-ball-outline {
  fill: none;
  stroke: #ffa500;
  stroke-width: 4.2;
  stroke-dasharray: 8 6;
  stroke-linecap: round;
}

.pass-flash-text {
  font-size: 18px;
  font-weight: 800;
  paint-order: stroke;
  stroke: #0a0f1e;
  stroke-width: 3px;
  letter-spacing: 0.5px;
}

@keyframes pass-flash-line {
  0% { opacity: 1; stroke-width: 10; }
  60% { opacity: 0.75; stroke-width: 6; }
  100% { opacity: 0; stroke-width: 2; }
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
  stroke-width: 4.5;
  stroke-dasharray: 12 8;
  stroke-linecap: round;
}

.shot-projectile-shaft {
  stroke-width: 7.5;
  stroke-linecap: round;
}

.shot-projectile-head {
  stroke: none;
}

.shot-ball-outline {
  fill: none;
  stroke: #ffa500;
  stroke-width: 4.2;
  stroke-dasharray: 8 6;
  stroke-linecap: round;
}

.shot-projectile-impact {
  fill: none;
  stroke-width: 3;
}

</style>
