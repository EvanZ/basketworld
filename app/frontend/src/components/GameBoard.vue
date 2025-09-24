<script setup>
import { computed, ref, watch, onMounted } from 'vue';
import { getShotProbability } from '@/services/api';

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
});

const emit = defineEmits(['update:activePlayerId']);

// ------------------------------------------------------------
//  HEXAGON GEOMETRY — POINTY-TOP, ODD-R OFFSET  (matches Python)
// ------------------------------------------------------------

const HEX_RADIUS = 24;  // pixel radius of one hexagon corner-to-center

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

function onPlayerClick(player) {
  // Only allow selecting offensive players from the board
  if (!player || !player.isOffense) return;
  emit('update:activePlayerId', player.id);
}

const currentGameState = computed(() => {
  return props.gameHistory.length > 0 ? props.gameHistory[props.gameHistory.length - 1] : null;
});

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

const basketPosition = computed(() => {
    if (!currentGameState.value) return { x: 0, y: 0 };
    const [q, r] = currentGameState.value.basket_position;
    // The basket axial coordinates already match the environment; no offset needed.
    return axialToCartesian(q, r);
});

const threePointSegments = computed(() => {
  const segs = [];
  const gs = currentGameState.value;
  if (!gs) return segs;
  const dist3 = gs.three_point_distance ?? 4;
  for (let r_off = 0; r_off < gs.court_height; r_off++) {
    for (let c_off = 0; c_off < gs.court_width; c_off++) {
      const { q, r } = offsetToAxial(c_off, r_off);
      const dq = Math.abs(q - gs.basket_position[0]);
      const dr = Math.abs(r - gs.basket_position[1]);
      const ds = Math.abs((q + r) - (gs.basket_position[0] + gs.basket_position[1]));
      const hexDist = (dq + dr + ds) / 2;
      if (hexDist === dist3) {
        const { x, y } = axialToCartesian(q, r);
        segs.push({ x, y, key: `tp-${q},${r}` });
      }
    }
  }
  return segs;
});

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

// This is a direct mapping from our ActionType enum for moves
const moveActionIndices = {
    MOVE_E: 1, MOVE_NE: 2, MOVE_NW: 3, MOVE_W: 4, MOVE_SW: 5, MOVE_SE: 6
};
const hexDirections = [
    {q: +1, r:  0}, {q: +1, r: -1}, {q:  0, r: -1}, 
    {q: -1, r:  0}, {q: -1, r: +1}, {q:  0, r: +1}
];

const policySuggestions = computed(() => {
    // Corrected the guard clause to explicitly check for null, fixing the "Player 0" bug.
    if (props.activePlayerId === null || !props.policyProbabilities || !props.policyProbabilities[props.activePlayerId]) {
        console.log('[GameBoard] No suggestions to render.');
        return [];
    }
    console.log(`[GameBoard] Calculating suggestions for Player ${props.activePlayerId}`);
    const currentPlayerPos = currentGameState.value.positions[props.activePlayerId];
    const probs = props.policyProbabilities[props.activePlayerId];

    const suggestions = [];
    for (let i = 0; i < hexDirections.length; i++) {
        const dir = hexDirections[i];
        const moveActionIndex = i + 1; // MOVE_E .. MOVE_SE
        const passActionIndex = 8 + i; // PASS_E .. PASS_SE

        const targetPos = { q: currentPlayerPos[0] + dir.q, r: currentPlayerPos[1] + dir.r };
        const cartesianPos = axialToCartesian(targetPos.q, targetPos.r);

        suggestions.push({
            x: cartesianPos.x,
            y: cartesianPos.y,
            moveProb: probs[moveActionIndex] ?? 0,
            passProb: probs[passActionIndex] ?? 0,
            key: `sugg-${i}`
        });
    }
    console.log('[GameBoard] Generated suggestions:', suggestions);
    return suggestions;
});

const ballHandlerShotProb = computed(() => {
    if (!currentGameState.value || currentGameState.value.ball_holder === null || !props.policyProbabilities) {
        return null;
    }
    const ballHolderId = currentGameState.value.ball_holder;
    const probs = props.policyProbabilities[ballHolderId];
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
        const [q, r] = pos;
        const bq = currentGameState.value.basket_position[0];
        const br = currentGameState.value.basket_position[1];
        const dist = (Math.abs(q - bq) + Math.abs((q + r) - (bq + br)) + Math.abs(r - br)) / 2;
        const isThree = dist >= (currentGameState.value.three_point_distance ?? 4);
        const isDunk = dist === 0;
        return {
            type: shotResult.success ? 'MADE_SHOT' : 'MISSED_SHOT',
            isThree,
            isDunk,
            playerId: pid,
        };
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
  // Start from the second state, as moves happen between states.
  for (let step = 1; step < props.gameHistory.length; step++) {
    const previousGameState = props.gameHistory[step - 1];
    const currentGameState = props.gameHistory[step];
    // Opacity should match the destination ghost cell's opacity.
    const opacity = 0.1 + (0.9 * (step - 1) / (props.gameHistory.length - 1));

    for (let playerId = 0; playerId < currentGameState.positions.length; playerId++) {
      const prevPos = previousGameState.positions[playerId];
      const currentPos = currentGameState.positions[playerId];

      // Check if the position has changed
      if (prevPos[0] !== currentPos[0] || prevPos[1] !== currentPos[1]) {
        const { x: startX, y: startY } = axialToCartesian(prevPos[0], prevPos[1]);
        const { x: endX, y: endY } = axialToCartesian(currentPos[0], currentPos[1]);
        
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

</script>

<template>
  <div class="game-board-container">
    <svg :viewBox="viewBox" preserveAspectRatio="xMidYMid meet">
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
      </defs>
      <!-- Flip the whole court vertically with translation to keep it in view -->
      <g :transform="boardTransform">
        <!-- Draw the court hexes -->
        <polygon
          v-for="hex in courtLayout"
          :key="hex.key"
          :points="[...Array(6)].map((_, i) => {
            const angle_deg = 60 * i + 30; // 30° offset for pointy-topped hexes
            const angle_rad = Math.PI / 180 * angle_deg;
            const xPoint = hex.x + HEX_RADIUS * Math.cos(angle_rad);
            const yPoint = hex.y + HEX_RADIUS * Math.sin(angle_rad);
            return `${xPoint},${yPoint}`;
          }).join(' ')"
          class="court-hex"
        />

        <!-- 3PT line: red outlines on hexes at exactly the 3PT distance -->
        <polygon
          v-for="hex in threePointSegments"
          :key="hex.key"
          :points="[...Array(6)].map((_, i) => {
            const ang = 60 * i + 30;
            const rad = Math.PI / 180 * ang;
            const xPoint = hex.x + HEX_RADIUS * Math.cos(rad);
            const yPoint = hex.y + HEX_RADIUS * Math.sin(rad);
            return `${xPoint},${yPoint}`;
          }).join(' ')"
          fill="none"
          stroke="red"
          stroke-width="2"
        />

        <!-- Draw the basket -->
        <circle :cx="basketPosition.x" :cy="basketPosition.y" :r="HEX_RADIUS * 0.8" class="basket-rim" />

        <!-- Draw Ghost Trails -->
        <g 
          v-for="(gameState, step) in gameHistory" 
          :key="`step-${step}`" 
          :style="{ opacity: 0.1 + (0.9 * step / (gameHistory.length - 1)) }"
        >
          <g v-for="player in getRenderablePlayers(gameState)" :key="player.id">
            <circle 
              v-if="step < gameHistory.length - 1"
              :cx="player.x" 
              :cy="player.y" 
              :r="HEX_RADIUS * 0.6" 
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
        <g v-for="move in playerTransitions" :key="move.key" :style="{ opacity: move.opacity }">
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

        <!-- Draw the current players on top -->
        <g v-if="currentGameState">
          <g v-for="player in getRenderablePlayers(currentGameState)" :key="player.id">
            <circle 
              :cx="player.x" 
              :cy="player.y" 
              :r="HEX_RADIUS * 0.6" 
              :class="[
                player.isOffense ? 'player-offense' : 'player-defense',
                { 'active-player-hex': player.id === activePlayerId }
              ]"
              @click="onPlayerClick(player)"
            />
            <text :x="player.x" :y="player.y" dy="0.3em" text-anchor="middle" class="player-text" @click="onPlayerClick(player)">{{ player.id }}</text>
            <!-- NOOP probability label (index 0) for the player -->
            <text
              v-if="player.isOffense && policyProbabilities && policyProbabilities[player.id] && policyProbabilities[player.id][0] !== undefined"
              :x="player.x"
              :y="player.y"
              dy="1.2em"
              text-anchor="middle"
              class="noop-prob-text"
            >
              {{ Number(policyProbabilities[player.id][0]).toFixed(2) }}
            </text>
            <!-- Display policy attempt probability for ball handler -->
            <text 
              v-if="player.hasBall && ballHandlerShotProb !== null"
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
              v-if="player.hasBall && ballHandlerMakeProb !== null"
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
            <circle v-if="player.hasBall" :cx="player.x" :cy="player.y" :r="HEX_RADIUS * 0.8" class="ball-indicator" />
          </g>
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
            <tspan :x="sugg.x" dy="-0.4em">{{ Number(sugg.moveProb).toFixed(3) }}</tspan>
            <tspan :x="sugg.x" dy="1.4em">{{ Number(sugg.passProb).toFixed(3) }}</tspan>
          </text>
        </g>

        <!-- Draw Episode Outcome Indicators -->
        <g v-if="episodeOutcome" class="outcome-overlay">
            <!-- Basket Fill for Shots -->
            <circle v-if="episodeOutcome.type === 'MADE_SHOT'" :cx="basketPosition.x" :cy="basketPosition.y" :r="HEX_RADIUS" class="basket-fill-made" />
            <circle v-if="episodeOutcome.type === 'MISSED_SHOT'" :cx="basketPosition.x" :cy="basketPosition.y" :r="HEX_RADIUS" class="basket-fill-missed" />

            <!-- Turnover 'X' -->
            <text v-if="episodeOutcome.type === 'TURNOVER'" :x="episodeOutcome.x" :y="episodeOutcome.y" class="turnover-x">X</text>
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
      </g>

      <!-- MLflow Run ID label (top-left) -->
      <text
        v-if="currentGameState && currentGameState.run_id"
        x="0%"
        y="-15%"
        text-anchor="start"
        dominant-baseline="hanging"
        class="run-id-label"
      >
        {{ currentGameState.run_id }}
      </text>
    </svg>
    <div class="shot-clock-overlay">
      {{ currentGameState ? currentGameState.shot_clock : '' }}
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
  margin-bottom: 60px; /* Add space below the board for the clock */
  /* Parquet-style checkerboard background */
  background-color: #d2b48c; /* Base light wood color */
  background-image: 
    linear-gradient(45deg, #c19a6b 25%, transparent 25%, transparent 75%, #c19a6b 75%), 
    linear-gradient(45deg, #c19a6b 25%, transparent 25%, transparent 75%, #c19a6b 75%);
  background-size: 60px 60px;
  background-position: 0 0, 30px 30px;
}

.shot-clock-overlay {
  position: absolute;
  bottom: -55px; /* Position it below the container */
  left: 50%;
  transform: translateX(-50%); /* Center it horizontally */
  font-family: 'DSEG7 Classic', sans-serif;
  font-size: 48px;
  color: #ff4d4d; /* Bright red for the LED color */
  background-color: #1a1a1a; /* Dark background for contrast */
  padding: 2px 8px;
  border-radius: 5px;
  border: 1px solid #333;
  text-shadow: 0 0 5px #ff4d4d, 0 0 10px #ff4d4d; /* Glowing effect */
  pointer-events: none; /* Make it non-interactive */
}

/* Removed rotation; court now renders in original orientation */
svg {
  display: block;
  width: 100%;
  height: auto;
}
.court-hex {
  fill: rgba(255, 255, 255, 0.65); /* More transparent fill */
  stroke: #ffffff;
  stroke-width: 1;
}
.player-offense {
  fill: #007bff;
  stroke: #0056b3;
  stroke-width: 1;
}
.player-defense {
  fill: #dc3545;
  stroke: #b22222;
  stroke-width: 1;
}
.active-player-hex {
  stroke: #ffeb3b; /* Bright yellow */
  stroke-width: 4; /* Increased width for better visibility */
}
.player-text {
  fill: white;
  font-weight: bold;
  font-size: 12px;
  paint-order: stroke;
  stroke: black;
  stroke-width: 0.5;
}
.ghost-text {
  font-size: 10px;
  opacity: 0.7;
  stroke-width: 0.2;
}
.basket-rim {
  fill: none;
  stroke: #ff8c00;
  stroke-width: 3;
}
.ball-indicator {
  fill: none;
  stroke: orange;
  stroke-width: 3;
  stroke-dasharray: 6 3;
}
.ghost {
  stroke: none;
}
.policy-suggestion-text {
  font-size: 10px;
  font-weight: bold;
  fill: #333;
  paint-order: stroke;
  stroke: white;
  stroke-width: 0.5;
  pointer-events: none;
}
.shot-prob-text {
  font-size: 10px;
  font-weight: bold;
  fill: #000;
  paint-order: stroke;
  stroke: #fff;
  stroke-width: 0.5px;
}

.noop-prob-text {
  font-size: 10px;
  font-weight: 700;
  fill: #111;
  paint-order: stroke;
  stroke: #fff;
  stroke-width: 0.6px;
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
</style>

<style scoped>
.run-id-label {
  font-size: 14px;
  font-weight: bold;
  fill: white;
  paint-order: stroke;
  stroke: black;
  stroke-width: 1.2px;
  pointer-events: none;
}
</style> 