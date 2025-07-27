<script setup>
import { computed } from 'vue';

const props = defineProps({
  gameState: {
    type: Object,
    required: true,
  },
  activePlayerId: {
    type: Number,
    default: null,
  },
});

// ------------------------------------------------------------
//  HEXAGON GEOMETRY — POINTY-TOP, ODD-R OFFSET  (matches Python)
// ------------------------------------------------------------

const HEX_RADIUS = 24;  // pixel radius of one hexagon corner-to-center

// Axial (q,r) → pixel cartesian (x,y) for pointy-topped hexes.
// Formula identical to the one in basketworld_env_v2.py:_render_visual.
function axialToCartesian(q, r) {
  const x = HEX_RADIUS * (Math.sqrt(3) * q + Math.sqrt(3) / 2 * r);
  const y = HEX_RADIUS * (1.5 * r);
  return { x, y };
}
// Helper function from Python environment to get axial coordinates for "odd-r"
function offsetToAxial(col, row) {
  const q = col - ((row - (row & 1)) >> 1);
  const r = row;
  return { q, r };
}

const courtLayout = computed(() => {
    const hexes = [];
    if (!props.gameState.court_width || !props.gameState.court_height) {
        return [];
    }
    for (let r_off = 0; r_off < props.gameState.court_height; r_off++) {
        for (let c_off = 0; c_off < props.gameState.court_width; c_off++) {
            const { q, r } = offsetToAxial(c_off, r_off);
            const { x, y } = axialToCartesian(q, r);
            hexes.push({ q, r, x, y, key: `${q},${r}` });
        }
    }
    return hexes;
});

const renderedPlayers = computed(() => {
  return props.gameState.positions.map((pos, index) => {
    const [q, r] = pos;
    const { x, y } = axialToCartesian(q, r);
    const isOffense = props.gameState.offense_ids.includes(index);
    const hasBall = props.gameState.ball_holder === index;
    return { id: index, x, y, isOffense, hasBall };
  });
});

const basketPosition = computed(() => {
    if (!props.gameState.basket_position) return { x: 0, y: 0 };
    const [q, r] = props.gameState.basket_position;
    // Manually adjusting the basket 'up' by one row to match the original gym render.
    return axialToCartesian(q, r - 1);
});

const viewBox = computed(() => {
    if (courtLayout.value.length === 0) return "-100 -100 200 200";
    
    const allX = courtLayout.value.map(h => h.x);
    const allY = courtLayout.value.map(h => h.y);
    
    const minX = Math.min(...allX) - HEX_RADIUS;
    const maxX = Math.max(...allX) + HEX_RADIUS;
    const minY = Math.min(...allY) - HEX_RADIUS;
    const maxY = Math.max(...allY) + HEX_RADIUS;

    const width = maxX - minX;
    const height = maxY - minY;

    return `${minX} ${minY} ${width} ${height}`;
});

</script>

<template>
  <div class="game-board-container">
    <svg :viewBox="viewBox" preserveAspectRatio="xMidYMid meet">
      <g>
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

        <!-- Draw the basket -->
        <circle :cx="basketPosition.x" :cy="basketPosition.y" :r="HEX_RADIUS * 0.8" class="basket-rim" />

        <!-- Draw the players -->
        <g v-for="player in renderedPlayers" :key="player.id">
          <circle 
            :cx="player.x" 
            :cy="player.y" 
            :r="HEX_RADIUS * 0.6" 
            :class="[
              player.isOffense ? 'player-offense' : 'player-defense',
              { 'active-player-hex': player.id === activePlayerId }
            ]"
          />
          <text :x="player.x" :y="player.y" dy=".3em" text-anchor="middle" class="player-text">{{ player.id }}</text>
          <!-- Ball handler indicator -->
          <circle v-if="player.hasBall" :cx="player.x" :cy="player.y" :r="HEX_RADIUS * 0.8" class="ball-indicator" />
        </g>
      </g>
    </svg>
    <div class="shot-clock-overlay">
      {{ gameState.shot_clock }}
    </div>
  </div>
</template>

<style scoped>
.game-board-container {
  position: relative; /* Needed for overlay positioning */
  flex: 1; /* Allow this component to grow and fill available space */
  min-width: 400px; /* Ensure it doesn't get too small */
  max-width: 650px;
  margin: 0; /* Remove auto margin which conflicts with flexbox */
  border-radius: 8px;
  overflow: hidden;
}

.shot-clock-overlay {
  position: absolute;
  bottom: 10px;
  right: 15px;
  font-size: 48px;
  font-weight: bold;
  color: black;
  opacity: 0.3;
  pointer-events: none; /* Make it non-interactive */
}

/* Removed rotation; court now renders in original orientation */
svg {
  display: block;
  background-color: #f0f0f0;
}
.court-hex {
  fill: #e0e0e0;
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
</style> 