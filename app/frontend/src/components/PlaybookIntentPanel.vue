<script setup>
import { computed } from 'vue';
import GameBoard from './GameBoard.vue';

const props = defineProps({
  baseState: {
    type: Object,
    required: true,
  },
  panel: {
    type: Object,
    required: true,
  },
  offenseIds: {
    type: Array,
    default: () => [],
  },
});

const boardHistory = computed(() => (props.baseState ? [props.baseState] : []));

const playerHeatmapRows = computed(() => {
  const ids = Array.isArray(props.offenseIds) && props.offenseIds.length
    ? props.offenseIds
    : Object.keys(props.panel?.player_heatmaps || {}).map((pid) => Number(pid)).filter(Number.isFinite).sort((a, b) => a - b);
  return ids.map((pid) => ({
    playerId: pid,
    heatmap: props.panel?.player_heatmaps?.[String(pid)] || {},
  }));
});

const sortedPassLinks = computed(() => {
  const entries = Object.entries(props.panel?.pass_links || {});
  entries.sort((a, b) => Number(b[1] || 0) - Number(a[1] || 0));
  return entries;
});
</script>

<template>
  <div class="playbook-intent-panel">
    <div class="playbook-intent-header">
      <h5>z={{ panel.intent_index }}</h5>
      <div class="playbook-intent-metrics">
        <span>{{ panel.num_rollouts }} rollouts</span>
        <span>{{ panel.avg_steps?.toFixed?.(2) ?? '0.00' }} avg steps</span>
        <span>{{ panel.avg_passes?.toFixed?.(2) ?? '0.00' }} avg passes</span>
        <span>{{ (100 * (panel.terminated_rate || 0)).toFixed(0) }}% terminated</span>
      </div>
    </div>

    <div class="playbook-board-grid">
      <div class="playbook-board-card">
        <div class="playbook-board-title">Ball Occupancy</div>
        <GameBoard
          :game-history="boardHistory"
          :shot-accumulator="panel.ball_heatmap || {}"
          shot-chart-label="Ball"
          :minimal-chrome="true"
          :allow-position-drag="false"
          :allow-shot-clock-adjustment="false"
          :disable-backend-value-fetches="true"
          :disable-transitions="true"
          :selected-actions="{}"
          :policy-probabilities="null"
          :active-player-id="null"
        />
      </div>

      <div
        v-for="row in playerHeatmapRows"
        :key="`playbook-player-${panel.intent_index}-${row.playerId}`"
        class="playbook-board-card"
      >
        <div class="playbook-board-title">Player {{ row.playerId }} Occupancy</div>
        <GameBoard
          :game-history="boardHistory"
          :shot-accumulator="row.heatmap"
          :shot-chart-label="`P${row.playerId}`"
          :minimal-chrome="true"
          :allow-position-drag="false"
          :allow-shot-clock-adjustment="false"
          :disable-backend-value-fetches="true"
          :disable-transitions="true"
          :selected-actions="{}"
          :policy-probabilities="null"
          :active-player-id="null"
        />
      </div>
    </div>

    <div class="playbook-pass-summary">
      <div class="playbook-board-title">Pass Links</div>
      <div v-if="!sortedPassLinks.length" class="no-data">
        No successful passes recorded.
      </div>
      <div v-else class="playbook-pass-links">
        <span
          v-for="[link, count] in sortedPassLinks"
          :key="`pass-${panel.intent_index}-${link}`"
          class="playbook-pass-chip"
        >
          {{ link }}: {{ count }}
        </span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.playbook-intent-panel {
  display: flex;
  flex-direction: column;
  gap: 0.9rem;
  padding: 1rem;
  border: 1px solid rgba(140, 160, 190, 0.28);
  border-radius: 16px;
  background: rgba(17, 24, 39, 0.58);
}

.playbook-intent-header {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 0.75rem;
  flex-wrap: wrap;
}

.playbook-intent-header h5 {
  margin: 0;
  font-size: 1rem;
}

.playbook-intent-metrics {
  display: flex;
  gap: 0.65rem;
  flex-wrap: wrap;
  font-size: 0.82rem;
  color: rgba(226, 232, 240, 0.82);
}

.playbook-board-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 0.85rem;
}

.playbook-board-card {
  border: 1px solid rgba(140, 160, 190, 0.22);
  border-radius: 14px;
  padding: 0.65rem;
  background: rgba(15, 23, 42, 0.55);
}

.playbook-board-title {
  font-size: 0.8rem;
  font-weight: 600;
  letter-spacing: 0.03em;
  color: rgba(148, 163, 184, 0.96);
  margin-bottom: 0.4rem;
  text-transform: uppercase;
}

.playbook-pass-summary {
  border-top: 1px solid rgba(140, 160, 190, 0.18);
  padding-top: 0.75rem;
}

.playbook-pass-links {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
}

.playbook-pass-chip {
  display: inline-flex;
  align-items: center;
  padding: 0.28rem 0.6rem;
  border-radius: 999px;
  background: rgba(30, 41, 59, 0.88);
  border: 1px solid rgba(140, 160, 190, 0.18);
  font-size: 0.8rem;
  color: rgba(226, 232, 240, 0.92);
}
</style>
