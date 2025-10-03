<script setup>
import { ref, computed, watch, onMounted } from 'vue';
import { getPhiParams, setPhiParams, getPhiLog } from '../services/api';

const props = defineProps({
  gameState: Object
});

const loading = ref(false);
const error = ref(null);
const params = ref({ 
  enable_phi_shaping: false, 
  phi_beta: 0.0, 
  reward_shaping_gamma: 1.0, 
  phi_use_ball_handler_only: false, 
  phi_blend_weight: 0.0,
  phi_aggregation_mode: 'team_best'
});
const rawLogData = ref([]); // Store raw data from backend

async function loadParams() {
  try {
    const p = await getPhiParams();
    params.value = { ...params.value, ...p };
  } catch (e) {
    // ignore param fetch errors in isolation
  }
}

async function refreshLog() {
  loading.value = true;
  error.value = null;
  try {
    const { phi_log } = await getPhiLog();
    // Keep in ascending order by step (most recent at bottom)
    rawLogData.value = Array.isArray(phi_log) ? phi_log.slice(-200) : [];
  } catch (e) {
    error.value = String(e?.message || e);
  } finally {
    loading.value = false;
  }
}

// Recalculate Φ based on current parameters
function calculatePhi(row, mode, blendWeight, useBallHandlerOnly) {
  const ballEP = row.ball_handler_ep ?? 0;
  const ballHandlerIdx = row.ball_handler ?? -1;
  const offenseIds = row.offense_ids || [];
  const epByPlayer = row.ep_by_player || [];
  
  // Debug logging (remove after confirming it works)
  if (import.meta.env.DEV && row.step === 1) {
    console.log('[PhiShaping] Debug row data:', {
      step: row.step,
      ballHandlerIdx,
      offenseIds,
      epByPlayer,
      ballEP,
      mode,
      blendWeight
    });
  }
  
  if (useBallHandlerOnly) {
    return ballEP;
  }
  
  // Get EPs for offensive team
  const offenseEPs = offenseIds.map(id => epByPlayer[id] ?? 0).filter(ep => ep >= 0);
  
  if (offenseEPs.length === 0 || offenseEPs.every(ep => ep === 0)) {
    // Fallback if no ep_by_player data - use stored values
    console.warn('[PhiShaping] No ep_by_player data, using fallback');
    return mode === 'team_avg' ? (row.team_best_ep + ballEP) / 2 : 
           (1 - blendWeight) * row.team_best_ep + blendWeight * ballEP;
  }
  
  if (mode === 'team_avg') {
    // Average of all offensive players
    return offenseEPs.reduce((sum, ep) => sum + ep, 0) / offenseEPs.length;
  }
  
  // For other modes, separate ball handler from teammates
  const teammateEPs = offenseIds
    .filter(id => id !== ballHandlerIdx)
    .map(id => epByPlayer[id] ?? 0)
    .filter(ep => ep >= 0);
  
  if (teammateEPs.length === 0) {
    // No teammates (1v1 or edge case)
    return ballEP;
  }
  
  let aggregateEP;
  if (mode === 'teammates_best') {
    // Max of teammates excluding ball handler
    aggregateEP = Math.max(...teammateEPs);
  } else if (mode === 'teammates_avg') {
    // Average of teammates excluding ball handler
    aggregateEP = teammateEPs.reduce((sum, ep) => sum + ep, 0) / teammateEPs.length;
  } else {
    // 'team_best': Max including ball handler
    aggregateEP = Math.max(...offenseEPs);
  }
  
  // Blend aggregate with ball handler EP
  const w = Math.max(0, Math.min(1, blendWeight));
  return (1 - w) * aggregateEP + w * ballEP;
}

// Computed property that recalculates all rows with current parameters
const displayRows = computed(() => {
  const beta = Number(params.value.phi_beta) || 0;
  const gamma = Number(params.value.reward_shaping_gamma) || 1.0;
  const mode = params.value.phi_aggregation_mode || 'team_best';
  const blendWeight = Number(params.value.phi_blend_weight) || 0;
  const useBallHandlerOnly = params.value.phi_use_ball_handler_only || false;
  
  return rawLogData.value.map((row, idx) => {
    // Recalculate Φ values with current parameters
    // For terminal states, phi_next must be 0 to maintain policy invariance
    const phi_next = row.is_terminal ? 0.0 : calculatePhi(row, mode, blendWeight, useBallHandlerOnly);
    
    // For phi_prev:
    // - Step 0 (initial state): phi_prev = 0 (no previous state)
    // - Step 1 (first transition): phi_prev = 0 (Φ(s₀) = 0 by definition for proper PBRS)
    // - Step 2+: phi_prev = previous state's phi_next
    let phi_prev = 0;
    if (row.step > 1 && idx > 0) {
      const prevRow = rawLogData.value[idx - 1];
      phi_prev = prevRow.is_terminal ? 0.0 : calculatePhi(prevRow, mode, blendWeight, useBallHandlerOnly);
    }
    
    // Recalculate r_shape with current beta and gamma
    // Step 0 (initial state) has no transition yet, so r_shape = 0
    const r_shape = row.step === 0 ? 0 : beta * (gamma * phi_next - phi_prev);
    
    // Calculate best EP player from ep_by_player array
    let best_ep_player = -1;
    if (row.ep_by_player && row.offense_ids && row.offense_ids.length > 0) {
      const offenseEPs = row.offense_ids.map(pid => ({
        pid: pid,
        ep: row.ep_by_player[pid] || 0
      }));
      const bestPlayer = offenseEPs.reduce((max, curr) => curr.ep > max.ep ? curr : max);
      best_ep_player = bestPlayer.pid;
    }
    
    return {
      step: row.step,
      shot_clock: row.shot_clock ?? -1,
      phi_beta: beta,
      phi_prev: phi_prev,
      phi_next: phi_next,
      team_best_ep: row.team_best_ep ?? -1,
      ball_handler_ep: row.ball_handler_ep ?? -1,
      best_ep_player: best_ep_player,
      phi_r_shape: r_shape
    };
  });
});

// Auto-update environment whenever parameters change (debounced)
let updateTimeout = null;
async function updateEnvironment() {
  if (updateTimeout) clearTimeout(updateTimeout);
  updateTimeout = setTimeout(async () => {
    try {
      // Always enable phi shaping when using this tab
      const paramsToApply = { ...params.value, enable_phi_shaping: true };
      await setPhiParams(paramsToApply);
    } catch (e) {
      console.error('Failed to update environment params:', e);
    }
  }, 300); // 300ms debounce
}

// Watch for parameter changes and auto-update
watch(() => params.value, () => {
  updateEnvironment();
}, { deep: true });

// Auto-refresh when game state changes (after each step)
watch(() => props.gameState, () => {
  refreshLog();
}, { deep: true });

onMounted(() => {
  loadParams();
  refreshLog();
});

// Expose refresh for manual calls if needed
defineExpose({ refresh: refreshLog });
</script>

<template>
  <div class="phi-shaping">
    <h3>Phi Shaping</h3>
    <div v-if="loading">Loading...</div>
    <div v-if="error" class="error">{{ error }}</div>

    <div class="controls">
      <div class="info-message">
        Phi shaping is automatically enabled when using this tab.
      </div>
      <label>
        Beta
        <input type="number" step="0.01" v-model.number="params.phi_beta" />
      </label>
      <label>
        Gamma
        <input type="number" step="0.001" v-model.number="params.reward_shaping_gamma" />
      </label>
      <label>
        <input type="checkbox" v-model="params.phi_use_ball_handler_only" /> Ball-handler only Φ
      </label>
      <div class="aggregation-mode">
        <label>Aggregation Mode</label>
        <select v-model="params.phi_aggregation_mode">
          <option value="team_best">Team Best (includes ball handler)</option>
          <option value="teammates_best">Teammates Best (excludes ball handler)</option>
          <option value="teammates_avg">Teammates Avg (excludes ball handler)</option>
          <option value="team_avg">Team Avg (no blend)</option>
        </select>
      </div>
      <div class="blend" v-if="params.phi_aggregation_mode !== 'team_avg'">
        <label>Blend w (Aggregate vs Ball Φ)</label>
        <input type="range" min="0" max="1" step="0.05" v-model.number="params.phi_blend_weight" />
        <span>{{ params.phi_blend_weight.toFixed(2) }}</span>
      </div>
    </div>
    <div class="param-info">
      <small>💡 Adjusting parameters automatically updates both the table and environment. Changes take effect on the next step.</small>
    </div>

    <div class="log">
      <table>
        <thead>
          <tr>
            <th>Step</th>
            <th>Clock</th>
            <th>β</th>
            <th>Φprev</th>
            <th>Φnext</th>
            <th>TeamBestEP</th>
            <th>BallEP</th>
            <th>BestP</th>
            <th>r_shape/team</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="row in displayRows" :key="row.step">
            <td>{{ row.step }}</td>
            <td>{{ row.shot_clock >= 0 ? row.shot_clock : '-' }}</td>
            <td>{{ (row.phi_beta ?? -1).toFixed(3) }}</td>
            <td>{{ (row.phi_prev ?? -1).toFixed(3) }}</td>
            <td>{{ (row.phi_next ?? -1).toFixed(3) }}</td>
            <td>{{ (row.team_best_ep ?? -1).toFixed(3) }}</td>
            <td>{{ (row.ball_handler_ep ?? -1).toFixed(3) }}</td>
            <td>{{ row.best_ep_player >= 0 ? row.best_ep_player : '-' }}</td>
            <td>{{ (row.phi_r_shape ?? 0).toFixed(4) }}</td>
          </tr>
          <tr v-if="displayRows.length > 0">
            <td><strong>Total</strong></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td><strong>{{ displayRows.reduce((s, r) => s + (Number(r?.phi_r_shape) || 0), 0).toFixed(4) }}</strong></td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
  
</template>

<style scoped>
.phi-shaping { padding: 0.5rem; }
.controls { display: flex; flex-wrap: wrap; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem; }
.info-message { 
  width: 100%; 
  padding: 0.5rem; 
  background-color: #e3f2fd; 
  border-left: 3px solid #2196f3; 
  font-size: 0.9em; 
  margin-bottom: 0.5rem;
}
.param-info {
  width: 100%;
  padding: 0.4rem;
  background-color: #f5f5f5;
  border-left: 3px solid #9e9e9e;
  font-size: 0.85em;
  margin-bottom: 0.75rem;
  color: #666;
}
.blend { display: inline-flex; align-items: center; gap: 0.5rem; }
.aggregation-mode { display: inline-flex; align-items: center; gap: 0.5rem; }
.aggregation-mode select { padding: 0.25rem; }
label { display: inline-flex; align-items: center; gap: 0.4rem; }
.log { max-height: 300px; overflow: auto; }
table { width: 100%; border-collapse: collapse; }
th, td { border: 1px solid #ddd; padding: 0.25rem 0.5rem; text-align: right; }
th:first-child, td:first-child { text-align: left; }
.error { color: red; margin-bottom: 0.5rem; }
</style>


