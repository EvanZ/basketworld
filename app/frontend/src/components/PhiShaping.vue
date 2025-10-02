<script setup>
import { ref, onMounted } from 'vue';
import { getPhiParams, setPhiParams, getPhiLog } from '../services/api';

const loading = ref(false);
const error = ref(null);
const params = ref({ enable_phi_shaping: false, phi_beta: 0.0, reward_shaping_gamma: 1.0, phi_use_ball_handler_only: false, phi_blend_weight: 0.0 });
const logRows = ref([]);

async function loadParams() {
  try {
    const p = await getPhiParams();
    // Overwrite UI with server values only when explicitly loading params
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
    logRows.value = Array.isArray(phi_log) ? phi_log.slice(-200).reverse() : [];
  } catch (e) {
    error.value = String(e?.message || e);
  } finally {
    loading.value = false;
  }
}

async function applyParams() {
  loading.value = true;
  error.value = null;
  try {
    await setPhiParams(params.value);
    await loadParams();
    await refreshLog();
  } catch (e) {
    error.value = String(e?.message || e);
  } finally {
    loading.value = false;
  }
}

onMounted(() => {
  loadParams();
  refreshLog();
});

// Expose refresh so parent can call it after each step
defineExpose({ refresh: refreshLog });
</script>

<template>
  <div class="phi-shaping">
    <h3>Phi Shaping</h3>
    <div v-if="loading">Loading...</div>
    <div v-if="error" class="error">{{ error }}</div>

    <div class="controls">
      <label>
        <input type="checkbox" v-model="params.enable_phi_shaping" /> Enable Phi Shaping
      </label>
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
      <div class="blend">
        <label>Blend w (Team vs Ball Φ)</label>
        <input type="range" min="0" max="1" step="0.05" v-model.number="params.phi_blend_weight" />
        <span>{{ params.phi_blend_weight.toFixed(2) }}</span>
      </div>
      <button @click="applyParams" :disabled="loading">Apply</button>
      <button @click="refresh" :disabled="loading" style="margin-left: 0.5rem;">Refresh</button>
    </div>

    <div class="log">
      <table>
        <thead>
          <tr>
            <th>Step</th>
            <th>β</th>
            <th>Φprev</th>
            <th>Φnext</th>
            <th>TeamBestEP</th>
            <th>BallEP</th>
            <th>r_shape/team</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="row in logRows" :key="row.step">
            <td>{{ row.step }}</td>
            <td>{{ (row.phi_beta ?? -1).toFixed(3) }}</td>
            <td>{{ (row.phi_prev ?? -1).toFixed(3) }}</td>
            <td>{{ (row.phi_next ?? -1).toFixed(3) }}</td>
            <td>{{ (row.team_best_ep ?? -1).toFixed(3) }}</td>
            <td>{{ (row.ball_handler_ep ?? -1).toFixed(3) }}</td>
            <td>{{ (row.phi_r_shape ?? 0).toFixed(4) }}</td>
          </tr>
          <tr v-if="logRows.length > 0">
            <td><strong>Total</strong></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td><strong>{{ logRows.reduce((s, r) => s + (Number(r?.phi_r_shape) || 0), 0).toFixed(4) }}</strong></td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
  
</template>

<style scoped>
.phi-shaping { padding: 0.5rem; }
.controls { display: flex; flex-wrap: wrap; align-items: center; gap: 0.75rem; margin-bottom: 1rem; }
.
blend { display: inline-flex; align-items: center; gap: 0.5rem; }
label { display: inline-flex; align-items: center; gap: 0.4rem; }
.log { max-height: 300px; overflow: auto; }
table { width: 100%; border-collapse: collapse; }
th, td { border: 1px solid #ddd; padding: 0.25rem 0.5rem; text-align: right; }
th:first-child, td:first-child { text-align: left; }
.error { color: red; margin-bottom: 0.5rem; }
</style>


