<script setup>
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { api } from './api'

const defaultConfig = () => ({
  name: `halfcourt-${new Date().toISOString().slice(0, 10)}`,
  preset: 'halfcourt_multi_possession',
  mlflow_tracking_uri: 'http://localhost:5000',
  mlflow_experiment_name: 'halfcourt_multi_possessions',
  num_updates: 5000,
  policy_seed: 0,
  historical_eval_updates: null,
  historical_eval_episodes: 200,
  possession_limit_start: 1,
  possession_limit_end: 25,
  possession_limit_ramp_updates: 5000,
  made_basket_restart_mode: 'check',
  check_setup_steps: 0,
  multi_possession_use_inbounds: true,
  overrides: {},
})

const PIN_STORAGE_KEY = 'basketworld.training.pinned-configs.v1'
const PIN_MIGRATION_KEY = 'basketworld.training.pinned-configs.migration.v1'
const ENTROPY_DECAY_PIN_MIGRATION = 'entropy-decay-updates'
const METRIC_STORAGE_KEY = 'basketworld.training.sparkline-metrics.v1'
const MAX_SELECTED_METRICS = 16
const DEFAULT_METRICS = [
  'jax/train/train_loop_steps_per_sec',
  'jax/train/total_loss',
  'jax/train/policy_loss',
  'jax/train/value_loss',
  'jax/train/entropy_coef',
  'jax/train/intent_disc_auc_ovr_macro_holdout',
  'jax/train/mean_completed_episode_length',
  'jax/train/game_reward_mean',
]
const DEFAULT_PINNED_FIELDS = [
  'mlflow_run_name',
  'mlflow_experiment_name',
  'num_updates',
  'entropy_decay_updates',
  'policy_seed',
  'multi_possession_limit_start',
  'multi_possession_limit_end',
  'multi_possession_limit_ramp_updates',
  'historical_eval_episodes',
  'made_basket_restart_mode',
  'check_setup_steps',
  'multi_possession_use_inbounds',
]
const APP_CONFIG_KEYS = {
  mlflow_run_name: 'name',
  mlflow_experiment_name: 'mlflow_experiment_name',
  num_updates: 'num_updates',
  policy_seed: 'policy_seed',
  historical_eval_updates: 'historical_eval_updates',
  historical_eval_episodes: 'historical_eval_episodes',
  multi_possession_limit: 'possession_limit_end',
  multi_possession_limit_start: 'possession_limit_start',
  multi_possession_limit_end: 'possession_limit_end',
  multi_possession_limit_ramp_updates: 'possession_limit_ramp_updates',
  multi_possession_use_inbounds: 'multi_possession_use_inbounds',
  made_basket_restart_mode: 'made_basket_restart_mode',
  check_setup_steps: 'check_setup_steps',
}

function loadPinnedConfigNames() {
  try {
    const stored = JSON.parse(window.localStorage.getItem(PIN_STORAGE_KEY))
    const pinned = Array.isArray(stored)
      ? [...new Set(stored.filter((name) => typeof name === 'string'))]
      : [...DEFAULT_PINNED_FIELDS]
    if (window.localStorage.getItem(PIN_MIGRATION_KEY) !== ENTROPY_DECAY_PIN_MIGRATION) {
      if (!pinned.includes('entropy_decay_updates')) pinned.push('entropy_decay_updates')
      window.localStorage.setItem(PIN_STORAGE_KEY, JSON.stringify(pinned))
      window.localStorage.setItem(PIN_MIGRATION_KEY, ENTROPY_DECAY_PIN_MIGRATION)
    }
    return pinned
  } catch {
    return [...DEFAULT_PINNED_FIELDS]
  }
}

function loadSelectedMetricNames() {
  try {
    const stored = JSON.parse(window.localStorage.getItem(METRIC_STORAGE_KEY))
    return Array.isArray(stored)
      ? [...new Set(stored.filter((name) => typeof name === 'string'))].slice(0, MAX_SELECTED_METRICS)
      : [...DEFAULT_METRICS]
  } catch {
    return [...DEFAULT_METRICS]
  }
}

const runs = ref([])
const selected = ref(null)
const config = ref(defaultConfig())
const preview = ref(null)
const configSchema = ref([])
const configSearch = ref('')
const runConfigSearch = ref('')
const showFrozenConfigs = ref(false)
const showInternalConfigs = ref(false)
const pinnedConfigNames = ref(loadPinnedConfigNames())
const mlflowRunId = ref('')
const copiedMlflowRunId = ref(false)
const importingMlflow = ref(false)
const importResult = ref(null)
const loading = ref(false)
const launching = ref(false)
const error = ref('')
const logs = ref('')
const availableMetrics = ref([])
const selectedMetricNames = ref(loadSelectedMetricNames())
const metricSeries = ref({})
const metricToAdd = ref('')
const metricPickerOpen = ref(false)
const metricsLoading = ref(false)
const metricsError = ref('')
const activeView = ref('runs')
let refreshTimer = null
let previewTimer = null
let metricRefreshTimer = null
let lastMetricCatalogAt = 0

const activeRun = computed(() => runs.value.find((run) =>
  ['starting', 'running', 'pausing', 'stopping'].includes(run.status)
))
const canPause = computed(() => selected.value?.status === 'running')
const canResume = computed(() => selected.value?.status === 'paused')
const canCheckpoint = computed(() => selected.value?.status === 'running')
const canStop = computed(() => ['starting', 'running', 'pausing'].includes(selected.value?.status))
const filteredConfigSchema = computed(() => {
  const query = configSearch.value.trim().toLowerCase()
  return configSchema.value.filter((field) => {
    if (field.frozen && !showFrozenConfigs.value) return false
    if (field.internal && !showInternalConfigs.value) return false
    return !query || [field.name, field.help, field.category, field.type]
      .some((value) => String(value || '').toLowerCase().includes(query))
  })
})
const pinnedConfigFields = computed(() => pinnedConfigNames.value
  .map((name) => configSchema.value.find((field) => field.name === name))
  .filter(Boolean))
const filteredRunConfig = computed(() => {
  const query = runConfigSearch.value.trim().toLowerCase()
  const fields = selected.value?.resolved_config || []
  if (!query) return fields
  return fields.filter((field) => [
    field.name,
    field.help,
    field.category,
    field.type,
    JSON.stringify(field.value),
  ].some((value) => String(value || '').toLowerCase().includes(query)))
})
const availableMetricNames = computed(() => new Set(availableMetrics.value.map((metric) => metric.name)))
const addableMetrics = computed(() => availableMetrics.value.filter((metric) => !selectedMetricNames.value.includes(metric.name)))
const metricPickerResults = computed(() => {
  const query = metricToAdd.value.trim().toLowerCase()
  return addableMetrics.value
    .filter((metric) => !query || metric.name.toLowerCase().includes(query))
    .slice(0, 60)
})
const metricCards = computed(() => selectedMetricNames.value.map((name) => ({
  name,
  points: metricSeries.value[name] || [],
  available: availableMetricNames.value.has(name),
})))

const applicationConfigValues = computed(() => ({
  num_updates: config.value.num_updates,
  policy_seed: config.value.policy_seed,
  historical_eval_updates: config.value.historical_eval_updates,
  historical_eval_episodes: config.value.historical_eval_episodes,
  mlflow_run_name: config.value.name,
  mlflow_experiment_name: config.value.mlflow_experiment_name,
  run_train_loop: true,
  log_mlflow: true,
  enable_multi_possession: true,
  multi_possession_limit: config.value.possession_limit_end,
  multi_possession_limit_start: config.value.possession_limit_start,
  multi_possession_limit_end: config.value.possession_limit_end,
  multi_possession_limit_ramp_updates: config.value.possession_limit_ramp_updates,
  multi_possession_use_inbounds: config.value.multi_possession_use_inbounds,
  made_basket_restart_mode: config.value.made_basket_restart_mode,
  check_setup_steps: config.value.check_setup_steps,
}))

function formatDate(value) {
  return value ? new Date(value).toLocaleString() : '—'
}

function formatDuration(totalSeconds) {
  const seconds = Math.max(0, Math.floor(Number(totalSeconds) || 0))
  const days = Math.floor(seconds / 86400)
  const hours = Math.floor((seconds % 86400) / 3600)
  const minutes = Math.floor((seconds % 3600) / 60)
  const remainder = seconds % 60
  const clock = [hours, minutes, remainder].map((value) => String(value).padStart(2, '0')).join(':')
  return days ? `${days}d ${clock}` : clock
}

function formatRunConfigValue(value) {
  if (typeof value === 'string') return value || '""'
  const serialized = JSON.stringify(value)
  return serialized === undefined ? String(value) : serialized
}

async function copyMlflowRunId(runId) {
  if (!runId) return
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(runId)
    } else {
      const textarea = document.createElement('textarea')
      textarea.value = runId
      textarea.style.position = 'fixed'
      textarea.style.opacity = '0'
      document.body.appendChild(textarea)
      textarea.select()
      const copied = document.execCommand('copy')
      textarea.remove()
      if (!copied) throw new Error('Clipboard copy was rejected')
    }
    copiedMlflowRunId.value = true
    window.setTimeout(() => { copiedMlflowRunId.value = false }, 1600)
  } catch (cause) {
    error.value = `Could not copy the MLflow run ID: ${cause.message}`
  }
}

function metricLabel(name) {
  return name.replace(/^jax\/(train|eval)\//, '').replaceAll('_', ' ')
}

function formatMetricValue(value) {
  if (!Number.isFinite(value)) return '—'
  const absolute = Math.abs(value)
  if ((absolute > 0 && absolute < 0.001) || absolute >= 100000) return value.toExponential(2)
  return value.toLocaleString(undefined, { maximumFractionDigits: absolute < 10 ? 4 : 2 })
}

function metricStats(points) {
  if (!points.length) return { latest: null, min: null, max: null, step: null }
  const values = points.map((point) => point.value)
  const latest = points[points.length - 1]
  return {
    latest: latest.value,
    min: Math.min(...values),
    max: Math.max(...values),
    step: latest.step,
  }
}

function sparklinePoints(points) {
  if (!points.length) return ''
  const width = 300
  const height = 82
  const padding = 4
  const values = points.map((point) => point.value)
  const min = Math.min(...values)
  const max = Math.max(...values)
  const span = max - min
  return points.map((point, index) => {
    const x = points.length === 1
      ? width / 2
      : padding + (index / (points.length - 1)) * (width - padding * 2)
    const y = span === 0
      ? height / 2
      : padding + (1 - (point.value - min) / span) * (height - padding * 2)
    return `${x.toFixed(2)},${y.toFixed(2)}`
  }).join(' ')
}

function addMetric() {
  const name = metricToAdd.value.trim()
  if (!name || !availableMetricNames.value.has(name) || selectedMetricNames.value.includes(name)) return
  if (selectedMetricNames.value.length >= MAX_SELECTED_METRICS) {
    metricsError.value = `You can pin up to ${MAX_SELECTED_METRICS} metrics.`
    return
  }
  selectedMetricNames.value = [...selectedMetricNames.value, name]
  metricToAdd.value = ''
  metricPickerOpen.value = false
  metricsError.value = ''
}

function chooseMetric(name) {
  metricToAdd.value = name
  addMetric()
  metricPickerOpen.value = false
}

function closeMetricPicker() {
  window.setTimeout(() => {
    metricPickerOpen.value = false
  }, 120)
}

function removeMetric(name) {
  selectedMetricNames.value = selectedMetricNames.value.filter((metric) => metric !== name)
}

async function loadMetrics({ refreshCatalog = false } = {}) {
  const run = selected.value
  if (!run?.id || !run.mlflow_run_id || metricsLoading.value) return
  const runId = run.id
  metricsLoading.value = true
  metricsError.value = ''
  try {
    const shouldRefreshCatalog = !metricPickerOpen.value && (
      refreshCatalog
      || !availableMetrics.value.length
      || Date.now() - lastMetricCatalogAt > 30000
    )
    if (shouldRefreshCatalog) {
      const catalog = await api.getMetricCatalog(runId)
      if (selected.value?.id !== runId) return
      availableMetrics.value = catalog.metrics || []
      lastMetricCatalogAt = Date.now()
    }
    const histories = await api.getMetricHistories(runId, selectedMetricNames.value)
    if (selected.value?.id === runId) metricSeries.value = histories.series || {}
  } catch (cause) {
    if (selected.value?.id === runId) metricsError.value = cause.message
  } finally {
    metricsLoading.value = false
  }
}

function progress(run) {
  if (!run?.target_updates) return 0
  return Math.min(100, (run.current_update / run.target_updates) * 100)
}

function configHelp(name, fallback = '') {
  const field = configSchema.value.find((candidate) => candidate.name === name)
  if (!field) return fallback
  return `${field.help}\n\nType: ${field.type}${field.item_type ? ` of ${field.item_type}` : ''}\nParser default: ${JSON.stringify(field.default)}\nPreset default: ${JSON.stringify(field.effective_value)}`
}

function configFieldValue(field) {
  if (Object.prototype.hasOwnProperty.call(applicationConfigValues.value, field.name)) {
    return applicationConfigValues.value[field.name]
  }
  if (Object.prototype.hasOwnProperty.call(config.value.overrides, field.name)) {
    return config.value.overrides[field.name]
  }
  return field.effective_value
}

function fieldLabel(field) {
  const labels = {
    mlflow_run_name: 'Run name',
    mlflow_experiment_name: 'MLflow experiment',
    num_updates: 'Updates',
    policy_seed: 'Policy seed',
    multi_possession_limit_start: 'Start possessions',
    multi_possession_limit_end: 'Final possessions',
    multi_possession_limit_ramp_updates: 'Curriculum updates',
    historical_eval_episodes: 'Historical eval episodes',
    made_basket_restart_mode: 'Made-basket restart',
    check_setup_steps: 'Pre-check setup steps',
    multi_possession_use_inbounds: 'Use inbounds for dead-ball restarts',
  }
  return labels[field.name] || field.name.replaceAll('_', ' ')
}

function fieldIsEditable(field) {
  return field.editable || Object.prototype.hasOwnProperty.call(APP_CONFIG_KEYS, field.name)
}

function isPinned(field) {
  return pinnedConfigNames.value.includes(field.name)
}

function togglePinned(field) {
  pinnedConfigNames.value = isPinned(field)
    ? pinnedConfigNames.value.filter((name) => name !== field.name)
    : [...pinnedConfigNames.value, field.name]
}

function inputValue(field) {
  const value = configFieldValue(field)
  if (Array.isArray(value)) return value.join(', ')
  return value ?? ''
}

function parseConfigFieldValue(field, rawValue) {
  let value = rawValue
  if (field.type === 'int') value = rawValue === '' ? null : Number.parseInt(rawValue, 10)
  if (field.type === 'float') value = rawValue === '' ? null : Number.parseFloat(rawValue)
  if (field.type === 'list') {
    value = rawValue.trim() ? rawValue.split(/[\s,]+/).map((item) => {
      if (field.item_type === 'int') return Number.parseInt(item, 10)
      if (field.item_type === 'float') return Number.parseFloat(item)
      return item
    }) : []
  }
  return value
}

function setConfigField(field, rawValue) {
  let value = parseConfigFieldValue(field, rawValue)
  const appConfigKey = APP_CONFIG_KEYS[field.name]
  if (appConfigKey) {
    if (field.name === 'historical_eval_updates') {
      const items = Array.isArray(value)
        ? value
        : String(value || '').split(/[\s,]+/).filter(Boolean)
      value = items.length ? items.map((item) => Number.parseInt(item, 10)) : null
    }
    config.value[appConfigKey] = value
    return
  }
  config.value.overrides = { ...config.value.overrides, [field.name]: value }
}

function resetOverride(field) {
  const next = { ...config.value.overrides }
  delete next[field.name]
  config.value.overrides = next
}

function cloneRun(run) {
  const source = JSON.parse(JSON.stringify(run.config || {}))
  config.value = {
    ...defaultConfig(),
    ...source,
    name: `${run.name} copy`,
    overrides: { ...(source.overrides || {}) },
  }
  activeView.value = 'new'
}

async function importFromMlflow() {
  if (!mlflowRunId.value.trim()) return
  importingMlflow.value = true
  importResult.value = null
  error.value = ''
  try {
    const result = await api.importMlflowRun(config.value.mlflow_tracking_uri, mlflowRunId.value.trim())
    config.value = {
      ...defaultConfig(),
      ...result.config,
      overrides: { ...(result.config.overrides || {}) },
    }
    importResult.value = result
  } catch (cause) {
    error.value = cause.message
  } finally {
    importingMlflow.value = false
  }
}

async function loadRuns({ preserveSelection = true } = {}) {
  try {
    const nextRuns = await api.listRuns()
    runs.value = nextRuns
    if (preserveSelection && selected.value) {
      const refreshed = nextRuns.find((run) => run.id === selected.value.id)
      if (refreshed) {
        if (activeView.value === 'detail') {
          const [detail, logText] = await Promise.all([
            api.getRun(refreshed.id),
            api.getLogs(refreshed.id),
          ])
          selected.value = detail
          logs.value = logText
        } else {
          selected.value = refreshed
        }
      }
    }
  } catch (cause) {
    error.value = cause.message
  }
}

async function selectRun(run) {
  try {
    const [detail, logText] = await Promise.all([
      api.getRun(run.id),
      api.getLogs(run.id),
    ])
    selected.value = detail
    logs.value = logText
    activeView.value = 'detail'
    availableMetrics.value = []
    metricSeries.value = {}
    lastMetricCatalogAt = 0
    await loadMetrics({ refreshCatalog: true })
  } catch (cause) {
    error.value = cause.message
  }
}

async function loadPreview() {
  try {
    preview.value = await api.previewRun(config.value)
    error.value = ''
  } catch (cause) {
    preview.value = null
    error.value = cause.message
  }
}

watch(config, () => {
  window.clearTimeout(previewTimer)
  previewTimer = window.setTimeout(loadPreview, 250)
}, { deep: true })

watch(pinnedConfigNames, (names) => {
  window.localStorage.setItem(PIN_STORAGE_KEY, JSON.stringify(names))
}, { deep: true })

watch(selectedMetricNames, (names) => {
  window.localStorage.setItem(METRIC_STORAGE_KEY, JSON.stringify(names))
  if (selected.value && activeView.value === 'detail') loadMetrics()
}, { deep: true })

async function launch() {
  launching.value = true
  error.value = ''
  try {
    const run = await api.createRun(config.value, crypto.randomUUID())
    await loadRuns({ preserveSelection: false })
    await selectRun(run)
  } catch (cause) {
    error.value = cause.message
  } finally {
    launching.value = false
  }
}

async function runAction(action) {
  if (!selected.value) return
  loading.value = true
  error.value = ''
  try {
    const response = await api.action(selected.value.id, action)
    selected.value = response.run
    await loadRuns()
  } catch (cause) {
    error.value = cause.message
  } finally {
    loading.value = false
  }
}

onMounted(async () => {
  const [, , schema] = await Promise.all([loadRuns(), loadPreview(), api.getConfigSchema()])
  configSchema.value = schema
  refreshTimer = window.setInterval(() => loadRuns(), 2500)
  metricRefreshTimer = window.setInterval(() => {
    if (activeView.value === 'detail') loadMetrics()
  }, 10000)
})

onBeforeUnmount(() => {
  window.clearInterval(refreshTimer)
  window.clearInterval(metricRefreshTimer)
  window.clearTimeout(previewTimer)
})
</script>

<template>
  <div class="app-shell">
    <aside class="sidebar">
      <div class="brand">
        <div class="brand-ball">BW</div>
        <div><strong>BasketWorld</strong><span>Training Control</span></div>
      </div>
      <nav>
        <button :class="{ active: activeView === 'runs' }" @click="activeView = 'runs'">Runs</button>
        <button :class="{ active: activeView === 'new' }" @click="activeView = 'new'">New run</button>
        <button :disabled="!selected" :class="{ active: activeView === 'detail' }" @click="activeView = 'detail'">Run detail</button>
      </nav>
      <div class="worker-card">
        <span class="eyebrow">TRAINING WORKER</span>
        <strong>{{ activeRun ? 'Occupied' : 'Available' }}</strong>
        <small v-if="activeRun">{{ activeRun.name }}</small>
        <small v-else>One local training slot</small>
      </div>
    </aside>

    <main>
      <header>
        <div>
          <p class="eyebrow">STANDALONE CONTROL PLANE</p>
          <h1>{{ activeView === 'new' ? 'Launch training' : activeView === 'detail' ? 'Run detail' : 'Training runs' }}</h1>
        </div>
        <button class="ghost" @click="loadRuns">Refresh</button>
      </header>

      <div v-if="error" class="error-banner">{{ error }}</div>

      <section v-if="activeView === 'runs'" class="panel">
        <div class="panel-heading">
          <div><h2>Experiments</h2><p>Durable workers and their latest persisted state.</p></div>
          <button class="primary" :disabled="Boolean(activeRun)" @click="activeView = 'new'">New run</button>
        </div>
        <div v-if="!runs.length" class="empty">No training runs yet.</div>
        <button v-for="run in runs" :key="run.id" class="run-row" @click="selectRun(run)">
          <span class="status-dot" :class="run.status"></span>
          <span class="run-name"><strong>{{ run.name }}</strong><small>{{ formatDate(run.created_at) }}</small></span>
          <span class="status-pill" :class="run.status">{{ run.status }}</span>
          <span class="run-progress">
            <span>{{ run.current_update.toLocaleString() }} / {{ run.target_updates.toLocaleString() }}</span>
            <span class="progress-track"><i :style="{ width: `${progress(run)}%` }"></i></span>
          </span>
        </button>
      </section>

      <section v-else-if="activeView === 'new'" class="new-grid">
        <form class="panel form-panel" @submit.prevent="launch">
          <div class="panel-heading"><div><h2>Experiment</h2><p>Validated halfcourt multi-possession preset.</p></div></div>
          <section class="mlflow-import wide">
            <div>
              <strong>Copy an MLflow run</strong>
              <small>Paste a run ID to populate a new launch configuration.</small>
            </div>
            <div class="mlflow-import-row">
              <input v-model="mlflowRunId" placeholder="MLflow run ID" />
              <button type="button" :disabled="importingMlflow || !mlflowRunId.trim()" @click="importFromMlflow">{{ importingMlflow ? 'Importing…' : 'Import config' }}</button>
            </div>
            <p v-if="importResult" class="import-result" :class="{ exact: importResult.exact }">
              {{ importResult.exact ? 'Exact resolved configuration loaded' : 'Legacy parameter reconstruction loaded' }} from {{ importResult.source_run_name || importResult.source_run_id }} in {{ importResult.experiment_name }}. Matched {{ importResult.matched_fields.length }} of {{ configSchema.length }} settings<span v-if="!importResult.exact">; review the missing settings before launch</span>.
            </p>
          </section>
          <label class="wide"><span class="label-title">MLflow tracking URI <button type="button" class="info-button" :title="'Address of the MLflow tracking server used for parameters, metrics, and artifacts.\n\nType: str\nDefault: http://localhost:5000'">i</button></span><input v-model="config.mlflow_tracking_uri" required maxlength="2048" placeholder="http://localhost:5000" /></label>
          <section class="pinned-configs wide">
            <div class="pinned-heading">
              <div><strong>Pinned configs</strong><small>Keep frequently used settings at the top of the form.</small></div>
              <span>{{ pinnedConfigFields.length }} pinned</span>
            </div>
            <div v-if="pinnedConfigFields.length" class="pinned-grid">
              <article v-for="field in pinnedConfigFields" :key="field.name" class="pinned-field">
                <div class="pinned-field-heading">
                  <span class="label-title">{{ fieldLabel(field) }} <button type="button" class="info-button" :title="configHelp(field.name)">i</button></span>
                  <button type="button" class="pin-button active" :aria-label="`Unpin ${field.name}`" :title="`Unpin ${field.name}`" @click="togglePinned(field)">📌</button>
                </div>
                <label v-if="field.type === 'bool'" class="toggle config-input">
                  <input type="checkbox" :checked="Boolean(configFieldValue(field))" :disabled="!fieldIsEditable(field)" @change="setConfigField(field, $event.target.checked)" />
                  <span>{{ configFieldValue(field) ? 'true' : 'false' }}</span>
                </label>
                <select v-else-if="field.choices" class="config-input" :value="inputValue(field)" :disabled="!fieldIsEditable(field)" @change="setConfigField(field, $event.target.value)">
                  <option v-for="choice in field.choices" :key="choice" :value="choice">{{ choice }}</option>
                </select>
                <input v-else class="config-input" :type="['int', 'float'].includes(field.type) ? 'number' : 'text'" :step="field.type === 'float' ? 'any' : undefined" :value="inputValue(field)" :disabled="!fieldIsEditable(field)" @input="setConfigField(field, $event.target.value)" />
                <small class="field-meta">{{ field.name }} · {{ field.type }}<span v-if="!fieldIsEditable(field)"> · {{ field.frozen ? 'frozen' : 'internal' }}</span></small>
              </article>
            </div>
            <p v-else class="empty-pins">No configs pinned. Pin one from All training configs below.</p>
          </section>
          <details class="config-browser wide">
            <summary>All training configs <span>{{ filteredConfigSchema.length }} shown · {{ configSchema.length }} total</span></summary>
            <div class="config-tools">
              <input v-model="configSearch" placeholder="Search names, help, type, or category…" />
              <div class="config-visibility">
                <label class="toggle"><input v-model="showFrozenConfigs" type="checkbox" />Show frozen</label>
                <label class="toggle"><input v-model="showInternalConfigs" type="checkbox" />Show internal</label>
              </div>
              <small>Pin any setting to keep it at the top. Internal and frozen settings remain visible but read-only.</small>
            </div>
            <div class="config-fields">
              <article v-for="field in filteredConfigSchema" :key="field.name" class="config-field" :class="{ overridden: Object.prototype.hasOwnProperty.call(config.overrides, field.name) }">
                <div class="config-field-label">
                  <strong>{{ field.name }}</strong>
                  <button type="button" class="info-button" :title="`${field.help}\n\nType: ${field.type}${field.item_type ? ` of ${field.item_type}` : ''}\nParser default: ${JSON.stringify(field.default)}\nPreset default: ${JSON.stringify(field.effective_value)}`">i</button>
                  <span>{{ field.type }}</span>
                  <small>{{ field.category }}</small>
                </div>
                <label v-if="field.type === 'bool'" class="toggle config-input">
                  <input type="checkbox" :checked="Boolean(configFieldValue(field))" :disabled="!fieldIsEditable(field)" @change="setConfigField(field, $event.target.checked)" />
                  <span>{{ configFieldValue(field) ? 'true' : 'false' }}</span>
                </label>
                <select v-else-if="field.choices" class="config-input" :value="inputValue(field)" :disabled="!fieldIsEditable(field)" @change="setConfigField(field, $event.target.value)">
                  <option v-for="choice in field.choices" :key="choice" :value="choice">{{ choice }}</option>
                </select>
                <input v-else class="config-input" :type="['int', 'float'].includes(field.type) ? 'number' : 'text'" :step="field.type === 'float' ? 'any' : undefined" :value="inputValue(field)" :disabled="!fieldIsEditable(field)" @input="setConfigField(field, $event.target.value)" />
                <div class="field-actions">
                  <button type="button" class="pin-button" :class="{ active: isPinned(field) }" :aria-label="`${isPinned(field) ? 'Unpin' : 'Pin'} ${field.name}`" :title="`${isPinned(field) ? 'Unpin' : 'Pin'} ${field.name}`" @click="togglePinned(field)">📌</button>
                  <button v-if="Object.prototype.hasOwnProperty.call(config.overrides, field.name)" type="button" class="reset-button" title="Restore preset default" @click="resetOverride(field)">Reset</button>
                  <span v-else-if="!fieldIsEditable(field)" class="lock-label">{{ field.frozen ? 'frozen' : 'internal' }}</span>
                </div>
              </article>
            </div>
          </details>
          <div class="launch-row wide">
            <p v-if="activeRun">The single v0 worker is occupied by {{ activeRun.name }}.</p>
            <p v-else>Launches as a detached worker; closing this page will not stop it.</p>
            <button class="primary" :disabled="launching || Boolean(activeRun)">{{ launching ? 'Launching…' : 'Launch training' }}</button>
          </div>
        </form>
        <aside class="panel command-panel">
          <p class="eyebrow">RESOLVED COMMAND</p>
          <pre>{{ preview?.display_command || 'Waiting for valid configuration…' }}</pre>
          <dl v-if="preview">
            <div><dt>Checkpoints</dt><dd>{{ preview.checkpoint_dir }}</dd></div>
            <div><dt>Logs</dt><dd>{{ preview.log_file }}</dd></div>
          </dl>
        </aside>
      </section>

      <section v-else class="detail-grid">
        <div v-if="!selected" class="panel empty">Select a run from the Runs page.</div>
        <template v-else>
          <div class="panel hero-panel">
            <div>
              <p class="eyebrow">{{ selected.preset }}</p>
              <h2>{{ selected.name }}</h2>
              <span class="status-pill" :class="selected.status">{{ selected.status }}</span>
            </div>
            <div class="actions">
              <button @click="cloneRun(selected)">Copy as new run</button>
              <button :disabled="loading || !canCheckpoint" @click="runAction('checkpoint')">Checkpoint</button>
              <button :disabled="loading || !canPause" @click="runAction('pause')">Pause</button>
              <button class="primary" :disabled="loading || !canResume" @click="runAction('resume')">Resume</button>
              <button class="danger" :disabled="loading || !canStop" @click="runAction('stop')">Stop</button>
            </div>
          </div>
          <div class="stat-grid">
            <article class="panel stat"><span>Update</span><strong>{{ selected.current_update.toLocaleString() }}</strong><small>of {{ selected.target_updates.toLocaleString() }}</small></article>
            <article class="panel stat"><span>Progress</span><strong>{{ progress(selected).toFixed(1) }}%</strong><small>{{ selected.worker_status?.state || selected.status }}</small></article>
            <article class="panel stat"><span>Training time</span><strong class="duration-value">{{ formatDuration(selected.training_elapsed_seconds) }}</strong><small>active elapsed</small></article>
            <article class="panel stat"><span>Steps / sec</span><strong>{{ Math.round(selected.worker_status?.metrics?.end_to_end_steps_per_sec || 0).toLocaleString() }}</strong><small>end to end</small></article>
            <article class="panel stat">
              <span>MLflow</span>
              <button
                v-if="selected.mlflow_run_id"
                type="button"
                class="mlflow-run-copy"
                :title="`Copy MLflow run ID ${selected.mlflow_run_id}`"
                :aria-label="`Copy MLflow run ID ${selected.mlflow_run_id}`"
                @click="copyMlflowRunId(selected.mlflow_run_id)"
              >
                <strong class="mono small-value">{{ selected.mlflow_run_id }}</strong>
                <small>{{ copiedMlflowRunId ? 'Copied!' : 'Click to copy run ID' }}</small>
              </button>
              <template v-else><strong class="mono small-value">Pending</strong><small>run ID</small></template>
            </article>
          </div>
          <div class="panel progress-panel"><span class="progress-track large"><i :style="{ width: `${progress(selected)}%` }"></i></span></div>
          <div class="panel metrics-panel">
            <div class="detail-section-heading metrics-heading">
              <div><h3>MLflow metrics</h3><span>{{ selectedMetricNames.length }} pinned · {{ availableMetrics.length }} available</span></div>
              <div class="metric-picker">
                <div class="metric-picker-search">
                  <input v-model="metricToAdd" autocomplete="off" aria-label="Metric to add" placeholder="Search or paste a metric name…" @focus="metricPickerOpen = true" @input="metricPickerOpen = true" @blur="closeMetricPicker" @keyup.enter="addMetric" @keydown.esc="metricPickerOpen = false" />
                  <div v-if="metricPickerOpen" class="metric-picker-menu">
                    <button v-for="metric in metricPickerResults" :key="metric.name" type="button" @mousedown.prevent="chooseMetric(metric.name)">
                      <span>{{ metric.name }}</span><small>{{ formatMetricValue(metric.latest) }}</small>
                    </button>
                    <p v-if="!metricPickerResults.length">No matching unpinned metrics.</p>
                  </div>
                </div>
                <button type="button" :disabled="!availableMetricNames.has(metricToAdd.trim()) || selectedMetricNames.length >= MAX_SELECTED_METRICS" @click="addMetric">Add</button>
              </div>
            </div>
            <p v-if="metricsError" class="metric-error">{{ metricsError }}</p>
            <div v-if="!selected.mlflow_run_id" class="empty-metrics">Waiting for the MLflow run to start…</div>
            <div v-else-if="metricCards.length" class="sparkline-grid">
              <article v-for="metric in metricCards" :key="metric.name" class="sparkline-card">
                <div class="sparkline-card-heading">
                  <div><strong>{{ metricLabel(metric.name) }}</strong><small :title="metric.name">{{ metric.name }}</small></div>
                  <button type="button" :aria-label="`Unpin ${metric.name}`" :title="`Unpin ${metric.name}`" @click="removeMetric(metric.name)">×</button>
                </div>
                <template v-if="metric.points.length">
                  <div class="metric-value">{{ formatMetricValue(metricStats(metric.points).latest) }} <small>step {{ metricStats(metric.points).step.toLocaleString() }}</small></div>
                  <svg class="sparkline" viewBox="0 0 300 82" preserveAspectRatio="none" role="img" :aria-label="`${metric.name} trend`">
                    <polyline :points="sparklinePoints(metric.points)" />
                  </svg>
                  <div class="metric-range"><span>min {{ formatMetricValue(metricStats(metric.points).min) }}</span><span>max {{ formatMetricValue(metricStats(metric.points).max) }}</span></div>
                </template>
                <div v-else class="sparkline-empty">{{ metric.available ? (metricsLoading ? 'Loading…' : 'No history yet') : 'Not logged by this run yet' }}</div>
              </article>
            </div>
            <div v-else class="empty-metrics">Add a metric to begin monitoring.</div>
          </div>
          <div class="panel metadata">
            <h3>Runtime</h3>
            <dl>
              <div><dt>PID</dt><dd>{{ selected.pid || '—' }}</dd></div>
              <div><dt>MLflow tracking URI</dt><dd>{{ selected.config.mlflow_tracking_uri }}</dd></div>
              <div><dt>MLflow experiment</dt><dd>{{ selected.config.mlflow_experiment_name }}</dd></div>
              <div><dt>Started</dt><dd>{{ formatDate(selected.started_at) }}</dd></div>
              <div><dt>Latest checkpoint</dt><dd>{{ selected.checkpoint_path || 'Pending' }}</dd></div>
              <div><dt>Log file</dt><dd>{{ selected.log_file }}</dd></div>
              <div v-if="selected.error_message"><dt>Error</dt><dd class="error-text">{{ selected.error_message }}</dd></div>
            </dl>
          </div>
          <div class="panel run-command-panel">
            <div class="detail-section-heading">
              <div><h3>Launch command</h3><span>Equivalent CLI command for this worker invocation</span></div>
            </div>
            <pre>{{ selected.display_command }}</pre>
          </div>
          <div class="panel run-config-panel">
            <div class="detail-section-heading">
              <div><h3>Resolved configuration</h3><span>{{ selected.resolved_config?.length || 0 }} read-only settings used by this run</span></div>
              <input v-model="runConfigSearch" aria-label="Search run configuration" placeholder="Search config, value, type, or category…" />
            </div>
            <div class="readonly-config-list">
              <article v-for="field in filteredRunConfig" :key="field.name" class="readonly-config-row">
                <div class="readonly-config-name">
                  <strong>{{ field.name }}</strong>
                  <span>{{ field.type }}</span>
                  <span>{{ field.category }}</span>
                  <small>{{ field.help }}</small>
                </div>
                <code>{{ formatRunConfigValue(field.value) }}</code>
              </article>
              <p v-if="!filteredRunConfig.length" class="empty-config-search">No configuration settings match this search.</p>
            </div>
          </div>
          <div class="panel logs-panel">
            <div class="logs-heading"><h3>Worker log</h3><span>Last 100 KB</span></div>
            <pre>{{ logs || 'Waiting for worker output…' }}</pre>
          </div>
        </template>
      </section>
    </main>
  </div>
</template>
