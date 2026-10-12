<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue';
import {
  extendSkillRatingsStudy,
  getActiveSkillRatingsJob,
  getSkillRatingsJob,
  getSkillRatingsReport,
  launchSkillRatingsStudy,
  listSkillRatingsReports,
} from '../services/api';
import {
  latestReportedProgress,
  reportedProgressPercent,
  selectedAnalysisIdentity,
  skillRatingsReportOptionLabel,
} from '../utils/skillRatingsWorkbench';

const props = defineProps({
  checkpointPath: {
    type: String,
    default: '',
  },
  checkpointUpdate: {
    type: [Number, String],
    default: null,
  },
  modelBackend: {
    type: String,
    default: '',
  },
});

function studyStamp() {
  const now = new Date();
  const pad = (value) => String(value).padStart(2, '0');
  return `${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}-${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`;
}

function defaultConfig() {
  const stem = `artifacts/skill_ratings/${studyStamp()}`;
  return {
    poolPath: `${stem}/player_pool.json`,
    createPool: true,
    poolSize: 0,
    poolSeed: 0,
    schedulePath: `${stem}/lineup_schedule.json`,
    createSchedule: true,
    gameCount: 0,
    scheduleSeed: 1,
    artifactDir: stem,
    // This is the durable study library, deliberately separate from each
    // timestamped launch destination.  Results therefore survive new-study
    // paths, tab remounts, and completed jobs.
    resultsArtifactDir: 'artifacts/skill_ratings',
    actionMode: 'direct',
    batchSize: 64,
    evalSeed: 0,
    validationFraction: 0.2,
    splitSeed: 0,
    bootstrapSamples: 200,
    bootstrapSeed: 1,
  };
}

const config = ref(defaultConfig());
const job = ref(null);
const reports = ref([]);
const selectedReportPath = ref('');
const report = ref(null);
const error = ref('');
const isLaunching = ref(false);
const isExtending = ref(false);
const isLoadingReports = ref(false);
const copyNotice = ref('');
const additionalGames = ref(640);
const extensionScheduleSeed = ref(1);
let pollTimer = null;

const selectedCheckpointPath = computed(() => String(props.checkpointPath || '').trim());
const selectedPolicyIsJax = computed(() => String(props.modelBackend || '').toLowerCase().startsWith('jax'));
const canLaunch = computed(() => selectedPolicyIsJax.value && Boolean(selectedCheckpointPath.value));
const isRunning = computed(() => ['queued', 'running'].includes(job.value?.status));
const latestEvent = computed(() => job.value?.events?.at(-1) || null);
const batchProgress = computed(() => latestReportedProgress(job.value?.events || []));
const progressPercent = computed(() => reportedProgressPercent(batchProgress.value));
const selectedAnalysis = computed(() => selectedAnalysisIdentity(report.value));
const firstBatchStart = computed(() => (job.value?.events || []).find((event) => event.phase === 'batch_started') || null);
const latestCompletedBatch = computed(() => {
  const events = job.value?.events || [];
  for (let index = events.length - 1; index >= 0; index -= 1) {
    if (events[index].phase === 'batch_completed') return events[index];
  }
  return null;
});
const observedGamesPerSecond = computed(() => {
  const start = firstBatchStart.value;
  const complete = latestCompletedBatch.value;
  if (!start || !complete) return null;
  const games = Number(complete.completed) - Number(start.completed);
  const elapsed = Number(complete.elapsed_sec) - Number(start.elapsed_sec);
  return games > 0 && elapsed > 0 ? games / elapsed : null;
});
const estimatedGameSecondsRemaining = computed(() => {
  const rate = observedGamesPerSecond.value;
  const progress = batchProgress.value;
  if (!rate || !progress) return null;
  return Math.max(0, (Number(progress.total) - Number(progress.completed)) / rate);
});
const totalGames = computed(() => batchProgress.value?.total ?? null);
const totalBatchCount = computed(() => {
  const total = Number(totalGames.value);
  const batchSize = Number(job.value?.config?.batch_size || config.value.batchSize);
  return Number.isFinite(total) && total > 0 && batchSize > 0 ? Math.ceil(total / batchSize) : null;
});

function numberOr(value, fallback = 0) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function studyPayload() {
  return {
    checkpoint_paths: [selectedCheckpointPath.value],
    pool_path: config.value.poolPath.trim(),
    create_pool: config.value.createPool,
    pool_size: numberOr(config.value.poolSize),
    pool_seed: numberOr(config.value.poolSeed),
    schedule_path: config.value.schedulePath.trim(),
    create_schedule: config.value.createSchedule,
    game_count: numberOr(config.value.gameCount),
    schedule_seed: numberOr(config.value.scheduleSeed),
    artifact_dir: config.value.artifactDir.trim(),
    action_mode: config.value.actionMode,
    batch_size: numberOr(config.value.batchSize, 64),
    eval_seed: numberOr(config.value.evalSeed),
    validation_fraction: numberOr(config.value.validationFraction, 0.2),
    split_seed: numberOr(config.value.splitSeed),
    bootstrap_samples: numberOr(config.value.bootstrapSamples, 200),
    bootstrap_seed: numberOr(config.value.bootstrapSeed, 1),
  };
}

function formatNumber(value, digits = 2) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric.toFixed(digits) : '—';
}

function formatPercent(value, digits = 1) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? `${numeric.toFixed(digits)}%` : '—';
}

function formatSeed(value) {
  return value === null || value === undefined || value === '' ? 'Not recorded' : String(value);
}

function formatPhase(phase) {
  return String(phase || 'waiting').replaceAll('_', ' ');
}

function formatElapsed(seconds) {
  const numeric = Number(seconds);
  if (!Number.isFinite(numeric)) return '—';
  if (numeric < 60) return `${numeric.toFixed(1)}s`;
  return `${Math.floor(numeric / 60)}m ${Math.round(numeric % 60)}s`;
}

function formatRate(gamesPerSecond) {
  const numeric = Number(gamesPerSecond);
  return Number.isFinite(numeric) && numeric > 0 ? `${(numeric * 60).toFixed(1)} games/min` : '—';
}

function makeFreshStudyPaths() {
  const fresh = defaultConfig();
  config.value = { ...config.value, artifactDir: fresh.artifactDir, poolPath: fresh.poolPath, schedulePath: fresh.schedulePath, createPool: true, createSchedule: true };
}

function stopPolling() {
  if (pollTimer !== null) {
    window.clearInterval(pollTimer);
    pollTimer = null;
  }
}

function startPolling() {
  stopPolling();
  pollTimer = window.setInterval(refreshJob, 2000);
}

async function refreshJob() {
  if (!job.value?.id) return;
  try {
    job.value = await getSkillRatingsJob(job.value.id);
    if (!isRunning.value) {
      stopPolling();
      if (job.value.status === 'completed') {
        await refreshReports();
        const newestPath = job.value.report_paths?.[0];
        if (newestPath) await loadReport(newestPath);
      }
    }
  } catch (err) {
    error.value = err.message || 'Could not refresh Skill Ratings progress.';
    stopPolling();
  }
}

async function resumeActiveJob() {
  try {
    const response = await getActiveSkillRatingsJob();
    if (!response.job) return false;
    job.value = response.job;
    startPolling();
    return true;
  } catch (err) {
    // Results are still usable if the optional status lookup is unavailable.
    console.warn('Could not reattach to active Skill Ratings study:', err);
    return false;
  }
}

async function launch() {
  error.value = '';
  if (!canLaunch.value) {
    error.value = 'Skill Ratings requires the selected JAX multi-possession policy to expose its checkpoint path.';
    return;
  }
  isLaunching.value = true;
  try {
    job.value = await launchSkillRatingsStudy(studyPayload());
    startPolling();
  } catch (err) {
    error.value = err.message || 'Could not launch Skill Ratings study.';
    if (/already running/i.test(error.value) && await resumeActiveJob()) error.value = '';
  } finally {
    isLaunching.value = false;
  }
}

async function refreshReports() {
  error.value = '';
  isLoadingReports.value = true;
  try {
    const response = await listSkillRatingsReports(config.value.resultsArtifactDir.trim());
    reports.value = response.reports || [];
  } catch (err) {
    error.value = err.message || 'Could not list Skill Ratings reports.';
  } finally {
    isLoadingReports.value = false;
  }
}

async function loadReport(path = selectedReportPath.value) {
  if (!path) return;
  error.value = '';
  try {
    selectedReportPath.value = path;
    report.value = await getSkillRatingsReport(path);
    const sourceGames = Number(report.value?.reproducibility?.config?.game_count);
    const sourceSeed = Number(report.value?.reproducibility?.config?.schedule_seed);
    if (Number.isFinite(sourceGames) && sourceGames > 0) additionalGames.value = sourceGames;
    if (Number.isFinite(sourceSeed)) extensionScheduleSeed.value = sourceSeed + 1;
  } catch (err) {
    error.value = err.message || 'Could not load Skill Ratings report.';
  }
}

async function copyReportConfig() {
  const reproducibility = report.value?.reproducibility;
  if (!reproducibility?.config) return;
  copyNotice.value = '';
  try {
    await navigator.clipboard.writeText(JSON.stringify(reproducibility.config, null, 2));
    copyNotice.value = 'Launch configuration copied.';
  } catch (err) {
    error.value = 'Could not copy the report configuration.';
  }
}

async function extendReport() {
  if (!selectedReportPath.value) return;
  const games = numberOr(additionalGames.value);
  if (games < 1) {
    error.value = 'Choose at least one additional game.';
    return;
  }
  error.value = '';
  isExtending.value = true;
  try {
    job.value = await extendSkillRatingsStudy({
      report_path: selectedReportPath.value,
      additional_games: games,
      schedule_seed: numberOr(extensionScheduleSeed.value),
      batch_size: numberOr(config.value.batchSize, 64),
    });
    startPolling();
  } catch (err) {
    error.value = err.message || 'Could not extend the Skill Ratings study.';
  } finally {
    isExtending.value = false;
  }
}

onMounted(async () => {
  await Promise.all([refreshReports(), resumeActiveJob()]);
});
onBeforeUnmount(stopPolling);
</script>

<template>
  <section class="skill-ratings-workbench">
    <header class="workbench-header">
      <div>
        <p class="eyebrow">Dev tools</p>
        <h1>Skill Ratings</h1>
        <p class="intro">Run a balanced multi-possession study, then inspect adjusted plus/minus, player skill profiles, and the three factors.</p>
      </div>
      <button class="secondary" type="button" @click="makeFreshStudyPaths">New study paths</button>
    </header>

    <p v-if="error" class="error-message">{{ error }}</p>

    <div class="workbench-grid">
      <form class="config-card" @submit.prevent="launch">
        <div class="card-heading">
          <h2>Configure study</h2>
          <span>Artifacts stay immutable once created.</span>
        </div>

        <details class="settings-guide">
          <summary>How to use these settings</summary>
          <p>For a first pass, leave the size and matchup counts at <strong>0</strong>, use <strong>Direct</strong> actions, and keep the analysis defaults. Change seeds only when you want a deliberately independent replicate.</p>
          <p>Pool and schedule files are immutable study inputs. “New study paths” creates fresh names so a later run cannot overwrite earlier evidence.</p>
        </details>

        <fieldset class="selected-policy">
          <legend>Selected policy</legend>
          <p v-if="canLaunch" class="selected-policy-name">
            This study will evaluate the JAX checkpoint currently loaded in the game.
          </p>
          <p v-else class="error-message">
            Load a compatible JAX multi-possession policy in Game setup before launching Skill Ratings.
          </p>
          <label>
            <span class="field-label">Current checkpoint <span class="help-tip" tabindex="0" title="This is read-only. Skill Ratings always evaluates the JAX checkpoint currently loaded in the game, so the result is tied to the policy you selected.">?</span></span>
            <input :value="selectedCheckpointPath || 'No compatible checkpoint loaded'" readonly />
          </label>
          <p v-if="checkpointUpdate !== null && checkpointUpdate !== undefined" class="muted">Checkpoint update {{ checkpointUpdate }}</p>
        </fieldset>

        <fieldset>
          <legend>Player pool and schedule</legend>
          <label class="toggle"><input v-model="config.createPool" type="checkbox" /> Create a new player pool <span class="help-tip" tabindex="0" title="A player pool is the synthetic set of player skill profiles being rated. Turn this off to reuse an existing immutable pool exactly.">?</span></label>
          <div class="two-column">
            <label>
              <span class="field-label">Player pool path <span class="help-tip" tabindex="0" title="Where the immutable player-pool JSON is written or loaded from. Use a new path when creating; use an existing path only when reusing that exact pool.">?</span></span>
              <input v-model="config.poolPath" required />
            </label>
            <label>
              <span class="field-label">Seeded players <span class="optional">0 = skill-scaled default</span> <span class="help-tip" tabindex="0" title="How many synthetic players to rate. Zero sizes the pool from the active skill registry (currently 64 for 8 skills). More players improve skill-model coverage but require a larger schedule.">?</span></span>
              <input v-model.number="config.poolSize" type="number" min="0" />
            </label>
          </div>
          <label>
            <span class="field-label">Pool seed <span class="help-tip" tabindex="0" title="Random seed used only when creating a pool. Reusing the same seed and configuration reproduces the same player profiles; change it for an independent replicate.">?</span></span>
            <input v-model.number="config.poolSeed" type="number" />
          </label>
          <label class="toggle"><input v-model="config.createSchedule" type="checkbox" /> Create a new balanced schedule <span class="help-tip" tabindex="0" title="The schedule chooses a distinct, balanced 5v5 lineup for every game, balances which team starts, and gives each game independent RNG streams. Turn this off to reuse an existing immutable schedule.">?</span></label>
          <div class="two-column">
            <label>
              <span class="field-label">Schedule path <span class="help-tip" tabindex="0" title="Where the immutable balanced-lineup schedule is written or loaded from. The schedule must match the chosen player pool.">?</span></span>
              <input v-model="config.schedulePath" required />
            </label>
            <label>
              <span class="field-label">Games <span class="optional">0 = skill-scaled default</span> <span class="help-tip" tabindex="0" title="Number of distinct balanced games. Zero chooses the skill-scaled default (currently 640 games for 64 players and 8 skills). More games narrow uncertainty intervals and take longer.">?</span></span>
              <input v-model.number="config.gameCount" type="number" min="0" />
            </label>
          </div>
          <label>
            <span class="field-label">Schedule seed <span class="help-tip" tabindex="0" title="Random seed used only when creating the balanced lineup schedule. Keep it to reproduce the same set of matchups; change it for an independent schedule.">?</span></span>
            <input v-model.number="config.scheduleSeed" type="number" />
          </label>
        </fieldset>

        <fieldset>
          <legend>Evaluation and analysis</legend>
          <div class="three-column">
            <label>
              <span class="field-label">Action mode <span class="help-tip" tabindex="0" title="Direct uses each policy's highest-probability action and is best for stable ratings. Sampled draws from action probabilities and measures a noisier behavioral distribution. Results from the two modes remain separate studies.">?</span></span>
              <select v-model="config.actionMode"><option value="direct">Direct (argmax)</option><option value="sampled">Sampled</option></select>
            </label>
            <label>
              <span class="field-label">Games per batch <span class="help-tip" tabindex="0" title="Number of games evaluated together in a compiled JAX batch. It does not change the study design. Smaller values provide heartbeats sooner and use less memory; larger values can improve throughput after compilation.">?</span></span>
              <input v-model.number="config.batchSize" type="number" min="1" />
            </label>
            <label>
              <span class="field-label">Evaluation seed <span class="help-tip" tabindex="0" title="Controls stochastic environment/evaluation draws. Keep it fixed to reproduce an evaluation; change it when running a new independent replicate.">?</span></span>
              <input v-model.number="config.evalSeed" type="number" />
            </label>
          </div>
          <div class="three-column">
            <div class="analysis-method">
              <span class="field-label">Ridge regularization <span class="help-tip" tabindex="0" title="Each ridge model selects its own lambda with deterministic grouped five-fold cross-validation inside the training games. Held-out games remain untouched for reporting.">?</span></span>
              <strong>Automatic 5-fold CV</strong>
            </div>
            <label>
              <span class="field-label">Validation fraction <span class="help-tip" tabindex="0" title="Fraction of games held out before fitting, used to report honest out-of-sample diagnostics. Default 0.20 holds out 20%; increasing it gives a larger check but less training data.">?</span></span>
              <input v-model.number="config.validationFraction" type="number" min="0" max="1" step="0.05" />
            </label>
            <label>
              <span class="field-label">Bootstrap samples <span class="help-tip" tabindex="0" title="Number of game-level resamples used for player-rating uncertainty intervals. More samples make interval estimates smoother but only add fitting time, not new games.">?</span></span>
              <input v-model.number="config.bootstrapSamples" type="number" min="1" />
            </label>
          </div>
          <div class="two-column">
            <label>
              <span class="field-label">Split seed <span class="help-tip" tabindex="0" title="Controls which games enter the training versus validation split. Use the same value for a reproducible comparison.">?</span></span>
              <input v-model.number="config.splitSeed" type="number" />
            </label>
            <label>
              <span class="field-label">Bootstrap seed <span class="help-tip" tabindex="0" title="Controls the resampling used to calculate uncertainty intervals. It affects interval repeatability, not the underlying games.">?</span></span>
              <input v-model.number="config.bootstrapSeed" type="number" />
            </label>
          </div>
          <label>
            <span class="field-label">Artifact destination <span class="help-tip" tabindex="0" title="Parent folder for raw games, immutable inputs, and derived reports from this launch. Choose a fresh folder for a new study so existing evidence cannot be overwritten.">?</span></span>
            <input v-model="config.artifactDir" required />
          </label>
        </fieldset>

        <button class="primary" type="submit" :disabled="isLaunching || isRunning || !canLaunch">
          {{ isLaunching ? 'Validating…' : isRunning ? 'Study running' : 'Launch Skill Ratings study' }}
        </button>
      </form>

      <aside class="side-column">
        <section class="status-card">
          <div class="card-heading">
            <h2>Study status</h2>
            <span :class="['status-badge', job?.status || 'idle']">{{ job?.status || 'idle' }}</span>
          </div>
          <template v-if="job">
            <p class="phase"><strong>{{ formatPhase(latestEvent?.phase) }}</strong><span v-if="latestEvent?.elapsed_sec !== undefined">{{ formatElapsed(latestEvent.elapsed_sec) }} elapsed</span></p>
            <template v-if="batchProgress">
              <div class="progress-label"><span>Games completed</span><strong>{{ batchProgress.completed }} / {{ batchProgress.total }}</strong></div>
              <div class="progress-track"><div class="progress-fill" :style="{ width: `${progressPercent}%` }"></div></div>
              <p class="muted">Compiled batches report progress only after they finish; this is evaluator-reported progress.</p>
              <p v-if="observedGamesPerSecond" class="timing-estimate">
                Observed {{ formatRate(observedGamesPerSecond) }} · about {{ formatElapsed(estimatedGameSecondsRemaining) }} of game simulation remaining
                <span v-if="totalBatchCount">({{ totalBatchCount }} batches total)</span>.
              </p>
              <p v-else-if="latestEvent?.phase === 'batch_started' || latestEvent?.phase === 'batch_running'" class="timing-estimate">
                Calibrating on the first batch. It includes JAX compilation; a liveness heartbeat arrives every 10 seconds until the batch completes.
              </p>
              <p class="muted">The estimate covers simulation batches; model fitting and report writing follow after the final game.</p>
            </template>
            <p v-else class="muted">Waiting for the evaluator’s first phase heartbeat.</p>
            <p v-if="job.error" class="error-message">{{ job.error }}</p>
            <details>
              <summary>Recent evaluator output ({{ job.logs?.length || 0 }})</summary>
              <pre>{{ job.logs?.map((entry) => entry.message).join('\n') }}</pre>
            </details>
          </template>
          <p v-else class="muted">Launching validates the checkpoint’s 5v5 multi-possession compatibility before it starts work.</p>
        </section>

        <section class="reports-card">
          <div class="card-heading">
            <h2>Existing results</h2>
            <button class="text-button" type="button" :disabled="isLoadingReports" @click="refreshReports">{{ isLoadingReports ? 'Loading…' : 'Refresh' }}</button>
          </div>
          <p class="muted">Completed reports are retained in this library. Leave it at the Skill Ratings root to see every study, or enter a narrower artifact folder.</p>
          <label>
            <span class="field-label">Results artifact location <span class="help-tip" tabindex="0" title="A report library or study folder to search. It is separate from the next launch destination, so new study paths never remove or hide prior reports.">?</span></span>
            <input v-model="config.resultsArtifactDir" @change="refreshReports" />
          </label>
          <select v-model="selectedReportPath" @change="loadReport()">
            <option value="">Choose a report…</option>
            <option v-for="item in reports" :key="item.path" :value="item.path">{{ skillRatingsReportOptionLabel(item) }}</option>
          </select>
        </section>
      </aside>
    </div>

    <section v-if="report" class="results-section">
      <header class="results-heading">
        <div>
          <p class="eyebrow">Study results</p>
          <h2>{{ report.summary.study_id || 'Skill Ratings report' }}</h2>
          <p class="muted">{{ report.summary.action_mode }} actions · checkpoint update {{ report.summary.checkpoint_update ?? 'unknown' }}</p>
        </div>
        <div v-if="report.sizing" class="study-size">
          <span>{{ report.sizing.seeded_players }} players</span><span>{{ report.sizing.actual_games }} games</span><span>{{ report.sizing.actual_possessions }} possessions</span>
        </div>
      </header>

      <section class="result-card selected-analysis-card">
        <h3>Selected analysis</h3>
        <dl class="analysis-identity">
          <dt>Revision</dt><dd>{{ selectedAnalysis.label }}</dd>
          <dt>Report path</dt><dd><code>{{ selectedAnalysis.path }}</code></dd>
        </dl>
        <details>
          <summary>Raw study dependencies ({{ selectedAnalysis.dependencies.length }})</summary>
          <dl class="analysis-dependencies">
            <template v-for="dependency in selectedAnalysis.dependencies" :key="dependency.key">
              <dt>{{ dependency.key }}</dt><dd><code>{{ dependency.value }}</code></dd>
            </template>
          </dl>
        </details>
      </section>

      <div class="results-grid">
        <section class="result-card factors-card">
          <h3>Three factors</h3>
          <p class="muted">Conventional OLS scenario weights. Free-throw rate is unavailable because the environment has no fouls or free throws.</p>
          <div v-for="(details, factor) in report.three_factors.scenario_weights?.factor_weights || {}" :key="factor" class="factor-row">
            <span>{{ report.three_factors.scenario_weights?.factor_definitions?.[factor] || factor }}</span>
            <strong>{{ formatPercent(details.percent_of_total_margin_delta) }}</strong>
          </div>
          <p v-for="(details, factor) in report.three_factors.unavailable_factors || {}" :key="factor" class="muted">{{ factor }}: {{ details.reason }}</p>
        </section>
        <section class="result-card">
          <h3>Model diagnostics</h3>
          <dl class="diagnostics">
            <template v-if="report.player_rating_fit"><dt>Skill model λ</dt><dd>{{ formatNumber(report.player_rating_fit.ridge_alpha, 3) }} · {{ report.player_rating_fit.ridge_selection?.fold_count || '—' }}-fold CV</dd><dt>Skill-model fit R²</dt><dd>{{ formatNumber(report.player_rating_fit.r_squared, 3) }}</dd><dt>Skill-model RMSE</dt><dd>{{ formatNumber(report.player_rating_fit.rmse, 2) }}</dd></template>
            <template v-if="report.player_rating_validation"><dt>Held-out skill R²</dt><dd>{{ formatNumber(report.player_rating_validation.r_squared, 3) }}</dd><dt>Held-out skill RMSE</dt><dd>{{ formatNumber(report.player_rating_validation.rmse, 2) }}</dd></template>
            <template v-if="report.apm_diagnostic_fit"><dt>APM diagnostic R²</dt><dd>{{ formatNumber(report.apm_diagnostic_fit.r_squared, 3) }}</dd></template>
            <template v-if="report.skill_apm_agreement"><dt title="Pearson correlation of direct skill ratings and adjusted plus/minus across all comparable players.">Skill/APM rating correlation</dt><dd>{{ formatNumber(report.skill_apm_agreement.rating_pearson_correlation, 3) }}</dd><dt title="Spearman correlation of the two player orderings. This measures agreement about who ranks above whom.">Skill/APM rank correlation</dt><dd>{{ formatNumber(report.skill_apm_agreement.ranking_spearman_correlation, 3) }}</dd><dt>Top-{{ report.skill_apm_agreement.overlap_k }} overlap</dt><dd>{{ report.skill_apm_agreement.top_k_overlap_count }} of {{ report.skill_apm_agreement.overlap_k }}</dd><dt>Bottom-{{ report.skill_apm_agreement.overlap_k }} overlap</dt><dd>{{ report.skill_apm_agreement.bottom_k_overlap_count }} of {{ report.skill_apm_agreement.overlap_k }}</dd></template>
          </dl>
        </section>
      </div>

      <section v-if="report.reproducibility" class="result-card reproducibility-card">
        <div class="card-heading">
          <div><h3>Reproduce or extend this study</h3><p class="muted">The pool and game schedule are immutable evidence. Extending evaluates only newly appended games, then fits one combined report.</p></div>
          <button class="secondary" type="button" @click="copyReportConfig">Copy config</button>
        </div>
        <p v-if="copyNotice" class="copy-notice">{{ copyNotice }}</p>
        <p v-if="report.reproducibility.missing_seed_keys?.length" class="error-message">
          This legacy report did not record {{ report.reproducibility.missing_seed_keys.join(', ').replaceAll('_', ' ') }}. It cannot be safely extended; launch a new study instead.
        </p>
        <dl class="reproducibility-seeds">
          <dt>Player pool seed</dt><dd>{{ formatSeed(report.reproducibility.seeds?.pool_seed) }}</dd>
          <dt>Original schedule seed</dt><dd>{{ formatSeed(report.reproducibility.seeds?.schedule_seed) }}</dd>
          <dt>Evaluation seed</dt><dd>{{ formatSeed(report.reproducibility.seeds?.evaluation_seed) }}</dd>
          <dt>Split seed</dt><dd>{{ formatSeed(report.reproducibility.seeds?.split_seed) }}</dd>
          <dt>Bootstrap seed</dt><dd>{{ formatSeed(report.reproducibility.seeds?.bootstrap_seed) }}</dd>
        </dl>
        <div class="extension-controls">
          <label>Additional games<input v-model="additionalGames" type="number" min="1" step="1" /></label>
          <label>New schedule seed<input v-model="extensionScheduleSeed" type="number" step="1" /></label>
          <button class="primary" type="button" :disabled="isRunning || isExtending || report.reproducibility.missing_seed_keys?.length" @click="extendReport">{{ isExtending ? 'Starting extension…' : 'Extend study' }}</button>
        </div>
      </section>

      <section class="result-card table-card">
        <h3>Player skill ratings and APM comparison</h3>
        <div class="table-scroll">
          <table>
            <thead><tr><th>Rank</th><th>Player</th><th>Skill rating</th><th>90% interval</th><th>APM check</th><th>APM − skill</th><th v-for="skill in report.skill_descriptors" :key="skill.key" :title="skill.semantic_group">{{ skill.display_label || skill.key }}</th></tr></thead>
            <tbody>
              <tr v-for="player in report.player_rows" :key="player.player_id">
                <td>{{ player.rank }}</td><td>{{ player.name }}</td><td>{{ formatNumber(player.net_rating) }}</td><td>{{ formatNumber(player.interval_p05) }} to {{ formatNumber(player.interval_p95) }}</td><td>{{ formatNumber(player.apm_rating) }}</td><td>{{ formatNumber(player.apm_minus_skill_rating) }}</td>
                <td v-for="skill in report.skill_descriptors" :key="skill.key">{{ formatNumber(player.skills.find((value) => value.key === skill.key)?.value) }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      <section class="result-card table-card">
        <h3>Direct skill effects</h3>
        <div class="table-scroll">
          <table>
            <thead><tr><th>Skill</th><th>Group</th><th>Net rating / pool SD</th><th>Pool mean</th><th>Pool SD</th></tr></thead>
            <tbody><tr v-for="skill in report.skill_coefficients" :key="skill.key"><td>{{ skill.label }}</td><td>{{ skill.group }}</td><td>{{ formatNumber(skill.net_rating_per_standard_deviation) }}</td><td>{{ formatNumber(skill.pool_mean) }}</td><td>{{ formatNumber(skill.pool_standard_deviation) }}</td></tr></tbody>
          </table>
        </div>
      </section>
    </section>
  </section>
</template>

<style scoped>
.skill-ratings-workbench { width: 100%; min-width: 0; color: #e9eef8; }
.workbench-header, .results-heading, .card-heading, .progress-label { display: flex; align-items: center; justify-content: space-between; gap: 1rem; }
.workbench-header { margin-bottom: 1.25rem; }
h1, h2, h3, p { margin-top: 0; } h1 { margin-bottom: .25rem; font-size: 2rem; } h2 { margin-bottom: .2rem; } h3 { margin-bottom: .75rem; }
.eyebrow { margin-bottom: .25rem; color: #8da9d9; font-size: .75rem; font-weight: 800; letter-spacing: .11em; text-transform: uppercase; }
.intro { max-width: 760px; color: #b8c6dc; } .muted, small, .optional { color: #9caec9; font-size: .83rem; } .optional { font-weight: normal; }
.workbench-grid { display: grid; grid-template-columns: minmax(0, 1.8fr) minmax(280px, .8fr); gap: 1rem; align-items: start; }
.status-card, .reports-card, .result-card { border: 1px solid #314662; border-radius: 12px; background: #101a2b; box-shadow: 0 8px 24px rgba(0,0,0,.18); }
/* The workbench already lives inside the dev-tab surface. Keep its launch
   form flat instead of nesting a card, a guide card, and bordered fieldsets. */
.config-card { padding: 0; background: transparent; } .side-column { display: grid; gap: 1rem; } .status-card, .reports-card, .result-card { padding: 1rem; }
.settings-guide { margin: 0 0 .75rem; border: 0; border-left: 3px solid #3b587d; padding: .25rem 0 .25rem .7rem; background: transparent; color: #c3d4ed; font-size: .85rem; }.settings-guide summary { cursor: pointer; color: #d9e7ff; font-weight: 800; }.settings-guide p { margin: .65rem 0 0; line-height: 1.45; }.settings-guide strong { color: #f0f6ff; }
fieldset { margin: 0; padding: .85rem 0; border: 0; border-top: 1px solid #304765; border-radius: 0; } fieldset:first-of-type { border-top: 0; padding-top: .25rem; } legend { padding: 0; color: #bcd2f8; font-weight: 700; }
label { display: grid; gap: .3rem; margin-bottom: .7rem; color: #cbd9ef; font-size: .88rem; } label:last-child { margin-bottom: 0; } .toggle { display: flex; align-items: center; gap: .5rem; } .toggle input { width: auto; }.field-label { display: inline-flex; align-items: center; flex-wrap: wrap; gap: .25rem; }.help-tip { display: inline-grid; place-items: center; width: 1.05rem; height: 1.05rem; border: 1px solid #7192be; border-radius: 50%; color: #d9eaff; cursor: help; font-size: .72rem; font-weight: 800; line-height: 1; }.help-tip:focus { outline: 2px solid #70b7ff; outline-offset: 2px; }
input, textarea, select { width: 100%; box-sizing: border-box; border: 1px solid #415c80; border-radius: 6px; padding: .55rem .65rem; background: #0a1220; color: #edf3ff; font: inherit; } textarea { resize: vertical; }
.two-column, .three-column { display: grid; gap: .7rem; } .two-column { grid-template-columns: repeat(2, minmax(0, 1fr)); } .three-column { grid-template-columns: repeat(3, minmax(0, 1fr)); }.analysis-method { display: grid; align-content: start; gap: .35rem; margin-bottom: .7rem; color: #cbd9ef; font-size: .88rem; }.analysis-method strong { color: #e8f1ff; font-size: .92rem; }
button { border-radius: 7px; padding: .6rem .85rem; border: 1px solid #5678a9; color: #f3f7ff; background: #1b3152; cursor: pointer; font-weight: 700; } button:hover:not(:disabled) { background: #284873; } button:disabled { cursor: not-allowed; opacity: .55; } .primary { width: 100%; background: #1e6c50; border-color: #3ca777; } .primary:hover:not(:disabled) { background: #27855f; } .secondary { white-space: nowrap; } .text-button { padding: .25rem .45rem; font-size: .8rem; }
.status-badge { border-radius: 999px; padding: .2rem .55rem; background: #2b3850; color: #d0ddf5; font-size: .75rem; font-weight: 800; text-transform: capitalize; }.status-badge.running { background: #155d45; color: #b5f4d7; }.status-badge.completed { background: #285887; color: #d6ebff; }.status-badge.failed { background: #7a303c; color: #ffe2e5; }
.phase { display: flex; align-items: center; justify-content: space-between; gap: .5rem; margin-bottom: .6rem; text-transform: capitalize; }.phase span { color: #9fb2d0; font-size: .82rem; }
.progress-track { overflow: hidden; height: .55rem; border-radius: 999px; background: #21324a; }.progress-fill { height: 100%; background: #48bb88; transition: width .25s ease; }.progress-label { margin: .75rem 0 .35rem; font-size: .85rem; }.timing-estimate { margin: .65rem 0 .25rem; color: #cde3ff; font-size: .84rem; line-height: 1.4; }
details { margin-top: .85rem; } summary { cursor: pointer; color: #bcd2f8; } pre { max-height: 250px; overflow: auto; white-space: pre-wrap; color: #bfcde3; font-size: .73rem; }
.error-message { border: 1px solid #8a3945; border-radius: 7px; padding: .65rem; background: #3d1d25; color: #ffdbe0; }
.results-section { margin-top: 1rem; }.study-size { display: flex; flex-wrap: wrap; gap: .45rem; }.study-size span { border-radius: 999px; padding: .35rem .6rem; background: #203653; color: #cde3ff; font-size: .8rem; }.selected-analysis-card { margin-top: 1rem; }.analysis-identity, .analysis-dependencies { display: grid; grid-template-columns: auto minmax(0, 1fr); gap: .45rem .9rem; margin: 0; }.analysis-identity dt, .analysis-dependencies dt { color: #aebfd9; }.analysis-identity dd, .analysis-dependencies dd { min-width: 0; margin: 0; }.analysis-identity code, .analysis-dependencies code { overflow-wrap: anywhere; white-space: normal; }.results-grid { display: grid; grid-template-columns: minmax(0, 1.15fr) minmax(260px, .85fr); gap: 1rem; margin: 1rem 0; }.factor-row { display: flex; justify-content: space-between; gap: 1rem; border-top: 1px solid #283d59; padding: .6rem 0; font-size: .9rem; }.factor-row span { color: #c6d4e9; }.diagnostics, .reproducibility-seeds { display: grid; grid-template-columns: 1fr auto; gap: .45rem .9rem; margin: 0; }.diagnostics dt, .reproducibility-seeds dt { color: #aebfd9; }.diagnostics dd, .reproducibility-seeds dd { margin: 0; font-weight: 700; }.reproducibility-card { margin-top: 1rem; }.reproducibility-card h3 { margin-bottom: .25rem; }.copy-notice { margin: .75rem 0 0; color: #9be7bd; font-size: .84rem; }.extension-controls { display: grid; grid-template-columns: minmax(150px, .55fr) minmax(150px, .55fr) auto; align-items: end; gap: .7rem; margin-top: 1rem; padding-top: 1rem; border-top: 1px solid #283d59; }.extension-controls label { margin: 0; }.extension-controls .primary { width: auto; white-space: nowrap; }.table-card { margin-top: 1rem; }.table-scroll { overflow: auto; } table { width: 100%; border-collapse: collapse; font-size: .83rem; } th, td { padding: .55rem .65rem; border-bottom: 1px solid #283d59; text-align: left; vertical-align: top; white-space: nowrap; } th { position: sticky; top: 0; background: #14223a; color: #c8dcfc; }
@media (max-width: 900px) { .workbench-grid, .results-grid, .extension-controls { grid-template-columns: 1fr; }.three-column { grid-template-columns: 1fr; } }
@media (max-width: 600px) { .workbench-header, .results-heading { align-items: flex-start; flex-direction: column; }.two-column { grid-template-columns: 1fr; } }
</style>
