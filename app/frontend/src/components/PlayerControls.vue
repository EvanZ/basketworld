<script setup>
import { ref, computed, watch, onMounted, nextTick } from 'vue';
import { defineExpose } from 'vue';
import HexagonControlPad from './HexagonControlPad.vue';
import { getActionValues, getRewards, mctsAdvise, setOffenseSkills, setPassTargetStrategy, setPassLogitBias, setBallHolder } from '@/services/api';
import { loadStats, saveStats, resetStatsStorage } from '@/services/stats';

// Import API_BASE_URL for policy probabilities fetch
const API_BASE_URL = import.meta.env?.VITE_API_BASE_URL || 'http://localhost:8080';

function formatParamCount(n) {
  if (n === null || n === undefined) return 'N/A';
  const num = Number(n);
  if (Number.isNaN(num)) return 'N/A';
  if (num >= 1_000_000) return `${(num / 1_000_000).toFixed(2)}M`;
  if (num >= 1_000) return `${(num / 1_000).toFixed(1)}k`;
  return String(num);
}

const props = defineProps({
  gameState: Object,
  activePlayerId: Number,
  disabled: {
    type: Boolean,
    default: false,
  },
  isReplaying: {
    type: Boolean,
    default: false,
  },
  isManualStepping: {
    type: Boolean,
    default: false,
  },
  isEvaluating: {
    type: Boolean,
    default: false,
  },
  aiMode: {
    type: Boolean,
    default: false,
  },
  deterministic: {
    type: Boolean,
    default: true,
  },
  opponentDeterministic: {
    type: Boolean,
    default: true,
  },
  moveHistory: {
    type: Array,
    default: () => [],
  },
  // Current shot clock value (for highlighting in moves table)
  currentShotClock: {
    type: Number,
    default: null,
    required: false,
  },
  // Cumulative defender pressure exposure for the current episode
  pressureExposure: {
    type: Number,
    default: 0,
  },
  // When provided, overrides internal selections to reflect actual applied actions
  externalSelections: {
    type: Object,
    default: null,
  },
  // When provided (during manual stepping), use these stored policy probs instead of fetching
  storedPolicyProbs: {
    type: Object,
    default: null,
  },
  policyOptions: {
    type: Array,
    default: () => [],
  },
  policiesLoading: {
    type: Boolean,
    default: false,
  },
  policyLoadError: {
    type: String,
    default: null,
  },
  isPolicySwapping: {
    type: Boolean,
    default: false,
  },
  initialUseMcts: {
    type: Boolean,
    default: false,
  },
  mctsStepRunning: {
    type: Boolean,
    default: false,
  },
  // MCTS results returned from the last step (when backend ran MCTS)
  mctsResults: {
    type: Object,
    default: null,
  },
  evalConfig: {
    type: Object,
    default: null,
  },
  evalNumEpisodes: {
    type: Number,
    default: 100,
  },
  perPlayerEvalStats: {
    type: Object,
    default: null,
  },
});

const emit = defineEmits(['actions-submitted', 'update:activePlayerId', 'move-recorded', 'policy-swap-requested', 'selections-changed', 'refresh-policies', 'mcts-options-changed', 'mcts-toggle-changed', 'state-updated', 'eval-config-changed', 'eval-run', 'active-tab-changed', 'ball-holder-updating', 'ball-holder-changed']);

const selectedActions = ref({});

const paramCounts = computed(() => props.gameState?.training_params?.param_counts || null);
const movesColumnCount = computed(() => (allPlayerIds.value?.length || 0) + 4);
const pressureExposureDisplay = computed(() => {
  const val = Number(props.pressureExposure);
  return Number.isFinite(val) ? val.toFixed(3) : '0.000';
});

// Debug: Watch for any changes to selectedActions
watch(selectedActions, (newActions, oldActions) => {
  console.log('[PlayerControls] 🔍 selectedActions changed from:', oldActions, 'to:', newActions);
  emit('selections-changed', { ...newActions });
}, { deep: true });

const actionValues = ref(null);
const valueRange = ref({ min: 0, max: 0 });

function applyStoredActionValues(storedValues) {
  if (!storedValues) {
    actionValues.value = null;
    valueRange.value = { min: 0, max: 0 };
    return;
  }

  const converted = {};
  const numericValues = [];

  Object.entries(storedValues).forEach(([playerId, actionDict]) => {
    const playerValues = {};
    Object.entries(actionDict || {}).forEach(([actionName, value]) => {
      const numeric = Number(value);
      if (!Number.isNaN(numeric)) {
        playerValues[actionName] = numeric;
        numericValues.push(numeric);
      }
    });
    converted[playerId] = playerValues;
  });

  actionValues.value = converted;
  if (numericValues.length > 0) {
    valueRange.value = {
      min: Math.min(...numericValues),
      max: Math.max(...numericValues),
    };
  } else {
    valueRange.value = { min: 0, max: 0 };
  }
}
// shot probability is displayed on the board, not in controls
const policyProbabilities = ref(null);

const availablePolicies = computed(() => props.policyOptions || []);
const policiesLoading = computed(() => props.policiesLoading);
const policyLoadError = computed(() => props.policyLoadError);
const userPolicySelection = ref('');
const opponentPolicySelection = ref('');

watch(
  () => props.gameState?.unified_policy_name,
  (name) => {
    userPolicySelection.value = name || '';
  },
  { immediate: true }
);

watch(
  () => props.gameState?.opponent_unified_policy_name,
  (name) => {
    opponentPolicySelection.value = name || '';
  },
  { immediate: true }
);

function handlePolicySelection(type, event) {
  if (props.isPolicySwapping || props.policiesLoading) return;
  const value = event?.target?.value ?? '';
  if (type === 'user') {
    if (value === (userPolicySelection.value || '')) return;
    if (!value) return;
    userPolicySelection.value = value;
    emit('policy-swap-requested', { target: 'user', policyName: value });
    return;
  }

  if (value === (opponentPolicySelection.value || '')) return;
  opponentPolicySelection.value = value;
  emit('policy-swap-requested', { target: 'opponent', policyName: value });
}

// Offense skill overrides (Environment tab)
const offenseSkillInputs = ref({ layup: [], three_pt: [], dunk: [] });
const offenseSkillSampled = ref({ layup: [], three_pt: [], dunk: [] });
const skillsUpdating = ref(false);
const skillsError = ref(null);
const passStrategyUpdating = ref(false);
const passStrategyError = ref(null);
const passLogitBiasInput = ref(0);
const passLogitBiasDefault = 0.0;
const passLogitBiasUpdating = ref(false);
const passLogitBiasError = ref(null);
const PASS_TARGET_STRATEGIES = [
  { value: 'nearest', label: 'Nearest (legacy)' },
  { value: 'best_ev', label: 'Best EV' },
];

const passTargetStrategyValue = computed(() => {
  const val = props.gameState?.pass_target_strategy || 'nearest';
  return String(val).toLowerCase();
});

function percentFromProb(prob) {
  const num = Number(prob ?? 0);
  if (Number.isNaN(num)) return 0;
  return Math.max(0, Math.min(100, Number((num * 100).toFixed(1))));
}

function fillOffenseSkills(targetRef, sourceSkills) {
  const count = props.gameState?.offense_ids?.length || 0;
  const safe = sourceSkills || {};
  targetRef.value = {
    layup: Array.from({ length: count }, (_, idx) => percentFromProb(safe?.layup?.[idx] ?? 0)),
    three_pt: Array.from({ length: count }, (_, idx) => percentFromProb(safe?.three_pt?.[idx] ?? 0)),
    dunk: Array.from({ length: count }, (_, idx) => percentFromProb(safe?.dunk?.[idx] ?? 0)),
  };
}

watch(() => props.gameState?.offense_shooting_pct_by_player, (skills) => {
  fillOffenseSkills(offenseSkillInputs, skills);
}, { immediate: true, deep: true });

watch(() => props.gameState?.offense_shooting_pct_sampled, (skills) => {
  fillOffenseSkills(
    offenseSkillSampled,
    skills || props.gameState?.offense_shooting_pct_by_player || null
  );
}, { immediate: true, deep: true });

watch(() => props.gameState?.offense_shooting_pct_by_player, (skills) => {
  if (!props.gameState?.offense_shooting_pct_sampled) {
    fillOffenseSkills(offenseSkillSampled, skills);
  }
}, { deep: true });

watch(() => props.gameState?.pass_logit_bias, (val) => {
  if (val === null || val === undefined) return;
  if (!passLogitBiasUpdating.value) {
    const num = Number(val);
    passLogitBiasInput.value = Number.isFinite(num) ? num : 0;
  }
}, { immediate: true });

const offenseSkillRows = computed(() => {
  const ids = props.gameState?.offense_ids || [];
  const lay = offenseSkillInputs.value?.layup || [];
  const three = offenseSkillInputs.value?.three_pt || [];
  const dunk = offenseSkillInputs.value?.dunk || [];
  const sampleLay = offenseSkillSampled.value?.layup || [];
  const sampleThree = offenseSkillSampled.value?.three_pt || [];
  const sampleDunk = offenseSkillSampled.value?.dunk || [];
  return ids.map((pid, idx) => ({
    playerId: pid,
    layup: Number(lay[idx] ?? 0),
    threePt: Number(three[idx] ?? 0),
    dunk: Number(dunk[idx] ?? 0),
    sampledLayup: Number(sampleLay[idx] ?? lay[idx] ?? 0),
    sampledThree: Number(sampleThree[idx] ?? three[idx] ?? 0),
    sampledDunk: Number(sampleDunk[idx] ?? dunk[idx] ?? 0),
  }));
});

function percentToProb(percent) {
  const num = Number(percent ?? 0);
  if (Number.isNaN(num)) return 0.01;
  const clamped = Math.max(1, Math.min(99, num));
  return clamped / 100;
}

const defaultEvalConfig = () => ({
  mode: 'default',
  placementEditing: false,
  positions: [],
  ballHolder: null,
  shootingMode: 'random',
  skills: { layup: [], three_pt: [], dunk: [] },
  randomizeOffensePermutation: false,
});

const evalConfigSafe = computed(() => {
  const base = defaultEvalConfig();
  const incoming = props.evalConfig || {};
  return {
    ...base,
    ...incoming,
    skills: { ...base.skills, ...(incoming.skills || {}) },
  };
});

const evalPlacementEditing = computed(() => evalConfigSafe.value.placementEditing && evalConfigSafe.value.mode === 'custom');
const evalModeIsCustom = computed(() => evalConfigSafe.value.mode === 'custom');
const evalOffenseIds = computed(() => props.gameState?.offense_ids || []);
const ballStartOptions = computed(() => {
  const ids = evalOffenseIds.value || [];
  return Array.isArray(ids) ? [...ids] : [];
});

const evalEpisodesInput = ref(props.evalNumEpisodes || 100);
watch(() => props.evalNumEpisodes, (val) => {
  const safe = Number.isFinite(val) ? Number(val) : 100;
  evalEpisodesInput.value = safe;
});

function updateEvalEpisodes(val) {
  const safe = Math.max(1, Number(val) || 1);
  evalEpisodesInput.value = safe;
}

function emitEvalConfigUpdate(patch = {}) {
  const base = evalConfigSafe.value;
  const next = {
    ...base,
    ...(patch || {}),
    skills: {
      ...base.skills,
      ...(patch.skills || {}),
    },
  };
  if (next.mode !== 'custom') {
    next.placementEditing = false;
  }
  emit('eval-config-changed', next);
}

function seedEvalConfigFromGameState(copySkills = true) {
  if (!props.gameState) return;
  const positions = (props.gameState.positions || []).map((pos) => [pos[0], pos[1]]);
  const offenseCount = props.gameState.offense_ids?.length || 0;
  const bhDefault = evalOffenseIds.value.length ? evalOffenseIds.value[0] : null;
  const sourceSkills =
    props.gameState.offense_shooting_pct_sampled ||
    props.gameState.offense_shooting_pct_by_player ||
    null;
  const skills = copySkills ? { layup: [], three_pt: [], dunk: [] } : (evalConfigSafe.value.skills || {});
  if (copySkills && sourceSkills) {
    skills.layup = Array.from({ length: offenseCount }, (_, idx) => percentFromProb(sourceSkills?.layup?.[idx]));
    skills.three_pt = Array.from({ length: offenseCount }, (_, idx) => percentFromProb(sourceSkills?.three_pt?.[idx]));
    skills.dunk = Array.from({ length: offenseCount }, (_, idx) => percentFromProb(sourceSkills?.dunk?.[idx]));
  }
  const nextBallHolder = (bhDefault !== null && evalOffenseIds.value.includes(props.gameState.ball_holder))
    ? props.gameState.ball_holder
    : bhDefault;

  const patch = {
    positions,
    ballHolder: nextBallHolder,
    mode: evalConfigSafe.value.mode,
    skills: copySkills ? skills : (evalConfigSafe.value.skills || skills),
  };
  emitEvalConfigUpdate(patch);
}

function toggleEvalPlacement(val) {
  if (val && (!evalConfigSafe.value.positions || evalConfigSafe.value.positions.length === 0)) {
    seedEvalConfigFromGameState(false);
  }
  emitEvalConfigUpdate({ placementEditing: val, mode: 'custom' });
  emit('active-tab-changed', 'eval');
}

function setEvalBallHolder(val) {
  const num = val === null || val === '' ? null : Number(val);
  if (num !== null && !evalOffenseIds.value.includes(num)) return;
  emitEvalConfigUpdate({ ballHolder: num });
}

function setEvalShootingMode(mode) {
  if (mode === 'fixed') {
    // If no skills are populated yet, seed from game state for convenience
    const skills = evalConfigSafe.value.skills || {};
    const hasSkills =
      (skills.layup && skills.layup.length > 0) ||
      (skills.three_pt && skills.three_pt.length > 0) ||
      (skills.dunk && skills.dunk.length > 0);
    if (!hasSkills) {
      seedEvalConfigFromGameState(true);
    }
  }
  emitEvalConfigUpdate({ shootingMode: mode, mode: 'custom' });
}

function setEvalRandomizePermutation(val) {
  emitEvalConfigUpdate({ randomizeOffensePermutation: !!val });
}

function updateEvalSkill(idx, key, value) {
  const offenseCount = props.gameState?.offense_ids?.length || 0;
  const base = evalConfigSafe.value.skills || {};
  const nextSkills = {
    ...base,
    [key]: Array.from({ length: offenseCount }, (_, i) => {
      if (i === idx) {
        return Number(value) || 0;
      }
      return base[key]?.[i] ?? 0;
    }),
  };
  emitEvalConfigUpdate({ skills: nextSkills });
}

function handleEvalRunClick() {
  emit('eval-run', {
    numEpisodes: evalEpisodesInput.value,
    config: evalConfigSafe.value,
  });
}

function handleEvalModeChange(mode) {
  if (mode === 'custom') {
    if (!evalPlacementEditing.value) {
      toggleEvalPlacement(true);
    }
    if (!evalConfigSafe.value.positions || evalConfigSafe.value.positions.length === 0) {
      seedEvalConfigFromGameState(true);
    }
    // Default to fixed skills when entering custom mode so controls appear
    const nextMode = evalConfigSafe.value.shootingMode || 'random';
    const patch = { mode: 'custom' };
    if (nextMode === 'random') {
      patch.shootingMode = 'fixed';
    }
    emitEvalConfigUpdate(patch);
  } else {
    emitEvalConfigUpdate({ mode: 'default', placementEditing: false });
  }
  emit('active-tab-changed', mode === 'custom' ? 'eval' : 'controls');
}

const evalOffenseSkillRows = computed(() => {
  const ids = props.gameState?.offense_ids || [];
  const skills = evalConfigSafe.value.skills || {};
  const lay = skills.layup || [];
  const three = skills.three_pt || [];
  const dunk = skills.dunk || [];
  return ids.map((pid, idx) => ({
    playerId: pid,
    layup: Number(lay[idx] ?? 0),
    threePt: Number(three[idx] ?? 0),
    dunk: Number(dunk[idx] ?? 0),
  }));
});

async function applyOffenseSkillOverrides() {
  if (!props.gameState) return;
  skillsUpdating.value = true;
  skillsError.value = null;
  try {
    const payload = {
      skills: {
        layup: (offenseSkillInputs.value?.layup || []).map(percentToProb),
        three_pt: (offenseSkillInputs.value?.three_pt || []).map(percentToProb),
        dunk: (offenseSkillInputs.value?.dunk || []).map(percentToProb),
      },
    };
    const res = await setOffenseSkills(payload);
    if (res?.status === 'success' && res.state) {
      emit('state-updated', res.state);
    } else {
      throw new Error(res?.detail || 'Failed to update skills');
    }
  } catch (err) {
    console.error('[PlayerControls] Failed to update offense skills', err);
    skillsError.value = err?.message || 'Failed to update skills';
  } finally {
    skillsUpdating.value = false;
  }
}

async function resetOffenseSkillsToSampled() {
  if (!props.gameState) return;
  skillsUpdating.value = true;
  skillsError.value = null;
  try {
    const res = await setOffenseSkills({ reset_to_sampled: true });
    if (res?.status === 'success' && res.state) {
      emit('state-updated', res.state);
    } else {
      throw new Error(res?.detail || 'Failed to reset skills');
    }
  } catch (err) {
    console.error('[PlayerControls] Failed to reset offense skills', err);
    skillsError.value = err?.message || 'Failed to reset skills';
  } finally {
    skillsUpdating.value = false;
  }
}

async function handlePassTargetStrategyChange(event) {
  if (!props.gameState) return;
  const value = String(event?.target?.value || '').toLowerCase();
  if (!value || value === passTargetStrategyValue.value) return;
  passStrategyUpdating.value = true;
  passStrategyError.value = null;
  try {
    const res = await setPassTargetStrategy(value);
    if (res?.status === 'success' && res.state) {
      emit('state-updated', res.state);
    } else {
      throw new Error(res?.detail || 'Failed to update pass target strategy');
    }
  } catch (err) {
    console.error('[PlayerControls] Failed to update pass target strategy', err);
    passStrategyError.value = err?.message || 'Failed to update pass target strategy';
  } finally {
    passStrategyUpdating.value = false;
  }
}

function normalizePassLogitBias(value) {
  const num = Number(value);
  return Number.isFinite(num) ? num : 0.0;
}

async function applyPassLogitBiasOverride() {
  if (!props.gameState) return;
  passLogitBiasUpdating.value = true;
  passLogitBiasError.value = null;
  try {
    const bias = normalizePassLogitBias(passLogitBiasInput.value);
    const res = await setPassLogitBias(bias);
    if (res?.status === 'success' && res.state) {
      emit('state-updated', res.state);
    } else {
      throw new Error(res?.detail || 'Failed to update pass logit bias');
    }
  } catch (err) {
    console.error('[PlayerControls] Failed to update pass logit bias', err);
    passLogitBiasError.value = err?.message || 'Failed to update pass logit bias';
  } finally {
    passLogitBiasUpdating.value = false;
  }
}

async function resetPassLogitBiasDefault() {
  if (!props.gameState) return;
  passLogitBiasInput.value = passLogitBiasDefault;
  await applyPassLogitBiasOverride();
}

async function handleBallHolderChange(val) {
  if (val === null || val === undefined) return;
  const pid = Number(val);
  if (Number.isNaN(pid) || !offenseIdsLive.value.includes(pid)) {
    return;
  }
  if (!props.gameState || props.isEvaluating || props.isReplaying) return;
  const prevBallHolder = props.gameState?.ball_holder;
  if (prevBallHolder === pid) return;
  ballHolderUpdating.value = true;
  emit('ball-holder-updating', true);
  ballHolderError.value = null;
  try {
    const res = await setBallHolder(pid);
    if (res?.status === 'success' && res.state) {
      emit('state-updated', res.state);
      emit('ball-holder-changed', { from: prevBallHolder, to: pid });
      // Clear any queued selections so we don't treat the manual ball-handler change as an action.
      selectedActions.value = {};
      emit('selections-changed', {});
    } else {
      throw new Error(res?.detail || 'Failed to set ball holder');
    }
  } catch (err) {
    console.error('[PlayerControls] Failed to set ball holder', err);
    ballHolderError.value = err?.message || 'Failed to set ball holder';
  } finally {
    ballHolderUpdating.value = false;
    emit('ball-holder-updating', false);
  }
}

// Add rewards tracking
const activeTab = ref('controls');
const rewardHistory = ref([]);
const episodeRewards = ref({ offense: 0.0, defense: 0.0 });
const rewardParams = ref(null);
const mlflowPhiParams = ref(null);

// Advisor (MCTS) state
const advisorMaxDepth = ref(3);
const advisorTimeBudget = ref(200);
const advisorExplorationC = ref(1.4);
const advisorUsePriors = ref(true);
const advisorResults = ref({}); // playerId -> advisor result
const advisorError = ref(null);
const advisorLoading = ref(false);
const ballHolderUpdating = ref(false);
const ballHolderError = ref(null);
const advisorSelectedPlayerIds = ref([]);
const advisorProgress = ref(0);
const useMctsForStep = ref(!!props.initialUseMcts);
const advisorPolicyTop = computed(() => {
  const topByPlayer = {};
  for (const [pid, res] of Object.entries(advisorResults.value || {})) {
    if (!res || !res.policy) continue;
    const pairs = res.policy.map((p, idx) => ({ idx, prob: p }));
    pairs.sort((a, b) => b.prob - a.prob);
    topByPlayer[pid] = pairs.slice(0, 5);
  }
  return topByPlayer;
});

// Auto-scroll to current shot clock in moves table
const isMounted = ref(false);

watch(() => props.currentShotClock, async (newShotClock) => {
  try {
    if (isMounted.value && newShotClock !== null) {
      await nextTick();
      if (activeTab.value === 'moves') {
        const currentRow = document.querySelector('.current-shot-clock-row');
        if (currentRow) {
          currentRow.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
      } else if (activeTab.value === 'rewards') {
        const currentRow = document.querySelector('.current-reward-row');
        if (currentRow) {
          currentRow.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
      }
    }
  } catch (err) {
    console.warn('Failed to scroll to current shot clock:', err);
  }
}, { flush: 'post' });

// Also scroll when switching tabs
watch(activeTab, async (newTab) => {
  try {
    emit('active-tab-changed', newTab);
    if (isMounted.value && props.currentShotClock !== null) {
      await nextTick();
      if (newTab === 'moves') {
        const currentRow = document.querySelector('.current-shot-clock-row');
        if (currentRow) {
          currentRow.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
      } else if (newTab === 'rewards') {
        const currentRow = document.querySelector('.current-reward-row');
        if (currentRow) {
          currentRow.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
      }
    }
  } catch (err) {
    console.warn('Failed to scroll to current shot clock on tab change:', err);
  }
}, { flush: 'post' });

// Move tracking is now handled by parent component

// --- Stats tracking (persistent across sessions) ---
const statsState = ref(loadStats());
function safeDiv(n, d) { return d > 0 ? (n / d) : 0; }
const totalAssists = computed(() => (statsState.value.dunk.assists + statsState.value.twoPt.assists + statsState.value.threePt.assists));
const totalPotentialAssists = computed(() => (
  statsState.value.dunk.potentialAssists
  + statsState.value.twoPt.potentialAssists
  + statsState.value.threePt.potentialAssists
));
const ppp = computed(() => safeDiv(statsState.value.points, Math.max(1, statsState.value.episodes)));
const avgRewardPerEp = computed(() => safeDiv(statsState.value.rewardSum, Math.max(1, statsState.value.episodes)));
const avgEpisodeLen = computed(() => safeDiv(statsState.value.episodeStepsSum, Math.max(1, statsState.value.episodes)));
const selectedShotChartTarget = ref('team');
const offensePlayerIdsForStats = computed(() => {
  const fromEval = Object.keys(props.perPlayerEvalStats || {}).map((k) => Number(k)).filter((n) => !Number.isNaN(n));
  if (fromEval.length > 0) {
    const offense = props.gameState?.offense_ids || [];
    if (Array.isArray(offense) && offense.length > 0) {
      return fromEval.filter((pid) => offense.includes(pid)).sort((a, b) => a - b);
    }
    return fromEval.sort((a, b) => a - b);
  }
  const offense = props.gameState?.offense_ids || [];
  return Array.isArray(offense) ? offense.map(Number) : [];
});

function aggregatePlayerStats(entries) {
  const base = {
    shots: 0,
    makes: 0,
    assists: 0,
    potential_assists: 0,
    turnovers: 0,
    points: 0,
    shot_types: { dunk: [0, 0], two: [0, 0], three: [0, 0] },
    shot_chart: {},
  };
  entries.forEach((entry) => {
    if (!entry) return;
    base.shots += Number(entry.shots || 0);
    base.makes += Number(entry.makes || 0);
    base.assists += Number(entry.assists || 0);
    base.potential_assists += Number(entry.potential_assists || 0);
    base.turnovers += Number(entry.turnovers || 0);
    base.points += Number(entry.points || 0);
    const st = entry.shot_types || {};
    ['dunk', 'two', 'three'].forEach((k) => {
      const vals = st[k] || [0, 0];
      base.shot_types[k][0] += Number(vals[0] || 0);
      base.shot_types[k][1] += Number(vals[1] || 0);
    });
    const chart = entry.shot_chart || {};
    Object.entries(chart).forEach(([loc, vals]) => {
      const target = base.shot_chart[loc] || [0, 0];
      target[0] += Number((vals || [])[0] || 0);
      target[1] += Number((vals || [])[1] || 0);
      base.shot_chart[loc] = target;
    });
  });
  return base;
}

const selectedEvalStats = computed(() => {
  const stats = props.perPlayerEvalStats || {};
  const offenseIds = offensePlayerIdsForStats.value;
  if (!stats || Object.keys(stats).length === 0 || offenseIds.length === 0) return null;
  if (selectedShotChartTarget.value === 'team') {
    const offenseEntries = offenseIds.map((pid) => stats[pid] || stats[String(pid)]).filter(Boolean);
    return aggregatePlayerStats(offenseEntries);
  }
  const key = selectedShotChartTarget.value;
  return stats[key] || stats[String(key)] || null;
});

const selectedEvalSummary = computed(() => {
  const s = selectedEvalStats.value;
  if (!s) return null;
  const attempts = Number(s.shots || 0);
  const makes = Number(s.makes || 0);
  const fgPct = attempts > 0 ? (makes / attempts) * 100 : 0;
  const turnovers = Number(s.turnovers || 0);
  const points = Number(s.points || 0);
  return {
    attempts,
    makes,
    fgPct,
    turnovers,
    points,
    shotTypes: s.shot_types || { dunk: [0, 0], two: [0, 0], three: [0, 0] },
    shotChart: s.shot_chart || {},
  };
});

const offensePlayerStatsTable = computed(() => {
  const stats = props.perPlayerEvalStats || {};
  const offenseIds = offensePlayerIdsForStats.value;
  if (!stats || offenseIds.length === 0) return [];
  return offenseIds.map((pid) => {
    const entry = stats[pid] || stats[String(pid)] || {};
    const attemptSum = Number(entry.shots || 0);
    const makeSum = Number(entry.makes || 0);
    const shotTypes = entry.shot_types || { dunk: [0, 0], two: [0, 0], three: [0, 0] };
    const getPair = (k) => {
      const vals = shotTypes[k] || [0, 0];
      return { att: Number(vals[0] || 0), mk: Number(vals[1] || 0) };
    };
    const dunk = getPair('dunk');
    const two = getPair('two');
    const three = getPair('three');
    const un = entry.unassisted || {};
    const af = entry.assist_full_by_type || {};
    return {
      playerId: pid,
      attempts: attemptSum,
      makes: makeSum,
      fg: attemptSum > 0 ? (makeSum / attemptSum) * 100 : 0,
      dunk,
      two,
      three,
      assists: Number(entry.assists || 0),
      potentialAssists: Number(entry.potential_assists || 0),
      turnovers: Number(entry.turnovers || 0),
      points: Number(entry.points || 0),
      unassisted: {
        dunk: Math.max(0, dunk.mk - Number(af.dunk || 0)),
        two: Math.max(0, two.mk - Number(af.two || 0)),
        three: Math.max(0, three.mk - Number(af.three || 0)),
      },
    };
  });
});

const CHART_HEX_RADIUS = 12;
function chartAxialToCartesian(q, r, radius = CHART_HEX_RADIUS) {
  const x = radius * (Math.sqrt(3) * q + (Math.sqrt(3) / 2) * r);
  const y = radius * (1.5 * r);
  return { x, y };
}

const shotChartConfig = computed(() => {
  const points = [];
  const stats = selectedEvalSummary.value;
  if (!stats || !stats.shotChart) {
    return { points: [], width: 0, height: 0, radius: CHART_HEX_RADIUS };
  }
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  Object.entries(stats.shotChart).forEach(([loc, vals]) => {
    const [q, r] = loc.split(',').map((v) => Number(v));
    const { x, y } = chartAxialToCartesian(q, r);
    const att = Number((vals || [])[0] || 0);
    const makes = Number((vals || [])[1] || 0);
    points.push({ x, y, attempts: att, makes, ratio: att > 0 ? makes / att : 0 });
    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y);
  });
  if (points.length === 0) {
    return { points: [], width: 0, height: 0, radius: CHART_HEX_RADIUS };
  }
  const margin = CHART_HEX_RADIUS * 2;
  const width = maxX - minX + margin * 2;
  const height = maxY - minY + margin * 2;
  const adjusted = points.map((p) => ({
    ...p,
    x: p.x - minX + margin,
    y: p.y - minY + margin,
  }));
  return { points: adjusted, width, height, radius: CHART_HEX_RADIUS };
});

async function recordEpisodeStats(finalState, skipApiCall = false, episodeData = null) {
  console.log('[Stats] recordEpisodeStats called - current episodes:', statsState.value.episodes);
  const results = finalState?.last_action_results || {};
  // Shot attempt (at most one at termination)
  const shots = results?.shots || {};
  const shotEntries = Object.entries(shots);
  if (shotEntries.length > 0) {
    const [, shot] = shotEntries[0];
    const distance = Number(shot?.distance ?? 9999);
    const isDunk = distance === 0;
    const isThree = !isDunk && distance >= Number(finalState?.three_point_distance ?? 4);
    const isTwo = !isDunk && !isThree;
    const made = Boolean(shot?.success);
    const assisted = Boolean(shot?.assist_full);
    const potentialAssisted = Boolean(shot?.assist_potential) && !made;

    if (isDunk) {
      statsState.value.dunk.attempts += 1;
      if (made) statsState.value.dunk.made += 1;
      if (assisted) statsState.value.dunk.assists += 1;
      if (potentialAssisted) statsState.value.dunk.potentialAssists += 1;
    } else if (isThree) {
      statsState.value.threePt.attempts += 1;
      if (made) statsState.value.threePt.made += 1;
      if (assisted) statsState.value.threePt.assists += 1;
      if (potentialAssisted) statsState.value.threePt.potentialAssists += 1;
    } else if (isTwo) {
      statsState.value.twoPt.attempts += 1;
      if (made) statsState.value.twoPt.made += 1;
      if (assisted) statsState.value.twoPt.assists += 1;
      if (potentialAssisted) statsState.value.twoPt.potentialAssists += 1;
    }

    if (made) {
      statsState.value.points += isThree ? 3 : 2;
    }
  }

  // Turnovers at termination (array contains a single turnover if present)
  const tovCount = Array.isArray(results?.turnovers) ? results.turnovers.length : 0;
  statsState.value.turnovers += Number(tovCount || 0);

  // Add episode reward for user's team
  // If episodeData is provided (from evaluation), use it directly
  // Otherwise, fetch from API if not skipping
  if (episodeData && episodeData.episode_rewards && episodeData.steps !== undefined) {
    const ep = episodeData.episode_rewards;
    const userTeam = finalState?.user_team_name || 'OFFENSE';
    statsState.value.rewardSum += Number(userTeam === 'OFFENSE' ? ep.offense : ep.defense) || 0;
    statsState.value.episodeStepsSum += Number(episodeData.steps || 0);
    console.log('[Stats] Using episodeData - reward:', userTeam === 'OFFENSE' ? ep.offense : ep.defense, 'steps:', episodeData.steps);
  } else if (!skipApiCall) {
    try {
      const data = await getRewards();
      const ep = data?.episode_rewards || { offense: 0.0, defense: 0.0 };
      const userTeam = finalState?.user_team_name || 'OFFENSE';
      statsState.value.rewardSum += Number(userTeam === 'OFFENSE' ? ep.offense : ep.defense) || 0;
      const steps = Array.isArray(data?.reward_history) ? data.reward_history.length : 0;
      statsState.value.episodeStepsSum += Number(steps || 0);
      console.log('[Stats] Using API data - reward:', userTeam === 'OFFENSE' ? ep.offense : ep.defense, 'steps:', steps);
    } catch (_) { /* ignore */ }
  }

  // Increment episode count last
  statsState.value.episodes += 1;
  console.log('[Stats] recordEpisodeStats completed - new episodes:', statsState.value.episodes);
  saveStats(statsState.value);
}

function resetStats() {
  statsState.value = resetStatsStorage();
}

async function copyStatsMarkdown() {
  try {
    const s = statsState.value;
    const fg = (made, att) => (safeDiv(made, Math.max(1, att)) * 100).toFixed(1) + '%';
    const summary = [
      ['Episodes', String(s.episodes)],
      ['PPP', ppp.value.toFixed(2)],
      ['Avg reward/ep', avgRewardPerEp.value.toFixed(2)],
      ['Avg ep length (steps)', safeDiv(s.episodeStepsSum, Math.max(1, s.episodes)).toFixed(1)],
      ['Total assists', String(s.dunk.assists + s.twoPt.assists + s.threePt.assists)],
      ['Total potential assists', String(s.dunk.potentialAssists + s.twoPt.potentialAssists + s.threePt.potentialAssists)],
      ['Total turnovers', String(s.turnovers)],
    ];
    const shotsHeader = ['Type', 'Attempts', 'Made', 'FG%', 'Assists', 'Potential assists (missed)'];
    const shotsRows = [
      ['Dunks', s.dunk.attempts, s.dunk.made, fg(s.dunk.made, s.dunk.attempts), s.dunk.assists, s.dunk.potentialAssists],
      ['2PT', s.twoPt.attempts, s.twoPt.made, fg(s.twoPt.made, s.twoPt.attempts), s.twoPt.assists, s.twoPt.potentialAssists],
      ['3PT', s.threePt.attempts, s.threePt.made, fg(s.threePt.made, s.threePt.attempts), s.threePt.assists, s.threePt.potentialAssists],
    ];
    const table = (rows) => rows.map(r => `| ${r.join(' | ')} |`).join('\n');
    const md = [
      '## Summary',
      '| Metric | Value |',
      '| --- | --- |',
      table(summary),
      '',
      '## Shots',
      `| ${shotsHeader.join(' | ')} |`,
      `| ${shotsHeader.map(()=>'---').join(' | ')} |`,
      table(shotsRows),
      '',
    ].join('\n');
    if (navigator?.clipboard?.writeText) {
      await navigator.clipboard.writeText(md);
      alert('Stats copied to clipboard as Markdown');
    } else {
      const ta = document.createElement('textarea');
      ta.value = md;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand('copy');
      document.body.removeChild(ta);
      alert('Stats copied to clipboard as Markdown');
    }
  } catch (e) {
    console.warn('[Stats] Failed to copy stats markdown', e);
    alert('Failed to copy stats');
  }
}

// Expose for parent (keyboard shortcut)
defineExpose({ resetStats, copyStatsMarkdown, submitActions, recordEpisodeStats, getSelectedActions });

const isDefense = computed(() => {
  if (!props.gameState || props.activePlayerId === null) return false;
  return props.gameState.defense_ids.includes(props.activePlayerId);
});

const userControlledPlayerIds = computed(() => {
  if (!props.gameState || !props.gameState.user_team_name) {
    return [];
  }
  return props.gameState.user_team_name === 'OFFENSE' 
    ? props.gameState.offense_ids 
    : props.gameState.defense_ids;
});

// All player IDs for AI mode
const allPlayerIds = computed(() => {
  if (!props.gameState) {
    return [];
  }
  return [...(props.gameState.offense_ids || []), ...(props.gameState.defense_ids || [])];
});
const offenseIdsLive = computed(() => props.gameState?.offense_ids || []);
const ballHolderSelection = computed(() => props.gameState?.ball_holder ?? null);

function normalizeLegalProbs(probArray, actionMask) {
  if (!Array.isArray(probArray)) return null;
  if (!Array.isArray(actionMask)) return probArray.map((v) => Number(v));

  let total = 0;
  const masked = probArray.map((raw, idx) => {
    const p = Number(raw);
    const allowed = actionMask[idx] > 0;
    if (!Number.isFinite(p) || p <= 0 || !allowed) return 0;
    total += p;
    return p;
  });

  if (total <= 0) return masked;
  return masked.map((p) => p / total);
}

function computeEntropy(probArray, actionMask) {
  const probs = normalizeLegalProbs(probArray, actionMask);
  if (!Array.isArray(probs)) return null;
  let entropy = 0;
  for (const raw of probs) {
    const p = Number(raw);
    if (!Number.isFinite(p) || p <= 0) continue;
    entropy -= p * Math.log(p);
  }
  return entropy;
}

const entropyRows = computed(() => {
  if (!props.gameState || !policyProbabilities.value) return [];
  const offenseIds = props.gameState.offense_ids || [];
  const defenseIds = props.gameState.defense_ids || [];
  const userTeam = props.gameState.user_team_name;
  const opponentHasOwnPolicy = !!props.gameState.opponent_unified_policy_name;

  return allPlayerIds.value.map((pid) => {
    const probs = policyProbabilities.value?.[pid] ?? policyProbabilities.value?.[String(pid)];
    const mask = props.gameState?.action_mask?.[pid];
    const entropy = computeEntropy(probs, mask);
    const isUserTeam =
      (userTeam === 'OFFENSE' && offenseIds.includes(pid)) ||
      (userTeam === 'DEFENSE' && defenseIds.includes(pid));
    const teamLabel = offenseIds.includes(pid) ? 'Offense' : 'Defense';
    const policyOwner = isUserTeam
      ? 'Player policy'
      : opponentHasOwnPolicy
        ? 'Opponent policy'
        : 'Player policy (mirror)';

    return { playerId: pid, teamLabel, policyOwner, entropy, isUserTeam };
  });
});

const entropyTotals = computed(() => {
  let playerPolicy = 0;
  let opponentPolicy = 0;
  let hasPlayer = false;
  let hasOpponent = false;

  entropyRows.value.forEach((row) => {
    if (row.entropy === null || row.entropy === undefined) return;
    if (row.isUserTeam) {
      playerPolicy += row.entropy;
      hasPlayer = true;
    } else {
      opponentPolicy += row.entropy;
      hasOpponent = true;
    }
  });

  return {
    playerPolicy: hasPlayer ? playerPolicy : null,
    opponentPolicy: hasOpponent ? opponentPolicy : null,
  };
});

const hasEntropyData = computed(() => entropyRows.value.some((row) => row.entropy !== null && row.entropy !== undefined));

// Shot probability display is handled on the board

// Fetch policy probabilities for probabilistic action sampling
async function fetchPolicyProbabilities() {
  if (!props.gameState || (props.gameState.done && !props.isManualStepping)) {
    console.log('[PlayerControls] Skipping fetchPolicyProbabilities - no game state or game done');
    return;
  }
  
  console.log('[PlayerControls] Attempting to fetch policy probabilities from:', `${API_BASE_URL}/api/policy_probabilities`);
  
  try {
    const response = await fetch(`${API_BASE_URL}/api/policy_probabilities`);
    console.log('[PlayerControls] Response status:', response.status, response.statusText);
    
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Failed to fetch policy probabilities: ${response.status} ${response.statusText} - ${errorText}`);
    }
    
    const probs = await response.json();
    policyProbabilities.value = probs;
    console.log('[PlayerControls] Fetched policy probabilities:', probs);
  } catch (error) {
    console.error('[PlayerControls] Failed to fetch policy probabilities:', error);
    policyProbabilities.value = null;
  }
}

// Sample an action from probability distribution
function sampleFromProbabilities(probabilities) {
  const cumSum = [];
  let sum = 0;
  
  for (let i = 0; i < probabilities.length; i++) {
    sum += probabilities[i];
    cumSum.push(sum);
  }
  
  if (sum === 0) return 0; // Default to NOOP if no valid probabilities
  
  const random = Math.random() * sum;
  for (let i = 0; i < cumSum.length; i++) {
    if (random <= cumSum[i]) {
      return i;
    }
  }
  return 0; // Fallback to NOOP
}

// Fetch action values for all players (needed for AI mode and display)
async function fetchAllActionValues() {
  if (!props.gameState || (props.gameState.done && !props.isManualStepping)) {
    console.log('[PlayerControls] Skipping fetchAllActionValues - no game state or game done');
    return;
  }
  
  const allValues = {};
  const allIds = allPlayerIds.value;
  
  console.log('[PlayerControls] Fetching action values for all players:', allIds);
  
  // Also calculate min/max for color scaling
  let allNumericValues = [];
  
  for (const playerId of allIds) {
    try {
      const values = await getActionValues(playerId);
      allValues[playerId] = values;
      // Collect numeric values for scaling
      const numericValues = Object.values(values).filter(v => typeof v === 'number');
      allNumericValues.push(...numericValues);
    } catch (error) {
      console.error(`[PlayerControls] Failed to fetch action values for player ${playerId}:`, error);
    }
  }
  
  actionValues.value = allValues;
  console.log('[PlayerControls] All action values:', allValues);
  
  // Set min/max for color scaling
  if (allNumericValues.length > 0) {
    valueRange.value = {
      min: Math.min(...allNumericValues),
      max: Math.max(...allNumericValues)
    };
  } else {
    valueRange.value = { min: 0, max: 0 };
  }
}

// Watch for game state changes to fetch all action values when needed
watch(() => props.gameState, async (newGameState) => {
  console.log('[PlayerControls] Game state changed, fetching AI data... Ball holder:', newGameState?.ball_holder, 'Manual stepping:', props.isManualStepping, 'Replaying:', props.isReplaying);
  
  let consumedStoredValues = false;
  let consumedStoredProbs = false;
  if (newGameState && newGameState.action_values) {
    console.log('[PlayerControls] Applying stored action values from game state snapshot');
    applyStoredActionValues(newGameState.action_values);
    consumedStoredValues = true;
  }
  if (newGameState && newGameState.policy_probabilities) {
    console.log('[PlayerControls] Applying stored policy probabilities from game state snapshot');
    policyProbabilities.value = newGameState.policy_probabilities;
    consumedStoredProbs = true;
  }

  const allowApiFetch = !props.isManualStepping && !props.isReplaying;
  const shouldFetchAIData = newGameState && (!newGameState.done || props.isManualStepping);
  const shouldFetchActionValues = shouldFetchAIData && allowApiFetch && !consumedStoredValues;
  const shouldFetchPolicyProbs = shouldFetchAIData && allowApiFetch && !consumedStoredProbs;

  if (shouldFetchActionValues || shouldFetchPolicyProbs) {
    try {
      if (shouldFetchActionValues) {
        console.log('[PlayerControls] Starting to fetch action values for ball holder:', newGameState.ball_holder);
        await fetchAllActionValues();
      }
      
      if (shouldFetchPolicyProbs) {
        console.log('[PlayerControls] Fetching policy probabilities from API for ball holder:', newGameState.ball_holder);
        await fetchPolicyProbabilities();
        console.log('[PlayerControls] Policy probabilities fetch completed for ball holder:', newGameState.ball_holder);
      } else if (consumedStoredProbs) {
        console.log('[PlayerControls] Skipping policy probability fetch - using stored snapshot');
      }
    } catch (error) {
      console.error('[PlayerControls] Error during AI data fetch:', error);
    }
  } else if (!newGameState) {
    console.log('[PlayerControls] Clearing AI data - no game state');
    actionValues.value = null;
    if (!props.isManualStepping) {
      policyProbabilities.value = null;
    }
    valueRange.value = { min: 0, max: 0 };
  } else if (!consumedStoredValues) {
    console.log('[PlayerControls] Game ended - retaining last action values and policy probabilities');
  }
}, { immediate: true });

// Watch for stored policy probs from parent (during manual stepping)
watch(() => props.storedPolicyProbs, (newStoredProbs) => {
  if (props.isManualStepping && newStoredProbs) {
    console.log('[PlayerControls] Using stored policy probabilities from replay state:', JSON.stringify(newStoredProbs).substring(0, 150));
    policyProbabilities.value = newStoredProbs;
  }
}, { immediate: true });


// Watch for the list of players to be populated, then set the first one as active.
// The `immediate` flag ensures this runs on component creation.
watch(allPlayerIds, (newPlayerIds) => {
    if (newPlayerIds && newPlayerIds.length > 0 && props.activePlayerId === null) {
        // Prefer first user-controlled player, otherwise just the first player
        const firstUser = userControlledPlayerIds.value.length > 0 ? userControlledPlayerIds.value[0] : newPlayerIds[0];
        emit('update:activePlayerId', firstUser);
    }
}, { immediate: true });

watch(allPlayerIds, (newPlayerIds) => {
  if (!newPlayerIds || !newPlayerIds.length) {
    advisorSelectedPlayerIds.value = [];
    return;
  }
  // Keep only IDs that still exist
  advisorSelectedPlayerIds.value = advisorSelectedPlayerIds.value.filter(pid => newPlayerIds.includes(pid));
  // Default selection if empty
  if (!advisorSelectedPlayerIds.value.length) {
    const bh = props.gameState?.ball_holder;
    const defaultId = (bh !== null && bh !== undefined && newPlayerIds.includes(bh)) ? bh : newPlayerIds[0];
    advisorSelectedPlayerIds.value = [defaultId];
  }
}, { immediate: true });

watch(() => props.gameState?.ball_holder, (bh) => {
  if (bh === null || bh === undefined) return;
  // If nothing is selected yet, default to the ball handler
  if (!advisorSelectedPlayerIds.value.length) {
    advisorSelectedPlayerIds.value = [bh];
    return;
  }
  // Keep selection in sync if the current selection is not on the floor
  if (!advisorSelectedPlayerIds.value.includes(bh)) {
    const newList = [...advisorSelectedPlayerIds.value, bh];
    advisorSelectedPlayerIds.value = Array.from(new Set(newList));
  }
}, { immediate: true });

watch([advisorSelectedPlayerIds, advisorMaxDepth, advisorTimeBudget, advisorExplorationC, advisorUsePriors], () => {
  if (useMctsForStep.value) {
    emit('mcts-options-changed', buildMctsOptions());
  }
});

watch(() => props.initialUseMcts, (val) => {
  useMctsForStep.value = !!val;
}, { immediate: true });

// When backend runs MCTS on step, surface results in the table
watch(() => props.mctsResults, (newResults) => {
  if (newResults && typeof newResults === 'object') {
    advisorResults.value = { ...newResults };
  } else if (newResults === null) {
    advisorResults.value = {};
  }
});

const actionNames = Object.values({
  0: "NOOP", 
  1: "MOVE_E", 2: "MOVE_NE", 3: "MOVE_NW", 4: "MOVE_W", 5: "MOVE_SW", 6: "MOVE_SE", 
  7: "SHOOT", 
  8: "PASS_E", 9: "PASS_NE", 10: "PASS_NW", 11: "PASS_W", 12: "PASS_SW", 13: "PASS_SE"
});

function getLegalActions(playerId) {
  if (!props.gameState.action_mask || !props.gameState.action_mask[playerId]) {
    console.log(`[getLegalActions] No action mask for player ${playerId}`);
    return [];
  }
  const mask = props.gameState.action_mask[playerId];
  const legalActions = [];
  for (let i = 0; i < mask.length; i++) {
    if (mask[i] === 1 && i < actionNames.length) {
      legalActions.push(actionNames[i]);
    }
  }
  
  // Debug logging for SHOOT/PASS actions
  const hasShoot = legalActions.includes('SHOOT');
  const hasPass = legalActions.some(action => action.startsWith('PASS_'));
  if (hasShoot || hasPass) {
    console.log(`[getLegalActions] 🚨 Player ${playerId} has SHOOT: ${hasShoot}, PASS: ${hasPass}, Ball holder: ${props.gameState.ball_holder}, Action mask:`, mask);
  }
  
  return legalActions;
}

function getPassStealProbability(move, playerId) {
  // Check if pass steal probabilities were calculated for this move
  // This shows the risk of passing TO this teammate (if they're in the dict)
  if (move.passStealProbabilities && move.passStealProbabilities[playerId] !== undefined) {
    return move.passStealProbabilities[playerId];
  }
  
  return null;
}

function getDefenderPressureProbability(move, playerId) {
  // Check if this player has defender pressure info
  if (!move.actionResults || !move.actionResults.defender_pressure) {
    return null;
  }
  
  const pressureInfo = move.actionResults.defender_pressure[playerId];
  if (!pressureInfo || pressureInfo.total_pressure_prob === undefined) {
    return null;
  }
  
  return pressureInfo.total_pressure_prob;
}

function handleActionSelected(action) {
  if (props.disabled) return; // ignore clicks when disabled
  if (props.activePlayerId !== null) {
    // If the same action is clicked again, deselect it. Otherwise, select the new one.
    if (selectedActions.value[props.activePlayerId] === action) {
      delete selectedActions.value[props.activePlayerId];
    } else {
      selectedActions.value[props.activePlayerId] = action;
      // Optional: automatically switch to next player only when a new action is chosen
      const currentIndex = userControlledPlayerIds.value.indexOf(props.activePlayerId);
      const nextIndex = (currentIndex + 1) % userControlledPlayerIds.value.length;
      emit('update:activePlayerId', userControlledPlayerIds.value[nextIndex]);
    }
    // If parent provided external selections (self-play), clear them so AI mode resumes fresh
    if (props.externalSelections) {
      // Emit a harmless update to notify parent to clear external selections if desired
      // Parent reads this indirectly by starting self-play; on manual override we stop mirroring
    }
  }
}

function buildMctsOptions() {
  if (!useMctsForStep.value) return null;
  const targets = getAdvisorTargets();
  return {
    use_mcts: true,
    player_ids: targets,
    max_depth: advisorMaxDepth.value,
    time_budget_ms: advisorTimeBudget.value,
    exploration_c: advisorExplorationC.value,
    use_priors: advisorUsePriors.value,
  };
}

function submitActions() {
  let actionsToSubmit = {};
  
  // When manually stepping, we want to collect actions for ALL players (user and opponent)
  // that have been selected manually.
  
  // Determine which players to iterate over. 
  // If manual mode (AI Mode off), we allow controlling everyone.
  // If AI mode on, we also want to send everyone (including AI pre-selections).
  const playersToSubmit = allPlayerIds.value; 
  const mctsTargets = useMctsForStep.value ? new Set(getAdvisorTargets()) : new Set();

  for (const playerId of playersToSubmit) {
    // If MCTS is set to override this player, omit the explicit action so the backend can fill it
    if (mctsTargets.has(playerId)) continue;
    // If a selection exists, use it. Otherwise send 0 (NOOP).
    // Note: In AI mode, selections are pre-filled. In Manual mode, they are filled by click.
    // If unselected in manual mode, it defaults to NOOP (0), which allows manual override of opponents.
    const actionName = selectedActions.value[playerId] || 'NOOP';
    const actionIndex = actionNames.indexOf(actionName);
    actionsToSubmit[playerId] = actionIndex !== -1 ? actionIndex : 0;
  }
  
  console.log('[PlayerControls] Emitting actions-submitted with payload:', actionsToSubmit);
  
  // Track moves for the selected team
  const currentTurn = props.moveHistory.filter(m => !m?.isNoteRow).length + 1;
  const teamMoves = {};
  
  for (const playerId of playersToSubmit) {
    const actionName = selectedActions.value[playerId] || 'NOOP';
    teamMoves[`Player ${playerId}`] = actionName;
  }
  
  emit('move-recorded', {
    turn: currentTurn,
    moves: teamMoves,
    mctsPlayers: Array.from(mctsTargets),
  });
  
  emit('actions-submitted', actionsToSubmit);

  // Emit current MCTS options (or null) so parent can include in step
  emit('mcts-options-changed', buildMctsOptions());
  
  if (!props.aiMode) {
    // Only clear selections in manual mode
    selectedActions.value = {};
    // Reset to first available player (prefer user team)
    if (userControlledPlayerIds.value.length > 0) {
      emit('update:activePlayerId', userControlledPlayerIds.value[0]);
    } else if (allPlayerIds.value.length > 0) {
      emit('update:activePlayerId', allPlayerIds.value[0]);
    }
  }
}

function getSelectedActions() {
  // Return current selections for parent to use
  return { ...selectedActions.value };
}

function getAdvisorTargets() {
  const ids = advisorSelectedPlayerIds.value || [];
  if (ids.length > 0) {
    return Array.from(new Set(ids.map(pid => Number(pid))));
  }
  // Fallback to active player if nothing is selected
  if (props.activePlayerId !== null && props.activePlayerId !== undefined) {
    return [Number(props.activePlayerId)];
  }
  return [];
}

async function runAdvisor() {
  advisorLoading.value = true;
  advisorError.value = null;
  advisorResults.value = {};
  advisorProgress.value = 0;
  const targets = getAdvisorTargets();
  if (!targets.length) {
    advisorError.value = 'Select at least one player.';
    advisorLoading.value = false;
    return;
  }
  try {
    const payload = {
      max_depth: advisorMaxDepth.value,
      time_budget_ms: advisorTimeBudget.value,
      exploration_c: advisorExplorationC.value,
      use_priors: advisorUsePriors.value,
    };
    const results = {};
    let completed = 0;
    for (const pid of targets) {
      const res = await mctsAdvise({ ...payload, player_id: pid });
      results[pid] = res?.advice || null;
      completed += 1;
      advisorProgress.value = targets.length ? completed / targets.length : 1;
    }
    advisorResults.value = results;
  } catch (err) {
    advisorError.value = err?.message || 'Failed to fetch advice';
  } finally {
    advisorProgress.value = 1;
    advisorLoading.value = false;
  }
}

function applyAdvisorAction() {
  const targets = getAdvisorTargets();
  if (!targets.length) return;
  let errorMsg = null;
  const newSelections = { ...selectedActions.value };
  for (const pid of targets) {
    const res = advisorResults.value?.[pid];
    if (!res || res.action === undefined || res.action === null) continue;
    const actionIdx = Number(res.action);
    const actionName = actionNames[actionIdx] || 'NOOP';
    const legal = getLegalActions(pid);
    if (legal.length > 0 && !legal.includes(actionName)) {
      errorMsg = `Advisor suggested ${actionName} for Player ${pid}, but it is not legal now.`;
      continue;
    }
    newSelections[pid] = actionName;
  }
  selectedActions.value = newSelections;
  emit('selections-changed', { ...selectedActions.value });
  advisorError.value = errorMsg;
}

function toggleUseMcts(val) {
  useMctsForStep.value = val;
  emit('mcts-toggle-changed', val);
  emit('mcts-options-changed', buildMctsOptions());
}

// Watch for AI mode or deterministic mode changes to pre-select actions
watch([() => props.aiMode, () => props.deterministic, () => props.opponentDeterministic], ([newAiMode, newDeterministic, newOpponentDeterministic]) => {
  // If parent is driving selections (self-play), don't override
  if (props.externalSelections) return;
  
  try {
    if (newAiMode) {
      // Pre-select AI actions for ALL players when AI mode is enabled
      const newSelections = {};
      const allIds = allPlayerIds.value;
      
      for (const playerId of allIds) {
        const legalActions = getLegalActions(playerId);
        
        if (legalActions.length > 0) {
          let selectedAction = null;
          
          // Determine if this is a user or opponent player to apply correct deterministic setting
          const isUserPlayer = userControlledPlayerIds.value.includes(playerId);
          const useDeterministic = isUserPlayer ? newDeterministic : newOpponentDeterministic;
          
          if (policyProbabilities.value && policyProbabilities.value[playerId]) {
            const probs = policyProbabilities.value[playerId];
            
            if (useDeterministic) {
              // Deterministic: mimic policy.predict(..., deterministic=True) → argmax of policy distribution
              let bestIdx = -1;
              let bestProb = -1;
              for (let i = 0; i < probs.length && i < actionNames.length; i++) {
                const name = actionNames[i];
                if (!legalActions.includes(name)) continue;
                if (probs[i] > bestProb) {
                  bestProb = probs[i];
                  bestIdx = i;
                }
              }
              if (bestIdx >= 0) {
                selectedAction = actionNames[bestIdx];
              }
            } else {
              // Probabilistic: Sample from policy probabilities
              const legalActionIndices = [];
              const legalProbs = [];
              
              for (let i = 0; i < probs.length && i < actionNames.length; i++) {
                if (legalActions.includes(actionNames[i])) {
                  legalActionIndices.push(i);
                  legalProbs.push(probs[i]);
                }
              }
              
              if (legalProbs.length > 0) {
                const sampledIndex = sampleFromProbabilities(legalProbs);
                const actionIndex = legalActionIndices[sampledIndex];
                selectedAction = actionNames[actionIndex];
              }
            }
          }
          
          if (selectedAction) {
            newSelections[playerId] = selectedAction;
          }
        }
      }
      
      selectedActions.value = newSelections;
      console.log('[PlayerControls] 📝 Updated selectedActions via AI mode:', selectedActions.value);
    } else {
      // Clear selections when AI mode is disabled
      console.log('[PlayerControls] Clearing AI mode selections');
      selectedActions.value = {};
    }
  } catch (error) {
    console.error('[PlayerControls] Error in AI mode watch:', error);
  }
});

// Re-sample probabilistic actions whenever policy probabilities update
watch(() => policyProbabilities.value, () => {
  if (props.externalSelections) return;
  try {
    if (!(props.aiMode && policyProbabilities.value)) {
      return;
    }

    const newSelections = {};
    const allIds = allPlayerIds.value;

    for (const playerId of allIds) {
      const legalActions = getLegalActions(playerId);
      const playerProbs = policyProbabilities.value?.[playerId];
      if (!playerProbs || legalActions.length === 0) continue;

      // Build list of legal (index, prob)
      const legalActionIndices = [];
      const legalProbs = [];
      for (let i = 0; i < playerProbs.length && i < actionNames.length; i++) {
        if (legalActions.includes(actionNames[i])) {
          legalActionIndices.push(i);
          legalProbs.push(playerProbs[i]);
        }
      }

      if (legalProbs.length === 0) continue;

      // Determine if this is a user or opponent player to apply correct deterministic setting
      const isUserPlayer = userControlledPlayerIds.value.includes(playerId);
      const useDeterministic = isUserPlayer ? props.deterministic : props.opponentDeterministic;

      if (useDeterministic) {
        // Deterministic: argmax among legal
        let bestIdxLocal = 0;
        let bestProb = -1;
        for (let j = 0; j < legalProbs.length; j++) {
          if (legalProbs[j] > bestProb) {
            bestProb = legalProbs[j];
            bestIdxLocal = j;
          }
        }
        const actionIndex = legalActionIndices[bestIdxLocal];
        const selectedAction = actionNames[actionIndex];
        newSelections[playerId] = selectedAction;
      } else {
        // Probabilistic: sample among legal
        const sampledIndex = sampleFromProbabilities(legalProbs);
        const actionIndex = legalActionIndices[sampledIndex];
        const selectedAction = actionNames[actionIndex];
        newSelections[playerId] = selectedAction;
      }
    }

    if (Object.keys(newSelections).length > 0) {
      selectedActions.value = newSelections;
    }
  } catch (error) {
    console.error('[PlayerControls] Error in policyProbabilities watch:', error);
  }
}, { immediate: true });

// Disable Q-value-driven deterministic preselection to match analytics behavior
watch(() => actionValues.value, () => { /* no-op for deterministic mode */ });

// Shot probability helpers removed

// Fetch rewards from API
const fetchRewards = async () => {
  try {
    const data = await getRewards();
    rewardHistory.value = data.reward_history || [];
    episodeRewards.value = data.episode_rewards || { offense: 0.0, defense: 0.0 };
    rewardParams.value = data.reward_params || null;
    mlflowPhiParams.value = data.mlflow_phi_params || null;
    console.log('[Rewards] Fetched rewards. History length:', rewardHistory.value.length, 'Episode totals:', episodeRewards.value, 'MLflow phi params:', mlflowPhiParams.value);
  } catch (error) {
    console.error('Failed to fetch rewards:', error);
  }
};

// Watch for game state changes to update rewards and clear moves
watch(() => props.gameState, (newState, oldState) => {
  if (newState) {
    console.log('[Rewards] Game state changed, fetching rewards. Done:', newState.done);
    fetchRewards();
    
    // Move history clearing is now handled by parent component
  }
}, { deep: true });

// Watch for when user switches to Rewards tab
watch(() => activeTab.value, (newTab) => {
  if (newTab === 'rewards') {
    console.log('[Rewards] Switched to Rewards tab, fetching rewards');
    fetchRewards();
  }
});

onMounted(() => {
  isMounted.value = true;
  fetchRewards();
  emit('active-tab-changed', activeTab.value);
});

// Record stats once on episode completion (but skip during evaluation mode)
watch(() => props.gameState?.done, async (done, prevDone) => {
  console.log('[Stats] Watch triggered - done:', done, 'prevDone:', prevDone, 'isReplaying:', props.isReplaying, 'isEvaluating:', props.isEvaluating);
  if (done && !prevDone && props.gameState && !props.isReplaying && !props.isEvaluating) {
    console.log('[Stats] Watch conditions met - recording stats from watch');
    try { await recordEpisodeStats(props.gameState); } catch (e) { console.warn('[Stats] record failed', e); }
  }
});

// Backend probability is declared at the top

const passProbabilities = computed(() => {
  // For now, return empty object - this was likely removed in previous changes
  return {};
});

// shotProbability computed removed

// Keep local selections in sync with externally applied selections during self-play
watch(() => props.externalSelections, (newSelections) => {
  if (newSelections && typeof newSelections === 'object') {
    // Replace entire selection map to match applied actions
    selectedActions.value = { ...newSelections };
  }
});

// Ensure Controls tab is visible when active player changes (e.g., from board clicks)
watch(() => props.activePlayerId, (newVal, oldVal) => {
  if (newVal !== oldVal) {
    activeTab.value = 'controls';
    emit('active-tab-changed', activeTab.value);
  }
});

import PhiShaping from './PhiShaping.vue';
import { ref as vueRef } from 'vue';
const phiRef = vueRef(null);

// Observation parsing utilities
function getAngleDescription(cosAngle) {
  if (!Number.isFinite(cosAngle)) return '';
  const angle = Math.max(-1, Math.min(1, cosAngle));
  const absAngle = Math.abs(angle);
  if (absAngle < 0.05) return '➡️ On basket line';
  if (absAngle < 0.25) return angle > 0 ? '↗️ Left of basket line' : '↘️ Right of basket line';
  if (absAngle < 0.5) return angle > 0 ? '⬆️ Far left of basket line' : '⬇️ Far right of basket line';
  if (absAngle < 0.75) return angle > 0 ? '⬅️ Behind-left' : '➡️ Behind-right';
  return angle > 0 ? '⬅️ Opposite direction' : '➡️ Opposite direction';
}

function formatAngleValue(cosAngle) {
  const numeric = Number.isFinite(cosAngle) ? cosAngle : 0;
  const clamped = Math.max(-1, Math.min(1, numeric));
  const degrees = clamped * 180;
  return `${clamped.toFixed(4)} (${degrees.toFixed(1)}°)`;
}

function formatTokenValue(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return '—';
  return numeric.toFixed(4);
}

function attentionColor(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return { r: 15, g: 23, b: 42, a: 0.0 };
  }
  const range = attentionRange.value;
  const denom = range.max - range.min;
  const normalized = denom > 0 ? (numeric - range.min) / denom : 0;
  const intensity = Math.max(0, Math.min(1, normalized));
  const start = { r: 59, g: 130, b: 246 }; // blue-500
  const end = { r: 249, g: 115, b: 22 }; // orange-500
  const r = Math.round(start.r + (end.r - start.r) * intensity);
  const g = Math.round(start.g + (end.g - start.g) * intensity);
  const b = Math.round(start.b + (end.b - start.b) * intensity);
  const a = 0.12 + intensity * 0.75;
  return { r, g, b, a };
}

function attentionCellStyle(value) {
  const color = attentionColor(value);
  return {
    backgroundColor: `rgba(${color.r}, ${color.g}, ${color.b}, ${color.a.toFixed(3)})`,
  };
}

// Computed properties for observation parsing
const offenseIds = computed(() => props.gameState?.offense_ids || []);
const defenseIds = computed(() => props.gameState?.defense_ids || []);

function formatOffenseId(index) {
  const ids = offenseIds.value;
  if (Number.isInteger(ids[index])) return ids[index];
  return index;
}

function formatDefenseId(index) {
  const ids = defenseIds.value;
  if (Number.isInteger(ids[index])) return ids[index];
  return offenseIds.value.length + index;
}

const numDefenders = computed(() => {
  if (!props.gameState) return 0;
  return props.gameState.defense_ids?.length || 0;
});

const numOffenders = computed(() => {
  if (!props.gameState) return 0;
  return props.gameState.offense_ids?.length || 0;
});

const playerPositionRows = computed(() => {
  if (!props.gameState || !props.gameState.obs) return [];
  const obs = props.gameState.obs;
  const nPlayers = (props.gameState.offense_ids?.length || 0) + (props.gameState.defense_ids?.length || 0);
  // First nPlayers*2 elements
  return obs.slice(0, nPlayers * 2);
});

const ballHolderOHE = computed(() => {
  if (!props.gameState || !props.gameState.obs) return [];
  const obs = props.gameState.obs;
  const nPlayers = (props.gameState.offense_ids?.length || 0) + (props.gameState.defense_ids?.length || 0);
  const startIdx = nPlayers * 2;
  return obs.slice(startIdx, startIdx + nPlayers);
});

const shotClockValue = computed(() => {
  if (!props.gameState || !props.gameState.obs) return 0;
  const obs = props.gameState.obs;
  const nPlayers = (props.gameState.offense_ids?.length || 0) + (props.gameState.defense_ids?.length || 0);
  const idx = nPlayers * 2 + nPlayers;
  return obs[idx] || 0;
});

const teamEncodingRows = computed(() => {
  if (!props.gameState || !props.gameState.obs) return [];
  const obs = props.gameState.obs;
  const nPlayers = (props.gameState.offense_ids?.length || 0) + (props.gameState.defense_ids?.length || 0);
  const idx = nPlayers * 2 + nPlayers + 1;
  return obs.slice(idx, idx + nPlayers);
});

const ballHandlerPositionRows = computed(() => {
  if (!props.gameState || !props.gameState.obs) return [];
  const obs = props.gameState.obs;
  const nPlayers = (props.gameState.offense_ids?.length || 0) + (props.gameState.defense_ids?.length || 0);
  const idx = nPlayers * 2 + nPlayers + 1 + nPlayers; // +nPlayers for team encoding
  return obs.slice(idx, idx + 2);
});

const hoopVectorRows = computed(() => {
  const meta = obsMeta.value;
  const obs = props.gameState?.obs;
  if (!meta || !obs || meta.hoopLen === 0) return [];
  return obs.slice(meta.hoopStart, meta.hoopStart + meta.hoopLen);
});

const obsMeta = computed(() => {
  const gs = props.gameState;
  if (!gs || !gs.obs) return null;
  const nOffense = gs.offense_ids?.length || 0;
  const nDefense = gs.defense_ids?.length || 0;
  const nPlayers = nOffense + nDefense;
  const allPairsSize = nOffense * nDefense;
  const offensePairs = (nOffense * (nOffense - 1)) / 2;
  const defensePairs = (nDefense * (nDefense - 1)) / 2;
  const teammatePairCount = offensePairs + defensePairs;
  const teammateDistanceSize = teammatePairCount;
  const teammateAngleSize = (nOffense * (nOffense - 1)) + (nDefense * (nDefense - 1));
  let offset = 0;
  offset += nPlayers * 2; // player positions
  offset += nPlayers; // ball holder one-hot
  const shotClockIdx = offset;
  offset += 1;
  const teamEncodingStart = offset;
  offset += nPlayers;
  const ballHandlerStart = offset;
  offset += 2;
  const hoopLen = gs.include_hoop_vector ? 2 : 0;
  const hoopStart = offset;
  offset += hoopLen;
  const allPairsDistancesStart = offset;
  offset += allPairsSize;
  const allPairsAnglesStart = offset;
  offset += allPairsSize;
  const teammateDistanceStart = offset;
  offset += teammateDistanceSize;
  const teammateAngleStart = offset;
  offset += teammateAngleSize;
  const laneStepsStart = offset;
  const laneStepsLen = nPlayers;
  offset += laneStepsLen;
  const expectedPointsStart = offset;
  const expectedPointsLen = nOffense;
  offset += expectedPointsLen;
  const turnoverStart = offset;
  offset += nOffense;
  const stealStart = offset;
  return {
    shotClockIdx,
    teamEncodingStart,
    ballHandlerStart,
    hoopStart,
    hoopLen,
    allPairsDistancesStart,
    allPairsAnglesStart,
    allPairsSize,
    teammateDistanceStart,
    teammateAngleStart,
    teammateDistanceSize,
    teammateAngleSize,
    laneStepsStart,
    laneStepsLen,
    expectedPointsStart,
    expectedPointsLen,
    turnoverStart,
    stealStart,
    nOffense,
    teammatePairCount,
  };
});

const obsTokens = computed(() => props.gameState?.obs_tokens || null);
const tokenFeatureLabels = [
  'q_norm',
  'r_norm',
  'role',
  'has_ball',
  'layup%',
  '3pt%',
  'dunk%',
  'steps',
  'EP',
  'tov%',
  'stl%',
  'dist_to_ball',
  'dist_to_best_ep',
  'dist_to_nearest_opp',
  'dist_to_nearest_team',
];
const tokenGlobalLabels = ['shot_clock', 'pressure_exposure', 'hoop_q_norm', 'hoop_r_norm'];

const tokenPlayers = computed(() => {
  const players = obsTokens.value?.players;
  return Array.isArray(players) ? players : [];
});

const tokenGlobals = computed(() => {
  const globals = obsTokens.value?.globals;
  return Array.isArray(globals) ? globals : [];
});

const tokenAttention = computed(() => obsTokens.value?.attention || null);
const tokenAttentionLabels = computed(() => tokenAttention.value?.labels || []);
const tokenAttentionAvgWeights = computed(() => tokenAttention.value?.weights_avg || []);
const tokenAttentionHeadWeights = computed(() => tokenAttention.value?.weights_heads || []);
const tokenAttentionHeads = computed(() => tokenAttention.value?.heads ?? null);
const attentionView = ref('avg');
const attentionHeadOptions = computed(() => {
  const count = tokenAttentionHeads.value || 0;
  return Array.from({ length: count }, (_, idx) => idx);
});
const tokenAttentionMatrix = computed(() => {
  if (attentionView.value === 'avg') {
    return tokenAttentionAvgWeights.value;
  }
  const idx = Number(attentionView.value);
  const heads = tokenAttentionHeadWeights.value;
  if (!Array.isArray(heads) || !Array.isArray(heads[idx])) {
    return tokenAttentionAvgWeights.value;
  }
  return heads[idx];
});
const tokenAttentionSubtitle = computed(() => {
  const count = tokenAttentionHeads.value;
  if (!count) return '';
  if (attentionView.value === 'avg') {
    return `Average of ${count} heads`;
  }
  const idx = Number(attentionView.value);
  if (!Number.isFinite(idx)) return `Average of ${count} heads`;
  return `Head ${idx + 1} of ${count}`;
});
const attentionRange = computed(() => {
  const weights = tokenAttentionMatrix.value;
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  for (const row of weights) {
    if (!Array.isArray(row)) continue;
    for (const val of row) {
      const num = Number(val);
      if (!Number.isFinite(num)) continue;
      if (num < min) min = num;
      if (num > max) max = num;
    }
  }
  if (!Number.isFinite(min) || !Number.isFinite(max)) {
    return { min: 0, max: 1 };
  }
  if (max <= min) {
    return { min, max: min + 1 };
  }
  return { min, max };
});

function downloadAttentionPng() {
  if (!tokenAttentionMatrix.value.length) {
    alert('No attention data available.');
    return;
  }
  const labels = tokenAttentionLabels.value;
  const matrix = tokenAttentionMatrix.value;
  const n = matrix.length;

  const padding = 12;
  const fontSize = 12;
  const font = `${fontSize}px "Courier New", monospace`;

  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  ctx.font = font;
  let maxLabelWidth = 0;
  labels.forEach((label) => {
    const width = ctx.measureText(String(label)).width;
    if (width > maxLabelWidth) maxLabelWidth = width;
  });
  const cellSize = Math.max(36, Math.ceil(maxLabelWidth) + 12);
  const headerSize = Math.max(cellSize + padding, Math.ceil(maxLabelWidth) + padding * 2);
  const width = headerSize + cellSize * n;
  const height = headerSize + cellSize * n;

  const scale = 2;
  canvas.width = Math.ceil(width * scale);
  canvas.height = Math.ceil(height * scale);
  ctx.scale(scale, scale);
  ctx.fillStyle = 'rgba(15, 23, 42, 0.95)';
  ctx.fillRect(0, 0, width, height);

  ctx.fillStyle = '#cbd5f5';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.font = font;

  for (let c = 0; c < n; c += 1) {
    const label = labels[c] ?? `T${c}`;
    const x = headerSize + c * cellSize + cellSize / 2;
    const y = headerSize / 2;
    ctx.fillText(String(label), x, y);
  }

  for (let r = 0; r < n; r += 1) {
    const label = labels[r] ?? `T${r}`;
    const x = headerSize / 2;
    const y = headerSize + r * cellSize + cellSize / 2;
    ctx.fillText(String(label), x, y);
  }

  for (let r = 0; r < n; r += 1) {
    for (let c = 0; c < n; c += 1) {
      const val = matrix[r]?.[c];
      const color = attentionColor(val);
      const x = headerSize + c * cellSize;
      const y = headerSize + r * cellSize;
      ctx.fillStyle = `rgba(${color.r}, ${color.g}, ${color.b}, ${color.a})`;
      ctx.fillRect(x, y, cellSize, cellSize);
      ctx.strokeStyle = 'rgba(148, 163, 184, 0.35)';
      ctx.strokeRect(x, y, cellSize, cellSize);
      ctx.fillStyle = '#0b1020';
      ctx.font = `10px "Courier New", monospace`;
      ctx.fillText(formatTokenValue(val), x + cellSize / 2, y + cellSize / 2);
    }
  }

  const link = document.createElement('a');
  link.download = 'attention_map.png';
  link.href = canvas.toDataURL('image/png');
  link.click();
}

const tokenRows = computed(() => {
  if (!props.gameState) return [];
  return tokenPlayers.value.map((row, idx) => {
    const teamLabel = offenseIds.value.includes(idx)
      ? 'Offense'
      : defenseIds.value.includes(idx)
        ? 'Defense'
        : 'Unknown';
    return {
      playerId: idx,
      teamLabel,
      features: Array.isArray(row) ? row : [],
    };
  });
});

const tokenGlobalRows = computed(() => {
  return tokenGlobals.value.map((value, idx) => ({
    label: tokenGlobalLabels[idx] || `global_${idx}`,
    value,
  }));
});

const allPairsDistances = computed(() => {
  const meta = obsMeta.value;
  const obs = props.gameState?.obs;
  if (!meta || !obs) return [];
  return obs.slice(meta.allPairsDistancesStart, meta.allPairsDistancesStart + meta.allPairsSize);
});

const allPairsAngles = computed(() => {
  const meta = obsMeta.value;
  const obs = props.gameState?.obs;
  if (!meta || !obs) return [];
  return obs.slice(meta.allPairsAnglesStart, meta.allPairsAnglesStart + meta.allPairsSize);
});

const teammateDistances = computed(() => {
  const meta = obsMeta.value;
  const obs = props.gameState?.obs;
  if (!meta || !obs || meta.teammateDistanceSize === 0) return [];
  return obs.slice(
    meta.teammateDistanceStart,
    meta.teammateDistanceStart + meta.teammateDistanceSize,
  );
});

const teammateAngles = computed(() => {
  const meta = obsMeta.value;
  const obs = props.gameState?.obs;
  if (!meta || !obs || meta.teammateAngleSize === 0) return [];
  return obs.slice(
    meta.teammateAngleStart,
    meta.teammateAngleStart + meta.teammateAngleSize,
  );
});

const teammateDistanceLabels = computed(() => {
  const meta = obsMeta.value;
  if (!meta || meta.teammatePairCount === 0) return [];
  const labels = [];
  const appendTeam = (teamIds, label) => {
    if (!teamIds || teamIds.length <= 1) return;
    for (let i = 0; i < teamIds.length - 1; i += 1) {
      const baseId = teamIds[i];
      for (let j = i + 1; j < teamIds.length; j += 1) {
        labels.push(`${label} P${baseId} → P${teamIds[j]}`);
      }
    }
  };
  appendTeam(props.gameState.offense_ids, 'Offense');
  appendTeam(props.gameState.defense_ids, 'Defense');
  return labels;
});

const teammateAngleLabels = computed(() => {
  const meta = obsMeta.value;
  if (!meta || meta.teammateAngleSize === 0) return [];
  const labels = [];
  const appendTeam = (teamIds, label) => {
    if (!teamIds || teamIds.length <= 1) return;
    for (let i = 0; i < teamIds.length; i += 1) {
      const baseId = teamIds[i];
      for (let j = 0; j < teamIds.length; j += 1) {
        if (j === i) continue;
        labels.push(`${label} P${baseId} → P${teamIds[j]}`);
      }
    }
  };
  appendTeam(props.gameState.offense_ids, 'Offense');
  appendTeam(props.gameState.defense_ids, 'Defense');
  return labels;
});

const laneSteps = computed(() => {
  const meta = obsMeta.value;
  const obs = props.gameState?.obs;
  if (!meta || !obs) return [];
  return obs.slice(meta.laneStepsStart, meta.laneStepsStart + meta.laneStepsLen);
});

const expectedPoints = computed(() => {
  const meta = obsMeta.value;
  const obs = props.gameState?.obs;
  if (!meta || !obs) return [];
  return obs.slice(
    meta.expectedPointsStart,
    meta.expectedPointsStart + meta.expectedPointsLen,
  );
});

const turnoverProbs = computed(() => {
  const meta = obsMeta.value;
  const obs = props.gameState?.obs;
  if (!meta || !obs) return [];
  return obs.slice(
    meta.turnoverStart,
    meta.turnoverStart + meta.nOffense,
  );
});

const stealRisks = computed(() => {
  const meta = obsMeta.value;
  const obs = props.gameState?.obs;
  if (!meta || !obs) return [];
  return obs.slice(
    meta.stealStart,
    meta.stealStart + meta.nOffense,
  );
});
</script>

<template>
  <div class="player-controls-container">
    <h3>Player Controls</h3>
    
    <!-- Tab Navigation -->
    <div class="tab-navigation">
      <button 
        :class="{ active: activeTab === 'controls' }"
        @click="activeTab = 'controls'"
      >
        Controls
      </button>
      <button 
        :class="{ active: activeTab === 'rewards' }"
        @click="activeTab = 'rewards'"
      >
        Rewards
      </button>
      <button 
        :class="{ active: activeTab === 'stats' }"
        @click="activeTab = 'stats'"
      >
        Stats
      </button>
      <button 
        :class="{ active: activeTab === 'entropy' }"
        @click="activeTab = 'entropy'"
      >
        Entropy
      </button>
      <button 
        :class="{ active: activeTab === 'advisor' }"
        @click="activeTab = 'advisor'"
      >
        Advisor
      </button>
      <button 
        :class="{ active: activeTab === 'moves' }"
        @click="activeTab = 'moves'"
      >
        Moves
      </button>
      <button 
        :class="{ active: activeTab === 'eval' }"
        @click="activeTab = 'eval'"
      >
        Eval
      </button>
      <button 
        :class="{ active: activeTab === 'environment' }"
        @click="activeTab = 'environment'"
      >
        Environment
      </button>
      <button 
        :class="{ active: activeTab === 'training' }"
        @click="activeTab = 'training'"
      >
        Training
      </button>
      <button 
        :class="{ active: activeTab === 'phi' }"
        @click="activeTab = 'phi'"
      >
        Phi Shaping
      </button>
      <button 
        :class="{ active: activeTab === 'observation' }"
        @click="activeTab = 'observation'"
      >
        Observation
      </button>
      <button 
        :class="{ active: activeTab === 'attention' }"
        @click="activeTab = 'attention'"
      >
        Attention
      </button>
    </div>

    <!-- Controls Tab -->
    <div v-if="activeTab === 'controls'" class="tab-content">
      <div class="ball-holder-row">
        <label>Ball handler</label>
        <select
          :value="ballHolderSelection ?? ''"
          @change="handleBallHolderChange($event.target.value)"
          :disabled="ballHolderUpdating || props.isEvaluating || props.isReplaying || offenseIdsLive.length === 0"
        >
          <option v-if="offenseIdsLive.length === 0" disabled value="">No offense players</option>
          <option v-for="pid in offenseIdsLive" :key="`bh-${pid}`" :value="pid">Player {{ pid }}</option>
        </select>
        <span v-if="ballHolderUpdating" class="status-note">Updating…</span>
        <span v-if="ballHolderError" class="error-note">{{ ballHolderError }}</span>
      </div>

      <div class="player-tabs">
          <button 
              v-for="playerId in allPlayerIds" 
              :key="playerId"
              :class="{ active: activePlayerId === playerId }"
              @click="$emit('update:activePlayerId', playerId)"
              :disabled="false"
          >
              Player {{ playerId }}
              <span v-if="selectedActions[playerId]">
                ({{ selectedActions[playerId].startsWith('MOVE') ? 'M' : selectedActions[playerId].startsWith('PASS') ? 'P' : selectedActions[playerId] }})
              </span>
          </button>
      </div>
      
      <div class="control-pad-wrapper" v-if="activePlayerId !== null">
          <HexagonControlPad 
              :legal-actions="getLegalActions(activePlayerId)"
              :selected-action="selectedActions[activePlayerId]"
              :pass-probabilities="passProbabilities"
              @action-selected="handleActionSelected"
              :action-values="actionValues && actionValues[activePlayerId] ? actionValues[activePlayerId] : null"
              :value-range="valueRange"
              :is-defense="isDefense"
          />
          <p v-if="selectedActions[activePlayerId]">
              Selected for Player {{ activePlayerId }}: <strong>{{ selectedActions[activePlayerId] }}</strong>
          </p>
      </div>
    </div>

    <!-- Advisor Tab -->
    <div v-if="activeTab === 'advisor'" class="tab-content advisor-tab">
      <div class="advisor-table-wrapper">
        <table class="advisor-table">
          <thead>
            <tr>
              <th>Select</th>
              <th>Player</th>
              <th>Recommended</th>
              <th>Q</th>
              <th>Stats</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="pid in allPlayerIds" :key="pid">
              <td>
                <input type="checkbox" :value="Number(pid)" v-model="advisorSelectedPlayerIds" />
              </td>
              <td>Player {{ pid }}</td>
              <td>
                <span v-if="advisorResults?.[pid] && advisorResults[pid]?.action !== undefined">
                  {{ actionNames[advisorResults[pid].action] || 'UNKNOWN' }}
                </span>
                <span v-else>—</span>
                <div v-if="advisorPolicyTop?.[pid]?.length" class="policy-row">
                  <span v-for="item in advisorPolicyTop[pid]" :key="item.idx" class="policy-chip">
                    {{ actionNames[item.idx] || 'UNKNOWN' }} {{ (item.prob * 100).toFixed(0) }}%
                  </span>
                </div>
              </td>
              <td>
                <span v-if="advisorResults?.[pid]?.q_estimate !== null && advisorResults?.[pid]?.q_estimate !== undefined">
                  {{ Number(advisorResults[pid].q_estimate).toFixed(2) }}
                </span>
                <span v-else>—</span>
              </td>
              <td class="advisor-stat-cell">
                <span v-if="advisorResults?.[pid]?.visits">Visits: {{ advisorResults[pid].visits?.[advisorResults[pid].action] || 0 }}</span>
                <span v-if="advisorResults?.[pid]?.nodes_expanded">Nodes: {{ advisorResults[pid].nodes_expanded }}</span>
                <span v-if="advisorResults?.[pid]?.max_depth_reached">Depth: {{ advisorResults[pid].max_depth_reached }}</span>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
      <div class="advisor-grid">
        <label>
          Max depth (plies)
          <input type="number" min="1" v-model.number="advisorMaxDepth" />
        </label>
        <label>
          Time budget (ms)
          <input type="number" min="10" step="10" v-model.number="advisorTimeBudget" />
        </label>
        <label>
          Exploration C
          <input type="number" step="0.1" v-model.number="advisorExplorationC" />
        </label>
        <label class="checkbox">
          <input type="checkbox" v-model="advisorUsePriors" /> Use policy priors
        </label>
        <label class="checkbox">
          <input type="checkbox" :checked="useMctsForStep" @change="toggleUseMcts($event.target.checked)" /> Use MCTS on next step
        </label>
        <div class="advisor-actions">
          <button @click="runAdvisor" :disabled="advisorLoading || !props.gameState || props.gameState.done">
            {{ advisorLoading ? 'Running...' : 'Get Advice for Selected' }}
          </button>
          <button @click="applyAdvisorAction" :disabled="!Object.keys(advisorResults || {}).length" title="Apply recommended actions to selection">
            Apply Actions
          </button>
        </div>
      </div>
      <div class="advisor-progress" v-if="advisorLoading">
        <div class="advisor-progress-bar">
          <div 
            class="advisor-progress-fill" 
            :class="{ indeterminate: advisorProgress < 1 }"
            :style="advisorProgress >= 1 ? { width: '100%' } : {}"
          ></div>
        </div>
        <span class="advisor-progress-text">Running MCTS…</span>
      </div>
      <div class="advisor-note" v-if="props.mctsResults && useMctsForStep">
        Showing MCTS results returned by the last step for selected players.
      </div>
      <div class="advisor-progress" v-if="useMctsForStep && props.mctsStepRunning">
        <div class="advisor-progress-bar">
          <div class="advisor-progress-fill indeterminate"></div>
        </div>
        <span class="advisor-progress-text">Running MCTS for turn…</span>
      </div>
      <div v-if="advisorLoading && !Object.keys(advisorResults || {}).length" class="advisor-results">Running advisor…</div>
      <div v-if="advisorError" class="advisor-error">{{ advisorError }}</div>
    </div>

    <!-- Rewards Tab -->
    <div v-if="activeTab === 'rewards'" class="tab-content">
      <div class="rewards-section">
        <h4>Reward Parameters</h4>
        <div class="parameters-grid" v-if="rewardParams">
          <div class="param-category">
            <h5>Shot Rewards</h5>
            <div class="param-item">
              <span class="param-name">Shot reward:</span>
              <span class="param-value">
                {{ rewardParams.shot_reward_description || 'Expected points (shot value × pressure-adjusted make probability, applies to makes and misses)' }}
              </span>
            </div>
            <div class="param-item">
              <span class="param-name">Turnover reward:</span>
              <span class="param-value">{{ rewardParams.turnover_reward !== undefined ? rewardParams.turnover_reward : 0 }}</span>
            </div>
          </div>
          <div class="param-category">
            <h5>Assist Shaping</h5>
            <div class="param-item"><span class="param-name">Potential assist % of shot:</span><span class="param-value">{{ (rewardParams.potential_assist_pct * 100).toFixed(1) }}%</span></div>
            <div class="param-item"><span class="param-name">Full assist bonus % of shot:</span><span class="param-value">{{ (rewardParams.full_assist_bonus_pct * 100).toFixed(1) }}%</span></div>
            <div class="param-item"><span class="param-name">Assist window (steps):</span><span class="param-value">{{ rewardParams.assist_window }}</span></div>
          </div>
          <div class="param-category">
            <h5>Other</h5>
            <div class="param-item"><span class="param-name">Pass reward:</span><span class="param-value">{{ rewardParams.pass_reward }}</span></div>
            <div class="param-item"><span class="param-name">Violation reward:</span><span class="param-value">{{ rewardParams.violation_reward }}</span></div>
          </div>
          <div class="param-category" v-if="mlflowPhiParams && mlflowPhiParams.enable_phi_shaping">
            <h5>Phi Shaping (from MLflow)</h5>
            <div class="param-item"><span class="param-name">Beta (β):</span><span class="param-value">{{ mlflowPhiParams.phi_beta }}</span></div>
            <div class="param-item"><span class="param-name">Gamma (γ):</span><span class="param-value">{{ mlflowPhiParams.reward_shaping_gamma }}</span></div>
            <div class="param-item"><span class="param-name">Aggregation mode:</span><span class="param-value">{{ mlflowPhiParams.phi_aggregation_mode }}</span></div>
            <div class="param-item" v-if="mlflowPhiParams.phi_blend_weight > 0"><span class="param-name">Blend weight:</span><span class="param-value">{{ mlflowPhiParams.phi_blend_weight.toFixed(2) }}</span></div>
          </div>
        </div>
        <div v-else class="no-rewards">No reward parameters available.</div>

        <h4>Episode Totals</h4>
        <div class="episode-totals">
          <div class="total-item">
            <span class="team-label offense">Offense:</span>
            <span class="reward-value">{{ episodeRewards.offense.toFixed(2) }}</span>
          </div>
          <div class="total-item">
            <span class="team-label defense">Defense:</span>
            <span class="reward-value">{{ episodeRewards.defense.toFixed(2) }}</span>
          </div>
        </div>

        <h4>Turn History</h4>
        <div class="reward-history">
          <div v-if="rewardHistory.length === 0" class="no-rewards">
            No rewards recorded yet.
          </div>
          <div v-else class="reward-table" :class="{ 'with-phi': mlflowPhiParams && mlflowPhiParams.enable_phi_shaping }">
            <div class="reward-header">
              <span>Turn</span>
              <span>Shot Clock</span>
              <span>Offense</span>
              <span>Off. Reason</span>
              <span>Defense</span>
              <span>Def. Reason</span>
              <span v-if="mlflowPhiParams && mlflowPhiParams.enable_phi_shaping">Φ</span>
            </div>
            <div 
              v-for="reward in rewardHistory" 
              :key="reward.step"
              class="reward-row"
              :class="{ 'current-reward-row': reward.shot_clock === props.currentShotClock }"
            >
              <span>{{ reward.step }}</span>
              <span class="shot-clock-cell">{{ reward.shot_clock !== undefined ? reward.shot_clock : '-' }}</span>
              <span :class="{ positive: reward.offense > 0, negative: reward.offense < 0 }">
                {{ reward.offense.toFixed(3) }}
              </span>
              <span class="reason-text">{{ reward.offense_reason }}</span>
              <span :class="{ positive: reward.defense > 0, negative: reward.defense < 0 }">
                {{ reward.defense.toFixed(3) }}
              </span>
              <span class="reason-text">{{ reward.defense_reason }}</span>
              <span v-if="mlflowPhiParams && mlflowPhiParams.enable_phi_shaping">
                {{ (reward.mlflow_phi_potential || 0).toFixed(3) }}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Stats Tab -->
    <div v-if="activeTab === 'stats'" class="tab-content">
      <div v-if="offensePlayerStatsTable.length" class="per-player-stats">
        <h4>Per-Player Offense Stats (Eval)</h4>
        <table class="per-player-table">
          <thead>
            <tr>
              <th>Player</th>
              <th>FG</th>
              <th>Dunk</th>
              <th>2PT</th>
              <th>3PT</th>
              <th>Assists</th>
              <th>Pot. Ast</th>
              <th>TOV</th>
              <th>Points</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="row in offensePlayerStatsTable" :key="`stat-${row.playerId}`">
              <td>Player {{ row.playerId }}</td>
              <td>{{ row.makes }}/{{ row.attempts }} ({{ row.fg.toFixed(1) }}%)</td>
              <td>{{ row.dunk.mk }}/{{ row.dunk.att }} ({{ row.unassisted.dunk }})</td>
              <td>{{ row.two.mk }}/{{ row.two.att }} ({{ row.unassisted.two }})</td>
              <td>{{ row.three.mk }}/{{ row.three.att }} ({{ row.unassisted.three }})</td>
              <td>{{ row.assists }}</td>
              <td>{{ row.potentialAssists }}</td>
              <td>{{ row.turnovers }}</td>
              <td>{{ row.points.toFixed(1) }}</td>
            </tr>
          </tbody>
        </table>
      </div>
      <div class="rewards-section">
        <h4>Episode Stats</h4>
        <div class="parameters-grid">
          <div class="param-category">
            <h5>Totals</h5>
            <div class="param-item"><span class="param-name">Episodes played:</span><span class="param-value">{{ statsState.episodes }}</span></div>
            <div class="param-item"><span class="param-name">Total assists:</span><span class="param-value">{{ totalAssists }}</span></div>
            <div class="param-item"><span class="param-name">Total potential assists (missed):</span><span class="param-value">{{ totalPotentialAssists }}</span></div>
            <div class="param-item"><span class="param-name">Total turnovers:</span><span class="param-value">{{ statsState.turnovers }}</span></div>
            <div class="param-item"><span class="param-name">PPP:</span><span class="param-value">{{ ppp.toFixed(2) }}</span></div>
            <div class="param-item"><span class="param-name">Avg reward/ep:</span><span class="param-value">{{ avgRewardPerEp.toFixed(2) }}</span></div>
            <div class="param-item"><span class="param-name">Avg ep length (steps):</span><span class="param-value">{{ avgEpisodeLen.toFixed(1) }}</span></div>
          </div>
          <div class="param-category">
            <h5>Dunks</h5>
            <div class="param-item"><span class="param-name">Attempts:</span><span class="param-value">{{ statsState.dunk.attempts }}</span></div>
            <div class="param-item"><span class="param-name">Made:</span><span class="param-value">{{ statsState.dunk.made }}</span></div>
            <div class="param-item"><span class="param-name">FG%:</span><span class="param-value">{{ (safeDiv(statsState.dunk.made, Math.max(1, statsState.dunk.attempts)) * 100).toFixed(1) }}%</span></div>
            <div class="param-item"><span class="param-name">Assists:</span><span class="param-value">{{ statsState.dunk.assists }}</span></div>
            <div class="param-item"><span class="param-name">Potential assists (missed):</span><span class="param-value">{{ statsState.dunk.potentialAssists }}</span></div>
          </div>
          <div class="param-category">
            <h5>2PT</h5>
            <div class="param-item"><span class="param-name">Attempts:</span><span class="param-value">{{ statsState.twoPt.attempts }}</span></div>
            <div class="param-item"><span class="param-name">Made:</span><span class="param-value">{{ statsState.twoPt.made }}</span></div>
            <div class="param-item"><span class="param-name">FG%:</span><span class="param-value">{{ (safeDiv(statsState.twoPt.made, Math.max(1, statsState.twoPt.attempts)) * 100).toFixed(1) }}%</span></div>
            <div class="param-item"><span class="param-name">Assists:</span><span class="param-value">{{ statsState.twoPt.assists }}</span></div>
            <div class="param-item"><span class="param-name">Potential assists (missed):</span><span class="param-value">{{ statsState.twoPt.potentialAssists }}</span></div>
          </div>
          <div class="param-category">
            <h5>3PT</h5>
            <div class="param-item"><span class="param-name">Attempts:</span><span class="param-value">{{ statsState.threePt.attempts }}</span></div>
            <div class="param-item"><span class="param-name">Made:</span><span class="param-value">{{ statsState.threePt.made }}</span></div>
            <div class="param-item"><span class="param-name">FG%:</span><span class="param-value">{{ (safeDiv(statsState.threePt.made, Math.max(1, statsState.threePt.attempts)) * 100).toFixed(1) }}%</span></div>
            <div class="param-item"><span class="param-name">Assists:</span><span class="param-value">{{ statsState.threePt.assists }}</span></div>
            <div class="param-item"><span class="param-name">Potential assists (missed):</span><span class="param-value">{{ statsState.threePt.potentialAssists }}</span></div>
          </div>
        </div>
        <div style="display:flex; gap: 0.5rem;">
          <button class="new-game-button" @click="resetStats">Reset Stats</button>
          <button class="submit-button" @click="copyStatsMarkdown">Copy</button>
        </div>
      </div>
    </div>

    <!-- Entropy Tab -->
    <div v-if="activeTab === 'entropy'" class="tab-content">
      <div class="entropy-section">
        <h4>Action Entropy</h4>
        <p class="entropy-note">Computed as -∑ p ln p from current policy probabilities.</p>

        <div v-if="!policyProbabilities || !hasEntropyData" class="no-data">
          No policy probabilities available yet.
        </div>
        <div v-else>
          <div class="episode-totals">
            <div class="total-item">
              <span class="team-label offense">Player policy</span>
              <span class="reward-value">
                {{ entropyTotals.playerPolicy !== null ? entropyTotals.playerPolicy.toFixed(3) : '—' }}
              </span>
            </div>
            <div class="total-item">
              <span class="team-label defense">Opponent policy</span>
              <span class="reward-value">
                {{ entropyTotals.opponentPolicy !== null ? entropyTotals.opponentPolicy.toFixed(3) : '—' }}
              </span>
            </div>
          </div>

          <div class="entropy-table-wrapper">
            <table class="entropy-table">
              <thead>
                <tr>
                  <th>Player</th>
                  <th>Team</th>
                  <th>Policy Owner</th>
                  <th>Entropy</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="row in entropyRows" :key="row.playerId">
                  <td>Player {{ row.playerId }}</td>
                  <td>{{ row.teamLabel }}</td>
                  <td>{{ row.policyOwner }}</td>
                  <td>{{ row.entropy !== null && row.entropy !== undefined ? row.entropy.toFixed(3) : '—' }}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>

    <!-- Moves Tab -->
    <div v-if="activeTab === 'moves'" class="tab-content">
      <div class="moves-section">
        <h4>Team Moves History ({{ props.gameState?.user_team_name || 'Unknown' }})</h4>
        <div class="moves-summary">
          <span class="summary-label">Pressure exposure:</span>
          <span class="summary-value">{{ pressureExposureDisplay }}</span>
        </div>
        <div v-if="props.moveHistory.length === 0" class="no-moves">
          No moves recorded yet.
        </div>
        <table v-else class="moves-table">
          <thead>
            <tr>
              <th>Turn</th>
              <th>Shot Clock</th>
              <th v-for="playerId in allPlayerIds" :key="playerId">
                Player {{ playerId }}
              </th>
              <th>Off Value</th>
              <th>Def Value</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="move in props.moveHistory" :key="move.id || move.turn" :class="{ 'current-shot-clock-row': !move.isNoteRow && (move.shotClock === props.currentShotClock || (move.isEndRow && props.gameState?.done)) }">
              <template v-if="move.isNoteRow">
                <td class="moves-note-cell" :colspan="movesColumnCount">
                  {{ move.noteText || 'Note' }}
                </td>
              </template>
              <template v-else>
              <td>{{ move.turn }}</td>
              <td class="shot-clock-cell">{{ move.shotClock !== undefined ? move.shotClock : '-' }}</td>
              <td v-for="playerId in allPlayerIds" :key="playerId" class="move-cell">
                <div class="move-action">
                  <span v-if="move.ballHolder === playerId" class="ball-holder-icon">🏀 </span>
                  <span v-if="move.mctsPlayers && move.mctsPlayers.includes(playerId)" class="mcts-icon" title="Selected via MCTS">🔍 </span>
                  {{ move.moves[`Player ${playerId}`] || 'NOOP' }}
                </div>
                <div v-if="getPassStealProbability(move, playerId) !== null" class="pass-steal-info">
                  ({{ (getPassStealProbability(move, playerId) * 100).toFixed(1) }}% steal risk)
                </div>
                <div v-if="getDefenderPressureProbability(move, playerId) !== null" class="defender-pressure-info">
                  ({{ (getDefenderPressureProbability(move, playerId) * 100).toFixed(1) }}% turnover risk)
                </div>
              </td>
              <td class="value-cell">
                {{ move.offensiveValue !== null && move.offensiveValue !== undefined ? move.offensiveValue.toFixed(3) : '-' }}
              </td>
              <td class="value-cell">
                {{ move.defensiveValue !== null && move.defensiveValue !== undefined ? move.defensiveValue.toFixed(3) : '-' }}
              </td>
              </template>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- Eval Tab -->
    <div v-if="activeTab === 'eval'" class="tab-content eval-tab">
      <div class="eval-row">
        <label class="inline-label">
          <input type="radio" value="default" :checked="!evalModeIsCustom" @change="handleEvalModeChange('default')" />
          Default (current behavior)
        </label>
        <label class="inline-label">
          <input type="radio" value="custom" :checked="evalModeIsCustom" @change="handleEvalModeChange('custom')" />
          Custom pinned setup
        </label>
      </div>

      <div class="eval-row">
        <label>Episodes</label>
        <input
          type="number"
          min="1"
          max="1000000"
          :value="evalEpisodesInput"
          @input="updateEvalEpisodes($event.target.value)"
          :disabled="props.isEvaluating"
        />
        <button
          class="eval-run-btn"
          @click="handleEvalRunClick"
          :disabled="props.isEvaluating || !props.gameState"
        >
          {{ props.isEvaluating ? 'Evaluating…' : 'Run Eval' }}
        </button>
        <span v-if="props.isEvaluating" class="eval-status">
          Running {{ evalEpisodesInput }} episodes…
        </span>
      </div>

      <div class="eval-row">
        <label class="inline-label">
          <input
            type="checkbox"
            :checked="evalConfigSafe.randomizeOffensePermutation"
            @change="setEvalRandomizePermutation($event.target.checked)"
          />
          Randomize offense player slots each episode (shuffle positions)
        </label>
      </div>

      <div v-if="evalModeIsCustom" class="eval-custom">
        <div class="eval-row">
          <label class="inline-label">
            <input type="checkbox" :checked="evalPlacementEditing" @change="toggleEvalPlacement($event.target.checked)" />
            Edit starting positions on board
          </label>
          <button class="ghost-btn" @click="seedEvalConfigFromGameState(false)">
            Use current board positions
          </button>
        </div>

        <div class="eval-row">
          <label>Ball starts with</label>
          <select :value="evalConfigSafe.ballHolder ?? ''" @change="setEvalBallHolder($event.target.value ? Number($event.target.value) : null)">
            <option v-if="!ballStartOptions.length" disabled value="">No offense players</option>
            <option v-for="pid in ballStartOptions" :key="`ball-${pid}`" :value="pid">Player {{ pid }}</option>
          </select>
        </div>

        <div class="eval-row">
          <label>Shooting skills</label>
          <div class="radio-row">
            <label class="inline-label">
              <input type="radio" value="random" :checked="evalConfigSafe.shootingMode === 'random'" @change="setEvalShootingMode('random')" />
              Random each episode
            </label>
            <label class="inline-label">
              <input type="radio" value="fixed" :checked="evalConfigSafe.shootingMode === 'fixed'" @change="setEvalShootingMode('fixed')" />
              Fixed overrides
            </label>
          </div>
        </div>

        <div v-if="evalConfigSafe.shootingMode === 'fixed'" class="eval-skills">
          <div class="skills-header">
            <span>Player</span>
            <span>Layup %</span>
            <span>3PT %</span>
            <span>Dunk %</span>
          </div>
          <div class="skills-row" v-for="(row, idx) in evalOffenseSkillRows" :key="`eval-skill-${row.playerId}`">
            <span class="skills-player">Player {{ row.playerId }}</span>
            <input type="number" min="1" max="99" step="0.1" :value="row.layup" @input="updateEvalSkill(idx, 'layup', $event.target.value)" />
            <input type="number" min="1" max="99" step="0.1" :value="row.threePt" @input="updateEvalSkill(idx, 'three_pt', $event.target.value)" />
            <input type="number" min="1" max="99" step="0.1" :value="row.dunk" @input="updateEvalSkill(idx, 'dunk', $event.target.value)" />
          </div>
          <div class="skills-actions">
            <button class="ghost-btn" @click="seedEvalConfigFromGameState(true)">
              Copy sampled skills
            </button>
          </div>
        </div>
      </div>
    </div>

    <!-- Environment Tab -->
    <div v-if="activeTab === 'environment'" class="tab-content">
      <div class="parameters-section">
        <h4>Environment Parameters</h4>
        <div v-if="!props.gameState" class="no-data">
          No game loaded
        </div>
        <div v-else class="parameters-grid">
          <div class="param-category">
            <h5>Environment Settings</h5>
            <div class="param-item" data-tooltip="Number of players on each team (offense and defense)">
              <span class="param-name">Players per side:</span>
              <span class="param-value">{{ Math.floor((props.gameState.offense_ids?.length || 0)) }}</span>
            </div>
            <div class="param-item" data-tooltip="Width × Height of the hexagonal court grid in hex cells">
              <span class="param-name">Court dimensions:</span>
              <span class="param-value">{{ props.gameState.court_width }}×{{ props.gameState.court_height }}</span>
            </div>
            <div class="param-item" data-tooltip="The player currently in possession of the ball">
              <span class="param-name">Ball holder:</span>
              <span class="param-value">Player {{ props.gameState.ball_holder }}</span>
            </div>
            <div class="param-item" data-tooltip="Remaining steps before shot clock violation (turnover). Decrements each turn.">
              <span class="param-name">Shot clock:</span>
              <span class="param-value">{{ props.gameState.shot_clock }}</span>
            </div>
            <div class="param-item" data-tooltip="Minimum shot clock value when the game resets for a new possession">
              <span class="param-name">Min shot clock at reset:</span>
              <span class="param-value">{{ props.gameState.min_shot_clock ?? 'N/A' }}</span>
            </div>
            <div class="param-item" data-tooltip="Hex distance from basket that marks the three-point line. Shots from this distance or further are worth 3 points.">
              <span class="param-name">Three point distance:</span>
              <span class="param-value">{{ props.gameState.three_point_distance || 'N/A' }}</span>
            </div>
            <div class="param-item" data-tooltip="Shorter three-point distance used for corner areas of the court">
              <span class="param-name">Three point short distance:</span>
              <span class="param-value">{{ props.gameState.three_point_short_distance || 'N/A' }}</span>
            </div>
          </div>

          <div class="param-category">
            <h5>Policies</h5>
            <div class="param-item policy-select-item" data-tooltip="Select the neural network policy controlling the player's team">
              <div class="policy-label">
                Player ({{ props.gameState.user_team_name || 'OFFENSE' }})
              </div>
              <div class="policy-select-wrapper">
                <select
                  :value="userPolicySelection || ''"
                  @change="handlePolicySelection('user', $event)"
                  :disabled="policiesLoading || props.isPolicySwapping"
                >
                  <option v-if="availablePolicies.length === 0 && !userPolicySelection" disabled value="">
                    No policies available
                  </option>
                  <option
                    v-for="policy in availablePolicies"
                    :key="`player-policy-${policy}`"
                    :value="policy"
                  >
                    {{ policy }}
                  </option>
                  <option
                    v-if="userPolicySelection && !availablePolicies.includes(userPolicySelection)"
                    :value="userPolicySelection"
                  >
                    {{ userPolicySelection }} (current)
                  </option>
                </select>
              </div>
            </div>
            <div class="param-item policy-select-item" data-tooltip="Select the neural network policy controlling the opponent team. 'Mirror' uses the same policy as the player.">
              <div class="policy-label">
                Opponent ({{ props.gameState.user_team_name === 'OFFENSE' ? 'DEFENSE' : 'OFFENSE' }})
              </div>
              <div class="policy-select-wrapper">
                <select
                  :value="opponentPolicySelection || ''"
                  @change="handlePolicySelection('opponent', $event)"
                  :disabled="policiesLoading || props.isPolicySwapping"
                >
                  <option value="">Mirror player policy</option>
                  <option
                    v-for="policy in availablePolicies"
                    :key="`opponent-policy-${policy}`"
                    :value="policy"
                  >
                    {{ policy }}
                  </option>
                  <option
                    v-if="opponentPolicySelection && opponentPolicySelection !== '' && !availablePolicies.includes(opponentPolicySelection)"
                    :value="opponentPolicySelection"
                  >
                    {{ opponentPolicySelection }} (current)
                  </option>
                </select>
              </div>
            </div>
            <div class="policy-actions">
              <button 
                class="refresh-policies-btn"
                @click="$emit('refresh-policies')"
                :disabled="policiesLoading || props.isPolicySwapping"
                title="Refresh policy list from MLflow"
              >
                <span v-if="policiesLoading">⟳ Loading...</span>
                <span v-else>⟳ Refresh Policies</span>
              </button>
            </div>
            <div class="policy-status" v-if="policiesLoading">
              Loading policies…
            </div>
            <div class="policy-status error" v-else-if="policyLoadError">
              {{ policyLoadError }}
            </div>
          </div>

          <div class="param-category">
            <h5>Shot Parameters</h5>
            <div v-if="props.gameState.shot_params" class="param-group">
              <div class="param-item" data-tooltip="Mean (average) base probability for layup shots (distance 1-2 hexes from basket)">
                <span class="param-name">Layup &mu;:</span>
                <span class="param-value">{{ (props.gameState.shot_params.layup_pct * 100).toFixed(1) }}%</span>
              </div>
              <div class="param-item" data-tooltip="Standard deviation for sampling individual player layup skill. Higher = more variance between players.">
                <span class="param-name">Layup &sigma;:</span>
                <span class="param-value">{{ (props.gameState.shot_params.layup_std * 100).toFixed(1) }}%</span>
              </div>
              <div class="param-item" data-tooltip="Mean (average) base probability for three-point shots">
                <span class="param-name">Three-point &mu;:</span>
                <span class="param-value">{{ (props.gameState.shot_params.three_pt_pct * 100).toFixed(1) }}%</span>
              </div>
              <div class="param-item" data-tooltip="Standard deviation for sampling individual player three-point skill. Higher = more variance between players.">
                <span class="param-name">Three-point &sigma;:</span>
                <span class="param-value">{{ (props.gameState.shot_params.three_pt_std * 100).toFixed(1) }}%</span>
              </div>
              <div class="param-item" data-tooltip="Mean (average) base probability for dunk shots (distance 0 from basket)">
                <span class="param-name">Dunk &mu;:</span>
                <span class="param-value">{{ (props.gameState.shot_params.dunk_pct * 100).toFixed(1) }}%</span>
              </div>
              <div class="param-item" data-tooltip="Standard deviation for sampling individual player dunk skill. Higher = more variance between players.">
                <span class="param-name">Dunk &sigma;</span>
                <span class="param-value">{{ (props.gameState.shot_params.dunk_std * 100).toFixed(1) }}%</span>
              </div>
              <div class="param-item" data-tooltip="Whether players can attempt dunks when standing on the basket hex">
                <span class="param-name">Dunks allowed:</span>
                <span class="param-value">{{ props.gameState.shot_params.allow_dunks ? '✓ Yes' : '✗ No' }}</span>
              </div>
            </div>
          </div>
          <div class="param-category" v-if="props.gameState.offense_shooting_pct_by_player">
            <h5>Sampled Player Skills (Offense)</h5>
            <div class="offense-skills-editor">
              <div class="offense-skills-row header">
                <span>Player</span>
                <span>Layup %</span>
                <span>3PT %</span>
                <span>Dunk %</span>
              </div>
              <div
                class="offense-skills-row"
                v-for="(row, idx) in offenseSkillRows"
                :key="`skill-${row.playerId}`"
                data-tooltip="Individual shooting percentages sampled from μ±σ distributions. Editable to override per-player skills."
              >
                <span class="skills-player">Player {{ row.playerId }}</span>
                <div class="offense-skill-input">
                  <input type="number" min="1" max="99" step="0.1" v-model.number="offenseSkillInputs.layup[idx]" :disabled="skillsUpdating" />
                  <span class="offense-skill-default">Sampled {{ row.sampledLayup.toFixed(1) }}%</span>
                </div>
                <div class="offense-skill-input">
                  <input type="number" min="1" max="99" step="0.1" v-model.number="offenseSkillInputs.three_pt[idx]" :disabled="skillsUpdating" />
                  <span class="offense-skill-default">Sampled {{ row.sampledThree.toFixed(1) }}%</span>
                </div>
                <div class="offense-skill-input">
                  <input type="number" min="1" max="99" step="0.1" v-model.number="offenseSkillInputs.dunk[idx]" :disabled="skillsUpdating" />
                  <span class="offense-skill-default">Sampled {{ row.sampledDunk.toFixed(1) }}%</span>
                </div>
              </div>
              <div class="offense-skill-actions">
                <button 
                  class="refresh-policies-btn" 
                  @click="resetOffenseSkillsToSampled" 
                  :disabled="skillsUpdating || !offenseSkillRows.length"
                  title="Reset to the values sampled when this game was created"
                >
                  Reset to Sampled
                </button>
                <button 
                  class="refresh-policies-btn" 
                  @click="applyOffenseSkillOverrides" 
                  :disabled="skillsUpdating || !offenseSkillRows.length"
                >
                  {{ skillsUpdating ? 'Saving...' : 'Apply Overrides' }}
                </button>
              </div>
              <div class="policy-status error" v-if="skillsError">
                {{ skillsError }}
              </div>
            </div>
          </div>
          <div class="param-category">
            <h5>Defender Turnover Pressure</h5>
            <div class="param-item" data-tooltip="Maximum hex distance at which defenders can apply turnover pressure to the ball handler">
              <span class="param-name">Pressure distance:</span>
              <span class="param-value">{{ props.gameState.defender_pressure_distance || 'N/A' }}</span>
            </div>
            <div class="param-item" data-tooltip="Base probability of turnover when a defender is adjacent (distance=1) to ball handler">
              <span class="param-name">Turnover chance:</span>
              <span class="param-value">{{ props.gameState.defender_pressure_turnover_chance || 'N/A' }}</span>
            </div>
            <div class="param-item" data-tooltip="Exponential decay rate for pressure. Higher values = pressure drops off faster with distance.">
              <span class="param-name">Decay lambda:</span>
              <span class="param-value">{{ props.gameState.defender_pressure_decay_lambda || 'N/A' }}</span>
            </div>
          </div>
          <div class="param-category">
            <h5>Pass Interception (Line-of-Sight)</h5>
            <div class="param-item" data-tooltip="Base probability that a defender intercepts a pass when directly on the pass line">
              <span class="param-name">Base steal rate:</span>
              <span class="param-value">{{ props.gameState.base_steal_rate ?? 'N/A' }}</span>
            </div>
            <div class="param-item" data-tooltip="How quickly steal probability drops as defender is further from pass line. Higher = faster decay.">
              <span class="param-name">Perpendicular decay:</span>
              <span class="param-value">{{ props.gameState.steal_perp_decay ?? 'N/A' }}</span>
            </div>
            <div class="param-item" data-tooltip="How pass distance affects interception chance. Longer passes are easier to intercept.">
              <span class="param-name">Distance factor:</span>
              <span class="param-value">{{ props.gameState.steal_distance_factor ?? 'N/A' }}</span>
            </div>
            <div class="param-item" data-tooltip="Minimum weight for defender's position along pass line (0=near passer, 1=near receiver)">
              <span class="param-name">Position weight min:</span>
              <span class="param-value">{{ props.gameState.steal_position_weight_min ?? 'N/A' }}</span>
            </div>
          </div>
          <div class="param-category">
            <h5>Spawn Distance</h5>
            <div class="param-item" data-tooltip="Minimum hex distance from basket where players spawn at episode start">
              <span class="param-name">Min spawn distance:</span>
              <span class="param-value">{{ props.gameState.spawn_distance || 'N/A' }}</span>
            </div>
            <div class="param-item" data-tooltip="Maximum hex distance from basket where players spawn. 'Unlimited' means anywhere on court.">
              <span class="param-name">Max spawn distance:</span>
              <span class="param-value">{{ props.gameState.max_spawn_distance ?? 'Unlimited' }}</span>
            </div>
            <div class="param-item" data-tooltip="Randomize defender spawn distance from matched offense player (0 = spawn adjacent; N = spawn 1-N hexes away)">
              <span class="param-name">Defender spawn distance:</span>
              <span class="param-value">{{ props.gameState.defender_spawn_distance || 'N/A' }}</span>
            </div>
            <div class="param-item" data-tooltip="Keep offense spawns away from the court boundary by this many hex rings (0 = no restriction)">
              <span class="param-name">Offense boundary margin:</span>
              <span class="param-value">{{ props.gameState.offense_spawn_boundary_margin ?? 'N/A' }}</span>
            </div>
            <div class="param-item" data-tooltip="Hex distance (N) within which a defender reset their lane counter if guarding an offensive player while in the lane. 0 disables guarding resets.">
              <span class="param-name">Defender guard distance:</span>
              <span class="param-value">{{ props.gameState.defender_guard_distance || 'N/A' }}</span>
            </div>
          </div>
          <div class="param-category">
            <h5>Shot Pressure</h5>
            <div class="param-item" data-tooltip="Whether nearby defenders reduce shot accuracy">
              <span class="param-name">Pressure enabled:</span>
              <span class="param-value">{{ props.gameState.shot_pressure_enabled || 'N/A' }}</span>
            </div>
            <div class="param-item" data-tooltip="Maximum percentage reduction in shot accuracy from defender pressure (e.g., 0.3 = up to 30% reduction)">
              <span class="param-name">Max pressure:</span>
              <span class="param-value">{{ props.gameState.shot_pressure_max || 'N/A' }}</span>
            </div>
            <div class="param-item" data-tooltip="Exponential decay rate for shot pressure over distance. Higher = pressure drops faster.">
              <span class="param-name">Pressure lambda:</span>
              <span class="param-value">{{ props.gameState.shot_pressure_lambda || 'N/A' }}</span>
            </div>
            <div class="param-item" data-tooltip="Angular width of the defensive pressure cone. Defenders outside this arc apply less pressure.">
              <span class="param-name">Pressure arc degrees:</span>
              <span class="param-value">{{ props.gameState.shot_pressure_arc_degrees || 'N/A' }}°</span>
            </div>
          </div>

          <div class="param-category">
            <h5>Pass & Action Policy</h5>
            <div class="param-item" data-tooltip="Angular width of valid pass directions. Passes outside this arc from the intended direction fail.">
              <span class="param-name">Pass arc degrees:</span>
              <span class="param-value">{{ props.gameState.pass_arc_degrees || 'N/A' }}°</span>
            </div>
            <div class="param-item" data-tooltip="Receiver selection when multiple teammates are in the pass arc.">
              <span class="param-name">Pass target strategy:</span>
              <div class="param-select-wrapper">
                <select
                  :value="passTargetStrategyValue"
                  @change="handlePassTargetStrategyChange"
                  :disabled="passStrategyUpdating || !props.gameState || props.gameState.done"
                >
                  <option
                    v-for="opt in PASS_TARGET_STRATEGIES"
                    :key="opt.value"
                    :value="opt.value"
                  >
                    {{ opt.label }}
                  </option>
                </select>
              </div>
              <div v-if="passStrategyError" class="policy-status error">
                {{ passStrategyError }}
              </div>
            </div>
            <div class="param-item" data-tooltip="Probability of turnover when a pass goes out of bounds (no teammate in direction)">
              <span class="param-name">Pass OOB turnover prob:</span>
              <span class="param-value">{{ props.gameState.pass_oob_turnover_prob != null ? (props.gameState.pass_oob_turnover_prob * 100).toFixed(0) + '%' : 'N/A' }}</span>
            </div>
            <div class="param-item" data-tooltip="Bias added to pass action logits in the policy network. Positive = encourage passing.">
              <span class="param-name">Pass logit bias:</span>
            </div>
            <div class="offense-skills-editor">
              <div class="offense-skills-row header">
                <span>Setting</span>
                <span>Bias</span>
                <span></span>
                <span></span>
              </div>
              <div class="offense-skills-row">
                <span class="skills-player">Pass logit bias</span>
                <div class="offense-skill-input">
                  <input type="number" step="0.05" v-model.number="passLogitBiasInput" :disabled="passLogitBiasUpdating" />
                  <span class="offense-skill-default">Default {{ passLogitBiasDefault.toFixed(2) }}</span>
                </div>
                <span></span>
                <span></span>
              </div>
              <div class="offense-skill-actions">
                <button
                  class="refresh-policies-btn"
                  @click="resetPassLogitBiasDefault"
                  :disabled="passLogitBiasUpdating"
                >
                  Reset to Default
                </button>
                <button
                  class="refresh-policies-btn"
                  @click="applyPassLogitBiasOverride"
                  :disabled="passLogitBiasUpdating"
                >
                  {{ passLogitBiasUpdating ? 'Saving...' : 'Apply Bias' }}
                </button>
              </div>
              <div class="policy-status error" v-if="passLogitBiasError">
                {{ passLogitBiasError }}
              </div>
            </div>
          </div>

          <div class="param-category">
            <h5>Team Configuration</h5>
            <div class="param-item" data-tooltip="Which team the user/player controls (offense or defense)">
              <span class="param-name">User team:</span>
              <span class="param-value">{{ props.gameState.user_team_name }}</span>
            </div>
            <div class="param-item" data-tooltip="Player IDs assigned to the offensive team">
              <span class="param-name">Offense IDs:</span>
              <span class="param-value">{{ props.gameState.offense_ids?.join(', ') || 'N/A' }}</span>
            </div>
            <div class="param-item" data-tooltip="Player IDs assigned to the defensive team">
              <span class="param-name">Defense IDs:</span>
              <span class="param-value">{{ props.gameState.defense_ids?.join(', ') || 'N/A' }}</span>
            </div>
          </div>

          <div class="param-category">
            <h5>3-Second Violation Rules</h5>
            <div class="param-item" data-tooltip="Width of the paint/lane area in hex cells, centered on the basket">
              <span class="param-name">Lane width (hexes):</span>
              <span class="param-value">{{ props.gameState.three_second_lane_width ?? 'N/A' }}</span>
            </div>
            <div class="param-item" data-tooltip="Height/depth of the paint/lane area extending from the baseline">
              <span class="param-name">Lane height (hexes):</span>
              <span class="param-value">{{ props.gameState.three_second_lane_height ?? 'N/A' }}</span>
            </div>
            <div class="param-item" data-tooltip="Maximum consecutive steps a player can stay in the lane before a violation">
              <span class="param-name">Max steps in lane:</span>
              <span class="param-value">{{ props.gameState.three_second_max_steps ?? 'N/A' }}</span>
            </div>
            <div class="param-item" data-tooltip="Whether offensive players get violations for staying in the paint too long (turnover)">
              <span class="param-name">Offensive 3-sec enabled:</span>
              <span class="param-value">{{ props.gameState.offensive_three_seconds_enabled ? '✓ Yes' : '✗ No' }}</span>
            </div>
            <div class="param-item" data-tooltip="Whether defensive players get violations for camping in the paint without guarding (technical foul)">
              <span class="param-name">Illegal defense enabled:</span>
              <span class="param-value">{{ props.gameState.illegal_defense_enabled ? '✓ Yes' : '✗ No' }}</span>
            </div>
            <div class="param-item" v-if="props.gameState.offensive_lane_hexes" data-tooltip="Total number of hex cells that make up the painted lane area">
              <span class="param-name">Lane hexes count:</span>
              <span class="param-value">{{ props.gameState.offensive_lane_hexes?.length || 0 }} hexes</span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Training Tab -->
    <div v-if="activeTab === 'training'" class="tab-content">
      <div class="parameters-section">
        <h4>Training Hyperparameters</h4>
        <div v-if="!props.gameState || !props.gameState.training_params" class="no-data">
          No training parameters available
        </div>
        <div v-else class="parameters-grid">
          <div class="param-category">
            <h5>PPO Core</h5>
            <div class="param-item" data-tooltip="Step size for gradient descent. Lower = slower but more stable training.">
              <span class="param-name">Learning rate:</span>
              <span class="param-value">{{ props.gameState.training_params.learning_rate?.toExponential(2) || 'N/A' }}</span>
            </div>
            <div class="param-item" data-tooltip="Number of environment steps collected per update. Higher = more stable gradients but slower iteration.">
              <span class="param-name">N steps:</span>
              <span class="param-value">{{ props.gameState.training_params.n_steps || 'N/A' }}</span>
            </div>
            <div class="param-item" data-tooltip="Number of samples per gradient update. Must divide n_steps × num_envs evenly.">
              <span class="param-name">Batch size:</span>
              <span class="param-value">{{ props.gameState.training_params.batch_size || 'N/A' }}</span>
            </div>
            <div class="param-item" data-tooltip="Number of passes through collected data per update. More epochs = better sample efficiency but risk of overfitting.">
              <span class="param-name">N epochs:</span>
              <span class="param-value">{{ props.gameState.training_params.n_epochs || 'N/A' }}</span>
            </div>
            <div class="param-item" data-tooltip="Discount factor for future rewards. Higher (closer to 1) = considers longer-term consequences.">
              <span class="param-name">Gamma (γ):</span>
              <span class="param-value">{{ props.gameState.training_params.gamma || 'N/A' }}</span>
            </div>
            <div class="param-item" data-tooltip="GAE lambda for advantage estimation. Higher = lower bias but higher variance.">
              <span class="param-name">GAE Lambda:</span>
              <span class="param-value">{{ props.gameState.training_params.gae_lambda || 'N/A' }}</span>
            </div>
            <div class="param-item" data-tooltip="PPO clipping parameter. Limits how much the policy can change per update.">
              <span class="param-name">Clip range:</span>
              <span class="param-value">{{ props.gameState.training_params.clip_range || 'N/A' }}</span>
            </div>
          </div>

          <div class="param-category">
            <h5>Loss Coefficients</h5>
            <div class="param-item" data-tooltip="Weight for value function loss in total loss. Higher = more emphasis on accurate value predictions.">
              <span class="param-name">VF coefficient:</span>
              <span class="param-value">{{ props.gameState.training_params.vf_coef || 'N/A' }}</span>
            </div>
            <div class="param-item" data-tooltip="Weight for entropy bonus. Higher = more exploration. Can be scheduled.">
              <span class="param-name">Entropy coefficient:</span>
              <span class="param-value">{{ props.gameState.training_params.ent_coef ?? 'N/A' }}</span>
            </div>
            <div class="param-item" v-if="props.gameState.training_params.ent_coef_start != null" data-tooltip="Starting value for entropy coefficient schedule.">
              <span class="param-name">Entropy start:</span>
              <span class="param-value">{{ props.gameState.training_params.ent_coef_start }}</span>
            </div>
            <div class="param-item" v-if="props.gameState.training_params.ent_coef_end != null" data-tooltip="Ending value for entropy coefficient schedule.">
              <span class="param-name">Entropy end:</span>
              <span class="param-value">{{ props.gameState.training_params.ent_coef_end }}</span>
            </div>
          </div>

          <div class="param-category">
            <h5>Network Architecture</h5>
            <div class="param-item" data-tooltip="Type of policy network. Dual critic has separate value heads for offense/defense.">
              <span class="param-name">Policy class:</span>
              <span class="param-value policy-class-value">{{ props.gameState.training_params.policy_class || 'N/A' }}</span>
            </div>
            <div class="param-item" data-tooltip="Whether using dual critic architecture (separate offense/defense value heads).">
              <span class="param-name">Dual critic:</span>
              <span class="param-value">{{ props.gameState.training_params.use_dual_critic ? '✓ Yes' : '✗ No' }}</span>
            </div>
            <div class="param-item" v-if="props.gameState.training_params.net_arch_used" data-tooltip="Actual network architecture used (logged after policy creation). Shows pi/vf layer sizes.">
              <span class="param-name">Net arch (actual):</span>
              <span class="param-value policy-class-value">{{ props.gameState.training_params.net_arch_used }}</span>
            </div>
            <div class="param-item" v-if="props.gameState.training_params.net_arch_pi" data-tooltip="Actor (policy) network hidden layer sizes from CLI args.">
              <span class="param-name">Net arch π:</span>
              <span class="param-value">{{ props.gameState.training_params.net_arch_pi }}</span>
            </div>
            <div class="param-item" v-if="props.gameState.training_params.net_arch_vf" data-tooltip="Critic (value) network hidden layer sizes from CLI args.">
              <span class="param-name">Net arch vf:</span>
              <span class="param-value">{{ props.gameState.training_params.net_arch_vf }}</span>
            </div>
            <div class="param-item" v-if="props.gameState.training_params.net_arch && !props.gameState.training_params.net_arch_used" data-tooltip="Shared network architecture (both actor and critic).">
              <span class="param-name">Net arch:</span>
              <span class="param-value">{{ props.gameState.training_params.net_arch }}</span>
            </div>
            <div class="param-item" v-if="paramCounts" data-tooltip="Total trainable parameters across trunk and heads.">
              <span class="param-name">Params (total):</span>
              <span class="param-value">{{ formatParamCount(paramCounts.total) }}</span>
            </div>
            <div class="param-item" v-if="paramCounts" data-tooltip="Shared trunk (features + MLP extractor) parameters.">
              <span class="param-name">Params (shared trunk):</span>
              <span class="param-value">{{ formatParamCount(paramCounts.shared_trunk) }}</span>
            </div>
            <div class="param-item" v-if="paramCounts" data-tooltip="Policy head parameters (including log_std).">
              <span class="param-name">Params (policy):</span>
              <span class="param-value">{{ formatParamCount(paramCounts.policy_heads) }}</span>
            </div>
            <div class="param-item" v-if="paramCounts" data-tooltip="Value head parameters (offense/defense critics).">
              <span class="param-name">Params (value):</span>
              <span class="param-value">{{ formatParamCount(paramCounts.value_heads) }}</span>
            </div>
          </div>

          <div class="param-category">
            <h5>Training Setup</h5>
            <div class="param-item" data-tooltip="Number of parallel environments used during training.">
              <span class="param-name">Num envs:</span>
              <span class="param-value">{{ props.gameState.training_params.num_envs || 'N/A' }}</span>
            </div>
            <div class="param-item" data-tooltip="Number of times to alternate training (each alternation loads a new opponent).">
              <span class="param-name">Alternations:</span>
              <span class="param-value">{{ props.gameState.training_params.alternations || 'N/A' }}</span>
            </div>
            <div class="param-item" data-tooltip="Number of PPO updates per alternation. Can be scheduled from start to end value.">
              <span class="param-name">Steps per alternation:</span>
              <span class="param-value" v-if="props.gameState.training_params.steps_per_alternation_end && props.gameState.training_params.steps_per_alternation_end !== props.gameState.training_params.steps_per_alternation">
                {{ props.gameState.training_params.steps_per_alternation }} → {{ props.gameState.training_params.steps_per_alternation_end }}
              </span>
              <span class="param-value" v-else>{{ props.gameState.training_params.steps_per_alternation ?? 'N/A' }}</span>
            </div>
            <div class="param-item" v-if="props.gameState.training_params.steps_per_alternation_end && props.gameState.training_params.steps_per_alternation_end !== props.gameState.training_params.steps_per_alternation" data-tooltip="Schedule type for steps per alternation: linear interpolates from start to end.">
              <span class="param-name">SPA schedule:</span>
              <span class="param-value">{{ props.gameState.training_params.steps_per_alternation_schedule || 'linear' }}</span>
            </div>
            <div class="param-item" data-tooltip="Timesteps per alternation = steps_per_alternation × num_envs × n_steps">
              <span class="param-name">Timesteps/alternation:</span>
              <span class="param-value">{{ props.gameState.training_params.steps_per_alternation && props.gameState.training_params.num_envs && props.gameState.training_params.n_steps ? ((props.gameState.training_params.steps_per_alternation * props.gameState.training_params.num_envs * props.gameState.training_params.n_steps) / 1e6).toFixed(2) + 'M' : 'N/A' }}</span>
            </div>
            <div class="param-item" data-tooltip="Total planned timesteps for training (exact value when using SPA schedule).">
              <span class="param-name">Total timesteps:</span>
              <span class="param-value" v-if="props.gameState.training_params.total_timesteps_planned">
                {{ (props.gameState.training_params.total_timesteps_planned / 1e6).toFixed(2) + 'M' }}
              </span>
              <span class="param-value" v-else>
                {{ props.gameState.training_params.alternations && props.gameState.training_params.steps_per_alternation && props.gameState.training_params.num_envs && props.gameState.training_params.n_steps ? ((props.gameState.training_params.alternations * props.gameState.training_params.steps_per_alternation * props.gameState.training_params.num_envs * props.gameState.training_params.n_steps) / 1e6).toFixed(1) + 'M' : 'N/A' }}
              </span>
            </div>
          </div>

          <div class="param-category">
            <h5>Self-Play & Opponents</h5>
            <div class="param-item" data-tooltip="Whether opponent uses deterministic action selection during training.">
              <span class="param-name">Deterministic opponent:</span>
              <span class="param-value">{{ props.gameState.training_params.deterministic_opponent ? '✓ Yes' : '✗ No' }}</span>
            </div>
            <div class="param-item" data-tooltip="Whether each parallel env samples different opponents (prevents forgetting).">
              <span class="param-name">Per-env opponent sampling:</span>
              <span class="param-value">{{ props.gameState.training_params.per_env_opponent_sampling ? '✓ Yes' : '✗ No' }}</span>
            </div>
            <div class="param-item" v-if="props.gameState.training_params.per_env_opponent_sampling" data-tooltip="Number of recent checkpoints to sample opponents from.">
              <span class="param-name">Opponent sample K:</span>
              <span class="param-value">{{ props.gameState.training_params.opponent_sample_k || 'N/A' }}</span>
            </div>
            <div class="param-item" v-if="props.gameState.training_params.opponent_pool_beta" data-tooltip="Geometric distribution parameter for opponent sampling. Higher = more recency bias.">
              <span class="param-name">Opponent pool beta:</span>
              <span class="param-value">{{ props.gameState.training_params.opponent_pool_beta || 'N/A' }}</span>
            </div>
            <div class="param-item" v-if="props.gameState.training_params.opponent_pool_exploration" data-tooltip="Geometric distribution parameter for opponent sampling. Higher = more recency bias.">
              <span class="param-name">Opponent pool exploration:</span>
              <span class="param-value">{{ props.gameState.training_params.opponent_pool_exploration || 'N/A' }}</span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Phi Shaping Tab -->
    <div v-if="activeTab === 'phi'" class="tab-content">
      <PhiShaping ref="phiRef" :game-state="props.gameState" />
    </div>

    <!-- Observation Tab -->
    <div v-if="activeTab === 'observation'" class="tab-content">
      <div class="observation-section">
        <h4>Current Observation Features</h4>
        <div v-if="!props.gameState || !props.gameState.obs" class="no-data">
          No observation data available.
        </div>
        <div v-else class="observation-table-wrapper">
          <table class="observation-table">
            <thead>
              <tr>
                <th>Feature Group</th>
                <th>Element</th>
                <th>Value</th>
                <th>Notes</th>
              </tr>
            </thead>
            <tbody>
              <!-- Player Positions -->
              <tr v-for="(pos, idx) in playerPositionRows" :key="`pos-${idx}`" class="group-player-pos">
                <td v-if="idx === 0" :rowspan="playerPositionRows.length" class="group-label">Player Positions (absolute)</td>
                <td>Player {{ Math.floor(idx / 2) }} - {{ idx % 2 === 0 ? 'Q' : 'R' }}</td>
                <td class="value-mono">{{ pos.toFixed(4) }}</td>
                <td class="notes">{{ idx % 2 === 0 ? 'Column' : 'Row' }}</td>
              </tr>

              <!-- Ball Holder One-Hot -->
              <tr v-for="(val, idx) in ballHolderOHE" :key="`bh-${idx}`" class="group-ball-holder">
                <td v-if="idx === 0" :rowspan="ballHolderOHE.length" class="group-label">Ball Holder (one-hot)</td>
                <td>Player {{ idx }}</td>
                <td class="value-mono">{{ val }}</td>
                <td v-if="val === 1" class="notes highlight">🏀 Ball holder</td>
                <td v-else class="notes"></td>
              </tr>

              <!-- Shot Clock -->
              <tr class="group-shot-clock">
                <td class="group-label">Shot Clock</td>
                <td>-</td>
                <td class="value-mono">{{ shotClockValue }}</td>
                <td class="notes">Current shot clock</td>
              </tr>

              <!-- Team Encoding -->
              <tr v-for="(team, idx) in teamEncodingRows" :key="`team-${idx}`" class="group-team-encoding">
                <td v-if="idx === 0" :rowspan="teamEncodingRows.length" class="group-label">Team Encoding</td>
                <td>Player {{ idx }}</td>
                <td class="value-mono">{{ team > 0 ? '+1' : '-1' }}</td>
                <td class="notes">{{ team > 0 ? '🏀 Offense' : '🛡️ Defense' }}</td>
              </tr>

              <!-- Ball Handler Position -->
              <tr v-for="(val, idx) in ballHandlerPositionRows" :key="`bhpos-${idx}`" class="group-ball-handler-pos">
                <td v-if="idx === 0" :rowspan="ballHandlerPositionRows.length" class="group-label">Ball Handler Position (absolute)</td>
                <td>{{ idx === 0 ? 'Q' : 'R' }}</td>
                <td class="value-mono">{{ val.toFixed(4) }}</td>
                <td class="notes">{{ idx === 0 ? 'Column' : 'Row' }} of ball handler</td>
              </tr>

              <!-- Hoop Vector -->
              <tr v-for="(val, idx) in hoopVectorRows" :key="`hoop-${idx}`" class="group-hoop">
                <td v-if="idx === 0" :rowspan="hoopVectorRows.length" class="group-label">Hoop Vector (absolute)</td>
                <td>{{ idx === 0 ? 'Q' : 'R' }}</td>
                <td class="value-mono">{{ val.toFixed(4) }}</td>
                <td class="notes">Basket position</td>
              </tr>

              <!-- All-Pairs Distances -->
              <tr v-for="(dist, idx) in allPairsDistances" :key="`dist-${idx}`" class="group-distances">
                <td v-if="idx === 0" :rowspan="allPairsDistances.length" class="group-label">All-Pairs Distances</td>
                <td>O{{ formatOffenseId(Math.floor(idx / numDefenders)) }} → D{{ formatDefenseId(idx % numDefenders) }}</td>
                <td class="value-mono">{{ dist.toFixed(4) }}</td>
                <td class="notes">Hex distance</td>
              </tr>

              <!-- All-Pairs Angles -->
              <tr v-for="(angle, idx) in allPairsAngles" :key="`angle-${idx}`" class="group-angles">
                <td v-if="idx === 0" :rowspan="allPairsAngles.length" class="group-label">All-Pairs Angles (signed)</td>
                <td>O{{ formatOffenseId(Math.floor(idx / numDefenders)) }} → D{{ formatDefenseId(idx % numDefenders) }}</td>
                <td class="value-mono">{{ formatAngleValue(angle) }}</td>
                <td class="notes">{{ getAngleDescription(angle) }}</td>
              </tr>

              <!-- Teammate Distances -->
              <tr
                v-for="(dist, idx) in teammateDistances"
                :key="`team-dist-${idx}`"
                class="group-teammate-distances"
              >
                <td
                  v-if="idx === 0"
                  :rowspan="teammateDistances.length"
                  class="group-label"
                >
                  Teammate Distances
                </td>
                <td>{{ teammateDistanceLabels[idx] || 'Teammate spacing' }}</td>
                <td class="value-mono">{{ dist.toFixed(4) }}</td>
                <td class="notes">Team spacing</td>
              </tr>

              <!-- Teammate Angles -->
              <tr
                v-for="(angle, idx) in teammateAngles"
                :key="`team-angle-${idx}`"
                class="group-teammate-angles"
              >
                <td
                  v-if="idx === 0"
                  :rowspan="teammateAngles.length"
                  class="group-label"
                >
                  Teammate Angles (signed)
                </td>
                <td>{{ teammateAngleLabels[idx] || 'Teammate direction' }}</td>
                <td class="value-mono">{{ formatAngleValue(angle) }}</td>
                <td class="notes">{{ getAngleDescription(angle) }}</td>
              </tr>

              <!-- Lane Steps -->
              <tr v-for="(steps, idx) in laneSteps" :key="`lane-${idx}`" class="group-lane-steps">
                <td v-if="idx === 0" :rowspan="laneSteps.length" class="group-label">Lane Steps</td>
                <td>Player {{ idx }}</td>
                <td class="value-mono">{{ steps }}</td>
                <td class="notes">Time in lane</td>
              </tr>

              <!-- Expected Points -->
              <tr v-for="(ep, idx) in expectedPoints" :key="`ep-${idx}`" class="group-ep">
                <td v-if="idx === 0" :rowspan="expectedPoints.length" class="group-label">Expected Points (EP)</td>
                <td>O{{ formatOffenseId(idx) }}</td>
                <td class="value-mono">{{ ep.toFixed(4) }}</td>
                <td class="notes">Shot quality estimate</td>
              </tr>

              <!-- Turnover Probabilities -->
              <tr v-for="(prob, idx) in turnoverProbs" :key="`turnover-${idx}`" class="group-turnover">
                <td v-if="idx === 0" :rowspan="turnoverProbs.length" class="group-label">Turnover Probs</td>
                <td>O{{ formatOffenseId(idx) }}</td>
                <td class="value-mono">{{ prob.toFixed(4) }}</td>
                <td v-if="prob > 0" class="notes highlight">🚨 Risk</td>
                <td v-else class="notes">No risk</td>
              </tr>

              <!-- Steal Risks -->
              <tr v-for="(risk, idx) in stealRisks" :key="`steal-${idx}`" class="group-steal">
                <td v-if="idx === 0" :rowspan="stealRisks.length" class="group-label">Steal Risks</td>
                <td>O{{ formatOffenseId(idx) }}</td>
                <td class="value-mono">{{ risk.toFixed(4) }}</td>
                <td v-if="risk > 0" class="notes highlight">⚠️ Risk</td>
                <td v-else class="notes">Safe</td>
              </tr>
            </tbody>
          </table>
        </div>

      </div>
    </div>

    <!-- Attention Tab -->
    <div v-if="activeTab === 'attention'" class="tab-content">
      <div class="observation-section">
        <div class="token-section">
          <h4>Token View (Set-Observation)</h4>
          <div v-if="!obsTokens" class="no-data">
            No token data available.
          </div>
          <div v-else class="token-table-wrapper">
            <div class="token-table-scroll">
              <table class="observation-table token-table">
                <thead>
                  <tr>
                    <th>Player</th>
                    <th>Team</th>
                    <th v-for="label in tokenFeatureLabels" :key="`token-head-${label}`">
                      {{ label }}
                    </th>
                  </tr>
                </thead>
                <tbody>
                  <tr
                    v-for="row in tokenRows"
                    :key="`token-${row.playerId}`"
                    :class="{ 'token-ball-holder': row.playerId === props.gameState.ball_holder }"
                  >
                    <td>Player {{ row.playerId }}</td>
                    <td>{{ row.teamLabel }}</td>
                    <td
                      v-for="(label, fIdx) in tokenFeatureLabels"
                      :key="`token-${row.playerId}-${label}`"
                      class="value-mono"
                    >
                      {{ formatTokenValue(row.features[fIdx]) }}
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>

            <h5 class="token-subtitle">Globals</h5>
            <table class="observation-table token-table">
              <thead>
                <tr>
                  <th>Global</th>
                  <th>Value</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="row in tokenGlobalRows" :key="`token-global-${row.label}`">
                  <td>{{ row.label }}</td>
                  <td class="value-mono">{{ formatTokenValue(row.value) }}</td>
                </tr>
              </tbody>
            </table>

            <div class="token-attn-section">
              <h5 class="token-subtitle">Attention Map</h5>
              <div v-if="tokenAttentionMatrix.length === 0" class="no-data">
                Attention weights are not available.
              </div>
              <div v-else class="token-table-wrapper">
                <div class="token-attn-controls" v-if="tokenAttentionHeads">
                  <label for="token-attn-view">View:</label>
                  <select id="token-attn-view" v-model="attentionView">
                    <option value="avg">Average</option>
                    <option v-for="idx in attentionHeadOptions" :key="`head-${idx}`" :value="String(idx)">
                      Head {{ idx + 1 }}
                    </option>
                  </select>
                  <button class="token-attn-download" type="button" @click="downloadAttentionPng">
                    Download PNG
                  </button>
                </div>
                <div class="token-attn-note" v-if="tokenAttentionSubtitle">
                  {{ tokenAttentionSubtitle }}
                </div>
                <div class="token-table-scroll">
                  <table class="observation-table token-table token-attn-table">
                    <thead>
                      <tr>
                        <th>From \ To</th>
                        <th v-for="label in tokenAttentionLabels" :key="`attn-head-${label}`">
                          {{ label }}
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr v-for="(row, rIdx) in tokenAttentionMatrix" :key="`attn-row-${rIdx}`">
                        <td>{{ tokenAttentionLabels[rIdx] || `T${rIdx}` }}</td>
                        <td
                          v-for="(val, cIdx) in row"
                          :key="`attn-${rIdx}-${cIdx}`"
                          class="value-mono"
                          :style="attentionCellStyle(val)"
                        >
                          {{ formatTokenValue(val) }}
                        </td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.player-controls-container {
  display: flex;
  flex-direction: column;
  height: 100%;
  min-height: 400px;
}

.controls-wrapper {
  display: flex;
  flex-direction: column;
  flex: 1;
  overflow-y: auto;
}

.tab-navigation {
  display: flex;
  gap: 0.5rem;
  border-bottom: 1px solid var(--app-panel-border);
  margin-bottom: 1rem;
  flex-wrap: wrap;
}

.tab-navigation button {
  padding: 0.5rem 1rem;
  border: 1px solid var(--app-panel-border);
  background-color: transparent;
  cursor: pointer;
  border-bottom: 1px solid transparent;
  font-weight: 500;
  color: var(--app-text-muted);
  transition: all 0.2s ease;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  font-size: 1rem;
}

.tab-navigation button:hover {
  color: var(--app-text);
  background-color: rgba(255, 255, 255, 0.03);
}

.tab-navigation button.active {
  border-bottom-color: var(--app-accent);
  color: var(--app-accent);
}

.tab-content {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

/* Player tabs */
.player-tabs {
  display: flex;
  gap: 0.5rem;
  flex-wrap: wrap;
  margin-bottom: 0.5rem;
}

.player-tabs button {
  padding: 0.4rem 0.9rem;
  border: 1px solid var(--app-panel-border);
  background-color: rgba(15, 23, 42, 0.4);
  color: var(--app-text-muted);
  cursor: pointer;
  border-radius: 999px;
  transition: all 0.2s ease;
  font-size: 0.8rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.player-tabs button:hover:not(:disabled) {
  border-color: var(--app-accent);
  color: var(--app-accent);
}

.player-tabs button.active {
  background-color: rgba(56, 189, 248, 0.15);
  color: var(--app-accent);
  border-color: var(--app-accent);
  box-shadow: 0 0 10px rgba(56, 189, 248, 0.2);
}

.player-tabs button:disabled {
  background-color: rgba(0, 0, 0, 0.2);
  color: var(--app-text-muted);
  opacity: 0.5;
  cursor: not-allowed;
  border-color: transparent;
}

/* Advisor */
.advisor-tab {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.advisor-table-wrapper {
  overflow-x: auto;
}

.advisor-table {
  width: 100%;
  border-collapse: collapse;
  background: rgba(255, 255, 255, 0.02);
  border: 1px solid var(--app-panel-border);
  border-radius: 12px;
}

.advisor-table th,
.advisor-table td {
  padding: 10px;
  border-bottom: 1px solid var(--app-panel-border);
  text-align: left;
  font-size: 14px;
}

.advisor-table th {
  color: var(--app-text-muted);
  font-weight: 600;
}

.advisor-table tr:last-child td {
  border-bottom: none;
}

.policy-row {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 6px;
}

.policy-chip {
  background: rgba(56, 189, 248, 0.12);
  color: var(--app-accent);
  padding: 2px 8px;
  border-radius: 999px;
  font-size: 12px;
}

.advisor-stat-cell {
  display: flex;
  flex-direction: column;
  gap: 2px;
  color: var(--app-text-muted);
}

.advisor-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
  gap: 10px;
  align-items: end;
}

.advisor-grid label {
  font-size: 14px;
  display: flex;
  flex-direction: column;
  gap: 4px;
  color: var(--app-text-muted);
}

.advisor-grid input,
.advisor-grid select {
  padding: 8px 10px;
  border: 1px solid var(--app-panel-border);
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.08);
  color: var(--app-text);
}

.advisor-grid .checkbox {
  flex-direction: row;
  align-items: center;
  gap: 8px;
}

.advisor-actions {
  display: flex;
  gap: 8px;
}

.advisor-actions button {
  padding: 10px 12px;
  border: none;
  border-radius: 8px;
  background: var(--app-accent);
  color: #fff;
  cursor: pointer;
  font-weight: 600;
}

.advisor-actions button:disabled {
  background: rgba(56, 189, 248, 0.4);
  cursor: not-allowed;
}

.advisor-results {
  background: rgba(56, 189, 248, 0.08);
  border: 1px solid rgba(56, 189, 248, 0.2);
  border-radius: 12px;
  padding: 12px;
}

.advisor-summary {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  align-items: center;
  font-size: 14px;
  color: var(--app-text);
}

.advisor-stats span {
  margin-right: 10px;
  color: var(--app-text-muted);
}

.advisor-policy {
  margin-top: 8px;
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.advisor-policy-row {
  display: flex;
  justify-content: space-between;
  font-size: 14px;
}

.policy-action {
  font-weight: 700;
}

.policy-prob {
  color: var(--app-accent);
}

.policy-visits {
  color: var(--app-text-muted);
}

.advisor-error {
  color: #fca5a5;
  font-weight: 700;
}

.advisor-progress {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 13px;
  color: var(--app-text-muted);
}

.advisor-progress-bar {
  position: relative;
  flex: 1;
  height: 8px;
  background: rgba(255, 255, 255, 0.08);
  border-radius: 6px;
  overflow: hidden;
  border: 1px solid var(--app-panel-border);
}

.advisor-progress-fill {
  position: absolute;
  left: 0;
  top: 0;
  bottom: 0;
  width: 0%;
  height: 100%;
  background: linear-gradient(90deg, rgba(56, 189, 248, 0.9), rgba(14, 165, 233, 0.9));
  transition: width 0.2s ease;
}

.advisor-progress-fill.indeterminate {
  width: 40%;
  animation: advisor-indeterminate 1s ease-in-out infinite;
}

@keyframes advisor-indeterminate {
  0% { left: -40%; }
  50% { left: 60%; }
  100% { left: -40%; }
}

.advisor-progress-text {
  min-width: 70px;
}

.advisor-note {
  font-size: 13px;
  color: var(--app-text-muted);
  background: rgba(56, 189, 248, 0.08);
  border: 1px solid rgba(56, 189, 248, 0.15);
  padding: 8px 10px;
  border-radius: 8px;
}

/* Rewards styles */
.rewards-section {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.episode-totals {
  display: flex;
  gap: 1rem;
  padding: 1rem;
  background-color: rgba(15, 23, 42, 0.6);
  border-radius: 16px;
  border: 1px solid var(--app-panel-border);
  justify-content: space-around;
}

.total-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.25rem;
}

.team-label {
  font-weight: 600;
  font-size: 0.85rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.team-label.offense {
  color: #fb7185;
}

.team-label.defense {
  color: var(--app-accent);
}

.reward-value {
  font-size: 1.3rem;
  font-weight: bold;
  font-family: 'DSEG7 Classic', monospace;
  color: var(--app-text);
  text-shadow: 0 0 10px rgba(255, 255, 255, 0.1);
}

.reward-history {
  max-height: 300px;
  overflow-y: auto;
  border-radius: 12px;
  border: 1px solid var(--app-panel-border);
}

.entropy-section {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.entropy-note {
  color: var(--app-text-muted);
  font-size: 0.9rem;
  margin: 0;
}

.entropy-table-wrapper {
  border: 1px solid var(--app-panel-border);
  border-radius: 12px;
  overflow: hidden;
}

.entropy-table {
  width: 100%;
  border-collapse: collapse;
}

.entropy-table th,
.entropy-table td {
  border-bottom: 1px solid rgba(148, 163, 184, 0.12);
  padding: 0.65rem 0.75rem;
  text-align: left;
  font-size: 0.9rem;
}

.entropy-table th {
  background: rgba(15, 23, 42, 0.8);
  color: var(--app-text-muted);
  text-transform: uppercase;
  letter-spacing: 0.04em;
  font-size: 0.8rem;
}

.entropy-table tr:nth-child(even) td {
  background: rgba(255, 255, 255, 0.02);
}

.entropy-table td:last-child {
  font-family: 'Courier New', monospace;
  color: var(--app-accent);
}

.no-rewards {
  text-align: center;
  padding: 2rem;
  color: var(--app-text-muted);
  font-style: italic;
}

.reward-table {
  background-color: transparent;
  width: 100%;
  border-collapse: collapse;
}

.reward-header {
  display: grid;
  grid-template-columns: 0.8fr 0.8fr 1fr 1.5fr 1fr 1.5fr;
  padding: 0.75rem;
  background-color: rgba(15, 23, 42, 0.8);
  font-weight: 600;
  border-bottom: 1px solid var(--app-panel-border);
  text-align: center;
  color: var(--app-text-muted);
  font-size: 0.85rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  position: sticky;
  top: 0;
}

.reward-table.with-phi .reward-header {
  grid-template-columns: 0.8fr 0.8fr 1fr 1.5fr 1fr 1.5fr 1fr;
}

.reward-row {
  display: grid;
  grid-template-columns: 0.8fr 0.8fr 1fr 1.5fr 1fr 1.5fr;
  padding: 0.6rem 0.75rem;
  border-bottom: 1px solid rgba(148, 163, 184, 0.1);
  text-align: center;
  font-size: 0.9rem;
  color: var(--app-text);
}

.reward-table.with-phi .reward-row {
  grid-template-columns: 0.8fr 0.8fr 1fr 1.5fr 1fr 1.5fr 1fr;
}

.reward-row:last-child {
  border-bottom: none;
}

.reward-row:hover {
  background-color: rgba(255, 255, 255, 0.03);
}

.reward-row.current-reward-row {
  background-color: rgba(251, 146, 60, 0.1) !important;
  border-top: 1px solid rgba(251, 146, 60, 0.3);
  border-bottom: 1px solid rgba(251, 146, 60, 0.3);
}

.reward-row.current-reward-row:hover {
  background-color: rgba(251, 146, 60, 0.15) !important;
}

.positive {
  color: var(--app-success);
  font-weight: 600;
}

.negative {
  color: #fb7185;
  font-weight: 600;
}

.reason-text {
  font-size: 0.8rem;
  color: var(--app-text-muted);
}

/* Existing control styles */
.control-pad-wrapper {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 1rem;
  padding: 1rem;
  background: rgba(15, 23, 42, 0.4);
  border-radius: 16px;
  border: 1px solid var(--app-panel-border);
}

.disabled {
  opacity: 0.5;
  pointer-events: none;
  filter: grayscale(100%);
}

/* Moves styles */
.moves-section {
  padding: 0.5rem;
}

.moves-summary {
  display: flex;
  justify-content: flex-end;
  gap: 0.4rem;
  margin: 0.2rem 0 0.8rem;
  color: var(--app-text-muted);
  font-size: 0.85rem;
}

.moves-summary .summary-label {
  text-transform: uppercase;
  letter-spacing: 0.04em;
}

.moves-summary .summary-value {
  color: var(--app-accent);
  font-family: 'Courier New', monospace;
  font-weight: 600;
}

.moves-table {
  width: 100%;
  border-collapse: collapse;
  margin-top: 0;
  border-radius: 8px;
  overflow: hidden;
}

.moves-table th,
.moves-table td {
  border: 1px solid rgba(148, 163, 184, 0.15);
  padding: 0.6rem;
  text-align: center;
  font-size: 0.9rem;
}

.moves-table th {
  background-color: rgba(15, 23, 42, 0.8);
  color: var(--app-text-muted);
  font-weight: 600;
  text-transform: uppercase;
  font-size: 0.8rem;
  letter-spacing: 0.05em;
}

.value-cell {
  font-family: 'Courier New', monospace;
  font-size: 0.9em;
  color: var(--app-accent);
  font-weight: 500;
}

.moves-table tr:nth-child(even) {
  background-color: rgba(255, 255, 255, 0.02);
}

.moves-table tr.current-shot-clock-row {
  background-color: rgba(56, 189, 248, 0.15) !important;
  box-shadow: inset 0 0 10px rgba(56, 189, 248, 0.1);
}

.moves-table tr.current-shot-clock-row td {
  border-top: 1px solid rgba(56, 189, 248, 0.4);
  border-bottom: 1px solid rgba(56, 189, 248, 0.4);
  color: var(--app-text);
}

.moves-table tr.current-shot-clock-row td:first-child {
  border-left: 1px solid rgba(56, 189, 248, 0.4);
}

.moves-table tr.current-shot-clock-row td:last-child {
  border-right: 1px solid rgba(56, 189, 248, 0.4);
}

.moves-note-cell {
  text-align: left;
  padding: 0.6rem 0.8rem;
  font-size: 0.85rem;
  color: var(--app-text-muted);
  background: rgba(148, 163, 184, 0.06);
}

.move-cell {
  padding: 6px 8px;
}

.move-action {
  font-weight: 500;
  color: var(--app-text);
}

.ball-holder-icon {
  font-size: 1em;
  color: var(--app-warning);
  margin-left: 4px;
}

.mcts-icon {
  color: var(--app-accent);
}

.pass-steal-info {
  font-size: 0.75em;
  color: #fb7185;
  font-style: italic;
}

.defender-pressure-info {
  font-size: 0.75em;
  color: var(--app-warning);
  font-style: italic;
}

.shot-clock-cell {
  font-weight: 600;
  color: var(--app-warning);
  font-family: 'DSEG7 Classic', monospace;
}

.no-moves {
  text-align: center;
  color: var(--app-text-muted);
  font-style: italic;
  padding: 20px;
}

/* Parameters styles */
.parameters-section {
  padding: 0;
}

.parameters-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 1rem;
}

.param-category {
  background: rgba(15, 23, 42, 0.4);
  border-radius: 12px;
  padding: 1rem;
  border: 1px solid var(--app-panel-border);
}

.param-category h5 {
  margin: 0 0 0.8rem 0;
  color: var(--app-accent);
  font-weight: 600;
  border-bottom: 1px solid var(--app-panel-border);
  padding-bottom: 0.5rem;
  text-transform: uppercase;
  font-size: 0.8rem;
  letter-spacing: 0.05em;
}

.param-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.4rem 0;
  border-bottom: 1px solid rgba(148, 163, 184, 0.1);
  position: relative;
  cursor: help;
}

/* Tooltip styles */
.param-item[data-tooltip] {
  cursor: help;
}

.param-item[data-tooltip]::before {
  content: attr(data-tooltip);
  position: absolute;
  bottom: calc(100% + 8px);
  left: 50%;
  transform: translateX(-50%);
  padding: 0.6rem 0.8rem;
  background: rgba(15, 23, 42, 0.98);
  color: var(--app-text);
  font-size: 0.8rem;
  font-weight: 400;
  line-height: 1.4;
  border-radius: 8px;
  border: 1px solid var(--app-accent);
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4), 0 0 15px rgba(56, 189, 248, 0.15);
  white-space: normal;
  width: max-content;
  max-width: 280px;
  z-index: 1000;
  opacity: 0;
  visibility: hidden;
  transition: opacity 0.2s ease, visibility 0.2s ease;
  pointer-events: none;
  text-align: left;
}

.param-item[data-tooltip]::after {
  content: '';
  position: absolute;
  bottom: calc(100% + 2px);
  left: 50%;
  transform: translateX(-50%);
  border: 6px solid transparent;
  border-top-color: var(--app-accent);
  z-index: 1001;
  opacity: 0;
  visibility: hidden;
  transition: opacity 0.2s ease, visibility 0.2s ease;
}

.param-item[data-tooltip]:hover::before,
.param-item[data-tooltip]:hover::after {
  opacity: 1;
  visibility: visible;
}

/* Adjust tooltip position for items near edges */
.param-item[data-tooltip]:first-child::before {
  left: 0;
  transform: translateX(0);
}

.param-item[data-tooltip]:first-child::after {
  left: 20px;
  transform: translateX(0);
}

/* Add a subtle help indicator */
.param-item[data-tooltip] .param-name::after {
  content: '?';
  display: inline-flex;
  align-items: center;
  justify-content: center;
  margin-left: 0.4rem;
  width: 14px;
  height: 14px;
  font-size: 0.65rem;
  font-weight: 600;
  background: rgba(56, 189, 248, 0.15);
  color: var(--app-accent);
  border-radius: 50%;
  border: 1px solid rgba(56, 189, 248, 0.3);
  opacity: 0.6;
  transition: opacity 0.2s ease;
}

.param-item[data-tooltip]:hover .param-name::after {
  opacity: 1;
  background: rgba(56, 189, 248, 0.25);
}

.policy-select-item {
  flex-direction: column;
  align-items: flex-start;
  gap: 0.5rem;
}

.policy-label {
  font-weight: 500;
  color: var(--app-text-muted);
  font-size: 0.9rem;
}

.policy-select-wrapper {
  width: 100%;
}

.policy-select-wrapper select {
  width: 100%;
  padding: 0.4rem 0.6rem;
  border: 1px solid var(--app-panel-border);
  border-radius: 8px;
  font-size: 0.9rem;
  background: rgba(13, 20, 38, 0.8);
  color: var(--app-text);
}

.policy-select-wrapper select:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.policy-status {
  font-size: 0.8rem;
  margin-top: 0.2rem;
  color: var(--app-text-muted);
}

.policy-status.error {
  color: #fb7185;
}

.policy-actions {
  margin-top: 0.5rem;
  display: flex;
  justify-content: flex-end;
}

.refresh-policies-btn {
  padding: 0.35rem 0.75rem;
  font-size: 0.8rem;
  border-radius: 6px;
  border: 1px solid var(--app-panel-border);
  background: rgba(56, 189, 248, 0.1);
  color: var(--app-text);
  cursor: pointer;
  transition: all 0.15s ease;
}

.refresh-policies-btn:hover:not(:disabled) {
  background: rgba(56, 189, 248, 0.25);
  border-color: var(--app-accent);
}

.refresh-policies-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.offense-skills-editor {
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
}

.offense-skills-row {
  display: grid;
  grid-template-columns: 0.9fr repeat(3, 1fr);
  gap: 0.5rem;
  align-items: center;
}

.offense-skills-row.header {
  font-size: 0.8rem;
  color: var(--app-text-muted);
  text-transform: uppercase;
  letter-spacing: 0.05em;
  border-bottom: 1px solid rgba(148, 163, 184, 0.15);
  padding-bottom: 0.25rem;
}

.offense-skill-input {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.offense-skill-input input {
  width: 100%;
  padding: 0.35rem 0.5rem;
  border: 1px solid var(--app-panel-border);
  border-radius: 6px;
  background: rgba(13, 20, 38, 0.7);
  color: var(--app-text);
}

.skills-player {
  font-weight: 600;
  color: var(--app-text);
}

.offense-skill-default {
  font-size: 0.75rem;
  color: var(--app-text-muted);
}

.offense-skill-actions {
  display: flex;
  justify-content: flex-end;
  gap: 0.5rem;
  margin-top: 0.25rem;
}

.param-item:last-child {
  border-bottom: none;
}

.param-name {
  font-weight: 500;
  color: var(--app-text-muted);
  font-size: 0.9rem;
}

.param-value {
  font-family: 'Courier New', monospace;
  background: rgba(0, 0, 0, 0.3);
  padding: 0.2rem 0.5rem;
  border-radius: 4px;
  border: 1px solid rgba(148, 163, 184, 0.2);
  font-size: 0.85em;
  color: var(--app-accent);
}

.param-value.policy-class-value {
  font-size: 0.75em;
  word-break: break-all;
  max-width: 180px;
  text-align: right;
}

/* Observation Tab Styles */
.observation-section {
  padding: 0;
}

.observation-table-wrapper {
  max-height: 600px;
  overflow-y: auto;
  border: 1px solid var(--app-panel-border);
  border-radius: 12px;
  background: rgba(15, 23, 42, 0.4);
}

.observation-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.9em;
  background: transparent;
}

.observation-table th {
  position: sticky;
  top: 0;
  background-color: rgba(15, 23, 42, 0.95);
  color: var(--app-text-muted);
  padding: 0.75rem;
  text-align: left;
  font-weight: 600;
  border-bottom: 1px solid var(--app-panel-border);
  z-index: 10;
  text-transform: uppercase;
  font-size: 0.8rem;
  letter-spacing: 0.05em;
}

.observation-table td {
  padding: 0.6rem 0.75rem;
  border-bottom: 1px solid rgba(148, 163, 184, 0.1);
  color: var(--app-text);
}

.observation-table tbody tr:hover {
  background-color: rgba(255, 255, 255, 0.03);
}

.token-section {
  margin-top: 1.5rem;
}

.token-table-wrapper {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.token-table {
  font-size: 0.9em;
}

.token-table-scroll {
  overflow-x: auto;
  max-width: 100%;
}

.token-table-scroll .token-table {
  min-width: max-content;
}

.token-ball-holder {
  background-color: rgba(251, 191, 36, 0.12);
}

.token-subtitle {
  margin: 0.25rem 0;
  font-size: 0.95rem;
  color: var(--app-text-muted);
}

.token-attn-section {
  margin-top: 0.75rem;
}

.token-attn-controls {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.85rem;
  color: var(--app-text-muted);
  flex-wrap: wrap;
}

.token-attn-controls select {
  padding: 0.25rem 0.5rem;
  border: 1px solid var(--app-panel-border);
  border-radius: 6px;
  background: rgba(15, 23, 42, 0.7);
  color: var(--app-text);
  font-size: 0.85rem;
}

.token-attn-download {
  margin-left: auto;
  padding: 0.3rem 0.6rem;
  border: 1px solid var(--app-panel-border);
  border-radius: 6px;
  background: rgba(15, 23, 42, 0.7);
  color: var(--app-text);
  font-size: 0.85rem;
  cursor: pointer;
}

.token-attn-download:hover {
  background: rgba(59, 130, 246, 0.2);
}

.token-attn-note {
  font-size: 0.85rem;
  color: var(--app-text-muted);
}

.token-attn-table th,
.token-attn-table td {
  text-align: center;
  white-space: nowrap;
}

.group-label {
  font-weight: 600;
  background-color: rgba(15, 23, 42, 0.6);
  color: var(--app-accent);
  min-width: 140px;
  border-right: 1px solid rgba(148, 163, 184, 0.1);
}

.ball-holder-row {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 12px;
}

.status-note {
  color: var(--app-text-muted);
  font-size: 0.85rem;
}

.error-note {
  color: #ff7676;
  font-size: 0.85rem;
}

.value-mono {
  font-family: 'Courier New', monospace;
  background: rgba(0, 0, 0, 0.3);
  padding: 0.2rem 0.4rem;
  border-radius: 3px;
  font-weight: 500;
  color: var(--app-warning);
  border: 1px solid rgba(148, 163, 184, 0.15);
}

.notes {
  font-size: 0.85em;
  color: var(--app-text-muted);
  font-style: italic;
}

.notes.highlight {
  color: #fb7185;
  font-weight: 600;
  font-style: normal;
}

.group-player-pos td:first-child { background-color: rgba(56, 189, 248, 0.05); }
.group-ball-holder td:first-child { background-color: rgba(251, 146, 60, 0.05); }
.group-shot-clock td:first-child { background-color: rgba(56, 189, 248, 0.08); }
.group-team-encoding td:first-child { background-color: rgba(168, 85, 247, 0.05); }
.group-ball-handler-pos td:first-child { background-color: rgba(251, 146, 60, 0.08); }
.group-hoop td:first-child { background-color: rgba(45, 212, 191, 0.05); }
.group-distances td:first-child { background-color: rgba(244, 114, 182, 0.05); }
.group-angles td:first-child { background-color: rgba(192, 132, 252, 0.05); }
.group-lane-steps td:first-child { background-color: rgba(251, 146, 60, 0.05); }
.group-ep td:first-child { background-color: rgba(45, 212, 191, 0.08); }
.group-turnover td:first-child { background-color: rgba(251, 113, 133, 0.08); }

/* Eval tab */
.eval-tab .eval-row {
  display: flex;
  gap: 0.75rem;
  align-items: center;
  flex-wrap: wrap;
  margin-bottom: 0.75rem;
}
.eval-tab .inline-label {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
}
.eval-run-btn {
  padding: 0.5rem 1rem;
  border-radius: 12px;
  background: #0ea5e9;
  color: #0b1221;
  border: none;
  cursor: pointer;
}
.eval-run-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}
.eval-status {
  color: #38bdf8;
}
.eval-custom {
  display: flex;
  flex-direction: column;
  gap: 0.9rem;
}
.radio-row {
  display: flex;
  gap: 1rem;
}
.eval-skills {
  border: 1px solid rgba(148, 163, 184, 0.2);
  border-radius: 12px;
  padding: 0.75rem;
  background: rgba(15, 23, 42, 0.5);
  display: flex;
  flex-direction: column;
  gap: 0.4rem;
}
.skills-header,
.skills-row {
  display: grid;
  grid-template-columns: 1.2fr 1fr 1fr 1fr;
  gap: 0.5rem;
  align-items: center;
}
.skills-header {
  font-size: 0.85rem;
  color: #cbd5e1;
  opacity: 0.8;
}
.skills-row input {
  width: 100%;
  padding: 0.35rem 0.5rem;
  border-radius: 10px;
  border: 1px solid rgba(148, 163, 184, 0.35);
  background: rgba(15, 23, 42, 0.7);
  color: #e2e8f0;
}
.skills-player {
  font-weight: 600;
}
.skills-actions {
  display: flex;
  justify-content: flex-end;
  margin-top: 0.25rem;
}
.ghost-btn {
  background: transparent;
  color: #38bdf8;
  border: 1px solid rgba(56, 189, 248, 0.4);
  border-radius: 10px;
  padding: 0.35rem 0.75rem;
  cursor: pointer;
}
.ghost-btn:hover {
  border-color: #7dd3fc;
}
.stats-selector {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.75rem;
}
.eval-stats-summary {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
  gap: 0.5rem;
  margin-bottom: 0.75rem;
}
.stat-chip {
  padding: 0.75rem;
  border-radius: 12px;
  background: rgba(15, 23, 42, 0.65);
  border: 1px solid rgba(148, 163, 184, 0.2);
  display: flex;
  flex-direction: column;
  gap: 0.2rem;
}
.stat-chip span {
  color: #94a3b8;
  font-size: 0.85rem;
}
.stat-chip strong {
  color: #e2e8f0;
  font-size: 1.05rem;
}
.shot-chart-wrapper {
  border: 1px solid rgba(148, 163, 184, 0.2);
  border-radius: 14px;
  background: rgba(15, 23, 42, 0.55);
  padding: 0.75rem;
  margin-bottom: 1rem;
}
.shot-chart {
  width: 100%;
  max-width: 420px;
  height: 280px;
}
.shot-chart-legend {
  margin-top: 0.35rem;
  color: #cbd5e1;
  font-size: 0.85rem;
  display: flex;
  align-items: center;
  gap: 0.35rem;
}
.legend-dot {
  display: inline-block;
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: #22c55e;
}
.legend-dot.made {
  background: #22c55e;
}
.per-player-stats {
  margin-top: 0.75rem;
}
.per-player-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.9rem;
}
.per-player-table th,
.per-player-table td {
  padding: 0.45rem 0.6rem;
  border-bottom: 1px solid rgba(148, 163, 184, 0.2);
}
.per-player-table th {
  text-align: left;
  color: #cbd5e1;
}
.per-player-table td {
  color: #e2e8f0;
}
.group-steal td:first-child { background-color: rgba(251, 113, 133, 0.08); }

.no-data {
  text-align: center;
  padding: 2rem;
  color: var(--app-text-muted);
  font-style: italic;
}
</style> 
