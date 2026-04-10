<script setup>
import { ref, computed, watch, onMounted, onBeforeUnmount, nextTick } from 'vue';
import { defineExpose } from 'vue';
import HexagonControlPad from './HexagonControlPad.vue';
import {
  applyStartTemplate,
  getActionValues,
  getPlaybookProgress,
  getRewards,
  importStartTemplateLibrary,
  loadStartTemplateLibrary,
  mctsAdvise,
  runPlaybookAnalysis,
  saveStartTemplateLibrary,
  setOffenseSkills,
  setPassTargetStrategy,
  setPassLogitBias,
  setBallHolder,
  setIntentState,
  setStartTemplateLibrary,
  captureCounterfactualSnapshot,
  restoreCounterfactualSnapshot,
  replayCounterfactualSnapshot,
  setShotPressureParams,
  setPassInterceptionParams,
  setDefenderPressureParams,
} from '@/services/api';
import { loadStats, saveStats, resetStatsStorage } from '@/services/stats';

function formatParamCount(n) {
  if (n === null || n === undefined) return 'N/A';
  const num = Number(n);
  if (Number.isNaN(num)) return 'N/A';
  if (num >= 1_000_000) return `${(num / 1_000_000).toFixed(2)}M`;
  if (num >= 1_000) return `${(num / 1_000).toFixed(1)}k`;
  return String(num);
}

function lookupPlayName(playNameMap, intentIndex) {
  const idx = Number(intentIndex);
  if (!Number.isFinite(idx) || !playNameMap || typeof playNameMap !== 'object') return null;
  const raw = playNameMap[String(idx)] ?? playNameMap[idx];
  if (typeof raw !== 'string') return null;
  const trimmed = raw.trim();
  return trimmed || null;
}

function formatPlayLabel(intentIndex, playNameMap, explicitName = null) {
  const idx = Number(intentIndex);
  if (!Number.isFinite(idx)) return 'Unknown play';
  const name = (typeof explicitName === 'string' && explicitName.trim())
    ? explicitName.trim()
    : lookupPlayName(playNameMap, idx);
  return name ? `${name} (z=${idx})` : `z=${idx}`;
}

function deepCloneJson(value, fallback = null) {
  try {
    return JSON.parse(JSON.stringify(value));
  } catch {
    return fallback;
  }
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
  showOpponentActions: {
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
  initialActiveTab: {
    type: String,
    default: 'environment',
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
  templateConfig: {
    type: Object,
    default: null,
  },
  evalNumEpisodes: {
    type: Number,
    default: 100,
  },
  evalProgress: {
    type: Object,
    default: null,
  },
  perPlayerEvalStats: {
    type: Object,
    default: null,
  },
  perIntentEvalStats: {
    type: Object,
    default: null,
  },
  tabsMountEl: {
    type: null,
    default: null,
  },
  tabsMountSelector: {
    type: String,
    default: '',
  },
});

const emit = defineEmits(['actions-submitted', 'update:activePlayerId', 'move-recorded', 'policy-swap-requested', 'swap-teams-requested', 'selections-changed', 'refresh-policies', 'mcts-options-changed', 'mcts-toggle-changed', 'state-updated', 'eval-config-changed', 'template-config-changed', 'eval-run', 'active-tab-changed', 'ball-holder-updating', 'ball-holder-changed', 'stats-reset', 'counterfactual-replay-loaded', 'playbook-analysis-loaded']);

const hasExternalTabsMount = computed(() => String(props.tabsMountSelector || '').trim().length > 0);
const resolvedTabsMount = computed(() => {
  return String(props.tabsMountSelector || '').trim();
});
const tabsTeleportEnabled = ref(false);
const tabsMountTargetEl = computed(() => props.tabsMountEl || null);
const resolvedTabsTeleportTarget = computed(() => {
  return tabsMountTargetEl.value || resolvedTabsMount.value || 'body';
});
const useExternalTabsTeleport = computed(() => {
  return Boolean(tabsMountTargetEl.value) || (hasExternalTabsMount.value && tabsTeleportEnabled.value);
});

function refreshTabsTeleportTarget() {
  if (tabsMountTargetEl.value) {
    tabsTeleportEnabled.value = true;
    return;
  }
  if (!hasExternalTabsMount.value || typeof document === 'undefined') {
    tabsTeleportEnabled.value = false;
    return;
  }
  tabsTeleportEnabled.value = Boolean(document.querySelector(resolvedTabsMount.value));
}

const selectedActions = ref({});
const selectedPassTargets = ref({});
const passMode = computed(() => String(props.gameState?.pass_mode || 'directional').toLowerCase());
const isPointerPassMode = computed(() => passMode.value === 'pointer_targeted');
const POINTER_PASS_SLOT_ACTIONS = ['PASS_E', 'PASS_NE', 'PASS_NW', 'PASS_W', 'PASS_SW', 'PASS_SE'];

function isBallHolderPlayer(playerId) {
  const bh = props.gameState?.ball_holder;
  if (bh === null || bh === undefined) return false;
  return String(playerId) === String(bh);
}

function getPointerPassTeammatesForPlayer(playerId) {
  const pid = Number(playerId);
  if (!Number.isFinite(pid)) return [];
  const offense = props.gameState?.offense_ids || [];
  const defense = props.gameState?.defense_ids || [];
  const sameTeam = offense.includes(pid) ? offense : defense.includes(pid) ? defense : [];
  const teammates = sameTeam
    .map((id) => Number(id))
    .filter((id) => Number.isFinite(id) && id !== pid);
  teammates.sort((a, b) => a - b);
  return teammates.slice(0, POINTER_PASS_SLOT_ACTIONS.length);
}

function resolvePointerPassTarget(playerId, actionName) {
  if (!isPointerPassMode.value) return null;
  const pid = Number(playerId);
  if (!Number.isFinite(pid)) return null;
  const teammates = getPointerPassTeammatesForPlayer(pid);
  const isValidTarget = (targetId) => teammates.includes(Number(targetId));

  const action = typeof actionName === 'string' ? actionName : '';
  if (action.startsWith('PASS->')) {
    const parsed = Number(action.replace('PASS->', ''));
    return Number.isFinite(parsed) && isValidTarget(parsed) ? parsed : null;
  }
  if (!action.startsWith('PASS_')) return null;

  const explicitTarget = selectedPassTargets.value?.[pid];
  if (Number.isFinite(Number(explicitTarget)) && isValidTarget(explicitTarget)) {
    return Number(explicitTarget);
  }

  const slotIdx = POINTER_PASS_SLOT_ACTIONS.indexOf(action);
  if (slotIdx < 0) return null;
  if (slotIdx >= teammates.length) return null;
  return Number(teammates[slotIdx]);
}

function getResolvedPointerPassTarget(playerId) {
  const effectiveAction = getEffectiveSelectedAction(playerId);
  return resolvePointerPassTarget(playerId, effectiveAction);
}

const activeResolvedPassTarget = computed(() => {
  const active = props.activePlayerId;
  if (active === null || active === undefined) return null;
  const resolved = getResolvedPointerPassTarget(active);
  return Number.isFinite(Number(resolved)) ? Number(resolved) : null;
});

const activeEffectiveAction = computed(() => {
  const active = props.activePlayerId;
  if (active === null || active === undefined) return '';
  return String(getEffectiveSelectedAction(active) || '');
});

const activeHasPassSelection = computed(() => {
  const action = activeEffectiveAction.value;
  return action.startsWith('PASS_') || action.startsWith('PASS->');
});

function isPointerPassButtonSelected(targetId) {
  if (!activeHasPassSelection.value) return false;
  return activeResolvedPassTarget.value !== null && activeResolvedPassTarget.value === Number(targetId);
}

function buildDisplaySelections(selections) {
  const displaySelections = {};
  for (const [pid, actionName] of Object.entries(selections || {})) {
    const action = typeof actionName === 'string' ? actionName : '';
    const isPassAction = action.startsWith('PASS_') || action.startsWith('PASS->');
    const numericPid = Number(pid);
    if (!isPassAction) {
      displaySelections[pid] = actionName;
      continue;
    }

    if (isPointerPassMode.value && !isBallHolderPlayer(pid)) {
      continue;
    }

    const resolvedTarget = resolvePointerPassTarget(numericPid, action);
    if (Number.isFinite(Number(resolvedTarget))) {
      displaySelections[pid] = `PASS->${Number(resolvedTarget)}`;
    } else {
      displaySelections[pid] = action;
    }
  }
  return displaySelections;
}

const paramCounts = computed(() => props.gameState?.training_params?.param_counts || null);
const selectorTrainingParams = computed(() => props.gameState?.training_params || {});
const selectorEnabled = computed(() =>
  Boolean(selectorTrainingParams.value?.intent_selector_enabled)
);
const selectorAlphaSummary = computed(() => {
  if (!selectorEnabled.value) return 'Disabled';
  const start = selectorTrainingParams.value?.intent_selector_alpha_start;
  const end = selectorTrainingParams.value?.intent_selector_alpha_end;
  if (start === null || start === undefined) return 'N/A';
  if (end === null || end === undefined || Number(start) === Number(end)) {
    return String(start);
  }
  return `${start} → ${end}`;
});
const selectorScheduleSummary = computed(() => {
  if (!selectorEnabled.value) return 'Disabled';
  const warmup = selectorTrainingParams.value?.intent_selector_alpha_warmup_steps;
  const ramp = selectorTrainingParams.value?.intent_selector_alpha_ramp_steps;
  const warmupText = warmup?.toLocaleString?.() || warmup || '0';
  const rampText = ramp?.toLocaleString?.() || ramp || '0';
  return `warmup ${warmupText}, ramp ${rampText}`;
});
const selectorEpsSummary = computed(() => {
  if (!selectorEnabled.value) return 'Disabled';
  const start = selectorTrainingParams.value?.intent_selector_eps_start;
  const end = selectorTrainingParams.value?.intent_selector_eps_end;
  if (start === null || start === undefined) return 'N/A';
  if (end === null || end === undefined || Number(start) === Number(end)) {
    return String(start);
  }
  return `${start} → ${end}`;
});
const selectorEpsScheduleSummary = computed(() => {
  if (!selectorEnabled.value) return 'Disabled';
  const warmup = selectorTrainingParams.value?.intent_selector_eps_warmup_steps;
  const ramp = selectorTrainingParams.value?.intent_selector_eps_ramp_steps;
  const warmupText = warmup?.toLocaleString?.() || warmup || '0';
  const rampText = ramp?.toLocaleString?.() || ramp || '0';
  return `warmup ${warmupText}, ramp ${rampText}`;
});
const selectorMultiselectEnabled = computed(() =>
  Boolean(selectorTrainingParams.value?.intent_selector_multiselect_enabled)
);
const selectorMinPlayStepsSummary = computed(() => {
  if (!selectorEnabled.value) return 'Disabled';
  if (!selectorMultiselectEnabled.value) return 'N/A (multiselect off)';
  return selectorTrainingParams.value?.intent_selector_min_play_steps ?? 'N/A';
});
const selectorDecisionBoundaryLabel = computed(() => {
  if (!selectorEnabled.value) return 'Disabled';
  if (selectorMultiselectEnabled.value) {
    return 'Possession start + completed-pass reselection';
  }
  return 'Possession start only';
});
const taskRewardScaleSummary = computed(() => {
  const start = selectorTrainingParams.value?.task_reward_scale_start;
  const end = selectorTrainingParams.value?.task_reward_scale_end;
  if (start === null || start === undefined) return 'Disabled';
  if (end === null || end === undefined || Number(start) === Number(end)) {
    return String(start);
  }
  return `${start} → ${end}`;
});
const taskRewardScheduleSummary = computed(() => {
  const start = selectorTrainingParams.value?.task_reward_scale_start;
  if (start === null || start === undefined) return 'Disabled';
  const warmup = selectorTrainingParams.value?.task_reward_scale_warmup_steps;
  const ramp = selectorTrainingParams.value?.task_reward_scale_ramp_steps;
  const warmupText = warmup?.toLocaleString?.() || warmup || '0';
  const rampText = ramp?.toLocaleString?.() || ramp || '0';
  return `warmup ${warmupText}, ramp ${rampText}`;
});
const selectorHeadContextLabel = computed(() => {
  const policyClass = String(selectorTrainingParams.value?.policy_class || '');
  return policyClass.includes('SetAttention') ? 'Shared set-attention encoder' : 'Shared policy encoder';
});
const selectorMode = computed(() =>
  String(selectorTrainingParams.value?.intent_selector_mode || 'callback').toLowerCase()
);
const selectorUsesIntegratedPath = computed(() => selectorMode.value === 'integrated');
const selectorImplementationLabel = computed(() => {
  return selectorUsesIntegratedPath.value ? 'Integrated PPO path' : 'Callback prototype';
});
const selectorValueCoef = computed(() => selectorTrainingParams.value?.intent_selector_value_coef);
const selectorCriticEnabled = computed(() =>
  Boolean(
    selectorEnabled.value
    && selectorUsesIntegratedPath.value
    && selectorValueCoef.value !== null
    && selectorValueCoef.value !== undefined
  )
);
const selectorHeadSummary = computed(() => {
  if (!selectorEnabled.value) return 'N/A';
  if (selectorCriticEnabled.value) return 'Intent logits + selector value head';
  return 'Intent logits head';
});
const selectorObjectiveLabel = computed(() => {
  if (!selectorEnabled.value) return 'Disabled';
  if (selectorCriticEnabled.value) return 'Clipped PPO selector loss + full-possession value baseline';
  if (selectorUsesIntegratedPath.value) return 'Integrated selector policy loss';
  return 'Callback-side selector prototype';
});
const selectorCreditAssignmentLabel = computed(() => {
  if (!selectorEnabled.value) return 'Disabled';
  return selectorCriticEnabled.value
    ? 'Full-possession return against selector value baseline'
    : 'Full-possession return';
});
const movesColumnCount = computed(() => (allPlayerIds.value?.length || 0) + 4);
const pressureExposureDisplay = computed(() => {
  const val = Number(props.pressureExposure);
  return Number.isFinite(val) ? val.toFixed(3) : '0.000';
});

watch(selectedActions, (newActions) => {
  emit('selections-changed', buildDisplaySelections(newActions));
}, { deep: true });

watch(selectedPassTargets, () => {
  emit('selections-changed', buildDisplaySelections(selectedActions.value));
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
const intentStateInput = ref({
  active: false,
  intent_index: 0,
  intent_age: 0,
});
const intentStateUpdating = ref(false);
const intentStateError = ref(null);
const counterfactualSnapshotUpdating = ref(false);
const counterfactualSnapshotError = ref(null);
const playbookIntentInput = ref('');
const playbookNumRollouts = ref(24);
const playbookMaxSteps = ref(8);
const playbookRunToEnd = ref(false);
const playbookUseSnapshot = ref(true);
const playbookPlayerDeterministic = ref(false);
const playbookOpponentDeterministic = ref(true);
const playbookRunning = ref(false);
const playbookError = ref(null);
const playbookResult = ref(null);
const playbookProgress = ref({
  running: false,
  completed: 0,
  total: 0,
  fraction: 0,
  status: 'idle',
  error: null,
});
let playbookProgressPollHandle = null;
let playbookProgressPollBusy = false;
const pressureParamsInput = ref({
  three_pt_extra_hex_decay: 0.05,
  shot_pressure_enabled: true,
  shot_pressure_max: 0.5,
  shot_pressure_lambda: 1.0,
  shot_pressure_arc_degrees: 60.0,
  base_steal_rate: 0.35,
  steal_perp_decay: 1.5,
  steal_distance_factor: 0.08,
  steal_position_weight_min: 0.3,
  defender_pressure_distance: 1,
  defender_pressure_turnover_chance: 0.05,
  defender_pressure_decay_lambda: 1.0,
});
const pressureParamsUpdating = ref(false);
const pressureParamsError = ref(null);
const pendingPressureSyncKeys = ref(null);
const suppressNextPressurePropsSync = ref(false);
const activePressureUpdateSection = ref(null);
const SHOT_PRESSURE_PARAM_KEYS = [
  'three_pt_extra_hex_decay',
  'shot_pressure_enabled',
  'shot_pressure_max',
  'shot_pressure_lambda',
  'shot_pressure_arc_degrees',
];
const DEFENDER_PRESSURE_PARAM_KEYS = [
  'defender_pressure_distance',
  'defender_pressure_turnover_chance',
  'defender_pressure_decay_lambda',
];
const PASS_INTERCEPTION_PARAM_KEYS = [
  'base_steal_rate',
  'steal_perp_decay',
  'steal_distance_factor',
  'steal_position_weight_min',
];

function isPressureSectionUpdating(section) {
  return (
    pressureParamsUpdating.value &&
    activePressureUpdateSection.value === section
  );
}
const PASS_TARGET_STRATEGIES = [
  { value: 'nearest', label: 'Nearest (legacy)' },
  { value: 'best_ev', label: 'Best EV' },
];
const intentIndexMax = computed(() =>
  Math.max(0, Number(props.gameState?.num_intents || 1) - 1)
);
const currentPlayOptions = computed(() => {
  const count = Math.max(0, Number(props.gameState?.num_intents || 0));
  return Array.from({ length: count }, (_, idx) => ({
    value: idx,
    label: formatPlayLabel(idx, props.gameState?.play_name_map),
  }));
});
const intentAgeMax = computed(() =>
  Math.max(0, Number(props.gameState?.intent_commitment_steps || 0))
);
const intentControlsDisabled = computed(() =>
  intentStateUpdating.value ||
  !props.gameState ||
  !props.gameState.enable_intent_learning ||
  props.gameState.done ||
  props.isEvaluating ||
  props.isReplaying ||
  props.isManualStepping
);
const counterfactualSnapshotAvailable = computed(() =>
  Boolean(props.gameState?.counterfactual_snapshot_available)
);
const counterfactualSnapshotStep = computed(() =>
  props.gameState?.counterfactual_snapshot_step ?? null
);
const counterfactualSnapshotShotClock = computed(() =>
  props.gameState?.counterfactual_snapshot_shot_clock ?? null
);
const counterfactualSnapshotBallHolder = computed(() =>
  props.gameState?.counterfactual_snapshot_ball_holder ?? null
);
const counterfactualSnapshotIntentSummary = computed(() => {
  if (!counterfactualSnapshotAvailable.value) return 'None';
  const active = Boolean(props.gameState?.counterfactual_snapshot_intent_active);
  if (!active) return 'Inactive';
  const idx = props.gameState?.counterfactual_snapshot_intent_index ?? 0;
  const age = props.gameState?.counterfactual_snapshot_intent_age ?? 0;
  return `${formatPlayLabel(idx, props.gameState?.play_name_map)}, age=${age}`;
});
const counterfactualSnapshotControlsDisabled = computed(() =>
  counterfactualSnapshotUpdating.value ||
  !props.gameState ||
  props.isEvaluating ||
  props.isReplaying
);
const playbookControlsDisabled = computed(() =>
  playbookRunning.value ||
  !props.gameState ||
  props.isEvaluating
);

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

watch(
  () => [
    props.gameState?.intent_active_current,
    props.gameState?.intent_index_current,
    props.gameState?.intent_age,
    props.gameState?.num_intents,
    props.gameState?.intent_commitment_steps,
  ],
  () => {
    intentStateInput.value = {
      active: Boolean(props.gameState?.intent_active_current),
      intent_index: Math.max(
        0,
        Math.min(intentIndexMax.value, Number(props.gameState?.intent_index_current ?? 0) || 0),
      ),
      intent_age: Math.max(
        0,
        Math.min(intentAgeMax.value, Number(props.gameState?.intent_age ?? 0) || 0),
      ),
    };
  },
  { immediate: true }
);

function syncPressureParamsInputs(state) {
  const src = state || {};
  const nextValues = {
    three_pt_extra_hex_decay: Number(src.three_pt_extra_hex_decay ?? 0.05),
    shot_pressure_enabled: Boolean(src.shot_pressure_enabled ?? true),
    shot_pressure_max: Number(src.shot_pressure_max ?? 0.5),
    shot_pressure_lambda: Number(src.shot_pressure_lambda ?? 1.0),
    shot_pressure_arc_degrees: Number(src.shot_pressure_arc_degrees ?? 60.0),
    base_steal_rate: Number(src.base_steal_rate ?? 0.35),
    steal_perp_decay: Number(src.steal_perp_decay ?? 1.5),
    steal_distance_factor: Number(src.steal_distance_factor ?? 0.08),
    steal_position_weight_min: Number(src.steal_position_weight_min ?? 0.3),
    defender_pressure_distance: Number(src.defender_pressure_distance ?? 1),
    defender_pressure_turnover_chance: Number(src.defender_pressure_turnover_chance ?? 0.05),
    defender_pressure_decay_lambda: Number(src.defender_pressure_decay_lambda ?? 1.0),
  };
  const scopedKeys = Array.isArray(pendingPressureSyncKeys.value)
    ? pendingPressureSyncKeys.value
    : null;
  if (scopedKeys && scopedKeys.length > 0) {
    const merged = { ...(pressureParamsInput.value || {}) };
    for (const key of scopedKeys) {
      if (Object.prototype.hasOwnProperty.call(nextValues, key)) {
        merged[key] = nextValues[key];
      }
    }
    pressureParamsInput.value = merged;
    pendingPressureSyncKeys.value = null;
    return;
  }
  pressureParamsInput.value = nextValues;
}

watch(
  () => [
    props.gameState?.three_pt_extra_hex_decay,
    props.gameState?.shot_pressure_enabled,
    props.gameState?.shot_pressure_max,
    props.gameState?.shot_pressure_lambda,
    props.gameState?.shot_pressure_arc_degrees,
    props.gameState?.base_steal_rate,
    props.gameState?.steal_perp_decay,
    props.gameState?.steal_distance_factor,
    props.gameState?.steal_position_weight_min,
    props.gameState?.defender_pressure_distance,
    props.gameState?.defender_pressure_turnover_chance,
    props.gameState?.defender_pressure_decay_lambda,
  ],
  () => {
    if (suppressNextPressurePropsSync.value) {
      suppressNextPressurePropsSync.value = false;
      return;
    }
    if (pressureParamsUpdating.value && !pendingPressureSyncKeys.value) return;
    syncPressureParamsInputs(props.gameState || {});
  },
  { immediate: true }
);

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
  intentSelectionMode: 'learned_sample',
});
const defaultTemplateConfig = () => ({
  positions: [],
  ballHolder: null,
  templateId: 'new_template',
  weight: 1.0,
  mirrorable: true,
  shotClock: 24,
  jitterByPlayer: {},
  roleByPlayer: {},
});
const playersPerSideForTemplates = computed(() => {
  const raw = Number(props.gameState?.players_per_side ?? props.gameState?.players ?? 3);
  return Number.isFinite(raw) && raw > 0 ? Math.round(raw) : 3;
});

function buildDefaultTemplateLibrary() {
  return {
    version: 1,
    players_per_side: playersPerSideForTemplates.value,
    templates: [],
  };
}

function normalizeTemplateLibraryDraft(library) {
  const source = library && typeof library === 'object' ? library : {};
  const rawTemplates = Array.isArray(source.templates) ? source.templates : [];
  return {
    version: 1,
    players_per_side: playersPerSideForTemplates.value,
    templates: rawTemplates.map((template) => deepCloneJson(template, {})).filter(Boolean),
  };
}

function buildTemplateLibrarySignature(library, source = '', path = '') {
  return JSON.stringify({
    library: normalizeTemplateLibraryDraft(library),
    source: String(source || ''),
    path: String(path || ''),
  });
}

const templateLibraryDraft = ref(buildDefaultTemplateLibrary());
const templateLibraryPathInput = ref('');
const templateLibraryFileInput = ref(null);
const templateLibraryImportedFilename = ref('');
const templateLibraryStatus = ref('');
const templateLibraryError = ref('');
const templateLibraryDirty = ref(false);
const templateLibraryRequestPending = ref(false);
const selectedEditableTemplateId = ref('');
let lastLoadedTemplateLibrarySignature = '';
const templateLibrarySource = computed(() => String(props.gameState?.start_template_library_source || '').trim());
const templateLibrarySessionPath = computed(() => String(props.gameState?.start_template_library_path || '').trim());
const templateLibrarySessionSignature = computed(() =>
  buildTemplateLibrarySignature(
    props.gameState?.start_template_library || null,
    templateLibrarySource.value,
    templateLibrarySessionPath.value,
  )
);

function clearTemplateLibraryFeedback() {
  templateLibraryStatus.value = '';
  templateLibraryError.value = '';
}

function uniqueTemplateId(baseId, excludeId = null) {
  const seed = sanitizeTemplateId(baseId || 'new_template');
  const existing = new Set(
    (templateLibraryDraft.value?.templates || [])
      .map((template) => String(template?.id || '').trim())
      .filter((id) => id && id !== String(excludeId || '').trim())
  );
  if (!existing.has(seed)) return seed;
  let idx = 2;
  while (existing.has(`${seed}_${idx}`)) {
    idx += 1;
  }
  return `${seed}_${idx}`;
}

const editableTemplateOptions = computed(() => {
  const templates = Array.isArray(templateLibraryDraft.value?.templates)
    ? templateLibraryDraft.value.templates
    : [];
  return templates
    .map((template) => {
      const id = String(template?.id || '').trim();
      return id ? { value: id, label: id } : null;
    })
    .filter(Boolean);
});

const selectedEditableTemplateIndex = computed(() => {
  const templates = Array.isArray(templateLibraryDraft.value?.templates)
    ? templateLibraryDraft.value.templates
    : [];
  const wantedId = String(selectedEditableTemplateId.value || '').trim();
  return templates.findIndex((template) => String(template?.id || '').trim() === wantedId);
});

const selectedEditableTemplate = computed(() => {
  const idx = selectedEditableTemplateIndex.value;
  const templates = Array.isArray(templateLibraryDraft.value?.templates)
    ? templateLibraryDraft.value.templates
    : [];
  return idx >= 0 ? templates[idx] || null : null;
});

const hasEditableTemplateLibrary = computed(() => editableTemplateOptions.value.length > 0);
const templateLibraryJsonExport = computed(() => JSON.stringify(templateLibraryDraft.value, null, 2));
const templateLibrarySourceLabel = computed(() => {
  if (templateLibrarySource.value === 'local_file') return 'Local file';
  if (templateLibrarySource.value === 'file_upload') return 'Imported file';
  if (templateLibrarySource.value === 'mlflow_artifact') return 'Run artifact';
  if (templateLibrarySource.value === 'session_editor') return 'Session draft';
  return 'Unsaved draft';
});

const suppressTemplateConfigBackfill = ref(false);

function syncTemplateLibraryDraftFromState() {
  const incomingLibrary = props.gameState?.start_template_library;
  const normalized = normalizeTemplateLibraryDraft(
    incomingLibrary && typeof incomingLibrary === 'object'
      ? incomingLibrary
      : buildDefaultTemplateLibrary()
  );
  templateLibraryDraft.value = normalized;
  if (templateLibrarySource.value === 'file_upload') {
    templateLibraryImportedFilename.value = templateLibrarySessionPath.value || '';
  } else {
    templateLibraryImportedFilename.value = '';
  }
  if (templateLibrarySessionPath.value && templateLibrarySource.value !== 'file_upload') {
    templateLibraryPathInput.value = templateLibrarySessionPath.value;
  } else if (!templateLibraryPathInput.value) {
    templateLibraryPathInput.value = 'configs/start_templates_v1.json';
  }
  const validIds = new Set(
    (normalized.templates || [])
      .map((template) => String(template?.id || '').trim())
      .filter(Boolean)
  );
  if (!validIds.has(selectedEditableTemplateId.value)) {
    selectedEditableTemplateId.value = normalized.templates?.[0]?.id || '';
  }
  templateLibraryDirty.value = false;
  clearTemplateLibraryFeedback();
  lastLoadedTemplateLibrarySignature = templateLibrarySessionSignature.value;
}

watch(
  templateLibrarySessionSignature,
  (nextSignature) => {
    if (!nextSignature || nextSignature === lastLoadedTemplateLibrarySignature) return;
    syncTemplateLibraryDraftFromState();
  },
  { immediate: true }
);

const startTemplateOptions = computed(() => {
  const templates = Array.isArray(props.gameState?.start_template_library?.templates)
    ? props.gameState.start_template_library.templates
    : [];
  return templates
    .map((template) => {
      const id = String(template?.id || '').trim();
      return id ? { value: id, label: id } : null;
    })
    .filter(Boolean);
});
const hasLoadedStartTemplates = computed(() => startTemplateOptions.value.length > 0);
const selectedStartTemplateDefinition = computed(() => {
  const templates = Array.isArray(props.gameState?.start_template_library?.templates)
    ? props.gameState.start_template_library.templates
    : [];
  const wantedId = String(selectedStartTemplateId.value || '').trim();
  return templates.find((template) => String(template?.id || '').trim() === wantedId) || null;
});
const selectedStartTemplateId = ref('');
const selectedStartTemplateMirrored = ref(false);
const startTemplateActionStatus = ref('');
const startTemplateActionError = ref('');

watch(
  startTemplateOptions,
  (options) => {
    const values = new Set(options.map((opt) => opt.value));
    if (!values.has(selectedStartTemplateId.value)) {
      selectedStartTemplateId.value = options[0]?.value || '';
    }
  },
  { immediate: true }
);

function clearStartTemplateFeedback() {
  startTemplateActionStatus.value = '';
  startTemplateActionError.value = '';
}

function stableStartTemplateSeed(templateId) {
  const text = String(templateId || '');
  let hash = 2166136261;
  for (let i = 0; i < text.length; i += 1) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return Math.abs(hash >>> 0);
}

function previewOffsetToAxial(col, row) {
  const safeCol = Number(col) || 0;
  const safeRow = Number(row) || 0;
  return {
    q: safeCol - ((safeRow - (safeRow & 1)) >> 1),
    r: safeRow,
  };
}

function previewAxialToOffset(q, r) {
  const safeQ = Number(q) || 0;
  const safeR = Number(r) || 0;
  return {
    col: safeQ + ((safeR - (safeR & 1)) >> 1),
    row: safeR,
  };
}

function previewMirrorAnchor(anchor, courtHeight) {
  const [q, r] = Array.isArray(anchor) ? anchor : [0, 0];
  const { col, row } = previewAxialToOffset(q, r);
  const mirroredRow = Math.max(0, Number(courtHeight || 0) - 1 - row);
  const mirrored = previewOffsetToAxial(col, mirroredRow);
  return [mirrored.q, mirrored.r];
}

function clampTemplateAnchorToCourt(anchor, courtWidth, courtHeight) {
  const [q, r] = Array.isArray(anchor) ? anchor : [0, 0];
  const width = Math.max(1, Math.round(Number(courtWidth) || 0));
  const height = Math.max(1, Math.round(Number(courtHeight) || 0));
  const { col, row } = previewAxialToOffset(q, r);
  const clampedCol = Math.max(0, Math.min(width - 1, Math.round(Number(col) || 0)));
  const clampedRow = Math.max(0, Math.min(height - 1, Math.round(Number(row) || 0)));
  const clamped = previewOffsetToAxial(clampedCol, clampedRow);
  return [clamped.q, clamped.r];
}

function previewAxialToCartesian(anchor) {
  const [q, r] = Array.isArray(anchor) ? anchor : [0, 0];
  return {
    x: Math.sqrt(3) * (Number(q) + Number(r) / 2),
    y: 1.5 * Number(r),
  };
}

const startTemplatePreviewModel = computed(() => {
  const template = selectedStartTemplateDefinition.value;
  const courtWidth = Number(props.gameState?.court_width || 0);
  const courtHeight = Number(props.gameState?.court_height || 0);
  if (!template || courtWidth <= 0 || courtHeight <= 0) return null;

  const boardPoints = [];
  for (let row = 0; row < courtHeight; row += 1) {
    for (let col = 0; col < courtWidth; col += 1) {
      const axial = previewOffsetToAxial(col, row);
      boardPoints.push(previewAxialToCartesian([axial.q, axial.r]));
    }
  }
  if (!boardPoints.length) return null;

  const xs = boardPoints.map((point) => point.x);
  const ys = boardPoints.map((point) => point.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);

  const viewBox = { width: 360, height: 240 };
  const frame = { x: 18, y: 14, width: 324, height: 212 };
  const rangeX = Math.max(1e-6, maxX - minX);
  const rangeY = Math.max(1e-6, maxY - minY);
  const scale = Math.min(frame.width / rangeX, frame.height / rangeY);
  const usedWidth = rangeX * scale;
  const usedHeight = rangeY * scale;
  const originX = frame.x + (frame.width - usedWidth) / 2 - minX * scale;
  const originY = frame.y + (frame.height - usedHeight) / 2 - minY * scale;

  const mirrored = Boolean(selectedStartTemplateMirrored.value && template?.mirrorable);
  const entries = [];
  for (const teamName of ['offense', 'defense']) {
    const teamEntries = Array.isArray(template?.[teamName]) ? template[teamName] : [];
    teamEntries.forEach((entry, idx) => {
      const anchor = clampTemplateAnchorToCourt(
        Array.isArray(entry?.anchor) ? [...entry.anchor] : [0, 0],
        courtWidth,
        courtHeight,
      );
      const effectiveAnchor = mirrored
        ? previewMirrorAnchor(anchor, courtHeight)
        : anchor;
      const cart = previewAxialToCartesian(effectiveAnchor);
      entries.push({
        team: teamName,
        marker: teamName === 'offense'
          ? (entry?.has_ball ? 'O.' : 'O')
          : 'X',
        x: originX + cart.x * scale,
        y: originY + cart.y * scale,
        role: String(entry?.role || ''),
        key: `${teamName}-${idx}`,
      });
    });
  }

  const court = {
    ...frame,
    backboardX: frame.x + 16,
    hoopX: frame.x + 24,
    hoopY: frame.y + frame.height / 2,
    hoopRadius: 4,
    laneX: frame.x,
    laneY: frame.y + frame.height * 0.28,
    laneWidth: frame.width * 0.21,
    laneHeight: frame.height * 0.44,
    arcLineX: frame.x + frame.width * 0.29,
    arcTopY: frame.y + frame.height * 0.16,
    arcBottomY: frame.y + frame.height * 0.84,
    arcRadius: frame.height * 0.39,
  };

  return {
    templateId: String(template?.id || ''),
    mirrored,
    viewBox,
    court,
    entries,
  };
});

async function handleApplyStartTemplateToBoard() {
  clearStartTemplateFeedback();
  if (!selectedStartTemplateId.value) return;
  try {
    const res = await applyStartTemplate(
      selectedStartTemplateId.value,
      selectedStartTemplateMirrored.value,
      true,
      stableStartTemplateSeed(selectedStartTemplateId.value),
    );
    if (res?.state) {
      emit('state-updated', res.state);
    }
    startTemplateActionStatus.value = `Loaded ${selectedStartTemplateId.value} onto the current board.`;
  } catch (err) {
    console.error('[PlayerControls] Failed to apply start template to board', err);
    startTemplateActionError.value = err?.message || 'Failed to load template onto board.';
  }
}

const evalConfigSafe = computed(() => {
  const base = defaultEvalConfig();
  const incoming = props.evalConfig || {};
  return {
    ...base,
    ...incoming,
    skills: { ...base.skills, ...(incoming.skills || {}) },
  };
});
const templateConfigSafe = computed(() => {
  const base = defaultTemplateConfig();
  const incoming = props.templateConfig || {};
  return {
    ...base,
    ...incoming,
    jitterByPlayer: { ...base.jitterByPlayer, ...(incoming.jitterByPlayer || {}) },
    roleByPlayer: { ...base.roleByPlayer, ...(incoming.roleByPlayer || {}) },
  };
});

const evalPlacementEditing = computed(() => evalConfigSafe.value.placementEditing && evalConfigSafe.value.mode === 'custom');
const evalModeIsCustom = computed(() => evalConfigSafe.value.mode === 'custom');
const evalOffenseIds = computed(() => props.gameState?.offense_ids || []);
const templateOffenseIds = computed(() => props.gameState?.offense_ids || []);
const templateDefenseIds = computed(() => props.gameState?.defense_ids || []);
const ballStartOptions = computed(() => {
  const ids = evalOffenseIds.value || [];
  return Array.isArray(ids) ? [...ids] : [];
});
const templateBallStartOptions = computed(() => {
  const ids = templateOffenseIds.value || [];
  return Array.isArray(ids) ? [...ids] : [];
});
const templateCopyStatus = ref('');

function sanitizeTemplateId(raw) {
  const text = String(raw ?? '').trim().toLowerCase();
  if (!text) return 'new_template';
  const cleaned = text.replace(/[^a-z0-9_ -]/g, '').replace(/\s+/g, '_').replace(/-+/g, '_');
  return cleaned || 'new_template';
}

function emitTemplateConfigUpdate(patch = {}) {
  const base = templateConfigSafe.value;
  const next = {
    ...base,
    ...(patch || {}),
    jitterByPlayer: {
      ...base.jitterByPlayer,
      ...(patch.jitterByPlayer || {}),
    },
    roleByPlayer: {
      ...base.roleByPlayer,
      ...(patch.roleByPlayer || {}),
    },
  };
  emit('template-config-changed', next);
}

function seedTemplateConfigFromGameState() {
  if (!props.gameState) return;
  const positions = (props.gameState.positions || []).map((pos) => [pos[0], pos[1]]);
  const jitterByPlayer = {};
  const roleByPlayer = {};
  const offenseIds = Array.isArray(props.gameState.offense_ids) ? props.gameState.offense_ids : [];
  const defenseIds = Array.isArray(props.gameState.defense_ids) ? props.gameState.defense_ids : [];
  const ballHolder = props.gameState.ball_holder ?? templateConfigSafe.value.ballHolder ?? null;

  offenseIds.forEach((pid) => {
    jitterByPlayer[pid] = Number(
      templateConfigSafe.value.jitterByPlayer?.[pid]
      ?? templateConfigSafe.value.jitterByPlayer?.[String(pid)]
      ?? (Number(pid) === Number(ballHolder) ? 0 : 1)
    );
    roleByPlayer[pid] = String(
      templateConfigSafe.value.roleByPlayer?.[pid]
      ?? templateConfigSafe.value.roleByPlayer?.[String(pid)]
      ?? (Number(pid) === Number(ballHolder) ? 'ball_handler' : '')
    );
  });
  defenseIds.forEach((pid) => {
    jitterByPlayer[pid] = Number(
      templateConfigSafe.value.jitterByPlayer?.[pid]
      ?? templateConfigSafe.value.jitterByPlayer?.[String(pid)]
      ?? 1
    );
    roleByPlayer[pid] = String(
      templateConfigSafe.value.roleByPlayer?.[pid]
      ?? templateConfigSafe.value.roleByPlayer?.[String(pid)]
      ?? ''
    );
  });

  emitTemplateConfigUpdate({
    positions,
    ballHolder,
    shotClock: Number(props.gameState.shot_clock ?? templateConfigSafe.value.shotClock ?? 24),
    jitterByPlayer,
    roleByPlayer,
  });
}

function setTemplateBallHolder(val) {
  const num = val === null || val === '' ? null : Number(val);
  if (num !== null && !templateBallStartOptions.value.includes(num)) return;
  const nextRoles = {};
  for (const pid of templateBallStartOptions.value) {
    const currentRole = String(
      templateConfigSafe.value.roleByPlayer?.[pid]
      ?? templateConfigSafe.value.roleByPlayer?.[String(pid)]
      ?? ''
    );
    nextRoles[pid] = Number(pid) === Number(num)
      ? 'ball_handler'
      : (currentRole === 'ball_handler' ? '' : currentRole);
  }
  emitTemplateConfigUpdate({
    ballHolder: num,
    roleByPlayer: nextRoles,
  });
}

function updateTemplatePlayerJitter(playerId, value) {
  const num = Math.max(0, Number(value) || 0);
  emitTemplateConfigUpdate({
    jitterByPlayer: {
      [playerId]: num,
    },
  });
}

function updateTemplatePlayerRole(playerId, value) {
  emitTemplateConfigUpdate({
    roleByPlayer: {
      [playerId]: String(value ?? ''),
    },
  });
}

function updateTemplatePlayerAnchor(playerId, axisIndex, value) {
  const positions = Array.isArray(templateConfigSafe.value.positions)
    ? [...templateConfigSafe.value.positions]
    : [];
  const current = Array.isArray(positions[playerId]) ? [...positions[playerId]] : [0, 0];
  current[axisIndex] = Math.round(Number(value) || 0);
  positions[playerId] = clampTemplateAnchorToCourt(
    current,
    props.gameState?.court_width,
    props.gameState?.court_height,
  );
  emitTemplateConfigUpdate({ positions });
}

function setTemplateId(value) {
  emitTemplateConfigUpdate({ templateId: sanitizeTemplateId(value) });
}

const templatePlayerRows = computed(() => {
  const positions = Array.isArray(templateConfigSafe.value.positions) ? templateConfigSafe.value.positions : [];
  const rows = [];
  const buildRows = (ids, teamLabel) => {
    (ids || []).forEach((pid, entryIndex) => {
      const pos = Array.isArray(positions[pid]) ? positions[pid] : [null, null];
      rows.push({
        playerId: pid,
        teamLabel,
        entryIndex,
        q: pos[0],
        r: pos[1],
        jitter: Number(
          templateConfigSafe.value.jitterByPlayer?.[pid]
          ?? templateConfigSafe.value.jitterByPlayer?.[String(pid)]
          ?? (teamLabel === 'Offense' && Number(pid) === Number(templateConfigSafe.value.ballHolder) ? 0 : 1)
        ),
        role: String(
          templateConfigSafe.value.roleByPlayer?.[pid]
          ?? templateConfigSafe.value.roleByPlayer?.[String(pid)]
          ?? ''
        ),
      });
    });
  };
  buildRows(templateOffenseIds.value, 'Offense');
  buildRows(templateDefenseIds.value, 'Defense');
  return rows;
});

function buildTemplateExportEntries(playerIds) {
  const positions = Array.isArray(templateConfigSafe.value.positions) ? templateConfigSafe.value.positions : [];
  return (playerIds || []).map((pid) => {
    const pos = clampTemplateAnchorToCourt(
      Array.isArray(positions[pid]) ? positions[pid] : [0, 0],
      props.gameState?.court_width,
      props.gameState?.court_height,
    );
    const entry = {
      anchor: [Number(pos[0]) || 0, Number(pos[1]) || 0],
      jitter_radius: Math.max(
        0,
        Number(
          templateConfigSafe.value.jitterByPlayer?.[pid]
          ?? templateConfigSafe.value.jitterByPlayer?.[String(pid)]
          ?? (Number(pid) === Number(templateConfigSafe.value.ballHolder) ? 0 : 1)
        ) || 0
      ),
    };
    const role = String(
      templateConfigSafe.value.roleByPlayer?.[pid]
      ?? templateConfigSafe.value.roleByPlayer?.[String(pid)]
      ?? ''
    ).trim();
    if (role) entry.role = role;
    if (Number(pid) === Number(templateConfigSafe.value.ballHolder)) {
      entry.has_ball = true;
    }
    return entry;
  });
}

const templateExportObject = computed(() => {
  const offenseIds = templateOffenseIds.value || [];
  const defenseIds = templateDefenseIds.value || [];
  const obj = {
    id: sanitizeTemplateId(templateConfigSafe.value.templateId),
    weight: Number(templateConfigSafe.value.weight ?? 1.0) || 1.0,
    mirrorable: Boolean(templateConfigSafe.value.mirrorable),
    shot_clock: Math.max(1, Number(templateConfigSafe.value.shotClock ?? 24) || 24),
    offense: buildTemplateExportEntries(offenseIds),
    defense: buildTemplateExportEntries(defenseIds),
  };
  return obj;
});

function loadTemplateIntoAuthoringConfig(template) {
  if (!template) return;
  const positions = deepCloneJson(props.gameState?.positions, []);
  const offenseIds = templateOffenseIds.value || [];
  const defenseIds = templateDefenseIds.value || [];
  const jitterByPlayer = {};
  const roleByPlayer = {};
  let ballHolder = null;

  template.offense?.forEach((entry, idx) => {
    const pid = offenseIds[idx];
    if (pid === undefined) return;
    positions[pid] = clampTemplateAnchorToCourt(
      [Number(entry?.anchor?.[0]) || 0, Number(entry?.anchor?.[1]) || 0],
      props.gameState?.court_width,
      props.gameState?.court_height,
    );
    jitterByPlayer[pid] = Math.max(0, Number(entry?.jitter_radius ?? 0) || 0);
    roleByPlayer[pid] = String(entry?.role || '');
    if (entry?.has_ball) {
      ballHolder = Number(pid);
    }
  });
  template.defense?.forEach((entry, idx) => {
    const pid = defenseIds[idx];
    if (pid === undefined) return;
    positions[pid] = clampTemplateAnchorToCourt(
      [Number(entry?.anchor?.[0]) || 0, Number(entry?.anchor?.[1]) || 0],
      props.gameState?.court_width,
      props.gameState?.court_height,
    );
    jitterByPlayer[pid] = Math.max(0, Number(entry?.jitter_radius ?? 0) || 0);
    roleByPlayer[pid] = String(entry?.role || '');
  });

  suppressTemplateConfigBackfill.value = true;
  emitTemplateConfigUpdate({
    positions,
    ballHolder,
    templateId: String(template?.id || 'new_template'),
    weight: Number(template?.weight ?? 1.0) || 1.0,
    mirrorable: Boolean(template?.mirrorable),
    shotClock: Math.max(1, Number(template?.shot_clock ?? 24) || 24),
    jitterByPlayer,
    roleByPlayer,
  });
  nextTick(() => {
    suppressTemplateConfigBackfill.value = false;
  });
}

function replaceSelectedDraftTemplate(nextTemplate, { markDirty = true } = {}) {
  if (!nextTemplate || selectedEditableTemplateIndex.value < 0) return;
  const normalized = deepCloneJson(nextTemplate, {});
  if (!normalized || typeof normalized !== 'object') return;
  const nextTemplates = Array.isArray(templateLibraryDraft.value?.templates)
    ? [...templateLibraryDraft.value.templates]
    : [];
  nextTemplates.splice(selectedEditableTemplateIndex.value, 1, normalized);
  templateLibraryDraft.value = {
    ...normalizeTemplateLibraryDraft(templateLibraryDraft.value),
    templates: nextTemplates,
  };
  const nextId = String(normalized.id || '').trim();
  if (nextId) {
    selectedEditableTemplateId.value = nextId;
  }
  if (markDirty) {
    templateLibraryDirty.value = true;
  }
}

watch(
  selectedEditableTemplateId,
  () => {
    if (!selectedEditableTemplate.value) return;
    loadTemplateIntoAuthoringConfig(selectedEditableTemplate.value);
  }
);

watch(
  templateExportObject,
  (nextTemplate) => {
    if (suppressTemplateConfigBackfill.value) return;
    if (selectedEditableTemplateIndex.value < 0) return;
    const selectedId = String(selectedEditableTemplateId.value || '').trim();
    const authoringId = String(sanitizeTemplateId(templateConfigSafe.value.templateId) || '').trim();
    if (selectedId && authoringId && selectedId !== authoringId) return;
    replaceSelectedDraftTemplate(nextTemplate, { markDirty: true });
  },
  { deep: true }
);

function yamlScalar(value) {
  if (typeof value === 'number') return String(value);
  if (typeof value === 'boolean') return value ? 'true' : 'false';
  return JSON.stringify(String(value ?? ''));
}

function formatTemplateYaml(obj) {
  const lines = [];
  lines.push(`id: ${yamlScalar(obj.id)}`);
  lines.push(`weight: ${obj.weight}`);
  lines.push(`mirrorable: ${obj.mirrorable ? 'true' : 'false'}`);
  lines.push(`shot_clock: ${obj.shot_clock}`);
  for (const teamName of ['offense', 'defense']) {
    lines.push(`${teamName}:`);
    for (const entry of obj[teamName]) {
      lines.push('  -');
      lines.push(`    anchor: [${entry.anchor[0]}, ${entry.anchor[1]}]`);
      lines.push(`    jitter_radius: ${entry.jitter_radius}`);
      if (entry.role) {
        lines.push(`    role: ${yamlScalar(entry.role)}`);
      }
      if (entry.has_ball) {
        lines.push('    has_ball: true');
      }
    }
  }
  return lines.join('\n');
}

function formatTemplateLibraryYaml(library) {
  const normalized = normalizeTemplateLibraryDraft(library);
  const lines = [];
  lines.push(`version: ${Number(normalized.version || 1)}`);
  lines.push(`players_per_side: ${Number(normalized.players_per_side || playersPerSideForTemplates.value)}`);
  lines.push('templates:');
  for (const template of normalized.templates || []) {
    const block = formatTemplateYaml(template).split('\n');
    lines.push('  -');
    block.forEach((line, idx) => {
      if (idx === 0) {
        lines.push(`    ${line}`);
      } else {
        lines.push(`    ${line}`);
      }
    });
  }
  return lines.join('\n');
}

function getTemplateLibrarySerializedPayload(filenameHint = '') {
  const filename = String(filenameHint || '').trim().toLowerCase();
  const library = normalizeTemplateLibraryDraft(templateLibraryDraft.value);
  const useYaml = filename.endsWith('.yaml') || filename.endsWith('.yml');
  return {
    contents: useYaml
      ? formatTemplateLibraryYaml(library)
      : JSON.stringify(library, null, 2),
    mimeType: useYaml ? 'text/yaml;charset=utf-8' : 'application/json;charset=utf-8',
    extension: useYaml ? (filename.endsWith('.yml') ? '.yml' : '.yaml') : '.json',
  };
}

const templateJsonExport = computed(() => JSON.stringify(templateExportObject.value, null, 2));
const templateYamlExport = computed(() => formatTemplateYaml(templateExportObject.value));

async function copyTemplateExport(format) {
  const text = format === 'json' ? templateJsonExport.value : templateYamlExport.value;
  try {
    if (navigator?.clipboard?.writeText) {
      await navigator.clipboard.writeText(text);
    } else {
      throw new Error('Clipboard API unavailable');
    }
    templateCopyStatus.value = `${String(format).toUpperCase()} copied`;
  } catch (err) {
    console.error('[PlayerControls] Failed to copy template export', err);
    templateCopyStatus.value = 'Copy failed';
  }
  window.setTimeout(() => {
    if (templateCopyStatus.value === `${String(format).toUpperCase()} copied` || templateCopyStatus.value === 'Copy failed') {
      templateCopyStatus.value = '';
    }
  }, 2000);
}

async function copyTemplateLibraryJson() {
  try {
    if (navigator?.clipboard?.writeText) {
      await navigator.clipboard.writeText(templateLibraryJsonExport.value);
    } else {
      throw new Error('Clipboard API unavailable');
    }
    templateCopyStatus.value = 'LIB copied';
  } catch (err) {
    console.error('[PlayerControls] Failed to copy template library export', err);
    templateCopyStatus.value = 'Copy failed';
  }
  window.setTimeout(() => {
    if (templateCopyStatus.value === 'LIB copied' || templateCopyStatus.value === 'Copy failed') {
      templateCopyStatus.value = '';
    }
  }, 2000);
}

function addTemplateFromCurrentBoard() {
  const nextTemplate = deepCloneJson(templateExportObject.value, {});
  nextTemplate.id = uniqueTemplateId(nextTemplate.id || 'new_template');
  const nextTemplates = Array.isArray(templateLibraryDraft.value?.templates)
    ? [...templateLibraryDraft.value.templates, nextTemplate]
    : [nextTemplate];
  templateLibraryDraft.value = {
    ...normalizeTemplateLibraryDraft(templateLibraryDraft.value),
    templates: nextTemplates,
  };
  selectedEditableTemplateId.value = nextTemplate.id;
  templateLibraryDirty.value = true;
  clearTemplateLibraryFeedback();
  templateLibraryStatus.value = `Added ${nextTemplate.id} from current board.`;
}

function duplicateSelectedTemplate() {
  const current = selectedEditableTemplate.value;
  if (!current) return;
  const copyTemplate = deepCloneJson(current, {});
  copyTemplate.id = uniqueTemplateId(`${current.id}_copy`);
  const nextTemplates = Array.isArray(templateLibraryDraft.value?.templates)
    ? [...templateLibraryDraft.value.templates, copyTemplate]
    : [copyTemplate];
  templateLibraryDraft.value = {
    ...normalizeTemplateLibraryDraft(templateLibraryDraft.value),
    templates: nextTemplates,
  };
  selectedEditableTemplateId.value = copyTemplate.id;
  templateLibraryDirty.value = true;
  clearTemplateLibraryFeedback();
  templateLibraryStatus.value = `Duplicated ${current.id} to ${copyTemplate.id}.`;
}

function deleteSelectedTemplate() {
  const current = selectedEditableTemplate.value;
  if (!current) return;
  const nextTemplates = (templateLibraryDraft.value?.templates || []).filter(
    (template) => String(template?.id || '').trim() !== String(current.id || '').trim()
  );
  templateLibraryDraft.value = {
    ...normalizeTemplateLibraryDraft(templateLibraryDraft.value),
    templates: nextTemplates,
  };
  selectedEditableTemplateId.value = nextTemplates[0]?.id || '';
  templateLibraryDirty.value = true;
  clearTemplateLibraryFeedback();
  templateLibraryStatus.value = `Deleted ${current.id}.`;
  if (selectedEditableTemplateId.value) {
    nextTick(() => {
      if (selectedEditableTemplate.value) {
        loadTemplateIntoAuthoringConfig(selectedEditableTemplate.value);
      }
    });
  }
}

async function handleLoadTemplateLibraryFile() {
  clearTemplateLibraryFeedback();
  const path = String(templateLibraryPathInput.value || '').trim();
  if (!path) {
    templateLibraryError.value = 'Provide a YAML or JSON template library path.';
    return;
  }
  templateLibraryRequestPending.value = true;
  try {
    const res = await loadStartTemplateLibrary(path);
    if (res?.state) {
      emit('state-updated', res.state);
    }
    templateLibraryDraft.value = normalizeTemplateLibraryDraft(res?.library || buildDefaultTemplateLibrary());
    templateLibraryPathInput.value = String(res?.path || path);
    selectedEditableTemplateId.value = templateLibraryDraft.value.templates?.[0]?.id || '';
    templateLibraryDirty.value = false;
    lastLoadedTemplateLibrarySignature = buildTemplateLibrarySignature(
      templateLibraryDraft.value,
      'local_file',
      templateLibraryPathInput.value,
    );
    templateLibraryStatus.value = `Loaded ${templateLibraryDraft.value.templates.length} templates from ${templateLibraryPathInput.value}.`;
    if (selectedEditableTemplate.value) {
      loadTemplateIntoAuthoringConfig(selectedEditableTemplate.value);
    }
  } catch (err) {
    console.error('[PlayerControls] Failed to load template library', err);
    templateLibraryError.value = err?.message || 'Failed to load template library.';
  } finally {
    templateLibraryRequestPending.value = false;
  }
}

function triggerTemplateLibraryChooser() {
  clearTemplateLibraryFeedback();
  templateLibraryFileInput.value?.click?.();
}

async function handleTemplateLibraryFileChosen(event) {
  clearTemplateLibraryFeedback();
  const file = event?.target?.files?.[0] || null;
  if (!file) return;
  templateLibraryRequestPending.value = true;
  try {
    const contents = await file.text();
    const res = await importStartTemplateLibrary(file.name, contents);
    if (res?.state) {
      emit('state-updated', res.state);
    }
    templateLibraryDraft.value = normalizeTemplateLibraryDraft(res?.library || buildDefaultTemplateLibrary());
    templateLibraryImportedFilename.value = String(res?.path || file.name || '');
    selectedEditableTemplateId.value = templateLibraryDraft.value.templates?.[0]?.id || '';
    templateLibraryDirty.value = false;
    lastLoadedTemplateLibrarySignature = buildTemplateLibrarySignature(
      templateLibraryDraft.value,
      res?.source || 'file_upload',
      templateLibraryImportedFilename.value,
    );
    templateLibraryStatus.value = `Imported ${templateLibraryDraft.value.templates.length} templates from ${file.name}.`;
    if (selectedEditableTemplate.value) {
      loadTemplateIntoAuthoringConfig(selectedEditableTemplate.value);
    }
  } catch (err) {
    console.error('[PlayerControls] Failed to import template library file', err);
    templateLibraryError.value = err?.message || 'Failed to import template library.';
  } finally {
    if (event?.target) {
      event.target.value = '';
    }
    templateLibraryRequestPending.value = false;
  }
}

async function handlePushTemplateLibraryDraft() {
  clearTemplateLibraryFeedback();
  templateLibraryRequestPending.value = true;
  try {
    const res = await setStartTemplateLibrary(
      templateLibraryDraft.value,
      'session_editor',
      templateLibraryPathInput.value || null,
    );
    if (res?.state) {
      emit('state-updated', res.state);
    }
    templateLibraryDraft.value = normalizeTemplateLibraryDraft(res?.library || templateLibraryDraft.value);
    templateLibraryDirty.value = false;
    lastLoadedTemplateLibrarySignature = buildTemplateLibrarySignature(
      templateLibraryDraft.value,
      res?.source || 'session_editor',
      res?.path || templateLibraryPathInput.value,
    );
    templateLibraryStatus.value = 'Draft pushed into the active session library.';
  } catch (err) {
    console.error('[PlayerControls] Failed to push template library draft', err);
    templateLibraryError.value = err?.message || 'Failed to update session template library.';
  } finally {
    templateLibraryRequestPending.value = false;
  }
}

async function handleSaveTemplateLibraryFile() {
  clearTemplateLibraryFeedback();
  const path = String(templateLibraryPathInput.value || '').trim();
  if (!path) {
    templateLibraryError.value = 'Provide a file path before saving.';
    return;
  }
  templateLibraryRequestPending.value = true;
  try {
    const res = await saveStartTemplateLibrary(path, templateLibraryDraft.value);
    if (res?.state) {
      emit('state-updated', res.state);
    }
    templateLibraryDraft.value = normalizeTemplateLibraryDraft(res?.library || templateLibraryDraft.value);
    templateLibraryPathInput.value = String(res?.path || path);
    templateLibraryDirty.value = false;
    lastLoadedTemplateLibrarySignature = buildTemplateLibrarySignature(
      templateLibraryDraft.value,
      res?.source || 'local_file',
      templateLibraryPathInput.value,
    );
    templateLibraryStatus.value = `Saved template library to ${templateLibraryPathInput.value}.`;
  } catch (err) {
    console.error('[PlayerControls] Failed to save template library', err);
    templateLibraryError.value = err?.message || 'Failed to save template library.';
  } finally {
    templateLibraryRequestPending.value = false;
  }
}

function handleReloadTemplateLibraryDraft() {
  syncTemplateLibraryDraftFromState();
  templateLibraryStatus.value = 'Reloaded the current session template library.';
}

function downloadTemplateLibraryFile(downloadName = '') {
  clearTemplateLibraryFeedback();
  try {
    const filename = String(
      downloadName
      || templateLibraryPathInput.value
      || templateLibraryImportedFilename.value
      || ''
    ).trim() || 'start_templates_library.json';
    const suggestedName = /\.(ya?ml|json)$/i.test(filename)
      ? filename.split('/').pop()
      : `${filename}.json`;
    const payload = getTemplateLibrarySerializedPayload(suggestedName);
    const blob = new Blob([payload.contents], { type: payload.mimeType });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = suggestedName;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
    templateLibraryStatus.value = `Downloaded ${suggestedName}.`;
  } catch (err) {
    console.error('[PlayerControls] Failed to download template library', err);
    templateLibraryError.value = 'Failed to download template library.';
  }
}

async function handleSaveTemplateLibraryChooser() {
  clearTemplateLibraryFeedback();
  try {
    const validated = await setStartTemplateLibrary(
      templateLibraryDraft.value,
      'session_editor',
      templateLibraryPathInput.value || null,
    );
    if (validated?.state) {
      emit('state-updated', validated.state);
    }
    templateLibraryDraft.value = normalizeTemplateLibraryDraft(validated?.library || templateLibraryDraft.value);
    templateLibraryDirty.value = false;

    const preferredName = String(
      templateLibraryImportedFilename.value
      || templateLibraryPathInput.value
      || 'start_templates_library.json'
    ).trim().split('/').pop() || 'start_templates_library.json';

    if (typeof window !== 'undefined' && typeof window.showSaveFilePicker === 'function') {
      const picker = await window.showSaveFilePicker({
        suggestedName: preferredName,
        types: [
          {
            description: 'Template libraries',
            accept: {
              'application/json': ['.json'],
              'text/yaml': ['.yaml', '.yml'],
            },
          },
        ],
      });
      const payload = getTemplateLibrarySerializedPayload(picker?.name || preferredName);
      const writable = await picker.createWritable();
      await writable.write(payload.contents);
      await writable.close();
      templateLibraryStatus.value = `Saved template library to ${picker?.name || preferredName}.`;
      return;
    }

    downloadTemplateLibraryFile(preferredName);
    templateLibraryStatus.value = `Browser save dialog unavailable; downloaded ${preferredName} instead.`;
  } catch (err) {
    if (err?.name === 'AbortError') {
      templateLibraryStatus.value = 'Save cancelled.';
      return;
    }
    console.error('[PlayerControls] Failed to save template library with chooser', err);
    templateLibraryError.value = err?.message || 'Failed to save template library.';
  }
}

const evalEpisodesInput = ref(props.evalNumEpisodes || 100);
watch(() => props.evalNumEpisodes, (val) => {
  const safe = Number.isFinite(val) ? Number(val) : 100;
  evalEpisodesInput.value = safe;
});

const evalProgressSafe = computed(() => {
  const incoming = props.evalProgress && typeof props.evalProgress === 'object'
    ? props.evalProgress
    : {};
  const total = Math.max(0, Number(incoming.total || evalEpisodesInput.value || 0));
  const rawCompleted = Math.max(0, Number(incoming.completed || 0));
  const completed = total > 0 ? Math.min(total, rawCompleted) : rawCompleted;
  const fraction = total > 0
    ? Math.max(0, Math.min(1, Number.isFinite(Number(incoming.fraction)) ? Number(incoming.fraction) : (completed / total)))
    : 0;
  return {
    running: Boolean(incoming.running),
    completed,
    total,
    fraction,
    status: String(incoming.status || 'idle'),
    error: incoming.error || null,
  };
});

const evalProgressPercent = computed(() => `${(evalProgressSafe.value.fraction * 100).toFixed(1)}%`);

const playbookProgressSafe = computed(() => {
  const incoming = playbookProgress.value && typeof playbookProgress.value === 'object'
    ? playbookProgress.value
    : {};
  const total = Math.max(0, Number(incoming.total || 0));
  const rawCompleted = Math.max(0, Number(incoming.completed || 0));
  const completed = total > 0 ? Math.min(total, rawCompleted) : rawCompleted;
  const fraction = total > 0
    ? Math.max(0, Math.min(1, Number.isFinite(Number(incoming.fraction)) ? Number(incoming.fraction) : (completed / total)))
    : 0;
  return {
    running: Boolean(incoming.running),
    completed,
    total,
    fraction,
    status: String(incoming.status || 'idle'),
    error: incoming.error || null,
  };
});

const playbookProgressPercent = computed(() => `${(playbookProgressSafe.value.fraction * 100).toFixed(1)}%`);

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

function setEvalIntentSelectionMode(mode) {
  const normalized = ['learned_sample', 'best_intent', 'uniform_random'].includes(String(mode))
    ? String(mode)
    : 'learned_sample';
  emitEvalConfigUpdate({ intentSelectionMode: normalized });
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
    activeTab.value = 'eval';
  } else {
    emitEvalConfigUpdate({ mode: 'default', placementEditing: false });
    if (activeTab.value === 'eval') {
      activeTab.value = 'rewards';
    }
  }
  emit('active-tab-changed', activeTab.value);
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

async function applyIntentStateOverride() {
  if (!props.gameState || intentControlsDisabled.value) return;
  intentStateUpdating.value = true;
  intentStateError.value = null;
  try {
    const payload = {
      active: Boolean(intentStateInput.value.active),
      intent_index: Math.max(
        0,
        Math.min(intentIndexMax.value, Number(intentStateInput.value.intent_index) || 0),
      ),
      intent_age: Math.max(
        0,
        Math.min(intentAgeMax.value, Number(intentStateInput.value.intent_age) || 0),
      ),
    };
    const res = await setIntentState(payload);
    if (res?.status === 'success' && res.state) {
      emit('state-updated', res.state);
    } else {
      throw new Error(res?.detail || 'Failed to update intent state');
    }
  } catch (err) {
    console.error('[PlayerControls] Failed to update intent state', err);
    intentStateError.value = err?.message || 'Failed to update intent state';
  } finally {
    intentStateUpdating.value = false;
  }
}

function resetIntentStateInputs() {
  intentStateInput.value = {
    active: Boolean(props.gameState?.intent_active_current),
    intent_index: Math.max(
      0,
      Math.min(intentIndexMax.value, Number(props.gameState?.intent_index_current ?? 0) || 0),
    ),
    intent_age: Math.max(
      0,
      Math.min(intentAgeMax.value, Number(props.gameState?.intent_age ?? 0) || 0),
    ),
  };
  intentStateError.value = null;
}

async function handleCaptureCounterfactualSnapshot() {
  if (!props.gameState || counterfactualSnapshotControlsDisabled.value) return;
  counterfactualSnapshotUpdating.value = true;
  counterfactualSnapshotError.value = null;
  try {
    const res = await captureCounterfactualSnapshot();
    if (res?.status === 'success' && res.state) {
      emit('state-updated', res.state);
    } else {
      throw new Error(res?.detail || 'Failed to capture snapshot');
    }
  } catch (err) {
    console.error('[PlayerControls] Failed to capture counterfactual snapshot', err);
    counterfactualSnapshotError.value = err?.message || 'Failed to capture snapshot';
  } finally {
    counterfactualSnapshotUpdating.value = false;
  }
}

async function handleRestoreCounterfactualSnapshot() {
  if (!props.gameState || counterfactualSnapshotControlsDisabled.value) return;
  counterfactualSnapshotUpdating.value = true;
  counterfactualSnapshotError.value = null;
  try {
    const res = await restoreCounterfactualSnapshot();
    if (res?.status === 'success' && res.state) {
      emit('state-updated', res.state);
    } else {
      throw new Error(res?.detail || 'Failed to restore snapshot');
    }
  } catch (err) {
    console.error('[PlayerControls] Failed to restore counterfactual snapshot', err);
    counterfactualSnapshotError.value = err?.message || 'Failed to restore snapshot';
  } finally {
    counterfactualSnapshotUpdating.value = false;
  }
}

async function handleReplayCounterfactualSnapshot() {
  if (
    !props.gameState ||
    counterfactualSnapshotControlsDisabled.value ||
    !counterfactualSnapshotAvailable.value ||
    props.gameState.done
  ) {
    return;
  }
  counterfactualSnapshotUpdating.value = true;
  counterfactualSnapshotError.value = null;
  try {
    const res = await replayCounterfactualSnapshot({
      player_deterministic: true,
      opponent_deterministic: true,
    });
    if (res?.status === 'success' && Array.isArray(res.states) && res.states.length > 0) {
      emit('counterfactual-replay-loaded', res);
    } else {
      throw new Error(res?.detail || 'Failed to replay from current branch state');
    }
  } catch (err) {
    console.error('[PlayerControls] Failed to replay counterfactual snapshot', err);
    counterfactualSnapshotError.value = err?.message || 'Failed to replay current state';
  } finally {
    counterfactualSnapshotUpdating.value = false;
  }
}

function resetPlaybookProgressState(total = 0) {
  playbookProgress.value = {
    running: total > 0,
    completed: 0,
    total: Math.max(0, Number(total || 0)),
    fraction: 0,
    status: total > 0 ? 'running' : 'idle',
    error: null,
  };
}

function stopPlaybookProgressPolling() {
  if (playbookProgressPollHandle !== null) {
    clearInterval(playbookProgressPollHandle);
    playbookProgressPollHandle = null;
  }
}

async function pollPlaybookProgressOnce() {
  if (playbookProgressPollBusy) return;
  playbookProgressPollBusy = true;
  try {
    const payload = await getPlaybookProgress();
    playbookProgress.value = {
      ...playbookProgress.value,
      ...(payload || {}),
    };
  } catch (err) {
    console.error('[PlayerControls] Failed to fetch playbook progress', err);
  } finally {
    playbookProgressPollBusy = false;
  }
}

function startPlaybookProgressPolling(total = 0) {
  stopPlaybookProgressPolling();
  resetPlaybookProgressState(total);
  void pollPlaybookProgressOnce();
  playbookProgressPollHandle = window.setInterval(() => {
    void pollPlaybookProgressOnce();
  }, 750);
}

function getPlaybookShotStatsForPanel(panel) {
  const rawByPlayer = panel?.shot_stats?.by_player || {};
  const offenseIds = Array.isArray(props.gameState?.offense_ids)
    ? props.gameState.offense_ids.map((id) => Number(id)).filter((id) => Number.isFinite(id))
    : [];
  const seen = new Set();
  const orderedIds = [];
  for (const pid of offenseIds) {
    const key = String(pid);
    seen.add(key);
    orderedIds.push(key);
  }
  for (const key of Object.keys(rawByPlayer)) {
    if (!seen.has(String(key))) {
      orderedIds.push(String(key));
    }
  }
  return orderedIds.map((key) => {
    const stats = rawByPlayer?.[key] || {};
    return {
      playerId: Number(key),
      attempts: Number(stats?.attempts || 0),
      makes: Number(stats?.makes || 0),
    };
  });
}

const playbookOffenseColumnIds = computed(() => {
  const ids = [];
  const seen = new Set();
  const pushId = (rawId) => {
    const playerId = Number(rawId);
    if (!Number.isFinite(playerId)) return;
    const key = String(playerId);
    if (seen.has(key)) return;
    seen.add(key);
    ids.push(playerId);
  };

  const liveOffenseIds = Array.isArray(props.gameState?.offense_ids) ? props.gameState.offense_ids : [];
  liveOffenseIds.forEach(pushId);

  const panels = Array.isArray(playbookResult.value?.panels) ? playbookResult.value.panels : [];
  panels.forEach((panel) => {
    getPlaybookShotStatsForPanel(panel).forEach((entry) => pushId(entry?.playerId));
  });

  return ids;
});

function formatPlaybookTotalShots(panel) {
  const attempts = Number(panel?.shot_stats?.total?.attempts || 0);
  const makes = Number(panel?.shot_stats?.total?.makes || 0);
  return `${attempts} / ${makes}`;
}

function formatPlaybookPlayerShotCell(panel, playerId) {
  const match = getPlaybookShotStatsForPanel(panel).find((entry) => Number(entry?.playerId) === Number(playerId));
  if (!match) return '0 / 0';
  return `${Number(match.attempts || 0)} / ${Number(match.makes || 0)}`;
}

function getPlaybookPrimaryShooterSummary(panel) {
  const raw = panel?.primary_shooter_distribution || {};
  const entries = Object.entries(raw)
    .map(([key, value]) => {
      const playerId = Number(key);
      const count = Number(value?.count || 0);
      const rate = Number(value?.rate || 0);
      if (!Number.isFinite(playerId) || count <= 0) return null;
      return { playerId, count, rate };
    })
    .filter(Boolean)
    .sort((a, b) => {
      if (b.count !== a.count) return b.count - a.count;
      return a.playerId - b.playerId;
    });
  if (!entries.length) return 'none';
  return entries
    .map((entry) => `P${entry.playerId} ${entry.count} (${(entry.rate * 100).toFixed(0)}%)`)
    .join(', ');
}

function formatPlaybookOutcomeSummary(panel) {
  const raw = panel?.terminal_outcomes || {};
  const entries = Object.entries(raw)
    .map(([key, count]) => ({
      key: String(key),
      count: Number(count || 0),
      label: formatPlaybookOutcomeKey(key),
    }))
    .filter((entry) => entry.count > 0)
    .sort((a, b) => {
      if (b.count !== a.count) return b.count - a.count;
      return a.label.localeCompare(b.label);
    });
  if (!entries.length) return 'none';
  return entries.map((entry) => `${entry.label}: ${entry.count}`).join(', ');
}

function formatPlaybookOutcomeKey(rawKey) {
  const key = String(rawKey || '').trim();
  switch (key) {
    case 'shot_make':
      return 'Made shots';
    case 'shot_miss':
      return 'Missed shots';
    case 'turnover':
      return 'Turnovers';
    case 'defensive_violation':
      return 'Defensive violations';
    case 'shot_clock_expiration':
      return 'Shot clock expirations';
    case 'horizon_cutoff':
      return 'Horizon cutoffs';
    case 'other_terminal':
      return 'Other terminal';
    default:
      return key.replaceAll('_', ' ');
  }
}

function formatPlaybookTurnoverReason(rawKey) {
  const key = String(rawKey || '').trim();
  switch (key) {
    case 'steal':
      return 'Steal';
    case 'out_of_bounds':
      return 'Out of bounds';
    case 'pressure':
      return 'Pressure';
    case 'offensive_three_seconds':
      return 'Offensive 3 seconds';
    case 'unknown':
      return 'Unknown';
    default:
      return key.replaceAll('_', ' ');
  }
}

function formatPlaybookTurnoverSummary(panel) {
  const raw = panel?.turnover_reasons || {};
  const entries = Object.entries(raw)
    .map(([key, count]) => ({
      key: String(key),
      count: Number(count || 0),
      label: formatPlaybookTurnoverReason(key),
    }))
    .filter((entry) => entry.count > 0)
    .sort((a, b) => {
      if (b.count !== a.count) return b.count - a.count;
      return a.label.localeCompare(b.label);
    });
  if (!entries.length) return 'none';
  return entries.map((entry) => `${entry.label}: ${entry.count}`).join(', ');
}

async function handleRunPlaybookAnalysis() {
  if (!props.gameState || playbookControlsDisabled.value) return;
  if (!playbookSelectedIntentIndices.value.length) {
    playbookError.value = 'Provide at least one valid intent index.';
    return;
  }
  if (playbookUseSnapshot.value && !counterfactualSnapshotAvailable.value) {
    playbookError.value = 'Capture a snapshot first, or switch the source to current state.';
    return;
  }

  playbookRunning.value = true;
  playbookError.value = null;
  const totalRollouts = playbookSelectedIntentIndices.value.length * Number(playbookNumRollouts.value || 16);
  startPlaybookProgressPolling(totalRollouts);
  try {
    const res = await runPlaybookAnalysis({
      intent_indices: playbookSelectedIntentIndices.value,
      num_rollouts: Number(playbookNumRollouts.value || 16),
      max_steps: Number(playbookMaxSteps.value || 8),
      run_to_end: Boolean(playbookRunToEnd.value),
      use_snapshot: Boolean(playbookUseSnapshot.value),
      player_deterministic: Boolean(playbookPlayerDeterministic.value),
      opponent_deterministic: Boolean(playbookOpponentDeterministic.value),
    });
    if (res?.status === 'success') {
      playbookResult.value = res;
      emit('playbook-analysis-loaded', res);
      playbookProgress.value = {
        ...playbookProgress.value,
        running: false,
        completed: Number(res?.total_rollouts || totalRollouts || 0),
        total: Number(res?.total_rollouts || totalRollouts || 0),
        fraction: 1,
        status: 'completed',
        error: null,
      };
    } else {
      throw new Error(res?.detail || 'Failed to generate playbook analysis');
    }
  } catch (err) {
    console.error('[PlayerControls] Failed to run playbook analysis', err);
    playbookError.value = err?.message || 'Failed to run playbook analysis';
    playbookProgress.value = {
      ...playbookProgress.value,
      running: false,
      status: 'failed',
      error: err?.message || 'Failed to run playbook analysis',
    };
  } finally {
    stopPlaybookProgressPolling();
    playbookRunning.value = false;
  }
}

function _normalizePressurePayload(input) {
  return {
    three_pt_extra_hex_decay: Number(input?.three_pt_extra_hex_decay ?? 0.05),
    shot_pressure_enabled: Boolean(input?.shot_pressure_enabled),
    shot_pressure_max: Number(input?.shot_pressure_max ?? 0.5),
    shot_pressure_lambda: Number(input?.shot_pressure_lambda ?? 1.0),
    shot_pressure_arc_degrees: Number(input?.shot_pressure_arc_degrees ?? 60.0),
    base_steal_rate: Number(input?.base_steal_rate ?? 0.35),
    steal_perp_decay: Number(input?.steal_perp_decay ?? 1.5),
    steal_distance_factor: Number(input?.steal_distance_factor ?? 0.08),
    steal_position_weight_min: Number(input?.steal_position_weight_min ?? 0.3),
    defender_pressure_distance: Math.round(Number(input?.defender_pressure_distance ?? 1)),
    defender_pressure_turnover_chance: Number(input?.defender_pressure_turnover_chance ?? 0.05),
    defender_pressure_decay_lambda: Number(input?.defender_pressure_decay_lambda ?? 1.0),
  };
}

function _buildPressureSubsetPayload(keys) {
  const normalized = _normalizePressurePayload(pressureParamsInput.value);
  const payload = {};
  for (const key of keys || []) {
    if (Object.prototype.hasOwnProperty.call(normalized, key)) {
      payload[key] = normalized[key];
    }
  }
  return payload;
}

function _buildSectionResetPayload(keys) {
  const defaults = props.gameState?.mlflow_env_defaults || {};
  const fallbackState = props.gameState || {};
  const payload = {};
  for (const key of keys || []) {
    const defaultVal = defaults[key];
    if (defaultVal !== undefined && defaultVal !== null) {
      payload[key] = defaultVal;
      continue;
    }
    const stateVal = fallbackState[key];
    if (stateVal !== undefined && stateVal !== null) {
      payload[key] = stateVal;
    }
  }
  return payload;
}

async function _submitPressureParams(
  requestFn,
  payload,
  fallbackErrorMessage,
  syncKeys = null,
  section = null
) {
  if (!props.gameState) return;
  if (pressureParamsUpdating.value) return;
  pressureParamsUpdating.value = true;
  pressureParamsError.value = null;
  activePressureUpdateSection.value = section || null;
  pendingPressureSyncKeys.value = Array.isArray(syncKeys) ? [...syncKeys] : null;
  try {
    const reqPayload = payload || {};
    const res = await requestFn(reqPayload);
    if (res?.status === 'success' && res.state) {
      // Ignore the immediate props-sync from this emitted state update.
      // We explicitly sync only the intended section below.
      suppressNextPressurePropsSync.value = true;
      emit('state-updated', res.state);
      syncPressureParamsInputs(res.state);
    } else {
      throw new Error(res?.detail || fallbackErrorMessage);
    }
  } catch (err) {
    console.error('[PlayerControls] Failed to update pressure parameters', err);
    pressureParamsError.value = err?.message || fallbackErrorMessage;
    pendingPressureSyncKeys.value = null;
  } finally {
    activePressureUpdateSection.value = null;
    pressureParamsUpdating.value = false;
  }
}

async function applyPressureParameterOverrides() {
  const payload = _buildPressureSubsetPayload(SHOT_PRESSURE_PARAM_KEYS);
  await _submitPressureParams(
    setShotPressureParams,
    payload,
    'Failed to update shot pressure parameters',
    SHOT_PRESSURE_PARAM_KEYS,
    'shot'
  );
}

async function resetPressureParametersToMlflowDefaults() {
  const payload = _buildSectionResetPayload(SHOT_PRESSURE_PARAM_KEYS);
  if (Object.keys(payload).length > 0) {
    await _submitPressureParams(
      setShotPressureParams,
      payload,
      'Failed to reset shot pressure parameters',
      SHOT_PRESSURE_PARAM_KEYS,
      'shot'
    );
    return;
  }
  await _submitPressureParams(
    setShotPressureParams,
    { reset_to_mlflow_defaults: true },
    'Failed to reset shot pressure parameters',
    SHOT_PRESSURE_PARAM_KEYS,
    'shot'
  );
}

async function applyDefenderTurnoverPressureOverrides() {
  const payload = _buildPressureSubsetPayload(DEFENDER_PRESSURE_PARAM_KEYS);
  await _submitPressureParams(
    setDefenderPressureParams,
    payload,
    'Failed to update defender turnover pressure parameters',
    DEFENDER_PRESSURE_PARAM_KEYS,
    'defender'
  );
}

async function resetDefenderTurnoverPressureToMlflowDefaults() {
  const payload = _buildSectionResetPayload(DEFENDER_PRESSURE_PARAM_KEYS);
  if (Object.keys(payload).length > 0) {
    await _submitPressureParams(
      setDefenderPressureParams,
      payload,
      'Failed to reset defender turnover pressure parameters',
      DEFENDER_PRESSURE_PARAM_KEYS,
      'defender'
    );
    return;
  }
  await _submitPressureParams(
    setDefenderPressureParams,
    { reset_to_mlflow_defaults: true },
    'Failed to reset defender turnover pressure parameters',
    DEFENDER_PRESSURE_PARAM_KEYS,
    'defender'
  );
}

async function applyPassInterceptionOverrides() {
  const payload = _buildPressureSubsetPayload(PASS_INTERCEPTION_PARAM_KEYS);
  await _submitPressureParams(
    setPassInterceptionParams,
    payload,
    'Failed to update pass interception parameters',
    PASS_INTERCEPTION_PARAM_KEYS,
    'pass'
  );
}

async function resetPassInterceptionToMlflowDefaults() {
  const payload = _buildSectionResetPayload(PASS_INTERCEPTION_PARAM_KEYS);
  if (Object.keys(payload).length > 0) {
    await _submitPressureParams(
      setPassInterceptionParams,
      payload,
      'Failed to reset pass interception parameters',
      PASS_INTERCEPTION_PARAM_KEYS,
      'pass'
    );
    return;
  }
  await _submitPressureParams(
    setPassInterceptionParams,
    { reset_to_mlflow_defaults: true },
    'Failed to reset pass interception parameters',
    PASS_INTERCEPTION_PARAM_KEYS,
    'pass'
  );
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
      selectedPassTargets.value = {};
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
const DEV_TABS_STORAGE_KEY = 'basketworld.dev.tab_order';
const DEFAULT_DEV_TABS = Object.freeze([
  { id: 'environment', label: 'Environment' },
  { id: 'template', label: 'Template' },
  { id: 'rewards', label: 'Rewards' },
  { id: 'stats', label: 'Stats' },
  { id: 'entropy', label: 'Entropy' },
  { id: 'policy', label: 'Policy' },
  { id: 'playbook', label: 'Playbook' },
  { id: 'advisor', label: 'Advisor' },
  { id: 'moves', label: 'Moves' },
  { id: 'eval', label: 'Eval' },
  { id: 'training', label: 'Training' },
  { id: 'phi', label: 'Phi Shaping' },
  { id: 'observation', label: 'Observation' },
  { id: 'attention', label: 'Attention' },
]);
const DEFAULT_DEV_TAB_ORDER = DEFAULT_DEV_TABS.map((tab) => tab.id);
const DEV_TAB_ID_SET = new Set(DEFAULT_DEV_TAB_ORDER);

function normalizeDevTabOrder(rawOrder) {
  const candidateOrder = Array.isArray(rawOrder) ? rawOrder : [];
  const seen = new Set();
  const normalized = [];
  for (const tabId of candidateOrder) {
    if (typeof tabId !== 'string' || seen.has(tabId) || !DEV_TAB_ID_SET.has(tabId)) continue;
    seen.add(tabId);
    normalized.push(tabId);
  }
  for (const tabId of DEFAULT_DEV_TAB_ORDER) {
    if (!seen.has(tabId)) normalized.push(tabId);
  }
  return normalized;
}

function loadStoredDevTabOrder() {
  if (typeof window === 'undefined') return [...DEFAULT_DEV_TAB_ORDER];
  try {
    const raw = window.localStorage.getItem(DEV_TABS_STORAGE_KEY);
    if (!raw) return [...DEFAULT_DEV_TAB_ORDER];
    return normalizeDevTabOrder(JSON.parse(raw));
  } catch (err) {
    console.warn('[PlayerControls] Failed to load tab order preference', err);
    return [...DEFAULT_DEV_TAB_ORDER];
  }
}

const activeTab = ref(String(props.initialActiveTab || 'environment'));
const devTabOrder = ref(loadStoredDevTabOrder());
const draggedDevTabId = ref(null);
const orderedDevTabs = computed(() => {
  const tabsById = new Map(DEFAULT_DEV_TABS.map((tab) => [tab.id, tab]));
  return devTabOrder.value
    .map((tabId) => tabsById.get(tabId))
    .filter((tab) => Boolean(tab));
});
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

watch(() => props.initialActiveTab, (nextTab) => {
  const normalized = String(nextTab || '').trim();
  if (!normalized || activeTab.value === normalized) return;
  if (!DEV_TAB_ID_SET.has(normalized)) return;
  activeTab.value = normalized;
});

watch(devTabOrder, (newOrder) => {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.setItem(DEV_TABS_STORAGE_KEY, JSON.stringify(newOrder));
  } catch (err) {
    console.warn('[PlayerControls] Failed to persist tab order preference', err);
  }
});

function handleDevTabDragStart(tabId, event) {
  draggedDevTabId.value = tabId;
  if (event?.dataTransfer) {
    event.dataTransfer.effectAllowed = 'move';
    event.dataTransfer.setData('text/plain', tabId);
  }
}

function handleDevTabDragOver(event) {
  if (!draggedDevTabId.value) return;
  event.dataTransfer.dropEffect = 'move';
}

function handleDevTabDrop(targetTabId) {
  const sourceTabId = draggedDevTabId.value;
  draggedDevTabId.value = null;
  if (!sourceTabId || sourceTabId === targetTabId) return;
  const nextOrder = [...devTabOrder.value];
  const sourceIndex = nextOrder.indexOf(sourceTabId);
  const targetIndex = nextOrder.indexOf(targetTabId);
  if (sourceIndex === -1 || targetIndex === -1) return;
  nextOrder.splice(sourceIndex, 1);
  nextOrder.splice(targetIndex, 0, sourceTabId);
  devTabOrder.value = nextOrder;
}

function handleDevTabDragEnd() {
  draggedDevTabId.value = null;
}

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
const totalViolations = computed(() => (
  Number(statsState.value?.violations?.defensiveLane || 0)
  + Number(statsState.value?.violations?.offensiveThreeSeconds || 0)
));
const ppp = computed(() => safeDiv(statsState.value.points, Math.max(1, statsState.value.episodes)));
const avgRewardPerEp = computed(() => safeDiv(statsState.value.rewardSum, Math.max(1, statsState.value.episodes)));
const avgEpisodeLen = computed(() => safeDiv(statsState.value.episodeStepsSum, Math.max(1, statsState.value.episodes)));

function ensureStatsDiagnosticFields(target) {
  if (!target || typeof target !== 'object') return;
  if (!target.intentSelectionCounts || typeof target.intentSelectionCounts !== 'object') {
    target.intentSelectionCounts = {};
  }
  target.intentInactiveCount = Number(target.intentInactiveCount || 0);
  if (!target.turnoverReasons || typeof target.turnoverReasons !== 'object') {
    target.turnoverReasons = {};
  }
  if (!target.actionMix || typeof target.actionMix !== 'object') {
    target.actionMix = {};
  }
  if (!target.rewardBreakdown || typeof target.rewardBreakdown !== 'object') {
    target.rewardBreakdown = {};
  }
  target.actionMix.noop = Number(target.actionMix.noop || 0);
  target.actionMix.move = Number(target.actionMix.move || 0);
  target.actionMix.shoot = Number(target.actionMix.shoot || 0);
  target.actionMix.pass = Number(target.actionMix.pass || 0);
  target.actionMix.other = Number(target.actionMix.other || 0);
  target.actionMix.total = Number(target.actionMix.total || 0);
  target.rewardBreakdown.totalReward = Number(target.rewardBreakdown.totalReward || 0);
  target.rewardBreakdown.expectedPoints = Number(target.rewardBreakdown.expectedPoints || 0);
  target.rewardBreakdown.passReward = Number(target.rewardBreakdown.passReward || 0);
  target.rewardBreakdown.violationReward = Number(target.rewardBreakdown.violationReward || 0);
  target.rewardBreakdown.assistPotential = Number(target.rewardBreakdown.assistPotential || 0);
  target.rewardBreakdown.assistFullBonus = Number(target.rewardBreakdown.assistFullBonus || 0);
  target.rewardBreakdown.phiShaping = Number(target.rewardBreakdown.phiShaping || 0);
  target.rewardBreakdown.unexplained = Number(target.rewardBreakdown.unexplained || 0);
}

ensureStatsDiagnosticFields(statsState.value);

function formatTurnoverReason(reason) {
  const map = {
    intercepted: 'Intercepted',
    steal: 'Intercepted',
    defender_pressure: 'Pressure',
    pass_out_of_bounds: 'Pass OOB',
    move_out_of_bounds: 'Move OOB',
    offensive_three_seconds: 'Offensive 3-second violation',
    shot_clock_violation: 'Shot Clock',
  };
  return map[String(reason)] || String(reason || 'unknown');
}

function getTurnoverReasonTooltip(reason) {
  const map = {
    intercepted: 'A defender intercepted a pass before it reached the target.',
    steal: 'Legacy label for an intercepted pass turnover.',
    defender_pressure: 'The ball handler turned it over due to defender pressure probability.',
    pass_out_of_bounds: 'The selected pass trajectory went out of bounds.',
    move_out_of_bounds: 'The ball handler moved out of bounds.',
    offensive_three_seconds: 'An offensive player stayed in the lane longer than the 3-second limit.',
    shot_clock_violation: 'Possession ended because the shot clock expired.',
  };
  return map[String(reason)] || 'Turnover category reported by the environment.';
}

function getActionMixTooltip(key) {
  const map = {
    noop: 'Player selected NOOP (no action this step).',
    move: 'Player selected one of the six movement directions.',
    shoot: 'Player selected SHOOT.',
    pass: 'Player selected a PASS action.',
    other: 'Any non-standard action id outside NOOP/MOVE/SHOOT/PASS buckets.',
  };
  return map[String(key)] || 'Action category count and rate over evaluated decisions.';
}

function getRewardBreakdownTooltip(key) {
  const map = {
    totalReward: 'Sum of all user-team rewards across evaluated episodes.',
    expectedPoints: 'Shot expected-points term (shot value × pressure-adjusted make probability).',
    passReward: 'Reward term from successful passes.',
    assistPotential: 'Potential-assist shaping reward component.',
    assistFullBonus: 'Full-assist bonus reward component.',
    violationReward: 'Reward term from defensive-lane violations.',
    phiShaping: 'Potential-based shaping component (if phi shaping is enabled).',
    unexplained: 'Residual: totalReward minus tracked components.',
  };
  return map[String(key)] || 'Reward component contribution.';
}

const turnoverReasonRows = computed(() => {
  const rows = Object.entries(statsState.value?.turnoverReasons || {}).map(([reason, count]) => ({
    reason,
    label: formatTurnoverReason(reason),
    count: Number(count || 0),
  }));
  rows.sort((a, b) => b.count - a.count);
  return rows;
});

const actionMixRows = computed(() => {
  const mix = statsState.value?.actionMix || {};
  const total = Number(mix.total || 0);
  const asRow = (key, label) => {
    const count = Number(mix[key] || 0);
    return {
      key,
      label,
      count,
      rate: total > 0 ? (count / total) * 100 : 0,
    };
  };
  return {
    total,
    rows: [
      asRow('noop', 'NOOP'),
      asRow('move', 'MOVE'),
      asRow('shoot', 'SHOOT'),
      asRow('pass', 'PASS'),
      asRow('other', 'OTHER'),
    ],
  };
});

const intentSelectionRows = computed(() => {
  const raw = statsState.value?.intentSelectionCounts || {};
  const rows = Object.entries(raw)
    .map(([intent, count]) => ({
      intent: Number(intent),
      label: formatPlayLabel(Number(intent), props.gameState?.play_name_map),
      count: Number(count || 0),
    }))
    .filter((row) => Number.isFinite(row.intent))
    .sort((a, b) => a.intent - b.intent);
  return {
    rows,
    inactiveCount: Number(statsState.value?.intentInactiveCount || 0),
    total: rows.reduce((acc, row) => acc + row.count, 0) + Number(statsState.value?.intentInactiveCount || 0),
  };
});

const rewardBreakdownRows = computed(() => {
  const rb = statsState.value?.rewardBreakdown || {};
  return [
    { key: 'totalReward', label: 'Total reward', value: Number(rb.totalReward || 0) },
    { key: 'expectedPoints', label: 'Expected points', value: Number(rb.expectedPoints || 0) },
    { key: 'passReward', label: 'Pass reward', value: Number(rb.passReward || 0) },
    { key: 'assistPotential', label: 'Potential assist', value: Number(rb.assistPotential || 0) },
    { key: 'assistFullBonus', label: 'Full assist bonus', value: Number(rb.assistFullBonus || 0) },
    { key: 'violationReward', label: 'Violation reward', value: Number(rb.violationReward || 0) },
    { key: 'phiShaping', label: 'Phi shaping', value: Number(rb.phiShaping || 0) },
    { key: 'unexplained', label: 'Unexplained', value: Number(rb.unexplained || 0) },
  ];
});

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

function getShotTypePair(shotTypes, key) {
  const vals = shotTypes?.[key] || [0, 0];
  return {
    att: Number(vals[0] || 0),
    mk: Number(vals[1] || 0),
  };
}

function buildEvalAggregateRow(entry) {
  const shotTypes = entry?.shot_types || { dunk: [0, 0], two: [0, 0], three: [0, 0] };
  const assistByType = entry?.assist_full_by_type || {};
  const dunk = getShotTypePair(shotTypes, 'dunk');
  const two = getShotTypePair(shotTypes, 'two');
  const three = getShotTypePair(shotTypes, 'three');
  const attempts = Number(entry?.shots || 0);
  const makes = Number(entry?.makes || 0);
  const episodes = Number(entry?.episodes || 0);
  const points = Number(entry?.points || 0);
  return {
    attempts,
    makes,
    fg: attempts > 0 ? (makes / attempts) * 100 : 0,
    dunk,
    two,
    three,
    assists: Number(entry?.assists || 0),
    potentialAssists: Number(entry?.potential_assists || 0),
    turnovers: Number(entry?.turnovers || 0),
    points,
    episodes,
    ppp: episodes > 0 ? points / episodes : 0,
    unassisted: {
      dunk: Math.max(0, dunk.mk - Number(assistByType.dunk || 0)),
      two: Math.max(0, two.mk - Number(assistByType.two || 0)),
      three: Math.max(0, three.mk - Number(assistByType.three || 0)),
    },
  };
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
    const row = buildEvalAggregateRow(entry);
    return {
      playerId: pid,
      ...row,
    };
  });
});

const perIntentEvalStatsTable = computed(() => {
  const stats = props.perIntentEvalStats || {};
  const rows = Object.entries(stats)
    .map(([intentKey, entry]) => {
      const base = buildEvalAggregateRow(entry || {});
      const idx = Number(intentKey);
      return {
        intentKey: String(intentKey),
        label:
          String(intentKey) === 'none'
            ? 'No intent'
            : (Number.isFinite(idx)
              ? formatPlayLabel(idx, props.gameState?.play_name_map)
              : String(intentKey)),
        ...base,
      };
    })
    .filter((row) => row.episodes > 0 || row.attempts > 0 || row.points > 0);
  rows.sort((a, b) => {
    if (a.intentKey === 'none') return 1;
    if (b.intentKey === 'none') return -1;
    const ai = Number(a.intentKey);
    const bi = Number(b.intentKey);
    if (Number.isFinite(ai) && Number.isFinite(bi)) return ai - bi;
    return a.label.localeCompare(b.label);
  });
  return rows;
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
  ensureStatsDiagnosticFields(statsState.value);
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
  const turnovers = Array.isArray(results?.turnovers) ? results.turnovers : [];
  const tovCount = turnovers.length;
  statsState.value.turnovers += Number(tovCount || 0);
  for (const turnover of turnovers) {
    const reason = String(turnover?.reason || 'unknown');
    statsState.value.turnoverReasons[reason] = Number(statsState.value.turnoverReasons[reason] || 0) + 1;
  }
  if (!statsState.value.violations) {
    statsState.value.violations = { defensiveLane: 0, offensiveThreeSeconds: 0 };
  }
  const defensiveLaneCount = Array.isArray(results?.defensive_lane_violations)
    ? results.defensive_lane_violations.length
    : 0;
  const offensiveThreeCount = turnovers.filter(
    (turnover) => String(turnover?.reason || '') === 'offensive_three_seconds'
  ).length;
  // Defensive 3-second violation awards offense 1 point.
  statsState.value.points += Number(defensiveLaneCount || 0);
  statsState.value.violations.defensiveLane += Number(defensiveLaneCount || 0);
  statsState.value.violations.offensiveThreeSeconds += Number(offensiveThreeCount || 0);

  // Add episode reward for user's team
  let episodeRewardAdded = 0;
  // If episodeData is provided (from evaluation), use it directly
  // Otherwise, fetch from API if not skipping
  if (episodeData && episodeData.episode_rewards && episodeData.steps !== undefined) {
    const ep = episodeData.episode_rewards;
    const userTeam = finalState?.user_team_name || 'OFFENSE';
    episodeRewardAdded = Number(userTeam === 'OFFENSE' ? ep.offense : ep.defense) || 0;
    statsState.value.rewardSum += episodeRewardAdded;
    statsState.value.episodeStepsSum += Number(episodeData.steps || 0);
  } else if (!skipApiCall) {
    try {
      const data = await getRewards();
      const ep = data?.episode_rewards || { offense: 0.0, defense: 0.0 };
      const userTeam = finalState?.user_team_name || 'OFFENSE';
      episodeRewardAdded = Number(userTeam === 'OFFENSE' ? ep.offense : ep.defense) || 0;
      statsState.value.rewardSum += episodeRewardAdded;
      const steps = Array.isArray(data?.reward_history) ? data.reward_history.length : 0;
      statsState.value.episodeStepsSum += Number(steps || 0);
    } catch (_) { /* ignore */ }
  }
  statsState.value.rewardBreakdown.totalReward += Number(episodeRewardAdded || 0);
  statsState.value.rewardBreakdown.unexplained += Number(episodeRewardAdded || 0);

  // Increment episode count last
  statsState.value.episodes += 1;
  saveStats(statsState.value);
}

function applyEvaluationStats(
  episodeResults = [],
  perPlayerStats = {},
  userTeamName = 'OFFENSE',
  evalDiagnostics = null,
) {
  const next = resetStatsStorage();
  ensureStatsDiagnosticFields(next);
  const statsByPlayer = perPlayerStats || {};

  const offenseIds = (props.gameState?.offense_ids || []).map((id) => Number(id));
  const defenseIds = (props.gameState?.defense_ids || []).map((id) => Number(id));

  let teamIds = userTeamName === 'DEFENSE' ? defenseIds : offenseIds;
  if (!Array.isArray(teamIds) || teamIds.length === 0) {
    const allIds = Object.keys(statsByPlayer)
      .map((k) => Number(k))
      .filter((n) => Number.isFinite(n))
      .sort((a, b) => a - b);
    const half = Math.floor(allIds.length / 2);
    teamIds = userTeamName === 'DEFENSE' ? allIds.slice(half) : allIds.slice(0, half);
  }

  for (const pid of teamIds) {
    const entry = statsByPlayer?.[pid] || statsByPlayer?.[String(pid)] || null;
    if (!entry) continue;

    const shotTypes = entry.shot_types || {};
    const assistByType = entry.assist_full_by_type || {};

    const dunk = shotTypes.dunk || [0, 0];
    const two = shotTypes.two || [0, 0];
    const three = shotTypes.three || [0, 0];

    next.dunk.attempts += Number(dunk[0] || 0);
    next.dunk.made += Number(dunk[1] || 0);
    next.dunk.assists += Number(assistByType.dunk || 0);

    next.twoPt.attempts += Number(two[0] || 0);
    next.twoPt.made += Number(two[1] || 0);
    next.twoPt.assists += Number(assistByType.two || 0);

    next.threePt.attempts += Number(three[0] || 0);
    next.threePt.made += Number(three[1] || 0);
    next.threePt.assists += Number(assistByType.three || 0);

    next.turnovers += Number(entry.turnovers || 0);
    next.points += Number(entry.points || 0);
  }

  const teamIdSet = new Set(teamIds.map((id) => Number(id)));
  for (const row of episodeResults || []) {
    const shots = row?.final_state?.last_action_results?.shots || {};
    for (const [shooterRaw, shot] of Object.entries(shots)) {
      const shooterId = Number(shooterRaw);
      if (!Number.isFinite(shooterId) || !teamIdSet.has(shooterId)) continue;
      const made = Boolean(shot?.success);
      const potentialAssisted = Boolean(shot?.assist_potential) && !made;
      if (!potentialAssisted) continue;
      const distance = Number(shot?.distance ?? 9999);
      const isDunk = distance === 0;
      const isThree = !isDunk && distance >= Number(row?.final_state?.three_point_distance ?? 4);
      if (isDunk) {
        next.dunk.potentialAssists += 1;
      } else if (isThree) {
        next.threePt.potentialAssists += 1;
      } else {
        next.twoPt.potentialAssists += 1;
      }
    }
  }

  let defensiveLaneCount = 0;
  let offensiveThreeCount = 0;
  for (const row of episodeResults || []) {
    const results = row?.final_state?.last_action_results || {};
    const defLane = Array.isArray(results?.defensive_lane_violations)
      ? results.defensive_lane_violations.length
      : 0;
    defensiveLaneCount += Number(defLane || 0);
    const turnovers = Array.isArray(results?.turnovers) ? results.turnovers : [];
    offensiveThreeCount += turnovers.filter(
      (turnover) => String(turnover?.reason || '') === 'offensive_three_seconds'
    ).length;
  }

  next.violations = {
    defensiveLane: Number(defensiveLaneCount || 0),
    offensiveThreeSeconds: Number(offensiveThreeCount || 0),
  };
  // Defensive lane violation awards one point to offense.
  if (String(userTeamName || 'OFFENSE').toUpperCase() === 'OFFENSE') {
    next.points += Number(defensiveLaneCount || 0);
  }

  for (const row of episodeResults || []) {
    next.episodes += 1;
    next.episodeStepsSum += Number(row?.steps || 0);
    const epRewards = row?.episode_rewards || {};
    const rewardVal = userTeamName === 'DEFENSE' ? epRewards?.defense : epRewards?.offense;
    next.rewardSum += Number(rewardVal || 0);
  }

  if (evalDiagnostics && typeof evalDiagnostics === 'object') {
    const intentRaw = evalDiagnostics.intent_selection_counts || {};
    next.intentSelectionCounts = {};
    for (const [intent, count] of Object.entries(intentRaw)) {
      const idx = Number(intent);
      if (!Number.isFinite(idx)) continue;
      next.intentSelectionCounts[String(idx)] = Number(count || 0);
    }
    next.intentInactiveCount = Number(evalDiagnostics.intent_inactive_count || 0);

    const reasonsRaw = evalDiagnostics.turnover_reasons || {};
    next.turnoverReasons = {};
    for (const [reason, count] of Object.entries(reasonsRaw)) {
      next.turnoverReasons[String(reason)] = Number(count || 0);
    }

    const mixRaw = evalDiagnostics.action_mix || {};
    next.actionMix = {
      noop: Number(mixRaw.noop || 0),
      move: Number(mixRaw.move || 0),
      shoot: Number(mixRaw.shoot || 0),
      pass: Number(mixRaw.pass || 0),
      other: Number(mixRaw.other || 0),
      total: Number(mixRaw.total || 0),
    };

    const rbRaw = evalDiagnostics.reward_breakdown || {};
    next.rewardBreakdown = {
      totalReward: Number(rbRaw.total_reward ?? next.rewardSum ?? 0),
      expectedPoints: Number(rbRaw.expected_points || 0),
      passReward: Number(rbRaw.pass_reward || 0),
      violationReward: Number(rbRaw.violation_reward || 0),
      assistPotential: Number(rbRaw.assist_potential || 0),
      assistFullBonus: Number(rbRaw.assist_full_bonus || 0),
      phiShaping: Number(rbRaw.phi_shaping || 0),
      unexplained: Number(rbRaw.unexplained ?? next.rewardSum ?? 0),
    };
  } else {
    next.rewardBreakdown.totalReward = Number(next.rewardSum || 0);
    next.rewardBreakdown.unexplained = Number(next.rewardSum || 0);
  }

  statsState.value = next;
  saveStats(statsState.value);
}

function resetStats() {
  statsState.value = resetStatsStorage();
  emit('stats-reset');
}

async function copyStatsMarkdown() {
  try {
    const s = statsState.value;
    ensureStatsDiagnosticFields(s);
    const fg = (made, att) => (safeDiv(made, Math.max(1, att)) * 100).toFixed(1) + '%';
    const summary = [
      ['Episodes', String(s.episodes)],
      ['PPP', ppp.value.toFixed(2)],
      ['Avg reward/ep', avgRewardPerEp.value.toFixed(2)],
      ['Avg ep length (steps)', safeDiv(s.episodeStepsSum, Math.max(1, s.episodes)).toFixed(1)],
      ['Total assists', String(s.dunk.assists + s.twoPt.assists + s.threePt.assists)],
      ['Total potential assists', String(s.dunk.potentialAssists + s.twoPt.potentialAssists + s.threePt.potentialAssists)],
      ['Total turnovers', String(s.turnovers)],
      ['Total violations', String((s.violations?.defensiveLane || 0) + (s.violations?.offensiveThreeSeconds || 0))],
      ['Illegal defense violations', String(s.violations?.defensiveLane || 0)],
      ['Offensive 3-second violations', String(s.violations?.offensiveThreeSeconds || 0)],
    ];
    const turnoverRows = Object.entries(s.turnoverReasons || {})
      .map(([reason, count]) => [formatTurnoverReason(reason), String(Number(count || 0))])
      .sort((a, b) => Number(b[1]) - Number(a[1]));
    const actionTotal = Number(s.actionMix?.total || 0);
    const actionRows = [
      ['NOOP', Number(s.actionMix?.noop || 0)],
      ['MOVE', Number(s.actionMix?.move || 0)],
      ['SHOOT', Number(s.actionMix?.shoot || 0)],
      ['PASS', Number(s.actionMix?.pass || 0)],
      ['OTHER', Number(s.actionMix?.other || 0)],
    ].map(([label, count]) => [
      label,
      String(count),
      `${(actionTotal > 0 ? (Number(count) / actionTotal) * 100 : 0).toFixed(1)}%`,
    ]);
    const intentRows = Object.entries(s.intentSelectionCounts || {})
      .map(([intent, count]) => [
        formatPlayLabel(Number(intent), props.gameState?.play_name_map),
        String(Number(count || 0)),
        Number(intent),
      ])
      .sort((a, b) => Number(a[2]) - Number(b[2]))
      .map(([label, count]) => [label, count]);
    if (Number(s.intentInactiveCount || 0) > 0) {
      intentRows.push(['No intent', String(Number(s.intentInactiveCount || 0))]);
    }
    const rb = s.rewardBreakdown || {};
    const rewardRows = [
      ['Total reward', Number(rb.totalReward || 0).toFixed(2)],
      ['Expected points', Number(rb.expectedPoints || 0).toFixed(2)],
      ['Pass reward', Number(rb.passReward || 0).toFixed(2)],
      ['Potential assist', Number(rb.assistPotential || 0).toFixed(2)],
      ['Full assist bonus', Number(rb.assistFullBonus || 0).toFixed(2)],
      ['Violation reward', Number(rb.violationReward || 0).toFixed(2)],
      ['Phi shaping', Number(rb.phiShaping || 0).toFixed(2)],
      ['Unexplained', Number(rb.unexplained || 0).toFixed(2)],
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
      '## Turnovers by Reason',
      '| Reason | Count |',
      '| --- | --- |',
      table(turnoverRows.length ? turnoverRows : [['(none)', '0']]),
      '',
      '## Action Mix',
      '| Action | Count | Rate |',
      '| --- | --- | --- |',
      table(actionRows),
      '',
      '## Offense Intent Starts',
      '| Intent | Count |',
      '| --- | --- |',
      table(intentRows.length ? intentRows : [['(none)', '0']]),
      '',
      '## Reward Decomposition',
      '| Component | Value |',
      '| --- | --- |',
      table(rewardRows),
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
defineExpose({ resetStats, copyStatsMarkdown, submitActions, recordEpisodeStats, applyEvaluationStats, getSelectedActions });

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

const controlsTabPlayerIds = computed(() => {
  if (props.showOpponentActions) {
    return allPlayerIds.value;
  }
  return userControlledPlayerIds.value;
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

// entropy debug removed

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
const policyTeamLabel = computed(() => props.gameState?.user_team_name || 'OFFENSE');
const selectorIntentPreferences = computed(() => {
  const payload = props.gameState?.selector_intent_preferences;
  const items = Array.isArray(payload?.intent_probs)
    ? [...payload.intent_probs].sort(
      (a, b) => Number(a?.intent_index ?? 0) - Number(b?.intent_index ?? 0),
    )
    : [];
  return {
    alphaCurrent: payload?.alpha_current ?? null,
    epsCurrent: payload?.eps_current ?? null,
    selectionMode: payload?.selection_mode ?? 'learned_sample',
    valueEstimate: payload?.value_estimate ?? null,
    currentIntentIndex: payload?.current_intent_index ?? null,
    items,
  };
});
const playbookSelectedIntentLabels = computed(() => {
  const indices = Array.isArray(playbookSelectedIntentIndices.value)
    ? playbookSelectedIntentIndices.value
    : [];
  return indices.map((intentIndex) => formatPlayLabel(intentIndex, props.gameState?.play_name_map));
});
function computeSelectorIntentDistributionStats(items, fieldName) {
  const rows = Array.isArray(items) ? items : [];
  if (!rows.length) {
    return {
      entropy: null,
      normalizedEntropy: null,
      klToUniform: null,
      normalizedKlToUniform: null,
    };
  }
  const probs = rows
    .map((item) => Number(item?.[fieldName]))
    .filter((prob) => Number.isFinite(prob) && prob > 0);
  if (!probs.length) {
    return {
      entropy: null,
      normalizedEntropy: null,
      klToUniform: null,
      normalizedKlToUniform: null,
    };
  }
  const entropy = -probs.reduce((sum, prob) => sum + (prob * Math.log(prob)), 0);
  const maxEntropy = Math.log(rows.length);
  const normalizedEntropy = maxEntropy > 0 ? entropy / maxEntropy : null;
  const klToUniform = maxEntropy > 0 ? (maxEntropy - entropy) : 0;
  const normalizedKlToUniform = maxEntropy > 0 ? (klToUniform / maxEntropy) : null;
  return {
    entropy,
    normalizedEntropy,
    klToUniform,
    normalizedKlToUniform,
  };
}

const selectorIntentDistributionStats = computed(() => {
  const items = selectorIntentPreferences.value.items || [];
  return {
    raw: computeSelectorIntentDistributionStats(items, 'raw_prob'),
    mixed: computeSelectorIntentDistributionStats(items, 'mixed_prob'),
    deployed: computeSelectorIntentDistributionStats(items, 'deployed_prob'),
  };
});
const SELECTOR_INTENT_PLOT_MODE_STORAGE_KEY = 'basketworld.selector_intent_plot_mode';
function loadStoredSelectorIntentPlotMode() {
  if (typeof window === 'undefined') return 'deployed';
  try {
    const raw = String(window.localStorage.getItem(SELECTOR_INTENT_PLOT_MODE_STORAGE_KEY) || '').trim().toLowerCase();
    return ['raw', 'mixed', 'deployed'].includes(raw) ? raw : 'deployed';
  } catch (err) {
    console.warn('[PlayerControls] Failed to load selector intent plot mode preference', err);
    return 'deployed';
  }
}
const selectedSelectorIntentPlotMode = ref(loadStoredSelectorIntentPlotMode());
const selectorIntentPlotModeOptions = Object.freeze([
  { value: 'raw', label: 'Raw' },
  { value: 'mixed', label: 'Mixed' },
  { value: 'deployed', label: 'Deployed' },
]);
const selectorIntentPlotField = computed(() => {
  return selectedSelectorIntentPlotMode.value === 'raw'
    ? 'raw_prob'
    : selectedSelectorIntentPlotMode.value === 'mixed'
      ? 'mixed_prob'
      : 'deployed_prob';
});
const selectorIntentPlotLabel = computed(() => {
  return selectedSelectorIntentPlotMode.value === 'raw'
    ? 'Raw'
    : selectedSelectorIntentPlotMode.value === 'mixed'
      ? 'Mixed'
      : 'Deployed';
});
const selectorIntentEntropy = computed(() => {
  const stats = selectorIntentDistributionStats.value?.[selectedSelectorIntentPlotMode.value];
  return stats || {
    entropy: null,
    normalizedEntropy: null,
    klToUniform: null,
    normalizedKlToUniform: null,
  };
});
function getSelectorIntentPlotProb(item) {
  const field = selectorIntentPlotField.value;
  const prob = Number(item?.[field] ?? 0);
  return Number.isFinite(prob) ? prob : 0;
}
watch(selectedSelectorIntentPlotMode, (nextMode) => {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.setItem(
      SELECTOR_INTENT_PLOT_MODE_STORAGE_KEY,
      String(nextMode || 'deployed'),
    );
  } catch (err) {
    console.warn('[PlayerControls] Failed to persist selector intent plot mode preference', err);
  }
});
const hasSelectorIntentPreferences = computed(() =>
  Array.isArray(selectorIntentPreferences.value.items)
  && selectorIntentPreferences.value.items.length > 0
);
const policyRowsByPlayer = computed(() => {
  if (!props.gameState || !policyProbabilities.value) return [];

  return userControlledPlayerIds.value.map((pid) => {
    const probs =
      policyProbabilities.value?.[pid] ?? policyProbabilities.value?.[String(pid)];
    const mask = props.gameState?.action_mask?.[pid];
    const normalized = normalizeLegalProbs(probs, mask);
    if (!Array.isArray(normalized)) {
      return { playerId: pid, actions: [] };
    }

    const actions = [];
    for (let i = 0; i < normalized.length && i < actionNames.length; i += 1) {
      const allowed = !Array.isArray(mask) || Number(mask[i]) > 0;
      if (!allowed) continue;
      const prob = Number(normalized[i]);
      if (!Number.isFinite(prob)) continue;
      actions.push({
        action: actionNames[i] || `ACTION_${i}`,
        prob,
      });
    }

    actions.sort((a, b) => String(a.action).localeCompare(String(b.action)));
    return { playerId: pid, actions };
  });
});
const hasPolicyData = computed(() =>
  policyRowsByPlayer.value.some((row) => Array.isArray(row.actions) && row.actions.length > 0)
);

function isSelectedPolicyAction(playerId, actionName) {
  const selected =
    selectedActions.value?.[playerId]
    ?? selectedActions.value?.[String(playerId)]
    ?? null;
  return selected === actionName;
}

const playbookSelectedIntentIndices = computed(() => {
  const maxIntent = Math.max(0, Number(props.gameState?.num_intents || 1) - 1);
  const raw = String(playbookIntentInput.value || '')
    .split(',')
    .map((part) => Number(part.trim()))
    .filter((val) => Number.isFinite(val));
  const out = [];
  for (const val of raw) {
    const idx = Math.max(0, Math.min(maxIntent, Math.round(val)));
    if (!out.includes(idx)) out.push(idx);
  }
  return out;
});

// Shot probability display is handled on the board

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

function hasAnyProbabilities(probsByPlayer) {
  if (!probsByPlayer) return false;
  const values = Object.values(probsByPlayer);
  for (const probs of values) {
    if (!Array.isArray(probs)) continue;
    for (const raw of probs) {
      const p = Number(raw);
      if (Number.isFinite(p) && p > 0) return true;
    }
  }
  return false;
}

// Fetch action values for all players (needed for AI mode and display)
async function fetchAllActionValues() {
  if (!props.gameState || (props.gameState.done && !props.isManualStepping)) {
    return;
  }
  
  const allValues = {};
  const allIds = allPlayerIds.value;
  
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
  let consumedStoredValues = false;
  if (newGameState && newGameState.action_values) {
    applyStoredActionValues(newGameState.action_values);
    consumedStoredValues = true;
  }
  const snapshotProbs = newGameState?.policy_probabilities;
  if (hasAnyProbabilities(snapshotProbs)) {
    policyProbabilities.value = snapshotProbs;
  } else {
    policyProbabilities.value = null;
  }

  const allowApiFetch = !props.isManualStepping && !props.isReplaying;
  const shouldFetchAIData = newGameState && (!newGameState.done || props.isManualStepping);
  const shouldFetchActionValues = shouldFetchAIData && allowApiFetch && !consumedStoredValues;
  if (shouldFetchActionValues) {
    try {
      await fetchAllActionValues();
    } catch (error) {
      console.error('[PlayerControls] Error during AI data fetch:', error);
    }
  } else if (!newGameState) {
    actionValues.value = null;
    if (!props.isManualStepping) {
      policyProbabilities.value = null;
    }
    valueRange.value = { min: 0, max: 0 };
  }
}, { immediate: true });

watch(
  () => [props.gameState?.num_intents, props.gameState?.intent_commitment_steps],
  ([numIntents, commitment]) => {
    if (!playbookIntentInput.value) {
      const count = Math.max(1, Math.min(4, Number(numIntents || 1)));
      playbookIntentInput.value = Array.from({ length: count }, (_, idx) => idx).join(', ');
    }
    if (!playbookMaxSteps.value || playbookMaxSteps.value <= 0) {
      playbookMaxSteps.value = Math.max(1, Number(commitment || 8));
    }
  },
  { immediate: true },
);


// Watch for the list of players to be populated, then set the first one as active.
// The `immediate` flag ensures this runs on component creation.
watch(allPlayerIds, (newPlayerIds) => {
    if (newPlayerIds && newPlayerIds.length > 0 && props.activePlayerId === null) {
        // Prefer first user-controlled player, otherwise just the first player
        const firstUser = userControlledPlayerIds.value.length > 0 ? userControlledPlayerIds.value[0] : newPlayerIds[0];
        emit('update:activePlayerId', firstUser);
    }
}, { immediate: true });

watch(
  [controlsTabPlayerIds, () => props.activePlayerId],
  ([visibleIds, activePlayerId]) => {
    if (!Array.isArray(visibleIds) || visibleIds.length === 0) return;
    if (activePlayerId === null || activePlayerId === undefined || !visibleIds.includes(activePlayerId)) {
      emit('update:activePlayerId', visibleIds[0]);
    }
  },
  { immediate: true },
);

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
    return [];
  }
  const mask = props.gameState.action_mask[playerId];
  const legalActions = [];
  for (let i = 0; i < mask.length; i++) {
    if (mask[i] === 1 && i < actionNames.length) {
      legalActions.push(actionNames[i]);
    }
  }

  // Hard invariant for UI correctness: only current ball holder can shoot/pass.
  if (!isBallHolderPlayer(playerId)) {
    return legalActions.filter((action) => action !== 'SHOOT' && !action.startsWith('PASS_'));
  }
  
  return legalActions;
}

function sanitizeSelectionsToCurrentLegality() {
  if (!props.gameState?.action_mask) return;
  const nextSelections = { ...selectedActions.value };
  const nextTargets = { ...selectedPassTargets.value };
  let changed = false;

  for (const [rawPid, rawAction] of Object.entries(nextSelections)) {
    const pid = Number(rawPid);
    const action = typeof rawAction === 'string' ? rawAction : '';
    if (!Number.isFinite(pid) || !action) {
      delete nextSelections[rawPid];
      delete nextTargets[rawPid];
      changed = true;
      continue;
    }

    const legal = getLegalActions(pid);
    const legalPassExists = legal.some((a) => a.startsWith('PASS_'));
    const isPassAction = action.startsWith('PASS_') || action.startsWith('PASS->');

    if (isPassAction) {
      if (isPointerPassMode.value && !isBallHolderPlayer(pid)) {
        delete nextSelections[rawPid];
        delete nextTargets[rawPid];
        changed = true;
        continue;
      }
      if (!legalPassExists) {
        delete nextSelections[rawPid];
        delete nextTargets[rawPid];
        changed = true;
        continue;
      }
      if (isPointerPassMode.value) {
        const resolvedTarget = resolvePointerPassTarget(pid, action);
        if (!Number.isFinite(Number(resolvedTarget))) {
          delete nextSelections[rawPid];
          delete nextTargets[rawPid];
          changed = true;
          continue;
        }
        if (Number(nextTargets[rawPid]) !== Number(resolvedTarget)) {
          nextTargets[rawPid] = Number(resolvedTarget);
          changed = true;
        }
      } else if (!legal.includes(action)) {
        delete nextSelections[rawPid];
        delete nextTargets[rawPid];
        changed = true;
      }
      continue;
    }

    if (!legal.includes(action)) {
      delete nextSelections[rawPid];
      delete nextTargets[rawPid];
      changed = true;
    }
  }

  for (const rawPid of Object.keys(nextTargets)) {
    const action = String(nextSelections[rawPid] || '');
    if (!action.startsWith('PASS')) {
      delete nextTargets[rawPid];
      changed = true;
    }
  }

  if (changed) {
    selectedActions.value = nextSelections;
    selectedPassTargets.value = nextTargets;
  }
}

function isSelectionLegalForPlayer(playerId, actionName) {
  const pid = Number(playerId);
  const action = typeof actionName === 'string' ? actionName : '';
  if (!Number.isFinite(pid) || !action) return false;
  if (action === 'SHOOT' && !isBallHolderPlayer(pid)) {
    return false;
  }

  const legal = getLegalActions(pid);
  if (!legal.length) return false;

  const isPassAction = action.startsWith('PASS_') || action.startsWith('PASS->');
  if (!isPassAction) {
    return legal.includes(action);
  }

  if (!isPointerPassMode.value) {
    return legal.includes(action);
  }

  if (!isBallHolderPlayer(pid)) {
    return false;
  }

  const hasLegalPass = legal.some((a) => a.startsWith('PASS_'));
  if (!hasLegalPass) return false;
  const resolvedTarget = resolvePointerPassTarget(pid, action);
  return Number.isFinite(Number(resolvedTarget));
}

function getEffectiveSelectedAction(playerId) {
  const selection = selectedActions.value?.[playerId];
  if (!selection) return '';
  return isSelectionLegalForPlayer(playerId, selection) ? selection : '';
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
      delete selectedPassTargets.value[props.activePlayerId];
    } else {
      selectedActions.value[props.activePlayerId] = action;
      if (!action.startsWith('PASS_') || !isPointerPassMode.value) {
        delete selectedPassTargets.value[props.activePlayerId];
      }
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

const activePassTargets = computed(() => {
  const active = props.activePlayerId;
  if (active === null || active === undefined) return [];
  return getPointerPassTeammatesForPlayer(active);
});

const activeCanSelectPointerPass = computed(() => {
  if (!isPointerPassMode.value) return false;
  if (props.activePlayerId === null || props.activePlayerId === undefined) return false;
  if (!isBallHolderPlayer(props.activePlayerId)) return false;
  const legal = getLegalActions(props.activePlayerId);
  return legal.some(action => action.startsWith('PASS_'));
});

function handlePointerPassTargetSelected(targetId) {
  if (props.disabled || !isPointerPassMode.value) return;
  if (props.activePlayerId === null || props.activePlayerId === undefined) return;
  const pid = Number(props.activePlayerId);
  const tid = Number(targetId);
  if (!Number.isFinite(tid)) return;
  if (!activeCanSelectPointerPass.value) return;

  const alreadySelected = (
    selectedActions.value[pid]
    && selectedActions.value[pid].startsWith('PASS_')
    && Number(selectedPassTargets.value[pid]) === tid
  );
  if (alreadySelected) {
    delete selectedActions.value[pid];
    delete selectedPassTargets.value[pid];
    return;
  }

  selectedActions.value[pid] = 'PASS_E';
  selectedPassTargets.value[pid] = tid;

  const currentIndex = userControlledPlayerIds.value.indexOf(pid);
  const nextIndex = (currentIndex + 1) % userControlledPlayerIds.value.length;
  emit('update:activePlayerId', userControlledPlayerIds.value[nextIndex]);
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
    const actionName = getEffectiveSelectedAction(playerId) || 'NOOP';
    const pointerTarget = resolvePointerPassTarget(playerId, actionName);
    const hasPointerTarget = Number.isFinite(Number(pointerTarget));
    if (isPointerPassMode.value && actionName.startsWith('PASS_') && hasPointerTarget) {
      actionsToSubmit[playerId] = {
        type: 'PASS',
        target: Number(pointerTarget),
      };
    } else {
      const actionIndex = actionNames.indexOf(actionName);
      actionsToSubmit[playerId] = actionIndex !== -1 ? actionIndex : 0;
    }
  }
  
  // Track moves for the selected team
  const currentTurn = props.moveHistory.filter(m => !m?.isNoteRow).length + 1;
  const teamMoves = {};
  
  for (const playerId of playersToSubmit) {
    const actionName = getEffectiveSelectedAction(playerId) || 'NOOP';
    const pointerTarget = resolvePointerPassTarget(playerId, actionName);
    const hasPointerTarget = Number.isFinite(Number(pointerTarget));
    if (isPointerPassMode.value && actionName.startsWith('PASS_') && hasPointerTarget) {
      teamMoves[`Player ${playerId}`] = `PASS->${Number(pointerTarget)}`;
    } else {
      teamMoves[`Player ${playerId}`] = actionName;
    }
  }
  
  emit('move-recorded', {
    turn: currentTurn,
    moves: teamMoves,
    mctsPlayers: Array.from(mctsTargets),
  });
  
  emit('actions-submitted', actionsToSubmit);

  // Emit current MCTS options (or null) so parent can include in step
  emit('mcts-options-changed', buildMctsOptions());
  
  // Clear local selections after submit; next state/policy snapshot repopulates as needed.
  selectedActions.value = {};
  selectedPassTargets.value = {};
  // Reset to first available player (prefer user team)
  if (userControlledPlayerIds.value.length > 0) {
    emit('update:activePlayerId', userControlledPlayerIds.value[0]);
  } else if (allPlayerIds.value.length > 0) {
    emit('update:activePlayerId', allPlayerIds.value[0]);
  }
}

function getSelectedActions() {
  // Return current selections for parent to use
  return { ...selectedActions.value };
}

function getControlPadLegalActions(playerId) {
  const legal = getLegalActions(playerId);
  if (!isPointerPassMode.value) return legal;
  return legal.filter(action => !action.startsWith('PASS_'));
}

function getSelectedActionDisplay(playerId) {
  const selection = getEffectiveSelectedAction(playerId);
  if (!selection || selection === 'NOOP') return '';
  const isPassAction = selection.startsWith('PASS_') || selection.startsWith('PASS->');
  if (!isPassAction) {
    return selection;
  }
  const target = resolvePointerPassTarget(playerId, selection);
  if (Number.isFinite(Number(target))) {
    return `PASS->${Number(target)}`;
  }
  return selection;
}

function getSelectedActionBadge(playerId) {
  const label = getSelectedActionDisplay(playerId);
  if (!label) return '';
  if (label.startsWith('MOVE_')) return 'M';
  if (label.startsWith('PASS->')) return `P${label.replace('PASS->', '')}`;
  if (label.startsWith('PASS_')) return 'P';
  return label;
}

function shouldDisplaySelectedAction(playerId) {
  if (playerId === null || playerId === undefined) return false;
  if (props.showOpponentActions) return true;
  return userControlledPlayerIds.value.includes(Number(playerId));
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
  emit('selections-changed', buildDisplaySelections(selectedActions.value));
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
      selectedPassTargets.value = {};
    } else {
      // Clear selections when AI mode is disabled
      selectedActions.value = {};
      selectedPassTargets.value = {};
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
      selectedPassTargets.value = {};
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
  } catch (error) {
    console.error('Failed to fetch rewards:', error);
  }
};

// Watch for game state changes to update rewards and clear moves
watch(() => props.gameState, (newState, oldState) => {
  if (newState) {
    fetchRewards();
    
    // Move history clearing is now handled by parent component
  }
}, { deep: true });

// Watch for when user switches to Rewards tab
watch(() => activeTab.value, (newTab) => {
  if (newTab === 'rewards') {
    fetchRewards();
  }
});

onMounted(() => {
  isMounted.value = true;
  fetchRewards();
  nextTick(() => {
    refreshTabsTeleportTarget();
    setTimeout(() => refreshTabsTeleportTarget(), 0);
  });
  emit('active-tab-changed', activeTab.value);
});

onBeforeUnmount(() => {
  stopPlaybookProgressPolling();
});

watch(resolvedTabsMount, () => {
  nextTick(() => refreshTabsTeleportTarget());
});

watch(tabsMountTargetEl, () => {
  nextTick(() => refreshTabsTeleportTarget());
});

// Record stats once on episode completion (but skip during evaluation mode)
watch(() => props.gameState?.done, async (done, prevDone) => {
  if (done && !prevDone && props.gameState && !props.isReplaying && !props.isEvaluating) {
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
    selectedPassTargets.value = {};
    sanitizeSelectionsToCurrentLegality();
  }
});

watch(
  [
    () => props.gameState?.action_mask,
    () => props.gameState?.ball_holder,
    () => props.gameState?.offense_ids,
    () => props.gameState?.defense_ids,
    isPointerPassMode,
  ],
  () => {
    sanitizeSelectionsToCurrentLegality();
  },
  { deep: true }
);

watch(isPointerPassMode, (enabled) => {
  if (!enabled) {
    selectedPassTargets.value = {};
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
  const meta = obsMeta.value;
  const obs = props.gameState?.obs;
  if (!meta || !obs) return 0;
  return Number(obs[meta.shotClockIdx] || 0);
});

const pressureExposureObsValue = computed(() => {
  const meta = obsMeta.value;
  const obs = props.gameState?.obs;
  if (!meta || !obs || meta.pressureExposureIdx === null) return 0;
  return Number(obs[meta.pressureExposureIdx] || 0);
});

const teamEncodingRows = computed(() => {
  const meta = obsMeta.value;
  const obs = props.gameState?.obs;
  if (!meta || !obs) return [];
  return obs.slice(meta.teamEncodingStart, meta.teamEncodingStart + meta.nPlayers);
});

const ballHandlerPositionRows = computed(() => {
  const meta = obsMeta.value;
  const obs = props.gameState?.obs;
  if (!meta || !obs) return [];
  return obs.slice(meta.ballHandlerStart, meta.ballHandlerStart + 2);
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
  // Pressure exposure was inserted after shot clock in the flat observation schema.
  const pressureExposureIdx = offset;
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
    nPlayers,
    shotClockIdx,
    pressureExposureIdx,
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
const tokenGlobalBaseLabels = ['shot_clock', 'pressure_exposure', 'hoop_q_norm', 'hoop_r_norm'];
const tokenGlobalIntentLabels = ['intent_index_norm', 'intent_active', 'intent_visible', 'intent_age_norm'];

const tokenPlayers = computed(() => {
  const players = obsTokens.value?.players;
  return Array.isArray(players) ? players : [];
});

const tokenGlobals = computed(() => {
  const globals = obsTokens.value?.globals;
  return Array.isArray(globals) ? globals : [];
});

const tokenGlobalLabels = computed(() => {
  const provided = obsTokens.value?.globals_labels;
  if (Array.isArray(provided) && provided.length > 0) {
    return provided;
  }
  const labels = [...tokenGlobalBaseLabels];
  const extras = Math.max(0, tokenGlobals.value.length - labels.length);
  if (extras > 0) {
    const intentTake = Math.min(tokenGlobalIntentLabels.length, extras);
    labels.push(...tokenGlobalIntentLabels.slice(0, intentTake));
    for (let idx = intentTake; idx < extras; idx += 1) {
      labels.push(`global_${tokenGlobalBaseLabels.length + idx}`);
    }
  }
  return labels;
});

const tokenAttention = computed(() => obsTokens.value?.attention || null);
const tokenAttentionLabels = computed(() => tokenAttention.value?.labels || []);
const tokenAttentionAvgWeights = computed(() => tokenAttention.value?.weights_avg || []);
const tokenAttentionHeadWeights = computed(() => tokenAttention.value?.weights_heads || []);
const tokenAttentionHeads = computed(() => tokenAttention.value?.heads ?? null);
const tokenAttentionRuntimeIntentIndex = computed(() => {
  const value = tokenAttention.value?.runtime_intent_index;
  return Number.isFinite(Number(value)) ? Number(value) : null;
});
const tokenAttentionRuntimeIntentGate = computed(() => Boolean(tokenAttention.value?.runtime_intent_gate));
const tokenAttentionRuntimeIntentActive = computed(() => Boolean(tokenAttention.value?.runtime_intent_active));
const tokenAttentionRuntimeIntentVisible = computed(() => Boolean(tokenAttention.value?.runtime_intent_visible));
const tokenAttentionObserverRole = computed(() => String(tokenAttention.value?.observer_role || ''));
const tokenAttentionRuntimeSummary = computed(() => {
  if (!tokenAttention.value) return '';
  const parts = [];
  if (tokenAttentionObserverRole.value) {
    parts.push(`Observer: ${tokenAttentionObserverRole.value}`);
  }
  if (tokenAttentionRuntimeIntentIndex.value !== null) {
    parts.push(
      `Intent: ${formatPlayLabel(tokenAttentionRuntimeIntentIndex.value, props.gameState?.play_name_map)}`
    );
  }
  parts.push(`Gate: ${tokenAttentionRuntimeIntentGate.value ? 'on' : 'off'}`);
  parts.push(`Active: ${tokenAttentionRuntimeIntentActive.value ? 'yes' : 'no'}`);
  parts.push(`Visible: ${tokenAttentionRuntimeIntentVisible.value ? 'yes' : 'no'}`);
  return parts.join(' · ');
});
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
    label: tokenGlobalLabels.value[idx] || `global_${idx}`,
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
    <div class="turn-controls-panel">
      <h3 class="turn-controls-title">Turn Controls</h3>
      <div class="ball-holder-row">
        <label>Ball handler</label>
        <select
          :value="ballHolderSelection ?? ''"
          @change="handleBallHolderChange($event.target.value)"
          :disabled="ballHolderUpdating || props.isEvaluating || props.isReplaying || props.isManualStepping || offenseIdsLive.length === 0"
        >
          <option v-if="offenseIdsLive.length === 0" disabled value="">No offense players</option>
          <option v-for="pid in offenseIdsLive" :key="`bh-${pid}`" :value="pid">Player {{ pid }}</option>
        </select>
        <span v-if="ballHolderUpdating" class="status-note">Updating…</span>
        <span v-if="ballHolderError" class="error-note">{{ ballHolderError }}</span>
      </div>

      <div class="player-tabs">
          <button
              v-for="playerId in controlsTabPlayerIds"
              :key="playerId"
              class="player-tab"
              :class="{ active: activePlayerId === playerId }"
              @click="$emit('update:activePlayerId', playerId)"
              :disabled="false"
          >
              Player {{ playerId }}
              <span v-if="shouldDisplaySelectedAction(playerId) && getSelectedActionDisplay(playerId)">
                ({{ getSelectedActionBadge(playerId) }})
              </span>
          </button>
      </div>

      <div class="control-pad-wrapper" v-if="activePlayerId !== null && controlsTabPlayerIds.includes(activePlayerId)">
          <HexagonControlPad
              :legal-actions="getControlPadLegalActions(activePlayerId)"
              :selected-action="shouldDisplaySelectedAction(activePlayerId) ? getEffectiveSelectedAction(activePlayerId) : ''"
              :pass-probabilities="passProbabilities"
              @action-selected="handleActionSelected"
              :action-values="actionValues && actionValues[activePlayerId] ? actionValues[activePlayerId] : null"
              :value-range="valueRange"
              :is-defense="isDefense"
              layout-variant="court"
          />
          <div v-if="isPointerPassMode" class="pointer-pass-controls pointer-pass-wrap" :class="{ 'has-pass-selection': activeHasPassSelection }">
            <p class="pointer-pass-label">Pass Target</p>
            <p v-if="props.gameState && !isBallHolderPlayer(activePlayerId)" class="pointer-pass-note">
              Select the ball handler to choose a teammate target.
            </p>
            <p v-else-if="!activeCanSelectPointerPass" class="pointer-pass-note">
              Passing is not legal from this state.
            </p>
            <div v-else class="pointer-pass-buttons">
              <button
                v-for="targetId in activePassTargets"
                :key="`pass-target-${activePlayerId}-${targetId}`"
                class="pointer-pass-button pointer-pass-btn"
                :class="{ selected: isPointerPassButtonSelected(targetId) }"
                :disabled="props.disabled"
                @click="handlePointerPassTargetSelected(targetId)"
              >
                Player {{ targetId }}
              </button>
            </div>
          </div>
          <p v-if="shouldDisplaySelectedAction(activePlayerId) && getSelectedActionDisplay(activePlayerId)">
              Selected for Player {{ activePlayerId }}: <strong>{{ getSelectedActionDisplay(activePlayerId) }}</strong>
          </p>
      </div>
    </div>
    
    <Teleport :to="resolvedTabsTeleportTarget" :disabled="!useExternalTabsTeleport">
      <div class="tabs-content-shell">
    <!-- Tab Navigation -->
    <div class="tab-navigation">
      <button
        v-for="tab in orderedDevTabs"
        :key="tab.id"
        :class="{ active: activeTab === tab.id, dragging: draggedDevTabId === tab.id }"
        draggable="true"
        @click="activeTab = tab.id"
        @dragstart="handleDevTabDragStart(tab.id, $event)"
        @dragover.prevent="handleDevTabDragOver"
        @drop="handleDevTabDrop(tab.id)"
        @dragend="handleDevTabDragEnd"
      >
        {{ tab.label }}
      </button>
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
      <div v-if="perIntentEvalStatsTable.length" class="per-player-stats">
        <h4>Per-Intent Offense Stats (Eval)</h4>
        <div class="status-note">
          The board shot-chart dropdown above the court now includes these intent labels.
        </div>
        <table class="per-player-table">
          <thead>
            <tr>
              <th>Play</th>
              <th>Episodes</th>
              <th>FG</th>
              <th>Dunk</th>
              <th>2PT</th>
              <th>3PT</th>
              <th>Assists</th>
              <th>Pot. Ast</th>
              <th>TOV</th>
              <th>Points</th>
              <th>PPP</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="row in perIntentEvalStatsTable" :key="`intent-stat-${row.intentKey}`">
              <td>{{ row.label }}</td>
              <td>{{ row.episodes }}</td>
              <td>{{ row.makes }}/{{ row.attempts }} ({{ row.fg.toFixed(1) }}%)</td>
              <td>{{ row.dunk.mk }}/{{ row.dunk.att }} ({{ row.unassisted.dunk }})</td>
              <td>{{ row.two.mk }}/{{ row.two.att }} ({{ row.unassisted.two }})</td>
              <td>{{ row.three.mk }}/{{ row.three.att }} ({{ row.unassisted.three }})</td>
              <td>{{ row.assists }}</td>
              <td>{{ row.potentialAssists }}</td>
              <td>{{ row.turnovers }}</td>
              <td>{{ row.points.toFixed(1) }}</td>
              <td>{{ row.ppp.toFixed(2) }}</td>
            </tr>
          </tbody>
        </table>
      </div>
      <div class="rewards-section">
        <h4>Episode Stats</h4>
        <div class="parameters-grid">
          <div class="param-category">
            <h5>
              Totals
              <span
                class="category-help"
                title="Aggregate metrics across all evaluated episodes for the user-controlled team."
                aria-label="Totals help"
                tabindex="0"
              >?</span>
            </h5>
            <div class="param-item" data-tooltip="Number of evaluated episodes included in these stats."><span class="param-name">Episodes played:</span><span class="param-value">{{ statsState.episodes }}</span></div>
            <div class="param-item" data-tooltip="Total credited assists on made shots by the user team."><span class="param-name">Total assists:</span><span class="param-value">{{ totalAssists }}</span></div>
            <div class="param-item" data-tooltip="Missed shots that still qualified as potential assists."><span class="param-name">Total potential assists (missed):</span><span class="param-value">{{ totalPotentialAssists }}</span></div>
            <div class="param-item" data-tooltip="Total turnovers committed by the user team."><span class="param-name">Total turnovers:</span><span class="param-value">{{ statsState.turnovers }}</span></div>
            <div class="param-item" data-tooltip="Total lane-rule violations (illegal defense + offensive 3-second)."><span class="param-name">Total violations:</span><span class="param-value">{{ totalViolations }}</span></div>
            <div class="param-item" data-tooltip="Defenders stayed in the lane too long without guarding; counts technical-style lane violations."><span class="param-name">Illegal defense violations:</span><span class="param-value">{{ statsState.violations?.defensiveLane || 0 }}</span></div>
            <div class="param-item" data-tooltip="Offense kept a player in the lane beyond the 3-second limit, causing turnovers."><span class="param-name">Offensive 3-second violations:</span><span class="param-value">{{ statsState.violations?.offensiveThreeSeconds || 0 }}</span></div>
            <div class="param-item" data-tooltip="Points per possession proxy here: total points scored by user team divided by episodes."><span class="param-name">PPP:</span><span class="param-value">{{ ppp.toFixed(2) }}</span></div>
            <div class="param-item" data-tooltip="Average total episode reward for the user team."><span class="param-name">Avg reward/ep:</span><span class="param-value">{{ avgRewardPerEp.toFixed(2) }}</span></div>
            <div class="param-item" data-tooltip="Average number of steps per episode."><span class="param-name">Avg ep length (steps):</span><span class="param-value">{{ avgEpisodeLen.toFixed(1) }}</span></div>
          </div>
          <div class="param-category">
            <h5>
              Turnovers by Reason
              <span
                class="category-help"
                title="Counts user-team turnovers grouped by turnover reason from action resolution."
                aria-label="Turnovers by reason help"
                tabindex="0"
              >?</span>
            </h5>
            <div
              v-for="row in turnoverReasonRows"
              :key="`turnover-reason-${row.reason}`"
              class="param-item"
              :data-tooltip="getTurnoverReasonTooltip(row.reason)"
            >
              <span class="param-name">{{ row.label }}:</span>
              <span class="param-value">{{ row.count }}</span>
            </div>
            <div v-if="turnoverReasonRows.length === 0" class="param-item" data-tooltip="No turnovers recorded for the user team in this evaluation window.">
              <span class="param-name">(none)</span>
              <span class="param-value">0</span>
            </div>
          </div>
          <div class="param-category">
            <h5>
              Action Mix
              <span
                class="category-help"
                title="Distribution of selected actions for the user team across all evaluated timesteps."
                aria-label="Action mix help"
                tabindex="0"
              >?</span>
            </h5>
            <div class="param-item" data-tooltip="Total number of user-team action selections counted in this evaluation."><span class="param-name">Total decisions:</span><span class="param-value">{{ actionMixRows.total }}</span></div>
            <div
              v-for="row in actionMixRows.rows"
              :key="`action-mix-${row.key}`"
              class="param-item"
              :data-tooltip="getActionMixTooltip(row.key)"
            >
              <span class="param-name">{{ row.label }}:</span>
              <span class="param-value">{{ row.count }} ({{ row.rate.toFixed(1) }}%)</span>
            </div>
          </div>
          <div class="param-category">
            <h5>
              Offense Intent Starts
              <span
                class="category-help"
                title="Count of offense intent index at episode start during evaluation. This reflects the active offense intent sampled/applied when each eval episode began."
                aria-label="Offense intent starts help"
                tabindex="0"
              >?</span>
            </h5>
            <div class="param-item" data-tooltip="Total number of evaluated episode starts counted for offense intent.">
              <span class="param-name">Total starts:</span>
              <span class="param-value">{{ intentSelectionRows.total }}</span>
            </div>
            <div
              v-for="row in intentSelectionRows.rows"
              :key="`intent-start-${row.intent}`"
              class="param-item"
              data-tooltip="Number of evaluation episodes that started with this offense intent index active."
            >
              <span class="param-name">{{ row.label }}:</span>
              <span class="param-value">{{ row.count }}</span>
            </div>
            <div
              v-if="intentSelectionRows.inactiveCount > 0"
              class="param-item"
              data-tooltip="Episodes where offense intent learning was enabled but no active intent was present at episode start."
            >
              <span class="param-name">No intent:</span>
              <span class="param-value">{{ intentSelectionRows.inactiveCount }}</span>
            </div>
            <div v-if="intentSelectionRows.total === 0" class="param-item" data-tooltip="No offense intent start counts were recorded in this evaluation window.">
              <span class="param-name">(none)</span>
              <span class="param-value">0</span>
            </div>
          </div>
          <div class="param-category">
            <h5>
              Reward Decomposition
              <span
                class="category-help"
                title="Breakdown of total user-team reward into environment reward components (expected points, pass, assist, violations, phi shaping)."
                aria-label="Reward decomposition help"
                tabindex="0"
              >?</span>
            </h5>
            <div
              v-for="row in rewardBreakdownRows"
              :key="`reward-breakdown-${row.key}`"
              class="param-item"
              :data-tooltip="getRewardBreakdownTooltip(row.key)"
            >
              <span class="param-name">{{ row.label }}:</span>
              <span class="param-value">{{ row.value.toFixed(2) }}</span>
            </div>
          </div>
          <div class="param-category">
            <h5>
              Dunks
              <span
                class="category-help"
                title="Dunk-only shot statistics for the user team."
                aria-label="Dunks help"
                tabindex="0"
              >?</span>
            </h5>
            <div class="param-item" data-tooltip="Total dunk attempts by the user team."><span class="param-name">Attempts:</span><span class="param-value">{{ statsState.dunk.attempts }}</span></div>
            <div class="param-item" data-tooltip="Made dunks by the user team."><span class="param-name">Made:</span><span class="param-value">{{ statsState.dunk.made }}</span></div>
            <div class="param-item" data-tooltip="Dunk field-goal percentage: made / attempts."><span class="param-name">FG%:</span><span class="param-value">{{ (safeDiv(statsState.dunk.made, Math.max(1, statsState.dunk.attempts)) * 100).toFixed(1) }}%</span></div>
            <div class="param-item" data-tooltip="Made dunks that were credited with a full assist."><span class="param-name">Assists:</span><span class="param-value">{{ statsState.dunk.assists }}</span></div>
            <div class="param-item" data-tooltip="Missed dunks that still qualified as potential assists."><span class="param-name">Potential assists (missed):</span><span class="param-value">{{ statsState.dunk.potentialAssists }}</span></div>
          </div>
          <div class="param-category">
            <h5>
              2PT
              <span
                class="category-help"
                title="Two-point, non-dunk shot statistics for the user team."
                aria-label="2PT help"
                tabindex="0"
              >?</span>
            </h5>
            <div class="param-item" data-tooltip="Total two-point (non-dunk) attempts by the user team."><span class="param-name">Attempts:</span><span class="param-value">{{ statsState.twoPt.attempts }}</span></div>
            <div class="param-item" data-tooltip="Made two-point (non-dunk) shots by the user team."><span class="param-name">Made:</span><span class="param-value">{{ statsState.twoPt.made }}</span></div>
            <div class="param-item" data-tooltip="2PT field-goal percentage: made / attempts."><span class="param-name">FG%:</span><span class="param-value">{{ (safeDiv(statsState.twoPt.made, Math.max(1, statsState.twoPt.attempts)) * 100).toFixed(1) }}%</span></div>
            <div class="param-item" data-tooltip="Made 2PT shots that were credited with a full assist."><span class="param-name">Assists:</span><span class="param-value">{{ statsState.twoPt.assists }}</span></div>
            <div class="param-item" data-tooltip="Missed 2PT shots that still qualified as potential assists."><span class="param-name">Potential assists (missed):</span><span class="param-value">{{ statsState.twoPt.potentialAssists }}</span></div>
          </div>
          <div class="param-category">
            <h5>
              3PT
              <span
                class="category-help"
                title="Three-point shot statistics for the user team."
                aria-label="3PT help"
                tabindex="0"
              >?</span>
            </h5>
            <div class="param-item" data-tooltip="Total three-point attempts by the user team."><span class="param-name">Attempts:</span><span class="param-value">{{ statsState.threePt.attempts }}</span></div>
            <div class="param-item" data-tooltip="Made three-point shots by the user team."><span class="param-name">Made:</span><span class="param-value">{{ statsState.threePt.made }}</span></div>
            <div class="param-item" data-tooltip="3PT field-goal percentage: made / attempts."><span class="param-name">FG%:</span><span class="param-value">{{ (safeDiv(statsState.threePt.made, Math.max(1, statsState.threePt.attempts)) * 100).toFixed(1) }}%</span></div>
            <div class="param-item" data-tooltip="Made 3PT shots that were credited with a full assist."><span class="param-name">Assists:</span><span class="param-value">{{ statsState.threePt.assists }}</span></div>
            <div class="param-item" data-tooltip="Missed 3PT shots that still qualified as potential assists."><span class="param-name">Potential assists (missed):</span><span class="param-value">{{ statsState.threePt.potentialAssists }}</span></div>
          </div>
        </div>
        <div style="display:flex; gap: 0.5rem;">
          <button type="button" class="new-game-button" @click="resetStats">Reset Stats</button>
          <button type="button" class="submit-button" @click="copyStatsMarkdown">Copy</button>
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

    <!-- Policy Tab -->
    <div v-if="activeTab === 'policy'" class="tab-content">
      <div class="entropy-section">
        <h4>Policy Probabilities ({{ policyTeamLabel }})</h4>
        <p class="entropy-note">Legal-action probabilities for the current state, filtered to the selected team.</p>

        <div v-if="hasSelectorIntentPreferences" class="param-category selector-intent-card">
          <h5>Intent Preferences</h5>
          <div class="eval-row selector-plot-row">
            <label>Plot</label>
            <select v-model="selectedSelectorIntentPlotMode">
              <option
                v-for="opt in selectorIntentPlotModeOptions"
                :key="`selector-plot-${opt.value}`"
                :value="opt.value"
              >
                {{ opt.label }}
              </option>
            </select>
            <span class="status-note">Bar widths and entropy summary use {{ selectorIntentPlotLabel.toLowerCase() }} probabilities.</span>
          </div>
          <p class="entropy-note">
            Selector preferences for the current offense state in fixed intent order.
            <span v-if="selectorIntentPreferences.selectionMode">
              Mode {{ String(selectorIntentPreferences.selectionMode) }}
            </span>
            <span v-if="selectorIntentPreferences.alphaCurrent !== null">
              Alpha {{ Number(selectorIntentPreferences.alphaCurrent).toFixed(3) }}
            </span>
            <span v-if="selectorIntentPreferences.epsCurrent !== null">
              · Eps {{ Number(selectorIntentPreferences.epsCurrent).toFixed(3) }}
            </span>
            <span v-if="selectorIntentPreferences.valueEstimate !== null">
              · Value {{ Number(selectorIntentPreferences.valueEstimate).toFixed(3) }}
            </span>
            <span v-if="selectorIntentEntropy.entropy !== null">
              · {{ selectorIntentPlotLabel }} Entropy {{ Number(selectorIntentEntropy.entropy).toFixed(3) }}
            </span>
            <span v-if="selectorIntentEntropy.normalizedEntropy !== null">
              (norm {{ Number(selectorIntentEntropy.normalizedEntropy).toFixed(3) }})
            </span>
            <span v-if="selectorIntentEntropy.klToUniform !== null">
              · {{ selectorIntentPlotLabel }} KL-U {{ Number(selectorIntentEntropy.klToUniform).toFixed(3) }}
            </span>
            <span v-if="selectorIntentEntropy.normalizedKlToUniform !== null">
              (norm {{ Number(selectorIntentEntropy.normalizedKlToUniform).toFixed(3) }})
            </span>
          </p>
          <p class="entropy-note">
            Raw = selector softmax. Mixed = selector-branch probabilities after epsilon floor. Deployed = final runtime distribution after alpha mixes the selector branch with uniform fallback.
          </p>
          <div class="entropy-table-wrapper">
            <table class="entropy-table">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Play</th>
                  <th>Raw</th>
                  <th>Mixed</th>
                  <th>Deployed</th>
                  <th>Logit</th>
                </tr>
              </thead>
              <tbody>
                <tr
                  v-for="(item, idx) in selectorIntentPreferences.items"
                  :key="`selector-intent-${item.intent_index}`"
                  class="selector-intent-row"
                  :class="{ 'current-intent-row': Number(item.intent_index) === Number(selectorIntentPreferences.currentIntentIndex) }"
                  :style="{ '--selector-prob-width': `${Math.max(0, Math.min(100, getSelectorIntentPlotProb(item) * 100)).toFixed(2)}%` }"
                >
                  <td>{{ idx + 1 }}</td>
                  <td>{{ formatPlayLabel(item.intent_index, props.gameState?.play_name_map, item.play_name) }}</td>
                  <td>{{ (Number(item.raw_prob ?? item.prob ?? 0) * 100).toFixed(2) }}%</td>
                  <td>{{ (Number(item.mixed_prob ?? item.raw_prob ?? item.prob ?? 0) * 100).toFixed(2) }}%</td>
                  <td>{{ (Number(item.deployed_prob ?? item.prob ?? 0) * 100).toFixed(2) }}%</td>
                  <td>{{ Number(item.logit).toFixed(3) }}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        <div v-if="!policyProbabilities || !hasPolicyData" class="no-data">
          No policy probabilities available yet.
        </div>
        <div v-else class="parameters-grid">
          <div
            v-for="row in policyRowsByPlayer"
            :key="`policy-${row.playerId}`"
            class="param-category"
          >
            <h5>Player {{ row.playerId }}</h5>
            <div v-if="!row.actions.length" class="no-data">
              No legal actions available.
            </div>
            <div v-else class="entropy-table-wrapper">
              <table class="entropy-table">
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Action</th>
                    <th>Prob</th>
                  </tr>
                </thead>
                <tbody>
                  <tr
                    v-for="(action, idx) in row.actions"
                    :key="`${row.playerId}-${action.action}`"
                    :class="[
                      'probability-bar-row',
                      { 'current-intent-row': isSelectedPolicyAction(row.playerId, action.action) }
                    ]"
                    :style="{ '--selector-prob-width': `${Math.max(0, Math.min(100, Number(action.prob) * 100)).toFixed(2)}%` }"
                  >
                    <td>{{ idx + 1 }}</td>
                    <td>{{ action.action }}</td>
                    <td>{{ (action.prob * 100).toFixed(2) }}%</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Playbook Tab -->
    <div v-if="activeTab === 'playbook'" class="tab-content">
      <div class="entropy-section">
        <h4>Playbook</h4>
        <p class="entropy-note">
          Aggregate intent-conditioned rollout patterns from the current state or a captured snapshot. The main board renders trajectory overlays after generation.
        </p>

        <div class="playbook-controls">
          <div class="eval-row">
            <label>Intent indices</label>
            <input
              type="text"
              v-model="playbookIntentInput"
              placeholder="0, 1, 2, 3"
              :disabled="playbookControlsDisabled"
            />
            <span class="status-note">Parsed: {{ playbookSelectedIntentLabels.join(', ') || 'none' }}</span>
          </div>

          <div class="eval-row">
            <label>Rollouts</label>
            <input
              type="number"
              min="1"
              max="512"
              v-model.number="playbookNumRollouts"
              :disabled="playbookControlsDisabled"
            />
            <label>Horizon</label>
            <input
              type="number"
              min="1"
              max="256"
              v-model.number="playbookMaxSteps"
              :disabled="playbookControlsDisabled || playbookRunToEnd"
            />
            <label class="inline-label">
              <input
                type="checkbox"
                v-model="playbookRunToEnd"
                :disabled="playbookControlsDisabled"
              />
              Run to end
            </label>
            <button
              class="eval-run-btn"
              @click="handleRunPlaybookAnalysis"
              :disabled="playbookControlsDisabled"
            >
              {{ playbookRunning ? 'Generating…' : 'Generate' }}
            </button>
          </div>

          <div v-if="playbookRunning" class="eval-progress-wrap">
            <div
              class="eval-progress-bar"
              :aria-valuenow="playbookProgressSafe.completed"
              :aria-valuemin="0"
              :aria-valuemax="Math.max(1, playbookProgressSafe.total)"
              role="progressbar"
            >
              <div
                class="eval-progress-fill"
                :class="{ indeterminate: playbookProgressSafe.total <= 0 }"
                :style="playbookProgressSafe.total > 0 ? { width: playbookProgressPercent } : null"
              />
            </div>
            <span class="eval-status">
              {{ playbookProgressSafe.completed }}/{{ playbookProgressSafe.total || (playbookSelectedIntentIndices.length * Number(playbookNumRollouts || 0)) }}
            </span>
            <span class="eval-status" v-if="playbookProgressSafe.total > 0">({{ playbookProgressPercent }})</span>
          </div>

          <div class="eval-row">
            <label class="inline-label">
              <input
                type="radio"
                :checked="playbookUseSnapshot"
                @change="playbookUseSnapshot = true"
                :disabled="playbookControlsDisabled"
              />
              Snapshot source
            </label>
            <label class="inline-label">
              <input
                type="radio"
                :checked="!playbookUseSnapshot"
                @change="playbookUseSnapshot = false"
                :disabled="playbookControlsDisabled"
              />
              Current state source
            </label>
            <span class="status-note" v-if="playbookUseSnapshot">
              {{ counterfactualSnapshotAvailable ? 'Using captured snapshot as pinned start state.' : 'No snapshot available yet.' }}
            </span>
          </div>

          <div class="eval-row">
            <label class="inline-label">
              <input
                type="checkbox"
                v-model="playbookPlayerDeterministic"
                :disabled="playbookControlsDisabled"
              />
              Player deterministic
            </label>
            <label class="inline-label">
              <input
                type="checkbox"
                v-model="playbookOpponentDeterministic"
                :disabled="playbookControlsDisabled"
              />
              Opponent deterministic
            </label>
            <span class="status-note">
              {{ playbookRunToEnd
                ? 'Rollouts continue until the possession ends; Horizon is ignored.'
                : 'Fully deterministic rollouts from the same source state collapse to a single trajectory.' }}
            </span>
          </div>
        </div>

        <div class="policy-status error" v-if="playbookError">
          {{ playbookError }}
        </div>

        <div v-if="!playbookResult?.panels?.length" class="no-data">
          No playbook preview generated yet.
        </div>
        <div v-else class="status-note">
          Generated {{ playbookResult.panels.length }} intent trajectories using
          {{ playbookResult.run_to_end ? 'full-possession rollouts' : `a ${playbookResult.max_steps}-step horizon` }}.
          Use the Playbook dropdown above the main board to inspect them.
        </div>
        <div v-if="playbookResult?.panels?.length" class="playbook-summary-table-wrap">
          <table class="playbook-summary-table">
            <thead>
              <tr>
                <th>Play</th>
                <th>Rollouts</th>
                <th>Total Shots</th>
                <th>Shot Rollout Rate</th>
                <th>Avg First Shot Step</th>
                <th>Avg Terminated Steps</th>
                <th>Primary Shooter</th>
                <th
                  v-for="playerId in playbookOffenseColumnIds"
                  :key="`playbook-summary-head-p${playerId}`"
                >
                  P{{ playerId }} Shots
                </th>
                <th>Terminal Outcomes</th>
                <th>Turnover Reasons</th>
              </tr>
            </thead>
            <tbody>
              <tr
                v-for="panel in playbookResult.panels"
                :key="`playbook-debug-${panel.intent_index}`"
              >
                <td class="playbook-summary-play-cell">
                  {{ formatPlayLabel(panel.intent_index, playbookResult?.play_name_map || props.gameState?.play_name_map, panel.play_name) }}
                </td>
                <td>{{ Number(panel?.num_rollouts || 0) }}</td>
                <td>{{ formatPlaybookTotalShots(panel) }}</td>
                <td>{{ ((Number(panel?.shot_rollout_rate || 0)) * 100).toFixed(0) }}%</td>
                <td>{{ panel?.avg_first_shot_step == null ? 'none' : Number(panel.avg_first_shot_step).toFixed(2) }}</td>
                <td>{{ panel?.avg_terminated_steps == null ? 'n/a' : Number(panel.avg_terminated_steps).toFixed(2) }}</td>
                <td>{{ getPlaybookPrimaryShooterSummary(panel) }}</td>
                <td
                  v-for="playerId in playbookOffenseColumnIds"
                  :key="`playbook-summary-${panel.intent_index}-p${playerId}`"
                >
                  {{ formatPlaybookPlayerShotCell(panel, playerId) }}
                </td>
                <td>{{ formatPlaybookOutcomeSummary(panel) }}</td>
                <td>{{ formatPlaybookTurnoverSummary(panel) }}</td>
              </tr>
            </tbody>
          </table>
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

    <!-- Template Tab -->
    <div v-if="activeTab === 'template'" class="tab-content eval-tab template-tab">
      <div class="template-library-shell">
        <div class="template-library-toolbar">
          <input
            ref="templateLibraryFileInput"
            type="file"
            accept=".json,.yaml,.yml,application/json,text/yaml,text/x-yaml"
            class="template-file-input-hidden"
            @change="handleTemplateLibraryFileChosen"
          />
          <div class="eval-row template-library-path-row">
            <label>
              Save path
              <span
                class="template-help"
                data-tooltip="Optional repo/backend file path used by 'Load from path' and 'Save file'. Use 'Choose file' when you just want to import a local YAML or JSON file through the browser."
              >?</span>
            </label>
            <input
              v-model="templateLibraryPathInput"
              class="template-path-input"
              type="text"
              placeholder="configs/start_templates_v1.yaml"
              title="Repo/backend file path used for direct load/save."
            />
            <button
              class="ghost-btn"
              @click="triggerTemplateLibraryChooser"
              :disabled="templateLibraryRequestPending"
              title="Import a local YAML or JSON template file through the browser. This does not save back to that original file."
            >
              {{ templateLibraryRequestPending ? 'Loading…' : 'Import file' }}
            </button>
            <button
              class="ghost-btn"
              @click="handleLoadTemplateLibraryFile"
              :disabled="templateLibraryRequestPending || !templateLibraryPathInput"
              title="Load the library from the repo/backend file path shown in Save path."
            >
              {{ templateLibraryRequestPending ? 'Loading…' : 'Load from path' }}
            </button>
            <button
              class="ghost-btn"
              @click="handleSaveTemplateLibraryChooser"
              :disabled="templateLibraryRequestPending"
              title="Open a save dialog and write the current draft to a location you choose. In browsers without save-dialog support, this falls back to a normal download."
            >
              Save file
            </button>
            <button
              class="ghost-btn"
              @click="handleSaveTemplateLibraryFile"
              :disabled="templateLibraryRequestPending"
              title="Write the current draft to the repo/backend file path shown in Save path."
            >
              {{ templateLibraryRequestPending ? 'Saving…' : 'Save to path' }}
            </button>
            <button
              class="ghost-btn"
              @click="handlePushTemplateLibraryDraft"
              :disabled="templateLibraryRequestPending || !templateLibraryDirty"
              title="Replace the active in-session template library used by the current UI session with this draft."
            >
              Push to session
            </button>
            <button
              class="ghost-btn"
              @click="handleReloadTemplateLibraryDraft"
              :disabled="templateLibraryRequestPending"
              title="Discard local draft changes and reload the current in-session template library."
            >
              Reload session
            </button>
          </div>
          <div class="template-library-meta">
            <span class="status-note">
              Source:
              <strong>{{ templateLibrarySourceLabel }}</strong>
            </span>
            <span class="status-note">
              Templates:
              <strong>{{ templateLibraryDraft.templates?.length || 0 }}</strong>
            </span>
            <span v-if="templateLibrarySessionPath && templateLibrarySource !== 'file_upload'" class="status-note">
              Active path:
              <strong>{{ templateLibrarySessionPath }}</strong>
            </span>
            <span v-if="templateLibraryImportedFilename" class="status-note">
              Imported file:
              <strong>{{ templateLibraryImportedFilename }}</strong>
            </span>
            <span v-if="templateLibraryDirty" class="status-note template-library-dirty">
              Local draft has unsaved changes.
            </span>
          </div>
          <div class="status-note">
            `Import file` loads a browser-selected file into the session only. `Save file` opens a browser save dialog. `Save to path` writes to the backend path shown in `Save path`.
          </div>
          <div v-if="templateLibraryStatus" class="status-note">
            {{ templateLibraryStatus }}
          </div>
          <div v-if="templateLibraryError" class="policy-status error">
            {{ templateLibraryError }}
          </div>
        </div>

        <div class="template-library-picker-card">
          <div class="eval-row">
            <label>
              Selected template
              <span
                class="template-help"
                data-tooltip="The template currently being edited. Selecting a different template loads its anchors, ball handler, jitter, and roles into the board authoring view."
              >?</span>
            </label>
            <select
              v-model="selectedEditableTemplateId"
              :disabled="!editableTemplateOptions.length"
              title="Select which template in the library you want to edit."
            >
              <option v-if="!editableTemplateOptions.length" disabled value="">
                No templates loaded
              </option>
              <option
                v-for="opt in editableTemplateOptions"
                :key="`editable-template-${opt.value}`"
                :value="opt.value"
              >
                {{ opt.label }}
              </option>
            </select>
            <button
              class="ghost-btn"
              @click="addTemplateFromCurrentBoard"
              title="Create a new template using the current board positions and current template editor settings."
            >
              Add from board
            </button>
            <button
              class="ghost-btn"
              @click="duplicateSelectedTemplate"
              :disabled="!selectedEditableTemplate"
              title="Copy the selected template to a new template id."
            >
              Duplicate
            </button>
            <button
              class="ghost-btn template-danger-btn"
              @click="deleteSelectedTemplate"
              :disabled="!selectedEditableTemplate"
              title="Delete the selected template from the draft library."
            >
              Delete
            </button>
            <button
              class="ghost-btn"
              @click="copyTemplateLibraryJson"
              title="Copy the entire current draft library as JSON."
            >
              Copy library JSON
            </button>
            <span v-if="templateCopyStatus" class="status-note">{{ templateCopyStatus }}</span>
          </div>
          <div
            v-if="editableTemplateOptions.length"
            class="template-library-chip-list"
          >
            <button
              v-for="opt in editableTemplateOptions"
              :key="`template-chip-${opt.value}`"
              class="template-chip-btn"
              :class="{ active: selectedEditableTemplateId === opt.value }"
              @click="selectedEditableTemplateId = opt.value"
            >
              {{ opt.label }}
            </button>
          </div>
        </div>

        <div v-if="selectedEditableTemplate" class="template-editor-shell">
          <div class="eval-row">
            <label>
              Template id
              <span
                class="template-help"
                data-tooltip="Stable identifier for this template. This is what training, eval, UI controls, and MLflow artifacts use to reference the template."
              >?</span>
            </label>
            <input
              type="text"
              :value="templateConfigSafe.templateId"
              @input="setTemplateId($event.target.value)"
              placeholder="wing_entry_help"
              title="Stable template identifier used everywhere else in the app."
            />
            <span class="status-note">Board drags update the selected draft template directly in this tab.</span>
          </div>

          <div class="eval-row">
            <label>
              Weight
              <span
                class="template-help"
                data-tooltip="Relative sampling weight when this library is used during training resets. Higher weight means this template is sampled more often."
              >?</span>
            </label>
            <input
              type="number"
              min="0.01"
              step="0.01"
              :value="templateConfigSafe.weight"
              @input="emitTemplateConfigUpdate({ weight: Math.max(0.01, Number($event.target.value) || 1.0) })"
              title="Relative reset sampling weight for this template."
            />
            <label class="inline-label">
              <input
                type="checkbox"
                :checked="templateConfigSafe.mirrorable"
                @change="emitTemplateConfigUpdate({ mirrorable: $event.target.checked })"
                title="Allow training/UI to mirror this template left-right."
              />
              Mirrorable
              <span
                class="template-help"
                data-tooltip="If enabled, the template can be mirrored left-right. This only permits mirroring; it does not force it."
              >?</span>
            </label>
            <label>
              Shot clock
              <span
                class="template-help"
                data-tooltip="Optional shot clock value applied when the template resolves. Useful when a template is supposed to represent a later-clock situation."
              >?</span>
            </label>
            <input
              type="number"
              min="1"
              step="1"
              :value="templateConfigSafe.shotClock"
              @input="emitTemplateConfigUpdate({ shotClock: Math.max(1, Number($event.target.value) || 24) })"
              title="Shot clock assigned when this template resolves."
            />
          </div>

          <div class="eval-row">
            <label>
              Ball starts with
              <span
                class="template-help"
                data-tooltip="Which offense player row gets the has_ball marker in this concrete authoring view. Training still randomizes same-team assignment when the template resolves."
              >?</span>
            </label>
            <select
              :value="templateConfigSafe.ballHolder ?? ''"
              @change="setTemplateBallHolder($event.target.value ? Number($event.target.value) : null)"
              title="Concrete authoring-time ball holder for this template."
            >
              <option v-if="!templateBallStartOptions.length" disabled value="">No offense players</option>
              <option v-for="pid in templateBallStartOptions" :key="`template-ball-${pid}`" :value="pid">Player {{ pid }}</option>
            </select>
            <button
              class="ghost-btn"
              @click="seedTemplateConfigFromGameState()"
              title="Overwrite the selected template’s anchors with the current board state."
            >
              Use current board positions
            </button>
          </div>

          <div class="eval-skills template-player-grid">
            <div class="skills-header template-grid-header">
              <span title="Concrete board player row used in the editor.">Player</span>
              <span title="Whether this row belongs to offense or defense.">Team</span>
              <span title="Template entry index within that team.">Entry</span>
              <span title="Anchor hex [q, r] for this entry. You can drag on the board or edit the numbers directly.">Anchor</span>
              <span title="Per-entry jitter radius before the library-wide jitter scale is applied.">Jitter</span>
              <span title="Optional semantic tag for later analysis or UI display.">Role</span>
            </div>
            <div class="skills-row template-grid-row" v-for="row in templatePlayerRows" :key="`template-player-${row.playerId}`">
              <span class="skills-player">P{{ row.playerId }}</span>
              <span>{{ row.teamLabel }}</span>
              <span>{{ row.entryIndex }}</span>
              <div class="template-anchor-inputs">
                <input
                  type="number"
                  step="1"
                  :value="row.q"
                  @input="updateTemplatePlayerAnchor(row.playerId, 0, $event.target.value)"
                  title="Anchor q coordinate."
                />
                <input
                  type="number"
                  step="1"
                  :value="row.r"
                  @input="updateTemplatePlayerAnchor(row.playerId, 1, $event.target.value)"
                  title="Anchor r coordinate."
                />
              </div>
              <input
                type="number"
                min="0"
                step="1"
                :value="row.jitter"
                @input="updateTemplatePlayerJitter(row.playerId, $event.target.value)"
                title="Per-entry jitter radius."
              />
              <input
                type="text"
                :value="row.role"
                @input="updateTemplatePlayerRole(row.playerId, $event.target.value)"
                :placeholder="row.teamLabel === 'Offense' && row.playerId === templateConfigSafe.ballHolder ? 'ball_handler' : 'role'"
                title="Optional semantic role label."
              />
            </div>
          </div>

          <div class="eval-row template-export-actions">
            <button
              class="ghost-btn"
              @click="copyTemplateExport('yaml')"
              title="Copy only the selected template entry as YAML."
            >
              Copy YAML
            </button>
            <button
              class="ghost-btn"
              @click="copyTemplateExport('json')"
              title="Copy only the selected template entry as JSON."
            >
              Copy JSON
            </button>
          </div>

          <div class="eval-custom template-export-grid">
            <div class="template-export-card">
              <h5 title="Selected template only, formatted as YAML.">YAML Template Entry</h5>
              <textarea class="template-export-textarea" readonly :value="templateYamlExport"></textarea>
            </div>
            <div class="template-export-card">
              <h5 title="Selected template only, formatted as JSON.">JSON Template Entry</h5>
              <textarea class="template-export-textarea" readonly :value="templateJsonExport"></textarea>
            </div>
            <div class="template-export-card">
              <h5 title="Entire current draft library, formatted as JSON.">JSON Library Draft</h5>
              <textarea class="template-export-textarea" readonly :value="templateLibraryJsonExport"></textarea>
            </div>
          </div>
        </div>

        <div v-else class="no-data">
          Load a template library or add a new template from the current board.
        </div>
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
        <div v-if="props.isEvaluating" class="eval-progress-wrap">
          <div class="eval-progress-bar" :aria-valuenow="evalProgressSafe.completed" :aria-valuemin="0" :aria-valuemax="Math.max(1, evalProgressSafe.total)" role="progressbar">
            <div
              class="eval-progress-fill"
              :class="{ indeterminate: evalProgressSafe.total <= 0 }"
              :style="evalProgressSafe.total > 0 ? { width: evalProgressPercent } : null"
            ></div>
          </div>
          <span class="eval-status">
            {{ evalProgressSafe.completed }}/{{ evalProgressSafe.total || evalEpisodesInput }}
            episodes
            <span v-if="evalProgressSafe.total > 0">({{ evalProgressPercent }})</span>
          </span>
        </div>
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

      <div class="eval-row">
        <label>Intent selection</label>
        <select
          :value="evalConfigSafe.intentSelectionMode"
          :disabled="props.isEvaluating || !selectorEnabled"
          @change="setEvalIntentSelectionMode($event.target.value)"
        >
          <option value="learned_sample">Learned sample</option>
          <option value="best_intent">Best intent (argmax)</option>
          <option value="uniform_random">Uniform random</option>
        </select>
        <span v-if="!selectorEnabled" class="status-note">
          Selector is not enabled for this checkpoint.
        </span>
      </div>

      <div v-if="evalModeIsCustom" class="eval-custom">
        <div class="eval-row">
          <label class="inline-label">
            <input type="checkbox" :checked="evalPlacementEditing" @change="toggleEvalPlacement($event.target.checked)" />
            Edit starting positions on board
          </label>
          <button class="ghost-btn" @click="seedEvalConfigFromGameState(false)" :disabled="!evalPlacementEditing">
            Use current board positions
          </button>
          <span v-if="!evalPlacementEditing" class="status-note">
            Spawn and ball handler randomize each episode
          </span>
        </div>

        <div class="eval-row">
          <label>Ball starts with</label>
          <select
            :value="evalConfigSafe.ballHolder ?? ''"
            :disabled="!evalPlacementEditing"
            @change="setEvalBallHolder($event.target.value ? Number($event.target.value) : null)"
          >
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
        <div v-else>
        <div class="parameters-grid">
          <div v-if="hasLoadedStartTemplates" class="param-category">
            <h5>Start Templates</h5>
            <div class="eval-row">
              <label>Template</label>
              <select v-model="selectedStartTemplateId">
                <option v-for="opt in startTemplateOptions" :key="`env-template-${opt.value}`" :value="opt.value">
                  {{ opt.label }}
                </option>
              </select>
            </div>
            <div class="eval-row">
              <label class="inline-label">
                <input type="checkbox" v-model="selectedStartTemplateMirrored" />
                Mirror L|R
              </label>
              <button
                class="ghost-btn"
                @click="handleApplyStartTemplateToBoard"
                :disabled="!selectedStartTemplateId"
              >
                Apply to board
              </button>
            </div>
            <div class="status-note">
              Apply a template here, then use Playbook with current state. For Eval custom setup, use the existing current-board copy flow in the Eval tab.
            </div>
            <div v-if="startTemplatePreviewModel" class="start-template-preview">
              <svg
                class="start-template-preview-svg"
                :viewBox="`0 0 ${startTemplatePreviewModel.viewBox.width} ${startTemplatePreviewModel.viewBox.height}`"
                role="img"
                :aria-label="`Schematic preview for ${startTemplatePreviewModel.templateId}${startTemplatePreviewModel.mirrored ? ', mirrored left-right' : ''}`"
              >
                <rect
                  class="start-template-preview-board"
                  :x="startTemplatePreviewModel.court.x"
                  :y="startTemplatePreviewModel.court.y"
                  :width="startTemplatePreviewModel.court.width"
                  :height="startTemplatePreviewModel.court.height"
                  rx="8"
                />
                <line
                  class="start-template-preview-line"
                  :x1="startTemplatePreviewModel.court.backboardX"
                  :y1="startTemplatePreviewModel.court.laneY"
                  :x2="startTemplatePreviewModel.court.backboardX"
                  :y2="startTemplatePreviewModel.court.laneY + startTemplatePreviewModel.court.laneHeight"
                />
                <circle
                  class="start-template-preview-line"
                  :cx="startTemplatePreviewModel.court.hoopX"
                  :cy="startTemplatePreviewModel.court.hoopY"
                  :r="startTemplatePreviewModel.court.hoopRadius"
                />
                <rect
                  class="start-template-preview-line"
                  :x="startTemplatePreviewModel.court.laneX"
                  :y="startTemplatePreviewModel.court.laneY"
                  :width="startTemplatePreviewModel.court.laneWidth"
                  :height="startTemplatePreviewModel.court.laneHeight"
                />
                <line
                  class="start-template-preview-line"
                  :x1="startTemplatePreviewModel.court.x"
                  :y1="startTemplatePreviewModel.court.arcTopY"
                  :x2="startTemplatePreviewModel.court.arcLineX"
                  :y2="startTemplatePreviewModel.court.arcTopY"
                />
                <line
                  class="start-template-preview-line"
                  :x1="startTemplatePreviewModel.court.x"
                  :y1="startTemplatePreviewModel.court.arcBottomY"
                  :x2="startTemplatePreviewModel.court.arcLineX"
                  :y2="startTemplatePreviewModel.court.arcBottomY"
                />
                <path
                  class="start-template-preview-line"
                  :d="`M ${startTemplatePreviewModel.court.arcLineX} ${startTemplatePreviewModel.court.arcTopY}
                       A ${startTemplatePreviewModel.court.arcRadius} ${startTemplatePreviewModel.court.arcRadius} 0 0 1 ${startTemplatePreviewModel.court.arcLineX} ${startTemplatePreviewModel.court.arcBottomY}`"
                />
                <text
                  v-for="entry in startTemplatePreviewModel.entries"
                  :key="entry.key"
                  :x="entry.x"
                  :y="entry.y"
                  :class="['start-template-preview-marker', `team-${entry.team}`]"
                  text-anchor="middle"
                  dominant-baseline="middle"
                >
                  {{ entry.marker }}
                </text>
              </svg>
              <div class="status-note">
                Anchor schematic only. Jitter is not shown.
              </div>
            </div>
            <div v-if="startTemplateActionStatus" class="status-note">
              {{ startTemplateActionStatus }}
            </div>
            <div v-if="startTemplateActionError" class="policy-status error">
              {{ startTemplateActionError }}
            </div>
          </div>
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
            <div class="param-item" data-tooltip="MLflow run ID used for currently loaded policies/state.">
              <span class="param-name">Run ID:</span>
              <span class="param-value">{{ props.gameState.run_id || 'N/A' }}</span>
            </div>
            <div class="param-item" data-tooltip="Stable nominal codename for the currently loaded model/run.">
              <span class="param-name">Model codename:</span>
              <span class="param-value">{{ props.gameState.model_codename || 'N/A' }}</span>
            </div>
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
                <span v-else>⟳ Refresh </span>
              </button>
              <button
                class="refresh-policies-btn"
                @click="$emit('swap-teams-requested')"
                :disabled="policiesLoading || props.isPolicySwapping || !props.gameState?.run_id"
                title="Reinitialize the current run with the opposite user-controlled team"
              >
                ↔ Swap Teams
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
              <div class="param-item" data-tooltip="Absolute FG% penalty per extra hex beyond (three_point_distance + 1). Example: 5% = -0.05 per extra hex.">
                <span class="param-name">3PT extra decay / hex:</span>
                <span class="param-value">{{ ((props.gameState.shot_params.three_pt_extra_hex_decay ?? 0) * 100).toFixed(1) }}%</span>
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
                  Reset 
                </button>
                <button 
                  class="refresh-policies-btn" 
                  @click="applyOffenseSkillOverrides" 
                  :disabled="skillsUpdating || !offenseSkillRows.length"
                >
                  {{ skillsUpdating ? 'Saving...' : 'Apply ' }}
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
              <input
                class="env-param-input"
                type="number"
                min="0"
                step="1"
                v-model.number="pressureParamsInput.defender_pressure_distance"
                :disabled="pressureParamsUpdating"
              />
            </div>
            <div class="param-item" data-tooltip="Base probability of turnover when a defender is adjacent (distance=1) to ball handler">
              <span class="param-name">Turnover chance:</span>
              <input
                class="env-param-input"
                type="number"
                min="0"
                max="1"
                step="0.01"
                v-model.number="pressureParamsInput.defender_pressure_turnover_chance"
                :disabled="pressureParamsUpdating"
              />
            </div>
            <div class="param-item" data-tooltip="Exponential decay rate for pressure. Higher values = pressure drops off faster with distance.">
              <span class="param-name">Decay lambda:</span>
              <input
                class="env-param-input"
                type="number"
                min="0"
                step="0.01"
                v-model.number="pressureParamsInput.defender_pressure_decay_lambda"
                :disabled="pressureParamsUpdating"
              />
            </div>
            <div class="offense-skill-actions">
              <button
                class="refresh-policies-btn"
                @click="resetDefenderTurnoverPressureToMlflowDefaults"
                :disabled="isPressureSectionUpdating('defender')"
                title="Reset defender turnover pressure values to MLflow defaults for this run"
              >
                Reset
              </button>
              <button
                class="refresh-policies-btn"
                @click="applyDefenderTurnoverPressureOverrides"
                :disabled="isPressureSectionUpdating('defender')"
              >
                {{ isPressureSectionUpdating('defender') ? 'Saving...' : 'Apply' }}
              </button>
            </div>
          </div>
          <div class="param-category">
            <h5>Pass Interception (Line-of-Sight)</h5>
            <div class="param-item" data-tooltip="Base probability that a defender intercepts a pass when directly on the pass line">
              <span class="param-name">Base steal rate:</span>
              <input
                class="env-param-input"
                type="number"
                min="0"
                max="1"
                step="0.01"
                v-model.number="pressureParamsInput.base_steal_rate"
                :disabled="pressureParamsUpdating"
              />
            </div>
            <div class="param-item" data-tooltip="How quickly steal probability drops as defender is further from pass line. Higher = faster decay.">
              <span class="param-name">Perpendicular decay:</span>
              <input
                class="env-param-input"
                type="number"
                min="0"
                step="0.01"
                v-model.number="pressureParamsInput.steal_perp_decay"
                :disabled="pressureParamsUpdating"
              />
            </div>
            <div class="param-item" data-tooltip="How pass distance affects interception chance. Longer passes are easier to intercept.">
              <span class="param-name">Distance factor:</span>
              <input
                class="env-param-input"
                type="number"
                min="0"
                step="0.01"
                v-model.number="pressureParamsInput.steal_distance_factor"
                :disabled="pressureParamsUpdating"
              />
            </div>
            <div class="param-item" data-tooltip="Minimum weight for defender's position along pass line (0=near passer, 1=near receiver)">
              <span class="param-name">Position weight min:</span>
              <input
                class="env-param-input"
                type="number"
                min="0"
                max="1"
                step="0.01"
                v-model.number="pressureParamsInput.steal_position_weight_min"
                :disabled="pressureParamsUpdating"
              />
            </div>
            <div class="offense-skill-actions">
              <button
                class="refresh-policies-btn"
                @click="resetPassInterceptionToMlflowDefaults"
                :disabled="isPressureSectionUpdating('pass')"
                title="Reset pass interception values to MLflow defaults for this run"
              >
                Reset
              </button>
              <button
                class="refresh-policies-btn"
                @click="applyPassInterceptionOverrides"
                :disabled="isPressureSectionUpdating('pass')"
              >
                {{ isPressureSectionUpdating('pass') ? 'Saving...' : 'Apply' }}
              </button>
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
            <div class="param-item" data-tooltip="Absolute FG% penalty applied per extra hex beyond (three_point_distance + 1). Example: 0.05 = minus 5 percentage points per extra hex.">
              <span class="param-name">3PT extra decay / hex:</span>
              <input
                class="env-param-input"
                type="number"
                min="0"
                max="1"
                step="0.01"
                v-model.number="pressureParamsInput.three_pt_extra_hex_decay"
                :disabled="pressureParamsUpdating"
              />
            </div>
            <div class="param-item" data-tooltip="Whether nearby defenders reduce shot accuracy">
              <span class="param-name">Pressure enabled:</span>
              <label class="inline-label env-checkbox">
                <input
                  type="checkbox"
                  v-model="pressureParamsInput.shot_pressure_enabled"
                  :disabled="pressureParamsUpdating"
                />
                {{ pressureParamsInput.shot_pressure_enabled ? 'Enabled' : 'Disabled' }}
              </label>
            </div>
            <div class="param-item" data-tooltip="Maximum percentage reduction in shot accuracy from defender pressure (e.g., 0.3 = up to 30% reduction)">
              <span class="param-name">Max pressure:</span>
              <input
                class="env-param-input"
                type="number"
                min="0"
                max="1"
                step="0.01"
                v-model.number="pressureParamsInput.shot_pressure_max"
                :disabled="pressureParamsUpdating"
              />
            </div>
            <div class="param-item" data-tooltip="Exponential decay rate for shot pressure over distance. Higher = pressure drops faster.">
              <span class="param-name">Pressure lambda:</span>
              <input
                class="env-param-input"
                type="number"
                min="0"
                step="0.01"
                v-model.number="pressureParamsInput.shot_pressure_lambda"
                :disabled="pressureParamsUpdating"
              />
            </div>
            <div class="param-item" data-tooltip="Angular width of the defensive pressure cone. Defenders outside this arc apply less pressure.">
              <span class="param-name">Pressure arc degrees:</span>
              <input
                class="env-param-input"
                type="number"
                min="0"
                max="360"
                step="1"
                v-model.number="pressureParamsInput.shot_pressure_arc_degrees"
                :disabled="pressureParamsUpdating"
              />
            </div>
            <div class="offense-skill-actions">
                <button
                  class="refresh-policies-btn"
                  @click="resetPressureParametersToMlflowDefaults"
                  :disabled="isPressureSectionUpdating('shot')"
                  title="Reset shot-distance decay and shot pressure values to MLflow defaults for this run"
                >
                  Reset
                </button>
              <button
                class="refresh-policies-btn"
                @click="applyPressureParameterOverrides"
                :disabled="isPressureSectionUpdating('shot')"
              >
                {{ isPressureSectionUpdating('shot') ? 'Saving...' : 'Apply ' }}
              </button>
            </div>
            <div class="policy-status error" v-if="pressureParamsError">
              {{ pressureParamsError }}
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
              <input
                class="env-param-input"
                type="number"
                step="0.05"
                v-model.number="passLogitBiasInput"
                :disabled="passLogitBiasUpdating"
              />
            </div>
            <div class="param-item" data-tooltip="Default pass logit bias from the loaded policy/config.">
              <span class="param-name">Default bias:</span>
              <span class="param-value">{{ passLogitBiasDefault.toFixed(2) }}</span>
            </div>
            <div class="offense-skill-actions">
              <button
                class="refresh-policies-btn"
                @click="resetPassLogitBiasDefault"
                :disabled="passLogitBiasUpdating"
              >
                Reset
              </button>
              <button
                class="refresh-policies-btn"
                @click="applyPassLogitBiasOverride"
                :disabled="passLogitBiasUpdating"
              >
                {{ passLogitBiasUpdating ? 'Saving...' : 'Apply' }}
              </button>
            </div>
            <div class="policy-status error" v-if="passLogitBiasError">
              {{ passLogitBiasError }}
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

          <div class="param-category">
            <h5>Counterfactual Snapshot</h5>
            <div class="param-item" data-tooltip="Whether a branch point snapshot is currently stored for restoring the exact current state later.">
              <span class="param-name">Snapshot stored:</span>
              <span class="param-value">{{ counterfactualSnapshotAvailable ? '✓ Yes' : '✗ No' }}</span>
            </div>
            <div class="param-item" data-tooltip="Step index captured when the snapshot was stored.">
              <span class="param-name">Captured step:</span>
              <span class="param-value">{{ counterfactualSnapshotStep ?? 'N/A' }}</span>
            </div>
            <div class="param-item" data-tooltip="Shot clock value in the stored snapshot.">
              <span class="param-name">Snapshot shot clock:</span>
              <span class="param-value">{{ counterfactualSnapshotShotClock ?? 'N/A' }}</span>
            </div>
            <div class="param-item" data-tooltip="Ball holder stored in the snapshot.">
              <span class="param-name">Snapshot ball holder:</span>
              <span class="param-value">{{ counterfactualSnapshotBallHolder ?? 'N/A' }}</span>
            </div>
            <div class="param-item" data-tooltip="Intent state stored in the snapshot.">
              <span class="param-name">Snapshot intent:</span>
              <span class="param-value">{{ counterfactualSnapshotIntentSummary }}</span>
            </div>
            <div class="offense-skill-actions">
              <button
                class="refresh-policies-btn"
                @click="handleCaptureCounterfactualSnapshot"
                :disabled="counterfactualSnapshotControlsDisabled"
              >
                <font-awesome-icon :icon="['fas', 'camera']" />
                <span>{{ counterfactualSnapshotUpdating ? 'Saving...' : 'Capture' }}</span>
              </button>
              <button
                class="refresh-policies-btn"
                @click="handleRestoreCounterfactualSnapshot"
                :disabled="counterfactualSnapshotControlsDisabled || !counterfactualSnapshotAvailable"
              >
                <font-awesome-icon :icon="['fas', 'redo']" />
                <span>{{ counterfactualSnapshotUpdating ? 'Restoring...' : 'Restore' }}</span>
              </button>
              <button
                class="refresh-policies-btn"
                @click="handleReplayCounterfactualSnapshot"
                :disabled="counterfactualSnapshotControlsDisabled || !counterfactualSnapshotAvailable || props.gameState?.done"
                data-tooltip="Autoplay deterministically from the current live state. Restore the snapshot first if you want to branch from the saved point."
              >
                <font-awesome-icon :icon="['fas', 'play']" />
                <span>{{ counterfactualSnapshotUpdating ? 'Playing...' : 'Play' }}</span>
              </button>
            </div>
            <div class="policy-status error" v-if="counterfactualSnapshotError">
              {{ counterfactualSnapshotError }}
            </div>
          </div>

          <div class="param-category">
            <h5>Intent Learning</h5>
            <div class="param-item" data-tooltip="Whether latent intent/play learning is enabled in this environment.">
              <span class="param-name">Enabled:</span>
              <span class="param-value">{{ props.gameState.enable_intent_learning ? '✓ Yes' : '✗ No' }}</span>
            </div>
            <div class="param-item" data-tooltip="Number of latent intent categories available to offense when intent is active.">
              <span class="param-name">Num intents:</span>
              <span class="param-value">{{ props.gameState.num_intents ?? 'N/A' }}</span>
            </div>
            <div class="param-item" data-tooltip="Intent observability mode (private_offense, public, hidden).">
              <span class="param-name">Observation mode:</span>
              <span class="param-value">{{ props.gameState.intent_obs_mode || 'N/A' }}</span>
            </div>
            <div class="param-item" data-tooltip="Configured commitment window for active intent state.">
              <span class="param-name">Commitment steps:</span>
              <span class="param-value">{{ props.gameState.intent_commitment_steps ?? 'N/A' }}</span>
            </div>
            <div class="param-item" data-tooltip="Configured probability of null/no-intent episodes.">
              <span class="param-name">Null intent prob:</span>
              <span class="param-value">{{ props.gameState.intent_null_prob != null ? (props.gameState.intent_null_prob * 100).toFixed(1) + '%' : 'N/A' }}</span>
            </div>
            <div class="param-item" data-tooltip="Configured probability of exposing intent to defense.">
              <span class="param-name">Visible-to-defense prob:</span>
              <span class="param-value">{{ props.gameState.intent_visible_to_defense_prob != null ? (props.gameState.intent_visible_to_defense_prob * 100).toFixed(1) + '%' : 'N/A' }}</span>
            </div>
            <div class="param-item" data-tooltip="Whether intent-diversity objective is enabled for the model run loaded in this session.">
              <span class="param-name">Diversity enabled:</span>
              <span class="param-value">{{ props.gameState.intent_diversity_enabled == null ? 'N/A' : (props.gameState.intent_diversity_enabled ? '✓ Yes' : '✗ No') }}</span>
            </div>
            <div class="param-item" data-tooltip="Whether current possession has an active latent intent.">
              <span class="param-name">Current active:</span>
              <template v-if="props.gameState.enable_intent_learning">
                <label class="param-value">
                  <input
                    type="checkbox"
                    v-model="intentStateInput.active"
                    :disabled="intentControlsDisabled"
                  />
                  <span>{{ intentStateInput.active ? ' Enabled' : ' Disabled' }}</span>
                </label>
              </template>
              <span v-else class="param-value">{{ props.gameState.intent_active_current ? '✓ Yes' : '✗ No' }}</span>
            </div>
            <div class="param-item" data-tooltip="Current latent play identity (masked elsewhere when hidden).">
              <span class="param-name">Current play:</span>
              <div v-if="props.gameState.enable_intent_learning" class="param-select-wrapper">
                <select
                  v-model.number="intentStateInput.intent_index"
                  :disabled="intentControlsDisabled"
                >
                  <option
                    v-for="opt in currentPlayOptions"
                    :key="opt.value"
                    :value="opt.value"
                  >
                    {{ opt.label }}
                  </option>
                </select>
              </div>
              <span v-else class="param-value">
                {{ formatPlayLabel(props.gameState.intent_index_current, props.gameState?.play_name_map, props.gameState?.current_play_name) }}
              </span>
            </div>
            <div class="param-item" data-tooltip="Current age of active intent (steps since sampled).">
              <span class="param-name">Current age:</span>
              <input
                v-if="props.gameState.enable_intent_learning"
                class="env-param-input"
                type="number"
                min="0"
                :max="intentAgeMax"
                step="1"
                v-model.number="intentStateInput.intent_age"
                :disabled="intentControlsDisabled"
              />
              <span v-else class="param-value">{{ props.gameState.intent_age ?? 'N/A' }}</span>
            </div>
            <div class="param-item" data-tooltip="Remaining commitment steps before intent can be reconsidered.">
              <span class="param-name">Commitment remaining:</span>
              <span class="param-value">{{ props.gameState.intent_commitment_remaining ?? 'N/A' }}</span>
            </div>
            <div class="param-item" data-tooltip="Whether this possession's intent is currently visible to the defense role.">
              <span class="param-name">Current defense-visible:</span>
              <span class="param-value">{{ props.gameState.intent_visible_to_defense_current ? '✓ Yes' : '✗ No' }}</span>
            </div>
            <div class="offense-skill-actions" v-if="props.gameState.enable_intent_learning">
              <button
                class="refresh-policies-btn"
                @click="resetIntentStateInputs"
                :disabled="intentStateUpdating"
              >
                Reset
              </button>
              <button
                class="refresh-policies-btn"
                @click="applyIntentStateOverride"
                :disabled="intentControlsDisabled"
              >
                {{ intentStateUpdating ? 'Saving...' : 'Apply' }}
              </button>
              <div class="policy-status error" v-if="intentStateError">
                {{ intentStateError }}
              </div>
            </div>
          </div>

          <div class="param-category">
            <h5>&mu; Selector</h5>
            <div class="param-item" data-tooltip="Whether the learned play selector mu(z|s) was enabled in the loaded training run.">
              <span class="param-name">Enabled:</span>
              <span class="param-value">{{ selectorEnabled ? '✓ Yes' : '✗ No' }}</span>
            </div>
            <div class="param-item" data-tooltip="Which selector implementation path trained this checkpoint.">
              <span class="param-name">Path:</span>
              <span class="param-value">{{ selectorImplementationLabel }}</span>
            </div>
            <div class="param-item" data-tooltip="Current implementation mixes uniform intent sampling with selector-driven play calling using alpha.">
              <span class="param-name">Alpha schedule:</span>
              <span class="param-value">{{ selectorAlphaSummary }}</span>
            </div>
            <div class="param-item" data-tooltip="Warmup and ramp window controlling when selector-driven play calling begins to replace uniform sampling.">
              <span class="param-name">Warmup / ramp:</span>
              <span class="param-value">{{ selectorScheduleSummary }}</span>
            </div>
            <div class="param-item" data-tooltip="Entropy regularization applied to the selector's categorical distribution over intents.">
              <span class="param-name">Entropy coef:</span>
              <span class="param-value">{{ selectorTrainingParams.intent_selector_entropy_coef ?? 'N/A' }}</span>
            </div>
            <div class="param-item" data-tooltip="Coverage regularizer that penalizes selector collapse by pushing average usage back toward uniform.">
              <span class="param-name">Usage reg coef:</span>
              <span class="param-value">{{ selectorTrainingParams.intent_selector_usage_reg_coef ?? 'N/A' }}</span>
            </div>
            <div class="param-item" data-tooltip="Integrated selector runs add a selector critic/value head on top of the shared selector context.">
              <span class="param-name">Selector critic:</span>
              <span class="param-value">{{ selectorCriticEnabled ? '✓ Enabled' : '✗ No / not logged' }}</span>
            </div>
            <div class="param-item" data-tooltip="Weight on the selector value loss when the integrated selector critic is enabled.">
              <span class="param-name">Value coef:</span>
              <span class="param-value">{{ selectorValueCoef ?? 'N/A' }}</span>
            </div>
            <div class="param-item" data-tooltip="The selector uses the shared set-attention state encoder and emits intent logits, plus a selector value estimate in the integrated critic path.">
              <span class="param-name">Architecture:</span>
              <span class="param-value">{{ selectorEnabled ? `${selectorHeadContextLabel}, ${selectorHeadSummary}` : 'N/A' }}</span>
            </div>
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
            <h5>Discriminator Architecture</h5>
            <div class="param-item" data-tooltip="Whether the DIAYN-style intent discriminator bonus was enabled during training.">
              <span class="param-name">Intent diversity:</span>
              <span class="param-value">{{ props.gameState.training_params.intent_diversity_enabled ? '✓ Enabled' : '✗ Disabled' }}</span>
            </div>
            <div class="param-item" data-tooltip="Encoder used to summarize the intent-conditioned trajectory before classification.">
              <span class="param-name">Encoder:</span>
              <span class="param-value">{{ props.gameState.training_params.intent_disc_encoder_type || 'N/A' }}</span>
            </div>
            <div class="param-item" data-tooltip="Hidden width of the discriminator MLP head.">
              <span class="param-name">Hidden dim:</span>
              <span class="param-value">{{ props.gameState.training_params.intent_disc_hidden_dim ?? 'N/A' }}</span>
            </div>
            <div
              class="param-item"
              v-if="props.gameState.training_params.intent_disc_encoder_type === 'gru'"
              data-tooltip="Per-step embedding width before the GRU consumes the trajectory."
            >
              <span class="param-name">Step dim:</span>
              <span class="param-value">{{ props.gameState.training_params.intent_disc_step_dim ?? 'N/A' }}</span>
            </div>
            <div class="param-item" data-tooltip="Dropout applied inside the discriminator network.">
              <span class="param-name">Dropout:</span>
              <span class="param-value">{{ props.gameState.training_params.intent_disc_dropout ?? 'N/A' }}</span>
            </div>
            <div class="param-item" data-tooltip="Maximum flattened observation features passed into the discriminator per step.">
              <span class="param-name">Max obs dim:</span>
              <span class="param-value">{{ props.gameState.training_params.intent_disc_max_obs_dim ?? 'N/A' }}</span>
            </div>
            <div class="param-item" data-tooltip="Maximum flattened action features passed into the discriminator per step.">
              <span class="param-name">Max action dim:</span>
              <span class="param-value">{{ props.gameState.training_params.intent_disc_max_action_dim ?? 'N/A' }}</span>
            </div>
            <div class="param-item" data-tooltip="Optimizer learning rate for the intent discriminator.">
              <span class="param-name">Disc LR:</span>
              <span class="param-value">{{ props.gameState.training_params.intent_disc_lr?.toExponential?.(2) || 'N/A' }}</span>
            </div>
            <div class="param-item" data-tooltip="Batch size for discriminator updates at rollout end.">
              <span class="param-name">Disc batch size:</span>
              <span class="param-value">{{ props.gameState.training_params.intent_disc_batch_size ?? 'N/A' }}</span>
            </div>
            <div class="param-item" data-tooltip="How many discriminator optimization passes run after each rollout.">
              <span class="param-name">Updates/rollout:</span>
              <span class="param-value">{{ props.gameState.training_params.intent_disc_updates_per_rollout ?? 'N/A' }}</span>
            </div>
            <div class="param-item" data-tooltip="Whether training exported the discriminator eval batch artifact after each alternation checkpoint.">
              <span class="param-name">Eval batch output:</span>
              <span class="param-value">{{ props.gameState.training_params.disc_eval_batch_output ? '✓ Enabled' : '✗ Disabled' }}</span>
            </div>
          </div>

          <div class="param-category">
            <h5>Intent Objective</h5>
            <div class="param-item" data-tooltip="Number of discrete latent plays available to the policy.">
              <span class="param-name">Num intents:</span>
              <span class="param-value">{{ props.gameState.training_params.num_intents ?? 'N/A' }}</span>
            </div>
            <div class="param-item" data-tooltip="How many steps an active intent stays in force before it deactivates.">
              <span class="param-name">Commitment steps:</span>
              <span class="param-value">{{ props.gameState.training_params.intent_commitment_steps ?? 'N/A' }}</span>
            </div>
            <div class="param-item" data-tooltip="Who can observe the latent intent in the model observation.">
              <span class="param-name">Obs mode:</span>
              <span class="param-value">{{ props.gameState.training_params.intent_obs_mode || 'N/A' }}</span>
            </div>
            <div class="param-item" data-tooltip="Probability of sampling no offense intent. If scheduled, this is the start value.">
              <span class="param-name">Null prob:</span>
              <span class="param-value" v-if="props.gameState.training_params.intent_null_prob_end != null">
                {{ props.gameState.training_params.intent_null_prob }} → {{ props.gameState.training_params.intent_null_prob_end }}
              </span>
              <span class="param-value" v-else>{{ props.gameState.training_params.intent_null_prob ?? 'N/A' }}</span>
            </div>
            <div class="param-item" data-tooltip="Probability that the offense intent is visible to defense. If scheduled, this is the start value.">
              <span class="param-name">Defense-visible prob:</span>
              <span class="param-value" v-if="props.gameState.training_params.intent_visible_to_defense_prob_end != null">
                {{ props.gameState.training_params.intent_visible_to_defense_prob }} → {{ props.gameState.training_params.intent_visible_to_defense_prob_end }}
              </span>
              <span class="param-value" v-else>{{ props.gameState.training_params.intent_visible_to_defense_prob ?? 'N/A' }}</span>
            </div>
            <div class="param-item" data-tooltip="Whether a separate latent intent is also trained for the defense.">
              <span class="param-name">Defense intent:</span>
              <span class="param-value">{{ props.gameState.training_params.enable_defense_intent_learning ? '✓ Enabled' : '✗ Disabled' }}</span>
            </div>
            <div class="param-item" v-if="props.gameState.training_params.enable_defense_intent_learning" data-tooltip="Probability of sampling no defense intent.">
              <span class="param-name">Defense null prob:</span>
              <span class="param-value">{{ props.gameState.training_params.defense_intent_null_prob ?? 'N/A' }}</span>
            </div>
            <div class="param-item" data-tooltip="Target scale for the DIAYN-style intrinsic bonus once fully ramped.">
              <span class="param-name">Beta target:</span>
              <span class="param-value">{{ props.gameState.training_params.intent_diversity_beta_target ?? 'N/A' }}</span>
            </div>
            <div class="param-item" data-tooltip="Number of timesteps before the intent diversity reward starts contributing.">
              <span class="param-name">Warmup steps:</span>
              <span class="param-value">{{ props.gameState.training_params.intent_diversity_warmup_steps?.toLocaleString?.() || 'N/A' }}</span>
            </div>
            <div class="param-item" data-tooltip="Number of timesteps used to ramp the diversity beta from zero to target.">
              <span class="param-name">Ramp steps:</span>
              <span class="param-value">{{ props.gameState.training_params.intent_diversity_ramp_steps?.toLocaleString?.() || 'N/A' }}</span>
            </div>
            <div class="param-item" data-tooltip="Clip range applied to the normalized discriminator bonus before reward shaping.">
              <span class="param-name">Bonus clip:</span>
              <span class="param-value">{{ props.gameState.training_params.intent_diversity_clip ?? 'N/A' }}</span>
            </div>
          </div>

          <div class="param-category">
            <h5>Task Reward Curriculum</h5>
            <div class="param-item" data-tooltip="Scale applied to the aggregated environment task reward returned to PPO. Values below 1.0 create a DIAYN-first curriculum by downweighting basketball reward early.">
              <span class="param-name">Task reward scale:</span>
              <span class="param-value">{{ taskRewardScaleSummary }}</span>
            </div>
            <div class="param-item" data-tooltip="Timesteps spent holding task reward at the start scale, then ramping it back to the end scale.">
              <span class="param-name">Task reward schedule:</span>
              <span class="param-value">{{ taskRewardScheduleSummary }}</span>
            </div>
          </div>

          <div class="param-category">
            <h5>Start Templates</h5>
            <div class="param-item" data-tooltip="Optional curriculum that resolves a sampled formation template into exact initial positions before reset. When disabled, the environment uses the default spawn generator only.">
              <span class="param-name">Template mode:</span>
              <span class="param-value">{{ props.gameState.training_params.start_template_enabled ? '✓ Enabled' : '✗ Disabled' }}</span>
            </div>
            <div class="param-item" data-tooltip="Probability that a reset uses a sampled start template instead of the default spawn logic.">
              <span class="param-name">Template prob:</span>
              <span class="param-value">{{ props.gameState.training_params.start_template_prob ?? 'N/A' }}</span>
            </div>
            <div class="param-item" data-tooltip="Global multiplier applied to per-player jitter radii inside the sampled start template.">
              <span class="param-name">Jitter scale:</span>
              <span class="param-value">{{ props.gameState.training_params.start_template_jitter_scale ?? 'N/A' }}</span>
            </div>
            <div class="param-item" data-tooltip="Probability of mirroring a mirrorable template left/right before resolving player anchors.">
              <span class="param-name">Mirror prob:</span>
              <span class="param-value">{{ props.gameState.training_params.start_template_mirror_prob ?? 'N/A' }}</span>
            </div>
            <div class="param-item" data-tooltip="If enabled, invalid or unreadable template libraries fail fast instead of silently disabling the start-template feature.">
              <span class="param-name">Strict mode:</span>
              <span class="param-value">{{ props.gameState.training_params.start_template_strict ? '✓ Enabled' : '✗ Disabled' }}</span>
            </div>
            <div class="param-item" data-tooltip="Path to the JSON/YAML template library used for the run.">
              <span class="param-name">Library:</span>
              <span class="param-value">{{ props.gameState.training_params.start_template_library || 'N/A' }}</span>
            </div>
            <div class="param-item" data-tooltip="MLflow artifact path for the canonical persisted start-template library logged with the run.">
              <span class="param-name">Artifact:</span>
              <span class="param-value">{{ props.gameState.training_params.start_template_library_artifact_path || 'N/A' }}</span>
            </div>
            <div class="param-item" data-tooltip="Number of templates available from the persisted library artifact currently loaded into the UI session.">
              <span class="param-name">Template count:</span>
              <span class="param-value">{{ props.gameState.start_template_library?.templates?.length ?? props.gameState.training_params.start_template_library_template_count ?? 'N/A' }}</span>
            </div>
          </div>

          <div class="param-category">
            <h5>&mu; Selector Architecture</h5>
            <div class="param-item" data-tooltip="Whether the learned high-level play selector head mu(z|s) was enabled for this run.">
              <span class="param-name">Selector enabled:</span>
              <span class="param-value">{{ selectorEnabled ? '✓ Enabled' : '✗ Disabled' }}</span>
            </div>
            <div class="param-item" data-tooltip="Current implementation uses the same shared set-attention encoder for low-level control and selector context.">
              <span class="param-name">Selector context:</span>
              <span class="param-value">{{ selectorHeadContextLabel }}</span>
            </div>
            <div class="param-item" data-tooltip="Hidden width of the selector head MLP before intent logits.">
              <span class="param-name">Head hidden dim:</span>
              <span class="param-value">{{ selectorTrainingParams.intent_selector_hidden_dim ?? 'N/A' }}</span>
            </div>
            <div class="param-item" data-tooltip="Whether the selector path includes a learned value baseline/critic in addition to intent logits.">
              <span class="param-name">Selector heads:</span>
              <span class="param-value">{{ selectorHeadSummary }}</span>
            </div>
            <div class="param-item" data-tooltip="Selector decision points. With multiselect off, mu chooses one play at offense possession start. With multiselect on, a completed pass can trigger reselection later in the same possession once the minimum play length is satisfied.">
              <span class="param-name">Decision boundary:</span>
              <span class="param-value">{{ selectorDecisionBoundaryLabel }}</span>
            </div>
            <div class="param-item" data-tooltip="Which selector implementation path trained this checkpoint.">
              <span class="param-name">Integration:</span>
              <span class="param-value">{{ selectorImplementationLabel }}</span>
            </div>
          </div>

          <div class="param-category">
            <h5>&mu; Selector Mechanics</h5>
            <div class="param-item" data-tooltip="mu does not fully replace uniform intent sampling immediately. It mixes between uniform play sampling and selector-driven play choice using alpha.">
              <span class="param-name">Alpha schedule:</span>
              <span class="param-value">{{ selectorAlphaSummary }}</span>
            </div>
            <div class="param-item" data-tooltip="Timesteps spent before selector usage begins, followed by the selector ramp window.">
              <span class="param-name">Schedule:</span>
              <span class="param-value">{{ selectorScheduleSummary }}</span>
            </div>
            <div class="param-item" data-tooltip="Uniform exploration floor mixed into selector sampling when the selector branch is active. Larger epsilon keeps broad play exploration alive even if raw selector logits start to collapse.">
              <span class="param-name">Selector eps:</span>
              <span class="param-value">{{ selectorEpsSummary }}</span>
            </div>
            <div class="param-item" data-tooltip="Timesteps spent before ramping the selector's internal uniform-exploration floor, then the ramp window from eps-start to eps-end.">
              <span class="param-name">Eps schedule:</span>
              <span class="param-value">{{ selectorEpsScheduleSummary }}</span>
            </div>
            <div class="param-item" data-tooltip="Entropy regularization on the selector distribution over intents. This is distinct from low-level PPO action entropy.">
              <span class="param-name">Selector entropy coef:</span>
              <span class="param-value">{{ selectorTrainingParams.intent_selector_entropy_coef ?? 'N/A' }}</span>
            </div>
            <div class="param-item" data-tooltip="KL-to-uniform regularization on average selector usage to prevent early play collapse.">
              <span class="param-name">Usage reg coef:</span>
              <span class="param-value">{{ selectorTrainingParams.intent_selector_usage_reg_coef ?? 'N/A' }}</span>
            </div>
            <div class="param-item" data-tooltip="Whether the selector is allowed to choose a new play again within the same possession. When disabled, one selected play remains fixed for the full possession.">
              <span class="param-name">Multiselect:</span>
              <span class="param-value">{{ selectorMultiselectEnabled ? '✓ Enabled' : '✗ Disabled' }}</span>
            </div>
            <div class="param-item" data-tooltip="Minimum segment length before a completed pass can trigger selector reselection. This only matters when multiselect is enabled.">
              <span class="param-name">Min play steps:</span>
              <span class="param-value">{{ selectorMinPlayStepsSummary }}</span>
            </div>
            <div class="param-item" data-tooltip="Weight on the selector value-loss term in the integrated selector critic path.">
              <span class="param-name">Value coef:</span>
              <span class="param-value">{{ selectorValueCoef ?? 'N/A' }}</span>
            </div>
            <div class="param-item" data-tooltip="How selector updates are optimized in this checkpoint.">
              <span class="param-name">Objective:</span>
              <span class="param-value">{{ selectorObjectiveLabel }}</span>
            </div>
            <div class="param-item" data-tooltip="Current integrated selector critic still trains on full-possession return; segment-return reselection is a later planned extension.">
              <span class="param-name">Credit target:</span>
              <span class="param-value">{{ selectorCreditAssignmentLabel }}</span>
            </div>
            <div class="param-item" data-tooltip="The action policy still produces low-level pi(a|s,z). The selector chooses z only at its configured decision boundaries; the low-level policy then executes under that active play id.">
              <span class="param-name">Low-level policy:</span>
              <span class="param-value">pi(a|s,z) under active z</span>
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

              <!-- Pressure Exposure -->
              <tr class="group-pressure-exposure">
                <td class="group-label">Pressure Exposure</td>
                <td>-</td>
                <td class="value-mono">{{ pressureExposureObsValue.toFixed(4) }}</td>
                <td class="notes">Cumulative defender-pressure turnover probability</td>
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
                <div class="token-attn-note" v-if="tokenAttentionRuntimeSummary">
                  {{ tokenAttentionRuntimeSummary }}
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
    </Teleport>
  </div>
</template>

<style scoped>
.player-controls-container {
  display: flex;
  flex-direction: column;
  height: 100%;
  min-height: 400px;
  padding: 0.2rem 0.4rem;
  gap: 0.9rem;
}

.player-controls-container > h3 {
  margin: 0;
  color: var(--app-accent);
  letter-spacing: 0.08em;
  text-transform: uppercase;
  font-size: 0.95rem;
}

.turn-controls-panel {
  display: flex;
  flex-direction: column;
  gap: 0.85rem;
  margin-bottom: 1rem;
  border: 1px solid rgba(148, 163, 184, 0.24);
  border-radius: 14px;
  padding: 0.75rem;
  background: rgba(15, 23, 42, 0.2);
}

.tabs-content-shell {
  display: flex;
  flex-direction: column;
  min-width: 0;
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
  cursor: grab;
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

.tab-navigation button.dragging {
  opacity: 0.6;
  cursor: grabbing;
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
  margin-bottom: 0;
}

.player-tabs button {
  border-radius: 999px;
  border: 1px solid rgba(148, 163, 184, 0.4);
  background: transparent;
  color: var(--app-text);
  padding: 0.4rem 0.85rem;
  font-size: 0.85rem;
  letter-spacing: 0.02em;
  cursor: pointer;
  transition: border-color 0.18s ease, color 0.18s ease, box-shadow 0.18s ease;
}

.player-tabs button:hover:not(:disabled) {
  border-color: var(--app-accent-strong);
  color: var(--app-accent);
  box-shadow: 0 0 12px rgba(56, 189, 248, 0.16);
}

.player-tabs button.active {
  background-color: transparent;
  color: var(--app-accent);
  border-color: var(--app-accent-strong);
  box-shadow: 0 0 14px rgba(56, 189, 248, 0.2);
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

/* entropy debug removed */

.entropy-table tr:nth-child(even) td {
  background: rgba(255, 255, 255, 0.02);
}

.entropy-table td:last-child {
  font-family: 'Courier New', monospace;
  color: var(--app-accent);
}

.selector-intent-card {
  margin-bottom: 1rem;
}

.probability-bar-row,
.selector-intent-row {
  --selector-prob-width: 0%;
  background-image: linear-gradient(90deg, rgba(56, 189, 248, 0.24), rgba(56, 189, 248, 0.24));
  background-repeat: no-repeat;
  background-size: var(--selector-prob-width) 100%;
  transition: background-size 160ms ease-out;
}

.probability-bar-row td,
.selector-intent-row td {
  position: relative;
  z-index: 1;
  background: transparent;
  text-shadow: 0 1px 0 rgba(2, 6, 23, 0.8);
}

.probability-bar-row:hover td,
.selector-intent-row:hover td {
  background: rgba(148, 163, 184, 0.06);
}

.current-intent-row {
  background-color: rgba(244, 114, 182, 0.06);
  background-image: linear-gradient(90deg, rgba(244, 114, 182, 0.24), rgba(244, 114, 182, 0.24));
  background-repeat: no-repeat;
  background-size: var(--selector-prob-width) 100%;
}

.current-intent-row td {
  font-weight: 600;
  font-size: larger;
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
  gap: 0.8rem;
  padding: 0;
  background: transparent;
  border: none;
  width: 100%;
}

.pointer-pass-controls {
  width: 100%;
  max-width: 100%;
}

.pointer-pass-label {
  margin: 0 0 0.4rem;
  font-size: 0.76rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--app-text-muted);
}

.pointer-pass-note {
  margin: 0;
  font-size: 0.76rem;
  color: var(--app-text-muted);
}

.pointer-pass-buttons {
  display: flex;
  flex-wrap: wrap;
  gap: 0.45rem;
}

.pointer-pass-button {
  border: 1px solid rgba(148, 163, 184, 0.4);
  border-radius: 999px;
  background: transparent;
  color: var(--app-text);
  padding: 0.32rem 0.72rem;
  font-size: 0.76rem;
  letter-spacing: 0.02em;
  cursor: pointer;
  transition: border-color 0.16s ease, color 0.16s ease, box-shadow 0.16s ease;
}

.pointer-pass-button:hover:not(:disabled) {
  border-color: var(--app-accent-strong);
  color: var(--app-accent);
  box-shadow: 0 0 10px rgba(56, 189, 248, 0.14);
}

.pointer-pass-button.selected {
  border-color: var(--app-accent-strong);
  background: transparent;
  color: var(--app-accent);
  box-shadow: 0 0 12px rgba(56, 189, 248, 0.2);
}

/* Fail-safe: when no pass is selected, never render highlighted target buttons. */
.pointer-pass-controls:not(.has-pass-selection) .pointer-pass-button.selected {
  border-color: rgba(148, 163, 184, 0.4);
  background: transparent;
  color: var(--app-text);
  box-shadow: none;
}

.pointer-pass-button:disabled {
  opacity: 0.55;
  cursor: not-allowed;
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

.start-template-preview {
  margin-top: 0.85rem;
  display: flex;
  flex-direction: column;
  gap: 0.45rem;
}

.start-template-preview-svg {
  width: 100%;
  max-width: 360px;
  background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
  border: 1px solid rgba(226, 232, 240, 0.16);
  border-radius: 10px;
  box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.03);
}

.start-template-preview-board {
  fill: transparent;
  stroke: rgba(248, 250, 252, 0.92);
  stroke-width: 2;
}

.start-template-preview-line {
  fill: none;
  stroke: rgba(248, 250, 252, 0.92);
  stroke-width: 2;
  stroke-linecap: round;
  stroke-linejoin: round;
}

.start-template-preview-marker {
  fill: rgba(248, 250, 252, 0.96);
  font-family: "Courier New", monospace;
  font-size: 16px;
  font-weight: 700;
  letter-spacing: 0.04em;
}

.start-template-preview-marker.team-defense {
  fill: rgba(226, 232, 240, 0.82);
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

.category-help {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  margin-left: 0.45rem;
  width: 15px;
  height: 15px;
  font-size: 0.68rem;
  line-height: 1;
  border-radius: 50%;
  border: 1px solid rgba(56, 189, 248, 0.45);
  background: rgba(56, 189, 248, 0.16);
  color: var(--app-accent);
  text-transform: none;
  cursor: help;
  vertical-align: middle;
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

.env-param-input {
  width: 110px;
  padding: 0.3rem 0.5rem;
  border: 1px solid var(--app-panel-border);
  border-radius: 6px;
  background: rgba(13, 20, 38, 0.85);
  color: var(--app-text);
  font-size: 0.85rem;
  text-align: right;
}

.env-param-input:disabled {
  opacity: 0.55;
  cursor: not-allowed;
}

.env-checkbox {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  color: var(--app-text);
  font-size: 0.85rem;
  cursor: pointer;
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
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
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
  flex-wrap: wrap;
  justify-content: center;
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
  margin-bottom: 0;
  flex-wrap: wrap;
}

.ball-holder-row label {
  font-size: 0.76rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--app-text-muted);
}

.ball-holder-row select {
  background: rgba(13, 20, 38, 0.85);
  border: 1px solid rgba(56, 189, 248, 0.35);
  color: var(--app-text);
  border-radius: 999px;
  padding: 0.28rem 0.6rem;
  font-size: 0.78rem;
}

.ball-holder-row select:focus {
  outline: none;
  border-color: var(--app-accent-strong);
  box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.2);
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
.playbook-controls .eval-row {
  display: flex;
  gap: 0.75rem;
  align-items: center;
  flex-wrap: wrap;
  margin-bottom: 0.75rem;
}
.playbook-controls .inline-label {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
}
.playbook-results-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
  gap: 1rem;
  margin-top: 1rem;
}
.playbook-summary-table-wrap {
  margin-top: 0.9rem;
  overflow-x: auto;
  border: 1px solid rgba(148, 163, 184, 0.2);
  border-radius: 12px;
  background: rgba(15, 23, 42, 0.52);
}
.playbook-summary-table {
  width: 100%;
  min-width: 1120px;
  border-collapse: collapse;
}
.playbook-summary-table th,
.playbook-summary-table td {
  padding: 0.65rem 0.75rem;
  border-bottom: 1px solid rgba(148, 163, 184, 0.16);
  text-align: left;
  vertical-align: top;
  color: #cbd5e1;
  font-size: 0.9rem;
  line-height: 1.4;
}
.playbook-summary-table th {
  position: sticky;
  top: 0;
  z-index: 1;
  background: rgba(15, 23, 42, 0.96);
  color: #94a3b8;
  font-size: 0.78rem;
  font-weight: 700;
  letter-spacing: 0.03em;
  text-transform: uppercase;
  white-space: nowrap;
}
.playbook-summary-table tbody tr:last-child td {
  border-bottom: none;
}
.playbook-summary-table tbody tr:hover td {
  background: rgba(30, 41, 59, 0.42);
}
.playbook-summary-play-cell {
  color: #38bdf8 !important;
  font-weight: 700;
  white-space: nowrap;
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
.eval-progress-wrap {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  min-width: 0;
}
.eval-progress-bar {
  width: 180px;
  height: 6px;
  background: rgba(15, 23, 42, 0.7);
  border-radius: 999px;
  border: 1px solid rgba(148, 163, 184, 0.35);
  overflow: hidden;
  flex: 0 0 auto;
}
.eval-progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #22c55e, #38bdf8);
}
.eval-progress-fill.indeterminate {
  animation: indeterminate-progress 1.5s linear infinite;
}
.eval-status {
  color: #38bdf8;
}
@keyframes indeterminate-progress {
  0% {
    transform: translateX(-100%);
    width: 35%;
  }
  50% {
    transform: translateX(150%);
  }
  100% {
    transform: translateX(-100%);
    width: 35%;
  }
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
.template-player-grid {
  gap: 0.55rem;
}
.template-library-shell {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}
.template-library-toolbar,
.template-library-picker-card,
.template-editor-shell {
  border: 1px solid rgba(148, 163, 184, 0.2);
  border-radius: 12px;
  padding: 0.85rem;
  background: rgba(15, 23, 42, 0.5);
}
.template-library-toolbar {
  display: flex;
  flex-direction: column;
  gap: 0.6rem;
}
.template-library-path-row {
  align-items: center;
}
.template-file-input-hidden {
  display: none;
}
.template-path-input {
  flex: 1 1 320px;
  min-width: 220px;
}
.template-library-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;
}
.template-library-dirty {
  color: #fbbf24;
}
.template-help {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  margin-left: 0.35rem;
  width: 15px;
  height: 15px;
  border-radius: 50%;
  border: 1px solid rgba(56, 189, 248, 0.45);
  background: rgba(56, 189, 248, 0.16);
  color: var(--app-accent);
  font-size: 0.68rem;
  font-weight: 700;
  line-height: 1;
  cursor: help;
  position: relative;
  vertical-align: middle;
}
.template-help[data-tooltip]::before {
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
.template-help[data-tooltip]::after {
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
.template-help[data-tooltip]:hover::before,
.template-help[data-tooltip]:hover::after {
  opacity: 1;
  visibility: visible;
}
.template-library-chip-list {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
}
.template-chip-btn {
  border: 1px solid rgba(148, 163, 184, 0.35);
  border-radius: 999px;
  padding: 0.32rem 0.75rem;
  background: rgba(15, 23, 42, 0.45);
  color: #cbd5e1;
  cursor: pointer;
  transition: border-color 0.16s ease, color 0.16s ease, background 0.16s ease;
}
.template-chip-btn:hover {
  border-color: #7dd3fc;
  color: #f8fafc;
}
.template-chip-btn.active {
  border-color: #38bdf8;
  background: rgba(56, 189, 248, 0.14);
  color: #38bdf8;
}
.template-grid-header,
.template-grid-row {
  grid-template-columns: 0.9fr 0.9fr 0.6fr 1fr 0.8fr 1.2fr;
}
.template-grid-row span {
  color: #e2e8f0;
}
.template-anchor-inputs {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 0.35rem;
}
.template-anchor-inputs input {
  width: 100%;
}
.template-export-actions {
  justify-content: flex-end;
}
.template-export-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 0.9rem;
}
.template-export-card {
  border: 1px solid rgba(148, 163, 184, 0.2);
  border-radius: 12px;
  padding: 0.75rem;
  background: rgba(15, 23, 42, 0.5);
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}
.template-export-card h5 {
  margin: 0;
  color: #cbd5e1;
}
.template-export-textarea {
  width: 100%;
  min-height: 260px;
  resize: vertical;
  border-radius: 10px;
  border: 1px solid rgba(148, 163, 184, 0.28);
  background: rgba(2, 6, 23, 0.72);
  color: #e2e8f0;
  padding: 0.75rem;
  font-family: 'JetBrains Mono', 'Courier New', monospace;
  font-size: 0.82rem;
  line-height: 1.45;
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
.ghost-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
.template-danger-btn {
  color: #fda4af;
  border-color: rgba(244, 63, 94, 0.35);
}
.template-danger-btn:hover:not(:disabled) {
  border-color: #fb7185;
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
