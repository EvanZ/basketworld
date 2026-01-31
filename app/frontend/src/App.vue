<script setup>
import { ref, computed, watch, onMounted, onBeforeUnmount, nextTick } from 'vue';
import GameSetup from './components/GameSetup.vue';
import GameBoard from './components/GameBoard.vue';
import PlayerControls from './components/PlayerControls.vue';
import { ref as vueRef } from 'vue';
import KeyboardLegend from './components/KeyboardLegend.vue';
import { initGame, stepGame, getPolicyProbs, saveEpisode, saveEpisodeFromPngs, startSelfPlay, replayLastEpisode, getPhiParams, setPhiParams, runEvaluation, getPassStealProbabilities, getStateValues, updatePlayerPosition, setShotClock, resetTurnState, swapPolicies, listPolicies, previewPassSteal } from './services/api';
import { resetStatsStorage } from './services/stats';

function cloneState(state) {
  return state ? JSON.parse(JSON.stringify(state)) : null;
}

function normalizeShotAccumulator(acc) {
  const result = {};
  if (!acc || typeof acc !== 'object') return result;
  for (const [key, val] of Object.entries(acc)) {
    if (!Array.isArray(val) || val.length < 2) continue;
    result[key] = [Number(val[0]) || 0, Number(val[1]) || 0];
  }
  return result;
}

function probToPercent(prob) {
  const num = Number(prob ?? 0);
  if (Number.isNaN(num)) return 0;
  return Math.max(0, Math.min(100, Number((num * 100).toFixed(1))));
}

function percentToProbClamp(percent) {
  const num = Number(percent ?? 0);
  if (Number.isNaN(num)) return 0.01;
  const clamped = Math.max(1, Math.min(99, num));
  return clamped / 100;
}

const gameState = ref(null);      // For current state and UI logic
const gameHistory = ref([]);     // For ghost trails
const policyProbs = ref(null);   // For AI suggestions
const isLoading = ref(false);
const error = ref(null);
const initialSetup = ref(null);
const activePlayerId = ref(null);
const shotAccumulator = ref({});
const shotChartTarget = ref('team');
const shotAccumulatorForBoard = computed(() => {
  if (shotChartTarget.value === 'team' || !shotChartTarget.value) {
    return shotAccumulator.value || {};
  }
  const pid = Number(shotChartTarget.value);
  const stats = perPlayerEvalStats.value?.[pid] || perPlayerEvalStats.value?.[String(pid)];
  if (!stats || !stats.shot_chart) return shotAccumulator.value || {};
  const converted = {};
  for (const [key, vals] of Object.entries(stats.shot_chart || {})) {
    if (!Array.isArray(vals) || vals.length < 2) continue;
    const [att, mk] = vals;
    converted[key] = [Number(att) || 0, Number(mk) || 0];
  }
  return converted;
});
// Reflect the actions being applied on each step to keep UI tabs in sync during self-play
const currentSelections = ref(null);
// Track user-selected actions for display on game board
const userSelections = ref({});
const isSelfPlaying = ref(false);
const canReplay = ref(false);
const isReplaying = ref(false);
// For manual stepping through replay
const replayStates = ref([]);
const currentStepIndex = ref(0);
const isManualStepping = ref(false);
// Force remount of PlayerControls to clear internal state between games
const controlsKey = ref(0);
const controlsRef = ref(null);
const gameBoardRef = ref(null);
const phiRef = vueRef(null);

// AI Mode for pre-selecting actions, not automatic play
const aiMode = ref(true);
const playerDeterministic = ref(false);
const opponentDeterministic = ref(true);

// Shared move tracking between manual and AI play
const moveHistory = ref([]);

// Track whether a shot-clock adjustment request is in flight
const isShotClockUpdating = ref(false);
const isPolicySwapping = ref(false);
const policyOptions = ref([]);
const policiesLoading = ref(false);
const policyLoadError = ref(null);
const mctsOptionsForStep = ref(null);
const lastMctsResults = ref(null);
const persistUseMcts = ref(false);
const isMctsStepRunning = ref(false);
const isStepRunning = ref(false);
const isBallHolderUpdating = ref(false);

async function refreshPolicyOptions(runId) {
  if (!runId) {
    policyOptions.value = [];
    policyLoadError.value = null;
    policiesLoading.value = false;
    return;
  }

  policiesLoading.value = true;
  policyLoadError.value = null;

  try {
    const result = await listPolicies(runId);
    policyOptions.value = Array.isArray(result?.unified) ? result.unified : [];
  } catch (err) {
    policyOptions.value = [];
    policyLoadError.value = err?.message || 'Failed to load policies';
  } finally {
    policiesLoading.value = false;
  }
}

// Current shot clock for highlighting in moves table
const currentShotClock = computed(() => {
  if (!gameState.value || gameState.value.shot_clock === undefined) {
    return null;
  }
  // Highlight the row with actions for the CURRENT board state
  // Board shows post-action, so highlight matching board value
  // (which represents the next actions to be taken)
  return gameState.value.shot_clock;
});

const pressureExposure = computed(() => {
  let total = 0;
  for (const move of moveHistory.value) {
    if (!move || move.isEndRow) continue;
    const pressure = move.actionResults?.defender_pressure;
    if (!pressure) continue;
    for (const info of Object.values(pressure)) {
      const val = Number(info?.total_pressure_prob);
      if (Number.isFinite(val)) {
        total += val;
      }
    }
  }
  return total;
});

// Evaluation mode state
const isEvaluating = ref(false);
const evalNumEpisodes = ref(100);
const defaultEvalConfig = () => ({
  mode: 'default',
  placementEditing: false,
  positions: [],
  ballHolder: null,
  shootingMode: 'random',
  skills: { layup: [], three_pt: [], dunk: [] },
  randomizeOffensePermutation: false,
});
const evalConfig = ref(defaultEvalConfig());
const perPlayerEvalStats = ref({});
const placementPassPreview = ref({});
const isEvalPlacementMode = computed(() => evalConfig.value.mode === 'custom');
const activeControlsTab = ref('controls');
const shotChartOptions = computed(() => {
  const opts = [{ label: 'Team', value: 'team' }];
  const offenseIds = gameState.value?.offense_ids || [];
  if (Array.isArray(offenseIds)) {
    offenseIds.forEach((pid) => opts.push({ label: `Player ${pid}`, value: String(pid) }));
  }
  return opts;
});
const showShotChartSelector = computed(() => !isEvalPlacementMode.value && !!shotAccumulator.value && Object.keys(shotAccumulator.value).length > 0);

const FLASH_CAPTURE_OFFSETS_MS = [0, 120, 240, 360, 480, 600, 720, 840];
const MOVE_CAPTURE_OFFSETS_MS = [0, 90, 180, 260];
const moveTransitionMs = computed(() => Math.max(120, gifStepDurationMs.value || BASE_STEP_DURATION_MS));
const disableTransitionsForCapture = ref(false);
const moveProgressForCapture = ref(1);
const BASE_STEP_DURATION_MS = 900;
const gifStepDurationMs = ref(BASE_STEP_DURATION_MS);

// Watch for when episodes end to stop auto-play behavior
watch(gameState, async (newState, oldState) => {
    if (newState?.policy_probabilities) {
        policyProbs.value = newState.policy_probabilities;
    }

    // Only fetch policy probs if NOT in manual stepping mode (to avoid overwriting stored historical probs)
    if (newState && !newState.done && !isManualStepping.value && !isReplaying.value) {
        try {
            console.log('[App] Fetching policy probs from API (not in manual stepping mode)');
            const response = await getPolicyProbs();
            policyProbs.value = response;
        } catch (err) {
            console.error('[App] Failed to fetch policy probs:', err);
        }
    } else if (newState && !newState.done && isManualStepping.value) {
        console.log('[App] Skipping policy probs fetch - in manual stepping mode, using stored probs');
    } else if (newState && !newState.done && isReplaying.value) {
        console.log('[App] Skipping policy probs fetch during replay - using stored probabilities in snapshots');
    }
    // When an episode ends, disable AI mode to allow starting a new game
    if (newState && newState.done && (!oldState || !oldState.done)) {
        aiMode.value = true;
        // Allow replay after any completed episode (manual or self-play)
        canReplay.value = true;
        // Automatically load episode for manual stepping (with small delay and error handling)
        setTimeout(async () => {
            try {
                await handleManualReplay();
            } catch (err) {
                console.error('[App] Failed to auto-load episode for stepping:', err);
                // Don't show alert here since it's automatic - just log the error
            }
        }, 100);
    }
});

async function handleGameStarted(setupData) {
  console.log('[App] Starting game with data:', setupData);
  
  // Preserve phi shaping parameters (always try to save them, not just when gameState exists)
  let savedPhiParams = null;
  try {
    savedPhiParams = await getPhiParams();
    console.log('[App] Saved phi shaping params for new game:', savedPhiParams);
  } catch (err) {
    console.warn('[App] Could not save phi params:', err);
  }
  
  // Clear move history for new game
  moveHistory.value = [];
  shotAccumulator.value = {};
  // Clear any self-play selections state
  currentSelections.value = null;
  userSelections.value = {};
  isSelfPlaying.value = false;
  // Bump key to reset PlayerControls' internal state (like selectedActions)
  controlsKey.value += 1;
  // Reset replay availability for a fresh episode
  canReplay.value = false;
  // Reset manual stepping state
  isManualStepping.value = false;
  replayStates.value = [];
  currentStepIndex.value = 0;
  
  // Start loading BEFORE clearing state to avoid interim UI flicker
  isLoading.value = true;
  error.value = null;
  policyProbs.value = null;
  lastMctsResults.value = null;
  initialSetup.value = setupData;
  // Keep board hidden but avoid flashing setup screen when we already have a setup
  gameState.value = null;      // Ensure old board is cleared
  gameHistory.value = [];      // Clear history
  activePlayerId.value = null;
  // Clear options; PlayerControls will re-emit with the persisted toggle state
  mctsOptionsForStep.value = null;
  try {
    const response = await initGame(
      setupData.runId,
      setupData.userTeam,
      setupData.offensePolicyName,
      setupData.defensePolicyName,
      setupData.unifiedPolicyName ?? null,
      setupData.opponentUnifiedPolicyName ?? null,
    );
    
    // Restore phi shaping parameters if we had them
    if (savedPhiParams && response.status === 'success') {
      try {
        await setPhiParams(savedPhiParams);
        console.log('[App] Restored phi shaping params:', savedPhiParams);
      } catch (err) {
        console.warn('[App] Could not restore phi params:', err);
      }
    }
    if (response.status === 'success') {
      gameState.value = response.state;
      gameHistory.value.push(cloneState(response.state));
      refreshPolicyOptions(response.state.run_id);
      // Don't create an initial row - wait for first actions
    } else {
      throw new Error(response.message || 'Failed to start game.');
    }
  } catch (err) {
    error.value = err.message;
    console.error(err);
  } finally {
    isLoading.value = false;
  }
}

async function handleActionsSubmitted(actions) {
  if (!gameState.value) return;
  if (isStepRunning.value) {
    console.warn('[App] Step already in progress, ignoring submit.');
    return;
  }
  if (isBallHolderUpdating.value) {
    console.warn('[App] Ball handler update in progress, ignoring submit.');
    return;
  }
  isStepRunning.value = true;
  const shouldUseMcts = !!(mctsOptionsForStep.value && mctsOptionsForStep.value.use_mcts);
  // No loading indicator for steps, feels more responsive
  try {
    if (shouldUseMcts) {
      isMctsStepRunning.value = true;
    }
    const response = await stepGame(actions, playerDeterministic.value, opponentDeterministic.value, mctsOptionsForStep.value);
     if (response.status === 'success') {
      gameState.value = response.state;
      gameHistory.value.push(cloneState(response.state));
      if (response.state?.policy_probabilities) {
        policyProbs.value = response.state.policy_probabilities;
      }
      lastMctsResults.value = response.mcts || null;
      
        // Update the last move with action results, shot clock, and state values BEFORE action
        if (moveHistory.value.length > 0) {
          const lastMove = moveHistory.value[moveHistory.value.length - 1];
          if (response.state.last_action_results) {
            lastMove.actionResults = response.state.last_action_results;
          }
          // Track which players were controlled by MCTS for this step
          if (response.mcts && typeof response.mcts === 'object') {
            lastMove.mctsPlayers = Object.keys(response.mcts).map(k => Number(k));
          } else if (!lastMove.mctsPlayers) {
            lastMove.mctsPlayers = [];
          }
          if ((!lastMove.mctsPlayers || lastMove.mctsPlayers.length === 0) && shouldUseMcts && mctsOptionsForStep.value) {
            const opt = mctsOptionsForStep.value;
            const players = Array.isArray(opt.player_ids) ? opt.player_ids : [];
            if (players.length > 0) {
              lastMove.mctsPlayers = players.map(p => Number(p));
            } else if (opt.player_id !== null && opt.player_id !== undefined) {
              lastMove.mctsPlayers = [Number(opt.player_id)];
            } else if (response.state?.ball_holder !== undefined && response.state?.ball_holder !== null) {
              lastMove.mctsPlayers = [Number(response.state.ball_holder)];
            }
          }
          // Update moves with actual actions taken (including opponent)
          if (response.actions_taken) {
              console.log('[App] Actions taken:', response.actions_taken);
              // Create a new object for reactivity
              const newMoves = { ...lastMove.moves };
              for (const [pid, actionName] of Object.entries(response.actions_taken)) {
                  newMoves[`Player ${pid}`] = actionName;
              }
              // Replace the whole object
              lastMove.moves = newMoves;
          }
          // Store shot clock when action was decided (before execution): board + 1
          if (response.state.shot_clock !== undefined) {
            lastMove.shotClock = response.state.shot_clock + 1;
          }
          // Store the value of the observation that is now visible on the board
          const applyStateValues = (values) => {
            if (!values) return;
            if (lastMove.offensiveValue === null || lastMove.offensiveValue === undefined) {
              lastMove.offensiveValue = values.offensive_value;
            }
            if (lastMove.defensiveValue === null || lastMove.defensiveValue === undefined) {
              lastMove.defensiveValue = values.defensive_value;
            }
          };

          applyStateValues(response.state?.state_values);
          if ((lastMove.offensiveValue === null || lastMove.offensiveValue === undefined) && response.pre_step_state_values) {
            applyStateValues(response.pre_step_state_values);
          }
        }
        
        // If game is done, add an END row
        if (response.state.done && moveHistory.value.length > 0) {
          const allIds = [...(response.state.offense_ids || []), ...(response.state.defense_ids || [])];
          const endMoves = {};
          allIds.forEach(pid => {
            endMoves[`Player ${pid}`] = 'END';
          });
          moveHistory.value.push({
            turn: moveHistory.value.length + 1,
            moves: endMoves,
            shotClock: response.state.shot_clock,
            isEndRow: true,
            offensiveValue: response.state?.state_values?.offensive_value ?? null,
            defensiveValue: response.state?.state_values?.defensive_value ?? null,
          });
        }
      
      try { controlsRef.value?.$refs?.phiRef?.refresh?.(); } catch (_) {}
    } else {
      throw new Error(response.message || 'Failed to process step.');
    }
  } catch (err) {
    error.value = err.message;
    console.error(err);
  } finally {
    isMctsStepRunning.value = false;
    isStepRunning.value = false;
  }
}

async function handleResetPositions() {
  if (!gameState.value) return;
  if (isShotClockUpdating.value) {
    console.log('[App] Reset already in progress.');
    return;
  }

  isShotClockUpdating.value = true;

  try {
    console.log('[App] Resetting turn via backend snapshot');
    const response = await resetTurnState();
    if (response.status !== 'success') {
      throw new Error(response.message || 'Failed to reset turn.');
    }

    gameState.value = response.state;
    if (gameHistory.value.length > 0) {
      gameHistory.value[gameHistory.value.length - 1] = cloneState(response.state);
    }

    const probs = await getPolicyProbs();
    policyProbs.value = probs;
  } catch (err) {
    console.error('[App] Failed to reset turn via backend:', err);
    alert(`Failed to reset turn: ${err.message}`);
  } finally {
    isShotClockUpdating.value = false;
  }
}

async function handlePlayerPositionUpdate({ playerId, q, r }) {
  if (!gameState.value || gameState.value.done) {
    console.warn('[App] Cannot update position: game not active or done.');
    return;
  }
  
  try {
    console.log(`[App] Updating position for Player ${playerId} to (${q}, ${r})`);
    const response = await updatePlayerPosition(playerId, q, r);
    if (response.status === 'success') {
      // Update the current game state
      gameState.value = response.state;
      
      // Update the history for the CURRENT step (so ghost trails and replay show the new position)
      // Since we haven't stepped yet, we are modifying the state "before" the step.
      // If we just append, it looks like a step. If we replace, it looks like a correction.
      // The user intent is "re-calculate... as if the step started with the player in that position".
      // So we should update the last entry in gameHistory.
      if (gameHistory.value.length > 0) {
        gameHistory.value[gameHistory.value.length - 1] = cloneState(response.state);
      }
      
      // Also refresh policy probs since state changed
      const probs = await getPolicyProbs();
      policyProbs.value = probs;
    }
  } catch (err) {
    console.error('[App] Failed to update player position:', err);
    alert(`Failed to move player: ${err.message}`);
  }
}

function handlePatchedGameState(newState) {
  if (!newState) return;
  gameState.value = newState;
  const clonedState = cloneState(newState);
  if (gameHistory.value.length > 0) {
    gameHistory.value[gameHistory.value.length - 1] = clonedState;
  } else {
    gameHistory.value = [clonedState];
  }
  if (newState.policy_probabilities) {
    policyProbs.value = { ...newState.policy_probabilities };
  }
}

async function handleShotClockAdjustment(delta) {
  if (!gameState.value || gameState.value.done) {
    console.warn('[App] Cannot adjust shot clock after episode has ended.');
    return;
  }
  if (isShotClockUpdating.value) {
    console.log('[App] Shot-clock adjustment already in progress, ignoring extra request.');
    return;
  }

  isShotClockUpdating.value = true;

  try {
    const response = await setShotClock(delta);
    if (response.status === 'success') {
      gameState.value = response.state;
      if (gameHistory.value.length > 0) {
        gameHistory.value[gameHistory.value.length - 1] = cloneState(response.state);
      }
      const probs = await getPolicyProbs();
      policyProbs.value = probs;
    }
  } catch (err) {
    console.error('[App] Failed to adjust shot clock:', err);
    alert(`Failed to adjust shot clock: ${err.message}`);
  } finally {
    isShotClockUpdating.value = false;
  }
}

async function handlePolicySwap({ target, policyName }) {
  if (!gameState.value || !target) return;
  if (isPolicySwapping.value) {
    console.log('[App] Policy swap already in progress, ignoring new request.');
    return;
  }

  const payload = {};
  if (target === 'user') {
    payload.user_policy_name = policyName;
  } else if (target === 'opponent') {
    payload.opponent_policy_name = policyName;
  } else {
    return;
  }

  isPolicySwapping.value = true;

  try {
    const response = await swapPolicies(payload);
    if (response.status !== 'success' && response.status !== 'no_change') {
      throw new Error(response.message || 'Failed to swap policies.');
    }

    const newState = response.state;
    if (newState) {
      gameState.value = newState;
      const clonedState = cloneState(newState);
      if (gameHistory.value.length > 0) {
        gameHistory.value[gameHistory.value.length - 1] = clonedState;
      } else {
        gameHistory.value = [clonedState];
      }

      if (newState.policy_probabilities) {
        policyProbs.value = newState.policy_probabilities;
      } else {
        try {
          const probs = await getPolicyProbs();
          policyProbs.value = probs;
        } catch (err) {
          console.warn('[App] Failed to refresh policy probabilities after swap:', err);
        }
      }

      // Update initialSetup so new games use the updated policy
      if (initialSetup.value) {
        if (target === 'user') {
          initialSetup.value = {
            ...initialSetup.value,
            unifiedPolicyName: policyName,
          };
          console.log('[App] Updated initialSetup user policy to:', policyName);
        } else if (target === 'opponent') {
          initialSetup.value = {
            ...initialSetup.value,
            opponentUnifiedPolicyName: policyName || null,
          };
          console.log('[App] Updated initialSetup opponent policy to:', policyName || 'Mirror');
        }
      }
    }
  } catch (err) {
    console.error('[App] Policy swap failed:', err);
    alert(`Failed to swap policies: ${err.message}`);
  } finally {
    isPolicySwapping.value = false;
  }
}

async function handleMoveRecorded(moveData) {
  console.log('[App] Recording move:', moveData);

  const moveEntry = { ...moveData };

  // Capture pre-step state immediately so async work doesn't race the step response.
  if (gameState.value && gameState.value.ball_holder !== undefined) {
    moveEntry.ballHolder = gameState.value.ball_holder;
  }
  const stateValues = gameState.value?.state_values;
  moveEntry.offensiveValue = stateValues?.offensive_value ?? null;
  moveEntry.defensiveValue = stateValues?.defensive_value ?? null;
  if (gameState.value && gameState.value.shot_clock !== undefined) {
    moveEntry.shotClock = gameState.value.shot_clock;
  }

  // Carry forward any MCTS selection info from the controls emit (so Moves table can show icons)
  if (moveEntry.mctsPlayers && Array.isArray(moveEntry.mctsPlayers)) {
    moveEntry.mctsPlayers = moveEntry.mctsPlayers.map(p => Number(p));
  } else {
    moveEntry.mctsPlayers = [];
  }

  // Ensure the row exists before any awaits so later updates can attach to it.
  moveEntry.passStealProbabilities = {};
  moveHistory.value.push(moveEntry);

  // Fetch pass steal probabilities before the action is taken (best-effort).
  try {
    const passStealProbs = await getPassStealProbabilities();
    moveEntry.passStealProbabilities = passStealProbs;
    console.log('[App] Added pass steal probs to move:', passStealProbs);
  } catch (err) {
    console.warn('[App] Failed to fetch pass steal probs for move:', err);
    moveEntry.passStealProbabilities = {};
  }
}

// Handler for selections changed from PlayerControls
function handleSelectionsChanged(selections) {
  userSelections.value = selections;
}

// Handler for refresh policies request from PlayerControls
async function handleRefreshPolicies() {
  if (gameState.value?.run_id) {
    console.log('[App] Refreshing policies for run:', gameState.value.run_id);
    await refreshPolicyOptions(gameState.value.run_id);
    console.log('[App] Policy refresh complete, found', policyOptions.value.length, 'policies');
  }
}

function handleMctsOptionsChanged(options) {
  mctsOptionsForStep.value = options;
}

function handleMctsToggleChanged(val) {
  persistUseMcts.value = !!val;
}

function handleActiveTabChanged(tab) {
  activeControlsTab.value = tab || 'controls';
}

// Computed: which selections to show on board (user or self-play)
// During manual stepping (replay), show the stored actions_taken from the NEXT state
// (because actions_taken in state N shows what was done to GET to N, but we want to show
// what will be done FROM the current state N, which is stored in state N+1)
const boardSelectedActions = computed(() => {
  // During animated replay, show the actions that led to the current state
  if (isReplaying.value && gameState.value?.actions_taken) {
    const actions = {};
    for (const [pid, action] of Object.entries(gameState.value.actions_taken)) {
      actions[parseInt(pid, 10)] = action;
    }
    return actions;
  }
  // During replay/manual stepping, show the stored actions from the next step
  if (isManualStepping.value && replayStates.value.length > 0) {
    const nextIndex = currentStepIndex.value + 1;
    // If we're at the last state, there are no more actions to show
    if (nextIndex >= replayStates.value.length) {
      return {};
    }
    const nextState = replayStates.value[nextIndex];
    if (nextState && nextState.actions_taken) {
      // Convert string keys to number keys for player IDs
      const actions = {};
      for (const [pid, action] of Object.entries(nextState.actions_taken)) {
        actions[parseInt(pid)] = action;
      }
      return actions;
    }
    return {};
  }
  if (isSelfPlaying.value && currentSelections.value) {
    return currentSelections.value;
  }
  return userSelections.value;
});

// Handler for Self-Play button click
function handleSelfPlayButton() {
  if (activeControlsTab.value === 'eval') {
    console.log('[App] Self-play disabled in Eval tab');
    return;
  }
  // Get current selections from PlayerControls
  const preselected = controlsRef.value?.getSelectedActions?.() || null;
  handleSelfPlay(preselected);
}

// New function for self-play mode (runs full episode)
async function handleSelfPlay(preselected = null) {
  if (!gameState.value || !aiMode.value) return;
  const mctsOptions = (mctsOptionsForStep.value && mctsOptionsForStep.value.use_mcts) ? mctsOptionsForStep.value : null;
  // Start deterministic self-play on backend: snapshot seed and initial state
  try {
    const res = await startSelfPlay();
    if (res && res.status === 'success' && res.state) {
      // Reset UI to backend's reset state so trajectories align
      gameState.value = res.state;
      gameHistory.value = [res.state];
      moveHistory.value = [];
      currentSelections.value = null;
    }
  } catch (e) {
    console.error('[App] Failed to start self-play on backend:', e);
  }
  isSelfPlaying.value = true;
  currentSelections.value = null;
  
  // Run full episode with AI controlling all players
  while (gameState.value && !gameState.value.done) {
    try {
      // Get AI actions for user-controlled players
      let aiActions = {};
      const mctsTargets = new Set();
      if (mctsOptions) {
        if (Array.isArray(mctsOptions.player_ids)) {
          mctsOptions.player_ids.forEach(pid => { if (pid !== null && pid !== undefined) mctsTargets.add(Number(pid)); });
        } else if (mctsOptions.player_id !== null && mctsOptions.player_id !== undefined) {
          mctsTargets.add(Number(mctsOptions.player_id));
        } else if (gameState.value?.ball_holder !== null && gameState.value?.ball_holder !== undefined) {
          mctsTargets.add(Number(gameState.value.ball_holder));
        }
      }
      
      if (policyProbs.value) {
        // Determine which players are user-controlled
        const userControlledIds = gameState.value.user_team_name === 'OFFENSE' 
          ? gameState.value.offense_ids 
          : gameState.value.defense_ids;
        
        // Get AI actions for user-controlled players
        for (const playerId of userControlledIds) {
          // If MCTS should override this player, skip explicit action
          if (mctsTargets.has(playerId)) {
            continue;
          }
          const probs = policyProbs.value[playerId];
          const actionMask = gameState.value.action_mask[playerId];

          // If this is the first loop iteration and preselected is provided, honor it
          if (preselected && preselected[`$${playerId}`] === undefined && preselected[playerId] === undefined) {
            // normalize to string and numeric lookup
          }
          if (preselected) {
            let chosen = null;
            const preVal = preselected[playerId] ?? preselected[`$${playerId}`];
            if (typeof preVal === 'string') {
              const names = [
                "NOOP","MOVE_E","MOVE_NE","MOVE_NW","MOVE_W","MOVE_SW","MOVE_SE",
                "SHOOT","PASS_E","PASS_NE","PASS_NW","PASS_W","PASS_SW","PASS_SE"
              ];
              const idx = names.indexOf(preVal);
              if (idx >= 0 && actionMask[idx] === 1) chosen = idx;
            } else if (typeof preVal === 'number') {
              if (preVal >= 0 && preVal < actionMask.length && actionMask[preVal] === 1) chosen = preVal;
            }
            if (chosen != null) {
              aiActions[playerId] = chosen;
              continue; // do not override a valid preselection
            }
          }
          
          if (Array.isArray(probs) && Array.isArray(actionMask)) {
            let selectedActionIndex = 0;
            
            if (playerDeterministic.value) {
              // Deterministic: Pick action with highest probability (argmax) among LEGAL actions only
              let bestProb = -1;
              
              for (let i = 0; i < probs.length && i < actionMask.length; i++) {
                // Only consider legal actions (action_mask[i] === 1)
                if (actionMask[i] === 1 && probs[i] > bestProb) {
                  bestProb = probs[i];
                  selectedActionIndex = i;
                }
              }
              console.log(`[App] Self-play DETERMINISTIC action ${selectedActionIndex} for player ${playerId} (prob: ${bestProb.toFixed(3)})`);
            } else {
              // Stochastic: Sample from probability distribution over LEGAL actions
              const legalIndices = [];
              const legalProbs = [];
              
              for (let i = 0; i < probs.length && i < actionMask.length; i++) {
                if (actionMask[i] === 1) {
                  legalIndices.push(i);
                  legalProbs.push(probs[i]);
                }
              }
              
              if (legalProbs.length > 0) {
                // Normalize probabilities
                const sum = legalProbs.reduce((a, b) => a + b, 0);
                const normalizedProbs = legalProbs.map(p => p / sum);
                
                // Sample from distribution
                const rand = Math.random();
                let cumulative = 0;
                for (let i = 0; i < normalizedProbs.length; i++) {
                  cumulative += normalizedProbs[i];
                  if (rand < cumulative) {
                    selectedActionIndex = legalIndices[i];
                    break;
                  }
                }
                console.log(`[App] Self-play STOCHASTIC action ${selectedActionIndex} for player ${playerId} (prob: ${probs[selectedActionIndex].toFixed(3)})`);
              }
            }
            
            aiActions[playerId] = selectedActionIndex;
          }
        }
      }
      // Clear preselected after applying for the first step
      preselected = null;
      
      // Track moves for AI self-play (even if all players are MCTS)
      const currentTurn = moveHistory.value.length + 1;
      const teamMoves = {};
      const appliedSelections = {};
      const actionNames = [
        "NOOP", "MOVE_E", "MOVE_NE", "MOVE_NW", "MOVE_W", "MOVE_SW", "MOVE_SE", 
        "SHOOT", "PASS_E", "PASS_NE", "PASS_NW", "PASS_W", "PASS_SW", "PASS_SE"
      ];

      for (const playerId of gameState.value.offense_ids.concat(gameState.value.defense_ids)) {
        if (mctsTargets.has(playerId)) {
          teamMoves[`Player ${playerId}`] = 'MCTS';
          appliedSelections[playerId] = 'MCTS';
        } else if (aiActions[playerId] !== undefined) {
          const idx = aiActions[playerId];
          const actionName = actionNames[idx] || 'UNKNOWN';
          teamMoves[`Player ${playerId}`] = actionName;
          appliedSelections[playerId] = actionName;
        } else {
          teamMoves[`Player ${playerId}`] = 'NOOP';
          appliedSelections[playerId] = 'NOOP';
        }
      }

      currentSelections.value = appliedSelections;
      
      // Capture ball holder before the action is taken
      const ballHolder = gameState.value?.ball_holder;
      
      // Fetch pass steal probabilities before the action is taken
      let passStealProbs = {};
      try {
        passStealProbs = await getPassStealProbabilities();
      } catch (err) {
        console.warn('[App Self-play] Failed to fetch pass steal probs:', err);
      }
      
      // Push move to history (state values will be added after step response)
      const currentStateValues = gameState.value?.state_values;
      moveHistory.value.push({
        turn: currentTurn,
        moves: teamMoves,
        ballHolder: ballHolder,
        passStealProbabilities: passStealProbs,
        mctsPlayers: Array.from(mctsTargets),
        offensiveValue: currentStateValues?.offensive_value ?? null,
        defensiveValue: currentStateValues?.defensive_value ?? null
      });
      
      const response = await stepGame(aiActions, playerDeterministic.value, opponentDeterministic.value, mctsOptions);
      if (response.status === 'success') {
        gameState.value = response.state;
        gameHistory.value.push(cloneState(response.state));
        if (response.state?.policy_probabilities) {
          policyProbs.value = response.state.policy_probabilities;
        }
        
        // Update the last move with action results, shot clock, and state values BEFORE action
        if (moveHistory.value.length > 0) {
          const lastMove = moveHistory.value[moveHistory.value.length - 1];
          if (response.state.last_action_results) {
            lastMove.actionResults = response.state.last_action_results;
          }
          if (response.mcts && typeof response.mcts === 'object') {
            lastMove.mctsPlayers = Object.keys(response.mcts).map(k => Number(k));
          } else if (!lastMove.mctsPlayers || lastMove.mctsPlayers.length === 0) {
            lastMove.mctsPlayers = Array.from(mctsTargets);
          }
          // Update moves with actual actions taken (including opponent)
          if (response.actions_taken) {
              console.log('[App] Actions taken:', response.actions_taken);
              // Create a new object for reactivity
              const newMoves = { ...lastMove.moves };
              for (const [pid, actionName] of Object.entries(response.actions_taken)) {
                  newMoves[`Player ${pid}`] = actionName;
              }
              // Replace the whole object
              lastMove.moves = newMoves;
          }
          // Store shot clock when action was decided (before execution): board + 1
          if (response.state.shot_clock !== undefined) {
            lastMove.shotClock = response.state.shot_clock + 1;
          }
          const stateValues = response.state?.state_values;
          if (stateValues) {
            lastMove.offensiveValue = stateValues.offensive_value;
            lastMove.defensiveValue = stateValues.defensive_value;
          } else if (response.pre_step_state_values) {
            lastMove.offensiveValue = response.pre_step_state_values.offensive_value;
            lastMove.defensiveValue = response.pre_step_state_values.defensive_value;
          }
        }
        
        // If game is done, add an END row
        if (response.state.done && moveHistory.value.length > 0) {
          const allIds = [...(response.state.offense_ids || []), ...(response.state.defense_ids || [])];
          const endMoves = {};
          allIds.forEach(pid => {
            endMoves[`Player ${pid}`] = 'END';
          });
          moveHistory.value.push({
            turn: moveHistory.value.length + 1,
            moves: endMoves,
            shotClock: response.state.shot_clock,
            isEndRow: true
          });
        }
        try { controlsRef.value?.$refs?.phiRef?.refresh?.(); } catch (_) {}
        // Small delay to make the progression visible
        await new Promise(resolve => setTimeout(resolve, 100));
      } else {
        throw new Error(response.message || 'Failed to process step.');
      }
    } catch (err) {
      error.value = err.message;
      console.error(err);
      break;
    }
  }
  // Self-play finished
  isSelfPlaying.value = false;
  currentSelections.value = null;
  canReplay.value = true;
}

function handleEvalConfigChanged(nextConfig) {
  const base = defaultEvalConfig();
  const current = evalConfig.value || base;
  const merged = {
    ...base,
    ...current,
    ...(nextConfig || {}),
    skills: {
      ...base.skills,
      ...(current.skills || {}),
      ...(nextConfig?.skills || {}),
    },
  };
  if (merged.mode !== 'custom') {
    merged.placementEditing = false;
  }
  evalConfig.value = merged;
  if (merged.mode === 'custom') {
    refreshPlacementPreview(merged.positions, merged.ballHolder);
  }
  if (merged.mode !== 'custom') {
    shotChartTarget.value = 'team';
  }
}

function seedEvalConfigFromGameState(copySkills = true) {
  if (!gameState.value) return;
  const positions = (gameState.value.positions || []).map((pos) => [pos[0], pos[1]]);
  const base = defaultEvalConfig();
  const offenseCount = gameState.value.offense_ids?.length || 0;
  const sourceSkills =
    gameState.value.offense_shooting_pct_sampled ||
    gameState.value.offense_shooting_pct_by_player ||
    null;
  const skills = { ...base.skills };
  if (copySkills && sourceSkills) {
    skills.layup = Array.from({ length: offenseCount }, (_, idx) => probToPercent(sourceSkills?.layup?.[idx]));
    skills.three_pt = Array.from({ length: offenseCount }, (_, idx) => probToPercent(sourceSkills?.three_pt?.[idx]));
    skills.dunk = Array.from({ length: offenseCount }, (_, idx) => probToPercent(sourceSkills?.dunk?.[idx]));
  }
  evalConfig.value = {
    ...base,
    ...evalConfig.value,
    positions,
    ballHolder: gameState.value.ball_holder ?? evalConfig.value.ballHolder,
    skills: copySkills ? skills : (evalConfig.value?.skills || skills),
  };
}

function ensureEvalPositionsInitialized() {
  if (!gameState.value) return;
  const nPlayers = (gameState.value.offense_ids?.length || 0) + (gameState.value.defense_ids?.length || 0);
  if (!Array.isArray(evalConfig.value.positions) || evalConfig.value.positions.length !== nPlayers) {
    seedEvalConfigFromGameState(false);
  }
}

function handleEvalPlacementUpdate({ playerId, q, r }) {
  if (playerId === undefined || playerId === null) return;
  ensureEvalPositionsInitialized();
  const positions = Array.isArray(evalConfig.value.positions)
    ? evalConfig.value.positions.map((pos) => [pos[0], pos[1]])
    : [];
  positions[playerId] = [q, r];
  handleEvalConfigChanged({ positions });
  refreshPlacementPreview(positions, evalConfig.value.ballHolder);
}

function buildCustomEvalSetup() {
  if (!gameState.value || evalConfig.value.mode !== 'custom') return null;
  const nPlayers = (gameState.value.offense_ids?.length || 0) + (gameState.value.defense_ids?.length || 0);
  let positions = Array.isArray(evalConfig.value.positions) ? evalConfig.value.positions : [];
  if (positions.length !== nPlayers && gameState.value.positions) {
    positions = (gameState.value.positions || []).map((pos) => [pos[0], pos[1]]);
  }
  if (positions.length !== nPlayers) {
    throw new Error(`Custom setup requires ${nPlayers} positions (found ${positions.length}).`);
  }
  const offenseIds = gameState.value.offense_ids || [];
  const defaultBall = offenseIds.length ? offenseIds[0] : 0;
  const rawBh = evalConfig.value.ballHolder ?? gameState.value.ball_holder ?? defaultBall;
  const ballHolder = offenseIds.includes(rawBh) ? rawBh : defaultBall;
  const shootingMode = evalConfig.value.shootingMode || 'random';
  const payload = {
    initial_positions: positions,
    ball_holder: ballHolder,
    shooting_mode: shootingMode,
  };
  if (shootingMode === 'fixed') {
    const offenseCount = gameState.value.offense_ids?.length || 0;
    const skills = evalConfig.value.skills || {};
    const toProb = (arrKey) =>
      Array.from({ length: offenseCount }, (_, idx) => percentToProbClamp(skills?.[arrKey]?.[idx] ?? 0));
    payload.offense_skills = {
      layup: toProb('layup'),
      three_pt: toProb('three_pt'),
      dunk: toProb('dunk'),
    };
  }
  return payload;
}

async function refreshPlacementPreview(positionsOverride = null, ballHolderOverride = null) {
  try {
    if (!gameState.value || evalConfig.value.mode !== 'custom') return;
    ensureEvalPositionsInitialized();
    const positions = positionsOverride || evalConfig.value.positions;
    const ballHolder = ballHolderOverride ?? evalConfig.value.ballHolder;
    if (!Array.isArray(positions) || ballHolder === null || ballHolder === undefined) return;
    const res = await previewPassSteal(positions, ballHolder);
    placementPassPreview.value = res?.pass_steal_probabilities || {};
  } catch (err) {
    console.warn('[App] Failed to refresh placement preview:', err);
  }
}

async function handleEvaluation() {
  if (!gameState.value || isEvaluating.value) return;
  
  const numEpisodes = Math.max(1, Math.min(evalNumEpisodes.value, 1000000));
  shotAccumulator.value = {};
  perPlayerEvalStats.value = {};
  
  // Reset stats at the beginning
  console.log('[App] Resetting stats before evaluation');
  if (controlsRef.value?.resetStats) {
    controlsRef.value.resetStats();
  }
  
  // Set UI state immediately to show we're evaluating
  isEvaluating.value = true;
  error.value = null;
  
  // Give UI a chance to update before blocking on API call
  await nextTick();
  
  try {
    console.log(`[App] Starting evaluation: ${numEpisodes} episodes, playerDeterministic=${playerDeterministic.value}, opponentDeterministic=${opponentDeterministic.value}`);
    
    let customSetup = null;
    if (evalConfig.value.mode === 'custom') {
      try {
        customSetup = buildCustomEvalSetup();
      } catch (setupErr) {
        throw new Error(setupErr.message || 'Invalid custom eval setup');
      }
    }
    
    // Run evaluation on backend (this will block until all episodes complete)
    const response = await runEvaluation(
      numEpisodes,
      playerDeterministic.value,
      opponentDeterministic.value,
      customSetup,
      !!evalConfig.value.randomizeOffensePermutation
    );
    
  if (response.status === 'success' && Array.isArray(response.results)) {
    console.log(`[App] Evaluation completed: ${response.results.length} episodes - processing all results immediately`);
    
    // Process all episode results and update stats immediately
    let lastEpisodeState = null;
      
      for (const result of response.results) {
        // Save the last episode state for display
        lastEpisodeState = result.final_state;
        
        // Record stats for this episode (skip API calls for speed)
        if (controlsRef.value?.recordEpisodeStats && result.final_state?.done) {
          await controlsRef.value.recordEpisodeStats(result.final_state, true, result);
        }
      }
      
      console.log('[App] All stats recorded');
      if (!response.shot_accumulator) {
        console.warn('[App] No shot_accumulator returned from evaluation');
      }
      shotAccumulator.value = normalizeShotAccumulator(response.shot_accumulator || {});
      perPlayerEvalStats.value = response.per_player_stats || {};
      console.log('[App] Shot accumulator keys:', Object.keys(shotAccumulator.value || {}).length, 'data:', shotAccumulator.value);
      
      // Update game state to the fresh state from backend (after evaluation reset)
      // This ensures the UI shows a playable game, not the last episode's final state
      if (response.current_state) {
        console.log('[App] Updating gameState to fresh state from backend');
        gameState.value = response.current_state;
        gameHistory.value = [response.current_state];
        // Exit placement/preview mode so shot charts and live controls render normally
        evalConfig.value = defaultEvalConfig();
        placementPassPreview.value = {};
      } else if (lastEpisodeState) {
        // Fallback for older backend versions
        console.log('[App] Updating gameState to show last episode (isEvaluating still true)');
        gameState.value = lastEpisodeState;
        gameHistory.value = [lastEpisodeState];
      }
      
      // Wait for Vue to process all reactive updates (including watchers) before setting isEvaluating to false
      await nextTick();
      console.log('[App] nextTick completed, watchers should have run with isEvaluating=true');
    } else {
      throw new Error('Invalid evaluation response');
    }
  } catch (err) {
    error.value = `Evaluation failed: ${err.message}`;
    console.error('[App] Evaluation error:', err);
  } finally {
    console.log('[App] Setting isEvaluating to false');
    isEvaluating.value = false;
  }
}

function handleEvalRunRequested(payload) {
  if (payload?.numEpisodes) {
    evalNumEpisodes.value = payload.numEpisodes;
  }
  if (payload?.config) {
    handleEvalConfigChanged(payload.config);
  }
  refreshPlacementPreview(payload?.config?.positions, payload?.config?.ballHolder);
  handleEvaluation();
}

const liveButtonsDisabled = computed(() => activeControlsTab.value === 'eval');

function stateHasAnimatedFlash(state) {
  const results = state?.last_action_results;
  if (!results) return false;

  const hasShot = results.shots && Object.keys(results.shots).length > 0;
  const hasPass =
    results.passes &&
    Object.values(results.passes).some(
      (passRes) => passRes && passRes.success && typeof passRes.target === 'number'
    );

  return Boolean(hasShot || hasPass);
}

function captureOffsetsForState(state) {
  const hasFlash = stateHasAnimatedFlash(state);
  const moveEnd = moveTransitionMs.value;
  if (hasFlash) {
    const merged = [...FLASH_CAPTURE_OFFSETS_MS, ...MOVE_CAPTURE_OFFSETS_MS, moveEnd];
    return Array.from(new Set(merged)).sort((a, b) => a - b);
  }
  const merged = [...MOVE_CAPTURE_OFFSETS_MS, moveEnd];
  return Array.from(new Set(merged)).sort((a, b) => a - b);
}

function interpolateState(prevState, currState, t) {
  if (!currState) return null;
  const clampedT = Math.max(0, Math.min(1, t));
  if (!prevState || !prevState.positions || !currState.positions) {
    return currState;
  }
  const positions = currState.positions.map((pos, idx) => {
    const prev = prevState.positions?.[idx] || pos;
    const [q1, r1] = prev;
    const [q2, r2] = pos;
    return [
      q1 + (q2 - q1) * clampedT,
      r1 + (r2 - r1) * clampedT,
    ];
  });
  return {
    ...currState,
    positions,
  };
}

async function handleSaveEpisode() {
  try {
    disableTransitionsForCapture.value = true;
    moveProgressForCapture.value = 1;
    await nextTick();
    // Check if we have episode states to render
    const states = replayStates.value.length > 0 ? replayStates.value : [gameState.value];
    
    if (!states || states.length === 0) {
      alert('No episode states available to save');
      return;
    }
    
    console.log(`[handleSaveEpisode] Generating ${states.length} frames...`);
    
    // Save current step index to restore later
    const savedStepIndex = currentStepIndex.value;
    
    // Generate PNG frames for each state
    const frames = [];
    const frameDurations = [];
    for (let i = 0; i < states.length; i++) {
      try {
        // Update currentStepIndex so boardSelectedActions shows correct actions for this frame
        currentStepIndex.value = i;
        
        // Temporarily set the game history to show this state
        const tempHistory = states.slice(0, i + 1);
        gameHistory.value = tempHistory;
        
        // Wait for Vue to update the DOM
        await nextTick();
        
        // Give a small delay to ensure rendering is complete
        await new Promise(resolve => setTimeout(resolve, 60));

        const offsets = captureOffsetsForState(states[i]);
        const stepDurationMs = Math.max(200, gifStepDurationMs.value || BASE_STEP_DURATION_MS);
        const frameDurationMs = stepDurationMs / offsets.length;

        let lastOffset = 0;
        for (const offset of offsets) {
          const waitMs = Math.max(offset - lastOffset, 0);
          if (waitMs > 0) {
            // eslint-disable-next-line no-await-in-loop
            await new Promise(resolve => setTimeout(resolve, waitMs));
          }

          const prevState = i > 0 ? states[i - 1] : states[i];
          const t = moveTransitionMs.value > 0 ? Math.min(1, offset / moveTransitionMs.value) : 1;
          moveProgressForCapture.value = t;
          const interpState = interpolateState(prevState, states[i], t);
          const tempHistory = [...states.slice(0, i), interpState];
          gameHistory.value = tempHistory;
          // eslint-disable-next-line no-await-in-loop
          await nextTick();
          // Give a tiny delay for the DOM to paint
          // eslint-disable-next-line no-await-in-loop
          await new Promise(resolve => setTimeout(resolve, 10));

          // Render this state as PNG
          if (gameBoardRef.value && gameBoardRef.value.renderStateToPng) {
            // eslint-disable-next-line no-await-in-loop
            const pngDataUrl = await gameBoardRef.value.renderStateToPng();
            if (pngDataUrl) {
              frames.push(pngDataUrl);
              frameDurations.push(frameDurationMs / 1000);
            }
          }
          lastOffset = offset;
        }
      } catch (err) {
        console.error(`Failed to render frame ${i}:`, err);
      }
    }
    
    // Restore the original state
    currentStepIndex.value = savedStepIndex;
    if (replayStates.value.length > 0) {
      gameHistory.value = replayStates.value.slice(0, savedStepIndex + 1);
    }
    moveProgressForCapture.value = 1;
    
    if (frames.length === 0) {
      alert('Failed to generate any frames');
      return;
    }
    
    if (frameDurations.length !== frames.length) {
      const fallbackDur = Math.max(0.2, (gifStepDurationMs.value || BASE_STEP_DURATION_MS) / 1000);
      while (frameDurations.length < frames.length) {
        frameDurations.push(fallbackDur);
      }
      frameDurations.length = frames.length;
    }

    // Send frames (with per-frame durations) to backend to create GIF
    const res = await saveEpisodeFromPngs(frames, frameDurations, gifStepDurationMs.value);
    alert(`Episode saved to ${res.file_path}`);
  } catch (e) {
    console.error('Failed to save episode:', e);
    alert(`Failed to save episode: ${e.message}`);
  } finally {
    disableTransitionsForCapture.value = false;
    moveProgressForCapture.value = 1;
  }
}

async function handleReplay() {
  try {
    const res = await replayLastEpisode();
    if (res.status === 'success' && Array.isArray(res.states) && res.states.length > 0) {
      // Animate the replay so it is visible in the UI
      isReplaying.value = true;
      isManualStepping.value = false;
      gameHistory.value = [];
      const stepDurationMs = Math.max(200, gifStepDurationMs.value || BASE_STEP_DURATION_MS);
      for (const s of res.states) {
        gameState.value = s;
        gameHistory.value.push(s);
        // Small delay between frames aligned to slider
        // eslint-disable-next-line no-await-in-loop
        await new Promise(resolve => setTimeout(resolve, stepDurationMs));
      }
      isReplaying.value = false;
      // After animated replay, reload for manual stepping (keep showing final state)
      await handleManualReplay(false, true);
    } else {
      throw new Error('Invalid replay response');
    }
  } catch (e) {
    alert(`Failed to replay episode: ${e.message}`);
  }
}

async function handleManualReplay(showAlert = false, keepCurrentView = true) {
  try {
    const res = await replayLastEpisode();
    if (res.status === 'success' && Array.isArray(res.states) && res.states.length > 0) {
      // Store states for manual stepping
      replayStates.value = res.states;
      isManualStepping.value = true;
      
      // Debug: check if policy probs are stored
      console.log('[handleManualReplay] First state has policy_probabilities:', !!res.states[0].policy_probabilities);
      console.log('[handleManualReplay] Last state has policy_probabilities:', !!res.states[res.states.length - 1].policy_probabilities);
      
      if (keepCurrentView) {
        // Keep showing current state (usually the final state), start from end
        currentStepIndex.value = res.states.length - 1;
        // Load policy probs from the final state
        if (res.states[res.states.length - 1].policy_probabilities) {
          policyProbs.value = res.states[res.states.length - 1].policy_probabilities;
        } else {
          policyProbs.value = null;
          console.warn('[handleManualReplay] No policy_probabilities in final state - episode may have been recorded before backend changes');
        }
        // Keep current gameState and gameHistory as-is
      } else {
        // Reset to beginning
        currentStepIndex.value = 0;
        gameState.value = res.states[0];
        gameHistory.value = [res.states[0]];
        // Load policy probs from the first state
        if (res.states[0].policy_probabilities) {
          policyProbs.value = res.states[0].policy_probabilities;
        } else {
          policyProbs.value = null;
          console.warn('[handleManualReplay] No policy_probabilities in first state - episode may have been recorded before backend changes');
        }
      }
    } else {
      throw new Error('Invalid replay response');
    }
  } catch (e) {
    console.error('[App] Failed to load episode for manual stepping:', e);
    if (showAlert) {
      alert(`Failed to load episode for manual stepping: ${e.message}`);
    }
    throw e; // Re-throw so caller knows it failed
  }
}

function stepForward() {
  if (!isManualStepping.value || replayStates.value.length === 0) return;
  if (currentStepIndex.value < replayStates.value.length - 1) {
    currentStepIndex.value += 1;
    const currentState = replayStates.value[currentStepIndex.value];
    
    // Update policy probabilities BEFORE gameState to avoid race conditions
    if (currentState.policy_probabilities) {
      console.log(`[stepForward] Step ${currentStepIndex.value} - updating policy probs FIRST:`, JSON.stringify(currentState.policy_probabilities).substring(0, 150));
      policyProbs.value = { ...currentState.policy_probabilities }; // Create new object to trigger reactivity
    } else {
      console.warn(`[stepForward] Step ${currentStepIndex.value} - no policy_probabilities in state`);
      policyProbs.value = null;
    }
    
    // Then update gameState and history
    gameState.value = currentState;
    gameHistory.value = replayStates.value.slice(0, currentStepIndex.value + 1);
    
    console.log(`[stepForward] Step ${currentStepIndex.value} complete. policyProbs ref value:`, JSON.stringify(policyProbs.value).substring(0, 100));
  }
}

function stepBackward() {
  if (!isManualStepping.value || replayStates.value.length === 0) return;
  if (currentStepIndex.value > 0) {
    currentStepIndex.value -= 1;
    const currentState = replayStates.value[currentStepIndex.value];
    
    // Update policy probabilities BEFORE gameState to avoid race conditions
    if (currentState.policy_probabilities) {
      console.log(`[stepBackward] Step ${currentStepIndex.value} - updating policy probs FIRST:`, JSON.stringify(currentState.policy_probabilities).substring(0, 150));
      policyProbs.value = { ...currentState.policy_probabilities }; // Create new object to trigger reactivity
    } else {
      console.warn(`[stepBackward] Step ${currentStepIndex.value} - no policy_probabilities in state`);
      policyProbs.value = null;
    }
    
    // Then update gameState and history
    gameState.value = currentState;
    gameHistory.value = replayStates.value.slice(0, currentStepIndex.value + 1);
    
    console.log(`[stepBackward] Step ${currentStepIndex.value} complete. policyProbs ref value:`, JSON.stringify(policyProbs.value).substring(0, 100));
  }
}

function handlePlayAgain() {
  gameState.value = null;
  gameHistory.value = [];
  policyProbs.value = null;
  shotAccumulator.value = {};
  placementPassPreview.value = {};
  evalConfig.value = defaultEvalConfig();
  activePlayerId.value = null;
  currentSelections.value = null;
  userSelections.value = {};
  isSelfPlaying.value = false;
  controlsKey.value += 1;
  // Reset manual stepping state
  isManualStepping.value = false;
  replayStates.value = [];
  currentStepIndex.value = 0;
  isEvaluating.value = false;
  perPlayerEvalStats.value = {};
  evalNumEpisodes.value = 100;
  if (initialSetup.value) {
    handleGameStarted(initialSetup.value);
  }
}

// --- Global keyboard shortcuts ---
function onKeydown(e) {
  const tag = (e.target?.tagName || '').toLowerCase();
  if (tag === 'input' || tag === 'textarea') return; // avoid typing conflicts
  const key = e.key?.toLowerCase();
  // Numeric keys: select player by exact ID if present on current team; otherwise by team index
  if (/^[0-9]$/.test(key)) {
    if (gameState.value) {
      const idx = Number(key);
      const allIds = [...(gameState.value.offense_ids || []), ...(gameState.value.defense_ids || [])];
      // Prefer exact player ID match on ANY team (useful when IDs are 2,3,...)
      if (allIds.includes(idx)) {
        activePlayerId.value = idx;
        return;
      }
    }
  }
  if (key === 'n') {
    // New Game
    if (initialSetup.value) handleGameStarted(initialSetup.value);
  } else if (key === 'p') {
    // Self-Play
    if (gameState.value && !gameState.value.done) handleSelfPlayButton();
  } else if (key === 's') {
    // Save Episode
    if (gameState.value?.done) handleSaveEpisode();
  } else if (key === 'y') {
    // Replay
    if (gameState.value?.done && canReplay.value) handleReplay();
  } else if (key === 'r') {
    // Reset Stats
    try { controlsRef.value?.resetStats?.(); } catch (_) {}
  } else if (key === 'c') {
    // Copy stats as Markdown
    try { controlsRef.value?.copyStatsMarkdown?.(); } catch (_) {}
  } else if (key === 't') {
    // Submit Turn
    if (isStepRunning.value || isBallHolderUpdating.value) return;
    try { controlsRef.value?.submitActions?.(); } catch (_) {}
  } else if (key === 'arrowright') {
    // Step forward in manual replay
    if (isManualStepping.value) stepForward();
  } else if (key === 'arrowleft') {
    // Step backward in manual replay
    if (isManualStepping.value) stepBackward();
  } else if (key === 'x') {
    // Reset Positions
    try { handleResetPositions(); } catch (_) {}
  }
}

onMounted(() => {
  // Clear persisted stats once per app load
  try { resetStatsStorage(); } catch (_) {}
  window.addEventListener('keydown', onKeydown);
});
onBeforeUnmount(() => {
  window.removeEventListener('keydown', onKeydown);
});
</script>

<template>
  <main>
    <div class="header-container">
      <img src="@/assets/basketworld-logo.jpg" alt="BasketWorld Logo" class="logo" />
      
      <div class="header-content">
        <header>
          <h1>Welcome to BasketWorld</h1>
        </header>

        <!-- Only render setup when no initialSetup has been chosen -->
        <GameSetup v-if="!initialSetup" @game-started="handleGameStarted" />
        
        <div v-if="isLoading" class="loading">Loading Game...</div>
        <div v-if="error" class="error-message">{{ error }}</div>
        
        <div v-if="gameState" class="ai-toggle">
          <button class="toggle-btn" @click="aiMode = !aiMode">
            <font-awesome-icon :icon="aiMode ? ['fas','toggle-on'] : ['fas','toggle-off']" />
            <span class="toggle-label">AI Mode</span>
          </button>
          <button class="toggle-btn" @click="playerDeterministic = !playerDeterministic" :disabled="!aiMode">
            <font-awesome-icon :icon="playerDeterministic ? ['fas','toggle-on'] : ['fas','toggle-off']" />
            <span class="toggle-label">Player Deterministic</span>
          </button>
          <button class="toggle-btn" @click="opponentDeterministic = !opponentDeterministic" :disabled="!aiMode">
            <font-awesome-icon :icon="opponentDeterministic ? ['fas','toggle-on'] : ['fas','toggle-off']" />
            <span class="toggle-label">Opponent Deterministic</span>
          </button>
        </div>

      </div>
    </div>

    <div v-if="gameState" class="game-container">
      <div class="board-area">
        <div class="run-title">{{ gameState.run_name || gameState.run_id }}</div>
        <div v-if="showShotChartSelector" class="shot-chart-selector">
          <label>Shot chart:</label>
          <select v-model="shotChartTarget">
            <option v-for="opt in shotChartOptions" :key="opt.value" :value="opt.value">
              {{ opt.label }}
            </option>
          </select>
        </div>
        <GameBoard 
          ref="gameBoardRef"
          :game-history="gameHistory" 
          v-model:activePlayerId="activePlayerId"
          :policy-probabilities="policyProbs"
          :is-manual-stepping="isManualStepping"
          :selected-actions="boardSelectedActions"
          :shot-accumulator="shotAccumulatorForBoard"
          :shot-chart-label="shotChartOptions.find(o => o.value === shotChartTarget)?.label || ''"
          :placement-mode="isEvalPlacementMode"
          :placement-editable="evalConfig.placementEditing"
          :placement-positions="evalConfig.positions"
          :placement-ball-holder="evalConfig.ballHolder"
          :placement-pass-probs="placementPassPreview"
          @update-player-position="handlePlayerPositionUpdate"
          @update-placement="handleEvalPlacementUpdate"
          @adjust-shot-clock="handleShotClockAdjustment"
          :is-shot-clock-updating="isShotClockUpdating"
          :disable-transitions="disableTransitionsForCapture"
          :move-progress="moveProgressForCapture"
        />
        <KeyboardLegend />
      </div>
      <div class="controls-area">
        <PlayerControls 
            :key="controlsKey"
            :game-state="gameState" 
            v-model:activePlayerId="activePlayerId"
            :disabled="false"
            :is-replaying="isReplaying"
            :is-manual-stepping="isManualStepping"
            :is-evaluating="isEvaluating"
            :is-policy-swapping="isPolicySwapping"
            :policy-options="policyOptions"
            :policies-loading="policiesLoading"
            :policy-load-error="policyLoadError"
            :stored-policy-probs="policyProbs"
            :mcts-results="lastMctsResults"
            :initial-use-mcts="persistUseMcts"
          :mcts-step-running="isMctsStepRunning"
          :ai-mode="aiMode"
          :deterministic="playerDeterministic"
          :opponent-deterministic="opponentDeterministic"
          :move-history="moveHistory"
          :current-shot-clock="currentShotClock"
          :pressure-exposure="pressureExposure"
          :external-selections="isSelfPlaying ? currentSelections : null"
          :eval-config="evalConfig"
          :eval-num-episodes="evalNumEpisodes"
          :per-player-eval-stats="perPlayerEvalStats"
          @actions-submitted="handleActionsSubmitted" 
          @move-recorded="handleMoveRecorded"
          @ball-holder-updating="isBallHolderUpdating = $event"
          @policy-swap-requested="handlePolicySwap"
          @selections-changed="handleSelectionsChanged"
          @refresh-policies="handleRefreshPolicies"
          @mcts-options-changed="handleMctsOptionsChanged"
          @mcts-toggle-changed="handleMctsToggleChanged"
          @state-updated="handlePatchedGameState"
          @eval-config-changed="handleEvalConfigChanged"
          @eval-run="handleEvalRunRequested"
          @active-tab-changed="handleActiveTabChanged"
          ref="controlsRef"
        />

        <div class="action-buttons">
          <button 
            @click="controlsRef?.submitActions?.()" 
            class="action-button submit-button" 
            :disabled="gameState.done || liveButtonsDisabled || isStepRunning || isBallHolderUpdating"
          >
            {{ gameState.done ? 'Game Over' : 'Submit Turn' }}
          </button>
          
          <button 
            @click="handleSelfPlayButton" 
            class="action-button self-play-button"
            :disabled="!aiMode || gameState.done || liveButtonsDisabled"
          >
            Self-Play
          </button>
          
          <button 
            v-if="gameState && !gameState.done"
            @click="handleResetPositions" 
            class="action-button reset-button" 
            title="Reset player positions to start of turn"
            :disabled="liveButtonsDisabled"
          >
            Reset Pos
          </button>

          <button 
            @click="handlePlayAgain" 
            class="action-button new-game-button"
          >
            New Game
          </button>
        </div>

        <div v-if="gameState.done || isManualStepping" class="save-episode-row">
          <div class="gif-speed-control">
            <label for="gif-speed-slider">GIF speed</label>
            <input
              id="gif-speed-slider"
              type="range"
              min="300"
              max="2400"
              step="100"
              v-model.number="gifStepDurationMs"
            />
            <span class="gif-speed-label">{{ (gifStepDurationMs / 1000).toFixed(2) }}s / step</span>
          </div>
          <button @click="handleSaveEpisode" class="save-episode-button">
            Save Episode
          </button>
        </div>
        
        <div v-if="isManualStepping && canReplay" class="replay-controls">
          <button @click="handleReplay" class="replay-button" title="Replay (animated)">
            <font-awesome-icon :icon="['fas', 'redo']" />
          </button>
          
          <div class="step-controls-inline">
            <button @click="stepBackward" class="step-button" :disabled="currentStepIndex === 0" title="Step back (← arrow)">
              <font-awesome-icon :icon="['fas', 'chevron-left']" />
            </button>
            <span class="step-indicator">
              {{ currentStepIndex + 1 }} / {{ replayStates.length }}
            </span>
            <button @click="stepForward" class="step-button" :disabled="currentStepIndex === replayStates.length - 1" title="Step forward (→ arrow)">
              <font-awesome-icon :icon="['fas', 'chevron-right']" />
            </button>
          </div>
        </div>

      </div>
    </div>
  </main>
</template>

<style scoped>
main {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

.header-container {
  display: flex;
  align-items: flex-start;
  gap: 2rem;
  margin-bottom: 0.5rem;
}

.logo {
  width: 200px;
  height: 200px;
  object-fit: contain;
  border-radius: 20px;
  box-shadow: 0 8px 24px rgba(56, 189, 248, 0.35);
  flex-shrink: 0;
}

.header-content {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

header {
  text-align: center;
  letter-spacing: 0.15em;
  text-transform: uppercase;
  color: var(--app-accent);
  font-size: 1.5rem;
}

.loading,
.error-message {
  text-align: center;
  font-size: 0.95rem;
  color: var(--app-text-muted);
}

.error-message {
  color: #fb7185;
}

.panel {
  background: var(--app-panel);
  border: 1px solid var(--app-panel-border);
  border-radius: 24px;
  padding: 1.5rem;
  box-shadow: 0 20px 45px rgba(2, 6, 23, 0.45);
}

.ai-toggle,
.eval-controls {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 0.8rem;
  margin-bottom: 0.5rem;
}

.toggle-btn {
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
  transition: color 0.15s ease, border 0.15s ease;
}

.toggle-btn:hover:not(:disabled) {
  color: var(--app-accent);
  border-color: var(--app-accent-strong);
}

.toggle-btn:disabled {
  opacity: 0.35;
  cursor: not-allowed;
}

.eval-controls-row {
  display: flex;
  align-items: center;
  gap: 0.75rem;
}

.eval-input {
  width: 110px;
  padding: 0.55rem 0.8rem;
  border-radius: 16px;
  border: 1px solid rgba(56, 189, 248, 0.35);
  background: rgba(13, 20, 38, 0.85);
  color: var(--app-text);
  letter-spacing: 0.05em;
  font-size: 1rem;
}

.eval-button {
  border-radius: 999px;
  border: 1px solid rgba(148, 163, 184, 0.4);
  background: transparent;
  color: var(--app-text);
  padding: 0.5rem 1.2rem;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  font-size: 1rem;
}

.eval-button:hover:not(:disabled) {
  border-color: var(--app-success);
  color: var(--app-success);
}

.eval-button:disabled {
  opacity: 0.4;
}

.eval-status {
  color: var(--app-text-muted);
  font-size: 0.85rem;
}

.eval-progress-bar {
  width: 260px;
  height: 6px;
  background: rgba(15, 23, 42, 0.7);
  border-radius: 999px;
  border: 1px solid rgba(148, 163, 184, 0.35);
  overflow: hidden;
}

.eval-progress-fill {
  height: 100%;
  background: linear-gradient(90deg, var(--app-success), var(--app-accent));
}

.eval-progress-fill.indeterminate {
  animation: indeterminate-progress 1.5s linear infinite;
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

.game-container {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

.board-area {
  display: flex;
  flex-direction: column;
  gap: 0.6rem;
  align-items: center;
}

.run-title {
  font-size: 1.25rem;
  color: var(--app-text-muted);
  letter-spacing: 0.1em;
  text-transform: uppercase;
  margin-bottom: 0.5rem;
  text-align: center;
}
.shot-chart-selector {
  margin: -0.25rem 0 0.25rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}
.shot-chart-selector label {
  color: var(--app-text-muted);
  font-size: 0.9rem;
}
.shot-chart-selector select {
  background: rgba(13, 20, 38, 0.85);
  border: 1px solid rgba(56, 189, 248, 0.35);
  color: var(--app-text);
  border-radius: 10px;
  padding: 0.25rem 0.5rem;
}

.controls-area {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.action-buttons {
  display: flex;
  gap: 0.6rem;
  justify-content: center;
}

.save-episode-row {
  display: flex;
  align-items: center;
  gap: 0.9rem;
  justify-content: center;
  flex-wrap: wrap;
}

.gif-speed-control {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.4rem 0.8rem;
  border: 1px solid rgba(148, 163, 184, 0.35);
  border-radius: 10px;
  background: rgba(15, 23, 42, 0.4);
  color: var(--app-text-muted);
}

.gif-speed-control input[type='range'] {
  accent-color: var(--app-accent);
}

.gif-speed-label {
  font-weight: 600;
  color: var(--app-text);
  min-width: 84px;
  text-align: right;
}

.action-button,
.save-episode-button,
.replay-button,
.step-button {
  flex: 1;
  border-radius: 999px;
  border: 1px solid rgba(148, 163, 184, 0.4);
  background: transparent;
  color: var(--app-text);
  padding: 0.55rem 1rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  font-size: 1rem;
  transition: border 0.15s ease, color 0.15s ease;
}

.action-button:hover:not(:disabled),
.save-episode-button:hover,
.replay-button:hover,
.step-button:hover:not(:disabled) {
  border-color: var(--app-accent);
  color: var(--app-accent);
}

.action-button:disabled,
.step-button:disabled {
  opacity: 0.3;
  cursor: not-allowed;
}

.replay-controls {
  display: flex;
  align-items: center;
  gap: 1rem;
  padding: 1rem;
  background: rgba(8, 11, 19, 0.85);
  border-radius: 24px;
  border: 1px solid rgba(148, 163, 184, 0.2);
  justify-content: center;
}

.step-controls-inline {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  color: var(--app-text-muted);
}

.step-indicator {
  font-size: 0.85rem;
  min-width: 60px;
  text-align: center;
}
</style>
