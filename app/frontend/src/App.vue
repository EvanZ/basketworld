<script setup>
import { ref, computed, watch, onMounted, onBeforeUnmount, nextTick } from 'vue';
import GameSetup from './components/GameSetup.vue';
import GameBoard from './components/GameBoard.vue';
import PlayerControls from './components/PlayerControls.vue';
import { ref as vueRef } from 'vue';
import KeyboardLegend from './components/KeyboardLegend.vue';
import { initGame, stepGame, getPolicyProbs, saveEpisode, startSelfPlay, replayLastEpisode, getPhiParams, setPhiParams, runEvaluation, getPassStealProbabilities } from './services/api';
import { resetStatsStorage } from './services/stats';

const gameState = ref(null);      // For current state and UI logic
const gameHistory = ref([]);     // For ghost trails
const policyProbs = ref(null);   // For AI suggestions
const isLoading = ref(false);
const error = ref(null);
const initialSetup = ref(null);
const activePlayerId = ref(null);
// Reflect the actions being applied on each step to keep UI tabs in sync during self-play
const currentSelections = ref(null);
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
const phiRef = vueRef(null);

// AI Mode for pre-selecting actions, not automatic play
const aiMode = ref(true);
const playerDeterministic = ref(false);
const opponentDeterministic = ref(true);

// Shared move tracking between manual and AI play
const moveHistory = ref([]);

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

// Evaluation mode state
const isEvaluating = ref(false);
const evalNumEpisodes = ref(100);

// Watch for when episodes end to stop auto-play behavior
watch(gameState, async (newState, oldState) => {
    // Only fetch policy probs if NOT in manual stepping mode (to avoid overwriting stored historical probs)
    if (newState && !newState.done && !isManualStepping.value) {
        try {
            console.log('[App] Fetching policy probs from API (not in manual stepping mode)');
            const response = await getPolicyProbs();
            policyProbs.value = response;
        } catch (err) {
            console.error('[App] Failed to fetch policy probs:', err);
        }
    } else if (newState && !newState.done && isManualStepping.value) {
        console.log('[App] Skipping policy probs fetch - in manual stepping mode, using stored probs');
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
  // Clear any self-play selections state
  currentSelections.value = null;
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
  initialSetup.value = setupData;
  // Keep board hidden but avoid flashing setup screen when we already have a setup
  gameState.value = null;      // Ensure old board is cleared
  gameHistory.value = [];      // Clear history
  activePlayerId.value = null;
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
      gameHistory.value.push(response.state);
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
  // No loading indicator for steps, feels more responsive
  try {
    const response = await stepGame(actions, playerDeterministic.value, opponentDeterministic.value);
     if (response.status === 'success') {
      gameState.value = response.state;
      gameHistory.value.push(response.state);
      
      // Update the last move with action results and shot clock BEFORE action
      if (moveHistory.value.length > 0) {
        const lastMove = moveHistory.value[moveHistory.value.length - 1];
        if (response.state.last_action_results) {
          lastMove.actionResults = response.state.last_action_results;
        }
        // Store shot clock when action was decided (before execution): board + 1
        if (response.state.shot_clock !== undefined) {
          lastMove.shotClock = response.state.shot_clock + 1;
        }
      }
      
      // If game is done, add an END row
      if (response.state.done && moveHistory.value.length > 0) {
        const userTeamIds = response.state.offense_ids?.includes(0) 
          ? response.state.offense_ids 
          : response.state.defense_ids;
        const endMoves = {};
        userTeamIds.forEach(pid => {
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
    } else {
      throw new Error(response.message || 'Failed to process step.');
    }
  } catch (err) {
    error.value = err.message;
    console.error(err);
  }
}

async function handleMoveRecorded(moveData) {
    console.log('[App] Recording move:', moveData);
    
    // Capture ball holder before the action is taken
    if (gameState.value && gameState.value.ball_holder !== undefined) {
        moveData.ballHolder = gameState.value.ball_holder;
    }
    
    // Fetch pass steal probabilities before the action is taken
    try {
        const passStealProbs = await getPassStealProbabilities();
        moveData.passStealProbabilities = passStealProbs;
        console.log('[App] Added pass steal probs to move:', passStealProbs);
    } catch (err) {
        console.warn('[App] Failed to fetch pass steal probs for move:', err);
        moveData.passStealProbabilities = {};
    }
    
    moveHistory.value.push(moveData);
}

// Handler for Self-Play button click
function handleSelfPlayButton() {
  // Get current selections from PlayerControls
  const preselected = controlsRef.value?.getSelectedActions?.() || null;
  handleSelfPlay(preselected);
}

// New function for self-play mode (runs full episode)
async function handleSelfPlay(preselected = null) {
  if (!gameState.value || !aiMode.value) return;
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
      
      if (policyProbs.value) {
        // Determine which players are user-controlled
        const userControlledIds = gameState.value.user_team_name === 'OFFENSE' 
          ? gameState.value.offense_ids 
          : gameState.value.defense_ids;
        
        // Get AI actions for user-controlled players
        for (const playerId of userControlledIds) {
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
      
      // Track moves for AI self-play
      if (Object.keys(aiActions).length > 0) {
        const currentTurn = moveHistory.value.length + 1;
        const teamMoves = {};
        
        // Convert action indices back to action names for tracking
        const actionNames = [
          "NOOP", "MOVE_E", "MOVE_NE", "MOVE_NW", "MOVE_W", "MOVE_SW", "MOVE_SE", 
          "SHOOT", "PASS_E", "PASS_NE", "PASS_NW", "PASS_W", "PASS_SW", "PASS_SE"
        ];
        
        const appliedSelections = {};
        for (const [playerId, actionIndex] of Object.entries(aiActions)) {
          const actionName = actionNames[actionIndex] || 'UNKNOWN';
          teamMoves[`Player ${playerId}`] = actionName;
          appliedSelections[playerId] = actionName;
        }
        // Update UI tabs to mirror applied actions
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
        
        moveHistory.value.push({
          turn: currentTurn,
          moves: teamMoves,
          ballHolder: ballHolder,
          passStealProbabilities: passStealProbs
        });
      }
      
      const response = await stepGame(aiActions, playerDeterministic.value, opponentDeterministic.value);
      if (response.status === 'success') {
        gameState.value = response.state;
        gameHistory.value.push(response.state);
        
        // Update the last move with action results and shot clock BEFORE action
        if (moveHistory.value.length > 0) {
          const lastMove = moveHistory.value[moveHistory.value.length - 1];
          if (response.state.last_action_results) {
            lastMove.actionResults = response.state.last_action_results;
          }
          // Store shot clock when action was decided (before execution): board + 1
          if (response.state.shot_clock !== undefined) {
            lastMove.shotClock = response.state.shot_clock + 1;
          }
        }
        
        // If game is done, add an END row
        if (response.state.done && moveHistory.value.length > 0) {
          const userTeamIds = response.state.offense_ids?.includes(0) 
            ? response.state.offense_ids 
            : response.state.defense_ids;
          const endMoves = {};
          userTeamIds.forEach(pid => {
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

async function handleEvaluation() {
  if (!gameState.value || isEvaluating.value) return;
  
  const numEpisodes = Math.max(1, Math.min(evalNumEpisodes.value, 10000));
  
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
    
    // Run evaluation on backend (this will block until all episodes complete)
    const response = await runEvaluation(numEpisodes, playerDeterministic.value, opponentDeterministic.value);
    
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
      
      // Now update the game state AFTER recording all stats
      if (lastEpisodeState) {
        console.log('[App] Updating gameState to show last episode (isEvaluating still true)');
        gameState.value = lastEpisodeState;
        gameHistory.value = [lastEpisodeState];
        
        // Wait for Vue to process all reactive updates (including watchers) before setting isEvaluating to false
        await nextTick();
        console.log('[App] nextTick completed, watchers should have run with isEvaluating=true');
      }
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

async function handleSaveEpisode() {
  try {
    const res = await saveEpisode();
    alert(`Episode saved to ${res.file_path}`);
  } catch (e) {
    alert(`Failed to save episode: ${e.message}`);
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
      for (const s of res.states) {
        gameState.value = s;
        gameHistory.value.push(s);
        // Small delay between frames
        // eslint-disable-next-line no-await-in-loop
        await new Promise(resolve => setTimeout(resolve, 300));
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
      if (res.states[0].policy_probabilities) {
        console.log('[handleManualReplay] Sample first state probs:', JSON.stringify(res.states[0].policy_probabilities).substring(0, 200));
      }
      if (res.states.length > 1 && res.states[1].policy_probabilities) {
        console.log('[handleManualReplay] Sample second state probs:', JSON.stringify(res.states[1].policy_probabilities).substring(0, 200));
      }
      
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
  activePlayerId.value = null;
  currentSelections.value = null;
  isSelfPlaying.value = false;
  controlsKey.value += 1;
  // Reset manual stepping state
  isManualStepping.value = false;
  replayStates.value = [];
  currentStepIndex.value = 0;
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
      const isOffense = gameState.value.user_team_name === 'OFFENSE';
      const teamIds = isOffense ? (gameState.value.offense_ids || []) : (gameState.value.defense_ids || []);
      // Prefer exact player ID match on current team (useful when IDs are 2,3,...)
      if (teamIds.includes(idx)) {
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
    try { controlsRef.value?.submitActions?.(); } catch (_) {}
  } else if (key === 'arrowright') {
    // Step forward in manual replay
    if (isManualStepping.value) stepForward();
  } else if (key === 'arrowleft') {
    // Step backward in manual replay
    if (isManualStepping.value) stepBackward();
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

    <div v-if="gameState && !gameState.done && !isSelfPlaying" class="eval-controls">
      <div class="eval-controls-row">
        <input 
          type="number" 
          v-model.number="evalNumEpisodes" 
          min="1" 
          max="10000" 
          class="eval-input"
          :disabled="isEvaluating"
          placeholder="Num episodes"
        />
        <button 
          @click="handleEvaluation" 
          class="eval-button"
          :disabled="isEvaluating || isSelfPlaying"
        >
          {{ isEvaluating ? 'Evaluating...' : 'Eval' }}
        </button>
        <span v-if="isEvaluating" class="eval-status">
          Running {{ evalNumEpisodes }} episodes...
        </span>
      </div>
      <div v-if="isEvaluating" class="eval-progress-bar">
        <div class="eval-progress-fill indeterminate"></div>
      </div>
    </div>

    <div v-if="gameState" class="game-container">
      <div class="board-area">
        <div class="run-title">{{ gameState.run_name || gameState.run_id }}</div>
        <GameBoard 
          :game-history="gameHistory" 
          v-model:activePlayerId="activePlayerId"
          :policy-probabilities="policyProbs"
          :is-manual-stepping="isManualStepping"
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
            :stored-policy-probs="policyProbs"
            :ai-mode="aiMode"
            :deterministic="playerDeterministic"
            :move-history="moveHistory"
            :current-shot-clock="currentShotClock"
            :external-selections="isSelfPlaying ? currentSelections : null"
            @actions-submitted="handleActionsSubmitted" 
            @move-recorded="handleMoveRecorded"
            ref="controlsRef"
        />

        <div class="action-buttons">
          <button 
            @click="controlsRef?.submitActions?.()" 
            class="action-button submit-button" 
            :disabled="gameState.done"
          >
            {{ gameState.done ? 'Game Over' : 'Submit Turn' }}
          </button>
          
          <button 
            @click="handleSelfPlayButton" 
            class="action-button self-play-button"
            :disabled="!aiMode || gameState.done"
          >
            Self-Play
          </button>
          
          <button 
            @click="handlePlayAgain" 
            class="action-button new-game-button"
          >
            New Game
          </button>
        </div>

        <button v-if="gameState.done || isManualStepping" @click="handleSaveEpisode" class="save-episode-button">
          Save Episode
        </button>
        
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
header {
  text-align: center;
  margin-bottom: 2rem;
}
.loading {
  text-align: center;
  font-size: 1.5rem;
  margin-top: 2rem;
}
.error-message {
  color: red;
  text-align: center;
  margin-top: 1rem;
}

.ai-toggle {
  text-align: center;
  margin-bottom: 1rem;
}
.toggle-btn {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.4rem 0.8rem;
  border: 1px solid #ddd;
  border-radius: 6px;
  background: #fff;
  cursor: pointer;
  margin: 0 0.5rem;
  font-size: 1.2rem;
}
.toggle-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
.toggle-label {
  font-weight: 500;
}

.eval-controls {
  text-align: center;
  margin-bottom: 1rem;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.5rem;
}

.eval-controls-row {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.75rem;
}

.eval-input {
  width: 100px;
  padding: 0.5rem;
  border: 1px solid #ddd;
  border-radius: 6px;
  font-size: 1rem;
}

.eval-button {
  padding: 0.5rem 1rem;
  border: 1px solid #ddd;
  border-radius: 6px;
  background: #4CAF50;
  color: white;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: background 0.2s;
}

.eval-button:hover:not(:disabled) {
  background: #45a049;
}

.eval-button:disabled {
  opacity: 0.6;
  cursor: not-allowed;
  background: #999;
}

.eval-status {
  font-size: 1rem;
  font-weight: 500;
  color: #333;
}

.eval-progress-bar {
  width: 300px;
  height: 8px;
  background-color: #ddd;
  border-radius: 4px;
  overflow: hidden;
  border: 1px solid #999;
}

.eval-progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #4CAF50 0%, #45a049 100%);
  transition: width 0.3s ease;
}

.eval-progress-fill.indeterminate {
  animation: indeterminate-progress 1.5s ease-in-out infinite;
}

@keyframes indeterminate-progress {
  0% {
    transform: translateX(-100%);
    width: 30%;
  }
  50% {
    transform: translateX(250%);
  }
  100% {
    transform: translateX(-100%);
    width: 30%;
  }
}
.game-container {
  display: flex;
  flex-direction: row; /* Changed to row */
  justify-content: flex-start; /* Left area fills available space */
  align-items: flex-start; /* Align items to the top */
  gap: 2rem; /* Increased gap */
}

.board-area {
  display: flex;
  flex-direction: column;
  align-items: center;
  flex: 1;
  min-width: 640; /* allow flex child to shrink properly */
}

.run-title {
  font-size: 1.1rem;
  font-weight: 600;
  margin-bottom: 0.5rem;
  text-align: center;
}

.controls-area {
  min-width: 400px; /* Give the controls area a fixed width */
  display: flex;
  flex: 1.5;
  flex-direction: column;
}

.action-buttons {
  display: flex;
  gap: 0.5rem;
  margin-top: 1rem;
}

.action-button {
  flex: 1;
  padding: 0.75rem 1rem;
  font-size: 1rem;
  font-weight: 600;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.2s ease;
}

.action-button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.submit-button {
  background: #007bff;
  color: white;
}

.submit-button:hover:not(:disabled) {
  background: #0056b3;
}

.self-play-button {
  background: #28a745;
  color: white;
}

.self-play-button:hover:not(:disabled) {
  background: #218838;
}

.new-game-button {
  background: orange;
  color: white;
}

.new-game-button:hover {
  background: #5a6268;
}

.save-episode-button {
  margin-top: 1rem;
  padding: 0.5rem 1rem;
}

.replay-controls {
  display: flex;
  align-items: center;
  gap: 1rem;
  margin-top: 1rem;
  padding: 0.75rem;
  background: rgba(255, 255, 255, 0.9);
  border: 1px solid #ddd;
  border-radius: 8px;
  box-shadow: 0 2px 6px rgba(0,0,0,0.1);
}

.replay-button {
  padding: 0.5rem 1rem;
  font-size: 1.1rem;
  background: #4CAF50;
  color: white;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  transition: background 0.2s;
}

.replay-button:hover {
  background: #45a049;
}

.step-controls-inline {
  display: flex;
  align-items: center;
  gap: 0.75rem;
}

.step-button {
  padding: 0.5rem 0.75rem;
  font-size: 1.2rem;
  background: #2196F3;
  color: white;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  transition: background 0.2s;
  min-width: 40px;
}

.step-button:hover:not(:disabled) {
  background: #1976D2;
}

.step-button:disabled {
  background: #ccc;
  cursor: not-allowed;
  opacity: 0.6;
}

.step-indicator {
  font-weight: 600;
  font-size: 1rem;
  color: #333;
  min-width: 60px;
  text-align: center;
}
</style>
