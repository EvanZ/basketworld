<script setup>
import { ref, computed, watch, onMounted, nextTick } from 'vue';
import { defineExpose } from 'vue';
import HexagonControlPad from './HexagonControlPad.vue';
import { getActionValues, getRewards } from '@/services/api';
import { loadStats, saveStats, resetStatsStorage } from '@/services/stats';

// Import API_BASE_URL for policy probabilities fetch
const API_BASE_URL = import.meta.env?.VITE_API_BASE_URL || 'http://localhost:8080';

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
});

const emit = defineEmits(['actions-submitted', 'update:activePlayerId', 'move-recorded']);

const selectedActions = ref({});

// Debug: Watch for any changes to selectedActions
watch(selectedActions, (newActions, oldActions) => {
  console.log('[PlayerControls] 🔍 selectedActions changed from:', oldActions, 'to:', newActions);
}, { deep: true });

const actionValues = ref(null);
const valueRange = ref({ min: 0, max: 0 });
// shot probability is displayed on the board, not in controls
const policyProbabilities = ref(null);

// Add rewards tracking
const activeTab = ref('controls');
const rewardHistory = ref([]);
const episodeRewards = ref({ offense: 0.0, defense: 0.0 });
const rewardParams = ref(null);
const mlflowPhiParams = ref(null);

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
const ppp = computed(() => safeDiv(statsState.value.points, Math.max(1, statsState.value.episodes)));
const avgRewardPerEp = computed(() => safeDiv(statsState.value.rewardSum, Math.max(1, statsState.value.episodes)));
const avgEpisodeLen = computed(() => safeDiv(statsState.value.episodeStepsSum, Math.max(1, statsState.value.episodes)));

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

    if (isDunk) {
      statsState.value.dunk.attempts += 1;
      if (made) statsState.value.dunk.made += 1;
      if (assisted) statsState.value.dunk.assists += 1;
    } else if (isThree) {
      statsState.value.threePt.attempts += 1;
      if (made) statsState.value.threePt.made += 1;
      if (assisted) statsState.value.threePt.assists += 1;
    } else if (isTwo) {
      statsState.value.twoPt.attempts += 1;
      if (made) statsState.value.twoPt.made += 1;
      if (assisted) statsState.value.twoPt.assists += 1;
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
      ['Total turnovers', String(s.turnovers)],
    ];
    const shotsHeader = ['Type', 'Attempts', 'Made', 'FG%', 'Assists'];
    const shotsRows = [
      ['Dunks', s.dunk.attempts, s.dunk.made, fg(s.dunk.made, s.dunk.attempts), s.dunk.assists],
      ['2PT', s.twoPt.attempts, s.twoPt.made, fg(s.twoPt.made, s.twoPt.attempts), s.twoPt.assists],
      ['3PT', s.threePt.attempts, s.threePt.made, fg(s.threePt.made, s.threePt.attempts), s.threePt.assists],
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

// Shot probability display is handled on the board

// Fetch policy probabilities for probabilistic action sampling
async function fetchPolicyProbabilities() {
  if (!props.gameState || props.gameState.done) {
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

// Fetch action values for all user-controlled players (needed for AI mode)
async function fetchAllActionValues() {
  if (!props.gameState || props.gameState.done) {
    console.log('[PlayerControls] Skipping fetchAllActionValues - no game state or game done');
    return;
  }
  
  const allValues = {};
  const controlledIds = userControlledPlayerIds.value;
  
  console.log('[PlayerControls] Fetching action values for all players:', controlledIds);
  
  // Also calculate min/max for color scaling
  let allNumericValues = [];
  
  for (const playerId of controlledIds) {
    try {
      const values = await getActionValues(playerId);
      allValues[playerId] = values;
      console.log(`[PlayerControls] Fetched action values for player ${playerId}:`, values);
      
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
  console.log('[PlayerControls] Game state changed, fetching AI data... Ball holder:', newGameState?.ball_holder, 'Manual stepping:', props.isManualStepping);
  
  if (newGameState && !newGameState.done) {
    // Fetch both action values and policy probabilities for AI mode
    try {
      console.log('[PlayerControls] Starting to fetch action values for ball holder:', newGameState.ball_holder);
      await fetchAllActionValues();
      
      // Only fetch policy probabilities from API if NOT in manual stepping mode
      if (!props.isManualStepping) {
        console.log('[PlayerControls] Fetching policy probabilities from API for ball holder:', newGameState.ball_holder);
        await fetchPolicyProbabilities();
        console.log('[PlayerControls] Policy probabilities fetch completed for ball holder:', newGameState.ball_holder);
      } else {
        console.log('[PlayerControls] Skipping API fetch - in manual stepping mode, will use stored probs');
      }
    } catch (error) {
      console.error('[PlayerControls] Error during AI data fetch:', error);
    }
  } else {
    console.log('[PlayerControls] Clearing AI data - no game state or game done');
    actionValues.value = null;
    // Don't clear policyProbabilities if in manual stepping mode (we're using stored ones)
    if (!props.isManualStepping) {
      policyProbabilities.value = null;
    }
    valueRange.value = { min: 0, max: 0 };
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
watch(userControlledPlayerIds, (newPlayerIds) => {
    if (newPlayerIds && newPlayerIds.length > 0 && props.activePlayerId === null) {
        emit('update:activePlayerId', newPlayerIds[0]);
    }
}, { immediate: true });

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

function submitActions() {
  let actionsToSubmit = {};
  
  if (props.aiMode) {
    // Use AI actions for user-controlled players (selected actions should already be set)
    for (const playerId of userControlledPlayerIds.value) {
      const actionName = selectedActions.value[playerId] || 'NOOP';
      const actionIndex = actionNames.indexOf(actionName);
      actionsToSubmit[playerId] = actionIndex !== -1 ? actionIndex : 0;
    }
  } else {
    // Use manually selected actions (only for user-controlled players)
    for (const playerId of userControlledPlayerIds.value) {
      const actionName = selectedActions.value[playerId] || 'NOOP';
      const actionIndex = actionNames.indexOf(actionName);
      actionsToSubmit[playerId] = actionIndex !== -1 ? actionIndex : 0;
    }
  }
  
  console.log('[PlayerControls] Emitting actions-submitted with payload:', actionsToSubmit);
  
  // Track moves for the selected team
  const currentTurn = props.moveHistory.length + 1;
  const teamMoves = {};
  
  for (const playerId of userControlledPlayerIds.value) {
    const actionName = selectedActions.value[playerId] || 'NOOP';
    teamMoves[`Player ${playerId}`] = actionName;
  }
  
  emit('move-recorded', {
    turn: currentTurn,
    moves: teamMoves
  });
  
  emit('actions-submitted', actionsToSubmit);
  
  if (!props.aiMode) {
    // Only clear selections in manual mode
    selectedActions.value = {};
    if (userControlledPlayerIds.value.length > 0) {
      emit('update:activePlayerId', userControlledPlayerIds.value[0]);
    }
  }
}

function getSelectedActions() {
  // Return current selections for parent to use
  return { ...selectedActions.value };
}

// Watch for AI mode or deterministic mode changes to pre-select actions
watch([() => props.aiMode, () => props.deterministic], ([newAiMode, newDeterministic]) => {
  // If parent is driving selections (self-play), don't override
  if (props.externalSelections) return;
  console.log('[PlayerControls] 🔄 AI mode watch triggered - AI:', newAiMode, 'Deterministic:', newDeterministic, 'Ball holder:', props.gameState?.ball_holder);
  console.log('[PlayerControls] actionValues.value:', actionValues.value);
  console.log('[PlayerControls] policyProbabilities.value:', policyProbabilities.value);
  console.log('[PlayerControls] userControlledPlayerIds.value:', userControlledPlayerIds.value);
  
  try {
    if (newAiMode) {
      // Pre-select AI actions for USER-CONTROLLED players only when AI mode is enabled
      const newSelections = {};
      const controlledIds = userControlledPlayerIds.value;
      
      for (const playerId of controlledIds) {
        const legalActions = getLegalActions(playerId);
        
        // Debug logging for action masking issues
        console.log(`[PlayerControls] Player ${playerId} - Ball holder: ${props.gameState?.ball_holder}, Legal actions:`, legalActions);
        if (legalActions.includes('SHOOT') && playerId !== props.gameState?.ball_holder) {
          console.log(`[PlayerControls] 🚨 BUG: Player ${playerId} can SHOOT but is NOT ball holder (${props.gameState?.ball_holder})`);
        }
        if (legalActions.some(action => action.startsWith('PASS_')) && playerId !== props.gameState?.ball_holder) {
          console.log(`[PlayerControls] 🚨 BUG: Player ${playerId} can PASS but is NOT ball holder (${props.gameState?.ball_holder})`);
        }
        
        if (legalActions.length > 0) {
          let selectedAction = null;
          
          if (newDeterministic && policyProbabilities.value && policyProbabilities.value[playerId]) {
            // Deterministic: mimic policy.predict(..., deterministic=True) → argmax of policy distribution
            const probs = policyProbabilities.value[playerId];
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
          } else if (!newDeterministic && policyProbabilities.value && policyProbabilities.value[playerId]) {
            // Probabilistic: Sample from policy probabilities
            const playerProbs = policyProbabilities.value[playerId];
            console.log(`[PlayerControls] Attempting probabilistic sampling for player ${playerId}`);
            console.log(`[PlayerControls] Player probs:`, playerProbs);
            console.log(`[PlayerControls] Legal actions:`, legalActions);
            
            // Filter probabilities to only include legal actions
            const legalActionIndices = [];
            const legalProbs = [];
            
            for (let i = 0; i < playerProbs.length && i < actionNames.length; i++) {
              if (legalActions.includes(actionNames[i])) {
                legalActionIndices.push(i);
                legalProbs.push(playerProbs[i]);
              }
            }
            
            if (legalProbs.length > 0) {
              const sampledIndex = sampleFromProbabilities(legalProbs);
              const actionIndex = legalActionIndices[sampledIndex];
              selectedAction = actionNames[actionIndex];
              console.log(`[PlayerControls] Selected PROBABILISTIC action for player ${playerId}: ${selectedAction} (prob: ${(playerProbs[actionIndex] * 100).toFixed(1)}%)`);
            } else {
              console.log(`[PlayerControls] No legal actions with probabilities for player ${playerId}`);
            }
          } else {
            console.log(`[PlayerControls] Cannot do probabilistic sampling for player ${playerId}:`);
            console.log(`  - newDeterministic: ${newDeterministic}`);
            console.log(`  - policyProbabilities.value: ${!!policyProbabilities.value}`);
            console.log(`  - policyProbabilities.value[${playerId}]: ${!!policyProbabilities.value?.[playerId]}`);
          }
          
          if (selectedAction) {
            newSelections[playerId] = selectedAction;
          } else {
            console.log(`[PlayerControls] No valid action selected for player ${playerId}`);
          }
        } else {
          console.log(`[PlayerControls] No legal actions for player ${playerId}`);
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
    const controlledIds = userControlledPlayerIds.value;

    for (const playerId of controlledIds) {
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

      if (props.deterministic) {
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
        console.log(`[PlayerControls] Deterministic argmax action for player ${playerId}: ${selectedAction} (prob: ${(playerProbs[actionIndex] * 100).toFixed(1)}%)`);
      } else {
        // Probabilistic: sample among legal
        const sampledIndex = sampleFromProbabilities(legalProbs);
        const actionIndex = legalActionIndices[sampledIndex];
        const selectedAction = actionNames[actionIndex];
        newSelections[playerId] = selectedAction;
        console.log(`[PlayerControls] Re-sampled PROB action for player ${playerId}: ${selectedAction} (prob: ${(playerProbs[actionIndex] * 100).toFixed(1)}%)`);
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
  }
});

import PhiShaping from './PhiShaping.vue';
import { ref as vueRef } from 'vue';
const phiRef = vueRef(null);

// Observation parsing utilities
function getAngleDescription(cosAngle) {
  if (cosAngle > 0.9) return '👍 In front (defender blocking)';
  if (cosAngle > 0.5) return '📍 Somewhat in front';
  if (cosAngle > -0.5) return '↔️ Side (help defense)';
  if (cosAngle > -0.9) return '📍 Somewhat behind';
  return '🔙 Behind (beaten)';
}

// Computed properties for observation parsing
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
  if (!props.gameState || !props.gameState.obs) return [];
  const obs = props.gameState.obs;
  const nPlayers = (props.gameState.offense_ids?.length || 0) + (props.gameState.defense_ids?.length || 0);
  const idx = nPlayers * 2 + nPlayers + 1 + nPlayers + 2;
  return obs.slice(idx, idx + 2);
});

const allPairsDistances = computed(() => {
  if (!props.gameState || !props.gameState.obs) return [];
  const obs = props.gameState.obs;
  const nPlayers = (props.gameState.offense_ids?.length || 0) + (props.gameState.defense_ids?.length || 0);
  const nOffense = props.gameState.offense_ids?.length || 0;
  const nDefense = props.gameState.defense_ids?.length || 0;
  const idx = nPlayers * 2 + nPlayers + 1 + nPlayers + 2 + 2;
  const size = nOffense * nDefense;
  return obs.slice(idx, idx + size);
});

const allPairsAngles = computed(() => {
  if (!props.gameState || !props.gameState.obs) return [];
  const obs = props.gameState.obs;
  const nPlayers = (props.gameState.offense_ids?.length || 0) + (props.gameState.defense_ids?.length || 0);
  const nOffense = props.gameState.offense_ids?.length || 0;
  const nDefense = props.gameState.defense_ids?.length || 0;
  const idx = nPlayers * 2 + nPlayers + 1 + nPlayers + 2 + 2 + (nOffense * nDefense);
  const size = nOffense * nDefense;
  return obs.slice(idx, idx + size);
});

const laneSteps = computed(() => {
  if (!props.gameState || !props.gameState.obs) return [];
  const obs = props.gameState.obs;
  const nPlayers = (props.gameState.offense_ids?.length || 0) + (props.gameState.defense_ids?.length || 0);
  const nOffense = props.gameState.offense_ids?.length || 0;
  const nDefense = props.gameState.defense_ids?.length || 0;
  const idx = nPlayers * 2 + nPlayers + 1 + nPlayers + 2 + 2 + (nOffense * nDefense) + (nOffense * nDefense);
  return obs.slice(idx, idx + nPlayers);
});

const expectedPoints = computed(() => {
  if (!props.gameState || !props.gameState.obs) return [];
  const obs = props.gameState.obs;
  const nPlayers = (props.gameState.offense_ids?.length || 0) + (props.gameState.defense_ids?.length || 0);
  const nOffense = props.gameState.offense_ids?.length || 0;
  const nDefense = props.gameState.defense_ids?.length || 0;
  const idx = nPlayers * 2 + nPlayers + 1 + nPlayers + 2 + 2 + (nOffense * nDefense) + (nOffense * nDefense) + nPlayers;
  return obs.slice(idx, idx + nOffense);
});

const turnoverProbs = computed(() => {
  if (!props.gameState || !props.gameState.obs) return [];
  const obs = props.gameState.obs;
  const nPlayers = (props.gameState.offense_ids?.length || 0) + (props.gameState.defense_ids?.length || 0);
  const nOffense = props.gameState.offense_ids?.length || 0;
  const nDefense = props.gameState.defense_ids?.length || 0;
  const idx = nPlayers * 2 + nPlayers + 1 + nPlayers + 2 + 2 + (nOffense * nDefense) + (nOffense * nDefense) + nPlayers + nOffense;
  return obs.slice(idx, idx + nOffense);
});

const stealRisks = computed(() => {
  if (!props.gameState || !props.gameState.obs) return [];
  const obs = props.gameState.obs;
  const nPlayers = (props.gameState.offense_ids?.length || 0) + (props.gameState.defense_ids?.length || 0);
  const nOffense = props.gameState.offense_ids?.length || 0;
  const nDefense = props.gameState.defense_ids?.length || 0;
  const idx = nPlayers * 2 + nPlayers + 1 + nPlayers + 2 + 2 + (nOffense * nDefense) + (nOffense * nDefense) + nPlayers + nOffense + nOffense;
  return obs.slice(idx, idx + nOffense);
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
        :class="{ active: activeTab === 'moves' }"
        @click="activeTab = 'moves'"
      >
        Moves
      </button>
      <button 
        :class="{ active: activeTab === 'parameters' }"
        @click="activeTab = 'parameters'"
      >
        Parameters
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
    </div>

    <!-- Controls Tab -->
    <div v-if="activeTab === 'controls'" class="tab-content">
      <div class="player-tabs">
          <button 
              v-for="playerId in userControlledPlayerIds" 
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

    <!-- Rewards Tab -->
    <div v-if="activeTab === 'rewards'" class="tab-content">
      <div class="rewards-section">
        <h4>Reward Parameters</h4>
        <div class="parameters-grid" v-if="rewardParams">
          <div class="param-category">
            <h5>Shot Rewards</h5>
            <div class="param-item"><span class="param-name">Made 2pt reward:</span><span class="param-value">{{ rewardParams.made_shot_reward_inside }}</span></div>
            <div class="param-item"><span class="param-name">Made 3pt reward:</span><span class="param-value">{{ rewardParams.made_shot_reward_three }}</span></div>
            <div class="param-item"><span class="param-name">Missed shot penalty:</span><span class="param-value">{{ rewardParams.missed_shot_penalty }}</span></div>
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
            <div class="param-item"><span class="param-name">Turnover penalty:</span><span class="param-value">{{ rewardParams.turnover_penalty }}</span></div>
            <div class="param-item"><span class="param-name">Violation reward:</span><span class="param-value">{{ rewardParams.violation_reward }}</span></div>
            <div class="param-item"><span class="param-name">Made shot reward inside:</span><span class="param-value">{{ rewardParams.made_shot_reward_inside }}</span></div>
            <div class="param-item"><span class="param-name">Made shot reward three:</span><span class="param-value">{{ rewardParams.made_shot_reward_three }}</span></div>
            <div class="param-item"><span class="param-name">Missed shot penalty:</span><span class="param-value">{{ rewardParams.missed_shot_penalty }}</span></div>
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
      <div class="rewards-section">
        <h4>Episode Stats</h4>
        <div class="parameters-grid">
          <div class="param-category">
            <h5>Totals</h5>
            <div class="param-item"><span class="param-name">Episodes played:</span><span class="param-value">{{ statsState.episodes }}</span></div>
            <div class="param-item"><span class="param-name">Total assists:</span><span class="param-value">{{ totalAssists }}</span></div>
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
          </div>
          <div class="param-category">
            <h5>2PT</h5>
            <div class="param-item"><span class="param-name">Attempts:</span><span class="param-value">{{ statsState.twoPt.attempts }}</span></div>
            <div class="param-item"><span class="param-name">Made:</span><span class="param-value">{{ statsState.twoPt.made }}</span></div>
            <div class="param-item"><span class="param-name">FG%:</span><span class="param-value">{{ (safeDiv(statsState.twoPt.made, Math.max(1, statsState.twoPt.attempts)) * 100).toFixed(1) }}%</span></div>
            <div class="param-item"><span class="param-name">Assists:</span><span class="param-value">{{ statsState.twoPt.assists }}</span></div>
          </div>
          <div class="param-category">
            <h5>3PT</h5>
            <div class="param-item"><span class="param-name">Attempts:</span><span class="param-value">{{ statsState.threePt.attempts }}</span></div>
            <div class="param-item"><span class="param-name">Made:</span><span class="param-value">{{ statsState.threePt.made }}</span></div>
            <div class="param-item"><span class="param-name">FG%:</span><span class="param-value">{{ (safeDiv(statsState.threePt.made, Math.max(1, statsState.threePt.attempts)) * 100).toFixed(1) }}%</span></div>
            <div class="param-item"><span class="param-name">Assists:</span><span class="param-value">{{ statsState.threePt.assists }}</span></div>
          </div>
        </div>
        <div style="display:flex; gap: 0.5rem;">
          <button class="new-game-button" @click="resetStats">Reset Stats</button>
          <button class="submit-button" @click="copyStatsMarkdown">Copy</button>
        </div>
      </div>
    </div>

    <!-- Moves Tab -->
    <div v-if="activeTab === 'moves'" class="tab-content">
      <div class="moves-section">
        <h4>Team Moves History ({{ props.gameState?.user_team_name || 'Unknown' }})</h4>
        <div v-if="props.moveHistory.length === 0" class="no-moves">
          No moves recorded yet.
        </div>
        <table v-else class="moves-table">
          <thead>
            <tr>
              <th>Turn</th>
              <th>Shot Clock</th>
              <th v-for="playerId in userControlledPlayerIds" :key="playerId">
                Player {{ playerId }}
              </th>
              <th>Off Value</th>
              <th>Def Value</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="move in props.moveHistory" :key="move.turn" :class="{ 'current-shot-clock-row': move.shotClock === props.currentShotClock || (move.isEndRow && props.gameState?.done) }">
              <td>{{ move.turn }}</td>
              <td class="shot-clock-cell">{{ move.shotClock !== undefined ? move.shotClock : '-' }}</td>
              <td v-for="playerId in userControlledPlayerIds" :key="playerId" class="move-cell">
                <div class="move-action">
                  <span v-if="move.ballHolder === playerId" class="ball-holder-icon">🏀 </span>{{ move.moves[`Player ${playerId}`] || 'NOOP' }}
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
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- Parameters Tab -->
    <div v-if="activeTab === 'parameters'" class="tab-content">
      <div class="parameters-section">
        <h4>MLflow Parameters</h4>
        <div v-if="!props.gameState" class="no-data">
          No game loaded
        </div>
        <div v-else class="parameters-grid">
          <div class="param-category">
            <h5>Environment Settings</h5>
            <div class="param-item">
              <span class="param-name">Players per side:</span>
              <span class="param-value">{{ Math.floor((props.gameState.offense_ids?.length || 0)) }}</span>
            </div>
            <div class="param-item">
              <span class="param-name">Court dimensions:</span>
              <span class="param-value">{{ props.gameState.court_width }}×{{ props.gameState.court_height }}</span>
            </div>
            <div class="param-item">
              <span class="param-name">Ball holder:</span>
              <span class="param-value">Player {{ props.gameState.ball_holder }}</span>
            </div>
            <div class="param-item">
              <span class="param-name">Shot clock:</span>
              <span class="param-value">{{ props.gameState.shot_clock }}</span>
            </div>
            <div class="param-item">
              <span class="param-name">Min shot clock at reset:</span>
              <span class="param-value">{{ props.gameState.min_shot_clock ?? 'N/A' }}</span>
            </div>
          </div>

          <div class="param-category">
            <h5>Policies</h5>
            <div class="param-item">
              <span class="param-name">Player 1 team:</span>
              <span class="param-value">{{ (props.gameState.user_team_name || 'OFFENSE') }} · {{ props.gameState.unified_policy_name || 'Latest unified' }}</span>
            </div>
            <div class="param-item">
              <span class="param-name">Opponent:</span>
              <span class="param-value">{{ (props.gameState.user_team_name === 'OFFENSE' ? 'DEFENSE' : 'OFFENSE') }} · {{ props.gameState.opponent_unified_policy_name || props.gameState.unified_policy_name || 'Same as unified' }}</span>
            </div>
          </div>

          <div class="param-category">
            <h5>Shot Parameters</h5>
            <div class="param-item">
              <span class="param-name">Three point distance:</span>
              <span class="param-value">{{ props.gameState.three_point_distance || 'N/A' }}</span>
            </div>
            <div v-if="props.gameState.shot_params" class="param-group">
              <div class="param-item">
                <span class="param-name">Layup mean:</span>
                <span class="param-value">{{ (props.gameState.shot_params.layup_pct * 100).toFixed(1) }}%</span>
              </div>
              <div class="param-item">
                <span class="param-name">Layup std:</span>
                <span class="param-value">{{ (props.gameState.shot_params.layup_std * 100).toFixed(1) }}%</span>
              </div>
              <div class="param-item">
                <span class="param-name">Three-point mean:</span>
                <span class="param-value">{{ (props.gameState.shot_params.three_pt_pct * 100).toFixed(1) }}%</span>
              </div>
              <div class="param-item">
                <span class="param-name">Three-point std:</span>
                <span class="param-value">{{ (props.gameState.shot_params.three_pt_std * 100).toFixed(1) }}%</span>
              </div>
              <div class="param-item">
                <span class="param-name">Dunk mean:</span>
                <span class="param-value">{{ (props.gameState.shot_params.dunk_pct * 100).toFixed(1) }}%</span>
              </div>
              <div class="param-item">
                <span class="param-name">Dunk std:</span>
                <span class="param-value">{{ (props.gameState.shot_params.dunk_std * 100).toFixed(1) }}%</span>
              </div>
              <div class="param-item">
                <span class="param-name">Dunks enabled:</span>
                <span class="param-value">{{ props.gameState.shot_params.allow_dunks ? 'Yes' : 'No' }}</span>
              </div>
            </div>
          </div>
          <div class="param-category" v-if="props.gameState.offense_shooting_pct_by_player">
            <h5>Sampled Player Skills (Offense)</h5>
            <div class="param-item" v-for="(pid, idx) in (props.gameState.offense_ids || [])" :key="`skill-${pid}`">
              <span class="param-name">Player {{ pid }}:</span>
              <span class="param-value">
                L {{ ((props.gameState.offense_shooting_pct_by_player.layup?.[idx] || 0) * 100).toFixed(1) }}% ·
                3 {{ ((props.gameState.offense_shooting_pct_by_player.three_pt?.[idx] || 0) * 100).toFixed(1) }}% ·
                D {{ ((props.gameState.offense_shooting_pct_by_player.dunk?.[idx] || 0) * 100).toFixed(1) }}%
              </span>
            </div>
          </div>
          <div class="param-category">
            <h5>Defender Turnover Pressure</h5>
            <div class="param-item">
              <span class="param-name">Pressure distance:</span>
              <span class="param-value">{{ props.gameState.defender_pressure_distance || 'N/A' }}</span>
            </div>
            <div class="param-item">
              <span class="param-name">Turnover chance:</span>
              <span class="param-value">{{ props.gameState.defender_pressure_turnover_chance || 'N/A' }}</span>
            </div>
            <div class="param-item">
              <span class="param-name">Decay lambda:</span>
              <span class="param-value">{{ props.gameState.defender_pressure_decay_lambda || 'N/A' }}</span>
            </div>
          </div>
          <div class="param-category">
            <h5>Pass Interception (Line-of-Sight)</h5>
            <div class="param-item">
              <span class="param-name">Base steal rate:</span>
              <span class="param-value">{{ props.gameState.base_steal_rate ?? 'N/A' }}</span>
            </div>
            <div class="param-item">
              <span class="param-name">Perpendicular decay:</span>
              <span class="param-value">{{ props.gameState.steal_perp_decay ?? 'N/A' }}</span>
            </div>
            <div class="param-item">
              <span class="param-name">Distance factor:</span>
              <span class="param-value">{{ props.gameState.steal_distance_factor ?? 'N/A' }}</span>
            </div>
            <div class="param-item">
              <span class="param-name">Position weight min:</span>
              <span class="param-value">{{ props.gameState.steal_position_weight_min ?? 'N/A' }}</span>
            </div>
          </div>
          <div class="param-category">
            <h5>Spawn Distance</h5>
            <div class="param-item">
              <span class="param-name">Min spawn distance:</span>
              <span class="param-value">{{ props.gameState.spawn_distance || 'N/A' }}</span>
            </div>
            <div class="param-item">
              <span class="param-name">Max spawn distance:</span>
              <span class="param-value">{{ props.gameState.max_spawn_distance ?? 'Unlimited' }}</span>
            </div>
          </div>
          <div class="param-category">
            <h5>Shot Pressure</h5>
            <div class="param-item">
              <span class="param-name">Pressure enabled:</span>
              <span class="param-value">{{ props.gameState.shot_pressure_enabled || 'N/A' }}</span>
            </div>
            <div class="param-item">
              <span class="param-name">Max pressure:</span>
              <span class="param-value">{{ props.gameState.shot_pressure_max || 'N/A' }}</span>
            </div>
            <div class="param-item">
              <span class="param-name">Pressure lambda:</span>
              <span class="param-value">{{ props.gameState.shot_pressure_lambda || 'N/A' }}</span>
            </div>
            <div class="param-item">
              <span class="param-name">Pressure arc degrees:</span>
              <span class="param-value">{{ props.gameState.shot_pressure_arc_degrees || 'N/A' }}°</span>
            </div>
          </div>

          <div class="param-category">
            <h5>Pass & Action Policy</h5>
            <div class="param-item">
              <span class="param-name">Pass arc degrees:</span>
              <span class="param-value">{{ props.gameState.pass_arc_degrees || 'N/A' }}°</span>
            </div>
            <div class="param-item">
              <span class="param-name">Pass OOB turnover prob:</span>
              <span class="param-value">{{ props.gameState.pass_oob_turnover_prob != null ? (props.gameState.pass_oob_turnover_prob * 100).toFixed(0) + '%' : 'N/A' }}</span>
            </div>
            <div class="param-item">
              <span class="param-name">Pass logit bias:</span>
              <span class="param-value">{{ props.gameState.pass_logit_bias != null ? props.gameState.pass_logit_bias.toFixed(2) : 'N/A' }}</span>
            </div>
            <div class="param-item">
              <span class="param-name">Illegal action policy:</span>
              <span class="param-value">{{ props.gameState.illegal_action_policy || 'N/A' }}</span>
            </div>
          </div>

          <div class="param-category">
            <h5>Team Configuration</h5>
            <div class="param-item">
              <span class="param-name">User team:</span>
              <span class="param-value">{{ props.gameState.user_team_name }}</span>
            </div>
            <div class="param-item">
              <span class="param-name">Offense IDs:</span>
              <span class="param-value">{{ props.gameState.offense_ids?.join(', ') || 'N/A' }}</span>
            </div>
            <div class="param-item">
              <span class="param-name">Defense IDs:</span>
              <span class="param-value">{{ props.gameState.defense_ids?.join(', ') || 'N/A' }}</span>
            </div>
          </div>

          <div class="param-category">
            <h5>3-Second Violation Rules</h5>
            <div class="param-item">
              <span class="param-name">Lane width (hexes):</span>
              <span class="param-value">{{ props.gameState.three_second_lane_width ?? 'N/A' }}</span>
            </div>
            <div class="param-item">
              <span class="param-name">Lane height (hexes):</span>
              <span class="param-value">{{ props.gameState.three_second_lane_height ?? 'N/A' }}</span>
            </div>
            <div class="param-item">
              <span class="param-name">Max steps in lane:</span>
              <span class="param-value">{{ props.gameState.three_second_max_steps ?? 'N/A' }}</span>
            </div>
            <div class="param-item">
              <span class="param-name">Offensive 3-sec enabled:</span>
              <span class="param-value">{{ props.gameState.offensive_three_seconds_enabled ? '✓ Yes' : '✗ No' }}</span>
            </div>
            <div class="param-item">
              <span class="param-name">Illegal defense enabled:</span>
              <span class="param-value">{{ props.gameState.illegal_defense_enabled ? '✓ Yes' : '✗ No' }}</span>
            </div>
            <div class="param-item" v-if="props.gameState.offensive_lane_hexes">
              <span class="param-name">Lane hexes count:</span>
              <span class="param-value">{{ props.gameState.offensive_lane_hexes?.length || 0 }} hexes</span>
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
                <td>O{{ Math.floor(idx / numDefenders) }} → D{{ idx % numDefenders }}</td>
                <td class="value-mono">{{ dist.toFixed(4) }}</td>
                <td class="notes">Hex distance</td>
              </tr>

              <!-- All-Pairs Angles -->
              <tr v-for="(angle, idx) in allPairsAngles" :key="`angle-${idx}`" class="group-angles">
                <td v-if="idx === 0" :rowspan="allPairsAngles.length" class="group-label">All-Pairs Angles (cos)</td>
                <td>O{{ Math.floor(idx / numDefenders) }} → D{{ idx % numDefenders }}</td>
                <td class="value-mono">{{ angle.toFixed(4) }}</td>
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
                <td>O{{ idx }}</td>
                <td class="value-mono">{{ ep.toFixed(4) }}</td>
                <td class="notes">Shot quality estimate</td>
              </tr>

              <!-- Turnover Probabilities -->
              <tr v-for="(prob, idx) in turnoverProbs" :key="`turnover-${idx}`" class="group-turnover">
                <td v-if="idx === 0" :rowspan="turnoverProbs.length" class="group-label">Turnover Probs</td>
                <td>O{{ idx }}</td>
                <td class="value-mono">{{ prob.toFixed(4) }}</td>
                <td v-if="prob > 0" class="notes highlight">🚨 Risk</td>
                <td v-else class="notes">No risk</td>
              </tr>

              <!-- Steal Risks -->
              <tr v-for="(risk, idx) in stealRisks" :key="`steal-${idx}`" class="group-steal">
                <td v-if="idx === 0" :rowspan="stealRisks.length" class="group-label">Steal Risks</td>
                <td>O{{ idx }}</td>
                <td class="value-mono">{{ risk.toFixed(4) }}</td>
                <td v-if="risk > 0" class="notes highlight">⚠️ Risk</td>
                <td v-else class="notes">Safe</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.player-controls-container {
  padding: 1rem;
  background-color: #f8f9fa;
  border-radius: 8px;
  display: flex;
  flex-direction: column;
  gap: 1rem;
  max-height: 80vh;
  overflow-y: auto;
}

.tab-navigation {
  display: flex;
  gap: 0.5rem;
  border-bottom: 1px solid #dee2e6;
  margin-bottom: 1rem;
}

.tab-navigation button {
  padding: 0.5rem 1rem;
  border: none;
  background-color: transparent;
  cursor: pointer;
  border-bottom: 2px solid transparent;
  font-weight: 500;
  transition: all 0.2s ease;
}

.tab-navigation button:hover {
  background-color: #e9ecef;
}

.tab-navigation button.active {
  border-bottom-color: #007bff;
  color: #007bff;
}

.tab-content {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

/* Existing styles */
.player-tabs {
  display: flex;
  gap: 0.5rem;
  flex-wrap: wrap;
}

.player-tabs button {
  padding: 0.5rem 1rem;
  border: 1px solid #ccc;
  background-color: white;
  cursor: pointer;
  border-radius: 4px;
  transition: all 0.2s ease;
}

.player-tabs button:hover {
  background-color: #e9ecef;
}

.player-tabs button.active {
  background-color: #007bff;
  color: white;
  border-color: #007bff;
}

.player-tabs button:disabled {
  background-color: #e9ecef;
  color: #6c757d;
  cursor: not-allowed;
  opacity: 0.6;
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
  background-color: white;
  border-radius: 4px;
  border: 1px solid #dee2e6;
}

.total-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.25rem;
}

.team-label {
  font-weight: bold;
  font-size: 0.9rem;
}

.team-label.offense {
  color: #dc3545;
}

.team-label.defense {
  color: #007bff;
}

.reward-value {
  font-size: 1.2rem;
  font-weight: bold;
}

.reward-history {
  max-height: 300px;
  overflow-y: auto;
}

.no-rewards {
  text-align: center;
  padding: 2rem;
  color: #6c757d;
  font-style: italic;
}

.reward-table {
  background-color: white;
  border-radius: 4px;
  border: 1px solid #dee2e6;
  overflow: hidden;
}

.reward-header {
  display: grid;
  grid-template-columns: 0.8fr 0.8fr 1fr 1.5fr 1fr 1.5fr;
  padding: 0.75rem;
  background-color: #f8f9fa;
  font-weight: bold;
  border-bottom: 1px solid #dee2e6;
  text-align: center;
}

.reward-table.with-phi .reward-header {
  grid-template-columns: 0.8fr 0.8fr 1fr 1.5fr 1fr 1.5fr 1fr;
}

.reward-row {
  display: grid;
  grid-template-columns: 0.8fr 0.8fr 1fr 1.5fr 1fr 1.5fr;
  padding: 0.5rem 0.75rem;
  border-bottom: 1px solid #f1f3f4;
  text-align: center;
}

.reward-table.with-phi .reward-row {
  grid-template-columns: 0.8fr 0.8fr 1fr 1.5fr 1fr 1.5fr 1fr;
}

.reward-row:last-child {
  border-bottom: none;
}

.reward-row:hover {
  background-color: #f8f9fa;
}

.reward-row.current-reward-row {
  background-color: #fff3cd !important;
  font-weight: 600;
  border-top: 2px solid #ffc107;
  border-bottom: 2px solid #ffc107;
}

.reward-row.current-reward-row:hover {
  background-color: #ffe69c !important;
}

.positive {
  color: #28a745;
  font-weight: bold;
}

.negative {
  color: #dc3545;
  font-weight: bold;
}

.reason-text {
  font-size: 0.85rem;
  color: #6c757d;
  font-style: italic;
}

/* Existing control styles */
.control-pad-wrapper {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 1rem;
}


.disabled {
  opacity: 0.7;
  pointer-events: none;
}

/* Moves styles */
.moves-section {
  padding: 1rem;
}

.moves-table {
  width: 100%;
  border-collapse: collapse;
  margin-top: 10px;
}

.moves-table th,
.moves-table td {
  border: 1px solid #ddd;
  padding: 8px;
  text-align: center;
}

.moves-table th {
  background-color: #f5f5f5;
  font-weight: bold;
}

.value-cell {
  font-family: 'Courier New', monospace;
  font-size: 0.9em;
  color: #333;
  font-weight: 500;
}

.moves-table tr:nth-child(even) {
  background-color: #f9f9f9;
}

.moves-table tr.current-shot-clock-row {
  background-color: #fff3cd !important;
  font-weight: 600;
}

.moves-table tr.current-shot-clock-row:hover {
  background-color: #ffe69c !important;
}

.moves-table tr.current-shot-clock-row td {
  border-top: 2px solid #ffc107;
  border-bottom: 2px solid #ffc107;
}

.moves-table tr.current-shot-clock-row td:first-child {
  border-left: 2px solid #ffc107;
}

.moves-table tr.current-shot-clock-row td:last-child {
  border-right: 2px solid #ffc107;
}

.move-cell {
  padding: 6px 8px;
}

.move-action {
  font-weight: 500;
  margin-bottom: 2px;
}

.ball-holder-icon {
  font-size: 1em;
}

.pass-steal-info {
  font-size: 0.8em;
  color: #dc3545;
  font-style: italic;
}

.defender-pressure-info {
  font-size: 0.8em;
  color: #ff6b35;
  font-style: italic;
}

.shot-clock-cell {
  font-weight: 600;
  color: #495057;
  background-color: #f8f9fa;
}

.no-moves {
  text-align: center;
  color: #666;
  font-style: italic;
  padding: 20px;
}

/* Parameters styles */
.parameters-section {
  padding: 1rem;
}

.parameters-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 1rem;
}

.param-category {
  background: #f8f9fa;
  border-radius: 6px;
  padding: 1rem;
}

.param-category h5 {
  margin: 0 0 0.5rem 0;
  color: #333;
  font-weight: 600;
  border-bottom: 1px solid #ddd;
  padding-bottom: 0.25rem;
}

.param-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.25rem 0;
  border-bottom: 1px solid #eee;
}

.param-item:last-child {
  border-bottom: none;
}

.param-name {
  font-weight: 500;
  color: #555;
}

.param-value {
  font-family: 'Courier New', monospace;
  background: #fff;
  padding: 0.2rem 0.4rem;
  border-radius: 3px;
  border: 1px solid #ddd;
  font-size: 0.9em;
}

/* Observation Tab Styles */
.observation-section {
  padding: 1rem;
}

.observation-table-wrapper {
  max-height: 600px;
  overflow-y: auto;
  border: 1px solid #dee2e6;
  border-radius: 4px;
}

.observation-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.9em;
  background: white;
}

.observation-table th {
  position: sticky;
  top: 0;
  background-color: #2c3e50;
  color: white;
  padding: 0.75rem;
  text-align: left;
  font-weight: 600;
  border-bottom: 2px solid #34495e;
  z-index: 10;
}

.observation-table td {
  padding: 0.6rem 0.75rem;
  border-bottom: 1px solid #ecf0f1;
}

.observation-table tbody tr:hover {
  background-color: #f8f9fa;
}

.group-label {
  font-weight: 600;
  background-color: #ecf0f1;
  color: #2c3e50;
  min-width: 140px;
}

.value-mono {
  font-family: 'Courier New', monospace;
  background: #f5f5f5;
  padding: 0.3rem 0.5rem;
  border-radius: 3px;
  font-weight: 500;
  color: #d35400;
}

.notes {
  font-size: 0.85em;
  color: #7f8c8d;
  font-style: italic;
}

.notes.highlight {
  color: #c0392b;
  font-weight: 600;
  font-style: normal;
}

.group-player-pos td:first-child { background-color: #e8f4f8; }
.group-ball-holder td:first-child { background-color: #fef5e7; }
.group-shot-clock td:first-child { background-color: #ebf5fb; }
.group-team-encoding td:first-child { background-color: #f0e6ff; }
.group-ball-handler-pos td:first-child { background-color: #fef9e7; }
.group-hoop td:first-child { background-color: #eafaf1; }
.group-distances td:first-child { background-color: #fdeef4; }
.group-angles td:first-child { background-color: #f4ecf7; }
.group-lane-steps td:first-child { background-color: #fef5e7; }
.group-ep td:first-child { background-color: #eafaf1; }
.group-turnover td:first-child { background-color: #fadbd8; }
.group-steal td:first-child { background-color: #fadbd8; }

.no-data {
  text-align: center;
  padding: 2rem;
  color: #7f8c8d;
  font-style: italic;
}
</style> 