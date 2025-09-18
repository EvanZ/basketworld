<script setup>
import { ref, computed, watch, onMounted } from 'vue';
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
  // When provided, overrides internal selections to reflect actual applied actions
  externalSelections: {
    type: Object,
    default: null,
  },
});

const emit = defineEmits(['actions-submitted', 'update:activePlayerId', 'play-again', 'self-play', 'move-recorded']);

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

// Move tracking is now handled by parent component

// --- Stats tracking (persistent across sessions) ---
const statsState = ref(loadStats());
function safeDiv(n, d) { return d > 0 ? (n / d) : 0; }
const totalAssists = computed(() => (statsState.value.dunk.assists + statsState.value.twoPt.assists + statsState.value.threePt.assists));
const ppp = computed(() => safeDiv(statsState.value.points, Math.max(1, statsState.value.episodes)));
const avgRewardPerEp = computed(() => safeDiv(statsState.value.rewardSum, Math.max(1, statsState.value.episodes)));
const avgEpisodeLen = computed(() => safeDiv(statsState.value.episodeStepsSum, Math.max(1, statsState.value.episodes)));

async function recordEpisodeStats(finalState) {
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
  try {
    const data = await getRewards();
    const ep = data?.episode_rewards || { offense: 0.0, defense: 0.0 };
    const userTeam = finalState?.user_team_name || 'OFFENSE';
    statsState.value.rewardSum += Number(userTeam === 'OFFENSE' ? ep.offense : ep.defense) || 0;
    const steps = Array.isArray(data?.reward_history) ? data.reward_history.length : 0;
    statsState.value.episodeStepsSum += Number(steps || 0);
  } catch (_) { /* ignore */ }

  // Increment episode count last
  statsState.value.episodes += 1;
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
defineExpose({ resetStats, copyStatsMarkdown, submitActions });

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
  console.log('[PlayerControls] Game state changed, fetching AI data... Ball holder:', newGameState?.ball_holder);
  if (newGameState && !newGameState.done) {
    // Fetch both action values and policy probabilities for AI mode
    try {
      console.log('[PlayerControls] Starting to fetch action values for ball holder:', newGameState.ball_holder);
      await fetchAllActionValues();
      console.log('[PlayerControls] Action values fetched, now fetching policy probabilities for ball holder:', newGameState.ball_holder);
      await fetchPolicyProbabilities();
      console.log('[PlayerControls] Policy probabilities fetch completed for ball holder:', newGameState.ball_holder);
    } catch (error) {
      console.error('[PlayerControls] Error during AI data fetch:', error);
    }
  } else {
    console.log('[PlayerControls] Clearing AI data - no game state or game done');
    actionValues.value = null;
    policyProbabilities.value = null;
    valueRange.value = { min: 0, max: 0 };
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

function triggerSelfPlay() {
  // Emit current selections so the parent can use them for the first self-play step
  const snapshot = { ...selectedActions.value };
  emit('self-play', snapshot);
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
    console.log('[Rewards] Fetched rewards. History length:', rewardHistory.value.length, 'Episode totals:', episodeRewards.value);
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

onMounted(() => {
  fetchRewards();
});

// Record stats once on episode completion
watch(() => props.gameState?.done, async (done, prevDone) => {
  if (done && !prevDone && props.gameState && !props.isReplaying) {
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

      <button @click="submitActions" class="submit-button" :disabled="gameState.done">
        {{ gameState.done ? 'Game Over' : 'Submit Turn' }}
      </button>
      
      <button 
        @click="triggerSelfPlay" 
        class="self-play-button"
        :disabled="!aiMode || gameState.done"
      >
        Self-Play
      </button>
      
      <button @click="$emit('play-again')" class="new-game-button">
        New Game
      </button>
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
          <div v-else class="reward-table">
            <div class="reward-header">
              <span>Turn</span>
              <span>Offense</span>
              <span>Off. Reason</span>
              <span>Defense</span>
              <span>Def. Reason</span>
            </div>
            <div 
              v-for="reward in rewardHistory" 
              :key="reward.step"
              class="reward-row"
            >
              <span>{{ reward.step }}</span>
              <span :class="{ positive: reward.offense > 0, negative: reward.offense < 0 }">
                {{ reward.offense.toFixed(2) }}
              </span>
              <span class="reason-text">{{ reward.offense_reason }}</span>
              <span :class="{ positive: reward.defense > 0, negative: reward.defense < 0 }">
                {{ reward.defense.toFixed(2) }}
              </span>
              <span class="reason-text">{{ reward.defense_reason }}</span>
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
              <th v-for="playerId in userControlledPlayerIds" :key="playerId">
                Player {{ playerId }}
              </th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="move in props.moveHistory" :key="move.turn">
              <td>{{ move.turn }}</td>
              <td v-for="playerId in userControlledPlayerIds" :key="playerId">
                {{ move.moves[`Player ${playerId}`] || 'NOOP' }}
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
          </div>
          <div class="param-category">
            <h5>Pass Interception</h5>
            <div class="param-item">
              <span class="param-name">Steal chance:</span>
              <span class="param-value">{{ props.gameState.steal_chance ?? 'N/A' }}</span>
            </div>
          </div>
          <div class="param-category">
            <h5>Spawn Distance</h5>
            <div class="param-item">
              <span class="param-name">Spawn distance:</span>
              <span class="param-value">{{ props.gameState.spawn_distance || 'N/A' }}</span>
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
  grid-template-columns: 1fr 1fr 1.5fr 1fr 1.5fr;
  padding: 0.75rem;
  background-color: #f8f9fa;
  font-weight: bold;
  border-bottom: 1px solid #dee2e6;
  text-align: center;
}

.reward-row {
  display: grid;
  grid-template-columns: 1fr 1fr 1.5fr 1fr 1.5fr;
  padding: 0.5rem 0.75rem;
  border-bottom: 1px solid #f1f3f4;
  text-align: center;
}

.reward-row:last-child {
  border-bottom: none;
}

.reward-row:hover {
  background-color: #f8f9fa;
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

.submit-button, .new-game-button {
  padding: 0.75rem 1.5rem;
  font-size: 1rem;
  font-weight: bold;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  transition: background-color 0.2s ease;
  min-width: 120px;
}

.submit-button {
  background-color: #28a745;
  color: white;
}

.submit-button:disabled {
  background-color: #6c757d;
  cursor: not-allowed;
}

.submit-button:hover:not(:disabled) {
  background-color: #218838;
}

.new-game-button {
  background-color: #007bff;
  color: white;
}

.new-game-button:hover {
  background-color: #0056b3;
}

.self-play-button {
  background-color: #6c757d;
  color: white;
  padding: 0.75rem 1.5rem;
  font-size: 1rem;
  font-weight: bold;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  transition: background-color 0.2s ease;
  min-width: 120px;
}

.self-play-button:hover:not(:disabled) {
  background-color: #5a6268;
}

.self-play-button:disabled {
  background-color: #e9ecef;
  color: #6c757d;
  cursor: not-allowed;
  opacity: 0.6;
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

.moves-table tr:nth-child(even) {
  background-color: #f9f9f9;
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
</style> 