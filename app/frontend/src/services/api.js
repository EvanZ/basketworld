// Determine the backend base URL.
// Priority:
//  1. Vite env variable VITE_API_BASE_URL (e.g., set in .env or at build time)
//  2. Default to localhost:8080 (common alternative when 8000 is occupied)
//     You can of course change this to any port that matches your FastAPI server.

const API_BASE_URL = import.meta.env?.VITE_API_BASE_URL || 'http://localhost:8080';

export async function initGame(runId, userTeamName, offensePolicyName = null, defensePolicyName = null, unifiedPolicyName = null, opponentUnifiedPolicyName = null) {
    const response = await fetch(`${API_BASE_URL}/api/init_game`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            run_id: runId,
            user_team_name: userTeamName,
            offense_policy_name: offensePolicyName,
            defense_policy_name: defensePolicyName,
            unified_policy_name: unifiedPolicyName,
            opponent_unified_policy_name: opponentUnifiedPolicyName,
        }),
    });
    if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Failed to parse error from initGame' }));
        console.error('[API] initGame failed:', response.status, errorData);
        throw new Error(errorData.detail || 'Failed to initialize game');
    }
    return response.json();
}

export async function stepGame(actions, playerDeterministic = null, opponentDeterministic = null) {
    console.log('[API] Sending step request with actions:', actions, 'playerDeterministic:', playerDeterministic, 'opponentDeterministic:', opponentDeterministic);
    const response = await fetch(`${API_BASE_URL}/api/step`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
            actions, 
            player_deterministic: playerDeterministic,
            opponent_deterministic: opponentDeterministic 
        }),
    });
    if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Failed to take step' }));
        console.error('[API] stepGame failed:', response.status, errorData);
        throw new Error(errorData.detail || 'Failed to take step');
    }
    return response.json();
}

export async function getPolicyProbs() {
    const response = await fetch(`${API_BASE_URL}/api/policy_probabilities`);
    if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Failed to fetch policy probabilities' }));
        console.error('[API] getPolicyProbs failed:', response.status, errorData);
        throw new Error(errorData.detail || 'Failed to fetch policy probabilities');
    }
    return response.json();
}

export async function getActionValues(playerId) {
    const response = await fetch(`${API_BASE_URL}/api/action_values/${playerId}`);
    if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Failed to fetch action values' }));
        console.error('[API] getActionValues failed:', response.status, errorData);
        throw new Error(errorData.detail || 'Failed to fetch action values');
    }
    return response.json();
}

export const getShotProbability = async (playerId) => {
  const url = `${API_BASE_URL}/api/shot_probability/${playerId}`;
  console.log('[API] getShotProbability →', url);
  const response = await fetch(url);
  if (!response.ok) {
    const text = await response.text().catch(() => '');
    console.error('[API] getShotProbability failed:', response.status, response.statusText, text);
    throw new Error(`Failed to get shot probability: ${response.statusText}`);
  }
  const json = await response.json();
  console.log('[API] getShotProbability ←', json);
  return json;
};

export const getPassStealProbabilities = async () => {
  const url = `${API_BASE_URL}/api/pass_steal_probabilities`;
  console.log('[API] getPassStealProbabilities →', url);
  const response = await fetch(url);
  if (!response.ok) {
    const text = await response.text().catch(() => '');
    console.error('[API] getPassStealProbabilities failed:', response.status, response.statusText, text);
    throw new Error(`Failed to get pass steal probabilities: ${response.statusText}`);
  }
  const json = await response.json();
  console.log('[API] getPassStealProbabilities ←', json);
  return json;
};

export const getStateValues = async () => {
  const url = `${API_BASE_URL}/api/state_values`;
  console.log('[API] getStateValues →', url);
  const response = await fetch(url);
  if (!response.ok) {
    const text = await response.text().catch(() => '');
    console.error('[API] getStateValues failed:', response.status, response.statusText, text);
    throw new Error(`Failed to get state values: ${response.statusText}`);
  }
  const json = await response.json();
  console.log('[API] getStateValues ←', json);
  return json;
};

export const getRewards = async () => {
  const response = await fetch(`${API_BASE_URL}/api/rewards`);
  if (!response.ok) {
    throw new Error(`Failed to get rewards: ${response.statusText}`);
  }
  return response.json();
};

export async function saveEpisode() {
    const response = await fetch(`${API_BASE_URL}/api/save_episode`, {
        method: 'POST'
    });
    if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Failed to save episode' }));
        console.error('[API] saveEpisode failed:', response.status, errorData);
        throw new Error(errorData.detail || 'Failed to save episode');
    }
    return response.json();
}

export async function startSelfPlay() {
    const response = await fetch(`${API_BASE_URL}/api/start_self_play`, {
        method: 'POST'
    });
    if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Failed to start self-play' }));
        console.error('[API] startSelfPlay failed:', response.status, errorData);
        throw new Error(errorData.detail || 'Failed to start self-play');
    }
    return response.json();
}

export async function replayLastEpisode() {
    const response = await fetch(`${API_BASE_URL}/api/replay_last_episode`, {
        method: 'POST'
    });
    if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Failed to replay last episode' }));
        console.error('[API] replayLastEpisode failed:', response.status, errorData);
        throw new Error(errorData.detail || 'Failed to replay last episode');
    }
    return response.json();
}

export async function listPolicies(runId) {
    const response = await fetch(`${API_BASE_URL}/api/list_policies`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ run_id: runId }),
    });
    if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Failed to list policies' }));
        console.error('[API] listPolicies failed:', response.status, errorData);
        throw new Error(errorData.detail || 'Failed to list policies');
    }
    return response.json();
} 

// --- Phi Shaping API ---
export async function getPhiParams() {
  const response = await fetch(`${API_BASE_URL}/api/phi_params`);
  if (!response.ok) throw new Error('Failed to get phi params');
  return response.json();
}

export async function setPhiParams(payload) {
  const response = await fetch(`${API_BASE_URL}/api/phi_params`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload || {}),
  });
  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.detail || 'Failed to set phi params');
  }
  return response.json();
}

export async function getPhiLog() {
  const response = await fetch(`${API_BASE_URL}/api/phi_log`);
  if (!response.ok) throw new Error('Failed to get phi log');
  return response.json();
}

export async function runEvaluation(numEpisodes = 100, playerDeterministic = false, opponentDeterministic = true) {
  const response = await fetch(`${API_BASE_URL}/api/run_evaluation`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ 
      num_episodes: numEpisodes, 
      player_deterministic: playerDeterministic,
      opponent_deterministic: opponentDeterministic
    }),
  });
  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.detail || 'Failed to run evaluation');
  }
  return response.json();
}

export async function updatePlayerPosition(playerId, q, r) {
  const response = await fetch(`${API_BASE_URL}/api/update_player_position`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ player_id: playerId, q, r }),
  });
  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.detail || 'Failed to update player position');
  }
  return response.json();
}

export async function batchUpdatePlayerPositions(updates) {
  const response = await fetch(`${API_BASE_URL}/api/batch_update_player_positions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ updates }),
  });
  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.detail || 'Failed to batch update player positions');
  }
  return response.json();
}

export async function setShotClock(delta) {
  const response = await fetch(`${API_BASE_URL}/api/set_shot_clock`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ delta }),
  });
  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.detail || 'Failed to set shot clock');
  }
  return response.json();
}

export async function resetTurnState() {
  const response = await fetch(`${API_BASE_URL}/api/reset_turn_state`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
  });
  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.detail || 'Failed to reset turn state');
  }
  return response.json();
}
