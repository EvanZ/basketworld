// Determine the backend base URL.
// Priority:
//  1. Vite env variable VITE_API_BASE_URL (e.g., set in .env or at build time)
//  2. Default to localhost:8080 (common alternative when 8000 is occupied)
//     You can of course change this to any port that matches your FastAPI server.

const API_BASE_URL = import.meta.env?.VITE_API_BASE_URL || 'http://localhost:8080';

export async function initGame(runId, userTeamName, offensePolicyName = null, defensePolicyName = null) {
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
        }),
    });
    if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Failed to parse error from initGame' }));
        console.error('[API] initGame failed:', response.status, errorData);
        throw new Error(errorData.detail || 'Failed to initialize game');
    }
    return response.json();
}

export async function stepGame(actions) {
    console.log('[API] Sending step request with actions:', actions);
    const response = await fetch(`${API_BASE_URL}/api/step`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ actions }),
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