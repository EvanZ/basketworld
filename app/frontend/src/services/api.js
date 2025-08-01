const API_BASE_URL = 'http://localhost:8000';

export async function initGame(runId, userTeamName) {
    const response = await fetch(`${API_BASE_URL}/api/init_game`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            run_id: runId,
            user_team_name: userTeamName,
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