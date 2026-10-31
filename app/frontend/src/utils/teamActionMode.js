export function policyTeamActionMode(gameState) {
  const candidates = [
    gameState?.team_action_mode,
    gameState?.model_metadata?.policy_spec?.team_action_mode,
    gameState?.training_params?.team_action_mode,
  ];
  for (const candidate of candidates) {
    const normalized = String(candidate ?? '').trim().toLowerCase();
    if (normalized) return normalized;
  }
  return 'independent';
}

export function usesBackendJointPolicyActions(gameState) {
  return policyTeamActionMode(gameState) === 'autoregressive';
}
