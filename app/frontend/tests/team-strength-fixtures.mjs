// Small deterministic study/game for unit tests and the browser harness.
export function teamStrengthFixture(size = 4) {
  const report = {
    summary: { checkpoint: '/models/example/update_2500', checkpoint_update: 2500, action_mode: 'direct', study_id: 'test-study', path: 'test-study' },
    reproducibility: { team_size: size },
    player_rating_fit: { feature_definition: 'sum(team_a_player_skills) - sum(team_b_player_skills)' },
    skill_descriptors: [{ key: 'three_pt', display_label: 'Three point' }, { key: 'rebound', display_label: 'Rebounding' }],
    skill_coefficients: [
      { key: 'three_pt', label: 'Three point', pool_mean: .4, pool_standard_deviation: .1, net_rating_per_standard_deviation: 2 },
      { key: 'rebound', label: 'Rebounding', pool_mean: 0, pool_standard_deviation: 2, net_rating_per_standard_deviation: 6 },
    ],
    player_rows: [],
    three_factors: {},
  };
  const metadata = { checkpoint_path: '/models/example/update_2500', update_index: 2500 };
  const state = {
    enable_multi_possession: true,
    model_metadata: { ...metadata }, opponent_model_metadata: { ...metadata },
    team_a_ids: Array.from({ length: size }, (_, i) => i),
    team_b_ids: Array.from({ length: size }, (_, i) => i + size),
    player_episode_skills: {}, player_shooting_skills: {}, player_rebound_skills: {},
  };
  for (let id = 0; id < size * 2; id += 1) {
    state.player_shooting_skills[id] = { three_pt: id < size ? .5 : .3 };
    state.player_rebound_skills[id] = id < size ? 1 : 0;
  }
  return { report, state };
}
