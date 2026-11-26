export function scoringMetricGroup(name) {
  const key = name.replace(/^jax\/[^/]+\//, '')
  if (/shot_defense_|^(blocks|learner_blocks|opponent_blocks|blocks_per_completed_possession)$/.test(key)) return 'Shot defense and blocks'
  if (/_points_per_completed_episode$|^(?:offense|defense)_(?:learner|opponent)_offensive_efficiency$|^mean_(?:final_)?(?:offense_score|defense_score|score_margin)$/.test(key)) return 'Legacy scoring (schema 1)'
  if (/(?:^|_)(?:offense|defense|learner|opponent)_points_(?:total|per_step)$/.test(key)) return 'Legacy scoring (schema 1)'
  if (/roster_(?:slot|assignment)|started_with_possession|opening_possession|regulation_segment_\d|(?:^|_)team_[ab]_(?:points|score)|^clock_team_[ab]/.test(key)) return 'Roster and opening-possession diagnostics'
  if (name.startsWith('jax/eval_deploy/') && /^(self_play_|completed_game_count|truncated_game_count)/.test(key)) return 'Self-play behavior'
  if (/^(learner_(?:points_|score_|completed_game_|completed_(?:offensive|defensive)_possessions)|completed_game_count|truncated_game_count)/.test(key)) return name.startsWith('jax/eval/') ? 'Learner evaluation' : 'Learner scoring'
  if (name.startsWith('jax/eval_historical/') || /\/pfsp\//.test(name)) return 'Opponent comparisons'
  return 'RL health and other diagnostics'
}

export function scoringMetricDefinition(name) {
  const group = scoringMetricGroup(name)
  if (group === 'Shot defense and blocks') return 'Shot-release diagnostics across observed active steps, including partial games. Blocks are credited to the defending team; shot-defense rows are grouped by shooting team. Eligible attempts are finishes with an adjacent defender while rim protection is enabled. Rates with no opportunities are unavailable. Open, conditional (if unblocked), and marginal make probabilities are reported separately.'
  if (group.startsWith('Legacy')) return 'Legacy scoring schema: historical roster-lane metrics may mix partial-game points with completed-game denominators. Retained for history only; not comparable to schema 2 scoring.'
  if (group.startsWith('Roster')) return 'Fairness diagnostic. Roster assignment and opening possession are separate axes; these are not offense/defense performance roles. Margins and win scores use completed games only; legacy raw totals are unnormalized.'
  if (name.startsWith('jax/train/learner_') && /_possession$|_observed$/.test(name)) return 'Observed-rollout PPP: sum actual scoreboard points (including violation awards), then divide by completed possessions across both learner roster assignments. Includes ongoing and truncated games. Zero possessions means unavailable.'
  if (group === 'Self-play behavior') return 'Same-policy self-play behavior, not strength against opponents. Scores and PPP include completed games only; PPP combines both teams’ points and completed offensive possessions before division. See completed and truncated game counts.'
  if (group === 'Learner scoring' || group === 'Learner evaluation') return 'Completed-game terminal scoreboard outcomes across both roster assignments. Partial and truncated game scores are excluded. Win rate counts wins only; win score gives ties one half. See completed_game_count; zero completed games means unavailable.'
  if (/reward/.test(name)) return 'Policy-learning reward, which may contain shaping and bonuses. This is not basketball points or PPP.'
  return ''
}
