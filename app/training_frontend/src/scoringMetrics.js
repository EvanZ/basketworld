export const ROSTER_METRICS = ['roster_persistent_persistent_pct', 'roster_mixed_pct', 'roster_fresh_fresh_pct', 'roster_pool_coverage_pct'].map(name => 'jax/train/' + name)

const compactBlockMetric = /^(?:(?:learner|opponent)_)?(?:block_pct|blocks_per_defensive_possession|blocks_per_possession|blocks_per_episode)$/
const legacyBlockMetric = /shot_defense_|^(blocks|learner_blocks|opponent_blocks|blocks_per_completed_possession)$/

export const RIM_CONTEST_FREQUENCY_METRICS = [
  'jax/train/learner_', 'jax/train/opponent_', 'jax/eval_deploy/',
].flatMap(prefix => ['rim_contested_shots_per_episode', 'rim_contested_shot_pct'].map(suffix => prefix + suffix))

export const LAYUP_METRICS = [
  'jax/train/learner_', 'jax/train/opponent_', 'jax/eval_deploy/',
].flatMap(prefix => ['layup_attempts_per_episode', 'layup_makes_per_episode', 'layup_fg_pct'].map(suffix => prefix + suffix))

export function isLegacyLayupMetric(name) {
  return /^jax\/(?:train|eval_offense|eval_defense|eval_deploy)\/(?:learner|opponent|all)_shot_defense_(?:open_|contested_)?layup_(?:attempts|makes|make_rate)$/.test(name)
}

export function metricPickerCatalog(loggedMetrics) {
  // Approved metrics can be pinned before their first observation. Never fill
  // missing measurements with zero or mark them as logged.
  const catalog = new Map([...RIM_CONTEST_FREQUENCY_METRICS, ...LAYUP_METRICS, ...ROSTER_METRICS].map(name => [name, { name, latest: null, logged: false }]))
  for (const metric of loggedMetrics) catalog.set(metric.name, { ...metric, logged: true })
  return [...catalog.values()].sort((a, b) => a.name.localeCompare(b.name))
}

export function scoringMetricGroup(name) {
  const key = name.replace(/^jax\/[^/]+\//, '')
  if (ROSTER_METRICS.includes(name)) return 'RL health and other diagnostics'
  if (compactBlockMetric.test(key) || legacyBlockMetric.test(key)) return 'RL health and other diagnostics'
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
  const key = name.replace(/^jax\/[^/]+\//, '')
  if (ROSTER_METRICS.includes(name)) {
    if (key === 'roster_pool_coverage_pct') return 'Distinct league teams selected on either side since the previous log, divided by all teams in the selected league, times 100. Each team counts once; interval coverage, not cumulative.'
    const kind = key.includes('persistent_persistent') ? 'two league teams' : key.includes('mixed') ? 'one league team and one fresh team' : 'two fresh teams'
    return `Newly assigned games with ${kind}, divided by all newly assigned games since the previous log, times 100. Assignments, not completed games or possessions. No new assignments means unavailable.`
  }
  if (LAYUP_METRICS.includes(name)) {
    const deploy = name.startsWith('jax/eval_deploy/')
    const definition = key.endsWith('fg_pct')
      ? `${deploy ? 'Both teams pooled.' : 'Credited to the named shooting team.'} Layup makes divided by layup attempts, multiplied by 100. No attempts means unavailable.`
      : `${deploy ? 'Both teams’ totals divided by twice the completed game count: average per team per game.' : 'The named shooting team’s total divided by completed games.'} Counts layup ${key.includes('attempts') ? 'attempts' : 'makes'}. No completed games means unavailable.`
    return `${definition} Uses completed games only, excluding partial and truncated games. Blocked layups count as missed attempts; dunks and jump shots are excluded. Training pools games completed since the previous log; deployment pools its evaluation batch.`
  }
  if (isLegacyLayupMetric(name)) return 'Historical layup diagnostics from observed steps, including partial games. No longer logged; retained for history. Use the completed-game layup attempts/makes per episode and layup FG% metrics.'
  if (/^(?:(?:learner|opponent)_)?rim_contested_(?:shots_per_episode|shot_pct)$/.test(key)) {
    const deploy = name.startsWith('jax/eval_deploy/')
    const definition = key.endsWith('shot_pct')
      ? `${deploy ? 'Both teams pooled.' : 'Credited to the named defending team.'} Rim-contested attempts divided by all opposing layup/dunk attempts, multiplied by 100. No layup/dunk attempts means unavailable.`
      : deploy
        ? 'Both teams’ rim-contested attempts divided by twice the completed game count: average per team per game. No completed games means unavailable.'
        : 'Layup/dunk attempts contested by the named defending team divided by completed episodes. No completed episodes means unavailable.'
    return `${definition} A rim contest requires an adjacent eligible defender with rim protection enabled; blocked attempts count. Uses completed games only, excluding partial and truncated games. Training pools games completed since the previous log; deployment pools its evaluation batch.`
  }
  if (/^(?:(?:learner|opponent)_)?rim_contest_mean_reduction_pp$/.test(key)) {
    const side = name.startsWith('jax/eval_deploy/') ? 'Both teams combined.' : 'Credited to the named defending team.'
    return `${side} Average percentage-point reduction from open-shot to conditional (if unblocked) make probability on contested layups and dunks with rim protection enabled. For example, 80% to 60% is 20 percentage points. Includes blocked attempts but excludes the block probability effect. Uses completed possessions; no qualifying attempts means unavailable.`
  }
  if (compactBlockMetric.test(key)) {
    const side = name.startsWith('jax/eval_deploy/') ? 'Both teams combined.' : 'Credited to the named defending team.'
    if (key.endsWith('block_pct')) return `${side} Blocks divided by opposing layup + dunk attempts, multiplied by 100. Includes open and contested finishes; excludes jump shots. Both counts use completed possessions. No attempts means unavailable.`
    if (key.endsWith('blocks_per_episode')) return `${side} Blocks per completed episode/game, including all possessions in that game. Partial and truncated games are excluded. No completed games means unavailable.`
    return `${side} Blocks divided by completed defensive possessions (all possessions for combined deploy). Both counts use the same completed possessions. No completed possessions means unavailable.`
  }
  if (legacyBlockMetric.test(key)) return 'Shot-release diagnostics across observed active steps, including partial games. Blocks are credited to the defending team; shot-defense rows are grouped by shooting team. Eligible attempts are finishes with an adjacent defender while rim protection is enabled. Rates with no opportunities are unavailable. Open, conditional (if unblocked), and marginal make probabilities are reported separately.'
  if (group.startsWith('Legacy')) return 'Legacy scoring schema: historical roster-lane metrics may mix partial-game points with completed-game denominators. Retained for history only; not comparable to schema 2 scoring.'
  if (group.startsWith('Roster')) return 'Fairness diagnostic. Roster assignment and opening possession are separate axes; these are not offense/defense performance roles. Margins and win scores use completed games only; legacy raw totals are unnormalized.'
  if (name.startsWith('jax/train/learner_') && /_possession$|_observed$/.test(name)) return 'Observed-rollout PPP: sum actual scoreboard points (including violation awards), then divide by completed possessions across both learner roster assignments. Includes ongoing and truncated games. Zero possessions means unavailable.'
  if (group === 'Self-play behavior') return 'Same-policy self-play behavior, not strength against opponents. Scores and PPP include completed games only; PPP combines both teams’ points and completed offensive possessions before division. See completed and truncated game counts.'
  if (group === 'Learner scoring' || group === 'Learner evaluation') return 'Completed-game terminal scoreboard outcomes across both roster assignments. Partial and truncated game scores are excluded. Win rate counts wins only; win score gives ties one half. See completed_game_count; zero completed games means unavailable.'
  if (/reward/.test(name)) return 'Policy-learning reward, which may contain shaping and bonuses. This is not basketball points or PPP.'
  return ''
}
