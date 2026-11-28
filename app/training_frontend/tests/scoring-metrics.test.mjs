import test from 'node:test'
import assert from 'node:assert/strict'
import { scoringMetricGroup, scoringMetricDefinition, metricPickerCatalog, RIM_CONTEST_FREQUENCY_METRICS, LAYUP_METRICS, ROSTER_METRICS, isLegacyLayupMetric } from '../src/scoringMetrics.js'

test('primary scoring separates learner outcomes, self-play, rewards, and fairness', () => {
  assert.equal(scoringMetricGroup('jax/train/learner_points_scored_per_completed_game'), 'Learner scoring')
  assert.equal(scoringMetricGroup('jax/eval/learner_completed_game_win_score'), 'Learner evaluation')
  assert.equal(scoringMetricGroup('jax/eval_deploy/self_play_points_per_team_per_completed_game'), 'Self-play behavior')
  assert.equal(scoringMetricGroup('jax/train/learner_roster_assignment_b_completed_game_count'), 'Roster and opening-possession diagnostics')
  assert.equal(scoringMetricGroup('jax/train/opponent_started_with_possession_win_score'), 'Roster and opening-possession diagnostics')
  assert.equal(scoringMetricGroup('jax/train/game_reward_mean'), 'RL health and other diagnostics')
})

test('league sampling metrics belong to the normally visible chart group', () => {
  assert.equal(ROSTER_METRICS.length, 4)
  for (const name of ROSTER_METRICS) {
    assert.equal(scoringMetricGroup(name), 'RL health and other diagnostics')
    assert.ok(metricPickerCatalog([]).some(metric => metric.name === name))
  }
})
test('old charts remain explicitly legacy and ratios describe their denominators', () => {
  for (const name of ['jax/train/offense_learner_points_total', 'jax/train/ppo_defense_opponent_points_per_step', 'jax/train/offense_learner_points_per_completed_episode', 'jax/eval_deploy/mean_offense_score', 'jax/eval_defense/mean_final_score_margin']) {
    assert.equal(scoringMetricGroup(name), 'Legacy scoring (schema 1)')
    assert.match(scoringMetricDefinition(name), /not comparable/)
  }
  assert.match(scoringMetricDefinition('jax/train/learner_points_scored_per_completed_offensive_possession'), /ongoing and truncated/)
  assert.match(scoringMetricDefinition('jax/eval_deploy/self_play_points_per_completed_offensive_possession'), /completed games only/)
  assert.match(scoringMetricDefinition('jax/train/game_reward_mean'), /not basketball points/)
})

test('block diagnostics retain defensive credit and explicit opportunity denominators', () => {
  for (const name of ['jax/train/blocks', 'jax/eval/learner_blocks', 'jax/eval_deploy/all_shot_defense_blocks_per_eligible_attempt']) {
    assert.equal(scoringMetricGroup(name), 'RL health and other diagnostics')
    assert.match(scoringMetricDefinition(name), /credited to the defending team/)
    assert.match(scoringMetricDefinition(name), /finishes with an adjacent defender/)
  }
})

test('compact block metrics explain finish percentage and completed denominators', () => {
  for (const prefix of ['jax/train/learner_', 'jax/train/opponent_', 'jax/eval_deploy/']) {
    assert.equal(scoringMetricGroup(`${prefix}block_pct`), 'RL health and other diagnostics')
    assert.match(scoringMetricDefinition(`${prefix}block_pct`), /layup \+ dunk attempts, multiplied by 100/)
    assert.match(scoringMetricDefinition(`${prefix}block_pct`), /completed possessions/)
    assert.match(scoringMetricDefinition(`${prefix}blocks_per_episode`), /Partial and truncated games are excluded/)
  }
  assert.match(scoringMetricDefinition('jax/train/learner_blocks_per_defensive_possession'), /completed defensive possessions/)
  assert.match(scoringMetricDefinition('jax/eval_deploy/blocks_per_possession'), /Both teams combined/)
})

test('rim contest metrics use the existing group and explain percentage points', () => {
  for (const prefix of ['jax/train/learner_', 'jax/train/opponent_', 'jax/eval_deploy/']) {
    const name = `${prefix}rim_contest_mean_reduction_pp`
    assert.equal(scoringMetricGroup(name), 'RL health and other diagnostics')
    assert.match(scoringMetricDefinition(name), /80% to 60% is 20 percentage points/)
    assert.match(scoringMetricDefinition(name), /Includes blocked attempts but excludes the block probability effect/)
  }
})

test('rim contest frequency uses finishes and completed games with per-team deploy counts', () => {
  for (const prefix of ['jax/train/learner_', 'jax/train/opponent_', 'jax/eval_deploy/']) {
    for (const suffix of ['rim_contested_shots_per_episode', 'rim_contested_shot_pct']) {
      const name = `${prefix}${suffix}`
      assert.equal(scoringMetricGroup(name), 'RL health and other diagnostics')
      assert.match(scoringMetricDefinition(name), /completed games only/)
      assert.match(scoringMetricDefinition(name), /blocked attempts count/)
    }
    assert.match(scoringMetricDefinition(`${prefix}rim_contested_shot_pct`), /all opposing layup\/dunk attempts/)
  }
  assert.match(scoringMetricDefinition('jax/eval_deploy/rim_contested_shots_per_episode'), /twice the completed game count/)
})

test('approved frequency and layup metrics are selectable before logging without fabricated values', () => {
  const pending = metricPickerCatalog([])
  assert.equal(pending.length, 19)
  assert.deepEqual(new Set(pending.map(metric => metric.name)), new Set([...RIM_CONTEST_FREQUENCY_METRICS, ...LAYUP_METRICS, ...ROSTER_METRICS]))
  assert.ok(pending.every(metric => metric.latest === null && metric.logged === false))
  const name = 'jax/train/learner_rim_contested_shot_pct'
  const updated = metricPickerCatalog([{ name, latest: 0 }, { name: 'jax/train/entropy_coef', latest: .01 }])
  assert.equal(updated.length, 20)
  assert.deepEqual(updated.find(metric => metric.name === name), { name, latest: 0, logged: true })
  assert.equal(updated.filter(metric => !metric.logged).length, 18)
})


test('nine layup metrics explain shooting credit, game counts, and percentages', () => {
  assert.equal(LAYUP_METRICS.length, 9)
  for (const name of LAYUP_METRICS) {
    assert.equal(scoringMetricGroup(name), 'RL health and other diagnostics')
    assert.equal(isLegacyLayupMetric(name), false)
    const definition = scoringMetricDefinition(name)
    assert.match(definition, /completed games only/)
    assert.match(definition, /Blocked layups count as missed attempts/)
    assert.match(definition, /dunks and jump shots are excluded/)
    if (name.endsWith('fg_pct')) assert.match(definition, /makes divided by layup attempts, multiplied by 100/)
    else if (name.startsWith('jax/eval_deploy/')) assert.match(definition, /twice the completed game count/)
    else assert.match(definition, /named shooting team/)
    assert.deepEqual(metricPickerCatalog([{name, latest: 0}]).find(metric => metric.name === name), {name, latest: 0, logged: true})
  }
})

test('all 108 historical layup diagnostics can be hidden without losing their history', () => {
  let count = 0
  for (const context of ['train', 'eval_offense', 'eval_defense', 'eval_deploy']) {
    for (const side of ['learner', 'opponent', 'all']) {
      for (const kind of ['', 'open_', 'contested_']) {
        for (const suffix of ['attempts', 'makes', 'make_rate']) {
          const name = `jax/${context}/${side}_shot_defense_${kind}layup_${suffix}`
          assert.ok(isLegacyLayupMetric(name))
          assert.match(scoringMetricDefinition(name), /retained for history/)
          assert.ok(metricPickerCatalog([{name, latest: 2}]).some(metric => metric.name === name && metric.logged))
          count++
        }
      }
    }
  }
  assert.equal(count, 108)
  assert.equal(isLegacyLayupMetric('jax/train/learner_shot_defense_dunk_attempts'), false)
})
