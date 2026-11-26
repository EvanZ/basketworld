import test from 'node:test'
import assert from 'node:assert/strict'
import { scoringMetricGroup, scoringMetricDefinition } from '../src/scoringMetrics.js'

test('primary scoring separates learner outcomes, self-play, rewards, and fairness', () => {
  assert.equal(scoringMetricGroup('jax/train/learner_points_scored_per_completed_game'), 'Learner scoring')
  assert.equal(scoringMetricGroup('jax/eval/learner_completed_game_win_score'), 'Learner evaluation')
  assert.equal(scoringMetricGroup('jax/eval_deploy/self_play_points_per_team_per_completed_game'), 'Self-play behavior')
  assert.equal(scoringMetricGroup('jax/train/learner_roster_assignment_b_completed_game_count'), 'Roster and opening-possession diagnostics')
  assert.equal(scoringMetricGroup('jax/train/opponent_started_with_possession_win_score'), 'Roster and opening-possession diagnostics')
  assert.equal(scoringMetricGroup('jax/train/game_reward_mean'), 'RL health and other diagnostics')
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
    assert.equal(scoringMetricGroup(name), 'Shot defense and blocks')
    assert.match(scoringMetricDefinition(name), /credited to the defending team/)
    assert.match(scoringMetricDefinition(name), /finishes with an adjacent defender/)
  }
})
