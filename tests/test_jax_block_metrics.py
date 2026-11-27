from types import SimpleNamespace as NS

import numpy as np
import pytest

from basketworld_jax.block_metrics import (
    TRAIN_BLOCK_METRICS, DEPLOY_BLOCK_METRICS, TrainingBlockMetrics,
    TRAIN_RIM_CONTEST_METRICS, DEPLOY_RIM_CONTEST_METRICS,
    TRAIN_LAYUP_METRICS, DEPLOY_LAYUP_METRICS,
    empty_block_metric_state, advance_block_metrics, summarize_block_metrics,
)
from basketworld_jax.shot_defense_metrics import (
    SHOT_DEFENSE_COUNTERS, SHOT_DEFENSE_SHAPE, summarize_shot_defense_diagnostics,
)


def trace(*events):
    """One lane; shooting role +1 means learner offense."""
    n = len(events)
    diagnostics = np.zeros((n, 1, *SHOT_DEFENSE_SHAPE), np.float32)
    fields = {key: np.zeros((n, 1), bool) for key in
              ('active_mask', 'episode_start', 'possession_ended', 'dones', 'truncated')}
    roles = np.zeros((n, 1), np.float32)
    for step, event in enumerate(events):
        roles[step, 0] = event.get('role', 1)
        for key in fields:
            fields[key][step, 0] = event.get(key, key == 'active_mask')
        for key in ('blocked', 'layup_attempts', 'layup_makes', 'dunk_attempts', 'jump_attempts',
                    'rim_opportunities', 'rim_contest_probability_drop_sum'):
            diagnostics[step, 0, 0 if roles[step, 0] > 0 else 1,
                        SHOT_DEFENSE_COUNTERS.index(key)] = event.get(key, 0)
    return NS(shot_defense_diagnostics=diagnostics, training_role=roles, **fields)


def consume(t):
    collector = TrainingBlockMetrics()
    collector.consume(t)
    return collector


def test_completed_possessions_and_game_carry_across_updates_and_logging():
    # Two blocks recovered by offense belong to the same defensive possession.
    first = dict(episode_start=True, blocked=1, layup_attempts=1)
    second = dict(blocked=1, dunk_attempts=1, possession_ended=True)
    third = dict(role=-1, layup_attempts=1, possession_ended=True, dones=True)
    collector = consume(trace(first))
    assert collector.flush() == {}  # No premature numerator or denominator.
    collector.consume(trace(second, third))
    result = collector.flush()
    assert result == consume(trace(first, second, third)).flush()
    assert result == {
        'learner_block_pct': 0, 'learner_blocks_per_defensive_possession': 0,
        'learner_blocks_per_episode': 0, 'opponent_block_pct': 100,
        'opponent_blocks_per_defensive_possession': 2, 'opponent_blocks_per_episode': 2,
        'learner_rim_contested_shots_per_episode': 0, 'learner_rim_contested_shot_pct': 0,
        'opponent_rim_contested_shots_per_episode': 0, 'opponent_rim_contested_shot_pct': 0,
        'learner_layup_attempts_per_episode': 1, 'learner_layup_makes_per_episode': 0, 'learner_layup_fg_pct': 0,
        'opponent_layup_attempts_per_episode': 1, 'opponent_layup_makes_per_episode': 0, 'opponent_layup_fg_pct': 0,
    }
    assert collector.flush() == {}


def test_percentage_excludes_jump_shots_and_combines_counts_not_lane_rates():
    collector = consume(trace(
        dict(episode_start=True, blocked=1, layup_attempts=1, possession_ended=True),
        dict(layup_attempts=3, dunk_attempts=1, possession_ended=True),
        dict(jump_attempts=100, possession_ended=True, dones=True),
    ))
    result = collector.flush()
    assert result['opponent_block_pct'] == 20
    assert result['opponent_blocks_per_defensive_possession'] == pytest.approx(1 / 3)
    assert 'learner_block_pct' not in result
    assert 'learner_blocks_per_defensive_possession' not in result
    assert result['learner_blocks_per_episode'] == 0


def test_truncation_and_unfinished_possessions_have_matched_populations():
    collector = consume(trace(
        dict(episode_start=True, layup_attempts=1, possession_ended=True),
        dict(blocked=1, dunk_attempts=1, possession_ended=True, dones=True, truncated=True),
        dict(episode_start=True, blocked=1, layup_attempts=1),
    ))
    assert collector.flush() == {
        'opponent_block_pct': 0, 'opponent_blocks_per_defensive_possession': 0,
    }


def test_resume_ignores_unknown_partial_game_and_first_partial_possession():
    collector = consume(trace(
        dict(blocked=1, layup_attempts=1, possession_ended=True),
        dict(role=-1, layup_attempts=1, possession_ended=True, dones=True),
    ))
    assert collector.flush() == {
        'learner_block_pct': 0, 'learner_blocks_per_defensive_possession': 0,
    }
    collector.consume(trace(dict(episode_start=True, blocked=1, dunk_attempts=1,
                                 possession_ended=True, dones=True)))
    assert collector.flush()['opponent_blocks_per_episode'] == 1


def test_combined_deploy_counts_games_once_and_ignores_inactive_lanes():
    t = trace(
        dict(episode_start=True, blocked=1, layup_attempts=1, possession_ended=True),
        dict(role=-1, blocked=1, dunk_attempts=1, possession_ended=True, dones=True),
        dict(active_mask=False, blocked=100, layup_attempts=100, possession_ended=True, dones=True),
    )
    totals = consume(t).state.totals
    assert summarize_block_metrics(totals, combined=True) == {
        'block_pct': 100, 'blocks_per_possession': 1, 'blocks_per_episode': 2,
        'rim_contested_shots_per_episode': 0, 'rim_contested_shot_pct': 0,
        'layup_attempts_per_episode': .5, 'layup_makes_per_episode': 0, 'layup_fg_pct': 0,
    }


def test_jitted_accounting_matches_host_across_multiple_lanes():
    import jax
    import jax.numpy as jnp

    t = trace(dict(episode_start=True, blocked=1, layup_attempts=1, possession_ended=True, dones=True,
                   rim_opportunities=1, rim_contest_probability_drop_sum=0.2))
    kwargs = dict(active=np.array([True, False]), episode_start=np.ones(2, bool),
                  possession_ended=np.ones(2, bool), done=np.ones(2, bool),
                  truncated=np.zeros(2, bool), training_role=np.array([1, -1]))
    diagnostics = np.repeat(t.shot_defense_diagnostics[0], 2, axis=0)
    expected = advance_block_metrics(empty_block_metric_state(2), diagnostics, **kwargs)
    actual = jax.jit(lambda state, diag, args: advance_block_metrics(state, diag, **args, xp=jnp))(
        empty_block_metric_state(2, jnp), diagnostics, kwargs)
    for left, right in zip(expected, actual):
        np.testing.assert_array_equal(left, right)


def test_rim_contest_uses_matched_completed_possessions_across_updates():
    collector = consume(trace(dict(episode_start=True, rim_opportunities=1,
                                   rim_contest_probability_drop_sum=.2)))
    assert collector.flush() == {}
    collector.consume(trace(
        dict(rim_opportunities=1, rim_contest_probability_drop_sum=.3, possession_ended=True),
        dict(role=-1, rim_opportunities=1, rim_contest_probability_drop_sum=.1, possession_ended=True),
        dict(rim_opportunities=1, rim_contest_probability_drop_sum=.9, possession_ended=True,
             truncated=True, dones=True),
    ))
    totals = collector.state.totals
    result = collector.flush()
    assert result['opponent_rim_contest_mean_reduction_pp'] == pytest.approx(25)
    assert result['learner_rim_contest_mean_reduction_pp'] == pytest.approx(10)
    assert summarize_block_metrics(totals, combined=True)['rim_contest_mean_reduction_pp'] == pytest.approx(20)
    assert collector.flush() == {}


def test_rim_contest_frequency_counts_complete_games_across_logging_windows():
    collector = consume(trace(dict(episode_start=True, blocked=1, layup_attempts=1,
                                   rim_opportunities=1, possession_ended=True)))
    first = collector.flush()
    assert not any('rim_contested_' in name for name in first)
    collector.consume(trace(
        dict(role=-1, layup_attempts=3, dunk_attempts=1, rim_opportunities=2,
             possession_ended=True),
        dict(layup_attempts=2, dunk_attempts=1, rim_opportunities=1, possession_ended=True),
        dict(jump_attempts=100, possession_ended=True, dones=True),
        # An unfinished next game must not leak into either numerator or denominator.
        dict(episode_start=True, layup_attempts=100, rim_opportunities=100, possession_ended=True),
    ))
    totals = collector.state.totals.copy()
    result = collector.flush()
    for side in ('learner', 'opponent'):
        assert result[f'{side}_rim_contested_shots_per_episode'] == 2
        assert result[f'{side}_rim_contested_shot_pct'] == 50
    deploy = summarize_block_metrics(totals, combined=True)
    assert deploy['rim_contested_shots_per_episode'] == 2  # Four contests / two team-games.
    assert deploy['rim_contested_shot_pct'] == 50  # Four contests / eight layups+dunks.
    collector.consume(trace(dict(dones=True, truncated=True, possession_ended=True)))
    assert not any('rim_contested_' in name for name in collector.flush())


def test_rim_contest_frequency_zero_counts_missing_denominators_and_partial_resume():
    assert summarize_block_metrics(empty_block_metric_state(1).totals) == {}
    no_shots = consume(trace(dict(episode_start=True, possession_ended=True, dones=True))).flush()
    for side in ('learner', 'opponent'):
        assert no_shots[f'{side}_rim_contested_shots_per_episode'] == 0
        assert f'{side}_rim_contested_shot_pct' not in no_shots
    resumed = consume(trace(dict(layup_attempts=1, rim_opportunities=1,
                                 possession_ended=True, dones=True))).flush()
    assert not any('rim_contested_' in name for name in resumed)
    # Unequal attempt populations must be pooled before dividing.
    totals = np.array([[0, 0, 0, 0, 2, 0, 0, 1, 2], [0, 0, 0, 0, 2, 0, 0, 3, 18]])
    deploy = summarize_block_metrics(totals, combined=True)
    assert deploy['rim_contested_shot_pct'] == 20
    assert deploy['rim_contested_shots_per_episode'] == 1


def test_rim_contest_probability_drop_masks_shot_types_and_excludes_block_effect():
    import jax
    import jax.numpy as jnp
    from basketworld_jax.env.minimal import SHOT_TYPE_DUNK, SHOT_TYPE_TWO, SHOT_TYPE_THREE
    from basketworld_jax.shot_defense_metrics import build_shot_defense_diagnostics

    static = NS(rim_protection_enabled=jnp.asarray(True), training_player_mask=jnp.array([1, 0]))
    fields = dict(
        shot_attempt=jnp.ones(6), shot_success=jnp.zeros(6),
        shot_is_finish=jnp.array([1, 1, 1, 0, 1, 1]),
        shot_contested=jnp.array([1, 1, 0, 1, 1, 1]),
        shot_blocked=jnp.array([1, 0, 0, 0, 0, 0]),
        offensive_rebound=jnp.zeros(6), defensive_rebound=jnp.zeros(6),
        shot_type=jnp.array([SHOT_TYPE_TWO, SHOT_TYPE_DUNK, SHOT_TYPE_TWO,
                            SHOT_TYPE_THREE, SHOT_TYPE_TWO, SHOT_TYPE_TWO]),
        shot_shooter=jnp.array([0, 0, 0, 0, 0, 1]), possession_ended=jnp.ones(6),
    )
    active = jnp.array([True, True, True, True, False, True])
    build = jax.jit(lambda opened, conditional, block: build_shot_defense_diagnostics(
        static, NS(**fields, shot_open_probability=opened,
                   shot_conditional_probability=conditional, shot_block_probability=block), active, jnp))
    opened = jnp.array([.8, .9, .8, .8, .8, .6])
    conditional = jnp.array([.6, .6, .8, .4, .1, .5])
    diag = np.asarray(build(opened, conditional, jnp.ones(6)))
    no_block = np.asarray(build(opened, conditional, jnp.zeros(6)))
    idx = SHOT_DEFENSE_COUNTERS.index('rim_contest_probability_drop_sum')
    np.testing.assert_allclose(diag[:, :, idx], no_block[:, :, idx])
    np.testing.assert_allclose(diag[:, :, idx].sum(axis=-1), [.2, .3, 0, 0, 0, .1], atol=1e-7)
    np.testing.assert_array_equal(diag[:, :, SHOT_DEFENSE_COUNTERS.index('rim_opportunities')].sum(axis=-1),
                                  [1, 1, 0, 0, 0, 1])
    np.testing.assert_array_equal(diag[:, :, SHOT_DEFENSE_COUNTERS.index('completed_possessions')].sum(axis=-1), active)
    disabled = build_shot_defense_diagnostics(
        NS(**{**vars(static), 'rim_protection_enabled': jnp.asarray(False)}),
        NS(**fields, shot_open_probability=opened, shot_conditional_probability=conditional,
           shot_block_probability=jnp.zeros(6)), active, jnp)
    np.testing.assert_array_equal(disabled[:, :, idx], 0)


def test_historical_diagnostics_remain_readable_without_inventing_contest_values():
    legacy = np.zeros((2, len(SHOT_DEFENSE_COUNTERS) - 1))
    legacy[0, SHOT_DEFENSE_COUNTERS.index('blocked')] = 2
    legacy[0, SHOT_DEFENSE_COUNTERS.index('completed_possessions')] = 4
    summary = summarize_shot_defense_diagnostics(legacy)
    assert summary['blocks_per_completed_possession'] == .5
    assert not any('rim_contest_probability_drop_sum' in name for name in summary)
    t = trace(dict(episode_start=True, rim_opportunities=1, possession_ended=True))
    t.shot_defense_diagnostics = t.shot_defense_diagnostics[..., :-1]
    assert not any('rim_contest' in name for name in consume(t).flush())


@pytest.mark.parametrize('bulk', [False, True])
def test_mlflow_sink_logs_only_approved_block_rim_contest_and_layup_names(bulk):
    from basketworld_jax.train.main import _log_mlflow_metrics

    logged = {}
    sink = NS(log_metrics=lambda metrics, **kwargs: logged.update(metrics)) if bulk else NS(
        log_metric=lambda key, value, **kwargs: logged.update({key: value}))
    metrics = summarize_shot_defense_diagnostics(np.ones(SHOT_DEFENSE_SHAPE))
    metrics.update({key: 1.0 for key in TRAIN_BLOCK_METRICS | DEPLOY_BLOCK_METRICS
                    | TRAIN_RIM_CONTEST_METRICS | DEPLOY_RIM_CONTEST_METRICS
                    | TRAIN_LAYUP_METRICS | DEPLOY_LAYUP_METRICS})
    metrics['unrelated'] = 2
    for context in ('train', 'eval_deploy', 'eval_offense', 'eval_defense'):
        _log_mlflow_metrics(sink, metrics, step=1, prefix=f'jax/{context}')
        assert logged.pop(f'jax/{context}/unrelated') == 2
    assert len(logged) == 27
    assert set(logged) == ({f'jax/train/{k}' for k in TRAIN_BLOCK_METRICS | TRAIN_RIM_CONTEST_METRICS | TRAIN_LAYUP_METRICS}
                           | {f'jax/eval_deploy/{k}' for k in DEPLOY_BLOCK_METRICS | DEPLOY_RIM_CONTEST_METRICS | DEPLOY_LAYUP_METRICS})


def test_deploy_tracing_and_summary_use_compact_block_totals():
    import jax
    import jax.numpy as jnp
    from tests.test_jax_pressure_metrics import fixture
    from basketworld_jax.env import build_policy_observation_batch, build_action_masks_batch
    from basketworld_jax.models import build_actor_critic_spec, init_actor_critic_params
    from basketworld_jax.train.runtime import (
        build_compiled_deploy_eval_runner, summarize_deploy_eval_outputs,
    )

    static, state = fixture()
    obs = build_policy_observation_batch(static, state, jnp, model_type='mlp', multi_possession_features=True)
    masks = build_action_masks_batch(static, state, jnp)[:, static.offense_ids, :]
    spec = build_actor_critic_spec(np.asarray(obs), np.asarray(masks), (16,), multi_possession_features=True)
    params = init_actor_critic_params(jax, jnp, spec, seed=80)
    runner = build_compiled_deploy_eval_runner(jax, jnp, spec)
    # Trace the complete scan to catch carry/shape/type integration errors;
    # the small accounting kernel itself is compiled in the parity test above.
    shaped = jax.eval_shape(lambda: runner(
        static, state, params, jax.random.PRNGKey(80), 1, False, False, 3))
    assert shaped.block_metric_totals.shape == (2, 11)
    output = jax.tree.map(lambda value: np.zeros(value.shape, value.dtype), shaped)
    output = output._replace(final_state=state)
    # Exercise merging across batches with unequal denominators. Two completed
    # games in the first batch; one in the second. Each row counts defending side.
    first = output._replace(block_metric_totals=np.array([[1, 4, 5, 1, 2, .2, 1, 1, 4, 6, 3], [2, 6, 7, 2, 2, .3, 2, 2, 6, 2, 2]]))
    second = output._replace(block_metric_totals=np.array([[1, 10, 2, 1, 1, .1, 1, 1, 10, 2, 0], [0, 0, 1, 0, 1, 0, 0, 0, 0, 10, 3]]))
    summary = summarize_deploy_eval_outputs([first, second], batch_size=4)
    assert summary['block_pct'] == 20
    assert summary['blocks_per_possession'] == pytest.approx(4 / 15)
    assert summary['blocks_per_episode'] == pytest.approx(4 / 3)

    assert summary['rim_contest_mean_reduction_pp'] == pytest.approx(15)

    assert summary['rim_contested_shots_per_episode'] == pytest.approx(4 / 6)
    assert summary['rim_contested_shot_pct'] == pytest.approx(20)

    assert summary['layup_attempts_per_episode'] == pytest.approx(20 / 6)
    assert summary['layup_makes_per_episode'] == pytest.approx(8 / 6)
    assert summary['layup_fg_pct'] == 40


def test_layups_use_shooting_side_complete_games_and_pooled_attempts():
    collector = consume(trace(dict(episode_start=True, blocked=1, layup_attempts=1)))
    assert collector.flush() == {}
    collector.consume(trace(
        dict(layup_attempts=3, layup_makes=2, possession_ended=True),
        dict(role=-1, layup_attempts=1, layup_makes=1, possession_ended=True),
        dict(dunk_attempts=50, jump_attempts=100, possession_ended=True, dones=True),
        dict(episode_start=True, layup_attempts=100, layup_makes=100, dones=True, truncated=True),
        dict(episode_start=True, layup_attempts=100, layup_makes=100),
    ))
    deploy = summarize_block_metrics(collector.state.totals, combined=True)
    result = collector.flush()
    assert result['learner_layup_attempts_per_episode'] == 4
    assert result['learner_layup_makes_per_episode'] == 2
    assert result['learner_layup_fg_pct'] == 50
    assert result['opponent_layup_attempts_per_episode'] == 1
    assert result['opponent_layup_makes_per_episode'] == 1
    assert result['opponent_layup_fg_pct'] == 100
    assert deploy['layup_attempts_per_episode'] == 2.5
    assert deploy['layup_makes_per_episode'] == 1.5
    assert deploy['layup_fg_pct'] == 60
    assert not any('layup' in key for key in collector.flush())


def test_layup_zero_counts_and_resume_do_not_fabricate_percentages():
    collector = consume(trace(dict(layup_attempts=10, layup_makes=9, dones=True)))
    assert not any('layup' in key for key in collector.flush())
    collector.consume(trace(
        dict(episode_start=True, dones=True),
        dict(active_mask=False, episode_start=True, layup_attempts=100, layup_makes=100, dones=True),
    ))
    result = collector.flush()
    for side in ('learner', 'opponent'):
        assert result[f'{side}_layup_attempts_per_episode'] == 0
        assert result[f'{side}_layup_makes_per_episode'] == 0
        assert f'{side}_layup_fg_pct' not in result
