from types import SimpleNamespace
import numpy as np
import pytest
from basketworld_jax.leagues import generate_league, snapshot_digest
from basketworld_jax.roster_sampling import compile_league, assign_rosters, evaluation_static, flush_roster_metrics, SKILLS, validate_roster_resume


def snapshot(size=3):
    s = generate_league(name='Sampling test', team_count=4, team_size=size, seed=17,
                       rules={'rim_protection_enabled': True, 'contested_shooting_enabled': True})
    s['version'] = 1
    s['sha256'] = snapshot_digest(s)
    return s


@pytest.mark.parametrize('size', [3, 4, 5])
def test_game_boundary_sampling_skills_rng_and_metrics(size):
    import jax
    import jax.numpy as jnp
    from basketworld_jax.config import JaxEnvironmentConfig
    from basketworld_jax.env.minimal import build_kernel_static_from_config, reset_batch_minimal, replace_done_states
    from basketworld_jax.train.main import _filter_mlflow_train_metrics, _restore_like_template
    s = snapshot(size)
    static = build_kernel_static_from_config(JaxEnvironmentConfig(players=size, enable_multi_possession=True), xp=jnp)
    static = static._replace(league_pool=compile_league(s, jnp), league_team_probability=jnp.array(1.))
    reset = jax.jit(lambda keys, prev=None: reset_batch_minimal(static, keys, jax, jnp, roster_state=prev))
    keys = jax.random.split(jax.random.PRNGKey(1), 12)
    state = reset(keys)
    # Mapping is the saved member order and follows stable sides.
    for i, d in enumerate(SKILLS):
        expected = np.asarray(static.league_pool.skills)[np.asarray(state.roster_ids), :, i].reshape(12, 2*size)
        np.testing.assert_array_equal(getattr(state, d.runtime_field), expected)
    # Future assignments depend on the saved roster RNG, not environment/policy reset keys.
    next_state = reset(keys, state)
    restored = _restore_like_template({k: np.asarray(v) for k, v in state._asdict().items()}, state)
    restored = jax.tree.map(lambda x, template: jnp.asarray(x, dtype=template.dtype), restored, state)
    resumed = reset(jax.random.split(jax.random.PRNGKey(999), 12), restored)
    np.testing.assert_array_equal(next_state.roster_ids, resumed.roster_ids)
    np.testing.assert_array_equal(next_state.roster_rng, resumed.roster_rng)
    # A real possession handoff changes offense, never player ownership or skills.
    from basketworld_jax.env.minimal import _finalize_possession_single, POSSESSION_END_TURNOVER
    def handoff(row):
        return _finalize_possession_single(
            static, row, possession_ended=jnp.array(True),
            possession_end_reason=jnp.array(POSSESSION_END_TURNOVER),
            next_offense_team=1-row.offense_team, requires_inbound=jnp.array(False),
            inbound_selection_key=jax.random.PRNGKey(5), jax=jax, jnp=jnp,
        )[0]
    switched = jax.jit(jax.vmap(handoff))(state)
    np.testing.assert_array_equal(switched.offense_team, 1-state.offense_team)
    for field in ['roster_ids', 'roster_rng'] + [d.runtime_field for d in SKILLS]:
        np.testing.assert_array_equal(getattr(switched, field), getattr(state, field))
    mask = jnp.arange(12) % 2 == 0
    transitioned = replace_done_states(state._replace(offense_team=1-state.offense_team), next_state, mask, jnp)
    np.testing.assert_array_equal(transitioned.roster_rng[~mask], state.roster_rng[~mask])
    np.testing.assert_array_equal(transitioned.layup_pct[~mask], state.layup_pct[~mask])
    metrics, cleared = flush_roster_metrics({'offense': transitioned}, jnp)
    assert metrics['roster_persistent_persistent_pct'] == 100
    assert sum(float(np.asarray(x).sum()) for x in [cleared['offense'].roster_counts, cleared['offense'].roster_seen]) == 0
    assert set(metrics) <= _filter_mlflow_train_metrics(metrics).keys()
    fresh_static = evaluation_static(static, jnp)
    fresh = jax.jit(lambda k: reset_batch_minimal(fresh_static, k, jax, jnp))(keys)
    assert (np.asarray(fresh.roster_ids) == -1).all()
    assert np.std(np.asarray(fresh.passing_skill)) > .5
    # Legacy reset produces no assignments or league metrics.
    legacy = reset_batch_minimal(static._replace(league_pool=None), keys, jax, jnp)
    assert legacy.roster_seen.shape == (12, 0)


def test_source_frequencies_same_team_and_generation_distributions():
    import jax
    import jax.numpy as jnp
    from collections import namedtuple
    s = snapshot()
    # Constant draws verify generation settings are honored exactly, including means.
    for v in s['sampling'].values():
        if 'std' in v: v['std'] = 0.
    s['sampling']['passing']['mean'] = 2.5
    s['sha256'] = snapshot_digest(s)
    pool = compile_league(s, jnp)
    fields = [d.runtime_field for d in SKILLS] + ['rebound_skill_specialist', 'roster_rng', 'roster_ids', 'roster_counts', 'roster_seen']
    State = namedtuple('State', fields)
    state = State(*([jnp.zeros(6)]*len(SKILLS)), jnp.zeros(6), jnp.zeros(2,dtype=jnp.uint32), jnp.full(2,-1), jnp.zeros(3,dtype=jnp.int32), jnp.zeros(4,dtype=bool))
    static = SimpleNamespace(league_pool=pool, league_team_probability=.75, offense_ids=jnp.arange(3), defense_ids=jnp.arange(3,6))
    sample = jax.jit(jax.vmap(lambda k: assign_rosters(static,state,k,state.roster_counts,state.roster_seen,jax,jnp)))
    out = sample(jax.random.split(jax.random.PRNGKey(87), 8192))
    proportions = np.asarray(out.roster_counts).mean(axis=0)
    np.testing.assert_allclose(proportions, [.5625,.375,.0625], atol=.02)
    ids=np.asarray(out.roster_ids)
    assert np.any((ids[:,0] == ids[:,1]) & (ids[:,0]>=0))
    passing=np.asarray(out.passing_skill).reshape(-1,2,3)
    np.testing.assert_allclose(passing[ids<0],2.5)
    counts=np.bincount(ids[ids>=0],minlength=4)
    np.testing.assert_allclose(counts/counts.sum(),.25,atol=.02)


def test_resume_rejects_changed_roster_settings_and_missing_state():
    s = snapshot()
    args=SimpleNamespace(roster_source='league',league_team_probability=.75,_league_snapshot=s)
    from basketworld_jax.roster_sampling import roster_contract
    payload={'trainer_config':roster_contract(args),'current_state':{'offense':{'positions':np.zeros((2,6,2))}}}
    with pytest.raises(ValueError,match='roster_rng'):
        validate_roster_resume(payload,args)
    payload['trainer_config']['league_team_probability']=.5
    with pytest.raises(ValueError,match='configuration mismatch'):
        validate_roster_resume(payload,args)
