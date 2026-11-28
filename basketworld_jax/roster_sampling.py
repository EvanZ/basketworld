"""Immutable league tensors and independent, checkpointed game-boundary draws."""
from typing import NamedTuple, Any
import numpy as np
from basketworld_jax.eval.skills import ACTIVE_PLAYER_SKILL_REGISTRY, registry_for_environment
from basketworld_jax.shot_defense import rim_skill_from_block_sample
from basketworld_jax.leagues import validate_snapshot

SKILLS = tuple(ACTIVE_PLAYER_SKILL_REGISTRY.descriptors)
METRICS = ('roster_persistent_persistent_pct', 'roster_mixed_pct',
           'roster_fresh_fresh_pct', 'roster_pool_coverage_pct')


class LeaguePool(NamedTuple):
    skills: Any
    means: Any
    stds: Any
    minimum: Any
    maximum: Any
    beta: Any
    active: Any


def compile_league(snapshot, xp):
    validate_snapshot(snapshot)
    registry = registry_for_environment(snapshot['rules']['parameters'])
    players = {p['id']: p for p in snapshot['players']}
    values = [[[players[m['player_id']]['skills'].get(d.key, 0.0) for d in SKILLS]
               for m in t['members']] for t in snapshot['rosters']]
    sampling = snapshot['sampling']
    beta = sampling.get('rim_protection', {'alpha': 1., 'beta': 3., 'maximum': .1, 'reference': .02})
    return LeaguePool(
        xp.asarray(values, dtype=xp.float32),
        xp.asarray([sampling.get(d.key, {}).get('mean', 0.) for d in SKILLS], dtype=xp.float32),
        xp.asarray([sampling.get(d.key, {}).get('std', 0.) for d in SKILLS], dtype=xp.float32),
        xp.asarray([d.minimum if d.minimum is not None else -np.inf for d in SKILLS], dtype=xp.float32),
        xp.asarray([d.maximum if d.maximum is not None else np.inf for d in SKILLS], dtype=xp.float32),
        xp.asarray([beta[k] for k in ('alpha', 'beta', 'maximum', 'reference')], dtype=xp.float32),
        xp.asarray([d.key in registry.keys for d in SKILLS], dtype=xp.bool_),
    )


def assign_rosters(static, state, roster_key, counts, seen, jax, jnp):
    """One game, two independent roster draws. Never consume policy-opponent RNG."""
    pool = static.league_pool
    next_key, source_key, team_key, normal_key, beta_key = jax.random.split(roster_key, 5)
    persistent = jax.random.uniform(source_key, (2,)) < static.league_team_probability
    indices = jax.random.randint(team_key, (2,), 0, pool.skills.shape[0])
    shape = (2, pool.skills.shape[1], len(SKILLS))
    fresh = jnp.clip(pool.means + pool.stds * jax.random.normal(normal_key, shape), pool.minimum, pool.maximum)
    rim_index = next(i for i, d in enumerate(SKILLS) if d.key == 'rim_protection')
    rim = rim_skill_from_block_sample(pool.beta[2] * jax.random.beta(beta_key, pool.beta[0], pool.beta[1], shape=shape[:2]), pool.beta[3], jnp)
    fresh = fresh.at[:, :, rim_index].set(rim)
    fresh = jnp.where(pool.active, fresh, 0.)
    values = jnp.where(persistent[:, None, None], pool.skills[indices], fresh)
    ids = jnp.where(persistent, indices, -1)
    player_ids = jnp.concatenate((static.offense_ids, static.defense_ids))
    updates = {}
    for i, descriptor in enumerate(SKILLS):
        updates[descriptor.runtime_field] = getattr(state, descriptor.runtime_field).at[player_ids].set(values[:, :, i].reshape(-1))
    kind = 2 - jnp.sum(persistent.astype(jnp.int32))
    seen = seen | jnp.any((jnp.arange(pool.skills.shape[0])[:, None] == ids[None, :]), axis=1)
    return state._replace(**updates, rebound_skill_specialist=jnp.zeros_like(state.rebound_skill_specialist),
                          roster_rng=next_key, roster_ids=ids,
                          roster_counts=counts.at[kind].add(1), roster_seen=seen)


def evaluation_static(static, xp):
    return static if static.league_pool is None else static._replace(league_team_probability=xp.asarray(0., dtype=xp.float32))


def flush_roster_metrics(states, jnp):
    counts = np.sum([np.asarray(s.roster_counts).sum(axis=0) for s in states.values()], axis=0)
    seen = np.any(np.concatenate([np.asarray(s.roster_seen) for s in states.values()], axis=0), axis=0)
    total = counts.sum()
    metrics = {key: float(100 * n / total) for key, n in zip(METRICS[:3], counts)} if total else {}
    if seen.size:
        metrics[METRICS[3]] = float(100 * seen.mean())
    reset = {role: s._replace(roster_counts=jnp.zeros_like(s.roster_counts), roster_seen=jnp.zeros_like(s.roster_seen)) for role, s in states.items()}
    return metrics, reset


def training_league(args):
    from basketworld_jax.leagues import load_league_snapshot
    source = getattr(args, 'roster_source', 'random')
    probability = float(getattr(args, 'league_team_probability', .75))
    if source not in ('random', 'league') or not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError('Invalid roster_source or league_team_probability (expected 0 through 1).')
    path = getattr(args, 'league_snapshot_path', '')
    if source == 'random':
        if path:
            raise ValueError('A league snapshot requires roster_source=league.')
        return None
    if not path:
        raise ValueError('League training requires --league-snapshot-path.')
    return load_league_snapshot(path, environment=vars(args))


def roster_contract(args):
    snapshot = getattr(args, '_league_snapshot', None)
    return {'roster_source': getattr(args, 'roster_source', 'random'),
            'league_team_probability': float(getattr(args, 'league_team_probability', .75)),
            'league_reference': ({k: snapshot[k] for k in ('league_id', 'version', 'sha256')} if snapshot else None)}


def validate_roster_resume(payload, args):
    expected = roster_contract(args)
    saved = payload.get('trainer_config', {})
    for key, default in [('roster_source', 'random'), ('league_team_probability', .75), ('league_reference', None)]:
        if saved.get(key, default) != expected[key]:
            raise ValueError(f'Resume roster configuration mismatch: {key}. Start a new run to change roster settings.')
    if expected['roster_source'] != 'league':
        return
    n = len(args._league_snapshot['rosters'])
    for state in payload['current_state'].values():
        get = state.get if isinstance(state, dict) else lambda k: getattr(state, k, None)
        batch = np.shape(get('positions'))[0]
        for field, shape in [('roster_rng', (batch, 2)), ('roster_ids', (batch, 2)),
                             ('roster_counts', (batch, 3)), ('roster_seen', (batch, n))]:
            if get(field) is None or np.shape(get(field)) != shape:
                raise ValueError(f'Resume checkpoint is missing valid {field}; roster state cannot be regenerated.')
