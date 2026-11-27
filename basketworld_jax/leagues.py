"""League artifact v1: exact player identities, skills and versioned membership.

Shared by trainer authoring and future training/evaluation/play consumers. No
worker, database, SB3 or JAX runtime dependency is needed to load a snapshot.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from uuid import UUID, uuid4, uuid5

import numpy as np

from basketworld_jax.eval.fixed_pool import generate_player_pool
from basketworld_jax.eval.skills import registry_for_environment, registry_for_schema
from basketworld_jax.league_names import FIRST_NAMES, LAST_NAMES, PLACES, NICKNAMES, NAME_SET_VERSION
from basketworld_jax.shot_defense import SHOT_DEFENSE_DEFAULTS, shot_defense_contract, validate_shot_defense

LEAGUE_SCHEMA_VERSION = 1
LEAGUE_TRAINING_AVAILABLE = False
UNAVAILABLE_MESSAGE = 'League-backed training is unavailable until #85 slice 2 is implemented. Published leagues can be authored and inspected now.'


def canonical_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def snapshot_digest(snapshot):
    return hashlib.sha256(canonical_bytes({k: v for k, v in snapshot.items() if k != 'sha256'})).hexdigest()


def required_name(value, label):
    if not isinstance(value, str) or not value.strip() or len(value) > 120:
        raise ValueError(f'{label} must contain 1–120 characters.')
    return value.strip()


def generation_settings(registry, overrides=None):
    overrides = overrides or {}
    if set(overrides) - set(registry.keys):
        raise ValueError('Generation settings contain skills unsupported by the selected rules.')
    result = {}
    for d in registry.descriptors:
        defaults = dict(d.sampling_defaults)
        default = ({k: defaults[k] for k in ('alpha', 'beta', 'maximum', 'reference')}
                   if d.sampling_distribution == 'scaled_beta_logit_centered'
                   else {'mean': defaults.get('default_mean', 0), 'std': 1.0})
        supplied = overrides.get(d.key, {})
        if not isinstance(supplied, dict) or set(supplied) - set(default):
            raise ValueError(f'Unknown distribution settings for {d.key}.')
        values = {**default, **supplied}
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values.values()):
            raise ValueError(f'{d.key} distribution settings must be finite numbers.')
        if 'std' in values and values['std'] < 0:
            raise ValueError(f'{d.key} standard deviation must be nonnegative.')
        # The shared sampler validates beta parameters; reject before generation too.
        if 'alpha' in values:
            validate_shot_defense({
                'rim_protection_beta_alpha': values['alpha'], 'rim_protection_beta_beta': values['beta'],
                'rim_protection_block_distribution_max': values['maximum'],
                'rim_protection_skill_block_reference': values['reference'],
            })
        result[d.key] = values
    return result


def generate_league(*, name, team_count=64, team_size=5, seed=0, rules=None, sampling=None, league_id=None):
    name = required_name(name, 'League name')
    if type(team_count) is not int or not 2 <= team_count <= 512:
        raise ValueError('Team count must be an integer from 2 to 512.')
    if type(team_size) is not int or team_size not in (3, 4, 5):
        raise ValueError('Team size must be 3, 4 or 5.')
    if type(seed) is not int or not 0 <= seed <= 2_147_483_647:
        raise ValueError('Seed must be an integer from 0 to 2147483647.')
    rules = dict(rules or {})
    if set(rules) - set(SHOT_DEFENSE_DEFAULTS):
        raise ValueError('Unknown skill-rule settings.')
    rules = {**SHOT_DEFENSE_DEFAULTS, **rules}
    for key in ('rim_protection_enabled', 'contested_shooting_enabled'):
        if type(rules[key]) is not bool:
            raise ValueError(f'{key} must be a boolean.')
    for key, default in SHOT_DEFENSE_DEFAULTS.items():
        if not isinstance(default, bool):
            value = rules[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f'{key} must be a finite number.')
    validate_shot_defense(rules)
    registry = registry_for_environment(rules)
    sampling = generation_settings(registry, sampling)
    league_id = str(UUID(league_id)) if league_id else str(uuid4())
    # Cosmetic draws and roster assignment never perturb the skill sampler.
    rng = np.random.default_rng(np.random.SeedSequence([seed, NAME_SET_VERSION, 86]))
    count = team_count * team_size
    name_indices = rng.choice(len(FIRST_NAMES) * len(LAST_NAMES), count, replace=False)
    names = [f'{FIRST_NAMES[int(i) // len(LAST_NAMES)]} {LAST_NAMES[int(i) % len(LAST_NAMES)]}' for i in name_indices]
    pool = generate_player_pool(player_count=count, seed=seed, registry=registry, sampling_overrides=sampling, names=names)
    players = []
    for original in pool.players:
        skills = dict(original.skills)
        players.append({'id': str(uuid5(UUID(league_id), original.player_id)), 'name': original.name,
                        'skills': skills, 'generated_skills': dict(skills), 'skill_overrides': {},
                        'provenance': {'pool_id': pool.pool_id, 'pool_player_id': original.player_id}})
    team_names = rng.choice(len(PLACES) * len(NICKNAMES), team_count, replace=False)
    order = rng.permutation(count)
    rosters = []
    for i, n in enumerate(team_names):
        jerseys = rng.choice(100, team_size, replace=False)
        rosters.append({'id': str(uuid5(UUID(league_id), f'team-{i}')),
                        'name': f'{PLACES[int(n) // len(NICKNAMES)]} {NICKNAMES[int(n) % len(NICKNAMES)]}',
                        'members': [{'player_id': players[int(order[i * team_size + j])]['id'],
                                     'jersey_number': int(jerseys[j])} for j in range(team_size)]})
    return {'schema_version': LEAGUE_SCHEMA_VERSION, 'league_id': league_id, 'name': name,
            'team_size': team_size, 'team_count': team_count, 'seed': seed, 'sampling': sampling,
            'skill_schema_version': registry.schema_version, 'skill_schema_hash': registry.schema_hash,
            'skill_descriptors': registry.schema_payload()['skills'], 'rules': shot_defense_contract(rules),
            'generation': {'sampler': 'fixed_player_pool', 'pool_id': pool.pool_id, 'name_set_version': NAME_SET_VERSION},
            'players': players, 'rosters': rosters}


def validate_league(draft, *, complete=False):
    """Validate drafts structurally; only publication requires complete membership."""
    if draft.get('schema_version') != LEAGUE_SCHEMA_VERSION:
        raise ValueError('Unsupported league artifact schema.')
    UUID(draft['league_id'])
    required_name(draft['name'], 'League name')
    if type(draft['team_size']) is not int or draft['team_size'] not in (3, 4, 5):
        raise ValueError('Team size must be 3, 4 or 5.')
    registry = registry_for_schema(draft['skill_schema_version'], draft['skill_schema_hash'])
    rules = draft['rules']['parameters']
    validate_shot_defense(rules)
    if draft['rules'] != shot_defense_contract(rules) or registry != registry_for_environment(rules):
        raise ValueError('League skill schema and rule contract are incompatible.')
    if draft['skill_descriptors'] != registry.schema_payload()['skills']:
        raise ValueError('League skill descriptors do not match its schema.')
    generation_settings(registry, draft['sampling'])
    players = {}
    for player in draft['players']:
        pid = player['id']
        UUID(pid)
        if pid in players:
            raise ValueError(f'Duplicate player identity: {pid}.')
        required_name(player['name'], 'Player name')
        for field in ('skills', 'generated_skills'):
            if set(player[field]) != set(registry.keys):
                raise ValueError(f'{player["name"]}: {field} must contain exactly the supported skills.')
            for d in registry.descriptors:
                try:
                    d.normalize_value(player[field][d.key])
                except ValueError as exc:
                    raise ValueError(f'{player["name"]}: {d.display_label} {exc}.') from exc
        expected = {k: v for k, v in player['skills'].items() if v != player['generated_skills'][k]}
        if player['skill_overrides'] != expected:
            raise ValueError(f'{player["name"]}: skill override provenance is inconsistent.')
        players[pid] = player
    if len(draft['rosters']) != draft['team_count'] or len(players) != draft['team_count'] * draft['team_size']:
        raise ValueError('Team/player counts do not match league generation settings.')
    team_ids, names, assigned = set(), set(), set()
    errors = []
    for roster in draft['rosters']:
        UUID(roster['id'])
        name = required_name(roster['name'], 'Team name')
        if roster['id'] in team_ids or name.casefold() in names:
            raise ValueError('Team identities and names must be unique within the league.')
        team_ids.add(roster['id'])
        names.add(name.casefold())
        numbers = set()
        if complete and len(roster['members']) != draft['team_size']:
            errors.append(f'{name} needs exactly {draft["team_size"]} players (has {len(roster["members"])}).')
        for member in roster['members']:
            pid, number = member['player_id'], member['jersey_number']
            if pid not in players or pid in assigned:
                raise ValueError(f'{name}: unknown player or duplicate membership ({pid}).')
            if type(number) is not int or not 0 <= number <= 99:
                raise ValueError(f'{name}: jersey numbers must be integers from 0–99.')
            if complete and number in numbers:
                errors.append(f'{name}: jersey #{number} is assigned more than once.')
            numbers.add(number)
            assigned.add(pid)
    if complete and set(players) - assigned:
        errors.append(f'{len(set(players) - assigned)} players are unassigned; assign every player before publishing.')
    if errors:
        raise ValueError(' '.join(errors))
    return draft


def validate_snapshot(snapshot, reference=None, environment=None):
    try:
        validate_league(snapshot, complete=True)
        if type(snapshot['version']) is not int or snapshot['version'] < 1:
            raise ValueError('Invalid published league version.')
        if snapshot.get('sha256') != snapshot_digest(snapshot):
            raise ValueError('League snapshot checksum mismatch.')
        if reference and any(snapshot[k] != reference[k] for k in ('league_id', 'version', 'sha256')):
            raise ValueError('League snapshot does not match the pinned version/checksum.')
        if environment is not None:
            if int(environment.get('players', 5)) != snapshot['team_size']:
                raise ValueError(f'League requires {snapshot["team_size"]} players per team; adjust the run team size.')
            if shot_defense_contract(environment) != snapshot['rules']:
                raise ValueError('League shot-defense rules do not match this run. Select a compatible published version or match its rule settings.')
    except (KeyError, TypeError) as exc:
        raise ValueError(f'Incomplete league artifact: {exc}.') from exc
    return snapshot


def load_league_snapshot(path, *, reference=None, environment=None):
    try:
        snapshot = json.loads(Path(path).read_bytes())
    except (OSError, ValueError) as exc:
        raise ValueError(f'Cannot load league snapshot {path}: {exc}') from exc
    return validate_snapshot(snapshot, reference, environment)
