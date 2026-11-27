from copy import deepcopy
import json
from pathlib import Path
from uuid import uuid4

import pytest

from app.training_backend.database import TrainingRepository
from app.training_backend.leagues import (
    DeleteLeagueRequest, DraftConflict, DraftUpdate, LeagueCreate, LeagueLibrary, LeagueReference, league_options,
)
from basketworld_jax.eval.fixed_pool import generate_player_pool
from basketworld_jax.eval.skills import registry_for_environment
from basketworld_jax.leagues import generate_league, load_league_snapshot, validate_snapshot


@pytest.fixture
def library(tmp_path):
    repository = TrainingRepository(tmp_path / 'custom-db' / 'trainer.duckdb')
    repository.initialize()
    return LeagueLibrary(repository, tmp_path / 'custom-data')


def edit_request(current):
    draft = current['draft']
    return {'revision': current['revision'], 'name': draft['name'],
            'players': [{k: deepcopy(p[k]) for k in ('id', 'name', 'skills')} for p in draft['players']],
            'rosters': deepcopy(draft['rosters'])}


def create(library, **kwargs):
    return library.create(LeagueCreate(name='Test league', team_count=2, **kwargs))


def reference(snapshot):
    return LeagueReference(**{k: snapshot[k] for k in ('league_id', 'version', 'sha256')})


@pytest.mark.parametrize('published', [False, True])
def test_delete_league_removes_only_selected_league(library, published):
    current = create(library)
    other = create(library)
    lid = current['draft']['league_id']
    other_snapshot = library.publish(other['draft']['league_id'], other['revision'])
    if published:
        library.publish(lid, current['revision'])
        library.archive(lid, True)
    request = DeleteLeagueRequest(revision=current['revision'], name=current['draft']['name'])
    assert library.delete(lid, request)['deleted']
    with pytest.raises(KeyError):
        library.get(lid)
    assert not library._path(lid, 1).parent.exists()
    assert not list((library.data_dir / 'leagues').glob('.deleted-*'))
    assert library.load(reference(other_snapshot)) == other_snapshot
    with library.repository._connection() as connection:
        assert connection.execute('SELECT count(*) FROM league_versions WHERE league_id=?', [lid]).fetchone()[0] == 0


def test_delete_requires_current_revision_and_name(library):
    current = create(library)
    lid = current['draft']['league_id']
    snapshot = library.publish(lid, current['revision'])
    for revision, name in [(2, 'Test league'), (1, 'Wrong name')]:
        with pytest.raises(DraftConflict):
            library.delete(lid, DeleteLeagueRequest(revision=revision, name=name))
        assert library.get(lid)['draft'] == current['draft']
        assert library.load(reference(snapshot)) == snapshot


def test_delete_blocks_saved_run_snapshot(library):
    current = create(library)
    lid = current['draft']['league_id']
    snapshot = library.publish(lid, current['revision'])
    path = library.data_dir / 'runs' / str(uuid4()) / 'league.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(snapshot))
    with pytest.raises(DraftConflict, match='saved run snapshot'):
        library.delete(lid, DeleteLeagueRequest(revision=1, name='Test league'))
    assert library.load(reference(snapshot)) == snapshot
    assert json.loads(path.read_text()) == snapshot


@pytest.mark.parametrize('size', [3, 4, 5])
@pytest.mark.parametrize('flags', [(False, False), (True, False), (False, True), (True, True)])
def test_generation_matches_existing_sampler_and_roundtrips_all_skills(library, size, flags):
    rules = dict(rim_protection_enabled=flags[0], contested_shooting_enabled=flags[1])
    current = create(library, team_size=size, seed=42, rules=rules,
                     sampling={'layup': {'mean': .6, 'std': .1}})
    draft = current['draft']
    registry = registry_for_environment(rules)
    original = generate_player_pool(player_count=2 * size, seed=42, registry=registry, sampling_overrides=draft['sampling'])
    assert [p['skills'] for p in draft['players']] == [dict(p.skills) for p in original.players]
    assert len(draft['players'][0]['skills']) == (11 if any(flags) else 9)
    assert all(len(t['members']) == size for t in draft['rosters'])
    published = library.publish(draft['league_id'], current['revision'])
    assert library.load(reference(published)) == published
    assert load_league_snapshot(library._path(draft['league_id'], 1)) == published


def test_generated_names_are_reproducible_unique_and_independent_of_skills():
    for seed in (0, 1, 2, 42, 99):
        a = generate_league(name='A', seed=seed)
        b = generate_league(name='B', seed=seed, sampling={'layup': {'mean': .8, 'std': .1}})
        assert len({p['name'] for p in a['players']}) == 320
        assert [p['name'] for p in a['players']] == [p['name'] for p in b['players']]
        assert len({t['name'] for t in a['rosters']}) == 64
        assert not ({p['id'] for p in a['players']} & {p['id'] for p in b['players']})
        for team in a['rosters']:
            assert len({m['jersey_number'] for m in team['members']}) == 5
    large = generate_league(name='Large', team_count=512)
    assert len({p['name'] for p in large['players']}) == 2560
    assert len({t['name'] for t in large['rosters']}) == 512


def test_drafts_and_exact_versions_survive_new_repository_instance(library):
    first = create(library)
    second = library.create(LeagueCreate(name='Other league', team_count=2, team_size=3))
    assert first['draft']['league_id'] != second['draft']['league_id']
    published = library.publish(first['draft']['league_id'], first['revision'])
    patch = edit_request(first)
    patch['players'][0]['name'] = 'Edited player'
    patch['players'][0]['skills']['layup'] = .7312345678901234
    member = patch['rosters'][0]['members'].pop()
    patch['rosters'][1]['members'].append(member)
    saved = library.update(first['draft']['league_id'], DraftUpdate(**patch))
    repository = TrainingRepository(library.repository.database_path)
    repository.initialize()
    restarted = LeagueLibrary(repository, library.data_dir)
    assert restarted.get(first['draft']['league_id']) == saved
    p = saved['draft']['players'][0]
    assert p['skill_overrides'] == {'layup': .7312345678901234}
    assert p['generated_skills'] == first['draft']['players'][0]['skills']
    assert p['provenance'] == first['draft']['players'][0]['provenance']
    assert restarted.load(reference(published)) == published
    with pytest.raises(ValueError, match='exactly 5'):
        restarted.publish(first['draft']['league_id'], saved['revision'])


def test_incomplete_drafts_publish_errors_and_optimistic_conflicts(library):
    current = create(library)
    lid = current['draft']['league_id']
    patch = edit_request(current)
    patch['rosters'][0]['members'].pop()
    updated = library.update(lid, DraftUpdate(**patch))
    with pytest.raises(ValueError, match='unassigned'):
        library.publish(lid, updated['revision'])
    for operation in (lambda: library.update(lid, DraftUpdate(**patch)), lambda: library.publish(lid, current['revision'])):
        with pytest.raises(DraftConflict, match='another tab'):
            operation()
    assert library.get(lid)['versions'] == []


@pytest.mark.parametrize('fault, message', [
    ('duplicate', 'duplicate membership'), ('unknown', 'unknown player'),
    ('team_name', 'unique'), ('missing_skill', 'exactly the supported skills'),
    ('extra_skill', 'exactly the supported skills'), ('range', 'between 0 and 1'),
    ('identity', 'preserve all players'),
])
def test_invalid_draft_edits_do_not_mutate_saved_data(library, fault, message):
    current = create(library)
    patch = edit_request(current)
    if fault == 'duplicate': patch['rosters'][1]['members'].append(patch['rosters'][0]['members'][0])
    if fault == 'unknown': patch['rosters'][0]['members'][0]['player_id'] = str(uuid4())
    if fault == 'team_name': patch['rosters'][1]['name'] = patch['rosters'][0]['name']
    if fault == 'missing_skill': del patch['players'][0]['skills']['layup']
    if fault == 'extra_skill': patch['players'][0]['skills']['rim_protection'] = 1
    if fault == 'range': patch['players'][0]['skills']['layup'] = 1.1
    if fault == 'identity': patch['players'][0]['id'] = str(uuid4())
    with pytest.raises(ValueError, match=message):
        library.update(current['draft']['league_id'], DraftUpdate(**patch))
    assert library.get(current['draft']['league_id']) == current


def test_duplicate_jerseys_block_publication_but_allow_draft_repair(library):
    current = create(library)
    patch = edit_request(current)
    patch['rosters'][0]['members'][1]['jersey_number'] = patch['rosters'][0]['members'][0]['jersey_number']
    updated = library.update(current['draft']['league_id'], DraftUpdate(**patch))
    with pytest.raises(ValueError, match='jersey #'):
        library.publish(current['draft']['league_id'], updated['revision'])


def test_versions_and_multiple_run_copies_stay_exact_when_draft_changes_or_archives(library):
    current = create(library)
    lid = current['draft']['league_id']
    v1 = library.publish(lid, current['revision'])
    ref = reference(v1)
    environment = {'players': 5, **v1['rules']['parameters']}
    runs = [str(uuid4()), str(uuid4())]
    for run in runs:
        assert library.pin_for_run(ref, run, environment=environment) == v1
        assert (library.data_dir / 'runs' / run / 'league.json').read_bytes() == library._path(lid, 1).read_bytes()
    patch = edit_request(current); patch['players'][0]['skills']['layup'] = .123
    updated = library.update(lid, DraftUpdate(**patch))
    v2 = library.publish(lid, updated['revision'])
    assert v2['version'] == 2 and v2['sha256'] != v1['sha256']
    with pytest.raises(ValueError, match='cannot be replaced'):
        library.pin_for_run(reference(v2), runs[0], environment=environment)
    library.archive(lid, True, 1); library.archive(lid, True)
    assert library.list() == [] and len(library.list(True)) == 1
    with pytest.raises(ValueError, match='archived'):
        library.pin_for_run(ref, str(uuid4()), environment=environment)
    assert library.load(ref, allow_archived=True) == v1
    # Restore/extension use the run's copy even if the library file is unavailable.
    library._path(lid, 1).unlink()
    for run in runs:
        assert library.restore_for_run(ref, run, environment=environment) == v1
    library.archive(lid, False); library.archive(lid, False, 1)
    with pytest.raises(ValueError, match='Cannot load'):
        library.load(ref)


def test_missing_corrupt_and_incompatible_artifacts_fail_explicitly(library):
    current = create(library)
    v = library.publish(current['draft']['league_id'], current['revision'])
    ref = reference(v)
    with pytest.raises(ValueError, match='requires 5 players'):
        library.load(ref, environment={'players': 3})
    with pytest.raises(ValueError, match='rules do not match'):
        library.load(ref, environment={'players': 5, 'rim_protection_enabled': True})
    bad = deepcopy(v); bad['players'][0]['name'] = 'Corrupted'
    with pytest.raises(ValueError, match='checksum mismatch'):
        validate_snapshot(bad)
    with pytest.raises(ValueError, match='Cannot load'):
        library.restore_for_run(ref, str(uuid4()), environment={'players': 5})
    ref.sha256 = '0' * 64
    with pytest.raises(ValueError, match='checksum'):
        library.load(ref)


def test_http_authoring_and_training_capability_gate(library, monkeypatch):
    from fastapi.testclient import TestClient
    from app.training_backend import main
    from app.training_backend.config import Settings
    from app.training_backend.jobs import JobManager
    settings = Settings(Path(__file__).resolve().parents[3], library.data_dir,
                        library.repository.database_path, (), 1)
    monkeypatch.setattr(main, 'repository', library.repository)
    monkeypatch.setattr(main, 'settings', settings)
    monkeypatch.setattr(main, 'manager', JobManager(settings, library.repository))
    client = TestClient(main.app)
    opts = client.get('/api/v1/leagues/options').json()
    assert opts['league_training_available'] is False
    assert len(opts['profiles']) == 4
    assert client.post('/api/v1/leagues', json={'name':'Bad', 'team_size': True}).status_code == 422
    created = client.post('/api/v1/leagues', json={'name': 'API League', 'team_count': 2}).json()
    lid = created['draft']['league_id']
    published = client.post(f'/api/v1/leagues/{lid}/versions', json={'revision': 1})
    assert published.status_code == 201
    ref = reference(published.json()).model_dump(mode='json')
    launch = client.post('/api/v1/runs', json={'config': {'league_reference': ref}, 'idempotency_key': 'unavailable-league'})
    assert launch.status_code == 409 and 'slice 2' in launch.json()['detail']
    assert library.repository.list_runs() == []
    assert client.get(f'/api/v1/leagues/{lid}/versions/1').json() == published.json()
    assert client.post(f'/api/v1/leagues/{lid}/archive', json={'archived': True}).status_code == 200
    assert client.get('/api/v1/leagues').json() == []
    assert len(client.get('/api/v1/leagues?include_archived=true').json()) == 1
    assert client.put(f'/api/v1/leagues/{lid}/draft', json=edit_request(created)).status_code == 409


# Use the real run lifecycle with a stubbed worker. The capability gate remains
# disabled in production; this exercises the slice-2 contract without training.
from app.training_backend.tests.test_jobs import paused_schedule_run


def test_delete_blocks_training_run_reference(paused_schedule_run):
    manager, repository, run, _ = paused_schedule_run
    lib = LeagueLibrary(repository, manager.settings.data_dir)
    current = create(lib)
    lid = current['draft']['league_id']
    snapshot = lib.publish(lid, current['revision'])
    config = {**run['config'], 'league_reference': reference(snapshot).model_dump(mode='json')}
    repository.update_run(run['id'], config_json=json.dumps(config))
    with pytest.raises(DraftConflict, match='referenced by a training run'):
        lib.delete(lid, DeleteLeagueRequest(revision=1, name='Test league'))
    assert lib.load(reference(snapshot)) == snapshot
    assert repository.get_run(run['id'])['config']['league_reference'] == config['league_reference']


def test_run_copy_resume_and_extension_keep_pinned_snapshot(paused_schedule_run, monkeypatch):
    from app.training_backend import jobs
    from app.training_backend.models import TrainingRunConfig
    from app.training_backend.application_training import resolve_run_training_config
    manager, repository, run, _ = paused_schedule_run
    lib = LeagueLibrary(repository, manager.settings.data_dir)
    environment = resolve_run_training_config(tuple(run['command']))
    from basketworld_jax.shot_defense import shot_defense_values
    current = lib.create(LeagueCreate(name='Lifecycle', team_count=2, team_size=environment['players'],
                                    rules=shot_defense_values(environment)))
    published = lib.publish(current['draft']['league_id'], current['revision'])
    ref = reference(published)
    lib.pin_for_run(ref, run['id'], environment=environment)
    config = {**run['config'], 'league_reference': ref.model_dump(mode='json')}
    repository.update_run(run['id'], config_json=json.dumps(config))
    monkeypatch.setattr(jobs, 'LEAGUE_TRAINING_AVAILABLE', True)
    resumed = manager.resume(run['id'], confirm_provenance_mismatch=True)
    assert resumed['config']['league_reference'] == config['league_reference']
    # Editing/publishing later does not change copied-run selection or snapshot.
    patch = edit_request(current); patch['players'][0]['skills']['layup'] = .314
    changed = lib.update(current['draft']['league_id'], DraftUpdate(**patch))
    lib.publish(current['draft']['league_id'], changed['revision'])
    repository.update_run(run['id'], status='completed')
    copied = manager.create_and_launch(config=TrainingRunConfig.model_validate(config), idempotency_key='league-copy-test')
    assert copied['config']['league_reference'] == config['league_reference']
    assert lib.restore_for_run(ref, copied['id'], environment=environment) == published
    repository.update_run(copied['id'], status='completed')
    lib.archive(current['draft']['league_id'], True)
    continued = manager.continue_completed(run['id'], additional_updates=1000, confirm_provenance_mismatch=True)
    assert continued['config']['league_reference'] == config['league_reference']
    assert lib.restore_for_run(ref, run['id'], environment=environment) == published
    # Resume cannot silently regenerate a missing snapshot.
    repository.update_run(run['id'], status='paused')
    (manager.settings.data_dir / 'runs' / run['id'] / 'league.json').unlink()
    with pytest.raises(ValueError, match='Cannot load league snapshot'):
        manager.resume(run['id'], confirm_provenance_mismatch=True)
    assert repository.get_run(run['id'])['status'] == 'paused'


@pytest.mark.parametrize('value', [None, True, 'invalid', float('nan')])
def test_invalid_rule_parameter_has_actionable_error(value):
    with pytest.raises(ValueError, match='rim_protection_beta_alpha must be a finite number'):
        generate_league(name='Bad rule', team_count=2, rules={'rim_protection_beta_alpha': value})
