"""Trainer league library: optimistic drafts in DuckDB, immutable JSON versions."""
from __future__ import annotations

import copy
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Annotated, Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, StrictInt

from app.training_backend.database import utc_now
from basketworld_jax.leagues import (
    canonical_bytes, generate_league, generation_settings, load_league_snapshot,
    snapshot_digest, validate_league,
)
from basketworld_jax.eval.skills import registry_for_environment
from basketworld_jax.shot_defense import SHOT_DEFENSE_DEFAULTS


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid')


class LeagueReference(StrictModel):
    league_id: UUID
    version: Annotated[StrictInt, Field(ge=1)]
    sha256: str = Field(pattern=r'^[0-9a-f]{64}$')


class LeagueCreate(StrictModel):
    name: str = Field(min_length=1, max_length=120)
    team_count: Annotated[StrictInt, Field(ge=2, le=512)] = 64
    team_size: Annotated[StrictInt, Field(ge=3, le=5)] = 5
    seed: Annotated[StrictInt, Field(ge=0, le=2_147_483_647)] = 0
    rules: dict[str, Any] = Field(default_factory=dict)
    sampling: dict[str, dict[str, StrictFloat]] = Field(default_factory=dict)


class EditablePlayer(StrictModel):
    id: UUID
    name: str
    skills: dict[str, StrictFloat]


class Member(StrictModel):
    player_id: UUID
    jersey_number: Annotated[StrictInt, Field(ge=0, le=99)]


class EditableRoster(StrictModel):
    id: UUID
    name: str
    members: list[Member]


class DraftUpdate(StrictModel):
    revision: Annotated[StrictInt, Field(ge=1)]
    name: str
    players: list[EditablePlayer]
    rosters: list[EditableRoster]


class PublishRequest(StrictModel):
    revision: Annotated[StrictInt, Field(ge=1)]


class ArchiveRequest(StrictModel):
    archived: StrictBool


class DeleteLeagueRequest(StrictModel):
    revision: Annotated[StrictInt, Field(ge=1)]
    name: str


class DraftConflict(RuntimeError):
    pass


def initialize_league_tables(connection):
    connection.execute('''CREATE TABLE IF NOT EXISTS leagues (
        id VARCHAR PRIMARY KEY, draft_json VARCHAR NOT NULL, revision INTEGER NOT NULL,
        archived BOOLEAN NOT NULL DEFAULT FALSE, created_at TIMESTAMPTZ NOT NULL,
        updated_at TIMESTAMPTZ NOT NULL)''')
    connection.execute('''CREATE TABLE IF NOT EXISTS league_versions (
        league_id VARCHAR NOT NULL, version INTEGER NOT NULL, sha256 VARCHAR NOT NULL,
        archived BOOLEAN NOT NULL DEFAULT FALSE, published_at TIMESTAMPTZ NOT NULL,
        PRIMARY KEY (league_id, version))''')


def _immutable_write(path, content):
    """Publish complete bytes atomically, never replacing an existing artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f'.{uuid4().hex}.tmp'
    try:
        with temporary.open('xb') as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class LeagueLibrary:
    def __init__(self, repository, data_dir):
        self.repository = repository
        self.data_dir = Path(data_dir)

    def _path(self, league_id, version):
        return self.data_dir / 'leagues' / str(UUID(str(league_id))) / f'v{int(version)}.json'

    def _get(self, connection, league_id):
        row = connection.execute('SELECT draft_json, revision, archived FROM leagues WHERE id=?', [str(league_id)]).fetchone()
        if row is None:
            raise KeyError('League not found.')
        versions = connection.execute('SELECT version, sha256, archived, published_at FROM league_versions WHERE league_id=? ORDER BY version DESC', [str(league_id)]).fetchall()
        return {'draft': json.loads(row[0]), 'revision': row[1], 'archived': row[2],
                'versions': [{'league_id': str(league_id), 'version': v, 'sha256': h, 'archived': a, 'published_at': t.isoformat()} for v, h, a, t in versions]}

    def get(self, league_id):
        with self.repository._lock, self.repository._connection() as connection:
            return self._get(connection, str(league_id))

    def list(self, include_archived=False):
        with self.repository._lock, self.repository._connection() as connection:
            rows = connection.execute('SELECT id, draft_json, revision, archived FROM leagues ORDER BY created_at DESC').fetchall()
            result = []
            for lid, raw, revision, archived in rows:
                if archived and not include_archived:
                    continue
                draft = json.loads(raw)
                result.append({k: draft[k] for k in ('league_id', 'name', 'team_size', 'team_count', 'skill_schema_version')} | {'revision': revision, 'archived': archived})
            return result

    def create(self, request):
        draft = generate_league(**request.model_dump())
        validate_league(draft, complete=True)
        now = utc_now()
        with self.repository._lock, self.repository._connection() as connection:
            connection.execute('INSERT INTO leagues VALUES (?, ?, 1, FALSE, ?, ?)', [draft['league_id'], canonical_bytes(draft).decode(), now, now])
            return self._get(connection, draft['league_id'])

    def update(self, league_id, request):
        with self.repository._lock, self.repository._connection() as connection:
            current = self._get(connection, str(league_id))
            self._require_editable(current, request.revision)
            draft = copy.deepcopy(current['draft'])
            patch = request.model_dump(mode='json')
            for field in ('players', 'rosters'):
                old_ids = {v['id'] for v in draft[field]}
                new_ids = [v['id'] for v in patch[field]]
                if len(new_ids) != len(old_ids) or set(new_ids) != old_ids:
                    raise ValueError(f'Draft edits must preserve all {field} identities.')
            draft['name'] = patch['name']
            players = {v['id']: v for v in draft['players']}
            for player in patch['players']:
                stored = players[player['id']]
                stored.update(player)
                stored['skill_overrides'] = {k: v for k, v in player['skills'].items() if v != stored['generated_skills'].get(k)}
            draft['rosters'] = patch['rosters']
            validate_league(draft)
            connection.execute('UPDATE leagues SET draft_json=?, revision=revision+1, updated_at=? WHERE id=?', [canonical_bytes(draft).decode(), utc_now(), str(league_id)])
            return self._get(connection, str(league_id))

    @staticmethod
    def _require_editable(current, revision):
        if current['revision'] != revision:
            raise DraftConflict('This draft changed in another tab. Reload the league before editing or publishing.')
        if current['archived']:
            raise DraftConflict('Restore this archived league before editing or publishing.')

    def publish(self, league_id, revision):
        with self.repository._lock, self.repository._connection() as connection:
            current = self._get(connection, str(league_id))
            self._require_editable(current, revision)
            validate_league(current['draft'], complete=True)
            version = max([v['version'] for v in current['versions']], default=0) + 1
            now = utc_now()
            snapshot = {**current['draft'], 'version': version, 'draft_revision': revision, 'published_at': now.isoformat()}
            snapshot['sha256'] = snapshot_digest(snapshot)
            path = self._path(league_id, version)
            _immutable_write(path, canonical_bytes(snapshot))
            try:
                connection.execute('INSERT INTO league_versions VALUES (?, ?, ?, FALSE, ?)', [str(league_id), version, snapshot['sha256'], now])
            except Exception:
                path.unlink()
                raise
            return snapshot

    def archive(self, league_id, archived, version=None):
        with self.repository._lock, self.repository._connection() as connection:
            current = self._get(connection, str(league_id))
            if version is None:
                connection.execute('UPDATE leagues SET archived=?, updated_at=? WHERE id=?', [archived, utc_now(), str(league_id)])
            else:
                if not any(v['version'] == version for v in current['versions']):
                    raise KeyError('Published version not found.')
                connection.execute('UPDATE league_versions SET archived=? WHERE league_id=? AND version=?', [archived, str(league_id), version])
            return self._get(connection, str(league_id))

    def delete(self, league_id, request):
        league_id = str(UUID(str(league_id)))
        directory = self.data_dir / 'leagues' / league_id
        staged = directory.with_name(f'.deleted-{league_id}-{uuid4().hex}')
        moved = False
        with self.repository._lock, self.repository._connection() as connection:
            current = self._get(connection, league_id)
            if request.revision != current['revision'] or request.name != current['draft']['name']:
                raise DraftConflict('The league changed. Reload it and confirm its current name before deleting.')
            preserved_run_ids = set()
            for run_id, status, raw in connection.execute('SELECT id, status, config_json FROM training_runs').fetchall():
                ref = json.loads(raw).get('league_reference') or {}
                if ref.get('league_id') == league_id:
                    if status not in {'stopped', 'completed', 'failed'}:
                        raise DraftConflict('This league is referenced by a training run that is active or paused. Stop the run before deleting the league, or archive it.')
                    try:
                        load_league_snapshot(self.data_dir / 'runs' / str(run_id) / 'league.json', reference=ref)
                    except ValueError as exc:
                        raise DraftConflict(f'Cannot delete this league: run {run_id} lacks a valid saved league snapshot. Archive it instead.') from exc
                    preserved_run_ids.add(str(run_id))
            # A run snapshot can also exist before its run record is created.
            for path in (self.data_dir / 'runs').glob('*/league.json'):
                if path.parent.name in preserved_run_ids:
                    continue
                if json.loads(path.read_text()).get('league_id') == league_id:
                    raise DraftConflict('This league has a saved run snapshot. Archive it instead.')
            connection.execute('BEGIN TRANSACTION')
            try:
                if directory.exists():
                    directory.rename(staged)
                    moved = True
                connection.execute('DELETE FROM league_versions WHERE league_id=?', [league_id])
                connection.execute('DELETE FROM leagues WHERE id=?', [league_id])
                connection.execute('COMMIT')
            except Exception:
                connection.execute('ROLLBACK')
                if moved:
                    staged.rename(directory)
                raise
        if moved:
            try:
                shutil.rmtree(staged)
            except OSError:
                logging.getLogger(__name__).exception('Could not clean up deleted league files at %s', staged)
        return {'deleted': True, 'league_id': league_id}

    def load(self, reference, *, environment=None, allow_archived=False):
        ref = reference.model_dump(mode='json') if isinstance(reference, LeagueReference) else reference
        current = self.get(ref['league_id'])
        version = next((v for v in current['versions'] if v['version'] == ref['version']), None)
        if version is None:
            raise ValueError('Published league version not found.')
        if not allow_archived and (current['archived'] or version['archived']):
            raise ValueError('This league or version is archived. Restore it before selecting it for a new run.')
        if ref['sha256'] != version['sha256']:
            raise ValueError('Published league checksum does not match the selected reference.')
        return load_league_snapshot(self._path(ref['league_id'], ref['version']), reference=ref, environment=environment)

    def pin_for_run(self, reference, run_id, *, environment):
        """Slice-2 integration contract: copy exact published bytes before launch."""
        UUID(str(run_id))
        snapshot = self.load(reference, environment=environment)
        path = self.data_dir / 'runs' / str(run_id) / 'league.json'
        content = canonical_bytes(snapshot)
        if path.exists():
            if path.read_bytes() != content:
                raise ValueError('Run already has a different league snapshot; it cannot be replaced.')
        else:
            _immutable_write(path, content)
        return snapshot

    def restore_for_run(self, reference, run_id, *, environment):
        """Resume/extension never consults mutable drafts or regenerates a pool."""
        UUID(str(run_id))
        ref = reference.model_dump(mode='json') if isinstance(reference, LeagueReference) else reference
        return load_league_snapshot(self.data_dir / 'runs' / str(run_id) / 'league.json', reference=ref, environment=environment)


def league_options(preset):
    profiles = []
    for label, flags in [('Legacy skills', (False, False)), ('Rim protection', (True, False)),
                         ('Contested shooting', (False, True)), ('Rim protection + contested shooting', (True, True))]:
        rules = {**SHOT_DEFENSE_DEFAULTS, 'rim_protection_enabled': flags[0], 'contested_shooting_enabled': flags[1]}
        registry = registry_for_environment(rules)
        sampling = generation_settings(registry)
        for d in registry.descriptors:
            defaults = dict(d.sampling_defaults)
            if 'mean' in sampling[d.key]:
                sampling[d.key]['mean'] = preset.get(defaults.get('mean_field'), sampling[d.key]['mean'])
            if d.semantic_group == 'shooting':
                sampling[d.key]['std'] = preset.get(defaults.get('std_field'), sampling[d.key]['std'])
        profiles.append({'label': label, 'rules': rules, 'skills': registry.schema_payload()['skills'],
                         'sampling': sampling, 'skill_schema_version': registry.schema_version})
    return {'profiles': profiles, 'team_sizes': [3, 4, 5], 'default_team_count': 64,
            'league_training_available': True}
