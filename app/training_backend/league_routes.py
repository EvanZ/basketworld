"""HTTP adapter for the persistent league library."""
from uuid import UUID

from fastapi import APIRouter, HTTPException

from app.training_backend.leagues import (
    ArchiveRequest, DeleteLeagueRequest, DraftConflict, DraftUpdate, LeagueCreate, LeagueLibrary,
    PublishRequest, league_options,
)
from app.training_backend.application_training import application_preset_values


def league_router(repository_getter, settings_getter):
    router = APIRouter(prefix='/api/v1/leagues', tags=['leagues'])

    def library():
        return LeagueLibrary(repository_getter(), settings_getter().data_dir)

    def invoke(operation):
        try:
            return operation()
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except DraftConflict as exc:
            raise HTTPException(409, str(exc)) from exc
        except (ValueError, FileExistsError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @router.get('/options')
    def options():
        return league_options(application_preset_values(settings_getter().repo_root))

    @router.get('')
    def list_leagues(include_archived: bool = False):
        return library().list(include_archived)

    @router.post('', status_code=201)
    def create(request: LeagueCreate):
        return invoke(lambda: library().create(request))

    @router.get('/{league_id}')
    def get(league_id: UUID):
        return invoke(lambda: library().get(league_id))

    @router.put('/{league_id}/draft')
    def update(league_id: UUID, request: DraftUpdate):
        return invoke(lambda: library().update(league_id, request))

    @router.delete('/{league_id}')
    def delete(league_id: UUID, request: DeleteLeagueRequest):
        return invoke(lambda: library().delete(league_id, request))

    @router.post('/{league_id}/versions', status_code=201)
    def publish(league_id: UUID, request: PublishRequest):
        return invoke(lambda: library().publish(league_id, request.revision))

    @router.get('/{league_id}/versions/{version}')
    def version(league_id: UUID, version: int):
        def read():
            lib = library()
            current = lib.get(league_id)
            ref = next((v for v in current['versions'] if v['version'] == version), None)
            if ref is None:
                raise KeyError('Published version not found.')
            return lib.load(ref, allow_archived=True)
        return invoke(read)

    @router.post('/{league_id}/archive')
    def archive(league_id: UUID, request: ArchiveRequest):
        return invoke(lambda: library().archive(league_id, request.archived))

    @router.post('/{league_id}/versions/{version}/archive')
    def archive_version(league_id: UUID, version: int, request: ArchiveRequest):
        return invoke(lambda: library().archive(league_id, request.archived, version))

    return router
