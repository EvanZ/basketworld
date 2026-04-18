from __future__ import annotations

from types import SimpleNamespace

from app.backend.policies import get_unified_policy_path, list_policies_from_run


class _FakeFileInfo:
    def __init__(self, path: str):
        self.path = path


class _FakeRunData:
    def __init__(self, tags=None):
        self.tags = dict(tags or {})


class _FakeRun:
    def __init__(self, tags=None):
        self.data = _FakeRunData(tags=tags)


class _FakeMlflowClient:
    def __init__(self, artifact_map, *, tags=None):
        self.artifact_map = {
            key: [_FakeFileInfo(path) for path in paths]
            for key, paths in artifact_map.items()
        }
        self.tags = dict(tags or {})
        self.download_calls = []

    def list_artifacts(self, run_id, path):
        return list(self.artifact_map.get(path, []))

    def get_run(self, run_id):
        return _FakeRun(tags=self.tags)

    def download_artifacts(self, run_id, artifact_path, cache_dir):
        self.download_calls.append((run_id, artifact_path, cache_dir))
        return f"{cache_dir}/{artifact_path.rsplit('/', 1)[-1]}"


def test_list_policies_from_run_includes_jax_checkpoints():
    client = _FakeMlflowClient(
        {
            "models": [
                "models/update_0000100",
                "models/update_0000200",
            ],
        }
    )

    paths = list_policies_from_run(client, "run-1")
    assert paths == [
        "models/update_0000100",
        "models/update_0000200",
    ]


def test_get_unified_policy_path_prefers_tagged_latest_jax_checkpoint():
    client = _FakeMlflowClient(
        {
            "models": [
                "models/update_0000100",
                "models/update_0000200",
            ],
        },
        tags={"jax_phase_a_latest_checkpoint_artifact": "models/update_0000100"},
    )

    local_path = get_unified_policy_path(client, "run-1", None)

    assert client.download_calls == [
        ("run-1", "models/update_0000100", "episodes/_policy_cache")
    ]
    assert local_path.endswith("update_0000100")


def test_get_unified_policy_path_still_resolves_latest_sb3_zip():
    client = _FakeMlflowClient(
        {
            "models": [
                "models/unified_policy_100.zip",
                "models/unified_policy_200.zip",
            ],
        }
    )

    local_path = get_unified_policy_path(client, "run-2", None)

    assert client.download_calls == [
        ("run-2", "models/unified_policy_200.zip", "episodes/_policy_cache")
    ]
    assert local_path.endswith("unified_policy_200.zip")
