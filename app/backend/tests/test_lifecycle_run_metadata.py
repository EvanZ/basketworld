from types import SimpleNamespace

from app.backend.routes import lifecycle_routes


class _RunClient:
    def __init__(self, params):
        self.params = params

    def get_run(self, _run_id):
        return SimpleNamespace(data=SimpleNamespace(params=self.params))


def test_run_metadata_recognizes_multi_possession_mode():
    assert lifecycle_routes._run_uses_multi_possession(
        _RunClient({"enable_multi_possession": "true"}), "run-1"
    ) is True
    assert lifecycle_routes._run_uses_multi_possession(
        _RunClient({"enable_multi_possession": "false"}), "run-2"
    ) is False
    assert lifecycle_routes._run_uses_multi_possession(
        _RunClient({"jax/env/enable_multi_possession": "true"}), "run-3"
    ) is True
