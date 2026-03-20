from app.backend.evaluation import (
    _accumulate_intent_selection,
    _init_eval_diagnostics,
    _merge_eval_diagnostics,
)


class _DummyEnv:
    def __init__(self, *, enabled=True, active=True, intent_index=0):
        self.enable_intent_learning = enabled
        self.intent_active = active
        self.intent_index = intent_index


def test_accumulate_intent_selection_counts_active_and_inactive():
    diagnostics = _init_eval_diagnostics()

    _accumulate_intent_selection(diagnostics, _DummyEnv(enabled=True, active=True, intent_index=3))
    _accumulate_intent_selection(diagnostics, _DummyEnv(enabled=True, active=True, intent_index=3))
    _accumulate_intent_selection(diagnostics, _DummyEnv(enabled=True, active=False, intent_index=7))

    assert diagnostics["intent_selection_counts"] == {"3": 2}
    assert diagnostics["intent_inactive_count"] == 1


def test_merge_eval_diagnostics_merges_intent_counts():
    dest = _init_eval_diagnostics()
    src = _init_eval_diagnostics()

    dest["intent_selection_counts"] = {"1": 2, "4": 1}
    dest["intent_inactive_count"] = 3

    src["intent_selection_counts"] = {"1": 5, "7": 4}
    src["intent_inactive_count"] = 2

    merged = _merge_eval_diagnostics(dest, src)

    assert merged["intent_selection_counts"] == {"1": 7, "4": 1, "7": 4}
    assert merged["intent_inactive_count"] == 5
