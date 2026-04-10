from types import SimpleNamespace

from train.train_utils import resolve_spa_schedule


def _args(**overrides):
    base = {
        "continue_run_id": None,
        "steps_per_alternation": 1,
        "steps_per_alternation_end": 1,
        "steps_per_alternation_schedule": "linear",
        "alternations": 1,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_resolve_spa_schedule_extend_uses_previous_schedule_metadata():
    args = _args(
        continue_run_id="run123",
        steps_per_alternation=1,
        steps_per_alternation_end=1,
        steps_per_alternation_schedule="linear",
        alternations=2,
    )
    previous = {
        "spa_start": 20,
        "spa_end": 150,
        "spa_schedule": "log",
    }

    spa_start, spa_end, spa_schedule, spa_offset, spa_total = resolve_spa_schedule(
        args,
        "extend",
        previous,
        base_alt_idx=7,
    )

    assert spa_start == 20
    assert spa_end == 150
    assert spa_schedule == "log"
    assert spa_offset == 7
    assert spa_total == 9


def test_resolve_spa_schedule_restart_uses_current_launch_values():
    args = _args(
        continue_run_id="run123",
        steps_per_alternation=1,
        steps_per_alternation_end=1,
        steps_per_alternation_schedule="log",
        alternations=1,
    )
    previous = {
        "spa_start": 20,
        "spa_end": 150,
        "spa_schedule": "log",
    }

    spa_start, spa_end, spa_schedule, spa_offset, spa_total = resolve_spa_schedule(
        args,
        "restart",
        previous,
        base_alt_idx=7,
    )

    assert spa_start == 1
    assert spa_end == 1
    assert spa_schedule == "log"
    assert spa_offset == 0
    assert spa_total == 1


def test_resolve_spa_schedule_constant_uses_current_launch_values_and_locks_steps():
    args = _args(
        continue_run_id="run123",
        steps_per_alternation=2,
        steps_per_alternation_end=5,
        steps_per_alternation_schedule="linear",
        alternations=3,
    )
    previous = {
        "spa_start": 20,
        "spa_end": 150,
        "spa_schedule": "log",
    }

    spa_start, spa_end, spa_schedule, spa_offset, spa_total = resolve_spa_schedule(
        args,
        "constant",
        previous,
        base_alt_idx=7,
    )

    assert spa_start == 5
    assert spa_end == 5
    assert spa_schedule == "linear"
    assert spa_offset == 0
    assert spa_total == 3
