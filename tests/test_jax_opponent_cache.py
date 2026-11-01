from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from basketworld_jax.train.opponent_cache import (
    OpponentSnapshotCache,
    validate_cache_config,
)
from basketworld_jax.train.pfsp import PFSPState


def make_cache(capacity=4, interval=10, candidates=None, state=None, pending=None):
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    return OpponentSnapshotCache(
        [] if candidates is None else candidates,
        capacity=capacity,
        interval=interval,
        template={"w": jnp.zeros(2)},
        jax=jax,
        jnp=jnp,
        state=state,
        pending_params=pending,
    )


def snapshot(cache, update, assignments=None):
    params = {"w": cache.jnp.full(2, float(update))}
    return cache.advance(
        update=update, assignments=assignments or {}, snapshot_params=params
    )


def test_fixed_shape_no_writes_between_snapshots_and_exact_masks():
    cache = make_cache()
    assert cache.bank["w"].shape == (4, 2)
    np.testing.assert_array_equal(cache.probabilities([]), np.zeros((2, 4)))
    snapshot(cache, 1)
    bank = cache.bank
    cache.advance(update=2, assignments={})
    assert cache.bank is bank
    assert cache.bank_writes == 1
    with pytest.raises(ValueError, match="cadence"):
        snapshot(cache, 3)
    snapshot(cache, 10)
    assert cache.bank["w"].shape == (4, 2)
    np.testing.assert_array_equal(
        cache.probabilities([0.25, 0.75]), [[0.25, 0.75, 0, 0]] * 2
    )
    np.testing.assert_array_equal(cache.bank["w"][:2], [[1, 1], [10, 10]])
    assert cache._write._cache_size() == 1


def test_retirement_drains_games_and_pending_is_bounded_and_resumable():
    cache = make_cache(capacity=2)
    snapshot(cache, 1)
    snapshot(cache, 10)
    snapshot(cache, 30, {"offense": np.array([0]), "defense": np.array([1])})
    assert cache.retiring == 0
    assert cache.pending_info["update_index"] == 30
    assert cache.bank_writes == 2
    np.testing.assert_array_equal(cache.bank["w"][0], [1, 1])
    np.testing.assert_array_equal(cache.probabilities([0.5, 0.5]), [[0, 1], [0, 1]])
    snapshot(cache, 40, {"offense": np.array([0])})
    assert cache.pending_info["update_index"] == 40
    assert cache.superseded_pending == 1
    restored = make_cache(
        capacity=2,
        candidates=deepcopy(cache.candidates),
        state=cache.payload(),
        pending=cache.pending_params,
    )
    for item in (cache, restored):
        event = item.advance(update=41, assignments={"offense": np.array([1])})
        assert event["snapshot"] == 40
        assert event["evicted"] == "self_u1"
        assert item.bank_writes == 3
        assert len(item.candidates) == 2
        assert item.pending_params is None
        assert item.retiring is None
        np.testing.assert_array_equal(item.bank["w"][0], [40, 40])
    assert restored.payload() == cache.payload()


def test_usage_is_recent_assignments_across_modes_not_lifetime_or_game_length():
    cache = make_cache()
    snapshot(cache, 1)
    snapshot(cache, 10)
    trace = SimpleNamespace(
        active_mask=np.ones((2, 3)),
        episode_start=np.array([[1, 1, 1], [0, 1, 0]]),
        opponent_assignment=np.array([[0, 1, -1], [0, 1, 1]]),
    )
    cache.observe_rollouts({"offense": trace}, update=10)
    np.testing.assert_array_equal(cache.usage, [1, 2, 0, 0])
    metrics = cache.usage_metrics()
    assert metrics["opponent_cache/usage_percent/self_u1"] == pytest.approx(100 / 3)
    assert metrics["opponent_cache/assigned_games_latest"] == 3
    cache.observe_rollouts({}, update=510)
    np.testing.assert_array_equal(cache.usage, [0.5, 1, 0, 0])
    assert cache.usage_metrics()["opponent_cache/assigned_games_latest"] == 0


def test_grace_and_latest_protection_then_low_usage_eviction():
    cache = make_cache()
    for update in (1, 10, 20, 30):
        snapshot(cache, update)
    cache.usage[:] = [100, 2, 1, 0]
    # U30 is newest; U20 still in its grace period at U35. U10 beats U1
    # on recent usage despite being younger and despite U30 having zero use.
    assert cache._victim(35) == 1
    assert cache._victim(40) == 2
    large = make_cache(capacity=12)
    for update in range(10, 130, 10):
        snapshot(large, update)
    large.usage[:] = 100
    large.usage[0] = 0  # Historical anchor remains protected.
    large.usage[2] = 1
    assert large._victim(200) == 2


def test_retiring_opponent_is_excluded_before_pfsp_normalization():
    cache = make_cache(capacity=2)
    snapshot(cache, 1)
    snapshot(cache, 10)
    snapshot(cache, 30, {"offense": [0]})
    probs, metrics = PFSPState().sampling_snapshot(
        cache.candidates, update=31, alpha=0.5, epsilon=0.1
    )
    np.testing.assert_array_equal(probs, [[0, 1], [0, 1]])
    assert metrics["pfsp/retiring/self_u1/sampled"] == 1
    assert metrics["pfsp/snapshot_update/self_u10/sampled"] == 31


def test_zero_probability_slots_never_sampled_and_empty_bank_is_legal_random():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from basketworld_jax.train.runtime import sample_pinned_opponent_assignment

    runner = jax.jit(
        lambda p, key: sample_pinned_opponent_assignment(
            p, jnp.zeros(1000, dtype=bool), key, jax, jnp
        )
    )
    np.testing.assert_array_equal(
        runner(jnp.zeros((2, 4)), jax.random.PRNGKey(1)), -np.ones(1000)
    )
    np.testing.assert_array_equal(
        runner(jnp.array([[0, 1, 0, 0]] * 2, dtype=float), jax.random.PRNGKey(2)),
        np.ones(1000),
    )
    assert runner._cache_size() == 1


def test_cache_config_cli_roundtrip_and_legacy_default():
    from basketworld_jax.train.main import (
        parse_args,
        validate_train_args,
        build_train_config_schema,
    )
    from app.training_backend.config_schema import compile_overrides

    assert parse_args([]).opponent_snapshot_every_updates == 0
    args = parse_args(
        [
            "--enable-multi-possession",
            "--opponent-snapshot-every-updates",
            "50",
            "--opponent-pool-size",
            "600",
        ]
    )
    validate_train_args(args)
    overrides = {"opponent_snapshot_every_updates": 50, "opponent_pool_size": 600}
    restored = parse_args(compile_overrides(overrides, build_train_config_schema(args)))
    assert restored.opponent_snapshot_every_updates == 50
    assert restored.opponent_pool_size == 600
    for interval, size in ((-1, 30), (50, 1), (50, 0)):
        with pytest.raises(ValueError):
            validate_cache_config(interval, size)
    with pytest.raises(ValueError):
        validate_cache_config(50, 30, disabled=True)


def test_cache_resume_configuration_mismatch_is_rejected():
    cache = make_cache()
    with pytest.raises(ValueError, match="differs"):
        make_cache(interval=50, state=cache.payload())
    with pytest.raises(ValueError, match="exceeds"):
        make_cache(capacity=2, candidates=[{}, {}, {}])


def test_recency_uses_snapshot_age_not_reused_slot_order():
    from basketworld_jax.train.main import _opponent_candidate_probabilities

    candidates = [{"info": {"update_index": update}} for update in (100, 50, 150)]
    args = SimpleNamespace(
        opponent_snapshot_every_updates=50,
        opponent_pool_size=3,
        opponent_pool_beta=1.0,
        opponent_pool_exploration=0.0,
    )
    np.testing.assert_array_equal(
        _opponent_candidate_probabilities(candidates, args=args), [0, 0, 1]
    )
    candidates[0]["info"]["update_index"] = 200
    np.testing.assert_array_equal(
        _opponent_candidate_probabilities(candidates, args=args), [1, 0, 0]
    )


def test_checkpoint_host_weights_remain_unchanged_after_donated_slot_replacement():
    from basketworld_jax.train.main import _build_opponent_pool_checkpoint_state
    from basketworld_jax.checkpoints import build_checkpoint_payload

    cache = make_cache(capacity=2)
    snapshot(cache, 1)
    snapshot(cache, 10)
    state = _build_opponent_pool_checkpoint_state(
        cache.candidates,
        {},
        {},
        opponent_rng=np.random.default_rng(0),
        enabled=True,
        jax=cache.jax,
        jnp=cache.jnp,
        snapshot_cache=cache,
    )
    payload = build_checkpoint_payload(
        update_index=10,
        trainer_config={},
        policy_spec={},
        frozen_config={},
        params={},
        opt_state={},
        current_state={},
        eval_initial_state={},
        base_key=None,
        eval_trajectories=[],
        last_metrics=None,
        opponent_pool_state=state,
    )
    snapshot(cache, 30)
    cache.jax.block_until_ready(cache.bank)
    np.testing.assert_array_equal(cache.bank["w"][0], [30, 30])
    np.testing.assert_array_equal(
        payload["state"]["opponent_candidate_params"]["w"], [[1, 1], [10, 10]]
    )
