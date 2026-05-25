import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from basketworld.envs.basketworld_env_v2 import HexagonBasketballEnv
from basketworld_jax.env.minimal import (
    build_kernel_static_from_env,
    build_pass_steal_probabilities_batch,
    snapshot_state_from_env,
    stack_state_snapshots,
)


def test_pass_steal_ignores_defender_beyond_receiver_projection():
    env = HexagonBasketballEnv(
        players=3,
        render_mode=None,
        pass_mode="pointer_targeted",
        allow_dunks=True,
        base_steal_rate=0.3,
    )
    env.reset(seed=17)
    static = build_kernel_static_from_env(env, xp=jnp)

    passer = np.asarray([-7, 14], dtype=np.int32)
    receiver = np.asarray([-6, 12], dtype=np.int32)
    defender_beyond_receiver = np.asarray([-5, 10], dtype=np.int32)
    other_receiver = np.asarray([-7, 15], dtype=np.int32)

    snapshot = snapshot_state_from_env(env)
    snapshot["positions"] = np.asarray(
        [
            passer,
            receiver,
            other_receiver,
            defender_beyond_receiver,
            passer,
            receiver,
        ],
        dtype=np.int32,
    )
    snapshot["ball_holder"] = int(env.offense_ids[0])
    state = stack_state_snapshots([snapshot], xp=jnp)

    steal_probs = np.asarray(build_pass_steal_probabilities_batch(static, state, jnp))[0]

    assert steal_probs[1] == pytest.approx(0.0, abs=1e-7)
