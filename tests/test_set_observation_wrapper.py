import numpy as np

import basketworld
from basketworld.utils.wrappers import SetObservationWrapper


def test_set_observation_wrapper_adds_tokens_and_globals():
    env = SetObservationWrapper(basketworld.HexagonBasketballEnv(players=3))
    obs, _ = env.reset(seed=0)

    assert isinstance(obs, dict)
    assert "players" in obs
    assert "globals" in obs

    players = obs["players"]
    globals_vec = obs["globals"]
    assert players.shape == (env.unwrapped.n_players, 15)
    assert globals_vec.shape == (4,)

    assert np.all(np.isfinite(players))
    assert np.all(np.isfinite(globals_vec))


def test_set_observation_wrapper_adds_intent_globals_when_enabled():
    env = SetObservationWrapper(
        basketworld.HexagonBasketballEnv(
            players=3,
            enable_intent_learning=True,
            intent_null_prob=0.0,
        )
    )
    obs, _ = env.reset(seed=3)
    globals_vec = obs["globals"]
    assert globals_vec.shape == (7,)
    assert np.all(np.isfinite(globals_vec))
