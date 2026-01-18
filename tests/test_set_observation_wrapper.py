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
    assert players.shape == (env.unwrapped.n_players, 8)
    assert globals_vec.shape == (3,)

    assert np.all(np.isfinite(players))
    assert np.all(np.isfinite(globals_vec))
