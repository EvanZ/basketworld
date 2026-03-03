import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from basketworld.utils.wrappers import EpisodeStatsWrapper


class _DummyEpisodeEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, pass_mode: str, action_results: dict):
        super().__init__()
        self.pass_mode = str(pass_mode)
        self.players_per_side = 3
        self.n_players = 6
        self.positions = [(0, 0)] * self.n_players
        self.basket_position = (0, 0)
        self.offense_ids = [0, 1, 2]
        self.defense_ids = [3, 4, 5]
        self.allow_dunks = False
        self.pressure_exposure = 0.0
        self.action_space = spaces.Discrete(1)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32
        )
        self._action_results = action_results

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros((1,), dtype=np.float32), {}

    def step(self, action):
        info = {"action_results": self._action_results}
        obs = np.zeros((1,), dtype=np.float32)
        reward = 0.0
        terminated = True
        truncated = False
        return obs, reward, terminated, truncated, info

    def _hex_distance(self, a, b):
        return int(abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1])))

    def is_three_point_location(self, coord):
        return False


def test_pointer_pass_episode_metrics_are_computed():
    action_results = {
        "passes": {
            0: {"success": True, "target": 1, "intended_target": 1},
            1: {"success": False, "reason": "illegal_target", "intended_target": 2},
        },
        "turnovers": [
            {"reason": "steal", "intended_target": 2},
        ],
    }
    wrapped = EpisodeStatsWrapper(
        _DummyEpisodeEnv(pass_mode="pointer_targeted", action_results=action_results)
    )
    wrapped.reset()
    _, _, done, _, info = wrapped.step(0)
    assert done is True

    # attempts = 2 pass records + 1 pass-steal turnover
    assert info["pointer_pass_attempts"] == 3.0
    # one exact intent/outcome match (the successful pass to intended target)
    assert math.isclose(info["pointer_pass_intent_match_rate"], 1.0 / 3.0, rel_tol=1e-6)

    # intended target counts: {1:1, 2:2}
    p1 = 1.0 / 3.0
    p2 = 2.0 / 3.0
    entropy = -(p1 * math.log(p1) + p2 * math.log(p2))
    max_entropy = math.log(2.0)  # players_per_side - 1
    kl_uniform = max_entropy - entropy
    assert math.isclose(info["pointer_pass_target_entropy"], entropy, rel_tol=1e-6)
    assert math.isclose(
        info["pointer_pass_target_entropy_norm"], entropy / max_entropy, rel_tol=1e-6
    )
    assert math.isclose(info["pointer_pass_target_kl_uniform"], kl_uniform, rel_tol=1e-6)


def test_directional_mode_pointer_metrics_stay_zero():
    action_results = {
        "passes": {
            0: {"success": True, "target": 1, "intended_target": None},
        },
        "turnovers": [],
    }
    wrapped = EpisodeStatsWrapper(
        _DummyEpisodeEnv(pass_mode="directional", action_results=action_results)
    )
    wrapped.reset()
    _, _, done, _, info = wrapped.step(0)
    assert done is True
    assert info["pointer_pass_attempts"] == 0.0
    assert info["pointer_pass_intent_match_rate"] == 0.0
    assert info["pointer_pass_target_entropy"] == 0.0
    assert info["pointer_pass_target_entropy_norm"] == 0.0
    assert info["pointer_pass_target_kl_uniform"] == 0.0


def test_potential_assist_counts_on_assisted_make_and_expected_points_is_exposed():
    action_results = {
        "passes": {},
        "turnovers": [],
        "shots": {
            0: {
                "success": True,
                "assist_potential": True,
                "assist_full": True,
                "expected_points": 1.25,
                "is_three": False,
            }
        },
    }
    env = _DummyEpisodeEnv(pass_mode="directional", action_results=action_results)
    env.positions[0] = (1, 0)
    wrapped = EpisodeStatsWrapper(env)
    wrapped.reset()
    _, _, done, _, info = wrapped.step(0)
    assert done is True
    assert info["assisted_2pt"] == 1.0
    assert info["potential_assisted_2pt"] == 1.0
    assert info["potential_assists"] == 1.0
    assert math.isclose(info["expected_points"], 1.25, rel_tol=1e-6)


def test_steal_reason_is_counted_as_intercepted_turnover_metric():
    action_results = {
        "passes": {},
        "turnovers": [
            {"reason": "steal", "player_id": 0},
        ],
    }
    wrapped = EpisodeStatsWrapper(
        _DummyEpisodeEnv(pass_mode="pointer_targeted", action_results=action_results)
    )
    wrapped.reset()
    _, _, done, _, info = wrapped.step(0)
    assert done is True
    assert info["turnover"] == 1.0
    assert info["turnover_intercepted"] == 1.0
    assert info["turnover_other"] == 0.0
