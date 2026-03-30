from __future__ import annotations

import gymnasium as gym

from app.backend.env_access import get_env_attr


class _BaseEnv(gym.Env):
    metadata = {}

    def __init__(self) -> None:
        super().__init__()
        self.pass_reward = 1.25
        self.assist_window = 4

    def reset(self, *, seed=None, options=None):
        return {}, {}

    def step(self, action):
        return {}, 0.0, False, False, {}


class _WrapperWithAttr(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.violation_reward = 2.0


def test_get_env_attr_reads_unwrapped_attr_without_wrapper_forwarding() -> None:
    env = gym.Wrapper(_BaseEnv())
    assert float(get_env_attr(env, "pass_reward", 0.0)) == 1.25
    assert int(get_env_attr(env, "assist_window", 0)) == 4


def test_get_env_attr_reads_wrapper_attr_via_get_wrapper_attr() -> None:
    env = _WrapperWithAttr(_BaseEnv())
    assert float(get_env_attr(env, "violation_reward", 0.0)) == 2.0


def test_get_env_attr_returns_default_when_missing() -> None:
    env = gym.Wrapper(_BaseEnv())
    assert float(get_env_attr(env, "full_assist_bonus_pct", 0.5)) == 0.5
