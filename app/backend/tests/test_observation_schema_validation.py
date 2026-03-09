import unittest

import gymnasium as gym
import numpy as np

from app.backend.observations import validate_policy_observation_schema


class _DummyPolicyObj:
    def __init__(self, observation_space):
        self.observation_space = observation_space


class _DummyPPO:
    def __init__(self, observation_space):
        self.policy = _DummyPolicyObj(observation_space)


class _DummyEnv:
    pass


class TestObservationSchemaValidation(unittest.TestCase):
    def test_dict_schema_validation_passes(self):
        obs_space = gym.spaces.Dict(
            {
                "obs": gym.spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32),
                "action_mask": gym.spaces.Box(low=0, high=1, shape=(2, 3), dtype=np.int8),
                "role_flag": gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
                "skills": gym.spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32),
            }
        )
        policy = _DummyPPO(obs_space)
        obs = {
            "obs": np.zeros((5,), dtype=np.float32),
            "action_mask": np.ones((2, 3), dtype=np.int8),
            "role_flag": np.array([1.0], dtype=np.float32),
            "skills": np.zeros((6,), dtype=np.float32),
        }
        validated = validate_policy_observation_schema(policy, _DummyEnv(), obs)
        self.assertIsInstance(validated, dict)
        self.assertEqual(set(validated.keys()), set(obs.keys()))

    def test_dict_schema_missing_key_raises(self):
        obs_space = gym.spaces.Dict(
            {
                "obs": gym.spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32),
                "action_mask": gym.spaces.Box(low=0, high=1, shape=(2, 3), dtype=np.int8),
            }
        )
        policy = _DummyPPO(obs_space)
        obs = {"obs": np.zeros((5,), dtype=np.float32)}
        with self.assertRaises(ValueError):
            validate_policy_observation_schema(policy, _DummyEnv(), obs)

    def test_dict_schema_shape_mismatch_raises(self):
        obs_space = gym.spaces.Dict(
            {"obs": gym.spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)}
        )
        policy = _DummyPPO(obs_space)
        obs = {"obs": np.zeros((4,), dtype=np.float32)}
        with self.assertRaises(ValueError):
            validate_policy_observation_schema(policy, _DummyEnv(), obs)

