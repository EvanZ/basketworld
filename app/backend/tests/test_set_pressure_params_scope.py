import math
from dataclasses import dataclass
import unittest
from unittest.mock import patch

from fastapi import HTTPException

from app.backend.routes import admin_routes
from app.backend.schemas import SetPressureParamsRequest
from app.backend.state import game_state


ALL_PARAM_KEYS = [
    "three_pt_extra_hex_decay",
    "shot_pressure_enabled",
    "shot_pressure_max",
    "shot_pressure_lambda",
    "shot_pressure_arc_degrees",
    "base_steal_rate",
    "steal_perp_decay",
    "steal_distance_factor",
    "steal_position_weight_min",
    "defender_pressure_distance",
    "defender_pressure_turnover_chance",
    "defender_pressure_decay_lambda",
]

DEFENDER_KEYS = [
    "defender_pressure_distance",
    "defender_pressure_turnover_chance",
    "defender_pressure_decay_lambda",
]

PASS_INTERCEPTION_KEYS = [
    "base_steal_rate",
    "steal_perp_decay",
    "steal_distance_factor",
    "steal_position_weight_min",
]


@dataclass
class DummyPressureEnv:
    three_pt_extra_hex_decay: float = 0.99
    shot_pressure_enabled: bool = False
    shot_pressure_max: float = 0.91
    shot_pressure_lambda: float = 9.0
    shot_pressure_arc_degrees: float = 111.0
    base_steal_rate: float = 0.81
    steal_perp_decay: float = 8.0
    steal_distance_factor: float = 0.71
    steal_position_weight_min: float = 0.61
    defender_pressure_distance: int = 9
    defender_pressure_turnover_chance: float = 0.51
    defender_pressure_decay_lambda: float = 7.0

    def __post_init__(self):
        self.shot_pressure_arc_rad = math.radians(self.shot_pressure_arc_degrees)

    def _get_observation(self):
        return [0.0]

    def _get_action_masks(self):
        return [[1]]


def _snapshot_env_params(env):
    return {key: getattr(env, key) for key in ALL_PARAM_KEYS}


class TestSetPressureParamsScope(unittest.TestCase):
    def setUp(self):
        self.prev_env = game_state.env
        self.prev_obs = game_state.obs
        self.prev_episode_states = game_state.episode_states
        self.prev_env_optional_params = game_state.env_optional_params
        self.prev_mlflow_defaults = game_state.mlflow_env_optional_defaults

        game_state.env = DummyPressureEnv()
        game_state.obs = {
            "obs": [0.0],
            "action_mask": [[1]],
            "role_flag": None,
            "skills": None,
        }
        game_state.episode_states = []
        game_state.env_optional_params = {}
        game_state.mlflow_env_optional_defaults = {
            "three_pt_extra_hex_decay": 0.05,
            "shot_pressure_enabled": True,
            "shot_pressure_max": 0.25,
            "shot_pressure_lambda": 1.0,
            "shot_pressure_arc_degrees": 300.0,
            "base_steal_rate": 0.3,
            "steal_perp_decay": 1.5,
            "steal_distance_factor": 0.2,
            "steal_position_weight_min": 0.3,
            "defender_pressure_distance": 3,
            "defender_pressure_turnover_chance": 0.05,
            "defender_pressure_decay_lambda": 1.0,
        }
        self._state_patch = patch.object(
            admin_routes, "get_full_game_state", side_effect=lambda **_: {"mock": True}
        )
        self._state_patch.start()

    def tearDown(self):
        self._state_patch.stop()
        game_state.env = self.prev_env
        game_state.obs = self.prev_obs
        game_state.episode_states = self.prev_episode_states
        game_state.env_optional_params = self.prev_env_optional_params
        game_state.mlflow_env_optional_defaults = self.prev_mlflow_defaults

    def test_scoped_reset_defender_updates_only_defender_fields(self):
        env = game_state.env
        before = _snapshot_env_params(env)

        response = admin_routes.set_pressure_params(
            SetPressureParamsRequest(
                reset_to_mlflow_defaults=True, scope="defender_pressure"
            )
        )

        self.assertEqual(response["status"], "success")
        self.assertEqual(response["applied_scope"], "defender_pressure")
        self.assertEqual(response["updated_keys"], sorted(DEFENDER_KEYS))

        for key in DEFENDER_KEYS:
            self.assertEqual(getattr(env, key), game_state.mlflow_env_optional_defaults[key])
        for key in ALL_PARAM_KEYS:
            if key not in DEFENDER_KEYS:
                self.assertEqual(getattr(env, key), before[key])

    def test_scoped_reset_pass_interception_updates_only_pass_fields(self):
        env = game_state.env
        before = _snapshot_env_params(env)

        response = admin_routes.set_pressure_params(
            SetPressureParamsRequest(
                reset_to_mlflow_defaults=True, scope="pass_interception"
            )
        )

        self.assertEqual(response["status"], "success")
        self.assertEqual(response["applied_scope"], "pass_interception")
        self.assertEqual(response["updated_keys"], sorted(PASS_INTERCEPTION_KEYS))

        for key in PASS_INTERCEPTION_KEYS:
            self.assertEqual(getattr(env, key), game_state.mlflow_env_optional_defaults[key])
        for key in ALL_PARAM_KEYS:
            if key not in PASS_INTERCEPTION_KEYS:
                self.assertEqual(getattr(env, key), before[key])

    def test_scoped_apply_filters_out_non_scope_fields(self):
        env = game_state.env
        before = _snapshot_env_params(env)

        response = admin_routes.set_pressure_params(
            SetPressureParamsRequest(
                scope="pass_interception",
                base_steal_rate=0.22,
                steal_perp_decay=2.2,
                defender_pressure_distance=1,  # should be ignored by pass_interception scope
                shot_pressure_max=0.4,  # should be ignored by pass_interception scope
            )
        )

        self.assertEqual(response["status"], "success")
        self.assertEqual(response["applied_scope"], "pass_interception")
        self.assertEqual(
            response["updated_keys"], sorted(["base_steal_rate", "steal_perp_decay"])
        )

        self.assertAlmostEqual(env.base_steal_rate, 0.22)
        self.assertAlmostEqual(env.steal_perp_decay, 2.2)
        self.assertEqual(env.defender_pressure_distance, before["defender_pressure_distance"])
        self.assertEqual(env.shot_pressure_max, before["shot_pressure_max"])

    def test_reset_to_defaults_requires_scope_or_reset_keys(self):
        with self.assertRaises(HTTPException) as exc_info:
            admin_routes.set_pressure_params(
                SetPressureParamsRequest(reset_to_mlflow_defaults=True)
            )
        self.assertEqual(exc_info.exception.status_code, 400)
        self.assertIn(
            "requires scope/reset_group or reset_keys",
            str(exc_info.exception.detail),
        )

    def test_defender_specific_route_forces_defender_scope(self):
        env = game_state.env
        before = _snapshot_env_params(env)

        response = admin_routes.set_defender_pressure_params(
            SetPressureParamsRequest(reset_to_mlflow_defaults=True)
        )

        self.assertEqual(response["status"], "success")
        self.assertEqual(response["applied_scope"], "defender_pressure")
        self.assertEqual(response["updated_keys"], sorted(DEFENDER_KEYS))
        for key in DEFENDER_KEYS:
            self.assertEqual(getattr(env, key), game_state.mlflow_env_optional_defaults[key])
        for key in ALL_PARAM_KEYS:
            if key not in DEFENDER_KEYS:
                self.assertEqual(getattr(env, key), before[key])

    def test_pass_specific_route_forces_pass_scope(self):
        env = game_state.env
        before = _snapshot_env_params(env)

        response = admin_routes.set_pass_interception_params(
            SetPressureParamsRequest(
                base_steal_rate=0.11,
                shot_pressure_max=0.02,  # should be ignored by pass route
            )
        )

        self.assertEqual(response["status"], "success")
        self.assertEqual(response["applied_scope"], "pass_interception")
        self.assertEqual(response["updated_keys"], ["base_steal_rate"])
        self.assertAlmostEqual(env.base_steal_rate, 0.11)
        self.assertEqual(env.shot_pressure_max, before["shot_pressure_max"])

    def test_shot_specific_route_forces_shot_scope(self):
        env = game_state.env
        before = _snapshot_env_params(env)

        response = admin_routes.set_shot_pressure_params(
            SetPressureParamsRequest(
                shot_pressure_max=0.33,
                base_steal_rate=0.77,  # should be ignored by shot route
            )
        )

        self.assertEqual(response["status"], "success")
        self.assertEqual(response["applied_scope"], "shot_pressure")
        self.assertEqual(response["updated_keys"], ["shot_pressure_max"])
        self.assertAlmostEqual(env.shot_pressure_max, 0.33)
        self.assertEqual(env.base_steal_rate, before["base_steal_rate"])
