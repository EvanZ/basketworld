from pathlib import Path
from types import SimpleNamespace

from basketworld.utils.mlflow_params import (
    get_mlflow_params,
    get_mlflow_phi_shaping_params,
    get_mlflow_start_template_library,
    get_mlflow_training_params,
)


class _FakeClient:
    def __init__(self, params):
        self._run = SimpleNamespace(data=SimpleNamespace(params=params))

    def get_run(self, run_id):
        return self._run


def test_get_mlflow_params_includes_opponent_sampling_settings():
    client = _FakeClient(
        {
            "grid_size": "16",
            "players": "3",
            "shot_clock": "24",
            "deterministic_opponent": "true",
            "per_env_opponent_sampling": "true",
            "opponent_pool_size": "12",
            "opponent_pool_beta": "0.61",
            "opponent_pool_exploration": "0.07",
        }
    )

    required, optional = get_mlflow_params(client, "dummy")

    assert required == {
        "grid_size": 16,
        "players": 3,
        "shot_clock": 24,
    }
    assert optional["deterministic_opponent"] is True
    assert optional["per_env_opponent_sampling"] is True
    assert optional["opponent_pool_size"] == 12
    assert optional["opponent_pool_beta"] == 0.61
    assert optional["opponent_pool_exploration"] == 0.07


def test_get_mlflow_params_includes_set_obs_and_mirror_settings():
    client = _FakeClient(
        {
            "grid_size": "16",
            "players": "3",
            "shot_clock": "24",
            "use_set_obs": "true",
            "mirror_episode_prob": "0.35",
        }
    )

    _, optional = get_mlflow_params(client, "dummy")

    assert optional["use_set_obs"] is True
    assert optional["mirror_episode_prob"] == 0.35


def test_get_mlflow_training_params_includes_disc_eval_batch_output():
    client = _FakeClient(
        {
            "disc_eval_batch_output": "true",
            "intent_disc_eval_holdout_fraction": "0.3",
            "intent_selector_mode": "integrated",
            "intent_selector_value_coef": "0.75",
        }
    )

    training = get_mlflow_training_params(client, "dummy")

    assert training["disc_eval_batch_output"] is True
    assert training["intent_disc_eval_holdout_fraction"] == 0.3
    assert training["intent_selector_mode"] == "integrated"
    assert training["intent_selector_value_coef"] == 0.75


def test_get_mlflow_training_params_includes_multiselect_and_disc_priors():
    client = _FakeClient(
        {
            "intent_selector_multiselect_enabled": "true",
            "intent_selector_min_play_steps": "4",
            "intent_disc_lambda_shot": "0.2",
            "intent_disc_lambda_q": "0.05",
        }
    )

    training = get_mlflow_training_params(client, "dummy")

    assert training["intent_selector_multiselect_enabled"] is True
    assert training["intent_selector_min_play_steps"] == 4
    assert training["intent_disc_lambda_shot"] == 0.2
    assert training["intent_disc_lambda_q"] == 0.05


def test_get_mlflow_training_params_supports_jax_start_template_aliases():
    client = _FakeClient(
        {
            "jax/env/start_template_enabled": "true",
            "jax/env/start_template_library": "configs/start_templates_5v5.json",
            "jax/env/start_template_library_artifact_path": "metadata/start_template_library.json",
            "jax/env/start_template_library_template_count": "12",
            "jax/env/start_template_prob": "0.8",
            "jax/env/start_template_jitter_scale": "1.25",
            "jax/env/start_template_mirror_prob": "0.4",
            "jax/env/start_template_strict": "true",
        }
    )

    training = get_mlflow_training_params(client, "dummy")

    assert training["start_template_enabled"] is True
    assert training["start_template_library"] == "configs/start_templates_5v5.json"
    assert training["start_template_library_artifact_path"] == "metadata/start_template_library.json"
    assert training["start_template_library_template_count"] == 12
    assert training["start_template_prob"] == 0.8
    assert training["start_template_jitter_scale"] == 1.25
    assert training["start_template_mirror_prob"] == 0.4
    assert training["start_template_strict"] is True


def test_get_mlflow_start_template_library_uses_jax_env_players():
    library_path = Path(__file__).resolve().parents[1] / "configs" / "start_templates_5v5.json"

    class _FakeArtifactClient(_FakeClient):
        def download_artifacts(self, run_id, artifact_path, dst_path):
            return str(library_path)

    client = _FakeArtifactClient(
        {
            "start_template_library_artifact_path": "metadata/start_template_library.json",
            "jax/env/players": "5",
        }
    )

    library = get_mlflow_start_template_library(client, "dummy")

    assert library is not None
    assert library["players_per_side"] == 5
    assert len(library["templates"]) == 12


def test_get_mlflow_training_params_supports_jax_selector_aliases():
    client = _FakeClient(
        {
            "jax/intent_selector_enabled": "true",
            "jax/intent_selector_hidden_dim": "96",
            "jax/intent_selector_alpha_start": "0.1",
            "jax/intent_selector_alpha_end": "0.8",
            "jax/intent_selector_alpha_warmup_updates": "20",
            "jax/intent_selector_alpha_ramp_updates": "100",
            "jax/intent_selector_eps_start": "0.5",
            "jax/intent_selector_eps_end": "0.05",
            "jax/intent_selector_eps_warmup_updates": "10",
            "jax/intent_selector_eps_ramp_updates": "50",
            "jax/intent_selector_entropy_coef": "0.02",
            "jax/intent_selector_usage_reg_coef": "0.03",
            "jax/intent_selector_value_coef": "0.4",
            "jax/intent_selector_train_every_rollouts": "2",
            "jax/intent_selector_max_samples_per_update": "4096",
            "jax/intent_selector_multiselect_enabled": "true",
            "jax/intent_selector_min_play_steps": "7",
        }
    )

    training = get_mlflow_training_params(client, "dummy")

    assert training["intent_selector_enabled"] is True
    assert training["intent_selector_mode"] == "integrated"
    assert training["intent_selector_hidden_dim"] == 96
    assert training["intent_selector_alpha_start"] == 0.1
    assert training["intent_selector_alpha_end"] == 0.8
    assert training["intent_selector_alpha_warmup_steps"] == 20
    assert training["intent_selector_alpha_ramp_steps"] == 100
    assert training["intent_selector_eps_start"] == 0.5
    assert training["intent_selector_eps_end"] == 0.05
    assert training["intent_selector_eps_warmup_steps"] == 10
    assert training["intent_selector_eps_ramp_steps"] == 50
    assert training["intent_selector_entropy_coef"] == 0.02
    assert training["intent_selector_usage_reg_coef"] == 0.03
    assert training["intent_selector_value_coef"] == 0.4
    assert training["intent_selector_train_every_rollouts"] == 2
    assert training["intent_selector_max_samples_per_update"] == 4096
    assert training["intent_selector_multiselect_enabled"] is True
    assert training["intent_selector_min_play_steps"] == 7


def test_get_mlflow_phi_shaping_params_supports_jax_env_aliases():
    client = _FakeClient(
        {
            "jax/env/enable_phi_shaping": "true",
            "jax/enable_phi_shaping": "false",
            "jax/env/reward_shaping_gamma": "0.97",
            "jax/reward_shaping_gamma": "0.50",
            "jax/env/phi_beta_start": "0.01",
            "jax/env/phi_beta_end": "0.15",
            "jax/phi_beta_end": "0.05",
            "jax/env/phi_blend_weight": "0.5",
            "jax/phi_blend_weight": "0.1",
            "jax/env/phi_aggregation_mode": "teammates_best",
            "jax/env/phi_use_ball_handler_only": "false",
        }
    )

    phi = get_mlflow_phi_shaping_params(client, "dummy")

    assert phi["enable_phi_shaping"] is True
    assert phi["reward_shaping_gamma"] == 0.97
    assert phi["phi_beta"] == 0.15
    assert phi["phi_blend_weight"] == 0.5
    assert phi["phi_aggregation_mode"] == "teammates_best"
    assert phi["phi_use_ball_handler_only"] is False
