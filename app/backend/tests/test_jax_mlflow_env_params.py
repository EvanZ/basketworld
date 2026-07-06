from __future__ import annotations

from basketworld.envs.basketworld_env_v2 import Team

from app.backend.routes.lifecycle_routes import (
    _jax_local_env_config_from_metadata,
    _overlay_jax_mlflow_env_params,
    _overlay_jax_mlflow_training_params,
)


def test_overlay_jax_mlflow_env_params_applies_skill_stds():
    optional = {
        "training_team": Team.OFFENSE,
        "layup_pct": 0.60,
        "three_pt_pct": 0.40,
        "dunk_pct": 0.85,
    }
    params = {
        "jax/env/layup_pct": "0.55",
        "jax/env/three_pt_pct": "0.37",
        "jax/env/dunk_pct": "0.6",
        "jax/env/layup_std": "0.05",
        "jax/env/three_pt_std": "0.05",
        "jax/env/dunk_std": "0.3",
        "jax/env/allow_dunks": "true",
        "jax/env/training_team": "offense",
        "jax/env/enable_intent_learning": "true",
        "jax/env/enable_defense_intent_learning": "true",
        "jax/env/num_intents": "6",
        "jax/env/intent_commitment_steps": "5",
        "jax/env/intent_null_prob": "0.1",
        "jax/env/defense_intent_null_prob": "0.2",
        "jax/env/intent_visible_to_defense_prob": "0.3",
        "jax/env/three_second_lane_width": "1",
        "jax/env/three_second_lane_height": "3",
        "jax/env/three_second_max_steps": "3",
        "jax/env/violation_reward": "2.0",
        "jax/env/enable_phi_shaping": "true",
        "jax/env/reward_shaping_gamma": "0.97",
        "jax/env/phi_beta_start": "0.01",
        "jax/env/phi_beta_end": "0.15",
        "jax/env/phi_beta_warmup_updates": "50",
        "jax/env/phi_beta_ramp_updates": "200",
        "jax/env/phi_blend_weight": "0.5",
        "jax/env/phi_aggregation_mode": "teammates_best",
        "jax/env/phi_use_ball_handler_only": "false",
        "jax/env/start_template_library": "/tmp/templates.json",
        "jax/env/start_template_prob": "1.0",
        "jax/env/rebound_skill_std": "0.75",
        "jax/env/rebound_skill_sampling_mode": "one_high_per_team",
        "jax/env/rebound_skill_high": "1.0",
        "jax/env/rebound_skill_low": "-0.25",
        "jax/env/rebound_skill_weight": "1.25",
        "jax/env/rebound_contest_mode": "local_contest",
        "jax/env/rebound_contest_radius": "1",
        "jax/env/rebound_obs_top_n_targets": "3",
    }

    merged = _overlay_jax_mlflow_env_params(optional, params)

    assert merged["training_team"] == Team.OFFENSE
    assert merged["allow_dunks"] is True
    assert merged["layup_pct"] == 0.55
    assert merged["three_pt_pct"] == 0.37
    assert merged["dunk_pct"] == 0.6
    assert merged["layup_std"] == 0.05
    assert merged["three_pt_std"] == 0.05
    assert merged["dunk_std"] == 0.3
    assert merged["enable_intent_learning"] is True
    assert merged["enable_defense_intent_learning"] is True
    assert merged["num_intents"] == 6
    assert merged["intent_commitment_steps"] == 5
    assert merged["intent_null_prob"] == 0.1
    assert merged["defense_intent_null_prob"] == 0.2
    assert merged["intent_visible_to_defense_prob"] == 0.3
    assert merged["three_second_lane_width"] == 1
    assert merged["three_second_lane_height"] == 3
    assert merged["three_second_max_steps"] == 3
    assert merged["violation_reward"] == 2.0
    assert merged["enable_phi_shaping"] is True
    assert merged["reward_shaping_gamma"] == 0.97
    assert merged["phi_beta_start"] == 0.01
    assert merged["phi_beta_end"] == 0.15
    assert merged["phi_beta"] == 0.15
    assert merged["phi_beta_warmup_updates"] == 50
    assert merged["phi_beta_ramp_updates"] == 200
    assert merged["phi_blend_weight"] == 0.5
    assert merged["phi_aggregation_mode"] == "teammates_best"
    assert merged["phi_use_ball_handler_only"] is False
    assert merged["start_template_library"] == "/tmp/templates.json"
    assert merged["start_template_prob"] == 1.0
    assert merged["rebound_skill_std"] == 0.75
    assert merged["rebound_skill_sampling_mode"] == "one_high_per_team"
    assert merged["rebound_skill_high"] == 1.0
    assert merged["rebound_skill_low"] == -0.25
    assert merged["rebound_skill_weight"] == 1.25
    assert merged["rebound_contest_mode"] == "local_contest"
    assert merged["rebound_contest_radius"] == 1
    assert merged["rebound_obs_top_n_targets"] == 3


def test_overlay_jax_mlflow_env_params_accepts_rebound_skill_aliases():
    merged = _overlay_jax_mlflow_env_params(
        {},
        {
            "jax/rebound_skill_std": "0.5",
            "jax/rebound_skill_sampling_mode": "one_high_per_team",
            "jax/rebound_skill_high": "2.0",
            "jax/rebound_skill_low": "-0.5",
            "jax/rebound_skill_weight": "1.5",
            "jax/rebound_terminal_reward_mode": "last_shot_ep",
            "jax/rebound_contest_mode": "local_contest",
            "jax/rebound_contest_initial_radius": "2",
            "jax/rebound_obs_top_n_targets": "2",
        },
    )

    assert merged["rebound_skill_std"] == 0.5
    assert merged["rebound_skill_sampling_mode"] == "one_high_per_team"
    assert merged["rebound_skill_high"] == 2.0
    assert merged["rebound_skill_low"] == -0.5
    assert merged["rebound_skill_weight"] == 1.5
    assert merged["rebound_terminal_reward_mode"] == "last_shot_ep"
    assert merged["rebound_contest_mode"] == "local_contest"
    assert merged["rebound_contest_radius"] == 2
    assert merged["rebound_obs_top_n_targets"] == 2


def test_overlay_jax_mlflow_training_params_exposes_selector_config():
    trainer = {
        "intent_selector_enabled": True,
        "intent_selector_multiselect_enabled": False,
        "intent_selector_min_play_steps": 3,
    }
    params = {
        "jax/mode": "train_loop",
        "jax/kernel_batch_size": "4096",
        "jax/rollout_horizon": "128",
        "jax/policy_update_epochs": "3",
        "jax/ppo_minibatches": "16",
        "jax/ppo_clip_range": "0.2",
        "jax/value_coef": "0.5",
        "jax/entropy_coef": "0.01",
        "jax/policy_model": "attention",
        "jax/attention_embed_dim": "64",
        "jax/attention_num_heads": "4",
        "jax/attention_token_mlp_dim": "64",
        "jax/intent_selector_enabled": "true",
        "jax/intent_selector_hidden_dim": "96",
        "jax/intent_selector_alpha_start": "0.1",
        "jax/intent_selector_alpha_end": "0.8",
        "jax/intent_selector_alpha_warmup_updates": "20",
        "jax/intent_selector_alpha_ramp_updates": "100",
        "jax/intent_selector_eps_start": "0.5",
        "jax/intent_selector_eps_end": "0.05",
        "jax/intent_selector_entropy_coef": "0.02",
        "jax/intent_selector_train_every_rollouts": "2",
        "jax/intent_selector_max_samples_per_update": "4096",
        "jax/intent_selector_multiselect_enabled": "true",
        "jax/intent_selector_min_play_steps": "7",
        "jax/env/layup_std": "0.05",
        "jax/env/enable_phi_shaping": "true",
        "jax/env/phi_beta_end": "0.15",
        "jax/env/phi_beta_warmup_updates": "50",
    }

    merged = _overlay_jax_mlflow_training_params(trainer, params)

    assert merged["intent_selector_enabled"] is True
    assert merged["intent_selector_mode"] == "integrated"
    assert merged["intent_selector_hidden_dim"] == 96
    assert merged["intent_selector_alpha_start"] == 0.1
    assert merged["intent_selector_alpha_end"] == 0.8
    assert merged["intent_selector_alpha_warmup_updates"] == 20
    assert merged["intent_selector_alpha_ramp_updates"] == 100
    assert merged["intent_selector_eps_start"] == 0.5
    assert merged["intent_selector_eps_end"] == 0.05
    assert merged["intent_selector_entropy_coef"] == 0.02
    assert merged["intent_selector_train_every_rollouts"] == 2
    assert merged["intent_selector_max_samples_per_update"] == 4096
    assert merged["intent_selector_multiselect_enabled"] is True
    assert merged["intent_selector_min_play_steps"] == 7
    assert merged["enable_phi_shaping"] is True
    assert merged["phi_beta_end"] == 0.15
    assert merged["phi_beta_warmup_updates"] == 50
    assert merged["num_envs"] == 4096
    assert merged["n_steps"] == 128
    assert merged["steps_per_update"] == 1048576
    assert merged["ppo_batch_size"] == 1048576
    assert merged["batch_size"] == 65536
    assert merged["n_epochs"] == 3
    assert merged["clip_range"] == 0.2
    assert merged["vf_coef"] == 0.5
    assert merged["ent_coef"] == 0.01
    assert merged["policy_class"] == "JAX attention actor-critic"
    assert merged["intent_selector_alpha_warmup_steps"] == 20971520
    assert merged["intent_selector_alpha_ramp_steps"] == 104857600
    assert "env/layup_std" not in merged


def test_jax_local_env_config_prefers_checkpoint_env_config():
    class Policy:
        metadata = {
            "frozen_config": {
                "training_team": "offense",
                "layup_std": 0.0,
            },
            "env_config": {
                "training_team": "offense",
                "layup_std": 0.05,
                "three_pt_std": 0.05,
                "dunk_std": 0.3,
                "offensive_three_seconds": False,
                "enable_intent_learning": True,
                "enable_defense_intent_learning": True,
                "num_intents": 6,
                "intent_commitment_steps": 5,
                "intent_null_prob": 0.1,
                "defense_intent_null_prob": 0.2,
                "intent_visible_to_defense_prob": 0.3,
                "enable_phi_shaping": True,
                "reward_shaping_gamma": 0.97,
                "phi_beta_start": 0.01,
                "phi_beta_end": 0.15,
                "phi_blend_weight": 0.5,
                "phi_aggregation_mode": "teammates_best",
                "phi_use_ball_handler_only": False,
            },
            "trainer_config": {"kernel_batch_size": 128},
            "last_metrics": {"phi_beta": 0.123},
        }

    required, optional, trainer, phi = _jax_local_env_config_from_metadata(Policy())

    assert required == {}
    assert optional["training_team"] == Team.OFFENSE
    assert optional["allow_dunks"] is True
    assert optional["layup_std"] == 0.05
    assert optional["three_pt_std"] == 0.05
    assert optional["dunk_std"] == 0.3
    assert optional["offensive_three_seconds_enabled"] is False
    assert optional["enable_intent_learning"] is True
    assert optional["enable_defense_intent_learning"] is True
    assert optional["num_intents"] == 6
    assert optional["intent_commitment_steps"] == 5
    assert optional["intent_null_prob"] == 0.1
    assert optional["defense_intent_null_prob"] == 0.2
    assert optional["intent_visible_to_defense_prob"] == 0.3
    assert optional["enable_phi_shaping"] is True
    assert optional["reward_shaping_gamma"] == 0.97
    assert optional["phi_beta"] == 0.123
    assert optional["phi_beta_end"] == 0.15
    assert optional["phi_blend_weight"] == 0.5
    assert "offensive_three_seconds" not in optional
    assert trainer == {"kernel_batch_size": 128}
    assert phi == {
        "enable_phi_shaping": True,
        "phi_beta": 0.123,
        "reward_shaping_gamma": 0.97,
        "phi_aggregation_mode": "teammates_best",
        "phi_use_ball_handler_only": False,
        "phi_blend_weight": 0.5,
    }
