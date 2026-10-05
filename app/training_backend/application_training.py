from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from app.training_backend.config_schema import compile_config_values


# Arguments inherited from the historical SB3/shared parser that are not read
# by either the JAX trainer or the environment bootstrap used by that trainer.
# Do not expose them as editable no-ops. Representative JAX replacements are:
# n_steps -> rollout_horizon, n_epochs -> policy_update_epochs,
# batch_size -> ppo_minibatches, num_envs -> kernel_batch_size,
# net_arch* -> policy_hidden_dims/attention_*, and eval_freq/eval_episodes ->
# eval_*_every_updates plus the corresponding JAX episode/batch controls.
JAX_INAPPLICABLE_FIELDS = frozenset(
    {
        "alternations",
        "batch_size",
        "continue_schedule_mode",
        "enable_env_profiling",
        "ent_bump_multiplier",
        "ent_bump_rollouts",
        "ent_bump_updates",
        "episode_sample_prob",
        "eval_episodes",
        "eval_freq",
        "init_critic_from_run",
        "intent_disc_console_log_every_rollouts",
        "intent_disc_lambda_q",
        "intent_disc_lambda_shot",
        "intent_disc_max_action_dim",
        "intent_disc_step_dim",
        "intent_null_prob_end",
        "intent_selector_template_metrics_log_every_rollouts",
        "intent_visible_to_defense_prob_end",
        "log_episode_artifacts",
        "mlflow_episode_log_every_rollouts",
        "mlflow_gradnorm_log_every_rollouts",
        "mlflow_sb3_log_every_writes",
        "mlflow_schedule_log_every_rollouts",
        "n_epochs",
        "n_steps",
        "net_arch",
        "net_arch_pi",
        "net_arch_vf",
        "num_envs",
        "output_json",
        "pass_arc_end",
        "pass_arc_power",
        "pass_logit_bias_enabled",
        "pass_logit_bias_end",
        "pass_logit_bias_start",
        "pass_oob_power",
        "pass_oob_turnover_prob_end",
        "pass_prob_min",
        "phi_bump_multiplier",
        "phi_bump_updates",
        "profiling_sample_rate",
        "restart_entropy_on_continue",
        "set_cls_tokens",
        "set_embed_dim",
        "set_head_activation",
        "set_heads",
        "set_intent_embedding_dim",
        "set_intent_embedding_enabled",
        "set_token_activation",
        "set_token_mlp_dim",
        "steps_per_alternation",
        "steps_per_alternation_end",
        "steps_per_alternation_schedule",
        "target_kl",
        "tensorboard_path",
        "use_vec_normalize",
    }
)


# This is the application's training recipe. It deliberately lives in Python
# instead of inheriting defaults by executing a user-facing shell script.
APPLICATION_PRESET_VALUES: dict[str, Any] = {
    "action_head_mode": "pointer_targeted",
    "base_passer_risk": 0.02,
    "base_receiver_risk": 0.5,
    "base_steal_rate": 0.2,
    "checkpoint_every_updates": 250,
    "defender_pressure_decay_lambda": 2.0,
    "defender_pressure_distance": 1,
    "defender_pressure_turnover_chance": 0.025,
    "defender_reach_radius": 0.5,
    "defender_reaction_time": 0.15,
    "defender_speed": 1.1,
    "disc_eval_batch_output": True,
    "dunk_pct": 0.6,
    "dunk_std": 0.3,
    "enable_intent_learning": True,
    "enable_multi_possession": True,
    "enable_rebounds": True,
    "ent_coef_end": 0.03,
    "ent_coef_start": 1.0,
    "ent_schedule": "exp",
    "eval_deploy_batches": 2,
    "eval_deploy_every_updates": 250,
    "eval_deploy_horizon": 2048,
    "eval_every_updates": 0,
    "game_winner_reward": 5.0,
    "gamma": 1.0,
    "historical_eval_horizon": 2048,
    "historical_eval_updates": "100,500,2500,5000",
    "illegal_defense_enabled": True,
    "intent_commitment_steps": 8,
    "intent_conditioning_scale": 5.0,
    "intent_disc_batch_size": 512,
    "intent_disc_encoder_type": "set_step",
    "intent_disc_eval_holdout_fraction": 0.1,
    "intent_disc_include_pressure_exposure": False,
    "intent_disc_include_shot_clock": False,
    "intent_disc_updates_per_rollout": 4,
    "intent_diversity_enabled": True,
    "intent_diversity_ramp_updates": 2500,
    "intent_diversity_warmup_updates": 2500,
    "intent_embedding_enabled": True,
    "intent_null_prob": 0.0,
    "intent_policy_sensitivity_log_every_rollouts": 10,
    "intent_policy_sensitivity_sample_states": 64,
    "intent_sample_dump_size": 4096,
    "intent_selector_alpha_ramp_updates": 2500,
    "intent_selector_alpha_warmup_updates": 2500,
    "intent_selector_enabled": True,
    "intent_selector_entropy_coef": 0.03,
    "intent_selector_eps_end": 0.15,
    "intent_selector_eps_ramp_updates": 2500,
    "intent_selector_eps_start": 0.5,
    "intent_selector_eps_warmup_updates": 2500,
    "intent_selector_learning_rate": 0.0001,
    "intent_selector_max_samples_per_update": 1024,
    "intent_selector_min_play_steps": 6,
    "intent_selector_multiselect_enabled": True,
    "intent_selector_train_every_rollouts": 8,
    "intent_selector_usage_reg_coef": 0.05,
    "kernel_batch_size": 512,
    "layup_std": 0.05,
    "learning_rate": 0.0003,
    "log_mlflow": True,
    "made_basket_restart_mode": "check",
    "min_shot_clock": 24,
    "mlflow_experiment_name": "halfcourt_multi_possessions",
    "multi_possession_limit_end": 25,
    "multi_possession_limit_ramp_updates": 5000,
    "multi_possession_limit_start": 1,
    "multi_possession_reward_mode": "scoring_events",
    "num_updates": 5000,
    "offensive_three_seconds": True,
    "opponent_deterministic_episode_prob_end": 0.8,
    "opponent_deterministic_episode_prob_ramp_updates": 15000,
    "opponent_deterministic_episode_prob_start": 0.2,
    "opponent_pool_beta": 0.9,
    "opponent_pool_exploration": 0.3,
    "opponent_pool_size": 30,
    "pass_interception_model": "reaction",
    "pass_speed": 5.0,
    "passer_pressure_decay": 1.0,
    "players": 5,
    "policy_model": "attention",
    "ppo_minibatches": 16,
    "rebound_basket_position_weight": 1.0,
    "rebound_contest_mode": "local_contest",
    "rebound_contest_radius": 2,
    "rebound_skill_std": 1.0,
    "rebound_skill_weight": 1.0,
    "run_train_loop": True,
    "score_potential_scale": 0.0,
    "steal_distance_factor": 0.5,
    "steal_position_weight_min": 0.8,
    "task_reward_scale_end": 1.0,
    "task_reward_scale_ramp_updates": 2000,
    "task_reward_scale_start": 0.1,
    "task_reward_scale_warmup_updates": 0,
    "three_pt_pct": 0.37,
    "three_pt_std": 0.05,
    "use_set_obs": True,
    "vf_coef": 0.75,
}


def application_preset_values(repo_root: Path) -> dict[str, Any]:
    return {
        **APPLICATION_PRESET_VALUES,
        "rebound_table_model_dir": str(
            repo_root
            / "analytics"
            / "rebound_physics"
            / "outputs"
            / "dataset_9x8"
            / "fitted_catch_model"
        ),
    }


@lru_cache(maxsize=4)
def load_application_config_schema(repo_root_text: str) -> list[dict[str, Any]]:
    from basketworld_jax.train.main import (
        build_train_config_schema,
        parse_args,
    )

    default_schema = [
        field
        for field in build_train_config_schema(parse_args([]))
        if field["name"] not in JAX_INAPPLICABLE_FIELDS
    ]
    preset_argv = compile_config_values(
        application_preset_values(Path(repo_root_text)),
        default_schema,
        require_editable=False,
    )
    return [
        field
        for field in build_train_config_schema(parse_args(preset_argv))
        if field["name"] not in JAX_INAPPLICABLE_FIELDS
    ]


@lru_cache(maxsize=128)
def resolve_run_config_schema(command: tuple[str, ...]) -> list[dict[str, Any]]:
    from basketworld_jax.train.main import build_train_config_schema, parse_args

    try:
        module_index = command.index("basketworld_jax.train.main")
    except ValueError as exc:
        raise ValueError("Run command does not invoke the JAX trainer module.") from exc
    trainer_argv = list(command[module_index + 1 :])
    return [
        {**field, "value": field["effective_value"]}
        for field in build_train_config_schema(parse_args(trainer_argv))
        if field["name"] not in JAX_INAPPLICABLE_FIELDS
    ]
