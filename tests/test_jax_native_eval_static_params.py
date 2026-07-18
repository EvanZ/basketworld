from basketworld_jax.eval.native import _split_native_env_params


def test_rebound_reward_params_are_jax_static_overrides():
    env_kwargs, static_overrides = _split_native_env_params(
        {
            "shot_pressure_enabled": True,
            "enable_rebound_reward_redistribution": "true",
            "offensive_rebound_reward_advance": "0.4",
            "rebound_reward_once_per_possession": "false",
        }
    )

    assert env_kwargs == {"shot_pressure_enabled": True}
    assert static_overrides == {
        "enable_rebound_reward_redistribution": True,
        "offensive_rebound_reward_advance": 0.4,
        "rebound_reward_once_per_possession": False,
    }
