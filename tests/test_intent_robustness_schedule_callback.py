from basketworld.utils.callbacks import IntentRobustnessScheduleCallback


class _DummyVecEnv:
    def __init__(self):
        self.values = {}

    def env_method(self, name, value):
        self.values[name] = float(value)


class _DummyModel:
    def __init__(self):
        self.num_timesteps = 0
        self._env = _DummyVecEnv()

    def get_env(self):
        return self._env


def test_intent_robustness_schedule_updates_env_methods():
    cb = IntentRobustnessScheduleCallback(
        null_start=0.2,
        null_end=0.8,
        visible_start=0.1,
        visible_end=0.0,
        total_planned_timesteps=100,
        log_freq_rollouts=1,
        timestep_offset=0,
    )
    model = _DummyModel()
    cb.init_callback(model)

    cb._on_training_start()
    assert abs(model.get_env().values["set_intent_null_prob"] - 0.2) < 1e-9
    assert (
        abs(model.get_env().values["set_intent_visible_to_defense_prob"] - 0.1)
        < 1e-9
    )

    model.num_timesteps = 100
    cb._on_rollout_end()
    assert abs(model.get_env().values["set_intent_null_prob"] - 0.8) < 1e-9
    assert (
        abs(model.get_env().values["set_intent_visible_to_defense_prob"] - 0.0)
        < 1e-9
    )
