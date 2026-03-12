import numpy as np

from basketworld.envs.basketworld_env_v2 import HexagonBasketballEnv, Team
from basketworld.utils.self_play_wrapper import SelfPlayEnvWrapper


class _DummyInnerPolicy:
    def __init__(self):
        self.pass_mode_calls: list[str] = []

    def set_pass_mode(self, mode: str) -> None:
        self.pass_mode_calls.append(str(mode))


class _DummySB3Policy:
    def __init__(self, output_actions):
        self.policy = _DummyInnerPolicy()
        self._output_actions = np.array(output_actions, dtype=int)
        self.last_obs = None

    def predict(self, obs, deterministic=False):
        self.last_obs = obs
        del deterministic
        return np.array(self._output_actions, dtype=int), None


def _make_env(training_team=Team.OFFENSE, pass_mode="pointer_targeted"):
    return HexagonBasketballEnv(
        players=3,
        render_mode=None,
        training_team=training_team,
        pass_mode=pass_mode,
    )


def test_wrapper_applies_pass_mode_to_opponent_policy_object():
    env = _make_env(training_team=Team.OFFENSE, pass_mode="pointer_targeted")
    opponent = _DummySB3Policy(output_actions=[0, 0, 0])

    wrapped = SelfPlayEnvWrapper(env, opponent_policy=opponent)
    wrapped.reset(seed=7)

    assert opponent.policy.pass_mode_calls, "Expected at least one pass mode application call."
    assert opponent.policy.pass_mode_calls[-1] == "pointer_targeted"


def test_wrapper_maps_full_length_opponent_actions_to_opponent_team(monkeypatch):
    env = _make_env(training_team=Team.OFFENSE, pass_mode="directional")
    n_players = env.n_players
    players_per_side = env.players_per_side

    # Legacy-style output: one action per absolute player id.
    # Defense ids are expected to receive the tail actions [4, 5, 6] for players=3.
    opponent = _DummySB3Policy(output_actions=[1, 2, 3, 4, 5, 6][:n_players])
    wrapped = SelfPlayEnvWrapper(env, opponent_policy=opponent)
    wrapped.reset(seed=11)

    # Keep resolver transparent so we can verify raw remapping behavior.
    monkeypatch.setattr(
        "basketworld.utils.self_play_wrapper.resolve_illegal_actions",
        lambda predicted_actions, action_mask, strategy, deterministic, probs_per_player=None: np.array(predicted_actions, dtype=int),
    )

    captured = {}

    def _fake_env_step(full_action):
        captured["full_action"] = np.array(full_action, dtype=int)
        obs = wrapped._last_obs
        reward = np.zeros(wrapped.env.unwrapped.n_players, dtype=np.float32)
        done = True
        truncated = False
        info = {"action_results": {}, "training_team": wrapped.env.unwrapped.training_team.name}
        return obs, reward, done, truncated, info

    monkeypatch.setattr(wrapped.env, "step", _fake_env_step)

    training_action = np.zeros(players_per_side, dtype=int)
    wrapped.step(training_action)

    assert "full_action" in captured
    full_action = captured["full_action"]
    defense_ids = wrapped.opponent_player_ids
    assert full_action[defense_ids].tolist() == [4, 5, 6][: len(defense_ids)]


def test_wrapper_applies_pass_mode_when_loading_opponent_from_path(monkeypatch):
    env = _make_env(training_team=Team.DEFENSE, pass_mode="pointer_targeted")
    fake_loaded = _DummySB3Policy(output_actions=[0, 0, 0])

    monkeypatch.setattr(
        "basketworld.utils.self_play_wrapper.PPO.load",
        lambda *args, **kwargs: fake_loaded,
    )

    wrapped = SelfPlayEnvWrapper(env, opponent_policy="/tmp/fake_opp.zip")
    wrapped.reset(seed=13)
    wrapped.step(np.zeros(env.players_per_side, dtype=int))

    assert fake_loaded.policy.pass_mode_calls, "Expected pass mode to be set on loaded opponent policy."
    assert fake_loaded.policy.pass_mode_calls[-1] == "pointer_targeted"


def test_wrapper_reconditions_intent_fields_for_opponent_role():
    env = HexagonBasketballEnv(
        players=3,
        render_mode=None,
        training_team=Team.DEFENSE,
        enable_intent_learning=True,
        intent_null_prob=0.0,
        intent_visible_to_defense_prob=0.0,
    )
    opponent = _DummySB3Policy(output_actions=[0, 0, 0])
    wrapped = SelfPlayEnvWrapper(env, opponent_policy=opponent)

    obs, _ = wrapped.reset(seed=5)
    assert float(obs["intent_active"][0]) == 0.0
    assert float(obs["intent_visible"][0]) == 0.0

    wrapped.step(np.zeros(env.players_per_side, dtype=int))
    assert opponent.last_obs is not None
    assert float(opponent.last_obs["role_flag"][0]) == 1.0
    assert float(opponent.last_obs["intent_visible"][0]) == 1.0
