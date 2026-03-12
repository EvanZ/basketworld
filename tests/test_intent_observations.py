import numpy as np

from basketworld.envs.basketworld_env_v2 import HexagonBasketballEnv, Team


def test_intent_fields_absent_when_disabled():
    env = HexagonBasketballEnv(players=3, enable_intent_learning=False)
    obs, _ = env.reset(seed=11)
    assert "intent_index" not in obs
    assert "intent_active" not in obs
    assert "intent_visible" not in obs


def test_intent_fields_present_and_visible_for_offense_training():
    env = HexagonBasketballEnv(
        players=3,
        training_team=Team.OFFENSE,
        enable_intent_learning=True,
        intent_null_prob=0.0,
        num_intents=8,
    )
    obs, _ = env.reset(seed=9)
    assert "intent_index" in obs
    assert float(obs["intent_active"][0]) == 1.0
    assert float(obs["intent_visible"][0]) == 1.0
    intent_index = float(obs["intent_index"][0])
    assert 0.0 <= intent_index <= 7.0


def test_intent_index_masked_for_defense_when_private():
    env = HexagonBasketballEnv(
        players=3,
        training_team=Team.DEFENSE,
        enable_intent_learning=True,
        intent_null_prob=0.0,
        intent_visible_to_defense_prob=0.0,
    )
    obs, _ = env.reset(seed=17)
    assert float(obs["intent_active"][0]) == 0.0
    assert float(obs["intent_visible"][0]) == 0.0
    assert float(obs["intent_index"][0]) == 0.0
    assert np.isfinite(obs["intent_age_norm"]).all()


def test_defense_observer_sees_own_defense_intent_when_enabled():
    env = HexagonBasketballEnv(
        players=3,
        training_team=Team.DEFENSE,
        enable_intent_learning=True,
        intent_null_prob=0.0,
        enable_defense_intent_learning=True,
        defense_intent_null_prob=0.0,
    )
    obs, _ = env.reset(seed=23)
    assert float(obs["intent_active"][0]) == 1.0
    assert float(obs["intent_visible"][0]) == 1.0
    assert 0.0 <= float(obs["intent_index"][0]) <= 7.0
