from basketworld.envs.basketworld_env_v2 import HexagonBasketballEnv, Team
from basketworld.utils.wrappers import SetObservationWrapper

from app.backend.state import game_state, get_full_game_state


def test_get_full_game_state_preserves_three_point_hexes_with_wrapped_env():
    original_state = game_state.__dict__.copy()
    try:
        env = HexagonBasketballEnv(players=3, render_mode=None)
        wrapped_env = SetObservationWrapper(env)
        obs, _ = wrapped_env.reset()

        game_state.env = wrapped_env
        game_state.obs = obs
        game_state.user_team = Team.OFFENSE
        game_state.unified_policy = None
        game_state.offense_policy = None
        game_state.defense_policy = None
        game_state.reward_history = []
        game_state.episode_rewards = {"offense": 0.0, "defense": 0.0}
        game_state.shot_log = []
        game_state.phi_log = []
        game_state.actions_log = []
        game_state.episode_states = []
        game_state.mlflow_training_params = None
        game_state.mlflow_phi_shaping_params = None

        state = get_full_game_state()

        assert state["three_point_hexes"]
        assert sorted(state["three_point_hexes"]) == sorted(
            [(int(q), int(r)) for q, r in env._three_point_hexes]
        )
        assert sorted(state["three_point_line_hexes"]) == sorted(
            [(int(q), int(r)) for q, r in env._three_point_line_hexes]
        )
    finally:
        game_state.__dict__.clear()
        game_state.__dict__.update(original_state)
