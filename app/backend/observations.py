import copy
from typing import Dict, List, Tuple

import numpy as np
import torch
import gymnasium as gym
from basketworld.envs.basketworld_env_v2 import ActionType, Team
from stable_baselines3 import PPO

from basketworld.utils.action_resolution import (
    IllegalActionStrategy,
    get_policy_action_probabilities,
    resolve_illegal_actions,
)
from basketworld.utils.intent_policy_sensitivity import (
    sync_policy_runtime_intent_override_from_env,
)
from basketworld.utils.wrappers import SetObservationWrapper

from .env_access import env_view
from .state import GameState, _role_flag_value_for_team, game_state

_predict_failure_count = 0


def _resolve_base_env(env):
    return getattr(env, "unwrapped", env)


def _apply_observation_wrappers(env, obs):
    """Apply observation wrappers explicitly without deprecated wrapper forwarding."""
    if env is None:
        return obs
    base_env = _resolve_base_env(env)
    wrappers = []
    current = env
    visited = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if current is base_env:
            break
        if isinstance(current, gym.ObservationWrapper):
            wrappers.append(current)
        current = getattr(current, "env", None)
    rebuilt = obs
    for wrapper in reversed(wrappers):
        rebuilt = wrapper.observation(rebuilt)
    return rebuilt


def _apply_intent_fields_for_role(cloned: Dict, env, role_flag_value: float) -> Dict:
    """Recondition role-dependent intent fields for cloned observations."""
    if env is None:
        return cloned
    if "players" in cloned and "globals" in cloned:
        return cloned
    base_env = _resolve_base_env(env)
    observer_is_offense = bool(float(role_flag_value) > 0.0)
    try:
        fields = base_env.get_intent_observation_fields(observer_is_offense)
    except Exception:
        fields = {}
    if fields:
        for key, value in fields.items():
            cloned[key] = np.array(value, dtype=np.float32, copy=True)
    if "globals" in cloned:
        try:
            cloned["globals"] = base_env.patch_globals_with_intent_features(
                cloned["globals"], observer_is_offense
            )
        except Exception:
            pass
    return cloned


def _clone_obs_with_role_flag(obs: Dict, role_flag_value: float) -> Dict:
    cloned = {
        "obs": np.copy(obs["obs"]),
        "action_mask": obs["action_mask"],
        "role_flag": np.array([role_flag_value], dtype=np.float32),
    }
    skills = obs.get("skills")
    if skills is not None:
        cloned["skills"] = np.copy(skills)
    else:
        cloned["skills"] = None
    players = obs.get("players") if isinstance(obs, dict) else None
    globals_vec = obs.get("globals") if isinstance(obs, dict) else None
    if players is not None:
        cloned["players"] = np.copy(players)
    if globals_vec is not None:
        cloned["globals"] = np.copy(globals_vec)
    if isinstance(obs, dict):
        for key, value in obs.items():
            if key in cloned:
                continue
            if key in {"intent_index", "intent_active", "intent_visible", "intent_age_norm"}:
                continue
            if isinstance(value, np.ndarray):
                cloned[key] = np.copy(value)
    return cloned


def _ensure_set_obs(policy: PPO | None, env, obs: dict | None) -> dict | None:
    if policy is None or env is None or obs is None:
        return obs
    try:
        policy_obj = getattr(policy, "policy", None)
        obs_space = getattr(policy_obj, "observation_space", None)
        if not isinstance(obs_space, gym.spaces.Dict):
            return obs
        if "players" not in obs_space.spaces or "globals" not in obs_space.spaces:
            return obs
    except Exception:
        return obs
    if isinstance(obs, dict) and "players" in obs and "globals" in obs:
        return obs
    try:
        wrapper = SetObservationWrapper(env)
        return wrapper.observation(obs)
    except Exception:
        return obs


def validate_policy_observation_schema(
    policy: PPO | None,
    env,
    obs: dict | None,
    policy_label: str = "policy",
) -> dict | None:
    """Validate that the observation payload matches the policy observation space.

    Raises ValueError with actionable details on mismatch.
    """
    if policy is None:
        return obs
    policy_obj = getattr(policy, "policy", None)
    if policy_obj is None:
        return obs
    obs_space = getattr(policy_obj, "observation_space", None)
    if obs_space is None:
        return obs

    prepared_obs = _ensure_set_obs(policy, env, obs)

    if isinstance(obs_space, gym.spaces.Dict):
        if not isinstance(prepared_obs, dict):
            raise ValueError(
                f"{policy_label}: expected dict observation, received {type(prepared_obs).__name__}"
            )
        expected_keys = set(obs_space.spaces.keys())
        actual_keys = set(prepared_obs.keys())
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        if missing or extra:
            raise ValueError(
                f"{policy_label}: observation keys mismatch. missing={missing} extra={extra}"
            )
        for key, space in obs_space.spaces.items():
            arr = np.asarray(prepared_obs[key])
            expected_shape = tuple(getattr(space, "shape", ()))
            actual_shape = tuple(arr.shape)
            if actual_shape != expected_shape:
                raise ValueError(
                    f"{policy_label}: key='{key}' shape mismatch expected={expected_shape} actual={actual_shape}"
                )
        return prepared_obs

    # Non-dict observation spaces: use canonical "obs" key when available.
    if isinstance(prepared_obs, dict):
        if "obs" not in prepared_obs:
            raise ValueError(
                f"{policy_label}: non-dict policy expects flat obs but payload has no 'obs' key"
            )
        arr = np.asarray(prepared_obs["obs"])
    else:
        arr = np.asarray(prepared_obs)
    expected_shape = tuple(getattr(obs_space, "shape", ()))
    actual_shape = tuple(arr.shape)
    if actual_shape != expected_shape:
        raise ValueError(
            f"{policy_label}: observation shape mismatch expected={expected_shape} actual={actual_shape}"
        )
    return prepared_obs


def _team_player_ids(env, team: Team) -> List[int]:
    env_read = env_view(env)
    if team == Team.OFFENSE:
        return list(env_read.offense_ids or [])
    return list(env_read.defense_ids or [])


def _resolve_role_flag_value_for_team(
    team: Team,
    *,
    role_flag_offense: float | None = None,
    role_flag_defense: float | None = None,
) -> float:
    if team == Team.OFFENSE and role_flag_offense is not None:
        return float(role_flag_offense)
    if team == Team.DEFENSE and role_flag_defense is not None:
        return float(role_flag_defense)
    return _role_flag_value_for_team(team)


def rebuild_observation_from_env(
    env,
    *,
    current_obs: dict | None = None,
    role_flag_value: float | None = None,
) -> dict:
    """Rebuild an observation dict from the current env state for the current viewer."""
    base_env = _resolve_base_env(env)
    if role_flag_value is None:
        if (
            current_obs is not None
            and isinstance(current_obs, dict)
            and current_obs.get("role_flag") is not None
        ):
            role_flag_value = float(
                np.asarray(current_obs.get("role_flag"), dtype=np.float32).reshape(-1)[0]
            )
        else:
            role_flag_value = 1.0 if base_env.training_team == Team.OFFENSE else -1.0
    observer_is_offense = bool(float(role_flag_value) > 0.0)
    if hasattr(base_env, "_build_observation_dict"):
        rebuilt_obs = base_env._build_observation_dict(observer_is_offense)
    else:
        rebuilt_obs = {
            "obs": base_env._get_observation(),
            "action_mask": base_env._get_action_masks(),
            "skills": base_env._get_offense_skills_array(),
        }
    rebuilt_obs = _apply_observation_wrappers(env, rebuilt_obs)
    rebuilt_obs["role_flag"] = np.array([float(role_flag_value)], dtype=np.float32)
    return rebuilt_obs


def _predict_actions_for_team(
    policy: PPO,
    base_obs: Dict,
    env,
    team: Team,
    deterministic: bool,
    strategy: IllegalActionStrategy,
    *,
    role_flag_offense: float | None = None,
    role_flag_defense: float | None = None,
) -> Tuple[Dict[int, int], Dict[int, np.ndarray]]:
    actions_by_player: Dict[int, int] = {}
    probs_by_player: Dict[int, np.ndarray] = {}

    if policy is None or base_obs is None or env is None:
        return actions_by_player, probs_by_player

    base_obs = _ensure_set_obs(policy, env, base_obs)
    team_ids = _team_player_ids(env, team)
    if not team_ids:
        return actions_by_player, probs_by_player
    env_read = env_view(env)
    num_players = int(env_read.n_players or 0)

    role_flag_value = _resolve_role_flag_value_for_team(
        team,
        role_flag_offense=role_flag_offense,
        role_flag_defense=role_flag_defense,
    )
    conditioned_obs = _clone_obs_with_role_flag(base_obs, role_flag_value)
    conditioned_obs = _apply_intent_fields_for_role(conditioned_obs, env, role_flag_value)

    raw_actions = None
    try:
        sync_policy_runtime_intent_override_from_env(
            policy,
            env,
            observer_is_offense=bool(float(role_flag_value) > 0.0),
        )
        raw_actions, _ = policy.predict(conditioned_obs, deterministic=deterministic)
    except Exception as err:
        global _predict_failure_count
        _predict_failure_count += 1
        if _predict_failure_count <= 5:
            team_name = getattr(team, "name", str(team))
            print(
                f"[OBS][WARN] policy.predict failed for team={team_name} "
                f"(deterministic={deterministic}) count={_predict_failure_count}: {err}"
            )
        raw_actions = None

    if raw_actions is None:
        raw_actions = np.zeros(len(team_ids), dtype=int)
    raw_actions = np.array(raw_actions).reshape(-1)
    action_len = raw_actions.shape[0]
    team_mask = base_obs["action_mask"][team_ids]

    # Legacy policies output actions for every player; new policies output players_per_side only.
    if action_len == len(team_ids):
        team_pred_actions = raw_actions
    elif action_len == num_players:
        team_pred_actions = raw_actions[team_ids]
    else:
        # Fallback: truncate/pad to team size
        team_pred_actions = raw_actions[: len(team_ids)]

    probs = get_policy_action_probabilities(policy, conditioned_obs)
    if probs is not None:
        probs = [np.asarray(p, dtype=np.float32) for p in probs]
        if len(probs) == num_players:
            team_probs = [probs[int(pid)] for pid in team_ids]
        else:
            team_probs = probs[: len(team_ids)]
    else:
        team_probs = None

    # Deterministic policy execution should select argmax among legal actions from
    # the model's probability distribution directly. This avoids edge cases where
    # policy.predict(..., deterministic=True) may not match joint action argmax.
    if deterministic and team_probs is not None and len(team_probs) == len(team_ids):
        resolved_actions = np.zeros(len(team_ids), dtype=int)
        for idx in range(len(team_ids)):
            legal = np.where(team_mask[idx] == 1)[0]
            if len(legal) == 0:
                resolved_actions[idx] = 0
                continue
            p = np.asarray(team_probs[idx], dtype=np.float32)
            if p.shape[0] <= int(np.max(legal)):
                # Fallback to existing resolver if probability vector is malformed.
                resolved_actions = resolve_illegal_actions(
                    np.array(team_pred_actions),
                    team_mask,
                    strategy,
                    deterministic,
                    team_probs,
                )
                break
            masked = p[legal]
            resolved_actions[idx] = int(legal[int(np.argmax(masked))])
    else:
        resolved_actions = resolve_illegal_actions(
            np.array(team_pred_actions),
            team_mask,
            strategy,
            deterministic,
            team_probs,
        )

    for idx, pid in enumerate(team_ids):
        actions_by_player[int(pid)] = int(resolved_actions[idx])
        if team_probs is not None and idx < len(team_probs):
            probs_by_player[int(pid)] = np.asarray(team_probs[idx], dtype=np.float32)

    return actions_by_player, probs_by_player


def _predict_policy_actions(
    policy: PPO,
    base_obs: Dict,
    env,
    deterministic: bool,
    strategy: IllegalActionStrategy,
    *,
    role_flag_offense: float | None = None,
    role_flag_defense: float | None = None,
) -> Tuple[np.ndarray | None, List[np.ndarray] | None]:
    if policy is None or base_obs is None or env is None:
        return None, None

    env_read = env_view(env)
    num_players = int(env_read.n_players or 0)
    num_actions = len(ActionType)
    full_actions = np.zeros(num_players, dtype=int)
    probs_per_player: List[np.ndarray] = [
        np.zeros(num_actions, dtype=np.float32) for _ in range(num_players)
    ]

    for team in (Team.OFFENSE, Team.DEFENSE):
        team_actions, team_probs = _predict_actions_for_team(
            policy,
            base_obs,
            env,
            team,
            deterministic,
            strategy,
            role_flag_offense=role_flag_offense,
            role_flag_defense=role_flag_defense,
        )
        for pid, action in team_actions.items():
            full_actions[int(pid)] = int(action)
        for pid, prob_vec in team_probs.items():
            probs_per_player[int(pid)] = prob_vec

    return full_actions, probs_per_player


def calculate_phi_from_ep_data(
    ep_by_player: list[float],
    ball_handler_id: int,
    offense_ids: list[int],
    phi_params: dict,
) -> float:
    """Calculate phi value from EP data using specified parameters."""
    if not ep_by_player or ball_handler_id < 0 or not offense_ids:
        return 0.0

    ball_ep = ep_by_player[ball_handler_id] if ball_handler_id < len(ep_by_player) else 0.0
    if phi_params.get("phi_use_ball_handler_only", False):
        return ball_ep

    mode = phi_params.get("phi_aggregation_mode", "team_best")
    blend_weight = phi_params.get("phi_blend_weight", 0.0)

    offense_eps = [ep_by_player[pid] for pid in offense_ids if pid < len(ep_by_player)]
    if not offense_eps:
        return ball_ep

    teammate_eps = [
        ep_by_player[pid] for pid in offense_ids if pid != ball_handler_id and pid < len(ep_by_player)
    ]
    if not teammate_eps:
        teammate_aggregate = ball_ep
    elif mode == "teammates_best":
        teammate_aggregate = max(teammate_eps)
    elif mode == "teammates_avg":
        teammate_aggregate = sum(teammate_eps) / len(teammate_eps)
    elif mode == "teammates_worst":
        teammate_aggregate = min(teammate_eps)
    elif mode == "team_worst":
        teammate_aggregate = min(min(teammate_eps), ball_ep)
    elif mode == "team_avg":
        teammate_aggregate = sum(offense_eps) / len(offense_eps)
    else:  # team_best (default)
        teammate_aggregate = max(max(teammate_eps), ball_ep)

    w = max(0.0, min(1.0, blend_weight))
    return (1.0 - w) * teammate_aggregate + w * ball_ep


def _build_role_conditioned_obs(base_obs: dict | None, role_flag_value: float):
    """Prepare an observation payload with a specific role flag for value prediction."""
    if base_obs is None or "obs" not in base_obs or "action_mask" not in base_obs:
        return None
    conditioned = {
        "obs": np.copy(base_obs["obs"]),
        "action_mask": base_obs["action_mask"],
        "role_flag": np.array([role_flag_value], dtype=np.float32),
        "skills": np.copy(base_obs.get("skills")) if base_obs.get("skills") is not None else None,
    }
    players = base_obs.get("players") if isinstance(base_obs, dict) else None
    globals_vec = base_obs.get("globals") if isinstance(base_obs, dict) else None
    if players is not None:
        conditioned["players"] = np.copy(players)
    if globals_vec is not None:
        conditioned["globals"] = np.copy(globals_vec)
    if isinstance(base_obs, dict):
        for key, value in base_obs.items():
            if key in conditioned:
                continue
            if isinstance(value, np.ndarray):
                conditioned[key] = np.copy(value)
    try:
        conditioned = _apply_intent_fields_for_role(
            conditioned, game_state.env, role_flag_value
        )
    except Exception:
        pass
    return conditioned


def _compute_state_values_from_obs(obs_dict: dict | None):
    """Compute offensive/defensive value estimates for a given observation snapshot."""
    if not game_state.unified_policy or obs_dict is None:
        return None

    value_policy = game_state.unified_policy
    if not hasattr(value_policy, "policy"):
        return None

    try:
        obs_dict = _ensure_set_obs(value_policy, game_state.env, obs_dict)
        offense_obs = _build_role_conditioned_obs(obs_dict, game_state.role_flag_offense)
        defense_obs = _build_role_conditioned_obs(obs_dict, game_state.role_flag_defense)
        if offense_obs is None or defense_obs is None:
            return None

        sync_policy_runtime_intent_override_from_env(
            value_policy, game_state.env, observer_is_offense=True
        )
        offense_tensor, _ = value_policy.policy.obs_to_tensor(offense_obs)
        sync_policy_runtime_intent_override_from_env(
            value_policy, game_state.env, observer_is_offense=False
        )
        defense_tensor, _ = value_policy.policy.obs_to_tensor(defense_obs)

        with torch.no_grad():
            offense_value = float(value_policy.policy.predict_values(offense_tensor).item())
            defense_value = float(value_policy.policy.predict_values(defense_tensor).item())

        return {"offensive_value": offense_value, "defensive_value": defense_value}
    except Exception as err:
        print(f"[STATE_VALUES] Failed to compute state values: {err}")
        import traceback

        traceback.print_exc()
        return None


def _compute_q_values_for_player(player_id: int, state: GameState) -> dict:
    """Compute Q-values for all actions for a given player."""
    action_values = {}
    if not state.env or state.obs is None:
        return action_values

    value_policy = state.unified_policy
    gamma = value_policy.gamma
    possible_actions = [action.name for action in ActionType]
    state_env_read = env_view(state.env)
    state_offense_ids = set(state_env_read.offense_ids or [])
    state_defense_ids = set(state_env_read.defense_ids or [])
    state_num_players = int(state_env_read.n_players or 0)

    for action_name in possible_actions:
        action_id = ActionType[action_name].value
        temp_env = copy.deepcopy(state.env)
        temp_env_read = env_view(temp_env)
        temp_num_players = int(temp_env_read.n_players or state_num_players)
        sim_action = np.zeros(temp_num_players, dtype=int)

        full_actions_main, _ = _predict_policy_actions(
            state.unified_policy,
            state.obs,
            state.env,
            deterministic=True,
            strategy=IllegalActionStrategy.BEST_PROB,
        )
        if full_actions_main is None:
            full_actions_main = np.zeros(temp_num_players, dtype=int)

        full_actions_opponent, _ = _predict_policy_actions(
            state.defense_policy,
            state.obs,
            state.env,
            deterministic=True,
            strategy=IllegalActionStrategy.BEST_PROB,
        )

        is_player_on_user_team = (
            (player_id in state_offense_ids and state.user_team == Team.OFFENSE)
            or (player_id in state_defense_ids and state.user_team == Team.DEFENSE)
        )

        for i in range(temp_num_players):
            if i == player_id:
                sim_action[i] = action_id
            else:
                is_i_on_user_team = (
                    (i in state_offense_ids and state.user_team == Team.OFFENSE)
                    or (i in state_defense_ids and state.user_team == Team.DEFENSE)
                )
                if is_i_on_user_team:
                    sim_action[i] = full_actions_main[i]
                elif full_actions_opponent is not None:
                    sim_action[i] = full_actions_opponent[i]
                else:
                    sim_action[i] = full_actions_main[i]

        try:
            next_obs, reward, _, _, _ = temp_env.step(sim_action)
        except ValueError:
            # Episode ended; return empty to avoid crashing endpoints when in terminal states
            return {}
        next_obs = _ensure_set_obs(value_policy, temp_env, next_obs)
        role_flag_value = (
            state.role_flag_offense if player_id in state_offense_ids else state.role_flag_defense
        )
        conditioned_next_obs = {
            "obs": np.copy(next_obs["obs"]),
            "action_mask": next_obs["action_mask"],
            "role_flag": np.array([role_flag_value], dtype=np.float32),
            "skills": np.copy(next_obs.get("skills")) if next_obs.get("skills") is not None else None,
        }
        players = next_obs.get("players") if isinstance(next_obs, dict) else None
        globals_vec = next_obs.get("globals") if isinstance(next_obs, dict) else None
        if players is not None:
            conditioned_next_obs["players"] = np.copy(players)
        if globals_vec is not None:
            conditioned_next_obs["globals"] = np.copy(globals_vec)
        if isinstance(next_obs, dict):
            for key, value in next_obs.items():
                if key in conditioned_next_obs:
                    continue
                if isinstance(value, np.ndarray):
                    conditioned_next_obs[key] = np.copy(value)
        conditioned_next_obs = _apply_intent_fields_for_role(
            conditioned_next_obs, temp_env, role_flag_value
        )

        sync_policy_runtime_intent_override_from_env(
            value_policy,
            temp_env,
            observer_is_offense=bool(float(role_flag_value) > 0.0),
        )
        next_obs_tensor, _ = value_policy.policy.obs_to_tensor(conditioned_next_obs)
        with torch.no_grad():
            next_value = value_policy.policy.predict_values(next_obs_tensor)

        team_reward = reward[player_id]
        q_value = team_reward + gamma * next_value.item()
        if abs(q_value) > 2.5:
            print(
                f"[Q-VALUE WARNING] Player {player_id} action {action_name}: "
                f"Q={q_value:.3f}, r={team_reward:.3f}, V(s')={next_value.item():.3f}, "
                f"role_flag={role_flag_value}, gamma={gamma}"
            )

        action_values[action_name] = q_value

    return action_values


def _compute_policy_probabilities_for_obs(base_obs: dict, env) -> dict | None:
    """Compute policy probabilities for a provided observation snapshot."""
    if not game_state.unified_policy or base_obs is None or env is None:
        return None
    try:
        _, raw_probs_main = _predict_policy_actions(
            game_state.unified_policy,
            base_obs,
            env,
            deterministic=False,
            strategy=IllegalActionStrategy.SAMPLE_PROB,
        )

        raw_probs_opponent = None
        if game_state.defense_policy is not None:
            _, raw_probs_opponent = _predict_policy_actions(
                game_state.defense_policy,
                base_obs,
                env,
                deterministic=False,
                strategy=IllegalActionStrategy.SAMPLE_PROB,
            )

        if raw_probs_main is None:
            return None

        action_mask = base_obs["action_mask"]
        probs_list = []
        env_read = env_view(env)
        offense_ids = set(env_read.offense_ids or [])
        defense_ids = set(env_read.defense_ids or [])

        for pid in range(int(env_read.n_players or 0)):
            is_user_team = (
                (pid in offense_ids and game_state.user_team == Team.OFFENSE)
                or (pid in defense_ids and game_state.user_team == Team.DEFENSE)
            )
            if is_user_team or raw_probs_opponent is None:
                probs = raw_probs_main[pid]
            else:
                probs = raw_probs_opponent[pid]
            mask = action_mask[pid]
            masked = probs * mask
            total = float(np.sum(masked))
            if total <= 0:
                legal = (mask > 0).astype(np.float32)
                if float(np.sum(legal)) > 0:
                    raw_total = float(np.sum(probs))
                    if raw_total > 0:
                        masked = probs * legal
                        total = float(np.sum(masked))
                    if total <= 0:
                        masked = legal
                        total = float(np.sum(masked))
            if total > 0:
                masked = masked / total
            probs_list.append(masked.tolist())

        return {player_id: probs for player_id, probs in enumerate(probs_list)}
    except Exception as err:
        print(f"[policy_prob_preview] Failed to compute policy probabilities: {err}")
        return None


def compute_policy_probabilities():
    """
    Helper function to compute policy probabilities for the current observation.
    Returns a dict mapping player_id to their action probabilities, or None if error.
    """
    if not game_state.env or not game_state.unified_policy or game_state.obs is None:
        return None

    try:
        _, raw_probs_main = _predict_policy_actions(
            game_state.unified_policy,
            game_state.obs,
            game_state.env,
            deterministic=False,
            strategy=IllegalActionStrategy.SAMPLE_PROB,
        )

        raw_probs_opponent = None
        if game_state.defense_policy is not None:
            _, raw_probs_opponent = _predict_policy_actions(
                game_state.defense_policy,
                game_state.obs,
                game_state.env,
                deterministic=False,
                strategy=IllegalActionStrategy.SAMPLE_PROB,
            )

        if raw_probs_main is None:
            return None

        action_mask = game_state.obs["action_mask"]  # shape (n_players, n_actions)
        probs_list = []
        env_read = env_view(game_state.env)
        offense_ids = set(env_read.offense_ids or [])
        defense_ids = set(env_read.defense_ids or [])

        for pid in range(int(env_read.n_players or 0)):
            is_user_team = (
                (pid in offense_ids and game_state.user_team == Team.OFFENSE)
                or (pid in defense_ids and game_state.user_team == Team.DEFENSE)
            )

            if is_user_team or raw_probs_opponent is None:
                probs = raw_probs_main[pid]
            else:
                probs = raw_probs_opponent[pid]

            mask = action_mask[pid]
            masked = probs * mask
            total = float(np.sum(masked))
            if total <= 0:
                legal = (mask > 0).astype(np.float32)
                if float(np.sum(legal)) > 0:
                    raw_total = float(np.sum(probs))
                    if raw_total > 0:
                        masked = probs * legal
                        total = float(np.sum(masked))
                    if total <= 0:
                        masked = legal
                        total = float(np.sum(masked))
            if total > 0:
                masked = masked / total
            probs_list.append(masked.tolist())

        return {player_id: probs for player_id, probs in enumerate(probs_list)}
    except Exception as e:
        print(f"[compute_policy_probabilities] Error: {e}")
        return None
