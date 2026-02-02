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
from basketworld.utils.wrappers import SetObservationWrapper

from .state import GameState, _role_flag_value_for_team, game_state


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
    if "players" in obs:
        cloned["players"] = np.copy(obs["players"])
    if "globals" in obs:
        cloned["globals"] = np.copy(obs["globals"])
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


def _team_player_ids(env, team: Team) -> List[int]:
    if team == Team.OFFENSE:
        return list(getattr(env, "offense_ids", []))
    return list(getattr(env, "defense_ids", []))


def _predict_actions_for_team(
    policy: PPO,
    base_obs: Dict,
    env,
    team: Team,
    deterministic: bool,
    strategy: IllegalActionStrategy,
) -> Tuple[Dict[int, int], Dict[int, np.ndarray]]:
    actions_by_player: Dict[int, int] = {}
    probs_by_player: Dict[int, np.ndarray] = {}

    if policy is None or base_obs is None or env is None:
        return actions_by_player, probs_by_player

    base_obs = _ensure_set_obs(policy, env, base_obs)
    team_ids = _team_player_ids(env, team)
    if not team_ids:
        return actions_by_player, probs_by_player

    role_flag_value = _role_flag_value_for_team(team)
    conditioned_obs = _clone_obs_with_role_flag(base_obs, role_flag_value)

    raw_actions = None
    try:
        raw_actions, _ = policy.predict(conditioned_obs, deterministic=deterministic)
    except Exception:
        raw_actions = None

    if raw_actions is None:
        raw_actions = np.zeros(len(team_ids), dtype=int)
    raw_actions = np.array(raw_actions).reshape(-1)
    action_len = raw_actions.shape[0]
    team_mask = base_obs["action_mask"][team_ids]

    # Legacy policies output actions for every player; new policies output players_per_side only.
    if action_len == len(team_ids):
        team_pred_actions = raw_actions
    elif action_len == getattr(env, "n_players", action_len):
        team_pred_actions = raw_actions[team_ids]
    else:
        # Fallback: truncate/pad to team size
        team_pred_actions = raw_actions[: len(team_ids)]

    probs = get_policy_action_probabilities(policy, conditioned_obs)
    if probs is not None:
        probs = [np.asarray(p, dtype=np.float32) for p in probs]
        if len(probs) == getattr(env, "n_players", len(probs)):
            team_probs = [probs[int(pid)] for pid in team_ids]
        else:
            team_probs = probs[: len(team_ids)]
    else:
        team_probs = None

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
) -> Tuple[np.ndarray | None, List[np.ndarray] | None]:
    if policy is None or base_obs is None or env is None:
        return None, None

    num_players = env.n_players
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
    if "players" in base_obs:
        conditioned["players"] = np.copy(base_obs["players"])
    if "globals" in base_obs:
        conditioned["globals"] = np.copy(base_obs["globals"])
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

        offense_tensor, _ = value_policy.policy.obs_to_tensor(offense_obs)
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

    for action_name in possible_actions:
        action_id = ActionType[action_name].value
        temp_env = copy.deepcopy(state.env)
        sim_action = np.zeros(temp_env.n_players, dtype=int)

        full_actions_main, _ = _predict_policy_actions(
            state.unified_policy,
            state.obs,
            state.env,
            deterministic=True,
            strategy=IllegalActionStrategy.BEST_PROB,
        )
        if full_actions_main is None:
            full_actions_main = np.zeros(temp_env.n_players, dtype=int)

        full_actions_opponent, _ = _predict_policy_actions(
            state.defense_policy,
            state.obs,
            state.env,
            deterministic=True,
            strategy=IllegalActionStrategy.BEST_PROB,
        )

        is_player_on_user_team = (
            (player_id in state.env.offense_ids and state.user_team == Team.OFFENSE)
            or (player_id in state.env.defense_ids and state.user_team == Team.DEFENSE)
        )

        for i in range(temp_env.n_players):
            if i == player_id:
                sim_action[i] = action_id
            else:
                is_i_on_user_team = (
                    (i in state.env.offense_ids and state.user_team == Team.OFFENSE)
                    or (i in state.env.defense_ids and state.user_team == Team.DEFENSE)
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
        role_flag_value = state.role_flag_offense if player_id in state.env.offense_ids else state.role_flag_defense
        conditioned_next_obs = {
            "obs": np.copy(next_obs["obs"]),
            "action_mask": next_obs["action_mask"],
            "role_flag": np.array([role_flag_value], dtype=np.float32),
            "skills": np.copy(next_obs.get("skills")) if next_obs.get("skills") is not None else None,
        }
        if "players" in next_obs:
            conditioned_next_obs["players"] = np.copy(next_obs["players"])
        if "globals" in next_obs:
            conditioned_next_obs["globals"] = np.copy(next_obs["globals"])

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

        for pid in range(env.n_players):
            is_user_team = (
                (pid in env.offense_ids and game_state.user_team == Team.OFFENSE)
                or (pid in env.defense_ids and game_state.user_team == Team.DEFENSE)
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

        for pid in range(game_state.env.n_players):
            is_user_team = (
                (pid in game_state.env.offense_ids and game_state.user_team == Team.OFFENSE)
                or (pid in game_state.env.defense_ids and game_state.user_team == Team.DEFENSE)
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
