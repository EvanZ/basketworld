import copy
import hashlib
import math
import time
from typing import Optional

import numpy as np
import torch
from fastapi import HTTPException
from basketworld.envs.basketworld_env_v2 import ActionType, Team
from stable_baselines3 import PPO

from basketworld.utils.action_resolution import IllegalActionStrategy

from app.backend.observations import _build_role_conditioned_obs, _predict_policy_actions
from app.backend.state import game_state


class _MCTSNode:
    def __init__(self, legal_actions: list[int], priors: Optional[np.ndarray]):
        self.legal_actions = legal_actions
        self.priors = priors if priors is not None else None
        self.children: dict[int, dict] = {}  # action -> {"visits", "value_sum", "state_key"}
        self.visits = 0
        self.value_sum = 0.0

    def add_value(self, value: float):
        self.visits += 1
        self.value_sum += float(value)


class MCTSAdvisor:
    def __init__(
        self,
        unified_policy: PPO,
        opponent_policy: Optional[PPO],
        user_team: Team,
        role_flag_offense: float,
        role_flag_defense: float,
        target_player_id: int,
        max_depth: int = 3,
        time_budget_ms: int = 200,
        exploration_c: float = 1.4,
        use_priors: bool = True,
    ):
        self.unified_policy = unified_policy
        self.opponent_policy = opponent_policy
        self.user_team = user_team
        self.role_flag_offense = role_flag_offense
        self.role_flag_defense = role_flag_defense
        self.target_player_id = target_player_id
        self.max_depth = max(1, int(max_depth))
        self.time_budget_ms = max(1, int(time_budget_ms))
        self.exploration_c = float(exploration_c)
        self.use_priors = bool(use_priors)
        self.gamma = float(getattr(unified_policy, "gamma", 0.99))

        self.nodes: dict[str, _MCTSNode] = {}
        self.nodes_expanded = 0
        self.max_depth_reached = 0
        self.cache_hits = 0

    # --- Hashing helpers (partial state) ---
    def _hash_state(self, env, obs: dict) -> str:
        positions_raw = getattr(env, "positions", None) or []
        positions = tuple((int(p[0]), int(p[1])) for p in positions_raw)
        bh_val = getattr(env, "ball_holder", -1)
        ball_holder = -1 if bh_val is None else int(bh_val)
        sc_val = getattr(env, "shot_clock", 0)
        shot_clock = 0 if sc_val is None else int(sc_val)
        offense_lane_dict = getattr(env, "_offensive_lane_steps", None) or {}
        offense_lane = tuple(sorted((int(k), int(v)) for k, v in offense_lane_dict.items()))
        defense_lane_dict = getattr(env, "_defender_in_key_steps", None) or {}
        defense_lane = tuple(sorted((int(k), int(v)) for k, v in defense_lane_dict.items()))
        skills = None
        try:
            skill_arr = obs.get("skills") if isinstance(obs, dict) else None
            if skill_arr is not None:
                skills = tuple(float(x) for x in np.asarray(skill_arr).flatten())
        except Exception:
            skills = None

        key_tuple = (positions, ball_holder, shot_clock, offense_lane, defense_lane, skills)
        return hashlib.sha1(str(key_tuple).encode("utf-8")).hexdigest()

    def _clone_env(self, env):
        try:
            return copy.deepcopy(env)
        except Exception as err:
            print(f"[MCTS] Failed to clone env with deepcopy: {err}")
            raise

    # --- Policy helpers ---
    def _team_for_player(self, env, player_id: int) -> Team:
        if player_id in getattr(env, "offense_ids", []):
            return Team.OFFENSE
        return Team.DEFENSE

    def _policy_for_player(self, player_id: int):
        player_team = self._team_for_player(game_state.env, player_id)
        if player_team == self.user_team:
            return self.unified_policy
        return self.opponent_policy or self.unified_policy

    def _build_policy_priors(self, obs: dict, env, player_id: int) -> Optional[np.ndarray]:
        policy = self._policy_for_player(player_id)
        if policy is None or not self.use_priors:
            return None
        try:
            _, probs = _predict_policy_actions(
                policy,
                obs,
                env,
                deterministic=True,
                strategy=IllegalActionStrategy.BEST_PROB,
            )
            if probs is None or player_id >= len(probs):
                return None
            return np.asarray(probs[player_id], dtype=np.float32)
        except Exception as err:
            print(f"[MCTS] Failed to build priors: {err}")
            return None

    def _estimate_value(self, obs: dict, env) -> float:
        policy = self.unified_policy
        if policy is None or not hasattr(policy, "policy"):
            return 0.0
        try:
            role_flag = (
                self.role_flag_offense
                if self.target_player_id in getattr(env, "offense_ids", [])
                else self.role_flag_defense
            )
            conditioned = _build_role_conditioned_obs(obs, role_flag)
            if conditioned is None:
                return 0.0
            obs_tensor, _ = policy.policy.obs_to_tensor(conditioned)
            with torch.no_grad():
                return float(policy.policy.predict_values(obs_tensor).item())
        except Exception as err:
            print(f"[MCTS] Value estimation failed: {err}")
            return 0.0

    def _reward_for_player(self, reward, player_id: int) -> float:
        if isinstance(reward, np.ndarray):
            reward_list = reward.tolist()
        elif isinstance(reward, (list, tuple)):
            reward_list = list(reward)
        else:
            reward_list = [reward]
        if player_id < len(reward_list):
            return float(reward_list[player_id])
        return float(reward_list[0]) if reward_list else 0.0

    def _build_action_array(self, obs: dict, env, target_action: int) -> np.ndarray:
        resolved_unified, _ = _predict_policy_actions(
            self.unified_policy,
            obs,
            env,
            deterministic=True,
            strategy=IllegalActionStrategy.BEST_PROB,
        )
        if resolved_unified is None:
            resolved_unified = np.zeros(env.n_players, dtype=int)

        resolved_opponent = None
        if self.opponent_policy is not None:
            resolved_opponent, _ = _predict_policy_actions(
                self.opponent_policy,
                obs,
                env,
                deterministic=True,
                strategy=IllegalActionStrategy.BEST_PROB,
            )

        final_action = np.zeros(env.n_players, dtype=np.int32)
        if self.user_team == Team.OFFENSE:
            for idx in env.offense_ids:
                final_action[idx] = resolved_unified[idx]
            for idx in env.defense_ids:
                final_action[idx] = resolved_opponent[idx] if resolved_opponent is not None else resolved_unified[idx]
        else:
            for idx in env.defense_ids:
                final_action[idx] = resolved_unified[idx]
            for idx in env.offense_ids:
                final_action[idx] = resolved_opponent[idx] if resolved_opponent is not None else resolved_unified[idx]

        if 0 <= self.target_player_id < env.n_players:
            final_action[self.target_player_id] = int(target_action)

        return final_action

    def _select_action(self, node: _MCTSNode) -> int:
        best_action = node.legal_actions[0]
        best_score = float("-inf")
        log_parent = math.log(max(1, node.visits)) if node.visits > 0 else 0.0
        for action in node.legal_actions:
            child_stats = node.children.get(action)
            visits = child_stats["visits"] if child_stats else 0
            value_sum = child_stats["value_sum"] if child_stats else 0.0
            prior_bonus = 0.0
            if node.priors is not None and action < len(node.priors):
                prior_bonus = float(node.priors[action])

            if visits == 0:
                score = float("inf")
            else:
                q = value_sum / visits
                score = q + self.exploration_c * math.sqrt(log_parent / visits) + prior_bonus

            if score > best_score:
                best_score = score
                best_action = action
        return best_action

    def _simulate(self, env, obs: dict, depth: int) -> float:
        state_key = self._hash_state(env, obs)
        node = self.nodes.get(state_key)
        if node is None:
            legal_mask = obs.get("action_mask") if isinstance(obs, dict) else None
            if legal_mask is None or self.target_player_id >= len(legal_mask):
                return 0.0
            player_mask = legal_mask[self.target_player_id]
            legal_actions = [int(i) for i, allowed in enumerate(player_mask) if allowed == 1]
            priors = self._build_policy_priors(obs, env, self.target_player_id)
            node = _MCTSNode(legal_actions, priors)
            self.nodes[state_key] = node
            self.nodes_expanded += 1
            leaf_value = self._estimate_value(obs, env)
            node.add_value(leaf_value)
            self.max_depth_reached = max(self.max_depth_reached, depth)
            return leaf_value

        self.cache_hits += 1

        if depth >= self.max_depth or not node.legal_actions:
            value = self._estimate_value(obs, env)
            node.add_value(value)
            self.max_depth_reached = max(self.max_depth_reached, depth)
            return value

        action = self._select_action(node)
        next_env = self._clone_env(env)
        full_action = self._build_action_array(obs, next_env, action)
        next_obs, reward, terminated, truncated, _ = next_env.step(full_action)
        done = terminated or truncated
        immediate_reward = self._reward_for_player(reward, self.target_player_id)

        if done or depth + 1 >= self.max_depth:
            rollout_value = 0.0 if done else self._estimate_value(next_obs, next_env)
        else:
            rollout_value = self._simulate(next_env, next_obs, depth + 1)

        total_value = immediate_reward + (self.gamma * rollout_value)

        child_stats = node.children.get(action)
        if child_stats is None:
            child_stats = {"visits": 0, "value_sum": 0.0, "state_key": None}
            node.children[action] = child_stats
        child_stats["visits"] += 1
        child_stats["value_sum"] += float(total_value)
        child_stats["state_key"] = self._hash_state(next_env, next_obs)

        node.add_value(total_value)
        self.max_depth_reached = max(self.max_depth_reached, depth)
        return total_value

    def advise(self, env, obs: dict) -> dict:
        if env is None or obs is None:
            raise HTTPException(status_code=400, detail="Game not initialized")

        root_key = self._hash_state(env, obs)
        start = time.perf_counter()
        iterations = 0

        while (time.perf_counter() - start) * 1000.0 < self.time_budget_ms:
            sim_env = self._clone_env(env)
            sim_obs = copy.deepcopy(obs)
            self._simulate(sim_env, sim_obs, depth=0)
            iterations += 1

        root_node = self.nodes.get(root_key)
        if root_node is None or not root_node.legal_actions:
            raise HTTPException(status_code=500, detail="MCTS failed to build root node")

        action_visits = {}
        for act in root_node.legal_actions:
            child = root_node.children.get(act)
            action_visits[act] = child["visits"] if child else 0

        best_action = max(action_visits.items(), key=lambda kv: kv[1])[0]
        total_child_visits = sum(action_visits.values())
        if total_child_visits > 0:
            policy = [action_visits.get(i, 0) / total_child_visits for i in range(len(ActionType))]
        else:
            policy = [0.0 for _ in range(len(ActionType))]

        best_stats = root_node.children.get(best_action)
        q_estimate = None
        if best_stats and best_stats["visits"] > 0:
            q_estimate = best_stats["value_sum"] / best_stats["visits"]

        duration_ms = (time.perf_counter() - start) * 1000.0
        return {
            "action": int(best_action),
            "policy": policy,
            "q_estimate": q_estimate,
            "visits": action_visits,
            "nodes_expanded": self.nodes_expanded,
            "iterations": iterations,
            "duration_ms": duration_ms,
            "nodes_per_sec": (iterations / (duration_ms / 1000.0)) if duration_ms > 0 else None,
            "max_depth_reached": self.max_depth_reached,
            "cache_hits": self.cache_hits,
        }


def _run_mcts_advisor(
    player_id: Optional[int],
    obs: dict,
    env,
    max_depth: Optional[int] = None,
    time_budget_ms: Optional[int] = None,
    exploration_c: Optional[float] = None,
    use_priors: Optional[bool] = True,
):
    if env is None:
        raise HTTPException(status_code=400, detail="Game not initialized")

    target_player = player_id if player_id is not None else getattr(env, "ball_holder", 0)
    if target_player is None:
        target_player = 0

    advisor = MCTSAdvisor(
        unified_policy=game_state.unified_policy,
        opponent_policy=game_state.defense_policy,
        user_team=game_state.user_team,
        role_flag_offense=game_state.role_flag_offense,
        role_flag_defense=game_state.role_flag_defense,
        target_player_id=int(target_player),
        max_depth=max_depth or 3,
        time_budget_ms=time_budget_ms or 200,
        exploration_c=exploration_c if exploration_c is not None else 1.4,
        use_priors=True if use_priors is None else bool(use_priors),
    )
    return advisor.advise(env, obs)
