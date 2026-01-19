import numpy as np
import torch
from fastapi.encoders import jsonable_encoder
from basketworld.envs.basketworld_env_v2 import ActionType, Team
from basketworld.utils.wrappers import SetObservationWrapper


class GameState:
    """Lightweight container for backend session state (single-user demo)."""

    def __init__(self):
        self.env = None
        self.offense_policy = None
        self.defense_policy = None
        self.unified_policy = None
        self.user_team: Team = None
        self.obs = None
        self.frames = []  # List of RGB frames for the current episode
        self.reward_history = []  # Track rewards for each step
        self.episode_rewards = {"offense": 0.0, "defense": 0.0}  # Running totals
        self.shot_log = []  # Per-step shot attempts with probability and result
        self.phi_log = []  # Per-step Phi diagnostics and EPs
        # Track which policies are currently loaded so we can persist logs across episodes
        self.offense_policy_key: str | None = None
        self.defense_policy_key: str | None = None
        self.unified_policy_key: str | None = None
        # Opponent unified policy (if different from unified)
        self.opponent_unified_policy_key: str | None = None
        # Self-play / replay tracking
        self.self_play_active: bool = False
        self.replay_seed: int | None = None
        self.replay_initial_positions: list[tuple[int, int]] | None = None
        self.replay_ball_holder: int | None = None
        self.replay_shot_clock: int | None = None
        self.replay_offense_skills: dict | None = None  # Store sampled skills for consistency
        self.sampled_offense_skills: dict | None = None  # Baseline skills from initial game creation
        self.actions_log: list[list[int]] = []  # full action arrays per step
        # General replay buffers (manual or AI). We store full game states for instant replay
        self.episode_states: list[dict] = []
        # MLflow run metadata
        self.run_id: str | None = None
        self.run_name: str | None = None
        # MLflow phi shaping parameters (used for Rewards tab calculations)
        # This is separate from env.phi_beta etc which can be modified in Phi Shaping tab
        self.mlflow_phi_shaping_params: dict | None = None
        # MLflow training parameters (PPO hyperparameters)
        self.mlflow_training_params: dict | None = None
        # Role flag encoding (for backward compatibility with old models)
        self.role_flag_offense: float = 1.0  # Default to new encoding
        self.role_flag_defense: float = -1.0  # Default to new encoding
        # Cache previous observation to handle race condition between move-recorded and step
        self.prev_obs: dict | None = None
        # Turn-start snapshot for frontend resets
        self.turn_start_positions: list[tuple[int, int]] | None = None
        self.turn_start_ball_holder: int | None = None
        self.turn_start_shot_clock: int | None = None
        # Parallel evaluation support - store params/paths for worker recreation
        self.env_required_params: dict | None = None
        self.env_optional_params: dict | None = None
        self.unified_policy_path: str | None = None
        self.opponent_policy_path: str | None = None


game_state = GameState()


def _role_flag_value_for_team(team: Team) -> float:
    """Resolve the role_flag value for offense vs defense with backward compatibility."""
    if team == Team.OFFENSE:
        value = getattr(game_state, "role_flag_offense", None)
        return float(value if value is not None else 1.0)
    value = getattr(game_state, "role_flag_defense", None)
    return float(value if value is not None else -1.0)


def _capture_turn_start_snapshot():
    """Store current positions/ball holder/shot clock as the baseline for the turn."""
    if not game_state.env:
        return
    env = game_state.env
    try:
        game_state.turn_start_positions = [
            (int(pos[0]), int(pos[1])) for pos in getattr(env, "positions", [])
        ]
    except Exception:
        game_state.turn_start_positions = None
    game_state.turn_start_ball_holder = (
        int(env.ball_holder) if getattr(env, "ball_holder", None) is not None else None
    )
    game_state.turn_start_shot_clock = int(getattr(env, "shot_clock", 0))


def get_full_game_state(
    include_policy_probs: bool = False,
    include_action_values: bool = False,
    include_state_values: bool = False,
):
    """Construct a JSON-friendly snapshot of the current game state."""
    if not game_state.env:
        return {}

    # Local imports to avoid circular dependencies at module load time
    from app.backend.observations import (
        compute_policy_probabilities,
        _compute_q_values_for_player,
        _compute_state_values_from_obs,
    )

    # Use FastAPI's jsonable_encoder for numpy-safe encoding
    custom_encoder = {
        np.integer: int,
        np.floating: float,
        np.bool_: bool,
    }

    last_action_results_py = jsonable_encoder(
        game_state.env.last_action_results, custom_encoder=custom_encoder
    )

    # Convert numpy types to standard Python types for JSON serialization
    positions_py = [(int(q), int(r)) for q, r in game_state.env.positions]
    ball_holder_py = (
        int(game_state.env.ball_holder)
        if game_state.env.ball_holder is not None
        else None
    )
    basket_pos_py = (
        int(game_state.env.basket_position[0]),
        int(game_state.env.basket_position[1]),
    )
    action_mask_py = game_state.obs["action_mask"].tolist()
    obs_tokens = None
    if game_state.obs:
        players_tokens = game_state.obs.get("players")
        globals_tokens = game_state.obs.get("globals")
        if players_tokens is not None or globals_tokens is not None:
            obs_tokens = {}
            if players_tokens is not None:
                obs_tokens["players"] = (
                    players_tokens.tolist()
                    if hasattr(players_tokens, "tolist")
                    else players_tokens
                )
            if globals_tokens is not None:
                obs_tokens["globals"] = (
                    globals_tokens.tolist()
                    if hasattr(globals_tokens, "tolist")
                    else globals_tokens
                )
    if obs_tokens is None and game_state.env and game_state.obs:
        try:
            wrapper = SetObservationWrapper(game_state.env)
            derived = wrapper.observation(game_state.obs)
            players_tokens = derived.get("players")
            globals_tokens = derived.get("globals")
            if players_tokens is not None or globals_tokens is not None:
                obs_tokens = {}
                if players_tokens is not None:
                    obs_tokens["players"] = (
                        players_tokens.tolist()
                        if hasattr(players_tokens, "tolist")
                        else players_tokens
                    )
                if globals_tokens is not None:
                    obs_tokens["globals"] = (
                        globals_tokens.tolist()
                        if hasattr(globals_tokens, "tolist")
                        else globals_tokens
                    )
        except Exception:
            obs_tokens = obs_tokens

    attention_payload = None
    if obs_tokens is not None and game_state.unified_policy is not None:
        try:
            policy_obj = getattr(game_state.unified_policy, "policy", None)
            extractor = getattr(policy_obj, "features_extractor", None)
            if (
                extractor is not None
                and hasattr(extractor, "token_mlp")
                and hasattr(extractor, "attn")
                and obs_tokens.get("players") is not None
                and obs_tokens.get("globals") is not None
            ):
                players_np = np.asarray(obs_tokens["players"], dtype=np.float32)
                globals_np = np.asarray(obs_tokens["globals"], dtype=np.float32)
                device = next(extractor.parameters()).device
                with torch.no_grad():
                    players_t = torch.as_tensor(players_np, device=device).unsqueeze(0)
                    globals_t = torch.as_tensor(globals_np, device=device).unsqueeze(0)
                    g = globals_t.unsqueeze(1).expand(-1, players_t.size(1), -1)
                    tokens = torch.cat([players_t, g], dim=-1)
                    emb = extractor.token_mlp(tokens)
                    cls_tokens = getattr(extractor, "cls_tokens", None)
                    if cls_tokens is not None:
                        cls = cls_tokens.unsqueeze(0).expand(emb.size(0), -1, -1)
                        emb = torch.cat([emb, cls], dim=1)
                    _, attn_weights = extractor.attn(
                        emb, emb, emb, need_weights=True, average_attn_weights=False
                    )
                    per_head = attn_weights[0].detach().cpu().numpy()
                    avg_weights = per_head.mean(axis=0).tolist()
                    per_head_weights = per_head.tolist()
                labels = []
                for pid in range(players_np.shape[0]):
                    if pid in getattr(game_state.env, "offense_ids", []):
                        labels.append(f"O{pid}")
                    elif pid in getattr(game_state.env, "defense_ids", []):
                        labels.append(f"D{pid}")
                    else:
                        labels.append(f"P{pid}")
                num_cls = int(getattr(extractor, "num_cls_tokens", 0))
                if num_cls >= 1:
                    labels.append("CLS_OFF")
                if num_cls >= 2:
                    labels.append("CLS_DEF")
                attention_payload = {
                    "weights_avg": avg_weights,
                    "weights_heads": per_head_weights,
                    "labels": labels,
                    "heads": int(getattr(extractor.attn, "num_heads", 0)),
                }
        except Exception:
            attention_payload = None

    # Calculate ball handler's pressure-adjusted shot probability for replay
    ball_handler_shot_prob = None
    if ball_holder_py is not None:
        try:
            player_pos = game_state.env.positions[ball_holder_py]
            basket_pos = game_state.env.basket_position
            distance = game_state.env._hex_distance(player_pos, basket_pos)
            ball_handler_shot_prob = float(
                game_state.env._calculate_shot_probability(ball_holder_py, distance)
            )
        except Exception:
            ball_handler_shot_prob = None

    # Calculate pass steal probabilities for replay
    pass_steal_probs = {}
    if ball_holder_py is not None:
        try:
            steal_probs = game_state.env.calculate_pass_steal_probabilities(ball_holder_py)
            pass_steal_probs = {int(k): float(v) for k, v in steal_probs.items()}
        except Exception as e:
            print(f"[get_full_game_state] Failed to calculate pass steal probabilities: {e}")
            pass_steal_probs = {}

    # Calculate EP (expected points) for all players
    ep_by_player = []
    try:
        env = game_state.env
        for pid in range(env.n_players):
            pos = env.positions[pid]
            dist = env._hex_distance(pos, env.basket_position)
            is_three = env.is_three_point_location(pos)
            if getattr(env, "allow_dunks", True) and dist == 0:
                shot_value = 2.0
            else:
                shot_value = 3.0 if is_three else 2.0
            p = float(env._calculate_shot_probability(pid, dist))
            ep = float(shot_value * p)
            ep_by_player.append(ep)
    except Exception:
        ep_by_player = []

    sampled_offense_skills = getattr(game_state, "sampled_offense_skills", None) or {}

    state = {
        "players_per_side": int(getattr(game_state.env, "players_per_side", 3)),
        "players": int(getattr(game_state.env, "players_per_side", 3)),
        "positions": positions_py,
        "ball_holder": ball_holder_py,
        "ball_handler_shot_probability": ball_handler_shot_prob,
        "pass_steal_probabilities": pass_steal_probs,
        "shot_clock": int(game_state.env.shot_clock),
        "min_shot_clock": int(getattr(game_state.env, "min_shot_clock", 10)),
        "shot_clock_steps": int(
            getattr(game_state.env, "shot_clock_steps", getattr(game_state.env, "shot_clock", 24))
        ),
        "user_team_name": game_state.user_team.name,
        "done": game_state.env.episode_ended,
        "training_team": (
            getattr(game_state.env, "training_team", None).name
            if getattr(game_state.env, "training_team", None)
            else None
        ),
        "action_space": {action.name: action.value for action in ActionType},
        "action_mask": action_mask_py,
        "obs": game_state.obs["obs"].tolist() if game_state.obs and "obs" in game_state.obs else [],
        "obs_tokens": (
            {**obs_tokens, "attention": attention_payload} if obs_tokens is not None else None
        ),
        "obs_tokens_version": 1 if obs_tokens is not None else 0,
        "last_action_results": last_action_results_py,
        "offense_ids": game_state.env.offense_ids,
        "defense_ids": game_state.env.defense_ids,
        "basket_position": basket_pos_py,
        "court_width": game_state.env.court_width,
        "court_height": game_state.env.court_height,
        "three_point_distance": float(getattr(game_state.env, "three_point_distance", 4.0)),
        "three_point_short_distance": (
            float(getattr(game_state.env, "three_point_short_distance"))
            if getattr(game_state.env, "three_point_short_distance", None) is not None
            else None
        ),
        "three_point_hexes": [
            (int(q), int(r))
            for q, r in getattr(game_state.env, "_three_point_hexes", set())
        ],
        "three_point_line_hexes": [
            (int(q), int(r))
            for q, r in getattr(game_state.env, "_three_point_line_hexes", set())
        ],
        "three_point_outline": [
            (float(x), float(y))
            for x, y in getattr(game_state.env, "_three_point_outline_points", [])
        ],
        "shot_probs": getattr(game_state.env, "shot_probs", None),
        "shot_params": {
            "layup_pct": float(getattr(game_state.env, "layup_pct", 0.0)),
            "three_pt_pct": float(getattr(game_state.env, "three_pt_pct", 0.0)),
            "dunk_pct": float(getattr(game_state.env, "dunk_pct", 0.0)),
            "layup_std": float(getattr(game_state.env, "layup_std", 0.0)),
            "three_pt_std": float(getattr(game_state.env, "three_pt_std", 0.0)),
            "dunk_std": float(getattr(game_state.env, "dunk_std", 0.0)),
            "allow_dunks": bool(getattr(game_state.env, "allow_dunks", False)),
        },
        "defender_pressure_distance": int(
            getattr(game_state.env, "defender_pressure_distance", 1)
        ),
        "defender_pressure_turnover_chance": float(
            getattr(game_state.env, "defender_pressure_turnover_chance", 0.05)
        ),
        "defender_pressure_decay_lambda": float(
            getattr(game_state.env, "defender_pressure_decay_lambda", 1.0)
        ),
        "base_steal_rate": float(getattr(game_state.env, "base_steal_rate", 0.35)),
        "steal_perp_decay": float(getattr(game_state.env, "steal_perp_decay", 1.5)),
        "steal_distance_factor": float(getattr(game_state.env, "steal_distance_factor", 0.08)),
        "steal_position_weight_min": float(getattr(game_state.env, "steal_position_weight_min", 0.3)),
        "spawn_distance": int(getattr(game_state.env, "spawn_distance", 3)),
        "max_spawn_distance": (
            int(getattr(game_state.env, "max_spawn_distance", None))
            if getattr(game_state.env, "max_spawn_distance", None) is not None
            else None
        ),
        "defender_spawn_distance": int(getattr(game_state.env, "defender_spawn_distance", 0)),
        "defender_guard_distance": int(getattr(game_state.env, "defender_guard_distance", 1)),
        "shot_pressure_enabled": bool(
            getattr(game_state.env, "shot_pressure_enabled", True)
        ),
        "shot_pressure_max": float(getattr(game_state.env, "shot_pressure_max", 0.5)),
        "shot_pressure_lambda": float(
            getattr(game_state.env, "shot_pressure_lambda", 1.0)
        ),
        "shot_pressure_arc_degrees": float(
            getattr(game_state.env, "shot_pressure_arc_degrees", 60.0)
        ),
        "mask_occupied_moves": bool(
            getattr(game_state.env, "mask_occupied_moves", False)
        ),
        "three_second_lane_width": int(
            getattr(game_state.env, "three_second_lane_width", 1)
        ),
        "three_second_lane_height": int(
            getattr(game_state.env, "three_second_lane_height", 3)
        ),
        "three_second_max_steps": int(
            getattr(game_state.env, "three_second_max_steps", 3)
        ),
        "illegal_defense_enabled": bool(
            getattr(game_state.env, "illegal_defense_enabled", False)
        ),
        "offensive_three_seconds_enabled": bool(
            getattr(game_state.env, "offensive_three_seconds_enabled", False)
        ),
        "include_hoop_vector": bool(
            getattr(game_state.env, "include_hoop_vector", False)
        ),
        "offensive_lane_hexes": [
            (int(q), int(r))
            for q, r in getattr(game_state.env, "offensive_lane_hexes", set())
        ],
        "defensive_lane_hexes": [
            (int(q), int(r))
            for q, r in getattr(game_state.env, "defensive_lane_hexes", set())
        ],
        "offensive_lane_steps": {
            int(pid): int(steps)
            for pid, steps in getattr(game_state.env, "_offensive_lane_steps", {}).items()
        },
        "defensive_lane_steps": {
            int(pid): int(steps)
            for pid, steps in getattr(game_state.env, "_defender_in_key_steps", {}).items()
        },
        "pass_arc_degrees": float(getattr(game_state.env, "pass_arc_degrees", 60.0)),
        "pass_oob_turnover_prob": float(
            getattr(game_state.env, "pass_oob_turnover_prob", 1.0)
        ),
        "pass_target_strategy": getattr(game_state.env, "pass_target_strategy", "nearest"),
        "illegal_action_policy": (
            getattr(game_state.env, "illegal_action_policy", None).value
            if getattr(game_state.env, "illegal_action_policy", None)
            else "noop"
        ),
        "pass_logit_bias": float(
            getattr(game_state.unified_policy.policy, "pass_logit_bias", 0.0)
            if game_state.unified_policy
            and hasattr(game_state.unified_policy, "policy")
            else 0.0
        ),
        "run_id": getattr(game_state, "run_id", None),
        "run_name": getattr(game_state, "run_name", None),
        "training_params": getattr(game_state, "mlflow_training_params", None),
        "unified_policy_name": getattr(game_state, "unified_policy_key", None),
        "opponent_unified_policy_name": getattr(
            game_state, "opponent_unified_policy_key", None
        ),
        "offense_shooting_pct_by_player": {
            "layup": [
                float(x)
                for x in getattr(game_state.env, "offense_layup_pct_by_player", [])
            ],
            "three_pt": [
                float(x)
                for x in getattr(game_state.env, "offense_three_pt_pct_by_player", [])
            ],
            "dunk": [
                float(x)
                for x in getattr(game_state.env, "offense_dunk_pct_by_player", [])
            ],
        },
        "offense_shooting_pct_sampled": {
            "layup": [
                float(x)
                for x in sampled_offense_skills.get("layup", [])
            ],
            "three_pt": [
                float(x)
                for x in sampled_offense_skills.get("three_pt", [])
            ],
            "dunk": [
                float(x)
                for x in sampled_offense_skills.get("dunk", [])
            ],
        },
        "ep_by_player": ep_by_player,
    }

    if include_policy_probs:
        policy_probs = compute_policy_probabilities()
        if policy_probs is not None:
            state["policy_probabilities"] = policy_probs

    if include_action_values:
        try:
            action_values_by_player = {}
            for pid in range(game_state.env.n_players):
                action_values_by_player[str(pid)] = _compute_q_values_for_player(pid, game_state)
            state["action_values"] = action_values_by_player
        except Exception as e:
            print(f"[get_full_game_state] Failed to compute action values: {e}")

    if include_state_values:
        state_values = _compute_state_values_from_obs(game_state.obs)
        if state_values:
            state["state_values"] = state_values

    return state
