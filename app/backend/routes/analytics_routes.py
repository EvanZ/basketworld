from fastapi import APIRouter, HTTPException

from app.backend.state import game_state, get_full_game_state
from app.backend.observations import calculate_phi_from_ep_data


router = APIRouter()


@router.get("/api/shot_stats")
def get_shot_stats():
    """Return raw shot log and simple aggregates to compare displayed probabilities vs outcomes."""
    logs = list(game_state.shot_log)
    total = len(logs)
    made = sum(1 for s in logs if s.get("success"))
    avg_prob = (sum(s.get("probability", 0.0) for s in logs) / total) if total else 0.0
    avg_base = sum(
        s.get("base_probability", 0.0)
        for s in logs
        if s.get("base_probability", -1.0) >= 0
    ) / max(1, sum(1 for s in logs if s.get("base_probability", -1.0) >= 0))
    avg_pressure_mult = sum(
        s.get("pressure_multiplier", 0.0)
        for s in logs
        if s.get("pressure_multiplier", -1.0) >= 0
    ) / max(1, sum(1 for s in logs if s.get("pressure_multiplier", -1.0) >= 0))
    total_three = sum(1 for s in logs if s.get("is_three"))
    made_three = sum(1 for s in logs if s.get("is_three") and s.get("success"))
    avg_prob_three = (
        sum(s.get("probability", 0.0) for s in logs if s.get("is_three")) / total_three
        if total_three
        else 0.0
    )
    avg_base_three = sum(
        s.get("base_probability", 0.0)
        for s in logs
        if s.get("is_three") and s.get("base_probability", -1.0) >= 0
    ) / max(
        1,
        sum(1 for s in logs if s.get("is_three") and s.get("base_probability", -1.0) >= 0),
    )
    avg_pressure_three = sum(
        s.get("pressure_multiplier", 0.0)
        for s in logs
        if s.get("is_three") and s.get("pressure_multiplier", -1.0) >= 0
    ) / max(
        1,
        sum(1 for s in logs if s.get("is_three") and s.get("pressure_multiplier", -1.0) >= 0),
    )
    by_distance = {}
    for s in logs:
        dist = s.get("distance", -1)
        try:
            d = int(dist)
        except Exception:
            d = -1
        if d not in by_distance:
            by_distance[d] = {
                "attempts": 0,
                "made": 0,
                "avg_prob": 0.0,
                "_prob_sum": 0.0,
            }
        by_distance[d]["attempts"] += 1
        by_distance[d]["made"] += 1 if s.get("success") else 0
        by_distance[d]["_prob_sum"] += float(s.get("probability", 0.0))
    for d, agg in by_distance.items():
        attempts = max(1, agg["attempts"])
        agg["avg_prob"] = agg["_prob_sum"] / attempts
        del agg["_prob_sum"]
    return {
        "total_attempts": total,
        "made": made,
        "make_rate": (made / total) if total else 0.0,
        "avg_prob": avg_prob,
        "avg_base_probability": avg_base,
        "avg_pressure_multiplier": avg_pressure_mult,
        "three_point": {
            "attempts": total_three,
            "made": made_three,
            "make_rate": (made_three / total_three) if total_three else 0.0,
            "avg_prob": avg_prob_three,
            "avg_base_probability": avg_base_three,
            "avg_pressure_multiplier": avg_pressure_three,
        },
        "by_distance": by_distance,
        "log": logs[-100:],
    }


@router.post("/api/replay_last_episode")
def replay_last_episode():
    """Return the recorded sequence of states for the last episode (manual or self-play)."""
    if not game_state.env:
        raise HTTPException(status_code=400, detail="Game not initialized.")

    if getattr(game_state, "episode_states", None) and len(game_state.episode_states) > 0:
        return {"status": "success", "states": list(game_state.episode_states)}

    if (
        game_state.replay_seed is None
        or game_state.replay_initial_positions is None
        or game_state.actions_log is None
    ):
        raise HTTPException(status_code=400, detail="No episode available to replay.")

    options = {
        "initial_positions": game_state.replay_initial_positions,
        "ball_holder": game_state.replay_ball_holder,
        "shot_clock": game_state.replay_shot_clock,
        "offense_skills": game_state.replay_offense_skills,
    }
    obs, _ = game_state.env.reset(seed=game_state.replay_seed, options=options)

    states = [get_full_game_state(include_state_values=True)]
    for action in game_state.actions_log:
        obs, _, _, _, _ = game_state.env.step(action)
        try:
            frame = game_state.env.render()
            if frame is not None:
                game_state.frames.append(frame)
        except Exception:
            pass
        states.append(get_full_game_state(include_state_values=True))

    game_state.obs = obs
    return {"status": "success", "states": states}


@router.get("/api/pass_steal_probabilities")
def get_pass_steal_probabilities():
    """Get steal probabilities for passes from ball handler to each teammate."""
    if game_state.env is None:
        raise HTTPException(status_code=400, detail="Game not initialized")

    if game_state.env.ball_holder is None:
        return {}

    try:
        steal_probs = game_state.env.calculate_pass_steal_probabilities(game_state.env.ball_holder)
        return {int(k): float(v) for k, v in steal_probs.items()}
    except Exception as e:
        print(f"[ERROR] Failed to calculate pass steal probabilities: {e}")
        import traceback

        traceback.print_exc()
        return {}


@router.get("/api/rewards")
def get_rewards():
    """Get the current reward history and episode totals."""
    import sys

    print("=" * 80, flush=True)
    sys.stdout.flush()

    mlflow_phi_params = game_state.mlflow_phi_shaping_params
    mlflow_phi_r_shape_values = []
    if mlflow_phi_params and mlflow_phi_params.get("enable_phi_shaping", False):
        beta = mlflow_phi_params.get("phi_beta", 0.0)
        gamma = mlflow_phi_params.get("reward_shaping_gamma", 1.0)

        phi_prev = 0.0
        if game_state.phi_log and len(game_state.phi_log) > 0:
            initial_entry = game_state.phi_log[0]
            if initial_entry.get("step") == 0:
                initial_ep = initial_entry.get("ep_by_player", [])
                initial_ball = initial_entry.get("ball_handler", -1)
                initial_offense = initial_entry.get("offense_ids", [])
                if initial_ep and initial_ball >= 0 and initial_offense:
                    phi_prev = calculate_phi_from_ep_data(
                        initial_ep, initial_ball, initial_offense, mlflow_phi_params
                    )

        for i, reward in enumerate(game_state.reward_history):
            ep_by_player = reward.get("ep_by_player", [])
            ball_handler = reward.get("ball_handler", -1)
            offense_ids = reward.get("offense_ids", [])
            is_terminal = reward.get("is_terminal", False)

            phi_next = 0.0
            if not is_terminal and ep_by_player:
                phi_next = calculate_phi_from_ep_data(
                    ep_by_player, ball_handler, offense_ids, mlflow_phi_params
                )
            elif not is_terminal and not ep_by_player:
                print(f"[MLflow Phi] WARNING: No EP data for step {i+1}, cannot calculate phi")

            r_shape = beta * (gamma * phi_next - phi_prev)
            mlflow_phi_r_shape_values.append(r_shape)
            phi_prev = phi_next
    else:
        mlflow_phi_r_shape_values = [0.0] * len(game_state.reward_history)

    mlflow_phi_potential_values = []
    if mlflow_phi_params and mlflow_phi_params.get("enable_phi_shaping", False):
        for reward in game_state.reward_history:
            ep_by_player = reward.get("ep_by_player", [])
            ball_handler = reward.get("ball_handler", -1)
            offense_ids = reward.get("offense_ids", [])
            is_terminal = reward.get("is_terminal", False)

            phi_potential = 0.0
            if not is_terminal and ep_by_player:
                phi_potential = calculate_phi_from_ep_data(
                    ep_by_player, ball_handler, offense_ids, mlflow_phi_params
                )
            mlflow_phi_potential_values.append(phi_potential)
    else:
        mlflow_phi_potential_values = [0.0] * len(game_state.reward_history)

    serialized_history = []

    if mlflow_phi_params and mlflow_phi_params.get("enable_phi_shaping", False):
        initial_phi = 0.0
        if game_state.phi_log and len(game_state.phi_log) > 0:
            initial_entry = game_state.phi_log[0]
            if initial_entry.get("step") == 0:
                initial_ep = initial_entry.get("ep_by_player", [])
                initial_ball = initial_entry.get("ball_handler", -1)
                initial_offense = initial_entry.get("offense_ids", [])
                if initial_ep and initial_ball >= 0 and initial_offense:
                    initial_phi = calculate_phi_from_ep_data(
                        initial_ep, initial_ball, initial_offense, mlflow_phi_params
                    )

        serialized_history.append(
            {
                "step": 0,
                "shot_clock": 24,
                "offense": 0.0,
                "defense": 0.0,
                "offense_reason": "Initial State",
                "defense_reason": "Initial State",
                "mlflow_phi_potential": float(initial_phi),
            }
        )

    for i, reward in enumerate(game_state.reward_history):
        mlflow_phi_r_shape = (
            mlflow_phi_r_shape_values[i] if i < len(mlflow_phi_r_shape_values) else 0.0
        )
        mlflow_phi_potential = (
            mlflow_phi_potential_values[i]
            if i < len(mlflow_phi_potential_values)
            else 0.0
        )

        env_phi_r_shape_per_player = reward.get("phi_r_shape", 0.0)
        offense_ids = reward.get("offense_ids", [])
        num_offensive_players = len(offense_ids) if offense_ids else 3
        env_phi_r_shape_total = env_phi_r_shape_per_player * num_offensive_players

        base_offense = float(reward["offense"]) - env_phi_r_shape_total
        base_defense = float(reward["defense"]) + env_phi_r_shape_total

        offense_with_mlflow = base_offense + mlflow_phi_r_shape
        defense_with_mlflow = base_defense - mlflow_phi_r_shape

        serialized_history.append(
            {
                "step": int(reward["step"]),
                "shot_clock": int(reward.get("shot_clock", 0)),
                "offense": float(offense_with_mlflow),
                "defense": float(defense_with_mlflow),
                "offense_reason": reward.get("offense_reason", "Unknown"),
                "defense_reason": reward.get("defense_reason", "Unknown"),
                "mlflow_phi_potential": float(mlflow_phi_potential),
            }
        )

    env = game_state.env
    reward_params = {}
    try:
        reward_params = {
            "pass_reward": float(getattr(env, "pass_reward", 0.0)),
            "turnover_reward": 0.0,
            "shot_reward_type": "expected_points",
            "shot_reward_description": "Reward = shot_value × pressure-adjusted make probability (applies to makes and misses)",
            "violation_reward": float(getattr(env, "violation_reward", 0.0)),
            "potential_assist_pct": float(getattr(env, "potential_assist_pct", 0.0)),
            "full_assist_bonus_pct": float(getattr(env, "full_assist_bonus_pct", 0.0)),
            "assist_window": int(getattr(env, "assist_window", 2)),
        }
    except Exception:
        reward_params = {}

    mlflow_phi_params_serialized = {}
    if mlflow_phi_params:
        try:
            mlflow_phi_params_serialized = {
                "enable_phi_shaping": bool(mlflow_phi_params.get("enable_phi_shaping", False)),
                "phi_beta": float(mlflow_phi_params.get("phi_beta", 0.0)),
                "reward_shaping_gamma": float(mlflow_phi_params.get("reward_shaping_gamma", 1.0)),
                "phi_aggregation_mode": str(mlflow_phi_params.get("phi_aggregation_mode", "team_best")),
                "phi_use_ball_handler_only": bool(
                    mlflow_phi_params.get("phi_use_ball_handler_only", False)
                ),
                "phi_blend_weight": float(mlflow_phi_params.get("phi_blend_weight", 0.0)),
            }
        except Exception:
            pass

    return {
        "reward_history": serialized_history,
        "episode_rewards": {
            "offense": float(game_state.episode_rewards["offense"]),
            "defense": float(game_state.episode_rewards["defense"]),
        },
        "reward_params": reward_params,
        "mlflow_phi_params": mlflow_phi_params_serialized,
    }

