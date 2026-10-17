from __future__ import annotations

from typing import Any

from .state import GameState, game_state


def validate_policy_observation_schema(
    policy: Any,
    env: Any,
    obs: dict | None,
    policy_label: str = "policy",
) -> dict | None:
    """Validate the application-side observation contract for JAX inference."""
    del env
    if policy is None or obs is None:
        return obs
    if not isinstance(obs, dict):
        raise ValueError(
            f"{policy_label}: expected a JAX observation dictionary, "
            f"received {type(obs).__name__}"
        )
    missing = sorted({"obs", "action_mask"} - set(obs))
    if missing:
        raise ValueError(f"{policy_label}: observation keys missing={missing}")
    return obs


def calculate_phi_from_ep_data(
    ep_by_player: list[float],
    ball_handler_id: int,
    offense_ids: list[int],
    phi_params: dict,
) -> float:
    """Calculate phi from expected-points data using the configured aggregation."""
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
        ep_by_player[pid]
        for pid in offense_ids
        if pid != ball_handler_id and pid < len(ep_by_player)
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
    else:
        teammate_aggregate = max(max(teammate_eps), ball_ep)
    weight = max(0.0, min(1.0, float(blend_weight)))
    return (1.0 - weight) * teammate_aggregate + weight * ball_ep


def _compute_state_values_from_obs(obs_dict: dict | None):
    """Return JAX value estimates for the live runtime."""
    if obs_dict is None or game_state.unified_policy is None:
        return None
    runtime = game_state.jax_runtime
    if runtime is not None:
        try:
            return runtime.state_values()
        except Exception as err:
            print(f"[STATE_VALUES] Failed to compute JAX state values: {err}")
            return None
    value_fn = getattr(game_state.unified_policy, "state_value", None)
    if not callable(value_fn) or game_state.env is None:
        return None
    try:
        return {
            "offensive_value": float(value_fn(game_state.env, observer_is_offense=True)),
            "defensive_value": float(value_fn(game_state.env, observer_is_offense=False)),
        }
    except Exception as err:
        print(f"[STATE_VALUES] Failed to compute JAX adapter state values: {err}")
        return None


def _compute_q_values_for_player(player_id: int, state: GameState) -> dict:
    """Q-value counterfactuals are not exposed by the JAX inference contract."""
    del player_id, state
    return {}
