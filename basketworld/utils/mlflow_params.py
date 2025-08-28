import mlflow
from typing import Any, Callable, Dict, Tuple


def _get_param(params_dict: Dict[str, str], names: list[str], cast: Callable[[Any], Any], default: Any) -> Any:
    """Try multiple parameter names, cast if found, else return default.

    Params in MLflow are stored as strings. This helper searches for the first
    available alias in names, attempts to cast it, and falls back to default if
    missing or invalid.
    """
    for name in names:
        if name in params_dict and params_dict[name] != "":
            try:
                return cast(params_dict[name])
            except Exception:
                pass
    return default


def get_mlflow_params(client: mlflow.tracking.MlflowClient, run_id: str) -> Tuple[dict, dict]:
    """Fetch and normalize environment-related params from an MLflow run.

    Returns a tuple (required, optional) where:
      - required contains required fields (grid_size, players, shot_clock)
      - optional contains all optional knobs with sensible defaults
    """
    run = client.get_run(run_id)
    params = run.data.params

    # Required
    required = {
        "grid_size": int(params["grid_size"]) if "grid_size" in params else 16,
        "players": int(params["players"]) if "players" in params else 3,
        "shot_clock": int(params["shot_clock"]) if "shot_clock" in params else 24,
    }

    # Optional
    optional = {}
    optional["three_point_distance"] = _get_param(params, [
        "three_point_distance", "three-point-distance", "three_pt_distance", "three-pt-distance"
    ], int, 4)
    optional["layup_pct"] = _get_param(params, ["layup_pct", "layup-pct"], float, 0.60)
    optional["three_pt_pct"] = _get_param(params, ["three_pt_pct", "three-pt-pct"], float, 0.37)
    optional["spawn_distance"] = _get_param(params, ["spawn_distance", "spawn-distance"], int, 3)
    optional["allow_dunks"] = _get_param(params, ["allow_dunks", "allow-dunks"], lambda v: str(v).lower() in ["1","true","yes","y","t"], False)
    optional["dunk_pct"] = _get_param(params, ["dunk_pct", "dunk-pct"], float, 0.90)
    optional["shot_pressure_enabled"] = _get_param(params, ["shot_pressure_enabled", "shot-pressure-enabled"], lambda v: str(v).lower() in ["1","true","yes","y","t"], True)
    optional["shot_pressure_max"] = _get_param(params, ["shot_pressure_max", "shot-pressure-max"], float, 0.5)
    optional["shot_pressure_lambda"] = _get_param(params, ["shot_pressure_lambda", "shot-pressure-lambda"], float, 1.0)
    optional["shot_pressure_arc_degrees"] = _get_param(params, ["shot_pressure_arc_degrees", "shot-pressure-arc-degrees"], float, 60.0)
    optional["defender_pressure_distance"] = _get_param(params, ["defender_pressure_distance", "defender-pressure-distance"], int, 1)
    optional["defender_pressure_turnover_chance"] = _get_param(params, ["defender_pressure_turnover_chance", "defender-pressure-turnover-chance"], float, 0.05)
    optional["mask_occupied_moves"] = _get_param(params, ["mask_occupied_moves", "mask-occupied-moves"], lambda v: str(v).lower() in ["1","true","yes","y","t"], False)
    optional["illegal_defense_enabled"] = _get_param(params, ["illegal_defense_enabled", "illegal-defense-enabled"], lambda v: str(v).lower() in ["1","true","yes","y","t"], False)
    optional["illegal_defense_max_steps"] = _get_param(params, ["illegal_defense_max_steps", "illegal-defense-max-steps"], int, 3)
    
    # Observation controls (optional; used by backend/main.py)
    optional["use_egocentric_obs"] = _get_param(params, ["use_egocentric_obs", "use-egocentric-obs"], lambda v: str(v).lower() in ["1","true","yes","y","t"], True)
    optional["egocentric_rotate_to_hoop"] = _get_param(params, ["egocentric_rotate_to_hoop", "egocentric-rotate-to-hoop"], lambda v: str(v).lower() in ["1","true","yes","y","t"], True)
    optional["include_hoop_vector"] = _get_param(params, ["include_hoop_vector", "include-hoop-vector"], lambda v: str(v).lower() in ["1","true","yes","y","t"], True)
    optional["normalize_obs"] = _get_param(params, ["normalize_obs", "normalize-obs"], lambda v: str(v).lower() in ["1","true","yes","y","t"], True)

    return required, optional


