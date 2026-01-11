import numpy as np
import mlflow


def log_vecenv_profile_stats(vecenv, prefix: str, step: int) -> None:
    """Aggregate and log profiling stats from a VecEnv."""
    try:
        print(f"\nCollecting {prefix} profiling stats...")
        stats_list = vecenv.env_method("get_profile_stats")
        aggregated = {}
        for env_idx, stats in enumerate(stats_list):
            for section_name, metrics in stats.items():
                if section_name not in aggregated:
                    aggregated[section_name] = {
                        "total_ms": 0.0,
                        "total_calls": 0,
                        "avg_us_list": [],
                    }
                aggregated[section_name]["total_ms"] += metrics["total_ms"]
                aggregated[section_name]["total_calls"] += int(metrics["calls"])
                aggregated[section_name]["avg_us_list"].append(metrics["avg_us"])

        for section_name, metrics in aggregated.items():
            mean_avg_us = np.mean(metrics["avg_us_list"])
            mlflow.log_metric(f"{prefix}_{section_name}_avg_us", mean_avg_us, step=step)
            mlflow.log_metric(
                f"{prefix}_{section_name}_total_ms", metrics["total_ms"], step=step
            )
            mlflow.log_metric(
                f"{prefix}_{section_name}_total_calls",
                metrics["total_calls"],
                step=step,
            )
        print(f"Logged profiling stats for {len(aggregated)} sections")
        vecenv.env_method("reset_profile_stats")
    except Exception as e:
        print(f"Warning: Could not collect profiling stats: {e}")
