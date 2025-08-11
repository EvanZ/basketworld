import os
import imageio
import mlflow
import re
from collections import defaultdict

def get_outcome_category(outcome_str: str) -> str:
    """Categorizes a detailed outcome string into a simple category for filenames."""
    if "Made 2" in outcome_str:
        return "made_2pt"
    if "Made 3" in outcome_str:
        return "made_3pt"
    if "Missed 2" in outcome_str:
        return "missed_2pt"
    if "Missed 3" in outcome_str:
        return "missed_3pt"
    if "Turnover (Pressure)" in outcome_str:
        return "tov-pressure"
    if "Turnover (Intercepted)" in outcome_str:
        return "tov-intercepted"
    if "Turnover (OOB)" in outcome_str:
        return "tov-oob"
    if "Turnover (Shot Clock Violation)" in outcome_str:
        return "tov-shotclock"
    if "Turnover" in outcome_str:
        return "turnover" # Generic fallback
    return "unknown"

def create_and_log_gif(frames, episode_num: int, outcome: str, temp_dir: str, artifact_path: str = "gifs"):
    """
    Saves frames as a GIF and logs it to a structured path in MLflow.
    The artifact_path can be customized for different evaluation contexts.
    """
    if not frames:
        return

    outcome_category = get_outcome_category(outcome)
    gif_filename = f"episode_{episode_num:03d}_{outcome_category}.gif"
    local_path = os.path.join(temp_dir, gif_filename)
    
    # Save the GIF
    imageio.mimsave(local_path, frames, fps=2, loop=0)
    
    # Log to MLflow in the categorized folder
    mlflow.log_artifact(local_path, artifact_path=artifact_path) 