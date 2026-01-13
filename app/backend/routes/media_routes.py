import os
from datetime import datetime
from typing import List

import imageio
import numpy as np
from fastapi import APIRouter, HTTPException

from basketworld.utils.evaluation_helpers import get_outcome_category
from app.backend.schemas import SaveEpisodeRequest
from app.backend.state import game_state


router = APIRouter()


@router.get("/api/debug/frames")
def debug_frames():
    """Debug endpoint to check frame capture status."""
    return {
        "frames_count": len(game_state.frames) if game_state.frames else 0,
        "env_exists": game_state.env is not None,
        "render_mode": getattr(game_state.env, "render_mode", None) if game_state.env else None,
        "has_offensive_lane_hexes": hasattr(game_state.env, "offensive_lane_hexes") if game_state.env else False,
    }


@router.post("/api/save_episode")
def save_episode():
    """Saves the recorded episode frames to a GIF in ./episodes and returns the file path."""
    print(f"[SAVE_EPISODE] Frames count: {len(game_state.frames)}")
    print(f"[SAVE_EPISODE] Env exists: {game_state.env is not None}")

    if not game_state.frames:
        raise HTTPException(
            status_code=400,
            detail=f"No episode frames to save. Frames list is empty (length: {len(game_state.frames) if game_state.frames else 0}).",
        )

    base_dir = "episodes"
    if getattr(game_state, "run_id", None):
        base_dir = os.path.join(base_dir, str(game_state.run_id))
    os.makedirs(base_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    outcome = "Unknown"
    category = None
    try:
        ar = game_state.env.last_action_results or {}
        if ar.get("shots"):
            shooter_id_str = list(ar["shots"].keys())[0]
            shot_res = ar["shots"][shooter_id_str]
            distance = int(shot_res.get("distance", 999))
            is_dunk = distance == 0
            is_three = bool(
                shot_res.get("is_three") if "is_three" in shot_res else distance >= game_state.env.three_point_distance
            )
            success = bool(shot_res.get("success"))
            assist_full = bool(shot_res.get("assist_full", False))
            assist_potential = bool(shot_res.get("assist_potential", False))

            shot_type = "dunk" if is_dunk else ("3pt" if is_three else "2pt")
            if success:
                outcome = f"Made {shot_type.upper()}" if shot_type != "dunk" else "Made Dunk"
                category = f"made_assisted_{shot_type}" if assist_full else f"made_unassisted_{shot_type}"
            else:
                outcome = f"Missed {shot_type.upper()}" if shot_type != "dunk" else "Missed Dunk"
                if assist_potential:
                    category = f"missed_potentially_assisted_{shot_type}"
                else:
                    category = f"missed_{shot_type}"
        elif ar.get("turnovers"):
            reason = ar["turnovers"][0].get("reason", "turnover")
            if reason == "intercepted":
                outcome = "Turnover (Intercepted)"
            elif reason in ("pass_out_of_bounds", "move_out_of_bounds"):
                outcome = "Turnover (OOB)"
            elif reason == "defender_pressure":
                outcome = "Turnover (Pressure)"
            else:
                outcome = f"Turnover ({reason})"
        elif getattr(game_state.env, "shot_clock", 1) <= 0:
            outcome = "Turnover (Shot Clock Violation)"
    except Exception:
        pass

    if category is None:
        category = get_outcome_category(outcome)
    file_path = os.path.join(base_dir, f"episode_{timestamp}_{category}.gif")

    try:
        valid_frames = [f for f in game_state.frames if f is not None]
        if not valid_frames:
            raise HTTPException(status_code=400, detail="No valid frames to save.")
        frames_to_save: List[np.ndarray] = []
        for f in valid_frames:
            try:
                frames_to_save.append(np.array(f, copy=True))
            except Exception:
                frames_to_save.append(f)
        imageio.mimsave(file_path, frames_to_save, fps=1, loop=0)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save GIF: {e}")

    game_state.frames = []
    return {"status": "success", "file_path": file_path}


@router.post("/api/save_episode_from_pngs")
def save_episode_from_pngs(request: SaveEpisodeRequest):
    """Saves episode from base64-encoded PNG frames sent from frontend."""
    import base64
    from PIL import Image
    import io

    if not request.frames or len(request.frames) == 0:
        raise HTTPException(status_code=400, detail="No frames provided")

    base_dir = "episodes"
    if getattr(game_state, "run_id", None):
        base_dir = os.path.join(base_dir, str(game_state.run_id))
    os.makedirs(base_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    outcome = "Unknown"
    category = None
    try:
        ar = game_state.env.last_action_results or {}
        if ar.get("shots"):
            shooter_id_str = list(ar["shots"].keys())[0]
            shot_res = ar["shots"][shooter_id_str]
            distance = int(shot_res.get("distance", 999))
            is_dunk = distance == 0
            is_three = bool(
                shot_res.get("is_three") if "is_three" in shot_res else distance >= game_state.env.three_point_distance
            )
            success = bool(shot_res.get("success"))
            assist_full = bool(shot_res.get("assist_full", False))
            assist_potential = bool(shot_res.get("assist_potential", False))

            shot_type = "dunk" if is_dunk else ("3pt" if is_three else "2pt")
            if success:
                outcome = f"Made {shot_type.upper()}" if shot_type != "dunk" else "Made Dunk"
                category = f"made_assisted_{shot_type}" if assist_full else f"made_unassisted_{shot_type}"
            else:
                outcome = f"Missed {shot_type.upper()}" if shot_type != "dunk" else "Missed Dunk"
                if assist_potential:
                    category = f"missed_potentially_assisted_{shot_type}"
                else:
                    category = f"missed_{shot_type}"
        elif ar.get("turnovers"):
            reason = ar["turnovers"][0].get("reason", "turnover")
            if reason == "intercepted":
                outcome = "Turnover (Intercepted)"
            elif reason in ("pass_out_of_bounds", "move_out_of_bounds"):
                outcome = "Turnover (OOB)"
            elif reason == "defender_pressure":
                outcome = "Turnover (Pressure)"
            else:
                outcome = f"Turnover ({reason})"
        elif getattr(game_state.env, "shot_clock", 1) <= 0:
            outcome = "Turnover (Shot Clock Violation)"
    except Exception:
        pass

    if category is None:
        category = get_outcome_category(outcome)

    file_path = os.path.join(base_dir, f"episode_{timestamp}_{category}.gif")

    try:
        pil_frames = []
        for base64_frame in request.frames:
            if "," in base64_frame:
                base64_frame = base64_frame.split(",")[1]
            img_bytes = base64.b64decode(base64_frame)
            img = Image.open(io.BytesIO(img_bytes))
            if img.mode == "RGBA":
                background = Image.new("RGB", img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])
                img = background
            elif img.mode != "RGB":
                img = img.convert("RGB")
            pil_frames.append(img)

        durations_sec = None
        if request.durations and len(request.durations) > 0:
            durations_sec = [max(0.01, float(d)) for d in request.durations]
        elif request.step_duration_ms:
            dur = max(10.0, float(request.step_duration_ms))
            durations_sec = [dur / 1000.0] * len(pil_frames)
        else:
            durations_sec = [1.0] * len(pil_frames)

        if len(durations_sec) != len(pil_frames):
            if len(durations_sec) < len(pil_frames):
                last_d = durations_sec[-1] if durations_sec else 1.0
                durations_sec.extend([last_d] * (len(pil_frames) - len(durations_sec)))
            durations_sec = durations_sec[: len(pil_frames)]

        durations_ms = [max(10, int(round(d * 1000))) for d in durations_sec]

        if not pil_frames:
            raise HTTPException(status_code=400, detail="No frames provided after decoding")

        pil_frames[0].save(
            file_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=durations_ms,
            loop=0,
            optimize=False,
            disposal=2,
        )

        return {"status": "success", "file_path": file_path}
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to save GIF from PNGs: {e}")

