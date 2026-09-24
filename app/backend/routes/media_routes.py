import os
import io
import json
from datetime import datetime
from typing import List

import imageio
import numpy as np
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from basketworld.utils.evaluation_helpers import get_outcome_category
from app.backend.schemas import SaveEpisodeRequest, EpisodeExportFrameRequest, EpisodeExportFinishRequest
from app.backend.episode_gif import EpisodeGifExports, quantize_gif_frame
from app.backend.state import game_state


router = APIRouter()
episode_exports = EpisodeGifExports()
EPISODE_EXPORT_BATCH_MAX_FRAMES = 8
EPISODE_EXPORT_BATCH_MAX_BYTES = 64 * 1024 * 1024


def _is_public_mode() -> bool:
    return str(os.getenv("BW_PUBLIC_MODE", "")).strip().lower() in {"1", "true", "yes", "on"}


def _decode_png_frames_and_durations(request: SaveEpisodeRequest):
    import base64
    from PIL import Image

    if not request.frames or len(request.frames) == 0:
        raise HTTPException(status_code=400, detail="No frames provided")

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

    # A browser resize can change the captured dimensions. GIF's logical canvas
    # is taken from its first frame: pad so later frames cannot be clipped.
    canvas_size = (
        max(frame.width for frame in pil_frames),
        max(frame.height for frame in pil_frames),
    )
    for index, frame in enumerate(pil_frames):
        if frame.size != canvas_size:
            padded = Image.new("RGB", canvas_size, (10, 15, 30))
            padded.paste(frame, (0, 0))
            frame.close()
            pil_frames[index] = padded

    # Short board animations and full episode exports use the same color
    # conversion; do not let Pillow fall back to undithered median-cut on save.
    for index, frame in enumerate(pil_frames):
        pil_frames[index] = quantize_gif_frame(frame)
        frame.close()

    return pil_frames, durations_ms


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
    if _is_public_mode():
        raise HTTPException(status_code=403, detail="Saving full episodes is disabled in public mode.")

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


def _episode_png_output_path():
    if _is_public_mode():
        raise HTTPException(status_code=403, detail="Saving full episodes is disabled in public mode.")

    base_dir = "episodes"
    if getattr(game_state, "run_id", None):
        base_dir = os.path.join(base_dir, str(game_state.run_id))
    os.makedirs(base_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

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

    return os.path.join(base_dir, f"episode_{timestamp}_{category}.gif")


@router.post("/api/save_episode_from_pngs")
def save_episode_from_pngs(request: SaveEpisodeRequest):
    """Legacy bulk upload; the UI uses bounded episode_exports uploads."""
    file_path = _episode_png_output_path()

    try:
        pil_frames, durations_ms = _decode_png_frames_and_durations(request)

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


@router.post("/api/render_gif_from_pngs")
def render_gif_from_pngs(request: SaveEpisodeRequest):
    """Render a GIF from base64-encoded PNG frames and return bytes without persisting."""
    try:
        pil_frames, durations_ms = _decode_png_frames_and_durations(request)
        buffer = io.BytesIO()
        pil_frames[0].save(
            buffer,
            format="GIF",
            save_all=True,
            append_images=pil_frames[1:],
            duration=durations_ms,
            loop=0,
            optimize=False,
            disposal=2,
        )
        return Response(content=buffer.getvalue(), media_type="image/gif")
    except HTTPException:
        raise
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to render GIF from PNGs: {e}")


def _get_episode_export(export_id):
    if _is_public_mode():
        raise HTTPException(status_code=403, detail="Saving full episodes is disabled in public mode.")
    try:
        return episode_exports.get(export_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


def _decode_episode_export_batch(body: bytes):
    """Decode a compact [metadata length][JSON][PNG bytes...] upload.

    Keeping this as a binary request avoids base64 expansion and does not add
    a multipart parser dependency to the local development backend. The
    browser only holds one small batch at a time.
    """
    if len(body) < 4:
        raise ValueError("GIF frame batch is missing metadata")
    if len(body) > EPISODE_EXPORT_BATCH_MAX_BYTES:
        raise ValueError("GIF frame batch exceeds the 64 MiB limit")
    metadata_size = int.from_bytes(body[:4], byteorder="big")
    metadata_end = 4 + metadata_size
    if metadata_size <= 0 or metadata_end > len(body):
        raise ValueError("GIF frame batch has invalid metadata")
    try:
        metadata = json.loads(body[4:metadata_end].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("GIF frame batch metadata is invalid") from error
    if not isinstance(metadata, list) or not metadata:
        raise ValueError("GIF frame batch must contain at least one frame")
    if len(metadata) > EPISODE_EXPORT_BATCH_MAX_FRAMES:
        raise ValueError(f"GIF frame batches may contain at most {EPISODE_EXPORT_BATCH_MAX_FRAMES} frames")

    frames = []
    offset = metadata_end
    for item in metadata:
        if not isinstance(item, dict):
            raise ValueError("GIF frame batch metadata is invalid")
        index = item.get("index")
        length = item.get("length")
        duration = item.get("duration")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError("GIF frame batch has an invalid frame index")
        if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
            raise ValueError("GIF frame batch has an invalid frame length")
        try:
            duration = float(duration)
        except (TypeError, ValueError) as error:
            raise ValueError("GIF frame batch has an invalid frame duration") from error
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError("GIF frame batch has an invalid frame duration")
        end = offset + length
        if end > len(body):
            raise ValueError("GIF frame batch is shorter than its metadata")
        frames.append((index, body[offset:end], duration))
        offset = end
    if offset != len(body):
        raise ValueError("GIF frame batch has unexpected trailing data")
    return frames


@router.post("/api/episode_exports")
def create_episode_export():
    destination = _episode_png_output_path()
    try:
        return {"export_id": episode_exports.create(destination)}
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@router.post("/api/episode_exports/{export_id}/frames")
def append_episode_export(export_id: str, request: EpisodeExportFrameRequest):
    export = _get_episode_export(export_id)
    with export.lock:
        try:
            export.append(request.index, request.frame, request.duration)
        except (ValueError, OSError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {"frame_count": len(export.durations)}


@router.post("/api/episode_exports/{export_id}/frame_batch")
async def append_episode_export_batch(export_id: str, request: Request):
    """Append a bounded run of binary PNGs, preserving frame order."""
    export = _get_episode_export(export_id)
    try:
        frames = _decode_episode_export_batch(await request.body())
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    with export.lock:
        expected_indices = list(range(len(export.durations), len(export.durations) + len(frames)))
        if [index for index, _data, _duration in frames] != expected_indices:
            raise HTTPException(status_code=400, detail=f"Expected frame {len(export.durations)}")
        try:
            for index, data, duration in frames:
                export.append_png_bytes(index, data, duration)
        except (ValueError, OSError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {"frame_count": len(export.durations)}


@router.post("/api/episode_exports/{export_id}/finish")
def finish_episode_export(export_id: str, request: EpisodeExportFinishRequest):
    export = _get_episode_export(export_id)
    try:
        with export.lock:
            return export.finish(request.frame_count)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    finally:
        episode_exports.remove(export_id)


@router.delete("/api/episode_exports/{export_id}")
def cancel_episode_export(export_id: str):
    if _is_public_mode():
        raise HTTPException(status_code=403, detail="Saving full episodes is disabled in public mode.")
    episode_exports.remove(export_id)
    return {"status": "cancelled"}
