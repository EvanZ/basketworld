import base64
import io
import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image, ImageSequence

from app.backend.routes import media_routes
from app.backend.episode_gif import EpisodeGifExports, quantize_gif_frame


def png_data_url(size, color):
    stream = io.BytesIO()
    Image.new("RGB", size, color).save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()


def png_bytes(size, color):
    stream = io.BytesIO()
    Image.new("RGB", size, color).save(stream, format="PNG")
    return stream.getvalue()


def png_batch(frames):
    metadata = json.dumps([
        {"index": index, "duration": duration, "length": len(data)}
        for index, data, duration in frames
    ]).encode()
    return len(metadata).to_bytes(4, byteorder="big") + metadata + b"".join(
        data for _index, data, _duration in frames
    )


@pytest.mark.parametrize("endpoint", ["render_gif_from_pngs", "save_episode_from_pngs"])
def test_gif_preserves_larger_later_frames_and_final_hold(endpoint, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("BW_PUBLIC_MODE", raising=False)
    monkeypatch.setattr(media_routes, "game_state", SimpleNamespace(env=None, run_id=None))
    app = FastAPI()
    app.include_router(media_routes.router)
    # The middle frame models added scoreboard/caption height; the last shrinks
    # again. Both dimensions, padding, and the one-second End Game hold survive.
    response = TestClient(app).post(f"/api/{endpoint}", json={
        "frames": [
            png_data_url((20, 30), "red"),
            png_data_url((30, 50), "blue"),
            png_data_url((20, 30), "green"),
        ],
        "durations": [0.2, 0.3, 1.0],
    })
    assert response.status_code == 200, response.text
    if endpoint == "render_gif_from_pngs":
        assert response.headers["content-type"] == "image/gif"
        source = io.BytesIO(response.content)
    else:
        source = tmp_path / response.json()["file_path"]
        assert source.is_file()
    with Image.open(source) as gif:
        assert gif.size == (30, 50)
        frames = list(ImageSequence.Iterator(gif))
        assert len(frames) == 3
        durations = []
        for index, frame in enumerate(ImageSequence.Iterator(gif)):
            durations.append(frame.info["duration"])
            rgb = frame.convert("RGB")
            assert rgb.size == (30, 50)
            if index == 1:
                assert rgb.getpixel((29, 49)) == (0, 0, 255)
            else:
                assert rgb.getpixel((29, 49)) == (10, 15, 30)
        assert durations == [200, 300, 1000]


def test_gif_rejects_empty_frames():
    app = FastAPI()
    app.include_router(media_routes.router)
    response = TestClient(app).post("/api/render_gif_from_pngs", json={"frames": []})
    assert response.status_code == 400


@pytest.fixture
def export_client(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("BW_PUBLIC_MODE", raising=False)
    monkeypatch.setattr(media_routes, "game_state", SimpleNamespace(env=None, run_id=None))
    monkeypatch.setattr(media_routes, "episode_exports", EpisodeGifExports())
    app = FastAPI()
    app.include_router(media_routes.router)
    return TestClient(app)


def test_streamed_episode_preserves_all_frames_palettes_sizes_and_hold(export_client):
    key = export_client.post("/api/episode_exports").json()["export_id"]
    prefix = f"/api/episode_exports/{key}"
    export = media_routes.episode_exports.get(key)
    temp_path = export.temp.name
    count = 160
    for index in range(count):
        color = (index, 255 - index, 40)
        size = (20, 30) if index % 2 == 0 else (30, 50)
        response = export_client.post(prefix + "/frames", json={
            "index": index, "frame": png_data_url(size, color),
            "duration": 1 if index == count - 1 else 0.1,
        })
        assert response.status_code == 200, response.text
        assert response.json()["frame_count"] == index + 1
    response = export_client.post(prefix + "/finish", json={"frame_count": count})
    assert response.status_code == 200, response.text
    assert response.json()["frame_count"] == count
    with Image.open(response.json()["file_path"]) as gif:
        assert gif.n_frames == count
        assert gif.size == (30, 50)
        assert gif.info["loop"] == 0
        for index, frame in enumerate(ImageSequence.Iterator(gif)):
            rgb = frame.convert("RGB")
            assert rgb.getpixel((10, 10)) == (index, 255 - index, 40)
            if index % 2 == 0:
                assert rgb.getpixel((29, 49)) == (10, 15, 30)
            assert frame.info["duration"] == (1000 if index == count - 1 else 100)
    from pathlib import Path
    assert not Path(temp_path).exists()
    assert key not in media_routes.episode_exports.sessions


def test_streamed_episode_accepts_bounded_binary_png_batches(export_client):
    key = export_client.post("/api/episode_exports").json()["export_id"]
    prefix = f"/api/episode_exports/{key}"
    count = 16
    for first in range(0, count, 8):
        frames = []
        for index in range(first, first + 8):
            size = (20, 30) if index % 2 == 0 else (30, 50)
            frames.append((index, png_bytes(size, (index, 255 - index, 40)), 1 if index == count - 1 else 0.1))
        response = export_client.post(
            prefix + "/frame_batch",
            content=png_batch(frames),
            headers={"content-type": "application/octet-stream"},
        )
        assert response.status_code == 200, response.text
        assert response.json()["frame_count"] == first + 8
    response = export_client.post(prefix + "/finish", json={"frame_count": count})
    assert response.status_code == 200, response.text
    with Image.open(response.json()["file_path"]) as gif:
        assert gif.n_frames == count
        assert gif.size == (30, 50)
        assert gif.info["duration"] == 100
        gif.seek(count - 1)
        assert gif.info["duration"] == 1000


def test_binary_png_batch_rejects_out_of_order_frames(export_client):
    key = export_client.post("/api/episode_exports").json()["export_id"]
    prefix = f"/api/episode_exports/{key}"
    response = export_client.post(
        prefix + "/frame_batch",
        content=png_batch([(1, png_bytes((10, 10), "red"), 1)]),
        headers={"content-type": "application/octet-stream"},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Expected frame 0"
    export_client.delete(prefix)


def test_streamed_export_rejects_missing_frames_and_cleans_up(export_client):
    key = export_client.post("/api/episode_exports").json()["export_id"]
    prefix = f"/api/episode_exports/{key}"
    export = media_routes.episode_exports.get(key)
    frame = png_data_url((10, 10), "red")
    assert export_client.post(prefix + "/frames", json={"index": 1, "frame": frame, "duration": 1}).status_code == 400
    assert export_client.post(prefix + "/frames", json={"index": 0, "frame": "broken", "duration": 1}).status_code == 400
    assert export_client.post(prefix + "/frames", json={"index": 0, "frame": frame, "duration": 1}).status_code == 200
    assert export_client.post(prefix + "/finish", json={"frame_count": 2}).status_code == 400
    assert not export.destination.exists()
    assert key not in media_routes.episode_exports.sessions


def test_streamed_export_cancel_and_public_mode(export_client, monkeypatch):
    key = export_client.post("/api/episode_exports").json()["export_id"]
    assert export_client.delete(f"/api/episode_exports/{key}").status_code == 200
    assert key not in media_routes.episode_exports.sessions
    monkeypatch.setenv("BW_PUBLIC_MODE", "true")
    assert export_client.post("/api/episode_exports").status_code == 403
    assert export_client.delete(f"/api/episode_exports/{key}").status_code == 403


def test_encoder_failure_never_leaves_partial_gif(export_client, monkeypatch):
    from app.backend import episode_gif
    key = export_client.post("/api/episode_exports").json()["export_id"]
    prefix = f"/api/episode_exports/{key}"
    export = media_routes.episode_exports.get(key)
    export_client.post(prefix + "/frames", json={
        "index": 0, "frame": png_data_url((10, 10), "red"), "duration": 1,
    })
    def fail(*args, **kwargs):
        raise OSError("encoder failed")
    monkeypatch.setattr(episode_gif.GifImagePlugin, "getdata", fail)
    with pytest.raises(OSError, match="encoder failed"):
        export_client.post(prefix + "/finish", json={"frame_count": 1})
    assert not export.destination.exists()
    assert not list(export.destination.parent.glob("*.gif.tmp"))
    assert key not in media_routes.episode_exports.sessions


def test_streamed_export_bounds_combined_canvas_not_just_each_frame(export_client):
    key = export_client.post("/api/episode_exports").json()["export_id"]
    prefix = f"/api/episode_exports/{key}"
    assert export_client.post(prefix + "/frames", json={
        "index": 0, "frame": png_data_url((4001, 10), "red"), "duration": 1,
    }).status_code == 200
    response = export_client.post(prefix + "/frames", json={
        "index": 1, "frame": png_data_url((10, 4001), "blue"), "duration": 1,
    })
    assert response.status_code == 400
    assert "canvas" in response.json()["detail"]
    export_client.delete(prefix)


@pytest.mark.parametrize("endpoint", ["render_gif_from_pngs", "save_episode_from_pngs", "stream"])
def test_all_gif_paths_use_dithered_color_conversion(export_client, endpoint):
    import numpy as np

    # More than 256 colors, including small warm highlights against a large
    # dark blue area. Exact decoded pixels catch accidental re-quantization or
    # a short-animation endpoint silently using the old color conversion.
    y, x = np.indices((96, 160))
    pixels = np.stack((x, y, (x + y) % 256), axis=-1).astype(np.uint8)
    pixels[10:18, 10:18] = (251, 191, 36)
    with Image.fromarray(pixels) as source:
        output = io.BytesIO()
        source.save(output, format="PNG")
        with quantize_gif_frame(source) as palette:
            expected = palette.convert("RGB").tobytes()
        # A palette-only pass lacks error diffusion and must not be equivalent.
        with source.quantize(colors=256, method=Image.Quantize.FASTOCTREE) as undithered:
            assert expected != undithered.convert("RGB").tobytes()
    png = "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()
    if endpoint == "stream":
        key = export_client.post("/api/episode_exports").json()["export_id"]
        prefix = f"/api/episode_exports/{key}"
        assert export_client.post(prefix + "/frames", json={"index": 0, "frame": png, "duration": 1}).status_code == 200
        response = export_client.post(prefix + "/finish", json={"frame_count": 1})
    else:
        response = export_client.post(f"/api/{endpoint}", json={"frames": [png], "durations": [1]})
    assert response.status_code == 200, response.text
    gif_source = io.BytesIO(response.content) if endpoint == "render_gif_from_pngs" else response.json()["file_path"]
    with Image.open(gif_source) as result:
        assert result.size == (160, 96)
        assert result.info["duration"] == 1000
        assert result.convert("RGB").tobytes() == expected
