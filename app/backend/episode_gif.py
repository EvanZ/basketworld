"""Disk-backed episode export: memory use is bounded by one decoded frame."""

import base64
import io
import os
import tempfile
import threading
import time
import uuid
from pathlib import Path

from PIL import GifImagePlugin, Image


def quantize_gif_frame(frame):
    """Keep small bright UI details and smooth glows within GIF's 256 colors."""
    # Median-cut spends most colors on the large dark court and bands the LED
    # glows. Octree preserves those smaller color regions. Dithering must be a
    # separate palette-mapping pass: quantize(colors=...) does not apply it.
    with frame.quantize(colors=256, method=Image.Quantize.FASTOCTREE) as palette:
        return frame.quantize(palette=palette, dither=Image.Dither.FLOYDSTEINBERG)


class EpisodeGifExport:
    def __init__(self, destination):
        self.destination = Path(destination)
        self.temp = tempfile.TemporaryDirectory(prefix="basketworld-gif-")
        self.durations = []
        self.size = (0, 0)
        self.updated = time.monotonic()
        self.lock = threading.Lock()

    def append(self, index, frame, duration):
        if index != len(self.durations):
            raise ValueError(f"Expected frame {len(self.durations)}, received {index}")
        data = base64.b64decode(frame.split(",", 1)[-1], validate=True)
        with Image.open(io.BytesIO(data)) as image:
            if image.format != "PNG":
                raise ValueError("Export frames must be PNG images")
            size = image.size
            canvas_size = tuple(max(a, b) for a, b in zip(self.size, size))
            if max(canvas_size) > 65535 or canvas_size[0] * canvas_size[1] > 16_000_000:
                raise ValueError("Export canvas exceeds the GIF size limit (16 million pixels)")
            image.verify()
        path = Path(self.temp.name) / f"{index}.png"
        path.write_bytes(data)
        self.size = canvas_size
        self.durations.append(max(10, int(round(duration * 1000))))
        self.updated = time.monotonic()

    def finish(self, expected_frames):
        if not self.durations or expected_frames != len(self.durations):
            raise ValueError(f"Expected {expected_frames} frames; received {len(self.durations)}")
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        # Never expose a partial GIF or overwrite another export with the same
        # timestamp. The caller supplies a unique destination for this session.
        with tempfile.NamedTemporaryFile(dir=self.destination.parent, suffix=".gif.tmp", delete=False) as output:
            staging_path = Path(output.name)
            try:
                for index, duration in enumerate(self.durations):
                    with Image.open(Path(self.temp.name) / f"{index}.png") as source:
                        with Image.new("RGB", self.size, (10, 15, 30)) as canvas:
                            rgba = source.convert("RGBA")
                            canvas.paste(rgba, (0, 0), rgba)
                            rgba.close()
                            # Each frame has its own palette. Pillow's save_all
                            # collects the entire decoded animation in memory;
                            # getdata writes just this frame, then releases it.
                            with quantize_gif_frame(canvas) as palette_frame:
                                if index == 0:
                                    header, _ = GifImagePlugin.getheader(
                                        palette_frame, info={"loop": 0, "optimize": False},
                                    )
                                    output.writelines(header)
                                output.writelines(GifImagePlugin.getdata(
                                    palette_frame, duration=duration, disposal=2,
                                    include_color_table=True,
                                ))
                output.write(b";")
                output.flush()
                os.replace(staging_path, self.destination)
            finally:
                staging_path.unlink(missing_ok=True)
        return {
            "status": "success", "file_path": str(self.destination),
            "frame_count": len(self.durations), "width": self.size[0], "height": self.size[1],
        }

    def close(self):
        self.temp.cleanup()


class EpisodeGifExports:
    """Dev-only upload sessions, with cancellation and abandoned-upload cleanup."""

    def __init__(self):
        self.sessions = {}
        self.lock = threading.Lock()

    def create(self, destination):
        with self.lock:
            for key, export in list(self.sessions.items()):
                if time.monotonic() - export.updated > 3600 and export.lock.acquire(blocking=False):
                    try:
                        export.close()
                        del self.sessions[key]
                    finally:
                        export.lock.release()
            if len(self.sessions) >= 4:
                raise ValueError("Too many active GIF exports; finish or cancel an existing export")
            key = uuid.uuid4().hex
            self.sessions[key] = EpisodeGifExport(destination)
            return key

    def get(self, key):
        with self.lock:
            if key not in self.sessions:
                raise KeyError("GIF export expired or backend restarted; please retry saving")
            return self.sessions[key]

    def remove(self, key):
        with self.lock:
            export = self.sessions.pop(key, None)
        if export:
            with export.lock:
                export.close()
