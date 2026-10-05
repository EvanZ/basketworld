"""Disk-backed episode export: memory use is bounded by one decoded frame."""

import base64
import io
import os
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path

from PIL import GifImagePlugin, Image


class Mp4EncoderUnavailableError(RuntimeError):
    """Raised when neither the bundled nor system FFmpeg can be used."""


class Mp4EncodingError(RuntimeError):
    """Raised when FFmpeg fails to encode a validated episode."""


def quantize_gif_frame(frame):
    """Keep small bright UI details and smooth glows within GIF's 256 colors."""
    # Median-cut spends most colors on the large dark court and bands the LED
    # glows. Octree preserves those smaller color regions. Dithering must be a
    # separate palette-mapping pass: quantize(colors=...) does not apply it.
    with frame.quantize(colors=256, method=Image.Quantize.FASTOCTREE) as palette:
        return frame.quantize(palette=palette, dither=Image.Dither.FLOYDSTEINBERG)


class EpisodeGifExport:
    def __init__(self, destination, *, temp_prefix="basketworld-gif-"):
        self.destination = Path(destination)
        self.temp = tempfile.TemporaryDirectory(prefix=temp_prefix)
        self.durations = []
        self.size = (0, 0)
        self.updated = time.monotonic()
        self.lock = threading.Lock()

    def append_png_bytes(self, index, data, duration):
        """Validate and spool one already-binary PNG frame to the export."""
        if index != len(self.durations):
            raise ValueError(f"Expected frame {len(self.durations)}, received {index}")
        with Image.open(io.BytesIO(data)) as image:
            if image.format != "PNG":
                raise ValueError("Export frames must be PNG images")
            size = image.size
            canvas_size = tuple(max(a, b) for a, b in zip(self.size, size))
            if max(canvas_size) > 65535 or canvas_size[0] * canvas_size[1] > 16_000_000:
                raise ValueError("Export canvas exceeds the 16-million-pixel limit")
            image.verify()
        path = Path(self.temp.name) / f"{index}.png"
        path.write_bytes(data)
        self.size = canvas_size
        self.durations.append(max(10, int(round(duration * 1000))))
        self.updated = time.monotonic()

    def append(self, index, frame, duration):
        """Backward-compatible data-URL upload path for older clients."""
        data = base64.b64decode(frame.split(",", 1)[-1], validate=True)
        self.append_png_bytes(index, data, duration)

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


def _ffmpeg_executable():
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    try:
        import imageio_ffmpeg
    except ImportError as error:
        raise Mp4EncoderUnavailableError(
            "MP4 export requires FFmpeg. Install imageio-ffmpeg in the backend environment."
        ) from error
    try:
        executable = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as error:
        raise Mp4EncoderUnavailableError(
            "MP4 export could not locate FFmpeg. Reinstall imageio-ffmpeg or install ffmpeg."
        ) from error
    if not executable or not Path(executable).is_file():
        raise Mp4EncoderUnavailableError(
            "MP4 export could not locate FFmpeg. Reinstall imageio-ffmpeg or install ffmpeg."
        )
    return executable


class EpisodeMp4Export(EpisodeGifExport):
    """Disk-backed H.264 export with per-frame timing and bounded memory."""

    frames_per_second = 20

    def __init__(self, destination):
        super().__init__(destination, temp_prefix="basketworld-mp4-")

    def _normalize_frames(self):
        width = self.size[0] + (self.size[0] % 2)
        height = self.size[1] + (self.size[1] % 2)
        normalized_dir = Path(self.temp.name) / "normalized"
        normalized_dir.mkdir()
        normalized_paths = []
        for index in range(len(self.durations)):
            source_path = Path(self.temp.name) / f"{index}.png"
            destination_path = normalized_dir / f"{index:08d}.png"
            with Image.open(source_path) as source:
                with Image.new("RGB", (width, height), (10, 15, 30)) as canvas:
                    rgba = source.convert("RGBA")
                    canvas.paste(rgba, (0, 0), rgba)
                    rgba.close()
                    canvas.save(destination_path, format="PNG", optimize=False)
            normalized_paths.append(destination_path)
        return normalized_paths, (width, height)

    def _build_constant_rate_sequence(self, normalized_paths):
        """Represent variable source durations as a precise CFR image sequence.

        Hard links keep repeated holds effectively free on disk. Accumulating
        the target frame total prevents per-source-frame rounding drift across
        long episodes.
        """
        sequence_dir = Path(self.temp.name) / "sequence"
        sequence_dir.mkdir()
        elapsed_ms = 0
        emitted_frames = 0
        for source_path, duration_ms in zip(normalized_paths, self.durations):
            elapsed_ms += duration_ms
            target_total = max(
                emitted_frames + 1,
                int(round(elapsed_ms * self.frames_per_second / 1000)),
            )
            for _ in range(target_total - emitted_frames):
                destination_path = sequence_dir / f"{emitted_frames:08d}.png"
                try:
                    os.link(source_path, destination_path)
                except OSError:
                    shutil.copyfile(source_path, destination_path)
                emitted_frames += 1
        return sequence_dir, emitted_frames

    def finish(self, expected_frames):
        if not self.durations or expected_frames != len(self.durations):
            raise ValueError(f"Expected {expected_frames} frames; received {len(self.durations)}")
        executable = _ffmpeg_executable()
        normalized_paths, output_size = self._normalize_frames()
        sequence_dir, encoded_frame_count = self._build_constant_rate_sequence(
            normalized_paths
        )

        self.destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=self.destination.parent,
            suffix=".mp4.tmp",
            delete=False,
        ) as output:
            staging_path = Path(output.name)
        try:
            command = [
                executable,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-framerate",
                str(self.frames_per_second),
                "-i",
                str(sequence_dir / "%08d.png"),
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                "-r",
                str(self.frames_per_second),
                "-video_track_timescale",
                "1000",
                "-movflags",
                "+faststart",
                "-f",
                "mp4",
                str(staging_path),
            ]
            try:
                result = subprocess.run(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                    text=True,
                )
            except OSError as error:
                raise Mp4EncoderUnavailableError(
                    f"MP4 export could not start FFmpeg: {error}"
                ) from error
            if result.returncode != 0:
                detail = result.stderr.strip().splitlines()
                message = detail[-1] if detail else f"FFmpeg exited with status {result.returncode}"
                raise Mp4EncodingError(f"Failed to encode MP4: {message}")
            if not staging_path.is_file() or staging_path.stat().st_size == 0:
                raise Mp4EncodingError("Failed to encode MP4: FFmpeg produced an empty file")
            os.replace(staging_path, self.destination)
        finally:
            staging_path.unlink(missing_ok=True)
        return {
            "status": "success",
            "file_path": str(self.destination),
            "format": "mp4",
            "codec": "h264",
            "frame_count": len(self.durations),
            "width": output_size[0],
            "height": output_size[1],
            "frames_per_second": self.frames_per_second,
            "encoded_frame_count": encoded_frame_count,
            "duration_seconds": encoded_frame_count / self.frames_per_second,
        }


class EpisodeGifExports:
    """Dev-only GIF/MP4 sessions, with cancellation and abandoned-upload cleanup."""

    def __init__(self):
        self.sessions = {}
        self.lock = threading.Lock()

    def create(self, destination, export_format="gif"):
        with self.lock:
            for key, export in list(self.sessions.items()):
                if time.monotonic() - export.updated > 3600 and export.lock.acquire(blocking=False):
                    try:
                        export.close()
                        del self.sessions[key]
                    finally:
                        export.lock.release()
            if len(self.sessions) >= 4:
                raise ValueError("Too many active episode exports; finish or cancel an existing export")
            key = uuid.uuid4().hex
            if export_format == "gif":
                export = EpisodeGifExport(destination)
            elif export_format == "mp4":
                export = EpisodeMp4Export(destination)
            else:
                raise ValueError("Episode export format must be gif or mp4")
            self.sessions[key] = export
            return key

    def get(self, key):
        with self.lock:
            if key not in self.sessions:
                raise KeyError("Episode export expired or backend restarted; please retry saving")
            return self.sessions[key]

    def remove(self, key):
        with self.lock:
            export = self.sessions.pop(key, None)
        if export:
            with export.lock:
                export.close()
