from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import threading
from pathlib import Path


_poster_slots = threading.BoundedSemaphore(2)


def cached_video_poster(source: Path, runtime: Path) -> Path:
    """Decode one frame into a disposable local cache, never into the archive."""
    source = source.resolve()
    stat = source.stat()
    identity = {
        "schema_version": "visioncortex-video-poster/1",
        "source": str(source), "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns, "ctime_ns": stat.st_ctime_ns,
        "frame": "first_decoded_frame", "maximum_width": 960,
    }
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache = runtime.resolve() / "cache/web-video-posters"
    if not cache.resolve().is_relative_to(runtime.resolve()):
        raise ValueError("Poster cache must remain inside local runtime")
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / f"{key}.jpg"
    if destination.is_file() and not destination.is_symlink():
        return destination
    if not _poster_slots.acquire(timeout=10):
        raise TimeoutError("Video preview generation is busy")
    try:
        if destination.is_file() and not destination.is_symlink():
            return destination
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
             "-threads", "1", "-i", str(source), "-map", "0:v:0",
             "-frames:v", "1", "-an", "-sn", "-filter_threads", "1",
             "-vf", "scale='min(960,iw)':-2", "-threads", "1",
             "-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1"],
            capture_output=True, check=False, timeout=15,
        )
        if result.returncode or not result.stdout.startswith(b"\xff\xd8"):
            raise ValueError("Video preview could not be decoded")
        current = source.stat()
        if (current.st_size, current.st_mtime_ns, current.st_ctime_ns) != (
            stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns
        ):
            raise ValueError("Video changed while generating preview")
        with tempfile.NamedTemporaryFile(dir=cache, suffix=".partial", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(result.stdout)
        try:
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination
    finally:
        _poster_slots.release()
