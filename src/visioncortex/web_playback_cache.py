"""Small on-demand browser clips, kept outside the evidence archive."""
from pathlib import Path
import threading
import time
import uuid

from .device_day_contract import digest

_SLOTS = threading.BoundedSemaphore(1)


def cached_preview(source, runtime, start_ms, duration_ms, source_identity):
    from .video_io import extract_clip
    if not 0 < duration_ms <= 35_000:
        raise ValueError('Only short playback windows are allowed')
    source = Path(source)
    stat = source.stat()
    identity = (str(source), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    key = digest([identity, source_identity, start_ms, duration_ms, 'web-playback-v1'])
    root = Path(runtime).resolve()/'cache/WebPlayback'
    root.mkdir(parents=True, exist_ok=True)
    target = root/(key+'.mp4')
    if target.is_file() and not target.is_symlink():
        return target
    if not _SLOTS.acquire(timeout=20):
        raise TimeoutError('Preview queue busy')
    temp = root/(key+'.'+uuid.uuid4().hex+'.partial.mp4')
    try:
        if target.is_file() and not target.is_symlink():
            return target
        # Reuse the same accurate-seek exporter as ProcessedClips. This is a
        # disposable viewing derivative, not an additional evidence artifact.
        extract_clip(source, temp, start_ms, duration_ms)
        current = source.stat()
        if identity != (str(source), current.st_size, current.st_mtime_ns, current.st_ctime_ns):
            raise ValueError('Source changed during preview')
        temp.replace(target)
        # Only this cache's own expired files; never touch NAS/raw media.
        files = sorted((p for p in root.glob('*.mp4') if not p.is_symlink()), key=lambda p: p.stat().st_mtime)
        total = sum(p.stat().st_size for p in files)
        for p in files:
            if p != target and (time.time()-p.stat().st_mtime > 1800 or total > 512*1024**2):
                try:
                    size = p.stat().st_size
                    p.unlink()
                    total -= size
                except OSError:
                    pass
        return target
    finally:
        temp.unlink(missing_ok=True)
        _SLOTS.release()
