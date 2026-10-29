"""Evict only verified working copies whose producer original is still intact."""
from __future__ import annotations

import fcntl
import time

from .media import hash_file, safe_path


def prune(core, settings, *, now=None, limit=100):
    now = time.time() if now is None else now
    removed, pinned, unverifiable, reclaimed = 0, 0, 0, 0
    lock = settings.cache / '.admission.lock'
    with lock.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        # Content hashes may be shared. Pin every running reader, including another
        # customer using this local cache, before deleting any physical object.
        with core.db.transaction(readonly=True) as conn:
            active = {r['verified_sha256'] for r in conn.execute("SELECT a.verified_sha256 FROM media_asset a JOIN processing_job j ON j.customer_id=a.customer_id AND j.capture_id=a.capture_id WHERE j.status='running'").fetchall()}
            for task in conn.execute("SELECT request FROM hj_timeline_task WHERE status='leased' AND lease_until>now()").fetchall():
                active.update(ref['sha256'] for window in (task['request'] or {}).get('windows', []) for ref in window['input_spans'])
            locations = conn.execute("SELECT l.relative_path,a.id,a.verified_sha256,a.byte_size FROM media_location l JOIN media_asset a ON a.customer_id=l.customer_id AND a.id=l.media_id WHERE l.customer_id=%s AND l.storage='cache' AND l.availability='available' ORDER BY l.relative_path", (core.customer,)).fetchall()
        grouped = {}
        for row in locations:
            grouped.setdefault(row['relative_path'], []).append(row)
        for relative, rows in grouped.items():
            if removed >= limit:
                break
            path = safe_path(settings.cache, relative, must_exist=False)
            if not path.exists() or now - path.stat().st_mtime < settings.cache_ttl_seconds:
                continue
            if any(r['verified_sha256'] in active for r in rows):
                pinned += 1; continue
            verified = True
            for row in rows:
                with core.db.transaction(readonly=True) as conn:
                    originals = conn.execute("SELECT relative_path FROM media_location WHERE customer_id=%s AND media_id=%s AND storage='incoming'", (core.customer, row['id'])).fetchall()
                intact = False
                for original in originals:
                    try:
                        digest, size = hash_file(safe_path(settings.incoming, original['relative_path']), max(settings.max_native_bytes, settings.max_capture_bytes))
                        intact = digest == row['verified_sha256'] and size == row['byte_size']
                        if intact:
                            break
                    except (OSError, ValueError):
                        continue
                verified = verified and intact
            if not verified:
                unverifiable += 1; continue
            size = path.stat().st_size
            # All intake and worker admission use the same lock. A new running
            # reader cannot appear between the pin check and this unlink.
            path.unlink()
            with core.db.transaction() as conn:
                conn.execute("UPDATE media_location SET availability='evicted_verified_original' WHERE customer_id=%s AND storage='cache' AND relative_path=%s", (core.customer, relative))
            removed += 1; reclaimed += size
    return dict(removed=removed, reclaimed_bytes=reclaimed, pinned=pinned, original_unverified=unverifiable, ttl_seconds=settings.cache_ttl_seconds)


class CachePruner:
    def __init__(self, core, settings):
        self.core, self.settings, self.last = core, settings, None

    def tick(self):
        clock = time.monotonic()
        if self.last is not None and clock - self.last < 60:
            return None
        result = prune(self.core, self.settings, limit=5)
        self.last = clock
        return result
