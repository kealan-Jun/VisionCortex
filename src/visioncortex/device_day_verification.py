"""Bounded process-local integrity checks for immutable device-day artifacts."""
from collections import OrderedDict
from pathlib import Path
import threading

from .device_day_contract import file_hash, safe_child


class ArtifactVerifier:
    """Share successful byte checks; never persist trust across service restarts."""

    def __init__(self, maximum=4096):
        self.maximum = maximum
        self._verified = OrderedDict()
        self._lock = threading.Lock()
        self._stripes = [threading.Lock() for _ in range(32)]

    @staticmethod
    def _identity(path):
        stat = path.stat()
        return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)

    def verify(self, root: Path, reference: dict) -> bool:
        try:
            path = safe_child(root, reference['path'])
            if not path.is_file():
                return False
            expected = (reference['size_bytes'], reference['sha256'])
            with self._stripes[hash(str(path)) % len(self._stripes)]:
                identity = self._identity(path)
                if identity[2] != expected[0]:
                    return False
                key = (str(path), identity, expected)
                with self._lock:
                    cached = key in self._verified
                    if cached:
                        self._verified.move_to_end(key)
                if cached:
                    return True
                checksum = file_hash(path)
                if checksum != expected[1] or identity != self._identity(path):
                    return False
                with self._lock:
                    self._verified[key] = True
                    while len(self._verified) > self.maximum:
                        self._verified.popitem(last=False)
                return True
        except (OSError, ValueError, KeyError, TypeError):
            return False


_VERIFIER = ArtifactVerifier()
verify_artifact_cached = _VERIFIER.verify
