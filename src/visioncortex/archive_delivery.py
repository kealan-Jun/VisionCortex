"""Retryable NAS delivery for profiles that compute into a local archive.

A NAS-direct profile still needs its selected filesystem to write outputs. This
adapter does not change configured roots or copy original input media.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

from .archive import write_json
from .storage import IncrementalArchivePublisher


class DeferredArchivePublisher(IncrementalArchivePublisher):
    def __init__(self, local_root: Path, nas_root: Path):
        super().__init__(local_root, nas_root)
        self._delivery_lock = threading.RLock()
        self.journal = self.local_root / "JSON-Config-Files/archive_delivery.json"
        try:
            saved = json.loads(self.journal.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            saved = {}
        self.pending = dict(saved.get("pending", {})) if saved.get("destination") == str(nas_root) else {}

    def _save(self):
        with self._delivery_lock:
            write_json(self.journal, {"schema_version": "visioncortex-archive-delivery/1",
                                     "destination": str(self.nas_root), "pending": dict(self.pending),
                                     "status": "pending" if self.pending else "synced",
                                     "source_media_copied": False})

    def publish_file(self, source: Path) -> Path:
        relative = source.resolve().relative_to(self.local_root).as_posix()
        if relative == "JSON-Config-Files/archive_delivery.json":
            return self.nas_root / relative
        try:
            result = super().publish_file(source)
            with self._delivery_lock:
                if self.pending.pop(relative, None) is not None:
                    self._save()
            return result
        except OSError as exc:
            with self._delivery_lock:
                self.pending[relative] = {"error_type": type(exc).__name__, "reason": "目标保存目录暂时无法写入；本地产物已保留"}
                self._save()
            return self.nas_root / relative

    def publish_status(self, payload: dict) -> Path:
        path = self.local_root / "JSON-Config-Files/pipeline_status.json"
        write_json(path, payload)
        return self.publish_file(path)

    def flush(self) -> bool:
        with self._delivery_lock:
            pending = list(self.pending)
        for relative in pending:
            # The journal is data, never authority to publish another directory.
            path = self.local_root / relative
            if Path(relative).is_absolute() or ".." in Path(relative).parts or path.is_symlink():
                continue
            self.publish_file(path)
        self._save()
        return not self.pending
