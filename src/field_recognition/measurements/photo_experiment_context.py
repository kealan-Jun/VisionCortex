"""Operator-assigned experiment windows for generated NAS triplet receipts.

Capture time selects an immutable window, not the import/retry time. This only
adds provenance; it never confirms a measurement or changes an OCR value.
"""

from datetime import datetime
import json
import os
from pathlib import Path
from field_recognition.runtime.paths import PROJECT_ROOT


def _instant(value):
    result = datetime.fromisoformat(value)
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("experiment window requires an explicit timezone")
    return result


def resolve_context(camera, photos):
    data = Path(os.environ.get("FIELD_DEMO_DATA", str(PROJECT_ROOT / "Data")))
    path = data / "Runtime" / "PhotoExperimentContexts.json"
    if not path.exists():
        return None
    with path.open("rb") as handle:
        raw = handle.read(65537)
    if len(raw) > 65536:
        raise ValueError("experiment window config exceeds limit")
    config = json.loads(raw)
    bindings = config.get("bindings")
    if config.get("version") != 1 or not isinstance(bindings, list) or len(bindings) > 128:
        raise ValueError("invalid experiment window config")
    windows = []
    for binding in bindings:
        if binding.get("camera_id") != camera:
            continue
        start, end = _instant(binding["start_at"]), _instant(binding["end_at"])
        ref = binding.get("experiment_context_ref")
        if start >= end or not isinstance(ref, str) or not ref.strip() or len(ref) > 2000:
            raise ValueError("invalid experiment window")
        if any(start < old_end and old_start < end for old_start, old_end, _ in windows):
            raise ValueError("overlapping experiment windows")
        windows.append((start, end, ref))
    if not photos:
        return None
    moments = [_instant(photo["captured_at"]) for photo in photos]
    matching = [ref for start, end, ref in windows if all(start <= at < end for at in moments)]
    return matching[0] if matching else None
