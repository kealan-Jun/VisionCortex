"""Bounded isolated reader for the pinned FR photo-directory protocol."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def read(request):
    from visioncortex.lab.storage import verify_root
    from visioncortex.lab.components import enable_components
    root = verify_root(Path(request['root']), os.environ.get('VC_LAB_NAS_MARKER_SHA256'))
    enable_components()
    from field_recognition.runtime import source_io
    from visioncortex.lab.input_layout import photo_location
    # Only this short-lived child receives the path binding. Original component
    # bytes and the parent's FR module remain unchanged; grouping stays native.
    source_io.photo_location = photo_location
    result = source_io.read_one(dict(request, root=str(root)))
    # Stable bytes are independently rehashed by the caller. No large base64
    # image crosses the subprocess boundary.
    result.pop('image_base64', None)
    return result


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    try:
        result = read(json.loads(sys.stdin.read(65536)))
    except Exception as exc:
        result = dict(error=type(exc).__name__, reason=str(exc)[:200])
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
