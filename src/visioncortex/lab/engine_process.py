"""Isolated customer model bridge. No old model defaults, downloads or paid calls."""
from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))


def run(request):
    model, kind = request["model"], request["kind"]
    result = dict(model_revision=model["revision"], model_invoked=True, normalized_fields=[], association_status="unknown")
    if kind == "vision":
        from visioncortex.lab.native_vision import screen
        result.update(screen(request))
    elif kind == "ocr":
        from visioncortex.lab.native_ocr import read
        result.update(read(request))
    elif kind == "asr":
        from visioncortex.lab.native_asr import transcribe
        result.update(transcribe(request))
    elif kind == 'understanding':
        from visioncortex.lab.native_pipeline import deliver
        result.update(deliver(request))
    else:
        raise ValueError("Unsupported engine")
    return result


if __name__ == "__main__":
    request = json.loads(sys.stdin.read(16 * 1024**2))
    with contextlib.redirect_stdout(sys.stderr):
        result = run(request)
    sys.stdout.write(json.dumps(result, ensure_ascii=False, allow_nan=False))
