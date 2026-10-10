from __future__ import annotations

import os
from pathlib import Path


def configure_third_party_runtime() -> Path:
    """Isolate third-party mutable settings from a user's global config.

    Ultralytics creates ``settings.json`` at import time.  Setting its
    supported configuration directory before any lazy model import keeps that
    mutation inside a VisionCortex-owned, auditable location.
    """

    configured = os.environ.get("VISIONCORTEX_ULTRALYTICS_CONFIG_DIR")
    if configured:
        destination = Path(configured).expanduser().resolve()
    else:
        state_root = Path(
            os.environ.get("XDG_STATE_HOME") or Path.home() / ".local" / "state"
        )
        # YOLO_CONFIG_DIR is a *parent* directory; Ultralytics appends its own
        # ``Ultralytics`` subdirectory.
        destination = (state_root / "VisionCortex" / "ThirdParty").resolve()
    os.environ.setdefault("YOLO_CONFIG_DIR", str(destination))
    return Path(os.environ["YOLO_CONFIG_DIR"])
