"""Stable paths for tests, independent of their domain directory."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "src/visioncortex/web"


def web_source() -> str:
    """Legacy characterization tests can locate functions across domain files."""
    return "\n".join((WEB / name).read_text(encoding="utf-8") for name in (
        "app.js", "uploads.js", "archives.js", "materials.js", "tasks.js",
    ))
