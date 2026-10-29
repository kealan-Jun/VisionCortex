"""Historical experiment JSON serialization contract; distinct from device-day atomic_json."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "value"):
        return value.value
    raise TypeError(type(value).__name__)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    temporary.replace(path)
