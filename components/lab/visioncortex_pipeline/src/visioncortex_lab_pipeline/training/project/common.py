"""Common responsibilities for explicitly authorized training."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from ...project_training_evidence import durable_json



def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()



def canonical_sha(value: Any) -> str:
    text = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(text.encode()).hexdigest()



def write_json(path: Path, value: Any) -> None:
    durable_json(path, value)
