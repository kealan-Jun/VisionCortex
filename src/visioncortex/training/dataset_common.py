"""Dataset common responsibilities for explicitly authorized training."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any
import yaml

DATASET_SCHEMA = "visioncortex-yolo-training-dataset/1"
TRAINING_SCHEMA = "visioncortex-yolo-training-run/1"
EVALUATION_SCHEMA = "visioncortex-yolo-human-truth-evaluation/1"
PUBLIC_DATASET_SCHEMA = "visioncortex-public-yolo-training-dataset/1"
PUBLIC_UNION_SCHEMA = "visioncortex-mapped-public-yolo-union/1"
DATASET_INTEGRITY_AUDIT_SCHEMA = "visioncortex-yolo-dataset-integrity-audit/1"
TRAINING_TRUTH_STATUSES = frozenset(
    {"reviewed_ground_truth", "public_human_annotations"}
)
_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()



def _safe_name(value: Any) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value).strip())
    return normalized.strip(".-") or "image"



def _validate_review_authority(payload: dict[str, Any]) -> None:
    schema = str(payload.get("schema_version") or "")
    if not schema.startswith("visioncortex-yolo-"):
        raise ValueError("Training input must declare a VisionCortex YOLO schema")
    truth_status = str(payload.get("truth_status") or "").lower()
    if "pseudo" in truth_status or "pseudo" in schema:
        raise ValueError("Pseudo-labels cannot be used as reviewed training truth")



def _dataset_yaml_classes(path: Path) -> list[str]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}
    names = payload.get("names")
    if isinstance(names, list):
        classes = [str(item).strip() for item in names]
    elif isinstance(names, dict):
        indexed = {int(key): str(value).strip() for key, value in names.items()}
        if sorted(indexed) != list(range(len(indexed))):
            raise RuntimeError(f"YOLO class ids are not contiguous: {path}")
        classes = [indexed[index] for index in range(len(indexed))]
    else:
        raise RuntimeError(f"YOLO dataset does not declare names: {path}")
    if not classes or any(not item for item in classes) or len(classes) != len(set(classes)):
        raise RuntimeError(f"YOLO dataset classes are invalid: {path}")
    return classes
