from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any


REGISTRY_SCHEMA = "visioncortex-closed-set-model-registry/1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model_registry(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if payload.get("schema_version") != REGISTRY_SCHEMA:
        raise ValueError(f"Unsupported closed-set model registry: {path}")
    models = payload.get("models")
    if not isinstance(models, dict) or set(models) != {
        "first_person",
        "third_person",
    }:
        raise ValueError("Closed-set model registry must define both view roles")
    return payload


def validate_model_registry(path: Path, *, inspect_ontology: bool = True) -> dict[str, Any]:
    registry = load_model_registry(path)
    expected_classes = [str(item).replace("-", "_") for item in registry["ontology"]["classes"]]
    if len(expected_classes) != int(registry["ontology"]["class_count"]):
        raise ValueError("Closed-set registry ontology class_count mismatch")
    if len(set(expected_classes)) != len(expected_classes):
        raise ValueError("Closed-set registry ontology contains duplicate classes")

    records = []
    for role, configured in registry["models"].items():
        model_path = Path(str(configured.get("path") or "")).resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f"Registered {role} model is missing: {model_path}")
        actual_size = model_path.stat().st_size
        expected_size = int(configured.get("size_bytes") or 0)
        if actual_size != expected_size:
            raise RuntimeError(
                f"Registered {role} model size mismatch: expected={expected_size} actual={actual_size}"
            )
        actual_hash = _sha256(model_path)
        expected_hash = str(configured.get("sha256") or "").lower()
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"Registered {role} model hash mismatch: expected={expected_hash} actual={actual_hash}"
            )
        record: dict[str, Any] = {
            "role": role,
            "path": str(model_path),
            "size_bytes": actual_size,
            "sha256": actual_hash,
            "status": "hash_validated",
        }
        if inspect_ontology:
            from ultralytics import YOLO

            model = YOLO(str(model_path))
            names = [
                str(value).replace("-", "_")
                for _, value in sorted(model.names.items(), key=lambda item: int(item[0]))
            ]
            if names != expected_classes:
                raise RuntimeError(f"Registered {role} model ontology mismatch")
            record["class_count"] = len(names)
            record["status"] = "hash_and_ontology_validated"
        records.append(record)
    return {
        "schema_version": "visioncortex-closed-set-model-validation/1",
        "status": "passed",
        "registry": str(path.resolve()),
        "class_count": len(expected_classes),
        "models": records,
        "source_copy_bytes": 0,
        "nas_accessed": False,
    }


def install_registered_models(
    registry_path: Path,
    sources: dict[str, Path],
) -> dict[str, Any]:
    """Install verified source weights without silently replacing a model."""

    registry = load_model_registry(registry_path)
    installed = []
    for role in ("first_person", "third_person"):
        source = sources[role].resolve()
        configured = registry["models"][role]
        destination = Path(str(configured["path"])).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Source model is missing for {role}: {source}")
        expected_hash = str(configured["sha256"]).lower()
        if source.stat().st_size != int(configured["size_bytes"]) or _sha256(source) != expected_hash:
            raise RuntimeError(f"Source model does not match registry for {role}: {source}")
        if destination.exists():
            if not destination.is_file() or _sha256(destination) != expected_hash:
                raise FileExistsError(
                    f"Refusing to overwrite non-matching registered model: {destination}"
                )
            installed.append(
                {"role": role, "path": str(destination), "status": "reused_exact"}
            )
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.partial-{uuid.uuid4().hex[:8]}")
        try:
            shutil.copyfile(source, temporary)
            if _sha256(temporary) != expected_hash:
                raise RuntimeError(f"Copied model failed verification for {role}")
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        installed.append(
            {"role": role, "path": str(destination), "status": "installed_verified"}
        )
    return {
        "schema_version": "visioncortex-closed-set-model-installation/1",
        "status": "completed",
        "registry": str(registry_path.resolve()),
        "models": installed,
        "nas_accessed": False,
    }
