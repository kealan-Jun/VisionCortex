from __future__ import annotations

import hashlib
import json

import pytest

from labvision_evidence.model_registry import (
    install_registered_models,
    validate_model_registry,
)


def _registry(tmp_path, first: bytes = b"first", third: bytes = b"third"):
    registry = tmp_path / "registry.json"
    paths = {
        "first_person": tmp_path / "models" / "first.pt",
        "third_person": tmp_path / "models" / "third.pt",
    }
    payload = {
        "schema_version": "visioncortex-closed-set-model-registry/1",
        "ontology": {"class_count": 2, "classes": ["hand", "paper"]},
        "models": {
            "first_person": {
                "path": str(paths["first_person"]),
                "size_bytes": len(first),
                "sha256": hashlib.sha256(first).hexdigest(),
            },
            "third_person": {
                "path": str(paths["third_person"]),
                "size_bytes": len(third),
                "sha256": hashlib.sha256(third).hexdigest(),
            },
        },
    }
    registry.write_text(json.dumps(payload), encoding="utf-8")
    return registry, paths


def test_install_and_hash_validate_registered_models(tmp_path):
    registry, paths = _registry(tmp_path)
    first_source = tmp_path / "first-source.pt"
    third_source = tmp_path / "third-source.pt"
    first_source.write_bytes(b"first")
    third_source.write_bytes(b"third")

    installed = install_registered_models(
        registry,
        {"first_person": first_source, "third_person": third_source},
    )
    report = validate_model_registry(registry, inspect_ontology=False)

    assert installed["status"] == "completed"
    assert paths["first_person"].read_bytes() == b"first"
    assert report["status"] == "passed"
    assert report["nas_accessed"] is False


def test_install_refuses_non_matching_existing_destination(tmp_path):
    registry, paths = _registry(tmp_path)
    paths["first_person"].parent.mkdir(parents=True)
    paths["first_person"].write_bytes(b"wrong")
    first_source = tmp_path / "first-source.pt"
    third_source = tmp_path / "third-source.pt"
    first_source.write_bytes(b"first")
    third_source.write_bytes(b"third")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        install_registered_models(
            registry,
            {"first_person": first_source, "third_person": third_source},
        )


def test_registry_rejects_duplicate_ontology(tmp_path):
    registry, _paths = _registry(tmp_path)
    payload = json.loads(registry.read_text(encoding="utf-8"))
    payload["ontology"] = {"class_count": 2, "classes": ["hand", "hand"]}
    registry.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate"):
        validate_model_registry(registry, inspect_ontology=False)
