from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from labvision_evidence.public_datasets import prepare_public_dataset


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _registry(path: Path, archive: Path, *, allowed: bool = True) -> Path:
    payload = {
        "schema_version": "visioncortex-public-dataset-registry/1",
        "datasets": {
            "fixture": {
                "title": "Fixture",
                "license": "MIT",
                "source_record": "https://example.invalid/fixture",
                "scope": "test",
                "automated_acquisition_allowed": allowed,
                "artifact": {
                    "url": "https://example.invalid/fixture.zip",
                    "filename": archive.name,
                    "sha256": _sha(archive),
                    "maximum_uncompressed_bytes": 1024,
                },
            }
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_existing_public_dataset_is_hash_checked_and_safely_extracted(tmp_path: Path):
    source = tmp_path / "source.zip"
    with zipfile.ZipFile(source, "w") as bundle:
        bundle.writestr("sample/Image.png", b"image")
    destination = tmp_path / "datasets"
    fixture = destination / "fixture"
    fixture.mkdir(parents=True)
    archive = fixture / source.name
    archive.write_bytes(source.read_bytes())

    receipt = prepare_public_dataset(
        _registry(tmp_path / "registry.json", source),
        "fixture",
        destination,
    )

    assert receipt["nas_accessed"] is False
    assert receipt["acquisition"]["status"] == "reused_verified"
    assert (fixture / "extracted" / "sample" / "Image.png").read_bytes() == b"image"

    reused = prepare_public_dataset(
        _registry(tmp_path / "registry.json", source),
        "fixture",
        destination,
    )
    assert reused["extraction"]["status"] == "reused_receipt"
    assert reused["extraction"]["content_revalidated"] is False


def test_public_dataset_rejects_unsafe_zip_member(tmp_path: Path):
    source = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(source, "w") as bundle:
        bundle.writestr("../escape.txt", b"escape")
    destination = tmp_path / "datasets"
    fixture = destination / "fixture"
    fixture.mkdir(parents=True)
    (fixture / source.name).write_bytes(source.read_bytes())

    with pytest.raises(RuntimeError, match="Unsafe public dataset ZIP member"):
        prepare_public_dataset(
            _registry(tmp_path / "registry.json", source),
            "fixture",
            destination,
        )


def test_public_dataset_rejects_nas_destination(tmp_path: Path):
    source = tmp_path / "source.zip"
    with zipfile.ZipFile(source, "w") as bundle:
        bundle.writestr("sample.txt", b"sample")

    with pytest.raises(RuntimeError, match="local non-NAS"):
        prepare_public_dataset(
            _registry(tmp_path / "registry.json", source),
            "fixture",
            Path("/home/x1/桌面/nas/forbidden"),
        )
