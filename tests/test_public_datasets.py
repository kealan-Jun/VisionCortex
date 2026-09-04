from __future__ import annotations

import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

import pytest

from visioncortex.public_datasets import (
    _safe_extract_rar,
    _validate_rar_listing,
    prepare_public_dataset,
)


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
    assert not (fixture / "extracted").exists()


def test_public_dataset_rejects_windows_drive_zip_member(tmp_path: Path):
    source = tmp_path / "unsafe-drive.zip"
    with zipfile.ZipFile(source, "w") as bundle:
        bundle.writestr("C:/escape.txt", b"escape")
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


def test_public_dataset_zip_extraction_publishes_atomically(tmp_path: Path):
    source = tmp_path / "source.zip"
    with zipfile.ZipFile(source, "w") as bundle:
        bundle.writestr("dataset/sample.txt", b"sample")
    destination = tmp_path / "datasets"
    fixture = destination / "fixture"
    fixture.mkdir(parents=True)
    (fixture / source.name).write_bytes(source.read_bytes())

    receipt = prepare_public_dataset(
        _registry(tmp_path / "registry.json", source),
        "fixture",
        destination,
    )

    assert receipt["extraction"]["file_count"] == 1
    assert receipt["extraction"]["uncompressed_bytes"] == 6
    assert (fixture / "extracted" / "dataset" / "sample.txt").read_bytes() == b"sample"
    assert not list(fixture.glob(".extracted.partial-*"))


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


def test_rar_listing_rejects_links_and_path_traversal(monkeypatch, tmp_path):
    outputs = iter(
        [
            subprocess.CompletedProcess([], 0, "../escape.txt\n", ""),
            subprocess.CompletedProcess([], 0, "-rw-r--r-- 0 0 0 4 Jan 1 2025 ../escape.txt\n", ""),
        ]
    )
    monkeypatch.setattr(
        "visioncortex.public_datasets._run_archive_tool",
        lambda command: next(outputs),
    )

    with pytest.raises(RuntimeError, match="Unsafe public dataset RAR member"):
        _validate_rar_listing(tmp_path / "fixture.rar", "bsdtar", 100)


def test_safe_rar_extraction_is_atomic_and_size_checked(monkeypatch, tmp_path):
    archive = tmp_path / "fixture.rar"
    archive.write_bytes(b"rar")
    destination = tmp_path / "extracted"

    def fake_listing(_archive, _tool, _maximum):
        return ["dataset/", "dataset/sample.txt"], 4, 1

    def fake_run(command):
        if "-xmf" in command:
            root = Path(command[command.index("-C") + 1]) / "dataset"
            root.mkdir()
            (root / "sample.txt").write_bytes(b"data")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(
        "visioncortex.public_datasets.shutil.which", lambda name: "/bin/bsdtar"
    )
    monkeypatch.setattr(
        "visioncortex.public_datasets._validate_rar_listing", fake_listing
    )
    monkeypatch.setattr(
        "visioncortex.public_datasets._run_archive_tool", fake_run
    )

    receipt = _safe_extract_rar(
        archive,
        destination,
        expected_archive_sha256="a" * 64,
        maximum_uncompressed_bytes=100,
    )

    assert receipt["archive_format"] == "rar"
    assert receipt["file_count"] == 1
    assert receipt["uncompressed_bytes"] == 4
    assert (destination / "dataset" / "sample.txt").read_bytes() == b"data"
