import hashlib
import json
import os
from pathlib import Path

import pytest
import yaml

from visioncortex.source_path_migrations import (
    SourcePathResolver,
    create_migration_receipt,
    load_migration_receipt,
    resolve_manifest_copy,
)


def _write_receipt(tmp_path: Path) -> tuple[Path, Path, Path]:
    old_root = tmp_path / "old"
    new_root = tmp_path / "new"
    target = new_root / "camera" / "segment.mp4"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"immutable-source")
    payload = create_migration_receipt(old_root, new_root)
    receipt = tmp_path / "migration.json"
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    return receipt, old_root / "camera" / "segment.mp4", target


def test_exact_migration_resolves_and_verifies_source_identity(tmp_path):
    receipt, old_path, target = _write_receipt(tmp_path)
    resolver = SourcePathResolver([receipt])

    assert resolver.resolve(old_path) == target
    resolution = resolver.resolution_receipt(subject="test")
    assert resolution["status"] == "verified"
    assert resolution["resolved_path_count"] == 1
    assert resolution["records"][0]["sha256"] == hashlib.sha256(
        target.read_bytes()
    ).hexdigest()


def test_existing_original_path_wins_without_relocation(tmp_path):
    receipt, old_path, _target = _write_receipt(tmp_path)
    old_path.parent.mkdir(parents=True)
    old_path.write_bytes(b"original-still-present")
    resolver = SourcePathResolver([receipt])

    assert resolver.resolve(old_path) == old_path
    assert resolver.resolution_receipt()["status"] == "not_needed"


def test_changed_mapped_source_fails_closed(tmp_path):
    receipt, old_path, target = _write_receipt(tmp_path)
    resolver = SourcePathResolver([receipt])
    target.write_bytes(b"changed-after-receipt")

    with pytest.raises(ValueError, match="size mismatch|SHA-256 mismatch"):
        resolver.resolve(old_path)


def test_same_size_and_mtime_content_change_still_fails_closed(tmp_path):
    receipt, old_path, target = _write_receipt(tmp_path)
    resolver = SourcePathResolver([receipt])
    before = target.stat()
    target.write_bytes(b"tampered-source!")
    assert target.stat().st_size == before.st_size
    os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns))

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        resolver.resolve(old_path)


def test_changed_mapping_digest_fails_closed(tmp_path):
    receipt, _old_path, _target = _write_receipt(tmp_path)
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["entries"][0]["new_path"] += ".different"
    receipt.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="relative path mismatch|mapping digest mismatch"):
        load_migration_receipt(receipt)


def test_manifest_replay_writes_copy_and_preserves_frozen_manifest(tmp_path):
    receipt, old_path, target = _write_receipt(tmp_path)
    source_manifest = tmp_path / "archive" / "manifest.yaml"
    source_manifest.parent.mkdir()
    source_manifest.write_text(
        yaml.safe_dump(
            {
                "experiment_id": "historical",
                "views": [
                    {
                        "view_id": "fp",
                        "role": "first_person",
                        "segments": [{"video": str(old_path), "timestamps_csv": None}],
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    frozen_hash = hashlib.sha256(source_manifest.read_bytes()).hexdigest()

    resolved, resolution_path, resolution = resolve_manifest_copy(
        source_manifest, tmp_path / "replay", [receipt]
    )

    assert hashlib.sha256(source_manifest.read_bytes()).hexdigest() == frozen_hash
    assert yaml.safe_load(resolved.read_text(encoding="utf-8"))["views"][0][
        "segments"
    ][0]["video"] == str(target)
    assert resolution["original_manifest_sha256"] == frozen_hash
    assert json.loads(resolution_path.read_text(encoding="utf-8"))[
        "resolved_path_count"
    ] == 1
