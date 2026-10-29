"""Synthetic populated archive binding, anchored storage guards; no real NAS."""

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

import pytest

from visioncortex.lab import (
    archive_layout,
    existing_archive_binding as binding,
    storage,
)
from visioncortex.lab.components import pipeline_module
from visioncortex.lab.config import Settings
from visioncortex.lab.domain import Conflict
from visioncortex.lab.scan_source import ScanSourcePublisher

from .test_scan_source import configuration, metadata, pixels, session


def write(path, value):
    path.write_text(json.dumps(value))
    path.chmod(0o600)


@pytest.fixture
def bound(tmp_path, monkeypatch):
    root = tmp_path / "archive"
    raw = root / "2026-10-29_fixture-camera-0/MetaVideo/original.bin"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"immutable recorder fixture")
    guide = root / "DataRetentionGuide.docx"
    guide.write_bytes(b"neutral synthetic existing guide; not a real user document")
    manifest = tmp_path / "binding.json"
    document = dict(
        schema_version=binding.SCHEMA,
        bindings=[
            dict(
                mode="existing_native_root",
                archive_root=str(root),
                guide_relative_path=guide.name,
                guide_sha256=hashlib.sha256(guide.read_bytes()).hexdigest(),
                installed_contract_sha256=hashlib.sha256(
                    archive_layout.contract_bytes()
                ).hexdigest(),
                layout=deepcopy(binding.LAYOUT),
            )
        ],
    )
    write(manifest, document)
    monkeypatch.setenv(binding.ENV, str(manifest))
    return dict(root=root, guide=guide, manifest=manifest, document=document, raw=raw)


def test_explicit_populated_root_binding_is_read_only_and_keeps_existing_layout(bound):
    f = bound
    before = {
        str(p.relative_to(f["root"])): p.read_bytes()
        for p in f["root"].rglob("*")
        if p.is_file()
    }
    assert archive_layout.verify_archive_contract(f["root"]) == f["guide"]
    assert before == {
        str(p.relative_to(f["root"])): p.read_bytes()
        for p in f["root"].rglob("*")
        if p.is_file()
    }
    assert not (f["root"] / ".System").exists()


def test_existing_native_guide_cannot_be_used_as_empty_archive_bootstrap(bound):
    f = bound
    settings = Settings(
        database_url="postgresql://fixture/unused",
        runtime_root=f["root"].parent,
        customer_code="fixture",
        source_root=f["root"],
        unified_archive_required=True,
    )
    before = f["guide"].read_bytes()
    with pytest.raises(Conflict, match="empty bootstrap"):
        archive_layout.prepare_archive(settings)
    assert f["guide"].read_bytes() == before
    assert not (f["root"] / ".System").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("archive_root", "/fixture/unrelated"),
        ("guide_sha256", "0" * 64),
        ("installed_contract_sha256", "1" * 64),
        ("guide_relative_path", "../guide.docx"),
        ("unexpected", "reject"),
    ],
)
def test_exact_private_root_hashes_paths_and_unknown_fields_fail_closed(
    bound, field, value
):
    document = deepcopy(bound["document"])
    document["bindings"][0][field] = value
    write(bound["manifest"], document)
    with pytest.raises(Conflict):
        archive_layout.verify_archive_contract(bound["root"])


def test_layout_and_existing_wrong_contract_cannot_fall_back_to_guide(bound):
    document = deepcopy(bound["document"])
    document["bindings"][0]["layout"]["result_directories"].append("NewResult")
    write(bound["manifest"], document)
    with pytest.raises(Conflict):
        archive_layout.verify_archive_contract(bound["root"])
    write(bound["manifest"], bound["document"])
    (bound["root"] / ".System").mkdir()
    (bound["root"] / archive_layout.CONTRACT_PATH).write_bytes(
        b"wrong existing contract"
    )
    with pytest.raises(Conflict):
        archive_layout.verify_archive_contract(bound["root"])


@pytest.mark.parametrize("kind", ["mode", "owner", "symlink", "oversized"])
def test_manifest_is_private_owned_bounded_and_canonical(bound, monkeypatch, kind):
    f = bound
    if kind == "mode":
        f["manifest"].chmod(0o644)
    elif kind == "owner":
        monkeypatch.setattr(
            binding.os, "getuid", lambda: os.stat(f["manifest"]).st_uid + 1
        )
    elif kind == "symlink":
        link = f["manifest"].with_name("alias.json")
        link.symlink_to(f["manifest"])
        monkeypatch.setenv(binding.ENV, str(link))
    else:
        f["manifest"].write_bytes(b" " * 65537)
    with pytest.raises(Conflict):
        archive_layout.verify_archive_contract(f["root"])


def test_guide_symlink_and_between_read_mutation_are_rejected(bound, monkeypatch):
    f = bound
    original = f["guide"].read_bytes()
    target = f["guide"].with_name("other.docx")
    target.write_bytes(original)
    f["guide"].unlink()
    f["guide"].symlink_to(target)
    with pytest.raises(Conflict):
        archive_layout.verify_archive_contract(f["root"])
    f["guide"].unlink()
    f["guide"].write_bytes(original)
    read = binding._read
    calls = []

    def mutate(root, relative, *args, **kwargs):
        raw = read(root, relative, *args, **kwargs)
        if relative == f["guide"].name and not calls:
            calls.append(1)
            f["guide"].write_bytes(b"changed after first stable read")
        return raw

    monkeypatch.setattr(binding, "_read", mutate)
    with pytest.raises(Conflict, match="changed"):
        archive_layout.verify_archive_contract(f["root"])


def synthetic_mounts(f, tmp_path, monkeypatch):
    # The bytes and dirfds are real local fixtures. Only kernel/findmnt metadata
    # is simulated, so this is never a real CIFS/field acceptance claim.
    original_root = f["root"]
    new_root = tmp_path / "archive-mount/native-root"
    new_root.parent.mkdir()
    original_root.rename(new_root)
    f["root"] = new_root
    f["guide"] = new_root / "DataRetentionGuide.docx"
    f["raw"] = new_root / f["raw"].relative_to(original_root)
    f["document"]["bindings"][0]["archive_root"] = str(new_root)
    write(f["manifest"], f["document"])
    bindings = []
    for label, mount, project in [
        ("host", tmp_path / "host-mount", tmp_path / "host-mount/project"),
        ("archive", f["root"].parent, f["root"]),
    ]:
        project.mkdir(parents=True, exist_ok=True)
        marker = mount / (label + "-marker.json")
        write(marker, dict(id="fixture-" + label))
        bindings.append(
            dict(
                mount_path=str(mount),
                share="//fixture.invalid/" + label,
                project_root=str(project),
                marker_path=str(marker),
                marker_sha256=hashlib.sha256(marker.read_bytes()).hexdigest(),
            )
        )
    manifest = tmp_path / "storage.json"
    write(manifest, dict(schema_version="vc-lab-storage/1", bindings=bindings))
    monkeypatch.setenv("VC_LAB_STORAGE_MANIFEST", str(manifest))
    original = storage._mount_info

    def info(path):
        matches = [
            b for b in bindings if Path(path).is_relative_to(Path(b["mount_path"]))
        ]
        if not matches:
            return original(path)
        b = max(matches, key=lambda x: len(x["mount_path"]))
        return [dict(target=b["mount_path"], source=b["share"], fstype="cifs")]

    monkeypatch.setattr(storage, "_mount_info", info)
    guard = pipeline_module("device_day_write_guard")
    drift = []

    def snapshot():
        return {
            Path(b["mount_path"]): dict(
                mount_id=str(i + 1) + ("changed" if drift else ""),
                device=os.stat(b["mount_path"]).st_dev,
                filesystem="cifs",
                source=b["share"],
            )
            for i, b in enumerate(bindings)
        }

    monkeypatch.setattr(guard, "mount_snapshot", snapshot)
    recovery = pipeline_module("device_day_recovery_worker")
    monkeypatch.setattr(
        recovery, "verify_storage", lambda config: storage.verify_required_roots()
    )
    return bindings, drift


def test_current_mount_and_host_anchor_guards_are_not_bypassed(
    bound, tmp_path, monkeypatch
):
    bindings, drift = synthetic_mounts(bound, tmp_path, monkeypatch)
    assert archive_layout.verify_archive_contract(bound["root"]) == bound["guide"]
    marker = Path(bindings[0]["marker_path"])
    before = marker.read_bytes()
    marker.write_bytes(b"wrong host anchor")
    with pytest.raises(OSError):
        archive_layout.verify_archive_contract(bound["root"])
    marker.write_bytes(before)
    read = binding._read

    def detach(*args, **kwargs):
        raw = read(*args, **kwargs)
        drift.append(1)
        return raw

    monkeypatch.setattr(binding, "_read", detach)
    with pytest.raises(Conflict):
        archive_layout.verify_archive_contract(bound["root"])


def test_adopted_identity_file_allows_only_new_source_frame_with_pinned_guard(
    bound, tmp_path, monkeypatch
):
    synthetic_mounts(bound, tmp_path, monkeypatch)
    document = configuration(tmp_path / "source-runtime")
    document["incoming_root"] = str(bound["root"])
    publisher = ScanSourcePublisher(document)
    before = bound["raw"].read_bytes()
    guide = bound["guide"].read_bytes()
    image = pixels(["11111111-1111-4111-8111-111111111111"])
    from field_recognition.capture.qr_decode import decode_qr

    result = publisher.publish(image, metadata(), session(), decode_qr(image))
    path = bound["root"] / result["relative_path"]
    assert path.is_file() and "/MetaData/VideoEvidence/" in result["relative_path"]
    assert path.with_name("CaptureReceipt.json").is_file()
    assert bound["raw"].read_bytes() == before and bound["guide"].read_bytes() == guide
    assert not (bound["root"] / archive_layout.CONTRACT_PATH).exists()
