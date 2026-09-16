"""Exercise cold/warm portable verification without starting models or a desktop."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pytest


@pytest.fixture
def package(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    import rtx4050_integrity as integrity
    import rtx4050_portable as portable

    root = tmp_path / "package"
    root.mkdir()
    for name, data in {"app.py": b"print('ready')\n", "model.bin": b"model" * 10000}.items():
        (root / name).write_bytes(data)

    def seal():
        files = [{"path": name, "size_bytes": (root / name).stat().st_size,
                  "sha256": portable.sha256(root / name)} for name in ("app.py", "model.bin")]
        (root / "SHA256SUMS.json").write_text(json.dumps({"files": files, "file_count": len(files)}))

    seal()
    # Real filesystem timestamps must be outside the verifier's 1-second
    # ambiguity window before a native warm-cache hit is expected.
    time.sleep(1.05)
    reads = []
    original = portable.sha256

    def hash_file(path, **kwargs):
        reads.append(path.name)
        return original(path, **kwargs)

    monkeypatch.setattr(portable, "sha256", hash_file)
    def verify(**kw):
        return portable.verify_package(root, use_cache=True, **kw)

    def receipt():
        return json.loads((root / "Runtime/Logs/package-verification.json").read_text())

    return root, integrity, portable, verify, receipt, reads, seal


def test_second_launch_does_not_reread_unchanged_files(package):
    root, _, _, verify, receipt, reads, _ = package
    verify()
    assert reads == ["app.py", "model.bin"]
    assert receipt()["mode"] == "full"
    reads.clear()
    progress = []
    verify(progress=progress.append)
    assert reads == []
    assert receipt()["hashed_bytes"] == 0
    assert receipt()["reused_files"] == 2
    assert receipt()["fresh_full_hash_verification"] is False
    assert progress[-1]["checked_bytes"] == progress[-1]["total_bytes"]
    assert sorted(p.name for p in (root / "Runtime/Cache/PackageIntegrity").iterdir()) == ["authentication.key", "verified-files.json"]


def test_changed_source_is_rehashed_while_unchanged_model_is_reused(package):
    root, _, _, verify, receipt, reads, seal = package
    verify()
    (root / "app.py").write_bytes(b"print('updated')\n")
    seal()
    reads.clear()
    verify()
    assert reads == ["app.py"]
    assert receipt()["reused_bytes"] == (root / "model.bin").stat().st_size


@pytest.mark.parametrize("change", ["same_size", "restored_mtime", "missing", "replaced"])
def test_file_damage_cannot_reuse_a_previous_success(package, change):
    root, _, _, verify, _, _, _ = package
    verify()
    path = root / "model.bin"
    before = path.stat()
    if change == "missing":
        path.unlink()
    elif change == "replaced":
        temporary = root / "replacement"
        temporary.write_bytes(b"x" * before.st_size)
        os.replace(temporary, path)
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    else:
        path.write_bytes(b"x" * before.st_size)
        if change == "restored_mtime":
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises((OSError, RuntimeError)):
        verify()
    assert not (root / "Runtime/Cache/PackageIntegrity/verified-files.json").exists()


@pytest.mark.parametrize("value", [b"broken json", b'"not an object"', b'{"signature":"forged"}'])
def test_damaged_or_edited_cache_falls_back_to_full_hashing(package, value):
    root, _, _, verify, receipt, reads, _ = package
    verify()
    (root / "Runtime/Cache/PackageIntegrity/verified-files.json").write_bytes(value)
    reads.clear()
    verify()
    assert reads == ["app.py", "model.bin"]
    assert receipt()["fresh_full_hash_verification"] is True


def test_matching_file_metadata_without_valid_cache_authentication_is_not_trusted(package):
    root, integrity, _, verify, _, _, _ = package
    verify()
    path = root / "model.bin"
    path.write_bytes(b"x" * path.stat().st_size)
    cache_path = root / "Runtime/Cache/PackageIntegrity/verified-files.json"
    value = json.loads(cache_path.read_bytes())
    value["files"]["model.bin"]["identity"] = integrity.fingerprint(path)
    cache_path.write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match="校验失败"):
        verify()


def test_unreliable_change_metadata_requires_full_hashing(package, monkeypatch):
    _, integrity, _, verify, receipt, reads, _ = package
    monkeypatch.setattr(integrity, "fingerprint", lambda _: None)
    verify()
    reads.clear()
    verify()
    assert reads == ["app.py", "model.bin"]
    assert receipt()["reused_files"] == 0


def test_copying_a_cached_package_requires_new_verification(package, tmp_path):
    root, _, portable, verify, _, reads, _ = package
    verify()
    copy = tmp_path / "relocated"
    shutil.copytree(root, copy)
    reads.clear()
    portable.verify_package(copy, use_cache=True)
    assert reads == ["app.py", "model.bin"]


def test_explicit_full_check_refreshes_the_same_cache(package):
    _, _, _, verify, receipt, reads, _ = package
    verify()
    reads.clear()
    verify(force_full=True)
    assert reads == ["app.py", "model.bin"]
    assert receipt()["fresh_full_hash_verification"] is True
    reads.clear()
    verify()
    assert reads == []


def test_file_changes_during_hashing_cannot_create_a_cache(package, monkeypatch):
    root, _, portable, verify, _, _, _ = package

    def racing_hash(path, *, on_chunk):
        data = path.read_bytes()
        on_chunk(len(data))
        path.write_bytes(data + b"changed")
        return hashlib.sha256(data).hexdigest()

    monkeypatch.setattr(portable, "sha256", racing_hash)
    with pytest.raises(RuntimeError, match="发生变化"):
        verify()
    assert not (root / "Runtime/Cache/PackageIntegrity/verified-files.json").exists()


def test_unwritable_cache_does_not_block_a_valid_package(package, monkeypatch):
    _, integrity, _, verify, _, reads, _ = package
    monkeypatch.setattr(integrity, "_atomic", lambda *_: (_ for _ in ()).throw(PermissionError()))
    verify()
    reads.clear()
    verify()
    assert reads == ["app.py", "model.bin"]


def test_recent_or_future_timestamps_are_never_reused(package, monkeypatch):
    _, integrity, _, verify, receipt, reads, _ = package
    verify()
    monkeypatch.setattr(integrity, "_settled", lambda _: False)
    reads.clear()
    verify()
    assert reads == ["app.py", "model.bin"]
    assert receipt()["reused_files"] == 0


def test_a_separate_launch_process_reuses_the_verified_files(package):
    root, _, _, verify, receipt, _, _ = package
    verify()
    script = """
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from rtx4050_portable import verify_package
verify_package(Path(sys.argv[2]), use_cache=True)
"""
    subprocess.run([sys.executable, "-I", "-B", "-c", script,
                    str(Path(__file__).resolve().parents[1] / "tools"), str(root)], check=True, timeout=15)
    assert receipt()["hashed_bytes"] == 0
    assert receipt()["reused_files"] == 2


def test_manual_full_verification_does_not_start_hardware_or_ai(package, monkeypatch):
    root, _, portable, _, receipt, _, _ = package
    monkeypatch.setattr(portable, "ROOT", root)
    monkeypatch.setattr(sys, "argv", ["portable", "--verify-package-only"])
    monkeypatch.setattr(portable, "hardware_preflight", lambda: pytest.fail("must not start hardware"))
    monkeypatch.setattr(portable, "configure_environment", lambda _: pytest.fail("must not prepare models"))
    monkeypatch.chdir(root)
    assert portable.main() == 0
    assert receipt()["fresh_full_hash_verification"] is True


def test_cache_key_for_another_account_is_replaced_after_full_verification(package, monkeypatch):
    _, integrity, _, verify, receipt, reads, _ = package
    verify()
    original = integrity._protect
    fail_once = [True]

    def another_account(data, *, decrypt=False):
        if decrypt and fail_once[0]:
            fail_once[0] = False
            raise OSError("synthetic account change")
        return original(data, decrypt=decrypt)

    monkeypatch.setattr(integrity, "_protect", another_account)
    reads.clear()
    verify()
    assert receipt()["fresh_full_hash_verification"] is True
    assert reads == ["app.py", "model.bin"]
    reads.clear()
    verify()
    assert reads == []


def test_verifier_update_requires_a_new_full_verification(package, monkeypatch):
    _, integrity, _, verify, receipt, reads, _ = package
    verify()
    monkeypatch.setattr(integrity, "VERIFIER_SHA256", "0" * 64)
    reads.clear()
    verify()
    assert reads == ["app.py", "model.bin"]
    assert receipt()["fresh_full_hash_verification"] is True
