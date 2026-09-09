from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="Git is required for source updates")


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def commit(source):
    subprocess.run(["git", "-C", str(source), "add", "--", "src", "configs", "tools", "deployment", "docs", "README.md", "AGENTS.md", "pyproject.toml"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(source), "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                    "commit", "-m", "synthetic source revision"], check=True, capture_output=True)


def seal(root, update):
    records = [{"path": p.relative_to(root).as_posix(), "size_bytes": p.stat().st_size,
                "sha256": update.digest(p.read_bytes())}
               for p in sorted(root.rglob("*")) if p.is_file() and p.name != "SHA256SUMS.json"
               and "Runtime" not in p.relative_to(root).parts]
    (root / "SHA256SUMS.json").write_bytes(update.encoded({"files": records, "file_count": len(records),
                                                         "total_bytes": sum(r["size_bytes"] for r in records)}))


@pytest.fixture
def installation(tmp_path):
    update = load(ROOT / "tools/update_rtx4050_source.py", "source_update_test")
    apply = load(ROOT / "deployment/rtx4050-windows/apply-update.py", "source_apply_test")
    source, root, patch = [tmp_path / n for n in ("git", "installed", "patch")]
    source.mkdir()
    root.mkdir()
    patch.mkdir()
    contents = {name: b"synthetic source\n" for name in update.MAPPINGS}
    contents.update({name: b"# synthetic executor\n" for name in update.EXECUTORS})
    contents.update({"pyproject.toml": b'[project]\ndependencies = []\n',
                     "deployment/rtx4050-windows/assets-lock.json": b'{}\n',
                     "src/visioncortex/keep.py": b'identity = "retained"\n',
                     "src/visioncortex/new.py": b'identity = "new"\n',
                     "configs/profile.yaml": b'{}\n'})
    previous = []
    for name, data in contents.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        if name in update.EXECUTORS:
            continue
        if name in {"src/visioncortex/new.py", "tools/rtx4050_hardware.py"}:
            continue
        target = update.MAPPINGS.get(name, name)
        path = root / target
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        previous.append({"path": name, "package_path": target, "sha256": update.digest(data)})
    removed = root / "src/visioncortex/removed.py"
    removed.write_bytes(b"old source\n")
    previous.append({"path": "src/visioncortex/removed.py", "package_path": "src/visioncortex/removed.py"})
    (root / "BUNDLE-METADATA.json").write_bytes(update.encoded({"entry_point": "VisionCortex.exe"}))
    (root / "receipts/source-snapshot.json").write_bytes(update.encoded({"source_files": previous}))
    for name in ("models/model.bin", "python/python.exe", "VisionCortex.exe", "Runtime/user-data.txt"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"unchanged\n")
    seal(root, update)
    subprocess.run(["git", "init", str(source)], check=True, capture_output=True)
    commit(source)
    return update, apply, source, root, patch


def test_git_update_adds_removes_and_preserves_runtime_then_is_idempotent(installation):
    update, apply, source, root, patch = installation
    update.prepare(root, source, patch)
    result = apply.apply(root, patch)
    assert result["status"] == "applied"
    assert (root / "src/visioncortex/new.py").is_file()
    assert (root / "tools/rtx4050_hardware.py").is_file()
    assert not (root / "src/visioncortex/removed.py").exists()
    for name in ("models/model.bin", "python/python.exe", "Runtime/user-data.txt"):
        assert (root / name).read_bytes() == b"unchanged\n"
    manifest = json.loads((root / "SHA256SUMS.json").read_text(encoding="utf-8"))
    for record in manifest["files"]:
        assert update.digest((root / record["path"]).read_bytes()) == record["sha256"]
    backups = list((root / "Runtime/Updates").iterdir())
    assert update.prepare(root, source, patch)["status"] == "already_applied"
    assert list((root / "Runtime/Updates").iterdir()) == backups


def test_failure_before_manifest_replacement_restores_added_and_deleted_sources(installation, monkeypatch):
    update, apply, source, root, patch = installation
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    update.prepare(root, source, patch)
    replace = apply.os.replace

    def fail(source, target):
        if target.name == "SHA256SUMS.json":
            raise PermissionError("synthetic locked manifest")
        replace(source, target)

    monkeypatch.setattr(apply.os, "replace", fail)
    with pytest.raises(PermissionError):
        apply.apply(root, patch)
    for path, content in before.items():
        assert (root / path).read_bytes() == content
    assert not (root / "src/visioncortex/new.py").exists()
    assert not (root / "tools/rtx4050_hardware.py").exists()


@pytest.mark.parametrize("kind", ["dirty_git", "runtime_change", "python_change", "tampered_source", "unexpected_mapping"])
def test_unverifiable_updates_fail_before_changing_installed_files(installation, kind):
    update, apply, source, root, patch = installation
    if kind == "dirty_git":
        (source / "src/visioncortex/keep.py").write_bytes(b"uncommitted edit")
    elif kind == "runtime_change":
        (source / "deployment/rtx4050-windows/assets-lock.json").write_bytes(b'{"new-runtime":true}\n')
        commit(source)
    elif kind == "python_change":
        (source / "pyproject.toml").write_bytes(b'[project]\ndependencies = []\nrequires-python = ">=3.13"\n')
        commit(source)
    elif kind == "tampered_source":
        (root / "src/visioncortex/keep.py").write_bytes(b"unexpected edit")
    else:
        record = json.loads((root / "receipts/source-snapshot.json").read_text(encoding="utf-8"))
        record["source_files"][-1]["package_path"] = "../../escape.py"
        (root / "receipts/source-snapshot.json").write_bytes(update.encoded(record))
        seal(root, update)
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    with pytest.raises(RuntimeError):
        update.prepare(root, source, patch)
    assert {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()} == before
    assert not (root / "Runtime/Updates").exists()


def test_source_update_rejects_path_traversal(installation):
    update, apply, source, root, patch = installation
    for name in ("../escape", "/absolute", "src\\escape.py", "src/alternate:stream.py"):
        with pytest.raises(RuntimeError):
            update.safe(patch, name)
        with pytest.raises(RuntimeError):
            apply.safe(root, name)


def test_update_executor_is_committed_and_staged_before_use(installation):
    update, _, source, root, patch = installation
    update.prepare(root, source, patch)
    assert (patch / "apply-update.py").read_bytes() == (source / update.EXECUTORS[1]).read_bytes()
    executor = source / update.EXECUTORS[1]
    executor.unlink()
    commit(source)
    executor.write_bytes(b"# untracked replacement\n")
    before = (root / "SHA256SUMS.json").read_bytes()
    with pytest.raises(RuntimeError, match="executor"):
        update.prepare(root, source, patch)
    assert (root / "SHA256SUMS.json").read_bytes() == before


def test_windows_crlf_checkout_keeps_the_committed_executor(installation):
    update, _, source, root, patch = installation
    checkout = source.with_name("windows-checkout")
    subprocess.run(["git", "clone", "--config", "core.autocrlf=true", "--no-local", str(source), str(checkout)],
                   check=True, capture_output=True)
    assert b"\r\n" in (checkout / update.EXECUTORS[0]).read_bytes()
    update.prepare(root, checkout, patch)
    assert (patch / "apply-update.py").read_bytes() == b"# synthetic executor\n"
