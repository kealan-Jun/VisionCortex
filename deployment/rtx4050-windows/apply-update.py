"""Apply a manifest-bound desktop update using only bundled Python's stdlib."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def safe(root, name):
    pure = PurePosixPath(name)
    if not name or pure.is_absolute() or ".." in pure.parts or "\\" in name or ":" in name:
        raise RuntimeError("Invalid update path")
    unresolved = root.resolve() / pure
    for entry in (unresolved, *unresolved.parents):
        if entry == root.resolve():
            break
        if entry.is_symlink() or (hasattr(entry, "is_junction") and entry.is_junction()):
            raise RuntimeError("Update paths cannot contain links")
    target = unresolved.resolve()
    if root.resolve() not in target.parents:
        raise RuntimeError("Update path escapes application")
    return target


def apply(root, patch):
    root, patch = root.resolve(), patch.resolve()
    spec = json.loads((patch / "update.json").read_text(encoding="utf-8"))
    records = spec["files"]
    names = [item["path"].casefold() for item in records]
    if len(names) != len(set(names)) or names.count("sha256sums.json") != 1:
        raise RuntimeError("Invalid update manifest")
    if any(name.startswith("runtime/") for name in names):
        raise RuntimeError("Updates cannot replace user data")
    current = digest(root / "SHA256SUMS.json")
    if current == spec["updated_manifest_sha256"]:
        for item in records:
            path = safe(root, item["path"])
            actual = digest(path) if path.is_file() else None
            if actual != item["after_sha256"]:
                raise RuntimeError("Installed update files are damaged")
        return {"status": "already_applied"}
    if current != spec["base_manifest_sha256"]:
        raise RuntimeError("This update does not match your application version. No files changed.")
    original = json.loads((root / "SHA256SUMS.json").read_text(encoding="utf-8"))
    indexed = {item["path"]: item for item in original["files"]}
    for item in records:
        name = item["path"]
        if name != "SHA256SUMS.json" and indexed.get(name, {}).get("sha256") != item["before_sha256"]:
            raise RuntimeError("Update is not bound to the original file manifest")
        path = safe(root, name)
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise RuntimeError("Update target must be a regular file")
        actual = digest(path) if path.is_file() else None
        if actual != item["before_sha256"]:
            raise RuntimeError("Original application files were changed. No files changed.")
        payload = safe(patch / "payload", name)
        after = digest(payload) if payload.is_file() else None
        if after != item["after_sha256"]:
            raise RuntimeError("Update payload is damaged. No files changed.")
        if item["before_sha256"] is None and item["after_sha256"] is None:
            raise RuntimeError("Update entry cannot be empty")
    manifest_record = next(item for item in records if item["path"] == "SHA256SUMS.json")
    if manifest_record["after_sha256"] != spec["updated_manifest_sha256"]:
        raise RuntimeError("Updated manifest identity mismatch")
    updated = json.loads((patch / "payload/SHA256SUMS.json").read_text(encoding="utf-8"))
    expected = {name: dict(item) for name, item in indexed.items()}
    for item in records:
        if item["path"] != "SHA256SUMS.json":
            if item["after_sha256"] is None:
                expected.pop(item["path"], None)
            else:
                expected[item["path"]] = dict(path=item["path"], sha256=item["after_sha256"],
                                             size_bytes=safe(patch / "payload", item["path"]).stat().st_size)
    if (updated["files"] != list(expected.values()) or updated["file_count"] != len(expected)
            or updated.get("total_bytes") != sum(item["size_bytes"] for item in expected.values())):
        raise RuntimeError("Update unexpectedly changes unrelated manifest entries")
    updates = safe(root, "Runtime/Updates")
    updates.mkdir(parents=True, exist_ok=True)
    backup = Path(tempfile.mkdtemp(prefix="source-" if spec.get("kind") == "git_source_update" else "connection-", dir=updates))
    ordered = sorted(records, key=lambda item: item["path"] == "SHA256SUMS.json")
    # Back up and stage everything before the first replacement. Seal manifest last.
    for item in ordered:
        name = item["path"]
        for folder, source in [("before", safe(root, name)), ("staged", safe(patch / "payload", name))]:
            if not source.is_file():
                continue
            target = safe(backup / folder, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    replaced = []
    try:
        for item in ordered:
            name = item["path"]
            target = safe(root, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            if item["after_sha256"] is None:
                target.unlink()
            else:
                os.replace(safe(backup / "staged", name), target)
            replaced.append(name)
        for item in ordered:
            target = safe(root, item["path"])
            actual = digest(target) if target.is_file() else None
            if actual != item["after_sha256"]:
                raise RuntimeError("Installed update checksum mismatch")
    except Exception:
        for name in reversed(replaced):
            before = safe(backup / "before", name)
            if before.is_file():
                shutil.copyfile(before, safe(root, name))
            else:
                safe(root, name).unlink(missing_ok=True)
        (backup / "FAILED.txt").write_text("Update failed; original files restored.\n", encoding="utf-8")
        raise
    receipt = {"status": "applied", "base_manifest_sha256": current,
               "updated_manifest_sha256": spec["updated_manifest_sha256"], "backup": str(backup),
               "windows_runtime": "NOT_PROVEN", "real_video_quality": "NOT_PROVEN"}
    if spec.get("source_git_commit"):
        receipt["source_git_commit"] = spec["source_git_commit"]
    (backup / "result.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(apply(args.root, Path(__file__).resolve().parent), ensure_ascii=True))
    except Exception as error:
        parser.exit(1, f"Update failed: {error}\n")
