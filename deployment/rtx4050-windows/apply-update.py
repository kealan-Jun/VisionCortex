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
    target = (root / name).resolve()
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
            if digest(safe(root, item["path"])) != item["after_sha256"]:
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
        if digest(safe(root, name)) != item["before_sha256"]:
            raise RuntimeError("Original application files were changed. No files changed.")
        if digest(safe(patch / "payload", name)) != item["after_sha256"]:
            raise RuntimeError("Update payload is damaged. No files changed.")
    manifest_record = next(item for item in records if item["path"] == "SHA256SUMS.json")
    if manifest_record["after_sha256"] != spec["updated_manifest_sha256"]:
        raise RuntimeError("Updated manifest identity mismatch")
    updated = json.loads((patch / "payload/SHA256SUMS.json").read_text(encoding="utf-8"))
    expected = {name: dict(item) for name, item in indexed.items()}
    for item in records:
        if item["path"] != "SHA256SUMS.json":
            expected[item["path"]].update(sha256=item["after_sha256"], size_bytes=safe(patch / "payload", item["path"]).stat().st_size)
    if updated["files"] != [expected[item["path"]] for item in original["files"]] or updated["file_count"] != original["file_count"]:
        raise RuntimeError("Update unexpectedly changes unrelated manifest entries")
    updates = safe(root, "Runtime/Updates")
    updates.mkdir(parents=True, exist_ok=True)
    backup = Path(tempfile.mkdtemp(prefix="connection-", dir=updates))
    ordered = sorted(records, key=lambda item: item["path"] == "SHA256SUMS.json")
    # Back up and stage everything before the first replacement. Seal manifest last.
    for item in ordered:
        name = item["path"]
        for folder, source in [("before", safe(root, name)), ("staged", safe(patch / "payload", name))]:
            target = safe(backup / folder, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    replaced = []
    try:
        for item in ordered:
            name = item["path"]
            os.replace(safe(backup / "staged", name), safe(root, name))
            replaced.append(name)
        for item in ordered:
            if digest(safe(root, item["path"])) != item["after_sha256"]:
                raise RuntimeError("Installed update checksum mismatch")
    except Exception:
        for name in reversed(replaced):
            shutil.copyfile(safe(backup / "before", name), safe(root, name))
        (backup / "FAILED.txt").write_text("Update failed; original files restored.\n", encoding="utf-8")
        raise
    receipt = {"status": "applied", "base_manifest_sha256": current,
               "updated_manifest_sha256": spec["updated_manifest_sha256"], "backup": str(backup),
               "windows_runtime": "NOT_PROVEN", "real_video_quality": "NOT_PROVEN"}
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
