"""Update application text from a Git commit, retaining the installed offline runtime."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import subprocess
import tempfile
import tomllib


EXECUTORS = ("tools/update_rtx4050_source.py", "deployment/rtx4050-windows/apply-update.py")


MAPPINGS = {
    "README.md": "SOURCE-README.md", "pyproject.toml": "pyproject.toml", "AGENTS.md": "AGENTS.md",
    "docs/DUAL-REPOSITORY-RELEASE-POLICY.md": "docs/DUAL-REPOSITORY-RELEASE-POLICY.md",
    "docs/RTX4050-GIT-UPDATES.md": "docs/RTX4050-GIT-UPDATES.md",
    "deployment/rtx4050-windows/README.md": "README.md",
    "deployment/rtx4050-windows/assets-lock.json": "receipts/assets-lock.json",
    "deployment/rtx4050-windows/desktop/sitecustomize.py": "python/Lib/sitecustomize.py",
}
for _name in ("rtx4050_portable.py", "rtx4050_hardware.py", "desktop_storage.py",
              "windows_desktop_lifecycle.py", "verify_mllm_connection.py"):
    MAPPINGS[f"tools/{_name}"] = f"tools/{_name}"
for _name in ("package.json", "main.cjs", "controller.cjs", "connection.cjs", "storage.cjs",
              "preload.cjs", "setup.html", "setup.css", "setup.js", "ensure-vc-runtime.ps1"):
    MAPPINGS[f"deployment/rtx4050-windows/desktop/{_name}"] = f"resources/app/{_name}"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()


def git(source, *args):
    return subprocess.check_output(["git", "-C", str(source), *args], stderr=subprocess.PIPE)


def safe(root, name):
    relative = PurePosixPath(name)
    if not name or relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name:
        raise RuntimeError("Invalid source update path")
    path = root.resolve() / relative
    for part in (path, *path.parents):
        if part == root.resolve():
            break
        if part.is_symlink() or (hasattr(part, "is_junction") and part.is_junction()):
            raise RuntimeError("Source update paths cannot contain links")
    return path


def committed_source(source):
    revision = git(source, "rev-parse", "HEAD").decode().strip()
    # The scripts used to apply the update must also match the selected commit.
    if git(source, "status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("Source checkout has tracked edits. Commit them or use a clean checkout before updating.")
    selected = []
    for raw in git(source, "ls-tree", "-r", "-z", "HEAD").split(b"\0"):
        if not raw:
            continue
        header, name = raw.split(b"\t", 1)
        mode, kind, oid = header.decode().split()
        name = name.decode()
        if name not in MAPPINGS and name not in EXECUTORS and not name.startswith(("src/", "configs/", "examples/")):
            continue
        if mode not in {"100644", "100755"} or kind != "blob":
            raise RuntimeError("Source update requires regular committed files")
        if Path(name).suffix.lower() not in {".py", ".yaml", ".yml", ".json", ".csv", ".html", ".css", ".js", ".md", ".toml", ".cjs", ".ps1"}:
            raise RuntimeError("Unexpected source artifact: " + name)
        selected.append((name, oid))
    data = {}
    process = subprocess.Popen(["git", "-C", str(source), "cat-file", "--batch"],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        for name, oid in selected:
            process.stdin.write((oid + "\n").encode())
            process.stdin.flush()
            identity, kind, size = process.stdout.readline().decode().split()
            if identity != oid or kind != "blob" or int(size) > 8 * 1024**2:
                raise RuntimeError("Invalid or oversized source blob")
            content = process.stdout.read(int(size))
            if len(content) != int(size) or process.stdout.read(1) != b"\n":
                raise RuntimeError("Incomplete Git source blob")
            data[name] = content
        process.stdin.close()
        if process.wait(timeout=10):
            raise RuntimeError("Git source read failed")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        process.stdout.close()
        if not process.stdin.closed:
            process.stdin.close()
    if git(source, "rev-parse", "HEAD").decode().strip() != revision:
        raise RuntimeError("Source revision changed during update preparation")
    if not set(MAPPINGS) <= set(data):
        raise RuntimeError("Git revision is missing required desktop source files")
    if not set(EXECUTORS) <= set(data):
        raise RuntimeError("Git revision is missing a required update executor")
    executors = {name: data.pop(name) for name in EXECUTORS}
    for name, content in executors.items():
        checkout = safe(source, name).read_bytes()
        # Git may materialize text as CRLF in a clean Windows checkout.
        if checkout.replace(b"\r\n", b"\n") != content.replace(b"\r\n", b"\n"):
            raise RuntimeError("Update executor does not match the selected commit")
    return revision, data, executors


def prepare(root, source, patch):
    root, source = root.resolve(), source.resolve()
    if root == source or root in source.parents or source in root.parents:
        raise RuntimeError("Keep the Git checkout separate from the installed application")
    revision, files, executors = committed_source(source)
    original_bytes = (root / "SHA256SUMS.json").read_bytes()
    original = json.loads(original_bytes)
    indexed = {item["path"]: item for item in original["files"]}
    if len(indexed) != original["file_count"] or len({n.casefold() for n in indexed}) != len(indexed):
        raise RuntimeError("Invalid installed package manifest")

    def installed(name):
        data = safe(root, name).read_bytes()
        if digest(data) != indexed[name]["sha256"]:
            raise RuntimeError("Installed package identity mismatch: " + name)
        return data

    metadata = json.loads(installed("BUNDLE-METADATA.json"))
    previous = json.loads(installed("receipts/source-snapshot.json"))
    if metadata.get("entry_point") != "VisionCortex.exe":
        raise RuntimeError("Select an extracted RTX 4050 desktop application")
    if installed("receipts/assets-lock.json") != files["deployment/rtx4050-windows/assets-lock.json"]:
        raise RuntimeError("This revision changes the offline runtime assets; a compatible runtime update is required")
    old_project = tomllib.loads(installed("pyproject.toml").decode())["project"]
    new_project = tomllib.loads(files["pyproject.toml"].decode())["project"]
    if (old_project.get("requires-python") != new_project.get("requires-python")
            or old_project["dependencies"] != new_project["dependencies"] or any(
        old_project.get("optional-dependencies", {}).get(extra) != new_project.get("optional-dependencies", {}).get(extra)
        for extra in ("models", "tensorrt", "sam2")
    )):
        raise RuntimeError("This revision changes required runtime dependencies; a compatible runtime update is required")
    records = [{"path": name, "package_path": MAPPINGS.get(name, name), "sha256": digest(content)}
               for name, content in sorted(files.items())]
    if metadata.get("source_git_commit") == revision and previous.get("source_git_commit") == revision:
        if previous["source_files"] != records:
            raise RuntimeError("Installed source receipt does not match its Git revision")
        for record in records:
            if digest(installed(record["package_path"])) != record["sha256"]:
                raise RuntimeError("Installed source is damaged")
        return {"status": "already_applied", "source_git_commit": revision}
    desired = {MAPPINGS.get(name, name): content for name, content in files.items()}
    if len(desired) != len(records) or len({n.casefold() for n in desired}) != len(desired):
        raise RuntimeError("Source package mapping collision")
    snapshot = {"base_commit": revision, "source_git_commit": revision, "working_tree_snapshot": False,
                "source_files": records, "stable_release_readiness": "NOT_PROVEN",
                "delivery_revision": "git-" + revision[:12], "desktop_base_manifest_sha256": digest(original_bytes)}
    desired["receipts/source-snapshot.json"] = encoded(snapshot)
    metadata.update(source_base_commit=revision, source_git_commit=revision, working_tree_snapshot=False,
                    source_refreshed=True, delivery_revision=snapshot["delivery_revision"],
                    delivery_refresh_receipt="receipts/delivery-refresh.json", windows_runtime="NOT_PROVEN",
                    real_video_quality="NOT_PROVEN", stable_release_readiness="NOT_PROVEN")
    desired["BUNDLE-METADATA.json"] = encoded(metadata)
    desired["receipts/delivery-refresh.json"] = encoded({
        "kind": "git_source_update", "source_git_commit": revision,
        "revision": snapshot["delivery_revision"], "base_manifest_sha256": digest(original_bytes),
        "source_snapshot_sha256": digest(desired["receipts/source-snapshot.json"]),
        "source_from_committed_git_objects": "PROVEN", "windows_runtime": "NOT_PROVEN",
        "real_video_quality": "NOT_PROVEN", "stable_release_readiness": "NOT_PROVEN",
    })
    for item in previous["source_files"]:
        name = item["path"]
        target = item.get("package_path", name)
        if name in MAPPINGS or name.startswith(("src/", "configs/", "examples/")):
            if target != MAPPINGS.get(name, name):
                raise RuntimeError("Unexpected installed source mapping")
            if target not in desired:
                desired[target] = None
    changes = []
    expected = {name: dict(item) for name, item in indexed.items()}
    payload = patch / "payload"
    for name, content in sorted(desired.items()):
        current_path = safe(root, name)
        before = indexed.get(name, {}).get("sha256")
        after = digest(content) if content is not None else None
        if before is None:
            if current_path.exists():
                raise RuntimeError("Untracked application file blocks update: " + name)
        elif not current_path.is_file() or digest(current_path.read_bytes()) != before:
            raise RuntimeError("Installed source was changed: " + name)
        if before == after:
            continue
        changes.append({"path": name, "before_sha256": before, "after_sha256": after})
        if content is None:
            expected.pop(name)
        else:
            path = safe(payload, name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            expected[name] = {"path": name, "size_bytes": len(content), "sha256": after}
    updated = encoded({"files": list(expected.values()), "file_count": len(expected),
                       "total_bytes": sum(r["size_bytes"] for r in expected.values())})
    (payload / "SHA256SUMS.json").write_bytes(updated)
    changes.append({"path": "SHA256SUMS.json", "before_sha256": digest(original_bytes), "after_sha256": digest(updated)})
    spec = {"kind": "git_source_update", "source_git_commit": revision, "files": changes,
            "base_manifest_sha256": digest(original_bytes), "updated_manifest_sha256": digest(updated),
            "python_sha256": indexed["python/python.exe"]["sha256"]}
    (patch / "update.json").write_bytes(encoded(spec))
    # Execute the immutable blob we reviewed, not a later checkout version.
    (patch / "apply-update.py").write_bytes(executors["deployment/rtx4050-windows/apply-update.py"])
    return spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="visioncortex-source-update-") as directory:
        patch = Path(directory)
        spec = prepare(args.root, args.source, patch)
        if spec.get("status") == "already_applied":
            print(json.dumps(spec))
            return
        if args.check_only:
            print(json.dumps({"status": "prepared", "source_git_commit": spec["source_git_commit"],
                              "changed_files": len(spec["files"])}))
            return
        module_spec = importlib.util.spec_from_file_location("source_updater", patch / "apply-update.py")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        print(json.dumps(module.apply(args.root, patch)))


if __name__ == "__main__":
    main()
