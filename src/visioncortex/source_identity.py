"""Recursive source inventories for versioned execution identities.

Names are package-relative paths, so moved files and equally named modules in
different packages cannot disappear from the identity. Consumers choose their
own exclusions; a build identity and a stage identity are different contracts.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Collection


SOURCE_RECIPE = "visioncortex-python-source/2"


def _file_hash(path: Path, snapshot: tuple[int, ...]) -> str:
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    if _snapshot(path) != snapshot:
        raise ValueError(f"Source changed while computing identity: {path.name}")
    return checksum


def _snapshot(path: Path) -> tuple[int, ...]:
    info = path.stat()
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def source_manifest(
    directory: Path | None = None, *, exclude: Collection[str] = ()
) -> dict[str, str]:
    root = (directory or Path(__file__).parent).resolve()
    excluded = frozenset(exclude)
    result: dict[str, str] = {}
    paths = _python_paths(root)
    snapshots = {}
    for path in paths:
        name = path.relative_to(root).as_posix()
        if name in excluded:
            continue
        snapshots[path] = _snapshot(path)
        result[name] = _file_hash(path, snapshots[path])
    # Timestamps can collide on rapid equal-size rewrites. Recheck content and
    # membership after the complete pass, including files hashed early in it.
    if _python_paths(root) != paths:
        raise ValueError("Source inventory changed while computing identity")
    for path, snapshot in snapshots.items():
        if _file_hash(path, snapshot) != result[path.relative_to(root).as_posix()]:
            raise ValueError(f"Source changed while computing identity: {path.name}")
    return result


def _python_paths(root: Path) -> list[Path]:
    if not root.is_dir():
        raise ValueError(f"Source root must be an existing directory: {root}")

    def scan_error(error: OSError) -> None:
        raise error

    paths = []
    for directory_path, directories, files in os.walk(
        root, followlinks=False, onerror=scan_error
    ):
        for name in [*directories, *files]:
            path = Path(directory_path) / name
            if path.is_symlink():
                raise ValueError(f"Source identity requires a regular source tree: {path.relative_to(root)}")
        directories[:] = [name for name in directories if name != '__pycache__']
        paths.extend(Path(directory_path) / name for name in files if name.endswith('.py'))
    return sorted(paths)


def source_digest(manifest: dict[str, str]) -> str:
    payload = {"recipe": SOURCE_RECIPE, "files": manifest}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
