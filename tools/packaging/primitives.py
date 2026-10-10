"""Small shared artifact operations, independent of release/admission policy."""
from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile
import tempfile


def sha256_file(path: Path, *, on_chunk=None, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(chunk_size), b""):
            digest.update(chunk)
            if on_chunk is not None:
                on_chunk(len(chunk))
    return digest.hexdigest()


def relative_path(value: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or any(ord(c) < 32 or ord(c) == 127 for c in value):
        raise ValueError("Invalid package path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value or ":" in value or str(path) != value or value == ".":
        raise ValueError("Package path escapes its root")
    return path


def safe_path(root: Path, value: str, *, reject_links: bool = False) -> Path:
    root = root.resolve()
    path = root / relative_path(value)
    if reject_links:
        for part in (path, *path.parents):
            if part == root:
                break
            if part.is_symlink() or (hasattr(part, "is_junction") and part.is_junction()):
                raise ValueError("Package paths cannot contain links")
    resolved = path.resolve()
    if root not in resolved.parents:
        raise ValueError("Package path escapes its root")
    return path if reject_links else resolved


def export_git_commit(repo: Path, commit: str, destination: Path) -> None:
    """Export only an immutable commit; candidate/readiness gates stay with callers."""
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Git export requires a full immutable commit SHA")
    actual = subprocess.check_output(["git", "rev-parse", f"{commit}^{{commit}}"], cwd=repo, text=True).strip()
    if actual != commit:
        raise ValueError("Git export commit identity differs")
    destination.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() or any(destination.iterdir()):
        raise FileExistsError("Git export destination must be an empty owned directory")
    with tempfile.TemporaryDirectory(prefix=".visioncortex-export-", dir=destination.parent) as temporary:
        archive = Path(temporary) / "source.tar"
        with archive.open("wb") as stream:
            subprocess.run(["git", "archive", "--format=tar", commit], cwd=repo, check=True, stdout=stream)
        with tarfile.open(archive, "r") as bundle:
            for member in bundle.getmembers():
                safe_path(destination, member.name.rstrip("/"))
            bundle.extractall(destination, filter="data")
