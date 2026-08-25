from __future__ import annotations

import ntpath
import os
from pathlib import Path, PureWindowsPath
from typing import Union


PathLike = Union[str, os.PathLike[str]]


def _windows_path_text(value: PathLike) -> str:
    """Return one lexical spelling for normal and extended Windows paths."""

    text = str(value).replace("/", "\\")
    folded = text.casefold()
    if folded.startswith("\\\\?\\unc\\"):
        text = "\\\\" + text[8:]
    elif folded.startswith("\\\\?\\"):
        text = text[4:]
    return ntpath.normpath(text)


def _looks_like_windows_path(value: PathLike) -> bool:
    text = str(value).replace("/", "\\")
    return bool(
        text.startswith("\\\\")
        or text.startswith("\\\\?\\")
        or (len(text) >= 2 and text[1] == ":")
    )


def archive_relative_posix(path: PathLike, root: PathLike) -> str:
    """Return an archive-relative path with UNC namespace equivalence.

    Windows may expose the same long NAS path as either
    ``\\\\server\\share\\...`` or ``\\\\?\\UNC\\server\\share\\...``.
    ``Path.relative_to`` treats those spellings as unrelated.  Normalize only
    the Windows namespace prefix, then use component-aware path containment.
    """

    if _looks_like_windows_path(path) or _looks_like_windows_path(root):
        candidate = PureWindowsPath(_windows_path_text(path))
        archive_root = PureWindowsPath(_windows_path_text(root))
        try:
            relative = candidate.relative_to(archive_root)
        except ValueError as exc:
            raise ValueError(
                f"Artifact path is outside archive root: {path!s} (root={root!s})"
            ) from exc
        return relative.as_posix()

    candidate = Path(path).resolve(strict=False)
    archive_root = Path(root).resolve(strict=False)
    try:
        return candidate.relative_to(archive_root).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"Artifact path is outside archive root: {path!s} (root={root!s})"
        ) from exc


def archive_contains(path: PathLike, root: PathLike) -> bool:
    try:
        archive_relative_posix(path, root)
    except ValueError:
        return False
    return True
