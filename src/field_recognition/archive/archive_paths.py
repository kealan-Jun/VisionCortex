"""Resumable path moves, hash deduplication and immutable-receipt compatibility.

Only the archive writer mutates this ledger. A move is journaled before rename;
readers accept the old location until the atomic rename has completed.
"""

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from field_recognition.archive.archive_progress import advance

LEDGER = ".System/MigrationMap.json"


def validate_name(relative):
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("Invalid archive path")
    parts = PurePosixPath(relative)
    if parts.is_absolute() or ".." in parts.parts or relative.startswith("/"):
        raise ValueError("Invalid archive path")


def checked(root, relative, *, resolved_root=None):
    validate_name(relative)
    path = (root / relative).resolve()
    path.relative_to(root.resolve() if resolved_root is None else resolved_root)
    advance()
    return path


def load(root, *, resolved_root=None):
    path = checked(root, LEDGER, resolved_root=resolved_root)
    if not path.exists():
        return {
            "schema": "field-recognition-paths/1",
            "aliases": {},
            "assets": {},
            "events": {},
            "views": {},
        }
    value = json.loads(path.read_text())
    if value.get("schema") != "field-recognition-paths/1":
        raise ValueError("Unknown archive path ledger")
    return value


def resolve_entry(root, relative, aliases=None, *, resolved_root=None):
    """Resolve once per operation; never cache existence or alias results."""
    aliases = load(root, resolved_root=resolved_root)["aliases"] if aliases is None else aliases
    seen = set()
    while True:
        path = checked(root, relative, resolved_root=resolved_root)
        present = path.is_file()
        # During an interrupted move the old file is still authoritative.
        if present or relative not in aliases:
            return relative, path, present
        if relative in seen:
            raise ValueError("Archive alias cycle")
        seen.add(relative)
        relative = aliases[relative]


def resolve_name(root, relative, aliases=None):
    return resolve_entry(root, relative, aliases)[0]


def resolve_file(root, relative):
    return resolve_entry(root, relative)[1]


class ArchivePaths:
    def __init__(self, root, *, layout="events", source_roots=()):
        from field_recognition.archive.archive_store import canonical

        # The root is stable for this single-writer operation. Every child path
        # still undergoes a real resolve before filesystem access.
        self.root = Path(root).resolve()
        self.layout = layout
        self.source_roots = tuple(source_roots)
        self.data = load(self.root, resolved_root=self.root)
        self.saved = canonical(self.data)

    def _checked(self, relative):
        return checked(self.root, relative, resolved_root=self.root)

    def _resolve(self, relative):
        return resolve_entry(self.root, relative, self.data["aliases"], resolved_root=self.root)

    def _set_alias(self, source, target):
        validate_name(source)
        validate_name(target)
        if source != target:
            self.data["aliases"][source] = target

    def save(self):
        from field_recognition.archive.archive_store import canonical, replace_view

        snapshot = canonical(self.data)
        path = self._checked(LEDGER)
        if snapshot == self.saved and path.is_file():
            return
        replace_view(path, snapshot)
        self.saved = snapshot

    def alias(self, source, target):
        self._checked(source)
        self._checked(target)
        self._set_alias(source, target)

    def name(self, relative):
        return self._resolve(relative)[0]

    def move(self, source, target):
        source_path, target_path = self._checked(source), self._checked(target)
        self._move(source, target, source_path, target_path)

    def _move(self, source, target, source_path, target_path):
        if source == target:
            return
        if source_path.is_file() and target_path.exists():
            if source_path.read_bytes() != target_path.read_bytes():
                raise ValueError("Archive migration destination conflict")
        self._set_alias(source, target)
        self.save()  # Must precede rename or removal of a duplicate.
        if source_path.is_file():
            target_path.parent.mkdir(parents=True, exist_ok=True)
            if target_path.exists():
                # The target may have appeared since the pre-journal check.
                # An external conflicting file must never make us delete source.
                if source_path.read_bytes() != target_path.read_bytes():
                    raise ValueError("Archive migration destination conflict")
                source_path.unlink()
            else:
                source_path.rename(target_path)

    def asset(self, logical, raw, *, save=True):
        """Stage bytes; row writers may save once before publishing their receipt.

        Deferral changes only the in-memory ledger. Every object is still written
        immutably, and path moves continue to commit their own rename journal.
        """
        from field_recognition.archive.archive_store import immutable_write

        digest = hashlib.sha256(raw).hexdigest()
        stored = self.data["assets"].get(digest)
        relative, path, present = self._resolve(stored["path"] if stored else logical)
        if not present:
            relative = ".System/PendingAssets/" + digest + Path(logical).suffix
            path = self._checked(relative)
        immutable_write(path, raw)
        result = {"path": relative, "sha256": digest, "size_bytes": len(raw)}
        self.data["assets"][digest] = result
        if logical != relative:
            self._checked(logical)
        self._set_alias(logical, relative)
        if save:
            self.save()
        return result

    def place(self, artifact, target, *, pin_source=False):
        """Move a hash-addressed image into its event; aliases preserve all receipts."""
        digest = artifact["sha256"]
        stored = self.data["assets"].get(digest, artifact)
        source, path, _ = self._resolve(stored["path"])
        target_path = path if source == target else self._checked(target)
        # Always validate actual bytes, including an unchanged physical place.
        # Size/mtime and the ledger cannot establish an immutable object digest.
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest or len(raw) != artifact["size_bytes"]:
            raise ValueError("Cannot migrate corrupt archive evidence")
        if stored.get("source_owned"):
            # Capture-owned originals are pinned. Recognition never renames or
            # unlinks an uploader's photograph to arrange its own outputs.
            old = artifact["path"]
            if (
                old != source
                and not artifact.get("source_owned")
                and (old.startswith(("Objects/", ".System/PendingAssets/")) or "/Photos/" in old)
            ):
                old_name, old_path, present = self._resolve(old)
                if present and old_name != source:
                    # Reclaim only a verified processing-owned duplicate. The
                    # uploader's original and its other capture files stay put.
                    if old_path.read_bytes() != raw:
                        raise ValueError("Historical original differs from uploaded source")
                    self._move(old_name, source, old_path, path)
            return stored | {"path": source}
        if pin_source and any(
            self.root.joinpath(*PurePosixPath(target).parts[:i]).is_symlink()
            for i in range(1, len(PurePosixPath(target).parts) + 1)
        ):
            raise ValueError("Source photograph cannot contain symlinks")
        if artifact["path"] not in (source, target):
            self._checked(artifact["path"])
        if (
            not pin_source
            and source == target
            and self.data["assets"].get(digest) == (artifact | {"path": target})
        ):
            self._set_alias(artifact["path"], target)
            return self.data["assets"][digest]
        self._move(source, target, path, target_path)
        self._set_alias(artifact["path"], target)
        self.data["assets"][digest] = (
            artifact | {"path": target} | ({"source_owned": True} if pin_source else {})
        )
        self.save()
        return self.data["assets"][digest]

    def adopt_source(self, relative, digest):
        """Reference an uploaded original only after verifying its actual bytes."""
        path = self._checked(relative)
        if any(
            (self.root.joinpath(*PurePosixPath(relative).parts[:i])).is_symlink()
            for i in range(1, len(PurePosixPath(relative).parts) + 1)
        ):
            raise ValueError("Source photograph cannot contain symlinks")
        before = path.stat()
        if not 0 < before.st_size <= 16 * 1024 * 1024:
            raise ValueError("Invalid source photograph size")
        raw = path.read_bytes()
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("Source photograph changed while archiving")
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError("Source photograph SHA-256 mismatch")
        result = {"path": relative, "sha256": digest, "size_bytes": len(raw), "source_owned": True}
        # Existing immutable receipts may still name their historical artifact.
        # Never delete that artifact here; migration verifies it separately.
        previous = self.data["assets"].get(digest)
        if previous and previous["path"] != relative:
            self._set_alias(previous["path"], relative)
        self.data["assets"][digest] = result
        return result

    def directory(self, key, proposed):
        # Allocation is in-memory; every filesystem operation still uses checked().
        validate_name(proposed)
        # Value/owner edits never rename a measurement already delivered.
        return self.data["events"].setdefault(key, proposed)

    def result_file(self, event, instrument, directory):
        """Keep one stable six-field file per instrument, including later burst members."""
        slots = self.data.setdefault("result_files", {}).setdefault(event, {})
        if instrument not in slots:
            number = len(slots) + 1
            name = "Result.json" if number == 1 else f"Result{number:02d}.json"
            slots[instrument] = directory + "/" + name
        self._checked(slots[instrument])
        return slots[instrument]

    def retire_views(self, views, *, partial=False):
        """Remove only byte-verified generated views, never handwritten files."""
        from field_recognition.archive.archive_store import replace_view

        previous = self.data.get("views", {})
        for relative, digest in previous.items():
            if partial or relative in views:
                continue
            path = self._checked(relative)
            if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == digest:
                path.unlink()
        # Result files precede Evidence manifests. This is a publication order,
        # not an atomic multi-file transaction; readers must verify manifest hashes.
        ordered = sorted(
            views.items(), key=lambda item: PurePosixPath(item[0]).name == "Evidence.json"
        )
        for relative, raw in ordered:
            path = self._checked(relative)
            if previous.get(relative) == hashlib.sha256(raw).hexdigest() and path.is_file():
                continue
            if not path.is_file() or path.read_bytes() != raw:
                replace_view(path, raw)
        self.data["views"] = (previous if partial else {}) | {
            p: hashlib.sha256(raw).hexdigest() for p, raw in views.items()
        }
        self.save()


def empty_directories(root):
    # Yield after each actual directory visit; collecting the whole NAS tree
    # before reporting progress can exceed the worker watchdog interval.
    for directory, _, _ in os.walk(root, topdown=False, followlinks=False):
        path = Path(directory)
        if path != root and not path.is_symlink():
            try:
                path.rmdir()
            except OSError:
                pass
        advance("removing_empty_directories")
