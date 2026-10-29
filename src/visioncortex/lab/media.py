from __future__ import annotations

import hashlib
import os
import shutil
import fcntl
from pathlib import Path, PurePosixPath

from .domain import Conflict, Missing, ident


def safe_path(root: Path, relative: str, *, must_exist=True) -> Path:
    parts = PurePosixPath(relative).parts
    if not parts or PurePosixPath(relative).is_absolute() or "\\" in relative or any(p in {"", ".", ".."} for p in relative.split("/")):
        raise ValueError("Only a normalized storage-relative path is accepted")
    if root.is_symlink():
        raise ValueError("Storage roots cannot be symbolic links")
    path = root
    for part in parts:
        path = path / part
        if path.is_symlink():
            raise ValueError("Symbolic links are not allowed in media locations")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Media escaped its configured storage root")
    if must_exist and not path.is_file():
        raise FileNotFoundError("Media location unavailable")
    return path


def hash_file(path: Path, max_bytes: int) -> tuple[str, int]:
    if path.is_symlink():
        raise ValueError("Cannot hash a symbolic link")
    before = path.stat()
    if before.st_size > max_bytes:
        raise ValueError("Media exceeds the configured per-capture bound")
    digest, size = hashlib.sha256(), 0
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(block)
            if size > max_bytes:
                raise ValueError("Media grew beyond the configured bound")
            digest.update(block)
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns) or size != after.st_size:
        raise Conflict("Source media changed during verification")
    return digest.hexdigest(), size


class Media:
    def __init__(self, core, settings):
        self.core, self.settings = core, settings

    @property
    def roots(self):
        return {"cache": self.settings.cache, "incoming": self.settings.incoming}

    def cache_file(self, relative, declared_hash):
        with (self.settings.cache / ".admission.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            return self._cache_file(relative, declared_hash)

    def original_file(self, relative, declared_hash=None):
        """Large sealed native captures stay at their producer-owned path."""
        source = safe_path(self.settings.incoming, relative)
        digest, size = hash_file(source, self.settings.max_native_bytes)
        if declared_hash is not None and declared_hash != digest:
            raise Conflict('Original bytes differ from their sealed receipt')
        return dict(sha256=digest, byte_size=size, relative_path=relative, storage='incoming')

    def _cache_file(self, relative, declared_hash):
        source = safe_path(self.settings.incoming, relative)
        digest, size = hash_file(source, self.settings.max_capture_bytes)
        if declared_hash is not None and declared_hash != digest:
            raise Conflict("Declared media hash does not match independently verified bytes")
        suffix = source.suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png", ".mp4", ".opus", ".wav", ".m4a", ".flac"}:
            raise ValueError("Unsupported media extension")
        target_relative = f"objects/{digest[:2]}/{digest}{suffix}"
        target = safe_path(self.settings.cache, target_relative, must_exist=False)
        if target.exists():
            actual, _ = hash_file(target, self.settings.max_capture_bytes)
            if actual != digest:
                raise Conflict("Existing cached object has an integrity conflict")
        else:
            used = sum(p.stat().st_size for p in self.settings.cache.rglob("*") if p.is_file())
            if used + size > self.settings.cache_max_bytes:
                raise Conflict("Working media cache capacity reached; archive verified evidence before admitting more")
            if shutil.disk_usage(self.settings.runtime_root).free < 80 * 1024**3:
                raise Conflict("Local data filesystem free-space guard reached")
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + "." + str(ident()) + ".partial")
            try:
                with source.open("rb") as inp, temporary.open("xb") as out:
                    shutil.copyfileobj(inp, out, length=1024 * 1024)
                    out.flush()
                    os.fsync(out.fileno())
                copied, _ = hash_file(temporary, self.settings.max_capture_bytes)
                if copied != digest:
                    raise Conflict("Source changed while being cached")
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
        return dict(sha256=digest, byte_size=size, relative_path=target_relative, storage='cache')

    def locate(self, media_id):
        with self.core.db.transaction(readonly=True) as conn:
            asset = conn.execute("SELECT * FROM media_asset WHERE customer_id=%s AND id=%s", (self.core.customer, ident(media_id))).fetchone()
            if not asset:
                raise Missing("Media outside customer scope")
            locations = conn.execute("SELECT * FROM media_location WHERE customer_id=%s AND media_id=%s ORDER BY storage", (self.core.customer, ident(media_id))).fetchall()
        if asset["verified_sha256"] is None:
            return dict(status="unverified", media_id=str(media_id), path=None)
        for location in locations:
            if location["storage"] not in self.roots:
                continue
            try:
                path = safe_path(self.roots[location["storage"]], location["relative_path"])
                digest, size = hash_file(path, self.settings.max_native_bytes if location['storage'] == 'incoming' else self.settings.max_capture_bytes)
                if digest != asset["verified_sha256"] or size != asset["byte_size"]:
                    return dict(status="integrity_failed", media_id=str(media_id), path=None)
                return dict(status="available", media_id=str(media_id), path=path, version=asset["version"])
            except FileNotFoundError:
                continue
        return dict(status="unavailable", media_id=str(media_id), path=None)

    def add_location(self, key, media_id, storage, relative):
        if storage not in self.roots:
            raise ValueError("Unknown storage root")
        path = safe_path(self.roots[storage], relative)
        digest, size = hash_file(path, self.settings.max_capture_bytes)

        def action(conn):
            asset = conn.execute("SELECT * FROM media_asset WHERE customer_id=%s AND id=%s FOR UPDATE", (self.core.customer, ident(media_id))).fetchone()
            if not asset:
                raise Missing("Media outside customer scope")
            if digest != asset["verified_sha256"] or size != asset["byte_size"]:
                raise Conflict("Location bytes do not match the stable media identity")
            conn.execute("INSERT INTO media_location(customer_id,id,media_id,storage,relative_path,availability) VALUES(%s,%s,%s,%s,%s,'available') ON CONFLICT DO NOTHING", (self.core.customer, ident(), ident(media_id), storage, relative))
            return dict(media_id=media_id, storage=storage, relative_path=relative)
        return self.core.command(key, dict(media=media_id, storage=storage, relative=relative, sha256=digest), action)
