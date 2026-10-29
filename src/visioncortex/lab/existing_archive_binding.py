"""Explicit read-only binding to a native archive without ArchiveContract.

The private descriptor only authorizes future paths. It does not certify old
results, bootstrap the archive, create metadata, or replace its existing guide.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat

from .components import pipeline_module
from .domain import Conflict
from .storage import _mount_info, _REMOTE_FS, is_nas, storage_binding, verify_root

ENV = "VC_LAB_ARCHIVE_COMPATIBILITY_MANIFEST"
SCHEMA = "vc-lab-existing-native-root/1"
LAYOUT = {
    "device_day": "YYYY-MM-DD_<camera_key>",
    "timezone": "Asia/Shanghai",
    "raw_directories": ["MetaData", "MetaVideo"],
    "result_directories": [
        "ProcessedClips",
        "MultimodalUnderstanding",
        "LaboratoryDailyReport",
        "Comment",
    ],
    "historical_paths_preserved": True,
    "originals_mutable_by_processing": False,
}


def _unique(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise Conflict("Duplicate existing archive binding key")
        value[key] = item
    return value


def private_json(path):
    """Bounded, canonical local descriptor owned by this process user, mode 0600."""
    path = Path(path)
    if (
        not path.is_absolute()
        or ".." in path.parts
        or is_nas(path)
        or path.resolve(strict=True) != path
    ):
        raise Conflict("Private descriptor must be a canonical local non-NAS file")
    mounts = _mount_info(path)
    if (
        len(mounts) != 1
        or not mounts[0].get("fstype")
        or mounts[0]["fstype"] in _REMOTE_FS
        or str(mounts[0].get("source", "")).startswith("//")
    ):
        raise Conflict("Private descriptor local filesystem identity is unverified")
    with os.fdopen(
        os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb"
    ) as stream:
        before = os.fstat(stream.fileno())
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0o600
            or before.st_uid != os.getuid()
            or before.st_size > 65536
        ):
            raise Conflict(
                "Private descriptor requires current ownership, mode 0600 and a 64KiB bound"
            )
        raw = stream.read(65537)
        after = os.fstat(stream.fileno())
    if len(raw) > 65536 or (before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise Conflict("Private descriptor changed during its bounded read")

    def invalid(value):
        raise Conflict("Nonfinite private JSON")

    value = json.loads(raw, object_pairs_hook=_unique, parse_constant=invalid)
    if type(value) is not dict:
        raise Conflict("Private descriptor must be a JSON object")
    return value


def _read(root, relative, limit, guard, *, optional=False):
    """No mkdir: open NAS components relative to the current pinned superblock."""
    guard.validate()
    mount, path = guard.mount_for(root / relative)
    base = mount.path if mount else root
    descriptor = (
        os.dup(mount.descriptor)
        if mount
        else os.open(base, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    )
    try:
        parts = path.relative_to(base).parts
        for part in parts[:-1]:
            try:
                child = os.open(
                    part,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=descriptor,
                )
            except FileNotFoundError:
                if not optional:
                    raise
                guard.validate()
                return None
            os.close(descriptor)
            descriptor = child
            if mount and os.fstat(descriptor).st_dev != mount.snapshot["device"]:
                raise OSError("Archive identity crossed an unverified filesystem")
            guard.validate(volume=False)
        try:
            fd = os.open(
                parts[-1],
                os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                dir_fd=descriptor,
            )
        except FileNotFoundError:
            if not optional:
                raise
            guard.validate()
            return None
        with os.fdopen(fd, "rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
                raise Conflict("Archive identity must be a bounded regular file")
            raw = stream.read(limit + 1)
            after = os.fstat(stream.fileno())
        if len(raw) > limit or (before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise Conflict("Archive identity changed while reading")
        guard.validate()
        return raw
    finally:
        os.close(descriptor)


def verify_existing_native_root(root, installed):
    name = os.environ.get(ENV)
    if not name:
        raise Conflict(
            "Archive contract absent; explicit existing-native-root binding required"
        )
    try:
        document = private_json(name)
        if (
            set(document) != {"schema_version", "bindings"}
            or document.get("schema_version") != SCHEMA
        ):
            raise Conflict(
                "Explicit existing-native-root descriptor required for a contract-free archive"
            )
        bindings = document["bindings"]
        if type(bindings) is not list or not 1 <= len(bindings) <= 8:
            raise Conflict("Bounded explicit existing archive bindings required")
        selected, seen = [], set()
        installed_sha = hashlib.sha256(installed).hexdigest()
        for binding in bindings:
            if type(binding) is not dict or set(binding) != {
                "mode",
                "archive_root",
                "guide_relative_path",
                "guide_sha256",
                "installed_contract_sha256",
                "layout",
            }:
                raise Conflict("Unknown or missing existing archive binding field")
            archive = binding["archive_root"]
            if (
                type(archive) is not str
                or not Path(archive).is_absolute()
                or str(Path(archive)) != archive
                or ".." in Path(archive).parts
            ):
                raise Conflict("Exact canonical archive root required")
            if archive in seen:
                raise Conflict("Duplicate existing archive root binding")
            seen.add(archive)
            if (
                binding["mode"] != "existing_native_root"
                or binding["guide_relative_path"] != "DataRetentionGuide.docx"
                or json.dumps(binding["layout"], sort_keys=True)
                != json.dumps(LAYOUT, sort_keys=True)
            ):
                raise Conflict("Existing native layout constraints cannot be changed")
            if any(
                type(binding[k]) is not str
                or not re.fullmatch("[0-9a-f]{64}", binding[k])
                for k in ("guide_sha256", "installed_contract_sha256")
            ):
                raise Conflict(
                    "Exact guide and installed contract SHA256 pins required"
                )
            if (
                archive == str(root)
                and binding["installed_contract_sha256"] == installed_sha
            ):
                selected.append(binding)
        if len(selected) != 1:
            raise Conflict(
                "Archive root and installed future layout do not match the private binding"
            )
        binding = selected[0]
        root = verify_root(root)
        guard_module = pipeline_module("device_day_write_guard")
        with guard_module.storage_guard(storage_binding(root)) as guard:
            if (
                _read(
                    root,
                    ".System/ArchiveContract.json",
                    128 * 1024,
                    guard,
                    optional=True,
                )
                is not None
            ):
                raise Conflict(
                    "Existing contract cannot fall back to guide-only binding"
                )
            raw = _read(root, binding["guide_relative_path"], 2 * 1024**2, guard)
            if hashlib.sha256(raw).hexdigest() != binding["guide_sha256"]:
                raise Conflict("Existing native guide SHA256 mismatch")
            if _read(root, binding["guide_relative_path"], 2 * 1024**2, guard) != raw:
                raise Conflict("Existing native guide changed during verification")
        return root / binding["guide_relative_path"]
    except Conflict:
        raise
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise Conflict(
            "Existing native archive binding is unavailable or invalid"
        ) from exc
