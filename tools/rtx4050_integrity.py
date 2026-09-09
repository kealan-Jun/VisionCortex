"""Reuse authenticated hashes only while the filesystem's file identity is unchanged."""
from __future__ import annotations

import ctypes
from functools import lru_cache
import hashlib
import hmac
import json
import os
import stat
import sys
import time


SCHEMA = "visioncortex-package-hash-cache/1"
with open(__file__, "rb") as _source:
    VERIFIER_SHA256 = hashlib.sha256(_source.read()).hexdigest()


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.partial")
    created = False
    try:
        with temporary.open("xb") as handle:
            created = True
            if sys.platform != "win32":
                os.chmod(temporary, 0o600)
            handle.write(data)
        os.replace(temporary, path)
    finally:
        if created:
            temporary.unlink(missing_ok=True)


def _protect(data, *, decrypt=False):
    """Protect the local cache authentication key with the current Windows account."""
    if sys.platform != "win32":
        return data

    class Blob(ctypes.Structure):
        _fields_ = [("size", ctypes.c_uint32), ("data", ctypes.c_void_p)]

    crypt = ctypes.WinDLL("crypt32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    function = crypt.CryptUnprotectData if decrypt else crypt.CryptProtectData
    function.argtypes = [ctypes.POINTER(Blob), ctypes.c_void_p, ctypes.c_void_p,
                         ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(Blob)]
    function.restype = ctypes.c_int
    kernel.LocalFree.argtypes, kernel.LocalFree.restype = [ctypes.c_void_p], ctypes.c_void_p
    buffer = ctypes.create_string_buffer(data)
    source, output = Blob(len(data), ctypes.cast(buffer, ctypes.c_void_p)), Blob()
    if not function(ctypes.byref(source), None, None, None, None, 1, ctypes.byref(output)):
        raise OSError("Package cache key protection unavailable")
    try:
        return ctypes.string_at(output.data, output.size)
    finally:
        kernel.LocalFree(output.data)


def _key(directory):
    path = directory / "authentication.key"
    directory.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            if sys.platform != "win32":
                os.chmod(path, 0o600)
            handle.write(_protect(os.urandom(32)))
    except FileExistsError:
        pass
    if path.is_symlink() or path.stat().st_size > 16384:
        raise ValueError("Invalid cache key")
    if sys.platform != "win32" and path.stat().st_mode & 0o077:
        raise ValueError("Cache key permissions are not private")
    key = _protect(path.read_bytes(), decrypt=True)
    if len(key) != 32:
        raise ValueError("Invalid cache key")
    return key


class _FileBasicInformation(ctypes.Structure):
    _fields_ = [(name, ctypes.c_int64) for name in ("creation", "access", "write", "change")] + [("attributes", ctypes.c_uint32)]


class _FileIdentityInformation(ctypes.Structure):
    _fields_ = [("volume", ctypes.c_uint64), ("file_id", ctypes.c_ubyte * 16)]


@lru_cache(maxsize=1)
def _file_api():
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetDriveTypeW.argtypes, kernel.GetDriveTypeW.restype = [ctypes.c_wchar_p], ctypes.c_uint32
    kernel.GetVolumeInformationW.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint32]
    kernel.GetVolumeInformationW.restype = ctypes.c_int
    kernel.CreateFileW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32,
        ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
    kernel.CreateFileW.restype = ctypes.c_void_p
    kernel.GetFileInformationByHandleEx.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
    kernel.GetFileInformationByHandleEx.restype = ctypes.c_int
    kernel.CloseHandle.argtypes, kernel.CloseHandle.restype = [ctypes.c_void_p], ctypes.c_int
    return kernel


@lru_cache(maxsize=8)
def _windows_filesystem(anchor):
    kernel = _file_api()
    filesystem = ctypes.create_unicode_buffer(32)
    # FAT/exFAT and network shares do not provide the change-time contract used here.
    return kernel.GetDriveTypeW(anchor) in (2, 3) and bool(kernel.GetVolumeInformationW(
        anchor, None, 0, None, None, None, filesystem, len(filesystem)
    )) and filesystem.value.upper() in {"NTFS", "REFS"}


def fingerprint(path):
    """Use Windows ChangeTime, not st_ctime (creation time on Python 3.12 Windows)."""
    info = path.stat()
    if not stat.S_ISREG(info.st_mode):
        raise RuntimeError(f"离线包包含非普通文件：{path.name}")
    if sys.platform != "win32":
        return ["posix-change-time", info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]
    try:
        if not _windows_filesystem(path.anchor):
            return None

        kernel = _file_api()
        handle = kernel.CreateFileW(str(path), 0x80, 7, None, 3, 0, None)
        if handle in (None, ctypes.c_void_p(-1).value):
            return None
        try:
            basic, identity = _FileBasicInformation(), _FileIdentityInformation()
            if (not kernel.GetFileInformationByHandleEx(handle, 0, ctypes.byref(basic), ctypes.sizeof(basic))
                    or not kernel.GetFileInformationByHandleEx(handle, 18, ctypes.byref(identity), ctypes.sizeof(identity))
                    or basic.attributes & 0x410):
                return None
            return ["windows-change-time", identity.volume, bytes(identity.file_id).hex(),
                    info.st_size, basic.creation, basic.write, basic.change]
        finally:
            kernel.CloseHandle(handle)
    except (OSError, AttributeError):
        return None


def _settled(identity):
    # Files changed within a timestamp granularity window must be read again.
    # This also covers a same-size rewrite before a coarse clock advances.
    if identity is None:
        return False
    latest = max(identity[-2:])
    if identity[0] == "windows-change-time":
        latest = (latest - 116444736000000000) * 100
    return 0 < latest <= time.time_ns() - 1_000_000_000


def verify(root, *, hash_file, safe_path, progress=None, force_full=False):
    root = root.resolve()
    started = time.monotonic()
    manifest_bytes = (root / "SHA256SUMS.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    files = manifest["files"]
    if len(files) != manifest["file_count"]:
        raise RuntimeError("Package manifest count mismatch")
    seen = set()
    for item in files:
        name = item["path"]
        if name.casefold() in seen:
            raise RuntimeError("Duplicate package manifest path")
        seen.add(name.casefold())
        if not isinstance(item["size_bytes"], int) or item["size_bytes"] < 0:
            raise RuntimeError("Invalid package manifest size")
    directory = root / "Runtime/Cache/PackageIntegrity"
    cache_path = directory / "verified-files.json"
    records, key = {}, None
    cache_status = "forced_full" if force_full else "cold"
    try:
        key = _key(directory)
        if not force_full and cache_path.stat().st_size <= 64 * 1024**2:
            cache = json.loads(cache_path.read_bytes())
            if not isinstance(cache, dict):
                raise ValueError("Invalid cache record")
            signature = cache.pop("signature")
            if (cache.get("schema_version") == SCHEMA and cache.get("package_root") == str(root)
                    and cache.get("verifier_sha256") == VERIFIER_SHA256
                    and hmac.compare_digest(signature, hmac.new(key, _json(cache), "sha256").hexdigest())
                    and isinstance(cache.get("files"), dict)):
                records = cache["files"]
                cache_status = "available"
    except (OSError, ValueError, KeyError, TypeError):
        # Missing/corrupt/unreadable caches never bypass the original full hash check.
        cache_status = "unavailable"
    counters = {"checked_bytes": 0, "total_bytes": sum(x["size_bytes"] for x in files),
                "checked_files": 0, "total_files": len(files), "hashed_bytes": 0, "reused_bytes": 0,
                "hashed_files": 0, "reused_files": 0}
    updated = {}

    def report(size=0):
        counters["checked_bytes"] += size
        counters["hashed_bytes"] += size
        if progress:
            progress(dict(counters))

    report()
    try:
        for entry in files:
            name, expected = entry["path"], entry["sha256"]
            path = safe_path(root, name)
            if path.stat().st_size != entry["size_bytes"]:
                raise RuntimeError(f"文件校验失败，请重新完整解压：{name}")
            before = fingerprint(path)
            previous = records.get(name)
            if (_settled(before) and isinstance(previous, dict)
                    and previous.get("identity") == before and previous.get("sha256") == expected):
                counters["checked_bytes"] += entry["size_bytes"]
                counters["reused_bytes"] += entry["size_bytes"]
                counters["reused_files"] += 1
            else:
                digest = hash_file(path, on_chunk=report)
                if digest != expected:
                    raise RuntimeError(f"文件校验失败，请重新完整解压：{name}")
                after = fingerprint(path)
                if before != after or path.stat().st_size != entry["size_bytes"]:
                    raise RuntimeError(f"校验期间文件发生变化，请关闭更新程序后重试：{name}")
                counters["hashed_files"] += 1
            if before is not None:
                updated[name] = {"identity": before, "sha256": expected}
            counters["checked_files"] += 1
            report()
        if (root / "SHA256SUMS.json").read_bytes() != manifest_bytes:
            raise RuntimeError("校验期间应用版本发生变化，请完成更新后重试。")
    except Exception:
        # Never retain a partially successful verification as a trusted new cache.
        try:
            cache_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    if key is None:
        # A copied package may contain a key protected for another Windows user.
        # Re-key only after every file has passed a fresh full verification.
        try:
            candidate_key = os.urandom(32)
            _atomic(directory / "authentication.key", _protect(candidate_key))
            key = candidate_key
        except (OSError, ValueError):
            pass
    if key is not None:
        cache = {"schema_version": SCHEMA, "package_root": str(root), "files": updated,
                 "verifier_sha256": VERIFIER_SHA256}
        cache["signature"] = hmac.new(key, _json(cache), "sha256").hexdigest()
        try:
            _atomic(cache_path, _json(cache))
            cache_status = "saved"
        except OSError:
            cache_status = "write_unavailable"
    receipt = {"schema_version": "visioncortex-package-verification/1", "status": "passed",
               "mode": "incremental" if counters["reused_files"] else "full", "cache_status": cache_status,
               "manifest_sha256": manifest_sha, "package_root": str(root), **counters,
               "verifier_sha256": VERIFIER_SHA256,
               "elapsed_seconds": round(time.monotonic() - started, 3),
               "checked_at": time.time(), "fresh_full_hash_verification": not bool(counters["reused_files"])}
    try:
        _atomic(root / "Runtime/Logs/package-verification.json", _json(receipt))
    except OSError:
        pass
    return manifest
