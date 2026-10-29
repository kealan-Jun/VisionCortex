"""Bounded, read-only NAS I/O. Child processes never import the application/GPU."""

import base64
import hashlib
import json
import os
from pathlib import Path
from field_recognition.capture.photo_sources import photo_location
from field_recognition.runtime.paths import PROJECT_ROOT
import re
import subprocess
import sys
import threading
from datetime import datetime
from zoneinfo import ZoneInfo
import time

_LOCK = threading.Lock()
_ORPHANS = []
_SLOTS = threading.BoundedSemaphore(4)


class SourceError(Exception):
    def __init__(self, detail, status=503):
        super().__init__(detail)
        self.status = status


def call(request, *, timeout=None):
    if not _SLOTS.acquire(blocking=False):
        raise SourceError("NAS 读取槽位已满，稍后重试")
    try:
        return _call(request, timeout=timeout)
    finally:
        _SLOTS.release()


def _call(request, *, timeout=None):
    timeout = timeout or float(os.environ.get("FIELD_PHOTO_IO_TIMEOUT_SECONDS", "4"))
    with _LOCK:
        _ORPHANS[:] = [p for p in _ORPHANS if p.poll() is None]
        if len(_ORPHANS) >= 2:
            raise SourceError("NAS 读取进程尚未退出，等待存储恢复")
    proc = subprocess.Popen(
        [sys.executable, "-m", "field_recognition.runtime.source_io", "--worker"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        cwd=PROJECT_ROOT,
    )
    try:
        raw, _ = proc.communicate(json.dumps(request), timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        # Do not wait on an uninterruptible filesystem syscall. Bound orphan count.
        with _LOCK:
            _ORPHANS.append(proc)
        for pipe in (proc.stdin, proc.stdout):
            if pipe:
                pipe.close()
        raise SourceError("NAS 读取超时，文件留在持久化队列") from None
    if proc.returncode:
        raise SourceError("NAS 读取进程异常退出")
    reply = json.loads(raw)
    if "error" in reply:
        raise SourceError(reply["error"], reply.get("status", 503))
    return reply


def safe_path(root, supplied):
    root = Path(root).resolve(strict=True)
    path = Path(supplied)
    path = path if path.is_absolute() else root / path
    # Reject symlink components, including same-root links which could change targets.
    relative = path.relative_to(root)
    for i in range(1, len(relative.parts) + 1):
        if (root.joinpath(*relative.parts[:i])).is_symlink():
            raise SourceError("照片路径不能包含符号链接", 422)
    path = path.resolve(strict=True)
    path.relative_to(root)
    return path, relative


def read_stable(path, limit):
    before = path.stat()
    if not path.is_file() or not 0 < before.st_size <= limit:
        raise SourceError("文件为空或超过大小上限", 413)
    with path.open("rb") as handle:
        raw = handle.read(limit + 1)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or len(
        raw
    ) != before.st_size:
        raise SourceError("照片或回执仍在写入", 409)
    return raw, after


def read(request):
    roots = request.get("roots") or [request["root"]]
    supplied = Path(request["path"])
    outside = supplied.is_absolute()
    for root in roots:
        if supplied.is_absolute():
            try:
                supplied.relative_to(Path(root).absolute())
            except ValueError:
                continue
            outside = False
        attempts = [request["path"]]
        if not supplied.is_absolute():
            try:
                location = photo_location(supplied.as_posix())
                attempts.extend([location.metadata_path, location.identity])
            except ValueError:
                pass
        for candidate in dict.fromkeys(attempts):
            try:
                return read_one(request | {"root": root, "path": candidate})
            except FileNotFoundError:
                continue
    if outside:
        raise SourceError("照片路径不属于已配置的输入目录", 422)
    raise SourceError("照片存储暂不可读", 503)


def read_one(request):
    path, relative = safe_path(request["root"], request["path"])
    try:
        location = photo_location(relative.as_posix())
    except ValueError:
        raise SourceError("照片目录不匹配", 422) from None
    if location.camera != request["camera"]:
        raise SourceError("照片相机不匹配", 409)
    if (path.parent / ".uploading").exists():
        raise SourceError("照片目录仍在上传，保留队列等待", 503)
    raw, info = read_stable(path, request["max_bytes"])
    receipt = path.parent / "PhotoReceipt.json"
    metadata = None
    receipt_basis = "nas_photo_receipt"
    if receipt.exists():
        if receipt.is_symlink():
            raise SourceError("连拍回执不能为符号链接", 422)
        content, _ = read_stable(receipt, 128 * 1024)
        try:
            metadata = json.loads(content)
        except (ValueError, UnicodeError):
            raise SourceError("连拍回执格式无效", 422) from None
    elif request.get("batch_mode") == "directory_triplet":
        # Explicit directory contract: one planned triplet has one stable batch
        # identity. Available photographs can be read before the others arrive.
        metadata = directory_triplet(request, path, Path(location.identity))
        receipt_basis = "configured_directory_triplet"
    elif request.get("require_receipt"):
        raise SourceError("等待连拍回执 PhotoReceipt.json", 503)
    return {
        "relative": location.identity,
        "source_relative": relative.as_posix(),
        "path": str(path),
        "mtime": info.st_mtime,
        "signature": f"{info.st_size}:{info.st_mtime_ns}",
        "image_base64": base64.b64encode(raw).decode(),
        "receipt": metadata,
        "receipt_basis": receipt_basis if metadata else None,
    }


def directory_triplet(request, path, relative):
    def members():
        return sorted(
            p for p in path.parent.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )

    paths = members()
    if len(paths) > 3:
        raise SourceError("目录超过三张照片，不能确定一次拍照边界", 409)
    photos, signatures = [], {}
    for member in paths:
        safe, physical = safe_path(request["root"], member)
        try:
            rel = Path(photo_location(physical.as_posix()).identity)
        except ValueError:
            raise SourceError("三连拍目录不符合上传协议", 422) from None
        match = re.fullmatch(r"(\d{8})_(\d{6})(?:_\d+)?\.(?:jpg|jpeg|png)", safe.name, re.I)
        if not match:
            raise SourceError("三连拍文件名不符合已配置的目录协议", 422)
        captured = datetime.strptime(match[1] + match[2], "%Y%m%d%H%M%S").replace(
            tzinfo=ZoneInfo(request["zone"])
        )
        if rel.parts[:3] != relative.parts[:3] or rel.parts[1:3] != (
            captured.strftime("%Y-%m-%d"),
            captured.strftime("%H-%M-%S"),
        ):
            raise SourceError("三连拍目录与采集时间不一致", 409)
        try:
            raw, info = read_stable(safe, request["max_bytes"])
            if time.time() - info.st_mtime < 0.5:
                raise SourceError("照片仍在写入", 503)
        except SourceError as exc:
            if safe == path or exc.status not in {409, 413, 503}:
                raise
            continue  # Another member's upload does not block this stable photo.
        signatures[safe] = (info.st_size, info.st_mtime_ns)
        photos.append(
            {
                "filename": safe.name,
                "capture_id": rel.as_posix(),
                "captured_at": captured.isoformat(),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    if paths != members() or any(
        signature != (p.stat().st_size, p.stat().st_mtime_ns)
        for p, signature in signatures.items()
    ):
        raise SourceError("语音三连拍目录发生变化，等待稳定", 503)
    batch = hashlib.sha256(
        ("directory-triplet/1:" + relative.parent.as_posix()).encode()
    ).hexdigest()
    from field_recognition.measurements.photo_experiment_context import resolve_context

    try:
        context_ref = resolve_context(request["camera"], photos)
    except (ValueError, KeyError, TypeError):
        raise SourceError("实验关联配置无效，请核对相机、时间范围和实验编号", 422) from None
    return {
        "schema_version": "field-photo-receipt/1"
        if len(photos) == 3
        else "field-photo-directory-snapshot/1",
        "camera_id": request["camera"],
        "measurement": {
            "burst_id": "directory-triplet-" + batch,
            "expected_photos": 3,
            **({"experiment_context_ref": context_ref} if context_ref else {}),
        },
        "photos": photos,
    }


def discover(request):
    files, statuses, errors = {}, [], []
    for root in request.get("roots") or [request["root"]]:
        try:
            result = discover_one(request | {"root": root})
        except (OSError, SourceError) as exc:
            if isinstance(exc, SourceError) and exc.status != 503:
                raise
            statuses.append({"root": root, "status": "unavailable"})
            continue
        statuses.append(
            {
                "root": root,
                "status": "missing_camera" if result.get("missing_camera") else "watching",
                "discovered_files": len(result["files"]),
            }
        )
        errors.extend(result.get("errors", []))
        for row in result["files"]:
            # Relative capture identities survive a source-root move. The saved
            # photo transaction still checks bytes before reusing any OCR job.
            files.setdefault(photo_location(row[0]).identity, row)
    if statuses and all(s["status"] == "unavailable" for s in statuses):
        raise SourceError("照片存储暂不可读", 503)
    return {
        "files": list(files.values()),
        "errors": errors,
        "source_roots": statuses,
        "missing_camera": all(s["status"] != "watching" for s in statuses),
    }


def discover_one(request):
    root = Path(request["root"])
    camera = request["camera"]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", camera):
        raise SourceError("相机目录无效", 422)
    if not root.is_dir():
        raise SourceError("照片存储未连接")
    zone = ZoneInfo(request["zone"])
    started = datetime.fromisoformat(request["since"]).astimezone(zone)
    today = datetime.now(zone).strftime("%Y-%m-%d")
    directories = []
    # Only MetaData is an input under the day/camera tree. Regions and exported
    # Photos are never traversed, even when their filenames resemble uploads.
    for parent in sorted(root.iterdir(), reverse=True):
        if (
            len(parent.name) > 11
            and parent.name[10:] == "_" + camera
            and re.fullmatch(r"\d{4}-\d{2}-\d{2}", parent.name[:10])
            and not parent.is_symlink()
            and parent.is_dir()
        ):
            metadata = parent / "MetaData"
            if metadata.is_dir() and not metadata.is_symlink():
                directories.append((metadata, parent.name[:10]))
    folder = root / camera
    if folder.is_symlink():
        raise SourceError("相机目录不能为符号链接", 422)
    if folder.is_dir():
        directories.extend(
            (day, day.name)
            for day in sorted(folder.iterdir(), reverse=True)
            if not day.is_symlink() and day.is_dir()
        )
    rows, errors = [], []
    for day, day_name in directories:
        if (
            not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day_name)
            or not started.strftime("%Y-%m-%d") <= day_name <= today
            or (request.get("recent") and day_name != today)
        ):
            continue
        try:
            moments = sorted(day.iterdir(), reverse=True)
        except OSError:
            errors.append(str(day.relative_to(root)))
            continue
        if request.get("recent"):
            moments = moments[:32]
        for moment in moments:
            if (
                not re.fullmatch(r"\d{2}-\d{2}-\d{2}", moment.name)
                or moment.is_symlink()
                or not moment.is_dir()
                or (moment / ".uploading").exists()
            ):
                continue
            try:
                for path in sorted(moment.iterdir()):
                    if path.is_symlink() or not path.is_file():
                        continue
                    try:
                        location = photo_location(path.relative_to(root).as_posix())
                        captured = datetime.strptime(
                            location.day + location.moment, "%Y-%m-%d%H-%M-%S"
                        ).replace(tzinfo=zone)
                    except ValueError:
                        continue
                    if location.camera != camera or captured < started:
                        continue
                    stat = path.stat()
                    rows.append(
                        [
                            path.relative_to(root).as_posix(),
                            f"{stat.st_size}:{stat.st_mtime_ns}",
                            stat.st_mtime,
                        ]
                    )
            except OSError:
                errors.append(str(moment.relative_to(root)))
    return {"files": rows, "errors": errors, "missing_camera": not directories}


if __name__ == "__main__":
    try:
        request = json.load(sys.stdin)
        reply = {"read": read, "discover": discover}[request["action"]](request)
    except SourceError as exc:
        reply = {"error": str(exc), "status": exc.status}
    except (ValueError, KeyError):
        reply = {"error": "照片路径或回执无效", "status": 422}
    except OSError:
        reply = {"error": "照片存储暂不可读", "status": 503}
    print(json.dumps(reply))
