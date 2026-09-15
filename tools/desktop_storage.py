"""Bounded desktop storage discovery and a small write/rename/read probe."""

from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path, PureWindowsPath
import shutil
import subprocess
import sys
import tempfile


def discover() -> dict:
    """Inspect existing Windows network mappings, never enumerate share contents."""
    if sys.platform != "win32":
        return {"candidates": [], "message": "当前系统未检测 Windows 网络映射。"}
    script = (
        "[Console]::OutputEncoding=[System.Text.Encoding]::UTF8; "
        "$ErrorActionPreference='Stop'; "
        "@(Get-CimInstance Win32_LogicalDisk -Filter 'DriveType=4' | "
        "Select-Object DeviceID,ProviderName) | ConvertTo-Json -Compress"
    )
    executable = Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32/WindowsPowerShell/v1.0/powershell.exe"
    try:
        result = subprocess.run(
            [str(executable), "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True, text=True, encoding="utf-8-sig", check=True,
            timeout=6, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        rows = json.loads(result.stdout or "[]")
        if isinstance(rows, dict):
            rows = [rows]
        candidates = []
        for row in rows:
            remote = str(row.get("ProviderName") or "")
            if not remote.startswith("\\\\"):
                continue
            candidate = {"data_root": str(PureWindowsPath(remote) / "VisionCortexData"),
                         "label": f"网络盘 {row.get('DeviceID', '')} · {remote}", "kind": "network"}
            if candidate["data_root"] not in {item["data_root"] for item in candidates}:
                candidates.append(candidate)
        return {"candidates": candidates, "message": "已检测当前用户连接的网络盘；保存时会检查可用性。"}
    except (OSError, ValueError, subprocess.SubprocessError):
        return {"candidates": [], "message": "未检测到可用网络映射，可选择本地目录或手动填写 NAS 路径。"}


def normalize_root(value: str, package_root: Path) -> Path:
    if not isinstance(value, str) or not value.strip() or len(value) > 240 or any(ord(c) < 32 for c in value):
        raise ValueError("请输入有效的保存目录，建议使用较短路径。")
    if sys.platform == "win32":
        windows = PureWindowsPath(value.strip())
        if not windows.is_absolute() or str(windows).startswith(("\\\\?\\", "\\\\.\\")):
            raise ValueError("请选择完整盘符路径或 NAS 共享路径。")
        for part in windows.parts[1:]:
            if part.endswith((".", " ")) or any(c in part for c in '<>:"|?*') or PureWindowsPath(part).is_reserved():
                raise ValueError("目录包含 Windows 不支持的名称。")
    target = Path(value.strip())
    if not target.is_absolute():
        raise ValueError("请选择绝对路径。")
    target = target.resolve()
    if target == Path(target.anchor):
        raise ValueError("请选择磁盘或共享下面的专用文件夹。")
    package_root = package_root.resolve()
    runtime = package_root / "Runtime"
    if target == package_root or (package_root in target.parents and target != runtime and runtime not in target.parents):
        raise ValueError("应用目录内只允许选择 Runtime，或选择应用目录外的专用文件夹。")
    return target


def input_directory(value: str, package_root: Path, data_root: Path) -> Path:
    source = normalize_root(value, package_root)
    if not source.is_dir():
        raise OSError("原视频目录不可读取，请检查连接或清空该选项后使用文件上传。")
    if source == data_root or source in data_root.parents or data_root in source.parents:
        raise ValueError("原视频目录和产出目录必须分开放置，避免重复索引系统生成的视频。")
    with os.scandir(source) as entries:
        next(entries, None)
    return source


def probe(value: str, package_root: Path, source_root: str = "") -> dict:
    target = normalize_root(value, package_root)
    source = input_directory(source_root, package_root, target) if source_root else None
    if target.exists() and not target.is_dir():
        raise ValueError("所选路径是文件，请选择文件夹。")
    if not Path(target.anchor).is_dir():
        raise OSError("磁盘或 NAS 共享不可用，请检查连接或选择其他目录。")
    kind = "local"
    if sys.platform == "win32":
        get_type = ctypes.WinDLL("kernel32", use_last_error=True).GetDriveTypeW
        get_type.argtypes = [ctypes.c_wchar_p]
        get_type.restype = ctypes.c_uint
        drive_type = get_type(target.anchor)
        if drive_type not in {2, 3, 4}:
            raise OSError("请选择可用的本地磁盘或网络共享。")
        kind = "network" if drive_type == 4 else "local"
    target.mkdir(parents=True, exist_ok=True)
    # Probe the exact archive destination too; an existing Archives folder may
    # have different permissions from its parent. Never touch user media.
    archive = target / "Archives"
    if archive.exists() and not archive.is_dir():
        raise ValueError("该位置已有名为 Archives 的文件，请选择其他文件夹。")
    archive.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".visioncortex-write-check-", dir=archive) as temporary:
        first = Path(temporary) / "write.tmp"
        second = Path(temporary) / "renamed.tmp"
        with first.open("xb") as handle:
            handle.write(b"VisionCortex storage check\n")
            handle.flush()
            os.fsync(handle.fileno())
        first.replace(second)
        if second.read_bytes() != b"VisionCortex storage check\n":
            raise OSError("保存目录读写校验失败。")
    free = shutil.disk_usage(archive).free
    if free < 1024**3:
        raise OSError("保存盘可用空间不足 1 GiB，请选择空间更充足的目录。")
    return {"data_root": str(target), "archive_root": str(archive), "kind": kind,
            "source_root": str(source) if source else "",
            "free_bytes": free, "write_rename_read": "PROVEN"}


def main() -> int:
    try:
        value = json.loads(sys.stdin.readline(8192))
        if value.get("action") == "discover":
            result = discover()
        elif value.get("action") == "probe":
            result = probe(value.get("data_root"), Path(__file__).resolve().parents[1], value.get("source_root") or "")
        else:
            raise ValueError("未知的保存位置操作。")
        print(json.dumps({"ok": True, **result}, ensure_ascii=False), flush=True)
        return 0
    except PermissionError:
        print(json.dumps({"ok": False, "message": "当前用户没有保存目录的读写权限，请选择其他目录或检查共享权限。"}, ensure_ascii=False), flush=True)
        return 1
    except Exception as error:
        print(json.dumps({"ok": False, "message": str(error)[:1000]}, ensure_ascii=False), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
