"""Discover recorder sidecars; user selections become immutable input manifests.

No producer index is required. Source files are never copied or modified, and
camera roles / experiment membership are never inferred from directory names.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

FIELDS = (
    "experiment_id", "experiment_prefix", "camera_key", "camera_id", "camera_view",
    "recording_start_us", "recording_end_us", "recording_start_time",
    "recording_end_time", "segment_count", "rgb_file", "frames_file", "updated_at",
    "sync_error",
)


def enabled(config: dict[str, Any]) -> bool:
    ingest = config.get("collection_ingest") or {}
    return bool(ingest.get("enabled") and ingest.get("mode") == "directory_metadata")


def _root(config: dict[str, Any]) -> Path:
    if not enabled(config):
        raise ValueError("NAS 目录采集未启用")
    root = Path(config["collection_ingest"]["source_root"])
    if not root.is_dir():
        raise OSError("NAS 未连接，请先连接 NAS")
    return root


def _catalog_root(config: dict[str, Any]) -> Path:
    _root(config)  # Do not silently create a local substitute for a missing mount.
    return Path(config["storage"]["local_cache_root"]) / "Collection-Catalog" / "Selections"


def _json(path: Path) -> dict[str, Any]:
    if path.stat().st_size > 2 * 1024 * 1024:
        raise ValueError("采集说明文件过大")
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError("采集说明格式无效")
    return data


def _child(folder: Path, value: Any) -> Path:
    name = str(value or "")
    if not name or Path(name).name != name or name in {".", ".."} or ";" in name or "\\" in name:
        raise ValueError("采集文件名无效")
    path = folder / name
    if path.is_symlink() or path.resolve().parent != folder.resolve():
        raise ValueError("采集文件不能指向目录之外")
    return path


def _iso(value: int) -> str:
    return datetime.fromtimestamp(value / 1_000_000, timezone.utc).isoformat() if value else ""


def _plain_clock_bounds(clock: Path) -> tuple[int, int]:
    """Read bounded CSV edges; full media/clock validation remains a run gate."""
    with clock.open("rb") as handle:
        header = handle.readline(65536).decode("utf-8-sig").strip()
        head = handle.read(262144).decode("utf-8", errors="ignore").splitlines()
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - 262144))
        tail = handle.read().decode("utf-8", errors="ignore").splitlines()
    fields = next(csv.reader([header]))
    time_column = next((field for field in ("global_timestamp_us", "frame_system_timestamp_us", "wallclock_us", "epoch_us") if field in fields), None)
    if time_column is None or "rgb_video_frame_index" not in fields:
        return 0, 0
    def values(lines: list[str]) -> list[int]:
        result = []
        for cells in csv.reader(lines):
            if len(cells) != len(fields):
                continue
            row = dict(zip(fields, cells, strict=True))
            if row.get("rgb_recorded") == "0" or not row.get("rgb_video_frame_index"):
                continue
            try:
                result.append(int(row[time_column]))
            except (ValueError, TypeError):
                continue
        return result
    first, last = values(head[:-1]), values(tail[1:])
    return (first[0], last[-1]) if first and last else (0, 0)


def _inspect(root: Path, video: Path, now: float, settle: float, allow_plain: bool = False) -> dict[str, Any]:
    relative = video.relative_to(root).as_posix()
    prefix = video.name[:-len("rgb.mp4")] if video.name.endswith("rgb.mp4") else None
    meta_path = video.with_name(f"{prefix}meta.json") if prefix is not None else video.with_suffix(".json")
    ready_path = video.with_name(f"{prefix}recording_ready.json") if prefix is not None else video.with_suffix(".ready.json")
    metadata = _json(meta_path) if meta_path.is_file() and not meta_path.is_symlink() else {}
    ready = _json(ready_path) if ready_path.is_file() and not ready_path.is_symlink() else {}
    merged = metadata | ready
    plain = allow_plain and not meta_path.exists() and not ready_path.exists()
    issues: list[str] = []
    if merged.get("rgb_file") and _child(video.parent, merged["rgb_file"]) != video:
        issues.append("视频与采集说明不匹配")
    if metadata and ready and metadata.get("recording_session_id") != ready.get("recording_session_id"):
        issues.append("采集会话信息不一致")
    clock_name = merged.get("frames_file") or (f"{prefix}frames.csv" if prefix is not None else f"{video.stem}.csv")
    if plain and not video.with_name(clock_name).is_file() and video.stem.lower().endswith("_rgb"):
        clock_name = video.stem[:-4] + "_帧时间戳.csv"
    clock = _child(video.parent, clock_name)
    snapshots = []
    for path in (video, clock, meta_path, ready_path):
        if path.is_file() and not path.is_symlink():
            stat = path.stat()
            snapshots.append((path.relative_to(root).as_posix(), stat.st_size, stat.st_mtime_ns))
    video_stat = video.stat()
    if video_stat.st_size == 0:
        issues.append("视频尚未写入")
    if not clock.is_file():
        issues.append("缺少配对的时间戳 CSV")
    else:
        with clock.open(encoding="utf-8-sig", newline="") as handle:
            header = next(csv.reader([handle.readline(65536)]), [])
        if not {"global_timestamp_us", "frame_system_timestamp_us", "wallclock_us", "epoch_us"}.intersection(header) or "rgb_video_frame_index" not in header:
            issues.append("CSV 缺少采集时间或视频帧序号")
    start = int(
        merged.get("recording_quality_window_start_global_us")
        or merged.get("recording_window_first_valid_global_us")
        or merged.get("recording_window_start_global_us")
        or merged.get("segment_start_us")
        or 0
    )
    end = int(
        merged.get("recording_quality_window_end_global_us")
        or merged.get("recording_window_last_valid_global_us")
        or merged.get("recording_window_end_global_us")
        or merged.get("segment_end_us")
        or 0
    )
    if plain and clock.is_file():
        start, end = _plain_clock_bounds(clock)
    if not start or end <= start:
        issues.append("缺少完整的采集时间范围")
    if not plain and (not ready or ready.get("ready") is not True or metadata.get("closed") is not True or metadata.get("frames_publish_state") != "finalized"):
        issues.append("等待采集完成标记")
    if ready and (ready.get("recording_complete") is not True or ready.get("recording_quality_status") != "complete"):
        issues.append("采集程序报告数据不完整")
    latest = max(item[2] for item in snapshots) / 1e9
    if now - latest < settle:
        issues.append("等待文件写入稳定")
    # This capture format embeds a physical camera identifier, not a view role.
    # Keep successive segments of that camera together; roles remain explicit.
    capture_name = re.fullmatch(r"分段\d{6}_(.+)_RGB", video.stem, re.IGNORECASE) if plain else None
    camera_identity = capture_name.group(1) if capture_name else relative
    camera_key = (
        ("capture-" if capture_name else "view-") + hashlib.sha256(camera_identity.encode()).hexdigest()[:12]
        if plain else str(merged.get("camera_key") or video.relative_to(root).parts[0])
    )
    return {
        "recording_id": hashlib.sha256(relative.encode()).hexdigest()[:24],
        "camera_key": camera_key,
        "camera_identity_source": "capture_filename" if capture_name else "relative_path" if plain else "recorder_metadata" if merged.get("camera_key") else "directory_layout",
        "relative_path": relative,
        "video_path": str(video),
        "frames_path": str(clock),
        "size_bytes": video_stat.st_size,
        "recording_session_id": str(merged.get("recording_session_id") or ""),
        "recording_start_us": start,
        "recording_end_us": end,
        "recording_start_time": _iso(start),
        "recording_end_time": _iso(end),
        "duration_seconds": (end - start) / 1e6 if start and end > start else None,
        "available": not issues,
        "requires_completion_confirmation": plain,
        "completion_source": "user_confirmation_required" if plain else "recorder_sidecars",
        "status": "available" if not issues else "attention",
        "issues": issues,
        "source_signature": hashlib.sha256(json.dumps(snapshots).encode()).hexdigest(),
        "updated_at": datetime.fromtimestamp(latest, timezone.utc).isoformat(),
    }


def _camera_directories(root: Path, settings: dict[str, Any], allow_plain: bool) -> list[Path]:
    configured = settings.get("camera_directories") or []
    if configured:
        if not isinstance(configured, list):
            raise ValueError("采集相机目录配置无效")
        cameras = []
        for value in configured:
            name = str(value or "")
            if not name or Path(name).name != name or name in {".", ".."}:
                raise ValueError("采集相机目录配置无效")
            cameras.append(root / name)
        return cameras
    if allow_plain:
        return [root]
    return sorted(root.glob(settings.get("camera_directory_glob", "*_cam*")))


def _merged_recording_intervals(
    items: list[dict[str, Any]],
) -> list[tuple[int, int]]:
    intervals = sorted(
        (
            int(item.get("recording_start_us") or 0),
            int(item.get("recording_end_us") or 0),
        )
        for item in items
        if int(item.get("recording_end_us") or 0)
        > int(item.get("recording_start_us") or 0)
    )
    merged: list[tuple[int, int]] = []
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def _intersect_recording_intervals(
    left: list[tuple[int, int]], right: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    intersections: list[tuple[int, int]] = []
    left_index = 0
    right_index = 0
    while left_index < len(left) and right_index < len(right):
        left_start, left_end = left[left_index]
        right_start, right_end = right[right_index]
        start = max(left_start, right_start)
        end = min(left_end, right_end)
        if end > start:
            intersections.append((start, end))
        if left_end <= right_end:
            left_index += 1
        else:
            right_index += 1
    return intersections


def _recording_batches(
    recordings: list[dict[str, Any]], settings: dict[str, Any]
) -> list[dict[str, Any]]:
    """Group recorder-native files into one-click batches by shared session id."""

    role_map = {
        str(camera): str(role)
        for camera, role in (settings.get("camera_role_map") or {}).items()
        if str(role) in {"first_person", "third_person"}
    }
    expected_cameras = {
        str(item) for item in (settings.get("camera_directories") or []) if str(item)
    }
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in recordings:
        session_id = str(item.get("recording_session_id") or "")
        if session_id:
            grouped[session_id].append(item)
    batches = []
    for session_id, items in grouped.items():
        cameras = {str(item.get("camera_key") or "") for item in items}
        items_by_camera: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in items:
            items_by_camera[str(item.get("camera_key") or "")].append(item)
        camera_intervals = {
            camera: _merged_recording_intervals(camera_items)
            for camera, camera_items in items_by_camera.items()
        }
        common_intervals: list[tuple[int, int]] | None = None
        for intervals in camera_intervals.values():
            common_intervals = (
                intervals
                if common_intervals is None
                else _intersect_recording_intervals(common_intervals, intervals)
            )
            if not common_intervals:
                break
        common_intervals = common_intervals or []
        overlap_start_us = common_intervals[0][0] if common_intervals else 0
        overlap_end_us = common_intervals[-1][1] if common_intervals else 0
        overlap_duration_us = sum(end - start for start, end in common_intervals)
        roles = {role_map.get(camera) for camera in cameras} - {None}
        issues = sorted(
            {
                str(issue)
                for item in items
                for issue in item.get("issues") or []
                if str(issue)
            }
        )
        missing_cameras = sorted(expected_cameras - cameras)
        if missing_cameras:
            issues.append("等待另一视角完成同一采集批次")
        unconfigured_cameras = sorted(cameras - role_map.keys())
        if unconfigured_cameras:
            issues.append("采集相机视角待确认")
        if roles != {"first_person", "third_person"}:
            issues.append("采集相机视角尚未配置完整")
        if not common_intervals:
            issues.append("同一采集会话的相机时间范围不重叠")
        available = bool(items) and all(item.get("available") for item in items) and not issues
        source_start_us = min(int(item.get("recording_start_us") or 0) for item in items)
        source_end_us = max(int(item.get("recording_end_us") or 0) for item in items)
        start_us = overlap_start_us if common_intervals else source_start_us
        end_us = overlap_end_us if common_intervals else source_end_us
        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "session_id": session_id,
                    "sources": sorted(
                        (item["recording_id"], item["source_signature"]) for item in items
                    ),
                    "roles": sorted(role_map.items()),
                },
                ensure_ascii=False,
            ).encode()
        ).hexdigest()
        batches.append(
            {
                "batch_id": f"nas-batch-{fingerprint[:24]}",
                "recording_session_id": session_id,
                "recording_start_us": start_us,
                "recording_end_us": end_us,
                "recording_start_time": _iso(start_us),
                "recording_end_time": _iso(end_us),
                "duration_seconds": (
                    overlap_duration_us / 1e6
                    if common_intervals
                    else (end_us - start_us) / 1e6
                    if end_us > start_us
                    else None
                ),
                "cross_view_overlap_start_us": overlap_start_us,
                "cross_view_overlap_end_us": overlap_end_us,
                "cross_view_overlap_seconds": overlap_duration_us / 1e6,
                "cross_view_overlap_window_count": len(common_intervals),
                "cross_view_overlap_windows": [
                    {"start_us": start, "end_us": end}
                    for start, end in common_intervals
                ],
                "source_recording_start_us": source_start_us,
                "source_recording_end_us": source_end_us,
                "camera_count": len(cameras),
                "recording_count": len(items),
                "size_bytes": sum(int(item.get("size_bytes") or 0) for item in items),
                "available": available,
                "issues": issues,
                "unconfigured_cameras": unconfigured_cameras,
                "recordings": [
                    {
                        "recording_id": item["recording_id"],
                        "camera_key": item["camera_key"],
                        "role": role_map.get(str(item.get("camera_key") or "")),
                        "relative_path": item["relative_path"],
                    }
                    for item in sorted(items, key=lambda row: row["relative_path"])
                ],
            }
        )
    batches.sort(
        key=lambda item: (item["recording_start_us"], item["batch_id"]), reverse=True
    )
    return batches


def scan_recordings(config: dict[str, Any]) -> dict[str, Any]:
    root = _root(config)
    settings = config["collection_ingest"]
    now = time.time()
    recordings, errors = [], []
    excluded = {Path(config["storage"][key]).resolve() for key in ("archive_root", "local_cache_root")}
    visited = 0
    truncated = False
    history_window_limited = False
    allow_plain = bool(settings.get("discover_plain_video_csv", False))
    # Bound discovery and exclude archives/caches at every depth.
    cameras = _camera_directories(root, settings, allow_plain)
    monitored_camera_directories: list[str] = []
    for camera in cameras:
        if not camera.is_dir() or (camera != root and camera.is_symlink()) or camera.resolve() in excluded:
            continue
        monitored_camera_directories.append(
            "." if camera == root else camera.relative_to(root).as_posix()
        )
        camera_recording_count = 0
        camera_recording_limit = int(settings.get("max_recordings_per_camera") or 0)
        def scan_error(exc: OSError) -> None:
            errors.append({"path": str(exc.filename), "message": "无法读取采集目录"})

        for folder, directories, files in os.walk(camera, followlinks=False, onerror=scan_error):
            current = Path(folder)
            depth = len(current.relative_to(camera).parts)
            max_depth = min(8, int(settings.get("max_scan_depth", 6))) if allow_plain else 2
            directories[:] = sorted(
                (
                    name
                    for name in directories
                    if not name.startswith((".", "#"))
                    and not (current / name).is_symlink()
                    and (current / name).resolve() not in excluded
                ),
                reverse=True,
            ) if depth < max_depth else []
            visited += 1
            if visited > int(settings.get("max_scan_directories", 20000)):
                truncated = True
                break
            for name in sorted(files):
                if not name.lower().endswith((".mp4", ".mov", ".mkv", ".avi", ".webm")) or name.startswith("."):
                    continue
                if not allow_plain and not name.lower().endswith("rgb.mp4"):
                    # Recorder-native batches publish RGB evidence alongside
                    # depth containers. Only RGB is a pipeline view source.
                    continue
                path = current / name
                if path.is_symlink():
                    continue
                try:
                    item = _inspect(
                        root,
                        path,
                        now,
                        float(settings.get("settle_seconds", 120)),
                        allow_plain,
                    )
                    item["configured_role"] = (settings.get("camera_role_map") or {}).get(
                        item["camera_key"]
                    )
                    recordings.append(item)
                    camera_recording_count += 1
                except (OSError, ValueError, TypeError, OverflowError) as exc:
                    errors.append({"path": path.relative_to(root).as_posix(), "message": str(exc)})
                if (
                    camera_recording_limit > 0
                    and camera_recording_count >= camera_recording_limit
                ):
                    history_window_limited = True
                    break
            if len(recordings) >= int(settings.get("max_recordings", 5000)):
                truncated = True
                break
            if (
                camera_recording_limit > 0
                and camera_recording_count >= camera_recording_limit
            ):
                break
        if truncated:
            break
    recordings.sort(key=lambda item: (item["recording_start_us"], item["relative_path"]), reverse=True)
    return {
        "mode": "directory_metadata", "recordings": recordings,
        "recording_count": len(recordings), "errors": errors, "truncated": truncated,
        "batches": _recording_batches(recordings, settings),
        "camera_directories": monitored_camera_directories,
        "camera_directory_count": len(monitored_camera_directories),
        "history_window_limited": history_window_limited,
        "max_recordings_per_camera": int(
            settings.get("max_recordings_per_camera") or 0
        ),
        "unconfigured_camera_directories": sorted(
            camera
            for camera in monitored_camera_directories
            if camera != "." and camera not in (settings.get("camera_role_map") or {})
        ),
        "generated_at": _iso(round(now * 1e6)),
        "source_copy_bytes": 0, "video_decode": False,
    }


def selection_path(config: dict[str, Any], collection_id: str, suffix: str = ".json") -> Path:
    if not re.fullmatch(r"nas-[a-f0-9]{24}", collection_id):
        raise ValueError("无效的 NAS 实验编号")
    return _catalog_root(config) / f"{collection_id}{suffix}"


def validate_selection(config: dict[str, Any], collection_id: str) -> dict[str, Any]:
    selected = _json(selection_path(config, collection_id))
    root = _root(config)
    for item in selected["recordings"]:
        video = root / item["relative_path"]
        if not video.resolve().is_relative_to(root.resolve()) or video.is_symlink():
            raise ValueError("素材已移出 NAS 采集目录")
        current = _inspect(root, video, time.time(), float(config["collection_ingest"].get("settle_seconds", 120)), bool(config["collection_ingest"].get("discover_plain_video_csv", False)))
        if current["requires_completion_confirmation"] and not item.get("completion_confirmed_by_user"):
            raise ValueError("请确认所选视频已采集完成")
        if not current["available"] or current["source_signature"] != item["source_signature"]:
            raise ValueError("所选素材已变化，请刷新后重新选择")
    return selected


def create_selection(config: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    name = str(payload.get("experiment_name") or "").strip()
    specs = payload.get("recordings")
    if not name or len(name) > 120 or not isinstance(specs, list) or not 2 <= len(specs) <= 1000:
        raise ValueError("请命名实验，并选择至少两个视角的素材")
    inventory = {item["recording_id"]: item for item in scan_recordings(config)["recordings"]}
    selected, seen = [], set()
    by_camera: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for spec in specs:
        if not isinstance(spec, dict):
            raise ValueError("素材选择格式无效")
        if not isinstance(spec.get("recording_id"), str):
            raise ValueError("素材选择格式无效")
        item = inventory.get(spec.get("recording_id"))
        role = spec.get("role") or (item or {}).get("configured_role")
        if item is None or not item["available"] or item["recording_id"] in seen:
            raise ValueError("素材不存在、尚未完成或被重复选择")
        if role not in {"first_person", "third_person"}:
            raise ValueError("请确认每个相机的拍摄视角")
        if item["requires_completion_confirmation"] and payload.get("recording_complete_confirmed") is not True:
            raise ValueError("请确认所选视频已采集完成且不再写入")
        seen.add(item["recording_id"])
        item = item | {"role": role, "completion_confirmed_by_user": payload.get("recording_complete_confirmed") is True}
        selected.append(item)
        by_camera[item["camera_key"]].append(item)
    if len(by_camera) < 2 or {item["role"] for item in selected} != {"first_person", "third_person"}:
        raise ValueError("至少需要两台相机，包含第一人称和第三人称")
    windows = []
    for items in by_camera.values():
        if len({item["role"] for item in items}) != 1 or len({item["recording_session_id"] for item in items}) != 1:
            raise ValueError("同一相机的片段必须属于同一采集会话和视角")
        items.sort(key=lambda item: (item["recording_start_us"], item["relative_path"]))
        intervals = [(item["recording_start_us"], item["recording_end_us"]) for item in items]
        if len(set(intervals)) != len(intervals):
            raise ValueError("同一机位包含重复采集时段，请只选择一份原片")
        windows.append((min(i["recording_start_us"] for i in items), max(i["recording_end_us"] for i in items)))
    if max(start for start, _ in windows) >= min(end for _, end in windows):
        raise ValueError("所选相机的采集时间不重叠，请选择同一实验的不同视角")
    collection_id = str(payload.get("_collection_id") or "")
    if collection_id and not re.fullmatch(r"nas-[a-f0-9]{24}", collection_id):
        raise ValueError("无效的 NAS 实验编号")
    if not collection_id:
        collection_id = "nas-" + uuid.uuid4().hex[:24]
    rows = []
    for camera, items in by_camera.items():
        start = min(i["recording_start_us"] for i in items)
        end = max(i["recording_end_us"] for i in items)
        rows.append(dict(zip(FIELDS, (
            collection_id, name, camera, camera, items[0]["role"], start, end,
            _iso(start), _iso(end), len(items),
            ";".join(i["video_path"] for i in items),
            ";".join(i["frames_path"] for i in items),
            max(i["updated_at"] for i in items), "",
        ), strict=True)))
    automated_batch = bool(payload.get("_batch_id"))
    receipt = {"schema_version": "visioncortex-nas-selection/1", "collection_id": collection_id,
               "experiment_name": name, "selected_at": _iso(round(time.time() * 1e6)),
               "membership_source": (
                   "user_selected_capture_batch" if automated_batch else "user_selection"
               ), "role_source": (
                   "configured_camera_role_map"
                   if all(not spec.get("role") for spec in specs)
                   else "user_selection"
               ),
               "capture_batch_id": payload.get("_batch_id") if automated_batch else None,
               "source_copy_bytes": 0, "recordings": selected, "rows": rows}
    root = _catalog_root(config)
    root.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    # Publish the receipt last: it is the authoritative catalog entry.
    for suffix, content in ((".csv", buffer.getvalue()), (".json", json.dumps(receipt, ensure_ascii=False, indent=2))):
        destination = selection_path(config, collection_id, suffix)
        temporary = destination.with_suffix(suffix + ".partial")
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    return receipt


def selected_rows(config: dict[str, Any]) -> tuple[list[dict[str, str]], dict[str, Any]]:
    root = _catalog_root(config)
    rows = []
    for path in sorted(root.glob("nas-*.json")):
        receipt = _json(path)
        rows.extend({key: str(value) for key, value in row.items()} for row in receipt["rows"])
    return rows, {"path": str(root), "cache_hit": False, "row_count": len(rows), "mode": "user_selected_recordings"}
