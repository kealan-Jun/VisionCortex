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



def _capture_session(value):
    """Read recorder integer IDs and their exact string representation."""
    if type(value) is int:
        return str(value) if value > 0 else None
    return value if isinstance(value, str) and value else None


def _derived_audio_clock(timing, metadata, ready, start, end):
    """Admit a declared replay sample clock, never invent packet receipts.

    Partial capture quality remains partial. The replay clock locates encoded
    samples on the replay timeline; it does not verify the original clock.
    """
    if any(row.get('artifact_kind') != 'derived_replay' or row.get('schema_version') != 2
           or row.get('task_audio') is not True
           or row.get('packet_receipt_evidence') != 'unavailable_for_derived_media'
           for row in (metadata, ready)) or metadata.get('global_timestamp_source') != 'derived_cfr_source_lineage' or (
               'global_timestamp_source' in ready and ready['global_timestamp_source'] != 'derived_cfr_source_lineage'):
        raise ValueError('derived_audio_clock_schema_unverified')
    lo, hi = metadata.get('segment_window_start_global_us'), metadata.get('segment_window_end_global_us')
    rate, samples, duration = (metadata.get(k) for k in ('sample_rate', 'audio_sample_count', 'audio_duration_us'))
    if (any(type(n) is not int for n in (lo, hi, rate, samples, duration)) or rate <= 0 or samples <= 0
            or hi <= lo or duration != hi-lo or start != lo
            or abs(samples*1000000/rate-duration) > 1 or abs(hi-end-1000000/rate) > 1
            or metadata.get('audio_duration_basis') != 'derived_decoded_sample_grid'
            or metadata.get('audio_timestamp_basis') != 'derived_decoded_sample_origin'
            or any(ready.get(k) != metadata.get(k) for k in (
                'segment_window_start_global_us', 'segment_window_end_global_us', 'segment_id'))):
        raise ValueError('derived_audio_sample_clock_mismatch')
    quality = metadata.get('quality_evidence')
    flags = metadata.get('quality_flags')
    if (not isinstance(quality, list) or len(quality) > 4096 or not isinstance(flags, list)
            or not all(isinstance(flag, str) for flag in flags)
            or ready.get('quality_flags') != flags or ready.get('quality_status') != metadata.get('quality_status')
            or any(not isinstance(row, dict) or type(row.get('start_us')) is not int
                or type(row.get('end_us')) is not int or not lo <= row['start_us'] < row['end_us'] <= hi
                for row in quality)
            or any(flag in flags and not any(row.get('kind') == flag for row in quality)
                   for flag in ('synthetic_silence', 'source_outage'))):
        raise ValueError('derived_audio_quality_ranges_unverified')
    previous, count = lo, 0
    with timing.open(encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            left, right = int(row['window_start_global_us']), int(row['window_end_global_us'])
            if (int(row['global_timestamp_us']) != left or left != previous or right-left != 1000000
                    or right > hi or row.get('time_basis') != 'derived_cfr_source_lineage'
                    or row.get('received_packets') not in ('', None)
                    or row.get('received_packets_basis') != 'unavailable_for_derived_media'):
                raise ValueError('derived_audio_window_clock_mismatch')
            evidence = json.loads(row['quality_evidence_json'])
            if not isinstance(evidence, list) or any(not isinstance(part, dict)
                    or type(part.get('start_us')) is not int or type(part.get('end_us')) is not int
                    or not left <= part['start_us'] < part['end_us'] <= right
                    or not any(part.get('kind') == original.get('kind')
                        and original['start_us'] <= part['start_us'] < part['end_us'] <= original['end_us']
                        for original in quality) for part in evidence):
                raise ValueError('derived_audio_window_quality_mismatch')
            previous, count = right, count+1
    if previous != hi or not count:
        raise ValueError('derived_audio_window_clock_incomplete')
    return {'kind': 'derived_replay_decoded_sample_clock', 'rows': count, 'received_packets': None,
            'source_clock_verified': False, 'capture_quality': metadata.get('quality_status'),
            'quality_flags': metadata.get('quality_flags', []), 'time_alignment': 'PARTIAL_EVIDENCE'}


def _independent_audio_clock(timing, audio_meta, audio_ready, start, end):
    if timing.stat().st_size > 2*1024*1024:
        raise ValueError('independent_audio_clock_exceeds_bound')
    if audio_meta.get('artifact_kind') == 'derived_replay':
        return _derived_audio_clock(timing, audio_meta, audio_ready, start, end)
    previous, count, received = None, 0, 0
    first_nonempty, last_nonempty, empty_windows = None, None, []
    with timing.open(encoding='utf-8-sig', newline='') as handle:
        rows = csv.DictReader(handle)
        fields = set(rows.fieldnames or [])
        if 'global_timestamp_us' not in fields:
            raise ValueError('independent_audio_clock_header_invalid')
        windows = bool(fields & {'window_start_global_us', 'window_end_global_us'})
        if windows and (not {'window_start_global_us', 'window_end_global_us', 'received_packets',
                            'clock_sync_valid'}.issubset(fields)
                or any(row.get('schema_version') != 2 or row.get('task_audio') is not True
                       for row in (audio_meta, audio_ready))
                or audio_meta.get('global_timestamp_source') != 'sender_system_timestamp_us_plus_clock_offset'):
            raise ValueError('independent_audio_window_clock_schema_unverified')
        for row in rows:
            stamp = int(row['global_timestamp_us'])
            if windows:
                left, right, packets = (int(row[k]) for k in
                    ('window_start_global_us', 'window_end_global_us', 'received_packets'))
                if (stamp != left or right-left != 1000000 or packets < 0
                        or previous is not None and left != previous
                        or packets and row['clock_sync_valid'] != '1'):
                    raise ValueError('independent_audio_window_clock_time_mismatch')
                if packets:
                    first_nonempty = first_nonempty or (left, right)
                    last_nonempty = (left, right)
                else:
                    empty_windows.append({'start_us': left, 'end_us': right, 'reason': 'no_received_packets'})
                previous = right
                received += packets
            else:
                if not start <= stamp <= end or previous is not None and stamp < previous:
                    raise ValueError('independent_audio_clock_time_mismatch')
                if previous is None and stamp != start:
                    raise ValueError('independent_audio_clock_origin_mismatch')
                previous = stamp
            count += 1
        if not count:
            raise ValueError('independent_audio_clock_empty')
        if windows:
            if first_nonempty is None or not first_nonempty[0] <= start < first_nonempty[1]:
                raise ValueError('independent_audio_window_clock_origin_mismatch')
            if not last_nonempty[0] < end <= last_nonempty[1]:
                raise ValueError('independent_audio_window_clock_end_mismatch')
            if (received <= 0 or any(type(row.get('received_packets')) is not int
                    or row['received_packets'] != received for row in (audio_meta, audio_ready))
                    or type(audio_meta.get('clock_sync_valid_packets')) is not int
                    or audio_meta['clock_sync_valid_packets'] != received):
                raise ValueError('independent_audio_window_clock_packet_mismatch')
    return {'kind': 'recorder_task_audio_v2_window_summary' if windows else 'explicit_audio_timestamp_rows',
            'rows': count, 'received_packets': received if windows else None,
            **({'empty_packet_windows': empty_windows} if windows else {})}


def _independent_stt(record, video, metadata, ready, now, settle):
    """Audio readiness has its own identity/clock/producer completion gate."""
    from .device_day_contract import digest
    result = {'stt_processable': False, 'stt_readiness_reasons': []}
    reasons = result['stt_readiness_reasons']
    audio = record['audio']
    if audio.get('status') != 'provided':
        reasons.append('independent_audio_not_ready')
        return result
    try:
        paths = {row['kind']: Path(row['path']) for row in audio.get('files') or []}
        if not all(kind in paths for kind in ('audio', 'metadata', 'ready', 'timing')):
            raise ValueError('independent_audio_files_incomplete')
        audio_meta, audio_ready = _json(paths['metadata']), _json(paths['ready'])
        session = _capture_session(record.get('recording_session_id'))
        if not session or any(_capture_session(row.get('recording_session_id')) != session
                              for row in (metadata, ready, audio_meta, audio_ready)):
            raise ValueError('independent_audio_session_unverified')
        if any(row.get('camera_key') and row['camera_key'] != record['camera_key']
               for row in (metadata, ready, audio_meta, audio_ready)):
            raise ValueError('independent_audio_camera_mismatch')
        if any(len({str(row[field]) for row in (metadata, ready, audio_meta, audio_ready)
                    if row.get(field) is not None}) > 1 for field in ('camera_id', 'sender_id')):
            raise ValueError('independent_audio_capture_identity_mismatch')
        prefix = video.name[:-len('rgb.mp4')] if video.name.endswith('rgb.mp4') else None
        identity_paths = ([video.with_name(prefix + suffix) for suffix in ('meta.json', 'recording_ready.json')]
                          if prefix is not None else [video.with_suffix('.json'), video.with_suffix('.ready.json')])
        if any(not path.is_file() or path.is_symlink() or now - path.stat().st_mtime < settle for path in identity_paths):
            raise ValueError('capture_identity_not_stable')
        derived = audio_meta.get('artifact_kind') == 'derived_replay'
        if audio_ready.get('ready') is not True or not derived and (audio_meta | audio_ready).get('audio_valid') is not True:
            raise ValueError('independent_audio_completion_unverified')
        start, end = audio.get('start_us'), audio.get('end_us')
        if type(start) is not int or type(end) is not int or not 0 < start < end:
            raise ValueError('independent_audio_time_unverified')
        if not record.get('recording_start_us') or record['recording_end_us'] <= record['recording_start_us']:
            raise ValueError('capture_time_identity_unverified')
        unsafe = [issue for issue in record['issues'] if issue not in {
            '视频尚未写入', '缺少原始视频', '缺少配对的时间戳 CSV', 'CSV 缺少采集时间或视频帧序号',
            '缺少完整的采集时间范围', '等待采集完成标记', '采集程序报告数据不完整', '等待文件写入稳定'}]
        if unsafe or audio.get('association_issues'):
            raise ValueError('capture_identity_or_audio_association_mismatch')
        clock_proof = _independent_audio_clock(paths['timing'], audio_meta, audio_ready, start, end)
        identity = {'schema_version': 'visioncortex-independent-recorder-audio/1',
            'recording_id': record['recording_id'], 'camera_key': record['camera_key'],
            'recording_session_id': session, 'start_us': start, 'end_us': end,
            'audio_source_signature': audio['source_signature']}
        result.update(stt_processable=True, stt_identity=identity, stt_source_signature=digest(identity),
                      stt_clock_evidence=clock_proof)
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as exc:
        reasons.append(str(exc)[:200] or type(exc).__name__)
    return result


def _inspect(root: Path, video: Path, now: float, settle: float, allow_plain: bool = False) -> dict[str, Any]:
    from .device_day_audio import recorder_audio
    relative = video.relative_to(root).as_posix()
    prefix = video.name[:-len("rgb.mp4")] if video.name.endswith("rgb.mp4") else None
    meta_path = video.with_name(f"{prefix}meta.json") if prefix is not None else video.with_suffix(".json")
    ready_path = video.with_name(f"{prefix}recording_ready.json") if prefix is not None else video.with_suffix(".ready.json")
    metadata = _json(meta_path) if meta_path.is_file() and not meta_path.is_symlink() else {}
    ready = _json(ready_path) if ready_path.is_file() and not ready_path.is_symlink() else {}
    merged = metadata | ready
    audio = recorder_audio(video, metadata, ready, now, settle)
    plain = allow_plain and not meta_path.exists() and not ready_path.exists()
    issues: list[str] = []
    if merged.get("rgb_file") and _child(video.parent, merged["rgb_file"]) != video:
        issues.append("视频与采集说明不匹配")
    if metadata and ready and metadata.get("recording_session_id") != ready.get("recording_session_id"):
        issues.append("采集会话信息不一致")
    clock_name = merged.get("frames_file") or (f"{prefix}frames.csv" if prefix is not None else f"{video.stem}.csv")
    if plain and not video.with_name(clock_name).is_file() and video.stem.lower().endswith("_rgb"):
        clock_name = video.stem[:-4] + "_帧时间戳.csv"
    if plain and not video.with_name(clock_name).is_file() and video.with_name(f"{video.stem}_frames.csv").is_file():
        clock_name = f"{video.stem}_frames.csv"
    clock = _child(video.parent, clock_name)
    snapshots = []
    for path in (video, clock, meta_path, ready_path):
        if path.is_file() and not path.is_symlink():
            stat = path.stat()
            snapshots.append((path.relative_to(root).as_posix(), stat.st_size, stat.st_mtime_ns))
    video_stat = video.stat() if video.is_file() else None
    if video_stat is None:
        issues.append('缺少原始视频')
    elif video_stat.st_size == 0:
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
    if clock.is_file() and (plain or not start or end <= start):
        start, end = _plain_clock_bounds(clock)
    if not start or end <= start:
        issues.append("缺少完整的采集时间范围")
        if (type(audio.get('start_us')) is int and type(audio.get('end_us')) is int
                and 0 < audio['start_us'] < audio['end_us']):
            start, end = audio['start_us'], audio['end_us']
    if not plain and (not ready or ready.get("ready") is not True or metadata.get("closed") is not True or metadata.get("frames_publish_state") != "finalized"):
        issues.append("等待采集完成标记")
    if ready and (ready.get("recording_complete") is not True or ready.get("recording_quality_status") != "complete"):
        issues.append("采集程序报告数据不完整")
    latest = max((item[2] for item in snapshots), default=round(now*1e9)) / 1e9
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
    from .capture_layout import METADATA_LAYOUT, metadata_location
    location = metadata_location(video)
    input_layout = {}
    if location:
        camera_key = str(merged.get('camera_key') or location['camera_key'])
        from .device_day_contract import archive_name
        wrong_camera = any(sidecar.get('camera_key') and str(sidecar['camera_key']) != location['camera_key']
                           for sidecar in (metadata, ready))
        if wrong_camera or (start and archive_name(camera_key, start) != location['archive']):
            issues.append(f"{location['root'].name} 目录与采集设备或日期不一致")
        input_layout = {'input_layout': METADATA_LAYOUT}
        if location['root'].name != 'MetaData':
            # Keep existing MetaData observation/receipt identities unchanged.
            input_layout['source_raw_directory'] = location['root'].name
    result = input_layout | {
        "recording_id": hashlib.sha256(relative.encode()).hexdigest()[:24],
        "camera_key": camera_key,
        "camera_identity_source": "device_day_metadata" if location else "capture_filename" if capture_name else "relative_path" if plain else "recorder_metadata" if merged.get("camera_key") else "directory_layout",
        "relative_path": relative,
        "video_path": str(video),
        "frames_path": str(clock),
        "size_bytes": video_stat.st_size if video_stat else None,
        "recording_session_id": str(merged.get("recording_session_id") or ""),
        "recording_start_us": start,
        "recording_end_us": end,
        "recording_start_time": _iso(start),
        "recording_end_time": _iso(end),
        "duration_seconds": (end - start) / 1e6 if start and end > start else None,
        "available": not issues,
        # Capture completeness is a quality attribute, not proof the file is
        # still being written. Device/day processing includes closed partials.
        "processable": not [issue for issue in issues if issue != "采集程序报告数据不完整"],
        "capture_complete": ready.get("recording_complete") is True and ready.get("recording_quality_status") == "complete",
        "requires_completion_confirmation": plain,
        "completion_source": "user_confirmation_required" if plain else "recorder_sidecars",
        "status": "available" if not issues else "attention",
        "issues": issues,
        "capture_quality": {
            "status": ready.get("recording_quality_status"),
            "reason": ready.get("recording_quality_reason"),
            "rgb_coverage_ratio": ready.get("rgb_coverage_ratio"),
            "rgb_max_frame_gap_us": ready.get("rgb_max_frame_gap_us"),
            "rgb_timestamp_out_of_order_count": ready.get("rgb_timestamp_out_of_order_count"),
            "scope": "recorder_reported_not_analysis_accuracy",
        },
        "source_signature": hashlib.sha256(json.dumps(snapshots).encode()).hexdigest(),
        "audio": audio,
        "updated_at": datetime.fromtimestamp(latest, timezone.utc).isoformat(),
    }
    return result | _independent_stt(result, video, metadata, ready, now, settle)


def _camera_directories(root: Path, settings: dict[str, Any], allow_plain: bool) -> list[Path]:
    from .capture_layout import configured_camera_names, discovery_mode
    configured = configured_camera_names(settings)
    if discovery_mode(settings) == 'metadata_only':
        return []
    if configured:
        return [root / name for name in configured]
    if allow_plain:
        return [root]
    from .capture_layout import validate
    validate(settings)
    cameras = set(root.glob(settings.get("camera_directory_glob", "*_cam*")))
    cameras.update(root / name for name in settings.get('additional_camera_directories') or [])
    return sorted(cameras)


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
    from .capture_layout import camera_group
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in recordings:
        session_id = str(item.get("recording_session_id") or "")
        if session_id:
            grouped[item.get('source_group', camera_group(settings, item['camera_key'])), session_id].append(item)
    batches = []
    for (source_group, session_id), items in grouped.items():
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
                "source_group": source_group,
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
                "device_day_processable": bool(items) and all(item.get("processable", item.get("available")) for item in items) and not unconfigured_cameras,
                "issues": issues,
                "issue_details": [
                    {"camera_key": item.get("camera_key"), "relative_path": item.get("relative_path"),
                     "issues": item.get("issues", []), "capture_quality": item.get("capture_quality", {})}
                    for item in items if item.get("issues")
                ],
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


def scan_recordings(config: dict[str, Any], *, on_record=None, skip_folders=()) -> dict[str, Any]:
    root = _root(config)
    settings = config["collection_ingest"]
    skip_folders = set(skip_folders)
    now = time.time()
    recordings, errors = [], []
    excluded = {Path(config["storage"][key]).resolve() for key in ("archive_root", "local_cache_root")}
    visited = 0
    truncated = False
    history_window_limited = False
    allow_plain = bool(settings.get("discover_plain_video_csv", False))
    # Bound discovery and exclude archives/caches at every depth.
    from .capture_layout import discovery_mode, metadata_directories
    from .device_day_contract import RAW_DIRECTORIES
    archive_root = Path(config['storage']['archive_root']).resolve()
    # Derived output siblings remain excluded even when source == archive.
    cameras = [p for p in _camera_directories(root, settings, allow_plain)
               if not p.resolve().is_relative_to(archive_root)]
    cameras.extend(metadata_directories(config))
    monitored_camera_directories: list[str] = []
    for camera in cameras:
        if not camera.is_dir() or (camera != root and camera.is_symlink()) or camera.resolve() in excluded:
            continue
        monitored_camera_directories.append(
            camera.parent.name[11:] if camera.name in RAW_DIRECTORIES else
            "." if camera == root else camera.relative_to(root).as_posix()
        )
        if str(camera) in skip_folders:
            continue
        camera_recording_count = 0
        camera_recording_limit = int(settings.get("max_recordings_per_camera") or 0)
        def scan_error(exc: OSError) -> None:
            errors.append({"path": str(exc.filename), "message": "无法读取采集目录"})

        for folder, directories, files in os.walk(camera, followlinks=False, onerror=scan_error):
            current = Path(folder)
            depth = len(current.relative_to(camera).parts)
            max_depth = min(8, int(settings.get("max_scan_depth", 6))) if allow_plain or camera.name in RAW_DIRECTORIES else 2
            directories[:] = sorted(
                (
                    name
                    for name in directories
                    if not name.startswith((".", "#"))
                    and str(current / name) not in skip_folders
                    and not (depth == 0 and re.fullmatch(r"\d{4}-\d{2}-\d{2}", name)
                             and ((settings.get("capture_date") and name != settings["capture_date"])
                                  or (settings.get("capture_since_date") and name < settings["capture_since_date"])))
                    and not (current / name).is_symlink()
                    and (current / name).resolve() not in excluded
                ),
                reverse=True,
            ) if depth < max_depth else []
            visited += 1
            if visited > int(settings.get("max_scan_directories", 20000)):
                truncated = True
                break
            candidates = set(files)
            if not allow_plain:
                # Reuse the same bounded metadata walk for independently
                # ready recorder audio; no parallel scanner or second queue.
                for sidecar_name in files:
                    if sidecar_name.endswith('meta.json') and not sidecar_name.endswith('audio_meta.json'):
                        sidecar = current/sidecar_name
                        try:
                            info = _json(sidecar) if not sidecar.is_symlink() else {}
                            declared = info.get('rgb_file')
                            if isinstance(declared, str) and _child(current, declared).parent == current:
                                candidates.add(declared)
                        except (OSError, ValueError, TypeError):
                            continue
            for name in sorted(candidates):
                if not name.lower().endswith((".mp4", ".mov", ".mkv", ".avi", ".webm")) or name.startswith("."):
                    continue
                if not allow_plain and not name.lower().endswith("rgb.mp4"):
                    # Recorder-native batches publish RGB evidence alongside
                    # depth containers. Only RGB is a pipeline view source.
                    continue
                path = current / name
                if path.is_symlink():
                    continue
                if camera.name == 'MetaVideo':
                    prefix = path.name[:-len('rgb.mp4')] if path.name.endswith('rgb.mp4') else None
                    native_sidecars = ([path.with_name(prefix + suffix) for suffix in ('meta.json', 'recording_ready.json')]
                                       if prefix is not None else [path.with_suffix('.json'), path.with_suffix('.ready.json')])
                    if not any(sidecar.is_file() and not sidecar.is_symlink() for sidecar in native_sidecars):
                        # Historical retained videos also live in MetaVideo.
                        # Their presence alone cannot establish new capture
                        # identity, readiness, clocks or a producer handoff.
                        continue
                try:
                    item = _inspect(
                        root,
                        path,
                        now,
                        float(settings.get("settle_seconds", 120)),
                        allow_plain,
                    )
                    from .input_availability import configured_record
                    item = configured_record(config, item)
                    recordings.append(item)
                    if on_record is not None:
                        on_record(item)
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
    monitored_camera_directories = list(dict.fromkeys(monitored_camera_directories))
    recordings.sort(key=lambda item: (item["recording_start_us"], item["relative_path"]), reverse=True)
    batches = _recording_batches(recordings, settings)
    for batch in batches:
        batch["analysis_ready"] = batch["device_day_processable"] if (config.get("device_day") or {}).get("enabled") else batch["available"]
    return {
        "mode": "directory_metadata", "discovery_mode": discovery_mode(settings), "recordings": recordings,
        "recording_count": len(recordings), "errors": errors, "truncated": truncated,
        "batches": batches,
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
            and not any(rule['directory_glob'].startswith(camera+'/')
                        and rule['camera_key'] in (settings.get('camera_role_map') or {})
                        for rule in settings.get('directory_camera_bindings') or [])
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
    from .capture_layout import camera_group
    if len({camera_group(config['collection_ingest'], item['camera_key']) for item in selected}) != 1:
        raise ValueError('不同实验室的素材不能合并为同一次实验')
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
