"""Resumable upload session operations and capacity admission."""

from __future__ import annotations
from .dependencies import ports
import hashlib
import hmac
import json
import math
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Any
from .contracts import ApplicationError
from ..device_registry import load_device_registry, resolve_view_role
from ..schemas import RunManifest, VideoSegmentInput, ViewInput
from ..storage import (
    initialize_nas_archive,
    safe_archive_name,
)
from ..upload_sessions import StorageReservationError, UploadSessionStore

from . import archive_read_model, run_submission_service, runtime_host


def _require_upload_store(settings: dict[str, Any]) -> UploadSessionStore:
    if runtime_host._upload_sessions is None:
        # Focused unit calls may not enter the ASGI lifespan. The production
        # server initializes this same store before accepting requests.
        runtime_host._upload_sessions = UploadSessionStore(
            runtime_host._queue_database_path(settings)
        )
    return runtime_host._upload_sessions


def _expire_stale_upload_sessions(settings: dict[str, Any]) -> None:
    store = _require_upload_store(settings)
    archive_root = archive_read_model._archive_root(settings).resolve()
    for expired in store.expire_stale():
        candidate = Path(expired["archive_root"]).resolve()
        if candidate.parent != archive_root or not candidate.name:
            continue
        if candidate.is_dir():
            try:
                shutil.rmtree(candidate)
            except OSError:
                # The remaining bytes are still reflected by disk_usage, so a
                # cleanup failure cannot over-admit the next reservation.
                continue


def _upload_policy(settings: dict[str, Any]) -> dict[str, Any]:
    configured = settings.get("web_upload") or {}
    chunk_size_mib = max(1, min(64, int(configured.get("chunk_size_mib", 16))))
    session_ttl_hours = max(1.0, float(configured.get("session_ttl_hours", 168.0)))
    processing_ratio = max(
        0.0, float(configured.get("processing_headroom_ratio", 0.35))
    )
    safety_ratio = max(0.0, float(configured.get("safety_headroom_ratio", 0.10)))
    minimum_safety_gib = max(
        0.0, float(configured.get("minimum_safety_headroom_gib", 10.0))
    )
    parallel_files = max(1, min(4, int(configured.get("parallel_files", 2))))
    full_file_sha256_max_gib = max(
        0.0, float(configured.get("full_file_sha256_max_gib", 4.0))
    )
    return {
        "chunk_size_bytes": chunk_size_mib * 1024 * 1024,
        "session_ttl_seconds": session_ttl_hours * 3600.0,
        "processing_headroom_ratio": processing_ratio,
        "safety_headroom_ratio": safety_ratio,
        "minimum_safety_bytes": math.ceil(minimum_safety_gib * 1024**3),
        "parallel_files": parallel_files,
        "full_file_sha256_max_bytes": math.ceil(full_file_sha256_max_gib * 1024**3),
        "prequeue_media_preflight_enabled": bool(
            configured.get("prequeue_media_preflight_enabled", True)
        ),
        "chunk_sha256_required": bool(configured.get("chunk_sha256_required", True)),
    }


def _upload_retention_mode(settings: dict[str, Any]) -> str:
    if settings["storage"].get("sync_to_nas"):
        return "nas_only"
    return "local_only"


def _unique_upload_archive_name(settings: dict[str, Any], experiment_name: str) -> str:
    base_name = safe_archive_name(experiment_name)
    archive_root = archive_read_model._archive_root(settings)
    with runtime_host._lock:
        if not (archive_root / base_name).exists():
            return base_name
        suffix = ports.datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"{base_name}-{suffix}-{uuid.uuid4().hex[:4]}"


def _parse_upload_session_files(
    payload: dict[str, Any],
    archive_path: Path,
    session_id: str,
    settings: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    specs = payload.get("view_specs")
    raw_files = payload.get("files")
    if not isinstance(specs, list) or not specs:
        raise ApplicationError(400, "view_specs 必须是非空数组")
    if not isinstance(raw_files, list) or not raw_files:
        raise ApplicationError(400, "files 必须是非空数组")

    normalized_files: list[dict[str, Any]] = []
    file_ids: set[str] = set()
    by_kind_index: dict[tuple[str, int], dict[str, Any]] = {}
    for position, item in enumerate(raw_files):
        if not isinstance(item, dict):
            raise ApplicationError(400, f"files[{position}] 必须是对象")
        kind = str(item.get("kind") or "")
        if kind not in {"video", "timestamp_csv", "audio"}:
            raise ApplicationError(400, f"files[{position}].kind 无效")
        try:
            file_index = int(item.get("file_index"))
            expected_bytes = int(item.get("size"))
        except (TypeError, ValueError) as exc:
            raise ApplicationError(400, f"files[{position}] 的序号或大小无效") from exc
        if file_index < 0 or expected_bytes <= 0:
            raise ApplicationError(400, f"files[{position}] 的序号或大小无效")
        file_id = str(item.get("file_id") or f"{kind}-{file_index}")
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", file_id) or file_id in file_ids:
            raise ApplicationError(400, f"files[{position}].file_id 无效或重复")
        key = (kind, file_index)
        if key in by_kind_index:
            raise ApplicationError(400, f"files[{position}] 的类型与序号重复")
        normalized = {
            "file_id": file_id,
            "kind": kind,
            "file_index": file_index,
            "source_name": str(item.get("name") or f"{kind}-{file_index}"),
            "expected_bytes": expected_bytes,
        }
        file_ids.add(file_id)
        by_kind_index[key] = normalized
        normalized_files.append(normalized)

    videos = sorted(
        (item for item in normalized_files if item["kind"] == "video"),
        key=lambda item: int(item["file_index"]),
    )
    if len(videos) < 2 or [item["file_index"] for item in videos] != list(
        range(len(videos))
    ):
        raise ApplicationError(
            400, "视频至少需要两路，且 video_index 必须从 0 连续编号"
        )
    if len(specs) < 2:
        raise ApplicationError(400, "至少需要两个视角")

    validation_views: list[ViewInput] = []
    normalized_specs: list[dict[str, Any]] = []
    used_video_indexes: set[int] = set()
    used_csv_indexes: set[int] = set()
    used_audio_indexes: set[int] = set()
    used_paths: set[Path] = set()
    try:
        registry = load_device_registry(
            ((settings or {}).get("storage") or {}).get("device_registry_path")
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ApplicationError(500, f"设备角色注册表不可用: {exc}") from exc
    for position, raw_spec in enumerate(specs):
        if not isinstance(raw_spec, dict):
            raise ApplicationError(400, f"view_specs[{position}] 必须是对象")
        try:
            calibration_hint_ms = float(raw_spec.get("calibration_hint_ms", 0.0))
        except (TypeError, ValueError) as exc:
            raise ApplicationError(
                400, f"view_specs[{position}] 的文件映射无效"
            ) from exc
        raw_segments = raw_spec.get("segments")
        segmented_layout = raw_segments is not None
        if segmented_layout:
            if not isinstance(raw_segments, list) or not raw_segments:
                raise ApplicationError(
                    400, f"view_specs[{position}].segments 必须是非空数组"
                )
            mapping_items = raw_segments
        else:
            mapping_items = [
                {
                    "video_index": raw_spec.get("video_index", position),
                    "csv_index": raw_spec.get("csv_index"),
                    "audio_index": raw_spec.get("audio_index"),
                    "audio_offset_ms": raw_spec.get("audio_offset_ms"),
                }
            ]
        mappings: list[
            tuple[int, int | None, dict[str, Any], dict[str, Any] | None]
        ] = []
        for segment_position, raw_mapping in enumerate(mapping_items):
            if not isinstance(raw_mapping, dict):
                raise ApplicationError(
                    400,
                    f"view_specs[{position}].segments[{segment_position}] 必须是对象",
                )
            try:
                video_index = int(raw_mapping.get("video_index"))
                csv_value = raw_mapping.get("csv_index")
                csv_index = int(csv_value) if csv_value is not None else None
            except (TypeError, ValueError) as exc:
                raise ApplicationError(
                    400,
                    f"view_specs[{position}] 第 {segment_position + 1} 段映射无效",
                ) from exc
            video_file = by_kind_index.get(("video", video_index))
            if video_file is None:
                raise ApplicationError(
                    400,
                    f"view_specs[{position}] 第 {segment_position + 1} 段引用了不存在的视频",
                )
            if video_index in used_video_indexes:
                raise ApplicationError(400, "同一个视频不能映射到多个视角或分片")
            used_video_indexes.add(video_index)
            csv_file = (
                by_kind_index.get(("timestamp_csv", csv_index))
                if csv_index is not None
                else None
            )
            if csv_index is not None and csv_file is None:
                raise ApplicationError(
                    400,
                    f"view_specs[{position}] 第 {segment_position + 1} 段引用了不存在的 CSV",
                )
            if csv_index is not None:
                if csv_index in used_csv_indexes:
                    raise ApplicationError(400, "同一个 CSV 不能映射到多个视角或分片")
                used_csv_indexes.add(csv_index)
            mappings.append((video_index, csv_index, video_file, csv_file))
        view_id = run_submission_service._safe_file_name(
            str(raw_spec.get("view_id") or f"view-{position + 1:02d}"),
            f"view-{position + 1:02d}",
        )
        requested_role = str(raw_spec.get("role") or "")
        role_resolution = resolve_view_role(
            registry,
            archive_path.name,
            view_id,
            requested_role,
        )
        role_resolution["input_source"] = "browser_user_declared"
        if view_id not in (registry.get("devices") or {}):
            role_resolution["resolution_source"] = "browser_user_declared"
        elif role_resolution.get("resolution_source") == "experiment_record_index":
            role_resolution["resolution_source"] = (
                "browser_user_declared_confirmed_by_device_registry"
            )
        if role_resolution.get("blocking_reasons") or not role_resolution.get(
            "resolved_role"
        ):
            raise ApplicationError(
                400,
                {
                    "message": f"机位 {view_id} 的视角角色与设备注册表冲突",
                    "view_id": view_id,
                    "blocking_reasons": role_resolution.get("blocking_reasons"),
                    "requested_role": requested_role,
                    "registry_expected_role": role_resolution.get(
                        "registry_expected_role"
                    ),
                },
            )
        role = str(role_resolution["resolved_role"])
        normalized_spec = {
            "view_id": view_id,
            "role": role,
            "calibration_hint_ms": calibration_hint_ms,
            "source_layout": "segments" if segmented_layout else "single",
            "role_resolution": role_resolution,
        }
        normalized_mappings = [
            {
                "video_index": video_index,
                **({"csv_index": csv_index} if csv_index is not None else {}),
            }
            for video_index, csv_index, _video_file, _csv_file in mappings
        ]
        audio_mappings = []
        for raw_mapping, normalized_mapping in zip(
            mapping_items, normalized_mappings, strict=True
        ):
            audio_file = None
            if raw_mapping.get("audio_index") is not None:
                try:
                    audio_index = int(raw_mapping["audio_index"])
                    raw_offset = raw_mapping.get("audio_offset_ms")
                    audio_offset = float(raw_offset) if raw_offset is not None else None
                    if audio_offset is not None and not math.isfinite(audio_offset):
                        raise ValueError("nonfinite audio offset")
                except (TypeError, ValueError) as exc:
                    raise ApplicationError(400, "录音映射或时间偏移无效") from exc
                audio_file = by_kind_index.get(("audio", audio_index))
                if audio_file is None or audio_index in used_audio_indexes:
                    raise ApplicationError(400, "录音文件不存在或重复映射")
                used_audio_indexes.add(audio_index)
                normalized_mapping.update(
                    audio_index=audio_index, audio_offset_ms=audio_offset
                )
            elif raw_mapping.get("audio_offset_ms") is not None:
                raise ApplicationError(400, "录音时间偏移必须关联录音文件")
            audio_mappings.append(audio_file)
        if segmented_layout:
            normalized_spec["segments"] = normalized_mappings
        else:
            normalized_spec.update(normalized_mappings[0])
        normalized_specs.append(normalized_spec)

        for segment_position, (
            _video_index,
            _csv_index,
            video_file,
            csv_file,
        ) in enumerate(mappings, 1):
            for file_item in (
                video_file,
                csv_file,
                audio_mappings[segment_position - 1],
            ):
                if file_item is None:
                    continue
                source_name = run_submission_service._safe_file_name(
                    file_item["source_name"],
                    "video" if file_item["kind"] == "video" else "timestamps",
                )
                prefix = (
                    f"segment-{segment_position:04d}-video-"
                    if file_item["kind"] == "video"
                    else f"segment-{segment_position:04d}-audio-"
                    if file_item["kind"] == "audio"
                    else f"segment-{segment_position:04d}-timestamps-"
                )
                stored_name = f"{prefix}{source_name}"
                final_path = (
                    archive_path / "Original-Experiment-Videos" / view_id / stored_name
                )
                if final_path in used_paths:
                    raise ApplicationError(400, "上传文件的目标路径发生冲突")
                used_paths.add(final_path)
                file_item.update(
                    {
                        "view_id": view_id,
                        "segment_ordinal": segment_position,
                        "stored_name": stored_name,
                        "final_path": final_path,
                        "partial_path": final_path.with_name(
                            f".{stored_name}.upload-{session_id}.partial"
                        ),
                    }
                )
        try:
            if segmented_layout:
                validation_views.append(
                    ViewInput(
                        view_id=view_id,
                        role=role,
                        segments=[
                            VideoSegmentInput(
                                video=Path(f"/{view_id}-{index:04d}.mp4"),
                                timestamps_csv=(
                                    Path(f"/{view_id}-{index:04d}.csv")
                                    if csv_file is not None
                                    else None
                                ),
                            )
                            for index, (_v, _c, _video_file, csv_file) in enumerate(
                                mappings, 1
                            )
                        ],
                        calibration_hint_ms=calibration_hint_ms,
                    )
                )
            else:
                _video_index, _csv_index, _video_file, csv_file = mappings[0]
                validation_views.append(
                    ViewInput(
                        view_id=view_id,
                        role=role,
                        video=Path(f"/{view_id}.mp4"),
                        timestamps_csv=(Path(f"/{view_id}.csv") if csv_file else None),
                        calibration_hint_ms=calibration_hint_ms,
                    )
                )
        except ValueError as exc:
            raise ApplicationError(400, f"view_specs[{position}] 无效: {exc}") from exc

    unused_audio = {
        int(item["file_index"]) for item in normalized_files if item["kind"] == "audio"
    } - used_audio_indexes
    if unused_audio:
        raise ApplicationError(400, "存在未映射的录音文件")
    unused_csvs = {
        int(item["file_index"])
        for item in normalized_files
        if item["kind"] == "timestamp_csv"
    } - used_csv_indexes
    if unused_csvs:
        raise ApplicationError(400, f"存在未映射的 CSV: {sorted(unused_csvs)}")
    unused_videos = {int(item["file_index"]) for item in videos} - used_video_indexes
    if unused_videos:
        raise ApplicationError(400, f"存在未映射的视频: {sorted(unused_videos)}")
    try:
        RunManifest(experiment_id=archive_path.name, views=validation_views)
    except ValueError as exc:
        raise ApplicationError(400, f"视角配置无效: {exc}") from exc

    return (
        normalized_specs,
        normalized_files,
        sum(int(item["expected_bytes"]) for item in normalized_files),
    )


def _public_upload_session(session: dict[str, Any]) -> dict[str, Any]:
    files = [
        {
            "file_id": item["file_id"],
            "kind": item["kind"],
            "file_index": int(item["file_index"]),
            "name": item["source_name"],
            "size": int(item["expected_bytes"]),
            "uploaded_bytes": int(item["uploaded_bytes"]),
            "completed": bool(item.get("content_hash") or item.get("sha256")),
            "sha256": item.get("sha256"),
            "content_hash": item.get("content_hash"),
            "content_hash_algorithm": item.get("content_hash_algorithm"),
        }
        for item in session["files"]
    ]
    return {
        "session_id": session["session_id"],
        "status": session["status"],
        "archive_name": session["archive_name"],
        "retention_mode": session["retention_mode"],
        "expected_source_bytes": int(session["expected_source_bytes"]),
        "reserved_bytes": int(session["reserved_bytes"]),
        "processing_headroom_bytes": int(session["processing_headroom_bytes"]),
        "safety_headroom_bytes": int(session["safety_headroom_bytes"]),
        "uploaded_bytes": sum(int(item["uploaded_bytes"]) for item in files),
        "expires_at_epoch": float(session["expires_at"]),
        "run_id": session.get("run_id"),
        "files": files,
    }


def _load_upload_session(
    settings: dict[str, Any], session_id: str, *, refresh_expiry: bool = False
) -> dict[str, Any]:
    if refresh_expiry:
        _expire_stale_upload_sessions(settings)
    store = _require_upload_store(settings)
    session = store.get(session_id)
    if session is None:
        raise ApplicationError(404, "上传会话不存在")
    policy = _upload_policy(settings)
    for item in session["files"]:
        final_path = Path(item["final_path"])
        partial_path = Path(item["partial_path"])
        active_path = final_path if final_path.is_file() else partial_path
        actual_bytes = active_path.stat().st_size if active_path.is_file() else 0
        if actual_bytes > int(item["expected_bytes"]):
            raise ApplicationError(
                500, f"服务器暂存文件超过声明大小: {item['file_id']}"
            )
        if actual_bytes != int(item["uploaded_bytes"]):
            store.update_progress(
                session_id,
                str(item["file_id"]),
                actual_bytes,
                expires_at=ports.time.time() + float(policy["session_ttl_seconds"]),
            )
            item["uploaded_bytes"] = actual_bytes
    if refresh_expiry and session["status"] == "open" and session["files"]:
        first = session["files"][0]
        store.update_progress(
            session_id,
            str(first["file_id"]),
            int(first["uploaded_bytes"]),
            expires_at=ports.time.time() + float(policy["session_ttl_seconds"]),
        )
        session = store.get(session_id) or session
    return session


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def create_upload_session(payload: dict[str, Any]) -> dict[str, Any]:
    settings = runtime_host._settings()
    _expire_stale_upload_sessions(settings)
    archive_root = archive_read_model._archive_root(settings).resolve()
    archive_root.mkdir(parents=True, exist_ok=True)
    experiment_name = str(payload.get("experiment_name") or "").strip()
    if not experiment_name:
        raise ApplicationError(400, "experiment_name 不能为空")

    session_id = uuid.uuid4().hex
    archive_name = _unique_upload_archive_name(settings, experiment_name)
    archive_path = archive_root / archive_name
    specs, files, expected_source_bytes = _parse_upload_session_files(
        payload, archive_path, session_id, settings
    )
    policy = _upload_policy(settings)
    processing_headroom = math.ceil(
        expected_source_bytes * float(policy["processing_headroom_ratio"])
    )
    safety_headroom = max(
        math.ceil(expected_source_bytes * float(policy["safety_headroom_ratio"])),
        int(policy["minimum_safety_bytes"]),
    )
    disk = shutil.disk_usage(archive_root)
    store = _require_upload_store(settings)
    try:
        capacity = store.reserve(
            session_id=session_id,
            archive_name=archive_name,
            archive_root=archive_path,
            storage_key=str(archive_root),
            experiment_name=experiment_name,
            view_specs=specs,
            retention_mode=_upload_retention_mode(settings),
            expected_source_bytes=expected_source_bytes,
            processing_headroom_bytes=processing_headroom,
            safety_headroom_bytes=safety_headroom,
            free_bytes=int(disk.free),
            files=files,
            expires_at=ports.time.time() + float(policy["session_ttl_seconds"]),
        )
    except StorageReservationError as exc:
        raise ApplicationError(
            507,
            {
                "message": "NAS 可用空间不足，任务尚未创建",
                "required_bytes": exc.required_bytes,
                "available_bytes": exc.available_bytes,
                "missing_bytes": exc.missing_bytes,
                "free_bytes": exc.free_bytes,
                "already_reserved_bytes": exc.already_reserved_bytes,
            },
        ) from exc
    try:
        settings["storage"]["active_archive_path"] = str(archive_path)
        initialize_nas_archive(settings, archive_name)
    except Exception:
        store.release(session_id)
        raise
    session = store.get(session_id)
    if session is None:
        raise ApplicationError(500, "上传会话创建后无法读取")
    return {
        **_public_upload_session(session),
        "chunk_size_bytes": int(policy["chunk_size_bytes"]),
        "parallel_files": int(policy["parallel_files"]),
        "chunk_sha256_required": bool(policy["chunk_sha256_required"]),
        "capacity": capacity,
        "resume_url": f"/api/upload-sessions/{session_id}",
        "finalize_url": f"/api/upload-sessions/{session_id}/finalize",
    }


def get_upload_session(session_id: str) -> dict[str, Any]:
    settings = runtime_host._settings()
    session = _load_upload_session(settings, session_id, refresh_expiry=True)
    return {
        **_public_upload_session(session),
        "chunk_size_bytes": int(_upload_policy(settings)["chunk_size_bytes"]),
        "parallel_files": int(_upload_policy(settings)["parallel_files"]),
        "chunk_sha256_required": bool(
            _upload_policy(settings)["chunk_sha256_required"]
        ),
    }


def cancel_upload_session(session_id: str) -> dict[str, Any]:
    settings = runtime_host._settings()
    store = _require_upload_store(settings)
    with runtime_host._upload_finalize_lock:
        session = store.get(session_id)
        if session is None:
            raise ApplicationError(404, "上传会话不存在")
        if session["status"] != "open" or session.get("run_id"):
            raise ApplicationError(409, "只有尚未提交分析的上传会话可以取消")
        archive_root = archive_read_model._archive_root(settings).resolve()
        candidate = Path(session["archive_root"]).resolve()
        if candidate.parent != archive_root or not candidate.name:
            raise ApplicationError(500, "上传会话归档路径超出允许清理范围")
        if not store.cancel(session_id):
            raise ApplicationError(409, "上传会话状态已经变化，未执行清理")
        cleanup_status = "not_present"
        if candidate.is_dir():
            try:
                shutil.rmtree(candidate)
                cleanup_status = "removed"
            except OSError:
                cleanup_status = "pending_retry"
        return {
            "session_id": session_id,
            "status": "cancelled",
            "released_bytes": int(session["reserved_bytes"]),
            "archive_cleanup": cleanup_status,
        }


async def upload_session_chunk(
    request: Any, session_id: str, file_id: str
) -> dict[str, Any]:
    settings = runtime_host._settings()
    store = _require_upload_store(settings)
    session = _load_upload_session(settings, session_id)
    if session["status"] != "open":
        raise ApplicationError(409, "上传会话已经结束，不能继续写入")
    item = next(
        (entry for entry in session["files"] if entry["file_id"] == file_id), None
    )
    if item is None:
        raise ApplicationError(404, "上传文件不存在")
    if item.get("content_hash") or item.get("sha256"):
        return {
            "session_id": session_id,
            "file_id": file_id,
            "uploaded_bytes": int(item["expected_bytes"]),
            "completed": True,
        }

    try:
        offset = int(request.headers.get("upload-offset", ""))
    except ValueError as exc:
        raise ApplicationError(400, "Upload-Offset 请求头无效") from exc
    partial_path = Path(item["partial_path"])
    final_path = Path(item["final_path"])
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    active_path = final_path if final_path.is_file() else partial_path
    actual_offset = active_path.stat().st_size if active_path.is_file() else 0
    if offset < 0 or offset > actual_offset:
        raise ApplicationError(
            409,
            f"上传位置不一致，服务器应从 {actual_offset} 字节继续",
            headers={"Upload-Offset": str(actual_offset)},
        )

    policy = _upload_policy(settings)
    maximum_chunk = int(policy["chunk_size_bytes"])
    expected_chunk_sha = request.headers.get("x-chunk-sha256")
    if expected_chunk_sha is not None:
        expected_chunk_sha = expected_chunk_sha.strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", expected_chunk_sha):
            raise ApplicationError(400, "X-Chunk-SHA256 请求头无效")
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            declared_length = int(content_length)
        except ValueError as exc:
            raise ApplicationError(400, "Content-Length 请求头无效") from exc
        if declared_length <= 0 or declared_length > maximum_chunk:
            raise ApplicationError(413, f"单个分块必须在 1 到 {maximum_chunk} 字节之间")
    expected_bytes = int(item["expected_bytes"])
    if offset < actual_offset:
        replay = bytearray()
        async for block in request.stream():
            if not block:
                continue
            replay.extend(block)
            if len(replay) > maximum_chunk or offset + len(replay) > actual_offset:
                raise ApplicationError(409, "续传校验范围超过服务器已保存的内容")
        if not replay:
            raise ApplicationError(400, "续传校验分块不能为空")
        replay_digest = hashlib.sha256(replay).hexdigest()
        if expected_chunk_sha and not hmac.compare_digest(
            expected_chunk_sha, replay_digest
        ):
            raise ApplicationError(422, "续传校验分块 SHA-256 无效")
        with active_path.open("rb") as handle:
            handle.seek(offset)
            retained = handle.read(len(replay))
        if not hmac.compare_digest(bytes(replay), retained):
            raise ApplicationError(409, "重新选择的文件与服务器断点内容不一致")
        return {
            "session_id": session_id,
            "file_id": file_id,
            "uploaded_bytes": actual_offset,
            "completed": actual_offset == expected_bytes,
            "replayed": True,
        }
    if actual_offset == expected_bytes:
        return {
            "session_id": session_id,
            "file_id": file_id,
            "uploaded_bytes": actual_offset,
            "completed": True,
        }
    if policy["chunk_sha256_required"] and not expected_chunk_sha:
        raise ApplicationError(428, "每个新上传分块必须提供 X-Chunk-SHA256")
    digest = hashlib.sha256()
    written = 0
    try:
        with partial_path.open("ab") as handle:
            async for block in request.stream():
                if not block:
                    continue
                written += len(block)
                if written > maximum_chunk or actual_offset + written > expected_bytes:
                    raise ApplicationError(413, "分块超过服务器声明的大小或文件边界")
                handle.write(block)
                digest.update(block)
            if written <= 0:
                raise ApplicationError(400, "上传分块不能为空")
            handle.flush()
            os.fsync(handle.fileno())
        if expected_chunk_sha and not hmac.compare_digest(
            expected_chunk_sha, digest.hexdigest()
        ):
            raise ApplicationError(422, "上传分块 SHA-256 校验失败，请重试该分块")
    except Exception:
        if partial_path.is_file():
            with partial_path.open("r+b") as handle:
                handle.truncate(actual_offset)
                handle.flush()
                os.fsync(handle.fileno())
        raise

    uploaded_bytes = actual_offset + written
    store.update_progress(
        session_id,
        file_id,
        uploaded_bytes,
        expires_at=ports.time.time() + float(policy["session_ttl_seconds"]),
    )
    store.record_chunk(
        session_id,
        file_id,
        actual_offset,
        written,
        digest.hexdigest(),
    )
    return {
        "session_id": session_id,
        "file_id": file_id,
        "uploaded_bytes": uploaded_bytes,
        "completed": uploaded_bytes == expected_bytes,
    }
