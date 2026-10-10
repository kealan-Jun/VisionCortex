"""Source-aware preparation and common durable task submission."""

from __future__ import annotations
from .dependencies import ports
import hashlib
import json
import os
import re
import shutil
import unicodedata
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import quote
import yaml
from .contracts import ApplicationError, DeferredTasks, UploadedFile
from ..input_seal import build_input_seal, write_input_seal
from ..schemas import RunManifest, VideoSegmentInput, ViewInput
from ..storage import (
    initialize_nas_archive,
    safe_archive_name,
)

from . import archive_read_model, run_read_model, runtime_host, upload_service


def _safe_file_name(value: str, fallback_stem: str = "file") -> str:
    original = Path(value).name
    suffix = Path(original).suffix
    stem = original[: -len(suffix)] if suffix else original
    ascii_stem = (
        unicodedata.normalize("NFKD", stem).encode("ascii", "ignore").decode("ascii")
    )
    ascii_suffix = (
        unicodedata.normalize("NFKD", suffix).encode("ascii", "ignore").decode("ascii")
    )
    cleaned_stem = (
        re.sub(r"[^A-Za-z0-9_.-]+", "-", ascii_stem).strip(" .-")
        or re.sub(r"[^A-Za-z0-9_.-]+", "-", fallback_stem).strip(" .-")
        or "file"
    )
    cleaned_suffix = re.sub(r"[^A-Za-z0-9.]", "", ascii_suffix).lower()
    return f"{cleaned_stem[:160]}{cleaned_suffix[:20]}"


def _reserve_archive(
    settings: dict[str, Any], experiment_name: str
) -> tuple[str, Path]:
    base_name = safe_archive_name(experiment_name)
    archive_root = archive_read_model._archive_root(settings)
    with runtime_host._lock:
        archive_name = base_name
        if (archive_root / archive_name).exists():
            suffix = ports.datetime.now().strftime("%Y%m%d-%H%M%S")
            archive_name = f"{base_name}-{suffix}-{uuid.uuid4().hex[:4]}"
        nas_root = initialize_nas_archive(settings, archive_name)
    return archive_name, nas_root


def _prepare_formal_run_staging(
    settings: dict[str, Any],
    archive_name: str,
    run_id: str,
    fixed_root: Path,
) -> tuple[Path, Path, Path]:
    """Route every Web-created derived package through the same promotion gate."""

    expected_fixed, staging_root, history_root = ports.fixed_archive_staging_paths(
        settings, archive_name, run_id
    )
    if expected_fixed.resolve() != fixed_root.resolve():
        raise RuntimeError(
            f"Formal archive reservation mismatch: {expected_fixed} != {fixed_root}"
        )
    settings["storage"]["active_archive_path"] = str(staging_root)
    settings["storage"]["run_output_mode"] = "nas_direct"
    settings["storage"]["formal_fixed_root"] = str(expected_fixed)
    settings["storage"]["formal_history_root"] = str(history_root)
    settings["storage"]["formal_promotion_required"] = True
    initialize_nas_archive(settings, archive_name)
    source_json = fixed_root / "JSON-Config-Files"
    if source_json.is_dir():
        shutil.copytree(
            source_json,
            staging_root / "JSON-Config-Files",
            dirs_exist_ok=True,
        )
    return expected_fixed, staging_root, history_root


def _manifest_source_receipts(
    manifest: RunManifest,
    known_sha256: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Describe manifest inputs without reopening large media files."""

    known_sha256 = known_sha256 or {}
    receipts: list[dict[str, Any]] = []
    for view in manifest.views:
        source_pairs: list[tuple[str, int | None, Path]] = []
        if view.video is not None:
            source_pairs.append(("video", None, Path(view.video)))
            if view.timestamps_csv is not None:
                source_pairs.append(("timestamp_csv", None, Path(view.timestamps_csv)))
            if view.audio is not None:
                source_pairs.append(("audio", None, Path(view.audio)))
        else:
            for ordinal, segment in enumerate(view.segments):
                source_pairs.append(("video", ordinal, Path(segment.video)))
                if segment.audio is not None:
                    source_pairs.append(("audio", ordinal, Path(segment.audio)))
                if segment.timestamps_csv is not None:
                    source_pairs.append(
                        ("timestamp_csv", ordinal, Path(segment.timestamps_csv))
                    )
        for kind, ordinal, path in source_pairs:
            stat = path.stat()
            resolved = str(path.resolve())
            sha256 = known_sha256.get(resolved) or known_sha256.get(str(path))
            receipt = {
                "kind": kind,
                "view_id": view.view_id,
                "segment_ordinal": ordinal,
                "path": str(path),
                "size_bytes": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "source_fingerprint_algorithm": "path_size_mtime",
            }
            if sha256:
                receipt.update(
                    {
                        "sha256": sha256,
                        "content_hash": sha256,
                        "content_hash_algorithm": "sha256",
                    }
                )
            receipts.append(receipt)
    return receipts


def _declared_role_resolution(manifest: RunManifest, source: str) -> dict[str, Any]:
    return {
        "schema_version": "visioncortex-view-role-resolution-ledger/1",
        "experiment_id": manifest.experiment_id,
        "input_source": source,
        "status": "resolved",
        "views": [
            {
                "camera_key": view.view_id,
                "resolved_role": view.role.value,
                "resolution_source": source,
                "blocking_reasons": [],
            }
            for view in manifest.views
        ],
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{uuid.uuid4().hex[:8]}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def _queue_recovery_receipt_path(nas_root: Path) -> Path:
    return nas_root / "JSON-Config-Files" / "Input-Manifests" / "queue_recovery.json"


def _write_queue_recovery_receipt(
    nas_root: Path,
    *,
    run_id: str,
    state: str,
    manifest_path: Path,
    input_seal_path: Path | None,
    ingest: dict[str, Any] | None,
    error: str | None = None,
    recovered_from_archive_receipt: bool = False,
    job_kind: str = "run",
    recovery_context: dict[str, Any] | None = None,
) -> Path:
    receipt_path = _queue_recovery_receipt_path(nas_root)
    previous = run_read_model._read_json(receipt_path, {}) or {}
    payload = {
        "schema_version": "visioncortex-queue-recovery/1",
        "run_id": run_id,
        "job_kind": job_kind,
        "state": state,
        "updated_at": ports.datetime.now().astimezone().isoformat(),
        "archive_root": str(nas_root),
        "manifest_relative_path": manifest_path.relative_to(nas_root).as_posix(),
        "input_seal_relative_path": (
            input_seal_path.relative_to(nas_root).as_posix()
            if input_seal_path is not None
            else None
        ),
        "ingest": ingest,
        "recovery_context": recovery_context,
        "error": error,
        "recovered_from_archive_receipt": bool(
            recovered_from_archive_receipt
            or previous.get("recovered_from_archive_receipt")
        ),
        "recovery_count": int(previous.get("recovery_count") or 0),
    }
    if recovered_from_archive_receipt:
        payload["recovery_count"] += 1
    _write_json_atomic(receipt_path, payload)
    return receipt_path


async def _save_upload_to_local_and_nas(
    upload: UploadedFile,
    local_destination: Path,
    nas_destination: Path,
    *,
    retain_local_copy: bool = True,
    archive_is_network: bool = True,
) -> dict[str, Any]:
    started = ports.time.perf_counter()
    if retain_local_copy:
        local_destination.parent.mkdir(parents=True, exist_ok=True)
    nas_destination.parent.mkdir(parents=True, exist_ok=True)
    local_partial = local_destination.with_name(
        f".{local_destination.name}.partial-{uuid.uuid4().hex[:8]}"
    )
    nas_partial = nas_destination.with_name(
        f".{nas_destination.name}.partial-{uuid.uuid4().hex[:8]}"
    )
    total = 0
    digest = hashlib.sha256()
    local_handle = None
    try:
        local_handle = local_partial.open("wb") if retain_local_copy else None
        with nas_partial.open("wb") as nas_handle:
            while block := await upload.read(8 * 1024 * 1024):
                if local_handle is not None:
                    local_handle.write(block)
                nas_handle.write(block)
                total += len(block)
                digest.update(block)
            if local_handle is not None:
                local_handle.flush()
                os.fsync(local_handle.fileno())
            nas_handle.flush()
            os.fsync(nas_handle.fileno())
        if local_handle is not None:
            local_handle.close()
            local_handle = None
            os.replace(local_partial, local_destination)
        os.replace(nas_partial, nas_destination)
    finally:
        if local_handle is not None:
            local_handle.close()
        for partial in (local_partial, nas_partial):
            if partial.exists():
                partial.unlink()
        await upload.close()
    duration_seconds = max(ports.time.perf_counter() - started, 1e-9)
    return {
        "filename": upload.filename,
        "bytes": total,
        "local_path": str(local_destination)
        if retain_local_copy
        else (str(nas_destination) if not archive_is_network else None),
        "nas_path": str(nas_destination) if archive_is_network else None,
        "analysis_path": str(
            local_destination if retain_local_copy else nas_destination
        ),
        "retention_mode": ("local_and_nas" if retain_local_copy else "nas_only")
        if archive_is_network
        else "local_only",
        "local_write_bytes": total
        * (int(retain_local_copy) + int(not archive_is_network)),
        "nas_write_bytes": total if archive_is_network else 0,
        "sha256": digest.hexdigest(),
        "duration_seconds": round(duration_seconds, 6),
        "effective_source_throughput_mib_s": round(
            total / duration_seconds / (1024 * 1024), 3
        ),
    }


def finalize_upload_session(
    session_id: str, background_tasks: DeferredTasks
) -> dict[str, Any]:
    settings = runtime_host._settings()
    store = upload_service._require_upload_store(settings)
    with runtime_host._upload_finalize_lock:
        session = upload_service._load_upload_session(settings, session_id)
        if session["status"] in {"finalized", "released"} and session.get("run_id"):
            run_id = str(session["run_id"])
            run = runtime_host._runs.get(run_id, {})
            return {
                "run_id": run_id,
                "state": run.get("state", "queued"),
                "status_url": f"/api/runs/{run_id}",
                "nas_output": str(session["archive_root"]),
                "archive_url": f"/?archive={quote(str(session['archive_name']))}",
                "queue_persistence": "sqlite",
            }
        if session["status"] != "open":
            raise ApplicationError(409, "上传会话不能提交分析")
        incomplete = [
            item
            for item in session["files"]
            if int(item["uploaded_bytes"]) != int(item["expected_bytes"])
        ]
        if incomplete:
            raise ApplicationError(
                409,
                {
                    "message": "仍有文件没有上传完成",
                    "files": [item["file_id"] for item in incomplete],
                },
            )

        policy = upload_service._upload_policy(settings)
        for item in session["files"]:
            if item.get("content_hash") or item.get("sha256"):
                continue
            partial_path = Path(item["partial_path"])
            final_path = Path(item["final_path"])
            source_path = final_path if final_path.is_file() else partial_path
            if not source_path.is_file() or source_path.stat().st_size != int(
                item["expected_bytes"]
            ):
                raise ApplicationError(409, f"暂存文件不完整: {item['file_id']}")
            expected_bytes = int(item["expected_bytes"])
            chunk_tree = store.chunk_tree_identity(
                session_id,
                str(item["file_id"]),
                expected_bytes,
            )
            sha256 = None
            if chunk_tree is None or expected_bytes <= int(
                policy["full_file_sha256_max_bytes"]
            ):
                sha256 = upload_service._sha256_path(source_path)
            content_hash = sha256 or chunk_tree
            content_hash_algorithm = (
                "sha256" if sha256 else "visioncortex-upload-chunk-tree-v1"
            )
            if not content_hash:
                raise ApplicationError(
                    409, f"上传分块完整性账本不完整: {item['file_id']}"
                )
            if source_path == partial_path:
                final_path.parent.mkdir(parents=True, exist_ok=True)
                os.replace(partial_path, final_path)
            store.complete_file(
                session_id,
                str(item["file_id"]),
                sha256,
                content_hash=content_hash,
                content_hash_algorithm=content_hash_algorithm,
            )

        session = store.get(session_id)
        if session is None:
            raise ApplicationError(500, "上传会话在完成校验后丢失")
        archive_name = str(session["archive_name"])
        nas_root = Path(session["archive_root"])
        files_by_key = {
            (str(item["kind"]), int(item["file_index"])): item
            for item in session["files"]
        }
        views: list[ViewInput] = []
        for spec in session["view_specs"]:
            if spec.get("source_layout") == "segments" or spec.get("segments"):
                views.append(
                    ViewInput(
                        view_id=str(spec["view_id"]),
                        role=spec["role"],
                        segments=[
                            VideoSegmentInput(
                                video=Path(
                                    files_by_key[
                                        ("video", int(mapping["video_index"]))
                                    ]["final_path"]
                                ),
                                timestamps_csv=(
                                    Path(
                                        files_by_key[
                                            (
                                                "timestamp_csv",
                                                int(mapping["csv_index"]),
                                            )
                                        ]["final_path"]
                                    )
                                    if mapping.get("csv_index") is not None
                                    else None
                                ),
                            )
                            for mapping in spec["segments"]
                        ],
                        calibration_hint_ms=float(spec.get("calibration_hint_ms", 0.0)),
                    )
                )
            else:
                video = files_by_key[("video", int(spec["video_index"]))]
                csv_index = spec.get("csv_index")
                timestamp_csv = (
                    Path(files_by_key[("timestamp_csv", int(csv_index))]["final_path"])
                    if csv_index is not None
                    else None
                )
                views.append(
                    ViewInput(
                        view_id=str(spec["view_id"]),
                        role=spec["role"],
                        video=Path(video["final_path"]),
                        timestamps_csv=timestamp_csv,
                        calibration_hint_ms=float(spec.get("calibration_hint_ms", 0.0)),
                    )
                )
        for view, spec in zip(views, session["view_specs"], strict=True):
            mappings = spec.get("segments") or [spec]
            for part, mapping in zip(view.segments or [view], mappings, strict=True):
                if mapping.get("audio_index") is not None:
                    part.audio = Path(
                        files_by_key[("audio", int(mapping["audio_index"]))][
                            "final_path"
                        ]
                    )
                    part.audio_offset_ms = mapping.get("audio_offset_ms")
        manifest = RunManifest(experiment_id=archive_name, views=views)
        manifest_path = nas_root / "JSON-Config-Files" / "input_manifest.yaml"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            yaml.safe_dump(
                manifest.model_dump(mode="json"), allow_unicode=True, sort_keys=False
            ),
            encoding="utf-8",
        )
        role_resolution = {
            "schema_version": "visioncortex-view-role-resolution-ledger/1",
            "experiment_id": archive_name,
            "input_source": "browser_user_declared_with_device_registry_validation",
            "status": "resolved",
            "resolved_first_person_views": sum(
                spec["role"] == "first_person" for spec in session["view_specs"]
            ),
            "resolved_third_person_views": sum(
                spec["role"] == "third_person" for spec in session["view_specs"]
            ),
            "views": [
                spec.get("role_resolution")
                or {
                    "camera_key": spec["view_id"],
                    "resolved_role": spec["role"],
                    "status": "legacy_session_without_registry_receipt",
                }
                for spec in session["view_specs"]
            ],
        }
        role_resolution_path = (
            nas_root / "JSON-Config-Files" / "view_role_resolution.json"
        )
        _write_json_atomic(role_resolution_path, role_resolution)
        preflight_path = (
            nas_root
            / "JSON-Config-Files"
            / "Input-Manifests"
            / "prequeue_input_preflight.json"
        )
        if policy["prequeue_media_preflight_enabled"]:
            try:
                preflight = ports.preflight_manifest_inputs(manifest, settings)
            except (OSError, RuntimeError, ValueError) as exc:
                _write_json_atomic(
                    preflight_path,
                    {
                        "schema_version": "visioncortex-prequeue-input-preflight/1",
                        "status": "failed",
                        "completed_at": ports.datetime.now().astimezone().isoformat(),
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )
                raise ApplicationError(
                    422,
                    {
                        "message": "输入文件未通过媒体或时钟预检，任务未进入 GPU 队列",
                        "error": f"{type(exc).__name__}: {exc}",
                        "receipt": str(preflight_path),
                    },
                ) from exc
        else:
            preflight = {
                "schema_version": "visioncortex-prequeue-input-preflight/1",
                "status": "skipped_by_configuration",
            }
        _write_json_atomic(preflight_path, preflight)

        completed_at = ports.datetime.now().astimezone().isoformat()
        duration_seconds = max(0.0, ports.time.time() - float(session["created_at"]))
        retention_mode = str(session["retention_mode"])
        upload_ledger = [
            {
                "filename": item["source_name"],
                "bytes": int(item["expected_bytes"]),
                "local_path": (
                    item["final_path"] if retention_mode == "local_only" else None
                ),
                "nas_path": (
                    item["final_path"] if retention_mode == "nas_only" else None
                ),
                "analysis_path": item["final_path"],
                "retention_mode": retention_mode,
                "local_write_bytes": (
                    int(item["expected_bytes"]) if retention_mode == "local_only" else 0
                ),
                "nas_write_bytes": (
                    int(item["expected_bytes"]) if retention_mode == "nas_only" else 0
                ),
                "sha256": item["sha256"],
                "content_hash": item.get("content_hash") or item.get("sha256"),
                "content_hash_algorithm": item.get("content_hash_algorithm")
                or "sha256",
                "file_id": item["file_id"],
                "kind": item["kind"],
                "view_id": item["view_id"],
                "segment_ordinal": item.get("segment_ordinal"),
            }
            for item in session["files"]
        ]
        ingest = {
            "request_started_perf": None,
            "request_started_epoch": float(session["created_at"]),
            "request_received_at": ports.datetime.fromtimestamp(
                float(session["created_at"])
            )
            .astimezone()
            .isoformat(),
            "original_retention_completed_at": completed_at,
            "duration_seconds": round(duration_seconds, 6),
            "file_count": len(upload_ledger),
            "video_count": sum(item["kind"] == "video" for item in session["files"]),
            "timestamp_csv_count": sum(
                item["kind"] == "timestamp_csv" for item in session["files"]
            ),
            "total_bytes": int(session["expected_source_bytes"]),
            "local_write_bytes": sum(
                int(item["local_write_bytes"]) for item in upload_ledger
            ),
            "nas_write_bytes": sum(
                int(item["nas_write_bytes"]) for item in upload_ledger
            ),
            "retention_mode": retention_mode,
            "destinations": [
                "nas_original_experiment_videos"
                if retention_mode == "nas_only"
                else "local_archive_original_experiment_videos"
            ],
            "upload_protocol": "resumable_chunks_v2",
            "protocol_compatibility": ["resumable_chunks_v1"],
            "chunk_integrity": "persistent_sha256_ledger",
            "full_file_sha256_max_bytes": int(policy["full_file_sha256_max_bytes"]),
            "upload_session_id": session_id,
            "storage_reservation_bytes": int(session["reserved_bytes"]),
        }
        ingest["effective_source_throughput_mib_s"] = round(
            ingest["total_bytes"] / max(duration_seconds, 1e-9) / (1024 * 1024), 3
        )
        input_seal = build_input_seal(
            manifest,
            source_mode="browser_resumable_upload",
            sources=[
                {
                    "file_id": item["file_id"],
                    "kind": item["kind"],
                    "view_id": item["view_id"],
                    "segment_ordinal": item.get("segment_ordinal"),
                    "path": item["analysis_path"],
                    "size_bytes": item["bytes"],
                    "sha256": item["sha256"],
                    "content_hash": item["content_hash"],
                    "content_hash_algorithm": item["content_hash_algorithm"],
                }
                for item in upload_ledger
            ],
            role_resolution=role_resolution,
            copied_source_bytes=int(session["expected_source_bytes"]),
            preflight=preflight,
        )
        input_seal_path = write_input_seal(
            nas_root / "JSON-Config-Files" / "Input-Manifests" / "input_seal.json",
            input_seal,
        )
        ingest["input_seal"] = {
            "path": str(input_seal_path),
            "sha256": input_seal["seal_sha256"],
            "source_mode": input_seal["source_mode"],
        }
        upload_record = {
            "run_id": f"upload-{session_id[:12]}",
            "archive_name": archive_name,
            "uploaded_at": completed_at,
            "files": upload_ledger,
            "manifest": manifest.model_dump(mode="json"),
            "web_ingest": {
                key: value
                for key, value in ingest.items()
                if key not in {"request_started_perf", "request_started_epoch"}
            },
        }
        _write_json_atomic(
            nas_root / "JSON-Config-Files" / "original_upload_manifest.json",
            upload_record,
        )
        _write_json_atomic(
            nas_root / "JSON-Config-Files" / "Stage-Receipts" / "original_ingest.json",
            {
                "schema_version": "visioncortex-stage-receipt/1",
                "stage": "original_ingest",
                "status": "completed",
                "completed_at": completed_at,
                "run_elapsed_seconds": round(duration_seconds, 6),
                "stage_duration_seconds": round(duration_seconds, 6),
                "archive_mode": settings["storage"].get("run_output_mode", "local"),
                "archive_root": str(nas_root),
                "retention_mode": retention_mode,
                "upload_protocol": "resumable_chunks_v2",
                "source_copy_bytes": int(session["expected_source_bytes"]),
                "storage_reservation_bytes": int(session["reserved_bytes"]),
                "artifacts": [
                    "Original-Experiment-Videos",
                    "JSON-Config-Files/original_upload_manifest.json",
                    "JSON-Config-Files/Input-Manifests/input_seal.json",
                    "JSON-Config-Files/Input-Manifests/prequeue_input_preflight.json",
                    "JSON-Config-Files/view_role_resolution.json",
                ],
                "token_ledger": "JSON-Config-Files/run_metrics.json",
            },
        )

        run_id = str(upload_record["run_id"])
        settings["storage"]["sync_to_nas"] = retention_mode == "nas_only"
        submitted = submission_service.submit_prepared_input(
            manifest=manifest,
            settings=settings,
            archive_name=archive_name,
            run_id=run_id,
            archive_root=nas_root,
            background_tasks=background_tasks,
            ingest=ingest,
            before_enqueue=lambda: store.assign_run(session_id, run_id),
            include_archive_url=True,
            include_recovery=True,
        )
        try:
            store.finalize(session_id, run_id)
        except ValueError:
            refreshed = store.get(session_id)
            if refreshed is None or refreshed["status"] != "released":
                raise
        return submitted


async def create_run(
    request: Any,
    background_tasks: DeferredTasks,
    experiment_name: str,
    view_specs_json: str,
    videos: list[UploadedFile],
    timestamp_csvs: list[UploadedFile] | None = None,
) -> dict[str, Any]:
    try:
        specs = json.loads(view_specs_json)
        if not isinstance(specs, list) or not specs:
            raise ValueError("必须是非空数组")
    except (json.JSONDecodeError, ValueError) as exc:
        raise ApplicationError(400, f"view_specs_json 无效: {exc}") from exc
    settings = runtime_host._settings()
    archive_name, nas_root = _reserve_archive(settings, experiment_name)
    settings["storage"]["active_archive_path"] = str(nas_root)
    run_id = uuid.uuid4().hex[:12]
    local_root = Path(settings["storage"]["local_input_root"]) / archive_name / run_id
    retention_mode = str(
        settings["storage"].get("web_upload_retention_mode", "local_and_nas")
    ).lower()
    if retention_mode not in {"local_and_nas", "nas_only", "local_only"}:
        raise ApplicationError(
            500, f"Unsupported web_upload_retention_mode: {retention_mode}"
        )
    retain_local_copy = retention_mode == "local_and_nas"
    saved_videos: list[Path] = []
    saved_csvs: list[Path] = []
    upload_ledger: list[dict[str, Any]] = []
    try:
        for index, upload in enumerate(videos):
            spec = next(
                (item for item in specs if int(item.get("video_index", -1)) == index),
                {},
            )
            view_id = _safe_file_name(
                str(spec.get("view_id") or f"view-{index + 1:02d}"),
                f"view-{index + 1:02d}",
            )
            filename = _safe_file_name(
                upload.filename or f"video-{index + 1:02d}.mp4",
                f"video-{index + 1:02d}",
            )
            local_destination = local_root / view_id / filename
            nas_destination = (
                nas_root / "Original-Experiment-Videos" / view_id / filename
            )
            upload_ledger.append(
                await _save_upload_to_local_and_nas(
                    upload,
                    local_destination,
                    nas_destination,
                    retain_local_copy=retain_local_copy,
                    archive_is_network=bool(settings["storage"].get("sync_to_nas")),
                )
            )
            saved_videos.append(Path(upload_ledger[-1]["analysis_path"]))
        for index, upload in enumerate(timestamp_csvs or []):
            spec = next(
                (item for item in specs if int(item.get("csv_index", -1)) == index), {}
            )
            view_id = _safe_file_name(
                str(spec.get("view_id") or f"view-{index + 1:02d}"),
                f"view-{index + 1:02d}",
            )
            filename = _safe_file_name(
                upload.filename or f"timestamps-{index + 1:02d}.csv",
                f"timestamps-{index + 1:02d}",
            )
            local_destination = local_root / view_id / filename
            nas_destination = (
                nas_root / "Original-Experiment-Videos" / view_id / filename
            )
            upload_ledger.append(
                await _save_upload_to_local_and_nas(
                    upload,
                    local_destination,
                    nas_destination,
                    retain_local_copy=retain_local_copy,
                    archive_is_network=bool(settings["storage"].get("sync_to_nas")),
                )
            )
            saved_csvs.append(Path(upload_ledger[-1]["analysis_path"]))
        view_inputs = []
        for position, spec in enumerate(specs):
            video_index = int(spec.get("video_index", position))
            csv_index = spec.get("csv_index")
            view_inputs.append(
                ViewInput(
                    view_id=str(spec["view_id"]),
                    role=spec["role"],
                    video=saved_videos[video_index],
                    timestamps_csv=saved_csvs[int(csv_index)]
                    if csv_index is not None
                    else None,
                    calibration_hint_ms=float(spec.get("calibration_hint_ms", 0.0)),
                )
            )
        manifest = RunManifest(experiment_id=archive_name, views=view_inputs)
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ApplicationError(400, f"视角映射无效: {exc}") from exc
    manifest_payload = yaml.safe_dump(
        manifest.model_dump(mode="json"), allow_unicode=True, sort_keys=False
    )
    manifest_path = nas_root / "JSON-Config-Files" / "input_manifest.yaml"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(manifest_payload, encoding="utf-8")
    if retain_local_copy:
        local_manifest_path = local_root / "manifest.yaml"
        local_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        local_manifest_path.write_text(manifest_payload, encoding="utf-8")
    role_resolution = _declared_role_resolution(
        manifest, "browser_legacy_upload_declared"
    )
    _write_json_atomic(
        nas_root / "JSON-Config-Files" / "view_role_resolution.json",
        role_resolution,
    )
    sha_by_path = {
        str(Path(item["analysis_path"]).resolve()): str(item["sha256"])
        for item in upload_ledger
    }
    input_seal = build_input_seal(
        manifest,
        source_mode="browser_legacy_upload",
        sources=_manifest_source_receipts(manifest, sha_by_path),
        role_resolution=role_resolution,
        copied_source_bytes=sum(int(item["bytes"]) for item in upload_ledger),
    )
    write_input_seal(
        nas_root / "JSON-Config-Files" / "Input-Manifests" / "input_seal.json",
        input_seal,
    )
    upload_record = {
        "run_id": run_id,
        "archive_name": archive_name,
        "uploaded_at": ports.datetime.now().isoformat(),
        "files": upload_ledger,
        "manifest": manifest.model_dump(mode="json"),
    }
    ingest_ended_at = ports.datetime.now().astimezone().isoformat()
    ingest_started_perf = float(
        getattr(request.state, "ingest_started_perf", ports.time.perf_counter())
    )
    ingest = {
        "request_started_perf": ingest_started_perf,
        "request_started_epoch": float(
            getattr(request.state, "ingest_started_epoch", ports.time.time())
        ),
        "request_received_at": getattr(
            request.state,
            "ingest_started_at",
            ports.datetime.now().astimezone().isoformat(),
        ),
        "original_retention_completed_at": ingest_ended_at,
        "duration_seconds": round(ports.time.perf_counter() - ingest_started_perf, 6),
        "file_count": len(upload_ledger),
        "video_count": len(saved_videos),
        "timestamp_csv_count": len(saved_csvs),
        "total_bytes": sum(int(item["bytes"]) for item in upload_ledger),
        "local_write_bytes": sum(
            int(item["local_write_bytes"]) for item in upload_ledger
        ),
        "nas_write_bytes": sum(int(item["nas_write_bytes"]) for item in upload_ledger),
        "retention_mode": retention_mode,
        "destinations": (
            ["local_archive_original_experiment_videos"]
            if not settings["storage"].get("sync_to_nas")
            else ["local_input", "nas_original_experiment_videos"]
            if retain_local_copy
            else ["nas_original_experiment_videos"]
        ),
    }
    ingest["effective_source_throughput_mib_s"] = round(
        ingest["total_bytes"] / max(ingest["duration_seconds"], 1e-9) / (1024 * 1024),
        3,
    )
    upload_record["web_ingest"] = {
        key: value
        for key, value in ingest.items()
        if key not in {"request_started_perf", "request_started_epoch"}
    }
    record_path = nas_root / "JSON-Config-Files" / "original_upload_manifest.json"
    record_path.write_text(
        json.dumps(upload_record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_json_atomic(
        nas_root / "JSON-Config-Files" / "Stage-Receipts" / "original_ingest.json",
        {
            "schema_version": "visioncortex-stage-receipt/1",
            "stage": "original_ingest",
            "status": "completed",
            "completed_at": ingest_ended_at,
            "run_elapsed_seconds": ingest["duration_seconds"],
            "stage_duration_seconds": ingest["duration_seconds"],
            "archive_mode": settings["storage"].get("run_output_mode", "local"),
            "archive_root": str(nas_root),
            "retention_mode": retention_mode,
            "source_copy_bytes": ingest["local_write_bytes"],
            "artifacts": [
                "Original-Experiment-Videos",
                "JSON-Config-Files/original_upload_manifest.json",
                "JSON-Config-Files/Input-Manifests/input_seal.json",
                "JSON-Config-Files/view_role_resolution.json",
            ],
            "token_ledger": "JSON-Config-Files/run_metrics.json",
        },
    )
    return submission_service.submit_prepared_input(
        manifest=manifest,
        settings=settings,
        archive_name=archive_name,
        run_id=run_id,
        archive_root=nas_root,
        background_tasks=background_tasks,
        ingest=ingest,
        include_archive_url=True,
    )


def _create_run_from_paths(
    payload,
    background_tasks,
    *,
    reserved_run_id=None,
    reserved_archive=None,
    runtime_settings=None,
):
    try:
        manifest = RunManifest.model_validate(payload)
    except ValueError as exc:
        raise ApplicationError(400, str(exc)) from exc
    settings = runtime_settings or runtime_host._settings()
    archive_name, nas_root = reserved_archive or _reserve_archive(
        settings, manifest.experiment_id
    )
    if archive_name != manifest.experiment_id:
        manifest = manifest.model_copy(update={"experiment_id": archive_name})
    run_id = reserved_run_id or uuid.uuid4().hex[:12]
    fixed_root, staging_root, _history_root = _prepare_formal_run_staging(
        settings, archive_name, run_id, nas_root
    )
    manifest_path = staging_root / "JSON-Config-Files" / "input_manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            manifest.model_dump(mode="json"), allow_unicode=True, sort_keys=False
        ),
        encoding="utf-8",
    )
    try:
        role_resolution = _declared_role_resolution(manifest, "api_from_paths_declared")
        _write_json_atomic(
            staging_root / "JSON-Config-Files" / "view_role_resolution.json",
            role_resolution,
        )
        input_seal = build_input_seal(
            manifest,
            source_mode="api_from_paths",
            sources=_manifest_source_receipts(manifest),
            role_resolution=role_resolution,
            copied_source_bytes=0,
        )
        write_input_seal(
            staging_root / "JSON-Config-Files" / "Input-Manifests" / "input_seal.json",
            input_seal,
        )
    except OSError as exc:
        raise ApplicationError(422, f"输入路径不可读取: {exc}") from exc
    return submission_service.submit_prepared_input(
        manifest=manifest,
        settings=settings,
        archive_name=archive_name,
        run_id=run_id,
        archive_root=nas_root,
        background_tasks=background_tasks,
        prepared_paths=(fixed_root, staging_root, _history_root),
    )


class RunSubmissionService:
    """Commit prepared source input through one recoverable queue boundary.

    Source adapters retain their seal algorithm, receipts and storage reservation.
    This tail owns staging, recovery receipt, queued state and durable enqueue.
    """

    @staticmethod
    def submit_prepared_input(
        *,
        manifest: RunManifest,
        settings: dict[str, Any],
        archive_name: str,
        run_id: str,
        archive_root: Path,
        background_tasks: DeferredTasks,
        ingest: dict[str, Any] | None = None,
        prepared_paths: tuple[Path, Path, Path] | None = None,
        before_enqueue=None,
        include_archive_url: bool = False,
        include_recovery: bool = False,
    ) -> dict[str, Any]:
        fixed_root, staging_root, history_root = (
            prepared_paths
            or _prepare_formal_run_staging(settings, archive_name, run_id, archive_root)
        )
        receipt = _write_queue_recovery_receipt(
            staging_root,
            run_id=run_id,
            state="queued",
            manifest_path=staging_root / "JSON-Config-Files" / "input_manifest.yaml",
            input_seal_path=staging_root
            / "JSON-Config-Files"
            / "Input-Manifests"
            / "input_seal.json",
            ingest=ingest,
            recovery_context={
                "formal_fixed_root": str(fixed_root),
                "formal_history_root": str(history_root),
            },
        )
        if before_enqueue is not None:
            before_enqueue()
        existing = (
            runtime_host._persistent_queue.get_job(run_id)
            if runtime_host._persistent_queue is not None
            else None
        )
        if existing is None:
            runtime_host._update(
                run_id,
                state="queued",
                progress=0.0,
                experiment_id=manifest.experiment_id,
                nas_output=str(fixed_root),
                nas_staging=str(staging_root),
            )
            fallback_args = (run_id, manifest, settings, staging_root)
            if ingest is not None:
                fallback_args += (ingest,)
            persistence = runtime_host._schedule_job(
                background_tasks,
                run_id=run_id,
                kind="run",
                payload={
                    "manifest": manifest.model_dump(mode="json"),
                    "settings": settings,
                    "nas_root": str(staging_root),
                    "ingest": ingest,
                },
                fallback=runtime_host._execute,
                fallback_args=fallback_args,
            )
        else:
            persistence = "sqlite"
        result = {
            "run_id": run_id,
            "state": "queued",
            "status_url": f"/api/runs/{run_id}",
            "nas_output": str(fixed_root),
            "nas_staging": str(staging_root),
            "queue_persistence": persistence,
        }
        if include_archive_url:
            result["archive_url"] = f"/?archive={quote(archive_name)}"
        if include_recovery:
            result.update(
                queue_disaster_recovery="archive_receipt",
                queue_recovery_receipt=str(receipt),
            )
        return result


submission_service = RunSubmissionService()
