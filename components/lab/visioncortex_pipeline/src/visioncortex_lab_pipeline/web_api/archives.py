"""HTTP adapters for archives"""

from __future__ import annotations
from ..application.dependencies import ports
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.parse import quote
from ..pathing import archive_contains
from fastapi import HTTPException, Query
from fastapi.responses import FileResponse
from .. import speech, speech_worker
from ..media_preview import cached_video_poster

from ..application import archive_read_model, run_read_model, runtime_host

from fastapi import APIRouter

router = APIRouter()


def staging_archive_detail(run_id: str, section: str = "all") -> dict[str, Any]:
    if section not in {"all", "library-materials", "library-reports"}:
        raise HTTPException(status_code=400, detail="Unsupported staging section")
    root = run_read_model._resolve_staging_run(run_id)
    record = runtime_host._runs.get(run_id) or {}
    name = record.get("experiment_id") or root.parent.name
    result = archive_read_model._archive_detail_from_root(
        root,
        name,
        staging_run_id=run_id,
        library_section=section if section != "all" else None,
    )
    if record.get("read_only"):
        result.update(read_only=True, retry_available=False)
    return result


def staging_file(run_id: str, path: str, poster: bool = False) -> FileResponse:
    root = run_read_model._resolve_staging_run(run_id)
    from ..artifact_reader import resolve, storage_status

    try:
        candidate = resolve(root, path, historical=True)
    except (OSError, ValueError) as exc:
        code, message = storage_status(exc)
        raise HTTPException(code, message) from exc
    if poster:
        return _video_poster_response(candidate, root)
    # Staging media can be rebuilt at the same URL. Revalidate cached ranges so
    # the browser does not combine an earlier MP4 index with newer media bytes.
    return FileResponse(
        candidate,
        headers={
            "Cache-Control": "private, no-cache",
            "X-Content-Type-Options": "nosniff",
        },
    )


def _video_poster_response(candidate: Path, root: Path) -> FileResponse:
    relative = candidate.relative_to(root.resolve())
    if candidate.suffix.lower() != ".mp4" or relative.parts[0] not in {
        "Experiment-Clips",
        "Key-Materials",
    }:
        raise HTTPException(400, "仅为已有实验片段和素材生成封面")
    runtime = runtime_host._settings().get("storage", {}).get("local_runtime_root")
    if not runtime:
        raise HTTPException(503, "本地封面缓存尚未配置")
    try:
        path = cached_video_poster(candidate, Path(runtime))
    except (OSError, ValueError, TimeoutError, subprocess.SubprocessError) as exc:
        raise HTTPException(503, "封面暂不可用，仍可点击播放视频") from exc
    return FileResponse(
        path,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "private, no-cache",
            "X-Content-Type-Options": "nosniff",
            "X-VisionCortex-Preview": "first-decoded-frame",
        },
    )


def _folder_open_command(
    root: Path,
    *,
    os_name: str | None = None,
    platform: str | None = None,
) -> list[str]:
    effective_os_name = os.name if os_name is None else os_name
    effective_platform = sys.platform if platform is None else platform
    if effective_os_name == "nt":
        return ["explorer.exe", str(root)]
    if effective_platform == "darwin":
        return ["open", str(root)]
    opener = shutil.which("xdg-open")
    if opener:
        return [opener, str(root)]
    raise RuntimeError("No supported desktop folder opener is installed")


def open_archive_folder(archive_name: str) -> dict[str, str]:
    root = archive_read_model._resolve_archive(archive_name)
    try:
        command = _folder_open_command(root)
    except RuntimeError as exc:
        raise HTTPException(501, str(exc)) from exc
    subprocess.Popen(command, start_new_session=True)
    return {"status": "opened", "path": str(root)}


def open_staging_folder(run_id: str) -> dict[str, str]:
    """Open an already resolved run directory without promoting its evidence."""
    root = run_read_model._resolve_staging_run(run_id)
    try:
        command = _folder_open_command(root)
    except RuntimeError as exc:
        raise HTTPException(501, str(exc)) from exc
    subprocess.Popen(command, start_new_session=True)
    return {"status": "opened", "path": str(root)}


def _experiment_speech_response(
    root: Path,
    name: str,
    staging: bool,
    query: str,
    offset: int,
    limit: int,
    chunk: str | None,
    release: str | None = None,
    *,
    fold: bool = False,
    phrase: str | None = None,
    aliases: bool = True,
    hint: str | None = None,
) -> dict[str, Any]:
    pointer = ports.read_current_release_pointer(root) or {} if not staging else {}
    current = str(pointer.get("release_id") or "") or None
    if release is not None and release != current:
        raise HTTPException(409, "实验归档已更新，请刷新页面")
    try:
        if current:
            if (
                ports.lightweight_release_integrity(root, pointer)
                != "release_manifest_verified"
            ):
                raise ValueError("invalid release manifest")
            manifest_path = (
                root / str(pointer.get("release_manifest") or "")
            ).resolve()
            if not archive_contains(manifest_path, root):
                raise ValueError("invalid release path")
            for filename in (
                "speech.json",
                "speech_understanding.json",
                "speech_timeline.json",
                "speech_search.json",
                "speech_search_receipt.json",
                "capture_quality.json",
                "speech_group_understanding.json",
            ):
                index = root / "JSON-Config-Files" / filename
                if index.is_file():
                    release_manifest = (
                        run_read_model._read_json(manifest_path, {}) or {}
                    )
                    expected = next(
                        (
                            item
                            for item in release_manifest.get("manifests", {}).get(
                                "JSON-Config-Files", []
                            )
                            if item.get("path") == filename
                        ),
                        None,
                    )
                    observed = speech_worker.file_record(index)
                    if not expected or observed != {
                        "size": expected.get("size_bytes"),
                        "sha256": expected.get("sha256"),
                    }:
                        raise ValueError("speech index does not match release")
        result = speech.archive_result(
            root,
            query,
            offset,
            limit,
            chunk,
            fold=fold,
            phrase=phrase,
            aliases=aliases,
            hint=hint,
        )
        result["refresh_targets"] = []
        recording_path = root / "JSON-Config-Files/speech_understanding.json"
        if staging and recording_path.is_file():
            result["refresh_targets"] = [
                {
                    "id": f"recording:{i}",
                    "label": f"录音理解第 {i + 1} 段",
                    "revision": speech_worker.sha256(recording_path),
                    "start_global_ms": part["speech_context"]["start_global_ms"],
                    "end_global_ms": part["speech_context"]["end_global_ms"],
                }
                for i, part in enumerate(
                    (result.get("model_understanding") or {}).get("parts", [])
                )
                if part.get("speech_context")
            ]
        groups_path = root / "JSON-Config-Files/experiment_group_understanding.json"
        if groups_path.is_file():
            from ..speech_refresh import apply, input_path

            groups = apply(
                root,
                (run_read_model._read_json(groups_path, {}) or {}).get("groups", []),
            )
            result["group_understanding"] = groups
            if staging:
                result["refresh_targets"].extend(
                    {
                        "id": "group:" + group["group_id"],
                        "label": group.get("experiment_name") or group["group_id"],
                        "revision": speech_worker.sha256(groups_path),
                        "start_global_ms": group["global_start_ms"],
                        "end_global_ms": group["global_end_ms"],
                    }
                    for group in groups
                    if input_path(root, group["group_id"]).is_file()
                )
        from ..speech_timeline import load as load_speech_timeline

        timeline = load_speech_timeline(root)
        if timeline:
            prefix = f"/api/{'staging-runs' if staging else 'archives'}/{quote(name)}/speech-video"
            timeline["videos"] = [
                {key: value for key, value in video.items() if not key.startswith("_")}
                | {
                    "url": prefix
                    + f"?view={quote(video['view_id'])}&part={video['segment_ordinal']}&timeline={timeline['sha256']}"
                    + (f"&release={quote(current)}" if current else "")
                }
                for video in timeline["videos"]
            ]
        result["timeline"] = timeline
        for source in result["sources"]:
            original = (source.get("original") or {}).get("file")
            if original:
                source_path = (root / original["path"]).resolve()
                if (
                    not archive_contains(source_path, root)
                    or not source_path.is_file()
                    or source_path.stat().st_size != original["size"]
                ):
                    raise ValueError("原始录音归档不可用")
                original["url"] = (
                    archive_read_model._staging_file_url(name, original["path"])
                    if staging
                    else archive_read_model._file_url(name, original["path"], current)
                )
            for part in source["chunks"]:
                for spec in part["files"].values():
                    spec["url"] = (
                        archive_read_model._staging_file_url(name, spec["path"])
                        if staging
                        else archive_read_model._file_url(name, spec["path"], current)
                    )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise HTTPException(409, "实验录音产物不可用或完整性检查未通过") from exc
    if not staging and (ports.read_current_release_pointer(root) or {}) != pointer:
        raise HTTPException(409, "实验归档正在更新，请刷新页面")
    result["release_id"] = current
    return result


def archive_speech(
    name: str,
    q: str = Query(default="", max_length=200),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=500),
    chunk: str | None = None,
    release: str | None = None,
    fold: bool = False,
    phrase: str | None = None,
    aliases: bool = True,
    hint: str | None = None,
) -> dict[str, Any]:
    return _experiment_speech_response(
        archive_read_model._resolve_archive(name),
        name,
        False,
        q,
        offset,
        limit,
        chunk,
        release,
        fold=fold,
        phrase=phrase,
        aliases=aliases,
        hint=hint,
    )


def staging_speech(
    run_id: str,
    q: str = Query(default="", max_length=200),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=500),
    chunk: str | None = None,
    fold: bool = False,
    phrase: str | None = None,
    aliases: bool = True,
    hint: str | None = None,
) -> dict[str, Any]:
    return _experiment_speech_response(
        run_read_model._resolve_staging_run(run_id),
        run_id,
        True,
        q,
        offset,
        limit,
        chunk,
        fold=fold,
        phrase=phrase,
        aliases=aliases,
        hint=hint,
    )


def search_speech_library(
    q: str = Query(min_length=1, max_length=200),
    archive_offset: int = Query(default=0, ge=0),
    aliases: bool = True,
) -> dict:
    roots = [
        (name, root, False)
        for name, root in archive_read_model._search_archive_roots(None)
    ]
    for run_id, state in sorted(list(runtime_host._runs.items())):
        if state.get("parent_run_id") or state.get("state") not in {
            "failed",
            "completed",
            "partial",
            "interrupted",
        }:
            continue
        try:
            roots.append((run_id, run_read_model._resolve_staging_run(run_id), True))
        except HTTPException:
            continue
    rows, unavailable = [], []
    for name, root, staging in roots[archive_offset : archive_offset + 10]:
        if not (root / "JSON-Config-Files/speech.json").is_file():
            continue
        try:
            if not staging:
                root = archive_read_model._resolve_archive(name)
            result = _experiment_speech_response(
                root, name, staging, q, 0, 100, None, aliases=aliases
            )
            for row in result["segments"]:
                params = (
                    f"chunk={quote(row['chunk_id'])}&t={row['playback_start_seconds']}"
                )
                rows.append(
                    {
                        **row,
                        "experiment": root.parent.name if staging else name,
                        "run_id": name if staging else None,
                        "experiment_matches": result["total"],
                        "href": f"#/{'stage' if staging else 'archive'}/{quote(name)}/speech?{params}",
                    }
                )
        except HTTPException:
            unavailable.append({"name": name, "reason": "来源或索引完整性未通过"})
    return {
        "segments": rows,
        "unavailable": unavailable,
        "searched_experiments": min(10, max(0, len(roots) - archive_offset)),
        "total_experiments": len(roots),
        "per_experiment_limit": 100,
        "next_archive_offset": archive_offset + 10
        if archive_offset + 10 < len(roots)
        else None,
        "evidence_kind": "spoken_mention",
        "physical_action_confirmation": False,
    }


def evaluate_speech(collection: str, name: str, reference: dict) -> dict:
    from ..speech_search import evaluate

    if collection not in {"archives", "staging-runs"}:
        raise HTTPException(404, "实验不存在")
    staging = collection == "staging-runs"
    root = (
        run_read_model._resolve_staging_run(name)
        if staging
        else archive_read_model._resolve_archive(name)
    )
    _experiment_speech_response(root, name, staging, "", 0, 1, None)
    try:
        return evaluate(root, reference)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise HTTPException(422, str(exc)) from exc


def staging_speech_video(
    run_id: str, view: str, timeline: str, part: int = Query(ge=0)
) -> FileResponse:
    from ..speech_timeline import video_path

    try:
        path = video_path(
            run_read_model._resolve_staging_run(run_id), timeline, view, part
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise HTTPException(409, "同步视频不可用或输入身份已变化") from exc
    return FileResponse(
        path,
        media_type="video/mp4",
        headers={
            "Cache-Control": "private, no-cache",
            "X-Content-Type-Options": "nosniff",
        },
    )


def archive_speech_video(
    name: str,
    view: str,
    timeline: str,
    part: int = Query(ge=0),
    release: str | None = None,
) -> FileResponse:
    from ..speech_timeline import video_path

    root = archive_read_model._resolve_archive(name)
    # Reuse the release manifest check before granting original-source access.
    _experiment_speech_response(root, name, False, "", 0, 1, None, release)
    try:
        path = video_path(root, timeline, view, part)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise HTTPException(409, "同步视频不可用或输入身份已变化") from exc
    return FileResponse(
        path,
        media_type="video/mp4",
        headers={
            "Cache-Control": "private, no-cache",
            "X-Content-Type-Options": "nosniff",
        },
    )


def search_key_events(
    archive: str | None = None,
    q: str | None = None,
    action_type: str | None = None,
    parent_event_id: str | None = None,
    cross_view: bool | None = None,
    liquid_state_status: str | None = None,
    liquid_present: bool | None = None,
    visible_flow: bool | None = None,
    material_ready: bool | None = None,
    start_us: int | None = None,
    end_us: int | None = None,
    cursor: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
) -> dict[str, Any]:
    return archive_read_model.search_key_events(
        archive=archive,
        q=q,
        action_type=action_type,
        parent_event_id=parent_event_id,
        cross_view=cross_view,
        liquid_state_status=liquid_state_status,
        liquid_present=liquid_present,
        visible_flow=visible_flow,
        material_ready=material_ready,
        start_us=start_us,
        end_us=end_us,
        cursor=cursor,
        limit=limit,
    )


router.get("/api/key-events")(search_key_events)


def search_staging_key_events(
    run_id: str,
    q: str | None = None,
    action_type: str | None = None,
    parent_event_id: str | None = None,
    cross_view: bool | None = None,
    liquid_state_status: str | None = None,
    liquid_present: bool | None = None,
    visible_flow: bool | None = None,
    start_us: int | None = None,
    end_us: int | None = None,
    cursor: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
) -> dict[str, Any]:
    return archive_read_model.search_staging_key_events(
        run_id=run_id,
        q=q,
        action_type=action_type,
        parent_event_id=parent_event_id,
        cross_view=cross_view,
        liquid_state_status=liquid_state_status,
        liquid_present=liquid_present,
        visible_flow=visible_flow,
        start_us=start_us,
        end_us=end_us,
        cursor=cursor,
        limit=limit,
    )


router.get("/api/staging-runs/{run_id}/key-events")(search_staging_key_events)


def staging_key_event(run_id: str, event_uid: str) -> dict[str, Any]:
    return archive_read_model.staging_key_event(run_id=run_id, event_uid=event_uid)


router.get("/api/staging-runs/{run_id}/key-events/{event_uid}")(staging_key_event)


def indexed_physical_changes(
    archive: str | None = None,
    object_id: str | None = None,
    object_role: str | None = None,
    action_type: str | None = None,
    parent_event_id: str | None = None,
    start_us: int | None = None,
    end_us: int | None = None,
    cursor: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
) -> dict[str, Any]:
    return archive_read_model.indexed_physical_changes(
        archive=archive,
        object_id=object_id,
        object_role=object_role,
        action_type=action_type,
        parent_event_id=parent_event_id,
        start_us=start_us,
        end_us=end_us,
        cursor=cursor,
        limit=limit,
    )


router.get("/api/physical-changes")(indexed_physical_changes)


def indexed_key_event(event_uid: str, archive: str | None = None) -> dict[str, Any]:
    return archive_read_model.indexed_key_event(event_uid=event_uid, archive=archive)


router.get("/api/key-events/{event_uid}")(indexed_key_event)


def indexed_evidence(evidence_uid: str, archive: str | None = None) -> dict[str, Any]:
    return archive_read_model.indexed_evidence(
        evidence_uid=evidence_uid, archive=archive
    )


router.get("/api/evidence/{evidence_uid:path}")(indexed_evidence)


def archive_detail(
    archive_name: str,
    section: str = "all",
    cursor: str | None = None,
    limit: int = Query(default=24, ge=1, le=100),
) -> dict[str, Any]:
    return archive_read_model.archive_detail(
        archive_name=archive_name, section=section, cursor=cursor, limit=limit
    )


router.get("/api/archives/{archive_name}")(archive_detail)

router.get("/api/staging-runs/{run_id}/archive")(staging_archive_detail)


def archive_file(
    archive: str,
    path: str,
    release: str | None = None,
    poster: bool = False,
) -> FileResponse:
    root = archive_read_model._resolve_archive(archive).resolve()
    from ..artifact_reader import resolve, storage_status

    try:
        candidate = resolve(root, path, historical=True)
    except (OSError, ValueError) as exc:
        code, message = storage_status(exc)
        raise HTTPException(code, message) from exc
    headers = {
        "Accept-Ranges": "bytes",
        "X-Content-Type-Options": "nosniff",
        "Cache-Control": "private, no-cache",
    }
    if release:
        pointer = ports.read_current_release_pointer(root) or {}
        if str(pointer.get("release_id") or "") != release:
            raise HTTPException(409, "该素材链接对应的归档版本已更新，请刷新页面")
        if (
            ports.lightweight_release_integrity(root, pointer)
            != "release_manifest_verified"
        ):
            raise HTTPException(409, "正式归档发布清单完整性校验失败")
        manifest_path = (root / str(pointer.get("release_manifest") or "")).resolve()
        if not archive_contains(manifest_path, root) or not manifest_path.is_file():
            raise HTTPException(409, "正式归档发布清单不可用")
        manifest = run_read_model._read_json(manifest_path, {}) or {}
        relative = candidate.relative_to(root)
        directory = relative.parts[0] if relative.parts else ""
        inner_path = (
            Path(*relative.parts[1:]).as_posix() if len(relative.parts) > 1 else ""
        )
        receipt = next(
            (
                item
                for item in (manifest.get("manifests") or {}).get(directory, [])
                if str(item.get("path") or "") == inner_path
            ),
            None,
        )
        if (
            receipt is None
            or int(receipt.get("size_bytes") or -1) != candidate.stat().st_size
        ):
            raise HTTPException(409, "正式归档素材与发布清单不一致")
        digest = str(receipt.get("sha256") or "")
        headers.update(
            {
                "Cache-Control": "private, max-age=31536000, immutable",
                "ETag": f'"sha256-{digest}"',
                "X-VisionCortex-Release": release,
                "X-VisionCortex-Integrity": "release-manifest-size-matched",
            }
        )
    if poster:
        return _video_poster_response(candidate, root)
    return FileResponse(candidate, headers=headers)


router.get("/api/archive-file")(archive_file)

router.get("/api/staging-file")(staging_file)

router.post("/api/archives/{archive_name}/open")(open_archive_folder)

router.post("/api/staging-runs/{run_id}/open")(open_staging_folder)

router.get("/api/archives/{name}/speech")(archive_speech)

router.get("/api/staging-runs/{run_id}/speech")(staging_speech)

router.get("/api/speech-search")(search_speech_library)

router.post("/api/{collection}/{name}/speech-evaluation")(evaluate_speech)

router.get("/api/staging-runs/{run_id}/speech-video")(staging_speech_video)

router.get("/api/archives/{name}/speech-video")(archive_speech_video)


@router.get("/api/archives")
def list_archives(
    q: str | None = None,
    cursor: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
) -> dict[str, Any]:
    return archive_read_model.list_archives(q=q, cursor=cursor, limit=limit)
