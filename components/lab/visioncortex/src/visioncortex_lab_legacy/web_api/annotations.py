"""HTTP adapters for annotations"""

from __future__ import annotations
import json
from typing import Any
from fastapi import HTTPException, Query
from fastapi.responses import FileResponse
from ..annotation_workspace import (
    export_reviewed_ground_truth,
    load_annotation_workspace,
    record_annotation_decision,
    resolve_annotation_image,
)

from ..application import runtime_host

from fastapi import APIRouter

router = APIRouter()


def annotation_workspace(
    priority: str | None = None,
    review_status: str | None = None,
    q: str | None = None,
    offset: int = Query(0, ge=0),
    limit: int = Query(24, ge=1, le=200),
) -> dict[str, Any]:
    settings = runtime_host._settings()
    if not bool(
        (settings.get("developer_tools") or {}).get("yolo_annotation_workspace_enabled")
    ):
        raise HTTPException(404, "内部 YOLO 标注工具未启用")
    try:
        return load_annotation_workspace(
            settings,
            priority=priority,
            review_status=review_status,
            query=q,
            offset=offset,
            limit=limit,
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise HTTPException(500, f"无法读取 YOLO 标注工作台：{exc}") from exc


def annotation_workspace_image(item_id: str) -> FileResponse:
    settings = runtime_host._settings()
    if not bool(
        (settings.get("developer_tools") or {}).get("yolo_annotation_workspace_enabled")
    ):
        raise HTTPException(404, "内部 YOLO 标注工具未启用")
    try:
        path = resolve_annotation_image(settings, item_id)
    except KeyError as exc:
        raise HTTPException(404, "标注项不存在") from exc
    except FileNotFoundError as exc:
        raise HTTPException(404, "badcase 图片不存在") from exc
    return FileResponse(path)


def save_annotation_workspace_decision(
    item_id: str, payload: dict[str, Any]
) -> dict[str, Any]:
    settings = runtime_host._settings()
    if not bool(
        (settings.get("developer_tools") or {}).get("yolo_annotation_workspace_enabled")
    ):
        raise HTTPException(404, "内部 YOLO 标注工具未启用")
    try:
        decision = record_annotation_decision(settings, item_id, payload)
    except KeyError as exc:
        raise HTTPException(404, "标注项不存在") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"status": "saved", "decision": decision}


def annotation_workspace_export() -> dict[str, Any]:
    settings = runtime_host._settings()
    if not bool(
        (settings.get("developer_tools") or {}).get("yolo_annotation_workspace_enabled")
    ):
        raise HTTPException(404, "内部 YOLO 标注工具未启用")
    try:
        return export_reviewed_ground_truth(settings)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise HTTPException(500, f"无法导出审核真值：{exc}") from exc


router.get("/api/annotation-workspace")(annotation_workspace)

router.get("/api/annotation-workspace/items/{item_id}/image")(
    annotation_workspace_image
)

router.post("/api/annotation-workspace/items/{item_id}/decision")(
    save_annotation_workspace_decision
)

router.get("/api/annotation-workspace/export")(annotation_workspace_export)
