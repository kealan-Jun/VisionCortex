"""Explicit runtime actions for readouts."""

from __future__ import annotations

import json
import re
import uuid
from fastapi import HTTPException
from fastapi.responses import FileResponse
from field_recognition.measurements.activity import readout_page
from field_recognition.measurements.readout_timing import update_timing
from field_recognition.measurements.photo_measurements import (
    attach as attach_measurement,
    check_retry,
)
from field_recognition.api.models import OcrRequest
from field_recognition.api.security import require_camera


def submit_ocr(_runtime, body: OcrRequest):
    with _runtime["ocr_submit_lock"]:
        return _runtime["enqueue_ocr"](body)


def enqueue_ocr(_runtime, body, *, trigger="explicit_request", precomputed_local=None):
    with _runtime["db"]() as conn:
        from field_recognition.bindings.binding_policy import expire

        expire(conn, _runtime["now"]())
        binding = conn.execute(
            "SELECT * FROM bindings WHERE id=?", (str(body.binding_id),)
        ).fetchone()
        capture = conn.execute(
            "SELECT document FROM scans WHERE id=?", (str(body.capture_id),)
        ).fetchone()
        if not capture:
            raise HTTPException(404, "图片不存在")
        capture = json.loads(capture["document"])
        require_camera(capture["camera_id"])
        if binding and binding["camera"] != capture["camera_id"]:
            raise HTTPException(409, "图片来源与绑定相机不一致")
        linked = json.loads(binding["document"]) if binding else None
        from field_recognition.measurements.readout_context import binding_contains_photo

        historical_binding = bool(linked and binding_contains_photo(linked, capture))
        if body.binding_id is not None and (
            not binding
            or ((binding["ended"] or capture.get("external_photo")) and not historical_binding)
        ):
            raise HTTPException(409, "照片不属于该仪器的有效使用时段")
        if linked and not historical_binding:
            current_asset = _runtime["get_instrument"](linked["instrument"]["id"])
            if not current_asset or current_asset["scene"] != linked["instrument"]["scene"]:
                raise HTTPException(409, "仪器所属场景已更改，请重新绑定")
            visit_id = linked.get("scene_visit_id")
            if (
                visit_id
                and not conn.execute(
                    "SELECT 1 FROM scene_visits WHERE id=? AND ended IS NULL", (visit_id,)
                ).fetchone()
            ):
                raise HTTPException(409, "场景已变化，请重新绑定仪器")
            if (
                capture["received_at"] < linked["started_at"]
                and capture["scan_id"] != linked["scan_id"]
            ):
                raise HTTPException(409, "图片早于本次绑定，请重新拍照")
        seen = {item["id"] for item in capture["matches"]}
        if (
            linked
            and seen
            and (linked["instrument"]["id"] not in seen or (len(seen) > 1 and body.crop is None))
        ):
            raise HTTPException(409, "图片仪器二维码与当前绑定冲突，请重新选择仪器")
        from field_recognition.measurements.readout_context import at_capture

        context = at_capture(
            _runtime,
            conn,
            capture,
            linked=linked,
            automatic=body.auto_associate
            or bool(capture["matches"])
            or bool(capture.get("external_photo"))
            or trigger in {"voice_photo_directory", "video_stream"},
        )
        linked = context.pop("resolved_binding")
        binding_id = linked["binding_id"] if linked else None
        crop = body.crop if body.crop is not None else [0, 0, capture["width"], capture["height"]]
        if (
            len(crop) != 4
            or min(crop[:2]) < 0
            or min(crop[2:]) < 16
            or crop[0] + crop[2] > capture["width"]
            or crop[1] + crop[3] > capture["height"]
        ):
            raise HTTPException(422, "面板选框超出图片，或区域太小")
        repeatable = _runtime["RECORD_MODE"] == "production" or body.measurement is not None
        lookup = (
            "SELECT status,document FROM jobs WHERE json_extract(document,'$.capture_id')=?"
            + ("" if repeatable else " AND status IN ('queued','running')")
            + " ORDER BY rowid DESC"
        )
        for pending in conn.execute(lookup, (str(body.capture_id),)):
            existing = json.loads(pending["document"])
            if (
                existing["binding_id"] == binding_id
                and existing["capture_id"] == str(body.capture_id)
                and existing["crop"] == crop
            ):
                check_retry(existing, body.measurement, _runtime["RECORD_MODE"])
                return existing | {"status": pending["status"]}
        if conn.execute(
            "SELECT count(*) FROM jobs WHERE status IN ('queued','running')"
        ).fetchone()[0] >= (1000 if _runtime["RUNTIME_ENABLED"] else 4):
            raise HTTPException(429, "OCR 正在处理，请稍候再提交")
        document = {
            "job_id": str(uuid.uuid4()),
            "binding_id": binding_id,
            "capture_id": str(body.capture_id),
            "crop": crop,
            "submitted_at": _runtime["now"](),
            "status": "queued",
            "image_url": capture["image_url"],
            "image_sha256": capture["image_sha256"],
            "instrument": linked["instrument"] if linked else None,
            "operator": linked["operator"] if linked else capture.get("operator"),
            "camera_id": capture["camera_id"],
            "operator_basis": "binding_snapshot"
            if linked
            else capture.get("operator_basis", "not_recorded"),
            "operator_registration": capture.get("operator_registration"),
            "scene_visit_id": linked.get("scene_visit_id") if linked else None,
            "scene": linked["scene"] if linked else None,
            "scene_basis": linked.get("scene_basis", "decoded_scene_qr") if linked else "unbound",
            "scene_qr_verified": linked.get("scene_qr_verified", bool(linked.get("scene_visit_id")))
            if linked
            else False,
            "instrument_association": (
                "same_image_qr" if seen else "operator_selected_current_binding"
            )
            if linked
            else "unbound_photo",
            "source": capture["source"],
            "frame_metadata": capture.get("frame_metadata"),
            "timing": dict(capture.get("photo_observation") or {}),
            "external_photo": capture.get("external_photo"),
            "request_trigger": trigger,
        }
        document["resume_pending"] = trigger != "video_stream"
        document["video_observation"] = capture.get("video_observation")
        document.update(context)
        if context["attribution_status"] == "needs_review":
            document.update(operator=None, wearer_id=None, operator_basis="pending_attribution")
            if (
                context["attribution_reason"] == "historical_binding_missing"
                and capture.get("operator_basis") == "camera_registration"
            ):
                document.update(
                    operator=capture.get("operator"),
                    wearer_id=capture.get("wearer_id"),
                    operator_basis="camera_registration",
                )
        else:
            people = {
                (b.get("operator"), b.get("wearer_id")) for b in context["all_binding_snapshots"]
            }
            if len(people) == 1:
                document["operator"], document["wearer_id"] = next(iter(people))
        document["field_rules"] = {
            b["instrument"]["id"]: (_runtime["get_instrument"](b["instrument"]["id"]) or {}).get(
                "measurement_ranges", {}
            )
            for b in document.get("all_binding_snapshots", document.get("binding_snapshots", []))
        }
        if not linked and context["instrument_candidates"]:
            document["instrument_association"] = (
                "multiple_candidates"
                if len(context["instrument_candidates"]) > 1
                else "same_image_qr_unbound"
            )
            if len(context["instrument_candidates"]) == 1:
                # A decoded registered QR establishes identity without fabricating
                # a new session binding for this photograph.
                document["instrument"] = context["instrument_candidates"][0]["instrument"]
                document["scene"] = context["workbench"]
                document["scene_basis"] = (context["workbench"] or {}).get(
                    "basis", "instrument_registration"
                )
        document["instrument_identity_basis"] = (
            "decoded_photo_qr"
            if seen and document["instrument"]
            else "session_binding"
            if linked
            else "multiple_candidates"
            if context["instrument_candidates"]
            else "not_observed"
        )
        if precomputed_local is not None:
            document["precomputed_local"] = precomputed_local
        update_timing(document)
        attach_measurement(_runtime, conn, document, body.measurement)
        if _runtime["RUNTIME_ENABLED"]:
            from field_recognition.runtime.task_queue import QueueFull

            try:
                document = _runtime["task_queue"].enqueue(document, conn)
            except QueueFull as exc:
                raise HTTPException(429, str(exc)) from None
        else:
            conn.execute(
                "INSERT INTO jobs(id,status,document) VALUES(?,?,?)",
                (document["job_id"], "queued", json.dumps(document)),
            )
    if not _runtime["RUNTIME_ENABLED"]:
        _runtime["readout_pool"].submit(_runtime["run_ocr"], dict(document))
    return document


def get_job(_runtime, job_id: uuid.UUID):
    with _runtime["db"]() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id=?", (str(job_id),)).fetchone()
    if not row:
        raise HTTPException(404, "任务不存在")
    document = json.loads(row["document"]) | {"status": row["status"]}
    from field_recognition.measurements.measurement_records import display_fields

    document["archive"] = _runtime["archive_store"].job_receipt(document)
    update_timing(document, preserve_recorded_durations=True)
    return document | {"display_fields": display_fields(document)}


def export(_runtime):
    result = _runtime["state"]()
    result["exported_at"] = _runtime["now"]()
    return result


def get_measurements(_runtime, job_id: uuid.UUID):
    document = _runtime["get_job"](job_id)
    from field_recognition.measurements.measurement_records import build_records, RULE_VERSION

    records = build_records(document)
    return {
        "job_id": str(job_id),
        "records": records,
        "rule_version": RULE_VERSION,
        "status": document["status"],
        "record_scope": document.get("record_scope", "legacy_test_only"),
        "measurement_id": document.get("measurement_id"),
        "format_available": "measurement_records" in document or bool(records),
    }


def readouts(_runtime, limit: int = 20, before: int | None = None, related_only: bool = False):
    if not 1 <= limit <= 50 or (before is not None and before < 1):
        raise HTTPException(422, "页大小需为 1–50，游标需为正整数")
    camera = (
        _runtime["receiver_camera"].target
        if _runtime["receiver_camera"]
        else _runtime.config.camera_id
    )
    with _runtime["db"]() as conn:
        return readout_page(conn, camera, limit=limit, before=before, related_only=related_only)


def workbenches(_runtime, limit: int = 200, related_only: bool = False):
    if not 1 <= limit <= 500:
        raise HTTPException(422, "记录范围需为 1–500")
    from field_recognition.measurements.workbench_records import summarize

    camera = (
        _runtime["receiver_camera"].target
        if _runtime["receiver_camera"]
        else _runtime.config.camera_id
    )
    with _runtime["db"]() as conn:
        return summarize(conn, camera, limit=limit, related_only=related_only)


def archive_file(_runtime, relative: str):
    from field_recognition.archive.archive_integrity import relative_file
    from field_recognition.archive.archive_paths import checked, resolve_name

    if not _runtime["archive_store"].enabled():
        raise HTTPException(404, "未启用 NAS 留存")
    if (
        relative not in {"Readme.html", "Audit.html", "Index.json", "00_归档导航.html"}
        and not re.fullmatch(
            r"(?:Receipts|Objects|Integrity)/[A-Za-z0-9_./-]+\.(?:json|png|jpg|jpeg|webp|bmp|tiff)",
            relative,
        )
        and not re.fullmatch(r"(?:Browse|Records)/[A-Za-z0-9_./-]+\.(?:html|json)", relative)
        and not re.fullmatch(r"01_业务数据/[\w./-]+\.(?:html|json)", relative)
        and not re.fullmatch(
            r"(?:\d{4}-\d{2}-\d{2}_[A-Za-z0-9_-]+|\.System)/[A-Za-z0-9_./-]+\.(?:html|json|png|jpg|jpeg|webp|bmp|tiff)",
            relative,
        )
    ):
        raise HTTPException(404, "归档文件不存在")
    try:
        root = _runtime["archive_store"]._root(create=False)
        checked(root, relative)
        canonical = resolve_name(root, relative)
        path = relative_file(root, canonical)
        if not path.is_file():
            raise FileNotFoundError()
    except (OSError, ValueError):
        raise HTTPException(404, "归档文件暂不可用")
    if canonical != relative:
        from fastapi.responses import RedirectResponse
        from urllib.parse import quote

        return RedirectResponse(
            "/api/archive/files/" + quote(canonical, safe="/"),
            headers={"Cache-Control": "no-store"},
        )
    return FileResponse(path, headers={"Cache-Control": "no-store"})
