"""Read-only activity timeline from stored receipts, including unbound QR hits."""

import json
from datetime import datetime
from field_recognition.archive.archive_store import receipt_status
from field_recognition.measurements.readout_rules import READOUT
from field_recognition.measurements.history_records import (
    panel_readings,
    job_rows,
    current_version_sql,
)
from field_recognition.measurements.measurement_records import display_fields
from field_recognition.measurements.readout_timing import update_timing


def readout_event(conn, row, *, related_only=False):
    doc = json.loads(row["document"])
    doc["archive"] = receipt_status(
        conn, "jobs", doc["job_id"], (doc.get("timing") or {}).get("source_written_at")
    )
    update_timing(doc, preserve_recorded_durations=True)
    status = row["status"]
    if related_only:
        relevant = panel_readings(doc, status)
        assets = {r["instrument"]["id"]: r["instrument"] for r in relevant}
        doc = dict(
            doc,
            readings=relevant,
            lines=relevant,
            instrument=next(iter(assets.values())) if len(assets) == 1 else None,
            instrument_candidates=[{"instrument": a} for a in assets.values()],
            binding_ids=list(
                dict.fromkeys(r["binding_id"] for r in relevant if r.get("binding_id"))
            ),
        )
    lines = [
        line
        for line in doc.get("lines", [])
        if line.get("text")
        and (related_only or doc.get("device") == "cloud" or READOUT.fullmatch(line["text"]))
    ]
    label = {
        "queued": "等待识别",
        "running": "正在识别",
        "failed": "识别失败",
        "interrupted": "识别中断",
        "cancelled": "识别取消",
    }.get(status, status)
    if status == "completed":
        label = "自动识别完成" if lines else "未读出完整数字"
    if doc.get("reading_mode") == "parallel_photo":
        label = {
            "running": "本地初读 · 云端复核中"
            if doc.get("result_available")
            else "本地与云端正在识别",
            "completed": "模型复核完成 · 未经人工核实",
            "failed": "本地初读 · 云端复核未完成",
            "conflict": "复核存在冲突 · 待核对",
            "unreadable": "云端未读清 · 待核对",
        }.get((doc.get("parallel_review") or {}).get("status"), label)
        if (doc.get("parallel_review") or {}).get("crosscheck_pending"):
            label = "初步读数 · 本地交叉检查中"
    notes = []
    if doc.get("panel_selection") == "user_selected_crop_or_full_photo" and not doc.get(
        "panel_detection"
    ):
        notes.append("历史整图或手动选框识别，面板归属未核验")
    for reading in doc.get("readings", []):
        issue = {
            "decimal_uncertain": "小数点可能缺失，不能直接作为数值",
            "possible_display_self_test": "疑似屏幕自检，不能作为测量值",
        }.get(reading.get("quality_issue"))
        if issue and issue not in notes:
            notes.append(issue)
    if doc.get("recognition_skipped"):
        notes.append("未检测到已绑定仪器面板，已跳过识别")
    fields = display_fields(doc)
    detail = "、".join(
        (f["instrument_name"] or "")
        + " "
        + f["name"]
        + " "
        + (
            str(f["value"]) + (" " + f["unit"] if f["unit"] else "")
            if f["value"] is not None
            else f["display_state"] or "未取得有效数值"
        )
        for f in fields
    )
    return {
        "event_id": "readout:" + doc["job_id"],
        "job_id": doc["job_id"],
        "kind": "readout",
        "raw_status": status,
        "display_fields": fields,
        "recognition_revision": doc.get("recognition_revision", 1),
        "reading_mode": doc.get("reading_mode"),
        "result_available": doc.get("result_available"),
        "result_stage": doc.get("result_stage"),
        "result_revision": doc.get("result_revision"),
        "parallel_review": doc.get("parallel_review"),
        "occurred_at": doc.get("finished_at") or doc["submitted_at"],
        "submitted_at": doc["submitted_at"],
        "captured_at": (doc.get("external_photo") or {}).get("captured_at"),
        "camera_id": doc["camera_id"],
        "operator": doc.get("operator"),
        "target": (doc.get("instrument") or {}).get("name")
        or (
            "、".join(c["instrument"]["name"] for c in doc.get("instrument_candidates", []))
            + (
                " · 面板分别定位"
                if doc.get("association_status") == "localized_panels"
                else " · 归属待确认"
            )
            if doc.get("instrument_candidates")
            else "未绑定仪器 · 照片读数"
        ),
        "status": label,
        "detail": detail if fields else "、".join(line["text"] for line in lines),
        "quality_notes": notes,
        "measurement_url": "/api/jobs/" + doc["job_id"] + "/measurements"
        if doc.get("measurement_records")
        else None,
        "image_url": doc.get("image_url") or doc.get("crop_image_url"),
        "result_url": "/api/jobs/" + doc["job_id"],
        "timing": doc.get("timing") or {},
        "readings": doc.get("readings", []),
        "binding_ids": doc.get("binding_ids", []),
        "workbench": doc.get("workbench"),
        "workbenches": doc.get("workbenches", []),
        "input_mode": "video" if doc.get("request_trigger") == "video_stream" else "photo",
        "fallback": doc.get("fallback")
        and {key: doc["fallback"].get(key) for key in ("status", "reason", "error", "http_status")},
        "archive": doc["archive"],
    }


def readout_page(conn, camera_id=None, *, limit=20, before=None, related_only=False):
    """Stable keyset pages in submission order; no photo bytes or raw model output."""
    conditions, params = [current_version_sql()], []
    if camera_id:
        conditions.append("json_extract(document,'$.camera_id')=?")
        params.append(camera_id)
    if before is not None:
        conditions.append("rowid<?")
        params.append(before)
    where = " WHERE " + " AND ".join(conditions) if conditions else ""
    rows = (
        job_rows(conn, camera_id, limit=limit + 1, before=before)
        if related_only
        else conn.execute(
            "SELECT rowid AS cursor,* FROM jobs" + where + " ORDER BY rowid DESC LIMIT ?",
            params + [limit + 1],
        ).fetchall()
    )
    return {
        "items": [readout_event(conn, row, related_only=related_only) for row in rows[:limit]],
        "next_cursor": rows[limit - 1]["cursor"] if len(rows) > limit else None,
        "camera_id": camera_id,
        "order": "submitted_desc",
        "related_only": related_only,
    }


def recent_activity(conn, camera_id=None, limit=50):
    events = []

    def rows(table, timestamp):
        where, params = (
            ("", [])
            if camera_id is None
            else ("WHERE json_extract(document,'$.camera_id')=?", [camera_id])
        )
        if table == "scans":
            where += (
                (" AND " if where else "WHERE ")
                + "(coalesce(json_array_length(document,'$.matches'),0)>0 OR coalesce(json_array_length(document,'$.scene_matches'),0)>0)"
            )
        occurred_at = (
            "coalesce(json_extract(document,'$.external_photo.captured_at'),json_extract(document,'$.received_at'))"
            if table == "scans"
            else f"json_extract(document,'$.{timestamp}')"
        )
        # Table/field names come only from the fixed calls below.
        return conn.execute(
            f"SELECT * FROM {table} {where} ORDER BY julianday({occurred_at}) DESC,rowid DESC LIMIT ?",
            params + [limit],
        ).fetchall()

    def add(doc, kind, timestamp, target, status, evidence=None, detail="", operator=None):
        events.append(
            {
                "event_id": f"{kind}:{doc.get('scan_id') if kind in {'scan', 'photo'} else doc.get('job_id') or doc.get('binding_id') or doc.get('visit_id')}",
                "kind": kind,
                "occurred_at": timestamp,
                "camera_id": doc["camera_id"],
                "operator": doc.get("operator") if operator is None else operator,
                "target": target,
                "status": status,
                "detail": detail,
                "image_url": evidence,
            }
        )
        entity = (
            "scans"
            if kind in {"scan", "photo"}
            else "bindings"
            if kind.startswith("binding_")
            else "scene_visits"
        )
        ident = (
            doc.get("scan_id")
            if entity == "scans"
            else doc.get("binding_id")
            if entity == "bindings"
            else doc.get("visit_id")
        )
        events[-1]["archive"] = receipt_status(conn, entity, ident)

    for row in rows("scans", "received_at"):
        doc = json.loads(row["document"])
        hits = doc.get("matches", []) + doc.get("scene_matches", [])
        linked = conn.execute(
            "SELECT document FROM bindings WHERE json_extract(document,'$.scan_id')=? LIMIT 1",
            (doc["scan_id"],),
        ).fetchone()
        binding = json.loads(linked["document"]) if linked else None
        if doc.get("matches"):
            detail = "已建立绑定" if binding else "未建立仪器绑定"
        else:
            visit = conn.execute(
                "SELECT 1 FROM scene_visits WHERE json_extract(document,'$.scan_id')=? LIMIT 1",
                (doc["scan_id"],),
            ).fetchone()
            detail = "已记录场景进入" if visit else "场景二维码已识别"
        add(
            doc,
            "scan",
            (doc.get("external_photo") or {}).get("captured_at") or doc["received_at"],
            "、".join(hit["name"] for hit in hits),
            "已识别",
            doc["image_url"],
            detail,
        )

    for row in rows("bindings", "started_at"):
        doc = json.loads(row["document"])
        add(
            doc,
            "binding_started",
            doc["started_at"],
            doc["instrument"]["name"],
            "绑定成功",
            doc["image_url"],
        )
        for correction in doc.get("lifecycle_corrections", []):
            add(
                doc,
                "binding_end_corrected",
                correction["previous_ended_at"],
                doc["instrument"]["name"],
                "原结束判定已撤销",
                doc["image_url"],
                correction["detail"],
            )
            add(
                doc,
                "binding_restored",
                correction["corrected_at"],
                doc["instrument"]["name"],
                "原绑定已恢复",
                doc["image_url"],
                correction["detail"],
            )
    for row in rows("bindings", "ended_at"):
        doc = json.loads(row["document"])
        if doc.get("ended_at"):
            add(
                doc,
                "binding_ended",
                doc["ended_at"],
                doc["instrument"]["name"],
                "绑定结束",
                doc["image_url"],
            )

    for row in rows("scene_visits", "started_at"):
        doc = json.loads(row["document"])
        add(
            doc,
            "scene_entered",
            doc["started_at"],
            doc["scene"]["name"],
            "已进入场景",
            doc["image_url"],
        )
        for correction in doc.get("lifecycle_corrections", []):
            add(
                doc,
                "scene_end_corrected",
                correction["previous_ended_at"],
                doc["scene"]["name"],
                "原结束判定已撤销",
                doc["image_url"],
                correction["detail"],
            )
            add(
                doc,
                "scene_restored",
                correction["corrected_at"],
                doc["scene"]["name"],
                "原场景关联已恢复",
                doc["image_url"],
                correction["detail"],
            )
    for row in rows("scene_visits", "ended_at"):
        doc = json.loads(row["document"])
        if doc.get("ended_at"):
            add(
                doc,
                "scene_left",
                doc["ended_at"],
                doc["scene"]["name"],
                "场景关系已结束",
                doc["image_url"],
            )

    for row in job_rows(conn, camera_id, limit=limit, recent=True):
        events.append(readout_event(conn, row, related_only=True))
    return sorted(
        events,
        key=lambda event: (datetime.fromisoformat(event["occurred_at"]), event["event_id"]),
        reverse=True,
    )[:limit]
