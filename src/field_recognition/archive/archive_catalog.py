"""Portable full-history index derived from immutable business snapshots."""

import json
from datetime import datetime
import re
from zoneinfo import ZoneInfo

from field_recognition.measurements.readout_timing import elapsed_ms
from field_recognition.archive.archive_progress import advance
from field_recognition.runtime.paths import PROJECT_ROOT


def build_index(rows, instance, timestamp, integrity):
    items = []
    for row in rows:
        advance("building_navigation")
        doc = json.loads(row["document"])
        occurred = next(
            (
                doc.get(k)
                for k in (
                    "finished_at",
                    "ended_at",
                    "started_at",
                    "submitted_at",
                    "received_at",
                    "registered_at",
                )
                if doc.get(k)
            ),
            row["recorded_at"],
        )
        local = datetime.fromisoformat(occurred).astimezone(ZoneInfo("Asia/Shanghai"))
        instruments = (
            [doc["instrument"]]
            if doc.get("instrument")
            else (
                [c["instrument"] for c in doc.get("instrument_candidates", [])]
                or doc.get("matches", [])
            )
        )
        instruments = [{"id": i["id"], "name": i["name"]} for i in instruments]
        target = (
            "、".join(i["name"] for i in instruments)
            or (doc.get("scene") or {}).get("name")
            or "、".join(i["name"] for i in doc.get("scene_matches", []))
        )
        if row["entity"] == "jobs" and not doc.get("binding_id"):
            if not instruments:
                target = "未识别仪器 · 照片读数"
            elif (
                not doc.get("instrument")
                and len(instruments) > 1
                and doc.get("association_status") != "localized_panels"
            ):
                target += " · 归属待确认"
        image_hash = doc.get("image_sha256")
        image = (
            f"Objects/{image_hash[:2]}/{image_hash}.png"
            if re.fullmatch("[a-f0-9]{64}", image_hash or "")
            else None
        )
        published_at = row["published_at"] if "published_at" in row.keys() else None
        readable_at = row["readable_at"] if "readable_at" in row.keys() else None
        items.append(
            {
                "sequence": row["seq"],
                "entity": row["entity"],
                "entity_id": row["entity_id"],
                "camera_id": doc.get("camera_id")
                or (row["entity_id"] if row["entity"] == "automation_settings" else None),
                "occurred_at": occurred,
                "local_time": local.strftime("%Y-%m-%d %H:%M:%S"),
                "day": local.date().isoformat(),
                "captured_at": (doc.get("external_photo") or {}).get("captured_at"),
                "operator": doc.get("operator"),
                "instruments": instruments,
                "target": target,
                "status": doc.get("status")
                or (
                    "已结束"
                    if doc.get("ended_at")
                    else "使用中"
                    if row["entity"] in {"bindings", "scene_visits"}
                    else "已记录"
                ),
                "readings": "、".join(
                    str(i["text"]) for i in doc.get("lines", []) if i.get("text")
                ),
                "reading_regions": doc.get("readings", []),
                "binding_ids": doc.get("binding_ids", []),
                "workbench": doc.get("workbench")
                or (
                    {"name": doc["instrument"]["scene"], "basis": "stored_instrument_snapshot"}
                    if (doc.get("instrument") or {}).get("scene")
                    else doc.get("scene")
                ),
                "workbenches": doc.get("workbenches", []),
                "input_mode": "video" if doc.get("request_trigger") == "video_stream" else "photo",
                "record_scope": doc.get("record_scope", "legacy_test_only")
                if row["entity"] == "jobs"
                else doc.get("record_scope"),
                "measurement_id": doc.get("measurement_id"),
                "receipt": row["receipt_path"],
                "image": image,
                "archived_at": row["archived_at"],
                "archive_published_at": published_at,
                "archive_readable_at": readable_at,
                "archive_readability_basis": row["readability_basis"]
                if "readability_basis" in row.keys()
                else None,
                "archive_queue_ms": elapsed_ms(
                    row["recorded_at"], published_at or row["archived_at"]
                ),
                "write_to_archive_ms": elapsed_ms(
                    (doc.get("timing") or {}).get("source_written_at"),
                    published_at or row["archived_at"],
                ),
                "archive_readability_ms": elapsed_ms(published_at, readable_at),
            }
        )
        from field_recognition.archive.archive_store import publication_timing

        items[-1].update(publication_timing(row))
    return {
        "schema": "field-recognition-index/2",
        "timezone": "Asia/Shanghai",
        "source_instance": instance,
        "updated_at": timestamp,
        "archived_receipts": len(items),
        "listed_receipts": len(items),
        "unique_records": len({(i["entity"], i["entity_id"]) for i in items}),
        "scope": "all_acknowledged_versions",
        "integrity": integrity,
        "items": items,
    }


def render_index(index):
    static = PROJECT_ROOT / "static"
    # Inline data permits opening the NAS HTML directly without a running server or fetch/file CORS.
    chunks = []
    encoder = json.JSONEncoder(ensure_ascii=False, separators=(",", ":"))
    for number, chunk in enumerate(encoder.iterencode(index), 1):
        chunks.append(chunk)
        if number % 1024 == 0:
            advance("rendering_navigation")
    advance("rendering_navigation")
    data = "".join(chunks).replace("&", "\\u0026").replace("<", "\\u003c")
    return (
        (static / "archive.html")
        .read_text()
        .replace("/*ARCHIVE_SCRIPT*/", (static / "archive.js").read_text())
        .replace("/*VIEW_UTILS*/", (static / "view-utils.js").read_text())
        .replace("/*ARCHIVE_DATA*/", data)
    ).encode()
