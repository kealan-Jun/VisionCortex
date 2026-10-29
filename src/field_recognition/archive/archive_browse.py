"""Human-readable NAS folders derived from acknowledged immutable receipts."""

import hashlib
import html
import json
import re
from datetime import datetime
from pathlib import PurePosixPath
from zoneinfo import ZoneInfo


CATEGORIES = {
    "Bindings": ("绑定记录", "按绑定开始日期、相机查谁在何时绑定仪器或场景。"),
    "InstrumentReadings": ("仪器读数", "按仪器编号、拍摄日期查读数；Unassigned 为归属待确认。"),
    "VoicePhotos": ("语音拍照读数", "按相机、拍摄日期查原语音照片文件及识别结果。"),
    "WorkbenchReadings": ("实验台总览", "按实验台、日期汇总仪器读数和未确定仪器归属的读数。"),
    "Photos": ("照片留存", "按照片日期、相机查看原图、面板图及 SHA-256。"),
    "PhotoDrafts": (
        "拍照测量草稿",
        "一次连拍对应一份草稿，保留原文、字段证据和修订历史；不代表已提交。",
    ),
    "ExperimentRecords": (
        "已确认实验记录",
        "仅包含生产模式人工确认的测量，每次测量只提交一条记录。",
    ),
    "Handoffs": (
        "设备交接",
        "查看谁申请接收、原使用人何时交出、接收人何时确认，原绑定不会被覆盖。",
    ),
}


def token(value):
    value = str(value or "Unassigned")
    return (
        value
        if re.fullmatch(r"[A-Za-z0-9_-]{1,100}", value)
        else hashlib.sha256(value.encode()).hexdigest()[:24]
    )


def local_time(value):
    return datetime.fromisoformat(value).astimezone(ZoneInfo("Asia/Shanghai"))


def artifact_path(digest, suffix=".png"):
    if re.fullmatch(r"[a-f0-9]{64}", digest or "") and suffix in {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".bmp",
        ".tiff",
    }:
        return f"Objects/{digest[:2]}/{digest}{suffix}"
    return None


def page(title, body):
    return (
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>"
        + html.escape(title)
        + "</title><style>body{max-width:1100px;margin:32px auto;padding:0 20px;font:15px/1.7 system-ui;color:#183c43;background:#f4f8f8}"
        "a{color:#246a70}article{background:white;border:1px solid #dce7e8;border-radius:12px;padding:24px;margin:16px 0}"
        "dt{color:#61787e}dd{margin:0 0 12px;overflow-wrap:anywhere}img{max-width:100%;max-height:650px;object-fit:contain}li{margin:10px 0}small{color:#61787e}"
        "table{width:100%;border-collapse:collapse}td,th{text-align:left;border-bottom:1px solid #dce7e8;padding:10px;overflow-wrap:anywhere}input{font:inherit;padding:8px;max-width:100%}[hidden]{display:none!important}"
        ".business-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}.business-grid article{margin:0;padding:16px 20px}.business-grid h2{font-size:18px;margin:0 0 8px}.business-grid p{margin:5px 0;overflow-wrap:anywhere}@media(max-width:650px){.business-grid{grid-template-columns:1fr}}</style>"
        "<h1>" + html.escape(title) + "</h1>" + body + "</html>"
    ).encode()


def build_views(rows):
    """No NAS reads or source mutations. Relative links work with file:// and HTTP."""
    latest, versions = {}, {}
    for row in rows:
        key = (row["entity"], row["entity_id"])
        versions.setdefault(key, []).append(
            {"sequence": row["seq"], "receipt": row["receipt_path"]}
        )
        if key not in latest or row["seq"] > latest[key]["seq"]:
            latest[key] = row
    captures = {
        ident: json.loads(row["document"])
        for (entity, ident), row in latest.items()
        if entity == "scans"
    }
    views, catalog, business = {}, {key: [] for key in CATEGORIES}, []
    esc = lambda value: html.escape(str(value if value is not None else "未记录"))

    def record(category, parts, title, data):
        # One entity has one canonical view. Classification never duplicates its payload.
        directory = PurePosixPath(
            "Records",
            "".join(w.title() for w in data["entity"].split("_")),
            token(data["entity_id"]),
        )
        business.append((category, data, str(directory / "Readme.html")))
        catalog[category].append(
            {
                "title": title,
                "local_time": data["local_time"],
                "record": str(directory / "Record.json"),
                "page": str(directory / "Readme.html"),
                "entity_id": data["entity_id"],
                "instrument_id": data.get("folder_instrument_id"),
            }
        )

    for (entity, ident), row in latest.items():
        if entity in {"photo_measurements", "experiment_records"}:
            doc = json.loads(row["document"])
            local = local_time(
                doc.get("confirmed_at") or doc.get("created_at") or row["recorded_at"]
            )
            category = "ExperimentRecords" if entity == "experiment_records" else "PhotoDrafts"
            photos = {}
            for source in doc.get("sources", []):
                for kind, digest, suffix in [
                    ("image", source.get("image_sha256"), ".png"),
                    (
                        "original",
                        source.get("source_sha256"),
                        PurePosixPath(source.get("original_blob") or "").suffix,
                    ),
                ]:
                    path = artifact_path(digest, suffix)
                    if path:
                        photos.setdefault(
                            "image"
                            if not photos and kind == "image"
                            else source["capture_id"] + "_" + kind,
                            {"path": path, "sha256": digest},
                        )
                for region in source.get("panel_regions", []):
                    path = artifact_path(region.get("image_sha256"))
                    if path:
                        photos[source["capture_id"] + "_" + region["panel_id"]] = {
                            "path": path,
                            "sha256": region["image_sha256"],
                        }
            title = (
                local.strftime("%Y-%m-%d %H:%M:%S")
                + " · "
                + ("已确认测量" if entity == "experiment_records" else "测量草稿")
            )
            record(
                category,
                [local.date().isoformat(), doc["camera_id"], ident],
                title,
                {
                    "entity": entity,
                    "entity_id": ident,
                    "camera_id": doc["camera_id"],
                    "local_time": local.strftime("%Y-%m-%d %H:%M:%S"),
                    "status": doc["status"],
                    "record_scope": doc["record_scope"],
                    "measurement_draft": doc,
                    "photos": photos,
                    "operator": (doc.get("confirmation") or {}).get("actor"),
                    "receipt_versions": versions[(entity, ident)],
                },
            )
            continue
        if entity not in {"bindings", "scene_visits", "jobs", "scans", "binding_handoffs"}:
            continue
        doc = json.loads(row["document"])
        capture = (
            doc
            if entity == "scans"
            else captures.get(doc.get("capture_id") or doc.get("scan_id"), {})
        )
        external = doc.get("external_photo") or capture.get("external_photo") or {}
        moment = (
            (doc.get("started_at") if entity in {"bindings", "scene_visits"} else None)
            or external.get("captured_at")
            or (doc.get("video_observation") or {}).get("observed_at")
            or capture.get("received_at")
            or doc.get("submitted_at")
            or row["recorded_at"]
        )
        local = local_time(moment)
        day = local.date().isoformat()
        camera = doc.get("camera_id")
        photos = {}
        for name, digest, suffix in [
            ("image", capture.get("image_sha256"), ".png"),
            (
                "original",
                capture.get("source_sha256"),
                PurePosixPath(capture.get("original_blob") or "").suffix,
            ),
            ("panel", doc.get("crop_image_sha256"), ".png"),
        ]:
            path = artifact_path(digest, suffix)
            if path:
                photos[name] = {"path": path, "sha256": digest}
        from field_recognition.measurements.reading_results import build_readings

        for region in doc.get("panel_regions", []):
            path = artifact_path(region.get("image_sha256"), ".png")
            if path:
                photos[
                    "面板 "
                    + (region.get("instrument") or {}).get("name", region["panel_id"])
                    + " "
                    + region["panel_id"]
                ] = {"path": path, "sha256": region["image_sha256"]}
        readings = doc.get("readings")
        if readings is None:
            readings = build_readings(doc) if entity == "jobs" else []
        asset = doc.get("instrument") or {}
        workbench = doc.get("workbench") or {
            "name": asset.get("scene") or (doc.get("scene") or {}).get("name")
        }
        target = (
            asset.get("name")
            or (
                "、".join(c["instrument"]["name"] for c in doc.get("instrument_candidates", []))
                if doc.get("association_status") == "localized_panels"
                else ""
            )
            or (doc.get("scene") or {}).get("name")
            or "归属待确认"
        )
        common = {
            "entity": entity,
            "entity_id": ident,
            "camera_id": camera,
            "operator": doc.get("operator"),
            "event_at": moment,
            "local_time": local.strftime("%Y-%m-%d %H:%M:%S"),
            "target": target,
            "started_at": doc.get("started_at"),
            "ended_at": doc.get("ended_at"),
            "finished_at": doc.get("finished_at"),
            "status": doc.get("status") or ("已结束" if doc.get("ended_at") else "已记录"),
            "capture_id": capture.get("capture_id"),
            "job_id": doc.get("job_id"),
            "binding_id": doc.get("binding_id"),
            "binding_ids": doc.get("binding_ids", []),
            "instrument": doc.get("instrument"),
            "instrument_candidates": doc.get("instrument_candidates", []),
            "workbench": workbench,
            "readings": readings,
            "panel_regions": doc.get("panel_regions", []),
            "panel_detection": doc.get("panel_detection"),
            "source_ref": external.get("source_ref"),
            "external_capture_id": external.get("capture_id"),
            "measurement_records": doc.get("measurement_records", []),
            "record_scope": doc.get("record_scope", "legacy_test_only")
            if entity == "jobs"
            else "evidence",
            "captured_at": external.get("captured_at"),
            "source_written_at": external.get("source_written_at"),
            "input_mode": "voice_photo"
            if external
            else "video"
            if doc.get("request_trigger") == "video_stream"
            else "other",
            "timing": doc.get("timing"),
            "photos": photos,
            "receipt_versions": sorted(versions[(entity, ident)], key=lambda v: v["sequence"]),
        }
        title = common["local_time"] + " · " + target
        if entity == "binding_handoffs":
            record("Handoffs", [day, camera, ident], title, common | {"handoff": doc})
        elif entity in {"bindings", "scene_visits"}:
            record(
                "Bindings",
                [day, camera, ident],
                title
                + " · "
                + str(doc.get("operator") or "未登记人员").replace("Demo验证", "样张验证"),
                common,
            )
        elif entity == "jobs":
            if doc.get("record_scope") == "draft":
                # Production OCR receipts stay in Receipts; the grouped draft/confirmed
                # record, not every image, is the business browsing unit.
                continue
            groups = {}
            measurements = {r["instrument_id"]: r for r in doc.get("measurement_records", [])}
            for reading in readings:
                groups.setdefault(
                    (reading.get("instrument") or {}).get("id") or "Unassigned", []
                ).append(reading)
            for instrument_id in measurements:
                groups.setdefault(instrument_id, [])
            if not groups:
                candidates = doc.get("instrument_candidates") or []
                sole = asset.get("id") or (
                    candidates[0]["instrument"]["id"] if len(candidates) == 1 else "Unassigned"
                )
                groups[sole] = []
            for instrument_id, values in groups.items():
                name = next(
                    (
                        (v.get("instrument") or {}).get("name")
                        for v in values
                        if (v.get("instrument") or {}).get("name")
                    ),
                    target,
                )
                measurement = measurements.get(instrument_id, {})
                record(
                    "InstrumentReadings",
                    [instrument_id, day, ident],
                    common["local_time"] + " · " + name,
                    common
                    | {
                        "target": name,
                        "readings": values,
                        "folder_instrument_id": instrument_id,
                        "measurement": measurement.get("record"),
                        "measurement_evidence": measurement.get("evidence"),
                    },
                )
            if external:
                filename = PurePosixPath(
                    external.get("source_ref") or external.get("capture_id") or ident
                ).name
                record(
                    "VoicePhotos",
                    [camera, day, local.strftime("%H-%M-%S") + "_" + ident],
                    title + " · " + filename,
                    common,
                )
            record(
                "WorkbenchReadings",
                [workbench.get("id") or workbench.get("name"), day, ident],
                title,
                common,
            )
        elif photos:
            record(
                "Photos",
                [day, camera, local.strftime("%H-%M-%S") + "_" + ident],
                common["local_time"] + " · 照片",
                common,
            )

    from field_recognition.archive.archive_readable import build_readable

    return build_readable(business, page)
