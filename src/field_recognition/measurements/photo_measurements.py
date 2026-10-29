"""Photo/burst drafts. Raw OCR receipts never become production records implicitly."""

import copy
from functools import lru_cache
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import re
import uuid

from fastapi import HTTPException
from fastapi.responses import FileResponse
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, FiniteFloat

from field_recognition.measurements.measurement_records import (
    number,
    unit_evidence,
    field_issues,
    measurement_value,
    binding_for_region,
)
from field_recognition.measurements.readout_rules import UNITS

from field_recognition.runtime.paths import source_path

RULE_VERSION = "photo-measurement/5"
TERMINAL = {"completed", "failed", "cancelled", "interrupted"}


class MeasurementContext(BaseModel):
    model_config = ConfigDict(extra="forbid")
    burst_id: str | None = Field(default=None, min_length=1, max_length=100)
    expected_photos: int = Field(default=1, ge=1, le=16)
    experiment_context_ref: str | None = Field(default=None, min_length=1, max_length=2000)
    instrument_ids: list[uuid.UUID] = Field(default_factory=list, max_length=16)


class FieldCorrection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    field_id: str
    instrument_id: uuid.UUID
    name: str
    value: FiniteFloat = Field(strict=True)
    unit: str = Field(min_length=1, max_length=20)


class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    revision: int = Field(ge=1)
    actor: str = Field(min_length=1, max_length=100)


class Revision(Decision):
    reason: str = Field(min_length=1, max_length=1000)
    fields: list[FieldCorrection] = Field(default_factory=list, max_length=64)
    experiment_context_ref: str | None = Field(default=None, min_length=1, max_length=2000)
    capture_times: dict[str, AwareDatetime] = Field(default_factory=dict)


class Rejection(Decision):
    reason: str = Field(min_length=1, max_length=1000)


def configured_mode():
    mode = os.environ.get("FIELD_RECORD_MODE", "test").lower().strip()
    if mode not in {"test", "production"}:
        raise ValueError("FIELD_RECORD_MODE must be test or production")
    return mode


@lru_cache(maxsize=1)
def versions():
    packages = {}
    for package in ("paddleocr", "paddlepaddle-gpu", "paddlepaddle"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            pass
    return {
        "rules": RULE_VERSION,
        "packages": packages,
        "code_sha256": {
            name: hashlib.sha256(source_path(name).read_bytes()).hexdigest()
            for name in (
                "photo_measurements.py",
                "panel_regions.py",
                "field_recognition/recognition/panel_readout.py",
                "field_recognition/recognition/parallel_photo.py",
                "field_recognition/recognition/photo_orientation.py",
                "digit_regions.py",
                "field_recognition/recognition/red_geometry.py",
                "led_digits.py",
                "red_segments.py",
                "panel_layout.py",
                "reading_results.py",
                "measurement_records.py",
                "ocr_runtime.py",
                "readout_context.py",
                "instrument_ownership.py",
                "binding_policy.py",
                "result_reprocessing.py",
                "field_recognition/measurements/readout_rules.py",
                "field_recognition/recognition/config.py",
                "field_recognition/recognition/pipeline_context.py",
                "field_recognition/measurements/measurement_actions.py",
                "field_recognition/bindings/handoff_actions.py",
                "field_recognition/bindings/binding_actions.py",
                "field_recognition/capture/capture_actions.py",
                "field_recognition/recognition/service.py",
                "field_recognition/api/models.py",
                "field_recognition/api/readouts.py",
                "field_recognition/api/measurements_routes.py",
                "field_recognition/api/bindings_routes.py",
                "field_recognition/runtime/factory.py",
                "field_recognition/runtime/bootstrap.py",
                "field_recognition/runtime/context.py",
                "field_recognition/runtime/components.py",
                "field_recognition/runtime/lifecycle.py",
                "field_recognition/runtime/runtime_rpc.py",
                "field_recognition/runtime/production_config.py",
                "field_recognition/runtime/paths.py",
            )
        },
    }


def attach(core, conn, job, context):
    """Called in the same transaction as INSERT jobs; group only on explicit burst ID."""
    mode = core["RECORD_MODE"]
    job.update(record_mode=mode, record_scope="test_only", recognition_versions=versions())
    if job.get("request_trigger") == "video_stream":
        # The production photo workflow does not promote video observations.
        job["record_scope"] = "field_observation" if mode == "production" else "test_only"
        return
    if mode == "test" and context is None:
        return
    context = context or MeasurementContext()
    if context.expected_photos > 1 and not context.burst_id:
        raise HTTPException(422, "连拍必须提供 burst_id 和预期照片数量")
    metadata = context.model_dump(mode="json")
    job["measurement_context"] = metadata
    row = (
        conn.execute(
            "SELECT document FROM photo_measurements WHERE camera=? AND burst_key=?",
            (job["camera_id"], context.burst_id),
        ).fetchone()
        if context.burst_id
        else None
    )
    if row:
        group = json.loads(row["document"])
        if group["context"] != metadata or group["record_mode"] != mode:
            raise HTTPException(409, "同一连拍编号的上下文、照片数量或模式不一致")
        if (
            group["status"] != "collecting"
            or len(group["job_ids"]) >= group["context"]["expected_photos"]
        ):
            raise HTTPException(409, "这次测量已收齐照片或已结束，请使用新的连拍编号")
    else:
        group = {
            "measurement_id": str(uuid.uuid4()),
            "camera_id": job["camera_id"],
            "record_mode": mode,
            "record_scope": "draft" if mode == "production" else "test_only",
            "context": metadata,
            "status": "collecting",
            "revision": 0,
            "job_ids": [],
            "sources": [],
            "fields": [],
            "corrections": [],
            "created_at": core["now"](),
        }
        conn.execute(
            "INSERT INTO photo_measurements(id,camera,burst_key,status,document) VALUES(?,?,?,?,?)",
            (
                group["measurement_id"],
                job["camera_id"],
                context.burst_id,
                group["status"],
                json.dumps(group),
            ),
        )
    group["job_ids"].append(job["job_id"])
    group["revision"] += 1
    conn.execute(
        "UPDATE photo_measurements SET document=? WHERE id=?",
        (json.dumps(group), group["measurement_id"]),
    )
    job.update(measurement_id=group["measurement_id"], record_scope=group["record_scope"])


def check_retry(job, context, mode):
    requested = context.model_dump(mode="json") if context else None
    stored = job.get("measurement_context")
    if requested != stored and not (
        requested is None and stored == MeasurementContext().model_dump(mode="json")
    ):
        raise HTTPException(409, "照片已属于另一测量上下文，不能通过重试重新归组")
    if job.get("record_mode", "test") != mode:
        raise HTTPException(409, "这张照片已有其他模式的识别回执，不能自动改成生产记录")


def candidates(jobs):
    fields = {}
    for job in jobs:
        regions = {
            r["panel_id"]: r
            for r in job.get("panel_regions", [])
            if binding_for_region(job, r) is not None
        }
        readings = copy.deepcopy(job.get("readings", []))
        # Preserve a visible but unreadable region so it can be explicitly corrected.
        for pid, region in regions.items():
            if not any(r.get("panel_id") == pid for r in readings):
                readings.append(
                    {
                        "panel_id": pid,
                        "instrument": region.get("instrument"),
                        "measurement_name": region.get("measurement_name"),
                        "value": None,
                        "text": (region.get("display_state") or {}).get("raw_text")
                        or (region.get("display_state") or {}).get("text"),
                        "binding_id": region.get("binding_id"),
                        "quality_issue": None if region.get("display_state") else "unreadable",
                    }
                )
        for reading in readings:
            region = regions.get(reading.get("panel_id"), {})
            if job.get("panel_detection") and not region:
                continue
            asset = reading.get("instrument") or {}
            if region and (
                (region.get("instrument") or {}).get("id") != asset.get("id")
                or reading.get("binding_id", region.get("binding_id")) != region.get("binding_id")
            ):
                continue
            iid, name = asset.get("id"), reading.get("measurement_name")
            key = (
                iid,
                name,
                None if name else reading.get("panel_id") or reading.get("reading_id"),
            )
            fid = hashlib.sha256(json.dumps(key).encode()).hexdigest()[:20]
            field = fields.setdefault(
                fid,
                {
                    "field_id": fid,
                    "instrument": asset or None,
                    "name": name,
                    "original_candidates": [],
                    "corrected": False,
                },
            )
            unit, unit_basis = unit_evidence(name, reading, asset)
            bindings = job.get("all_binding_snapshots", job.get("binding_snapshots", []))
            binding = next(
                (b for b in bindings if b.get("binding_id") == reading.get("binding_id")), {}
            )
            qr = next((q for q in job.get("qr_matches", []) if q.get("id") == iid), {})
            digest = qr.get("qr_hash") or binding.get("qr_hash")
            field["original_candidates"].append(
                {
                    "job_id": job["job_id"],
                    "capture_id": job["capture_id"],
                    "captured_at": (job.get("external_photo") or {}).get("captured_at"),
                    "image_sha256": job.get("image_sha256"),
                    "raw_text": reading.get("text"),
                    "raw_value": reading.get("value"),
                    "value": measurement_value(name, reading, asset)[0]["value"],
                    "display_state": (region.get("display_state") or {}).get("text"),
                    "display_state_source": region.get("display_state"),
                    "unit": unit,
                    "unit_basis": unit_basis,
                    "quality_issue": next(iter(field_issues(name, reading, asset)), None),
                    "panel_id": reading.get("panel_id"),
                    "bbox": region.get("bbox"),
                    "polygon": reading.get("polygon"),
                    "digit_region": region.get("digit_region"),
                    "panel_image_url": region.get("image_url"),
                    "panel_image_sha256": region.get("image_sha256"),
                    "confidence": reading.get("confidence"),
                    "confidence_basis": "model_score_not_measured_accuracy",
                    "clarity": reading.get("clarity"),
                    "qr_hash": digest if re.fullmatch("[a-f0-9]{64}", digest or "") else None,
                    "qr_scan_id": job["capture_id"] if qr else binding.get("scan_id"),
                    "qr_conflict": bool(job.get("qr_matches"))
                    and iid not in {q["id"] for q in job["qr_matches"]},
                    "operator": binding.get("operator"),
                    "wearer_id": binding.get("wearer_id"),
                    "identity_basis": reading.get("association_basis"),
                    "binding_id": reading.get("binding_id"),
                }
            )
    for field in fields.values():
        summarize_candidates(field)
    return list(fields.values())


def summarize_candidates(field):
    """Use readable candidates; keep rejected OCR evidence without vetoing the burst.

    Identity/session conflicts still inspect every candidate. A clean candidate
    does not repair missing attribution, override another usable value or imply
    human confirmation. This also works on frozen draft evidence for revalidation.
    """
    rows = field["original_candidates"]
    usable = [
        r
        for r in rows
        if not r.get("quality_issue")
        and (r.get("value") is not None or r.get("display_state") == "OFF")
    ]
    excluded = [r for r in rows if r not in usable]
    unique = {(r["value"], r["unit"]) for r in usable if r["value"] is not None}
    states = {r.get("display_state") for r in usable}
    state_conflict = any(states) and (len(states) > 1 or bool(unique))
    session_conflict = (
        len({(r.get("binding_id"), r.get("operator"), r.get("wearer_id")) for r in rows}) > 1
    )
    units = {r["unit"] for r in usable}
    varying = (
        len(unique) > 1
        and len(units) == 1
        and None not in units
        and not any(states)
        and not session_conflict
        and len({r["capture_id"] for r in usable}) == len(usable)
    )
    field.update(
        observation_status="varying"
        if varying
        else "consistent"
        if len(unique) == 1
        else "unresolved",
        observation_count=len({r["capture_id"] for r in usable}),
        value=next(iter(unique))[0]
        if len(unique) == 1 and not state_conflict and not session_conflict
        else None,
        display_state=next(iter(states))
        if len(states) == 1 and not unique and not session_conflict
        else None,
        unit=next(iter(units)) if len(units) == 1 else None,
        agreement={
            "matching_photos": len(
                {r["image_sha256"] or r["capture_id"] for r in usable if r["value"] is not None}
            )
            if len(unique) == 1 and not state_conflict and not session_conflict
            else 0,
            "photos_with_region": len({r["capture_id"] for r in rows}),
            "basis": "exact_value_and_unit_agreement",
            "accuracy": None,
        },
        validation_rules=RULE_VERSION,
        excluded_candidates=[
            {
                "job_id": r["job_id"],
                "capture_id": r["capture_id"],
                "reason": r.get("quality_issue") or "unreadable",
            }
            for r in excluded
        ],
        recognition_warnings=["部分原始候选不可用，已保留供核对；汇总采用可用候选"]
        if excluded and usable
        else [],
    )
    field["display_value"] = (
        next((r["raw_value"] for r in usable if r["value"] == field["value"]), None)
        if field["value"] is not None
        else None
    )
    issues = []
    if session_conflict:
        issues.append("跨使用时段或实际人员的读数不能合并")
    if (len(unique) > 1 and not varying) or state_conflict:
        issues.append("多图读数或单位冲突")
    if not usable:
        issues.append("没有可用读数：原始结果不可读或格式异常")
    if not field["instrument"] or not rows or not all(r["qr_hash"] and r["bbox"] for r in rows):
        issues.append("二维码身份或面板归属不明确")
    if any(r["qr_conflict"] for r in rows):
        issues.append("照片二维码与面板仪器不一致")
    field["recognition_issues"] = issues


def blockers(group):
    issues = []
    if group["status"] == "collecting":
        issues.append("连拍尚未收齐或识别尚未完成")
    if not group["context"].get("experiment_context_ref"):
        issues.append("缺少实验上下文引用")
    if not group["fields"]:
        issues.append("没有可定位的读数字段，请补拍清晰面板")
    overrides = group.get("capture_time_corrections", {})
    for source in group["sources"]:
        if not source.get("captured_at") and source["capture_id"] not in overrides:
            issues.append("缺少采集时间：" + source["capture_id"])
        if source["status"] != "completed":
            issues.append("有照片未完成识别：" + source["job_id"])
    seen = set()
    for field in group["fields"]:
        prefix = (field.get("name") or "未定位字段") + "："
        if not field["corrected"]:
            issues.extend(prefix + issue for issue in field["recognition_issues"])
        iid = (field.get("instrument") or {}).get("id")
        if not iid or field.get("name") not in UNITS:
            issues.append(prefix + "需要指定仪器及指标")
        if field_binding_issue(group, field):
            issues.append(prefix + "缺少采集时有效绑定，不能确认仪器归属")
        if field.get("value") is None and field.get("display_state") != "OFF":
            issues.append(
                prefix
                + (
                    "已逐张记录不同读数；确认单值前请选择并核对读数"
                    if field.get("observation_status") == "varying"
                    else "需要校正数值"
                )
            )
        if field.get("display_state") != "OFF" and field.get("unit") not in UNITS.get(
            field.get("name"), set()
        ):
            issues.append(prefix + "单位不符或未知")
        if field_issues(
            field.get("name"), field | {"corrected": True}, field.get("instrument") or {}
        ):
            issues.append(prefix + "数值不符合仪器登记的单位、量程或显示精度")
        declared = group["context"]["instrument_ids"]
        if declared and iid not in declared:
            issues.append(prefix + "仪器与输入标识不符")
        key = (iid, field.get("name"))
        if key in seen:
            issues.append(prefix + "同一仪器指标存在重复字段，请重新采集清晰区域")
        seen.add(key)
    return list(dict.fromkeys(issues))


def field_binding_issue(group, field):
    """A reviewer may correct digits, but cannot grant historical occupancy."""
    iid = (field.get("instrument") or {}).get("id")
    captures = {r.get("capture_id") for r in field.get("original_candidates", [])}
    sources = [s for s in group.get("sources", []) if s.get("capture_id") in captures]
    if not iid or not captures or len(sources) != len(captures):
        return "historical_binding_missing"
    sessions = set()
    for source in sources:
        at = group.get("capture_time_corrections", {}).get(source["capture_id"]) or source.get(
            "captured_at"
        )
        if not at or source.get("attribution_status") == "needs_review":
            return "capture_attribution_unconfirmed"
        eligible = []
        for binding in source.get("binding_snapshots", []):
            if (binding.get("instrument") or {}).get("id") != iid:
                continue
            region = {"instrument": binding["instrument"], "binding_id": binding.get("binding_id")}
            document = {
                "camera_id": group.get("camera_id"),
                "external_photo": {"captured_at": at},
                "binding_snapshots": source.get("binding_snapshots", []),
            }
            if binding_for_region(document, region) is not None:
                eligible.append(binding)
        if len(eligible) != 1:
            return "historical_binding_missing" if not eligible else "ambiguous_binding_history"
        binding = eligible[0]
        sessions.add((binding.get("binding_id"), binding.get("operator"), binding.get("wearer_id")))
    if len(sessions) > 1:
        return "binding_session_conflict"
    return None


def instrument_completeness(group):
    """Reviewing a subset of fields does not claim that a whole panel was read."""
    expected = {}
    for source in group.get("sources", []):
        for region in source.get("panel_regions", []):
            iid = (region.get("instrument") or {}).get("id")
            names = (
                ["温度", "转速"]
                if region.get("class_id") == 0
                else ["质量"]
                if region.get("class_id") == 1
                else []
            )
            if iid and names:
                expected.setdefault(iid, names)
    result = []
    for iid, names in expected.items():
        fields = [
            f for f in group.get("fields", []) if (f.get("instrument") or {}).get("id") == iid
        ]
        present = {
            f["name"]
            for f in fields
            if (
                f.get("value") is not None
                and f.get("unit") in UNITS.get(f["name"], set())
                or f.get("display_state") == "OFF"
            )
            and not field_binding_issue(group, f)
        }
        missing = [name for name in names if name not in present]
        result.append(
            {
                "instrument_id": iid,
                "expected_fields": names,
                "missing_fields": missing,
                "status": "incomplete" if missing else "complete",
            }
        )
    return result


def verify_evidence(core, group):
    """Confirmation verifies the preserved byte evidence, not recognition accuracy."""
    root = core["DATA"].resolve()
    for source in group["sources"]:
        files = [
            (
                Path("Images") / (str(uuid.UUID(source["capture_id"])) + ".png"),
                source["image_sha256"],
            ),
            (source.get("original_blob"), source.get("source_sha256")),
        ]
        for region in source.get("panel_regions", []):
            if region.get("image_sha256"):
                files.append(
                    (
                        Path("Images") / (str(uuid.UUID(region["evidence_id"])) + ".png"),
                        region["image_sha256"],
                    )
                )
        for relative, digest in files:
            try:
                path = (root / relative).resolve()
                path.relative_to(root)
                if (
                    not re.fullmatch("[a-f0-9]{64}", digest or "")
                    or hashlib.sha256(path.read_bytes()).hexdigest() != digest
                ):
                    raise ValueError("hash mismatch")
            except (TypeError, ValueError, OSError):
                raise HTTPException(409, "测量图片证据缺失或哈希不一致，不能确认提交") from None


def verify_ownership(conn, group):
    from datetime import datetime

    for field in group["fields"]:
        iid = field["instrument"]["id"]
        captures = {r["capture_id"] for r in field["original_candidates"]}
        others = [
            json.loads(row[0])
            for row in conn.execute(
                "SELECT document FROM bindings WHERE camera<>? AND json_extract(document,'$.instrument.id')=?",
                (group["camera_id"], iid),
            )
        ]
        for source in group["sources"]:
            if source["capture_id"] not in captures:
                continue
            moment = datetime.fromisoformat(
                group.get("capture_time_corrections", {}).get(source["capture_id"])
                or source["captured_at"]
            )
            for other in others:
                from field_recognition.bindings.binding_policy import contains

                if contains(other, moment):
                    raise HTTPException(
                        409,
                        "采集时该仪器由 "
                        + other["operator"]
                        + "（"
                        + other["camera_id"]
                        + "）使用；材料保留，归属须另行处理",
                    )


def refresh(core, conn, mid):
    row = conn.execute("SELECT document FROM photo_measurements WHERE id=?", (mid,)).fetchone()
    if not row:
        return
    group = json.loads(row["document"])
    if group["status"] != "collecting":
        return
    jobs, sources = [], []
    for jid in group["job_ids"]:
        row = conn.execute("SELECT status,document FROM jobs WHERE id=?", (jid,)).fetchone()
        if not row:
            return
        job = json.loads(row["document"]) | {"status": row["status"]}
        capture = json.loads(
            conn.execute("SELECT document FROM scans WHERE id=?", (job["capture_id"],)).fetchone()[
                0
            ]
        )
        jobs.append(job)
        sources.append(
            {
                "job_id": jid,
                "capture_id": job["capture_id"],
                "status": row["status"],
                "result_stage": job.get("result_stage"),
                "parallel_review": job.get("parallel_review"),
                "captured_at": (job.get("external_photo") or {}).get("captured_at"),
                "external_photo": job.get("external_photo"),
                "operator": job.get("operator"),
                "operator_registration": job.get("operator_registration"),
                "binding_snapshots": job.get(
                    "all_binding_snapshots", job.get("binding_snapshots", [])
                ),
                "binding_time_basis": job.get("binding_time_basis"),
                "ownership_conflicts": job.get("ownership_conflicts", []),
                "attribution_status": job.get("attribution_status"),
                "attribution_reason": job.get("attribution_reason"),
                "image_url": capture["image_url"],
                "image_sha256": capture["image_sha256"],
                "original_blob": capture.get("original_blob"),
                "source_sha256": capture.get("source_sha256"),
                "original_url": f"/api/captures/{job['capture_id']}/original",
                "width": capture["width"],
                "height": capture["height"],
                "raw_lines": job.get("lines", []),
                "local_ocr": job.get("local_ocr"),
                "fallback": job.get("fallback"),
                "panel_regions": job.get("panel_regions", []),
                "panel_detection": job.get("panel_detection"),
                "versions": job.get("recognition_versions"),
                "model": job.get("model"),
                "device": job.get("device"),
            }
        )
    previous_sources = group["sources"]
    group["sources"] = sources
    if len(jobs) == group["context"]["expected_photos"] and all(
        j["status"] in TERMINAL and not j.get("resume_pending") for j in jobs
    ):
        group["fields"] = candidates(jobs)
        group["status"] = "draft"
    else:
        if sources == previous_sources:
            return
        if jobs and all(
            j["status"] in TERMINAL
            and not j.get("resume_pending")
            and (j.get("external_photo") or {}).get("batch_basis") == "configured_directory_triplet"
            for j in jobs
        ):
            # Keep the original three-photo context open for late arrivals, but
            # expose all completed observations now. This never confirms a value.
            group["fields"] = candidates(jobs)
    group["revision"] += 1
    group["updated_at"] = core["now"]()
    group["instrument_completeness"] = instrument_completeness(group)
    group["blockers"] = blockers(group)
    conn.execute(
        "UPDATE photo_measurements SET status=?,document=? WHERE id=?",
        (group["status"], json.dumps(group), mid),
    )


def install(core):
    """Compatibility installer; business actions are independent of HTTP adapters."""
    from field_recognition.api.measurements_routes import install as install_routes

    return install_routes(core)
