"""Date/camera event bundles. A burst, retries and confirmation share one result."""

import copy
import hashlib
import html
import json
import posixpath
import time
from datetime import datetime
from pathlib import PurePosixPath
from zoneinfo import ZoneInfo

from field_recognition.archive.archive_browse import page, token
from field_recognition.archive.archive_readable import raw_json
from field_recognition.measurements.measurement_records import (
    build_records,
    capture_clock,
    measurement_value,
    number,
    RULE_VERSION,
)
from field_recognition.measurements.evidence_timeline import acquisition
from field_recognition.archive.archive_progress import advance

KINDS = {
    "Bindings": "二维码与绑定",
    "VideoReadings": "视频面板读数",
    "VoicePhotoReadings": "语音照片读数",
    "PhotoReadings": "上传照片读数",
}


def processing_outcome(job):
    """Recognition completion is independent of later human confirmation."""
    status = job.get("status")
    terminal = status in {"completed", "failed", "cancelled", "interrupted"}
    if job.get("resume_pending") or job.get("phase") in {
        "retry_wait",
        "replayed",
        "waiting_service",
    }:
        terminal = False
    skipped = job.get(
        "recognition_skipped", (job.get("local_ocr") or {}).get("recognition_skipped")
    )
    return {
        "task_id": job.get("job_id"),
        "capture_id": job.get("capture_id"),
        "status": status,
        "terminal": terminal,
        "outcome": ("failed" if status != "completed" else "skipped" if skipped else "succeeded")
        if terminal
        else "pending",
        "reason": (
            job.get("skip_reason") or (job.get("local_ocr") or {}).get("skip_reason")
            if skipped
            else job.get("error")
        ),
    }


def aware(value):
    try:
        moment = datetime.fromisoformat(value)
        return moment.astimezone(ZoneInfo("Asia/Shanghai")) if moment.tzinfo else None
    except (TypeError, ValueError, OverflowError):
        return None


def latest_rows(rows):
    latest, versions = {}, {}
    for number, row in enumerate(sorted(rows, key=lambda r: r["seq"]), 1):
        advance("preparing_event_versions", number, len(rows))
        key = (row["entity"], row["entity_id"])
        current = dict(row)
        for field in (
            "published_at",
            "readable_at",
            "readability_basis",
            "publication_started_at",
            "receipt_publication_ms",
        ):
            current.setdefault(field, None)
        latest[key] = current
        versions.setdefault(key, []).append(
            {
                "sequence": row["seq"],
                "receipt": row["receipt_path"],
                "queued_at": row["recorded_at"],
                "published_at": current["published_at"],
                "readable_at": current["readable_at"],
                "readability_basis": current["readability_basis"],
            }
        )
    for number, (key, row) in enumerate(latest.items(), 1):
        advance("decoding_latest_events", number, len(latest))
        latest[key] = row | {"doc": json.loads(row["document"])}
    return latest, versions


def event_rows(conn, sequences, paths):
    """Load complete histories of changed events, without loading unrelated JSON.

    Only entity IDs and sequence numbers are enumerated across the archive.
    Previously published aliases supply reverse event membership. Business
    tables supply additional IDs while an event is first being published;
    the values used below always come from the latest ARCHIVED snapshot.
    Photograph identities and their job source IDs are retained by the project.
    """
    metadata = {}
    for row in conn.execute(
        "SELECT entity,entity_id,max(seq) AS seq FROM archive_outbox "
        "WHERE archived_at IS NOT NULL GROUP BY entity,entity_id"
    ):
        advance("selecting_event_entities")
        metadata[(row["entity"], row["entity_id"])] = row["seq"]
    entities_by_alias = {}
    for entity, ident in metadata:
        advance("selecting_event_aliases")
        entities_by_alias[
            "Records/"
            + "".join(word.title() for word in entity.split("_"))
            + "/"
            + token(ident)
            + "/Record.json"
        ] = (entity, ident)
    members = {}
    aliases = paths.data["aliases"]

    def alias_target(target):
        seen = set()
        while target in aliases and target not in seen:
            seen.add(target)
            target = aliases[target]
        return target

    for alias, target in aliases.items():
        advance("selecting_event_members")
        entity_key = entities_by_alias.get(alias)
        if entity_key is None:
            continue
        target = alias_target(target)
        members.setdefault(target.rsplit("/", 1)[0], set()).add(entity_key)
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    wanted, todo, visited = set(), [], set()

    def add(entity, ident):
        key = (entity, ident)
        if key in metadata and key not in wanted:
            wanted.add(key)
            todo.append(key)

    for sequence in sequences:
        advance("selecting_pending_events")
        row = conn.execute(
            "SELECT entity,entity_id FROM archive_outbox WHERE seq=? AND archived_at IS NOT NULL",
            (sequence,),
        ).fetchone()
        if row:
            add(row["entity"], row["entity_id"])

    captures, measurements = set(), set()
    while todo:
        advance("selecting_event_closure", len(visited), None)
        key = todo.pop()
        if key in visited:
            continue
        visited.add(key)
        entity, ident = key
        row = conn.execute(
            "SELECT document FROM archive_outbox WHERE seq=?", (metadata[key],)
        ).fetchone()
        doc = json.loads(row["document"])
        event_keys = []
        capture = None
        measurement = None
        if entity in {"bindings", "scene_visits", "binding_handoffs"}:
            capture = doc.get("scan_id")
            event_keys.append(entity + ":" + ident)
        elif entity == "scans":
            capture = doc.get("capture_id") or ident
        elif entity == "jobs":
            capture = doc.get("capture_id")
            measurement = doc.get("measurement_id")
            for job_id in (doc.get("recognition_root_job_id"), doc.get("supersedes_job_id")):
                add("jobs", job_id)
        elif entity in {"photo_measurements", "experiment_records"}:
            measurement = doc.get("measurement_id")
            for job_id in doc.get("job_ids", []) + doc.get("superseded_job_ids", []):
                add("jobs", job_id)
            for source in doc.get("sources", []):
                add("scans", source.get("capture_id"))
        if capture:
            add("scans", capture)
            event_keys.append("capture:" + capture)
            if capture not in captures:
                captures.add(capture)
                if "jobs" in tables:
                    # This lookup uses jobs_capture_lookup; retrieve IDs only.
                    for related in conn.execute(
                        "SELECT id FROM jobs WHERE json_extract(document,'$.capture_id')=?",
                        (capture,),
                    ):
                        advance()
                        add("jobs", related["id"])
                for table in ("bindings", "scene_visits", "binding_handoffs"):
                    if table in tables:
                        for related in conn.execute(
                            f"SELECT id FROM {table} WHERE json_extract(document,'$.scan_id')=?",
                            (capture,),
                        ):
                            advance()
                            add(table, related["id"])
        if measurement:
            add("photo_measurements", measurement)
            event_keys.append("measurement:" + measurement)
            if measurement not in measurements:
                measurements.add(measurement)
                if "experiment_records" in tables:
                    for related in conn.execute(
                        "SELECT id FROM experiment_records WHERE json_extract(document,'$.measurement_id')=?",
                        (measurement,),
                    ):
                        advance()
                        add("experiment_records", related["id"])
        alias = (
            "Records/"
            + "".join(word.title() for word in entity.split("_"))
            + "/"
            + token(ident)
            + "/Record.json"
        )
        directories = (
            {alias_target(aliases[alias]).rsplit("/", 1)[0]} if alias in aliases else set()
        )
        directories.update(
            paths.data["events"][event] for event in event_keys if event in paths.data["events"]
        )
        for directory in directories:
            for related_entity, related_id in members.get(directory, ()):
                add(related_entity, related_id)
    rows = []
    for number, (entity, ident) in enumerate(sorted(wanted), 1):
        advance("loading_event_versions", number, len(wanted))
        rows.extend(
            conn.execute(
                "SELECT * FROM archive_outbox WHERE entity=? AND entity_id=? "
                "AND archived_at IS NOT NULL ORDER BY seq",
                (entity, ident),
            ).fetchall()
        )
    return rows


def clock_for(doc, capture):
    value, basis = capture_clock(doc)
    if not aware(value):
        value, basis = capture_clock(capture)
    return (value, basis) if aware(value) else (None, None)


def prepare(rows, paths):
    latest, versions = latest_rows(rows)
    captures = {i: r["doc"] for (e, i), r in latest.items() if e == "scans"}
    jobs = {i: r for (e, i), r in latest.items() if e == "jobs"}
    groups = {
        r["doc"]["measurement_id"]: r for (e, _), r in latest.items() if e == "photo_measurements"
    }
    measurement_keys = {}
    # A confirmed record supersedes its draft view; both immutable histories remain linked.
    for (entity, _), row in latest.items():
        advance("preparing_measurement_groups")
        if entity in {"photo_measurements", "experiment_records"}:
            measurement_keys.setdefault(row["doc"]["measurement_id"], set()).add(
                (entity, row["entity_id"])
            )
        if entity == "experiment_records":
            mid = row["doc"]["measurement_id"]
            current = groups.get(mid)
            if (
                not current
                or current["doc"].get("status") == "confirmed"
                and row["doc"].get("record_revision", 1) >= current["doc"].get("record_revision", 1)
            ):
                groups[mid] = row
    bundles, used = [], set()

    def add(key, kind, members, source_ids, job_rows=(), group=None):
        first = min(members, key=lambda r: r["seq"])
        doc = first["doc"]
        sources = [captures[i] for i in dict.fromkeys(source_ids) if i in captures]
        timestamps = [
            clock_for(j["doc"], captures.get(j["doc"].get("capture_id"), {})) for j in job_rows
        ]
        if not timestamps:
            timestamps = [clock_for(s, s) for s in sources]
        timestamp, basis = min(
            (t for t in timestamps if t[0]), key=lambda t: aware(t[0]), default=(None, None)
        )
        if kind == "Bindings":
            timestamp = doc.get("started_at") or doc.get("requested_at") or doc.get("received_at")
            basis = "binding_event_time" if aware(timestamp) else None
        # Folder fallback is explicit; never written as a photo capture timestamp.
        local = aware(timestamp) or aware(first["recorded_at"])
        camera = doc.get("camera_id") or next(
            (s.get("camera_id") for s in sources if s.get("camera_id")), None
        )
        day = local.strftime("%Y-%m-%d") + "_" + token(camera)
        stamp = local.strftime("%H-%M-%S.%f")[:-3] if basis else "UnknownTime"
        proposed = f"{day}/{kind}/{stamp}_{token(key.split(':', 1)[1])}"
        if kind == "SourceMaterials":
            proposed = ".System/SourceMaterials/" + proposed
        directory = paths.directory(key, proposed)
        file = (
            "Binding.json"
            if kind == "Bindings"
            else "Evidence.json"
            if kind != "SourceMaterials"
            else "Source.json"
        )
        keys = {(r["entity"], r["entity_id"]) for r in members}
        for sid in source_ids:
            if ("scans", sid) in latest:
                keys.add(("scans", sid))
        if group:
            keys.update(measurement_keys.get(group["measurement_id"], ()))
        history = sorted(
            (v for k in keys for v in versions.get(k, [])), key=lambda v: v["sequence"]
        )
        bundles.append(
            {
                "key": key,
                "kind": kind,
                "directory": directory,
                "file": file,
                "camera_id": camera,
                "event_at": timestamp if basis else None,
                "time_basis": basis,
                "folder_time_basis": basis or "receipt_date_capture_time_unknown",
                "day": directory.split("/")[0],
                "members": members,
                "sources": sources,
                "jobs": list(job_rows),
                "group": group,
                "history": history,
                "entity_keys": keys,
            }
        )
        used.update(source_ids)

    for (entity, ident), row in latest.items():
        advance("preparing_binding_events")
        if entity in {"bindings", "scene_visits", "binding_handoffs"}:
            add(entity + ":" + ident, "Bindings", [row], [row["doc"].get("scan_id")])
    grouped_jobs = set()
    for mid, row in groups.items():
        advance("preparing_group_events")
        group = row["doc"]
        member_ids = list(dict.fromkeys(group["job_ids"] + group.get("superseded_job_ids", [])))
        linked = [jobs[jid] for jid in member_ids if jid in jobs]
        grouped_jobs.update(member_ids)
        ids = [j["doc"]["capture_id"] for j in linked] + [
            s["capture_id"] for s in group.get("sources", [])
        ]
        kind = (
            "VoicePhotoReadings"
            if any((captures.get(i) or {}).get("external_photo") for i in ids)
            else "PhotoReadings"
        )
        add("measurement:" + mid, kind, [row] + linked, ids, linked, group)
    standalone = {}
    for jid, row in jobs.items():
        advance("preparing_standalone_jobs")
        if jid not in grouped_jobs:
            standalone.setdefault(row["doc"].get("capture_id") or row["entity_id"], []).append(row)
    for cid, linked in standalone.items():
        advance("preparing_capture_events")
        doc = linked[0]["doc"]
        source = captures.get(cid, {})
        kind = (
            "VideoReadings"
            if doc.get("request_trigger") == "video_stream"
            else (
                "VoicePhotoReadings"
                if doc.get("external_photo") or source.get("external_photo")
                else "PhotoReadings"
            )
        )
        add("capture:" + cid, kind, linked, [cid], linked)
    for cid, source in captures.items():
        advance("preparing_source_events")
        if cid in used:
            continue
        kind = (
            "Bindings"
            if source.get("matches") or source.get("scene_matches")
            else ("VoicePhotoReadings" if source.get("external_photo") else "SourceMaterials")
        )
        add("capture:" + cid, kind, [latest[("scans", cid)]], [cid])
    return bundles, latest


def standard_records(job_rows, group, *, diagnostics=None):
    """Keep the single-value review projection separate from per-capture observations."""
    by_capture = {}
    from field_recognition.measurements.result_reprocessing import preferred_jobs

    preferred = {d["job_id"] for d in preferred_jobs([r["doc"] for r in job_rows])}
    for row in sorted(job_rows, key=lambda r: r["seq"]):
        if row["doc"]["job_id"] not in preferred:
            continue
        by_capture[row["doc"].get("capture_id") or row["entity_id"]] = row["doc"]
    entries = {}
    for original in by_capture.values():
        job = copy.deepcopy(original)
        if group and job.get("capture_id") in group.get("capture_time_corrections", {}):
            job["external_photo"] = (job.get("external_photo") or {}) | {
                "captured_at": group["capture_time_corrections"][job["capture_id"]]
            }
        # Recompute derived output under current explicit rules, without rewriting receipts.
        for entry in build_records(job):
            snapshot = next(
                (
                    b
                    for b in job.get("all_binding_snapshots", job.get("binding_snapshots", []))
                    if b.get("binding_id") == entry["evidence"].get("binding_id")
                ),
                {},
            )
            entry["evidence"]["operator"] = snapshot.get("operator", job.get("operator"))
            entries.setdefault(entry["instrument_id"], []).append(entry)
    # Corrections remain in the decision/evidence history. Only an instrument
    # established by build_records from the task's binding/region evidence may
    # receive a Result; a correction cannot manufacture a missing QR binding.
    result = []
    for iid, parts in entries.items():
        sessions = {
            (
                part["evidence"].get("binding_id"),
                part["record"].get("wearer_id"),
                part["evidence"].get("operator"),
            )
            for part in parts
        }
        if len(sessions) > 1:
            if diagnostics is not None:
                diagnostics.append(
                    {
                        "instrument_id": iid,
                        "reason": "binding_session_conflict",
                        "status": "needs_review",
                        "observations": [part["evidence"] for part in parts],
                    }
                )
            continue  # A burst cannot collapse usage periods into its first wearer.
        first = copy.deepcopy(parts[0])
        conflicts, varying = [], []
        for value in first["record"]["values"]:
            name = value["name"]
            alternatives = [v for p in parts for v in p["record"]["values"] if v["name"] == name]
            unique = {(v["value"], v["unit"], v.get("display_state")) for v in alternatives}
            if len(unique) > 1:
                value.update(
                    value=None,
                    unit=value["unit"] if len({v["unit"] for v in alternatives}) == 1 else None,
                )
                value.pop("display_state", None)
                if len({v["unit"] for v in alternatives}) == 1 and all(
                    v["value"] is not None and not v.get("display_state") for v in alternatives
                ):
                    varying.append(name)
                else:
                    conflicts.append(name)
            field = next(
                (
                    f
                    for f in (group or {}).get("fields", [])
                    if (f.get("instrument") or {}).get("id") == iid and f.get("name") == name
                ),
                None,
            )
            if field:
                # Group fields are the explicitly reconciled/corrected result, with raw candidates retained.
                revised, _ = measurement_value(name, field, field["instrument"])
                if not field.get("corrected") and any(
                    c.get("unit_basis") == "unit_conflict"
                    for c in field.get("original_candidates", [])
                ):
                    revised.update(value=None, unit=None, range=[None, None])
                value.update(revised)
                if field.get("corrected") or revised["value"] is not None:
                    value.pop("display_state", None)
                if field.get("corrected") and name in conflicts:
                    conflicts.remove(name)
                if field.get("corrected") and name in varying:
                    varying.remove(name)
        first["evidence"] = {
            "schema_version": "instrument-measurement/1",
            "rule_version": RULE_VERSION,
            "observations": [p["evidence"] for p in parts],
            "conflicting_fields": conflicts,
            "varying_fields": varying,
            "human_verified": bool(group and group.get("status") == "confirmed"),
        }
        result.append(first)
    return result


def photo_candidates(jobs, sources, *, unbound_only=True):
    """Expose per-photo candidates without creating instrument or burst identity."""
    captures = {s["capture_id"]: s for s in sources}
    result = []
    for job in jobs:
        captured_at, basis = clock_for(job, captures.get(job.get("capture_id"), {}))
        for reading in job.get("readings", []):
            if unbound_only and (reading.get("instrument") or reading.get("binding_id")):
                continue
            issues = list(
                dict.fromkeys(
                    [
                        *(reading.get("quality_issues") or []),
                        *([reading["quality_issue"]] if reading.get("quality_issue") else []),
                    ]
                )
            )
            normalized = reading.get("normalized_value")
            if isinstance(normalized, dict) and normalized.get("value") is None and not issues:
                issues.append("normalized_value_unavailable")
            value = number(reading) if not issues else None
            if value is None and not issues:
                issues.append("invalid_or_uncertain_number")
            normalization = copy.deepcopy(reading.get("normalization"))
            result.append(
                {
                    "job_id": job["job_id"],
                    "capture_id": job.get("capture_id"),
                    "recognition_root_job_id": job.get("recognition_root_job_id", job["job_id"]),
                    "recognition_revision": job.get("recognition_revision", 1),
                    "result_stage": job.get("result_stage"),
                    "result_revision": job.get("result_revision"),
                    "parallel_review": job.get("parallel_review"),
                    "captured_at": captured_at,
                    "capture_time_basis": basis,
                    "observed_at": (job.get("video_observation") or {}).get("observed_at"),
                    "reading_id": reading.get("reading_id"),
                    "instrument_id": (reading.get("instrument") or {}).get("id")
                    if not unbound_only
                    else None,
                    "qr_hash": None,
                    "binding_id": reading.get("binding_id") if not unbound_only else None,
                    "identity_basis": reading.get("association_basis") or "unbound",
                    "raw_text": reading.get("text"),
                    "raw_value": reading.get(
                        "raw_value", (normalization or {}).get("raw_value", reading.get("value"))
                    ),
                    "raw_unit": reading.get("unit"),
                    "value": value,
                    "display_value": str(reading["value"]) if value is not None else None,
                    "unit": None
                    if any("unit_conflict" in issue for issue in issues)
                    else (
                        normalized.get("unit")
                        if isinstance(normalized, dict)
                        else reading.get("unit")
                    ),
                    "measurement_name": reading.get("measurement_name")
                    or (normalized or {}).get("name"),
                    "quality_issue": issues[0] if issues else None,
                    "quality_issues": issues,
                    "normalization": normalization,
                    "normalized_value": normalized,
                    "display_rule": reading.get("display_rule"),
                    "status": "needs_review",
                    "human_verified": False,
                    "activity_confirmed": False,
                }
            )
    return result


def photo_readme(
    content, *, jobs=(), measurement_files=(), records=(), aggregation_issues=(), missing_assets=()
):
    esc = lambda value: html.escape(str(value if value is not None else "未知"), quote=True)
    unbound = content.get("record_scope") == "unbound_candidates"
    body = (
        "<p><strong>"
        + (
            "仪器身份未确认；以下是待核实候选，不是正式测量结果。"
            if unbound
            else "逐张识别候选与合并结果分别展示；候选未经人工核实。"
        )
        + "</strong>每张照片的读数分别记录；数值变化不自动判为识别错误，也不取平均或多数值。</p>"
    )
    body += '<p><a href="Readings.json">逐次读数 JSON</a> · <a href="Evidence.json">完整证据与历史版本</a>'
    if unbound:
        body += ' · <a href="UnboundReadings.json">未绑定候选 JSON</a>'
    for measurement in measurement_files:
        body += (
            ' · <a href="'
            + esc(PurePosixPath(measurement["result"]).name)
            + '">'
            + esc(measurement["instrument_name"] or measurement["instrument_id"])
            + " 单值复核结果</a>"
        )
    body += "</p>"
    for job in jobs:
        if job.get("reading_mode") == "parallel_photo":
            review_status = (job.get("parallel_review") or {}).get("status")
            label = {
                "running": "本地初读 · 云端复核中",
                "failed": "本地初读 · 云端复核未完成",
                "conflict": "复核存在冲突 · 待核对",
                "unreadable": "云端未读清 · 待核对",
                "completed": "模型复核完成 · 未经人工核实",
            }.get(review_status, "本地与云端正在识别")
            if (job.get("parallel_review") or {}).get("result") == "unverified":
                label = "仅整图候选 · 面板未定位 · 待核实"
            if (job.get("parallel_review") or {}).get("crosscheck_pending"):
                label = "初步读数 · 本地交叉检查中"
            body += "<p>" + esc(job.get("job_id")) + "：" + esc(label) + "</p>"
    if missing_assets:
        body += (
            f"<p><strong>历史图片证据缺失 {len(missing_assets)} 项，"
            "已在此前完整性巡检中发现；未恢复的图片不可查看。"
            "读数与原回执保留，缺失明细见 Evidence.json。</strong></p>"
        )
    if jobs and all(
        (j.get("external_photo") or {}).get("batch_basis") == "configured_directory_triplet"
        for j in jobs
    ):
        count = len({j["capture_id"] for j in jobs})
        body += (
            f"<p>同一三连拍批次已到达 {count}/3 张；现有照片分别处理并记录，"
            "后续照片继续加入此批次，重复上传复用原任务。</p>"
        )
    conflicts = list(
        dict.fromkeys(
            name for entry in records for name in entry["evidence"].get("conflicting_fields", [])
        )
    )
    if conflicts or aggregation_issues:
        body += (
            "<p><strong>连拍存在冲突，已识别的逐张数字见下表。冲突字段的合并值保持 null，不能视为有效读数。</strong> "
            + esc("、".join(conflicts) or "归属/使用时段冲突")
            + "</p>"
        )
    varying = list(
        dict.fromkeys(
            name for entry in records for name in entry["evidence"].get("varying_fields", [])
        )
    )
    if varying:
        body += (
            "<p>"
            + esc("、".join(varying))
            + "读数有变化，所有逐张数值已记录如下；单值复核视图暂不选定一个数值。</p>"
        )
    body += "<table><tr><th>拍摄时间 / 任务</th><th>面板原文</th><th>候选值 / 单位</th><th>规则与质量问题</th></tr>"
    for reading in content["readings"]:
        value = esc(reading["display_value"]) if reading["value"] is not None else "不可作为有效值"
        body += (
            "<tr><td>"
            + esc(reading["captured_at"])
            + "<br>"
            + esc(reading["job_id"])
            + "</td><td>"
            + esc(reading["raw_text"])
            + "</td><td>"
            + value
            + " "
            + esc(reading["unit"])
            + "<br>待核实</td><td>"
            + esc("、".join(reading["quality_issues"]) or "未发现已知规则冲突；未经人工核实")
            + "<br>"
            + esc((reading["normalization"] or {}).get("method"))
            + "</td></tr>"
        )
    if not content["readings"]:
        state = (
            "已完成识别，未读出结构化数值。"
            if jobs and all(j.get("status") == "completed" for j in jobs)
            else "当前没有结构化候选。"
        )
        body += '<tr><td colspan="4">' + state + "处理状态与原因请查看 Evidence.json。</td></tr>"
    body += "</table>"
    for job in jobs:
        cloud = (job.get("fallback") or {}).get("answer") or {}
        if cloud.get("status") == "unreadable":
            body += (
                "<p>已完成识别，未读出仪器数值：" + esc(cloud.get("notes") or "面板不可读") + "</p>"
            )
        body += (
            "<details><summary>逐张原始识别文字 · "
            + esc(job["job_id"])
            + "</summary><pre>"
            + esc(json.dumps(job.get("lines", []), ensure_ascii=False, indent=2))
            + "</pre></details>"
        )
    return page("逐次读数记录 · 未绑定" if unbound else "逐次读数记录", body)


def make_views(
    bundles,
    paths,
    receipts,
    *,
    preserve_asset_locations=False,
    placed_assets=None,
    known_missing_assets=frozenset(),
):
    views, days = {}, {}
    owners = {}
    from field_recognition.archive.source_layout import source_relative

    uploads = {
        source["source_sha256"]: relative
        for bundle in bundles
        for source in bundle["sources"]
        if paths.layout == "metadata"
        and source.get("source_sha256")
        and (relative := source_relative(paths.root, source, source_roots=paths.source_roots))
    }
    # Choose one physical location per hash, even if QR and readout share a photograph.
    for bundle in sorted(bundles, key=lambda b: (b["kind"] == "SourceMaterials", b["directory"])):
        artifacts = {}
        for version in bundle["history"]:
            for label, item in receipts[version["sequence"]].get("artifacts", {}).items():
                if isinstance(item, dict):
                    artifacts.setdefault(item["sha256"], (label, item))
        bundle["artifacts"] = artifacts
        region_names = {
            r.get("image_sha256"): {"温度": "Temperature", "转速": "Speed", "质量": "Mass"}.get(
                r.get("measurement_name"), "Panel"
            )
            for job in bundle["jobs"]
            for r in job["doc"].get("panel_regions", [])
        }
        for digest, (label, item) in artifacts.items():
            suffix = PurePosixPath(item["path"]).suffix
            role = (
                region_names.get(digest, "Panel")
                if "panel" in label
                else "Original"
                if "original" in label
                else "Frame"
                if bundle["kind"] == "VideoReadings"
                else "Photo"
            )
            folder = "Regions" if "panel" in label else "Photos"
            owners.setdefault(digest, f"{bundle['directory']}/{folder}/{role}{digest[:16]}{suffix}")
            if digest in uploads:
                owners[digest] = uploads[digest]
            stored = paths.data["assets"].get(digest, {}).get("path", "")
            if (
                preserve_asset_locations
                and paths.layout != "metadata"
                and stored
                and not stored.startswith((".System/PendingAssets/", "Objects/"))
            ):
                owners[digest] = stored
    # The timing confirmation repeats this derivation while the same writer
    # still holds the migration gate. Reuse only assets placed in that pass;
    # a later publication must validate/place its own dependencies again.
    placed = {} if placed_assets is None else placed_assets
    for bundle in bundles:
        for digest, (_, item) in bundle["artifacts"].items():
            if digest not in placed:
                stored = paths.data["assets"].get(digest, item)
                if item["path"] in known_missing_assets and not paths._resolve(stored["path"])[2]:
                    # Rebuild navigation around a previously audited loss. Do
                    # not manufacture bytes, move aliases or claim restoration.
                    placed[digest] = item | {
                        "available": False,
                        "availability_basis": "previous_integrity_report_missing",
                    }
                    continue
                placed[digest] = (
                    paths.place(item, owners[digest], pin_source=True)
                    if digest in uploads
                    else paths.place(item, owners[digest])
                )
    for bundle in bundles:
        directory = bundle["directory"]
        file = directory + "/" + bundle["file"]
        history = [{**v, "receipt": paths.name(v["receipt"])} for v in bundle["history"]]
        photos = {
            digest: placed[digest] | {"kind": label}
            for digest, (label, _) in bundle["artifacts"].items()
        }

        def evidence(digest):
            return photos.get(digest)

        sources = []
        for source in bundle["sources"]:
            timestamp, basis = clock_for(source, source)
            sources.append(
                {
                    "capture_id": source["capture_id"],
                    "captured_at": timestamp,
                    "capture_time_basis": basis,
                    "received_at": source.get("received_at"),
                    "external_photo": source.get("external_photo"),
                    "video_observation": source.get("video_observation"),
                    "frame_metadata": source.get("frame_metadata"),
                    "camera_id": source.get("camera_id"),
                    "operator": source.get("operator"),
                    "original": evidence(source.get("source_sha256")),
                    "image": evidence(source.get("image_sha256")),
                    "qr_matches": source.get("matches", []),
                    "scene_matches": source.get("scene_matches", []),
                }
            )
        group = bundle["group"]
        doc = max(bundle["members"], key=lambda r: r["seq"])["doc"]
        merge_started = time.monotonic()
        aggregation_issues = []
        records = standard_records(bundle["jobs"], group, diagnostics=aggregation_issues)
        candidate_merge_ms = round((time.monotonic() - merge_started) * 1000, 2)
        observations = []
        for job in bundle["jobs"]:
            row = job
            job = copy.deepcopy(row["doc"])
            from field_recognition.archive.archive_store import publication_timing

            job["archive"] = {
                "queued_at": row["recorded_at"],
                "published_at": row["published_at"],
                "readable_at": row["readable_at"],
                "readability_basis": row["readability_basis"],
            } | publication_timing(row)
            if job.get("submitted_at"):
                from field_recognition.measurements.readout_timing import update_timing

                update_timing(job, preserve_recorded_durations=True)
            observations.append(
                {
                    k: job.get(k)
                    for k in (
                        "job_id",
                        "capture_id",
                        "status",
                        "reading_mode",
                        "result_available",
                        "result_stage",
                        "result_revision",
                        "result_updated_at",
                        "first_result_at",
                        "parallel_review",
                        "lines",
                        "readings",
                        "operator",
                        "wearer_id",
                        "operator_basis",
                        "attribution_status",
                        "attribution_reason",
                        "binding_snapshot_version",
                        "binding_effective_at",
                        "binding_time_basis",
                        "allowed_instrument_ids",
                        "allowed_instrument_ids_basis",
                        "local_ocr",
                        "localized_attempt",
                        "whole_photo_recovery",
                        "fallback",
                        "panel_regions",
                        "recognition_versions",
                        "panel_detection",
                        "photo_orientation",
                        "timing",
                        "archive",
                        "instrument",
                        "all_binding_snapshots",
                        "binding_snapshots",
                        "workbench",
                        "workbenches",
                        "ownership_conflicts",
                        "frame_metadata",
                        "video_observation",
                        "model",
                        "device",
                        "submitted_at",
                        "finished_at",
                        "request_trigger",
                        "field_rules",
                        "recognition_root_job_id",
                        "recognition_revision",
                        "supersedes_job_id",
                        "reprocessing",
                        "display_fields",
                        "actual_model_invocation",
                        "recognition_skipped",
                        "skip_reason",
                        "failure_reason",
                        "error",
                        "outcome",
                    )
                }
            )
        content = {
            "schema": "field-recognition-event/2",
            "derived_view": True,
            "event_id": bundle["key"],
            "event_at": bundle["event_at"],
            "time_basis": bundle["time_basis"],
            "folder_time_basis": bundle["folder_time_basis"],
            "camera_id": bundle["camera_id"],
            "operator": doc.get("operator")
            or next((s["operator"] for s in sources if s["operator"]), None),
            "category": bundle["kind"],
            "record_scope": (group or doc).get(
                "record_scope", "evidence" if bundle["kind"] == "Bindings" else "legacy_test_only"
            ),
            "status": (group or doc).get("status")
            or (
                "ended"
                if doc.get("ended_at")
                else "active"
                if doc.get("binding_id") or doc.get("visit_id")
                else "source_retained"
            ),
            "sources": sources,
            "photos": photos,
            "receipt_versions": history,
        }
        missing_assets = [p for p in photos.values() if p.get("available") is False]
        if missing_assets:
            content["evidence_integrity"] = {
                "status": "historical_assets_missing",
                "missing_assets": missing_assets,
                "bytes_reconstructed": False,
                "immutable_receipts_rewritten": False,
            }
        if observations:
            people = {
                (b.get("operator"), b.get("wearer_id"))
                for observation in observations
                for b in observation.get("all_binding_snapshots")
                or observation.get("binding_snapshots")
                or []
            }
            if len(people) > 1:
                content.update(
                    operator=None, personnel_attribution="multiple_sessions_see_observations"
                )
        if bundle["kind"] in {"VideoReadings", "VoicePhotoReadings"}:
            content["acquisition"] = [
                acquisition(row["doc"], group)
                | {"job_id": row["doc"]["job_id"], "capture_id": row["doc"].get("capture_id")}
                for row in bundle["jobs"]
            ]
        latest_member = max(bundle["members"], key=lambda r: r["seq"])
        content["archive"] = {
            "queued_at": latest_member["recorded_at"],
            "published_at": latest_member["published_at"],
            "readable_at": latest_member["readable_at"],
            "readability_basis": latest_member["readability_basis"],
        }
        from field_recognition.archive.archive_store import publication_timing

        content["archive"].update(publication_timing(latest_member))
        if "archive_history" in bundle:
            content["archive"]["history"] = bundle["archive_history"]
        if bundle["kind"] == "Bindings":
            content["binding"] = doc
        else:
            measurement_files = []
            for entry in sorted(records, key=lambda r: r["instrument_id"]):
                result_file = paths.result_file(bundle["key"], entry["instrument_id"], directory)
                raw = raw_json(entry["record"])
                views[result_file] = raw
                assets = [
                    r.get("instrument") or {}
                    for j in bundle["jobs"]
                    for r in j["doc"].get("panel_regions", [])
                ]
                assets += [f.get("instrument") or {} for f in (group or {}).get("fields", [])]
                name = next(
                    (
                        a.get("name")
                        for a in assets
                        if a.get("id") == entry["instrument_id"] and a.get("name")
                    ),
                    None,
                )
                measurement_files.append(
                    {
                        "instrument_id": entry["instrument_id"],
                        "instrument_name": name,
                        "result": result_file,
                        "sha256": hashlib.sha256(raw).hexdigest(),
                        "evidence": entry["evidence"],
                    }
                )
            content.update(
                measurement_id=group.get("measurement_id")
                if group
                else bundle["key"].split(":", 1)[1],
                measurement_files=measurement_files,
                observations=observations,
                aggregation_issues=aggregation_issues,
            )
            # Completed is a worker lifecycle state, not proof that OCR was invoked.
            from field_recognition.measurements.result_reprocessing import preferred_jobs

            current_jobs = preferred_jobs([r["doc"] for r in bundle["jobs"]])
            candidates = photo_candidates(current_jobs, bundle["sources"])
            if bundle["kind"] == "VoicePhotoReadings" and (candidates or not measurement_files):
                candidate_file = directory + "/UnboundReadings.json"
                candidate_content = {
                    "schema": "field-unbound-readings/1",
                    "derived_view": True,
                    "title": "未绑定照片读数候选（待核实，非正式测量结果）",
                    "event_id": bundle["key"],
                    "camera_id": bundle["camera_id"],
                    "instrument_id": None,
                    "qr_hash": None,
                    "human_verified": False,
                    "record_scope": "unbound_candidates",
                    "status": "needs_review",
                    "aggregation": "per_photo_candidates_not_merged",
                    "readings": candidates,
                    "evidence": file,
                    "receipt_versions": history,
                }
                raw = raw_json(candidate_content)
                views[candidate_file] = raw
                content["unbound_readings"] = {
                    "path": candidate_file,
                    "readme": directory + "/Readme.html",
                    "count": len(candidates),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "human_verified": False,
                }
            if bundle["kind"] in {"VoicePhotoReadings", "PhotoReadings", "VideoReadings"}:
                readings = photo_candidates(current_jobs, bundle["sources"], unbound_only=False)
                # A compact observation ledger. Full raw/model/rule history stays
                # in Evidence.json and immutable receipts, rather than each row.
                reading_file = directory + "/Readings.json"
                reading_content = {
                    "schema": "field-numeric-observations/1",
                    "event_id": bundle["key"],
                    "camera_id": bundle["camera_id"],
                    "record_scope": "test_only"
                    if content["record_scope"] == "test_only"
                    else "field_observation",
                    "human_verified": False,
                    "aggregation": "per_capture_not_merged",
                    "readings": [
                        {
                            k: r[k]
                            for k in (
                                "capture_id",
                                "captured_at",
                                "capture_time_basis",
                                "observed_at",
                                "job_id",
                                "reading_id",
                                "instrument_id",
                                "binding_id",
                                "identity_basis",
                                "measurement_name",
                                "value",
                                "display_value",
                                "unit",
                                "quality_issues",
                                "result_stage",
                                "result_revision",
                                "parallel_review",
                            )
                        }
                        for r in readings
                    ],
                    "evidence": "Evidence.json",
                }
                raw = raw_json(reading_content)
                views[reading_file] = raw
                content["reading_observations"] = {
                    "path": reading_file,
                    "count": len(readings),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "human_verified": False,
                }
                content["readings_readme"] = directory + "/Readme.html"
                views[content["readings_readme"]] = photo_readme(
                    {
                        "record_scope": "unbound_candidates"
                        if not measurement_files
                        else "per_photo_candidates",
                        "readings": readings,
                    },
                    jobs=current_jobs,
                    measurement_files=measurement_files,
                    records=records,
                    aggregation_issues=aggregation_issues,
                    missing_assets=missing_assets,
                )
            latest_attempts = {}
            for row in bundle["jobs"]:
                job = row["doc"]
                key = job.get("recognition_root_job_id", job["job_id"])
                if key not in latest_attempts or job.get(
                    "recognition_revision", 1
                ) > latest_attempts[key].get("recognition_revision", 1):
                    latest_attempts[key] = job
            processing_jobs = list(latest_attempts.values())
            content["group_id"] = (group or {}).get("context", {}).get("burst_id") or content[
                "measurement_id"
            ]
            content["task_ids"] = [j["job_id"] for j in processing_jobs]
            content["task_outcomes"] = [processing_outcome(j) for j in processing_jobs]
            expected = (group or {}).get("context", {}).get("expected_photos", 1)
            arrived = len({j.get("capture_id") for j in processing_jobs})
            directory_batch = bool(processing_jobs) and all(
                (j.get("external_photo") or {}).get("batch_basis") == "configured_directory_triplet"
                for j in processing_jobs
            )
            content["photo_collection"] = {
                "arrived_photos": arrived,
                "expected_photos": expected,
                "complete": arrived >= expected,
                "processing_requires_all_expected_photos": not directory_batch,
            }
            content["processing_terminal"] = (
                bool(processing_jobs)
                and all(t["terminal"] for t in content["task_outcomes"])
                and (directory_batch or arrived >= expected)
            )
            outcomes = {t["outcome"] for t in content["task_outcomes"]}
            content["processing_outcome"] = (
                (
                    "failed"
                    if "failed" in outcomes
                    else "skipped"
                    if outcomes == {"skipped"}
                    else "succeeded"
                )
                if content["processing_terminal"]
                else "pending"
            )
            content["processing_status"] = content["status"]
            if content["status"] == "completed" and not records:
                skipped = current_jobs and all(
                    processing_outcome(j)["outcome"] == "skipped" for j in current_jobs
                )
                content["status"] = (
                    "skipped"
                    if skipped
                    else "unassigned"
                    if any(j.get("readings") for j in current_jobs)
                    else "unreadable"
                )
            content["skip_reasons"] = list(
                dict.fromkeys(
                    task["reason"]
                    for task in content["task_outcomes"]
                    if task["outcome"] == "skipped" and task["reason"]
                )
            )
            if not measurement_files:
                # Keep old HTTP links readable without pretending that a measurement exists.
                paths.alias(directory + "/Result.json", file)
            if group:
                # Sources/evidence are normalized above; the decision history stays intact.
                content["decision"] = {k: v for k, v in group.items() if k != "sources"}
            # Each file is atomic independently. Readers use this manifest and
            # validate hashes; a new Result with an old Evidence is not a snapshot.
            content["publication"] = {
                "protocol": "evidence_manifest_sha256/1",
                "atomic_scope": "one_file_same_directory_rename",
                "reader_rule": "parse Evidence, parse every measurement_file and verify sha256; retry mismatches",
                "result_count": len(measurement_files),
                "generation": hashlib.sha256(
                    json.dumps(
                        {
                            "receipts": [v["sequence"] for v in history],
                            "results": [(m["result"], m["sha256"]) for m in measurement_files],
                            "unbound_readings": content.get("unbound_readings"),
                        },
                        sort_keys=True,
                    ).encode()
                ).hexdigest(),
            }
            # Keep the first measured rebuild for this generation. Re-reading or
            # re-indexing unchanged history must not rewrite its evidence bytes.
            timings = paths.data.setdefault("event_stage_timings", {})
            previous = timings.get(bundle["key"], {})
            if previous.get("generation") != content["publication"]["generation"]:
                timings[bundle["key"]] = {
                    "generation": content["publication"]["generation"],
                    "group_id": content["group_id"],
                    "task_ids": content["task_ids"],
                    "durations_ms": {"candidate_merge_ms": candidate_merge_ms},
                    "duration_clock_basis": {"candidate_merge_ms": "process_monotonic"},
                    "scope": "first_derived_view_rebuild_for_generation",
                }
            content["timing"] = timings[bundle["key"]]
        views[file] = raw_json(content)
        for entity, ident in bundle["entity_keys"]:
            old = "Records/" + "".join(w.title() for w in entity.split("_")) + "/" + token(ident)
            paths.alias(old + "/Record.json", file)
            paths.alias(
                old + "/Readme.html",
                bundle["day"] + "/DailyReport/DailyReport.html"
                if bundle["kind"] != "SourceMaterials"
                else "Readme.html",
            )
        if bundle["kind"] != "SourceMaterials":
            target = (
                (doc.get("instrument") or {}).get("name")
                or (doc.get("scene") or {}).get("name")
                or "、".join(
                    dict.fromkeys(
                        r["instrument"].get("name") or r["instrument"]["id"]
                        for j in bundle["jobs"]
                        for r in j["doc"].get("panel_regions", [])
                        if r.get("instrument")
                    )
                )
            )
            target = target or "、".join(
                v.get("name") or v["id"]
                for v in doc.get("matches", []) + doc.get("scene_matches", [])
            )
            days.setdefault(bundle["day"], []).append(
                {
                    "event_id": bundle["key"],
                    "event_at": bundle["event_at"],
                    "category": bundle["kind"],
                    "target": target or "归属待确认",
                    "operator": content["operator"],
                    "status": content["status"],
                    "result": file,
                    "acquisition": content.get("acquisition", []),
                    "unbound_readings": content.get("unbound_readings"),
                    "readings_readme": content.get("readings_readme"),
                    "measurement_files": [
                        {k: m[k] for k in ("instrument_id", "instrument_name", "result")}
                        for m in content.get("measurement_files", [])
                    ],
                    "photo": next(
                        (
                            v["path"]
                            for v in photos.values()
                            if "original" in v["kind"] and v.get("available") is not False
                        ),
                        next(
                            (v["path"] for v in photos.values() if v.get("available") is not False),
                            None,
                        ),
                    ),
                }
            )
    esc = lambda x: html.escape(str(x if x is not None else "未记录"), quote=True)
    statuses = {
        "source_retained": "原始材料已留存",
        "active": "已绑定",
        "ended": "已结束",
        "completed": "已识别",
        "failed": "识别失败",
        "skipped": "未执行读数识别",
        "unassigned": "读数未归属",
        "unreadable": "未读出结构化读数",
        "queued": "等待识别",
        "running": "识别中",
        "collecting": "连拍处理中",
        "draft": "待确认",
        "confirmed": "已确认",
        "rejected": "已驳回",
        "cancelled": "已取消",
        "interrupted": "已中断",
    }
    intro = "<p>按日期与相机打开当日报告，再从事件打开结果和照片。一个测量目录对应一次拍照或一次明确的连拍。</p>"
    intro += "<article><ul><li><b>Bindings</b>：Binding.json 记录二维码、人员、仪器或场景、开始/结束和交接。</li><li><b>VideoReadings / VoicePhotoReadings / PhotoReadings</b>：一次视频观测、语音照片或明确连拍共用一个事件目录。</li><li><b>Result.json、Result02.json…</b>：每台仪器独立一个文件，直接保存 wearer_id、device_model、device_no、qr_hash、photo_time、values 六字段。搅拌器为温度和转速，天平为质量；未知值为 null。文件与仪器的对应关系在 Evidence.json 中，后续增加仪器不会改变已有对应关系。</li><li><b>Evidence.json</b>：来源照片、拍摄时间依据、绑定快照、原始 OCR、模型与规则版本、处理状态和确认记录。measurement_files 指向各台仪器的结果，不重复保存结果正文。</li><li><b>Photos / Regions</b>：原图、处理图及面板/数字裁剪；按哈希共用，同一字节只保留一份。</li><li><b>DailyReport</b>：当日事件索引，直接打开每台仪器的读数或追溯文件。</li></ul><p>未绑定且跳过识别时只留 Evidence.json 和照片，不生成伪造的 Result.json。未知仪器不能默认当作 A 或 B；已明确仪器但数字看不清时，其固定字段的 value 为 null。所有图片和结果路径相对于归档根目录；正式写入状态以 Evidence.json 的 record_scope 和 decision 为准。</p></article>"
    intro += "<article><h2>日期与相机</h2><ul>"
    for day, entries in sorted(days.items(), reverse=True):
        base = day + "/DailyReport"
        entries.sort(
            key=lambda e: aware(e["event_at"]).timestamp()
            if aware(e["event_at"])
            else float("-inf"),
            reverse=True,
        )
        views[base + "/DailyReport.json"] = raw_json(
            {"schema": "field-recognition-day/1", "timezone": "Asia/Shanghai", "events": entries}
        )
        body = '<p><a href="../../Readme.html">归档首页</a> · <a href="DailyReport.json">当日索引 JSON</a></p>'
        body += "<p>两条采集线：视频自动留证、语音三连拍。按采集时间排列；语音缺少批次回执时保留逐张证据，不能确认属于同一连拍。时间来源见追溯文件，不将接收或处理时间当作拍摄时间。</p>"
        body += "<p>点击结果查看结构化读数、来源与历史回执；原图可直接在浏览器打开。</p><table><tr><th>北京时间</th><th>事件 / 对象</th><th>人员 / 状态</th><th>结果与照片</th></tr>"
        for entry in entries:
            moment = aware(entry["event_at"])
            link = lambda target: esc(posixpath.relpath(target, base))
            body += (
                "<tr><td>"
                + esc(moment.strftime("%H:%M:%S.%f")[:-3] if moment else "拍摄时间未知")
                + "</td><td>"
                + esc(KINDS[entry["category"]])
                + "<br>"
                + esc(entry["target"])
                + "</td><td>"
                + esc(entry["operator"])
                + "<br>"
                + esc(statuses.get(entry["status"], entry["status"]))
                + '</td><td><a href="'
                + link(entry["result"])
                + '">'
                + ("绑定记录" if entry["category"] == "Bindings" else "追溯与状态")
                + "</a>"
            )
            for measurement in entry.get("measurement_files", []):
                body += (
                    ' · <a href="'
                    + link(measurement["result"])
                    + '">'
                    + esc(measurement["instrument_name"] or measurement["instrument_id"])
                    + " 读数</a>"
                )
            if entry.get("readings_readme"):
                body += (
                    ' · <a href="'
                    + link(entry["readings_readme"])
                    + '">'
                    + (
                        "未绑定读数候选（待核实）"
                        if not entry["measurement_files"]
                        else "逐张读数与合并结果"
                    )
                    + "</a>"
                )
            if entry["photo"]:
                body += ' · <a href="' + link(entry["photo"]) + '">原图</a>'
            body += "</td></tr>"
        views[base + "/DailyReport.html"] = page(day, body + "</table>")
        intro += (
            '<li><a href="'
            + esc(base + "/DailyReport.html")
            + '">'
            + esc(day)
            + "</a> · "
            + str(len(entries))
            + " 个事件</li>"
        )
    intro += '</ul></article><details><summary>维护与溯源</summary><p>.System 为隐藏维护目录：Receipts 保存不可覆盖的原始回执；Integrity 保存完整性报告；MigrationMap.json 保存旧路径映射和照片位置；SourceMaterials 保留历史未分类材料；PendingAssets 为发布过程中待归位的图片。无需在这些目录日常查数。</p><a href=".System/Audit.html">全量历史版本</a></details>'
    views["Readme.html"] = page("现场识别归档", intro)
    return views
