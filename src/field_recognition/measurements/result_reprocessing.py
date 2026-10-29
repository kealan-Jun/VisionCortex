"""Explicit, idempotent recognition revisions; original jobs and ownership stay intact."""

import copy
import hashlib
import json
import uuid
from field_recognition.measurements.reading_results import result_ready

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field
from field_recognition.measurements.photo_measurements import versions, blockers
from field_recognition.api.security import require_camera, require_role, actor_name


class ReprocessRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    request_id: uuid.UUID
    actor: str = Field(min_length=1, max_length=100)
    reason: str = Field(min_length=1, max_length=1000)


# Copy input evidence only. Results/lease state/timers never leak into a new run.
INPUT_FIELDS = (
    "binding_id",
    "capture_id",
    "crop",
    "image_url",
    "image_sha256",
    "instrument",
    "operator",
    "camera_id",
    "operator_basis",
    "operator_registration",
    "scene_visit_id",
    "scene",
    "scene_basis",
    "scene_qr_verified",
    "instrument_association",
    "source",
    "frame_metadata",
    "external_photo",
    "request_trigger",
    "video_observation",
    "all_binding_snapshots",
    "binding_snapshots",
    "binding_ids",
    "instrument_candidates",
    "workbench",
    "workbenches",
    "ownership_conflicts",
    "binding_time_basis",
    "qr_matches",
    "association_status",
    "instrument_identity_basis",
    "measurement_id",
    "measurement_context",
    "record_mode",
    "record_scope",
    "capture_association",
    "attribution_status",
    "attribution_reason",
    "binding_snapshot_version",
    "binding_effective_at",
    "allowed_instrument_ids",
    "allowed_instrument_ids_basis",
    "scene_snapshots",
    "qr_scene_matches",
)


def preferred_jobs(documents):
    """A failed/running revision does not displace the last completed revision."""
    groups = {}
    for doc in documents:
        groups.setdefault(doc.get("recognition_root_job_id", doc["job_id"]), []).append(doc)
    result = []
    for group in groups.values():
        finished = [j for j in group if result_ready(j)]
        result.append(max(finished or group, key=lambda j: j.get("recognition_revision", 1)))
    return result


def install(core):
    with core["db"]() as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS recognition_revisions(
            request_id TEXT PRIMARY KEY, request_digest TEXT NOT NULL, root_job_id TEXT NOT NULL,
            revision INTEGER NOT NULL, job_id TEXT NOT NULL UNIQUE, UNIQUE(root_job_id,revision))""")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS jobs_recognition_root ON jobs(coalesce(json_extract(document,'$.recognition_root_job_id'),id),status)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS jobs_capture_lookup ON jobs(json_extract(document,'$.capture_id'))"
        )

    @core["app"].post("/api/jobs/{job_id}/reprocess", status_code=202)
    def reprocess(job_id: uuid.UUID, body: ReprocessRequest):
        require_role("admin", "reviewer")
        actor = actor_name(body.actor)
        payload = {"job_id": str(job_id), "actor": actor, "reason": body.reason}
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        with core["ocr_submit_lock"], core["db"]() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT status,document FROM jobs WHERE id=?", (str(job_id),)
            ).fetchone()
            if not row:
                raise HTTPException(404, "原识别任务不存在")
            original = json.loads(row["document"]) | {"status": row["status"]}
            require_camera(original["camera_id"])
            retry = conn.execute(
                "SELECT * FROM recognition_revisions WHERE request_id=?", (str(body.request_id),)
            ).fetchone()
            if retry:
                if retry["request_digest"] != digest:
                    raise HTTPException(409, "同一重识别请求编号不能更改目标或理由")
                return json.loads(
                    conn.execute(
                        "SELECT document FROM jobs WHERE id=?", (retry["job_id"],)
                    ).fetchone()[0]
                )
            if original["status"] not in {"completed", "failed"}:
                raise HTTPException(409, "仅已完成或失败的任务可以创建识别新版本")
            root = original.get("recognition_root_job_id", str(job_id))
            last = conn.execute(
                "SELECT revision,job_id FROM recognition_revisions WHERE root_job_id=? ORDER BY revision DESC LIMIT 1",
                (root,),
            ).fetchone()
            replaced_job_id = str(job_id)
            if last and last["job_id"] != str(job_id):
                latest = json.loads(
                    conn.execute(
                        "SELECT document FROM jobs WHERE id=?", (last["job_id"],)
                    ).fetchone()[0]
                )
                if latest["status"] != "failed":
                    raise HTTPException(
                        409,
                        {
                            "message": "请使用最新识别版本；正在处理的版本不能重复提交",
                            "job_id": last["job_id"],
                        },
                    )
                replaced_job_id = last["job_id"]
            path = core["DATA"] / "Images" / f"{original['capture_id']}.png"
            try:
                if hashlib.sha256(path.read_bytes()).hexdigest() != original["image_sha256"]:
                    raise ValueError("hash")
            except (OSError, ValueError):
                raise HTTPException(409, "原照片缺失或哈希不一致，不能重识别") from None
            document = {k: copy.deepcopy(original[k]) for k in INPUT_FIELDS if k in original}
            revision = (last["revision"] if last else 1) + 1
            document.update(
                job_id=str(uuid.uuid4()),
                status="queued",
                phase="recognition_revision",
                submitted_at=core["now"](),
                resume_pending=True,
                recognition_versions=versions(),
                recognition_root_job_id=root,
                recognition_revision=revision,
                supersedes_job_id=str(job_id),
                reprocessing={
                    "request_id": str(body.request_id),
                    "actor": actor,
                    "reason": body.reason,
                    "at": core["now"](),
                },
                timing={"is_reprocessing": True, "original_job_id": str(job_id)},
            )
            document["field_rules"] = {
                b["instrument"]["id"]: (core["get_instrument"](b["instrument"]["id"]) or {}).get(
                    "measurement_ranges", {}
                )
                for b in document.get(
                    "all_binding_snapshots", document.get("binding_snapshots", [])
                )
            }
            mid = original.get("measurement_id")
            if mid:
                group = json.loads(
                    conn.execute(
                        "SELECT document FROM photo_measurements WHERE id=?", (mid,)
                    ).fetchone()[0]
                )
                if group["status"] == "collecting":
                    raise HTTPException(409, "测量仍在收集或重识别，请等待完成")
                previous = {
                    k: copy.deepcopy(v) for k, v in group.items() if k != "revision_history"
                }
                group.setdefault("revision_history", []).append(previous)
                group.setdefault("superseded_job_ids", []).append(replaced_job_id)
                group["job_ids"] = [
                    document["job_id"] if j == replaced_job_id else j for j in group["job_ids"]
                ]
                if document["job_id"] not in group["job_ids"]:
                    raise HTTPException(409, "原任务已不是该测量的当前版本")
                if group["status"] == "confirmed":
                    group["supersedes_record_id"] = group.get("record_id", mid)
                    group["record_revision"] = group.get("record_revision", 1) + 1
                group.update(
                    status="collecting",
                    record_scope="draft" if group["record_mode"] == "production" else "test_only",
                    revision=group["revision"] + 1,
                    updated_at=core["now"](),
                    fields=[],
                    sources=[],
                )
                document["record_scope"] = group["record_scope"]
                group["blockers"] = blockers(group)
                conn.execute(
                    "UPDATE photo_measurements SET status=?,document=? WHERE id=?",
                    ("collecting", json.dumps(group), mid),
                )
            if core["RUNTIME_ENABLED"]:
                from field_recognition.runtime.task_queue import QueueFull

                try:
                    document = core["task_queue"].enqueue(document, conn)
                except QueueFull as exc:
                    raise HTTPException(429, str(exc)) from None
            else:
                if (
                    conn.execute(
                        "SELECT count(*) FROM jobs WHERE status IN ('queued','running')"
                    ).fetchone()[0]
                    >= 4
                ):
                    raise HTTPException(429, "识别队列已满")
                conn.execute(
                    "INSERT INTO jobs(id,status,document) VALUES(?,?,?)",
                    (document["job_id"], "queued", json.dumps(document)),
                )
            conn.execute(
                "INSERT INTO recognition_revisions VALUES(?,?,?,?,?)",
                (str(body.request_id), digest, root, revision, document["job_id"]),
            )
        if not core["RUNTIME_ENABLED"]:
            core["readout_pool"].submit(core["run_ocr"], copy.deepcopy(document))
        return document

    @core["app"].get("/api/jobs/{job_id}/versions")
    def history(job_id: uuid.UUID):
        with core["db"]() as conn:
            row = conn.execute("SELECT document FROM jobs WHERE id=?", (str(job_id),)).fetchone()
            if not row:
                raise HTTPException(404, "任务不存在")
            doc = json.loads(row[0])
            require_camera(doc["camera_id"])
            root = doc.get("recognition_root_job_id", str(job_id))
            ids = [root] + [
                r[0]
                for r in conn.execute(
                    "SELECT job_id FROM recognition_revisions WHERE root_job_id=? ORDER BY revision",
                    (root,),
                )
            ]
            docs = [
                json.loads(
                    conn.execute("SELECT document FROM jobs WHERE id=?", (jid,)).fetchone()[0]
                )
                for jid in ids
            ]
        return {
            "root_job_id": root,
            "current_job_id": preferred_jobs(docs)[0]["job_id"],
            "items": [
                {
                    k: j.get(k)
                    for k in (
                        "job_id",
                        "status",
                        "recognition_revision",
                        "supersedes_job_id",
                        "reprocessing",
                        "recognition_versions",
                        "finished_at",
                    )
                }
                for j in docs
            ],
        }
