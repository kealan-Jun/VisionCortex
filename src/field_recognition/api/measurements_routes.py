"""HTTP adapters for evidence-backed measurements actions."""

import hashlib
import json
import uuid
from fastapi import HTTPException
from fastapi.responses import FileResponse
from field_recognition.measurements.photo_measurements import Decision, Revision, Rejection
from field_recognition.measurements.measurement_actions import MeasurementActions


def install(core):
    actions = MeasurementActions(core)
    with core["db"]() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS photo_measurements(
                id TEXT PRIMARY KEY, camera TEXT NOT NULL, burst_key TEXT, status TEXT NOT NULL, document TEXT NOT NULL,
                UNIQUE(camera,burst_key));
            CREATE TABLE IF NOT EXISTS experiment_records(id TEXT PRIMARY KEY,document TEXT NOT NULL);
            CREATE TRIGGER IF NOT EXISTS experiment_records_no_update BEFORE UPDATE ON experiment_records
                BEGIN SELECT RAISE(ABORT,'Confirmed records are immutable'); END;
            CREATE TRIGGER IF NOT EXISTS experiment_records_no_delete BEFORE DELETE ON experiment_records
                BEGIN SELECT RAISE(ABORT,'Confirmed records are immutable'); END;
        """)

    @core["app"].get("/photo-measurements")
    def page():
        return FileResponse(
            core["BASE"] / "static" / "photo-measurements.html",
            headers={"Cache-Control": "no-store"},
        )

    @core["app"].get("/api/photo-measurements")
    def listing(limit: int = 30):
        from field_recognition.api.security import visible

        if not 1 <= limit <= 100:
            raise HTTPException(422, "数量需为 1–100")
        with core["db"]() as conn:
            rows = conn.execute(
                "SELECT document FROM photo_measurements ORDER BY rowid DESC LIMIT ?", (limit,)
            ).fetchall()
        return {
            "record_mode": core["RECORD_MODE"],
            "items": [json.loads(r[0]) for r in rows if visible(json.loads(r[0]))],
        }

    @core["app"].get("/api/photo-measurements/{mid}")
    def detail(mid: uuid.UUID):
        return actions.detail(mid)

    @core["app"].get("/api/captures/{capture_id}/original")
    def original(capture_id: uuid.UUID):
        with core["db"]() as conn:
            row = conn.execute(
                "SELECT document FROM scans WHERE id=?", (str(capture_id),)
            ).fetchone()
        capture = json.loads(row[0]) if row else {}
        relative = capture.get("original_blob")
        if not relative:
            raise HTTPException(404, "这条历史记录没有原始字节副本")
        path = (core["DATA"] / relative).resolve()
        try:
            path.relative_to((core["DATA"] / "Originals").resolve())
            if hashlib.sha256(path.read_bytes()).hexdigest() != capture["source_sha256"]:
                raise ValueError("hash mismatch")
        except (OSError, KeyError, ValueError):
            raise HTTPException(409, "原件不可用或哈希不匹配") from None
        return FileResponse(path, headers={"Cache-Control": "no-store"})

    @core["app"].post("/api/photo-measurements/{mid}/revisions")
    def revise(mid: uuid.UUID, body: Revision):
        return actions.revise(mid, body)

    @core["app"].post("/api/photo-measurements/{mid}/revalidate")
    def revalidate(mid: uuid.UUID, body: Decision):
        return actions.revalidate(mid, body)

    @core["app"].post("/api/photo-measurements/{mid}/confirm")
    def confirm(mid: uuid.UUID, body: Decision):
        return actions.confirm(mid, body)

    @core["app"].post("/api/photo-measurements/{mid}/reject")
    def reject(mid: uuid.UUID, body: Rejection):
        return actions.reject(mid, body)

    @core["app"].get("/api/experiment-records")
    def records(limit: int = 30):
        from field_recognition.api.security import require_role

        require_role("admin", "reviewer")
        if not 1 <= limit <= 100:
            raise HTTPException(422, "数量需为 1–100")
        with core["db"]() as conn:
            return {
                "items": [
                    json.loads(row[0])
                    for row in conn.execute(
                        "SELECT document FROM experiment_records AS r WHERE NOT EXISTS (SELECT 1 FROM experiment_records AS n WHERE json_extract(n.document,'$.measurement_id')=json_extract(r.document,'$.measurement_id') AND n.rowid>r.rowid) ORDER BY rowid DESC LIMIT ?",
                        (limit,),
                    )
                ]
            }

    @core["app"].get("/api/experiment-records/{record_id}")
    def read_record(record_id: uuid.UUID):
        with core["db"]() as conn:
            row = conn.execute(
                "SELECT document FROM experiment_records WHERE id=?", (str(record_id),)
            ).fetchone()
        if not row:
            raise HTTPException(404, "已确认实验记录不存在")
        return json.loads(row[0])

    core["measurement_actions"] = {
        "detail": actions.detail,
        "revise": actions.revise,
        "confirm": actions.confirm,
        "reject": actions.reject,
    }
