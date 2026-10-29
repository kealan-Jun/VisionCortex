"""Versioned aliases and operational routes attached to the actual application."""

import asyncio
import json
import time
from pathlib import Path

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse, PlainTextResponse
from fastapi.routing import APIRoute

from field_recognition.api.security import current, require_role, visible
from field_recognition.runtime.task_queue import TaskQueue
from pydantic import BaseModel, Field, ConfigDict
from field_recognition.capture.capture_adapter import CaptureEvent
from field_recognition.bindings.instrument_config import MeasurementDefinition


class RegisterCamera(BaseModel):
    model_config = ConfigDict(extra="forbid")
    camera_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,100}$")
    display_name: str = Field(min_length=1, max_length=100)
    receiver_url: str | None = None
    nas_photo_root: str | None = None


class CreateType(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=100)
    measurements: list[MeasurementDefinition] = Field(min_length=1, max_length=20)


def install(core):
    app = core["app"]

    @app.get("/health/live")
    def live():
        return {"status": "alive"}

    @app.get("/health/ready")
    @app.get("/health")
    def ready():
        try:
            with core["db"]() as conn:
                conn.execute("SELECT 1").fetchone()
                pending = conn.execute(
                    "SELECT count(*) FROM jobs WHERE status IN ('queued','running')"
                ).fetchone()[0]
                workers = {}
                if core["RUNTIME_ENABLED"]:
                    workers = {
                        r["name"]: (
                            time.time() - r["updated_at"] < 30
                            and json.loads(r["document"]).get("status") != "stopped"
                        )
                        for r in conn.execute("SELECT * FROM runtime_status")
                    }
                    required = {"ocr", "camera-supervisor"}
                    required.update(
                        "camera:" + r[0]
                        for r in conn.execute("""SELECT r.camera_id FROM camera_registry r
                        JOIN camera_active_users c USING(camera_id) JOIN users u ON c.user_id=u.id
                        WHERE r.enabled=1 AND u.disabled=0""")
                    )
                    if any(not workers.get(name) for name in required):
                        return JSONResponse(
                            {"status": "unavailable", "database": "ready", "workers": workers},
                            status_code=503,
                        )
            return {
                "status": "ready"
                if not core["RUNTIME_ENABLED"] or workers.get("archive")
                else "degraded",
                "database": "ready",
                "queued_or_running": pending,
            }
        except Exception:
            return JSONResponse(
                {"status": "unavailable", "database": "unavailable"}, status_code=503
            )

    @app.get("/api/admin/status")
    def status():
        require_role("admin")
        with core["db"]() as conn:
            runtimes = (
                [dict(r) for r in conn.execute("SELECT * FROM runtime_status")]
                if core["RUNTIME_ENABLED"]
                else []
            )
        ocr = next(
            (
                json.loads(r["document"]).get("ocr")
                for r in runtimes
                if r["name"] == "ocr" and time.time() - r["updated_at"] < 30
            ),
            None,
        )
        return {
            "version": Path(core["BASE"] / "VERSION").read_text().strip()
            if (core["BASE"] / "VERSION").exists()
            else "development",
            "record_mode": core["RECORD_MODE"],
            "role": core["SERVICE_ROLE"],
            "queue": TaskQueue(core["database"]).get_queue_stats()
            if core["RUNTIME_ENABLED"]
            else {},
            "components": runtimes,
            "ocr": ocr or core["ocr_state"],
            "archive": core["archive_store"].snapshot(),
        }

    @app.post("/api/admin/jobs/{job_id}/replay")
    def replay(job_id: str, body: dict):
        require_role("admin")
        actor = current().user_id if current() else body.get("actor")
        if not isinstance(actor, str) or not actor.strip() or len(actor) > 200:
            raise HTTPException(422, "请填写本次重放的操作人")
        return TaskQueue(core["database"]).replay(job_id, actor.strip(), body.get("reason", ""))

    @app.get("/api/version")
    def version():
        return {
            "api_version": "v1",
            "service": "FieldRecognition",
            "record_mode": core["RECORD_MODE"],
        }

    @app.get("/api/events")
    async def events(request: Request):
        async def stream():
            last = request.headers.get("last-event-id")
            while not await request.is_disconnected():
                # Database-backed IDs survive reconnects and API process changes.
                with core["db"]() as conn:
                    seq = conn.execute(
                        "SELECT coalesce(max(seq),0) FROM archive_outbox"
                    ).fetchone()[0]
                if str(seq) != last:
                    yield f'id: {seq}\nevent: changed\ndata: {{"refresh":true}}\n\n'
                    last = str(seq)
                else:
                    yield ": heartbeat\n\n"
                await asyncio.sleep(2)

        return StreamingResponse(
            stream(), media_type="text/event-stream", headers={"Cache-Control": "no-store"}
        )

    @app.get("/metrics")
    def metrics():
        require_role("admin")
        with core["db"]() as conn:
            jobs = dict(conn.execute("SELECT status,count(*) FROM jobs GROUP BY status"))
            pending = conn.execute(
                "SELECT count(*) FROM archive_outbox WHERE archived_at IS NULL"
            ).fetchone()[0]
        lines = ["# TYPE field_jobs gauge"] + [
            f'field_jobs{{status="{state}"}} {jobs.get(state, 0)}'
            for state in ("queued", "running", "completed", "failed", "interrupted")
        ]
        lines += ["# TYPE field_archive_pending gauge", f"field_archive_pending {pending}"]
        return PlainTextResponse("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")

    @app.get("/api/cameras")
    def cameras():
        with core["db"]() as conn:
            rows = (
                conn.execute(
                    "SELECT camera_id,display_name,enabled FROM camera_registry"
                ).fetchall()
                if core["RUNTIME_ENABLED"]
                else []
            )
        return {"items": [dict(r) for r in rows if visible(dict(r))]}

    @app.post("/api/admin/cameras")
    def register_camera(body: RegisterCamera):
        from field_recognition.bindings.camera_registry import CameraRegistry

        require_role("admin")
        return CameraRegistry(core["database"]).register_camera(**body.model_dump()).model_dump()

    @app.get("/api/admin/instrument-types")
    def instrument_types():
        from field_recognition.bindings.instrument_config import InstrumentConfig

        require_role("admin")
        return {
            "items": [
                t.model_dump() for t in InstrumentConfig(core["database"]).list_instrument_types()
            ]
        }

    @app.post("/api/admin/instrument-types")
    def create_type(body: CreateType):
        from field_recognition.bindings.instrument_config import (
            InstrumentConfig,
        )

        require_role("admin")
        return (
            InstrumentConfig(core["database"])
            .create_instrument_type(body.name, body.measurements)
            .model_dump()
        )

    @app.put("/api/admin/instruments/{instrument_id}/type")
    def assign_type(instrument_id: str, body: dict):
        require_role("admin")
        with core["database"].transaction("IMMEDIATE") as conn:
            if not conn.execute(
                "SELECT 1 FROM instrument_types WHERE id=?", (body.get("type_id"),)
            ).fetchone():
                raise HTTPException(422, "仪器类型不存在")
            row = conn.execute(
                "UPDATE instruments SET type_id=? WHERE id=?", (body["type_id"], instrument_id)
            )
            if not row.rowcount:
                raise HTTPException(404, "仪器不存在")
        return core["get_instrument"](instrument_id)

    @app.post("/api/capture-events")
    def ingest(body: CaptureEvent):
        from field_recognition.capture.capture_adapter import ingest_event

        return ingest_event(core, body)

    # Aliases execute the SAME endpoint, dependencies and security boundary.
    for route in list(app.routes):
        if (
            isinstance(route, APIRoute)
            and route.path.startswith("/api/")
            and not route.path.startswith("/api/v1/")
        ):
            app.add_api_route(
                "/api/v1/" + route.path[len("/api/") :],
                route.endpoint,
                methods=route.methods,
                response_model=route.response_model,
                status_code=route.status_code,
                name="v1_" + route.name,
                dependencies=route.dependencies,
            )
