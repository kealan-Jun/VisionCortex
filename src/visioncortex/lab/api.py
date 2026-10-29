from __future__ import annotations

import secrets

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

from . import __version__
from .domain import Conflict, Core, Missing, clean, ident
from .media import Media
from .material_batch import MaterialBatch
from .projections import Projections
from .retention import Retention
from .rules import Rules
from .timeline_runtime import TimelineRuntime
from .timeline_commands import command_methods


def create_app(core: Core, settings):
    """Internal operations API only. Agent-specific integration remains deferred."""
    if not settings.token:
        raise ValueError("VC_LAB_API_TOKEN is required for the internal HTTP service")
    app = FastAPI(title="VisionCortex Lab internal data operations", version=__version__, docs_url=None, redoc_url=None)
    media = Media(core, settings)
    timeline = TimelineRuntime(core, settings)

    @app.middleware("http")
    async def auth(request: Request, call_next):
        if request.url.path != "/health":
            expected = "Bearer " + settings.token
            if not secrets.compare_digest(request.headers.get("Authorization", ""), expected):
                return JSONResponse({"detail": "Unauthorized"}, status_code=401)
        return await call_next(request)

    @app.exception_handler(Conflict)
    async def conflict(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.exception_handler(Missing)
    async def missing(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=404)

    @app.exception_handler(ValueError)
    async def invalid(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=422)

    @app.get("/health")
    def health():
        with core.db.transaction(readonly=True) as conn:
            conn.execute("SELECT 1")
        return {"status": "ready", "version": __version__, "agent_integration": "deferred"}

    @app.get("/internal/status")
    def status():
        with core.db.transaction(readonly=True) as conn:
            jobs = conn.execute("SELECT status,count(*) AS count FROM processing_job WHERE customer_id=%s GROUP BY status", (core.customer,)).fetchall()
            intake = conn.execute("SELECT relative_path,error_code FROM intake_item WHERE customer_id=%s AND status='needs_attention' LIMIT 50", (core.customer,)).fetchall()
        return clean(dict(jobs=jobs, timeline=timeline.status(), intake_attention=intake, retention=Retention(core, settings).status(), processing_rules=Rules().identity(), **Projections(core).status()))

    @app.get("/internal/jobs/{job_id}")
    def job(job_id: str):
        with core.db.transaction(readonly=True) as conn:
            row = conn.execute("SELECT id,capture_id,worker_kind,model_revision,status,analysis_outcome,association_status,fact_status,index_status,attempts,error_code,result FROM processing_job WHERE customer_id=%s AND id=%s", (core.customer, ident(job_id))).fetchone()
            if not row:
                raise Missing("Job outside customer scope")
            return clean(row)

    @app.get("/internal/materials/{material_id}")
    def material(material_id: str):
        return core.balance(material_id)

    @app.get("/internal/materials/source-lots/{lot_id}/summary")
    def source_lot_summary(lot_id: str):
        return MaterialBatch(core).source_lot(lot_id)

    @app.get("/internal/materials/process-batches/{batch_id}/summary")
    def process_batch_summary(batch_id: str):
        return MaterialBatch(core).process_batch(batch_id)

    @app.get("/internal/evidence/{evidence_id}")
    def evidence(evidence_id: str):
        with core.db.transaction(readonly=True) as conn:
            row = conn.execute("SELECT * FROM evidence_ref WHERE customer_id=%s AND id=%s", (core.customer, ident(evidence_id))).fetchone()
            if not row:
                raise Missing("Evidence outside customer scope")
        result = media.locate(row["media_id"])
        result.pop("path", None)
        return clean(dict(row, availability=result, media_url="/internal/media/" + str(row["media_id"])))

    @app.get("/internal/media/{media_id}")
    def content(media_id: str):
        result = media.locate(media_id)
        if result["status"] != "available":
            raise HTTPException(status_code=409 if result["status"] == "integrity_failed" else 404, detail=result["status"])
        return FileResponse(result["path"], headers={"ETag": '"' + str(media_id) + ':' + str(result["version"]) + '"', "Cache-Control": "private, no-cache"})

    @app.post("/internal/commands/{name}")
    def command(name: str, body: dict):
        methods = {"assign-camera-person": core.assign_camera_person, "reserve-equipment-catalogue": core.reserve_equipment_catalogue, "issue-equipment": core.issue_equipment, "activate-equipment": core.activate_equipment, "retire-equipment": core.retire_equipment, "issue-identity": core.issue_identity, "register": core.register, "register-material": core.register_material, "assign-tag": core.assign_tag, "bind-container": core.bind_container, "equipment-use": core.equipment_use, "operate": core.operate, "correct-operation": core.correct_operation, "correct-fact": core.correct_fact, "confirm-candidate": core.confirm_candidate, "confirm-usage": core.confirm_usage, "reserve": core.reserve}
        methods.update(command_methods(timeline))
        if name not in methods:
            raise Missing("Unknown internal command")
        try:
            return methods[name](**body)
        except TypeError as exc:
            raise HTTPException(422, "Command arguments do not match its contract") from exc

    @app.get('/internal/timeline/trace')
    def trace(start: str, end: str):
        return timeline.views.trace(start, end)

    @app.get('/internal/timeline/experiments/{run_id}/clips')
    def clips(run_id: str, include_low_priority: bool = False):
        return timeline.views.clip_index(run_id, include_low_priority=include_low_priority)

    @app.get('/internal/timeline/readouts/{session_id}')
    def readout(session_id: str):
        return timeline.views.readout(session_id)

    @app.get('/internal/timeline/reports/{day}')
    def daily_report(day: str):
        return timeline.views.snapshot(day)

    @app.get('/internal/timeline/person-days/{person_id}/{day}')
    def person_day(person_id: str, day: str):
        return timeline.views.person_day(person_id, day)

    @app.get('/internal/timeline/metrics/{day}')
    def timeline_metrics(day: str):
        return timeline.metrics.day(day)

    @app.get('/internal/timeline/tasks/{task_id}')
    def timeline_task(task_id: str):
        with core.db.transaction(readonly=True) as conn:
            row = conn.execute('SELECT id,media_id,kind,analysis_revision,status,attempts,next_attempt_at,error_code,model_revision,result FROM hj_timeline_task WHERE customer_id=%s AND id=%s', (core.customer, ident(task_id))).fetchone()
            if not row:
                raise Missing('Timeline task outside customer scope')
            windows = conn.execute('SELECT window_index,input_fingerprint,provider_fingerprint,result_fingerprint,execution_receipt FROM hj_timeline_window_result WHERE customer_id=%s AND task_id=%s ORDER BY created_at,window_index', (core.customer, ident(task_id))).fetchall()
        return clean(dict(row, checkpoints=windows))

    return app
