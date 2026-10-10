"""Opt-in scoped machine API; native queue/pipeline remain authoritative."""

import copy
import hashlib
import hmac
import json
import os
import re
import sqlite3
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from .device_day import exclusive
from .application.contracts import ApplicationError
from .schemas import RunManifest
from .sqlite_store import connection
from .storage import safe_archive_name


def checksum(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str, separators=(",", ":")).encode()
    ).hexdigest()


class Principal(BaseModel):
    model_config = ConfigDict(extra="forbid")
    client_id: str = Field(pattern=r"^[A-Za-z0-9_.-]{1,100}$")
    token_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    default_scope: str
    scopes: dict[str, list[Path]]


class MachineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    clients: list[Principal] = Field(min_length=1)
    max_pending: int = Field(default=50, ge=5, le=1000)


class Submissions:
    def __init__(self, database):
        self.path = Path(database)
        self.root = self.path.parent
        with connection(self.path) as db:
            db.execute("""CREATE TABLE IF NOT EXISTS machine_submissions(
                client TEXT,scope TEXT,key TEXT,digest TEXT,run_id TEXT UNIQUE,
                archive TEXT UNIQUE,payload TEXT NOT NULL,state TEXT NOT NULL,
                error TEXT, revision TEXT NOT NULL, PRIMARY KEY(client,scope,key))""")

    def claim(self, client, scope, key, body, archive_root, limit, revision):
        digest = checksum([body, revision])
        with connection(self.path) as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT * FROM machine_submissions WHERE client=? AND scope=? AND key=?",
                (client, scope, key),
            ).fetchone()
            if row:
                if row["digest"] != digest:
                    raise HTTPException(409, "IDEMPOTENCY_CONFLICT")
                return dict(row)
            active = db.execute("""SELECT count(*) FROM machine_submissions s
                LEFT JOIN run_jobs j ON j.run_id=s.run_id
                WHERE s.state='accepting' OR j.status IN ('queued','running')""").fetchone()[
                0
            ]
            if active >= limit:
                raise HTTPException(429, "QUEUE_FULL", headers={"Retry-After": "3"})
            base_archive = safe_archive_name(body["experiment_id"])
            archive = base_archive
            while (archive_root / archive).exists() or db.execute(
                "SELECT 1 FROM machine_submissions WHERE archive=?", (archive,)
            ).fetchone():
                suffix = (
                    "-"
                    + datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
                    + "-"
                    + uuid.uuid4().hex[:4]
                )
                # Preserve the collision suffix inside the native 120-character limit.
                archive = safe_archive_name(base_archive[: 120 - len(suffix)] + suffix)
            row = {
                "client": client,
                "scope": scope,
                "key": key,
                "digest": digest,
                "run_id": uuid.uuid4().hex[:12],
                "archive": archive,
                "payload": json.dumps(body),
                "state": "accepting",
                "error": None,
                "revision": revision,
            }
            db.execute(
                "INSERT INTO machine_submissions VALUES(:client,:scope,:key,:digest,:run_id,:archive,:payload,:state,:error,:revision)",
                row,
            )
            return row

    def find(self, client, scope, *, key=None, run_id=None, archive=None):
        column, value = (
            ("key", key)
            if key is not None
            else ("run_id", run_id)
            if run_id is not None
            else ("archive", archive)
        )
        with connection(self.path, readonly=True) as db:
            row = db.execute(
                f"SELECT * FROM machine_submissions WHERE client=? AND scope=? AND {column}=?",
                (client, scope, value),
            ).fetchone()
        if row is None:
            raise HTTPException(404, "TASK_NOT_FOUND")
        return dict(row)

    def pending(self):
        with connection(self.path, readonly=True) as db:
            return [
                dict(row)
                for row in db.execute(
                    "SELECT * FROM machine_submissions WHERE state='accepting' LIMIT 50"
                )
            ]

    def mark(self, run_id, state, error=None):
        with connection(self.path) as db:
            db.execute(
                "UPDATE machine_submissions SET state=?,error=? WHERE run_id=?",
                (state, error, run_id),
            )


def validate_paths(manifest, roots):
    resolved_roots = []
    for root in roots:
        if not root.is_absolute() or not root.is_dir() or root.is_symlink():
            raise HTTPException(503, "STORAGE_UNAVAILABLE")
        resolved_roots.append(root.resolve(strict=True))
    for view in manifest.views:
        sources = (
            [(view.video, view.timestamps_csv, view.audio)]
            if view.video
            else [
                (segment.video, segment.timestamps_csv, segment.audio)
                for segment in view.segments
            ]
        )
        for source_group in sources:
            for source in source_group:
                if source is None:
                    continue
                if not source.is_absolute() or ".." in source.parts:
                    raise HTTPException(422, "INPUT_PATH_INVALID")
                try:
                    resolved = source.resolve(strict=True)
                    if (
                        not any(resolved.is_relative_to(r) for r in resolved_roots)
                        or not resolved.is_file()
                    ):
                        raise HTTPException(403, "SOURCE_DENIED")
                    if any(p.is_symlink() for p in [source, *source.parents]):
                        raise HTTPException(422, "SOURCE_LINK_UNSUPPORTED")
                except OSError:
                    raise HTTPException(503, "STORAGE_UNAVAILABLE") from None


def create_app(config=None, native=None):
    config = config or MachineConfig.model_validate(
        json.loads(Path(os.environ["VISIONCORTEX_MACHINE_CONFIG"]).read_text())
    )
    if len({c.client_id for c in config.clients}) != len(config.clients) or len(
        {c.token_sha256 for c in config.clients}
    ) != len(config.clients):
        raise ValueError("Machine clients and credentials must be unique")
    if any(c.default_scope not in c.scopes for c in config.clients):
        raise ValueError("Client default scope must be explicitly allowed")
    if native is None:
        from .application.runtime_host import default_host
        native = default_host
    settings = native.settings()
    from .build_identity import identity

    revision = checksum({"settings": settings, "build": identity()})
    stop = threading.Event()
    submissions = None

    def roots_for(client, scope):
        owner = next((c for c in config.clients if c.client_id == client), None)
        if owner is None or scope not in owner.scopes:
            raise HTTPException(403, "SCOPE_DENIED")
        return owner.scopes[scope]

    def prepare(row):
        # Preallocated run/archive make recovery after any preparation boundary repeatable.
        with exclusive(submissions.root / ("Machine-" + row["run_id"] + ".lock")):
            if native.queue.get_job(row["run_id"]) is not None:
                submissions.mark(row["run_id"], "queued")
                return
            if row["revision"] != revision:
                submissions.mark(row["run_id"], "failed", "CONFIGURATION_CHANGED")
                return
            manifest = RunManifest.model_validate(json.loads(row["payload"]))
            validate_paths(manifest, roots_for(row["client"], row["scope"]))
            manifest = manifest.model_copy(update={"experiment_id": row["archive"]})
            from .storage import initialize_nas_archive

            destination = initialize_nas_archive(settings, row["archive"])
            native.submit_paths(
                manifest.model_dump(mode="json"),
                BackgroundTasks(),
                reserved_run_id=row["run_id"],
                reserved_archive=(row["archive"], destination),
                runtime_settings=copy.deepcopy(settings),
            )
            submissions.mark(row["run_id"], "queued")

    def recover():
        while not stop.is_set():
            try:
                for row in submissions.pending():
                    if stop.is_set():
                        break
                    try:
                        prepare(row)
                    except (HTTPException, ApplicationError) as exc:
                        if exc.status_code < 500:
                            submissions.mark(row["run_id"], "failed", "INPUT_REJECTED")
                    except (OSError, sqlite3.Error):
                        break
                    except Exception:  # noqa: BLE001 - persist a sanitized preparation failure
                        submissions.mark(row["run_id"], "failed", "PREPARATION_FAILED")
            except (OSError, sqlite3.Error):
                pass
            stop.wait(1)

    @asynccontextmanager
    async def lifespan(_):
        nonlocal submissions
        if native.maintenance_active():
            raise RuntimeError("Storage maintenance is active")
        native.initialize_queue(settings)
        submissions = Submissions(native.queue_database_path(settings))
        thread = threading.Thread(
            target=recover, daemon=True, name="machine-input-preparation"
        )
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join(timeout=5)

    app = FastAPI(title="VisionCortex Machine API", version="1.0.0", lifespan=lifespan)

    @app.middleware("http")
    async def boundary(request, call_next):
        if request.url.path == "/health/live":
            return JSONResponse({"alive": True})
        if native.maintenance_active():
            return JSONResponse({"code": "STORAGE_MAINTENANCE"}, status_code=503)
        authorization = request.headers.get("Authorization", "")
        digest = hashlib.sha256(authorization[7:].encode()).hexdigest()
        client = (
            next(
                (
                    c
                    for c in config.clients
                    if hmac.compare_digest(c.token_sha256, digest)
                ),
                None,
            )
            if authorization.startswith("Bearer ") and len(authorization) <= 1024
            else None
        )
        if client is None:
            return JSONResponse({"code": "AUTH_REQUIRED"}, status_code=401)
        scope = request.headers.get("X-Execution-Scope", client.default_scope)
        if scope not in client.scopes:
            return JSONResponse({"code": "SCOPE_DENIED"}, status_code=403)
        request.state.client, request.state.scope = client.client_id, scope
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > 65536:
                return JSONResponse({"code": "INPUT_TOO_LARGE"}, status_code=413)
        request._body = bytes(body)
        try:
            response = await call_next(request)
        except (OSError, sqlite3.Error):
            response = JSONResponse({"code": "SERVICE_UNAVAILABLE"}, status_code=503)
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(ApplicationError)
    @app.exception_handler(HTTPException)
    async def failure(_, exc):
        detail = (
            exc.detail
            if isinstance(exc.detail, str) and re.fullmatch(r"[A-Z_]{1,80}", exc.detail)
            else "REQUEST_FAILED"
        )
        return JSONResponse(
            {"code": detail}, status_code=exc.status_code, headers=exc.headers
        )

    def owned(request, **criteria):
        row = submissions.find(request.state.client, request.state.scope, **criteria)
        # No NAS access on status polling; current scope removal is already enforced.
        roots_for(row["client"], row["scope"])
        return row

    def reply(row):
        return {
            "run_id": row["run_id"],
            "state": "queued" if row["state"] == "accepting" else row["state"],
            "phase": "preparing_input" if row["state"] == "accepting" else row["state"],
            "experiment_id": row["archive"],
            "queue_persistence": "sqlite",
            "status_url": "/api/runs/" + row["run_id"],
            "error_code": row["error"],
        }

    @app.get("/api/health")
    def health():
        return {"api_ready": True, "queue_persistence": "sqlite", "role": "machine-api"}

    @app.post("/api/runs/from-paths", status_code=202)
    def submit(payload: RunManifest, request: Request):
        key = request.headers.get("Idempotency-Key") or payload.experiment_id
        if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,240}", key):
            raise HTTPException(422, "IDEMPOTENCY_KEY_INVALID")
        validate_paths(payload, roots_for(request.state.client, request.state.scope))
        row = submissions.claim(
            request.state.client,
            request.state.scope,
            key,
            payload.model_dump(mode="json"),
            Path(settings["storage"]["archive_root"]),
            config.max_pending,
            revision,
        )
        return reply(row)

    @app.get("/api/run-submissions/{key}")
    def lookup(key: str, request: Request):
        return reply(owned(request, key=key))

    @app.get("/api/runs/{run_id}")
    def status(run_id: str, request: Request):
        row = owned(request, run_id=run_id)
        if row["state"] != "queued":
            return reply(row)
        return native.get_run(run_id)

    def archive_allowed(request, archive):
        row = owned(request, archive=archive)
        validate_paths(
            RunManifest.model_validate(json.loads(row["payload"])),
            roots_for(row["client"], row["scope"]),
        )
        status = native.queue.load_run(row["run_id"]) or {}
        if status.get("state") != "completed":
            raise HTTPException(409, "RESULT_NOT_READY")

    @app.get("/api/archives/{archive}")
    def archive_detail(archive: str, request: Request):
        archive_allowed(request, archive)
        return native.archive_detail(archive, section="all", cursor=None, limit=24)

    @app.get("/api/key-events")
    def events(
        request: Request,
        archive: str,
        q: str | None = None,
        cursor: str | None = None,
        limit: int = 20,
    ):
        archive_allowed(request, archive)
        if not 1 <= limit <= 100 or q is not None and len(q) > 200:
            raise HTTPException(422, "QUERY_INVALID")
        return native.search_key_events(
            archive=archive, q=q, cursor=cursor, limit=limit
        )

    @app.get("/api/key-events/{uid}")
    def event(uid: str, archive: str, request: Request):
        archive_allowed(request, archive)
        return native.indexed_key_event(uid, archive=archive)

    @app.get("/api/evidence/{uid:path}")
    def evidence(uid: str, archive: str, request: Request):
        archive_allowed(request, archive)
        return native.indexed_evidence(uid, archive=archive)

    return app
