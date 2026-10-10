"""Request idempotency across analysis entry points; replay never reruns work."""

from contextvars import ContextVar
import hashlib
import json
from pathlib import Path
import re
import tempfile
import time
import uuid
from .sqlite_store import connection

_CURRENT = ContextVar("submission", default=None)


class Submissions:
    def __init__(self, root):
        self.path = Path(root) / "state" / "Submissions.sqlite3"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with connection(self.path) as db:
            db.execute("""CREATE TABLE IF NOT EXISTS submissions(
                key TEXT PRIMARY KEY,digest TEXT NOT NULL,owner TEXT NOT NULL,state TEXT NOT NULL,
                created REAL,task_id TEXT,response TEXT,code INTEGER)""")

    def claim(self, key, digest):
        owner = uuid.uuid4().hex
        with connection(self.path) as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM submissions WHERE key=?", (key,)).fetchone()
            if row:
                if row["digest"] != digest:
                    raise ValueError("同一幂等键不能提交不同内容")
                return dict(row), False
            db.execute(
                "INSERT INTO submissions VALUES(?,?,?,'pending',?,NULL,NULL,NULL)",
                (key, digest, owner, time.time()),
            )
            return {"key": key, "owner": owner, "state": "pending"}, True

    def bind(self, key, task_id):
        with connection(self.path) as db:
            db.execute(
                "UPDATE submissions SET task_id=? WHERE key=? AND state='pending'",
                (task_id, key),
            )

    def finish(self, key, owner, response, code):
        with connection(self.path) as db:
            db.execute(
                "UPDATE submissions SET state='completed',response=?,code=? WHERE key=? AND owner=?",
                (response, code, key, owner),
            )


def bind_task(task_id):
    context = _CURRENT.get()
    if context:
        context[0].bind(context[1], task_id)


def eligible(path):
    return (
        path
        in {
            "/api/runs",
            "/api/runs/from-paths",
            "/api/upload-sessions",
            "/api/knowledge/ask",
        }
        or bool(
            re.fullmatch(r"/api/(collections|benchmarks|nas-batches)/[^/]+/runs", path)
        )
        or bool(re.fullmatch(r"/api/upload-sessions/[^/]+/finalize", path))
        or bool(re.fullmatch(r"/api/runs/[^/]+/(retry|refresh/[^/]+)", path))
    )


async def handle(request, call_next, settings):
    from fastapi.responses import JSONResponse, Response

    if request.method != "POST" or not eligible(request.url.path):
        return await call_next(request)
    key = request.headers.get("Idempotency-Key")
    # JSON submissions deduplicate by canonical content when the caller does
    # not send a key. Multipart callers supply a key for upload retries.
    content_type = request.headers.get("content-type", "")
    # A new user retry is distinct from retrying the same HTTP request. These
    # routes already enforce queue/revision CAS; clients can add an explicit key.
    if not key and re.fullmatch(
        r"/api/runs/[^/]+/(retry|refresh/[^/]+)", request.url.path
    ):
        return await call_next(request)
    if key and (len(key) > 200 or not re.fullmatch(r"[\w.:-]+", key)):
        return JSONResponse({"detail": "无效的 Idempotency-Key"}, status_code=400)
    principal = getattr(request.state, "web_identity", {}).get("username", "local")
    original_receive = request._receive
    # Bounded memory, including legacy multipart uploads. The spooled request
    # belongs to this request only and is removed in finally.
    with tempfile.SpooledTemporaryFile(max_size=1024 * 1024) as spool:
        digest = hashlib.sha256()
        async for chunk in request.stream():
            spool.write(chunk)
            digest.update(chunk)
        spool.seek(0)
        body_digest = digest.hexdigest()
        if "application/json" in content_type and spool.seek(0, 2) <= 8 * 1024 * 1024:
            spool.seek(0)
            try:
                body_digest = hashlib.sha256(
                    json.dumps(
                        json.loads(spool.read()), sort_keys=True, separators=(",", ":")
                    ).encode()
                ).hexdigest()
            except (ValueError, UnicodeError):
                pass
        spool.seek(0)
        # Multipart boundaries change on retries. Hash semantic fields/files,
        # not transport framing; restore the owned spool before dispatch.
        if "multipart/form-data" in content_type:

            async def form_receive():
                chunk = spool.read(1024 * 1024)
                return {"type": "http.request", "body": chunk, "more_body": bool(chunk)}

            request._receive = form_receive
            request._stream_consumed = False
            parts = []
            try:
                async with request.form() as form:
                    for name, value in form.multi_items():
                        if hasattr(value, "read"):
                            checksum = hashlib.sha256()
                            while chunk := await value.read(1024 * 1024):
                                checksum.update(chunk)
                            parts.append([name, value.filename, checksum.hexdigest()])
                        else:
                            parts.append([name, str(value)])
                body_digest = hashlib.sha256(
                    json.dumps(sorted(parts)).encode()
                ).hexdigest()
            except Exception:
                return JSONResponse({"detail": "上传表单无效"}, status_code=400)
            spool.seek(0)
        body_digest = hashlib.sha256(
            json.dumps(
                [body_digest, sorted(request.query_params.multi_items())]
            ).encode()
        ).hexdigest()
        from .build_identity import identity

        config_digest = hashlib.sha256(
            json.dumps(
                {
                    k: settings.get(k)
                    for k in ("pipeline", "detection", "mllm", "device_day")
                },
                sort_keys=True,
                default=str,
            ).encode()
        ).hexdigest()
        if request.url.path == "/api/knowledge/ask":
            from .knowledge import Knowledge

            knowledge = Knowledge(settings)
            with connection(knowledge.path, readonly=True) as db:
                evidence_revision = list(
                    db.execute("SELECT id,revision,status FROM sources ORDER BY id")
                )
            config_digest += hashlib.sha256(
                json.dumps([list(r) for r in evidence_revision]).encode()
            ).hexdigest()
        auto_key = (
            body_digest
            + ":"
            + config_digest
            + ":"
            + str(identity().get("source_digest"))
        )
        scope = hashlib.sha256(
            json.dumps([principal, request.url.path, key or auto_key]).encode()
        ).hexdigest()
        store = Submissions(settings["storage"]["local_runtime_root"])
        try:
            row, acquired = store.claim(scope, body_digest)
        except ValueError as exc:
            return JSONResponse({"detail": str(exc)}, status_code=409)
        if not acquired:
            if row["state"] == "completed":
                return Response(
                    row["response"],
                    status_code=row["code"],
                    media_type="application/json",
                    headers={"Idempotency-Replayed": "true"},
                )
            return JSONResponse(
                {
                    "status": "submission_pending",
                    "run_id": row.get("task_id"),
                    "status_url": (
                        f"/api/knowledge/answers/{row['task_id']}"
                        if request.url.path == "/api/knowledge/ask"
                        else f"/api/tasks/{row['task_id']}/history"
                    )
                    if row.get("task_id")
                    else None,
                    "detail": "请求已接收，请查询原任务；不会重复启动分析",
                },
                status_code=202,
                headers={"Retry-After": "3"},
            )
        sent = False

        async def receive():
            nonlocal sent
            if sent:
                return await original_receive()
            chunk = spool.read(1024 * 1024)
            sent = not chunk
            return {"type": "http.request", "body": chunk, "more_body": not sent}

        request._receive = receive
        request._stream_consumed = False
        token = _CURRENT.set((store, scope))
        try:
            response = await call_next(request)
            body = b"".join([part async for part in response.body_iterator])
            if response.headers.get("content-type", "").startswith("application/json"):
                store.finish(
                    scope, row["owner"], body.decode("utf-8"), response.status_code
                )
            return Response(
                body,
                status_code=response.status_code,
                headers=dict(response.headers),
                background=response.background,
            )
        finally:
            _CURRENT.reset(token)
