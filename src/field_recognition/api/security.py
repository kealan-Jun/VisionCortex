"""Authentication boundary shared by HTTP, browser and Agent tools."""

import contextvars
import hashlib
import json
import os
import re
import secrets
import time
import uuid
from dataclasses import dataclass

from fastapi import HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.concurrency import run_in_threadpool
from pydantic import ValidationError, BaseModel, Field, ConfigDict
from fastapi.exceptions import RequestValidationError
from typing import Literal

from field_recognition.runtime.auth import AuthService

identity = contextvars.ContextVar("field_identity", default=None)


class BoundedBody:
    """Bound all request bodies before JSON/multipart parsers allocate memory."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        chunks = []
        size = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            size += len(message.get("body", b""))
            if size > 32 * 1024 * 1024:
                return await JSONResponse(
                    {"error": {"code": "request_too_large", "message": "请求超过 32 MB"}},
                    status_code=413,
                )(scope, receive, send)
            chunks.append(message)
            if not message.get("more_body"):
                break

        async def replay():
            return chunks.pop(0) if chunks else await receive()

        await self.app(scope, replay, send)


class CreateUser(BaseModel):
    model_config = ConfigDict(extra="forbid")
    username: str = Field(min_length=1, max_length=100)
    display_name: str = Field(min_length=1, max_length=100)
    role: Literal["admin", "operator", "reviewer"]
    password: str | None = Field(default=None, min_length=12, max_length=1000)


@dataclass(frozen=True)
class Principal:
    user_id: str
    display_name: str
    role: str
    cameras: frozenset
    device: bool = False


def current():
    return identity.get()


def require_camera(camera_id):
    actor = current()
    if actor and actor.role not in {"admin", "reviewer"} and camera_id not in actor.cameras:
        raise HTTPException(403, "无权访问此相机的材料或操作")
    if actor and actor.device and camera_id not in actor.cameras:
        raise HTTPException(403, "相机凭证与来源不一致")


def require_role(*roles):
    actor = current()
    if actor and (actor.device or actor.role not in roles):
        raise HTTPException(403, "当前账号无此操作权限")


def actor_name(declared=None):
    actor = current()
    return actor.display_name if actor else declared


def visible(document):
    actor = current()
    if not actor or actor.role in {"admin", "reviewer"}:
        return True
    if not isinstance(document, dict):
        return True
    camera = document.get("camera_key") or document.get("camera_id") or document.get("camera")
    if document.get("configured") and document.get("mode") and document.get("id"):
        camera = document["id"]  # Receiver status also contains the local channel cam01.
    elif document.get("sender_id") and document.get("camera_id") and not document.get("camera_key"):
        camera = str(document["sender_id"]) + "_" + str(document["camera_id"])
    return not isinstance(camera, str) or not camera or camera in actor.cameras


def filter_payload(payload):
    if isinstance(payload, list):
        return [filter_payload(x) for x in payload if visible(x)]
    if isinstance(payload, dict):
        return {k: filter_payload(v) for k, v in payload.items()} if visible(payload) else None
    return payload


class SecurityMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, *, core, enabled):
        super().__init__(app)
        self.core = core
        self.enabled = enabled

    async def dispatch(self, request, call_next):
        request_id = request.headers.get("x-request-id", "")
        if not re.fullmatch(r"[A-Za-z0-9._-]{1,80}", request_id):
            request_id = uuid.uuid4().hex
        request.state.request_id = request_id
        path = request.url.path
        if path.startswith("/api/v1/"):
            path = "/api/" + path[len("/api/v1/") :]
        public = path in {"/login", "/api/auth/login", "/health/live", "/health/ready", "/health"}
        principal = None
        try:
            if self.enabled and not public:
                auth = self.core["auth_service"]
                device = request.headers.get("x-camera-id")
                authorization = request.headers.get("authorization", "")
                if authorization.startswith("Device "):
                    if not device:
                        raise HTTPException(401, "缺少相机编号")
                    await run_in_threadpool(auth.verify_device, device, authorization[7:])
                    with self.core["db"]() as conn:
                        row = conn.execute(
                            """SELECT u.* FROM users u JOIN camera_active_users a ON a.user_id=u.id
                                            JOIN camera_registry r ON r.camera_id=a.camera_id
                                            WHERE a.camera_id=? AND u.disabled=0 AND r.enabled=1""",
                            (device,),
                        ).fetchone()
                    if not row:
                        raise HTTPException(403, "相机尚未分配给有效账号")
                    principal = Principal(
                        row["id"], row["display_name"], "operator", frozenset([device]), True
                    )
                else:
                    token = request.headers.get("x-session-id") or (
                        authorization[7:]
                        if authorization.startswith("Bearer ")
                        else request.cookies.get("field_session")
                    )
                    user = await run_in_threadpool(auth.verify_session, token)
                    with self.core["db"]() as conn:
                        cameras = frozenset(
                            r[0]
                            for r in conn.execute(
                                "SELECT camera_id FROM camera_users WHERE user_id=?", (user.id,)
                            )
                        )
                    principal = Principal(user.id, user.display_name, user.role, cameras)
                    # Cookies are HTTP-only; unsafe browser requests require same-origin or CSRF proof.
                    if (
                        request.method not in {"GET", "HEAD", "OPTIONS"}
                        and request.cookies.get("field_session")
                        and not authorization
                        and not request.headers.get("x-session-id")
                    ):
                        origin = request.headers.get("origin")
                        if origin and origin.rstrip("/") != str(request.base_url).rstrip("/"):
                            raise HTTPException(403, "请求来源不匹配")
                        csrf = request.headers.get("x-csrf-token")
                        if not csrf or not secrets.compare_digest(
                            csrf, hashlib.sha256(token.encode()).hexdigest()
                        ):
                            raise HTTPException(403, "缺少有效请求校验令牌")
                request.state.principal = principal
            from field_recognition.runtime.runtime_rpc import selected_camera

            target = request.headers.get("x-camera-id") or request.query_params.get("camera_id")
            if not target and principal and principal.role == "operator":
                target = next(iter(sorted(principal.cameras)), None)
            camera_marker = selected_camera.set(target)
            marker = identity.set(principal)
            try:
                if self.enabled and not public:
                    if target:
                        require_camera(target)
                    await self.authorize(request, path, principal)
                response = await call_next(request)
            finally:
                identity.reset(marker)
                selected_camera.reset(camera_marker)
        except HTTPException as exc:
            if (
                exc.status_code == 401
                and request.method == "GET"
                and "text/html" in request.headers.get("accept", "")
            ):
                response = RedirectResponse("/login", status_code=303)
            else:
                response = JSONResponse(
                    {
                        "error": {
                            "code": "auth_required"
                            if exc.status_code == 401
                            else "permission_denied",
                            "message": exc.detail,
                        },
                        "request_id": request_id,
                    },
                    status_code=exc.status_code,
                )
        response.headers["x-request-id"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        if path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    async def authorize(self, request, path, actor):
        if (
            path.startswith("/api/admin")
            or path == "/metrics"
            or (path.startswith("/api/instruments") and request.method != "GET")
        ):
            require_role("admin")
        if path.startswith("/api/archive/") or path == "/api/export":
            require_role("admin", "reviewer")
        if re.search(r"/photo-measurements/[^/]+/(confirm|revisions|reject)$", path):
            require_role("admin", "reviewer")
        if actor.device and (
            path.startswith("/api/auth")
            or "/handoffs" in path
            or path.startswith("/api/automation")
        ):
            raise HTTPException(403, "相机凭证不能用于人员管理或确认交接")
        if (
            actor.role == "reviewer"
            and request.method not in {"GET", "HEAD"}
            and not ("/photo-measurements/" in path or path == "/api/auth/logout")
        ):
            raise HTTPException(403, "审核员不能控制采集或修改仪器登记")
        payload = {}
        if request.headers.get("content-type", "").startswith("application/json"):
            raw = await request.body()
            if len(raw) > 32 * 1024 * 1024:
                raise HTTPException(413, "请求过大")
            try:
                payload = json.loads(raw) if raw else {}
            except ValueError:
                raise HTTPException(422, "JSON 无效") from None
            if not isinstance(payload, dict):
                raise HTTPException(422, "请求必须为对象")
        self.check_references(
            payload, handoff=path == "/api/handoffs" or path.endswith("/request_device_handoff")
        )
        routes = {
            "jobs": "jobs",
            "images": "scans",
            "captures": "scans",
            "bindings": "bindings",
            "photo-measurements": "photo_measurements",
            "experiment-records": "experiment_records",
        }
        match = re.match(r"/api/([^/]+)/([^/]+)", path)
        if match and match[1] in routes:
            with self.core["db"]() as conn:
                row = conn.execute(
                    "SELECT document FROM " + routes[match[1]] + " WHERE id=?", (match[2],)
                ).fetchone()
                if not row and match[1] == "images":
                    row = conn.execute(
                        "SELECT document FROM jobs WHERE id=?", (match[2],)
                    ).fetchone()
                    if not row:
                        row = conn.execute(
                            "SELECT document FROM jobs WHERE EXISTS(SELECT 1 FROM json_tree(jobs.document) WHERE type='text' AND value=?) LIMIT 1",
                            ("/api/images/" + match[2],),
                        ).fetchone()
            if row:
                document = json.loads(row[0])
                require_camera(document.get("camera_id") or document.get("camera"))
            elif match[1] == "images" and actor.role not in {"admin", "reviewer"}:
                raise HTTPException(403, "图片归属无法验证")
        if (
            path.startswith("/api/camera/")
            or path.startswith("/api/automation")
            or path.startswith("/api/video-ocr")
        ):
            camera = self.core.get("receiver_camera")
            config = self.core.get("config")
            target = (
                camera.target
                if camera
                else (config.camera_id if config else os.environ.get("FIELD_CAMERA_ID"))
            )
            if target:
                require_camera(target)
        # A legacy actor/name is display metadata; it must match authenticated identity.
        for key in ("actor", "operator", "recipient_operator"):
            if payload.get(key) and payload[key] != actor.display_name:
                raise HTTPException(403, "操作人必须与登录账号一致")
        if payload.get("wearer_id") and payload["wearer_id"] != actor.user_id:
            raise HTTPException(403, "佩戴人编号与账号不一致")

    def check_references(self, payload, *, handoff=False):
        references = {
            "capture_id": "scans",
            "scan_id": "scans",
            "recipient_scan_id": "scans",
            "binding_id": "bindings",
            "job_id": "jobs",
            "measurement_id": "photo_measurements",
        }
        for key, value in payload.items():
            if key == "camera_id" and isinstance(value, str):
                require_camera(value)
            if (
                key in references
                and isinstance(value, str)
                and not (handoff and key == "binding_id")
            ):
                with self.core["db"]() as conn:
                    row = conn.execute(
                        "SELECT document FROM " + references[key] + " WHERE id=?", (value,)
                    ).fetchone()
                if row:
                    document = json.loads(row[0])
                    require_camera(document.get("camera_id") or document.get("camera"))
            if isinstance(value, dict):
                self.check_references(value)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        self.check_references(item)


def install(core, enabled):
    app = core["app"]
    config = core.get("config")
    core["auth_service"] = (
        AuthService(
            core["database"], session_duration_hours=config.session_duration_hours if config else 8
        )
        if enabled
        else None
    )
    app.add_middleware(SecurityMiddleware, core=core, enabled=enabled)
    app.add_middleware(BoundedBody)

    async def invalid(request, exc):
        # Pydantic's default errors include input values (photos/passwords).
        return JSONResponse(
            {
                "error": {"code": "invalid_request", "message": "请求字段无效"},
                "request_id": getattr(request.state, "request_id", None),
            },
            status_code=422,
        )

    app.add_exception_handler(RequestValidationError, invalid)
    app.add_exception_handler(ValidationError, invalid)
    app.add_exception_handler(ValueError, invalid)

    @app.get("/login", include_in_schema=False)
    def login_page():
        if not enabled:
            return RedirectResponse("/", status_code=303, headers={"Cache-Control": "no-store"})
        return FileResponse(
            core["BASE"] / "static" / "login.html", headers={"Cache-Control": "no-store"}
        )

    @app.post("/api/auth/login")
    def login(body: dict, request: Request):
        if not enabled:
            raise HTTPException(404, "登录功能已关闭，请直接打开工作台")
        username = body.get("username", "")
        password = body.get("password", "")
        if (
            not isinstance(username, str)
            or not isinstance(password, str)
            or len(username) > 100
            or len(password) > 1000
        ):
            raise HTTPException(422, "登录参数无效")
        # Persistent throttling applies across API processes and service restarts.
        key = hashlib.sha256(
            (username + "|" + (request.client.host if request.client else "")).encode()
        ).hexdigest()
        now = time.time()
        with core["database"].transaction("IMMEDIATE") as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS login_attempts(key TEXT PRIMARY KEY,started REAL,count INTEGER)"
            )
            row = conn.execute("SELECT * FROM login_attempts WHERE key=?", (key,)).fetchone()
            if row and now - row["started"] < 300 and row["count"] >= 10:
                raise HTTPException(429, "登录尝试过多，请稍后重试")
            conn.execute(
                "INSERT OR REPLACE INTO login_attempts VALUES(?,?,?)",
                (
                    key,
                    row["started"] if row and now - row["started"] < 300 else now,
                    row["count"] + 1 if row and now - row["started"] < 300 else 1,
                ),
            )
        session, user = core["auth_service"].login(username, password)
        with core["database"].transaction() as conn:
            conn.execute("DELETE FROM login_attempts WHERE key=?", (key,))
        response = JSONResponse(
            {
                "user": user.model_dump(),
                "csrf_token": hashlib.sha256(session.id.encode()).hexdigest(),
            }
        )
        response.set_cookie(
            "field_session",
            session.id,
            httponly=True,
            samesite="strict",
            secure=request.url.scheme == "https",
            max_age=core["auth_service"].session_duration_hours * 3600,
        )
        return response

    @app.get("/api/auth/me")
    def me(request: Request):
        if not enabled:
            return {"auth_enabled": False, "user_id": None, "csrf_token": None}
        p = current()
        if not p:
            raise HTTPException(401, "请登录")
        token = request.cookies.get("field_session", "")
        return {
            "auth_enabled": True,
            "user_id": p.user_id,
            "display_name": p.display_name,
            "role": p.role,
            "camera_ids": sorted(p.cameras),
            "csrf_token": hashlib.sha256(token.encode()).hexdigest() if token else None,
        }

    @app.post("/api/auth/logout")
    def logout(request: Request):
        token = (
            request.cookies.get("field_session")
            or request.headers.get("x-session-id")
            or request.headers.get("authorization", "").removeprefix("Bearer ")
        )
        if enabled:
            core["auth_service"].logout(token)
        response = JSONResponse({"logged_out": True})
        response.delete_cookie("field_session")
        return response

    def require_account_admin():
        if not enabled:
            raise HTTPException(404, "登录功能已关闭")
        require_role("admin")

    @app.post("/api/admin/users")
    def create_user(body: CreateUser):
        require_account_admin()
        return (
            core["auth_service"]
            .create_user(
                body.username, body.display_name, body.role, body.password, current().user_id
            )
            .model_dump()
        )

    @app.put("/api/admin/users/{user_id}/disabled")
    def disable_user(user_id: str, disabled: bool):
        require_account_admin()
        with core["database"].transaction("IMMEDIATE") as conn:
            row = conn.execute("SELECT role FROM users WHERE id=?", (user_id,)).fetchone()
            if not row:
                raise HTTPException(404, "账号不存在")
            if (
                disabled
                and row[0] == "admin"
                and conn.execute(
                    "SELECT count(*) FROM users WHERE role='admin' AND disabled=0 AND id<>?",
                    (user_id,),
                ).fetchone()[0]
                == 0
            ):
                raise HTTPException(409, "不能禁用最后一个管理员")
            conn.execute("UPDATE users SET disabled=? WHERE id=?", (int(disabled), user_id))
            if disabled:
                conn.execute("DELETE FROM sessions WHERE user_id=?", (user_id,))
        return {"user_id": user_id, "disabled": disabled}

    @app.put("/api/admin/cameras/{camera_id}/owner")
    def assign(camera_id: str, body: dict):
        require_role("admin")
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", camera_id):
            raise HTTPException(422, "相机编号无效")
        with core["database"].transaction("IMMEDIATE") as conn:
            from field_recognition.bindings.binding_operator import (
                request_fingerprint,
                replay_request,
                remember_request,
                check_revision,
                camera_revision,
            )

            fingerprint = request_fingerprint(
                {
                    "operation": "assign_owner",
                    "camera_id": camera_id,
                    "body": body,
                    "principal": current().user_id if current() else None,
                }
            )
            replay = replay_request(conn, body.get("request_id"), fingerprint)
            if replay is not None:
                return replay
            check_revision(conn, camera_id, body.get("expected_revision"))
            user = conn.execute(
                "SELECT id FROM users WHERE id=? AND disabled=0", (body.get("user_id"),)
            ).fetchone()
            if not user:
                raise HTTPException(422, "账号不存在")
            from field_recognition.bindings.binding_operator import activate_member

            activate_member(conn, camera_id, user[0], core["now"]())
            members = conn.execute(
                """SELECT u.id,u.display_name FROM camera_users c JOIN users u ON u.id=c.user_id
                                    WHERE c.camera_id=? AND u.disabled=0 ORDER BY u.display_name,u.id""",
                (camera_id,),
            ).fetchall()
            result = {
                "camera_id": camera_id,
                "user_id": user[0],
                "revision": camera_revision(conn, camera_id),
                "members": [
                    {"user_id": r["id"], "display_name": r["display_name"]} for r in members
                ],
            }
            return remember_request(conn, body.get("request_id"), camera_id, fingerprint, result)

    @app.post("/api/admin/cameras/{camera_id}/members")
    def register_member(camera_id: str, body: dict):
        require_role("admin")
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", camera_id):
            raise HTTPException(422, "相机编号无效")
        with core["database"].transaction("IMMEDIATE") as conn:
            if not conn.execute(
                "SELECT 1 FROM users WHERE id=? AND disabled=0", (body.get("user_id"),)
            ).fetchone():
                raise HTTPException(422, "人员不存在或已停用")
            conn.execute(
                "INSERT OR IGNORE INTO camera_users(camera_id,user_id) VALUES(?,?)",
                (camera_id, body["user_id"]),
            )
        return camera_members(camera_id)

    @app.get("/api/cameras/{camera_id}/members")
    def camera_members(camera_id: str):
        require_camera(camera_id)
        with core["database"].connection() as conn:
            from field_recognition.bindings.binding_operator import camera_revision

            rows = conn.execute(
                """SELECT u.id,u.display_name FROM camera_users c JOIN users u ON u.id=c.user_id
                                 WHERE c.camera_id=? AND u.disabled=0 ORDER BY u.display_name,u.id""",
                (camera_id,),
            ).fetchall()
            active = conn.execute(
                "SELECT user_id FROM camera_active_users WHERE camera_id=?", (camera_id,)
            ).fetchone()
            return {
                "camera_id": camera_id,
                "active_user_id": active[0] if active else None,
                "revision": camera_revision(conn, camera_id),
                "members": [{"user_id": r["id"], "display_name": r["display_name"]} for r in rows],
            }

    @app.put("/api/cameras/{camera_id}/active-user")
    def select_member(camera_id: str, body: dict):
        require_camera(camera_id)
        actor = current()
        if actor and (
            actor.device
            or actor.role not in {"admin", "operator"}
            or actor.role != "admin"
            and body.get("user_id") != actor.user_id
        ):
            raise HTTPException(403, "只能选择自己的人员身份；管理员可明确交接给登记人员")
        with core["database"].transaction("IMMEDIATE") as conn:
            from field_recognition.bindings.binding_operator import (
                activate_member,
                check_revision,
                camera_revision,
                request_fingerprint,
                replay_request,
                remember_request,
            )

            fingerprint = request_fingerprint(
                {
                    "operation": "select_member",
                    "camera_id": camera_id,
                    "body": body,
                    "principal": actor.user_id if actor else None,
                }
            )
            replay = replay_request(conn, body.get("request_id"), fingerprint)
            if replay is not None:
                return replay
            check_revision(conn, camera_id, body.get("expected_revision"))
            if not conn.execute(
                """SELECT 1 FROM camera_users c JOIN users u ON c.user_id=u.id
                                   WHERE c.camera_id=? AND c.user_id=? AND u.disabled=0""",
                (camera_id, body.get("user_id")),
            ).fetchone():
                raise HTTPException(422, "请先登记此人员的相机使用资格")
            activate_member(conn, camera_id, body["user_id"], core["now"]())
            return remember_request(
                conn,
                body.get("request_id"),
                camera_id,
                fingerprint,
                {
                    "camera_id": camera_id,
                    "active_user_id": body["user_id"],
                    "revision": camera_revision(conn, camera_id),
                },
            )

    @app.get("/api/admin/cameras/{camera_id}/members")
    def members(camera_id: str):
        require_role("admin")
        return camera_members(camera_id)

    @app.delete("/api/admin/cameras/{camera_id}/members/{user_id}")
    def remove_member(camera_id: str, user_id: str):
        require_role("admin")
        with core["database"].transaction("IMMEDIATE") as conn:
            active = conn.execute(
                "SELECT user_id FROM camera_active_users WHERE camera_id=?", (camera_id,)
            ).fetchone()
            if active and active["user_id"] == user_id:
                raise HTTPException(409, "请先切换当前使用人再移除此成员")
            conn.execute(
                "DELETE FROM camera_users WHERE camera_id=? AND user_id=?", (camera_id, user_id)
            )
            if not conn.execute(
                "SELECT 1 FROM camera_users WHERE camera_id=?", (camera_id,)
            ).fetchone():
                raise HTTPException(409, "相机至少需要保留一名登记人员")
            rows = conn.execute(
                """SELECT u.id,u.display_name FROM camera_users c JOIN users u ON u.id=c.user_id
                                 WHERE c.camera_id=? AND u.disabled=0 ORDER BY u.display_name,u.id""",
                (camera_id,),
            ).fetchall()
        return {
            "camera_id": camera_id,
            "members": [{"user_id": r["id"], "display_name": r["display_name"]} for r in rows],
        }

    @app.post("/api/admin/cameras/{camera_id}/credential")
    def credential(camera_id: str):
        require_account_admin()
        token = secrets.token_urlsafe(32)
        core["auth_service"].register_device(camera_id, "neck_camera", token, current().user_id)
        return {"camera_id": camera_id, "credential": token, "display_once": True}
