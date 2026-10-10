"""Web authentication, admission and stream middleware."""

from __future__ import annotations
from ..application.dependencies import ports
import re
from pathlib import Path
from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.responses import Response
from ..provider_credentials import key_configured
from ..model_certification import audit_production_model_certification
from ..web_access import (
    authenticate_basic_authorization,
    is_allowed_lan_client,
    web_access_mode,
    web_https_required,
)

from ..application import runtime_host
from . import health as http_health


async def enforce_web_access(request: Request, call_next):
    """Keep the default local service open and fail closed for LAN service mode."""

    if runtime_host._storage_maintenance() and request.url.path not in {
        "/api/device-day-progress",
        "/health/live",
        "/health/automation",
    }:
        message = "修复版代码已加载。NAS 正在维护，数据读取、提交与后台处理暂不开放；历史队列保留，未触发全量重跑。"
        headers = {"Retry-After": "60", "Cache-Control": "no-store"}
        if request.url.path.startswith("/api/"):
            return JSONResponse(
                {"status": "storage_maintenance", "detail": message},
                status_code=503,
                headers=headers,
            )
        return HTMLResponse(
            "<!doctype html><meta charset='utf-8'><title>VisionCortex 维护状态</title>"
            "<main style='max-width:760px;margin:80px auto;font:20px sans-serif'>"
            "<h1>VisionCortex · NAS 维护中</h1><p>" + message + "</p></main>",
            status_code=503,
            headers=headers,
        )
    started = ports.time.perf_counter()
    identity: dict[str, str] | None = None
    mode = "unknown"

    def audited(response: Response) -> Response:
        if mode == "lan" and request.url.path.startswith("/api/"):
            try:
                settings = runtime_host._settings()
                audit_path = (
                    Path(settings["storage"]["local_runtime_root"])
                    / "state"
                    / f"web_access_audit-{ports.datetime.now().astimezone():%Y-%m-%d}.jsonl"
                )
                with runtime_host._web_access_audit_lock:
                    ports.append_access_audit(
                        audit_path,
                        {
                            "schema_version": "visioncortex-web-access-audit/1",
                            "observed_at": ports.datetime.now()
                            .astimezone()
                            .isoformat(),
                            "username": (identity or {}).get("username") or "anonymous",
                            "role": (identity or {}).get("role"),
                            "client": request.client.host if request.client else None,
                            "method": request.method,
                            "path": request.url.path,
                            "status_code": response.status_code,
                            "duration_ms": round(
                                (ports.time.perf_counter() - started) * 1000.0, 3
                            ),
                        },
                    )
            except (OSError, KeyError, RuntimeError, TypeError):
                pass
        return response

    async def dispatch():
        from ..submission import handle, eligible

        if request.method != "POST" or not eligible(request.url.path):
            return await call_next(request)
        return await handle(request, call_next, runtime_host._settings())

    try:
        mode = web_access_mode()
        if mode == "local":
            identity = {"username": "local", "role": "admin"}
            request.state.web_identity = identity
            return audited(await dispatch())
        client_host = request.client.host if request.client else None
        if not is_allowed_lan_client(client_host):
            return audited(
                Response(
                    "VisionCortex LAN access is not allowed from this network.",
                    status_code=403,
                    headers={"Cache-Control": "no-store"},
                )
            )
        if web_https_required() and request.url.scheme != "https":
            return audited(
                Response(
                    "VisionCortex LAN access requires HTTPS.",
                    status_code=426,
                    headers={"Cache-Control": "no-store"},
                )
            )
        identity = authenticate_basic_authorization(
            request.headers.get("authorization")
        )
        if identity is None:
            return audited(
                Response(
                    "VisionCortex login required.",
                    status_code=401,
                    headers={
                        "WWW-Authenticate": 'Basic realm="VisionCortex 3090 Ti", charset="UTF-8"',
                        "Cache-Control": "no-store",
                    },
                )
            )
        if identity["role"] == "viewer" and request.method not in {
            "GET",
            "HEAD",
            "OPTIONS",
        }:
            return audited(
                Response(
                    "VisionCortex viewer accounts are read-only.",
                    status_code=403,
                    headers={"Cache-Control": "no-store"},
                )
            )
        if identity["role"] != "admin" and (
            request.url.path.startswith("/api/annotation-workspace")
            or (
                request.method == "POST"
                and request.url.path.startswith("/api/archives/")
                and request.url.path.endswith("/open")
            )
        ):
            return audited(
                Response(
                    "VisionCortex administrator access is required.",
                    status_code=403,
                    headers={"Cache-Control": "no-store"},
                )
            )
        request.state.web_identity = identity
    except RuntimeError:
        return audited(
            Response(
                "VisionCortex Web access configuration is invalid.",
                status_code=503,
                headers={"Cache-Control": "no-store"},
            )
        )
    return audited(await dispatch())


async def record_web_ingest_start(request: Request, call_next):
    """Capture the request boundary before multipart parsing starts."""

    if request.method == "POST" and request.url.path == "/api/runs":
        request.state.ingest_started_perf = ports.time.perf_counter()
        request.state.ingest_started_epoch = ports.time.time()
        request.state.ingest_started_at = ports.datetime.now().astimezone().isoformat()
    return await call_next(request)


async def require_analysis_certification(request: Request, call_next):
    path = request.url.path
    starts_analysis = (
        path in {"/api/runs", "/api/runs/from-paths", "/api/upload-sessions"}
        or bool(re.fullmatch(r"/api/(collections|benchmarks)/[^/]+/runs", path))
        or bool(re.fullmatch(r"/api/nas-batches/[^/]+/runs", path))
        or bool(re.fullmatch(r"/api/upload-sessions/[^/]+/finalize", path))
        or bool(re.fullmatch(r"/api/runs/[^/]+/retry", path))
    )
    if request.method == "POST" and starts_analysis:
        settings = runtime_host._settings()
        from ..config import require_configured_site
        try:
            require_configured_site(settings)
        except ValueError:
            return JSONResponse(status_code=503, content={"detail": {
                "code": "site_configuration_required",
                "message": "目标硬件模板尚未配置私有站点，当前可浏览本地页面。",
            }})
        if settings.get("mllm", {}).get("credential_ref") and not key_configured(
            settings["mllm"]
        ):
            return JSONResponse(
                status_code=409,
                content={"detail": "AI 服务配置需要重新验证，请打开 AI 服务设置。"},
            )
        if settings["storage"].get("sync_to_nas"):
            try:
                audit_production_model_certification(settings)
            except (OSError, RuntimeError, ValueError, KeyError, TypeError):
                return JSONResponse(
                    status_code=503,
                    content={
                        "detail": {
                            "code": "model_certification_required",
                            "message": "分析模型尚未完成质量验收，当前可浏览和整理 NAS 素材。",
                        }
                    },
                )
            if not http_health._nas_storage_available(settings):
                return JSONResponse(
                    status_code=503,
                    content={
                        "detail": {
                            "code": "nas_unavailable",
                            "message": "NAS 归档或缓存目录不可用，请检查连接后重试。",
                        }
                    },
                )
    return await call_next(request)


async def limit_archive_streams(request: Request, call_next):
    """Bound concurrent NAS-backed streams for predictable multi-user playback."""

    is_stream = request.url.path in {"/api/archive-file", "/api/staging-file"} or bool(
        re.fullmatch(
            r"/api/(staging-runs|archives)/[^/]+/speech-video", request.url.path
        )
    )
    if not is_stream or request.query_params.get("poster") == "true":
        return await call_next(request)
    if not runtime_host._archive_stream_slots.acquire(blocking=False):
        return Response(
            "VisionCortex archive streaming is busy; retry shortly.",
            status_code=429,
            headers={"Retry-After": "2", "Cache-Control": "no-store"},
        )
    try:
        response = await call_next(request)
    except BaseException:
        runtime_host._archive_stream_slots.release()
        raise
    original_iterator = response.body_iterator

    async def guarded_body():
        try:
            async for chunk in original_iterator:
                yield chunk
        finally:
            runtime_host._archive_stream_slots.release()

    response.body_iterator = guarded_body()
    return response


def install_middleware(app):
    app.middleware("http")(enforce_web_access)
    app.middleware("http")(record_web_ingest_start)
    app.middleware("http")(require_analysis_certification)
    app.middleware("http")(limit_archive_streams)
