"""HTTP adapters for settings"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from .. import ai_settings

from ..application import runtime_host

from fastapi import APIRouter

router = APIRouter()


def _require_local_ai_settings(request: Request) -> None:
    if not ai_settings.enabled():
        raise HTTPException(404, "当前服务未启用本机 AI 设置。")
    identity = getattr(request.state, "web_identity", {})
    if (
        not request.client
        or request.client.host not in {"127.0.0.1", "::1"}
        or request.url.hostname not in {"127.0.0.1", "localhost", "::1"}
        or identity.get("role") != "admin"
    ):
        raise HTTPException(403, "请从本机工作台以管理员身份设置 AI 服务。")
    origin = request.headers.get("origin")
    expected = f"{request.url.scheme}://{request.url.netloc}"
    if (origin and origin != expected) or request.headers.get(
        "sec-fetch-site"
    ) == "cross-site":
        raise HTTPException(403, "AI 设置请求来源无效。")
    if request.method == "POST" and (
        origin != expected
        or request.headers.get("content-type", "").split(";", 1)[0]
        != "application/json"
    ):
        raise HTTPException(403, "请通过本机 AI 服务设置页面提交。")


def get_ai_settings(request: Request) -> JSONResponse:
    _require_local_ai_settings(request)
    try:
        result = ai_settings.public_settings(runtime_host._settings())
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        raise HTTPException(
            503, "本机 AI 配置无法读取，请检查本地配置目录权限。"
        ) from exc
    return JSONResponse(result, headers={"Cache-Control": "no-store"})


async def verify_ai_settings(request: Request) -> JSONResponse:
    _require_local_ai_settings(request)
    raw = bytearray()
    async for chunk in request.stream():
        raw.extend(chunk)
        if len(raw) > 16384:
            raise HTTPException(413, "AI 配置请求过大。")
    try:
        value = json.loads(raw)
        connection = value.get("connection")
        key = value.get("api_key", "")
        if (
            not isinstance(connection, dict)
            or not isinstance(key, str)
            or len(key) > 4096
            or any(ord(c) < 32 for c in key)
        ):
            raise ValueError("invalid settings")
    except (ValueError, AttributeError, TypeError) as exc:
        raise HTTPException(400, "请填写有效的厂商、视觉模型和 API 密钥。") from exc
    try:
        action = (
            ai_settings.discover_available_models
            if request.url.path.endswith("/models")
            else ai_settings.verify_and_activate
        )
        result = await run_in_threadpool(
            action, connection, key, runtime_host._settings()
        )
    except ai_settings.DiscoveryError as exc:
        raise HTTPException(400, str(exc)) from exc
    except BlockingIOError as exc:
        raise HTTPException(409, "已有连接正在验证，请稍候。") from exc
    except ValueError as exc:
        raise HTTPException(
            400, "请核对厂商、HTTPS 接口地址、视觉模型及 API 密钥。"
        ) from exc
    except (OSError, RuntimeError, KeyError, TypeError) as exc:
        raise HTTPException(
            503, "连接验证或配置保存未完成，请重试并检查本机配置目录权限。"
        ) from exc
    return JSONResponse(result, headers={"Cache-Control": "no-store"})


def model_candidates() -> dict[str, Any]:
    """Expose the committed, non-production model quality ledger to local Web."""

    registry_path = (
        Path(__file__).resolve().parents[3]
        / "configs"
        / "models"
        / "public-apparatus-candidates.json"
    )
    try:
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(503, "模型候选质量账本不可用") from exc
    candidates = payload.get("candidates")
    if payload.get(
        "schema_version"
    ) != "visioncortex-public-apparatus-candidate-registry/1" or not isinstance(
        candidates, dict
    ):
        raise HTTPException(503, "模型候选质量账本格式无效")
    records = []
    for candidate_id, raw in candidates.items():
        if not isinstance(raw, dict):
            raise HTTPException(503, "模型候选质量账本包含无效记录")
        records.append({**raw, "candidate_id": str(candidate_id)})
    return {
        "schema_version": payload["schema_version"],
        "production_configuration_changed": False,
        "candidate_count": len(records),
        "candidates": records,
    }


router.get("/api/ai-settings")(get_ai_settings)

router.post("/api/ai-settings/models")(verify_ai_settings)

router.post("/api/ai-settings/verify")(verify_ai_settings)

router.get("/api/model-candidates")(model_candidates)
