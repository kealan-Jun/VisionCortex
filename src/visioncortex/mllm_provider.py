"""Shared request transport and configuration identity for vision providers."""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any
from urllib.parse import urlsplit, urlunsplit


PROTOCOLS = {"ark_responses", "chat_completions"}


def normalize_connection(value: dict[str, Any]) -> dict[str, str]:
    result = {}
    for field in ("provider", "base_url", "model", "api_protocol"):
        item = value.get(field)
        if not isinstance(item, str) or not item.strip() or len(item) > 2048 or any(ord(c) < 32 for c in item):
            raise ValueError(f"请填写有效的 {field}。")
        result[field] = item.strip()
    if not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", result["provider"]) or result["api_protocol"] not in PROTOCOLS:
        raise ValueError("请选择支持的服务厂商和接口类型。")
    parsed = urlsplit(result["base_url"])
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("接口地址必须是完整的 HTTPS Base URL，不能包含密钥、查询参数或片段。")
    # Accessing port validates malformed authority strings too.
    _ = parsed.port
    endpoint = parsed.path.rstrip("/")
    if endpoint.endswith(("/responses", "/chat/completions")):
        raise ValueError("请填写 Base URL，不要包含 /responses 或 /chat/completions。")
    result["base_url"] = urlunsplit(("https", parsed.netloc.lower(), endpoint, "", ""))
    result["quality_mode"] = value.get("quality_mode", "quality")
    if result["quality_mode"] not in {"quality", "balanced"}:
        raise ValueError("请选择质量优先或速度与成本优先。")
    return result


def connection_identity(connection: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(normalize_connection(connection), sort_keys=True).encode()).hexdigest()


def vision_request_identity(settings: dict[str, Any]) -> str:
    """Bind semantic caches to the effective transport policy, excluding secrets."""
    url, request = build_vision_request(settings, "", [])
    # Delivery framing does not change evidence, prompts or generation policy.
    # Preserve validated semantic cache identities from non-streaming requests.
    if "stream" in request:
        request["stream"] = False
        request.pop("stream_options", None)
    policy = {"url": url, "request": request,
              "image_max_edge": int(settings.get("request_image_max_edge", 0)),
              "image_jpeg_quality": int(settings.get("request_image_jpeg_quality", 85))}
    return hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest()


def build_vision_request(settings: dict[str, Any], system_prompt: str, content: list[dict[str, Any]]) -> tuple[str, dict]:
    protocol = settings.get("api_protocol", "ark_responses")
    base = str(settings["base_url"]).rstrip("/")
    tokens = int(settings.get("max_output_tokens", 4096))
    quality = settings.get("quality_mode") == "quality"
    required_thinking = settings.get("provider") == "zhipu" and str(settings["model"]).lower().startswith("glm-5.3-flash")
    thinking = "enabled" if quality or required_thinking else "disabled"
    if quality or required_thinking:
        tokens = max(tokens, 8192)
    if protocol == "ark_responses":
        return base + "/responses", {
            "model": settings["model"], "instructions": system_prompt,
            "input": [{"type": "message", "role": "user", "content": content}],
            "store": False, "thinking": {"type": thinking}, "max_output_tokens": tokens,
        }
    if protocol != "chat_completions":
        raise ValueError("Unsupported multimodal API protocol")
    messages = []
    for item in content:
        if item["type"] == "input_text":
            messages.append({"type": "text", "text": item["text"]})
        elif item["type"] == "input_image":
            url = item["image_url"]
            # The GLM vision API documents raw Base64 as well as remote URLs.
            if settings.get("provider") == "zhipu" and not required_thinking and url.startswith("data:"):
                url = url.split(",", 1)[1]
            messages.append({"type": "image_url", "image_url": {"url": url}})
        else:
            raise ValueError("Unsupported vision evidence content")
    request = {"model": settings["model"], "stream": False, "max_tokens": tokens,
               "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": messages}]}
    if settings.get("provider") == "aliyun":
        request["enable_thinking"] = quality
        if quality:
            request.update(stream=True, stream_options={"include_usage": True})
    elif settings.get("provider") == "zhipu":
        request["thinking"] = {"type": thinking}
    return base + "/chat/completions", request
