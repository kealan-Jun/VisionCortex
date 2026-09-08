"""Bounded model discovery at the explicitly selected compatible endpoint.

Discovery is provider metadata, never a vision verification or activation receipt.
"""
from __future__ import annotations

import json

import httpx


class DiscoveryError(ValueError):
    """A safe, locally generated message; never upstream response/error text."""


def discover_models(connection: dict, api_key: str) -> dict:
    url = connection["base_url"] + "/models"
    try:
        # Never forward a credential to redirects, guessed vendors or env proxies.
        with httpx.Client(timeout=20, follow_redirects=False, trust_env=False) as client:
            with client.stream("GET", url, headers={"Authorization": f"Bearer {api_key}"}) as response:
                if response.status_code in {401, 403}:
                    raise DiscoveryError("模型列表访问被拒绝，请检查所选服务的密钥与权限。")
                if response.status_code in {404, 405}:
                    raise DiscoveryError("此服务未提供兼容模型列表，请手动填写视觉模型或部署 ID 后验证。")
                if response.status_code != 200:
                    raise DiscoveryError("模型列表暂时无法读取，请重试或手动填写模型 ID。")
                raw = bytearray()
                for chunk in response.iter_bytes():
                    raw.extend(chunk)
                    if len(raw) > 4 * 1024 * 1024:
                        raise DiscoveryError("模型列表过大，请手动填写视觉模型 ID。")
        value = json.loads(raw)
    except httpx.HTTPError as exc:
        raise DiscoveryError("模型列表连接未完成，请核对接口地址后重试。") from exc
    except (ValueError, UnicodeError) as exc:
        if isinstance(exc, DiscoveryError):
            raise
        raise DiscoveryError("服务未返回兼容模型列表，请手动填写模型 ID 后验证。") from exc
    data = value.get("data") if isinstance(value, dict) else None
    if not isinstance(data, list):
        raise DiscoveryError("服务未返回兼容模型列表，请手动填写模型 ID 后验证。")
    models = {}
    excluded = 0
    for item in data[:2000]:
        if not isinstance(item, dict):
            continue
        model_id = item.get("id")
        if (not isinstance(model_id, str) or not model_id.strip() or len(model_id) > 256
                or any(ord(c) < 32 for c in model_id) or api_key in model_id):
            continue
        architecture = item.get("architecture")
        modalities = architecture.get("input_modalities") if isinstance(architecture, dict) else None
        if not isinstance(modalities, list):
            modalities = item.get("input_modalities")
        vision = "image" in modalities if isinstance(modalities, list) and modalities else None
        if vision is False:
            excluded += 1
            continue
        label = item.get("name")
        if not isinstance(label, str) or not label.strip() or len(label) > 256 or api_key in label:
            label = model_id
        models[model_id] = {"id": model_id, "label": label, "vision_declared": vision}
    return {
        "models": sorted(models.values(), key=lambda m: (m["vision_declared"] is not True, m["id"])),
        "source": "provider_model_list",
        "excluded_non_image": excluded,
        "truncated": len(data) > 2000 or bool(value.get("has_more")),
        "model_invocation": "NOT_PROVEN",
        "activated": False,
    }
