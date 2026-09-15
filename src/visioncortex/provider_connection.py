"""Live, bounded vision connection check using the production analyzer transport."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import secrets
import tempfile
from typing import Any

import cv2
import numpy as np

from .mllm import ArkAnalyzer
from .mllm_provider import connection_identity, normalize_connection


def adapter_identity() -> str:
    digest = hashlib.sha256()
    for name in ("mllm.py", "mllm_provider.py", "provider_connection.py"):
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


def verification_matches(connection: dict, receipt: dict) -> bool:
    return bool(
        receipt.get("status") == "verified"
        and receipt.get("model_invocation") == "PROVEN"
        and receipt.get("connection_sha256") == connection_identity(connection)
        and receipt.get("adapter_sha256") == adapter_identity()
    )


def connection_health(settings: dict, key_configured: bool) -> dict:
    receipt = settings.get("connection_verification") or {}
    try:
        verified = key_configured and verification_matches(settings, receipt)
    except (ValueError, TypeError, AttributeError):
        verified = False
    return {"status": "verified" if verified else "not_verified",
            "provider": settings.get("provider", "volcengine"), "model": settings.get("model"),
            "api_protocol": settings.get("api_protocol", "ark_responses"),
            "checked_at": receipt.get("checked_at") if verified else None,
            "request_id": receipt.get("request_id") if verified else None,
            "usage": receipt.get("usage") if verified else None,
            "scope": "synthetic_multi_image_connection_only", "real_video_quality": "NOT_PROVEN"}


def _challenge(root: Path) -> tuple[list[tuple[str, Path]], list[dict[str, str]]]:
    paths, expected = [], []
    palette = {"red": (0, 0, 230), "green": (0, 155, 0), "blue": (220, 0, 0)}
    for index in range(2):
        item = {"code": str(secrets.randbelow(9000) + 1000),
                "color": secrets.choice(list(palette)), "shape": secrets.choice(["circle", "square"])}
        frame = np.full((512, 512, 3), 255, dtype=np.uint8)
        cv2.putText(frame, item["code"], (125, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 6, cv2.LINE_AA)
        if item["shape"] == "circle":
            cv2.circle(frame, (256, 325), 110, palette[item["color"]], -1)
        else:
            cv2.rectangle(frame, (146, 215), (366, 435), palette[item["color"]], -1)
        path = root / f"image-{index + 1}.png"
        if not cv2.imwrite(str(path), frame):
            raise RuntimeError("无法创建连接验证图片。")
        paths.append((f"image-{index + 1}", path))
        expected.append(item)
    return paths, expected


def verify_connection(connection: dict, api_key: str, work_root: Path, *, analyzer_factory=ArkAnalyzer) -> dict[str, Any]:
    selected = normalize_connection(connection)
    if not isinstance(api_key, str) or not api_key.strip() or len(api_key) > 4096 or any(ord(c) < 32 for c in api_key):
        raise ValueError("请输入有效的 API 密钥。")
    settings = dict(selected, enabled=True, api_key_env="VISIONCORTEX_UNUSED_PROBE_KEY",
                    timeout_seconds=180, max_retries=1, workers=1, group_workers=1,
                    stream_total_timeout_seconds=180, stream_read_timeout_seconds=30,
                    max_images_per_event=2, max_output_tokens=512)
    analyzer = analyzer_factory({"mllm": settings})
    analyzer.api_key = api_key.strip()
    work_root.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix="connection-", dir=work_root) as temporary:
            images, expected = _challenge(Path(temporary))
            result = analyzer._call(
                'Read both images in the given order. Return only a JSON object with this schema: '
                '{"images":[{"code":"four printed digits","color":"red|green|blue","shape":"circle|square"}, ...]}. '
                'Return exactly two items. Read the digits and colored shape from each image. Do not guess.',
                {"task": "Read image-1 and image-2; this is a synthetic connection test, not experiment evidence."},
                images,
            )
    finally:
        analyzer.close()
    base = {"connection": selected, "connection_sha256": connection_identity(selected),
            "adapter_sha256": adapter_identity(), "checked_at": datetime.now(timezone.utc).isoformat(),
            "input_kind": "synthetic_connection_challenge_not_experiment_evidence",
            "real_video_quality": "NOT_PROVEN", "usage": result.get("usage"),
            "request_id": result.get("request_id"), "response_model": result.get("response_model"),
            "latency_seconds": result.get("latency_seconds"),
            "transport": next((item["transport"] for item in reversed(result.get("attempt_receipts") or []) if item.get("transport")), None)}
    if result.get("status") != "completed":
        status = result.get("http_status")
        reason = {401: "密钥认证失败，请检查密钥是否属于所选服务和地域。",
                  403: "当前账号没有该服务或模型的调用权限。",
                  404: "接口地址或模型不存在，请核对控制台中的配置。",
                  429: "服务限流或额度不足，请检查账号额度后重试。",
                  400: "服务拒绝了图像请求，请确认模型支持多图理解及当前接口。"}.get(
                      status, "连接或响应验证失败，请检查网络、接口地址与模型配置后重试。")
        return dict(base, status="failed", model_invocation="NOT_PROVEN", http_status=status,
                    message=reason, detail=str(result.get("error", ""))[:1000].replace(api_key.strip(), "[REDACTED]"))
    if result.get("images") != expected:
        return dict(base, status="failed", model_invocation="PARTIAL_EVIDENCE",
                    message="服务返回了内容，但未正确识别两张验证图片，暂不能启用。请确认选择了图像理解模型，或重试。")
    return dict(base, status="verified", model_invocation="PROVEN", multi_image_json_check="PROVEN",
                challenge_sha256=hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest(),
                message="已实际调用所选模型，多图理解与 JSON 响应验证通过。")
