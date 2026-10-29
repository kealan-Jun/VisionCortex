"""Explicit runtime actions for capture actions."""

from __future__ import annotations

import hashlib
import io
import json
import uuid
import cv2
import httpx
import numpy as np
from fastapi import File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError
from field_recognition.api.security import require_camera, actor_name


def save_image(_runtime, data, source, camera):
    if not data or len(data) > _runtime["MAX_BYTES"]:
        raise HTTPException(413, "图片为空或超过 12 MB")
    try:
        with Image.open(io.BytesIO(data)) as original:
            original.load()
            original_extension = {
                "JPEG": ".jpg",
                "PNG": ".png",
                "WEBP": ".webp",
                "BMP": ".bmp",
                "TIFF": ".tiff",
            }.get(original.format, ".bin")
            if original.width * original.height > 16_000_000:
                raise ValueError("图片像素过大")
            # EXIF orientation must match what the user sees and crops.
            from PIL import ImageOps

            image = ImageOps.exif_transpose(original).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise HTTPException(422, "无法解码图片，或图片像素超过上限") from exc
    ident = str(uuid.uuid4())
    path = _runtime["DATA"] / "Images" / f"{ident}.png"
    image.save(path)
    source_hash = hashlib.sha256(data).hexdigest()
    original_blob = f"Originals/{source_hash}{original_extension}"
    from field_recognition.archive.archive_store import immutable_write

    immutable_write(_runtime["DATA"] / original_blob, data)
    return {
        "capture_id": ident,
        "image_url": f"/api/images/{ident}",
        "source": source,
        "camera_id": camera,
        "received_at": _runtime["now"](),
        "source_sha256": source_hash,
        "original_blob": original_blob,
        "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "width": image.width,
        "height": image.height,
        "path": str(path),
    }


def qr_matches(_runtime, values, points):
    matches, scene_matches, unknown = [], [], []
    for ordinal, value in enumerate(values):
        if not value:
            continue
        try:
            payload = json.loads(value)
            if (
                not isinstance(payload, dict)
                or payload.get("v") != 1
                or payload.get("type") not in {"instrument", "scene"}
            ):
                raise ValueError("unknown schema")
            instrument_id = str(uuid.UUID(payload["id"]))
        except (ValueError, TypeError, KeyError):
            unknown.append("不是本系统的仪器码")
            continue
        if payload["type"] == "scene":
            scene = next((s for s in _runtime["scene_records"]() if s["id"] == instrument_id), None)
            if scene:
                if not any(s["id"] == instrument_id for s in scene_matches):
                    scene_matches.append(
                        scene
                        | {
                            "polygon": np.asarray(points[ordinal]).tolist(),
                            "qr_hash": hashlib.sha256(value.encode("utf-8")).hexdigest(),
                        }
                    )
            else:
                unknown.append("场景编号尚未登记：" + instrument_id)
            continue
        asset = _runtime["get_instrument"](instrument_id)
        if not asset:
            unknown.append("仪器编号尚未登记：" + instrument_id)
            continue
        if not any(row["id"] == instrument_id for row in matches):
            matches.append(
                asset
                | {
                    "polygon": np.asarray(points[ordinal]).tolist(),
                    "qr_hash": hashlib.sha256(value.encode("utf-8")).hexdigest(),
                }
            )
    return {
        "matches": matches,
        "scene_matches": scene_matches,
        "unknown": unknown,
        "status": "matched"
        if matches or scene_matches
        else "not_registered"
        if unknown
        else "no_qr",
        "signature_status": "unsigned_demo_label",
    }


def scan_image(
    _runtime,
    data,
    source,
    camera,
    *,
    decoded=None,
    metadata=None,
    operator=None,
    scan_session_id=None,
    persist=True,
):
    require_camera(camera)
    operator = actor_name(operator)
    capture = _runtime["save_image"](data, source, camera)
    if decoded is None:
        from field_recognition.capture.qr_decode import decode_qr

        decoded = decode_qr(cv2.imread(capture["path"]))
    values, points, diagnostics = decoded
    result = (
        capture
        | _runtime["qr_matches"](values, points)
        | {"scan_id": capture["capture_id"], "qr_diagnostics": diagnostics}
    )
    if metadata is not None:
        result["frame_metadata"] = metadata
    if operator is not None:
        result["operator"] = operator
        result["scan_session_id"] = scan_session_id
    if persist:
        with _runtime["db"]() as conn:
            conn.execute("INSERT INTO scans VALUES(?,?)", (result["scan_id"], json.dumps(result)))
    return result


def save_camera_scan(
    _runtime, frame, metadata, *, decoded=None, operator=None, scan_session_id=None
):
    ok, encoded = cv2.imencode(".png", frame)
    if not ok:
        raise HTTPException(500, "画面编码失败")
    return _runtime["scan_image"](
        encoded.tobytes(),
        "neck_camera_gwhp_main",
        _runtime["receiver_camera"].target,
        decoded=decoded,
        metadata=metadata,
        operator=operator,
        scan_session_id=scan_session_id,
    )


async def scan(_runtime, file: UploadFile = File(...), camera_id: str = Form("UploadedPhoto")):
    if not camera_id.strip() or len(camera_id) > 100:
        raise HTTPException(422, "相机编号无效")
    data = await file.read(_runtime["MAX_BYTES"] + 1)
    from starlette.concurrency import run_in_threadpool

    return await run_in_threadpool(_runtime["scan_image"], data, "uploaded_photo", camera_id)


def camera_capture(_runtime):
    if _runtime["receiver_camera"]:
        try:
            frame, metadata = _runtime["receiver_camera"].frame()
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        return _runtime["save_camera_scan"](frame, metadata)
    url = _runtime.config.camera_snapshot_url
    camera = _runtime.config.camera_id
    if not url or not camera:
        raise HTTPException(409, "挂脖设备尚未连接：需要设备编号及返回 JPEG/PNG 的取图接口")
    started = _runtime["time"].monotonic()
    try:
        with httpx.stream(
            "GET", url, timeout=12, follow_redirects=False, trust_env=False
        ) as response:
            response.raise_for_status()
            data = bytearray()
            for block in response.iter_bytes(65536):
                data.extend(block)
                if len(data) > _runtime["MAX_BYTES"] or _runtime["time"].monotonic() - started > 15:
                    raise ValueError("capture limit")
        return _runtime["scan_image"](bytes(data), "neck_camera_http_snapshot", camera)
    except (httpx.HTTPError, ValueError) as exc:
        raise HTTPException(502, "挂脖设备取图失败，请检查连接和取图服务") from exc


def image_file(_runtime, capture_id: uuid.UUID):
    path = _runtime["DATA"] / "Images" / f"{capture_id}.png"
    if not path.is_file():
        raise HTTPException(404, "图片不存在")
    return FileResponse(path)
