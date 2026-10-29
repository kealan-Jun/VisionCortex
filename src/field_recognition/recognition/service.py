"""Explicit runtime actions for service."""

from __future__ import annotations

import json
import re
import numpy as np
from field_recognition.recognition.ocr_runtime import (
    create_model,
    OcrDeviceUnavailable,
    OcrModelUnavailable,
)
from field_recognition.recognition.panel_readout import run as run_panel_readout


def create_ocr_model(_runtime):
    return create_model(_runtime["OCR_DEVICE"], remote_config=_runtime["ocr_service_config"])


def load_ocr(_runtime):
    with _runtime["ocr_lock"]:
        if _runtime["ocr_model"] is None:
            if _runtime["time"].monotonic() < _runtime["ocr_retry_at"]:
                raise OcrModelUnavailable("模型加载失败，等待自动重试")
            started = _runtime["time"].monotonic()
            _runtime["ocr_state"].update(
                status="loading", error=None, error_detail=None, resident=False
            )
            try:
                _runtime["ocr_model"] = _runtime["create_ocr_model"]()
            except Exception as exc:
                _runtime["ocr_load_failures"] += 1
                delay = min(300, 5 * 2 ** min(_runtime["ocr_load_failures"] - 1, 6))
                _runtime["ocr_retry_at"] = _runtime["time"].monotonic() + delay
                _runtime["ocr_state"].update(
                    status="error",
                    error=type(exc).__name__,
                    resident=False,
                    error_detail=str(exc)
                    if isinstance(exc, (OcrDeviceUnavailable, OcrModelUnavailable))
                    else "模型加载失败，请检查本机模型与服务日志",
                    retry_after_seconds=delay,
                    load_failures=_runtime["ocr_load_failures"],
                )
                raise
            _runtime["ocr_retry_at"], _runtime["ocr_load_failures"] = 0.0, 0
            _runtime["ocr_state"].update(
                loaded_at=_runtime["now"](),
                load_count=_runtime["ocr_state"]["load_count"] + 1,
                load_seconds=round(_runtime["time"].monotonic() - started, 3),
                retry_after_seconds=0,
                load_failures=0,
            )
        if hasattr(_runtime["ocr_model"], "health"):
            _runtime["ocr_state"].update(_runtime["ocr_model"].health())
        else:
            _runtime["ocr_state"].update(
                status="ready", resident=True, error=None, error_detail=None
            )
        return _runtime["ocr_model"]


def warm_ocr(_runtime):
    try:
        _runtime["load_ocr"]()
        _runtime["panel_detector"].warmup()
    except Exception:
        # load_ocr exposes the failure in /api/state; the successful binding is retained.
        return False
    return True


def queue_ocr_warmup(_runtime):
    with _runtime["ocr_warmup_lock"]:
        if (
            _runtime["ocr_model"] is None
            and _runtime["time"].monotonic() < _runtime["ocr_retry_at"]
        ):
            return _runtime["ocr_warmup_future"]
        if _runtime["ocr_model"] is not None or (
            _runtime["ocr_warmup_future"] is not None and not _runtime["ocr_warmup_future"].done()
        ):
            return _runtime["ocr_warmup_future"]
        _runtime["ocr_state"].update(status="queued", error=None)
        try:
            # Same single worker as inference, so loading/prediction never run in parallel.
            _runtime["ocr_warmup_future"] = _runtime["ocr_pool"].submit(_runtime["warm_ocr"])
        except RuntimeError as exc:
            _runtime["ocr_state"].update(status="error", error=type(exc).__name__)
        return _runtime["ocr_warmup_future"]


def predict_panel(_runtime, panel, x=0, y=0):
    """Manual requests and saved-photo events share the resident model."""
    started = _runtime["time"].monotonic()
    ocr_started_at = _runtime["now"]()
    invoked = False
    lock_acquired = prepared = call_started = call_finished = None
    model_call_scope = None

    def stages():
        return {
            "ocr_model_lock_wait_ms": round((lock_acquired - started) * 1000, 3)
            if lock_acquired is not None
            else None,
            "ocr_model_prepare_ms": round((prepared - lock_acquired) * 1000, 3)
            if prepared is not None
            else None,
            "ocr_model_call_ms": round((call_finished - call_started) * 1000, 3)
            if call_finished is not None
            else None,
            "ocr_model_call_count": int(call_started is not None),
            "ocr_model_call_scope": model_call_scope,
        }

    try:
        with _runtime["ocr_lock"]:
            lock_acquired = _runtime["time"].monotonic()
            model = _runtime["load_ocr"]()
            invoked = not hasattr(model, "last_receipt")
            prepared = _runtime["time"].monotonic()
            model_call_scope = (
                "adapter_call_including_remote_queue_and_transport"
                if hasattr(model, "last_receipt")
                else "local_model_predict_iteration"
            )
            call_started = _runtime["time"].monotonic()
            try:
                output = list(model.predict(panel))
            finally:
                call_finished = _runtime["time"].monotonic()
            receipt = getattr(model, "last_receipt", None)
            invoked = bool(receipt["execution"]["actual_model_invocation"]) if receipt else invoked
            lines = []
            for result in output:
                for text, score, polygon in zip(
                    result["rec_texts"], result["rec_scores"], result["rec_polys"]
                ):
                    poly = np.asarray(polygon).astype(float)
                    poly[:, 0] += x
                    poly[:, 1] += y
                    lines.append(
                        {
                            "text": text,
                            "confidence": float(score),
                            "polygon": poly.tolist(),
                            "numeric_candidates": re.findall(r"[-+]?\d+(?:\.\d+)?", text),
                        }
                    )
        return {
            "status": "completed",
            "lines": lines,
            "model": _runtime["ocr_state"]["engine"],
            "ocr_started_at": ocr_started_at,
            "ocr_finished_at": _runtime["now"](),
            "actual_model_invocation": invoked,
            "device": _runtime["OCR_DEVICE"],
            **stages(),
            **({"remote_inference": receipt} if receipt else {}),
            "wall_seconds": round(_runtime["time"].monotonic() - started, 3),
        }
    except Exception as exc:
        from field_recognition.recognition.remote_ocr import InferenceUnavailable

        if isinstance(exc, InferenceUnavailable):
            _runtime["ocr_state"].update(
                status="waiting_service", resident=False, error="InferenceUnavailable"
            )
            raise
        _runtime["ocr_state"].update(status="error", error=type(exc).__name__)
        return {
            "status": "failed",
            "lines": [],
            "error": type(exc).__name__,
            "ocr_started_at": ocr_started_at,
            "ocr_finished_at": _runtime["now"](),
            "model": _runtime["ocr_state"]["engine"],
            "device": _runtime["OCR_DEVICE"],
            "actual_model_invocation": invoked,
            **stages(),
            "wall_seconds": round(_runtime["time"].monotonic() - started, 3),
        }


def run_ocr(_runtime, document):
    run_panel_readout(_runtime, document)


def current_panel_bindings(_runtime, camera_id):
    from field_recognition.bindings.binding_policy import sweep

    sweep(_runtime)
    with _runtime["db"]() as conn:
        rows = conn.execute(
            "SELECT document FROM bindings WHERE camera=? AND ended IS NULL", (camera_id,)
        ).fetchall()
    return [
        b for row in rows if _runtime["current_readout_binding"](b := json.loads(row["document"]))
    ]


def predict_readout(
    _runtime,
    panel,
    x=0,
    y=0,
    *,
    video=False,
    binding_snapshots=None,
    field_rules=None,
    localization_only=False,
    detection=None,
):
    from field_recognition.recognition.panel_regions import predict

    if binding_snapshots is None:
        binding_snapshots = (
            _runtime["current_panel_bindings"](_runtime["receiver_camera"].target)
            if video and _runtime["receiver_camera"]
            else []
        )
    local_core = _runtime
    if field_rules is not None:
        local_core = dict(
            _runtime,
            get_instrument=lambda iid: (_runtime["get_instrument"](iid) or {})
            | {"measurement_ranges": field_rules.get(iid, {})},
        )
    from field_recognition.recognition.remote_ocr import inference_request

    with inference_request():
        return predict(
            local_core,
            panel,
            x,
            y,
            video=video,
            localization_only=localization_only,
            detection=detection,
            allowed_instrument_ids={
                b["instrument"]["id"]
                for b in binding_snapshots
                if b.get("attribution_status") != "needs_review"
            },
        )


def current_readout_binding(_runtime, document):
    if _runtime["stopping"].is_set():
        return False
    if (
        document.get("job_id")
        and document.get("capture_id")
        and (
            document.get("request_trigger") in {"voice_photo_directory", "video_stream"}
            or document.get("binding_time_basis") == "source_capture_time"
        )
    ):
        # The persisted photograph and its binding-at-capture snapshot survive a
        # later device disconnect. Do not discard a requested historical reading.
        return True
    if document.get("binding_id") is None:
        return bool(
            document.get("job_id")
            and document.get("capture_id")
            and document.get("instrument_association")
            in {"unbound_photo", "multiple_candidates", "same_image_qr_unbound"}
        )
    with _runtime["db"]() as conn:
        row = conn.execute(
            "SELECT * FROM bindings WHERE id=?", (document["binding_id"],)
        ).fetchone()
        if not row or row["ended"]:
            return False
        binding = json.loads(row["document"])
        from field_recognition.bindings.binding_policy import contains

        if not contains(binding, _runtime["now"]()):
            return False
        asset = _runtime["get_instrument"](binding["instrument"]["id"])
        if not asset or asset["scene"] != binding["instrument"]["scene"]:
            return False
        visit_id = binding.get("scene_visit_id")
        if (
            visit_id
            and not conn.execute(
                "SELECT 1 FROM scene_visits WHERE id=? AND ended IS NULL", (visit_id,)
            ).fetchone()
        ):
            return False
        service = conn.execute(
            "SELECT document FROM camera_service_state WHERE camera=?", (binding["camera_id"],)
        ).fetchone()
        return not service or json.loads(service["document"])["online"]


def reserve_fallback(_runtime, binding_id):
    with _runtime["db"]() as conn:
        conn.execute("BEGIN IMMEDIATE")
        previous = conn.execute(
            "SELECT attempted_at FROM fallback_attempts WHERE binding_id=?", (binding_id,)
        ).fetchone()
        current = _runtime["time"].time()
        if previous and current - previous["attempted_at"] < 30:
            return False
        conn.execute("INSERT OR REPLACE INTO fallback_attempts VALUES(?,?)", (binding_id, current))
    return True
