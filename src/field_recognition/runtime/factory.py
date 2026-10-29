"""Construct a fresh runtime without starting services or loading OCR models."""

from __future__ import annotations

import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import field_recognition.recognition.aliyun_vision as aliyun_vision
from fastapi import APIRouter
from PIL import Image

from field_recognition.api import catalog, dashboard, models, readouts
from field_recognition.bindings import binding_actions
from field_recognition.capture import capture_actions
from field_recognition.recognition import service
from field_recognition.recognition.panel_detector import PanelDetector
from field_recognition.recognition.remote_ocr import RemoteOcrConfig
from field_recognition.runtime.bootstrap import initialize_database
from field_recognition.runtime.components import install_components
from field_recognition.runtime.context import RuntimeContext
from field_recognition.runtime.database import Database
from field_recognition.runtime.lifecycle import lifespan
from field_recognition.runtime.paths import PROJECT_ROOT
from field_recognition.runtime.production_config import ProductionConfig
from field_recognition.runtime.runtime_rpc import CameraProxy, DetectorProxy, call
from field_recognition.runtime.task_queue import TaskQueue


def now():
    return datetime.now(timezone.utc).isoformat()


def _remote_panel(runtime, panel, x=0, y=0):
    return call("ocr", "predict_panel", panel, x, y, data_dir=runtime.data_dir)


def _remote_readout(runtime, panel, x=0, y=0, **kwargs):
    return call("ocr", "predict_readout", panel, x, y, data_dir=runtime.data_dir, **kwargs)


def _remote_warmup(runtime):
    try:
        call("ocr", "warmup", data_dir=runtime.data_dir)
        runtime["ocr_state"].update(call("ocr", "status", data_dir=runtime.data_dir))
    except Exception:
        runtime["ocr_state"].update(status="worker_unavailable", resident=False)


def _queue_remote_warmup(runtime):
    with runtime["ocr_warmup_lock"]:
        future = runtime["ocr_warmup_future"]
        if future is None or future.done():
            runtime["ocr_warmup_future"] = runtime["ocr_pool"].submit(runtime["remote_warmup"])
        return runtime["ocr_warmup_future"]


def _bind_actions(runtime):
    runtime.bind(
        instrument_record=catalog.instrument_record,
        get_instrument=catalog.get_instrument,
        devices=catalog.devices,
        edit_instrument=catalog.edit_instrument,
        index=dashboard.index,
        state=dashboard.state,
        register_qr_identity=binding_actions.register_qr_identity,
        validate_capture_epoch=binding_actions.validate_capture_epoch,
        bind=binding_actions.bind,
        save_binding=binding_actions.save_binding,
        persist_binding=binding_actions.persist_binding,
        end_binding=binding_actions.end_binding,
        finish_binding=binding_actions.finish_binding,
        end_relations=binding_actions.end_relations,
        save_image=capture_actions.save_image,
        qr_matches=capture_actions.qr_matches,
        scan_image=capture_actions.scan_image,
        save_camera_scan=capture_actions.save_camera_scan,
        scan=capture_actions.scan,
        camera_capture=capture_actions.camera_capture,
        image_file=capture_actions.image_file,
        create_ocr_model=service.create_ocr_model,
        load_ocr=service.load_ocr,
        warm_ocr=service.warm_ocr,
        queue_ocr_warmup=service.queue_ocr_warmup,
        predict_panel=service.predict_panel,
        run_ocr=service.run_ocr,
        current_panel_bindings=service.current_panel_bindings,
        predict_readout=service.predict_readout,
        current_readout_binding=service.current_readout_binding,
        reserve_fallback=service.reserve_fallback,
        submit_ocr=readouts.submit_ocr,
        enqueue_ocr=readouts.enqueue_ocr,
        get_job=readouts.get_job,
        export=readouts.export,
        get_measurements=readouts.get_measurements,
        readouts=readouts.readouts,
        workbenches=readouts.workbenches,
        archive_file=readouts.archive_file,
    )


def create_runtime(
    config: ProductionConfig | None = None,
    *,
    project_root: Path | None = None,
    data_dir: Path | None = None,
    service_role: str | None = None,
    production_enabled: bool | None = None,
) -> RuntimeContext:
    """Prepare an instance; startup happens only through its explicit lifecycle.

    A worker needs this factory alone; ``create_app(runtime)`` separately adds
    the HTTP transport and security. Configuration is captured at construction.
    """
    config = config if config is not None else ProductionConfig.from_environment()
    root = Path(project_root or PROJECT_ROOT).expanduser().resolve()
    role = service_role if service_role is not None else config.service_role
    if role not in {"combined", "api", "capture", "ocr", "archive"}:
        raise ValueError("FIELD_SERVICE_ROLE must be combined, api, capture, ocr or archive")
    enabled = config.production_enabled if production_enabled is None else production_enabled
    if (config.receiver_url or config.camera_snapshot_url) and not (config.camera_id or "").strip():
        raise ValueError(
            "FIELD_CAMERA_ID is required when FIELD_RECEIVER_URL or FIELD_CAMERA_SNAPSHOT_URL is configured"
        )
    database_path = Path(config.database_path).expanduser()
    if not database_path.is_absolute():
        database_path = root / database_path
    data = Path(data_dir or config.data_dir or database_path.parent).expanduser()
    if not data.is_absolute():
        data = root / data
    data = data.resolve()
    if (
        data_dir is not None or config.data_dir is not None
    ) and config.database_path == "Data/Demo.sqlite3":
        database_path = data / "Demo.sqlite3"
    config = config.model_copy(
        update={
            "database_path": str(database_path),
            "data_dir": str(data),
            "service_role": role,
            "production_enabled": enabled,
        }
    )
    device = config.ocr_device.strip().lower()
    if not re.fullmatch(r"cpu|gpu:\d+", device):
        raise ValueError("FIELD_OCR_DEVICE must be cpu or gpu:<index>, for example gpu:0")
    data.mkdir(parents=True, exist_ok=True)
    (data / "Images").mkdir(exist_ok=True)
    Image.MAX_IMAGE_PIXELS = 16_000_000
    runtime = RuntimeContext(config, root)
    runtime.update(
        DATA=data,
        OCR_DEVICE=device,
        RECORD_MODE=config.record_mode,
        RUNTIME_ENABLED=enabled,
        AUTH_ENABLED=config.auth_enabled,
        SERVICE_ROLE=role,
        MAX_BYTES=12 * 1024 * 1024,
        DEVICE_TYPE_NAMES={"builtin-stirrer-v1": "搅拌器", "builtin-balance-v1": "质量测量仪"},
        database=Database(database_path),
        now=now,
        app=APIRouter(),
        json=json,
        time=time,
        aliyun_vision=aliyun_vision,
        MeasurementRange=models.MeasurementRange,
        InstrumentEdit=models.InstrumentEdit,
        BindingRequest=models.BindingRequest,
        OcrRequest=models.OcrRequest,
        EndRelations=models.EndRelations,
    )
    runtime["db"] = runtime.database.connection
    initialize_database(runtime)
    runtime.update(
        ocr_pool=ThreadPoolExecutor(max_workers=1, thread_name_prefix="panel-ocr"),
        readout_pool=ThreadPoolExecutor(max_workers=4, thread_name_prefix="panel-request"),
        vision_pool=ThreadPoolExecutor(max_workers=4, thread_name_prefix="panel-review"),
        vision_call_lock=threading.Lock(),
        ocr_submit_lock=threading.RLock(),
        stopping=threading.Event(),
        ocr_lock=threading.RLock(),
        ocr_warmup_lock=threading.Lock(),
        ocr_warmup_future=None,
        ocr_retry_at=0.0,
        ocr_load_failures=0,
        ocr_model=None,
        ocr_state={
            "status": "not_loaded",
            "device": device,
            "engine": "PaddleOCR / PP-OCRv5 mobile",
            "resident": False,
            "load_count": 0,
            "loaded_at": None,
        },
        panel_detector=DetectorProxy(data_dir=data)
        if role in {"api", "capture"}
        else PanelDetector(),
        receiver_camera=None,
        ocr_service_config=(
            RemoteOcrConfig.from_environment(service_url=config.recognition.ocr_service_url)
            if config.recognition.ocr_service_url
            else None
        ),
    )
    if role == "api":
        runtime["receiver_camera"] = CameraProxy(
            config.camera_id or "UnconfiguredNeckCamera", data_dir=data
        )
    elif role not in {"ocr", "archive"} and config.receiver_url:
        from field_recognition.capture.receiver import ReceiverCamera

        runtime["receiver_camera"] = ReceiverCamera(config.receiver_url, config.camera_id)
    _bind_actions(runtime)
    runtime["lifespan"] = partial(lifespan, runtime)
    runtime["task_queue"] = TaskQueue(runtime.database)
    if role in {"api", "capture"}:
        runtime.bind(
            predict_panel=_remote_panel,
            predict_readout=_remote_readout,
            remote_warmup=_remote_warmup,
            queue_ocr_warmup=_queue_remote_warmup,
        )
    try:
        install_components(runtime)
    except Exception:
        runtime.close()
        raise
    return runtime
