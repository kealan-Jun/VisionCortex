from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .telemetry import ResourceMonitor
from .yolo_calibration import AUDIT_SCHEMA, _load_dataset, _split_pairs


CANDIDATE_RUNTIME_SCHEMA = "visioncortex-yolo-candidate-tensorrt/1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.partial-{uuid.uuid4().hex[:8]}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def _exact_batches(sources: list[str], batch_size: int) -> list[list[str]]:
    """Return only complete fixed-shape TensorRT batches."""

    if batch_size < 1:
        raise ValueError("TensorRT candidate batch must be positive")
    usable = len(sources) // batch_size * batch_size
    return [
        sources[offset : offset + batch_size]
        for offset in range(0, usable, batch_size)
    ]


def export_and_benchmark_yolo_candidate(
    model_path: Path,
    dataset_root: Path,
    audit_receipt_path: Path,
    output: Path,
    *,
    split: str = "val",
    image_size: int = 640,
    export_batch: int = 4,
    benchmark_image_limit: int = 256,
    workspace_gib: float = 3.0,
    device: str = "0",
) -> dict[str, Any]:
    """Export a non-production TensorRT candidate and measure local throughput."""

    model_path = model_path.resolve()
    root, _, classes = _load_dataset(dataset_root)
    audit_receipt_path = audit_receipt_path.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"YOLO candidate runtime output already exists: {output}")
    if not model_path.is_file():
        raise FileNotFoundError(f"YOLO candidate model is missing: {model_path}")
    if (
        split not in {"val", "test"}
        or image_size < 64
        or export_batch < 1
        or benchmark_image_limit < export_batch
        or workspace_gib <= 0.0
    ):
        raise ValueError("YOLO candidate TensorRT limits are invalid")
    audit = json.loads(audit_receipt_path.read_text(encoding="utf-8"))
    dataset_receipt_path = root / "dataset-receipt.json"
    if (
        audit.get("schema_version") != AUDIT_SCHEMA
        or audit.get("passed") is not True
        or audit.get("dataset_receipt_sha256") != _sha256(dataset_receipt_path)
        or int(audit.get("cross_split_content_hash_count") or 0) != 0
        or int(audit.get("source_copy_bytes") or 0) != 0
        or audit.get("nas_accessed") is not False
    ):
        raise RuntimeError("A passing matching dataset integrity audit is required")
    pairs = _split_pairs(root, split)
    sources = [str(image.resolve(strict=True)) for _, image, _ in pairs]
    sources = sources[: min(len(sources), benchmark_image_limit)]
    if len(sources) < export_batch:
        raise RuntimeError("Not enough audited local images for TensorRT benchmark")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.partial-{uuid.uuid4().hex[:8]}")
    temporary.mkdir()
    build_root = Path(
        tempfile.mkdtemp(prefix=f".{output.name}-build-", dir=output.parent)
    )
    staged_model = build_root / "candidate.pt"
    telemetry_path = temporary / "resource-telemetry.json"
    monitor = ResourceMonitor(telemetry_path, interval_seconds=0.25)
    started = datetime.now(timezone.utc)
    monitor.start()
    export_started = time.perf_counter()
    try:
        shutil.copy2(model_path, staged_model)
        from ultralytics import YOLO

        monitor.set_stage("candidate_tensorrt_export")
        exported = Path(
            YOLO(str(staged_model)).export(
                format="engine",
                imgsz=image_size,
                half=True,
                # Keep the benchmark profile spatially static.  Ultralytics'
                # dynamic TensorRT export expands a 960 px request to a
                # 2880 px maximum profile, which can exhaust a 24 GiB GPU
                # during tactic selection even though the requested runtime
                # workload is strictly 960 px.
                dynamic=False,
                batch=export_batch,
                workspace=workspace_gib,
                device=device,
            )
        )
        export_seconds = time.perf_counter() - export_started
        engine_path = temporary / "candidate.engine"
        shutil.move(str(exported), engine_path)

        monitor.set_stage("candidate_tensorrt_benchmark")
        runtime_wrapper_started = time.perf_counter()
        runtime = YOLO(str(engine_path), task="detect")
        runtime_wrapper_seconds = time.perf_counter() - runtime_wrapper_started
        runtime_names = [str(runtime.names[index]) for index in sorted(runtime.names)]
        if runtime_names != classes:
            raise RuntimeError("TensorRT candidate ontology does not match audited dataset")
        rows: list[dict[str, Any]] = []
        # A static TensorRT engine only accepts the exact batch embedded in
        # its input binding.  Benchmarking smaller batches would either fail
        # the backend shape assertion or silently measure padding, neither of
        # which is a truthful runtime result.
        for batch_size in [export_batch]:
            warmup_sources = sources[:batch_size]
            for _ in range(2):
                warmup_results = runtime.predict(
                    source=warmup_sources,
                    stream=True,
                    imgsz=image_size,
                    batch=batch_size,
                    device=device,
                    conf=0.25,
                    save=False,
                    verbose=False,
                )
                if sum(1 for _ in warmup_results) != len(warmup_sources):
                    raise RuntimeError("TensorRT warmup dropped benchmark images")
            benchmark_batches = _exact_batches(sources, batch_size)
            benchmark_source_count = sum(len(chunk) for chunk in benchmark_batches)
            benchmark_started = time.perf_counter()
            result_count = 0
            for benchmark_sources in benchmark_batches:
                results = runtime.predict(
                    source=benchmark_sources,
                    stream=True,
                    imgsz=image_size,
                    batch=batch_size,
                    device=device,
                    conf=0.25,
                    save=False,
                    verbose=False,
                )
                result_count += sum(1 for _ in results)
            elapsed = time.perf_counter() - benchmark_started
            if result_count != benchmark_source_count:
                raise RuntimeError("TensorRT benchmark dropped audited images")
            rows.append(
                {
                    "batch": batch_size,
                    "image_count": result_count,
                    "elapsed_seconds": round(elapsed, 6),
                    "images_per_second": round(result_count / elapsed, 3),
                    "milliseconds_per_image": round(elapsed * 1000.0 / result_count, 3),
                }
            )
    finally:
        telemetry = monitor.stop()
        shutil.rmtree(build_root, ignore_errors=True)
    ended = datetime.now(timezone.utc)
    best = max(rows, key=lambda item: float(item["images_per_second"]))
    try:
        import tensorrt as trt

        tensorrt_version = trt.__version__
    except ImportError:
        tensorrt_version = None
    payload = {
        "schema_version": CANDIDATE_RUNTIME_SCHEMA,
        "status": "completed",
        "production_certified": False,
        "deployment_status": "benchmark_only_not_promoted",
        "production_configuration_changed": False,
        "model": str(model_path),
        "model_sha256": _sha256(model_path),
        "engine": str(output / engine_path.name),
        "engine_sha256": _sha256(engine_path),
        "engine_bytes": engine_path.stat().st_size,
        "dataset_root": str(root),
        "dataset_receipt": str(dataset_receipt_path),
        "dataset_receipt_sha256": _sha256(dataset_receipt_path),
        "dataset_integrity_audit": str(audit_receipt_path),
        "dataset_integrity_audit_sha256": _sha256(audit_receipt_path),
        "benchmark_split": split,
        "benchmark_image_selection": "first_sorted_audited_images",
        "benchmark_image_limit": benchmark_image_limit,
        "export": {
            "format": "TensorRT engine",
            "tensorrt_version": tensorrt_version,
            "image_size": image_size,
            "half": True,
            "dynamic": False,
            "shape_policy": "fixed_spatial_and_export_batch",
            "benchmark_batch_policy": "exact_static_export_batch_only",
            "benchmark_scope": (
                "ultralytics_file_input_end_to_end_including_decode_preprocess_"
                "backend_setup_inference_and_postprocess"
            ),
            "steady_state_inference_only_claimed": False,
            "maximum_batch": export_batch,
            "workspace_gib": workspace_gib,
            "device": device,
            "elapsed_seconds": round(export_seconds, 6),
            "runtime_wrapper_initialization_seconds": round(
                runtime_wrapper_seconds, 6
            ),
        },
        "benchmarks": rows,
        "best_throughput": best,
        "started_at": started.isoformat(),
        "ended_at": ended.isoformat(),
        "elapsed_seconds": (ended - started).total_seconds(),
        "resource_telemetry": str(output / telemetry_path.name),
        "resource_summary": telemetry.get("stage_summaries") or {},
        "telemetry_health": telemetry.get("monitor_health") or {},
        "source_copy_bytes": 0,
        "model_staging_copy_bytes": model_path.stat().st_size,
        "ark_calls": 0,
        "token_usage": 0,
        "nas_accessed": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(temporary / "candidate-runtime-receipt.json", payload)
    os.replace(temporary, output)
    return payload
