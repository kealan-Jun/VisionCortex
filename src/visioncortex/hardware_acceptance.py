from __future__ import annotations

import json
import platform
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2

from .detection import FramePacket, RoleScanner
from .schemas import ViewInput, ViewRole
from .telemetry import ResourceMonitor


HARDWARE_ACCEPTANCE_SCHEMA = "visioncortex-rtx3090ti-hardware-acceptance/1"


def _gpu_inventory() -> dict[str, str]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total,power.limit",
            "--format=csv,noheader,nounits",
            "-i",
            "0",
        ],
        capture_output=True,
        check=True,
        text=True,
        timeout=10,
    )
    values = [item.strip() for item in result.stdout.splitlines()[0].split(",")]
    if len(values) != 5:
        raise RuntimeError("Unexpected nvidia-smi inventory output")
    return dict(
        zip(
            ("name", "uuid", "driver_version", "memory_total_mib", "power_limit_w"),
            values,
            strict=True,
        )
    )


def _video_codec(path: Path) -> str:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        check=True,
        text=True,
        timeout=15,
    )
    return result.stdout.strip()


def _start_decode_process(path: Path, seconds: float) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-stream_loop",
            "-1",
            "-hwaccel",
            "cuda",
            "-i",
            str(path),
            "-t",
            f"{seconds:.3f}",
            "-map",
            "0:v:0",
            "-f",
            "null",
            "-",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )


def _packets(path: Path, role: ViewRole, count: int = 4) -> list[FramePacket]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open hardware acceptance video: {path}")
    packets = []
    previous_gray = None
    try:
        for frame_index in range(count):
            ok, frame = capture.read()
            if not ok:
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(f"Cannot decode benchmark frame: {path}")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            packets.append(
                FramePacket(
                    view=ViewInput(
                        view_id=f"benchmark-{role.value}", role=role, video=path
                    ),
                    frame_index=frame_index,
                    local_ms=frame_index * 100.0,
                    frame=frame,
                    gray=gray,
                    previous_gray=previous_gray,
                    motion_score=1.0,
                )
            )
            previous_gray = gray
    finally:
        capture.release()
    return packets


def _inference_worker(
    role: ViewRole,
    replica: int,
    path: Path,
    config: dict[str, Any],
    deadline: float,
    stop: threading.Event,
) -> dict[str, Any]:
    scanner = RoleScanner(role, config, batch_size=4)
    packets = _packets(path, role)
    batches = 0
    frames = 0
    detections = 0
    started = time.perf_counter()
    try:
        while time.perf_counter() < deadline and not stop.is_set():
            result = scanner.infer(packets)
            batches += 1
            frames += len(packets)
            detections += sum(len(item) for item in result)
        elapsed = time.perf_counter() - started
        return {
            "role": role.value,
            "replica": replica,
            "backend": "TensorRT",
            "engine": str(scanner.model_path),
            "requested_batch_size": scanner.requested_batch_size,
            "engine_build_batch": scanner.engine_build_batch,
            "effective_batch_size": scanner.batch_size,
            "batch_contractions": scanner.batch_contractions,
            "batches": batches,
            "frames": frames,
            "detections": detections,
            "elapsed_seconds": round(elapsed, 6),
            "frames_per_second": round(frames / max(elapsed, 1e-9), 3),
        }
    finally:
        scanner.close()


def _peak(telemetry: dict[str, Any], key: str) -> float | None:
    values = [
        sample.get("gpu", {}).get(key)
        for sample in telemetry.get("samples") or []
    ]
    filtered = [float(value) for value in values if value is not None]
    return round(max(filtered), 3) if filtered else None


def run_hardware_acceptance(
    output: Path,
    config: dict[str, Any],
    media_paths: list[Path],
    *,
    duration_seconds: float = 60.0,
    workers_per_role: int = 1,
) -> Path:
    """Stress six CUDA decode lanes and both real TensorRT role engines."""

    if duration_seconds < 10 or duration_seconds > 300:
        raise ValueError("Hardware acceptance duration must be between 10 and 300 seconds")
    if workers_per_role < 1 or workers_per_role > 3:
        raise ValueError("Hardware acceptance workers_per_role must be between 1 and 3")
    resolved_media = [path.resolve() for path in media_paths]
    if len(resolved_media) < 6 or any(not path.is_file() for path in resolved_media):
        raise RuntimeError("Hardware acceptance requires six existing local video files")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    receipt_path = output / "hardware-acceptance.json"
    if receipt_path.is_file():
        existing = json.loads(receipt_path.read_text(encoding="utf-8"))
        if existing.get("passed") is True:
            return receipt_path
        raise RuntimeError(f"Existing hardware acceptance did not pass: {receipt_path}")

    inventory = _gpu_inventory()
    if "RTX 3090 Ti" not in inventory["name"]:
        raise RuntimeError(f"Expected RTX 3090 Ti, got {inventory['name']}")
    codecs = [_video_codec(path) for path in resolved_media[:6]]
    if any(codec not in {"h264", "hevc"} for codec in codecs):
        raise RuntimeError(f"NVDEC acceptance requires H.264/HEVC inputs: {codecs}")

    monitor = ResourceMonitor(output / "resource-telemetry.json", interval_seconds=0.25)
    monitor.set_stage("six_lane_nvdec_plus_dual_tensorrt")
    monitor.start()
    started = time.perf_counter()
    decode_processes: list[subprocess.Popen[str]] = []
    stop = threading.Event()
    try:
        decode_processes = [
            _start_decode_process(path, duration_seconds)
            for path in resolved_media[:6]
        ]
        deadline = time.perf_counter() + duration_seconds
        role_workers = [
            (role, replica)
            for role in (ViewRole.FIRST_PERSON, ViewRole.THIRD_PERSON)
            for replica in range(workers_per_role)
        ]
        with ThreadPoolExecutor(max_workers=len(role_workers)) as executor:
            futures = [
                executor.submit(
                    _inference_worker,
                    role,
                    replica,
                    resolved_media[index % len(resolved_media)],
                    config,
                    deadline,
                    stop,
                )
                for index, (role, replica) in enumerate(role_workers)
            ]
            inference = [future.result() for future in futures]
        decode_failures = []
        for index, process in enumerate(decode_processes):
            stderr = process.communicate(timeout=max(30.0, duration_seconds + 10))[1]
            if process.returncode != 0:
                decode_failures.append(
                    {"lane": index, "returncode": process.returncode, "error": stderr[-1000:]}
                )
        if decode_failures:
            raise RuntimeError(f"CUDA decode lane failure: {decode_failures}")
    finally:
        stop.set()
        for process in decode_processes:
            if process.poll() is None:
                process.terminate()
        telemetry = monitor.stop()

    elapsed = time.perf_counter() - started
    gpu_peak = _peak(telemetry, "utilization.gpu")
    gpu_memory_peak = _peak(telemetry, "memory.used")
    nvdec_peak = _peak(telemetry, "utilization.decoder")
    power_peak = _peak(telemetry, "power.draw")
    total_fps = round(sum(item["frames"] for item in inference) / elapsed, 3)
    passed = bool(
        elapsed >= duration_seconds * 0.90
        and all(item["frames"] > 0 for item in inference)
        and all(not item["batch_contractions"] for item in inference)
        and gpu_peak is not None
    )
    bottleneck = (
        "gpu_compute_saturated_or_near_saturated"
        if gpu_peak is not None and gpu_peak >= 90
        else "host_or_decode_feed_limited"
    )
    payload = {
        "schema_version": HARDWARE_ACCEPTANCE_SCHEMA,
        "status": "completed",
        "passed": passed,
        "synthetic_local_media": True,
        "production_quality_claim": False,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "host": {"platform": platform.platform(), "gpu": inventory},
        "workload": {
            "duration_target_seconds": duration_seconds,
            "elapsed_seconds": round(elapsed, 6),
            "cuda_decode_lanes": 6,
            "video_codecs": codecs,
            "workers_per_role": workers_per_role,
            "tensor_rt_role_workers": 2 * workers_per_role,
            "inference": inference,
            "aggregate_tensor_rt_frames_per_second": total_fps,
        },
        "peaks": {
            "gpu_compute_percent": gpu_peak,
            "gpu_memory_used_mib": gpu_memory_peak,
            "nvdec_percent": nvdec_peak,
            "gpu_power_w": power_peak,
        },
        "bottleneck": bottleneck,
        "source_copy_bytes": 0,
        "ark_calls": 0,
        "token_usage": 0,
        "nas_accessed": False,
        "telemetry_path": str(output / "resource-telemetry.json"),
    }
    receipt_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "hardware-acceptance.md").write_text(
        "# VisionCortex RTX 3090 Ti local hardware acceptance\n\n"
        f"- Passed: `{passed}`\n"
        f"- Workload: 6 CUDA decode lanes + {2 * workers_per_role} TensorRT role workers\n"
        f"- Duration: {elapsed:.3f} s\n"
        f"- TensorRT throughput: {total_fps:.3f} frames/s\n"
        f"- GPU peak: {gpu_peak}%\n"
        f"- GPU memory peak: {gpu_memory_peak} MiB\n"
        f"- NVDEC peak: {nvdec_peak}%\n"
        f"- Power peak: {power_peak} W\n"
        f"- Bottleneck: `{bottleneck}`\n"
        "- Scope: synthetic local hardware stress; no production quality claim.\n"
        "- NAS/Ark/tokens/source copies: 0.\n",
        encoding="utf-8",
    )
    if not passed:
        raise RuntimeError(f"Hardware acceptance failed: {receipt_path}")
    return receipt_path
