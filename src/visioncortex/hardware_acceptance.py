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
HARDWARE_TUNING_SCHEMA = "visioncortex-rtx3090ti-hardware-tuning/1"
_NETWORK_FILESYSTEMS = frozenset(
    {"9p", "cifs", "fuse.sshfs", "nfs", "nfs4", "smb3", "sshfs"}
)


def _mount_filesystem_type(path: Path) -> str:
    try:
        lines = Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise RuntimeError("Cannot prove that hardware tuning media is local") from exc
    matches: list[tuple[int, str]] = []
    for line in lines:
        fields = line.split()
        try:
            separator = fields.index("-")
            mount_value = (
                fields[4]
                .replace("\\040", " ")
                .replace("\\011", "\t")
                .replace("\\012", "\n")
                .replace("\\134", "\\")
            )
            mount_point = Path(mount_value)
            path.relative_to(mount_point)
        except (ValueError, IndexError):
            continue
        matches.append((len(mount_point.parts), fields[separator + 1].casefold()))
    if not matches:
        raise RuntimeError(f"Cannot determine media filesystem type: {path}")
    return max(matches)[1]


def _require_local_media(path: Path) -> Path:
    resolved = path.resolve()
    normalized = resolved.as_posix().casefold()
    forbidden = (
        "/home/x1/桌面/nas",
        "/visioncortexexperimentarchive",
        "/visioncortexexperimentcache",
    )
    if any(marker in normalized for marker in forbidden):
        raise RuntimeError(f"Hardware tuning requires local non-NAS media: {resolved}")
    filesystem_type = _mount_filesystem_type(resolved)
    if filesystem_type in _NETWORK_FILESYSTEMS:
        raise RuntimeError(
            "Hardware tuning requires local non-NAS media: "
            f"{resolved} is on {filesystem_type}"
        )
    return resolved


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
    resolved_media = [_require_local_media(path) for path in media_paths]
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
            "media": [
                {
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "codec": codec,
                }
                for path, codec in zip(resolved_media[:6], codecs, strict=True)
            ],
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


def tune_hardware_acceptance(
    output: Path,
    config: dict[str, Any],
    media_paths: list[Path],
    *,
    duration_seconds: float = 20.0,
    worker_candidates: tuple[int, ...] = (1, 2, 3),
) -> Path:
    """Measure a bounded worker matrix and retain the fastest stable profile."""

    if duration_seconds < 10.0 or duration_seconds > 120.0:
        raise ValueError("Hardware tuning duration must be between 10 and 120 seconds")
    candidates = tuple(dict.fromkeys(worker_candidates))
    if not candidates or any(item < 1 or item > 3 for item in candidates):
        raise ValueError("Hardware tuning workers must be unique values from 1 to 3")
    output = output.resolve()
    receipt_path = output / "hardware-tuning.json"
    if output.exists():
        if receipt_path.is_file():
            existing = json.loads(receipt_path.read_text(encoding="utf-8"))
            if existing.get("passed") is True:
                return receipt_path
            raise RuntimeError(f"Existing hardware tuning did not pass: {receipt_path}")
        raise FileExistsError(f"Hardware tuning output already exists: {output}")
    output.mkdir(parents=True)

    runs: list[dict[str, Any]] = []
    for workers_per_role in candidates:
        run_root = output / f"workers-{workers_per_role}"
        try:
            run_receipt = run_hardware_acceptance(
                run_root,
                config,
                media_paths,
                duration_seconds=duration_seconds,
                workers_per_role=workers_per_role,
            )
            payload = json.loads(run_receipt.read_text(encoding="utf-8"))
            runs.append(
                {
                    "workers_per_role": workers_per_role,
                    "status": "passed",
                    "receipt": str(run_receipt),
                    "aggregate_tensor_rt_frames_per_second": payload["workload"][
                        "aggregate_tensor_rt_frames_per_second"
                    ],
                    "peaks": payload.get("peaks") or {},
                }
            )
        except Exception as exc:
            runs.append(
                {
                    "workers_per_role": workers_per_role,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    passed_runs = [item for item in runs if item["status"] == "passed"]
    if not passed_runs:
        failed_payload = {
            "schema_version": HARDWARE_TUNING_SCHEMA,
            "status": "failed",
            "passed": False,
            "runs": runs,
            "source_copy_bytes": 0,
            "ark_calls": 0,
            "token_usage": 0,
            "nas_accessed": False,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        receipt_path.write_text(
            json.dumps(failed_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        raise RuntimeError(f"Every bounded hardware tuning profile failed: {receipt_path}")
    maximum_fps = max(
        float(item["aggregate_tensor_rt_frames_per_second"])
        for item in passed_runs
    )
    near_maximum = [
        item
        for item in passed_runs
        if float(item["aggregate_tensor_rt_frames_per_second"])
        >= maximum_fps * 0.98
    ]
    selected = min(near_maximum, key=lambda item: item["workers_per_role"])
    baseline = next(
        (item for item in passed_runs if item["workers_per_role"] == 1), None
    )
    baseline_fps = (
        float(baseline["aggregate_tensor_rt_frames_per_second"])
        if baseline is not None
        else None
    )
    selected_fps = float(selected["aggregate_tensor_rt_frames_per_second"])
    gpu_peak = selected["peaks"].get("gpu_compute_percent")
    payload = {
        "schema_version": HARDWARE_TUNING_SCHEMA,
        "status": "completed",
        "passed": True,
        "production_quality_claim": False,
        "selection_rule": (
            "fewest workers within 2% of maximum measured TensorRT throughput"
        ),
        "duration_seconds_per_profile": duration_seconds,
        "worker_candidates": list(candidates),
        "runs": runs,
        "selected_workers_per_role": selected["workers_per_role"],
        "selected_aggregate_tensor_rt_frames_per_second": selected_fps,
        "selected_gpu_compute_peak_percent": gpu_peak,
        "selected_gpu_memory_peak_mib": selected["peaks"].get(
            "gpu_memory_used_mib"
        ),
        "throughput_change_vs_one_worker_percent": (
            round((selected_fps / baseline_fps - 1.0) * 100.0, 3)
            if baseline_fps
            else None
        ),
        "hardware_compute_saturated": gpu_peak is not None and float(gpu_peak) >= 90.0,
        "recommendation": (
            "benchmark capacity only; keep production concurrency unchanged until "
            "ordered multi-view tracking is verified by a real six-view A/B run"
        ),
        "production_configuration_changed": False,
        "deployment_status": "benchmark_only_not_promoted",
        "source_copy_bytes": 0,
        "ark_calls": 0,
        "token_usage": 0,
        "nas_accessed": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    receipt_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "hardware-tuning.md").write_text(
        "# VisionCortex bounded RTX 3090 Ti tuning\n\n"
        f"- Selected workers per role: `{selected['workers_per_role']}`\n"
        f"- Selected TensorRT throughput: `{selected_fps:.3f} frames/s`\n"
        f"- Selected GPU peak: `{gpu_peak}%`\n"
        f"- Selected GPU memory peak: `{selected['peaks'].get('gpu_memory_used_mib')} MiB`\n"
        f"- Candidate profiles: `{list(candidates)}`\n"
        "- Selection: fewest workers within 2% of maximum measured throughput.\n"
        "- Scope: synthetic local six-lane stress; no production quality claim.\n"
        "- NAS/Ark/tokens/source copies: 0.\n",
        encoding="utf-8",
    )
    return receipt_path
