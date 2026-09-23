"""Build bounded-shape TensorRT candidates and time real-frame product inference.

Artifacts stay under the supplied output directory. This never changes a live
engine, downloads a model, calls a provider, or claims experiment accuracy.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from copy import deepcopy
import hashlib
import json
import shutil
import struct
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fresh_output_directory(path: Path) -> Path:
    """Never resume a candidate from someone else's checkpoint or receipt."""
    output = path.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"Benchmark output directory must be empty: {output}")
    return output


def comparison_identity(config: dict, manifest: Path) -> dict:
    """Pin comparison semantics while allowing only engine/batch replacements."""
    normalized = deepcopy(config)
    for key in ("batch_size", "fine_batch_size", "coarse_batch_size"):
        normalized.get("performance", {}).pop(key, None)
    models = normalized.get("models", {})
    for key in list(models):
        if key.endswith("_engine"):
            del models[key]
    for key in ("local_cache_root", "local_runtime_root"):
        normalized.get("storage", {}).pop(key, None)
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False).encode("utf-8")
    repository = Path(__file__).resolve().parents[1]
    commit = subprocess.run(["git", "-C", str(repository), "rev-parse", "HEAD"],
                            check=True, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(repository), "status", "--porcelain",
                            "--untracked-files=all"], check=True,
                           capture_output=True, text=True).stdout
    return {"code_sha": commit, "git_dirty": bool(dirty),
            "config_sha256": hashlib.sha256(encoded).hexdigest(),
            "manifest_sha256": sha256(manifest)}


def require_requested_batch(scanner, requested: int) -> None:
    if not scanner.engine_build_batch or scanner.engine_build_batch < requested:
        raise ValueError(f"Requested batch {requested} exceeds verified engine capacity "
                         f"{scanner.engine_build_batch}; candidate capacity is NOT_PROVEN")
    if scanner.batch_size < requested:
        raise ValueError(f"Requested batch {requested} contracted to {scanner.batch_size}; "
                         "candidate capacity is NOT_PROVEN")


def record_engine_batches(histogram: Counter, scanner) -> None:
    sizes = scanner.last_engine_batch_sizes
    if not sizes or any(type(size) is not int or size < 1 for size in sizes):
        raise ValueError("Actual engine batch sizes are missing; capacity is NOT_PROVEN")
    histogram.update(sizes)


def model_identities(config: dict) -> dict:
    """Snapshot weights, engine bytes and any recorded candidate-build origin."""
    identities = {}
    for key, value in config["models"].items():
        if not value or (key not in ("first_person", "third_person") and not key.endswith("_engine")):
            continue
        path = Path(value).resolve(strict=True)
        before = path.stat()
        identity = {"path": str(path), "sha256": sha256(path)}
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"Model identity changed while reading: {key}")
        if key.endswith("_engine"):
            sidecar = path.with_suffix(".build.json")
            if sidecar.exists():
                with sidecar.open("rb") as handle:
                    encoded = handle.read(1024 * 1024 + 1)
                if len(encoded) > 1024 * 1024:
                    raise ValueError(f"Engine build receipt exceeds 1 MiB: {key}")
                identity["build_receipt"] = json.loads(encoded)
                identity["build_receipt_sha256"] = hashlib.sha256(encoded).hexdigest()
        identities[key] = identity
    return identities


def verify_model_identities(config: dict, expected: dict) -> None:
    if model_identities(config) != expected:
        raise RuntimeError("Model, engine or build receipt changed during the scan; comparison is NOT_PROVEN")


def build(args: argparse.Namespace) -> None:
    root = fresh_output_directory(args.output)
    import tensorrt as trt

    destination = root / f"batch-{args.batch}.engine"
    source = args.weights.resolve(strict=True)
    weights_sha256 = sha256(source)
    copied = root / "source.pt"
    shutil.copy2(source, copied)
    if sha256(copied) != weights_sha256:
        raise ValueError("Frozen weights differ from requested weights")
    onnx = copied.with_suffix(".onnx")
    if onnx.exists():
        raise FileExistsError("Candidate builds cannot reuse an existing ONNX graph")
    # CPU export changes CUDA_VISIBLE_DEVICES in Ultralytics; isolate it
    # so the TensorRT builder still sees the original GPU environment.
    script = ("""import sys
from ultralytics import YOLO
from ultralytics.engine.exporter import Exporter
class GraphOnly(Exporter):
    def export_engine(self, *args, **kwargs):
        return self.export_onnx()
GraphOnly(overrides=dict(format='engine', imgsz=int(sys.argv[2]), batch=4,
    dynamic=True, half=True, simplify=True, device=0))(model=YOLO(sys.argv[1], task='detect').model)
""" if args.export_like_reference else
        "import sys; from ultralytics import YOLO; "
        "YOLO(sys.argv[1], task='detect').export(format='onnx', "
        "imgsz=int(sys.argv[2]), batch=1, dynamic=True, simplify=False, "
        "opset=17, device='cpu')")
    subprocess.run([sys.executable, "-c", script, str(copied), str(args.image_size)], check=True)
    with args.reference_engine.open("rb") as handle:
        length = struct.unpack("<I", handle.read(4))[0]
        if not 0 < length < 1024 * 1024:
            raise ValueError("Reference engine has no bounded metadata header")
        metadata = json.loads(handle.read(length))
    import onnx as onnx_module

    exported_metadata = {p.key: p.value for p in onnx_module.load(onnx, load_external_data=False).metadata_props}
    names = {str(k): v for k, v in ast.literal_eval(exported_metadata["names"]).items()}
    if names != metadata.get("names"):
        raise ValueError("Exported class identities differ from the reference engine")
    metadata["args"] = ast.literal_eval(exported_metadata["args"])
    metadata["end2end"] = exported_metadata.get("end2end", "False") == "True"
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx)):
        raise RuntimeError("ONNX parse failed: " + "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(args.workspace_gib * 1024**3))
    timing_path = root / "tactics.cache"
    cache = config.create_timing_cache(timing_path.read_bytes() if timing_path.exists() else b"")
    config.set_timing_cache(cache, ignore_mismatch=False)
    profile = builder.create_optimization_profile()
    # The end-to-end detector's TopK needs at least 300 anchors. A 32px
    # minimum is invalid even when all actual inference images are larger.
    optimal_batch = args.optimal_batch or args.batch
    shapes = ((1, 3, 160, 160), (optimal_batch, 3, args.image_size, args.image_size),
              (args.batch, 3, args.image_size, args.image_size))
    if network.num_inputs != 1:
        raise ValueError("Only a single image input is supported")
    profile.set_shape(network.get_input(0).name, *shapes)
    if not profile:
        raise ValueError("Invalid optimization profile")
    config.add_optimization_profile(profile)
    started = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT build failed")
    timing_path.write_bytes(bytes(config.get_timing_cache().serialize()))
    metadata.update(batch=args.batch, dynamic=True, imgsz=[args.image_size, args.image_size])
    metadata.setdefault("args", {}).update(batch=args.batch, dynamic=True, half=True)
    header = json.dumps(metadata).encode()
    if sha256(source) != weights_sha256 or sha256(copied) != weights_sha256:
        raise ValueError("Source or frozen weights changed during the build; candidate will not be published")
    with destination.open("xb") as handle:
        handle.write(struct.pack("<I", len(header)))
        handle.write(header)
        handle.write(serialized)
    receipt = {"schema_version": "visioncortex-engine-capacity-build/1", "engine": str(destination),
               "weights_sha256": weights_sha256, "onnx_sha256": sha256(onnx),
               "engine_sha256": sha256(destination), "tensorrt": trt.__version__,
               "min_opt_max_shapes": shapes, "workspace_gib": args.workspace_gib,
               "build_seconds": time.perf_counter() - started,
               "quality_evidence": "NOT_PROVEN"}
    destination.with_suffix(".build.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt), flush=True)


def infer(args: argparse.Namespace) -> None:
    import cv2
    import torch
    from visioncortex.config import load_config, load_manifest
    from visioncortex.detection import FramePacket, RoleScanner
    from visioncortex.schemas import ViewRole
    from visioncortex.telemetry import ResourceMonitor

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    receipt_path = output / "result.json"
    if receipt_path.exists():
        raise FileExistsError(receipt_path)
    config = load_config(args.config)
    role = ViewRole(args.role)
    config["models"][f"{role.value}_engine"] = str(args.engine.resolve(strict=True))
    config["performance"]["batch_size"] = args.batch
    manifest = load_manifest(args.manifest)
    views = [v for v in manifest.views if v.role == role]
    packets, identities = [], []
    for view in views:
        if view.video is None:
            raise ValueError("Use a continuous real-video manifest for engine microbenchmarks")
        capture = cv2.VideoCapture(str(view.video))
        if not capture.isOpened():
            raise RuntimeError(f"Cannot decode {view.video}")
        try:
            fps = capture.get(cv2.CAP_PROP_FPS)
            count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
            duration = count / fps
            for index in range(args.samples):
                target = (index + .5) * duration / args.samples
                capture.set(cv2.CAP_PROP_POS_MSEC, target * 1000)
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError("Real benchmark frame decode failed")
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                packets.append(FramePacket(view=view, frame_index=index,
                    local_ms=capture.get(cv2.CAP_PROP_POS_MSEC), frame=frame, gray=gray,
                    previous_gray=None, motion_score=0))
                identities.append({"view_id": view.view_id, "requested_ms": target * 1000,
                    "decoded_ms": capture.get(cv2.CAP_PROP_POS_MSEC),
                    "pixel_sha256": hashlib.sha256(frame.tobytes()).hexdigest()})
        finally:
            capture.release()
    scanner = RoleScanner(role, config, image_size=args.image_size, batch_size=args.batch)
    monitor = ResourceMonitor(output / "telemetry.json", .5)
    monitor.start()
    try:
        require_requested_batch(scanner, args.batch)
        batch = [packets[i % len(packets)] for i in range(args.batch)]
        for _ in range(5):
            scanner.infer(batch)
        require_requested_batch(scanner, args.batch)
        torch.cuda.synchronize()
        start = time.perf_counter()
        frames, latencies, offset = 0, [], 0
        engine_batches = Counter()
        while time.perf_counter() - start < args.seconds:
            batch = [packets[(offset + i) % len(packets)] for i in range(args.batch)]
            before = time.perf_counter()
            scanner.infer(batch)
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - before)
            record_engine_batches(engine_batches, scanner)
            frames += len(batch)
            offset += len(batch)
        elapsed = time.perf_counter() - start
        timed_effective_batch = scanner.batch_size
        requested_batch_reached = bool(engine_batches) and (
            min(engine_batches) >= args.batch and timed_effective_batch >= args.batch
        )
        predictions = scanner.infer(packets)
        quality = [[box.model_dump(mode="json") for box in boxes] for boxes in predictions]
        receipt = {"schema_version": "visioncortex-engine-capacity-inference/1",
            "scope": "Predecoded real-frame product inference; excludes video IO and complete preprocessing",
            "role": args.role, "engine_sha256": sha256(args.engine),
            "requested_batch": args.batch, "effective_batch": scanner.batch_size,
            "engine_capacity": scanner.engine_build_batch, "contractions": scanner.batch_contractions,
            "timed_effective_batch": timed_effective_batch,
            "timed_engine_batch_histogram": dict(sorted(engine_batches.items())),
            "timed_engine_batch_histogram_scope": "Successful timed engine invocations only; excludes warmup and quality prediction",
            "requested_batch_reached": requested_batch_reached,
            "capacity_evidence": "PROVEN" if requested_batch_reached else "NOT_PROVEN",
            "frames": frames, "seconds": elapsed, "frames_per_second": frames / elapsed,
            "batch_latency_ms_mean": sum(latencies) / len(latencies) * 1000,
            "frame_identities": identities, "predictions": quality,
            "quality_evidence": "PARTIAL_EVIDENCE; comparison required; no human ground truth"}
        receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2))
        print(json.dumps({k: v for k, v in receipt.items() if k not in {"frame_identities", "predictions"}}), flush=True)
        if not requested_batch_reached:
            raise RuntimeError("Timed inference did not sustain the requested batch; see result.json")
    finally:
        monitor.stop()
        scanner.close()


def movement(args: argparse.Namespace) -> None:
    import cv2
    from visioncortex.config import load_config, load_manifest
    from visioncortex.movement_verification import verify_movement_candidates
    from visioncortex.schemas import ActionCandidate, VideoInfo

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "result.json").exists():
        raise FileExistsError(output / "result.json")
    config = load_config(args.config)
    config["segmentation"]["movement_visual_verification"].update(
        workers=args.workers, frame_cache_size=args.cache_frames)
    config_path = args.baseline / "JSON-Config-Files"
    candidates = [ActionCandidate.model_validate(c) for c in json.loads(
        (config_path / "candidate_layer.json").read_text())]
    infos = {k: VideoInfo.model_validate(v) for k, v in json.loads(
        (config_path / "video_probe.json").read_text()).items()}
    manifest = load_manifest(args.manifest)
    identity = json.loads((config_path / "cache_identity.json").read_text())
    detections = (Path(config["storage"]["local_cache_root"]) / manifest.experiment_id
                  / identity["cache_key"] / "detections-fine/indexed-detections")
    cv2.setNumThreads(args.opencv_threads)
    result = verify_movement_candidates(candidates, manifest.views, infos,
        {v.view_id: detections / f"{v.view_id}.detections.jsonl" for v in manifest.views}, config)
    expected = json.loads((config_path / "movement_visual_verification.json").read_text())
    result["comparison"] = {"candidate_receipts_equal": result["candidates"] == expected["candidates"],
        "counts_equal": result.get("counts") == expected.get("counts"),
        "baseline_seconds": expected["duration_seconds"], "opencv_threads": cv2.getNumThreads(),
        "workers_requested": args.workers, "cache_frames": args.cache_frames,
        "source_receipt_sha256": sha256(config_path / "movement_visual_verification.json")}
    (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in {"candidates", "policy"}}, ensure_ascii=False), flush=True)


def concurrent(args: argparse.Namespace) -> None:
    """Measure actual product contexts on replayed real frames, not ten sources."""
    import cv2
    from visioncortex.config import load_config, load_manifest
    from visioncortex.detection import FramePacket, RoleScanner
    from visioncortex.schemas import ViewRole
    from visioncortex.telemetry import ResourceMonitor
    from ultralytics.utils import ops

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "result.json").exists():
        raise FileExistsError(output / "result.json")
    config = load_config(args.config)
    for role_name in ("first_person", "third_person"):
        override = getattr(args, f"{role_name}_engine", None)
        if override is not None:
            config["models"][f"{role_name}_engine"] = str(override.resolve(strict=True))
    manifest = load_manifest(args.manifest)
    packets, identities = {}, []
    for role in (ViewRole.FIRST_PERSON, ViewRole.THIRD_PERSON):
        view = next(v for v in manifest.views if v.role == role and v.video is not None)
        capture = cv2.VideoCapture(str(view.video))
        if not capture.isOpened():
            raise RuntimeError(f"Cannot decode {view.view_id}")
        packets[role] = []
        try:
            duration = capture.get(cv2.CAP_PROP_FRAME_COUNT) / capture.get(cv2.CAP_PROP_FPS)
            for index in range(16):
                capture.set(cv2.CAP_PROP_POS_MSEC, (index + .5) * duration / 16 * 1000)
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError("Real frame decode failed")
                packets[role].append(FramePacket(view=view, frame_index=index,
                    local_ms=capture.get(cv2.CAP_PROP_POS_MSEC), frame=frame,
                    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), previous_gray=None, motion_score=0))
                identities.append({"view_id": view.view_id, "frame_index": index,
                    "decoded_ms": capture.get(cv2.CAP_PROP_POS_MSEC),
                    "pixel_sha256": hashlib.sha256(frame.tobytes()).hexdigest()})
        finally:
            capture.release()
    gpu_command = ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader"]
    gpu_before = subprocess.run(gpu_command, capture_output=True, text=True).stdout
    monitor = ResourceMonitor(output / "telemetry.json", .5)
    monitor.set_stage("warmup")
    monitor.start()
    scanners, references, resident_inputs = [], {}, {}
    original_profiler = ops.Profile
    try:
        for index in range(args.contexts):
            role = ViewRole.FIRST_PERSON if index % 2 == 0 else ViewRole.THIRD_PERSON
            scanner = RoleScanner(role, config, image_size=640, batch_size=args.batch)
            scanners.append((role, scanner))
            quality = [[b.model_dump(mode="json") for b in boxes] for boxes in scanner.infer(packets[role])]
            references.setdefault(role, quality)
            if quality != references[role]:
                raise RuntimeError("Sequential contexts disagree before the timed measurement")
            if args.gpu_resident:
                if args.batch > scanner.batch_size:
                    raise ValueError("Resident benchmark batch exceeds the engine capacity")
                predictor = scanner.model.predictor
                tensor = predictor.preprocess([packets[role][i % 16].frame for i in range(args.batch)])
                resident_inputs[id(scanner)] = tensor
                predictor.inference(tensor)
        started = []

        def begin():
            monitor.set_stage("concurrent_inference")
            started.append(time.perf_counter())

        barrier = Barrier(args.contexts, action=begin)

        if args.host_only_profiler:
            # Isolated process experiment: change timing instrumentation only.
            # TensorRT execute_v2 and output CPU transfers still complete work.
            # Warm-up references above were obtained with the original profiler.
            class HostOnlyProfile(original_profiler):
                def time(self):
                    return time.perf_counter()

            ops.Profile = HostOnlyProfile

        def run(item):
            role, scanner = item
            frames = 0
            barrier.wait(timeout=60)
            while time.perf_counter() - started[0] < args.seconds:
                if args.gpu_resident:
                    scanner.model.predictor.inference(resident_inputs[id(scanner)])
                else:
                    batch = [packets[role][(frames + i) % 16] for i in range(args.batch)]
                    scanner.infer(batch)
                frames += args.batch
            if args.gpu_resident:
                import torch

                torch.cuda.synchronize()
            ended = time.perf_counter()
            predictions = [[b.model_dump(mode="json") for b in boxes] for boxes in scanner.infer(packets[role])]
            return {"role": role.value, "frames": frames, "ended": ended,
                "effective_batch": scanner.batch_size, "contractions": scanner.batch_contractions,
                "prediction_receipts_equal": predictions == references[role]}

        with ThreadPoolExecutor(max_workers=args.contexts) as executor:
            results = list(executor.map(run, scanners))
        elapsed = max(row.pop("ended") for row in results) - started[0]
        receipt = {"schema_version": "visioncortex-product-concurrency-capacity/1",
            "scope": (
                "GPU-resident real-frame engine benchmark; excludes preprocessing, uploads, postprocessing and source decoding. NOT application throughput."
                if args.gpu_resident else
                "Replayed predecoded real frames; excludes simultaneous source decoding. NOT ten independent experiments."),
            "contexts": args.contexts, "requested_batch": args.batch,
            "gpu_resident_engine_only": args.gpu_resident,
            "host_only_profiler_experiment": args.host_only_profiler,
            "seconds": elapsed, "frames": sum(r["frames"] for r in results),
            "frames_per_second": sum(r["frames"] for r in results) / elapsed,
            "workers": results, "frame_identities": identities, "gpu_before": gpu_before,
            "reference_predictions": {role.value: values for role, values in references.items()},
            "engine_sha256": {role: sha256(Path(config["models"][f"{role}_engine"]))
                for role in ("first_person", "third_person")},
            "gpu_after": subprocess.run(gpu_command, capture_output=True, text=True).stdout,
            "quality_evidence": "PARTIAL_EVIDENCE; context consistency only, no human ground truth"}
        (output / "result.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2))
        print(json.dumps({k: v for k, v in receipt.items() if k not in {"frame_identities", "reference_predictions"}}), flush=True)
    finally:
        ops.Profile = original_profiler
        monitor.stop()
        for _role, scanner in scanners:
            scanner.close()


def scan(args: argparse.Namespace) -> None:
    """Cold shared-product scans, with ordered frame ledgers and action audit.

    Inputs are referenced in place; no raw media or NAS archive is written.
    Timing covers decode, inference, tracking and action audit, not rendering.
    """
    output = fresh_output_directory(args.output)
    from visioncortex.config import load_config, load_manifest
    from visioncortex.device_day_models import DeviceDayModels, check_coverage, device_scan_plan
    from visioncortex.actions import generate_candidates
    from visioncortex.detection import iter_frame_evidence
    from visioncortex.device_day_contract import DeviceDayLayout, digest
    from visioncortex.actions import merge_activity_intervals
    from visioncortex.scan_scheduler import scan_views_concurrently
    from visioncortex.schemas import AlignmentTransform
    from visioncortex.video_io import probe_video

    config = load_config(args.config)
    for role in ("first_person", "third_person"):
        engine = getattr(args, role + "_engine")
        if engine:
            config["models"][role + "_engine"] = str(engine.resolve(strict=True))
    perf = config["performance"]
    perf.update(batch_size=args.batch, coarse_batch_size=args.batch, fine_batch_size=args.batch,
                inference_batch_wait_ms=args.wait_ms, coarse_decode_lanes=[args.coarse_decode])
    # Owned benchmark outputs only. Source paths come solely from the manifest.
    config["storage"]["local_cache_root"] = str(output / "Cache")
    config["storage"]["local_runtime_root"] = str(output / "Runtime")
    config = DeviceDayModels(config).config
    perf = config["performance"]
    identity = comparison_identity(config, args.manifest)
    models = model_identities(config)
    manifest = load_manifest(args.manifest)
    views = manifest.views
    if any(view.video is None for view in views):
        raise ValueError("Use continuous real sources for bounded scan comparisons")
    infos = {v.view_id: probe_video(v.video) for v in views}
    windows = {v.view_id: [(args.start_seconds * 1000,
                           min(infos[v.view_id].duration_ms, (args.start_seconds + args.duration_seconds) * 1000))]
               for v in views}
    if any(end <= start for ranges in windows.values() for start, end in ranges):
        raise ValueError("Benchmark window is outside a source")
    transforms = {v.view_id: AlignmentTransform(view_id=v.view_id, reference_view_id=v.view_id,
                   state="uncertain", alignment_basis="device_local_activity_only") for v in views}
    # Exactly one recorder job per camera, as in the production device/day path.
    first_phase = "coarse" if args.phase == "preprocess" else args.phase
    image_size = int(perf["coarse_image_size"] if first_phase == "coarse" else perf["image_size"])
    fps = float(perf["coarse_detection_fps"] if first_phase == "coarse" else perf["detection_fps"])
    def work(view):
        start = time.perf_counter()
        directory = output / view.view_id
        ledgers = scan_views_concurrently(config, [view], infos, transforms, directory,
            windows={view.view_id: windows[view.view_id]}, sample_fps=fps, image_size=image_size, phase=first_phase)
        scanned = time.perf_counter() - start
        candidates = generate_candidates([view], ledgers, config)
        audit = None
        if args.phase == "preprocess":
            model = DeviceDayModels(config)
            lo, hi = windows[view.view_id][0]
            coverage = check_coverage(list(iter_frame_evidence(ledgers[view.view_id])), lo, hi, fps)
            coarse, fine_windows, plan = device_scan_plan(view, ledgers, config, infos[view.view_id], lo, hi, fps)
            layout = DeviceDayLayout(output / "Archive", view.view_id, 1789005600000000,
                                     Path(config["storage"]["local_cache_root"]))
            key = digest([str(view.video), lo, hi, perf, config["models"]])
            selected = {}
            audit = {"intervals": [], "selected_key_events": []}
            if fine_windows:
                fine = scan_views_concurrently(config, [view], infos, transforms, directory / "Fine",
                    windows={view.view_id: fine_windows}, sample_fps=float(perf["detection_fps"]),
                    image_size=int(perf["image_size"]), phase="fine")
                fine, fine_coverage, _ = model._index_fine(layout, {"recording_id": view.view_id},
                    key, 0, view, infos[view.view_id], fine, fine_windows)
                candidates = generate_candidates([view], fine, config)
                intervals, audit = model._audit_activity(view, infos[view.view_id], fine, candidates, coarse, fine_windows)
                selected = model._key_frame_choices(view, fine, audit)
                audit = {"intervals": merge_activity_intervals(intervals, lo, hi),
                         "fine_coverage": fine_coverage, **audit}
            audit.update(coarse_coverage=coverage, plan=plan, fine_windows=fine_windows,
                         selected_frames=selected)
            (directory / "ActivityAudit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2))
        if args.phase == "fine":
            intervals, audit = DeviceDayModels(config)._audit_activity(
                view, infos[view.view_id], ledgers, candidates, [], windows[view.view_id])
            audit = {"intervals": intervals, **audit}
            (directory / "ActivityAudit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2))
        (directory / "Candidates.json").write_text(json.dumps([c.model_dump(mode="json") for c in candidates], ensure_ascii=False))
        result = {"view_id": view.view_id, "role": view.role.value, "source": str(view.video),
                  "source_stat": [view.video.stat().st_size, view.video.stat().st_mtime_ns],
                  "windows": windows[view.view_id], "scan_seconds": scanned,
                  "scan_and_audit_seconds": time.perf_counter() - start,
                  "candidates": len(candidates), "ledger": str(ledgers[view.view_id])}
        print(json.dumps(result), flush=True)
        return result
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(views)) as pool:
        rows = list(pool.map(work, views))
    elapsed = time.perf_counter() - start
    verify_model_identities(config, models)
    receipt = {"scope": "Real-source per-camera concurrent decode/inference/tracking/action audit; excludes media rendering",
               "comparison_identity": identity,
               "phase": args.phase, "wall_seconds": elapsed, "rows": rows,
               "models": models,
               "performance": perf, "evidence_status": "PARTIAL_EVIDENCE; no human action ground truth"}
    (output / "result.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2))
    print(json.dumps({"phase": args.phase, "wall_seconds": receipt["wall_seconds"]}), flush=True)


def compare(args: argparse.Namespace) -> None:
    from visioncortex.fine_batch_acceptance import compare_fine_scans

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    receipt = compare_fine_scans(args.baseline, args.candidate, target_batch=args.target_batch)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, ensure_ascii=False, indent=2)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)
    if receipt["comparison_status"] != "passed":
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    b = commands.add_parser("build")
    b.add_argument("--weights", type=Path, required=True)
    b.add_argument("--reference-engine", type=Path, required=True)
    b.add_argument("--workspace-gib", type=float, default=3)
    b.add_argument("--optimal-batch", type=int)
    b.add_argument("--export-like-reference", action="store_true",
                   help="Use the original GPU TensorRT export graph preparation and simplification")
    i = commands.add_parser("infer")
    i.add_argument("--engine", type=Path, required=True)
    i.add_argument("--config", type=Path, required=True)
    i.add_argument("--manifest", type=Path, required=True)
    i.add_argument("--role", choices=["first_person", "third_person"], required=True)
    i.add_argument("--seconds", type=float, default=30)
    i.add_argument("--samples", type=int, default=16)
    s = commands.add_parser("scan")
    s.add_argument("--config", type=Path, required=True)
    s.add_argument("--manifest", type=Path, required=True)
    s.add_argument("--output", type=Path, required=True)
    s.add_argument("--first-person-engine", type=Path)
    s.add_argument("--third-person-engine", type=Path)
    s.add_argument("--batch", type=int, default=16)
    s.add_argument("--wait-ms", type=float, default=25)
    s.add_argument("--start-seconds", type=float, default=0)
    s.add_argument("--duration-seconds", type=float, default=70)
    s.add_argument("--coarse-decode", choices=["cpu", "cuda"], default="cpu")
    s.add_argument("--phase", choices=["coarse", "fine", "preprocess"], required=True)
    c = commands.add_parser("concurrent")
    c.add_argument("--config", type=Path, required=True)
    c.add_argument("--manifest", type=Path, required=True)
    c.add_argument("--output", type=Path, required=True)
    c.add_argument("--contexts", type=int, required=True)
    c.add_argument("--batch", type=int, default=4)
    c.add_argument("--seconds", type=float, default=25)
    c.add_argument("--first-person-engine", type=Path)
    c.add_argument("--third-person-engine", type=Path)
    c.add_argument("--host-only-profiler", action="store_true")
    c.add_argument("--gpu-resident", action="store_true")
    m = commands.add_parser("movement")
    m.add_argument("--config", type=Path, required=True)
    m.add_argument("--manifest", type=Path, required=True)
    m.add_argument("--baseline", type=Path, required=True)
    m.add_argument("--output", type=Path, required=True)
    m.add_argument("--workers", type=int, default=1)
    m.add_argument("--cache-frames", type=int, default=96)
    m.add_argument("--opencv-threads", type=int, default=1)
    comparison = commands.add_parser("compare")
    comparison.add_argument("--baseline", type=Path, required=True)
    comparison.add_argument("--candidate", type=Path, required=True)
    comparison.add_argument("--output", type=Path, required=True,
                            help="New JSON receipt file; existing files are never overwritten")
    comparison.add_argument("--target-batch", type=int, default=16)
    for command in (b, i):
        command.add_argument("--batch", type=int, required=True)
        command.add_argument("--image-size", type=int, default=640)
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "compare":
        if not 1 <= args.target_batch <= 512:
            parser.error("target-batch must be in [1, 512]")
        compare(args)
        return
    if args.command == "scan":
        if not 1 <= args.batch <= 64 or not 1 <= args.wait_ms <= 2000:
            parser.error("batch must be in [1,64], wait-ms in [1,2000]")
        if args.start_seconds < 0 or not 0 < args.duration_seconds <= 900:
            parser.error("start must be nonnegative and duration in (0,900]")
        scan(args)
        return
    if args.command == "concurrent":
        if not 2 <= args.contexts <= 16 or args.contexts % 2 or not 1 <= args.batch <= 128 or not 5 <= args.seconds <= 300:
            parser.error("contexts must be even in [2,16], batch in [1,128], seconds in [5,300]")
        concurrent(args)
        return
    if args.command == "movement":
        if not 1 <= args.workers <= 8 or not 1 <= args.cache_frames <= 4096 or not 1 <= args.opencv_threads <= 24:
            parser.error("workers/cache-frames/opencv-threads exceed bounded capacities")
        movement(args)
        return
    if not 1 <= args.batch <= 512 or args.image_size < 160 or args.image_size % 32:
        parser.error("batch must be in [1, 512], image-size at least 160 and a multiple of 32")
    if args.command == "infer" and (not 5 <= args.seconds <= 300 or not 1 <= args.samples <= 64):
        parser.error("seconds must be in [5, 300], samples in [1, 64]")
    if args.command == "build" and not .25 <= args.workspace_gib <= 12:
        parser.error("workspace-gib must be in [.25, 12]")
    if args.command == "build" and args.optimal_batch is not None and not 1 <= args.optimal_batch <= args.batch:
        parser.error("optimal-batch must be in [1, batch]")
    (build if args.command == "build" else infer)(args)


if __name__ == "__main__":
    main()
