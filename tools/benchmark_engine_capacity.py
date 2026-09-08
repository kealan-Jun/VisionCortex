"""Build bounded-shape TensorRT candidates and time real-frame product inference.

Artifacts stay under the supplied output directory. This never changes a live
engine, downloads a model, calls a provider, or claims experiment accuracy.
"""
from __future__ import annotations

import argparse
import ast
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


def build(args: argparse.Namespace) -> None:
    import tensorrt as trt

    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"batch-{args.batch}.engine"
    if destination.exists():
        raise FileExistsError(destination)
    source = args.weights.resolve(strict=True)
    copied = root / "source.pt"
    if not copied.exists():
        shutil.copy2(source, copied)
    if sha256(copied) != sha256(source):
        raise ValueError("Frozen weights differ from requested weights")
    onnx = copied.with_suffix(".onnx")
    if not onnx.exists():
        # CPU export changes CUDA_VISIBLE_DEVICES in Ultralytics; isolate it
        # so the TensorRT builder still sees the original GPU environment.
        subprocess.run([sys.executable, "-c",
            "import sys; from ultralytics import YOLO; "
            "YOLO(sys.argv[1], task='detect').export(format='onnx', "
            "imgsz=int(sys.argv[2]), batch=1, dynamic=True, simplify=False, "
            "opset=17, device='cpu')", str(copied), str(args.image_size)], check=True)
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
    with destination.open("xb") as handle:
        handle.write(struct.pack("<I", len(header)))
        handle.write(header)
        handle.write(serialized)
    receipt = {"schema_version": "visioncortex-engine-capacity-build/1", "engine": str(destination),
               "weights_sha256": sha256(source), "onnx_sha256": sha256(onnx),
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
        batch = [packets[i % len(packets)] for i in range(args.batch)]
        for _ in range(5):
            scanner.infer(batch)
        torch.cuda.synchronize()
        start = time.perf_counter()
        frames, latencies, offset = 0, [], 0
        while time.perf_counter() - start < args.seconds:
            batch = [packets[(offset + i) % len(packets)] for i in range(args.batch)]
            before = time.perf_counter()
            scanner.infer(batch)
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - before)
            frames += len(batch)
            offset += len(batch)
        elapsed = time.perf_counter() - start
        predictions = scanner.infer(packets)
        quality = [[box.model_dump(mode="json") for box in boxes] for boxes in predictions]
        receipt = {"schema_version": "visioncortex-engine-capacity-inference/1",
            "scope": "Predecoded real-frame product inference; excludes video IO and complete preprocessing",
            "role": args.role, "engine_sha256": sha256(args.engine),
            "requested_batch": args.batch, "effective_batch": scanner.batch_size,
            "engine_capacity": scanner.engine_build_batch, "contractions": scanner.batch_contractions,
            "frames": frames, "seconds": elapsed, "frames_per_second": frames / elapsed,
            "batch_latency_ms_mean": sum(latencies) / len(latencies) * 1000,
            "frame_identities": identities, "predictions": quality,
            "quality_evidence": "PARTIAL_EVIDENCE; comparison required; no human ground truth"}
        receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2))
        print(json.dumps({k: v for k, v in receipt.items() if k not in {"frame_identities", "predictions"}}), flush=True)
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    b = commands.add_parser("build")
    b.add_argument("--weights", type=Path, required=True)
    b.add_argument("--reference-engine", type=Path, required=True)
    b.add_argument("--workspace-gib", type=float, default=3)
    b.add_argument("--optimal-batch", type=int)
    i = commands.add_parser("infer")
    i.add_argument("--engine", type=Path, required=True)
    i.add_argument("--config", type=Path, required=True)
    i.add_argument("--manifest", type=Path, required=True)
    i.add_argument("--role", choices=["first_person", "third_person"], required=True)
    i.add_argument("--seconds", type=float, default=30)
    i.add_argument("--samples", type=int, default=16)
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
    for command in (b, i):
        command.add_argument("--batch", type=int, required=True)
        command.add_argument("--image-size", type=int, default=640)
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
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
