"""Offline, relocatable Windows launcher. Never downloads models or dependencies."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser


ROOT = Path(__file__).resolve().parents[1]
DESKTOP_MODE = False


def desktop_state(stage: str, message: str, *, status: str = "preparing", **details) -> None:
    if DESKTOP_MODE:
        print("VISIONCORTEX_DESKTOP_STATE " + json.dumps(
            {"stage": stage, "status": status, "message": message, **details}, ensure_ascii=False), flush=True)


def sha256(path: Path, *, on_chunk=None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
            if on_chunk is not None:
                on_chunk(len(block))
    return digest.hexdigest()


def safe_path(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if (
        not relative or pure.is_absolute() or ".." in pure.parts
        or "\\" in relative or ":" in relative
    ):
        raise RuntimeError("Invalid package manifest path")
    result = (root / relative).resolve()
    if root.resolve() not in result.parents:
        raise RuntimeError("Package manifest path escapes root")
    return result


def verify_package(root: Path, *, progress=None) -> dict:
    manifest = json.loads((root / "SHA256SUMS.json").read_text(encoding="utf-8"))
    seen = set()
    total_bytes = sum(entry["size_bytes"] for entry in manifest["files"])
    checked_bytes = checked_files = 0

    def report(size=0):
        nonlocal checked_bytes
        checked_bytes += size
        if progress is not None:
            progress({"checked_bytes": checked_bytes, "total_bytes": total_bytes,
                      "checked_files": checked_files, "total_files": manifest["file_count"]})

    report()
    for entry in manifest["files"]:
        name = entry["path"]
        if name.casefold() in seen:
            raise RuntimeError("Duplicate package manifest path")
        seen.add(name.casefold())
        path = safe_path(root, name)
        if path.stat().st_size != entry["size_bytes"]:
            raise RuntimeError(f"文件校验失败，请重新完整解压：{name}")
        digest = sha256(path, on_chunk=report) if progress is not None else sha256(path)
        if digest != entry["sha256"]:
            raise RuntimeError(f"文件校验失败，请重新完整解压：{name}")
        checked_files += 1
        if progress is None:
            checked_bytes += entry["size_bytes"]
        report()
    if len(seen) != manifest["file_count"]:
        raise RuntimeError("Package manifest count mismatch")
    return manifest


def data_runtime_root(root: Path, selection: dict) -> Path:
    if not selection:
        return root / "Runtime"
    from desktop_storage import normalize_root
    selected = normalize_root(selection.get("data_root"), root)
    if selection.get("kind") == "local":
        return selected
    if selection.get("kind") == "network":
        identity = hashlib.sha256(str(selected).casefold().encode()).hexdigest()[:20]
        # Keep each share's queue separate, including from the default local
        # folder. Returning to local mode must not hydrate old NAS job paths.
        return root / "Runtime/NetworkStores" / identity
    raise RuntimeError("保存位置类型无效。")


def configure_environment(root: Path) -> None:
    desktop_connection = os.environ.get("VISIONCORTEX_DESKTOP_CONNECTION") if DESKTOP_MODE else None
    desktop_storage = os.environ.get("VISIONCORTEX_DESKTOP_STORAGE") if DESKTOP_MODE else None
    # Old NAS, model, default-config and Web settings must not enter this package.
    for name in list(os.environ):
        if name.upper().startswith("VISIONCORTEX_"):
            del os.environ[name]
    selection = json.loads(desktop_storage or "{}")
    runtime = data_runtime_root(root, selection)
    for name in ("Temp", "Cache", "Logs", "ThirdParty", "Archives"):
        (runtime / name).mkdir(parents=True, exist_ok=True)
    os.environ.update({
        "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "HF_HOME": str(runtime / "Cache/HuggingFace"),
        "TORCH_HOME": str(runtime / "Cache/Torch"),
        "MPLCONFIGDIR": str(runtime / "Cache/Matplotlib"),
        "XDG_CACHE_HOME": str(runtime / "Cache"),
        "YOLO_OFFLINE": "true", "YOLO_AUTOINSTALL": "false",
        "YOLO_CONFIG_DIR": str(runtime / "ThirdParty"),
        "VISIONCORTEX_ULTRALYTICS_CONFIG_DIR": str(runtime / "ThirdParty"),
        "NVIDIA_TENSORRT_DISABLE_INTERNAL_PIP": "1",
        "PIP_NO_INDEX": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PIP_CACHE_DIR": str(runtime / "Cache/Pip"),
        "TEMP": str(runtime / "Temp"), "TMP": str(runtime / "Temp"),
    })
    if DESKTOP_MODE:
        os.environ["VISIONCORTEX_DESKTOP_MODE"] = "1"
        if desktop_connection:
            os.environ["VISIONCORTEX_DESKTOP_CONNECTION"] = desktop_connection
        if desktop_storage:
            os.environ["VISIONCORTEX_DESKTOP_STORAGE"] = desktop_storage
    libraries = root / "python/Lib/site-packages"
    dll_roots = [root / "vendor/ffmpeg/bin", libraries / "torch/lib",
                 libraries / "tensorrt_libs"]
    os.environ["PATH"] = os.pathsep.join(map(str, dll_roots)) + os.pathsep + os.environ.get("PATH", "")
    threads = max(1, min(12, (os.cpu_count() or 4) // 2))
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def hardware_preflight() -> dict:
    from rtx4050_hardware import hardware_preflight as check

    return check(ROOT, desktop_state)


def effective_config(root: Path, hardware: dict) -> tuple[Path, dict]:
    import yaml
    from visioncortex.config import load_config

    profile = root / "configs/rtx4050-6gb-windows-local.yaml"
    config = load_config(profile)

    def relocate(value):
        if isinstance(value, dict):
            return {key: relocate(item) for key, item in value.items()}
        if isinstance(value, list):
            return [relocate(item) for item in value]
        if isinstance(value, str) and value.startswith(("./Runtime/", "./models/", "./configs/")):
            return str((root / value).resolve())
        if value == "./Runtime":
            return str(root / "Runtime")
        return value

    config = relocate(config)
    selected_storage = json.loads(os.environ.get("VISIONCORTEX_DESKTOP_STORAGE", "{}")) if DESKTOP_MODE else {}
    data_root = (root / "Runtime").resolve()
    if selected_storage:
        from desktop_storage import normalize_root
        data_root = normalize_root(selected_storage.get("data_root"), root)
        if selected_storage.get("kind") not in {"local", "network"}:
            raise RuntimeError("保存位置类型无效，请重新选择保存位置。")
    network = selected_storage.get("kind") == "network"
    runtime = data_runtime_root(root, selected_storage).resolve()
    archive_root = data_root / "Archives"
    config["storage"].update(
        index_csv=str(runtime / "local-experiment-index.csv"),
        device_registry_path=str(runtime / "local-device-registry.json"),
        archive_root=str(archive_root), local_input_root=str(runtime / "Input-Manifests"),
        local_runtime_root=str(runtime), local_cache_root=str(runtime / "Cache"),
        local_staging_root=str(archive_root / "Processing"), staging_directory_name="Processing",
        sync_to_nas=network, web_upload_retention_mode="nas_only" if network else "local_only",
    )
    config["project"]["output_root"] = str(runtime / "runs")
    config["collection_ingest"].update(source_root=str(runtime / "Input-Manifests"), snapshot_path=str(runtime / "collection-snapshot.json"))
    if selected_storage.get("source_root"):
        from desktop_storage import input_directory
        source = input_directory(selected_storage["source_root"], root, data_root)
        config["collection_ingest"].update(
            enabled=True, mode="directory_metadata", source_root=str(source),
            discover_plain_video_csv=True, camera_directories=[],
        )
    for directory in (runtime, runtime / "Cache", runtime / "Input-Manifests", archive_root, archive_root / "Processing"):
        directory.mkdir(parents=True, exist_ok=True)
    from visioncortex.device_registry import DEVICE_REGISTRY_SCHEMA_VERSION, load_device_registry
    registry = runtime / "local-device-registry.json"
    if not registry.exists():
        # A new installation has no registered cameras. Explicit roles selected
        # in the upload form remain authoritative; never import another lab's
        # device identities or overwrite an existing user's registry.
        write_json(registry, {"schema_version": DEVICE_REGISTRY_SCHEMA_VERSION,
                              "devices": {}, "experiment_role_overrides": {}})
    load_device_registry(registry)
    index = runtime / "local-experiment-index.csv"
    if not index.exists():
        with index.open("x", encoding="utf-8", newline="") as handle:
            handle.write("experiment_id,camera_key,camera_view\n")
    if DESKTOP_MODE:
        from visioncortex.mllm_provider import normalize_connection
        from visioncortex.provider_connection import verification_matches
        saved = json.loads(os.environ.get("VISIONCORTEX_DESKTOP_CONNECTION", "{}"))
        connection = normalize_connection(saved.get("connection", {}))
        if not os.environ.get("MLLM_API_KEY") or not verification_matches(connection, saved.get("verification", {})):
            raise RuntimeError("AI 服务配置未通过当前版本的真实连接验证，请在应用设置中重新验证。")
        config["mllm"].update(connection)
        config["mllm"]["api_key_env"] = "MLLM_API_KEY"
        config["mllm"]["connection_verification"] = saved["verification"]
    config["performance"]["cpu_decode_threads"] = max(1, min(4, (os.cpu_count() or 4) // 4))
    signature = hashlib.sha256(json.dumps({
        "hardware": hardware, "config": {key: value for key, value in config.items() if key not in {"mllm", "storage", "project", "collection_ingest"}},
        "source_manifest": sha256(root / "SHA256SUMS.json"),
    }, sort_keys=True).encode()).hexdigest()[:20]
    for role in ("first_person", "third_person"):
        config["models"][f"{role}_engine"] = str(root / "Runtime/Engines" / signature / f"{role}.engine")
    for key in ("index_csv", "device_registry_path", "archive_root", "local_input_root",
                "local_runtime_root", "local_cache_root", "local_staging_root"):
        path = Path(config["storage"][key]).resolve()
        boundary = data_root if key in {"archive_root", "local_staging_root"} else runtime
        if path != boundary and boundary not in path.parents:
            raise RuntimeError(f"本地存储路径越界：{key}")
    destination = root / "Runtime/active-config.yaml"
    destination.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    os.environ["VISIONCORTEX_DEFAULT_CONFIG"] = str(destination)
    os.environ["VISIONCORTEX_CONFIG"] = str(destination)
    return destination, config


def verify_engine_receipts(config: dict) -> None:
    for role in ("first_person", "third_person"):
        engine = Path(config["models"][f"{role}_engine"])
        if not engine.is_file():
            continue
        receipt_path = engine.with_suffix(".engine.build.json")
        if not receipt_path.is_file():
            raise RuntimeError(f"缺少引擎身份回执：{receipt_path}")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (receipt["engine_sha256"] != sha256(engine)
                or receipt["source_sha256"] != sha256(Path(config["models"][role]))
                or receipt["image_size"] != config["performance"]["image_size"]
                or receipt["selected_batch"] not in config["performance"]["engine_batch_candidates"]):
            raise RuntimeError("引擎身份校验失败，请保留日志并重新解压到新目录。")


def command(config: Path, *arguments: str) -> None:
    run_logged([sys.executable, "-B", "-m", "visioncortex", *arguments, "--config", str(config)])


def run_logged(arguments: list[str]) -> None:
    log = ROOT / "Runtime/Logs/startup.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(arguments, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace")
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                handle.write(line)
                handle.flush()
            if process.wait():
                raise RuntimeError(f"离线准备失败，详细日志：{log}")
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def model_smoke(stage: str, config_path: Path) -> None:
    """One isolated process per model; synthetic input proves execution only."""
    import cv2
    import numpy as np
    from visioncortex.config import load_config

    config = load_config(config_path)
    frame = np.zeros((256, 384, 3), dtype=np.uint8)
    cv2.rectangle(frame, (96, 64), (288, 192), (180, 180, 180), -1)
    root = Path(config["models"]["first_person_engine"]).parent / "startup-smoke"
    root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if stage == "world":
        from visioncortex.open_vocabulary_runtime import load_yolo_world_with_local_clip
        model = load_yolo_world_with_local_clip(config["models"]["open_vocabulary_key_frame"])
        model.set_classes(["laboratory beaker", "gloved hand"])
        model.predict(frame, imgsz=640, device=0, verbose=False)
    elif stage == "dino":
        from visioncortex.archive import _grounding_dino_key_frame_detections
        _, receipt = _grounding_dino_key_frame_detections(
            frame, {"beaker", "container", "gloved_hand"}, config["models"]["open_vocabulary_key_frame"])
        if receipt.get("status") != "executed":
            raise RuntimeError("Grounding DINO startup inference did not execute")
    elif stage == "labpics":
        from visioncortex.liquid_semantic import predict_liquid_masks
        masks, _ = predict_liquid_masks(frame, config)
        if not masks:
            raise RuntimeError("LabPics startup inference returned no semantic maps")
    elif stage == "sam2":
        from visioncortex.local_model_acceptance import _write_bounded_clip
        from visioncortex.temporal_segmentation import audit_participant_continuity
        clip = root / "synthetic-nine-frames.mp4"
        _write_bounded_clip(frame, clip)
        _, receipt = audit_participant_continuity(
            clip, frame, [{"class_name": "vessel", "confidence": 1.0,
                           "xyxy_norm": [0.25, 0.25, 0.75, 0.75], "seed_source": "synthetic_not_ground_truth"}],
            root / "sam2-work", config, event_id="SYNTHETIC-STARTUP", view_id="synthetic",
            action_type="liquid_movement", seed_fraction=0.5)
        if receipt.get("status") != "completed":
            raise RuntimeError("SAM2 startup inference did not execute")
    write_json(root / f"{stage}.json", {"stage": stage, "model_invocation": "PROVEN",
               "input_kind": "synthetic_not_ground_truth", "real_video_quality": "NOT_PROVEN",
               "elapsed_seconds": round(time.perf_counter() - started, 3),
               "config_sha256": sha256(config_path)})


def ensure_model_smoke(config_path: Path, config: dict) -> None:
    root = Path(config["models"]["first_person_engine"]).parent / "startup-smoke"
    for stage in ("world", "dino", "labpics", "sam2"):
        receipt = root / f"{stage}.json"
        if receipt.is_file():
            record = json.loads(receipt.read_text(encoding="utf-8"))
            if record.get("model_invocation") == "PROVEN" and record.get("config_sha256") == sha256(config_path):
                continue
        print(f"本地模型首次自检：{stage}（合成输入，仅验证可执行性）", flush=True)
        desktop_state("models", {"world": "正在准备目标识别模型…", "dino": "正在准备物体复核模型…",
                                "labpics": "正在准备材料分析模型…", "sam2": "正在准备视频分割模型…"}[stage])
        run_logged([sys.executable, "-B", str(ROOT / "tools/rtx4050_portable.py"),
                    "--smoke-stage", stage, "--config", str(config_path)])


def serve(root: Path, config: Path, port: int, no_browser: bool) -> None:
    from visioncortex.config import load_config
    expected_archive = Path(load_config(config)["storage"]["archive_root"]).resolve()
    with socket.socket() as probe:
        try:
            probe.bind(("127.0.0.1", port))
        except OSError as exc:
            raise RuntimeError(f"端口 {port} 已被占用，请关闭已有服务或使用 --port。") from exc
    log = root / "Runtime/Logs/web.log"
    with log.open("a", encoding="utf-8") as output:
        process = subprocess.Popen([sys.executable, "-B", "-m", "visioncortex", "serve", "--host", "127.0.0.1",
                                    "--port", str(port), "--config", str(config)], stdout=output, stderr=output)
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            for _ in range(120):
                if process.poll() is not None:
                    raise RuntimeError(f"网页服务退出，日志：{log}")
                try:
                    with opener.open(f"http://127.0.0.1:{port}/api/health", timeout=1) as response:
                        health = json.load(response)
                    if (health.get("product_name") == "VisionCortex"
                            and Path(health["archive_root"]).resolve() == expected_archive):
                        break
                except (OSError, ValueError, KeyError):
                    pass
                time.sleep(0.5)
            else:
                raise RuntimeError(f"网页启动超时，日志：{log}")
            url = f"http://127.0.0.1:{port}/#/home"
            desktop_state("ready", "应用已准备完成。", status="ready", url=url)
            if not DESKTOP_MODE:
                print(f"网页已启动：{url}\n请保留此窗口；按 Ctrl+C 停止。", flush=True)
            if not no_browser:
                webbrowser.open(url)
            process.wait()
            if process.returncode:
                raise RuntimeError(f"网页服务异常退出，日志：{log}")
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def main() -> int:
    global DESKTOP_MODE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--smoke-stage", choices=("world", "dino", "labpics", "sam2"), help=argparse.SUPPRESS)
    parser.add_argument("--config", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--desktop", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--parent-pid", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    os.chdir(ROOT)
    DESKTOP_MODE = args.desktop
    if DESKTOP_MODE:
        if args.parent_pid is None:
            parser.error("--parent-pid is required for desktop mode")
        from windows_desktop_lifecycle import supervise_desktop_parent
        supervise_desktop_parent(args.parent_pid)
    if args.smoke_stage:
        if args.config is None:
            parser.error("--config is required for a smoke stage")
        model_smoke(args.smoke_stage, args.config)
        return 0
    print("正在核验离线包完整性，请稍候……", flush=True)
    desktop_state("integrity", "正在检查应用文件完整性…")
    verify_package(ROOT)
    configure_environment(ROOT)
    desktop_state("hardware", "正在检查本机运行环境…")
    hardware = hardware_preflight()
    write_json(ROOT / "Runtime/target-preflight.json", hardware)
    config_path, config = effective_config(ROOT, hardware)
    if shutil.disk_usage(ROOT).free < 15 * 1024**3:
        raise RuntimeError("解压盘可用空间不足 15 GiB；视频容量另由上传前预检核算。")
    verify_engine_receipts(config)
    if args.check_only:
        print("环境与文件自检通过；未构建引擎、未分析视频。")
        return 0
    print("首次启动将离线构建并实测 TensorRT 引擎，可能需要数分钟。", flush=True)
    desktop_state("engines", "正在为这台电脑准备加速引擎，首次使用需要几分钟…")
    command(config_path, "prepare-engine")
    verify_engine_receipts(config)
    command(config_path, "validate-models")
    ensure_model_smoke(config_path, config)
    print("模型文件与引擎检查完成；真实视频质量仍需本机运行验收。", flush=True)
    desktop_state("interface", "正在打开分析工作空间…")
    serve(ROOT, config_path, args.port, args.no_browser)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("VisionCortex 已停止。")
    except Exception as error:
        # Startup never prints environment values or stored credentials.
        print(f"启动失败：{type(error).__name__}: {error}", file=sys.stderr)
        desktop_state("error", str(error), status="error")
        raise SystemExit(1) from None
