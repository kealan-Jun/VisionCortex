"""Prepare and launch one isolated, host-local VisionCortex user-acceptance instance."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request

REPOSITORY = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def clean_environment():
    # Do not let the workstation's production/NAS deployment override this instance.
    return {key: value for key, value in os.environ.items() if not key.startswith("VISIONCORTEX_")}


def local_config(settings, runtime, source, connection):
    from copy import deepcopy

    settings = deepcopy(settings)
    settings["project"].update(output_root=str(runtime / "Staging"), portable_desktop=True,
                               run_purpose="local_user_acceptance")
    settings["storage"].update(
        index_csv=str(runtime / "Catalog/index.csv"), device_registry_path=str(runtime / "Catalog/devices.json"),
        archive_root=str(runtime / "Archives"), local_input_root=str(runtime / "Input-Manifests"),
        local_runtime_root=str(runtime), local_cache_root=str(runtime / "Cache"),
        local_staging_root=str(runtime / "Staging"), sync_to_nas=False,
        run_output_mode="local", manifest_storage="local", require_nas_source_paths=False,
        web_upload_retention_mode="local_only", source_path_migration_receipts=[])
    settings["collection_ingest"].update(
        enabled=True, mode="directory_metadata", source_root=str(source), discover_plain_video_csv=True,
        camera_directories=[], camera_directory_glob="*", camera_role_map={},
        snapshot_path=str(runtime / "Catalog/collection-snapshot.json"), max_scan_directories=32,
        max_recordings=32, max_recordings_per_camera=4)
    settings["mllm"].update(connection, enabled=True, api_key_env="VISIONCORTEX_LOCAL_EXPERIENCE_KEY")
    for field in ("credential_ref", "connection_verification"):
        settings["mllm"].pop(field, None)
    return settings


def initialize_storage(settings):
    """Create a fresh local catalog without replacing user-selected device roles."""
    from visioncortex.device_registry import DEVICE_REGISTRY_SCHEMA_VERSION

    registry = Path(settings["storage"]["device_registry_path"])
    if not registry.exists():
        write_json(registry, {"schema_version": DEVICE_REGISTRY_SCHEMA_VERSION,
                              "devices": {}, "experiment_role_overrides": {}})
    index = Path(settings["storage"]["index_csv"])
    index.parent.mkdir(parents=True, exist_ok=True)
    if not index.exists():
        with index.open("x", encoding="utf-8") as handle:
            handle.write("experiment_id,camera_key,camera_view,rgb_file,frames_file\n")


def prepare(root, source):
    import yaml
    from visioncortex.ai_settings import read_index
    from visioncortex.config import load_config
    from visioncortex.provider_credentials import read_revision, settings_directory

    if root == REPOSITORY or REPOSITORY in root.parents:
        raise RuntimeError("体验目录必须位于源码仓库之外。")
    if (root / "instance.json").exists():
        raise RuntimeError("本机体验版本已经存在；请在同一目录修复，不重复创建。")
    if not source.is_dir():
        raise RuntimeError("原视频目录不存在。")
    index = read_index()
    ref = index.get("active_ref")
    if not ref:
        raise RuntimeError("请先在本机 AI 服务设置中配置厂商。")
    # Keep the secret in its approved private store; the instance contains only a reference.
    record = read_revision(ref)
    connection = record["connection"]
    credential_source = settings_directory()
    runtime, app = root / "Runtime", root / "App"
    paths = subprocess.check_output(["git", "ls-files", "-c", "-o", "--exclude-standard", "-z", "src", "configs"], cwd=REPOSITORY).decode().split("\0")
    paths += ["pyproject.toml", "tools/local_experience.py"]
    records = []
    for name in sorted(set(paths)):
        if not name:
            continue
        path = REPOSITORY / name
        if not path.exists():
            continue
        if path.is_symlink() or path.suffix not in {".py", ".json", ".yaml", ".yml", ".html", ".js", ".css", ".toml"}:
            raise RuntimeError(f"Unexpected source artifact: {name}")
        before = sha(path)
        target = app / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        if before != sha(target) or before != sha(path):
            raise RuntimeError(f"源码在准备过程中变化，请在同一目录完成修复：{name}")
        records.append({"path": name, "sha256": before})
    # Configuration evaluation runs in a clean environment, then is stored fully resolved.
    previous = dict(os.environ)
    try:
        os.environ.clear()
        os.environ.update(cleaned := {k: v for k, v in previous.items() if not k.startswith("VISIONCORTEX_")})
        settings = load_config(REPOSITORY / "configs/rtx3090ti-ubuntu-local.yaml")
    finally:
        os.environ.clear()
        os.environ.update(previous)
    del cleaned
    settings = local_config(settings, runtime, source, connection)
    initialize_storage(settings)
    runtime.mkdir(parents=True, exist_ok=True)
    (app / "configs/local-experience.yaml").write_text(yaml.safe_dump(settings, allow_unicode=True, sort_keys=False), encoding="utf-8")
    private = Path.home() / ".config/VisionCortex/local-experience-ai"
    info = {"root": str(root), "python": sys.executable, "config": str(app / "configs/local-experience.yaml"),
            "source_root": str(source), "credential_source": str(credential_source), "credential_ref": ref,
            "private_settings": str(private), "connection": connection,
            "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True).strip(),
            "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=REPOSITORY, text=True).strip(),
            "working_tree_snapshot": True, "shared_host_dependencies": True, "stable_release_readiness": "NOT_PROVEN"}
    write_json(root / "instance.json", info)
    write_json(root / "Acceptance/source-snapshot.json", records)
    start = root / "打开 VisionCortex.sh"
    start.write_text('#!/usr/bin/env bash\nset -Eeuo pipefail\nroot=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)\n'
                     f'exec {sys.executable!r} "$root/App/tools/local_experience.py" launch --root "$root" "$@"\n', encoding="utf-8")
    start.chmod(0o755)
    stop_script = root / "关闭 VisionCortex.sh"
    stop_script.write_text(start.read_text().replace('local_experience.py" launch ', 'local_experience.py" stop '), encoding="utf-8")
    stop_script.chmod(0o755)
    return {"root": str(root), "source_files": len(records), "source_media_copied": 0, "model_files_copied": 0}


def check(root):
    import yaml

    info = json.loads((root / "instance.json").read_text())
    config = yaml.safe_load(Path(info["config"]).read_text())
    versions = {name: importlib.metadata.version(name) for name in
                ("torch", "ultralytics", "numpy", "httpx", "fastapi", "tensorrt-cu12-bindings")}
    paths = {Path(config["models"][name]) for name in ("first_person", "third_person", "first_person_engine", "third_person_engine")}
    def model_paths(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in {"model_path", "clip_model_path", "checkpoint_path"} and isinstance(item, str):
                    path = Path(item)
                    if path.is_dir():
                        paths.update(p for p in path.rglob("*") if p.is_file())
                    else:
                        paths.add(path)
                else:
                    model_paths(item)
    model_paths(config["models"])
    models = [{"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)} for path in sorted(paths)]
    storage = config["storage"]
    local = root / "Runtime"
    for name in ("archive_root", "local_input_root", "local_runtime_root", "local_cache_root", "local_staging_root", "index_csv", "device_registry_path"):
        path = Path(storage[name]).resolve()
        if path != local and local not in path.parents:
            raise RuntimeError(f"Local storage escapes instance: {name}")
    if storage["sync_to_nas"] or storage["require_nas_source_paths"] or storage["run_output_mode"] != "local":
        raise RuntimeError("Local instance inherited a NAS contract")
    inputs = [{"path": str(path), "bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
              for path in sorted(Path(info["source_root"]).iterdir()) if path.is_file() and path.suffix.lower() in {".mp4", ".csv"}]
    report = {"python": sys.executable, "versions": versions, "config_sha256": sha(Path(info["config"])),
              "models": models, "inputs": inputs, "free_disk_bytes": shutil.disk_usage(root).free,
              "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total,memory.used", "--format=csv,noheader"], text=True).strip(),
              "model_execution": "NOT_PROVEN", "source_video_copy_bytes": 0}
    if report["free_disk_bytes"] < 15 * 1024**3:
        raise RuntimeError("本地空间不足 15 GiB，请先释放空间。")
    write_json(root / "Acceptance/preflight.json", report)
    return {"models_checked": len(models), "input_files": len(inputs), "free_disk_gib": round(report["free_disk_bytes"] / 1024**3, 1)}


def request(url):
    with urllib.request.urlopen(url, timeout=2) as response:
        return json.load(response)


def owned_server(root):
    state = root / "Runtime/server.json"
    if not state.exists():
        return None
    value = json.loads(state.read_text())
    try:
        arguments = Path(f"/proc/{int(value['pid'])}/cmdline").read_bytes().split(b"\0")
        if str(root / "App/configs/local-experience.yaml").encode() not in arguments:
            return None
        health = request(value["url"] + "/api/health")
        if health["archive_root"] != str(root / "Runtime/Archives"):
            return None
        return value
    except (OSError, ValueError, KeyError):
        return None


def open_browser(url):
    browser = shutil.which("google-chrome") or shutil.which("chromium")
    if not browser:
        raise RuntimeError("本机未找到 Chrome/Chromium，请在浏览器打开：" + url)
    subprocess.Popen([browser, "--new-window", "--app=" + url, "--class=VisionCortexLocal"],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)


def launch(root, no_browser):
    import fcntl  # This launcher targets the prepared Ubuntu workstation.

    runtime = root / "Runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    with (runtime / "launcher.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        current = owned_server(root)
        if not current:
            from visioncortex.provider_credentials import read_revision

            info = json.loads((root / "instance.json").read_text())
            old = os.environ.get("VISIONCORTEX_AI_SETTINGS_DIR")
            try:
                os.environ["VISIONCORTEX_AI_SETTINGS_DIR"] = info["credential_source"]
                key = read_revision(info["credential_ref"])["api_key"]
            finally:
                if old is None:
                    os.environ.pop("VISIONCORTEX_AI_SETTINGS_DIR", None)
                else:
                    os.environ["VISIONCORTEX_AI_SETTINGS_DIR"] = old
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            env = clean_environment()
            env.update(PYTHONPATH=str(root / "App/src"), PYTHONDONTWRITEBYTECODE="1",
                       VISIONCORTEX_CONFIG=info["config"], VISIONCORTEX_WEB_AI_SETTINGS="1",
                       VISIONCORTEX_AI_SETTINGS_DIR=info["private_settings"],
                       VISIONCORTEX_LOCAL_EXPERIENCE_KEY=key, VISIONCORTEX_WEB_ACCESS_MODE="local",
                       VISIONCORTEX_ULTRALYTICS_CONFIG_DIR=str(runtime / "ThirdParty"),
                       HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
            logs = runtime / "Logs"
            logs.mkdir(parents=True, exist_ok=True)
            with (logs / "web.log").open("ab") as log:
                process = subprocess.Popen([info["python"], "-B", "-m", "visioncortex", "serve", "--host", "127.0.0.1", "--port", str(port), "--config", info["config"]],
                                           cwd=root / "App", env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
            current = {"pid": process.pid, "url": f"http://127.0.0.1:{port}"}
            write_json(runtime / "server.json", current)
            for _ in range(120):
                if owned_server(root):
                    break
                if process.poll() is not None:
                    raise RuntimeError("本机体验启动失败，请查看 Runtime/Logs/web.log。")
                time.sleep(.5)
            else:
                process.terminate()
                raise RuntimeError("本机体验启动等待超时，请查看 Runtime/Logs/web.log。")
        url = current["url"] + "/#/ai-settings"
        if not no_browser:
            open_browser(url)
        return {"url": url, "pid": current["pid"]}


def stop(root):
    current = owned_server(root)
    if not current:
        return {"stopped": False}
    data = request(current["url"] + "/api/runs")
    if any(item.get("state") not in {"completed", "partial", "failed", "interrupted", "cancelled"} for item in data.get("runs", [])):
        raise RuntimeError("请等待当前分析任务结束再关闭本机服务。")
    os.kill(current["pid"], signal.SIGTERM)
    return {"stopped": True, "pid": current["pid"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "check", "launch", "stop"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    if args.action == "prepare":
        if not args.source:
            parser.error("prepare requires --source")
        result = prepare(root, args.source.expanduser().resolve())
    elif args.action == "check":
        result = check(root)
    elif args.action == "stop":
        result = stop(root)
    else:
        result = launch(root, args.no_browser)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    sys.path.insert(0, str(REPOSITORY / "src"))
    main()
