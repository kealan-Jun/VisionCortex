"""Render portable units and verify deployment identity without loading models."""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import shlex
import socket
import sys
import urllib.request


def absolute(value: str) -> str:
    if not value or any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError("Deployment paths must not contain control characters.")
    return str(Path(value).expanduser().resolve())


def settings_for(project: str, config: str, mode: str) -> dict:
    if sys.version_info[:2] not in {(3, 11), (3, 12)}:
        raise ValueError("Python 3.11 or 3.12 required.")
    os.chdir(project)
    sys.path.insert(0, str(Path(project) / "src"))
    from visioncortex.config import load_config

    try:
        settings = load_config(Path(config))
    except Exception as error:
        raise ValueError("Invalid deployment configuration (" + type(error).__name__ + ").") from None
    if mode == "production":
        if settings.get("project", {}).get("site_configuration_required", False):
            raise ValueError("Production site configuration is unprepared; supply a private site configuration.")
        if settings.get("runtime", {}).get("local_only", False):
            raise ValueError("Production requires an explicitly prepared non-local site configuration.")
    elif not settings.get("runtime", {}).get("local_only", False):
        raise ValueError("Local startup requires runtime.local_only=true.")
    return settings


def paths_for(settings: dict) -> dict:
    storage = settings["storage"]
    return {
        "runtime_root": absolute(storage["local_runtime_root"]),
        "archive_root": absolute(storage["archive_root"]),
        "input_root": absolute(storage["local_input_root"]),
        "cache_root": absolute(storage["local_cache_root"]),
        "staging_root": absolute(storage["local_staging_root"]),
        "output_root": absolute(settings["project"]["output_root"]),
        "index_csv": absolute(storage["index_csv"]),
        "first_weight": absolute(settings["models"]["first_person"]),
        "third_weight": absolute(settings["models"]["third_person"]),
        "first_engine": absolute(settings["models"]["first_person_engine"]),
        "third_engine": absolute(settings["models"]["third_person_engine"]),
        "mllm_enabled": str(bool(settings["mllm"].get("enabled", False))).lower(),
        "api_key_env": settings["mllm"].get("api_key_env", "VISIONCORTEX_MODEL_API_KEY"),
    }


def quoted(value: str, *, argument: bool = False) -> str:
    value = value.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%")
    if argument:
        value = value.replace("$", "$$")
    return '"' + value + '"'


def render(project: str, config: str, mode: str, template: str, target: str, port: str) -> None:
    project, config = absolute(project), absolute(config)
    if "\\" in project or project.endswith(" "):
        raise ValueError("Project directory must not contain backslashes or trailing spaces.")
    settings = settings_for(project, config, mode)
    paths = paths_for(settings)
    environment = {
        "VISIONCORTEX_PROJECT_ROOT": project,
        "VISIONCORTEX_PYTHON": os.path.abspath(sys.executable),
        "VISIONCORTEX_CONFIG": config,
        "VISIONCORTEX_DEFAULT_CONFIG": absolute(os.environ.get("VISIONCORTEX_DEFAULT_CONFIG", str(Path(project) / "configs/default.yaml"))),
        "VISIONCORTEX_DEPLOYMENT_MODE": mode,
        "VISIONCORTEX_WEB_PORT": port,
        "VISIONCORTEX_WEB_HOST": "127.0.0.1" if mode == "local" else "0.0.0.0",
        "PYTHONPATH": str(Path(project) / "src"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "TMPDIR": str(Path(paths["runtime_root"]) / "tmp"),
    }
    environment.update({
        "VISIONCORTEX_NAS_INDEX_CSV": paths["index_csv"],
        "VISIONCORTEX_NAS_ARCHIVE_ROOT": paths["archive_root"],
        "VISIONCORTEX_LOCAL_INPUT_ROOT": paths["input_root"],
        "VISIONCORTEX_LOCAL_RUNTIME_ROOT": paths["runtime_root"],
        "VISIONCORTEX_LOCAL_CACHE_ROOT": paths["cache_root"],
        "VISIONCORTEX_LOCAL_STAGING_ROOT": paths["staging_root"],
        "VISIONCORTEX_OUTPUT_ROOT": paths["output_root"],
        "VISIONCORTEX_FIRST_PERSON_ENGINE": paths["first_engine"],
        "VISIONCORTEX_THIRD_PERSON_ENGINE": paths["third_engine"],
        "VISIONCORTEX_RUNTIME_ROLE": settings["runtime"].get("role", "combined"),
        "VISIONCORTEX_TENSORRT": str(settings["performance"].get("tensor_rt", "off")),
    })
    # Forward safe policy and file paths only. Never embed a credential value.
    for name in ("VISIONCORTEX_WEB_USERNAME", "VISIONCORTEX_WEB_PASSWORD_FILE", "VISIONCORTEX_WEB_ALLOWED_NETWORKS",
                 "VISIONCORTEX_ARK_API_KEY_FILE", "VISIONCORTEX_MODEL_API_KEY_FILE", "VISIONCORTEX_WEB_AI_SETTINGS"):
        if name in os.environ:
            environment[name] = os.environ[name]
    if mode == "production":
        environment["VISIONCORTEX_WEB_ACCESS_MODE"] = os.environ.get("VISIONCORTEX_WEB_ACCESS_MODE", "lan")
    if any(any(ord(char) < 32 or ord(char) == 127 for char in value) for value in environment.values()):
        raise ValueError("Service values must not contain control characters.")
    text = Path(template).read_text(encoding="utf-8")
    values = {
        "@PROJECT_ROOT@": project.replace("%", "%%"),
        "@ENVIRONMENT@": "\n".join("Environment=" + quoted(name + "=" + value) for name, value in environment.items()),
        "@RUNNER@": quoted(absolute(os.environ.get("VISIONCORTEX_DEPLOYMENT_RUNNER", str(Path(project) / "deployment/rtx3090ti-ubuntu/06-Run-LAN-Server.sh"))), argument=True),
    }
    if "@PREFLIGHT@" in text:
        values["@PREFLIGHT@"] = quoted(str(Path(project) / "deployment/rtx3090ti-ubuntu/00-Preflight.sh"), argument=True)
    for marker, value in values.items():
        if text.count(marker) != 1:
            raise ValueError("Invalid service template.")
        text = text.replace(marker, value)
    Path(target).write_text(text, encoding="utf-8")


def main() -> None:
    command, *args = sys.argv[1:]
    if command == "snapshot":
        project, config, mode = args
        for name, value in paths_for(settings_for(absolute(project), absolute(config), mode)).items():
            print(name + "=" + shlex.quote(value))
    elif command == "render":
        render(*args)
    elif command == "render-rtp":
        project, python, config, template, target = args
        project, config = absolute(project), absolute(config)
        if not Path(config).is_file() or not os.access(python, os.X_OK):
            raise ValueError("Prepared receiver configuration or Python is missing.")
        json.loads(Path(config).read_text(encoding="utf-8"))
        text = Path(template).read_text(encoding="utf-8")
        for marker, value in {
            "@PROJECT_ROOT@": project.replace("%", "%%"),
            "@ENVIRONMENT@": "Environment=" + quoted("PYTHONPATH=" + str(Path(project) / "src")) + "\nEnvironment=PYTHONDONTWRITEBYTECODE=1",
            "@PYTHON@": quoted(os.path.abspath(python)),
            "@CONFIG@": quoted(config, argument=True),
        }.items():
            if text.count(marker) != 1:
                raise ValueError("Invalid receiver service template.")
            text = text.replace(marker, value)
        Path(target).write_text(text, encoding="utf-8")
    elif command == "owner":
        project, unit = args
        expected = "WorkingDirectory=" + absolute(project).replace("%", "%%")
        if expected not in Path(unit).read_text(encoding="utf-8").splitlines():
            raise ValueError("The selected service belongs to another checkout; choose a unique VISIONCORTEX_SERVICE_NAME and port.")
    elif command == "desktop":
        project, config, service, port, target = args
        values = ["/usr/bin/env", "VISIONCORTEX_PROJECT_ROOT=" + project,
                  "VISIONCORTEX_CONFIG=" + config, "VISIONCORTEX_SERVICE_NAME=" + service,
                  "VISIONCORTEX_WEB_PORT=" + port, "VISIONCORTEX_PYTHON=" + sys.executable,
                  "VISIONCORTEX_DEPLOYMENT_MODE=" + os.environ.get("VISIONCORTEX_DEPLOYMENT_MODE", "local"),
                  "/usr/bin/bash", absolute(os.environ.get("VISIONCORTEX_DESKTOP_RUNNER", str(Path(project) / "deployment/rtx3090ti-ubuntu/Open-VisionCortex.sh")))]
        # Forward only policy and credential file locations, never key values.
        for name in ("VISIONCORTEX_DEFAULT_CONFIG", "VISIONCORTEX_WEB_ACCESS_MODE",
                     "VISIONCORTEX_WEB_USERNAME", "VISIONCORTEX_WEB_PASSWORD_FILE",
                     "VISIONCORTEX_MODEL_API_KEY_FILE", "VISIONCORTEX_ARK_API_KEY_FILE",
                     "VISIONCORTEX_WEB_AI_SETTINGS"):
            if name in os.environ:
                values.insert(-2, name + "=" + os.environ[name])
        def desktop_quote(value):
            for char in ("\\", '"', "`", "$"):
                value = value.replace(char, "\\" + char)
            return '"' + value.replace("%", "%%") + '"'
        Path(target).write_text(
            "[Desktop Entry]\nType=Application\nName=VisionCortex\nExec="
            + " ".join(desktop_quote(value) for value in values)
            + "\nIcon=applications-science\nTerminal=false\nCategories=Science;Education;\n"
              "StartupNotify=true\nStartupWMClass=VisionCortex\n", encoding="utf-8")
    elif command == "port":
        with socket.socket() as listener:
            try:
                listener.bind(("0.0.0.0", int(args[0])))
            except OSError as exc:
                raise ValueError("The selected port is occupied; choose a separate instance port.") from exc
    elif command == "process":
        project, config, pid = args
        process = Path("/proc") / pid
        argv = (process / "cmdline").read_bytes().split(b"\0")
        decoded = [part.decode() for part in argv if part]
        if Path(process / "cwd").resolve() != Path(project).resolve() or decoded[1:5] != ["-m", "visioncortex", "serve", "--host"]:
            raise ValueError("Recorded PID is not owned by this checkout.")
        if "--config" not in decoded or Path(decoded[decoded.index("--config") + 1]).resolve() != Path(config).resolve():
            raise ValueError("Recorded PID uses another configuration.")
    elif command in {"health", "health-stop"}:
        port, archive, pid = args
        headers = {}
        if os.environ.get("VISIONCORTEX_WEB_ACCESS_MODE") == "lan":
            password = Path(os.environ["VISIONCORTEX_WEB_PASSWORD_FILE"]).read_text().strip()
            username = os.environ.get("VISIONCORTEX_WEB_USERNAME", "visioncortex")
            headers["Authorization"] = "Basic " + base64.b64encode(f"{username}:{password}".encode()).decode()
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        request = urllib.request.Request(f"http://127.0.0.1:{port}/api/health", headers=headers)
        with opener.open(request, timeout=2) as response:
            data = json.loads(response.read(65537))
        if data.get("status") != "ok" or data.get("product_name") != "VisionCortex" or Path(data.get("archive_root", "")).resolve() != Path(archive).resolve():
            raise ValueError("Health response belongs to another application or runtime.")
        identity_request = urllib.request.Request(f"http://127.0.0.1:{port}/health/automation", headers=headers)
        with opener.open(identity_request, timeout=2) as response:
            identity = json.loads(response.read(65537))
        if identity.get("schema_version") != "visioncortex-automation-readiness/1":
            raise ValueError("Startup identity response is invalid.")
        if pid and (type(identity.get("pid")) is not int or identity["pid"] != int(pid)):
            raise ValueError("Health response belongs to another process.")
        config = absolute(os.environ["VISIONCORTEX_CONFIG"])
        default = absolute(os.environ["VISIONCORTEX_DEFAULT_CONFIG"])
        settings = settings_for(absolute(os.environ["VISIONCORTEX_PROJECT_ROOT"]), config,
                                os.environ.get("VISIONCORTEX_DEPLOYMENT_MODE", "local"))
        from visioncortex.automation_readiness import configuration_digest

        expected = {"config_path": config, "default_config_path": default,
                    "settings_sha256": configuration_digest(settings)}
        if identity.get("configuration") != expected:
            raise ValueError("Running service configuration differs; wait for tasks and stop it explicitly before reinstalling.")
        if command == "health-stop":
            safety = data.get("shutdown_safety") or {}
            if safety.get("provable") is not True or safety.get("active_tasks") != 0:
                raise ValueError("Active or unprovable work prevents stopping; wait for tasks to finish.")
    else:
        raise ValueError("Unsupported deployment helper command.")


if __name__ == "__main__":
    try:
        main()
    except (KeyError, OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1) from None
