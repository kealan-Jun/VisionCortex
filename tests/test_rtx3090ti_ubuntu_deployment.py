from __future__ import annotations

import os
import json
import re
import shlex
import shutil
import socket
import tempfile
import subprocess
import sys
from pathlib import Path

import pytest

from visioncortex.api import _folder_open_command


ROOT = Path(__file__).resolve().parents[1]
DEPLOYMENT = ROOT / "deployment" / "rtx3090ti-ubuntu"


def test_ubuntu_shell_scripts_are_syntactically_valid_and_gpu_scoped():
    if os.name == "nt":
        pytest.skip("Ubuntu shell syntax is validated by the Ubuntu CI job")
    for path in DEPLOYMENT.glob("*.sh"):
        subprocess.run(["bash", "-n", str(path)], check=True)
    preflight = (DEPLOYMENT / "00-Preflight.sh").read_text()
    installer = (DEPLOYMENT / "01-Install-And-Validate.sh").read_text()
    assert "RTX 3090 Ti" in preflight and "--production" in preflight
    assert "minimum_free_gib=30" in preflight
    assert "prepare-engine" in installer and "prepare-public-models" in installer
    assert 'export PATH="$venv/bin:$PATH"' in installer
    assert "sha256sum" in installer
    for path in [*DEPLOYMENT.glob("*.sh"), *DEPLOYMENT.glob("*.service")]:
        assert not re.search(r"/home/[^/\s]+/", path.read_text())
        assert not re.search(r"/srv/[^/\s]+/(?:VisionCortex[^/\s]*|LocalWorkspaces)/", path.read_text())


def test_ubuntu_dependency_set_is_linux_pinned():
    requirements = (DEPLOYMENT / "requirements-lock.txt").read_text(encoding="utf-8")

    assert "torch==2.6.0" in requirements
    assert "torchvision==0.21.0" in requirements
    assert "https://mirrors.aliyun.com/pypi/simple/" in requirements
    assert "tensorrt-cu12==10.16.1.11" in requirements
    assert "onnx==1.22.0" in requirements
    assert "onnxruntime-gpu==1.29.0" in requirements
    assert "onnxslim==0.1.96" in requirements
    assert "ml-dtypes==0.5.4" in requirements
    assert "transformers==4.57.6" in requirements
    assert "scipy==1.17.1" in requirements
    assert "hydra-core==1.3.2" in requirements
    assert "iopath==0.1.10" in requirements
    assert "ultralytics/CLIP.git@68dce32140994dfcb645a1320c4ebdc034fc19fd" in requirements
    assert "https://pypi.nvidia.com/" in requirements
    assert "win_amd64" not in requirements

    installer = (DEPLOYMENT / "01-Install-And-Validate.sh").read_text(
        encoding="utf-8"
    )
    assert "SAM2_BUILD_CUDA=0" in installer
    assert "--no-build-isolation --no-deps" in installer
    assert "2b90b9f5ceec907a1c18123530e92e794ad901a4" in installer


def test_web_lifecycle_is_local_and_pid_scoped():
    start = (DEPLOYMENT / "02-Start-Web.sh").read_text()
    stop = (DEPLOYMENT / "03-Stop-Web.sh").read_text()
    assert "mode=local" in start and "--production" in start
    assert "--host 127.0.0.1" in start
    assert "render_service.py\" process" in start
    assert "render_service.py\" process" in stop
    assert "kill -9" not in stop


def test_systemd_services_are_verified_templates_and_never_stop_other_instances():
    for name in ("local", "lan", "analysis"):
        unit = (DEPLOYMENT / f"visioncortex-{name}.service").read_text()
        assert all(marker in unit for marker in ("@PROJECT_ROOT@", "@ENVIRONMENT@", "@RUNNER@"))
        assert "WantedBy=default.target" in unit and "UMask=0077" in unit
    for name in ("05-Install-Local-Service.sh", "07-Install-LAN-Server.sh"):
        installer = (DEPLOYMENT / name).read_text()
        assert 'systemd-analyze --user verify "$rendered_unit"' in installer
        assert "deployment_unit_guard" in installer
        assert "disable --now" not in installer
        assert 'install -m 0644 -- "$unit_source"' not in installer
    launcher = (DEPLOYMENT / "Open-VisionCortex.sh").read_text()
    assert "render_service.py\" owner" in launcher
    assert '--user-data-dir="$browser_profile"' in launcher
    assert "browser_scale='1.5'" in launcher
    assert '--force-device-scale-factor="$browser_scale"' in launcher
    assert '--app="$url"' in launcher


def test_analysis_service_keeps_runtime_identity_and_readiness_gate():
    installer = (DEPLOYMENT / "09-Install-Analysis-Service.sh").read_text()
    assert "explicit prepared VISIONCORTEX_CONFIG" in installer
    assert 'systemd-analyze --user verify "$rendered_unit"' in installer
    assert 'mv -fT -- "$unit_temp" "$unit_target"' in installer
    assert "configuration_blockers(settings, role(settings))" in installer
    assert "--property MainPID --value" in installer
    assert '"http://127.0.0.1:$port/health/automation"' in installer


@pytest.fixture
def relocated_analysis_host(tmp_path, monkeypatch, request):
    if os.name == "nt":
        pytest.skip("systemd user-unit rendering targets Ubuntu")
    if sys.version_info[:2] not in {(3, 11), (3, 12)}:
        pytest.skip("the prepared deployment interpreter must be Python 3.11 or 3.12")
    import yaml

    project = tmp_path / '客户 repo $(touch injected) % "quoted"'
    deployment = project / "deployment/rtx3090ti-ubuntu"
    deployment.mkdir(parents=True)
    for name in ("09-Install-Analysis-Service.sh", "06-Run-LAN-Server.sh", "visioncortex-analysis.service", "render_service.py", "_common.sh"):
        shutil.copy2(DEPLOYMENT / name, deployment / name)
    (project / "src").symlink_to(ROOT / "src", target_is_directory=True)
    configs = project / "configs"
    configs.mkdir()
    shutil.copy2(ROOT / "configs/default.yaml", configs / "default.yaml")
    nas = tmp_path / "prepared NAS"
    archive = nas / "VisionCortexExperimentArchive"
    staging = archive / ".VisionCortex-Run-Staging"
    cache = nas / "VisionCortexExperimentCache"
    runtime = tmp_path / "local runtime"
    for directory in (staging, cache, runtime):
        directory.mkdir(parents=True)
    engine_root = tmp_path / "prepared Engines"
    engine_root.mkdir()
    for role in ("first_person", "third_person"):
        (engine_root / f"{role}.engine").write_bytes(b"test-only engine placeholder")
    config = configs / "customer.yaml"
    config.write_text(yaml.safe_dump({
        "runtime": {"role": "combined", "local_only": False},
        "device_day": {"enabled": True, "paused_stages": ["understanding", "report"], "delete_capture_sources": False},
        "storage": {"archive_root": str(archive), "local_input_root": str(runtime / "Input-Manifests"),
                    "local_runtime_root": str(runtime), "local_cache_root": str(cache), "local_staging_root": str(staging),
                    "sync_to_nas": True},
        "project": {"output_root": str(staging), "run_purpose": "analysis"},
        "collection_ingest": {"source_root": str(nas), "enabled": True, "mode": "directory_metadata"},
        "models": {f"{role}_engine": str(engine_root / f"{role}.engine") for role in ("first_person", "third_person")},
    }), encoding="utf-8")
    home = tmp_path / "customer home"
    home.mkdir()
    bus_temp = tempfile.TemporaryDirectory(prefix="vc-user-bus-")
    request.addfinalizer(bus_temp.cleanup)
    user_bus = Path(bus_temp.name)
    bus = socket.socket(socket.AF_UNIX)
    bus.bind(str(user_bus / "bus"))
    bus.close()
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    log = tmp_path / "commands.log"
    verified = tmp_path / "verified.service"
    readiness = tmp_path / "readiness.json"
    from visioncortex.automation_readiness import configuration_digest
    from visioncortex.config import load_config

    with monkeypatch.context() as prepared_environment:
        for name in list(os.environ):
            if name.startswith("VISIONCORTEX_"):
                prepared_environment.delenv(name)
        prepared_environment.setenv("VISIONCORTEX_DEFAULT_CONFIG", str(configs / "default.yaml"))
        prepared_environment.setenv("VISIONCORTEX_TENSORRT", "required")
        prepared_settings = load_config(config)
    readiness.write_text(json.dumps({
        "schema_version": "visioncortex-automation-readiness/1", "ready": True, "pid": 4242,
        "configuration": {"config_path": str(config), "default_config_path": str(configs / "default.yaml"),
                          "settings_sha256": configuration_digest(prepared_settings)},
        "storage_maintenance": False,
        "runtime_role": "combined", "worker": {"status": "running", "pid": 4242},
        "nas_monitor": {"thread_alive": True, "status": "watching"},
        "device_day": {"thread_alive": True, "runner_initialized": True,
                       "status": "waiting_for_nas_monitor", "paused_stages": ["understanding", "report"]},
        "blockers": [],
    }), encoding="utf-8")
    commands = {
        "systemctl": 'printf "systemctl %s\\n" "$*" >> "$TEST_COMMAND_LOG"\n'
                     'expected_service=${TEST_EXPECTED_SERVICE:-visioncortex-analysis.service}\n'
                     'if [[ $* == "--user show $expected_service --property MainPID --value" ]]; then\n'
                     '  printf "%s\\n" "${TEST_MAIN_PID:-4242}"\n'
                     'elif [[ $* == "--user is-active --quiet $expected_service" ]]; then\n'
                     '  [[ ${TEST_SERVICE_ACTIVE:-0} == 1 ]]\n'
                     'elif [[ $* != "--user daemon-reload" && $* != "--user enable --now $expected_service"\n'
                     '     && $* != "--user --no-pager --full status $expected_service" ]]; then exit 99\n'
                     'fi\n',
        "loginctl": 'printf "yes\\n"\n',
        "curl": 'printf "curl %s\\n" "$*" >> "$TEST_COMMAND_LOG"\n'
                '[[ ${TEST_CURL_FAILURE:-0} == 0 ]] || exit 22\n'
                'output=""; url=""\n'
                'while (( $# )); do\n'
                '  if [[ $1 == --output ]]; then output=$2; shift 2;\n'
                '  elif [[ $1 == http://* ]]; then url=$1; shift; else shift; fi\n'
                'done\n'
                '[[ $url == "http://127.0.0.1:${TEST_EXPECTED_PORT:-8001}/health/automation" ]]\n'
                'count=$(cat "$TEST_CURL_COUNT" 2>/dev/null || printf "0")\n'
                'count=$((count + 1))\n'
                'printf "%s\\n" "$count" > "$TEST_CURL_COUNT"\n'
                'if [[ $count == 1 && -n ${TEST_FIRST_READINESS_BODY:-} ]]; then\n'
                '  cp -- "$TEST_FIRST_READINESS_BODY" "$output"\n'
                'else cp -- "$TEST_READINESS_BODY" "$output"; fi\n',
        "seq": '[[ $* == "1 180" ]]\n'
               'printf "1\\n2\\n3\\n"\n',
        "sleep": 'printf "sleep %s\\n" "$*" >> "$TEST_COMMAND_LOG"\n',
        "systemd-analyze": 'printf "verify %s\\n" "$*" >> "$TEST_COMMAND_LOG"\n'
                           '[[ ${TEST_REJECT_UNIT:-0} == 0 ]] || exit 1\n'
                           'cp -- "${@: -1}" "$TEST_VERIFIED_UNIT"\n',
    }
    for name, body in commands.items():
        command = fake_bin / name
        command.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + body, encoding="utf-8")
        command.chmod(0o755)
    env = {name: value for name, value in os.environ.items() if not name.startswith("VISIONCORTEX_")}
    env.update(HOME=str(home), XDG_CONFIG_HOME=str(home / ".config"), XDG_RUNTIME_DIR=str(user_bus),
               PATH=f"{fake_bin}{os.pathsep}{env['PATH']}", PYTHONDONTWRITEBYTECODE="1",
               VISIONCORTEX_PYTHON=sys.executable, VISIONCORTEX_CONFIG=str(config),
               TEST_COMMAND_LOG=str(log), TEST_VERIFIED_UNIT=str(verified), TEST_READINESS_BODY=str(readiness),
               TEST_CURL_COUNT=str(tmp_path / "curl-count"))
    return {"project": project, "installer": deployment / "09-Install-Analysis-Service.sh", "config": config,
            "runtime": runtime, "archive": archive, "env": env, "log": log, "verified": verified,
            "unit": home / ".config/systemd/user/visioncortex-analysis.service", "readiness": readiness}


def test_analysis_install_renders_relocated_paths_and_preserves_prepared_config(relocated_analysis_host):
    host = relocated_analysis_host
    original = host["config"].read_bytes()
    result = subprocess.run(["bash", str(host["installer"])], env=host["env"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    unit = host["unit"].read_text(encoding="utf-8")
    assert host["verified"].read_text(encoding="utf-8") == unit
    assert "@PROJECT_ROOT@" not in unit and "@ENVIRONMENT@" not in unit and "@RUNNER@" not in unit
    environment = {}
    for line in unit.splitlines():
        if line.startswith("Environment="):
            name, value = shlex.split(line.removeprefix("Environment="))[0].split("=", 1)
            environment[name] = value.replace("%%", "%")
    assert environment["VISIONCORTEX_CONFIG"] == str(host["config"])
    assert environment["VISIONCORTEX_PYTHON"] == sys.executable
    assert environment["VISIONCORTEX_NAS_ARCHIVE_ROOT"] == str(host["archive"])
    assert environment["VISIONCORTEX_LOCAL_RUNTIME_ROOT"] == str(host["runtime"])
    assert environment["VISIONCORTEX_ARK_API_KEY_FILE"] == str(Path(host["env"]["XDG_CONFIG_HOME"]) / "VisionCortex/ark_api_key")
    assert environment["VISIONCORTEX_WEB_HOST"] == "127.0.0.1"
    assert environment["VISIONCORTEX_WEB_PORT"] == "8001"
    assert environment["VISIONCORTEX_RUNTIME_ROLE"] == "combined"
    assert environment["PYTHONDONTWRITEBYTECODE"] == "1"
    probe = host["runtime"] / "unit_cache_probe.py"
    probe.write_text('value = "owned unit import probe"\n')
    probe_environment = {name: value for name, value in os.environ.items() if name != "PYTHONPYCACHEPREFIX"}
    probe_environment.update(PYTHONDONTWRITEBYTECODE=environment["PYTHONDONTWRITEBYTECODE"],
                             PYTHONPATH=str(host["runtime"]))
    probe_result = subprocess.run([sys.executable, "-c", "import unit_cache_probe; print(unit_cache_probe.value)"],
                                  cwd=host["runtime"], env=probe_environment, capture_output=True, text=True, check=True)
    assert probe_result.stdout.strip() == "owned unit import probe"
    assert not list(host["runtime"].rglob('*.pyc'))
    exec_line = next(line for line in unit.splitlines() if line.startswith("ExecStart="))
    executable, argument = shlex.split(exec_line.removeprefix("ExecStart="))
    assert executable == "/usr/bin/bash"
    command = argument.replace("%%", "%").replace("$$", "$")
    assert command == str(host["installer"].with_name("06-Run-LAN-Server.sh"))
    assert not (host["project"] / "injected").exists()
    assert host["config"].read_bytes() == original
    commands = host["log"].read_text().splitlines()
    assert commands[0].startswith("verify --user verify ")
    assert "systemctl --user enable --now visioncortex-analysis.service" in commands
    assert "NAS 自动处理后台已启动" in result.stdout
    assert "视频理解、日报" in result.stdout
    assert "不代表真实模型执行" in result.stdout


def test_analysis_install_targets_only_named_second_instance(relocated_analysis_host):
    host = relocated_analysis_host
    service = "visioncortex-selftest-20261009.service"
    host["unit"].parent.mkdir(parents=True)
    host["unit"].write_text("existing production service", encoding="utf-8")
    inode = host["unit"].stat().st_ino
    env = host["env"] | {"VISIONCORTEX_SERVICE_NAME": service, "VISIONCORTEX_WEB_PORT": "18002",
                          "TEST_EXPECTED_SERVICE": service, "TEST_EXPECTED_PORT": "18002"}
    result = subprocess.run(["bash", str(host["installer"])], env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    unit = host["unit"].with_name(service)
    assert unit.is_file()
    assert unit.read_bytes() == host["verified"].read_bytes()
    assert 'Environment="VISIONCORTEX_WEB_PORT=18002"' in unit.read_text()
    assert 'Environment="VISIONCORTEX_WEB_HOST=127.0.0.1"' in unit.read_text()
    assert host["unit"].read_text() == "existing production service"
    assert host["unit"].stat().st_ino == inode
    commands = host["log"].read_text()
    assert f"systemctl --user enable --now {service}" in commands
    assert f"systemctl --user show {service} --property MainPID --value" in commands
    assert "visioncortex-analysis.service" not in commands
    assert "http://127.0.0.1:18002/#/home" in result.stdout
    analyzer = shutil.which("systemd-analyze")
    if analyzer is not None:
        parsed = subprocess.run([analyzer, "--system", "verify", str(unit)], capture_output=True, text=True)
        assert parsed.returncode == 0, parsed.stderr


@pytest.mark.parametrize("name", ["", "sshd.service", "Visioncortex-test.service", "visioncortex-.service",
                                 "../visioncortex-test.service", "visioncortex-test/other.service",
                                 "visioncortex-test.service\nExecStart=/bin/false", "visioncortex-test@other.service",
                                 "visioncortex-客户.service", "visioncortex-test%u.service",
                                 "visioncortex-" + "a" * 64 + ".service"])
def test_analysis_install_rejects_unsafe_instance_name_before_any_action(relocated_analysis_host, name):
    host = relocated_analysis_host
    result = subprocess.run(["bash", str(host["installer"])], env=host["env"] | {"VISIONCORTEX_SERVICE_NAME": name},
                            capture_output=True, text=True, timeout=15)
    assert result.returncode != 0
    assert "Invalid VisionCortex service name" in result.stderr
    assert not host["unit"].exists()
    assert not host["log"].exists()


@pytest.mark.parametrize("port", ["", "0", "1023", "65536", "-1", "+8001", "8.5", "８００１", " 8001",
                                 "8001\n", "8001;touch injected", "999999999999999999999"])
def test_analysis_install_rejects_invalid_port_before_any_action(relocated_analysis_host, port):
    host = relocated_analysis_host
    result = subprocess.run(["bash", str(host["installer"])], env=host["env"] | {"VISIONCORTEX_WEB_PORT": port},
                            capture_output=True, text=True, timeout=15)
    assert result.returncode != 0
    assert "VisionCortex Web port" in result.stderr
    assert not host["unit"].exists()
    assert not host["log"].exists()
    assert not (host["project"] / "injected").exists()


@pytest.mark.parametrize("port", ["1024", "65535", "08002"])
def test_analysis_install_normalizes_valid_custom_port(relocated_analysis_host, port):
    host = relocated_analysis_host
    normalized = str(int(port))
    env = host["env"] | {"VISIONCORTEX_WEB_PORT": port, "TEST_EXPECTED_PORT": normalized}
    result = subprocess.run(["bash", str(host["installer"])], env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert f'Environment="VISIONCORTEX_WEB_PORT={normalized}"' in host["unit"].read_text()
    assert f"http://127.0.0.1:{normalized}/#/home" in result.stdout


@pytest.mark.parametrize("invalid", ["web_role", "worker_role", "environment_web_role", "device_day_disabled",
                                     "ingest_disabled", "wrong_ingest_mode", "nas_disabled", "local_only"])
def test_analysis_install_rejects_non_automatic_profile_before_mutation(relocated_analysis_host, invalid):
    import yaml

    host = relocated_analysis_host
    settings = yaml.safe_load(host["config"].read_text())
    env = host["env"].copy()
    if invalid in {"web_role", "worker_role"}:
        settings["runtime"]["role"] = invalid.removesuffix("_role")
    elif invalid == "environment_web_role":
        env["VISIONCORTEX_RUNTIME_ROLE"] = "web"
    elif invalid == "device_day_disabled":
        settings["device_day"]["enabled"] = False
    elif invalid == "ingest_disabled":
        settings["collection_ingest"]["enabled"] = False
    elif invalid == "wrong_ingest_mode":
        settings["collection_ingest"]["mode"] = "index_metadata_poll"
    elif invalid == "nas_disabled":
        settings["storage"]["sync_to_nas"] = False
    else:
        settings["runtime"]["local_only"] = True
    host["config"].write_text(yaml.safe_dump(settings), encoding="utf-8")
    original = host["config"].read_bytes()
    host["unit"].parent.mkdir(parents=True)
    host["unit"].write_text("existing unit", encoding="utf-8")
    result = subprocess.run(["bash", str(host["installer"])], env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode != 0
    assert host["unit"].read_text() == "existing unit"
    assert host["config"].read_bytes() == original
    assert not (host["runtime"] / "Input-Manifests").exists()
    assert not host["log"].exists()


def test_analysis_install_waits_for_background_readiness(relocated_analysis_host):
    host = relocated_analysis_host
    initial = host["readiness"].with_name("not-ready.json")
    payload = json.loads(host["readiness"].read_text())
    payload.update(ready=False, blockers=["nas_monitor_starting"])
    initial.write_text(json.dumps(payload), encoding="utf-8")
    env = host["env"] | {"TEST_FIRST_READINESS_BODY": str(initial)}
    result = subprocess.run(["bash", str(host["installer"])], env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    commands = host["log"].read_text().splitlines()
    assert sum(line.startswith("curl ") for line in commands) == 2
    assert "sleep 1" in commands
    assert "NAS 自动处理后台已启动" in result.stdout


@pytest.mark.parametrize("failure", ["not_ready", "wrong_pid", "wrong_config", "wrong_default_config", "invalid_json",
                                     "wrong_schema", "truthy_ready", "wrong_role", "wrong_worker_pid", "worker_not_running",
                                     "monitor_stopped", "dispatcher_stopped", "runner_missing", "invalid_paused",
                                     "missing_main_pid", "request_failure", "wrong_settings_digest", "missing_settings_digest",
                                     "monitor_retrying", "dispatcher_failed", "storage_maintenance"])
def test_analysis_install_cannot_succeed_on_http_response_alone(relocated_analysis_host, failure):
    host = relocated_analysis_host
    payload = json.loads(host["readiness"].read_text())
    env = host["env"].copy()
    if failure == "not_ready":
        payload["ready"] = False
    elif failure == "wrong_pid":
        payload["pid"] = 4243
    elif failure == "wrong_config":
        payload["configuration"]["config_path"] = "/other/profile.yaml"
    elif failure == "wrong_default_config":
        payload["configuration"]["default_config_path"] = "/other/default.yaml"
    elif failure == "wrong_schema":
        payload["schema_version"] = "wrong/1"
    elif failure == "truthy_ready":
        payload["ready"] = "true"
    elif failure == "wrong_role":
        payload["runtime_role"] = "web"
    elif failure == "wrong_worker_pid":
        payload["worker"]["pid"] = 4243
    elif failure == "worker_not_running":
        payload["worker"]["status"] = "not_started"
    elif failure == "monitor_stopped":
        payload["nas_monitor"]["thread_alive"] = False
    elif failure == "dispatcher_stopped":
        payload["device_day"]["thread_alive"] = False
    elif failure == "runner_missing":
        payload["device_day"]["runner_initialized"] = False
    elif failure == "invalid_paused":
        payload["device_day"]["paused_stages"] = ["unknown"]
    elif failure == "missing_main_pid":
        env["TEST_MAIN_PID"] = "0"
    elif failure == "request_failure":
        env["TEST_CURL_FAILURE"] = "1"
    elif failure == "wrong_settings_digest":
        payload["configuration"]["settings_sha256"] = "0" * 64
    elif failure == "missing_settings_digest":
        del payload["configuration"]["settings_sha256"]
    elif failure == "monitor_retrying":
        payload["nas_monitor"]["status"] = "retrying"
    elif failure == "dispatcher_failed":
        payload["device_day"]["status"] = "failed"
    elif failure == "storage_maintenance":
        payload["storage_maintenance"] = True
    host["readiness"].write_text("not JSON" if failure == "invalid_json" else json.dumps(payload), encoding="utf-8")
    result = subprocess.run(["bash", str(host["installer"])], env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode != 0
    assert "NAS 自动处理后台未就绪" in result.stderr
    assert "NAS 自动处理后台已启动" not in result.stdout
    commands = host["log"].read_text().splitlines()
    assert sum(line == "sleep 1" for line in commands) == 3
    assert sum(line.startswith("curl ") for line in commands) == (0 if failure == "missing_main_pid" else 3)
    assert "systemctl --user --no-pager --full status visioncortex-analysis.service" in commands


def test_analysis_install_rejects_active_old_settings_at_same_config_path(relocated_analysis_host):
    import yaml

    host = relocated_analysis_host
    first = subprocess.run(["bash", str(host["installer"])], env=host["env"], capture_output=True, text=True, timeout=15)
    assert first.returncode == 0, first.stderr
    original_unit = host["unit"].read_bytes()
    inode = host["unit"].stat().st_ino
    host["log"].unlink()
    settings = yaml.safe_load(host["config"].read_text())
    settings["collection_ingest"]["poll_seconds"] = 17
    host["config"].write_text(yaml.safe_dump(settings), encoding="utf-8")
    # The old active service still reports its original startup settings digest.
    result = subprocess.run(["bash", str(host["installer"])], env=host["env"] | {"TEST_SERVICE_ACTIVE": "1"},
                            capture_output=True, text=True, timeout=15)
    assert result.returncode != 0
    assert "后台实际配置与本次安装配置不一致" in result.stderr
    assert host["unit"].read_bytes() == original_unit
    assert host["unit"].stat().st_ino == inode
    commands = host["log"].read_text()
    assert "restart" not in commands and "systemctl --user stop" not in commands


@pytest.mark.parametrize("failure", ["verification", "missing_storage", "control_character", "relative_config", "project_backslash"])
def test_analysis_install_rejects_invalid_prepared_paths_before_replacing_unit(relocated_analysis_host, failure):
    host = relocated_analysis_host
    host["unit"].parent.mkdir(parents=True)
    host["unit"].write_text("original installed unit", encoding="utf-8")
    env = host["env"].copy()
    if failure == "verification":
        env["TEST_REJECT_UNIT"] = "1"
    elif failure == "missing_storage":
        shutil.rmtree(host["archive"])
    elif failure == "control_character":
        env["VISIONCORTEX_ARK_API_KEY_FILE"] = str(host["runtime"] / "key\nExecStart=/bin/false")
    elif failure == "relative_config":
        env["VISIONCORTEX_CONFIG"] = "relative-config.yaml"
    else:
        env["VISIONCORTEX_PROJECT_ROOT"] = str(host["project"].parent / "trailing-backslash\\")
    result = subprocess.run(["bash", str(host["installer"])], env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert host["unit"].read_text(encoding="utf-8") == "original installed unit"
    assert not host["runtime"].joinpath("Input-Manifests").exists()
    assert not host["log"].exists() or "systemctl" not in host["log"].read_text()


def test_analysis_rendered_unit_passes_real_systemd_static_parser(relocated_analysis_host):
    analyzer = shutil.which("systemd-analyze")
    if analyzer is None:
        pytest.skip("systemd-analyze is unavailable")
    host = relocated_analysis_host
    rendered = subprocess.run(["bash", str(host["installer"])], env=host["env"], capture_output=True, text=True)
    assert rendered.returncode == 0, rendered.stderr
    # --system only parses the owned temporary unit; it neither starts a manager
    # nor executes the service. User-manager verification may lack a login bus.
    verified = subprocess.run([analyzer, "--system", "verify", str(host["unit"])], capture_output=True, text=True)
    assert verified.returncode == 0, verified.stderr


def test_analysis_install_refuses_to_replace_different_active_service(relocated_analysis_host):
    host = relocated_analysis_host
    host["unit"].parent.mkdir(parents=True)
    host["unit"].write_text("WorkingDirectory=" + str(host["project"]).replace("%", "%%") + "\nprevious configuration", encoding="utf-8")
    env = host["env"] | {"TEST_SERVICE_ACTIVE": "1"}
    result = subprocess.run(["bash", str(host["installer"])], env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert "先等待任务结束" in result.stderr
    assert "systemctl --user stop visioncortex-analysis.service" in result.stderr
    assert "previous configuration" in host["unit"].read_text()
    assert not (host["runtime"] / "Input-Manifests").exists()
    assert not (host["runtime"] / "tmp").exists()
    commands = host["log"].read_text()
    assert "is-active --quiet visioncortex-analysis.service" in commands
    assert "daemon-reload" not in commands and "enable --now" not in commands and "curl" not in commands
    assert not list(host["unit"].parent.glob(".visioncortex-analysis.service.*"))


def test_analysis_install_reuses_identical_active_service_without_replacing_file(relocated_analysis_host):
    host = relocated_analysis_host
    first = subprocess.run(["bash", str(host["installer"])], env=host["env"], capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    original = host["unit"].read_bytes()
    inode = host["unit"].stat().st_ino
    host["log"].unlink()
    env = host["env"] | {"TEST_SERVICE_ACTIVE": "1"}
    repeated = subprocess.run(["bash", str(host["installer"])], env=env, capture_output=True, text=True)
    assert repeated.returncode == 0, repeated.stderr
    assert host["unit"].read_bytes() == original
    assert host["unit"].stat().st_ino == inode
    commands = host["log"].read_text()
    assert "systemctl --user enable --now visioncortex-analysis.service" in commands
    assert "restart" not in commands and "systemctl --user stop" not in commands


def test_analysis_install_atomically_replaces_inactive_service(relocated_analysis_host):
    host = relocated_analysis_host
    host["unit"].parent.mkdir(parents=True)
    host["unit"].write_text("WorkingDirectory=" + str(host["project"]).replace("%", "%%") + "\nprevious configuration", encoding="utf-8")
    inode = host["unit"].stat().st_ino
    result = subprocess.run(["bash", str(host["installer"])], env=host["env"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert host["unit"].read_bytes() == host["verified"].read_bytes()
    assert host["unit"].stat().st_ino != inode
    assert "is-active --quiet visioncortex-analysis.service" in host["log"].read_text()
    assert not list(host["unit"].parent.glob(".visioncortex-analysis.service.*"))


def test_linux_archive_folder_uses_xdg_open(monkeypatch, tmp_path):
    monkeypatch.setattr("visioncortex.api.shutil.which", lambda name: "/usr/bin/xdg-open")

    command = _folder_open_command(tmp_path, os_name="posix", platform="linux")

    assert command == ["/usr/bin/xdg-open", str(tmp_path)]


def test_local_profile_cannot_inherit_production_nas_input_contract():
    from visioncortex.config import load_config

    settings = load_config(ROOT / "configs/rtx3090ti-ubuntu-local.yaml")
    storage = settings["storage"]
    assert storage["run_output_mode"] == "local"
    assert storage["manifest_storage"] == "local"
    assert storage["require_nas_source_paths"] is False
    assert storage["sync_to_nas"] is False
    for name in ("source_root", "snapshot_path"):
        value = settings["collection_ingest"][name]
        assert value is None or Path(value).is_relative_to(Path(storage["local_runtime_root"]))


@pytest.fixture
def relocated_web_host(tmp_path, request):
    if os.name == "nt":
        pytest.skip("systemd installation is Linux-only")
    import yaml

    project = tmp_path / 'customer clone % "quoted" $(touch injected)'
    deployment = project / "deployment/rtx3090ti-ubuntu"
    deployment.mkdir(parents=True)
    for name in ("_common.sh", "render_service.py", "05-Install-Local-Service.sh", "07-Install-LAN-Server.sh",
                 "06-Run-LAN-Server.sh", "00-Preflight.sh", "Open-VisionCortex.sh",
                 "visioncortex-local.service", "visioncortex-lan.service"):
        shutil.copy2(DEPLOYMENT / name, deployment / name)
    (project / "src").symlink_to(ROOT / "src", target_is_directory=True)
    (project / "configs").mkdir()
    shutil.copy2(ROOT / "configs/default.yaml", project / "configs/default.yaml")
    runtime = tmp_path / "owned runtime"
    config = project / "configs/customer.yaml"
    config.write_text(yaml.safe_dump({
        "project": {"run_purpose": "analysis", "output_root": str(runtime / "runs")},
        "runtime": {"local_only": True}, "mllm": {"enabled": False},
        "storage": {"local_runtime_root": str(runtime), "archive_root": str(runtime / "archives"),
                    "local_input_root": str(runtime / "input"), "local_cache_root": str(runtime / "cache"),
                    "local_staging_root": str(runtime / "runs"), "sync_to_nas": False},
    }), encoding="utf-8")
    # Mock only the service/health/probe boundaries; real configuration and unit
    # rendering are exercised without GPU, NAS or a running service manager.
    with (deployment / "_common.sh").open("a") as stream:
        stream.write('\ndeployment_health() { printf "health mocked\\n" >> "$TEST_COMMAND_LOG"; }\n')
    (deployment / "00-Preflight.sh").write_text(
        '#!/usr/bin/env bash\nprintf "preflight mocked %s\\n" "$*" >> "$TEST_COMMAND_LOG"\n')
    (deployment / "00-Preflight.sh").chmod(0o755)
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    log = tmp_path / "commands.log"
    for name, body in {
        "systemctl": 'printf "systemctl %s\\n" "$*" >> "$TEST_COMMAND_LOG"\n'
                     'if [[ $* == *"is-active --quiet"* ]]; then [[ ${TEST_SERVICE_ACTIVE:-0} == 1 ]];\n'
                     'elif [[ $* == *"--property MainPID --value"* ]]; then printf "4242\\n";\n'
                     'elif [[ $* == *"stop"* || $* == *"disable"* || $* == *"restart"* ]]; then exit 98; fi\n',
        "systemd-analyze": 'printf "verify %s\\n" "$*" >> "$TEST_COMMAND_LOG"\n'
                           '[[ ${TEST_REJECT_UNIT:-0} == 0 ]]\n',
        "nvidia-smi": 'printf "FORBIDDEN GPU\\n" >> "$TEST_COMMAND_LOG"; exit 98\n',
        "ffmpeg": 'printf "FORBIDDEN FFmpeg\\n" >> "$TEST_COMMAND_LOG"; exit 98\n',
        "df": 'printf "FORBIDDEN disk probe\\n" >> "$TEST_COMMAND_LOG"; exit 98\n',
    }.items():
        path = fake_bin / name
        path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + body)
        path.chmod(0o755)
    bus_temp = tempfile.TemporaryDirectory(prefix="vc-web-bus-")
    request.addfinalizer(bus_temp.cleanup)
    bus = socket.socket(socket.AF_UNIX)
    bus.bind(str(Path(bus_temp.name) / "bus"))
    bus.close()
    # Port allocation is a local socket bind only, with no listener or service.
    with socket.socket() as port_probe:
        port_probe.bind(("127.0.0.1", 0))
        port = str(port_probe.getsockname()[1])
    home = tmp_path / "customer home"
    home.mkdir()
    env = {name: value for name, value in os.environ.items() if not name.startswith("VISIONCORTEX_")}
    env.update(HOME=str(home), XDG_CONFIG_HOME=str(home / ".config"), XDG_DATA_HOME=str(home / ".local/share"),
               XDG_RUNTIME_DIR=bus_temp.name, PATH=f"{fake_bin}{os.pathsep}{env['PATH']}",
               VISIONCORTEX_PYTHON=sys.executable, VISIONCORTEX_CONFIG=str(config), VISIONCORTEX_WEB_PORT=port,
               TEST_COMMAND_LOG=str(log), PYTHONDONTWRITEBYTECODE="1")
    return {"project": project, "deployment": deployment, "config": config, "runtime": runtime,
            "env": env, "home": home, "log": log, "port": port,
            "unit": home / ".config/systemd/user/visioncortex-local.service"}


def test_local_service_customer_clone_is_rendered_without_models_credentials_or_nas(relocated_web_host):
    host = relocated_web_host
    result = subprocess.run(["bash", str(host["deployment"] / "05-Install-Local-Service.sh")],
                            env=host["env"], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    unit = host["unit"].read_text()
    assert "@PROJECT_ROOT@" not in unit and "@ENVIRONMENT@" not in unit
    assert 'VISIONCORTEX_DEPLOYMENT_MODE=local' in unit
    assert f'VISIONCORTEX_WEB_PORT={host["port"]}' in unit
    assert str(host["project"]).replace("%", "%%") in unit
    assert "FORBIDDEN" not in host["log"].read_text()
    assert "preflight mocked" not in host["log"].read_text()
    assert not (host["project"] / "injected").exists()
    desktop = host["home"] / ".local/share/applications/visioncortex-local.desktop"
    assert desktop.is_file()
    assert "VISIONCORTEX_CONFIG=" in desktop.read_text() and "configs/customer.yaml" in desktop.read_text()
    assert not (host["home"] / ".config/autostart").exists()
    assert "sudo loginctl" not in host["log"].read_text()
    analyzer = shutil.which("systemd-analyze")
    if analyzer:
        verified = subprocess.run([analyzer, "--system", "verify", str(host["unit"])], capture_output=True, text=True)
        assert verified.returncode == 0, verified.stderr


@pytest.mark.parametrize("state", ["other_checkout", "active_different", "symlink", "invalid_template"])
def test_local_install_preserves_existing_instance_before_mutation(relocated_web_host, state):
    host = relocated_web_host
    host["unit"].parent.mkdir(parents=True)
    owner = str(host["project"]).replace("%", "%%") if state == "active_different" else "/another/checkout"
    original = "WorkingDirectory=" + owner + "\nprevious service\n"
    if state == "symlink":
        other = host["unit"].with_name("other.service")
        other.write_text(original)
        host["unit"].symlink_to(other)
    else:
        host["unit"].write_text(original)
    env = host["env"] | ({"TEST_SERVICE_ACTIVE": "1"} if state == "active_different" else {})
    if state == "invalid_template":
        env["TEST_REJECT_UNIT"] = "1"
    result = subprocess.run(["bash", str(host["deployment"] / "05-Install-Local-Service.sh")],
                            env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode != 0
    assert host["unit"].read_text() == original
    assert not host["runtime"].exists()
    commands = host["log"].read_text()
    assert "daemon-reload" not in commands and "enable --now" not in commands
    assert "stop" not in commands and "disable" not in commands


def test_local_install_second_instance_does_not_change_default_service(relocated_web_host):
    host = relocated_web_host
    host["unit"].parent.mkdir(parents=True)
    host["unit"].write_text("another customer's service")
    result = subprocess.run(["bash", str(host["deployment"] / "05-Install-Local-Service.sh")],
                            env=host["env"] | {"VISIONCORTEX_SERVICE_NAME": "visioncortex-customer.service"},
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert host["unit"].read_text() == "another customer's service"
    assert host["unit"].with_name("visioncortex-customer.service").is_file()
    assert "visioncortex-local.service" not in host["log"].read_text()


def test_lan_install_uses_prepared_customer_site_and_keeps_local_service(relocated_web_host):
    import yaml

    host = relocated_web_host
    settings = yaml.safe_load(host["config"].read_text())
    settings["runtime"]["local_only"] = False
    host["config"].write_text(yaml.safe_dump(settings))
    password = host["home"] / "mock web password"
    password.write_text("test-only synthetic password\n")
    password.chmod(0o600)
    host["unit"].parent.mkdir(parents=True)
    host["unit"].write_text("unrelated local instance")
    env = host["env"] | {"VISIONCORTEX_WEB_PASSWORD_FILE": str(password), "VISIONCORTEX_WEB_USERNAME": "customer"}
    result = subprocess.run(["bash", str(host["deployment"] / "07-Install-LAN-Server.sh")],
                            env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    unit = host["unit"].with_name("visioncortex-lan.service").read_text()
    assert "test-only synthetic password" not in unit
    assert 'VISIONCORTEX_WEB_USERNAME=customer' in unit
    assert "ExecStartPre=/usr/bin/bash" in unit and "--production" in unit
    assert host["unit"].read_text() == "unrelated local instance"
    commands = host["log"].read_text()
    assert "stop" not in commands and "disable" not in commands and "FORBIDDEN" not in commands


@pytest.mark.parametrize("installer", ["07-Install-LAN-Server.sh", "09-Install-Analysis-Service.sh"])
def test_production_installer_requires_explicit_private_configuration(relocated_web_host, installer):
    host = relocated_web_host
    if installer.startswith("09"):
        shutil.copy2(DEPLOYMENT / installer, host["deployment"] / installer)
    env = {name: value for name, value in host["env"].items() if name != "VISIONCORTEX_CONFIG"}
    result = subprocess.run(["bash", str(host["deployment"] / installer)], env=env,
                            capture_output=True, text=True, timeout=15)
    assert result.returncode != 0 and "explicit prepared" in result.stderr
    assert not host["log"].exists() and not host["runtime"].exists()


def test_committed_production_template_cannot_be_used_as_prepared_site(relocated_web_host):
    host = relocated_web_host
    result = subprocess.run(["bash", str(host["deployment"] / "07-Install-LAN-Server.sh")],
                            env=host["env"] | {"VISIONCORTEX_CONFIG": str(ROOT / "configs/rtx3090ti-ubuntu-production.yaml")},
                            capture_output=True, text=True, timeout=15)
    assert result.returncode != 0
    assert not host["log"].exists() and not host["runtime"].exists()


def test_deployment_preflight_local_never_queries_gpu_nas_or_credentials(relocated_web_host):
    host = relocated_web_host
    shutil.copy2(DEPLOYMENT / "00-Preflight.sh", host["deployment"] / "00-Preflight.sh")
    result = subprocess.run(["bash", str(host["deployment"] / "00-Preflight.sh"), "--local"],
                            env=host["env"], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert "GPU/NAS/model execution not checked" in result.stdout
    assert not host["log"].exists()


@pytest.fixture
def deployment_helper():
    import importlib.util

    spec = importlib.util.spec_from_file_location("deployment_render", DEPLOYMENT / "render_service.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("failure", ["active_work", "unprovable", "wrong_pid", "wrong_digest", "wrong_product", "wrong_archive"])
def test_deployment_health_refuses_wrong_runtime_or_unsafe_shutdown(deployment_helper, monkeypatch, tmp_path, failure):
    from visioncortex.automation_readiness import configuration_digest
    from visioncortex.config import load_config

    project = ROOT
    config = ROOT / "configs/development-local.yaml"
    for name in list(os.environ):
        if name.startswith("VISIONCORTEX_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("VISIONCORTEX_PROJECT_ROOT", str(project))
    monkeypatch.setenv("VISIONCORTEX_CONFIG", str(config))
    monkeypatch.setenv("VISIONCORTEX_DEFAULT_CONFIG", str(ROOT / "configs/default.yaml"))
    settings = load_config(config)
    archive = settings["storage"]["archive_root"]
    health = {"status": "ok", "product_name": "VisionCortex", "archive_root": archive,
              "shutdown_safety": {"provable": True, "active_tasks": 0}}
    identity = {"schema_version": "visioncortex-automation-readiness/1", "pid": 4242,
                "configuration": {"config_path": str(config), "default_config_path": str(ROOT / "configs/default.yaml"),
                                  "settings_sha256": configuration_digest(settings)}}
    if failure == "active_work":
        health["shutdown_safety"]["active_tasks"] = 1
    elif failure == "unprovable":
        health["shutdown_safety"]["provable"] = False
    elif failure == "wrong_pid":
        identity["pid"] = 4243
    elif failure == "wrong_digest":
        identity["configuration"]["settings_sha256"] = "0" * 64
    elif failure == "wrong_product":
        health["product_name"] = "another service"
    else:
        health["archive_root"] = str(tmp_path / "another archive")
    class Response:
        def __init__(self, payload):
            self.payload = payload
        def __enter__(self):
            return self
        def __exit__(self, *_):
            return False
        def read(self, limit):
            assert limit == 65537
            return json.dumps(self.payload).encode()
    class Opener:
        def open(self, request, timeout):
            assert timeout == 2
            return Response(identity if request.full_url.endswith("/health/automation") else health)
    monkeypatch.setattr(deployment_helper.urllib.request, "build_opener", lambda *_: Opener())
    monkeypatch.setattr(sys, "argv", ["helper", "health-stop", "8002", archive, "4242"])
    with pytest.raises(ValueError):
        deployment_helper.main()


def test_rtp_receiver_template_renders_customer_paths_with_static_systemd_verification(tmp_path):
    project = tmp_path / 'customer clone % "quoted"'
    project.mkdir()
    config = project / "receiver config.json"
    config.write_text('{}\n')
    unit = tmp_path / "visioncortex-rtp-opus.service"
    result = subprocess.run([sys.executable, str(DEPLOYMENT / "render_service.py"), "render-rtp", str(project),
                             sys.executable, str(config), str(ROOT / "deployment/rtp-opus/visioncortex-rtp-opus.service"),
                             str(unit)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    text = unit.read_text()
    assert "@PROJECT_ROOT@" not in text and "@CONFIG@" not in text
    assert str(config).replace('"', '\\"').replace('%', '%%') in text
    analyzer = shutil.which("systemd-analyze")
    if analyzer:
        parsed = subprocess.run([analyzer, "--system", "verify", str(unit)], capture_output=True, text=True)
        assert parsed.returncode == 0, parsed.stderr
