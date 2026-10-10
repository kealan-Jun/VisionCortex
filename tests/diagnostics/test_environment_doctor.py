from __future__ import annotations

from repo_paths import ROOT

import json
import os
import csv
import shutil
import subprocess
import sys
import tomllib

import pytest
import yaml

from tools.diagnostics import doctor


REPOSITORY_ROOT = ROOT


def test_parse_nvidia_smi_and_choose_memory_profiles():
    output = "NVIDIA GeForce RTX 4060, 8188, 591.44\nNVIDIA RTX 3090 Ti, 24564, 591.44"

    assert doctor.parse_nvidia_smi(output) == [
        {"name": "NVIDIA GeForce RTX 4060", "memory_mb": 8188, "driver_version": "591.44"},
        {"name": "NVIDIA RTX 3090 Ti", "memory_mb": 24564, "driver_version": "591.44"},
    ]
    assert doctor.gpu_memory_profile(6144) == "gpu-6gb"
    assert doctor.gpu_memory_profile(8188) == "gpu-8gb"
    assert doctor.gpu_memory_profile(12288) == "gpu-12gb"
    assert doctor.gpu_memory_profile(24564) == "gpu-24gb"


def test_environment_doctor_does_not_probe_storage_by_default(monkeypatch):
    commands = []

    def command_output(command, **kwargs):
        commands.append(command)
        return None

    monkeypatch.setattr(doctor, "_command_output", command_output)
    monkeypatch.setattr(doctor, "_import_failures", lambda *_: [])

    report = doctor.inspect_environment(REPOSITORY_ROOT)

    assert report["storage"] == {"checked": False, "paths": {}}
    storage_check = next(item for item in report["checks"] if item["name"] == "NAS存储")
    assert storage_check["status"] == "skipped"
    assert report["evidence_boundary"]["model_inference"] == "NOT_PROVEN"
    assert report["evidence_boundary"]["real_video"] == "NOT_PROVEN"
    assert report["evidence_boundary"]["web_start"] == "NOT_PROVEN"
    assert report["gpu_checked"] is False
    assert not any(command[0] == "nvidia-smi" for command in commands)
    assert next(item for item in report["checks"] if item["name"] == "GPU")["status"] == "skipped"


def test_environment_doctor_core_coverage_matches_declared_dependencies():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    declared = {item.split(">", 1)[0].split("[", 1)[0] for item in project["dependencies"]}
    assert declared == set(doctor.CORE_MODULES.values()) - {"visioncortex"}


def test_missing_core_dependency_blocks_preflight(monkeypatch):
    original = doctor.importlib.util.find_spec
    monkeypatch.setattr(doctor.importlib.util, "find_spec", lambda module:
                        None if module == "ijson" else original(module))
    monkeypatch.setattr(doctor, "_command_output", lambda *_args, **_kwargs: None)
    report = doctor.inspect_environment(ROOT)
    assert report["launch_ready"] is False
    assert report["evidence_boundary"]["dependency_preflight"] == "NOT_PROVEN"
    assert any("ijson" in item["message"] for item in report["checks"])


def test_import_failure_blocks_preflight_without_claiming_browser_evidence(monkeypatch):
    monkeypatch.setattr(doctor, "_command_output", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(doctor, "_import_failures", lambda *_: ["cv2:ImportError"])
    report = doctor.inspect_environment(ROOT)
    assert report["launch_ready"] is False
    assert report["evidence_boundary"]["web_start"] == "NOT_PROVEN"
    assert any("OpenCV" in message for message in report["remediation"])


def test_import_preflight_rejects_a_different_checkout(tmp_path):
    failures = doctor._import_failures(tmp_path, ["visioncortex"])
    assert failures == ["visioncortex:DifferentCheckout"]


def test_web_and_cli_import_without_optional_model_stack():
    assert doctor._import_failures(ROOT, [*doctor.CORE_MODULES, "visioncortex.cli", "visioncortex.api"]) == []


def test_gpu_probe_requires_explicit_option(monkeypatch):
    def command_output(command, **kwargs):
        return "NVIDIA GeForce RTX 4060, 8188, 591.44" if command[0] == "nvidia-smi" else None

    monkeypatch.setattr(doctor, "_command_output", command_output)
    monkeypatch.setattr(doctor, "_import_failures", lambda *_: [])
    report = doctor.inspect_environment(ROOT, check_gpu=True)
    assert report["gpu_checked"] is True
    assert report["gpus"][0]["memory_mb"] == 8188
    assert report["evidence_boundary"]["model_inference"] == "NOT_PROVEN"


def test_local_collection_index_is_empty_and_has_the_contract_columns():
    with (ROOT / "examples/development-index.csv").open(newline="") as handle:
        rows = csv.DictReader(handle)
        assert {"experiment_id", "camera_key", "camera_view", "rgb_file", "frames_file"} <= set(rows.fieldnames)
        assert list(rows) == []


def test_default_launcher_profile_is_local_only():
    profile = yaml.safe_load(
        (REPOSITORY_ROOT / "configs" / "development-local.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert profile["storage"]["sync_to_nas"] is False
    assert profile["storage"]["web_upload_retention_mode"] == "local_only"
    assert profile["collection_ingest"]["enabled"] is False
    assert profile["mllm"]["enabled"] is False


def test_environment_doctor_json_cli_is_machine_readable():
    environment = dict(os.environ)
    environment["PYTHONIOENCODING"] = "utf-8"
    completed = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "tools" / "doctor.py"),
            "--project-root",
            str(REPOSITORY_ROOT),
            "--json",
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=environment,
        check=False,
    )

    payload = json.loads(completed.stdout)
    assert completed.returncode == (0 if payload["launch_ready"] else 1)
    assert payload["schema_version"] == doctor.SCHEMA_VERSION
    assert payload["project_root"] == str(REPOSITORY_ROOT)
    assert payload["storage"]["checked"] is False
    assert payload["gpu_checked"] is False
    assert payload["evidence_boundary"]["web_start"] == "NOT_PROVEN"


@pytest.mark.skipif(sys.platform == "win32", reason="Unix environment selection uses Bash.")
@pytest.mark.parametrize("selected", ["active", "explicit", "repository", "conda"])
def test_unix_launcher_uses_the_requested_environment_before_repo_fallback(tmp_path, selected):
    project = tmp_path / "Project"
    project.mkdir()
    (project / "configs").mkdir()
    (project / "configs/development-local.yaml").write_text("{}\n")
    launcher = project / "start-visioncortex.sh"
    shutil.copy2(ROOT / launcher.name, launcher)
    marker = tmp_path / "selected.txt"
    interpreters = {}
    for name in ("active", "explicit", "repository", "conda"):
        root = project / ".venv" if name == "repository" else tmp_path / name
        executable = root / "bin/python"
        executable.parent.mkdir(parents=True)
        executable.write_text(
            f'#!/usr/bin/env bash\nprintf "%s\\n" "{name}" > "$VC_TEST_INTERPRETER_MARKER"\n'
            'if [[ "$1" == "-" ]]; then\n'
            '  printf "%s\\n" "$PWD/Runtime" "$PWD/Runtime/runs" "$PWD/Runtime/cache" '
            '"$PWD/Runtime/input" "$PWD/Runtime/archives"\nfi\n'
        )
        executable.chmod(0o700)
        interpreters[name] = executable
    environment = dict(os.environ, VC_TEST_INTERPRETER_MARKER=str(marker))
    environment.pop("VIRTUAL_ENV", None)
    environment.pop("CONDA_PREFIX", None)
    if selected in {"active", "explicit"}:
        environment["VIRTUAL_ENV"] = str(interpreters["active"].parents[1])
    if selected == "conda":
        environment["CONDA_PREFIX"] = str(interpreters["conda"].parents[1])
    arguments = ["--check-only"]
    if selected == "explicit":
        arguments += ["--python", str(interpreters["explicit"])]
    subprocess.run(["bash", str(launcher), *arguments], env=environment,
                   check=True, capture_output=True, timeout=10)
    assert marker.read_text().strip() == selected


@pytest.mark.skipif(sys.platform == "win32", reason="Bash syntax check runs on Unix CI.")
def test_unix_launchers_have_valid_bash_syntax():
    for relative_path in ("start-visioncortex.sh", "Start-VisionCortex.command"):
        subprocess.run(
            ["bash", "-n", str(REPOSITORY_ROOT / relative_path)],
            check=True,
            capture_output=True,
            text=True,
        )


@pytest.mark.skipif(sys.platform != "win32", reason="PowerShell syntax check runs on Windows CI.")
def test_windows_launcher_has_valid_powershell_syntax():
    script = REPOSITORY_ROOT / "Start-VisionCortex.ps1"
    command = f"[void][scriptblock]::Create((Get-Content -LiteralPath '{script}' -Raw))"
    subprocess.run(
        ["powershell.exe", "-NoLogo", "-NoProfile", "-Command", command],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
