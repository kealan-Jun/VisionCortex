from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from labvision_evidence.runtime_preflight import (
    REQUIRED_RUNTIME_MODULES,
    inspect_project_runtime,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"


def test_runtime_preflight_imports_full_project_runtime_from_expected_source():
    result = inspect_project_runtime(SOURCE_ROOT)

    assert result["status"] == "passed"
    assert Path(str(result["package_source"])).is_relative_to(SOURCE_ROOT)
    assert set(result["dependency_versions"]) == set(REQUIRED_RUNTIME_MODULES)
    assert result["cli_imported"] is True
    assert result["quality_replay_imported"] is True
    assert result["video_operations"] == 0
    assert result["model_calls"] == 0
    assert result["token_usage"] == 0


def test_runtime_preflight_module_executes_without_inline_python_source():
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(SOURCE_ROOT)

    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "labvision_evidence.runtime_preflight",
            "--expected-source",
            str(SOURCE_ROOT),
        ],
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "passed"
    assert Path(payload["package_source"]).is_relative_to(SOURCE_ROOT)
    assert payload["python_executable"] == sys.executable
