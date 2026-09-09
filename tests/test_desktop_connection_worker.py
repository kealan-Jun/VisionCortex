"""Exercise the desktop worker protocol without cloud calls or model startup."""
from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def worker(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "tools"))
    spec = importlib.util.spec_from_file_location("connection_worker_test", ROOT / "tools/verify_mllm_connection.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps({"connection": {"provider": "aliyun"}, "api_key": "synthetic-worker-key"})))
    return module


def records(capsys):
    return [json.loads(line.split(" ", 1)[1]) for line in capsys.readouterr().out.splitlines()]


def test_slow_package_reports_progress_before_loading_and_calling(worker, monkeypatch, tmp_path, capsys):
    clock = [0.0]
    monkeypatch.setattr(worker.time, "monotonic", lambda: clock[0])
    stages = []

    def verify(root, *, progress, use_cache):
        assert root == tmp_path
        assert use_cache is True
        stages.append("package")
        progress({"checked_bytes": 0, "total_bytes": 100, "checked_files": 0, "total_files": 1})
        clock[0] = 600.0
        progress({"checked_bytes": 100, "total_bytes": 100, "checked_files": 1, "total_files": 1})

    def request(connection, key, destination):
        assert stages == ["package"]
        assert key == "synthetic-worker-key"
        assert destination == tmp_path / "Runtime/Temp"
        stages.append("request")
        clock[0] += 175
        return {"status": "verified", "model_invocation": "PROVEN", "connection": connection}

    monkeypatch.setattr(worker, "verify_package", verify)
    fake = ModuleType("visioncortex.provider_connection")
    fake.verify_connection = request
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    assert worker.main(tmp_path) == 0
    output = records(capsys)
    assert [row["stage"] for row in output[:-1]] == ["package", "package", "package", "loading", "request"]
    assert output[-1]["phase_seconds"] == {"package": 600, "loading": 0, "request": 175}
    assert output[-1]["verification_elapsed_seconds"] == 775
    assert "synthetic-worker-key" not in json.dumps(output)


def test_corrupt_package_never_imports_or_calls_the_provider(worker, monkeypatch, tmp_path, capsys):
    def corrupt(*args, **kwargs):
        raise RuntimeError("bad local file synthetic-worker-key")

    monkeypatch.setattr(worker, "verify_package", corrupt)
    monkeypatch.setitem(sys.modules, "visioncortex.provider_connection", None)
    assert worker.main(tmp_path) == 2
    receipt = records(capsys)[-1]
    assert receipt["verification_stage"] == "package"
    assert receipt["model_invocation"] == "NOT_PROVEN"
    assert "尚未调用 AI" in receipt["message"]
    assert "[REDACTED]" in receipt["detail"]
    assert "loading" not in receipt["phase_seconds"]


def test_dependency_failure_is_distinct_from_cloud_failure(worker, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(worker, "verify_package", lambda *args, **kwargs: {})
    monkeypatch.setitem(sys.modules, "visioncortex.provider_connection", None)
    assert worker.main(tmp_path) == 2
    receipt = records(capsys)[-1]
    assert receipt["verification_stage"] == "loading"
    assert "组件加载失败" in receipt["message"]
    assert "request" not in receipt["phase_seconds"]


def test_cloud_failure_is_redacted_and_has_request_timing(worker, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(worker, "verify_package", lambda *args, **kwargs: {})
    fake = ModuleType("visioncortex.provider_connection")

    def failure(*args):
        raise TimeoutError("synthetic-worker-key")

    fake.verify_connection = failure
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    assert worker.main(tmp_path) == 2
    receipt = records(capsys)[-1]
    assert receipt["verification_stage"] == "request"
    assert receipt["detail"] == "[REDACTED]"
    assert "request" in receipt["phase_seconds"]
