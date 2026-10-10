"""Application ownership and publication order, using fake work only."""

from contextlib import contextmanager
from types import SimpleNamespace
import ast
import asyncio

import pytest

from repo_paths import ROOT
from visioncortex import api
from visioncortex.application import runtime_host
from visioncortex.application.collection_workflow import (
    CollectionWorkflow,
    CollectionPolicy,
)
from visioncortex.application.dependencies import ports


def test_application_services_do_not_depend_on_transport_or_commands():
    for source in (ROOT / "src/visioncortex/application").glob("*.py"):
        for node in ast.walk(ast.parse(source.read_text())):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                assert module not in {"api", "cli", "machine_api", "web_api"}, (
                    source.name
                )
                assert not module.startswith(("fastapi", "web_api.")), source.name


def test_legacy_dependency_patch_has_one_port_owner(monkeypatch):
    target = object()
    original = ports.EvidencePipeline
    monkeypatch.setattr(api, "EvidencePipeline", target)
    assert ports.EvidencePipeline is target
    assert api.EvidencePipeline is target
    assert api._COMPATIBILITY_EXPORTS["EvidencePipeline"] == (ports, "EvidencePipeline")
    monkeypatch.undo()
    assert ports.EvidencePipeline is original


def workflow(tmp_path, *, partial=False, reject=False):
    events = []
    path = tmp_path / "input_manifest.yaml"
    manifest = SimpleNamespace(model_dump=lambda **_: {"experiment_id": "owned"})

    def prepare(settings, experiment_id, progress):
        events.append("prepare")
        return manifest, path, {"copied_source_bytes": 0}

    def analyze(value):
        assert path.is_file()
        events.append("analyze")
        return tmp_path

    def acceptance(output):
        events.append("acceptance")
        if reject:
            raise ValueError("required audit failed")
        return {"audit": "verified"}

    work = CollectionWorkflow(
        prepare=prepare,
        analyze=analyze,
        is_partial=lambda _: partial,
        promote=lambda *_: events.append("promote") or {"verification": "verified"},
    )
    policy = CollectionPolicy(require_cold_execution=True, before_publish=acceptance)
    arguments = dict(
        settings={"project": {"cache_mode": "cold"}},
        experiment_id="owned",
        staging_root=tmp_path,
        fixed_root=tmp_path / "fixed",
        history_root=tmp_path / "history",
        policy=policy,
        on_partial=lambda _: events.append("partial"),
    )
    return work, arguments, events


def test_partial_collection_never_runs_formal_acceptance_or_promotion(tmp_path):
    work, arguments, events = workflow(tmp_path, partial=True)
    result = work.execute(**arguments)
    assert result.partial and result.promotion is None
    assert events == ["prepare", "analyze", "partial"]


def test_required_collection_audit_blocks_promotion(tmp_path):
    work, arguments, events = workflow(tmp_path, reject=True)
    with pytest.raises(ValueError, match="required audit failed"):
        work.execute(**arguments)
    assert events == ["prepare", "analyze", "acceptance"]


def test_cold_collection_requirement_rejects_before_input_read(tmp_path):
    work, arguments, events = workflow(tmp_path)
    arguments["settings"] = {"project": {"cache_mode": "reuse"}}
    with pytest.raises(ValueError, match="cold"):
        work.execute(**arguments)
    assert events == []


def test_runtime_host_lifecycle_uses_ordered_shared_producers(monkeypatch):
    events = []
    settings = {"runtime": {"role": "combined"}}
    host = runtime_host.RuntimeHost()
    monkeypatch.setattr(host, "settings", lambda: settings)
    monkeypatch.setattr(host, "maintenance_active", lambda: False)
    monkeypatch.setattr(host, "initialize_queue", lambda _: events.append("queue"))
    monkeypatch.setattr(
        runtime_host, "_automation_configuration_identity", lambda _: {}
    )
    from visioncortex import runtime_process, owned_subprocess, shared_inference
    from visioncortex.application import upload_service

    @contextmanager
    def owner(_):
        events.append("owner")
        try:
            yield
        finally:
            events.append("release")

    @contextmanager
    def guard():
        events.append("decoder_guard")
        yield

    monkeypatch.setattr(runtime_process, "role", lambda _: "combined")
    monkeypatch.setattr(runtime_process, "worker_owner", owner)
    monkeypatch.setattr(owned_subprocess, "decoder_shutdown_guard", guard)
    monkeypatch.setattr(
        shared_inference, "close_pools", lambda: events.append("pools_stop")
    )
    monkeypatch.setattr(
        upload_service,
        "_expire_stale_upload_sessions",
        lambda _: events.append("expire"),
    )
    for method in (
        "_recover_jobs_from_archive_receipts",
        "_recover_orphaned_tasks",
        "_start_queue_worker",
        "_start_nas_monitor",
        "_stop_nas_monitor",
        "_stop_queue_worker",
    ):
        monkeypatch.setattr(
            runtime_host, method, lambda *_, step=method: events.append(step)
        )
    monkeypatch.setattr(
        runtime_host,
        "_device_day_service",
        SimpleNamespace(
            start=lambda: events.append("device_start"),
            stop=lambda: events.append("device_stop"),
        ),
    )

    async def run():
        async with host.lifespan():
            events.append("active")

    asyncio.run(run())
    assert events == [
        "queue",
        "owner",
        "expire",
        "_recover_jobs_from_archive_receipts",
        "_recover_orphaned_tasks",
        "_start_queue_worker",
        "_start_nas_monitor",
        "device_start",
        "active",
        "decoder_guard",
        "device_stop",
        "_stop_nas_monitor",
        "_stop_queue_worker",
        "pools_stop",
        "release",
    ]
