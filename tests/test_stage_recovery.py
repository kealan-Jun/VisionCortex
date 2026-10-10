"""Fault injection contracts; no model, NAS or real-video accuracy claims."""
import json
import time
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from types import MethodType

import pytest

from visioncortex.archive import ArchiveLayout, write_json
from visioncortex.config import load_config
from visioncortex.pipeline import EvidencePipeline
from visioncortex.schemas import RunManifest, ViewInput, VideoInfo
from visioncortex.stage_recovery import DEPENDENCIES, StageRunner, decode, encode, index


def fixture(tmp_path, failed=(), *, resume=False, identity="source-v1"):
    config = load_config(Path("configs/development-local.yaml"))
    config["storage"].update(local_cache_root=str(tmp_path / "cache"), local_runtime_root=str(tmp_path / "runtime"))
    config["project"]["resume_stages"] = resume
    layout = ArchiveLayout(tmp_path / "output")
    layout.create()
    pipeline = EvidencePipeline(config)
    pipeline._run_started_perf = time.perf_counter()
    pipeline._run_started_iso = datetime.now(timezone.utc).isoformat()
    pipeline._active_layout = layout
    manifest = RunManifest(experiment_id="test", views=[ViewInput(view_id=name, role=role, video=tmp_path / (name+".mp4")) for name, role in (("fp", "first_person"), ("tp", "third_person"))])
    calls = Counter()
    for stage in DEPENDENCIES:
        def callback(self, context, received_layout, received_manifest, name=stage):
            calls[name] += 1
            if name in failed:
                raise RuntimeError("injected " + name)
            if name == "preflight":
                context.infos = {"fp": VideoInfo(path=tmp_path / "source.mp4", duration_ms=1000, fps=1, width=16, height=16, frame_count=1)}
                context.disk_report = {}
            path = layout.json_config / f"test-{name}.json"
            write_json(path, {"stage": name, "fixture": True})
            self._complete_stage(layout, name, [path])
        setattr(pipeline, "_stage_" + stage, MethodType(callback, pipeline))
    runner = StageRunner(pipeline, layout, manifest, {"inputs": [identity], "models": {}})
    pipeline._stage_runner = runner
    return runner, calls, layout


def descendants(stage):
    blocked = {stage}
    for name, deps in DEPENDENCIES.items():
        if any(d in blocked for d in deps):
            blocked.add(name)
    return blocked


@pytest.mark.parametrize("failed", list(DEPENDENCIES))
def test_only_required_dependants_stop(tmp_path, failed):
    runner, calls, layout = fixture(tmp_path, {failed})
    if failed in {"speech", "capture_quality"}:
        runner.run()
        assert calls["finalizing"] == 1
    else:
        with pytest.raises(RuntimeError, match="injected " + failed):
            runner.run()
        for name in descendants(failed) - {failed}:
            assert calls[name] == 0
            assert runner.outcomes[name] == "blocked"
    for name in set(DEPENDENCIES) - descendants(failed):
        assert calls[name] == 1
    receipt = json.loads((layout.json_config / "Stage-Receipts" / f"{failed}.json").read_text())
    assert receipt["status"] == "failed"


def test_resume_sam_failure_does_not_repeat_scans_clips_or_cloud(tmp_path):
    first, _, layout = fixture(tmp_path, {"material_refinement"})
    with pytest.raises(RuntimeError):
        first.run()
    video = layout.experiment_clips / "retained.mp4"
    video.write_bytes(b"fixture-not-video")
    before = video.stat()
    second, calls, _ = fixture(tmp_path, resume=True)
    second.run()
    assert calls["preflight"] == 1  # Always revalidate the environment.
    for name in ("alignment", "motion_probe", "candidate_coarse", "candidate_fine", "candidate_audit", "experiment_clips", "key_materials", "mllm"):
        assert calls[name] == 0, name
        assert second.pipeline._stage_outcomes[name]["reused"]
    assert calls["material_refinement"] == calls["semantic_refinement"] == 1
    assert video.stat().st_mtime_ns == before.st_mtime_ns
    assert video.read_bytes() == b"fixture-not-video"
    assert len(list((layout.json_config / "Recovery").glob("*.json"))) > len(DEPENDENCIES)


@pytest.mark.parametrize("changed", ["source", "checkpoint", "snapshot", "media", "model"])
def test_changed_inputs_or_outputs_cannot_reuse_unverified_stage(tmp_path, changed):
    first, _, layout = fixture(tmp_path, {"material_refinement"})
    with pytest.raises(RuntimeError):
        first.run()
    stage = "candidate_fine"
    if changed == "checkpoint":
        (layout.root / index(layout.root)["stages"][stage]["path"]).write_text("{}")
    if changed in {"snapshot", "media"}:
        item = index(layout.root)["stages"][stage]
        checkpoint = json.loads((layout.root / item["path"]).read_text())
        version_path = layout.root / checkpoint["receipt"]["version_manifest"]
        version = json.loads(version_path.read_text())
        if changed == "snapshot":
            (layout.root / version["files"][0]["snapshot"]).write_text("tampered")
        else:
            version["files"].append({"path": "missing.mp4", "size_bytes": 20, "mtime_ns": 1})
            write_json(version_path, version)
    second, calls, _ = fixture(tmp_path, resume=True, identity="changed" if changed == "source" else "source-v1")
    if changed == "model":
        second.identity["models"] = {"sha256": "changed"}
    second.run()
    assert calls[stage] == calls["candidate_audit"] == calls["mllm"] == 1


def test_legacy_result_without_recovery_points_is_not_inferred_complete(tmp_path):
    runner, calls, layout = fixture(tmp_path, resume=True)
    write_json(layout.json_config / "Stage-Receipts/mllm.json", {"stage": "mllm", "status": "completed"})
    runner.run()
    assert all(calls[name] == 1 for name in DEPENDENCIES)


def test_quality_attention_retries_archive_after_partial_report_is_written(tmp_path, monkeypatch):
    runner, _, layout = fixture(tmp_path)
    finished = []
    original_package = runner.pipeline._stage_package

    def package(context, current_layout, manifest):
        original_package(context, current_layout, manifest)
        write_json(layout.json_config / "quality_acceptance.json", {"passed": False})
        context.quality_attention = True

    def report(*_):
        finished.append("report_written")
        return layout.root

    def flush():
        finished.append("archive_retry")
        return True

    monkeypatch.setattr(runner.pipeline, "_stage_package", package)
    monkeypatch.setattr(runner.pipeline, "_finish_quality_attention", report)
    monkeypatch.setattr(runner, "flush_archive", flush)
    assert runner.run() == layout.root
    assert finished[-2:] == ["report_written", "archive_retry"]


def test_failed_stage_retries_archive_after_saving_partial_outputs(tmp_path, monkeypatch):
    runner, _, layout = fixture(tmp_path, {"material_refinement"})
    reports_at_flush = []

    def flush():
        reports_at_flush.append((layout.root / "Partial-Results/Analysis-Result.json").is_file())
        return True

    monkeypatch.setattr(runner, "flush_archive", flush)
    with pytest.raises(RuntimeError, match="material_refinement"):
        runner.run()
    assert reports_at_flush[-1] is True


def test_data_codec_preserves_event_aliases_and_rejects_executable_types(tmp_path):
    value = {"some": [1, 2]}
    shared = {"a": value, "b": value, "path": tmp_path, "set": {"a", "b"}}
    result = decode(encode(deepcopy(shared)))
    assert result["a"] is result["b"]
    assert result["path"] == tmp_path
    with pytest.raises(ValueError, match="Unsupported"):
        encode(lambda: None)


def test_nas_delivery_failure_keeps_local_files_and_retries_without_computation(tmp_path, monkeypatch):
    from visioncortex.archive_delivery import DeferredArchivePublisher
    from visioncortex.storage import IncrementalArchivePublisher
    local, nas = tmp_path / "local", tmp_path / "nas"
    local.mkdir()
    source = local / "result.txt"
    source.write_text("阶段结果", encoding="utf-8")
    publisher = DeferredArchivePublisher(local, nas)
    original = IncrementalArchivePublisher.publish_file
    monkeypatch.setattr(IncrementalArchivePublisher, "publish_file", lambda *_: (_ for _ in ()).throw(OSError("disconnected")))
    publisher.publish_file(source)
    assert source.read_text(encoding="utf-8") == "阶段结果"
    assert publisher.pending
    assert not (nas / "result.txt").exists()
    monkeypatch.setattr(IncrementalArchivePublisher, "publish_file", original)
    recovered = DeferredArchivePublisher(local, nas)
    assert recovered.flush()
    assert (nas / "result.txt").read_bytes() == source.read_bytes()
    assert not recovered.pending


def test_extracted_production_stages_run_with_synthetic_detection_ledgers(tmp_path, monkeypatch):
    """Exercise actual stage bodies, replacing GPU/video/model I/O only."""
    from visioncortex import pipeline as module
    from visioncortex.schemas import AlignmentTransform, FrameEvidence
    from test_pipeline_preflight import _NoopResourceMonitor
    config = load_config(Path("configs/development-local.yaml"))
    config["project"].update(output_root=str(tmp_path / "results"), preprocessing_acceptance_only=True)
    config["storage"].update(local_cache_root=str(tmp_path / "cache"), local_runtime_root=str(tmp_path / "runtime"))
    config["capture_quality"]["enabled"] = False
    config["speech_recognition"]["enabled"] = False
    config["performance"].update(media_pipeline_preflight_enabled=False, motion_probe_sparse_strategy="indexed_seek",
        fine_progressive_cross_view=False, coarse_frame_index_enabled=True, coarse_coverage_gate_enabled=False,
        fine_frame_index_enabled=True, fine_coverage_gate_enabled=False, coarse_open_vocabulary_recall_enabled=False,
        fine_roi_open_vocabulary_recall_enabled=False, automatic_short_timeline_exhaustive=False,
        coarse_full_timeline_scan=True, coarse_shared_motion_probe_enabled=True)
    views = []
    for name, role in (("fp", "first_person"), ("tp", "third_person")):
        source = tmp_path / (name + ".mp4")
        source.write_bytes(b"synthetic-input-identity-only")
        views.append(ViewInput(view_id=name, role=role, video=source))
    manifest = RunManifest(experiment_id="synthetic-stage-contract", views=views)
    infos = {v.view_id: VideoInfo(path=v.video, duration_ms=60000, fps=1, width=16, height=16, frame_count=60) for v in views}
    transforms = {v.view_id: AlignmentTransform(view_id=v.view_id, reference_view_id="fp") for v in views}
    monkeypatch.setattr(module, "ResourceMonitor", _NoopResourceMonitor)
    monkeypatch.setattr(module, "validate_models", lambda _: {})
    monkeypatch.setattr(module, "video_encoder_preflight", lambda _: {"selected_encoder": "libx264"})
    monkeypatch.setattr(module, "probe_views", lambda *_a, **_k: infos)
    monkeypatch.setattr(module, "build_alignments", lambda *_a: (transforms, []))
    monkeypatch.setattr(module, "alignment_quality_report", lambda *_a: {"formal_evidence_ready": True})
    monkeypatch.setattr(module, "write_aligned_csv", lambda path, *_a: path.write_text("time\n"))
    monkeypatch.setattr(EvidencePipeline, "_run_boundary_precheck", lambda *_a, **_k: {"passed": True})
    scanned = []
    def scan(self, received_manifest, received_infos, received_transforms, work, **options):
        scanned.append(options["phase"])
        work.mkdir(parents=True, exist_ok=True)
        paths = {}
        for view in received_manifest.views:
            path = work / (view.view_id + ".jsonl")
            frames = [FrameEvidence(view_id=view.view_id, role=view.role, frame_index=i, local_ms=i*1000,
                      global_ms=i*1000, width=16, height=16, motion_score=0.9, raw_motion_score=0.9) for i in range(60)]
            path.write_text("\n".join(f.model_dump_json() for f in frames)+"\n", encoding="utf-8")
            paths[view.view_id] = path
        return paths
    monkeypatch.setattr(EvidencePipeline, "_scan_all_views_concurrently", scan)
    def runtime(self, layout, work, phase, **_kwargs):
        write_json(layout.json_config / f"scan_runtime_{phase}.json", {"synthetic": True})
    monkeypatch.setattr(EvidencePipeline, "_archive_scan_runtime", runtime)
    # The real precheck writes this file; keep the fixture's side effect too.
    def precheck(self, layout, *_a, **_k):
        write_json(layout.json_config / "boundary_precheck.json", {"passed": True, "synthetic": True})
        return {"passed": True}
    monkeypatch.setattr(EvidencePipeline, "_run_boundary_precheck", precheck)
    output = EvidencePipeline(config).run(manifest)
    receipt = json.loads((output / "JSON-Config-Files/Stage-Receipts/candidate_audit.json").read_text())
    assert receipt["status"] == "completed"
    assert "candidate_fine" in index(output)["stages"]
    assert scanned
    assert not (output / "release.json").exists()


def test_local_work_ledger_is_bound_before_reuse(tmp_path):
    first, _, layout = fixture(tmp_path, {"material_refinement"})
    ledger = layout.work / "detections.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text('{"fixture":1}\n')
    original = first.pipeline._stage_candidate_fine
    def fine(self, context, received_layout, manifest):
        original(context, received_layout, manifest)
        context.detection_paths = {"fp": ledger}
    first.pipeline._stage_candidate_fine = MethodType(fine, first.pipeline)
    with pytest.raises(RuntimeError):
        first.run()
    ledger.write_text('{"fixture":2}\n')
    second, calls, _ = fixture(tmp_path, resume=True)
    second.run()
    assert calls["candidate_coarse"] == 0
    assert calls["candidate_fine"] == 1


def test_recovered_call_identity_survives_deterministic_relabel_and_counts_failure(tmp_path):
    from visioncortex.stage_recovery import call_identity
    runner, _, _ = fixture(tmp_path)
    call = {"provider": "fixture", "model": "fixture", "request_id": "request-a", "stage": "experiment_group_understanding",
            "cache_reused": False, "usage": {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10}}
    assert call_identity(call) == call_identity(dict(call, stage="experiment_group_understanding_pre_curation"))
    runner.pipeline._retained_failed_calls = [call]
    metrics = runner.pipeline._metrics([], [])
    assert metrics["tokens"]["run_total"]["total_tokens"] == 10
    assert metrics["mllm_calls"][0]["retained_from_failed_stage"] is True
