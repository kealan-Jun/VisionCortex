import hashlib
import json
from pathlib import Path

import pytest

from fastapi.testclient import TestClient

from visioncortex import api
from visioncortex.archive import ArchiveLayout, _raise_for_incomplete_semantic_results, write_json
from visioncortex.partial_delivery import register_partial_result, write_partial_delivery
from visioncortex.run_queue import DurableRunQueue
from visioncortex.storage import fixed_archive_staging_paths


def test_partial_report_preserves_unknown_usage_and_does_not_read_source_media(tmp_path, monkeypatch):
    layout = ArchiveLayout(tmp_path / "archive")
    layout.create()
    write_json(layout.json_config / "pipeline_status.json", {
        "stage": "failed", "failed_stage": "package",
        "completed_stages": ["key_materials", "semantic_refinement"],
    })
    _raise_for_incomplete_semantic_results(
        layout, stage="key_material", results=[("EVT-1", {"status": "failed"})],
        fail_run=False,
    )
    original = layout.root / "Original-Experiment-Videos" / "source.mp4"
    original.parent.mkdir(exist_ok=True)
    original.write_bytes(b"source is never read")
    original_read = Path.read_bytes

    def bounded_read(path):
        assert path.suffix != ".mp4"
        return original_read(path)

    monkeypatch.setattr(Path, "read_bytes", bounded_read)
    metrics = {"tokens": {"run_total": {"total_tokens": None}}}
    write_json(layout.json_config / "run_metrics.json", metrics)
    report = write_partial_delivery(layout.root, metrics)

    assert report["status"] == "awaiting_semantic_recovery"
    assert report["formal_archive_promotion_allowed"] is False
    assert report["run_metrics"]["tokens"]["run_total"]["total_tokens"] is None
    assert report["original_media_read_bytes"] == 0
    assert report["pending_semantic_results"][0]["subject_id"] == "EVT-1"
    for reference in report["artifact_references"]:
        assert hashlib.sha256((layout.root / reference["path"]).read_bytes()).hexdigest() == reference["sha256"]
    assert "阶段成果" in (layout.root / report["report"]).read_text()
    assert report["evidence_classification"] == "PARTIAL_EVIDENCE"
    assert not (layout.json_config / "quality_acceptance.json").exists()
    assert not (layout.root / ".VisionCortex-Current-Release.json").exists()


def test_semantic_recovery_clears_previous_failure_receipt(tmp_path):
    layout = ArchiveLayout(tmp_path)
    _raise_for_incomplete_semantic_results(
        layout, stage="key_material", results=[("EVT-1", {"status": "failed"})],
        fail_run=False,
    )
    _raise_for_incomplete_semantic_results(
        layout, stage="key_material", results=[("EVT-1", {"status": "completed"})],
        fail_run=False,
    )
    report = json.loads((layout.json_config / "key_material_semantic_failures.json").read_text())
    assert report["incomplete"] == []
    assert report["status"] == "completed"


def test_material_failures_have_separate_retry_scope_in_partial_report(tmp_path):
    failure = {"event_id": "EVT-1", "view_id": "fp", "artifact_kind": "key_frame", "status": "failed"}
    write_json(tmp_path / "JSON-Config-Files/key_material_materialization_runtime.json", {
        "status": "partial", "incomplete_artifacts": [failure], "retry_event_ids": ["EVT-1"],
    })
    report = write_partial_delivery(tmp_path, {})
    assert report["pending_material_outputs"] == [failure]
    assert report["material_retry_event_ids"] == ["EVT-1"]
    assert report["pending_semantic_results"] == []
    assert report["formal_archive_promotion_allowed"] is False
    assert "待补全素材：1" in (tmp_path / report["report"]).read_text()
    assert any(item["path"].endswith("key_material_materialization_runtime.json") for item in report["artifact_references"])


def test_partial_report_distinguishes_quality_failure_and_counts_all_retained_candidates(tmp_path):
    write_json(tmp_path / "JSON-Config-Files/quality_acceptance.json", {"passed": False})
    write_json(tmp_path / "JSON-Config-Files/key_material_semantic_failures.json", {
        "incomplete": [{"subject_id": "EVT-1", "status": "failed"}],
    })
    for relative, ids in (
        ("Machine-Quarantine/Machine-Quarantine-Index.json", ["EVT-1", "EVT-2"]),
        ("Review-Candidates/Candidate-Index.json", ["EVT-2", "EVT-3"]),
    ):
        write_json(tmp_path / "Key-Materials" / relative, {"candidates": [{"event_id": value} for value in ids]})
    report = write_partial_delivery(tmp_path, {})
    assert report["status"] == "quality_attention"
    assert report["quarantined_event_count"] == 3
    assert report["pending_semantic_results"]
    assert not report["formal_archive_promotion_allowed"]
    html = (tmp_path / report["report"]).read_text()
    assert "未通过自动质量检查" in html
    assert "服务恢复" not in html


def test_partial_report_records_timeline_cost_and_quality_gap_without_releasing(tmp_path):
    quality = {"passed": False, "segmentation_integrity": {"groups": [
        {"group_id": "GROUP-0002", "passed": False},
    ]}}
    quality_path = tmp_path / "JSON-Config-Files/quality_acceptance.json"
    write_json(quality_path, quality)
    before = quality_path.read_bytes()
    write_json(tmp_path / "JSON-Config-Files/experiment_group_understanding.json", {
        "groups": [{"global_start_ms": 1000, "global_end_ms": 5000,
                    "model_understanding": {"steps": [{}], "overall_summary": "未观察到开盖",
                                            "uncertainties": ["<script>unknown</script>"]}}],
    })
    metrics = {"total_duration_seconds": 120, "tokens": {"run_total": {"total_tokens": 800}},
               "mllm_calls": [{"attempts": 2, "usage": {"total_tokens": 800, "unknown_attempt_count": 1}},
                              {"cache_reused": True, "attempts": 5}],
               "stage_durations": [{"stage": "mllm", "status": "completed", "duration_seconds": 100}]}
    report = write_partial_delivery(tmp_path, metrics)
    document = (tmp_path / report["report"]).read_text()
    assert "GROUP-0002 缺少第一与第三人称共同支持" in document
    assert "1.0 秒 → 5.0 秒" in document
    assert "未观察到开盖" in document
    assert "&lt;script&gt;unknown&lt;/script&gt;" in document
    assert "Token" not in document
    assert "quality_acceptance.json" not in document
    assert "控制文件" not in document
    assert "查看已保存步骤" in document
    exported = json.loads((tmp_path / report["export"]).read_text())
    assert exported["run_metrics"] == metrics
    assert exported["evidence_classification"] == "PARTIAL_EVIDENCE"
    assert exported["formal_archive_promotion_allowed"] is False
    assert exported["experiment_groups"][0]["model_understanding"]["overall_summary"] == "未观察到开盖"
    assert exported["quality_acceptance"] == quality
    assert quality_path.read_bytes() == before
    assert report["formal_archive_promotion_allowed"] is False
    assert not (tmp_path / "Lab-Daily-Reports").exists()


def test_quarantine_media_remains_previewable_without_becoming_formal(tmp_path, monkeypatch):
    settings = {"storage": {"archive_root": str(tmp_path), "staging_directory_name": "Processing"}}
    _, root, _ = fixed_archive_staging_paths(settings, "experiment", "run-partial")
    write_json(root / "JSON-Config-Files/pipeline_status.json", {"stage": "failed"})
    relative = "Key-Materials/Machine-Quarantine/EVT-1/Key-Frames/aligned_first_third.jpg"
    frame = root / relative
    frame.parent.mkdir(parents=True)
    frame.write_bytes(b"quarantined-frame")
    outside = tmp_path / "private.jpg"
    outside.write_bytes(b"outside")
    write_json(root / "Key-Materials/Machine-Quarantine/Machine-Quarantine-Index.json", {
        "candidates": [{"event_id": "EVT-1", "key_global_ms": 500,
                        "media": [str(outside), "../../private.jpg", relative]}],
    })
    monkeypatch.setattr(api, "_settings", lambda: settings)
    detail = api.staging_archive_detail("run-partial")
    assert detail["key_events"] == []
    assert detail["counts"]["key_events"] == 0
    assert len(detail["quarantined_materials"]) == 1
    item = detail["quarantined_materials"][0]
    assert item["evidence_classification"] == "PARTIAL_EVIDENCE"
    assert "private.jpg" not in item["frame_url"]
    client = TestClient(api.app)
    response = client.get(item["frame_url"])
    assert response.status_code == 200
    assert response.content == b"quarantined-frame"


@pytest.mark.parametrize("previous_state", ["failed", "partial"])
@pytest.mark.parametrize("mode", ["full", "resume"])
def test_retry_preserves_job_input_and_cache_and_survives_restart(tmp_path, monkeypatch, previous_state, mode):
    settings = {"project": {"cache_mode": "cold"}, "mllm": {"enabled": False}, "storage": {
        "archive_root": str(tmp_path / "nas"), "staging_directory_name": "Processing",
        "local_runtime_root": str(tmp_path / "runtime"),
    }}
    _, root, _ = fixed_archive_staging_paths(settings, "experiment", "run-retry")
    write_json(root / "JSON-Config-Files/pipeline_status.json", {"stage": previous_state})
    write_json(root / "JSON-Config-Files/partial_delivery.json", {"status": "incomplete"})
    store = DurableRunQueue(tmp_path / "queue.sqlite3")
    payload = {"settings": settings, "manifest": {"experiment_id": "experiment"}, "nas_root": str(root)}
    store.save_run("run-retry", {"state": previous_state, "nas_staging": str(root)})
    store.enqueue("run-retry", "run", payload)
    store.claim_next("old-worker", lease_seconds=60)
    store.finish("run-retry", "old-worker", previous_state)
    store.save_run("waiting", {"state": "queued"})
    store.enqueue("waiting", "run", {"other": True})
    monkeypatch.setattr(api, "_settings", lambda: settings)
    monkeypatch.setattr(api, "_persistent_queue", store)
    monkeypatch.setattr(api, "_runs", {})
    client = TestClient(api.app)
    plan = client.get("/api/runs/run-retry/recovery-plan").json()
    assert plan["actions"]["retry"] is True
    assert plan["actions"]["reports"] is True
    assert plan["actions"]["operations"] is False
    assert "key" not in plan and "credential_ref" not in plan
    assert store.get_job("run-retry")["status"] in {"failed", "completed"}
    rejected = client.post("/api/runs/run-retry/retry?revision=outdated")
    assert rejected.status_code == 409
    assert (root / "JSON-Config-Files/partial_delivery.json").is_file()
    video = root / "Experiment-Clips/kept.mp4"
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(b"synthetic-media-identity")
    before = video.stat().st_mtime_ns
    if mode == "resume":
        assert client.post("/api/runs/run-retry/retry?mode=resume").status_code == 422
    response = client.post("/api/runs/run-retry/retry", params={"mode": mode, "revision": plan["revision"]})
    if mode == "resume":
        assert video.read_bytes() == b"synthetic-media-identity"
        assert video.stat().st_mtime_ns == before
    assert response.status_code == 202
    assert response.json()["source_copy_bytes"] == 0
    queued = client.get("/api/runs/run-retry").json()
    assert queued["observability"]["status"]["stage"] == "queued"
    assert queued["observability"]["previous_attempt_status"]["stage"] == previous_state
    assert client.post("/api/runs/run-retry/retry").status_code == 409
    reopened = DurableRunQueue(store.database)
    job = reopened.get_job("run-retry")
    assert job["payload"]["manifest"] == payload["manifest"]
    assert job["payload"]["nas_root"] == str(root)
    assert job["payload"]["settings"]["project"]["cache_mode"] == "reuse"
    assert reopened.load_runs()["run-retry"]["state"] == "queued"
    assert len(reopened.load_runs()["run-retry"]["attempt_history"]) == 1
    monkeypatch.setattr(api, "_runs", reopened.load_runs())
    after_restart = client.get("/api/runs/run-retry").json()["observability"]
    assert after_restart["status"]["stage"] == "queued"
    assert after_restart["previous_attempt_status"]["stage"] == previous_state
    assert after_restart["partial_delivery"] == ({"status": "incomplete"} if mode == "resume" else {})
    assert bool(job["payload"]["settings"]["project"].get("resume_stages")) == (mode == "resume")
    assert reopened.claim_next("new-worker", lease_seconds=60).run_id == "waiting"
    assert (root / "JSON-Config-Files/Retry-Attempts/1/partial_delivery.json").is_file()


def test_registered_local_result_survives_restart_without_publishing_or_enqueueing(tmp_path, monkeypatch):
    runtime = tmp_path / "runtime"
    root = runtime / "independent-run/Archives/experiment"
    layout = ArchiveLayout(root)
    layout.create()
    settings = {"storage": {"local_runtime_root": str(runtime), "archive_root": str(tmp_path / "nas")}}
    write_json(layout.json_config / "run_manifest.json", {"experiment_id": "experiment", "views": []})
    write_json(root / "run_status.json", {"stage": "failed", "failed_stage": "package"})
    write_json(layout.json_config / "experiment_group_understanding.json", {"groups": [{"group_id": "G1"}]})
    write_json(layout.json_config / "Stage-Receipts/experiment_clips.json", {
        "stage": "experiment_clips", "status": "completed", "artifacts": ["Experiment-Clips"],
    })
    write_partial_delivery(root, {})
    media = root / "Experiment-Clips/clip.mp4"
    media.write_bytes(b"retained-video")
    result = register_partial_result(root, settings)
    assert result["enqueued"] is False
    assert result["media_copied_bytes"] == 0
    assert register_partial_result(root, settings)["run_id"] == result["run_id"]
    store = DurableRunQueue(runtime / "state/web_run_queue.sqlite3")
    assert store.get_job(result["run_id"]) is None
    assert store.claim_next("worker", lease_seconds=60) is None
    monkeypatch.setattr(api, "_settings", lambda: settings)
    monkeypatch.setattr(api, "_runs", store.load_runs())
    monkeypatch.setattr(api, "_persistent_queue", store)
    client = TestClient(api.app)
    assert client.get("/api/archives").json()["archives"] == []
    run = client.get("/api/runs").json()["runs"][0]
    assert run["result_available"] is True
    assert run["retry_available"] is False
    assert run["observability"]["retained_experiment_count"] == 1
    detail = client.get(f"/api/staging-runs/{result['run_id']}/archive").json()
    assert detail["name"] == "experiment"
    assert detail["read_only"] is True
    assert detail["retry_available"] is False
    assert client.get(detail["links"]["partial_report"]).status_code == 200
    assert client.get("/api/staging-file", params={"run_id": result["run_id"], "path": "Experiment-Clips/clip.mp4"}).content == b"retained-video"
    assert not (tmp_path / "nas").exists()
    assert not (root / ".VisionCortex-Current-Release.json").exists()
    # Replacing the registered manifest must revoke access, not silently expose
    # a different result under the same record.
    write_json(layout.json_config / "run_manifest.json", {"experiment_id": "different"})
    assert client.get(f"/api/staging-runs/{result['run_id']}/archive").status_code == 404
    assert client.get("/api/runs").json()["runs"][0]["result_available"] is False


def test_partial_registration_rejects_outside_runtime_and_symlink_escape(tmp_path):
    import pytest

    runtime = tmp_path / "runtime"
    runtime.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    alias = runtime / "alias"
    alias.symlink_to(outside, target_is_directory=True)
    settings = {"storage": {"local_runtime_root": str(runtime)}}
    for root in (outside, alias, runtime):
        with pytest.raises(ValueError, match="configured local runtime root"):
            register_partial_result(root, settings)
    assert not (runtime / "state").exists()


def test_retained_candidate_priority_is_traceable_and_never_discards_short_actions():
    from copy import deepcopy
    from visioncortex.partial_delivery import prioritize_retained_candidates

    def item(name, start, end, **overrides):
        return {"event_id": name, "start_ms": start, "end_ms": end, "timestamp_ms": start,
                "source_views": ["fp", "tp"], "source_roles": ["first_person", "third_person"],
                "group_id": "G1", "cv_action_type": "container_state_change",
                "cv_objects": ["hand", "tube_cap"], "frame_url": "/frame", "clip_url": "/clip",
                "disposition": "machine_quarantined_semantic_unavailable", **overrides}
    candidates = [item("long", 0, 2000), item("similar", 100, 1900),
                  item("next-action", 2100, 4100), item("quick", 5000, 5150),
                  item("same-role", 6000, 8000, source_roles=["third_person"]),
                  item("other-group", 0, 2000, group_id="G2"),
                  item("rejected", 0, 2000, disposition="machine_quarantined_semantic_rejected")]
    frozen = deepcopy(candidates)
    reviewed, receipt = prioritize_retained_candidates(candidates)
    assert candidates == frozen
    assert len(reviewed) == len(candidates)
    assert receipt["priority_count"] == 3
    assert receipt["deleted_count"] == 0
    assert receipt["formal_accuracy_claim_allowed"] is False
    indexed = {item["event_id"]: item for item in reviewed}
    assert indexed["similar"]["preview_review"]["related_event_id"] == "long"
    assert indexed["next-action"]["preview_review"]["priority"] is True
    assert indexed["quick"]["preview_review"]["reason_codes"] == ["short_candidate"]
    assert "missing_cross_role_sources" in indexed["same-role"]["preview_review"]["reason_codes"]
    assert [item["timestamp_ms"] for item in reviewed] == sorted(item["timestamp_ms"] for item in reviewed)


def test_context_camera_is_separate_from_primary_candidate_media(tmp_path, monkeypatch):
    settings = {'storage': {'archive_root': str(tmp_path), 'staging_directory_name': 'Processing'}}
    _, root, _ = fixed_archive_staging_paths(settings, 'experiment', 'run-context')
    write_json(root/'JSON-Config-Files/pipeline_status.json', {'stage': 'failed'})
    media = []
    for directory, suffix in [('Key-Frames', '.jpg'), ('Key-Clips', '.mp4')]:
        for name in ['First-Person', 'Third-Person', 'Aligned_First+Third']:
            relative = f'Key-Materials/Machine-Quarantine/E/{directory}/{name}{suffix}'
            path = root/relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'retained-media')
            media.append(relative)
    write_json(root/'Key-Materials/Machine-Quarantine/Machine-Quarantine-Index.json', {
        'candidates': [{'event_id':'E', 'key_global_ms':27600, 'media':media, 'view_pairing': {
            'first_person_view':'fp', 'third_person_view':'context',
            'pair_evidence_status':'context_only_missing_key_time_support',
            'same_action_pair_verified':False,
            'candidates': {'first_person':[{'view_id':'fp','candidate_supported_at_key':True}],
                           'third_person':[{'view_id':'context','candidate_supported_at_key':False}]},
        }}]})
    monkeypatch.setattr(api, '_settings', lambda: settings)
    detail = api.staging_archive_detail('run-context')
    item = detail['quarantined_materials'][0]
    assert 'First-Person.jpg' in item['frame_url']
    assert 'First-Person.mp4' in item['clip_url']
    assert 'Third-Person.jpg' in item['context_media']['frame_url']
    assert 'Third-Person.mp4' in item['context_media']['clip_url']
    assert detail['counts']['key_events'] == 0
    assert len(media) == 6


def test_motion_rejection_is_visible_without_promoting_candidates(tmp_path, monkeypatch):
    settings = {"storage": {"archive_root": str(tmp_path), "staging_directory_name": "Processing"}}
    _, root, _ = fixed_archive_staging_paths(settings, "experiment", "motion-result")
    write_json(root / "JSON-Config-Files/pipeline_status.json", {"stage": "failed"})
    receipt = {"enabled": True, "counts": {"contradicted": 201}, "candidates": [
        {"candidate_id": f"move-{i}", "view_id": "fp", "start_ms": i * 1000,
         "end_ms": i * 1000 + 500, "objects": ["pipette"], "status": "contradicted",
         "checks": [{"input_gray_sha256": ["a", "b"]}]} for i in range(201)]}
    write_json(root / "JSON-Config-Files/movement_visual_verification.json", receipt)
    monkeypatch.setattr(api, "_settings", lambda: settings)
    detail = api.staging_archive_detail("motion-result")
    assert detail["key_events"] == []
    review = detail["movement_screening"]
    assert review["physical_action_confirmed"] is False
    assert review["total_candidates"] == 201
    assert len(review["candidates"]) == 200
    assert "checks" not in review["candidates"][0]
    assert TestClient(api.app).get(review["report_url"]).json() == receipt


def test_independent_speech_and_understanding_survive_visual_failure(tmp_path):
    from visioncortex.partial_delivery import component_results
    speech_path = tmp_path / 'JSON-Config-Files/speech.json'
    write_json(speech_path, {'status':'completed', 'sources':[{'chunks':[{'segment_count':156}]}]})
    write_json(tmp_path / 'JSON-Config-Files/speech_understanding.json', {
        'status':'completed', 'index_sha256':hashlib.sha256(speech_path.read_bytes()).hexdigest(), 'parts':[{}]})
    write_json(tmp_path / 'JSON-Config-Files/quality_acceptance.json', {'passed':False})
    result = {item['key']:item['state'] for item in component_results(tmp_path)}
    assert result == {'speech':'completed', 'understanding':'completed', 'visual':'insufficient', 'reports':'not_generated'}
    speech_path.write_text('{}')
    assert component_results(tmp_path)[1]['state'] == 'not_available'


def test_report_refresh_never_calls_models_or_weakens_quality(tmp_path, monkeypatch):
    from visioncortex.stage_refresh import refresh
    from visioncortex import daily_reports, speech_semantics
    def forbidden(*args, **kwargs):
        raise AssertionError('quality failure must not generate formal report or invoke model')
    monkeypatch.setattr(daily_reports, 'generate_daily_report_from_archive', forbidden)
    monkeypatch.setattr(speech_semantics, 'refresh_recording_understanding', forbidden)
    quality = tmp_path / 'JSON-Config-Files/quality_acceptance.json'
    write_json(quality, {'passed':False})
    before = quality.read_bytes()
    result = refresh(tmp_path, 'reports', {})
    assert quality.read_bytes() == before
    assert result['model_calls'] == []
    assert result['cv_invocations'] == result['asr_invocations'] == result['source_media_decodes'] == 0
    assert (tmp_path / 'Partial-Results/Partial-Evidence-Report.html').is_file()


def test_refresh_queue_is_durable_and_excludes_parent_retry(tmp_path, monkeypatch):
    settings = {'project':{}, 'mllm':{'enabled':False}, 'storage':{
        'archive_root':str(tmp_path/'nas'), 'staging_directory_name':'Processing',
        'local_runtime_root':str(tmp_path/'runtime')}}
    _, root, _ = fixed_archive_staging_paths(settings, 'experiment', 'parent')
    write_json(root/'JSON-Config-Files/quality_acceptance.json', {'passed':False})
    store = DurableRunQueue(tmp_path/'queue.sqlite3')
    store.save_run('parent', {'state':'failed','nas_staging':str(root)})
    store.enqueue('parent','run',{'settings':settings})
    store.claim_next('previous', lease_seconds=60)
    store.finish('parent','previous','failed')
    monkeypatch.setattr(api,'_settings',lambda:settings)
    monkeypatch.setattr(api,'_persistent_queue',store)
    monkeypatch.setattr(api,'_runs',store.load_runs())
    client = TestClient(api.app)
    response = client.post('/api/runs/parent/refresh/reports')
    assert response.status_code == 202, response.text
    assert client.post('/api/runs/parent/refresh/reports').status_code == 409
    assert client.post('/api/runs/parent/retry').status_code == 409
    reopened = DurableRunQueue(store.database)
    job = reopened.claim_next('next', lease_seconds=60)
    assert job.kind == 'stage_refresh'
    assert job.payload['parent_run_id'] == 'parent'
    api._dispatch_persisted_job(job)
    assert api._runs[job.run_id]['state'] == 'completed'
    assert api._runs['parent']['state'] == 'failed'
    assert (root/'JSON-Config-Files/reports_refresh.json').is_file()


def test_readable_partial_reports_bind_exports_without_promoting_and_hide_tampering(tmp_path, monkeypatch):
    layout = ArchiveLayout(tmp_path)
    layout.create()
    write_json(layout.json_config / "pipeline_status.json", {"stage":"failed"})
    write_json(layout.json_config / "quality_acceptance.json", {"passed":False})
    write_json(layout.json_config / "run_metrics.json", {"tokens":{"run_total":{"input_tokens":0,"output_tokens":None,"total_tokens":None}}})
    quality_before = (layout.json_config / "quality_acceptance.json").read_bytes()
    receipt = write_partial_delivery(tmp_path, {"tokens":{"run_total":{"input_tokens":0,"output_tokens":None,"total_tokens":None}}})
    assert "readable_report_error" not in receipt
    for item in receipt["readable_reports"].values():
        assert hashlib.sha256((tmp_path/item["path"]).read_bytes()).hexdigest() == item["sha256"]
    assert (tmp_path/receipt["readable_reports"]["pdf"]["path"]).read_bytes().startswith(b'%PDF-')
    assert (layout.json_config/"quality_acceptance.json").read_bytes() == quality_before
    assert not (layout.json_config/"daily_report_manifest.json").exists()
    export = json.loads((tmp_path/receipt["export"]).read_text())
    assert export["artifact_references"]
    assert export["traceability"]["source_video_bodies_revalidated"] is False
    monkeypatch.setattr(api,"_resolve_staging_run",lambda _:tmp_path)
    detail = api.staging_archive_detail('R')
    assert '/api/staging-file?' in detail['links']['partial_pdf']
    (tmp_path/receipt['readable_reports']['pdf']['path']).write_bytes(b'tampered')
    assert 'partial_pdf' not in api.staging_archive_detail('R')['links']


def test_stage_report_images_reference_only_linked_retained_frames(tmp_path):
    from visioncortex.partial_reports import retained_report_visuals
    frame = tmp_path / 'Key-Materials/frame.jpg'
    frame.parent.mkdir()
    frame.write_bytes(b'existing derived image')
    groups = [{"group_id":"G","model_understanding":{"steps":[{"supporting_event_ids":["E"]}]}}]
    events = [{"event_id":"E","key_frames":{"aligned_first_third":"Key-Materials/frame.jpg"}},
              {"event_id":"unused","key_frames":{"aligned_first_third":"Key-Materials/missing.jpg"}}]
    rows = retained_report_visuals(tmp_path,groups,events)
    assert len(rows) == 1 and rows[0]['event_id'] == 'E'
    assert rows[0]['sha256'] == hashlib.sha256(frame.read_bytes()).hexdigest()
    assert rows[0]['evidence_classification'] == 'PARTIAL_EVIDENCE'
    events[0]['key_frames']['aligned_first_third'] = '../outside.jpg'
    assert retained_report_visuals(tmp_path,groups,events) == []
