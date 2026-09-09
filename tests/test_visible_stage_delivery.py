import json
import time
from pathlib import Path

import pytest

from visioncortex import api
from visioncortex.archive import ArchiveLayout
from visioncortex.config import load_config
from visioncortex.pipeline import EvidencePipeline
from visioncortex.schemas import ActionType, EvidenceEvent
from visioncortex.storage import (
    IncrementalArchivePublisher,
    fixed_archive_staging_paths,
    initialize_nas_archive,
    promote_fixed_archive,
    run_staging_roots,
    safe_archive_name,
)


def test_stage_versions_retain_overwritten_json_and_do_not_copy_media(tmp_path):
    from visioncortex.stage_versions import save_version
    root = tmp_path / 'archive'
    directory = root / 'JSON-Config-Files'
    directory.mkdir(parents=True)
    source = directory / 'groups.json'
    source.write_text('{"phase":1}')
    video = root / 'clip.mp4'
    video.write_bytes(b'fixture-video-not-real')
    first = save_version(root, [directory, video], {'stage':'clips'})
    source.write_text('{"phase":2}')
    second = save_version(root, [directory], {'stage':'review'})
    payload = json.loads(first.read_text())
    row = next(item for item in payload['files'] if item['path'].endswith('groups.json'))
    assert (root / row['snapshot']).read_text() == '{"phase":1}'
    import hashlib
    assert hashlib.sha256((root / row['snapshot']).read_bytes()).hexdigest() == row['sha256']
    assert payload['media_bodies_revalidated'] is False
    assert not any(first.parent.rglob('*.mp4'))
    assert len(json.loads(second.read_text())['files']) == 1  # no recursive snapshot copying
    assert first != second


def test_completed_stage_visible_before_final_acceptance(tmp_path):
    config = {"storage": {"archive_root": str(tmp_path / "nas"), "staging_directory_name": "Processing"}}
    final, staging, history = fixed_archive_staging_paths(config, "experiment", "run-1")
    config["storage"]["active_archive_path"] = str(staging)
    initialize_nas_archive(config, "experiment")
    local = tmp_path / "work"
    source = local / "Experiment-Clips" / "first.mp4"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"completed-stage-media")
    publisher = IncrementalArchivePublisher(local, staging)
    publisher.publish_file(source)
    publisher.publish_status({"stage": "key_materials", "progress": 0.84})
    assert (staging / "Experiment-Clips/first.mp4").read_bytes() == source.read_bytes()
    assert (staging / "处理状态.txt").is_file()
    assert not final.exists()
    assert api._find_staging_run(config, "run-1") == staging
    with pytest.raises(RuntimeError):
        promote_fixed_archive(staging, final, history)
    assert not final.exists()
    assert (staging / "Experiment-Clips/first.mp4").is_file()


def test_visible_staging_preserves_legacy_run_discovery(tmp_path):
    assert safe_archive_name("Processing") == "Experiment-Processing"
    config = {"storage": {"archive_root": str(tmp_path), "staging_directory_name": "Processing"}}
    old = tmp_path / ".VisionCortex-Run-Staging/experiment/old-run"
    status = old / "JSON-Config-Files/pipeline_status.json"
    status.parent.mkdir(parents=True)
    status.write_text(json.dumps({"stage": "failed"}))
    assert api._find_staging_run(config, "old-run") == old
    config["storage"]["staging_directory_name"] = "../outside"
    with pytest.raises(ValueError):
        run_staging_roots(config)


def test_preflight_failure_can_retry_without_pipeline_status(tmp_path, monkeypatch):
    config = {"storage": {"archive_root": str(tmp_path / "archive")}}
    _, staging, _ = fixed_archive_staging_paths(config, "experiment", "run-preflight")
    staging.mkdir(parents=True)
    monkeypatch.setattr(api, "_runs", {"run-preflight": {"nas_staging": str(staging)}})
    assert api._find_staging_run(config, "run-preflight") == staging.resolve()
    outside = tmp_path / "elsewhere/experiment/run-preflight"
    outside.mkdir(parents=True)
    api._runs["run-preflight"]["nas_staging"] = str(outside)
    assert api._find_staging_run(config, "run-preflight") is None
    api._runs["another-run"] = {"nas_staging": str(staging)}
    assert api._find_staging_run(config, "another-run") is None


def test_stage_preview_reads_completed_clips_without_final_package(tmp_path, monkeypatch):
    config = {"storage": {"archive_root": str(tmp_path), "staging_directory_name": "Processing"}}
    _, root, _ = fixed_archive_staging_paths(config, "experiment", "run-preview")
    json_dir = root / "JSON-Config-Files"
    json_dir.mkdir(parents=True)
    (json_dir / "pipeline_status.json").write_text(json.dumps({"stage": "mllm"}))
    (json_dir / "Stage-Receipts").mkdir()
    (json_dir / "Stage-Receipts/experiment_clips.json").write_text(json.dumps({
        "stage": "experiment_clips", "status": "completed",
    }))
    (json_dir / "experiment_group_understanding.json").write_text(json.dumps({
        "groups": [{"group_id": "GROUP-1", "archive_folder": "weighing",
                    "experiment_name": "称量", "global_start_ms": 0,
                    "global_end_ms": 1000, "model_understanding": {"steps": []}}]
    }))
    clip = root / "Experiment-Clips/weighing/Aligned_First+Third.mp4"
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"completed-stage-media")
    (clip.parent / "First-Person.mp4").write_bytes(b"first-person-media")
    (clip.parent / "Third-Person.mp4").write_bytes(b"third-person-media")
    monkeypatch.setattr(api, "_settings", lambda: config)
    preview = api.staging_archive_detail("run-preview")
    assert preview["path"] == str(root.resolve())
    assert preview["observability"]["status"]["stage"] == "mllm"
    assert preview["experiments"][0]["name"] == "称量"
    assert preview["experiments"][0]["aligned_video_url"].startswith(
        "/api/staging-file?run_id=run-preview&path="
    )
    assert preview["experiments"][0]["first_person_video_url"].startswith(
        "/api/staging-file?run_id=run-preview&path="
    )
    assert preview["experiments"][0]["third_person_video_url"].startswith(
        "/api/staging-file?run_id=run-preview&path="
    )
    archived = api._archive_detail_from_root(root, "experiment")
    assert archived["experiments"][0]["aligned_video_url"].startswith(
        "/api/archive-file?archive=experiment&path="
    )
    assert archived["experiments"][0]["first_person_video_url"].startswith(
        "/api/archive-file?archive=experiment&path="
    )
    assert archived["experiments"][0]["third_person_video_url"].startswith(
        "/api/archive-file?archive=experiment&path="
    )
    assert not (tmp_path / "experiment").exists()
    with pytest.raises(api.HTTPException):
        api.staging_file("run-preview", "../../../../outside")

    material = root / "Key-Materials/frame.jpg"
    material.parent.mkdir()
    material.write_bytes(b"preview-frame")
    (material.parent / "Key-Material-Category-Index.json").write_text(json.dumps({
        "experiments": [{"action_categories": [{"events": [{
            "event_id": "EVT-1", "peak_timestamp_us": 123000,
            "key_frames": {"aligned_first_third": "Key-Materials/frame.jpg"}
        }]}]}]
    }))
    # A folder alone must not make an unfinished stage appear delivered.
    assert api.staging_archive_detail("run-preview")["preliminary_materials"] == []
    receipts = json_dir / "Stage-Receipts"
    receipts.mkdir(exist_ok=True)
    (receipts / "key_materials.json").write_text(json.dumps({"stage": "key_materials", "status": "completed"}))
    preview = api.staging_archive_detail("run-preview")
    assert preview["key_events"] == []
    assert preview["preliminary_materials"][0]["review_status"] == "pending_semantic_review"
    assert preview["preliminary_materials"][0]["timestamp_ms"] == 123


def test_stage_api_keeps_auxiliary_video_without_counting_it_as_experiment(tmp_path):
    from visioncortex.activity_review import binding, save
    root = tmp_path / 'archive'
    controls = root / 'JSON-Config-Files'
    controls.mkdir(parents=True)
    (controls / 'pipeline_status.json').write_text(json.dumps({'stage':'failed'}))
    group = {'group_id':'G', 'archive_folder':'one', 'global_start_ms':0,
             'global_end_ms':1000, 'model_understanding':{'steps':[]}}
    (controls / 'experiment_group_understanding.json').write_text(json.dumps({'groups':[group]}))
    clip = root / 'Experiment-Clips/one/First-Person.mp4'
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b'fixture')
    save(root, group, binding(root, group), {'kind':'equipment_organization','workflow_relation':'standalone','reason':'整理仪器'})
    data = api._archive_detail_from_root(root, 'example', staging_run_id='run-1')
    assert data['counts']['experiments'] == 0
    assert data['counts']['auxiliary_activities'] == 1
    assert data['experiments'][0]['first_person_video_url']
    assert data['experiments'][0]['activity_assessment']['label'] == '器材整理'


def test_stage_preview_includes_both_candidate_indexes_without_double_counting(tmp_path, monkeypatch):
    config = {"storage": {"archive_root": str(tmp_path), "staging_directory_name": "Processing"}}
    _, root, _ = fixed_archive_staging_paths(config, "experiment", "run-candidates")
    json_dir = root / "JSON-Config-Files"
    json_dir.mkdir(parents=True)
    (json_dir / "pipeline_status.json").write_text(json.dumps({"stage": "failed"}))
    for directory, filename, ids in (
        ("Machine-Quarantine", "Machine-Quarantine-Index.json", ["EVT-1", "EVT-2"]),
        ("Review-Candidates", "Candidate-Index.json", ["EVT-2", "EVT-3"]),
    ):
        folder = root / "Key-Materials" / directory
        folder.mkdir(parents=True)
        entries = []
        for event_id in ids:
            media = folder / f"{event_id}.jpg"
            media.write_bytes(b"contract-fixture-not-a-real-image")
            entries.append({"event_id": event_id, "media": [str(media.relative_to(root))]})
        (folder / filename).write_text(json.dumps({"candidates": entries}))
    monkeypatch.setattr(api, "_settings", lambda: config)
    preview = api.staging_archive_detail("run-candidates")
    assert {item["event_id"] for item in preview["quarantined_materials"]} == {"EVT-1", "EVT-2", "EVT-3"}
    assert len(preview["quarantined_materials"]) == 3
    assert not preview["key_events"]


def test_understanding_checkpoint_survives_later_revision_and_failure(tmp_path):
    config = load_config()
    config["storage"]["run_output_mode"] = "nas_direct"
    pipeline = EvidencePipeline(config)
    pipeline._run_started_perf = time.perf_counter()
    layout = ArchiveLayout(tmp_path / "archive")
    layout.create()
    event = EvidenceEvent(
        event_id="EVT-1", action_type=ActionType.HAND_OBJECT_CONTACT,
        global_start_ms=0, global_end_ms=1000, key_global_ms=500,
        confidence=0.9, accepted=True,
        objects=["hand", "tube"], audit_reason="checkpoint regression",
        supporting_views=[], supporting_roles=[], candidates=[],
        model_understanding={
            "status": "completed", "current_step": "initial observation",
            "usage": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
        },
    )
    pipeline._status(layout, "mllm", 0.92, "understanding")
    paths = pipeline._checkpoint_key_material_understanding(layout, "mllm", [event], [])
    pipeline._complete_stage(layout, "mllm", paths)
    initial = layout.json_config / "Stage-Outputs/mllm.json"
    initial_bytes = initial.read_bytes()
    initial_receipt = (layout.json_config / "Stage-Receipts/mllm.json").read_bytes()

    pipeline._status(layout, "material_refinement", 0.93, "refinement")
    event.accepted = False
    event.model_understanding["current_step"] = "revised observation"
    paths = pipeline._checkpoint_key_material_understanding(
        layout, "material_refinement", [event], [], {"records": [{"event_id": "EVT-1"}]}
    )
    pipeline._complete_stage(layout, "material_refinement", paths)
    pipeline._status(layout, "semantic_refinement", 0.95, "step review")
    pipeline._status(layout, "failed", 0.95, "step review failed")

    assert initial.read_bytes() == initial_bytes
    assert (layout.json_config / "Stage-Receipts/mllm.json").read_bytes() == initial_receipt
    assert json.loads(initial_bytes)["events"][0]["accepted"] is True
    latest = json.loads((layout.json_config / "key_material_model_understanding.json").read_text())
    assert latest["events"][0]["accepted"] is False
    assert latest["review_status"] == "pending_semantic_refinement"
    # Rejected events still consumed tokens; later curation must not erase them.
    metrics = json.loads((layout.json_config / "run_metrics_live.json").read_text())
    assert metrics["tokens"]["run_total"]["total_tokens"] == 120
    snapshot = api._run_snapshot_from_root(layout.root)
    assert snapshot["status"]["failed_stage"] == "semantic_refinement"
    assert {item["stage"] for item in snapshot["stage_receipts"]} == {"mllm", "material_refinement"}
    assert all(item["available"] for receipt in snapshot["stage_receipts"] for item in receipt["artifacts"])


def test_checkpoint_receipt_waits_for_successful_publication(tmp_path, monkeypatch):
    pipeline = EvidencePipeline(load_config())
    pipeline._run_started_perf = time.perf_counter()
    layout = ArchiveLayout(tmp_path / "work")
    layout.create()
    nas = tmp_path / "nas"
    publisher = IncrementalArchivePublisher(layout.root, nas)
    pipeline._publisher = publisher
    first = pipeline._checkpoint_key_material_understanding(layout, "mllm", [], [])
    pipeline._complete_stage(layout, "mllm", first)
    preserved = (nas / "JSON-Config-Files/Stage-Outputs/mllm.json").read_bytes()
    inventory_path = nas / "阶段产出清单.json"
    inventory_before = inventory_path.read_bytes()
    inventory = json.loads(inventory_before)
    assert inventory["formal_release"] is False
    assert [item["stage"] for item in inventory["stages"]] == ["mllm"]
    assert all((nas / name).exists() for item in inventory["stages"] for name in item["artifacts"])
    next_paths = pipeline._checkpoint_key_material_understanding(
        layout, "material_refinement", [], []
    )

    def fail_publish(_source):
        raise OSError("storage unavailable")

    monkeypatch.setattr(publisher, "publish_file", fail_publish)
    with pytest.raises(OSError, match="storage unavailable"):
        pipeline._complete_stage(layout, "material_refinement", next_paths)
    assert (nas / "JSON-Config-Files/Stage-Outputs/mllm.json").read_bytes() == preserved
    assert (nas / "JSON-Config-Files/Stage-Receipts/mllm.json").exists()
    assert not (layout.json_config / "Stage-Receipts/material_refinement.json").exists()
    assert not (nas / "JSON-Config-Files/Stage-Receipts/material_refinement.json").exists()
    assert inventory_path.read_bytes() == inventory_before


def test_task_page_exposes_each_completed_stage_output():
    app_js = (
        Path(__file__).parents[1]
        / "src"
        / "visioncortex"
        / "web"
        / "app.js"
    ).read_text(encoding="utf-8")

    assert 'summary>查看各环节与生成文件</summary>' in app_js
    assert 'runObservabilityCard(run,false)' in app_js
    assert '${outputsOpen ? "open" : ""}' in app_js
    assert "function stageArtifactUrl" in app_js
    assert "function stageResultRoute" in app_js
    assert 'class="journey-artifacts"' in app_js
    assert '"原始输入索引"' in app_js
    assert '<h2>质量验收</h2>' in app_js
    assert 'videoTile("同步双视角", experiment.aligned_video_url' in app_js
    assert 'videoTile("第一人称", experiment.first_person_video_url)' in app_js
    assert 'videoTile("第三人称", experiment.third_person_video_url)' in app_js


def test_operations_page_shows_selected_output_and_cache_locations():
    app_js = (
        Path(__file__).parents[1]
        / "src"
        / "visioncortex"
        / "web"
        / "app.js"
    ).read_text(encoding="utf-8")

    operations = app_js.split("function renderOperations()", 1)[1].split(
        "async function loadArchive", 1
    )[0]
    assert "NAS / VisionCortexExperimentArchive" not in operations
    assert "NAS / VisionCortexExperimentCache" not in operations
    assert 'isNasMode() ? "NAS 存储" : "本地存储"' in operations
    assert 'esc(health.archive_root || "未配置")' in operations
    assert 'esc(health.fixed_benchmark?.local_cache_root || "未配置")' in operations
    assert "health.fixed_benchmark?.enabled !== false" in operations
    assert "管理员工具" in operations
    assert "modelCandidatePanel" not in operations
    assert "health.nas_archive_root" not in operations
    assert "local_runtime_root" not in operations
    assert "benchmark.index_csv" not in operations
    assert "health.model" not in operations


def test_retry_preview_uses_current_groups_and_receipts(tmp_path):
    root = tmp_path / "archive"
    config = root / "JSON-Config-Files"
    receipts = config / "Stage-Receipts"
    receipts.mkdir(parents=True)
    def save(path, value):
        path.write_text(json.dumps(value))
    save(config / "pipeline_status.json", {
        "stage": "mllm", "updated_at": "2026-09-07T10:05:00+00:00",
        "elapsed_seconds": 300,
    })
    save(config / "run_metrics.json", {"run_started_at": "2026-09-07T09:00:00+00:00", "total_duration_seconds": 2400})
    save(config / "run_metrics_live.json", {"run_started_at": "2026-09-07T10:00:00+00:00", "total_duration_seconds": 300})
    save(config / "delivery_metrics.json", {"total_duration_seconds": 2500})
    old = {"group_id": "GROUP-1", "archive_folder": "old", "experiment_name": "previous"}
    current = {"group_id": "GROUP-1", "archive_folder": "current", "experiment_name": "current", "model_understanding": {"steps": [{"description": "observed"}]}}
    save(config / "evidence_package.json", {"experiment_groups": [old]})
    save(config / "experiment_group_understanding.json", {"groups": [current]})
    for name in ("old", "current"):
        folder = root / "Experiment-Clips" / name
        folder.mkdir(parents=True)
        (folder / "First-Person.mp4").write_bytes(b"test-media")
    save(receipts / "experiment_understanding.json", {"stage": "experiment_understanding", "status": "completed", "completed_at": "2026-09-07T10:04:00+00:00"})
    save(receipts / "experiment_clips.json", {"stage": "experiment_clips", "status": "completed", "completed_at": "2026-09-07T10:04:30+00:00"})
    save(receipts / "semantic_refinement.json", {"stage": "semantic_refinement", "status": "completed", "completed_at": "2026-09-07T09:40:00+00:00"})
    detail = api._archive_detail_from_root(root, "experiment", staging_run_id="retry")
    assert [item["name"] for item in detail["experiments"]] == ["current"]
    assert detail["experiments"][0]["steps"] == [{"description": "observed"}]
    assert detail["metrics"]["total_duration_seconds"] == 300
    assert {item["stage"] for item in detail["observability"]["stage_receipts"]} == {"experiment_understanding", "experiment_clips"}
    assert (root / "Experiment-Clips/old/First-Person.mp4").exists()
    # The completed package also excludes orphan folders from older attempts.
    save(config / "evidence_package.json", {"experiment_groups": [current]})
    assert api._archive_detail_from_root(root, "experiment")["counts"]["experiments"] == 1


def test_retry_retains_derived_media_without_touching_originals(tmp_path):
    root = tmp_path / "archive"
    sources = {"Original-Experiment-Videos/fp/video.mp4": b"original",
               "JSON-Config-Files/Input-Manifests/input_seal.json": b"seal",
               "JSON-Config-Files/run_manifest.json": b"input-manifest"}
    derived = {"Key-Materials/Machine-Quarantine/event/Key-Frames/frame.jpg": b"old-frame",
               "Experiment-Clips/old/First-Person.mp4": b"old-clip",
               "JSON-Config-Files/evidence_package.json": b"old-package",
               "JSON-Config-Files/Stage-Receipts/mllm.json": b"old-receipt"}
    for relative, payload in {**sources, **derived}.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    api._retain_retry_outputs(root, 3)
    for relative, payload in sources.items():
        assert (root / relative).read_bytes() == payload
    for relative, payload in derived.items():
        assert not (root / relative).exists()
        assert (root / "JSON-Config-Files/Retry-Attempts/3/Derived" / relative).read_bytes() == payload
    fresh = root / "Key-Materials/Machine-Quarantine/event/Key-Frames/frame.jpg"
    fresh.parent.mkdir(parents=True)
    fresh.write_bytes(b"new-frame")
    api._retain_retry_outputs(root, 4)
    assert (root / "JSON-Config-Files/Retry-Attempts/4/Derived/Key-Materials/Machine-Quarantine/event/Key-Frames/frame.jpg").read_bytes() == b"new-frame"


def test_running_video_preview_retains_queue_lease_and_recovers_early_clip_paths(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from visioncortex.run_queue import DurableRunQueue

    root = tmp_path / "archive"
    config = root / "JSON-Config-Files"
    receipts = config / "Stage-Receipts"
    receipts.mkdir(parents=True)
    def save(path, value):
        path.write_text(json.dumps(value))
    save(config / "pipeline_status.json", {"stage": "mllm", "updated_at": "2026-09-07T10:05:00+00:00", "elapsed_seconds": 300})
    group = {"group_id": "G1", "archive_folder": None, "global_start_ms": 0, "global_end_ms": 1000,
             "experiment_name": "current", "model_understanding": {"steps": []}}
    save(config / "experiment_group_understanding.json", {"groups": [group]})
    for stage in ("experiment_understanding", "experiment_clips"):
        save(receipts / f"{stage}.json", {"stage": stage, "status": "completed", "completed_at": "2026-09-07T10:04:30+00:00"})
    folder = root / "Experiment-Clips/001_current"
    folder.mkdir(parents=True)
    video = folder / "First-Person.mp4"
    video.write_bytes(b"range-contract-fixture")
    save(folder / "First-Person.json", {"artifact_type": "experiment_view_video", "group": {**group, "archive_folder": folder.name}})
    store = DurableRunQueue(tmp_path / "queue.sqlite3")
    store.save_run("R", {"state": "mllm"})
    store.enqueue("R", "run", {})
    assert store.claim_next("worker", lease_seconds=600)
    before = store.get_job("R")
    monkeypatch.setattr(api, "_resolve_staging_run", lambda _: root)
    monkeypatch.setattr(api, "_runs", {"R": {"run_id": "R", "experiment_id": "current", "state": "mllm"}})
    # No ASGI lifespan: start no runtime workers in this contract test.
    client = TestClient(api.app, client=("127.0.0.1", 30000))
    detail = client.get("/api/staging-runs/R/archive").json()
    assert detail["experiments"][0]["folder"] == folder.name
    response = client.get(detail["experiments"][0]["first_person_video_url"], headers={"Range": "bytes=0-4"})
    assert response.status_code == 206 and response.content == b"range"
    assert store.get_job("R") == before
    assert api._runs["R"]["state"] == "mllm"
    assert store.finish("R", "worker", "completed")
    # Before clips finish, unfinished MP4s are not advertised as playable.
    (receipts / "experiment_clips.json").unlink()
    assert api._archive_detail_from_root(root, "current", staging_run_id="R")["experiments"] == []


def test_old_clip_metadata_cannot_recover_a_different_current_boundary(tmp_path):
    folder = tmp_path / "clip"
    folder.mkdir()
    source = {"group_id": "G1", "archive_folder": "clip", "global_start_ms": 0, "global_end_ms": 1000}
    (folder / "First-Person.json").write_text(json.dumps({"artifact_type": "experiment_view_video", "group": source}))
    groups = {"G1": {**source, "archive_folder": None, "global_end_ms": 2000}}
    assert api._stage_clip_group(folder, groups, {}) is None


@pytest.mark.parametrize("outcome", ["skipped", "failed"])
def test_optional_component_outcome_is_visible_without_failing_video(tmp_path, outcome):
    layout = ArchiveLayout(tmp_path / "archive")
    layout.create()
    pipeline = EvidencePipeline(load_config())
    pipeline._run_started_iso = "2026-01-01T00:00:00+00:00"
    pipeline._run_started_perf = time.perf_counter()
    receipts = layout.json_config / "Stage-Receipts"
    receipts.mkdir()
    (receipts / "old.json").write_text(json.dumps({"stage": "old", "completed_at": "2025-01-01T00:00:00+00:00"}))
    pipeline._status(layout, "speech", .09, "录音")
    pipeline._complete_stage(layout, "speech", status=outcome, reason="没有可用转写，视频分析继续")
    pipeline._status(layout, "motion_probe", .1, "视频处理中")
    status = json.loads((layout.root / "run_status.json").read_text())
    assert status["stage"] == "motion_probe" and status["failed_stage"] is None
    assert "speech" not in status["completed_stages"]
    assert status["stage_outcomes"]["speech"]["status"] == outcome
    inventory = json.loads((layout.root / "阶段产出清单.json").read_text())
    assert [item["stage"] for item in inventory["stages"]] == ["speech"]
    assert inventory["stages"][0]["reason"] == "没有可用转写，视频分析继续"
