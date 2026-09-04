import json
import time
from pathlib import Path

import pytest

from labvision_evidence import api
from labvision_evidence.archive import ArchiveLayout
from labvision_evidence.config import load_config
from labvision_evidence.pipeline import EvidencePipeline
from labvision_evidence.schemas import ActionType, EvidenceEvent
from labvision_evidence.storage import (
    IncrementalArchivePublisher,
    fixed_archive_staging_paths,
    initialize_nas_archive,
    promote_fixed_archive,
    run_staging_roots,
    safe_archive_name,
)


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


def test_stage_preview_reads_completed_clips_without_final_package(tmp_path, monkeypatch):
    config = {"storage": {"archive_root": str(tmp_path), "staging_directory_name": "Processing"}}
    _, root, _ = fixed_archive_staging_paths(config, "experiment", "run-preview")
    json_dir = root / "JSON-Config-Files"
    json_dir.mkdir(parents=True)
    (json_dir / "pipeline_status.json").write_text(json.dumps({"stage": "mllm"}))
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
    receipts.mkdir()
    (receipts / "key_materials.json").write_text(json.dumps({"stage": "key_materials", "status": "completed"}))
    preview = api.staging_archive_detail("run-preview")
    assert preview["key_events"] == []
    assert preview["preliminary_materials"][0]["review_status"] == "pending_semantic_review"
    assert preview["preliminary_materials"][0]["timestamp_ms"] == 123


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


def test_task_page_exposes_each_completed_stage_output():
    app_js = (
        Path(__file__).parents[1]
        / "src"
        / "labvision_evidence"
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


def test_operations_page_exposes_only_user_facing_nas_locations():
    app_js = (
        Path(__file__).parents[1]
        / "src"
        / "labvision_evidence"
        / "web"
        / "app.js"
    ).read_text(encoding="utf-8")

    operations = app_js.split("function renderOperations()", 1)[1].split(
        "async function loadArchive", 1
    )[0]
    assert "NAS / VisionCortexExperimentArchive" in operations
    assert "NAS / VisionCortexExperimentCache" in operations
    assert "管理员工具" in operations
    assert "modelCandidatePanel" not in operations
    assert "health.archive_root" not in operations
    assert "health.nas_archive_root" not in operations
    assert "local_runtime_root" not in operations
    assert "local_cache_root" not in operations
    assert "benchmark.index_csv" not in operations
    assert "health.model" not in operations
