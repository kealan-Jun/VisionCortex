import hashlib
import json
import threading
from pathlib import Path

from fastapi.testclient import TestClient

from visioncortex import api, indexing
from visioncortex.archive import _artifact_json
from visioncortex.decisions import decision_receipt
from visioncortex.indexing import (
    ARTIFACT_REGISTRY_NAME,
    DECISION_REGISTRY_NAME,
    EVIDENCE_REGISTRY_NAME,
    INDEX_DB_NAME,
    INDEX_MANIFEST_NAME,
    PHYSICAL_CHANGE_REGISTRY_NAME,
    build_archive_index,
    get_indexed_evidence,
    search_archive_index,
    search_decision_receipts,
    search_physical_changes,
    stable_evidence_uid,
    stable_event_uid,
)
from visioncortex.schemas import (
    ActionCandidate,
    ActionType,
    AlignmentTransform,
    EvidenceEvent,
    ExperimentGroup,
    VideoInfo,
    ViewRole,
)


def _indexed_archive(
    root: Path,
    *,
    event_count: int = 1,
    stable_group_uids: bool = False,
):
    archive_id = root.name
    transforms = {
        view_id: AlignmentTransform(
            view_id=view_id,
            reference_view_id="fp",
            state="aligned",
            confidence=1.0,
        )
        for view_id in ("fp", "tp")
    }
    events = []
    groups = []
    normalized = []
    for ordinal in range(1, event_count + 1):
        event_id = f"EVT-{ordinal:06d}"
        group_id = f"GROUP-{ordinal:04d}"
        candidate = ActionCandidate(
            candidate_id=f"CAND-fp-{ordinal:06d}",
            action_type=ActionType.LIQUID_MOVEMENT,
            view_id="fp",
            role=ViewRole.FIRST_PERSON,
            local_start_ms=12_000 + ordinal * 100,
            local_end_ms=13_000 + ordinal * 100,
            global_start_ms=12_000 + ordinal * 100,
            global_end_ms=13_000 + ordinal * 100,
            key_global_ms=12_500 + ordinal * 100,
            objects=["pipette", "tube"],
            confidence=0.9,
            evidence=[{"frame_index": 360 + ordinal, "motion_score": 0.8}],
        )
        event = EvidenceEvent(
            event_id=event_id,
            action_type=ActionType.LIQUID_MOVEMENT,
            global_start_ms=12_000 + ordinal * 100,
            global_end_ms=13_000 + ordinal * 100,
            key_global_ms=12_500 + ordinal * 100,
            objects=["pipette", "tube"],
            confidence=0.91,
            accepted=True,
            audit_reason="dual-view evidence",
            supporting_views=["fp", "tp"],
            supporting_roles=[ViewRole.FIRST_PERSON, ViewRole.THIRD_PERSON],
            candidates=[candidate],
            model_understanding={
                "status": "completed",
                "model": "doubao-seed-2.1-pro",
                "current_step": f"第{ordinal}步：从源容器吸取液体",
                "next_step": "移动到离心管并释放液体",
                "action_type_confirmed": "liquid_movement",
                "objects": ["pipette", "tube"],
                "cross_view_consistency": "consistent",
                "confidence": 0.9,
                "physical_change": {
                    "before": "outside_source",
                    "after": "withdrawn_from_target",
                },
            },
        )
        group = ExperimentGroup(
            group_id=group_id,
            group_uid=(f"GRP-stable-{ordinal:04d}" if stable_group_uids else None),
            continuity_type="independent",
            atomic_experiment_ids=[f"EXP-{ordinal:04d}"],
            global_start_ms=10_000,
            global_end_ms=15_000,
            participating_views=["fp", "tp"],
            first_person_view="fp",
            third_person_view="tp",
            continuity_reason="independent",
            experiment_name="移液实验",
            experiment_name_en="Pipetting-Experiment",
            archive_folder=f"{ordinal:03d}_Pipetting-Experiment",
            key_event_ids=[event_id],
        )
        action_folder = "03-Liquid-Movement"
        frame_root = (
            root
            / "Key-Materials"
            / "Key-Frames"
            / group.archive_folder
            / action_folder
            / event_id
        )
        clip_root = (
            root
            / "Key-Materials"
            / "Key-Clips"
            / group.archive_folder
            / action_folder
            / event_id
        )
        for collection, media_root, suffix in (
            (event.key_frames, frame_root, ".jpg"),
            (event.key_clips, clip_root, ".mp4"),
        ):
            for view_id in ("fp", "tp", "aligned_first_third"):
                media = media_root / f"{view_id}{suffix}"
                media.parent.mkdir(parents=True, exist_ok=True)
                media.write_bytes(f"{event_id}:{view_id}:{suffix}".encode())
                media.with_suffix(".json").write_text(
                    json.dumps({"event_id": event_id}), encoding="utf-8"
                )
                collection[view_id] = media.relative_to(root).as_posix()
        payload = _artifact_json(
            group,
            event,
            "key_material_event_index",
            "",
            None,
            transforms,
            archive_id,
        )
        events.append(event)
        groups.append(group)
        normalized.append(payload)
    source = root / "source-fp.mp4"
    infos = {
        "fp": VideoInfo(
            path=source,
            duration_ms=60_000,
            fps=30,
            width=1920,
            height=1080,
            frame_count=1800,
        )
    }
    manifest = build_archive_index(
        root, archive_id, normalized, events, groups, infos, hash_workers=2
    )
    return archive_id, events, groups, normalized, manifest


def test_index_uid_uses_stable_group_uid_but_keeps_display_group_id(tmp_path):
    root = tmp_path / "Archive-Stable-UID"
    archive_id, events, groups, normalized, manifest = _indexed_archive(
        root, stable_group_uids=True
    )

    payload = normalized[0]
    expected = stable_event_uid(
        archive_id, groups[0].group_uid, events[0].event_id
    )
    assert payload["parent_event_id"] == groups[0].group_id
    assert payload["parent_event_uid"] == groups[0].group_uid
    assert payload["provenance"]["index"]["event_uid"] == expected
    assert manifest["schema_version"] == "visioncortex-evidence-index/4"


def test_build_archive_index_preserves_one_hop_artifact_and_source_references(tmp_path):
    root = tmp_path / "Archive-001"
    archive_id, events, groups, normalized, manifest = _indexed_archive(root)

    json_root = root / "JSON-Config-Files"
    assert (json_root / INDEX_DB_NAME).is_file()
    assert (json_root / INDEX_MANIFEST_NAME).is_file()
    assert manifest["counts"]["key_events"] == 1
    assert manifest["counts"]["artifacts"] == 6
    assert manifest["counts"]["evidence"] == 2
    assert manifest["counts"]["artifact_hashes_computed"] == 6
    assert manifest["counts"]["physical_changes"] == 1
    assert (json_root / PHYSICAL_CHANGE_REGISTRY_NAME).is_file()
    event_uid = stable_event_uid(archive_id, groups[0].group_id, events[0].event_id)
    assert normalized[0]["provenance"]["index"]["event_uid"] == event_uid
    assert normalized[0]["provenance"]["index"]["evidence_refs"][0][
        "evidence_uid"
    ].startswith(f"{event_uid}:evidence:")

    artifacts = [
        json.loads(line)
        for line in (json_root / ARTIFACT_REGISTRY_NAME).read_text(encoding="utf-8").splitlines()
    ]
    assert len(artifacts) == 6
    first_artifact = artifacts[0]
    media = root / first_artifact["path"]
    assert first_artifact["sha256"] == hashlib.sha256(media.read_bytes()).hexdigest()
    assert (root / first_artifact["sidecar_path"]).is_file()
    assert first_artifact["integrity_status"] == "verified"

    evidence = [
        json.loads(line)
        for line in (json_root / EVIDENCE_REGISTRY_NAME).read_text(encoding="utf-8").splitlines()
    ]
    frame = next(item for item in evidence if item["evidence_kind"] == "frame_evidence")
    assert frame["source"]["source_segment_path"].endswith("source-fp.mp4")
    assert frame["json_references"][0]["path"].endswith("evidence_package.json")
    assert frame["json_references"][0]["json_pointer"].startswith("/events/0/candidates/0")
    assert get_indexed_evidence(root, frame["evidence_uid"])["event_uid"] == event_uid

    rebuilt = build_archive_index(
        root,
        archive_id,
        normalized,
        events,
        groups,
        {
            "fp": VideoInfo(
                path=root / "source-fp.mp4",
                duration_ms=60_000,
                fps=30,
                width=1920,
                height=1080,
                frame_count=1800,
            )
        },
        hash_workers=2,
    )
    assert rebuilt["counts"]["artifact_hashes_reused"] == 6
    assert rebuilt["counts"]["artifact_hashes_computed"] == 0


def test_archive_sqlite_is_built_outside_archive_before_publish(tmp_path, monkeypatch):
    archive_root = tmp_path / "Archive-Local-Scratch"
    connected_paths = []
    original_connect = indexing.sqlite3.connect

    def record_connect(path, *args, **kwargs):
        connected_paths.append(Path(path).resolve())
        return original_connect(path, *args, **kwargs)

    monkeypatch.setattr(indexing.sqlite3, "connect", record_connect)

    _indexed_archive(archive_root)

    assert len(connected_paths) == 1
    assert archive_root.resolve() not in connected_paths[0].parents
    assert (archive_root / "JSON-Config-Files" / INDEX_DB_NAME).is_file()


def test_archive_index_registers_key_material_recall_receipt(tmp_path):
    root = tmp_path / "Archive-Recall"
    receipt = root / "JSON-Config-Files" / "key_material_recall_eval.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text(
        json.dumps({"status": "not_evaluated", "evaluated": False}),
        encoding="utf-8",
    )

    _, _, _, _, manifest = _indexed_archive(root)

    assert manifest["files"]["key_material_recall_eval"] == (
        "key_material_recall_eval.json"
    )
    integrity = manifest["integrity"]["key_material_recall_eval.json"]
    assert integrity["size_bytes"] == receipt.stat().st_size
    assert integrity["sha256"] == hashlib.sha256(receipt.read_bytes()).hexdigest()


def test_archive_detail_exposes_key_material_recall_receipt(
    monkeypatch, tmp_path
):
    archive_root = tmp_path / "archives"
    root = archive_root / "Archive-Recall-Web"
    _indexed_archive(root)
    receipt = root / "JSON-Config-Files" / "key_material_recall_eval.json"
    receipt.write_text(
        json.dumps(
            {
                "status": "evaluated",
                "evaluated": True,
                "threshold_results": [
                    {
                        "temporal_iou_threshold": 0.5,
                        "precision": 0.5,
                        "recall": 0.4,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(api, "_archive_root", lambda settings=None: archive_root)

    payload = TestClient(api.app).get(
        f"/api/archives/{root.name}"
    ).json()

    assert payload["key_material_recall_eval"]["evaluated"] is True
    assert payload["links"]["key_material_recall_eval"].endswith(
        "key_material_recall_eval.json"
    )


def test_search_archive_index_filters_full_text_and_returns_material_hashes(
    tmp_path, monkeypatch
):
    root = tmp_path / "Archive-Search"
    _, _, _, _, _ = _indexed_archive(root, event_count=2)
    statements = []
    original_connect = indexing.sqlite3.connect

    def traced_connect(*args, **kwargs):
        connection = original_connect(*args, **kwargs)
        connection.set_trace_callback(statements.append)
        return connection

    monkeypatch.setattr(indexing.sqlite3, "connect", traced_connect)

    results = search_archive_index(
        root,
        query="吸取液体",
        action_type="liquid_transfer",
        cross_view=True,
        limit=10,
    )

    assert len(results) == 2
    assert all(item["event_uid"] for item in results)
    assert all(len(item["artifact_references"]) == 6 for item in results)
    assert all(item["artifact_references"][0]["sha256"] for item in results)
    artifact_queries = [
        statement for statement in statements if "FROM artifacts" in statement
    ]
    assert len(artifact_queries) == 1
    assert " IN (" in artifact_queries[0]


def test_quality_decision_receipts_are_indexed_by_rule_verdict_and_subject(tmp_path):
    root = tmp_path / "Archive-Decisions"
    archive_id, events, groups, normalized, _ = _indexed_archive(root)
    receipt = decision_receipt(
        decision_type="experiment_continuity_edge",
        rule_id="QF2-STABLE-OBJECT-IDENTITY",
        verdict="rejected",
        subject_ids=["EXP-LEFT", "EXP-RIGHT"],
        reason_codes=["class_overlap_without_stable_identity"],
        facts={"shared_object_labels": ["tube", "tube_rack"]},
    )
    json_root = root / "JSON-Config-Files"
    (json_root / "audit_layer.json").write_text(
        json.dumps(
            {"quality_decision_receipts": [receipt]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    infos = {
        "fp": VideoInfo(
            path=root / "source-fp.mp4",
            duration_ms=60_000,
            fps=30,
            width=1920,
            height=1080,
            frame_count=1800,
        )
    }

    manifest = build_archive_index(
        root, archive_id, normalized, events, groups, infos
    )
    results = search_decision_receipts(
        root,
        rule_id="QF2-STABLE-OBJECT-IDENTITY",
        verdict="rejected",
        subject_id="EXP-RIGHT",
    )

    assert manifest["counts"]["decision_receipts"] == 1
    assert (json_root / DECISION_REGISTRY_NAME).is_file()
    assert [item["decision_id"] for item in results] == [receipt["decision_id"]]
    assert results[0]["json_references"][0]["path"].endswith(
        "audit_layer.json"
    )


def test_physical_change_index_only_projects_explicit_known_state_differences(tmp_path):
    root = tmp_path / "Archive-Changes"
    _, _, _, normalized, _ = _indexed_archive(root, event_count=2)

    changes = search_physical_changes(
        root,
        object_role="tool",
        action_type="liquid_transfer",
        limit=10,
    )

    assert len(changes) == 2
    assert {item["state_before"] for item in changes} == {"outside_source"}
    assert {item["state_after"] for item in changes} == {"withdrawn_from_target"}
    assert all(item["provenance"]["new_inference_performed"] is False for item in changes)
    assert all(item["event_uid"] for item in changes)
    assert normalized[0]["state_before"]["source"] == "unknown"
    assert not search_physical_changes(root, object_role="source", limit=10)


def test_key_event_api_paginates_and_resolves_event_and_evidence(monkeypatch, tmp_path):
    archive_root = tmp_path / "archives"
    root = archive_root / "Archive-Web"
    archive_id, events, groups, _, _ = _indexed_archive(root, event_count=2)
    monkeypatch.setattr(api, "_archive_root", lambda settings=None: archive_root)
    client = TestClient(api.app)

    first_page = client.get(
        "/api/key-events",
        params={"archive": root.name, "q": "吸取液体", "limit": 1},
    )
    assert first_page.status_code == 200
    first_payload = first_page.json()
    assert first_payload["count"] == 1
    assert first_payload["next_cursor"]
    assert first_payload["items"][0]["artifact_references"][0]["url"]

    second_page = client.get(
        "/api/key-events",
        params={
            "archive": root.name,
            "q": "吸取液体",
            "limit": 1,
            "cursor": first_payload["next_cursor"],
        },
    )
    assert second_page.status_code == 200
    assert second_page.json()["items"][0]["event_uid"] != first_payload["items"][0]["event_uid"]

    mismatched_cursor = client.get(
        "/api/key-events",
        params={
            "archive": root.name,
            "q": "different-filter",
            "limit": 1,
            "cursor": first_payload["next_cursor"],
        },
    )
    assert mismatched_cursor.status_code == 400

    first_change_page = client.get(
        "/api/physical-changes",
        params={"archive": root.name, "object_role": "tool", "limit": 1},
    )
    assert first_change_page.status_code == 200
    change_payload = first_change_page.json()
    assert change_payload["count"] == 1
    assert change_payload["next_cursor"]
    assert change_payload["items"][0]["event_url"].startswith("/api/key-events/")

    second_change_page = client.get(
        "/api/physical-changes",
        params={
            "archive": root.name,
            "object_role": "tool",
            "limit": 1,
            "cursor": change_payload["next_cursor"],
        },
    )
    assert second_change_page.status_code == 200
    assert second_change_page.json()["items"][0]["change_uid"] != change_payload["items"][0]["change_uid"]

    event_uid = stable_event_uid(archive_id, groups[0].group_id, events[0].event_id)
    detail = client.get(
        f"/api/key-events/{event_uid}", params={"archive": root.name}
    )
    assert detail.status_code == 200
    assert detail.json()["event_uid"] == event_uid

    evidence_uid = stable_evidence_uid(event_uid, events[0].candidates[0].candidate_id)
    evidence = client.get(
        f"/api/evidence/{evidence_uid}", params={"archive": root.name}
    )
    assert evidence.status_code == 200
    assert evidence.json()["event_url"].startswith("/api/key-events/")


def test_archive_catalog_paginates_and_global_search_uses_one_read_model(
    monkeypatch, tmp_path
):
    archive_root = tmp_path / "archives"
    first = archive_root / "Archive-Alpha"
    second = archive_root / "Archive-Beta"
    _indexed_archive(first, event_count=2)
    _indexed_archive(second, event_count=1)
    catalog = tmp_path / "runtime" / "archive-catalog.sqlite3"
    monkeypatch.setattr(api, "_archive_root", lambda settings=None: archive_root)
    monkeypatch.setattr(
        api, "_archive_catalog_database", lambda settings=None, archive_root=None: catalog
    )
    client = TestClient(api.app)

    first_page = client.get("/api/archives", params={"limit": 1}).json()
    assert first_page["count"] == 1
    assert first_page["total_count"] == 2
    assert first_page["totals"]["key_events"] == 3
    assert first_page["next_cursor"]

    second_page = client.get(
        "/api/archives",
        params={"limit": 1, "cursor": first_page["next_cursor"]},
    ).json()
    assert second_page["archives"][0]["name"] != first_page["archives"][0]["name"]

    search = client.get(
        "/api/key-events", params={"q": "移液", "limit": 10}
    ).json()
    assert search["count"] == 3
    assert search["total_count"] == 3
    assert {item["archive_name"] for item in search["items"]} == {
        "Archive-Alpha",
        "Archive-Beta",
    }
    assert all(item["aligned_frame_url"] for item in search["items"])
    literal_wildcard = client.get(
        "/api/key-events", params={"q": "%", "limit": 10}
    ).json()
    assert literal_wildcard["total_count"] == 0


def test_archive_summary_is_lightweight_and_experiment_section_is_paginated(
    monkeypatch, tmp_path
):
    archive_root = tmp_path / "archives"
    root = archive_root / "Archive-Sections"
    _, _, groups, _, _ = _indexed_archive(root, event_count=2)
    analysis = {
        "experiment_groups": [
            {
                **group.model_dump(mode="json"),
                "model_understanding": {
                    "overall_summary": f"summary-{index}",
                    "steps": [],
                },
            }
            for index, group in enumerate(groups, 1)
        ]
    }
    (root / "JSON-Config-Files" / "Experiment-Groups-Step-Level-Analysis.json").write_text(
        json.dumps(analysis, ensure_ascii=False), encoding="utf-8"
    )
    monkeypatch.setattr(api, "_archive_root", lambda settings=None: archive_root)
    client = TestClient(api.app)

    summary = client.get(
        f"/api/archives/{root.name}", params={"section": "summary"}
    ).json()
    assert summary["counts"] == {"experiments": 0, "key_events": 2}
    assert "key_events" not in summary
    assert "daily_report" not in summary

    first_page = client.get(
        f"/api/archives/{root.name}",
        params={"section": "experiments", "limit": 1},
    ).json()
    assert len(first_page["experiments"]) == 1
    assert first_page["next_cursor"]
    second_page = client.get(
        f"/api/archives/{root.name}",
        params={
            "section": "experiments",
            "limit": 1,
            "cursor": first_page["next_cursor"],
        },
    ).json()
    assert second_page["experiments"][0]["name"] == "移液实验"

    (root / "JSON-Config-Files" / "evidence_package.json").write_text(
        json.dumps(analysis, ensure_ascii=False), encoding="utf-8"
    )
    (root / "JSON-Config-Files" / "Experiment-Groups-Step-Level-Analysis.json").unlink()
    legacy_page = client.get(
        f"/api/archives/{root.name}",
        params={"section": "experiments", "limit": 1},
    ).json()
    assert legacy_page["experiments"][0]["name"] == "移液实验"


def test_catalog_refreshes_pipeline_state_without_rebuilding_event_index(monkeypatch, tmp_path):
    archive_root = tmp_path / "archives"
    root = archive_root / "Archive-State"
    _, _, _, normalized, _ = _indexed_archive(root)
    (root / "Key-Materials/Key-Materials-Model-Understanding.json").write_text(
        json.dumps(normalized), encoding="utf-8"
    )
    status_path = root / "JSON-Config-Files/pipeline_status.json"
    status_path.write_text('{"stage":"candidate_fine"}', encoding="utf-8")
    catalog = tmp_path / "runtime/catalog.sqlite3"
    monkeypatch.setattr(api, "_archive_root", lambda settings=None: archive_root)
    monkeypatch.setattr(api, "_archive_catalog_database", lambda **_kwargs: catalog)
    client = TestClient(api.app)
    assert client.get("/api/archives").json()["archives"][0]["pipeline_stage"] == "candidate_fine"
    status_path.write_text('{"stage":"failed","failed_stage":"mllm"}', encoding="utf-8")
    assert client.get("/api/archives").json()["archives"][0]["pipeline_stage"] == "failed"
    detail = client.get(f"/api/archives/{root.name}?section=summary").json()
    assert detail["observability"]["status"]["failed_stage"] == "mllm"
    assert len(detail["key_events"]) == 1


def test_paginated_sections_keep_role_videos_and_professional_report(monkeypatch, tmp_path):
    archive_root = tmp_path / "archives"
    root = archive_root / "Archive-Sections-Roles"
    _, _, groups, _, _ = _indexed_archive(root)
    folder = root / "Experiment-Clips" / groups[0].archive_folder
    folder.mkdir(parents=True)
    for name in ("First-Person.mp4", "Third-Person.mp4", "Aligned_First+Third.mp4"):
        (folder / name).write_bytes(b"structural-media-fixture")
    json_root = root / "JSON-Config-Files"
    (json_root / "Experiment-Groups-Step-Level-Analysis.json").write_text(
        json.dumps({"experiment_groups": [group.model_dump(mode="json") for group in groups]})
    )
    (json_root / "pipeline_status.json").write_text('{"stage":"completed"}')
    (json_root / "daily_report_manifest.json").write_text('{"json":"report.json"}')
    (root / "report.json").write_text('{"report_id":"R1"}')
    monkeypatch.setattr(api, "_archive_root", lambda settings=None: archive_root)
    client = TestClient(api.app)
    detail = client.get(f"/api/archives/{root.name}?section=experiments").json()
    experiment = detail["experiments"][0]
    assert all(experiment[field] for field in (
        "first_person_video_url", "third_person_video_url", "aligned_video_url"
    ))
    metrics = client.get(f"/api/archives/{root.name}?section=metrics").json()
    assert metrics["daily_report"]["report_id"] == "R1"
    assert metrics["observability"]["status"]["stage"] == "completed"


def test_versioned_archive_file_rejects_stale_or_size_mismatched_content(
    monkeypatch, tmp_path
):
    archive_root = tmp_path / "archives"
    root = archive_root / "Archive-Media"
    json_root = root / "JSON-Config-Files"
    json_root.mkdir(parents=True)
    (json_root / "evidence_package.json").write_text("{}", encoding="utf-8")
    report = root / "Lab-Daily-Reports" / "report.txt"
    report.parent.mkdir(parents=True)
    report.write_text("verified report", encoding="utf-8")
    release_manifest = root / ".release-manifest.json"
    release_payload = {
        "release_id": "release-001",
        "manifests": {
            "Lab-Daily-Reports": [
                {
                    "path": "report.txt",
                    "size_bytes": report.stat().st_size,
                    "sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
                }
            ]
        },
    }
    release_manifest.write_text(json.dumps(release_payload), encoding="utf-8")
    pointer = {
        "release_id": "release-001",
        "release_manifest": release_manifest.name,
        "release_manifest_file_sha256": hashlib.sha256(
            release_manifest.read_bytes()
        ).hexdigest(),
    }
    (root / ".VisionCortex-Current-Release.json").write_text(
        json.dumps(pointer), encoding="utf-8"
    )
    monkeypatch.setattr(api, "_archive_root", lambda settings=None: archive_root)
    client = TestClient(api.app)

    current = client.get(
        "/api/archive-file",
        params={
            "archive": root.name,
            "path": "Lab-Daily-Reports/report.txt",
            "release": "release-001",
        },
    )
    assert current.status_code == 200
    assert current.headers["cache-control"].endswith("immutable")
    assert current.headers["x-visioncortex-release"] == "release-001"
    assert current.headers["etag"].startswith('"sha256-')

    stale = client.get(
        "/api/archive-file",
        params={
            "archive": root.name,
            "path": "Lab-Daily-Reports/report.txt",
            "release": "release-old",
        },
    )
    assert stale.status_code == 409

    report.write_text("tampered report is larger", encoding="utf-8")
    mismatched = client.get(
        "/api/archive-file",
        params={
            "archive": root.name,
            "path": "Lab-Daily-Reports/report.txt",
            "release": "release-001",
        },
    )
    assert mismatched.status_code == 409


def test_archive_stream_limit_returns_retryable_busy_response(monkeypatch):
    semaphore = threading.BoundedSemaphore(1)
    assert semaphore.acquire(blocking=False)
    monkeypatch.setattr(api, "_archive_stream_slots", semaphore)

    response = TestClient(api.app).get(
        "/api/archive-file", params={"archive": "unused", "path": "unused"}
    )

    assert response.status_code == 429
    assert response.headers["retry-after"] == "2"
    semaphore.release()


def test_staging_run_web_endpoints_are_indexable_and_fail_closed(
    monkeypatch, tmp_path
):
    archive_root = tmp_path / "archives"
    run_id = "staging-20260822-000001-abc123"
    root = archive_root / ".VisionCortex-Run-Staging" / "Audit-Campaign" / run_id
    archive_id, events, groups, _, _ = _indexed_archive(root, event_count=2)
    json_root = root / "JSON-Config-Files"
    (json_root / "pipeline_status.json").write_text(
        json.dumps({"stage": "completed", "progress": 1.0}),
        encoding="utf-8",
    )
    retained = root / "Web-Retention" / "report.html"
    retained.parent.mkdir(parents=True)
    retained.write_text("<h1>staging report</h1>", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("must not be served", encoding="utf-8")
    (root / "outside-link.txt").symlink_to(outside)
    monkeypatch.setattr(
        api,
        "_settings",
        lambda: {"storage": {"archive_root": str(archive_root)}},
    )
    client = TestClient(api.app)

    first_page = client.get(
        f"/api/staging-runs/{run_id}/key-events",
        params={"q": "吸取液体", "limit": 1},
    )
    assert first_page.status_code == 200
    first_payload = first_page.json()
    assert first_payload["count"] == 1
    assert first_payload["next_cursor"]
    assert first_payload["formal_archive_promotion"] is False
    assert first_payload["items"][0]["artifact_references"][0]["url"].startswith(
        "/api/staging-file?"
    )

    second_page = client.get(
        f"/api/staging-runs/{run_id}/key-events",
        params={
            "q": "吸取液体",
            "limit": 1,
            "cursor": first_payload["next_cursor"],
        },
    )
    assert second_page.status_code == 200
    assert (
        second_page.json()["items"][0]["event_uid"]
        != first_payload["items"][0]["event_uid"]
    )

    event_uid = stable_event_uid(archive_id, groups[0].group_id, events[0].event_id)
    detail = client.get(
        f"/api/staging-runs/{run_id}/key-events/{event_uid}"
    )
    assert detail.status_code == 200
    assert detail.json()["event_uid"] == event_uid

    report = client.get(
        "/api/staging-file",
        params={"run_id": run_id, "path": "Web-Retention/report.html"},
    )
    assert report.status_code == 200
    assert "staging report" in report.text
    assert client.get(
        "/api/staging-file",
        params={"run_id": run_id, "path": "../../outside.txt"},
    ).status_code == 404
    assert client.get(
        "/api/staging-file",
        params={"run_id": run_id, "path": "outside-link.txt"},
    ).status_code == 404


def test_material_link_lookup_finds_exact_event_beyond_first_page(monkeypatch, tmp_path):
    archive_root = tmp_path / "archives"
    root = archive_root / "Archive-Material-Link"
    _, events, groups, _, _ = _indexed_archive(root, event_count=30)
    monkeypatch.setattr(api, "_archive_root", lambda settings=None: archive_root)
    monkeypatch.setattr(
        api, "_archive_catalog_database", lambda **_kwargs: tmp_path / "catalog.sqlite3"
    )
    client = TestClient(api.app)
    first = client.get(
        "/api/key-events", params={"archive": root.name, "material_ready": True, "limit": 24}
    ).json()
    target = events[-1]
    assert target.event_id not in {item["event_id"] for item in first["items"]}
    response = client.get(
        "/api/key-events",
        params={
            "archive": root.name, "q": target.event_id,
            "parent_event_id": groups[-1].group_id, "material_ready": True, "limit": 24,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["total_count"] == 1
    assert payload["items"][0]["event_id"] == target.event_id
    assert payload["items"][0]["event_uid"]
    assert payload["items"][0]["dual_view_material_ready"] is True
    assert payload["items"][0]["archive_name"] == root.name
    assert "release_id" in payload["items"][0]
    assert payload["next_cursor"] is None
