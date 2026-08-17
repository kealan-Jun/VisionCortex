import hashlib
import json
from pathlib import Path

from fastapi.testclient import TestClient

from labvision_evidence import api
from labvision_evidence.archive import _artifact_json
from labvision_evidence.indexing import (
    ARTIFACT_REGISTRY_NAME,
    EVIDENCE_REGISTRY_NAME,
    INDEX_DB_NAME,
    INDEX_MANIFEST_NAME,
    build_archive_index,
    get_indexed_evidence,
    search_archive_index,
    stable_evidence_uid,
    stable_event_uid,
)
from labvision_evidence.schemas import (
    ActionCandidate,
    ActionType,
    AlignmentTransform,
    EvidenceEvent,
    ExperimentGroup,
    VideoInfo,
    ViewRole,
)


def _indexed_archive(root: Path, *, event_count: int = 1):
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
            },
        )
        group = ExperimentGroup(
            group_id=group_id,
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
        frame_root = root / "Key-Materials" / "Key-Frames" / group.archive_folder / event_id
        clip_root = root / "Key-Materials" / "Key-Clips" / group.archive_folder / event_id
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


def test_search_archive_index_filters_full_text_and_returns_material_hashes(tmp_path):
    root = tmp_path / "Archive-Search"
    _, _, _, _, _ = _indexed_archive(root, event_count=2)

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
