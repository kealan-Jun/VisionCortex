from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from visioncortex import api
from visioncortex.annotation_workspace import (
    export_reviewed_ground_truth,
    load_annotation_workspace,
    record_annotation_decision,
)


def _settings(tmp_path: Path) -> tuple[dict, Path]:
    runtime = tmp_path / "runtime"
    queue = runtime / "Model-Audits" / "audit-001" / "YOLO-Annotation-Queue.json"
    image = queue.parent / "Badcase-Images" / "event-001.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"fake-jpeg")
    queue.write_text(
        json.dumps(
            {
                "schema_version": "visioncortex-yolo-annotation-queue/1.0.0",
                "priority_counts": {"P0": 1},
                "items": [
                    {
                        "event_id": "EVT-001",
                        "role": "first_person",
                        "class_name": "paper",
                        "action_type": "hand_object_contact",
                        "issue_type": "persistent_miss",
                        "representative_phase_index": "2",
                        "priority": "P0",
                        "image_path": str(image),
                        "review_status": "pending",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return {
        "storage": {"local_runtime_root": str(runtime)},
        "developer_tools": {"yolo_annotation_workspace_enabled": True},
    }, image


def test_workspace_discovers_latest_queue_and_persists_review(tmp_path: Path):
    settings, _ = _settings(tmp_path)
    workspace = load_annotation_workspace(settings)
    item_id = workspace["items"][0]["item_id"]

    decision = record_annotation_decision(
        settings,
        item_id,
        {
            "decision": "confirmed_false_negative",
            "corrected_class": "paper",
            "xyxy": [1, 2, 30, 40],
            "visibility": "partial",
            "occlusion": "hand",
            "reviewer": "reviewer-01",
            "notes": "visible edge",
        },
    )
    refreshed = load_annotation_workspace(settings)
    exported = export_reviewed_ground_truth(settings)

    assert decision["decision"] == "confirmed_false_negative"
    assert refreshed["summary"]["reviewed"] == 1
    assert refreshed["items"][0]["effective_review_status"] == "reviewed"
    assert exported["annotation_count"] == 1
    assert exported["annotations"][0]["xyxy"] == [1.0, 2.0, 30.0, 40.0]


def test_workspace_rejects_fabricated_or_invalid_boxes(tmp_path: Path):
    settings, _ = _settings(tmp_path)
    item_id = load_annotation_workspace(settings)["items"][0]["item_id"]

    try:
        record_annotation_decision(
            settings,
            item_id,
            {
                "decision": "confirmed_false_negative",
                "xyxy": [30, 2, 1, 40],
                "reviewer": "reviewer-01",
            },
        )
    except ValueError as exc:
        assert "positive-area" in str(exc)
    else:
        raise AssertionError("invalid box was accepted")


def test_annotation_workspace_api(monkeypatch, tmp_path: Path):
    settings, image = _settings(tmp_path)
    monkeypatch.setattr(api, "_settings", lambda: settings)
    client = TestClient(api.app)

    listing = client.get("/api/annotation-workspace")
    assert listing.status_code == 200
    item_id = listing.json()["items"][0]["item_id"]
    image_response = client.get(f"/api/annotation-workspace/items/{item_id}/image")
    save = client.post(
        f"/api/annotation-workspace/items/{item_id}/decision",
        json={
            "decision": "class_error",
            "corrected_class": "weighing_paper",
            "xyxy": [1, 2, 30, 40],
            "reviewer": "reviewer-02",
        },
    )
    exported = client.get("/api/annotation-workspace/export")

    assert image_response.status_code == 200
    assert image_response.content == image.read_bytes()
    assert save.status_code == 200
    assert exported.json()["annotation_count"] == 1


def test_annotation_workspace_is_hidden_from_normal_product(monkeypatch, tmp_path: Path):
    settings, _ = _settings(tmp_path)
    settings["developer_tools"]["yolo_annotation_workspace_enabled"] = False
    monkeypatch.setattr(api, "_settings", lambda: settings)

    response = TestClient(api.app).get("/api/annotation-workspace")

    assert response.status_code == 404
    assert "内部 YOLO 标注工具未启用" in response.json()["detail"]
