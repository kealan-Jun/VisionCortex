from copy import deepcopy
import json

import cv2
import numpy as np
import pytest

from visioncortex import archive
from visioncortex import participant_visual_review as review_module
from visioncortex.config import load_config
from visioncortex.pipeline import EvidencePipeline
from visioncortex.schemas import ActionType, EvidenceEvent, ExperimentGroup, ViewRole
from visioncortex.validation import validate_experiment_and_material_quality


def _event():
    return EvidenceEvent(
        event_id="event", action_type=ActionType.HAND_OBJECT_CONTACT,
        global_start_ms=0, global_end_ms=1000, key_global_ms=500,
        confidence=0.9, objects=["gloved_hand", "paper"], accepted=True,
        audit_reason="test fixture", candidates=[],
        supporting_views=["fp", "tp"],
        supporting_roles=[ViewRole.FIRST_PERSON, ViewRole.THIRD_PERSON],
        model_understanding={"current_step": "hold a sheet"},
    )


@pytest.fixture
def setup(tmp_path, monkeypatch):
    config = load_config()
    config["storage"]["local_cache_root"] = str(tmp_path / "cache")
    config["models"]["open_vocabulary_key_frame"] = {"grounding_dino_fallback": {}}
    config["mllm"].update({"enabled": True, "api_key_env": "TEST_PARTICIPANT_REVIEW_KEY"})
    config["key_materials"]["participant_visual_review"] = {"enabled": True, "max_calls_per_run": 1}
    monkeypatch.setenv("TEST_PARTICIPANT_REVIEW_KEY", "unit-test-placeholder")
    calls = []

    class FakeAnalyzer:
        def __init__(self, api_config):
            assert api_config["mllm"]["max_retries"] == 1

        def close(self):
            pass

        def _call(self, prompt, metadata, images, *, max_images):
            assert len(images) <= 4
            calls.append(deepcopy(metadata))
            return {
                "status": "completed", "usage": {"total_tokens": 120, "input_tokens": 100, "output_tokens": 20},
                "provider": "aliyun", "api_protocol": "chat_completions",
                "request_id": "synthetic-review-request", "response_model": "synthetic-vision-model",
                "views": [{"view_id": view["view_id"], "selected_candidate_ids": view["candidate_ids"][:1], "target_visible": True, "reason": "visible in the candidate"} for view in metadata["views"]],
            }

    monkeypatch.setattr(review_module, "ArkAnalyzer", FakeAnalyzer)
    views = []
    for view_id, role in [("fp", "First-Person"), ("tp", "Third-Person")]:
        path = tmp_path / f"{role}.jpg"
        cv2.imwrite(str(path), np.full((200, 300, 3), 120, np.uint8))
        views.append({"view_id": view_id, "role_label": role, "raw_path": path, "detections": [{"class_name": "gloved_hand", "confidence": 0.9, "xyxy_norm": [0.2,0.2,0.4,0.5]}]})

    def detector(frame, classes, settings):
        return [{"class_name": "paper", "confidence": 0.8, "xyxy_norm": [0.3,0.3,0.5,0.5]}], {"status": "executed"}

    reviewer = review_module.ParticipantVisualReviewer(config, tmp_path / "work", tmp_path / "output", detector)
    return reviewer, config, views, calls


def test_completed_visual_request_reused_with_original_candidate_only(setup, tmp_path):
    reviewer, config, views, calls = setup
    event = _event()
    record, plan = reviewer.review(event, views)
    assert len(calls) == 1
    assert all(len(view["boxes"]) == 1 for view in plan.values())
    assert record["cache_reused"] is False
    new = review_module.ParticipantVisualReviewer(config, tmp_path / "next-work", tmp_path / "next-output", reviewer.detector)
    reused, _ = new.review(_event(), views)
    assert len(calls) == 1
    assert reused["cache_reused"] is True
    # Re-entry in the same run must retain its original spending in the ledger.
    same, _ = reviewer.review(event, views)
    assert same["cache_reused"] is False
    metrics = EvidencePipeline(config)._metrics([event, event])
    assert metrics["tokens"]["participant_visual_review"]["executed_call_count"] == 1
    assert metrics["tokens"]["run_total"]["total_tokens"] == 120
    call = metrics["mllm_calls"][0]
    assert call["provider"] == "aliyun"
    assert call["request_id"] == "synthetic-review-request"
    assert call["response_model"] == "synthetic-vision-model"


def test_changed_thinking_policy_cannot_reuse_visual_answer(setup, tmp_path):
    reviewer, config, views, calls = setup
    first, _ = reviewer.review(_event(), views)
    config["mllm"]["quality_mode"] = "quality"
    new = review_module.ParticipantVisualReviewer(config, tmp_path / "next-work", tmp_path / "next-output", reviewer.detector)
    second, _ = new.review(_event(), views)
    assert len(calls) == 2
    assert second["input_fingerprint"] != first["input_fingerprint"]
    assert second["cache_reused"] is False


def test_changed_input_respects_persistent_run_budget(setup):
    reviewer, config, views, calls = setup
    reviewer.review(_event(), views)
    cv2.imwrite(str(views[0]["raw_path"]), np.full((200,300,3), 220, np.uint8))
    resumed = review_module.ParticipantVisualReviewer(config, reviewer.work_root.parent, reviewer.index_path.parent, reviewer.detector)
    with pytest.raises(RuntimeError, match="budget_exhausted"):
        resumed.review(_event(), views)
    assert len(calls) == 1
    assert any(record["status"] == "budget_exhausted" for record in resumed.records.values())


@pytest.mark.parametrize("cached_status", ["pending", "failed", "invalid_response"])
def test_uncertain_or_failed_request_is_not_resubmitted(setup, cached_status):
    reviewer, config, views, calls = setup
    record, _ = reviewer.review(_event(), views)
    cache_path = reviewer.cache / "requests" / f"{record['input_fingerprint']}.json"
    saved = json.loads(cache_path.read_text())
    saved["status"] = cached_status
    saved["usage"] = {}
    cache_path.write_text(json.dumps(saved))
    event = _event()
    with pytest.raises(RuntimeError, match=cached_status):
        reviewer.review(event, views)
    assert len(calls) == 1
    usage = EvidencePipeline(config)._metrics([event])["tokens"]["participant_visual_review"]
    assert usage["unknown_usage_call_count"] == 1
    assert usage["total_tokens"] is None


@pytest.mark.parametrize("cached_status", ["pending", "failed", "invalid_response"])
def test_explicit_recovery_retries_failed_visual_request_once_with_new_budget(
    setup, cached_status,
):
    reviewer, config, views, calls = setup
    record, _ = reviewer.review(_event(), views)
    cache_path = reviewer.cache / "requests" / f"{record['input_fingerprint']}.json"
    saved = json.loads(cache_path.read_text())
    saved.update(status=cached_status, usage={})
    cache_path.write_text(json.dumps(saved))
    config["project"]["semantic_recovery_attempt"] = "original-run:2"
    recovered = review_module.ParticipantVisualReviewer(
        config, reviewer.work_root.parent, reviewer.index_path.parent, reviewer.detector
    )
    result, _ = recovered.review(_event(), views)
    assert result["status"] == "completed"
    assert len(calls) == 2
    recovered.review(_event(), views)
    assert len(calls) == 2
    history = list((reviewer.cache / "request-recovery").rglob("*.previous.json"))
    assert len(history) == 1
    assert json.loads(history[0].read_text())["status"] == cached_status
    assert list((reviewer.index_path.parent / "participant-visual-review-attempts").glob("*.json"))


def test_explicit_invisible_selection_is_safely_normalized_without_box(
    setup, monkeypatch
):
    reviewer, _, views, calls = setup

    class ContradictoryAnalyzer:
        def __init__(self, config):
            pass

        def close(self):
            pass

        def _call(self, prompt, metadata, images, **kwargs):
            calls.append(deepcopy(metadata))
            return {
                "status": "completed",
                "usage": {"total_tokens": 10},
                "views": [
                    {
                        "view_id": item["view_id"],
                        "selected_candidate_ids": item["candidate_ids"][:1],
                        "target_visible": False,
                        "reason": "no participant is visibly being operated",
                    }
                    for item in metadata["views"]
                ],
            }

    monkeypatch.setattr(review_module, "ArkAnalyzer", ContradictoryAnalyzer)
    record, plan = reviewer.review(_event(), views)

    assert record["status"] == "completed"
    assert len(record["response_normalizations"]) == 2
    assert all(row["selected_candidate_ids"] for row in record["raw_views"])
    assert all(not row["selected_candidate_ids"] for row in record["views"])
    assert all(not item["boxes"] and not item["target_visible"] for item in plan.values())


def test_cached_invalid_invisible_selection_is_recovered_without_new_call(
    setup, tmp_path
):
    reviewer, config, views, calls = setup
    record, _ = reviewer.review(_event(), views)
    cache_path = reviewer.cache / "requests" / f"{record['input_fingerprint']}.json"
    saved = json.loads(cache_path.read_text())
    saved["status"] = "invalid_response"
    saved["validation_error"] = "Inconsistent target visibility"
    for row in saved["views"]:
        row["target_visible"] = False
    cache_path.write_text(json.dumps(saved))

    resumed = review_module.ParticipantVisualReviewer(
        config, tmp_path / "recovered-work", tmp_path / "recovered-output", reviewer.detector
    )
    recovered, plan = resumed.review(_event(), views)

    assert len(calls) == 1
    assert recovered["status"] == "completed"
    assert recovered["cache_reused"] is True
    assert recovered["recovered_from_status"] == "invalid_response"
    assert all(not item["boxes"] and not item["target_visible"] for item in plan.values())


@pytest.mark.parametrize("mutation", ["unknown", "cross_view", "duplicate_view", "missing_view", "multiple", "invisible_selected"])
def test_invalid_selection_cannot_create_a_box(mutation):
    views = [{"view_id": name, "candidates": [{"candidate_id": name+"-01"}]} for name in ["a", "b"]]
    result = {"views": [{"view_id": name, "selected_candidate_ids": [name+"-01"], "target_visible": True, "reason": "visible"} for name in ["a", "b"]]}
    first = result["views"][0]
    if mutation == "unknown":
        first["selected_candidate_ids"] = ["invented"]
    if mutation == "cross_view":
        first["selected_candidate_ids"] = ["b-01"]
    if mutation == "duplicate_view":
        result["views"][1] = deepcopy(first)
    if mutation == "missing_view":
        result["views"].pop()
    if mutation == "multiple":
        first["selected_candidate_ids"] *= 2
    if mutation == "invisible_selected":
        first["target_visible"] = False
    with pytest.raises(ValueError):
        review_module.validate_selection(result, views)


@pytest.mark.parametrize("participant_class", ["paper", "bottle_cap"])
def test_empty_selection_never_restores_background_box(setup, tmp_path, monkeypatch, participant_class):
    reviewer, config, views, calls = setup
    config["key_materials"]["participant_visual_review"]["classes"] = [participant_class]
    config["models"]["temporal_participant_segmentation"] = {"enabled": False}
    config["models"]["liquid_semantic_sidecar"] = {"enabled": False}

    class EmptyAnalyzer:
        def __init__(self, config):
            pass

        def close(self):
            pass

        def _call(self, prompt, metadata, images, **kwargs):
            return {"status": "completed", "usage": {"total_tokens": 10}, "views": [{"view_id": item["view_id"], "selected_candidate_ids": [], "target_visible": False, "reason": "no visible paper"} for item in metadata["views"]]}

    monkeypatch.setattr(review_module, "ArkAnalyzer", EmptyAnalyzer)
    monkeypatch.setattr(archive, "_grounding_dino_key_frame_detections", reviewer.detector)
    # Any accidentally executed legacy supplement must also be unable to restore paper.
    monkeypatch.setattr(archive, "_open_vocabulary_key_frame_supplement", lambda *args, **kwargs: ([{"class_name":participant_class, "confidence":0.99, "xyxy_norm":[0.3,0.3,0.5,0.5]}], {"status":"executed", "closed_set_replaced_classes":[participant_class]}))
    event = _event()
    event.objects = ["gloved_hand", participant_class]
    monkeypatch.setattr(archive, "_grounding_dino_key_frame_detections", lambda *args: ([{"class_name":participant_class, "confidence":0.9, "xyxy_norm":[0.3,0.3,0.5,0.5]}], {"status":"executed"}))
    layout = archive.ArchiveLayout(tmp_path / "annotation")
    layout.create()
    raw = layout.work / "key-material-annotation-inputs" / event.event_id
    raw.mkdir(parents=True)
    for view in views:
        (raw / (view["role_label"]+".jpg")).write_bytes(view["raw_path"].read_bytes())
        (raw / (view["role_label"]+".json")).write_text(json.dumps({"detections": [*view["detections"], {"class_name":participant_class, "confidence":0.99, "xyxy_norm":[0.3,0.3,0.5,0.5]}]}))
        event.key_frames[view["view_id"]] = f"Key-Materials/Key-Frames/{view['view_id']}.jpg"
    event.key_frames["aligned_first_third"] = "Key-Materials/Key-Frames/aligned.jpg"
    group = ExperimentGroup(group_id="group", continuity_type="independent", atomic_experiment_ids=["atom"], global_start_ms=0, global_end_ms=1000, participating_views=["fp","tp"], first_person_view="fp", third_person_view="tp", continuity_reason="fixture", key_event_ids=[event.event_id])
    report = archive._rerender_curated_participant_annotations(layout, [event], [group], config)
    assert all(participant_class not in row["rendered_classes"] for row in report["records"])
    assert event.observability["key_material_annotation"]["participant_visual_review"]["localized_view_count"] == 0
    quality = validate_experiment_and_material_quality([group], [event], None, require_participant_only_annotations=True)
    assert quality["key_materials"]["participant_only_annotation_pass_count"] == 0


def test_irrelevant_participant_does_not_call_models(setup):
    reviewer, config, views, calls = setup
    event = _event()
    event.objects = ["hand", "reagent_bottle"]
    record, plan = reviewer.review(event, views)
    assert record["status"] == "not_applicable"
    assert plan == {} and calls == []


def test_balance_review_applies_only_to_device_panel_operations(setup):
    reviewer, config, views, calls = setup
    config["key_materials"]["participant_visual_review"]["classes"] = ["balance"]
    reviewer = review_module.ParticipantVisualReviewer(
        config, reviewer.work_root.parent / "balance-work",
        reviewer.index_path.parent / "balance-output", reviewer.detector,
    )
    event = _event()
    event.objects = ["gloved_hand", "balance"]
    assert reviewer.eligible_classes(event) == []
    event.action_type = ActionType.DEVICE_PANEL_OPERATION
    assert reviewer.eligible_classes(event) == ["balance"]


def test_balance_panel_review_uses_strict_device_contact_prompt(setup, monkeypatch):
    reviewer, config, views, calls = setup
    config["key_materials"]["participant_visual_review"]["classes"] = ["balance"]

    def detector(frame, classes, settings):
        name = next(iter(classes))
        return [{"class_name": name, "confidence": 0.8, "xyxy_norm": [0.2,0.1,0.8,0.9]}], {"status":"executed"}

    captured = []

    class EmptyAnalyzer:
        def __init__(self, api_config):
            pass

        def close(self):
            pass

        def _call(self, prompt, metadata, images, **kwargs):
            captured.append(prompt)
            return {
                "status": "completed", "usage": {"total_tokens": 10},
                "views": [
                    {"view_id": item["view_id"], "selected_candidate_ids": [],
                     "target_visible": False, "reason": "hand is on an adjacent device"}
                    for item in metadata["views"]
                ],
            }

    monkeypatch.setattr(review_module, "ArkAnalyzer", EmptyAnalyzer)
    reviewer = review_module.ParticipantVisualReviewer(
        config, reviewer.work_root.parent / "strict-balance-work",
        reviewer.index_path.parent / "strict-balance-output", detector,
    )
    event = _event()
    event.action_type = ActionType.DEVICE_PANEL_OPERATION
    event.objects = ["gloved_hand", "balance"]
    record, plan = reviewer.review(event, views, participant_class="balance")
    assert record["status"] == "completed"
    assert all(not row["boxes"] for row in plan.values())
    assert captured and "相邻设备" in captured[0] and "控制面板" in captured[0]


def test_visual_review_failure_quarantines_event_without_aborting_archive(
    setup, tmp_path, monkeypatch
):
    reviewer, config, views, _ = setup
    config["key_materials"]["participant_visual_review"]["classes"] = ["paper"]
    config["models"]["temporal_participant_segmentation"] = {"enabled": False}
    config["models"]["liquid_semantic_sidecar"] = {"enabled": False}

    class FailingReviewer:
        def __init__(self, *_args, **_kwargs):
            self.enabled = True
            self.records = {}

        def eligible_classes(self, event):
            return ["paper"]

        def review(self, event, visual_views, *, participant_class):
            fingerprint = "failed-review-fingerprint"
            event.observability.setdefault("participant_visual_review", {})[
                fingerprint
            ] = self.records[fingerprint] = {
                "input_fingerprint": fingerprint,
                "participant_class": participant_class,
                "status": "invalid_response",
                "request_attempted": True,
                "usage": {"total_tokens": 10},
            }
            raise RuntimeError(
                "Participant visual review invalid_response; retained receipt "
                + fingerprint
            )

    monkeypatch.setattr(archive, "ParticipantVisualReviewer", FailingReviewer)
    layout = archive.ArchiveLayout(tmp_path / "archive")
    layout.create()
    event = _event()
    raw = layout.work / "key-material-annotation-inputs" / event.event_id
    raw.mkdir(parents=True)
    for view in views:
        role = view["role_label"]
        (raw / f"{role}.jpg").write_bytes(view["raw_path"].read_bytes())
        (raw / f"{role}.json").write_text(
            json.dumps(
                {
                    "detections": [
                        *view["detections"],
                        {
                            "class_name": "paper",
                            "confidence": 0.99,
                            "xyxy_norm": [0.3, 0.3, 0.5, 0.5],
                        },
                    ]
                }
            )
        )
        event.key_frames[view["view_id"]] = (
            f"Key-Materials/Key-Frames/{view['view_id']}.jpg"
        )
    event.key_frames["aligned_first_third"] = (
        "Key-Materials/Key-Frames/aligned.jpg"
    )
    group = ExperimentGroup(
        group_id="group",
        continuity_type="independent",
        atomic_experiment_ids=["atom"],
        global_start_ms=0,
        global_end_ms=1000,
        participating_views=["fp", "tp"],
        first_person_view="fp",
        third_person_view="tp",
        continuity_reason="fixture",
        archive_folder="group-folder",
        key_event_ids=[event.event_id],
    )

    report = archive._rerender_curated_participant_annotations(
        layout, [event], [group], config
    )

    review = event.observability["key_material_annotation"][
        "participant_visual_review"
    ]
    assert review["status"] == "completed_with_quarantined_review_failures"
    assert review["quarantined_review_failure_count"] == 1
    assert all("paper" not in row["rendered_classes"] for row in report["records"])
    assert any("待复核" in note for note in event.uncertainty)

    event.semantic_review = {
        "model_action_type": "hand_object_contact",
        "pre_curation_action_type": "hand_object_contact",
        "pre_curation_objects": list(event.objects),
        "final_delivery_accepted": True,
        "final_participant_objects": list(event.objects),
    }
    semantic_curation = {
        "candidate_count": 1,
        "records": [
            {
                "event_id": event.event_id,
                "disposition": "accepted_confirmed",
                "final_delivery_accepted": True,
                "final_participant_objects": list(event.objects),
            }
        ],
    }
    retained, reconciled = archive.reconcile_visually_reviewed_participants(
        layout, [event], [group], semantic_curation
    )
    assert retained == []
    assert event.accepted is False
    excluded = reconciled["visual_participant_reconciliation"][
        "excluded_events"
    ]
    assert excluded[0]["unsupported_classes"] == ["paper"]
    assert (layout.key_materials / "Review-Candidates" / "Candidate-Index.json").is_file()


def test_cap_and_paper_requests_have_separate_candidates_and_shared_budget(setup, tmp_path):
    _reviewer, config, views, calls = setup
    config["key_materials"]["participant_visual_review"].update(classes=["paper", "bottle_cap"], max_calls_per_run=2)

    def detector(frame, classes, settings):
        name = next(iter(classes))
        return [{"class_name": name, "confidence": 0.8, "xyxy_norm": [0.3,0.3,0.5,0.5]}], {"status":"executed"}

    reviewer = review_module.ParticipantVisualReviewer(config, tmp_path / "both-work", tmp_path / "both-output", detector)
    event = _event()
    event.objects.append("bottle_cap")
    paper, _ = reviewer.review(event, views)
    cap, plan = reviewer.review(event, views, participant_class="bottle_cap")
    assert paper["input_fingerprint"] != cap["input_fingerprint"]
    assert [row["participant_class"] for row in calls] == ["paper", "bottle_cap"]
    assert all(box["class_name"] == "bottle_cap" for row in plan.values() for box in row["boxes"])
    reviewer.review(event, views, participant_class="bottle_cap")
    assert len(calls) == 2


def test_present_paper_does_not_hide_missing_cap_in_annotation_gate():
    event = _event()
    event.objects.append("bottle_cap")
    event.observability["key_material_annotation"] = {
        "mode": "event_participants_only",
        "views": {view: {"rendered_classes":["gloved_hand", "paper"], "extraneous_rendered_classes":[]} for view in ["fp","tp"]},
        "participant_visual_review": {
            "status":"completed", "localized_view_count":2,
            "reviews":[
                {"participant_class":"paper", "status":"completed", "localized_view_count":2},
                {"participant_class":"bottle_cap", "status":"completed", "localized_view_count":0},
            ],
        },
    }
    group = ExperimentGroup(group_id="group", continuity_type="independent", atomic_experiment_ids=["atom"], global_start_ms=0, global_end_ms=1000, participating_views=["fp","tp"], first_person_view="fp", third_person_view="tp", continuity_reason="fixture", key_event_ids=[event.event_id])
    quality = validate_experiment_and_material_quality([group], [event], None, require_participant_only_annotations=True)
    assert quality["key_materials"]["participant_only_annotation_pass_count"] == 0


def test_post_review_curation_prunes_optional_class_and_quarantines_required_failure(tmp_path):
    layout = archive.ArchiveLayout(tmp_path / "archive")
    layout.create()

    optional = _event()
    optional.event_id = "optional"
    optional.objects.append("bottle_cap")
    optional.observability["key_material_annotation"] = {
        "mode": "event_participants_only",
        "views": {
            view: {
                "rendered_classes": ["gloved_hand", "paper"],
                "extraneous_rendered_classes": [],
            }
            for view in ("fp", "tp")
        },
        "participant_visual_review": {
            "status": "completed",
            "reviews": [
                {
                    "participant_class": "paper",
                    "status": "completed",
                    "localized_view_count": 2,
                },
                {
                    "participant_class": "bottle_cap",
                    "status": "completed",
                    "localized_view_count": 0,
                },
            ],
        },
    }
    required = _event()
    required.event_id = "required"
    required.observability["key_material_annotation"] = {
        "mode": "event_participants_only",
        "views": {
            view: {
                "rendered_classes": ["gloved_hand"],
                "extraneous_rendered_classes": [],
            }
            for view in ("fp", "tp")
        },
        "participant_visual_review": {
            "status": "completed",
            "reviews": [
                {
                    "participant_class": "paper",
                    "status": "completed",
                    "localized_view_count": 0,
                }
            ],
        },
    }
    group = ExperimentGroup(
        group_id="group",
        continuity_type="independent",
        atomic_experiment_ids=["atom"],
        global_start_ms=0,
        global_end_ms=1000,
        participating_views=["fp", "tp"],
        first_person_view="fp",
        third_person_view="tp",
        continuity_reason="fixture",
        archive_folder="group-folder",
        key_event_ids=[optional.event_id, required.event_id],
    )
    for event in (optional, required):
        for attribute, root, kind, suffix in (
            ("key_frames", layout.key_frames, "frames", ".jpg"),
            ("key_clips", layout.key_clips, "clips", ".mp4"),
        ):
            directory = root / "group-folder" / "old" / event.event_id
            directory.mkdir(parents=True)
            collection = {}
            for view_id in ("fp", "tp", "aligned_first_third"):
                path = directory / f"{view_id}{suffix}"
                path.write_bytes(f"{event.event_id}-{kind}-{view_id}".encode())
                collection[view_id] = path.relative_to(layout.root).as_posix()
            setattr(event, attribute, collection)
        event.semantic_review = {
            "model_action_type": "hand_object_contact",
            "pre_curation_action_type": "hand_object_contact",
            "pre_curation_objects": list(event.objects),
            "final_delivery_accepted": True,
            "final_participant_objects": list(event.objects),
        }

    semantic_curation = {
        "candidate_count": 2,
        "accepted_count": 2,
        "excluded_count": 0,
        "confirmed_count": 2,
        "review_candidate_count": 0,
        "records": [
            {
                "event_id": event.event_id,
                "disposition": "accepted_confirmed",
                "final_action_type": "hand_object_contact",
                "final_delivery_accepted": True,
                "final_participant_objects": list(event.objects),
            }
            for event in (optional, required)
        ],
    }

    retained, report = archive.reconcile_visually_reviewed_participants(
        layout, [optional, required], [group], semantic_curation
    )

    assert [event.event_id for event in retained] == ["optional"]
    assert optional.objects == ["gloved_hand", "paper"]
    assert optional.accepted is True
    assert all((layout.root / path).is_file() for path in optional.key_frames.values())
    assert required.accepted is False
    assert required.key_frames == {} and required.key_clips == {}
    assert group.key_event_ids == ["optional"]
    assert report["accepted_count"] == 1 and report["excluded_count"] == 1
    reconciliation = report["visual_participant_reconciliation"]
    assert reconciliation["pruned_event_count"] == 1
    assert reconciliation["excluded_event_count"] == 1
    candidate_index = json.loads(
        (layout.key_materials / "Review-Candidates" / "Candidate-Index.json").read_text()
    )
    assert candidate_index["candidate_count"] == 1
    assert candidate_index["candidates"][0]["event_id"] == "required"
    assert candidate_index["candidates"][0]["media"]


def test_post_review_curation_quarantines_wrong_device_identity_without_optional_pruning(
    tmp_path,
):
    layout = archive.ArchiveLayout(tmp_path / "archive")
    layout.create()
    candidate = _event()
    candidate.event_id = "wrong-device"
    candidate.action_type = ActionType.DEVICE_PANEL_OPERATION
    candidate.objects = ["gloved_hand", "tube", "tube_rack"]
    candidate.observability["key_material_annotation"] = {
        "mode": "event_participants_only",
        "views": {
            view: {
                "rendered_classes": ["gloved_hand", "tube"],
                "extraneous_rendered_classes": [],
            }
            for view in ("fp", "tp")
        },
    }
    group = ExperimentGroup(
        group_id="group",
        continuity_type="independent",
        atomic_experiment_ids=["atom"],
        global_start_ms=0,
        global_end_ms=1000,
        participating_views=["fp", "tp"],
        first_person_view="fp",
        third_person_view="tp",
        continuity_reason="fixture",
        archive_folder="group-folder",
        key_event_ids=[candidate.event_id],
    )
    for attribute, root, kind, suffix in (
        ("key_frames", layout.key_frames, "frames", ".jpg"),
        ("key_clips", layout.key_clips, "clips", ".mp4"),
    ):
        directory = root / "group-folder" / "old" / candidate.event_id
        directory.mkdir(parents=True)
        collection = {}
        for view_id in ("fp", "tp", "aligned_first_third"):
            path = directory / f"{view_id}{suffix}"
            path.write_bytes(f"{kind}-{view_id}".encode())
            collection[view_id] = path.relative_to(layout.root).as_posix()
        setattr(candidate, attribute, collection)
    candidate.semantic_review = {
        "model_action_type": "device_panel_operation",
        "pre_curation_action_type": "device_panel_operation",
        "pre_curation_objects": list(candidate.objects),
        "final_delivery_accepted": True,
        "final_participant_objects": list(candidate.objects),
    }
    semantic_curation = {
        "candidate_count": 1,
        "accepted_count": 1,
        "excluded_count": 0,
        "confirmed_count": 1,
        "review_candidate_count": 0,
        "records": [
            {
                "event_id": candidate.event_id,
                "disposition": "accepted_confirmed",
                "final_action_type": "device_panel_operation",
                "final_delivery_accepted": True,
                "final_participant_objects": list(candidate.objects),
            }
        ],
    }

    retained, report = archive.reconcile_visually_reviewed_participants(
        layout, [candidate], [group], semantic_curation
    )

    assert retained == []
    assert candidate.accepted is False
    assert candidate.key_frames == {} and candidate.key_clips == {}
    assert group.key_event_ids == []
    reconciliation = report["visual_participant_reconciliation"]
    assert reconciliation["pruned_event_count"] == 0
    assert reconciliation["excluded_event_count"] == 1
    assert reconciliation["excluded_events"][0]["unsupported_classes"] == []


def test_missing_second_hand_is_supplemented_without_repeating_cloud_review(setup):
    reviewer, config, views, cloud_calls = setup
    detector_calls = []

    def detector(frame, classes, settings):
        detector_calls.append(classes)
        return [{"class_name":"gloved_hand", "confidence":0.9, "xyxy_norm":[0.7,0.7,0.9,0.9]}], {"status":"executed"}

    reviewer.detector = detector
    view = {"raw_path":views[0]["raw_path"], "actors":views[0]["detections"]}
    cap = [{"class_name":"bottle_cap", "confidence":0.8, "xyxy_norm":[0.75,0.75,0.8,0.8]}]
    actors, receipt = reviewer._cap_actor_support(view, cap)
    assert len(actors) == 2 and receipt["status"] == "supplemented"
    assert len(detector_calls) == 1 and not cloud_calls
    again, receipt = reviewer._cap_actor_support(view, cap)
    assert again == actors and receipt["cache_reused"]
    assert len(detector_calls) == 1 and not cloud_calls
