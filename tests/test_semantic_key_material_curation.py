import json
from pathlib import Path

import cv2
import numpy as np

from labvision_evidence.archive import (
    ArchiveLayout,
    _key_material_event_folder_name,
    curate_semantically_reviewed_key_materials,
)
from labvision_evidence.schemas import (
    ActionType,
    EvidenceEvent,
    ExperimentGroup,
    ViewRole,
)


def _event(
    event_id: str,
    action: ActionType,
    verdict: str,
    model_action: str,
    confidence: float = 0.9,
) -> EvidenceEvent:
    return EvidenceEvent(
        event_id=event_id,
        action_type=action,
        global_start_ms=1000,
        global_end_ms=2000,
        key_global_ms=1500,
        objects=["hand", "tube"],
        confidence=0.9,
        accepted=True,
        audit_reason="test",
        supporting_views=["fp", "tp"],
        supporting_roles=[ViewRole.FIRST_PERSON, ViewRole.THIRD_PERSON],
        candidates=[],
        model_understanding={
            "status": "completed",
            "hand_object_interactions": [
                {"hand": "right", "object": "tube", "contact": "grasp"}
            ],
        },
        semantic_review={
            "verdict": verdict,
            "model_evidence_verdict": verdict,
            "model_action_type": model_action,
            "model_confidence": confidence,
            "model_status": "completed",
        },
    )


def _group(events: list[EvidenceEvent]) -> ExperimentGroup:
    return ExperimentGroup(
        group_id="GROUP-0001",
        continuity_type="independent",
        atomic_experiment_ids=["EXP-1"],
        global_start_ms=1000,
        global_end_ms=2000,
        participating_views=["fp", "tp"],
        first_person_view="fp",
        third_person_view="tp",
        continuity_reason="test",
        experiment_name="Test",
        experiment_name_en="Test",
        archive_folder="001-Test",
        key_event_ids=[event.event_id for event in events],
    )


def _materialize_stub(layout: ArchiveLayout, event: EvidenceEvent) -> None:
    action_folder = {
        ActionType.HAND_OBJECT_CONTACT: "01-Hand-Object-Contact",
        ActionType.OBJECT_MOVEMENT: "02-Object-Movement",
        ActionType.LIQUID_MOVEMENT: "03-Liquid-Movement",
        ActionType.CONTAINER_STATE_CHANGE: "04-Container-State-Change",
        ActionType.DEVICE_PANEL_OPERATION: "05-Device-Panel-Operation",
        ActionType.PIPETTE_TRANSFER_OPERATION: "06-Pipette-Transfer-Operation",
    }[event.action_type]
    for attribute, root, suffix in (
        ("key_frames", layout.key_frames, ".jpg"),
        ("key_clips", layout.key_clips, ".mp4"),
    ):
        folder = root / "001-Test" / action_folder / event.event_id
        folder.mkdir(parents=True, exist_ok=True)
        paths = {}
        for view_id in ("fp", "tp", "aligned_first_third"):
            path = folder / f"{view_id}{suffix}"
            path.write_bytes(b"generated-test-media")
            paths[view_id] = path.relative_to(layout.root).as_posix()
        setattr(event, attribute, paths)


def test_curates_confirmed_relabelled_and_rejected_media(tmp_path: Path):
    layout = ArchiveLayout(tmp_path / "archive")
    layout.create()
    layout.work = tmp_path / "cache" / "run"
    confirmed = _event(
        "EVT-CONFIRMED",
        ActionType.HAND_OBJECT_CONTACT,
        "confirmed",
        "hand_object_contact",
    )
    relabelled = _event(
        "EVT-RELABEL",
        ActionType.DEVICE_PANEL_OPERATION,
        "relabel_suggested",
        "hand_object_contact",
    )
    relabelled.global_start_ms = 3000
    relabelled.global_end_ms = 4000
    relabelled.key_global_ms = 3500
    rejected = _event(
        "EVT-REJECTED",
        ActionType.DEVICE_PANEL_OPERATION,
        "rejected",
        "unknown",
    )
    events = [confirmed, relabelled, rejected]
    group = _group(events)
    for item in events:
        _materialize_stub(layout, item)

    curated, report = curate_semantically_reviewed_key_materials(
        layout,
        events,
        [group],
        {"key_materials": {"semantic_relabel_min_confidence": 0.7}},
    )

    assert [item.event_id for item in curated] == [
        "EVT-CONFIRMED",
        "EVT-RELABEL",
    ]
    assert relabelled.action_type == ActionType.HAND_OBJECT_CONTACT
    assert "01-Hand-Object-Contact" in relabelled.key_frames["fp"]
    assert rejected.accepted is False
    assert rejected.key_frames == {}
    assert rejected.key_clips == {}
    assert report["accepted_count"] == 2
    assert report["strict_relabel_count"] == 1
    assert report["excluded_count"] == 1
    assert group.key_event_ids == ["EVT-CONFIRMED", "EVT-RELABEL"]
    assert (
        layout.key_materials
        / "Review-Candidates"
        / "EVT-REJECTED"
        / "Key-Frames"
    ).is_dir()
    review_index = json.loads(
        (
            layout.key_materials
            / "Review-Candidates"
            / "Candidate-Index.json"
        ).read_text(encoding="utf-8")
    )
    assert review_index["candidate_count"] == 1
    assert review_index["candidates"][0]["event_id"] == "EVT-REJECTED"
    assert review_index["candidates"][0]["media"]


def test_same_action_refines_cv_participant_from_structured_interaction(
    tmp_path: Path,
):
    layout = ArchiveLayout(tmp_path / "archive")
    layout.create()
    layout.work = tmp_path / "cache" / "run"
    event = _event(
        "EVT-SAME-ACTION-PARTICIPANT",
        ActionType.HAND_OBJECT_CONTACT,
        "confirmed",
        "hand_object_contact",
        confidence=0.92,
    )
    event.objects = ["gloved_hand", "spearhead"]
    event.model_understanding["hand_object_interactions"] = [
        {"hand": "right", "object": "pipette", "contact": "grasp"}
    ]
    group = _group([event])
    _materialize_stub(layout, event)
    original_frame_folder = Path(event.key_frames["fp"]).parent

    curated, report = curate_semantically_reviewed_key_materials(
        layout,
        [event],
        [group],
        {"key_materials": {"semantic_relabel_min_confidence": 0.7}},
    )

    assert curated == [event]
    assert event.action_type == ActionType.HAND_OBJECT_CONTACT
    assert event.objects == ["gloved_hand", "pipette"]
    assert Path(event.key_frames["fp"]).parent != original_frame_folder
    assert event.semantic_review["semantic_participant_refined"] is True
    assert event.semantic_review[
        "semantic_participant_refinement_changed"
    ] is True
    assert event.state_machine["derivation"]["source"] == (
        "semantic_participant_refinement"
    )
    assert report["records"][0]["final_participant_objects"] == [
        "gloved_hand",
        "pipette",
    ]


def test_same_action_alias_refinement_retains_identical_semantic_media_path(
    tmp_path: Path,
):
    layout = ArchiveLayout(tmp_path / "archive")
    layout.create()
    layout.work = tmp_path / "cache" / "run"
    event = _event(
        "EVT-SAME-SEMANTIC-PATH",
        ActionType.HAND_OBJECT_CONTACT,
        "confirmed",
        "hand_object_contact",
        confidence=0.92,
    )
    event.objects = ["gloved_hand", "称量纸"]
    event.model_understanding["hand_object_interactions"] = [
        {"hand": "right", "object": "weighing_paper", "contact": "touch"}
    ]
    group = _group([event])
    event_folder = _key_material_event_folder_name(layout, "001-Test", event)
    for attribute, root, suffix in (
        ("key_frames", layout.key_frames, ".jpg"),
        ("key_clips", layout.key_clips, ".mp4"),
    ):
        folder = root / "001-Test" / "01-Hand-Object-Contact" / event_folder
        folder.mkdir(parents=True, exist_ok=True)
        paths = {}
        for view_id in ("fp", "tp", "aligned_first_third"):
            path = folder / f"{view_id}{suffix}"
            path.write_bytes(b"generated-test-media")
            paths[view_id] = path.relative_to(layout.root).as_posix()
        setattr(event, attribute, paths)
    original_paths = {
        "key_frames": dict(event.key_frames),
        "key_clips": dict(event.key_clips),
    }

    curated, report = curate_semantically_reviewed_key_materials(
        layout,
        [event],
        [group],
        {"key_materials": {"semantic_relabel_min_confidence": 0.7}},
    )

    assert curated == [event]
    assert event.objects == ["gloved_hand", "paper"]
    assert event.key_frames == original_paths["key_frames"]
    assert event.key_clips == original_paths["key_clips"]
    assert report["records"][0]["moved_media"] == []
    assert report["records"][0]["retained_media"] == [
        {
            "media_kind": "Key-Frames",
            "path": (
                "Key-Materials/Key-Frames/001-Test/01-Hand-Object-Contact/"
                + event_folder
            ),
            "operation": "retained_in_place",
        },
        {
            "media_kind": "Key-Clips",
            "path": (
                "Key-Materials/Key-Clips/001-Test/01-Hand-Object-Contact/"
                + event_folder
            ),
            "operation": "retained_in_place",
        },
    ]


def test_two_strong_views_can_relabel_direct_contact_at_low_summary_confidence(
    tmp_path: Path,
):
    layout = ArchiveLayout(tmp_path / "archive")
    layout.create()
    layout.work = tmp_path / "cache" / "run"
    event = _event(
        "EVT-CAP",
        ActionType.CONTAINER_STATE_CHANGE,
        "relabel_suggested",
        "hand_object_contact",
        confidence=0.40,
    )
    event.semantic_review["confirmed_action_support_by_view"] = [
        {
            "view_id": "fp",
            "supports_confirmed_action": True,
            "confidence": 0.95,
        },
        {
            "view_id": "tp",
            "supports_confirmed_action": True,
            "confidence": 0.90,
        },
    ]
    event.semantic_review["model_evidence_verdict"] = "uncertain"
    group = _group([event])
    _materialize_stub(layout, event)

    curated, report = curate_semantically_reviewed_key_materials(
        layout,
        [event],
        [group],
        {"key_materials": {"semantic_relabel_min_confidence": 0.7}},
    )

    assert curated == [event]
    assert event.action_type == ActionType.HAND_OBJECT_CONTACT
    assert event.semantic_review["curation_disposition"] == "accepted_strict_relabel"
    assert event.semantic_review["relabel_safety_gate"][
        "strong_direct_relabel_consensus"
    ] is True
    assert report["strict_relabel_count"] == 1


def test_raw_candidate_rejection_can_relabel_with_two_strong_direct_views(
    tmp_path: Path,
):
    layout = ArchiveLayout(tmp_path / "archive")
    layout.create()
    layout.work = tmp_path / "cache" / "run"
    event = _event(
        "EVT-CAP-REJECTED",
        ActionType.CONTAINER_STATE_CHANGE,
        "relabel_suggested",
        "hand_object_contact",
        confidence=0.90,
    )
    event.semantic_review["model_evidence_verdict"] = "rejected"
    event.semantic_review["confirmed_action_support_by_view"] = [
        {
            "view_id": "fp",
            "supports_confirmed_action": True,
            "confidence": 0.95,
        },
        {
            "view_id": "tp",
            "supports_confirmed_action": True,
            "confidence": 0.90,
        },
    ]
    event.supporting_views = ["tp"]
    event.supporting_roles = [ViewRole.THIRD_PERSON]
    event.observability = {
        "semantic_recall_admission": {
            "context_view_id": "fp",
            "context_role": "first_person",
        }
    }
    group = _group([event])
    _materialize_stub(layout, event)

    curated, report = curate_semantically_reviewed_key_materials(
        layout,
        [event],
        [group],
        {"key_materials": {"semantic_relabel_min_confidence": 0.7}},
    )

    assert curated == [event]
    assert event.action_type == ActionType.HAND_OBJECT_CONTACT
    assert event.supporting_views == ["fp", "tp"]
    assert event.supporting_roles == [
        ViewRole.FIRST_PERSON,
        ViewRole.THIRD_PERSON,
    ]
    assert event.observability["semantic_direct_support"]["view_ids"] == [
        "fp",
        "tp",
    ]
    assert event.semantic_review["curation_disposition"] == "accepted_strict_relabel"
    assert report["strict_relabel_count"] == 1


def test_low_summary_confidence_contact_uses_one_strong_and_one_clear_view(
    tmp_path: Path,
):
    layout = ArchiveLayout(tmp_path / "archive")
    layout.create()
    layout.work = tmp_path / "cache" / "run"
    event = _event(
        "EVT-LIQUID-DOWNCLASS",
        ActionType.LIQUID_MOVEMENT,
        "relabel_suggested",
        "hand_object_contact",
        confidence=0.20,
    )
    event.objects = ["pipette", "sample_bottle"]
    event.model_understanding.update(
        {
            "current_step": "戴蓝色手套的双手接触并扶住带蓝色瓶盖的试剂瓶。",
            "hand_object_interactions": [
                {
                    "hand": "left",
                    "object": "sample_bottle_blue_cap",
                    "contact": "接触",
                },
                {
                    "hand": "right",
                    "object": "sample_bottle",
                    "contact": "释放",
                },
            ],
        }
    )
    event.semantic_review.update(
        {
            "model_evidence_verdict": "rejected",
            "cross_view_consistency": "conflict",
            "candidate_action_support_by_view": [
                {"view_id": "fp", "supports_candidate_action": False},
                {"view_id": "tp", "supports_candidate_action": False},
            ],
            "confirmed_action_support_by_view": [
                {
                    "view_id": "fp",
                    "supports_confirmed_action": True,
                    "confidence": 0.90,
                },
                {
                    "view_id": "tp",
                    "supports_confirmed_action": True,
                    "confidence": 0.75,
                },
            ],
        }
    )
    event.supporting_views = ["fp"]
    event.supporting_roles = [ViewRole.FIRST_PERSON]
    event.observability = {
        "semantic_recall_admission": {
            "context_view_id": "tp",
            "context_role": "third_person",
        }
    }
    group = _group([event])
    _materialize_stub(layout, event)

    curated, report = curate_semantically_reviewed_key_materials(
        layout,
        [event],
        [group],
        {"key_materials": {"semantic_relabel_min_confidence": 0.7}},
    )

    assert curated == [event]
    assert event.action_type == ActionType.HAND_OBJECT_CONTACT
    assert event.supporting_views == ["fp", "tp"]
    assert event.semantic_review["relabel_safety_gate"][
        "low_risk_direct_relabel_consensus"
    ] is True
    assert event.semantic_review["curation_disposition"] == "accepted_strict_relabel"
    assert report["strict_relabel_count"] == 1


def test_recuration_preserves_prior_strict_relabel_request(tmp_path: Path):
    layout = ArchiveLayout(tmp_path / "archive")
    layout.create()
    layout.work = tmp_path / "cache" / "run"
    event = _event(
        "EVT-RECURATE",
        ActionType.OBJECT_MOVEMENT,
        "confirmed",
        "hand_object_contact",
        confidence=0.90,
    )
    event.objects = ["pipette"]
    event.model_understanding["hand_object_interactions"] = [
        {"hand": "right", "object": "pipette", "contact": "grasp"}
    ]
    event.semantic_review.update(
        {
            "pre_curation_verdict": "relabel_suggested",
            "pre_curation_action_type": "object_movement",
            "model_evidence_verdict": "rejected",
            "confirmed_action_support_by_view": [
                {
                    "view_id": "fp",
                    "supports_confirmed_action": True,
                    "confidence": 0.95,
                },
                {
                    "view_id": "tp",
                    "supports_confirmed_action": True,
                    "confidence": 0.90,
                },
            ],
        }
    )
    group = _group([event])
    _materialize_stub(layout, event)

    curated, report = curate_semantically_reviewed_key_materials(
        layout,
        [event],
        [group],
        {"key_materials": {"semantic_relabel_min_confidence": 0.7}},
    )

    assert curated == [event]
    assert event.action_type == ActionType.HAND_OBJECT_CONTACT
    assert event.objects == ["hand", "pipette"]
    assert event.semantic_review["curation_disposition"] == "accepted_strict_relabel"
    assert event.semantic_review["relabel_safety_gate"][
        "semantic_relabel_requested"
    ] is True
    assert report["strict_relabel_count"] == 1


def test_post_relabel_duplicate_prefers_original_confirmed_event(tmp_path: Path):
    layout = ArchiveLayout(tmp_path / "archive")
    layout.create()
    layout.work = tmp_path / "cache" / "run"
    confirmed = _event(
        "EVT-CONFIRMED",
        ActionType.HAND_OBJECT_CONTACT,
        "confirmed",
        "hand_object_contact",
        confidence=0.8,
    )
    relabelled = _event(
        "EVT-RELABEL",
        ActionType.DEVICE_PANEL_OPERATION,
        "relabel_suggested",
        "hand_object_contact",
        confidence=0.99,
    )
    events = [confirmed, relabelled]
    group = _group(events)
    for item in events:
        _materialize_stub(layout, item)

    curated, report = curate_semantically_reviewed_key_materials(
        layout,
        events,
        [group],
        {"key_materials": {"semantic_relabel_min_confidence": 0.7}},
    )

    assert [item.event_id for item in curated] == ["EVT-CONFIRMED"]
    assert report["post_relabel_duplicate_count"] == 1
    assert report["post_relabel_deduplication"][0]["dropped_event_id"] == (
        "EVT-RELABEL"
    )
    assert relabelled.accepted is False
    assert relabelled.semantic_review["curation_disposition"] == (
        "excluded_post_relabel_duplicate"
    )


def test_completed_state_transition_subsumes_nested_generic_contact(
    tmp_path: Path,
):
    layout = ArchiveLayout(tmp_path / "archive")
    layout.create()
    layout.work = tmp_path / "cache" / "run"
    state_event = _event(
        "EVT-STATE",
        ActionType.CONTAINER_STATE_CHANGE,
        "confirmed",
        "container_state_change",
    )
    state_event.objects = ["gloved_hand", "bottle_cap", "reagent_bottle"]
    state_event.global_start_ms = 1000
    state_event.global_end_ms = 5000
    state_event.key_global_ms = 4000
    state_event.model_understanding.update(
        {
            "current_step": "双手将红色瓶盖拧回棕色试剂瓶。",
            "hand_object_interactions": [
                {"hand": "left", "object": "棕色试剂瓶", "contact": "握持"},
                {"hand": "right", "object": "红色瓶盖", "contact": "拧紧"},
            ],
            "physical_change": {
                "before": "瓶口敞开且瓶盖分离",
                "after": "瓶盖拧紧且瓶口封闭",
            },
            "action_proof": {
                "container_before_state_visible": True,
                "container_after_state_visible": True,
                "container_state_transition_completed": True,
            },
        }
    )
    contact = _event(
        "EVT-NESTED-CONTACT",
        ActionType.HAND_OBJECT_CONTACT,
        "confirmed",
        "hand_object_contact",
    )
    contact.objects = ["gloved_hand", "reagent_bottle"]
    contact.global_start_ms = 2500
    contact.global_end_ms = 4500
    contact.key_global_ms = 3500
    contact.model_understanding.update(
        {
            "current_step": "双手握持棕色试剂瓶。",
            "hand_object_interactions": [
                {"hand": "both", "object": "棕色试剂瓶", "contact": "握持"}
            ],
        }
    )
    events = [state_event, contact]
    group = _group(events)
    group.global_end_ms = 5000
    for event in events:
        _materialize_stub(layout, event)

    curated, report = curate_semantically_reviewed_key_materials(
        layout,
        events,
        [group],
        {"key_materials": {"semantic_relabel_min_confidence": 0.7}},
    )

    assert [event.event_id for event in curated] == ["EVT-STATE"]
    assert report["state_subsumed_contact_count"] == 1
    assert report["state_contact_subsumption"][0]["dropped_event_id"] == (
        "EVT-NESTED-CONTACT"
    )
    assert contact.accepted is False
    assert contact.semantic_review["semantic_subsumed_by"] == "EVT-STATE"
    assert group.key_event_ids == ["EVT-STATE"]


def test_relabel_corrects_participant_objects_before_final_box_render(tmp_path: Path):
    layout = ArchiveLayout(tmp_path / "archive")
    layout.create()
    layout.work = tmp_path / "cache" / "run"
    event = _event(
        "EVT-PAPER",
        ActionType.DEVICE_PANEL_OPERATION,
        "relabel_suggested",
        "object_movement",
    )
    event.objects = ["balance", "gloved_hand"]
    event.model_understanding["hand_object_interactions"] = [
        {"hand": "both", "object": "称量纸", "contact": "抓取"}
    ]
    group = _group([event])
    _materialize_stub(layout, event)
    input_root = layout.work / "key-material-annotation-inputs" / event.event_id
    input_root.mkdir(parents=True)
    detections = [
        {
            "class_name": class_name,
            "confidence": 0.9,
            "xyxy_norm": [0.1 + index * 0.2, 0.1, 0.25 + index * 0.2, 0.4],
        }
        for index, class_name in enumerate(("balance", "gloved_hand", "paper"))
    ]
    for role_label in ("First-Person", "Third-Person"):
        assert cv2.imwrite(
            str(input_root / f"{role_label}.jpg"),
            np.zeros((100, 100, 3), dtype=np.uint8),
        )
        (input_root / f"{role_label}.json").write_text(
            json.dumps({"detections": detections}), encoding="utf-8"
        )

    curated, report = curate_semantically_reviewed_key_materials(
        layout,
        [event],
        [group],
        {"key_materials": {"semantic_relabel_min_confidence": 0.7}},
    )

    assert curated[0].objects == ["gloved_hand", "paper"]
    assert curated[0].action_type == ActionType.OBJECT_MOVEMENT
    view_receipts = curated[0].observability["key_material_annotation"]["views"]
    assert set(view_receipts) == {"fp", "tp"}
    assert all(
        item["rendered_classes"] == ["gloved_hand", "paper"]
        for item in view_receipts.values()
    )
    assert all(
        "balance" in item["suppressed_background_classes"]
        for item in view_receipts.values()
    )
    assert report["final_annotation"]["rendered_view_count"] == 2


def test_direct_dual_role_cv_is_preserved_over_storyboard_relabel(tmp_path: Path):
    layout = ArchiveLayout(tmp_path / "archive")
    layout.create()
    layout.work = tmp_path / "cache" / "run"
    event = _event(
        "EVT-DIRECT-MOVE",
        ActionType.OBJECT_MOVEMENT,
        "relabel_suggested",
        "hand_object_contact",
        confidence=0.9,
    )
    event.objects = ["paper"]
    event.observability = {"can_cv_directly_prove_action": True}
    event.model_understanding["hand_object_interactions"] = [
        {"hand": "right", "object": "balance", "contact": "touch"}
    ]
    group = _group([event])
    _materialize_stub(layout, event)

    curated, report = curate_semantically_reviewed_key_materials(
        layout,
        [event],
        [group],
        {"key_materials": {"semantic_relabel_min_confidence": 0.7}},
    )

    assert curated == [event]
    assert event.action_type == ActionType.OBJECT_MOVEMENT
    assert event.objects == ["paper"]
    assert event.semantic_review["curation_disposition"] == (
        "accepted_objective_cv_preserved_over_sparse_semantics"
    )
    assert report["objective_cv_preserved_count"] == 1


def test_dual_role_object_movement_is_excluded_when_both_views_refute_it(
    tmp_path: Path,
):
    layout = ArchiveLayout(tmp_path / "archive")
    layout.create()
    layout.work = tmp_path / "cache" / "run"
    event = _event(
        "EVT-CAMERA-MOTION-FALSE-POSITIVE",
        ActionType.OBJECT_MOVEMENT,
        "uncertain",
        "unknown",
        confidence=0.25,
    )
    event.objects = ["tube"]
    event.observability = {"can_cv_directly_prove_action": True}
    event.semantic_review.update(
        {
            "model_evidence_verdict": "uncertain",
            "cross_view_consistency": "conflict",
            "candidate_action_support_by_view": [
                {
                    "view_id": "fp",
                    "supports_candidate_action": False,
                    "confidence": 0.30,
                },
                {
                    "view_id": "tp",
                    "supports_candidate_action": False,
                    "confidence": 0.10,
                },
            ],
            "confirmed_action_support_by_view": [],
        }
    )
    group = _group([event])
    _materialize_stub(layout, event)

    curated, report = curate_semantically_reviewed_key_materials(
        layout,
        [event],
        [group],
        {"key_materials": {"semantic_relabel_min_confidence": 0.7}},
    )

    assert curated == []
    assert event.accepted is False
    assert event.semantic_review["curation_disposition"] == (
        "excluded_unanimously_refuted_object_movement"
    )
    assert event.semantic_review["relabel_safety_gate"][
        "objective_movement_unanimously_refuted"
    ] is True
    assert report["excluded_count"] == 1


def test_camera_motion_confirmation_cannot_substitute_a_different_object(
    tmp_path: Path,
):
    layout = ArchiveLayout(tmp_path / "archive")
    layout.create()
    layout.work = tmp_path / "cache" / "run"
    event = _event(
        "EVT-CAMERA-MOTION-OBJECT-MISMATCH",
        ActionType.OBJECT_MOVEMENT,
        "confirmed",
        "object_movement",
        confidence=0.62,
    )
    event.objects = ["tube"]
    event.observability = {"can_cv_directly_prove_action": True}
    event.model_understanding.update(
        {
            "current_step": "蓝色手套握持黑色线缆，镜头发生平移",
            "hand_object_interactions": [
                {
                    "hand": "left",
                    "object": "黑色线缆",
                    "contact": "接触/抓取",
                }
            ],
        }
    )
    event.semantic_review.update(
        {
            "model_evidence_verdict": "confirmed",
            "cross_view_consistency": "conflict",
            "candidate_action_support_by_view": [
                {
                    "view_id": "fp",
                    "supports_candidate_action": True,
                    "confidence": 0.70,
                },
                {
                    "view_id": "tp",
                    "supports_candidate_action": False,
                    "confidence": 0.20,
                },
            ],
            "confirmed_action_support_by_view": [
                {
                    "view_id": "fp",
                    "supports_confirmed_action": True,
                    "confidence": 0.75,
                },
                {
                    "view_id": "tp",
                    "supports_confirmed_action": False,
                    "confidence": 0.25,
                },
            ],
        }
    )
    group = _group([event])
    _materialize_stub(layout, event)

    curated, report = curate_semantically_reviewed_key_materials(
        layout,
        [event],
        [group],
        {"key_materials": {"semantic_relabel_min_confidence": 0.7}},
    )

    assert curated == []
    assert event.accepted is False
    assert event.semantic_review["curation_disposition"] == (
        "review_candidate_semantic_object_identity_mismatch"
    )
    assert event.semantic_review["relabel_safety_gate"][
        "movement_semantic_object_mismatch"
    ] is True
    assert report["excluded_count"] == 1


def test_strong_dual_role_contact_survives_uncertain_sparse_storyboard(
    tmp_path: Path,
):
    layout = ArchiveLayout(tmp_path / "archive")
    layout.create()
    layout.work = tmp_path / "cache" / "run"
    event = _event(
        "EVT-STRONG-CONTACT",
        ActionType.HAND_OBJECT_CONTACT,
        "uncertain",
        "unknown",
        confidence=0.25,
    )
    event.confidence = 0.66
    event.observability = {
        "can_cv_directly_prove_action": False,
        "measurements": {
            "candidate_count": 2,
            "evidence_observation_count": 88,
            "stable_track_token_count": 16,
            "minimum_box_distance_norm": 0.0,
        },
    }
    group = _group([event])
    _materialize_stub(layout, event)

    curated, report = curate_semantically_reviewed_key_materials(
        layout,
        [event],
        [group],
        {"key_materials": {"semantic_relabel_min_confidence": 0.7}},
    )

    assert curated == [event]
    assert event.action_type == ActionType.HAND_OBJECT_CONTACT
    assert event.semantic_review["curation_disposition"] == (
        "accepted_objective_cv_preserved_over_sparse_semantics"
    )
    assert report["objective_cv_preserved_count"] == 1


def test_conflicting_high_risk_relabel_without_object_continuity_is_excluded(
    tmp_path: Path,
):
    layout = ArchiveLayout(tmp_path / "archive")
    layout.create()
    layout.work = tmp_path / "cache" / "run"
    event = _event(
        "EVT-UNSAFE-RELABEL",
        ActionType.DEVICE_PANEL_OPERATION,
        "relabel_suggested",
        "container_state_change",
        confidence=0.9,
    )
    event.objects = ["balance"]
    event.semantic_review["cross_view_consistency"] = "conflict"
    event.model_understanding["hand_object_interactions"] = [
        {"hand": "right", "object": "tube_cap", "contact": "touch"}
    ]
    event.model_understanding["action_proof"] = {
        "proof_type": "direct_other",
        "container_before_state_visible": True,
        "container_after_state_visible": True,
        "container_state_transition_completed": True,
    }
    group = _group([event])
    _materialize_stub(layout, event)

    curated, report = curate_semantically_reviewed_key_materials(
        layout,
        [event],
        [group],
        {"key_materials": {"semantic_relabel_min_confidence": 0.7}},
    )

    assert curated == []
    assert event.accepted is False
    assert event.semantic_review["curation_disposition"] == (
        "excluded_semantically_unconfirmed"
    )
    assert report["unsafe_relabel_excluded_count"] == 1


def test_single_strong_view_can_relabel_completed_container_state_with_context(
    tmp_path: Path,
):
    layout = ArchiveLayout(tmp_path / "archive")
    layout.create()
    layout.work = tmp_path / "cache" / "run"
    event = _event(
        "EVT-CAP-REPLACE",
        ActionType.LIQUID_MOVEMENT,
        "relabel_suggested",
        "container_state_change",
        confidence=0.75,
    )
    event.objects = ["pipette", "sample_bottle"]
    event.supporting_views = ["fp"]
    event.supporting_roles = [ViewRole.FIRST_PERSON]
    event.state_machine = {
        "schema_version": "visioncortex-continuous-action-state/1.0.0",
        "action_type": "liquid_transfer",
        "action_subtype": "pipette_transfer",
        "phases": ["source_contact", "transport", "target_contact"],
        "state_before": {"tool": "outside_source"},
        "state_after": {"tool": "withdrawn_from_target"},
        "object_identity": {
            "object_classes": ["pipette", "sample_bottle"],
            "track_tokens": ["tp:tool_track_id:6"],
            "identity_status": "tracked",
        },
    }
    event.semantic_review.update(
        {
            "model_evidence_verdict": "rejected",
            "cross_view_consistency": "conflict",
            "confirmed_action_support_by_view": [
                {
                    "view_id": "tp",
                    "supports_confirmed_action": True,
                    "confidence": 0.80,
                },
                {
                    "view_id": "fp",
                    "supports_confirmed_action": False,
                    "confidence": 0.20,
                },
            ],
        }
    )
    event.model_understanding.update(
        {
            "current_step": "戴手套的右手正在旋合 blue_bottle_cap",
            "objects": ["blue_bottle_cap", "transparent_bottle", "gloved_hand"],
            "hand_object_interactions": [
                {
                    "hand": "right",
                    "object": "blue_bottle_cap",
                    "contact": "grasp",
                },
                {
                    "hand": "left",
                    "object": "transparent_bottle",
                    "contact": "hold",
                },
            ],
            "action_proof": {
                "proof_type": "direct_other",
                "container_before_state_visible": True,
                "container_after_state_visible": True,
                "container_state_transition_completed": True,
            },
        }
    )
    group = _group([event])
    _materialize_stub(layout, event)

    curated, report = curate_semantically_reviewed_key_materials(
        layout,
        [event],
        [group],
        {"key_materials": {"semantic_relabel_min_confidence": 0.7}},
    )

    assert curated == [event]
    assert event.action_type == ActionType.CONTAINER_STATE_CHANGE
    assert event.objects == ["gloved_hand", "bottle_cap", "reagent_bottle"]
    assert event.supporting_views == ["tp"]
    assert event.supporting_roles == [ViewRole.THIRD_PERSON]
    assert event.semantic_review["curation_disposition"] == (
        "accepted_strict_relabel"
    )
    assert event.semantic_review["relabel_safety_gate"][
        "strict_single_view_state_relabel"
    ] is True
    assert event.semantic_review["relabel_safety_gate"][
        "strongest_direct_relabel_confidence"
    ] == 0.80
    assert event.observability["semantic_direct_support"]["view_ids"] == ["tp"]
    assert event.semantic_review["semantic_state_machine_rebuilt"] is True
    assert event.semantic_review["pre_curation_state_machine"]["action_type"] == (
        "liquid_transfer"
    )
    assert event.state_machine["action_type"] == "container_state_change"
    assert event.state_machine["action_subtype"] == "cap_remove_or_replace"
    assert event.state_machine["phases"] == [
        "state_before",
        "operation",
        "state_after",
        "state_hold",
    ]
    assert event.state_machine["object_identity"] == {
        "object_classes": ["bottle_cap", "reagent_bottle"],
        "track_tokens": [],
        "identity_status": "semantic_participant_classes",
    }
    assert event.state_machine["cross_view"]["supporting_views"] == ["tp"]
    assert all(
        item["source"] == "semantic_relabel_confirmed_state_proof"
        for item in event.state_machine["transition_trace"]
    )
    assert report["strict_relabel_count"] == 1


def test_overlapping_same_direction_state_relabels_are_deduplicated(
    tmp_path: Path,
):
    layout = ArchiveLayout(tmp_path / "archive")
    layout.create()
    layout.work = tmp_path / "cache" / "run"
    events = []
    for event_id, bounds, confidence in (
        ("EVT-OPEN-WIDE", (4300, 12500, 6100), 0.86),
        ("EVT-OPEN-TIGHT", (7700, 12300, 8100), 0.96),
    ):
        event = _event(
            event_id,
            ActionType.LIQUID_MOVEMENT,
            "relabel_suggested",
            "container_state_change",
            confidence=confidence,
        )
        event.global_start_ms, event.global_end_ms, event.key_global_ms = bounds
        event.objects = ["pipette", "sample_bottle_blue"]
        event.supporting_views = ["fp"]
        event.supporting_roles = [ViewRole.FIRST_PERSON]
        event.semantic_review.update(
            {
                "model_evidence_verdict": "rejected",
                "cross_view_consistency": "consistent",
                "confirmed_action_support_by_view": [
                    {
                        "view_id": "fp",
                        "supports_confirmed_action": True,
                        "confidence": 0.90,
                    },
                    {
                        "view_id": "tp",
                        "supports_confirmed_action": True,
                        "confidence": 0.92,
                    },
                ],
            }
        )
        before_after = (
            {
                "before": "目标透明试剂瓶被蓝色瓶盖盖住，瓶口不可见",
                "after": "瓶盖与瓶口分离，瓶口露出，瓶身呈开启状态",
            }
            if event_id == "EVT-OPEN-WIDE"
            else {
                "before": "目标透明试剂瓶由蓝色瓶盖盖住，瓶口封闭",
                "after": "蓝色瓶盖已从瓶口取下，目标试剂瓶瓶口开放",
            }
        )
        event.model_understanding.update(
            {
                "current_step": "双手拧下 blue_cap 并打开 sample_bottle_blue",
                "hand_object_interactions": [
                    {"hand": "left", "object": "blue_cap", "contact": "grasp"},
                    {
                        "hand": "right",
                        "object": "sample_bottle_blue",
                        "contact": "hold",
                    },
                ],
                "physical_change": before_after,
                "action_proof": {
                    "proof_type": "direct_other",
                    "container_before_state_visible": True,
                    "container_after_state_visible": True,
                    "container_state_transition_completed": True,
                },
            }
        )
        events.append(event)
    group = _group(events)
    for event in events:
        _materialize_stub(layout, event)

    curated, report = curate_semantically_reviewed_key_materials(
        layout,
        events,
        [group],
        {"key_materials": {"semantic_relabel_min_confidence": 0.7}},
    )

    assert [event.event_id for event in curated] == ["EVT-OPEN-TIGHT"]
    assert report["post_relabel_duplicate_count"] == 1
    duplicate = report["post_relabel_deduplication"][0]
    assert duplicate["dropped_event_id"] == "EVT-OPEN-WIDE"
    assert duplicate["retained_event_id"] == "EVT-OPEN-TIGHT"
    assert duplicate["state_transition_signature"] == "opening"
    assert duplicate["interval_overlap_ratio"] == 1.0
    assert duplicate["peak_distance_ms"] == 2000


def test_single_view_state_relabel_requires_completed_before_after_proof(
    tmp_path: Path,
):
    layout = ArchiveLayout(tmp_path / "archive")
    layout.create()
    layout.work = tmp_path / "cache" / "run"
    event = _event(
        "EVT-CAP-INCOMPLETE-RELABEL",
        ActionType.LIQUID_MOVEMENT,
        "relabel_suggested",
        "container_state_change",
        confidence=0.95,
    )
    event.objects = ["pipette", "sample_bottle"]
    event.semantic_review.update(
        {
            "model_evidence_verdict": "rejected",
            "cross_view_consistency": "partial",
            "confirmed_action_support_by_view": [
                {
                    "view_id": "tp",
                    "supports_confirmed_action": True,
                    "confidence": 0.99,
                }
            ],
        }
    )
    event.model_understanding.update(
        {
            "current_step": "右手接触 blue_bottle_cap 和 clear_reagent_bottle",
            "hand_object_interactions": [
                {
                    "hand": "right",
                    "object": "blue_bottle_cap",
                    "contact": "touch",
                },
                {
                    "hand": "left",
                    "object": "clear_reagent_bottle",
                    "contact": "hold",
                },
            ],
            "action_proof": {
                "proof_type": "direct_other",
                "container_before_state_visible": True,
                "container_after_state_visible": False,
                "container_state_transition_completed": False,
            },
        }
    )
    group = _group([event])
    _materialize_stub(layout, event)

    curated, _ = curate_semantically_reviewed_key_materials(
        layout,
        [event],
        [group],
        {"key_materials": {"semantic_relabel_min_confidence": 0.7}},
    )

    assert curated == []
    assert event.accepted is False
    assert event.semantic_review["relabel_safety_gate"][
        "strict_single_view_state_relabel"
    ] is False


def test_confirmed_liquid_with_explicit_transfer_denial_is_quarantined(
    tmp_path: Path,
):
    layout = ArchiveLayout(tmp_path / "archive")
    layout.create()
    layout.work = tmp_path / "cache" / "run"
    event = _event(
        "EVT-LIQUID-POSTURE",
        ActionType.LIQUID_MOVEMENT,
        "confirmed",
        "liquid_movement",
        confidence=0.9,
    )
    event.objects = ["pipette", "sample_bottle"]
    event.model_understanding.update(
        {
            "action_proof": {
                "proof_type": "posture_only",
                "visible_liquid_or_level_change": False,
            },
            "uncertainties": ["无法确认液体是否实际被吸取/转移"],
        }
    )
    group = _group([event])
    _materialize_stub(layout, event)

    curated, report = curate_semantically_reviewed_key_materials(
        layout,
        [event],
        [group],
        {"key_materials": {"semantic_relabel_min_confidence": 0.7}},
    )

    assert curated == []
    assert event.accepted is False
    assert event.semantic_review["curation_disposition"] == (
        "excluded_semantic_proof_contradiction"
    )
    assert report["semantic_proof_contradiction_excluded_count"] == 1


def test_incomplete_container_state_is_safely_downclassified_to_contact(
    tmp_path: Path,
):
    layout = ArchiveLayout(tmp_path / "archive")
    layout.create()
    layout.work = tmp_path / "cache" / "run"
    event = _event(
        "EVT-CAP-INCOMPLETE",
        ActionType.CONTAINER_STATE_CHANGE,
        "confirmed",
        "container_state_change",
        confidence=0.82,
    )
    event.objects = ["bottle_cap", "gloved_hand"]
    event.semantic_review.update(
        {
            "cross_view_consistency": "partial",
            "confirmed_action_support_by_view": [
                {
                    "view_id": "fp",
                    "supports_confirmed_action": True,
                    "confidence": 0.85,
                },
                {
                    "view_id": "tp",
                    "supports_confirmed_action": True,
                    "confidence": 0.75,
                },
            ],
        }
    )
    event.model_understanding.update(
        {
            "hand_object_interactions": [
                {"hand": "right", "object": "bottle_cap", "contact": "grasp"}
            ],
            "action_proof": {
                "proof_type": "direct_other",
                "container_before_state_visible": True,
                "container_after_state_visible": False,
                "container_state_transition_completed": False,
            },
        }
    )
    group = _group([event])
    _materialize_stub(layout, event)

    curated, report = curate_semantically_reviewed_key_materials(
        layout,
        [event],
        [group],
        {"key_materials": {"semantic_relabel_min_confidence": 0.7}},
    )

    assert curated == [event]
    assert event.action_type == ActionType.HAND_OBJECT_CONTACT
    assert event.objects == ["bottle_cap", "gloved_hand"]
    assert event.semantic_review["curation_disposition"] == (
        "accepted_state_safety_downclass_to_contact"
    )
    assert report["state_safety_downclass_count"] == 1


def test_direct_single_view_pipette_chain_is_strictly_relabelled(
    tmp_path: Path,
):
    layout = ArchiveLayout(tmp_path / "archive")
    layout.create()
    layout.work = tmp_path / "cache" / "run"
    event = _event(
        "EVT-PIPETTE-OP",
        ActionType.LIQUID_MOVEMENT,
        "relabel_suggested",
        "pipette_transfer_operation",
        confidence=0.84,
    )
    event.objects = ["gloved_hand", "pipette", "tube", "beaker"]
    event.semantic_review.update(
        {
            "cross_view_consistency": "conflict",
            "directly_supported_view_ids": ["fp"],
        }
    )
    event.model_understanding.update(
        {
            "current_step": "移液器从离心管抽离并进入目标烧杯",
            "hand_object_interactions": [
                {"hand": "right", "object": "移液器", "contact": "操作"},
                {"hand": "right", "object": "离心管", "contact": "进入"},
                {"hand": "right", "object": "烧杯", "contact": "进入"},
            ],
            "confirmed_action_support_by_view": [
                {
                    "view_id": "fp",
                    "supports_confirmed_action": True,
                    "confidence": 0.86,
                }
            ],
            "action_proof": {
                "proof_type": "pipette_operational_transfer_chain",
                "visible_liquid_or_level_change": False,
                "source_contact_visible": True,
                "withdrawal_or_transport_visible": True,
                "target_contact_visible": True,
                "release_or_plunger_change_visible": False,
            },
            "uncertainties": ["无法确认液体是否实际被转移"],
        }
    )
    group = _group([event])
    _materialize_stub(layout, event)

    curated, _ = curate_semantically_reviewed_key_materials(
        layout,
        [event],
        [group],
        {"key_materials": {"semantic_relabel_min_confidence": 0.7}},
    )

    assert curated == [event]
    assert event.action_type == ActionType.PIPETTE_TRANSFER_OPERATION
    assert event.semantic_review["curation_disposition"] == "accepted_strict_relabel"
    assert event.semantic_review["semantic_proof_contradictions"] == []
