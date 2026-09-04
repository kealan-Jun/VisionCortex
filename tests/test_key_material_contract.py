from visioncortex.archive import (
    _artifact_json,
    _event_key_frame_score,
    _event_participant_boxes,
    _missing_state_transition_fallback_classes,
)
from visioncortex.schemas import (
    ActionType,
    AlignmentTransform,
    EvidenceEvent,
    ExperimentGroup,
    FrameEvidence,
    ViewRole,
)


def test_key_material_json_uses_normalized_event_contract():
    event = EvidenceEvent(
        event_id="EVT-000001",
        action_type=ActionType.LIQUID_MOVEMENT,
        global_start_ms=12_000,
        global_end_ms=20_500,
        key_global_ms=16_700,
        objects=["pipette", "reagent_bottle_open", "tube"],
        confidence=0.88,
        accepted=True,
        audit_reason="cross-view evidence",
        supporting_views=["fp", "tp"],
        supporting_roles=[ViewRole.FIRST_PERSON, ViewRole.THIRD_PERSON],
        candidates=[],
        key_frames={"fp": "fp.jpg", "tp": "tp.jpg", "aligned_first_third": "aligned.jpg"},
        key_clips={"fp": "fp.mp4", "tp": "tp.mp4", "aligned_first_third": "aligned.mp4"},
        model_understanding={
            "status": "completed",
            "action_type_confirmed": "liquid_movement",
            "current_step": "移液器从源容器吸取液体",
            "next_step": "移动至目标离心管并释放液体",
            "physical_change": {"before": "移液器在源容器外", "after": "移液器从目标容器撤回"},
            "per_view_observations": [
                {"view_id": "fp", "observation": "移液器接触源容器"},
                {"view_id": "tp", "observation": "手将移液器移向离心管"},
            ],
            "cross_view_consistency": "consistent",
            "confidence": 0.84,
            "uncertainties": [],
            "usage": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
        },
    )
    group = ExperimentGroup(
        group_id="GROUP-0001",
        continuity_type="independent",
        atomic_experiment_ids=["EXP-0001"],
        global_start_ms=10_000,
        global_end_ms=22_000,
        participating_views=["fp", "tp"],
        first_person_view="fp",
        third_person_view="tp",
        continuity_reason="independent",
        key_event_ids=[event.event_id],
    )
    transforms = {
        view_id: AlignmentTransform(view_id=view_id, reference_view_id="fp", confidence=1.0)
        for view_id in ("fp", "tp")
    }
    payload = _artifact_json(group, event, "key_clip", "clip.mp4", "fp", transforms)
    assert list(payload) == [
        "event_id",
        "parent_event_id",
        "parent_event_uid",
        "actor_id",
        "workstation_id",
        "action_type",
        "action_subtype",
        "start_us",
        "end_us",
        "peak_timestamp_us",
        "time_uncertainty_us",
        "phases",
        "objects",
        "state_before",
        "state_after",
        "observations",
        "cross_view_associations",
        "decision",
        "scores",
        "key_frames",
        "key_clips",
        "evidence_ids",
        "provenance",
    ]
    assert payload["action_type"] == "liquid_transfer"
    assert payload["action_subtype"] == "pipette_transfer"
    assert payload["decision"]["observed_facts"]
    assert payload["decision"]["supported_inferences"]
    assert len(payload["key_frames"]) == 3
    assert len(payload["key_clips"]) == 3
    assert payload["provenance"]["mllm"]["usage"]["total_tokens"] == 120


def test_contact_state_change_is_attached_to_target_not_unknown_tool():
    event = EvidenceEvent(
        event_id="EVT-CONTACT-TARGET",
        action_type=ActionType.HAND_OBJECT_CONTACT,
        global_start_ms=1000,
        global_end_ms=2000,
        key_global_ms=1500,
        objects=["gloved_hand", "paper"],
        confidence=0.9,
        accepted=True,
        audit_reason="target contact",
        supporting_views=["fp", "tp"],
        supporting_roles=[ViewRole.FIRST_PERSON, ViewRole.THIRD_PERSON],
        candidates=[],
        model_understanding={
            "status": "completed",
            "action_type_confirmed": "hand_object_contact",
            "physical_change": {
                "before": "not_contacting",
                "after": "contacting",
            },
            "confidence": 0.9,
        },
    )
    group = ExperimentGroup(
        group_id="GROUP-CONTACT",
        continuity_type="independent",
        atomic_experiment_ids=["EXP-CONTACT"],
        global_start_ms=1000,
        global_end_ms=2000,
        participating_views=["fp", "tp"],
        first_person_view="fp",
        third_person_view="tp",
        continuity_reason="test",
        key_event_ids=[event.event_id],
    )
    transforms = {
        view_id: AlignmentTransform(
            view_id=view_id, reference_view_id="fp", confidence=1.0
        )
        for view_id in ("fp", "tp")
    }

    payload = _artifact_json(
        group, event, "key_material_event_index", "", None, transforms
    )

    assert set(payload["objects"]) == {"actor", "target"}
    assert payload["objects"]["target"].startswith("paper-")
    assert payload["state_before"] == {"target": "not_contacting"}
    assert payload["state_after"] == {"target": "contacting"}


def test_relabelled_key_material_contract_separates_cv_from_final_semantics():
    pre_curation_state = {
        "action_type": "liquid_transfer",
        "action_subtype": "pipette_transfer",
        "phases": ["source_contact", "transport", "target_contact"],
    }
    final_state = {
        "action_type": "container_state_change",
        "action_subtype": "cap_remove_or_replace",
        "phases": ["state_before", "operation", "state_after", "state_hold"],
        "state_before": {"container": "observed_before_state"},
        "state_after": {"container": "observed_changed_state"},
        "scores": {"phase_completeness": 1.0},
    }
    event = EvidenceEvent(
        event_id="EVT-RELABELLED-STATE",
        action_type=ActionType.CONTAINER_STATE_CHANGE,
        global_start_ms=1000,
        global_end_ms=3000,
        key_global_ms=2500,
        objects=["gloved_hand", "bottle_cap", "sample_bottle"],
        confidence=0.8,
        accepted=True,
        audit_reason="semantic relabel",
        supporting_views=["tp"],
        supporting_roles=[ViewRole.THIRD_PERSON],
        candidates=[],
        state_machine=final_state,
        semantic_review={
            "pre_curation_action_type": "liquid_movement",
            "pre_curation_objects": ["pipette", "sample_bottle"],
            "pre_curation_supporting_views": ["fp", "tp"],
            "pre_curation_state_machine": pre_curation_state,
            "final_action_type": "container_state_change",
            "final_participant_objects": [
                "gloved_hand",
                "bottle_cap",
                "sample_bottle",
            ],
        },
        model_understanding={
            "status": "completed",
            "action_type_confirmed": "container_state_change",
            "objects": ["pipette", "balance", "sample_bottle"],
            "physical_change": {"before": "瓶口开放", "after": "瓶盖已盖回"},
            "confidence": 0.9,
        },
    )
    group = ExperimentGroup(
        group_id="GROUP-1",
        continuity_type="independent",
        atomic_experiment_ids=["EXP-1"],
        global_start_ms=1000,
        global_end_ms=3000,
        participating_views=["fp", "tp"],
        first_person_view="fp",
        third_person_view="tp",
        continuity_reason="test",
        key_event_ids=[event.event_id],
    )
    transforms = {
        view_id: AlignmentTransform(
            view_id=view_id, reference_view_id="fp", confidence=1.0
        )
        for view_id in ("fp", "tp")
    }

    payload = _artifact_json(
        group, event, "key_material_event_index", "", None, transforms
    )

    assert payload["action_type"] == "container_state_change"
    assert payload["action_subtype"] == "cap_remove_or_replace"
    assert payload["phases"] == final_state["phases"]
    assert set(payload["objects"]) == {"actor", "container", "closure"}
    assert all("pipette" not in value for value in payload["objects"].values())
    assert payload["state_before"] == {"container": "瓶口开放"}
    assert payload["state_after"] == {"container": "瓶盖已盖回"}
    assert payload["provenance"]["cv"]["action_type"] == "liquid_movement"
    assert payload["provenance"]["cv"]["continuous_state_machine"] == (
        pre_curation_state
    )
    assert payload["provenance"]["semantic_final"]["action_type"] == (
        "container_state_change"
    )
    assert payload["provenance"]["semantic_final"][
        "continuous_state_machine"
    ] == final_state


def test_key_material_annotation_renders_only_event_participants():
    event = EvidenceEvent(
        event_id="EVT-BOX-FILTER",
        action_type=ActionType.HAND_OBJECT_CONTACT,
        global_start_ms=1000,
        global_end_ms=2000,
        key_global_ms=1500,
        objects=["gloved_hand", "paper"],
        confidence=0.9,
        accepted=True,
        audit_reason="test",
        supporting_views=["fp", "tp"],
        supporting_roles=[ViewRole.FIRST_PERSON, ViewRole.THIRD_PERSON],
        candidates=[],
    )
    detections = [
        {"class_name": "gloved_hand", "confidence": 0.9},
        {"class_name": "paper", "confidence": 0.8},
        {"class_name": "balance", "confidence": 0.95},
        {"class_name": "tube_rack", "confidence": 0.85},
    ]

    rendered, receipt = _event_participant_boxes(event, detections)

    assert [item["class_name"] for item in rendered] == ["gloved_hand", "paper"]
    assert receipt["mode"] == "event_participants_only"
    assert receipt["rendered_box_count"] == 2
    assert receipt["suppressed_background_box_count"] == 2
    assert receipt["extraneous_rendered_classes"] == []


def test_pipette_annotation_keeps_only_interacting_instances():
    event = EvidenceEvent(
        event_id="EVT-INSTANCE-FILTER",
        action_type=ActionType.PIPETTE_TRANSFER_OPERATION,
        global_start_ms=1000,
        global_end_ms=2000,
        key_global_ms=1500,
        objects=["pipette", "container"],
        confidence=0.9,
        accepted=True,
        audit_reason="test",
        supporting_views=["fp", "tp"],
        supporting_roles=[ViewRole.FIRST_PERSON, ViewRole.THIRD_PERSON],
        candidates=[],
    )
    detections = [
        {
            "class_name": "gloved_hand",
            "confidence": 0.92,
            "xyxy_norm": [0.40, 0.30, 0.60, 0.60],
            "track_id": 10,
            "roi_motion": 20.0,
        },
        {
            "class_name": "pipette",
            "confidence": 0.88,
            "xyxy_norm": [0.50, 0.40, 0.56, 0.75],
            "track_id": 11,
            "roi_motion": 25.0,
        },
        {
            "class_name": "beaker",
            "confidence": 0.95,
            "xyxy_norm": [0.48, 0.68, 0.70, 0.95],
            "track_id": 12,
            "roi_motion": 12.0,
        },
        {
            "class_name": "pipette",
            "confidence": 0.99,
            "xyxy_norm": [0.02, 0.02, 0.08, 0.25],
            "track_id": 90,
            "roi_motion": 1.0,
        },
        {
            "class_name": "pipette",
            "confidence": 0.97,
            "xyxy_norm": [0.12, 0.02, 0.18, 0.25],
            "track_id": 91,
            "roi_motion": 1.0,
        },
    ]

    rendered, receipt = _event_participant_boxes(event, detections, view_id="fp")

    assert [item["track_id"] for item in rendered] == [10, 11, 12]
    assert receipt["instance_policy"] == (
        "single_interacting_instance_per_semantic_slot"
    )
    assert receipt["suppressed_same_class_instance_count"] == 2
    assert receipt["rendered_track_ids"] == [10, 11, 12]


def test_pipette_annotation_does_not_invent_unlisted_tool_or_vessel_slots():
    event = EvidenceEvent(
        event_id="EVT-STRICT-PARTICIPANTS",
        action_type=ActionType.PIPETTE_TRANSFER_OPERATION,
        global_start_ms=1000,
        global_end_ms=2000,
        key_global_ms=1500,
        objects=["gloved_hand", "pipette"],
        confidence=0.9,
        accepted=True,
        audit_reason="test",
        supporting_views=["fp"],
        supporting_roles=[ViewRole.FIRST_PERSON],
        candidates=[],
    )
    detections = [
        {
            "class_name": "gloved_hand",
            "confidence": 0.9,
            "xyxy_norm": [0.40, 0.30, 0.60, 0.60],
        },
        {
            "class_name": "pipette",
            "confidence": 0.8,
            "xyxy_norm": [0.50, 0.40, 0.56, 0.75],
        },
        {
            "class_name": "spearhead",
            "confidence": 0.99,
            "xyxy_norm": [0.55, 0.40, 0.65, 0.70],
        },
        {
            "class_name": "sample_bottle",
            "confidence": 0.99,
            "xyxy_norm": [0.52, 0.65, 0.70, 0.95],
        },
    ]

    rendered, receipt = _event_participant_boxes(event, detections)

    assert [item["class_name"] for item in rendered] == [
        "gloved_hand",
        "pipette",
    ]
    assert "spearhead" in receipt["suppressed_background_classes"]
    assert "sample_bottle" in receipt["suppressed_background_classes"]


def test_contact_annotation_rejects_distant_static_participant_instance():
    event = EvidenceEvent(
        event_id="EVT-DISTANT-OBJECT",
        action_type=ActionType.HAND_OBJECT_CONTACT,
        global_start_ms=1000,
        global_end_ms=2000,
        key_global_ms=1500,
        objects=["gloved_hand", "balance"],
        confidence=0.9,
        accepted=True,
        audit_reason="test",
        supporting_views=["fp"],
        supporting_roles=[ViewRole.FIRST_PERSON],
        candidates=[],
    )
    detections = [
        {
            "class_name": "gloved_hand",
            "confidence": 0.9,
            "xyxy_norm": [0.70, 0.70, 0.85, 0.90],
        },
        {
            "class_name": "balance",
            "confidence": 0.99,
            "xyxy_norm": [0.05, 0.05, 0.30, 0.30],
        },
    ]

    rendered, receipt = _event_participant_boxes(event, detections)

    assert [item["class_name"] for item in rendered] == ["gloved_hand"]
    assert receipt["interaction_relation_gate"]["rejected_box_count"] == 1


def test_container_state_annotation_uses_nearest_bottle_alias_only():
    event = EvidenceEvent(
        event_id="EVT-CAP-ALIAS",
        action_type=ActionType.CONTAINER_STATE_CHANGE,
        global_start_ms=1000,
        global_end_ms=2000,
        key_global_ms=1500,
        objects=["gloved_hand", "bottle_cap", "reagent_bottle"],
        confidence=0.9,
        accepted=True,
        audit_reason="test",
        supporting_views=["fp", "tp"],
        supporting_roles=[ViewRole.FIRST_PERSON, ViewRole.THIRD_PERSON],
        candidates=[],
    )
    detections = [
        {
            "class_name": "gloved_hand",
            "confidence": 0.90,
            "xyxy_norm": [0.45, 0.45, 0.60, 0.70],
            "track_id": 1,
        },
        {
            # The detector calls the manipulated reagent bottle a blue sample
            # bottle; it is still the semantic container participant.
            "class_name": "sample_bottle_blue",
            "confidence": 0.80,
            "xyxy_norm": [0.50, 0.35, 0.62, 0.55],
            "track_id": 2,
        },
        {
            "class_name": "reagent_bottle",
            "confidence": 0.99,
            "xyxy_norm": [0.01, 0.01, 0.10, 0.20],
            "track_id": 99,
        },
        {
            "class_name": "beaker",
            "confidence": 0.99,
            "xyxy_norm": [0.55, 0.55, 0.80, 0.90],
            "track_id": 100,
        },
    ]

    rendered, receipt = _event_participant_boxes(event, detections, view_id="fp")

    assert [item["track_id"] for item in rendered] == [1, 2]
    assert receipt["rendered_classes"] == ["gloved_hand", "sample_bottle_blue"]
    assert receipt["suppressed_same_class_instance_count"] == 1
    assert "beaker" in receipt["suppressed_background_classes"]


def test_state_transition_fallback_requests_every_missing_physical_slot():
    active = {"bottle_cap", "reagent_bottle"}
    state_prompts = {"closed bottle", "capped bottle"}

    assert _missing_state_transition_fallback_classes(
        [],
        active_object_classes=active,
        state_prompts=state_prompts,
        actor_boxes=[
            {"class_name": "gloved_hand", "xyxy_norm": [0.4, 0.4, 0.6, 0.7]}
        ],
        maximum_actor_gap=0.08,
        closure_maximum_actor_gap=0.01,
        closure_minimum_confidence=0.12,
    ) == {"bottle_cap", "container", "reagent_bottle"}
    assert _missing_state_transition_fallback_classes(
        [
                {
                    "class_name": "bottle_cap",
                    "confidence": 0.5,
                    "grounding_prompt": "bottle cap",
                "xyxy_norm": [0.45, 0.45, 0.50, 0.50],
            },
            {
                "class_name": "container",
                "grounding_prompt": "closed bottle",
                "xyxy_norm": [0.45, 0.50, 0.58, 0.85],
            },
        ],
        active_object_classes=active,
        state_prompts=state_prompts,
        actor_boxes=[
            {"class_name": "gloved_hand", "xyxy_norm": [0.4, 0.4, 0.6, 0.7]}
        ],
        maximum_actor_gap=0.08,
        closure_maximum_actor_gap=0.01,
        closure_minimum_confidence=0.12,
    ) == set()


def test_participant_rich_action_frame_outscores_empty_nominal_peak():
    event = EvidenceEvent(
        event_id="EVT-KEY-FRAME",
        action_type=ActionType.PIPETTE_TRANSFER_OPERATION,
        global_start_ms=1000,
        global_end_ms=2000,
        key_global_ms=1200,
        objects=["pipette", "container"],
        confidence=0.9,
        accepted=True,
        audit_reason="test",
        supporting_views=["fp", "tp"],
        supporting_roles=[ViewRole.FIRST_PERSON, ViewRole.THIRD_PERSON],
        candidates=[],
    )
    empty = FrameEvidence(
        view_id="fp",
        role=ViewRole.FIRST_PERSON,
        frame_index=1,
        local_ms=1200,
        global_ms=1200,
        width=640,
        height=360,
        detections=[],
    )
    action = FrameEvidence(
        view_id="fp",
        role=ViewRole.FIRST_PERSON,
        frame_index=2,
        local_ms=1600,
        global_ms=1600,
        width=640,
        height=360,
        detections=[
            {
                "class_id": 0,
                "class_name": "gloved_hand",
                "confidence": 0.9,
                "xyxy_norm": [0.4, 0.3, 0.6, 0.6],
            },
            {
                "class_id": 1,
                "class_name": "pipette",
                "confidence": 0.9,
                "xyxy_norm": [0.5, 0.4, 0.56, 0.75],
            },
            {
                "class_id": 2,
                "class_name": "beaker",
                "confidence": 0.9,
                "xyxy_norm": [0.48, 0.68, 0.7, 0.95],
            },
        ],
    )

    empty_score, _ = _event_key_frame_score(event, empty)
    action_score, _ = _event_key_frame_score(event, action)

    assert action_score > empty_score + 300


def test_container_state_after_frame_wins_equal_participant_tie():
    event = EvidenceEvent(
        event_id="EVT-CAP-PHASE",
        action_type=ActionType.CONTAINER_STATE_CHANGE,
        global_start_ms=4000,
        global_end_ms=12000,
        key_global_ms=6000,
        objects=["gloved_hand", "bottle_cap", "reagent_bottle"],
        confidence=0.9,
        accepted=True,
        audit_reason="test",
        supporting_views=["fp", "tp"],
        supporting_roles=[ViewRole.FIRST_PERSON, ViewRole.THIRD_PERSON],
        candidates=[],
    )
    detections = [
        {
            "class_id": 0,
            "class_name": "gloved_hand",
            "confidence": 0.9,
            "xyxy_norm": [0.4, 0.4, 0.6, 0.7],
        },
        {
            "class_id": 1,
            "class_name": "sample_bottle_blue",
            "confidence": 0.8,
            "xyxy_norm": [0.5, 0.3, 0.65, 0.55],
        },
    ]
    before = FrameEvidence(
        view_id="fp",
        role=ViewRole.FIRST_PERSON,
        frame_index=1,
        local_ms=5200,
        global_ms=5200,
        width=640,
        height=360,
        detections=detections,
    )
    after = FrameEvidence(
        view_id="fp",
        role=ViewRole.FIRST_PERSON,
        frame_index=2,
        local_ms=10400,
        global_ms=10400,
        width=640,
        height=360,
        detections=detections,
    )

    before_score, before_receipt = _event_key_frame_score(event, before)
    after_score, after_receipt = _event_key_frame_score(event, after)

    assert after_score > before_score
    assert after_receipt["key_frame_phase_bias"] > before_receipt[
        "key_frame_phase_bias"
    ]
