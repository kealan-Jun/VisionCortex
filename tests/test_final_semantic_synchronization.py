from visioncortex.archive import (
    _filter_grounded_actor_boxes,
    _relabel_participant_objects,
    _select_manipulated_object_candidate,
    _select_state_container_candidate,
    _view_specific_participant_objects,
)
from visioncortex.pipeline import (
    _synchronize_final_event_state_receipts,
    _synchronize_segments_with_final_key_events,
)
from visioncortex.schemas import (
    ActionType,
    EvidenceEvent,
    ExperimentGroup,
    ExperimentSegment,
    ViewRole,
)


def _event(event_id: str, *, action: ActionType, start: float, end: float, peak: float):
    return EvidenceEvent(
        event_id=event_id,
        action_type=action,
        global_start_ms=start,
        global_end_ms=end,
        key_global_ms=peak,
        objects=["gloved_hand", "bottle_cap", "reagent_bottle"],
        confidence=0.9,
        accepted=True,
        audit_reason="accepted",
        supporting_views=["fp", "tp"],
        supporting_roles=[ViewRole.FIRST_PERSON, ViewRole.THIRD_PERSON],
        candidates=[],
        model_understanding={
            "action_proof": {
                "container_before_state_visible": True,
                "container_after_state_visible": True,
                "container_state_transition_completed": True,
            }
        },
    )


def test_final_segment_contains_only_curated_semantic_events():
    retained = _event(
        "EVT-RETAINED",
        action=ActionType.CONTAINER_STATE_CHANGE,
        start=7700.0,
        end=12300.0,
        peak=8100.0,
    )
    group = ExperimentGroup(
        group_id="GROUP-0001",
        continuity_type="independent",
        atomic_experiment_ids=["EXP-0001"],
        global_start_ms=2300.0,
        global_end_ms=15500.0,
        participating_views=["fp", "tp"],
        first_person_view="fp",
        third_person_view="tp",
        continuity_reason="one action",
        key_event_ids=[retained.event_id],
    )
    segment = ExperimentSegment(
        segment_id="EXP-0001",
        group_id=group.group_id,
        global_start_ms=2300.0,
        global_end_ms=15500.0,
        event_ids=["EVT-DUPLICATE", retained.event_id],
        participating_views=["fp", "tp"],
        micro_segments=[
            {
                "micro_segment_id": "MICRO-0001-0001",
                "evidence_event_id": "EVT-DUPLICATE",
                "action_type": "liquid_movement",
            },
            {
                "micro_segment_id": "MICRO-0001-0002",
                "evidence_event_id": retained.event_id,
                "action_type": "liquid_movement",
            },
        ],
    )

    receipts = _synchronize_segments_with_final_key_events(
        [segment], [group], [retained]
    )

    assert segment.event_ids == [retained.event_id]
    assert len(segment.micro_segments) == 1
    assert segment.micro_segments[0]["action_type"] == "container_state_change"
    assert segment.micro_segments[0]["objects"] == retained.objects
    assert receipts[0]["previous_micro_segment_count"] == 2
    assert receipts[0]["final_micro_segment_count"] == 1


def test_final_key_timestamp_resynchronizes_semantic_state_receipt():
    event = _event(
        "EVT-OPEN",
        action=ActionType.CONTAINER_STATE_CHANGE,
        start=4300.0,
        end=12500.0,
        peak=8300.0,
    )
    event.state_machine = {
        "action_type": "container_state_change",
        "peak_timestamp_us": 6100000,
        "publication": {"status": "primary"},
        "derivation": {
            "source": "semantic_relabel_confirmed_state_proof",
            "pre_curation_action_type": "liquid_movement",
        },
    }

    repairs = _synchronize_final_event_state_receipts([event], {})

    assert repairs[0]["final_peak_timestamp_us"] == 8300000
    assert event.state_machine["publication"] == {"status": "primary"}
    assert event.state_machine["object_identity"]["track_tokens"] == []
    assert (
        event.state_machine["derivation"][
            "timing_resynchronized_after_final_key_frame_selection"
        ]
        is True
    )


def test_participant_refinement_excludes_explicit_noncontact_object():
    event = _event(
        "EVT-DIRECT-PARTICIPANTS",
        action=ActionType.HAND_OBJECT_CONTACT,
        start=1000.0,
        end=2000.0,
        peak=1500.0,
    )
    event.objects = ["gloved_hand", "balance", "pipette"]
    event.model_understanding = {
        "current_step": "手握移液器并靠近天平",
        "hand_object_interactions": [
            {"hand": "right", "object": "pipette", "contact": "抓取"},
            {
                "hand": "left",
                "object": "centrifuge_tube_or_tube_box",
                "contact": "接触/抓取",
            },
            {
                "hand": "left",
                "object": "balance",
                "contact": "接近/未清晰证明接触",
            },
        ],
    }

    assert _relabel_participant_objects(event) == [
        "gloved_hand",
        "pipette",
        "tube",
    ]


def test_participant_refinement_resolves_generic_red_container_to_bottle_cap():
    event = _event(
        "EVT-BROWN-BOTTLE-CONTACT",
        action=ActionType.HAND_OBJECT_CONTACT,
        start=16600.0,
        end=21200.0,
        peak=18900.0,
    )
    event.objects = ["gloved_hand", "container"]
    event.model_understanding = {
        "current_step": (
            "戴蓝色手套的双手接触棕色带红环开口瓶和红色小盖容器。"
        ),
        "hand_object_interactions": [
            {
                "hand": "left",
                "object": "brown bottle with red ring",
                "contact": "接触/抓取",
            },
            {
                "hand": "right",
                "object": "red small container",
                "contact": "接触/抓取",
            },
        ],
    }

    assert _relabel_participant_objects(event) == [
        "gloved_hand",
        "reagent_bottle",
        "bottle_cap",
    ]


def test_participant_refinement_prefers_specific_gloved_actor_over_hand():
    event = _event(
        "EVT-GLOVED-PIPETTE",
        action=ActionType.HAND_OBJECT_CONTACT,
        start=1000.0,
        end=2000.0,
        peak=1500.0,
    )
    event.objects = ["gloved_hand", "hand", "pipette"]
    event.model_understanding = {
        "current_step": "戴蓝色手套的右手握持移液器。",
        "hand_object_interactions": [
            {"hand": "right", "object": "pipette", "contact": "grasp"}
        ],
    }

    assert _relabel_participant_objects(event) == ["gloved_hand", "pipette"]


def test_conflicting_views_use_only_their_observed_participant_object():
    event = _event(
        "EVT-CONFLICTING-CONTACTS",
        action=ActionType.HAND_OBJECT_CONTACT,
        start=1000.0,
        end=2000.0,
        peak=1500.0,
    )
    event.objects = ["gloved_hand", "pipette", "reagent_bottle"]
    event.semantic_review = {"cross_view_consistency": "conflict"}
    event.model_understanding = {
        "per_view_observations": [
            {"view_id": "fp", "observation": "右手从架上取下移液器并握持。"},
            {"view_id": "tp", "observation": "右手接触并抓取小试剂瓶。"},
        ],
        "confirmed_action_support_by_view": [
            {"view_id": "fp", "supports_confirmed_action": True},
            {"view_id": "tp", "supports_confirmed_action": True},
        ],
    }

    assert _view_specific_participant_objects(event, "fp") == [
        "gloved_hand",
        "pipette",
    ]
    assert _view_specific_participant_objects(event, "tp") == [
        "gloved_hand",
        "reagent_bottle",
    ]


def test_view_participants_ignore_objects_mentioned_only_in_negation():
    event = _event(
        "EVT-NEGATED-PIPETTE",
        action=ActionType.HAND_OBJECT_CONTACT,
        start=1000.0,
        end=2000.0,
        peak=1500.0,
    )
    event.objects = ["gloved_hand", "pipette", "reagent_bottle"]
    event.model_understanding = {
        "cross_view_consistency": "conflict",
        "per_view_observations": [
            {
                "view_id": "tp",
                "observation": "右手抓取小试剂瓶；未看到接触移液器。",
            }
        ],
        "confirmed_action_support_by_view": [
            {
                "view_id": "tp",
                "supports_confirmed_action": True,
                "reason": "该视角直接证明右手与小试剂瓶发生抓取。",
            }
        ],
    }

    assert _view_specific_participant_objects(event, "tp") == [
        "gloved_hand",
        "reagent_bottle",
    ]


def test_unsupported_view_keeps_only_actor_annotation():
    event = _event(
        "EVT-UNSUPPORTED-TRANSFER-VIEW",
        action=ActionType.LIQUID_MOVEMENT,
        start=1000.0,
        end=2000.0,
        peak=1500.0,
    )
    event.objects = ["gloved_hand", "pipette", "beaker"]
    event.model_understanding = {
        "per_view_observations": [
            {
                "view_id": "tp",
                "observation": "未拍到移液器与烧杯操作链。",
            }
        ],
        "confirmed_action_support_by_view": [
            {
                "view_id": "tp",
                "supports_confirmed_action": False,
                "reason": "该视角不能直接支持该动作。",
            }
        ],
    }

    assert _view_specific_participant_objects(event, "tp") == ["gloved_hand"]


def test_first_person_opening_prefers_container_below_operating_hand():
    actor = {
        "class_name": "gloved_hand",
        "confidence": 0.56,
        "xyxy_norm": [0.30, 0.49, 0.48, 0.72],
    }
    background_open_vial = {
        "class_name": "container",
        "confidence": 0.41,
        "xyxy_norm": [0.44, 0.50, 0.49, 0.61],
    }
    manipulated_open_jar = {
        "class_name": "container",
        "confidence": 0.04,
        "xyxy_norm": [0.43, 0.70, 0.49, 0.90],
    }

    selected, receipt = _select_state_container_candidate(
        [background_open_vial, manipulated_open_jar],
        [actor],
        view_role="First-Person",
        state_direction="opening",
        maximum_actor_gap=0.08,
    )

    assert selected is manipulated_open_jar
    assert receipt["candidate_count"] == 2
    assert receipt["eligible_candidate_count"] == 1
    assert receipt["fail_closed"] is False


def test_first_person_opening_fails_closed_without_spatial_candidate():
    actor = {
        "class_name": "gloved_hand",
        "confidence": 0.56,
        "xyxy_norm": [0.30, 0.49, 0.48, 0.72],
    }
    background_open_vial = {
        "class_name": "container",
        "confidence": 0.41,
        "xyxy_norm": [0.44, 0.50, 0.49, 0.61],
    }

    selected, receipt = _select_state_container_candidate(
        [background_open_vial],
        [actor],
        view_role="First-Person",
        state_direction="opening",
        maximum_actor_gap=0.08,
    )

    assert selected is None
    assert receipt["fail_closed"] is True


def test_closing_container_rejects_higher_confidence_background_bottle():
    actor = {
        "class_name": "gloved_hand",
        "confidence": 0.8,
        "xyxy_norm": [0.30, 0.45, 0.50, 0.75],
    }
    background = {
        "class_name": "container",
        "confidence": 0.95,
        "xyxy_norm": [0.75, 0.10, 0.90, 0.40],
    }
    manipulated = {
        "class_name": "reagent_bottle",
        "confidence": 0.30,
        "xyxy_norm": [0.45, 0.55, 0.58, 0.88],
    }

    selected, receipt = _select_state_container_candidate(
        [background, manipulated],
        [actor],
        view_role="First-Person",
        state_direction="closing",
        maximum_actor_gap=0.08,
    )

    assert selected is manipulated
    assert receipt["eligible_candidate_count"] == 1
    assert receipt["fail_closed"] is False


def test_closing_container_prefers_object_adjacent_to_active_closure():
    actor = {
        "class_name": "gloved_hand",
        "confidence": 0.8,
        "xyxy_norm": [0.30, 0.40, 0.70, 0.85],
    }
    closure = {
        "class_name": "bottle_cap",
        "confidence": 0.6,
        "xyxy_norm": [0.34, 0.50, 0.42, 0.59],
    }
    background_touching_hand = {
        "class_name": "container",
        "confidence": 0.95,
        "xyxy_norm": [0.58, 0.48, 0.68, 0.72],
    }
    manipulated = {
        "class_name": "reagent_bottle",
        "confidence": 0.30,
        "xyxy_norm": [0.33, 0.56, 0.44, 0.88],
    }

    selected, receipt = _select_state_container_candidate(
        [background_touching_hand, manipulated],
        [actor],
        view_role="First-Person",
        state_direction="closing",
        maximum_actor_gap=0.08,
        closure_boxes=[closure],
        maximum_closure_gap=0.03,
    )

    assert selected is manipulated
    assert receipt["closure_candidate_count"] == 1
    assert receipt["selected_closure_gap_norm"] == 0.0


def test_opening_container_accepts_cap_held_away_from_bottle_mouth():
    actor = {
        "class_name": "gloved_hand",
        "confidence": 0.8,
        "xyxy_norm": [0.30, 0.40, 0.70, 0.85],
    }
    held_cap = {
        "class_name": "bottle_cap",
        "confidence": 0.6,
        "xyxy_norm": [0.32, 0.48, 0.40, 0.57],
    }
    opened_bottle = {
        "class_name": "reagent_bottle",
        "confidence": 0.30,
        "xyxy_norm": [0.58, 0.55, 0.69, 0.88],
    }

    selected, receipt = _select_state_container_candidate(
        [opened_bottle],
        [actor],
        view_role="Third-Person",
        state_direction="opening",
        maximum_actor_gap=0.08,
        closure_boxes=[held_cap],
        maximum_closure_gap=0.03,
    )

    assert selected is opened_bottle
    assert receipt["eligible_candidate_count"] == 1
    assert receipt["rule"] == "actor_contact_with_independently_held_closure"
    assert receipt["selected_closure_gap_norm"] > 0.03
    assert receipt["fail_closed"] is False


def test_active_pipette_selection_rejects_background_and_hand_shaped_boxes():
    actor = {
        "class_name": "gloved_hand",
        "confidence": 0.90,
        "xyxy_norm": [0.70, 0.35, 0.85, 0.60],
    }
    background_rack_pipette = {
        "class_name": "pipette",
        "confidence": 0.95,
        "xyxy_norm": [0.20, 0.65, 0.24, 0.95],
    }
    hand_shaped_false_positive = {
        "class_name": "pipette",
        "confidence": 0.80,
        "xyxy_norm": [0.72, 0.38, 0.84, 0.55],
    }
    active_handheld_pipette = {
        "class_name": "pipette",
        "confidence": 0.16,
        "xyxy_norm": [0.64, 0.40, 0.72, 0.68],
    }

    selected, receipt = _select_manipulated_object_candidate(
        [
            background_rack_pipette,
            hand_shaped_false_positive,
            active_handheld_pipette,
        ],
        [actor],
        canonical_class="pipette",
        maximum_actor_gap=0.08,
        pipette_minimum_aspect_ratio=1.7,
    )

    assert selected is active_handheld_pipette
    assert receipt["eligible_candidate_count"] == 1
    assert receipt["shape_rejected_count"] == 1
    assert receipt["contact_rejected_count"] == 1
    assert receipt["selected_actor_gap_norm"] == 0.0
    assert receipt["fail_closed"] is False


def test_active_pipette_selection_fails_closed_without_actor_contact():
    selected, receipt = _select_manipulated_object_candidate(
        [
            {
                "class_name": "pipette",
                "confidence": 0.99,
                "xyxy_norm": [0.05, 0.05, 0.08, 0.30],
            }
        ],
        [],
        canonical_class="pipette",
        maximum_actor_gap=0.08,
        pipette_minimum_aspect_ratio=1.7,
    )

    assert selected is None
    assert receipt["fail_closed"] is True


def test_manipulated_bottle_prefers_explicit_brown_semantic_prompt():
    actor = {
        "class_name": "gloved_hand",
        "confidence": 0.9,
        "xyxy_norm": [0.35, 0.40, 0.55, 0.85],
    }
    unrelated_clear_bottle = {
        "class_name": "reagent_bottle",
        "confidence": 0.92,
        "xyxy_norm": [0.50, 0.42, 0.63, 0.80],
        "grounding_prompt": "glass reagent bottle",
    }
    operated_brown_bottle = {
        "class_name": "reagent_bottle",
        "confidence": 0.38,
        "xyxy_norm": [0.39, 0.60, 0.51, 0.95],
        "grounding_prompt": "brown reagent bottle",
    }

    selected, receipt = _select_manipulated_object_candidate(
        [unrelated_clear_bottle, operated_brown_bottle],
        [actor],
        canonical_class="reagent_bottle",
        maximum_actor_gap=0.08,
        pipette_minimum_aspect_ratio=1.7,
        preferred_grounding_terms=("brown", "amber"),
    )

    assert selected is operated_brown_bottle
    assert receipt["preferred_eligible_candidate_count"] == 1
    assert receipt["selected_grounding_prompt"] == "brown reagent bottle"


def test_grounded_actor_filter_rejects_oversized_weak_hand_union():
    compact_hand = {
        "class_name": "gloved_hand",
        "confidence": 0.28,
        "xyxy_norm": [0.38, 0.62, 0.52, 0.84],
    }
    second_hand = {
        "class_name": "gloved_hand",
        "confidence": 0.25,
        "xyxy_norm": [0.42, 0.74, 0.70, 0.99],
    }
    oversized_union = {
        "class_name": "gloved_hand",
        "confidence": 0.17,
        "xyxy_norm": [0.00, 0.58, 0.70, 1.00],
    }

    assert _filter_grounded_actor_boxes(
        [compact_hand, second_hand, oversized_union],
        maximum_area_norm=0.15,
        confidence_ratio=0.75,
    ) == [compact_hand, second_hand]


def test_grounding_dino_pipette_uses_detector_specific_shape_floor():
    actor = {
        "class_name": "gloved_hand",
        "confidence": 0.92,
        "xyxy_norm": [0.61, 0.10, 0.80, 0.46],
    }
    handheld_diagonal_box = {
        "class_name": "pipette",
        "confidence": 0.43,
        "xyxy_norm": [0.54, 0.18, 0.75, 0.38],
        "detector_source": "grounding_dino_base_key_frame_fallback",
        "image_aspect_ratio": 16.0 / 10.0,
    }

    selected, receipt = _select_manipulated_object_candidate(
        [handheld_diagonal_box],
        [actor],
        canonical_class="pipette",
        maximum_actor_gap=0.08,
        pipette_minimum_aspect_ratio=1.7,
        grounding_dino_pipette_minimum_aspect_ratio=1.4,
    )

    assert selected is handheld_diagonal_box
    assert receipt["grounding_dino_minimum_aspect_ratio"] == 1.4
    assert receipt["fail_closed"] is False


def test_relaxed_grounding_dino_pipette_shape_rejects_balance_box():
    actor = {
        "class_name": "gloved_hand",
        "confidence": 0.92,
        "xyxy_norm": [0.68, 0.20, 0.84, 0.60],
    }
    balance_shaped_false_positive = {
        "class_name": "pipette",
        "confidence": 0.42,
        "xyxy_norm": [0.38, 0.30, 0.63, 0.58],
        "detector_source": "grounding_dino_base_key_frame_fallback",
        "image_aspect_ratio": 1.6,
    }

    selected, receipt = _select_manipulated_object_candidate(
        [balance_shaped_false_positive],
        [actor],
        canonical_class="pipette",
        maximum_actor_gap=0.08,
        pipette_minimum_aspect_ratio=1.7,
        grounding_dino_pipette_minimum_aspect_ratio=1.4,
    )

    assert selected is None
    assert receipt["centre_rejected_count"] == 1
    assert receipt["fail_closed"] is True


def test_handheld_pipette_rejects_rack_instance_with_centre_outside_hand():
    actor = {
        "class_name": "gloved_hand",
        "confidence": 0.8,
        "xyxy_norm": [0.74, 0.25, 1.0, 0.90],
    }
    rack_pipette = {
        "class_name": "pipette",
        "confidence": 0.40,
        "xyxy_norm": [0.68, 0.25, 0.78, 0.58],
        "detector_source": "grounding_dino_base_key_frame_fallback",
        "image_aspect_ratio": 1.6,
    }

    selected, receipt = _select_manipulated_object_candidate(
        [rack_pipette],
        [actor],
        canonical_class="pipette",
        maximum_actor_gap=0.08,
        pipette_minimum_aspect_ratio=1.7,
        grounding_dino_pipette_minimum_aspect_ratio=1.4,
        minimum_confidence=0.1,
        pipette_center_must_overlap_actor=True,
    )

    assert selected is None
    assert receipt["centre_rejected_count"] == 1
    assert receipt["fail_closed"] is True


def test_handheld_pipette_rejects_actor_shaped_grounding_box():
    actor = {
        "class_name": "gloved_hand",
        "confidence": 0.92,
        "xyxy_norm": [0.70, 0.20, 0.85, 0.65],
    }
    actor_shaped_pipette = {
        "class_name": "pipette",
        "confidence": 0.49,
        "xyxy_norm": [0.70, 0.20, 0.85, 0.65],
        "detector_source": "grounding_dino_base_key_frame_fallback",
        "image_aspect_ratio": 16.0 / 9.0,
    }

    selected, receipt = _select_manipulated_object_candidate(
        [actor_shaped_pipette],
        [actor],
        canonical_class="pipette",
        maximum_actor_gap=0.08,
        pipette_minimum_aspect_ratio=1.7,
        grounding_dino_pipette_minimum_aspect_ratio=1.4,
        minimum_confidence=0.1,
        pipette_maximum_actor_iou=0.65,
    )

    assert selected is None
    assert receipt["actor_overlap_rejected_count"] == 1
    assert receipt["fail_closed"] is True


def test_state_container_expands_to_complete_overlapping_object():
    actor = {
        "class_name": "gloved_hand",
        "confidence": 0.56,
        "xyxy_norm": [0.30, 0.49, 0.48, 0.72],
    }
    bottle_mouth = {
        "class_name": "container",
        "confidence": 0.035,
        "xyxy_norm": [0.428, 0.702, 0.492, 0.813],
    }
    complete_bottle = {
        "class_name": "container",
        "confidence": 0.031,
        "xyxy_norm": [0.401, 0.701, 0.494, 0.891],
    }

    selected, receipt = _select_state_container_candidate(
        [bottle_mouth, complete_bottle],
        [actor],
        view_role="First-Person",
        state_direction="opening",
        maximum_actor_gap=0.08,
    )

    assert selected is complete_bottle
    assert receipt["extent_expansion_candidate_count"] == 2
