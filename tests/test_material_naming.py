from visioncortex.material_naming import key_material_semantic_name
from visioncortex.schemas import ActionType, EvidenceEvent


def event(action_type: ActionType, objects: list[str]) -> EvidenceEvent:
    return EvidenceEvent(
        event_id="EVT-000001",
        action_type=action_type,
        global_start_ms=1_000,
        global_end_ms=2_000,
        key_global_ms=1_500,
        objects=objects,
        confidence=0.9,
        accepted=True,
        audit_reason="test",
        supporting_views=[],
        supporting_roles=[],
        candidates=[],
    )


def test_liquid_name_uses_source_and_target_containers_not_tool():
    name = key_material_semantic_name(
        event(
            ActionType.LIQUID_MOVEMENT,
            ["pipette", "reagent_bottle", "tube"],
        )
    )

    assert name["file_stem"] == (
        "Transfer-Liquid-Reagent-Bottle-To-Tube_EVT-000001"
    )
    assert name["primary_object"] == "Reagent-Bottle"
    assert name["object_labels"] == [
        "Pipette",
        "Reagent-Bottle",
        "Tube",
    ]


def test_contact_name_translates_chinese_weighing_paper():
    name = key_material_semantic_name(
        event(
            ActionType.HAND_OBJECT_CONTACT,
            ["戴手套的手", "称量纸"],
        )
    )

    assert name["file_stem"] == (
        "Contact-Hand-With-Weighing-Paper_EVT-000001"
    )
    assert name["raw_object_labels"] == ["戴手套的手", "称量纸"]


def test_pipette_operation_name_does_not_claim_visible_liquid():
    name = key_material_semantic_name(
        event(
            ActionType.PIPETTE_TRANSFER_OPERATION,
            ["pipette", "tube", "beaker"],
        )
    )

    assert name["file_stem"] == (
        "Operate-Pipette-Tube-To-Beaker_EVT-000001"
    )
