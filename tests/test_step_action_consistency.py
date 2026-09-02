from visioncortex.pipeline import (
    normalize_final_group_action_language,
    validate_final_step_action_consistency,
)
from visioncortex.schemas import (
    ActionType,
    EvidenceEvent,
    ExperimentGroup,
    ViewRole,
)


def _event() -> EvidenceEvent:
    return EvidenceEvent(
        event_id="EVT-1",
        action_type=ActionType.HAND_OBJECT_CONTACT,
        global_start_ms=1000,
        global_end_ms=2000,
        key_global_ms=1500,
        objects=["gloved_hand", "balance"],
        confidence=0.9,
        accepted=True,
        audit_reason="test",
        supporting_views=["fp", "tp"],
        supporting_roles=[ViewRole.FIRST_PERSON, ViewRole.THIRD_PERSON],
        candidates=[],
    )


def _group(step: str) -> ExperimentGroup:
    return ExperimentGroup(
        group_id="GROUP-1",
        continuity_type="independent",
        atomic_experiment_ids=["EXP-1"],
        global_start_ms=1000,
        global_end_ms=2000,
        participating_views=["fp", "tp"],
        first_person_view="fp",
        third_person_view="tp",
        continuity_reason="test",
        key_event_ids=["EVT-1"],
        model_understanding={
            "status": "completed",
            "steps": [
                {
                    "step_index": 1,
                    "current_step": step,
                    "physical_change": "",
                }
            ],
        },
    )


def test_rejects_panel_claim_without_confirmed_panel_event():
    report = validate_final_step_action_consistency(
        [_group("手按压电子天平面板确认读数")], [_event()]
    )

    assert report["passed"] is False
    assert report["violations"][0]["unsupported_action_type"] == (
        "device_panel_operation"
    )


def test_allows_lower_level_contact_claim():
    report = validate_final_step_action_consistency(
        [_group("戴手套的手接触电子天平外壳")], [_event()]
    )

    assert report["passed"] is True


def test_pipette_object_name_does_not_assert_liquid_movement():
    report = validate_final_step_action_consistency(
        [
            _group(
                "戴手套的手接触并移动移液枪，移液器架保持静止，"
                "移液相关器材调整完成"
            )
        ],
        [_event()],
    )

    assert report["passed"] is True


def test_pipette_tip_object_name_does_not_assert_liquid_movement():
    report = validate_final_step_action_consistency(
        [_group("戴手套的手接触移液吸头并调整离心管架位置")],
        [_event()],
    )

    assert report["passed"] is True


def test_pipetting_operation_claim_requires_operational_event():
    report = validate_final_step_action_consistency(
        [_group("使用移液器执行移液操作")],
        [_event()],
    )

    assert report["passed"] is False
    assert report["violations"][0]["unsupported_action_type"] == (
        "pipette_transfer_operation"
    )


def test_posture_only_word_collision_is_narrowly_defunctionalized():
    group = _group("试剂瓶在台面上发生移位倾倒")

    receipt = normalize_final_group_action_language([group], [_event()])
    report = validate_final_step_action_consistency([group], [_event()])

    assert receipt["correction_count"] == 1
    assert group.model_understanding["steps"][0]["current_step"] == (
        "试剂瓶在台面上发生移位且姿态改变"
    )
    assert report["passed"] is True


def test_functional_pouring_claim_is_not_sanitized():
    group = _group("将液体倾倒入目标容器")

    receipt = normalize_final_group_action_language([group], [_event()])
    report = validate_final_step_action_consistency([group], [_event()])

    assert receipt["correction_count"] == 0
    assert report["passed"] is False


def test_explicit_liquid_uncertainty_is_not_a_positive_claim():
    report = validate_final_step_action_consistency(
        [
            _group(
                "执行源到目标移液器操作，移液器在容器间移动，"
                "液体转移状态不可确认"
            )
        ],
        [
            _event().model_copy(
                update={"action_type": ActionType.PIPETTE_TRANSFER_OPERATION}
            )
        ],
    )

    assert report["passed"] is True


def test_unproven_clean_water_pipetting_title_is_safely_defunctionalized():
    group = _group("执行源到目标的移液器操作，液体状态不可确认")
    group.experiment_name = "移液器吸头装配与清水移液实验"
    group.experiment_name_en = (
        "Pipette-Tip-Assembly-and-Clean-Water-Pipetting-Experiment"
    )
    event = _event().model_copy(
        update={"action_type": ActionType.PIPETTE_TRANSFER_OPERATION}
    )

    before = validate_final_step_action_consistency([group], [event])
    receipt = normalize_final_group_action_language([group], [event])
    after = validate_final_step_action_consistency([group], [event])

    assert before["passed"] is False
    assert group.experiment_name == (
        "移液器吸头装配与清水相关移液器操作实验"
    )
    assert group.experiment_name_en == (
        "Pipette-Tip-Assembly-and-Clean-Water-Related-Pipette-Operation-Experiment"
    )
    assert receipt["correction_count"] == 2
    assert after["passed"] is True


def test_explicit_no_liquid_transfer_is_not_a_positive_claim():
    report = validate_final_step_action_consistency(
        [_group("移液器位置发生改变，无液体转移发生")],
        [_event()],
    )

    assert report["passed"] is True


def test_aseptic_liquid_transfer_is_still_a_positive_claim():
    report = validate_final_step_action_consistency(
        [_group("执行无菌液体转移")],
        [_event()],
    )

    assert report["passed"] is False
    assert report["violations"][0]["unsupported_action_type"] == "liquid_movement"


def test_coordinated_no_action_list_does_not_create_positive_claims():
    report = validate_final_step_action_consistency(
        [
            _group(
                "无经确认的液体移动、容器开合、设备面板操作或物体移动动作发生"
            )
        ],
        [_event()],
    )

    assert report["passed"] is True


def test_unconfirmed_other_action_list_does_not_create_positive_claims():
    report = validate_final_step_action_consistency(
        [
            _group(
                "除此之外未确认存在其他移液、设备面板操作、"
                "液体移动或容器状态变化动作"
            )
        ],
        [_event()],
    )

    assert report["passed"] is True


def test_unproven_coordinated_action_list_does_not_create_positive_claims():
    report = validate_final_step_action_consistency(
        [
            _group(
                "无法证实存在拿取物体、旋盖、液体转移等实验操作"
            )
        ],
        [_event()],
    )

    assert report["passed"] is True


def test_unconfirmed_occurred_action_list_does_not_create_positive_claims():
    report = validate_final_step_action_consistency(
        [
            _group(
                "全程未确认发生容器开合、设备面板操作、液体移动、"
                "移液转移或物体移动等实验操作"
            )
        ],
        [_event()],
    )

    assert report["passed"] is True


def test_unobserved_action_list_does_not_create_positive_claims():
    report = validate_final_step_action_consistency(
        [
            _group(
                "全程未观测到设备面板操作、液体移动、移液转移或"
                "手与其他物体接触的已确认操作"
            )
        ],
        [_event()],
    )

    assert report["passed"] is True


def test_finally_audited_and_actually_unobserved_lists_are_not_positive_claims():
    report = validate_final_step_action_consistency(
        [
            _group(
                "无经最终审核确认的容器开合、液体移动、设备操作或移液；"
                "片段内未观察到实际液体转移、设备面板操作或完整开合流程"
            )
        ],
        [_event()],
    )

    assert report["passed"] is True


def test_unconfirmed_balance_readout_slash_action_list_is_not_positive_claims():
    report = validate_final_step_action_consistency(
        [
            _group(
                "全程仅可确认手部接触天平、容器与移液器；"
                "未确认天平读数/面板操作、容器开合、液体转移或完整移液流程。"
            )
        ],
        [_event()],
    )

    assert report["passed"] is True


def test_unoccurred_proven_action_list_does_not_create_positive_claims():
    report = validate_final_step_action_consistency(
        [
            _group(
                "未发生经证实的容器开合、液体移动、设备面板操作、"
                "手物接触移物或移液操作"
            )
        ],
        [_event()],
    )

    assert report["passed"] is True


def test_no_audited_event_support_list_does_not_create_positive_claims():
    report = validate_final_step_action_consistency(
        [
            _group(
                "但无已审核通过的事件支持开盖、移液、物体移动、"
                "设备面板操作或液体移动等动作确实发生"
            )
        ],
        [_event()],
    )

    assert report["passed"] is True


def test_unproven_list_does_not_mask_later_positive_liquid_claim():
    report = validate_final_step_action_consistency(
        [
            _group(
                "无法证实存在旋盖、液体转移，但随后将液体转移到烧杯"
            )
        ],
        [_event()],
    )

    assert report["passed"] is False
    assert report["violations"][0]["unsupported_action_type"] == (
        "liquid_movement"
    )


def test_no_confirmed_action_list_is_not_treated_as_positive_claims():
    report = validate_final_step_action_consistency(
        [
            _group(
                "该阶段无已确认的容器开合、液体转移或设备操作动作"
            )
        ],
        [_event()],
    )

    assert report["passed"] is True


def test_failed_parenthetical_action_audit_is_not_positive_claims():
    report = validate_final_step_action_consistency(
        [
            _group(
                "所有指定的功能性操作类别（容器状态改变、设备面板操作、"
                "液体移动、移液操作）均未通过最终事件审计"
            )
        ],
        [_event()],
    )

    assert report["passed"] is True


def test_visible_panel_without_operation_is_not_an_operation_claim():
    report = validate_final_step_action_consistency(
        [_group("磁力搅拌器面板可见显示内容但无面板操作")],
        [_event()],
    )

    assert report["passed"] is True


def test_whether_cap_was_tightened_is_an_uncertainty_not_a_claim():
    report = validate_final_step_action_consistency(
        [
            _group(
                "瓶盖被移动至试剂瓶瓶口位置，瓶盖是否完成旋紧封合无法确认"
            )
        ],
        [_event()],
    )

    assert report["passed"] is True


def test_uncertain_tightening_does_not_mask_later_positive_tightening():
    report = validate_final_step_action_consistency(
        [
            _group(
                "瓶盖是否完成旋紧封合无法确认，随后完成旋紧"
            )
        ],
        [_event()],
    )

    assert report["passed"] is False
    assert report["violations"][0]["unsupported_action_type"] == (
        "container_state_change"
    )


def test_prior_no_action_does_not_mask_later_positive_panel_claim():
    report = validate_final_step_action_consistency(
        [_group("无液体转移发生后又进行设备面板操作")],
        [_event()],
    )

    assert report["passed"] is False
    assert report["violations"][0]["unsupported_action_type"] == (
        "device_panel_operation"
    )


def test_negated_and_positive_liquid_mentions_do_not_mask_each_other():
    report = validate_final_step_action_consistency(
        [_group("开始时未观察到液体转移，随后将液体转移到烧杯")],
        [_event()],
    )

    assert report["passed"] is False
    assert report["violations"][0]["unsupported_action_type"] == "liquid_movement"
