import pytest

from visioncortex.pipeline import (
    normalize_final_group_action_language,
    refine_groups_from_final_events,
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


@pytest.mark.parametrize("text", [
    "全片段未看到烧杯被放下、倾倒或任何内容物转移完成。",
    "瓶盖未脱离，旋开或旋紧完成均未见。",
    "双手扶握瓶盖，未看到旋紧完成。",
    "清水标签瓶全程保持开盖，其紫色盖放在桌上未被操作。",
    "移液器末端伸入蓝色吸头盒内；烧杯与各瓶液面未见变化。",
])
def test_real_run_denials_and_separate_clauses_do_not_become_positive_actions(text):
    assert validate_final_step_action_consistency([_group(text)], [_event()])["passed"] is True


@pytest.mark.parametrize("text", [
    "未看到旋开，但随后旋紧瓶盖。",
    "旋开瓶盖，随后旋紧完成未见。",
    "清水瓶保持开盖，操作者拧开另一只瓶。",
    "移液器末端伸入烧杯，完成移液操作。",
])
def test_a_nearby_denial_or_static_state_does_not_hide_a_positive_action(text):
    assert validate_final_step_action_consistency([_group(text)], [_event()])["passed"] is False


def test_single_container_contact_is_not_a_source_to_target_transfer_claim():
    text = "移液器吸头伸入烧杯并提离，未见吸头进入第二个容器。"
    assert validate_final_step_action_consistency([_group(text)], [_event()])["passed"] is True
    text = "移液器从源烧杯到目标试剂瓶进行转移。"
    assert validate_final_step_action_consistency([_group(text)], [_event()])["passed"] is False


def test_explicit_before_state_and_unfinished_operation_are_not_completed_actions():
    group = _group("盖合/旋紧动作正在进行，未看到旋紧完成。")
    group.model_understanding["steps"][0]["physical_change"] = "各瓶开盖、瓶盖置于台面 → 各瓶仍开盖"
    assert validate_final_step_action_consistency([group], [_event()])["passed"] is True
    group.model_understanding["steps"][0]["physical_change"] = "操作者将瓶开盖 → 瓶口敞开"
    assert validate_final_step_action_consistency([group], [_event()])["passed"] is False


def test_final_group_steps_are_rebuilt_without_a_second_model_pass():
    event = _event()
    event.model_understanding = {
        "status": "completed",
        "current_step": "戴手套的手抓取离心管",
        "next_step": "移动离心管",
        "next_step_evidence": {
            "status": "inferred",
            "reason": "当前抓取姿态支持谨慎预测",
            "evidence_event_ids": [event.event_id],
        },
        "physical_change": {"before": "未抓取", "after": "已抓取"},
        "confidence": 0.91,
        "uncertainties": [],
    }
    group = _group("旧的未审核步骤")
    group.model_understanding["usage"] = {"total_tokens": 100}

    refine_groups_from_final_events([group], [event])

    assert group.model_understanding["refinement_model_call_count"] == 0
    assert group.model_understanding["operation_coverage"]["all_operator_steps_proven"] is False
    assert group.model_understanding["usage"] == {"total_tokens": 100}
    assert group.model_understanding["steps"] == [
        {
            "step_index": 1,
            "start_global_ms": 1000.0,
            "end_global_ms": 2000.0,
            "current_step": "戴手套的手抓取离心管",
            "next_step": "移动离心管",
            "next_step_status": "inferred",
            "next_step_evidence": {
                "status": "inferred",
                "reason": "当前抓取姿态支持谨慎预测",
                "evidence_event_ids": ["EVT-1"],
            },
            "supporting_event_ids": ["EVT-1"],
            "objects": ["gloved_hand", "balance"],
            "physical_change": "未抓取 → 已抓取",
            "supporting_views": ["fp", "tp"],
            "confidence": 0.91,
            "action_type": "hand_object_contact",
            "event_id": "EVT-1",
        }
    ]


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


def test_long_unobserved_action_enumeration_is_not_positive_claims():
    report = validate_final_step_action_consistency(
        [
            _group(
                "整个有界段内未观察到包装开启、内容物取出、液体或固体转移、"
                "天平面板操作或可读称量读数"
            )
        ],
        [_event()],
    )

    assert report["passed"] is True


def test_no_clear_readout_is_not_a_positive_panel_claim():
    report = validate_final_step_action_consistency(
        [_group("截至片段结束，天平显示屏无清晰可读读数")],
        [_event()],
    )

    assert report["passed"] is True


@pytest.mark.parametrize("text", [
    "手持续握持已开盖的棕色样品瓶，候选开盖动作未确认",
    "片段开始：瓶已开盖、瓶盖仰放 → 片段尾部：瓶仍开盖、瓶盖仍仰放",
    "未见取放、开盖或液体转移",
    "全程无拧盖/开盖动作，瓶保持开启状态",
    "无试管被取出、无盖被拧开、无可见液体移动",
])
def test_existing_cap_state_and_scoped_denials_are_not_action_claims(text):
    assert validate_final_step_action_consistency([_group(text)], [_event()])["passed"]


@pytest.mark.parametrize("text", [
    "手持已开盖的瓶子，随后开盖",
    "操作者已开盖",
    "未见取放、开盖或液体转移，但随后液体转移至烧杯",
    "全程无拧盖/开盖动作，随后完成开盖",
])
def test_static_states_and_denials_do_not_hide_later_actions(text):
    assert not validate_final_step_action_consistency([_group(text)], [_event()])["passed"]


def test_final_title_drops_rejected_candidate_action_and_preserves_original_receipt():
    group = _group("手接触称量纸")
    group.experiment_name = "天平面板操作与称量纸拿取实验"
    group.model_understanding["experiment_name"] = group.experiment_name
    event = _event()
    event.model_understanding = {"status": "completed", "current_step": "手接触称量纸"}
    refine_groups_from_final_events([group], [event])
    assert group.experiment_name == "手部与物体接触实验片段"
    assert "面板操作" in group.model_understanding["pre_curation_understanding"]["experiment_name"]
    assert validate_final_step_action_consistency([group], [event])["passed"]


def test_unconfirmed_long_list_with_open_close_is_not_positive_claims():
    report = validate_final_step_action_consistency(
        [
            _group(
                "由于未确认面板操作、容器状态变化和液体转移事件，"
                "也无法确认称量读数、开合盖或物料转移是否完成"
            )
        ],
        [_event()],
    )

    assert report["passed"] is True


def test_metrics_count_original_group_call_once_after_deterministic_curation(default_config):
    from visioncortex.pipeline import EvidencePipeline
    group = _group("observed")
    group.model_understanding.update(request_id="receipt-1", usage={"total_tokens": 100})
    refine_groups_from_final_events([group], [])
    metrics = EvidencePipeline(default_config)._metrics([], [group])
    assert metrics["tokens"]["experiment_groups"]["total_tokens"] == 100
    assert metrics["tokens"]["experiment_groups"]["call_count"] == 1
    # A genuine second model pass still counts as a second call.
    group.model_understanding.update(refinement_pass="model", request_id="receipt-2", usage={"total_tokens": 60})
    metrics = EvidencePipeline(default_config)._metrics([], [group])
    assert metrics["tokens"]["experiment_groups"]["total_tokens"] == 160
    assert metrics["tokens"]["experiment_groups"]["call_count"] == 2


def test_operation_title_is_preserved_and_cannot_bypass_action_evidence():
    event = _event()
    event.model_understanding = {"status": "completed", "operation_title": "旋开瓶盖",
                                 "current_step": "手接触瓶盖"}
    group = _group("手接触瓶盖")
    refine_groups_from_final_events([group], [event])
    assert group.model_understanding["steps"][0]["operation_title"] == "旋开瓶盖"
    report = validate_final_step_action_consistency([group], [event])
    assert not report["passed"]
    assert any(item["field"] == "operation_title" for item in report["violations"])
