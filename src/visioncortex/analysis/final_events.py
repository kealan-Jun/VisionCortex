"""Final evidence synchronization after semantic curation.

These deterministic rules reconcile final event state, segment membership and
step prose. They perform no model/media/queue work and preserve event receipts.
"""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Sequence

from ..action_semantics import attach_action_observability
from ..action_state_machine import build_event_state_receipt
from ..material_naming import ACTION_LABELS_ZH
from ..ordering import event_sort_key
from ..schemas import (
    ActionType,
    EvidenceEvent,
    ExperimentGroup,
    ExperimentSegment,
    ViewRole,
    event_is_formal,
)

FINAL_STEP_ACTION_PATTERNS: dict[str, tuple[str, ...]] = {
    "device_panel_operation": (
        r"(?:操作|点击|触碰|按下|调节|读取).{0,4}面板",
        r"面板.{0,4}(?:操作|点击|触碰|按下|调节|读取)",
        r"(?:按下|点击|触碰|操作).{0,3}按钮",
        r"(?:按下|点击|触碰|操作).{0,3}按键",
        r"读数",
        r"数值.{0,4}(?:确认|读取)",
    ),
    "liquid_movement": (
        r"吸液",
        r"排液",
        r"加液",
        r"液体转移",
        r"(?:清水|液体|试剂|溶液).{0,4}移液(?!器)",
        r"倾倒",
    ),
    "pipette_transfer_operation": (
        r"移液操作",
        r"移液流程",
        r"源[^，。；！？]{0,12}目标[^，。；！？]{0,12}移液器",
        r"移液器[^，。；！？]{0,12}源[^，。；！？]{0,12}(?:到|至|进入)[^，。；！？]{0,8}目标",
    ),
    "container_state_change": (
        r"开盖",
        r"合盖",
        r"旋开",
        r"旋紧",
        r"拧开",
        r"拧紧",
        r"盖回",
    ),
}

_SAFE_POSTURE_ONLY_REWRITES: tuple[tuple[re.Pattern[str], str], ...] = (
    # These phrases describe an object's visible posture while accidentally
    # using the same Chinese verb as a functional liquid-pouring claim.  The
    # rewrite is deliberately narrow: claims such as "将液体倾倒入容器" remain
    # untouched and therefore still fail the consistency gate.
    (re.compile(r"发生移位倾倒"), "发生移位且姿态改变"),
    (re.compile(r"移位倾倒"), "移位且姿态改变"),
    (re.compile(r"倾倒姿态"), "倾斜姿态"),
)


def _explicit_container_state_direction(
    current_step: str, physical_change: str
) -> str | None:
    """Recognize only an explicit model-described closure state transition."""

    current = str(current_step or "")
    physical = str(physical_change or "")
    text = f"{current} {physical}".lower()
    closure_named = bool(
        re.search(r"瓶盖|管盖|离心管盖|bottle[_ -]?cap|tube[_ -]?cap", text)
    )
    container_named = bool(
        re.search(r"试剂瓶|样品瓶|瓶身|离心管|容器|bottle|tube|container", text)
    )
    if not closure_named or not container_named:
        return None
    explicit_before_after = bool(
        re.search(
            r"(?:从|由).{0,18}(?:密封|闭合|关闭|盖合|有盖).{0,18}"
            r"(?:变为|变成|转为|成为).{0,18}(?:开口|打开|开启|无盖|分离)",
            physical,
        )
    )
    explicit_open_action = bool(
        re.search(
            r"(?:旋开|拧开|打开|开启|取下|移除).{0,8}(?:瓶盖|管盖)"
            r"|(?:瓶盖|管盖).{0,8}(?:旋开|拧开|打开|开启|取下|移除)"
            r"|(?:瓶盖|管盖).{0,8}(?:与|从).{0,8}(?:瓶身|容器).{0,6}分离",
            text,
        )
    )
    if explicit_before_after or explicit_open_action:
        return "opening"
    explicit_close_before_after = bool(
        re.search(
            r"(?:从|由).{0,18}(?:开口|打开|开启|无盖|分离).{0,18}"
            r"(?:变为|变成|转为|成为).{0,18}(?:密封|闭合|关闭|盖合|有盖)",
            physical,
        )
    )
    explicit_close_action = bool(
        re.search(
            r"(?:旋紧|拧紧|盖上|盖回|关闭|封闭).{0,8}(?:瓶盖|管盖|瓶|管)?"
            r"|(?:瓶盖|管盖).{0,8}(?:旋紧|拧紧|盖上|盖回|关闭|封闭)",
            text,
        )
    )
    return "closing" if explicit_close_before_after or explicit_close_action else None


def _semantic_state_participants(text: str) -> list[str]:
    normalized = str(text or "").lower()
    actor = "gloved_hand" if re.search(r"手套|glov(?:e|ed)", normalized) else "hand"
    if re.search(r"离心管|试管|tube", normalized):
        container = "tube"
        closure = "tube_cap"
    elif re.search(r"样品瓶|sample[_ -]?bottle", normalized):
        container = "sample_bottle"
        closure = "bottle_cap"
    elif re.search(r"试剂瓶|reagent[_ -]?bottle", normalized):
        container = "reagent_bottle"
        closure = "bottle_cap"
    else:
        container = "container"
        closure = "bottle_cap"
    return [actor, container, closure]


def _recover_group_storyboard_state_events(
    groups: Sequence[ExperimentGroup],
    segments: Sequence[ExperimentSegment],
    events: list[EvidenceEvent],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Create provisional state candidates from a real full-group Ark review.

    This is recall-only.  The recovered event is still materialized and sent
    through the independent event-level Ark action-proof contract; semantic
    curation removes it if before/after state proof is not confirmed.
    """

    minimum_group_confidence = float(
        config.get("mllm", {}).get("group_state_recall_minimum_confidence", 0.55)
    )
    minimum_step_confidence = float(
        config.get("mllm", {}).get("group_state_recall_step_minimum_confidence", 0.55)
    )
    next_number = (
        max(
            [
                int(match.group(1))
                for event in events
                if (match := re.fullmatch(r"EVT-(\d+)", event.event_id))
            ],
            default=0,
        )
        + 1
    )
    by_segment = {segment.segment_id: segment for segment in segments}
    recovered: list[dict[str, Any]] = []
    for group in groups:
        understanding = group.model_understanding or {}
        if (
            understanding.get("status") != "completed"
            or float(understanding.get("confidence") or 0.0) < minimum_group_confidence
        ):
            continue
        group_segments = [
            by_segment[segment_id]
            for segment_id in group.atomic_experiment_ids
            if segment_id in by_segment
        ]
        group_event_ids = {
            event_id for segment in group_segments for event_id in segment.event_ids
        }
        existing_state_events = [
            event
            for event in events
            if event.event_id in group_event_ids
            and event_is_formal(event)
            and event.action_type == ActionType.CONTAINER_STATE_CHANGE
        ]
        for step in understanding.get("steps") or []:
            if not isinstance(step, dict):
                continue
            step_confidence = float(step.get("confidence") or 0.0)
            if step_confidence < minimum_step_confidence:
                continue
            current_step = str(step.get("current_step") or "")
            physical_change = str(step.get("physical_change") or "")
            direction = _explicit_container_state_direction(
                current_step, physical_change
            )
            if direction is None:
                continue
            supporting_views = [
                str(item) for item in step.get("supporting_views") or [] if str(item)
            ]
            required_views = {
                group.first_person_view,
                group.third_person_view,
            }
            if not required_views.issubset(set(supporting_views)):
                continue
            start_ms = max(
                float(group.global_start_ms),
                float(step.get("start_global_ms") or group.global_start_ms),
            )
            end_ms = min(
                float(group.global_end_ms),
                float(step.get("end_global_ms") or group.global_end_ms),
            )
            if end_ms <= start_ms:
                continue
            if any(
                min(end_ms, event.global_end_ms) > max(start_ms, event.global_start_ms)
                for event in existing_state_events
            ):
                continue
            target_segment = max(
                group_segments,
                key=lambda segment: max(
                    0.0,
                    min(end_ms, segment.global_end_ms)
                    - max(start_ms, segment.global_start_ms),
                ),
                default=None,
            )
            if target_segment is None:
                continue
            event_id = f"EVT-{next_number:06d}"
            next_number += 1
            participants = _semantic_state_participants(
                f"{current_step} {physical_change}"
            )
            event = EvidenceEvent(
                event_id=event_id,
                action_type=ActionType.CONTAINER_STATE_CHANGE,
                global_start_ms=start_ms,
                global_end_ms=end_ms,
                key_global_ms=start_ms + (end_ms - start_ms) * 0.80,
                objects=participants,
                confidence=min(
                    float(understanding.get("confidence") or 0.0),
                    step_confidence,
                ),
                accepted=True,
                formal_admission_status="provisional",
                audit_reason=(
                    "完整组故事板视觉模型显式提出容器开合状态转换；"
                    "仅作为召回候选，必须通过独立事件级视觉模型状态证明"
                ),
                supporting_views=sorted(required_views),
                supporting_roles=[
                    ViewRole.FIRST_PERSON,
                    ViewRole.THIRD_PERSON,
                ],
                candidates=[],
                uncertainty=["该候选由组级语义召回产生，不代表事件级状态证明已通过"],
                observability={
                    "semantic_recall_admission": {
                        "schema_version": (
                            "visioncortex-group-storyboard-state-recall/1"
                        ),
                        "mode": "full_group_storyboard_explicit_state_transition",
                        "mandatory_semantic_review": True,
                        "candidate_action_directly_confirmed": False,
                        "source_group_id": group.group_id,
                        "transition_direction": direction,
                        "source_step_confidence": step_confidence,
                    }
                },
            )
            attach_action_observability([event])
            event.state_machine = build_event_state_receipt(event, config)
            event.state_machine["publication"] = {
                "status": "provisional_semantic_review",
                "suppressed_by_event_id": None,
                "reason": "independent_event_level_state_proof_required",
            }
            events.append(event)
            existing_state_events.append(event)
            target_segment.event_ids = [
                item.event_id
                for item in sorted(
                    [
                        candidate
                        for candidate in events
                        if candidate.event_id
                        in {
                            *target_segment.event_ids,
                            event_id,
                        }
                    ],
                    key=event_sort_key,
                )
            ]
            target_segment.micro_segments.append(
                {
                    "micro_segment_id": (
                        f"MICRO-{target_segment.segment_id}-SEM-{event_id}"
                    ),
                    "start_global_ms": start_ms,
                    "end_global_ms": end_ms,
                    "action_type": ActionType.CONTAINER_STATE_CHANGE.value,
                    "objects": participants,
                    "evidence_event_id": event_id,
                    "view_alignment_state": "aligned",
                    "next_event_id": None,
                    "uncertainty": list(event.uncertainty),
                    "source": "full_group_storyboard_semantic_recall",
                }
            )
            recovered.append(
                {
                    "event_id": event_id,
                    "group_id": group.group_id,
                    "segment_id": target_segment.segment_id,
                    "transition_direction": direction,
                    "global_start_ms": start_ms,
                    "global_end_ms": end_ms,
                    "participant_objects": participants,
                    "source_step_confidence": step_confidence,
                    "group_confidence": float(understanding.get("confidence") or 0.0),
                    "supporting_views": sorted(required_views),
                    "independent_event_level_review_required": True,
                }
            )
    return recovered


def normalize_final_group_action_language(
    groups: Sequence[ExperimentGroup],
    key_events: Sequence[EvidenceEvent],
) -> dict[str, Any]:
    """Conservatively de-functionalize only known posture-word collisions.

    This is not a general claim sanitizer.  It records every exact rewrite and
    leaves all other unsupported high-risk action words in place so the normal
    fail-closed consistency check rejects the archive.
    """

    event_by_id = {event.event_id: event for event in key_events}
    corrections: list[dict[str, Any]] = []
    for group in groups:
        confirmed_actions = {
            event_by_id[event_id].action_type.value
            for event_id in group.key_event_ids
            if event_id in event_by_id
        }
        if ActionType.LIQUID_MOVEMENT.value in confirmed_actions:
            continue
        understanding = deepcopy(group.model_understanding or {})
        if ActionType.PIPETTE_TRANSFER_OPERATION.value in confirmed_actions:
            identity_fields = (
                ("group.experiment_name", group.experiment_name),
                ("understanding.experiment_name", understanding.get("experiment_name")),
            )
            for field, value in identity_fields:
                before = str(value or "")
                after = re.sub(
                    r"(清水|液体|试剂|溶液)移液",
                    r"\1相关移液器操作",
                    before,
                )
                if after == before:
                    continue
                if field == "group.experiment_name":
                    group.experiment_name = after
                else:
                    understanding["experiment_name"] = after
                corrections.append(
                    {
                        "group_id": group.group_id,
                        "field": field,
                        "before": before,
                        "after": after,
                        "reason": "liquid_unproven_but_pipette_operation_confirmed",
                    }
                )
            english_fields = (
                ("group.experiment_name_en", group.experiment_name_en),
                (
                    "understanding.experiment_name_en",
                    understanding.get("experiment_name_en"),
                ),
            )
            for field, value in english_fields:
                before = str(value or "")
                after = re.sub(
                    r"(Clean-Water|Liquid|Reagent|Solution)-Pipetting",
                    r"\1-Related-Pipette-Operation",
                    before,
                    flags=re.IGNORECASE,
                )
                if after == before:
                    continue
                if field == "group.experiment_name_en":
                    group.experiment_name_en = after
                else:
                    understanding["experiment_name_en"] = after
                corrections.append(
                    {
                        "group_id": group.group_id,
                        "field": field,
                        "before": before,
                        "after": after,
                        "reason": "liquid_unproven_but_pipette_operation_confirmed",
                    }
                )
        for step in understanding.get("steps") or []:
            for field in ("current_step", "next_step", "physical_change"):
                before = str(step.get(field) or "")
                after = before
                for pattern, replacement in _SAFE_POSTURE_ONLY_REWRITES:
                    after = pattern.sub(replacement, after)
                if after == before:
                    continue
                step[field] = after
                corrections.append(
                    {
                        "group_id": group.group_id,
                        "step_index": step.get("step_index"),
                        "field": field,
                        "before": before,
                        "after": after,
                        "reason": "liquid_action_absent_posture_only_defunctionalization",
                    }
                )
        if corrections_for_group := [
            item for item in corrections if item["group_id"] == group.group_id
        ]:
            understanding["fail_closed_language_normalization"] = {
                "schema_version": "visioncortex-final-language-normalization/1",
                "policy": "narrow posture-only rewrite; all other unsupported claims remain fatal",
                "corrections": corrections_for_group,
            }
            group.model_understanding = understanding
    return {
        "schema_version": "visioncortex-final-language-normalization/1",
        "correction_count": len(corrections),
        "corrections": corrections,
    }


def refine_groups_from_final_events(
    groups: Sequence[ExperimentGroup],
    key_events: Sequence[EvidenceEvent],
) -> None:
    """Rebuild final group steps from adjudicated events without a second MLLM call.

    The initial group pass remains responsible for bounded-experiment naming and
    continuity. Event-level review is the stronger source for final action facts,
    so this pass deterministically replaces only the step narrative and keeps the
    initial model receipt and token usage available for audit.
    """

    from ..step_evidence import next_operation, timing_scope

    event_by_id = {
        event.event_id: event for event in key_events if event_is_formal(event)
    }
    for group in groups:
        previous = deepcopy(group.model_understanding or {})
        events = sorted(
            (
                event_by_id[event_id]
                for event_id in group.key_event_ids
                if event_id in event_by_id
            ),
            key=event_sort_key,
        )
        steps: list[dict[str, Any]] = []
        uncertainties = list(previous.get("uncertainties") or [])
        for index, event in enumerate(events, 1):
            understanding = dict(event.model_understanding or {})
            physical_change = dict(understanding.get("physical_change") or {})
            before = str(physical_change.get("before") or "")
            after = str(physical_change.get("after") or "")
            physical_change_text = (
                f"{before} → {after}" if before or after else "未观察到可确认的状态变化"
            )
            event_uncertainties = [
                str(item)
                for item in (
                    list(event.uncertainty)
                    + list(understanding.get("uncertainties") or [])
                )
                if item
            ]
            uncertainties.extend(event_uncertainties)
            steps.append(
                {
                    "step_index": index,
                    "start_global_ms": float(event.global_start_ms),
                    "end_global_ms": float(event.global_end_ms),
                    "current_step": str(
                        understanding.get("current_step")
                        or ACTION_LABELS_ZH[event.action_type.value]
                    ),
                    **next_operation(event, events, {event.event_id}),
                    "supporting_event_ids": [event.event_id],
                    "time_scope": timing_scope([event]),
                    "objects": list(event.objects),
                    "physical_change": physical_change_text,
                    "supporting_views": list(event.supporting_views),
                    "confidence": float(
                        understanding.get("confidence") or event.confidence
                    ),
                    "action_type": event.action_type.value,
                    "event_id": event.event_id,
                    **(
                        {"operation_title": understanding["operation_title"]}
                        if understanding.get("operation_title")
                        else {}
                    ),
                }
            )
        action_labels = [
            ACTION_LABELS_ZH[action_type]
            for action_type in dict.fromkeys(
                event.action_type.value for event in events
            )
        ]
        summary = (
            f"已整理 {len(events)} 条有事件证据支持的操作记录；"
            "具体过程、前后状态和后续判断见各步骤。"
            if events
            else "当前有界实验没有通过自动验收的关键动作。"
        )
        refined = dict(previous)
        refined.update(
            {
                "steps": steps,
                "overall_summary": summary,
                "uncertainties": sorted(set(uncertainties)),
                "pre_curation_understanding": previous,
                "refinement_pass": "deterministic_post_event_semantic_curation",
                "refinement_source": "final_adjudicated_key_events",
                "refinement_model_call_count": 0,
                "refinement_key_event_count": len(events),
                "operation_coverage": {
                    "status": "PARTIAL_EVIDENCE",
                    "basis": "adjudicated_event_intervals",
                    "all_operator_steps_proven": False,
                    "missing_gate": "real_video_operation_level_coverage_review",
                },
                "archive_folder_frozen_after_initial_materialization": True,
                "display_identity_refined_after_event_curation": False,
            }
        )
        group.model_understanding = refined
        from ..operation_review import coverage

        refined["operation_coverage"] = coverage(group.model_dump(mode="json"), steps)
        # A completed naming/step request is not an experiment-end receipt.
        # This applies to every future run, including profiles that skip the
        # dedicated boundary pass. Preserve the original response above.
        if group.completion_status != "observed_complete":
            refined["boundary_assessment"] = {
                **(refined.get("boundary_assessment") or {}),
                "end_complete": False,
                "localized_rescan_needed": True,
                "end_reason": group.completion_reason
                or "尚未取得实验完成的边界复核证据",
            }
        title_violations = [
            item
            for item in validate_final_step_action_consistency([group], events)[
                "violations"
            ]
            if item["field"] in {"experiment_name", "understanding.experiment_name"}
        ]
        if title_violations:
            # A candidate-stage title may claim an action removed by event
            # adjudication. Rebuild only that display title from final actions;
            # preserve the original model receipt and all material paths.
            title = (
                "、".join(action_labels) + "实验片段" if events else "待确认实验片段"
            )
            english_title = (
                "Reviewed Experiment Segment"
                if events
                else "Unconfirmed Experiment Segment"
            )
            refined["title_refinement"] = {
                "reason": "candidate_title_asserted_rejected_action",
                "previous_title": group.experiment_name,
                "previous_title_en": group.experiment_name_en,
                "final_title": title,
                "final_title_en": english_title,
                "unsupported_action_types": sorted(
                    {item["unsupported_action_type"] for item in title_violations}
                ),
            }
            group.experiment_name = title
            group.experiment_name_en = english_title
            refined["experiment_name"] = title
            refined["experiment_name_en"] = english_title
            refined["display_identity_refined_after_event_curation"] = True


def _synchronize_final_event_state_receipts(
    events: Sequence[EvidenceEvent], config: dict[str, Any]
) -> list[dict[str, Any]]:
    """Keep final state receipts aligned with participant key-frame selection.

    Key-frame selection is allowed to move the event peak inside the accepted
    bounds.  Rebuild only receipts whose peak or final action no longer agrees
    with the curated event, while retaining semantic-relabelling provenance and
    any component-publication decision from the original receipt.
    """

    repairs: list[dict[str, Any]] = []
    for event in events:
        previous = dict(event.state_machine or {})
        expected_action = (
            "liquid_transfer"
            if event.action_type == ActionType.LIQUID_MOVEMENT
            else event.action_type.value
        )
        expected_peak_us = round(event.key_global_ms * 1000.0)
        expected_object_classes = {
            str(item).strip().lower().replace("-", "_").replace(" ", "_")
            for item in event.objects
        } - {"hand", "gloved_hand", "lab_coat"}
        previous_object_classes = {
            str(item).strip().lower().replace("-", "_").replace(" ", "_")
            for item in (
                ((previous.get("object_identity") or {}).get("object_classes")) or []
            )
        }
        participant_identity_mismatch = bool(
            expected_object_classes != previous_object_classes
        )
        if (
            str(previous.get("action_type") or "") == expected_action
            and int(previous.get("peak_timestamp_us") or -1) == expected_peak_us
            and not participant_identity_mismatch
        ):
            continue
        rebuilt = build_event_state_receipt(event, config)
        derivation = dict(previous.get("derivation") or {})
        if derivation:
            derivation["timing_resynchronized_after_final_key_frame_selection"] = True
            rebuilt["derivation"] = derivation
            if derivation.get("source") == "semantic_relabel_confirmed_state_proof":
                for transition in rebuilt.get("transition_trace") or []:
                    transition["source"] = "semantic_relabel_confirmed_state_proof"
                identity = dict(rebuilt.get("object_identity") or {})
                identity["track_tokens"] = []
                identity["identity_status"] = "semantic_participant_classes"
                rebuilt["object_identity"] = identity
        if previous.get("publication"):
            rebuilt["publication"] = dict(previous["publication"])
        event.state_machine = rebuilt
        repairs.append(
            {
                "event_id": event.event_id,
                "previous_action_type": previous.get("action_type"),
                "final_action_type": rebuilt.get("action_type"),
                "previous_peak_timestamp_us": previous.get("peak_timestamp_us"),
                "final_peak_timestamp_us": rebuilt.get("peak_timestamp_us"),
                "previous_object_classes": sorted(previous_object_classes),
                "final_object_classes": sorted(expected_object_classes),
                "participant_identity_resynchronized": (participant_identity_mismatch),
            }
        )
    return repairs


def _synchronize_segments_with_final_key_events(
    segments: Sequence[ExperimentSegment],
    groups: Sequence[ExperimentGroup],
    events: Sequence[EvidenceEvent],
) -> list[dict[str, Any]]:
    """Remove rejected/duplicate CV hypotheses from final segment sidecars."""

    event_by_id = {event.event_id: event for event in events}
    group_by_id = {group.group_id: group for group in groups}
    receipts: list[dict[str, Any]] = []
    for segment in segments:
        group = group_by_id.get(str(segment.group_id or ""))
        if group is None:
            continue
        final_events = sorted(
            (
                event_by_id[event_id]
                for event_id in group.key_event_ids
                if event_id in event_by_id
                and event_by_id[event_id].global_end_ms >= segment.global_start_ms
                and event_by_id[event_id].global_start_ms <= segment.global_end_ms
            ),
            key=lambda event: (
                event.global_start_ms,
                event.key_global_ms,
                event.event_id,
            ),
        )
        previous_event_ids = list(segment.event_ids)
        previous_micro_count = len(segment.micro_segments)
        segment.event_ids = [event.event_id for event in final_events]
        synchronized_micro_segments: list[dict[str, Any]] = []
        for index, event in enumerate(final_events, start=1):
            synchronized_micro_segments.append(
                {
                    "micro_segment_id": (
                        f"MICRO-{segment.segment_id.removeprefix('EXP-')}-{index:04d}"
                    ),
                    "start_global_ms": float(event.global_start_ms),
                    "end_global_ms": float(event.global_end_ms),
                    "action_type": event.action_type.value,
                    "objects": list(event.objects),
                    "evidence_event_id": event.event_id,
                    "view_alignment_state": (
                        "aligned"
                        if {
                            role.value if isinstance(role, ViewRole) else str(role)
                            for role in event.supporting_roles
                        }
                        >= {
                            ViewRole.FIRST_PERSON.value,
                            ViewRole.THIRD_PERSON.value,
                        }
                        else "single_role_with_aligned_context"
                    ),
                    "next_event_id": (
                        final_events[index].event_id
                        if index < len(final_events)
                        else None
                    ),
                    "uncertainty": list(event.uncertainty),
                }
            )
        segment.micro_segments = synchronized_micro_segments
        receipts.append(
            {
                "segment_id": segment.segment_id,
                "previous_event_ids": previous_event_ids,
                "final_event_ids": list(segment.event_ids),
                "previous_micro_segment_count": previous_micro_count,
                "final_micro_segment_count": len(segment.micro_segments),
            }
        )
    return receipts


def validate_final_step_action_consistency(
    groups: Sequence[ExperimentGroup],
    key_events: Sequence[EvidenceEvent],
) -> dict[str, Any]:
    """Fail closed when final prose asserts an absent high-risk action."""

    def is_explicitly_negated(text: str, start: int, end: int) -> bool:
        """Return true only for a denial attached to this exact occurrence.

        A phrase such as ``液体转移状态不可确认`` is an uncertainty, not a
        claim that liquid moved.  Evaluate each regex occurrence separately so
        a denial earlier in the sentence cannot mask a later positive claim.
        """

        before = text[max(0, start - 12) : start]
        after = text[end : min(len(text), end + 14)]
        before_denial = re.search(
            r"(?:(?:未(?:观察到|观测到|看到|看见|见到|见|确认|证明|发生)?|"
            r"没有(?:观察到|看到|看见|确认|证明)?|"
            r"无法确认|不能确认|不可确认|不确定|"
            r"是否(?:已)?(?:完成|发生)?)"
            r"[^，。；！？]{0,4}|无(?:可见|明确|[^，。；！？]{0,2}被)?|"
            r"无已审核通过的事件支持|"
            r"未发生(?:经证实的|已确认的)?)$",
            before,
        )
        after_denial = re.match(
            r"(?:状态|动作)?(?:不可|无法|不能|未能|尚未|未)"
            r"(?:确认|判断|观察到|看见|证明|辨认)"
            r"|(?:不可见|不可读|无法读取|不能读取|未见|不确定|是否发生不可确认)"
            r"|是否(?:已)?(?:发生|完成)[^，。；！？]{0,8}(?:无法|不能|未能)确认"
            r"|(?:完成|发生)(?:均|都)?(?:未见|未看到|未观察到|无法确认)"
            r"|(?:动作)?正在进行(?=，未(?:看到|看见|观察到|见))"
            r"|(?:(?:或|、)[^，。；！？或、]{1,8}){0,3}"
            r"(?:完成(?:均|都)?未见|(?:均|都)?未见完成)"
            r"|[^，。；！？]{0,6}(?:无法|不能|未能|尚未)"
            r"(?:确认|判断|证实)",
            after,
        )
        clause_start = max(text.rfind(mark, 0, start) for mark in "，。；！？") + 1
        clause_before = text[clause_start:start]
        # A denial often governs a long Chinese enumeration separated with
        # ``、``/``或`` rather than clause punctuation.  Bind the marker to
        # every later item in the same clause, while the transition check
        # below still exposes a later positive claim such as ``但随后...``.
        # Keep the marker vocabulary explicit so ``无菌液体转移`` is never
        # mistaken for a denial.
        scoped_marker = re.search(
            r"(?:无已审核通过的事件支持|"
            r"无(?:经(?:最终审核)?确认的|已确认的|可见的|明确的|任何|"
            r"实际|清晰(?:可读|可见)?)|"
            r"未确认(?:存在|发生)?(?:其他)?|"
            r"未发生(?:经证实的|已确认的)?|"
            r"未(?:观察到|观测到|看到|看见|见到|见|证明)(?:实际|任何|其他|"
            r"明确的|完整的)?|"
            r"没有(?:观察到|看到|看见|确认|证明)(?:任何|其他)?|"
            r"无法(?:证实|确认|认定)(?:存在|发生)?(?:完整的)?|"
            r"不能确认|不可确认|无(?=拧盖|开盖|合盖|取放|液体|吸液|排液))"
            r"[^，。；！？]{0,96}$",
            clause_before,
        )
        scoped_list_denial = False
        if scoped_marker:
            marker_to_occurrence = clause_before[scoped_marker.start() :]
            scoped_list_denial = not re.search(
                r"(?:但|但是|随后|之后|然后|而后|后又|再进行|转而)",
                marker_to_occurrence,
            )
        clause_end_candidates = [
            position for mark in "，。；！？" if (position := text.find(mark, end)) >= 0
        ]
        clause_end = min(clause_end_candidates, default=len(text))
        open_parenthesis = text.rfind("（", clause_start, start)
        close_parenthesis = text.find("）", end, clause_end)
        parenthetical_list_denial = bool(
            open_parenthesis >= clause_start
            and close_parenthesis >= end
            and re.search(
                r"均(?:未通过|未确认|未被确认|无确认)",
                text[close_parenthesis + 1 : clause_end],
            )
        )
        return bool(
            before_denial
            or after_denial
            or scoped_list_denial
            or parenthetical_list_denial
        )

    def has_positive_occurrence(pattern: str, text: str, field: str = "") -> bool:
        def describes_existing_state(match: re.Match[str]) -> bool:
            if match.group() != "开盖":
                return False
            before = text[max(0, match.start() - 8) : match.start()]
            after = text[match.end() : match.end() + 1]
            clause_start = (
                max(text.rfind(mark, 0, match.start()) for mark in "，。；！？") + 1
            )
            subject = text[clause_start : match.start()]
            before_state = bool(
                field == "physical_change"
                and "→" in text
                and match.start() < text.index("→")
                and re.search(r"(?:瓶|容器)$", subject)
                and not re.search(r"操作者|手|将|把|使|拿|拧|旋|取|进行|完成", subject)
            )
            return bool(
                before_state
                or (after == "的" and re.search(r"(?:已|已经)$", before))
                or re.search(r"(?:瓶|容器)(?:已|仍然|仍|全程保持|始终保持)$", before)
            )

        return any(
            not is_explicitly_negated(text, match.start(), match.end())
            and not describes_existing_state(match)
            for match in re.finditer(pattern, text)
        )

    event_by_id = {event.event_id: event for event in key_events}
    violations: list[dict[str, Any]] = []
    group_reports: list[dict[str, Any]] = []
    for group in groups:
        confirmed_actions = {
            event_by_id[event_id].action_type.value
            for event_id in group.key_event_ids
            if event_id in event_by_id
        }
        understanding = group.model_understanding or {}
        steps = list(understanding.get("steps") or [])
        checked_claims: list[dict[str, Any]] = []
        group_level_claims = (
            ("experiment_name", str(group.experiment_name or "")),
            (
                "understanding.experiment_name",
                str(understanding.get("experiment_name") or ""),
            ),
            ("overall_summary", str(understanding.get("overall_summary") or "")),
        )
        for field, text in group_level_claims:
            if not text.strip():
                continue
            checked_claims.append({"step_index": None, "field": field, "text": text})
            for action_type, patterns in FINAL_STEP_ACTION_PATTERNS.items():
                if action_type in confirmed_actions:
                    continue
                matched = [
                    pattern
                    for pattern in patterns
                    if has_positive_occurrence(pattern, text, field)
                ]
                if matched:
                    violations.append(
                        {
                            "group_id": group.group_id,
                            "step_index": None,
                            "field": field,
                            "unsupported_action_type": action_type,
                            "matched_patterns": matched,
                            "text": text,
                        }
                    )
        for step in steps:
            step_index = step.get("step_index")
            references = step.get("supporting_event_ids") or []
            if not references and len(group.key_event_ids) == 1:
                references = group.key_event_ids
            step_actions = {
                event_by_id[eid].action_type.value
                for eid in references
                if eid in event_by_id
                and eid in group.key_event_ids
                and event_is_formal(event_by_id[eid])
            }
            for field in (
                "operation_title",
                "current_step",
                "observed_result",
                "physical_change",
            ):
                text = str(step.get(field) or "").strip()
                if not text:
                    continue
                checked_claims.append(
                    {"step_index": step_index, "field": field, "text": text}
                )
                for action_type, patterns in FINAL_STEP_ACTION_PATTERNS.items():
                    if action_type in step_actions:
                        continue
                    matched = [
                        pattern
                        for pattern in patterns
                        if has_positive_occurrence(pattern, text, field)
                    ]
                    if matched:
                        violations.append(
                            {
                                "group_id": group.group_id,
                                "step_index": step_index,
                                "field": field,
                                "unsupported_action_type": action_type,
                                "matched_patterns": matched,
                                "text": text,
                            }
                        )
        group_reports.append(
            {
                "group_id": group.group_id,
                "confirmed_action_types": sorted(confirmed_actions),
                "checked_claim_count": len(checked_claims),
            }
        )
    return {
        "schema_version": "visioncortex-final-step-action-consistency/1",
        "status": "passed" if not violations else "failed",
        "passed": not violations,
        "policy": "final prose cannot assert absent high-risk action classes",
        "groups": group_reports,
        "violations": violations,
    }
