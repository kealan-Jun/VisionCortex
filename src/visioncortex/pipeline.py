from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat as stat_module
import threading
import time
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np

from .actions import (
    audit_candidates,
    build_experiment_segments,
    build_physical_change_log,
    generate_candidates,
    generate_coarse_activity_candidates,
    generate_motion_burst_candidates,
    generate_motion_safety_candidates,
    fuse_motion_probe_candidates,
    refine_motion_candidates_with_coarse,
    refine_liquid_events_with_context,
    select_fine_scan_views,
)
from .alignment import alignment_quality_report, build_alignments
from .action_semantics import (
    attach_action_observability,
    build_semantic_review_plan,
)
from .action_state_machine import (
    attach_continuous_action_states,
    build_event_state_receipt,
)
from .archive import (
    ArchiveLayout,
    _rerender_curated_participant_annotations,
    analyze_experiment_groups,
    analyze_key_materials,
    curate_semantically_reviewed_key_materials,
    finalize_archive,
    materialize_experiment_clips,
    materialize_key_materials,
    key_material_action_folder,
    prepare_key_material_category_layout,
    reconcile_visually_reviewed_participants,
    refresh_key_material_metadata,
    write_aligned_csv,
    write_key_material_category_index,
    write_json,
)
from .grouping import (
    build_experiment_groups,
    is_experiment_start_anchor,
    normalize_experiment_segments,
    prepare_formal_experiment_segments,
    select_key_events,
)
from .pathing import archive_relative_posix
from .ordering import candidate_sort_key, event_sort_key
from .detection import iter_frame_evidence, scan_videos, validate_models
from .coarse_recall import (
    generate_open_vocabulary_coarse_candidates,
    generate_open_vocabulary_fine_candidates,
)
from .material_naming import ACTION_LABELS_ZH
from .candidate_index import (
    CoarseFrameIndex,
    FineFrameIndex,
    build_coarse_frame_index,
    create_fine_frame_index,
    fine_frame_coverage_report,
    ingest_fine_frame_ledgers,
)
from .daily_reports import generate_daily_report_archive
from .decisions import decision_receipt
from .model_certification import audit_production_model_certification
from .provenance import write_run_provenance
from .partial_delivery import write_partial_delivery
from .schema_contracts import (
    validate_archive_contracts_or_raise,
    write_archive_contract_manifest,
)
from . import speech
from .schemas import (
    ActionCandidate,
    ActionType,
    AlignmentTransform,
    EvidenceEvent,
    ExperimentGroup,
    ExperimentSegment,
    FrameEvidence,
    PhysicalChange,
    RunManifest,
    VideoInfo,
    ViewInput,
    ViewRole,
    event_is_formal,
)
from .video_io import (
    benchmark_sparse_decode_strategy,
    check_disk_capacity,
    create_grid_video,
    extract_view_clip,
    probe_video,
    probe_views,
    video_encoder_preflight,
    view_source_files,
    view_timestamp_files,
)
from .storage import (
    IncrementalArchivePublisher,
    initialize_nas_archive,
    snapshot_source_paths,
    source_cache_diagnostics,
)
from .telemetry import ResourceMonitor
from .reviewed_artifacts import load_dataset_scoped_json
from .validation import (
    evaluate_key_event_recall,
    finalize_quality_acceptance_claims,
    validate_experiment_and_material_quality,
)


ProgressCallback = Callable[[str, float, str], None]


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
    actor = (
        "gloved_hand"
        if re.search(r"手套|glov(?:e|ed)", normalized)
        else "hand"
    )
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
        config.get("mllm", {}).get(
            "group_state_recall_minimum_confidence", 0.55
        )
    )
    minimum_step_confidence = float(
        config.get("mllm", {}).get(
            "group_state_recall_step_minimum_confidence", 0.55
        )
    )
    next_number = max(
        [
            int(match.group(1))
            for event in events
            if (match := re.fullmatch(r"EVT-(\d+)", event.event_id))
        ],
        default=0,
    ) + 1
    by_segment = {segment.segment_id: segment for segment in segments}
    recovered: list[dict[str, Any]] = []
    for group in groups:
        understanding = group.model_understanding or {}
        if (
            understanding.get("status") != "completed"
            or float(understanding.get("confidence") or 0.0)
            < minimum_group_confidence
        ):
            continue
        group_segments = [
            by_segment[segment_id]
            for segment_id in group.atomic_experiment_ids
            if segment_id in by_segment
        ]
        group_event_ids = {
            event_id
            for segment in group_segments
            for event_id in segment.event_ids
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
                str(item)
                for item in step.get("supporting_views") or []
                if str(item)
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
                min(end_ms, event.global_end_ms)
                > max(start_ms, event.global_start_ms)
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
                uncertainty=[
                    "该候选由组级语义召回产生，不代表事件级状态证明已通过"
                ],
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
                        if candidate.event_id in {
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
                    "group_confidence": float(
                        understanding.get("confidence") or 0.0
                    ),
                    "supporting_views": sorted(required_views),
                    "independent_event_level_review_required": True,
                }
            )
    return recovered


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

    event_by_id = {event.event_id: event for event in key_events if event.accepted}
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
            next_step_evidence = dict(
                understanding.get("next_step_evidence") or {}
            )
            next_step_status = str(
                next_step_evidence.get("status") or "unknown"
            )
            if next_step_status not in {"observed", "inferred", "unknown"}:
                next_step_status = "unknown"
            supporting_event_ids = {
                event.event_id,
                *(
                    str(item)
                    for item in next_step_evidence.get("evidence_event_ids") or []
                    if item
                ),
            }
            before = str(physical_change.get("before") or "")
            after = str(physical_change.get("after") or "")
            physical_change_text = (
                f"{before} → {after}"
                if before or after
                else "未观察到可确认的状态变化"
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
                    "next_step": str(
                        understanding.get("next_step") or "未知"
                    ),
                    "next_step_status": next_step_status,
                    "next_step_evidence": next_step_evidence,
                    "supporting_event_ids": sorted(supporting_event_ids),
                    "objects": list(event.objects),
                    "physical_change": physical_change_text,
                    "supporting_views": list(event.supporting_views),
                    "confidence": float(
                        understanding.get("confidence") or event.confidence
                    ),
                    "action_type": event.action_type.value,
                    "event_id": event.event_id,
                    **({"operation_title": understanding["operation_title"]}
                       if understanding.get("operation_title") else {}),
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
        title_violations = [
            item for item in validate_final_step_action_consistency([group], events)["violations"]
            if item["field"] in {"experiment_name", "understanding.experiment_name"}
        ]
        if title_violations:
            # A candidate-stage title may claim an action removed by event
            # adjudication. Rebuild only that display title from final actions;
            # preserve the original model receipt and all material paths.
            title = "、".join(action_labels) + "实验片段" if events else "待确认实验片段"
            english_title = "Reviewed Experiment Segment" if events else "Unconfirmed Experiment Segment"
            refined["title_refinement"] = {
                "reason": "candidate_title_asserted_rejected_action",
                "previous_title": group.experiment_name,
                "previous_title_en": group.experiment_name_en,
                "final_title": title,
                "final_title_en": english_title,
                "unsupported_action_types": sorted({item["unsupported_action_type"] for item in title_violations}),
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
                ((previous.get("object_identity") or {}).get("object_classes"))
                or []
            )
        }
        participant_identity_mismatch = bool(
            expected_object_classes != previous_object_classes
        )
        if (
            str(previous.get("action_type") or "") == expected_action
            and int(previous.get("peak_timestamp_us") or -1)
            == expected_peak_us
            and not participant_identity_mismatch
        ):
            continue
        rebuilt = build_event_state_receipt(event, config)
        derivation = dict(previous.get("derivation") or {})
        if derivation:
            derivation["timing_resynchronized_after_final_key_frame_selection"] = (
                True
            )
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
                "participant_identity_resynchronized": (
                    participant_identity_mismatch
                ),
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
                            role.value
                            if isinstance(role, ViewRole)
                            else str(role)
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
            r"|(?:不可见|未见|不确定|是否发生不可确认)"
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
            position
            for mark in "，。；！？"
            if (position := text.find(mark, end)) >= 0
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
            before = text[max(0, match.start() - 8):match.start()]
            after = text[match.end():match.end() + 1]
            clause_start = max(text.rfind(mark, 0, match.start()) for mark in "，。；！？") + 1
            subject = text[clause_start:match.start()]
            before_state = bool(
                field == "physical_change" and "→" in text
                and match.start() < text.index("→")
                and re.search(r"(?:瓶|容器)$", subject)
                and not re.search(r"操作者|手|将|把|使|拿|拧|旋|取|进行|完成", subject)
            )
            return bool(
                before_state
                or
                (after == "的" and re.search(r"(?:已|已经)$", before))
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
            checked_claims.append(
                {"step_index": None, "field": field, "text": text}
            )
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
            for field in ("operation_title", "current_step", "physical_change"):
                text = str(step.get(field) or "").strip()
                if not text:
                    continue
                checked_claims.append(
                    {"step_index": step_index, "field": field, "text": text}
                )
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


def _noop_progress(stage: str, progress: float, message: str) -> None:
    del stage, progress, message


def run_media_pipeline_preflight(
    manifest: RunManifest,
    infos: dict[str, Any],
    work_root: Path,
    preferred_encoder: str,
    duration_seconds: float,
) -> dict[str, Any]:
    """Exercise source decode and paired delivery encoding before long scans."""

    first = next(
        (view for view in manifest.views if view.role == ViewRole.FIRST_PERSON),
        None,
    )
    third = next(
        (view for view in manifest.views if view.role == ViewRole.THIRD_PERSON),
        None,
    )
    if first is None or third is None:
        raise ValueError("media pipeline preflight requires first- and third-person views")
    requested_ms = max(200.0, float(duration_seconds) * 1000.0)
    smoke_root = work_root / "media-pipeline-preflight"
    smoke_root.mkdir(parents=True, exist_ok=True)
    clips: list[tuple[str, Path]] = []
    clip_reports: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for label, view in (("First-Person", first), ("Third-Person", third)):
            info = infos[view.view_id]
            duration_ms = min(requested_ms, float(info.duration_ms))
            if duration_ms <= 0.0:
                raise ValueError(f"view has no probeable duration: {view.view_id}")
            start_ms = min(1000.0, max(0.0, float(info.duration_ms) - duration_ms))
            destination = smoke_root / f"{label}.mp4"
            clip_started = time.perf_counter()
            extract_view_clip(
                view,
                info,
                destination,
                start_ms,
                duration_ms,
                preferred_encoder,
            )
            rendered = probe_video(destination)
            if rendered.duration_ms <= 0.0 or rendered.width <= 0 or rendered.height <= 0:
                raise RuntimeError(f"encoded smoke clip is invalid: {destination}")
            clips.append((label, destination))
            clip_reports.append(
                {
                    "view_id": view.view_id,
                    "role": view.role.value,
                    "source_start_ms": round(start_ms, 3),
                    "requested_duration_ms": round(duration_ms, 3),
                    "encoded_duration_ms": round(rendered.duration_ms, 3),
                    "width": rendered.width,
                    "height": rendered.height,
                    "bytes": destination.stat().st_size,
                    "elapsed_seconds": round(time.perf_counter() - clip_started, 6),
                }
            )
        aligned = smoke_root / "Aligned_First+Third.mp4"
        grid_started = time.perf_counter()
        create_grid_video(clips, aligned, preferred_encoder)
        rendered_grid = probe_video(aligned)
        if (
            rendered_grid.duration_ms <= 0.0
            or rendered_grid.width <= 0
            or rendered_grid.height <= 0
        ):
            raise RuntimeError(f"encoded aligned smoke clip is invalid: {aligned}")
        report = {
            "schema_version": "visioncortex-media-pipeline-preflight/1",
            "status": "passed",
            "source_mode": (
                "segmented_virtual_timeline"
                if all(view.segments for view in manifest.views)
                else "continuous_file"
            ),
            "selected_encoder": preferred_encoder,
            "views": clip_reports,
            "aligned_output": {
                "encoded_duration_ms": round(rendered_grid.duration_ms, 3),
                "width": rendered_grid.width,
                "height": rendered_grid.height,
                "bytes": aligned.stat().st_size,
                "elapsed_seconds": round(time.perf_counter() - grid_started, 6),
            },
            "elapsed_seconds": round(time.perf_counter() - started, 6),
            "temporary_artifacts_retained": False,
        }
        shutil.rmtree(smoke_root)
        return report
    except Exception:
        # Retain the tiny local smoke directory on failure for incident review.
        raise


def _key_material_selection_report(
    groups: list[ExperimentGroup],
    segments: list[ExperimentSegment],
    events: list[EvidenceEvent],
    decision_receipts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Expose material recall and suspiciously sparse experiment coverage."""

    by_segment = {segment.segment_id: segment for segment in segments}
    by_event = {event.event_id: event for event in events}
    records: list[dict[str, Any]] = []
    for group in groups:
        candidate_ids = {
            event_id
            for segment_id in group.atomic_experiment_ids
            for event_id in by_segment[segment_id].event_ids
            if event_id in by_event and by_event[event_id].accepted
        }
        selected = [
            by_event[event_id]
            for event_id in group.key_event_ids
            if event_id in by_event
        ]
        counts: dict[str, int] = {}
        for event in selected:
            counts[event.action_type.value] = counts.get(event.action_type.value, 0) + 1
        duration_seconds = max(
            0.0, (group.global_end_ms - group.global_start_ms) / 1000.0
        )
        # This is a warning, not a fabricated quota. It makes sparse evidence
        # visible without inventing events or weakening cross-view acceptance.
        sparse_threshold = max(3, min(12, round(duration_seconds / 30.0)))
        records.append(
            {
                "group_id": group.group_id,
                "experiment_name": group.experiment_name,
                "duration_seconds": round(duration_seconds, 3),
                "accepted_physical_events": len(candidate_ids),
                "selected_key_materials": len(selected),
                "selection_rate": round(len(selected) / len(candidate_ids), 4)
                if candidate_ids
                else 0.0,
                "action_type_counts": counts,
                "low_recall_warning": duration_seconds >= 60.0
                and len(selected) < sparse_threshold,
                "low_recall_threshold": sparse_threshold,
            }
        )
    return {
        "schema_version": "visioncortex-key-material-selection/2",
        "selection_rule": (
            "accepted formal physical actions; duplicate only when action/object "
            "identity, positive interval overlap, and peak proximity all agree"
        ),
        "decision_receipt_complete": bool(decision_receipts is not None)
        and len(decision_receipts)
        == sum(record["accepted_physical_events"] for record in records),
        "decision_receipts": list(decision_receipts or []),
        "groups": records,
        "totals": {
            "accepted_physical_events": sum(
                record["accepted_physical_events"] for record in records
            ),
            "selected_key_materials": sum(
                record["selected_key_materials"] for record in records
            ),
            "groups_with_low_recall_warning": sum(
                bool(record["low_recall_warning"]) for record in records
            ),
        },
    }


def _merge_frame_evidence_ledgers(
    existing_path: Path,
    supplement_path: Path,
    output_path: Path,
    *,
    track_id_namespace: int,
) -> dict[str, Any]:
    """Streaming-merge a recall pass without loading full ledgers in RAM."""

    existing_count = 0
    supplement_count = 0
    deduplicated = 0
    merged_count = 0
    offset = max(1, int(track_id_namespace)) * 1_000_000

    def existing_frames():
        nonlocal existing_count
        for frame in iter_frame_evidence(existing_path):
            existing_count += 1
            yield frame

    def supplement_frames():
        nonlocal supplement_count
        for frame in iter_frame_evidence(supplement_path):
            supplement_count += 1
            yield frame.model_copy(
                update={
                    "detections": [
                        detection.model_copy(
                            update={
                                "track_id": (
                                    detection.track_id + offset
                                    if detection.track_id is not None
                                    else None
                                )
                            }
                        )
                        for detection in frame.detections
                    ]
                }
            )

    def identity(frame: FrameEvidence) -> tuple[int, int]:
        return frame.frame_index, round(frame.local_ms * 1000.0)

    def ordering(frame: FrameEvidence) -> tuple[float, int]:
        return frame.local_ms, frame.frame_index

    left_iterator = iter(existing_frames())
    right_iterator = iter(supplement_frames())
    left = next(left_iterator, None)
    right = next(right_iterator, None)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", buffering=1024 * 1024) as handle:
        while left is not None or right is not None:
            if left is None:
                selected = right
                right = next(right_iterator, None)
            elif right is None:
                selected = left
                left = next(left_iterator, None)
            elif identity(left) == identity(right):
                selected = right
                deduplicated += 1
                left = next(left_iterator, None)
                right = next(right_iterator, None)
            elif ordering(left) < ordering(right):
                selected = left
                left = next(left_iterator, None)
            else:
                selected = right
                right = next(right_iterator, None)
            assert selected is not None
            handle.write(selected.model_dump_json() + "\n")
            merged_count += 1
    temporary.replace(output_path)
    return {
        "existing_frames": existing_count,
        "supplement_frames": supplement_count,
        "merged_frames": merged_count,
        "deduplicated_frames": deduplicated,
        "track_id_namespace": offset,
        "merge_strategy": "streaming_sorted_supplement_overwrites_duplicate",
        "output_path": str(output_path),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def build_cache_identity(config: dict[str, Any], manifest: RunManifest) -> dict[str, Any]:
    """Bind resumable ledgers to code, config, models and concrete inputs."""

    package_root = Path(__file__).resolve().parent
    code_digest = hashlib.sha256()
    # Shared orchestration/schema changes still invalidate conservatively. These
    # modules only affect downstream interpretation, reports, speech or the UI.
    downstream_modules = {
        "mllm.py", "mllm_provider.py", "provider_connection.py", "provider_credentials.py",
        "provider_discovery.py", "ai_settings.py", "speech.py", "speech_worker.py",
        "speech_semantics.py", "daily_reports.py", "report_presentations.py",
        "report_brand.py", "partial_delivery.py", "api.py", "cli.py", "stage_refresh.py", "speech_timeline.py",
        "speech_search.py", "speech_refresh.py", "capture_quality.py",
    }
    for source in sorted(package_root.glob("*.py")):
        if source.name in downstream_modules:
            continue
        code_digest.update(source.name.encode("utf-8"))
        code_digest.update(source.read_bytes())

    models = {}
    for role in ("first_person", "third_person"):
        path = Path(config["models"][role]).resolve()
        try:
            model_stat = path.stat()
        except OSError:
            model_stat = None
        if model_stat is not None and not stat_module.S_ISREG(model_stat.st_mode):
            model_stat = None
        models[role] = {
            "path": str(path),
            "size_bytes": model_stat.st_size if model_stat is not None else None,
            "sha256": _sha256_file(path) if model_stat is not None else None,
        }

    def absolute(path: Path) -> Path:
        return Path(os.path.abspath(str(path)))

    files_by_view = {
        view.view_id: {
            "videos": [absolute(path) for path in view_source_files(view)],
            "timestamps_csvs": [absolute(path) for path in view_timestamp_files(view)],
            "audio": [absolute(path) for path in speech.audio_files(view)],
        }
        for view in manifest.views
    }
    all_source_paths = [
        path
        for files in files_by_view.values()
        for file_type in ("videos", "timestamps_csvs", "audio")
        for path in files[file_type]
    ]
    source_snapshots, source_snapshot_report = snapshot_source_paths(
        all_source_paths,
        workers=int(config["performance"].get("source_stat_workers", 24)),
        max_age_seconds=float(
            config["performance"].get("source_stat_cache_ttl_seconds", 120.0)
        ),
    )
    inputs = []
    for view in manifest.views:
        source_files = files_by_view[view.view_id]["videos"]
        clock_files = files_by_view[view.view_id]["timestamps_csvs"]
        inputs.append(
            {
                "view_id": view.view_id,
                "role": view.role.value,
                "input_mode": "segmented" if view.segments else "continuous_file",
                "audio": [{"path": str(path), "snapshot": source_snapshots[path]} for path in files_by_view[view.view_id]["audio"]],
                "audio_offsets_ms": [part.audio_offset_ms for part in (view.segments or [view])],
                "videos": [
                    {
                        "path": str(path),
                        "size_bytes": source_snapshots[path]["size_bytes"],
                        "mtime_ns": source_snapshots[path]["mtime_ns"],
                    }
                    for path in source_files
                ],
                "timestamps_csvs": [
                    {
                        "path": str(path),
                        "size_bytes": source_snapshots[path]["size_bytes"],
                        "mtime_ns": source_snapshots[path]["mtime_ns"],
                    }
                    for path in clock_files
                ],
            }
        )

    stable_config = deepcopy(config)
    stable_config.get("project", {}).pop("output_root", None)
    # Execution policy must not fork the deterministic cache identity: a cold
    # run and its explicit hot replay intentionally share the same namespace.
    # ``cache_namespace`` remains in the stable config and is therefore the
    # auditable isolation boundary between independent benchmark campaigns.
    stable_config.get("project", {}).pop("cache_mode", None)
    # Ark response reuse is an execution policy independent of deterministic
    # CV artifacts.  Excluding it lets a recovery run reuse the exact verified
    # scan while forcing fresh semantic calls.
    stable_config.get("project", {}).pop("semantic_cache_mode", None)
    stable_config.get("project", {}).pop("semantic_recovery_attempt", None)
    # This switch changes only how far a run proceeds. Keeping it out of the
    # CV cache identity lets a later authorized full pipeline reuse the exact
    # accepted cold-start preprocessing ledgers without weakening provenance.
    stable_config.get("project", {}).pop("preprocessing_acceptance_only", None)
    for key in (
        "active_archive_path",
        "archive_root",
        "local_staging_root",
        "local_runtime_root",
        "local_input_root",
        "local_cache_root",
        "sync_to_nas",
    ):
        stable_config.get("storage", {}).pop(key, None)

    downstream_config = {key: stable_config.pop(key, None)
                         for key in ("mllm", "speech_recognition", "daily_report", "capture_quality")}
    audio_inputs = [{"view_id": item["view_id"], "audio": item.pop("audio"),
                     "audio_offsets_ms": item.pop("audio_offsets_ms")} for item in inputs]
    payload = {
        "schema_version": "visioncortex-cache-identity/2",
        "code_sha256": code_digest.hexdigest(),
        "config": stable_config,
        "models": models,
        "inputs": inputs,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    payload["cache_key"] = hashlib.sha256(encoded).hexdigest()[:20]
    payload["downstream_dependencies"] = {
        "config": downstream_config, "audio_inputs": audio_inputs,
        "identity_policy": "ASR worker/runtime/source identity; semantic prompt/provider/evidence identity",
    }
    payload["execution_cache_policy"] = {
        "mode": str(config.get("project", {}).get("cache_mode") or "reuse"),
        "semantic_mode": str(
            config.get("project", {}).get(
                "semantic_cache_mode",
                config.get("project", {}).get("cache_mode") or "reuse",
            )
        ),
        "namespace": config.get("project", {}).get("cache_namespace"),
    }
    payload["source_snapshot_report"] = source_snapshot_report
    return payload


class EvidencePipeline:
    def __init__(self, config: dict[str, Any], progress: ProgressCallback | None = None):
        self.config = config
        mllm = config.get("mllm") or {}
        configured_timeout = float(mllm.get("timeout_seconds", 180))
        # Selected-provider connections are verified with a 180 s budget.
        # Legacy hardware profiles (including retained jobs) must not silently
        # shorten that budget for the larger real-experiment request.
        selected_provider = bool(mllm.get("provider") and mllm.get("quality_mode"))
        effective_timeout = max(180.0, configured_timeout) if selected_provider else configured_timeout
        if effective_timeout != configured_timeout:
            self.config = {**config, "mllm": {**mllm, "timeout_seconds": effective_timeout}}
        self._mllm_timeout_policy = {
            "configured_timeout_seconds": configured_timeout,
            "effective_timeout_seconds": effective_timeout,
            "selected_provider_minimum_seconds": 180 if selected_provider else None,
            "request_content_changed": False,
        }
        self.progress = progress or _noop_progress
        self._run_started_perf = 0.0
        self._run_started_iso = ""
        self._active_stage: str | None = None
        self._active_stage_started = 0.0
        self._active_stage_started_iso = ""
        self._stage_metrics: list[dict[str, Any]] = []
        self._startup_metrics: dict[str, Any] = {}
        self._speech_understanding: dict[str, Any] = {}
        self._preprocessing_completed_seconds: float | None = None
        self._input_view_count = 0
        self._input_mode = "unknown"
        self._publisher: IncrementalArchivePublisher | None = None
        self._resource_monitor: ResourceMonitor | None = None
        self._view_runtime: dict[str, dict[str, Any]] = {}
        self._runtime_lock = threading.Lock()
        self._active_layout: ArchiveLayout | None = None
        self._current_experiment_id: str | None = None
        self._acceptance_baseline_selection: dict[str, Any] = {
            "configured": False,
            "applied": False,
            "reason": "not_evaluated",
        }
        self._model_certification_audit: dict[str, Any] = {
            "required": False,
            "status": "not_required_by_profile",
        }

    def _frame_index_path(self, work_dir: Path, filename: str) -> Path:
        # The evidence cache may live on SMB/NAS. SQLite locking and WAL shared
        # memory require a local filesystem; only rebuildable query indexes go
        # here. Authoritative JSONL ledgers remain in their configured cache.
        runtime_root = Path(self.config["storage"]["local_runtime_root"]).resolve()
        namespace = hashlib.sha256(str(work_dir.resolve()).encode("utf-8")).hexdigest()
        return runtime_root / "state" / "frame-indexes" / namespace / filename

    def _scan_progress(
        self, phase: str, view_id: str, completed_units: int, total_units: int
    ) -> None:
        with self._runtime_lock:
            runtime = self._view_runtime.setdefault(view_id, {})
            runtime.update(
                {
                    "state": f"{phase}_running"
                    if completed_units < total_units
                    else f"{phase}_completed",
                    "phase": phase,
                    "completed_units": completed_units,
                    "total_units": total_units,
                    "unit_progress": completed_units / total_units if total_units else 0.0,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            if self._active_layout is None:
                return
            payload = {
                "schema_version": "visioncortex-source-progress/1",
                "phase": phase,
                "updated_at": runtime["updated_at"],
                "views": self._view_runtime,
            }
            path = self._active_layout.json_config / "source_progress.json"
            write_json(path, payload)
            if self._publisher is not None:
                self._publisher.publish_file(path)

    def _status(self, layout: ArchiveLayout, stage: str, progress: float, message: str) -> None:
        if self._resource_monitor is not None:
            self._resource_monitor.set_stage(stage)
        now_perf = time.perf_counter()
        now_iso = datetime.now(timezone.utc).isoformat()
        if self._active_stage is not None and stage != self._active_stage:
            self._stage_metrics.append(
                {
                    "stage": self._active_stage,
                    "status": "failed" if stage == "failed" else "completed",
                    "started_at": self._active_stage_started_iso,
                    "ended_at": now_iso,
                    "duration_seconds": round(now_perf - self._active_stage_started, 6),
                }
            )
        if stage != self._active_stage:
            if stage in {"completed", "partial", "failed"}:
                self._active_stage = None
            else:
                self._active_stage = stage
                self._active_stage_started = now_perf
                self._active_stage_started_iso = now_iso
        self.progress(stage, progress, message)
        status_payload = {
            "stage": stage,
            "progress": progress,
            "message": message,
            "updated_at": now_iso,
            "elapsed_seconds": round(now_perf - self._run_started_perf, 6)
            if self._run_started_perf
            else 0.0,
            "input_view_count": self._input_view_count,
            "input_mode": self._input_mode,
            "views": self._view_runtime,
            "completed_stages": [
                item["stage"]
                for item in self._stage_metrics
                if item.get("status", "completed") == "completed"
            ],
            "failed_stage": next(
                (
                    item["stage"]
                    for item in reversed(self._stage_metrics)
                    if item.get("status") == "failed"
                ),
                None,
            ),
        }
        write_json(
            layout.root / "run_status.json",
            status_payload,
        )
        if self.config.get("storage", {}).get("run_output_mode") == "nas_direct":
            write_json(layout.json_config / "pipeline_status.json", status_payload)
            stage_note = "处理中" if stage not in {"completed", "partial", "failed"} else (
                "分析结束，阶段成果已保存" if stage == "partial" else
                "分析已结束，等待归档校验" if stage == "completed" else "处理失败，已完成的阶段产出保留"
            )
            note_path = layout.root / "处理状态.txt"
            temporary_note = note_path.with_suffix(".txt.partial")
            temporary_note.write_text(
                f"{stage_note}\n{message}\n更新时间：{now_iso}\n"
                "各阶段产出位于对应文件夹；最终结果以归档校验为准。\n", encoding="utf-8"
            )
            temporary_note.replace(note_path)
        if self._publisher is not None:
            self._publisher.publish_status(status_payload)

    def _finish_quality_attention(self, layout: ArchiveLayout, events, groups) -> Path:
        """Finish an evaluated run without granting formal publication rights."""
        self._status(layout, "quality_attention", .995, "质量检查已结束，正在保存阶段成果与证据缺口")
        metrics = self._metrics(events, groups)
        write_json(layout.json_config / "run_metrics.json", metrics)
        write_partial_delivery(layout.root, metrics, analysis_finished=True)
        if self._publisher is not None:
            self._publisher.publish_directory("Partial-Results")
            self._publisher.publish_directory("JSON-Config-Files")
        self._status(layout, "partial", 1.0, "分析结束，部分证据不足；阶段成果与缺口报告已保存")
        return layout.root

    def _complete_stage(
        self,
        layout: ArchiveLayout,
        stage: str,
        artifacts: list[Path] | tuple[Path, ...] = (),
    ) -> Path:
        """Publish a durable, atomic NAS receipt only after a stage succeeds."""

        completed_at = datetime.now(timezone.utc).isoformat()
        elapsed_seconds = round(time.perf_counter() - self._run_started_perf, 6)
        stage_duration = None
        if self._active_stage == stage:
            stage_duration = round(time.perf_counter() - self._active_stage_started, 6)
        else:
            stage_duration = next(
                (
                    item["duration_seconds"]
                    for item in reversed(self._stage_metrics)
                    if item["stage"] == stage
                ),
                None,
            )
        relative_artifacts: list[str] = []
        for artifact in artifacts:
            artifact = Path(artifact)
            if not artifact.exists():
                raise FileNotFoundError(f"Completed stage artifact is missing: {artifact}")
            relative = archive_relative_posix(artifact, layout.root)
            relative_artifacts.append(relative)
            if self._publisher is not None:
                if artifact.is_dir():
                    self._publisher.publish_directory(Path(relative))
                else:
                    self._publisher.publish_file(artifact)
        receipt = {
            "schema_version": "visioncortex-stage-receipt/1",
            "stage": stage,
            "status": "completed",
            "completed_at": completed_at,
            "run_elapsed_seconds": elapsed_seconds,
            "stage_duration_seconds": stage_duration,
            "archive_mode": self.config.get("storage", {}).get("run_output_mode", "local"),
            "archive_root": str(layout.root),
            "artifacts": relative_artifacts,
            "token_ledger": "JSON-Config-Files/run_metrics.json",
        }
        receipt_path = layout.json_config / "Stage-Receipts" / f"{stage}.json"
        write_json(receipt_path, receipt)
        if self._publisher is not None:
            self._publisher.publish_file(receipt_path)
        return receipt_path

    def _checkpoint_key_material_understanding(
        self,
        layout: ArchiveLayout,
        stage: str,
        events: list[EvidenceEvent],
        groups: list[ExperimentGroup],
        semantic_curation: dict[str, Any] | None = None,
    ) -> list[Path]:
        """Keep each completed understanding pass before later passes mutate it."""

        review_status = {
            "mllm": "pending_material_refinement",
            "material_refinement": "pending_semantic_refinement",
            "semantic_refinement": "pending_quality_acceptance",
        }[stage]
        understanding = {
            "schema_version": "visioncortex-key-material-understanding/1",
            "refinement_stage": stage,
            "review_status": review_status,
            "events": [event.model_dump(mode="json") for event in events],
            "semantic_curation": {
                key: value
                for key, value in (semantic_curation or {}).items()
                if key != "records"
            },
        }
        metrics = self._metrics(events, groups)
        snapshots = layout.json_config / "Stage-Outputs"
        payloads = {
            snapshots / f"{stage}.json": understanding,
            snapshots / f"{stage}-metrics.json": metrics,
            layout.json_config / "key_material_model_understanding.json": understanding,
            layout.json_config / "run_metrics_live.json": metrics,
        }
        for path, payload in payloads.items():
            write_json(path, payload)
        return list(payloads)

    def _acceptance_baseline(self) -> dict[str, Any] | None:
        configured = self.config.get("validation", {}).get("acceptance_baseline")
        payload, selection = load_dataset_scoped_json(
            configured,
            self._current_experiment_id,
            repository_root=Path(__file__).resolve().parents[2],
            artifact_label="验收基线",
        )
        self._acceptance_baseline_selection = selection
        return payload

    def _run_quality_acceptance(
        self,
        layout: ArchiveLayout,
        groups: list[ExperimentGroup],
        key_events: list[EvidenceEvent],
    ) -> dict[str, Any]:
        validation = self.config.get("validation", {})
        baseline = self._acceptance_baseline()
        report = validate_experiment_and_material_quality(
            groups,
            key_events,
            baseline,
            boundary_match_iou=float(validation.get("boundary_match_iou", 0.50)),
            max_start_error_seconds=float(validation.get("max_start_error_seconds", 8.0)),
            max_end_error_seconds=float(validation.get("max_end_error_seconds", 8.0)),
            minimum_cross_view_event_rate=float(
                validation.get("minimum_cross_view_event_rate", 0.25)
            ),
            require_participant_only_annotations=bool(
                validation.get("require_participant_only_annotations", False)
            ),
        )
        report["baseline_selection"] = dict(
            self._acceptance_baseline_selection
        )
        key_event_ground_truth, ground_truth_selection = (
            load_dataset_scoped_json(
                validation.get("key_event_ground_truth"),
                self._current_experiment_id,
                repository_root=Path(__file__).resolve().parents[2],
                artifact_label="关键事件真值",
            )
        )
        recall_report = evaluate_key_event_recall(
            key_events, key_event_ground_truth
        )
        recall_report["ground_truth_selection"] = ground_truth_selection
        recall_gate = {
            "evaluated": bool(recall_report.get("evaluated")),
            "passed": None,
            "temporal_iou_threshold": float(
                validation.get("key_event_recall_iou", 0.50)
            ),
            "minimum_precision": float(
                validation.get("minimum_key_event_precision", 0.80)
            ),
            "minimum_recall": float(
                validation.get("minimum_key_event_recall", 0.80)
            ),
            "precision": None,
            "recall": None,
            "small_sample_warning": recall_report.get(
                "small_sample_warning"
            ),
        }
        if recall_gate["evaluated"]:
            selected_threshold = next(
                (
                    item
                    for item in recall_report.get("threshold_results") or []
                    if abs(
                        float(item["temporal_iou_threshold"])
                        - recall_gate["temporal_iou_threshold"]
                    )
                    < 1e-9
                ),
                None,
            )
            if selected_threshold is None:
                raise ValueError(
                    "Configured key-event recall IoU is absent from evaluation thresholds"
                )
            recall_gate["precision"] = selected_threshold.get("precision")
            recall_gate["recall"] = selected_threshold.get("recall")
            recall_gate["passed"] = bool(
                recall_gate["precision"] is not None
                and recall_gate["recall"] is not None
                and float(recall_gate["precision"])
                >= recall_gate["minimum_precision"]
                and float(recall_gate["recall"])
                >= recall_gate["minimum_recall"]
            )
            report["passed"] = bool(report.get("passed")) and bool(
                recall_gate["passed"]
            )
            if not recall_gate["passed"]:
                report["status"] = "failed"
        report["key_event_recall"] = recall_gate
        step_consistency = validate_final_step_action_consistency(
            groups, key_events
        )
        report["step_action_consistency"] = step_consistency
        if not step_consistency["passed"]:
            report["passed"] = False
            report["status"] = "failed"
        finalize_quality_acceptance_claims(
            report, recall_gate, step_consistency
        )
        write_json(layout.json_config / "quality_acceptance.json", report)
        write_json(
            layout.json_config / "key_material_recall_eval.json",
            recall_report,
        )
        return report

    def _run_boundary_precheck(
        self,
        layout: ArchiveLayout,
        groups: list[ExperimentGroup],
        progressive_report: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        validation = self.config.get("validation", {})
        full_report = validate_experiment_and_material_quality(
            groups,
            [],
            self._acceptance_baseline(),
            boundary_match_iou=float(validation.get("boundary_match_iou", 0.50)),
            max_start_error_seconds=float(validation.get("max_start_error_seconds", 8.0)),
            max_end_error_seconds=float(validation.get("max_end_error_seconds", 8.0)),
            minimum_cross_view_event_rate=float(
                validation.get("minimum_cross_view_event_rate", 0.25)
            ),
            require_participant_only_annotations=bool(
                validation.get("require_participant_only_annotations", False)
            ),
        )
        boundary = full_report["experiment_boundaries"]
        boundary_evaluated = bool(boundary["evaluated"])
        boundary_passed = (
            bool(boundary["passed"]) if boundary_evaluated else True
        )
        progressive_evaluated = progressive_report is not None
        progressive_passed = bool(
            progressive_report.get("quality_complete", True)
            if progressive_report is not None
            else True
        )
        evaluated = boundary_evaluated or progressive_evaluated
        passed = boundary_passed and progressive_passed
        # A reviewed baseline regression invalidates the bounded experiment
        # itself and remains a blocking pre-model failure.  In contrast, an
        # unresolved recall candidate is event-level uncertainty: retain it in
        # the audit ledger, mark the run partial, and continue producing
        # machine-verifiable evidence for the candidates that did close across
        # views.  One uncertain candidate must not discard an otherwise usable
        # analysis run.
        blocking_failure = boundary_evaluated and not boundary_passed
        continuation_status = (
            "passed" if passed else "failed" if blocking_failure else "partial"
        )
        report = {
            "schema_version": "visioncortex-boundary-precheck/1",
            "status": continuation_status,
            "evaluated": evaluated,
            "passed": passed,
            "blocking_failure": blocking_failure,
            "analysis_continuation_allowed": not blocking_failure,
            "evidence_classification": (
                "PROVEN" if passed else "NOT_PROVEN" if blocking_failure else "PARTIAL_EVIDENCE"
            ),
            "baseline": full_report["baseline"],
            "baseline_selection": dict(self._acceptance_baseline_selection),
            "thresholds": full_report["thresholds"],
            "experiment_boundaries": boundary,
            "cross_view_cluster_completeness": {
                "evaluated": progressive_evaluated,
                "passed": progressive_passed,
                "unresolved_candidate_ids": (
                    progressive_report.get("unresolved_candidate_ids", [])
                    if progressive_report is not None
                    else []
                ),
                "group_local_recall": (
                    progressive_report.get("group_local_recall", {})
                    if progressive_report is not None
                    else {}
                ),
            },
            "gate_position": "before_experiment_and_key_material_model_calls",
        }
        path = layout.json_config / "boundary_precheck.json"
        write_json(path, report)
        if (
            blocking_failure
            and bool(validation.get("fail_before_model_on_boundary_regression", True))
        ):
            raise RuntimeError(
                "Bounded experiment quality precheck failed before model calls; "
                f"see {path}"
            )
        return report

    def _metrics(self, events, groups=()) -> dict[str, Any]:
        key_calls = []
        visual_calls_by_fingerprint: dict[str, dict[str, Any]] = {}
        for event in events:
            understanding = event.model_understanding or {}
            if "usage" in understanding or understanding.get("status") in {"completed", "failed"}:
                key_calls.append(
                    {
                        "stage": "key_material_understanding",
                        **{key: understanding.get(key) for key in ("provider", "api_protocol", "request_id", "response_model")},
                        "event_id": event.event_id,
                        "model": understanding.get("model", self.config["mllm"]["model"]),
                        "status": understanding.get("status"),
                        "latency_seconds": understanding.get("latency_seconds"),
                        "attempts": understanding.get("attempts"),
                        "cache_reused": bool(understanding.get("cache_reused")),
                        "attempt_receipts": understanding.get("attempt_receipts", []),
                        "usage": understanding.get("usage", {}),
                    }
                )
            for fingerprint, review in (event.observability.get("participant_visual_review") or {}).items():
                if not review.get("request_attempted"):
                    continue
                call = {
                    "stage": "participant_visual_review",
                    **{key: review.get(key) for key in ("provider", "api_protocol", "request_id", "response_model")},
                    "event_id": event.event_id,
                    "input_fingerprint": fingerprint,
                    "model": review.get("model", self.config["mllm"]["model"]),
                    "status": review.get("status"),
                    "latency_seconds": review.get("latency_seconds"),
                    "attempts": review.get("attempts"),
                    "cache_reused": bool(review.get("cache_reused")),
                    "usage": review.get("usage") or {},
                }
                prior = visual_calls_by_fingerprint.get(fingerprint)
                if prior is None or (prior["cache_reused"] and not call["cache_reused"]):
                    visual_calls_by_fingerprint[fingerprint] = call

        group_calls = []
        seen_boundary_reviews = set()
        for group in groups:
            for review in group.boundary_reviews:
                identity = review.get("input_fingerprint")
                if identity in seen_boundary_reviews:
                    continue
                seen_boundary_reviews.add(identity)
                candidate = review.get("result") or {}
                if "usage" in candidate:
                    group_calls.append({
                        "stage": "experiment_boundary_review", "group_id": group.group_id,
                        **{key: candidate.get(key) for key in ("provider", "model", "api_protocol", "request_id", "response_model", "status", "latency_seconds", "attempts")},
                        "cache_reused": bool(candidate.get("cache_reused")),
                        "attempt_receipts": candidate.get("attempt_receipts", []),
                        "usage": candidate.get("usage", {}),
                    })
            understanding = group.model_understanding or {}
            candidates = [
                (
                    "experiment_group_understanding_pre_curation",
                    understanding.get("pre_curation_understanding") or {},
                ),
                ("experiment_group_understanding", understanding),
            ]
            if (
                understanding.get("refinement_pass") == "deterministic_post_event_semantic_curation"
                and understanding.get("refinement_model_call_count") == 0
                and understanding.get("pre_curation_understanding")
            ):
                # Deterministic curation preserves the original call receipt in
                # both views; it does not make or charge a second model call.
                candidates = candidates[:1]
            for stage_name, candidate in candidates:
                if not (
                    "usage" in candidate
                    or candidate.get("status") in {"completed", "failed"}
                ):
                    continue
                group_calls.append(
                    {
                        "stage": stage_name,
                        **{key: candidate.get(key) for key in ("provider", "api_protocol", "request_id", "response_model")},
                        "group_id": group.group_id,
                        "model": candidate.get(
                            "model", self.config["mllm"]["model"]
                        ),
                        "status": candidate.get("status"),
                        "latency_seconds": candidate.get("latency_seconds"),
                        "attempts": candidate.get("attempts"),
                        "cache_reused": bool(candidate.get("cache_reused")),
                        "attempt_receipts": candidate.get("attempt_receipts", []),
                        "usage": candidate.get("usage", {}),
                    }
                )

        def token_sum(calls, field: str):
            values = [
                call["usage"].get(field)
                for call in calls
                if not call.get("cache_reused")
                and call.get("usage", {}).get(field) is not None
            ]
            return sum(values) if values else None

        key_materials = {
            "input_tokens": token_sum(key_calls, "input_tokens"),
            "output_tokens": token_sum(key_calls, "output_tokens"),
            "total_tokens": token_sum(key_calls, "total_tokens"),
            "cached_input_tokens": token_sum(key_calls, "cached_input_tokens"),
            "call_count": len(key_calls),
            "executed_call_count": sum(not call.get("cache_reused") for call in key_calls),
            "reused_call_count": sum(bool(call.get("cache_reused")) for call in key_calls),
            "server_reported_for_all_calls": bool(key_calls)
            and all(call.get("usage", {}).get("server_reported") for call in key_calls),
        }
        experiment_groups = {
            "input_tokens": token_sum(group_calls, "input_tokens"),
            "output_tokens": token_sum(group_calls, "output_tokens"),
            "total_tokens": token_sum(group_calls, "total_tokens"),
            "cached_input_tokens": token_sum(group_calls, "cached_input_tokens"),
            "call_count": len(group_calls),
            "executed_call_count": sum(not call.get("cache_reused") for call in group_calls),
            "reused_call_count": sum(bool(call.get("cache_reused")) for call in group_calls),
            "server_reported_for_all_calls": bool(group_calls)
            and all(call.get("usage", {}).get("server_reported") for call in group_calls),
        }
        visual_calls = list(visual_calls_by_fingerprint.values())
        visual_review_metrics = {
            "input_tokens": token_sum(visual_calls, "input_tokens"),
            "output_tokens": token_sum(visual_calls, "output_tokens"),
            "total_tokens": token_sum(visual_calls, "total_tokens"),
            "call_count": len(visual_calls),
            "executed_call_count": sum(not call["cache_reused"] for call in visual_calls),
            "reused_call_count": sum(call["cache_reused"] for call in visual_calls),
            "unknown_usage_call_count": sum(
                not call["cache_reused"] and call["usage"].get("total_tokens") is None
                for call in visual_calls
            ),
        }
        speech_calls = [
            {"stage": "recording_speech_understanding", **{
                key: part.get(key) for key in ("provider", "api_protocol", "request_id", "response_model", "status", "latency_seconds", "attempts", "input_fingerprint")
            }, "cache_reused": bool(part.get("cache_reused")), "usage": part.get("usage") or {}}
            for part in self._speech_understanding.get("parts", [])
        ]
        all_calls = group_calls + key_calls + visual_calls + speech_calls
        return {
            "run_started_at": self._run_started_iso,
            "run_ended_at": datetime.now(timezone.utc).isoformat(),
            "total_duration_seconds": round(time.perf_counter() - self._run_started_perf, 6),
            "preprocessing_sla": {
                "definition": "probe + alignment + coarse scan + fine scan + boundary audit",
                "target_seconds": float(self.config["performance"]["preprocessing_budget_seconds"]),
                "actual_seconds": self._preprocessing_completed_seconds,
                "met": (
                    self._preprocessing_completed_seconds
                    <= float(self.config["performance"]["preprocessing_budget_seconds"])
                    if self._preprocessing_completed_seconds is not None
                    else None
                ),
            },
            "stage_durations": list(self._stage_metrics),
            "startup_durations": dict(self._startup_metrics),
            "tokens": {
                "experiment_groups": experiment_groups,
                "key_materials": key_materials,
                "participant_visual_review": visual_review_metrics,
                "recording_speech_understanding": {
                    "call_count": len(speech_calls),
                    "executed_call_count": sum(not call["cache_reused"] for call in speech_calls),
                    **{field: token_sum(speech_calls, field) for field in ("input_tokens", "output_tokens", "total_tokens")},
                },
                "daily_report": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "call_count": 0,
                    "note": "Deterministic aggregation of already accepted evidence; no additional MLLM call.",
                },
                "run_total": {
                    "input_tokens": token_sum(all_calls, "input_tokens"),
                    "output_tokens": token_sum(all_calls, "output_tokens"),
                    "total_tokens": token_sum(all_calls, "total_tokens"),
                    "note": "MLLM step analysis and participant visual review consume model tokens; CV and FFmpeg consume no tokens.",
                },
            },
            "mllm_calls": all_calls,
            "mllm_request_policy": getattr(self, "_mllm_timeout_policy", None),
            "performance_mode": {
                "concurrent_input_views": self._input_view_count,
                "concurrent_role_scanners": bool(
                    self.config["performance"].get("concurrent_role_scanners", True)
                ),
                "bounded_streaming": True,
                "gpu_batching": True,
                "input_mode": self._input_mode,
                "source_copy_bytes": 0 if self._input_mode == "segmented_virtual_timeline" else None,
                "runtime_evidence": {
                    "resource_telemetry": "JSON-Config-Files/resource_telemetry.json",
                    "coarse_scan": "JSON-Config-Files/scan_runtime_coarse.json",
                    "fine_scan": "JSON-Config-Files/scan_runtime_fine.json",
                },
            },
        }

    def _scan_all_views_concurrently(
        self,
        manifest,
        infos,
        transforms,
        work_dir,
        *,
        windows=None,
        sample_fps=None,
        image_size=None,
        keyframes_only=False,
        phase="fine",
        decode_backends: dict[str, str] | None = None,
    ):
        role_groups = [
            [view for view in manifest.views if view.role == ViewRole.FIRST_PERSON],
            [view for view in manifest.views if view.role == ViewRole.THIRD_PERSON],
        ]
        role_groups = [group for group in role_groups if group]
        kwargs = {
            "windows": windows,
            "sample_fps": sample_fps,
            "image_size": image_size,
            "keyframes_only": keyframes_only,
            "phase": phase,
            "progress_callback": lambda view_id, completed, total: self._scan_progress(
                phase, view_id, completed, total
            ),
        }
        perf = self.config["performance"]
        concurrent_roles = (
            bool(perf.get("concurrent_role_scanners", True))
            and len(role_groups) > 1
        )
        workers_per_role = max(
            1, int(perf.get("yolo_inference_workers", 1))
        )
        same_role_workers = max(1, min(4, int(perf.get("fine_inference_workers_per_role", 1))))
        same_role_workers = min(
            same_role_workers, max(1, int(perf.get("fine_active_decode_slots", 1)))
        )
        parallel_same_role = (
            phase == "fine" and len(role_groups) == 1
            and same_role_workers > 1 and len(role_groups[0]) > 1
        )
        parallel_scanners = concurrent_roles or parallel_same_role
        if parallel_same_role:
            workers_per_role = same_role_workers
        scanner_groups: list[tuple[list[ViewInput], str | None]] = []
        for role_group in role_groups:
            worker_count = (
                min(workers_per_role, len(role_group))
                if parallel_scanners
                else 1
            )
            partitions = [role_group[index::worker_count] for index in range(worker_count)]
            partitions = [partition for partition in partitions if partition]
            for index, partition in enumerate(partitions, start=1):
                scanner_groups.append(
                    (
                        partition,
                        (
                            f"worker_{index:02d}"
                            if len(partitions) > 1
                            else None
                        ),
                    )
                )
        requested_sources = int(perf.get("source_workers", len(manifest.views)))
        if requested_sources < len(manifest.views):
            raise ValueError(
                f"source_workers={requested_sources} cannot keep {len(manifest.views)} views active"
            )
        lanes = list(
            perf.get(
                "coarse_decode_lanes"
                if phase in {"motion_probe", "coarse"}
                else "fine_decode_lanes",
                [],
            )
        )
        if not lanes:
            lanes = ["cuda" if perf.get("ffmpeg_hwaccel") else "cpu"] * len(manifest.views)
        if len(lanes) < len(manifest.views):
            lanes.extend([lanes[-1]] * (len(manifest.views) - len(lanes)))
        default_decode_backends = {
            view.view_id: lanes[index] for index, view in enumerate(manifest.views)
        }
        kwargs["decode_backends"] = {
            view.view_id: (decode_backends or default_decode_backends).get(
                view.view_id, default_decode_backends[view.view_id]
            )
            for view in manifest.views
        }
        if (
            phase == "coarse"
            and windows is None
            and perf.get("synchronized_segment_waves")
            and concurrent_roles
            and all(view.segments for view in manifest.views)
        ):
            segment_counts = {view.view_id: len(view.segments) for view in manifest.views}
            if len(set(segment_counts.values())) != 1:
                raise ValueError(
                    f"synchronized segment waves require equal segment counts: {segment_counts}"
                )
            kwargs["wave_barrier"] = threading.Barrier(len(manifest.views))
        for view in manifest.views:
            runtime = self._view_runtime.setdefault(view.view_id, {})
            runtime.update(
                {
                    "role": view.role.value,
                    "decode_backend": kwargs["decode_backends"][view.view_id],
                    "state": f"{phase}_running",
                }
            )
        work_dir.mkdir(parents=True, exist_ok=True)
        (work_dir / f"scheduler_{phase}.json").write_text(
            json.dumps(
                {
                    "schema_version": "visioncortex-role-scheduler/1",
                    "phase": phase,
                    "mode": (
                        "concurrent_same_role_workers"
                        if parallel_same_role
                        else
                        "concurrent_role_workers"
                        if concurrent_roles and len(scanner_groups) > len(role_groups)
                        else "concurrent_roles"
                        if concurrent_roles
                        else "sequential_role_residency"
                    ),
                    "role_order": [group[0].role.value for group in role_groups],
                    "yolo_inference_workers_per_role": workers_per_role,
                    "active_scanner_count": len(scanner_groups),
                    "scanner_groups": [
                        {
                            "scanner_id": scanner_id,
                            "role": group[0].role.value,
                            "view_ids": [view.view_id for view in group],
                        }
                        for group, scanner_id in scanner_groups
                    ],
                    "configured_decode_lanes": lanes,
                    "active_view_ids": [view.view_id for view in manifest.views],
                    "decode_backends": kwargs["decode_backends"],
                    "reason": (
                        "bounded same-role contexts sharing the configured decode-slot budget"
                        if parallel_same_role
                        else
                        "configured concurrent role scanners"
                        if concurrent_roles
                        else "one TensorRT role model resident at a time to preserve batch capacity"
                    ),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        if not parallel_scanners:
            result = {}
            for group in role_groups:
                group_kwargs = dict(kwargs)
                group_kwargs["decode_backends"] = {
                    view.view_id: kwargs["decode_backends"][view.view_id]
                    for view in group
                }
                if (
                    phase == "coarse"
                    and windows is None
                    and perf.get("synchronized_segment_waves")
                    and all(view.segments for view in group)
                    and len(group) > 1
                ):
                    group_kwargs["wave_barrier"] = threading.Barrier(len(group))
                for view in group:
                    self._view_runtime[view.view_id]["decode_backend"] = group_kwargs[
                        "decode_backends"
                    ][view.view_id]
                result.update(
                    scan_videos(
                        group,
                        infos,
                        transforms,
                        work_dir,
                        self.config,
                        **group_kwargs,
                    )
                )
            for view in manifest.views:
                self._view_runtime[view.view_id]["state"] = f"{phase}_completed"
            return result
        result = {}
        worker_config = self.config
        if parallel_same_role:
            worker_config = deepcopy(self.config)
            decode_slots = max(1, int(perf.get("fine_active_decode_slots", 1)))
            worker_config["performance"]["fine_active_decode_slots"] = max(
                1, decode_slots // len(scanner_groups)
            )
        with ThreadPoolExecutor(
            max_workers=len(scanner_groups),
            thread_name_prefix="role-scanner",
        ) as executor:
            futures = [
                executor.submit(
                    scan_videos,
                    group,
                    infos,
                    transforms,
                    work_dir,
                    worker_config,
                    scanner_id=scanner_id,
                    **kwargs,
                )
                for group, scanner_id in scanner_groups
            ]
            for future in futures:
                result.update(future.result())
        for view in manifest.views:
            self._view_runtime[view.view_id]["state"] = f"{phase}_completed"
        return result

    def _motion_probe_views(self, manifest: RunManifest) -> list[ViewInput]:
        """Choose sentinel views; bounded YOLO scans still use all eligible views."""

        perf = self.config["performance"]
        if perf.get("motion_probe_all_views", False):
            return list(manifest.views)
        first_limit = max(1, int(perf.get("motion_probe_first_person_views", 1)))
        third_limit = max(0, int(perf.get("motion_probe_third_person_views", 1)))
        first = [view for view in manifest.views if view.role == ViewRole.FIRST_PERSON]
        third = [view for view in manifest.views if view.role == ViewRole.THIRD_PERSON]
        return first[:first_limit] + third[:third_limit]

    def _coarse_scan_views(self, manifest: RunManifest) -> list[ViewInput]:
        """Choose boundary sentinels; fine validation still uses required views."""

        perf = self.config["performance"]
        if perf.get("coarse_all_views", False):
            return list(manifest.views)
        first_limit = max(1, int(perf.get("coarse_first_person_views", 1)))
        third_limit = max(1, int(perf.get("coarse_third_person_views", 1)))
        first = [view for view in manifest.views if view.role == ViewRole.FIRST_PERSON]
        third = [view for view in manifest.views if view.role == ViewRole.THIRD_PERSON]
        return first[:first_limit] + third[:third_limit]

    @staticmethod
    def _progressive_fine_view_order(
        fine_views: list[ViewInput],
        fine_view_report: dict[str, dict[str, Any]],
        initial_third_person_views: int,
        preferred_third_person_views: list[str] | None = None,
    ) -> tuple[list[ViewInput], list[ViewInput]]:
        """Return the initial views and ordered third-person fallback pool.

        An optional caller-provided preference may rank known deployment views.
        Otherwise each upload is ranked only by its own coarse anchor activity,
        with stable manifest order as the deterministic final tie-breaker.
        """

        positions = {view.view_id: index for index, view in enumerate(fine_views)}
        preferred_positions = {
            view_id: index
            for index, view_id in enumerate(preferred_third_person_views or [])
        }
        first = [view for view in fine_views if view.role == ViewRole.FIRST_PERSON]
        third = sorted(
            (view for view in fine_views if view.role == ViewRole.THIRD_PERSON),
            key=lambda view: (
                0 if view.view_id in preferred_positions else 1,
                preferred_positions.get(view.view_id, len(preferred_positions)),
                -int(fine_view_report.get(view.view_id, {}).get("active_anchor_frames", 0)),
                -int(fine_view_report.get(view.view_id, {}).get("anchor_frames", 0)),
                positions[view.view_id],
            ),
        )
        initial_count = min(len(third), max(0, int(initial_third_person_views)))
        return first + third[:initial_count], third[initial_count:]

    @staticmethod
    def _progressive_candidate_recall_bounds(
        candidate: ActionCandidate,
    ) -> tuple[float, float, str]:
        start = candidate.global_start_ms
        end = candidate.global_end_ms
        basis = "candidate_bounds"
        for item in candidate.evidence:
            if item.get("source") != "coarse_yolo_refinement":
                continue
            original_start = item.get("original_global_start_ms")
            original_end = item.get("original_global_end_ms")
            if original_start is None or original_end is None:
                continue
            start = min(start, float(original_start))
            end = max(end, float(original_end))
            basis = "original_motion_bounds_after_coarse_refinement"
        return start, end, basis

    @staticmethod
    def _quarantine_nonformal_progressive_gaps(
        target_status: list[dict[str, Any]],
        groups: list[ExperimentGroup],
    ) -> tuple[set[str], set[str]]:
        """Separate formal experiment gaps from exhausted single-view noise.

        Rejected first-person anchors are intentionally used while recall can
        still discover a corroborating third-person view. Once every eligible
        view is exhausted, a cluster containing only rejected anchors must not
        invalidate an otherwise proven dual-view experiment. It remains in the
        audit ledger and is never promoted into formal events or key materials.
        """

        unresolved: set[str] = set()
        quarantined: set[str] = set()
        for item in target_status:
            if item.get("status") != "needs_third_person_supplement":
                continue
            overlapping_groups = [
                group
                for group in groups
                if float(item["global_end_ms"]) >= group.global_start_ms
                and float(item["global_start_ms"]) <= group.global_end_ms
            ]
            candidate_id = str(item["candidate_id"])
            if not groups or not overlapping_groups:
                if not groups:
                    unresolved.add(candidate_id)
                    continue
                item["status"] = "quarantined_missing_dual_view"
                item["quarantine_reason"] = (
                    "all eligible third-person views exhausted and candidate does "
                    "not overlap any formal dual-view experiment group"
                )
                quarantined.add(candidate_id)
                continue
            uncovered_clusters = [
                cluster
                for cluster in item.get("anchor_clusters") or []
                if not cluster.get("covered")
            ]
            blocking_clusters = [
                cluster
                for cluster in uncovered_clusters
                if any(
                    float(cluster["global_end_ms"]) >= group.global_start_ms
                    and float(cluster["global_start_ms"]) <= group.global_end_ms
                    for group in overlapping_groups
                )
            ]
            anchor_windows = {
                str(anchor.get("event_id")): anchor
                for anchor in item.get("first_person_anchor_windows") or []
            }
            weak_blocking_clusters = [
                cluster
                for cluster in blocking_clusters
                if cluster.get("event_ids")
                and all(
                    event_id in anchor_windows
                    and anchor_windows[event_id].get("accepted") is False
                    for event_id in cluster["event_ids"]
                )
            ]
            strong_blocking_clusters = [
                cluster
                for cluster in blocking_clusters
                if cluster not in weak_blocking_clusters
            ]
            covered_clusters = [
                cluster
                for cluster in item.get("anchor_clusters") or []
                if cluster.get("covered")
                and any(
                    float(cluster["global_end_ms"]) >= group.global_start_ms
                    and float(cluster["global_start_ms"]) <= group.global_end_ms
                    for group in overlapping_groups
                )
            ]
            if strong_blocking_clusters or (blocking_clusters and not covered_clusters):
                unresolved.add(candidate_id)
                item["blocking_anchor_cluster_ids"] = [
                    str(cluster["cluster_id"])
                    for cluster in (
                        strong_blocking_clusters
                        if strong_blocking_clusters
                        else blocking_clusters
                    )
                ]
                continue
            item["status"] = (
                "cross_view_covered_with_quarantined_weak_anchor_context"
                if weak_blocking_clusters
                else "cross_view_covered_with_quarantined_single_view_context"
            )
            item["quarantine_reason"] = (
                "formal experiment has accepted dual-view anchor clusters; remaining "
                "exhausted first-person clusters contain only rejected audit candidates"
                if weak_blocking_clusters
                else "formal experiment clusters have dual-view support; remaining "
                "single-view context lies outside formal boundaries"
            )
            item["quarantined_anchor_cluster_ids"] = [
                str(cluster["cluster_id"])
                for cluster in (
                    weak_blocking_clusters
                    if weak_blocking_clusters
                    else uncovered_clusters
                )
            ]
            if not weak_blocking_clusters:
                quarantined.add(candidate_id)
        return unresolved, quarantined

    def _progressive_target_status(
        self,
        boundary_candidates: list[ActionCandidate],
        events: list[EvidenceEvent],
    ) -> list[dict[str, Any]]:
        """Classify which recalled windows still need another third-person view.

        A supplemental scan is demanded only by reliable first-person evidence.
        A window is covered only by an accepted, boundary-eligible event with
        both roles. This deliberately prevents a partial liquid hypothesis such
        as DEV-011 EVT-000444 from suppressing a needed supplemental scan.
        """

        audit_margin_ms = (
            float(
                self.config["performance"].get(
                    "fine_progressive_audit_margin_seconds", 5.0
                )
            )
            * 1000.0
        )
        results = []
        for candidate in boundary_candidates:
            recall_start, recall_end, recall_basis = (
                self._progressive_candidate_recall_bounds(candidate)
            )
            # Decode padding maximizes recall and may overlap adjacent experiments.
            # It must not be reused as the evidence-association window, otherwise
            # one event can falsely mark two nearby candidates as cross-view covered.
            # A coarse-refined candidate is the exception: its original motion
            # bounds remain the recall envelope so a late atomic action cannot be
            # suppressed by an early cross-view hit. Formal boundaries are still
            # computed only from audited evidence, never from this recall envelope.
            window_start = recall_start - audit_margin_ms
            window_end = recall_end + audit_margin_ms
            relevant = []
            for event in events:
                # The candidate/action ledger has already applied temporal and
                # physical-action construction. A first-person anchor rejected
                # only for missing cross-view support is exactly the condition
                # that must trigger a supplemental view.
                if event.action_type == ActionType.LIQUID_MOVEMENT:
                    complete_first_person_transfer = any(
                        candidate.role == ViewRole.FIRST_PERSON
                        and (
                            candidate.candidate_id.startswith("TRANSFER-SEQ-")
                            or any(
                                evidence.get("transfer_sequence")
                                == "source_transport_target"
                                for evidence in candidate.evidence
                            )
                        )
                        for candidate in event.candidates
                    )
                    # Tool/container proximity plus camera motion is only a
                    # semantic-review candidate.  It must not become a hard
                    # progressive boundary anchor before the model has proved
                    # liquid motion.  Deterministic CV may route recall only
                    # when it has a complete source/transport/target chain, or
                    # when observability explicitly says the action can define
                    # a boundary without semantic promotion.
                    reliable_start_signal = complete_first_person_transfer or (
                        event_is_formal(event)
                        and bool(
                            (event.observability or {}).get(
                                "can_define_boundary_without_semantic_promotion"
                            )
                        )
                    )
                else:
                    reliable_start_signal = is_experiment_start_anchor(
                        event, self.config
                    )
                    observability = event.observability or {}
                    requires_semantic_promotion = bool(
                        observability.get("semantic_review_priority") == "required"
                        and not observability.get(
                            "can_define_boundary_without_semantic_promotion",
                            False,
                        )
                    )
                    if requires_semantic_promotion:
                        reliable_start_signal = False
                if not reliable_start_signal:
                    continue
                midpoint = (event.global_start_ms + event.global_end_ms) / 2.0
                if window_start <= midpoint <= window_end:
                    relevant.append(event)
            first_signal = [
                event for event in relevant if ViewRole.FIRST_PERSON in event.supporting_roles
            ]
            cross_view = [
                event
                for event in first_signal
                if event_is_formal(event)
                and {ViewRole.FIRST_PERSON, ViewRole.THIRD_PERSON}.issubset(
                    set(event.supporting_roles)
                )
            ]
            cluster_gap_ms = (
                float(
                    self.config["performance"].get(
                        "fine_progressive_anchor_cluster_gap_seconds", 10.0
                    )
                )
                * 1000.0
            )
            clusters: list[list[EvidenceEvent]] = []
            for event in sorted(first_signal, key=event_sort_key):
                if (
                    clusters
                    and event.global_start_ms - max(
                        item.global_end_ms for item in clusters[-1]
                    )
                    <= cluster_gap_ms
                ):
                    clusters[-1].append(event)
                else:
                    clusters.append([event])
            cross_view_ids = {event.event_id for event in cross_view}
            cluster_receipts = []
            supplement_events: list[EvidenceEvent] = []
            for index, cluster in enumerate(clusters, start=1):
                cluster_cross_view = [
                    event for event in cluster if event.event_id in cross_view_ids
                ]
                covered = bool(cluster_cross_view)
                if not covered:
                    supplement_events.extend(cluster)
                cluster_receipts.append(
                    {
                        "cluster_id": f"{candidate.candidate_id}-ANCHOR-{index:03d}",
                        "global_start_ms": min(
                            event.global_start_ms for event in cluster
                        ),
                        "global_end_ms": max(
                            event.global_end_ms for event in cluster
                        ),
                        "event_ids": [event.event_id for event in cluster],
                        "cross_view_event_ids": [
                            event.event_id for event in cluster_cross_view
                        ],
                        "covered": covered,
                    }
                )
            if first_signal and not supplement_events:
                status = "cross_view_covered"
            elif first_signal:
                status = "needs_third_person_supplement"
            else:
                status = "no_reliable_first_person_activity"
            results.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "global_start_ms": candidate.global_start_ms,
                    "global_end_ms": candidate.global_end_ms,
                    "recall_window_basis": recall_basis,
                    "recall_window_start_ms": recall_start,
                    "recall_window_end_ms": recall_end,
                    "audit_window_start_ms": window_start,
                    "audit_window_end_ms": window_end,
                    "status": status,
                    "first_person_anchor_event_ids": [
                        event.event_id for event in first_signal
                    ],
                    "first_person_anchor_windows": [
                        {
                            "event_id": event.event_id,
                            "global_start_ms": event.global_start_ms,
                            "global_end_ms": event.global_end_ms,
                            "key_global_ms": event.key_global_ms,
                            "confidence": event.confidence,
                            "accepted": event.accepted,
                            "formal_admission_status": event.formal_admission_status,
                            "action_type": event.action_type.value,
                            "objects": list(event.objects),
                        }
                        for event in supplement_events
                    ],
                    "cross_view_anchor_event_ids": [event.event_id for event in cross_view],
                    "anchor_cluster_gap_ms": cluster_gap_ms,
                    "anchor_clusters": cluster_receipts,
                    "uncovered_anchor_cluster_ids": [
                        item["cluster_id"]
                        for item in cluster_receipts
                        if not item["covered"]
                    ],
                }
            )
        return results

    @staticmethod
    def _progressive_anchor_windows(
        target_status: list[dict[str, Any]],
        view_ids: list[str],
        infos,
        transforms,
        padding_seconds: float,
        candidate_ids: set[str] | None = None,
        peak_radius_seconds: float | None = None,
    ) -> dict[str, list[tuple[float, float]]]:
        """Build aligned narrow windows around reliable first-person events."""

        padding_ms = max(0.0, float(padding_seconds)) * 1000.0
        peak_radius_ms = (
            max(0.0, float(peak_radius_seconds)) * 1000.0
            if peak_radius_seconds is not None
            else None
        )
        grouped: dict[str, list[tuple[float, float]]] = {
            view_id: [] for view_id in view_ids
        }
        for item in target_status:
            if candidate_ids is not None and item["candidate_id"] not in candidate_ids:
                continue
            for anchor in item.get("first_person_anchor_windows") or []:
                if peak_radius_ms is None:
                    global_start = float(anchor["global_start_ms"]) - padding_ms
                    global_end = float(anchor["global_end_ms"]) + padding_ms
                else:
                    peak_ms = float(anchor["key_global_ms"])
                    global_start = peak_ms - peak_radius_ms
                    global_end = peak_ms + peak_radius_ms
                for view_id in view_ids:
                    local_start = max(
                        0.0, transforms[view_id].to_local(global_start)
                    )
                    local_end = min(
                        infos[view_id].duration_ms,
                        transforms[view_id].to_local(global_end),
                    )
                    if local_end > local_start:
                        grouped[view_id].append((local_start, local_end))

        merged: dict[str, list[tuple[float, float]]] = {}
        for view_id, windows in grouped.items():
            result: list[list[float]] = []
            for start, end in sorted(windows):
                if result and start <= result[-1][1]:
                    result[-1][1] = max(result[-1][1], end)
                else:
                    result.append([start, end])
            merged[view_id] = [(item[0], item[1]) for item in result]
        return merged

    @staticmethod
    def _progressive_scout_anchor_representatives(
        target_status: list[dict[str, Any]],
        candidate_ids: set[str],
        *,
        dedup_tolerance_seconds: float,
        cluster_gap_seconds: float,
        representatives_per_cluster: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Collapse repeated candidate references into sparse routing anchors.

        This affects only the low-FPS view-ranking scout. Formal 10 FPS evidence
        continues to use every aligned first-person action span.
        """

        occurrences: list[dict[str, Any]] = []
        for item in target_status:
            candidate_id = str(item["candidate_id"])
            if candidate_id not in candidate_ids:
                continue
            for raw_anchor in item.get("first_person_anchor_windows") or []:
                anchor = dict(raw_anchor)
                anchor["candidate_ids"] = [candidate_id]
                occurrences.append(anchor)

        by_event: dict[str, dict[str, Any]] = {}
        for index, anchor in enumerate(occurrences):
            event_id = str(anchor.get("event_id") or f"anonymous-{index:06d}")
            existing = by_event.get(event_id)
            if existing is None:
                anchor["event_id"] = event_id
                by_event[event_id] = anchor
                continue
            existing["candidate_ids"] = sorted(
                set(existing.get("candidate_ids") or [])
                | set(anchor.get("candidate_ids") or [])
            )

        tolerance_ms = max(0.0, float(dedup_tolerance_seconds)) * 1000.0
        peak_groups: list[list[dict[str, Any]]] = []
        for anchor in sorted(
            by_event.values(), key=lambda item: float(item["key_global_ms"])
        ):
            if (
                peak_groups
                and float(anchor["key_global_ms"])
                - float(peak_groups[-1][-1]["key_global_ms"])
                <= tolerance_ms
            ):
                peak_groups[-1].append(anchor)
            else:
                peak_groups.append([anchor])

        unique_peaks: list[dict[str, Any]] = []
        for group in peak_groups:
            representative = max(
                group,
                key=lambda item: (
                    float(item.get("confidence", 0.0)),
                    -max(
                        0.0,
                        float(item.get("global_end_ms", 0.0))
                        - float(item.get("global_start_ms", 0.0)),
                    ),
                    str(item.get("event_id", "")),
                ),
            ).copy()
            representative["event_ids"] = sorted(
                str(item["event_id"]) for item in group
            )
            representative["candidate_ids"] = sorted(
                {
                    candidate_id
                    for item in group
                    for candidate_id in item.get("candidate_ids") or []
                }
            )
            unique_peaks.append(representative)

        cluster_gap_ms = max(0.0, float(cluster_gap_seconds)) * 1000.0
        clusters: list[list[dict[str, Any]]] = []
        for anchor in unique_peaks:
            if (
                clusters
                and float(anchor["key_global_ms"])
                - float(clusters[-1][-1]["key_global_ms"])
                <= cluster_gap_ms
            ):
                clusters[-1].append(anchor)
            else:
                clusters.append([anchor])

        limit = max(1, int(representatives_per_cluster))
        selected: list[dict[str, Any]] = []
        cluster_reports: list[dict[str, Any]] = []
        for cluster_index, cluster in enumerate(clusters, 1):
            peak_values = [float(item["key_global_ms"]) for item in cluster]
            median_peak = float(np.median(np.asarray(peak_values, dtype=np.float64)))
            ranked = sorted(
                cluster,
                key=lambda item: (
                    abs(float(item["key_global_ms"]) - median_peak),
                    -float(item.get("confidence", 0.0)),
                    str(item.get("event_id", "")),
                ),
            )
            chosen = sorted(
                ranked[: min(limit, len(ranked))],
                key=lambda item: float(item["key_global_ms"]),
            )
            selected.extend(chosen)
            cluster_reports.append(
                {
                    "cluster_index": cluster_index,
                    "global_start_ms": min(peak_values),
                    "global_end_ms": max(peak_values),
                    "unique_peak_count": len(cluster),
                    "candidate_ids": sorted(
                        {
                            candidate_id
                            for item in cluster
                            for candidate_id in item.get("candidate_ids") or []
                        }
                    ),
                    "representative_event_ids": [
                        str(item["event_id"]) for item in chosen
                    ],
                    "representative_peak_ms": [
                        float(item["key_global_ms"]) for item in chosen
                    ],
                }
            )

        diagnostics = {
            "candidate_count": len(candidate_ids),
            "raw_anchor_occurrence_count": len(occurrences),
            "unique_event_count": len(by_event),
            "unique_peak_count": len(unique_peaks),
            "dedup_tolerance_seconds": float(dedup_tolerance_seconds),
            "cluster_gap_seconds": float(cluster_gap_seconds),
            "cluster_count": len(clusters),
            "representatives_per_cluster": limit,
            "representative_count": len(selected),
            "clusters": cluster_reports,
        }
        return selected, diagnostics

    @staticmethod
    def _merge_time_windows(
        windows: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        merged: list[list[float]] = []
        for start, end in sorted(
            (float(start), float(end))
            for start, end in windows
            if end > start
        ):
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        return [(start, end) for start, end in merged]

    @staticmethod
    def _window_fully_covered(
        start_ms: float,
        end_ms: float,
        coverage: list[tuple[float, float]],
        tolerance_ms: float = 100.0,
    ) -> bool:
        return any(
            covered_start <= start_ms + tolerance_ms
            and covered_end >= end_ms - tolerance_ms
            for covered_start, covered_end in coverage
        )

    @classmethod
    def _uncovered_time_windows(
        cls,
        requested: list[tuple[float, float]],
        coverage: list[tuple[float, float]],
        tolerance_ms: float = 100.0,
    ) -> list[tuple[float, float]]:
        """Return only intervals that can add new decoded evidence.

        Group-local recall may still regard a global FP anchor as unresolved
        when a TP recording ends just before that anchor.  Converting the
        anchor back to the TP timeline clips it at the physical media end.  If
        that clipped interval was already scanned, decoding it again cannot
        improve recall and used to repeat until the maximum-round guard fired.
        """

        merged_requested = cls._merge_time_windows(requested)
        merged_coverage = cls._merge_time_windows(coverage)
        uncovered: list[tuple[float, float]] = []
        for requested_start, requested_end in merged_requested:
            cursor = requested_start
            for covered_start, covered_end in merged_coverage:
                if covered_end <= cursor + tolerance_ms:
                    continue
                if covered_start >= requested_end - tolerance_ms:
                    break
                if covered_start > cursor + tolerance_ms:
                    uncovered.append((cursor, min(covered_start, requested_end)))
                cursor = max(cursor, covered_end)
                if cursor >= requested_end - tolerance_ms:
                    break
            if cursor < requested_end - tolerance_ms:
                uncovered.append((cursor, requested_end))
        return cls._merge_time_windows(uncovered)

    def _group_local_recall_plan(
        self,
        groups: list[ExperimentGroup],
        segments: list[ExperimentSegment],
        events: list[EvidenceEvent],
        candidates: list[ActionCandidate],
        fine_views: list[ViewInput],
        fine_view_report: dict[str, dict[str, Any]],
        actual_windows: dict[str, list[tuple[float, float]]],
        infos,
        transforms,
        boundary_candidates: list[ActionCandidate] | None = None,
    ) -> dict[str, Any]:
        """Plan TP supplementation per unresolved temporal cluster.

        A single early dual-role event cannot close a broad refined target.  FP
        anchors are evaluated across the entire overlapping coarse boundary,
        clustered in time, and each unresolved cluster is supplemented until a
        TP association is found or every eligible TP view is exhausted.
        """

        perf = self.config["performance"]
        minimum_unresolved = max(
            1, int(perf.get("fine_group_recall_min_unresolved_anchors", 3))
        )
        padding_ms = max(
            0.0, float(perf.get("fine_group_recall_padding_seconds", 0.0))
        ) * 1000.0
        cluster_gap_ms = max(
            0.0,
            float(perf.get("fine_group_recall_cluster_gap_seconds", 10.0))
            * 1000.0,
        )
        zero_prior_fallback = bool(
            perf.get("fine_group_recall_zero_prior_quality_fallback", True)
        )
        by_segment = {segment.segment_id: segment for segment in segments}
        third_views = [
            view for view in fine_views if view.role == ViewRole.THIRD_PERSON
        ]
        positions = {view.view_id: index for index, view in enumerate(fine_views)}
        reports: list[dict[str, Any]] = []
        selected_plans: list[dict[str, Any]] = []
        decision_receipts: list[dict[str, Any]] = []

        for group in groups:
            event_ids = {
                event_id
                for segment_id in group.atomic_experiment_ids
                if segment_id in by_segment
                for event_id in by_segment[segment_id].event_ids
            }
            group_events = [
                event
                for event in events
                if event_is_formal(event) and event.event_id in event_ids
            ]
            cross_view_events = [
                event
                for event in group_events
                if {ViewRole.FIRST_PERSON, ViewRole.THIRD_PERSON}.issubset(
                    set(event.supporting_roles)
                )
            ]
            overlapping_targets = [
                candidate
                for candidate in (boundary_candidates or [])
                if self._progressive_candidate_recall_bounds(candidate)[1]
                >= group.global_start_ms
                and self._progressive_candidate_recall_bounds(candidate)[0]
                <= group.global_end_ms
            ]
            target_start_ms = min(
                [group.global_start_ms]
                + [
                    self._progressive_candidate_recall_bounds(item)[0]
                    for item in overlapping_targets
                ]
            )
            target_end_ms = max(
                [group.global_end_ms]
                + [
                    self._progressive_candidate_recall_bounds(item)[1]
                    for item in overlapping_targets
                ]
            )
            fp_candidates = [
                candidate
                for candidate in candidates
                if candidate.role == ViewRole.FIRST_PERSON
                and candidate.global_end_ms >= target_start_ms
                and candidate.global_start_ms <= target_end_ms
            ]

            def candidate_supported(candidate: ActionCandidate) -> bool:
                for event in cross_view_events:
                    if event.action_type != candidate.action_type:
                        continue
                    if candidate.objects and event.objects and not (
                        set(candidate.objects) & set(event.objects)
                    ):
                        continue
                    overlap = min(
                        event.global_end_ms, candidate.global_end_ms
                    ) - max(event.global_start_ms, candidate.global_start_ms)
                    if overlap >= 0.0 or abs(
                        event.key_global_ms - candidate.key_global_ms
                    ) <= 1500.0:
                        return True
                return False

            raw_unresolved = [
                candidate
                for candidate in fp_candidates
                if not candidate_supported(candidate)
            ]
            raw_unresolved_clusters: list[list[ActionCandidate]] = []
            for candidate in sorted(raw_unresolved, key=candidate_sort_key):
                if (
                    raw_unresolved_clusters
                    and candidate.global_start_ms
                    <= max(
                        item.global_end_ms for item in raw_unresolved_clusters[-1]
                    )
                    + cluster_gap_ms
                ):
                    raw_unresolved_clusters[-1].append(candidate)
                else:
                    raw_unresolved_clusters.append([candidate])
            raw_cluster_reports: list[dict[str, Any]] = []
            cross_view_supported_candidate_ids: set[str] = set()
            for index, cluster in enumerate(raw_unresolved_clusters, 1):
                cluster_start = min(item.global_start_ms for item in cluster)
                cluster_end = max(item.global_end_ms for item in cluster)
                supporting_cross_view_events = [
                    event
                    for event in cross_view_events
                    if event.global_end_ms >= cluster_start
                    and event.global_start_ms <= cluster_end
                ]
                supported = bool(supporting_cross_view_events)
                if supported:
                    cross_view_supported_candidate_ids.update(
                        item.candidate_id for item in cluster
                    )
                raw_cluster_reports.append({
                    "cluster_id": f"{group.group_id}-UNRESOLVED-{index:03d}",
                    "global_start_ms": cluster_start,
                    "global_end_ms": cluster_end,
                    "candidate_ids": [item.candidate_id for item in cluster],
                    "candidate_count": len(cluster),
                    "cross_view_event_ids": [
                        event.event_id for event in supporting_cross_view_events
                    ],
                    "cross_view_supported": supported,
                    "status": (
                        "satisfied_by_temporally_overlapping_cross_view_event"
                        if supported
                        else "unresolved"
                    ),
                })
            unresolved = [
                candidate
                for candidate in raw_unresolved
                if candidate.candidate_id not in cross_view_supported_candidate_ids
            ]
            cluster_reports = [
                item
                for item in raw_cluster_reports
                if not item["cross_view_supported"]
            ]
            supported_cluster_reports = [
                item
                for item in raw_cluster_reports
                if item["cross_view_supported"]
            ]
            base_report: dict[str, Any] = {
                "group_id": group.group_id,
                "global_start_ms": group.global_start_ms,
                "global_end_ms": group.global_end_ms,
                "refined_target_start_ms": target_start_ms,
                "refined_target_end_ms": target_end_ms,
                "refined_target_candidate_ids": [
                    item.candidate_id for item in overlapping_targets
                ],
                "first_person_candidate_count": len(fp_candidates),
                "cross_view_event_count": len(cross_view_events),
                "raw_candidate_level_unresolved_anchor_count": len(raw_unresolved),
                "unresolved_anchor_count": len(unresolved),
                "minimum_unresolved_anchors": minimum_unresolved,
                "unresolved_candidate_ids": [
                    candidate.candidate_id for candidate in unresolved
                ],
                "unresolved_temporal_clusters": cluster_reports,
                "unresolved_temporal_cluster_count": len(cluster_reports),
                "cross_view_supported_temporal_clusters": supported_cluster_reports,
                "cross_view_supported_temporal_cluster_count": len(
                    supported_cluster_reports
                ),
            }
            if len(unresolved) < minimum_unresolved:
                base_report["status"] = "complete_below_recall_trigger"
                reports.append(base_report)
                decision_receipts.append(
                    decision_receipt(
                        decision_type="cross_view_cluster_recall",
                        rule_id="QF3-TEMPORAL-CLUSTER-COMPLETENESS",
                        verdict="not_required",
                        subject_ids=[group.group_id],
                        reason_codes=["below_unresolved_anchor_trigger"],
                        facts={
                            "unresolved_anchor_count": len(unresolved),
                            "cluster_count": len(cluster_reports),
                        },
                        thresholds={
                            "minimum_unresolved_anchors": minimum_unresolved,
                        },
                        evidence_refs=[
                            candidate.candidate_id for candidate in unresolved
                        ],
                        legacy={
                            "group_id": group.group_id,
                            "decision": "recall_not_required",
                        },
                    )
                )
                continue

            choices: list[dict[str, Any]] = []
            for view in third_views:
                view_id = view.view_id
                global_coverage = self._merge_time_windows(
                    [
                        (
                            transforms[view_id].to_global(start),
                            transforms[view_id].to_global(end),
                        )
                        for start, end in actual_windows.get(view_id, [])
                    ]
                )
                missing = [
                    candidate
                    for candidate in unresolved
                    if not self._window_fully_covered(
                        candidate.global_start_ms,
                        candidate.global_end_ms,
                        global_coverage,
                    )
                ]
                if not missing:
                    continue
                missing_ids = {candidate.candidate_id for candidate in missing}
                missing_clusters = [
                    item
                    for item in cluster_reports
                    if missing_ids & set(item["candidate_ids"])
                ]
                requested_local_windows = self._merge_time_windows(
                    [
                        (
                            max(
                                0.0,
                                transforms[view_id].to_local(
                                    max(
                                        target_start_ms,
                                        candidate.global_start_ms - padding_ms,
                                    )
                                ),
                            ),
                            min(
                                infos[view_id].duration_ms,
                                transforms[view_id].to_local(
                                    min(
                                        target_end_ms,
                                        candidate.global_end_ms + padding_ms,
                                    )
                                ),
                            ),
                        )
                        for candidate in missing
                    ]
                )
                local_windows = self._uncovered_time_windows(
                    requested_local_windows,
                    actual_windows.get(view_id, []),
                )
                if not local_windows:
                    continue
                local_support = sum(
                    view_id in event.supporting_views for event in group_events
                )
                local_candidate_count = sum(
                    candidate.view_id == view_id
                    and candidate.global_end_ms >= group.global_start_ms
                    and candidate.global_start_ms <= group.global_end_ms
                    for candidate in candidates
                )
                covered_group_seconds = sum(
                    max(
                        0.0,
                        min(end, transforms[view_id].to_local(group.global_end_ms))
                        - max(
                            start,
                            transforms[view_id].to_local(group.global_start_ms),
                        ),
                    )
                    for start, end in actual_windows.get(view_id, [])
                ) / 1000.0
                choices.append(
                    {
                        "view_id": view_id,
                        "windows": local_windows,
                        "requested_windows": requested_local_windows,
                        "incremental_window_policy": "uncovered_local_timeline_only",
                        "missing_anchor_count": len(missing),
                        "missing_candidate_ids": [
                            candidate.candidate_id for candidate in missing
                        ],
                        "missing_cluster_ids": [
                            str(item["cluster_id"]) for item in missing_clusters
                        ],
                        "existing_group_event_support": local_support,
                        "existing_group_candidate_count": local_candidate_count,
                        "covered_group_seconds": round(
                            covered_group_seconds, 6
                        ),
                        "evidence_yield_per_selected_second": round(
                            local_support / max(covered_group_seconds, 1e-9), 9
                        ),
                        "coarse_active_anchor_frames": int(
                            fine_view_report.get(view_id, {}).get(
                                "active_anchor_frames", 0
                            )
                        ),
                        "coarse_anchor_frames": int(
                            fine_view_report.get(view_id, {}).get(
                                "anchor_frames", 0
                            )
                        ),
                        "manifest_position": positions[view_id],
                        "positive_recall_prior": bool(
                            local_support
                            or local_candidate_count
                            or int(
                                fine_view_report.get(view_id, {}).get(
                                    "active_anchor_frames", 0
                                )
                            )
                            or int(
                                fine_view_report.get(view_id, {}).get(
                                    "anchor_frames", 0
                                )
                            )
                        ),
                    }
                )
            if not choices:
                base_report["status"] = "unresolved_all_tp_windows_exhausted"
                base_report["quality_fallback_exhausted"] = True
                reports.append(base_report)
                for cluster in cluster_reports:
                    decision_receipts.append(
                        decision_receipt(
                            decision_type="cross_view_cluster_recall",
                            rule_id="QF3-TEMPORAL-CLUSTER-COMPLETENESS",
                            verdict="exhausted",
                            subject_ids=[
                                group.group_id,
                                str(cluster["cluster_id"]),
                            ],
                            reason_codes=["all_eligible_tp_views_exhausted"],
                            facts=cluster,
                            thresholds={
                                "cluster_gap_ms": cluster_gap_ms,
                                "minimum_unresolved_anchors": minimum_unresolved,
                            },
                            evidence_refs=cluster["candidate_ids"],
                            legacy={
                                "group_id": group.group_id,
                                "cluster_id": cluster["cluster_id"],
                                "decision": "recall_views_exhausted",
                            },
                        )
                    )
                continue
            choices.sort(
                key=lambda item: (
                    -int(item["existing_group_event_support"]),
                    -int(item["existing_group_candidate_count"]),
                    -float(item["evidence_yield_per_selected_second"]),
                    -int(item["coarse_active_anchor_frames"]),
                    -int(item["coarse_anchor_frames"]),
                    int(item["manifest_position"]),
                )
            )
            base_report["ranked_view_choices"] = choices
            all_zero_prior = not any(
                bool(item["positive_recall_prior"]) for item in choices
            )
            if all_zero_prior and not zero_prior_fallback:
                base_report["status"] = "complete_no_positive_recall_prior"
                base_report["recall_guard_reason"] = (
                    "formal group already has first/third evidence and every "
                    "unscanned view has zero event, candidate, and coarse-anchor prior"
                )
                reports.append(base_report)
                continue
            chosen = choices[0]
            base_report.update(
                {
                    "status": "needs_group_local_recall",
                    "selected_view_id": chosen["view_id"],
                    "selected_windows": chosen["windows"],
                    "selection_mode": (
                        "zero_prior_quality_fallback"
                        if all_zero_prior
                        else "evidence_prior_ranked"
                    ),
                }
            )
            reports.append(base_report)
            selected_plans.append(
                {
                    "group_id": group.group_id,
                    "view_id": chosen["view_id"],
                    "windows": chosen["windows"],
                    "missing_anchor_count": chosen["missing_anchor_count"],
                    "missing_candidate_ids": chosen["missing_candidate_ids"],
                    "missing_cluster_ids": chosen["missing_cluster_ids"],
                    "refined_target_start_ms": target_start_ms,
                    "refined_target_end_ms": target_end_ms,
                }
            )
            for cluster in cluster_reports:
                if str(cluster["cluster_id"]) not in set(
                    chosen["missing_cluster_ids"]
                ):
                    continue
                decision_receipts.append(
                    decision_receipt(
                        decision_type="cross_view_cluster_recall",
                        rule_id="QF3-TEMPORAL-CLUSTER-COMPLETENESS",
                        verdict="selected",
                        subject_ids=[
                            group.group_id,
                            str(cluster["cluster_id"]),
                            str(chosen["view_id"]),
                        ],
                        reason_codes=[
                            "zero_prior_quality_fallback"
                            if all_zero_prior
                            else "ranked_tp_evidence_prior"
                        ],
                        facts={
                            **cluster,
                            "selected_view_id": chosen["view_id"],
                            "selected_windows": chosen["windows"],
                            "refined_target_start_ms": target_start_ms,
                            "refined_target_end_ms": target_end_ms,
                        },
                        thresholds={
                            "cluster_gap_ms": cluster_gap_ms,
                            "minimum_unresolved_anchors": minimum_unresolved,
                            "zero_prior_quality_fallback": zero_prior_fallback,
                        },
                        evidence_refs=cluster["candidate_ids"],
                        legacy={
                            "group_id": group.group_id,
                            "cluster_id": cluster["cluster_id"],
                            "decision": "selected_tp_supplemental_scan",
                            "selected_view_id": chosen["view_id"],
                        },
                    )
                )
        operational_complete = not selected_plans
        quality_complete = operational_complete and not any(
            item.get("status")
            in {
                "unresolved_all_tp_windows_exhausted",
                "complete_no_positive_recall_prior",
            }
            for item in reports
        )
        return {
            "schema_version": "visioncortex-group-local-recall-plan/2",
            "minimum_unresolved_anchors": minimum_unresolved,
            "cluster_gap_ms": cluster_gap_ms,
            "zero_prior_quality_fallback": zero_prior_fallback,
            "groups": reports,
            "selected_plans": selected_plans,
            "decision_receipts": decision_receipts,
            "complete": operational_complete,
            "quality_complete": quality_complete,
        }

    def _run_progressive_fine_scan(
        self,
        manifest: RunManifest,
        fine_views: list[ViewInput],
        fine_view_report: dict[str, dict[str, Any]],
        boundary_candidates: list[ActionCandidate],
        fine_windows: dict[str, list[tuple[float, float]]],
        infos,
        transforms,
        work_dir: Path,
    ) -> tuple[dict[str, Path], list[ViewInput], list[ActionCandidate], dict[str, Any]]:
        """Resolve dual-view gaps with optional scout-ranked aligned narrow windows."""

        perf = self.config["performance"]
        fine_frame_index: FineFrameIndex | None = None
        fine_index_ingest_reports: list[dict[str, Any]] = []
        if bool(perf.get("fine_frame_index_enabled", False)):
            fine_frame_index = create_fine_frame_index(
                self._frame_index_path(work_dir, "fine_frame_index.sqlite3")
            )
        ordered_initial, ordered_supplemental = self._progressive_fine_view_order(
            fine_views,
            fine_view_report,
            int(perf.get("fine_initial_third_person_views", 1)),
            list(perf.get("fine_preferred_third_person_views") or []),
        )
        if not any(
            view.role == ViewRole.THIRD_PERSON
            for view in ordered_initial + ordered_supplemental
        ):
            raise ValueError("progressive fine scan requires at least one third-person view")
        scout_enabled = bool(perf.get("fine_dynamic_cross_view_scout", False))
        if scout_enabled:
            initial_views = [
                view for view in fine_views if view.role == ViewRole.FIRST_PERSON
            ]
            supplemental_views = [
                view
                for view in ordered_initial + ordered_supplemental
                if view.role == ViewRole.THIRD_PERSON
            ]
        else:
            initial_views = ordered_initial
            supplemental_views = ordered_supplemental

        lanes = list(perf.get("fine_decode_lanes") or [])
        if not lanes:
            lanes = ["cuda" if perf.get("ffmpeg_hwaccel") else "cpu"] * len(
                manifest.views
            )
        if len(lanes) < len(manifest.views):
            lanes.extend([lanes[-1]] * (len(manifest.views) - len(lanes)))
        full_decode_backends = {
            view.view_id: lanes[index] for index, view in enumerate(manifest.views)
        }

        scanned: dict[str, ViewInput] = {}
        detection_paths: dict[str, Path] = {}
        actual_windows: dict[str, list[tuple[float, float]]] = {}
        pass_reports: list[dict[str, Any]] = []
        scout_summary: dict[str, Any] = {"enabled": scout_enabled}

        def generate_current_candidates(
            current_views: list[ViewInput],
        ) -> list[ActionCandidate]:
            if fine_frame_index is None:
                return generate_candidates(
                    current_views, detection_paths, self.config
                )
            return generate_candidates(
                current_views,
                detection_paths,
                self.config,
                fine_frame_index,
            )

        def refine_current_liquid_context(
            current_events: list[EvidenceEvent],
        ) -> list[dict[str, Any]]:
            if fine_frame_index is None:
                return refine_liquid_events_with_context(
                    current_events,
                    detection_paths,
                )
            return refine_liquid_events_with_context(
                current_events,
                detection_paths,
                frame_index=fine_frame_index,
                config=self.config,
            )

        def execute_pass(
            pass_index: int,
            pass_kind: str,
            pass_views: list[ViewInput],
            pass_candidates: list[ActionCandidate],
            target_snapshot: list[dict[str, Any]] | None = None,
            explicit_windows: dict[str, list[tuple[float, float]]] | None = None,
        ) -> list[dict[str, Any]]:
            pass_name = f"pass-{pass_index:02d}-{pass_kind}"
            if explicit_windows is not None:
                pass_windows = explicit_windows
                window_strategy = "formal_group_local_recall_windows"
            elif pass_kind == "primary":
                pass_windows = {
                    view.view_id: fine_windows[view.view_id] for view in pass_views
                }
                window_strategy = "coarse_candidate_windows"
            elif target_snapshot is not None:
                pass_windows = self._progressive_anchor_windows(
                    target_snapshot,
                    [view.view_id for view in pass_views],
                    infos,
                    transforms,
                    float(
                        perf.get(
                            "fine_progressive_anchor_padding_seconds", 10.0
                        )
                    ),
                    {candidate.candidate_id for candidate in pass_candidates},
                )
                if not any(pass_windows.values()):
                    all_pass_windows = self._fine_windows(
                        pass_candidates, infos, transforms
                    )
                    pass_windows = {
                        view.view_id: all_pass_windows[view.view_id]
                        for view in pass_views
                    }
                    window_strategy = "coarse_candidate_fallback"
                else:
                    window_strategy = "aligned_first_person_anchor_windows"
            else:
                all_pass_windows = self._fine_windows(
                    pass_candidates, infos, transforms
                )
                pass_windows = {
                    view.view_id: all_pass_windows[view.view_id] for view in pass_views
                }
                window_strategy = "coarse_candidate_windows"
            pass_manifest = manifest.model_copy(update={"views": pass_views})
            result = self._scan_all_views_concurrently(
                pass_manifest,
                infos,
                transforms,
                work_dir / pass_name,
                windows=pass_windows,
                sample_fps=float(perf["detection_fps"]),
                phase="fine",
                decode_backends={
                    view.view_id: full_decode_backends[view.view_id] for view in pass_views
                },
            )
            merge_reports: list[dict[str, Any]] = []
            index_ingest_report: dict[str, Any] | None = None
            if fine_frame_index is not None:
                index_ingest_report = ingest_fine_frame_ledgers(
                    fine_frame_index,
                    pass_views,
                    result,
                    source_pass=pass_name,
                    stitching_enabled=bool(
                        perf.get("fine_track_stitching_enabled", False)
                    ),
                    maximum_stitch_gap_ms=float(
                        perf.get("fine_track_stitch_max_gap_seconds", 2.5)
                    )
                    * 1000.0,
                    maximum_center_distance=float(
                        perf.get(
                            "fine_track_stitch_max_center_distance", 0.12
                        )
                    ),
                )
                fine_index_ingest_reports.append(index_ingest_report)
            for view in pass_views:
                scanned[view.view_id] = view
                existing_path = detection_paths.get(view.view_id)
                if existing_path is None:
                    detection_paths[view.view_id] = result[view.view_id]
                elif fine_frame_index is None:
                    merged_path = (
                        work_dir
                        / "merged-detections"
                        / f"{pass_name}-{view.view_id}.jsonl"
                    )
                    merge_reports.append(
                        {
                            "view_id": view.view_id,
                            **_merge_frame_evidence_ledgers(
                                existing_path,
                                result[view.view_id],
                                merged_path,
                                track_id_namespace=pass_index + 1,
                            ),
                        }
                    )
                    detection_paths[view.view_id] = merged_path
                actual_windows[view.view_id] = self._merge_time_windows(
                    [
                        *actual_windows.get(view.view_id, []),
                        *pass_windows[view.view_id],
                    ]
                )
            scanned_views = [
                view for view in fine_views if view.view_id in scanned
            ]
            current_candidates = generate_current_candidates(scanned_views)
            if fine_frame_index is None:
                current_events, _ = audit_candidates(
                    current_candidates,
                    transforms,
                    self.config,
                )
            else:
                current_events, _ = audit_candidates(
                    current_candidates,
                    transforms,
                    self.config,
                    fine_frame_index,
                )
            liquid_context_rejections = refine_current_liquid_context(
                current_events
            )
            target_status = self._progressive_target_status(
                boundary_candidates, current_events
            )
            pass_reports.append(
                {
                    "pass_index": pass_index,
                    "pass_kind": pass_kind,
                    "window_strategy": window_strategy,
                    "view_ids": [view.view_id for view in pass_views],
                    "target_candidate_ids": [
                        candidate.candidate_id for candidate in pass_candidates
                    ],
                    "decode_backends": {
                        view.view_id: full_decode_backends[view.view_id]
                        for view in pass_views
                    },
                    "coverage": self._window_coverage(pass_windows, infos),
                    "candidate_count_after_pass": len(current_candidates),
                    "liquid_context_rejection_count": len(
                        liquid_context_rejections
                    ),
                    "liquid_context_rejected_event_ids": [
                        item["event_id"] for item in liquid_context_rejections
                    ],
                    "target_status_after_pass": target_status,
                    "detection_ledger_merges": merge_reports,
                    "fine_index_ingest": index_ingest_report,
                }
            )
            return target_status

        target_status = execute_pass(
            0, "primary", initial_views, boundary_candidates
        )
        unresolved_ids = {
            item["candidate_id"]
            for item in target_status
            if item["status"] == "needs_third_person_supplement"
        }
        if scout_enabled and unresolved_ids:
            scout_view_ids = [view.view_id for view in supplemental_views]
            scout_anchors, scout_anchor_diagnostics = (
                self._progressive_scout_anchor_representatives(
                    target_status,
                    unresolved_ids,
                    dedup_tolerance_seconds=float(
                        perf.get(
                            "fine_scout_peak_dedup_tolerance_seconds", 0.25
                        )
                    ),
                    cluster_gap_seconds=float(
                        perf.get("fine_scout_peak_cluster_gap_seconds", 60.0)
                    ),
                    representatives_per_cluster=int(
                        perf.get("fine_scout_representatives_per_cluster", 1)
                    ),
                )
            )
            scout_status = [
                {
                    "candidate_id": "SCOUT-REPRESENTATIVES",
                    "first_person_anchor_windows": scout_anchors,
                }
            ]
            scout_windows = self._progressive_anchor_windows(
                scout_status,
                scout_view_ids,
                infos,
                transforms,
                0.0,
                None,
                peak_radius_seconds=float(
                    perf.get("fine_scout_anchor_radius_seconds", 3.0)
                ),
            )
            scout_manifest = manifest.model_copy(update={"views": supplemental_views})
            scout_paths = self._scan_all_views_concurrently(
                scout_manifest,
                infos,
                transforms,
                work_dir / "scout",
                windows=scout_windows,
                sample_fps=float(perf.get("fine_scout_fps", 1.0)),
                image_size=int(perf.get("fine_scout_image_size", 416)),
                keyframes_only=bool(perf.get("fine_scout_keyframes_only", False)),
                phase="fine_scout",
                decode_backends={
                    view.view_id: full_decode_backends[view.view_id]
                    for view in supplemental_views
                },
            )
            scout_config = deepcopy(self.config)
            _, scout_view_report = select_fine_scan_views(
                supplemental_views,
                scout_paths,
                boundary_candidates,
                scout_config,
            )
            ranked_initial, ranked_tail = self._progressive_fine_view_order(
                initial_views + supplemental_views,
                scout_view_report,
                1,
                list(perf.get("fine_preferred_third_person_views") or []),
            )
            supplemental_views = [
                view
                for view in ranked_initial + ranked_tail
                if view.role == ViewRole.THIRD_PERSON
            ]
            scout_rank = {
                view.view_id: index
                for index, view in enumerate(supplemental_views, 1)
            }
            for view in supplemental_views:
                scout_item = scout_view_report.get(view.view_id, {})
                fine_view_report.setdefault(view.view_id, {}).update(
                    {
                        "progressive_initial": False,
                        "progressive_supplemental_rank": scout_rank[view.view_id],
                        "scout_rank": scout_rank[view.view_id],
                        "scout_anchor_frames": int(
                            scout_item.get("anchor_frames", 0)
                        ),
                        "scout_active_anchor_frames": int(
                            scout_item.get("active_anchor_frames", 0)
                        ),
                        "scout_anchor_classes": list(
                            scout_item.get("anchor_classes") or []
                        ),
                        "scout_motion_threshold": scout_item.get(
                            "motion_threshold"
                        ),
                    }
                )
            scout_coverage = self._window_coverage(scout_windows, infos)
            scout_seconds = sum(
                float(item["selected_seconds"])
                for item in scout_coverage.values()
            )
            scout_fps = float(perf.get("fine_scout_fps", 1.0))
            scout_summary = {
                "enabled": True,
                "view_ids": scout_view_ids,
                "sample_fps": scout_fps,
                "image_size": int(perf.get("fine_scout_image_size", 416)),
                "keyframes_only": bool(
                    perf.get("fine_scout_keyframes_only", False)
                ),
                "anchor_radius_seconds": float(
                    perf.get("fine_scout_anchor_radius_seconds", 3.0)
                ),
                "anchor_selection": scout_anchor_diagnostics,
                "window_strategy": "aligned_first_person_peak_windows",
                "pre_merge_window_count_per_view": len(scout_anchors),
                "post_merge_window_count_by_view": {
                    view_id: len(windows)
                    for view_id, windows in scout_windows.items()
                },
                "total_post_merge_window_count": sum(
                    len(windows) for windows in scout_windows.values()
                ),
                "coverage": scout_coverage,
                "selected_seconds": round(scout_seconds, 3),
                "estimated_frames": int(round(scout_seconds * scout_fps)),
                "ranking": [
                    {
                        "view_id": view.view_id,
                        "rank": scout_rank[view.view_id],
                        "anchor_frames": fine_view_report[view.view_id][
                            "scout_anchor_frames"
                        ],
                        "active_anchor_frames": fine_view_report[view.view_id][
                            "scout_active_anchor_frames"
                        ],
                        "anchor_classes": fine_view_report[view.view_id][
                            "scout_anchor_classes"
                        ],
                    }
                    for view in supplemental_views
                ],
            }
        supplemental_batch_size = max(
            1, int(perf.get("fine_supplemental_view_batch_size", 1))
        )
        supplemental_waves = [
            supplemental_views[index : index + supplemental_batch_size]
            for index in range(0, len(supplemental_views), supplemental_batch_size)
        ]
        for pass_index, pass_views in enumerate(supplemental_waves, 1):
            if not unresolved_ids:
                break
            pass_candidates = [
                candidate
                for candidate in boundary_candidates
                if candidate.candidate_id in unresolved_ids
            ]
            target_status = execute_pass(
                pass_index,
                "supplemental",
                pass_views,
                pass_candidates,
                target_status,
            )
            unresolved_ids = {
                item["candidate_id"]
                for item in target_status
                if item["status"] == "needs_third_person_supplement"
            }

        formal_state_cache: tuple[
            list[ActionCandidate],
            list[EvidenceEvent],
            list[ExperimentSegment],
            list[ExperimentGroup],
            list[EvidenceEvent],
        ] | None = None

        def formal_state() -> tuple[
            list[ActionCandidate],
            list[EvidenceEvent],
            list[ExperimentSegment],
            list[ExperimentGroup],
            list[EvidenceEvent],
        ]:
            nonlocal formal_state_cache
            if formal_state_cache is not None:
                return formal_state_cache
            state_views = [view for view in fine_views if view.view_id in scanned]
            state_candidates = generate_current_candidates(state_views)
            if fine_frame_index is None:
                state_events, _ = audit_candidates(
                    state_candidates,
                    transforms,
                    self.config,
                )
            else:
                state_events, _ = audit_candidates(
                    state_candidates,
                    transforms,
                    self.config,
                    fine_frame_index,
                )
            refine_current_liquid_context(state_events)
            attach_action_observability(state_events)
            raw_state_segments = build_experiment_segments(
                state_events,
                manifest.views,
                self.config,
                coarse_windows=boundary_candidates,
            )
            normalized_state_segments = normalize_experiment_segments(
                raw_state_segments, state_events, manifest.views, self.config
            )
            state_segments, _ = prepare_formal_experiment_segments(
                normalized_state_segments,
                state_events,
                manifest.views,
                boundary_candidates,
                self.config,
            )
            state_groups = build_experiment_groups(
                state_segments,
                state_events,
                manifest.views,
                self.config,
                coarse_windows=boundary_candidates,
            )
            state_key_events = select_key_events(
                state_groups, state_segments, state_events, self.config
            )
            formal_state_cache = (
                state_candidates,
                state_events,
                state_segments,
                state_groups,
                state_key_events,
            )
            return formal_state_cache

        local_recall_rounds: list[dict[str, Any]] = []
        local_recall_enabled = bool(
            perf.get("fine_group_local_recall_enabled", False)
        )
        maximum_recall_rounds = max(
            0, int(perf.get("fine_group_recall_max_rounds", 5))
        )
        local_recall_plan: dict[str, Any] = {
            "schema_version": "visioncortex-group-local-recall-plan/2",
            "groups": [],
            "selected_plans": [],
            "complete": True,
        }
        if local_recall_enabled:
            for recall_round in range(1, maximum_recall_rounds + 1):
                (
                    recall_candidates,
                    recall_events,
                    recall_segments,
                    recall_groups,
                    recall_key_events,
                ) = formal_state()
                local_recall_plan = self._group_local_recall_plan(
                    recall_groups,
                    recall_segments,
                    recall_events,
                    recall_candidates,
                    fine_views,
                    fine_view_report,
                    actual_windows,
                    infos,
                    transforms,
                    boundary_candidates=boundary_candidates,
                )
                selected_plans = list(
                    local_recall_plan.get("selected_plans") or []
                )
                if not selected_plans:
                    break
                selected_view_ids: list[str] = []
                for item in selected_plans:
                    view_id = str(item["view_id"])
                    if view_id not in selected_view_ids:
                        selected_view_ids.append(view_id)
                    if len(selected_view_ids) >= supplemental_batch_size:
                        break
                windows_by_view: dict[str, list[tuple[float, float]]] = {}
                group_ids_by_view: dict[str, list[str]] = {}
                for item in selected_plans:
                    view_id = str(item["view_id"])
                    if view_id not in selected_view_ids:
                        continue
                    windows_by_view.setdefault(view_id, []).extend(
                        (float(start), float(end))
                        for start, end in item["windows"]
                    )
                    group_ids_by_view.setdefault(view_id, []).append(
                        str(item["group_id"])
                    )
                for view_id, view_windows in list(windows_by_view.items()):
                    windows_by_view[view_id] = self._merge_time_windows(
                        view_windows
                    )
                recall_views = [
                    view for view in fine_views if view.view_id in windows_by_view
                ]
                before_count = len(recall_key_events)
                recall_pass_index = len(pass_reports)
                execute_pass(
                    recall_pass_index,
                    f"group-recall-{recall_round:02d}",
                    recall_views,
                    [],
                    explicit_windows=windows_by_view,
                )
                formal_state_cache = None
                (
                    _after_candidates,
                    _after_events,
                    _after_segments,
                    _after_groups,
                    after_key_events,
                ) = formal_state()
                local_recall_rounds.append(
                    {
                        "round": recall_round,
                        "view_ids": [view.view_id for view in recall_views],
                        "group_ids_by_view": group_ids_by_view,
                        "windows_by_view": windows_by_view,
                        "selected_seconds": round(
                            sum(
                                end - start
                                for windows in windows_by_view.values()
                                for start, end in windows
                            )
                            / 1000.0,
                            6,
                        ),
                        "selected_key_events_before": before_count,
                        "selected_key_events_after": len(after_key_events),
                        "selected_key_event_gain": len(after_key_events)
                        - before_count,
                        "plan": local_recall_plan,
                    }
                )

            (
                final_recall_candidates,
                final_recall_events,
                final_recall_segments,
                final_recall_groups,
                final_recall_key_events,
            ) = formal_state()
            local_recall_plan = self._group_local_recall_plan(
                final_recall_groups,
                final_recall_segments,
                final_recall_events,
                final_recall_candidates,
                fine_views,
                fine_view_report,
                actual_windows,
                infos,
                transforms,
                boundary_candidates=boundary_candidates,
            )
            local_recall_summary = {
                "enabled": True,
                "rounds": local_recall_rounds,
                "round_count": len(local_recall_rounds),
                "selected_key_event_count": len(final_recall_key_events),
                "completeness_gate": local_recall_plan,
                "stopping_reason": (
                    "all_eligible_tp_views_exhausted"
                    if local_recall_plan.get("complete")
                    and any(
                        item.get("status")
                        == "unresolved_all_tp_windows_exhausted"
                        for item in local_recall_plan.get("groups", [])
                    )
                    else "no_positive_recall_prior_without_quality_fallback"
                    if local_recall_plan.get("complete")
                    and any(
                        item.get("status")
                        == "complete_no_positive_recall_prior"
                        for item in local_recall_plan.get("groups", [])
                    )
                    else "formal_groups_complete"
                    if local_recall_plan.get("complete")
                    else "maximum_group_recall_rounds_exhausted"
                ),
            }
        else:
            local_recall_summary = {
                "enabled": False,
                "rounds": [],
                "round_count": 0,
                "stopping_reason": "disabled",
            }

        scanned_views = [view for view in fine_views if view.view_id in scanned]
        final_candidates = generate_current_candidates(scanned_views)
        quarantined_ids: set[str] = set()
        if local_recall_rounds or unresolved_ids:
            (
                final_candidates,
                final_target_events,
                _final_segments,
                final_groups,
                _final_key_events,
            ) = formal_state()
            # The early progressive pass intentionally runs before the full
            # observability ledger exists.  The formal state above attaches
            # that ledger, so always recompute here—even when the group-level
            # completeness gate needed zero recall rounds.  Otherwise an
            # indirect state cue can remain a stale hard anchor in the final
            # quality gate after observability has correctly demoted it.
            target_status = self._progressive_target_status(
                boundary_candidates, final_target_events
            )
            unresolved_ids, quarantined_ids = (
                self._quarantine_nonformal_progressive_gaps(
                    target_status,
                    final_groups,
                )
            )
        all_coverage = self._window_coverage(fine_windows, infos)
        actual_coverage = self._window_coverage(actual_windows, infos)
        fine_index_report: dict[str, Any] | None = None
        if fine_frame_index is not None:
            fine_index_report = fine_frame_coverage_report(
                fine_frame_index,
                scanned_views,
                infos,
                actual_windows,
                sample_fps=float(perf["detection_fps"]),
                minimum_coverage_ratio=float(
                    perf.get("fine_minimum_coverage_ratio", 0.98)
                ),
                maximum_gap_periods=float(
                    perf.get("fine_maximum_gap_periods", 4.0)
                ),
                alignment_scales={
                    view.view_id: transforms[view.view_id].scale
                    for view in scanned_views
                },
            )
            detection_paths = fine_frame_index.materialize_ledgers(
                work_dir / "indexed-detections",
                scanned_views,
            )
        all_seconds = sum(float(item["selected_seconds"]) for item in all_coverage.values())
        actual_seconds = sum(
            float(item["selected_seconds"]) for item in actual_coverage.values()
        )
        sample_fps = float(perf["detection_fps"])
        scout_frames = int(scout_summary.get("estimated_frames", 0))
        full_fine_frames = int(round(actual_seconds * sample_fps))
        all_view_frames = int(round(all_seconds * sample_fps))
        # The post-recall unresolved/quarantine classification is authoritative.
        # A recall plan may have exhausted windows that are subsequently proven
        # to contain only rejected audit context; those are resolved, not failed.
        local_recall_was_enabled = bool(local_recall_summary.get("enabled"))
        group_recall_quality_complete = bool(
            not local_recall_was_enabled or not unresolved_ids
        )
        fine_coverage_quality_complete = bool(
            fine_index_report is None
            or fine_index_report.get("formal_evidence_ready", False)
        )
        progressive_quality_complete = bool(
            not unresolved_ids
            and group_recall_quality_complete
            and fine_coverage_quality_complete
        )
        if local_recall_was_enabled:
            local_recall_summary["post_quarantine_quality_complete"] = (
                group_recall_quality_complete
            )
            local_recall_summary["post_quarantine_unresolved_candidate_ids"] = sorted(
                unresolved_ids
            )
        report = {
            "schema_version": "visioncortex-progressive-fine-scan/3",
            "enabled": True,
            "selection_rule": (
                "scan the first-person boundary sensor; rank every third-person view "
                "with a low-FPS aligned scout; then exhaust required third-person views "
                "at full FPS only inside narrow first-person anchor windows"
                if scout_enabled
                else "scan first-person plus the configured initial third-person wave; "
                "scan remaining third-person views in bounded shared-model waves only for recalled windows "
                "with a reliable first-person start anchor but no valid dual-role anchor"
            ),
            "eligible_view_ids": [view.view_id for view in fine_views],
            "initial_view_ids": [view.view_id for view in initial_views],
            "preferred_third_person_views": list(
                perf.get("fine_preferred_third_person_views") or []
            ),
            "supplemental_priority": [view.view_id for view in supplemental_views],
            "supplemental_view_batch_size": supplemental_batch_size,
            "supplemental_waves": [
                [view.view_id for view in wave] for wave in supplemental_waves
            ],
            "dynamic_cross_view_scout": scout_summary,
            "group_local_recall": local_recall_summary,
            "fine_frame_index": fine_index_report,
            "fine_index_ingest_passes": fine_index_ingest_reports,
            "scanned_view_ids": [view.view_id for view in scanned_views],
            "not_scanned_view_ids": [
                view.view_id for view in fine_views if view.view_id not in scanned
            ],
            "passes": pass_reports,
            "final_target_status": target_status,
            "unresolved_candidate_ids": sorted(unresolved_ids),
            "quarantined_candidate_ids": sorted(quarantined_ids),
            "quality_complete": progressive_quality_complete,
            "stopping_reason": (
                "all_demanded_windows_have_dual_role_anchor"
                if progressive_quality_complete
                else "fine_frame_coverage_incomplete"
                if not fine_coverage_quality_complete
                else "group_local_evidence_exhausted_with_unresolved_clusters"
                if not group_recall_quality_complete
                else "all_eligible_third_person_views_exhausted"
            ),
            "all_view_selected_seconds": round(all_seconds, 3),
            "actual_selected_seconds": round(actual_seconds, 3),
            "avoided_selected_seconds": round(max(0.0, all_seconds - actual_seconds), 3),
            "all_view_estimated_frames": all_view_frames,
            "actual_estimated_frames": full_fine_frames,
            "scout_estimated_frames": scout_frames,
            "total_actual_estimated_frames": full_fine_frames + scout_frames,
            "avoided_estimated_frames": max(
                0, all_view_frames - full_fine_frames - scout_frames
            ),
            "actual_coverage": actual_coverage,
        }
        return detection_paths, scanned_views, final_candidates, report

    @staticmethod
    def _window_coverage(
        windows: dict[str, list[tuple[float, float]]], infos
    ) -> dict[str, dict[str, float | int]]:
        report: dict[str, dict[str, float | int]] = {}
        for view_id, view_windows in windows.items():
            seconds = sum(max(0.0, end - start) for start, end in view_windows) / 1000.0
            duration_seconds = infos[view_id].duration_ms / 1000.0
            report[view_id] = {
                "window_count": len(view_windows),
                "selected_seconds": round(seconds, 3),
                "source_seconds": round(duration_seconds, 3),
                "coverage_ratio": (
                    round(seconds / duration_seconds, 6) if duration_seconds > 0 else 0.0
                ),
            }
        return report

    @staticmethod
    def _input_volume_report(manifest: RunManifest, infos) -> dict[str, Any]:
        listed_video_paths = [
            os.path.abspath(path)
            for view in manifest.views
            for path in view_source_files(view)
        ]
        listed_clock_paths = [
            os.path.abspath(path)
            for view in manifest.views
            for path in view_timestamp_files(view)
        ]
        views = []
        for view in manifest.views:
            info = infos[view.view_id]
            duration_seconds = info.duration_ms / 1000.0
            views.append(
                {
                    "view_id": view.view_id,
                    "role": view.role.value,
                    "segment_count": len(view.segments) if view.segments else 1,
                    "width": info.width,
                    "height": info.height,
                    "fps": info.fps,
                    "duration_seconds": round(duration_seconds, 3),
                    "video_bytes": info.size_bytes,
                    "video_gib": round(info.size_bytes / 1024**3, 3),
                    "estimated_video_mbps": (
                        round(info.size_bytes * 8.0 / duration_seconds / 1_000_000.0, 3)
                        if duration_seconds > 0
                        else 0.0
                    ),
                }
            )
        total_video_bytes = sum(int(item["video_bytes"]) for item in views)
        return {
            "schema_version": "visioncortex-input-volume/1",
            "listed_video_path_count": len(listed_video_paths),
            "unique_video_path_count": len(set(listed_video_paths)),
            "duplicate_video_path_count": len(listed_video_paths) - len(set(listed_video_paths)),
            "listed_clock_path_count": len(listed_clock_paths),
            "unique_clock_path_count": len(set(listed_clock_paths)),
            "total_video_bytes": total_video_bytes,
            "total_video_gib": round(total_video_bytes / 1024**3, 3),
            "total_video_gb_decimal": round(total_video_bytes / 1_000_000_000.0, 3),
            "views": views,
        }

    @staticmethod
    def _archive_scan_runtime(
        layout: ArchiveLayout,
        work_dir: Path,
        phase: str,
        progressive_report: dict[str, Any] | None = None,
    ) -> None:
        role_reports = []
        role_values = {role.value for role in ViewRole}

        def scanner_role(value: str) -> str | None:
            return next(
                (
                    role
                    for role in role_values
                    if value == role or value.startswith(f"{role}_")
                ),
                None,
            )

        for path in sorted(work_dir.rglob(f"runtime_{phase}_*.json")):
            runtime_role = path.stem.removeprefix(f"runtime_{phase}_")
            if scanner_role(runtime_role) is None:
                continue
            report = json.loads(path.read_text(encoding="utf-8"))
            report["scan_pass"] = str(path.parent.relative_to(work_dir)).replace("\\", "/")
            role_reports.append(report)
        source_activity = []
        for path in sorted(work_dir.rglob(f"source_activity_{phase}_*.jsonl")):
            activity_role = path.stem.removeprefix(f"source_activity_{phase}_")
            if scanner_role(activity_role) is None:
                continue
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    if item.get("phase") not in {None, phase}:
                        continue
                    item["scan_pass"] = str(path.parent.relative_to(work_dir)).replace(
                        "\\", "/"
                    )
                    source_activity.append(item)
        computed_units = sum(
            item.get("event") == "source_unit_completed" for item in source_activity
        )
        reused_units = sum(
            item.get("event") == "source_unit_reused" for item in source_activity
        )
        started_units = sum(
            item.get("event") == "source_unit_started" for item in source_activity
        )
        decoder_receipts = [
            item["decoder_receipt"]
            for item in source_activity
            if item.get("event") == "source_unit_completed"
            and isinstance(item.get("decoder_receipt"), dict)
        ]
        frame_accounting = {
            "session_count": len(decoder_receipts),
            "exact_session_count": sum(
                int(receipt.get("frame_accounting_mismatch") or 0) == 0
                and not bool(receipt.get("frame_accounting_reconciled"))
                for receipt in decoder_receipts
            ),
            "reconciled_terminal_eof_session_count": sum(
                bool(receipt.get("frame_accounting_reconciled"))
                and int(receipt.get("terminal_eof_shortfall_frames") or 0) > 0
                for receipt in decoder_receipts
            ),
            "unreconciled_mismatch_session_count": sum(
                int(receipt.get("frame_accounting_mismatch") or 0) != 0
                and not bool(receipt.get("frame_accounting_reconciled"))
                for receipt in decoder_receipts
            ),
        }
        scheduler_reports = []
        for path in sorted(work_dir.rglob(f"scheduler_{phase}.json")):
            report = json.loads(path.read_text(encoding="utf-8"))
            if report.get("phase") not in {None, phase}:
                continue
            report["scan_pass"] = str(path.parent.relative_to(work_dir)).replace("\\", "/")
            scheduler_reports.append(report)
        if progressive_report is not None:
            scheduler = {
                "schema_version": "visioncortex-role-scheduler/2",
                "phase": phase,
                "mode": "progressive_cross_view",
                "passes": scheduler_reports,
            }
        elif len(scheduler_reports) == 1:
            scheduler = scheduler_reports[0]
        elif scheduler_reports:
            scheduler = {
                "schema_version": "visioncortex-role-scheduler/2",
                "phase": phase,
                "mode": "progressive_cross_view",
                "passes": scheduler_reports,
            }
        else:
            scheduler = None
        inference_calls = sum(int(item.get("inference_call_count", 0)) for item in role_reports)
        inference_frames = sum(int(item.get("inference_frame_count", 0)) for item in role_reports)
        queue_wait_seconds = sum(float(item.get("queue_wait_seconds", 0.0)) for item in role_reports)
        inference_seconds = sum(float(item.get("inference_seconds", 0.0)) for item in role_reports)
        postprocess_seconds = sum(
            float(item.get("tracking_and_ledger_seconds", 0.0)) for item in role_reports
        )
        role_seconds = [float(item.get("role_total_seconds", 0.0)) for item in role_reports]
        observed_seconds = sum(role_seconds)
        actual_batch_mean = inference_frames / inference_calls if inference_calls else 0.0
        effective_batch_slots = sum(
            int(item.get("inference_call_count", 0))
            * int(item.get("final_effective_batch_size", 0))
            for item in role_reports
        )
        batch_fill_ratio = inference_frames / effective_batch_slots if effective_batch_slots else 0.0
        effective_batch_capacity = (
            effective_batch_slots / inference_calls if inference_calls else 0.0
        )
        queue_wait_ratio = queue_wait_seconds / observed_seconds if observed_seconds else 0.0
        inference_ratio = inference_seconds / observed_seconds if observed_seconds else 0.0
        postprocess_ratio = postprocess_seconds / observed_seconds if observed_seconds else 0.0
        inference_frames_per_second = (
            inference_frames / inference_seconds if inference_seconds else 0.0
        )
        inference_milliseconds_per_call = (
            inference_seconds * 1000.0 / inference_calls if inference_calls else 0.0
        )
        if batch_fill_ratio < 0.60 and queue_wait_ratio >= 0.25:
            bottleneck = "decode_or_source_starved"
            next_action = "increase ordered decode supply before adding YOLO contexts"
        elif inference_ratio >= 0.65:
            bottleneck = "gpu_inference_bound"
            next_action = "reduce selected frames/windows or benchmark a faster engine"
        elif postprocess_ratio >= 0.40:
            bottleneck = "cpu_postprocess_bound"
            next_action = "parallelize tracking and ledger serialization"
        else:
            bottleneck = "mixed_or_balanced"
            next_action = "use per-role telemetry before changing concurrency"
        bottleneck_diagnosis = {
            "classification": bottleneck,
            "next_action": next_action,
            "observed_role_seconds": round(observed_seconds, 6),
            "queue_wait_seconds": round(queue_wait_seconds, 6),
            "inference_seconds": round(inference_seconds, 6),
            "tracking_and_ledger_seconds": round(postprocess_seconds, 6),
            "queue_wait_ratio": round(queue_wait_ratio, 6),
            "inference_ratio": round(inference_ratio, 6),
            "tracking_and_ledger_ratio": round(postprocess_ratio, 6),
            "inference_call_count": inference_calls,
            "inference_frame_count": inference_frames,
            "inference_frames_per_second": round(inference_frames_per_second, 3),
            "inference_milliseconds_per_call": round(
                inference_milliseconds_per_call, 3
            ),
            "actual_batch_size_mean": round(actual_batch_mean, 4),
            "effective_batch_capacity_mean": round(effective_batch_capacity, 4),
            "batch_fill_ratio": round(batch_fill_ratio, 6),
        }
        write_json(
            layout.json_config / f"scan_runtime_{phase}.json",
            {
                "schema_version": "visioncortex-scan-runtime/1",
                "phase": phase,
                "cold_start": reused_units == 0,
                "work_units": {
                    "started": started_units,
                    "computed": computed_units,
                    "reused": reused_units,
                },
                "role_reports": role_reports,
                "scheduler": scheduler,
                "progressive_cross_view": progressive_report,
                "frame_accounting": frame_accounting,
                "bottleneck_diagnosis": bottleneck_diagnosis,
                "source_activity": sorted(
                    source_activity, key=lambda item: float(item.get("timestamp", 0.0))
                ),
            },
        )

    def run(self, manifest: RunManifest) -> Path:
        self._run_started_perf = time.perf_counter()
        self._run_started_iso = datetime.now(timezone.utc).isoformat()
        self._active_stage = None
        self._stage_metrics = []
        self._startup_metrics = {}
        self._preprocessing_completed_seconds = None
        self._current_experiment_id = manifest.experiment_id
        self._acceptance_baseline_selection = {
            "configured": False,
            "applied": False,
            "reason": "not_evaluated",
            "current_experiment_id": manifest.experiment_id,
        }
        self._input_view_count = len(manifest.views)
        self._input_mode = (
            "segmented_virtual_timeline"
            if all(view.segments for view in manifest.views)
            else "continuous_file"
        )
        lanes = list(self.config["performance"].get("coarse_decode_lanes") or [])
        if not lanes:
            lanes = [
                "cuda" if self.config["performance"].get("ffmpeg_hwaccel") else "cpu"
            ] * len(manifest.views)
        if len(lanes) < len(manifest.views):
            lanes.extend([lanes[-1]] * (len(manifest.views) - len(lanes)))
        self._view_runtime = {
            view.view_id: {
                "role": view.role.value,
                "segment_count": len(view.segments) if view.segments else 1,
                "decode_backend": lanes[index],
                "state": "registered",
            }
            for index, view in enumerate(manifest.views)
        }
        storage = self.config.get("storage", {})
        if storage.get("run_output_mode") == "nas_direct":
            active_archive = storage.get("active_archive_path")
            if not active_archive:
                raise ValueError("nas_direct output requires storage.active_archive_path")
            # Keep a mapped-drive path mapped. Path.resolve() expands Y: into a
            # longer UNC path on Windows and can push otherwise valid artifact
            # names beyond MAX_PATH when LongPathsEnabled is disabled.
            layout = ArchiveLayout(Path(os.path.abspath(str(active_archive))))
        else:
            output_root = Path(self.config["project"]["output_root"]).resolve()
            layout = ArchiveLayout(output_root / manifest.experiment_id)
        cache_identity_started = time.perf_counter()
        cache_identity = build_cache_identity(self.config, manifest)
        self._startup_metrics["cache_identity_seconds"] = round(
            time.perf_counter() - cache_identity_started,
            6,
        )
        self._startup_metrics["cache_identity_source_snapshot"] = cache_identity.get(
            "source_snapshot_report", {}
        )
        layout.work = (
            Path(self.config["storage"]["local_cache_root"]).resolve()
            / manifest.experiment_id
            / cache_identity["cache_key"]
        )
        layout.create()
        self._active_layout = layout
        write_json(layout.json_config / "cache_identity.json", cache_identity)
        if storage.get("sync_to_nas") and storage.get("run_output_mode") != "nas_direct":
            nas_root = initialize_nas_archive(self.config, manifest.experiment_id)
            self._publisher = IncrementalArchivePublisher(layout.root, nas_root)
        else:
            self._publisher = None
        lock_path = layout.root / "run.lock"
        self._acquire_lock(lock_path)
        self._resource_monitor = ResourceMonitor(
            layout.json_config / "resource_telemetry.json",
            float(self.config.get("resource_limits", {}).get("telemetry_interval_seconds", 1.0)),
            (
                self._publisher.nas_root
                / "JSON-Config-Files"
                / "resource_telemetry_live.json"
                if self._publisher is not None
                else None
            ),
        )
        self._resource_monitor.start()
        try:
            self._status(layout, "preflight", 0.02, "检查输入、模型、视频与磁盘")
            preflight_breakdown: dict[str, Any] = {}
            preflight_step_started = time.perf_counter()
            source_paths = [
                path
                for view in manifest.views
                for path in view_source_files(view) + view_timestamp_files(view) + speech.audio_files(view)
            ]
            source_snapshots, source_validation = snapshot_source_paths(
                source_paths,
                workers=int(self.config["performance"].get("source_stat_workers", 24)),
                max_age_seconds=float(
                    self.config["performance"].get(
                        "source_stat_cache_ttl_seconds", 120.0
                    )
                ),
            )
            missing_sources = [
                str(path)
                for path, snapshot in source_snapshots.items()
                if not snapshot["is_file"]
            ]
            source_validation["cache_diagnostics"] = source_cache_diagnostics()
            write_json(
                layout.json_config / "source_validation.json",
                source_validation,
            )
            if missing_sources:
                raise FileNotFoundError(f"输入源文件不存在: {missing_sources[:4]}")
            preflight_breakdown["source_validation_seconds"] = round(
                time.perf_counter() - preflight_step_started,
                6,
            )
            preflight_breakdown["source_validation"] = source_validation
            from .capture_quality import inspect as inspect_capture
            capture = inspect_capture(manifest, self.config)
            write_json(layout.json_config / "capture_quality.json", capture)
            preflight_step_started = time.perf_counter()
            model_report = validate_models(self.config)
            self._model_certification_audit = (
                audit_production_model_certification(self.config)
            )
            write_json(
                layout.json_config / "model_certification_audit.json",
                self._model_certification_audit,
            )
            model_report["video_encoder"] = video_encoder_preflight(
                str(self.config["performance"].get("ffmpeg_video_encoder", "h264_nvenc"))
            )
            preflight_breakdown["model_validation_seconds"] = round(
                time.perf_counter() - preflight_step_started,
                6,
            )
            write_json(layout.json_config / "model_runtime_preflight.json", model_report)
            preflight_step_started = time.perf_counter()
            infos = probe_views(
                manifest.views,
                workers=int(self.config["performance"].get("preflight_probe_workers", 12)),
                prefer_clock_metadata=bool(
                    self.config["performance"].get(
                        "preflight_prefer_clock_metadata", True
                    )
                ),
            )
            preflight_breakdown["video_probe_seconds"] = round(
                time.perf_counter() - preflight_step_started,
                6,
            )
            write_json(
                layout.json_config / "input_volume_report.json",
                self._input_volume_report(manifest, infos),
            )
            preflight_step_started = time.perf_counter()
            disk_report = check_disk_capacity(layout.root, list(infos.values()))
            preflight_breakdown["disk_check_seconds"] = round(
                time.perf_counter() - preflight_step_started,
                6,
            )
            media_preflight_path = layout.json_config / "media_pipeline_preflight.json"
            if bool(
                self.config["performance"].get(
                    "media_pipeline_preflight_enabled", True
                )
            ):
                preflight_step_started = time.perf_counter()
                smoke_root = layout.work / "media-pipeline-preflight"
                try:
                    media_preflight = run_media_pipeline_preflight(
                        manifest,
                        infos,
                        layout.work,
                        str(model_report["video_encoder"]["selected_encoder"]),
                        float(
                            self.config["performance"].get(
                                "media_pipeline_preflight_seconds", 1.0
                            )
                        ),
                    )
                except Exception as exc:
                    write_json(
                        media_preflight_path,
                        {
                            "schema_version": "visioncortex-media-pipeline-preflight/1",
                            "status": "failed",
                            "error": f"{type(exc).__name__}: {exc}",
                            "temporary_artifacts_retained": smoke_root.exists(),
                            "temporary_artifact_path": str(smoke_root),
                            "elapsed_seconds": round(
                                time.perf_counter() - preflight_step_started, 6
                            ),
                        },
                    )
                    raise RuntimeError(
                        "media pipeline preflight failed before long-running scans"
                    ) from exc
                write_json(media_preflight_path, media_preflight)
                preflight_breakdown["media_pipeline_preflight_seconds"] = round(
                    time.perf_counter() - preflight_step_started,
                    6,
                )
            preflight_breakdown["source_cache_after_probe"] = source_cache_diagnostics()
            write_json(
                layout.json_config / "preflight_runtime.json",
                preflight_breakdown,
            )
            write_json(layout.json_config / "video_probe.json", {key: value.model_dump(mode="json") for key, value in infos.items()})
            self._complete_stage(
                layout,
                "preflight",
                [
                    layout.json_config / "source_validation.json",
                    layout.json_config / "model_runtime_preflight.json",
                    layout.json_config / "input_volume_report.json",
                    layout.json_config / "preflight_runtime.json",
                    layout.json_config / "video_probe.json",
                    *([media_preflight_path] if media_preflight_path.exists() else []),
                ],
            )

            self._status(layout, "alignment", 0.08, "最近邻时间戳拟合与视觉锚点校准")
            alignment_runtime: dict[str, Any] = {}
            alignment_step_started = time.perf_counter()
            transforms, _ = build_alignments(manifest.views, infos, self.config)
            alignment_runtime["fit_seconds"] = round(
                time.perf_counter() - alignment_step_started,
                6,
            )
            alignment_runtime["view_runtime"] = {
                view_id: transform.runtime
                for view_id, transform in transforms.items()
            }
            write_json(
                layout.json_config / "time_alignment.json",
                [transform.model_dump(mode="json") for transform in transforms.values()],
            )
            alignment_gate = alignment_quality_report(
                manifest.views,
                infos,
                transforms,
                self.config,
            )
            write_json(
                layout.json_config / "alignment_quality_gate.json",
                alignment_gate,
            )
            alignment_step_started = time.perf_counter()
            write_aligned_csv(
                layout.json_config / "aligned_timestamps.csv",
                manifest.views,
                infos,
                transforms,
                float(self.config["alignment"]["aligned_timestamps_fps"]),
            )
            alignment_runtime["aligned_csv_seconds"] = round(
                time.perf_counter() - alignment_step_started,
                6,
            )
            alignment_runtime["source_cache_after_alignment"] = source_cache_diagnostics()
            write_json(
                layout.json_config / "alignment_runtime.json",
                alignment_runtime,
            )
            if (
                bool(self.config["alignment"].get("quality_gate_enabled", True))
                and not bool(alignment_gate["formal_evidence_ready"])
            ):
                raise RuntimeError(
                    "alignment quality gate failed before GPU scans: "
                    + "; ".join(str(item) for item in alignment_gate["errors"])
                )
            self._complete_stage(
                layout,
                "alignment",
                [
                    layout.json_config / "time_alignment.json",
                    layout.json_config / "aligned_timestamps.csv",
                    layout.json_config / "alignment_runtime.json",
                    layout.json_config / "alignment_quality_gate.json",
                ],
            )

            self._status(layout, "speech", 0.09, "整理实验录音与转写")
            speech.run_stage(
                self.config, manifest, layout, infos, transforms,
                progress=lambda message: self._status(layout, "speech", 0.09, message),
            )
            if speech.enabled(self.config) and (layout.json_config / "Input-Manifests/input_seal.json").is_file():
                from .speech_timeline import build as build_speech_timeline
                build_speech_timeline(layout.root)
            self._complete_stage(layout, "speech", [
                layout.json_config / "speech.json",
                *([layout.json_config / "speech_timeline.json"] if (layout.json_config / "speech_timeline.json").is_file() else []),
                *([layout.key_materials / "Experiment-Audio"] if (layout.key_materials / "Experiment-Audio").is_dir() else []),
            ])

            motion_probe_views = self._motion_probe_views(manifest)
            preselected_coarse_views = self._coarse_scan_views(manifest)
            coarse_full_timeline = bool(
                self.config["performance"].get(
                    "coarse_full_timeline_scan", False
                )
            )
            shared_coarse_motion_scan = bool(
                self.config["performance"].get(
                    "coarse_shared_motion_probe_enabled", False
                )
                and coarse_full_timeline
                and {
                    view.view_id for view in motion_probe_views
                }
                == {view.view_id for view in preselected_coarse_views}
                and float(
                    self.config["performance"]["coarse_detection_fps"]
                )
                >= float(self.config["performance"]["motion_probe_fps"])
                and int(self.config["performance"]["coarse_image_size"])
                >= int(
                    self.config["performance"].get(
                        "motion_probe_max_width", 96
                    )
                )
                and not bool(
                    self.config["performance"].get(
                        "coarse_keyframes_only", False
                    )
                )
            )
            coarse_frame_index: CoarseFrameIndex | None = None
            coarse_index_report: dict[str, Any] | None = None

            def ensure_coarse_frame_index(
                paths: dict[str, Path], views: Sequence[ViewInput]
            ) -> CoarseFrameIndex | None:
                nonlocal coarse_frame_index, coarse_index_report
                if not self.config["performance"].get(
                    "coarse_frame_index_enabled", False
                ):
                    return None
                if coarse_frame_index is not None:
                    return coarse_frame_index
                coarse_frame_index, coarse_index_report = build_coarse_frame_index(
                    self._frame_index_path(layout.work, "coarse-frame-index.sqlite3"),
                    views,
                    infos,
                    paths,
                    sample_fps=float(
                        self.config["performance"]["coarse_detection_fps"]
                    ),
                    minimum_coverage_ratio=float(
                        self.config["performance"].get(
                            "coarse_minimum_coverage_ratio", 0.98
                        )
                    ),
                    maximum_gap_periods=float(
                        self.config["performance"].get(
                            "coarse_maximum_gap_periods", 4.0
                        )
                    ),
                )
                manifest_report = dict(coarse_index_report)
                manifest_report["index_path"] = str(coarse_frame_index.path)
                manifest_report["index_storage"] = "local_runtime_rebuildable"
                write_json(
                    layout.json_config / "coarse_frame_index_manifest.json",
                    manifest_report,
                )
                if (
                    self.config["performance"].get(
                        "coarse_coverage_gate_enabled", False
                    )
                    and not coarse_index_report["formal_evidence_ready"]
                ):
                    failed = [
                        item["view_id"]
                        for item in coarse_index_report["views"]
                        if not item["formal_evidence_ready"]
                    ]
                    raise RuntimeError(
                        "粗扫覆盖门禁失败，以下视角存在采样缺口: "
                        + ", ".join(failed)
                    )
                return coarse_frame_index
            probe_manifest = manifest.model_copy(update={"views": motion_probe_views})
            selected_probe_ids = {view.view_id for view in motion_probe_views}
            for view in manifest.views:
                self._view_runtime[view.view_id]["state"] = (
                    "motion_probe_running"
                    if view.view_id in selected_probe_ids
                    else "motion_probe_sentinel_not_selected"
                )
            self._status(
                layout,
                "motion_probe",
                0.12,
                (
                    "共享全时间轴粗扫解码并生成运动探针，不重复读取原视频"
                    if shared_coarse_motion_scan
                    else "Selecting and running the fastest real-source sparse motion probe"
                ),
            )
            sparse_strategy_path = layout.json_config / "motion_probe_sparse_strategy.json"
            configured_sparse_strategy = str(
                self.config["performance"].get(
                    "motion_probe_sparse_strategy", "indexed_seek"
                )
            )
            if configured_sparse_strategy == "auto":
                sentinel = motion_probe_views[0]
                sparse_report = benchmark_sparse_decode_strategy(
                    sentinel,
                    infos[sentinel.view_id],
                    sample_fps=float(self.config["performance"]["motion_probe_fps"]),
                    max_width=int(
                        self.config["performance"].get("motion_probe_max_width", 96)
                    ),
                    hwaccel=(
                        str(self.config["performance"].get("ffmpeg_hwaccel"))
                        if self.config["performance"].get("ffmpeg_hwaccel")
                        else None
                    ),
                    decoder_threads=int(
                        self.config["performance"].get("cpu_decode_threads", 0)
                    ),
                    cuda_scale=bool(
                        self.config["performance"].get("ffmpeg_cuda_scale", False)
                    ),
                    benchmark_seconds=float(
                        self.config["performance"].get(
                            "motion_probe_sparse_benchmark_seconds", 60.0
                        )
                    ),
                )
                selected_sparse_strategy = str(sparse_report["selected_strategy"])
            else:
                if configured_sparse_strategy not in {
                    "indexed_seek",
                    "sequential_keyframes",
                }:
                    raise ValueError(
                        "motion_probe_sparse_strategy must be auto, indexed_seek, "
                        "or sequential_keyframes"
                    )
                selected_sparse_strategy = configured_sparse_strategy
                sparse_report = {
                    "schema_version": "visioncortex-sparse-decode-benchmark/1",
                    "selected_strategy": selected_sparse_strategy,
                    "configured_strategy": configured_sparse_strategy,
                    "benchmark_skipped": True,
                }
            self.config["performance"][
                "motion_probe_sparse_strategy"
            ] = selected_sparse_strategy
            if configured_sparse_strategy == "auto":
                worker_key = (
                    "motion_probe_indexed_segment_workers"
                    if selected_sparse_strategy == "indexed_seek"
                    else "motion_probe_sequential_segment_workers"
                )
                selected_segment_workers = max(
                    1,
                    int(
                        self.config["performance"].get(
                            worker_key,
                            self.config["performance"].get(
                                "motion_probe_segment_workers", 1
                            ),
                        )
                    ),
                )
            else:
                selected_segment_workers = max(
                    1,
                    int(
                        self.config["performance"].get(
                            "motion_probe_segment_workers", 1
                        )
                    ),
                )
            self.config["performance"][
                "motion_probe_segment_workers"
            ] = selected_segment_workers
            sparse_report["selected_segment_workers"] = selected_segment_workers
            sparse_report["selection_reason"] = (
                "short real-source benchmark"
                if configured_sparse_strategy == "auto"
                else "explicit configuration"
            )
            sparse_report["shared_with_full_timeline_coarse_scan"] = (
                shared_coarse_motion_scan
            )
            write_json(sparse_strategy_path, sparse_report)
            selected_probe_ids = {view.view_id for view in motion_probe_views}
            for view in manifest.views:
                self._view_runtime[view.view_id]["state"] = (
                    "motion_probe_running"
                    if view.view_id in selected_probe_ids
                    else "motion_probe_sentinel_not_selected"
                )
            self._status(
                layout,
                "motion_probe",
                0.12,
                (
                    "全路全时间轴粗扫共享同一次解码，并同步生成0.5 FPS运动探针"
                    if shared_coarse_motion_scan
                    else "哨兵视角低分辨率运动探针；此阶段CUDA计算低占用属于预期"
                ),
            )
            if shared_coarse_motion_scan:
                shared_manifest = manifest.model_copy(
                    update={"views": preselected_coarse_views}
                )
                motion_paths = self._scan_all_views_concurrently(
                    shared_manifest,
                    infos,
                    transforms,
                    layout.work / "detections-coarse",
                    sample_fps=float(
                        self.config["performance"]["coarse_detection_fps"]
                    ),
                    image_size=int(
                        self.config["performance"]["coarse_image_size"]
                    ),
                    keyframes_only=bool(
                        self.config["performance"]["coarse_keyframes_only"]
                    ),
                    phase="coarse",
                )
                self._archive_scan_runtime(
                    layout, layout.work / "detections-coarse", "coarse"
                )
                shared_runtime = json.loads(
                    (layout.json_config / "scan_runtime_coarse.json").read_text(
                        encoding="utf-8"
                    )
                )
                write_json(
                    layout.json_config / "scan_runtime_motion_probe.json",
                    {
                        "schema_version": "visioncortex-shared-motion-probe-runtime/1",
                        "phase": "motion_probe",
                        "shared_source_phase": "coarse",
                        "additional_video_decode_bytes": 0,
                        "logical_sample_fps": float(
                            self.config["performance"]["motion_probe_fps"]
                        ),
                        "role_motion_scoring": [
                            {
                                "role": report.get("role"),
                                "motion_scoring": report.get("motion_scoring"),
                            }
                            for report in shared_runtime.get("role_reports", [])
                        ],
                    },
                )
                ensure_coarse_frame_index(
                    motion_paths, preselected_coarse_views
                )
            else:
                motion_paths = self._scan_all_views_concurrently(
                    probe_manifest,
                    infos,
                    transforms,
                    layout.work / "motion-probe",
                    sample_fps=float(
                        self.config["performance"]["motion_probe_fps"]
                    ),
                    image_size=int(
                        self.config["performance"].get(
                            "motion_probe_max_width", 96
                        )
                    ),
                    keyframes_only=bool(
                        self.config["performance"].get(
                            "motion_probe_keyframes_only", True
                        )
                    ),
                    phase="motion_probe",
                )
                self._archive_scan_runtime(
                    layout, layout.work / "motion-probe", "motion_probe"
                )
            probe_config = json.loads(json.dumps(self.config))
            probe_config["performance"]["motion_probe_require_objects"] = False
            probe_config["performance"][
                "motion_probe_use_embedded_coarse_scores"
            ] = shared_coarse_motion_scan
            probe_config["performance"]["motion_burst_percentile"] = float(
                self.config["performance"].get(
                    "motion_probe_percentile",
                    self.config["performance"]["motion_burst_percentile"],
                )
            )
            probe_config["performance"]["motion_burst_merge_gap_seconds"] = float(
                self.config["performance"].get(
                    "motion_probe_burst_merge_gap_seconds",
                    self.config["performance"]["motion_burst_merge_gap_seconds"],
                )
            )
            probe_config["performance"]["motion_burst_min_observations"] = int(
                self.config["performance"].get(
                    "motion_probe_min_observations",
                    self.config["performance"]["motion_burst_min_observations"],
                )
            )
            raw_motion_candidates = generate_motion_burst_candidates(
                motion_probe_views,
                motion_paths,
                probe_config,
                coarse_frame_index,
            )
            raw_motion_candidates = sorted(
                raw_motion_candidates,
                key=candidate_sort_key,
            )
            motion_candidates = fuse_motion_probe_candidates(
                raw_motion_candidates, probe_config
            )
            safety_fallback_used = False
            if not motion_candidates:
                safety_fallback_used = True
                motion_candidates = generate_motion_safety_candidates(
                    motion_probe_views,
                    motion_paths,
                    probe_config,
                    coarse_frame_index,
                )
            motion_candidates = sorted(motion_candidates, key=candidate_sort_key)
            if not motion_candidates:
                raise RuntimeError("输入视频没有产生任何可读运动帧，无法建立实验候选窗口")
            motion_windows = self._fine_windows(
                motion_candidates,
                infos,
                transforms,
                padding_seconds=float(
                    self.config["performance"].get(
                        "motion_probe_window_padding_seconds", 90.0
                    )
                ),
            )
            write_json(
                layout.json_config / "motion_probe_windows.json",
                {
                    "schema_version": "visioncortex-motion-probe/1",
                    "sentinel_views": [view.view_id for view in motion_probe_views],
                    "raw_candidate_count": len(raw_motion_candidates),
                    "fused_candidate_count": len(motion_candidates),
                    "safety_fallback_used": safety_fallback_used,
                    "coverage": self._window_coverage(motion_windows, infos),
                    "candidates": [
                        item.model_dump(mode="json") for item in motion_candidates
                    ],
                },
            )
            self._complete_stage(
                layout,
                "motion_probe",
                [
                    layout.json_config / "scan_runtime_motion_probe.json",
                    layout.json_config / "motion_probe_windows.json",
                    sparse_strategy_path,
                ],
            )

            reuse_motion_probe = (
                bool(
                    self.config["performance"].get(
                        "coarse_reuse_motion_probe", False
                    )
                )
                and bool(
                    self.config["performance"].get(
                        "motion_probe_run_yolo", False
                    )
                )
                and not coarse_full_timeline
            )
            coarse_scan_views = (
                preselected_coarse_views
                if shared_coarse_motion_scan
                else motion_probe_views
                if reuse_motion_probe
                else preselected_coarse_views
            )
            coarse_scan_ids = {view.view_id for view in coarse_scan_views}
            coarse_manifest = manifest.model_copy(update={"views": coarse_scan_views})
            for view in manifest.views:
                self._view_runtime[view.view_id]["state"] = (
                    "coarse_running"
                    if view.view_id in coarse_scan_ids
                    else "coarse_sentinel_not_selected"
                )
            self._status(
                layout,
                "candidate_coarse",
                0.28,
                (
                    "复用运动阶段已经完成的全路全时间轴YOLO粗扫"
                    if shared_coarse_motion_scan
                    else f"全时间轴多视角 {self.config['performance']['coarse_detection_fps']} FPS YOLO粗筛"
                    if coarse_full_timeline
                    else f"候选窗口内 {self.config['performance']['coarse_detection_fps']} FPS YOLO粗筛"
                ),
            )
            if shared_coarse_motion_scan:
                coarse_paths = motion_paths
                write_json(
                    layout.json_config / "coarse_reuse_motion_probe.json",
                    {
                        "schema_version": "visioncortex-coarse-reuse/1",
                        "reused": True,
                        "source_phase": "shared_coarse_motion_decode",
                        "source_view_ids": [
                            view.view_id for view in coarse_scan_views
                        ],
                        "sample_fps": float(
                            self.config["performance"]["coarse_detection_fps"]
                        ),
                        "logical_motion_probe_fps": float(
                            self.config["performance"]["motion_probe_fps"]
                        ),
                        "additional_video_decode_bytes": 0,
                    },
                )
            elif reuse_motion_probe:
                coarse_paths = motion_paths
                for view in manifest.views:
                    self._view_runtime[view.view_id]["state"] = (
                        "coarse_reused_motion_probe"
                        if view.view_id in coarse_scan_ids
                        else "coarse_sentinel_not_selected"
                    )
                write_json(
                    layout.json_config / "coarse_reuse_motion_probe.json",
                    {
                        "schema_version": "visioncortex-coarse-reuse/1",
                        "reused": True,
                        "source_phase": "motion_probe",
                        "source_view_ids": [view.view_id for view in coarse_scan_views],
                        "sample_fps": float(
                            self.config["performance"]["motion_probe_fps"]
                        ),
                        "additional_video_decode_bytes": 0,
                    },
                )
            else:
                coarse_paths = self._scan_all_views_concurrently(
                    coarse_manifest,
                    infos,
                    transforms,
                    layout.work / "detections-coarse",
                    windows=None if coarse_full_timeline else motion_windows,
                    sample_fps=float(self.config["performance"]["coarse_detection_fps"]),
                    image_size=int(self.config["performance"]["coarse_image_size"]),
                    keyframes_only=bool(self.config["performance"]["coarse_keyframes_only"]),
                    phase="coarse",
                )
                self._archive_scan_runtime(
                    layout, layout.work / "detections-coarse", "coarse"
                )
            ensure_coarse_frame_index(coarse_paths, coarse_scan_views)
            coarse_views = [
                view for view in coarse_scan_views if view.role == ViewRole.FIRST_PERSON
            ]
            coarse_config = json.loads(json.dumps(self.config))
            coarse_config["performance"][
                "candidate_alignment_uncertainty_ms_by_view"
            ] = {
                view_id: max(
                    float(transform.uncertainty_ms or 0.0),
                    float(transform.csv_rmse_ms or 0.0),
                )
                for view_id, transform in transforms.items()
            }
            coarse_config["segmentation"]["event_merge_gap_seconds"] = max(
                float(coarse_config["segmentation"]["event_merge_gap_seconds"]),
                1.5
                / float(
                    self.config["performance"][
                        "motion_probe_fps"
                        if reuse_motion_probe
                        else "coarse_detection_fps"
                    ]
                ),
            )
            coarse_config["segmentation"]["min_event_observations"] = 2
            coarse_candidates = generate_motion_burst_candidates(
                coarse_views,
                coarse_paths,
                coarse_config,
                coarse_frame_index,
            )
            comprehensive_coarse = bool(
                self.config["performance"].get(
                    "coarse_comprehensive_candidate_union", False
                )
            )
            if comprehensive_coarse:
                coarse_candidates.extend(
                    generate_coarse_activity_candidates(
                        coarse_scan_views,
                        coarse_paths,
                        coarse_config,
                        coarse_frame_index,
                    )
                )
            elif not coarse_candidates:
                fallback_views = [
                    view for view in coarse_scan_views if view.role == ViewRole.THIRD_PERSON
                ]
                fallback_paths = {view.view_id: coarse_paths[view.view_id] for view in fallback_views}
                coarse_candidates = generate_coarse_activity_candidates(
                    fallback_views,
                    fallback_paths,
                    coarse_config,
                    coarse_frame_index,
                )
            open_vocabulary_enabled = bool(
                self.config["performance"].get(
                    "coarse_open_vocabulary_recall_enabled", False
                )
            )
            if open_vocabulary_enabled:
                open_vocabulary_candidates, open_vocabulary_report = (
                    generate_open_vocabulary_coarse_candidates(
                        coarse_scan_views,
                        infos,
                        coarse_paths,
                        self.config,
                        coarse_frame_index,
                    )
                )
            else:
                open_vocabulary_candidates, open_vocabulary_report = [], None
            coverage_required = bool(
                self.config["performance"].get(
                    "coarse_coverage_gate_enabled", False
                )
            )
            coverage_ready = bool(
                not coverage_required
                or (
                    coarse_index_report is not None
                    and coarse_index_report.get("formal_evidence_ready")
                )
            )
            open_vocabulary_ready = bool(
                not open_vocabulary_enabled
                or (
                    open_vocabulary_report is not None
                    and open_vocabulary_report.get("formal_evidence_ready")
                )
            )
            candidate_gate_errors = []
            if not coverage_ready:
                candidate_gate_errors.append("coarse_frame_coverage_incomplete")
            if not open_vocabulary_ready:
                candidate_gate_errors.append(
                    "open_vocabulary_recall_unavailable_or_error_rate_exceeded"
                )
            candidate_discovery_gate = {
                "schema_version": "visioncortex-candidate-discovery-quality-gate/1",
                "formal_evidence_ready": not candidate_gate_errors,
                "errors": candidate_gate_errors,
                "coarse_coverage_required": coverage_required,
                "coarse_coverage_ready": coverage_ready,
                "open_vocabulary_enabled": open_vocabulary_enabled,
                "open_vocabulary_ready": open_vocabulary_ready,
                "open_vocabulary_status": (
                    open_vocabulary_report.get("status")
                    if open_vocabulary_report is not None
                    else "disabled"
                ),
            }
            write_json(
                layout.json_config / "candidate_discovery_quality_gate.json",
                candidate_discovery_gate,
            )
            if open_vocabulary_report is not None:
                write_json(
                    layout.json_config / "coarse_open_vocabulary_recall.json",
                    open_vocabulary_report,
                )
            if (
                self.config["performance"].get(
                    "candidate_discovery_quality_gate_enabled", False
                )
                and candidate_gate_errors
            ):
                raise RuntimeError(
                    "候选发现质量门禁失败: "
                    + "; ".join(candidate_gate_errors)
                )
            coarse_candidates.extend(open_vocabulary_candidates)
            coarse_candidates = sorted(coarse_candidates, key=candidate_sort_key)
            boundary_candidates, boundary_report = refine_motion_candidates_with_coarse(
                motion_candidates,
                coarse_candidates,
                coarse_config,
            )
            boundary_candidates = sorted(
                boundary_candidates,
                key=candidate_sort_key,
            )
            boundary_report["coarse_scan_view_ids"] = [
                view.view_id for view in coarse_scan_views
            ]
            if coarse_full_timeline or comprehensive_coarse or open_vocabulary_enabled:
                boundary_report["enhancements"] = {
                    "full_timeline_scan": coarse_full_timeline,
                    "comprehensive_candidate_union": comprehensive_coarse,
                    "open_vocabulary_candidate_ids": [
                        item.candidate_id for item in open_vocabulary_candidates
                    ],
                    "preserves_established_candidates": True,
                }
            write_json(
                layout.json_config / "coarse_boundary_refinement.json",
                boundary_report,
            )
            if open_vocabulary_report is not None:
                write_json(
                    layout.json_config / "coarse_open_vocabulary_recall.json",
                    open_vocabulary_report,
                )
            fine_views, fine_view_report = select_fine_scan_views(
                manifest.views,
                coarse_paths,
                boundary_candidates,
                self.config,
                coarse_frame_index,
            )
            short_timeline_ceiling_ms = float(
                self.config["performance"].get(
                    "auto_exhaustive_short_timeline_seconds", 0.0
                )
                or 0.0
            ) * 1000.0
            if (
                self.config["performance"].get(
                    "auto_exhaustive_short_timeline_enabled", False
                )
                and short_timeline_ceiling_ms > 0.0
                and infos
                and max(float(info.duration_ms) for info in infos.values())
                <= short_timeline_ceiling_ms
            ):
                self.config["performance"][
                    "exhaustive_full_timeline_scan"
                ] = True
                self.config["performance"][
                    "exhaustive_full_timeline_reason"
                ] = "automatic_short_probed_timeline_recall"
                self.config["performance"]["fine_progressive_cross_view"] = False
            progressive_enabled = bool(
                self.config["performance"].get("fine_progressive_cross_view", False)
            )
            if progressive_enabled:
                initial_fine_views, supplemental_fine_views = (
                    self._progressive_fine_view_order(
                        fine_views,
                        fine_view_report,
                        int(
                            self.config["performance"].get(
                                "fine_initial_third_person_views", 1
                            )
                        ),
                        list(
                            self.config["performance"].get(
                                "fine_preferred_third_person_views"
                            )
                            or []
                        ),
                    )
                )
                if bool(
                    self.config["performance"].get(
                        "fine_dynamic_cross_view_scout", False
                    )
                ):
                    initial_fine_views = [
                        view
                        for view in fine_views
                        if view.role == ViewRole.FIRST_PERSON
                    ]
                    supplemental_fine_views = [
                        view
                        for view in fine_views
                        if view.role == ViewRole.THIRD_PERSON
                    ]
                initial_fine_ids = {view.view_id for view in initial_fine_views}
                supplemental_rank = {
                    view.view_id: rank
                    for rank, view in enumerate(supplemental_fine_views, 1)
                }
                for view in fine_views:
                    item = fine_view_report[view.view_id]
                    item["progressive_initial"] = view.view_id in initial_fine_ids
                    item["progressive_supplemental_rank"] = supplemental_rank.get(
                        view.view_id
                    )
            write_json(layout.json_config / "fine_view_selection.json", fine_view_report)
            coarse_artifacts = [
                layout.json_config / "coarse_boundary_refinement.json",
                layout.json_config / "fine_view_selection.json",
            ]
            open_vocabulary_path = (
                layout.json_config / "coarse_open_vocabulary_recall.json"
            )
            if open_vocabulary_path.exists():
                coarse_artifacts.append(open_vocabulary_path)
            coarse_runtime = layout.json_config / "scan_runtime_coarse.json"
            if coarse_runtime.exists():
                coarse_artifacts.append(coarse_runtime)
            reuse_report = layout.json_config / "coarse_reuse_motion_probe.json"
            if reuse_report.exists():
                coarse_artifacts.append(reuse_report)
            coarse_index_manifest = (
                layout.json_config / "coarse_frame_index_manifest.json"
            )
            if coarse_index_manifest.exists():
                coarse_artifacts.append(coarse_index_manifest)
            candidate_quality_gate = (
                layout.json_config / "candidate_discovery_quality_gate.json"
            )
            if candidate_quality_gate.exists():
                coarse_artifacts.append(candidate_quality_gate)
            self._complete_stage(layout, "candidate_coarse", coarse_artifacts)
            fine_windows = self._fine_windows(
                boundary_candidates, infos, transforms
            )
            fine_windows = {
                view.view_id: fine_windows[view.view_id]
                for view in fine_views
            }
            fine_windows, fine_availability_report = (
                self._intersect_fine_windows_with_usable_alignment(
                    fine_windows,
                    infos,
                    transforms,
                    fine_views,
                )
            )
            for view in list(fine_views):
                if fine_windows.get(view.view_id):
                    continue
                fine_view_report[view.view_id]["selected"] = False
                fine_view_report[view.view_id]["reason"] = (
                    "alignment quality gate exposes no usable fine window"
                )
            fine_views = [
                view for view in fine_views if fine_windows.get(view.view_id)
            ]
            write_json(
                layout.json_config / "fine_view_selection.json",
                fine_view_report,
            )
            fine_coverage = self._window_coverage(fine_windows, infos)
            fine_sample_fps = float(self.config["performance"]["detection_fps"])
            exhaustive_negative_audit = bool(
                self.config["performance"].get(
                    "exhaustive_negative_audit_full_timeline", False
                )
            )
            exhaustive_full_timeline = bool(
                exhaustive_negative_audit
                or self.config["performance"].get(
                    "exhaustive_full_timeline_scan", False
                )
            )
            fine_window_report = {
                "schema_version": "visioncortex-fine-scan-windows/2",
                "strategy": (
                    "exhaustive_all_view_full_timeline_negative_audit"
                    if exhaustive_negative_audit
                    else "exhaustive_all_view_full_timeline"
                    if exhaustive_full_timeline
                    else "progressive_cross_view"
                    if progressive_enabled
                    else "all_selected_views"
                ),
                "eligible_view_ids": [view.view_id for view in fine_views],
                "selected_view_ids": [view.view_id for view in fine_views],
                "sample_fps": fine_sample_fps,
                "full_timeline_reason": self.config["performance"].get(
                    "exhaustive_full_timeline_reason"
                ),
                "image_size": int(self.config["performance"]["image_size"]),
                "padding_seconds": float(
                    self.config["performance"]["fine_window_padding_seconds"]
                ),
                "merge_gap_seconds": float(
                    self.config["performance"].get(
                        "fine_window_merge_gap_seconds", 0.0
                    )
                ),
                "coverage": fine_coverage,
                "usable_alignment_windows": fine_availability_report,
                "estimated_sampled_frames": int(
                    round(
                        sum(float(item["selected_seconds"]) for item in fine_coverage.values())
                        * fine_sample_fps
                    )
                ),
            }
            if (
                self.config["performance"].get(
                    "fine_risk_window_expansion_enabled", False
                )
                or self.config["performance"].get(
                    "fine_low_alignment_extra_padding_enabled", False
                )
            ):
                fine_window_report["risk_window_expansion"] = {
                    "mode": "expand_only_preserve_baseline_windows",
                    "action_types": list(
                        self.config["performance"].get(
                            "fine_risk_action_types", []
                        )
                    ),
                    "low_confidence_threshold": float(
                        self.config["performance"].get(
                            "fine_risk_low_confidence_threshold", 0.70
                        )
                    ),
                    "extra_padding_seconds": float(
                        self.config["performance"].get(
                            "fine_risk_extra_padding_seconds", 30.0
                        )
                    ),
                    "low_alignment_extra_padding_enabled": bool(
                        self.config["performance"].get(
                            "fine_low_alignment_extra_padding_enabled", False
                        )
                    ),
                    "low_alignment_confidence_threshold": float(
                        self.config["performance"].get(
                            "fine_low_alignment_confidence_threshold", 0.80
                        )
                    ),
                    "low_alignment_extra_padding_seconds": float(
                        self.config["performance"].get(
                            "fine_low_alignment_extra_padding_seconds", 30.0
                        )
                    ),
                }
            write_json(layout.json_config / "fine_scan_windows.json", fine_window_report)
            if (
                self.config["performance"].get(
                    "fine_coverage_gate_enabled", False
                )
                and not fine_availability_report["formal_evidence_ready"]
            ):
                raise RuntimeError(
                    "精扫没有同时可用的第一/第三人称对齐窗口；查看 "
                    f"{layout.json_config / 'fine_scan_windows.json'}"
                )
            eligible_fine_ids = {view.view_id for view in fine_views}
            for view in manifest.views:
                self._view_runtime[view.view_id]["state"] = (
                    "fine_pending" if view.view_id in eligible_fine_ids else "fine_not_selected"
                )
            progressive_report = None
            self._status(
                layout,
                "candidate_fine",
                0.48,
                (
                    "渐进精扫：先核验主双视角，缺失窗口按对齐时间戳补扫"
                    if progressive_enabled
                    else "候选窗精扫并收紧动作边界"
                ),
            )
            fine_work_dir = layout.work / "detections-fine"
            fine_frame_index_report: dict[str, Any] | None = None
            fine_runtime_index: FineFrameIndex | None = None
            if progressive_enabled:
                detection_paths, scanned_fine_views, candidates, progressive_report = (
                    self._run_progressive_fine_scan(
                        manifest,
                        fine_views,
                        fine_view_report,
                        boundary_candidates,
                        fine_windows,
                        infos,
                        transforms,
                        fine_work_dir,
                    )
                )
                write_json(
                    layout.json_config / "progressive_fine_scan.json",
                    progressive_report,
                )
                write_json(
                    layout.json_config / "fine_view_selection.json",
                    fine_view_report,
                )
                fine_window_report.update(
                    {
                        "selected_view_ids": progressive_report["scanned_view_ids"],
                        "coverage": progressive_report["actual_coverage"],
                        "estimated_sampled_frames": progressive_report[
                            "total_actual_estimated_frames"
                        ],
                        "full_fine_estimated_frames": progressive_report[
                            "actual_estimated_frames"
                        ],
                        "scout_estimated_frames": progressive_report[
                            "scout_estimated_frames"
                        ],
                        "full_pool_coverage": fine_coverage,
                        "full_pool_estimated_sampled_frames": progressive_report[
                            "all_view_estimated_frames"
                        ],
                        "avoided_selected_seconds": progressive_report[
                            "avoided_selected_seconds"
                        ],
                        "avoided_estimated_frames": progressive_report[
                            "avoided_estimated_frames"
                        ],
                    }
                )
                write_json(
                    layout.json_config / "fine_scan_windows.json", fine_window_report
                )
                fine_frame_index_report = progressive_report.get(
                    "fine_frame_index"
                )
                if fine_frame_index_report is not None:
                    fine_runtime_index = FineFrameIndex(
                        Path(str(fine_frame_index_report["index_path"]))
                    )
            else:
                fine_manifest = manifest.model_copy(update={"views": fine_views})
                detection_paths = self._scan_all_views_concurrently(
                    fine_manifest,
                    infos,
                    transforms,
                    fine_work_dir,
                    windows=fine_windows,
                    sample_fps=fine_sample_fps,
                    phase="fine",
                )
                scanned_fine_views = fine_views
                fine_frame_index: FineFrameIndex | None = None
                if self.config["performance"].get(
                    "fine_frame_index_enabled", False
                ):
                    fine_frame_index = create_fine_frame_index(
                        self._frame_index_path(fine_work_dir, "fine_frame_index.sqlite3")
                    )
                    fine_runtime_index = fine_frame_index
                    ingest_report = ingest_fine_frame_ledgers(
                        fine_frame_index,
                        scanned_fine_views,
                        detection_paths,
                        source_pass="single-pass",
                        stitching_enabled=bool(
                            self.config["performance"].get(
                                "fine_track_stitching_enabled", False
                            )
                        ),
                        maximum_stitch_gap_ms=float(
                            self.config["performance"].get(
                                "fine_track_stitch_max_gap_seconds", 2.5
                            )
                        )
                        * 1000.0,
                        maximum_center_distance=float(
                            self.config["performance"].get(
                                "fine_track_stitch_max_center_distance", 0.12
                            )
                        ),
                    )
                    fine_frame_index_report = fine_frame_coverage_report(
                        fine_frame_index,
                        scanned_fine_views,
                        infos,
                        fine_windows,
                        sample_fps=fine_sample_fps,
                        minimum_coverage_ratio=float(
                            self.config["performance"].get(
                                "fine_minimum_coverage_ratio", 0.98
                            )
                        ),
                        maximum_gap_periods=float(
                            self.config["performance"].get(
                                "fine_maximum_gap_periods", 4.0
                            )
                        ),
                        alignment_scales={
                            view.view_id: transforms[view.view_id].scale
                            for view in scanned_fine_views
                        },
                    )
                    fine_frame_index_report["ingest_passes"] = [
                        ingest_report
                    ]
                    detection_paths = fine_frame_index.materialize_ledgers(
                        fine_work_dir / "indexed-detections",
                        scanned_fine_views,
                    )
                candidates = (
                    generate_candidates(
                        scanned_fine_views,
                        detection_paths,
                        self.config,
                    )
                    if fine_frame_index is None
                    else generate_candidates(
                        scanned_fine_views,
                        detection_paths,
                        self.config,
                        fine_frame_index,
                    )
                )
            fine_roi_candidates, fine_roi_report = (
                generate_open_vocabulary_fine_candidates(
                    scanned_fine_views,
                    infos,
                    detection_paths,
                    fine_windows,
                    self.config,
                    fine_runtime_index,
                )
            )
            candidates = sorted(
                [*candidates, *fine_roi_candidates], key=candidate_sort_key
            )
            write_json(
                layout.json_config / "fine_roi_open_vocabulary_recall.json",
                fine_roi_report,
            )
            if fine_frame_index_report is not None:
                write_json(
                    layout.json_config / "fine_frame_index_manifest.json",
                    fine_frame_index_report,
                )
            scanned_fine_ids = {view.view_id for view in scanned_fine_views}
            for view in fine_views:
                if view.view_id not in scanned_fine_ids:
                    self._view_runtime[view.view_id]["state"] = (
                        "fine_not_scanned_no_remaining_evidence_gap"
                    )
            self._archive_scan_runtime(
                layout,
                fine_work_dir,
                "fine",
                progressive_report=progressive_report,
            )
            if (
                self.config["performance"].get(
                    "fine_coverage_gate_enabled", False
                )
                and (
                    fine_frame_index_report is None
                    or not fine_frame_index_report.get(
                        "formal_evidence_ready", False
                    )
                )
            ):
                raise RuntimeError(
                    "精扫实际帧覆盖门禁失败；查看 "
                    f"{layout.json_config / 'fine_frame_index_manifest.json'}"
                )
            if (
                self.config["performance"].get(
                    "fine_roi_open_vocabulary_recall_enabled", False
                )
                and not fine_roi_report.get("formal_evidence_ready", False)
            ):
                raise RuntimeError(
                    "精扫手部 ROI 开放词汇补救门禁失败；查看 "
                    f"{layout.json_config / 'fine_roi_open_vocabulary_recall.json'}"
                )
            scout_work_dir = fine_work_dir / "scout"
            if scout_work_dir.is_dir():
                self._archive_scan_runtime(
                    layout,
                    scout_work_dir,
                    "fine_scout",
                )
            from .movement_verification import verify_movement_candidates

            movement_report = verify_movement_candidates(
                candidates, scanned_fine_views, infos, detection_paths, self.config,
                progress=lambda done, total: self._status(
                    layout, "candidate_fine", 0.68, f"移动画面核验 {done}/{total}"
                ),
            )
            write_json(layout.json_config / "movement_visual_verification.json", movement_report)
            if self.config["archive"].get("keep_debug_candidates"):
                write_json(layout.json_config / "candidate_layer.json",
                           [candidate.model_dump(mode="json") for candidate in candidates])
            fine_artifacts = [
                layout.json_config / "movement_visual_verification.json",
                layout.json_config / "scan_runtime_fine.json",
                layout.json_config / "fine_scan_windows.json",
            ]
            fine_index_manifest_path = (
                layout.json_config / "fine_frame_index_manifest.json"
            )
            if fine_index_manifest_path.exists():
                fine_artifacts.append(fine_index_manifest_path)
            fine_roi_path = (
                layout.json_config / "fine_roi_open_vocabulary_recall.json"
            )
            if fine_roi_path.exists():
                fine_artifacts.append(fine_roi_path)
            progressive_path = layout.json_config / "progressive_fine_scan.json"
            if progressive_path.exists():
                fine_artifacts.append(progressive_path)
            scout_runtime_path = (
                layout.json_config / "scan_runtime_fine_scout.json"
            )
            if scout_runtime_path.exists():
                fine_artifacts.append(scout_runtime_path)
            candidate_layer = layout.json_config / "candidate_layer.json"
            if candidate_layer.exists():
                fine_artifacts.append(candidate_layer)
            self._complete_stage(layout, "candidate_fine", fine_artifacts)

            self._status(layout, "candidate_audit", 0.68, "持续性、动作密度与跨视角一致性审计")
            candidates = sorted(candidates, key=candidate_sort_key)
            if fine_runtime_index is None:
                events, rejected = audit_candidates(
                    candidates,
                    transforms,
                    self.config,
                )
            else:
                events, rejected = audit_candidates(
                    candidates,
                    transforms,
                    self.config,
                    fine_runtime_index,
                )
            if fine_runtime_index is None:
                rejected.extend(
                    refine_liquid_events_with_context(events, detection_paths)
                )
            else:
                rejected.extend(
                    refine_liquid_events_with_context(
                        events,
                        detection_paths,
                        frame_index=fine_runtime_index,
                        config=self.config,
                    )
                )
            state_machine_ledger = attach_continuous_action_states(events, self.config)
            observability_receipts = attach_action_observability(events)
            semantic_review_plan = build_semantic_review_plan(events, self.config)
            observability_path = layout.json_config / "action_observability.json"
            semantic_review_plan_path = (
                layout.json_config / "semantic_review_plan.json"
            )
            state_machine_path = (
                layout.json_config / "continuous_action_state_ledger.json"
            )
            write_json(state_machine_path, state_machine_ledger)
            write_json(
                observability_path,
                {
                    "schema_version": "visioncortex-action-observability-ledger/1",
                    "event_count": len(events),
                    "receipts": observability_receipts,
                },
            )
            write_json(semantic_review_plan_path, semantic_review_plan)
            raw_boundary_receipts: list[dict[str, Any]] = []
            segmentation_boundary_candidates = (
                [] if exhaustive_full_timeline else boundary_candidates
            )
            if exhaustive_full_timeline:
                raw_boundary_receipts.append(
                    decision_receipt(
                        decision_type="full_timeline_segmentation_basis",
                        rule_id="QF1-EXHAUSTIVE-FINE-EVIDENCE-TIMELINE",
                        verdict="accepted",
                        subject_ids=[manifest.experiment_id],
                        reason_codes=[
                            "sparse_motion_windows_not_used_for_event_routing"
                        ],
                        facts={
                            "full_timeline_fine_scan": True,
                            "fine_view_ids": sorted(scanned_fine_ids),
                            "sparse_motion_candidate_ids": [
                                candidate.candidate_id
                                for candidate in boundary_candidates
                            ],
                            "segmentation_boundary_candidate_ids": [],
                            "formal_membership_changed": False,
                        },
                        evidence_refs=[
                            candidate.candidate_id
                            for candidate in boundary_candidates
                        ],
                    )
                )
            raw_segments = build_experiment_segments(
                events,
                manifest.views,
                self.config,
                coarse_windows=segmentation_boundary_candidates,
                decision_receipts=raw_boundary_receipts,
            )
            normalization_receipts: list[dict[str, Any]] = []
            normalized_segments = normalize_experiment_segments(
                raw_segments,
                events,
                manifest.views,
                self.config,
                decision_receipts=normalization_receipts,
            )
            segments, formal_segment_receipts = prepare_formal_experiment_segments(
                normalized_segments,
                events,
                manifest.views,
                segmentation_boundary_candidates,
                self.config,
            )
            continuity_receipts: list[dict[str, Any]] = []
            groups = build_experiment_groups(
                segments,
                events,
                manifest.views,
                self.config,
                decision_receipts=continuity_receipts,
                coarse_windows=segmentation_boundary_candidates,
            )
            selection_decisions: list[dict[str, Any]] = []
            precheck_key_events = select_key_events(
                groups,
                segments,
                events,
                self.config,
                decision_receipts=selection_decisions,
            )
            key_selection_path = (
                layout.json_config / "key_material_selection_preview.json"
            )
            selection_report = _key_material_selection_report(
                groups,
                segments,
                events,
                selection_decisions,
            )
            write_json(key_selection_path, selection_report)
            self._preprocessing_completed_seconds = round(time.perf_counter() - self._run_started_perf, 6)
            write_json(
                layout.json_config / "audit_layer.json",
                {
                    "events": [event.model_dump(mode="json") for event in events],
                    "rejected": rejected,
                    "boundary_candidates": [
                        candidate.model_dump(mode="json")
                        for candidate in sorted(
                            boundary_candidates,
                            key=candidate_sort_key,
                        )
                    ],
                    "raw_segments": [
                        segment.model_dump(mode="json") for segment in raw_segments
                    ],
                    "normalized_segments": [
                        segment.model_dump(mode="json")
                        for segment in normalized_segments
                    ],
                    "segments": [segment.model_dump(mode="json") for segment in segments],
                    "raw_boundary_decision_receipts": raw_boundary_receipts,
                    "formal_segment_receipts": formal_segment_receipts,
                    "normalization_decision_receipts": normalization_receipts,
                    "continuity_decision_receipts": continuity_receipts,
                    "quality_decision_receipts": [
                        *raw_boundary_receipts,
                        *normalization_receipts,
                        *formal_segment_receipts,
                        *continuity_receipts,
                        *selection_decisions,
                    ],
                    "experiment_groups": [group.model_dump(mode="json") for group in groups],
                    "action_observability": observability_receipts,
                    "continuous_action_state": state_machine_ledger,
                    "semantic_review_plan": semantic_review_plan,
                },
            )
            boundary_precheck = self._run_boundary_precheck(
                layout, groups, progressive_report=progressive_report
            )
            self._complete_stage(
                layout,
                "candidate_audit",
                [
                    layout.json_config / "audit_layer.json",
                    layout.json_config / "boundary_precheck.json",
                    key_selection_path,
                    observability_path,
                    state_machine_path,
                    semantic_review_plan_path,
                ],
            )

            if bool(
                self.config.get("project", {}).get(
                    "preprocessing_acceptance_only", False
                )
            ):
                self._status(
                    layout,
                    "preprocessing_acceptance",
                    0.99,
                    "验证有界双视角实验与关键事件选择，不调用模型或导出媒体",
                )
                key_events = precheck_key_events
                baseline = self._acceptance_baseline() or {}
                minimum_key_events = baseline.get("minimum_selected_key_events")
                key_event_gate_passed = (
                    minimum_key_events is None
                    or len(key_events) >= int(minimum_key_events)
                )
                acceptance_passed = bool(boundary_precheck["passed"]) and bool(
                    key_event_gate_passed
                )
                acceptance_path = (
                    layout.json_config / "preprocessing_acceptance.json"
                )
                write_json(
                    acceptance_path,
                    {
                        "schema_version": (
                            "visioncortex-preprocessing-acceptance/1"
                        ),
                        "status": (
                            "passed" if acceptance_passed else "failed"
                        ),
                        "run_mode": "preprocessing_acceptance_only",
                        "formal_archive_promotion_allowed": False,
                        "model_api_calls": 0,
                        "token_usage": {
                            "input_tokens": 0,
                            "output_tokens": 0,
                            "total_tokens": 0,
                        },
                        "preprocessing_seconds": (
                            self._preprocessing_completed_seconds
                        ),
                        "experiment_group_count": len(groups),
                        "selected_key_event_count": len(key_events),
                        "minimum_selected_key_events": minimum_key_events,
                        "key_event_gate_passed": key_event_gate_passed,
                        "baseline_selection": dict(
                            self._acceptance_baseline_selection
                        ),
                        "experiment_groups": [
                            group.model_dump(mode="json") for group in groups
                        ],
                        "selected_key_event_ids": [
                            event.event_id for event in key_events
                        ],
                        "boundary_precheck": boundary_precheck,
                        "key_material_selection_preview": selection_report,
                        "limitations": [
                            "No MLLM experiment naming or step understanding was run.",
                            "No experiment clips, key frames, key clips, daily report, or PDF were materialized.",
                            "This staging result must not replace the accepted fixed archive.",
                        ],
                    },
                )
                self._status(
                    layout,
                    "preprocessing_completed",
                    1.0,
                    "预处理性能与边界质量验收完成；正式归档未提升",
                )
                run_metrics = self._metrics(key_events, groups)
                run_metrics["run_mode"] = "preprocessing_acceptance_only"
                run_metrics["formal_archive_promotion_allowed"] = False
                write_json(layout.json_config / "run_metrics.json", run_metrics)
                if not acceptance_passed:
                    raise RuntimeError(
                        "Preprocessing acceptance failed after evidence selection; "
                        f"see {acceptance_path}"
                    )
                self._complete_stage(
                    layout,
                    "preprocessing_acceptance",
                    [
                        acceptance_path,
                        key_selection_path,
                        layout.json_config / "run_metrics.json",
                    ],
                )
                return layout.root

            self._status(layout, "experiment_understanding", 0.72, "用完整有界双视角故事板命名实验并核验连续性")
            if groups:
                (layout.json_config / "speech_understanding.json").unlink(missing_ok=True)
            if not groups and speech.enabled(self.config):
                from .speech_semantics import analyze_unsegmented_recording

                self._status(layout, "experiment_understanding", 0.72, "未发现可确认实验片段，正在用实际画面核对实验录音")
                try:
                    self._speech_understanding = analyze_unsegmented_recording(
                        layout, manifest.views, infos, transforms, self.config
                    )
                finally:
                    recording_receipt = layout.json_config / "speech_understanding.json"
                    if recording_receipt.is_file():
                        self._speech_understanding = json.loads(recording_receipt.read_text(encoding="utf-8"))
            from .boundary_review import review_experiment_boundaries

            groups = review_experiment_boundaries(
                layout, groups, segments, events, manifest.views, infos, transforms, self.config
            )
            if any(group.boundary_reviews for group in groups):
                # Retain the original CV partition and publish the reconciled
                # group identities before names, clips and key materials exist.
                audit_path = layout.json_config / "audit_layer.json"
                audit = json.loads(audit_path.read_text(encoding="utf-8-sig"))
                audit["experiment_groups_before_boundary_review"] = audit.get("experiment_groups", [])
                audit["experiment_groups"] = [g.model_dump(mode="json") for g in groups]
                audit["segments"] = [s.model_dump(mode="json") for s in segments]
                write_json(audit_path, audit)
                write_json(layout.json_config / "boundary_precheck_before_semantic_review.json", boundary_precheck)
                boundary_precheck = self._run_boundary_precheck(layout, groups, progressive_report=progressive_report)
            analyze_experiment_groups(
                layout, groups, segments, events, manifest.views, infos, transforms, self.config
            )
            group_semantic_recall = _recover_group_storyboard_state_events(
                groups,
                segments,
                events,
                self.config,
            )
            group_semantic_recall_path = (
                layout.json_config / "group_storyboard_semantic_recall.json"
            )
            write_json(
                group_semantic_recall_path,
                {
                    "schema_version": (
                        "visioncortex-group-storyboard-semantic-recall/1"
                    ),
                    "policy": (
                        "explicit dual-view closure transition; recall-only; "
                        "independent event-level Ark proof required"
                    ),
                    "recovered_event_count": len(group_semantic_recall),
                    "events": group_semantic_recall,
                },
            )
            if group_semantic_recall:
                # The first ledgers are written before Ark so preprocessing can
                # fail closed without any model call.  A real group review may
                # add only provisional recall events; refresh the shadow
                # ledgers so their semantic origin and incomplete state remain
                # explicit until event-level adjudication.
                observability_receipts = attach_action_observability(events)
                state_machine_ledger = attach_continuous_action_states(
                    events, self.config
                )
                semantic_review_plan = build_semantic_review_plan(
                    events, self.config
                )
                write_json(observability_path, {
                    "schema_version": "visioncortex-action-observability-ledger/1",
                    "event_count": len(events),
                    "receipts": observability_receipts,
                })
                write_json(state_machine_path, state_machine_ledger)
                write_json(semantic_review_plan_path, semantic_review_plan)
                audit_layer_path = layout.json_config / "audit_layer.json"
                audit_layer = json.loads(
                    audit_layer_path.read_text(encoding="utf-8-sig")
                )
                audit_layer.update(
                    {
                        "events": [
                            event.model_dump(mode="json") for event in events
                        ],
                        "segments": [
                            segment.model_dump(mode="json")
                            for segment in segments
                        ],
                        "experiment_groups": [
                            group.model_dump(mode="json") for group in groups
                        ],
                        "action_observability": observability_receipts,
                        "continuous_action_state": state_machine_ledger,
                        "semantic_review_plan": semantic_review_plan,
                        "group_storyboard_semantic_recall": {
                            "path": group_semantic_recall_path.relative_to(
                                layout.root
                            ).as_posix(),
                            "events": group_semantic_recall,
                        },
                    }
                )
                write_json(audit_layer_path, audit_layer)
            group_understanding_path = layout.json_config / "experiment_group_understanding.json"
            write_json(
                group_understanding_path,
                {
                    "schema_version": "visioncortex-experiment-group-understanding/1",
                    "groups": [group.model_dump(mode="json") for group in groups],
                },
            )
            write_json(layout.json_config / "run_metrics_live.json", self._metrics(events, groups))
            final_selection_decisions: list[dict[str, Any]] = []
            key_events = select_key_events(
                groups,
                segments,
                events,
                self.config,
                decision_receipts=final_selection_decisions,
            )
            key_selection_path = layout.json_config / "key_material_selection.json"
            write_json(
                key_selection_path,
                _key_material_selection_report(
                    groups,
                    segments,
                    events,
                    final_selection_decisions,
                ),
            )
            self._complete_stage(
                layout,
                "experiment_understanding",
                [
                    group_understanding_path,
                    *([layout.json_config / "speech_understanding.json", layout.root / "Key-Materials/Experiment-Audio"] if (layout.json_config / "speech_understanding.json").is_file() else []),
                    group_semantic_recall_path,
                    key_selection_path,
                    layout.json_config / "run_metrics_live.json",
                ],
            )

            self._status(layout, "experiment_clips", 0.78, "按模型实验名归档第一/第三/并排三份有界视频")
            materialize_experiment_clips(
                layout,
                groups,
                segments,
                events,
                manifest.views,
                infos,
                transforms,
                self.config,
                publisher=self._publisher,
            )
            self._complete_stage(
                layout,
                "experiment_clips",
                [
                    layout.experiment_clips,
                    layout.json_config / "experiment_clip_materialization_runtime.json",
                ],
            )

            self._status(layout, "key_materials", 0.84, "按实验组提取去重后的分层动作关键素材")
            materialize_key_materials(
                layout,
                key_events,
                groups,
                manifest.views,
                infos,
                transforms,
                detection_paths,
                self.config,
                publisher=self._publisher,
                archive_id=manifest.experiment_id,
            )
            self._complete_stage(
                layout,
                "key_materials",
                [
                    layout.key_materials,
                    layout.json_config / "key_material_materialization_runtime.json",
                ],
            )

            self._status(layout, "mllm", 0.92, "调用已配置的图像模型理解去重后的关键动作当前/下一步骤")
            analyze_key_materials(
                layout, key_events, self.config,
                progress_callback=lambda done, total: self._status(
                    layout, "mllm", 0.92 + 0.009 * done / max(1, total),
                    f"关键动作模型理解已处理 {done}/{total}，正在核验画面证据",
                ),
            )
            reviewed_key_events = list(key_events)
            self._complete_stage(
                layout,
                "mllm",
                self._checkpoint_key_material_understanding(
                    layout, "mllm", reviewed_key_events, groups
                ),
            )
            self._status(layout, "material_refinement", 0.93, "复核关键画面与动作参与对象")
            key_events, semantic_curation = curate_semantically_reviewed_key_materials(
                layout,
                reviewed_key_events,
                groups,
                self.config,
                publisher=self._publisher,
            )
            segment_semantic_repairs = _synchronize_segments_with_final_key_events(
                segments, groups, key_events
            )
            # Semantic adjudication can replace the action, participants, or
            # strongest supporting role. Re-materialize only events whose final
            # evidence can change the selected frame/boxes instead of repeating
            # every accepted clip after the model pass.
            pre_final_key_timestamps = {
                event.event_id: float(event.key_global_ms) for event in key_events
            }
            curation_record_by_event = {
                str(record["event_id"]): record
                for record in semantic_curation.get("records") or []
            }
            rematerialize_event_ids: set[str] = set()
            for event in key_events:
                record = curation_record_by_event.get(event.event_id) or {}
                selected_source_view = str(
                    (
                        (event.observability or {}).get("key_frame_selection")
                        or {}
                    ).get("source_view_id")
                    or ""
                )
                direct_view_ids = {
                    str(item)
                    for item in (event.semantic_review or {}).get(
                        "directly_supported_view_ids", []
                    )
                    if item
                }
                if (
                    record.get("semantic_participant_refinement_changed")
                    or record.get("cv_action_type")
                    != record.get("final_action_type")
                    or (
                        direct_view_ids
                        and selected_source_view not in direct_view_ids
                    )
                ):
                    rematerialize_event_ids.add(event.event_id)
            if rematerialize_event_ids:
                materialize_key_materials(
                    layout,
                    key_events,
                    groups,
                    manifest.views,
                    infos,
                    transforms,
                    detection_paths,
                    self.config,
                    publisher=self._publisher,
                    archive_id=manifest.experiment_id,
                    materialize_event_ids=rematerialize_event_ids,
                    progress_callback=lambda done, total: self._status(
                        layout,
                        "material_refinement",
                        0.93 + 0.01 * done / max(1, total),
                        f"复核关键画面：{done}/{total}",
                    ),
                )
            state_receipt_repairs = _synchronize_final_event_state_receipts(
                key_events, self.config
            )
            self._status(layout, "material_refinement", 0.94, "校验参与对象标注与连续性")
            final_annotation = _rerender_curated_participant_annotations(
                layout, key_events, groups, self.config
            )
            key_events, semantic_curation = (
                reconcile_visually_reviewed_participants(
                    layout,
                    key_events,
                    groups,
                    semantic_curation,
                )
            )
            visual_reconciliation = semantic_curation.get(
                "visual_participant_reconciliation"
            ) or {}
            if (
                int(visual_reconciliation.get("pruned_event_count") or 0)
                or int(visual_reconciliation.get("excluded_event_count") or 0)
            ):
                # The first render is the evidence used to adjudicate unstable
                # paper/cap participants.  Rebuild the surviving user tree from
                # the reconciled participant sets.  All cloud requests are
                # content-addressed and reused; removed events stay in the
                # formal Review-Candidates tree.
                state_receipt_repairs.extend(
                    _synchronize_final_event_state_receipts(
                        key_events, self.config
                    )
                )
                segment_semantic_repairs.extend(
                    _synchronize_segments_with_final_key_events(
                        segments, groups, key_events
                    )
                )
                write_key_material_category_index(
                    layout,
                    groups,
                    key_events,
                    include_empty_categories=bool(
                        self.config.get("archive", {}).get(
                            "include_empty_action_categories", True
                        )
                    ),
                )
                final_annotation = _rerender_curated_participant_annotations(
                    layout, key_events, groups, self.config
                )
            semantic_curation["final_annotation"] = {
                key: value
                for key, value in final_annotation.items()
                if key != "records"
            }
            semantic_curation["post_semantic_participant_key_frame_selection"] = {
                "schema_version": (
                    "visioncortex-post-semantic-participant-key-frame-selection/1"
                ),
                "policy": (
                    "immutable-fine-ledger; selective refresh for changed final evidence"
                ),
                "full_scan_repeated": False,
                "source_copy_bytes": 0,
                "rematerialized_event_ids": sorted(rematerialize_event_ids),
                "unchanged_media_reused_event_ids": sorted(
                    event.event_id
                    for event in key_events
                    if event.event_id not in rematerialize_event_ids
                ),
                "events": [
                    {
                        "event_id": event.event_id,
                        "previous_key_global_ms": pre_final_key_timestamps[
                            event.event_id
                        ],
                        "selected_key_global_ms": float(event.key_global_ms),
                        "selection_offset_ms": round(
                            float(event.key_global_ms)
                            - pre_final_key_timestamps[event.event_id],
                            3,
                        ),
                        "media_rematerialized": (
                            event.event_id in rematerialize_event_ids
                        ),
                    }
                    for event in key_events
                ],
                "state_receipt_repairs": state_receipt_repairs,
                "segment_semantic_repairs": segment_semantic_repairs,
            }
            write_json(
                layout.json_config / "semantic_key_material_curation.json",
                semantic_curation,
            )
            # The initial group pass names the bounded experiment and verifies
            # continuity. Final steps come from the stronger event-level
            # adjudication, so rebuild them deterministically instead of paying
            # for a duplicate group MLLM pass over the same evidence.
            refine_groups_from_final_events(groups, key_events)
            normalize_final_group_action_language(groups, key_events)
            for group in groups:
                for segment in segments:
                    if segment.group_id != group.group_id:
                        continue
                    segment.experiment_name = group.experiment_name
                    segment.experiment_name_en = group.experiment_name_en
                    segment.semantic_understanding = group.model_understanding
            write_json(
                group_understanding_path,
                {
                    "schema_version": "visioncortex-experiment-group-understanding/2",
                    "refinement_pass": "post_event_semantic_curation",
                    "groups": [group.model_dump(mode="json") for group in groups],
                },
            )
            refresh_key_material_metadata(
                layout,
                key_events,
                groups,
                transforms,
                archive_id=manifest.experiment_id,
            )
            self._complete_stage(
                layout,
                "semantic_refinement",
                [
                    layout.key_materials,
                    group_understanding_path,
                    *self._checkpoint_key_material_understanding(
                        layout, "semantic_refinement", reviewed_key_events,
                        groups, semantic_curation,
                    ),
                ],
            )
            # Only semantically curated formal key events may create durable
            # physical-change claims. Accepted CV recall candidates that never
            # reached a formal dual-view segment must not leak into the report
            # as liquid_transferred/container_state_changed facts.
            physical_changes = build_physical_change_log(key_events)

            self._status(layout, "package", 0.96, "归档证据包并执行 evidence-package-eval")
            summary = finalize_archive(
                layout,
                manifest,
                infos,
                transforms,
                events,
                segments,
                groups,
                key_events,
                physical_changes,
                rejected,
                self.config,
                disk_report,
            )
            # Validate the archive-level experiment groups. A continuous group
            # may contain multiple atomic segments but must count as one bounded
            # experiment in the user-facing output and boundary evaluation.
            self._run_sidecar_validation(layout, manifest, groups)
            quality_acceptance = self._run_quality_acceptance(
                layout, groups, key_events
            )
            if quality_acceptance.get("passed") is not True:
                return self._finish_quality_attention(layout, events, groups)
            # A successful retry must not publish the previous attempt's
            # current partial report. The retry endpoint preserves its history.
            (layout.json_config / "partial_delivery.json").unlink(missing_ok=True)
            (layout.root / "Partial-Results/Partial-Evidence-Report.html").unlink(missing_ok=True)
            self._complete_stage(layout, "package", [layout.json_config])
            self._status(layout, "daily_report", 0.98, "从已验收证据生成实验室日报并执行一致性校验")
            generate_daily_report_archive(
                layout, summary, self._metrics(events, groups), self.config
            )
            self._complete_stage(
                layout,
                "daily_report",
                [layout.daily_reports, layout.professional_pdfs],
            )
            if not self.config["archive"].get("keep_debug_candidates") and layout.work.exists():
                shutil.rmtree(layout.work)
            self._status(
                layout,
                "finalizing",
                0.995,
                "封存最终运行指标、溯源清单和归档契约",
            )
            run_metrics = self._metrics(events, groups)
            summary.stats["run_metrics"] = run_metrics
            summary.stats["run_provenance"] = {
                "path": "JSON-Config-Files/run_provenance.json"
            }
            write_json(
                layout.json_config / "run_metrics.json",
                run_metrics,
            )
            write_json(
                layout.json_config / "evidence_package.json",
                summary.model_dump(mode="json"),
            )
            write_run_provenance(
                layout.root,
                self.config,
                self._model_certification_audit,
                repository_root=Path(__file__).resolve().parents[2],
            )
            write_archive_contract_manifest(layout.root)
            validate_archive_contracts_or_raise(layout.root)
            self._complete_stage(
                layout,
                "finalizing",
                [
                    layout.experiment_clips,
                    layout.key_materials,
                    layout.json_config,
                    layout.daily_reports,
                    layout.professional_pdfs,
                ],
            )
            self._status(layout, "completed", 1.0, "处理完成并通过自动发布前验收")
            self._complete_stage(layout, "completed", [layout.json_config])
            return layout.root
        except Exception as exc:
            self._status(layout, "failed", 1.0, f"{type(exc).__name__}: {exc}")
            partial_metrics = self._metrics(
                locals().get("events", []), locals().get("groups", [])
            )
            write_json(
                layout.json_config / "run_metrics.json",
                partial_metrics,
            )
            try:
                write_partial_delivery(layout.root, partial_metrics)
                if self._publisher is not None:
                    self._publisher.publish_directory("Partial-Results")
            except (OSError, ValueError, KeyError, TypeError) as report_exc:
                # Retain the original failure and existing media even if a
                # damaged control file prevents the supplemental report.
                write_json(layout.json_config / "partial_delivery_error.json", {
                    "status": "failed",
                    "exception_class": type(report_exc).__name__,
                })
            if self._publisher is not None:
                self._publisher.publish_directory("JSON-Config-Files")
            raise
        finally:
            if self._resource_monitor is not None:
                self._resource_monitor.stop()
                if self._publisher is not None:
                    self._publisher.publish_file(layout.json_config / "resource_telemetry.json")
            # A stale run lock blocks all future processing for the same
            # experiment. Cleanup failure is therefore a pipeline failure, not
            # an ignorable housekeeping warning.
            lock_path.unlink(missing_ok=True)

    @staticmethod
    def _acquire_lock(lock_path: Path) -> None:
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            owner = lock_path.read_text(encoding="utf-8", errors="replace") if lock_path.is_file() else "unknown"
            raise RuntimeError(f"该实验已有运行实例: {owner}") from exc
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps({"pid": os.getpid(), "started_at": datetime.now(timezone.utc).isoformat()}))

    def _fine_windows(
        self,
        candidates,
        infos,
        transforms,
        padding_seconds: float | None = None,
    ) -> dict[str, list[tuple[float, float]]]:
        if bool(
            self.config["performance"].get(
                "exhaustive_negative_audit_full_timeline", False
            )
            or self.config["performance"].get(
                "exhaustive_full_timeline_scan", False
            )
        ):
            return {
                view_id: [(0.0, float(info.duration_ms))]
                for view_id, info in infos.items()
                if float(info.duration_ms) > 0.0
            }
        padding = float(
            self.config["performance"]["fine_window_padding_seconds"]
            if padding_seconds is None
            else padding_seconds
        ) * 1000.0
        merge_gap = max(
            0.0,
            float(
                self.config["performance"].get(
                    "fine_window_merge_gap_seconds", 0.0
                )
            )
            * 1000.0,
        )
        candidate_list = list(candidates)
        risk_extra_by_id: dict[str, float] = {}
        perf = self.config["performance"]
        if perf.get("fine_risk_window_expansion_enabled", False):
            configured_actions = {
                str(item).strip()
                for item in perf.get("fine_risk_action_types", [])
                if str(item).strip()
            }
            low_confidence_threshold = float(
                perf.get("fine_risk_low_confidence_threshold", 0.70)
            )
            extra_padding_ms = max(
                0.0,
                float(perf.get("fine_risk_extra_padding_seconds", 30.0))
                * 1000.0,
            )
            conflict_gap_ms = max(
                0.0,
                float(perf.get("fine_risk_conflict_gap_seconds", 15.0))
                * 1000.0,
            )
            for candidate in candidate_list:
                if (
                    candidate.action_type.value in configured_actions
                    or float(candidate.confidence) < low_confidence_threshold
                ):
                    risk_extra_by_id[candidate.candidate_id] = extra_padding_ms
            active_candidates: list[ActionCandidate] = []
            for current in sorted(
                candidate_list,
                key=lambda item: (item.global_start_ms, item.global_end_ms),
            ):
                active_candidates = [
                    previous
                    for previous in active_candidates
                    if previous.global_end_ms + conflict_gap_ms
                    >= current.global_start_ms
                ]
                for previous in active_candidates:
                    if previous.action_type == current.action_type:
                        continue
                    risk_extra_by_id[previous.candidate_id] = extra_padding_ms
                    risk_extra_by_id[current.candidate_id] = extra_padding_ms
                active_candidates.append(current)
        grouped: dict[str, list[tuple[float, float]]] = {view_id: [] for view_id in infos}
        for candidate in candidate_list:
            candidate_padding = padding + risk_extra_by_id.get(
                candidate.candidate_id, 0.0
            )
            for view_id, info in infos.items():
                alignment_extra = 0.0
                transform = transforms[view_id]
                if (
                    perf.get("fine_low_alignment_extra_padding_enabled", False)
                    and (
                        transform.state != "aligned"
                        or float(transform.confidence)
                        < float(
                            perf.get(
                                "fine_low_alignment_confidence_threshold", 0.80
                            )
                        )
                    )
                ):
                    alignment_extra = max(
                        0.0,
                        float(
                            perf.get(
                                "fine_low_alignment_extra_padding_seconds", 30.0
                            )
                        )
                        * 1000.0,
                    )
                global_start = (
                    candidate.global_start_ms - candidate_padding - alignment_extra
                )
                global_end = (
                    candidate.global_end_ms + candidate_padding + alignment_extra
                )
                local_start = max(0.0, transforms[view_id].to_local(global_start))
                local_end = min(info.duration_ms, transforms[view_id].to_local(global_end))
                if local_end > local_start:
                    grouped[view_id].append((local_start, local_end))
        merged: dict[str, list[tuple[float, float]]] = {}
        for view_id, windows in grouped.items():
            result: list[list[float]] = []
            for start, end in sorted(windows):
                if result and start <= result[-1][1] + merge_gap:
                    result[-1][1] = max(result[-1][1], end)
                else:
                    result.append([start, end])
            merged[view_id] = [(item[0], item[1]) for item in result]
        return merged

    def _intersect_fine_windows_with_usable_alignment(
        self,
        windows: dict[str, list[tuple[float, float]]],
        infos: dict[str, VideoInfo],
        transforms: dict[str, AlignmentTransform],
        views: Sequence[ViewInput],
    ) -> tuple[dict[str, list[tuple[float, float]]], dict[str, Any]]:
        """Remove only intervals already declared unavailable by alignment."""

        view_by_id = {view.view_id: view for view in views}
        intersected: dict[str, list[tuple[float, float]]] = {}
        reports: dict[str, dict[str, Any]] = {}
        for view_id, requested in windows.items():
            transform = transforms[view_id]
            info = infos[view_id]
            if transform.state == "failed":
                usable: list[tuple[float, float]] = []
            elif transform.segment_transforms:
                usable = [
                    (
                        max(0.0, float(segment.local_start_ms)),
                        min(float(info.duration_ms), float(segment.local_end_ms)),
                    )
                    for segment in transform.segment_transforms
                    if segment.state != "failed"
                    and segment.local_end_ms > segment.local_start_ms
                ]
            elif (
                transform.local_coverage_start_ms is not None
                and transform.local_coverage_end_ms is not None
            ):
                usable = [
                    (
                        max(0.0, float(transform.local_coverage_start_ms)),
                        min(
                            float(info.duration_ms),
                            float(transform.local_coverage_end_ms),
                        ),
                    )
                ]
            else:
                usable = [(0.0, float(info.duration_ms))]
            selected = self._merge_time_windows(
                [
                    (max(start, usable_start), min(end, usable_end))
                    for start, end in requested
                    for usable_start, usable_end in usable
                    if min(end, usable_end) > max(start, usable_start)
                ]
            )
            requested_seconds = sum(
                max(0.0, end - start) for start, end in requested
            ) / 1000.0
            selected_seconds = sum(
                max(0.0, end - start) for start, end in selected
            ) / 1000.0
            intersected[view_id] = selected
            reports[view_id] = {
                "role": view_by_id[view_id].role.value,
                "alignment_state": transform.state,
                "requested_window_count": len(requested),
                "usable_interval_count": len(usable),
                "selected_window_count": len(selected),
                "requested_seconds": round(requested_seconds, 6),
                "selected_seconds": round(selected_seconds, 6),
                "excluded_unavailable_seconds": round(
                    max(0.0, requested_seconds - selected_seconds), 6
                ),
                "usable_intervals": usable,
            }
        first_ready = any(
            intersected.get(view.view_id)
            for view in views
            if view.role == ViewRole.FIRST_PERSON
        )
        third_ready = any(
            intersected.get(view.view_id)
            for view in views
            if view.role == ViewRole.THIRD_PERSON
        )
        return intersected, {
            "schema_version": "visioncortex-fine-usable-alignment-windows/1",
            "policy": "intersect_only_fail_closed_alignment_intervals",
            "first_person_window_available": first_ready,
            "third_person_window_available": third_ready,
            "formal_evidence_ready": bool(first_ready and third_ready),
            "views": reports,
        }

    def _run_sidecar_validation(self, layout: ArchiveLayout, manifest: RunManifest, groups) -> None:
        if not self.config.get("validation", {}).get("use_sidecar_annotations"):
            return
        from .validation import validate_against_sidecars

        report = validate_against_sidecars(
            manifest.views,
            groups,
            str(self.config["validation"].get("sidecar_name", "video.annotation.json")),
        )
        if report is not None:
            write_json(layout.json_config / "validation_against_annotations.json", report)


def create_dry_run(output: Path, config: dict[str, Any]) -> Path:
    layout = ArchiveLayout(output.resolve())
    layout.create()
    views = [
        ViewInput(view_id="fp01", role=ViewRole.FIRST_PERSON, video=Path("dry-run-first.mp4")),
        ViewInput(view_id="tp01", role=ViewRole.THIRD_PERSON, video=Path("dry-run-third.mp4")),
    ]
    manifest = RunManifest(experiment_id=output.name or "dry-run", views=views)
    transforms = {
        view.view_id: AlignmentTransform(
            view_id=view.view_id,
            reference_view_id="fp01",
            confidence=0.98,
            state="aligned",
            csv_match_count=100,
            csv_match_ratio=1.0,
            visual_confidence=0.95,
        )
        for view in views
    }
    candidates = [
        ActionCandidate(
            candidate_id=f"CAND-{view.view_id}-000001",
            action_type=ActionType.HAND_OBJECT_CONTACT,
            view_id=view.view_id,
            role=view.role,
            local_start_ms=10_000.0,
            local_end_ms=12_000.0,
            global_start_ms=10_000.0,
            global_end_ms=12_000.0,
            key_global_ms=11_000.0,
            objects=["gloved_hand", "pipette"],
            confidence=0.91,
        )
        for view in views
    ]
    event = EvidenceEvent(
        event_id="EVT-000001",
        action_type=ActionType.HAND_OBJECT_CONTACT,
        global_start_ms=10_000.0,
        global_end_ms=12_000.0,
        key_global_ms=11_000.0,
        objects=["gloved_hand", "pipette"],
        confidence=0.96,
        accepted=True,
        audit_reason="dry-run: 第一/第三人称一致",
        supporting_views=[view.view_id for view in views],
        supporting_roles=[ViewRole.FIRST_PERSON, ViewRole.THIRD_PERSON],
        candidates=candidates,
        model_understanding={
            "status": "completed",
            "execution_mode": "dry_run",
            "current_step": "戴手套的手接触移液器",
            "next_step": "未知",
            "confidence": 0.9,
        },
    )
    image = np.full((360, 640, 3), 35, dtype=np.uint8)
    cv2.putText(image, "DRY RUN - ALIGNED KEY FRAME", (45, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 255), 2)
    segment = ExperimentSegment(
        segment_id="EXP-0001",
        global_start_ms=8_000.0,
        global_end_ms=15_000.0,
        event_ids=[event.event_id],
        participating_views=[view.view_id for view in views],
        clips={view.view_id: f"dry-run://{view.view_id}/experiment-clip" for view in views},
        micro_segments=[
            {
                "micro_segment_id": "MICRO-0001-0001",
                "start_global_ms": 10_000.0,
                "end_global_ms": 12_000.0,
                "action_type": event.action_type.value,
                "objects": event.objects,
            }
        ],
    )
    group = ExperimentGroup(
        group_id="GROUP-0001",
        continuity_type="independent",
        atomic_experiment_ids=[segment.segment_id],
        global_start_ms=segment.global_start_ms,
        global_end_ms=segment.global_end_ms,
        participating_views=[view.view_id for view in views],
        first_person_view="fp01",
        third_person_view="tp01",
        continuity_reason="dry-run 独立实验",
        experiment_name="移液操作实验",
        experiment_name_en="Pipetting-Operation-Experiment",
        archive_folder="001_Pipetting-Operation-Experiment",
        key_event_ids=[event.event_id],
        videos={
            "first-person": "dry-run://fp01/experiment-clip",
            "third-person": "dry-run://tp01/experiment-clip",
            "aligned_first_third": "dry-run://aligned/experiment-clip",
        },
        video_json={
            "first-person": "dry-run://fp01/experiment-json",
            "third-person": "dry-run://tp01/experiment-json",
            "aligned_first_third": "dry-run://aligned/experiment-json",
        },
        model_understanding={
            "status": "completed",
            "execution_mode": "dry_run",
            "steps": [
                {
                    "step_index": 1,
                    "start_global_ms": event.global_start_ms,
                    "end_global_ms": event.global_end_ms,
                    "current_step": "戴手套的手接触移液器",
                    "next_step": "未知",
                    "next_step_status": "unknown",
                    "supporting_event_ids": [event.event_id],
                    "objects": event.objects,
                    "supporting_views": event.supporting_views,
                    "confidence": event.confidence,
                }
            ],
        },
    )
    prepare_key_material_category_layout(layout, [group])
    action_folder = key_material_action_folder(event.action_type)
    for view in views:
        path = (
            layout.key_frames
            / str(group.archive_folder)
            / action_folder
            / event.event_id
            / f"{view.view_id}.jpg"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), image)
        event.key_frames[view.view_id] = archive_relative_posix(path, layout.root)
        event.key_clips[view.view_id] = f"dry-run://{view.view_id}/key-clip"
    aligned_path = (
        layout.key_frames
        / str(group.archive_folder)
        / action_folder
        / event.event_id
        / "Aligned_First+Third.jpg"
    )
    cv2.imwrite(str(aligned_path), image)
    event.key_frames["aligned_first_third"] = archive_relative_posix(
        aligned_path, layout.root
    )
    event.key_clips["aligned_first_third"] = "dry-run://aligned/key-clip"
    write_key_material_category_index(layout, [group], [event])
    physical = [
        PhysicalChange(
            change_id="CHANGE-000001",
            event_id=event.event_id,
            global_ms=11_000.0,
            change_type="contact_started",
            object_names=event.objects,
            supporting_views=event.supporting_views,
            confidence=event.confidence,
        )
    ]
    summary = finalize_archive(
        layout,
        manifest,
        {},
        transforms,
        [event],
        [segment],
        [group],
        [event],
        physical,
        [],
        config,
        {"required_bytes": 0, "free_bytes": 0},
        {
            "total_duration_seconds": 0.0,
            "stage_durations": [],
            "tokens": {
                "key_materials": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                "run_total": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            },
            "dry_run": True,
        },
    )
    write_json(
        layout.json_config / "evidence_package_eval.json",
        {
            "passed": True,
            "dry_run": True,
            "checks": [
                {"check": "schema_serialization", "passed": True},
                {"check": "archive_layout", "passed": True},
                {"check": "no_video_or_ffmpeg_required", "passed": True},
            ],
        },
    )
    write_json(
        layout.json_config / "audit_layer.json",
        {
            "events": [event.model_dump(mode="json")],
            "rejected": [],
            "raw_segments": [segment.model_dump(mode="json")],
            "normalized_segments": [segment.model_dump(mode="json")],
            "segments": [segment.model_dump(mode="json")],
            "formal_segment_receipts": [],
            "experiment_groups": [group.model_dump(mode="json")],
            "dry_run": True,
        },
    )
    dry_quality = validate_experiment_and_material_quality(
        [group],
        [event],
        None,
        minimum_cross_view_event_rate=0.0,
    )
    dry_quality["dry_run"] = True
    dry_quality["structural_passed"] = bool(dry_quality.get("passed"))
    dry_quality["evidence_level"] = "synthetic_structural_only"
    dry_quality["formal_accuracy_claim_allowed"] = False
    write_json(layout.json_config / "quality_acceptance.json", dry_quality)
    dry_metrics = {
        "total_duration_seconds": 0.0,
        "preprocessing_sla": {"actual_seconds": 0.0, "target_seconds": 1200.0, "met": True},
        "stage_durations": [],
        "tokens": {
            "experiment_groups": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "call_count": 0},
            "key_materials": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "call_count": 0},
            "daily_report": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "call_count": 0},
            "run_total": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        },
        "dry_run": True,
    }
    write_json(layout.json_config / "run_metrics.json", dry_metrics)
    generate_daily_report_archive(layout, summary, dry_metrics, config)
    write_run_provenance(
        layout.root,
        config,
        {"required": False, "status": "not_required_by_profile"},
        repository_root=Path(__file__).resolve().parents[2],
    )
    write_archive_contract_manifest(layout.root)
    validate_archive_contracts_or_raise(layout.root)
    write_json(layout.root / "run_status.json", {"stage": "completed", "progress": 1.0, "dry_run": True})
    return layout.root
