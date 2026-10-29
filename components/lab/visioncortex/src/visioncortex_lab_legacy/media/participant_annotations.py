"""Curated participant annotation rendering with explicit visual/model ports."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2

from ..evidence.layout import ArchiveLayout
from ..material_naming import key_material_action_folder
from ..schemas import EvidenceEvent, ExperimentGroup


@dataclass(frozen=True)
class ParticipantAnnotationsServices:
    ParticipantVisualReviewer: Callable[..., Any]
    SelectiveVerificationBudget: Callable[..., Any]
    _event_participant_boxes: Callable[..., Any]
    _grounding_dino_key_frame_detections: Callable[..., Any]
    _key_material_view_pair: Callable[..., Any]
    _open_vocabulary_key_frame_supplement: Callable[..., Any]
    _park_auxiliary_model_caches: Callable[..., Any]
    _view_specific_participant_objects: Callable[..., Any]
    _write_aligned_frame: Callable[..., Any]
    analyze_liquid_semantics: Callable[..., Any]
    archive_relative_posix: Callable[..., Any]
    audit_participant_continuity: Callable[..., Any]
    plan_selective_key_material_verification: Callable[..., Any]
    write_annotated_frame: Callable[..., Any]
    write_json: Callable[..., Any]


def _rerender_curated_participant_annotations(
    layout: ArchiveLayout,
    events: Sequence[EvidenceEvent],
    groups: Sequence[ExperimentGroup],
    config: dict[str, Any] | None = None,
    *,
    services: ParticipantAnnotationsServices,
) -> dict[str, Any]:
    """Render final boxes after semantic relabeling corrected participants."""

    verification_started = time.perf_counter()
    verification_settings = dict(
        ((config or {}).get("key_materials") or {}).get("selective_verification") or {}
    )
    verification_budget = services.SelectiveVerificationBudget.from_settings(
        verification_settings
    )
    group_by_event = {
        event_id: group for group in groups for event_id in group.key_event_ids
    }
    records: list[dict[str, Any]] = []
    segmentation_records: list[dict[str, Any]] = []
    liquid_semantic_records: list[dict[str, Any]] = []
    visual_reviewer = (
        services.ParticipantVisualReviewer(
            config,
            layout.work,
            layout.json_config,
            services._grounding_dino_key_frame_detections,
        )
        if config is not None
        and config.get("key_materials", {})
        .get("participant_visual_review", {})
        .get("enabled")
        else None
    )
    for event in events:
        input_root = layout.work / "key-material-annotation-inputs" / event.event_id
        if not input_root.is_dir():
            continue
        group = group_by_event[event.event_id]
        first_material_view, third_material_view = services._key_material_view_pair(
            group, event
        )
        visual_plan: dict[str, dict[str, Any]] = {}
        visual_receipts: list[dict[str, Any]] = []
        review_classes = (
            visual_reviewer.eligible_classes(event)
            if visual_reviewer is not None
            else []
        )
        if review_classes:
            if config.get("performance", {}).get(
                "release_auxiliary_models_between_stages"
            ):
                services._park_auxiliary_model_caches(config)
            for participant_class in review_classes:
                visual_views = []
                for role_label, view_id in (
                    ("First-Person", first_material_view),
                    ("Third-Person", third_material_view),
                ):
                    if (
                        participant_class
                        not in services._view_specific_participant_objects(
                            event, view_id
                        )
                    ):
                        continue
                    visual_views.append(
                        {
                            "view_id": view_id,
                            "role_label": role_label,
                            "raw_path": input_root / f"{role_label}.jpg",
                            "detections": json.loads(
                                (input_root / f"{role_label}.json").read_text()
                            )["detections"],
                        }
                    )
                if not visual_views:
                    continue
                try:
                    visual_receipt, class_plan = visual_reviewer.review(
                        event, visual_views, participant_class=participant_class
                    )
                except RuntimeError as exc:
                    message = str(exc)
                    if not message.startswith(
                        (
                            "Participant visual review ",
                            "Invalid cached participant review",
                        )
                    ):
                        raise
                    review_records = [
                        row
                        for row in (
                            event.observability.get("participant_visual_review") or {}
                        ).values()
                        if isinstance(row, dict)
                        and row.get("participant_class") == participant_class
                    ]
                    if not review_records:
                        raise
                    visual_receipt = review_records[-1]
                    fingerprint = str(
                        visual_receipt.get("input_fingerprint") or "unrecorded"
                    )
                    class_plan = {}
                    for view in visual_views:
                        actors = [
                            dict(box)
                            for box in view["detections"]
                            if str(box.get("class_name") or "")
                            .strip()
                            .lower()
                            .replace("-", "_")
                            .replace(" ", "_")
                            in {"hand", "gloved_hand"}
                        ]
                        class_plan[view["view_id"]] = {
                            "boxes": [],
                            "actors": actors,
                            "selected_candidate_ids": [],
                            "target_visible": False,
                            "reason": (
                                "参与对象视觉复核未产生可验证选择；该对象从已确认"
                                "画面中移除并交由待复核素材处理。"
                            ),
                            "input_fingerprint": fingerprint,
                            "review_status": visual_receipt.get("status"),
                            "failure_reason": message,
                        }
                    visual_receipt = {
                        **visual_receipt,
                        "input_fingerprint": fingerprint,
                        "status": "review_failed_quarantined",
                        "source_status": visual_receipt.get("status"),
                        "failure_reason": message,
                    }
                    note = (
                        f"{participant_class} 参与对象视觉复核未完成；"
                        "相关候选已从已确认素材中移除并保留待复核记录。"
                    )
                    if note not in event.uncertainty:
                        event.uncertainty.append(note)
                visual_receipts.append(
                    {
                        "input_fingerprint": visual_receipt["input_fingerprint"],
                        "status": visual_receipt["status"],
                        "participant_class": participant_class,
                        "localized_view_count": sum(
                            bool(item["boxes"]) for item in class_plan.values()
                        ),
                        **(
                            {
                                "source_status": visual_receipt.get("source_status"),
                                "failure_reason": visual_receipt.get("failure_reason"),
                            }
                            if visual_receipt["status"] == "review_failed_quarantined"
                            else {}
                        ),
                    }
                )
                for view_id, selected in class_plan.items():
                    plan = visual_plan.setdefault(
                        view_id,
                        {
                            "boxes": [],
                            "actors": selected["actors"],
                            "input_fingerprint": selected["input_fingerprint"],
                            "input_fingerprints": [],
                            "reviewed_classes": [],
                            "reviews": [],
                        },
                    )
                    plan["boxes"].extend(selected["boxes"])
                    plan["input_fingerprints"].append(selected["input_fingerprint"])
                    plan["reviewed_classes"].append(participant_class)
                    plan["reviews"].append(
                        {
                            "participant_class": participant_class,
                            **{
                                key: value
                                for key, value in selected.items()
                                if key not in {"boxes", "actors"}
                            },
                        }
                    )
        rendered_frames: dict[str, Path] = {}
        annotation = {
            "schema_version": "visioncortex-key-material-annotation/1",
            "mode": "event_participants_only",
            "render_pass": "post_semantic_curation",
            "views": {},
        }
        if visual_receipts:
            failed_review_count = sum(
                item["status"] == "review_failed_quarantined"
                for item in visual_receipts
            )
            annotation["participant_visual_review"] = {
                "input_fingerprint": visual_receipts[0]["input_fingerprint"],
                "status": (
                    "completed_with_quarantined_review_failures"
                    if failed_review_count
                    else "completed"
                ),
                "participant_class": visual_receipts[0]["participant_class"]
                if len(visual_receipts) == 1
                else "multiple",
                "reviews": visual_receipts,
                "localized_view_count": sum(
                    bool(item["boxes"]) for item in visual_plan.values()
                ),
                "quarantined_review_failure_count": failed_review_count,
            }
        for role_label, view_id in (
            ("First-Person", first_material_view),
            ("Third-Person", third_material_view),
        ):
            raw_path = input_root / f"{role_label}.jpg"
            detection_path = input_root / f"{role_label}.json"
            if not raw_path.is_file() or not detection_path.is_file():
                raise RuntimeError(
                    f"Final annotation input is incomplete: {event.event_id}/{role_label}"
                )
            raw_frame = cv2.imread(str(raw_path))
            if raw_frame is None:
                raise RuntimeError(
                    f"Final annotation raw frame is unreadable: {raw_path}"
                )
            detected_boxes = json.loads(detection_path.read_text(encoding="utf-8"))[
                "detections"
            ]
            view_event = event.model_copy(deep=True)
            view_event.objects = services._view_specific_participant_objects(
                event, view_id
            )
            maximum_interaction_gap_norm = float(
                (
                    (config or {})
                    .get("models", {})
                    .get("open_vocabulary_key_frame", {})
                    .get("manipulated_object_max_actor_gap_norm", 0.08)
                )
            )
            _, closed_set_participant_receipt = services._event_participant_boxes(
                view_event,
                detected_boxes,
                view_id=view_id,
                maximum_interaction_gap_norm=maximum_interaction_gap_norm,
            )
            verification_decision = services.plan_selective_key_material_verification(
                view_event,
                view_id,
                detected_boxes,
                closed_set_participant_receipt,
                verification_settings,
                verification_budget,
            )
            supplement_receipt: dict[str, Any] | None = None
            phase_isolation = bool(
                (config or {})
                .get("performance", {})
                .get("release_auxiliary_models_between_stages")
            )
            if phase_isolation:
                services._park_auxiliary_model_caches(config)
            if view_id in visual_plan and set(view_event.objects) <= {
                "hand",
                "gloved_hand",
                *visual_plan[view_id]["reviewed_classes"],
            }:
                supplement_receipt = {
                    "status": "replaced_by_visual_candidate_review",
                    "input_fingerprint": visual_plan[view_id]["input_fingerprint"],
                    "input_fingerprints": visual_plan[view_id]["input_fingerprints"],
                    "purpose": "Use the reviewed DINO proposal without redundant YOLO-World inference",
                }
            elif config is not None and verification_decision["should_run"]:
                supplement_boxes, supplement_receipt = (
                    services._open_vocabulary_key_frame_supplement(
                        raw_frame,
                        view_event,
                        detected_boxes,
                        config,
                        view_role=role_label,
                    )
                )
                if supplement_receipt.get("status") == "executed":
                    # The supplement is a fail-closed replacement for ambiguous
                    # closed-set instances, not an additive source. Remove the
                    # named classes even when grounding finds no eligible
                    # hand-adjacent object; the visibility gate must then fail
                    # instead of silently drawing a background rack instance.
                    replaced_classes = set(
                        supplement_receipt.get("closed_set_replaced_classes") or []
                    )
                    detected_boxes = [
                        box
                        for box in detected_boxes
                        if str(box.get("class_name") or "")
                        .strip()
                        .lower()
                        .replace("-", "_")
                        .replace(" ", "_")
                        not in replaced_classes
                    ]
                    detected_boxes = [*detected_boxes, *supplement_boxes]
            elif config is not None:
                supplement_receipt = {
                    "schema_version": "visioncortex-open-vocabulary-key-frame/1",
                    "status": "skipped_by_selective_verification",
                    "scope": "final accepted key frames only",
                    "full_timeline_inference": False,
                    "source_copy_bytes": 0,
                    "token_usage": 0,
                    "ark_calls": 0,
                }
            if view_id in visual_plan:
                selected = visual_plan[view_id]
                # A validated empty selection also replaces the old target box;
                # SAM2 must never propagate a rejected background instance.
                detected_boxes = [
                    box
                    for box in detected_boxes
                    if box.get("class_name") not in selected["reviewed_classes"]
                ]
                detected_boxes.extend(selected["boxes"])
                if "bottle_cap" in selected["reviewed_classes"] or not any(
                    box.get("class_name") in {"hand", "gloved_hand"}
                    for box in detected_boxes
                ):
                    detected_boxes.extend(
                        actor
                        for actor in selected["actors"]
                        if actor not in detected_boxes
                    )
            boxes, receipt = services._event_participant_boxes(
                view_event,
                detected_boxes,
                view_id=view_id,
                maximum_interaction_gap_norm=maximum_interaction_gap_norm,
            )
            receipt["selective_verification"] = verification_decision
            if view_id in visual_plan:
                receipt["participant_visual_review"] = {
                    key: value
                    for key, value in visual_plan[view_id].items()
                    if key not in {"boxes", "actors"}
                }
            if supplement_receipt is not None:
                receipt["open_vocabulary_supplement"] = supplement_receipt
            if verification_decision["status"] == "deferred_budget_exhausted":
                deferred_note = (
                    "关键素材本地二次复核预算已用尽；保留闭集检测证据，"
                    f"未执行开放词汇补全（{verification_decision['reason']}）"
                )
                if deferred_note not in event.uncertainty:
                    event.uncertainty.append(deferred_note)
            if phase_isolation:
                services._park_auxiliary_model_caches(config)
            segmentation_receipt: dict[str, Any] | None = None
            if config is not None:
                segmentation_settings = (config.get("models") or {}).get(
                    "temporal_participant_segmentation"
                ) or {}
                if segmentation_settings.get("enabled"):
                    relative_clip = event.key_clips.get(view_id)
                    if not relative_clip:
                        if segmentation_settings.get(
                            "required_for_final_key_material", False
                        ):
                            raise RuntimeError(
                                "SAM2 final participant audit requires a bounded "
                                f"key clip: {event.event_id}/{view_id}"
                            )
                        segmentation_receipt = {"status": "not_available_no_key_clip"}
                    else:
                        clip_pre_ms = (
                            float(config["segmentation"]["key_clip_pre_seconds"])
                            * 1000.0
                        )
                        clip_post_ms = (
                            float(config["segmentation"]["key_clip_post_seconds"])
                            * 1000.0
                        )
                        clip_start_ms = max(0.0, event.global_start_ms - clip_pre_ms)
                        clip_end_ms = event.global_end_ms + clip_post_ms
                        seed_fraction = (event.key_global_ms - clip_start_ms) / max(
                            1.0, clip_end_ms - clip_start_ms
                        )
                        boxes, segmentation_receipt = (
                            services.audit_participant_continuity(
                                layout.root / relative_clip,
                                raw_frame,
                                boxes,
                                layout.work / "sam2-participant-continuity",
                                config,
                                event_id=event.event_id,
                                view_id=view_id,
                                action_type=event.action_type.value,
                                seed_fraction=seed_fraction,
                            )
                        )
                    receipt["temporal_participant_segmentation"] = segmentation_receipt
                    segmentation_records.append(
                        {
                            "event_id": event.event_id,
                            "view_id": view_id,
                            "role_label": role_label,
                            **dict(segmentation_receipt or {}),
                        }
                    )
                liquid_settings = (config.get("models") or {}).get(
                    "liquid_semantic_sidecar"
                ) or {}
                enabled_actions = {
                    str(item) for item in liquid_settings.get("enabled_actions") or []
                }
                if liquid_settings.get("enabled") and (
                    not enabled_actions or event.action_type.value in enabled_actions
                ):
                    if phase_isolation:
                        services._park_auxiliary_model_caches(config)
                    liquid_output = (
                        layout.key_materials
                        / "Liquid-State-Observations"
                        / str(group.archive_folder or group.group_id)
                        / key_material_action_folder(event.action_type)
                        / event.event_id
                    )
                    liquid_receipt = services.analyze_liquid_semantics(
                        raw_frame,
                        config,
                        liquid_output,
                        artifact_stem=view_id,
                    )
                    for path_key in ("overlay_path", "mask_path"):
                        liquid_receipt[path_key] = services.archive_relative_posix(
                            Path(str(liquid_receipt[path_key])), layout.root
                        )
                    receipt["liquid_semantic_sidecar"] = liquid_receipt
                    event.observability.setdefault("liquid_semantic_sidecar", {})[
                        view_id
                    ] = liquid_receipt
                    liquid_semantic_records.append(
                        {
                            "event_id": event.event_id,
                            "view_id": view_id,
                            "role_label": role_label,
                            **liquid_receipt,
                        }
                    )
            destination = layout.root / event.key_frames[view_id]
            services.write_annotated_frame(raw_frame, boxes, destination)
            receipt["rendered_classes"] = sorted({box["class_name"] for box in boxes})
            receipt["rendered_box_count"] = len(boxes)
            rendered_frames[role_label] = destination
            annotation["views"][view_id] = receipt
            records.append(
                {
                    "event_id": event.event_id,
                    "view_id": view_id,
                    "role_label": role_label,
                    "participant_objects": list(view_event.objects),
                    "event_participant_objects": list(event.objects),
                    **receipt,
                }
            )
        aligned = layout.root / event.key_frames["aligned_first_third"]
        services._write_aligned_frame(
            rendered_frames["First-Person"],
            rendered_frames["Third-Person"],
            aligned,
            (first_material_view, third_material_view),
        )
        if visual_receipts:
            for review in visual_receipts:
                review["reviewed_localized_view_count"] = review["localized_view_count"]
                review["localized_view_count"] = sum(
                    review["participant_class"] in item.get("rendered_classes", [])
                    for item in annotation["views"].values()
                )
            annotation["participant_visual_review"]["localized_view_count"] = sum(
                bool(set(item.get("rendered_classes", [])) & set(review_classes))
                for item in annotation["views"].values()
            )
        event.observability["key_material_annotation"] = annotation
        if bool(
            ((config or {}).get("performance") or {}).get(
                "release_auxiliary_models_after_event", False
            )
        ):
            services._park_auxiliary_model_caches(config)
    decisions = [
        record["selective_verification"]
        for record in records
        if isinstance(record.get("selective_verification"), dict)
    ]
    decision_status_counts: dict[str, int] = {}
    for decision in decisions:
        status = str(decision.get("status") or "unknown")
        decision_status_counts[status] = decision_status_counts.get(status, 0) + 1
    supplements = [
        record["open_vocabulary_supplement"]
        for record in records
        if isinstance(record.get("open_vocabulary_supplement"), dict)
    ]
    grounding_receipts = [
        supplement["grounding_dino_fallback"]
        for supplement in supplements
        if isinstance(supplement.get("grounding_dino_fallback"), dict)
    ]
    verification_summary = {
        "schema_version": "visioncortex-selective-key-material-verification-index/1",
        "enabled": bool(verification_settings.get("enabled", False)),
        "mode": str(verification_settings.get("mode") or "ambiguous_or_high_risk"),
        "policy": (
            "closed-set evidence for every accepted event; expensive local models "
            "only for high-risk or ambiguous final role frames"
        ),
        "decision_count": len(decisions),
        "decision_status_counts": dict(sorted(decision_status_counts.items())),
        "open_vocabulary_executed_count": sum(
            item.get("status") == "executed" for item in supplements
        ),
        "grounding_dino_executed_count": sum(
            item.get("status") == "executed" for item in grounding_receipts
        ),
        "open_vocabulary_model_load_seconds": round(
            sum(float(item.get("model_load_seconds") or 0.0) for item in supplements),
            6,
        ),
        "open_vocabulary_inference_seconds": round(
            sum(float(item.get("inference_seconds") or 0.0) for item in supplements),
            6,
        ),
        "grounding_dino_model_load_seconds": round(
            sum(
                float(item.get("model_load_seconds") or 0.0)
                for item in grounding_receipts
            ),
            6,
        ),
        "grounding_dino_inference_seconds": round(
            sum(
                float(item.get("inference_seconds") or 0.0)
                for item in grounding_receipts
            ),
            6,
        ),
        "wall_seconds": round(time.perf_counter() - verification_started, 6),
        "budget": verification_budget.summary(),
        "full_timeline_inference": False,
        "source_copy_bytes": 0,
        "ark_calls": 0,
        "token_usage": 0,
    }
    report = {
        "schema_version": "visioncortex-final-key-material-annotation/1",
        "mode": "event_participants_only",
        "render_pass": "post_semantic_curation",
        "event_count": len(events),
        "rendered_view_count": len(records),
        "selective_verification": verification_summary,
        "participant_visual_review": {
            "enabled": bool(visual_reviewer is not None and visual_reviewer.enabled),
            "requests": visual_reviewer.records if visual_reviewer is not None else {},
            "purpose": "participant localization only; not action or quality certification",
        },
        "records": records,
    }
    services.write_json(
        layout.json_config / "final_key_material_annotation.json", report
    )
    services.write_json(
        layout.json_config / "temporal_participant_segmentation.json",
        {
            "schema_version": "visioncortex-sam2-participant-continuity-index/1",
            "policy": (
                "bounded final key clips only; continuity and box refinement, "
                "never action confirmation"
            ),
            "record_count": len(segmentation_records),
            "passed_count": sum(
                item.get("status") == "completed" and item.get("passed") is True
                for item in segmentation_records
            ),
            "failed_count": sum(
                item.get("status") == "completed" and item.get("passed") is False
                for item in segmentation_records
            ),
            "records": segmentation_records,
        },
    )
    services.write_json(
        layout.json_config / "liquid_semantic_observations.json",
        {
            "schema_version": "visioncortex-labpics-liquid-semantic-index/1",
            "policy": "bounded final key frames; observation only; never action confirmation",
            "record_count": len(liquid_semantic_records),
            "records": liquid_semantic_records,
        },
    )
    return report
