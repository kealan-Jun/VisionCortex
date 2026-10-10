"""Semantic adjudication and curated participant/material reconciliation."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from ..action_semantics import DEVICE_CLASSES
from ..material_naming import key_material_action_folder
from ..schemas import (
    ActionType,
    EvidenceEvent,
    ExperimentGroup,
    ViewRole,
    set_event_admission,
)
from .layout import ArchiveLayout

RELABEL_OBJECT_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("tube_cap", (r"管盖", r"离心管盖", r"tube[_ -]?cap")),
    (
        "bottle_cap",
        (
            r"瓶盖",
            r"红盖",
            r"红色小盖(?:容器)?",
            r"bottle[_ -]?cap",
            r"(?:blue|red)[_ -]?cap",
        ),
    ),
    (
        "paper",
        (
            r"称量纸",
            r"白色纸片",
            r"纸片",
            r"纸张",
            r"持纸",
            r"^纸$",
            r"weighing[_ -]?paper",
        ),
    ),
    ("spatula", (r"药匙", r"药勺", r"勺状(?:金属)?工具", r"spatula")),
    ("pipette", (r"移液器", r"移液枪", r"pipette")),
    ("spearhead", (r"枪头", r"吸头", r"pipette[_ -]?tip", r"spearhead")),
    (
        "reagent_bottle",
        (
            r"试剂瓶",
            r"棕色瓶",
            r"透明(?:玻璃)?(?:试剂)?瓶",
            r"reagent[_ -]?bottle",
            r"(?:brown|amber)[_ -]?(?:reagent[_ -]?)?bottle",
            r"(?:clear|transparent|glass)[_ -]?(?:reagent[_ -]?)?bottle",
        ),
    ),
    ("sample_bottle", (r"样品瓶", r"sample[_ -]?bottle")),
    ("tube_rack", (r"离心管架", r"试管架", r"tube[_ -]?rack")),
    (
        "tube",
        (
            r"离心管",
            r"试管",
            r"\btube\b",
            r"(?:centrifuge[_ -]?)?tube(?:[_ /-]|$)",
        ),
    ),
    ("balance", (r"天平", r"balance")),
    ("beaker", (r"烧杯", r"beaker")),
    ("magnetic_stirrer", (r"磁力搅拌", r"magnetic[_ -]?stirrer")),
    ("container", (r"容器", r"container")),
)


@dataclass(frozen=True)
class SemanticCurationServices:
    """Named execution ports; supplied explicitly at the composition boundary."""

    _annotation_supports_curated_action: Callable[..., Any]
    _interaction_participant_classes: Callable[..., Any]
    _key_material_event_folder_name: Callable[..., Any]
    _move_curated_event_media: Callable[..., Any]
    _participant_class_mentions: Callable[..., Any]
    _relabel_participant_objects: Callable[..., Any]
    _relative: Callable[..., Any]
    _rerender_curated_participant_annotations: Callable[..., Any]
    _safe_folder_name: Callable[..., Any]
    _semantic_interaction_is_direct: Callable[..., Any]
    _semantic_participant_conflicts: Callable[..., Any]
    action_participant_visibility: Callable[..., Any]
    build_event_state_receipt: Callable[..., Any]
    event_stable_identities: Callable[..., Any]
    normalize_participant_class: Callable[..., Any]
    semantic_action_proof_contradictions: Callable[..., Any]
    write_json: Callable[..., Any]
    write_key_material_category_index: Callable[..., Any]


def _semantic_interaction_is_direct(item: dict[str, Any]) -> bool:
    """Accept only structured interactions that assert physical contact."""

    contact = str(item.get("contact") or "").strip().lower()
    if not contact:
        return False
    negative_terms = (
        "未",
        "接近",
        "靠近",
        "不清晰",
        "无法确认",
        "no contact",
        "not clear",
        "unclear",
        "near",
        "approach",
    )
    if any(term in contact for term in negative_terms):
        return False
    positive_terms = (
        "接触",
        "抓取",
        "握持",
        "按压",
        "触碰",
        "拿取",
        "操作",
        "contact",
        "touch",
        "grasp",
        "grip",
        "hold",
        "press",
    )
    return any(term in contact for term in positive_terms)


def _participant_class_mentions(
    text: str, *, services: SemanticCurationServices
) -> list[tuple[str, int, int]]:
    """Recognize canonical structured labels as well as natural-language aliases."""

    mentions: list[tuple[str, int, int]] = []
    for class_name, patterns in RELABEL_OBJECT_PATTERNS:
        canonical_pattern = rf"(?<!\w){re.escape(class_name)}(?!\w)"
        for pattern in (canonical_pattern, *patterns):
            mentions.extend(
                (class_name, match.start(), match.end())
                for match in re.finditer(pattern, text.lower())
            )
    return mentions


def _semantic_participant_conflicts(
    event: EvidenceEvent, *, services: SemanticCurationServices
) -> list[dict[str, str]]:
    """Record explicit background-only claims contradicting a contact record.

    Use the model's aggregate action proof, not one occluded view or an absent
    device state change. The result marks an internal contradiction; it does
    not establish that physical contact never occurred in the source video.
    """

    understanding = event.model_understanding or {}
    direct_classes = {
        class_name
        for item in understanding.get("hand_object_interactions") or []
        if isinstance(item, dict) and services._semantic_interaction_is_direct(item)
        for class_name, _start, _end in services._participant_class_mentions(
            str(item.get("object") or "")
        )
    }
    reason = str((understanding.get("action_proof") or {}).get("reason") or "")
    conflicts: list[dict[str, str]] = []
    for clause in re.split(r"[，,。；;.!?\n]", reason):
        background = re.search(
            r"(?:仅|只)(?:作为|是|在|出现在)?[^，,。；;\n]{0,6}背景"
            r"|\bonly\s+(?:(?:as|in)\s+(?:a\s+|the\s+)?)?background\b",
            clause.lower(),
        )
        if background is None:
            continue
        prefix = clause[: background.start()].lower()
        if re.search(r"(?:并非|不是|不仅|不只是|不|not)\s*$", prefix):
            continue
        mentions = [
            (class_name, end)
            for class_name, _start, end in services._participant_class_mentions(prefix)
            if len(prefix) - end <= 24
        ]
        if not mentions:
            continue
        nearest_end = max(end for _class_name, end in mentions)
        for class_name in sorted(
            {name for name, end in mentions if end == nearest_end}
        ):
            if class_name in direct_classes:
                conflicts.append(
                    {
                        "class_name": class_name,
                        "source": "action_proof.reason",
                        "statement": clause.strip(),
                        "reason": "direct_contact_and_background_only_claims_conflict",
                    }
                )
    return conflicts


def _interaction_participant_classes(
    text: str, *, services: SemanticCurationServices
) -> list[str]:
    classes = list(
        dict.fromkeys(
            name
            for name, _start, _end in services._participant_class_mentions(text.lower())
        )
    )
    # A cap used to describe a bottle is not a separately manipulated cap.
    # Match a whole, single noun phrase; explicit lists retain both objects.
    capped_bottle = bool(
        re.fullmatch(
            r"(?:带(?:有)?|装有|配有|盖有)?[^/，,。；;和与及、]{0,12}"
            r"(?:瓶盖|盖子|盖)的(?:棕色|玻璃|塑料|透明)?"
            r"(?:试剂瓶|样品瓶|棕色瓶子|瓶子|瓶)",
            text.strip(),
        )
    )
    if capped_bottle and set(classes) & {
        "reagent_bottle",
        "sample_bottle",
        "sample_bottle_blue",
        "container",
    }:
        classes = [name for name in classes if name != "bottle_cap"]
    return classes


def _relabel_participant_objects(
    event: EvidenceEvent, *, services: SemanticCurationServices
) -> list[str]:
    """Map model-described interaction participants back to detector classes."""

    understanding = event.model_understanding or {}
    structured_interactions = [
        item
        for item in (understanding.get("hand_object_interactions") or [])
        if isinstance(item, dict) and str(item.get("object") or "").strip()
    ]
    direct_interactions = [
        item
        for item in structured_interactions
        if services._semantic_interaction_is_direct(item)
    ]
    interaction_objects = [
        str(item.get("object") or "") for item in direct_interactions
    ]
    primary_text = " ".join(interaction_objects).lower()
    fallback_text = " ".join(
        str(understanding.get(key) or "") for key in ("current_step",)
    ).lower()
    interaction_matches = list(
        dict.fromkeys(
            name
            for text in interaction_objects
            for name in services._interaction_participant_classes(text)
        )
    )
    selected_observations = [
        item
        for item in understanding.get("selected_keyframe_observations") or []
        if isinstance(item, dict)
    ]
    selected_interaction_labels = [
        str(label)
        for item in selected_observations
        for label in item.get("directly_interacting_objects") or []
        if str(label).strip()
    ]
    selected_interaction_matches = list(
        dict.fromkeys(
            name
            for text in selected_interaction_labels
            for name in services._interaction_participant_classes(text)
            if name not in {"hand", "gloved_hand"}
        )
    )
    if event.action_type == ActionType.DEVICE_PANEL_OPERATION:
        selected_interaction_matches = [
            item for item in selected_interaction_matches if item in DEVICE_CLASSES
        ]
    elif event.action_type == ActionType.PIPETTE_TRANSFER_OPERATION:
        selected_interaction_matches = [
            item
            for item in selected_interaction_matches
            if item in {"pipette", "spearhead"}
        ]
    step_matches = list(
        dict.fromkeys(
            name
            for name, _start, _end in services._participant_class_mentions(
                fallback_text
            )
        )
    )
    # Structured hand-object interactions are already participant-scoped.
    # Do not drop one explicit participant merely because the free-text step
    # repeats another participant but omits this one's class name.
    matched = interaction_matches if structured_interactions else step_matches
    if selected_observations:
        # Final annotations explain the selected key frame, so the explicit
        # selected-frame interaction list outranks objects touched elsewhere
        # in a long event window.  An unmapped selected-frame object remains
        # empty and is quarantined downstream; a convenient mapped background
        # object must never substitute for it.
        matched = selected_interaction_matches
    # Ark may use a deliberately generic structured label such as
    # ``red small container`` while the participant-scoped current-step text
    # identifies the same object as a cap.  Refine only a generic container
    # through specific container/closure aliases from that current-step text;
    # never pull tools or unrelated background inventory into the participant
    # list.  This keeps the final detector from boxing a nearby bottle or rack
    # merely because the semantic participant was left as ``container``.
    if structured_interactions and "container" in matched:
        container_refinement_classes = {
            "tube_cap",
            "bottle_cap",
            "reagent_bottle",
            "sample_bottle",
            "tube",
            "beaker",
        }
        refinements = [
            item for item in step_matches if item in container_refinement_classes
        ]
        if refinements:
            matched = [item for item in matched if item != "container"]
            matched = list(dict.fromkeys([*matched, *refinements]))
    proof = understanding.get("action_proof") or {}
    if (
        event.action_type == ActionType.OBJECT_MOVEMENT
        and proof.get("source_contact_visible")
        and proof.get("withdrawal_or_transport_visible")
        and proof.get("target_contact_visible")
    ):
        # Long solid-transfer windows can mention incidental cap or package
        # touches. Keep only classes named by the proof of the confirmed
        # source-to-target movement. If the proof cannot be mapped, retain the
        # structured interactions and fail closed downstream.
        proof_classes = services._interaction_participant_classes(
            str(proof.get("reason") or "")
        )
        if proof_classes:
            matched = [item for item in matched if item in proof_classes]
        paper_interactions = [
            text
            for text in interaction_objects
            if "paper" in services._interaction_participant_classes(text)
        ]
        package_pattern = r"(?:package|packaging|packet|wrapper|包装|纸包)"
        if (
            "paper" in matched
            and paper_interactions
            and all(
                re.search(package_pattern, text.lower()) for text in paper_interactions
            )
            and not re.search(package_pattern, str(proof.get("reason") or "").lower())
        ):
            # A transfer window may contain a direct touch of the weighing-
            # paper package while the confirmed movement ends on a separate
            # sheet. Both phrases map to ``paper`` but they are different
            # physical instances. The receiving sheet is not a hand-object
            # participant unless a direct interaction names the sheet itself.
            matched = [item for item in matched if item != "paper"]
    conflicting_classes = {
        item["class_name"] for item in services._semantic_participant_conflicts(event)
    }
    matched = [item for item in matched if item not in conflicting_classes]
    actor_classes = [
        item
        for item in event.objects
        if str(item).strip().lower().replace("-", "_") in {"hand", "gloved_hand"}
    ]
    # ``hand`` is the generic superclass of ``gloved_hand``. Keeping both as
    # separate semantic slots makes open-vocabulary grounding request a bare
    # hand even when Ark explicitly observed blue gloves; a weak background
    # false positive can then suppress the real hand-held tool. Prefer the
    # more specific gloved actor whenever it is present.
    if "gloved_hand" in actor_classes:
        actor_classes = ["gloved_hand"]
    if not actor_classes and interaction_objects:
        actor_text = " ".join(
            [
                primary_text,
                fallback_text,
                " ".join(map(str, understanding.get("objects") or [])),
            ]
        ).lower()
        actor_classes = [
            "gloved_hand" if re.search(r"手套|glov(?:e|ed)", actor_text) else "hand"
        ]
    return list(dict.fromkeys([*actor_classes, *matched]))


def _view_specific_participant_objects(
    event: EvidenceEvent, view_id: str, *, services: SemanticCurationServices
) -> list[str]:
    """Constrain final boxes to objects Ark actually described in one view."""

    actor_objects = [
        item
        for item in event.objects
        if services.normalize_participant_class(item) in {"hand", "gloved_hand"}
    ]
    non_actor_objects = [
        item
        for item in event.objects
        if services.normalize_participant_class(item) not in {"hand", "gloved_hand"}
    ]
    understanding = event.model_understanding or {}
    observation = next(
        (
            str(item.get("observation") or "")
            for item in understanding.get("per_view_observations") or []
            if isinstance(item, dict) and str(item.get("view_id") or "") == str(view_id)
        ),
        "",
    )
    support = next(
        (
            item
            for item in understanding.get("confirmed_action_support_by_view") or []
            if isinstance(item, dict) and str(item.get("view_id") or "") == str(view_id)
        ),
        None,
    )
    confirmed_support = [
        item
        for item in understanding.get("confirmed_action_support_by_view") or []
        if isinstance(item, dict)
    ]
    if support is not None and support.get("supports_confirmed_action") is False:
        return list(dict.fromkeys(actor_objects))
    if support is None and confirmed_support:
        # A populated per-view support ledger is authoritative.  If this view
        # has no entry, do not propagate objects proved only by another view.
        return list(dict.fromkeys(actor_objects))
    support_reason = str((support or {}).get("reason") or "")
    # Prefer the participant-scoped support reason.  Full observations often
    # contain explicit negative statements (for example "未看到接触移液器");
    # naïve keyword matching would turn that absence into a rendered object.
    scoped_text = (support_reason or observation).lower()
    pattern_map = dict(RELABEL_OBJECT_PATTERNS)
    canonical_alias = {
        "weighing_paper": "paper",
        "sample_bottle_blue": "sample_bottle",
    }
    matched_non_actors: list[str] = []
    for item in non_actor_objects:
        canonical = canonical_alias.get(
            services.normalize_participant_class(item),
            services.normalize_participant_class(item),
        )
        patterns = pattern_map.get(canonical, ())
        if re.search(
            rf"(?<![a-z0-9_]){re.escape(canonical)}(?![a-z0-9_])", scoped_text
        ) or any(re.search(pattern, scoped_text) for pattern in patterns):
            matched_non_actors.append(item)
    if matched_non_actors:
        return list(dict.fromkeys([*actor_objects, *matched_non_actors]))
    if (
        observation
        and str(
            (event.semantic_review or {}).get("cross_view_consistency")
            or understanding.get("cross_view_consistency")
            or ""
        )
        == "conflict"
    ):
        # The view explicitly depicts another concurrent object. Do not carry
        # the primary view's class into it merely to fill an annotation slot.
        return list(dict.fromkeys(actor_objects))
    return list(event.objects)


def _move_curated_event_media(
    layout: ArchiveLayout,
    event: EvidenceEvent,
    group: ExperimentGroup,
    quarantine_root: Path,
    *,
    destination_action: ActionType | None,
    services: SemanticCurationServices,
) -> dict[str, Any]:
    """Move one event atomically between formal and review material trees."""

    moved: list[dict[str, str]] = []
    retained: list[dict[str, str]] = []
    experiment_folder = group.archive_folder or services._safe_folder_name(
        group.group_id
    )
    for attribute, media_root, media_kind in (
        ("key_frames", layout.key_frames, "Key-Frames"),
        ("key_clips", layout.key_clips, "Key-Clips"),
    ):
        collection = dict(getattr(event, attribute))
        if not collection:
            continue
        source_files = [layout.root / relative for relative in collection.values()]
        source_directories = {path.parent.resolve(strict=True) for path in source_files}
        if len(source_directories) != 1:
            raise RuntimeError(
                f"{event.event_id} {attribute} spans multiple event directories"
            )
        source_directory = source_directories.pop()
        if not source_directory.is_relative_to(media_root.resolve(strict=True)):
            raise RuntimeError(
                f"{event.event_id} media escaped the formal key-material root"
            )
        if destination_action is None:
            destination_directory = quarantine_root / event.event_id / media_kind
        else:
            destination_directory = (
                media_root
                / experiment_folder
                / key_material_action_folder(destination_action)
                / services._key_material_event_folder_name(
                    layout, experiment_folder, event
                )
            )
        if destination_directory.exists():
            resolved_destination = destination_directory.resolve(strict=True)
            if resolved_destination == source_directory:
                retained.append(
                    {
                        "media_kind": media_kind,
                        "path": services._relative(source_directory, layout.root),
                        "operation": "retained_in_place",
                    }
                )
                continue
            raise RuntimeError(
                f"Semantic curation destination already exists: {destination_directory}"
            )
        destination_directory.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source_directory), str(destination_directory))
        if destination_action is None:
            setattr(event, attribute, {})
        else:
            setattr(
                event,
                attribute,
                {
                    view_id: services._relative(
                        destination_directory / Path(relative).name,
                        layout.root,
                    )
                    for view_id, relative in collection.items()
                },
            )
        moved.append(
            {
                "media_kind": media_kind,
                "from": services._relative(source_directory, layout.root),
                "to": services._relative(destination_directory, layout.root),
            }
        )
    return {"moved_media": moved, "retained_media": retained}


def curate_semantically_reviewed_key_materials(
    layout: ArchiveLayout,
    events: Sequence[EvidenceEvent],
    groups: Sequence[ExperimentGroup],
    config: dict[str, Any],
    publisher: Any | None = None,
    *,
    services: SemanticCurationServices,
) -> tuple[list[EvidenceEvent], dict[str, Any]]:
    """Keep only semantically confirmed actions in final Key-Materials.

    CV acceptance remains in each semantic-review receipt. Rejected or
    uncertain generated media moves into a machine-quarantine tree, so recall
    disputes remain visually auditable without becoming a manual completion
    gate or being presented as confirmed key material.
    """

    if publisher is not None:
        raise RuntimeError(
            "Semantic key-material curation requires direct staging output; "
            "incremental publication cannot retract rejected candidates safely"
        )
    relabel_min_confidence = float(
        config.get("key_materials", {}).get("semantic_relabel_min_confidence", 0.70)
    )
    known_actions = {item.value: item for item in ActionType}
    group_by_event = {
        event_id: group for group in groups for event_id in group.key_event_ids
    }
    quarantine_root = layout.key_materials / "Machine-Quarantine"
    records: list[dict[str, Any]] = []
    curated: list[EvidenceEvent] = []

    def move_event_media(
        event: EvidenceEvent,
        *,
        destination_action: ActionType | None,
    ) -> dict[str, Any]:
        return services._move_curated_event_media(
            layout,
            event,
            group_by_event[event.event_id],
            quarantine_root,
            destination_action=destination_action,
        )

    for event in events:
        review = event.semantic_review or {}
        original_action = event.action_type
        original_objects = list(event.objects)
        original_supporting_views = list(event.supporting_views)
        original_supporting_roles = [
            role.value if isinstance(role, ViewRole) else str(role)
            for role in event.supporting_roles
        ]
        original_state_machine = dict(event.state_machine or {})
        original_paths = {
            "key_frames": dict(event.key_frames),
            "key_clips": dict(event.key_clips),
        }
        # A deterministic presentation repair may re-run this curation over an
        # already curated package.  Preserve the earliest CV hypothesis and
        # media/state receipts instead of replacing provenance with the final
        # semantic action on the second pass.
        provenance_action_type = str(
            review.get("pre_curation_action_type") or original_action.value
        )
        provenance_objects = list(
            review.get("pre_curation_objects") or original_objects
        )
        provenance_supporting_views = list(
            review.get("pre_curation_supporting_views") or original_supporting_views
        )
        provenance_supporting_roles = list(
            review.get("pre_curation_supporting_roles") or original_supporting_roles
        )
        provenance_state_machine = dict(
            review.get("pre_curation_state_machine") or original_state_machine
        )
        provenance_media = dict(review.get("pre_curation_media") or original_paths)
        if str(review.get("model_status") or "") != "completed":
            movement = move_event_media(event, destination_action=None)
            set_event_admission(event, "rejected")
            disposition = "machine_quarantined_semantic_unavailable"
            review.update(
                {
                    "pre_curation_verdict": (
                        review.get("pre_curation_verdict")
                        or review.get("verdict")
                        or None
                    ),
                    "pre_curation_action_type": provenance_action_type,
                    "pre_curation_objects": provenance_objects,
                    "pre_curation_supporting_views": provenance_supporting_views,
                    "pre_curation_supporting_roles": provenance_supporting_roles,
                    "pre_curation_state_machine": provenance_state_machine,
                    "pre_curation_media": provenance_media,
                    "verdict": "semantic_unavailable",
                    "final_delivery_accepted": False,
                    "final_action_type": None,
                    "final_participant_objects": [],
                    "curation_disposition": disposition,
                    "curation_policy": "semantic-final-key-material-v1",
                    "evidence_classification": "PARTIAL_EVIDENCE",
                    "retryable": True,
                    **movement,
                }
            )
            event.semantic_review = review
            records.append(
                {
                    "event_id": event.event_id,
                    "cv_action_type": provenance_action_type,
                    "cv_objects": provenance_objects,
                    "model_action_type": None,
                    "pre_curation_verdict": review.get("pre_curation_verdict"),
                    "model_confidence": 0.0,
                    "disposition": disposition,
                    "final_action_type": None,
                    "final_participant_objects": [],
                    "semantic_participant_refined": False,
                    "semantic_participant_refinement_changed": False,
                    "final_delivery_accepted": False,
                    "semantic_state_machine_rebuilt": False,
                    "evidence_classification": "PARTIAL_EVIDENCE",
                    "retryable": True,
                    **movement,
                }
            )
            continue
        model_action_value = str(review.get("model_action_type") or "")
        model_confidence = float(review.get("model_confidence") or 0.0)
        review_verdict = str(review.get("verdict") or "")
        raw_model_verdict = str(review.get("model_evidence_verdict") or "")
        participant_conflicts = services._semantic_participant_conflicts(event)
        relabel_objects = services._relabel_participant_objects(event)
        relabel_non_actor_objects = {
            item for item in relabel_objects if item not in {"hand", "gloved_hand"}
        }
        original_non_actor_objects = {
            item for item in original_objects if item not in {"hand", "gloved_hand"}
        }
        shared_relabel_objects = original_non_actor_objects & relabel_non_actor_objects
        observability = event.observability or {}
        measurements = observability.get("measurements") or {}
        both_roles = {role.value for role in event.supporting_roles} >= {
            "first_person",
            "third_person",
        }
        source_direct_cv = bool(
            observability.get("can_cv_directly_prove_action") and both_roles
        )
        minimum_box_distance = measurements.get("minimum_box_distance_norm")
        strong_dual_role_contact = bool(
            original_action == ActionType.HAND_OBJECT_CONTACT
            and both_roles
            and event.confidence >= 0.60
            and int(measurements.get("candidate_count") or 0) >= 2
            and int(measurements.get("evidence_observation_count") or 0) >= 12
            and int(measurements.get("stable_track_token_count") or 0) >= 4
            and minimum_box_distance is not None
            and float(minimum_box_distance) <= 0.01
        )
        objective_cv_preserve = bool(source_direct_cv or strong_dual_role_contact)
        direct_action_values = {
            ActionType.HAND_OBJECT_CONTACT.value,
            ActionType.OBJECT_MOVEMENT.value,
            # Direct here refers to a visually complete tool-operation chain,
            # not to visible fluid. Its independent proof gate below requires
            # source contact, transport, distinct target contact and one
            # directly supporting view.
            ActionType.PIPETTE_TRANSFER_OPERATION.value,
        }
        confirmed_support = [
            item
            for item in review.get("confirmed_action_support_by_view") or []
            if isinstance(item, dict)
            and item.get("supports_confirmed_action") is True
            and str(item.get("view_id") or "").strip()
        ]
        reviewed_view_ids = {
            str(item.get("view_id") or "").strip()
            for item in [
                *(review.get("candidate_action_support_by_view") or []),
                *(review.get("confirmed_action_support_by_view") or []),
            ]
            if isinstance(item, dict) and str(item.get("view_id") or "").strip()
        }
        event_group = group_by_event[event.event_id]
        semantic_context_both_roles = bool(
            event_group.first_person_view in reviewed_view_ids
            and event_group.third_person_view in reviewed_view_ids
        )
        strong_direct_relabel_views = sorted(
            {
                str(item["view_id"])
                for item in confirmed_support
                if float(item.get("confidence") or 0.0) >= 0.80
            }
        )
        strong_direct_relabel_consensus = bool(
            model_action_value in direct_action_values
            and len(strong_direct_relabel_views) >= 2
        )
        low_risk_direct_relabel_views = sorted(
            {
                str(item["view_id"])
                for item in confirmed_support
                if float(item.get("confidence") or 0.0) >= 0.70
            }
        )
        low_risk_direct_relabel_consensus = bool(
            model_action_value == ActionType.HAND_OBJECT_CONTACT.value
            and len(low_risk_direct_relabel_views) >= 2
            and any(
                float(item.get("confidence") or 0.0) >= 0.80
                for item in confirmed_support
            )
        )
        direct_relabel_support_views = (
            low_risk_direct_relabel_views
            if low_risk_direct_relabel_consensus
            else strong_direct_relabel_views
        )
        strongest_direct_relabel_confidence = max(
            (
                float(item.get("confidence") or 0.0)
                for item in confirmed_support
                if str(item.get("view_id") or "") in strong_direct_relabel_views
            ),
            default=0.0,
        )
        action_proof = (event.model_understanding or {}).get("action_proof") or {}
        state_relabel_object_complete = bool(
            set(relabel_objects) & {"bottle_cap", "tube_cap"}
            and set(relabel_objects)
            & {
                "container",
                "reagent_bottle",
                "sample_bottle",
                "tube",
            }
        )
        strict_single_view_state_relabel = bool(
            model_action_value == ActionType.CONTAINER_STATE_CHANGE.value
            and semantic_context_both_roles
            # The aggregate is intentionally allowed to reflect a weak or
            # occluded context view.  Direct state proof keeps the stricter
            # threshold below, so cross-view image quality cannot turn a
            # clearly completed single-view transition into a false negative.
            and model_confidence >= relabel_min_confidence
            and len(strong_direct_relabel_views) >= 1
            and strongest_direct_relabel_confidence >= 0.80
            and action_proof.get("container_before_state_visible") is True
            and action_proof.get("container_after_state_visible") is True
            and action_proof.get("container_state_transition_completed") is True
            # Cross-view identity conflict is retained as uncertainty, but it
            # cannot erase a completed transition directly visible in one
            # view.  This mirrors the event semantic contract.
            and state_relabel_object_complete
        )
        relabel_confidence_gate = bool(
            model_confidence >= relabel_min_confidence
            or strong_direct_relabel_consensus
            or low_risk_direct_relabel_consensus
        )
        semantic_relabel_requested = bool(
            review_verdict == "relabel_suggested"
            or (
                str(review.get("pre_curation_verdict") or "") == "relabel_suggested"
                and model_action_value != original_action.value
            )
            or (
                review_verdict == "confirmed"
                and raw_model_verdict in {"uncertain", "rejected"}
                and model_action_value != original_action.value
                and (
                    strong_direct_relabel_consensus or low_risk_direct_relabel_consensus
                )
            )
        )
        semantic_relabel_overrides_objective_cv = bool(
            raw_model_verdict in {"uncertain", "rejected"}
            and (strong_direct_relabel_consensus or low_risk_direct_relabel_consensus)
        )
        relabel_base_conditions = all(
            (
                semantic_relabel_requested,
                # ``record_semantic_review`` independently derives the
                # effective verdict from the per-view support contract.  A
                # model can conservatively mark the *input candidate* as
                # uncertain while still naming a different directly visible
                # action and supporting it strongly in two views.  Accept
                # that narrow case; never override a raw rejection and never
                # apply it to state/liquid/device classes.
                raw_model_verdict == "relabel_suggested"
                or (
                    raw_model_verdict in {"uncertain", "rejected"}
                    and (
                        strong_direct_relabel_consensus
                        or low_risk_direct_relabel_consensus
                        or strict_single_view_state_relabel
                    )
                ),
                model_action_value in known_actions,
                model_action_value != original_action.value,
                relabel_confidence_gate,
                str(review.get("model_status") or "") == "completed",
                bool(relabel_non_actor_objects),
            )
        )
        # A model may see only three storyboard instants while deterministic CV
        # measures the complete interval.  Do not let a conflicting storyboard
        # overwrite a direct, dual-role tracked action.  For high-risk semantic
        # classes, also require object continuity and no cross-view conflict;
        # this prevents an unrelated nearby device from becoming a panel event.
        high_risk_relabel_safe = bool(
            model_action_value in direct_action_values
            or strict_single_view_state_relabel
            or (
                shared_relabel_objects
                and str(review.get("cross_view_consistency") or "") != "conflict"
            )
        )
        proposed_action = (
            known_actions.get(model_action_value, original_action)
            if semantic_relabel_requested
            else original_action
        )
        proof_contradictions = services.semantic_action_proof_contradictions(
            proposed_action,
            event.model_understanding,
            event.observability,
        )
        state_contact_support = [
            item
            for item in review.get("confirmed_action_support_by_view") or []
            if isinstance(item, dict)
            and item.get("supports_confirmed_action") is True
            and float(item.get("confidence") or 0.0) >= 0.70
            and str(item.get("view_id") or "").strip()
        ]
        state_contact_interactions = [
            item
            for item in (event.model_understanding or {}).get(
                "hand_object_interactions"
            )
            or []
            if isinstance(item, dict)
            and str(item.get("hand") or "").strip()
            and str(item.get("object") or "").strip()
            and str(item.get("contact") or "").strip()
        ]
        safe_state_to_contact_fallback = bool(
            proof_contradictions
            and original_action == ActionType.CONTAINER_STATE_CHANGE
            and model_action_value == ActionType.CONTAINER_STATE_CHANGE.value
            and str(review.get("model_status") or "") == "completed"
            and model_confidence >= 0.70
            and str(review.get("cross_view_consistency") or "") != "conflict"
            and len({str(item["view_id"]) for item in state_contact_support}) >= 2
            and bool(set(original_objects) & {"hand", "gloved_hand"})
            and bool(original_non_actor_objects)
            and bool(state_contact_interactions)
        )
        candidate_support_receipts = [
            item
            for item in review.get("candidate_action_support_by_view") or []
            if isinstance(item, dict) and str(item.get("view_id") or "").strip()
        ]
        objective_movement_unanimously_refuted = bool(
            original_action == ActionType.OBJECT_MOVEMENT
            and source_direct_cv
            and semantic_context_both_roles
            and raw_model_verdict in {"uncertain", "rejected"}
            and model_action_value not in known_actions
            and len(candidate_support_receipts) >= 2
            and all(
                item.get("supports_candidate_action") is False
                for item in candidate_support_receipts
            )
            and str(review.get("cross_view_consistency") or "") == "conflict"
        )
        movement_semantic_object_mismatch = bool(
            original_action == ActionType.OBJECT_MOVEMENT
            and source_direct_cv
            and semantic_context_both_roles
            and model_action_value == ActionType.OBJECT_MOVEMENT.value
            and str(review.get("cross_view_consistency") or "") == "conflict"
            and not shared_relabel_objects
            and not relabel_non_actor_objects
            and len(confirmed_support) == 1
            and any(
                item.get("supports_candidate_action") is False
                for item in candidate_support_receipts
            )
        )
        final_action: ActionType | None = None
        disposition = "excluded_semantically_unconfirmed"
        unmapped_interaction_participants = bool(
            proposed_action == ActionType.HAND_OBJECT_CONTACT
            and model_action_value == ActionType.HAND_OBJECT_CONTACT.value
            and str(review.get("model_status") or "") == "completed"
            and model_confidence >= relabel_min_confidence
            and original_non_actor_objects
            and not relabel_non_actor_objects
            and any(
                isinstance(item, dict)
                and str(item.get("object") or "").strip()
                and services._semantic_interaction_is_direct(item)
                for item in (event.model_understanding or {}).get(
                    "hand_object_interactions", []
                )
                or []
            )
        )
        if unmapped_interaction_participants:
            # A known action type does not confirm the CV target's identity.
            # For example, handling gloves cannot validate a nearby paper box.
            # Preserve the media for review instead of retaining unrelated CV
            # targets when the explicit semantic participants are unmapped.
            disposition = (
                "review_candidate_conflicting_interaction_participants"
                if participant_conflicts
                else "review_candidate_unmapped_interaction_participants"
            )
        elif safe_state_to_contact_fallback:
            # A hand visibly manipulating a cap can safely establish contact
            # even when the before/after evidence is too weak to publish the
            # higher-risk open/close state transition.  Keep only the CV
            # participant objects so background objects named by the model do
            # not leak into the final annotations.
            final_action = ActionType.HAND_OBJECT_CONTACT
            disposition = "accepted_state_safety_downclass_to_contact"
        elif proof_contradictions:
            disposition = "excluded_semantic_proof_contradiction"
        elif objective_movement_unanimously_refuted:
            disposition = "excluded_unanimously_refuted_object_movement"
        elif movement_semantic_object_mismatch:
            disposition = "review_candidate_semantic_object_identity_mismatch"
        elif (
            relabel_base_conditions
            and high_risk_relabel_safe
            and semantic_relabel_overrides_objective_cv
        ):
            final_action = known_actions[model_action_value]
            disposition = "accepted_strict_relabel"
        elif review_verdict == "confirmed":
            final_action = original_action
            disposition = "accepted_confirmed"
        elif objective_cv_preserve and (
            review_verdict in {"uncertain", "relabel_suggested"}
            or (source_direct_cv and review_verdict == "rejected")
        ):
            final_action = original_action
            disposition = "accepted_objective_cv_preserved_over_sparse_semantics"
        elif relabel_base_conditions and high_risk_relabel_safe:
            final_action = known_actions[model_action_value]
            disposition = "accepted_strict_relabel"

        movement: dict[str, Any] = {
            "moved_media": [],
            "retained_media": [],
        }
        if final_action is None:
            movement = move_event_media(event, destination_action=None)
            set_event_admission(event, "rejected")
        elif final_action != original_action:
            event.action_type = final_action
            event.objects = (
                original_objects
                if disposition == "accepted_state_safety_downclass_to_contact"
                else relabel_objects
            )
            movement = move_event_media(event, destination_action=final_action)
            set_event_admission(event, "formal")
        else:
            set_event_admission(event, "formal")

        semantic_participant_refined = bool(review.get("semantic_participant_refined"))
        semantic_participant_refinement_changed = False
        participant_refinement_actions = {
            ActionType.HAND_OBJECT_CONTACT,
            ActionType.OBJECT_MOVEMENT,
            ActionType.CONTAINER_STATE_CHANGE,
            ActionType.DEVICE_PANEL_OPERATION,
            ActionType.PIPETTE_TRANSFER_OPERATION,
        }
        structured_interactions = [
            item
            for item in (event.model_understanding or {}).get(
                "hand_object_interactions"
            )
            or []
            if isinstance(item, dict)
            and str(item.get("object") or "").strip()
            and str(item.get("contact") or "").strip()
        ]
        semantic_participant_refinement_safe = bool(
            final_action is not None
            and final_action in participant_refinement_actions
            and model_action_value == final_action.value
            and str(review.get("model_status") or "") == "completed"
            and model_confidence >= relabel_min_confidence
            and structured_interactions
            and relabel_non_actor_objects
        )
        if semantic_participant_refinement_safe:
            refined_objects = list(relabel_objects)
            if refined_objects != list(event.objects):
                event.objects = refined_objects
                semantic_participant_refined = True
                semantic_participant_refinement_changed = True
                if final_action == original_action:
                    movement = move_event_media(event, destination_action=final_action)

        semantic_relabel_direct_support = bool(
            final_action is not None
            and final_action != original_action
            and direct_relabel_support_views
            and disposition == "accepted_strict_relabel"
        )
        if semantic_relabel_direct_support:
            # The per-view semantic contract can directly prove a relabelled
            # action in views that were only context for the original CV
            # candidate. Replace the original-candidate topology with those
            # *semantic* supports so the final action never inherits direct
            # support claims from a rejected class. Aligned opposite-role
            # media remains available as explicitly labelled context.
            role_by_view: dict[str, ViewRole] = {
                candidate.view_id: candidate.role for candidate in event.candidates
            }
            group = group_by_event[event.event_id]
            if group.first_person_view:
                role_by_view.setdefault(group.first_person_view, ViewRole.FIRST_PERSON)
            if group.third_person_view:
                role_by_view.setdefault(group.third_person_view, ViewRole.THIRD_PERSON)
            if len(event.supporting_views) == len(event.supporting_roles) == 1:
                role_by_view.setdefault(
                    event.supporting_views[0], event.supporting_roles[0]
                )
            recall_admission = (event.observability or {}).get(
                "semantic_recall_admission"
            ) or {}
            context_view_id = str(recall_admission.get("context_view_id") or "").strip()
            context_role_value = str(recall_admission.get("context_role") or "").strip()
            if context_view_id and context_role_value in {
                item.value for item in ViewRole
            }:
                role_by_view[context_view_id] = ViewRole(context_role_value)
            supported_role_pairs = [
                (view_id, role_by_view[view_id])
                for view_id in direct_relabel_support_views
                if view_id in role_by_view
            ]
            event.supporting_views = sorted(
                view_id for view_id, _role in supported_role_pairs
            )
            event.supporting_roles = sorted(
                {
                    *(role for _view_id, role in supported_role_pairs),
                },
                key=lambda role: role.value,
            )
            observability_update = dict(event.observability or {})
            observability_update["semantic_direct_support"] = {
                "schema_version": "visioncortex-semantic-direct-support/1",
                "action_type": final_action.value,
                "view_ids": [view_id for view_id, _role in supported_role_pairs],
                "roles": [role.value for _view_id, role in supported_role_pairs],
                "source": "confirmed_action_support_by_view",
                "cv_direct_support_unchanged": True,
            }
            event.observability = observability_update

        semantic_state_machine_rebuilt = False
        if final_action is not None and (
            final_action != original_action or semantic_participant_refinement_changed
        ):
            # A deterministic CV state machine is valid only for the action it
            # was built from.  Once semantic adjudication relabels the event,
            # retain that receipt as pre-curation provenance and derive a new
            # final state contract from the curated action, participants and
            # direct-support topology.  Candidate track ids are deliberately
            # cleared because they belong to the rejected CV hypothesis.
            rebuilt_state = services.build_event_state_receipt(event, config)
            rebuilt_state["derivation"] = {
                "source": (
                    "semantic_relabel_confirmed_state_proof"
                    if final_action != original_action
                    else "semantic_participant_refinement"
                ),
                "pre_curation_action_type": original_action.value,
                "final_action_type": final_action.value,
                "model_confidence": model_confidence,
                "direct_supporting_views": list(event.supporting_views),
            }
            for transition in rebuilt_state.get("transition_trace") or []:
                transition["source"] = rebuilt_state["derivation"]["source"]
            identity = rebuilt_state.get("object_identity") or {}
            identity["track_tokens"] = []
            identity["identity_status"] = "semantic_participant_classes"
            rebuilt_state["object_identity"] = identity
            event.state_machine = rebuilt_state
            semantic_state_machine_rebuilt = True

        review.update(
            {
                "pre_curation_verdict": (
                    review.get("pre_curation_verdict") or review_verdict or None
                ),
                "pre_curation_action_type": provenance_action_type,
                "pre_curation_objects": provenance_objects,
                "pre_curation_supporting_views": provenance_supporting_views,
                "pre_curation_supporting_roles": provenance_supporting_roles,
                "pre_curation_state_machine": provenance_state_machine,
                "final_delivery_accepted": final_action is not None,
                "final_action_type": final_action.value if final_action else None,
                "final_participant_objects": (
                    list(event.objects) if final_action else []
                ),
                "semantic_participant_refined": semantic_participant_refined,
                "semantic_participant_refinement_changed": (
                    semantic_participant_refinement_changed
                ),
                "semantic_participant_refinement_safe": (
                    semantic_participant_refinement_safe
                ),
                "curation_disposition": disposition,
                "curation_policy": "semantic-final-key-material-v1",
                "curation_relabel_min_confidence": relabel_min_confidence,
                "semantic_state_machine_rebuilt": semantic_state_machine_rebuilt,
                "semantic_proof_contradictions": proof_contradictions,
                "semantic_participant_conflicts": participant_conflicts,
                "relabel_safety_gate": {
                    "source_direct_dual_role_cv": source_direct_cv,
                    "strong_dual_role_contact": strong_dual_role_contact,
                    "objective_cv_preserve": objective_cv_preserve,
                    "target_is_direct_action": model_action_value
                    in direct_action_values,
                    "strong_direct_relabel_view_ids": (strong_direct_relabel_views),
                    "strong_direct_relabel_consensus": (
                        strong_direct_relabel_consensus
                    ),
                    "low_risk_direct_relabel_view_ids": (low_risk_direct_relabel_views),
                    "low_risk_direct_relabel_consensus": (
                        low_risk_direct_relabel_consensus
                    ),
                    "strict_single_view_state_relabel": (
                        strict_single_view_state_relabel
                    ),
                    "semantic_context_both_roles": semantic_context_both_roles,
                    "semantic_reviewed_view_ids": sorted(reviewed_view_ids),
                    "state_relabel_object_complete": (state_relabel_object_complete),
                    "state_relabel_model_confidence_min": (relabel_min_confidence),
                    "state_relabel_direct_confidence_min": 0.80,
                    "strongest_direct_relabel_confidence": (
                        strongest_direct_relabel_confidence
                    ),
                    "relabel_confidence_gate": relabel_confidence_gate,
                    "semantic_relabel_requested": semantic_relabel_requested,
                    "semantic_relabel_overrides_objective_cv": (
                        semantic_relabel_overrides_objective_cv
                    ),
                    "shared_non_actor_objects": sorted(shared_relabel_objects),
                    "cross_view_consistency": review.get("cross_view_consistency"),
                    "high_risk_relabel_safe": high_risk_relabel_safe,
                    "objective_movement_unanimously_refuted": (
                        objective_movement_unanimously_refuted
                    ),
                    "movement_semantic_object_mismatch": (
                        movement_semantic_object_mismatch
                    ),
                },
                "pre_curation_media": provenance_media,
                **movement,
            }
        )
        if final_action is not None:
            review["verdict"] = "confirmed"
            curated.append(event)
        event.semantic_review = review
        records.append(
            {
                "event_id": event.event_id,
                "cv_action_type": provenance_action_type,
                "cv_objects": provenance_objects,
                "model_action_type": model_action_value or None,
                "pre_curation_verdict": review_verdict or None,
                "model_confidence": model_confidence,
                "disposition": disposition,
                "final_action_type": final_action.value if final_action else None,
                "final_participant_objects": (
                    list(event.objects) if final_action else []
                ),
                "semantic_participant_refined": semantic_participant_refined,
                "semantic_participant_refinement_changed": (
                    semantic_participant_refinement_changed
                ),
                "final_delivery_accepted": final_action is not None,
                "semantic_state_machine_rebuilt": semantic_state_machine_rebuilt,
                **movement,
            }
        )

    # A strict relabel can collide with an event that was already selected in
    # the corrected class. Run a second physical-action deduplication after
    # relabeling so the final user tree never contains semantic duplicates.
    actor_objects = {"hand", "gloved_hand", "lab_coat"}
    record_by_event = {item["event_id"]: item for item in records}
    duplicate_event_ids: set[str] = set()
    deduplication_records: list[dict[str, Any]] = []
    subsumed_event_ids: set[str] = set()
    state_subsumption_records: list[dict[str, Any]] = []
    separation_ms = (
        float(config.get("key_materials", {}).get("minimum_separation_seconds", 1.5))
        * 1000.0
    )

    def retention_rank(event: EvidenceEvent) -> tuple[int, float, float, str]:
        record = record_by_event[event.event_id]
        return (
            int(record["disposition"] == "accepted_confirmed"),
            float(record.get("model_confidence") or 0.0),
            float(event.confidence),
            event.event_id,
        )

    def state_transition_signature(event: EvidenceEvent) -> str | None:
        if event.action_type != ActionType.CONTAINER_STATE_CHANGE:
            return None
        physical_change = (event.model_understanding or {}).get("physical_change") or {}
        before = str(physical_change.get("before") or "").lower()
        after = str(physical_change.get("after") or "").lower()
        # Ark commonly describes the same transition with nearby Chinese
        # variants (for example ``瓶盖盖住`` -> ``瓶口露出`` or
        # ``瓶口封闭`` -> ``瓶口开放``).  Directional semantic
        # deduplication must recognise those variants; otherwise two heavily
        # overlapping CV candidates can both be published as the same opening.
        closed_terms = (
            "closed",
            "capped",
            "盖合",
            "盖住",
            "盖着",
            "封闭",
            "密封",
            "瓶盖在瓶口",
            "瓶盖覆盖",
        )
        open_terms = (
            "open",
            "uncapped",
            "打开",
            "开启",
            "开放",
            "敞口",
            "瓶口开放",
            "瓶口露出",
            "瓶口暴露",
            "瓶盖与瓶口分离",
            "瓶口敞开",
            "瓶盖已被取下",
            "瓶盖被取下",
            "瓶盖离开瓶口",
        )
        before_closed = any(term in before for term in closed_terms)
        before_open = any(term in before for term in open_terms)
        after_closed = any(term in after for term in closed_terms)
        after_open = any(term in after for term in open_terms)
        if before_closed and after_open:
            return "opening"
        if before_open and after_closed:
            return "closing"
        return None

    for group in groups:
        bucket: list[EvidenceEvent] = []
        group_events = sorted(
            (event for event in curated if event.event_id in group.key_event_ids),
            key=lambda event: (event.key_global_ms, event.event_id),
        )
        for event in group_events:
            duplicate_of: EvidenceEvent | None = None
            for existing in bucket:
                if existing.action_type != event.action_type:
                    continue
                shared_objects = (
                    set(existing.objects) & set(event.objects)
                ) - actor_objects
                existing_identities = services.event_stable_identities(existing)
                event_identities = services.event_stable_identities(event)
                stable_identity_available = bool(
                    existing_identities and event_identities
                )
                shared_stable_identities = existing_identities & event_identities
                identity_compatible = bool(
                    shared_stable_identities
                    if stable_identity_available
                    else shared_objects
                )
                interval_overlap_ms = max(
                    0.0,
                    min(existing.global_end_ms, event.global_end_ms)
                    - max(existing.global_start_ms, event.global_start_ms),
                )
                peak_distance_ms = abs(existing.key_global_ms - event.key_global_ms)
                minimum_duration_ms = max(
                    1.0,
                    min(
                        existing.global_end_ms - existing.global_start_ms,
                        event.global_end_ms - event.global_start_ms,
                    ),
                )
                overlap_ratio = interval_overlap_ms / minimum_duration_ms
                existing_transition = state_transition_signature(existing)
                event_transition = state_transition_signature(event)
                same_completed_state_transition = bool(
                    existing_transition
                    and existing_transition == event_transition
                    and overlap_ratio >= 0.50
                )
                if (
                    identity_compatible
                    and interval_overlap_ms > 0.0
                    and (
                        peak_distance_ms < separation_ms
                        or same_completed_state_transition
                    )
                ):
                    duplicate_of = existing
                    break
            if duplicate_of is None:
                bucket.append(event)
                continue
            retained, dropped = sorted(
                (duplicate_of, event),
                key=retention_rank,
                reverse=True,
            )
            if retained is event:
                bucket[bucket.index(duplicate_of)] = event
            duplicate_event_ids.add(dropped.event_id)
            movement = move_event_media(dropped, destination_action=None)
            set_event_admission(dropped, "rejected")
            dropped_review = dropped.semantic_review or {}
            dropped_review.update(
                {
                    "verdict": "duplicate",
                    "final_delivery_accepted": False,
                    "final_action_type": None,
                    "curation_disposition": "excluded_post_relabel_duplicate",
                    "semantic_duplicate_of": retained.event_id,
                    "moved_media": [
                        *(dropped_review.get("moved_media") or []),
                        *movement["moved_media"],
                    ],
                }
            )
            dropped.semantic_review = dropped_review
            record = record_by_event[dropped.event_id]
            record.update(
                {
                    "disposition": "excluded_post_relabel_duplicate",
                    "final_action_type": None,
                    "final_delivery_accepted": False,
                    "semantic_duplicate_of": retained.event_id,
                    "moved_media": [
                        *(record.get("moved_media") or []),
                        *movement["moved_media"],
                    ],
                }
            )
            deduplication_records.append(
                {
                    "dropped_event_id": dropped.event_id,
                    "retained_event_id": retained.event_id,
                    "final_action_type": retained.action_type.value,
                    "shared_objects": sorted(
                        (set(retained.objects) & set(dropped.objects)) - actor_objects
                    ),
                    "shared_stable_identities": sorted(
                        services.event_stable_identities(retained)
                        & services.event_stable_identities(dropped)
                    ),
                    "stable_identity_required_when_available": True,
                    "interval_overlap_ms": round(
                        max(
                            0.0,
                            min(retained.global_end_ms, dropped.global_end_ms)
                            - max(retained.global_start_ms, dropped.global_start_ms),
                        ),
                        3,
                    ),
                    "peak_distance_ms": round(
                        abs(retained.key_global_ms - dropped.key_global_ms), 3
                    ),
                    "interval_overlap_ratio": round(
                        max(
                            0.0,
                            min(retained.global_end_ms, dropped.global_end_ms)
                            - max(retained.global_start_ms, dropped.global_start_ms),
                        )
                        / max(
                            1.0,
                            min(
                                retained.global_end_ms - retained.global_start_ms,
                                dropped.global_end_ms - dropped.global_start_ms,
                            ),
                        ),
                        4,
                    ),
                    "state_transition_signature": state_transition_signature(retained),
                    "retention_policy": (
                        "prefer_original_confirmed_then_model_confidence_then_cv_confidence"
                    ),
                }
            )

    # A confirmed, completed container transition already contains the
    # operator-to-cap/bottle contact that causes it. Publishing nested generic
    # hand-contact candidates as separate key materials adds duplicate frames
    # and invites background-instance errors without adding a distinct lab
    # fact. Keep the richer state transition and retain the lower-level
    # candidates only in the persistent audit cache.
    for group in groups:
        live_group_events = [
            event
            for event in curated
            if event.event_id in group.key_event_ids
            and event.event_id not in duplicate_event_ids
        ]
        completed_states = [
            event
            for event in live_group_events
            if event.action_type == ActionType.CONTAINER_STATE_CHANGE
            and bool(
                ((event.model_understanding or {}).get("action_proof") or {}).get(
                    "container_state_transition_completed"
                )
            )
        ]
        for contact in live_group_events:
            if contact.action_type != ActionType.HAND_OBJECT_CONTACT:
                continue
            contact_duration_ms = max(
                1.0, contact.global_end_ms - contact.global_start_ms
            )
            for state_event in completed_states:
                shared_objects = (
                    set(contact.objects) & set(state_event.objects)
                ) - actor_objects
                contact_identities = services.event_stable_identities(contact)
                state_identities = services.event_stable_identities(state_event)
                stable_identity_available = bool(
                    contact_identities and state_identities
                )
                shared_stable_identities = contact_identities & state_identities
                identity_compatible = bool(
                    shared_stable_identities
                    if stable_identity_available
                    else shared_objects
                )
                interval_overlap_ms = max(
                    0.0,
                    min(contact.global_end_ms, state_event.global_end_ms)
                    - max(contact.global_start_ms, state_event.global_start_ms),
                )
                overlap_ratio = interval_overlap_ms / contact_duration_ms
                if not (
                    identity_compatible
                    and overlap_ratio >= 0.80
                    and state_event.global_start_ms
                    <= contact.key_global_ms
                    <= state_event.global_end_ms
                ):
                    continue
                subsumed_event_ids.add(contact.event_id)
                movement = move_event_media(contact, destination_action=None)
                set_event_admission(contact, "rejected")
                contact_review = contact.semantic_review or {}
                contact_review.update(
                    {
                        "verdict": "subsumed",
                        "final_delivery_accepted": False,
                        "final_action_type": None,
                        "curation_disposition": (
                            "excluded_subsumed_by_container_state_transition"
                        ),
                        "semantic_subsumed_by": state_event.event_id,
                        "moved_media": [
                            *(contact_review.get("moved_media") or []),
                            *movement["moved_media"],
                        ],
                    }
                )
                contact.semantic_review = contact_review
                record = record_by_event[contact.event_id]
                record.update(
                    {
                        "disposition": (
                            "excluded_subsumed_by_container_state_transition"
                        ),
                        "final_action_type": None,
                        "final_delivery_accepted": False,
                        "semantic_subsumed_by": state_event.event_id,
                        "moved_media": [
                            *(record.get("moved_media") or []),
                            *movement["moved_media"],
                        ],
                    }
                )
                state_subsumption_records.append(
                    {
                        "dropped_event_id": contact.event_id,
                        "retained_event_id": state_event.event_id,
                        "dropped_action_type": contact.action_type.value,
                        "retained_action_type": state_event.action_type.value,
                        "shared_objects": sorted(shared_objects),
                        "shared_stable_identities": sorted(shared_stable_identities),
                        "stable_identity_required_when_available": True,
                        "interval_overlap_ms": round(interval_overlap_ms, 3),
                        "contact_interval_overlap_ratio": round(overlap_ratio, 4),
                        "policy": (
                            "generic_contact_subsumed_by_completed_container_state_transition"
                        ),
                    }
                )
                break

    curated = [
        event
        for event in curated
        if event.event_id not in duplicate_event_ids | subsumed_event_ids
    ]

    curated_ids = {event.event_id for event in curated}
    for group in groups:
        group.key_event_ids = [
            event_id for event_id in group.key_event_ids if event_id in curated_ids
        ]
    final_annotation = services._rerender_curated_participant_annotations(
        layout, curated, groups, config
    )
    services.write_key_material_category_index(
        layout,
        groups,
        curated,
        include_empty_categories=bool(
            config.get("archive", {}).get("include_empty_action_categories", True)
        ),
    )
    event_by_id = {event.event_id: event for event in events}
    review_candidates: list[dict[str, Any]] = []
    for record in records:
        if record.get("final_delivery_accepted") is True:
            continue
        event_id = str(record["event_id"])
        event = event_by_id[event_id]
        media_root = quarantine_root / event_id
        review_candidates.append(
            {
                "event_id": event_id,
                "status": "machine_quarantined_not_confirmed_key_material",
                "disposition": record.get("disposition"),
                "cv_action_type": record.get("cv_action_type"),
                "cv_objects": record.get("cv_objects") or [],
                "model_action_type": record.get("model_action_type"),
                "global_start_ms": event.global_start_ms,
                "global_end_ms": event.global_end_ms,
                "key_global_ms": event.key_global_ms,
                "source_views": list(
                    (event.semantic_review or {}).get("pre_curation_supporting_views")
                    or event.supporting_views
                ),
                "view_pairing": dict(
                    event.observability.get("key_material_view_selection") or {}
                ),
                "media": sorted(
                    services._relative(path, layout.root)
                    for path in media_root.rglob("*")
                    if path.is_file()
                ),
                "semantic_receipt": (
                    "JSON-Config-Files/semantic_key_material_curation.json"
                ),
                "source_reference": (
                    "Original-Experiment-Videos/Original-Video-Index.json"
                ),
            }
        )
    review_candidate_index_path = quarantine_root / "Machine-Quarantine-Index.json"
    services.write_json(
        review_candidate_index_path,
        {
            "schema_version": "visioncortex-machine-quarantine-index/1",
            "policy": (
                "unconfirmed candidates are automatically quarantined and indexed; "
                "they never require manual fallback and are never counted as "
                "confirmed key material"
            ),
            "candidate_count": len(review_candidates),
            "candidates": review_candidates,
        },
    )
    report = {
        "schema_version": "visioncortex-semantic-key-material-curation/1",
        "policy": "semantic-final-key-material-v1",
        "candidate_count": len(events),
        "accepted_count": len(curated),
        "excluded_count": len(events) - len(curated),
        "confirmed_count": sum(
            item["disposition"] == "accepted_confirmed"
            and item["final_delivery_accepted"]
            for item in records
        ),
        "strict_relabel_count": sum(
            item["disposition"] == "accepted_strict_relabel"
            and item["final_delivery_accepted"]
            for item in records
        ),
        "objective_cv_preserved_count": sum(
            item["disposition"]
            == "accepted_objective_cv_preserved_over_sparse_semantics"
            and item["final_delivery_accepted"]
            for item in records
        ),
        "state_safety_downclass_count": sum(
            item["disposition"] == "accepted_state_safety_downclass_to_contact"
            and item["final_delivery_accepted"]
            for item in records
        ),
        "unsafe_relabel_excluded_count": sum(
            item["pre_curation_verdict"] == "relabel_suggested"
            and item["disposition"] == "excluded_semantically_unconfirmed"
            for item in records
        ),
        "semantic_proof_contradiction_excluded_count": sum(
            item["disposition"] == "excluded_semantic_proof_contradiction"
            for item in records
        ),
        "post_relabel_duplicate_count": len(duplicate_event_ids),
        "post_relabel_deduplication": deduplication_records,
        "state_subsumed_contact_count": len(subsumed_event_ids),
        "state_contact_subsumption": state_subsumption_records,
        "semantic_relabel_min_confidence": relabel_min_confidence,
        "unconfirmed_media_retention": "automatic_machine_quarantine_with_source_reference",
        "unconfirmed_media_formally_published": True,
        "unconfirmed_media_counted_as_confirmed": False,
        "manual_fallback_required": False,
        "review_candidate_count": len(review_candidates),
        "review_candidate_index": services._relative(
            review_candidate_index_path, layout.root
        ),
        "final_annotation": {
            key: value for key, value in final_annotation.items() if key != "records"
        },
        "records": records,
    }
    services.write_json(
        layout.json_config / "semantic_key_material_curation.json", report
    )
    return curated, report


def _annotation_supports_curated_action(
    event: EvidenceEvent,
    annotation: dict[str, Any],
    *,
    services: SemanticCurationServices,
) -> bool:
    """Apply the final same-view participant contract before publication."""

    rendered_by_view = {
        str(view_id): {
            str(item).strip().lower().replace("-", "_").replace(" ", "_")
            for item in (receipt.get("rendered_classes") or [])
        }
        for view_id, receipt in (annotation.get("views") or {}).items()
        if isinstance(receipt, dict)
    }
    return bool(
        services.action_participant_visibility(
            event.action_type, event.objects, rendered_by_view
        )["passed"]
    )


def reconcile_visually_reviewed_participants(
    layout: ArchiveLayout,
    events: Sequence[EvidenceEvent],
    groups: Sequence[ExperimentGroup],
    semantic_curation: dict[str, Any],
    *,
    services: SemanticCurationServices,
) -> tuple[list[EvidenceEvent], dict[str, Any]]:
    """Prune unsupported optional participants and quarantine invalid events.

    Semantic understanding can name several objects observed across a long
    event.  Final participant review is tied to the selected key frame.  A
    paper or cap that cannot be rendered there must not remain a claimed final
    participant.  If other rendered objects still satisfy the action contract,
    retain the event with the unsupported class removed.  Otherwise move the
    event to the formal review-candidate tree instead of failing the complete
    archive or publishing an unverifiable key material.
    """

    group_by_event = {
        event_id: group for group in groups for event_id in group.key_event_ids
    }
    record_by_event = {
        str(record.get("event_id")): record
        for record in semantic_curation.get("records") or []
        if isinstance(record, dict) and record.get("event_id")
    }
    quarantine_root = layout.key_materials / "Review-Candidates"
    retained: list[EvidenceEvent] = []
    pruned_records: list[dict[str, Any]] = []
    excluded_records: list[dict[str, Any]] = []

    for event in events:
        annotation = (event.observability or {}).get("key_material_annotation") or {}
        visual_review = annotation.get("participant_visual_review") or {}
        unsupported_classes = sorted(
            {
                str(review.get("participant_class") or "")
                for review in visual_review.get("reviews") or []
                if isinstance(review, dict)
                and (
                    (
                        review.get("status") == "completed"
                        and int(review.get("localized_view_count") or 0) == 0
                        and str(review.get("participant_class") or "")
                        in {"paper", "bottle_cap"}
                    )
                    or (
                        review.get("status") == "review_failed_quarantined"
                        and str(review.get("participant_class") or "")
                        in {"paper", "bottle_cap", "balance"}
                    )
                )
            }
        )
        previous_objects = list(event.objects)
        event.objects = [
            item for item in event.objects if item not in unsupported_classes
        ]
        supported = services._annotation_supports_curated_action(event, annotation)
        if not unsupported_classes and supported:
            retained.append(event)
            continue
        group = group_by_event[event.event_id]
        destination_action = event.action_type if supported else None
        movement = services._move_curated_event_media(
            layout,
            event,
            group,
            quarantine_root,
            destination_action=destination_action,
        )
        receipt = {
            "schema_version": "visioncortex-visual-participant-reconciliation/1",
            "event_id": event.event_id,
            "unsupported_classes": unsupported_classes,
            "previous_participant_objects": previous_objects,
            "final_participant_objects": list(event.objects) if supported else [],
            "action_participant_visibility_after_pruning": supported,
            "disposition": (
                "accepted_after_visual_participant_pruning"
                if supported
                else "review_candidate_missing_required_visual_participant"
            ),
            "policy": (
                "remove only visually unsupported reviewed classes; retain the "
                "event only when remaining rendered objects satisfy its action contract"
            ),
            **movement,
        }
        review = event.semantic_review or {}
        review["visual_participant_reconciliation"] = receipt
        review["final_participant_objects"] = receipt["final_participant_objects"]
        record = record_by_event.get(event.event_id)
        if record is not None:
            record["visual_participant_reconciliation"] = receipt
            record["final_participant_objects"] = receipt["final_participant_objects"]
            record["post_annotation_disposition"] = receipt["disposition"]
            record["moved_media"] = [
                *(record.get("moved_media") or []),
                *(movement.get("moved_media") or []),
            ]
            record["retained_media"] = [
                *(record.get("retained_media") or []),
                *(movement.get("retained_media") or []),
            ]
        if supported:
            event.accepted = True
            review["final_delivery_accepted"] = True
            review["post_annotation_disposition"] = receipt["disposition"]
            if record is not None:
                record["final_delivery_accepted"] = True
            pruned_records.append(receipt)
            retained.append(event)
        else:
            event.accepted = False
            review.update(
                {
                    "verdict": "visually_unconfirmed",
                    "final_delivery_accepted": False,
                    "final_action_type": None,
                    "curation_disposition": receipt["disposition"],
                }
            )
            if record is not None:
                record.update(
                    {
                        "disposition": receipt["disposition"],
                        "final_delivery_accepted": False,
                        "final_action_type": None,
                    }
                )
            excluded_records.append(receipt)
        event.semantic_review = review

    retained_ids = {event.event_id for event in retained}
    for group in groups:
        group.key_event_ids = [
            event_id for event_id in group.key_event_ids if event_id in retained_ids
        ]

    if excluded_records:
        index_path = quarantine_root / "Candidate-Index.json"
        index = (
            json.loads(index_path.read_text(encoding="utf-8-sig"))
            if index_path.is_file()
            else {
                "schema_version": "visioncortex-review-candidate-index/1",
                "policy": (
                    "unconfirmed candidates remain visible and indexed; they are "
                    "never counted as confirmed key material"
                ),
                "candidates": [],
            }
        )
        by_id = {
            str(item.get("event_id")): item
            for item in index.get("candidates") or []
            if isinstance(item, dict) and item.get("event_id")
        }
        event_by_id = {event.event_id: event for event in events}
        for receipt in excluded_records:
            event = event_by_id[receipt["event_id"]]
            media_root = quarantine_root / event.event_id
            by_id[event.event_id] = {
                "event_id": event.event_id,
                "status": "review_candidate_not_confirmed_key_material",
                "disposition": receipt["disposition"],
                "cv_action_type": (
                    (event.semantic_review or {}).get("pre_curation_action_type")
                    or event.action_type.value
                ),
                "cv_objects": (
                    (event.semantic_review or {}).get("pre_curation_objects")
                    or receipt["previous_participant_objects"]
                ),
                "model_action_type": (
                    (event.semantic_review or {}).get("model_action_type")
                ),
                "global_start_ms": event.global_start_ms,
                "global_end_ms": event.global_end_ms,
                "key_global_ms": event.key_global_ms,
                "source_views": list(event.supporting_views),
                "media": sorted(
                    services._relative(path, layout.root)
                    for path in media_root.rglob("*")
                    if path.is_file()
                ),
                "semantic_receipt": (
                    "JSON-Config-Files/semantic_key_material_curation.json"
                ),
                "source_reference": (
                    "Original-Experiment-Videos/Original-Video-Index.json"
                ),
            }
        index["candidates"] = list(by_id.values())
        index["candidate_count"] = len(index["candidates"])
        services.write_json(index_path, index)
        semantic_curation["review_candidate_count"] = index["candidate_count"]

    records = semantic_curation.get("records") or []
    accepted_count = sum(
        bool(record.get("final_delivery_accepted"))
        for record in records
        if isinstance(record, dict)
    )
    semantic_curation.update(
        {
            "accepted_count": accepted_count,
            "excluded_count": int(semantic_curation.get("candidate_count") or 0)
            - accepted_count,
            "confirmed_count": sum(
                record.get("disposition") == "accepted_confirmed"
                and bool(record.get("final_delivery_accepted"))
                for record in records
                if isinstance(record, dict)
            ),
            "strict_relabel_count": sum(
                record.get("disposition") == "accepted_strict_relabel"
                and bool(record.get("final_delivery_accepted"))
                for record in records
                if isinstance(record, dict)
            ),
            "objective_cv_preserved_count": sum(
                record.get("disposition")
                == "accepted_objective_cv_preserved_over_sparse_semantics"
                and bool(record.get("final_delivery_accepted"))
                for record in records
                if isinstance(record, dict)
            ),
            "state_safety_downclass_count": sum(
                record.get("disposition")
                == "accepted_state_safety_downclass_to_contact"
                and bool(record.get("final_delivery_accepted"))
                for record in records
                if isinstance(record, dict)
            ),
            "visual_participant_reconciliation": {
                "schema_version": (
                    "visioncortex-visual-participant-reconciliation-index/1"
                ),
                "policy": (
                    "prune unsupported reviewed participants and quarantine an "
                    "event if its remaining final frame cannot satisfy the action contract"
                ),
                "pruned_event_count": len(pruned_records),
                "excluded_event_count": len(excluded_records),
                "pruned_events": pruned_records,
                "excluded_events": excluded_records,
            },
        }
    )
    return retained, semantic_curation
