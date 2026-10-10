"""Key-material metadata, view pairing and category indexes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from ..material_naming import (
    ACTION_CATEGORY_FOLDERS,
    key_material_action_folder,
)
from ..material_naming import (
    key_material_semantic_name as _key_material_semantic_name,
)
from ..schemas import (
    ActionType,
    AlignmentTransform,
    EvidenceEvent,
    ExperimentGroup,
    VideoInfo,
    ViewInput,
    ViewRole,
)
from .layout import ArchiveLayout


@dataclass(frozen=True)
class ArtifactsServices:
    """Named execution ports; supplied explicitly at the composition boundary."""

    _artifact_json: Callable[..., Any]
    _key_material_event_folder_name: Callable[..., Any]
    _key_material_view_pair: Callable[..., Any]
    _relative: Callable[..., Any]
    _safe_folder_name: Callable[..., Any]
    _safe_slug: Callable[..., Any]
    _select_key_material_view_pair: Callable[..., Any]
    event_view_actor_identities: Callable[..., Any]
    event_view_stable_identities: Callable[..., Any]
    stable_artifact_uid: Callable[..., Any]
    stable_event_uid: Callable[..., Any]
    stable_evidence_uid: Callable[..., Any]
    write_json: Callable[..., Any]


def _key_material_view_pair(
    group: ExperimentGroup,
    event: EvidenceEvent,
) -> tuple[str, str]:
    selection = event.observability.get("key_material_view_selection") or {}
    return (
        str(selection.get("first_person_view") or group.first_person_view),
        str(selection.get("third_person_view") or group.third_person_view),
    )


def _select_key_material_view_pair(
    group: ExperimentGroup,
    event: EvidenceEvent,
    views: Sequence[ViewInput],
    infos: dict[str, VideoInfo],
    transforms: dict[str, AlignmentTransform],
    *,
    services: ArtifactsServices,
) -> tuple[tuple[str, str], dict[str, Any]]:
    """Choose real same-role views that contain the event key timestamp.

    Short synchronized recordings can have unequal physical tails.  A group
    representative may therefore be valid for the experiment clip but no
    longer contain a late event key frame.  Select another existing view of
    the same role only when its aligned local timestamp is physically inside
    the decodable source; never clamp the timestamp or synthesize evidence.
    """

    direct = {
        str(item)
        for item in (event.semantic_review or {}).get("directly_supported_view_ids", [])
    }
    supported = {str(item) for item in event.supporting_views}
    by_role = {
        ViewRole.FIRST_PERSON: group.first_person_view,
        ViewRole.THIRD_PERSON: group.third_person_view,
    }
    candidates_receipt: dict[str, list[dict[str, Any]]] = {}
    for role, preferred in by_role.items():
        role_candidates: list[dict[str, Any]] = []
        for view in views:
            view_id = view.view_id
            if view.role != role or view_id not in infos or view_id not in transforms:
                continue
            local_ms = float(transforms[view_id].to_local(event.key_global_ms))
            duration_ms = float(infos[view_id].duration_ms)
            in_bounds = 0.0 <= local_ms <= duration_ms
            margin_ms = min(local_ms, duration_ms - local_ms) if in_bounds else -1.0
            view_candidates = [
                item for item in event.candidates if item.view_id == view_id
            ]
            uncertainty_ms = max(0.0, float(transforms[view_id].uncertainty_ms))
            supported_at_key = view_id in supported and (
                any(
                    item.global_start_ms - uncertainty_ms
                    <= event.key_global_ms
                    <= item.global_end_ms + uncertainty_ms
                    for item in view_candidates
                )
                if event.candidates
                else True
            )
            role_candidates.append(
                {
                    "view_id": view_id,
                    "role": role.value,
                    "local_key_ms": round(local_ms, 6),
                    "duration_ms": round(duration_ms, 6),
                    "in_physical_bounds": in_bounds,
                    "directly_supported": view_id in direct,
                    "candidate_supported": view_id in supported,
                    "candidate_supported_at_key": supported_at_key,
                    "candidate_time_bounds": [
                        [item.global_start_ms, item.global_end_ms]
                        for item in view_candidates
                    ],
                    "group_preferred": view_id == preferred,
                    "physical_margin_ms": round(margin_ms, 6),
                    "stable_object_identities": sorted(
                        services.event_view_stable_identities(event, view_id)
                    ),
                    "stable_actor_track_ids": sorted(
                        services.event_view_actor_identities(event, view_id)
                    ),
                }
            )
        candidates_receipt[role.value] = role_candidates
        if not any(item["in_physical_bounds"] for item in role_candidates):
            raise ValueError(
                f"{event.event_id} has no {role.value} view containing global "
                f"key timestamp {event.key_global_ms:.3f} ms"
            )

    pair_candidates: list[dict[str, Any]] = []
    for first in candidates_receipt[ViewRole.FIRST_PERSON.value]:
        if not first["in_physical_bounds"]:
            continue
        for third in candidates_receipt[ViewRole.THIRD_PERSON.value]:
            if not third["in_physical_bounds"]:
                continue
            first_classes = {str(item[0]) for item in first["stable_object_identities"]}
            third_classes = {str(item[0]) for item in third["stable_object_identities"]}
            shared_identity_classes = sorted(first_classes & third_classes)
            both_action_supported = bool(
                first["candidate_supported_at_key"]
                and third["candidate_supported_at_key"]
            )
            identity_conflict = bool(
                both_action_supported
                and first_classes
                and third_classes
                and not shared_identity_classes
            )
            pair_candidates.append(
                {
                    "first_person_view": first["view_id"],
                    "third_person_view": third["view_id"],
                    "both_views_directly_supported": bool(
                        first["directly_supported"] and third["directly_supported"]
                    ),
                    "direct_support_count": int(first["directly_supported"])
                    + int(third["directly_supported"]),
                    "both_views_candidate_supported": both_action_supported,
                    "candidate_support_count": int(first["candidate_supported"])
                    + int(third["candidate_supported"]),
                    "key_time_support_count": int(first["candidate_supported_at_key"])
                    + int(third["candidate_supported_at_key"]),
                    "shared_identity_classes": shared_identity_classes,
                    "stable_identity_available_in_both_views": bool(
                        first_classes and third_classes
                    ),
                    "actor_identity_available_in_both_views": bool(
                        first["stable_actor_track_ids"]
                        and third["stable_actor_track_ids"]
                    ),
                    "identity_conflict": identity_conflict,
                    "group_preferred_pair": bool(
                        first["group_preferred"] and third["group_preferred"]
                    ),
                    "minimum_physical_margin_ms": min(
                        float(first["physical_margin_ms"]),
                        float(third["physical_margin_ms"]),
                    ),
                }
            )
    eligible_pairs = [item for item in pair_candidates if not item["identity_conflict"]]
    if not eligible_pairs:
        raise ValueError(
            f"{event.event_id} has no identity-compatible first/third-person "
            f"view pair at {event.key_global_ms:.3f} ms"
        )
    winner = max(
        eligible_pairs,
        key=lambda item: (
            bool(item["both_views_directly_supported"]),
            int(item["direct_support_count"]),
            bool(item["both_views_candidate_supported"]),
            int(item["key_time_support_count"]),
            bool(item["group_preferred_pair"]),
            int(item["candidate_support_count"]),
            bool(item["shared_identity_classes"]),
            bool(item["stable_identity_available_in_both_views"]),
            bool(item["actor_identity_available_in_both_views"]),
            bool(item["group_preferred_pair"]),
            float(item["minimum_physical_margin_ms"]),
            str(item["first_person_view"]),
            str(item["third_person_view"]),
        ),
    )
    pair = (str(winner["first_person_view"]), str(winner["third_person_view"]))
    receipt = {
        "schema_version": "visioncortex-key-material-view-selection/1",
        "policy": "same-role real source containing aligned key timestamp",
        "timestamp_clamped": False,
        "synthetic_cross_view_evidence": False,
        "key_global_ms": float(event.key_global_ms),
        "first_person_view": pair[0],
        "third_person_view": pair[1],
        "group_first_person_view": group.first_person_view,
        "group_third_person_view": group.third_person_view,
        "fallback_applied": pair != (group.first_person_view, group.third_person_view),
        "candidates": candidates_receipt,
        "pair_candidates": pair_candidates,
        "selected_pair_identity": {
            key: value
            for key, value in winner.items()
            if key
            not in {
                "first_person_view",
                "third_person_view",
                "minimum_physical_margin_ms",
                "group_preferred_pair",
            }
        },
        "pair_evidence_status": (
            "directly_supported"
            if winner["both_views_directly_supported"]
            else "candidate_cooccurrence_unverified"
            if winner["both_views_candidate_supported"]
            else "context_only_missing_key_time_support"
        ),
        "same_action_pair_verified": bool(winner["both_views_directly_supported"]),
        "identity_policy": "shared classes and local track IDs do not establish cross-view physical identity",
    }
    return pair, receipt


def _select_key_material_view_pair_with_peak_fallback(
    group: ExperimentGroup,
    event: EvidenceEvent,
    views: Sequence[ViewInput],
    infos: dict[str, VideoInfo],
    transforms: dict[str, AlignmentTransform],
    accepted_peak_global_ms: float,
    *,
    services: ArtifactsServices,
) -> tuple[tuple[str, str], dict[str, Any]]:
    """Restore the accepted CV peak if a re-ranked frame lacks dual coverage."""

    selected_key_global_ms = float(event.key_global_ms)
    try:
        return services._select_key_material_view_pair(
            group, event, views, infos, transforms
        )
    except ValueError as selected_error:
        accepted_peak_global_ms = float(accepted_peak_global_ms)
        if (
            accepted_peak_global_ms == selected_key_global_ms
            or not event.global_start_ms
            <= accepted_peak_global_ms
            <= event.global_end_ms
        ):
            raise
        event.key_global_ms = accepted_peak_global_ms
        try:
            pair, receipt = services._select_key_material_view_pair(
                group, event, views, infos, transforms
            )
        except ValueError as fallback_error:
            event.key_global_ms = selected_key_global_ms
            raise selected_error from fallback_error
        receipt["key_timestamp_fallback"] = {
            "applied": True,
            "reason": (
                "participant-ranked frame lacked real dual-role physical "
                "coverage; restored accepted CV event peak"
            ),
            "participant_ranked_global_ms": selected_key_global_ms,
            "accepted_event_peak_global_ms": accepted_peak_global_ms,
            "timestamp_clamped": False,
        }
        return pair, receipt


def _artifact_json(
    group: ExperimentGroup,
    event: EvidenceEvent,
    artifact_type: str,
    artifact_file: str,
    view_id: str | None,
    transforms: dict[str, AlignmentTransform],
    archive_id: str | None = None,
    *,
    services: ArtifactsServices,
) -> dict[str, Any]:
    first_material_view, third_material_view = services._key_material_view_pair(
        group, event
    )
    liquid_state = event.liquid_state
    understanding = event.model_understanding or {}
    physical_change = understanding.get("physical_change") or {}
    before_state = str(physical_change.get("before") or "unknown")
    after_state = str(physical_change.get("after") or "unknown")
    semantic_review = event.semantic_review or {}
    final_participants = [
        str(item)
        for item in semantic_review.get("final_participant_objects") or []
        if str(item).strip()
    ]
    # The top-level normalized contract describes the curated event only.
    # Background/model-context objects and the rejected CV hypothesis remain
    # available in provenance, but must never leak back into the final object
    # slots or participant-only presentation.
    object_names = list(dict.fromkeys(final_participants or event.objects))
    material_name = _key_material_semantic_name(event)

    def contains(name: str, keywords: tuple[str, ...]) -> bool:
        lowered = name.lower()
        return any(keyword in lowered for keyword in keywords)

    def first_object(
        keywords: tuple[str, ...], excluded: set[str] | None = None
    ) -> str | None:
        excluded = excluded or set()
        return next(
            (
                name
                for name in object_names
                if name not in excluded and contains(name, keywords)
            ),
            None,
        )

    actor = first_object(("gloved_hand", "hand", "手套", "手"))
    closure = first_object(("bottle_cap", "tube_cap", "瓶盖", "管盖"))
    tool = first_object(
        (
            "pipette",
            "spearhead",
            "spatula",
            "dropper",
            "移液",
            "吸头",
            "药勺",
            "滴管",
        )
    )
    source = first_object(
        ("reagent_bottle", "bottle", "试剂瓶", "试剂容器", "瓶"),
        {item for item in (tool, closure) if item},
    )
    target = first_object(
        (
            "tube",
            "paper",
            "balance",
            "beaker",
            "flask",
            "离心管",
            "称量纸",
            "天平",
            "烧杯",
            "锥形瓶",
        ),
        {item for item in (tool, source) if item},
    )

    def tracked_id(name: str | None, fallback: str) -> str:
        if not name:
            return "unknown"
        slug = services._safe_slug(name).lower()
        if slug == "unknown":
            known = {
                "移液器": "pipette",
                "移液枪": "pipette",
                "枪头": "pipette-tip",
                "吸头": "pipette-tip",
                "药勺": "spatula",
                "称量纸": "weighing-paper",
                "试剂瓶": "reagent-bottle",
                "离心管": "centrifuge-tube",
                "分析天平": "analytical-balance",
            }
            slug = next(
                (value for key, value in known.items() if key in name), fallback
            )
        track_id = None
        for candidate in event.candidates:
            if name not in candidate.objects:
                continue
            for evidence in candidate.evidence:
                for key in (
                    "object_track_id",
                    "track_id",
                    "tool_track_id",
                    "container_track_id",
                ):
                    if evidence.get(key) is not None:
                        track_id = evidence[key]
                        break
                if track_id is not None:
                    break
            if track_id is not None:
                break
        return (
            f"{slug}-{int(track_id):03d}"
            if track_id is not None
            else f"{slug}-unresolved"
        )

    normalized_action = {
        "liquid_movement": "liquid_transfer",
    }.get(event.action_type.value, event.action_type.value)
    raw_state_receipt = event.state_machine or {}
    raw_state_action = str(raw_state_receipt.get("action_type") or "")
    state_receipt = raw_state_receipt if raw_state_action == normalized_action else {}
    combined = " ".join(object_names).lower()
    if normalized_action in {"liquid_transfer", "pipette_transfer_operation"}:
        action_subtype = (
            "pipette_transfer"
            if any(
                keyword in combined
                for keyword in ("pipette", "spearhead", "移液", "吸头")
            )
            else "container_pour"
        )
        phases = [
            "source_approach",
            "source_contact",
            "source_withdraw",
            "transport",
            "target_contact",
            "target_release",
        ]
    elif normalized_action == "container_state_change":
        action_subtype = "container_open_close"
        phases = [
            "container_approach",
            "container_contact",
            "state_transition",
            "release",
        ]
    elif normalized_action == "device_panel_operation":
        action_subtype = "device_control_operation"
        phases = ["panel_approach", "panel_contact", "control_activation", "withdraw"]
    elif normalized_action == "object_movement":
        action_subtype = "tool_movement" if tool else "object_relocation"
        phases = ["object_approach", "grasp", "lift", "transport", "place", "release"]
    else:
        action_subtype = "tool_contact" if tool else "generic_hand_object_contact"
        phases = ["object_approach", "contact", "manipulation", "release"]
    action_subtype = str(state_receipt.get("action_subtype") or action_subtype)
    phases = list(state_receipt.get("phases") or phases)

    observations = []
    for index, item in enumerate(understanding.get("per_view_observations") or [], 1):
        observation_view = str(item.get("view_id") or "unknown")
        observations.append(
            {
                "observation_id": f"{event.event_id}-obs-{index:02d}",
                "view_id": observation_view,
                "view_role": (
                    "first_person"
                    if observation_view == first_material_view
                    else "third_person"
                    if observation_view == third_material_view
                    else "unknown"
                ),
                "timestamp_us": round(event.key_global_ms * 1000.0),
                "observed_fact": str(item.get("observation") or ""),
            }
        )
    if not observations:
        observations = [
            {
                "observation_id": f"{event.event_id}-obs-{index:02d}",
                "view_id": supported_view,
                "view_role": (
                    "first_person"
                    if supported_view == first_material_view
                    else "third_person"
                ),
                "timestamp_us": round(event.key_global_ms * 1000.0),
                "observed_fact": f"CV accepted {event.action_type.value}: {', '.join(event.objects)}",
            }
            for index, supported_view in enumerate(event.supporting_views, 1)
        ]
    if liquid_state is not None:
        for index, item in enumerate(
            liquid_state.per_view_observations, len(observations) + 1
        ):
            facts = item.observed_facts or [
                f"liquid_present={item.liquid_present}; "
                f"fill_ratio={item.fill_ratio}; visible_flow={item.visible_flow}"
            ]
            observations.extend(
                {
                    "observation_id": f"{event.event_id}-obs-{index:02d}-{fact_index:02d}",
                    "view_id": item.view_id,
                    "view_role": item.view_role.value,
                    "timestamp_us": item.timestamp_us,
                    "observed_fact": fact,
                }
                for fact_index, fact in enumerate(facts, 1)
            )

    alignment_uncertainty_us = max(
        80_000,
        round(
            max(
                (
                    max(
                        float(transforms[item].uncertainty_ms),
                        float(transforms[item].csv_rmse_ms or 0.0),
                    )
                    for item in (first_material_view, third_material_view)
                    if item in transforms
                ),
                default=80.0,
            )
            * 1000.0
        ),
    )
    cross_view_associations = [
        {
            "association_id": f"{event.event_id}-first-third",
            "source_view_id": first_material_view,
            "target_view_id": third_material_view,
            "global_timestamp_us": round(event.key_global_ms * 1000.0),
            "source_local_timestamp_us": round(
                transforms[first_material_view].to_local(event.key_global_ms) * 1000.0
            ),
            "target_local_timestamp_us": round(
                transforms[third_material_view].to_local(event.key_global_ms) * 1000.0
            ),
            "time_uncertainty_us": alignment_uncertainty_us,
            "consistency": understanding.get("cross_view_consistency", "unreviewed"),
            "both_views_support_action": bool(
                (event.observability.get("key_material_view_selection") or {}).get(
                    "same_action_pair_verified", False
                )
            )
            and all(
                item in event.supporting_views
                for item in (first_material_view, third_material_view)
            ),
        }
    ]

    current_step = str(understanding.get("current_step") or "")
    next_step = str(understanding.get("next_step") or "")
    next_step_evidence = dict(understanding.get("next_step_evidence") or {})
    next_step_status = str(next_step_evidence.get("status") or "unknown")
    if next_step_status not in {"observed", "inferred", "unknown"}:
        next_step_status = "unknown"
    uncertainties = list(
        dict.fromkeys(
            [
                *[str(item) for item in event.uncertainty],
                *[
                    f"CV未直接观察: {item}"
                    for item in (
                        event.observability.get("unmet_visual_requirements") or []
                        if event.observability
                        else []
                    )
                ],
                *[str(item) for item in understanding.get("uncertainties") or []],
            ]
        )
    )
    if liquid_state is not None:
        uncertainties = list(
            dict.fromkeys([*uncertainties, *liquid_state.uncertain_claims])
        )
    consistency = str(understanding.get("cross_view_consistency") or "unreviewed")
    model_confidence = float(understanding.get("confidence") or 0.0)
    confirmed_action = str(understanding.get("action_type_confirmed") or "unknown")
    status = "confirmed"
    if understanding.get("status") != "completed":
        status = "provisional_cv_only"
    elif (
        confirmed_action == "unknown"
        or consistency == "conflict"
        or model_confidence < 0.55
    ):
        status = "uncertain"
    if state_receipt.get("lifecycle_state") == "incomplete_end":
        status = "incomplete_observation"

    role_for = (
        "aligned_first_third"
        if view_id is None
        else "first_person"
        if view_id == first_material_view
        else "third_person"
    )
    event_uid = (
        services.stable_event_uid(
            archive_id,
            group.group_uid or group.group_id,
            event.event_id,
        )
        if archive_id
        else None
    )
    key_frames = [
        {
            "view_id": item,
            "view_role": (
                "aligned_first_third"
                if item == "aligned_first_third"
                else "first_person"
                if item == first_material_view
                else "third_person"
            ),
            "timestamp_us": round(event.key_global_ms * 1000.0),
            "path": path,
            **(
                {
                    "artifact_uid": services.stable_artifact_uid(
                        event_uid, "key_frame", item
                    ),
                    "sidecar_path": str(Path(path).with_suffix(".json")).replace(
                        "\\", "/"
                    ),
                }
                if event_uid
                else {}
            ),
        }
        for item, path in event.key_frames.items()
    ]
    key_clips = [
        {
            "view_id": item,
            "view_role": (
                "aligned_first_third"
                if item == "aligned_first_third"
                else "first_person"
                if item == first_material_view
                else "third_person"
            ),
            "start_us": round(max(0.0, event.global_start_ms - 2000.0) * 1000.0),
            "end_us": round((event.global_end_ms + 3000.0) * 1000.0),
            "path": path,
            **(
                {
                    "artifact_uid": services.stable_artifact_uid(
                        event_uid, "key_clip", item
                    ),
                    "sidecar_path": str(Path(path).with_suffix(".json")).replace(
                        "\\", "/"
                    ),
                }
                if event_uid
                else {}
            ),
        }
        for item, path in event.key_clips.items()
    ]
    candidate_ids = [candidate.candidate_id for candidate in event.candidates]
    frame_evidence_ids = [
        f"{candidate.view_id}:frame-{item['frame_index']}"
        for candidate in event.candidates
        for item in candidate.evidence
        if item.get("frame_index") is not None
    ]
    evidence_ids = list(dict.fromkeys([*candidate_ids, *frame_evidence_ids]))
    observed_facts = [item for item in [current_step] if item]
    observed_facts.extend(
        item["observed_fact"] for item in observations if item.get("observed_fact")
    )
    supported_inferences = []
    if next_step and next_step not in {"未知", "unknown", "不确定"}:
        if next_step_status == "observed":
            observed_facts.append(f"已观察后续动作：{next_step}")
        elif next_step_status == "inferred":
            supported_inferences.append(f"预测下一步：{next_step}")
    if liquid_state is not None:
        observed_facts = list(
            dict.fromkeys([*observed_facts, *liquid_state.observed_facts])
        )
        supported_inferences = list(
            dict.fromkeys([*supported_inferences, *liquid_state.supported_inferences])
        )
    contradictions = []
    if consistency == "conflict":
        contradictions.append("第一人称与第三人称观察发生冲突，详见 observations")

    cross_view_score = {
        "consistent": 0.95,
        "partial": 0.65,
        "conflict": 0.20,
        "single_view": 0.40,
    }.get(consistency, 0.35)
    change_observed = before_state != "unknown" and after_state != "unknown"
    action_agrees = confirmed_action in {event.action_type.value, normalized_action}

    def liquid_container_state(value: Any, fallback: str) -> str:
        if value is None:
            return fallback
        fragments = []
        if value.liquid_present is not None:
            fragments.append(f"liquid_present={str(value.liquid_present).lower()}")
        if value.fill_ratio is not None:
            fragments.append(f"fill_ratio={value.fill_ratio:.4f}")
        return "; ".join(fragments) or fallback

    source_before_state = liquid_container_state(
        liquid_state.source_before if liquid_state else None, "unknown"
    )
    source_after_state = liquid_container_state(
        liquid_state.source_after if liquid_state else None, "unknown"
    )
    target_before_state = liquid_container_state(
        liquid_state.target_before if liquid_state else None, "unknown"
    )
    target_after_state = liquid_container_state(
        liquid_state.target_after if liquid_state else None, "unknown"
    )
    state_change_score = (
        liquid_state.confidence
        if liquid_state is not None and liquid_state.state_change_confirmed is True
        else model_confidence
        if change_observed
        else 0.25
    )
    if normalized_action == ActionType.CONTAINER_STATE_CHANGE.value:
        normalized_objects = {
            "actor": tracked_id(actor, "actor"),
            "container": tracked_id(source, "container"),
            "closure": tracked_id(closure, "closure"),
        }
        receipt_before = str(
            (state_receipt.get("state_before") or {}).get("container") or "unknown"
        )
        receipt_after = str(
            (state_receipt.get("state_after") or {}).get("container") or "unknown"
        )
        normalized_state_before = {
            "container": before_state if before_state != "unknown" else receipt_before
        }
        normalized_state_after = {
            "container": after_state if after_state != "unknown" else receipt_after
        }
    elif normalized_action in {
        ActionType.HAND_OBJECT_CONTACT.value,
        ActionType.DEVICE_PANEL_OPERATION.value,
    }:
        normalized_objects = {
            "actor": tracked_id(actor, "actor"),
            "target": tracked_id(target, "target"),
        }
        normalized_state_before = {
            "target": before_state,
            **dict(state_receipt.get("state_before") or {}),
        }
        normalized_state_after = {
            "target": after_state,
            **dict(state_receipt.get("state_after") or {}),
        }
    else:
        normalized_objects = {
            "tool": tracked_id(tool, "tool"),
            "source": tracked_id(source, "source"),
            "target": tracked_id(target, "target"),
        }
        normalized_state_before = {
            "tool": before_state,
            "source": source_before_state,
            "target": target_before_state,
            **dict(state_receipt.get("state_before") or {}),
        }
        normalized_state_after = {
            "tool": after_state,
            "source": source_after_state,
            "target": target_after_state,
            **dict(state_receipt.get("state_after") or {}),
        }

    pre_curation_action = str(
        semantic_review.get("pre_curation_action_type") or event.action_type.value
    )
    pre_curation_objects = list(
        semantic_review.get("pre_curation_objects") or event.objects
    )
    pre_curation_state_machine = (
        semantic_review.get("pre_curation_state_machine") or raw_state_receipt
    )

    return {
        "event_id": event.event_id,
        "parent_event_id": group.group_id,
        "parent_event_uid": group.group_uid or group.group_id,
        "actor_id": "operator-01",
        "workstation_id": (
            "weighing-station-01"
            if any(keyword in combined for keyword in ("balance", "天平"))
            else "pipetting-station-01"
            if any(
                keyword in combined
                for keyword in (
                    "pipette",
                    "spearhead",
                    "tube",
                    "移液",
                    "吸头",
                    "离心管",
                )
            )
            else "wet-lab-bench-01"
        ),
        "action_type": normalized_action,
        "action_subtype": action_subtype,
        "start_us": round(event.global_start_ms * 1000.0),
        "end_us": round(event.global_end_ms * 1000.0),
        "peak_timestamp_us": round(event.key_global_ms * 1000.0),
        "time_uncertainty_us": alignment_uncertainty_us,
        "phases": phases,
        "objects": normalized_objects,
        "state_before": normalized_state_before,
        "state_after": normalized_state_after,
        "observations": observations,
        "cross_view_associations": cross_view_associations,
        "decision": {
            "status": status,
            "observed_facts": observed_facts,
            "supported_inferences": supported_inferences,
            "uncertain_claims": uncertainties,
            "contradictions": contradictions,
        },
        "scores": {
            "temporal_continuity": round(
                min(1.0, 0.5 + float(event.confidence) * 0.5), 4
            ),
            "object_identity": round(
                min(
                    1.0,
                    0.35 + min(len(event.objects), 4) * 0.08 + model_confidence * 0.3,
                ),
                4,
            ),
            "phase_completeness": round(
                (0.45 + model_confidence * 0.5)
                if change_observed
                else (0.25 + model_confidence * 0.35),
                4,
            ),
            "cross_view_support": cross_view_score,
            "state_change_support": round(state_change_score, 4),
            "model_agreement": round(
                (float(event.confidence) + model_confidence) / 2.0
                if action_agrees
                else model_confidence * 0.5,
                4,
            ),
            "contradiction_penalty": 0.8 if consistency == "conflict" else 0.0,
            **dict(state_receipt.get("scores") or {}),
        },
        "key_frames": key_frames,
        "key_clips": key_clips,
        "evidence_ids": evidence_ids,
        "provenance": {
            **(
                {"liquid_state": liquid_state.model_dump(mode="json")}
                if liquid_state is not None
                else {}
            ),
            "schema_version": "key-material-event-v1.0.0",
            "time_base": "aligned_global_timeline_microseconds",
            "experiment_group_id": group.group_id,
            "experiment_name": group.experiment_name,
            "continuity_type": group.continuity_type,
            "atomic_experiment_ids": group.atomic_experiment_ids,
            "archive_classification": {
                "hierarchy_version": "2.0.0",
                "experiment_folder": (
                    group.archive_folder or services._safe_folder_name(group.group_id)
                ),
                "action_type": event.action_type.value,
                "action_category_folder": key_material_action_folder(event.action_type),
                "semantic_file_stem": material_name["file_stem"],
                "primary_object": material_name["primary_object"],
                "object_labels": material_name["object_labels"],
            },
            "sidecar_for": {
                "artifact_type": artifact_type,
                "artifact_file": artifact_file,
                "view_id": view_id,
                "view_role": role_for,
            },
            "frame_sources": {
                key: value
                for key, value in (
                    event.observability.get("key_frame_sources") or {}
                ).items()
                if view_id is None or key == view_id
            }
            if artifact_type in {"key_frame", "aligned_first_third_key_frame"}
            else {},
            "cv": {
                "action_type": pre_curation_action,
                "objects": pre_curation_objects,
                "confidence": event.confidence,
                "audit_reason": event.audit_reason,
                "supporting_views": list(
                    semantic_review.get("pre_curation_supporting_views")
                    or event.supporting_views
                ),
                "observability": event.observability,
                "semantic_review": event.semantic_review,
                "continuous_state_machine": pre_curation_state_machine,
            },
            "semantic_final": {
                "action_type": event.action_type.value,
                "participant_objects": object_names,
                "supporting_views": event.supporting_views,
                "continuous_state_machine": state_receipt,
            },
            "mllm": {
                "model": understanding.get("model"),
                "status": understanding.get("status"),
                "usage": understanding.get("usage") or {},
                "latency_seconds": understanding.get("latency_seconds"),
                "attempts": understanding.get("attempts"),
                "current_step": current_step or None,
                "next_step": next_step or None,
                "next_step_status": next_step_status,
                "next_step_evidence": next_step_evidence,
            },
            "alignment": {
                item: transforms[item].model_dump(mode="json")
                for item in (first_material_view, third_material_view)
            },
            "score_method": "deterministic_cv_mllm_evidence_mapping_v1",
            **(
                {
                    "index": {
                        "event_uid": event_uid,
                        "database": "JSON-Config-Files/evidence_index.sqlite",
                        "artifact_registry": "JSON-Config-Files/artifact_registry.jsonl",
                        "evidence_registry": "JSON-Config-Files/evidence_registry.jsonl",
                        "decision_receipt_registry": "JSON-Config-Files/decision_receipt_registry.jsonl",
                        "evidence_refs": [
                            {
                                "evidence_id": evidence_id,
                                "evidence_uid": services.stable_evidence_uid(
                                    event_uid, evidence_id
                                ),
                            }
                            for evidence_id in evidence_ids
                        ],
                    }
                }
                if event_uid
                else {}
            ),
        },
    }


def prepare_key_material_category_layout(
    layout: ArchiveLayout,
    groups: Sequence[ExperimentGroup],
    events: Sequence[EvidenceEvent] | None = None,
    *,
    include_empty_categories: bool = True,
    services: ArtifactsServices,
) -> None:
    """Create the stable experiment/action hierarchy before materialization."""

    event_by_id = {event.event_id: event for event in (events or []) if event.accepted}
    for group in groups:
        experiment_folder = group.archive_folder or services._safe_folder_name(
            group.group_id
        )
        observed_actions = {
            event_by_id[event_id].action_type.value
            for event_id in group.key_event_ids
            if event_id in event_by_id
        }
        for action_type, action_folder in ACTION_CATEGORY_FOLDERS.items():
            if (
                events is not None
                and not include_empty_categories
                and action_type not in observed_actions
            ):
                continue
            (layout.key_frames / experiment_folder / action_folder).mkdir(
                parents=True, exist_ok=True
            )
            (layout.key_clips / experiment_folder / action_folder).mkdir(
                parents=True, exist_ok=True
            )


def write_key_material_category_index(
    layout: ArchiveLayout,
    groups: Sequence[ExperimentGroup],
    events: Sequence[EvidenceEvent],
    publisher: Any | None = None,
    *,
    include_empty_categories: bool = True,
    services: ArtifactsServices,
) -> Path:
    """Write a human-browsable and machine-indexable six-category manifest."""

    event_by_id = {event.event_id: event for event in events if event.accepted}
    experiments: list[dict[str, Any]] = []
    for group in groups:
        experiment_folder = group.archive_folder or services._safe_folder_name(
            group.group_id
        )
        group_events = [
            event_by_id[event_id]
            for event_id in group.key_event_ids
            if event_id in event_by_id
        ]
        categories: list[dict[str, Any]] = []
        for action_type, action_folder in ACTION_CATEGORY_FOLDERS.items():
            category_events = [
                event
                for event in group_events
                if event.action_type.value == action_type
            ]
            frame_category_folder = (
                layout.key_frames / experiment_folder / action_folder
            )
            clip_category_folder = layout.key_clips / experiment_folder / action_folder
            category_materialized = bool(category_events or include_empty_categories)
            category_summary = {
                "schema_version": "visioncortex-key-material-category/1.0.0",
                "group_id": group.group_id,
                "experiment_name": group.experiment_name,
                "experiment_name_en": group.experiment_name_en,
                "experiment_folder": experiment_folder,
                "action_type": action_type,
                "action_category_folder": action_folder,
                "event_count": len(category_events),
                "coverage_status": ("observed" if category_events else "not_observed"),
                "absence_reason": (
                    None
                    if category_events
                    else (
                        "No accepted evidence event of this action type was "
                        "observed in the bounded experiment."
                    )
                ),
                "event_ids": [event.event_id for event in category_events],
                "events": [
                    {
                        "event_id": event.event_id,
                        "event_folder": services._key_material_event_folder_name(
                            layout, experiment_folder, event
                        ),
                        "material_name": _key_material_semantic_name(event),
                    }
                    for event in category_events
                ],
            }
            frame_summary_path = frame_category_folder / "Category.json"
            clip_summary_path = clip_category_folder / "Category.json"
            if category_materialized:
                frame_category_folder.mkdir(parents=True, exist_ok=True)
                clip_category_folder.mkdir(parents=True, exist_ok=True)
                services.write_json(
                    frame_summary_path,
                    {**category_summary, "media_kind": "key_frame"},
                )
                services.write_json(
                    clip_summary_path,
                    {**category_summary, "media_kind": "key_clip"},
                )
                if publisher is not None:
                    publisher.publish_file(frame_summary_path)
                    publisher.publish_file(clip_summary_path)
            else:
                # A later semantic or visual-quality pass can quarantine every
                # event that initially populated a category.  Remove the stale
                # summary and its now-empty directory so the user-facing NAS
                # tree never claims that a JSON-only category contains media.
                for summary_path, category_folder in (
                    (frame_summary_path, frame_category_folder),
                    (clip_summary_path, clip_category_folder),
                ):
                    summary_path.unlink(missing_ok=True)
                    try:
                        category_folder.rmdir()
                    except OSError:
                        # Preserve any unexpected material for audit instead of
                        # deleting a non-empty directory here.
                        pass
            categories.append(
                {
                    "action_type": action_type,
                    "folder": action_folder if category_materialized else None,
                    "event_count": len(category_events),
                    "coverage_status": category_summary["coverage_status"],
                    "absence_reason": category_summary["absence_reason"],
                    "materialized": category_materialized,
                    "key_frames_folder": (
                        Path("Key-Materials")
                        / "Key-Frames"
                        / experiment_folder
                        / action_folder
                    ).as_posix()
                    if category_materialized
                    else None,
                    "key_clips_folder": (
                        Path("Key-Materials")
                        / "Key-Clips"
                        / experiment_folder
                        / action_folder
                    ).as_posix()
                    if category_materialized
                    else None,
                    "key_frames_category_summary": services._relative(
                        frame_summary_path, layout.root
                    )
                    if category_materialized
                    else None,
                    "key_clips_category_summary": services._relative(
                        clip_summary_path, layout.root
                    )
                    if category_materialized
                    else None,
                    "events": [
                        {
                            "event_id": event.event_id,
                            "event_folder": services._key_material_event_folder_name(
                                layout, experiment_folder, event
                            ),
                            "material_name": _key_material_semantic_name(event),
                            "peak_timestamp_us": round(event.key_global_ms * 1000.0),
                            "key_frames": dict(event.key_frames),
                            "key_clips": dict(event.key_clips),
                        }
                        for event in category_events
                    ],
                }
            )
        experiments.append(
            {
                "group_id": group.group_id,
                "experiment_name": group.experiment_name,
                "experiment_name_en": group.experiment_name_en,
                "experiment_folder": experiment_folder,
                "key_event_count": len(group_events),
                "action_categories": categories,
            }
        )
    path = layout.key_materials / "Key-Material-Category-Index.json"
    services.write_json(
        path,
        {
            "schema_version": "visioncortex-key-material-category-index/1.0.0",
            "hierarchy": (
                "Key-Materials/{Key-Frames|Key-Clips}/"
                "{Experiment}/{Action-Category}/{Event}"
            ),
            "category_count": len(ACTION_CATEGORY_FOLDERS),
            "categories": [
                {
                    "order": index,
                    "action_type": action_type,
                    "folder": action_folder,
                }
                for index, (action_type, action_folder) in enumerate(
                    ACTION_CATEGORY_FOLDERS.items(), 1
                )
            ],
            "experiments": experiments,
        },
    )
    if publisher is not None:
        publisher.publish_file(path)
    return path


def refresh_key_material_metadata(
    layout: ArchiveLayout,
    events: Sequence[EvidenceEvent],
    groups: Sequence[ExperimentGroup],
    transforms: dict[str, AlignmentTransform],
    archive_id: str | None = None,
    *,
    services: ArtifactsServices,
) -> None:
    """Rewrite sidecars after MLLM so JSON and media never disagree."""
    group_by_event = {
        event_id: group for group in groups for event_id in group.key_event_ids
    }
    for event in events:
        group = group_by_event[event.event_id]
        for collection, artifact in (
            (event.key_frames, "key_frame"),
            (event.key_clips, "key_clip"),
        ):
            for view_id, relative in collection.items():
                media = layout.root / relative
                json_path = media.with_suffix(".json")
                aligned = view_id == "aligned_first_third"
                services.write_json(
                    json_path,
                    services._artifact_json(
                        group,
                        event,
                        f"aligned_first_third_{artifact}" if aligned else artifact,
                        relative,
                        None if aligned else view_id,
                        transforms,
                        archive_id,
                    ),
                )
