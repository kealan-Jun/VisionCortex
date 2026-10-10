from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Mapping

from .schemas import ActionCandidate, EvidenceEvent, ExperimentSegment


OBJECT_FAMILY_ALIASES = {
    "sample_bottle_blue": "sample_bottle",
    "reagent_bottle_open": "reagent_bottle",
}


def canonical_object_families(objects: Iterable[str]) -> tuple[str, ...]:
    return tuple(
        sorted({OBJECT_FAMILY_ALIASES.get(str(item), str(item)) for item in objects})
    )


def _ms(value: float) -> float:
    return round(float(value), 6)


def _fingerprint(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def stable_candidate_fingerprint(candidate: ActionCandidate) -> str:
    evidence = sorted(
        json.dumps(
            item,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        for item in candidate.evidence
    )
    return _fingerprint(
        {
            "action_type": candidate.action_type.value,
            "view_id": candidate.view_id,
            "role": candidate.role.value,
            "global_start_ms": _ms(candidate.global_start_ms),
            "global_end_ms": _ms(candidate.global_end_ms),
            "key_global_ms": _ms(candidate.key_global_ms),
            "object_families": canonical_object_families(candidate.objects),
            "confidence": round(float(candidate.confidence), 9),
            "evidence": evidence,
        }
    )


def candidate_sort_key(candidate: ActionCandidate) -> tuple[Any, ...]:
    return (
        _ms(candidate.global_start_ms),
        _ms(candidate.global_end_ms),
        candidate.action_type.value,
        canonical_object_families(candidate.objects),
        candidate.view_id,
        candidate.role.value,
        _ms(candidate.key_global_ms),
        stable_candidate_fingerprint(candidate),
        candidate.candidate_id,
    )


def stable_event_fingerprint(event: EvidenceEvent) -> str:
    if event.event_fingerprint:
        return event.event_fingerprint
    return _fingerprint(
        {
            "action_type": event.action_type.value,
            "global_start_ms": _ms(event.global_start_ms),
            "global_end_ms": _ms(event.global_end_ms),
            "key_global_ms": _ms(event.key_global_ms),
            "object_families": canonical_object_families(event.objects),
            "supporting_views": sorted(event.supporting_views),
            "supporting_roles": sorted(role.value for role in event.supporting_roles),
            "accepted": bool(event.accepted),
            "formal_admission_status": event.formal_admission_status,
            "candidate_fingerprints": sorted(
                stable_candidate_fingerprint(candidate)
                for candidate in event.candidates
            ),
        }
    )


def event_sort_key(event: EvidenceEvent) -> tuple[Any, ...]:
    return (
        _ms(event.global_start_ms),
        _ms(event.global_end_ms),
        event.action_type.value,
        canonical_object_families(event.objects),
        tuple(sorted(event.supporting_views)),
        _ms(event.key_global_ms),
        stable_event_fingerprint(event),
        event.event_id,
    )


def stable_segment_fingerprint(
    segment: ExperimentSegment,
    events_by_id: Mapping[str, EvidenceEvent],
) -> str:
    return _fingerprint(
        {
            "global_start_ms": _ms(segment.global_start_ms),
            "global_end_ms": _ms(segment.global_end_ms),
            "participating_views": sorted(segment.participating_views),
            "event_fingerprints": sorted(
                stable_event_fingerprint(events_by_id[event_id])
                for event_id in segment.event_ids
                if event_id in events_by_id
            ),
        }
    )


def stable_segment_uid(
    segment: ExperimentSegment,
    events_by_id: Mapping[str, EvidenceEvent],
) -> str:
    """Return an immutable segment identity independent of display ordering."""

    return f"SEG-{stable_segment_fingerprint(segment, events_by_id)[:20]}"


def stable_group_uid(
    segments: Iterable[ExperimentSegment],
    events_by_id: Mapping[str, EvidenceEvent],
) -> str:
    """Return an immutable group identity from its ordered atomic membership."""

    fingerprints = [
        stable_segment_fingerprint(segment, events_by_id)
        for segment in sorted(
            segments,
            key=lambda item: segment_sort_key(item, events_by_id),
        )
    ]
    return f"GRP-{_fingerprint({'segment_fingerprints': fingerprints})[:20]}"


def segment_sort_key(
    segment: ExperimentSegment,
    events_by_id: Mapping[str, EvidenceEvent],
) -> tuple[Any, ...]:
    return (
        _ms(segment.global_start_ms),
        _ms(segment.global_end_ms),
        stable_segment_fingerprint(segment, events_by_id),
        segment.segment_id,
    )
