"""Fine-scan routing rules and coverage plans."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..analysis.windows import merge_time_windows
from ..decisions import decision_receipt
from ..grouping import is_experiment_start_anchor
from ..ordering import candidate_sort_key, event_sort_key
from ..schemas import (
    ActionCandidate,
    ActionType,
    EvidenceEvent,
    ExperimentGroup,
    ExperimentSegment,
    ViewInput,
    ViewRole,
    event_is_formal,
)


class FineScanPlanner:
    """Deterministic coverage and recall planning, with no scanner side effects."""

    def __init__(self, config: dict[str, Any]):
        self.config = config

    @staticmethod
    def progressive_fine_view_order(
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
                -int(
                    fine_view_report.get(view.view_id, {}).get(
                        "active_anchor_frames", 0
                    )
                ),
                -int(fine_view_report.get(view.view_id, {}).get("anchor_frames", 0)),
                positions[view.view_id],
            ),
        )
        initial_count = min(len(third), max(0, int(initial_third_person_views)))
        return first + third[:initial_count], third[initial_count:]

    @staticmethod
    def progressive_candidate_recall_bounds(
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
    def quarantine_nonformal_progressive_gaps(
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

    def progressive_target_status(
        self,
        boundary_candidates: list[ActionCandidate],
        events: list[EvidenceEvent],
    ) -> list[dict[str, Any]]:
        """Classify which recalled windows still need another third-person view.

        A supplemental scan is demanded only by reliable first-person evidence.
        A window is covered only by an accepted, boundary-eligible event with
        both roles. This deliberately prevents a partial liquid hypothesis
        with an inconsistent frame count from suppressing a needed supplemental scan.
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
                self.progressive_candidate_recall_bounds(candidate)
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
                event
                for event in relevant
                if ViewRole.FIRST_PERSON in event.supporting_roles
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
                    and event.global_start_ms
                    - max(item.global_end_ms for item in clusters[-1])
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
                        "global_end_ms": max(event.global_end_ms for event in cluster),
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
                    "cross_view_anchor_event_ids": [
                        event.event_id for event in cross_view
                    ],
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
    def progressive_anchor_windows(
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
                    local_start = max(0.0, transforms[view_id].to_local(global_start))
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
    def progressive_scout_anchor_representatives(
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
    def merge_time_windows(
        windows: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        return merge_time_windows(windows)

    @staticmethod
    def window_fully_covered(
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
    def uncovered_time_windows(
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

        merged_requested = cls.merge_time_windows(requested)
        merged_coverage = cls.merge_time_windows(coverage)
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
        return cls.merge_time_windows(uncovered)

    def group_local_recall_plan(
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
        padding_ms = (
            max(0.0, float(perf.get("fine_group_recall_padding_seconds", 0.0))) * 1000.0
        )
        cluster_gap_ms = max(
            0.0,
            float(perf.get("fine_group_recall_cluster_gap_seconds", 10.0)) * 1000.0,
        )
        zero_prior_fallback = bool(
            perf.get("fine_group_recall_zero_prior_quality_fallback", True)
        )
        by_segment = {segment.segment_id: segment for segment in segments}
        third_views = [
            view for view in fine_views if view.role == ViewRole.THIRD_PERSON
        ]
        tp_global_coverage = {
            view.view_id: self.merge_time_windows(
                [
                    (
                        transforms[view.view_id].to_global(start),
                        transforms[view.view_id].to_global(end),
                    )
                    for start, end in actual_windows.get(view.view_id, [])
                ]
            )
            for view in third_views
        }
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
                if self.progressive_candidate_recall_bounds(candidate)[1]
                >= group.global_start_ms
                and self.progressive_candidate_recall_bounds(candidate)[0]
                <= group.global_end_ms
            ]
            target_start_ms = min(
                [group.global_start_ms]
                + [
                    self.progressive_candidate_recall_bounds(item)[0]
                    for item in overlapping_targets
                ]
            )
            target_end_ms = max(
                [group.global_end_ms]
                + [
                    self.progressive_candidate_recall_bounds(item)[1]
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

            def candidate_scan_covered(
                candidate: ActionCandidate, supporting_events: list[EvidenceEvent]
            ) -> bool:
                # A same-clock camera or an early event is not proof that the
                # event's TP view was scanned throughout this FP fragment.
                supporting_tp_views = {
                    view_id
                    for event in supporting_events
                    for view_id in event.supporting_views
                    if view_id in tp_global_coverage
                }
                return any(
                    self.window_fully_covered(
                        max(target_start_ms, candidate.global_start_ms),
                        min(target_end_ms, candidate.global_end_ms),
                        tp_global_coverage[view_id],
                    )
                    for view_id in supporting_tp_views
                )

            def candidate_supported(candidate: ActionCandidate) -> bool:
                for event in cross_view_events:
                    if event.action_type != candidate.action_type:
                        continue
                    if (
                        candidate.objects
                        and event.objects
                        and not (set(candidate.objects) & set(event.objects))
                    ):
                        continue
                    overlap = min(event.global_end_ms, candidate.global_end_ms) - max(
                        event.global_start_ms, candidate.global_start_ms
                    )
                    if (
                        overlap >= 0.0
                        or abs(event.key_global_ms - candidate.key_global_ms) <= 1500.0
                    ) and candidate_scan_covered(candidate, [event]):
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
                    <= max(item.global_end_ms for item in raw_unresolved_clusters[-1])
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
                covered_ids = {
                    item.candidate_id
                    for item in cluster
                    # Clustering schedules recall; it must not transfer an
                    # early workstation's proof to later, unrelated activity.
                    if candidate_scan_covered(
                        item,
                        [
                            event
                            for event in supporting_cross_view_events
                            if event.global_end_ms >= item.global_start_ms
                            and event.global_start_ms <= item.global_end_ms
                        ],
                    )
                }
                cross_view_supported_candidate_ids.update(covered_ids)
                supported = len(covered_ids) == len(cluster)
                remaining = [
                    item for item in cluster if item.candidate_id not in covered_ids
                ]
                reported_candidates = cluster if supported else remaining
                raw_cluster_reports.append(
                    {
                        "cluster_id": f"{group.group_id}-UNRESOLVED-{index:03d}",
                        "global_start_ms": cluster_start,
                        "global_end_ms": cluster_end,
                        "candidate_ids": [
                            item.candidate_id for item in reported_candidates
                        ],
                        "candidate_count": len(reported_candidates),
                        "source_cluster_candidate_count": len(cluster),
                        "scan_covered_candidate_ids": sorted(covered_ids),
                        "cross_view_event_ids": [
                            event.event_id for event in supporting_cross_view_events
                        ],
                        "cross_view_supported": supported,
                        "status": (
                            "satisfied_by_cross_view_event_and_tp_scan_coverage"
                            if supported
                            else "unresolved_missing_supporting_tp_scan_coverage"
                            if supporting_cross_view_events
                            else "unresolved"
                        ),
                    }
                )
            unresolved = [
                candidate
                for candidate in raw_unresolved
                if candidate.candidate_id not in cross_view_supported_candidate_ids
            ]
            cluster_reports = [
                item for item in raw_cluster_reports if not item["cross_view_supported"]
            ]
            supported_cluster_reports = [
                item for item in raw_cluster_reports if item["cross_view_supported"]
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
                "coverage_policy": "each_candidate_requires_local_event_and_supporting_tp_scan_coverage",
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
                global_coverage = self.merge_time_windows(
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
                    if not self.window_fully_covered(
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
                requested_local_windows = self.merge_time_windows(
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
                local_windows = self.uncovered_time_windows(
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
                covered_group_seconds = (
                    sum(
                        max(
                            0.0,
                            min(end, transforms[view_id].to_local(group.global_end_ms))
                            - max(
                                start,
                                transforms[view_id].to_local(group.global_start_ms),
                            ),
                        )
                        for start, end in actual_windows.get(view_id, [])
                    )
                    / 1000.0
                )
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
                        "covered_group_seconds": round(covered_group_seconds, 6),
                        "evidence_yield_per_selected_second": round(
                            local_support / max(covered_group_seconds, 1e-9), 9
                        ),
                        "coarse_active_anchor_frames": int(
                            fine_view_report.get(view_id, {}).get(
                                "active_anchor_frames", 0
                            )
                        ),
                        "coarse_anchor_frames": int(
                            fine_view_report.get(view_id, {}).get("anchor_frames", 0)
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
                if str(cluster["cluster_id"]) not in set(chosen["missing_cluster_ids"]):
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

    @staticmethod
    def window_coverage(
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
                    round(seconds / duration_seconds, 6)
                    if duration_seconds > 0
                    else 0.0
                ),
            }
        return report
