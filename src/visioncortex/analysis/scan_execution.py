"""Progressive scan execution using explicit planner and runtime ports.

The executor owns pass-local detection paths, coverage and index state. Pipeline
status/queue/archive lifecycle remains outside this service.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..candidate_index import FineFrameIndex
from ..schemas import (
    ActionCandidate,
    EvidenceEvent,
    ExperimentGroup,
    ExperimentSegment,
    RunManifest,
    ViewInput,
    ViewRole,
)
from .experiments import ExperimentAnalysisOperations, compose_experiments
from .scan_planning import FineScanPlanner


@dataclass(frozen=True)
class ScanExecutionPorts:
    scan_views: Callable[..., dict[str, Path]]
    frame_index_path: Callable[[Path, str], Path]
    fine_windows: Callable[..., dict[str, list[tuple[float, float]]]]


@dataclass(frozen=True)
class ScanAlgorithms:
    merge_frame_evidence_ledgers: Callable[..., Any]
    attach_action_observability: Callable[..., Any]
    audit_candidates: Callable[..., Any]
    build_experiment_groups: Callable[..., Any]
    build_experiment_segments: Callable[..., Any]
    create_fine_frame_index: Callable[..., Any]
    fine_frame_coverage_report: Callable[..., Any]
    generate_candidates: Callable[..., Any]
    ingest_fine_frame_ledgers: Callable[..., Any]
    normalize_experiment_segments: Callable[..., Any]
    prepare_formal_experiment_segments: Callable[..., Any]
    refine_liquid_events_with_context: Callable[..., Any]
    select_fine_scan_views: Callable[..., Any]
    select_key_events: Callable[..., Any]


class ProgressiveScanExecutor:
    def __init__(
        self,
        config: dict[str, Any],
        *,
        planner: FineScanPlanner,
        ports: ScanExecutionPorts,
        operations: ScanAlgorithms,
    ):
        self.config = config
        self.planner = planner
        self.ports = ports
        self.operations = operations

    def run(
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
            fine_frame_index = self.operations.create_fine_frame_index(
                self.ports.frame_index_path(work_dir, "fine_frame_index.sqlite3")
            )
        ordered_initial, ordered_supplemental = (
            self.planner.progressive_fine_view_order(
                fine_views,
                fine_view_report,
                int(perf.get("fine_initial_third_person_views", 1)),
                list(perf.get("fine_preferred_third_person_views") or []),
            )
        )
        if not any(
            view.role == ViewRole.THIRD_PERSON
            for view in ordered_initial + ordered_supplemental
        ):
            raise ValueError(
                "progressive fine scan requires at least one third-person view"
            )
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
                return self.operations.generate_candidates(
                    current_views, detection_paths, self.config
                )
            return self.operations.generate_candidates(
                current_views,
                detection_paths,
                self.config,
                fine_frame_index,
            )

        def refine_current_liquid_context(
            current_events: list[EvidenceEvent],
        ) -> list[dict[str, Any]]:
            if fine_frame_index is None:
                return self.operations.refine_liquid_events_with_context(
                    current_events,
                    detection_paths,
                )
            return self.operations.refine_liquid_events_with_context(
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
                pass_windows = self.planner.progressive_anchor_windows(
                    target_snapshot,
                    [view.view_id for view in pass_views],
                    infos,
                    transforms,
                    float(perf.get("fine_progressive_anchor_padding_seconds", 10.0)),
                    {candidate.candidate_id for candidate in pass_candidates},
                )
                if not any(pass_windows.values()):
                    all_pass_windows = self.ports.fine_windows(
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
                all_pass_windows = self.ports.fine_windows(
                    pass_candidates, infos, transforms
                )
                pass_windows = {
                    view.view_id: all_pass_windows[view.view_id] for view in pass_views
                }
                window_strategy = "coarse_candidate_windows"
            pass_manifest = manifest.model_copy(update={"views": pass_views})
            result = self.ports.scan_views(
                pass_manifest,
                infos,
                transforms,
                work_dir / pass_name,
                windows=pass_windows,
                sample_fps=float(perf["detection_fps"]),
                phase="fine",
                decode_backends={
                    view.view_id: full_decode_backends[view.view_id]
                    for view in pass_views
                },
            )
            merge_reports: list[dict[str, Any]] = []
            index_ingest_report: dict[str, Any] | None = None
            if fine_frame_index is not None:
                index_ingest_report = self.operations.ingest_fine_frame_ledgers(
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
                        perf.get("fine_track_stitch_max_center_distance", 0.12)
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
                            **self.operations.merge_frame_evidence_ledgers(
                                existing_path,
                                result[view.view_id],
                                merged_path,
                                track_id_namespace=pass_index + 1,
                            ),
                        }
                    )
                    detection_paths[view.view_id] = merged_path
                actual_windows[view.view_id] = self.planner.merge_time_windows(
                    [
                        *actual_windows.get(view.view_id, []),
                        *pass_windows[view.view_id],
                    ]
                )
            scanned_views = [view for view in fine_views if view.view_id in scanned]
            current_candidates = generate_current_candidates(scanned_views)
            if fine_frame_index is None:
                current_events, _ = self.operations.audit_candidates(
                    current_candidates,
                    transforms,
                    self.config,
                )
            else:
                current_events, _ = self.operations.audit_candidates(
                    current_candidates,
                    transforms,
                    self.config,
                    fine_frame_index,
                )
            liquid_context_rejections = refine_current_liquid_context(current_events)
            target_status = self.planner.progressive_target_status(
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
                    "coverage": self.planner.window_coverage(pass_windows, infos),
                    "candidate_count_after_pass": len(current_candidates),
                    "liquid_context_rejection_count": len(liquid_context_rejections),
                    "liquid_context_rejected_event_ids": [
                        item["event_id"] for item in liquid_context_rejections
                    ],
                    "target_status_after_pass": target_status,
                    "detection_ledger_merges": merge_reports,
                    "fine_index_ingest": index_ingest_report,
                }
            )
            return target_status

        target_status = execute_pass(0, "primary", initial_views, boundary_candidates)
        unresolved_ids = {
            item["candidate_id"]
            for item in target_status
            if item["status"] == "needs_third_person_supplement"
        }
        if scout_enabled and unresolved_ids:
            scout_view_ids = [view.view_id for view in supplemental_views]
            scout_anchors, scout_anchor_diagnostics = (
                self.planner.progressive_scout_anchor_representatives(
                    target_status,
                    unresolved_ids,
                    dedup_tolerance_seconds=float(
                        perf.get("fine_scout_peak_dedup_tolerance_seconds", 0.25)
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
            scout_windows = self.planner.progressive_anchor_windows(
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
            scout_paths = self.ports.scan_views(
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
            _, scout_view_report = self.operations.select_fine_scan_views(
                supplemental_views,
                scout_paths,
                boundary_candidates,
                scout_config,
            )
            ranked_initial, ranked_tail = self.planner.progressive_fine_view_order(
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
                view.view_id: index for index, view in enumerate(supplemental_views, 1)
            }
            for view in supplemental_views:
                scout_item = scout_view_report.get(view.view_id, {})
                fine_view_report.setdefault(view.view_id, {}).update(
                    {
                        "progressive_initial": False,
                        "progressive_supplemental_rank": scout_rank[view.view_id],
                        "scout_rank": scout_rank[view.view_id],
                        "scout_anchor_frames": int(scout_item.get("anchor_frames", 0)),
                        "scout_active_anchor_frames": int(
                            scout_item.get("active_anchor_frames", 0)
                        ),
                        "scout_anchor_classes": list(
                            scout_item.get("anchor_classes") or []
                        ),
                        "scout_motion_threshold": scout_item.get("motion_threshold"),
                    }
                )
            scout_coverage = self.planner.window_coverage(scout_windows, infos)
            scout_seconds = sum(
                float(item["selected_seconds"]) for item in scout_coverage.values()
            )
            scout_fps = float(perf.get("fine_scout_fps", 1.0))
            scout_summary = {
                "enabled": True,
                "view_ids": scout_view_ids,
                "sample_fps": scout_fps,
                "image_size": int(perf.get("fine_scout_image_size", 416)),
                "keyframes_only": bool(perf.get("fine_scout_keyframes_only", False)),
                "anchor_radius_seconds": float(
                    perf.get("fine_scout_anchor_radius_seconds", 3.0)
                ),
                "anchor_selection": scout_anchor_diagnostics,
                "window_strategy": "aligned_first_person_peak_windows",
                "pre_merge_window_count_per_view": len(scout_anchors),
                "post_merge_window_count_by_view": {
                    view_id: len(windows) for view_id, windows in scout_windows.items()
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

        formal_state_cache: (
            tuple[
                list[ActionCandidate],
                list[EvidenceEvent],
                list[ExperimentSegment],
                list[ExperimentGroup],
                list[EvidenceEvent],
            ]
            | None
        ) = None

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
                state_events, _ = self.operations.audit_candidates(
                    state_candidates,
                    transforms,
                    self.config,
                )
            else:
                state_events, _ = self.operations.audit_candidates(
                    state_candidates,
                    transforms,
                    self.config,
                    fine_frame_index,
                )
            refine_current_liquid_context(state_events)
            self.operations.attach_action_observability(state_events)
            state_analysis = compose_experiments(
                state_events,
                manifest.views,
                boundary_candidates,
                self.config,
                operations=ExperimentAnalysisOperations(
                    self.operations.build_experiment_segments,
                    self.operations.normalize_experiment_segments,
                    self.operations.prepare_formal_experiment_segments,
                    self.operations.build_experiment_groups,
                    self.operations.select_key_events,
                ),
                collect_decisions=False,
            )
            state_segments = state_analysis.segments
            state_groups = state_analysis.groups
            state_key_events = state_analysis.selected_key_events
            formal_state_cache = (
                state_candidates,
                state_events,
                state_segments,
                state_groups,
                state_key_events,
            )
            return formal_state_cache

        local_recall_rounds: list[dict[str, Any]] = []
        local_recall_enabled = bool(perf.get("fine_group_local_recall_enabled", False))
        maximum_recall_rounds = max(0, int(perf.get("fine_group_recall_max_rounds", 5)))
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
                local_recall_plan = self.planner.group_local_recall_plan(
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
                selected_plans = list(local_recall_plan.get("selected_plans") or [])
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
                        (float(start), float(end)) for start, end in item["windows"]
                    )
                    group_ids_by_view.setdefault(view_id, []).append(
                        str(item["group_id"])
                    )
                for view_id, view_windows in list(windows_by_view.items()):
                    windows_by_view[view_id] = self.planner.merge_time_windows(
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
                        "selected_key_event_gain": len(after_key_events) - before_count,
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
            local_recall_plan = self.planner.group_local_recall_plan(
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
                        item.get("status") == "unresolved_all_tp_windows_exhausted"
                        for item in local_recall_plan.get("groups", [])
                    )
                    else "no_positive_recall_prior_without_quality_fallback"
                    if local_recall_plan.get("complete")
                    and any(
                        item.get("status") == "complete_no_positive_recall_prior"
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
            target_status = self.planner.progressive_target_status(
                boundary_candidates, final_target_events
            )
            unresolved_ids, quarantined_ids = (
                self.planner.quarantine_nonformal_progressive_gaps(
                    target_status,
                    final_groups,
                )
            )
        all_coverage = self.planner.window_coverage(fine_windows, infos)
        actual_coverage = self.planner.window_coverage(actual_windows, infos)
        fine_index_report: dict[str, Any] | None = None
        if fine_frame_index is not None:
            fine_index_report = self.operations.fine_frame_coverage_report(
                fine_frame_index,
                scanned_views,
                infos,
                actual_windows,
                sample_fps=float(perf["detection_fps"]),
                minimum_coverage_ratio=float(
                    perf.get("fine_minimum_coverage_ratio", 0.98)
                ),
                maximum_gap_periods=float(perf.get("fine_maximum_gap_periods", 4.0)),
                alignment_scales={
                    view.view_id: transforms[view.view_id].scale
                    for view in scanned_views
                },
            )
            detection_paths = fine_frame_index.materialize_ledgers(
                work_dir / "indexed-detections",
                scanned_views,
            )
        all_seconds = sum(
            float(item["selected_seconds"]) for item in all_coverage.values()
        )
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
            "avoided_selected_seconds": round(
                max(0.0, all_seconds - actual_seconds), 3
            ),
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
