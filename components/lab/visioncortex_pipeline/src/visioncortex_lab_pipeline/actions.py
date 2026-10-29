"""Compatibility imports for analysis modules.

New code imports the owning analysis module. These aliases remain until all
historical clients and delivery packages use the documented package boundary.
No analysis implementation lives here.
"""
from __future__ import annotations

from .analysis.observations import (
    CAP_CLASSES as CAP_CLASSES,
    CONTAINER_CLASSES as CONTAINER_CLASSES,
    DEVICE_CLASSES as DEVICE_CLASSES,
    HAND_CLASSES as HAND_CLASSES,
    NON_ACTION_CLASSES as NON_ACTION_CLASSES,
    SUPPORT_ANCHOR_CLASSES as SUPPORT_ANCHOR_CLASSES,
    TRANSFER_TOOL_CLASSES as TRANSFER_TOOL_CLASSES,
    _ACTIVITY_PHYSICAL_TYPES as _ACTIVITY_PHYSICAL_TYPES,
    _Observation as _Observation,
    _box_area as _box_area,
    _box_center as _box_center,
    _box_distance as _box_distance,
    _container_family as _container_family,
    _container_state as _container_state,
    _container_state_key as _container_state_key,
    _frame_observations as _frame_observations,
    _instance_evidence as _instance_evidence,
    _instance_key as _instance_key,
    _observation_evidence as _observation_evidence,
    _observation_instance_signature as _observation_instance_signature,
    _observation_key as _observation_key,
    _observation_state_uncertainty as _observation_state_uncertainty,
    _representative_observation_evidence as _representative_observation_evidence,
)
from .analysis.candidates import (
    _LiquidContactRun as _LiquidContactRun,
    _RunAccumulator as _RunAccumulator,
    _StreamingLiquidSequences as _StreamingLiquidSequences,
    _candidate_from_accumulator as _candidate_from_accumulator,
    _generate_candidates_streaming as _generate_candidates_streaming,
    _infer_liquid_transfer_sequences as _infer_liquid_transfer_sequences,
    _legacy_candidate_covered as _legacy_candidate_covered,
    _merge_observations as _merge_observations,
    generate_candidates as generate_candidates,
    generate_coarse_activity_candidates as generate_coarse_activity_candidates,
)
from .analysis.motion import (
    _adaptive_motion_thresholds as _adaptive_motion_thresholds,
    _candidate_object_identity as _candidate_object_identity,
    _coarse_candidates_compatible as _coarse_candidates_compatible,
    _motion_probe_frames as _motion_probe_frames,
    _semantic_coarse_clusters as _semantic_coarse_clusters,
    fuse_motion_probe_candidates as fuse_motion_probe_candidates,
    generate_motion_burst_candidates as generate_motion_burst_candidates,
    generate_motion_safety_candidates as generate_motion_safety_candidates,
    refine_motion_candidates_with_coarse as refine_motion_candidates_with_coarse,
    select_fine_scan_views as select_fine_scan_views,
)
from .analysis.audit import (
    _appearance_similarity as _appearance_similarity,
    _candidate_alignment_profile as _candidate_alignment_profile,
    _cluster_confidence_receipt as _cluster_confidence_receipt,
    _instance_association_receipt as _instance_association_receipt,
    _objects_overlap as _objects_overlap,
    _weighted_quantile as _weighted_quantile,
    audit_candidates as audit_candidates,
    refine_liquid_events_with_context as refine_liquid_events_with_context,
)
from .analysis.segmentation import (
    _activity_objects as _activity_objects,
    _event_core_interval as _event_core_interval,
    _split_rich_repeated_primary_sequences as _split_rich_repeated_primary_sequences,
    activity_seed_intervals as activity_seed_intervals,
    build_activity_intervals as build_activity_intervals,
    build_experiment_segments as build_experiment_segments,
    build_physical_change_log as build_physical_change_log,
    build_view_activity_intervals as build_view_activity_intervals,
    fine_scan_windows as fine_scan_windows,
    merge_activity_intervals as merge_activity_intervals,
)
