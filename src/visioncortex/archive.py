"""Historical archive API compatibility and dependency composition.

Implementation lives in evidence/ and media/. New callers needing only layout
or serialization should import those lightweight owners directly. Operational
adapters supply named service ports so historical monkeypatches remain valid.
Compatibility exports are reviewed for removal after all supported delivery
clients have migrated; see the architecture migration guide.
"""

from __future__ import annotations

import csv as csv
import gc as gc
import hashlib as hashlib
import json as json
import math as math
import os as os
import re as re
import shutil as shutil
import subprocess as subprocess
import threading as threading
import time as time
import uuid as uuid
from concurrent.futures import (
    FIRST_COMPLETED as FIRST_COMPLETED,
)
from concurrent.futures import (
    ThreadPoolExecutor as ThreadPoolExecutor,
)
from concurrent.futures import (
    as_completed as as_completed,
)
from concurrent.futures import (
    wait as wait,
)
from itertools import product as product
from pathlib import Path as Path
from typing import Any as Any
from typing import Callable as Callable
from typing import Sequence as Sequence

import cv2 as cv2
import numpy as np
from openpyxl import Workbook as Workbook

from .action_semantics import (
    DEVICE_CLASSES as DEVICE_CLASSES,
)
from .action_semantics import (
    action_participant_visibility as action_participant_visibility,
)
from .action_semantics import (
    normalize_participant_class,
)
from .action_semantics import (
    record_semantic_review as record_semantic_review,
)
from .action_semantics import (
    semantic_action_proof_contradictions as semantic_action_proof_contradictions,
)
from .action_state_machine import build_event_state_receipt as build_event_state_receipt
from .alignment import iter_aligned_rows as iter_aligned_rows
from .detection import (
    iter_frame_evidence as iter_frame_evidence,
)
from .detection import (
    nearest_frame_evidence_many as nearest_frame_evidence_many,
)
from .evidence import archive_cache as _evidence_archive_cache
from .evidence import artifacts as _evidence_artifacts
from .evidence import layout as _evidence_layout
from .evidence import publication as _evidence_publication
from .evidence import semantic_curation as _evidence_semantic_curation
from .evidence import semantic_errors as _evidence_semantic_errors
from .evidence import semantic_review as _evidence_semantic_review
from .evidence import serialization as _evidence_serialization
from .grouping import (
    event_stable_identities as event_stable_identities,
)
from .grouping import (
    event_view_actor_identities as event_view_actor_identities,
)
from .grouping import (
    event_view_stable_identities as event_view_stable_identities,
)
from .identity import PRODUCT_NAME as PRODUCT_NAME
from .indexing import (
    build_archive_index as build_archive_index,
)
from .indexing import (
    stable_artifact_uid as stable_artifact_uid,
)
from .indexing import (
    stable_event_uid as stable_event_uid,
)
from .indexing import (
    stable_evidence_uid as stable_evidence_uid,
)
from .key_material_verification import (
    SelectiveVerificationBudget as SelectiveVerificationBudget,
)
from .key_material_verification import (
    plan_selective_key_material_verification as plan_selective_key_material_verification,
)
from .liquid_semantic import (
    analyze_liquid_semantics as analyze_liquid_semantics,
)
from .liquid_semantic import (
    release_liquid_semantic_model_cache as release_liquid_semantic_model_cache,
)
from .material_naming import (
    ACTION_CATEGORY_FOLDERS as ACTION_CATEGORY_FOLDERS,
)
from .material_naming import (
    key_material_action_folder as key_material_action_folder,
)
from .media import experiment_materials as _media_experiment_materials
from .media import key_materials as _media_key_materials
from .media import participant_annotations as _media_participant_annotations
from .media import participant_grounding as _media_participant_grounding
from .mllm import (
    EVENT_SYSTEM_PROMPT as EVENT_SYSTEM_PROMPT,
)
from .mllm import (
    FINAL_GROUP_SYSTEM_PROMPT as FINAL_GROUP_SYSTEM_PROMPT,
)
from .mllm import (
    GROUP_SYSTEM_PROMPT as GROUP_SYSTEM_PROMPT,
)
from .mllm import (
    ArkStepAnalyzer as ArkStepAnalyzer,
)
from .mllm import (
    normalize_uncalibrated_hand_identity as normalize_uncalibrated_hand_identity,
)
from .mllm_provider import vision_request_identity as vision_request_identity
from .open_vocabulary_runtime import (
    park_open_vocabulary_model as park_open_vocabulary_model,
)
from .open_vocabulary_runtime import (
    run_with_cuda_oom_cpu_fallback as run_with_cuda_oom_cpu_fallback,
)
from .open_vocabulary_runtime import (
    serialized_open_vocabulary as serialized_open_vocabulary,
)
from .open_vocabulary_runtime import (
    yolo_world_prediction_device as yolo_world_prediction_device,
)
from .participant_visual_review import (
    ParticipantVisualReviewer as ParticipantVisualReviewer,
)
from .pathing import archive_relative_posix as archive_relative_posix
from .schemas import (
    ActionType as ActionType,
)
from .schemas import (
    AlignmentTransform as AlignmentTransform,
)
from .schemas import (
    EvidenceEvent as EvidenceEvent,
)
from .schemas import (
    ExperimentGroup as ExperimentGroup,
)
from .schemas import (
    ExperimentSegment as ExperimentSegment,
)
from .schemas import (
    FrameEvidence as FrameEvidence,
)
from .schemas import (
    PhysicalChange as PhysicalChange,
)
from .schemas import (
    RunManifest as RunManifest,
)
from .schemas import (
    RunSummary as RunSummary,
)
from .schemas import (
    VideoInfo as VideoInfo,
)
from .schemas import (
    ViewInput as ViewInput,
)
from .schemas import (
    ViewRole as ViewRole,
)
from .schemas import (
    event_is_formal as event_is_formal,
)
from .schemas import (
    set_event_admission as set_event_admission,
)
from .source_frames import read_evidence_frame as read_evidence_frame
from .speech_semantics import (
    SpeechContext as SpeechContext,
)
from .speech_semantics import (
    prompt_with_speech as prompt_with_speech,
)
from .temporal_segmentation import (
    audit_participant_continuity as audit_participant_continuity,
)
from .temporal_segmentation import (
    release_temporal_segmentation_model_cache as release_temporal_segmentation_model_cache,
)
from .video_io import (
    ViewFrameReader as ViewFrameReader,
)
from .video_io import (
    create_grid_video as create_grid_video,
)
from .video_io import (
    extract_view_clip as extract_view_clip,
)
from .video_io import (
    probe_video as probe_video,
)
from .video_io import (
    select_video_encoder as select_video_encoder,
)
from .video_io import (
    view_source_files as view_source_files,
)
from .video_io import (
    write_annotated_frame as write_annotated_frame,
)

_json_default = _evidence_serialization._json_default
write_json = _evidence_serialization.write_json
SemanticAnalysisUnavailable = _evidence_semantic_errors.SemanticAnalysisUnavailable
_semantic_cache_reads_enabled = _evidence_archive_cache._semantic_cache_reads_enabled
_execution_cache_reads_enabled = _evidence_archive_cache._execution_cache_reads_enabled
_read_semantic_cache = _evidence_archive_cache._read_semantic_cache
_sha256_file = _evidence_archive_cache._sha256_file
_derived_media_cache_enabled = _evidence_archive_cache._derived_media_cache_enabled
_link_or_copy_immutable = _evidence_archive_cache._link_or_copy_immutable
_generate_detached_media = _evidence_archive_cache._generate_detached_media
_time_slug = _evidence_layout._time_slug
_safe_slug = _evidence_layout._safe_slug
_safe_folder_name = _evidence_layout._safe_folder_name
_component_budget = _evidence_layout._component_budget
ArchiveLayout = _evidence_layout.ArchiveLayout
_storyboard_times = _media_experiment_materials._storyboard_times
_write_aligned_frame = _media_experiment_materials._write_aligned_frame
_key_material_view_pair = _evidence_artifacts._key_material_view_pair
_shared_candidate_time = _media_participant_grounding._shared_candidate_time
_semantic_interaction_is_direct = (
    _evidence_semantic_curation._semantic_interaction_is_direct
)
_box_edge_gap_norm = _media_participant_grounding._box_edge_gap_norm
_box_iou = _media_participant_grounding._box_iou
_box_aspect_ratio = _media_participant_grounding._box_aspect_ratio
_filter_grounded_actor_boxes = _media_participant_grounding._filter_grounded_actor_boxes
_preferred_grounding_terms_by_class = (
    _media_participant_grounding._preferred_grounding_terms_by_class
)
_canonical_grounding_label = _media_participant_grounding._canonical_grounding_label
_select_key_material_media_source = (
    _media_key_materials._select_key_material_media_source
)
_prune_replaced_key_material_views = (
    _media_key_materials._prune_replaced_key_material_views
)
_attempt_key_material = _media_key_materials._attempt_key_material
_with_selected_keyframe_review_images = (
    _evidence_semantic_review._with_selected_keyframe_review_images
)
_run_bounded_semantic_waves = _evidence_semantic_review._run_bounded_semantic_waves
_timestamp_rows = _evidence_publication._timestamp_rows
_DERIVED_MEDIA_POPULATED_THIS_PROCESS = (
    _evidence_archive_cache._DERIVED_MEDIA_POPULATED_THIS_PROCESS
)
RELABEL_OBJECT_PATTERNS = _evidence_semantic_curation.RELABEL_OBJECT_PATTERNS
_OPEN_VOCABULARY_MODEL_CACHE = _media_participant_grounding._OPEN_VOCABULARY_MODEL_CACHE
_OPEN_VOCABULARY_ASSET_VALIDATION = (
    _media_participant_grounding._OPEN_VOCABULARY_ASSET_VALIDATION
)
_GROUNDING_DINO_MODEL_CACHE = _media_participant_grounding._GROUNDING_DINO_MODEL_CACHE
_GROUNDING_DINO_ASSET_VALIDATION = (
    _media_participant_grounding._GROUNDING_DINO_ASSET_VALIDATION
)


def _services_evidence_semantic_review() -> (
    _evidence_semantic_review.SemanticReviewServices
):
    return _evidence_semantic_review.SemanticReviewServices(
        ArkStepAnalyzer=ArkStepAnalyzer,
        SpeechContext=SpeechContext,
        ViewFrameReader=ViewFrameReader,
        _group_folder_name=_group_folder_name,
        _raise_for_incomplete_semantic_results=_raise_for_incomplete_semantic_results,
        _read_semantic_cache=_read_semantic_cache,
        _relative=_relative,
        _run_bounded_semantic_waves=_run_bounded_semantic_waves,
        _safe_slug=_safe_slug,
        _semantic_cache_path=_semantic_cache_path,
        _semantic_cache_reads_enabled=_semantic_cache_reads_enabled,
        _semantic_fingerprint=_semantic_fingerprint,
        _sha256_file=_sha256_file,
        _storyboard_times=_storyboard_times,
        _with_selected_keyframe_review_images=_with_selected_keyframe_review_images,
        _write_aligned_frame=_write_aligned_frame,
        _write_semantic_cache=_write_semantic_cache,
        extract_temporal_review_frames=extract_temporal_review_frames,
        key_material_review_images=key_material_review_images,
        normalize_uncalibrated_hand_identity=normalize_uncalibrated_hand_identity,
        prompt_with_speech=prompt_with_speech,
        record_semantic_review=record_semantic_review,
        write_json=write_json,
    )


def _services_evidence_archive_cache() -> _evidence_archive_cache.ArchiveCacheServices:
    return _evidence_archive_cache.ArchiveCacheServices(
        _DERIVED_MEDIA_POPULATED_THIS_PROCESS=_DERIVED_MEDIA_POPULATED_THIS_PROCESS,
        _derived_media_cache_enabled=_derived_media_cache_enabled,
        _derived_media_cache_reads_enabled=_derived_media_cache_reads_enabled,
        _derived_media_fingerprint=_derived_media_fingerprint,
        _execution_cache_reads_enabled=_execution_cache_reads_enabled,
        _generate_detached_media=_generate_detached_media,
        _json_default=_json_default,
        _link_or_copy_immutable=_link_or_copy_immutable,
        _safe_slug=_safe_slug,
        _sha256_file=_sha256_file,
        select_video_encoder=select_video_encoder,
        vision_request_identity=vision_request_identity,
        write_json=write_json,
    )


def _services_media_experiment_materials() -> (
    _media_experiment_materials.ExperimentMaterialsServices
):
    return _media_experiment_materials.ExperimentMaterialsServices(
        _group_folder_name=_group_folder_name,
        _materialize_derived_media=_materialize_derived_media,
        _relative=_relative,
        _safe_slug=_safe_slug,
        create_grid_video=create_grid_video,
        extract_view_clip=extract_view_clip,
        iter_aligned_rows=iter_aligned_rows,
        view_source_files=view_source_files,
        write_json=write_json,
    )


def _services_evidence_artifacts() -> _evidence_artifacts.ArtifactsServices:
    return _evidence_artifacts.ArtifactsServices(
        _artifact_json=_artifact_json,
        _key_material_event_folder_name=_key_material_event_folder_name,
        _key_material_view_pair=_key_material_view_pair,
        _relative=_relative,
        _safe_folder_name=_safe_folder_name,
        _safe_slug=_safe_slug,
        _select_key_material_view_pair=_select_key_material_view_pair,
        event_view_actor_identities=event_view_actor_identities,
        event_view_stable_identities=event_view_stable_identities,
        stable_artifact_uid=stable_artifact_uid,
        stable_event_uid=stable_event_uid,
        stable_evidence_uid=stable_evidence_uid,
        write_json=write_json,
    )


def _services_media_participant_grounding() -> (
    _media_participant_grounding.ParticipantGroundingServices
):
    return _media_participant_grounding.ParticipantGroundingServices(
        ViewFrameReader=ViewFrameReader,
        _GROUNDING_DINO_ASSET_VALIDATION=_GROUNDING_DINO_ASSET_VALIDATION,
        _GROUNDING_DINO_MODEL_CACHE=_GROUNDING_DINO_MODEL_CACHE,
        _OPEN_VOCABULARY_ASSET_VALIDATION=_OPEN_VOCABULARY_ASSET_VALIDATION,
        _OPEN_VOCABULARY_MODEL_CACHE=_OPEN_VOCABULARY_MODEL_CACHE,
        _box_aspect_ratio=_box_aspect_ratio,
        _box_edge_gap_norm=_box_edge_gap_norm,
        _box_iou=_box_iou,
        _canonical_grounding_label=_canonical_grounding_label,
        _event_key_frame_score=_event_key_frame_score,
        _event_participant_boxes=_event_participant_boxes,
        _filter_grounded_actor_boxes=_filter_grounded_actor_boxes,
        _grounding_dino_key_frame_detections=_grounding_dino_key_frame_detections,
        _grounding_dino_key_frame_detections_once=_grounding_dino_key_frame_detections_once,
        _missing_state_transition_fallback_classes=_missing_state_transition_fallback_classes,
        _preferred_grounding_terms_by_class=_preferred_grounding_terms_by_class,
        _release_auxiliary_model_caches=_release_auxiliary_model_caches,
        _select_manipulated_object_candidate=_select_manipulated_object_candidate,
        _select_state_container_candidate=_select_state_container_candidate,
        _sha256_file=_sha256_file,
        _shared_candidate_time=_shared_candidate_time,
        iter_frame_evidence=iter_frame_evidence,
        normalize_participant_class=normalize_participant_class,
        park_open_vocabulary_model=park_open_vocabulary_model,
        release_liquid_semantic_model_cache=release_liquid_semantic_model_cache,
        release_temporal_segmentation_model_cache=release_temporal_segmentation_model_cache,
        run_with_cuda_oom_cpu_fallback=run_with_cuda_oom_cpu_fallback,
        yolo_world_prediction_device=yolo_world_prediction_device,
    )


def _services_evidence_semantic_curation() -> (
    _evidence_semantic_curation.SemanticCurationServices
):
    return _evidence_semantic_curation.SemanticCurationServices(
        _annotation_supports_curated_action=_annotation_supports_curated_action,
        _interaction_participant_classes=_interaction_participant_classes,
        _key_material_event_folder_name=_key_material_event_folder_name,
        _move_curated_event_media=_move_curated_event_media,
        _participant_class_mentions=_participant_class_mentions,
        _relabel_participant_objects=_relabel_participant_objects,
        _relative=_relative,
        _rerender_curated_participant_annotations=_rerender_curated_participant_annotations,
        _safe_folder_name=_safe_folder_name,
        _semantic_interaction_is_direct=_semantic_interaction_is_direct,
        _semantic_participant_conflicts=_semantic_participant_conflicts,
        action_participant_visibility=action_participant_visibility,
        build_event_state_receipt=build_event_state_receipt,
        event_stable_identities=event_stable_identities,
        normalize_participant_class=normalize_participant_class,
        semantic_action_proof_contradictions=semantic_action_proof_contradictions,
        write_json=write_json,
        write_key_material_category_index=write_key_material_category_index,
    )


def _services_media_key_materials() -> _media_key_materials.KeyMaterialsServices:
    return _media_key_materials.KeyMaterialsServices(
        ParticipantVisualReviewer=ParticipantVisualReviewer,
        ViewFrameReader=ViewFrameReader,
        _artifact_json=_artifact_json,
        _attempt_key_material=_attempt_key_material,
        _best_event_frames_many=_best_event_frames_many,
        _bounded_grounding_dino_temporal_rescue=_bounded_grounding_dino_temporal_rescue,
        _event_participant_boxes=_event_participant_boxes,
        _generate_detached_media=_generate_detached_media,
        _grounding_dino_key_frame_detections=_grounding_dino_key_frame_detections,
        _key_material_event_folder_name=_key_material_event_folder_name,
        _materialize_derived_media=_materialize_derived_media,
        _prune_replaced_key_material_views=_prune_replaced_key_material_views,
        _relative=_relative,
        _release_auxiliary_model_caches=_release_auxiliary_model_caches,
        _review_bounded_cap_keyframe=_review_bounded_cap_keyframe,
        _safe_folder_name=_safe_folder_name,
        _select_key_material_media_source=_select_key_material_media_source,
        _select_key_material_view_pair_with_peak_fallback=_select_key_material_view_pair_with_peak_fallback,
        _shared_candidate_time=_shared_candidate_time,
        _write_aligned_frame=_write_aligned_frame,
        create_grid_video=create_grid_video,
        extract_view_clip=extract_view_clip,
        nearest_frame_evidence_many=nearest_frame_evidence_many,
        prepare_key_material_category_layout=prepare_key_material_category_layout,
        probe_video=probe_video,
        read_evidence_frame=read_evidence_frame,
        view_source_files=view_source_files,
        write_annotated_frame=write_annotated_frame,
        write_json=write_json,
        write_key_material_category_index=write_key_material_category_index,
    )


def _services_evidence_publication() -> _evidence_publication.PublicationServices:
    return _evidence_publication.PublicationServices(
        _artifact_json=_artifact_json,
        _key_material_view_pair=_key_material_view_pair,
        _time_slug=_time_slug,
        _timestamp_rows=_timestamp_rows,
        build_archive_index=build_archive_index,
        evidence_package_eval=evidence_package_eval,
        write_json=write_json,
        write_screening_notes=write_screening_notes,
        write_timestamp_tables=write_timestamp_tables,
    )


def _raise_for_incomplete_semantic_results(
    layout: "ArchiveLayout",
    *,
    stage: str,
    results: Sequence[tuple[str, dict[str, Any]]],
    fail_run: bool = True,
) -> list[str]:
    return _evidence_semantic_review._raise_for_incomplete_semantic_results(
        layout,
        stage=stage,
        results=results,
        fail_run=fail_run,
        services=_services_evidence_semantic_review(),
    )


def _semantic_fingerprint(
    kind: str,
    config: dict[str, Any],
    prompt: str,
    evidence: dict[str, Any],
    images: Sequence[tuple[str, Path]],
) -> str:
    """Fingerprint exactly the evidence that can change an MLLM answer.

    Run ids, cache identities and archive folder names are deliberately absent.
    A failed run can therefore resume on the same machine without paying for an
    identical request, while any boundary, CV event, prompt, model or image
    change forces a new call.
    """
    return _evidence_archive_cache._semantic_fingerprint(
        kind,
        config,
        prompt,
        evidence,
        images,
        services=_services_evidence_archive_cache(),
    )


def _semantic_cache_path(config: dict[str, Any], kind: str, fingerprint: str) -> Path:
    return _evidence_archive_cache._semantic_cache_path(
        config, kind, fingerprint, services=_services_evidence_archive_cache()
    )


def _write_semantic_cache(
    path: Path, fingerprint: str, result: dict[str, Any]
) -> dict[str, Any]:
    return _evidence_archive_cache._write_semantic_cache(
        path, fingerprint, result, services=_services_evidence_archive_cache()
    )


def _derived_media_cache_reads_enabled(config: dict[str, Any]) -> bool:
    return _evidence_archive_cache._derived_media_cache_reads_enabled(
        config, services=_services_evidence_archive_cache()
    )


def _derived_media_fingerprint(
    kind: str,
    config: dict[str, Any],
    request: dict[str, Any],
    inputs: Sequence[Path],
    *,
    content_address_inputs: bool = False,
) -> str:
    return _evidence_archive_cache._derived_media_fingerprint(
        kind,
        config,
        request,
        inputs,
        content_address_inputs=content_address_inputs,
        services=_services_evidence_archive_cache(),
    )


def _materialize_derived_media(
    destination: Path,
    kind: str,
    request: dict[str, Any],
    inputs: Sequence[Path],
    config: dict[str, Any],
    generator: Callable[[], None],
    *,
    content_address_inputs: bool = False,
) -> dict[str, Any]:
    """Reuse one immutable derived video only when its complete identity matches.

    Original source video is never copied into this cache.  The request binds
    source identity, exact time bounds, selected encoder, and transformation
    parameters; the sidecar additionally binds the cached output bytes.  Any
    missing or inconsistent receipt fails closed instead of serving uncertain
    media.
    """
    return _evidence_archive_cache._materialize_derived_media(
        destination,
        kind,
        request,
        inputs,
        config,
        generator,
        content_address_inputs=content_address_inputs,
        services=_services_evidence_archive_cache(),
    )


_relative = _evidence_layout._relative


_bounded_component = _evidence_layout._bounded_component


_group_folder_name = _evidence_layout._group_folder_name


_group_folder_budget = _evidence_layout._group_folder_budget


_key_material_event_folder_name = _evidence_layout._key_material_event_folder_name


def write_aligned_csv(
    path: Path,
    views: Sequence[ViewInput],
    infos: dict[str, VideoInfo],
    transforms: dict[str, AlignmentTransform],
    output_fps: float,
) -> None:
    return _media_experiment_materials.write_aligned_csv(
        path,
        views,
        infos,
        transforms,
        output_fps,
        services=_services_media_experiment_materials(),
    )


def materialize_experiment_clips(
    layout: ArchiveLayout,
    groups: Sequence[ExperimentGroup],
    segments: list[ExperimentSegment],
    events: Sequence[EvidenceEvent],
    views: Sequence[ViewInput],
    infos: dict[str, VideoInfo],
    transforms: dict[str, AlignmentTransform],
    config: dict[str, Any],
    publisher: Any | None = None,
) -> None:
    return _media_experiment_materials.materialize_experiment_clips(
        layout,
        groups,
        segments,
        events,
        views,
        infos,
        transforms,
        config,
        publisher,
        services=_services_media_experiment_materials(),
    )


def analyze_experiment_groups(
    layout: ArchiveLayout,
    groups: Sequence[ExperimentGroup],
    segments: Sequence[ExperimentSegment],
    events: Sequence[EvidenceEvent],
    views: Sequence[ViewInput],
    infos: dict[str, VideoInfo],
    transforms: dict[str, AlignmentTransform],
    config: dict[str, Any],
    *,
    final_adjudicated: bool = False,
) -> None:
    return _evidence_semantic_review.analyze_experiment_groups(
        layout,
        groups,
        segments,
        events,
        views,
        infos,
        transforms,
        config,
        final_adjudicated=final_adjudicated,
        services=_services_evidence_semantic_review(),
    )


def _select_key_material_view_pair(
    group: ExperimentGroup,
    event: EvidenceEvent,
    views: Sequence[ViewInput],
    infos: dict[str, VideoInfo],
    transforms: dict[str, AlignmentTransform],
) -> tuple[tuple[str, str], dict[str, Any]]:
    """Choose real same-role views that contain the event key timestamp.

    Short synchronized recordings can have unequal physical tails.  A group
    representative may therefore be valid for the experiment clip but no
    longer contain a late event key frame.  Select another existing view of
    the same role only when its aligned local timestamp is physically inside
    the decodable source; never clamp the timestamp or synthesize evidence.
    """
    return _evidence_artifacts._select_key_material_view_pair(
        group, event, views, infos, transforms, services=_services_evidence_artifacts()
    )


def _select_key_material_view_pair_with_peak_fallback(
    group: ExperimentGroup,
    event: EvidenceEvent,
    views: Sequence[ViewInput],
    infos: dict[str, VideoInfo],
    transforms: dict[str, AlignmentTransform],
    accepted_peak_global_ms: float,
) -> tuple[tuple[str, str], dict[str, Any]]:
    """Restore the accepted CV peak if a re-ranked frame lacks dual coverage."""
    return _evidence_artifacts._select_key_material_view_pair_with_peak_fallback(
        group,
        event,
        views,
        infos,
        transforms,
        accepted_peak_global_ms,
        services=_services_evidence_artifacts(),
    )


def _artifact_json(
    group: ExperimentGroup,
    event: EvidenceEvent,
    artifact_type: str,
    artifact_file: str,
    view_id: str | None,
    transforms: dict[str, AlignmentTransform],
    archive_id: str | None = None,
) -> dict[str, Any]:
    return _evidence_artifacts._artifact_json(
        group,
        event,
        artifact_type,
        artifact_file,
        view_id,
        transforms,
        archive_id,
        services=_services_evidence_artifacts(),
    )


def prepare_key_material_category_layout(
    layout: ArchiveLayout,
    groups: Sequence[ExperimentGroup],
    events: Sequence[EvidenceEvent] | None = None,
    *,
    include_empty_categories: bool = True,
) -> None:
    """Create the stable experiment/action hierarchy before materialization."""
    return _evidence_artifacts.prepare_key_material_category_layout(
        layout,
        groups,
        events,
        include_empty_categories=include_empty_categories,
        services=_services_evidence_artifacts(),
    )


def write_key_material_category_index(
    layout: ArchiveLayout,
    groups: Sequence[ExperimentGroup],
    events: Sequence[EvidenceEvent],
    publisher: Any | None = None,
    *,
    include_empty_categories: bool = True,
) -> Path:
    """Write a human-browsable and machine-indexable six-category manifest."""
    return _evidence_artifacts.write_key_material_category_index(
        layout,
        groups,
        events,
        publisher,
        include_empty_categories=include_empty_categories,
        services=_services_evidence_artifacts(),
    )


def _event_participant_boxes(
    event: EvidenceEvent,
    detections: Sequence[dict[str, Any]],
    view_id: str | None = None,
    maximum_interaction_gap_norm: float = 0.08,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Filter a delivery frame to the objects participating in one event.

    Full detector output remains in the frame-evidence ledger. Key-material
    images are explanatory evidence, so background detections must not appear.
    """
    return _media_participant_grounding._event_participant_boxes(
        event,
        detections,
        view_id,
        maximum_interaction_gap_norm,
        services=_services_media_participant_grounding(),
    )


def _event_key_frame_score(
    event: EvidenceEvent, frame: FrameEvidence
) -> tuple[float, dict[str, Any]]:
    return _media_participant_grounding._event_key_frame_score(
        event, frame, services=_services_media_participant_grounding()
    )


def _best_event_frames_many(
    path: Path,
    events: Sequence[EvidenceEvent],
    transforms: dict[str, AlignmentTransform] | None = None,
) -> dict[str, tuple[FrameEvidence, float, dict[str, Any]] | None]:
    """Select participant-rich frames for all events in one ledger pass."""
    return _media_participant_grounding._best_event_frames_many(
        path, events, transforms, services=_services_media_participant_grounding()
    )


def _bounded_grounding_dino_temporal_rescue(
    event: EvidenceEvent,
    group: ExperimentGroup,
    views: Sequence[ViewInput],
    infos: dict[str, VideoInfo],
    transforms: dict[str, AlignmentTransform],
    detection_paths: dict[str, Path],
    config: dict[str, Any],
) -> tuple[float | None, dict[str, Any]]:
    """Find an explanatory participant frame without repeating a video scan.

    The immutable fine ledgers provide candidate timestamps and actor boxes.
    At most a configured number of real source frames per eligible view are
    decoded in place.  Grounding DINO is then asked only for the missing
    semantic participant classes.  No Ark request or source copy is possible
    in this helper.
    """
    return _media_participant_grounding._bounded_grounding_dino_temporal_rescue(
        event,
        group,
        views,
        infos,
        transforms,
        detection_paths,
        config,
        services=_services_media_participant_grounding(),
    )


def _participant_class_mentions(text: str) -> list[tuple[str, int, int]]:
    """Recognize canonical structured labels as well as natural-language aliases."""
    return _evidence_semantic_curation._participant_class_mentions(
        text, services=_services_evidence_semantic_curation()
    )


def _semantic_participant_conflicts(event: EvidenceEvent) -> list[dict[str, str]]:
    """Record explicit background-only claims contradicting a contact record.

    Use the model's aggregate action proof, not one occluded view or an absent
    device state change. The result marks an internal contradiction; it does
    not establish that physical contact never occurred in the source video.
    """
    return _evidence_semantic_curation._semantic_participant_conflicts(
        event, services=_services_evidence_semantic_curation()
    )


def _interaction_participant_classes(text: str) -> list[str]:
    return _evidence_semantic_curation._interaction_participant_classes(
        text, services=_services_evidence_semantic_curation()
    )


def _relabel_participant_objects(event: EvidenceEvent) -> list[str]:
    """Map model-described interaction participants back to detector classes."""
    return _evidence_semantic_curation._relabel_participant_objects(
        event, services=_services_evidence_semantic_curation()
    )


def _view_specific_participant_objects(event: EvidenceEvent, view_id: str) -> list[str]:
    """Constrain final boxes to objects Ark actually described in one view."""
    return _evidence_semantic_curation._view_specific_participant_objects(
        event, view_id, services=_services_evidence_semantic_curation()
    )


def _rerender_curated_participant_annotations(
    layout: ArchiveLayout,
    events: Sequence[EvidenceEvent],
    groups: Sequence[ExperimentGroup],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Render final boxes after semantic relabeling corrected participants."""
    return _media_participant_annotations._rerender_curated_participant_annotations(
        layout,
        events,
        groups,
        config,
        services=_services_media_participant_annotations(),
    )


def _release_auxiliary_model_caches(*, retain_on_cpu: bool = False) -> dict[str, int]:
    """Release VRAM between models; optionally retain weights in host RAM."""
    return _media_participant_grounding._release_auxiliary_model_caches(
        retain_on_cpu=retain_on_cpu, services=_services_media_participant_grounding()
    )


def _park_auxiliary_model_caches(config: dict[str, Any] | None) -> dict[str, Any]:
    """Keep warm weights only while the configured host-memory reserve exists."""
    return _media_participant_grounding._park_auxiliary_model_caches(
        config, services=_services_media_participant_grounding()
    )


def _select_manipulated_object_candidate(
    candidates: Sequence[dict[str, Any]],
    actor_boxes: Sequence[dict[str, Any]],
    *,
    canonical_class: str,
    maximum_actor_gap: float,
    pipette_minimum_aspect_ratio: float,
    grounding_dino_pipette_minimum_aspect_ratio: float | None = None,
    minimum_confidence: float = 0.0,
    pipette_center_must_overlap_actor: bool = False,
    pipette_maximum_actor_iou: float | None = None,
    preferred_grounding_terms: Sequence[str] = (),
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Choose one active object instance using contact before confidence.

    Crowded laboratory benches can contain many high-confidence static tools.
    A final participant annotation is allowed to show only the instance that
    touches or nearly touches the operator. Pipettes and tips additionally
    need an elongated box so a hand-shaped open-vocabulary false positive
    cannot satisfy the interaction gate.
    """
    return _media_participant_grounding._select_manipulated_object_candidate(
        candidates,
        actor_boxes,
        canonical_class=canonical_class,
        maximum_actor_gap=maximum_actor_gap,
        pipette_minimum_aspect_ratio=pipette_minimum_aspect_ratio,
        grounding_dino_pipette_minimum_aspect_ratio=grounding_dino_pipette_minimum_aspect_ratio,
        minimum_confidence=minimum_confidence,
        pipette_center_must_overlap_actor=pipette_center_must_overlap_actor,
        pipette_maximum_actor_iou=pipette_maximum_actor_iou,
        preferred_grounding_terms=preferred_grounding_terms,
        services=_services_media_participant_grounding(),
    )


def _select_state_container_candidate(
    candidates: Sequence[dict[str, Any]],
    actor_boxes: Sequence[dict[str, Any]],
    *,
    view_role: str,
    state_direction: str,
    maximum_actor_gap: float,
    first_person_minimum_top_offset: float = 0.0,
    closure_boxes: Sequence[dict[str, Any]] = (),
    maximum_closure_gap: float | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Select a state-bearing container without preferring background vessels.

    In a first-person post-opening frame the manipulated vessel starts at or
    below the operating hand's vertical centre.  Crowded benches often contain
    higher, unrelated open vessels that score much better in open-vocabulary
    detection.  Treat the spatial relation as a fail-closed eligibility rule,
    then rank only eligible candidates by confidence.
    """
    return _media_participant_grounding._select_state_container_candidate(
        candidates,
        actor_boxes,
        view_role=view_role,
        state_direction=state_direction,
        maximum_actor_gap=maximum_actor_gap,
        first_person_minimum_top_offset=first_person_minimum_top_offset,
        closure_boxes=closure_boxes,
        maximum_closure_gap=maximum_closure_gap,
        services=_services_media_participant_grounding(),
    )


def _grounding_dino_key_frame_detections(
    frame: np.ndarray, canonical_classes: set[str], settings: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return _media_participant_grounding._grounding_dino_key_frame_detections(
        frame,
        canonical_classes,
        settings,
        services=_services_media_participant_grounding(),
    )


def _grounding_dino_key_frame_detections_once(
    frame: np.ndarray, canonical_classes: set[str], settings: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Ground only missing participant classes with the pinned local model."""
    return _media_participant_grounding._grounding_dino_key_frame_detections_once(
        frame,
        canonical_classes,
        settings,
        services=_services_media_participant_grounding(),
    )


def _missing_state_transition_fallback_classes(
    grounded: Sequence[dict[str, Any]],
    *,
    active_object_classes: set[str],
    state_prompts: set[str],
    actor_boxes: Sequence[dict[str, Any]],
    maximum_actor_gap: float,
    closure_maximum_actor_gap: float,
    closure_minimum_confidence: float,
) -> set[str]:
    """Return only missing closure/container slots for one final state frame."""
    return _media_participant_grounding._missing_state_transition_fallback_classes(
        grounded,
        active_object_classes=active_object_classes,
        state_prompts=state_prompts,
        actor_boxes=actor_boxes,
        maximum_actor_gap=maximum_actor_gap,
        closure_maximum_actor_gap=closure_maximum_actor_gap,
        closure_minimum_confidence=closure_minimum_confidence,
        services=_services_media_participant_grounding(),
    )


def _open_vocabulary_key_frame_supplement(
    frame: np.ndarray,
    event: EvidenceEvent,
    closed_set_detections: Sequence[dict[str, Any]],
    config: dict[str, Any],
    *,
    view_role: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Ground missing manipulated objects only on final accepted key frames.

    This is deliberately not a video-wide detector.  It runs at most on the
    two final role frames for an accepted event, and admits a grounded object
    only when its box touches or nearly touches an actor box.  Thus an open-set
    model cannot reintroduce the crowded-background boxes that the final
    participant-only contract is designed to suppress.
    """
    return _media_participant_grounding._open_vocabulary_key_frame_supplement(
        frame,
        event,
        closed_set_detections,
        config,
        view_role=view_role,
        services=_services_media_participant_grounding(),
    )


def _review_bounded_cap_keyframe(
    layout: ArchiveLayout,
    event: EvidenceEvent,
    pair: tuple[str, str],
    views: Sequence[ViewInput],
    infos: dict[str, VideoInfo],
    transforms: dict[str, AlignmentTransform],
    detection_paths: dict[str, Path],
    config: dict[str, Any],
) -> tuple[float | None, dict[str, Any]]:
    """Validate the cap instance on a few neighboring event frames.

    A missing visible cap does not justify deleting a real sequence-level
    action. Keep the current frame first, then inspect bounded alternatives;
    only the reviewer may admit an existing cap proposal. All requests share
    the final-annotation budget and content-addressed cache.
    """
    return _media_key_materials._review_bounded_cap_keyframe(
        layout,
        event,
        pair,
        views,
        infos,
        transforms,
        detection_paths,
        config,
        services=_services_media_key_materials(),
    )


def materialize_key_materials(
    layout: ArchiveLayout,
    events: Sequence[EvidenceEvent],
    groups: Sequence[ExperimentGroup],
    views: Sequence[ViewInput],
    infos: dict[str, VideoInfo],
    transforms: dict[str, AlignmentTransform],
    detection_paths: dict[str, Path],
    config: dict[str, Any],
    publisher: Any | None = None,
    archive_id: str | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    materialize_event_ids: set[str] | None = None,
) -> None:
    return _media_key_materials.materialize_key_materials(
        layout,
        events,
        groups,
        views,
        infos,
        transforms,
        detection_paths,
        config,
        publisher,
        archive_id,
        progress_callback,
        materialize_event_ids,
        services=_services_media_key_materials(),
    )


def extract_temporal_review_frames(
    clip_path: Path, output_dir: Path, view_id: str, samples_per_view: int = 3
) -> list[tuple[str, Path]]:
    """Sample a bounded timeline from an already-made key clip.

    This avoids another seek through the original NAS video and gives the MLLM
    temporal evidence instead of asking it to infer an action from one still.
    """
    return _media_experiment_materials.extract_temporal_review_frames(
        clip_path,
        output_dir,
        view_id,
        samples_per_view,
        services=_services_media_experiment_materials(),
    )


def key_material_review_images(
    layout: ArchiveLayout, event: EvidenceEvent, config: dict[str, Any]
) -> list[tuple[str, Path]]:
    """Select bounded temporal evidence without discarding dual-view context.

    Timeline positions never claim an action peak. When image budget permits,
    separately include both retained key frames with their recorded times.
    """
    return _evidence_semantic_review.key_material_review_images(
        layout, event, config, services=_services_evidence_semantic_review()
    )


def analyze_key_materials(
    layout: ArchiveLayout,
    events: Sequence[EvidenceEvent],
    config: dict[str, Any],
    *,
    progress_callback: Callable[[int, int], None] | None = None,
) -> None:
    return _evidence_semantic_review.analyze_key_materials(
        layout,
        events,
        config,
        progress_callback=progress_callback,
        services=_services_evidence_semantic_review(),
    )


def _move_curated_event_media(
    layout: ArchiveLayout,
    event: EvidenceEvent,
    group: ExperimentGroup,
    quarantine_root: Path,
    *,
    destination_action: ActionType | None,
) -> dict[str, Any]:
    """Move one event atomically between formal and review material trees."""
    return _evidence_semantic_curation._move_curated_event_media(
        layout,
        event,
        group,
        quarantine_root,
        destination_action=destination_action,
        services=_services_evidence_semantic_curation(),
    )


def curate_semantically_reviewed_key_materials(
    layout: ArchiveLayout,
    events: Sequence[EvidenceEvent],
    groups: Sequence[ExperimentGroup],
    config: dict[str, Any],
    publisher: Any | None = None,
) -> tuple[list[EvidenceEvent], dict[str, Any]]:
    """Keep only semantically confirmed actions in final Key-Materials.

    CV acceptance remains in each semantic-review receipt. Rejected or
    uncertain generated media moves into a machine-quarantine tree, so recall
    disputes remain visually auditable without becoming a manual completion
    gate or being presented as confirmed key material.
    """
    return _evidence_semantic_curation.curate_semantically_reviewed_key_materials(
        layout,
        events,
        groups,
        config,
        publisher,
        services=_services_evidence_semantic_curation(),
    )


def _annotation_supports_curated_action(
    event: EvidenceEvent, annotation: dict[str, Any]
) -> bool:
    """Apply the final same-view participant contract before publication."""
    return _evidence_semantic_curation._annotation_supports_curated_action(
        event, annotation, services=_services_evidence_semantic_curation()
    )


def reconcile_visually_reviewed_participants(
    layout: ArchiveLayout,
    events: Sequence[EvidenceEvent],
    groups: Sequence[ExperimentGroup],
    semantic_curation: dict[str, Any],
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
    return _evidence_semantic_curation.reconcile_visually_reviewed_participants(
        layout,
        events,
        groups,
        semantic_curation,
        services=_services_evidence_semantic_curation(),
    )


def refresh_key_material_metadata(
    layout: ArchiveLayout,
    events: Sequence[EvidenceEvent],
    groups: Sequence[ExperimentGroup],
    transforms: dict[str, AlignmentTransform],
    archive_id: str | None = None,
) -> None:
    """Rewrite sidecars after MLLM so JSON and media never disagree."""
    return _evidence_artifacts.refresh_key_material_metadata(
        layout,
        events,
        groups,
        transforms,
        archive_id,
        services=_services_evidence_artifacts(),
    )


def write_timestamp_tables(
    layout: ArchiveLayout, events: Sequence[EvidenceEvent], create_xlsx: bool
) -> None:
    return _evidence_publication.write_timestamp_tables(
        layout, events, create_xlsx, services=_services_evidence_publication()
    )


def write_screening_notes(
    layout: ArchiveLayout,
    segments: Sequence[ExperimentSegment],
    events: Sequence[EvidenceEvent],
    rejected_candidates: Sequence[dict[str, Any]],
    all_views: Sequence[ViewInput],
) -> None:
    return _evidence_publication.write_screening_notes(
        layout,
        segments,
        events,
        rejected_candidates,
        all_views,
        services=_services_evidence_publication(),
    )


def evidence_package_eval(
    root: Path,
    groups: Sequence[ExperimentGroup],
    segments: Sequence[ExperimentSegment],
    key_events: Sequence[EvidenceEvent],
    transforms: dict[str, AlignmentTransform],
) -> dict[str, Any]:
    return _evidence_publication.evidence_package_eval(
        root,
        groups,
        segments,
        key_events,
        transforms,
        services=_services_evidence_publication(),
    )


def finalize_archive(
    layout: ArchiveLayout,
    manifest: RunManifest,
    infos: dict[str, VideoInfo],
    transforms: dict[str, AlignmentTransform],
    events: list[EvidenceEvent],
    segments: list[ExperimentSegment],
    groups: list[ExperimentGroup],
    key_events: list[EvidenceEvent],
    physical_changes: list[PhysicalChange],
    rejected_candidates: Sequence[dict[str, Any]],
    config: dict[str, Any],
    disk_report: dict[str, int],
    run_metrics: dict[str, Any] | None = None,
) -> RunSummary:
    return _evidence_publication.finalize_archive(
        layout,
        manifest,
        infos,
        transforms,
        events,
        segments,
        groups,
        key_events,
        physical_changes,
        rejected_candidates,
        config,
        disk_report,
        run_metrics,
        services=_services_evidence_publication(),
    )


def _services_media_participant_annotations() -> (
    _media_participant_annotations.ParticipantAnnotationsServices
):
    return _media_participant_annotations.ParticipantAnnotationsServices(
        ParticipantVisualReviewer=ParticipantVisualReviewer,
        SelectiveVerificationBudget=SelectiveVerificationBudget,
        _event_participant_boxes=_event_participant_boxes,
        _grounding_dino_key_frame_detections=_grounding_dino_key_frame_detections,
        _key_material_view_pair=_key_material_view_pair,
        _open_vocabulary_key_frame_supplement=_open_vocabulary_key_frame_supplement,
        _park_auxiliary_model_caches=_park_auxiliary_model_caches,
        _view_specific_participant_objects=_view_specific_participant_objects,
        _write_aligned_frame=_write_aligned_frame,
        analyze_liquid_semantics=analyze_liquid_semantics,
        archive_relative_posix=archive_relative_posix,
        audit_participant_continuity=audit_participant_continuity,
        plan_selective_key_material_verification=plan_selective_key_material_verification,
        write_annotated_frame=write_annotated_frame,
        write_json=write_json,
    )
