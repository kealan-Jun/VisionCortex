from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ViewRole(str, Enum):
    FIRST_PERSON = "first_person"
    THIRD_PERSON = "third_person"


class ActionType(str, Enum):
    HAND_OBJECT_CONTACT = "hand_object_contact"
    OBJECT_MOVEMENT = "object_movement"
    LIQUID_MOVEMENT = "liquid_movement"
    CONTAINER_STATE_CHANGE = "container_state_change"
    DEVICE_PANEL_OPERATION = "device_panel_operation"
    # A source-contact -> withdrawal/transport -> distinct-target-contact
    # pipette chain is directly observable even when the microlitre payload is
    # not. Keep that operational fact separate from LIQUID_MOVEMENT so an
    # invisible fluid or occluded plunger is never promoted to visible liquid.
    PIPETTE_TRANSFER_OPERATION = "pipette_transfer_operation"


class VideoSegmentInput(BaseModel):
    """One immutable recorder segment on a view's virtual timeline."""

    model_config = ConfigDict(extra="forbid")

    video: Path
    timestamps_csv: Path | None = None
    audio: Path | None = None
    audio_offset_ms: float | None = Field(default=None, allow_inf_nan=False)


class ViewInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    view_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    role: ViewRole
    video: Path | None = None
    timestamps_csv: Path | None = None
    audio: Path | None = None
    audio_offset_ms: float | None = Field(default=None, allow_inf_nan=False)
    segments: list[VideoSegmentInput] = Field(default_factory=list)
    calibration_hint_ms: float = 0.0

    @model_validator(mode="after")
    def validate_source(self) -> "ViewInput":
        if self.video is None and not self.segments:
            raise ValueError("view must provide either video or segments")
        if self.segments and (self.audio is not None or self.audio_offset_ms is not None):
            raise ValueError("segmented views must declare audio on each segment")
        if self.video is not None and self.segments:
            raise ValueError("view cannot provide both video and segments")
        return self


class RunManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    views: list[ViewInput] = Field(min_length=2)

    @model_validator(mode="after")
    def validate_views(self) -> "RunManifest":
        ids = [v.view_id for v in self.views]
        if len(ids) != len(set(ids)):
            raise ValueError("view_id 必须唯一")
        roles = {v.role for v in self.views}
        if roles != {ViewRole.FIRST_PERSON, ViewRole.THIRD_PERSON}:
            raise ValueError("至少需要一路 first_person 和一路 third_person")
        return self


class VideoSegmentInfo(BaseModel):
    path: Path
    timestamps_csv: Path | None = None
    virtual_start_ms: float
    virtual_end_ms: float
    frame_start_index: int
    duration_ms: float
    fps: float
    width: int
    height: int
    frame_count: int
    size_bytes: int = 0
    # The recorder clock can cover pauses during which no RGB sample was
    # written.  Keep that wall-clock coverage for audit, but never use it as
    # the physical MP4 decode duration.
    source_clock_duration_ms: float | None = None
    media_timing_source: str | None = None


class VideoInfo(BaseModel):
    path: Path
    duration_ms: float
    fps: float
    width: int
    height: int
    frame_count: int
    size_bytes: int = 0
    source_clock_duration_ms: float | None = None
    media_timing_source: str | None = None
    segments: list[VideoSegmentInfo] = Field(default_factory=list)


class TimestampPoint(BaseModel):
    frame_index: int
    local_ms: float
    source_ms: float | None = None
    clock_sync_valid: bool | None = None
    source_column: str | None = None
    row_number: int | None = None


class AlignmentSegmentTransform(BaseModel):
    """One physical recorder segment's mapping onto the shared timeline."""

    segment_index: int
    local_start_ms: float
    local_end_ms: float
    scale: float = 1.0
    offset_ms: float = 0.0
    csv_sample_count: int = 0
    csv_rmse_ms: float | None = None
    confidence: float = 0.0
    uncertainty_ms: float = 0.0
    state: Literal["aligned", "uncertain", "failed"] = "uncertain"
    clock_anomalies: list[str] = Field(default_factory=list)

    def to_global(self, local_ms: float) -> float:
        return self.scale * local_ms + self.offset_ms

    def to_local(self, global_ms: float) -> float:
        return (global_ms - self.offset_ms) / self.scale


class AlignmentTransform(BaseModel):
    view_id: str
    reference_view_id: str
    scale: float = 1.0
    offset_ms: float = 0.0
    csv_match_count: int = 0
    csv_match_ratio: float = 0.0
    csv_rmse_ms: float | None = None
    visual_correction_ms: float = 0.0
    visual_confidence: float = 0.0
    confidence: float = 0.0
    state: Literal["aligned", "uncertain", "failed"] = "uncertain"
    failure_reason: str | None = None
    anchor_details: list[dict[str, Any]] = Field(default_factory=list)
    alignment_basis: str = "legacy_affine"
    clock_sample_count: int = 0
    clock_valid_sample_count: int = 0
    clock_anomalies: list[str] = Field(default_factory=list)
    uncertainty_ms: float = 0.0
    local_coverage_start_ms: float | None = None
    local_coverage_end_ms: float | None = None
    segment_transforms: list[AlignmentSegmentTransform] = Field(default_factory=list)
    runtime: dict[str, Any] = Field(default_factory=dict)

    def _segment_for_local(self, local_ms: float) -> AlignmentSegmentTransform | None:
        if not self.segment_transforms:
            return None
        containing = next(
            (
                segment
                for index, segment in enumerate(self.segment_transforms)
                if segment.local_start_ms <= local_ms
                and (
                    local_ms < segment.local_end_ms
                    or (
                        index == len(self.segment_transforms) - 1
                        and local_ms <= segment.local_end_ms
                    )
                )
                and segment.state != "failed"
            ),
            None,
        )
        if containing is not None:
            return containing
        return min(
            (segment for segment in self.segment_transforms if segment.state != "failed"),
            key=lambda segment: min(
                abs(local_ms - segment.local_start_ms),
                abs(local_ms - segment.local_end_ms),
            ),
            default=None,
        )

    def _segment_for_global(self, global_ms: float) -> AlignmentSegmentTransform | None:
        if not self.segment_transforms:
            return None
        adjusted = global_ms - self.visual_correction_ms
        containing = next(
            (
                segment
                for index, segment in enumerate(self.segment_transforms)
                if min(
                    segment.to_global(segment.local_start_ms),
                    segment.to_global(segment.local_end_ms),
                )
                <= adjusted
                and (
                    adjusted
                    < max(
                        segment.to_global(segment.local_start_ms),
                        segment.to_global(segment.local_end_ms),
                    )
                    or (
                        index == len(self.segment_transforms) - 1
                        and adjusted
                        <= max(
                            segment.to_global(segment.local_start_ms),
                            segment.to_global(segment.local_end_ms),
                        )
                    )
                )
                and segment.state != "failed"
            ),
            None,
        )
        if containing is not None:
            return containing
        return min(
            (segment for segment in self.segment_transforms if segment.state != "failed"),
            key=lambda segment: min(
                abs(adjusted - segment.to_global(segment.local_start_ms)),
                abs(adjusted - segment.to_global(segment.local_end_ms)),
            ),
            default=None,
        )

    def to_global(self, local_ms: float) -> float:
        segment = self._segment_for_local(local_ms)
        if segment is not None:
            return segment.to_global(local_ms) + self.visual_correction_ms
        return self.scale * local_ms + self.offset_ms + self.visual_correction_ms

    def to_local(self, global_ms: float) -> float:
        segment = self._segment_for_global(global_ms)
        if segment is not None:
            return segment.to_local(global_ms - self.visual_correction_ms)
        return (global_ms - self.offset_ms - self.visual_correction_ms) / self.scale

    def is_available_at_global(self, global_ms: float) -> bool:
        local_ms = self.to_local(global_ms)
        if self.segment_transforms:
            return any(
                segment.state != "failed"
                and segment.local_start_ms <= local_ms <= segment.local_end_ms
                for segment in self.segment_transforms
            )
        if self.local_coverage_start_ms is None or self.local_coverage_end_ms is None:
            return self.state != "failed"
        return (
            self.state != "failed"
            and self.local_coverage_start_ms <= local_ms <= self.local_coverage_end_ms
        )


class BoxEvidence(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    xyxy_norm: tuple[float, float, float, float]
    track_id: int | None = None
    roi_motion: float = 0.0
    appearance_signature: tuple[float, ...] = ()

    @field_validator("class_name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        return value.strip().replace("-", "_")


class DuplicateBoxRemoval(BaseModel):
    model_config = ConfigDict(extra="forbid")

    removed_input_index: int = Field(ge=0)
    retained_input_index: int = Field(ge=0)
    iou: float = Field(gt=0.0, le=1.0)


class DetectionDuplicateSuppression(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["visioncortex-detection-duplicate-suppression/1"] = (
        "visioncortex-detection-duplicate-suppression/1"
    )
    iou_threshold: float = Field(gt=0.0, le=1.0)
    raw_detections: list[BoxEvidence]
    retained_input_indices: list[int]
    removals: list[DuplicateBoxRemoval]
    scope: Literal["one_source_frame_before_tracking"] = "one_source_frame_before_tracking"

    @model_validator(mode="after")
    def validate_index_partition(self) -> "DetectionDuplicateSuppression":
        kept = self.retained_input_indices
        removed = [item.removed_input_index for item in self.removals]
        if sorted(kept + removed) != list(range(len(self.raw_detections))):
            raise ValueError("Duplicate suppression must account for every raw detection once")
        if any(item.retained_input_index not in kept for item in self.removals):
            raise ValueError("A suppressed box must reference a retained raw detection")
        return self


class SourceFrameIdentity(BaseModel):
    """Physical decoder provenance, distinct from the requested sampling grid."""

    schema_version: Literal["decoder-position-pts/1"] = "decoder-position-pts/1"
    source_path: Path
    source_size_bytes: int = Field(ge=0)
    source_mtime_ns: int = Field(ge=0)
    status: Literal["resolved", "unavailable", "ambiguous"]
    packet_position: int | None = Field(default=None, ge=0)
    source_pts: int | None = None
    time_base: str | None = None
    source_frame_index: int | None = Field(default=None, ge=0)
    decoded_pixels_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reason: str | None = None
    terminal_frame_hold: bool = False

    @model_validator(mode="after")
    def validate_resolved_identity(self) -> "SourceFrameIdentity":
        from fractions import Fraction

        if self.status == "resolved":
            if self.packet_position is None or self.source_pts is None or self.time_base is None:
                raise ValueError("Resolved source frames require decoder position and native PTS")
            try:
                time_base = Fraction(self.time_base)
            except (ValueError, ZeroDivisionError) as exc:
                raise ValueError("Invalid source frame time base") from exc
            if time_base <= 0:
                raise ValueError("Source frame time base must be positive")
        elif any(value is not None for value in (self.source_pts, self.time_base, self.source_frame_index)):
            raise ValueError("Unresolved source frames cannot claim native timestamps or indices")
        return self


class FrameEvidence(BaseModel):
    view_id: str
    role: ViewRole
    frame_index: int
    local_ms: float
    global_ms: float | None = None
    width: int
    height: int
    motion_score: float = 0.0
    raw_motion_score: float = 0.0
    motion_probe_score: float | None = None
    motion_probe_raw_score: float | None = None
    camera_motion_compensated: bool = False
    camera_motion_method: str | None = None
    motion_quality_state: str = "usable"
    detections: list[BoxEvidence] = Field(default_factory=list)
    duplicate_suppression: DetectionDuplicateSuppression | None = None
    source_frame: SourceFrameIdentity | None = None


class ActionCandidate(BaseModel):
    candidate_id: str
    action_type: ActionType
    view_id: str
    role: ViewRole
    local_start_ms: float
    local_end_ms: float
    global_start_ms: float
    global_end_ms: float
    key_global_ms: float
    objects: list[str]
    confidence: float
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    uncertainty: list[str] = Field(default_factory=list)
    instance_signature: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)


class EvidenceEvent(BaseModel):
    event_id: str
    action_type: ActionType
    global_start_ms: float
    global_end_ms: float
    key_global_ms: float
    objects: list[str]
    confidence: float
    accepted: bool
    formal_admission_status: Literal[
        "formal", "provisional", "rejected"
    ] | None = None
    audit_reason: str
    supporting_views: list[str]
    supporting_roles: list[ViewRole]
    candidates: list[ActionCandidate]
    uncertainty: list[str] = Field(default_factory=list)
    observability: dict[str, Any] = Field(default_factory=dict)
    state_machine: dict[str, Any] = Field(default_factory=dict)
    semantic_review: dict[str, Any] | None = None
    key_frames: dict[str, str] = Field(default_factory=dict)
    key_clips: dict[str, str] = Field(default_factory=dict)
    model_understanding: dict[str, Any] | None = None
    event_fingerprint: str | None = None
    core_global_start_ms: float | None = None
    core_global_end_ms: float | None = None
    evidence_global_start_ms: float | None = None
    evidence_global_end_ms: float | None = None

    @model_validator(mode="after")
    def normalize_formal_admission(self) -> EvidenceEvent:
        if self.formal_admission_status is None:
            self.formal_admission_status = (
                "formal" if self.accepted else "rejected"
            )
        self.accepted = self.formal_admission_status != "rejected"
        if self.core_global_start_ms is None:
            self.core_global_start_ms = float(self.global_start_ms)
        if self.core_global_end_ms is None:
            self.core_global_end_ms = float(self.global_end_ms)
        if self.evidence_global_start_ms is None:
            self.evidence_global_start_ms = float(self.global_start_ms)
        if self.evidence_global_end_ms is None:
            self.evidence_global_end_ms = float(self.global_end_ms)
        return self


def event_is_formal(event: EvidenceEvent) -> bool:
    """Return the single admission predicate used by delivery stages."""

    return bool(
        event.accepted and event.formal_admission_status == "formal"
    )


def set_event_admission(
    event: EvidenceEvent,
    status: Literal["formal", "provisional", "rejected"],
) -> None:
    event.formal_admission_status = status
    event.accepted = status != "rejected"


class ExperimentSegment(BaseModel):
    segment_id: str
    # Stable across insertions of unrelated earlier segments. ``segment_id``
    # remains the human-readable timeline position for backward compatibility.
    segment_uid: str | None = None
    global_start_ms: float
    global_end_ms: float
    event_ids: list[str]
    participating_views: list[str]
    rejected_views: dict[str, str] = Field(default_factory=dict)
    clips: dict[str, str] = Field(default_factory=dict)
    aligned_multiview_clip: str | None = None
    micro_segments: list[dict[str, Any]] = Field(default_factory=list)
    experiment_name: str | None = None
    experiment_name_en: str | None = None
    semantic_understanding: dict[str, Any] | None = None
    group_id: str | None = None


class ExperimentGroup(BaseModel):
    """One independently archived experiment or one continuous experiment chain."""

    group_id: str
    # Stable evidence identity; ``group_id`` remains the display/order label.
    group_uid: str | None = None
    continuity_type: Literal["independent", "continuous"]
    atomic_experiment_ids: list[str]
    global_start_ms: float
    global_end_ms: float
    participating_views: list[str]
    first_person_view: str
    third_person_view: str
    continuity_reason: str
    # Original-video reviews of proposed cuts; not physical-action acceptance.
    boundary_reviews: list[dict[str, Any]] = Field(default_factory=list)
    # Semantic workflow units are distinct from CV atomic_experiment_ids.
    workflow_units: list[dict[str, Any]] = Field(default_factory=list)
    workflow_kind: Literal["unresolved", "independent_experiment", "continuous_workflow"] = "unresolved"
    completion_status: Literal["unreviewed", "observed_complete", "ongoing_at_recording_end", "unresolved"] = "unreviewed"
    completion_reason: str = ""
    boundary_extension_requires_step_review: bool = False
    # Half-open global intervals; an absent third-person view means no verified
    # workstation correspondence, not permission to reuse the previous camera.
    view_timeline: list[dict[str, Any]] = Field(default_factory=list)
    experiment_name: str = "待模型命名实验"
    experiment_name_en: str = "Unnamed-Experiment"
    archive_folder: str | None = None
    source_archive_folders: list[str] = Field(default_factory=list)
    model_understanding: dict[str, Any] | None = None
    videos: dict[str, str] = Field(default_factory=dict)
    video_json: dict[str, str] = Field(default_factory=dict)
    key_event_ids: list[str] = Field(default_factory=list)


class PhysicalChange(BaseModel):
    change_id: str
    event_id: str
    global_ms: float
    change_type: str
    object_names: list[str]
    before_state: str | None = None
    after_state: str | None = None
    supporting_views: list[str]
    confidence: float
    uncertainty: list[str] = Field(default_factory=list)


class RunSummary(BaseModel):
    schema_version: str = "2.0.0"
    experiment_id: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    views: list[ViewInput]
    alignments: list[AlignmentTransform]
    events: list[EvidenceEvent]
    segments: list[ExperimentSegment]
    experiment_groups: list[ExperimentGroup] = Field(default_factory=list)
    physical_change_log: list[PhysicalChange]
    stats: dict[str, Any] = Field(default_factory=dict)
