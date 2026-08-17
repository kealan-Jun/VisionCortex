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


class LiquidStateStatus(str, Enum):
    """How strongly the visual evidence supports a liquid-state claim."""

    NOT_EVALUATED = "not_evaluated"
    OBSERVED = "observed"
    INFERRED = "inferred"
    UNCERTAIN = "uncertain"
    OCCLUDED = "occluded"


class LiquidFlowDirection(str, Enum):
    SOURCE_TO_TARGET = "source_to_target"
    TARGET_TO_SOURCE = "target_to_source"
    INTO_TOOL = "into_tool"
    OUT_OF_TOOL = "out_of_tool"
    UNKNOWN = "unknown"


class NormalizedPoint(BaseModel):
    """One image-space point expressed independently of source resolution."""

    model_config = ConfigDict(extra="forbid")

    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)


class LiquidViewObservation(BaseModel):
    """Direct, view-specific liquid evidence inside a bounded container ROI."""

    model_config = ConfigDict(extra="forbid")

    view_id: str = Field(min_length=1)
    view_role: ViewRole
    timestamp_us: int = Field(ge=0)
    container_id: str | None = None
    liquid_present: bool | None = None
    mask_ref: str | None = None
    meniscus_polyline: list[NormalizedPoint] = Field(default_factory=list)
    fill_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    visible_flow: bool | None = None
    flow_direction: LiquidFlowDirection = LiquidFlowDirection.UNKNOWN
    visibility: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    observed_facts: list[str] = Field(default_factory=list)
    uncertainty: list[str] = Field(default_factory=list)


class LiquidContainerState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    container_id: str
    liquid_present: bool | None = None
    fill_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    observed_by_views: list[str] = Field(default_factory=list)


class LiquidStateEvidence(BaseModel):
    """Structured specialist evidence; absent unless the opt-in expert runs."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "visioncortex-liquid-state/1.0.0"
    status: LiquidStateStatus = LiquidStateStatus.NOT_EVALUATED
    backend: str = "disabled"
    source_before: LiquidContainerState | None = None
    source_after: LiquidContainerState | None = None
    target_before: LiquidContainerState | None = None
    target_after: LiquidContainerState | None = None
    per_view_observations: list[LiquidViewObservation] = Field(default_factory=list)
    visible_flow: bool | None = None
    flow_direction: LiquidFlowDirection = LiquidFlowDirection.UNKNOWN
    state_change_confirmed: bool | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    observed_facts: list[str] = Field(default_factory=list)
    supported_inferences: list[str] = Field(default_factory=list)
    uncertain_claims: list[str] = Field(default_factory=list)
    model_receipt: dict[str, Any] = Field(default_factory=dict)


class LiquidExpertSample(BaseModel):
    """A reference-only request passed to a registered specialist adapter."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str = Field(min_length=1)
    event_id: str = Field(min_length=1)
    group_id: str = Field(min_length=1)
    view_id: str = Field(min_length=1)
    view_role: ViewRole
    timestamp_us: int = Field(ge=0)
    media_ref: str = Field(min_length=1)
    container_id: str | None = None
    roi_xyxy_norm: tuple[float, float, float, float] | None = None

    @field_validator("roi_xyxy_norm")
    @classmethod
    def validate_roi(
        cls, value: tuple[float, float, float, float] | None
    ) -> tuple[float, float, float, float] | None:
        if value is None:
            return None
        x1, y1, x2, y2 = value
        if not all(0.0 <= item <= 1.0 for item in value) or x2 <= x1 or y2 <= y1:
            raise ValueError("roi_xyxy_norm must be a normalized non-empty box")
        return x1, y1, x2, y2


class LiquidBenchmarkSample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_id: str
    event_id: str
    group_id: str
    split: Literal["train", "validation", "test"]
    image_refs: list[str] = Field(default_factory=list)
    clip_refs: list[str] = Field(default_factory=list)
    ground_truth: LiquidStateEvidence | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class LiquidBenchmarkManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "visioncortex-liquid-benchmark/1.0.0"
    benchmark_id: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source_archive: str
    media_policy: Literal["references_only"] = "references_only"
    samples: list[LiquidBenchmarkSample] = Field(default_factory=list)


class VideoSegmentInput(BaseModel):
    """One immutable recorder segment on a view's virtual timeline."""

    model_config = ConfigDict(extra="forbid")

    video: Path
    timestamps_csv: Path | None = None


class ViewInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    view_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    role: ViewRole
    video: Path | None = None
    timestamps_csv: Path | None = None
    segments: list[VideoSegmentInput] = Field(default_factory=list)
    calibration_hint_ms: float = 0.0

    @model_validator(mode="after")
    def validate_source(self) -> "ViewInput":
        if self.video is None and not self.segments:
            raise ValueError("view must provide either video or segments")
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


class VideoInfo(BaseModel):
    path: Path
    duration_ms: float
    fps: float
    width: int
    height: int
    frame_count: int
    size_bytes: int = 0
    segments: list[VideoSegmentInfo] = Field(default_factory=list)


class TimestampPoint(BaseModel):
    frame_index: int
    local_ms: float
    source_ms: float | None = None


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

    def to_global(self, local_ms: float) -> float:
        return self.scale * local_ms + self.offset_ms + self.visual_correction_ms

    def to_local(self, global_ms: float) -> float:
        return (global_ms - self.offset_ms - self.visual_correction_ms) / self.scale


class BoxEvidence(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    xyxy_norm: tuple[float, float, float, float]
    track_id: int | None = None
    roi_motion: float = 0.0

    @field_validator("class_name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        return value.strip().replace("-", "_")


class FrameEvidence(BaseModel):
    view_id: str
    role: ViewRole
    frame_index: int
    local_ms: float
    global_ms: float | None = None
    width: int
    height: int
    motion_score: float = 0.0
    detections: list[BoxEvidence] = Field(default_factory=list)


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


class EvidenceEvent(BaseModel):
    event_id: str
    action_type: ActionType
    global_start_ms: float
    global_end_ms: float
    key_global_ms: float
    objects: list[str]
    confidence: float
    accepted: bool
    audit_reason: str
    supporting_views: list[str]
    supporting_roles: list[ViewRole]
    candidates: list[ActionCandidate]
    uncertainty: list[str] = Field(default_factory=list)
    key_frames: dict[str, str] = Field(default_factory=dict)
    key_clips: dict[str, str] = Field(default_factory=dict)
    model_understanding: dict[str, Any] | None = None
    liquid_state: LiquidStateEvidence | None = None


class ExperimentSegment(BaseModel):
    segment_id: str
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
    continuity_type: Literal["independent", "continuous"]
    atomic_experiment_ids: list[str]
    global_start_ms: float
    global_end_ms: float
    participating_views: list[str]
    first_person_view: str
    third_person_view: str
    continuity_reason: str
    experiment_name: str = "待模型命名实验"
    experiment_name_en: str = "Unnamed-Experiment"
    archive_folder: str | None = None
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
