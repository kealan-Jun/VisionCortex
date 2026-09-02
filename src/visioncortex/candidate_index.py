from __future__ import annotations

import math
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Sequence

from .detection import iter_frame_evidence
from .schemas import FrameEvidence, VideoInfo, ViewInput


_ACTOR_CLASSES = {"hand", "gloved_hand"}
_NON_ACTION_CLASSES = {"person", "face", "background", "lab_coat", "PPE_Storage"}
_ANCHOR_CLASSES = {
    "balance",
    "magnetic_stirrer",
    "beaker",
    "reagent_bottle",
    "reagent_bottle_open",
    "sample_bottle",
    "sample_bottle_blue",
    "tube",
    "container",
    "tube_cap",
    "bottle_cap",
    "tube_rack",
    "magnetic_stir_bar",
}


class CoarseFrameIndex:
    """Bounded-memory, rebuildable index over coarse frame ledgers."""

    def __init__(self, path: Path):
        self.path = path

    def iter_frames(
        self,
        view_id: str,
        *,
        start_ms: float | None = None,
        end_ms: float | None = None,
        activity_only: bool = False,
        anchors_only: bool = False,
        missing_actor_or_object: bool = False,
    ) -> Iterable[FrameEvidence]:
        clauses = ["view_id = ?"]
        parameters: list[Any] = [view_id]
        if start_ms is not None:
            clauses.append("global_ms >= ?")
            parameters.append(float(start_ms))
        if end_ms is not None:
            clauses.append("global_ms <= ?")
            parameters.append(float(end_ms))
        if activity_only:
            clauses.append(
                "((has_actor = 1 AND object_count > 0) OR object_count >= 2)"
            )
        if anchors_only:
            clauses.append("anchor_count > 0")
        if missing_actor_or_object:
            clauses.append("(has_actor = 0 OR object_count = 0)")
        query = (
            "SELECT payload_json FROM coarse_frames WHERE "
            + " AND ".join(clauses)
            + " ORDER BY global_ms, frame_index"
        )
        connection = sqlite3.connect(self.path)
        try:
            for (payload,) in connection.execute(query, parameters):
                yield FrameEvidence.model_validate_json(payload)
        finally:
            connection.close()

    def iter_motion_samples(
        self, view_id: str
    ) -> Iterable[tuple[float, float]]:
        connection = sqlite3.connect(self.path)
        try:
            for global_ms, motion_score in connection.execute(
                "SELECT global_ms, motion_score FROM coarse_frames "
                "WHERE view_id = ? ORDER BY global_ms, frame_index",
                (view_id,),
            ):
                yield float(global_ms), float(motion_score)
        finally:
            connection.close()


def _expected_samples(info: VideoInfo, sample_fps: float) -> int:
    periods = [
        max(0.0, float(segment.duration_ms))
        for segment in info.segments
    ] if info.segments else [max(0.0, float(info.duration_ms))]
    return sum(
        max(1, int(math.ceil(duration / 1000.0 * sample_fps - 1e-9)))
        for duration in periods
    )


def _same_physical_span(info: VideoInfo, left_ms: float, right_ms: float) -> bool:
    if not info.segments:
        return True
    return any(
        segment.virtual_start_ms <= left_ms <= segment.virtual_end_ms
        and segment.virtual_start_ms <= right_ms <= segment.virtual_end_ms
        for segment in info.segments
    )


def build_coarse_frame_index(
    index_path: Path,
    views: Sequence[ViewInput],
    infos: dict[str, VideoInfo],
    detection_paths: dict[str, Path],
    *,
    sample_fps: float,
    minimum_coverage_ratio: float = 0.98,
    maximum_gap_periods: float = 4.0,
) -> tuple[CoarseFrameIndex, dict[str, Any]]:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    if index_path.exists():
        index_path.unlink()
    connection = sqlite3.connect(index_path)
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.executescript(
        """
        CREATE TABLE coarse_frames (
            view_id TEXT NOT NULL,
            frame_index INTEGER NOT NULL,
            local_ms REAL NOT NULL,
            global_ms REAL NOT NULL,
            motion_score REAL NOT NULL,
            raw_motion_score REAL NOT NULL,
            has_actor INTEGER NOT NULL,
            object_count INTEGER NOT NULL,
            anchor_count INTEGER NOT NULL,
            minimum_confidence REAL,
            payload_json TEXT NOT NULL,
            PRIMARY KEY (view_id, local_ms)
        ) WITHOUT ROWID;
        CREATE INDEX coarse_frames_global
            ON coarse_frames(view_id, global_ms);
        CREATE INDEX coarse_frames_activity
            ON coarse_frames(view_id, has_actor, object_count, anchor_count, global_ms);
        """
    )
    period_ms = 1000.0 / max(sample_fps, 1e-9)
    view_reports: list[dict[str, Any]] = []
    try:
        for view in views:
            info = infos[view.view_id]
            actual = 0
            first_local_ms: float | None = None
            last_local_ms: float | None = None
            previous_local_ms: float | None = None
            maximum_gap_ms = 0.0
            gaps: list[dict[str, float]] = []
            batch: list[tuple[Any, ...]] = []
            for frame in iter_frame_evidence(detection_paths[view.view_id]):
                global_ms = float(
                    frame.global_ms if frame.global_ms is not None else frame.local_ms
                )
                classes = {box.class_name for box in frame.detections}
                objects = classes - _ACTOR_CLASSES - _NON_ACTION_CLASSES
                anchors = classes & _ANCHOR_CLASSES
                confidences = [float(box.confidence) for box in frame.detections]
                batch.append(
                    (
                        view.view_id,
                        int(frame.frame_index),
                        float(frame.local_ms),
                        global_ms,
                        float(frame.motion_score),
                        float(frame.raw_motion_score),
                        int(bool(classes & _ACTOR_CLASSES)),
                        len(objects),
                        len(anchors),
                        min(confidences) if confidences else None,
                        frame.model_dump_json(),
                    )
                )
                if len(batch) >= 1000:
                    connection.executemany(
                        "INSERT OR REPLACE INTO coarse_frames VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        batch,
                    )
                    batch.clear()
                actual += 1
                first_local_ms = (
                    frame.local_ms if first_local_ms is None else first_local_ms
                )
                if previous_local_ms is not None and _same_physical_span(
                    info, previous_local_ms, frame.local_ms
                ):
                    gap_ms = float(frame.local_ms - previous_local_ms)
                    maximum_gap_ms = max(maximum_gap_ms, gap_ms)
                    if gap_ms > period_ms * maximum_gap_periods and len(gaps) < 100:
                        gaps.append(
                            {
                                "start_local_ms": previous_local_ms,
                                "end_local_ms": float(frame.local_ms),
                                "gap_ms": gap_ms,
                            }
                        )
                previous_local_ms = float(frame.local_ms)
                last_local_ms = float(frame.local_ms)
            if batch:
                connection.executemany(
                    "INSERT OR REPLACE INTO coarse_frames VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    batch,
                )
            expected = _expected_samples(info, sample_fps)
            ratio = min(1.0, actual / max(1, expected))
            view_reports.append(
                {
                    "view_id": view.view_id,
                    "expected_sample_count": expected,
                    "actual_sample_count": actual,
                    "coverage_ratio": round(ratio, 6),
                    "first_local_ms": first_local_ms,
                    "last_local_ms": last_local_ms,
                    "maximum_internal_gap_ms": round(maximum_gap_ms, 3),
                    "unexpected_gap_count": len(gaps),
                    "unexpected_gaps": gaps,
                    "formal_evidence_ready": bool(
                        actual > 0
                        and ratio >= minimum_coverage_ratio
                        and not gaps
                    ),
                }
            )
        connection.commit()
    finally:
        connection.close()
    report = {
        "schema_version": "visioncortex-coarse-frame-index/1",
        "index_path": str(index_path),
        "sample_fps": sample_fps,
        "minimum_coverage_ratio": minimum_coverage_ratio,
        "maximum_gap_periods": maximum_gap_periods,
        "view_count": len(view_reports),
        "row_count": sum(item["actual_sample_count"] for item in view_reports),
        "formal_evidence_ready": all(
            item["formal_evidence_ready"] for item in view_reports
        ),
        "views": view_reports,
    }
    return CoarseFrameIndex(index_path), report
