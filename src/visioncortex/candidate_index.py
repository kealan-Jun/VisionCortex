from __future__ import annotations

import hashlib
import math
import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Sequence

from .detection import iter_frame_evidence
from .schemas import ActionCandidate, FrameEvidence, VideoInfo, ViewInput


def local_frame_index_path(config: dict, work_dir: Path, filename: str) -> Path:
    """Both entry points keep rebuildable SQLite/WAL indexes off SMB storage."""
    runtime_root = Path(config["storage"]["local_runtime_root"]).resolve()
    namespace = hashlib.sha256(str(work_dir.resolve()).encode("utf-8")).hexdigest()
    return runtime_root / "state" / "frame-indexes" / namespace / filename


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


class FineFrameIndex:
    """Incremental, rebuildable index over fine-scan frame shards.

    JSONL ledgers remain the authoritative frame evidence.  This SQLite file is
    a bounded-memory query and merge layer used only while the existing fine
    stage is running.
    """

    def __init__(self, path: Path):
        self.path = path

    def iter_local_times(self, view_id: str) -> Iterable[float]:
        """Read coverage keys without reparsing detection payloads."""
        connection = sqlite3.connect(self.path)
        try:
            for (local_ms,) in connection.execute(
                "SELECT local_ms FROM fine_frames WHERE view_id = ? ORDER BY local_ms",
                (view_id,),
            ):
                yield float(local_ms)
        finally:
            connection.close()

    def iter_frames(
        self,
        view_id: str,
        *,
        start_ms: float | None = None,
        end_ms: float | None = None,
    ) -> Iterable[FrameEvidence]:
        clauses = ["view_id = ?"]
        parameters: list[Any] = [view_id]
        if start_ms is not None:
            clauses.append("local_ms >= ?")
            parameters.append(float(start_ms))
        if end_ms is not None:
            clauses.append("local_ms <= ?")
            parameters.append(float(end_ms))
        connection = sqlite3.connect(self.path)
        try:
            query = (
                "SELECT payload_json FROM fine_frames WHERE "
                + " AND ".join(clauses)
                + " ORDER BY local_ms, frame_index"
            )
            for (payload,) in connection.execute(query, parameters):
                yield FrameEvidence.model_validate_json(payload)
        finally:
            connection.close()

    def materialize_ledgers(
        self,
        output_dir: Path,
        views: Sequence[ViewInput],
    ) -> dict[str, Path]:
        """Write one final authoritative-compatible JSONL ledger per view."""

        output_dir.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        outputs: dict[str, Path] = {}
        try:
            for view in views:
                output = output_dir / f"{view.view_id}.detections.jsonl"
                temporary = output.with_suffix(output.suffix + ".tmp")
                with temporary.open(
                    "w", encoding="utf-8", buffering=1024 * 1024
                ) as handle:
                    for (payload,) in connection.execute(
                        "SELECT payload_json FROM fine_frames "
                        "WHERE view_id = ? ORDER BY local_ms, frame_index",
                        (view.view_id,),
                    ):
                        handle.write(payload + "\n")
                temporary.replace(output)
                outputs[view.view_id] = output
        finally:
            connection.close()
        return outputs

    def iter_global_frames(
        self,
        view_id: str,
        *,
        start_ms: float,
        end_ms: float,
    ) -> Iterable[FrameEvidence]:
        connection = sqlite3.connect(self.path)
        try:
            for (payload,) in connection.execute(
                "SELECT payload_json FROM fine_frames WHERE view_id = ? "
                "AND global_ms >= ? AND global_ms <= ? "
                "ORDER BY global_ms, frame_index",
                (view_id, float(start_ms), float(end_ms)),
            ):
                yield FrameEvidence.model_validate_json(payload)
        finally:
            connection.close()

    def replace_audit_candidates(
        self, candidates: Sequence[ActionCandidate]
    ) -> int:
        """Refresh the derived time index used by candidate_audit."""

        connection = sqlite3.connect(self.path)
        try:
            connection.execute("DELETE FROM fine_audit_candidates")
            connection.executemany(
                "INSERT INTO fine_audit_candidates VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (
                        candidate.candidate_id,
                        candidate.action_type.value,
                        candidate.view_id,
                        float(candidate.global_start_ms),
                        float(candidate.global_end_ms),
                        candidate.model_dump_json(),
                    )
                    for candidate in candidates
                ],
            )
            connection.commit()
        finally:
            connection.close()
        return len(candidates)

    def iter_audit_candidates(
        self,
        *,
        start_ms: float,
        end_ms: float,
        action_type: str | None = None,
    ) -> Iterable[ActionCandidate]:
        clauses = ["global_end_ms >= ?", "global_start_ms <= ?"]
        parameters: list[Any] = [float(start_ms), float(end_ms)]
        if action_type is not None:
            clauses.append("action_type = ?")
            parameters.append(str(action_type))
        query = (
            "SELECT payload_json FROM fine_audit_candidates WHERE "
            + " AND ".join(clauses)
            + " ORDER BY global_start_ms, global_end_ms, candidate_id"
        )
        connection = sqlite3.connect(self.path)
        try:
            for (payload,) in connection.execute(query, parameters):
                yield ActionCandidate.model_validate_json(payload)
        finally:
            connection.close()


def create_fine_frame_index(index_path: Path) -> FineFrameIndex:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    if index_path.exists():
        index_path.unlink()
    connection = sqlite3.connect(index_path)
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.executescript(
            """
            CREATE TABLE fine_frames (
                view_id TEXT NOT NULL,
                frame_index INTEGER NOT NULL,
                local_ms REAL NOT NULL,
                global_ms REAL NOT NULL,
                motion_score REAL NOT NULL,
                has_actor INTEGER NOT NULL,
                object_count INTEGER NOT NULL,
                payload_json TEXT NOT NULL,
                source_pass TEXT NOT NULL,
                PRIMARY KEY (view_id, local_ms)
            ) WITHOUT ROWID;
            CREATE INDEX fine_frames_global
                ON fine_frames(view_id, global_ms);
            CREATE INDEX fine_frames_activity
                ON fine_frames(view_id, has_actor, object_count, global_ms);
            CREATE TABLE fine_tracks (
                view_id TEXT NOT NULL,
                unified_track_id INTEGER NOT NULL,
                class_name TEXT NOT NULL,
                first_local_ms REAL NOT NULL,
                last_local_ms REAL NOT NULL,
                last_center_x REAL NOT NULL,
                last_center_y REAL NOT NULL,
                source_refs_json TEXT NOT NULL,
                PRIMARY KEY (view_id, unified_track_id)
            ) WITHOUT ROWID;
            CREATE INDEX fine_tracks_recent
                ON fine_tracks(view_id, class_name, last_local_ms);
            CREATE TABLE fine_track_stitches (
                source_pass TEXT NOT NULL,
                view_id TEXT NOT NULL,
                source_track_id INTEGER NOT NULL,
                unified_track_id INTEGER NOT NULL,
                match_method TEXT NOT NULL,
                match_gap_ms REAL,
                center_distance REAL,
                PRIMARY KEY (source_pass, view_id, source_track_id)
            ) WITHOUT ROWID;
            CREATE TABLE fine_ingest_passes (
                source_pass TEXT NOT NULL,
                view_id TEXT NOT NULL,
                input_path TEXT NOT NULL,
                input_frames INTEGER NOT NULL,
                indexed_frames INTEGER NOT NULL,
                replaced_frames INTEGER NOT NULL,
                stitched_tracks INTEGER NOT NULL,
                new_tracks INTEGER NOT NULL,
                PRIMARY KEY (source_pass, view_id)
            ) WITHOUT ROWID;
            CREATE TABLE fine_audit_candidates (
                candidate_id TEXT NOT NULL PRIMARY KEY,
                action_type TEXT NOT NULL,
                view_id TEXT NOT NULL,
                global_start_ms REAL NOT NULL,
                global_end_ms REAL NOT NULL,
                payload_json TEXT NOT NULL
            ) WITHOUT ROWID;
            CREATE INDEX fine_audit_candidates_time
                ON fine_audit_candidates(global_start_ms, global_end_ms);
            CREATE INDEX fine_audit_candidates_action_time
                ON fine_audit_candidates(action_type, global_start_ms, global_end_ms);
            """
        )
        connection.commit()
    finally:
        connection.close()
    return FineFrameIndex(index_path)


def _box_center_from_norm(box: Sequence[float]) -> tuple[float, float]:
    return (float(box[0] + box[2]) / 2.0, float(box[1] + box[3]) / 2.0)


def _box_iou(left: Sequence[float], right: Sequence[float]) -> float:
    x1 = max(float(left[0]), float(right[0]))
    y1 = max(float(left[1]), float(right[1]))
    x2 = min(float(left[2]), float(right[2]))
    y2 = min(float(left[3]), float(right[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, float(left[2]) - float(left[0])) * max(
        0.0, float(left[3]) - float(left[1])
    )
    right_area = max(0.0, float(right[2]) - float(right[0])) * max(
        0.0, float(right[3]) - float(right[1])
    )
    return intersection / max(left_area + right_area - intersection, 1e-9)


def _next_track_ids(connection: sqlite3.Connection) -> dict[str, int]:
    return {
        str(view_id): int(maximum or 0) + 1
        for view_id, maximum in connection.execute(
            "SELECT view_id, MAX(unified_track_id) FROM fine_tracks GROUP BY view_id"
        )
    }


def _existing_frame_payload_at(
    connection: sqlite3.Connection,
    view_id: str,
    local_ms: float,
) -> str | None:
    row = connection.execute(
        "SELECT payload_json FROM fine_frames WHERE view_id = ? AND local_ms = ?",
        (view_id, float(local_ms)),
    ).fetchone()
    return str(row[0]) if row else None


def _select_unified_track(
    connection: sqlite3.Connection,
    *,
    view_id: str,
    class_name: str,
    local_ms: float,
    box: Sequence[float],
    overlapping: FrameEvidence | None,
    maximum_gap_ms: float,
    maximum_center_distance: float,
    used_tracks: set[int],
) -> tuple[int | None, str, float | None, float | None]:
    if overlapping is not None:
        matches = [
            (
                _box_iou(box, item.xyxy_norm),
                int(item.track_id),
            )
            for item in overlapping.detections
            if item.track_id is not None
            and item.class_name == class_name
            and int(item.track_id) not in used_tracks
        ]
        if matches:
            overlap, track_id = max(matches)
            if overlap >= 0.20:
                return track_id, "overlap_iou", 0.0, 1.0 - overlap

    center_x, center_y = _box_center_from_norm(box)
    candidates = []
    for track_id, last_ms, last_x, last_y in connection.execute(
        "SELECT unified_track_id, last_local_ms, last_center_x, last_center_y "
        "FROM fine_tracks WHERE view_id = ? AND class_name = ? "
        "AND last_local_ms <= ? AND last_local_ms >= ?",
        (
            view_id,
            class_name,
            float(local_ms),
            float(local_ms - maximum_gap_ms),
        ),
    ):
        track_id = int(track_id)
        if track_id in used_tracks:
            continue
        distance = math.hypot(center_x - float(last_x), center_y - float(last_y))
        if distance <= maximum_center_distance:
            candidates.append(
                (distance, float(local_ms - last_ms), track_id)
            )
    if not candidates:
        return None, "new_track", None, None
    distance, gap_ms, track_id = min(candidates)
    return track_id, "recent_endpoint", gap_ms, distance


def ingest_fine_frame_ledgers(
    index: FineFrameIndex,
    views: Sequence[ViewInput],
    detection_paths: dict[str, Path],
    *,
    source_pass: str,
    stitching_enabled: bool = True,
    maximum_stitch_gap_ms: float = 2500.0,
    maximum_center_distance: float = 0.12,
    timings: Any | None = None,
) -> dict[str, Any]:
    """Append one fine-scan pass and stitch track identities when provable."""

    connection = sqlite3.connect(index.path)
    next_ids = _next_track_ids(connection)
    reports: list[dict[str, Any]] = []
    try:
        for view in views:
            path = detection_paths[view.view_id]
            source_map: dict[int, int] = {}
            # A source pass already has distinct tracker identities. Reserve
            # each mapped identity for the entire pass, including occlusions
            # and detections that occur later in the current frame. Otherwise
            # a nearby new object can steal an existing mapping and both
            # objects subsequently share one trajectory.
            assigned_tracks: set[int] = set()
            # Assigned tracks cannot participate in another match this pass.
            # Keep their endpoints in memory and publish each track once;
            # unassigned tracks stay queryable in SQLite for stitching.
            track_states: dict[int, list[Any]] = {}
            used_by_frame: set[int] = set()
            input_frames = 0
            replaced_frames = 0
            stitched_tracks: set[int] = set()
            new_tracks: set[int] = set()
            frames = iter(iter_frame_evidence(path))
            while True:
                try:
                    if timings is None:
                        frame = next(frames)
                    else:
                        with timings.measure("ledger_read_parse_seconds", cpu=True):
                            frame = next(frames)
                except StopIteration:
                    break
                input_frames += 1
                overlapping_payload = _existing_frame_payload_at(
                    connection, view.view_id, frame.local_ms
                )
                overlapping = None
                if overlapping_payload is not None:
                    replaced_frames += 1
                used_by_frame.clear()
                normalized_detections = []
                for detection in frame.detections:
                    source_track_id = detection.track_id
                    if source_track_id is None:
                        normalized_detections.append(detection)
                        continue
                    unified_track_id = source_map.get(int(source_track_id))
                    method = "same_pass"
                    gap_ms: float | None = None
                    center_distance: float | None = None
                    if unified_track_id is None:
                        if stitching_enabled:
                            if overlapping is None and overlapping_payload is not None:
                                overlapping = FrameEvidence.model_validate_json(overlapping_payload)
                            (
                                unified_track_id,
                                method,
                                gap_ms,
                                center_distance,
                            ) = _select_unified_track(
                                connection,
                                view_id=view.view_id,
                                class_name=detection.class_name,
                                local_ms=frame.local_ms,
                                box=detection.xyxy_norm,
                                overlapping=overlapping,
                                maximum_gap_ms=maximum_stitch_gap_ms,
                                maximum_center_distance=maximum_center_distance,
                                used_tracks=used_by_frame | assigned_tracks,
                            )
                        if unified_track_id is None:
                            unified_track_id = next_ids.get(view.view_id, 1)
                            next_ids[view.view_id] = unified_track_id + 1
                            method = "new_track"
                            new_tracks.add(unified_track_id)
                        else:
                            stitched_tracks.add(unified_track_id)
                        source_map[int(source_track_id)] = unified_track_id
                        assigned_tracks.add(unified_track_id)
                        connection.execute(
                            "INSERT OR REPLACE INTO fine_track_stitches VALUES (?, ?, ?, ?, ?, ?, ?)",
                            (
                                source_pass,
                                view.view_id,
                                int(source_track_id),
                                unified_track_id,
                                method,
                                gap_ms,
                                center_distance,
                            ),
                        )
                    used_by_frame.add(unified_track_id)
                    normalized_detections.append(
                        detection.model_copy(
                            update={"track_id": unified_track_id}
                        )
                    )
                    center_x, center_y = _box_center_from_norm(
                        detection.xyxy_norm
                    )
                    state = track_states.get(unified_track_id)
                    if state is None:
                        existing_track = connection.execute(
                            "SELECT first_local_ms, last_local_ms, last_center_x, "
                            "last_center_y, source_refs_json FROM fine_tracks "
                            "WHERE view_id = ? AND unified_track_id = ?",
                            (view.view_id, unified_track_id),
                        ).fetchone()
                        source_ref = f"{source_pass}:{int(source_track_id)}"
                        if existing_track:
                            refs = set(json.loads(existing_track[4]))
                            refs.add(source_ref)
                            state = [detection.class_name, *map(float, existing_track[:4]), json.dumps(sorted(refs))]
                        else:
                            state = [detection.class_name, float(frame.local_ms), float(frame.local_ms),
                                     center_x, center_y, json.dumps([source_ref])]
                        track_states[unified_track_id] = state
                    state[1] = min(state[1], float(frame.local_ms))
                    if float(frame.local_ms) >= state[2]:
                        state[2:5] = [float(frame.local_ms), center_x, center_y]
                normalized = frame.model_copy(
                    update={"detections": normalized_detections}
                )
                classes = {item.class_name for item in normalized_detections}
                objects = classes - _ACTOR_CLASSES - _NON_ACTION_CLASSES
                global_ms = float(
                    frame.global_ms
                    if frame.global_ms is not None
                    else frame.local_ms
                )
                connection.execute(
                    "INSERT OR REPLACE INTO fine_frames VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        view.view_id,
                        int(frame.frame_index),
                        float(frame.local_ms),
                        global_ms,
                        float(frame.motion_score),
                        int(bool(classes & _ACTOR_CLASSES)),
                        len(objects),
                        normalized.model_dump_json(),
                        source_pass,
                    ),
                )
            connection.executemany(
                "INSERT INTO fine_tracks VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(view_id, unified_track_id) DO UPDATE SET "
                "first_local_ms = excluded.first_local_ms, last_local_ms = excluded.last_local_ms, "
                "last_center_x = excluded.last_center_x, last_center_y = excluded.last_center_y, "
                "source_refs_json = excluded.source_refs_json",
                ((view.view_id, track_id, *state) for track_id, state in track_states.items()),
            )
            indexed_frames = int(
                connection.execute(
                    "SELECT COUNT(*) FROM fine_frames WHERE view_id = ?",
                    (view.view_id,),
                ).fetchone()[0]
            )
            connection.execute(
                "INSERT OR REPLACE INTO fine_ingest_passes VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    source_pass,
                    view.view_id,
                    str(path),
                    input_frames,
                    indexed_frames,
                    replaced_frames,
                    len(stitched_tracks),
                    len(new_tracks),
                ),
            )
            reports.append(
                {
                    "view_id": view.view_id,
                    "input_path": str(path),
                    "input_frames": input_frames,
                    "indexed_frames": indexed_frames,
                    "replaced_frames": replaced_frames,
                    "stitched_tracks": len(stitched_tracks),
                    "new_tracks": len(new_tracks),
                }
            )
        connection.commit()
    finally:
        connection.close()
    return {
        "source_pass": source_pass,
        "track_identity_policy": "injective_source_track_mapping_per_view_and_pass",
        "views": reports,
        "input_frames": sum(item["input_frames"] for item in reports),
        "stitched_tracks": sum(item["stitched_tracks"] for item in reports),
        "new_tracks": sum(item["new_tracks"] for item in reports),
    }


def _expected_window_samples(
    info: VideoInfo,
    start_ms: float,
    end_ms: float,
    sample_fps: float,
) -> int:
    if end_ms <= start_ms:
        return 0
    spans = (
        [
            (
                max(start_ms, float(segment.virtual_start_ms)),
                min(end_ms, float(segment.virtual_end_ms)),
            )
            for segment in info.segments
        ]
        if info.segments
        else [(max(0.0, start_ms), min(float(info.duration_ms), end_ms))]
    )
    return sum(
        max(1, int(math.ceil((right - left) / 1000.0 * sample_fps - 1e-9)))
        for left, right in spans
        if right > left
    )


def fine_frame_coverage_report(
    index: FineFrameIndex,
    views: Sequence[ViewInput],
    infos: dict[str, VideoInfo],
    windows: dict[str, list[tuple[float, float]]],
    *,
    sample_fps: float,
    minimum_coverage_ratio: float = 0.98,
    maximum_gap_periods: float = 4.0,
    alignment_scales: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Compare intended fine windows with actually indexed frame timestamps."""

    connection = sqlite3.connect(index.path)
    view_reports: list[dict[str, Any]] = []
    try:
        for view in views:
            info = infos[view.view_id]
            effective_sample_fps = sample_fps * max(
                1e-9,
                float((alignment_scales or {}).get(view.view_id, 1.0)),
            )
            period_ms = 1000.0 / max(effective_sample_fps, 1e-9)
            allowed_gap_ms = period_ms * maximum_gap_periods
            expected = 0
            actual_keys: set[int] = set()
            gaps: list[dict[str, float]] = []
            for window_index, (start_ms, end_ms) in enumerate(
                windows.get(view.view_id, [])
            ):
                expected += _expected_window_samples(
                    info, start_ms, end_ms, effective_sample_fps
                )
                times = [
                    float(item[0])
                    for item in connection.execute(
                        "SELECT local_ms FROM fine_frames WHERE view_id = ? "
                        "AND local_ms >= ? AND local_ms <= ? ORDER BY local_ms",
                        (view.view_id, float(start_ms), float(end_ms)),
                    )
                ]
                actual_keys.update(round(item * 1000.0) for item in times)
                if not times:
                    gaps.append(
                        {
                            "window_index": float(window_index),
                            "start_local_ms": float(start_ms),
                            "end_local_ms": float(end_ms),
                            "gap_ms": float(end_ms - start_ms),
                        }
                    )
                    continue
                boundary_pairs = [
                    (float(start_ms), times[0]),
                    *zip(times, times[1:], strict=False),
                    (times[-1], float(end_ms)),
                ]
                for left, right in boundary_pairs:
                    gap_ms = float(right - left)
                    if gap_ms <= allowed_gap_ms:
                        continue
                    if not _same_physical_span(info, left, right):
                        continue
                    if len(gaps) < 100:
                        gaps.append(
                            {
                                "window_index": float(window_index),
                                "start_local_ms": left,
                                "end_local_ms": right,
                                "gap_ms": gap_ms,
                            }
                        )
            actual = len(actual_keys)
            coverage_ratio = (
                1.0 if expected == 0 else min(1.0, actual / max(1, expected))
            )
            formal_ready = bool(
                (expected == 0 or actual > 0)
                and coverage_ratio >= minimum_coverage_ratio
                and not gaps
            )
            view_reports.append(
                {
                    "view_id": view.view_id,
                    "window_count": len(windows.get(view.view_id, [])),
                    "effective_local_sample_fps": round(
                        effective_sample_fps, 9
                    ),
                    "expected_sample_count": expected,
                    "actual_sample_count": actual,
                    "coverage_ratio": round(coverage_ratio, 6),
                    "maximum_allowed_gap_ms": round(allowed_gap_ms, 3),
                    "unexpected_gap_count": len(gaps),
                    "unexpected_gaps": gaps,
                    "formal_evidence_ready": formal_ready,
                }
            )
    finally:
        connection.close()
    return {
        "schema_version": "visioncortex-fine-frame-index/1",
        "index_path": str(index.path),
        "sample_fps": sample_fps,
        "minimum_coverage_ratio": minimum_coverage_ratio,
        "maximum_gap_periods": maximum_gap_periods,
        "row_count": sum(item["actual_sample_count"] for item in view_reports),
        "formal_evidence_ready": all(
            item["formal_evidence_ready"] for item in view_reports
        ),
        "views": view_reports,
    }


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
