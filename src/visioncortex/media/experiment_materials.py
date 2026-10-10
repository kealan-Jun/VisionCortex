"""Aligned experiment clips, CSV projection and bounded review frames."""

from __future__ import annotations

import csv
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np

from ..evidence.layout import ArchiveLayout
from ..schemas import (
    AlignmentTransform,
    EvidenceEvent,
    ExperimentGroup,
    ExperimentSegment,
    VideoInfo,
    ViewInput,
)


@dataclass(frozen=True)
class ExperimentMaterialsServices:
    """Named execution ports; supplied explicitly at the composition boundary."""

    _group_folder_name: Callable[..., Any]
    _materialize_derived_media: Callable[..., Any]
    _relative: Callable[..., Any]
    _safe_slug: Callable[..., Any]
    create_grid_video: Callable[..., Any]
    extract_view_clip: Callable[..., Any]
    iter_aligned_rows: Callable[..., Any]
    view_source_files: Callable[..., Any]
    write_json: Callable[..., Any]


def write_aligned_csv(
    path: Path,
    views: Sequence[ViewInput],
    infos: dict[str, VideoInfo],
    transforms: dict[str, AlignmentTransform],
    output_fps: float,
    *,
    services: ExperimentMaterialsServices,
) -> None:
    rows = services.iter_aligned_rows(views, infos, transforms, output_fps)
    first = next(iter(rows), None)
    if first is None:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(first))
        writer.writeheader()
        writer.writerow(first)
        writer.writerows(rows)


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
    *,
    services: ExperimentMaterialsServices,
) -> None:
    from ..workflow_video import covered_intervals, materialize_third_person_timeline

    by_view = {view.view_id: view for view in views}
    by_segment = {segment.segment_id: segment for segment in segments}
    encoder = config["performance"]["ffmpeg_video_encoder"]
    workers = max(
        1, min(2, int(config["performance"].get("materialization_workers", 2)))
    )
    runtime_records: list[dict[str, Any]] = []
    overlap_aligned = bool(
        publisher is None
        and config.get("performance", {}).get("overlap_aligned_experiment_clips", False)
    )
    aligned_executor = (
        ThreadPoolExecutor(
            max_workers=max(
                1,
                int(
                    config.get("performance", {}).get(
                        "aligned_experiment_clip_workers", 1
                    )
                ),
            ),
            thread_name_prefix="experiment-aligned",
        )
        if overlap_aligned
        else None
    )
    aligned_jobs: list[Any] = []

    def materialize_aligned(
        group: ExperimentGroup,
        aligned: Path,
        aligned_json: Path,
        aligned_inputs: list[tuple[str, Path]],
    ) -> dict[str, Any]:
        aligned_started = time.perf_counter()
        aligned_cache = services._materialize_derived_media(
            aligned,
            "experiment-aligned-clip",
            {
                "group_id": group.group_id,
                "global_start_ms": group.global_start_ms,
                "global_end_ms": group.global_end_ms,
                "layout": "first_person_left,third_person_right",
                "grid_shape": "2x1@640x360_each",
                "view_timeline": group.view_timeline,
            },
            [path for _, path in aligned_inputs],
            config,
            lambda: services.create_grid_video(aligned_inputs, aligned, encoder),
            content_address_inputs=True,
        )
        services.write_json(
            aligned_json,
            {
                "schema_version": "2.0.0",
                "artifact_type": "aligned_first_third_experiment_video",
                "group": group.model_dump(mode="json", exclude={"video_json"}),
                "video_file": services._relative(aligned, layout.root),
                "layout": "first_person_left, third_person_right",
                "global_start_ms": group.global_start_ms,
                "global_end_ms": group.global_end_ms,
                "views": group.participating_views,
                "view_timeline": group.view_timeline,
                "alignments": {
                    view_id: transforms[view_id].model_dump(mode="json")
                    for view_id in group.participating_views
                },
                "atomic_experiments": [
                    by_segment[item].model_dump(mode="json")
                    for item in group.atomic_experiment_ids
                ],
            },
        )
        if publisher is not None:
            publisher.publish_file(aligned)
            publisher.publish_file(aligned_json)
        return {
            "group_id": group.group_id,
            "role_label": "Aligned-First-Third",
            "view_id": "aligned_first_third",
            "duration_seconds": round(time.perf_counter() - aligned_started, 6),
            "output_bytes": aligned.stat().st_size,
            "overlapped_with_next_group": overlap_aligned,
            **aligned_cache,
        }

    for index, group in enumerate(groups, 1):
        folder_name = services._group_folder_name(
            layout,
            index,
            group.experiment_name,
            group.experiment_name_en,
        )
        group.archive_folder = folder_name
        group_root = layout.experiment_clips / folder_name
        videos_dir = group_root
        json_dir = group_root
        group_root.mkdir(parents=True, exist_ok=True)
        extraction_jobs = []

        def extract_role(role_label: str, view_id: str) -> dict[str, Any]:
            started = time.perf_counter()
            source_transform = transforms[view_id]
            needs_coverage_route = (
                source_transform.to_local(group.global_start_ms) < 0
                or source_transform.to_local(group.global_end_ms)
                > infos[view_id].duration_ms + 1
            )
            if (
                role_label == "Third-Person" and group.view_timeline
            ) or needs_coverage_route:
                destination = videos_dir / f"{role_label}.mp4"
                route_group = (
                    group
                    if role_label == "Third-Person" and group.view_timeline
                    else group.model_copy(
                        update={
                            "view_timeline": [
                                {
                                    "start_ms": group.global_start_ms,
                                    "end_ms": group.global_end_ms,
                                    "third_person_view": view_id,
                                }
                            ]
                        }
                    )
                )
                rows = covered_intervals(route_group, infos, transforms)
                source_ids = sorted(
                    {
                        row["third_person_view"]
                        for row in rows
                        if row.get("third_person_view")
                    }
                )
                cache = services._materialize_derived_media(
                    destination,
                    "workflow-third-person-timeline",
                    {
                        "coverage_policy": "explicit_source_gaps_v1",
                        "timeline": rows,
                        "alignments": {
                            v: transforms[v].model_dump(mode="json") for v in source_ids
                        },
                    },
                    [
                        p
                        for v in source_ids
                        for p in services.view_source_files(by_view[v])
                    ],
                    config,
                    lambda: materialize_third_person_timeline(
                        route_group, by_view, infos, transforms, destination, encoder
                    ),
                )
                return {
                    "group_id": group.group_id,
                    "role_label": role_label,
                    "view_id": "routed",
                    "duration_seconds": round(time.perf_counter() - started, 6),
                    "source_coverage": rows,
                    "output_bytes": destination.stat().st_size,
                    **cache,
                }
            view = by_view[view_id]
            transform = transforms[view_id]
            local_start = max(0.0, transform.to_local(group.global_start_ms))
            local_end = min(
                infos[view_id].duration_ms,
                transform.to_local(group.global_end_ms),
            )
            if local_end <= local_start:
                raise ValueError(
                    f"{group.group_id}/{view_id} global boundary is outside the source video"
                )
            destination = videos_dir / f"{role_label}.mp4"
            cache = services._materialize_derived_media(
                destination,
                "experiment-view-clip",
                {
                    "group_id": group.group_id,
                    "role_label": role_label,
                    "view_id": view_id,
                    "local_start_ms": local_start,
                    "local_end_ms": local_end,
                    "global_start_ms": group.global_start_ms,
                    "global_end_ms": group.global_end_ms,
                },
                services.view_source_files(view),
                config,
                lambda: services.extract_view_clip(
                    view,
                    infos[view_id],
                    destination,
                    local_start,
                    local_end - local_start,
                    encoder,
                ),
            )
            return {
                "group_id": group.group_id,
                "role_label": role_label,
                "view_id": view_id,
                "duration_seconds": round(time.perf_counter() - started, 6),
                "output_bytes": destination.stat().st_size,
                **cache,
            }

        with ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="experiment-media",
        ) as executor:
            for role_label, view_id in (
                ("First-Person", group.first_person_view),
                ("Third-Person", group.third_person_view),
            ):
                extraction_jobs.append(
                    executor.submit(extract_role, role_label, view_id)
                )
            failures = []
            for role_label, job in zip(
                ("First-Person", "Third-Person"), extraction_jobs, strict=True
            ):
                try:
                    runtime_records.append(job.result())
                except (OSError, ValueError, RuntimeError) as exc:
                    failures.append((role_label, exc))
            if failures:
                runtime_path = (
                    layout.json_config / "experiment_clip_materialization_runtime.json"
                )
                services.write_json(
                    runtime_path,
                    {
                        "schema_version": "visioncortex-materialization-runtime/1",
                        "status": "partial",
                        "records": runtime_records,
                        "failures": [
                            {
                                "group_id": group.group_id,
                                "role_label": role_label,
                                "requested_start_ms": group.global_start_ms,
                                "requested_end_ms": group.global_end_ms,
                                "error_type": type(exc).__name__,
                                "message": str(exc),
                            }
                            for role_label, exc in failures
                        ],
                    },
                )
                if publisher is not None:
                    publisher.publish_file(runtime_path)
                raise failures[0][1]
        role_paths: dict[str, tuple[str, Path]] = {}
        for role_label, view_id in (
            ("First-Person", group.first_person_view),
            ("Third-Person", group.third_person_view),
        ):
            view = by_view[view_id]
            transform = transforms[view_id]
            local_start = max(0.0, transform.to_local(group.global_start_ms))
            local_end = min(
                infos[view_id].duration_ms, transform.to_local(group.global_end_ms)
            )
            if local_end <= local_start:
                raise ValueError(
                    f"{group.group_id}/{view_id} 全局边界映射后不在视频范围内"
                )
            base = role_label
            destination = videos_dir / f"{base}.mp4"
            # Both role clips were exported concurrently above. Metadata and
            # publication remain ordered and atomic for a stable archive.
            relative = services._relative(destination, layout.root)
            group.videos[role_label.lower()] = relative
            role_paths[role_label] = (view_id, destination)
            metadata = {
                "schema_version": "2.0.0",
                "artifact_type": "experiment_view_video",
                "group": group.model_dump(mode="json", exclude={"video_json"}),
                "view_id": view_id,
                "view_role": view.role.value,
                "video_file": relative,
                "global_start_ms": group.global_start_ms,
                "global_end_ms": group.global_end_ms,
                "local_start_ms": local_start,
                "local_end_ms": local_end,
                "alignment": transform.model_dump(mode="json"),
                "source_coverage": next(
                    (
                        r.get("source_coverage")
                        for r in runtime_records
                        if r.get("group_id") == group.group_id
                        and r.get("role_label") == role_label
                    ),
                    None,
                ),
                "atomic_experiments": [
                    by_segment[item].model_dump(mode="json")
                    for item in group.atomic_experiment_ids
                ],
            }
            if role_label == "Third-Person" and group.view_timeline:
                metadata.update(
                    view_id=None,
                    alignment=None,
                    local_start_ms=None,
                    local_end_ms=None,
                    view_timeline=covered_intervals(group, infos, transforms),
                    alignments={
                        v: transforms[v].model_dump(mode="json")
                        for v in group.participating_views
                    },
                )
            json_path = json_dir / f"{base}.json"
            services.write_json(json_path, metadata)
            if publisher is not None:
                publisher.publish_file(destination)
                publisher.publish_file(json_path)
            group.video_json[role_label.lower()] = services._relative(
                json_path, layout.root
            )

        aligned = videos_dir / "Aligned_First+Third.mp4"
        aligned_inputs = [
            (
                f"First-Person {role_paths['First-Person'][0]}",
                role_paths["First-Person"][1],
            ),
            (
                f"Third-Person {role_paths['Third-Person'][0]}",
                role_paths["Third-Person"][1],
            ),
        ]
        group.videos["aligned_first_third"] = services._relative(aligned, layout.root)
        aligned_json = json_dir / "Aligned_First+Third.json"
        group.video_json["aligned_first_third"] = services._relative(
            aligned_json, layout.root
        )
        for segment_id in group.atomic_experiment_ids:
            segment = by_segment[segment_id]
            segment.clips = {group.first_person_view: group.videos["first-person"]}
            if not group.view_timeline:
                segment.clips[group.third_person_view] = group.videos["third-person"]
            segment.aligned_multiview_clip = group.videos["aligned_first_third"]
        aligned_arguments = (
            group,
            aligned,
            aligned_json,
            list(aligned_inputs),
        )
        if aligned_executor is None:
            runtime_records.append(materialize_aligned(*aligned_arguments))
        else:
            aligned_jobs.append(
                aligned_executor.submit(materialize_aligned, *aligned_arguments)
            )

    if aligned_executor is not None:
        try:
            runtime_records.extend(job.result() for job in aligned_jobs)
        finally:
            aligned_executor.shutdown(wait=True, cancel_futures=True)
        group_order = {group.group_id: index for index, group in enumerate(groups)}
        role_order = {
            "First-Person": 0,
            "Third-Person": 1,
            "Aligned-First-Third": 2,
        }
        runtime_records.sort(
            key=lambda item: (
                group_order.get(str(item.get("group_id")), len(group_order)),
                role_order.get(str(item.get("role_label")), len(role_order)),
            )
        )

    runtime_path = layout.json_config / "experiment_clip_materialization_runtime.json"
    services.write_json(
        runtime_path,
        {
            "schema_version": "visioncortex-materialization-runtime/1",
            "workers": workers,
            "aligned_workers": (
                int(
                    config.get("performance", {}).get(
                        "aligned_experiment_clip_workers", 1
                    )
                )
                if overlap_aligned
                else 0
            ),
            "overlap_aligned": overlap_aligned,
            "records": runtime_records,
        },
    )
    if publisher is not None:
        publisher.publish_file(runtime_path)


def _storyboard_times(
    group: ExperimentGroup,
    events: Sequence[EvidenceEvent],
    limit: int,
    segments: Sequence[ExperimentSegment] = (),
) -> list[float]:
    """Cover boundaries, atomic segments and action classes before uniform fill."""

    if limit < 2:
        return [(group.global_start_ms + group.global_end_ms) / 2.0]
    duration = max(1.0, group.global_end_ms - group.global_start_ms)
    uniform = [
        group.global_start_ms + duration * index / (limit - 1) for index in range(limit)
    ]
    accepted = [
        event
        for event in events
        if event.accepted
        and group.global_start_ms <= event.key_global_ms <= group.global_end_ms
    ]
    selected = [uniform[0], uniform[-1]]

    def add(value: float) -> None:
        if len(selected) >= limit:
            return
        bounded = min(group.global_end_ms, max(group.global_start_ms, float(value)))
        if bounded not in selected:
            selected.append(bounded)

    for segment in sorted(segments, key=lambda item: item.global_start_ms):
        segment_events = [
            event for event in accepted if event.event_id in set(segment.event_ids)
        ]
        representative = (
            max(
                segment_events,
                key=lambda event: (event.confidence, -event.key_global_ms),
            ).key_global_ms
            if segment_events
            else (segment.global_start_ms + segment.global_end_ms) / 2.0
        )
        add(representative)
    for action_type in sorted({event.action_type.value for event in accepted}):
        add(
            max(
                (event for event in accepted if event.action_type.value == action_type),
                key=lambda event: (event.confidence, -event.key_global_ms),
            ).key_global_ms
        )

    candidates = sorted(set(uniform + [event.key_global_ms for event in accepted]))
    while len(selected) < limit and candidates:
        best = max(
            candidates, key=lambda value: min(abs(value - item) for item in selected)
        )
        add(best)
        candidates.remove(best)
    return sorted(set(selected))[:limit]


def _write_aligned_frame(
    left: Path, right: Path, destination: Path, labels: tuple[str, str]
) -> None:
    images = [cv2.imread(str(left)), cv2.imread(str(right))]
    if any(image is None for image in images):
        raise RuntimeError(f"无法读取对齐关键帧: {left}, {right}")
    assert images[0] is not None and images[1] is not None
    target_height = min(images[0].shape[0], images[1].shape[0])
    rendered = []
    for image, label in zip(images, labels, strict=True):
        scale = target_height / image.shape[0]
        resized = cv2.resize(
            image, (max(1, round(image.shape[1] * scale)), target_height)
        )
        cv2.rectangle(resized, (0, 0), (resized.shape[1], 36), (0, 0, 0), -1)
        cv2.putText(
            resized, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2
        )
        rendered.append(resized)
    combined = np.hstack(rendered)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), combined):
        raise RuntimeError(f"写入对齐关键帧失败: {destination}")


def extract_temporal_review_frames(
    clip_path: Path,
    output_dir: Path,
    view_id: str,
    samples_per_view: int = 3,
    *,
    services: ExperimentMaterialsServices,
) -> list[tuple[str, Path]]:
    """Sample a bounded timeline from an already-made key clip.

    This avoids another seek through the original NAS video and gives the MLLM
    temporal evidence instead of asking it to infer an action from one still.
    """

    # High-risk pipette review needs several frames inside each source,
    # transport and target dwell, not just one uniform snapshot per phase.
    # Fifteen remains bounded and is read from the already-materialized clip.
    count = max(1, min(15, int(samples_per_view)))
    capture = cv2.VideoCapture(str(clip_path))
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count <= 0:
            return []
        if count == 1:
            fractions = [0.5]
            phases = ["clip_middle"]
        elif count == 2:
            fractions = [0.2, 0.8]
            phases = ["clip_early", "clip_late"]
        elif count == 3:
            fractions = [0.15, 0.5, 0.85]
            phases = ["clip_early", "clip_middle", "clip_late"]
        elif count in {4, 5}:
            fractions = [0.1 + 0.8 * index / (count - 1) for index in range(count)]
            # These are positions in a clip, not observed action phases.
            # Its midpoint need not coincide with the selected key frame.
            if count == 4:
                phases = ["clip_early", "clip_mid_early", "clip_mid_late", "clip_late"]
            else:
                phases = [
                    "clip_early",
                    "clip_mid_early",
                    "clip_middle",
                    "clip_mid_late",
                    "clip_late",
                ]
        else:
            fractions = [0.1 + 0.8 * index / (count - 1) for index in range(count)]
            phases = [f"timeline_{index:02d}" for index in range(1, count + 1)]
        samples: list[tuple[str, Path]] = []
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_view = services._safe_slug(view_id)
        for phase, fraction in zip(phases, fractions, strict=True):
            frame_index = min(
                frame_count - 1, max(0, round((frame_count - 1) * fraction))
            )
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            path = output_dir / f"{safe_view}_{phase}.jpg"
            if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 88]):
                continue
            nominal_ms = (
                round(frame_index * 1000 / fps, 3)
                if math.isfinite(fps) and fps > 0
                else "unknown"
            )
            samples.append(
                (
                    f"view_id={view_id}; sample_scope=clip_timeline; "
                    f"temporal_phase={phase}; key_clip_frame={frame_index}; "
                    f"nominal_clip_time_ms={nominal_ms}",
                    path,
                )
            )
        return samples
    finally:
        capture.release()
