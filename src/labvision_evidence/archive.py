from __future__ import annotations

import csv
import hashlib
import json
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
from openpyxl import Workbook

from .alignment import iter_aligned_rows
from .action_semantics import record_semantic_review
from .detection import nearest_frame_evidence_many
from .indexing import (
    build_archive_index,
    stable_artifact_uid,
    stable_event_uid,
    stable_evidence_uid,
)
from .mllm import ArkStepAnalyzer, EVENT_SYSTEM_PROMPT, GROUP_SYSTEM_PROMPT
from .material_naming import (
    ACTION_CATEGORY_FOLDERS,
    ACTION_SLUGS,
    key_material_action_folder,
    key_material_semantic_name as _key_material_semantic_name,
)
from .schemas import (
    AlignmentTransform,
    EvidenceEvent,
    ExperimentGroup,
    ExperimentSegment,
    PhysicalChange,
    RunManifest,
    RunSummary,
    VideoInfo,
    ViewInput,
)
from .video_io import (
    ViewFrameReader,
    create_grid_video,
    extract_view_clip,
    write_annotated_frame,
)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "value"):
        return value.value
    raise TypeError(type(value).__name__)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8"
    )
    temporary.replace(path)


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

    digest = hashlib.sha256()
    header = {
        "schema": "visioncortex-semantic-cache/1",
        "kind": kind,
        "model": str(config["mllm"]["model"]),
        "base_url": str(config["mllm"].get("base_url") or ""),
        "response_language": str(config["mllm"].get("response_language") or ""),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "evidence": evidence,
        "image_order": [label for label, _ in images],
    }
    digest.update(
        json.dumps(header, ensure_ascii=False, sort_keys=True, default=_json_default).encode(
            "utf-8"
        )
    )
    for label, image_path in images:
        digest.update(b"\0label\0")
        digest.update(label.encode("utf-8"))
        digest.update(b"\0image\0")
        digest.update(image_path.read_bytes())
    return digest.hexdigest()


def _semantic_cache_path(
    config: dict[str, Any], kind: str, fingerprint: str
) -> Path:
    return (
        Path(str(config["storage"]["local_cache_root"]))
        / "semantic-results-v1"
        / _safe_slug(str(config["mllm"]["model"]))
        / kind
        / f"{fingerprint}.json"
    )


def _read_semantic_cache(path: Path, fingerprint: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        cached = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        cached.get("status") != "completed"
        or cached.get("input_fingerprint") != fingerprint
        or cached.get("semantic_cache_schema") != "visioncortex-semantic-cache/1"
    ):
        return None
    cached["cache_reused"] = True
    return cached


def _write_semantic_cache(
    path: Path, fingerprint: str, result: dict[str, Any]
) -> dict[str, Any]:
    persisted = dict(result)
    persisted["input_fingerprint"] = fingerprint
    persisted["semantic_cache_schema"] = "visioncortex-semantic-cache/1"
    persisted["cache_reused"] = False
    write_json(path, persisted)
    return persisted


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _time_slug(ms: float) -> str:
    total = max(0.0, ms) / 1000.0
    hours = int(total // 3600)
    minutes = int(total % 3600 // 60)
    seconds = total % 60
    return f"{hours:02d}h{minutes:02d}m{seconds:06.3f}s"


def _safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-_") or "unknown"


def _safe_folder_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(" .-_")
    if cleaned:
        return cleaned[:120]
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]
    return f"Unnamed-Experiment-{digest}"


def _bounded_component(value: str, maximum_chars: int) -> str:
    cleaned = _safe_folder_name(value)
    maximum_chars = max(12, int(maximum_chars))
    if len(cleaned) <= maximum_chars:
        return cleaned
    digest = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()[:8]
    prefix = cleaned[: max(1, maximum_chars - len(digest) - 1)].rstrip(" .-_")
    return f"{prefix}-{digest}"


def _component_budget(parent: Path, reserved_tail_chars: int, maximum_chars: int = 72) -> int:
    # 235 leaves headroom below classic Windows MAX_PATH for FFmpeg/OpenCV
    # temporary suffixes when LongPathsEnabled is disabled.
    return max(
        12,
        min(maximum_chars, 235 - len(str(parent)) - int(reserved_tail_chars) - 1),
    )


def _group_folder_name(
    layout: "ArchiveLayout",
    index: int,
    experiment_name: str,
    experiment_name_en: str,
) -> str:
    del experiment_name  # Chinese display name stays in JSON/report content only.
    desired = f"{index:03d}_{_safe_slug(experiment_name_en)}"
    budget = _group_folder_budget(layout)
    return _bounded_component(desired, budget)


def _group_folder_budget(layout: "ArchiveLayout") -> int:
    return min(
        _component_budget(layout.experiment_clips, 28),
        # Action category + event folder + separators + longest aligned sidecar.
        _component_budget(layout.key_frames, 104),
        _component_budget(layout.key_clips, 104),
    )


def _key_material_event_folder_name(
    layout: "ArchiveLayout",
    experiment_folder: str,
    event: EvidenceEvent,
) -> str:
    action_folder = key_material_action_folder(event.action_type)
    desired = _key_material_semantic_name(event)["file_stem"]
    budget = min(
        _component_budget(
            layout.key_frames / experiment_folder / action_folder,
            26,
            maximum_chars=40,
        ),
        _component_budget(
            layout.key_clips / experiment_folder / action_folder,
            26,
            maximum_chars=40,
        ),
    )
    return _bounded_component(desired, budget)


class ArchiveLayout:
    def __init__(self, root: Path):
        self.root = root
        self.experiment_clips = root / "Experiment-Clips"
        self.json_config = root / "JSON-Config-Files"
        self.key_materials = root / "Key-Materials"
        self.key_clips = self.key_materials / "Key-Clips"
        self.key_frames = self.key_materials / "Key-Frames"
        self.daily_reports = root / "Lab-Daily-Reports"
        self.original_videos = root / "Original-Experiment-Videos"
        self.professional_pdfs = root / "Professional-PDFs"
        self.work = root / ".work"

    def create(self) -> None:
        for path in (
            self.experiment_clips,
            self.json_config,
            self.key_clips,
            self.key_frames,
            self.daily_reports,
            self.original_videos,
            self.professional_pdfs,
            self.work,
        ):
            path.mkdir(parents=True, exist_ok=True)


def write_aligned_csv(
    path: Path,
    views: Sequence[ViewInput],
    infos: dict[str, VideoInfo],
    transforms: dict[str, AlignmentTransform],
    output_fps: float,
) -> None:
    rows = iter_aligned_rows(views, infos, transforms, output_fps)
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
) -> None:
    by_view = {view.view_id: view for view in views}
    by_segment = {segment.segment_id: segment for segment in segments}
    encoder = config["performance"]["ffmpeg_video_encoder"]
    workers = max(1, min(2, int(config["performance"].get("materialization_workers", 2))))
    runtime_records: list[dict[str, Any]] = []
    for index, group in enumerate(groups, 1):
        folder_name = _group_folder_name(
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
            extract_view_clip(
                view,
                infos[view_id],
                destination,
                local_start,
                local_end - local_start,
                encoder,
            )
            return {
                "group_id": group.group_id,
                "role_label": role_label,
                "view_id": view_id,
                "duration_seconds": round(time.perf_counter() - started, 6),
                "output_bytes": destination.stat().st_size,
            }

        with ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="experiment-media",
        ) as executor:
            for role_label, view_id in (
                ("First-Person", group.first_person_view),
                ("Third-Person", group.third_person_view),
            ):
                extraction_jobs.append(executor.submit(extract_role, role_label, view_id))
            runtime_records.extend(job.result() for job in extraction_jobs)
        role_paths: dict[str, tuple[str, Path]] = {}
        for role_label, view_id in (
            ("First-Person", group.first_person_view),
            ("Third-Person", group.third_person_view),
        ):
            view = by_view[view_id]
            transform = transforms[view_id]
            local_start = max(0.0, transform.to_local(group.global_start_ms))
            local_end = min(infos[view_id].duration_ms, transform.to_local(group.global_end_ms))
            if local_end <= local_start:
                raise ValueError(f"{group.group_id}/{view_id} 全局边界映射后不在视频范围内")
            base = role_label
            destination = videos_dir / f"{base}.mp4"
            # Both role clips were exported concurrently above. Metadata and
            # publication remain ordered and atomic for a stable archive.
            relative = _relative(destination, layout.root)
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
                "atomic_experiments": [
                    by_segment[item].model_dump(mode="json") for item in group.atomic_experiment_ids
                ],
            }
            json_path = json_dir / f"{base}.json"
            write_json(json_path, metadata)
            if publisher is not None:
                publisher.publish_file(destination)
                publisher.publish_file(json_path)
            group.video_json[role_label.lower()] = _relative(json_path, layout.root)

        aligned = videos_dir / "Aligned_First+Third.mp4"
        create_grid_video(
            [
                (f"First-Person {role_paths['First-Person'][0]}", role_paths["First-Person"][1]),
                (f"Third-Person {role_paths['Third-Person'][0]}", role_paths["Third-Person"][1]),
            ],
            aligned,
            encoder,
        )
        group.videos["aligned_first_third"] = _relative(aligned, layout.root)
        aligned_json = json_dir / "Aligned_First+Third.json"
        write_json(
            aligned_json,
            {
                "schema_version": "2.0.0",
                "artifact_type": "aligned_first_third_experiment_video",
                "group": group.model_dump(mode="json", exclude={"video_json"}),
                "video_file": _relative(aligned, layout.root),
                "layout": "first_person_left, third_person_right",
                "global_start_ms": group.global_start_ms,
                "global_end_ms": group.global_end_ms,
                "views": [group.first_person_view, group.third_person_view],
                "alignments": {
                    view_id: transforms[view_id].model_dump(mode="json")
                    for view_id in (group.first_person_view, group.third_person_view)
                },
                "atomic_experiments": [
                    by_segment[item].model_dump(mode="json") for item in group.atomic_experiment_ids
                ],
            },
        )
        group.video_json["aligned_first_third"] = _relative(aligned_json, layout.root)
        if publisher is not None:
            publisher.publish_file(aligned)
            publisher.publish_file(aligned_json)
        for segment_id in group.atomic_experiment_ids:
            segment = by_segment[segment_id]
            segment.clips = {
                group.first_person_view: group.videos["first-person"],
                group.third_person_view: group.videos["third-person"],
            }
            segment.aligned_multiview_clip = group.videos["aligned_first_third"]

    runtime_path = layout.json_config / "experiment_clip_materialization_runtime.json"
    write_json(
        runtime_path,
        {
            "schema_version": "visioncortex-materialization-runtime/1",
            "workers": workers,
            "records": runtime_records,
        },
    )
    if publisher is not None:
        publisher.publish_file(runtime_path)


def _storyboard_times(group: ExperimentGroup, events: Sequence[EvidenceEvent], limit: int) -> list[float]:
    if limit < 2:
        return [(group.global_start_ms + group.global_end_ms) / 2.0]
    duration = max(1.0, group.global_end_ms - group.global_start_ms)
    uniform = [
        group.global_start_ms + duration * index / (limit - 1)
        for index in range(limit)
    ]
    event_times = [
        event.key_global_ms
        for event in events
        if event.accepted and group.global_start_ms <= event.key_global_ms <= group.global_end_ms
    ]
    candidates = sorted(set(uniform + event_times))
    selected = [uniform[0], uniform[-1]]
    while len(selected) < limit and candidates:
        best = max(candidates, key=lambda value: min(abs(value - item) for item in selected))
        selected.append(best)
        candidates.remove(best)
    return sorted(set(selected))[:limit]


def analyze_experiment_groups(
    layout: ArchiveLayout,
    groups: Sequence[ExperimentGroup],
    segments: Sequence[ExperimentSegment],
    events: Sequence[EvidenceEvent],
    views: Sequence[ViewInput],
    infos: dict[str, VideoInfo],
    transforms: dict[str, AlignmentTransform],
    config: dict[str, Any],
) -> None:
    analyzer = ArkStepAnalyzer(config)
    cache_root = layout.work / "mllm-cache" / "experiment-groups"
    by_view = {view.view_id: view for view in views}
    by_segment = {segment.segment_id: segment for segment in segments}
    max_pairs = max(2, int(config["mllm"].get("storyboard_pairs_per_group", 6)))

    def analyze(group: ExperimentGroup) -> tuple[ExperimentGroup, dict[str, Any]]:
        storyboard: list[tuple[str, Path]] = []
        storyboard_dir = layout.work / "group-storyboards" / group.group_id
        with ViewFrameReader(max_open=2) as frame_reader:
            for index, global_ms in enumerate(_storyboard_times(group, events, max_pairs), 1):
                for role_label, view_id in (
                    ("first_person", group.first_person_view),
                    ("third_person", group.third_person_view),
                ):
                    local_ms = transforms[view_id].to_local(global_ms)
                    if not 0.0 <= local_ms <= infos[view_id].duration_ms:
                        continue
                    frame = frame_reader.read(by_view[view_id], infos[view_id], local_ms)
                    if frame is None:
                        continue
                    path = storyboard_dir / f"{index:02d}_{role_label}_{_safe_slug(view_id)}.jpg"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 88]):
                        continue
                    storyboard.append(
                        (
                            f"t={global_ms:.3f}ms; role={role_label}; view_id={view_id}",
                            path,
                        )
                    )
        atomic = [by_segment[item] for item in group.atomic_experiment_ids]
        event_ids = {event_id for segment in atomic for event_id in segment.event_ids}
        group_events = [event for event in events if event.event_id in event_ids]
        semantic_evidence = {
            "continuity_type": group.continuity_type,
            "continuity_reason": group.continuity_reason,
            "global_start_ms": group.global_start_ms,
            "global_end_ms": group.global_end_ms,
            "first_person_view": group.first_person_view,
            "third_person_view": group.third_person_view,
            "atomic_boundaries": [
                segment.model_dump(
                    mode="json",
                    exclude={"semantic_understanding", "clips", "aligned_multiview_clip"},
                )
                for segment in atomic
            ],
            "cv_events": [
                event.model_dump(
                    mode="json",
                    exclude={"model_understanding", "key_frames", "key_clips"},
                )
                for event in group_events
                if event.accepted
            ],
        }
        fingerprint = _semantic_fingerprint(
            "experiment-group",
            config,
            GROUP_SYSTEM_PROMPT,
            semantic_evidence,
            storyboard,
        )
        persistent_cache_path = _semantic_cache_path(
            config, "experiment-groups", fingerprint
        )
        run_cache_path = cache_root / f"{fingerprint}.json"
        cached = _read_semantic_cache(persistent_cache_path, fingerprint)
        if cached is None:
            cached = _read_semantic_cache(run_cache_path, fingerprint)
        if cached is not None:
            return group, cached
        result = analyzer.analyze_group(group, atomic, group_events, storyboard)
        if result.get("status") == "completed":
            result = _write_semantic_cache(persistent_cache_path, fingerprint, result)
            write_json(run_cache_path, result)
        return group, result

    workers = max(1, int(config["mllm"].get("group_workers", 2)))
    try:
        with ThreadPoolExecutor(max_workers=min(workers, max(1, len(groups)))) as executor:
            futures = [executor.submit(analyze, group) for group in groups]
            for future in as_completed(futures):
                group, result = future.result()
                group.model_understanding = result
                if result.get("status") == "completed":
                    group.experiment_name = str(
                        result.get("experiment_name") or group.experiment_name
                    )
                    group.experiment_name_en = _safe_slug(
                        str(result.get("experiment_name_en") or group.experiment_name_en)
                    )
                    confirmed = result.get("continuity_type_confirmed")
                    # Rule-based continuous chains may be split only by an explicit
                    # model conflict; rule-based independent groups never merge here.
                    if confirmed == "continuous" and group.continuity_type == "continuous":
                        group.continuity_reason += "；模型确认连续"
                    elif confirmed not in {group.continuity_type, "uncertain", None}:
                        result.setdefault("uncertainties", []).append(
                            "模型连续性判断与物理规则冲突，归档采用物理规则并保留冲突"
                        )
                index = int(group.group_id.rsplit("-", 1)[-1])
                group.archive_folder = _group_folder_name(
                    layout,
                    index,
                    group.experiment_name,
                    group.experiment_name_en,
                )
                for segment_id in group.atomic_experiment_ids:
                    segment = by_segment[segment_id]
                    segment.experiment_name = group.experiment_name
                    segment.experiment_name_en = group.experiment_name_en
                    segment.semantic_understanding = result
    finally:
        analyzer.close()


def _write_aligned_frame(left: Path, right: Path, destination: Path, labels: tuple[str, str]) -> None:
    images = [cv2.imread(str(left)), cv2.imread(str(right))]
    if any(image is None for image in images):
        raise RuntimeError(f"无法读取对齐关键帧: {left}, {right}")
    assert images[0] is not None and images[1] is not None
    target_height = min(images[0].shape[0], images[1].shape[0])
    rendered = []
    for image, label in zip(images, labels, strict=True):
        scale = target_height / image.shape[0]
        resized = cv2.resize(image, (max(1, round(image.shape[1] * scale)), target_height))
        cv2.rectangle(resized, (0, 0), (resized.shape[1], 36), (0, 0, 0), -1)
        cv2.putText(resized, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
        rendered.append(resized)
    combined = np.hstack(rendered)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), combined):
        raise RuntimeError(f"写入对齐关键帧失败: {destination}")


def _artifact_json(
    group: ExperimentGroup,
    event: EvidenceEvent,
    artifact_type: str,
    artifact_file: str,
    view_id: str | None,
    transforms: dict[str, AlignmentTransform],
    archive_id: str | None = None,
) -> dict[str, Any]:
    views = [view_id] if view_id else [group.first_person_view, group.third_person_view]
    understanding = event.model_understanding or {}
    physical_change = understanding.get("physical_change") or {}
    before_state = str(physical_change.get("before") or "unknown")
    after_state = str(physical_change.get("after") or "unknown")
    model_objects = [str(item) for item in understanding.get("objects") or []]
    object_names = list(dict.fromkeys([*event.objects, *model_objects]))
    material_name = _key_material_semantic_name(event)

    def contains(name: str, keywords: tuple[str, ...]) -> bool:
        lowered = name.lower()
        return any(keyword in lowered for keyword in keywords)

    def first_object(keywords: tuple[str, ...], excluded: set[str] | None = None) -> str | None:
        excluded = excluded or set()
        return next(
            (
                name
                for name in object_names
                if name not in excluded and contains(name, keywords)
            ),
            None,
        )

    tool = first_object(
        (
            "pipette",
            "spearhead",
            "spatula",
            "dropper",
            "移液",
            "吸头",
            "药勺",
            "滴管",
        )
    )
    source = first_object(
        ("reagent_bottle", "bottle", "试剂瓶", "试剂容器", "瓶"),
        {tool} if tool else set(),
    )
    target = first_object(
        ("tube", "paper", "balance", "beaker", "flask", "离心管", "称量纸", "天平", "烧杯", "锥形瓶"),
        {item for item in (tool, source) if item},
    )

    def tracked_id(name: str | None, fallback: str) -> str:
        if not name:
            return "unknown"
        slug = _safe_slug(name).lower()
        if slug == "unknown":
            known = {
                "移液器": "pipette",
                "移液枪": "pipette",
                "枪头": "pipette-tip",
                "吸头": "pipette-tip",
                "药勺": "spatula",
                "称量纸": "weighing-paper",
                "试剂瓶": "reagent-bottle",
                "离心管": "centrifuge-tube",
                "分析天平": "analytical-balance",
            }
            slug = next((value for key, value in known.items() if key in name), fallback)
        track_id = None
        for candidate in event.candidates:
            if name not in candidate.objects:
                continue
            for evidence in candidate.evidence:
                for key in ("object_track_id", "track_id", "tool_track_id", "container_track_id"):
                    if evidence.get(key) is not None:
                        track_id = evidence[key]
                        break
                if track_id is not None:
                    break
            if track_id is not None:
                break
        return f"{slug}-{int(track_id):03d}" if track_id is not None else f"{slug}-unresolved"

    normalized_action = {
        "liquid_movement": "liquid_transfer",
    }.get(event.action_type.value, event.action_type.value)
    combined = " ".join(object_names).lower()
    if normalized_action == "liquid_transfer":
        action_subtype = "pipette_transfer" if any(
            keyword in combined for keyword in ("pipette", "spearhead", "移液", "吸头")
        ) else "container_pour"
        phases = [
            "source_approach",
            "source_contact",
            "source_withdraw",
            "transport",
            "target_contact",
            "target_release",
        ]
    elif normalized_action == "container_state_change":
        action_subtype = "container_open_close"
        phases = ["container_approach", "container_contact", "state_transition", "release"]
    elif normalized_action == "device_panel_operation":
        action_subtype = "device_control_operation"
        phases = ["panel_approach", "panel_contact", "control_activation", "withdraw"]
    elif normalized_action == "object_movement":
        action_subtype = "tool_movement" if tool else "object_relocation"
        phases = ["object_approach", "grasp", "lift", "transport", "place", "release"]
    else:
        action_subtype = "tool_contact" if tool else "generic_hand_object_contact"
        phases = ["object_approach", "contact", "manipulation", "release"]

    observations = []
    for index, item in enumerate(understanding.get("per_view_observations") or [], 1):
        observation_view = str(item.get("view_id") or "unknown")
        observations.append(
            {
                "observation_id": f"{event.event_id}-obs-{index:02d}",
                "view_id": observation_view,
                "view_role": (
                    "first_person"
                    if observation_view == group.first_person_view
                    else "third_person"
                    if observation_view == group.third_person_view
                    else "unknown"
                ),
                "timestamp_us": round(event.key_global_ms * 1000.0),
                "observed_fact": str(item.get("observation") or ""),
            }
        )
    if not observations:
        observations = [
            {
                "observation_id": f"{event.event_id}-obs-{index:02d}",
                "view_id": supported_view,
                "view_role": (
                    "first_person" if supported_view == group.first_person_view else "third_person"
                ),
                "timestamp_us": round(event.key_global_ms * 1000.0),
                "observed_fact": f"CV accepted {event.action_type.value}: {', '.join(event.objects)}",
            }
            for index, supported_view in enumerate(event.supporting_views, 1)
        ]

    alignment_uncertainty_us = max(
        80_000,
        round(
            max(
                (
                    transforms[item].csv_rmse_ms
                    for item in (group.first_person_view, group.third_person_view)
                    if item in transforms and transforms[item].csv_rmse_ms is not None
                ),
                default=80.0,
            )
            * 1000.0
        ),
    )
    cross_view_associations = [
        {
            "association_id": f"{event.event_id}-first-third",
            "source_view_id": group.first_person_view,
            "target_view_id": group.third_person_view,
            "global_timestamp_us": round(event.key_global_ms * 1000.0),
            "source_local_timestamp_us": round(
                transforms[group.first_person_view].to_local(event.key_global_ms) * 1000.0
            ),
            "target_local_timestamp_us": round(
                transforms[group.third_person_view].to_local(event.key_global_ms) * 1000.0
            ),
            "time_uncertainty_us": alignment_uncertainty_us,
            "consistency": understanding.get("cross_view_consistency", "unreviewed"),
            "both_views_support_action": all(
                item in event.supporting_views
                for item in (group.first_person_view, group.third_person_view)
            ),
        }
    ]

    current_step = str(understanding.get("current_step") or "")
    next_step = str(understanding.get("next_step") or "")
    uncertainties = list(
        dict.fromkeys(
            [
                *[str(item) for item in event.uncertainty],
                *[
                    f"CV未直接观察: {item}"
                    for item in (
                        event.observability.get("unmet_visual_requirements")
                        if event.observability
                        else []
                    )
                ],
                *[str(item) for item in understanding.get("uncertainties") or []],
            ]
        )
    )
    consistency = str(understanding.get("cross_view_consistency") or "unreviewed")
    model_confidence = float(understanding.get("confidence") or 0.0)
    confirmed_action = str(understanding.get("action_type_confirmed") or "unknown")
    status = "confirmed"
    if understanding.get("status") != "completed":
        status = "provisional_cv_only"
    elif confirmed_action == "unknown" or consistency == "conflict" or model_confidence < 0.55:
        status = "uncertain"

    role_for = (
        "aligned_first_third"
        if view_id is None
        else "first_person"
        if view_id == group.first_person_view
        else "third_person"
    )
    event_uid = (
        stable_event_uid(archive_id, group.group_id, event.event_id)
        if archive_id
        else None
    )
    key_frames = [
        {
            "view_id": item,
            "view_role": (
                "aligned_first_third"
                if item == "aligned_first_third"
                else "first_person"
                if item == group.first_person_view
                else "third_person"
            ),
            "timestamp_us": round(event.key_global_ms * 1000.0),
            "path": path,
            **(
                {
                    "artifact_uid": stable_artifact_uid(event_uid, "key_frame", item),
                    "sidecar_path": str(Path(path).with_suffix(".json")).replace("\\", "/"),
                }
                if event_uid
                else {}
            ),
        }
        for item, path in event.key_frames.items()
    ]
    key_clips = [
        {
            "view_id": item,
            "view_role": (
                "aligned_first_third"
                if item == "aligned_first_third"
                else "first_person"
                if item == group.first_person_view
                else "third_person"
            ),
            "start_us": round(max(0.0, event.global_start_ms - 2000.0) * 1000.0),
            "end_us": round((event.global_end_ms + 3000.0) * 1000.0),
            "path": path,
            **(
                {
                    "artifact_uid": stable_artifact_uid(event_uid, "key_clip", item),
                    "sidecar_path": str(Path(path).with_suffix(".json")).replace("\\", "/"),
                }
                if event_uid
                else {}
            ),
        }
        for item, path in event.key_clips.items()
    ]
    candidate_ids = [candidate.candidate_id for candidate in event.candidates]
    frame_evidence_ids = [
        f"{candidate.view_id}:frame-{item['frame_index']}"
        for candidate in event.candidates
        for item in candidate.evidence
        if item.get("frame_index") is not None
    ]
    evidence_ids = list(dict.fromkeys([*candidate_ids, *frame_evidence_ids]))
    observed_facts = [item for item in [current_step] if item]
    observed_facts.extend(
        item["observed_fact"] for item in observations if item.get("observed_fact")
    )
    supported_inferences = []
    if next_step and next_step not in {"未知", "unknown", "不确定"}:
        supported_inferences.append(f"下一步：{next_step}")
    contradictions = []
    if consistency == "conflict":
        contradictions.append("第一人称与第三人称观察发生冲突，详见 observations")

    cross_view_score = {
        "consistent": 0.95,
        "partial": 0.65,
        "conflict": 0.20,
        "single_view": 0.40,
    }.get(consistency, 0.35)
    change_observed = before_state != "unknown" and after_state != "unknown"
    action_agrees = confirmed_action in {event.action_type.value, normalized_action}
    return {
        "event_id": event.event_id,
        "parent_event_id": group.group_id,
        "actor_id": "operator-01",
        "workstation_id": (
            "weighing-station-01"
            if any(keyword in combined for keyword in ("balance", "天平"))
            else "pipetting-station-01"
            if any(keyword in combined for keyword in ("pipette", "spearhead", "tube", "移液", "吸头", "离心管"))
            else "wet-lab-bench-01"
        ),
        "action_type": normalized_action,
        "action_subtype": action_subtype,
        "start_us": round(event.global_start_ms * 1000.0),
        "end_us": round(event.global_end_ms * 1000.0),
        "peak_timestamp_us": round(event.key_global_ms * 1000.0),
        "time_uncertainty_us": alignment_uncertainty_us,
        "phases": phases,
        "objects": {
            "tool": tracked_id(tool, "tool"),
            "source": tracked_id(source, "source"),
            "target": tracked_id(target, "target"),
        },
        "state_before": {
            "tool": before_state,
            "source": "unknown",
            "target": "unknown",
        },
        "state_after": {
            "tool": after_state,
            "source": "unknown",
            "target": "unknown",
        },
        "observations": observations,
        "cross_view_associations": cross_view_associations,
        "decision": {
            "status": status,
            "observed_facts": observed_facts,
            "supported_inferences": supported_inferences,
            "uncertain_claims": uncertainties,
            "contradictions": contradictions,
        },
        "scores": {
            "temporal_continuity": round(min(1.0, 0.5 + float(event.confidence) * 0.5), 4),
            "object_identity": round(min(1.0, 0.35 + min(len(event.objects), 4) * 0.08 + model_confidence * 0.3), 4),
            "phase_completeness": round((0.45 + model_confidence * 0.5) if change_observed else (0.25 + model_confidence * 0.35), 4),
            "cross_view_support": cross_view_score,
            "state_change_support": round(model_confidence if change_observed else 0.25, 4),
            "model_agreement": round((float(event.confidence) + model_confidence) / 2.0 if action_agrees else model_confidence * 0.5, 4),
            "contradiction_penalty": 0.8 if consistency == "conflict" else 0.0,
        },
        "key_frames": key_frames,
        "key_clips": key_clips,
        "evidence_ids": evidence_ids,
        "provenance": {
            "schema_version": "key-material-event-v1.0.0",
            "time_base": "aligned_global_timeline_microseconds",
            "experiment_group_id": group.group_id,
            "experiment_name": group.experiment_name,
            "continuity_type": group.continuity_type,
            "atomic_experiment_ids": group.atomic_experiment_ids,
            "archive_classification": {
                "hierarchy_version": "2.0.0",
                "experiment_folder": (
                    group.archive_folder or _safe_folder_name(group.group_id)
                ),
                "action_type": event.action_type.value,
                "action_category_folder": key_material_action_folder(event.action_type),
                "semantic_file_stem": material_name["file_stem"],
                "primary_object": material_name["primary_object"],
                "object_labels": material_name["object_labels"],
            },
            "sidecar_for": {
                "artifact_type": artifact_type,
                "artifact_file": artifact_file,
                "view_id": view_id,
                "view_role": role_for,
            },
            "cv": {
                "action_type": event.action_type.value,
                "objects": event.objects,
                "confidence": event.confidence,
                "audit_reason": event.audit_reason,
                "supporting_views": event.supporting_views,
                "observability": event.observability,
                "semantic_review": event.semantic_review,
            },
            "mllm": {
                "model": understanding.get("model"),
                "status": understanding.get("status"),
                "usage": understanding.get("usage") or {},
                "latency_seconds": understanding.get("latency_seconds"),
                "attempts": understanding.get("attempts"),
                "current_step": current_step or None,
                "next_step": next_step or None,
            },
            "alignment": {
                item: transforms[item].model_dump(mode="json")
                for item in (group.first_person_view, group.third_person_view)
            },
            "score_method": "deterministic_cv_mllm_evidence_mapping_v1",
            **(
                {
                    "index": {
                        "event_uid": event_uid,
                        "database": "JSON-Config-Files/evidence_index.sqlite",
                        "artifact_registry": "JSON-Config-Files/artifact_registry.jsonl",
                        "evidence_registry": "JSON-Config-Files/evidence_registry.jsonl",
                        "decision_receipt_registry": "JSON-Config-Files/decision_receipt_registry.jsonl",
                        "evidence_refs": [
                            {
                                "evidence_id": evidence_id,
                                "evidence_uid": stable_evidence_uid(
                                    event_uid, evidence_id
                                ),
                            }
                            for evidence_id in evidence_ids
                        ],
                    }
                }
                if event_uid
                else {}
            ),
        },
    }


def prepare_key_material_category_layout(
    layout: ArchiveLayout,
    groups: Sequence[ExperimentGroup],
) -> None:
    """Create the stable experiment/action hierarchy before materialization."""

    for group in groups:
        experiment_folder = group.archive_folder or _safe_folder_name(group.group_id)
        for action_folder in ACTION_CATEGORY_FOLDERS.values():
            (layout.key_frames / experiment_folder / action_folder).mkdir(
                parents=True, exist_ok=True
            )
            (layout.key_clips / experiment_folder / action_folder).mkdir(
                parents=True, exist_ok=True
            )


def write_key_material_category_index(
    layout: ArchiveLayout,
    groups: Sequence[ExperimentGroup],
    events: Sequence[EvidenceEvent],
    publisher: Any | None = None,
) -> Path:
    """Write a human-browsable and machine-indexable five-category manifest."""

    event_by_id = {event.event_id: event for event in events if event.accepted}
    experiments: list[dict[str, Any]] = []
    for group in groups:
        experiment_folder = group.archive_folder or _safe_folder_name(group.group_id)
        group_events = [
            event_by_id[event_id]
            for event_id in group.key_event_ids
            if event_id in event_by_id
        ]
        categories: list[dict[str, Any]] = []
        for action_type, action_folder in ACTION_CATEGORY_FOLDERS.items():
            category_events = [
                event
                for event in group_events
                if event.action_type.value == action_type
            ]
            frame_category_folder = (
                layout.key_frames / experiment_folder / action_folder
            )
            clip_category_folder = (
                layout.key_clips / experiment_folder / action_folder
            )
            category_summary = {
                "schema_version": "visioncortex-key-material-category/1.0.0",
                "group_id": group.group_id,
                "experiment_name": group.experiment_name,
                "experiment_name_en": group.experiment_name_en,
                "experiment_folder": experiment_folder,
                "action_type": action_type,
                "action_category_folder": action_folder,
                "event_count": len(category_events),
                "coverage_status": (
                    "observed" if category_events else "not_observed"
                ),
                "absence_reason": (
                    None
                    if category_events
                    else (
                        "No accepted evidence event of this action type was "
                        "observed in the bounded experiment."
                    )
                ),
                "event_ids": [event.event_id for event in category_events],
                "events": [
                    {
                        "event_id": event.event_id,
                        "event_folder": _key_material_event_folder_name(
                            layout, experiment_folder, event
                        ),
                        "material_name": _key_material_semantic_name(event),
                    }
                    for event in category_events
                ],
            }
            frame_summary_path = frame_category_folder / "Category.json"
            clip_summary_path = clip_category_folder / "Category.json"
            write_json(
                frame_summary_path,
                {**category_summary, "media_kind": "key_frame"},
            )
            write_json(
                clip_summary_path,
                {**category_summary, "media_kind": "key_clip"},
            )
            if publisher is not None:
                publisher.publish_file(frame_summary_path)
                publisher.publish_file(clip_summary_path)
            categories.append(
                {
                    "action_type": action_type,
                    "folder": action_folder,
                    "event_count": len(category_events),
                    "key_frames_folder": (
                        Path("Key-Materials")
                        / "Key-Frames"
                        / experiment_folder
                        / action_folder
                    ).as_posix(),
                    "key_clips_folder": (
                        Path("Key-Materials")
                        / "Key-Clips"
                        / experiment_folder
                        / action_folder
                    ).as_posix(),
                    "key_frames_category_summary": _relative(
                        frame_summary_path, layout.root
                    ),
                    "key_clips_category_summary": _relative(
                        clip_summary_path, layout.root
                    ),
                    "events": [
                        {
                            "event_id": event.event_id,
                            "event_folder": _key_material_event_folder_name(
                                layout, experiment_folder, event
                            ),
                            "material_name": _key_material_semantic_name(event),
                            "peak_timestamp_us": round(
                                event.key_global_ms * 1000.0
                            ),
                            "key_frames": dict(event.key_frames),
                            "key_clips": dict(event.key_clips),
                        }
                        for event in category_events
                    ],
                }
            )
        experiments.append(
            {
                "group_id": group.group_id,
                "experiment_name": group.experiment_name,
                "experiment_name_en": group.experiment_name_en,
                "experiment_folder": experiment_folder,
                "key_event_count": len(group_events),
                "action_categories": categories,
            }
        )
    path = layout.key_materials / "Key-Material-Category-Index.json"
    write_json(
        path,
        {
            "schema_version": "visioncortex-key-material-category-index/1.0.0",
            "hierarchy": (
                "Key-Materials/{Key-Frames|Key-Clips}/"
                "{Experiment}/{Action-Category}/{Event}"
            ),
            "category_count": len(ACTION_CATEGORY_FOLDERS),
            "categories": [
                {
                    "order": index,
                    "action_type": action_type,
                    "folder": action_folder,
                }
                for index, (action_type, action_folder) in enumerate(
                    ACTION_CATEGORY_FOLDERS.items(), 1
                )
            ],
            "experiments": experiments,
        },
    )
    if publisher is not None:
        publisher.publish_file(path)
    return path


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
) -> None:
    by_view = {view.view_id: view for view in views}
    before = float(config["segmentation"]["key_clip_pre_seconds"]) * 1000.0
    after = float(config["segmentation"]["key_clip_post_seconds"]) * 1000.0
    encoder = config["performance"]["ffmpeg_video_encoder"]
    workers = max(1, min(2, int(config["performance"].get("materialization_workers", 2))))
    group_by_event = {event_id: group for group in groups for event_id in group.key_event_ids}
    prepare_key_material_category_layout(layout, groups)
    runtime_records: list[dict[str, Any]] = []
    stage_started = time.perf_counter()
    accepted_timestamps = [
        event.key_global_ms
        for event in events
        if event.accepted and event.event_id in group_by_event
    ]
    lookup_started = time.perf_counter()
    nearest_by_view = {
        view_id: nearest_frame_evidence_many(path, accepted_timestamps)
        for view_id, path in detection_paths.items()
    }
    lookup_seconds = time.perf_counter() - lookup_started
    for event in events:
        if not event.accepted or event.event_id not in group_by_event:
            continue
        event_started = time.perf_counter()
        group = group_by_event[event.event_id]
        folder = group.archive_folder or _safe_folder_name(group.group_id)
        action_folder = key_material_action_folder(event.action_type)
        event_folder = _key_material_event_folder_name(layout, folder, event)
        frame_dir = layout.key_frames / folder / action_folder / event_folder
        clip_dir = layout.key_clips / folder / action_folder / event_folder
        frame_dir.mkdir(parents=True, exist_ok=True)
        clip_dir.mkdir(parents=True, exist_ok=True)
        frame_paths: dict[str, Path] = {}
        clip_paths: dict[str, Path] = {}

        def extract_role(role_label: str, view_id: str) -> dict[str, Any]:
            role_started = time.perf_counter()
            view = by_view[view_id]
            transform = transforms[view_id]
            local_key_ms = transform.to_local(event.key_global_ms)
            if not 0.0 <= local_key_ms <= infos[view_id].duration_ms:
                raise ValueError(
                    f"{event.event_id}/{view_id} key timestamp is outside the source video"
                )
            frame_started = time.perf_counter()
            frame = None
            used_offset_ms = 0.0
            frame_reader = ViewFrameReader(max_open=1)
            try:
                for offset_ms in (0.0, -100.0, 100.0, -250.0, 250.0):
                    candidate_ms = local_key_ms + offset_ms
                    if not 0.0 <= candidate_ms <= infos[view_id].duration_ms:
                        continue
                    frame = frame_reader.read(view, infos[view_id], candidate_ms)
                    if frame is not None:
                        used_offset_ms = offset_ms
                        break
            finally:
                frame_reader.close()
            if frame is None:
                raise RuntimeError(f"{event.event_id}/{view_id} key frame decode failed")
            frame_seconds = time.perf_counter() - frame_started
            nearest = nearest_by_view[view_id].get(float(event.key_global_ms))
            boxes = [box.model_dump() for box in nearest.detections] if nearest else []
            base = role_label
            frame_path = frame_dir / f"{base}.jpg"
            write_annotated_frame(frame, boxes, frame_path)
            clip_start_global = max(event.global_start_ms - before, 0.0)
            clip_end_global = event.global_end_ms + after
            local_start = max(0.0, transform.to_local(clip_start_global))
            local_end = min(infos[view_id].duration_ms, transform.to_local(clip_end_global))
            if local_end <= local_start:
                raise ValueError(
                    f"{event.event_id}/{view_id} key clip boundary is outside the source video"
                )
            clip_path = clip_dir / f"{base}.mp4"
            clip_started = time.perf_counter()
            extract_view_clip(
                view, infos[view_id], clip_path, local_start, local_end - local_start, encoder
            )
            clip_seconds = time.perf_counter() - clip_started
            return {
                "event_id": event.event_id,
                "experiment_group_id": group.group_id,
                "action_type": event.action_type.value,
                "action_category_folder": action_folder,
                "role_label": role_label,
                "view_id": view_id,
                "frame_path": frame_path,
                "clip_path": clip_path,
                "frame_decode_offset_ms": used_offset_ms,
                "frame_duration_seconds": round(frame_seconds, 6),
                "clip_duration_seconds": round(clip_seconds, 6),
                "duration_seconds": round(time.perf_counter() - role_started, 6),
                "clip_source_duration_seconds": round((local_end - local_start) / 1000.0, 6),
                "frame_output_bytes": frame_path.stat().st_size,
                "clip_output_bytes": clip_path.stat().st_size,
            }

        role_results: dict[str, dict[str, Any]] = {}
        with ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="key-material-media",
        ) as executor:
            futures = {
                role_label: executor.submit(extract_role, role_label, view_id)
                for role_label, view_id in (
                    ("First-Person", group.first_person_view),
                    ("Third-Person", group.third_person_view),
                )
            }
            for role_label, future in futures.items():
                role_results[role_label] = future.result()

        # Keep archive mutation and incremental publication ordered. Readers
        # never see a sidecar before the corresponding media is complete.
        for role_label, view_id in (
            ("First-Person", group.first_person_view),
            ("Third-Person", group.third_person_view),
        ):
            result = role_results[role_label]
            frame_path = Path(result["frame_path"])
            clip_path = Path(result["clip_path"])
            relative_frame = _relative(frame_path, layout.root)
            relative_clip = _relative(clip_path, layout.root)
            event.key_frames[view_id] = relative_frame
            event.key_clips[view_id] = relative_clip
            frame_paths[role_label] = frame_path
            clip_paths[role_label] = clip_path
            if result["frame_decode_offset_ms"]:
                event.uncertainty.append(
                    f"{view_id} key frame decode offset {result['frame_decode_offset_ms']:+.0f} ms"
                )
            frame_json = frame_dir / f"{role_label}.json"
            clip_json = clip_dir / f"{role_label}.json"
            write_json(
                frame_json,
                _artifact_json(
                    group,
                    event,
                    "key_frame",
                    relative_frame,
                    view_id,
                    transforms,
                    archive_id,
                ),
            )
            write_json(
                clip_json,
                _artifact_json(
                    group,
                    event,
                    "key_clip",
                    relative_clip,
                    view_id,
                    transforms,
                    archive_id,
                ),
            )
            if publisher is not None:
                for artifact in (frame_path, frame_json, clip_path, clip_json):
                    publisher.publish_file(artifact)
            runtime_records.append(
                {
                    key: value
                    for key, value in result.items()
                    if key not in {"frame_path", "clip_path"}
                }
            )

        aligned_started = time.perf_counter()
        aligned_frame = frame_dir / "Aligned_First+Third.jpg"
        _write_aligned_frame(
            frame_paths["First-Person"],
            frame_paths["Third-Person"],
            aligned_frame,
            (group.first_person_view, group.third_person_view),
        )
        aligned_frame_relative = _relative(aligned_frame, layout.root)
        event.key_frames["aligned_first_third"] = aligned_frame_relative
        write_json(
            frame_dir / "Aligned_First+Third.json",
            _artifact_json(
                group,
                event,
                "aligned_first_third_key_frame",
                aligned_frame_relative,
                None,
                transforms,
                archive_id,
            ),
        )
        if publisher is not None:
            publisher.publish_file(aligned_frame)
            publisher.publish_file(frame_dir / "Aligned_First+Third.json")
        aligned_clip = clip_dir / "Aligned_First+Third.mp4"
        create_grid_video(
            [
                (group.first_person_view, clip_paths["First-Person"]),
                (group.third_person_view, clip_paths["Third-Person"]),
            ],
            aligned_clip,
            encoder,
        )
        aligned_clip_relative = _relative(aligned_clip, layout.root)
        event.key_clips["aligned_first_third"] = aligned_clip_relative
        write_json(
            clip_dir / "Aligned_First+Third.json",
            _artifact_json(
                group,
                event,
                "aligned_first_third_key_clip",
                aligned_clip_relative,
                None,
                transforms,
                archive_id,
            ),
        )
        if publisher is not None:
            publisher.publish_file(aligned_clip)
            publisher.publish_file(clip_dir / "Aligned_First+Third.json")
        runtime_records.append(
            {
                "event_id": event.event_id,
                "experiment_group_id": group.group_id,
                "action_type": event.action_type.value,
                "action_category_folder": action_folder,
                "role_label": "Aligned-First-Third",
                "view_id": "aligned_first_third",
                "duration_seconds": round(time.perf_counter() - aligned_started, 6),
                "frame_output_bytes": aligned_frame.stat().st_size,
                "clip_output_bytes": aligned_clip.stat().st_size,
                "event_wall_duration_seconds": round(time.perf_counter() - event_started, 6),
            }
        )

    category_index_path = write_key_material_category_index(
        layout,
        groups,
        events,
        publisher=publisher,
    )
    runtime_path = layout.json_config / "key_material_materialization_runtime.json"
    write_json(
        runtime_path,
        {
            "schema_version": "visioncortex-key-materialization-runtime/1",
            "workers": workers,
            "total_duration_seconds": round(time.perf_counter() - stage_started, 6),
            "detection_ledger_lookup_seconds": round(lookup_seconds, 6),
            "detection_ledger_passes": len(nearest_by_view),
            "archive_hierarchy_version": "2.0.0",
            "category_index": _relative(category_index_path, layout.root),
            "accepted_event_count": sum(
                bool(event.accepted and event.event_id in group_by_event) for event in events
            ),
            "records": runtime_records,
        },
    )
    if publisher is not None:
        publisher.publish_file(runtime_path)


def extract_temporal_review_frames(
    clip_path: Path,
    output_dir: Path,
    view_id: str,
    samples_per_view: int = 3,
) -> list[tuple[str, Path]]:
    """Sample a tiny before/peak/after storyboard from an already-made key clip.

    This avoids another seek through the original NAS video and gives the MLLM
    temporal evidence instead of asking it to infer an action from one still.
    """

    count = max(1, min(5, int(samples_per_view)))
    capture = cv2.VideoCapture(str(clip_path))
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count <= 0:
            return []
        if count == 1:
            fractions = [0.5]
            phases = ["peak"]
        elif count == 2:
            fractions = [0.2, 0.8]
            phases = ["before", "after"]
        elif count == 3:
            fractions = [0.15, 0.5, 0.85]
            phases = ["before", "peak", "after"]
        else:
            fractions = [0.1 + 0.8 * index / (count - 1) for index in range(count)]
            phases = [f"temporal_{index + 1:02d}" for index in range(count)]
        samples: list[tuple[str, Path]] = []
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_view = _safe_slug(view_id)
        for phase, fraction in zip(phases, fractions):
            frame_index = min(frame_count - 1, max(0, round((frame_count - 1) * fraction)))
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            path = output_dir / f"{safe_view}_{phase}.jpg"
            if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 88]):
                continue
            samples.append(
                (
                    f"view_id={view_id}; temporal_phase={phase}; key_clip_frame={frame_index}",
                    path,
                )
            )
        return samples
    finally:
        capture.release()


def analyze_key_materials(layout: ArchiveLayout, events: Sequence[EvidenceEvent], config: dict[str, Any]) -> None:
    analyzer = ArkStepAnalyzer(config)
    accepted = [event for event in events if event.accepted]
    cache_root = layout.work / "mllm-cache" / "key-materials"

    def analyze(event: EvidenceEvent) -> tuple[EvidenceEvent, dict[str, Any]]:
        fallback_images = [
            (view_id, layout.root / relative)
            for view_id, relative in event.key_frames.items()
            if view_id != "aligned_first_third"
        ]
        images: list[tuple[str, Path]] = []
        if bool(config["mllm"].get("use_key_clip_temporal_samples", True)):
            sample_dir = layout.work / "mllm-temporal-samples" / event.event_id
            samples_per_view = int(config["mllm"].get("temporal_samples_per_view", 3))
            by_phase: dict[str, list[tuple[str, Path]]] = {
                "before": [],
                "peak": [],
                "after": [],
            }
            remaining: list[tuple[str, Path]] = []
            for view_id, relative in event.key_clips.items():
                if view_id == "aligned_first_third":
                    continue
                for label, path in extract_temporal_review_frames(
                    layout.root / relative,
                    sample_dir,
                    view_id,
                    samples_per_view=samples_per_view,
                ):
                    phase = next(
                        (
                            item
                            for item in ("before", "peak", "after")
                            if f"temporal_phase={item}" in label
                        ),
                        None,
                    )
                    if phase is None:
                        remaining.append((label, path))
                    else:
                        by_phase[phase].append((label, path))
            for phase in ("before", "peak", "after"):
                images.extend(sorted(by_phase[phase], key=lambda item: item[0]))
            images.extend(sorted(remaining, key=lambda item: item[0]))
        if not images:
            images = fallback_images
        semantic_evidence = event.model_dump(
            mode="json",
            exclude={"model_understanding", "key_frames", "key_clips"},
        )
        fingerprint = _semantic_fingerprint(
            "key-material",
            config,
            EVENT_SYSTEM_PROMPT,
            semantic_evidence,
            images,
        )
        persistent_cache_path = _semantic_cache_path(
            config, "key-materials", fingerprint
        )
        run_cache_path = cache_root / f"{fingerprint}.json"
        cached = _read_semantic_cache(persistent_cache_path, fingerprint)
        if cached is None:
            cached = _read_semantic_cache(run_cache_path, fingerprint)
        if cached is not None:
            return event, cached
        result = analyzer.analyze_event(event, images)
        if result.get("status") == "completed":
            result = _write_semantic_cache(persistent_cache_path, fingerprint, result)
            write_json(run_cache_path, result)
        return event, result

    workers = max(1, int(config["mllm"].get("workers", 4)))
    try:
        with ThreadPoolExecutor(max_workers=min(workers, max(1, len(accepted)))) as executor:
            futures = [executor.submit(analyze, event) for event in accepted]
            for future in as_completed(futures):
                event, result = future.result()
                event.model_understanding = result
                record_semantic_review(event, result)
    finally:
        analyzer.close()


def refresh_key_material_metadata(
    layout: ArchiveLayout,
    events: Sequence[EvidenceEvent],
    groups: Sequence[ExperimentGroup],
    transforms: dict[str, AlignmentTransform],
    archive_id: str | None = None,
) -> None:
    """Rewrite sidecars after MLLM so JSON and media never disagree."""
    group_by_event = {event_id: group for group in groups for event_id in group.key_event_ids}
    for event in events:
        group = group_by_event[event.event_id]
        for collection, artifact in (
            (event.key_frames, "key_frame"),
            (event.key_clips, "key_clip"),
        ):
            for view_id, relative in collection.items():
                media = layout.root / relative
                json_path = media.with_suffix(".json")
                aligned = view_id == "aligned_first_third"
                write_json(
                    json_path,
                    _artifact_json(
                        group,
                        event,
                        f"aligned_first_third_{artifact}" if aligned else artifact,
                        relative,
                        None if aligned else view_id,
                        transforms,
                        archive_id,
                    ),
                )


def _timestamp_rows(events: Sequence[EvidenceEvent]) -> list[dict[str, Any]]:
    rows = []
    for event in events:
        if not event.accepted:
            continue
        rows.append(
            {
                "event_id": event.event_id,
                "action_type": event.action_type.value,
                "global_start_ms": round(event.global_start_ms, 3),
                "global_end_ms": round(event.global_end_ms, 3),
                "key_global_ms": round(event.key_global_ms, 3),
                "objects": ", ".join(event.objects),
                "supporting_views": ", ".join(event.supporting_views),
                "supporting_roles": ", ".join(role.value for role in event.supporting_roles),
                "confidence": round(event.confidence, 5),
                "cross_view": len(event.supporting_views) > 1,
                "uncertainty": "；".join(event.uncertainty),
            }
        )
    return rows


def write_timestamp_tables(layout: ArchiveLayout, events: Sequence[EvidenceEvent], create_xlsx: bool) -> None:
    rows = _timestamp_rows(events)
    csv_path = layout.key_materials / "Key-Material-Timestamps.csv"
    fields = list(rows[0]) if rows else ["event_id", "action_type", "global_start_ms", "global_end_ms"]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    if create_xlsx:
        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "Key Materials"
        sheet.append(fields)
        for row in rows:
            sheet.append([row.get(field) for field in fields])
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions
        for column in sheet.columns:
            width = min(60, max(12, max(len(str(cell.value or "")) for cell in column) + 2))
            sheet.column_dimensions[column[0].column_letter].width = width
        workbook.save(layout.key_materials / "Key-Material-Timestamps.xlsx")


def write_screening_notes(
    layout: ArchiveLayout,
    segments: Sequence[ExperimentSegment],
    events: Sequence[EvidenceEvent],
    rejected_candidates: Sequence[dict[str, Any]],
    all_views: Sequence[ViewInput],
) -> None:
    lines = [
        "LabVision 有界实验片段筛选记录",
        "= 输入路数不等于输出路数；仅通过物理动作持续性与边界审计的视角会生成 MP4。",
        f"输入视角: {len(all_views)}",
        f"接受事件: {sum(event.accepted for event in events)}",
        f"拒绝事件: {sum(not event.accepted for event in events)}",
        f"有效实验段: {len(segments)}",
        "",
    ]
    for segment in segments:
        lines.append(
            f"[{segment.segment_id}] {_time_slug(segment.global_start_ms)} - {_time_slug(segment.global_end_ms)}; "
            f"有效视角={','.join(segment.participating_views)}"
        )
        for view_id, reason in segment.rejected_views.items():
            lines.append(f"  REJECT_VIEW {view_id}: {reason}")
    for item in rejected_candidates:
        lines.append(f"REJECT_EVENT {item['event_id']}: {item['reason']} confidence={item['confidence']:.3f}")
    (layout.key_materials / "Screening-Notes.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def evidence_package_eval(
    root: Path,
    groups: Sequence[ExperimentGroup],
    segments: Sequence[ExperimentSegment],
    key_events: Sequence[EvidenceEvent],
    transforms: dict[str, AlignmentTransform],
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    group_by_event = {
        event_id: group for group in groups for event_id in group.key_event_ids
    }
    for group in groups:
        checks.append(
            {
                "check": "experiment_group_has_atomic_boundaries",
                "id": group.group_id,
                "passed": bool(group.atomic_experiment_ids),
            }
        )
        for artifact in ("first-person", "third-person", "aligned_first_third"):
            relative = group.videos.get(artifact)
            ok = bool(relative and (root / relative).is_file() and (root / relative).stat().st_size > 0)
            checks.append(
                {"check": "experiment_video_exists", "id": f"{group.group_id}:{artifact}", "passed": ok}
            )
            metadata = group.video_json.get(artifact)
            metadata_ok = bool(metadata and (root / metadata).is_file())
            checks.append(
                {"check": "experiment_video_json_exists", "id": f"{group.group_id}:{artifact}", "passed": metadata_ok}
            )
        for view_id in (group.first_person_view, group.third_person_view):
            checks.append(
                {
                    "check": "alignment_not_failed",
                    "id": f"{group.group_id}:{view_id}",
                    "passed": transforms[view_id].state != "failed",
                }
            )
    for event in key_events:
        group = group_by_event.get(event.event_id)
        expected_keys = (
            {group.first_person_view, group.third_person_view, "aligned_first_third"}
            if group is not None
            else set()
        )
        frame_keys = set(event.key_frames)
        clip_keys = set(event.key_clips)
        frame_media_complete = bool(expected_keys) and frame_keys == expected_keys and all(
            (root / event.key_frames[key]).is_file()
            and (root / event.key_frames[key]).stat().st_size > 0
            for key in expected_keys
        )
        clip_media_complete = bool(expected_keys) and clip_keys == expected_keys and all(
            (root / event.key_clips[key]).is_file()
            and (root / event.key_clips[key]).stat().st_size > 0
            for key in expected_keys
        )
        checks.append(
            {
                "check": "aligned_dual_view_key_frames_exist",
                "id": event.event_id,
                "passed": frame_media_complete,
            }
        )
        checks.append(
            {
                "check": "aligned_dual_view_key_clips_exist",
                "id": event.event_id,
                "passed": clip_media_complete,
            }
        )
        checks.append(
            {
                "check": "cross_view_or_explicit_uncertainty",
                "id": event.event_id,
                "passed": len(event.supporting_views) > 1 or bool(event.uncertainty),
            }
        )
    failures = [check for check in checks if not check["passed"]]
    return {
        "passed": not failures and bool(segments),
        "checks": checks,
        "failures": failures,
        "segment_count": len(segments),
        "experiment_group_count": len(groups),
        "key_event_count": len(key_events),
        "cross_view_event_count": sum(len(event.supporting_views) > 1 for event in key_events),
        "dual_view_material_count": sum(
            all(
                check["passed"]
                for check in checks
                if check["id"] == event.event_id
                and check["check"]
                in {"aligned_dual_view_key_frames_exist", "aligned_dual_view_key_clips_exist"}
            )
            for event in key_events
        ),
    }


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
    write_timestamp_tables(layout, key_events, bool(config["archive"]["create_xlsx"]))
    write_screening_notes(layout, segments, events, rejected_candidates, manifest.views)
    summary = RunSummary(
        experiment_id=manifest.experiment_id,
        views=manifest.views,
        alignments=list(transforms.values()),
        events=events,
        segments=segments,
        experiment_groups=groups,
        physical_change_log=physical_changes,
        stats={
            "input_view_count": len(manifest.views),
            "output_clip_view_count": len({view for segment in segments for view in segment.participating_views}),
            "accepted_event_count": sum(event.accepted for event in events),
            "rejected_event_count": sum(not event.accepted for event in events),
            "experiment_group_count": len(groups),
            "key_event_count": len(key_events),
            "disk_preflight": disk_report,
            "run_metrics": run_metrics or {},
        },
    )
    write_json(layout.json_config / "run_manifest.json", manifest.model_dump(mode="json"))
    write_json(layout.json_config / "time_alignment.json", [item.model_dump(mode="json") for item in transforms.values()])
    write_json(layout.json_config / "physical_change_log.json", [item.model_dump(mode="json") for item in physical_changes])
    write_json(layout.json_config / "evidence_package.json", summary.model_dump(mode="json"))
    if run_metrics is not None:
        write_json(layout.json_config / "run_metrics.json", run_metrics)
    normalized_events = [
        _artifact_json(
            next(group for group in groups if event.event_id in group.key_event_ids),
            event,
            "key_material_event_index",
            "",
            None,
            transforms,
            manifest.experiment_id,
        )
        for event in key_events
    ]
    write_json(
        layout.key_materials / "Key-Materials-Model-Understanding.json",
        normalized_events,
    )
    clip_analysis = {
        "experiment_id": manifest.experiment_id,
        "global_timeline": True,
        "segments": [segment.model_dump(mode="json") for segment in segments],
        "experiment_groups": [group.model_dump(mode="json") for group in groups],
        "steps": [
            {
                "event_id": event.event_id,
                "action_type": event.action_type.value,
                "start_global_ms": event.global_start_ms,
                "end_global_ms": event.global_end_ms,
                "supporting_views": event.supporting_views,
                "model_understanding": event.model_understanding,
            }
            for event in key_events
        ],
    }
    write_json(layout.json_config / "Experiment-Groups-Step-Level-Analysis.json", clip_analysis)
    evaluation = evidence_package_eval(layout.root, groups, segments, key_events, transforms)
    write_json(layout.json_config / "evidence_package_eval.json", evaluation)
    index_manifest = build_archive_index(
        layout.root,
        manifest.experiment_id,
        normalized_events,
        events,
        groups,
        infos,
        hash_workers=int(config.get("performance", {}).get("io_workers", 4)),
    )
    summary.stats["evidence_index"] = {
        "schema_version": index_manifest["schema_version"],
        "manifest": "JSON-Config-Files/evidence_index_manifest.json",
        "counts": index_manifest["counts"],
        "fts5_enabled": index_manifest["fts5_enabled"],
    }
    index_validation = index_manifest.get("validation") or {}
    index_check = {
        "id": manifest.experiment_id,
        "check": "evidence_index_and_one_hop_registries",
        "passed": index_validation.get("passed") is True
        and index_validation.get("all_artifacts_have_integrity_receipts") is True,
        "details": index_validation,
    }
    evaluation["checks"].append(index_check)
    if not index_check["passed"]:
        evaluation["failures"].append(
            "Evidence index or one-hop artifact/evidence registry validation failed"
        )
        evaluation["passed"] = False
    write_json(layout.json_config / "evidence_package_eval.json", evaluation)
    write_json(layout.json_config / "evidence_package.json", summary.model_dump(mode="json"))
    return summary
