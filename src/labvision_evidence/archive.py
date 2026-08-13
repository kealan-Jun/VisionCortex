from __future__ import annotations

import csv
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
from openpyxl import Workbook

from .alignment import iter_aligned_rows
from .detection import nearest_frame_evidence
from .mllm import ArkStepAnalyzer
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
    create_grid_video,
    extract_view_clip,
    read_view_frame_at,
    write_annotated_frame,
)


ACTION_SLUGS = {
    "hand_object_contact": "Hand-Object-Contact",
    "object_movement": "Object-Movement",
    "liquid_movement": "Liquid-Transfer",
    "container_state_change": "Container-State-Change",
    "device_panel_operation": "Device-Operation",
}


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
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "-", value).strip(" .-")
    return cleaned[:120] or "待命名实验"


class ArchiveLayout:
    def __init__(self, root: Path):
        self.root = root
        self.experiment_clips = root / "Experiment-Clips"
        self.json_config = root / "JSON-Config-Files"
        self.key_materials = root / "Key-Materials"
        self.key_clips = self.key_materials / "Key-Clips"
        self.key_frames = self.key_materials / "Key-Frames"
        self.daily_reports = root / "Lab-Daily-Reports"
        self.professional_pdfs = root / "Professional-PDFs"
        self.work = root / ".work"

    def create(self) -> None:
        for path in (
            self.experiment_clips,
            self.json_config,
            self.key_clips,
            self.key_frames,
            self.daily_reports,
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
    for index, group in enumerate(groups, 1):
        folder_name = group.archive_folder or _safe_folder_name(
            f"{index:03d}_{group.experiment_name}_{group.experiment_name_en}"
        )
        group.archive_folder = folder_name
        group_root = layout.experiment_clips / folder_name
        videos_dir = group_root
        json_dir = group_root
        group_root.mkdir(parents=True, exist_ok=True)
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
            base = f"{role_label}_{_safe_slug(view_id)}"
            destination = videos_dir / f"{base}.mp4"
            extract_view_clip(
                view, infos[view_id], destination, local_start, local_end - local_start, encoder
            )
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
    by_view = {view.view_id: view for view in views}
    by_segment = {segment.segment_id: segment for segment in segments}
    max_pairs = max(2, int(config["mllm"].get("storyboard_pairs_per_group", 6)))

    def analyze(group: ExperimentGroup) -> tuple[ExperimentGroup, dict[str, Any]]:
        storyboard: list[tuple[str, Path]] = []
        storyboard_dir = layout.work / "group-storyboards" / group.group_id
        for index, global_ms in enumerate(_storyboard_times(group, events, max_pairs), 1):
            for role_label, view_id in (
                ("first_person", group.first_person_view),
                ("third_person", group.third_person_view),
            ):
                local_ms = transforms[view_id].to_local(global_ms)
                if not 0.0 <= local_ms <= infos[view_id].duration_ms:
                    continue
                frame = read_view_frame_at(by_view[view_id], infos[view_id], local_ms)
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
        return group, analyzer.analyze_group(group, atomic, group_events, storyboard)

    workers = max(1, int(config["mllm"].get("group_workers", 2)))
    with ThreadPoolExecutor(max_workers=min(workers, max(1, len(groups)))) as executor:
        futures = [executor.submit(analyze, group) for group in groups]
        for future in as_completed(futures):
            group, result = future.result()
            group.model_understanding = result
            if result.get("status") == "completed":
                group.experiment_name = str(result.get("experiment_name") or group.experiment_name)
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
            group.archive_folder = _safe_folder_name(
                f"{index:03d}_{group.experiment_name}_{group.experiment_name_en}"
            )
            for segment_id in group.atomic_experiment_ids:
                segment = by_segment[segment_id]
                segment.experiment_name = group.experiment_name
                segment.experiment_name_en = group.experiment_name_en
                segment.semantic_understanding = result


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
) -> dict[str, Any]:
    views = [view_id] if view_id else [group.first_person_view, group.third_person_view]
    understanding = event.model_understanding or {}
    physical_change = understanding.get("physical_change") or {}
    before_state = str(physical_change.get("before") or "unknown")
    after_state = str(physical_change.get("after") or "unknown")
    model_objects = [str(item) for item in understanding.get("objects") or []]
    object_names = list(dict.fromkeys([*event.objects, *model_objects]))

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
        "evidence_ids": list(dict.fromkeys([*candidate_ids, *frame_evidence_ids])),
        "provenance": {
            "schema_version": "key-material-event-v1.0.0",
            "time_base": "aligned_global_timeline_microseconds",
            "experiment_group_id": group.group_id,
            "experiment_name": group.experiment_name,
            "continuity_type": group.continuity_type,
            "atomic_experiment_ids": group.atomic_experiment_ids,
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
        },
    }


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
) -> None:
    by_view = {view.view_id: view for view in views}
    before = float(config["segmentation"]["key_clip_pre_seconds"]) * 1000.0
    after = float(config["segmentation"]["key_clip_post_seconds"]) * 1000.0
    encoder = config["performance"]["ffmpeg_video_encoder"]
    group_by_event = {event_id: group for group in groups for event_id in group.key_event_ids}
    for event in events:
        if not event.accepted or event.event_id not in group_by_event:
            continue
        group = group_by_event[event.event_id]
        folder = group.archive_folder or _safe_folder_name(group.group_id)
        event_folder = _safe_folder_name(
            f"{event.event_id}_{event.action_type.value}_{_time_slug(event.key_global_ms)}"
        )
        frame_dir = layout.key_frames / folder / event_folder
        clip_dir = layout.key_clips / folder / event_folder
        frame_dir.mkdir(parents=True, exist_ok=True)
        clip_dir.mkdir(parents=True, exist_ok=True)
        frame_paths: dict[str, Path] = {}
        clip_paths: dict[str, Path] = {}
        # Materialize the same global instant from every participating camera,
        # not only the camera that originally triggered the CV event.
        for role_label, view_id in (
            ("First-Person", group.first_person_view),
            ("Third-Person", group.third_person_view),
        ):
            view = by_view[view_id]
            transform = transforms[view_id]
            local_key_ms = transform.to_local(event.key_global_ms)
            if not 0.0 <= local_key_ms <= infos[view_id].duration_ms:
                event.uncertainty.append(f"{view_id} 关键时间超出视频范围")
                continue
            frame = read_view_frame_at(view, infos[view_id], local_key_ms)
            if frame is None:
                event.uncertainty.append(f"{view_id} 关键帧解码失败")
                continue
            nearest = nearest_frame_evidence(detection_paths[view_id], event.key_global_ms)
            boxes = [box.model_dump() for box in nearest.detections] if nearest else []
            base = f"{role_label}_{_safe_slug(view_id)}"
            frame_path = frame_dir / f"{base}.jpg"
            write_annotated_frame(frame, boxes, frame_path)
            relative_frame = _relative(frame_path, layout.root)
            event.key_frames[view_id] = relative_frame
            frame_paths[role_label] = frame_path
            write_json(
                frame_dir / f"{base}.json",
                _artifact_json(
                    group, event, "key_frame", relative_frame, view_id, transforms
                ),
            )
            if publisher is not None:
                publisher.publish_file(frame_path)
                publisher.publish_file(frame_dir / f"{base}.json")

            clip_start_global = max(event.global_start_ms - before, 0.0)
            clip_end_global = event.global_end_ms + after
            local_start = max(0.0, transform.to_local(clip_start_global))
            local_end = min(infos[view_id].duration_ms, transform.to_local(clip_end_global))
            if local_end > local_start:
                clip_path = clip_dir / f"{base}.mp4"
                extract_view_clip(
                    view, infos[view_id], clip_path, local_start, local_end - local_start, encoder
                )
                relative_clip = _relative(clip_path, layout.root)
                event.key_clips[view_id] = relative_clip
                clip_paths[role_label] = clip_path
                write_json(
                    clip_dir / f"{base}.json",
                    _artifact_json(
                        group, event, "key_clip", relative_clip, view_id, transforms
                    ),
                )
                if publisher is not None:
                    publisher.publish_file(clip_path)
                    publisher.publish_file(clip_dir / f"{base}.json")

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
                group, event, "aligned_first_third_key_frame", aligned_frame_relative, None, transforms
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
        )
        aligned_clip_relative = _relative(aligned_clip, layout.root)
        event.key_clips["aligned_first_third"] = aligned_clip_relative
        write_json(
            clip_dir / "Aligned_First+Third.json",
            _artifact_json(
                group, event, "aligned_first_third_key_clip", aligned_clip_relative, None, transforms
            ),
        )
        if publisher is not None:
            publisher.publish_file(aligned_clip)
            publisher.publish_file(clip_dir / "Aligned_First+Third.json")


def analyze_key_materials(layout: ArchiveLayout, events: Sequence[EvidenceEvent], config: dict[str, Any]) -> None:
    analyzer = ArkStepAnalyzer(config)
    accepted = [event for event in events if event.accepted]

    def analyze(event: EvidenceEvent) -> tuple[EvidenceEvent, dict[str, Any]]:
        images = [
            (view_id, layout.root / relative)
            for view_id, relative in event.key_frames.items()
            if view_id != "aligned_first_third"
        ]
        return event, analyzer.analyze_event(event, images)

    workers = max(1, int(config["mllm"].get("workers", 4)))
    with ThreadPoolExecutor(max_workers=min(workers, max(1, len(accepted)))) as executor:
        futures = [executor.submit(analyze, event) for event in accepted]
        for future in as_completed(futures):
            event, result = future.result()
            event.model_understanding = result


def refresh_key_material_metadata(
    layout: ArchiveLayout,
    events: Sequence[EvidenceEvent],
    groups: Sequence[ExperimentGroup],
    transforms: dict[str, AlignmentTransform],
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
        checks.append({"check": "three_key_frames_present", "id": event.event_id, "passed": len(event.key_frames) == 3})
        checks.append({"check": "three_key_clips_present", "id": event.event_id, "passed": len(event.key_clips) == 3})
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
    write_json(
        layout.key_materials / "Key-Materials-Model-Understanding.json",
        [
            _artifact_json(
                next(group for group in groups if event.event_id in group.key_event_ids),
                event,
                "key_material_event_index",
                "",
                None,
                transforms,
            )
            for event in key_events
        ],
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
    return summary
