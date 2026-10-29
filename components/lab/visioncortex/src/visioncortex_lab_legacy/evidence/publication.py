"""Final evidence tables, package evaluation and archive publication."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from openpyxl import Workbook

from ..identity import PRODUCT_NAME
from ..schemas import (
    AlignmentTransform,
    EvidenceEvent,
    ExperimentGroup,
    ExperimentSegment,
    PhysicalChange,
    RunManifest,
    RunSummary,
    VideoInfo,
    ViewInput,
    event_is_formal,
)
from .layout import ArchiveLayout


@dataclass(frozen=True)
class PublicationServices:
    """Named execution ports; supplied explicitly at the composition boundary."""

    _artifact_json: Callable[..., Any]
    _key_material_view_pair: Callable[..., Any]
    _time_slug: Callable[..., Any]
    _timestamp_rows: Callable[..., Any]
    build_archive_index: Callable[..., Any]
    evidence_package_eval: Callable[..., Any]
    write_json: Callable[..., Any]
    write_screening_notes: Callable[..., Any]
    write_timestamp_tables: Callable[..., Any]


def _timestamp_rows(events: Sequence[EvidenceEvent]) -> list[dict[str, Any]]:
    rows = []
    for event in events:
        if not event_is_formal(event):
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
                "supporting_roles": ", ".join(
                    role.value for role in event.supporting_roles
                ),
                "confidence": round(event.confidence, 5),
                "cross_view": len(event.supporting_views) > 1,
                "uncertainty": "；".join(event.uncertainty),
            }
        )
    return rows


def write_timestamp_tables(
    layout: ArchiveLayout,
    events: Sequence[EvidenceEvent],
    create_xlsx: bool,
    *,
    services: PublicationServices,
) -> None:
    rows = services._timestamp_rows(events)
    csv_path = layout.key_materials / "Key-Material-Timestamps.csv"
    fields = (
        list(rows[0])
        if rows
        else ["event_id", "action_type", "global_start_ms", "global_end_ms"]
    )
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
            width = min(
                60, max(12, max(len(str(cell.value or "")) for cell in column) + 2)
            )
            sheet.column_dimensions[column[0].column_letter].width = width
        workbook.save(layout.key_materials / "Key-Material-Timestamps.xlsx")


def write_screening_notes(
    layout: ArchiveLayout,
    segments: Sequence[ExperimentSegment],
    events: Sequence[EvidenceEvent],
    rejected_candidates: Sequence[dict[str, Any]],
    all_views: Sequence[ViewInput],
    *,
    services: PublicationServices,
) -> None:
    lines = [
        f"{PRODUCT_NAME} 有界实验片段筛选记录",
        "= 输入路数不等于输出路数；仅通过物理动作持续性与边界审计的视角会生成 MP4。",
        f"输入视角: {len(all_views)}",
        f"正式事件: {sum(event_is_formal(event) for event in events)}",
        f"待复核事件: {sum(event.formal_admission_status == 'provisional' for event in events)}",
        f"拒绝事件: {sum(not event.accepted for event in events)}",
        f"有效实验段: {len(segments)}",
        "",
    ]
    for segment in segments:
        lines.append(
            f"[{segment.segment_id}] {services._time_slug(segment.global_start_ms)} - {services._time_slug(segment.global_end_ms)}; "
            f"有效视角={','.join(segment.participating_views)}"
        )
        for view_id, reason in segment.rejected_views.items():
            lines.append(f"  REJECT_VIEW {view_id}: {reason}")
    for item in rejected_candidates:
        lines.append(
            f"REJECT_EVENT {item['event_id']}: {item['reason']} confidence={item['confidence']:.3f}"
        )
    (layout.key_materials / "Screening-Notes.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def evidence_package_eval(
    root: Path,
    groups: Sequence[ExperimentGroup],
    segments: Sequence[ExperimentSegment],
    key_events: Sequence[EvidenceEvent],
    transforms: dict[str, AlignmentTransform],
    *,
    services: PublicationServices,
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
            ok = bool(
                relative
                and (root / relative).is_file()
                and (root / relative).stat().st_size > 0
            )
            checks.append(
                {
                    "check": "experiment_video_exists",
                    "id": f"{group.group_id}:{artifact}",
                    "passed": ok,
                }
            )
            metadata = group.video_json.get(artifact)
            metadata_ok = bool(metadata and (root / metadata).is_file())
            checks.append(
                {
                    "check": "experiment_video_json_exists",
                    "id": f"{group.group_id}:{artifact}",
                    "passed": metadata_ok,
                }
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
            {*services._key_material_view_pair(group, event), "aligned_first_third"}
            if group is not None
            else set()
        )
        frame_keys = set(event.key_frames)
        clip_keys = set(event.key_clips)
        frame_media_complete = (
            bool(expected_keys)
            and frame_keys == expected_keys
            and all(
                (root / event.key_frames[key]).is_file()
                and (root / event.key_frames[key]).stat().st_size > 0
                for key in expected_keys
            )
        )
        clip_media_complete = (
            bool(expected_keys)
            and clip_keys == expected_keys
            and all(
                (root / event.key_clips[key]).is_file()
                and (root / event.key_clips[key]).stat().st_size > 0
                for key in expected_keys
            )
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
    empty_observation = not segments
    return {
        # Package integrity and observational sufficiency are separate claims.
        # A fully scanned input with no accepted segment is still a valid,
        # reportable package; it is not proof that no physical action occurred.
        "passed": not failures,
        "package_integrity_status": (
            "passed_empty_observation"
            if not failures and empty_observation
            else "passed"
            if not failures
            else "failed"
        ),
        "observation_status": (
            "no_accepted_observation" if empty_observation else "observations_present"
        ),
        "observation_evidence_classification": (
            "PARTIAL_EVIDENCE" if empty_observation else "PROVEN"
        ),
        "negative_action_claim_supported": False if empty_observation else None,
        "checks": checks,
        "failures": failures,
        "segment_count": len(segments),
        "experiment_group_count": len(groups),
        "key_event_count": len(key_events),
        "cross_view_event_count": sum(
            len(event.supporting_views) > 1 for event in key_events
        ),
        "dual_view_material_count": sum(
            all(
                check["passed"]
                for check in checks
                if check["id"] == event.event_id
                and check["check"]
                in {
                    "aligned_dual_view_key_frames_exist",
                    "aligned_dual_view_key_clips_exist",
                }
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
    *,
    services: PublicationServices,
) -> RunSummary:
    services.write_timestamp_tables(
        layout, key_events, bool(config["archive"]["create_xlsx"])
    )
    services.write_screening_notes(
        layout, segments, events, rejected_candidates, manifest.views
    )
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
            "output_clip_view_count": len(
                {view for segment in segments for view in segment.participating_views}
            ),
            "accepted_event_count": sum(event_is_formal(event) for event in events),
            "provisional_event_count": sum(
                event.formal_admission_status == "provisional" for event in events
            ),
            "rejected_event_count": sum(not event.accepted for event in events),
            "experiment_group_count": len(groups),
            "key_event_count": len(key_events),
            "disk_preflight": disk_report,
            "run_metrics": run_metrics or {},
        },
    )
    speech_path = layout.json_config / "speech.json"
    if speech_path.is_file():
        speech_result = json.loads(speech_path.read_text(encoding="utf-8"))
        summary.stats["speech"] = {
            "manifest": "JSON-Config-Files/speech.json",
            "status": speech_result["status"],
            "source_count": len(speech_result["sources"]),
            "transcript_segments": sum(
                chunk["segment_count"]
                for source in speech_result["sources"]
                for chunk in source["chunks"]
            ),
            "accuracy": "NOT_PROVEN",
            "physical_action_confirmation": False,
        }
    services.write_json(
        layout.json_config / "run_manifest.json", manifest.model_dump(mode="json")
    )
    services.write_json(
        layout.json_config / "time_alignment.json",
        [item.model_dump(mode="json") for item in transforms.values()],
    )
    services.write_json(
        layout.json_config / "physical_change_log.json",
        [item.model_dump(mode="json") for item in physical_changes],
    )
    services.write_json(
        layout.json_config / "evidence_package.json", summary.model_dump(mode="json")
    )
    if run_metrics is not None:
        services.write_json(layout.json_config / "run_metrics.json", run_metrics)
    normalized_events = [
        services._artifact_json(
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
    services.write_json(
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
    services.write_json(
        layout.json_config / "Experiment-Groups-Step-Level-Analysis.json", clip_analysis
    )
    evaluation = services.evidence_package_eval(
        layout.root, groups, segments, key_events, transforms
    )
    services.write_json(layout.json_config / "evidence_package_eval.json", evaluation)
    index_manifest = services.build_archive_index(
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
    services.write_json(layout.json_config / "evidence_package_eval.json", evaluation)
    services.write_json(
        layout.json_config / "evidence_package.json", summary.model_dump(mode="json")
    )
    return summary
