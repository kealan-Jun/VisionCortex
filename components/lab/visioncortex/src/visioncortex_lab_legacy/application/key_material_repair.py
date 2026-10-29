"""Bounded key-material repair; no CLI or HTTP command coupling."""

from __future__ import annotations
import hashlib
import json
import shutil
import time
import uuid
from datetime import datetime
from pathlib import Path
from ..actions import build_physical_change_log
from ..action_state_machine import build_event_state_receipt
from ..archive import (
    ArchiveLayout,
    _artifact_json,
    _rerender_curated_participant_annotations,
    curate_semantically_reviewed_key_materials,
    evidence_package_eval,
    materialize_key_materials,
    refresh_key_material_metadata,
    write_json,
    write_timestamp_tables,
)
from ..config import load_config
from ..daily_reports import generate_daily_report_from_archive
from ..indexing import build_archive_index
from ..model_certification import (
    audit_production_model_certification,
)
from ..pipeline import (
    _synchronize_final_event_state_receipts,
    _synchronize_segments_with_final_key_events,
    normalize_final_group_action_language,
    validate_final_step_action_consistency,
)
from ..provenance import write_run_provenance
from ..recall_evaluation import evaluate_key_event_recall
from ..reviewed_artifacts import load_dataset_scoped_json
from ..schema_contracts import (
    validate_archive_contracts_or_raise,
    write_archive_contract_manifest,
)
from ..schemas import RunManifest, RunSummary, VideoInfo, event_is_formal
from ..storage import (
    DERIVED_ARCHIVE_DIRECTORIES,
    ORIGINAL_REFERENCE_NAMES,
    promote_fixed_archive,
    read_current_release_pointer,
    safe_archive_name,
    validate_formal_archive_release,
)
from ..validation import (
    finalize_quality_acceptance_claims,
    validate_experiment_and_material_quality,
)


def _relocate_unreferenced_key_material_event_directories(
    layout: ArchiveLayout, events: list
) -> list[dict[str, str]]:
    """Move stale event folders out of the user-facing key-material tree."""

    relocated: list[dict[str, str]] = []
    for attribute, media_root, media_kind in (
        ("key_frames", layout.key_frames, "Key-Frames"),
        ("key_clips", layout.key_clips, "Key-Clips"),
    ):
        referenced = {
            (layout.root / relative).parent.resolve(strict=True)
            for event in events
            for relative in dict(getattr(event, attribute)).values()
        }
        if not media_root.is_dir():
            continue
        candidates = sorted(
            path
            for experiment_root in media_root.iterdir()
            if experiment_root.is_dir()
            for action_root in experiment_root.iterdir()
            if action_root.is_dir()
            for path in action_root.iterdir()
            if path.is_dir()
        )
        for candidate in candidates:
            if candidate.is_symlink():
                raise RuntimeError(
                    f"Key-material event directory cannot be a symlink: {candidate}"
                )
            resolved = candidate.resolve(strict=True)
            if not resolved.is_relative_to(media_root.resolve(strict=True)):
                raise RuntimeError(
                    f"Key-material event directory escaped media root: {candidate}"
                )
            if resolved in referenced:
                continue
            relative = candidate.relative_to(media_root)
            destination = (
                layout.work
                / "presentation-repair-obsolete-media"
                / media_kind
                / relative
            )
            if destination.exists():
                destination = destination.with_name(
                    f"{destination.name}-retry-{uuid.uuid4().hex[:8]}"
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(candidate), str(destination))
            relocated.append(
                {
                    "media_kind": media_kind,
                    "from": candidate.relative_to(layout.root).as_posix(),
                    "to": str(destination.resolve()),
                    "operation": "relocated_to_persistent_cache",
                }
            )
    return relocated


def _refresh_repaired_quality_acceptance(
    layout: ArchiveLayout,
    settings: dict,
    experiment_id: str,
    groups: list,
    key_events: list,
) -> dict:
    """Re-evaluate final repaired semantics and annotations before promotion."""

    validation = settings.get("validation", {})
    repository_root = Path(__file__).resolve().parents[3]
    baseline, baseline_selection = load_dataset_scoped_json(
        validation.get("acceptance_baseline"),
        experiment_id,
        repository_root=repository_root,
        artifact_label="验收基线",
    )
    report = validate_experiment_and_material_quality(
        groups,
        key_events,
        baseline,
        boundary_match_iou=float(validation.get("boundary_match_iou", 0.50)),
        max_start_error_seconds=float(validation.get("max_start_error_seconds", 8.0)),
        max_end_error_seconds=float(validation.get("max_end_error_seconds", 8.0)),
        minimum_cross_view_event_rate=float(
            validation.get("minimum_cross_view_event_rate", 0.25)
        ),
        require_participant_only_annotations=bool(
            validation.get("require_participant_only_annotations", False)
        ),
    )
    report["baseline_selection"] = dict(baseline_selection)
    ground_truth, ground_truth_selection = load_dataset_scoped_json(
        validation.get("key_event_ground_truth"),
        experiment_id,
        repository_root=repository_root,
        artifact_label="关键事件真值",
    )
    recall_report = evaluate_key_event_recall(key_events, ground_truth)
    recall_report["ground_truth_selection"] = ground_truth_selection
    recall_gate = {
        "evaluated": bool(recall_report.get("evaluated")),
        "passed": None,
        "temporal_iou_threshold": float(validation.get("key_event_recall_iou", 0.50)),
        "minimum_precision": float(validation.get("minimum_key_event_precision", 0.80)),
        "minimum_recall": float(validation.get("minimum_key_event_recall", 0.80)),
        "precision": None,
        "recall": None,
        "small_sample_warning": recall_report.get("small_sample_warning"),
    }
    if recall_gate["evaluated"]:
        selected_threshold = next(
            (
                item
                for item in recall_report.get("threshold_results") or []
                if abs(
                    float(item["temporal_iou_threshold"])
                    - recall_gate["temporal_iou_threshold"]
                )
                < 1e-9
            ),
            None,
        )
        if selected_threshold is None:
            raise ValueError(
                "Configured key-event recall IoU is absent from evaluation thresholds"
            )
        recall_gate["precision"] = selected_threshold.get("precision")
        recall_gate["recall"] = selected_threshold.get("recall")
        recall_gate["passed"] = bool(
            recall_gate["precision"] is not None
            and recall_gate["recall"] is not None
            and float(recall_gate["precision"]) >= recall_gate["minimum_precision"]
            and float(recall_gate["recall"]) >= recall_gate["minimum_recall"]
        )
        report["passed"] = bool(report.get("passed")) and bool(recall_gate["passed"])
        if not recall_gate["passed"]:
            report["status"] = "failed"
    report["key_event_recall"] = recall_gate
    step_consistency = validate_final_step_action_consistency(groups, key_events)
    report["step_action_consistency"] = step_consistency
    if not step_consistency["passed"]:
        report["passed"] = False
        report["status"] = "failed"
    finalize_quality_acceptance_claims(report, recall_gate, step_consistency)
    write_json(layout.json_config / "quality_acceptance.json", report)
    write_json(layout.json_config / "key_material_recall_eval.json", recall_report)
    return report


def _stage_published_archive_for_repair(archive: Path) -> tuple[Path, Path]:
    """Copy one committed release into an isolated repair candidate."""

    run_id = f"repair-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
    archive_name = safe_archive_name(archive.name)
    staging_root = archive.parent / ".VisionCortex-Run-Staging" / archive_name / run_id
    history_root = archive.parent / ".VisionCortex-Run-History" / archive_name / run_id
    for directory in DERIVED_ARCHIVE_DIRECTORIES:
        source = archive / directory
        if not source.is_dir():
            raise ValueError(f"Published archive directory is missing: {source}")
        shutil.copytree(source, staging_root / directory)
    source_references = archive / "Original-Experiment-Videos"
    staged_references = staging_root / "Original-Experiment-Videos"
    staged_references.mkdir(parents=True, exist_ok=True)
    for source in sorted(source_references.iterdir()):
        if source.is_file() and (
            source.name in ORIGINAL_REFERENCE_NAMES
            or source.suffix.lower() == ".ffconcat"
        ):
            shutil.copy2(source, staged_references / source.name)
    return staging_root, history_root


def repair_key_material_presentation(archive: Path, config: Path, *, emit=None) -> dict:
    """Re-select and re-render participant key frames without model calls.

    This command reuses a completed run's immutable detection ledgers and only
    reads the exact source frames/clips needed for accepted key events. It never
    re-runs the full scan, copies source video, or calls Ark.
    """

    emit = emit or (lambda _message: None)
    started = time.perf_counter()
    archive = archive.resolve()
    layout = ArchiveLayout(archive)
    lock_path = archive / "run.lock"
    if lock_path.exists():
        raise ValueError(f"Archive is currently locked: {lock_path}")
    settings = load_config(config)
    if read_current_release_pointer(archive) is not None:
        try:
            validate_formal_archive_release(archive)
        except (OSError, RuntimeError, ValueError) as exc:
            raise ValueError(
                f"Published archive integrity failed before repair: {exc}"
            ) from exc
        staging_root, history_root = _stage_published_archive_for_repair(archive)
        emit(f"published_archive_repair_staging={staging_root}")
        repair_key_material_presentation(staging_root, config, emit=emit)
        promotion = promote_fixed_archive(staging_root, archive, history_root)
        emit(
            json.dumps(
                {
                    "status": "published",
                    "formal_archive": str(archive),
                    "repair_staging": str(staging_root),
                    "previous_release_history": str(history_root),
                    "promotion_verification": promotion.get("verification"),
                    "release_manifest_sha256": promotion.get("release_manifest_sha256"),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return promotion
    model_certification = audit_production_model_certification(settings)
    package_path = layout.json_config / "evidence_package.json"
    package_before_sha256 = hashlib.sha256(package_path.read_bytes()).hexdigest()
    summary = RunSummary.model_validate_json(
        package_path.read_text(encoding="utf-8-sig")
    )
    manifest = RunManifest.model_validate_json(
        (layout.json_config / "run_manifest.json").read_text(encoding="utf-8-sig")
    )
    infos = {
        view_id: VideoInfo.model_validate(payload)
        for view_id, payload in json.loads(
            (layout.json_config / "video_probe.json").read_text(encoding="utf-8-sig")
        ).items()
    }
    transforms = {item.view_id: item for item in summary.alignments}
    cache_identity = json.loads(
        (layout.json_config / "cache_identity.json").read_text(encoding="utf-8-sig")
    )
    layout.work = (
        Path(settings["storage"]["local_cache_root"]).resolve()
        / summary.experiment_id
        / str(cache_identity["cache_key"])
    )
    key_event_ids = {
        event_id
        for group in summary.experiment_groups
        for event_id in group.key_event_ids
    }
    reviewed_key_events = [
        event for event in summary.events if event.event_id in key_event_ids
    ]
    prior_curation_path = layout.json_config / "semantic_key_material_curation.json"
    prior_curation = (
        json.loads(prior_curation_path.read_text(encoding="utf-8-sig"))
        if prior_curation_path.is_file()
        else {}
    )
    if len(reviewed_key_events) > 1:
        key_events, semantic_curation = curate_semantically_reviewed_key_materials(
            layout,
            reviewed_key_events,
            summary.experiment_groups,
            settings,
            publisher=None,
        )
    else:
        key_events = [event for event in reviewed_key_events if event_is_formal(event)]
        semantic_curation = prior_curation
    segment_semantic_repairs = _synchronize_segments_with_final_key_events(
        summary.segments, summary.experiment_groups, key_events
    )
    semantic_state_machine_repairs: list[dict[str, object]] = []
    for event in key_events:
        review = dict(event.semantic_review or {})
        pre_curation_action = str(
            review.get("pre_curation_action_type") or event.action_type.value
        )
        expected_state_action = (
            "liquid_transfer"
            if event.action_type.value == "liquid_movement"
            else event.action_type.value
        )
        current_state_action = str((event.state_machine or {}).get("action_type") or "")
        if (
            pre_curation_action == event.action_type.value
            or current_state_action == expected_state_action
        ):
            continue
        pre_curation_state = dict(
            review.get("pre_curation_state_machine") or event.state_machine or {}
        )
        rebuilt_state = build_event_state_receipt(event, settings)
        rebuilt_state["derivation"] = {
            "source": "semantic_relabel_confirmed_state_proof",
            "pre_curation_action_type": pre_curation_action,
            "final_action_type": event.action_type.value,
            "model_confidence": float(review.get("model_confidence") or 0.0),
            "direct_supporting_views": list(event.supporting_views),
        }
        for transition in rebuilt_state.get("transition_trace") or []:
            transition["source"] = "semantic_relabel_confirmed_state_proof"
        identity = rebuilt_state.get("object_identity") or {}
        identity["track_tokens"] = []
        identity["identity_status"] = "semantic_participant_classes"
        rebuilt_state["object_identity"] = identity
        event.state_machine = rebuilt_state
        review["pre_curation_state_machine"] = pre_curation_state
        review["semantic_state_machine_rebuilt"] = True
        event.semantic_review = review
        semantic_state_machine_repairs.append(
            {
                "event_id": event.event_id,
                "pre_curation_action_type": pre_curation_action,
                "pre_curation_state_action": current_state_action,
                "final_action_type": event.action_type.value,
                "final_state_action": expected_state_action,
            }
        )
    required_view_ids = {
        view_id
        for group in summary.experiment_groups
        for view_id in (group.first_person_view, group.third_person_view)
    }
    detection_paths = {
        view_id: layout.work / "detections-fine" / f"{view_id}.detections.jsonl"
        for view_id in sorted(required_view_ids)
    }
    missing_ledgers = [
        str(path) for path in detection_paths.values() if not path.is_file()
    ]
    if missing_ledgers:
        raise ValueError(
            "Immutable fine-scan ledger is missing: " + "; ".join(missing_ledgers)
        )
    previous_keys = {event.event_id: event.key_global_ms for event in key_events}
    previous_names = {
        group.group_id: {
            "experiment_name": group.experiment_name,
            "experiment_name_en": group.experiment_name_en,
        }
        for group in summary.experiment_groups
    }
    materialize_key_materials(
        layout,
        key_events,
        summary.experiment_groups,
        manifest.views,
        infos,
        transforms,
        detection_paths,
        settings,
        publisher=None,
        archive_id=summary.experiment_id,
    )
    obsolete_media_relocated = _relocate_unreferenced_key_material_event_directories(
        layout, key_events
    )
    final_state_receipt_repairs = _synchronize_final_event_state_receipts(
        key_events, settings
    )
    for event in key_events:
        review = dict(event.semantic_review or {})
        pre_curation_action = str(
            review.get("pre_curation_action_type") or event.action_type.value
        )
        expected_peak_us = round(event.key_global_ms * 1000.0)
        if (
            pre_curation_action == event.action_type.value
            or int((event.state_machine or {}).get("peak_timestamp_us") or -1)
            == expected_peak_us
        ):
            continue
        rebuilt_state = build_event_state_receipt(event, settings)
        rebuilt_state["derivation"] = {
            "source": "semantic_relabel_confirmed_state_proof",
            "pre_curation_action_type": pre_curation_action,
            "final_action_type": event.action_type.value,
            "model_confidence": float(review.get("model_confidence") or 0.0),
            "direct_supporting_views": list(event.supporting_views),
        }
        for transition in rebuilt_state.get("transition_trace") or []:
            transition["source"] = "semantic_relabel_confirmed_state_proof"
        identity = rebuilt_state.get("object_identity") or {}
        identity["track_tokens"] = []
        identity["identity_status"] = "semantic_participant_classes"
        rebuilt_state["object_identity"] = identity
        event.state_machine = rebuilt_state
        review["semantic_state_machine_rebuilt"] = True
        event.semantic_review = review
        semantic_state_machine_repairs.append(
            {
                "event_id": event.event_id,
                "pre_curation_action_type": pre_curation_action,
                "pre_curation_state_action": str(
                    (review.get("pre_curation_state_machine") or {}).get("action_type")
                    or ""
                ),
                "final_action_type": event.action_type.value,
                "final_state_action": str(rebuilt_state["action_type"]),
                "timing_resynchronized_after_key_frame_selection": True,
            }
        )
    final_annotation = _rerender_curated_participant_annotations(
        layout, key_events, summary.experiment_groups, settings
    )
    language_normalization = normalize_final_group_action_language(
        summary.experiment_groups, key_events
    )
    group_by_id = {group.group_id: group for group in summary.experiment_groups}
    for segment in summary.segments:
        group = group_by_id.get(str(segment.group_id))
        if group is None:
            continue
        segment.experiment_name = group.experiment_name
        segment.experiment_name_en = group.experiment_name_en
        segment.semantic_understanding = group.model_understanding
    refreshed_experiment_clip_sidecars: list[str] = []
    for group in summary.experiment_groups:
        atomic_ids = set(group.atomic_experiment_ids)
        atomic_segments = [
            segment.model_dump(mode="json")
            for segment in summary.segments
            if segment.segment_id in atomic_ids
        ]
        for relative in group.video_json.values():
            sidecar = archive / relative
            if not sidecar.is_file():
                continue
            payload = json.loads(sidecar.read_text(encoding="utf-8-sig"))
            payload["group"] = group.model_dump(mode="json")
            payload["atomic_experiments"] = atomic_segments
            write_json(sidecar, payload)
            refreshed_experiment_clip_sidecars.append(str(relative))
    write_json(
        layout.json_config / "experiment_group_understanding.json",
        {
            "schema_version": "visioncortex-experiment-group-understanding/2",
            "refinement_pass": "post_event_semantic_curation",
            "presentation_repair": True,
            "groups": [
                group.model_dump(mode="json") for group in summary.experiment_groups
            ],
        },
    )
    key_understanding_path = (
        layout.json_config / "key_material_model_understanding.json"
    )
    if key_understanding_path.is_file():
        key_understanding = json.loads(
            key_understanding_path.read_text(encoding="utf-8-sig")
        )
        repaired_by_id = {event.event_id: event for event in key_events}
        key_understanding["events"] = [
            repaired_by_id.get(str(payload.get("event_id")), None).model_dump(
                mode="json"
            )
            if repaired_by_id.get(str(payload.get("event_id")), None) is not None
            else payload
            for payload in key_understanding.get("events") or []
        ]
        write_json(key_understanding_path, key_understanding)
    curation_path = layout.json_config / "semantic_key_material_curation.json"
    if curation_path.is_file():
        curation = json.loads(curation_path.read_text(encoding="utf-8-sig"))
        repaired_ids = {
            str(item["event_id"])
            for item in [
                *semantic_state_machine_repairs,
                *final_state_receipt_repairs,
            ]
        }
        for record in curation.get("records") or []:
            if str(record.get("event_id")) in repaired_ids:
                record["semantic_state_machine_rebuilt"] = True
        curation["final_annotation"] = {
            key: value for key, value in final_annotation.items() if key != "records"
        }
        curation["presentation_repair"] = {
            "schema_version": "visioncortex-semantic-presentation-repair/1",
            "full_scan_repeated": False,
            "source_copy_bytes": 0,
            "post_relabel_duplicate_count": semantic_curation.get(
                "post_relabel_duplicate_count", 0
            ),
            "state_receipt_repairs": [
                *semantic_state_machine_repairs,
                *final_state_receipt_repairs,
            ],
            "segment_semantic_repairs": segment_semantic_repairs,
            "obsolete_media_relocated": obsolete_media_relocated,
        }
        write_json(curation_path, curation)
    refresh_key_material_metadata(
        layout,
        key_events,
        summary.experiment_groups,
        transforms,
        archive_id=summary.experiment_id,
    )
    write_timestamp_tables(
        layout, key_events, bool(settings.get("archive", {}).get("create_xlsx", True))
    )
    normalized_events = [
        _artifact_json(
            next(
                group
                for group in summary.experiment_groups
                if event.event_id in group.key_event_ids
            ),
            event,
            "key_material_event_index",
            "",
            None,
            transforms,
            summary.experiment_id,
        )
        for event in key_events
    ]
    write_json(
        layout.key_materials / "Key-Materials-Model-Understanding.json",
        normalized_events,
    )
    summary.physical_change_log = build_physical_change_log(key_events)
    write_json(
        layout.json_config / "physical_change_log.json",
        [item.model_dump(mode="json") for item in summary.physical_change_log],
    )
    write_json(package_path, summary.model_dump(mode="json"))
    quality_report = _refresh_repaired_quality_acceptance(
        layout,
        settings,
        summary.experiment_id,
        summary.experiment_groups,
        key_events,
    )
    evaluation = evidence_package_eval(
        archive,
        summary.experiment_groups,
        summary.segments,
        key_events,
        transforms,
    )
    index_manifest = build_archive_index(
        archive,
        summary.experiment_id,
        normalized_events,
        summary.events,
        summary.experiment_groups,
        infos,
        hash_workers=int(settings.get("performance", {}).get("io_workers", 4)),
    )
    summary.stats["evidence_index"] = {
        "schema_version": index_manifest["schema_version"],
        "manifest": "JSON-Config-Files/evidence_index_manifest.json",
        "counts": index_manifest["counts"],
        "fts5_enabled": index_manifest["fts5_enabled"],
    }
    index_validation = index_manifest.get("validation") or {}
    index_check = {
        "id": summary.experiment_id,
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
    write_json(package_path, summary.model_dump(mode="json"))
    report_manifest = generate_daily_report_from_archive(archive, settings)
    write_run_provenance(
        archive,
        settings,
        model_certification,
        repository_root=Path(__file__).resolve().parents[3],
    )
    write_archive_contract_manifest(archive)
    validate_archive_contracts_or_raise(archive)
    receipt = {
        "schema_version": "visioncortex-key-material-presentation-repair/1",
        "status": (
            "completed"
            if evaluation.get("passed") and quality_report.get("passed")
            else "failed"
        ),
        "experiment_id": summary.experiment_id,
        "model_calls": 0,
        "token_usage": 0,
        "full_scan_repeated": False,
        "source_copy_bytes": 0,
        "source_frame_access": "bounded accepted-key-events only",
        "immutable_detection_ledgers": [str(path) for path in detection_paths.values()],
        "key_frame_changes": [
            {
                "event_id": event.event_id,
                "previous_key_global_ms": previous_keys[event.event_id],
                "selected_key_global_ms": event.key_global_ms,
                "selection_offset_ms": round(
                    event.key_global_ms - previous_keys[event.event_id], 3
                ),
            }
            for event in key_events
        ],
        "experiment_name_changes": [
            {
                "group_id": group.group_id,
                "before": previous_names[group.group_id],
                "after": {
                    "experiment_name": group.experiment_name,
                    "experiment_name_en": group.experiment_name_en,
                },
            }
            for group in summary.experiment_groups
            if previous_names[group.group_id]
            != {
                "experiment_name": group.experiment_name,
                "experiment_name_en": group.experiment_name_en,
            }
        ],
        "language_normalization": language_normalization,
        "semantic_state_machine_repairs": semantic_state_machine_repairs,
        "final_state_receipt_repairs": final_state_receipt_repairs,
        "semantic_recuration": {
            "candidate_count": semantic_curation.get("candidate_count", 0),
            "accepted_count": semantic_curation.get("accepted_count", 0),
            "excluded_count": semantic_curation.get("excluded_count", 0),
            "post_relabel_duplicate_count": semantic_curation.get(
                "post_relabel_duplicate_count", 0
            ),
            "post_relabel_deduplication": semantic_curation.get(
                "post_relabel_deduplication", []
            ),
            "segment_semantic_repairs": segment_semantic_repairs,
        },
        "obsolete_media_relocated": obsolete_media_relocated,
        "final_annotation": {
            key: value for key, value in final_annotation.items() if key != "records"
        },
        "refreshed_experiment_clip_sidecars": sorted(
            refreshed_experiment_clip_sidecars
        ),
        "evidence_package_eval_passed": bool(evaluation.get("passed")),
        "quality_acceptance_passed": bool(quality_report.get("passed")),
        "quality_acceptance_status": quality_report.get("status"),
        "interaction_pair_annotation_gate_passed": bool(
            (quality_report.get("key_materials") or {}).get(
                "interaction_pair_annotation_gate_passed"
            )
        ),
        "action_participant_visibility_gate_passed": bool(
            (quality_report.get("key_materials") or {}).get(
                "action_participant_visibility_gate_passed"
            )
        ),
        "daily_report_passed": bool(report_manifest.get("passed")),
        "package_before_sha256": package_before_sha256,
        "package_after_sha256": hashlib.sha256(package_path.read_bytes()).hexdigest(),
        "duration_seconds": round(time.perf_counter() - started, 6),
    }
    receipt_path = layout.json_config / "key_material_presentation_repair.json"
    write_json(receipt_path, receipt)
    if (
        not evaluation.get("passed")
        or not quality_report.get("passed")
        or not report_manifest.get("passed")
    ):
        raise RepairQualityError(receipt)
    return receipt


class RepairQualityError(ValueError):
    """Repair wrote its receipt but did not satisfy publication quality gates."""

    def __init__(self, receipt):
        super().__init__("Key-material repair did not pass all quality gates")
        self.receipt = receipt
