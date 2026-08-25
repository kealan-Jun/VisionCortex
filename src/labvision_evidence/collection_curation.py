from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


CURATION_SCHEMA_VERSION = "visioncortex-experiment-curation/1"
CONTENT_PROBE_SCHEMA_VERSION = "visioncortex-content-probe/2"
FULL_TIMELINE_SWEEP_SCHEMA_VERSION = (
    "visioncortex-full-timeline-content-sweep/1"
)
CONTENT_PROBE_DISTRIBUTION_STRATEGY = (
    "full_timeline_stratified_anchors_plus_motion_peaks"
)
CONTENT_PROBE_ANCHOR_FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)

REAL_EXPERIMENT_PREFIXES = (
    "CapToggle_",
    "CustomFlow_",
    "DissolveStir_",
    "FullProcess_",
    "Pipetting_",
    "TubeCleanDry_",
    "TubePlaceDilute_",
    "Weighing_",
)
HARDWARE_TEST_PREFIXES = ("recording_endurance_", "recording_matrix_")


def evaluate_full_timeline_semantic_consensus(
    sweep: dict[str, Any],
) -> dict[str, Any]:
    """Classify a full sweep without treating mere object motion as an experiment.

    The strongest mode remains unanimous strict-negative Ark verdicts. A
    fail-closed fallback is permitted only when every call completed, every
    source view was sampled, no chunk claims a real experiment, and every
    chunk explicitly says that no purposeful experimental action chain exists.
    The fallback still requires an independent visual audit and exhaustive CV
    in the downstream adjudicator.
    """

    chunks = sweep.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        return {"eligible": False, "mode": None, "reason": "chunks_missing"}
    if sweep.get("all_ark_completed") is not True:
        return {"eligible": False, "mode": None, "reason": "ark_incomplete"}
    if sweep.get("all_source_views_sampled") is not True:
        return {
            "eligible": False,
            "mode": None,
            "reason": "source_view_coverage_incomplete",
        }
    if (
        sweep.get("status")
        == "no_visible_experimental_action_across_full_timeline"
        and sweep.get("all_chunks_negative") is True
        and all(
            chunk.get("ark_status") == "completed"
            and chunk.get("gated_verdict") == "recording_or_hardware_test"
            for chunk in chunks
        )
    ):
        return {
            "eligible": True,
            "mode": "unanimous_strict_negative",
            "reason": "all_chunks_passed_the_strict_negative_gate",
            "inconclusive_chunk_indices": [],
        }

    inconclusive_indices: list[int] = []
    for position, chunk in enumerate(chunks, start=1):
        model_result = chunk.get("model_result") or {}
        if chunk.get("ark_status") != "completed":
            return {"eligible": False, "mode": None, "reason": "ark_incomplete"}
        if (
            model_result.get("status") != "completed"
            or model_result.get("verdict") == "real_experiment"
            or model_result.get("purposeful_experimental_action_chain") is not False
            or bool(model_result.get("action_chain_steps"))
        ):
            return {
                "eligible": False,
                "mode": None,
                "reason": "purposeful_action_chain_not_unanimously_absent",
            }
        if chunk.get("gated_verdict") != "recording_or_hardware_test":
            inconclusive_indices.append(int(chunk.get("chunk_index") or position))
    return {
        "eligible": True,
        "mode": "purposeful_action_chain_absent",
        "reason": "every_ark_chunk_explicitly_rejected_a_purposeful_action_chain",
        "inconclusive_chunk_indices": inconclusive_indices,
    }


def _prefixes(config: dict[str, Any], key: str, default: tuple[str, ...]) -> tuple[str, ...]:
    values = (config.get("collection_curation") or {}).get(key, default)
    return tuple(str(value).strip() for value in values if str(value).strip())


def _matches_prefix(value: str, prefixes: tuple[str, ...]) -> bool:
    folded = value.casefold()
    return any(folded.startswith(prefix.casefold()) for prefix in prefixes)


def _receipt_dir(config: dict[str, Any]) -> Path:
    configured = (config.get("collection_curation") or {}).get(
        "content_probe_receipt_dir"
    )
    return (
        Path(str(configured))
        if configured
        else Path(config["storage"]["local_cache_root"])
        / "Collection-Catalog"
        / "Content-Probe-Receipts"
    )


def _receipt_path(config: dict[str, Any], experiment_id: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", experiment_id).strip("-.")
    if not safe:
        raise ValueError("unsafe empty experiment_id")
    return _receipt_dir(config) / f"{safe}.json"


def _source_readiness_receipt_path(
    config: dict[str, Any], experiment_id: str
) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", experiment_id).strip("-.")
    if not safe:
        raise ValueError("unsafe empty experiment_id")
    return (
        Path(config["storage"]["local_cache_root"])
        / "Collection-Catalog"
        / "Content-Probe-Source-Readiness"
        / f"{safe}.json"
    )


def write_probe_source_readiness(
    config: dict[str, Any],
    experiment_id: str,
    *,
    status: str,
    error: str | None = None,
) -> Path:
    if status not in {"blocked", "ready"}:
        raise ValueError("source readiness status must be blocked or ready")
    path = _source_readiness_receipt_path(config, experiment_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "visioncortex-probe-source-readiness/1",
        "experiment_id": experiment_id,
        "status": status,
        "error": error if status == "blocked" else None,
        "source_copy_bytes": 0,
        "updated_at": datetime.now().astimezone().isoformat(),
    }
    temporary = path.with_name(f".{path.name}.partial-{uuid.uuid4().hex[:8]}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return path


def _read_probe_source_readiness(
    config: dict[str, Any], experiment_id: str
) -> dict[str, Any] | None:
    path = _source_readiness_receipt_path(config, experiment_id)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        payload.get("schema_version") != "visioncortex-probe-source-readiness/1"
        or payload.get("experiment_id") != experiment_id
        or payload.get("status") not in {"blocked", "ready"}
        or int(payload.get("source_copy_bytes", -1)) != 0
    ):
        return None
    return {**payload, "path": str(path)}


def _read_probe(
    config: dict[str, Any],
    experiment_id: str,
    max_seconds: float,
    max_views: int,
    expected_timeline_seconds: float | None,
) -> dict[str, Any]:
    path = _receipt_path(config, experiment_id)
    result = {
        "path": str(path),
        "status": "missing",
        "valid": False,
        "verdict": None,
        "errors": [],
    }
    if not path.is_file():
        return result
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {**result, "status": "invalid", "errors": [type(exc).__name__]}
    errors: list[str] = []
    if payload.get("schema_version") != CONTENT_PROBE_SCHEMA_VERSION:
        errors.append("schema_version_mismatch")
    if payload.get("experiment_id") != experiment_id:
        errors.append("experiment_id_mismatch")
    if payload.get("scope") != "classification_only":
        errors.append("scope_must_be_classification_only")
    if payload.get("production_completion_eligible") is not False:
        errors.append("probe_cannot_be_production_completion")
    if int(payload.get("source_copy_bytes", -1)) != 0:
        errors.append("source_copy_bytes_must_be_zero")
    duration = float(payload.get("decoded_timeline_seconds", -1.0))
    if duration < 0.0 or duration > max_seconds:
        errors.append("decoded_timeline_exceeds_probe_limit")
    timeline_duration = float(payload.get("timeline_duration_seconds", -1.0))
    if timeline_duration <= 0.0:
        errors.append("timeline_duration_must_be_positive")
    elif expected_timeline_seconds is not None and abs(
        timeline_duration - expected_timeline_seconds
    ) > max(1.0, expected_timeline_seconds * 0.001):
        errors.append("timeline_duration_mismatch")
    distribution_strategy = payload.get("distribution_strategy")
    if distribution_strategy != CONTENT_PROBE_DISTRIBUTION_STRATEGY:
        errors.append("distributed_full_timeline_strategy_required")
    intervals = payload.get("sampled_intervals") or []
    valid_intervals: list[tuple[float, float, str]] = []
    if not isinstance(intervals, list) or not intervals:
        errors.append("sampled_intervals_required")
    else:
        for interval in intervals:
            if not isinstance(interval, dict):
                errors.append("sampled_interval_must_be_object")
                continue
            try:
                start = float(interval["start_seconds"])
                end = float(interval["end_seconds"])
            except (KeyError, TypeError, ValueError):
                errors.append("sampled_interval_bounds_invalid")
                continue
            reason = str(interval.get("selection_reason") or "")
            if reason not in {"uniform_anchor", "motion_peak"}:
                errors.append("sampled_interval_selection_reason_invalid")
            if start < 0.0 or end <= start or (
                timeline_duration > 0.0 and end > timeline_duration + 0.001
            ):
                errors.append("sampled_interval_out_of_bounds")
                continue
            valid_intervals.append((start, end, reason))
    ordered = sorted(valid_intervals)
    if any(current[0] < previous[1] - 0.001 for previous, current in zip(ordered, ordered[1:])):
        errors.append("sampled_intervals_overlap")
    sampled_duration = sum(end - start for start, end, _ in valid_intervals)
    if abs(sampled_duration - duration) > 0.1:
        errors.append("decoded_timeline_interval_sum_mismatch")
    anchor_bands_verified = False
    motion_peak_verified = False
    if timeline_duration > 0.0 and valid_intervals:
        if timeline_duration <= max_seconds:
            anchor_bands_verified = (
                min(start for start, _, _ in valid_intervals) <= 0.001
                and max(end for _, end, _ in valid_intervals)
                >= timeline_duration - 0.001
            )
            if not anchor_bands_verified:
                errors.append("short_probe_must_cover_full_timeline")
            motion_peak_verified = True
        else:
            band_half_width = timeline_duration * 0.05
            anchor_bands_verified = all(
                any(
                    reason == "uniform_anchor"
                    and end >= max(0.0, timeline_duration * fraction - band_half_width)
                    and start <= min(
                        timeline_duration,
                        timeline_duration * fraction + band_half_width,
                    )
                    for start, end, reason in valid_intervals
                )
                for fraction in CONTENT_PROBE_ANCHOR_FRACTIONS
            )
            if not anchor_bands_verified:
                errors.append("full_timeline_anchor_bands_missing")
            motion_peak_verified = any(
                reason == "motion_peak" for _, _, reason in valid_intervals
            )
            if not motion_peak_verified:
                errors.append("motion_peak_interval_required")
    views = payload.get("views") or []
    if not isinstance(views, list) or not 1 <= len(views) <= max_views:
        errors.append("decoded_view_count_exceeds_probe_limit")
    roles = {
        str(view.get("role")) for view in views if isinstance(view, dict)
    }
    if len(views) == 2 and not {"first_person", "third_person"}.issubset(roles):
        errors.append("two_view_probe_must_be_cross_view")
    verdict = payload.get("verdict")
    if verdict not in {
        "real_experiment",
        "recording_or_hardware_test",
        "inconclusive",
    }:
        errors.append("unsupported_verdict")
    full_timeline_adjudication_valid = False
    if verdict == "recording_or_hardware_test" and timeline_duration > max_seconds:
        adjudication_errors = _validate_full_timeline_adjudication(
            config,
            payload,
            experiment_id,
            expected_timeline_seconds,
        )
        if adjudication_errors:
            errors.extend(adjudication_errors)
            errors.append("negative_long_timeline_probe_cannot_exclude_experiment")
        else:
            full_timeline_adjudication_valid = True
    return {
        **result,
        "status": "valid" if not errors else "invalid",
        "valid": not errors,
        "verdict": verdict if not errors else None,
        "errors": errors,
        "decoded_timeline_seconds": payload.get("decoded_timeline_seconds"),
        "timeline_duration_seconds": payload.get("timeline_duration_seconds"),
        "distribution_strategy": distribution_strategy,
        "sampled_interval_count": len(intervals) if isinstance(intervals, list) else None,
        "anchor_bands_verified": anchor_bands_verified,
        "motion_peak_verified": motion_peak_verified,
        "view_count": len(views) if isinstance(views, list) else None,
        "completed_at": payload.get("completed_at"),
        "review_backend": payload.get("review_backend"),
        "full_timeline_adjudication_valid": full_timeline_adjudication_valid,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_full_timeline_adjudication(
    config: dict[str, Any],
    probe: dict[str, Any],
    experiment_id: str,
    expected_timeline_seconds: float | None,
) -> list[str]:
    """Validate the independent semantic-sweep plus exhaustive-CV receipt."""

    errors: list[str] = []
    adjudication = probe.get("full_timeline_adjudication")
    if not isinstance(adjudication, dict):
        return ["full_timeline_adjudication_required"]
    if (
        adjudication.get("schema_version")
        != "visioncortex-full-timeline-cv-adjudication/1"
    ):
        errors.append("full_timeline_adjudication_schema_mismatch")
    if adjudication.get("experiment_id") != experiment_id:
        errors.append("full_timeline_adjudication_experiment_mismatch")
    if int(adjudication.get("source_copy_bytes", -1)) != 0:
        errors.append("full_timeline_adjudication_source_copy_must_be_zero")

    sweep_ref = adjudication.get("semantic_sweep_receipt")
    if not isinstance(sweep_ref, dict):
        return [*errors, "semantic_sweep_receipt_reference_required"]
    try:
        sweep_path = Path(str(sweep_ref["path"])).resolve(strict=True)
        allowed_sweep_root = (
            Path(config["storage"]["local_cache_root"])
            / "Collection-Catalog"
            / "Full-Timeline-Sweep-Receipts"
        ).resolve(strict=True)
    except (KeyError, OSError):
        return [*errors, "semantic_sweep_receipt_missing"]
    if not sweep_path.is_relative_to(allowed_sweep_root):
        errors.append("semantic_sweep_receipt_outside_cache_root")
    actual_sweep_sha = _sha256(sweep_path)
    if sweep_ref.get("sha256") != actual_sweep_sha:
        errors.append("semantic_sweep_receipt_sha256_mismatch")
    try:
        sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return [*errors, "semantic_sweep_receipt_invalid_json"]
    if sweep.get("schema_version") != FULL_TIMELINE_SWEEP_SCHEMA_VERSION:
        errors.append("semantic_sweep_schema_mismatch")
    if sweep.get("experiment_id") != experiment_id:
        errors.append("semantic_sweep_experiment_mismatch")
    if int(sweep.get("source_copy_bytes", -1)) != 0:
        errors.append("semantic_sweep_source_copy_must_be_zero")
    semantic_consensus = evaluate_full_timeline_semantic_consensus(sweep)
    if semantic_consensus.get("eligible") is not True:
        errors.append("semantic_sweep_purposeful_action_chain_absence_required")
    recorded_consensus = adjudication.get("semantic_consensus")
    legacy_unanimous_negative = (
        recorded_consensus is None
        and semantic_consensus.get("mode") == "unanimous_strict_negative"
    )
    if not legacy_unanimous_negative and (
        not isinstance(recorded_consensus, dict)
        or recorded_consensus != semantic_consensus
    ):
        errors.append("semantic_sweep_consensus_receipt_mismatch")
    if semantic_consensus.get("mode") == "purposeful_action_chain_absent":
        visual_ref = adjudication.get("independent_visual_audit")
        if not isinstance(visual_ref, dict):
            errors.append("semantic_sweep_independent_visual_audit_required")
        else:
            try:
                visual_path = Path(str(visual_ref["path"])).resolve(strict=True)
                allowed_visual_root = (
                    Path(config["storage"]["local_cache_root"])
                    / "Collection-Catalog"
                    / "Full-Timeline-Visual-Audit-Receipts"
                ).resolve(strict=True)
                visual = json.loads(visual_path.read_text(encoding="utf-8"))
            except (KeyError, OSError, json.JSONDecodeError):
                errors.append("semantic_sweep_independent_visual_audit_missing")
                visual = {}
                visual_path = None
            if visual_path is not None:
                if not visual_path.is_relative_to(allowed_visual_root):
                    errors.append("semantic_sweep_visual_audit_outside_cache_root")
                if visual_ref.get("sha256") != _sha256(visual_path):
                    errors.append("semantic_sweep_visual_audit_sha256_mismatch")
            if (
                visual.get("schema_version")
                != "visioncortex-full-timeline-visual-audit/1"
                or visual.get("experiment_id") != experiment_id
                or int(visual.get("source_copy_bytes", -1)) != 0
                or visual.get("sweep_receipt_sha256") != actual_sweep_sha
                or visual.get("verdict") != "no_purposeful_action_chain"
                or sorted(
                    int(item)
                    for item in visual.get("reviewed_inconclusive_chunk_indices")
                    or []
                )
                != sorted(
                    int(item)
                    for item in semantic_consensus.get(
                        "inconclusive_chunk_indices"
                    )
                    or []
                )
            ):
                errors.append("semantic_sweep_independent_visual_audit_invalid")
    try:
        recording_duration = float(sweep["recording_timeline_duration_seconds"])
        media_duration = float(sweep["media_timeline_duration_seconds"])
        decoded_duration = float(sweep["decoded_timeline_seconds"])
        coverage_start = float(sweep["coverage_start_seconds"])
        coverage_end = float(sweep["coverage_end_seconds"])
        coverage_gap = float(sweep["coverage_gap_seconds"])
    except (KeyError, TypeError, ValueError):
        errors.append("semantic_sweep_coverage_fields_invalid")
        recording_duration = media_duration = decoded_duration = -1.0
        coverage_start = coverage_end = coverage_gap = -1.0
    if expected_timeline_seconds is not None and abs(
        recording_duration - expected_timeline_seconds
    ) > max(1.0, expected_timeline_seconds * 0.001):
        errors.append("semantic_sweep_recording_timeline_mismatch")
    if media_duration <= 0.0 or media_duration > recording_duration + 1.0:
        errors.append("semantic_sweep_media_timeline_invalid")
    if abs(decoded_duration - media_duration) > 0.1:
        errors.append("semantic_sweep_decoded_timeline_mismatch")
    if (
        abs(coverage_start) > 0.001
        or abs(coverage_end - media_duration) > 0.1
        or abs(coverage_gap) > 0.001
    ):
        errors.append("semantic_sweep_full_media_coverage_required")

    chunks = sweep.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        errors.append("semantic_sweep_chunks_required")
    else:
        cursor = 0.0
        for chunk in chunks:
            try:
                start = float(chunk["start_seconds"])
                end = float(chunk["end_seconds"])
            except (KeyError, TypeError, ValueError):
                errors.append("semantic_sweep_chunk_bounds_invalid")
                continue
            if abs(start - cursor) > 0.001:
                errors.append("semantic_sweep_chunk_gap_or_overlap")
            if end <= start or end - start > 300.001:
                errors.append("semantic_sweep_chunk_limit_invalid")
            if chunk.get("ark_status") != "completed":
                errors.append("semantic_sweep_chunk_ark_incomplete")
            if (
                semantic_consensus.get("mode") == "unanimous_strict_negative"
                and chunk.get("gated_verdict") != "recording_or_hardware_test"
            ):
                errors.append("semantic_sweep_chunk_not_negative")
            cursor = end
        if media_duration > 0.0 and abs(cursor - media_duration) > 0.1:
            errors.append("semantic_sweep_chunk_tail_uncovered")

    cv_ref = adjudication.get("exhaustive_cv")
    if not isinstance(cv_ref, dict):
        return [*errors, "exhaustive_cv_reference_required"]
    try:
        staging_root = Path(str(cv_ref["staging_root"])).resolve(strict=True)
        allowed_staging_root = Path(
            config["storage"]["local_staging_root"]
        ).resolve(strict=True)
    except (KeyError, OSError):
        return [*errors, "exhaustive_cv_staging_missing"]
    if not staging_root.is_relative_to(allowed_staging_root):
        errors.append("exhaustive_cv_outside_staging_root")
    for key in (
        "boundary_precheck",
        "fine_scan_windows",
        "run_metrics",
        "original_ingest",
    ):
        reference = cv_ref.get(key)
        if not isinstance(reference, dict):
            errors.append(f"exhaustive_cv_{key}_reference_required")
            continue
        try:
            evidence_path = Path(str(reference["path"])).resolve(strict=True)
        except (KeyError, OSError):
            errors.append(f"exhaustive_cv_{key}_missing")
            continue
        if not evidence_path.is_relative_to(staging_root):
            errors.append(f"exhaustive_cv_{key}_outside_staging")
        elif reference.get("sha256") != _sha256(evidence_path):
            errors.append(f"exhaustive_cv_{key}_sha256_mismatch")
    if int(cv_ref.get("predicted_experiment_count", -1)) != 0:
        errors.append("exhaustive_cv_zero_groups_required")
    eligible = {str(item) for item in cv_ref.get("eligible_view_ids") or []}
    scanned = {str(item) for item in cv_ref.get("scanned_view_ids") or []}
    if not eligible or scanned != eligible:
        errors.append("exhaustive_cv_all_eligible_views_required")
    if cv_ref.get("not_scanned_view_ids"):
        errors.append("exhaustive_cv_unscanned_views_present")
    if int(cv_ref.get("production_model_calls", -1)) != 0:
        errors.append("exhaustive_cv_must_precede_model_calls")
    return list(dict.fromkeys(errors))


def _curate(collection: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    settings = config.get("collection_curation") or {}
    max_seconds = float(settings.get("content_probe_max_seconds", 300.0))
    max_views = int(settings.get("content_probe_max_views", 2))
    experiment_id = str(collection["experiment_id"])
    display_name = str(collection.get("display_name") or experiment_id)
    duration = collection.get("duration_seconds")
    duration = float(duration) if duration is not None else None
    if _matches_prefix(
        display_name,
        _prefixes(config, "hardware_test_prefixes", HARDWARE_TEST_PREFIXES),
    ):
        metadata_class = "recording_or_hardware_test"
    elif _matches_prefix(
        display_name,
        _prefixes(config, "real_experiment_prefixes", REAL_EXPERIMENT_PREFIXES),
    ):
        metadata_class = "real_experiment"
    else:
        metadata_class = "ambiguous"
    probe = _read_probe(
        config,
        experiment_id,
        max_seconds,
        max_views,
        duration,
    )
    classification = metadata_class
    classification_source = "metadata_prefix_policy"
    failed_real_prefix_negative_probe = bool(
        metadata_class == "real_experiment"
        and str((collection.get("processing") or {}).get("state")) == "failed"
        and probe["valid"]
        and probe["verdict"] == "recording_or_hardware_test"
        and duration is not None
        and duration <= max_seconds
        and probe.get("anchor_bands_verified") is True
    )
    if metadata_class == "ambiguous" and probe["valid"]:
        classification = str(probe["verdict"])
        classification_source = "bounded_content_probe"
    elif failed_real_prefix_negative_probe:
        # A name prefix is only metadata, not ground truth. Permit a strict
        # override after a real cold production attempt found no evidence and
        # an independent Ark probe reviewed the entire short timeline across
        # a first/third-person pair. Long recordings still require the stronger
        # full-timeline sweep plus exhaustive-CV adjudication contract.
        classification = "recording_or_hardware_test"
        classification_source = "full_short_probe_after_empty_production"
    collection_status = str(collection.get("status") or "unknown")
    processing_state = str(
        (collection.get("processing") or {}).get("state") or "not_processed"
    )
    if processing_state == "archived":
        classification = "real_experiment"
        classification_source = "accepted_formal_archive"
    source_readiness = _read_probe_source_readiness(config, experiment_id)
    blocking_issues = list(collection.get("blocking_issues") or [])
    if source_readiness and source_readiness.get("status") == "blocked":
        collection_status = "attention"
        blocking_issues.append(
            {
                "code": "probe_source_file_missing",
                "message": source_readiness.get("error"),
                "receipt": source_readiness.get("path"),
            }
        )
    if processing_state == "archived":
        disposition = "already_formally_archived"
    elif classification == "recording_or_hardware_test":
        disposition = "excluded_test"
    elif collection_status != "ready":
        disposition = "blocked_source_readiness"
    elif classification in {"ambiguous", "inconclusive"}:
        disposition = "requires_bounded_content_probe"
    else:
        disposition = "eligible_full_timeline_production"
    production_eligible = disposition == "eligible_full_timeline_production"
    resolved = collection.get("resolved_view_counts") or {}
    return {
        "experiment_id": experiment_id,
        "display_name": display_name,
        "collection_status": collection_status,
        "processing_state": processing_state,
        "classification": classification,
        "classification_source": classification_source,
        "metadata_classification": metadata_class,
        "disposition": disposition,
        "production_eligible": production_eligible,
        "blocking_issues": blocking_issues,
        "content_probe": {
            "required": metadata_class == "ambiguous",
            "maximum_timeline_seconds": max_seconds,
            "maximum_view_count": max_views,
            "minimum_cross_view_pair": True,
            "distribution_strategy": CONTENT_PROBE_DISTRIBUTION_STRATEGY,
            "required_anchor_fractions": list(CONTENT_PROBE_ANCHOR_FRACTIONS),
            "motion_peak_interval_required_for_long_timeline": True,
            "scope": "classification_only",
            "may_satisfy_production_completion": False,
            "receipt": probe,
        },
        "production_run": {
            "required_if_eligible": True,
            "full_timeline": True,
            "timeline_start_seconds": 0.0,
            "timeline_end_seconds": duration,
            "timeline_duration_seconds": duration,
            "input_camera_count": int(collection.get("camera_count") or 0),
            "resolved_first_person_view_count": int(
                resolved.get("first_person") or 0
            ),
            "resolved_third_person_view_count": int(
                resolved.get("third_person") or 0
            ),
            "source_copy_bytes_required": 0,
            "probe_receipt_is_completion_receipt": False,
            "formal_archive_required": True,
        },
        "source_catalog_fingerprint_sha256": collection.get("fingerprint_sha256"),
    }


def build_curation_catalog(
    collection_catalog: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    experiments = [
        _curate(collection, config)
        for collection in collection_catalog.get("collections") or []
    ]
    dispositions = Counter(item["disposition"] for item in experiments)
    return {
        "schema_version": CURATION_SCHEMA_VERSION,
        "generated_at": datetime.now().astimezone().isoformat(),
        "source_index": collection_catalog.get("index"),
        "policy": {
            "content_probe_maximum_timeline_seconds": float(
                (config.get("collection_curation") or {}).get(
                    "content_probe_max_seconds", 300.0
                )
            ),
            "content_probe_may_satisfy_production_completion": False,
            "content_probe_distribution_strategy": (
                CONTENT_PROBE_DISTRIBUTION_STRATEGY
            ),
            "content_probe_required_anchor_fractions": list(
                CONTENT_PROBE_ANCHOR_FRACTIONS
            ),
            "content_probe_motion_peak_interval_required_for_long_timeline": True,
            "production_full_timeline_required": True,
            "production_truncation_allowed": False,
            "production_zero_copy_required": True,
            "one_formal_archive_directory_per_experiment": True,
            "execution_order": "serial_single_gpu",
        },
        "summary": {
            "total": len(experiments),
            "dispositions": dict(sorted(dispositions.items())),
        },
        "experiments": experiments,
    }


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{uuid.uuid4().hex[:8]}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_curation_catalog(
    config: dict[str, Any], payload: dict[str, Any]
) -> dict[str, str]:
    root = Path(config["storage"]["local_cache_root"]) / "Collection-Catalog"
    json_path = root / "experiment_curation_catalog.json"
    csv_path = root / "experiment_curation_catalog.csv"
    _atomic_write(json_path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    fields = (
        "experiment_id",
        "display_name",
        "collection_status",
        "classification",
        "disposition",
        "production_eligible",
        "content_probe_required",
        "production_full_timeline",
        "production_timeline_duration_seconds",
        "source_copy_bytes_required",
    )
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for item in payload.get("experiments") or []:
        writer.writerow(
            {
                "experiment_id": item["experiment_id"],
                "display_name": item["display_name"],
                "collection_status": item["collection_status"],
                "classification": item["classification"],
                "disposition": item["disposition"],
                "production_eligible": item["production_eligible"],
                "content_probe_required": item["content_probe"]["required"],
                "production_full_timeline": item["production_run"]["full_timeline"],
                "production_timeline_duration_seconds": item["production_run"][
                    "timeline_duration_seconds"
                ],
                "source_copy_bytes_required": item["production_run"][
                    "source_copy_bytes_required"
                ],
            }
        )
    _atomic_write(csv_path, buffer.getvalue())
    return {"json": str(json_path), "csv": str(csv_path)}
