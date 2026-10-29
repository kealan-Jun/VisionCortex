"""Concurrent photo readers; only the task owner publishes the current projection.

Bound photos share one detector result before both readers start. Cloud reads
the same authorized source crops, independently of local OCR/refinement. A
cloud-first candidate may be shown while the local cross-check is pending.
"""

import copy
import time
from dataclasses import replace
from decimal import Decimal
import re

from field_recognition.recognition.config import RecognitionConfig
from field_recognition.recognition.panel_regions import finish as finish_regions
from field_recognition.measurements.reading_results import build_readings
from field_recognition.measurements.readout_rules import numeric_review_conflict
from field_recognition.measurements.measurement_records import build_records, display_fields


def _project(core, document, local, image, pipeline, valid, clock, started):
    """Use the existing association/validation transaction without another model call."""
    if local.get("panel_regions"):
        finish_regions(
            core,
            document,
            local,
            image,
            pipeline=pipeline,
            valid=valid,
            clock=clock,
            started=started,
        )
    else:
        document.update(copy.deepcopy(local))
        document["local_ocr"] = copy.deepcopy(local)


def _cloud(core, original, template, image, panel, pipeline, valid, started):
    from field_recognition.recognition.panel_readout import primary_unbound_photo

    document = copy.deepcopy(original)
    try:
        if template.get("panel_regions"):
            finish_regions(
                core,
                document,
                template,
                image,
                pipeline=pipeline,
                valid=valid,
                started=started,
            )
        else:
            primary_unbound_photo(
                core,
                document,
                panel,
                pipeline=pipeline,
                valid=valid,
                clock=time.monotonic,
                pause=time.sleep,
                started=started,
            )
        from field_recognition.recognition.led_digits import native_diagnostics

        for region in document.get("panel_regions", []):
            if region.get("class_id") == 0:
                from field_recognition.recognition.photo_orientation import source_panel

                region["source_pixel_review"] = native_diagnostics(source_panel(image, region))
    except Exception as exc:
        document["fallback"] = {"status": "failed", "error": type(exc).__name__}
    return document


def _source_conflict(cloud_region, initial_values, cloud_values):
    """Cloud confidence cannot override insufficient native stroke evidence."""
    native = cloud_region.get("source_pixel_review") or {}
    candidates = [
        line["value"]
        for line in cloud_region.get("lines", [])
        if re.fullmatch(r"[+-]?\d+(?:[.,]\d+)?", str(line.get("value", "")))
    ]
    if not candidates:
        return None
    digit_count = min(sum(char.isdigit() for char in value) for value in candidates)
    if native.get("group_count", 0) > digit_count:
        return "incomplete_cloud_readout"
    ambiguous = native.get("low_resolution_collapsed_row") or (
        native.get("below_resolution_gate")
        and (native.get("merged_glyphs_possible") or native.get("possible_decimal_point"))
    )
    if ambiguous and (not initial_values or initial_values != cloud_values):
        return "ambiguous_source_pixels"
    return None


def _merge(document, reviewed, local_ready):
    """Replace eligible fields within this photograph; failures keep the initial view."""
    fallback = reviewed.get("fallback", {})

    def values(doc):
        return {
            (
                r.get("panel_id"),
                Decimal(r["value"]),
                (r.get("normalized_value") or {}).get("unit") or r.get("unit"),
            )
            for r in build_readings(doc)
            if not r.get("quality_issue") and r.get("value") is not None
        }

    before = values(document)
    cloud_values = values(reviewed)
    document["parallel_review"].pop("conflicts", None)
    if local_ready:
        document["parallel_review"].pop("crosscheck_pending", None)

    def conflict_for(panel_id):
        initial_values = [(value, unit) for pid, value, unit in before if pid == panel_id]
        reviewed_values = [(value, unit) for pid, value, unit in cloud_values if pid == panel_id]
        if len(initial_values) != 1 or len(reviewed_values) != 1:
            return None
        (initial_value, initial_unit), (reviewed_value, reviewed_unit) = (
            initial_values[0],
            reviewed_values[0],
        )
        if initial_unit and reviewed_unit and initial_unit != reviewed_unit:
            return None  # Existing field/unit validation handles that conflict.
        reason = numeric_review_conflict(initial_value, reviewed_value)
        if reason:
            conflict = {
                "panel_id": panel_id,
                "reason": reason,
                "initial_value": str(initial_value),
                "reviewed_value": str(reviewed_value),
            }
            document["parallel_review"].setdefault("conflicts", []).append(conflict)
            return conflict
        return None

    def source_conflict_for(region):
        panel_id = region["panel_id"]
        initial = {(value, unit) for pid, value, unit in before if pid == panel_id}
        cloud = {(value, unit) for pid, value, unit in cloud_values if pid == panel_id}
        reason = _source_conflict(region, initial, cloud)
        if reason:
            conflict = {
                "panel_id": panel_id,
                "reason": reason,
                "initial_values": sorted(str(value) for value, _ in initial),
                "reviewed_values": sorted(str(value) for value, _ in cloud),
            }
            document["parallel_review"].setdefault("conflicts", []).append(conflict)
            return conflict
        return None

    document["fallback"] = copy.deepcopy(fallback)
    cloud_timing = reviewed.get("timing", {}).get("monotonic_durations_ms", {})
    document.setdefault("timing", {}).setdefault("monotonic_durations_ms", {}).update(cloud_timing)
    states = []
    if reviewed.get("panel_regions"):
        current = {r["panel_id"]: r for r in document.get("panel_regions", [])}
        merged = []
        for cloud_region in reviewed["panel_regions"]:
            initial = current.get(cloud_region["panel_id"])
            region = copy.deepcopy(initial or cloud_region)
            if initial is None:
                region.update(lines=[], display_state=None)
            attempt = cloud_region.get("fallback", {})
            region["fallback"] = copy.deepcopy(attempt)
            if "source_pixel_review" in cloud_region:
                region["source_pixel_review"] = copy.deepcopy(cloud_region["source_pixel_review"])
            if attempt.get("status") != "completed":
                states.append("failed")
            elif initial and (
                initial.get("measurement_name") != cloud_region.get("measurement_name")
                or initial.get("display_state_rejection")
                or (initial.get("local_ocr") or {}).get("display_state_rejection")
            ):
                states.append("conflict")
            elif any(
                line.get("quality_issue")
                or line.get("quality_issues")
                or line.get("label") not in (None, region.get("measurement_name"))
                for line in cloud_region.get("lines", [])
            ):
                states.append("conflict")
            elif conflict := conflict_for(cloud_region["panel_id"]):
                region["review_conflict"] = conflict
                states.append("conflict")
            elif conflict := source_conflict_for(cloud_region):
                region["review_conflict"] = conflict
                if conflict["reason"] == "incomplete_cloud_readout" and all(
                    sum(char.isdigit() for char in value)
                    < cloud_region["source_pixel_review"]["group_count"]
                    for value in conflict["initial_values"]
                ):
                    region["lines"] = [
                        dict(line, quality_issue="incomplete_source_readout")
                        for line in region.get("lines", [])
                    ]
                states.append("conflict")
            else:
                region["lines"] = copy.deepcopy(cloud_region.get("lines", []))
                region["display_state"] = copy.deepcopy(cloud_region.get("display_state"))
                region.pop("display_state_rejection", None)
                states.append(
                    "completed" if region["lines"] or region["display_state"] else "unreadable"
                )
            merged.append(region)
        document["panel_regions"] = merged
        document["lines"] = [line for region in merged for line in region.get("lines", [])]
        if not current:
            # These facts come exclusively from shared localization and the task's
            # capture-time bindings, never from the cloud's transcribed identity.
            for key in (
                "instrument",
                "binding_id",
                "binding_ids",
                "instrument_candidates",
                "instrument_association",
                "instrument_identity_basis",
                "association_status",
                "workbench",
                "workbenches",
                "panel_detection",
                "photo_orientation",
                "panel_selection",
            ):
                if key in reviewed:
                    document[key] = copy.deepcopy(reviewed[key])
    elif fallback.get("status") == "completed":
        if conflict_for(None) or any(
            line.get("quality_issue") or line.get("quality_issues")
            for line in reviewed.get("lines", [])
        ):
            states.append("conflict")
        else:
            if document.get("panel_detection"):
                # Whole-photo transcription supplies candidates, not a detected
                # panel. Keep the failed localized branch as evidence instead of
                # applying its region-only filter to the independent answer.
                document["localized_attempt"] = {
                    key: copy.deepcopy(document[key])
                    for key in (
                        "panel_detection",
                        "panel_regions",
                        "local_ocr",
                        "recognition_skipped",
                        "skip_reason",
                        "failure_reason",
                        "error",
                    )
                    if key in document
                }
                document["whole_photo_recovery"] = {
                    "status": "completed",
                    "reason": "no_localized_readings",
                    "association": "unbound_candidates",
                }
                document.pop("panel_detection")
            for key in (
                "lines",
                "model",
                "device",
                "actual_model_invocation",
                "instrument",
                "binding_id",
                "binding_ids",
                "instrument_candidates",
                "panel_regions",
                "instrument_association",
                "instrument_identity_basis",
                "association_status",
                "outcome",
                "vision_display_rules",
            ):
                if key in reviewed:
                    document[key] = copy.deepcopy(reviewed[key])
            states.append("completed" if document.get("lines") else "unreadable")
    else:
        states.append("failed")
    if not local_ready and not document.get("local_ocr"):
        document["local_ocr"] = {"status": "running", "lines": []}
    status = next((s for s in ("conflict", "failed", "unreadable") if s in states), "completed")
    document["parallel_review"].update(
        status=status, finished_at=reviewed.get("fallback", {}).get("finished_at")
    )
    accepted = any(s in {"completed", "unreadable"} for s in states)
    if accepted:
        document["parallel_review"]["result"] = (
            "unverified"
            if document.get("whole_photo_recovery", {}).get("status") == "completed"
            else "consistent"
            if before == values(document)
            else "updated"
        )
        if document.get("whole_photo_recovery", {}).get("status") == "completed":
            document["parallel_review"]["reason"] = "local_panel_not_localized"
        for key in ("recognition_skipped", "skip_reason", "failure_reason", "error"):
            document.pop(key, None)
        # Explicit task outcome overrides the nested local branch's skip while
        # retaining that branch unchanged for diagnosis and old receipt readers.
        document["recognition_skipped"] = False
        document["outcome"] = (
            "numeric_readout"
            if build_readings(document)
            else "display_state"
            if any(r.get("display_state") for r in document.get("panel_regions", []))
            else "no_numeric_readout"
        )
    document["actual_model_invocation"] = bool(
        document.get("actual_model_invocation")
        or fallback.get("actual_model_invocation")
        or any(r.get("actual_model_invocation") for r in fallback.get("regions", []))
    )
    return accepted


def run(core, document, image, panel, x, y, *, pipeline, valid, clock, pause, started):
    from field_recognition.recognition.panel_readout import submit_local

    config = pipeline.recognition or RecognitionConfig.from_environment()
    # A single immutable config snapshot is used by both threads. The background
    # reader never saves documents or touches the owner's task lease.
    local_pipeline = replace(pipeline, recognition=replace(config, enabled=False))
    cloud_pipeline = replace(
        pipeline, save_document=lambda _: None, recognition=replace(config, photo_mode="primary")
    )
    snapshots = document.get("all_binding_snapshots", document.get("binding_snapshots", []))
    template = (
        core["predict_readout"](
            panel,
            x,
            y,
            binding_snapshots=snapshots,
            field_rules=document.get("field_rules"),
            localization_only=True,
        )
        if snapshots
        else {"panel_regions": []}
    )
    document.update(
        reading_mode="parallel_photo",
        phase="parallel_reading",
        result_available=document.get("result_available", False),
        result_revision=document.get("result_revision", 0),
        parallel_review={"status": "running", "started_at": core["now"]()},
    )
    pipeline.save_document(document)
    local_future = submit_local(
        core, document, panel, x, y, clock=clock, detection=template.get("panel_detection")
    )
    cloud_future = core["vision_pool"].submit(
        _cloud,
        core,
        copy.deepcopy(document),
        template,
        image,
        panel,
        cloud_pipeline,
        valid,
        time.monotonic(),
    )
    local_ready = False
    reviewed = None
    cloud_published = False
    while valid():
        if local_future.done() and not local_ready:
            local_ready = True
            try:
                local = local_future.result()
            except Exception as exc:
                local = {"status": "failed", "lines": [], "error": type(exc).__name__}
            _project(core, document, local, image, local_pipeline, valid, clock, started)
            document.update(
                status="running",
                phase="review_pending",
                result_available=bool(
                    document.get("lines")
                    or any(r.get("display_state") for r in document.get("panel_regions", []))
                ),
                result_stage="preliminary",
                result_updated_at=core["now"](),
            )
            if document["result_available"] and reviewed is None:
                document.setdefault("first_result_at", document["result_updated_at"])
                document["result_revision"] += 1
            document["readings"] = build_readings(document)
            document["measurement_records"] = build_records(document)
            document["display_fields"] = display_fields(document)
            if reviewed is None:
                pipeline.save_document(document)
        if cloud_future.done() and reviewed is None:
            reviewed = cloud_future.result()
        if reviewed is not None and (local_ready or not cloud_published):
            accepted = _merge(document, reviewed, local_ready)
            if local_ready:
                document.update(
                    status="failed"
                    if not accepted and document.get("local_ocr", {}).get("status") == "failed"
                    else "completed",
                    phase="completed",
                    result_available=bool(
                        document.get("lines")
                        or any(r.get("display_state") for r in document.get("panel_regions", []))
                    ),
                    result_stage="reviewed" if accepted else "preliminary",
                    result_updated_at=core["now"](),
                )
                document.setdefault("first_result_at", document["result_updated_at"])
                document["result_revision"] += 1
                if document["status"] == "completed":
                    document.pop("error", None)
                return local_future
            # Both readers started together. An early cloud candidate remains
            # provisional until the local result can participate in comparison.
            document.update(
                status="running",
                phase="local_crosscheck_pending",
                result_available=bool(
                    document.get("lines")
                    or any(r.get("display_state") for r in document.get("panel_regions", []))
                ),
                result_stage="preliminary",
                result_updated_at=core["now"](),
            )
            document["parallel_review"].update(status="running", crosscheck_pending=True)
            if document["result_available"]:
                document.setdefault("first_result_at", document["result_updated_at"])
                document["result_revision"] += 1
            document["readings"] = build_readings(document)
            document["measurement_records"] = build_records(document)
            document["display_fields"] = display_fields(document)
            pipeline.save_document(document)
            cloud_published = True
        if reviewed is not None and clock() - started >= 90:
            document.update(
                status="completed" if document.get("result_available") else "failed",
                phase="completed",
                error="local_ocr_timeout",
            )
            document["parallel_review"].update(status="failed", reason="local_crosscheck_timeout")
            return local_future
        pause(0.02)
    document.update(
        status="cancelled",
        phase="completed",
        error="binding_ended_or_changed",
        lines=[],
        result_available=False,
    )
    if not cloud_future.cancel():
        # An in-flight paid request cannot be cancelled safely. Keep its raw
        # receipt before the owner publishes cancellation/interruption; do not
        # let that answer restore a released binding or a cancelled reading.
        reviewed = cloud_future.result()
        document["fallback"] = copy.deepcopy(reviewed.get("fallback", {}))
    document["parallel_review"].update(status="cancelled", finished_at=core["now"]())
    return local_future
