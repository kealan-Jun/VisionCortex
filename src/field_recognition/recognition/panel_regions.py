"""Per-panel OCR, immutable crop evidence and capture-time instrument attribution."""

import copy
import hashlib
import time
import uuid
import re

import cv2
import numpy as np
from field_recognition.recognition.aliyun_vision import answer_lines
from field_recognition.measurements.readout_rules import NUMBER, READOUT, fallback_reason
from field_recognition.recognition.pipeline_context import ReadoutPipeline
from field_recognition.recognition.digit_regions import refine_digits


def _orient_pair_roles(regions):
    """Use both source OFF rows to establish a device direction, never a number."""
    groups = {}
    for region in regions:
        if region["class_id"] == 0:
            groups.setdefault(region["instrument_id"], []).append(region)
    evidence = []
    for pair in groups.values():
        if len(pair) != 2:
            continue
        directions = [r["local_ocr"].get("red_display_orientation", {}) for r in pair]
        degrees = [
            d.get("orientation_degrees") if d.get("status") in {"consistent", "ocr_only"} else None
            for d in directions
        ]
        if 180 not in degrees:
            continue
        item = {
            "panel_ids": [r["panel_id"] for r in pair],
            "basis": "whole_pair_native_off_bars_and_ocr_views",
            "status": "unconfirmed",
        }
        geometries = [r["local_ocr"].get("red_display_geometry", {}) for r in pair]
        if (
            degrees == [180, 180]
            and all(d.get("status") == "consistent" for d in directions)
            and all(r["measurement_name"] for r in pair)
        ):
            rolls = [g["roll_degrees"] for g in geometries]
            centers = [np.asarray(g["polygon"]).mean(axis=0) for g in geometries]
            radians = np.deg2rad(sum(rolls) / 2)
            normal = np.asarray([-np.sin(radians), np.cos(radians)])
            # Both rows must occupy one horizontal physical pair in source
            # coordinates. Opposite/missing direction and separate rows abstain.
            if abs(rolls[0] - rolls[1]) <= 12 and abs(
                np.dot(centers[1] - centers[0], normal)
            ) <= min(g["native_size"][1] for g in geometries):
                pair[0]["measurement_name"], pair[1]["measurement_name"] = (
                    pair[1]["measurement_name"],
                    pair[0]["measurement_name"],
                )
                item.update(
                    status="consistent",
                    orientation_degrees=180,
                    supporting_views=[r["panel_id"] for r in pair],
                )
        if item["status"] != "consistent":
            for region in pair:
                region["measurement_name"] = None
        for region in pair:
            region["source_orientation"] = copy.deepcopy(item)
        evidence.append(item)
    return evidence


def _prepare_detection(core, detection, allowed, started, begin):
    from field_recognition.recognition.panel_layout import resolve_type_boxes

    detection["boxes"] = resolve_type_boxes(
        detection["boxes"], allowed, core.get("get_instrument", lambda _: {})
    )
    detection.update(started_at=started, finished_at=core["now"]())
    detection["allowed_instrument_ids"] = sorted(allowed)
    detection["skipped_unbound_panels"] = [
        b for b in detection["boxes"] if b["instrument_id"] not in allowed
    ]
    diagnostics = detection.setdefault("diagnostics", {})
    invoked = detection["status"] not in {"waiting_binding", "failed"}
    diagnostics.setdefault("raw_candidate_count", len(detection["boxes"]) if invoked else None)
    diagnostics.setdefault(
        "raw_count_basis",
        "detector_returned_boxes"
        if invoked
        else "detector_not_run"
        if not allowed
        else "detector_failed",
    )
    diagnostics.setdefault("quality_filtered_count", len(detection["boxes"]) if invoked else None)
    diagnostics.setdefault("excluded_candidates", []).extend(
        dict(b, reason=b.get("association_issue") or "target_not_allowed")
        for b in detection["skipped_unbound_panels"]
    )
    diagnostics["binding_matched_count"] = len(detection["boxes"]) - len(
        detection["skipped_unbound_panels"]
    )
    detection["duration_ms"] = round((time.monotonic() - begin) * 1000, 3)


def predict(
    core,
    image,
    x=0,
    y=0,
    *,
    video=False,
    allowed_instrument_ids=(),
    localization_only=False,
    detection=None,
):
    detector = core["panel_detector"]
    if localization_only and (not detector.enabled() or not allowed_instrument_ids):
        return {"status": "completed", "lines": [], "panel_regions": []}
    if not detector.enabled():
        return core["predict_panel"](image, x, y)
    started = core["now"]()
    begin = time.monotonic()
    allowed = set(allowed_instrument_ids)
    from field_recognition.recognition.photo_orientation import Rotation, recover, restore

    orientation = (detection or {}).get("photo_orientation") or {}
    if orientation.get("status") == "recovered":
        degree = orientation["degrees_ccw"]
        if (
            video
            or degree not in (90, 180, 270)
            or orientation["source_size"] != [image.shape[1], image.shape[0]]
            or orientation["source_offset"] != [x, y]
            or set(orientation["oriented_detection"].get("allowed_instrument_ids", [])) != allowed
            or hashlib.sha256(image.tobytes()).hexdigest() != orientation["source_pixels_sha256"]
        ):
            raise ValueError("Prepared photo orientation does not match the source input")
        rotation = Rotation(degree, image.shape[1], image.shape[0], x, y)
        result = predict(
            core,
            rotation.image(image),
            allowed_instrument_ids=allowed,
            localization_only=localization_only,
            detection=orientation["oriented_detection"],
        )
        proof = {k: v for k, v in orientation.items() if k != "oriented_detection"}
        return restore(result, rotation, proof)
    # A saved photo remains readable without an instrument identity. Keep its
    # raw OCR separate from detector-bound measurements; video still requires
    # an active binding before spending inference on background frames.
    if not allowed and not video:
        result = dict(core["predict_panel"](image, x, y))
        result.update(
            panel_regions=[],
            panel_selection="unbound_photo_text",
            instrument_identity_basis="unconfirmed",
        )
        return result
    prepared_detection = detection is not None
    detection = (
        copy.deepcopy(detection)
        if prepared_detection
        else (
            detector.predict(image, photo=True)
            if allowed and not video
            else detector.predict(image)
            if allowed
            else {"status": "waiting_binding", "boxes": [], "model": "YOLO11n"}
        )
    )
    from field_recognition.recognition.panel_layout import roles_for_boxes

    if not prepared_detection:
        _prepare_detection(core, detection, allowed, started, begin)
        if not video and allowed and detection["status"] == "completed" and not detection["boxes"]:
            rotation, rotated_detection, proof = recover(core, image, allowed)
            proof["original_detection"] = copy.deepcopy(detection)
            if rotation:
                _prepare_detection(core, rotated_detection, allowed, started, begin)
                rotation = Rotation(rotation.degrees, image.shape[1], image.shape[0], x, y)
                result = predict(
                    core,
                    rotation.image(image),
                    allowed_instrument_ids=allowed,
                    localization_only=localization_only,
                    detection=rotated_detection,
                )
                return restore(result, rotation, proof)
            detection["photo_orientation"] = proof
    diagnostics = detection.setdefault("diagnostics", {})
    ocr_started = time.monotonic()
    model_calls = []

    def predict_panel(*args, **kwargs):
        # Refinement can invoke OCR several times. Collect each adapter call once,
        # rather than adding nested panel_ocr/digit_ocr copies of the same receipt.
        result = core["predict_panel"](*args, **kwargs)
        model_calls.append(result)
        return result

    roles = roles_for_boxes(detection["boxes"])
    regions, lines, ordinals = [], [], {}
    for box in detection["boxes"]:
        if box["instrument_id"] not in allowed:
            continue
        ordinal = ordinals.get(box["class_id"], 0)
        ordinals[box["class_id"]] = ordinal + 1
        x1, y1, x2, y2 = box["xyxy"]
        # Include a small border for text detection, using original pixels for OCR.
        pad = max(2, round(min(x2 - x1, y2 - y1) * 0.08))
        cx, cy = max(0, x1 - pad), max(0, y1 - pad)
        ex, ey = min(image.shape[1], x2 + pad), min(image.shape[0], y2 + pad)
        panel = image[cy:ey, cx:ex].copy()
        crop_issue = None
        if (
            not video
            and box["class_id"] == 0
            and box.get("localization_method") == "photo_weak_pair_context_v1"
        ):
            from field_recognition.recognition.panel_recovery import weak_crop_boundary_issue

            crop_issue = weak_crop_boundary_issue(image, box, [cx, cy, ex, ey])
        local = (
            {
                "status": "completed",
                "lines": [],
                "actual_model_invocation": False,
                "recognition_skipped": True,
                "failure_reason": "source_crop_truncated",
                "source_crop_quality": crop_issue,
                "ocr_model_call_count": 0,
            }
            if crop_issue
            else {"status": "running", "lines": [], "actual_model_invocation": False}
            if localization_only
            else predict_panel(panel, x + cx, y + cy)
        )
        if not localization_only and not crop_issue and box["class_id"] == 1 and not video:
            from field_recognition.recognition.lcd_digits import refine_lcd

            instrument = core.get("get_instrument", lambda _: {})(box["instrument_id"]) or {}
            local = refine_lcd(
                predict_panel,
                panel,
                local,
                x + cx,
                y + cy,
                source_image=image,
                source_offset=(x, y),
                instrument=instrument,
            )
        elif not localization_only and not crop_issue:
            # A syntactically numeric OCR token (e.g. "5G") can still violate
            # the registered field rule. Retry its source pixels, not a guessed
            # character substitution, while preserving the untouched receipt.
            raw_checked = (
                validate_lines(
                    copy.deepcopy(local.get("lines", [])),
                    roles.get(
                        ((str(box["class_id"]), box.get("instrument_id")), tuple(box["xyxy"]))
                    ),
                    core.get("get_instrument", lambda _: {})(box["instrument_id"]) or {},
                )
                if box["class_id"] == 0 and not video
                else []
            )
            local = refine_digits(
                predict_panel,
                panel,
                local,
                x + cx,
                y + cy,
                source_image=image,
                source_offset=(x, y),
                recover_red=box["class_id"] == 0 and not video,
                red_readout_rejected=any(line.get("quality_issue") for line in raw_checked)
                and not any(
                    READOUT.fullmatch(line.get("text", "")) and not line.get("quality_issue")
                    for line in raw_checked
                ),
                require_native_conflict_confirmation=box["class_id"] == 0 and not video,
            )
        if not localization_only and not crop_issue and box["class_id"] == 0:
            from field_recognition.recognition.led_digits import verify_digits

            local = verify_digits(panel, local, x + cx, y + cy)
            raw_lines = local.get("panel_ocr", {}).get("lines", [])
            try:
                raw_confidence = float(raw_lines[0].get("confidence") or 0) if raw_lines else 0
            except (TypeError, ValueError):
                raw_confidence = 0
            if (
                not local.get("lines")
                and not local.get("display_state")
                and not local.get("display_state_rejection")
                and len(raw_lines) == 1
                and raw_lines[0].get("text", "").strip().upper() == "OFF"
                and np.isfinite(raw_confidence)
                and raw_confidence >= 0.6
            ):
                local["display_state"] = {
                    "text": "OFF",
                    "basis": "recognized_display_text",
                    "raw_text": raw_lines[0]["text"],
                    "source": "local_ocr.panel_ocr",
                    "confidence": raw_lines[0]["confidence"],
                }
        region = {
            "panel_id": f"panel-{box['class_id']}-{ordinal}",
            "class_id": box["class_id"],
            "measurement_name": roles.get(
                ((str(box["class_id"]), box.get("instrument_id")), tuple(box["xyxy"]))
            ),
            "instrument_id": box["instrument_id"],
            "detector_confidence": box["confidence"],
            "identity_evidence": box.get("identity_evidence")
            or {
                "basis": "detector_manifest_asset_mapping",
                "class_id": box["class_id"],
                "weights_sha256": detection.get("weights_sha256"),
                "model_version": detection.get("model_version"),
            },
            "localization_method": box.get("localization_method", "original"),
            "localization_supporting_views": box.get("supporting_views", []),
            "bbox": [x + x1, y + y1, x + x2, y + y2],
            "crop": [x + cx, y + cy, ex - cx, ey - cy],
            "local_ocr": local,
            "digit_region": local.get("digit_region"),
            "display_state": local.get("display_state"),
        }
        regions.append(region)
    if not video and not localization_only:
        orientations = _orient_pair_roles(regions)
        if orientations:
            diagnostics["source_orientations"] = orientations
    for region in regions:
        local = region["local_ocr"]
        local["lines"] = validate_lines(
            [dict(line, panel_id=region["panel_id"]) for line in local.get("lines", [])],
            region["measurement_name"],
            core.get("get_instrument", lambda _: {})(region["instrument_id"]) or {},
        )
        lines.extend(local["lines"])
    diagnostics["ocr_input_region_count"] = len(regions)
    ocr_elapsed_ms = round((time.monotonic() - ocr_started) * 1000, 3) if regions else None
    result = {
        "status": "completed",
        "lines": lines,
        "panel_regions": regions,
        "panel_detection": detection,
        "model": core["ocr_state"]["engine"],
        "device": core["OCR_DEVICE"],
        "actual_model_invocation": any(
            r["local_ocr"].get("actual_model_invocation") for r in regions
        ),
        "ocr_started_at": regions[0]["local_ocr"].get("ocr_started_at", detection["finished_at"])
        if regions
        else detection["finished_at"],
        "ocr_finished_at": core["now"](),
        "ocr_elapsed_ms": ocr_elapsed_ms,
        "wall_seconds": round(time.monotonic() - begin, 3),
        "panel_selection": "detected_display_regions",
    }
    result["ocr_model_call_count"] = (
        sum(call["ocr_model_call_count"] for call in model_calls)
        if all(call.get("ocr_model_call_count") is not None for call in model_calls)
        else None
    )
    for name in ("ocr_model_lock_wait_ms", "ocr_model_prepare_ms", "ocr_model_call_ms"):
        values = [call.get(name) for call in model_calls]
        result[name] = round(sum(values), 3) if all(value is not None for value in values) else None
    result["ocr_model_call_scopes"] = sorted(
        {call["ocr_model_call_scope"] for call in model_calls if call.get("ocr_model_call_scope")}
    )
    if detection.get("photo_orientation"):
        result["photo_orientation"] = copy.deepcopy(detection["photo_orientation"])
        from field_recognition.recognition.photo_orientation import record_calls

        record_calls(result, detection["photo_orientation"])
    if not regions:
        result.update(
            panel_selection="no_bound_panel",
            recognition_skipped=True,
            skip_reason="no_active_instrument_binding" if not allowed else "no_visible_bound_panel",
        )
        result["failure_reason"] = (
            "no_active_instrument_binding"
            if not allowed
            else "panel_orientation_unconfirmed"
            if any(
                attempt["authorized_box_count"]
                for attempt in (detection.get("photo_orientation") or {}).get("attempts", [])
            )
            else "instrument_identity_unconfirmed"
            if any(
                b.get("association_issue") == "instance_evidence_required"
                for b in detection["boxes"]
            )
            else "target_not_allowed"
            if detection["skipped_unbound_panels"]
            else "panel_candidates_filtered"
            if diagnostics.get("raw_candidate_count")
            else "panel_not_detected"
        )
    if detection["status"] == "failed":
        result["detector_error"] = detection["error"]
        result.update(
            status="failed", error="panel_detection_failed", failure_reason="processing_exception"
        )
    return result


def reading_fallback_reason(lines, error=None):
    checked = []
    for line in lines:
        if (
            (line.get("normalization") or {}).get("method")
            in {"registered_fixed_decimal_v1", "registered_fixed_decimal_v2"}
            and line.get("value") is not None
            and not line.get("quality_issue")
        ):
            checked.append(dict(line, text=str(line["value"])))
        else:
            checked.append(line)
    return fallback_reason(checked, error)


def needs_fallback(local):
    if local.get("panel_regions"):
        if any(
            l.get("quality_issue")
            for r in local["panel_regions"]
            for l in r["local_ocr"].get("lines", [])
        ):
            return "field_rule_conflict"
        return next(
            (
                reason
                for r in local["panel_regions"]
                if not r.get("display_state")
                if (r["local_ocr"].get("source_crop_quality") or {}).get("status") != "rejected"
                if (
                    reason := reading_fallback_reason(
                        r["local_ocr"].get("lines", []), r["local_ocr"].get("error")
                    )
                )
            ),
            None,
        )
    return fallback_reason(local.get("lines", []), local.get("error"))


def finish(
    core,
    document,
    local,
    image,
    *,
    valid,
    pipeline: ReadoutPipeline,
    clock=time.monotonic,
    started=0,
):
    document.update(copy.deepcopy(local))
    # A completed model call is not a completed business task. Keep the lease
    # while saving crop evidence and waiting for the optional provider response.
    document["status"] = "running"
    document["local_ocr"] = copy.deepcopy(local)
    document["capture_association"] = {
        k: document.get(k)
        for k in ("instrument", "binding_id", "instrument_candidates", "workbench")
    }
    snapshots = document.get("all_binding_snapshots", document.get("binding_snapshots", []))
    candidates, benches, combined, attempts = {}, {}, [], []
    for region in document["panel_regions"]:
        binding = next(
            (b for b in snapshots if b["instrument"]["id"] == region["instrument_id"]), None
        )
        if binding is None:
            raise ValueError("panel has no capture-time instrument binding")
        region.update(
            instrument=binding["instrument"],
            binding_id=binding["binding_id"],
            association_basis="panel_detector_session_binding",
        )
        if region["instrument_id"] in document.get("field_rules", {}):
            region["instrument"] = dict(
                region["instrument"],
                measurement_ranges=document["field_rules"][region["instrument_id"]],
            )
        instrument = region["instrument"]
        if instrument:
            with core["db"]() as conn:
                scene = conn.execute(
                    "SELECT * FROM scenes WHERE name=?", (instrument.get("scene"),)
                ).fetchone()
            bench = {
                "id": scene["id"] if scene else None,
                "name": instrument.get("scene"),
                "basis": "instrument_registration",
            }
            region["workbench"] = bench
            if bench["name"]:
                benches[bench["name"]] = bench
            candidates[instrument["id"]] = {
                "instrument": instrument,
                "binding_id": region["binding_id"],
                "basis": region["association_basis"],
            }
        from field_recognition.recognition.photo_orientation import source_panel

        crop = source_panel(image, region, digit_crop=True)
        ident = str(uuid.uuid4())
        path = core["DATA"] / "Images" / f"{ident}.png"
        if not cv2.imwrite(str(path), crop):
            raise ValueError("panel evidence encoding failed")
        region.update(
            evidence_id=ident,
            image_url=f"/api/images/{ident}",
            image_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        result = region["local_ocr"]
        crop_rejected = (result.get("source_crop_quality") or {}).get("status") == "rejected"
        lines = copy.deepcopy(result.get("lines", []))
        lines = validate_lines(lines, region.get("measurement_name"), region["instrument"])
        reason = (
            None
            if region.get("display_state")
            else "field_rule_conflict"
            if any(l.get("quality_issue") for l in lines)
            else reading_fallback_reason(lines, result.get("error"))
        )
        primary = pipeline.primary_photos() and document.get("request_trigger") != "video_stream"
        if primary and not region.get("display_state"):
            reason = "primary_photo"
        if crop_rejected:
            region["fallback"] = {
                "panel_id": region["panel_id"],
                "status": "skipped",
                "reason": "source_crop_truncated",
                "queue_wait_ms": 0,
            }
            attempts.append(region["fallback"])
        if not crop_rejected and reason and pipeline.public_config()["available"] and valid():
            attempt = {"panel_id": region["panel_id"], "trigger": reason}
            slot_started = clock()
            acquired = (
                pipeline.acquire_primary_slot(valid, clock=clock)
                if primary
                else core["vision_call_lock"].acquire(blocking=False)
            )
            attempt.update(
                queue_wait_ms=round((clock() - slot_started) * 1000, 3),
                queue_wait_clock_basis="process_monotonic",
                queue_wait_scope="panel_provider_slot_acquisition",
            )
            if not acquired:
                attempt.update(status="skipped", reason="busy")
            else:
                try:
                    scope = f"panel:{document['camera_id']}:{region['instrument_id']}:{region['panel_id']}"
                    if not primary and not core["reserve_fallback"](scope):
                        attempt.update(status="skipped", reason="cooldown")
                    elif valid():
                        attempt.update(
                            status="running",
                            attempted_at=core["now"](),
                            trigger_elapsed_seconds=round(clock() - started, 3),
                        )
                        region["fallback"] = attempt
                        document.update(phase="extended_reading")
                        pipeline.save_document(document)
                        provider_started = clock()
                        try:
                            name = region.get("measurement_name")
                            registered = (region["instrument"].get("measurement_ranges") or {}).get(
                                name
                            ) or {}
                            display_rules = (
                                [
                                    {
                                        "measurement_name": name,
                                        **{
                                            key: registered[key]
                                            for key in (
                                                "unit",
                                                "decimal_places",
                                                "fixed_decimal_display",
                                            )
                                            if key in registered
                                        },
                                    }
                                ]
                                if primary and registered.get("fixed_decimal_display") is True
                                else None
                            )
                            cloud = pipeline.read_panel(crop, display_rules=display_rules)
                        finally:
                            attempt.update(
                                duration_ms=round((clock() - provider_started) * 1000, 3),
                                clock_basis="process_monotonic",
                            )
                        attempt.update(cloud, finished_at=core["now"]())
                        answer = cloud.get("answer", {})
                        if answer.get("readings") or answer.get("display_states"):
                            lines = answer_lines(answer)
                            states = answer.get("display_states", [])
                            if states:
                                # Only a single isolated field under the existing task
                                # snapshot may acquire a formal state. Multi-screen or
                                # conflicting outputs remain reviewable raw evidence.
                                if (
                                    len(states) == 1
                                    and not answer.get("readings")
                                    and name
                                    and states[0].get("label") in (None, name)
                                    and not result.get("display_state_rejection")
                                    and not region.get("display_state_rejection")
                                ):
                                    region["display_state"] = {
                                        "text": "OFF",
                                        "raw_text": states[0]["text"],
                                        "basis": "vision_transcription_of_localized_field",
                                        "source": "fallback.answer.display_states",
                                        "human_verified": False,
                                    }
                                else:
                                    region["display_state_rejection"] = {
                                        "basis": "ambiguous_or_conflicting_vision_state"
                                    }
                                    lines = []
                            if primary and display_rules:
                                from field_recognition.measurements.measurement_records import (
                                    apply_fixed_decimal,
                                )

                                for line in lines:
                                    if line.get("display_state"):
                                        continue
                                    normalized = apply_fixed_decimal(
                                        name, dict(line, text=line["value"]), region["instrument"]
                                    )
                                    if normalized.get("normalization"):
                                        line.update(
                                            value=normalized["value"],
                                            raw_value=line["value"],
                                            normalization=dict(
                                                normalized["normalization"], raw_text=line["text"]
                                            ),
                                            numeric_candidates=[normalized["value"]],
                                        )
                        elif primary and cloud.get("status") == "completed":
                            # A model-declared unreadable panel must not be replaced by
                            # the very OCR candidate the primary mode was meant to review.
                            lines = []
                finally:
                    core["vision_call_lock"].release()
            region["fallback"] = attempt
            attempts.append(attempt)
        lines = validate_lines(lines, region.get("measurement_name"), region["instrument"])
        region["lines"] = [dict(line, panel_id=region["panel_id"]) for line in lines]
        combined.extend(region["lines"])
    if not valid():
        document.update(status="cancelled", error="binding_ended_or_changed", lines=[])
        return
    values = list(candidates.values())
    unique = values[0] if len(values) == 1 else None
    document.update(
        lines=combined,
        status="completed",
        phase="completed",
        instrument=unique["instrument"] if unique else None,
        binding_id=unique["binding_id"] if unique else None,
        instrument_candidates=values,
        binding_ids=[v["binding_id"] for v in values if v["binding_id"]],
        instrument_association="panel_detector",
        instrument_identity_basis="panel_model_class",
        association_status="localized_panels",
        workbenches=list(benches.values()),
        workbench=next(iter(benches.values())) if len(benches) == 1 else None,
        outcome="numeric_readout"
        if any(READOUT.fullmatch(l.get("text", "")) or l.get("value") for l in combined)
        else "display_state"
        if any(r.get("display_state") for r in document["panel_regions"])
        else "no_numeric_readout",
    )
    if all(
        r["local_ocr"].get("status") == "failed"
        and not r.get("fallback", {}).get("answer", {}).get("readings")
        and not r.get("fallback", {}).get("answer", {}).get("display_states")
        for r in document["panel_regions"]
    ):
        document.update(status="failed", error="panel_ocr_failed")
    if attempts:
        document["fallback"] = dict(
            attempts[-1],
            regions=attempts,
            queue_wait_ms=sum(a["queue_wait_ms"] for a in attempts),
            queue_wait_scope="sum_of_this_task_panel_provider_slot_acquisitions",
            duration_ms=sum(a["duration_ms"] for a in attempts if a.get("duration_ms") is not None)
            if any(a.get("duration_ms") is not None for a in attempts)
            else None,
            attempted_at=next((a["attempted_at"] for a in attempts if a.get("attempted_at")), None),
            finished_at=next(
                (a["finished_at"] for a in reversed(attempts) if a.get("finished_at")), None
            ),
        )


def validate_lines(lines, name, instrument):
    from field_recognition.measurements.measurement_records import (
        field_issues,
        measurement_value,
        apply_fixed_decimal,
    )

    if not name:
        return lines
    result = []
    for line in lines:
        line = apply_fixed_decimal(name, line, instrument)
        match = re.match(r"\s*(" + NUMBER + ")", line.get("text", ""))
        if not match and line.get("value") is None:
            result.append(line)
            continue
        unit = (
            line.get("unit")
            if line.get("reading_source") == "vision"
            else line.get("unit")
            or (line["text"][match.end() :].strip() if match else None)
            or None
        )
        raw = dict(
            line, value=str(line["value"]) if line.get("value") is not None else match[1], unit=unit
        )
        issues = field_issues(name, raw, instrument)
        result.append(
            dict(
                line,
                quality_issue=next(iter(issues), None),
                measurement_name=name,
                normalized_value=measurement_value(name, raw, instrument)[0],
            )
        )
    return result
