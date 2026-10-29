"""Separate numeric regions with explicit, reviewable identity ambiguity."""

import re
from field_recognition.measurements.readout_rules import READOUT, NUMBER


def result_ready(document):
    """A durable preliminary projection is readable while its review owns the lease."""
    return document.get("status") == "completed" or (
        document.get("status") == "running"
        and document.get("reading_mode") == "parallel_photo"
        and document.get("result_available") is True
    )


def build_readings(document):
    from field_recognition.measurements.measurement_records import binding_for_region

    candidates = document.get("instrument_candidates") or (
        [
            {
                "instrument": document["instrument"],
                "binding_id": document.get("binding_id"),
                "basis": document.get("instrument_association"),
            }
        ]
        if document.get("instrument")
        else []
    )
    result = []
    region_ordinals = {}
    regions = {r["panel_id"]: r for r in document.get("panel_regions", [])}
    for line in document.get("lines", []):
        text = line.get("text", "")
        if (
            document.get("device") != "cloud"
            and line.get("reading_source") != "vision"
            and not READOUT.fullmatch(text)
        ):
            continue
        match = re.match(r"\s*(" + NUMBER + ")", text)
        if not match and line.get("value") is None:
            continue
        region = regions.get(line.get("panel_id"))
        # Background candidates remain in raw lines/local_ocr for diagnostics.
        # A detected region may only become a reading under its task snapshot.
        if (region is not None and binding_for_region(document, region) is None) or (
            document.get("panel_detection") and region is None
        ):
            continue
        local_candidates = (
            [
                {
                    "instrument": region["instrument"],
                    "binding_id": region.get("binding_id"),
                    "basis": region["association_basis"],
                }
            ]
            if region and region.get("instrument")
            else []
            if document.get("panel_detection")
            else candidates
        )
        # A QR somewhere in a whole photo or the sole active binding does not
        # establish which screen produced an unlocalized number.
        unique = local_candidates[0] if region is not None and len(local_candidates) == 1 else None
        value = str(line["value"]) if line.get("value") is not None else match[1]
        issue = "decimal_uncertain" if re.fullmatch(r"[+-]?0\d+", value) else None
        if re.fullmatch(r"[+-]?8{5,}(?:\.8+)?", value):
            issue = "possible_display_self_test"
        vision_line = document.get("device") == "cloud" or line.get("reading_source") == "vision"
        observed_unit = (
            line.get("unit")
            if vision_line
            else line.get("unit") or (text[match.end() :].strip() if match else None) or None
        )
        region_id = region.get("panel_id") if region else None
        region_ordinals[region_id] = region_ordinals.get(region_id, 0) + 1
        reading_id = (
            f"{document['job_id']}:{region_id}:{region_ordinals[region_id]}"
            if document.get("reading_mode") == "parallel_photo" and region_id
            else f"{document['job_id']}:{len(result) + 1}"
        )
        result.append(
            {
                "reading_id": reading_id,
                "text": text,
                "value": value,
                "unit": observed_unit,
                "polygon": line.get("polygon"),
                "confidence": line.get("confidence"),
                "clarity": line.get("clarity"),
                "normalization": line.get("normalization"),
                "raw_value": line.get("raw_value"),
                "display_rule": line.get("display_rule"),
                "normalized_value": line.get("normalized_value"),
                "instrument": unique["instrument"] if unique else None,
                "binding_id": unique.get("binding_id") if unique else None,
                "association_basis": unique.get("basis")
                if unique
                else "ambiguous"
                if len(local_candidates) > 1
                else "unlocalized"
                if local_candidates or document.get("panel_detection")
                else "unbound",
                "instrument_candidates": local_candidates,
                "quality_issue": line.get("quality_issue") or issue,
                "quality_issues": line.get("quality_issues", []),
                "reading_source": line.get("reading_source"),
                "human_verified": False,
                "status": "needs_review",
                "workbench": region.get("workbench") if region else document.get("workbench"),
                "workbenches": document.get("workbenches", []),
                "activity_confirmed": False,
                "image_url": document.get("image_url"),
                "capture_id": document["capture_id"],
                "panel_id": line.get("panel_id"),
                "panel_bbox": region.get("bbox") if region else None,
                "measurement_name": region.get("measurement_name")
                if region
                else line.get("measurement_name"),
                "digit_region": region.get("digit_region") if region else None,
                "temporal_confirmation": line.get("temporal_confirmation")
                or {"status": "single_frame"},
                "panel_image_url": region.get("image_url") if region else None,
                "panel_image_sha256": region.get("image_sha256") if region else None,
                "detector_confidence": region.get("detector_confidence") if region else None,
                "detector_weights_sha256": (document.get("panel_detection") or {}).get(
                    "weights_sha256"
                ),
            }
        )
    from field_recognition.measurements.measurement_records import field_issues, measurement_value

    for reading in result:
        if reading.get("measurement_name") and reading.get("instrument"):
            issues = field_issues(reading["measurement_name"], reading, reading["instrument"])
            reading["quality_issues"] = issues
            reading["quality_issue"] = issues[0] if issues else None
            reading["normalized_value"] = measurement_value(
                reading["measurement_name"], reading, reading["instrument"]
            )[0]
    return result
