"""One instrument per photograph, with nulls for facts the evidence cannot supply."""

import math
import re
from decimal import Decimal, InvalidOperation
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from field_recognition.measurements.readout_rules import UNITS
from field_recognition.measurements.reading_results import result_ready


SCHEMA_VERSION = "instrument-measurement/1"
RULE_VERSION = "instrument-measurement/6"


def binding_for_region(document, region):
    """Recheck a region against this task's immutable capture-time authorization."""
    if document.get("attribution_status") == "needs_review":
        return None
    iid = (region.get("instrument") or {}).get("id")
    if not iid or not region.get("binding_id") or region.get("association_issue"):
        return None
    if region.get("instrument_id", iid) != iid:
        return None
    snapshots = document.get("all_binding_snapshots", document.get("binding_snapshots", []))
    matches = [
        b
        for b in snapshots
        if b.get("binding_id") == region["binding_id"]
        and (b.get("instrument") or {}).get("id") == iid
    ]
    if len(matches) != 1:
        return None
    binding = matches[0]
    if binding.get("attribution_status") == "needs_review":
        return None
    if (
        binding.get("camera_id")
        and document.get("camera_id")
        and binding["camera_id"] != document["camera_id"]
    ):
        return None
    captured_at, _ = capture_clock(document)
    if captured_at and binding.get("started_at"):
        from field_recognition.bindings.binding_policy import contains

        try:
            if not contains(binding, captured_at):
                return None
        except (TypeError, ValueError, AttributeError):
            return None
    return binding


def apply_fixed_decimal(name, line, instrument):
    """Place a registered fixed decimal in a complete integer glyph string.

    decimal_places alone remains a validation rule. An operator must also
    confirm fixed_decimal_display; neither class nor model supplies this fact.
    """
    spec = (instrument.get("measurement_ranges") or {}).get(name) or {}
    places = spec.get("decimal_places")
    raw = line.get("text", "").strip()
    if (
        spec.get("fixed_decimal_display") is not True
        or type(places) is not int
        or not 1 <= places <= 8
        or (line.get("value") is not None and str(line["value"]) != raw)
        or line.get("quality_issue")
        or not re.fullmatch(r"[+-]?(?:0|[1-9]\d*)\d{" + str(places) + "}", raw)
    ):
        return line
    sign = raw[0] if raw[0] in "+-" else ""
    digits = raw.lstrip("+-")
    value = sign + digits[:-places] + "." + digits[-places:]
    return dict(
        line,
        value=value,
        normalization={
            "method": "registered_fixed_decimal_v2",
            "raw_text": raw,
            "decimal_places": places,
            "fixed_decimal_display": True,
            "instrument_id": instrument.get("id"),
            "value": value,
        },
    )


def capture_clock(document):
    """Never treat a receive/processing clock or unsynchronized device tick as UTC."""
    if document.get("external_photo") is not None:
        return (document.get("external_photo") or {}).get("captured_at"), "source_capture_time"
    video = document.get("video_observation") or {}
    if (
        video.get("clock_sync_valid") is True
        and video.get("capture_time_basis") == "synchronized_frame_timestamp"
    ):
        return video.get("captured_at"), video.get("capture_time_basis")
    if video or document.get("request_trigger") == "video_stream":
        clock = video_clock(document.get("frame_metadata") or {})
        return clock["captured_at"], clock["capture_time_basis"]
    return None, None


def video_clock(metadata):
    value = metadata.get("global_timestamp_us")
    if metadata.get("clock_sync_valid") is True and type(value) is int and value > 0:
        try:
            moment = datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(microseconds=value)
            return {
                "captured_at": moment.isoformat(),
                "capture_time_basis": "synchronized_frame_timestamp",
                "global_timestamp_us": value,
                "clock_sync_valid": True,
            }
        except (OverflowError, ValueError):
            pass
    return {
        "captured_at": None,
        "capture_time_basis": None,
        "global_timestamp_us": value,
        "clock_sync_valid": metadata.get("clock_sync_valid") is True,
    }


def photo_time(document):
    value, _ = capture_clock(document)
    try:
        moment = datetime.fromisoformat(value)
        if moment.tzinfo is None:
            raise ValueError("timezone unknown")
        delta = moment.astimezone(timezone.utc) - datetime(1970, 1, 1, tzinfo=timezone.utc)
        milliseconds = (delta.days * 86400 + delta.seconds) * 1000 + delta.microseconds // 1000
        return {
            "timestamp_ms": milliseconds,
            "time": moment.astimezone(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S.%f")[
                :-3
            ],
        }
    except (TypeError, ValueError, OverflowError):
        return {"timestamp_ms": None, "time": None}


def string(value):
    return value.strip() or None if isinstance(value, str) else None


def number(reading):
    raw = str(reading.get("value", ""))
    if reading.get("quality_issue") or not re.fullmatch(r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)", raw):
        return None
    # Never turn a missing decimal such as 02287 into 2287.
    if re.fullmatch(r"[+-]?0\d+(?:\.\d+)?", raw):
        return None
    try:
        value = float(raw) if "." in raw else int(raw)
        return value if math.isfinite(value) else None
    except (OverflowError, ValueError):
        return None


def unit_evidence(name, reading, instrument):
    observed = string(reading.get("unit"))
    registered = string(((instrument.get("measurement_ranges") or {}).get(name) or {}).get("unit"))
    allowed = UNITS.get(name, set())
    conflict = bool(
        (observed and observed not in allowed)
        or (registered and registered not in allowed)
        or (observed and registered and observed != registered)
    )
    if conflict:
        return None, "unit_conflict"
    return (
        (observed, "recognition_text")
        if observed
        else (registered, "instrument_registry")
        if registered
        else (None, "unknown")
    )


def measurement_value(name, reading, instrument):
    unit, basis = unit_evidence(name, reading, instrument)
    bounds = (instrument.get("measurement_ranges") or {}).get(name) or {}
    limits = bounds.get("range") if unit and bounds.get("unit") == unit else None
    if (
        not isinstance(limits, list)
        or len(limits) != 2
        or any(
            v is not None and (type(v) not in (float, int) or not math.isfinite(v)) for v in limits
        )
        or all(v is not None for v in limits)
        and limits[0] > limits[1]
    ):
        limits = [None, None]
    return {
        "name": name,
        "value": None if field_issues(name, reading, instrument) else number(reading),
        "unit": unit,
        "range": limits,
    }, basis


def field_issues(name, reading, instrument):
    """Only enforce registered facts; never repair decimals or invent limits."""
    issues = [reading["quality_issue"]] if reading.get("quality_issue") else []
    # Keep supporting causes while the primary issue is active. A corrected
    # reading clears that primary issue; stale diagnostics must not reject it.
    extra = reading.get("quality_issues")
    if issues and isinstance(extra, list):
        issues.extend(issue for issue in extra if isinstance(issue, str) and issue)
    unit, basis = unit_evidence(name, reading, instrument)
    if name in UNITS and basis == "unit_conflict":
        issues.append(basis)
    value = number(reading)
    if value is None:
        if reading.get("value") is not None:
            issues.append("invalid_number")
        return list(dict.fromkeys(issues))
    spec = (instrument.get("measurement_ranges") or {}).get(name) or {}
    bounds = spec.get("range")
    if unit and unit == spec.get("unit") and isinstance(bounds, (list, tuple)) and len(bounds) == 2:
        low, high = bounds
        valid = all(v is None or type(v) in (int, float) and math.isfinite(v) for v in bounds)
        if valid and not (low is not None and high is not None and low > high):
            if (low is not None and value < low) or (high is not None and value > high):
                issues.append("outside_registered_range")
    places = spec.get("decimal_places")
    if type(places) is int and 0 <= places <= 8:
        try:
            # Display precision concerns observed text; user corrections are numeric.
            actual = max(0, -Decimal(str(reading["value"])).as_tuple().exponent)
            if actual != places and not reading.get("corrected"):
                issues.append("display_precision_mismatch")
            elif reading.get("corrected") and Decimal(str(value)).quantize(
                Decimal(1).scaleb(-places)
            ) != Decimal(str(value)):
                issues.append("display_precision_mismatch")
        except InvalidOperation:
            issues.append("invalid_number")
    return list(dict.fromkeys(issues))


def select_field_reading(matches, name, instrument):
    """Same-frame duplicate OCR candidates may differ in quality, not in value."""
    if not matches:
        return {}, None
    if len(matches) == 1:
        return matches[0], None
    # Different windows cannot silently vote for one physical field.
    if len({r.get("panel_id") for r in matches}) != 1:
        return {}, "multiple_field_regions"
    signatures = set()
    for reading in matches:
        value, basis = measurement_value(name, reading, instrument)
        if value["value"] is None or basis == "unit_conflict":
            return {}, "readout_conflict"
        signatures.add((Decimal(str(reading["value"])), value["unit"]))
    if len(signatures) != 1:
        return {}, "readout_conflict"
    return max(
        matches,
        key=lambda r: (
            {"clear": 2, "medium": 1}.get(r.get("clarity"), 0),
            r.get("confidence") or 0,
        ),
    ), None


def display_fields(document):
    """Common derived field view for API, browser and archived results."""
    result = []
    for entry in build_records(document):
        iid = entry["instrument_id"]
        for value in entry["record"]["values"]:
            regions = [
                r
                for r in document.get("panel_regions", [])
                if (r.get("instrument") or {}).get("id") == iid
                and r.get("measurement_name") == value["name"]
                and binding_for_region(document, r) is not None
            ]
            allowed = {(r["panel_id"], r.get("binding_id")) for r in regions}
            matches = [
                r
                for r in document.get("readings", [])
                if (r.get("instrument") or {}).get("id") == iid
                and r.get("measurement_name") == value["name"]
                and (r.get("panel_id"), r.get("binding_id")) in allowed
            ]
            region = regions[0] if len(regions) == 1 else {}
            reading, conflict = select_field_reading(
                matches, value["name"], region.get("instrument") or {}
            )
            state = (region.get("display_state") or {}).get("text")
            issues = field_issues(value["name"], reading, region.get("instrument") or {})
            if conflict:
                issues.append(conflict)
            unit_basis = entry["evidence"]["unit_basis"].get(value["name"])
            if value["value"] is not None and unit_basis == "unknown":
                issues.append("unit_unknown")
            result.append(
                value
                | {
                    "instrument_id": iid,
                    "instrument_name": (region.get("instrument") or {}).get("name"),
                    "panel_id": region.get("panel_id"),
                    "panel_image_url": region.get("image_url"),
                    "raw_text": reading.get("text")
                    or ((region.get("display_state") or {}).get("raw_text") if state else None),
                    "display_value": reading.get("value"),
                    "unit_basis": unit_basis,
                    "display_state": state,
                    "issues": issues,
                    "status": "display_state"
                    if state
                    else "needs_correction"
                    if issues
                    else "recognized"
                    if value["value"] is not None
                    else "unreadable",
                }
            )
    return result


def build_records(document):
    """The record follows the user's schema; provenance lives outside it."""
    groups = {}
    for region in document.get("panel_regions", []):
        if binding_for_region(document, region) is not None:
            groups.setdefault(region["instrument"]["id"], []).append(region)
    results = []
    for iid, regions in groups.items():
        instrument = regions[0]["instrument"]
        classes = {r.get("class_id") for r in regions}
        names = ["温度", "转速"] if classes == {0} else ["质量"] if classes == {1} else []
        if not names:
            continue
        binding = next(
            (
                b
                for b in document.get(
                    "all_binding_snapshots", document.get("binding_snapshots", [])
                )
                if b.get("binding_id") == regions[0]["binding_id"]
            ),
            {},
        )
        qr = next(
            (q for q in document.get("qr_matches", []) if q["id"] == iid and q.get("qr_hash")), None
        )
        digest = qr["qr_hash"] if qr else binding.get("qr_hash")
        digest = (
            digest if isinstance(digest, str) and re.fullmatch("[a-f0-9]{64}", digest) else None
        )
        panel_ids = {r["panel_id"] for r in regions}
        candidates = [
            r
            for r in document.get("readings", [])
            if (r.get("instrument") or {}).get("id") == iid
            and r.get("panel_id") in panel_ids
            and r.get("binding_id", regions[0]["binding_id"]) == regions[0]["binding_id"]
        ]
        values, unit_bases, field_statuses = [], {}, {}
        for name in names:
            matches = [r for r in candidates if r.get("measurement_name") == name]
            reading, conflict = select_field_reading(matches, name, instrument)
            if not result_ready(document):
                reading = {}
            value, unit_bases[name] = measurement_value(name, reading, instrument)
            displays = [r for r in regions if r.get("measurement_name") == name]
            if (
                result_ready(document)
                and len(displays) == 1
                and (displays[0].get("display_state") or {}).get("text") == "OFF"
            ):
                value.update(value=None, display_state="OFF")
            field_statuses[name] = (
                "display_state"
                if value.get("display_state")
                else conflict
                if conflict
                else "field_role_unconfirmed"
                if not displays and any(not r.get("measurement_name") for r in regions)
                else "field_not_detected"
                if not displays
                else "field_rule_conflict"
                if field_issues(name, reading, instrument)
                else "unreadable"
                if value["value"] is None
                else "unit_unknown"
                if unit_bases[name] == "unknown"
                else "recognized"
            )
            values.append(value)
        record = {
            "wearer_id": string(binding.get("wearer_id"))
            or string((document.get("operator_registration") or {}).get("wearer_id")),
            "device_model": string(instrument.get("model")),
            "device_no": string(instrument.get("device_no")),
            "qr_hash": digest,
            "photo_time": photo_time(document),
            "values": values,
        }
        results.append(
            {
                "instrument_id": iid,
                "record": record,
                "evidence": {
                    "job_id": document["job_id"],
                    "capture_id": document["capture_id"],
                    "image_url": document.get("image_url"),
                    "image_sha256": document.get("image_sha256"),
                    "source_ref": (document.get("external_photo") or {}).get("source_ref"),
                    "binding_id": regions[0]["binding_id"],
                    "qr_basis": "same_photo_decode" if qr else "binding_decode" if digest else None,
                    "qr_scan_id": document["capture_id"]
                    if qr
                    else binding.get("scan_id")
                    if digest
                    else None,
                    "photo_time_basis": capture_clock(document)[1]
                    if record["photo_time"]["timestamp_ms"] is not None
                    else None,
                    "schema_version": SCHEMA_VERSION,
                    "rule_version": RULE_VERSION,
                    "unit_basis": unit_bases,
                    "field_statuses": field_statuses,
                    "field_completeness": "complete"
                    if all(s in {"recognized", "display_state"} for s in field_statuses.values())
                    else "incomplete",
                    "panel_ids": [r["panel_id"] for r in regions],
                    "candidate_readings": candidates,
                    "human_verified": False,
                    "value_meaning": "displayed_value_setpoint_or_actual_unknown",
                },
            }
        )
    return results
