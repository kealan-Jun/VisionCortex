"""Explicit user-confirmed display type templates, without instrument identity."""

import json
import os
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class DisplayRule:
    instrument_type: str
    measurement_name: str
    unit: str
    decimal_places: int
    fixed_decimal_display: bool
    confirmed_by_user: bool

    def __post_init__(self):
        if (
            self.instrument_type != "balance"
            or self.measurement_name != "质量"
            or self.unit != "g"
            or self.confirmed_by_user is not True
            or self.fixed_decimal_display is not True
            or type(self.decimal_places) is not int
            or not 1 <= self.decimal_places <= 8
        ):
            raise ValueError("Display rule requires an explicit supported format confirmation")

    def as_dict(self):
        return asdict(self)


def configured_rules(raw: str | None = None):
    """Parse an explicit source; omitted sources preserve the legacy env adapter."""
    raw = os.environ.get("FIELD_VISION_DISPLAY_RULES", "[]") if raw is None else raw
    if not isinstance(raw, str) or len(raw) > 4096:
        return []
    try:
        rules = json.loads(raw)
    except (TypeError, ValueError):
        return []
    if not isinstance(rules, list) or len(rules) > 4:
        return []
    accepted = []
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        if (
            rule.get("instrument_type") != "balance"
            or rule.get("measurement_name") != "质量"
            or rule.get("unit") != "g"
            or rule.get("confirmed_by_user") is not True
            or rule.get("fixed_decimal_display") is not True
            or type(rule.get("decimal_places")) is not int
            or not 1 <= rule["decimal_places"] <= 8
        ):
            continue
        accepted.append(
            {
                key: rule[key]
                for key in (
                    "instrument_type",
                    "measurement_name",
                    "unit",
                    "decimal_places",
                    "fixed_decimal_display",
                    "confirmed_by_user",
                )
            }
        )
    # Conflicting templates cannot select a precision by array order.
    return accepted[:1] if accepted and len({r["decimal_places"] for r in accepted}) == 1 else []


def apply_rule(line, rules):
    if line.get("display_state") or line.get("value") is None:
        return line
    matching = [
        r
        for r in rules
        if line.get("label") == r["measurement_name"] and line.get("unit") == r["unit"]
    ]
    if len(matching) != 1:
        return line
    from field_recognition.measurements.measurement_records import (
        apply_fixed_decimal,
        field_issues,
        measurement_value,
    )

    rule = matching[0]
    name = rule["measurement_name"]
    instrument = {"measurement_ranges": {name: rule}}
    # The provider transcription stays in text/raw_value; only a complete integer
    # glyph string may receive the explicitly confirmed fixed decimal placement.
    normalized = apply_fixed_decimal(name, dict(line, text=line["value"]), instrument)
    result = dict(
        line,
        value=normalized.get("value"),
        measurement_name=name,
        display_rule=dict(rule, basis="user_confirmed_display_type"),
        raw_value=line["value"],
    )
    if normalized.get("normalization"):
        result["normalization"] = dict(
            normalized["normalization"],
            basis="user_confirmed_display_type",
            raw_text=line["text"],
            raw_value=line["value"],
        )
        result["numeric_candidates"] = [result["value"]]
    issues = field_issues(name, result, instrument)
    result.update(
        quality_issue=next(iter(issues), None),
        quality_issues=issues,
        normalized_value=measurement_value(name, result, instrument)[0],
    )
    return result
