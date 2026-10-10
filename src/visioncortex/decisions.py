from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence


DECISION_RECEIPT_SCHEMA_VERSION = "visioncortex-decision-receipt/1.0.0"
QUALITY_RULE_VERSION = "dev043-quality-v2"


def decision_receipt(
    *,
    decision_type: str,
    rule_id: str,
    verdict: str,
    subject_ids: Sequence[str],
    reason_codes: Sequence[str],
    facts: Mapping[str, Any] | None = None,
    thresholds: Mapping[str, Any] | None = None,
    evidence_refs: Sequence[str] | None = None,
    legacy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a deterministic, machine-indexable quality decision receipt.

    ``legacy`` keeps the established top-level fields consumed by the Web UI
    and existing archives.  The namespaced receipt fields are stable across
    every quality rule and deliberately omit wall-clock creation time so a
    ledger replay produces byte-for-byte identical decisions.
    """

    normalized_subjects = sorted({str(item) for item in subject_ids if item})
    normalized_reasons = sorted({str(item) for item in reason_codes if item})
    normalized_evidence = sorted(
        {str(item) for item in (evidence_refs or []) if item}
    )
    identity = {
        "decision_type": str(decision_type),
        "rule_id": str(rule_id),
        "rule_version": QUALITY_RULE_VERSION,
        "subject_ids": normalized_subjects,
        "verdict": str(verdict),
    }
    digest = hashlib.sha256(
        json.dumps(identity, ensure_ascii=True, sort_keys=True).encode("utf-8")
    ).hexdigest()[:20]
    receipt = dict(legacy or {})
    receipt.update(
        {
            "receipt_schema_version": DECISION_RECEIPT_SCHEMA_VERSION,
            "decision_id": f"DEC-{digest}",
            "decision_type": str(decision_type),
            "rule_id": str(rule_id),
            "rule_version": QUALITY_RULE_VERSION,
            "verdict": str(verdict),
            "subject_ids": normalized_subjects,
            "reason_codes": normalized_reasons,
            "facts": dict(facts or {}),
            "thresholds": dict(thresholds or {}),
            "evidence_refs": normalized_evidence,
        }
    )
    return receipt
