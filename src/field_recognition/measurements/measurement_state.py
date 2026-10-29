"""Compatibility facade over the evidence-backed photo measurement workflow.

Database-only draft creation was removed: groups are created during photo ingestion,
using capture provenance and a camera-scoped burst key.
"""

from enum import Enum
from collections.abc import Mapping
from pydantic import BaseModel, FiniteFloat
from field_recognition.measurements.photo_measurements import Decision, Revision, Rejection


class MeasurementState(str, Enum):
    DRAFT = "draft"
    COLLECTING = "collecting"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"


class FieldValue(BaseModel):
    field_id: str
    instrument_id: str
    name: str
    value: FiniteFloat
    unit: str


class MeasurementStateMachine:
    def __init__(self, core):
        if not isinstance(core, Mapping) or "measurement_actions" not in core:
            raise ValueError(
                "Use the installed photo measurement workflow; a database alone cannot establish evidence"
            )
        self.actions = core["measurement_actions"]

    def get_measurement(self, measurement_id):
        return self.actions["detail"](measurement_id)

    def update_fields(self, measurement_id, fields, actor, reason, *, expected_revision):
        return self.actions["revise"](
            measurement_id,
            Revision(
                actor=actor,
                reason=reason,
                revision=expected_revision,
                fields=[f.model_dump() if hasattr(f, "model_dump") else f for f in fields],
            ),
        )

    def confirm_measurement(self, measurement_id, actor, expected_revision, reason=None):
        record = self.actions["confirm"](
            measurement_id, Decision(actor=actor, revision=expected_revision)
        )
        return record, record["record_id"]

    def reject_measurement(self, measurement_id, actor, expected_revision, reason):
        return self.actions["reject"](
            measurement_id, Rejection(actor=actor, revision=expected_revision, reason=reason)
        )
