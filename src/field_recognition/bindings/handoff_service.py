"""Explicit handoff facade; uses the same QR evidence and ownership checks as HTTP."""

from enum import Enum
from collections.abc import Mapping
from field_recognition.bindings.instrument_ownership import HandoffRequest, HandoffDecision


class HandoffState(str, Enum):
    REQUESTED = "requested"
    RELEASED = "released"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class HandoffService:
    def __init__(self, core):
        if not isinstance(core, Mapping) or "handoff_actions" not in core:
            raise ValueError(
                "Use the installed ownership workflow; fresh recipient QR evidence is required"
            )
        self.actions = core["handoff_actions"]

    def request_handoff(self, **request):
        return self.actions["request"](HandoffRequest(**request))

    def transition(self, handoff_id, action, **decision):
        return self.actions["transition"](handoff_id, action, HandoffDecision(**decision))

    def list_handoffs(self):
        return self.actions["listing"]()
