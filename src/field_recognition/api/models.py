"""HTTP request schemas shared with the Agent compatibility API."""

from __future__ import annotations

import uuid
from typing import Literal

from pydantic import BaseModel, Field, FiniteFloat, model_validator
from field_recognition.measurements.photo_measurements import MeasurementContext


class MeasurementRange(BaseModel):
    unit: str | None = Field(default=None, max_length=20)
    range: tuple[FiniteFloat | None, FiniteFloat | None] = (None, None)
    decimal_places: int | None = Field(default=None, ge=0, le=8, strict=True)
    fixed_decimal_display: bool | None = None

    @model_validator(mode="after")
    def ordered(self):
        low, high = self.range
        if low is not None and high is not None and low > high:
            raise ValueError("范围最小值不能大于最大值")
        if self.fixed_decimal_display and (self.decimal_places is None or self.decimal_places == 0):
            raise ValueError("固定小数显示需要确认 1 至 8 位小数")
        return self


class InstrumentEdit(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    scene: str = Field(min_length=1, max_length=100)
    model: str = Field(default="", max_length=100)
    device_no: str | None = Field(default=None, max_length=100)
    device_category: str | None = Field(default=None, max_length=50)
    measurement_ranges: dict[str, MeasurementRange] | None = None


class BindingRequest(BaseModel):
    scan_id: uuid.UUID
    instrument_id: uuid.UUID
    operator: str = Field(min_length=1, max_length=80)
    wearer_id: str | None = Field(default=None, max_length=100)
    request_id: uuid.UUID | None = None
    expected_binding_id: uuid.UUID | None = None
    expected_revision: int | None = Field(default=None, ge=0)
    action: Literal["reuse", "refresh"] = "reuse"


class OcrRequest(BaseModel):
    binding_id: uuid.UUID | None = None
    capture_id: uuid.UUID
    crop: list[int] | None = None
    auto_associate: bool = False
    measurement: MeasurementContext | None = None


class EndRelations(BaseModel):
    camera_id: str = Field(min_length=1, max_length=100)
