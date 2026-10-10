"""Recorder upload completion receipt endpoint."""

from fastapi import HTTPException
from pydantic import BaseModel, Field


class UploadCompletion(BaseModel):
    recording_id: str = Field(min_length=1, max_length=200)
    source_signature: str = Field(min_length=1, max_length=200)
    completed_at: float = Field(gt=0, allow_inf_nan=False)


def install_routes(app, settings):
    @app.post("/api/receiver/upload-completed")
    def upload_complete(payload: UploadCompletion):
        from ..device_day_latency import upload_completed

        try:
            return upload_completed(settings(), payload.model_dump())
        except (ValueError, KeyError) as exc:
            raise HTTPException(409, str(exc)) from None
