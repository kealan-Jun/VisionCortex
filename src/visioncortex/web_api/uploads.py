"""HTTP adapters for uploads"""

from __future__ import annotations
from typing import Annotated, Any
from fastapi import BackgroundTasks, File, Form, Request, UploadFile

from ..application import run_submission_service, upload_service

from fastapi import APIRouter

router = APIRouter()


def create_upload_session(payload: dict[str, Any]) -> dict[str, Any]:
    return upload_service.create_upload_session(payload=payload)


router.post("/api/upload-sessions", status_code=201)(create_upload_session)


def get_upload_session(session_id: str) -> dict[str, Any]:
    return upload_service.get_upload_session(session_id=session_id)


router.get("/api/upload-sessions/{session_id}")(get_upload_session)


def cancel_upload_session(session_id: str) -> dict[str, Any]:
    return upload_service.cancel_upload_session(session_id=session_id)


router.delete("/api/upload-sessions/{session_id}")(cancel_upload_session)


async def upload_session_chunk(
    request: Request, session_id: str, file_id: str
) -> dict[str, Any]:
    return await upload_service.upload_session_chunk(
        request=request, session_id=session_id, file_id=file_id
    )


router.patch("/api/upload-sessions/{session_id}/files/{file_id}")(upload_session_chunk)


def finalize_upload_session(
    session_id: str, background_tasks: BackgroundTasks
) -> dict[str, Any]:
    return run_submission_service.finalize_upload_session(
        session_id=session_id, background_tasks=background_tasks
    )


router.post("/api/upload-sessions/{session_id}/finalize", status_code=202)(
    finalize_upload_session
)


async def create_run(
    request: Request,
    background_tasks: BackgroundTasks,
    experiment_name: Annotated[str, Form()],
    view_specs_json: Annotated[str, Form()],
    videos: Annotated[list[UploadFile], File()],
    timestamp_csvs: Annotated[list[UploadFile] | None, File()] = None,
) -> dict[str, Any]:
    return await run_submission_service.create_run(
        request=request,
        background_tasks=background_tasks,
        experiment_name=experiment_name,
        view_specs_json=view_specs_json,
        videos=videos,
        timestamp_csvs=timestamp_csvs,
    )


router.post("/api/runs", status_code=202)(create_run)
