"""Unified task delivery, receiver timing and evidence retrieval HTTP contracts."""

import hashlib
from pathlib import Path
from fastapi import HTTPException, Request
from pydantic import BaseModel, Field
from .knowledge import Knowledge
from .task_events import EventStore


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    day: str | None = None
    camera: str | None = None


class Acknowledgement(BaseModel):
    cursor: int = Field(ge=0)


class UploadCompletion(BaseModel):
    recording_id: str = Field(min_length=1, max_length=200)
    source_signature: str = Field(min_length=1, max_length=200)
    completed_at: float = Field(gt=0, allow_inf_nan=False)


def install_routes(app, settings):
    def store():
        return EventStore(Path(settings()["storage"]["local_runtime_root"]))

    def consumer_key(request, name):
        user = getattr(request.state, "web_identity", {}).get("username", "local")
        return "http:" + hashlib.sha256((user + "\0" + name).encode()).hexdigest()

    @app.get("/api/task-events")
    def events(
        request: Request, after: int = 0, limit: int = 100, consumer: str | None = None
    ):
        return store().read(
            after=after,
            limit=limit,
            consumer=consumer_key(request, consumer) if consumer else None,
        )

    @app.get("/api/tasks/{task_id}/history")
    def history(task_id: str, after: int = 0, limit: int = 100):
        if task_id.startswith("device-day-"):
            from .device_day_contract import safe_child, read_json

            try:
                path = safe_child(
                    Path(settings()["storage"]["local_runtime_root"])
                    / "device-day"
                    / "requests",
                    task_id + ".json",
                )
                request = read_json(path)
            except (OSError, ValueError):
                raise HTTPException(404, "批次任务不存在") from None
            return store().read(
                task_ids=[r["recording_id"] for r in request["recordings"]],
                after=after,
                limit=limit,
            )
        return store().read(task_id=task_id, after=after, limit=limit)

    @app.post("/api/task-events/consumers/{consumer}/ack")
    def acknowledge(request: Request, consumer: str, payload: Acknowledgement):
        try:
            store().acknowledge(consumer_key(request, consumer), payload.cursor)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from None
        return {"acknowledged_cursor": payload.cursor}

    @app.post("/api/receiver/upload-completed")
    def upload_complete(payload: UploadCompletion):
        from .device_day_latency import upload_completed

        try:
            return upload_completed(settings(), payload.model_dump())
        except (ValueError, KeyError) as exc:
            raise HTTPException(409, str(exc)) from None

    @app.get("/api/knowledge/search")
    def search(
        q: str = "",
        day: str | None = None,
        camera: str | None = None,
        kind: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ):
        return Knowledge(settings()).search(
            q, day=day, camera=camera, kind=kind, limit=limit, offset=offset
        )

    @app.get("/api/knowledge/evidence/{identifier}")
    def evidence(identifier: str, revision: str):
        try:
            return Knowledge(settings()).evidence(identifier, revision)
        except KeyError:
            raise HTTPException(404, "证据不存在") from None
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from None

    @app.post("/api/knowledge/ask")
    def ask(payload: Question):
        try:
            result = Knowledge(settings()).ask(
                payload.question, day=payload.day, camera=payload.camera
            )
            from .submission import bind_task

            bind_task(result["id"])
            return result
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None

    @app.get("/api/knowledge/answers/{identifier}")
    def answer(identifier: str):
        try:
            return Knowledge(settings()).answer(identifier)
        except KeyError:
            raise HTTPException(404, "回答回执不存在") from None
