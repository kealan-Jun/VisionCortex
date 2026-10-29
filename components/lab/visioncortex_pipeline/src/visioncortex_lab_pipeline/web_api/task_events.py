"""Task event delivery and persistent consumer cursors."""

import hashlib
from pathlib import Path
from fastapi import HTTPException, Request
from pydantic import BaseModel, Field
from ..task_events import EventStore


class Acknowledgement(BaseModel):
    cursor: int = Field(ge=0)


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
            from ..device_day_contract import safe_child, read_json

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
