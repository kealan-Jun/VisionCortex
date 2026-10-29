"""Evidence search and source-grounded questions and answers."""

from fastapi import HTTPException
from pydantic import BaseModel, Field
from ..knowledge import Knowledge


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    day: str | None = None
    camera: str | None = None


def install_routes(app, settings):
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
            from ..submission import bind_task

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
