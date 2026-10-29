"""Transport-neutral application errors and deferred task scheduling."""

from __future__ import annotations

from typing import Any, Protocol


class ApplicationError(Exception):
    """An application rejection mapped to a response by the transport adapter."""

    def __init__(
        self, status_code: int, detail: Any, headers: dict[str, str] | None = None
    ):
        super().__init__(str(detail))
        self.status_code = status_code
        self.detail = detail
        self.headers = headers


class DeferredTasks(Protocol):
    def add_task(self, function, *args, **kwargs) -> None: ...


class UploadedFile(Protocol):
    filename: str | None

    async def read(self, size: int = -1) -> bytes: ...
