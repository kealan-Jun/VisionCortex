"""One runtime's dependencies, with a mapping adapter for existing components."""

from __future__ import annotations

import inspect
from collections.abc import Iterator, MutableMapping
from functools import partial
from pathlib import Path
from types import ModuleType
from typing import Any, get_type_hints

from field_recognition.runtime.production_config import ProductionConfig


class RuntimeContext(MutableMapping[str, Any]):
    """Explicit instance state; canonical imports never create this object.

    Existing components use ``core[key]`` and retain this adapter. Only the
    historical root ``app`` entry point attaches its module dictionary, so
    existing ``setattr(app, ...)`` doubles and ``vars(app)`` callers keep seeing
    the same state. Ordinary factories and workers own independent dictionaries.
    """

    def __init__(self, config: ProductionConfig, project_root: Path):
        self._closed = False
        self._values: dict[str, Any] = {"config": config, "BASE": project_root}
        self._values["runtime"] = self

    @property
    def config(self) -> ProductionConfig:
        return self["config"]

    @property
    def project_root(self) -> Path:
        return self["BASE"]

    @property
    def database(self):
        return self["database"]

    @property
    def data_dir(self) -> Path:
        return self["DATA"]

    @property
    def service_role(self) -> str:
        return self["SERVICE_ROLE"]

    def __getitem__(self, key: str) -> Any:
        return self._values[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._values[key] = value

    def __delitem__(self, key: str) -> None:
        del self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def bind(self, **actions) -> None:
        """Bind explicitly named actions without changing their HTTP signatures."""
        for name, action in actions.items():
            bound = partial(action, self)
            hints = get_type_hints(action)
            signature = inspect.signature(bound)
            bound.__signature__ = signature.replace(
                parameters=[
                    parameter.replace(annotation=hints.get(parameter.name, parameter.annotation))
                    for parameter in signature.parameters.values()
                ],
                return_annotation=hints.get("return", signature.return_annotation),
            )
            bound.__name__ = name
            bound.__doc__ = action.__doc__
            self[name] = bound

    def attach_compatibility_module(self, module: ModuleType) -> None:
        """Export an initialized runtime only for the legacy ``import app`` API."""
        module.__dict__.update(self._values)
        self._values = module.__dict__

    def close(self, *, wait: bool = False) -> None:
        """Dispose an explicitly constructed runtime, including unused factories."""
        if self._closed:
            return
        self._closed = True
        if self.get("stopping") is not None:
            self["stopping"].set()
        error = None

        def dispose(action):
            nonlocal error
            try:
                action()
            except Exception as exc:
                if error is None:
                    error = exc

        for name in (
            "video_ocr",
            "automatic_runner",
            "saved_photo_watcher",
            "archive_store",
            "live_scanner",
            "receiver_camera",
        ):
            component = self.get(name)
            if component is not None:
                dispose(component.close)
        for name in ("readout_pool", "vision_pool", "ocr_pool"):
            pool = self.get(name)
            if pool is not None:
                dispose(partial(pool.shutdown, wait=wait, cancel_futures=True))
        # Waiting workers finish model calls before closing a loaded client.
        # Disposing a cold factory never loads an OCR model.
        model = self.get("ocr_model")
        close_model = getattr(model, "close", None) if model is not None else None
        if callable(close_model):
            dispose(close_model)
        if self.get("panel_detector") is not None:
            dispose(self["panel_detector"].close)
        if error is not None:
            raise error
