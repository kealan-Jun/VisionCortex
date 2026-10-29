"""Side-effect-free HTTP application factory for an explicit runtime instance."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from field_recognition.api.agent_tools import install_tools
from field_recognition.api.production_routes import install as install_production_routes
from field_recognition.api.security import install as install_security
from field_recognition.runtime.context import RuntimeContext
from field_recognition.runtime.factory import create_runtime
from field_recognition.runtime.production_config import ProductionConfig


def create_app(
    runtime: RuntimeContext | None = None,
    *,
    config: ProductionConfig | None = None,
    **runtime_options,
) -> FastAPI:
    """Create an independent HTTP app, or attach HTTP to an existing runtime."""
    if runtime is not None and (config is not None or runtime_options):
        raise TypeError("Pass either an existing runtime or factory settings")
    runtime = runtime if runtime is not None else create_runtime(config, **runtime_options)
    existing = runtime.get("http_app")
    if existing is not None:
        return existing
    app = FastAPI(
        title="FieldRecognition · 现场识别", version="0.1.0", lifespan=runtime["lifespan"]
    )
    app.state.runtime = runtime
    app.mount("/static", StaticFiles(directory=runtime.project_root / "static"), name="static")
    for path, method, action, options in (
        ("/", "GET", "index", {}),
        ("/api/state", "GET", "state", {}),
        ("/api/devices", "GET", "devices", {}),
        ("/api/instruments/{instrument_id}", "PUT", "edit_instrument", {}),
        ("/api/scans", "POST", "scan", {}),
        ("/api/camera/capture", "POST", "camera_capture", {}),
        ("/api/images/{capture_id}", "GET", "image_file", {}),
        ("/api/bindings", "POST", "bind", {}),
        ("/api/bindings/{binding_id}/end", "POST", "end_binding", {}),
        ("/api/camera/relations/end", "POST", "end_relations", {}),
        ("/api/ocr", "POST", "submit_ocr", {"status_code": 202}),
        ("/api/jobs/{job_id}", "GET", "get_job", {}),
        ("/api/export", "GET", "export", {}),
        ("/api/jobs/{job_id}/measurements", "GET", "get_measurements", {}),
        ("/api/readouts", "GET", "readouts", {}),
        ("/api/workbenches", "GET", "workbenches", {}),
        ("/api/archive/files/{relative:path}", "GET", "archive_file", {}),
    ):
        app.add_api_route(path, runtime[action], methods=[method], **options)
    app.include_router(runtime["app"])
    runtime["app"] = app
    install_tools(app, runtime)
    install_security(runtime, runtime["AUTH_ENABLED"])
    install_production_routes(runtime)
    runtime["http_app"] = app
    return app
