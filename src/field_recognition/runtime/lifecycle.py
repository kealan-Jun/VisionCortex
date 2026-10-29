"""Start and stop the components belonging to an explicitly constructed runtime."""

from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(runtime, application, *, wait=False):
    camera = None
    camera_started = False
    try:
        if runtime["RUNTIME_ENABLED"] and runtime.service_role == "combined":
            raise RuntimeError("Managed deployment requires api/capture/ocr/archive roles")
        if runtime.service_role == "api":
            # Request processes own no background consumers; close on all exits.
            yield
            return
        camera = runtime["receiver_camera"]
        if camera:
            runtime["live_scanner"].observe_service(
                {"online": False, "failure_reason": "capture_service_restarted"}
            )
            camera.start()
            camera_started = True
        with runtime["db"]() as conn:
            has_binding = conn.execute(
                "SELECT 1 FROM bindings WHERE ended IS NULL LIMIT 1"
            ).fetchone()
        if (
            has_binding
            or runtime["saved_photo_watcher"].enabled()
            or runtime["video_ocr"].enabled()
        ):
            runtime["queue_ocr_warmup"]()
        if not runtime["RUNTIME_ENABLED"]:
            from field_recognition.runtime.photo_job_queue import recover

            recover(runtime)
        runtime["saved_photo_watcher"].start()
        if not runtime["RUNTIME_ENABLED"]:
            runtime["archive_store"].start()
        runtime["automatic_runner"].start()
        runtime["video_ocr"].start()
        yield
    finally:
        runtime["stopping"].set()
        try:
            if camera_started:
                runtime["live_scanner"].observe_service(
                    {"online": False, "failure_reason": "capture_service_stopped"}
                )
        finally:
            runtime.close(wait=wait)
