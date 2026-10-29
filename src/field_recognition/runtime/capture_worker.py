"""One isolated capture runtime per registered camera, with shared GPU inference IPC."""

import asyncio
import os
import signal
import uuid

from field_recognition.runtime.runtime_rpc import RpcServer
from field_recognition.runtime.worker_support import process_lock, heartbeat


async def run(config=None, core=None):
    from field_recognition.runtime.factory import create_runtime

    core = core if core is not None else create_runtime(config, service_role="capture")
    try:
        await _run_capture(core)
    finally:
        runtime = core.get("runtime")
        if runtime is not None:
            runtime.close(wait=True)


async def _run_capture(core):
    from field_recognition.capture.automation import AutomationSettings
    from field_recognition.capture.video_ocr import VideoSettings
    from field_recognition.capture.live_scan import preview_part, recognition_part

    camera = core["receiver_camera"]
    target = core["config"].camera_id
    if not target or camera is None:
        raise RuntimeError("Capture runtime requires a camera id and receiver")

    def preview(recognition, previous):
        # JSON transport changes tuple tokens to lists; restore their shape so
        # the same frame is not encoded and delivered repeatedly.
        def token(value):
            return tuple(token(v) for v in value) if isinstance(value, list) else value

        previous = token(previous)
        return (
            recognition_part(core["video_ocr"], camera, previous)
            if recognition
            else preview_part(camera, previous)
        )

    with process_lock(core["DATA"], "capture-" + target):
        handlers = {
            "receiver.snapshot": camera.snapshot,
            "receiver.frame": camera.frame,
            "receiver.preview": preview,
            "scanner.snapshot": lambda: core["live_scanner"].session,
            "scanner.active_bindings": core["live_scanner"].active_bindings,
            "scanner.active_binding": core["live_scanner"].active_binding,
            "scanner.start": core["live_scanner"].start,
            "scanner.refresh_bindings": core["live_scanner"].refresh_bindings,
            "scanner.read": lambda sid, **kw: core["live_scanner"].read(uuid.UUID(sid), **kw),
            "scanner.observe_service": core["live_scanner"].observe_service,
            "automation.snapshot": core["automatic_runner"].snapshot,
            "automation.settings": core["automatic_runner"].settings,
            "automation.configure": lambda body: core["automatic_runner"].configure(
                AutomationSettings(**body)
            ),
            "automation.pause": core["automatic_runner"].pause,
            "automation._save": core["automatic_runner"]._save,
            "video.snapshot": core["video_ocr"].snapshot,
            "video.configure": lambda body: core["video_ocr"].configure(VideoSettings(**body)),
            "video.preview": core["video_ocr"].preview,
            "watcher.snapshot": core["saved_photo_watcher"].snapshot,
            "watcher.enabled": core["saved_photo_watcher"].enabled,
        }
        server = RpcServer("camera:" + target, handlers, data_dir=core["DATA"])
        try:
            server.start()
            stop = asyncio.Event()
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, stop.set)
            async with core["lifespan"](core["app"], wait=True):
                while not stop.is_set():
                    # This status is local only; slow NAS polling occurs on its own thread.
                    heartbeat(
                        core["database"],
                        "camera:" + target,
                        {"status": "running", "camera_id": target, "receiver": camera.snapshot()},
                    )
                    try:
                        await asyncio.wait_for(stop.wait(), timeout=2)
                    except asyncio.TimeoutError:
                        pass
        finally:
            server.close()
            heartbeat(core["database"], "camera:" + target, {"status": "stopped"})


if __name__ == "__main__":
    os.environ["FIELD_SERVICE_ROLE"] = "capture"
    asyncio.run(run())
