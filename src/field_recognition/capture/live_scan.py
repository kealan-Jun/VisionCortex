"""Live preview and opt-in, bounded continuous QR binding on the selected camera."""

import asyncio
import copy
import json
import logging
import threading
import time
import uuid

import cv2
import numpy as np
from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from field_recognition.capture.qr_decode import decode_qr

ACTIVE = {"scanning", "waiting_camera"}
LEASE_SECONDS = 20
QR_DECODE_RECOVERY_SECONDS = 30
logger = logging.getLogger(__name__)


def frame_key(metadata):
    return tuple(
        metadata.get(k)
        for k in ("decoded_frame_id", "sequence", "timestamp_us", "sender_id", "camera_id")
    )


class StartScan(BaseModel):
    operator: str = Field(min_length=1, max_length=80)
    wearer_id: str | None = Field(default=None, max_length=100)


class LiveScanner:
    def __init__(self, core):
        self.core = core
        self.lock = threading.RLock()
        self.session = None
        self.last_seen = 0
        self.worker = None
        self.shutdown = threading.Event()
        self.last_error = None

    def camera(self):
        camera = self.core["receiver_camera"]
        if camera is None:
            raise HTTPException(409, "连续视频需要配置挂脖相机主码流；当前仍可上传或拍照")
        return camera

    def active_bindings(self):
        from field_recognition.bindings.binding_policy import sweep

        sweep(self.core)
        with self.core["db"]() as conn:
            rows = conn.execute(
                "SELECT document FROM bindings WHERE camera=? AND ended IS NULL ORDER BY rowid",
                (self.camera().target,),
            ).fetchall()
        return [json.loads(row["document"]) for row in rows]

    def active_binding(self):
        bindings = self.active_bindings()
        return bindings[0] if bindings else None

    def binding_dependencies(self, instruments, scenes):
        """Retry a rejected code when its registration or occupancy changes."""
        from field_recognition.bindings.binding_policy import moment

        ids = {item["id"] for item in instruments}
        hashes = {item.get("qr_hash") for item in instruments} - {None}
        with self.core["db"]() as conn:
            occupied = [
                json.loads(row[0])
                for row in conn.execute("SELECT document FROM bindings WHERE ended IS NULL")
            ]
            transfers = [
                json.loads(row[0])
                for row in conn.execute(
                    "SELECT document FROM binding_handoffs WHERE status IN ('requested','released')"
                )
            ]
        timestamp = moment(self.core["now"]())
        state = {
            "identities": [
                {k: v for k, v in item.items() if k != "polygon"} for item in instruments + scenes
            ],
            "occupied": sorted(
                b["binding_id"]
                for b in occupied
                if b["instrument"]["id"] in ids or b.get("qr_hash") in hashes
            ),
            "transfers": sorted(
                (t["handoff_id"], t["status"], t["revision"])
                for t in transfers
                if t["instrument_id"] in ids
                and (not t.get("expires_at") or moment(t["expires_at"]) > timestamp)
            ),
            "scenes": sorted(
                v["visit_id"]
                for v in self.core["scene_visits"]()
                if v["camera_id"] == self.camera().target
            ),
        }
        return json.dumps(state, sort_keys=True)

    def refresh_bindings(self):
        bindings = self.active_bindings()
        visits = [v for v in self.core["scene_visits"]() if v["camera_id"] == self.camera().target]
        if self.session is None:
            return
        scan = self.session.get("scan") or {}
        instrument_ids = {b["instrument"]["id"] for b in bindings}
        scene_ids = {v["scene"]["id"] for v in visits}
        if (
            scan
            and all(m["id"] in instrument_ids for m in scan.get("matches", []))
            and all(m["id"] in scene_ids for m in scan.get("scene_matches", []))
        ):
            self.session["binding_errors"] = []
        self.session.update(
            bindings=bindings,
            binding=bindings[0] if len(bindings) == 1 else None,
            scene_visits=visits,
            message=f"已绑定 {len(bindings)} 台仪器、{len(visits)} 个场景；持续扫描新二维码",
        )
        errors = self.session.get("binding_errors", [])
        if errors:
            self.session["message"] += " · 最近扫码提示：" + "；".join(dict.fromkeys(errors))

    def start(self, operator, *, owner="browser", wearer_id=None):
        if not operator.strip():
            raise HTTPException(422, "请先填写实验员姓名或编号")
        camera = self.camera()
        if self.core.get("RUNTIME_ENABLED"):
            from field_recognition.bindings.binding_operator import resolve_wearer

            with self.core["db"]() as conn:
                wearer_id = resolve_wearer(conn, camera.target, operator.strip(), wearer_id)
            if not wearer_id:
                raise HTTPException(422, "请选择相机已登记的实际使用人员；同名人员需提供 wearer_id")
        with self.lock:
            if self.session and self.session["status"] in ACTIVE:
                raise HTTPException(409, "已有连续扫码正在运行，请先停止")
            self.session = {
                "session_id": str(uuid.uuid4()),
                "status": "scanning",
                "camera_id": camera.target,
                "operator": operator.strip(),
                "owner": owner,
                "wearer_id": wearer_id,
                "started_at": self.core["now"](),
                "frames_scanned": 0,
                "message": "连续扫码中：对准仪器码即可绑定；场景码可单独识别",
                "scan": None,
                "binding": None,
                "last_error": copy.deepcopy(self.last_error),
                "consecutive_decode_errors": 0,
            }
            if owner == "automation":
                self.session["after_frame_id"] = camera.snapshot().get("decoded_frames", 0)
            self.last_seen = time.monotonic()
            self.refresh_bindings()
            self.worker = threading.Thread(
                target=self._run,
                args=(self.session["session_id"],),
                name="continuous-qr",
                daemon=True,
            )
            self.worker.start()
            return copy.deepcopy(self.session)

    def read(self, session_id, *, cancel=False):
        with self.lock:
            if not self.session or self.session["session_id"] != str(session_id):
                raise HTTPException(404, "连续扫码会话不存在，请重新开始")
            self.last_seen = time.monotonic()
            if cancel and self.session["status"] in ACTIVE:
                self.session.update(status="stopped", message="连续扫码已停止，已有绑定继续保留")
            return copy.deepcopy(self.session)

    def _running(self, session_id):
        if self.shutdown.is_set() or not self.session or self.session["session_id"] != session_id:
            return False
        if self.session["status"] not in ACTIVE:
            return False
        if (
            self.session.get("owner") != "automation"
            and time.monotonic() - self.last_seen > LEASE_SECONDS
        ):
            self.session.update(
                status="stopped", message="页面已断开，连续扫码已停止；已有绑定继续保留"
            )
            return False
        return True

    def _run(self, session_id):
        previous, last_full, last_match = None, time.monotonic(), None
        decode_failed_since = None
        while not self.shutdown.wait(0.015):
            with self.lock:
                if not self._running(session_id):
                    return
            try:
                frame, metadata = self.camera().frame()
            except ValueError:
                with self.lock:
                    if self._running(session_id):
                        self.session.update(
                            status="waiting_camera", message="等待相机新画面，恢复连接后继续扫码"
                        )
                self.shutdown.wait(0.2)
                continue
            token = frame_key(metadata)
            if token == previous:
                continue
            previous = token
            acquired = time.monotonic()
            from field_recognition.bindings.binding_policy import moment, ZONE

            observed_day = moment(self.core["now"]()).astimezone(ZONE).date()
            stage = "qr_decode"
            try:
                # Fast native decoding on fresh frames; periodically retry difficult labels.
                full = acquired - last_full >= 1
                try:
                    decoded = decode_qr(frame, fast=not full)
                except (cv2.error, ValueError, RuntimeError) as exc:
                    with self.lock:
                        if not self._running(session_id):
                            return
                        if decode_failed_since is None:
                            decode_failed_since = acquired
                            logger.exception("QR frame decode failed for %s", self.camera().target)
                        self.last_error = {
                            "stage": stage,
                            "error_type": type(exc).__name__,
                            "occurred_at": self.core["now"](),
                            "frame": dict(zip(("decoded_frame_id", "sequence"), token[:2])),
                        }
                        self.session.update(
                            last_error=copy.deepcopy(self.last_error),
                            consecutive_decode_errors=self.session["consecutive_decode_errors"] + 1,
                            message="当前画面扫码失败，正在重试；已有绑定保留",
                        )
                    if acquired - decode_failed_since < QR_DECODE_RECOVERY_SECONDS:
                        self.shutdown.wait(0.2)
                        continue
                    raise
                decode_failed_since = None
                if full:
                    last_full = acquired
                stage = "qr_matches"
                matches = self.core["qr_matches"](*decoded[:2])
                with self.lock:
                    if not self._running(session_id):
                        return
                    if self.session.get("owner") == "automation":
                        info = self.camera().snapshot()
                        if not info.get("service_status_available") or not info.get(
                            "service_status", {}
                        ).get("online"):
                            continue
                        if metadata.get("decoded_frame_id", 0) <= self.session["after_frame_id"]:
                            continue
                    self.session.update(
                        status="scanning",
                        frames_scanned=self.session["frames_scanned"] + 1,
                        last_frame_metadata=metadata,
                        decode_ms=decoded[2]["elapsed_ms"],
                    )
                    if self.session["consecutive_decode_errors"]:
                        self.session["consecutive_decode_errors"] = 0
                        self.refresh_bindings()
                    if self.session["message"] == "等待相机新画面，恢复连接后继续扫码":
                        self.session["message"] = "连续扫码中：对准仪器码即可绑定；场景码可单独识别"
                    # Slow decoding must not commit an old frame as a new observation.
                    if observed_day != moment(self.core["now"]()).astimezone(ZONE).date():
                        continue
                    if time.monotonic() - acquired + metadata.get("frame_age_ms", 0) / 1000 > 1:
                        continue
                    if not (matches["matches"] or matches["scene_matches"]):
                        if matches["unknown"]:
                            self.session["message"] = "读到了未登记二维码，请对准场景或仪器码"
                        continue
                    existing_ids = {b["instrument"]["id"] for b in self.active_bindings()}
                    visits = [
                        v
                        for v in self.core["scene_visits"]()
                        if v["camera_id"] == self.camera().target
                    ]
                    scene_ids = {v["scene"]["id"] for v in visits}
                    new_instruments = [m for m in matches["matches"] if m["id"] not in existing_ids]
                    new_scenes = [m for m in matches["scene_matches"] if m["id"] not in scene_ids]
                    if not new_instruments and not new_scenes:
                        last_match = None
                        self.refresh_bindings()
                        continue
                    key = (observed_day, self.binding_dependencies(new_instruments, new_scenes))
                    if key == last_match:
                        continue
                    last_match = key
                    stage = "accept_scan"
                    self._accept(frame, metadata, decoded, matches)
            except Exception as exc:
                logger.exception(
                    "Continuous QR scan failed at %s for %s", stage, self.camera().target
                )
                with self.lock:
                    if self._running(session_id):
                        self.last_error = {
                            "stage": stage,
                            "error_type": type(exc).__name__,
                            "occurred_at": self.core["now"](),
                        }
                        self.observe_service(
                            {"online": False, "failure_reason": "qr_scanner_error"}
                        )
                        self.session.update(
                            status="failed",
                            message=f"连续扫码失败（{type(exc).__name__}），请重新开始",
                            last_error=copy.deepcopy(self.last_error),
                        )
                return

    def _accept(self, frame, metadata, decoded, matches):
        # Caller holds the same lock used by stop/reset: no late write after stop returns.
        if self.core.get("RUNTIME_ENABLED"):
            with self.core["db"]() as conn:
                active = conn.execute(
                    "SELECT user_id FROM camera_active_users WHERE camera_id=?",
                    (self.camera().target,),
                ).fetchone()
            if active and active[0] != self.session.get("wearer_id"):
                self.session.update(status="stopped", message="实际使用人已交接，请刷新扫码会话")
                return
        result = self.core["save_camera_scan"](
            frame,
            metadata,
            decoded=decoded,
            operator=self.session["operator"],
            scan_session_id=self.session["session_id"],
        )
        self.session["scan"] = result
        errors = []
        # Every identity comes from this decoded frame; each relation is independent.
        for scene in matches["scene_matches"]:
            try:
                visit = self.core["enter_scene"](
                    self.core["SceneEntry"](
                        scan_id=result["scan_id"],
                        scene_id=scene["id"],
                        operator=self.session["operator"],
                        wearer_id=self.session.get("wearer_id"),
                    ),
                    automatic=True,
                )
                self.session["scene_visit"] = visit
            except HTTPException as exc:
                errors.append(str(exc.detail))
        for instrument in matches["matches"]:
            try:
                self.core["bind"](
                    self.core["BindingRequest"](
                        scan_id=result["scan_id"],
                        instrument_id=instrument["id"],
                        operator=self.session["operator"],
                        wearer_id=self.session.get("wearer_id"),
                    ),
                    automatic=True,
                )
            except HTTPException as exc:
                if isinstance(exc.detail, dict):
                    detail = exc.detail
                    errors.append(
                        detail.get("message", "需要处理交接")
                        + (
                            f"：{detail['operator']} · {detail['camera_id']} · 开始于 {detail['started_at']}"
                            if detail.get("operator")
                            else ""
                        )
                    )
                else:
                    errors.append(str(exc.detail))
        self.session.update(status="scanning", binding_errors=errors)
        self.refresh_bindings()

    def observe_service(self, observation):
        """Offline, failed and replaced acquisition sessions release their instruments."""
        from field_recognition.bindings.binding_policy import observe_camera_service

        with self.lock:
            with self.core["db"]() as conn:
                conn.execute("BEGIN IMMEDIATE")
                reason = observe_camera_service(
                    conn, self.camera().target, observation, self.core["now"]()
                )
            if reason and self.session:
                self.session.update(
                    status="stopped",
                    binding=None,
                    bindings=[],
                    scene_visits=[],
                    binding_errors=[],
                    message="设备采集会话已结束，恢复后请重新扫码绑定",
                )
            return reason

    def close(self):
        self.shutdown.set()
        if self.worker:
            self.worker.join(timeout=3)


def preview_part(camera, previous):
    try:
        frame, metadata = camera.frame()
        token = frame_key(metadata)
        if token == previous:
            return None, token
        # Preview is scaled independently; recognition always uses the original image.
        if frame.shape[1] > 960:
            frame = cv2.resize(frame, (960, round(frame.shape[0] * 960 / frame.shape[1])))
    except ValueError:
        token = "offline"
        if previous == token:
            return None, token
        frame = np.full((480, 768, 3), 235, np.uint8)
        cv2.putText(
            frame,
            "Waiting for live camera...",
            (65, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (65, 85, 90),
            2,
        )
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise ValueError("Preview encoding failed")
    jpeg = encoded.tobytes()
    return b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(
        len(jpeg)
    ).encode() + b"\r\n\r\n" + jpeg + b"\r\n", token


def recognition_part(reader, camera, previous):
    from field_recognition.capture.recognition_preview import multipart

    info = camera.snapshot()
    if info.get("service_status_available") and info.get("service_status", {}).get("online"):
        cached = reader.preview(epoch=info.get("service_status", {}).get("media_session_id"))
        if cached:
            frame_token, jpeg = cached
            token = ("recognition", frame_token)
            if token == previous:
                return None, token
            return multipart(jpeg), token
    # Expired recognition frames must not keep a stale panel box on screen.
    raw_previous = previous[1] if isinstance(previous, tuple) and previous[0] == "raw" else None
    part, token = preview_part(camera, raw_previous)
    return part, ("raw", token)


def install(core):
    from field_recognition.runtime.runtime_rpc import camera_component
    import os

    config = core.get("config")
    scanner = camera_component(
        "scanner",
        config.camera_id if config else os.environ.get("FIELD_CAMERA_ID"),
        role=core.get("SERVICE_ROLE"),
        data_dir=core["DATA"] if config is not None else None,
    ) or LiveScanner(core)
    app = core["app"]

    @app.post("/api/camera/scan-sessions")
    def start(body: StartScan):
        from field_recognition.api.security import current

        return scanner.start(
            body.operator, wearer_id=current().user_id if current() else body.wearer_id
        )

    @app.get("/api/camera/scan-sessions/{session_id}")
    def status(session_id: uuid.UUID):
        return scanner.read(session_id)

    @app.delete("/api/camera/scan-sessions/{session_id}")
    def stop(session_id: uuid.UUID):
        with scanner.lock:
            scanner.read(session_id)  # Validate the target before changing automatic mode.
            runner = core.get("automatic_runner")
            if runner and runner.settings()["enabled"]:
                runner.pause()
            return scanner.read(session_id, cancel=True)

    @app.get("/api/camera/preview.mjpg")
    async def preview(request: Request, recognition: bool = False):
        camera = scanner.camera()

        async def frames():
            previous = None
            while not await request.is_disconnected():
                started = time.monotonic()
                from field_recognition.runtime.runtime_rpc import CameraProxy

                if isinstance(camera, CameraProxy):
                    try:
                        part, previous = await run_in_threadpool(
                            camera.preview_part, recognition, previous
                        )
                    except (OSError, HTTPException):
                        part = None
                elif recognition:
                    part, previous = await run_in_threadpool(
                        recognition_part, core["video_ocr"], camera, previous
                    )
                else:
                    part, previous = await run_in_threadpool(preview_part, camera, previous)
                if part:
                    yield part
                # Select the latest frame after each send; slow clients cannot
                # accumulate an application queue. Encoding time is in the budget.
                await asyncio.sleep(max(0, 1 / 30 - (time.monotonic() - started)))

        return StreamingResponse(
            frames(),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    if core["receiver_camera"]:
        core["receiver_camera"].on_service_status = scanner.observe_service
    return scanner
