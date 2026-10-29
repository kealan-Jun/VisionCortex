"""Persistent, browser-independent scanning for the configured local camera."""

import copy
import json
import os
import threading
import time
import uuid

from fastapi import HTTPException
from pydantic import BaseModel, Field

from field_recognition.capture.live_scan import ACTIVE


class AutomationSettings(BaseModel):
    enabled: bool
    operator: str | None = Field(default=None, max_length=80)
    wearer_id: str | None = Field(default=None, max_length=100)
    request_id: uuid.UUID | None = None
    expected_revision: int | None = Field(default=None, ge=0)


class AutomaticRunner:
    def __init__(self, core):
        self.core = core
        self.scanner = core["live_scanner"]
        self.lock = self.scanner.lock
        self.stop = threading.Event()
        self.worker = None
        self.retry_at = 0
        self.failed_session = None
        self.state = {"status": "starting", "message": "正在检查自动运行配置"}
        with core["db"]() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS automation_settings(camera TEXT PRIMARY KEY, document TEXT NOT NULL)"
            )

    def target(self):
        camera = self.core["receiver_camera"]
        config = self.core.get("config")
        if config is not None:
            return config.camera_id or (camera.target if camera else "UnconfiguredNeckCamera")
        return (
            camera.target if camera else os.environ.get("FIELD_CAMERA_ID", "UnconfiguredNeckCamera")
        )

    def settings(self):
        config = self.core.get("config")
        default_enabled = (
            config.auto_run_enabled
            if config is not None
            else os.environ.get("FIELD_AUTO_RUN_ENABLED", "0").lower() in {"1", "true", "yes"}
        )
        with self.core["db"]() as conn:
            row = conn.execute(
                "SELECT document FROM automation_settings WHERE camera=?", (self.target(),)
            ).fetchone()
            settings = (
                json.loads(row["document"])
                if row
                else {
                    "enabled": default_enabled,
                    "operator": "",
                    "registered_at": None,
                    "pause_reason": None,
                }
            )
            if self.core.get("RUNTIME_ENABLED"):
                owner = conn.execute(
                    """SELECT u.id,u.display_name FROM camera_active_users a
                                      JOIN users u ON u.id=a.user_id
                                      WHERE a.camera_id=? AND u.disabled=0""",
                    (self.target(),),
                ).fetchone()
                if not owner:
                    from field_recognition.bindings.binding_operator import camera_revision

                    return settings | {
                        "enabled": False,
                        "operator": "",
                        "wearer_id": None,
                        "revision": camera_revision(conn, self.target()),
                        "pause_reason": "unassigned_camera",
                    }
                settings.update(operator=owner["display_name"], wearer_id=owner["id"])
            from field_recognition.bindings.binding_operator import camera_revision

            settings["revision"] = camera_revision(conn, self.target())
            return settings

    def _save(self, settings):
        with self.core["db"]() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO automation_settings VALUES(?,?)",
                (self.target(), json.dumps(settings)),
            )

    def configure(self, body):
        with self.lock:
            settings = self.settings()
            operator = settings["operator"] if body.operator is None else body.operator.strip()
            if (body.enabled or body.operator is not None) and not operator:
                raise HTTPException(422, "请先登记实验员姓名或编号，再开启自动运行")
            timestamp = self.core["now"]()
            if body.enabled or body.operator is not None:
                from field_recognition.bindings.binding_operator import (
                    activate_member,
                    changed_document,
                    resolve_wearer,
                    update_camera_relations,
                )

                with self.core["db"]() as conn:
                    conn.execute("BEGIN IMMEDIATE")
                    from field_recognition.bindings.binding_operator import (
                        request_fingerprint,
                        replay_request,
                        remember_request,
                        check_revision,
                        camera_revision,
                    )

                    fingerprint = request_fingerprint(
                        {
                            "operation": "automation",
                            "camera_id": self.target(),
                            "body": body.model_dump(mode="json"),
                        }
                    )
                    replay = replay_request(conn, body.request_id, fingerprint)
                    if replay is not None:
                        return replay
                    check_revision(conn, self.target(), body.expected_revision)
                    wearer_id = None
                    if self.core.get("RUNTIME_ENABLED"):
                        requested = body.wearer_id or (
                            settings.get("wearer_id") if body.operator is None else None
                        )
                        wearer_id = resolve_wearer(conn, self.target(), operator, requested)
                        if not wearer_id:
                            raise HTTPException(
                                422, "请选择已登记的相机人员；同名人员请提供 wearer_id"
                            )
                        activate_member(conn, self.target(), wearer_id, timestamp)
                    else:
                        update_camera_relations(
                            conn,
                            self.target(),
                            operator,
                            timestamp=timestamp,
                            reason="camera_operator_setting_changed",
                        )
                    row = conn.execute(
                        "SELECT document FROM automation_settings WHERE camera=?", (self.target(),)
                    ).fetchone()
                    prior = json.loads(row[0]) if row else settings
                    settings, changed = changed_document(
                        prior,
                        operator,
                        wearer_id,
                        timestamp,
                        reason="camera_operator_setting_changed",
                    )
                    if changed or not settings.get("registered_at"):
                        settings["registered_at"] = timestamp
                    settings.update(
                        enabled=body.enabled, pause_reason=None if body.enabled else "user_paused"
                    )
                    settings["revision"] = camera_revision(conn, self.target())
                    conn.execute(
                        "INSERT OR REPLACE INTO automation_settings VALUES(?,?)",
                        (self.target(), json.dumps(settings)),
                    )
                    remember_request(conn, body.request_id, self.target(), fingerprint, settings)
                if self.core.get("receiver_camera"):
                    self.scanner.refresh_bindings()
            else:
                settings.update(enabled=False, pause_reason="user_paused")
                self._save(settings)
            self._stop_scan("自动运行设置已更新")
            self.failed_session, self.retry_at = None, 0
            # A reviewed resume starts a fresh scan, including after a multi-code pause.
            if self.scanner.session and self.scanner.session.get("owner") == "automation":
                self.scanner.session = None
            self.step()
            return self.snapshot()

    def pause(self, reason="user_paused"):
        with self.lock:
            settings = self.settings()
            settings.update(enabled=False, pause_reason=reason)
            self._save(settings)
            self._stop_scan("自动扫码已暂停；已有绑定保留")
            self.state = {
                "status": "paused",
                "message": {
                    "binding_ended": "本次绑定已结束，自动扫码已暂停；下次使用时点击保存并开启",
                    "needs_selection": "二维码需要人工选择，自动扫码已暂停；请核对画面或重新开始",
                }.get(reason, "自动扫码已暂停；已有绑定和照片监控保持原有状态"),
            }

    def _stop_scan(self, message):
        session = self.scanner.session
        if session and session.get("owner") == "automation" and session["status"] in ACTIVE:
            session.update(status="stopped", message=message)

    def snapshot(self):
        with self.lock:
            settings = self.settings()
            session = self.scanner.session
            return copy.deepcopy(
                settings
                | self.state
                | {
                    "binding_ids": [b["binding_id"] for b in self.scanner.active_bindings()]
                    if self.core["receiver_camera"]
                    else [],
                    "camera_id": self.target(),
                    "browser_required": False,
                    "session": session
                    if session and session.get("owner") == "automation"
                    else None,
                }
            )

    def step(self):
        with self.lock:
            if self.stop.is_set():
                return
            from field_recognition.bindings.binding_policy import sweep

            if sweep(self.core):
                self._stop_scan("已跨日，请重新扫码建立今天的绑定")
                self.failed_session, self.retry_at = None, 0
            camera = self.core["receiver_camera"]
            info = camera.snapshot() if camera else {}
            if camera and "service_status_available" in info:
                observation = (
                    info.get("service_status")
                    if (
                        info["service_status_available"]
                        or (info.get("service_status") or {}).get("observed_at")
                    )
                    else {"online": False, "failure_reason": "receiver_status_unavailable"}
                )
                self.scanner.observe_service(observation or {"online": False})
            settings = self.settings()
            if not settings["enabled"]:
                self._stop_scan("自动扫码未开启")
                if self.state["status"] != "paused":
                    self.state = {
                        "status": "paused",
                        "message": "自动扫码已暂停，登记实验员后可开启",
                    }
                return
            if not settings["operator"]:
                self._stop_scan("等待实验员登记")
                self.state = {
                    "status": "waiting_operator",
                    "message": "请先登记实验员姓名或编号，保存后自动扫码",
                }
                return
            camera = self.core["receiver_camera"]
            if camera is None:
                self.state = {"status": "waiting_config", "message": "等待配置挂脖相机接收端"}
                return
            # Never authorize unattended binding from a persisted or unavailable status.
            if not info.get("service_status_available") or not info.get("service_status", {}).get(
                "online"
            ):
                self._stop_scan("等待相机采集服务状态恢复")
                self.state = {
                    "status": "waiting_camera",
                    "message": "等待相机采集服务上线，恢复后自动扫码",
                }
                return
            session = self.scanner.session
            if (
                session
                and session.get("owner") == "automation"
                and (
                    session.get("operator") != settings["operator"]
                    or session.get("wearer_id") != settings.get("wearer_id")
                )
            ):
                self._stop_scan("实际使用人已交接，正在建立新的扫码会话")
                self.scanner.session = None
                session = None
            if session and session.get("owner") != "automation" and session["status"] in ACTIVE:
                self.state = {"status": "manual_scan", "message": "当前由手动扫码会话控制"}
                return
            if session and session.get("owner") == "automation":
                if session["status"] == "needs_selection":
                    self.pause("needs_selection")
                    return
                if session["status"] == "failed" and self.failed_session != session["session_id"]:
                    self.failed_session = session["session_id"]
                    self.retry_at = time.monotonic() + 30
                if time.monotonic() < self.retry_at:
                    self.state = {"status": "retrying", "message": "扫码暂不可用，30 秒后自动重试"}
                    return
                if session["status"] in ACTIVE:
                    self.scanner.refresh_bindings()
                    self.state = {
                        "status": session["status"],
                        "message": session["message"],
                        "binding_ids": [b["binding_id"] for b in session["bindings"]],
                    }
                    return
            session = self.scanner.start(
                settings["operator"], owner="automation", wearer_id=settings.get("wearer_id")
            )
            self.state = {"status": session["status"], "message": session["message"]}

    def start(self):
        if self.worker is not None:
            return
        self.worker = threading.Thread(target=self._run, name="automatic-field-runner", daemon=True)
        self.worker.start()

    def _run(self):
        while not self.stop.is_set():
            try:
                self.step()
            except Exception as exc:
                with self.lock:
                    self._stop_scan("自动运行暂不可用，等待重试")
                    self.state = {
                        "status": "retrying",
                        "message": "自动运行暂不可用，正在等待重试",
                        "error_type": type(exc).__name__,
                    }
                self.stop.wait(5)
            self.stop.wait(1)

    def close(self):
        self.stop.set()
        with self.lock:
            self._stop_scan("本地识别服务正在停止")
        if self.worker:
            self.worker.join(timeout=3)


def install(core):
    from field_recognition.runtime.runtime_rpc import camera_component

    config = core.get("config")
    runner = camera_component(
        "automation",
        config.camera_id if config else os.environ.get("FIELD_CAMERA_ID"),
        role=core.get("SERVICE_ROLE"),
        data_dir=core["DATA"] if config is not None else None,
    ) or AutomaticRunner(core)

    @core["app"].get("/api/automation")
    def state():
        return runner.snapshot()

    @core["app"].put("/api/automation")
    def configure(body: AutomationSettings):
        from field_recognition.api.security import current

        actor = current()
        if actor and (body.enabled or body.operator is not None):
            body = body.model_copy(
                update={"operator": actor.display_name, "wearer_id": actor.user_id}
            )
        return runner.configure(body)

    return runner
