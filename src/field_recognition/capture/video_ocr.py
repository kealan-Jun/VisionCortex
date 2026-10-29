"""Bounded live inference; ordinary frames stay in memory, evidence is event based."""

import copy
import json
import os
import threading
import time

from fastapi import HTTPException
from pydantic import BaseModel

from field_recognition.measurements.readout_rules import READOUT, fallback_reason
from field_recognition.recognition.aliyun_vision import no_digits_seconds
from field_recognition.recognition.config import runtime_recognition_config
from field_recognition.capture.live_scan import frame_key
from field_recognition.capture.qr_decode import decode_qr
from field_recognition.measurements.panel_confirmation import PanelConfirmation


class VideoSettings(BaseModel):
    enabled: bool


class VideoOcr:
    interval = 0.5

    def __init__(self, core, *, clock=time.monotonic):
        self.core, self.clock = core, clock
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.worker = None
        self.pending = None
        self.previous_frame = None
        self.next_sample = 0
        self.signature = None
        self.stable_count = 0
        self.last_saved_signature = None
        self.last_saved = -1000
        self.no_digits_since = None
        self.epoch = None
        self.confirmation = PanelConfirmation()
        # Repeated stable readings also need a fresh frame for later time lookup.
        # Invalid settings fail at startup instead of silently disabling evidence.
        self.periodic_seconds = float(os.environ.get("FIELD_VIDEO_EVIDENCE_INTERVAL_SECONDS", "30"))
        if not 2 <= self.periodic_seconds <= 3600:
            raise ValueError("FIELD_VIDEO_EVIDENCE_INTERVAL_SECONDS must be 2–3600")
        self.preview_cache = None
        self.state = {
            "status": "starting",
            "frames_inferred": 0,
            "evidence_saved": 0,
            "latest_lines": [],
            "last_error": None,
            "background_frames_skipped": 0,
        }

    def enabled(self):
        settings = self.core["automatic_runner"].settings()
        config = self.core.get("config")
        default = (
            config.video_ocr_enabled
            if config is not None
            else os.environ.get("FIELD_VIDEO_OCR_ENABLED", "0").lower() in {"1", "true", "yes"}
        )
        return bool(settings.get("video_ocr_enabled", default))

    def snapshot(self):
        with self.lock:
            return copy.deepcopy(self.state) | {
                "enabled": self.enabled(),
                "sample_interval_seconds": self.interval,
                "target_fps": 1 / self.interval,
                "saves_every_frame": False,
                "stable_samples": 2,
                "stability_required_for_recording": False,
                "observation_policy": "readable_changes/1",
                "periodic_evidence_seconds": self.periodic_seconds,
                "confirmation_rule": "2_of_3_distinct_frames",
                "confirmation_window_seconds": 1.6,
                "min_evidence_interval_seconds": 2,
                "unreadable_evidence_interval_seconds": 30,
                "empty_background_saved": False,
                "unreadable_requires_visible_instrument_qr": not self.core[
                    "panel_detector"
                ].enabled(),
                "panel_detector": self.core["panel_detector"].snapshot(),
            }

    def preview(self, epoch=None):
        with self.lock:
            if (
                not self.preview_cache
                or self.clock() - self.preview_cache[0] > 1.6
                or (epoch is not None and epoch != self.epoch)
                or not self.enabled()
                or self.state["status"]
                in {"paused", "waiting_camera", "waiting_binding", "waiting_operator", "error"}
            ):
                return None
            return self.preview_cache[1:]

    def configure(self, body: VideoSettings):
        runner = self.core["automatic_runner"]
        with runner.lock:
            settings = runner.settings()
            settings["video_ocr_enabled"] = body.enabled
            runner._save(settings)
        if body.enabled:
            self.core["queue_ocr_warmup"]()
        return self.snapshot()

    def start(self):
        self.worker = threading.Thread(target=self._run, name="video-ocr", daemon=True)
        self.worker.start()

    def close(self):
        self.stop.set()
        if self.worker:
            self.worker.join(timeout=3)

    def _run(self):
        while not self.stop.is_set():
            try:
                with self.lock:
                    self.step()
            except Exception as exc:
                with self.lock:
                    self.state.update(status="error", last_error=type(exc).__name__)
            self.stop.wait(0.05)

    def step(self):
        camera = self.core["receiver_camera"]
        if not self.enabled() or not camera:
            self.confirmation.reset()
            self.state["status"] = "paused" if camera else "unconfigured"
            if self.pending and self.pending[0].done():
                self.pending = None
            return
        settings = self.core["automatic_runner"].settings()
        if not settings.get("operator"):
            self.state["status"] = "waiting_operator"
            return
        info = camera.snapshot()
        service = info.get("service_status", {})
        if not info.get("service_status_available") or not service.get("online"):
            self.confirmation.reset()
            self.state["status"] = "waiting_camera"
            self.no_digits_since = None
            return
        epoch = service.get("media_session_id")
        if epoch != self.epoch:
            self.confirmation.reset()
            self.preview_cache = None
            self.epoch, self.signature, self.last_saved_signature = epoch, None, None
            self.no_digits_since, self.stable_count = None, 0
        snapshots = (
            self.core["current_panel_bindings"](camera.target)
            if self.core["panel_detector"].enabled()
            else None
        )
        if snapshots == []:
            self.confirmation.reset()
            if self.pending:
                self.pending[0].cancel()
                if self.pending[0].done():
                    self.pending = None
            self.state.update(status="waiting_binding", latest_lines=[], latest_panels=[])
            self.no_digits_since, self.stable_count, self.signature = None, 0, None
            return
        if self.pending:
            future, frame, metadata, observed, old_epoch, sampled = self.pending
            if not future.done():
                self.state["status"] = "inferring"
                return
            self.pending = None
            local = future.result()
            if old_epoch == epoch and self.clock() - sampled < 3 and not self.stop.is_set():
                self.accept(frame, metadata, observed, local)
        if self.clock() < self.next_sample:
            return
        # A video worker owns at most one OCR future; saved photographs take priority.
        with self.core["db"]() as conn:
            priority = conn.execute(
                "SELECT 1 FROM jobs WHERE status='queued' OR (status='running' "
                "AND json_extract(document,'$.phase')='local_ocr') LIMIT 1"
            ).fetchone()
        photos = self.core["saved_photo_watcher"].snapshot()
        if (
            priority
            or photos.get("ready_files", photos.get("pending_files", 0))
            or photos.get("importing_file")
        ):
            self.state["status"] = "yielding_to_photo"
            return
        if self.core["ocr_state"]["status"] != "ready":
            self.state["status"] = (
                "model_error" if self.core["ocr_state"]["status"] == "error" else "loading_model"
            )
            self.core["queue_ocr_warmup"]()
            return
        try:
            frame, metadata = camera.frame()
        except ValueError:
            self.confirmation.reset()
            self.state["status"] = "waiting_camera"
            self.no_digits_since, self.stable_count = None, 0
            return
        token = frame_key(metadata)
        if token == self.previous_frame or metadata.get("frame_age_ms", 0) > 1000:
            return
        self.previous_frame = token
        self.next_sample = self.clock() + self.interval
        observed = self.core["now"]()
        future = self.core["ocr_pool"].submit(
            self.core["predict_readout"], frame, 0, 0, video=True, binding_snapshots=snapshots
        )
        self.pending = (future, frame, metadata, observed, epoch, self.clock())
        self.state.update(status="inferring", last_sample_at=observed)

    def accept(self, frame, metadata, observed, local):
        current = self.clock()
        if "panel_detection" in local:
            bindings = self.core["current_panel_bindings"](self.core["receiver_camera"].target)
            allowed = {b["instrument"]["id"] for b in bindings}
            # Discard a result whose binding ended during inference.
            if any(r["instrument_id"] not in allowed for r in local.get("panel_regions", [])):
                self.confirmation.reset()
                self.preview_cache = None
                self.state.update(latest_lines=[], latest_panels=[])
                self.no_digits_since, self.stable_count, self.signature = None, 0, None
                return
        lines = [
            line
            for line in local.get("lines", [])
            if not line.get("quality_issue")
            and not line.get("quality_issues")
            and READOUT.fullmatch(line.get("text", ""))
        ]
        decoded = decode_qr(frame, fast=True)
        hits = self.core["qr_matches"](*decoded[:2])["matches"]
        panels = local.get("panel_regions", [])
        localized = "panel_detection" in local
        visible_ids = tuple(sorted(hit["id"] for hit in hits))
        if localized:
            visible_ids = tuple(sorted({r["instrument_id"] for r in panels}))
            lines = self.confirmation.observe(
                panels,
                frame_key(metadata),
                observed,
                current,
                (self.epoch, tuple(sorted(b["binding_id"] for b in bindings))),
            )
            self.state["pending_panels"] = sum(
                r["temporal_confirmation"]["status"] == "pending"
                and any(
                    READOUT.fullmatch(l.get("text", "")) for l in r["local_ocr"].get("lines", [])
                )
                for r in panels
            )
        elif not visible_ids:
            with self.core["db"]() as conn:
                visible_ids = tuple(
                    sorted(
                        json.loads(row["document"])["instrument"]["id"]
                        for row in conn.execute(
                            "SELECT document FROM bindings WHERE camera=? AND ended IS NULL",
                            (self.core["receiver_camera"].target,),
                        )
                    )
                )
        panel_map = {r["panel_id"]: r for r in panels}
        signature = (
            tuple(
                sorted(
                    (panel_map.get(line.get("panel_id"), {}).get("instrument_id", ""), line["text"])
                    for line in lines
                )
            ),
            visible_ids,
        )
        self.stable_count = self.stable_count + 1 if signature == self.signature else 1
        self.signature = signature
        self.state.update(
            status="watching",
            frames_inferred=self.state["frames_inferred"] + 1,
            latest_lines=lines,
            last_result_at=self.core["now"](),
            last_error=local.get("detector_error") or local.get("error"),
            latest_qr_instruments=[{k: hit[k] for k in ("id", "name")} for hit in hits],
            last_ocr_seconds=local.get("wall_seconds"),
            stable_count=self.stable_count,
            latest_frame_metadata=metadata,
        )
        self.state["latest_panels"] = [
            {
                k: r[k]
                for k in ("panel_id", "class_id", "instrument_id", "bbox", "detector_confidence")
            }
            for r in panels
        ]
        from field_recognition.capture.recognition_preview import render

        preview_panels = [
            region
            | {"instrument_name": self.core["get_instrument"](region["instrument_id"])["name"]}
            for region in panels
        ]
        self.preview_cache = (current, frame_key(metadata), render(frame, preview_panels))
        for line in lines:
            region = panel_map.get(line.get("panel_id"))
            if region:
                line["instrument"] = self.core["get_instrument"](region["instrument_id"])
        # An absent reading is not itself a reason to store the room, cables or desk.
        # A session binding alone does not establish that an instrument is in this frame.
        if not lines and not (panels if localized else hits):
            self.no_digits_since = None
            self.state["background_frames_skipped"] += 1
            return
        from field_recognition.recognition.panel_regions import needs_fallback

        reason = needs_fallback(local)
        if reason:
            if self.no_digits_since is None:
                self.no_digits_since = current
        else:
            self.no_digits_since = None
        changed_panels = (
            [
                r
                for r in panels
                if self.confirmation.changed(r) and self.confirmation.can_save(r, current)
            ]
            if localized
            else []
        )
        change = (
            bool(changed_panels)
            if localized
            else bool(lines) and signature != self.last_saved_signature
        )
        periodic_panels = (
            [r for r in panels if self.confirmation.periodic_due(r, current, self.periodic_seconds)]
            if localized
            else []
        )
        periodic = (
            bool(periodic_panels)
            if localized
            else bool(lines) and current - self.last_saved >= self.periodic_seconds
        )
        unreadable = (
            self.no_digits_since is not None
            and current - self.no_digits_since
            >= no_digits_seconds(runtime_recognition_config(self.core))
            and current - self.last_saved >= 30
            and self.stable_count >= 2
            and signature != self.last_saved_signature
        )
        if (
            not (change and (localized or current - self.last_saved >= 2))
            and not periodic
            and not unreadable
        ):
            return
        evidence_panels = {r["panel_id"]: r for r in changed_panels + periodic_panels}
        if localized and (change or periodic):
            local = copy.deepcopy(local)
            selected = set(evidence_panels)
            local["panel_regions"] = [
                r for r in local["panel_regions"] if r["panel_id"] in selected
            ]
            local["lines"] = [l for l in lines if l.get("panel_id") in selected]
            for region in local["panel_regions"]:
                region["local_ocr"]["lines"] = [
                    l for l in local["lines"] if l.get("panel_id") == region["panel_id"]
                ]
        # Readable changes are observations even before temporal agreement.
        # Keep the existing rate/queue bounds and reuse this exact frame's OCR.
        with self.core["ocr_submit_lock"]:
            with self.core["db"]() as conn:
                if (
                    conn.execute(
                        "SELECT count(*) FROM jobs WHERE status IN ('queued','running')"
                    ).fetchone()[0]
                    >= 4
                ):
                    return
            import cv2

            ok, encoded = cv2.imencode(".png", frame)
            if not ok:
                raise ValueError("video evidence encoding failed")
            capture = self.core["scan_image"](
                encoded.tobytes(),
                "neck_camera_video_ocr",
                self.core["receiver_camera"].target,
                decoded=decoded,
                metadata=metadata,
                operator=self.core["automatic_runner"].settings()["operator"],
            )
            elapsed = current - self.no_digits_since if self.no_digits_since is not None else 0
            from field_recognition.measurements.measurement_records import video_clock

            capture.update(
                received_at=observed,
                video_observation={
                    **video_clock(metadata),
                    "observed_at": observed,
                    "evidence_reason": "reading_changed"
                    if change
                    else "reading_periodic"
                    if periodic
                    else "unreadable_timeout",
                    "periodic_evidence_seconds": self.periodic_seconds,
                    "observation_policy": "readable_changes/1",
                    "stability_required_for_recording": False,
                    "stable_samples": min(
                        r["temporal_confirmation"]["votes"] for r in evidence_panels.values()
                    )
                    if evidence_panels
                    else self.stable_count,
                    "confirmation_rule": "2_of_3_distinct_frames"
                    if localized and (change or periodic)
                    else None,
                    "no_digits_elapsed_seconds": elapsed,
                    "sample_interval_seconds": self.interval,
                },
            )
            with self.core["db"]() as conn:
                conn.execute(
                    "UPDATE scans SET document=? WHERE id=?",
                    (json.dumps(capture), capture["capture_id"]),
                )
            job = self.core["enqueue_ocr"](
                self.core["OcrRequest"](capture_id=capture["capture_id"], auto_associate=True),
                trigger="video_stream",
                precomputed_local=local,
            )
            self.last_saved, self.last_saved_signature = current, signature
            if localized:
                self.confirmation.mark_saved(local["panel_regions"], current)
            self.state.update(
                last_job_id=job["job_id"],
                last_evidence_at=self.core["now"](),
                evidence_saved=self.state["evidence_saved"] + 1,
            )


def install(core):
    from field_recognition.runtime.runtime_rpc import camera_component

    config = core.get("config")
    reader = camera_component(
        "video",
        config.camera_id if config else os.environ.get("FIELD_CAMERA_ID"),
        role=core.get("SERVICE_ROLE"),
        data_dir=core["DATA"] if config is not None else None,
    ) or VideoOcr(core)

    @core["app"].get("/api/video-ocr")
    def status():
        return reader.snapshot()

    @core["app"].put("/api/video-ocr")
    def configure(body: VideoSettings):
        return reader.configure(body)

    return reader
