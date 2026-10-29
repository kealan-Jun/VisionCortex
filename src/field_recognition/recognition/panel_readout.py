"""One user-requested photograph: resident local OCR first, timed vision fallback."""

import hashlib
import copy
import json
import os
import time
from concurrent.futures import Future
from functools import partial

import cv2

import field_recognition.recognition.aliyun_vision as vision
from field_recognition.recognition.panel_regions import needs_fallback, finish as finish_regions
from field_recognition.recognition.config import runtime_recognition_config
from field_recognition.recognition.pipeline_context import ReadoutPipeline
from field_recognition.measurements.readout_rules import READOUT, fallback_reason
from field_recognition.measurements.readout_timing import update_timing, record_duration


def save(core, document):
    if document["status"] in {"completed", "failed", "cancelled", "interrupted"}:
        shutdown = bool(core.get("stopping") and core["stopping"].is_set())
        document["resume_pending"] = shutdown and document.get("request_trigger") != "video_stream"
        if document["resume_pending"]:
            document["status"] = "interrupted"
    update_timing(document)
    if document.get("lease_holder"):
        from field_recognition.measurements.photo_measurements import refresh

        callback = (
            (lambda conn: refresh(core, conn, document["measurement_id"]))
            if document.get("measurement_id")
            else None
        )
        if document["status"] == "failed":
            if document.get("error") in {
                "InferenceUnavailable",
                "ImportError",
                "ModuleNotFoundError",
            }:
                core["task_queue"].defer_task(
                    document["job_id"], document=document, error=document["error"]
                )
                return
            transient = document.get("error") in {
                "local_ocr_timeout",
                "TimeoutError",
                "OSError",
                "ConnectionError",
                "worker_unavailable",
            }
            core["task_queue"].fail_task(
                document["job_id"],
                document.get("error", "ocr_failed"),
                retry=transient,
                document=document,
                refresh=callback,
            )
        else:
            core["task_queue"].publish(document, refresh=callback)
        return
    with core["db"]() as conn:
        conn.execute(
            "UPDATE jobs SET status=?,document=? WHERE id=?",
            (document["status"], json.dumps(document), document["job_id"]),
        )
        if document.get("measurement_id"):
            from field_recognition.measurements.photo_measurements import refresh

            refresh(core, conn, document["measurement_id"])


def acquire_primary_slot(core, valid, *, clock=time.monotonic, pause=time.sleep):
    """Each requested photo gets a bounded turn; no burst is dropped as busy/cooldown."""
    deadline = clock() + 120
    while valid():
        if core["vision_call_lock"].acquire(blocking=False):
            return True
        if clock() >= deadline:
            return False
        pause(min(0.1, deadline - clock()))
    return False


def pipeline_context(core):
    return ReadoutPipeline(
        save_document=partial(save, core),
        acquire_primary_slot=partial(acquire_primary_slot, core),
        recognition=runtime_recognition_config(core),
    )


def primary_unbound_photo(core, document, panel, *, valid, clock, pause, started, pipeline=None):
    """Direct model transcription, explicitly without physical instrument attribution."""
    pipeline = pipeline or pipeline_context(core)
    from field_recognition.recognition.vision_display_rules import apply_rule

    rules = pipeline.display_rules()
    document.update(
        phase="vision_primary",
        reading_mode="vision_primary_photo",
        vision_display_rules=rules,
        fallback={"status": "waiting", "trigger": "primary_photo"},
    )
    pipeline.save_document(document)
    slot_started = clock()
    acquired = acquire_primary_slot(core, valid, clock=clock, pause=pause)
    record_duration(document, "vision_queue_wait_ms", slot_started, clock())
    if not acquired:
        document["fallback"].update(status="failed", error="vision_slot_timeout")
        return False
    try:
        if not valid():
            return False
        document["fallback"].update(
            status="running",
            attempted_at=core["now"](),
            trigger_elapsed_seconds=round(clock() - started, 3),
        )
        pipeline.save_document(document)
        vision_started = clock()
        cloud = pipeline.read_panel(panel, display_rules=rules)
        record_duration(document, "vision_ms", vision_started, clock())
        document["fallback"].update(cloud, finished_at=core["now"]())
    finally:
        core["vision_call_lock"].release()
    if cloud.get("status") != "completed":
        return False  # Preserve the provider failure and try local OCR once.
    readings = cloud["answer"]["readings"]
    states = cloud["answer"].get("display_states", [])
    document["capture_association"] = {
        k: document.get(k)
        for k in ("instrument", "binding_id", "instrument_candidates", "workbench")
    }
    document.update(
        status="completed",
        phase="completed",
        model=cloud["model"],
        device="cloud",
        actual_model_invocation=True,
        local_ocr={"status": "skipped", "reason": "vision_primary_photo", "lines": []},
        instrument=None,
        binding_id=None,
        binding_ids=[],
        instrument_candidates=[],
        panel_regions=[],
        instrument_association="unbound_photo",
        instrument_identity_basis="unconfirmed",
        association_status="unbound",
        lines=[apply_rule(line, rules) for line in vision.answer_lines(cloud["answer"])],
        outcome="numeric_readout"
        if readings
        else "display_state"
        if states
        else "no_numeric_readout",
    )
    document.pop("error", None)
    return True


def recover_primary_photo(core, document, panel, *, valid, clock, pause, started, pipeline=None):
    """Recover missing panel localization as unassigned photo candidates only."""
    pipeline = pipeline or pipeline_context(core)
    if (
        not pipeline.primary_photos()
        or not pipeline.public_config()["available"]
        or document.get("request_trigger") == "video_stream"
        or not document.get("all_binding_snapshots", document.get("binding_snapshots", []))
    ):
        return False
    if any(r.get("display_state") for r in document.get("panel_regions", [])):
        return False
    if any(
        line.get("value") is not None or READOUT.fullmatch(line.get("text", ""))
        for line in document.get("lines", [])
    ):
        return False
    # The failed crop/region path remains independently inspectable. Whole-photo
    # transcription cannot inherit that path's physical instrument identity.
    previous = copy.deepcopy(document)
    evidence_keys = (
        "panel_detection",
        "panel_regions",
        "local_ocr",
        "fallback",
        "lines",
        "instrument",
        "binding_id",
        "instrument_candidates",
        "workbench",
        "recognition_skipped",
        "skip_reason",
        "failure_reason",
        "error",
    )
    evidence = {key: copy.deepcopy(document[key]) for key in evidence_keys if key in document}
    if primary_unbound_photo(
        core,
        document,
        panel,
        valid=valid,
        clock=clock,
        pause=pause,
        started=started,
        pipeline=pipeline,
    ):
        document["localized_attempt"] = evidence
        document["whole_photo_recovery"] = {
            "status": "completed",
            "reason": "no_localized_readings",
            "association": "unbound_candidates",
        }
        for key in (
            "panel_detection",
            "recognition_skipped",
            "skip_reason",
            "failure_reason",
            "detector_error",
        ):
            document.pop(key, None)
        document["recognition_skipped"] = False
        return True
    recovery = copy.deepcopy(document.get("fallback", {}))
    document.clear()
    document.update(
        previous,
        whole_photo_recovery={
            "status": "failed",
            "reason": "no_localized_readings",
            "fallback": recovery,
        },
    )
    return False


def submit_local(core, document, panel, x, y, *, clock=time.monotonic, detection=None):
    precomputed = document.pop("precomputed_local", None)
    if precomputed is not None:
        future = Future()
        future.set_result(precomputed)
    else:
        predictor = (
            partial(
                core["predict_readout"],
                binding_snapshots=document.get(
                    "all_binding_snapshots", document.get("binding_snapshots", [])
                ),
                field_rules=document.get("field_rules"),
                **({"detection": detection} if detection is not None else {}),
            )
            if "predict_readout" in core
            else core["predict_panel"]
        )
        scheduled = clock()
        scheduled_at = core["now"]()

        def predict_request(pixels, offset_x, offset_y):
            from field_recognition.recognition.remote_ocr import inference_request

            execution_started = clock()
            execution_started_at = core["now"]()
            with inference_request(document["job_id"]):
                result = dict(predictor(pixels, offset_x, offset_y))
            result.update(
                ocr_scheduled_at=scheduled_at,
                ocr_execution_started_at=execution_started_at,
                ocr_execution_finished_at=core["now"](),
                ocr_scheduler_wait_ms=round((execution_started - scheduled) * 1000, 3),
                ocr_execution_ms=round((clock() - execution_started) * 1000, 3),
                ocr_execution_scope="predict_readout_with_prepared_localization"
                if detection is not None
                else "predict_readout_including_detection_preprocessing_and_model_wait",
            )
            return result

        future = core["ocr_pool"].submit(predict_request, panel, x, y)
    return future


def run(core, document, *, clock=time.monotonic, pause=time.sleep):
    pipeline = pipeline_context(core)
    started = clock()
    future = None
    local = None
    cloud = None
    document.update(
        status="running",
        phase="local_ocr",
        started_at=core["now"](),
        reading_mode="one_requested_photo",
        no_digits_timeout_seconds=pipeline.no_digits_seconds(),
    )
    save(core, document)

    def valid():
        return core["current_readout_binding"](document)

    def result_if_ready():
        return future.result() if future.done() and not future.cancelled() else None

    def cancelled():
        document.update(status="cancelled", error="binding_ended_or_changed", lines=[])

    try:
        if not valid():
            cancelled()
            return
        image = cv2.imread(str(core["DATA"] / "Images" / f"{document['capture_id']}.png"))
        x, y, w, h = document["crop"]
        panel = image[y : y + h, x : x + w].copy()
        path = core["DATA"] / "Images" / f"{document['job_id']}.png"
        if not cv2.imwrite(str(path), panel):
            raise ValueError("Panel evidence encoding failed")
        document.update(
            crop_image_url=f"/api/images/{document['job_id']}",
            crop_image_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            accuracy="not_human_verified",
            panel_selection="user_selected_crop_or_full_photo",
        )
        configured = pipeline.public_config()["available"]
        if (
            configured
            and pipeline.public_config()["photo_mode"] == "parallel"
            and document.get("request_trigger") != "video_stream"
        ):
            from field_recognition.recognition.parallel_photo import run as run_parallel

            future = run_parallel(
                core,
                document,
                image,
                panel,
                x,
                y,
                pipeline=pipeline,
                valid=valid,
                clock=clock,
                pause=pause,
                started=started,
            )
            return
        snapshots = document.get("all_binding_snapshots", document.get("binding_snapshots", []))
        primary_attempted = bool(
            configured
            and pipeline.primary_photos()
            and document.get("request_trigger") != "video_stream"
            and not snapshots
            and not document.get("binding_id")
        )
        if primary_attempted:
            completed = primary_unbound_photo(
                core,
                document,
                panel,
                valid=valid,
                clock=clock,
                pause=pause,
                started=started,
                pipeline=pipeline,
            )
            if not valid():
                cancelled()
                return
            if completed:
                return
            # One cloud request per attempt. Its failure receipt survives the local fallback.
            configured = False
            document.update(phase="local_ocr_after_vision_failure")
        future = submit_local(core, document, panel, x, y, clock=clock)
        video_elapsed = (document.get("video_observation") or {}).get(
            "no_digits_elapsed_seconds", 0
        )
        deadline = started + (
            max(0, pipeline.no_digits_seconds() - video_elapsed) if configured else 90
        )
        # A successful numeric result returns immediately. A miss cannot trigger a paid
        # request before the deadline, and slow local inference cannot block the watchdog.
        while True:
            if not valid():
                cancelled()
                return
            local = result_if_ready()
            if local and local.get("recognition_skipped"):
                break
            if local and (
                not configured
                or needs_fallback(local) is None
                or (pipeline.primary_photos() and document.get("request_trigger") != "video_stream")
            ):
                break
            # An unavailable remote worker must return a durable wait, not trigger
            # a cloud call or finalize an empty localization before it responds.
            remote = (
                pipeline.recognition.ocr_service_url
                if pipeline.recognition is not None
                else os.environ.get("FIELD_OCR_SERVICE_URL")
            )
            remaining = (
                max(deadline, started + 75) if remote and local is None else deadline
            ) - clock()
            if remaining <= 0:
                break
            if local and document["phase"] != "waiting_readout":
                document.update(phase="waiting_readout", local_ocr=local)
                save(core, document)
            pause(min(0.1, remaining))
        local = result_if_ready()
        if local and local.get("recognition_skipped"):
            document.update(local)
            document.update(
                local_ocr=local,
                outcome="processing_failed"
                if local.get("status") == "failed"
                else "skipped_unbound_or_unlocalized_panel",
                phase="completed",
            )
            recover_primary_photo(
                core,
                document,
                panel,
                valid=valid,
                clock=clock,
                pause=pause,
                started=started,
                pipeline=pipeline,
            )
            if not valid():
                cancelled()
            return
        if local and local.get("panel_regions"):
            finish_regions(
                core,
                document,
                local,
                image,
                valid=valid,
                clock=clock,
                started=started,
                pipeline=pipeline,
            )
            if document.get("status") != "cancelled":
                recover_primary_photo(
                    core,
                    document,
                    panel,
                    valid=valid,
                    clock=clock,
                    pause=pause,
                    started=started,
                    pipeline=pipeline,
                )
            if not valid():
                cancelled()
            return
        if (
            core.get("panel_detector")
            and core["panel_detector"].enabled()
            and not (local and local.get("panel_selection") == "unbound_photo_text")
        ):
            # Missing/slow localization must never fall back to reading the entire
            # frame: that could include an unbound laptop or another instrument.
            document.update(
                status="failed",
                error="bound_panel_localization_unavailable",
                lines=[],
                fallback={"status": "skipped", "reason": "no_verified_bound_panel"},
                panel_detection={"status": "unavailable"},
                panel_regions=[],
            )
            recover_primary_photo(
                core,
                document,
                panel,
                valid=valid,
                clock=clock,
                pause=pause,
                started=started,
                pipeline=pipeline,
            )
            if not valid():
                cancelled()
            return
        reason = (
            fallback_reason(local["lines"], local.get("error")) if local else "local_ocr_timeout"
        )
        if reason and configured:
            if not valid():
                cancelled()
                return
            fallback_scope = document.get("binding_id") or "unbound-camera:" + document["camera_id"]
            # Timers for separate photos can overlap, but paid requests must not.
            # A busy provider slot does not consume the persistent cooldown.
            if not core["vision_call_lock"].acquire(blocking=False):
                document["fallback"] = {"status": "skipped", "reason": "busy"}
            else:
                try:
                    if not valid():
                        cancelled()
                        return
                    if core["reserve_fallback"](fallback_scope):
                        document.update(
                            phase="extended_reading",
                            fallback={
                                "status": "running",
                                "trigger": reason,
                                "trigger_elapsed_seconds": round(
                                    clock() - started + video_elapsed, 3
                                ),
                                "attempted_at": core["now"](),
                            },
                        )
                        save(core, document)
                        vision_started = clock()
                        cloud = pipeline.read_panel(panel)
                        record_duration(document, "vision_ms", vision_started, clock())
                        document["fallback"].update(cloud)
                        document["fallback"]["finished_at"] = core["now"]()
                    else:
                        document["fallback"] = {
                            "status": "skipped",
                            "reason": "cooldown",
                            "cooldown_seconds": 30,
                        }
                finally:
                    core["vision_call_lock"].release()
        local = result_if_ready()
        document["local_ocr"] = local or {
            "status": "running",
            "lines": [],
            "model": core["ocr_state"]["engine"],
        }
        if not valid():
            cancelled()
            return
        if local:
            document.update(local)
        else:
            document.update(status="failed", error="local_ocr_timeout", lines=[])
        # When local OCR finishes during the vision call, prefer its valid numeric result.
        local_has_digits = local and fallback_reason(local["lines"], local.get("error")) is None
        cloud_lines = vision.answer_lines(cloud.get("answer", {})) if cloud else []
        if cloud_lines and not local_has_digits:
            document.update(
                status="completed",
                model=cloud["model"],
                device="cloud",
                actual_model_invocation=True,
                lines=cloud_lines,
            )
            document.pop("error", None)
        document["outcome"] = (
            "numeric_readout"
            if local_has_digits or (cloud and cloud.get("answer", {}).get("readings"))
            else "display_state"
            if cloud_lines
            else "no_numeric_readout"
        )
    except Exception as exc:
        document.update(status="failed", error=type(exc).__name__, lines=[])
    finally:
        if document.get("panel_detection") and not document.get("panel_regions"):
            document["capture_association"] = {
                k: document.get(k)
                for k in ("instrument", "binding_id", "instrument_candidates", "workbench")
            }
            document.update(
                instrument=None,
                binding_id=None,
                instrument_candidates=[],
                binding_ids=[],
                instrument_association="unbound_photo",
                instrument_identity_basis="not_localized",
                association_status="unlocalized",
            )
        generation_started = clock()
        document["result_generation_started_at"] = core["now"]()
        from field_recognition.measurements.reading_results import build_readings

        document["readings"] = build_readings(document)
        from field_recognition.measurements.measurement_records import build_records, display_fields

        document["measurement_records"] = build_records(document)
        document["display_fields"] = display_fields(document)
        document["result_generated_at"] = core["now"]()
        record_duration(document, "result_generation_ms", generation_started, clock())
        document.update(finished_at=core["now"](), wall_seconds=round(clock() - started, 3))
        record_duration(document, "processing_ms", started, clock())
        if future and not document.get("local_ocr"):
            document["local_ocr"] = result_if_ready() or {"status": "running", "lines": []}
        save(core, document)
        if future and document.get("local_ocr", {}).get("status") == "running":
            # An in-flight OCR operation cannot be killed safely. Its eventual raw
            # result is kept for audit, without replacing the answer already delivered.
            def late_result(completed):
                if completed.cancelled():
                    late = {"status": "cancelled", "lines": []}
                else:
                    try:
                        late = completed.result()
                    except Exception as exc:
                        late = {"status": "failed", "error": type(exc).__name__, "lines": []}
                with core["db"]() as conn:
                    conn.execute("BEGIN IMMEDIATE")
                    row = conn.execute(
                        "SELECT document FROM jobs WHERE id=?", (document["job_id"],)
                    ).fetchone()
                    if not row:
                        return
                    stored = json.loads(row["document"])
                    if (
                        document.get("lease_holder")
                        and stored.get("lease_holder") != document["lease_holder"]
                    ):
                        return  # A late attempt cannot replace a newer attempt's evidence.
                    if stored.get("phase") in {"retry_wait", "replayed"}:
                        return
                    stored.update(local_ocr=late, local_ocr_finished_late=True)
                    update_timing(stored)
                    conn.execute(
                        "UPDATE jobs SET document=? WHERE id=?",
                        (json.dumps(stored), document["job_id"]),
                    )

            future.add_done_callback(late_result)
            future.cancel()
