"""Measured stage times; NAS wall-clock durations explicitly retain their basis."""

from datetime import datetime
import math


def record_duration(document, name, started, finished):
    """Store a duration measured in one process; never persist monotonic epochs."""
    duration = (finished - started) * 1000
    if math.isfinite(duration) and duration >= 0:
        document.setdefault("timing", {}).setdefault("monotonic_durations_ms", {})[name] = round(
            duration, 2
        )


def elapsed_ms(start, end):
    if not start or not end:
        return None
    try:
        first, last = datetime.fromisoformat(start), datetime.fromisoformat(end)
        if first.tzinfo is None or last.tzinfo is None:
            return None
        value = (last - first).total_seconds() * 1000
    except (ValueError, TypeError, OverflowError):
        return None
    return round(value, 2) if value >= 0 else None


def update_timing(document, *, preserve_recorded_durations=False):
    timing = document.setdefault("timing", {})
    timing["task_id"] = document.get("job_id")
    timing["group_id"] = (document.get("measurement_context") or {}).get(
        "burst_id"
    ) or document.get("measurement_id")
    timing["queued_at"] = document["submitted_at"]
    for source, target in [
        ("started_at", "run_started_at"),
        ("finished_at", "result_finished_at"),
        ("result_generation_started_at", "result_generation_started_at"),
        ("result_generated_at", "result_generated_at"),
        ("archive_published_at", "archive_published_at"),
        ("archive_readable_at", "archive_readable_at"),
        ("first_result_at", "first_result_at"),
        ("result_updated_at", "result_updated_at"),
    ]:
        if document.get(source):
            timing[target] = document[source]
    # Archive status is joined at API/readout time for historical jobs. Keep
    # those timestamps in the same timing namespace when a caller supplies it.
    archive = document.get("archive") or {}
    for source, target in [
        ("published_at", "archive_published_at"),
        ("readable_at", "archive_readable_at"),
    ]:
        if archive.get(source):
            timing[target] = archive[source]
    local = document.get("local_ocr") or {}
    detection = local.get("panel_detection") or document.get("panel_detection") or {}
    for key in ("started_at", "finished_at"):
        if detection.get(key):
            timing["panel_detection_" + key] = detection[key]
    video = document.get("video_observation") or {}
    if video.get("observed_at"):
        timing["frame_observed_at"] = video["observed_at"]
    for key in (
        "ocr_started_at",
        "ocr_finished_at",
        "ocr_scheduled_at",
        "ocr_execution_started_at",
        "ocr_execution_finished_at",
    ):
        if local.get(key):
            timing[key] = local[key]
    fallback = document.get("fallback") or {}
    if fallback.get("attempted_at"):
        timing["vision_started_at"] = fallback["attempted_at"]
    if fallback.get("finished_at"):
        timing["vision_finished_at"] = fallback["finished_at"]
    pairs = {
        "panel_detection_ms": ("panel_detection_started_at", "panel_detection_finished_at"),
        "write_to_detect_ms": ("source_written_at", "first_observed_at"),
        "file_stability_ms": ("first_observed_at", "stable_at"),
        "ingest_wait_ms": ("stable_at", "read_started_at"),
        "file_read_ms": ("read_started_at", "photo_read_at"),
        "import_ms": ("photo_read_at", "imported_at"),
        "queue_wait_ms": ("queued_at", "run_started_at"),
        "ocr_wait_ms": ("run_started_at", "ocr_started_at"),
        "ocr_ms": ("ocr_started_at", "ocr_finished_at"),
        "ocr_scheduler_wait_ms": ("ocr_scheduled_at", "ocr_execution_started_at"),
        "ocr_execution_ms": ("ocr_execution_started_at", "ocr_execution_finished_at"),
        "processing_ms": ("run_started_at", "result_finished_at"),
        "result_generation_ms": ("result_generation_started_at", "result_generated_at"),
        "result_finalize_ms": ("result_generated_at", "result_finished_at"),
        "vision_ms": ("vision_started_at", "vision_finished_at"),
        "write_to_ocr_ms": ("source_written_at", "ocr_started_at"),
        "write_to_result_ms": ("source_written_at", "result_finished_at"),
        "write_to_readable_result_ms": ("source_written_at", "archive_readable_at"),
        "archive_publication_ms": ("result_finished_at", "archive_published_at"),
        "archive_readability_ms": ("archive_published_at", "archive_readable_at"),
        "detect_to_result_ms": ("first_observed_at", "result_finished_at"),
        "frame_to_result_ms": ("frame_observed_at", "result_finished_at"),
        "write_to_first_result_ms": ("source_written_at", "first_result_at"),
        "run_to_first_result_ms": ("run_started_at", "first_result_at"),
    }
    recorded = (timing.get("durations_ms") or {}) if preserve_recorded_durations else {}
    timing["durations_ms"] = {
        name: elapsed_ms(timing[start], timing[end])
        if timing.get(start) and timing.get(end)
        else recorded.get(name)
        for name, (start, end) in pairs.items()
    }
    timing["duration_clock_basis"] = {
        name: "wall_clock_timestamp_difference"
        for name, value in timing["durations_ms"].items()
        if value is not None
    }
    measured = dict(timing.get("monotonic_durations_ms") or {})
    for name in (
        "ocr_scheduler_wait_ms",
        "ocr_execution_ms",
        "ocr_model_lock_wait_ms",
        "ocr_model_prepare_ms",
        "ocr_model_call_ms",
    ):
        timing["durations_ms"].setdefault(name, recorded.get(name))
        if local.get(name) is not None:
            measured[name] = local[name]
    if detection.get("duration_ms") is not None:
        measured["panel_detection_ms"] = detection["duration_ms"]
    if (
        fallback.get("duration_ms") is not None
        and fallback.get("clock_basis") == "process_monotonic"
    ):
        measured["vision_ms"] = fallback["duration_ms"]
    if (
        fallback.get("queue_wait_ms") is not None
        and fallback.get("queue_wait_clock_basis") == "process_monotonic"
    ):
        measured["vision_queue_wait_ms"] = fallback["queue_wait_ms"]
    if local.get("ocr_elapsed_ms") is not None:
        measured["ocr_ms"] = local["ocr_elapsed_ms"]
    elif not local.get("panel_detection") and local.get("wall_seconds") is not None:
        # predict_panel measures one call including its local lock/model wait.
        measured["ocr_ms"] = local["wall_seconds"] * 1000
    for name, value in measured.items():
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and value >= 0
        ):
            timing["durations_ms"][name] = round(value, 2)
            timing["duration_clock_basis"][name] = "process_monotonic"
    timing["ocr_scope"] = (
        "localized_region_processing_including_preprocessing_and_model_wait"
        if local.get("panel_detection")
        else "model_call_including_lock_and_model_wait"
    )
    timing["duration_scopes"] = {
        "vision_queue_wait_ms": "provider_slot_wait; panel tasks sum_across_this_tasks_panels",
        "vision_ms": "provider_calls; excludes_provider_slot_wait_when_monotonic",
        "ocr_wait_ms": "run_start_to_reported_ocr_start_including_scheduler_and_detection",
        "ocr_scheduler_wait_ms": "single_worker_executor_submission_to_execution_start",
        "ocr_execution_ms": local.get("ocr_execution_scope")
        or "predict_readout_including_detection_preprocessing_and_model_wait",
        "ocr_model_lock_wait_ms": "sum_of_model_adapter_lock_waits",
        "ocr_model_prepare_ms": "sum_of_model_preparation_including_load_or_remote_health_checks",
        "ocr_model_call_ms": "sum_of_model_adapter_calls_excluding_local_lock_and_preparation",
    }
    timing["ocr_stage_relationship"] = (
        "execution_includes_detection_and_model_stages; overlapping_stages_must_not_be_added"
    )
    if local.get("ocr_model_call_count") is not None:
        timing["ocr_model_call_count"] = local["ocr_model_call_count"]
    scopes = local.get("ocr_model_call_scopes")
    if scopes is None and local.get("ocr_model_call_scope"):
        scopes = [local["ocr_model_call_scope"]]
    if scopes is not None:
        timing["ocr_model_call_scopes"] = scopes
    timing["source_clock_basis"] = {
        "nas_file_mtime": "NAS file mtime; clocks not independently synchronized",
        "agent_receipt": "Agent-declared write time; clock not independently verified",
        "not_recorded": "Write time not available",
    }.get(timing.get("source_written_basis"), "Source write clock not independently verified")
    if timing.get("source_written_at") and timing.get("first_observed_at"):
        timing["source_clock_ahead"] = (
            elapsed_ms(timing["source_written_at"], timing["first_observed_at"]) is None
        )
    return timing
